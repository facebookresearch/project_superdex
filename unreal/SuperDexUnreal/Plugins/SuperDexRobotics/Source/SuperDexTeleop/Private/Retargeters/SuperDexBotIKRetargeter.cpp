/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Retargeters/SuperDexBotIKRetargeter.h"

#include <mochi_core/utils/math_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <algorithm>
#include "Engine/Engine.h"
#include "Engine/SkeletalMesh.h"
#include "MochiConversions.h"
#include "SuperDexBot.h"
#include "SuperDexIKSolverObject.h"
#include "SuperDexPoseFunctionLibrary.h"
#include "SuperDexTeleopModule.h"
#include "TeleopNode.h"
#if WITH_EDITOR
#include "Misc/DataValidation.h"
#endif

namespace {
bool UnitRotation(const FQuat& Rotation) {
  return !Rotation.ContainsNaN() && FMath::IsFinite(Rotation.SizeSquared()) &&
      FMath::Abs(Rotation.SizeSquared() - 1.0) < 1e-4;
}

const TCHAR* const kAwaitingInputPose = TEXT("Awaiting a valid input pose from its provider.");

// Input poses should fit the InputBot's joints, leaving only rounding error. Anything larger that a
// joint can't produce (swing off a revolute axis, translation of a rotational joint, ...) points at
// a mismatched skeleton, bone frames or pose source.
constexpr double kMaxInputRotationResidual = 1e-4; // [rad]
constexpr double kMaxInputTranslationResidual = 1e-4; // [nominal InputBot units]
// Log the first frame whose input exceeds them, then every this many such frames.
constexpr int32 kInputResidualLogInterval = 200;

// Logs (throttled by LargeResidualFrames) when an input pose leaves larger residuals than rounding
// error. Translations are only checked when the pose has BoneLocations: a rotation-only pose has
// none to fit.
void LogLargeInputResiduals(
    const UObject& Retargeter,
    const superdex::robotics::BotPrefab& InputBot,
    const mochi::DynamicArray<superdex::robotics::LinkPoseResidual>& Residuals,
    bool bCheckTranslations,
    int32& LargeResidualFrames) {
  int32 NumLarge = 0;
  int32 Worst = INDEX_NONE;
  double WorstExcess = 1.0;
  for (int32 I = 0; I < static_cast<int32>(Residuals.size()); ++I) {
    const double Excess = FMath::Max(
        Residuals[I].rotation / kMaxInputRotationResidual,
        bCheckTranslations ? Residuals[I].translation / kMaxInputTranslationResidual : 0.0);
    if (Excess > 1.0) {
      ++NumLarge;
      if (Excess > WorstExcess) {
        WorstExcess = Excess;
        Worst = I;
      }
    }
  }
  if (NumLarge == 0 || LargeResidualFrames++ % kInputResidualLogInterval != 0) {
    return;
  }
  UE_LOG(
      LogSuperDexTeleop,
      Warning,
      TEXT(
          "BotIK retargeter %s: the input pose doesn't fit the InputBot's joints (%d frame(s) so far, logged every %d). This frame %d link(s) exceed %.4f deg or %.4f units; worst '%s': rotation %.4f deg, translation %.5f units%s."),
      *Retargeter.GetPathName(),
      LargeResidualFrames,
      kInputResidualLogInterval,
      NumLarge,
      FMath::RadiansToDegrees(kMaxInputRotationResidual),
      kMaxInputTranslationResidual,
      UTF8_TO_TCHAR(InputBot.links[Worst].name.c_str()),
      FMath::RadiansToDegrees(static_cast<double>(Residuals[Worst].rotation)),
      static_cast<double>(Residuals[Worst].translation),
      bCheckTranslations ? TEXT("") : TEXT(" (not checked: the pose has no BoneLocations)"));
}

bool ValidInputTree(const superdex::robotics::BotPrefab& Bot, FString& Diagnostic) {
  const int32 Count = static_cast<int32>(Bot.links.size());
  if (Count == 0 || Bot.joints.size() != Bot.links.size() ||
      Bot.links[0].parentLink != INDEX_NONE ||
      (Bot.joints[0].type != mochi::ArticulatedJointType::Free &&
       Bot.joints[0].type != mochi::ArticulatedJointType::Hard) ||
      !mochi::IsFinite(Bot.scale) || Bot.scale <= 0) {
    Diagnostic = TEXT(
        "InputBot requires a nonempty prepared tree, a Free/Hard root and positive native scale.");
    return false;
  }
  TSet<FName> Names;
  int32 Dof = 0;
  for (int32 I = 0; I < Count; ++I) {
    const auto& Link = Bot.links[I];
    const auto& Joint = Bot.joints[I];
    const FName Name(UTF8_TO_TCHAR(Link.name.c_str()));
    if (Link.name.empty() || Name.IsNone() || Names.Contains(Name) ||
        (I > 0 && (Link.parentLink < 0 || Link.parentLink >= I))) {
      Diagnostic = TEXT(
          "InputBot link names must be nonempty/noncolliding and each link must follow its parent.");
      return false;
    }
    Names.Add(Name);
    for (const auto& Frame : {Joint.parentLinkFromJoint, Link.parentJointFromLink}) {
      if (!mochi::IsFinite(Frame) ||
          !mochi::NearEqual(mochi::Norm(Frame.GetRotation()), mochi::real(1), mochi::real(1e-4))) {
        Diagnostic = TEXT("InputBot joint/link rest frames must be finite and normalized.");
        return false;
      }
    }
    if (I == 0) {
      continue;
    }
    int32 Width = 0;
    switch (Joint.type) {
      case mochi::ArticulatedJointType::Hard:
        break;
      case mochi::ArticulatedJointType::Revolute:
      case mochi::ArticulatedJointType::Prismatic:
        Width = 1;
        if (!mochi::IsFinite(Joint.axis) ||
            !mochi::NearEqual(mochi::Norm(Joint.axis), mochi::real(1), mochi::real(1e-4))) {
          Diagnostic = TEXT("InputBot revolute/prismatic axes must be finite and normalized.");
          return false;
        }
        break;
      case mochi::ArticulatedJointType::Spherical:
        Width = 3;
        break;
      default:
        Diagnostic =
            TEXT("InputBot supports only Hard/Revolute/Prismatic/Spherical non-root joints.");
        return false;
    }
    for (int32 K = 0; K < Width; ++K, ++Dof) {
      if (Dof >= static_cast<int32>(Bot._dofIndices.size()) || Bot._dofIndices[Dof] != I) {
        Diagnostic = TEXT("InputBot prepared joint DOF ordering is inconsistent.");
        return false;
      }
    }
  }
  if (Dof != Bot._numDofs || Dof != static_cast<int32>(Bot._dofIndices.size()) ||
      (!Bot.defaultPose.empty() && static_cast<int32>(Bot.defaultPose.size()) != Dof)) {
    Diagnostic = TEXT("InputBot prepared DOF width/default pose is inconsistent with its joints.");
    return false;
  }
  return true;
}

bool MatchingOutputMesh(const USuperDexBot* Bot, FString& Diagnostic) {
  const auto* Mesh = Bot->GetSkeletalMesh();
  const auto& Prefab = Bot->GetBotPrefab();
  if (!Mesh || Mesh->GetRefSkeleton().GetNum() != static_cast<int32>(Prefab.links.size())) {
    Diagnostic = TEXT("Config.OutputBot requires an imported mesh in its prefab link order.");
    return false;
  }
  const auto& Ref = Mesh->GetRefSkeleton();
  for (int32 I = 0; I < Ref.GetNum(); ++I) {
    if (Ref.GetBoneName(I) != FName(UTF8_TO_TCHAR(Prefab.links[I].name.c_str())) ||
        Ref.GetParentIndex(I) != Prefab.links[I].parentLink) {
      Diagnostic = FString::Printf(
          TEXT("Config.OutputBot mesh bone %d differs from the prefab name/parent order."), I);
      return false;
    }
  }
  return true;
}
} // namespace

USuperDexBotIKRetargeter::USuperDexBotIKRetargeter() {
  bGetTransformedPoseDoesNotNeedValidSourcePose = true;
  bShow_EditorViewportPose = false;
  ErrorMessageKey = reinterpret_cast<uint64>(this);
}

bool USuperDexBotIKRetargeter::ValidateConfiguration(FString& Diagnostic) const {
  Diagnostic.Reset();
  if (!Config || !Config->OutputBot || !Config->InputBot) {
    Diagnostic =
        TEXT("Assign a complete SuperDex Bot IK Config with both OutputBot and InputBot resolved.");
    return false;
  }
  if (!Config->Validate(Diagnostic)) {
    return false;
  }
  if (!Config->CreateParams.bFixedBase) {
    Diagnostic = TEXT(
        "This bot-to-bot retargeter requires a fixed OUTPUT base; tracked root/clutch placement is separate from joint IK.");
    return false;
  }
  mochi::Error InputValidation;
  superdex::robotics::Validate(Config->InputBot->GetBotPrefab(), nullptr, InputValidation);
  if (!InputValidation.IsOK()) {
    Diagnostic = FString::Printf(
        TEXT("InputBot native model validation failed: %hs"), InputValidation.GetDescription());
    return false;
  }
  if (!ValidInputTree(Config->InputBot->GetBotPrefab(), Diagnostic) ||
      !MatchingOutputMesh(Config->OutputBot, Diagnostic)) {
    return false;
  }
  const auto Spec = Config->ToNative();
  const auto& Robot = Config->OutputBot->GetBotPrefab();
  if (!Spec.outputBotConfiguration.empty() &&
      static_cast<int32>(Spec.outputBotConfiguration.size()) != Robot._numDofs) {
    Diagnostic = TEXT(
        "Authored OutputBotConfiguration width does not match Config.OutputBot joint coordinates.");
    return false;
  }
  mochi::DynamicString NativeDiagnostic;
  if (!superdex::robotics::ValidateIKJointCouplings(Robot, Spec.jointCouplings, NativeDiagnostic)) {
    Diagnostic = UTF8_TO_TCHAR(NativeDiagnostic.c_str());
    return false;
  }
  bool Active = false;
  const auto& InputBot = Config->InputBot->GetBotPrefab();
  for (const auto& Goal : Spec.goals) {
    const auto HasLink = [](const superdex::robotics::BotPrefab& Model, std::string_view Name) {
      return std::any_of(Model.links.begin(), Model.links.end(), [&](const auto& Link) {
        return Link.name == Name;
      });
    };
    if (!HasLink(Robot, Goal.outputBotLink) ||
        (!Goal.inputBotLink.empty() && !HasLink(InputBot, Goal.inputBotLink))) {
      Diagnostic = FString::Printf(
          TEXT("IK goal '%hs' references a missing OutputBot or InputBot link."),
          Goal.name.c_str());
      return false;
    }
    Active |= Goal.enabled && (Goal.positionStiffness > 0 || Goal.rotationStiffness > 0);
  }
  if (!Active) {
    Diagnostic = TEXT("The IK configuration has no active pose goals.");
    return false;
  }
  if (!FMath::IsFinite(JointAngleMultiplier)) {
    Diagnostic = TEXT("Hand joint bias must be finite.");
    return false;
  }
  if (bPublishTeleopHandJointTargets || bSubscribeToTeleopHandJointPositions) {
    const auto* Node =
        GetOwner() ? GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>() : nullptr;
    if (GetOwner() && (!Node || Node->NumHandJoints != Robot._numDofs)) {
      Diagnostic =
          TEXT("Teleop hand bridge requires an operator node matching Config.OutputBot hand DOFs.");
      return false;
    }
  }
  return ValidateSourceProvider(Diagnostic);
}

bool USuperDexBotIKRetargeter::ValidateSourceProvider(FString& Diagnostic) const {
  // Same priority as the pose framework. Input poses are assumed to match InputBot's link layout;
  // only their bone count is checked, per pose.
  if (!PoseSourceOverride &&
      (SourcePoseableOverride ? SourcePoseableOverride : GetChildSourcePoseable()) == this) {
    Diagnostic = TEXT("Retargeter cannot be its own pose source.");
    return false;
  }
  return true;
}

void USuperDexBotIKRetargeter::SetStatus(
    bool bValid,
    const FString& Diagnostic,
    bool bConfigurationError) {
  bool Changed;
  {
    FScopeLock Lock(&ResultMutex);
    Changed = bCurrentValid != bValid || LastDiagnostic != Diagnostic;
    bCurrentValid = bValid;
    LastDiagnostic = Diagnostic;
    if (Changed) {
      ++StatusSequence;
      bPendingStatus = true;
    }
  }
  if (!bValid && bPublishTeleopHandJointTargets && IsInGameThread() && GetOwner()) {
    if (auto* Node = GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>()) {
      Node->InvalidateHandJointTargets();
    }
  }
  if (bConfigurationError && Changed) {
    UE_LOG(LogSuperDexTeleop, Error, TEXT("BotIK retargeter %s: %s"), *GetPathName(), *Diagnostic);
  }
  if (GEngine && IsInGameThread()) {
    if (bConfigurationError) {
      GEngine->AddOnScreenDebugMessage(
          ErrorMessageKey,
          3600,
          FColor::Red,
          FString::Printf(TEXT("BotIK retargeter %s: %s"), *GetName(), *Diagnostic));
    } else {
      GEngine->RemoveOnScreenDebugMessage(ErrorMessageKey);
    }
  }
}

bool USuperDexBotIKRetargeter::InitializeRetargeter() {
  if (!IsInGameThread()) {
    return false;
  }
  ReleaseSolver();
  FString Diagnostic;
  if (!ValidateConfiguration(Diagnostic)) {
    SetStatus(false, Diagnostic, true);
    return false;
  }
  NativeConfig = Config->ToNative();
  OutputModel = Config->OutputBot->GetBotPrefab();
  InputModel = Config->InputBot->GetBotPrefab();
  OutputModelJson = Config->OutputBot->GetBotPrefabJson();
  InputModelJson = Config->InputBot->GetBotPrefabJson();
  Solver = USuperDexIKSolverObject::CreateBotIKSolver(this, Config->OutputBot, NativeConfig);
  if (!Solver) {
    SetStatus(false, TEXT("Could not instantiate the configured private IK solver."), true);
    return false;
  }
  InitializedConfig = Config;
  InitializedOutputBot = Config->OutputBot;
  InitializedInputBot = Config->InputBot;
  SetSkinnedAssetAndUpdate(Config->OutputBot->GetSkeletalMesh());
  InputRootFrameFromOutputRootFrame = FTransform::Identity;
  // Capture the root-driven Mochi pose contract from the real component/source relation, not
  // outputBaseFrameFromInputBaseFrame. That authored frame aligns solver goals only. In a
  // roboperator, using its inverse here adds a fresh arm-target offset whenever the clutch resets
  // from arm feedback. Resolve metadata only: never invoke the runtime world/controller pose-source
  // resolver here.
  const auto* Source = PoseSourceOverride
      ? nullptr
      : (SourcePoseableOverride ? SourcePoseableOverride : GetChildSourcePoseable());
  if (Source) {
    auto WorldFrameFromOutputRootFrame = GetBoneTransform(0);
    auto WorldFrameFromInputRootFrame = Source->GetComponentTransform();
    WorldFrameFromOutputRootFrame.SetScale3D(FVector::OneVector);
    WorldFrameFromInputRootFrame.SetScale3D(FVector::OneVector);
    InputRootFrameFromOutputRootFrame =
        USuperDexPoseFunctionLibrary::MultiplyTransformsStrippingScale(
            WorldFrameFromOutputRootFrame, WorldFrameFromInputRootFrame.Inverse());
    if (InputRootFrameFromOutputRootFrame.ContainsNaN() ||
        !UnitRotation(InputRootFrameFromOutputRootFrame.GetRotation())) {
      ReleaseSolver();
      SetStatus(false, TEXT("Output/source root calibration must be finite and rigid."), true);
      return false;
    }
  }
  bInputBotHasPrismatic =
      std::any_of(InputModel.joints.begin(), InputModel.joints.end(), [](const auto& Joint) {
        return Joint.type == mochi::ArticulatedJointType::Prismatic;
      });
  SetStatus(false, kAwaitingInputPose, false);
  return true;
}

bool USuperDexBotIKRetargeter::GetTransformedPose_Implementation(
    const FSuperDexPose& SourcePose,
    bool bSourceIsValid,
    FSuperDexPose& TransformedPose) {
  if (!IsInGameThread()) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "BotIK retargeter solves require the game thread; use cached results on physics threads."));
    return false;
  }
  bool bModelsChanged = !Solver || !Config || Config != InitializedConfig ||
      Config->OutputBot != InitializedOutputBot || Config->InputBot != InitializedInputBot;
#if WITH_EDITOR
  // Only the editor changes a bot asset in place (e.g. a reimport).
  bModelsChanged = bModelsChanged || Config->OutputBot->GetBotPrefabJson() != OutputModelJson ||
      Config->InputBot->GetBotPrefabJson() != InputModelJson;
#endif
  if (bModelsChanged) {
    SetStatus(
        false,
        TEXT(
            "Retargeter is uninitialized or its model references changed; call InitializeRetargeter."),
        true);
    return false;
  }
  if (!bSourceIsValid || SourcePose.bIsInInfiniteLoop ||
      SourcePose.SourceName == FName(TEXT("RefSkeleton")) ||
      SourcePose.tags.Contains(FName(TEXT("InEditorOnly"))) ||
      (SourcePose.bIsTrackingEnabled && !SourcePose.bIsTrackingConfident)) {
    SetStatus(false, kAwaitingInputPose, false);
    return false;
  }
  const int32 InputLinks = static_cast<int32>(InputModel.links.size());
  const bool bHasLocations = !SourcePose.BoneLocations.IsEmpty();
  if (SourcePose.BoneRotations.Num() != InputLinks ||
      (bHasLocations && SourcePose.BoneLocations.Num() != InputLinks)) {
    SetStatus(false, TEXT("Input pose bone count must match the InputBot link count."), true);
    return false;
  }
  if (bInputBotHasPrismatic && !bHasLocations) {
    SetStatus(
        false,
        TEXT("InputBot prismatic joints need input pose BoneLocations; rotations can't pose them."),
        true);
    return false;
  }
  // Parent-from-link transforms in nominal InputBot space; only prismatic joints read translations.
  FUnrealToMochiConverter ToNative(kSuperDexSpace);
  mochi::DynamicArray<mochi::TransformRT> ParentFromLinks(InputLinks);
  for (int32 I = 0; I < InputLinks; ++I) {
    ParentFromLinks[I] = mochi::TransformRT(
        ToNative.RotationToMochi(SourcePose.BoneRotations[I]),
        bHasLocations
            ? ToNative.TranslationToMochi(SourcePose.BoneLocations[I]) * (1 / InputModel.scale)
            : mochi::Real3{});
  }
  // The root link's local transform moves the whole input bot, so it places its base.
  const mochi::TransformRT InputRootLocal(
      ParentFromLinks[0].GetRotation(),
      ParentFromLinks[0].GetTranslation() * (InputModel.scale * NativeConfig.inputBotScale));
  mochi::DynamicArray<mochi::real> InputPositions;
  mochi::DynamicArray<superdex::robotics::LinkPoseResidual> InputResiduals;
  mochi::Error Error;
  // The previous frame's positions pick the turn of revolute joints that spin past +-pi.
  superdex::robotics::ComputePoseFromLinkTransforms(
      InputModel,
      ParentFromLinks,
      InputPositions,
      superdex::robotics::LinkTransformSpace::ParentFromLink,
      Error,
      PreviousInputPositions,
      &InputResiduals);
  if (Error.IsOK()) {
    LogLargeInputResiduals(
        *this, InputModel, InputResiduals, bHasLocations, LargeInputResidualFrames);
  }
  if (!Error.IsOK() ||
      !Solver->SetBotInputConfiguration(
          InputModel,
          InputPositions,
          NativeConfig.outputBaseFrameFromInputBaseFrame * InputRootLocal,
          NativeConfig.inputBotScale,
          NativeConfig.goals)) {
    SetStatus(false, TEXT("Applying the input pose to the IK goals failed."), true);
    return false;
  }
  PreviousInputPositions = MoveTemp(InputPositions);
  bool Reached = false;
  if (!Solver->SolveBotIK(Reached)) {
    SetStatus(false, TEXT("Native IK solve failed; no fresh robot command was published."), true);
    return false;
  }
  TArray<double> Positions = Solver->GetSolvedDofs();
  if (Positions.Num() != OutputModel._numDofs) {
    SetStatus(false, TEXT("IK result width does not match the output bot configuration."), true);
    return false;
  }
  for (double Q : Positions) {
    if (!FMath::IsFinite(Q)) {
      SetStatus(false, TEXT("IK produced nonfinite joint coordinates."), true);
      return false;
    }
  }
  for (int32 I = 0; I < Positions.Num(); ++I) {
    if (!JointAngleMultiplierExcludeIndex.Contains(I)) {
      Positions[I] *= JointAngleMultiplier;
    }
  }
  TArray<double> DisplayPositions = Positions;
  if (bSubscribeToTeleopHandJointPositions && GetOwner()) {
    auto* Node = GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>();
    TArray<double> Feedback;
    if (Node && Node->GetHandJointStates(Feedback)) {
      if (Feedback.Num() != Positions.Num() ||
          !std::all_of(
              Feedback.begin(), Feedback.end(), [](double Q) { return FMath::IsFinite(Q); })) {
        SetStatus(
            false,
            TEXT("Teleop hand feedback does not match the finite hand-only DOF order."),
            true);
        return false;
      }
      DisplayPositions = MoveTemp(Feedback);
    }
  }
  FSuperDexPose Candidate = SourcePose;
  Candidate.SkeletonName = FName(*InitializedOutputBot->GetName());
  Candidate.tags.AddUnique(FName(TEXT("SuperDexBotIK")));
  Candidate.SkeletonRootToVrRoot =
      USuperDexPoseFunctionLibrary::MultiplyTransformsWithoutScalePreservingScale(
          InputRootFrameFromOutputRootFrame, SourcePose.SkeletonRootToVrRoot);
  if (Candidate.SkeletonRootToVrRoot.ContainsNaN()) {
    SetStatus(false, TEXT("Output root composition must remain finite."), true);
    return false;
  }
  FString PackingDiagnostic;
  if (!ComputeSuperDexPoseFromJointPositions(DisplayPositions, Candidate, PackingDiagnostic)) {
    SetStatus(false, PackingDiagnostic, true);
    return false;
  }
  TransformedPose = MoveTemp(Candidate);
  {
    FScopeLock Lock(&ResultMutex);
    JointCache = MoveTemp(Positions);
    bHasSolution = true;
    bLastGoalsReached = Reached;
    ++SolutionSequence;
  }
  SetStatus(true, Reached ? TEXT("Goals reached") : TEXT("Best effort: goals unreached"), false);
  return true;
}

bool USuperDexBotIKRetargeter::ComputeSuperDexPoseFromJointPositions(
    TConstArrayView<double> DisplayPositions,
    FSuperDexPose& OutPose,
    FString& Diagnostic) {
  // Same contract as USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions: pack
  // parent-relative bones ONLY, leaving the caller's root/metadata intact. Use native snapshots,
  // not the legacy metadata FK: native prefab scale, prismatic coordinates and nonidentity child
  // rest frames must survive.
  mochi::DynamicArray<mochi::TransformRT> Links;
  for (const auto& Link : Solver->GetOutputBotWorldFromLinks()) {
    Links.push_back(Link);
  }
  // Solved commands and optional feedback visualization stay distinct; native FK restores the
  // solver's warm start. Never splice a 20-coordinate hand pose into 27 arm+hand bones.
  if (JointAngleMultiplier != 1.0 || bSubscribeToTeleopHandJointPositions) {
    if (!Solver->CopyOutputBotFKAtJointConfiguration(DisplayPositions, Links)) {
      Diagnostic = TEXT("Native hand feedback/bias FK failed.");
      return false;
    }
  }
  if (Links.size() != OutputModel.links.size()) {
    Diagnostic = TEXT("Solved link snapshots do not match the output robot topology.");
    return false;
  }
  FMochiToUnrealConverter ToUnreal(kSuperDexSpace);
  OutPose.BoneRotations.SetNum(static_cast<int32>(Links.size()));
  OutPose.BoneLocations.SetNum(static_cast<int32>(Links.size()));
  for (int32 I = 0; I < OutPose.BoneRotations.Num(); ++I) {
    const auto Local = I == 0 ? mochi::TransformRT::Identity()
                              : mochi::Invert(Links[OutputModel.links[I].parentLink]) * Links[I];
    const auto Link = ToUnreal.TransformToUnreal(Local);
    OutPose.BoneRotations[I] = Link.GetRotation();
    OutPose.BoneLocations[I] = Link.GetTranslation();
  }
  return true;
}

bool USuperDexBotIKRetargeter::GetPose(FSuperDexPose& OutPose) {
  if (!IsInGameThread()) {
    return false;
  }
  if (bEvaluatingPose) {
    // The outer SuperDexPose request may still own CacheLock while invoking its source. No cache
    // read or callback publication is safe from a nested source evaluation.
    return false;
  }
  if (bPublishing) {
    return GetCachedPose(OutPose);
  }
  bool Result;
  {
    TGuardValue<bool> EvaluationGuard(bEvaluatingPose, true);
    Result = Super::GetPose(OutPose);
  }
  bool WasValid;
  {
    FScopeLock Lock(&ResultMutex);
    WasValid = bCurrentValid;
  }
  if (!Result && WasValid) {
    SetStatus(false, TEXT("Pose source unavailable, disabled or cyclic; no fresh command."), false);
  }
  // Super holds CacheLock during GetTransformedPose. Listener callbacks run only after return.
  PublishPending();
  FScopeLock Lock(&ResultMutex);
  return Result && bCurrentValid;
}

bool USuperDexBotIKRetargeter::GetCachedPose(FSuperDexPose& OutPose) const {
  // Same lock order as GetTransformedPose: SuperDexPose cache first, result snapshot second.
  FReadScopeLock CacheGuard(CacheLock);
  FScopeLock ResultGuard(&ResultMutex);
  if (!CacheInitialized || !bCurrentValid) {
    return false;
  }
  OutPose = CachedOutputPose;
  return true;
}

bool USuperDexBotIKRetargeter::GetCachedJointConfiguration(TArray<double>& Positions, bool& bValid)
    const {
  FScopeLock Lock(&ResultMutex);
  bValid = bCurrentValid;
  if (bHasSolution) {
    Positions = JointCache;
  }
  return bHasSolution;
}

FString USuperDexBotIKRetargeter::GetRetargetingDiagnostic() const {
  FScopeLock Lock(&ResultMutex);
  return LastDiagnostic;
}

void USuperDexBotIKRetargeter::PublishPending() {
  if (!IsInGameThread() || bPublishing) {
    return;
  }
  TArray<double> Positions;
  FString Diagnostic;
  bool Valid, Reached, PublishSolution, PublishStatus;
  uint64 Generation, Sequence;
  {
    FScopeLock Lock(&ResultMutex);
    Generation = SolverGeneration;
    Sequence = SolutionSequence;
    Valid = bCurrentValid;
    Reached = bLastGoalsReached;
    Diagnostic = LastDiagnostic;
    PublishSolution = Valid && SolutionSequence != PublishedSequence;
    PublishStatus = StatusSequence != PublishedStatusSequence;
    PublishedSequence = SolutionSequence;
    PublishedStatusSequence = StatusSequence;
    bPendingStatus = false;
    if (PublishSolution) {
      Positions = JointCache;
    }
  }
  bPublishing = true;
  if (PublishStatus) {
    OnRetargetingStatusChanged.Broadcast(Valid, Diagnostic);
  }
  if (PublishSolution &&
      (!Config || Config != InitializedConfig || Config->OutputBot != InitializedOutputBot ||
       Config->InputBot != InitializedInputBot)) {
    SetStatus(
        false,
        TEXT("Configuration references changed during result publication; reinitialize."),
        true);
  }
  {
    FScopeLock Lock(&ResultMutex);
    PublishSolution &=
        bCurrentValid && SolverGeneration == Generation && SolutionSequence == Sequence;
  }
  if (PublishSolution && Solver) {
    if (bPublishTeleopHandJointTargets) {
      auto* Node =
          GetOwner() ? GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>() : nullptr;
      if (IsValid(Node) && Node->NumHandJoints == Positions.Num()) {
        Node->SetHandJointTargets(Positions);
      } else {
        if (IsValid(Node)) {
          Node->InvalidateHandJointTargets();
        }
        SetStatus(
            false,
            TEXT("Teleop hand node is missing or its width changed; no command published."),
            true);
        PublishSolution = false;
      }
    }
    if (PublishSolution) {
      OnJointConfigurationSolved.Broadcast(Positions, Reached);
    }
  }
  bPublishing = false;
}

void USuperDexBotIKRetargeter::ReleaseSolver() {
  if (bPublishTeleopHandJointTargets && GetOwner()) {
    if (auto* Node = GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>()) {
      Node->InvalidateHandJointTargets();
    }
  }
  if (Solver) {
    Solver->DestroyMochiResources();
    Solver = nullptr;
  }
  InitializedConfig = nullptr;
  InitializedOutputBot = nullptr;
  InitializedInputBot = nullptr;
  PreviousInputPositions.clear();
  LargeInputResidualFrames = 0;
  FScopeLock Lock(&ResultMutex);
  ++SolverGeneration;
  bHasSolution = bCurrentValid = false;
  JointCache.Reset();
  PublishedSequence = SolutionSequence;
}

void USuperDexBotIKRetargeter::BeginPlay() {
  Super::BeginPlay();
  InitializeRetargeter();
}
void USuperDexBotIKRetargeter::EndPlay(const EEndPlayReason::Type Reason) {
  ReleaseSolver();
  if (GEngine) {
    GEngine->RemoveOnScreenDebugMessage(ErrorMessageKey);
  }
  Super::EndPlay(Reason);
}
void USuperDexBotIKRetargeter::OnComponentDestroyed(bool bDestroyingHierarchy) {
  ReleaseSolver();
  Super::OnComponentDestroyed(bDestroyingHierarchy);
}
#if WITH_EDITOR
void USuperDexBotIKRetargeter::PostEditChangeProperty(FPropertyChangedEvent& Event) {
  Super::PostEditChangeProperty(Event);
  ReleaseSolver();
}
EDataValidationResult USuperDexBotIKRetargeter::IsDataValid(FDataValidationContext& Context) const {
  const auto Base = Super::IsDataValid(Context);
  FString Diagnostic;
  if (!ValidateConfiguration(Diagnostic)) {
    Context.AddError(FText::FromString(Diagnostic));
    return EDataValidationResult::Invalid;
  }
  return Base == EDataValidationResult::Invalid ? Base : EDataValidationResult::Valid;
}
#endif
