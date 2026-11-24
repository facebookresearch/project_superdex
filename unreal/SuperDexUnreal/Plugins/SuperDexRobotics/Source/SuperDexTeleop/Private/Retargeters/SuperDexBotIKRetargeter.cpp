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
#include <superdex_robotics/ik/bot_ik_config.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <algorithm>
#include "DrawDebugHelpers.h"
#include "Engine/Engine.h"
#include "Engine/SkeletalMesh.h"
#include "Materials/MaterialInterface.h"
#include "MochiConversions.h"
#include "SuperDexBot.h"
#include "SuperDexIKSolverObject.h"
#include "SuperDexPoseFunctionLibrary.h"
#include "SuperDexTeleopModule.h"
#include "TeleopNode.h"
#include "UObject/ConstructorHelpers.h"
#if WITH_EDITOR
#include "Misc/DataValidation.h"
#endif

namespace {
bool UnitRotation(const FQuat& Rotation) {
  return !Rotation.ContainsNaN() && FMath::IsFinite(Rotation.SizeSquared()) &&
      FMath::Abs(Rotation.SizeSquared() - 1.0) < 1e-4;
}

const TCHAR* const AwaitingInputPose = TEXT("Awaiting a valid input pose from its provider.");

// Input poses should fit the InputBot's joints, leaving only rounding error. Anything larger that a
// joint can't produce (swing off a revolute axis, translation of a rotational joint, ...) points at
// a mismatched skeleton, bone frames or pose source.
constexpr double MaxInputRotationResidual = 1e-4; // [rad]
constexpr double MaxInputTranslationResidual = 1e-4; // [nominal InputBot units]
// Log the first frame whose input exceeds them, then every this many such frames.
constexpr int32 InputResidualLogInterval = 200;

// Logs (throttled by LargeResidualFrames) when an input pose leaves larger residuals than rounding
// error. Translations are only checked when the InputBot has prismatic joints: its rotational
// joints ignore them, so the link offsets of a pose on another bot of its skeleton may differ.
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
        Residuals[I].rotation / MaxInputRotationResidual,
        bCheckTranslations ? Residuals[I].translation / MaxInputTranslationResidual : 0.0);
    if (Excess > 1.0) {
      ++NumLarge;
      if (Excess > WorstExcess) {
        WorstExcess = Excess;
        Worst = I;
      }
    }
  }
  if (NumLarge == 0 || LargeResidualFrames++ % InputResidualLogInterval != 0) {
    return;
  }
  UE_LOG(
      LogSuperDexTeleop,
      Warning,
      TEXT(
          "BotIK retargeter %s: the input pose doesn't fit the InputBot's joints (%d frame(s) so far, logged every %d). This frame %d link(s) exceed %.4f deg or %.4f units; worst '%s': rotation %.4f deg, translation %.5f units%s."),
      *Retargeter.GetPathName(),
      LargeResidualFrames,
      InputResidualLogInterval,
      NumLarge,
      FMath::RadiansToDegrees(MaxInputRotationResidual),
      MaxInputTranslationResidual,
      UTF8_TO_TCHAR(InputBot.links[Worst].name.c_str()),
      FMath::RadiansToDegrees(static_cast<double>(Residuals[Worst].rotation)),
      static_cast<double>(Residuals[Worst].translation),
      bCheckTranslations ? TEXT("") : TEXT(" (not checked: the InputBot has no prismatic joints)"));
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
    const superdex::robotics::BotLinkPrefab& Link = Bot.links[I];
    const superdex::robotics::BotJointPrefab& Joint = Bot.joints[I];
    const FName Name(UTF8_TO_TCHAR(Link.name.c_str()));
    if (Link.name.empty() || Name.IsNone() || Names.Contains(Name) ||
        (I > 0 && (Link.parentLink < 0 || Link.parentLink >= I))) {
      Diagnostic = TEXT(
          "InputBot link names must be nonempty/noncolliding and each link must follow its parent.");
      return false;
    }
    Names.Add(Name);
    for (const mochi::TransformRT& Frame : {Joint.parentLinkFromJoint, Link.parentJointFromLink}) {
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
  const USkeletalMesh* Mesh = Bot->GetSkeletalMesh();
  const superdex::robotics::BotPrefab& Prefab = Bot->GetBotPrefab();
  if (!Mesh || Mesh->GetRefSkeleton().GetNum() != static_cast<int32>(Prefab.links.size())) {
    Diagnostic = TEXT("Config.OutputBot requires an imported mesh in its prefab link order.");
    return false;
  }
  const FReferenceSkeleton& Ref = Mesh->GetRefSkeleton();
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
  // Same default as UMochiIKRetargeter's input pose skeleton: the translucent tracked-hand look.
  static ConstructorHelpers::FObjectFinder<UMaterialInterface> DefaultInputPoseSkeletonMaterial(TEXT(
      "/Script/Engine.MaterialInstanceConstant'/SuperDexRobotics/Teleop/Operator/MI_Operator_TrackedHand.MI_Operator_TrackedHand'"));
  DebugDrawInputPoseSkeletonMaterialOverride = DefaultInputPoseSkeletonMaterial.Object;
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
  const superdex::robotics::BotIKConfig Spec = Config->ToNative();
  const superdex::robotics::BotPrefab& Robot = Config->OutputBot->GetBotPrefab();
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
  bool bActive = false;
  const superdex::robotics::BotPrefab& InputBot = Config->InputBot->GetBotPrefab();
  for (const superdex::robotics::BotIKGoalConfig& Goal : Spec.goals) {
    const auto HasLink = [](const superdex::robotics::BotPrefab& Model, std::string_view Name) {
      return std::any_of(
          Model.links.begin(),
          Model.links.end(),
          [&](const superdex::robotics::BotLinkPrefab& Link) { return Link.name == Name; });
    };
    if (!HasLink(Robot, Goal.outputBotLink) ||
        (!Goal.inputBotLink.empty() && !HasLink(InputBot, Goal.inputBotLink))) {
      Diagnostic = FString::Printf(
          TEXT("IK goal '%hs' references a missing OutputBot or InputBot link."),
          Goal.name.c_str());
      return false;
    }
    bActive |= Goal.enabled && (Goal.positionStiffness > 0 || Goal.rotationStiffness > 0);
  }
  if (!bActive) {
    Diagnostic = TEXT("The IK configuration has no active pose goals.");
    return false;
  }
  if (!FMath::IsFinite(JointAngleMultiplier)) {
    Diagnostic = TEXT("Hand joint bias must be finite.");
    return false;
  }
  if (bPublishTeleopHandJointTargets || bSubscribeToTeleopHandJointPositions) {
    const UTeleopNodeOperatorComponent* Node =
        GetOwner() ? GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>() : nullptr;
    if (GetOwner() && (!Node || Node->NumHandJoints != Robot._numDofs)) {
      Diagnostic =
          TEXT("Teleop hand bridge requires an operator node matching Config.OutputBot hand DOFs.");
      return false;
    }
  }
  return true;
}

void USuperDexBotIKRetargeter::SetStatus(
    bool bValid,
    const FString& Diagnostic,
    bool bConfigurationError) {
  bool bChanged;
  {
    FScopeLock Lock(&ResultMutex);
    bChanged = bCurrentValid != bValid || LastDiagnostic != Diagnostic;
    bCurrentValid = bValid;
    LastDiagnostic = Diagnostic;
  }
  if (!bValid && bPublishTeleopHandJointTargets && IsInGameThread() && GetOwner()) {
    if (UTeleopNodeOperatorComponent* Node =
            GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>()) {
      Node->InvalidateHandJointTargets();
    }
  }
  if (bConfigurationError && bChanged) {
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
  PrepareBiasedCommandLimits();
  InputModel = Config->InputBot->GetBotPrefab();
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
  // The root-driven pose contract comes from the authored input/output base frame. An attached
  // source (normally the clutch) is placed at that frame too, so the clutch initializes from the
  // same relation and the arm target does not jump when it resets from arm feedback. Resolve
  // metadata only: never invoke the runtime world/controller pose-source resolver here.
  PlaceSourceAtConfigInputBaseFrame();
  InputRootFrameFromOutputRootFrame = ComputeInputRootFrameFromOutputRootFrame();
  if (InputRootFrameFromOutputRootFrame.ContainsNaN() ||
      !UnitRotation(InputRootFrameFromOutputRootFrame.GetRotation())) {
    ReleaseSolver();
    SetStatus(false, TEXT("Output/source root calibration must be finite and rigid."), true);
    return false;
  }
  bInputBotHasPrismatic = std::any_of(
      InputModel.joints.begin(),
      InputModel.joints.end(),
      [](const superdex::robotics::BotJointPrefab& Joint) {
        return Joint.type == mochi::ArticulatedJointType::Prismatic;
      });
  SetStatus(false, AwaitingInputPose, false);
  return true;
}

bool USuperDexBotIKRetargeter::GetTransformedPose_Implementation(
    const FSuperDexPose& SourcePose,
    bool bSourceIsValid,
    FSuperDexPose& TransformedPose) {
  const bool bModelsChanged = !Solver || !Config || Config != InitializedConfig ||
      Config->OutputBot != InitializedOutputBot || Config->InputBot != InitializedInputBot;
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
      SourcePose.Tags.Contains(FName(TEXT("InEditorOnly"))) ||
      (SourcePose.bIsTrackingEnabled && !SourcePose.bIsTrackingConfident)) {
    SetStatus(false, AwaitingInputPose, false);
    return false;
  }
  const int32 InputLinks = static_cast<int32>(InputModel.links.size());
  // The input pose's links relative to their parents, at its joint positions: the InputBot's own
  // poses, or those of another bot of its skeleton (one link per InputBot link, in its order).
  TArray<FTransform> SourceLinkToParent;
  if (!USuperDexPoseFunctionLibrary::ComputeSuperDexPoseLinkToParentTransforms(
          SourcePose, SourceLinkToParent) ||
      SourceLinkToParent.Num() != InputLinks) {
    SetStatus(
        false,
        TEXT("Input pose must hold joint positions of a bot with the InputBot's link count."),
        true);
    return false;
  }
  // Parent-from-link transforms in nominal InputBot space; only prismatic joints read translations.
  FUnrealToMochiConverter ToNative(SuperDexSpace);
  mochi::DynamicArray<mochi::TransformRT> ParentFromLinks(InputLinks);
  for (int32 I = 0; I < InputLinks; ++I) {
    ParentFromLinks[I] = ToNative.TransformToMochi_RT(SourceLinkToParent[I]);
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
        *this, InputModel, InputResiduals, bInputBotHasPrismatic, LargeInputResidualFrames);
  }
  const mochi::TransformRT WorldFrameFromInputBaseFrame =
      NativeConfig.outputBaseFrameFromInputBaseFrame * InputRootLocal;
  if (!Error.IsOK() ||
      !Solver->SetBotInputConfiguration(
          InputModel,
          InputPositions,
          WorldFrameFromInputBaseFrame,
          NativeConfig.inputBotScale,
          NativeConfig.goals)) {
    SetStatus(false, TEXT("Applying the input pose to the IK goals failed."), true);
    return false;
  }
  PreviousInputPositions = MoveTemp(InputPositions);
  bool bReached = false;
  if (!Solver->SolveBotIK(bReached)) {
    SetStatus(false, TEXT("Native IK solve failed; no fresh robot command was published."), true);
    return false;
  }
  if (bDebugDrawIKCoordinateFrames) {
    CaptureDebugGoalFrames(WorldFrameFromInputBaseFrame);
  } else {
    DebugGoalFrames.Reset();
  }
  if (bDebugDrawInputPoseSkeleton) {
    CaptureDebugInputPose(WorldFrameFromInputBaseFrame);
  } else {
    bHasDebugInputPose = false;
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
  if (!ApplyJointBiasAndLimits(Positions)) {
    SetStatus(false, TEXT("Biased command must have a valid width and finite coordinates."), true);
    return false;
  }
  TArray<double> DisplayPositions = Positions;
  if (bSubscribeToTeleopHandJointPositions && GetOwner()) {
    UTeleopNodeOperatorComponent* Node =
        GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>();
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
  Candidate.Tags.AddUnique(FName(TEXT("SuperDexBotIK")));
  Candidate.SkeletonRootToVrRoot =
      USuperDexPoseFunctionLibrary::MultiplyTransformsWithoutScalePreservingScale(
          InputRootFrameFromOutputRootFrame, SourcePose.SkeletonRootToVrRoot);
  if (Candidate.SkeletonRootToVrRoot.ContainsNaN()) {
    SetStatus(false, TEXT("Output root composition must remain finite."), true);
    return false;
  }
  if (!USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(
          InitializedOutputBot, DisplayPositions, Candidate)) {
    SetStatus(false, TEXT("IK result width does not match the output bot."), true);
    return false;
  }
  if (bPublishTeleopHandJointTargets) {
    auto* Node =
        GetOwner() ? GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>() : nullptr;
    if (!IsValid(Node) || Node->NumHandJoints != Positions.Num()) {
      SetStatus(
          false,
          TEXT("Teleop hand node is missing or its width changed; no command published."),
          true);
      return false;
    }
    Node->SetHandJointTargets(Positions);
  }
  TransformedPose = MoveTemp(Candidate);
  {
    FScopeLock Lock(&ResultMutex);
    JointCache = MoveTemp(Positions);
    bHasSolution = true;
  }
  SetStatus(true, bReached ? TEXT("Goals reached") : TEXT("Best effort: goals unreached"), false);
  return true;
}

bool USuperDexBotIKRetargeter::GetPose(FSuperDexPose& OutPose) {
  if (!IsInGameThread()) {
    return false;
  }
  const bool Result = Super::GetPose(OutPose);
  bool bValid;
  {
    FScopeLock Lock(&ResultMutex);
    bValid = bCurrentValid;
  }
  if (!Result && bValid) {
    SetStatus(false, TEXT("Pose source unavailable, disabled or cyclic; no fresh command."), false);
    return false;
  }
  return Result && bValid;
}

bool USuperDexBotIKRetargeter::GetCachedPose(FSuperDexPose& OutPose) const {
  // Same lock order as GetTransformedPose: SuperDexPose cache first, result snapshot second. A
  // thread computing this retargeter's pose already holds the cache.
  TOptional<FReadScopeLock> CacheGuard;
  if (!IsCacheWriterThread()) {
    CacheGuard.Emplace(CacheLock);
  }
  FScopeLock ResultGuard(&ResultMutex);
  if (!bCacheInitialized || !bCurrentValid) {
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

void USuperDexBotIKRetargeter::PrepareBiasedCommandLimits() {
  BiasedCommandLimits.Init(FBiasedCommandLimit{}, OutputModel._numDofs);
  int32 PreviousJoint = INDEX_NONE;
  int32 Component = 0;
  for (int32 I = 0; I < BiasedCommandLimits.Num(); ++I) {
    const int32 JointIndex = OutputModel._dofIndices[I];
    Component = JointIndex == PreviousJoint ? Component + 1 : 0;
    PreviousJoint = JointIndex;
    const superdex::robotics::BotJointPrefab& Joint = OutputModel.joints[JointIndex];
    const bool bScalar = Joint.type == mochi::ArticulatedJointType::Revolute ||
        Joint.type == mochi::ArticulatedJointType::Prismatic;
    if (!bScalar && Joint.type != mochi::ArticulatedJointType::Spherical) {
      continue;
    }
    // Match MakeBotPose's authored-limit semantics, but never its random fallback ranges.
    // Native prismatic commands/FK use instantiated metres, unlike nominal prefab seeds.
    const mochi::real Scale =
        Joint.type == mochi::ArticulatedJointType::Prismatic ? OutputModel.scale : mochi::real(1);
    FBiasedCommandLimit& Limit = BiasedCommandLimits[I];
    if (Joint.minLimit.has_value() && mochi::IsFinite(*Joint.minLimit)) {
      Limit.bHasMinimum = true;
      Limit.Minimum = static_cast<double>(
          (bScalar ? mochi::Dot(*Joint.minLimit, Joint.axis) : (*Joint.minLimit)[Component]) *
          Scale);
    }
    if (Joint.maxLimit.has_value() && mochi::IsFinite(*Joint.maxLimit)) {
      Limit.bHasMaximum = true;
      Limit.Maximum = static_cast<double>(
          (bScalar ? mochi::Dot(*Joint.maxLimit, Joint.axis) : (*Joint.maxLimit)[Component]) *
          Scale);
    }
  }
}

bool USuperDexBotIKRetargeter::ApplyJointBiasAndLimits(TArray<double>& Positions) const {
  for (int32 I = 0; I < Positions.Num(); ++I) {
    if (!JointAngleMultiplierExcludeIndex.Contains(I)) {
      Positions[I] *= JointAngleMultiplier;
    }
  }
  // Preserve the unbiased path exactly. Exclusions bypass bias, not authored bounds.
  if (JointAngleMultiplier != 1.0) {
    if (Positions.Num() != BiasedCommandLimits.Num()) {
      return false;
    }
    for (int32 I = 0; I < Positions.Num(); ++I) {
      const FBiasedCommandLimit& Limit = BiasedCommandLimits[I];
      if (Limit.bHasMinimum) {
        Positions[I] = FMath::Max(Positions[I], Limit.Minimum);
      }
      if (Limit.bHasMaximum) {
        Positions[I] = FMath::Min(Positions[I], Limit.Maximum);
      }
    }
    // Feedback FK may use a different finite vector, so validate the command independently.
    return std::all_of(
        Positions.begin(), Positions.end(), [](double Q) { return FMath::IsFinite(Q); });
  }
  return true;
}

void USuperDexBotIKRetargeter::ReleaseSolver() {
  if (bPublishTeleopHandJointTargets && GetOwner()) {
    if (UTeleopNodeOperatorComponent* Node =
            GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>()) {
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
  BiasedCommandLimits.Reset();
  LargeInputResidualFrames = 0;
  DebugGoalFrames.Reset();
  bHasDebugInputPose = false;
  FScopeLock Lock(&ResultMutex);
  bHasSolution = bCurrentValid = false;
  JointCache.Reset();
}

void USuperDexBotIKRetargeter::OnRegister() {
  Super::OnRegister();
  PlaceSourceAtConfigInputBaseFrame();
}

void USuperDexBotIKRetargeter::OnChildAttached(USceneComponent* ChildComponent) {
  Super::OnChildAttached(ChildComponent);
  PlaceSourceAtConfigInputBaseFrame();
}

USuperDexPoseableComponent* USuperDexBotIKRetargeter::GetComponentPoseSource() const {
  if (PoseSourceOverride) {
    return nullptr;
  }
  return SourcePoseableOverride ? SourcePoseableOverride : GetChildSourcePoseable();
}

bool USuperDexBotIKRetargeter::ComputeSourcePlacement(
    const USuperDexPoseableComponent& Source,
    FTransform& OutSourceRelative) const {
  if (Config == nullptr || Source.GetAttachParent() != this) {
    return false;
  }
  // UE order throughout: A * B is A expressed in B's frame. Each bot's base is its root link, at
  // its component (the skeleton root of the poses it shows).
  FTransform SourceInComponent = Config->OutputBaseFrameFromInputBaseFrame;
  SourceInComponent.SetScale3D(FVector::OneVector);
  FTransform ComponentInAttachFrame = FTransform::Identity;
  if (!Source.GetAttachSocketName().IsNone()) {
    FTransform AttachFrameInComponent =
        GetSocketTransform(Source.GetAttachSocketName(), RTS_Component);
    AttachFrameInComponent.SetScale3D(FVector::OneVector);
    ComponentInAttachFrame = AttachFrameInComponent.Inverse();
  }
  OutSourceRelative = SourceInComponent * ComponentInAttachFrame;
  return true;
}

FTransform USuperDexBotIKRetargeter::ComputeInputRootFrameFromOutputRootFrame() const {
  if (Config == nullptr) {
    return FTransform::Identity;
  }
  // The output root in the source root's frame: both are their components' skeleton roots, so with
  // the source placed per ComputeSourcePlacement this equals the measured relation.
  FTransform InputBaseInOutputBase = Config->OutputBaseFrameFromInputBaseFrame;
  InputBaseInOutputBase.SetScale3D(FVector::OneVector);
  return InputBaseInOutputBase.Inverse();
}

void USuperDexBotIKRetargeter::PlaceSourceAtConfigInputBaseFrame() {
  USuperDexPoseableComponent* Source = GetComponentPoseSource();
  if (Source == nullptr || Source == this) {
    return;
  }
  FTransform SourceRelative;
  if (ComputeSourcePlacement(*Source, SourceRelative)) {
    Source->SetRelativeLocationAndRotation(
        SourceRelative.GetLocation(), SourceRelative.GetRotation());
  }
}

void USuperDexBotIKRetargeter::BeginPlay() {
  Super::BeginPlay();
  InitializeRetargeter();
}

void USuperDexBotIKRetargeter::CaptureDebugGoalFrames(
    const mochi::TransformRT& WorldFrameFromInputBaseFrame) {
  DebugGoalFrames.Reset();
  if (!Solver) {
    return;
  }
  // Same resolution as the native BotIK::SetInputBotConfiguration, in the solver's world frame.
  mochi::Error Error;
  mochi::DynamicArray<mochi::TransformRT> WorldFromInputLinks;
  superdex::robotics::ComputeIKInputBotTransforms(
      InputModel,
      PreviousInputPositions,
      WorldFrameFromInputBaseFrame,
      NativeConfig.inputBotScale,
      WorldFromInputLinks,
      Error);
  const auto WorldFromOutputLinks = Solver->GetOutputBotWorldFromLinks();
  if (!Error.IsOK() || WorldFromOutputLinks.empty()) {
    return;
  }
  // Express everything relative to the output base (its root link), which is where this
  // component's root bone is drawn.
  const mochi::TransformRT OutputBaseFromWorld = mochi::Invert(WorldFromOutputLinks[0]);
  FMochiToUnrealConverter ToUnreal(SuperDexSpace);
  const auto InOutputBase = [&](const mochi::TransformRT& WorldFromFrame) {
    return ToUnreal.TransformToUnreal(OutputBaseFromWorld * WorldFromFrame);
  };
  for (const auto& Goal : NativeConfig.goals) {
    if (!Goal.enabled || (Goal.positionStiffness <= 0 && Goal.rotationStiffness <= 0)) {
      continue;
    }
    const int OutputLink = superdex::robotics::FindLinkIndexByName(OutputModel, Goal.outputBotLink);
    if (OutputLink < 0 || OutputLink >= static_cast<int>(WorldFromOutputLinks.size())) {
      continue;
    }
    mochi::Error GoalError;
    mochi::TransformRT WorldFromGoal;
    superdex::robotics::ComputeIKGoalWorldConstraintTransform(
        Goal,
        OutputModel.worldFromRoot,
        &InputModel,
        WorldFromInputLinks,
        NativeConfig.inputBotScale,
        WorldFromGoal,
        GoalError);
    if (!GoalError.IsOK()) {
      continue;
    }
    FDebugGoalFrames Frames;
    Frames.GoalInOutputBase = InOutputBase(WorldFromGoal);
    Frames.OutputLinkInOutputBase = InOutputBase(WorldFromOutputLinks[OutputLink]);
    Frames.ConstraintInOutputBase =
        InOutputBase(WorldFromOutputLinks[OutputLink] * Goal.linkFrameFromConstraintFrame);
    if (!Goal.inputBotLink.empty()) {
      const int InputLink = superdex::robotics::FindLinkIndexByName(InputModel, Goal.inputBotLink);
      if (InputLink >= 0 && InputLink < static_cast<int>(WorldFromInputLinks.size())) {
        Frames.InputLinkInOutputBase = InOutputBase(WorldFromInputLinks[InputLink]);
        Frames.InputLinkName = FName(UTF8_TO_TCHAR(Goal.inputBotLink.c_str()));
        Frames.bHasInputLink = true;
      }
    }
    DebugGoalFrames.Add(Frames);
  }
}

void USuperDexBotIKRetargeter::CaptureDebugInputPose(
    const mochi::TransformRT& WorldFrameFromInputBaseFrame) {
  bHasDebugInputPose = false;
  const auto WorldFromOutputLinks =
      Solver ? Solver->GetOutputBotWorldFromLinks() : mochi::Span<mochi::TransformRT const>{};
  if (WorldFromOutputLinks.empty()) {
    return;
  }
  // Nominal FK at the solver's input positions; the input base placement and the display scale
  // reproduce ComputeIKInputBotTransforms, which scales only root-relative translations.
  mochi::Error Error;
  mochi::DynamicArray<mochi::TransformRT> ParentFromLinks;
  superdex::robotics::ComputeLinkTransformsFromPose(
      InputModel,
      PreviousInputPositions,
      ParentFromLinks,
      superdex::robotics::LinkTransformSpace::ParentFromLink,
      Error);
  if (!Error.IsOK()) {
    return;
  }
  FMochiToUnrealConverter ToUnreal(SuperDexSpace);
  DebugInputBaseInOutputBase = ToUnreal.TransformToUnreal(
      mochi::Invert(WorldFromOutputLinks[0]) * WorldFrameFromInputBaseFrame);
  DebugInputBaseInOutputBase.SetScale3D(FVector(InputModel.scale * NativeConfig.inputBotScale));
  DebugInputLinksInParent.SetNum(static_cast<int32>(ParentFromLinks.size()));
  for (int32 I = 0; I < DebugInputLinksInParent.Num(); ++I) {
    DebugInputLinksInParent[I] = ToUnreal.TransformToUnreal(ParentFromLinks[I]);
  }
  bHasDebugInputPose = true;
}

void USuperDexBotIKRetargeter::UpdateDebugInputPoseMesh(bool bFreshSolve) {
  USkeletalMesh* Mesh =
      (Config && Config->InputBot) ? Config->InputBot->GetSkeletalMesh() : nullptr;
  // Only a fresh solve: a held input pose would show stale tracking.
  const bool bShow = bDebugDrawInputPoseSkeleton && bFreshSolve && bHasDebugInputPose && Mesh &&
      GetOwner() && GetNumBones() > 0 &&
      DebugInputLinksInParent.Num() == static_cast<int32>(InputModel.links.size());
  if (!bShow) {
    if (DebugInputPoseMesh) {
      DebugInputPoseMesh->SetVisibility(false);
    }
    return;
  }
  if (!DebugInputPoseMesh) {
    DebugInputPoseMesh = NewObject<UPoseableMeshComponent>(GetOwner(), NAME_None, RF_Transient);
    DebugInputPoseMesh->SetupAttachment(this);
    DebugInputPoseMesh->SetUsingAbsoluteLocation(true);
    DebugInputPoseMesh->SetUsingAbsoluteRotation(true);
    DebugInputPoseMesh->SetUsingAbsoluteScale(true);
    DebugInputPoseMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    DebugInputPoseMesh->SetCastShadow(false);
    DebugInputPoseMesh->RegisterComponent();
  }
  if (DebugInputPoseMesh->GetSkinnedAsset() != Mesh) {
    DebugInputPoseMesh->SetSkinnedAssetAndUpdate(Mesh);
    // Bones are posed by index, so the mesh must list the input bot's links in order.
    bDebugInputPoseMeshMatchesBot =
        DebugInputPoseMesh->GetNumBones() == DebugInputLinksInParent.Num();
    for (int32 I = 0; bDebugInputPoseMeshMatchesBot && I < DebugInputLinksInParent.Num(); ++I) {
      bDebugInputPoseMeshMatchesBot = DebugInputPoseMesh->GetBoneName(I) ==
          FName(UTF8_TO_TCHAR(InputModel.links[I].name.c_str()));
    }
    if (!bDebugInputPoseMeshMatchesBot) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT(
              "[%hs] Input bot mesh [%s] bones do not follow the input bot's links; the input"
              " pose skeleton is not drawn."),
          __FUNCTION__,
          *Mesh->GetName());
    }
  }
  if (!bDebugInputPoseMeshMatchesBot) {
    DebugInputPoseMesh->SetVisibility(false);
    return;
  }
  // Like UMochiIKRetargeter: the override replaces every material; none keeps the mesh's own.
  if (DebugDrawInputPoseSkeletonMaterialOverride) {
    for (int32 Slot = 0; Slot < DebugInputPoseMesh->GetNumMaterials(); ++Slot) {
      if (DebugInputPoseMesh->GetMaterial(Slot) != DebugDrawInputPoseSkeletonMaterialOverride) {
        DebugInputPoseMesh->SetMaterial(Slot, DebugDrawInputPoseSkeletonMaterialOverride);
      }
    }
  } else if (!DebugInputPoseMesh->OverrideMaterials.IsEmpty()) {
    DebugInputPoseMesh->EmptyOverrideMaterials();
  }
  FTransform OutputBaseToWorld = GetComponentTransform();
  OutputBaseToWorld.SetScale3D(FVector::OneVector);
  DebugInputPoseMesh->SetWorldTransform(DebugInputBaseInOutputBase * OutputBaseToWorld);
  // Parent-relative transforms by index: SetBoneTransformByName in component space would read the
  // parents' cached component transforms, which are stale until the next refresh.
  for (int32 I = 0; I < DebugInputLinksInParent.Num(); ++I) {
    DebugInputPoseMesh->BoneSpaceTransforms[I] = DebugInputLinksInParent[I];
  }
  DebugInputPoseMesh->RefreshBoneTransforms();
  DebugInputPoseMesh->SetVisibility(true);
}

void USuperDexBotIKRetargeter::DrawDebugGoalFrames() const {
  UWorld* World = GetWorld();
  if (World == nullptr || DebugGoalFrames.IsEmpty() || GetNumBones() == 0) {
    return;
  }
  FTransform OutputBaseToWorld = GetComponentTransform();
  OutputBaseToWorld.SetScale3D(FVector::OneVector);
  constexpr float kFrameSize = 1.0f;
  constexpr float kLinkRadius = 0.2f;
  const auto DrawFrame = [&](const FTransform& FrameToWorld) {
    DrawDebugCoordinateSystem(
        World,
        FrameToWorld.GetLocation(),
        FrameToWorld.Rotator(),
        kFrameSize,
        false,
        -1.f,
        SDPG_Foreground);
  };
  const auto DrawLink =
      [&](const FTransform& LinkToWorld, const FTransform& FrameToWorld, FColor Color) {
        DrawDebugLine(
            World,
            FrameToWorld.GetLocation(),
            LinkToWorld.GetLocation(),
            Color,
            false,
            -1.f,
            SDPG_Foreground);
        DrawDebugCircle(
            World, LinkToWorld.GetLocation(), kLinkRadius, 16, Color, false, -1.f, SDPG_Foreground);
      };
  for (const FDebugGoalFrames& Frames : DebugGoalFrames) {
    const FTransform Goal = Frames.GoalInOutputBase * OutputBaseToWorld;
    const FTransform Constraint = Frames.ConstraintInOutputBase * OutputBaseToWorld;
    DrawFrame(Goal);
    DrawFrame(Constraint);
    DrawDebugLine(
        World,
        Goal.GetLocation(),
        Constraint.GetLocation(),
        FColor::Black,
        false,
        -1.f,
        SDPG_Foreground);
    DrawLink(Frames.OutputLinkInOutputBase * OutputBaseToWorld, Constraint, FColor::White);
    if (Frames.bHasInputLink) {
      DrawLink(Frames.InputLinkInOutputBase * OutputBaseToWorld, Goal, FColor::Yellow);
    }
  }
}

void USuperDexBotIKRetargeter::TickComponent(
    float DeltaTime,
    enum ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  bool bValid;
  {
    FScopeLock Lock(&ResultMutex);
    bValid = bCurrentValid;
  }
  // Only frames from a fresh solve: held goals would show stale tracking.
  if (bDebugDrawIKCoordinateFrames && bValid) {
    DrawDebugGoalFrames();
  }
  UpdateDebugInputPoseMesh(bValid);
}

void USuperDexBotIKRetargeter::OnTickManageCache(FSuperDexPose& Pose, bool bIsValid) {
  Super::OnTickManageCache(Pose, bIsValid);
  // A fresh solve already displayed feedback (see GetTransformedPose). Otherwise (tracking lost,
  // unconfident, clutch not yet initialized, ...) keep the mesh following the hand's reported
  // state, like the arm does. Display only: bCurrentValid, the published command and the
  // GetPose/GetCachedPose results are untouched, so consumers still see no valid pose.
  if (bIsValid || !bSubscribeToTeleopHandJointPositions || !Solver || !GetOwner()) {
    return;
  }
  auto* Node = GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>();
  TArray<double> Feedback;
  if (!IsValid(Node) || !Node->GetHandJointStates(Feedback) ||
      Feedback.Num() != OutputModel._numDofs ||
      !std::all_of(Feedback.begin(), Feedback.end(), [](double Q) { return FMath::IsFinite(Q); })) {
    return;
  }
  FWriteScopeLock Lock(CacheLock);
  // Without a previous pose there is no root to keep; don't invent one for a self-placing mesh.
  if (!bCacheInitialized && bUpdateComponentToRootTransformOnTick) {
    return;
  }
  FSuperDexPose Display = CachedOutputPose;
  if (USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(
          InitializedOutputBot, Feedback, Display)) {
    CachedOutputPose = MoveTemp(Display);
    bCacheInitialized = true;
  }
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
  if (DebugInputPoseMesh) {
    DebugInputPoseMesh->DestroyComponent();
    DebugInputPoseMesh = nullptr;
  }
  Super::OnComponentDestroyed(bDestroyingHierarchy);
}
#if WITH_EDITOR
void USuperDexBotIKRetargeter::PostEditChangeProperty(FPropertyChangedEvent& Event) {
  Super::PostEditChangeProperty(Event);
  ReleaseSolver();
  PlaceSourceAtConfigInputBaseFrame();
}
EDataValidationResult USuperDexBotIKRetargeter::IsDataValid(FDataValidationContext& Context) const {
  const EDataValidationResult Base = Super::IsDataValid(Context);
  FString Diagnostic;
  if (!ValidateConfiguration(Diagnostic)) {
    Context.AddError(FText::FromString(Diagnostic));
    return EDataValidationResult::Invalid;
  }
  return Base == EDataValidationResult::Invalid ? Base : EDataValidationResult::Valid;
}
#endif
