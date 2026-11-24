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

#include "SuperDexIKSolverObject.h"

#include "MochiConversions.h"
#include "MochiPhysicsModule.h"
#include "SuperDexBot.h"
#include "SuperDexBotLoader.h"

#include <mochi_core/utils/profile.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <utility>

namespace {

mochi::DynamicArray<superdex::robotics::BotIKJointCouplingParams> ConvertLegacyBotCouplings(
    superdex::robotics::BotPrefab const& Prefab,
    const TArray<FMochiJointCoupling>& JointCouplings,
    const FString& SolverName) {
  mochi::DynamicArray<superdex::robotics::BotIKJointCouplingParams> Result;
  // FName matching is case-insensitive in the legacy API. Canonicalize only that representation;
  // native validation and transmission creation are shared with every BotIK client.
  auto CanonicalName = [&Prefab](FName Name) -> mochi::DynamicString {
    for (superdex::robotics::BotJointPrefab const& Joint : Prefab.joints) {
      if (Name == FName(UTF8_TO_TCHAR(Joint.name.c_str()))) {
        return Joint.name;
      }
    }
    return TCHAR_TO_UTF8(*Name.ToString());
  };
  for (const FMochiJointCoupling& Coupling : JointCouplings) {
    if (!Coupling.bEnabled) {
      continue;
    }
    superdex::robotics::BotIKJointCouplingParams Native;
    Native.jointA = CanonicalName(Coupling.JointAName);
    Native.jointB = CanonicalName(Coupling.JointBName);
    Native.ratio = static_cast<mochi::real>(Coupling.Ratio);
    Native.offset = static_cast<mochi::real>(Coupling.Offset);
    Native.stiffness = static_cast<mochi::real>(Coupling.Stiffness);
    // Keep the legacy warning-and-skip and strictly-positive stiffness policy. New document
    // callers are validated atomically instead; zero-stiffness documents are legal inactive rows.
    const bool bLegacyValid = FMath::IsFinite(Coupling.Ratio) &&
        !FMath::IsNearlyZero(Coupling.Ratio) && FMath::IsFinite(Coupling.Offset) &&
        FMath::IsFinite(Coupling.Stiffness) && Coupling.Stiffness > 0 && Native.stiffness > 0;
    mochi::Error Error;
    if (bLegacyValid) {
      superdex::robotics::BotIK::ValidateJointCouplings(Prefab, {&Native, 1}, Error);
    }
    if (!bLegacyValid || !Error.IsOK()) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT("USuperDexIKSolverObject::[%s] ignoring joint coupling [%s]->[%s]: %hs"),
          *SolverName,
          *Coupling.JointAName.ToString(),
          *Coupling.JointBName.ToString(),
          bLegacyValid ? Error.GetDescription() : "invalid legacy ratio, offset or stiffness");
      continue;
    }
    Result.push_back(std::move(Native));
  }
  return Result;
}

} // namespace

USuperDexIKSolverObject* USuperDexIKSolverObject::CreateBotIKSolver(
    UObject* Outer,
    const USuperDexBot* OutputBot,
    const superdex::robotics::BotIKCreateParams& Params) {
  return CreateBotIKSolver(Outer, OutputBot, Params, {});
}

USuperDexIKSolverObject* USuperDexIKSolverObject::CreateBotIKSolver(
    UObject* Outer,
    const USuperDexBot* OutputBot,
    const superdex::robotics::BotIKCreateParams& Params,
    mochi::Span<superdex::robotics::BotIKJointCouplingConfig const> JointCouplings) {
  if (!IsInGameThread() || !Outer || !OutputBot) {
    return nullptr;
  }
  superdex::robotics::BotPrefab Prefab = OutputBot->GetBotPrefab();
  Prefab.worldFromRoot = mochi::TransformRT{};
  return CreatePreparedBotIKSolver(Outer, Prefab, Params, JointCouplings);
}

USuperDexIKSolverObject* USuperDexIKSolverObject::CreateBotIKSolver(
    UObject* Outer,
    const USuperDexBot* OutputBot,
    const superdex::robotics::BotIKConfig& Config) {
  if (!IsInGameThread() || !Outer || !OutputBot) {
    return nullptr;
  }
  mochi::DynamicString Diagnostic;
  if (!superdex::robotics::ValidateIKConfig(Config, Diagnostic)) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("OutputBot IK config invalid: %hs"), Diagnostic.c_str());
    return nullptr;
  }
  superdex::robotics::BotPrefab Prefab = OutputBot->GetBotPrefab();
  Prefab.worldFromRoot = mochi::TransformRT{};
  if (!Config.outputBotConfiguration.empty()) {
    Prefab.defaultPose = Config.outputBotConfiguration;
  }
  for (superdex::robotics::BotIKGoalConfig const& Goal : Config.goals) {
    if (superdex::robotics::FindLinkIndexByName(Prefab, Goal.outputBotLink) == INDEX_NONE) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT("OutputBot IK config source link not found: %hs"),
          Goal.outputBotLink.c_str());
      return nullptr;
    }
  }
  return CreatePreparedBotIKSolver(Outer, Prefab, Config.createParams, Config.jointCouplings);
}

USuperDexIKSolverObject* USuperDexIKSolverObject::CreatePreparedBotIKSolver(
    UObject* Outer,
    superdex::robotics::BotPrefab const& Prefab,
    superdex::robotics::BotIKCreateParams const& Params,
    mochi::Span<superdex::robotics::BotIKJointCouplingConfig const> JointCouplings) {
  mochi::DynamicString Diagnostic;
  if (!superdex::robotics::ValidateIKJointCouplings(Prefab, JointCouplings, Diagnostic)) {
    UE_LOG(
        LogMochiPhysics, Warning, TEXT("Bot IK joint couplings invalid: %hs"), Diagnostic.c_str());
    return nullptr;
  }
  mochi::DynamicArray<superdex::robotics::BotIKJointCouplingParams> NativeCouplings =
      superdex::robotics::ToBotIKJointCouplingParams(JointCouplings);
  FSuperDexBotLoader Loader;
  mochi::Error Error;
  superdex::robotics::BotIK* Native = superdex::robotics::BotIK::Create(
      IMochiPhysicsModule::Get()->GetMochiContext(),
      Prefab,
      Params,
      NativeCouplings,
      Loader,
      Error);
  if (!Error.IsOK() || !Native) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("Bot IK creation failed: %hs"), Error.GetDescription());
    superdex::robotics::BotIK::Destroy(Native);
    return nullptr;
  }
  USuperDexIKSolverObject* Object = NewObject<USuperDexIKSolverObject>(Outer);
  Object->NativeIK = Native;
  Object->bNativeJointLayout = true;
  Object->bIsInitialized = true;
  Object->NumLinks = Object->NativeIK->GetNumLinks();
  Object->UpdateBotResults();
  return Object;
}

bool USuperDexIKSolverObject::SetBotGoal(
    int32 OutputBotLinkIndex,
    const mochi::TransformRT& WorldFrameFromConstraintFrame,
    mochi::real PositionStiffness,
    mochi::real RotationStiffness) {
  return SetBotGoal(
      OutputBotLinkIndex,
      mochi::TransformRT{},
      WorldFrameFromConstraintFrame,
      PositionStiffness,
      RotationStiffness);
}

bool USuperDexIKSolverObject::SetBotGoal(
    int32 OutputBotLinkIndex,
    const mochi::TransformRT& LinkFrameFromConstraintFrame,
    const mochi::TransformRT& WorldFrameFromConstraintFrame,
    mochi::real PositionStiffness,
    mochi::real RotationStiffness) {
  if (!IsInGameThread() || !bNativeJointLayout || !NativeIK) {
    return false;
  }
  // Validate the whole update before touching either independent goal component.
  const bool bValid = OutputBotLinkIndex >= 0 && OutputBotLinkIndex < NativeIK->GetNumLinks() &&
      mochi::IsFinite(LinkFrameFromConstraintFrame) &&
      mochi::IsFinite(WorldFrameFromConstraintFrame) &&
      mochi::NearEqual(mochi::Norm(LinkFrameFromConstraintFrame.GetRotation()),
                       mochi::real(1),
                       mochi::real(1e-4)) &&
      mochi::NearEqual(mochi::Norm(WorldFrameFromConstraintFrame.GetRotation()),
                       mochi::real(1),
                       mochi::real(1e-4)) &&
      mochi::IsFinite(PositionStiffness) && PositionStiffness >= 0 &&
      mochi::IsFinite(RotationStiffness) && RotationStiffness >= 0;
  if (!bValid) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("Bot IK attachment goal update has invalid input"));
    return false;
  }
  mochi::Error Error;
  if (PositionStiffness > 0) {
    NativeIK->SetPositionGoal(
        OutputBotLinkIndex,
        LinkFrameFromConstraintFrame.GetTranslation(),
        WorldFrameFromConstraintFrame.GetTranslation(),
        PositionStiffness,
        Error);
  } else {
    NativeIK->ClearPositionGoal(OutputBotLinkIndex, Error);
  }
  if (RotationStiffness > 0) {
    NativeIK->SetRotationGoal(
        OutputBotLinkIndex,
        LinkFrameFromConstraintFrame.GetRotation(),
        WorldFrameFromConstraintFrame.GetRotation(),
        RotationStiffness,
        Error);
  } else {
    NativeIK->ClearRotationGoal(OutputBotLinkIndex, Error);
  }
  if (!Error.IsOK()) {
    UE_LOG(
        LogMochiPhysics, Warning, TEXT("Bot IK goal update failed: %hs"), Error.GetDescription());
  }
  return Error.IsOK();
}

bool USuperDexIKSolverObject::SetBotInputConfiguration(
    superdex::robotics::BotPrefab const& InputBot,
    mochi::Span<mochi::real const> Configuration,
    mochi::TransformRT const& WorldFrameFromInputBaseFrame,
    mochi::real InputBotScale,
    mochi::Span<superdex::robotics::BotIKGoalConfig const> Goals) {
  if (!IsInGameThread() || !bNativeJointLayout || !NativeIK) {
    return false;
  }
  mochi::Error Error;
  NativeIK->SetInputBotConfiguration(
      InputBot, Configuration, WorldFrameFromInputBaseFrame, InputBotScale, Goals, Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("Bot IK input configuration update failed: %hs"),
        Error.GetDescription());
  }
  return Error.IsOK();
}

bool USuperDexIKSolverObject::SolveBotIK(bool& bOutGoalsReached) {
  bOutGoalsReached = false;
  if (!IsInGameThread() || !bNativeJointLayout || !NativeIK) {
    return false;
  }
  mochi::Error Error;
  bOutGoalsReached = NativeIK->SolveIK(Error);
  if (!Error.IsOK()) {
    bOutGoalsReached = false;
    UE_LOG(LogMochiPhysics, Warning, TEXT("Bot IK solve failed: %hs"), Error.GetDescription());
    return false;
  }
  UpdateBotResults();
  return true;
}

mochi::Span<mochi::TransformRT const> USuperDexIKSolverObject::GetOutputBotWorldFromLinks() const {
  if (!IsInGameThread() || !bNativeJointLayout || !NativeIK) {
    return {};
  }
  return NativeIK->GetWorldFromLinks();
}

bool USuperDexIKSolverObject::CopyOutputBotFKAtJointConfiguration(
    TConstArrayView<double> Positions,
    mochi::DynamicArray<mochi::TransformRT>& OutWorldFromLinks) {
  if (!IsInGameThread() || !bNativeJointLayout || !NativeIK ||
      Positions.Num() != NativeIK->GetNumDofs()) {
    return false;
  }
  mochi::DynamicArray<mochi::real> Before, Display;
  for (const mochi::real Q : NativeIK->GetJointPositions()) {
    Before.push_back(Q);
  }
  for (const double Q : Positions) {
    if (!FMath::IsFinite(Q)) {
      return false;
    }
    Display.push_back(static_cast<mochi::real>(Q));
  }
  mochi::Error Error;
  NativeIK->SetJointPositions(Display, Error);
  if (!Error.IsOK()) {
    return false;
  }
  mochi::DynamicArray<mochi::TransformRT> Snapshot;
  for (const mochi::TransformRT& Link : NativeIK->GetWorldFromLinks()) {
    Snapshot.push_back(Link);
  }
  NativeIK->SetJointPositions(Before, Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogMochiPhysics, Error, TEXT("Could not restore the private BotIK warm start after FK."));
    return false;
  }
  OutWorldFromLinks = MoveTemp(Snapshot);
  return true;
}

USuperDexIKSolverObject* USuperDexIKSolverObject::CreateIKSolver(
    UObject* Outer,
    FString SolverName,
    const UMochiArticulationBase* Articulation,
    const FTransform& WorldFromRoot,
    const TArray<FTransform>& RootFromLinks,
    bool bFixedBase,
    const FMochiIKSolverParams& IKSolverParams,
    const TArray<FMochiJointCoupling>& JointCouplings,
    bool bExcludeCollisionShapes) {
  MOCHI_PROFILE_SCOPE();
  // check that outer is valid
  if (Outer == nullptr) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "USuperDexIKSolverObject::Failed to create USuperDexIKSolverObject because Outer was null!"))
    return nullptr;
  }

  // Create new object
  FString NameStr = SolverName + "___IKSolver";
  USuperDexIKSolverObject* IKSolverObj = NewObject<USuperDexIKSolverObject>(Outer, FName(*NameStr));
  check(IKSolverObj);

  const USuperDexBot* BotAsset = Cast<USuperDexBot>(Articulation);
  if (BotAsset) {
    superdex::robotics::BotPrefab Prefab = BotAsset->GetBotPrefab();
    FUnrealToMochiConverter U2M;
    Prefab.worldFromRoot = U2M.TransformToMochi_RT(WorldFromRoot);
    mochi::DynamicArray<superdex::robotics::BotIKJointCouplingParams> NativeCouplings =
        ConvertLegacyBotCouplings(Prefab, JointCouplings, SolverName);

    superdex::robotics::BotIKCreateParams Params;
    Params.fixedBase = bFixedBase;
    Params.excludeCollisionShapes = bExcludeCollisionShapes;
    Params.solverParams = ::ToMochi(IKSolverParams);
    FSuperDexBotLoader Loader;
    mochi::Error Error;
    IKSolverObj->NativeIK = superdex::robotics::BotIK::Create(
        IMochiPhysicsModule::Get()->GetMochiContext(),
        Prefab,
        Params,
        NativeCouplings,
        Loader,
        Error);
    if (!Error.IsOK()) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("USuperDexIKSolverObject::Failed to add bot to IK solver scene: %hs"),
          Error.GetDescription());
      return nullptr;
    }

    IKSolverObj->NumLinks = IKSolverObj->NativeIK->GetNumLinks();
    IKSolverObj->NumBotBaseDofs =
        !bFixedBase && Prefab.joints[0].type == mochi::ArticulatedJointType::Free ? 6 : 0;
    IKSolverObj->NumMochiBaseDofs = bFixedBase ? 0 : 6;
    if (!bFixedBase && IKSolverObj->NumBotBaseDofs == 0) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "USuperDexIKSolverObject::[%s] was created with bFixedBase=false but the"
              " articulation has a welded root, so GetSolvedDofs() will drop 6 real joint dofs."
              " Set bUseFixedArticulationBase on the IK params to solve this articulation"
              " correctly."),
          *SolverName);
    }
    IKSolverObj->bIsInitialized = true;
    return IKSolverObj;
  }

  const UMochiArticulation* ArticulationAsset = Cast<UMochiArticulation>(Articulation);
  if (!ArticulationAsset) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "USuperDexIKSolverObject::Articulation was not a UMochiArticulation or USuperDexBot!"));
    return nullptr;
  }

  // The IKSolver uses a normal scene and controls its solve/step functions. Destruction will not
  // work correctly with an AsyncScene. The IKSolver will override the settings upon creation so we
  // just pass in default. The IKSolver also handles destroying the scene so we indicate that
  // through our friend class status.
  IKSolverObj->MochiSceneObject =
      UMochiSceneObject::CreateScene(IKSolverObj, EMochiSceneType::Sync, SolverName, {}, true);
  check(IKSolverObj->MochiSceneObject);
  IKSolverObj->MochiSceneObject->bDestructionOwnedByFriend = true;

  if (ArticulationAsset) {
    // UMochiArticulation path: use CreateArticulation directly
    if (bExcludeCollisionShapes) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "USuperDexIKSolverObject::[%s] bExcludeCollisionShapes is not supported for"
              " UMochiArticulation; building with collision geometry. Use a USuperDexBot"
              " articulation to exclude collision shapes from the IK solve."),
          *SolverName);
    }
    IKSolverObj->ArticulatedActorObject = UMochiArticulatedActorObject::CreateArticulation(
        IKSolverObj,
        SolverName,
        ArticulationAsset,
        WorldFromRoot,
        RootFromLinks,
        /* ContactParams */ nullptr,
        /* Layer */ Layer,
        bFixedBase,
        /* bHasGravity */ false,
        /* InitialJointVelocities */ {},
        [IKSolverObj](
            const TArray<mochi::TransformRT>& WorldFromLink,
            const TArray<mochi::real>& JointPositions,
            mochi::Actor* ArticulatedActor,
            mochi::Scene* Scene) {
          IKSolverObj->OnArticulatedActorGetPose(
              WorldFromLink, JointPositions, ArticulatedActor, Scene);
        },
        /* bAddPoseController */ false,
        /* PoseControllerGains */ {},
        /* ControlCallback */ nullptr,
        /* LinksThatQueryContactPoints */ TArray<FName>{},
        /* bSkinQueriesContactPoints */ false,
        IKSolverObj->MochiSceneObject);
  }
  check(IKSolverObj->ArticulatedActorObject);

  // Legacy value. Only the fixed base path below is allowed to change it, because some existing
  // bot assets already have a Hard root joint while every caller passes bFixedBase=false, and
  // correcting their dof slice here would change existing behaviour.
  IKSolverObj->NumMochiBaseDofs = bFixedBase ? 0 : 6;

  // TODO: consider defering the creation of the IKSolver till the first attempt to solve so that
  // the scene can still be edited after the system is created in case we need to add things in the
  // future.

  // Create MochiIKSolver and finalize/lock scene
  IKSolverObj->MochiSceneObject->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
    // Read the articulation's shape while the scene is still available. ArticulatedShapeInfo hands
    // back non-owning Spans into scene storage, so everything we keep must be copied out here.
    mochi::ErrorLog ShapeErr{};
    mochi::Actor* ArticulatedActor =
        SyncScene->GetActor(IKSolverObj->ArticulatedActorObject->GetArticulatedActorHandle());
    if (ArticulatedActor) {
      const mochi::ArticulatedShapeInfo ShapeInfo =
          ArticulatedActor->GetArticulatedShapeInfo(ShapeErr);
      // parents and jointTypes are both per-link, but take the smaller extent so a malformed
      // articulation cannot make the parent walk below read out of bounds.
      const int NumShapeLinks = ShapeErr.IsOK()
          ? static_cast<int>(FMath::Min(ShapeInfo.jointTypes.size(), ShapeInfo.parents.size()))
          : 0;
      if (NumShapeLinks > 0) {
        // A Hard root joint is how both the bot prefab and UMochiArticulation express a base
        // welded to the world. Unlike ExtractArticulatedActorState, which only treats a Free root
        // as base DOFs, this keeps the legacy slicing described above.
        const bool bActorIsFixedBase = ShapeInfo.jointTypes[0] == mochi::ArticulatedJointType::Hard;
        if (bFixedBase) {
          IKSolverObj->NumMochiBaseDofs = bActorIsFixedBase ? 0 : 6;
        } else if (bActorIsFixedBase) {
          // Pre-existing hazard, deliberately not corrected here: the articulation has no base dofs
          // but we keep stripping 6, so GetSolvedDofs() returns the wrong slice. Harmless while the
          // caller never solves; if you hit this, fix the caller rather than this line, since other
          // callers depend on the current slicing.
          UE_LOG(
              LogMochiPhysics,
              Warning,
              TEXT(
                  "USuperDexIKSolverObject::[%s] was created with bFixedBase=false but the"
                  " articulation has a welded root, so GetSolvedDofs() will drop 6 real joint dofs."
                  " Set bUseFixedArticulationBase on the IK params to solve this articulation"
                  " correctly."),
              *SolverName);
        }

        // A link is rigid wrt the root when every joint on its path to the root is a weld. Bound
        // both the hop count and the parent index so a malformed articulation cannot spin or read
        // out of bounds here.
        IKSolverObj->LinkIsRigidlyAttachedToRootLink.Init(false, NumShapeLinks);
        for (int LinkIndex = 0; LinkIndex < NumShapeLinks; ++LinkIndex) {
          int CurrentIndex = LinkIndex;
          bool bIsRigid = true;
          for (int HopIndex = 0; CurrentIndex > 0 && HopIndex <= NumShapeLinks; ++HopIndex) {
            if (ShapeInfo.jointTypes[CurrentIndex] != mochi::ArticulatedJointType::Hard) {
              bIsRigid = false;
              break;
            }
            const int ParentIndex = static_cast<int>(ShapeInfo.parents[CurrentIndex]);
            if (ParentIndex < 0 || ParentIndex >= NumShapeLinks) {
              bIsRigid = false;
              break;
            }
            CurrentIndex = ParentIndex;
          }
          IKSolverObj->LinkIsRigidlyAttachedToRootLink[LinkIndex] = bIsRigid && (CurrentIndex == 0);
        }

        // Tie pairs of joints together for the solve. Done here because we already hold the actor
        // and its shape. Transmissions are read from the actor at step time, so this only has to
        // happen before the first solve, but doing it here keeps it to a single scene command.
        // jointNames and dofInfo are both per-joint; take the smaller extent so a malformed
        // articulation cannot make the lookups below read out of bounds.
        const int NumJoints =
            static_cast<int>(FMath::Min(ShapeInfo.jointNames.size(), ShapeInfo.dofInfo.size()));
        auto FindJoint = [&ShapeInfo, NumJoints](FName Name) -> int {
          for (int JointIndex = 0; JointIndex < NumJoints; ++JointIndex) {
            if (Name == FName(UTF8_TO_TCHAR(ShapeInfo.jointNames[JointIndex].c_str()))) {
              return JointIndex;
            }
          }
          return INDEX_NONE;
        };
        for (const FMochiJointCoupling& Coupling : JointCouplings) {
          if (!Coupling.bEnabled) {
            continue;
          }
          const int JointA = FindJoint(Coupling.JointAName);
          const int JointB = FindJoint(Coupling.JointBName);

          // Validate here rather than letting Mochi error out, so that one bad entry produces a
          // readable message and is skipped instead of aborting the whole solver setup.
          const TCHAR* Rejection = nullptr;
          if (JointA == INDEX_NONE || JointB == INDEX_NONE) {
            Rejection = TEXT("a joint name does not exist in the articulation");
          } else if (JointA == JointB) {
            Rejection = TEXT("both names resolve to the same joint");
          } else if (
              ShapeInfo.dofInfo[JointA].GetSize() != 1 ||
              ShapeInfo.dofInfo[JointB].GetSize() != 1) {
            Rejection = TEXT("only single-dof joints can be coupled, and one of these is not");
          } else if (!FMath::IsFinite(Coupling.Ratio) || FMath::IsNearlyZero(Coupling.Ratio)) {
            Rejection = TEXT("Ratio must be finite and non-zero");
          } else if (!FMath::IsFinite(Coupling.Offset)) {
            Rejection = TEXT("Offset must be finite");
          } else if (!FMath::IsFinite(Coupling.Stiffness) || Coupling.Stiffness <= 0.0) {
            Rejection = TEXT("Stiffness must be finite and positive");
          }
          if (Rejection) {
            UE_LOG(
                LogMochiPhysics,
                Warning,
                TEXT("USuperDexIKSolverObject::[%s] ignoring joint coupling [%s]->[%s]: %s."),
                *SolverName,
                *Coupling.JointAName.ToString(),
                *Coupling.JointBName.ToString(),
                Rejection);
            continue;
          }

          // displacement = Ratio * jointA - jointB, driven to -Offset, which is exactly
          // jointB = Ratio * jointA + Offset.
          mochi::experimental::LinearTransmissionParams TransmissionParams;
          TransmissionParams.jointIndices = {JointA, JointB};
          TransmissionParams.jointCoefficients = {
              static_cast<mochi::real>(Coupling.Ratio), static_cast<mochi::real>(-1.0)};

          // Plain Error rather than ErrorLog: the failure is reported explicitly below, and
          // ErrorLog would also log it from its destructor.
          mochi::Error CouplingErr;
          const int TransmissionIndex = mochi::experimental::AddLinearTransmission(
              ArticulatedActor, TransmissionParams, CouplingErr);
          if (CouplingErr.IsOK()) {
            mochi::experimental::DisplacementControlActuatorParams ActuatorParams;
            ActuatorParams.targetDisplacement = static_cast<mochi::real>(-Coupling.Offset);
            ActuatorParams.stiffness = static_cast<mochi::real>(Coupling.Stiffness);
            // Damping keeps its native default of zero: the IK solve steps with an infinite
            // time step, so any damping term contributes nothing.
            // A joint coupling is a rigid linkage, so it must be able to push as well as pull.
            // Left false, the actuator clamps to zero force on one side and only half couples.
            ActuatorParams.allowCompressiveForce = true;
            mochi::experimental::AttachDisplacementControlActuator(
                ArticulatedActor, TransmissionIndex, ActuatorParams, CouplingErr);
          }
          if (!CouplingErr.IsOK()) {
            UE_LOG(
                LogMochiPhysics,
                Error,
                TEXT(
                    "USuperDexIKSolverObject::[%s] failed to create joint coupling [%s]->[%s]: %s"),
                *SolverName,
                *Coupling.JointAName.ToString(),
                *Coupling.JointBName.ToString(),
                *FString(CouplingErr.GetDescription()));
          }
        }
      }
    }

    // remove collision before creating IK solver since scene properties are locked after
    SyncScene->EnableLayerContactSymmetric(Layer, Layer, false, mochi::ErrorAssert{});

    // Get MochiPhysics world
    mochi::Context* MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
    check(MochiContext);
    IKSolverObj->IKSolver =
        mochi::experimental::CreateIKSolver(SyncScene, MochiContext, mochi::ErrorAssert{});
    mochi::experimental::IKSolverParams Params = ::ToMochi(IKSolverParams);
    IKSolverObj->IKSolver->SetSolverParams(Params);
    check(IKSolverObj->IKSolver);
  });
  IKSolverObj->MochiSceneObject->WaitForQueuedCommands();

  IKSolverObj->NumLinks = IKSolverObj->ArticulatedActorObject->GetNestedLinkActorHandles().Num();

  IKSolverObj->bIsInitialized = true;

  return IKSolverObj;
}

// The CreatePositionTarget() function will remove any existing position target for the specified
// actor then add a new constraint. It adds the constraint as a normal spring stiffness constraint.
// So weight is actually the stiffness of a spring in standard mochi stiffness units N/m or Nm/rad.
void USuperDexIKSolverObject::CreatePositionTargetForLink(
    int LinkIndex,
    FVector LocalPositionToLink,
    FVector TargetPositionToWorld,
    float Weight) {
  if (!bIsInitialized) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::Cannot create goals on uninitialized IKSolver."));
    return;
  }
  if (LinkIndex < 0 || LinkIndex > NumLinks - 1) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "USuperDexIKSolverObject::Attempted to create target for out of bounds link index. Returning without action."));
    return;
  }
  mochi::Error Err;
  FUnrealToMochiConverter U2M;
  if (NativeIK) {
    NativeIK->SetPositionGoal(
        LinkIndex,
        U2M.TranslationToMochi(LocalPositionToLink),
        U2M.TranslationToMochi(TargetPositionToWorld),
        Weight,
        Err);
  } else {
    MochiSceneObject->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
      mochi::ActorHandle LinkHandle =
          ArticulatedActorObject->GetNestedLinkActorHandles()[LinkIndex];
      IKSolver->CreatePositionTarget(
          LinkHandle,
          U2M.TranslationToMochi(LocalPositionToLink),
          U2M.TranslationToMochi(TargetPositionToWorld),
          Weight,
          Err);
    });
    MochiSceneObject->WaitForQueuedCommands();
  }
  if (!Err.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::failure to create constraint: %s"),
        *FString(Err.GetDescription()));
  }
}

void USuperDexIKSolverObject::CreateRotationTargetForLink(
    int LinkIndex,
    FQuat LocalRotationToLink,
    FQuat TargetRotationToWorld,
    float Weight) {
  if (!bIsInitialized) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::Cannot create goals on uninitialized IKSolver."));
    return;
  }
  if (LinkIndex < 0 || LinkIndex > NumLinks - 1) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "USuperDexIKSolverObject::Attempted to create target for out of bounds link index. Returning without action."));
    return;
  }
  mochi::Error Err;
  FUnrealToMochiConverter U2M;
  if (NativeIK) {
    NativeIK->SetRotationGoal(
        LinkIndex,
        U2M.RotationToMochi(LocalRotationToLink),
        U2M.RotationToMochi(TargetRotationToWorld),
        Weight,
        Err);
  } else {
    MochiSceneObject->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
      mochi::ActorHandle LinkHandle =
          ArticulatedActorObject->GetNestedLinkActorHandles()[LinkIndex];
      IKSolver->CreateRotationTarget(
          LinkHandle,
          U2M.RotationToMochi(LocalRotationToLink).ToRotationVector(),
          U2M.RotationToMochi(TargetRotationToWorld).ToRotationVector(),
          Weight,
          Err);
    });
    MochiSceneObject->WaitForQueuedCommands();
  }
  if (!Err.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::failure to create constraint: %s"),
        *FString(Err.GetDescription()));
  }
}

void USuperDexIKSolverObject::ClearPositionTargetForLink(int LinkIndex) {
  if (!bIsInitialized) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::Cannot clear goals on uninitialized IKSolver."));
    return;
  }
  if (LinkIndex < 0 || LinkIndex > NumLinks - 1) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "USuperDexIKSolverObject::Attempted to clear goals for out of bounds link index. Returning without action."));
    return;
  }
  mochi::Error Err;
  if (NativeIK) {
    NativeIK->ClearPositionGoal(LinkIndex, Err);
  } else {
    MochiSceneObject->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
      mochi::ActorHandle LinkHandle =
          ArticulatedActorObject->GetNestedLinkActorHandles()[LinkIndex];
      IKSolver->ClearPositionTarget(LinkHandle, Err);
    });
    MochiSceneObject->WaitForQueuedCommands();
  }
  if (!Err.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::Could not clear IK target from link. %s"),
        *FString(Err.GetDescription()));
  }
}
void USuperDexIKSolverObject::ClearRotationTargetForLink(int LinkIndex) {
  if (!bIsInitialized) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::Cannot clear goals on uninitialized IKSolver."));
    return;
  }
  if (LinkIndex < 0 || LinkIndex > NumLinks - 1) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "USuperDexIKSolverObject::Attempted to clear goals for out of bounds link index. Returning without action."));
    return;
  }
  mochi::Error Err;
  if (NativeIK) {
    NativeIK->ClearRotationGoal(LinkIndex, Err);
  } else {
    MochiSceneObject->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
      mochi::ActorHandle LinkHandle =
          ArticulatedActorObject->GetNestedLinkActorHandles()[LinkIndex];
      IKSolver->ClearRotationTarget(LinkHandle, Err);
    });
    MochiSceneObject->WaitForQueuedCommands();
  }
  if (!Err.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::Could not clear IK target from link. %s"),
        *FString(Err.GetDescription()));
  }
}

void USuperDexIKSolverObject::ClearAllTargets() {
  if (!bIsInitialized) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::Cannot clear goals on uninitialized IKSolver."));
    return;
  }
  mochi::Error ErrPos;
  mochi::Error ErrRot;
  if (NativeIK) {
    NativeIK->ClearAllGoals(ErrPos);
  } else {
    MochiSceneObject->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
      TArray<mochi::ActorHandle> const& LinkHandles =
          ArticulatedActorObject->GetNestedLinkActorHandles();
      for (int LinkIndex = 0; LinkIndex < NumLinks; LinkIndex++) {
        IKSolver->ClearPositionTarget(LinkHandles[LinkIndex], ErrPos);
        IKSolver->ClearRotationTarget(LinkHandles[LinkIndex], ErrRot);
      }
    });
    MochiSceneObject->WaitForQueuedCommands();
  }
  if ((!ErrPos.IsOK()) || (!ErrRot.IsOK())) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::Could not clear IK target from link. pos[%s] rot[%s]"),
        *FString(ErrPos.GetDescription()),
        *FString(ErrRot.GetDescription()));
  }
}

bool USuperDexIKSolverObject::SolveIK() {
  if (!IsValid()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("USuperDexIKSolverObject::Cannot solve uninitialized IK."));
    return false;
  }
  bool bGoalsReached = false;
  mochi::Error Err;
  if (NativeIK) {
    if (!ensureMsgf(IsInGameThread(), TEXT("Bot IK must be evaluated on the game thread."))) {
      return false;
    }
    bGoalsReached = NativeIK->SolveIK(Err);
    if (Err.IsOK()) {
      UpdateBotResults();
    }
  } else {
    MochiSceneObject->QueueOrExecuteCommand(
        [&](mochi::Scene* SyncScene) { bGoalsReached = IKSolver->SolveIK(Err); });
    MochiSceneObject->WaitForQueuedCommands();
  }
  if (!Err.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("USuperDexIKSolverObject::Error during solve. %s"),
        *FString(Err.GetDescription()));
  }
  return bGoalsReached;
}

void USuperDexIKSolverObject::UpdateBotResults() {
  FMochiToUnrealConverter M2U(
      bNativeJointLayout ? SuperDexSpace : UMochiPhysicsSettings::GetActiveSimulationSpace());
  const mochi::Span<mochi::TransformRT const> WorldFromLinks = NativeIK->GetWorldFromLinks();
  SolvedLinkToWorld.SetNum(static_cast<int>(WorldFromLinks.size()));
  for (int LinkIndex = 0; LinkIndex < SolvedLinkToWorld.Num(); ++LinkIndex) {
    SolvedLinkToWorld[LinkIndex] = M2U.TransformToUnreal(WorldFromLinks[LinkIndex]);
  }

  const mochi::Span<mochi::real const> JointPositions = NativeIK->GetJointPositions();
  // Keep the historical requested-free/actually-welded slice local to UE. BotIK itself always
  // returns the correct bot-space joint layout.
  const int ExtraDofsToStrip = FMath::Max(0, NumMochiBaseDofs - NumBotBaseDofs);
  const int NumJointPositions = static_cast<int>(JointPositions.size());
  ensureMsgf(
      NumJointPositions >= ExtraDofsToStrip,
      TEXT("IK solve returned %d joint dofs but %d are being stripped."),
      NumJointPositions,
      ExtraDofsToStrip);
  SolvedDofs.SetNum(FMath::Max(0, NumJointPositions - ExtraDofsToStrip));
  for (int DofIndex = 0; DofIndex < SolvedDofs.Num(); ++DofIndex) {
    SolvedDofs[DofIndex] = JointPositions[DofIndex + ExtraDofsToStrip];
  }
}

// Legacy articulation query callback.
void USuperDexIKSolverObject::OnArticulatedActorGetPose(
    const TArray<mochi::TransformRT>& InWorldFromLink,
    const TArray<mochi::real>& InJointDofs,
    mochi::Actor* ArticulatedActor,
    mochi::Scene* /*Scene*/) {
  MOCHI_PROFILE_SCOPE();
  if (!bIsInitialized) {
    return;
  }
  FMochiToUnrealConverter M2U;
  SolvedLinkToWorld.SetNum(InWorldFromLink.Num());
  for (int LinkIndex = 0; LinkIndex < InWorldFromLink.Num(); LinkIndex++) {
    SolvedLinkToWorld[LinkIndex] = M2U.TransformToUnreal(InWorldFromLink[LinkIndex]);
  }
  // A shortfall means the solver and the actor disagree about the base, which would silently
  // produce a truncated pose. Clamp so we cannot SetNum a negative count, but say so.
  ensureMsgf(
      InJointDofs.Num() >= NumMochiBaseDofs,
      TEXT("IK solve returned %d dofs but %d base dofs are being stripped."),
      InJointDofs.Num(),
      NumMochiBaseDofs);
  SolvedDofs.SetNum(FMath::Max(0, InJointDofs.Num() - NumMochiBaseDofs));
  for (int DofIndex = NumMochiBaseDofs; DofIndex < InJointDofs.Num(); DofIndex++) {
    const int JointIndex = DofIndex - NumMochiBaseDofs;
    SolvedDofs[JointIndex] = InJointDofs[DofIndex];
  }
}

const TArray<FTransform>& USuperDexIKSolverObject::GetSolvedLinkToWorldTransforms() const {
  return SolvedLinkToWorld;
}

const TArray<double>& USuperDexIKSolverObject::GetSolvedDofs() const {
  return SolvedDofs;
}

bool USuperDexIKSolverObject::IsValid() const {
  if (NativeIK) {
    return bIsInitialized;
  }
  return (MochiSceneObject != nullptr) && MochiSceneObject->IsValid() && (IKSolver != nullptr) &&
      (ArticulatedActorObject != nullptr) && ArticulatedActorObject->IsValid();
}

int USuperDexIKSolverObject::GetNumDofs() const {
  if (NativeIK) {
    return NativeIK->GetNumDofs() + (bNativeJointLayout ? 0 : NumBotBaseDofs);
  }
  return ArticulatedActorObject ? ArticulatedActorObject->GetNumDofs() : 0;
}

bool USuperDexIKSolverObject::IsLinkRigidlyAttachedToRootLink(int32 LinkIndex) const {
  if (NativeIK) {
    return NativeIK->IsLinkRigidlyAttachedToRootLink(LinkIndex);
  }
  return LinkIsRigidlyAttachedToRootLink.IsValidIndex(LinkIndex) &&
      LinkIsRigidlyAttachedToRootLink[LinkIndex];
}

void USuperDexIKSolverObject::DestroyMochiResources() {
  MOCHI_PROFILE_SCOPE();
  if (bNativeJointLayout && !ensure(IsInGameThread())) {
    return;
  }
  // Destruction order matters here; see the comments below.

  // Clear IK goals before destroying the articulation - its actor removal cascade-deletes
  // constraints, leaving dangling pointers in the IKSolver's maps.
  if (bIsInitialized && IKSolver && MochiSceneObject && ArticulatedActorObject) {
    ClearAllTargets();
  }
  bIsInitialized = false;
  superdex::robotics::BotIK::Destroy(NativeIK);
  NativeIK = nullptr;
  // Destroy the legacy actor wrapper before the solver destroys its scene.
  if (ArticulatedActorObject) {
    ArticulatedActorObject->DestroyMochiResources();
    ArticulatedActorObject->MarkAsGarbage();
    ArticulatedActorObject = nullptr;
  }
  // Finally, destroy the IK solver which will actually destroy the mochi::Scene.
  if (IKSolver) {
    mochi::Context* MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
    check(MochiContext != nullptr);
    mochi::experimental::DestroyIKSolver(IKSolver, MochiContext, mochi::ErrorAssert{});
    IKSolver = nullptr;
  }
  // The mochi::Scene should already be destroyed by the IKSolver, calling this to guarantee UE
  // object state
  if (MochiSceneObject) {
    MochiSceneObject->DestroyScene();
    MochiSceneObject->MarkAsGarbage();
    MochiSceneObject = nullptr;
  }
  LinkIsRigidlyAttachedToRootLink.Empty();
}

void USuperDexIKSolverObject::BeginDestroy() {
  MOCHI_PROFILE_SCOPE();
  // Destroy Mochi resources before allowing UE GC to run since this UObject
  // owns other UObjects that should be alive during DestroyMochiResources.
  DestroyMochiResources();
  UObject::BeginDestroy();
}
