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

#include "Controllers/Controller_MochiArticulatedPose.h"

#include "SuperDexTeleopModule.h"

#include "ISuperDexRoboticsModule.h"
#include "MochiConversions.h"
#include "MochiPhysicsModule.h"
#include "SuperDexBot.h"
#include "SuperDexBotComponent.h"
#include "SuperDexPoseableComponent.h"

namespace {
mochi::DynamicArray<mochi::PoseTrackingParams> ConvertTrackingArray(
    const TArray<FMochiPoseTrackingParams>& In) {
  mochi::DynamicArray<mochi::PoseTrackingParams> Out(In.Num());
  for (int i = 0; i < In.Num(); ++i) {
    Out[i].stiffness = static_cast<mochi::real>(In[i].Stiffness);
    Out[i].damping = static_cast<mochi::real>(In[i].Damping);
    Out[i].saturation = static_cast<mochi::real>(In[i].Saturation);
  }
  return Out;
}
} // namespace

superdex::robotics::ControllerMochiArticulatedPose::Params
UController_MochiArticulatedPose::ConvertToMochiPoseParams(
    const FMochiPoseControllerParams& InParams) {
  superdex::robotics::ControllerMochiArticulatedPose::Params Out;
  Out.poseControllerParams.linkPosTracking = ConvertTrackingArray(InParams.LinkPosTracking);
  Out.poseControllerParams.linkRotTracking = ConvertTrackingArray(InParams.LinkRotTracking);
  Out.poseControllerParams.jointTracking = ConvertTrackingArray(InParams.JointTracking);
  return Out;
}

FMochiArticulatedPoseGeometry FMochiArticulatedPoseGeometry::FromBotPrefab(
    const superdex::robotics::BotPrefab& Prefab,
    double InScale) {
  FMochiArticulatedPoseGeometry Out;
  Out.Scale = InScale;
  const int NumLinks = mochi::isize(Prefab.links);
  if (mochi::isize(Prefab.joints) != NumLinks) {
    return Out;
  }
  Out.JointOffsets.SetNum(NumLinks);
  Out.LinkOffsets.SetNum(NumLinks);
  for (int i = 0; i < NumLinks; ++i) {
    const mochi::TransformRT& JointToLink = Prefab.links[i].parentJointFromLink;
    Out.JointOffsets[i] = Prefab.joints[i].parentLinkFromJoint.GetTranslation();
    Out.LinkOffsets[i] = JointToLink.GetRotation().GetConjugate() * JointToLink.GetTranslation();
  }
  return Out;
}

UController_MochiArticulatedPose::UController_MochiArticulatedPose() {}

USuperDexPoseableComponent* UController_MochiArticulatedPose::FindChildSourcePoseable(
    const USuperDexBotComponent* Component,
    const FString& NameContains) {
  const TArray<USceneComponent*> AttachedChildren = Component->GetAttachChildren();
  for (USceneComponent* Child : AttachedChildren) {
    USuperDexPoseableComponent* ChildSource = Cast<USuperDexPoseableComponent>(Child);
    if (IsValid(ChildSource)) {
      if (ChildSource->GetName().Contains(NameContains)) {
        return ChildSource;
      }
    }
  }
  return nullptr;
}

USuperDexPoseableComponent* UController_MochiArticulatedPose::FindChildSourcePoseable(
    const USuperDexBotComponent* Component) const {
  return FindChildSourcePoseable(Component, ChildSourceNameContainsSubstring);
}

bool UController_MochiArticulatedPose::Initialize(const USuperDexBotComponent* Component) {
  SourcePoseable = FindChildSourcePoseable(Component);
  if (SourcePoseable == nullptr) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[UController_MochiArticulatedPose] No child USuperDexPoseableComponent found matching substring '%s'"),
        *ChildSourceNameContainsSubstring);
    return false;
  }
  return true;
}

EMochiPoseTargetResult UController_MochiArticulatedPose::BuildPoseTarget(
    const FSuperDexPose& Pose,
    const FTransform& VrRootToWorld,
    const FMochiArticulatedPoseGeometry& Geometry,
    superdex::robotics::ControllerMochiArticulatedPose::Target& OutTarget) {
  const int NumBones = Pose.BoneRotations.Num();
  if (NumBones == 0) {
    return EMochiPoseTargetResult::EmptyPose;
  }

  FUnrealToMochiConverter u2m;

  // The pose may or may not carry per-bone translations. Sources like the OpenXR pose source
  // fill only BoneRotations (and SkeletonRootToVrRoot), leaving BoneLocations empty. The
  // link translations then come from the bot's own geometry at the scale it was spawned at, so the
  // targets match the bot whatever size the tracked hand is. Substituting zero instead would drop
  // the offsets and leave the first frames before the PD controller settles with grossly wrong link
  // targets.
  const bool bHasLocations = Pose.BoneLocations.Num() == NumBones;
  if (!bHasLocations && Geometry.JointOffsets.Num() != NumBones) {
    return EMochiPoseTargetResult::BoneCountMismatch;
  }

  // Bone locations supplied by the source are in its own skeleton units and presented at the pose's
  // (uniform) scale, so the effective offset is scale * boneLocation.
  const FVector PoseScale3D = Pose.SkeletonRootToVrRoot.GetScale3D();
  double PoseScale = PoseScale3D.X;
  if (!FMath::IsFinite(PoseScale) || FMath::IsNearlyZero(PoseScale)) {
    PoseScale = 1.0;
  }

  // Build per-link local-to-parent transforms from the FSuperDexPose bone data
  mochi::DynamicArray<mochi::TransformRT> LocalToParentTransforms(NumBones);
  const mochi::real Scale = static_cast<mochi::real>(Geometry.Scale);
  for (int i = 0; i < NumBones; ++i) {
    if (bHasLocations) {
      const FTransform BoneTransform(Pose.BoneRotations[i], Pose.BoneLocations[i] * PoseScale);
      LocalToParentTransforms[i] = u2m.TransformToMochi_RT(BoneTransform);
    } else {
      const mochi::Quaternion Rotation =
          u2m.TransformToMochi_RT(FTransform(Pose.BoneRotations[i])).GetRotation();
      LocalToParentTransforms[i] = mochi::TransformRT(
          Rotation, (Geometry.JointOffsets[i] + Rotation * Geometry.LinkOffsets[i]) * Scale);
    }
  }

  // Compose world-space root from the skeleton-root-to-vr-root and the source actor's world
  // transform. Strip the actor's own scale (matching
  // USuperDexPoseableComponent::OnTickUpdateComponentActorRelativeTransformFromCache) so the bot is
  // driven from the same VrRoot the mesh uses. The pose's own (hand) scale does not reach the root:
  // the bot's size is fixed at spawn (USuperDexBotComponent::BotScale).
  FTransform UnscaledVrRootToWorld = VrRootToWorld;
  UnscaledVrRootToWorld.SetScale3D(FVector::OneVector);
  const FTransform WorldFromRoot = Pose.SkeletonRootToVrRoot * UnscaledVrRootToWorld;

  OutTarget.localToParentTransforms = MoveTemp(LocalToParentTransforms);
  OutTarget.worldFromRoot = u2m.TransformToMochi_RT(WorldFromRoot);
  return EMochiPoseTargetResult::Built;
}

void UController_MochiArticulatedPose::CacheBotGeometry(const USuperDexBotComponent* Component) {
  const USuperDexBot* BotAsset = Component->GetBotAsset();
  if (BotAsset == nullptr) {
    BotGeometry = FMochiArticulatedPoseGeometry{};
    BotGeometry.Scale = Component->GetBotScale();
    return;
  }
  BotGeometry = FMochiArticulatedPoseGeometry::FromBotPrefab(
      BotAsset->GetBotPrefab(), Component->GetBotScale());
}

void UController_MochiArticulatedPose::OnBotSpawned(USuperDexBotComponent* Component) {
  // Initialize() resolves SourcePoseable; if it failed there is nothing to drive, so do not create
  // the underlying Mochi controller (otherwise StepController would run with isInitialized == true
  // and a missing source).
  if (SourcePoseable == nullptr) {
    return;
  }

  auto* ArticulatedActor = Component->GetMochiArticulatedActor();
  if (ArticulatedActor == nullptr) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[UController_MochiArticulatedPose] OnBotSpawned: MochiArticulatedActor is null"));
    return;
  }

  RobotActorHandle = ArticulatedActor->GetArticulatedActorHandle();
  CacheBotGeometry(Component);

  UMochiSceneObject* mochiScene = ArticulatedActor->GetMochiScene();
  if (mochiScene == nullptr) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[UController_MochiArticulatedPose] OnBotSpawned: MochiScene is null"));
    return;
  }

  mochi::Error err;
  mochiScene->QueueOrExecuteCommand([this, &err](mochi::Scene* scene) {
    mochi::Actor* robot = scene->GetActor(RobotActorHandle);
    auto* botsCtx = ISuperDexRoboticsModule::Get()->GetRoboticsContext();
    PoseControllerHandle = botsCtx->CreateController(
        "MOCHI_ARTICULATED_POSE", /*prefab*/ nullptr, robot, /*name*/ "", err);
    if (!err.IsOK()) {
      return;
    }
    PoseController = static_cast<superdex::robotics::ControllerMochiArticulatedPose*>(
        botsCtx->GetController(PoseControllerHandle));
    if (PoseController) {
      MochiPoseParams = ConvertToMochiPoseParams(PoseControllerDetailsParams);
      PoseController->SetParams(MochiPoseParams, err);
      if (!err.IsOK()) {
        return;
      }
      PoseController->Initialize(bRemoveExistingController, err);
    }
  });
  mochiScene->WaitForQueuedCommands();
  if (!err.IsOK()) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[UController_MochiArticulatedPose] Error during OnBotSpawned: %hs"),
        err.GetDescription());
  }
  isInitialized = err.IsOK() && PoseController;
}

void UController_MochiArticulatedPose::Shutdown(const USuperDexBotComponent* Component) {
  isInitialized = false;

  if (PoseController) {
    // The articulated actor (and its scene) may already be torn down by the time Shutdown runs; the
    // PoseController guard above does not protect against that. If the scene is gone the controller
    // went with it, so just drop our pointer. Otherwise queue the explicit destroy.
    auto* ArticulatedActor = Component->GetMochiArticulatedActor();
    UMochiSceneObject* mochiScene =
        (ArticulatedActor != nullptr) ? ArticulatedActor->GetMochiScene() : nullptr;
    if (mochiScene != nullptr) {
      mochiScene->QueueOrExecuteCommand([this](mochi::Scene* scene) {
        auto* botsCtx = ISuperDexRoboticsModule::Get()->GetRoboticsContext();
        botsCtx->DestroyController(PoseControllerHandle);
      });
      mochiScene->WaitForQueuedCommands();
    }
    PoseController = nullptr;
  }

  SourcePoseable = nullptr;
}

void UController_MochiArticulatedPose::TickController(
    const USuperDexBotComponent* Component,
    double GameDeltaTime) {
  if (!isInitialized || SourcePoseable == nullptr) {
    return;
  }

  // TODO: The constraints below exist only because an IK retargeter
  // source runs its IK solve synchronously, inline on whatever thread pulls its pose, dispatching
  // to the shared Mochi (Marl) scheduler. That single fact produced three separate
  // deadlocks while wiring this controller up:
  //   1. CacheLock lock-ordering: pulling the source from StepController (Mochi sim thread) via
  //      GetCachedPose took the retargeter's CacheLock (read) while the game-thread solve held it
  //      (write) and waited on the Mochi scheduler -> circular wait.
  //   2. Marl-worker starvation (this site): calling GetPose here (which solves) while the game
  //      thread holds BotControllerMutex blocks the sim-thread Marl worker on that OS mutex, the
  //      solve's tasks never drain, and the sim hangs permanently after a few frames.
  // Once IK retargeters do their solves on their own dedicated thread (or otherwise stop blocking
  // the shared scheduler from arbitrary caller threads), these constraints can be relaxed - e.g. it
  // would then be safe to call GetPose() here for a same-frame, lag-free target.
  //
  // Until then: read the source's most recent cached pose (GetCachedPose) - deliberately NOT
  // GetPose. This function runs under the bot component's BotControllerMutex
  // (USuperDexBotComponent::TickComponent), which the Mochi simulation thread also acquires in
  // OnArticulatedActorControl -> StepController, so we must not block here. GetCachedPose only
  // takes the source's CacheLock briefly and issues no Mochi work. The source's own TickComponent
  // performs the IK solve on the game thread OUTSIDE this mutex, so the cache read here is valid
  // (it may be up to one frame stale depending on component tick order; the PD pose controller
  // absorbs that).
  FSuperDexPose Pose;
  if (!SourcePoseable->GetCachedPose(Pose)) {
    return;
  }

  superdex::robotics::ControllerMochiArticulatedPose::Target Target;
  const EMochiPoseTargetResult Result =
      BuildPoseTarget(Pose, SourcePoseable->GetOwner()->GetActorTransform(), BotGeometry, Target);
  if (Result == EMochiPoseTargetResult::BoneCountMismatch && !bWarnedBoneCountMismatch) {
    bWarnedBoneCountMismatch = true;
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[UController_MochiArticulatedPose] Pose has %d bones but the bot has %d links; not driving the bot."),
        Pose.BoneRotations.Num(),
        BotGeometry.JointOffsets.Num());
  }
  if (Result != EMochiPoseTargetResult::Built) {
    return;
  }

  // Resolve params from the current details-panel values, also on the game thread.
  superdex::robotics::ControllerMochiArticulatedPose::Params params =
      ConvertToMochiPoseParams(PoseControllerDetailsParams);

  // Stash the sampled Target for the Mochi-thread StepController to consume. Target owns its pose
  // data, so moving into it here is self-contained. TickController and StepController are mutually
  // exclusive via the bot component's BotControllerMutex; we also guard explicitly.
  FScopeLock lock(&TargetMutex);
  CachedTarget = MoveTemp(Target);
  CachedParams = MoveTemp(params);
  bHasCachedTarget = true;
}

bool UController_MochiArticulatedPose::StepController(
    const USuperDexBotComponent* Component,
    const FMochiArticulationState& State,
    double MochiDeltaTime,
    mochi::Actor* ArticulatedActor,
    mochi::Scene* Scene) {
  if (!isInitialized || PoseController == nullptr) {
    return false;
  }

  // Consume the data sampled on the game thread in TickController(). We must NOT pull from the
  // source poseable here: it may be an IK retargeter that runs a blocking Mochi IK solve under
  // its CacheLock on the game thread, and acquiring that same lock from this Mochi-thread step
  // callback deadlocks the simulation. See TickController() and TargetMutex.
  //
  // Copy the Target out under the lock so TickController cannot overwrite it mid-step. The copy is
  // deep -- Target owns its pose data -- so it stays valid for the ComputeOutput call below.
  superdex::robotics::ControllerMochiArticulatedPose::Target target;
  superdex::robotics::ControllerMochiArticulatedPose::Params params;
  {
    FScopeLock lock(&TargetMutex);
    if (!bHasCachedTarget) {
      return false;
    }
    target = CachedTarget;
    params = CachedParams;
  }

  mochi::Error err;
  PoseController->SetParams(params, err);
  if (!err.IsOK()) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[UController_MochiArticulatedPose] SetParams failed: %hs"),
        err.GetDescription());
    return false;
  }

  superdex::robotics::ControllerMochiArticulatedPose::Obsv obsv;
  PoseController->ComputeOutput(obsv, target, err);
  if (!err.IsOK()) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("[UController_MochiArticulatedPose] ComputeOutput failed: %hs"),
        err.GetDescription());
    return false;
  }

  return true;
}

TArray<superdex::robotics::ControllerEntry>
UController_MochiArticulatedPose::ExportControllerEntries(const FString& OutputDir) const {
  TArray<superdex::robotics::ControllerEntry> Entries;
  mochi::Error Error;
  auto Params = ConvertToMochiPoseParams(PoseControllerDetailsParams);
  FString ParamsPath = FPaths::Combine(OutputDir, TEXT("pose.superdex_controller"));
  Params.SaveToFile(TCHAR_TO_UTF8(*ParamsPath), Error);
  if (Error.IsOK()) {
    superdex::robotics::ControllerEntry Entry;
    Entry.type = "mochi_articulated_pose";
    Entry.name = "pose";
    Entry.params = TCHAR_TO_ANSI(*ParamsPath);
    Entries.Add(MoveTemp(Entry));
  } else {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("ExportControllerEntries: Failed to save pose controller params: %hs"),
        Error.ToString().c_str());
  }
  return Entries;
}
