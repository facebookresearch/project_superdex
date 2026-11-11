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

#pragma once

#include "SuperDexBotController.h"

#include "HAL/CriticalSection.h"

#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>
#include <superdex_robotics/core/context.h>

#include <atomic>

#include "Controller_MochiArticulatedPose.generated.h"

class USuperDexPoseableComponent;
struct FSuperDexPose;

// A bot's link geometry, as a pose target is built against it (see
// UController_MochiArticulatedPose::BuildPoseTarget). Per link, in nominal bot units: the joint's
// offset in its parent link (parentLinkFromJoint), and the link origin's offset from the joint
// expressed in the link frame. A link's translation in its parent is then
// Scale * (JointOffset + R * LinkOffset) for its local rotation R. Used when an incoming
// FSuperDexPose omits BoneLocations.
struct SUPERDEXTELEOP_API FMochiArticulatedPoseGeometry {
  TArray<mochi::Real3> JointOffsets;
  TArray<mochi::Real3> LinkOffsets;
  double Scale = 1.0;

  // Empty offsets when the prefab's link and joint counts disagree.
  static FMochiArticulatedPoseGeometry FromBotPrefab(
      const superdex::robotics::BotPrefab& Prefab,
      double InScale);
};

enum class EMochiPoseTargetResult : uint8 {
  Built,
  // The pose carries no bones.
  EmptyPose,
  // The pose omits bone locations and its bone count differs from the bot's link count.
  BoneCountMismatch,
};

// Details struct mirroring mochi::PoseTrackingParams
USTRUCT(BlueprintType)
struct FMochiPoseTrackingParams {
  GENERATED_BODY()

  UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (Tooltip = "Spring stiffness [N/m or Nm/rad]"))
  double Stiffness = 0;

  UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (Tooltip = "Damping coefficient"))
  double Damping = 0;

  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (Tooltip = "Force/torque saturation cap (-1 = no limit)"))
  double Saturation = -1;
};

// Details struct mirroring mochi::PoseControllerParams
USTRUCT(BlueprintType)
struct FMochiPoseControllerParams {
  GENERATED_BODY()

  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (Tooltip = "Per-link position tracking (size 1 broadcasts to all links)"))
  TArray<FMochiPoseTrackingParams> LinkPosTracking;

  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (Tooltip = "Per-link rotation tracking (size 1 broadcasts to all links)"))
  TArray<FMochiPoseTrackingParams> LinkRotTracking;

  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (Tooltip = "Per-joint tracking (size 1 broadcasts to all joints)"))
  TArray<FMochiPoseTrackingParams> JointTracking;
};

UCLASS(Blueprintable)
class SUPERDEXTELEOP_API UController_MochiArticulatedPose : public USuperDexBotController {
  GENERATED_BODY()
 public:
  // Substring used to disambiguate among multiple child USuperDexPoseableComponents.
  // Empty string matches any child. Mirrors the pattern from GetChildSourcePoseable.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Source")
  FString ChildSourceNameContainsSubstring = "";

  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pose Controller")
  FMochiPoseControllerParams PoseControllerDetailsParams;

  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Pose Controller",
      meta = (Tooltip = "Remove existing ArticulatedPoseController on actor if present"))
  bool bRemoveExistingController = true;

 public:
  UController_MochiArticulatedPose();

  static superdex::robotics::ControllerMochiArticulatedPose::Params ConvertToMochiPoseParams(
      const FMochiPoseControllerParams& InParams);

  // Build the target that poses a bot like Pose, rooted at Pose.SkeletonRootToVrRoot under
  // VrRootToWorld (whose scale is ignored). Call on the game thread: see TickController.
  static EMochiPoseTargetResult BuildPoseTarget(
      const FSuperDexPose& Pose,
      const FTransform& VrRootToWorld,
      const FMochiArticulatedPoseGeometry& Geometry,
      superdex::robotics::ControllerMochiArticulatedPose::Target& OutTarget);

  // The first child poseable of Component whose name contains NameContains (empty matches any).
  static USuperDexPoseableComponent* FindChildSourcePoseable(
      const USuperDexBotComponent* Component,
      const FString& NameContains);

 protected:
  virtual bool Initialize(const USuperDexBotComponent* Component) override;
  virtual void OnBotSpawned(USuperDexBotComponent* Component) override;
  virtual void Shutdown(const USuperDexBotComponent* Component) override;
  virtual bool StepController(
      const USuperDexBotComponent* Component,
      const FMochiArticulationState& State,
      double MochiDeltaTime,
      mochi::Actor* ArticulatedActor,
      mochi::Scene* Scene) override;
  // Sample the source poseable and build the controller target on the game thread. This must NOT be
  // done in StepController (Mochi simulation thread): the source may be an IK retargeter whose
  // pose cache is produced by a blocking Mochi IK solve (USuperDexRetargeter::GetPose holds
  // CacheLock while USuperDexIKSolverObject::SolveIK waits on the shared Mochi scheduler). Reading
  // that source from the Mochi step callback via GetCachedPose (which also takes CacheLock)
  // deadlocks: the game-thread solve holds the write lock and waits on the scheduler, while the
  // Mochi step waits on the read lock. We break the cycle by only ever touching the source from the
  // game thread here.
  virtual void TickController(const USuperDexBotComponent* Component, double GameDeltaTime)
      override;
  virtual TArray<superdex::robotics::ControllerEntry> ExportControllerEntries(
      const FString& OutputDir) const override;

 public:
  // The child poseable this controller tracks (see ChildSourceNameContainsSubstring).
  USuperDexPoseableComponent* FindChildSourcePoseable(const USuperDexBotComponent* Component) const;

 protected:
  // Caches the spawned bot's link geometry and scale. Pose sources like the OpenXR one fill only
  // FSuperDexPose::BoneRotations; the link translations of the target then come from the bot
  // itself, so the target always matches the bot's geometry at its scale. See BuildPoseTarget.
  void CacheBotGeometry(const USuperDexBotComponent* Component);

 protected:
  // Written on the game thread (OnBotSpawned/Shutdown) and read on the Mochi simulation thread
  // (StepController), so it is atomic to avoid a data race on the lifecycle gate.
  std::atomic<bool> isInitialized = false;
  mochi::ActorHandle RobotActorHandle{};

  superdex::robotics::ControllerHandle PoseControllerHandle{};
  superdex::robotics::ControllerMochiArticulatedPose* PoseController = nullptr;
  superdex::robotics::ControllerMochiArticulatedPose::Params MochiPoseParams;

  USuperDexPoseableComponent* SourcePoseable = nullptr;

  FMochiArticulatedPoseGeometry BotGeometry;
  bool bWarnedBoneCountMismatch = false;

  // The controller target is sampled on the game thread in TickController() and consumed on the
  // Mochi simulation thread in StepController(), guarded by TargetMutex (so the Mochi thread never
  // touches the source poseable and its CacheLock). bHasCachedTarget gates the first valid sample.
  //
  // Target owns its pose data (its fields are DynamicArrays, not Spans), so the sampled Target is
  // stashed directly and StepController copies it out under the mutex. See TickController() for
  // why sampling must stay on the game thread.
  mutable FCriticalSection TargetMutex;
  superdex::robotics::ControllerMochiArticulatedPose::Target CachedTarget;
  superdex::robotics::ControllerMochiArticulatedPose::Params CachedParams;
  bool bHasCachedTarget = false;
};
