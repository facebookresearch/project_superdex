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

#include "MochiArticulationBase.h"
#include "Retargeters/SuperDexRetargeter.h"

#include "SuperDexEEGoalPublisher.generated.h"

class UTeleopNodeOperatorComponent;

/**
 * Publishes an arm end-effector goal derived from its source pose, and poses its own mesh from the
 * robot's reported arm joint states. No IK is solved.
 *
 * Each frame:
 *  1. Goal out: if the source pose is valid, sends
 *       EndEffectorToSourceRoot * SourcePose.SkeletonRootToVrRoot * ArmBaseToActor^-1
 *     to the TeleopNode via SetArmPoseTarget(); otherwise calls InvalidateArmPoseTarget().
 *  2. Robot state in: reads every arm joint from the TeleopNode (GetArmJointStates()) and runs
 *     forward kinematics on Articulation to produce the output pose. With no robot state yet, no
 *     pose is produced and the last cached pose is kept.
 *
 * The source pose is usually the hand poseable attached below this component. It is used for its
 * validity and its SkeletonRootToVrRoot only.
 *
 * The arm's joints are the articulation's DOFs that move the chain from its root to
 * EndEffectorBoneName (see ComputeArmDofIndices), and their count must match the TeleopNode's
 * NumArmJoints. So Articulation can be the arm alone or the whole robot, e.g. an arm + hand bot:
 * its other DOFs (the hand's) stay at their default positions, and the mesh bones they move are
 * usually hidden (bHideBonesBelowSourceAttachment).
 */
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API USuperDexEEGoalPublisher : public USuperDexRetargeter {
  GENERATED_BODY()

 public:
  // The arm's articulation, or a bot of the whole robot with the arm at its root (e.g. an arm +
  // hand bot). Provides the arm's joints (ComputeArmDofIndices) and the forward kinematics that
  // pose this component's mesh. The mesh's first bones must be the articulation's links in order
  // (see IsMeshCompatibleWithArticulation); further bones keep their reference pose relative to
  // their parent. In the editor, assigning an articulation whose links the current mesh does not
  // start with replaces the mesh with the articulation's own, and the mesh is posed at the
  // articulation's default joint positions.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EEGoalPublisher")
  TObjectPtr<UMochiArticulationBase> Articulation = nullptr;

  // The bone (link) of this component's mesh whose pose the published goal targets, e.g. the
  // arm's flange. The arm's joints are the DOFs that move the chain from the root to it. Used to
  // derive EndEffectorToSourceRoot on BeginPlay.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EEGoalPublisher")
  FName EndEffectorBoneName = NAME_None;

  // If true, every bone below the bone (or socket) the source poseable is attached to is hidden
  // from rendering; that bone itself stays visible. With a combined arm + hand mesh and the hand
  // retargeter attached at the hand's mount bone, this draws the arm and its mount but not the
  // hand, which the source poseable draws. Nothing is hidden when the source is not attached to
  // a socket or its bone has no children. Rendering only: bone and socket transforms are
  // unaffected.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EEGoalPublisher|Rendering")
  bool bHideBonesBelowSourceAttachment = true;

  // If true, EndEffectorToSourceRoot is used as set. If false, it is derived on BeginPlay as the
  // EndEffectorBoneName bone's transform relative to the child source poseable component (both
  // with scale stripped), i.e. the fixed mount offset between the end effector and the hand.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EEGoalPublisher")
  bool bOverrideEndEffectorToSourceRoot = false;

  // Transform from the end effector to the source skeleton root. See
  // bOverrideEndEffectorToSourceRoot.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (EditCondition = "bOverrideEndEffectorToSourceRoot"),
      Category = "EEGoalPublisher")
  FTransform EndEffectorToSourceRoot = FTransform::Identity;

  // Debug (runtime): draw each published end-effector goal as a coordinate frame.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EEGoalPublisher|Debug")
  bool bDebugDrawEndEffectorGoal = false;

  // The last robot arm joint positions [rad or m] that produced a valid pose, in arm joint order
  // (GetArmDofIndices).
  UFUNCTION(BlueprintPure, Category = "EEGoalPublisher")
  const TArray<double>& GetCachedJointPositions() const {
    return JointPositionsCache;
  }

  // The articulation's DOF index of each arm joint, in the TeleopNode's arm joint order, found by
  // InitializeEEGoalPublisher (ComputeArmDofIndices). Empty until then.
  const TArray<int32>& GetArmDofIndices() const {
    return ArmDofIndices;
  }

  // Whether InitializeEEGoalPublisher found everything this component needs. When false it
  // publishes nothing and produces no pose.
  UFUNCTION(BlueprintPure, Category = "EEGoalPublisher")
  bool IsInitialized() const {
    return bInitialized;
  }

  /**
   * Find the TeleopNode, validate the articulation, mesh and joint count, and latch
   * EndEffectorToSourceRoot and the arm base frame. Returns whether this component is ready to
   * publish. Called from BeginPlay; call it again after configuring a component that has already
   * begun play (e.g. on a spawned roboperator).
   */
  UFUNCTION(BlueprintCallable, Category = "EEGoalPublisher")
  bool InitializeEEGoalPublisher();

  /**
   * The goal sent to SetArmPoseTarget(): the end effector expressed in the arm base frame.
   *
   * @param InEndEffectorToSourceRoot Fixed end effector -> source skeleton root offset.
   * @param SourceRootToVrRoot The source pose's SkeletonRootToVrRoot.
   * @param InArmBaseToActor This component (the arm base) relative to the actor root.
   */
  static FTransform ComputeEndEffectorGoal(
      const FTransform& InEndEffectorToSourceRoot,
      const FTransform& SourceRootToVrRoot,
      const FTransform& InArmBaseToActor);

  /**
   * Whether FK on InArticulation can pose Mesh: the mesh's first bones must be the articulation's
   * links, by name and in order. The mesh may have more bones after them.
   *
   * @param OutReason If not null, set to why the mesh is incompatible.
   */
  static bool IsMeshCompatibleWithArticulation(
      const USkinnedAsset* Mesh,
      const UMochiArticulationBase* InArticulation,
      FString* OutReason = nullptr);

  /**
   * The arm's joints: the DOFs of InArticulation that move a link on the chain from its root link
   * to InEndEffectorBoneName (that link included), in DOF order. With InEndEffectorBoneName unset,
   * or an articulation that cannot report the link each DOF moves (anything but a USuperDexBot),
   * every DOF is an arm joint.
   *
   * @param Mesh The mesh FK poses, compatible with InArticulation
   * (IsMeshCompatibleWithArticulation); its skeleton gives the chain.
   * @param OutReason If not null, set to why the arm joints cannot be found.
   * @return false if InEndEffectorBoneName is not one of InArticulation's links or no DOF moves its
   *     chain.
   */
  static bool ComputeArmDofIndices(
      const UMochiArticulationBase* InArticulation,
      const USkinnedAsset* Mesh,
      FName InEndEffectorBoneName,
      TArray<int32>& OutArmDofIndices,
      FString* OutReason = nullptr);

  virtual bool GetTransformedPose_Implementation(
      FSuperDexPose const& SourcePose,
      bool bSourceIsValid,
      FSuperDexPose& TransformedPose) override;

 protected:
  USuperDexEEGoalPublisher();
  virtual void OnRegister() override;
  virtual void BeginPlay() override;
  // A child (re)attaching or detaching can change the source's attach socket.
  virtual void OnChildAttached(USceneComponent* ChildComponent) override;
  virtual void OnChildDetached(USceneComponent* ChildComponent) override;
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

  // Outside a game world, pose the mesh at the articulation's default joint positions.
  void PoseMeshAtDefaultJointPositionsInEditor();

  // Hide the bones below the source poseable's attach bone (see bHideBonesBelowSourceAttachment),
  // unhiding any others. Must run again whenever the mesh is (re)assigned, the component
  // re-registers (both reset bone visibility), or the source's attachment changes.
  void ApplyHiddenBones();

  UPROPERTY(Transient)
  TObjectPtr<UTeleopNodeOperatorComponent> TeleopNode = nullptr;

  // This component relative to the actor root, latched on BeginPlay. It is the output pose's
  // SkeletonRootToVrRoot and the base frame of the published goal.
  FTransform ArmBaseToActor = FTransform::Identity;

  TArray<double> JointPositionsCache;
  // See GetArmDofIndices.
  TArray<int32> ArmDofIndices;
  // A joint configuration of Articulation: its default positions, with the arm joints overwritten
  // by each received robot state.
  TArray<double> ArticulationJointPositions;
  bool bInitialized = false;
  // Warn once about a robot state of the wrong size; cleared on a valid read.
  bool bLoggedJointStateSizeMismatch = false;
};
