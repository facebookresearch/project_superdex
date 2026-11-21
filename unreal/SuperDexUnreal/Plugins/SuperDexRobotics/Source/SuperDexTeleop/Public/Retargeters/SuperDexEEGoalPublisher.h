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
 * validity and its SkeletonRootToVrRoot only. The arm's joint count is the articulation's DOF count
 * and must match the TeleopNode's NumArmJoints.
 */
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API USuperDexEEGoalPublisher : public USuperDexRetargeter {
  GENERATED_BODY()

 public:
  // The arm's articulation. Provides the DOF count and the forward kinematics that pose this
  // component's mesh. In the editor, assigning it also assigns its skeletal mesh and poses the
  // mesh at the articulation's default joint positions.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EEGoalPublisher")
  TObjectPtr<UMochiArticulationBase> Articulation = nullptr;

  // The bone (link) of this component's mesh whose pose the published goal targets, e.g. the
  // arm's flange. Used to derive EndEffectorToSourceRoot on BeginPlay.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EEGoalPublisher")
  FName EndEffectorBoneName = NAME_None;

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

  // The last robot arm joint positions [rad or m] that produced a valid pose.
  UFUNCTION(BlueprintPure, Category = "EEGoalPublisher")
  const TArray<double>& GetCachedJointPositions() const {
    return JointPositionsCache;
  }

  // Whether BeginPlay found everything this component needs. When false it publishes nothing and
  // produces no pose.
  UFUNCTION(BlueprintPure, Category = "EEGoalPublisher")
  bool IsInitialized() const {
    return bInitialized;
  }

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

  virtual bool GetTransformedPose_Implementation(
      FSuperDexPose const& SourcePose,
      bool bSourceIsValid,
      FSuperDexPose& TransformedPose) override;

 protected:
  USuperDexEEGoalPublisher();
  virtual void OnRegister() override;
  virtual void BeginPlay() override;
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

  // Outside a game world, pose the mesh at the articulation's default joint positions.
  void PoseMeshAtDefaultJointPositionsInEditor();

  UPROPERTY(Transient)
  TObjectPtr<UTeleopNodeOperatorComponent> TeleopNode = nullptr;

  // This component relative to the actor root, latched on BeginPlay. It is the output pose's
  // SkeletonRootToVrRoot and the base frame of the published goal.
  FTransform ArmBaseToActor = FTransform::Identity;

  TArray<double> JointPositionsCache;
  bool bInitialized = false;
  // Warn once about a robot state of the wrong size; cleared on a valid read.
  bool bLoggedJointStateSizeMismatch = false;
};
