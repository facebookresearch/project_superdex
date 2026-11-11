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
#include "Skeletons/SuperDexSkeleton_OpenXR.h"
#include "TeleopNode.h"

#include "OpenXR2RobotiqRetargeter.generated.h"

UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SUPERDEXTELEOP_API UOpenXR2RobotiqRetargeter : public USuperDexRetargeter {
  GENERATED_BODY()
 public:
  // The gripper's articulation. Its joint configuration poses this component's mesh.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mesh")
  TObjectPtr<UMochiArticulationBase> Articulation;

  // If true, RobotiqRootToSourceSkeletonRoot will be set to its value. If false, it will be
  // derived from the ChildSource of this SuperDexPoseableComponent on BeginPlay. If false and no
  // valid child component is present, it will be identity.
  UPROPERTY(EditAnywhere, BlueprintREADONLY, Category = "Teleop")
  bool bOverrideRobotiqRootToSourceSkeletonRoot = false;
  // Transform between the Robotiq root and your source hand root.
  // TransformedPose.SkeletonRootToVrRoot = RobotiqRootToSourceSkeletonRoot *
  // SourcePose.SkeletonRootToVrRoot. This multiplication ignores scale to preserve the actual
  // relative position and then sets scale to the combination of the two.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (EditCondition = "bOverrideRobotiqRootToSourceSkeletonRoot"),
      Category = "Teleop")
  FTransform RobotiqRootToSourceSkeletonRoot = FTransform::Identity;
  // The thumb tip socket (or bone) of the source hand mesh; its distance to the index tip drives
  // the gripper. The default is the OpenXR hand's (xr_05_thumb_tip).
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Teleop")
  FName ThumbTipName = kSuperDexBoneNames_OXR[static_cast<int32>(ESuperDexBone_OXR::ThumbTip)];
  // The index finger tip socket (or bone) of the source hand mesh. The default is the OpenXR
  // hand's (xr_10_index_tip).
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Teleop")
  FName IndexTipName = kSuperDexBoneNames_OXR[static_cast<int32>(ESuperDexBone_OXR::IndexTip)];
  // Default behavior is to directly map
  // finger distance to pincher distance; setting
  // this to true allows you to define custom pinch mappings below.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Teleop")
  bool bUseCustomPinchRange = false;
  // Pinch is "1.0" at or below this index-to-thumb-tip distance.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (EditCondition = "bUseCustomPinchRange"),
      Category = "Teleop")
  double MinPinchDistance = 1.0;
  // Pinch is "0.0" at or above this index-to-thumb-tip distance.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      meta = (EditCondition = "bUseCustomPinchRange"),
      Category = "Teleop")
  double MaxPinchDistance = 10.0;
  // If true, the TransformedPose will be based on received ros poses. If false, the TransformedPose
  // will be based on the solved IK poses.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Teleop")
  bool bReceiveRosPoses = false;
  // If true, will send the IK computed pose to the Ros interface. If false and bReceiveRosPoses is
  // true, the system will not compute or need any of the IK solvers.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Teleop")
  bool bIssueRosCommands = false;

  // Returns the current cached joint positionsof the robot as an array of joint positions [rad or
  // m]. These are updated when GetTransformedPose is called and returns a valid pose.
  UFUNCTION(BlueprintPure, Category = "Teleop")
  const TArray<double>& GetCachedJointPositions() const {
    return JointPositionsCache;
  }

 public:
  virtual void BeginPlay() override;
  virtual bool GetTransformedPose_Implementation(
      FSuperDexPose const& SourcePose,
      bool bSourceIsValid,
      FSuperDexPose& TransformedPose) override;
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif
  UTeleopNodeOperatorComponent* RosInterface = nullptr;

  /** A cache containing the last valid joint positions from GetTransformedPose. */
  TArray<double> JointPositionsCache;
};
