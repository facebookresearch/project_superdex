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

#include "Retargeters/OpenXR2RobotiqRetargeter.h"

#include "Engine/SkeletalMesh.h"
#include "SuperDexPoseFunctionLibrary.h"
#include "SuperDexPoseUtils.h"
#include "SuperDexTeleopModule.h"

// Skeleton name of the Robotiq 2F-85 output pose (its skeleton enum header was deleted as unused).
static const FName kSkeletonName_Robotiq2F_85 = "Robotiq2F_85";

void UOpenXR2RobotiqRetargeter::BeginPlay() {
  Super::BeginPlay();
  // the articulation must exist to function
  if (Articulation == nullptr) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[%hs::%d] No articulation assigned to actor [%s]."
            " This component will not function."),
        __FUNCTION__,
        __LINE__,
        *GetOwner()->GetName());
  }
  // If we are using ros at all make sure we have the component
  if (bIssueRosCommands || bReceiveRosPoses) {
    RosInterface = GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>();
    if (RosInterface == nullptr) {
      UE_LOG(
          LogSuperDexTeleop,
          Error,
          TEXT(
              "[%hs::%d] (bIssueRosCommands || bReceiveRosPoses) == true;"
              " but could not find the TeleopNode on this actor [%s]."
              " This component will not function."),
          __FUNCTION__,
          __LINE__,
          *GetOwner()->GetName());
    }
  }
  if (!bOverrideRobotiqRootToSourceSkeletonRoot) {
    RobotiqRootToSourceSkeletonRoot = FTransform::Identity;
    USuperDexPoseableComponent* ChildSource = GetChildSourcePoseable();
    if (ChildSource != nullptr) {
      FTransform transform = ChildSource->GetRelativeTransform();
      transform.SetScale3D(FVector(1));
      RobotiqRootToSourceSkeletonRoot = transform.Inverse();
    }
  }
}

bool UOpenXR2RobotiqRetargeter::GetTransformedPose_Implementation(
    FSuperDexPose const& SourcePose,
    bool /*bSourceIsValid*/,
    FSuperDexPose& TransformedPose) {
  if (Articulation == nullptr) {
    return false;
  }
  if ((bIssueRosCommands || bReceiveRosPoses) && (RosInterface == nullptr)) {
    return false;
  }
  USuperDexPoseableComponent* openXrHand = GetChildSourcePoseable();
  if (openXrHand == nullptr) {
    return false;
  }

  TransformedPose = SourcePose;
  TransformedPose.tags.Add(TEXT("OpenXR2Robotiq"));
  TransformedPose.SkeletonName = kSkeletonName_Robotiq2F_85;
  TransformedPose.SkeletonRootToVrRoot =
      USuperDexPoseFunctionLibrary::MultiplyTransformsWithoutScalePreservingScale(
          RobotiqRootToSourceSkeletonRoot, SourcePose.SkeletonRootToVrRoot);
  auto thumbTipLoc = openXrHand->GetSocketLocation(ThumbTipName);
  auto indexTipLoc = openXrHand->GetSocketLocation(IndexTipName);
  constexpr double fingerTipRadius = 0.6;
  double fingerTipDist = (thumbTipLoc - indexTipLoc).Length() - 2 * fingerTipRadius;
  double theta = 0;
  constexpr double kMaxTheta = 0.8;
  constexpr double kMaxGripperDist = 8.5;
  if (bUseCustomPinchRange) {
    double grip01 = FMath::GetMappedRangeValueClamped(
        FVector2D{MaxPinchDistance, MinPinchDistance}, FVector2D{0, 1}, fingerTipDist);
    theta = kMaxTheta * grip01;
  } else {
    theta = FMath::Acos(fingerTipDist / kMaxGripperDist) / 2;
  }
  theta = FMath::Clamp(theta, 0, kMaxTheta);
  TArray<double> jointAngles = {theta, theta, -theta, -theta, -theta, theta};
  if (bIssueRosCommands) {
    RosInterface->SetHandJointTargets(jointAngles);
  }
  bool bHaveAngles = false;
  if (bReceiveRosPoses) {
    // get joint angles
    bHaveAngles = RosInterface->GetHandJointStates(jointAngles);
  } else {
    if (!jointAngles.IsEmpty()) {
      bHaveAngles = true;
    }
  }
  if (bHaveAngles) {
    bool validPose = USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(
        Articulation, jointAngles, TransformedPose);
    JointPositionsCache = jointAngles;
    return validPose;
  }
  return false;
}

#if WITH_EDITOR
void UOpenXR2RobotiqRetargeter::PostEditChangeProperty(
    FPropertyChangedEvent& PropertyChangedEvent) {
  Super::PostEditChangeProperty(PropertyChangedEvent);
  if (PropertyChangedEvent.GetMemberPropertyName() ==
      GET_MEMBER_NAME_CHECKED(UOpenXR2RobotiqRetargeter, Articulation)) {
    SetSkinnedAssetAndUpdate(Articulation ? Articulation->GetSkeletalMesh() : nullptr);
  }
}
#endif
