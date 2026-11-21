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

#include "Retargeters/SuperDexEEGoalPublisher.h"

#include "Engine/SkeletalMesh.h"
#include "SuperDexPoseFunctionLibrary.h"
#include "SuperDexPoseUtils.h"
#include "SuperDexTeleopModule.h"
#include "TeleopNode.h"

USuperDexEEGoalPublisher::USuperDexEEGoalPublisher() {
  // Run GetTransformedPose even when the source pose is invalid, so the goal is invalidated on
  // tracking loss and the mesh keeps following the robot.
  bGetTransformedPoseDoesNotNeedValidSourcePose = true;
  // This class poses its own editor preview from the articulation.
  bShow_EditorViewportPose = false;
}

FTransform USuperDexEEGoalPublisher::ComputeEndEffectorGoal(
    const FTransform& InEndEffectorToSourceRoot,
    const FTransform& SourceRootToVrRoot,
    const FTransform& InArmBaseToActor) {
  const FTransform EndEffectorToActor = InEndEffectorToSourceRoot * SourceRootToVrRoot;
  return EndEffectorToActor * InArmBaseToActor.Inverse();
}

void USuperDexEEGoalPublisher::OnRegister() {
  Super::OnRegister();
  PoseMeshAtDefaultJointPositionsInEditor();
}

void USuperDexEEGoalPublisher::PoseMeshAtDefaultJointPositionsInEditor() {
  // Only in the editor: in a game world the mesh must stay in its reference pose until BeginPlay,
  // which measures EndEffectorToSourceRoot from it.
  UWorld* World = GetWorld();
  if (World == nullptr || World->IsGameWorld() || Articulation == nullptr ||
      GetSkinnedAsset() == nullptr) {
    return;
  }
  FSuperDexPose Pose;
  if (!USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(
          Articulation, Articulation->GetDefaultJointPositions(), Pose)) {
    return;
  }
  {
    FWriteScopeLock Lock(CacheLock);
    CachedOutputPose = Pose;
    CacheInitialized = true;
  }
  ApplyCachedPoseToMesh();
  RefreshBoneTransforms();
}

void USuperDexEEGoalPublisher::BeginPlay() {
  Super::BeginPlay();
  bInitialized = false;
  const FString OwnerName = GetNameSafe(GetOwner());

  if (Articulation == nullptr) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[%hs] No Articulation assigned to [%s] on actor [%s]. This component will not function."),
        __FUNCTION__,
        *GetName(),
        *OwnerName);
    return;
  }

  TeleopNode =
      GetOwner() ? GetOwner()->GetComponentByClass<UTeleopNodeOperatorComponent>() : nullptr;
  if (TeleopNode == nullptr) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[%hs] No UTeleopNodeOperatorComponent found on actor [%s] for [%s]."
            " This component will not function."),
        __FUNCTION__,
        *OwnerName,
        *GetName());
    return;
  }

  const int32 NumDofs = Articulation->GetNumDofs();
  if (TeleopNode->NumArmJoints != NumDofs) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[%hs] TeleopNode NumArmJoints (%d) does not match the DOFs (%d) of articulation [%s]"
            " on [%s] of actor [%s]. This component will not function."),
        __FUNCTION__,
        TeleopNode->NumArmJoints,
        NumDofs,
        *Articulation->GetName(),
        *GetName(),
        *OwnerName);
    return;
  }

  if (!bOverrideEndEffectorToSourceRoot) {
    const int32 EndEffectorBoneIndex = GetBoneIndex(EndEffectorBoneName);
    if (EndEffectorBoneIndex == INDEX_NONE) {
      UE_LOG(
          LogSuperDexTeleop,
          Error,
          TEXT(
              "[%hs] EndEffectorBoneName [%s] is not a bone of the mesh on [%s] of actor [%s]."
              " This component will not function."),
          __FUNCTION__,
          *EndEffectorBoneName.ToString(),
          *GetName(),
          *OwnerName);
      return;
    }
    USuperDexPoseableComponent* ChildSource = GetChildSourcePoseable();
    if (ChildSource == nullptr) {
      UE_LOG(
          LogSuperDexTeleop,
          Error,
          TEXT(
              "[%hs] [%s] on actor [%s] has no child source poseable to derive"
              " EndEffectorToSourceRoot from. Attach the source poseable below this component or"
              " set bOverrideEndEffectorToSourceRoot. This component will not function."),
          __FUNCTION__,
          *GetName(),
          *OwnerName);
      return;
    }
    FTransform EndEffectorToWorld = GetBoneTransform(EndEffectorBoneIndex);
    EndEffectorToWorld.SetScale3D(FVector(1));
    FTransform SourceRootToWorld = ChildSource->GetComponentTransform();
    SourceRootToWorld.SetScale3D(FVector(1));
    EndEffectorToSourceRoot = USuperDexPoseFunctionLibrary::MultiplyTransformsStrippingScale(
        EndEffectorToWorld, SourceRootToWorld.Inverse());
  }

  ArmBaseToActor = USuperDexPoseUtilsFunctionLibrary::GetRelativeTransform_CompToRoot(this);
  bInitialized = true;
}

bool USuperDexEEGoalPublisher::GetTransformedPose_Implementation(
    FSuperDexPose const& SourcePose,
    bool bSourceIsValid,
    FSuperDexPose& TransformedPose) {
  if (!bInitialized || TeleopNode == nullptr || Articulation == nullptr) {
    return false;
  }

  // 1. Goal out.
  if (bSourceIsValid) {
    TeleopNode->SetArmPoseTarget(ComputeEndEffectorGoal(
        EndEffectorToSourceRoot, SourcePose.SkeletonRootToVrRoot, ArmBaseToActor));
  } else {
    TeleopNode->InvalidateArmPoseTarget();
  }

  // 2. Robot state in.
  TArray<double> JointPositions;
  if (!TeleopNode->GetArmJointStates(JointPositions)) {
    return false;
  }
  if (JointPositions.Num() != Articulation->GetNumDofs()) {
    if (!bLoggedJointStateSizeMismatch) {
      bLoggedJointStateSizeMismatch = true;
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT(
              "[%hs] %s/%s: received %d arm joint states, articulation [%s] has %d DOFs;"
              " skipping."),
          __FUNCTION__,
          *GetNameSafe(GetOwner()),
          *GetName(),
          JointPositions.Num(),
          *Articulation->GetName(),
          Articulation->GetNumDofs());
    }
    return false;
  }
  bLoggedJointStateSizeMismatch = false;

  TransformedPose = SourcePose;
  TransformedPose.tags.Add(TEXT("EEGoalPublisher"));
  TransformedPose.SkeletonRootToVrRoot = ArmBaseToActor;
  // Puts the pose on the arm's articulation (its Bot) at the robot's joint positions.
  if (!USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(
          Articulation, JointPositions, TransformedPose)) {
    return false;
  }
  JointPositionsCache = MoveTemp(JointPositions);
  return true;
}

#if WITH_EDITOR
void USuperDexEEGoalPublisher::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) {
  Super::PostEditChangeProperty(PropertyChangedEvent);
  if (PropertyChangedEvent.GetMemberPropertyName() ==
      GET_MEMBER_NAME_CHECKED(USuperDexEEGoalPublisher, Articulation)) {
    SetSkinnedAssetAndUpdate(Articulation ? Articulation->GetSkeletalMesh() : nullptr);
    PoseMeshAtDefaultJointPositionsInEditor();
  }
}
#endif
