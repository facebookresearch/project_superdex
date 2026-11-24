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

#include "DrawDebugHelpers.h"
#include "Engine/SkeletalMesh.h"
#include "SuperDexBot.h"
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
  // Registration reallocates the bone visibility states, so re-apply the hidden bones.
  ApplyHiddenBones();
}

bool USuperDexEEGoalPublisher::IsMeshCompatibleWithArticulation(
    const USkinnedAsset* Mesh,
    const UMochiArticulationBase* InArticulation,
    FString* OutReason) {
  auto Fail = [OutReason](FString Reason) {
    if (OutReason != nullptr) {
      *OutReason = MoveTemp(Reason);
    }
    return false;
  };
  if (Mesh == nullptr || InArticulation == nullptr) {
    return Fail(TEXT("the mesh or the articulation is not set"));
  }
  const FReferenceSkeleton& RefSkeleton = Mesh->GetRefSkeleton();
  const TArray<FName> LinkNames = InArticulation->GetLinkNames();
  if (RefSkeleton.GetNum() < LinkNames.Num()) {
    return Fail(
        FString::Printf(
            TEXT("mesh [%s] has %d bones but articulation [%s] has %d links"),
            *Mesh->GetName(),
            RefSkeleton.GetNum(),
            *InArticulation->GetName(),
            LinkNames.Num()));
  }
  for (int32 Index = 0; Index < LinkNames.Num(); ++Index) {
    if (RefSkeleton.GetBoneName(Index) != LinkNames[Index]) {
      return Fail(
          FString::Printf(
              TEXT("bone %d of mesh [%s] is [%s] but link %d of articulation [%s] is [%s]"),
              Index,
              *Mesh->GetName(),
              *RefSkeleton.GetBoneName(Index).ToString(),
              Index,
              *InArticulation->GetName(),
              *LinkNames[Index].ToString()));
    }
  }
  return true;
}

bool USuperDexEEGoalPublisher::ComputeArmDofIndices(
    const UMochiArticulationBase* InArticulation,
    const USkinnedAsset* Mesh,
    FName InEndEffectorBoneName,
    TArray<int32>& OutArmDofIndices,
    FString* OutReason) {
  auto Fail = [OutReason](FString Reason) {
    if (OutReason != nullptr) {
      *OutReason = MoveTemp(Reason);
    }
    return false;
  };
  OutArmDofIndices.Reset();
  if (InArticulation == nullptr) {
    return Fail(TEXT("the articulation is not set"));
  }
  const int32 NumDofs = InArticulation->GetNumDofs();
  const auto* Bot = Cast<USuperDexBot>(InArticulation);
  if (Bot == nullptr || InEndEffectorBoneName.IsNone()) {
    for (int32 Dof = 0; Dof < NumDofs; ++Dof) {
      OutArmDofIndices.Add(Dof);
    }
    return true;
  }
  FString MeshReason;
  if (!IsMeshCompatibleWithArticulation(Mesh, InArticulation, &MeshReason)) {
    return Fail(MeshReason);
  }
  // The mesh's first bones are the articulation's links, so its skeleton gives the links' parents.
  const FReferenceSkeleton& RefSkeleton = Mesh->GetRefSkeleton();
  const int32 EndEffectorIndex = RefSkeleton.FindBoneIndex(InEndEffectorBoneName);
  if (EndEffectorIndex == INDEX_NONE || EndEffectorIndex >= InArticulation->GetLinkNames().Num()) {
    return Fail(
        FString::Printf(
            TEXT("end effector [%s] is not a link of articulation [%s]"),
            *InEndEffectorBoneName.ToString(),
            *InArticulation->GetName()));
  }
  TSet<FName> Chain;
  for (int32 Bone = EndEffectorIndex; Bone != INDEX_NONE; Bone = RefSkeleton.GetParentIndex(Bone)) {
    Chain.Add(RefSkeleton.GetBoneName(Bone));
  }
  for (int32 Dof = 0; Dof < NumDofs; ++Dof) {
    if (Chain.Contains(Bot->GetDofLinkName(Dof))) {
      OutArmDofIndices.Add(Dof);
    }
  }
  if (OutArmDofIndices.IsEmpty()) {
    return Fail(
        FString::Printf(
            TEXT("no DOF of articulation [%s] moves the chain to end effector [%s]"),
            *InArticulation->GetName(),
            *InEndEffectorBoneName.ToString()));
  }
  return true;
}

void USuperDexEEGoalPublisher::ApplyHiddenBones() {
  const USkinnedAsset* Mesh = GetSkinnedAsset();
  if (Mesh == nullptr) {
    return;
  }
  const FReferenceSkeleton& RefSkeleton = Mesh->GetRefSkeleton();
  const int32 NumBones = RefSkeleton.GetNum();
  for (int32 BoneIndex = 0; BoneIndex < NumBones; ++BoneIndex) {
    if (IsBoneHidden(BoneIndex)) {
      UnHideBone(BoneIndex);
    }
  }
  // GetChildSourcePoseable() only returns direct children, so the source is attached to us and
  // its socket names one of our bones or sockets.
  const USuperDexPoseableComponent* Source =
      bHideBonesBelowSourceAttachment ? GetChildSourcePoseable() : nullptr;
  const FName AttachSocket = Source ? Source->GetAttachSocketName() : NAME_None;
  if (!AttachSocket.IsNone()) {
    const FName AttachBone = GetSocketBoneName(AttachSocket);
    const int32 KeptBoneIndex =
        AttachBone.IsNone() ? INDEX_NONE : RefSkeleton.FindBoneIndex(AttachBone);
    if (KeptBoneIndex == INDEX_NONE) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT(
              "[%hs] Source [%s] is attached to [%s], which is not a bone or socket of mesh [%s]"
              " on [%s] of actor [%s]; no bones are hidden."),
          __FUNCTION__,
          *Source->GetName(),
          *AttachSocket.ToString(),
          *Mesh->GetName(),
          *GetName(),
          *GetNameSafe(GetOwner()));
    } else {
      // Bones are ordered parents first, so the children follow their parent. Hiding a child
      // hides its whole subtree.
      for (int32 BoneIndex = KeptBoneIndex + 1; BoneIndex < NumBones; ++BoneIndex) {
        if (RefSkeleton.GetParentIndex(BoneIndex) == KeptBoneIndex) {
          HideBone(BoneIndex, PBO_None);
        }
      }
    }
  }
  MarkRenderDynamicDataDirty();
}

void USuperDexEEGoalPublisher::OnChildAttached(USceneComponent* ChildComponent) {
  Super::OnChildAttached(ChildComponent);
  ApplyHiddenBones();
}

void USuperDexEEGoalPublisher::OnChildDetached(USceneComponent* ChildComponent) {
  Super::OnChildDetached(ChildComponent);
  ApplyHiddenBones();
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
    bCacheInitialized = true;
  }
  ApplyCachedPoseToMesh();
  RefreshBoneTransforms();
}

void USuperDexEEGoalPublisher::BeginPlay() {
  Super::BeginPlay();
  InitializeEEGoalPublisher();
}

bool USuperDexEEGoalPublisher::InitializeEEGoalPublisher() {
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
    return false;
  }

  FString MeshReason;
  if (!IsMeshCompatibleWithArticulation(GetSkinnedAsset(), Articulation, &MeshReason)) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[%hs] The mesh of [%s] on actor [%s] cannot be posed by its articulation: %s."
            " This component will not function."),
        __FUNCTION__,
        *GetName(),
        *OwnerName,
        *MeshReason);
    return false;
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
    return false;
  }

  FString ArmReason;
  if (!ComputeArmDofIndices(
          Articulation, GetSkinnedAsset(), EndEffectorBoneName, ArmDofIndices, &ArmReason)) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[%hs] Cannot find the arm joints of [%s] on actor [%s]: %s."
            " This component will not function."),
        __FUNCTION__,
        *GetName(),
        *OwnerName,
        *ArmReason);
    return false;
  }
  if (TeleopNode->NumArmJoints != ArmDofIndices.Num()) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[%hs] TeleopNode NumArmJoints (%d) does not match the %d arm DOFs (of %d) of"
            " articulation [%s] up to end effector [%s] on [%s] of actor [%s]. This component will"
            " not function."),
        __FUNCTION__,
        TeleopNode->NumArmJoints,
        ArmDofIndices.Num(),
        Articulation->GetNumDofs(),
        *Articulation->GetName(),
        *EndEffectorBoneName.ToString(),
        *GetName(),
        *OwnerName);
    return false;
  }
  ArticulationJointPositions = Articulation->GetDefaultJointPositions();
  if (ArticulationJointPositions.Num() != Articulation->GetNumDofs()) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT(
            "[%hs] Articulation [%s] on [%s] of actor [%s] has %d default joint positions for %d"
            " DOFs. This component will not function."),
        __FUNCTION__,
        *Articulation->GetName(),
        *GetName(),
        *OwnerName,
        ArticulationJointPositions.Num(),
        Articulation->GetNumDofs());
    return false;
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
      return false;
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
      return false;
    }
    FTransform EndEffectorToWorld = GetBoneTransform(EndEffectorBoneIndex);
    EndEffectorToWorld.SetScale3D(FVector(1));
    FTransform SourceRootToWorld = ChildSource->GetComponentTransform();
    SourceRootToWorld.SetScale3D(FVector(1));
    EndEffectorToSourceRoot = USuperDexPoseFunctionLibrary::MultiplyTransformsStrippingScale(
        EndEffectorToWorld, SourceRootToWorld.Inverse());
  }

  ArmBaseToActor = USuperDexPoseUtilsFunctionLibrary::GetRelativeTransform_CompToRoot(this);
  // Children are attached by now; re-derive the hidden bones from the source's socket.
  ApplyHiddenBones();
  bInitialized = true;
  return true;
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
    const FTransform Goal = ComputeEndEffectorGoal(
        EndEffectorToSourceRoot, SourcePose.SkeletonRootToVrRoot, ArmBaseToActor);
    TeleopNode->SetArmPoseTarget(Goal);
    const USceneComponent* ActorRoot = GetOwner()->GetRootComponent();
    if (bDebugDrawEndEffectorGoal && IsInGameThread() && ActorRoot != nullptr) {
      // The goal is the end effector in the arm base frame, which is ArmBaseToActor in the actor
      // root's frame.
      FTransform ActorToWorld = ActorRoot->GetComponentTransform();
      ActorToWorld.SetScale3D(FVector::OneVector);
      const FTransform GoalToWorld = Goal * ArmBaseToActor * ActorToWorld;
      DrawDebugCoordinateSystem(
          GetWorld(),
          GoalToWorld.GetLocation(),
          GoalToWorld.Rotator(),
          10.f,
          false,
          -1.f,
          SDPG_Foreground,
          0.6f);
    }
  } else {
    TeleopNode->InvalidateArmPoseTarget();
  }

  // 2. Robot state in.
  TArray<double> JointPositions;
  if (!TeleopNode->GetArmJointStates(JointPositions)) {
    return false;
  }
  if (JointPositions.Num() != ArmDofIndices.Num()) {
    if (!bLoggedJointStateSizeMismatch) {
      bLoggedJointStateSizeMismatch = true;
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT(
              "[%hs] %s/%s: received %d arm joint states, articulation [%s] has %d arm DOFs;"
              " skipping."),
          __FUNCTION__,
          *GetNameSafe(GetOwner()),
          *GetName(),
          JointPositions.Num(),
          *Articulation->GetName(),
          ArmDofIndices.Num());
    }
    return false;
  }
  bLoggedJointStateSizeMismatch = false;
  // The arm joints into the articulation's configuration; its other DOFs keep their defaults.
  for (int32 Joint = 0; Joint < ArmDofIndices.Num(); ++Joint) {
    ArticulationJointPositions[ArmDofIndices[Joint]] = JointPositions[Joint];
  }

  TransformedPose = SourcePose;
  TransformedPose.Tags.Add(TEXT("EEGoalPublisher"));
  TransformedPose.SkeletonRootToVrRoot = ArmBaseToActor;
  // Puts the pose on the articulation (its Bot) at the robot's arm joint positions.
  if (!USuperDexPoseFunctionLibrary::ComputeSuperDexPoseFromJointPositions(
          Articulation, ArticulationJointPositions, TransformedPose)) {
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
    // Keep a mesh that FK on the new articulation can still pose (e.g. a combined arm + hand
    // mesh); otherwise switch to the articulation's own.
    if (Articulation == nullptr) {
      SetSkinnedAssetAndUpdate(nullptr);
    } else if (!IsMeshCompatibleWithArticulation(GetSkinnedAsset(), Articulation)) {
      SetSkinnedAssetAndUpdate(Articulation->GetSkeletalMesh());
    }
  }
  PoseMeshAtDefaultJointPositionsInEditor();
  ApplyHiddenBones();
}
#endif
