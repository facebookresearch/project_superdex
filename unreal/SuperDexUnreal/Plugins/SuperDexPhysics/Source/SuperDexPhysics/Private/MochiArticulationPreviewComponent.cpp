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

#include "MochiArticulationPreviewComponent.h"

#include "MochiTypes.h"

UMochiArticulationPreviewComponent::UMochiArticulationPreviewComponent() {
  if (Articulation && Articulation->GetSkeletalMesh()) {
    SetSkinnedAssetAndUpdate(Articulation->GetSkeletalMesh());
  }
}

void UMochiArticulationPreviewComponent::SetArticulation(UMochiArticulationBase* NewArticulation) {
  if (NewArticulation) {
    Articulation = NewArticulation;
    if (Articulation->GetSkeletalMesh()) {
      SetSkinnedAssetAndUpdate(Articulation->GetSkeletalMesh());
    }
    SetJointPositions(Articulation->GetDefaultJointPositions());
  } else {
    Articulation = nullptr;
    SetSkinnedAssetAndUpdate(nullptr);
    JointPositions.Reset();
  }
}

UMochiArticulationBase* UMochiArticulationPreviewComponent::GetArticulation() const {
  return Articulation;
}

void UMochiArticulationPreviewComponent::SetJointPositions(
    const TArray<double>& NewJointPositions) {
  if (Articulation) {
    ensure(NewJointPositions.Num() == Articulation->GetDefaultJointPositions().Num());
    JointPositions = NewJointPositions;
    TArray<FTransform> transforms;
    auto names = Articulation->GetLinkNames();
    if (Articulation->ComputeLinkTransformsFromJointPositions(
            JointPositions, transforms, EMochiArticulationLinkTransformSpace::LinkToParent)) {
      for (int iLink = 0; iLink < transforms.Num(); ++iLink) {
        SetBoneLocalTransformByName(names[iLink], transforms[iLink]);
      }
    }
    TickPose(0, true);
    RefreshBoneTransforms();
    RecreateRenderState_Concurrent();
  }
}

void UMochiArticulationPreviewComponent::SetBoneLocalTransformByName(
    FName BoneName,
    const FTransform& InTransform) {
  if (!GetSkinnedAsset() || !RequiredBones.IsValid()) {
    return;
  }
  const int32 boneIndex = GetBoneIndex(BoneName);
  if (boneIndex >= 0 && boneIndex < BoneSpaceTransforms.Num()) {
    BoneSpaceTransforms[boneIndex] = InTransform;
    MarkRefreshTransformDirty();
  }
}

void UMochiArticulationPreviewComponent::SetBoneLocalRotations(const TArray<FQuat>& InRotations) {
  if (!GetSkinnedAsset() || !RequiredBones.IsValid()) {
    return;
  }
  if (InRotations.Num() == GetNumBones()) {
    for (int boneIndex = 0; boneIndex < GetNumBones(); ++boneIndex) {
      BoneSpaceTransforms[boneIndex].SetRotation(InRotations[boneIndex]);
    }
  }
  RefreshBoneTransforms();
}

#if WITH_EDITOR
void UMochiArticulationPreviewComponent::PostEditChangeProperty(
    FPropertyChangedEvent& PropertyChangedEvent) {
  Super::PostEditChangeProperty(PropertyChangedEvent);
  const FName PropertyName = (PropertyChangedEvent.Property != nullptr)
      ? PropertyChangedEvent.Property->GetFName()
      : NAME_None;
  if (PropertyName == GET_MEMBER_NAME_CHECKED(UMochiArticulationPreviewComponent, Articulation)) {
    SetArticulation(Articulation);
  }
  if (PropertyName == GET_MEMBER_NAME_CHECKED(UMochiArticulationPreviewComponent, JointPositions)) {
    SetJointPositions(JointPositions);
  }
}
#endif

void UMochiArticulationPreviewComponent::OnRegister() {
  Super::OnRegister();
  if (Articulation) {
    JointPositions =
        JointPositions.Num() == 0 ? Articulation->GetDefaultJointPositions() : JointPositions;
    SetJointPositions(JointPositions);
  }
}
