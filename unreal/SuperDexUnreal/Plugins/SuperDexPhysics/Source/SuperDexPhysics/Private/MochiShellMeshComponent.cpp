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

#include "MochiShellMeshComponent.h"

#include "MochiAssets.h"
#include "MochiDynamicMeshUtil.h"
#include "MochiPhysicsModule.h"

#include <mochi_core/utils/profile.h>

void UMochiShellMeshComponent::CreateMochiActor() {
  MOCHI_PROFILE_SCOPE();
  if (!GetOwner()->HasAuthority()) {
    return;
  }

  DestroyMochiActor();

  if (!Model) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[%hs] Failed: Model is not set."), __FUNCTION__);
    return;
  }

  const mochi::ShapeHandle Shape =
      Model->GetMochiShape(GetComponentTransform().GetScale3D(), FTransform::Identity);
  if (!Shape.IsValid()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[%hs] Failed: shape handle is invalid."), __FUNCTION__);
    return;
  }

  mochi::PointCloudColliderParams PointCloudParams;
  PointCloudParams.radius = mochi::real(PointCloudRadius);
  PointCloudParams.selfContact = bPointCloudSelfContact;
  PointCloudParams.selfContactExclusionRatio = mochi::real(PointCloudSelfContactExclusionRatio);
  if (PointCloudColliderTriangleElementType.IsSet()) {
    PointCloudParams.colliderTriangleElementType =
        ToMochi(PointCloudColliderTriangleElementType.GetValue());
  }

  MochiActorObject = UMochiActorObject::CreateShell(
      this,
      GetName(),
      GetComponentTransform(),
      Shape,
      ShellMaterialParams,
      ContactParams,
      ColliderType,
      Layer,
      bHasGravity,
      &PointCloudParams,
      bUseContactSkin,
      ContactElementType,
      true, /* surface positions */
      bUseMochiNodeNormals,
      bEnableContacts);

  // The UDynamicMesh was created at registration, but we'll update with runtime settings here.
  EditMesh(
      [this](FDynamicMesh3& Mesh) {
        PrepareDynamicMeshForRuntime(GetComponentTransform().GetScale3D(), Mesh);
      },
      EDynamicMeshComponentRenderUpdateMode::FastUpdate);
}

UMochiShellMeshComponent::UMochiShellMeshComponent(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  // Load a default two-sided material so shell renders correctly from both sides.
  static ConstructorHelpers::FObjectFinder<UMaterialInterface> DefaultShellMaterial(
      TEXT("/SuperDexPhysics/Materials/M_MochiShellDefault"));
  if (DefaultShellMaterial.Succeeded()) {
    SetMaterial(0, DefaultShellMaterial.Object);
  }
}

void UMochiShellMeshComponent::OnRegister() {
  MOCHI_PROFILE_SCOPE();
  ConfigureShading();
  RebuildPreviewMesh();
  ValidateMaterialSlots();
  Super::OnRegister();
}

#if WITH_EDITOR
void UMochiShellMeshComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) {
  MOCHI_PROFILE_SCOPE();
  const FName PropName = PropertyChangedEvent.GetPropertyName();
  if (PropName == GET_MEMBER_NAME_CHECKED(UMochiShellMeshComponent, Model)) {
    UE_LOG(LogMochiPhysics, Display, TEXT("[%hs] %s changed"), __FUNCTION__, *PropName.ToString());
    RebuildPreviewMesh();
    ValidateMaterialSlots();
  }
  Super::PostEditChangeProperty(PropertyChangedEvent);
}
#endif
