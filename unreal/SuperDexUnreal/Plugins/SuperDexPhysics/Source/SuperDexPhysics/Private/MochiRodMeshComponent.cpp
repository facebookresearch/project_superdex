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

#include "MochiRodMeshComponent.h"

#include "MochiAssets.h"
#include "MochiDynamicMeshUtil.h"
#include "MochiPhysicsModule.h"
#include "PhysicsEngine/BodySetup.h"

#include <mochi_core/utils/profile.h>

void UMochiRodMeshComponent::RebuildPreviewMesh() {
  MOCHI_PROFILE_SCOPE();
  if (IsTemplate()) {
    return;
  }
  EditMesh([this](UE::Geometry::FDynamicMesh3& Mesh) {
    Mesh.Clear();
    PopulateMeshFromMochiModel(
        &Mesh,
        Model,
        FTransform::Identity,
        false /* use visual mesh if present */,
        true /* use contact skin if present */);
    ComputeOrDiscardVertexNormals(Mesh, bUseMochiNodeNormals);
  });
}

void UMochiRodMeshComponent::CreateMochiActor() {
  MOCHI_PROFILE_SCOPE();
  DestroyMochiActor();

  if (!GetOwner()->HasAuthority()) {
    return;
  }

  if (!Model) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[%hs] Failed: Model is not set."), __FUNCTION__);
    return;
  }

  const mochi::ShapeHandle shape =
      Model->GetMochiShape(GetComponentTransform().GetScale3D(), FTransform::Identity);
  if (!shape.IsValid()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[%hs] Failed: shape handle is invalid."), __FUNCTION__);
    return;
  }

  mochi::PointCloudColliderParams pointCloudParams;
  pointCloudParams.radius = mochi::real(PointCloudRadius);
  pointCloudParams.selfContact = bPointCloudSelfContact;
  pointCloudParams.selfContactExclusionRatio = mochi::real(PointCloudSelfContactExclusionRatio);
  if (PointCloudColliderSegmentElementType.IsSet()) {
    pointCloudParams.colliderSegmentElementType =
        ToMochi(PointCloudColliderSegmentElementType.GetValue());
  }

  MochiActorObject = UMochiActorObject::CreateRod(
      this,
      GetName(),
      GetComponentTransform(),
      shape,
      RodMaterialParams,
      ContactParams,
      ColliderType,
      Layer,
      bHasGravity,
      &pointCloudParams,
      bUseContactSkin,
      ContactElementType,
      ContactSkinElementType,
      true, /* surface positions */
      bUseMochiNodeNormals,
      bEnableContacts);

  // The UDynamicMesh was created at registration, but we'll update with runtime settings here.
  EditMesh(
      [this](UE::Geometry::FDynamicMesh3& Mesh) {
        PrepareDynamicMeshForRuntime(GetComponentTransform().GetScale3D(), Mesh);
      },
      EDynamicMeshComponentRenderUpdateMode::FastUpdate);
}

void UMochiRodMeshComponent::EnsureBodySetupPublic() {
  // If the engine ever lazily creates a UBodySetup on this component while it is
  // a CDO (Blueprint reinstancing, property propagation, Replace-With-Blueprint),
  // mark it RF_Public so it serializes legally and does not trip
  // "Illegal reference to private object" (e.g. Cable_GEN_VARIABLE) on save.
  if (UBodySetup* BS = GetBodySetup()) {
    BS->SetFlags(RF_Public);
  }
}

void UMochiRodMeshComponent::OnRegister() {
  MOCHI_PROFILE_SCOPE();
  Super::OnRegister();
  SetEnableFlatShading(!bUseMochiNodeNormals);
  RebuildPreviewMesh();
  EnsureBodySetupPublic();
  ValidateMaterialSlots();
}

#if WITH_EDITOR
void UMochiRodMeshComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) {
  MOCHI_PROFILE_SCOPE();
  Super::PostEditChangeProperty(PropertyChangedEvent);
  const FName PropName = PropertyChangedEvent.GetPropertyName();
  if (PropName == GET_MEMBER_NAME_CHECKED(UMochiRodMeshComponent, Model)) {
    UE_LOG(LogMochiPhysics, Display, TEXT("[%hs] %s changed"), __FUNCTION__, *PropName.ToString());
    RebuildPreviewMesh();
    ValidateMaterialSlots();
  }
  EnsureBodySetupPublic();
}
#endif
