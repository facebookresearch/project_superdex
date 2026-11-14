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

#include "MochiSoftMeshComponent.h"

#include "MochiDynamicMeshUtil.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsModule.h"

#include "Engine/GameInstance.h"

#include <mochi_core/utils/profile.h>

void UMochiSoftMeshComponent::CreateMochiActor() {
  MOCHI_PROFILE_SCOPE();
  if (!GetOwner()->HasAuthority()) {
    return; // The server or solo player has authority. Multiplayer clients don't.
  }

  DestroyMochiActor();
  if (!Model && !MochiExistingActor.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] failed because not all params were set."),
        __FUNCTION__);
    return;
  }

  // If an existing mochi::ActorHandle is already given, just create a wrapper around that
  if (MochiExistingActor.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[%hs] Creating wrapper for %s, verify type is supported (i.e. %s)"),
        __FUNCTION__,
        *GET_MEMBER_NAME_CHECKED(UMochiSoftMeshComponent, MochiExistingActor).ToString(),
        *GET_ENUMERATOR_NAME_CHECKED(EMochiActorType, Soft).ToString());
    MochiActorObject = UMochiActorObject::CreateFromExistingActor(
        this,
        MochiExistingActor,
        true, /* surface positions */
        false, /* volume positions */
        bUseMochiNodeNormals,
        bEnableContacts,
        nullptr, /* scene */
        true /* use contact skin if present */);
  } else {
    MochiActorObject = UMochiActorObject::CreateSoft(
        this,
        GetName(),
        GetComponentTransform(),
        Model,
        MaterialParams,
        ContactParams,
        ColliderType,
        Layer,
        bHasGravity,
        BoundaryElementType,
        bUseContactSkin,
        bEnableRom ? &RomParams : nullptr,
        bUseRecentering,
        true,
        bEnableVolumeNodePositionsQuery, /* volume positions */
        bUseMochiNodeNormals,
        bEnableContacts,
        bEnableElasticEnergyQuery);
  }

  // generate reference node positions
  if (bEnableVolumeNodePositionsQuery) {
    ConvertVolumeNodeReferencePositions(
        Model, GetComponentTransform().GetScale3D(), VolumeNodeReferencePositions);
  }

  if (MaterialModifiers) {
    ApplyMaterialModifiers(MaterialModifiers);
  }

  // The UDynamicMesh was created at registration, but we'll update with runtime settings here
  EditMesh(
      [this](FDynamicMesh3& Mesh) {
        PrepareDynamicMeshForRuntime(GetComponentTransform().GetScale3D(), Mesh);
      },
      EDynamicMeshComponentRenderUpdateMode::FastUpdate);
}

const TArray<FVector>& UMochiSoftMeshComponent::GetVolumeNodeReferencePositions() const {
  if (!bEnableVolumeNodePositionsQuery) {
    UE_LOG(
        LogTemp,
        Warning,
        TEXT("[%hs] Called but %s is not enabled!"),
        __FUNCTION__,
        *GET_MEMBER_NAME_CHECKED(UMochiSoftMeshComponent, bEnableVolumeNodePositionsQuery)
             .ToString());
  }
  return VolumeNodeReferencePositions;
}

const TArray<FVector>& UMochiSoftMeshComponent::GetVolumeNodePositions() const {
  if (!bEnableVolumeNodePositionsQuery) {
    UE_LOG(
        LogTemp,
        Warning,
        TEXT("[%hs] Called but %s is not enabled!"),
        __FUNCTION__,
        *GET_MEMBER_NAME_CHECKED(UMochiSoftMeshComponent, bEnableVolumeNodePositionsQuery)
             .ToString());
  }
  return VolumeNodePositions;
}

const void UMochiSoftMeshComponent::ApplyMaterialModifiers(
    UMochiModelMaterialModifiers* Modifiers) const {
  if (Modifiers && MochiActorObject) {
    MochiActorObject->ApplyMaterialModifiers(Modifiers->MaterialModifiers);
  }
}

UMochiSoftMeshComponent::UMochiSoftMeshComponent(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {}

static bool PopulateFromMochiMesh_SoftInternal(
    FDynamicMesh3& Mesh,
    UMochiModel* Model,
    bool bUseMochiNodeNormals,
    TConstArrayView<FVector2f> RenderModelUVs,
    TConstArrayView<FLinearColor> RenderModelColors,
    mochi::ActorHandle const& MochiExistingActor,
    UWorld* World) {
  MOCHI_PROFILE_SCOPE();
  Mesh.Clear();

  if (Model) {
    PopulateMeshFromMochiModel(
        &Mesh,
        Model,
        FTransform::Identity,
        true /* use visual mesh if present */,
        true /* use contact skin if present */);
  } else if (MochiExistingActor.IsValid()) {
    // If this dynamic mesh was created at run time with existing actor, get the mesh from there.
    const auto game = World->GetGameInstance();
    const auto subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
    if (subsys) {
      PopulateMeshFromMochiActor(
          &Mesh,
          MochiExistingActor,
          subsys->GetScene(),
          true /* use visual mesh if present */,
          true /* use contact skin if present */);
    }
  }

  ComputeOrDiscardVertexNormals(Mesh, bUseMochiNodeNormals);
  // Vertex IDs are visual-mesh node indices. A count mismatch means the model changed after import,
  // or the mesh came from the contact skin or surface instead. Flat shading is not supported: it
  // draws face normals, but the material's tangents are built from the normal overlay.
  const auto fits = [&Mesh](int32 Num) { return Num == 0 || Num == Mesh.MaxVertexID(); };
  if (bUseMochiNodeNormals && !(RenderModelUVs.IsEmpty() && RenderModelColors.IsEmpty())) {
    if (fits(RenderModelUVs.Num()) && fits(RenderModelColors.Num())) {
      EnableRenderModelAttributes(Mesh, RenderModelUVs, RenderModelColors);
      return true;
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "Render model UVs or colors aren't drawn on Mochi model %s: they don't match the %d "
              "vertices of its drawn mesh. Reimport the prefab."),
          *GetNameSafe(Model),
          Mesh.MaxVertexID());
    }
  }
  return false;
}

void UMochiSoftMeshComponent::SetMochiModel(UMochiModel* NewMochiModel) {
  MOCHI_PROFILE_SCOPE();
  Model = NewMochiModel;
  EditMesh([this](UE::Geometry::FDynamicMesh3& Mesh) {
    SetHasRenderModelAttributes(PopulateFromMochiMesh_SoftInternal(
        Mesh,
        Model,
        bUseMochiNodeNormals,
        RenderModelUVs,
        RenderModelColors,
        MochiExistingActor,
        GetWorld()));
  });
  ValidateMaterialSlots();
}

void UMochiSoftMeshComponent::OnRegister() {
  MOCHI_PROFILE_SCOPE();
  SetEnableFlatShading(!bUseMochiNodeNormals);
  if (bUseMochiNodeNormals) {
    if (!RenderModelUVs.IsEmpty()) {
      // Normal maps need tangents that follow the deformation.
      SetTangentsType(EDynamicMeshComponentTangentsMode::AutoCalculated);
    }
    if (!RenderModelColors.IsEmpty()) {
      // Converts the linear colors as a static mesh does.
      SetVertexColorSpaceTransformMode(EDynamicMeshVertexColorTransformMode::LinearToSRGB);
    }
  }
  // Populate the underlying UDynamicMesh from the Model data
  EditMesh([this](UE::Geometry::FDynamicMesh3& Mesh) {
    SetHasRenderModelAttributes(PopulateFromMochiMesh_SoftInternal(
        Mesh,
        Model,
        bUseMochiNodeNormals,
        RenderModelUVs,
        RenderModelColors,
        MochiExistingActor,
        GetWorld()));
  });
  ValidateMaterialSlots();
  Super::OnRegister();
}

void UMochiSoftMeshComponent::BeginPlay() {
  MOCHI_PROFILE_SCOPE();
  WarnIfRenderModelAttributesNotDrawn();
  Super::BeginPlay();
}

void UMochiSoftMeshComponent::WarnIfRenderModelAttributesNotDrawn() const {
  if (!bUseMochiNodeNormals && (!RenderModelUVs.IsEmpty() || !RenderModelColors.IsEmpty())) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "%s has render model UVs or colors, which aren't drawn because Use Mochi Node Normals "
            "is off: flat shading can't give the render model's material matching tangents."),
        *GetPathName());
  }
}

void UMochiSoftMeshComponent::SetExistingMochiActorHandle(mochi::ActorHandle Handle) {
  MochiExistingActor = Handle;
}

void UMochiSoftMeshComponent::OnQueryDataApplied(const FMochiActorQueryData& QueryData) {
  // update elastic energy
  if (bEnableElasticEnergyQuery) {
    ElasticEnergy = QueryData.ElasticEnergy;
  }
  // update volume node positions
  if (bEnableVolumeNodePositionsQuery) {
    ConvertNodePositionsToUnreal(
        QueryData.VolumeNodePositions, VolumeNodePositions, GetComponentTransform().GetScale3D());
  }
}

#if WITH_EDITOR
void UMochiSoftMeshComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) {
  MOCHI_PROFILE_SCOPE();
  // Check for changes in the Model property, and regenerate dynamic mesh if needed
  const FName PropName = PropertyChangedEvent.GetPropertyName();
  if (PropName == GET_MEMBER_NAME_CHECKED(UMochiSoftMeshComponent, bUseMochiNodeNormals)) {
    WarnIfRenderModelAttributesNotDrawn();
  }
  if (PropName == GET_MEMBER_NAME_CHECKED(UMochiSoftMeshComponent, Model)) {
    UE_LOG(LogMochiPhysics, Display, TEXT("[%hs] %s changed"), __FUNCTION__, *PropName.ToString());

    EditMesh([this](UE::Geometry::FDynamicMesh3& Mesh) {
      SetHasRenderModelAttributes(PopulateFromMochiMesh_SoftInternal(
          Mesh,
          Model,
          bUseMochiNodeNormals,
          RenderModelUVs,
          RenderModelColors,
          MochiExistingActor,
          GetWorld()));
    });
    ValidateMaterialSlots();
  }

  Super::PostEditChangeProperty(PropertyChangedEvent);
}
#endif
