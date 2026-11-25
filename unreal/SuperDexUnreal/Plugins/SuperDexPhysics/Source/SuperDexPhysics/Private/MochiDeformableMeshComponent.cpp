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

#include "MochiDeformableMeshComponent.h"

#include "MochiComponentTags.h"
#include "MochiDynamicMeshUtil.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsModule.h"

#include "Engine/GameInstance.h"
#include "Engine/World.h"

#include <mochi_core/utils/profile.h>

const TArray<FMochiContactPoint>& UMochiDeformableMeshComponent::GetContactPoints() const {
  return ContactEventHandler.ContactPoints;
}

EMochiRenderMeshSource UMochiDeformableMeshComponent::PopulateMeshFromExistingMochiActor(
    UE::Geometry::FDynamicMesh3& Mesh) const {
  MOCHI_PROFILE_SCOPE();
  const UWorld* World = GetWorld();
  const UGameInstance* Game = World ? World->GetGameInstance() : nullptr;
  const UMochiGameInstanceSubsystem* Subsys =
      Game ? Game->GetSubsystem<UMochiGameInstanceSubsystem>() : nullptr;
  UMochiSceneObject* Scene = Subsys ? Subsys->GetScene() : nullptr;
  EMochiRenderMeshSource Source = EMochiRenderMeshSource::Surface;
  PopulateMeshFromMochiActor(
      &Mesh, MochiExistingActor, Scene, true /* use contact skin if present */, true, &Source);
  return Source;
}

void UMochiDeformableMeshComponent::RebuildPreviewMesh() {
  EditMesh([this](UE::Geometry::FDynamicMesh3& Mesh) { PopulateMesh(Mesh); });
}

void UMochiDeformableMeshComponent::PopulateMesh(UE::Geometry::FDynamicMesh3& Mesh) {
  MOCHI_PROFILE_SCOPE();
  Mesh.Clear();
  bHasRenderModelAttributes = false;
  EMochiRenderMeshSource Source = EMochiRenderMeshSource::Surface;
  if (HasExistingMochiActor()) {
    Source = PopulateMeshFromExistingMochiActor(Mesh);
  } else if (Model && Model->IsValid() && IMochiPhysicsModule::Get()->GetMochiContext()) {
    const mochi::ShapeHandle Shape = Model->GetMochiShape(FVector::OneVector, FTransform::Identity);
    PopulateMeshFromMochiShape(&Mesh, Shape, true /* use contact skin if present */, true, &Source);
  }
  if (Mesh.VertexCount() > 0) {
    ComputeOrDiscardVertexNormals(Mesh, bUseMochiNodeNormals);
    bHasRenderModelAttributes = AddRenderModelAttributes(Mesh, Source);
  }
}

bool UMochiDeformableMeshComponent::AddRenderModelAttributes(
    UE::Geometry::FDynamicMesh3& Mesh,
    EMochiRenderMeshSource Source) const {
  // Render-model attributes are indexed by visual-mesh node, not contact-skin or surface node.
  // Flat shading is not supported: the material's tangents need the normal overlay.
  if (!bUseMochiNodeNormals || (RenderModelUVs.IsEmpty() && RenderModelColors.IsEmpty())) {
    return false;
  }
  const auto Fits = [&Mesh](int32 Num) { return Num == 0 || Num == Mesh.MaxVertexID(); };
  if (Source == EMochiRenderMeshSource::Visual && Fits(RenderModelUVs.Num()) &&
      Fits(RenderModelColors.Num())) {
    EnableRenderModelAttributes(Mesh, RenderModelUVs, RenderModelColors);
    return true;
  }
  UE_LOG(
      LogMochiPhysics,
      Warning,
      TEXT(
          "Render model UVs or colors aren't drawn on Mochi model %s: its drawn mesh is not the "
          "matching visual mesh (%d vertices). Reimport the prefab."),
      *GetNameSafe(Model),
      Mesh.MaxVertexID());
  return false;
}

void UMochiDeformableMeshComponent::DestroyMochiActor() {
  if (MochiActorObject) {
    MochiActorObject->DestroyMochiResources();
    MochiActorObject = nullptr;
  }
}

UMochiActorObject* UMochiDeformableMeshComponent::GetMochiActor() const {
  return MochiActorObject;
}

UMochiModel* UMochiDeformableMeshComponent::GetMochiModel() const {
  return Model;
}

bool UMochiDeformableMeshComponent::GetContactsEnabled() const {
  return bEnableContacts;
}

UMochiDeformableMeshComponent::UMochiDeformableMeshComponent(
    const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  PrimaryComponentTick.bCanEverTick = true;
  Mobility = EComponentMobility::Movable;
}

void UMochiDeformableMeshComponent::BeginPlay() {
  MOCHI_PROFILE_SCOPE();

  if (!MochiComponentTags::AreComponentsDisabled(GetOwner()) && MochiActorObject == nullptr) {
    CreateMochiActor();
  }
  Super::BeginPlay();
}

void UMochiDeformableMeshComponent::EndPlay(const EEndPlayReason::Type EndPlayReason) {
  MOCHI_PROFILE_SCOPE();
  Super::EndPlay(EndPlayReason);
  DestroyMochiActor();
}

void UMochiDeformableMeshComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  MOCHI_PROFILE_SCOPE();
  if (MochiActorObject == nullptr) {
    return;
  }
  MochiActorObject->TryGetQueryData([this](const FMochiActorQueryData& QueryData) {
    // set mesh transform
    SetWorldLocationAndRotation(
        QueryData.RootTransform.GetLocation(), QueryData.RootTransform.GetRotation());
    // update dynamic mesh
    EditMesh(
        [&](FDynamicMesh3& Mesh) {
          UpdateDynamicMeshFromMochiQuery(
              QueryData.SurfaceNodePositions,
              QueryData.NodeNormals,
              bUseMochiNodeNormals,
              bHasRenderModelAttributes,
              FVector::OneVector / GetComponentTransform().GetScale3D(),
              Mesh);
        },
        EDynamicMeshComponentRenderUpdateMode::FastUpdate);
    // update contact events
    if (bEnableContacts) {
      ContactEventHandler.Update(
          this,
          OnMochiComponentBeginContact,
          OnMochiComponentEndContact,
          OnMochiComponentContact,
          QueryData.ContactPoints,
          bFilterSelfContacts);
    }
    OnQueryDataApplied(QueryData);
  });
}

UMaterialInterface* UMochiDeformableMeshComponent::GetMaterial(int32 ElementIndex) const {
  MOCHI_PROFILE_SCOPE();
  // UDynamicMeshComponent's material handling is strange, so we'll go with the UMeshComponent
  // ancestor's version first.
  UMaterialInterface* OutMaterial = UMeshComponent::GetMaterial(ElementIndex);
  if (!OutMaterial) {
    OutMaterial = Super::GetMaterial(ElementIndex);
  }
  return OutMaterial;
}

void UMochiDeformableMeshComponent::SetMaterial(int32 ElementIndex, UMaterialInterface* Material) {
  MOCHI_PROFILE_SCOPE();
  // UDynamicMeshComponent misses a few important things, so we'll explicitly call the one in
  // UMeshComponent ancestor first.
  UMeshComponent::SetMaterial(ElementIndex, Material);
  Super::SetMaterial(ElementIndex, Material);
}
