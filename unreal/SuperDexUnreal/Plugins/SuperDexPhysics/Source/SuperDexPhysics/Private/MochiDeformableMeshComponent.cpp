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

#include <mochi_core/utils/profile.h>

const TArray<FMochiContactPoint>& UMochiDeformableMeshComponent::GetContactPoints() const {
  return ContactEventHandler.ContactPoints;
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
