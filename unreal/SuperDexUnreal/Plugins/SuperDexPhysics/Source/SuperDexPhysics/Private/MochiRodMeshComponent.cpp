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

#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/profile.h>
#include <mochi_physics/mochi_physics_experimental.h>

void UMochiRodMeshComponent::RebuildPreviewMesh() {
  MOCHI_PROFILE_SCOPE();
  if (IsTemplate() || !Model || !Model->IsValid()) {
    return;
  }
  const FVector Scale = GetComponentTransform().GetScale3D();
  bool modelHasContactSkin = false;
  mochi::ShapeHandle shape = CreateRodShape(Scale, modelHasContactSkin);
  bModelHasContactSkin = modelHasContactSkin;
  EditMesh(
      [this, shape, Scale](UE::Geometry::FDynamicMesh3& Mesh) {
        Mesh.Clear();
        if (shape.IsValid()) {
          PopulateMeshFromMochiShape(
              &Mesh,
              shape,
              false /* use visual mesh if present */,
              true /* use contact skin if present */);
          ComputeOrDiscardVertexNormals(Mesh, bUseMochiNodeNormals);
          PrepareDynamicMeshForRuntime(Scale, Mesh);
        }
      },
      EDynamicMeshComponentRenderUpdateMode::FastUpdate);
  // Cache the scale that was actually used for the bake. Updating it here
  // (rather than at each call site) keeps the cached scale paired with the
  // geometry and prevents drift if a new caller forgets to refresh it.
  LastBakedScale = Scale;
}

void UMochiRodMeshComponent::RebakeIfNeeded() {
  // Funnel for editor-only lifecycle hooks. RebuildPreviewMesh itself also
  // checks IsTemplate(); the additional HasBegunPlay() guard here keeps these
  // hooks from stomping the simulation-owned dynamic mesh during PIE.
  if (IsTemplate() || HasBegunPlay()) {
    return;
  }
  RebuildPreviewMesh();
}

mochi::ShapeHandle UMochiRodMeshComponent::CreateRodShape(
    const FVector& BakeScale,
    bool& bOutModelHasContactSkin) const {
  MOCHI_PROFILE_SCOPE();
  bOutModelHasContactSkin = false;
  if (!Model || !Model->IsValid()) {
    return {};
  }

  mochi::Context* context = IMochiPhysicsModule::Get()->GetMochiContext();
  if (!context) {
    return {};
  }

  // Load polyline shape from H5 model
  mochi::ShapeHandle polylineShape = Model->GetMochiShape(BakeScale, FTransform::Identity);
  if (!polylineShape.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] Failed: polyline shape handle is invalid."),
        __FUNCTION__);
    return {};
  }

  // If the model already provides a contact skin, prefer it. Otherwise fall back to
  // generating a tubular contact skin from the centerline below.
  {
    mochi::ErrorLog probeError{};
    mochi::MeshDataView modelContactSkin =
        context->GetShapeContactSkinMesh(polylineShape, probeError);
    bOutModelHasContactSkin = probeError.IsOK() && !modelContactSkin.IsEmpty();
  }
  if (bUseModelContactSkin && bOutModelHasContactSkin) {
    return polylineShape;
  }

  // Extract node positions from the polyline shape
  mochi::ErrorLog e{};
  mochi::MeshDataView meshView = context->GetShapeMesh(polylineShape, e);
  if (!e.IsOK() || meshView.IsEmpty()) {
    UE_LOG(
        LogMochiPhysics, Error, TEXT("[%hs] Failed to get mesh data for rod shape."), __FUNCTION__);
    return {};
  }
  mochi::Span<mochi::Real3 const> nodes =
      mochi::Unflatten<mochi::Real3 const>(meshView.coordinates);
  bool const isClosedLoop = mochi::IsPolylineClosedLoop(meshView);

  // Generate a tubular contact skin from the polyline centerline.
  mochi::ModelData model = mochi::experimental::GenerateTubularRodModelData(
      nodes,
      {} /* elementFrameAxes - auto-generate via parallel transport */,
      mochi::real(ProceduralTubeRadius),
      NumCrossSectionSegments,
      isClosedLoop,
      e);
  if (!e.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] Failed to generate tubular rod model data."),
        __FUNCTION__);
    return {};
  }

  // Create a shape with the polyline simulation mesh and triangulated contact skin.
  mochi::ShapeHandle combinedShape = context->CreateModelShape(model, e);
  if (!e.IsOK()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[%hs] Failed to create rod model shape."), __FUNCTION__);
    return {};
  }

  return combinedShape;
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

  const FVector Scale = GetComponentTransform().GetScale3D();
  bool modelHasContactSkin = false;
  mochi::ShapeHandle shape = CreateRodShape(Scale, modelHasContactSkin);
  bModelHasContactSkin = modelHasContactSkin;
  if (!shape.IsValid()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[%hs] Failed: shape handle is invalid."), __FUNCTION__);
    return;
  }

  mochi::PointCloudColliderParams pointCloudParams;
  pointCloudParams.radius = mochi::real(PointCloudRadius);
  pointCloudParams.selfContact = bPointCloudSelfContact;
  pointCloudParams.selfContactExclusionRatio = mochi::real(PointCloudSelfContactExclusionRatio);
  pointCloudParams.colliderSegmentElementType = ToMochi(ColliderSegmentElementType);

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
}

UMochiRodMeshComponent::UMochiRodMeshComponent(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  // Required so OnUpdateTransform is invoked on transform edits, so the
  // editor preview can rebake the tube when the user scales the actor.
  bWantsOnUpdateTransform = true;
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
  // Super first so ComponentToWorld is final before we bake.
  Super::OnRegister();
  check(bWantsOnUpdateTransform);
  SetEnableFlatShading(!bUseMochiNodeNormals);
  RebuildPreviewMesh();
  EnsureBodySetupPublic();
  ValidateMaterialSlots();
}

void UMochiRodMeshComponent::OnUpdateTransform(
    EUpdateTransformFlags UpdateTransformFlags,
    ETeleportType Teleport) {
  Super::OnUpdateTransform(UpdateTransformFlags, Teleport);
  if (IsTemplate()) {
    return;
  }
  const FVector Scale = GetComponentTransform().GetScale3D();
  // After BeginPlay the simulation actor's geometry is frozen at the scale
  // baked into the polyline by CreateMochiActor. Snap the world scale back so
  // the rendered tube does not desync from the simulated contact geometry.
  if (HasBegunPlay()) {
    if (Scale != LastBakedScale) {
      if (!bWarnedRuntimeScaleChange) {
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT(
                "[%hs] Runtime scale change on rod '%s' from (%s) to (%s) is not "
                "supported; simulation geometry is frozen at BeginPlay. Snapping "
                "the component's world scale back."),
            __FUNCTION__,
            *GetName(),
            *LastBakedScale.ToString(),
            *Scale.ToString());
        bWarnedRuntimeScaleChange = true;
      }
      SetWorldScale3D(LastBakedScale);
    }
    return;
  }
  // Exact compare: any per-axis discrepancy affects the inverse-scale bake.
  if (Scale == LastBakedScale) {
    return;
  }
  RebuildPreviewMesh();
}

void UMochiRodMeshComponent::PostLoad() {
  Super::PostLoad();
  RebakeIfNeeded();
}

void UMochiRodMeshComponent::PostEditImport() {
  Super::PostEditImport();
  RebakeIfNeeded();
}

void UMochiRodMeshComponent::PostDuplicate(bool bDuplicateForPIE) {
  Super::PostDuplicate(bDuplicateForPIE);
  RebakeIfNeeded();
}

#if WITH_EDITOR
bool UMochiRodMeshComponent::CanEditChange(const FProperty* InProperty) const {
  if (!Super::CanEditChange(InProperty)) {
    return false;
  }
  // During PIE, the Mochi simulation actor's geometry is frozen at the
  // vertex count baked in CreateMochiActor. Disabling these vertex-count-
  // affecting knobs in the Details panel prevents the user from rebuilding
  // the dynamic mesh under the live simulation, which would desync the
  // rendered tube from the simulated contact geometry. PostEditChangeProperty
  // provides a programmatic backstop for any path that bypasses the UI.
  if (HasBegunPlay() && InProperty) {
    const FName Name = InProperty->GetFName();
    if (Name == GET_MEMBER_NAME_CHECKED(UMochiRodMeshComponent, Model) ||
        Name == GET_MEMBER_NAME_CHECKED(UMochiRodMeshComponent, ProceduralTubeRadius) ||
        Name == GET_MEMBER_NAME_CHECKED(UMochiRodMeshComponent, NumCrossSectionSegments) ||
        Name == GET_MEMBER_NAME_CHECKED(UMochiRodMeshComponent, bUseModelContactSkin)) {
      return false;
    }
  }
  return true;
}

void UMochiRodMeshComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) {
  MOCHI_PROFILE_SCOPE();
  Super::PostEditChangeProperty(PropertyChangedEvent);
  const FName PropName = PropertyChangedEvent.GetPropertyName();
  if (PropName == GET_MEMBER_NAME_CHECKED(UMochiRodMeshComponent, Model) ||
      PropName == GET_MEMBER_NAME_CHECKED(UMochiRodMeshComponent, ProceduralTubeRadius) ||
      PropName == GET_MEMBER_NAME_CHECKED(UMochiRodMeshComponent, NumCrossSectionSegments) ||
      PropName == GET_MEMBER_NAME_CHECKED(UMochiRodMeshComponent, bUseModelContactSkin)) {
    UE_LOG(LogMochiPhysics, Display, TEXT("[%hs] %s changed"), __FUNCTION__, *PropName.ToString());
    // Route through RebakeIfNeeded so property edits during PIE don't desync
    // the rendered contact skin from the frozen simulation geometry.
    RebakeIfNeeded();
    ValidateMaterialSlots();
  }
  EnsureBodySetupPublic();
}

void UMochiRodMeshComponent::PostEditUndo() {
  Super::PostEditUndo();
  RebakeIfNeeded();
}
#endif
