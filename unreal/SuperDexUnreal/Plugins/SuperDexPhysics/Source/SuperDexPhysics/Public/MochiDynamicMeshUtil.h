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

#pragma once

#include "CoreMinimal.h"
#include "MochiModel.h"
#include "MochiSceneObject.h"
#include "UDynamicMesh.h"

#include <mochi_core/mochi_config.h>

void SUPERDEXPHYSICS_API PopulateMeshFromMochiModel(
    UE::Geometry::FDynamicMesh3* OutMesh,
    UMochiModel* MochiModel,
    const FTransform& ShapeTransform = FTransform::Identity,
    bool bUseVisualMeshIfPresent = true,
    bool bUseContactSkinIfPresent = false);

void SUPERDEXPHYSICS_API PopulateMeshFromMochiActor(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ActorHandle const& MochiActor,
    UMochiSceneObject* MochiScene,
    bool bUseVisualMeshIfPresent = true,
    bool bUseContactSkinIfPresent = false);

void SUPERDEXPHYSICS_API PopulateMeshFromMochiShape(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ShapeHandle shape,
    bool bUseVisualMeshIfPresent,
    bool bUseContactSkinIfPresent);

void SUPERDEXPHYSICS_API ConvertNodePositionsToUnreal(
    const mochi::Span<mochi::real const>& InMochi,
    TArray<FVector>& OutUnreal,
    FVector Scale3D = FVector::OneVector);

void SUPERDEXPHYSICS_API ConvertNodePositionsToUnreal(
    const TArray<mochi::real>& InMochi,
    TArray<FVector>& OutUnreal,
    FVector Scale3D = FVector::OneVector);

// currently only used for editor tools, but could be useful later
// i.e. respond to deformation on the element level?
void SUPERDEXPHYSICS_API ConvertElementBarycenterReferencePositions(
    const UMochiModel* MochiModel,
    FVector Scale3D,
    TArray<FVector>& OutReferencePositionsUnreal);

void SUPERDEXPHYSICS_API ConvertVolumeNodeReferencePositions(
    const UMochiModel* MochiModel,
    FVector Scale3D,
    TArray<FVector>& OutReferencePositionsUnreal);

/// Give each vertex of a mesh built from a Mochi mesh its render-model attributes, in overlays
/// whose element IDs are the vertex IDs: a normal overlay starting from the vertex normals, and a
/// UV or color overlay when UVs or Colors (linear RGBA) is non-empty. Each non-empty array holds
/// one value per vertex.
void SUPERDEXPHYSICS_API EnableRenderModelAttributes(
    UE::Geometry::FDynamicMesh3& Mesh,
    TConstArrayView<FVector2f> UVs,
    TConstArrayView<FLinearColor> Colors);

/// Update vertex positions (and optionally normals) on a dynamic mesh from Mochi query data.
/// bUpdateRenderModelNormalOverlay requires the one-overlay-element-per-vertex layout built by
/// EnableRenderModelAttributes. UV and color overlays are left unchanged.
void SUPERDEXPHYSICS_API UpdateDynamicMeshFromMochiQuery(
    const TArray<mochi::real>& NodePositions,
    const TArray<mochi::real>& NodeNormals,
    bool bUseMochiNodeNormals,
    bool bUpdateRenderModelNormalOverlay,
    FVector InvScale3D,
    UE::Geometry::FDynamicMesh3& Mesh);

/// Bake the inverse of the given scale into mesh vertex positions (for runtime deformation).
void SUPERDEXPHYSICS_API
PrepareDynamicMeshForRuntime(FVector Scale3D, UE::Geometry::FDynamicMesh3& Mesh);

/// Compute and store vertex normals if requested, otherwise discard them.
void SUPERDEXPHYSICS_API
ComputeOrDiscardVertexNormals(UE::Geometry::FDynamicMesh3& Mesh, bool bComputeNormals);
