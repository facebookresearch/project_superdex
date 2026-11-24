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

/**
 * Populate dynamic render geometry from a model asset.
 *
 * @param OutMesh Existing mesh storage supplied to the operation.
 * @param MochiModel Model asset supplying geometry.
 * @param ShapeTransform Transform applied to the model geometry.
 * @param bUseVisualMeshIfPresent Whether available visual geometry is preferred.
 * @param bUseContactSkinIfPresent Whether an available contact skin is preferred.
 */
void SUPERDEXPHYSICS_API PopulateMeshFromMochiModel(
    UE::Geometry::FDynamicMesh3* OutMesh,
    UMochiModel* MochiModel,
    const FTransform& ShapeTransform = FTransform::Identity,
    bool bUseVisualMeshIfPresent = true,
    bool bUseContactSkinIfPresent = false);

/**
 * Populate dynamic render geometry from a native actor.
 *
 * @param OutMesh Mesh storage populated by the operation.
 * @param MochiActor Native actor handle supplying the mesh.
 * @param MochiScene Scene wrapper associated with the native actor.
 * @param bUseVisualMeshIfPresent Whether available visual geometry is preferred.
 * @param bUseContactSkinIfPresent Whether to use an available model contact skin.
 */
void SUPERDEXPHYSICS_API PopulateMeshFromMochiActor(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ActorHandle const& MochiActor,
    UMochiSceneObject* MochiScene,
    bool bUseVisualMeshIfPresent = true,
    bool bUseContactSkinIfPresent = false);

/**
 * Populate dynamic render geometry from a native shape.
 *
 * @param OutMesh Mesh storage populated by the operation.
 * @param Shape Native shape handle supplying actor geometry.
 * @param bUseVisualMeshIfPresent Whether available visual geometry is preferred.
 * @param bUseContactSkinIfPresent Whether to use an available model contact skin.
 */
void SUPERDEXPHYSICS_API PopulateMeshFromMochiShape(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ShapeHandle Shape,
    bool bUseVisualMeshIfPresent,
    bool bUseContactSkinIfPresent);

/**
 * Convert native node coordinates into Unreal positions.
 *
 * @param InMochi Native node coordinates to convert.
 * @param OutUnreal Array populated with converted Unreal positions.
 * @param Scale3D Scale applied to the geometry.
 */
void SUPERDEXPHYSICS_API ConvertNodePositionsToUnreal(
    const mochi::Span<mochi::real const>& InMochi,
    TArray<FVector>& OutUnreal,
    FVector Scale3D = FVector::OneVector);

/**
 * Convert native node coordinates into Unreal positions.
 *
 * @param InMochi Native node coordinates to convert.
 * @param OutUnreal Array populated with converted Unreal positions.
 * @param Scale3D Scale applied to the geometry.
 */
void SUPERDEXPHYSICS_API ConvertNodePositionsToUnreal(
    const TArray<mochi::real>& InMochi,
    TArray<FVector>& OutUnreal,
    FVector Scale3D = FVector::OneVector);

/**
 * currently only used for editor tools, but could be useful later
 * i.e. respond to deformation on the element level?
 *
 * @param MochiModel Model asset supplied to the operation.
 * @param Scale3D Scale applied to the geometry.
 * @param OutReferencePositionsUnreal Array populated with converted reference positions.
 */
void SUPERDEXPHYSICS_API ConvertElementBarycenterReferencePositions(
    const UMochiModel* MochiModel,
    FVector Scale3D,
    TArray<FVector>& OutReferencePositionsUnreal);

/**
 * Convert volume-node reference positions into Unreal space.
 *
 * @param MochiModel Model asset supplied to the operation.
 * @param Scale3D Scale applied to the geometry.
 * @param OutReferencePositionsUnreal Array populated with converted reference positions.
 */
void SUPERDEXPHYSICS_API ConvertVolumeNodeReferencePositions(
    const UMochiModel* MochiModel,
    FVector Scale3D,
    TArray<FVector>& OutReferencePositionsUnreal);

/**
 * Give each vertex of a mesh built from a Mochi mesh its render-model attributes, in overlays
 * whose element IDs are the vertex IDs: a normal overlay starting from the vertex normals, and a
 * UV or color overlay when UVs or Colors (linear RGBA) is non-empty. Each non-empty array holds
 * one value per vertex.
 *
 * @param Mesh Existing dynamic mesh to populate or update.
 * @param UVs Texture coordinates to add to the render mesh.
 * @param Colors Vertex colors to add to the render mesh.
 */
void SUPERDEXPHYSICS_API EnableRenderModelAttributes(
    UE::Geometry::FDynamicMesh3& Mesh,
    TConstArrayView<FVector2f> UVs,
    TConstArrayView<FLinearColor> Colors);

/**
 * Update vertex positions (and optionally normals) on a dynamic mesh from Mochi query data.
 * bUpdateRenderModelNormalOverlay requires the one-overlay-element-per-vertex layout built by
 * EnableRenderModelAttributes. UV and color overlays are left unchanged.
 *
 * @param NodePositions Native node coordinates used to update the mesh.
 * @param NodeNormals Native node normals used to update the mesh.
 * @param bUseMochiNodeNormals Whether native node normals are used.
 * @param bUpdateRenderModelNormalOverlay Whether the render-model normal overlay is updated.
 * @param InvScale3D Inverse scale applied to converted coordinates.
 * @param Mesh Existing dynamic mesh to populate or update.
 */
void SUPERDEXPHYSICS_API UpdateDynamicMeshFromMochiQuery(
    const TArray<mochi::real>& NodePositions,
    const TArray<mochi::real>& NodeNormals,
    bool bUseMochiNodeNormals,
    bool bUpdateRenderModelNormalOverlay,
    FVector InvScale3D,
    UE::Geometry::FDynamicMesh3& Mesh);

/**
 * Bake the inverse of the given scale into mesh vertex positions (for runtime deformation).
 *
 * @param Scale3D Scale applied to the geometry.
 * @param Mesh Existing dynamic mesh to populate or update.
 */
void SUPERDEXPHYSICS_API
PrepareDynamicMeshForRuntime(FVector Scale3D, UE::Geometry::FDynamicMesh3& Mesh);

/**
 * Compute and store vertex normals if requested, otherwise discard them.
 *
 * @param Mesh Existing dynamic mesh to populate or update.
 * @param bComputeNormals Whether vertex normals are computed or discarded.
 */
void SUPERDEXPHYSICS_API
ComputeOrDiscardVertexNormals(UE::Geometry::FDynamicMesh3& Mesh, bool bComputeNormals);
