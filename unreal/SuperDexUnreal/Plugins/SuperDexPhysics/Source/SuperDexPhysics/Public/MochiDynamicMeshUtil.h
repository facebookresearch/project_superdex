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

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/mochi_config.h>
#include <mochi_physics/cpp_api/mochi_enums.h>

/**
 * Which mesh of a Mochi shape or actor is drawn. The drawn mesh and the node queries that deform
 * it must use the same one, so that vertex IDs are node indices.
 * Surface is first so value-initialization gives the fallback used when no mesh is selected.
 */
enum class EMochiRenderMeshSource : uint8 { Surface, Visual, ContactSkin };

/**
 * Whether actors of Type deform (soft, shell, rod).
 *
 * @param Type Native actor type to classify.
 * @return Whether the actor type is deformable.
 */
bool SUPERDEXPHYSICS_API IsDeformableActorType(mochi::ActorType Type);

/**
 * The visual mesh if usable; else the contact skin if usable, whether or not it is used for
 * contact; else the surface mesh. Pass an empty view for a mesh that is not wanted. An embedded
 * mesh is usable if present and, if bPosedByNodeQueries, skinned, since skinning is what ties its
 * vertices to the actor's nodes. Takes shape views, which carry skinning for every embedding,
 * rods' included; a deformable actor's embedding is built from that skinning.
 *
 * @param VisualMesh Shape visual mesh, or an empty view if not wanted.
 * @param ContactSkinMesh Shape contact skin, or an empty view if not wanted.
 * @param bPosedByNodeQueries False for a mesh only ever moved by a rigid transform.
 * @return The selected mesh source.
 */
EMochiRenderMeshSource SUPERDEXPHYSICS_API ChooseRenderMeshSource(
    mochi::MeshDataView const& VisualMesh,
    mochi::MeshDataView const& ContactSkinMesh,
    bool bPosedByNodeQueries);

/** A mesh view of a Mochi shape, which mesh it is, and the shape whose storage it views. */
struct FMochiRenderMesh {
  mochi::ShapeHandle Shape;
  mochi::MeshDataView Data;
  EMochiRenderMeshSource Source = EMochiRenderMeshSource::Surface;
};

/**
 * The mesh of Actor's reference shape that Actor's node queries pose: ChooseRenderMeshSource's
 * choice, posed by node queries if Actor is deformable, among the meshes whose node queries Actor
 * supports. A rod with no usable visual mesh or contact skin returns empty Data without error.
 * Otherwise, sets Error if the selected mesh has no supported node query. Call on the Mochi
 * thread.
 *
 * @param Actor Native actor whose reference shape is drawn.
 * @param bUseContactSkinIfPresent Whether a usable contact skin may be selected.
 * @param Error Error set if no drawable mesh has a supported node query.
 * @return The drawn mesh, its source, and its shape.
 */
FMochiRenderMesh SUPERDEXPHYSICS_API
GetDrawnMochiMesh(mochi::Actor const& Actor, bool bUseContactSkinIfPresent, mochi::Error& Error);

/**
 * Populate dynamic render geometry from a native actor. If bPosedByNodeQueries, use
 * GetDrawnMochiMesh's actor query selection; otherwise, select a mesh for rigid posing.
 *
 * @param OutMesh Mesh storage populated by the operation.
 * @param MochiActor Native actor handle supplying the mesh.
 * @param MochiScene Scene wrapper associated with the native actor.
 * @param bUseContactSkinIfPresent Whether to use an available model contact skin.
 * @param bPosedByNodeQueries Whether the mesh is posed by node queries; see ChooseRenderMeshSource.
 * @param OutSource Optional; set to the populated mesh's source, or Surface if nothing is drawn.
 */
void SUPERDEXPHYSICS_API PopulateMeshFromMochiActor(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ActorHandle const& MochiActor,
    UMochiSceneObject* MochiScene,
    bool bUseContactSkinIfPresent = false,
    bool bPosedByNodeQueries = true,
    EMochiRenderMeshSource* OutSource = nullptr);

/**
 * Populate dynamic render geometry from a native shape.
 *
 * @param OutMesh Mesh storage populated by the operation.
 * @param Shape Native shape handle supplying actor geometry.
 * @param bUseContactSkinIfPresent Whether to use an available model contact skin.
 * @param bPosedByNodeQueries Whether the mesh is posed by node queries; see ChooseRenderMeshSource.
 * @param OutSource Optional; set to the populated mesh's source, or Surface if nothing is drawn.
 */
void SUPERDEXPHYSICS_API PopulateMeshFromMochiShape(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ShapeHandle Shape,
    bool bUseContactSkinIfPresent,
    bool bPosedByNodeQueries = true,
    EMochiRenderMeshSource* OutSource = nullptr);

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
