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

#include "mochi_common_components.h"
#include "mochi_discretization_components.h"
#include "mochi_ecs.h"
#include "mochi_query.h"
#include "mochi_shape.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace mochi {

// Forward declarations for ecs::Excluded in the auxiliary mesh query systems
struct CRodVisualMeshEmbedding;
struct CRodContactSkinEmbedding;

/**************************************************************************
  Common ECS Utils related to discretization
*/

// Compute the local-space positions of each volume node and store them in CQueryNodePositions
void UpdateQueryNodePositions(
    CSimplicialMesh const& simplicial,
    CFinalDisplacementRef<TimeStep::Current> const& currSol,
    CQueryNodePositions& outQuery);

// Computes the positions of all the embedded nodes: their reference positions plus the weighted
// displacements of their source nodes. The reference and output spans hold one entry per embedded
// node.
void UpdateLinearEmbeddedNodePositionsFromDisplacements(
    LinearMeshEmbedding const& embedding,
    Span<Real3 const> embeddedReferencePositions,
    Span<Real3 const> displacements,
    Span<Real3> outPositions);

// Compute the local-space positions of each node in the surface mesh
// and store them in CQuerySurfaceNodePositions
void UpdateQuerySurfaceNodePositions(
    CSurfaceMesh const& simplicial,
    CFinalDisplacementRef<TimeStep::Current> const* currSol,
    ecs::OptionalTag<TagRigidActor> isRigid,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CQuerySurfaceNodePositions& outQuery);

// Compute the local-space normals of each boundary node and store them in
// CQuerySurfaceNodeNormals
void UpdateQuerySurfaceNodeNormals(
    CQuerySurfaceNodePositions const& queryPositions,
    CSurfaceMesh const& triMesh,
    ecs::OptionalTag<TagRigidActor> isRigid,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CQuerySurfaceNodeNormals& outQuery);

// Compute per-node normals of an auxiliary (visual or contact skin) mesh (area-weighted
// per-triangle normals). For rigid actors, normals are computed once. Normals of unreferenced
// nodes are zero.
void UpdateAuxiliaryMeshNodeNormals(
    bool isRigid,
    TriangularMesh const& mesh,
    Span<real const> positions,
    DynamicArray<real>& outNormals);

void UpdateQueryVisualNodePositionsAndNormals(
    ecs::Excluded<CRodVisualMeshEmbedding>,
    ecs::OptionalTag<TagRigidActor> isRigidDynamic,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CVisualMesh const& visualMesh,
    CQueryNodePositions const* posQuery,
    CQueryVisualNodePositions& outVisPosQuery,
    CQueryVisualNodeNormals* outVisNormQuery);

void UpdateQueryContactSkinNodePositionsAndNormals(
    ecs::Excluded<CRodContactSkinEmbedding>,
    CContactSkinMesh const& contactSkin,
    CQueryNodePositions const& posQuery,
    CQueryContactSkinNodePositions& outPosQuery,
    CQueryContactSkinNodeNormals* outNormQuery);

/**************************************************************************
  Overload set for creating active VOLUME elements
*/

CActiveVolumeElements CreateActiveVolumeElements(
    experimental::SampleMeshInitFromFile const& params,
    TetrahedralMeshShape const& tetMeshShape);

CActiveVolumeElements CreateActiveVolumeElements(
    experimental::SampleMeshInitRandomSampling const& params,
    TetrahedralMeshShape const& tetMeshShape);

CActiveVolumeElements CreateActiveVolumeElements(
    experimental::SampleMeshInitFromSpecificVolumeElements const& params,
    TetrahedralMeshShape const& tetMeshShape);

/**************************************************************************
Overload set for creating active BOUNDARY faces
*/

// Picks one index at random, with a fixed seed, in each of numPicks near-equal windows of
// consecutive indices that partition [0, numItems). The picks are ascending.
[[nodiscard]] std::vector<int> PickOneIndexPerWindow(int numItems, int numPicks);

template <typename BoundaryTraceT>
CActiveBoundaryFaces CreateActiveBoundaryFaces(
    experimental::SampleMeshInitRandomSampling const& params,
    std::shared_ptr<TetrahedralMesh const> const& tetMesh,
    std::vector<BoundaryTraceT> const& allBoundaryTraces) {
  static_assert(
      BoundaryTraceT::kSpaceDimParam == 3,
      "Overload requires BoundaryTraceT to be the trace of a 3D element");
  std::vector<int> activeBoundaryTraceInds;
  int const stepSize = params.stepSizeForBoundaryElementsSelection;
  if (stepSize != rom::hyper::kNoElements) {
    MOCHI_ASSERT(
        stepSize >= 1,
        "Invalid stepSize: must be kNoElements (%d) or >= 1",
        rom::hyper::kNoElements);

    int const numFaces = isize(allBoundaryTraces);
    activeBoundaryTraceInds = PickOneIndexPerWindow(numFaces, (numFaces + stepSize - 1) / stepSize);
  }
  return CActiveBoundaryFaces(tetMesh, activeBoundaryTraceInds, MakeConstSpan(allBoundaryTraces));
}

CActiveBoundaryFaces CreateActiveBoundaryFaces(
    experimental::SampleMeshInitFromFile const& params,
    TetrahedralMeshShape const& tetMeshShape,
    CFemBoundaryDiscretization const& boundaryDiscrVariant);

CActiveBoundaryFaces CreateActiveBoundaryFaces(
    experimental::SampleMeshInitRandomSampling const& params,
    TetrahedralMeshShape const& tetMeshShape,
    CFemBoundaryDiscretization const& boundaryDiscrVariant);

CActiveBoundaryFaces CreateActiveBoundaryFaces(
    experimental::SampleMeshInitFromSpecificVolumeElements const& params,
    TetrahedralMeshShape const& tetMeshShape,
    CFemBoundaryDiscretization const& boundaryDiscrVariant);

CActiveBoundaryFaces CreateActiveBoundaryFaces(
    BoundarySubsamplingParams const& params,
    TetrahedralMeshShape const& tetMeshShape,
    CFemBoundaryDiscretization const& boundaryDiscrVariant);

CActiveBoundaryFaces CreateActiveBoundaryFaces(
    BoundarySubsamplingParams const& params,
    std::shared_ptr<TriangularMesh const> const& triMesh,
    CFemSurfaceDiscretization const& surfaceDiscrVariant);

namespace discretization {
void InitializeOnce(entt::registry& reg);
} // namespace discretization

} // namespace mochi
