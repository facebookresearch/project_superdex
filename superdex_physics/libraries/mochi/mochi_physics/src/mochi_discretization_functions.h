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
#include <mochi_physics/mochi_physics_experimental.h>

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

namespace mochi {

// Forward declaration for ecs::Excluded in UpdateQueryVisualNodePositionsAndNormals
struct CRodVisualMeshEmbedding;

/**************************************************************************
  Common ECS Utils related to discretization
*/

// Compute the local-space positions of each volume node and store them in CQueryNodePositions
void UpdateQueryNodePositions(
    CSimplicialMesh const& simplicial,
    CFinalDisplacementRef<TimeStep::Current> const& currSol,
    CQueryNodePositions& outQuery);

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

// Compute the local-space positions and (optionally) normals of visual mesh nodes.
// For rigid actors, the reference visual mesh positions are used directly.
// For deformable actors, posQuery must be non-null and have been computed at input, and is
// mapped through the visual mesh embedding.
void UpdateQueryVisualNodePositionsAndNormals(
    ecs::Excluded<CRodVisualMeshEmbedding>,
    ecs::OptionalTag<TagRigidActor> isRigidDynamic,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CVisualMesh const& visualMesh,
    CQueryNodePositions const* posQuery,
    CQueryVisualNodePositions& outVisPosQuery,
    CQueryVisualNodeNormals* outVisNormQuery);

// Compute per-vertex normals from visual mesh positions (area-weighted per-triangle normals).
// Shared utility used by both soft/shell and rod visual query implementations.
void UpdateQueryVisualNodeNormals(
    bool isRigid,
    CVisualMesh const& visualMesh,
    CQueryVisualNodePositions const& visPosQuery,
    CQueryVisualNodeNormals& outVisNormQuery);

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
    activeBoundaryTraceInds.reserve((isize(allBoundaryTraces) + stepSize - 1) / stepSize);
    for (int i = 0; i < isize(allBoundaryTraces); i += stepSize) {
      activeBoundaryTraceInds.push_back(i);
    }
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

/**************************************************************************
Overload set for updating active elements
*/

template <typename BoundaryTraceT>
void UpdateActiveElementsAndNodes(
    experimental::SampleMeshInitFromFile const& /*s*/,
    TetrahedralMeshShape const& /*tetMeshShape*/,
    std::vector<BoundaryTraceT> const& /*allBoundaryTraces*/,
    CActiveVolumeElements& /*activeVolElements*/,
    CActiveBoundaryFaces& /*activeBdFaces*/,
    CActiveUniqueNodes& /*activeUniqueNodes*/) {
  static_assert(
      BoundaryTraceT::kSpaceDimParam == 3, "BoundaryTraceT must be the trace of a 3D element");
  MOCHI_ASSERT(false, "missing impl");
}

template <typename BoundaryTraceT>
void UpdateActiveElementsAndNodes(
    experimental::SampleMeshInitRandomSampling const& s,
    TetrahedralMeshShape const& tetMeshShape,
    std::vector<BoundaryTraceT> const& allBoundaryTraces,
    CActiveVolumeElements& activeVolElements,
    CActiveBoundaryFaces& activeBdFaces,
    CActiveUniqueNodes& activeUniqueNodes) {
  static_assert(
      BoundaryTraceT::kSpaceDimParam == 3, "BoundaryTraceT must be the trace of a 3D element");
  activeVolElements = CreateActiveVolumeElements(s, tetMeshShape);
  activeBdFaces = CreateActiveBoundaryFaces(s, *tetMeshShape.GetMesh(), allBoundaryTraces);
  activeUniqueNodes.Recompute(activeVolElements, activeBdFaces);
}

template <typename BoundaryTraceT>
void UpdateActiveElementsAndNodes(
    experimental::SampleMeshInitFromSpecificVolumeElements const& s,
    TetrahedralMeshShape const& tetMeshShape,
    std::vector<BoundaryTraceT> const& allBoundaryTraces,
    CActiveVolumeElements& activeVolElements,
    CActiveBoundaryFaces& activeBdFaces,
    CActiveUniqueNodes& activeUniqueNodes) {
  static_assert(
      BoundaryTraceT::kSpaceDimParam == 3, "BoundaryTraceT must be the trace of a 3D element");
  auto const& tetMesh = *tetMeshShape.GetMesh();

  // Volume elements
  activeVolElements.Recompute(s.ids);

  // Surface elements
  std::unordered_set<int> activeVolumeElementInds(s.ids.begin(), s.ids.end());
  std::vector<int> activeBoundaryFaceInds;
  int const numBdFaces = tetMesh.GetNumBoundaryFaces();
  auto const faces = tetMesh.GetBoundaryFaces();
  for (int i = 0; i < numBdFaces; ++i) {
    if (activeVolumeElementInds.count(faces[i].element) == 1) {
      activeBoundaryFaceInds.emplace_back(i);
    }
  }
  activeBdFaces.Recompute(activeBoundaryFaceInds, allBoundaryTraces);

  // Unique volume nodes
  activeUniqueNodes.Recompute(activeVolElements, activeBdFaces);
}

namespace discretization {
void InitializeOnce(entt::registry& reg);
} // namespace discretization

} // namespace mochi
