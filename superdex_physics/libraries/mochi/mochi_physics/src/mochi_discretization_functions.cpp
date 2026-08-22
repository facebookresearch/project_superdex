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

#include "mochi_discretization_functions.h"

#include "mochi_ecs_utils.h"
#include "mochi_rod.h"
#include "mochi_simulation.h"

#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/rand_utils.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::experimental;

namespace mochi::discretization {
void InitializeOnce(entt::registry& reg) {
  // for consistency, we follow below the same order
  // as the declaration in mochi_discretization_components.h

  ecs::RegisterComponent<CMeshPivot>(reg);
  ecs::RegisterComponent<CSimplicialMesh>(reg);
  ecs::RegisterComponent<CTetrahedralMesh>(reg);
  ecs::RegisterComponent<CTriangularMesh>(reg);
  ecs::RegisterComponent<CSurfaceMesh>(reg);
  ecs::RegisterComponent<CVisualMesh>(reg);
  ecs::RegisterComponent<CLocal2GlobalMap>(reg);
  ecs::RegisterComponent<CBoundaryLocal2GlobalMap>(reg);
  ecs::RegisterComponent<CContactLocal2GlobalMap>(reg);
  ecs::RegisterComponent<CFullSparsityPattern>(reg);
  ecs::RegisterComponent<CReducedSparsityPattern>(reg);
  ecs::RegisterComponent<CFemVolumeDiscretizationP1Q1>(reg);
  ecs::RegisterComponent<CFemVolumeDiscretizationP1Q4>(reg);
  ecs::RegisterComponent<CFemBoundaryDiscretizationP1Q1_1>(reg);
  ecs::RegisterComponent<CFemBoundaryDiscretizationP1Q1_3>(reg);
  ecs::RegisterComponent<CFemBoundaryDiscretizationP1Q1_6>(reg);
  ecs::RegisterComponent<CFemBoundaryDiscretizationP1Q1_7>(reg);
  ecs::RegisterComponent<CFemBoundaryDiscretizationP1Q1_12>(reg);
  ecs::RegisterComponent<CFemBoundaryDiscretizationP1Q1_16>(reg);
  ecs::RegisterComponent<CFemBoundaryDiscretization>(reg);
  ecs::RegisterComponent<CFemSurfaceDiscretizationP1Q1>(reg);
  ecs::RegisterComponent<CFemSurfaceDiscretizationP1Q3>(reg);
  ecs::RegisterComponent<CFemSurfaceDiscretizationP1Q6>(reg);
  ecs::RegisterComponent<CFemSurfaceDiscretizationP1Q7>(reg);
  ecs::RegisterComponent<CFemSurfaceDiscretizationP1Q12>(reg);
  ecs::RegisterComponent<CFemSurfaceDiscretizationP1Q16>(reg);
  ecs::RegisterComponent<CFemSurfaceDiscretization>(reg);
  ecs::RegisterComponent<CFemSurfaceDiscretizationLite>(reg);
  ecs::RegisterComponent<CFemSegmentDiscretizationP1Q1>(reg);
  ecs::RegisterComponent<CFemSegmentDiscretizationP1Q2>(reg);
  ecs::RegisterComponent<CFemSegmentDiscretizationP1Q3>(reg);
  ecs::RegisterComponent<CFemSegmentDiscretization>(reg);
  ecs::RegisterComponent<CActiveVolumeElements>(reg);
  ecs::RegisterComponent<CActiveBoundaryFaces>(reg);
  ecs::RegisterComponent<CActiveUniqueNodes>(reg);
}
} // namespace mochi::discretization

void mochi::UpdateQueryNodePositions(
    CSimplicialMesh const& simplicial,
    CFinalDisplacementRef<TimeStep::Current> const& currSol,
    CQueryNodePositions& outQuery) {
  MOCHI_PROFILE_SCOPE();
  auto const* mesh = simplicial.mesh.get();

  auto displacements = currSol.value.GetConstSpan();

  // Ensure that the output vector is of the correct size.
  int const numValues = kSpaceDim3 * mesh->GetNumNodes();
  outQuery.nodePositions.resize(numValues);

  // Retrieve reference positions.
  Span<real const> referencePositions = mochi::Flatten(mesh->GetNodeCoordinates());

  // Compute reference positions + displacements.
  ArrayAdd(MakeSpan(outQuery.nodePositions), referencePositions, displacements);
}

void mochi::UpdateQuerySurfaceNodePositions(
    CSurfaceMesh const& simplicial,
    CFinalDisplacementRef<TimeStep::Current> const* currSol,
    ecs::OptionalTag<TagRigidActor> isRigid,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CQuerySurfaceNodePositions& outQuery) {
  MOCHI_PROFILE_SCOPE();

  auto const* mesh = simplicial.mesh.get();

  // Since rigid actors don't deform, we only need to initialize the query output once.
  if (isStatic || isRigid) {
    if (outQuery.nodePositions.empty()) {
      Span<real const> refPositions = Flatten(simplicial.mesh->GetActiveNodeCoordinates());
      outQuery.nodePositions.assign(refPositions.begin(), refPositions.end());
    }
  } else {
    auto displacementsVolume = currSol->value.GetConstSpan();

    // Ensure that the output vector is of the correct size.
    int const numValues = kSpaceDim3 * mesh->GetNumActiveNodes();
    outQuery.nodePositions.clear();
    outQuery.nodePositions.reserve(numValues);
    auto activeNodes = mesh->GetActiveNodes();
    for (int i : activeNodes) {
      outQuery.nodePositions.push_back(displacementsVolume[i * 3 + 0]);
      outQuery.nodePositions.push_back(displacementsVolume[i * 3 + 1]);
      outQuery.nodePositions.push_back(displacementsVolume[i * 3 + 2]);
    }

    // Retrieve reference positions.
    Span<real const> referencePositions = Flatten(mesh->GetActiveNodeCoordinates());

    // Compute reference positions + displacements.
    ArrayAdd(
        MakeSpan(outQuery.nodePositions),
        MakeConstSpan(outQuery.nodePositions),
        referencePositions);
  }
}

void mochi::UpdateQuerySurfaceNodeNormals(
    CQuerySurfaceNodePositions const& queryPositions,
    CSurfaceMesh const& triMesh,
    ecs::OptionalTag<TagRigidActor> isRigid,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CQuerySurfaceNodeNormals& outQuery) {
  MOCHI_PROFILE_SCOPE();

  // Since rigid actors don't deform, we only need to initialize the query output once.
  if ((isRigid || isStatic) && !outQuery.nodeNormals.empty()) {
    return;
  }

  // Requires node positions
  if (queryPositions.nodePositions.empty()) {
    return;
  }

  TriangularMesh const* mesh = triMesh.mesh.get();

  // Make sure the output buffers are big enough
  int const numNodes = mesh->GetNumActiveNodes();
  real const* nodePositions = queryPositions.nodePositions.data();
  Span<int const> connectivity = mesh->GetActiveNodesFlatConnectivity();
  outQuery.faceCrossProducts.resize(mesh->GetNumElements());
  outQuery.nodeNormals.resize(kSpaceDim3 * numNodes + 1); // +1 for SIMD padding

  // First, compute the cross product of the legs of each triangle.
  int constexpr kMinPerTask = 1000; // TODO: Tune later
  ParallelForN("FaceCrossProduct", (int)mesh->GetNumElements(), kMinPerTask, [&](int iFace) {
    Vec4r v0 = Load<3, Vec4r>(nodePositions + (connectivity[iFace * 3 + 0] * kSpaceDim3));
    Vec4r v1 = Load<3, Vec4r>(nodePositions + (connectivity[iFace * 3 + 1] * kSpaceDim3));
    Vec4r v2 = Load<3, Vec4r>(nodePositions + (connectivity[iFace * 3 + 2] * kSpaceDim3));
    outQuery.faceCrossProducts[iFace] = Cross3(v1 - v0, v2 - v1);
  });

  // Then, for each boundary node, add the cross products of the adjacent faces
  // to get an area-weighted average of the adjacent face normals.
  ParallelForEach("ComputeNormal", mesh->GetActiveNodes(), kMinPerTask, [&](int iNode) {
    Span<int const> faces = mesh->GetAdjacentElements(iNode);
    int const numFaces = isize(faces);
    MOCHI_ASSERT_VERBOSE(
        numFaces != 0, "Every boundary node should be part at least one boundary face");
    Vec4r sum = outQuery.faceCrossProducts[faces[0]];
    for (int i = 1; i < numFaces; ++i) {
      sum += outQuery.faceCrossProducts[faces[i]];
    }
    Vec4r norm = Normalize<3>(sum);
    Store<3>(&outQuery.nodeNormals[mesh->GetAllToActiveNodesIndexMap(iNode) * kSpaceDim3], norm);
  });

  outQuery.nodeNormals.resize(kSpaceDim3 * numNodes); // trim SIMD padding
}

static void UpdateQueryVisualNodePositions(
    bool isRigid,
    CVisualMesh const& visualMesh,
    CQueryNodePositions const* positionQuery,
    CQueryVisualNodePositions& outQuery) {
  MOCHI_PROFILE_SCOPE();

  // Rigid actors don't deform: use reference positions, computed once.
  if (isRigid) {
    if (outQuery.nodePositions.empty()) {
      Span<real const> coordinates = Flatten(visualMesh.mesh->GetNodeCoordinates());
      outQuery.nodePositions.assign(coordinates.begin(), coordinates.end());
    }
    return;
  }

  // Deformable actors: map deformed positions through embedding.
  MOCHI_ASSERT(
      visualMesh.embedding && positionQuery && !positionQuery->nodePositions.empty(),
      "CQueryVisualNodePositions prerequisites for a deformable actor not satisfied.");
  auto const numValues = static_cast<size_t>(kSpaceDim3) * visualMesh.mesh->GetNumNodes();
  outQuery.nodePositions.resize(numValues);
  auto dstCoords = Unflatten<Real3>(MakeSpan(outQuery.nodePositions));
  auto srcCoords = Unflatten<Real3 const>(MakeSpan(positionQuery->nodePositions));
  visualMesh.embedding->Update(srcCoords, dstCoords);
}

void mochi::UpdateQueryVisualNodeNormals(
    bool isRigid,
    CVisualMesh const& visualMesh,
    CQueryVisualNodePositions const& visPosQuery,
    CQueryVisualNodeNormals& outVisNormQuery) {
  MOCHI_PROFILE_SCOPE();

  // Rigid actors don't deform: compute normals once and cache.
  if (isRigid && !outVisNormQuery.nodeNormals.empty()) {
    return;
  }

  TriangularMesh const* visMesh = visualMesh.mesh.get();

  // TODO[Nate] Compute the cross product only once per triangle, like
  // UpdateQuerySurfaceNodeNormals. Requires adjacency information, which TriangularMesh currently
  // doesn't have.
  auto const numValues = static_cast<size_t>(kSpaceDim3) * visMesh->GetNumNodes();
  outVisNormQuery.nodeNormals.clear(); // must be cleared to zeros for this algorithm to work
  outVisNormQuery.nodeNormals.resize(numValues, 0_r);
  Span<Int3 const> elements = visMesh->GetElementConnectivity();
  auto const& visualPositions = visPosQuery.nodePositions;
  auto& outVisualNormals = outVisNormQuery.nodeNormals;
  for (Int3 const& elem : elements) {
    Vec4r v0 = Load<3, Vec4r>(&visualPositions[kSpaceDim3 * elem[0]]);
    Vec4r v1 = Load<3, Vec4r>(&visualPositions[kSpaceDim3 * elem[1]]);
    Vec4r v2 = Load<3, Vec4r>(&visualPositions[kSpaceDim3 * elem[2]]);
    Vec4r n = ToSimdDirection(Cross3(v1 - v0, v2 - v0));

    real* dst = &outVisualNormals[kSpaceDim3 * elem[0]];
    Store<3>(dst, n + Load<3, Vec4r>(dst));
    dst = &outVisualNormals[kSpaceDim3 * elem[1]];
    Store<3>(dst, n + Load<3, Vec4r>(dst));
    dst = &outVisualNormals[kSpaceDim3 * elem[2]];
    Store<3>(dst, n + Load<3, Vec4r>(dst));
  }

  for (int i = 0; i < numValues; i += kSpaceDim3) {
    real* dst = &outVisualNormals[i];
    Vec4r n = Normalize<3>(Load<3, Vec4r>(dst));
    Store<3>(dst, n);
  }
}

void mochi::UpdateQueryVisualNodePositionsAndNormals(
    ecs::Excluded<CRodVisualMeshEmbedding>,
    ecs::OptionalTag<TagRigidActor> isRigidDynamic,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CVisualMesh const& visualMesh,
    CQueryNodePositions const* posQuery,
    CQueryVisualNodePositions& outVisPosQuery,
    CQueryVisualNodeNormals* outVisNormQuery) {
  bool const isRigid = isRigidDynamic || isStatic;

  // Compute visual positions first.
  UpdateQueryVisualNodePositions(isRigid, visualMesh, posQuery, outVisPosQuery);

  // Optionally, compute visual normals (requires the updated visual node positions).
  if (outVisNormQuery) {
    UpdateQueryVisualNodeNormals(isRigid, visualMesh, outVisPosQuery, *outVisNormQuery);
  }
}

/**************************************************************************
  Overload set for creating active VOLUME elements
*/

CActiveVolumeElements mochi::CreateActiveVolumeElements(
    SampleMeshInitFromFile const& params,
    TetrahedralMeshShape const& tetMeshShape) {
  auto it = tetMeshShape.GetSampleMeshes().find(std::string(params.source));
  MOCHI_ASSERT(it != tetMeshShape.GetSampleMeshes().end(), "Could not find sample mesh");

  auto const& sampleMeshInfo = it->second;
  auto const& elems = sampleMeshInfo.volumeElements;
  auto const& weights = sampleMeshInfo.volumeElementWeights;
  MOCHI_ASSERT(elems.size() == weights.size());
  return {tetMeshShape.GetMesh(), elems, weights};
}

CActiveVolumeElements mochi::CreateActiveVolumeElements(
    SampleMeshInitRandomSampling const& params,
    TetrahedralMeshShape const& tetMeshShape) {
  auto const& tetMesh = tetMeshShape.GetMesh();

  // if no elements are requested, then we need to still create the object
  // but it contains an empty set of indices
  int const stepSize = params.stepSizeForInteriorElementsSelection;
  if (stepSize == rom::hyper::kNoElements) {
    return {tetMesh, Span<int const>{}};
  }
  MOCHI_ASSERT(
      stepSize >= 1, "Invalid stepSize: must be kNoElements (%d) or >= 1", rom::hyper::kNoElements);

  int const numElements = tetMesh->GetNumElements();
  std::vector<int> activeVolElems;
  activeVolElems.reserve((numElements + stepSize - 1) / stepSize);
  for (int i = 0; i < numElements; i += stepSize) {
    activeVolElems.emplace_back(i);
  }
  return {tetMesh, activeVolElems};
}

CActiveVolumeElements mochi::CreateActiveVolumeElements(
    SampleMeshInitFromSpecificVolumeElements const& params,
    TetrahedralMeshShape const& tetMeshShape) {
  return {tetMeshShape.GetMesh(), params.ids};
}

/**************************************************************************
Overload set for creating active BOUNDARY TRACES elements
*/

CActiveBoundaryFaces mochi::CreateActiveBoundaryFaces(
    SampleMeshInitFromFile const& params,
    TetrahedralMeshShape const& tetMeshShape,
    CFemBoundaryDiscretization const& boundaryDiscrVariant) {
  return boundaryDiscrVariant.Visit([&](auto& boundaryDiscr) -> auto {
    auto it = tetMeshShape.GetSampleMeshes().find(std::string(params.source));
    MOCHI_ASSERT(it != tetMeshShape.GetSampleMeshes().end(), "Could not find sample mesh");

    auto const& sampleMeshInfo = it->second;
    auto const& activeBoundaryFaceIndices = sampleMeshInfo.boundaryElements;
    auto const& activeBoundaryFaceWeights = sampleMeshInfo.boundaryElementWeights;
    MOCHI_ASSERT(activeBoundaryFaceIndices.size() == activeBoundaryFaceWeights.size());
    return CActiveBoundaryFaces(
        tetMeshShape.GetMesh(),
        activeBoundaryFaceIndices,
        MakeConstSpan(boundaryDiscr.femElements),
        activeBoundaryFaceWeights);
  });
}

CActiveBoundaryFaces mochi::CreateActiveBoundaryFaces(
    SampleMeshInitRandomSampling const& params,
    TetrahedralMeshShape const& tetMeshShape,
    CFemBoundaryDiscretization const& boundaryDiscrVariant) {
  return boundaryDiscrVariant.Visit([&](auto& boundaryDiscr) -> auto {
    return CreateActiveBoundaryFaces(params, tetMeshShape.GetMesh(), boundaryDiscr.femElements);
  });
}

CActiveBoundaryFaces mochi::CreateActiveBoundaryFaces(
    SampleMeshInitFromSpecificVolumeElements const& params,
    TetrahedralMeshShape const& tetMeshShape,
    CFemBoundaryDiscretization const& boundaryDiscrVariant) {
  return boundaryDiscrVariant.Visit([&](auto& boundaryDiscr) -> auto {
    std::unordered_set<int> activeElementIndices(params.ids.begin(), params.ids.end());
    std::vector<int> activeBoundaryFaceIndices;
    auto const& tetMesh = tetMeshShape.GetMesh();
    int const numBoundaryFaces = tetMesh->GetNumBoundaryFaces();
    auto const faces = tetMesh->GetBoundaryFaces();
    for (int i = 0; i < numBoundaryFaces; ++i) {
      if (activeElementIndices.count(faces[i].element) == 1) {
        activeBoundaryFaceIndices.emplace_back(i);
      }
    }

    return CActiveBoundaryFaces(
        tetMesh, activeBoundaryFaceIndices, MakeConstSpan(boundaryDiscr.femElements));
  });
}

static std::vector<int> SelectActiveBoundaryFaceIndices(
    BoundarySubsamplingParams const& params,
    TriangularMesh const& triMesh) {
  MOCHI_ASSERT(
      params.subsamplingDensity >= 0_r && params.subsamplingDensity <= 1_r,
      "Invalid subsampling density.");

  int const numBoundaryFaces = triMesh.GetNumElements();
  int const numActiveBoundaryFaces =
      static_cast<int>(Ceil(numBoundaryFaces * params.subsamplingDensity));
  std::vector<int> activeBoundaryFaceInds;
  activeBoundaryFaceInds.reserve(numActiveBoundaryFaces);

  if (params.strategy == BoundarySubsamplingStrategy::AreaProportional) {
    // Select active boundary faces using A-Res method for Weighted Reservoir Sampling.
    constexpr unsigned int kSeed = 21;
    auto generator = RandomGenerator(kSeed);

    std::vector<std::pair<double, int>> weightedIndices;
    weightedIndices.reserve(numBoundaryFaces);
    for (int i = 0; i < numBoundaryFaces; i++) {
      // Select faces with the highest r_i = u_i^(1/w_i), where u_i is drawn from U([0,1]) and w_i
      // is the face area. This transformation ensures the selection probability is proportional to
      // the face area.
      // Reference: Weighted Random Sampling with a Reservoir, Efraimidis and Spirakis (2006)
      // https://doi.org/10.1016/j.ipl.2005.11.003
      double log_r =
          std::log(RandomUniformValue(generator, 0.0, 1.0)) / triMesh.GetElementMeasure(i);
      weightedIndices.emplace_back(log_r, i);
    }

    // Sort by weighted values and extract the top indices.
    // Higher "r" values correspond to faces that should be selected.
    std::partial_sort(
        weightedIndices.begin(),
        weightedIndices.begin() + numActiveBoundaryFaces,
        weightedIndices.end(),
        [](auto const& a, auto const& b) { return a.first > b.first; });

    for (int i = 0; i < numActiveBoundaryFaces; i++) {
      activeBoundaryFaceInds.push_back(weightedIndices[i].second);
    }

  } else if (params.strategy == BoundarySubsamplingStrategy::UniformProbability) {
    for (int i = 0; i < numActiveBoundaryFaces; ++i) {
      activeBoundaryFaceInds.push_back(
          static_cast<int>(int64_t(i) * numBoundaryFaces / numActiveBoundaryFaces));
    }
  } else {
    MOCHI_ASSERT(false, "Unsupported boundary subsampling strategy.");
  }
  static_assert(
      static_cast<int>(BoundarySubsamplingStrategy::Count) == 2,
      "Please update the if statement above if BoundarySubsamplingStrategy enumerator changes");

  return activeBoundaryFaceInds;
}

CActiveBoundaryFaces mochi::CreateActiveBoundaryFaces(
    BoundarySubsamplingParams const& params,
    TetrahedralMeshShape const& tetMeshShape,
    CFemBoundaryDiscretization const& boundaryDiscrVariant) {
  return boundaryDiscrVariant.Visit([&](auto const& boundaryDiscr) -> auto {
    auto const& boundaryMesh = *tetMeshShape.GetMesh()->GetBoundaryMesh();
    MOCHI_ASSERT(
        boundaryMesh.GetElementConnectivity().size() == boundaryDiscr.femElements.size(),
        "Inconsistent tetrahedral mesh and boundary discretization.");
    auto activeBoundaryFaceInds = SelectActiveBoundaryFaceIndices(params, boundaryMesh);
    return CActiveBoundaryFaces(
        tetMeshShape.GetMesh(), activeBoundaryFaceInds, MakeConstSpan(boundaryDiscr.femElements));
  });
}

CActiveBoundaryFaces mochi::CreateActiveBoundaryFaces(
    BoundarySubsamplingParams const& params,
    std::shared_ptr<TriangularMesh const> const& triMesh,
    CFemSurfaceDiscretization const& surfaceDiscrVariant) {
  return surfaceDiscrVariant.Visit([&](auto const& surfaceDiscr) -> auto {
    MOCHI_ASSERT(
        triMesh->GetElementConnectivity().size() == surfaceDiscr.femElements.size(),
        "Inconsistent triangular mesh and surface discretization.");
    auto activeBoundaryFaceInds = SelectActiveBoundaryFaceIndices(params, *triMesh);
    return CActiveBoundaryFaces(
        triMesh, activeBoundaryFaceInds, MakeConstSpan(surfaceDiscr.femElements));
  });
}
