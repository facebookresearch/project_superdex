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

#include <mochi_core/rom/rom_hyper_reduction.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/subset_map.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::rom;
using namespace mochi::rom::hyper;

static void Verify(SampleMeshData const& sampleMesh) {
  MOCHI_ASSERT(
      sampleMesh.weighting.boundaryFaceElements.size() ==
      sampleMesh.boundaryElementSubset.GetSubsetSize());
  MOCHI_ASSERT(
      sampleMesh.weighting.volumeElements.size() == sampleMesh.elementSubset.GetSubsetSize());
  MOCHI_ASSERT(sampleMesh.mesh->GetNumNodes() == sampleMesh.nodeSubset.GetSubsetSize());
  MOCHI_ASSERT(sampleMesh.mesh->GetNumVolumes() == sampleMesh.elementSubset.GetSubsetSize());
  MOCHI_ASSERT(sampleMesh.mesh->GetNumFaces() == sampleMesh.boundaryElementSubset.GetSubsetSize());
}

SampleMeshSubsetResult mochi::rom::hyper::SampleMeshSubset(
    TetrahedralMesh const& fullMesh,
    BoundaryAndInternalElementsSubsamplingParameters subsamplingParams,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  int const stepSizeForBoundaryElementsSelection =
      subsamplingParams.stepSizeForBoundaryElementsSelection;
  int const stepSizeForInteriorElementsSelection =
      subsamplingParams.stepSizeForInteriorElementsSelection;

  //-----------------------------------------------------------------------------
  // preconditions
  //-----------------------------------------------------------------------------

  // stepSizeForElementsSelection = kNoElements, no elements are chosen
  // stepSizeForElementsSelection = kAllElements, all elements are chosen
  // stepSizeForElementsSelection > 0, some elements are chosen
  MOCHI_ERROR_IF(
      stepSizeForInteriorElementsSelection != kNoElements &&
          stepSizeForInteriorElementsSelection < kAllElements,
      error,
      "Invalid interior hyper-reduction settings");
  MOCHI_ERROR_IF(
      stepSizeForBoundaryElementsSelection != kNoElements &&
          stepSizeForBoundaryElementsSelection < kAllElements,
      error,
      "Invalid boundary hyper-reduction settings");
  MOCHI_ERROR_IF(
      stepSizeForBoundaryElementsSelection == kNoElements &&
          stepSizeForInteriorElementsSelection == kNoElements,
      error,
      "Must sample the boundary and/or the interior with hyper-reduction");
  MOCHI_ERROR_RETURN(error, {});

  // Compute the subset of boundary elements
  SubsetMap fullBoundary = [&fullMesh]() {
    std::vector<int> result;
    auto bdFaces = fullMesh.GetBoundaryFaces();
    for (auto bdFace : bdFaces) {
      result.push_back(bdFace.element);
    }
    // This function removes duplicates
    return SubsetMap::FromUnsortedList(result, fullMesh.GetNumElements());
  }();

  // Sample the boundary
  SubsetMap sampleBoundary = stepSizeForBoundaryElementsSelection == kNoElements
      ? fullBoundary.EmptySubset()
      : fullBoundary.EveryNth(stepSizeForBoundaryElementsSelection);

  // Sample the interior elements from the remaining elements
  SubsetMap fullInterior = fullBoundary.Complement();
  SubsetMap sampleInterior = stepSizeForInteriorElementsSelection == kNoElements
      ? fullInterior.EmptySubset()
      : fullInterior.EveryNth(stepSizeForInteriorElementsSelection);

  // All sample elements are the union of the interior and boundary samples
  SubsetMap sampleElements = Union(sampleInterior, sampleBoundary);

  return SampleMeshSubsetResult{
      .fullInterior = std::move(fullInterior),
      .fullBoundary = std::move(fullBoundary),
      .sampleInterior = std::move(sampleInterior),
      .sampleBoundary = std::move(sampleBoundary),
      .sampleElements = std::move(sampleElements)};
}

SampleMeshSubsetResult mochi::rom::hyper::SampleMeshSubset(
    TetrahedralMesh const& fullMesh,
    std::vector<int> const& volumeElements,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Compute the subset of boundary elements
  SubsetMap fullBoundary = [&fullMesh]() {
    std::vector<int> result;
    auto bdFaces = fullMesh.GetBoundaryFaces();
    for (auto bdFace : bdFaces) {
      result.push_back(bdFace.element);
    }
    // This function removes duplicates
    int const numElem = fullMesh.GetNumElements();
    return SubsetMap::FromUnsortedList(result, numElem);
  }();
  SubsetMap fullInterior = fullBoundary.Complement();

  // figure out which ones are vol elements near the boundary
  std::vector<int> bdVolElems;
  std::vector<int> interiorVolElems;
  for (int e : volumeElements) {
    bool const isBd = fullBoundary.Contains(e);
    bool const isInterior = fullInterior.Contains(e);
    // in theory, because of the code above, an element is
    // either near boundary or an interior one, not both
    MOCHI_ASSERT((isBd && !isInterior) || (!isBd && isInterior));

    if (fullBoundary.Contains(e)) {
      bdVolElems.push_back(e);
    } else {
      interiorVolElems.push_back(e);
    }
  }

  SubsetMap sampleBoundary = SubsetMap::FromUnsortedList(bdVolElems, fullMesh.GetNumElements());
  SubsetMap sampleInterior =
      SubsetMap::FromUnsortedList(interiorVolElems, fullMesh.GetNumElements());

  // All sample elements are the union of the interior and boundary samples
  SubsetMap sampleElements = Union(sampleInterior, sampleBoundary);

  return SampleMeshSubsetResult{
      .fullInterior = std::move(fullInterior),
      .fullBoundary = std::move(fullBoundary),
      .sampleInterior = std::move(sampleInterior),
      .sampleBoundary = std::move(sampleBoundary),
      .sampleElements = std::move(sampleElements)};
}

SampleMeshWeighting mochi::rom::hyper::GenerateSampleMeshWeights(
    TetrahedralMesh const& fullMesh,
    SampleMeshSubsetResult const& generationResult,
    UnweightedSampleMeshData const& sampleMesh,
    bool scaleBoundaryAndInternalVolumeElementsSeparately) {
  // Compute weights
  std::vector<real> volumeMeasures = [&fullMesh]() {
    std::vector<real> result;
    result.reserve(fullMesh.GetNumElements());
    for (int i = 0; i < fullMesh.GetNumElements(); ++i) {
      result.emplace_back(fullMesh.GetElementMeasure(i));
    }
    return result;
  }();

  real internalReweightFactor = {};
  real boundaryReweightFactor = {};

  if (scaleBoundaryAndInternalVolumeElementsSeparately) {
    // Computes a reweight factor by dividing the weight on the full mesh ids by the weight on
    // the sample mesh ids.
    auto computeReweightFactor = [](std::vector<real> const& allMeasures,
                                    std::vector<int> const& fullMeshIds,
                                    std::vector<int> const& sampleMeshIds) {
      if (!sampleMeshIds.empty()) {
        std::vector<real> fullWeights = Extract<real, int>(allMeasures, fullMeshIds);
        std::vector<real> sampleWeights = Extract<real, int>(allMeasures, sampleMeshIds);

        real const fullWeight = std::accumulate(fullWeights.begin(), fullWeights.end(), 0_r);
        real const sampleWeight = std::accumulate(sampleWeights.begin(), sampleWeights.end(), 0_r);

        return fullWeight / sampleWeight;
      } else {
        return 0_r;
      }
    };

    internalReweightFactor = computeReweightFactor(
        volumeMeasures,
        generationResult.fullInterior.GetStorage(),
        generationResult.sampleInterior.GetStorage());
    boundaryReweightFactor = computeReweightFactor(
        volumeMeasures,
        generationResult.fullBoundary.GetStorage(),
        generationResult.sampleBoundary.GetStorage());
  } else {
    auto sampleWeightsInnElem =
        Extract<real, int>(volumeMeasures, generationResult.sampleInterior.GetStorage());
    auto sampleWeightsBdElem =
        Extract<real, int>(volumeMeasures, generationResult.sampleBoundary.GetStorage());
    std::vector<real> activeWeights = sampleWeightsInnElem;
    for (real w : sampleWeightsBdElem) {
      activeWeights.push_back(w);
    }
    real const activeSum = std::accumulate(activeWeights.begin(), activeWeights.end(), 0_r);

    real const fullWeight = std::accumulate(volumeMeasures.begin(), volumeMeasures.end(), 0_r);
    real const reweightFactor = fullWeight / activeSum;

    internalReweightFactor = reweightFactor;
    boundaryReweightFactor = reweightFactor;
  }

  // Compute weights for subsampled elements based on which element subset they fall into
  std::vector<real> volElementWeights =
      [&generationResult, &internalReweightFactor, &boundaryReweightFactor]() {
        std::vector<real> result;
        result.reserve(generationResult.sampleElements.GetSubsetSize());
        std::transform(
            generationResult.sampleElements.begin(),
            generationResult.sampleElements.end(),
            std::back_inserter(result),
            [&](int idx) {
              if (generationResult.sampleInterior.Contains(idx)) {
                return internalReweightFactor;
              } else if (generationResult.sampleBoundary.Contains(idx)) {
                return boundaryReweightFactor;
              } else {
                MOCHI_ASSERT(false, "Unexpected index.");
                return 0_r;
              }
            });
        return result;
      }();

  // Get weights of all surfaces
  std::vector<real> surfaceFaceMeasures = [&fullMesh]() {
    std::vector<real> result(fullMesh.GetBoundaryMesh()->GetNumElements());
    for (int i = 0; i < result.size(); ++i) {
      result[i] = fullMesh.GetBoundaryMesh()->GetElementMeasure(i);
    }
    return result;
  }();

  // Compute reweighting factor for the face elements
  real faceReweightFactor = [&surfaceFaceMeasures, &samples = sampleMesh.boundaryElementSubset]() {
    real fullSum = std::accumulate(surfaceFaceMeasures.begin(), surfaceFaceMeasures.end(), 0_r);
    auto weightSubset = samples.Extract<real>(surfaceFaceMeasures);
    real partialSum = std::accumulate(weightSubset.begin(), weightSubset.end(), 0_r);
    return fullSum / partialSum;
  }();

  std::vector<real> faceWeights(sampleMesh.boundaryElementSubset.GetSubsetSize());
  std::fill(faceWeights.begin(), faceWeights.end(), faceReweightFactor);

  return SampleMeshWeighting{
      .volumeElements = std::move(volElementWeights),
      .boundaryFaceElements = std::move(faceWeights)};
}

namespace {
struct SampleMeshTetMeshResult {
  std::unique_ptr<TetrahedralMesh> mesh;
  SubsetMap nodeSubset;
};

SampleMeshTetMeshResult CreateSampleMeshTetMesh(
    TetrahedralMesh const& fullMesh,
    SubsetMap const& volumeElements,
    SubsetMap const& boundaryElements) {
  MOCHI_PROFILE_SCOPE();

  // Compute node subset
  // This constitutes all nodes touched by the volume elements
  SubsetMap nodeSubset = [&volumeElements, &fullMesh]() {
    MOCHI_PROFILE_SCOPE_N("Compute Node Subset");

    std::unordered_set<int> nodes;

    auto connectivity = fullMesh.GetElementConnectivity();
    for (auto idx : volumeElements) {
      for (auto node : connectivity[idx]) {
        nodes.emplace(node);
      }
    }

    return SubsetMap::FromUnsortedList(nodes, fullMesh.GetNumNodes());
  }();

  // Extract new coordinates (do no require relabelling)
  std::vector<Real3> newCoords = nodeSubset.Extract(fullMesh.GetNodeCoordinates());

  std::vector<Int4> newConnectivity = [&fullMesh, &nodeSubset, &volumeElements]() {
    MOCHI_PROFILE_SCOPE_N("Compute New Connectivity");

    // Extract element connectivity from full mesh
    auto result = volumeElements.Extract(fullMesh.GetElementConnectivity());
    auto newConnectivityFlat = Flatten(MakeSpan(result));

    // Relabel tetrahedral connectivity with new node indices
    nodeSubset.GetSubsetIndicesFromFullIndices(
        newConnectivityFlat.begin(),
        newConnectivityFlat.end(),
        newConnectivityFlat.begin(),
        ErrorAssert{});
    return result;
  }();

  // Extract and relabel boundary face connectivity
  std::vector<Int3> boundaryFacesConnectivity = [&fullMesh, &boundaryElements, &nodeSubset]() {
    MOCHI_PROFILE_SCOPE_N("Compute Boundary Face Connectivity");

    auto result = boundaryElements.Extract(fullMesh.GetBoundaryFacesConnectivity());

    // Relabel boundary faces with new node indices
    auto newConnectivityFlat = Flatten(MakeSpan(result));
    nodeSubset.GetSubsetIndicesFromFullIndices(
        newConnectivityFlat.begin(),
        newConnectivityFlat.end(),
        newConnectivityFlat.begin(),
        ErrorAssert{});

    return result;
  }();

  // Compute new face information of above boundary face subset
  std::vector<TetrahedralMesh::BoundaryFaceInfo> newFaceInfo =
      [&fullMesh, &boundaryElements, &volumeElements]() {
        MOCHI_PROFILE_SCOPE_N("Compute New Face Info");

        auto newFaces = boundaryElements.Extract(fullMesh.GetBoundaryFaces());

        // Relabel face elements with new node indices
        std::for_each(newFaces.begin(), newFaces.end(), [&volumeElements](auto& face) {
          auto element = volumeElements.GetSubsetIndexFromFullIndex(face.element);
          MOCHI_ASSERT(element);
          face.element = *element;
        });

        return newFaces;
      }();

  return {
      .mesh = std::make_unique<TetrahedralMesh>(
          newCoords, newConnectivity, &newFaceInfo, &boundaryFacesConnectivity),
      .nodeSubset = std::move(nodeSubset)};
}
} // namespace

UnweightedSampleMeshData mochi::rom::hyper::CreateUnweightedSampleMeshFromVolumeElements(
    TetrahedralMesh const& fullMesh,
    SubsetMap const& elements) {
  MOCHI_ASSERT(fullMesh.GetBoundaryFaces());
  MOCHI_ASSERT(fullMesh.GetBoundaryFacesConnectivity());
  MOCHI_ASSERT(elements.GetFullSetSize() == fullMesh.GetElementConnectivity().size());

  // Compute boundary faces subset (i.e., faces that touch an element in newElements)
  SubsetMap boundaryFacesSubset = [&fullMesh, &elements]() {
    std::vector<int> result;
    auto oldFaces = fullMesh.GetBoundaryFaces();
    for (int faceIdx = 0; faceIdx < oldFaces.size(); ++faceIdx) {
      auto const& face = oldFaces[faceIdx];
      if (elements.Contains(face.element)) {
        result.emplace_back(faceIdx);
      }
    }
    return SubsetMap::FromUnsortedList(result, isize(oldFaces));
  }();

  auto [mesh, nodeSubset] = CreateSampleMeshTetMesh(fullMesh, elements, boundaryFacesSubset);
  return UnweightedSampleMeshData{
      .mesh = std::move(mesh),
      .nodeSubset = std::move(nodeSubset),
      .elementSubset = elements,
      .boundaryElementSubset = std::move(boundaryFacesSubset)};
}

SampleMeshData mochi::rom::hyper::CreateSampleMeshAndWeights(
    TetrahedralMesh const& fullMesh,
    BoundaryAndInternalElementsSubsamplingParameters subsamplingParams,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  auto meshSubset = SampleMeshSubset(fullMesh, subsamplingParams, error);
  MOCHI_ERROR_RETURN(error, {});
  auto sampleMeshUnweighted =
      CreateUnweightedSampleMeshFromVolumeElements(fullMesh, meshSubset.sampleElements);
  auto sampleMeshWeights = GenerateSampleMeshWeights(fullMesh, meshSubset, sampleMeshUnweighted);

  auto result = SampleMeshData{std::move(sampleMeshUnweighted), std::move(sampleMeshWeights)};
  Verify(result);
  return result;
}
