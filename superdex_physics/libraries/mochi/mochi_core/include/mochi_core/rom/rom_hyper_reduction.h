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

#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/rom/rom_hyper_reduction_params.h>
#include <mochi_core/utils/dskinning.h>
#include <mochi_core/utils/subset_map.h>

#include <memory>
#include <vector>

namespace mochi::rom::hyper {

struct UnweightedSampleMeshData {
  std::unique_ptr<TetrahedralMesh> mesh;
  SubsetMap nodeSubset;
  SubsetMap elementSubset;
  SubsetMap boundaryElementSubset;
};

struct SampleMeshWeighting {
  std::vector<real> volumeElements;
  std::vector<real> boundaryFaceElements;
};

struct SampleMeshData : public UnweightedSampleMeshData {
  SampleMeshWeighting weighting;
};

UnweightedSampleMeshData CreateUnweightedSampleMeshFromVolumeElements(
    TetrahedralMesh const& fullMesh,
    SubsetMap const& elements);

struct SampleMeshSubsetResult {
  SubsetMap fullInterior;
  SubsetMap fullBoundary;
  SubsetMap sampleInterior;
  SubsetMap sampleBoundary;
  SubsetMap sampleElements;
};

SampleMeshSubsetResult SampleMeshSubset(
    TetrahedralMesh const& fullMesh,
    BoundaryAndInternalElementsSubsamplingParameters subsamplingParams,
    Error& error);

SampleMeshSubsetResult SampleMeshSubset(
    TetrahedralMesh const& fullMesh,
    std::vector<int> const& volumeElements,
    Error& error);

SampleMeshWeighting GenerateSampleMeshWeights(
    TetrahedralMesh const& fullMesh,
    SampleMeshSubsetResult const& generationResult,
    UnweightedSampleMeshData const& sampleMesh,
    bool scaleBoundaryAndInternalVolumeElementsSeparately = true);

SampleMeshData CreateSampleMeshAndWeights(
    TetrahedralMesh const& fullMesh,
    BoundaryAndInternalElementsSubsamplingParameters subsamplingParams,
    Error& error);

} // namespace mochi::rom::hyper
