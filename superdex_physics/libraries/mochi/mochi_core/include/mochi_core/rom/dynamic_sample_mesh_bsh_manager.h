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

#include <mochi_core/contact/contact_samples_bsh.h>
#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/rom/rom_hyper_reduction_params.h>
#include <mochi_core/utils/dynamic_array.h>

#include <limits>

namespace mochi::rom::hyper {

/*
BSH dynamic sample mesh manager. The current implementation assumes the relative position of the
sample points may change over time, e.g. the underlying actor is soft. For rigid actors, the
implementation could be optimized further.
*/
struct DynamicSampleMeshBshManager {
  // Distance of a sample point with respect to the closest collider at a given time.
  struct SampleData {
    // Time at which the sample was taken [s].
    double time = 0.0;

    // Position the sample was taken at [m].
    Real3 position = {};

    // Distance to the closest collider at the time that the sample was taken [m].
    real distance = -std::numeric_limits<real>::infinity();

    // Integral of max velocity over time since the last sample [m].
    real velocityPenalty = 0_r;
  };

  DynamicSampleMeshBshManager() = delete;
  MOCHI_DECLARE_MOVE_ONLY(DynamicSampleMeshBshManager);

  DynamicSampleMeshBshManager(
      ContactSamplesBsh&& bsh,
      SdfLowerBoundAnchorSelection const& anchorSelectionMode);

  // Update BSH radii and SDF lower bounds using the SDF measurements for this time step.
  // NOTE: The current sample distances (currentSampleDistances) are only up-to-date where finite.
  // A distance of +infinity signals a missing measurement and leaves the previous lower-bound
  // anchor in place.
  void Update(
      Span<Real3 const> currentSamplePositions,
      Span<int const> activeSampleIndices,
      Span<real const> currentSampleDistances,
      double currentTime,
      real maxVelocitySinceLastUpdate,
      real maxDistance);

  // Compute active boundary faces.
  void ComputeActiveBoundaryFaces(int maxActiveSamples, real sampleActivationThreshold);

  // Evaluate the SDF lower bound between a given node and a given position.
  real EvaluateSdfLowerBound(int nodeIdx, Real3 const& position) const;

  // Active boundary face indices.
  Span<int const> ActiveBoundaryFaceIndices() const {
    return _activeBoundaryFaceIndices;
  }

  // Extra weights of the active boundary faces. The size is equal to the number of active boundary
  // faces. ActiveBoundaryFaceWeightMultipliers()[i] = X means the weight of the i-th active
  // boundary face should be X times larger than without hyper-reduction.
  Span<real const> ActiveBoundaryFaceWeightMultipliers() const {
    return _activeBoundaryFaceWeightMultipliers;
  }

  // Last data collected for a given sample point.
  SampleData const& LastSampleData(int sampleIdx) const {
    return _lastSamplesData[sampleIdx];
  }

  auto const& GetBsh() const {
    return _bsh;
  }

 private:
  ContactSamplesBsh _bsh;
  SdfLowerBoundAnchorSelection _anchorSelectionMode;
  double _lastUpdateTime = 0.0;
  DynamicArray<SampleData> _lastSamplesData{};
  DynamicArray<int> _activeBoundaryFaceIndices{};
  DynamicArray<real> _activeBoundaryFaceWeightMultipliers{};
  DynamicArray<int> _traversalBuffer{};
};

} // namespace mochi::rom::hyper
