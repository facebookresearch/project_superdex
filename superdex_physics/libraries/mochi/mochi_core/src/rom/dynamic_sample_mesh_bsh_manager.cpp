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

#include <mochi_core/rom/dynamic_sample_mesh_bsh_manager.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <cstdint>
#include <utility>

using namespace mochi;
using namespace mochi::rom::hyper;

real DynamicSampleMeshBshManager::EvaluateSdfLowerBound(int nodeIdx, Real3 const& position) const {
  MOCHI_ASSERT_VERBOSE(nodeIdx >= 0 && nodeIdx < isize(_bsh), "Invalid node index.");
  auto LowerBoundFromSample = [&](int sampleIdx) -> real {
    auto const& data = _lastSamplesData[sampleIdx];
    real lowerBound = data.distance - Norm(data.position - position) - data.velocityPenalty;
    MOCHI_ASSERT_VERBOSE(IsFinite(lowerBound) || (lowerBound == -kInf))
    return lowerBound;
  };

  // Using the anchor selection specified, combine lower bounds to get a tighter lower bound.
  int const nodeSampleIdx = _bsh.GetNodeData(nodeIdx).sampleIdx;
  real maxLowerBound = (nodeSampleIdx >= 0) ? LowerBoundFromSample(nodeSampleIdx) : -kInf;
  switch (_anchorSelectionMode) {
    case SdfLowerBoundAnchorSelection::Self: {
      return maxLowerBound;
    }
    case SdfLowerBoundAnchorSelection::Ancestor: {
      for (int ancestorIdx : _bsh.Ancestors(nodeIdx)) {
        int sampleIdx = _bsh.GetNodeData(ancestorIdx).sampleIdx;
        if (sampleIdx >= 0) {
          maxLowerBound = Max(maxLowerBound, LowerBoundFromSample(sampleIdx));
        }
      }
      return maxLowerBound;
    }
    case SdfLowerBoundAnchorSelection::AncestorSibling: {
      if (auto parentIdx = _bsh.Parent(nodeIdx)) {
        for (int siblingIdx : _bsh.Children(*parentIdx)) {
          int sampleIdx = _bsh.GetNodeData(siblingIdx).sampleIdx;
          if (sampleIdx >= 0) {
            maxLowerBound = Max(maxLowerBound, LowerBoundFromSample(sampleIdx));
          }
        }
        for (int ancestorIdx : _bsh.Ancestors(nodeIdx)) {
          int sampleIdx = _bsh.GetNodeData(ancestorIdx).sampleIdx;
          if (sampleIdx >= 0) {
            maxLowerBound = Max(maxLowerBound, LowerBoundFromSample(sampleIdx));
          }
        }
      }
      return maxLowerBound;
    }
    default: {
      MOCHI_ASSERT(false, "Unexpected SdfLowerBoundAnchorSelection.");
      return {};
    }
  }
}

// NOTE: The current sample distances (currentSampleDistances) are only up-to-date for the active
// samples. The distance of inactive samples is ignored.
void DynamicSampleMeshBshManager::Update(
    Span<Real3 const> currentSamplePositions,
    Span<int const> activeSampleIndices,
    Span<real const> currentSampleDistances,
    double currentTime,
    real maxVelocitySinceLastUpdate,
    real maxDistance) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(isize(currentSamplePositions) == _bsh.NumSamples());
  MOCHI_ASSERT_VERBOSE(isize(currentSampleDistances) == isize(currentSamplePositions));
  MOCHI_ASSERT_VERBOSE(maxVelocitySinceLastUpdate >= 0_r);
  MOCHI_ASSERT_VERBOSE(currentTime >= _lastUpdateTime);

  double const deltaTime = currentTime - _lastUpdateTime;
  _lastUpdateTime = currentTime;

  // Update BSH node spheres.
  _bsh.Update(currentSamplePositions);

  // Update the velocity penalties using the maximum velocity since the last update.
  for (auto& lastData : _lastSamplesData) {
    lastData.velocityPenalty += static_cast<real>(deltaTime * maxVelocitySinceLastUpdate);
  }

  // Update last samples data. A +infinity current distance means this sample did not receive an SDF
  // measurement in the current pass. Do not reinterpret "not reported" as "at least maxDistance
  // away". Keep the previous lower-bound anchor and let the accumulated motion penalties make it
  // more conservative.
  for (auto sampleIdx : activeSampleIndices) {
    MOCHI_ASSERT_VERBOSE(sampleIdx >= 0 && sampleIdx < _bsh.NumSamples());
    real const currentDistance = currentSampleDistances[sampleIdx];
    MOCHI_ASSERT_VERBOSE(IsFinite(currentDistance) || currentDistance == kInf);
    if (!IsFinite(currentDistance)) {
      continue;
    }

    auto& lastData = _lastSamplesData[sampleIdx];

    // Store time, position and distance.
    lastData.position = currentSamplePositions[sampleIdx];
    lastData.time = currentTime;
    lastData.distance = Min(maxDistance, currentDistance);

    // Reset the velocity penalty. The data is fresh.
    lastData.velocityPenalty = 0.0;
  }
}

void DynamicSampleMeshBshManager::ComputeActiveBoundaryFaces(
    int maxActiveSamples,
    real sampleActivationThreshold) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(maxActiveSamples >= 0, "Max number of samples must be positive.");
  maxActiveSamples = Min(maxActiveSamples, _bsh.NumSamples());

  _activeBoundaryFaceIndices.clear();
  _activeBoundaryFaceIndices.reserve(_bsh.NumSamples());

  // Perform breadth-first search of the BSH to find all samples whose SDF lower bound is under the
  // activation threshold.
  int bufferIdx = 0;
  _traversalBuffer.resize_noinit(_bsh.size());
  _traversalBuffer[bufferIdx++] = ContactSamplesBsh::RootIdx();
  while (bufferIdx > 0) {
    // NOTE: EvaluateSdfLowerBound could optionally terminate early if the lower bound is already
    // above the threshold "sampleActivationThreshold + data.radius". TBD whether it would be
    // faster.
    int bshNodeIdx = _traversalBuffer[--bufferIdx];
    auto const& data = _bsh.GetNodeData(bshNodeIdx);
    auto const sdfLowerBound = EvaluateSdfLowerBound(bshNodeIdx, data.position);
    if (data.sampleIdx >= 0 && sdfLowerBound <= sampleActivationThreshold) {
      int bdFaceIdx = data.sampleIdx; // TODO(T224856535): Assumes one sample per boundary face.
      _activeBoundaryFaceIndices.push_back(bdFaceIdx);
    }

    // Pruning optimization: Skip traversing children if the SDF lower bound for all points in the
    // sphere exceeds the activation threshold.
    if (sdfLowerBound - data.radius <= sampleActivationThreshold) {
      for (int childIdx : _bsh.Children(bshNodeIdx)) {
        _traversalBuffer[bufferIdx++] = childIdx;
      }
    }
  }

  real const extraWeight =
      static_cast<real>(Max(isize(_activeBoundaryFaceIndices), maxActiveSamples)) /
      Max(1, maxActiveSamples);
  if (isize(_activeBoundaryFaceIndices) > maxActiveSamples) {
    constexpr uint32_t kSeed = 2654435761; // Prime number with good bit distribution.
    RandomSubset(_activeBoundaryFaceIndices, maxActiveSamples, XorShift32Generator(kSeed));
  }

  _activeBoundaryFaceWeightMultipliers.clear();
  _activeBoundaryFaceWeightMultipliers.resize(_activeBoundaryFaceIndices.size(), extraWeight);
}

DynamicSampleMeshBshManager::DynamicSampleMeshBshManager(
    ContactSamplesBsh&& bsh,
    SdfLowerBoundAnchorSelection const& anchorSelectionMode)
    : _bsh(std::move(bsh)), _anchorSelectionMode(anchorSelectionMode) {
  MOCHI_ASSERT(_bsh.size() > 0, "BSH must not be empty.");
  _lastSamplesData.resize(_bsh.NumSamples());
}
