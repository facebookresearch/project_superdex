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

#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

namespace mochi::fem {

// Batched element vector (displacement, residual).
template <int kBatchSize, class ElementT, int kNumFields = ElementT::kSpaceDim>
using BatchElementVector = NdArray<BatchReal<kBatchSize>, ElementT::kNumDofs * kNumFields>;

// Batched element matrix (dresidual).
template <int kBatchSize, class ElementT, int kNumFields = ElementT::kSpaceDim>
using BatchElementMatrix = NdArray<
    BatchReal<kBatchSize>,
    ElementT::kNumDofs * kNumFields * ElementT::kNumDofs * kNumFields>;

// Compute per-element extra weights.
template <size_t kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<real, kBatchSize> ComputeExtraWeights(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<real const> perElementExtraWeight) {
  NdArray<real, kBatchSize> extraWeight MOCHI_NO_INIT;
  for (size_t b = 0; b < kBatchSize; ++b) {
    extraWeight[b] = perElementExtraWeight.empty() ? 1_r : perElementExtraWeight[elementIndices[b]];
  }
  return extraWeight;
}

// Pack basis functions and quadrature weights.
template <class ElementT, size_t kBatchSize>
MOCHI_FORCE_INLINE void PackBasisAndQuadWeight(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<ElementT const> elements,
    NdArray<real, kBatchSize> const& extraWeight,
    int q,
    NdArray<BatchReal<kBatchSize>, ElementT::kNumDofs>& batchedBasis,
    BatchReal<kBatchSize>& batchedQuadWeight) {
  using V = BatchReal<kBatchSize>;
  constexpr int kNumNodes = ElementT::kNumDofs;
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int f = 0; f < kNumNodes; ++f) {
    for (size_t b = 0; b < kBatchSize; ++b) {
      staging[b] = elements[elementIndices[b]].basisEvaluated[q][f];
    }
    batchedBasis[f] = Load<V>(staging);
  }
  for (size_t b = 0; b < kBatchSize; ++b) {
    staging[b] = elements[elementIndices[b]].quadWeights[q] * extraWeight[b];
  }
  batchedQuadWeight = Load<V>(staging);
}

/// @brief Gather the stage-start target for a batch of elements: the inertia predictor u₀ + Δt·v₀,
///   or the stage-start displacement u₀ only when @p kGatherVelocity is false.
///
/// @tparam kBatchSize Batch size.
/// @tparam ElementT Element type.
/// @tparam kGatherVelocity If true, add Δt·v₀. If false, gather u₀ only; @p predVeloRaw and
///   @p dtStage are then unused and @p predVeloRaw may be empty.
/// @tparam kNumFields Number of fields per node. Defaults to ElementT::kSpaceDim.
/// @tparam kStride Stride (in DoFs) between consecutive elements in @p indicesFlat.
///   Defaults to ElementT::kNumDofs * kNumFields (matching a uniform-stride L2G).
///   Set to a larger value when gathering a sub-stencil from a padded L2G with wider stride
///   (e.g., 3-node triangle inertia from a 6-node shell bending stencil with stride 18).
template <
    int kBatchSize,
    class ElementT,
    bool kGatherVelocity = true,
    int kNumFields = ElementT::kSpaceDim,
    int kStride = ElementT::kNumDofs * kNumFields>
MOCHI_FORCE_INLINE void GatherPredTarget(
    Span<int const> indicesFlat,
    NdArray<int, kBatchSize> const& elementIndices,
    Span<real const> predDispRaw,
    [[maybe_unused]] Span<real const> predVeloRaw,
    [[maybe_unused]] real dtStage,
    BatchElementVector<kBatchSize, ElementT, kNumFields>& outPredTarget) {
  static_assert(ElementT::kSpaceDim == 3, "GatherPredTarget requires kSpaceDim == 3");
  static_assert(
      kNumFields == ElementT::kSpaceDim, "GatherPredTarget requires kNumFields == kSpaceDim");
  static_assert(kStride >= ElementT::kNumDofs * kNumFields, "Stride must be >= element DoF count.");
  using V = BatchReal<kBatchSize>;
  constexpr int kNumNodes = ElementT::kNumDofs;
  alignas(alignof(V)) real nodeStaging[V::kSize * 3] MOCHI_NO_INIT;
  for (int n = 0; n < kNumNodes; ++n) {
    for (int b = 0; b < V::kSize; ++b) {
      int const globalDof =
          indicesFlat[elementIndices[Min(b, kBatchSize - 1)] * kStride + n * kNumFields];
      Vec4r nodeTarget = Load<3, Vec4r>(&predDispRaw[globalDof]);
      if constexpr (kGatherVelocity) {
        nodeTarget += Vec4r{dtStage} * Load<3, Vec4r>(&predVeloRaw[globalDof]);
      }
      Store<3>(&nodeStaging[b * 3], nodeTarget);
    }
    LoadTransposed<V::kSize>(
        nodeStaging,
        outPredTarget[n * kNumFields + 0],
        outPredTarget[n * kNumFields + 1],
        outPredTarget[n * kNumFields + 2]);
  }
}

} // namespace mochi::fem
