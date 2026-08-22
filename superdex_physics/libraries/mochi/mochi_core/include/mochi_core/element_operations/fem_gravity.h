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

#include <mochi_core/element_operations/batched_element_utils.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

namespace mochi::fem {

/// @brief Compute gravity work for a batch of elements.
///
/// @param[in] elementIndices  Indices into @p elements for each batch lane.
/// @param[in] elements  Element data array.
/// @param[in] disp  Batched displacements. Only the first ElementT::kNumDofs * ElementT::kSpaceDim
/// entries are read, so it may be a prefix span into a larger element vector.
/// @param[in,out] outEnergy  If non-null, accumulates per-element energy.
/// @param[in,out] outRes  If non-empty, accumulates per-element residual into the first
/// ElementT::kNumDofs * ElementT::kSpaceDim entries.
/// @param[in] gravity  Gravity acceleration vector [m/s²].
/// @param[in] density  Material density [kg/m³].
/// @param[in] perElementExtraWeight  Optional per-element quadrature weight multiplier.
/// @return true if outputs were written.
///
/// @note This is a spatial-only operation. It uses kNumFields == kSpaceDim implicitly. For element
/// types with extra DoFs per node (e.g., rod twist), use dedicated gravity functions.
template <int kBatchSize, class ElementT>
bool GravityWork(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<ElementT const> elements,
    Span<BatchReal<kBatchSize> const> disp,
    BatchDouble<kBatchSize>* outEnergy,
    Span<BatchReal<kBatchSize>> outRes,
    Real3 gravity,
    real density,
    Span<real const> perElementExtraWeight = {}) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  MOCHI_ASSERT_VERBOSE(!elements.empty(), "Elements span is empty.");
  MOCHI_ASSERT_VERBOSE(
      Min(elementIndices) >= 0 && Max(elementIndices) < isize(elements),
      "Element index out of range.");
  MOCHI_ASSERT_VERBOSE(
      perElementExtraWeight.empty() || (perElementExtraWeight.size() == elements.size()),
      "Inconsistent sizes.");

  constexpr auto kSpaceDim = ElementT::kSpaceDim;
  constexpr auto kNumNodes = ElementT::kNumDofs;
  constexpr auto kNumQuads = ElementT::kNumQuadPoints;
  static_assert(kSpaceDim == 3, "Only 3D elements are supported");
  static_assert(kNumNodes == 3 || kNumNodes == 4);
  MOCHI_ASSERT_VERBOSE(disp.size() >= kNumNodes * kSpaceDim, "Displacement span is too small.");
  MOCHI_ASSERT_VERBOSE(
      outRes.empty() || outRes.size() >= kNumNodes * kSpaceDim, "Residual span is too small.");

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = !outRes.empty();
  if (!evalObj && !evalRes) {
    return false;
  }

  auto const extraWeight = ComputeExtraWeights(elementIndices, perElementExtraWeight);

  V3 const gd = {V{gravity[0] * density}, V{gravity[1] * density}, V{gravity[2] * density}};

  alignas(alignof(V)) real staging[V::kSize]{};

  for (int q = 0; q < kNumQuads; ++q) {
    // NOTE: When evalObj is true, basis, quad-weight, and map-position gathers could be fused into
    // one per-lane pass. Measured slightly faster in benchmarks but not enough to justify the
    // additional complexity.
    NdArray<V, kNumNodes> basis MOCHI_NO_INIT;
    V quadWeight MOCHI_NO_INIT;
    PackBasisAndQuadWeight(elementIndices, elements, extraWeight, q, basis, quadWeight);

    if (evalObj) {
      // y_q = map_q + Σ_f N_f(q) * u_f  (interpolate displacement to quad point)
      V3 y = {};
      for (int f = 0; f < kNumNodes; ++f) {
        y += basis[f] * V3{disp[f * kSpaceDim], disp[f * kSpaceDim + 1], disp[f * kSpaceDim + 2]};
      }

      // Add map_q (reference position at quad point).
      // NOTE: The reference position is a constant offset and could be excluded from the objective.
      for (int d = 0; d < kSpaceDim; ++d) {
        for (int b = 0; b < kBatchSize; ++b) {
          staging[b] = elements[elementIndices[b]].mapEvaluated[q][d];
        }
        y[d] += Load<V>(staging);
      }

      // merit -= dot(y, gravity*density) * w_q
      *outEnergy -= StaticCast<Vd>(Dot(y, gd) * quadWeight);
    }

    if (evalRes) {
      // res[f] -= gravity*density * N_f(q) * w_q
      for (int f = 0; f < kNumNodes; ++f) {
        V const basisScaled = basis[f] * quadWeight;
        outRes[f * kSpaceDim + 0] -= gd[0] * basisScaled;
        outRes[f * kSpaceDim + 1] -= gd[1] * basisScaled;
        outRes[f * kSpaceDim + 2] -= gd[2] * basisScaled;
      }
    }
  }

  return true;
}

template <int kBatchSize, class ElementT>
MOCHI_FORCE_INLINE bool GravityWork(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<ElementT const> elements,
    BatchElementVector<kBatchSize, ElementT> const& disp,
    BatchDouble<kBatchSize>* outEnergy,
    BatchElementVector<kBatchSize, ElementT>* outRes,
    Real3 gravity,
    real density,
    Span<real const> perElementExtraWeight = {}) {
  return GravityWork<kBatchSize, ElementT>(
      elementIndices,
      elements,
      MakeConstSpan(disp),
      outEnergy,
      outRes ? MakeSpan(*outRes) : Span<BatchReal<kBatchSize>>{},
      gravity,
      density,
      perElementExtraWeight);
}

} // namespace mochi::fem
