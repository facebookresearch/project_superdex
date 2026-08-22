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
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/nd_array.h>

namespace mochi::fem {

/// @brief Compute traction work for a batch of elements.
///
/// @param[in] elementIndices  Indices into @p elements for each batch lane.
/// @param[in] elements  Element data array.
/// @param[in,out] outEnergy  If non-null, accumulates per-element energy.
/// @param[in,out] outRes  If non-null, accumulates per-element residual.
/// @param[in,out] outDRes  If non-null, accumulates per-element dresidual.
/// @param[in] tractionFunc  Batched traction callback, invoked once per quadrature point
///   with the signature:
///   @code
///     void tractionFunc(
///         NdArray<int, kBatchSize> const& elementIndices,
///         int quadPointIndex,
///         // Non-null iff energy is requested.
///         BatchDouble<kBatchSize>* outEnergy,
///         // Non-null iff residual is requested. Values may be uninitialized at input.
///         BatchReal3<kBatchSize>* outForce,
///         // Non-null iff dresidual is requested. Values may be uninitialized at input.
///         NdArray<BatchReal3<kBatchSize>, 3>* outDForce,
///         // True for each batch point that produced force.
///         NdArray<bool, kBatchSize>& hasForce);
///   @endcode
///   The callback is templated so it can be inlined into the assembly loop. Pass it as a lambda or
///   functor.
/// @param[in] perElementExtraWeight  Optional per-element quadrature weight multiplier.
///
/// @return true if any element in the batch produced output.
///
/// @note Traction acts only on spatial components. If @p kNumFields includes extra non-spatial
/// fields, those residual entries and dresidual rows/columns are left unchanged.
template <int kBatchSize, class ElementT, int kNumFields, class BatchedTractionFn>
bool TractionWork(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<ElementT const> elements,
    BatchDouble<kBatchSize>* outEnergy,
    BatchElementVector<kBatchSize, ElementT, kNumFields>* outRes,
    BatchElementMatrix<kBatchSize, ElementT, kNumFields>* outDRes,
    BatchedTractionFn const& tractionFunc,
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
  constexpr auto kNumDofs = kNumNodes * kNumFields;
  static_assert(kSpaceDim == 3, "Only 3D elements are supported");
  static_assert(kNumFields >= kSpaceDim && kNumFields <= 4);

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = (outRes != nullptr);
  bool const evalDRes = (outDRes != nullptr);
  MOCHI_ASSERT_VERBOSE(evalObj || evalRes || evalDRes, "Must assemble something.");

  auto const extraWeight = ComputeExtraWeights(elementIndices, perElementExtraWeight);

  alignas(alignof(V)) real staging[V::kSize]{};
  bool hasAnyOutput = false;

  for (int q = 0; q < kNumQuads; ++q) {
    NdArray<V, kNumNodes> basis MOCHI_NO_INIT;
    V quadWeight MOCHI_NO_INIT;
    PackBasisAndQuadWeight(elementIndices, elements, extraWeight, q, basis, quadWeight);

    // Call the batched traction callback.
    Vd cbEnergy = Vd{0.0};
    V3 cbForce MOCHI_NO_INIT;
    NdArray<V3, kSpaceDim> cbDForce MOCHI_NO_INIT;
    NdArray<bool, kBatchSize> hasForce = {};

    tractionFunc(
        elementIndices,
        q,
        evalObj ? &cbEnergy : nullptr,
        evalRes ? &cbForce : nullptr,
        evalDRes ? &cbDForce : nullptr,
        hasForce);

    // Check if any element has output.
    bool anyHasForce = false;
    for (int b = 0; b < kBatchSize; ++b) {
      anyHasForce |= hasForce[b];
    }
    if (!anyHasForce) {
      continue;
    }
    hasAnyOutput = true;

    // Build a mask: zero out contributions for elements without force.
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = hasForce[b] ? 1_r : 0_r;
    }
    V const zero = V{0_r};
    V const mask = Load<V>(staging) > zero;

    // Energy accumulation.
    if (evalObj) {
      MOCHI_ASSERT_VERBOSE(IsFinite(cbEnergy), "Traction energy must be finite.");
      Vd const maskedQuadWeight = StaticCast<Vd>(Select(mask, quadWeight, zero));
      *outEnergy += cbEnergy * maskedQuadWeight;
    }

    // Residual: res[f*nFields+d] -= force[d] * N_f(q) * w_q
    // NOTE: basisScaled is computed separately in the evalRes and evalDRes blocks rather than
    // hoisted above based on benchmark results.
    if (evalRes) {
      V const maskedForce[kSpaceDim] = {
          Select(mask, cbForce[0], zero),
          Select(mask, cbForce[1], zero),
          Select(mask, cbForce[2], zero)};
      for (int f = 0; f < kNumNodes; ++f) {
        V const basisScaled = basis[f] * quadWeight;
        (*outRes)[f * kNumFields + 0] -= maskedForce[0] * basisScaled;
        (*outRes)[f * kNumFields + 1] -= maskedForce[1] * basisScaled;
        (*outRes)[f * kNumFields + 2] -= maskedForce[2] * basisScaled;
      }
    }

    // DRes: dres[(f*nFields+r)*nDof + (g*nFields+c)] -= dforce[r][c] * N_f(q)*w_q * N_g(q)
    if (evalDRes) {
      NdArray<V3, kSpaceDim> maskedDForce MOCHI_NO_INIT;
      for (int r = 0; r < kSpaceDim; ++r) {
        for (int c = 0; c < kSpaceDim; ++c) {
          maskedDForce[r][c] = Select(mask, cbDForce[r][c], zero);
        }
      }

      for (int f = 0; f < kNumNodes; ++f) {
        V const basisScaled = basis[f] * quadWeight;
        for (int r = 0; r < kSpaceDim; ++r) {
          V const rowCoef[kSpaceDim] = {
              maskedDForce[r][0] * basisScaled,
              maskedDForce[r][1] * basisScaled,
              maskedDForce[r][2] * basisScaled};
          for (int g = 0; g < kNumNodes; ++g) {
            V const bg = basis[g];
            for (int c = 0; c < kSpaceDim; ++c) {
              (*outDRes)[(f * kNumFields + r) * kNumDofs + (g * kNumFields + c)] -= rowCoef[c] * bg;
            }
          }
        }
      }
    }
  }

  return hasAnyOutput;
}

} // namespace mochi::fem
