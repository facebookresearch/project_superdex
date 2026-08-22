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
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

namespace mochi::fem {

template <class ElementT>
constexpr auto kNumDisplacementDofs = ElementT::kNumDofs * ElementT::kSpaceDim;

/// @brief Compute inertia work for a batch of elements.
///
/// Implements the inertia term from Brown et al. 2018, "Accurate Dissipative Forces in
/// Optimization Integrators": (1/(h²α²))·M·(x − x̃), where x̃ = xₚ + α·h·vₚ is the stage-start
/// position (integration-method-dependent).
///
/// @param[in] elementIndices  Indices into @p elements for each batch lane.
/// @param[in] elements  Element data array.
/// @param[in] disp  Batched displacements. Only the first ElementT::kNumDofs * ElementT::kSpaceDim
/// entries are read, so it may be a prefix span into a larger element vector.
/// @param[in] stageStartDispl  Batched stage-start displacements (x̃ = xₚ + α·h·vₚ). Only the first
/// ElementT::kNumDofs * ElementT::kSpaceDim entries are read.
/// @param[in,out] outEnergy  If non-null, accumulates per-element energy.
/// @param[in,out] outRes  If non-empty, accumulates per-element residual into the first
/// ElementT::kNumDofs * ElementT::kSpaceDim entries.
/// @param[in] density  Material density [kg/m³].
/// @param[in] dtfi2  Inverse squared time step factor: 1 / (Δt·α)².
/// @param[in] perElementExtraWeight  Optional per-element quadrature weight multiplier.
/// @return true if outputs were written.
///
/// @note DRes is not computed here. It is handled globally via the precomputed mass matrix.
/// @note This is a spatial-only operation. It uses kNumFields == kSpaceDim implicitly. For element
/// types with extra DoFs per node (e.g., rod twist), use dedicated inertia functions.
template <int kBatchSize, class ElementT>
bool InertiaWork(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<ElementT const> elements,
    Span<BatchReal<kBatchSize> const> disp,
    Span<BatchReal<kBatchSize> const> stageStartDispl,
    BatchDouble<kBatchSize>* outEnergy,
    Span<BatchReal<kBatchSize>> outRes,
    real density,
    real dtfi2,
    Span<real const> perElementExtraWeight = {}) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
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
  constexpr auto kNumDofs = kNumNodes * kSpaceDim;
  static_assert(kSpaceDim == 3, "Only 3D elements are supported");
  MOCHI_ASSERT_VERBOSE(disp.size() >= kNumDofs, "Displacement span is too small.");
  MOCHI_ASSERT_VERBOSE(stageStartDispl.size() >= kNumDofs, "Stage-start span is too small.");
  MOCHI_ASSERT_VERBOSE(outRes.empty() || outRes.size() >= kNumDofs, "Residual span is too small.");

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = !outRes.empty();
  if (!evalObj && !evalRes) {
    return false;
  }

  auto const extraWeight = ComputeExtraWeights(elementIndices, perElementExtraWeight);

  // Compute nodal acceleration.
  BatchElementVector<kBatchSize, ElementT> nodalAccel MOCHI_NO_INIT;
  for (int d = 0; d < kNumDofs; ++d) {
    nodalAccel[d] = disp[d] - stageStartDispl[d];
  }

  for (int q = 0; q < kNumQuads; ++q) {
    NdArray<V, kNumNodes> basis MOCHI_NO_INIT;
    V quadWeight MOCHI_NO_INIT;
    PackBasisAndQuadWeight(elementIndices, elements, extraWeight, q, basis, quadWeight);

    // Interpolate nodalAccel to quadrature point: quadVal = Σ_f N_f(q) * nodalAccel_f
    V3 qv = {};
    for (int f = 0; f < kNumNodes; ++f) {
      qv[0] += basis[f] * nodalAccel[f * kSpaceDim + 0];
      qv[1] += basis[f] * nodalAccel[f * kSpaceDim + 1];
      qv[2] += basis[f] * nodalAccel[f * kSpaceDim + 2];
    }

    if (evalObj) {
      // energy += (ρ / 2h²a²) * ‖quadVal‖² * w_q
      *outEnergy += StaticCast<Vd>(density * 0.5_r * dtfi2 * NormSqr(qv) * quadWeight);
    }

    if (evalRes) {
      // res[f] += (ρ / h²a²) * N_f(q) * w_q * quadVal
      V const coeff = V{dtfi2 * density};
      for (int f = 0; f < kNumNodes; ++f) {
        V const weightedBasis = coeff * basis[f] * quadWeight;
        outRes[f * kSpaceDim + 0] += weightedBasis * qv[0];
        outRes[f * kSpaceDim + 1] += weightedBasis * qv[1];
        outRes[f * kSpaceDim + 2] += weightedBasis * qv[2];
      }
    }
  }

  return true;
}

template <int kBatchSize, class ElementT>
MOCHI_FORCE_INLINE bool InertiaWork(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<ElementT const> elements,
    BatchElementVector<kBatchSize, ElementT> const& disp,
    BatchElementVector<kBatchSize, ElementT> const& stageStartDispl,
    BatchDouble<kBatchSize>* outEnergy,
    BatchElementVector<kBatchSize, ElementT>* outRes,
    real density,
    real dtfi2,
    Span<real const> perElementExtraWeight = {}) {
  return InertiaWork<kBatchSize, ElementT>(
      elementIndices,
      elements,
      MakeConstSpan(disp),
      MakeConstSpan(stageStartDispl),
      outEnergy,
      outRes ? MakeSpan(*outRes) : Span<BatchReal<kBatchSize>>{},
      density,
      dtfi2,
      perElementExtraWeight);
}

/// @brief Gather precomputed per-element mass matrices and add them to batched element dresiduals.
///
/// The mass matrix dimension @p kMassDof may be smaller than the element dresidual dimension
/// (@ref ElementT::kNumDofs * @ref ElementT::kSpaceDim). In that case the mass matrix is written
/// into the top-left @p kMassDof × @p kMassDof block, leaving the remaining (padded-stencil)
/// positions zero. This supports assembling, e.g., a 3-node triangle mass matrix through a 6-node
/// shell bending stencil, whose first three nodes are the triangle vertices.
///
/// @param[in] elementIndices  Indices into @p mmPerElem for each batch lane.
/// @param[in] mmPerElem  Precomputed per-element mass matrices (see @ref
/// ComputeMassMatrixPerElement).
/// @param[in,out] outDRes  Accumulates the per-element mass matrix for each lane.
/// @return true.
///
/// @warning Unlike @ref InertiaWork, it scatters the raw mass matrix M, not the time-scaled inertia
/// dresidual (dtfi2 * M). Callers that use this for inertia dresidual assembly must apply the
/// time-step scale when inserting the precomputed global mass matrix.
template <int kBatchSize, class ElementT, size_t kMassDof>
bool AddMassMatrixToDRes(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<NdArray<real, kMassDof, kMassDof> const> mmPerElem,
    BatchElementMatrix<kBatchSize, ElementT>& outDRes) {
  using V = BatchReal<kBatchSize>;
  constexpr auto kNumDof = ElementT::kNumDofs * ElementT::kSpaceDim;
  static_assert(kMassDof <= kNumDof, "Mass matrix must fit in the element dresidual.");
  MOCHI_ASSERT_VERBOSE(!mmPerElem.empty(), "mmPerElem span is empty.");
  MOCHI_ASSERT_VERBOSE(
      Min(elementIndices) >= 0 && Max(elementIndices) < isize(mmPerElem),
      "Element index out of range.");

  alignas(alignof(V)) real staging[V::kSize]{};
  for (size_t i = 0; i < kMassDof; ++i) {
    for (size_t j = 0; j < kMassDof; ++j) {
      for (int b = 0; b < kBatchSize; ++b) {
        staging[b] = mmPerElem[elementIndices[b]][i][j];
      }
      outDRes[i * kNumDof + j] += Load<V>(staging);
    }
  }
  return true;
}

template <class ElementT>
void ComputeMassMatrixPerElement(
    Span<ElementT const> elements,
    real const& density,
    Span<NdArray<real, kNumDisplacementDofs<ElementT>, kNumDisplacementDofs<ElementT>>> mmPerElem,
    Span<int const> activeVolumeGlobalIndices = {},
    Span<real const> activeVolWeights = {}) {
  int const numElem = isize(elements);
  MOCHI_ASSERT_VERBOSE(
      mmPerElem.size() == numElem, "Invalid size of the element mass matrix array");

  MOCHI_ASSERT_VERBOSE(
      (activeVolumeGlobalIndices.empty() && activeVolWeights.empty()) ||
          (!activeVolumeGlobalIndices.empty() && (isize(activeVolWeights) == isize(elements))),
      "Either both active vol elements and weights are empty, or active vol weights must have the same size as the FEM elements.");

  auto PerElementOp = [&](int const elementId) {
    constexpr auto kSpaceDim = ElementT::kSpaceDim;
    constexpr auto kNumQuads = ElementT::kNumQuadPoints;
    constexpr auto kNumNodes = ElementT::kNumDofs;
    constexpr auto kNumDofs = kNumDisplacementDofs<ElementT>;

    ElementT const* element = &elements[elementId];
    NdArray<real, kNumDofs, kNumDofs>& mm = mmPerElem[elementId];

    real const thisElementWeight = activeVolWeights.empty() ? 1_r : activeVolWeights[elementId];
    auto const quadWeights = element->quadWeights * thisElementWeight;

    // Loop over each quadrature point
    for (int q = 0; q < kNumQuads; ++q) {
      NdArray<real, kNumNodes> const kQuadBasis = element->basisEvaluated[q];
      real const kQuadWeight = quadWeights[q];
      for (int f = 0; f < kNumNodes; ++f) {
        for (int g = 0; g < kNumNodes; ++g) {
          for (int i = 0; i < kSpaceDim; ++i) {
            mm[f * kSpaceDim + i][g * kSpaceDim + i] +=
                density * kQuadBasis[f] * kQuadBasis[g] * kQuadWeight;
          }
        }
      }
    }
  };

  bool const useActiveElements = !activeVolumeGlobalIndices.empty();
  int const loopBound = useActiveElements ? isize(activeVolumeGlobalIndices) : numElem;
  for (int i = 0; i < loopBound; ++i) {
    int const elementIndex = useActiveElements ? activeVolumeGlobalIndices[i] : i;
    PerElementOp(elementIndex);
  }
}
} // namespace mochi::fem
