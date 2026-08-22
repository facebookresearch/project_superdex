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
#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

namespace mochi::fem {

/// @brief Augment the in-plane deformation gradient with the out-of-plane normal rotation
/// for triangle elements (plane-strain formulation).
///
/// @note Implements plane-strain kinematics: no transverse shear or out-of-plane stretch, but the
/// surface normal may rotate. Safe with general 3D material models, but parameters may need
/// adjustment to approximate plane-stress for unconstrained shell structures.
///
/// @see StressWork
template <int kBatchSize, class ElementT>
MOCHI_FORCE_INLINE void AugmentDeformationGradientTriangle(
    BatchElementVector<kBatchSize, ElementT> const& disp,
    NdArray<int, kBatchSize> const& elementIndices,
    Span<ElementT const> elements,
    BatchReal3x3<kBatchSize>& F,
    real* staging) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  constexpr auto kSpaceDim = ElementT::kSpaceDim;
  static_assert(ElementT::kNumDofs == 3);

  // Pack reference node positions into SoA.
  NdArray<V, 3 * kSpaceDim> refPos MOCHI_NO_INIT;
  for (int d = 0; d < 3 * kSpaceDim; ++d) {
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = elements[elementIndices[b]].nodesCrdsPhys[d / kSpaceDim][d % kSpaceDim];
    }
    refPos[d] = Load<V>(staging);
  }

  // Current positions = reference + displacement.
  NdArray<V, 3 * kSpaceDim> curPos = refPos + disp;

  // Edge vectors: e1 = pos[1] - pos[0], e2 = pos[2] - pos[0]
  auto edge = [](auto const& pos, int a, int b, int d) { return pos[a * 3 + d] - pos[b * 3 + d]; };
  auto edge3 = [&](auto const& pos, int a, int b) -> V3 {
    return {edge(pos, a, b, 0), edge(pos, a, b, 1), edge(pos, a, b, 2)};
  };

  V3 const refNormal = Normalize(Cross(edge3(refPos, 1, 0), edge3(refPos, 2, 0)));
  V3 const curNormal = Normalize(Cross(edge3(curPos, 1, 0), edge3(curPos, 2, 0)));
  F += Outer(curNormal - refNormal, refNormal);
}

/// @brief Compute stress work for a batch of elements.
///
/// @param[in] elementIndices  Indices into @p elements for each batch lane.
/// @param[in] elements  Element data array.
/// @param[in] disp  Batched displacement DoF vector.
/// @param[out] outEnergy  If non-null, accumulates per-element energy.
/// @param[out] outRes  If non-null, accumulates per-element residual.
/// @param[out] outDRes  If non-null, accumulates per-element stiffness.
/// @param[in] projectPsd  If true, project the tangent modulus to be positive semi-definite.
/// @param[in] constitutiveResponse  Batched constitutive response callback, invoked once per
///   quadrature point with the signature:
///   @code
///     void constitutiveResponse(
///         // Deformation gradient.
///         BatchReal3x3<kBatchSize> const& F,
///         // Non-null iff energy is requested.
///         BatchDouble<kBatchSize>* outEnergy,
///         // Non-null iff first Piola-Kirchhoff stress tensor is requested. Values may be
///         // uninitialized at input.
///         BatchReal3x3<kBatchSize>* outPk1,
///         // Non-null iff tangent tensor is requested. Values may be uninitialized at input. If
///         // requested, must be symmetric at output.
///         NdArray<BatchReal3x3<kBatchSize>, 3, 3>* outTangent,
///         // If true, project the tangent tensor to positive semi-definiteness.
///         bool projectPsd);
///   @endcode
///   The callback is templated so it can be inlined into the assembly loop. Pass it as a lambda or
///   functor.
/// @param[in] perElementExtraWeight  Optional per-element quadrature weight multiplier.
/// @return true if outputs were written.
///
/// @note Triangle dresidual is not implemented because it would require the derivative of the
/// plane-strain deformation gradient with respect to the current normal; @p outDRes must be null
/// for triangle elements.
/// @note This is a spatial-only operation. It uses kNumFields == kSpaceDim implicitly. For element
/// types with extra DoFs per node (e.g., rod twist), use dedicated stress functions.
template <int kBatchSize, class ElementT, class ConstitutiveResponseFn>
bool StressWork(
    NdArray<int, kBatchSize> const& elementIndices,
    Span<ElementT const> elements,
    BatchElementVector<kBatchSize, ElementT> const& disp,
    BatchDouble<kBatchSize>* outEnergy,
    BatchElementVector<kBatchSize, ElementT>* outRes,
    BatchElementMatrix<kBatchSize, ElementT>* outDRes,
    bool projectPsd,
    ConstitutiveResponseFn const& constitutiveResponse,
    Span<real const> perElementExtraWeight = {}) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;
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

  if constexpr (kNumNodes == 3) {
    MOCHI_ASSERT_VERBOSE(outDRes == nullptr, "DRes implementation is incomplete for triangles.");
  }

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = (outRes != nullptr);
  bool const evalDRes = (outDRes != nullptr);
  MOCHI_ASSERT_VERBOSE(evalObj || evalRes || evalDRes, "Must assemble something.");

  auto const extraWeight = ComputeExtraWeights(elementIndices, perElementExtraWeight);

  // Staging buffer for packing element geometry data.
  alignas(alignof(V)) real staging[V::kSize]{};

  for (int q = 0; q < kNumQuads; ++q) {
    // Pack basis derivatives and quad weights.
    NdArray<V, kNumNodes * kSpaceDim> dbasis MOCHI_NO_INIT;
    V quadWeight MOCHI_NO_INIT;

    for (int d = 0; d < kNumNodes * kSpaceDim; ++d) {
      for (int b = 0; b < kBatchSize; ++b) {
        staging[b] = elements[elementIndices[b]].dBasisEvaluated[q][d / kSpaceDim][d % kSpaceDim];
      }
      dbasis[d] = Load<V>(staging);
    }
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = elements[elementIndices[b]].quadWeights[q] * extraWeight[b];
    }
    quadWeight = Load<V>(staging);

    // Deformation gradient.
    // F[r][c] = δ_rc + Σ_f disp[f*3+r] * dbasis[f*3+c]
    V3x3 F = Eye<3, V>();
    for (int f = 0; f < kNumNodes; ++f) {
      for (int r = 0; r < kSpaceDim; ++r) {
        V const u_fr = disp[f * kSpaceDim + r];
        for (int c = 0; c < kSpaceDim; ++c) {
          F[r][c] += u_fr * dbasis[f * kSpaceDim + c];
        }
      }
    }

    // Plane-strain augmentation for triangles.
    if constexpr (kNumNodes == 3) {
      AugmentDeformationGradientTriangle<kBatchSize, ElementT>(
          disp, elementIndices, elements, F, staging);
    }

    // Constitutive response.
    Vd energy{0.0};
    V3x3 pk1 MOCHI_NO_INIT;
    NdArray<V3x3, 3, 3> tangent MOCHI_NO_INIT;

    constitutiveResponse(
        elementIndices,
        F,
        evalObj ? &energy : nullptr,
        evalRes ? &pk1 : nullptr,
        evalDRes ? &tangent : nullptr,
        projectPsd);

    if (evalObj) {
      *outEnergy += energy * StaticCast<Vd>(quadWeight);
    }

    // Residual: res[f*3+r] += Σ_c PK1[r][c] * dbasisScaled[f*3+c]
    // NOTE: dbasisScaled is computed separately in the evalRes and evalDRes blocks rather than
    // hoisted above based on benchmark results.
    if (evalRes) {
      NdArray<V, kNumNodes * kSpaceDim> dbasisScaled MOCHI_NO_INIT;
      for (int d = 0; d < kNumNodes * kSpaceDim; ++d) {
        dbasisScaled[d] = dbasis[d] * quadWeight;
      }

      for (int f = 0; f < kNumNodes; ++f) {
        for (int r = 0; r < kSpaceDim; ++r) {
          (*outRes)[f * kSpaceDim + r] += pk1[r][0] * dbasisScaled[f * kSpaceDim + 0] +
              pk1[r][1] * dbasisScaled[f * kSpaceDim + 1] +
              pk1[r][2] * dbasisScaled[f * kSpaceDim + 2];
        }
      }
    }

    // Stiffness: exploit symmetry, per-row Cf to reduce register pressure.
    if (evalDRes) {
      NdArray<V, kNumNodes * kSpaceDim> dbasisScaled MOCHI_NO_INIT;
      for (int d = 0; d < kNumNodes * kSpaceDim; ++d) {
        dbasisScaled[d] = dbasis[d] * quadWeight;
      }

      for (int f = 0; f < kNumNodes; ++f) {
        for (int r = 0; r < kSpaceDim; ++r) {
          V3x3 Cf_r MOCHI_NO_INIT;
          for (int rr = 0; rr < 3; ++rr) {
            for (int cc = 0; cc < 3; ++cc) {
              Cf_r[rr][cc] = dbasisScaled[f * kSpaceDim + 0] * tangent[r][0][rr][cc] +
                  dbasisScaled[f * kSpaceDim + 1] * tangent[r][1][rr][cc] +
                  dbasisScaled[f * kSpaceDim + 2] * tangent[r][2][rr][cc];
            }
          }

          for (int g = f; g < kNumNodes; ++g) {
            for (int c = 0; c < kSpaceDim; ++c) {
              (*outDRes)[(f * kSpaceDim + r) * kNumDofs + (g * kSpaceDim + c)] +=
                  Cf_r[c][0] * dbasis[g * kSpaceDim + 0] + Cf_r[c][1] * dbasis[g * kSpaceDim + 1] +
                  Cf_r[c][2] * dbasis[g * kSpaceDim + 2];
            }
          }
        }
      }
    }
  }

  // Mirror dresidual.
  if (evalDRes) {
    for (int f = 0; f < kNumNodes; ++f) {
      for (int g = f + 1; g < kNumNodes; ++g) {
        for (int r = 0; r < kSpaceDim; ++r) {
          for (int c = 0; c < kSpaceDim; ++c) {
            (*outDRes)[(g * kSpaceDim + c) * kNumDofs + (f * kSpaceDim + r)] =
                (*outDRes)[(f * kSpaceDim + r) * kNumDofs + (g * kSpaceDim + c)];
          }
        }
      }
    }
  }

  return true;
}

} // namespace mochi::fem
