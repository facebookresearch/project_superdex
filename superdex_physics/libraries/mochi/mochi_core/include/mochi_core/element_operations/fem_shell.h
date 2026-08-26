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
#include <mochi_core/element_operations/element_operation_utils.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

#include <limits>

namespace mochi::fem {

// Bending stencil constants.
inline int constexpr kBendingStencilNodes = 6;
inline int constexpr kSpaceDim3 = 3;
inline int constexpr kBendingStencilDofs = kBendingStencilNodes * kSpaceDim3;
inline int constexpr kTriangleNodes = 3;

/// @brief Compile-time element tag for the 6-node shell bending assembly stencil.
///
/// @details Used as the @p ElementT parameter for FEM assembly and batched element
/// vectors/matrices. Nodes 0-2 are the triangle vertices; nodes 3-5 are the opposite hinge
/// vertices used by bending.
struct ShellStencilElement {
  static constexpr int kNumDofs = kBendingStencilNodes;
  static constexpr int kSpaceDim = kSpaceDim3;
};

/// @brief Compile-time element tag for the 3-node shell triangle sub-stencil.
///
/// @details Used for spatial-only shell gravity and inertia work inside the 6-node shell assembly
/// path. The sub-stencil corresponds to nodes 0-2 of @ref ShellStencilElement.
struct ShellTriangleElement {
  static constexpr int kNumDofs = kTriangleNodes;
  static constexpr int kSpaceDim = kSpaceDim3;
};

// Use the global DoF indices and their positions within the stencil to fill out a fixed-size
// stencil of global nodes, with placeholder indices.
inline NdArray<int, kBendingStencilNodes> GlobalStencilIndices(
    Span<int const> const& globalIndices,
    Span<int const> const& stencilIndices) {
  int const numEleDofs = isize(globalIndices);
  MOCHI_ASSERT_VERBOSE(
      isize(stencilIndices) == numEleDofs,
      "Every global index must have a corresponding stencil index");
  // Return value:
  NdArray<int, kBendingStencilNodes> stencilGlobalIndices;
  // Fast path for common case of a full stencil:
  if (numEleDofs == kBendingStencilDofs) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    for (int i = 0; i < kBendingStencilDofs; ++i) {
      MOCHI_ASSERT_VERBOSE(
          stencilIndices[i] == i, "Full-stencil fast path requires identity stencil indices.");
    }
#endif
    for (int i = 0; i < kBendingStencilNodes; i++) {
      stencilGlobalIndices[i] = globalIndices[i * kSpaceDim3] / kSpaceDim3;
    }
    return stencilGlobalIndices;
  }
  // Otherwise, initialize output to sentinel index value.
  for (int i = 0; i < kBendingStencilNodes; i++) {
    stencilGlobalIndices[i] = kSentinelIndex;
  }
  // Fill in global indices for valid stencil nodes.
  for (int i = 0; i < numEleDofs; i += kSpaceDim3) {
    MOCHI_ASSERT_VERBOSE(
        stencilIndices[i] < kBendingStencilDofs && stencilIndices[i] >= 0,
        "Invalid local stencil index");
    stencilGlobalIndices[stencilIndices[i] / kSpaceDim3] = globalIndices[i] / kSpaceDim3;
  }
  return stencilGlobalIndices;
}

/**************************************************************************************************
  Position helpers.
*/

template <int kBatchSize>
MOCHI_FORCE_INLINE void ExtrapolateStencilPositions(
    NdArray<int, kBendingStencilNodes, kBatchSize> const& stencilGlobalNodes,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs>& outPos) {
  using V = BatchReal<kBatchSize>;
  alignas(alignof(V)) real staging[V::kSize]{};
  // Triangle vertices in cyclic order. The hinge node opposite v0 lives at stencil slot (v0 +
  // kTriangleNodes). When missing it is extrapolated as x[v1] + x[v2] - x[v0].
  for (auto const& v : {Int3{0, 1, 2}, Int3{1, 2, 0}, Int3{2, 0, 1}}) {
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = (stencilGlobalNodes[v[0] + kTriangleNodes][b] == kSentinelIndex) ? 1_r : 0_r;
    }
    V const isMissing = (Load<V>(staging) > V{0_r});
    int const hingeIdx = (v[0] + kTriangleNodes) * kSpaceDim3;
    Int3 const vIdx = v * kSpaceDim3;
    for (int d = 0; d < kSpaceDim3; ++d) {
      outPos[hingeIdx + d] = Select(
          isMissing,
          outPos[vIdx[1] + d] + outPos[vIdx[2] + d] - outPos[vIdx[0] + d],
          outPos[hingeIdx + d]);
    }
  }
}

/**************************************************************************************************
  Metric tensor and strain.
*/

// Avoids catastrophic cancellation in 0.5*(Metric(x) - Metric(X)) by expanding in terms of
// reference edges and displacement differences.
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal2x2<kBatchSize> MembraneStrain(
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs> const& refPos,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs> const& disp) {
  using V = BatchReal<kBatchSize>;
  BatchReal3<kBatchSize> E0 MOCHI_NO_INIT, E1 MOCHI_NO_INIT, du0 MOCHI_NO_INIT, du1 MOCHI_NO_INIT;
  for (int d = 0; d < kSpaceDim3; ++d) {
    E0[d] = refPos[1 * kSpaceDim3 + d] - refPos[0 * kSpaceDim3 + d];
    E1[d] = refPos[2 * kSpaceDim3 + d] - refPos[0 * kSpaceDim3 + d];
    du0[d] = disp[1 * kSpaceDim3 + d] - disp[0 * kSpaceDim3 + d];
    du1[d] = disp[2 * kSpaceDim3 + d] - disp[0 * kSpaceDim3 + d];
  }
  return SymMatrix2x2(
      Dot(E0 + V{0.5_r} * du0, du0),
      V{0.5_r} * (Dot(E0, du1) + Dot(du0, E1 + du1)),
      Dot(E1 + V{0.5_r} * du1, du1));
}

template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal2x2<kBatchSize> Metric(
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs> const& pos) {
  BatchReal3<kBatchSize> e0 MOCHI_NO_INIT, e1 MOCHI_NO_INIT;
  for (int d = 0; d < kSpaceDim3; ++d) {
    e0[d] = pos[1 * kSpaceDim3 + d] - pos[0 * kSpaceDim3 + d];
    e1[d] = pos[2 * kSpaceDim3 + d] - pos[0 * kSpaceDim3 + d];
  }
  return SymMatrix2x2(Dot(e0, e0), Dot(e0, e1), Dot(e1, e1));
}

template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<BatchSymMatrix2x2<kBatchSize>, kTriangleNodes, kSpaceDim3>
DMetricDx(NdArray<BatchReal<kBatchSize>, kBendingStencilDofs> const& pos) {
  using V = BatchReal<kBatchSize>;
  BatchReal3<kBatchSize> e0 MOCHI_NO_INIT, e1 MOCHI_NO_INIT;
  for (int d = 0; d < kSpaceDim3; ++d) {
    e0[d] = pos[1 * kSpaceDim3 + d] - pos[0 * kSpaceDim3 + d];
    e1[d] = pos[2 * kSpaceDim3 + d] - pos[0 * kSpaceDim3 + d];
  }
  NdArray<BatchSymMatrix2x2<kBatchSize>, kTriangleNodes, kSpaceDim3> da_dx MOCHI_NO_INIT;
  V const two{2_r};
  for (int i = 0; i < kSpaceDim3; ++i) {
    da_dx[0][i] = Sym2x2Components(-two * e0[i], -e0[i] - e1[i], -two * e1[i]);
    da_dx[1][i] = Sym2x2Components(two * e0[i], e1[i], V{0_r});
    da_dx[2][i] = Sym2x2Components(V{0_r}, e0[i], two * e1[i]);
  }
  return da_dx;
}

/// @brief Returns the second derivatives of the metric tensor with respect to nodal positions.
/// Result is indexed as d2a_dx2[nodeA][nodeB] -> raw [00, 01, 11] components representing the
/// second derivative d^2(a)/d(x[nodeA][i])d(x[nodeB][i]).
///
/// @note The second derivatives are non-zero only when the spatial indices match (i == j). This is
/// because the metric tensor depends on edge vectors e0 = x[1] - x[0] and e1 = x[2] - x[0], and
/// de_k[i]/dx[node][j] = delta_{ij} * (coefficient). Therefore, this function returns a 2D array
/// indexed by [nodeA][nodeB], where each entry stores the diagonal values for all spatial
/// components i in one symmetric 2x2 component vector (exploiting that the values are the same for
/// all i).
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE
    NdArray<BatchSymMatrix2x2<kBatchSize>, kTriangleNodes, kTriangleNodes>
    D2MetricDx2() {
  using V = BatchReal<kBatchSize>;
  NdArray<BatchSymMatrix2x2<kBatchSize>, kTriangleNodes, kTriangleNodes> d2a MOCHI_NO_INIT;
  d2a[0][0] = Sym2x2Components(V{2_r}, V{2_r}, V{2_r});
  d2a[0][1] = Sym2x2Components(V{-2_r}, V{-1_r}, V{0_r});
  d2a[0][2] = Sym2x2Components(V{0_r}, V{-1_r}, V{-2_r});
  d2a[1][0] = d2a[0][1];
  d2a[1][1] = Sym2x2Components(V{2_r}, V{0_r}, V{0_r});
  d2a[1][2] = Sym2x2Components(V{0_r}, V{1_r}, V{0_r});
  d2a[2][0] = d2a[0][2];
  d2a[2][1] = d2a[1][2];
  d2a[2][2] = Sym2x2Components(V{0_r}, V{0_r}, V{2_r});
  return d2a;
}

/**************************************************************************************************
  SVK constitutive response (shared by membrane and bending).
*/

/// @brief St. Venant–Kirchhoff energy density (per unit reference area) in terms of tr(ε) and
/// tr(ε²), with Lamé parameters that absorb thickness. Shared by membrane and bending.
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchDouble<kBatchSize> PsiSVK(
    BatchReal2x2<kBatchSize> const& strain,
    BatchReal<kBatchSize> lambda,
    BatchReal<kBatchSize> mu) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  V const tr = Trace(strain);
  V const tr2 = Trace(Dot(strain, strain));
  return StaticCast<Vd>(V{0.5_r} * lambda * tr * tr) + StaticCast<Vd>(mu * tr2);
}

template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal2x2<kBatchSize> DPsiSVK(
    BatchReal2x2<kBatchSize> const& strain,
    BatchReal2x2<kBatchSize> const& AInv,
    BatchReal<kBatchSize> lambda,
    BatchReal<kBatchSize> mu) {
  using V = BatchReal<kBatchSize>;
  V const tr = Trace(strain);
  BatchReal2x2<kBatchSize> const sAInv = Dot(strain, AInv);
  return lambda * tr * AInv + V{2_r} * mu * sAInv;
}

/// @brief Reduced SVK material tangent for symmetric 2x2 strain/stress tensors.
///
/// @details Rows and columns are ordered as raw [00, 01, 11] components. The six unique tangent
/// entries use the symmetric 3x3 layout [C00, C11, C22, C01, C02, C12]. Apply this tangent to a
/// symmetric tensor x using the weighted input [x00, 2*x01, x11]; contract the resulting symmetric
/// tensor with test tensors using @ref ColonSym2x2.
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchSymMatrix3x3<kBatchSize> D2PsiSVKSym2x2(
    BatchReal2x2<kBatchSize> const& AInv,
    BatchReal<kBatchSize> lambda,
    BatchReal<kBatchSize> mu) {
  using V = BatchReal<kBatchSize>;
  V const a00 = AInv[0][0];
  V const a01 = AInv[0][1];
  V const a11 = AInv[1][1];
  V const lambdaPlusTwoMu = lambda + V{2_r} * mu;

  V const c00 = lambdaPlusTwoMu * a00 * a00;
  V const c01 = lambdaPlusTwoMu * a00 * a01;
  V const c02 = V{2_r} * mu * a01 * a01 + lambda * a00 * a11;
  V const c11 = mu * (a00 * a11 + a01 * a01) + lambda * a01 * a01;
  V const c12 = lambdaPlusTwoMu * a01 * a11;
  V const c22 = lambdaPlusTwoMu * a11 * a11;
  return {c00, c11, c22, c01, c02, c12};
}

namespace details {

template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr BatchSymMatrix2x2<kBatchSize> ApplySym2x2Tangent(
    BatchSymMatrix3x3<kBatchSize> const& tangent,
    BatchSymMatrix2x2<kBatchSize> const& x) {
  using V = BatchReal<kBatchSize>;
  V const twoX01 = V{2_r} * x[1];
  return {
      tangent[0] * x[0] + tangent[3] * twoX01 + tangent[4] * x[2],
      tangent[3] * x[0] + tangent[1] * twoX01 + tangent[5] * x[2],
      tangent[4] * x[0] + tangent[5] * twoX01 + tangent[2] * x[2]};
}

} // namespace details

/**************************************************************************************************
  Membrane residual and dresidual.
*/

template <int kBatchSize>
MOCHI_FORCE_INLINE void AddMembraneResidual(
    NdArray<BatchSymMatrix2x2<kBatchSize>, kTriangleNodes, kSpaceDim3> const& da_dx,
    BatchReal2x2<kBatchSize> const& dpsi_da,
    BatchReal<kBatchSize> referenceArea,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs>& outRes) {
  BatchSymMatrix2x2<kBatchSize> const dpsi = Sym2x2Components(dpsi_da);
  for (int a = 0; a < kTriangleNodes; ++a) {
    for (int i = 0; i < kSpaceDim3; ++i) {
      outRes[a * kSpaceDim3 + i] += referenceArea * ColonSym2x2(da_dx[a][i], dpsi);
    }
  }
}

template <int kBatchSize>
MOCHI_FORCE_INLINE void AddMembraneDResidual(
    NdArray<BatchSymMatrix2x2<kBatchSize>, kTriangleNodes, kSpaceDim3> const& da_dx,
    BatchReal2x2<kBatchSize> const& dpsi_da,
    BatchReal2x2<kBatchSize> const& AInv,
    BatchReal<kBatchSize> referenceArea,
    BatchReal<kBatchSize> lambda,
    BatchReal<kBatchSize> mu,
    bool projectPsd,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs * kBendingStencilDofs>& outDRes) {
  using V = BatchReal<kBatchSize>;
  using V2x2 = BatchReal2x2<kBatchSize>;

  // Material stiffness: pre-absorb scale = 0.25 * area (two factors of 0.5 from dε/da).
  BatchSymMatrix3x3<kBatchSize> d2psi = D2PsiSVKSym2x2<kBatchSize>(AInv, lambda, mu);
  d2psi *= V{0.25_r} * referenceArea;
  for (int trialNode = 0; trialNode < kTriangleNodes; ++trialNode) {
    NdArray<BatchSymMatrix2x2<kBatchSize>, kSpaceDim3> tangentApplied MOCHI_NO_INIT;
    for (int j = 0; j < kSpaceDim3; ++j) {
      tangentApplied[j] = details::ApplySym2x2Tangent<kBatchSize>(d2psi, da_dx[trialNode][j]);
    }
    for (int testNode = 0; testNode <= trialNode; ++testNode) {
      for (int i = 0; i < kSpaceDim3; ++i) {
        BatchSymMatrix2x2<kBatchSize> const& test = da_dx[testNode][i];
        int const outBase =
            (testNode * kSpaceDim3 + i) * kBendingStencilDofs + trialNode * kSpaceDim3;
        for (int j = 0; j < kSpaceDim3; ++j) {
          outDRes[outBase + j] += ColonSym2x2(test, tangentApplied[j]);
        }
      }
    }
  }

  // Geometric stiffness: dpsi_da * d2a/dx2 (upper triangle only).
  V2x2 const dpsi_da_proj =
      projectPsd ? BatchedProjectPsdWithMetric<kBatchSize>(dpsi_da, AInv) : dpsi_da;
  BatchSymMatrix2x2<kBatchSize> const dpsi_da_projSym = Sym2x2Components(dpsi_da_proj);

  auto const d2a_dx2 = D2MetricDx2<kBatchSize>();
  for (int testNode = 0; testNode < kTriangleNodes; ++testNode) {
    for (int trialNode = testNode; trialNode < kTriangleNodes; ++trialNode) {
      V const contraction =
          referenceArea * ColonSym2x2(d2a_dx2[testNode][trialNode], dpsi_da_projSym);
      for (int i = 0; i < kSpaceDim3; ++i) {
        outDRes[(testNode * kSpaceDim3 + i) * kBendingStencilDofs + trialNode * kSpaceDim3 + i] +=
            contraction;
      }
    }
  }
}

/**************************************************************************************************
  Second fundamental form and derivatives.
*/

/// @brief Compute the 6 edge vectors of a bending stencil for a batch of elements.
///
/// @details Edges 0–2 are the triangle edges (v[i] = x_{(i+1)%3} - x_i), edges 3–5 connect triangle
/// vertices to their opposite hinge nodes (v[i+3] = x_{i+3} - x_{(i+1)%3}).
///
/// @param[in] pos  Batched 18-DoF position vector (6 nodes × 3 spatial, extrapolated).
/// @return Per-edge batched 3-vectors, indexed as result[edge][dim].
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<BatchReal3<kBatchSize>, kBendingStencilNodes> EdgeVectors(
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs> const& pos) {
  NdArray<BatchReal3<kBatchSize>, kBendingStencilNodes> v MOCHI_NO_INIT;
  for (int i = 0; i < kTriangleNodes; ++i) {
    int const ip1 = (i + 1) % 3;
    for (int d = 0; d < kSpaceDim3; ++d) {
      v[i][d] = pos[ip1 * kSpaceDim3 + d] - pos[i * kSpaceDim3 + d];
      v[i + 3][d] = pos[(i + 3) * kSpaceDim3 + d] - pos[ip1 * kSpaceDim3 + d];
    }
  }
  return v;
}

/// @brief Kelvin inversion (inversion in the unit sphere) of a batched 3-vector: v / |v|^2.
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal3<kBatchSize> BatchKelvinInversion(
    BatchReal3<kBatchSize> const& v) {
  using V = BatchReal<kBatchSize>;
  V const invNormSqr = V{1_r} / (NormSqr(v) + V{std::numeric_limits<real>::min()});
  return invNormSqr * v;
}

/// @brief Derivative of @ref BatchKelvinInversion w.r.t. v: (1/|v|^2) * (I - 2 * vHat (x) vHat),
/// expanded to avoid a square root as invNormSqr * I - 2 * invNormSqr^2 * (v (x) v).
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE BatchReal3x3<kBatchSize> BatchDKelvinInversion(
    BatchReal3<kBatchSize> const& v) {
  using V = BatchReal<kBatchSize>;
  V const normSqr = NormSqr(v);
  V const invNormSqr = V{1_r} / (normSqr + V{std::numeric_limits<real>::min()});
  // Avoid squaring invNormSqr directly, to prevent overflow for normSqr near zero.
  V const twoInvNormSqrSqr = V{2_r} / (normSqr * normSqr + V{std::numeric_limits<real>::min()});
  return invNormSqr * Eye<3, V>() - twoInvNormSqrSqr * Outer(v, v);
}

/// @brief Compute the discrete second fundamental form for a batch of shell elements.
///
/// @details Uses averaged face normals from the 4 triangles formed by the bending stencil. Each
/// face normal n̂_i is computed from edge cross products. The averaged normal driving the form is
/// the Kelvin inversion of the mean of each pair (n̄_f = KelvinInversion(0.5 * (n̂_0 + n̂_{f+1}))),
/// producing 3 averaged normals. The 2×2 symmetric second fundamental form is assembled from dot
/// products of normal differences with edges.
///
/// @param[in] v  Batched edge vectors (6 edges, from @ref EdgeVectors).
/// @return Batched 2×2 second fundamental form matrix (symmetric, row-major).
template <int kBatchSize>
// Not MOCHI_FORCE_INLINE: forced inlining creates oversized ShellWork stack frames on MSVC.
[[nodiscard]] inline BatchReal2x2<kBatchSize> SecondFundamentalForm(
    NdArray<BatchReal3<kBatchSize>, kBendingStencilNodes> const& v) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;

  NdArray<V3, 4> nHat MOCHI_NO_INIT;
  for (int i = 0; i < 4; ++i) {
    int const e0 = i + 2, e1 = (e0 + 1) % 3;
    nHat[i] = Normalize(Cross(v[e0], v[e1]));
  }
  V const half{0.5_r};
  NdArray<V3, 3> nAvg MOCHI_NO_INIT;
  for (int f = 0; f < 3; ++f) {
    nAvg[f] = BatchKelvinInversion<kBatchSize>(half * (nHat[0] + nHat[f + 1]));
  }
  V3 const d10 = nAvg[1] - nAvg[0];
  V3 const d02 = nAvg[0] - nAvg[2];
  V const two{2_r};
  return SymMatrix2x2(two * Dot(d10, v[0]), two * Dot(nAvg[0], v[2]), two * Dot(d02, v[2]));
}

/// @brief Derivative of the second fundamental form with respect to edge vectors.
///
/// @details Computes db/dv where b is the discrete second fundamental form and v are the 6 stencil
/// edge vectors.
///
/// @param[in] v  Batched edge vectors (6 edges, from @ref EdgeVectors).
/// @return db_dv[edge][dim] — raw [00, 01, 11] derivative components per edge per spatial
/// dimension.
template <int kBatchSize>
// Not MOCHI_FORCE_INLINE: forced inlining creates oversized ShellWork stack frames on MSVC.
[[nodiscard]] inline NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3>
DSecondFundamentalFormDEdges(NdArray<BatchReal3<kBatchSize>, kBendingStencilNodes> const& v) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;

  NdArray<V3, 4> n MOCHI_NO_INIT, nHat MOCHI_NO_INIT;
  for (int i = 0; i < 4; ++i) {
    int const e0 = i + 2, e1 = (e0 + 1) % 3;
    n[i] = Cross(v[e0], v[e1]);
    nHat[i] = Normalize(n[i]);
  }
  NdArray<V3, 3> nMid MOCHI_NO_INIT, nAvg MOCHI_NO_INIT;
  V const half{0.5_r};
  for (int f = 0; f < 3; ++f) {
    nMid[f] = half * (nHat[0] + nHat[f + 1]);
    nAvg[f] = BatchKelvinInversion<kBatchSize>(nMid[f]);
  }
  // Taking nSum = 2 * nMid, d(nAvg)/d(nSum) = 0.5 * d(KelvinInversion)/d(nMid).
  NdArray<V3x3, 3> const dnAvg_dnSum = {
      half * BatchDKelvinInversion<kBatchSize>(nMid[0]),
      half * BatchDKelvinInversion<kBatchSize>(nMid[1]),
      half * BatchDKelvinInversion<kBatchSize>(nMid[2])};

  // Precompute dnHat/dn for each face normal (reused across per-edge loop).
  NdArray<V3x3, 4> const dnHat_dn = {
      DNormalize(n[0]), DNormalize(n[1]), DNormalize(n[2]), DNormalize(n[3])};

  NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3> db_dv{};
  V const two{2_r};
  V3 const twoNAvg1Minus0 = two * (nAvg[1] - nAvg[0]);
  V3 const twoNAvg0Minus2 = two * (nAvg[0] - nAvg[2]);
  for (int i = 0; i < kSpaceDim3; ++i) {
    db_dv[0][i][0] = twoNAvg1Minus0[i];
    db_dv[2][i] = Sym2x2Components(V{0_r}, two * nAvg[0][i], twoNAvg0Minus2[i]);
  }

  V3 const v0d1 = DotVecMat(v[0], dnAvg_dnSum[1]);
  V3 const v0d0 = DotVecMat(v[0], dnAvg_dnSum[0]);
  V3 const v2d0 = DotVecMat(v[2], dnAvg_dnSum[0]);
  V3 const v2d2 = DotVecMat(v[2], dnAvg_dnSum[2]);

  for (int vi = 0; vi < kBendingStencilNodes; ++vi) {
    V3x3 dn0{};
    if (vi == 2) {
      dn0 = -Dot(dnHat_dn[0], Skew(v[0]));
    } else if (vi == 0) {
      dn0 = Dot(dnHat_dn[0], Skew(v[2]));
    }

    NdArray<V3x3, 3> dnSum_dv_vi MOCHI_NO_INIT;
    for (int ns = 0; ns < 3; ++ns) {
      int const ni = ns + 1;
      int const e0 = ni + 2, e1 = (e0 + 1) % 3;
      V3x3 dn_ni{};
      if (vi == e0) {
        dn_ni = -Dot(dnHat_dn[ni], Skew(v[e1]));
      } else if (vi == e1) {
        dn_ni = Dot(dnHat_dn[ni], Skew(v[e0]));
      }
      dnSum_dv_vi[ns] = dn0 + dn_ni;
    }

    V3 const v0_dnAvg1 = DotVecMat(v0d1, dnSum_dv_vi[1]);
    V3 const v0_dnAvg0 = DotVecMat(v0d0, dnSum_dv_vi[0]);
    V3 const v2_dnAvg0 = DotVecMat(v2d0, dnSum_dv_vi[0]);
    V3 const v2_dnAvg2 = DotVecMat(v2d2, dnSum_dv_vi[2]);
    for (int j = 0; j < kSpaceDim3; ++j) {
      db_dv[vi][j] += Sym2x2Components(
          two * (v0_dnAvg1[j] - v0_dnAvg0[j]),
          two * v2_dnAvg0[j],
          two * (v2_dnAvg0[j] - v2_dnAvg2[j]));
    }
  }
  return db_dv;
}

/// Fused computation of the second fundamental form (b) and its edge derivative (db_dv).
/// TODO: This function improves performance at the cost of duplicating code from
/// SecondFundamentalForm and DSecondFundamentalFormDEdges. Consider implementing them as a single
/// templated function.
template <int kBatchSize>
// Not MOCHI_FORCE_INLINE: forced inlining creates oversized ShellWork stack frames on MSVC.
inline void SecondFundamentalFormAndDEdges(
    NdArray<BatchReal3<kBatchSize>, kBendingStencilNodes> const& v,
    BatchReal2x2<kBatchSize>& outB,
    NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3>& outDb_dv) {
  using V = BatchReal<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;

  NdArray<V3, 4> n MOCHI_NO_INIT, nHat MOCHI_NO_INIT;
  for (int i = 0; i < 4; ++i) {
    int const e0 = i + 2, e1 = (e0 + 1) % 3;
    n[i] = Cross(v[e0], v[e1]);
    nHat[i] = Normalize(n[i]);
  }
  NdArray<V3, 3> nMid MOCHI_NO_INIT, nAvg MOCHI_NO_INIT;
  V const half{0.5_r};
  for (int f = 0; f < 3; ++f) {
    nMid[f] = half * (nHat[0] + nHat[f + 1]);
    nAvg[f] = BatchKelvinInversion<kBatchSize>(nMid[f]);
  }

  V const two{2_r};
  V3 const d10 = nAvg[1] - nAvg[0];
  V3 const d02 = nAvg[0] - nAvg[2];
  outB = SymMatrix2x2(two * Dot(d10, v[0]), two * Dot(nAvg[0], v[2]), two * Dot(d02, v[2]));

  // nMid = 0.5 * nSum, so d(nAvg)/d(nSum) = 0.5 * d(KelvinInversion)/d(nMid).
  NdArray<V3x3, 3> const dnAvg_dnSum = {
      half * BatchDKelvinInversion<kBatchSize>(nMid[0]),
      half * BatchDKelvinInversion<kBatchSize>(nMid[1]),
      half * BatchDKelvinInversion<kBatchSize>(nMid[2])};

  NdArray<V3x3, 4> const dnHat_dn = {
      DNormalize(n[0]), DNormalize(n[1]), DNormalize(n[2]), DNormalize(n[3])};

  outDb_dv = {};
  V3 const twoNAvg1Minus0 = two * d10;
  V3 const twoNAvg0Minus2 = two * d02;
  for (int i = 0; i < kSpaceDim3; ++i) {
    outDb_dv[0][i][0] = twoNAvg1Minus0[i];
    outDb_dv[2][i] = Sym2x2Components(V{0_r}, two * nAvg[0][i], twoNAvg0Minus2[i]);
  }
  V3 const v0d1 = DotVecMat(v[0], dnAvg_dnSum[1]);
  V3 const v0d0 = DotVecMat(v[0], dnAvg_dnSum[0]);
  V3 const v2d0 = DotVecMat(v[2], dnAvg_dnSum[0]);
  V3 const v2d2 = DotVecMat(v[2], dnAvg_dnSum[2]);
  for (int vi = 0; vi < kBendingStencilNodes; ++vi) {
    V3x3 dn0{};
    if (vi == 2) {
      dn0 = -Dot(dnHat_dn[0], Skew(v[0]));
    } else if (vi == 0) {
      dn0 = Dot(dnHat_dn[0], Skew(v[2]));
    }
    NdArray<V3x3, 3> dnSum_dv_vi MOCHI_NO_INIT;
    for (int ns = 0; ns < 3; ++ns) {
      int const ni = ns + 1;
      int const e0 = ni + 2, e1 = (e0 + 1) % 3;
      V3x3 dn_ni{};
      if (vi == e0) {
        dn_ni = -Dot(dnHat_dn[ni], Skew(v[e1]));
      } else if (vi == e1) {
        dn_ni = Dot(dnHat_dn[ni], Skew(v[e0]));
      }
      dnSum_dv_vi[ns] = dn0 + dn_ni;
    }
    V3 const v0_dnAvg1 = DotVecMat(v0d1, dnSum_dv_vi[1]);
    V3 const v0_dnAvg0 = DotVecMat(v0d0, dnSum_dv_vi[0]);
    V3 const v2_dnAvg0 = DotVecMat(v2d0, dnSum_dv_vi[0]);
    V3 const v2_dnAvg2 = DotVecMat(v2d2, dnSum_dv_vi[2]);
    for (int j = 0; j < kSpaceDim3; ++j) {
      outDb_dv[vi][j] += Sym2x2Components(
          two * (v0_dnAvg1[j] - v0_dnAvg0[j]),
          two * v2_dnAvg0[j],
          two * (v2_dnAvg0[j] - v2_dnAvg2[j]));
    }
  }
}

/// @brief Derivative of the second fundamental form w.r.t. the (non-extrapolated) node positions,
/// computed directly from the per-edge derivative db/dv and the stencil topology.
///
/// @details db/dx = (db/dv) . (dv/dx). For an interior stencil dv/dx is the constant edge-incidence
/// matrix kC (each of the 6 edge rows has exactly one +1 and one -1), so the fast path is just the
/// column contraction db_dx[n] = sum_e kC[e][n] * db_dv[e]:
///   n0: db_dv[2]-db_dv[0]-db_dv[5]   n1: db_dv[0]-db_dv[1]-db_dv[3]
///   n2: db_dv[1]-db_dv[2]-db_dv[4]   n3: db_dv[3]   n4: db_dv[4]   n5: db_dv[5]
/// On a boundary, a missing opposite stencil node x_{i+3} is extrapolated as
/// x_{i+3} = x_{(i+1)%3} + x_{(i+2)%3} - x_i, which redistributes that edge's contribution onto the
/// triangle vertices. The slow path applies this per lane via the missingOppositeNode /
/// oppositeNodePresent factors.
///
/// @param[in] db_dv  Per-edge derivative of the second fundamental form, db/dv[edge][dim].
/// @param[in] stencilGlobalNodes  Per-lane 6-node global indices. @ref kSentinelIndex marks a
/// missing opposite stencil node.
///
/// @return db_dx[node][dim] — derivative of the second fundamental form w.r.t. the 6 physical node
/// positions, stored as raw [00, 01, 11] components.
template <int kBatchSize>
[[nodiscard]] MOCHI_FORCE_INLINE
    NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3>
    DSecondFundamentalFormDx(
        NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3> const& db_dv,
        NdArray<int, kBendingStencilNodes, kBatchSize> const& stencilGlobalNodes) {
  using V = BatchReal<kBatchSize>;

  alignas(alignof(V)) real staging[V::kSize]{};
  NdArray<V, kTriangleNodes> missingOppositeNode MOCHI_NO_INIT;
  bool anyMissingOppositeNode = false;
  for (int h = 0; h < kTriangleNodes; ++h) {
    for (int lane = 0; lane < kBatchSize; ++lane) {
      bool const missing = (stencilGlobalNodes[h + kTriangleNodes][lane] == kSentinelIndex);
      staging[lane] = missing ? 1_r : 0_r;
      anyMissingOppositeNode = anyMissingOppositeNode || missing;
    }
    missingOppositeNode[h] = Load<V>(staging);
  }

  NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3> db_dx MOCHI_NO_INIT;
  if (!anyMissingOppositeNode) {
    for (int i = 0; i < kSpaceDim3; ++i) {
      db_dx[0][i] = db_dv[2][i] - db_dv[0][i] - db_dv[5][i];
      db_dx[1][i] = db_dv[0][i] - db_dv[1][i] - db_dv[3][i];
      db_dx[2][i] = db_dv[1][i] - db_dv[2][i] - db_dv[4][i];
      db_dx[3][i] = db_dv[3][i];
      db_dx[4][i] = db_dv[4][i];
      db_dx[5][i] = db_dv[5][i];
    }
    return db_dx;
  }

  V const one{1_r};
  V const oppositeNode0Present = one - missingOppositeNode[0];
  V const oppositeNode1Present = one - missingOppositeNode[1];
  V const oppositeNode2Present = one - missingOppositeNode[2];
  for (int i = 0; i < kSpaceDim3; ++i) {
    db_dx[0][i] = db_dv[2][i] - db_dv[0][i] - missingOppositeNode[0] * db_dv[3][i] +
        missingOppositeNode[1] * db_dv[4][i] - oppositeNode2Present * db_dv[5][i];
    db_dx[1][i] = db_dv[0][i] - db_dv[1][i] - oppositeNode0Present * db_dv[3][i] -
        missingOppositeNode[1] * db_dv[4][i] + missingOppositeNode[2] * db_dv[5][i];
    db_dx[2][i] = db_dv[1][i] - db_dv[2][i] + missingOppositeNode[0] * db_dv[3][i] -
        oppositeNode1Present * db_dv[4][i] - missingOppositeNode[2] * db_dv[5][i];
    db_dx[3][i] = oppositeNode0Present * db_dv[3][i];
    db_dx[4][i] = oppositeNode1Present * db_dv[4][i];
    db_dx[5][i] = oppositeNode2Present * db_dv[5][i];
  }
  return db_dx;
}

/**************************************************************************************************
  Bending residual and dresidual.
*/

template <int kBatchSize>
MOCHI_FORCE_INLINE void AddBendingResidual(
    NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3> const& db_dx,
    BatchReal2x2<kBatchSize> const& s,
    BatchReal2x2<kBatchSize> const& AInv,
    BatchReal<kBatchSize> referenceArea,
    BatchReal<kBatchSize> alpha,
    BatchReal<kBatchSize> beta,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs>& outRes) {
  // dψ/db = -dψ/dκ because κ = B - b.
  // Bending uses SVK with (alpha, 0.5*beta) as (lambda, mu).
  BatchReal2x2<kBatchSize> const dpsi_db = -DPsiSVK<kBatchSize>(s, AInv, alpha, 0.5_r * beta);
  BatchSymMatrix2x2<kBatchSize> const dpsi = Sym2x2Components(dpsi_db);
  for (int a = 0; a < kBendingStencilNodes; ++a) {
    for (int i = 0; i < kSpaceDim3; ++i) {
      outRes[a * kSpaceDim3 + i] += referenceArea * ColonSym2x2(db_dx[a][i], dpsi);
    }
  }
}

template <int kBatchSize>
MOCHI_FORCE_INLINE void AddBendingDResidual(
    NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3> const& db_dx,
    BatchReal2x2<kBatchSize> const& AInv,
    BatchReal<kBatchSize> referenceArea,
    BatchReal<kBatchSize> alpha,
    BatchReal<kBatchSize> beta,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs * kBendingStencilDofs>& outDRes) {
  // Material bending stiffness (pre-absorb referenceArea into d2psi).
  BatchSymMatrix3x3<kBatchSize> d2psi = D2PsiSVKSym2x2<kBatchSize>(AInv, alpha, 0.5_r * beta);
  d2psi *= referenceArea;
  for (int trialNode = 0; trialNode < kBendingStencilNodes; ++trialNode) {
    NdArray<BatchSymMatrix2x2<kBatchSize>, kSpaceDim3> tangentApplied MOCHI_NO_INIT;
    for (int j = 0; j < kSpaceDim3; ++j) {
      tangentApplied[j] = details::ApplySym2x2Tangent<kBatchSize>(d2psi, db_dx[trialNode][j]);
    }
    for (int testNode = 0; testNode <= trialNode; ++testNode) {
      for (int i = 0; i < kSpaceDim3; ++i) {
        BatchSymMatrix2x2<kBatchSize> const& test = db_dx[testNode][i];
        int const outBase =
            (testNode * kSpaceDim3 + i) * kBendingStencilDofs + trialNode * kSpaceDim3;
        for (int j = 0; j < kSpaceDim3; ++j) {
          outDRes[outBase + j] += ColonSym2x2(test, tangentApplied[j]);
        }
      }
    }
  }

  // NOTE: The geometric bending stiffness is omitted here, following recommendations from the
  // graphics literature, e.g., Section 10.4.2 of
  // https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf, which recommends an
  // analogous simplification to the tangent of a related bending formulation.
}

/**************************************************************************************************
  Top-level shell work.
*/

/// @brief Compute combined membrane + bending work for a batch of shell elements.
///
/// @param[in] stencilGlobalNodes  Per-lane array of 6 global node indices.
/// @param[in] meshNodes  Global mesh node coordinates.
/// @param[in] disp  Batched 18-DoF displacement vector (6 nodes × 3 spatial).
/// @param[out] outEnergy  If non-null, accumulates per-element energy.
/// @param[out] outRes  If non-null, accumulates per-element residual.
/// @param[out] outDRes  If non-null, accumulates per-element stiffness.
/// @param[in] membraneLambda  Membrane Lamé first parameter [Pa·m].
/// @param[in] membraneMu  Membrane Lamé second parameter [Pa·m].
/// @param[in] bendingAlpha  Bending stiffness first parameter [Pa·m³].
/// @param[in] bendingBeta  Bending stiffness second parameter [Pa·m³].
/// @param[in] projectPsd  If true, project membrane geometric stiffness to be PSD.
/// @param[in] stiffnessDampingFactor  β/dt factor for stiffness damping (dimensionless). Unified
///   into the elastic response by scaling the material stiffnesses to (1+factor)·k and using
///   modified strains ε − (factor/(1+factor))·ε₀ (ε₀ = stage-start strain).
/// @param[in] stageStartDisp  Stage-start displacements for stiffness damping (18-DoF batched;
///   required when @p stiffnessDampingFactor > 0, may be null otherwise).
template <int kBatchSize>
bool ShellWork(
    NdArray<int, kBendingStencilNodes, kBatchSize> const& stencilGlobalNodes,
    Span<Real3 const> meshNodes,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs> const& disp,
    BatchDouble<kBatchSize>* outEnergy,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs>* outRes,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs * kBendingStencilDofs>* outDRes,
    real membraneLambda,
    real membraneMu,
    real bendingAlpha,
    real bendingBeta,
    bool projectPsd,
    real stiffnessDampingFactor = 0_r,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs> const* stageStartDisp = nullptr);

} // namespace mochi::fem
