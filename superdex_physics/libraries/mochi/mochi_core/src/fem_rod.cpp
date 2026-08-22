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

#include <mochi_core/element_operations/fem_rod.h>

#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/utils/batch_config.h>

namespace mochi::fem {

/// @brief Integrated curvature binormal for normalized edge vectors.
///
/// Equivalent to the un-normalized formula with the product of magnitudes in the denominator
/// simplified to 1. The denominator is regularized near exactly opposite orientations to avoid NaNs
/// or sign flips.
template <class V>
[[nodiscard]] static MOCHI_FORCE_INLINE NdArray<V, 3> IntegratedCurvatureBinormal(
    NdArray<V, 3> const& e0Hat,
    NdArray<V, 3> const& e1Hat) {
  V const onePlusDot = Max(V{1} + Dot(e0Hat, e1Hat), V{std::numeric_limits<real>::min()});
  return (V{2} / onePlusDot) * Cross(e0Hat, e1Hat);
}

/// @brief Partial derivatives of the integrated curvature binormal w.r.t. its two arguments.
template <class V>
[[nodiscard]] static MOCHI_FORCE_INLINE std::pair<NdArray<V, 3, 3>, NdArray<V, 3, 3>>
DIntegratedCurvatureBinormalDVecs(NdArray<V, 3> const& e0Hat, NdArray<V, 3> const& e1Hat) {
  NdArray<V, 3> const e0xe1 = Cross(e0Hat, e1Hat);
  NdArray<V, 3, 3> const de0xe1_de0 = -Skew(e1Hat);
  NdArray<V, 3, 3> const de0xe1_de1 = Skew(e0Hat);
  V const invDenom = V{1} / Max(V{1} + Dot(e0Hat, e1Hat), V{std::numeric_limits<real>::min()});
  V const twoOver1pd = V{2} * invDenom;
  return {
      twoOver1pd * (de0xe1_de0 - invDenom * Outer(e0xe1, e1Hat)),
      twoOver1pd * (de0xe1_de1 - invDenom * Outer(e0xe1, e0Hat))};
}

/// @brief Scatter one symmetric component (kI0, kJ0) of the 3x3 axial tangent into the stencil.
///
/// @details Since dx = x1 - x0, each component contributes with the sign pattern [+,-;-,+] across
/// the two node blocks. Off-diagonal components (kI0 != kJ0) also write the mirrored transpose
/// entries.
template <int kI0, int kJ0, class V>
static MOCHI_FORCE_INLINE void ScatterRodAxialTangentComponent(
    V const& value,
    NdArray<V, kNumRodStencilDofs * kNumRodStencilDofs>& outDRes) {
  static_assert(
      0 <= kI0 && kI0 <= kJ0 && kJ0 < RodStencilElement::kSpaceDim,
      "Component indices must address the upper triangle of a symmetric 3x3 tensor.");
  constexpr int kRow = kNumRodStencilDofs;
  constexpr int kI1 = kI0 + kNumRodFields;
  constexpr int kJ1 = kJ0 + kNumRodFields;
  outDRes[kI0 * kRow + kJ0] += value;
  outDRes[kI0 * kRow + kJ1] -= value;
  outDRes[kI1 * kRow + kJ0] -= value;
  outDRes[kI1 * kRow + kJ1] += value;
  if constexpr (kI0 != kJ0) {
    outDRes[kJ0 * kRow + kI0] += value;
    outDRes[kJ0 * kRow + kI1] -= value;
    outDRes[kJ1 * kRow + kI0] -= value;
    outDRes[kJ1 * kRow + kI1] += value;
  }
}

template <int kBatchSize>
bool RodAxialStress(
    Span<Real3 const> meshNodes,
    BatchRodVector<kBatchSize> const& disp,
    NdArray<int, kBatchSize> const& elementIndices,
    Span<int const> l2gFlat,
    BatchDouble<kBatchSize>* outEnergy,
    BatchRodVector<kBatchSize>* outRes,
    BatchRodMatrix<kBatchSize>* outDRes,
    real axialStiffness,
    bool projectPsd,
    real stiffnessDampingFactor,
    NdArray<BatchReal<kBatchSize>, kNumRodStencilDofs> const* stageStartDisp) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  constexpr auto kSpaceDim = RodStencilElement::kSpaceDim;

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = (outRes != nullptr);
  bool const evalDRes = (outDRes != nullptr);
  if (!evalObj && !evalRes && !evalDRes) {
    return false;
  }

  bool const hasDamping = (stiffnessDampingFactor > 0_r) && (stageStartDisp != nullptr);
  MOCHI_ASSERT_VERBOSE(
      stiffnessDampingFactor == 0_r || stageStartDisp != nullptr,
      "Nonzero stiffnessDampingFactor requires stageStartDisp.");

  // Per-lane stencil node indices (read from the L2G map so node 1 can wrap for periodic rods).
  // A 1-node boundary stencil has node 1 collapsed onto node 0; mask those lanes to zero.
  alignas(alignof(V)) real staging[V::kSize]{};
  NdArray<int, kBatchSize> node0Idx MOCHI_NO_INIT;
  NdArray<int, kBatchSize> node1Idx MOCHI_NO_INIT;
  for (int b = 0; b < kBatchSize; ++b) {
    node0Idx[b] = details::RodStencilNodeIndex(l2gFlat, elementIndices[b], 0);
    node1Idx[b] = details::RodStencilNodeIndex(l2gFlat, elementIndices[b], 1);
    staging[b] = (node0Idx[b] != node1Idx[b]) ? 1_r : 0_r;
  }
  V const isActive = (Load<V>(staging) > V{0});
  if (!AnyTrue(isActive))
    MOCHI_UNLIKELY {
      return false;
    }

  // Gather reference positions for nodes 0 and 1.
  V3 X0 MOCHI_NO_INIT;
  V3 X1 MOCHI_NO_INIT;
  for (int d = 0; d < kSpaceDim; ++d) {
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = meshNodes[node0Idx[b]][d];
    }
    X0[d] = Load<V>(staging);
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = meshNodes[node1Idx[b]][d];
    }
    X1[d] = Load<V>(staging);
  }

  // Current positions = reference + displacement.
  V3 const x0 = X0 + V3{disp[0], disp[1], disp[2]};
  V3 const x1 = X1 + V3{disp[kNumRodFields], disp[kNumRodFields + 1], disp[kNumRodFields + 2]};
  V3 const dX = X1 - X0;
  V3 const dx = x1 - x0;

  // Use q = dx/L for Green strain. The clamp keeps division defined in collapsed lanes; inactive
  // products may still overflow and are masked before contributing to outputs.
  V const L = Max(Norm(dX), V{std::numeric_limits<real>::min()});
  V const invL = V{1} / L;
  V3 q MOCHI_NO_INIT;
  for (int i = 0; i < kSpaceDim; ++i) {
    q[i] = Select(isActive, invL * dx[i], V{0});
  }
  V const strain = V{0.5_r} * (NormSqr(q) - V{1});

  // Stiffness damping unified into the elastic formulas: scale the material stiffness by
  // (1 + factor) and use a modified strain ε̃ = ε − (factor/(1+factor))·ε₀ (ε₀ = stage-start
  // strain). Feeding (effEA, effStrain) through the standard undamped quadratic reproduces the
  // damped stress σ = EA·[ε + factor·(ε − ε₀)] and its exact tangent, since the SVK response is
  // linear in strain. When damping is off, effEA = EA and effStrain = strain.
  real const effFactor = hasDamping ? stiffnessDampingFactor : 0_r;
  V const effEA{(1_r + effFactor) * axialStiffness};
  real const ssWeight = effFactor > 0_r ? effFactor / (1_r + effFactor) : 0_r;

  V strainSs = V{0};
  if (hasDamping) {
    V3 const x0Ss = X0 + V3{(*stageStartDisp)[0], (*stageStartDisp)[1], (*stageStartDisp)[2]};
    V3 const x1Ss = X1 +
        V3{(*stageStartDisp)[kNumRodFields],
           (*stageStartDisp)[kNumRodFields + 1],
           (*stageStartDisp)[kNumRodFields + 2]};
    V3 const dxSs = x1Ss - x0Ss;
    V3 qSs MOCHI_NO_INIT;
    for (int i = 0; i < kSpaceDim; ++i) {
      qSs[i] = Select(isActive, invL * dxSs[i], V{0});
    }
    strainSs = V{0.5_r} * (NormSqr(qSs) - V{1});
  }
  V const effStrain = strain - V{ssWeight} * strainSs;
  V const effStress = effEA * effStrain;

  if (evalObj) {
    // Standard quadratic in the modified strain; equals the elastic + dissipation potential up to
    // an x-independent constant (ε₀ is fixed within a stage), which is immaterial to the solver.
    *outEnergy += StaticCast<Vd>(Select(isActive, V{0.5_r} * effStress * effStrain * L, V{0}));
  }

  if (evalRes) {
    // q is exactly zero in inactive lanes, so the residual contribution vanishes there.
    for (int i = 0; i < kSpaceDim; ++i) {
      V const resComponent = effStress * q[i];
      (*outRes)[i] -= resComponent;
      (*outRes)[kNumRodFields + i] += resComponent;
    }
  }

  if (evalDRes) {
    V const materialCoeff = Select(isActive, effEA * invL, V{0});
    V geometricCoeff = effStress * invL;
    if (projectPsd) {
      V const eps{materials::kMinProjectedEigenvalue};
      // The geometric tangent is geometricCoeff * I, so its PSD projection reduces to flooring its
      // repeated eigenvalue at eps. Written as a comparison rather than Max(geometricCoeff, eps)
      // since Max has undefined NaN behavior, so it would silently swallow NaNs in active lanes.
      // Inactive lanes are masked below.
      geometricCoeff = Select(geometricCoeff < eps, eps, geometricCoeff);
    }
    geometricCoeff = Select(isActive, geometricCoeff, V{0});

    // Symmetric 3x3 tangent: material outer(q, q) plus the isotropic geometric term.
    V3 const materialTimesQ = materialCoeff * q;
    ScatterRodAxialTangentComponent<0, 0>(materialTimesQ[0] * q[0] + geometricCoeff, *outDRes);
    ScatterRodAxialTangentComponent<1, 1>(materialTimesQ[1] * q[1] + geometricCoeff, *outDRes);
    ScatterRodAxialTangentComponent<2, 2>(materialTimesQ[2] * q[2] + geometricCoeff, *outDRes);
    ScatterRodAxialTangentComponent<0, 1>(materialTimesQ[0] * q[1], *outDRes);
    ScatterRodAxialTangentComponent<0, 2>(materialTimesQ[0] * q[2], *outDRes);
    ScatterRodAxialTangentComponent<1, 2>(materialTimesQ[1] * q[2], *outDRes);
  }

  return true;
}

template <int kBatchSize>
bool RodBendTwistStress(
    Span<Real3 const> meshNodes,
    Span<Real3 const> frameAxes,
    Span<Real3 const> referenceAxes,
    BatchRodVector<kBatchSize> const& disp,
    NdArray<int, kBatchSize> const& elementIndices,
    Span<int const> l2gFlat,
    BatchDouble<kBatchSize>* outEnergy,
    BatchRodVector<kBatchSize>* outRes,
    BatchRodMatrix<kBatchSize>* outDRes,
    Real2 flexuralStiffness,
    real torsionalStiffness,
    real stiffnessDampingFactor,
    NdArray<BatchReal<kBatchSize>, kNumRodStencilDofs> const* stageStartDisp,
    Span<Real3 const> stageStartFrameAxes) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  using V3x3 = BatchReal3x3<kBatchSize>;

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = (outRes != nullptr);
  bool const evalDRes = (outDRes != nullptr);
  if (!evalObj && !evalRes && !evalDRes) {
    return false;
  }

  bool const hasDamping =
      (stiffnessDampingFactor > 0_r) && (stageStartDisp != nullptr) && !stageStartFrameAxes.empty();
  MOCHI_ASSERT_VERBOSE(
      stiffnessDampingFactor == 0_r || (stageStartDisp != nullptr && !stageStartFrameAxes.empty()),
      "Nonzero stiffnessDampingFactor requires stageStartDisp and stageStartFrameAxes.");

  constexpr int kDofs = kNumRodStencilDofs;

  // --- Per-lane stencil node indices + boundary mask (3 distinct nodes required) ---
  alignas(alignof(V)) real staging[V::kSize]{};
  NdArray<int, kBatchSize> node0Idx MOCHI_NO_INIT;
  NdArray<int, kBatchSize> node1Idx MOCHI_NO_INIT;
  NdArray<int, kBatchSize> node2Idx MOCHI_NO_INIT;
  for (int b = 0; b < kBatchSize; ++b) {
    node0Idx[b] = details::RodStencilNodeIndex(l2gFlat, elementIndices[b], 0);
    node1Idx[b] = details::RodStencilNodeIndex(l2gFlat, elementIndices[b], 1);
    node2Idx[b] = details::RodStencilNodeIndex(l2gFlat, elementIndices[b], 2);
    bool const distinct =
        node0Idx[b] != node1Idx[b] && node1Idx[b] != node2Idx[b] && node0Idx[b] != node2Idx[b];
    staging[b] = distinct ? 1_r : 0_r;
  }
  V const isActive = (Load<V>(staging) > V{0});
  if (!AnyTrue(isActive))
    MOCHI_UNLIKELY {
      return false;
    }

  // --- Gather reference node positions and current/reference frame axes ---
  NdArray<V3, kNumRodStencilNodes> X MOCHI_NO_INIT;
  for (int n = 0; n < kNumRodStencilNodes; ++n) {
    for (int d = 0; d < 3; ++d) {
      for (int b = 0; b < kBatchSize; ++b) {
        int const node = (n == 0) ? node0Idx[b] : (n == 1) ? node1Idx[b] : node2Idx[b];
        staging[b] = meshNodes[node][d];
      }
      X[n][d] = Load<V>(staging);
    }
  }
  // Frame axes and reference axes are stored per edge (one entry per element, indexed by the edge's
  // first node), so they are shorter than the node array. Boundary (collapsed) stencils are masked
  // out below, but their gathered node indices (node 0 and node 1, the two edge-first nodes) can
  // point one past the last edge; clamp the gather index to stay in bounds (the gathered value is
  // discarded for those inactive lanes).
  int const lastFrameAxis = isize(frameAxes) - 1;
  V3 a0 MOCHI_NO_INIT, a1 MOCHI_NO_INIT;
  for (int d = 0; d < 3; ++d) {
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = frameAxes[Min(node0Idx[b], lastFrameAxis)][d];
    }
    a0[d] = Load<V>(staging);
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = frameAxes[Min(node1Idx[b], lastFrameAxis)][d];
    }
    a1[d] = Load<V>(staging);
  }

  // --- Kinematics ---
  NdArray<V3, kNumRodStencilNodes> x MOCHI_NO_INIT;
  for (int n = 0; n < kNumRodStencilNodes; ++n) {
    for (int d = 0; d < 3; ++d) {
      x[n][d] = X[n][d] + disp[n * kNumRodFields + d];
    }
  }
  V3 const e0 = x[1] - x[0];
  V3 const e1 = x[2] - x[1];
  V const e0Norm = Max(Norm(e0), V{std::numeric_limits<real>::min()});
  V const e1Norm = Max(Norm(e1), V{std::numeric_limits<real>::min()});
  V3 const e0Hat = (V{1} / e0Norm) * e0;
  V3 const e1Hat = (V{1} / e1Norm) * e1;
  V3 const E0 = X[1] - X[0];
  V3 const E1 = X[2] - X[1];
  V const E0Norm = Max(Norm(E0), V{std::numeric_limits<real>::min()});
  V const E1Norm = Max(Norm(E1), V{std::numeric_limits<real>::min()});

  // Averaged tangent at the central node.
  V3 const eSum = e0Hat + e1Hat;
  V3 const eAvg = Normalize(eSum);

  // Parallel-transport current axes to the central node for finite differencing that mimics a
  // covariant derivative, i.e. deviation from parallel transport.
  V3x3 const paraTrans0Node = details::ParallelTransportOperator(e0Hat, eAvg);
  V3x3 const paraTrans1Node = details::ParallelTransportOperator(e1Hat, eAvg);
  V3 const a0Node = DotMatVec(paraTrans0Node, a0);
  V3 const a1Node = DotMatVec(paraTrans1Node, a1);
  V3 const b0Node = Cross(eAvg, a0Node);
  V3 const b1Node = Cross(eAvg, a1Node);

  // Averaged current material frame (a, b).
  V3 const aSum = a0Node + a1Node;
  V3 const bSum = b0Node + b1Node;
  V3 const aAvg = Normalize(aSum);
  V3 const bAvg = Normalize(bSum);

  // Averaged reference/current edge lengths and curvature binormals.
  V const L = V{0.5_r} * (E0Norm + E1Norm);
  V const l = V{0.5_r} * (e0Norm + e1Norm);
  V const Linv = V{1} / L;
  V const L2inv = Linv * Linv;
  V const lOverL2 = l * L2inv;
  V3 const kInt = IntegratedCurvatureBinormal(e0Hat, e1Hat);
  V3 const k = lOverL2 * kInt;
  V kaStrain MOCHI_NO_INIT;
  V kbStrain MOCHI_NO_INIT;
  V twistStrain MOCHI_NO_INIT;
  // Stiffness damping unified into the elastic formulas: scale each material stiffness by
  // (1 + factor) and use modified strains ε̃ = ε − (factor/(1+factor))·ε₀ (ε₀ = stage-start
  // strain). Feeding (eff*, eff*Strain) through the standard undamped formulas reproduces the
  // damped stress k·[ε + factor·(ε − ε₀)] and its exact tangent, since the response is linear in
  // the strains. When damping is off, eff* = k and the modified strains equal the plain strains.
  V effKa MOCHI_NO_INIT;
  V effKb MOCHI_NO_INIT;
  V effTwist MOCHI_NO_INIT;
  V const EI1 = V{flexuralStiffness[0]};
  V const EI2 = V{flexuralStiffness[1]};
  V const GJ = V{torsionalStiffness};
  real const effFactor = hasDamping ? stiffnessDampingFactor : 0_r;
  V const effEI1 = V{1_r + effFactor} * EI1;
  V const effEI2 = V{1_r + effFactor} * EI2;
  V const effGJ = V{1_r + effFactor} * GJ;
  real const ssWeight = effFactor > 0_r ? effFactor / (1_r + effFactor) : 0_r;

  // Helper computing the current-config projected curvatures/twist (ka, kb, twist) for a given
  // displacement stencil and per-edge frame axes. Used to re-evaluate the stage-start state for
  // stiffness damping. The reference projections (KA, KB, reference twist) are identical for the
  // current and stage-start states, so they cancel in the strain deltas and are not recomputed.
  auto computeBendTwistProjections = [&](auto const& dispStencil,
                                         V3 const& a0In,
                                         V3 const& a1In,
                                         V& outKa,
                                         V& outKb,
                                         V& outTwist) {
    NdArray<V3, kNumRodStencilNodes> xLoc MOCHI_NO_INIT;
    for (int n = 0; n < kNumRodStencilNodes; ++n) {
      for (int d = 0; d < 3; ++d) {
        xLoc[n][d] = X[n][d] + dispStencil[n * kNumRodFields + d];
      }
    }
    V3 const e0L = xLoc[1] - xLoc[0];
    V3 const e1L = xLoc[2] - xLoc[1];
    V const e0NormL = Max(Norm(e0L), V{std::numeric_limits<real>::min()});
    V const e1NormL = Max(Norm(e1L), V{std::numeric_limits<real>::min()});
    V3 const e0HatL = (V{1} / e0NormL) * e0L;
    V3 const e1HatL = (V{1} / e1NormL) * e1L;
    V3 const eAvgL = Normalize(e0HatL + e1HatL);
    V3 const a0NodeL = DotMatVec(details::ParallelTransportOperator(e0HatL, eAvgL), a0In);
    V3 const a1NodeL = DotMatVec(details::ParallelTransportOperator(e1HatL, eAvgL), a1In);
    V3 const aAvgL = Normalize(a0NodeL + a1NodeL);
    V3 const bAvgL = Normalize(Cross(eAvgL, a0NodeL) + Cross(eAvgL, a1NodeL));
    V const lL = V{0.5_r} * (e0NormL + e1NormL);
    V3 const kLoc = (lL * L2inv) * IntegratedCurvatureBinormal(e0HatL, e1HatL);
    outKa = Dot(kLoc, aAvgL);
    outKb = Dot(kLoc, bAvgL);
    outTwist = Dot(IntegratedCurvatureBinormal(a0NodeL, a1NodeL), eAvgL);
  };

  if (evalObj || evalRes) {
    V3 A0 MOCHI_NO_INIT, A1 MOCHI_NO_INIT;
    int const lastReferenceAxis = isize(referenceAxes) - 1;
    for (int d = 0; d < 3; ++d) {
      for (int b = 0; b < kBatchSize; ++b) {
        staging[b] = referenceAxes[Min(node0Idx[b], lastReferenceAxis)][d];
      }
      A0[d] = Load<V>(staging);
      for (int b = 0; b < kBatchSize; ++b) {
        staging[b] = referenceAxes[Min(node1Idx[b], lastReferenceAxis)][d];
      }
      A1[d] = Load<V>(staging);
    }

    V3 const E0Hat = (V{1} / E0Norm) * E0;
    V3 const E1Hat = (V{1} / E1Norm) * E1;
    V3 const EAvg = Normalize(E0Hat + E1Hat);
    V3 const A0Node = DotMatVec(details::ParallelTransportOperator(E0Hat, EAvg), A0);
    V3 const A1Node = DotMatVec(details::ParallelTransportOperator(E1Hat, EAvg), A1);
    V3 const B0Node = Cross(EAvg, A0Node);
    V3 const B1Node = Cross(EAvg, A1Node);
    V3 const AAvg = Normalize(A0Node + A1Node);
    V3 const BAvg = Normalize(B0Node + B1Node);

    // Project curvature binormals onto the material axes.
    V3 const K = Linv * IntegratedCurvatureBinormal(E0Hat, E1Hat);
    V const KA = Dot(K, AAvg);
    V const KB = Dot(K, BAvg);
    V const ka = Dot(k, aAvg);
    V const kb = Dot(k, bAvg);

    // Twist: reuse the binormal-curvature formula in finite difference to robustly block >180°
    // relative twist between adjacent elements.
    V3 const a01Binorm = IntegratedCurvatureBinormal(a0Node, a1Node);
    V const dCurrentTwist = Dot(a01Binorm, eAvg);
    V const dReferenceTwist = Dot(IntegratedCurvatureBinormal(A0Node, A1Node), EAvg);

    // Strains.
    kaStrain = ka - KA;
    kbStrain = kb - KB;
    twistStrain = dCurrentTwist - dReferenceTwist;

    // Stiffness-damping modified strains (equal to the plain strains when damping is disabled).
    effKa = kaStrain;
    effKb = kbStrain;
    effTwist = twistStrain;
    if (hasDamping) {
      // Gather the stage-start per-edge frame axes (same indexing/clamping as the current axes).
      int const lastStageStartAxis = isize(stageStartFrameAxes) - 1;
      V3 a0Ss MOCHI_NO_INIT, a1Ss MOCHI_NO_INIT;
      for (int d = 0; d < 3; ++d) {
        for (int b = 0; b < kBatchSize; ++b) {
          staging[b] = stageStartFrameAxes[Min(node0Idx[b], lastStageStartAxis)][d];
        }
        a0Ss[d] = Load<V>(staging);
        for (int b = 0; b < kBatchSize; ++b) {
          staging[b] = stageStartFrameAxes[Min(node1Idx[b], lastStageStartAxis)][d];
        }
        a1Ss[d] = Load<V>(staging);
      }
      V kaSs MOCHI_NO_INIT, kbSs MOCHI_NO_INIT, twistSs MOCHI_NO_INIT;
      computeBendTwistProjections(*stageStartDisp, a0Ss, a1Ss, kaSs, kbSs, twistSs);
      // Stage-start strains reuse the same reference projections (KA/KB/dReferenceTwist).
      effKa = kaStrain - V{ssWeight} * (kaSs - KA);
      effKb = kbStrain - V{ssWeight} * (kbSs - KB);
      effTwist = twistStrain - V{ssWeight} * (twistSs - dReferenceTwist);
    }

    if (evalObj) {
      // Standard quadratic in the modified strains; equals the elastic + dissipation potential up
      // to an x-independent constant (ε₀ is fixed within a stage), immaterial to the solver.
      V const bendEnergy = V{0.5_r} * (effEI1 * Sqr(effKa) + effEI2 * Sqr(effKb)) * L;
      V const twistEnergy = V{0.5_r} * effGJ * Sqr(effTwist) * Linv;
      *outEnergy += StaticCast<Vd>(Select(isActive, bendEnergy + twistEnergy, V{0}));
    }
  }

  if (!evalRes && !evalDRes) {
    return true;
  }

  // --- Derivatives (orthogonality-simplified) ---
  V3x3 const de0Hat_de0 = DNormalize(e0);
  V3x3 const de1Hat_de1 = DNormalize(e1);
  V3x3 const deAvg_deSum = DNormalize(eSum);

  // Simplified using orthogonality of the frame axis and the deformed tangent.
  V3x3 const da0_de0Hat = Outer(a0, e0Hat) - Outer(e0Hat, a0);
  V3x3 const da1_de1Hat = Outer(a1, e1Hat) - Outer(e1Hat, a1);
  V3 const da0_dtheta0 = Cross(e0Hat, a0);
  V3 const da1_dtheta1 = Cross(e1Hat, a1);

  auto const [dPT0_dn0_a0, dPT0_dn_a0] = details::DParallelTransportedVec(e0Hat, eAvg, a0);
  auto const [dPT1_dn0_a1, dPT1_dn_a1] = details::DParallelTransportedVec(e1Hat, eAvg, a1);
  V3x3 const da0Node_de1Hat = Dot(dPT0_dn_a0, deAvg_deSum);
  V3x3 const da1Node_de0Hat = Dot(dPT1_dn_a1, deAvg_deSum);
  V3x3 const da0Node_de0Hat = dPT0_dn0_a0 + da0Node_de1Hat + Dot(paraTrans0Node, da0_de0Hat);
  V3x3 const da1Node_de1Hat = dPT1_dn0_a1 + da1Node_de0Hat + Dot(paraTrans1Node, da1_de1Hat);
  V3 const da0Node_dtheta0 = DotMatVec(paraTrans0Node, da0_dtheta0);
  V3 const da1Node_dtheta1 = DotMatVec(paraTrans1Node, da1_dtheta1);

  V3x3 const dbNode_daNode = Skew(eAvg);
  V3x3 const db0Node_deAvg = -Skew(a0Node);
  V3x3 const db1Node_deAvg = -Skew(a1Node);
  V3x3 const db0Node_deSum = Dot(db0Node_deAvg, deAvg_deSum);
  V3x3 const db1Node_deSum = Dot(db1Node_deAvg, deAvg_deSum);
  V3x3 const db0Node_de0Hat = Dot(dbNode_daNode, da0Node_de0Hat) + db0Node_deSum;
  V3x3 const db0Node_de1Hat = Dot(dbNode_daNode, da0Node_de1Hat) + db0Node_deSum;
  V3x3 const db1Node_de0Hat = Dot(dbNode_daNode, da1Node_de0Hat) + db1Node_deSum;
  V3x3 const db1Node_de1Hat = Dot(dbNode_daNode, da1Node_de1Hat) + db1Node_deSum;
  V3 const db0Node_dtheta0 = DotMatVec(dbNode_daNode, da0Node_dtheta0);
  V3 const db1Node_dtheta1 = DotMatVec(dbNode_daNode, da1Node_dtheta1);

  V3x3 const daAvg_daSum = DNormalize(aSum);
  V3x3 const dbAvg_dbSum = DNormalize(bSum);
  V3x3 const daAvg_de0Hat = Dot(daAvg_daSum, da0Node_de0Hat + da1Node_de0Hat);
  V3x3 const daAvg_de1Hat = Dot(daAvg_daSum, da0Node_de1Hat + da1Node_de1Hat);
  V3x3 const dbAvg_de0Hat = Dot(dbAvg_dbSum, db0Node_de0Hat + db1Node_de0Hat);
  V3x3 const dbAvg_de1Hat = Dot(dbAvg_dbSum, db0Node_de1Hat + db1Node_de1Hat);
  V3 const daAvg_dtheta0 = DotMatVec(daAvg_daSum, da0Node_dtheta0);
  V3 const daAvg_dtheta1 = DotMatVec(daAvg_daSum, da1Node_dtheta1);
  V3 const dbAvg_dtheta0 = DotMatVec(dbAvg_dbSum, db0Node_dtheta0);
  V3 const dbAvg_dtheta1 = DotMatVec(dbAvg_dbSum, db1Node_dtheta1);

  V3x3 const daAvg_de0 = Dot(daAvg_de0Hat, de0Hat_de0);
  V3x3 const daAvg_de1 = Dot(daAvg_de1Hat, de1Hat_de1);
  V3x3 const dbAvg_de0 = Dot(dbAvg_de0Hat, de0Hat_de0);
  V3x3 const dbAvg_de1 = Dot(dbAvg_de1Hat, de1Hat_de1);

  auto const [dkInt_de0Hat, dkInt_de1Hat] = DIntegratedCurvatureBinormalDVecs(e0Hat, e1Hat);
  V3 const kIntOverL2Half = (V{0.5_r} * L2inv) * kInt;
  V3x3 const dk_de0 = Dot(lOverL2 * dkInt_de0Hat, de0Hat_de0) + Outer(kIntOverL2Half, e0Hat);
  V3x3 const dk_de1 = Dot(lOverL2 * dkInt_de1Hat, de1Hat_de1) + Outer(kIntOverL2Half, e1Hat);

  V3 const dka_de0 = DotVecMat(aAvg, dk_de0) + DotVecMat(k, daAvg_de0);
  V3 const dka_de1 = DotVecMat(aAvg, dk_de1) + DotVecMat(k, daAvg_de1);
  V3 const dkb_de0 = DotVecMat(bAvg, dk_de0) + DotVecMat(k, dbAvg_de0);
  V3 const dkb_de1 = DotVecMat(bAvg, dk_de1) + DotVecMat(k, dbAvg_de1);
  V const dka_dtheta0 = Dot(k, daAvg_dtheta0);
  V const dka_dtheta1 = Dot(k, daAvg_dtheta1);
  V const dkb_dtheta0 = Dot(k, dbAvg_dtheta0);
  V const dkb_dtheta1 = Dot(k, dbAvg_dtheta1);

  auto const [da01Binorm_da0Node, da01Binorm_da1Node] =
      DIntegratedCurvatureBinormalDVecs(a0Node, a1Node);
  V3x3 const da01Binorm_de0Hat =
      Dot(da01Binorm_da0Node, da0Node_de0Hat) + Dot(da01Binorm_da1Node, da1Node_de0Hat);
  V3x3 const da01Binorm_de1Hat =
      Dot(da01Binorm_da1Node, da1Node_de1Hat) + Dot(da01Binorm_da0Node, da0Node_de1Hat);
  V3 const da01Binorm_dtheta0 = DotMatVec(da01Binorm_da0Node, da0Node_dtheta0);
  V3 const da01Binorm_dtheta1 = DotMatVec(da01Binorm_da1Node, da1Node_dtheta1);
  // The second term of d(Dot(a01Binorm, eAvg))/d(eHat) — i.e., DotVecMat(a01Binorm, deAvg_deSum) —
  // is zero, because a01Binorm is parallel to eAvg and deAvg_deSum projects out the component along
  // eAvg.
  V3 const ddCurrentTwist_de0Hat = DotVecMat(eAvg, da01Binorm_de0Hat);
  V3 const ddCurrentTwist_de1Hat = DotVecMat(eAvg, da01Binorm_de1Hat);
  V3 const ddCurrentTwist_de0 = DotVecMat(ddCurrentTwist_de0Hat, de0Hat_de0);
  V3 const ddCurrentTwist_de1 = DotVecMat(ddCurrentTwist_de1Hat, de1Hat_de1);
  V const ddCurrentTwist_dtheta0 = Dot(da01Binorm_dtheta0, eAvg);
  V const ddCurrentTwist_dtheta1 = Dot(da01Binorm_dtheta1, eAvg);

  // Assemble derivatives w.r.t. the first 11 DoFs (the last node's twist DoF has zero derivative).
  constexpr int kActiveDofs = kDofs - 1;
  NdArray<V, kActiveDofs> dkaStrain_dDofs MOCHI_NO_INIT;
  NdArray<V, kActiveDofs> dkbStrain_dDofs MOCHI_NO_INIT;
  NdArray<V, kActiveDofs> ddCurrentTwist_dDofs MOCHI_NO_INIT;
  for (int i = 0; i < 3; ++i) {
    // Node 0 displacement.
    dkaStrain_dDofs[i] = -dka_de0[i];
    dkbStrain_dDofs[i] = -dkb_de0[i];
    ddCurrentTwist_dDofs[i] = -ddCurrentTwist_de0[i];
    // Node 1 displacement.
    dkaStrain_dDofs[i + kNumRodFields] = dka_de0[i] - dka_de1[i];
    dkbStrain_dDofs[i + kNumRodFields] = dkb_de0[i] - dkb_de1[i];
    ddCurrentTwist_dDofs[i + kNumRodFields] = ddCurrentTwist_de0[i] - ddCurrentTwist_de1[i];
    // Node 2 displacement.
    dkaStrain_dDofs[i + 2 * kNumRodFields] = dka_de1[i];
    dkbStrain_dDofs[i + 2 * kNumRodFields] = dkb_de1[i];
    ddCurrentTwist_dDofs[i + 2 * kNumRodFields] = ddCurrentTwist_de1[i];
  }
  // Twist angles (node 0 at kRodThetaDofOffset, node 1 at kNumRodFields + kRodThetaDofOffset).
  dkaStrain_dDofs[kRodThetaDofOffset] = dka_dtheta0;
  dkbStrain_dDofs[kRodThetaDofOffset] = dkb_dtheta0;
  ddCurrentTwist_dDofs[kRodThetaDofOffset] = ddCurrentTwist_dtheta0;
  dkaStrain_dDofs[kNumRodFields + kRodThetaDofOffset] = dka_dtheta1;
  dkbStrain_dDofs[kNumRodFields + kRodThetaDofOffset] = dkb_dtheta1;
  ddCurrentTwist_dDofs[kNumRodFields + kRodThetaDofOffset] = ddCurrentTwist_dtheta1;

  if (evalRes) {
    V const dbendA = effEI1 * effKa * L;
    V const dbendB = effEI2 * effKb * L;
    V const dtwist = effGJ * effTwist * Linv;
    for (int i = 0; i < kActiveDofs; ++i) {
      (*outRes)[i] += Select(
          isActive,
          dbendA * dkaStrain_dDofs[i] + dbendB * dkbStrain_dDofs[i] +
              dtwist * ddCurrentTwist_dDofs[i],
          V{0});
    }
  }

  if (evalDRes) {
    // Quadratic (automatically-PSD) "material stiffness" approximation, by analogy to how bending
    // is handled in shell and to geometrically-nonlinear constraints. Exploit symmetry: compute the
    // upper triangle and mirror. The effective stiffnesses (1 + factor)·k give the exact tangent of
    // the stiffness-damped residual, since this Gauss–Newton tangent omits geometric stiffness and
    // the damped stress is linear in the strains.
    V const EI1L = effEI1 * L;
    V const EI2L = effEI2 * L;
    V const tsOverL = effGJ * Linv;
    for (int i = 0; i < kActiveDofs; ++i) {
      V const sA_i = EI1L * dkaStrain_dDofs[i];
      V const sB_i = EI2L * dkbStrain_dDofs[i];
      V const tw_i = tsOverL * ddCurrentTwist_dDofs[i];
      for (int j = i; j < kActiveDofs; ++j) {
        V const val = Select(
            isActive,
            sA_i * dkaStrain_dDofs[j] + sB_i * dkbStrain_dDofs[j] + tw_i * ddCurrentTwist_dDofs[j],
            V{0});
        (*outDRes)[i * kDofs + j] += val;
        if (i != j) {
          (*outDRes)[j * kDofs + i] += val;
        }
      }
    }
  }

  return true;
}

static_assert(kDefaultFemBatchSize != 1, "Avoid duplicate explicit instantiations.");

#define MOCHI_INSTANTIATE_ROD_AXIAL_STRESS(BatchSize) \
  template bool RodAxialStress<BatchSize>(            \
      Span<Real3 const>,                              \
      BatchRodVector<BatchSize> const&,               \
      NdArray<int, BatchSize> const&,                 \
      Span<int const>,                                \
      BatchDouble<BatchSize>*,                        \
      BatchRodVector<BatchSize>*,                     \
      BatchRodMatrix<BatchSize>*,                     \
      real,                                           \
      bool,                                           \
      real,                                           \
      NdArray<BatchReal<BatchSize>, kNumRodStencilDofs> const*);

#define MOCHI_INSTANTIATE_ROD_BEND_TWIST_STRESS(BatchSize)      \
  template bool RodBendTwistStress<BatchSize>(                  \
      Span<Real3 const>,                                        \
      Span<Real3 const>,                                        \
      Span<Real3 const>,                                        \
      BatchRodVector<BatchSize> const&,                         \
      NdArray<int, BatchSize> const&,                           \
      Span<int const>,                                          \
      BatchDouble<BatchSize>*,                                  \
      BatchRodVector<BatchSize>*,                               \
      BatchRodMatrix<BatchSize>*,                               \
      Real2,                                                    \
      real,                                                     \
      real,                                                     \
      NdArray<BatchReal<BatchSize>, kNumRodStencilDofs> const*, \
      Span<Real3 const>);

MOCHI_INSTANTIATE_ROD_AXIAL_STRESS(1)
MOCHI_INSTANTIATE_ROD_AXIAL_STRESS(kDefaultFemBatchSize)

MOCHI_INSTANTIATE_ROD_BEND_TWIST_STRESS(1)
MOCHI_INSTANTIATE_ROD_BEND_TWIST_STRESS(kDefaultFemBatchSize)

#undef MOCHI_INSTANTIATE_ROD_AXIAL_STRESS
#undef MOCHI_INSTANTIATE_ROD_BEND_TWIST_STRESS

} // namespace mochi::fem
