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

#include <mochi_core/element_operations/fem_shell.h>

#include <mochi_core/utils/batch_config.h>

namespace mochi::fem {

// Regularization constants of the membrane collapse barrier (see ShellWork). Its energy and
// residual are exact until g ≈ gMax = 2¹⁶ in single precision (2¹²⁸ in double), and its tangent
// adds ε = gMax⁻⁵ to u⁵. This bounds the residual and tangent by O(k·gMax⁴) and O(k·gMax⁵), far
// from overflow, and the energy by 4k·A·gMax³.
static real constexpr kMembraneCollapseBarrierInvMaxG =
    MOCHI_USE_DOUBLE_PRECISION ? 0x1p-128_r : 0x1p-16_r;
static real constexpr kMembraneCollapseBarrierDResEps =
    Sqr(Sqr(kMembraneCollapseBarrierInvMaxG)) * kMembraneCollapseBarrierInvMaxG;

/// @brief Gather 6-node bending stencil positions into batched layout.
///
/// @warning Sentinel nodes (globalIdx == kSentinelIndex) are mapped to meshNodes[0], producing
/// invalid positions. Callers must follow with @ref ExtrapolateStencilPositions to overwrite them.
template <int kBatchSize>
static MOCHI_FORCE_INLINE void GatherStencilPositions(
    Span<Real3 const> meshNodes,
    NdArray<int, kBendingStencilNodes, kBatchSize> const& stencilGlobalNodes,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs>& outPos) {
  using V = BatchReal<kBatchSize>;
  alignas(alignof(V)) real staging[V::kSize * kSpaceDim3]{};
  for (int n = 0; n < kBendingStencilNodes; ++n) {
    for (int b = 0; b < kBatchSize; ++b) {
      int const globalIdx = stencilGlobalNodes[n][b];
      int const safeIdx = (globalIdx != kSentinelIndex) ? globalIdx : 0;
      for (int d = 0; d < kSpaceDim3; ++d) {
        staging[b * kSpaceDim3 + d] = meshNodes[safeIdx][d];
      }
    }
    LoadTransposed<V::kSize>(
        staging,
        outPos[n * kSpaceDim3 + 0],
        outPos[n * kSpaceDim3 + 1],
        outPos[n * kSpaceDim3 + 2]);
  }
}

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
    real membraneCollapseBarrierStiffness,
    real membraneStrainStiffeningStiffness,
    bool projectPsd,
    real stiffnessDampingFactor,
    NdArray<BatchReal<kBatchSize>, kBendingStencilDofs> const* stageStartDisp) {
  // TODO: Consider adding early exits for zero material parameters (hasMembrane/hasBending flags)
  // to skip membrane or bending computation entirely.
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V2x2 = BatchReal2x2<kBatchSize>;

  bool const evalObj = (outEnergy != nullptr);
  bool const evalRes = (outRes != nullptr);
  bool const evalDRes = (outDRes != nullptr);
  MOCHI_ASSERT_VERBOSE(evalObj || evalRes || evalDRes, "Must assemble something.");

  bool const hasDamping = (stiffnessDampingFactor > 0_r) && (stageStartDisp != nullptr);
  MOCHI_ASSERT_VERBOSE(
      stiffnessDampingFactor == 0_r || stageStartDisp != nullptr,
      "Nonzero stiffnessDampingFactor requires stageStartDisp.");

  V const vLambda{membraneLambda}, vMu{membraneMu}, vAlpha{bendingAlpha}, vBeta{bendingBeta};

  // Stiffness damping unified into the elastic formulas: scale the material stiffnesses by
  // (1 + factor) and use modified strains ε̃ = ε − (factor/(1+factor))·ε₀ (ε₀ = stage-start
  // strain). Feeding (eff*, modified strain) through the standard undamped formulas reproduces the
  // damped stress σ = k·[ε + factor·(ε − ε₀)] and its exact tangent, since the SVK response is
  // linear in both strain and the stiffness parameters. When damping is off, eff* = k and the
  // modified strains equal the plain strains.
  real const effFactor = hasDamping ? stiffnessDampingFactor : 0_r;
  V const effLambda = V{1_r + effFactor} * vLambda;
  V const effMu = V{1_r + effFactor} * vMu;
  V const effAlpha = V{1_r + effFactor} * vAlpha;
  V const effBeta = V{1_r + effFactor} * vBeta;
  real const ssWeight = effFactor > 0_r ? effFactor / (1_r + effFactor) : 0_r;

  // Gather and extrapolate positions.
  // NOTE: Sentinel hinge positions are invalid after gathering (mapped to an arbitrary mesh node)
  // and must be overwritten by ExtrapolateStencilPositions before use.
  NdArray<V, kBendingStencilDofs> refPosX MOCHI_NO_INIT;
  GatherStencilPositions<kBatchSize>(meshNodes, stencilGlobalNodes, refPosX);
  // Save pre-extrapolation reference positions for stage-start bending curvature computation.
  NdArray<V, kBendingStencilDofs> refPosXPreExtrap MOCHI_NO_INIT;
  if (hasDamping) {
    refPosXPreExtrap = refPosX;
  }

  NdArray<V, kBendingStencilDofs> curPosX = refPosX + disp;
  ExtrapolateStencilPositions<kBatchSize>(stencilGlobalNodes, refPosX);
  ExtrapolateStencilPositions<kBatchSize>(stencilGlobalNodes, curPosX);

  // Reference-configuration midsurface metric tensor. Clamp detA to avoid 1/0 → NaN in padding
  // lanes where all positions are zero (det = 0).
  V2x2 const A = Metric<kBatchSize>(refPosX);
  V const detA = Max(Det(A), V{std::numeric_limits<real>::min()});
  V const refArea = V{0.5_r} * Sqrt(detA);
  V2x2 const AInv = Invert(A, detA);

  // Fully-covariant associated tensor of Green–Lagrange strain (all indices lowered).
  V2x2 const epsilonFlat = MembraneStrain<kBatchSize>(refPosX, disp);
  // Mid-surface Green–Lagrange strain (first index raised, second index lowered).
  V2x2 const epsilon = Dot(AInv, epsilonFlat);

  // Modified membrane strain for stiffness damping (equals epsilon when damping is off).
  V2x2 epsilonEff = epsilon;
  if (hasDamping) {
    V2x2 const epsilonStageStartFlat = MembraneStrain<kBatchSize>(refPosX, *stageStartDisp);
    V2x2 const epsilonStageStart = Dot(AInv, epsilonStageStartFlat);
    epsilonEff = epsilon - V{ssWeight} * epsilonStageStart;
  }

  // Compute current edge vectors (shared by energy, residual, and derivative paths).
  auto const curEdges = EdgeVectors<kBatchSize>(curPosX);

  // When both b and db_dv are needed from curEdges, use fused function to share normals.
  bool const needCurB = (evalObj || evalRes);
  bool const needDerivs = (evalRes || evalDRes);

  V2x2 curB MOCHI_NO_INIT;
  NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3> db_dv_edges
      MOCHI_NO_INIT;
  if (needCurB && needDerivs) {
    SecondFundamentalFormAndDEdges<kBatchSize>(curEdges, curB, db_dv_edges);
  } else if (needCurB) {
    curB = SecondFundamentalForm<kBatchSize>(curEdges);
  } else {
    db_dv_edges = DSecondFundamentalFormDEdges<kBatchSize>(curEdges);
  }

  // Bending curvature.
  // NOTE: Some of these quantities may be needed for the tangent evaluation as well if a more
  // complex nonlinear bending model and/or geometric stiffness are added in the future.
  V2x2 s MOCHI_NO_INIT;
  V2x2 sEff MOCHI_NO_INIT;
  if (needCurB) {
    // TODO: The second fundamental forms are independent of the unknown current solution and could
    // be precomputed.
    auto const refEdges = EdgeVectors<kBatchSize>(refPosX);
    auto const B = SecondFundamentalForm<kBatchSize>(refEdges);
    V2x2 const kappa = B - curB;
    s = Dot(AInv, kappa);
    sEff = s;

    if (hasDamping) {
      // Stage-start bending curvature.
      NdArray<V, kBendingStencilDofs> curPosXStageStart = refPosXPreExtrap + *stageStartDisp;
      ExtrapolateStencilPositions<kBatchSize>(stencilGlobalNodes, curPosXStageStart);
      auto const curEdgesStageStart = EdgeVectors<kBatchSize>(curPosXStageStart);
      auto const curBStageStart = SecondFundamentalForm<kBatchSize>(curEdgesStageStart);
      V2x2 const kappaStageStart = B - curBStageStart;
      V2x2 const sStageStart = Dot(AInv, kappaStageStart);
      sEff = s - V{ssWeight} * sStageStart;
    }
  }

  if (evalObj) {
    // Standard quadratic in the modified strains with effective stiffnesses; equals the elastic +
    // dissipation potential up to an x-independent constant (ε₀ is fixed within a stage), which is
    // immaterial to the solver.
    *outEnergy += (PsiSVK<kBatchSize>(epsilonEff, effLambda, effMu) +
                   PsiSVK<kBatchSize>(sEff, effAlpha, V{0.5_r} * effBeta)) *
        StaticCast<Vd>(refArea);
  }

  NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3> db_dx MOCHI_NO_INIT;
  NdArray<BatchSymMatrix2x2<kBatchSize>, kTriangleNodes, kSpaceDim3> da_dx MOCHI_NO_INIT;
  V2x2 dpsi_da MOCHI_NO_INIT;
  BatchSymMatrix3x3<kBatchSize> d2psi_deps2 MOCHI_NO_INIT;
  if (needDerivs) {
    db_dx = DSecondFundamentalFormDx<kBatchSize>(db_dv_edges, stencilGlobalNodes);
    da_dx = DMetricDx<kBatchSize>(curPosX);
    dpsi_da = V{0.5_r} * DPsiSVK<kBatchSize>(epsilonEff, AInv, effLambda, effMu);
  }
  if (evalDRes) {
    // Material tangent with the effective stiffnesses (1 + factor)·k. Since the SVK tangent is
    // linear in the stiffness parameters, it and the geometric term at the effective stress dpsi_da
    // give the exact Hessian of the unified SVK objective.
    d2psi_deps2 = D2PsiSVKSym2x2<kBatchSize>(AInv, effLambda, effMu);
  }

  // Membrane strain stiffening k·Σᵢ⟨εᵢ⟩³, with its exact (PSD) stiffness. Evaluated at the plain
  // strain with unscaled k, so stiffness damping remains based on the SVK reference stiffness
  // alone.
  if (membraneStrainStiffeningStiffness > 0_r) {
    auto const positivePart = PositivePart<kBatchSize>(epsilon);
    V const vK{membraneStrainStiffeningStiffness};
    if (evalObj) {
      *outEnergy += PsiStrainStiffening<kBatchSize>(positivePart, vK) * StaticCast<Vd>(refArea);
    }
    if (needDerivs) {
      dpsi_da = dpsi_da + V{0.5_r} * DPsiStrainStiffening<kBatchSize>(positivePart, AInv, vK);
    }
    if (evalDRes) {
      d2psi_deps2 += D2PsiStrainStiffeningSym2x2<kBatchSize>(positivePart, AInv, vK);
    }
  }

  // Residual-only membrane stress; dpsi_da alone also drives the geometric stiffness.
  V2x2 membraneResStress MOCHI_NO_INIT;
  if (needDerivs) {
    membraneResStress = dpsi_da;
  }

  // Membrane collapse barrier k·⟨g⟩³, g = tr(C⁻¹) − 2. Evaluated at the plain strain with unscaled
  // k, so stiffness damping remains based on the SVK reference stiffness alone.
  if (membraneCollapseBarrierStiffness > 0_r) {
    // With mixed C = A⁻¹a = I + 2ε, g = −2 tr(adj(C)·ε) / det C. Taking adj(C) from the current
    // metric and det C = |e₀ × e₁|² / det A from the current edges keeps g accurate both near the
    // reference configuration (small ε) and near collapse (small C).
    V2x2 const C = Dot(AInv, Metric<kBatchSize>(curPosX));
    V2x2 const adjC = Transpose(Cofactor(C));
    V const detC = NormSqr(Cross(curEdges[0], curEdges[2])) / detA;

    // Normalizing by tr C, with u = det C / tr C = 1 / tr(C⁻¹) and Ĉ = adj(C) / tr C,
    //   g = p / u,  p = max(−2 tr(adj(C)·ε) / tr C, 0) = max(1 − 2u, 0) ∈ [0, 1],
    //   dg/da = −C⁻¹ C⁻¹ A⁻¹ = M̂ / u²,  M̂ = −Ĉ Ĉ A⁻¹,
    // where p and M̂ stay bounded (Ĉ has unit trace) and u ≈ 1 / g → 0 near collapse. Collapse to
    // a point (C = 0) makes p = 0/0, but u ≤ tr C / 4 gives p → 1 along every path, so p saturates
    // wherever the tr C floor is active. This also covers padding lanes, whose outputs are unused.
    V const trC = Trace(C);
    V const trCFloor{std::numeric_limits<real>::min()};
    V const invTrC = V{1_r} / Max(trC, trCFloor);
    V const u = detC * invTrC;
    V const p =
        Select(trC >= trCFloor, Max(V{-2_r} * Trace(Dot(adjC, epsilon)) * invTrC, V{0_r}), V{1_r});
    V const vK{membraneCollapseBarrierStiffness};

    // The exact energy k·A·p³/u³, stress 3k·p²·M̂/u⁴, and Gauss–Newton tangent
    // 6k·A·p/u⁵ · (M̂ : ∂a/∂x)⊗(M̂ : ∂a/∂x) diverge as u → 0. Below u₀ = 1/gMax, the energy
    // continues along its tangent line in u: with ū = max(u, u₀) and p̄ = p(ū), it is
    // k·A·(p̄³/ū³ + 3p̄²·(ū − u)/ū⁴), and the stress 3k·p̄²·M̂/ū⁴ is its exact derivative while the
    // tr C and det A floors are inactive. The tangent instead adds ε to u⁵, staying PSD but not
    // exact.
    V const u0{kMembraneCollapseBarrierInvMaxG};
    V const pBar = Select(u >= u0, p, V{1_r - 2_r * kMembraneCollapseBarrierInvMaxG});
    if (evalObj) {
      // Evaluated as k·A·ḡ²·(ḡ + 3(ū − u)/ū²) with ḡ = p̄/ū and ū − u set to 0 for u ≥ u₀, which
      // keeps inactive lanes zero when u overflows and avoids intermediate underflow for tiny k·A.
      Vd const uD = StaticCast<Vd>(u);
      Vd const u0D{static_cast<double>(kMembraneCollapseBarrierInvMaxG)};
      Vd const invUBarD = Vd{1.0} / StaticCast<Vd>(Max(u, u0));
      Vd const gBarD = StaticCast<Vd>(pBar) * invUBarD;
      Vd const uGapD = Select(uD >= u0D, Vd{0.0}, u0D - uD);
      *outEnergy +=
          StaticCast<Vd>(vK * refArea) * gBarD * gBarD * (gBarD + Vd{3.0} * uGapD * Sqr(invUBarD));
    }

    if (needDerivs) {
      V2x2 const adjCHat = invTrC * adjC;
      V2x2 const MHat = -Dot(Dot(adjCHat, adjCHat), AInv);
      V const u4 = Sqr(Sqr(u));
      if (evalRes) {
        V const uBar4 = Max(u4, V{Sqr(Sqr(kMembraneCollapseBarrierInvMaxG))});
        membraneResStress = membraneResStress + (V{3_r} * vK * pBar * pBar / uBar4) * MHat;
      }
      if (evalDRes) {
        AddMembraneCollapseBarrierDResidual<kBatchSize>(
            da_dx,
            MHat,
            V{6_r} * vK * refArea * p / (u4 * u + V{kMembraneCollapseBarrierDResEps}),
            *outDRes);
      }
    }
  }

  if (evalRes) {
    AddMembraneResidual<kBatchSize>(da_dx, membraneResStress, refArea, *outRes);
    AddBendingResidual<kBatchSize>(db_dx, sEff, AInv, refArea, effAlpha, effBeta, *outRes);
  }

  if (evalDRes) {
    AddMembraneDResidual<kBatchSize>(
        da_dx, dpsi_da, d2psi_deps2, AInv, refArea, projectPsd, *outDRes);
    AddBendingDResidual<kBatchSize>(db_dx, AInv, refArea, effAlpha, effBeta, *outDRes);

    // Mirror stiffness: copy upper triangle to lower triangle (node-pair blocks).
    for (int testNode = 0; testNode < kBendingStencilNodes; ++testNode) {
      for (int trialNode = testNode + 1; trialNode < kBendingStencilNodes; ++trialNode) {
        for (int i = 0; i < kSpaceDim3; ++i) {
          for (int j = 0; j < kSpaceDim3; ++j) {
            (*outDRes)
                [(trialNode * kSpaceDim3 + j) * kBendingStencilDofs + testNode * kSpaceDim3 + i] =
                    (*outDRes)
                        [(testNode * kSpaceDim3 + i) * kBendingStencilDofs +
                         trialNode * kSpaceDim3 + j];
          }
        }
      }
    }
  }

  return true;
}

static_assert(kDefaultFemBatchSize != 1, "Avoid duplicate explicit instantiations.");

#define MOCHI_INSTANTIATE_SHELL_WORK(BatchSize)                                  \
  template bool ShellWork<BatchSize>(                                            \
      NdArray<int, kBendingStencilNodes, BatchSize> const&,                      \
      Span<Real3 const>,                                                         \
      NdArray<BatchReal<BatchSize>, kBendingStencilDofs> const&,                 \
      BatchDouble<BatchSize>*,                                                   \
      NdArray<BatchReal<BatchSize>, kBendingStencilDofs>*,                       \
      NdArray<BatchReal<BatchSize>, kBendingStencilDofs * kBendingStencilDofs>*, \
      real,                                                                      \
      real,                                                                      \
      real,                                                                      \
      real,                                                                      \
      real,                                                                      \
      real,                                                                      \
      bool,                                                                      \
      real,                                                                      \
      NdArray<BatchReal<BatchSize>, kBendingStencilDofs> const*);

MOCHI_INSTANTIATE_SHELL_WORK(1)
MOCHI_INSTANTIATE_SHELL_WORK(kDefaultFemBatchSize)

#undef MOCHI_INSTANTIATE_SHELL_WORK

} // namespace mochi::fem
