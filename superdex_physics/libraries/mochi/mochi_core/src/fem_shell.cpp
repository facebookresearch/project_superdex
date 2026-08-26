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
  bool const needBendingDeriv = (evalRes || evalDRes);

  V2x2 curB MOCHI_NO_INIT;
  NdArray<BatchSymMatrix2x2<kBatchSize>, kBendingStencilNodes, kSpaceDim3> db_dv_edges
      MOCHI_NO_INIT;
  if (needCurB && needBendingDeriv) {
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
  if (needBendingDeriv) {
    db_dx = DSecondFundamentalFormDx<kBatchSize>(db_dv_edges, stencilGlobalNodes);
    da_dx = DMetricDx<kBatchSize>(curPosX);
    dpsi_da = V{0.5_r} * DPsiSVK<kBatchSize>(epsilonEff, AInv, effLambda, effMu);
  }

  if (evalRes) {
    AddMembraneResidual<kBatchSize>(da_dx, dpsi_da, refArea, *outRes);
    AddBendingResidual<kBatchSize>(db_dx, sEff, AInv, refArea, effAlpha, effBeta, *outRes);
  }

  if (evalDRes) {
    // Material tangent uses the effective stiffnesses (1 + factor)·k; the membrane geometric term
    // is evaluated at the effective stress (the new dpsi_da). Since the SVK tangent is linear in
    // the stiffness parameters, this is the exact Hessian of the unified objective.
    AddMembraneDResidual<kBatchSize>(
        da_dx, dpsi_da, AInv, refArea, effLambda, effMu, projectPsd, *outDRes);
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
      bool,                                                                      \
      real,                                                                      \
      NdArray<BatchReal<BatchSize>, kBendingStencilDofs> const*);

MOCHI_INSTANTIATE_SHELL_WORK(1)
MOCHI_INSTANTIATE_SHELL_WORK(kDefaultFemBatchSize)

#undef MOCHI_INSTANTIATE_SHELL_WORK

} // namespace mochi::fem
