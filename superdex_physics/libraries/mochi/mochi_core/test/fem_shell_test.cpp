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

#include <mochi_core/element_operations/element_operation_utils.h>
#include <mochi_core/element_operations/fem_shell.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/test/batch_helpers.h>
#include <mochi_core/test/batched_work_test_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>

#if MOCHI_USE_EIGEN
MOCHI_WARNING_PUSH()
MOCHI_WARNING_SUPPRESS_EIGEN()
#include <Eigen/Eigenvalues>
MOCHI_WARNING_POP()
#endif

using namespace mochi;
using namespace mochi::fem;
using namespace mochi::test;

// The batched shell kernel (membrane + bending over a 6-node stencil) is validated against a
// constitutive-agnostic oracle:
//   - residual is the gradient of the energy   (res ≈ ∂E/∂u, central differences)
//   - dresidual is the gradient of the residual (dRes ≈ ∂res/∂u, central differences)
//   - rest state (zero displacement, flat reference) has zero energy
//   - missing (boundary) stencil nodes contribute zero residual/dresidual (mask coverage)
//   - PSD projection yields a symmetric positive-semi-definite dresidual.
// Boundary configs (sentinel nodes) and mixed-lane batches exercise the per-lane masking that is
// the main correctness risk. Finite differences cover the assembly + masking.

static constexpr real kMembraneLambda = 100_r;
static constexpr real kMembraneMu = 50_r;
static constexpr real kBendingAlpha = 0.1_r;
static constexpr real kBendingBeta = 0.05_r;

namespace {
struct ShellStencilConfig {
  NdArray<Real3, kBendingStencilNodes> refPositions;
  NdArray<Real3, kBendingStencilNodes> displacements;
  NdArray<int, kBendingStencilNodes> globalNodeIndices;
};
} // namespace

static ShellStencilConfig MakeInteriorConfig() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0_r, 1_r, 0_r},
       Real3{1_r, 1_r, 0_r},
       Real3{-1_r, 1_r, 0_r},
       Real3{1_r, -1_r, 0_r}},
      {Real3{0.1_r, 0.2_r, 0.3_r},
       Real3{0.4_r, 0.05_r, 0.06_r},
       Real3{0.01_r, -0.02_r, 0.03_r},
       Real3{0.04_r, -0.05_r, 0.06_r},
       Real3{-0.01_r, -0.02_r, -0.03_r},
       Real3{-0.04_r, -0.05_r, -0.06_r}},
      {0, 1, 2, 3, 4, 5}};
}

static ShellStencilConfig PermuteCentralNodes(
    ShellStencilConfig const& cfg,
    std::array<int, kTriangleNodes> const& permutation) {
  ShellStencilConfig permuted{};
  for (int i = 0; i < kTriangleNodes; ++i) {
    int const oldCentralNode = permutation[i];
    permuted.refPositions[i] = cfg.refPositions[oldCentralNode];
    permuted.refPositions[i + kTriangleNodes] = cfg.refPositions[oldCentralNode + kTriangleNodes];
    permuted.displacements[i] = cfg.displacements[oldCentralNode];
    permuted.displacements[i + kTriangleNodes] = cfg.displacements[oldCentralNode + kTriangleNodes];
  }
  permuted.globalNodeIndices = {0, 1, 2, 3, 4, 5};
  return permuted;
}

static ShellStencilConfig MakeEquilateralConfig() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0.5_r, 0_r, kSqrt3Over2},
       Real3{1.5_r, 0_r, kSqrt3Over2},
       Real3{-0.5_r, 0_r, kSqrt3Over2},
       Real3{0.5_r, 0_r, -kSqrt3Over2}},
      {Real3{0.1_r, 0.5_r, 0.1_r},
       Real3{0.2_r, -0.3_r, 0.2_r},
       Real3{-0.1_r, 0.8_r, -0.1_r},
       Real3{0.3_r, 0.2_r, 0.3_r},
       Real3{-0.2_r, -0.1_r, 0.4_r},
       Real3{0.15_r, 0.4_r, -0.2_r}},
      {0, 1, 2, 3, 4, 5}};
}

static ShellStencilConfig MakeBoundaryNode3Missing() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0.5_r, 0_r, kSqrt3Over2},
       Real3{0_r, 0_r, 0_r},
       Real3{-0.5_r, 0_r, kSqrt3Over2},
       Real3{0.5_r, 0_r, -kSqrt3Over2}},
      {Real3{0.2_r, 0.5_r, 0.1_r},
       Real3{0.02_r, -0.3_r, 0.02_r},
       Real3{-0.1_r, 0.08_r, -0.01_r},
       Real3{0_r, 0_r, 0_r},
       Real3{-0.02_r, -0.01_r, 0.04_r},
       Real3{0.015_r, 0.04_r, -0.02_r}},
      {0, 1, 2, kSentinelIndex, 4, 5}};
}

static ShellStencilConfig MakeBoundaryNode4Missing() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0_r, 1_r, 0_r},
       Real3{1_r, 1_r, 0_r},
       Real3{0_r, 0_r, 0_r},
       Real3{1_r, -1_r, 0_r}},
      {Real3{0.1_r, 0.2_r, 0.3_r},
       Real3{0.4_r, 0.05_r, 0.06_r},
       Real3{0.01_r, -0.02_r, 0.03_r},
       Real3{0.04_r, -0.05_r, 0.06_r},
       Real3{0_r, 0_r, 0_r},
       Real3{-0.04_r, -0.05_r, -0.06_r}},
      {0, 1, 2, 3, kSentinelIndex, 5}};
}

static ShellStencilConfig MakeBoundaryNode5Missing() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0.5_r, 0_r, kSqrt3Over2},
       Real3{1.5_r, 0_r, kSqrt3Over2},
       Real3{-0.5_r, 0_r, kSqrt3Over2},
       Real3{0_r, 0_r, 0_r}},
      {Real3{0.1_r, 0.5_r, 0.1_r},
       Real3{0.2_r, -0.3_r, 0.2_r},
       Real3{-0.1_r, 0.8_r, -0.1_r},
       Real3{0.3_r, 0.2_r, 0.3_r},
       Real3{-0.2_r, -0.1_r, 0.4_r},
       Real3{0_r, 0_r, 0_r}},
      {0, 1, 2, 3, 4, kSentinelIndex}};
}

static ShellStencilConfig MakeBoundary34Missing() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0.5_r, 0_r, kSqrt3Over2},
       Real3{0_r, 0_r, 0_r},
       Real3{0_r, 0_r, 0_r},
       Real3{0.5_r, 0_r, -kSqrt3Over2}},
      {Real3{0.2_r, 0.5_r, 0.1_r},
       Real3{0.02_r, -0.3_r, 0.02_r},
       Real3{-0.1_r, 0.08_r, -0.01_r},
       Real3{0_r, 0_r, 0_r},
       Real3{0_r, 0_r, 0_r},
       Real3{0.015_r, 0.04_r, -0.02_r}},
      {0, 1, 2, kSentinelIndex, kSentinelIndex, 5}};
}

static ShellStencilConfig MakeBoundary35Missing() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0.5_r, 0_r, kSqrt3Over2},
       Real3{0_r, 0_r, 0_r},
       Real3{-0.5_r, 0_r, kSqrt3Over2},
       Real3{0_r, 0_r, 0_r}},
      {Real3{0.2_r, 0.5_r, 0.1_r},
       Real3{0.02_r, -0.3_r, 0.02_r},
       Real3{-0.1_r, 0.08_r, -0.01_r},
       Real3{0_r, 0_r, 0_r},
       Real3{-0.02_r, -0.01_r, 0.04_r},
       Real3{0_r, 0_r, 0_r}},
      {0, 1, 2, kSentinelIndex, 4, kSentinelIndex}};
}

static ShellStencilConfig MakeBoundary45Missing() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0.5_r, 0_r, kSqrt3Over2},
       Real3{1.5_r, 0_r, kSqrt3Over2},
       Real3{0_r, 0_r, 0_r},
       Real3{0_r, 0_r, 0_r}},
      {Real3{0.1_r, 0.5_r, 0.1_r},
       Real3{0.2_r, -0.3_r, 0.2_r},
       Real3{-0.1_r, 0.8_r, -0.1_r},
       Real3{0.3_r, 0.2_r, 0.3_r},
       Real3{0_r, 0_r, 0_r},
       Real3{0_r, 0_r, 0_r}},
      {0, 1, 2, 3, kSentinelIndex, kSentinelIndex}};
}

static ShellStencilConfig MakeBoundaryAllMissing() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0.5_r, 0_r, kSqrt3Over2},
       Real3{0_r, 0_r, 0_r},
       Real3{0_r, 0_r, 0_r},
       Real3{0_r, 0_r, 0_r}},
      {Real3{0.2_r, 0.5_r, 0.1_r},
       Real3{0.02_r, -0.3_r, 0.02_r},
       Real3{-0.1_r, 0.08_r, -0.01_r},
       Real3{0_r, 0_r, 0_r},
       Real3{0_r, 0_r, 0_r},
       Real3{0_r, 0_r, 0_r}},
      {0, 1, 2, kSentinelIndex, kSentinelIndex, kSentinelIndex}};
}

static ShellStencilConfig MakeFlatHelperConfig() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0_r, 1_r, 0_r},
       Real3{1_r, 1_r, 0_r},
       Real3{-1_r, 1_r, 0_r},
       Real3{1_r, -1_r, 0_r}},
      {Real3{1_r, 2_r, 3_r},
       Real3{4_r, 5_r, 6_r},
       Real3{1_r, -2_r, 3_r},
       Real3{4_r, -5_r, 6_r},
       Real3{-1_r, -2_r, -3_r},
       Real3{-4_r, -5_r, -6_r}},
      {0, 1, 2, 3, 4, 5}};
}

static ShellStencilConfig MakeRotatedHelperConfig() {
  real constexpr kInvSqrt2 = 1_r / kSqrt2;
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{kInvSqrt2, kInvSqrt2, 0_r},
       Real3{-kInvSqrt2, kInvSqrt2, 0_r},
       Real3{0_r, 2_r * kInvSqrt2, 0_r},
       Real3{-2_r * kInvSqrt2, 0_r, 0_r},
       Real3{2_r * kInvSqrt2, 0_r, 0_r}},
      {Real3{0.5_r, 0.5_r, 1_r},
       Real3{1_r, -0.5_r, 2_r},
       Real3{-0.5_r, 1_r, 1.5_r},
       Real3{1.5_r, 1_r, 2.5_r},
       Real3{-1_r, 0_r, 0.5_r},
       Real3{2_r, -1_r, 3_r}},
      {0, 1, 2, 3, 4, 5}};
}

static ShellStencilConfig MakeYZHelperConfig() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{0_r, 1_r, 0_r},
       Real3{0_r, 0_r, 1_r},
       Real3{0_r, 1_r, 1_r},
       Real3{0_r, -1_r, 1_r},
       Real3{0_r, 1_r, -1_r}},
      {Real3{2_r, 0.1_r, 0.1_r},
       Real3{-1_r, 0.2_r, -0.1_r},
       Real3{1.5_r, -0.1_r, 0.2_r},
       Real3{-0.5_r, 0.3_r, 0.3_r},
       Real3{0.8_r, -0.2_r, -0.2_r},
       Real3{-1.2_r, 0.15_r, 0.1_r}},
      {0, 1, 2, 3, 4, 5}};
}

static ShellStencilConfig MakeLargeDeformationHelperConfig() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{2_r, 0_r, 0_r},
       Real3{1_r, 2_r, 0_r},
       Real3{3_r, 2_r, 0_r},
       Real3{-1_r, 2_r, 0_r},
       Real3{1_r, -2_r, 0_r}},
      {Real3{5_r, 5_r, 5_r},
       Real3{-3_r, 4_r, 6_r},
       Real3{2_r, -4_r, 3_r},
       Real3{-2_r, -3_r, 4_r},
       Real3{4_r, 2_r, -2_r},
       Real3{-1_r, 3_r, -4_r}},
      {0, 1, 2, 3, 4, 5}};
}

static ShellStencilConfig MakeSmallPerturbationHelperConfig() {
  return {
      {Real3{0_r, 0_r, 0_r},
       Real3{1_r, 0_r, 0_r},
       Real3{0.5_r, 1_r, 0_r},
       Real3{1.5_r, 1_r, 0_r},
       Real3{-0.5_r, 1_r, 0_r},
       Real3{0.5_r, -1_r, 0_r}},
      {Real3{0.001_r, 0.002_r, 0.003_r},
       Real3{0.002_r, -0.001_r, 0.001_r},
       Real3{-0.001_r, 0.001_r, 0.002_r},
       Real3{0.0015_r, 0.0005_r, -0.001_r},
       Real3{-0.0005_r, -0.001_r, 0.001_r},
       Real3{0.001_r, 0.002_r, -0.0015_r}},
      {0, 1, 2, 3, 4, 5}};
}

static std::array<ShellStencilConfig, 6> MakeHelperConfigs() {
  return {
      MakeFlatHelperConfig(),
      MakeEquilateralConfig(),
      MakeRotatedHelperConfig(),
      MakeYZHelperConfig(),
      MakeLargeDeformationHelperConfig(),
      MakeSmallPerturbationHelperConfig()};
}

static std::array<ShellStencilConfig, 14> MakeShellWorkFdConfigs() {
  return {
      MakeInteriorConfig(),
      MakeEquilateralConfig(),
      MakeBoundaryNode3Missing(),
      MakeBoundaryNode4Missing(),
      MakeBoundaryNode5Missing(),
      MakeBoundary34Missing(),
      MakeBoundary35Missing(),
      MakeBoundary45Missing(),
      MakeBoundaryAllMissing(),
      MakeFlatHelperConfig(),
      MakeRotatedHelperConfig(),
      MakeYZHelperConfig(),
      MakeLargeDeformationHelperConfig(),
      MakeSmallPerturbationHelperConfig()};
}

static ShellStencilConfig WithBoundary35(ShellStencilConfig cfg) {
  cfg.globalNodeIndices = {0, 1, 2, kSentinelIndex, 4, kSentinelIndex};
  return cfg;
}

static NdArray<Real3, kBendingStencilNodes> HelperDirection() {
  return {
      Real3{1_r, 2_r, 3_r},
      Real3{4_r, 5_r, 6_r},
      Real3{7_r, 8_r, 9_r},
      Real3{10_r, 11_r, 12_r},
      Real3{13_r, 14_r, 15_r},
      Real3{16_r, 17_r, 18_r}};
}

template <int kBS>
using ShellBatchVector = NdArray<BatchReal<kBS>, kBendingStencilDofs>;

template <int kBS>
using ShellBatchMatrix = NdArray<BatchReal<kBS>, kBendingStencilDofs * kBendingStencilDofs>;

template <int kBS>
using ShellBatchStencil = NdArray<int, kBendingStencilNodes, kBS>;

namespace {
struct ShellFdOptions {
  bool checkEnergyResidual = true;
  bool checkResidualDResidual = true;
  bool checkPsdProjection = false;
  real dispScale = 1_r;
  real fdStep = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;
  real membraneLambda = kMembraneLambda;
  real membraneMu = kMembraneMu;
  real bendingAlpha = kBendingAlpha;
  real bendingBeta = kBendingBeta;
  real membraneCollapseBarrierStiffness = 0_r;
  real membraneStrainStiffeningStiffness = 0_r;
  real stiffnessDampingFactor = 0_r;
  real stageStartDispScale = 0_r; // Scale for stage-start displacements (0 = no damping data).
};
} // namespace

template <int kBS>
static ShellBatchVector<kBS> BroadcastNodeValues(
    NdArray<Real3, kBendingStencilNodes> const& values) {
  ShellBatchVector<kBS> out{};
  for (int dof = 0; dof < kBendingStencilDofs; ++dof) {
    out[dof] = BatchReal<kBS>{values[dof / kSpaceDim3][dof % kSpaceDim3]};
  }
  return out;
}

template <int kBS>
static NdArray<BatchReal3<kBS>, kBendingStencilNodes> BroadcastNodeTriples(
    NdArray<Real3, kBendingStencilNodes> const& values) {
  NdArray<BatchReal3<kBS>, kBendingStencilNodes> out{};
  for (int n = 0; n < kBendingStencilNodes; ++n) {
    for (int d = 0; d < kSpaceDim3; ++d) {
      out[n][d] = BatchReal<kBS>{values[n][d]};
    }
  }
  return out;
}

template <int kBS>
static ShellBatchStencil<kBS> BroadcastStencilNodes(
    NdArray<int, kBendingStencilNodes> const& nodes) {
  ShellBatchStencil<kBS> out;
  for (int n = 0; n < kBendingStencilNodes; ++n) {
    for (int b = 0; b < kBS; ++b) {
      out[n][b] = nodes[n];
    }
  }
  return out;
}

template <int kBS>
static ShellBatchVector<kBS> ExtrapolatedPositions(ShellStencilConfig const& cfg) {
  auto pos = BroadcastNodeValues<kBS>(cfg.refPositions);
  ExtrapolateStencilPositions<kBS>(BroadcastStencilNodes<kBS>(cfg.globalNodeIndices), pos);
  return pos;
}

template <int kBS>
static void MirrorNodePairBlocks(ShellBatchMatrix<kBS>& dres) {
  for (int testNode = 0; testNode < kBendingStencilNodes; ++testNode) {
    for (int trialNode = testNode + 1; trialNode < kBendingStencilNodes; ++trialNode) {
      for (int i = 0; i < kSpaceDim3; ++i) {
        for (int j = 0; j < kSpaceDim3; ++j) {
          dres[(trialNode * kSpaceDim3 + j) * kBendingStencilDofs + testNode * kSpaceDim3 + i] =
              dres[(testNode * kSpaceDim3 + i) * kBendingStencilDofs + trialNode * kSpaceDim3 + j];
        }
      }
    }
  }
}

namespace {
// Runs ShellWork for a batch of (per-lane) configs and a given displacement.
template <int kBS>
struct ShellHarness {
  DynamicArray<Real3> meshNodes;
  NdArray<int, kBendingStencilNodes, kBS> stencilGlobalNodes;
  NdArray<BatchReal<kBS>, kBendingStencilDofs> baseDisp{};
  ShellFdOptions fdOptions;

  explicit ShellHarness(
      std::array<ShellStencilConfig const*, kBS> const& cfgs,
      ShellFdOptions options = {})
      : fdOptions(options) {
    using V = BatchReal<kBS>;
    meshNodes.resize(kBS * kBendingStencilNodes);
    for (int b = 0; b < kBS; ++b) {
      int const offset = b * kBendingStencilNodes;
      for (int n = 0; n < kBendingStencilNodes; ++n) {
        int const idx = cfgs[b]->globalNodeIndices[n];
        if (idx != kSentinelIndex) {
          stencilGlobalNodes[n][b] = idx + offset;
          meshNodes[idx + offset] = cfgs[b]->refPositions[n];
        } else {
          stencilGlobalNodes[n][b] = kSentinelIndex;
        }
      }
    }
    alignas(alignof(V)) real staging[V::kSize]{};
    for (int dof = 0; dof < kBendingStencilDofs; ++dof) {
      for (int b = 0; b < kBS; ++b) {
        staging[b] =
            fdOptions.dispScale * cfgs[b]->displacements[dof / kSpaceDim3][dof % kSpaceDim3];
      }
      baseDisp[dof] = Load<V>(staging);
    }
    // Stage-start displacements for stiffness damping tests.
    if (fdOptions.stiffnessDampingFactor > 0_r) {
      stageStartDisp.emplace();
      for (int dof = 0; dof < kBendingStencilDofs; ++dof) {
        for (int b = 0; b < kBS; ++b) {
          staging[b] = fdOptions.stageStartDispScale *
              cfgs[b]->displacements[dof / kSpaceDim3][dof % kSpaceDim3];
        }
        (*stageStartDisp)[dof] = Load<V>(staging);
      }
    }
  }

  void Run(
      NdArray<BatchReal<kBS>, kBendingStencilDofs> const& disp,
      BatchDouble<kBS>* energy,
      NdArray<BatchReal<kBS>, kBendingStencilDofs>* res,
      NdArray<BatchReal<kBS>, kBendingStencilDofs * kBendingStencilDofs>* dres,
      bool projectPsd) const {
    ShellWork<kBS>(
        stencilGlobalNodes,
        MakeConstSpan(meshNodes),
        disp,
        energy,
        res,
        dres,
        fdOptions.membraneLambda,
        fdOptions.membraneMu,
        fdOptions.bendingAlpha,
        fdOptions.bendingBeta,
        fdOptions.membraneCollapseBarrierStiffness,
        fdOptions.membraneStrainStiffeningStiffness,
        projectPsd,
        fdOptions.stiffnessDampingFactor,
        stageStartDisp ? &*stageStartDisp : nullptr);
  }

  std::optional<NdArray<BatchReal<kBS>, kBendingStencilDofs>> stageStartDisp;
};

template <int kBS>
struct ShellOutputs {
  BatchDouble<kBS> energy{};
  NdArray<BatchReal<kBS>, kBendingStencilDofs> res{};
  NdArray<BatchReal<kBS>, kBendingStencilDofs * kBendingStencilDofs> dres{};
};
}; // namespace

template <int kBS>
static ShellOutputs<kBS>
RunShell(ShellHarness<kBS> const& h, OutputConfig cfg, bool projectPsd = false) {
  ShellOutputs<kBS> out;
  h.Run(
      h.baseDisp,
      cfg.energy ? &out.energy : nullptr,
      cfg.residual ? &out.res : nullptr,
      cfg.dresidual ? &out.dres : nullptr,
      projectPsd);
  return out;
}

template <int kBS>
static void ExpectNear2x2Lanes(
    BatchReal2x2<kBS> const& ref,
    BatchReal2x2<kBS> const& actual,
    real relTol,
    real absTol) {
  for (int b = 0; b < kBS; ++b) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        real const r = ref[i][j][b];
        EXPECT_NEAR_TOL(r, actual[i][j][b], Max(absTol, relTol * Abs(r)));
      }
    }
  }
}

template <typename T>
static NdArray<T, 2, 2> ToMatrix(NdArray<T, 3> const& sym) {
  return SymMatrix2x2(sym[0], sym[1], sym[2]);
}

template <int kBS>
static void ExpectNearVectorLanes(
    ShellBatchVector<kBS> const& ref,
    ShellBatchVector<kBS> const& actual,
    real relTol) {
  for (int b = 0; b < kBS; ++b) {
    ExpectNearL2(GetLane(ref, b), GetLane(actual, b), relTol);
  }
}

template <int kBS>
static void ExpectNearMatrixLanes(
    ShellBatchMatrix<kBS> const& ref,
    ShellBatchMatrix<kBS> const& actual,
    real relTol) {
  for (int b = 0; b < kBS; ++b) {
    ExpectNearL2(GetLane(ref, b), GetLane(actual, b), relTol);
  }
}

template <int kBS>
static BatchDouble<kBS> MembraneEnergy(
    ShellBatchVector<kBS> const& refPos,
    ShellBatchVector<kBS> const& disp) {
  using V = BatchReal<kBS>;
  using Vd = BatchDouble<kBS>;
  auto const A = Metric<kBS>(refPos);
  V const detA = Det(A);
  V const referenceArea = V{0.5_r} * Sqrt(detA);
  auto const AInv = Invert(A, detA);
  auto const epsilon = Dot(AInv, MembraneStrain<kBS>(refPos, disp));
  return PsiSVK<kBS>(epsilon, V{1_r}, V{2_r}) * StaticCast<Vd>(referenceArea);
}

template <int kBS>
static ShellBatchVector<kBS> MembraneResidual(
    ShellBatchVector<kBS> const& refPos,
    ShellBatchVector<kBS> const& disp) {
  using V = BatchReal<kBS>;
  auto const A = Metric<kBS>(refPos);
  V const detA = Det(A);
  V const referenceArea = V{0.5_r} * Sqrt(detA);
  auto const AInv = Invert(A, detA);
  auto const epsilon = Dot(AInv, MembraneStrain<kBS>(refPos, disp));
  auto const da_dx = DMetricDx<kBS>(refPos + disp);
  auto const dpsi_da = V{0.5_r} * DPsiSVK<kBS>(epsilon, AInv, V{1_r}, V{2_r});

  ShellBatchVector<kBS> res{};
  AddMembraneResidual<kBS>(da_dx, dpsi_da, referenceArea, res);
  return res;
}

template <int kBS>
static ShellBatchMatrix<kBS> MembraneDResidual(
    ShellBatchVector<kBS> const& refPos,
    ShellBatchVector<kBS> const& disp) {
  using V = BatchReal<kBS>;
  auto const A = Metric<kBS>(refPos);
  V const detA = Det(A);
  V const referenceArea = V{0.5_r} * Sqrt(detA);
  auto const AInv = Invert(A, detA);
  auto const epsilon = Dot(AInv, MembraneStrain<kBS>(refPos, disp));
  auto const da_dx = DMetricDx<kBS>(refPos + disp);
  auto const dpsi_da = V{0.5_r} * DPsiSVK<kBS>(epsilon, AInv, V{1_r}, V{2_r});

  ShellBatchMatrix<kBS> dres{};
  AddMembraneDResidual<kBS>(
      da_dx,
      dpsi_da,
      D2PsiSVKSym2x2<kBS>(AInv, V{1_r}, V{2_r}),
      AInv,
      referenceArea,
      /*projectPsd*/ false,
      dres);
  MirrorNodePairBlocks<kBS>(dres);
  return dres;
}

template <int kBS>
static BatchDouble<kBS> BendingEnergy(
    ShellBatchVector<kBS> const& refPos,
    ShellBatchVector<kBS> const& disp) {
  using V = BatchReal<kBS>;
  using Vd = BatchDouble<kBS>;
  auto const A = Metric<kBS>(refPos);
  V const detA = Det(A);
  V const referenceArea = V{0.5_r} * Sqrt(detA);
  auto const AInv = Invert(A, detA);
  auto const B = SecondFundamentalForm<kBS>(EdgeVectors<kBS>(refPos));
  auto const b = SecondFundamentalForm<kBS>(EdgeVectors<kBS>(refPos + disp));
  auto const s = Dot(AInv, B - b);
  return PsiSVK<kBS>(s, V{1_r}, V{1_r}) * StaticCast<Vd>(referenceArea);
}

template <int kBS>
static ShellBatchVector<kBS> BendingResidual(
    ShellBatchStencil<kBS> const& stencilGlobalNodes,
    ShellBatchVector<kBS> const& refPos,
    ShellBatchVector<kBS> const& disp) {
  using V = BatchReal<kBS>;
  auto const A = Metric<kBS>(refPos);
  V const detA = Det(A);
  V const referenceArea = V{0.5_r} * Sqrt(detA);
  auto const AInv = Invert(A, detA);
  auto const B = SecondFundamentalForm<kBS>(EdgeVectors<kBS>(refPos));
  auto const curEdges = EdgeVectors<kBS>(refPos + disp);
  auto const b = SecondFundamentalForm<kBS>(curEdges);
  auto const s = Dot(AInv, B - b);
  auto const db_dv = DSecondFundamentalFormDEdges<kBS>(curEdges);
  auto const db_dx = DSecondFundamentalFormDx<kBS>(db_dv, stencilGlobalNodes);

  ShellBatchVector<kBS> res{};
  AddBendingResidual<kBS>(db_dx, s, AInv, referenceArea, V{1_r}, V{2_r}, res);
  return res;
}

template <int kBS>
static ShellBatchMatrix<kBS> BendingDResidual(
    ShellBatchStencil<kBS> const& stencilGlobalNodes,
    ShellBatchVector<kBS> const& refPos,
    ShellBatchVector<kBS> const& disp) {
  using V = BatchReal<kBS>;
  auto const A = Metric<kBS>(refPos);
  V const detA = Det(A);
  V const referenceArea = V{0.5_r} * Sqrt(detA);
  auto const AInv = Invert(A, detA);
  auto const curEdges = EdgeVectors<kBS>(refPos + disp);
  auto const db_dv = DSecondFundamentalFormDEdges<kBS>(curEdges);
  auto const db_dx = DSecondFundamentalFormDx<kBS>(db_dv, stencilGlobalNodes);

  ShellBatchMatrix<kBS> dres{};
  AddBendingDResidual<kBS>(db_dx, AInv, referenceArea, V{1_r}, V{2_r}, dres);
  MirrorNodePairBlocks<kBS>(dres);
  return dres;
}

template <int kBS>
static void VerifyMetricAndStrainHelpers() {
  using V = BatchReal<kBS>;
  real constexpr kMetricEps = 1e-5_r;
  real constexpr kMetricTol = 1e-2_r;
  // Because DMetricDx is linear, we should be able to take arbitrarily large finite difference
  // steps here and still expect exactness.
  real constexpr kD2MetricEps = 1_r;
  real constexpr kD2MetricTol = 1e-5_r;

  for (auto const& cfg : MakeHelperConfigs()) {
    auto const refPos = ExtrapolatedPositions<kBS>(cfg);
    auto const disp = BroadcastNodeValues<kBS>(cfg.displacements);
    auto const direction = BroadcastNodeValues<kBS>(HelperDirection());

    auto const strainFromDelta = MembraneStrain<kBS>(refPos, disp);
    auto const strainFromMetric = V{0.5_r} * (Metric<kBS>(refPos + disp) - Metric<kBS>(refPos));
    ExpectNear2x2Lanes<kBS>(strainFromMetric, strainFromDelta, kMetricTol, kMetricTol);

    auto const da_dx = DMetricDx<kBS>(refPos);
    auto const metricFd = (V{1_r / kMetricEps}) *
        (Metric<kBS>(refPos + V{kMetricEps} * direction) - Metric<kBS>(refPos));
    BatchSymMatrix2x2<kBS> metricDirectional{};
    for (int node = 0; node < kTriangleNodes; ++node) {
      for (int d = 0; d < kSpaceDim3; ++d) {
        metricDirectional += da_dx[node][d] * direction[node * kSpaceDim3 + d];
      }
    }
    ExpectNear2x2Lanes<kBS>(metricFd, ToMatrix(metricDirectional), kMetricTol, kMetricTol);

    auto const d2a_dx2 = D2MetricDx2<kBS>();
    auto const da_dxPerturbed = DMetricDx<kBS>(refPos + V{kD2MetricEps} * direction);
    for (int testNode = 0; testNode < kTriangleNodes; ++testNode) {
      for (int d = 0; d < kSpaceDim3; ++d) {
        auto const fd =
            (V{1_r / kD2MetricEps}) * (da_dxPerturbed[testNode][d] - da_dx[testNode][d]);
        BatchSymMatrix2x2<kBS> directional{};
        for (int trialNode = 0; trialNode < kTriangleNodes; ++trialNode) {
          directional += d2a_dx2[testNode][trialNode] * direction[trialNode * kSpaceDim3 + d];
        }
        ExpectNear2x2Lanes<kBS>(ToMatrix(fd), ToMatrix(directional), kD2MetricTol, kD2MetricTol);
      }
    }
  }
}

template <int kBS>
static void VerifyBendingEdgeGeometryHelpers() {
  using V = BatchReal<kBS>;
  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-4_r;
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-3_r : 5e-3_r;

  for (auto const& cfg : MakeHelperConfigs()) {
    auto const pos = ExtrapolatedPositions<kBS>(cfg);
    auto const edges = EdgeVectors<kBS>(pos);
    auto const edgeDirection = BroadcastNodeTriples<kBS>(HelperDirection());
    auto const db_dv = DSecondFundamentalFormDEdges<kBS>(edges);

    BatchReal2x2<kBS> fusedB{};
    NdArray<BatchSymMatrix2x2<kBS>, kBendingStencilNodes, kSpaceDim3> fusedDbDv{};
    SecondFundamentalFormAndDEdges<kBS>(edges, fusedB, fusedDbDv);
    ExpectNear2x2Lanes<kBS>(SecondFundamentalForm<kBS>(edges), fusedB, kTol, kTol);
    for (int edge = 0; edge < kBendingStencilNodes; ++edge) {
      for (int d = 0; d < kSpaceDim3; ++d) {
        ExpectNear2x2Lanes<kBS>(ToMatrix(db_dv[edge][d]), ToMatrix(fusedDbDv[edge][d]), kTol, kTol);
      }
    }

    auto edgesPerturbed = edges;
    for (int edge = 0; edge < kBendingStencilNodes; ++edge) {
      for (int d = 0; d < kSpaceDim3; ++d) {
        edgesPerturbed[edge][d] += V{kEps} * edgeDirection[edge][d];
      }
    }
    auto const db_dvFd = (V{1_r / kEps}) *
        (SecondFundamentalForm<kBS>(edgesPerturbed) - SecondFundamentalForm<kBS>(edges));
    BatchSymMatrix2x2<kBS> db_dvDirectional{};
    for (int edge = 0; edge < kBendingStencilNodes; ++edge) {
      for (int d = 0; d < kSpaceDim3; ++d) {
        db_dvDirectional += db_dv[edge][d] * edgeDirection[edge][d];
      }
    }
    ExpectNear2x2Lanes<kBS>(db_dvFd, ToMatrix(db_dvDirectional), kTol, 0_r);
  }
}

template <int kBS>
static void VerifyBendingPositionGeometryHelpers() {
  using V = BatchReal<kBS>;
  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-4_r;
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-3_r : 5e-3_r;
  real constexpr kDbDxAbsTol = MOCHI_USE_DOUBLE_PRECISION ? kTol : 1e-2_r;

  for (auto const& cfg : MakeHelperConfigs()) {
    auto const boundaryCfg = WithBoundary35(cfg);
    auto const rawPos = BroadcastNodeValues<kBS>(boundaryCfg.refPositions);
    auto const originalPos = BroadcastNodeValues<kBS>(cfg.refPositions);
    auto const stencilGlobalNodes = BroadcastStencilNodes<kBS>(boundaryCfg.globalNodeIndices);
    auto extrapolatedPos = rawPos;
    ExtrapolateStencilPositions<kBS>(stencilGlobalNodes, extrapolatedPos);
    ExpectNearVectorLanes<kBS>(originalPos, extrapolatedPos, kTol);

    auto const extrapolatedEdges = EdgeVectors<kBS>(extrapolatedPos);
    auto const db_dx = DSecondFundamentalFormDx<kBS>(
        DSecondFundamentalFormDEdges<kBS>(extrapolatedEdges), stencilGlobalNodes);
    auto perturbedPos = rawPos + V{kEps} * BroadcastNodeValues<kBS>(HelperDirection());
    ExtrapolateStencilPositions<kBS>(stencilGlobalNodes, perturbedPos);
    auto const db_dxFd = (V{1_r / kEps}) *
        (SecondFundamentalForm<kBS>(EdgeVectors<kBS>(perturbedPos)) -
         SecondFundamentalForm<kBS>(extrapolatedEdges));
    BatchSymMatrix2x2<kBS> db_dxDirectional{};
    auto const direction = BroadcastNodeValues<kBS>(HelperDirection());
    for (int node = 0; node < kBendingStencilNodes; ++node) {
      for (int d = 0; d < kSpaceDim3; ++d) {
        db_dxDirectional += db_dx[node][d] * direction[node * kSpaceDim3 + d];
      }
    }
    ExpectNear2x2Lanes<kBS>(db_dxFd, ToMatrix(db_dxDirectional), 0_r, kDbDxAbsTol);
  }
}

template <int kBS>
static void VerifySvkHelpers() {
  using V = BatchReal<kBS>;
  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-4_r;
  real constexpr kTol = 1e-3_r;

  BatchReal2x2<kBS> const AInv = SymMatrix2x2(V{2_r}, V{-1_r}, V{3_r});
  V const lambda{2_r};
  V const mu{1_r};
  auto const strainFlat = SymMatrix2x2(V{1_r}, V{2_r}, V{3_r});
  auto const direction = SymMatrix2x2(V{4_r}, V{5_r}, V{6_r});
  auto const strain = Dot(AInv, strainFlat);
  auto const strainPerturbed = Dot(AInv, strainFlat + V{kEps} * direction);

  auto const psi = PsiSVK<kBS>(strain, lambda, mu);
  auto const psiPerturbed = PsiSVK<kBS>(strainPerturbed, lambda, mu);
  auto const dpsi = DPsiSVK<kBS>(strain, AInv, lambda, mu);
  auto const dpsiDirectional = Colon(dpsi, direction);
  for (int b = 0; b < kBS; ++b) {
    real const fd = static_cast<real>((psiPerturbed[b] - psi[b]) / static_cast<double>(kEps));
    EXPECT_NEAR_TOL(fd, dpsiDirectional[b], kTol * Abs(fd));
  }

  auto const d2psiSym = D2PsiSVKSym2x2<kBS>(AInv, lambda, mu);
  std::array<BatchReal2x2<kBS>, 3> const tangentDirections{
      SymMatrix2x2(V{1_r}, V{0_r}, V{0_r}),
      // The tangent doubles the raw shear component, so 0.5 is its unit coordinate.
      SymMatrix2x2(V{0_r}, V{0.5_r}, V{0_r}),
      SymMatrix2x2(V{0_r}, V{0_r}, V{1_r})};
  NdArray<BatchSymMatrix2x2<kBS>, 3> tangentColumns MOCHI_NO_INIT;
  for (int column = 0; column < 3; ++column) {
    auto const perturbedStrain = V{kEps} * Dot(AInv, tangentDirections[column]);
    auto const dpsiPerturbed = DPsiSVK<kBS>(perturbedStrain, AInv, lambda, mu);
    auto const dpsiColumn = V{1_r / kEps} * dpsiPerturbed;
    tangentColumns[column] = Sym2x2Components(dpsiColumn);
    auto const applied = fem::details::ApplySym2x2Tangent<kBS>(
        d2psiSym, Sym2x2Components(tangentDirections[column]));
    ExpectNear2x2Lanes<kBS>(dpsiColumn, ToMatrix(applied), kTol, 0_r);
  }

  BatchSymMatrix3x3<kBS> const expectedD2psiSym{
      tangentColumns[0][0],
      tangentColumns[1][1],
      tangentColumns[2][2],
      tangentColumns[1][0],
      tangentColumns[2][0],
      tangentColumns[2][1]};
  for (int b = 0; b < kBS; ++b) {
    ExpectNearL2(GetLane(expectedD2psiSym, b), GetLane(d2psiSym, b), kTol);
  }
}

// Principal strains ε₀ ≥ ε₁ of each sign combination, including equal and zero principal strains
// and one near zero on either side of a sign change.
static constexpr Real2 kPositivePartPrincipalStrains[] = {
    {0.3_r, 0.1_r},
    {0.2_r, 0.2_r},
    {0.25_r, 0_r},
    {0.3_r, 1e-4_r},
    {0.3_r, -1e-4_r},
    {0.3_r, -0.2_r},
    {0.2_r, -0.2_r},
    {0.1_r, -0.4_r},
    {1e-4_r, -0.3_r},
    {0_r, -0.25_r},
    {-0.1_r, -0.3_r},
    {-0.2_r, -0.2_r},
    {0_r, 0_r}};
static constexpr int kNumPositivePartCases = isize(kPositivePartPrincipalStrains);

// Square root L of the non-diagonal metric A = L² of MixedStrainWithPrincipalStrains.
static constexpr Matrix2x2r kMixedStrainMetricRoot{Real2{1.2_r, 0.3_r}, Real2{0.3_r, 0.8_r}};

// L⁻¹·(ε₀·r₀⊗r₀ + ε₁·r₁⊗r₁)·L for orthonormal rᵢ at the given angle, which is the mixed form A⁻¹·S
// of the covariant strain S = L·(ε₀·r₀⊗r₀ + ε₁·r₁⊗r₁)·L in the non-diagonal metric A = L². It has
// eigenvalues εᵢ and eigenprojectors L⁻¹·rᵢ⊗rᵢ·L.
static Matrix2x2r MixedStrainWithPrincipalStrains(Real2 const& principal, real angle) {
  Matrix2x2r const& L = kMixedStrainMetricRoot;
  Real2 const r0{Cos(angle), Sin(angle)};
  Real2 const r1{-r0[1], r0[0]};
  return Dot(Invert(L), Dot(principal[0] * Outer(r0, r0) + principal[1] * Outer(r1, r1), L));
}

template <int kBS>
static void VerifyPositivePartHelper() {
  real constexpr kTol = 64_r * std::numeric_limits<real>::epsilon();
  auto const angle = [](int c) { return 0.3_r + 0.7_r * StaticCast<real>(c); };
  auto const positive = [](Real2 const& e) { return Real2{Max(e[0], 0_r), Max(e[1], 0_r)}; };
  auto const decompose = [&](real scaling, int first) {
    std::array<Matrix2x2r, kBS> strains{};
    for (int i = 0; i < kBS; ++i) {
      int const c = (first + i) % kNumPositivePartCases;
      strains[i] =
          MixedStrainWithPrincipalStrains(scaling * kPositivePartPrincipalStrains[c], angle(c));
    }
    return PositivePart<kBS>(LoadBatchMatrix2x2<kBS>(MakeConstSpan(strains)));
  };

  // Each lane holds a different case.
  for (real scaling : {1e-12_r, 1e-6_r, 1_r, 1e6_r, 1e12_r}) {
    for (int first = 0; first < kNumPositivePartCases; first += kBS) {
      auto const part = decompose(scaling, first);
      for (int i = 0; i < kBS; ++i) {
        int const c = (first + i) % kNumPositivePartCases;
        Real2 const e = scaling * kPositivePartPrincipalStrains[c];
        SCOPED_TRACE(testing::Message() << "principal strains = {" << e[0] << ", " << e[1] << "}");
        real const tol = kTol * Max(Abs(e[0]), Abs(e[1]));
        EXPECT_NEAR(part.e0[i], e[0], tol);
        EXPECT_NEAR(part.e1[i], e[1], tol);
        EXPECT_NEAR_TOL(
            GetLane(part.epsPlus, i), MixedStrainWithPrincipalStrains(positive(e), angle(c)), tol);

        if (e[0] > 0_r && e[1] < 0_r) {
          real const r = e[0] / (e[0] - e[1]);
          EXPECT_NEAR(part.a[i], r, kTol);
          EXPECT_NEAR(part.b[i], -r * e[1], tol);
          EXPECT_NEAR_TOL(
              GetLane(part.projector, i),
              MixedStrainWithPrincipalStrains({1_r, 0_r}, angle(c)),
              kTol);
        } else {
          // With a zero principal strain, rounding may select either neighboring case, which give
          // the same positive part.
          EXPECT_NEAR(part.a[i], e[1] >= 0_r ? 1_r : 0_r, kTol);
          EXPECT_NEAR(part.b[i], 0_r, tol);
          if (e[0] * e[1] > 0_r) {
            EXPECT_EQ(0_r, Norm(GetLane(part.projector, i)));
          }
        }
      }
    }
  }

  // The outputs stay finite and the weights in range where the determinant is subnormal.
  real const tinyScaling = 0.1_r * Sqrt(std::numeric_limits<real>::min());
  for (int first = 0; first < kNumPositivePartCases; first += kBS) {
    auto const part = decompose(tinyScaling, first);
    for (int i = 0; i < kBS; ++i) {
      EXPECT_TRUE(IsFinite(part.e0[i]) && IsFinite(part.e1[i]));
      EXPECT_TRUE(IsFinite(GetLane(part.epsPlus, i)) && IsFinite(GetLane(part.projector, i)));
      EXPECT_TRUE(part.a[i] >= 0_r && part.a[i] <= 1_r);
      EXPECT_TRUE(part.b[i] >= 0_r && IsFinite(part.b[i]));
    }
  }
}

template <int kBS>
static void VerifyStrainStiffeningHelpers() {
  using V = BatchReal<kBS>;
  using V2x2 = BatchReal2x2<kBS>;
  // Small, because central differences are only first-order accurate across the tangent's kink
  // where an eigenvalue vanishes.
  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 1e-4_r;
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-5_r : 5e-3_r;
  double constexpr kEnergyRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9 : 1e-4;
  real constexpr kStiffness = 1.7_r;
  // Absolute, in units of the stiffness, because individual components may vanish.
  real constexpr kAbsTol = kTol * kStiffness;
  V const k{kStiffness};

  auto const broadcast = [](Matrix2x2r const& m) {
    std::array<Matrix2x2r, kBS> mats{};
    mats.fill(m);
    return LoadBatchMatrix2x2<kBS>(MakeConstSpan(mats));
  };
  // Covariant strains A·(A⁻¹·S) in the non-diagonal metric of MixedStrainWithPrincipalStrains.
  V2x2 const A = broadcast(Dot(kMixedStrainMetricRoot, kMixedStrainMetricRoot));
  V2x2 const AInv = Invert(A, Det(A));
  auto const strainWithPrincipalStrains = [&](Real2 const& e) {
    return Dot(A, broadcast(MixedStrainWithPrincipalStrains(e, 0.6_r)));
  };
  auto const psi = [&](V2x2 const& strain) {
    return PsiStrainStiffening<kBS>(PositivePart<kBS>(Dot(AInv, strain)), k);
  };
  auto const dpsi = [&](V2x2 const& strain) {
    return DPsiStrainStiffening<kBS>(PositivePart<kBS>(Dot(AInv, strain)), AInv, k);
  };
  auto const d2psi = [&](V2x2 const& strain) {
    return D2PsiStrainStiffeningSym2x2<kBS>(PositivePart<kBS>(Dot(AInv, strain)), AInv, k);
  };
  // Raw [00, 01, 11] coordinate directions; the shear one perturbs both 01 and 10.
  std::array<V2x2, 3> const directions{
      SymMatrix2x2(V{1_r}, V{0_r}, V{0_r}),
      SymMatrix2x2(V{0_r}, V{1_r}, V{0_r}),
      SymMatrix2x2(V{0_r}, V{0_r}, V{1_r})};

  // OuterSym and SymKroneckerSym2x2 reproduce the SVK tangent, pinning that their layouts match the
  // reduced tangent layout.
  {
    V const lambda{2_r};
    V const mu{1_r};
    BatchSymMatrix2x2<kBS> const aInv = Sym2x2Components(AInv);
    BatchSymMatrix3x3<kBS> const fromBuilders = lambda * OuterSym(aInv, aInv) +
        (V{2_r} * mu) * fem::details::SymKroneckerSym2x2<kBS>(aInv, aInv);
    auto const svk = D2PsiSVKSym2x2<kBS>(AInv, lambda, mu);
    for (int b = 0; b < kBS; ++b) {
      ExpectNearL2(GetLane(svk, b), GetLane(fromBuilders, b), kTol);
    }
  }

  // Principal strains both positive, equal, or with one zero, and with opposite signs for each sign
  // of the trace. (Cases with one near zero would need looser tolerances, since its absolute error
  // scales with the other.)
  for (Real2 const& e :
       {Real2{0.3_r, 0.1_r},
        Real2{0.2_r, 0.2_r},
        Real2{0.25_r, 0_r},
        Real2{0.3_r, -0.2_r},
        Real2{0.2_r, -0.2_r},
        Real2{0.1_r, -0.4_r}}) {
    SCOPED_TRACE(testing::Message() << "principal strains = {" << e[0] << ", " << e[1] << "}");
    V2x2 const strain = strainWithPrincipalStrains(e);
    auto const e0 = static_cast<double>(e[0]);
    auto const e1 = static_cast<double>(Max(e[1], 0_r));
    double const expectedPsi = static_cast<double>(kStiffness) * (e0 * e0 * e0 + e1 * e1 * e1);
    auto const actualPsi = psi(strain);
    V2x2 const stress = dpsi(strain);
    BatchSymMatrix3x3<kBS> const tangent = d2psi(strain);
    for (int b = 0; b < kBS; ++b) {
      EXPECT_NEAR(expectedPsi, actualPsi[b], kEnergyRelTol * expectedPsi);
    }

    // Central differences of ψ along each direction, and of dψ/dε for each tangent column.
    for (V2x2 const& direction : directions) {
      V2x2 const step = V{kEps} * direction;
      auto const psiP = psi(strain + step);
      auto const psiM = psi(strain - step);
      V const directional = Colon(stress, direction);
      for (int b = 0; b < kBS; ++b) {
        real const fd = static_cast<real>((psiP[b] - psiM[b]) / (2.0 * static_cast<double>(kEps)));
        EXPECT_NEAR_TOL(fd, directional[b], kAbsTol);
      }
      V2x2 const dpsiFd = V{0.5_r / kEps} * (dpsi(strain + step) - dpsi(strain - step));
      auto const applied =
          fem::details::ApplySym2x2Tangent<kBS>(tangent, Sym2x2Components(direction));
      ExpectNear2x2Lanes<kBS>(dpsiFd, ToMatrix(applied), 0_r, kAbsTol);
    }

    // The tangent is PSD as a quadratic form on symmetric strain increments.
    for (int t = 0; t < 5; ++t) {
      auto const delta = MakeRandomArray<3>(static_cast<unsigned int>(7 * t + 1), -1_r, 1_r);
      BatchSymMatrix2x2<kBS> const batchDelta{V{delta[0]}, V{delta[1]}, V{delta[2]}};
      V const q =
          ColonSym2x2(batchDelta, fem::details::ApplySym2x2Tangent<kBS>(tangent, batchDelta));
      for (int b = 0; b < kBS; ++b) {
        EXPECT_GE(q[b], -kTol * Norm(GetLane(tangent, b)) * NormSqr(delta));
      }
    }
  }

  // Strains with no positive eigenvalue give exact zeros.
  for (V2x2 const& strain :
       {strainWithPrincipalStrains(Real2{-0.1_r, -0.3_r}),
        strainWithPrincipalStrains(Real2{-0.2_r, -0.2_r}),
        SymMatrix2x2(V{0_r}, V{0_r}, V{0_r})}) {
    auto const zeroPsi = psi(strain);
    auto const zeroStress = Sym2x2Components(dpsi(strain));
    auto const zeroTangent = d2psi(strain);
    for (int b = 0; b < kBS; ++b) {
      EXPECT_EQ(0.0, zeroPsi[b]);
      EXPECT_SPAN_EQ((NdArray<real, 3>{}), GetLane(zeroStress, b));
      EXPECT_SPAN_EQ((NdArray<real, 6>{}), GetLane(zeroTangent, b));
    }
  }

  // Outputs are continuous where an eigenvalue changes sign.
  {
    SCOPED_TRACE("continuity");
    real constexpr kDelta = 1e-4_r;
    V2x2 const above = strainWithPrincipalStrains(Real2{0.3_r, kDelta});
    V2x2 const below = strainWithPrincipalStrains(Real2{0.3_r, -kDelta});
    auto const stressAbove = Sym2x2Components(dpsi(above));
    auto const stressBelow = Sym2x2Components(dpsi(below));
    auto const tangentAbove = d2psi(above);
    auto const tangentBelow = d2psi(below);
    for (int b = 0; b < kBS; ++b) {
      ExpectNearL2(GetLane(stressAbove, b), GetLane(stressBelow, b), 100_r * kDelta);
      ExpectNearL2(GetLane(tangentAbove, b), GetLane(tangentBelow, b), 100_r * kDelta);
    }
  }

  // With opposite-signed eigenvalues, the stress and tangent scale as the strain squared and the
  // strain, respectively, down to strains where cubing the eigenvalue gap would overflow in single
  // precision; they stay finite where the strain's determinant is subnormal there.
  for (Real2 const& e : {Real2{0.3_r, -0.2_r}, Real2{0.1_r, -0.4_r}}) {
    SCOPED_TRACE(testing::Message() << "eigenvalues = {" << e[0] << ", " << e[1] << "}");
    real constexpr kScale = 1e-15_r;
    V2x2 const strain = strainWithPrincipalStrains(e);
    auto const stress = Sym2x2Components(dpsi(strain));
    auto const tangent = d2psi(strain);
    auto const scaledStress =
        V{1_r / (kScale * kScale)} * Sym2x2Components(dpsi(V{kScale} * strain));
    auto const scaledTangent = V{1_r / kScale} * d2psi(V{kScale} * strain);
    auto const tinyTangent = d2psi(V{1e-21_r} * strain);
    for (int b = 0; b < kBS; ++b) {
      ExpectNearL2(GetLane(stress, b), GetLane(scaledStress, b), kTol);
      ExpectNearL2(GetLane(tangent, b), GetLane(scaledTangent, b), kTol);
      for (real const v : GetLane(tinyTangent, b)) {
        EXPECT_TRUE(IsFinite(v));
      }
    }
  }
}

template <int kBS>
static void VerifyKelvinInversionHelpers() {
  using V = BatchReal<kBS>;
  using V3 = BatchReal3<kBS>;
  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-4_r;
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-4_r : 5e-3_r;

  auto const broadcast = [](Real3 const& s) { return V3{V{s[0]}, V{s[1]}, V{s[2]}}; };

  // Direct value check pinning the v / |v|^2 contract, independent of the finite-difference
  // self-consistency checks below: {1, 2, 3} inverts to {1, 2, 3} / 14 (|v|^2 == 14).
  {
    V3 const kelvin = BatchKelvinInversion<kBS>(broadcast(Real3{1_r, 2_r, 3_r}));
    Real3 const expected{1_r / 14_r, 2_r / 14_r, 3_r / 14_r};
    for (int b = 0; b < kBS; ++b) {
      for (int d = 0; d < kSpaceDim3; ++d) {
        EXPECT_NEAR_TOL(expected[d], kelvin[d][b], kTol);
      }
    }
  }

  // Well-conditioned vectors (|v| bounded away from zero, where the divide-by-zero guard would
  // otherwise perturb the map) paired with independent perturbation directions.
  std::array<Real3, 4> const testVectors = {
      Real3{1_r, 2_r, 3_r},
      Real3{-2_r, 0.5_r, 1_r},
      Real3{0.3_r, -0.7_r, 0.2_r},
      Real3{4_r, -1_r, -2_r}};
  std::array<Real3, 4> const directions = {
      Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}, Real3{1_r, 1_r, 1_r}};

  for (auto const& vScalar : testVectors) {
    V3 const v = broadcast(vScalar);
    auto const jacobian = BatchDKelvinInversion<kBS>(v);

    for (auto const& dirScalar : directions) {
      V3 const dir = broadcast(dirScalar);
      V3 const vP = v + V{kEps} * dir;
      V3 const vM = v - V{kEps} * dir;
      // Central-difference directional derivative of BatchKelvinInversion.
      auto const fdDeriv =
          V{0.5_r / kEps} * (BatchKelvinInversion<kBS>(vP) - BatchKelvinInversion<kBS>(vM));

      // Analytic directional derivative: jacobian * dir.
      auto const analytic = DotMatVec(jacobian, dir);

      for (int b = 0; b < kBS; ++b) {
        ExpectNearL2(GetLane(fdDeriv, b), GetLane(analytic, b), kTol);
      }
    }
  }
}

template <int kBS>
static void VerifyMembraneResidualHelpers() {
  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;
  real constexpr kResTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-4_r : 1e-2_r;
  real constexpr kDResTol = 1e-3_r;

  for (auto const& cfg : MakeHelperConfigs()) {
    auto const refPos = ExtrapolatedPositions<kBS>(cfg);
    auto const disp = BroadcastNodeValues<kBS>(cfg.displacements);
    auto const res = MembraneResidual<kBS>(refPos, disp);
    auto const dres = MembraneDResidual<kBS>(refPos, disp);

    ShellBatchVector<kBS> resFd{};
    ShellBatchMatrix<kBS> dresFd{};
    for (int dof = 0; dof < kBendingStencilDofs; ++dof) {
      auto dispP = disp;
      auto dispM = disp;
      dispP[dof] += BatchReal<kBS>{kEps};
      dispM[dof] += BatchReal<kBS>{-kEps};
      auto const ep = MembraneEnergy<kBS>(refPos, dispP);
      auto const em = MembraneEnergy<kBS>(refPos, dispM);
      auto const rp = MembraneResidual<kBS>(refPos, dispP);
      auto const rm = MembraneResidual<kBS>(refPos, dispM);
      for (int b = 0; b < kBS; ++b) {
        resFd[dof] = Set(
            resFd[dof], b, static_cast<real>((ep[b] - em[b]) / (2.0 * static_cast<double>(kEps))));
      }
      for (int row = 0; row < kBendingStencilDofs; ++row) {
        dresFd[row * kBendingStencilDofs + dof] =
            (rp[row] - rm[row]) * BatchReal<kBS>{0.5_r / kEps};
      }
    }

    ExpectNearVectorLanes<kBS>(resFd, res, kResTol);
    ExpectNearMatrixLanes<kBS>(dresFd, dres, kDResTol);
  }
}

template <int kBS>
static void VerifyBendingResidualHelpers() {
  real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;
  real constexpr kResTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-3_r : 1e-2_r;
  real constexpr kDResTol = 1e-3_r;

  for (auto const& cfg : MakeHelperConfigs()) {
    auto const stencilGlobalNodes = BroadcastStencilNodes<kBS>(cfg.globalNodeIndices);
    auto const refPos = ExtrapolatedPositions<kBS>(cfg);
    auto const disp = BroadcastNodeValues<kBS>(cfg.displacements);
    auto const res = BendingResidual<kBS>(stencilGlobalNodes, refPos, disp);

    ShellBatchVector<kBS> resFd{};
    for (int dof = 0; dof < kBendingStencilDofs; ++dof) {
      auto dispP = disp;
      auto dispM = disp;
      dispP[dof] += BatchReal<kBS>{kEps};
      dispM[dof] += BatchReal<kBS>{-kEps};
      auto const ep = BendingEnergy<kBS>(refPos, dispP);
      auto const em = BendingEnergy<kBS>(refPos, dispM);
      for (int b = 0; b < kBS; ++b) {
        resFd[dof] = Set(
            resFd[dof], b, static_cast<real>((ep[b] - em[b]) / (2.0 * static_cast<double>(kEps))));
      }
    }
    ExpectNearVectorLanes<kBS>(resFd, res, kResTol);

    ShellBatchVector<kBS> zeroDisp{};
    auto const dres = BendingDResidual<kBS>(stencilGlobalNodes, refPos, zeroDisp);
    ShellBatchMatrix<kBS> dresFd{};
    for (int dof = 0; dof < kBendingStencilDofs; ++dof) {
      auto dispP = zeroDisp;
      auto dispM = zeroDisp;
      dispP[dof] += BatchReal<kBS>{kEps};
      dispM[dof] += BatchReal<kBS>{-kEps};
      auto const resP = BendingResidual<kBS>(stencilGlobalNodes, refPos, dispP);
      auto const resM = BendingResidual<kBS>(stencilGlobalNodes, refPos, dispM);
      for (int row = 0; row < kBendingStencilDofs; ++row) {
        dresFd[row * kBendingStencilDofs + dof] =
            (resP[row] - resM[row]) * BatchReal<kBS>{0.5_r / kEps};
      }
    }
    ExpectNearMatrixLanes<kBS>(dresFd, dres, kDResTol);
  }
}

template <int kDim>
static double QuadraticForm(NdArray<real, kDim * kDim> const& m, NdArray<real, kDim> const& v) {
  double q = 0.0;
  for (int i = 0; i < kDim; ++i) {
    for (int j = 0; j < kDim; ++j) {
      q += static_cast<double>(v[i]) * static_cast<double>(m[i * kDim + j]) *
          static_cast<double>(v[j]);
    }
  }
  return q;
}

// Expects m to be symmetric and to have no eigenvalue below −tol·‖m‖, with tolerances relative to
// the Frobenius norm ‖m‖, since rounding errors scale with it. Without Eigen, it checks only the
// necessary conditions on coordinate vectors and their pairwise sums and differences.
static void ExpectSymmetricPsd(NdArray<real, kBendingStencilDofs * kBendingStencilDofs> const& m) {
  constexpr int kN = kBendingStencilDofs;
  real constexpr kSymRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-5_r;
  real constexpr kPsdRelTol =
      (MOCHI_USE_DOUBLE_PRECISION ? 300_r : 100_r) * std::numeric_limits<real>::epsilon();
  real const norm = Norm(m);
  ASSERT_TRUE(IsFinite(norm));

  for (int i = 0; i < kN; ++i) {
    for (int j = i + 1; j < kN; ++j) {
      EXPECT_NEAR_TOL(m[i * kN + j], m[j * kN + i], kSymRelTol * norm);
    }
  }

  real const psdTol = kPsdRelTol * norm;
#if MOCHI_USE_EIGEN
  using EMatType = Eigen::Matrix<real, kN, kN, Eigen::RowMajor | Eigen::DontAlign>;
  Eigen::Map<EMatType const> const eigenMat(&m[0]);
  Eigen::SelfAdjointEigenSolver<EMatType> const eigenSolver(eigenMat, Eigen::EigenvaluesOnly);
  ASSERT_EQ(Eigen::Success, eigenSolver.info());
  EXPECT_GE(eigenSolver.eigenvalues().minCoeff(), -psdTol);
#else
  for (int i = 0; i < kN; ++i) {
    double const diagonal = static_cast<double>(m[i * kN + i]);
    EXPECT_GE(diagonal, -psdTol);
    for (int j = i + 1; j < kN; ++j) {
      double const diagonalSum = diagonal + static_cast<double>(m[j * kN + j]);
      double const offDiagonalSum =
          static_cast<double>(m[i * kN + j]) + static_cast<double>(m[j * kN + i]);
      EXPECT_GE(diagonalSum + offDiagonalSum, -2_r * psdTol);
      EXPECT_GE(diagonalSum - offDiagonalSum, -2_r * psdTol);
    }
  }
#endif
}

template <int kBS>
static void VerifyShellFd(
    std::array<ShellStencilConfig const*, kBS> const& cfgs,
    ShellFdOptions options = {}) {
  using V = BatchReal<kBS>;
  constexpr int kN = kBendingStencilDofs;
  ShellHarness<kBS> h(cfgs, options);

  if (options.checkEnergyResidual || options.checkResidualDResidual) {
    real constexpr kResTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-4_r : 1e-3_r;
    real constexpr kDResTol = MOCHI_USE_DOUBLE_PRECISION ? 2e-3_r : 5e-3_r;

    BatchDouble<kBS> e{};
    NdArray<V, kN> baseRes{};
    NdArray<V, kN * kN> baseDRes{};
    h.Run(
        h.baseDisp,
        options.checkEnergyResidual ? &e : nullptr,
        &baseRes,
        options.checkResidualDResidual ? &baseDRes : nullptr,
        /*projectPsd*/ false);

    // res == d(energy)/d(disp); dRes == d(res)/d(disp). Stored per-lane (AoS) since SIMD lanes are
    // not individually assignable.
    NdArray<NdArray<real, kN>, kBS> resFd{};
    NdArray<NdArray<real, kN * kN>, kBS> dresFd{};
    for (int j = 0; j < kN; ++j) {
      auto dispP = h.baseDisp;
      auto dispM = h.baseDisp;
      dispP[j] = dispP[j] + V{options.fdStep};
      dispM[j] = dispM[j] - V{options.fdStep};
      // Fresh outputs each call: ShellWork accumulates (+=) into them.
      BatchDouble<kBS> ep{}, em{};
      NdArray<V, kN> rp{}, rm{};
      h.Run(
          dispP,
          options.checkEnergyResidual ? &ep : nullptr,
          options.checkResidualDResidual ? &rp : nullptr,
          nullptr,
          /*projectPsd*/ false);
      h.Run(
          dispM,
          options.checkEnergyResidual ? &em : nullptr,
          options.checkResidualDResidual ? &rm : nullptr,
          nullptr,
          /*projectPsd*/ false);
      for (int b = 0; b < kBS; ++b) {
        if (options.checkEnergyResidual) {
          resFd[b][j] =
              static_cast<real>((ep[b] - em[b]) / (2.0 * static_cast<double>(options.fdStep)));
        }
        if (options.checkResidualDResidual) {
          for (int i = 0; i < kN; ++i) {
            dresFd[b][i * kN + j] = (rp[i][b] - rm[i][b]) / (2_r * options.fdStep);
          }
        }
      }
    }

    for (int b = 0; b < kBS; ++b) {
      if (options.checkEnergyResidual) {
        NdArray<real, kN> actRes{};
        for (int i = 0; i < kN; ++i) {
          actRes[i] = baseRes[i][b];
        }
        ExpectNearL2(resFd[b], actRes, kResTol);
      }

      if (options.checkResidualDResidual) {
        NdArray<real, kN * kN> actDRes{};
        for (int k = 0; k < kN * kN; ++k) {
          actDRes[k] = baseDRes[k][b];
        }
        ExpectNearL2(actDRes, dresFd[b], kDResTol);
      }
    }

    // Missing (sentinel) nodes contribute zero residual/dresidual.
    for (int b = 0; b < kBS; ++b) {
      for (int n = 0; n < kBendingStencilNodes; ++n) {
        if (cfgs[b]->globalNodeIndices[n] == kSentinelIndex) {
          for (int d = 0; d < kSpaceDim3; ++d) {
            int const dof = n * kSpaceDim3 + d;
            EXPECT_EQ(0_r, baseRes[dof][b]);
            if (options.checkResidualDResidual) {
              for (int j = 0; j < kN; ++j) {
                EXPECT_EQ(0_r, baseDRes[dof * kN + j][b]);
                EXPECT_EQ(0_r, baseDRes[j * kN + dof][b]);
              }
            }
          }
        }
      }
    }
  }

  if (options.checkPsdProjection) {
    NdArray<V, kN * kN> psdDRes{};
    h.Run(h.baseDisp, nullptr, nullptr, &psdDRes, /*projectPsd*/ true);
    for (int b = 0; b < kBS; ++b) {
      auto const mat = GetLane(psdDRes, b);
      ExpectSymmetricPsd(mat);
      for (int t = 0; t < 5; ++t) {
        auto const v = MakeRandomArray<kN>(static_cast<unsigned int>(13 * t + b), -1_r, 1_r);
        EXPECT_GE(QuadraticForm<kN>(mat, v), -1e-4);
      }
    }
  }
}

template <int kBS>
static void VerifyShellOutputModes(
    std::array<ShellStencilConfig const*, kBS> const& cfgs,
    ShellFdOptions const& options = {}) {
  ShellHarness<kBS> h(cfgs, options);
  auto const all = RunShell(h, {.energy = true, .residual = true, .dresidual = true});
  for (OutputConfig const cfg : kAllOutputConfigs) {
    auto const actual = RunShell(h, cfg);
    for (int b = 0; b < kBS; ++b) {
      if (cfg.energy) {
        ExpectNearEnergy(all.energy[b], actual.energy[b]);
      }
      if (cfg.residual) {
        ExpectNearL2(GetLane(all.res, b), GetLane(actual.res, b));
      }
      if (cfg.dresidual) {
        ExpectNearL2(GetLane(all.dres, b), GetLane(actual.dres, b));
      }
    }
  }
}

template <int kBS>
static std::array<ShellStencilConfig const*, kBS> BroadcastConfig(ShellStencilConfig const& cfg) {
  std::array<ShellStencilConfig const*, kBS> cfgs{};
  cfgs.fill(&cfg);
  return cfgs;
}

// Broadcasts one config to all lanes.
template <int kBS>
static void VerifyShellFdBroadcast(ShellStencilConfig const& cfg, ShellFdOptions options) {
  VerifyShellFd<kBS>(BroadcastConfig<kBS>(cfg), options);
}

template <int kBS>
static void VerifyShellOutputModesBroadcast(ShellStencilConfig const& cfg) {
  VerifyShellOutputModes<kBS>(BroadcastConfig<kBS>(cfg));
}

static void VerifyShellFdCanonicalConfigs(ShellFdOptions const& options) {
  auto const configs = MakeShellWorkFdConfigs();
  for (auto const& cfg : configs) {
    RunSupportedFemShellRodBatchSizes(
        [&]<int kBS>() { VerifyShellFdBroadcast<kBS>(cfg, options); });
  }
}

template <int kBS, class VerifyFn>
static void RunShellMixedLaneConfigs(VerifyFn const& verify) {
  static_assert(kBS > 1, "Batch size must be greater than 1 for mixed-lane tests");
  auto const interior = MakeEquilateralConfig();
  auto const b3 = MakeBoundaryNode3Missing();
  auto const b4 = MakeBoundaryNode4Missing();
  auto const b5 = MakeBoundaryNode5Missing();
  auto const b34 = MakeBoundary34Missing();
  auto const b35 = MakeBoundary35Missing();
  auto const b45 = MakeBoundary45Missing();
  auto const bAll = MakeBoundaryAllMissing();

  if constexpr (kBS == 4) {
    verify(std::array<ShellStencilConfig const*, 4>{&interior, &b3, &b4, &b35});
    verify(std::array<ShellStencilConfig const*, 4>{&b5, &b34, &b45, &interior});
    // Simulates the assembler padding convention: last real element duplicated into the tail.
    verify(std::array<ShellStencilConfig const*, 4>{&interior, &b3, &b4, &b4});
  } else if constexpr (kBS == 8) {
    verify(
        std::array<ShellStencilConfig const*, 8>{
            &b4, &b35, &bAll, &b34, &interior, &b3, &b5, &b45});
  } else {
    std::array<ShellStencilConfig const*, kBS> cfgs{};
    std::array<ShellStencilConfig const*, 8> const allConfigs = {
        &interior, &b3, &b4, &b5, &b34, &b35, &b45, &bAll};
    for (int lane = 0; lane < kBS; ++lane) {
      cfgs[lane] = allConfigs[lane % isize(allConfigs)];
    }
    verify(cfgs);

    cfgs[kBS - 1] = cfgs[kBS - 2];
    verify(cfgs);
  }
}

static void VerifyShellFdMixedLanes(ShellFdOptions const& options) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    if constexpr (kBS == 1) {
      // Batch size 1 is covered by broadcast tests, but cannot exercise mixed-lane behavior.
    } else {
      RunShellMixedLaneConfigs<kBS>([&](std::array<ShellStencilConfig const*, kBS> const& cfgs) {
        VerifyShellFd<kBS>(cfgs, options);
      });
    }
  });
}

// Rest state: zero displacement, flat reference => zero energy.
template <int kBS>
static void VerifyShellRest(ShellStencilConfig const& cfg) {
  ShellHarness<kBS> h(BroadcastConfig<kBS>(cfg));
  NdArray<BatchReal<kBS>, kBendingStencilDofs> zero{};
  BatchDouble<kBS> e{};
  h.Run(zero, &e, nullptr, nullptr, /*projectPsd*/ false);
  for (int b = 0; b < kBS; ++b) {
    EXPECT_NEAR(0.0, e[b], 1e-9);
  }
}

TEST(ShellHelpers, MetricAndStrain) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyMetricAndStrainHelpers<kBS>(); });
}

TEST(ShellHelpers, BendingEdgeGeometry) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyBendingEdgeGeometryHelpers<kBS>(); });
}

TEST(ShellHelpers, BendingPositionGeometry) {
  RunSupportedFemShellRodBatchSizes(
      [&]<int kBS>() { VerifyBendingPositionGeometryHelpers<kBS>(); });
}

TEST(ShellHelpers, DSecondFundamentalFormDxMixedLane) {
  // Lane-independence of the production fused DSecondFundamentalFormDx(db_dv, stencilGlobalNodes):
  // a mixed batch (a different missing-opposite-node pattern per lane, so anyMissingOppositeNode
  // forces the slow path) must match the single-lane result computed for each pattern on its own.
  // The interior lane also checks that the slow path reduces to the fast path. Synthetic distinct
  // db/dv entries make coefficient/sign mistakes observable while keeping db/dv broadcast so the
  // per-lane behavior is driven by the stencil pattern, not by db/dv.
  std::array<ShellStencilConfig, 8> const configs = {
      MakeInteriorConfig(),
      MakeBoundaryNode3Missing(),
      MakeBoundaryNode4Missing(),
      MakeBoundaryNode5Missing(),
      MakeBoundary34Missing(),
      MakeBoundary35Missing(),
      MakeBoundary45Missing(),
      MakeBoundaryAllMissing()};

  ShellBatchStencil<8> mixed{};
  for (int lane = 0; lane < static_cast<int>(configs.size()); ++lane) {
    for (int n = 0; n < kBendingStencilNodes; ++n) {
      mixed[n][lane] = configs[lane].globalNodeIndices[n];
    }
  }

  NdArray<BatchSymMatrix2x2<8>, kBendingStencilNodes, kSpaceDim3> dbDv8{};
  NdArray<BatchSymMatrix2x2<1>, kBendingStencilNodes, kSpaceDim3> dbDv1{};
  for (int edge = 0; edge < kBendingStencilNodes; ++edge) {
    for (int i = 0; i < kSpaceDim3; ++i) {
      auto const base = static_cast<real>(10 * edge + i + 1);
      dbDv8[edge][i] = Sym2x2Components(
          BatchReal<8>{base}, BatchReal<8>{base + 100_r}, BatchReal<8>{base + 200_r});
      dbDv1[edge][i] = Sym2x2Components(
          BatchReal<1>{base}, BatchReal<1>{base + 100_r}, BatchReal<1>{base + 200_r});
    }
  }

  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-12_r : 1e-5_r;
  auto const mixedDbDx = DSecondFundamentalFormDx<8>(dbDv8, mixed);
  for (int lane = 0; lane < static_cast<int>(configs.size()); ++lane) {
    auto const homogeneousDbDx = DSecondFundamentalFormDx<1>(
        dbDv1, BroadcastStencilNodes<1>(configs[lane].globalNodeIndices));
    for (int n = 0; n < kBendingStencilNodes; ++n) {
      for (int i = 0; i < kSpaceDim3; ++i) {
        for (int c = 0; c < 3; ++c) {
          real const ref = homogeneousDbDx[n][i][c][0];
          EXPECT_NEAR_TOL(ref, mixedDbDx[n][i][c][lane], Max(kTol, kTol * Abs(ref)));
        }
      }
    }
  }
}

TEST(ShellHelpers, SvkDerivatives) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifySvkHelpers<kBS>(); });
}

TEST(ShellHelpers, PositivePart) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyPositivePartHelper<kBS>(); });
}

TEST(ShellHelpers, StrainStiffeningDerivatives) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyStrainStiffeningHelpers<kBS>(); });
}

TEST(ShellHelpers, KelvinInversion) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyKelvinInversionHelpers<kBS>(); });
}

TEST(ShellHelpers, MembraneResiduals) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyMembraneResidualHelpers<kBS>(); });
}

TEST(ShellHelpers, BendingResiduals) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyBendingResidualHelpers<kBS>(); });
}

// Anchors the membrane energy magnitude, which finite differences cannot check, and the residual.
// The central triangle has a constant Green strain, ε = G⁻¹ (g - G) / 2 in terms of the reference
// and current edge metrics G and g, so its St. Venant-Kirchhoff energy is
// (μ tr(ε²) + λ/2 tr(ε)²) · area. Its gradient with respect to the current edge eᵢ is
// area · Σⱼ Sᵢⱼ eⱼ, with the symmetric stress S = (λ tr(ε) I + 2μ ε) G⁻¹.
TEST(ShellWork, MembraneMatchesClosedForm) {
  ShellFdOptions options;
  options.bendingAlpha = 0_r;
  options.bendingBeta = 0_r;
  options.membraneCollapseBarrierStiffness = 0_r;
  for (auto const& cfg : MakeHelperConfigs()) {
    auto const edges = [&](real dispScale) {
      auto const position = [&](int n) {
        return cfg.refPositions[n] + dispScale * cfg.displacements[n];
      };
      return std::array<Real3, 2>{position(1) - position(0), position(2) - position(0)};
    };
    auto const metric = [](std::array<Real3, 2> const& e) {
      return SymMatrix2x2(Dot(e[0], e[0]), Dot(e[0], e[1]), Dot(e[1], e[1]));
    };
    auto const e = edges(1_r);
    Matrix2x2r const G = metric(edges(0_r));
    Matrix2x2r const GInv = Invert(G);
    Matrix2x2r const strain = Dot(GInv, 0.5_r * (metric(e) - G));
    real const trace = Trace(strain);
    real const traceOfSquare = Trace(Dot(strain, strain));
    real const area = 0.5_r * Sqrt(Det(G));
    Matrix2x2r const stress =
        kMembraneLambda * trace * GInv + 2_r * kMembraneMu * Dot(strain, GInv);
    NdArray<real, kBendingStencilDofs> residual{};
    for (int i = 0; i < 2; ++i) {
      Real3 const edgeGradient = area * (stress[i][0] * e[0] + stress[i][1] * e[1]);
      for (int d = 0; d < kSpaceDim3; ++d) {
        residual[kSpaceDim3 * (i + 1) + d] = edgeGradient[d];
        residual[d] -= edgeGradient[d];
      }
    }
    auto const out = RunShell(
        ShellHarness<1>(BroadcastConfig<1>(cfg), options), {.energy = true, .residual = true});
    ExpectNearEnergy(
        (kMembraneMu * traceOfSquare + 0.5_r * kMembraneLambda * Sqr(trace)) * area, out.energy[0]);
    ExpectNearL2(residual, GetLane(out.res, 0));
  }
}

TEST(ShellWork, BendingEnergyInvariantUnderCentralNodePermutations) {
  auto const cfg = MakeInteriorConfig();
  ShellFdOptions options;
  options.membraneLambda = 0_r;
  options.membraneMu = 0_r;

  ShellHarness<1> const referenceHarness(BroadcastConfig<1>(cfg), options);
  double const referenceEnergy = RunShell(referenceHarness, {.energy = true}).energy[0];
  EXPECT_NE(referenceEnergy, 0.0);

  std::array<int, kTriangleNodes> permutation = {0, 1, 2};
  while (std::next_permutation(permutation.begin(), permutation.end())) {
    SCOPED_TRACE(
        testing::Message() << "permutation = {" << permutation[0] << ", " << permutation[1] << ", "
                           << permutation[2] << "}");
    auto const permutedCfg = PermuteCentralNodes(cfg, permutation);
    ShellHarness<1> const permutedHarness(BroadcastConfig<1>(permutedCfg), options);
    double const permutedEnergy = RunShell(permutedHarness, {.energy = true}).energy[0];
    ExpectNearEnergy(referenceEnergy, permutedEnergy);
  }
}

TEST(ShellWork, EnergyResidualFd) {
  ShellFdOptions options;
  options.checkResidualDResidual = false;
  VerifyShellFdCanonicalConfigs(options);
  VerifyShellFdMixedLanes(options);
}

TEST(ShellWork, ResidualDResidualFdAtRest) {
  ShellFdOptions options;
  options.checkEnergyResidual = false;
  options.dispScale = 0_r;
  VerifyShellFdCanonicalConfigs(options);
  VerifyShellFdMixedLanes(options);
}

TEST(ShellWork, ResidualDResidualFdMembraneOnly) {
  ShellFdOptions options;
  options.checkEnergyResidual = false;
  options.bendingAlpha = 0_r;
  options.bendingBeta = 0_r;
  VerifyShellFdCanonicalConfigs(options);
  VerifyShellFdMixedLanes(options);
}

TEST(ShellWork, PsdProjection) {
  ShellFdOptions options;
  options.checkEnergyResidual = false;
  options.checkResidualDResidual = false;
  options.checkPsdProjection = true;
  VerifyShellFdCanonicalConfigs(options);
  VerifyShellFdMixedLanes(options);
}

TEST(ShellWork, RestState) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    VerifyShellRest<kBS>(MakeInteriorConfig());
    VerifyShellRest<kBS>(MakeEquilateralConfig());
    VerifyShellRest<kBS>(MakeBoundaryNode3Missing());
  });
}

TEST(ShellWork, OutputModes) {
  auto const interior = MakeEquilateralConfig();
  auto const boundary = MakeBoundaryNode3Missing();
  auto const allMissing = MakeBoundaryAllMissing();
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    VerifyShellOutputModesBroadcast<kBS>(interior);
    VerifyShellOutputModesBroadcast<kBS>(boundary);
  });
  std::array<ShellStencilConfig const*, kDefaultFemBatchSize> mixedConfigs{};
  for (int b = 0; b < kDefaultFemBatchSize; ++b) {
    mixedConfigs[b] = (b % 3 == 0) ? &interior : (b % 3 == 1) ? &boundary : &allMissing;
  }
  VerifyShellOutputModes<kDefaultFemBatchSize>(mixedConfigs);
}

// ---------------------------------------------------------------------------
// Stiffness damping FD consistency tests
// ---------------------------------------------------------------------------
//
// Stiffness-proportional damping is unified into the elastic response: each material stiffness is
// scaled to (1+factor)·k and the modified strain ε̃ = ε − (factor/(1+factor))·ε_stageStart is fed
// through the same formulas as the undamped path. When factor == 0 the kernel runs the existing
// undamped path. The stage-start state is held fixed during a stage, so the FD scaffolding (which
// perturbs `disp` only) carries damping through transparently.

TEST(ShellWork, EnergyResidualFdWithStiffnessDamping) {
  ShellFdOptions options;
  options.checkResidualDResidual = false;
  options.stiffnessDampingFactor = 3_r;
  options.stageStartDispScale = 0.5_r;
  VerifyShellFdCanonicalConfigs(options);
}

TEST(ShellWork, ResidualDResidualFdAtRestWithStiffnessDamping) {
  ShellFdOptions options;
  options.checkEnergyResidual = false;
  options.dispScale = 0_r;
  options.stiffnessDampingFactor = 3_r;
  VerifyShellFdCanonicalConfigs(options);
}

TEST(ShellWork, EnergyResidualFdWithStiffnessDampingMembraneOnly) {
  ShellFdOptions options;
  options.checkResidualDResidual = false;
  options.bendingAlpha = 0_r;
  options.bendingBeta = 0_r;
  options.stiffnessDampingFactor = 3_r;
  options.stageStartDispScale = 0.5_r;
  VerifyShellFdCanonicalConfigs(options);
}

TEST(ShellWork, ResidualDResidualFdWithStiffnessDampingMembraneOnly) {
  ShellFdOptions options;
  options.checkEnergyResidual = false;
  options.bendingAlpha = 0_r;
  options.bendingBeta = 0_r;
  options.stiffnessDampingFactor = 3_r;
  options.stageStartDispScale = 0.5_r;
  VerifyShellFdCanonicalConfigs(options);
}

TEST(ShellWork, EnergyResidualFdWithStiffnessDampingBendingOnly) {
  ShellFdOptions options;
  options.checkResidualDResidual = false;
  options.membraneLambda = 0_r;
  options.membraneMu = 0_r;
  options.stiffnessDampingFactor = 3_r;
  options.stageStartDispScale = 0.5_r;
  VerifyShellFdCanonicalConfigs(options);
}

// NOTE: Bending-only DRes with stiffness damping is not tested separately because the
// approximate bending tangent (which omits geometric stiffness) has larger error when damping
// amplifies the stress. This is tested indirectly through the combined membrane+bending DRes test.

// ---------------------------------------------------------------------------
// Membrane collapse barrier k·⟨tr(C⁻¹) − 2⟩³
// ---------------------------------------------------------------------------

static constexpr real kCollapseBarrierStiffness = 50_r;

// cfg with homogeneous principal stretches along the world axes, plus perturbationScale times its
// own displacements.
static ShellStencilConfig
MakeStretchedConfig(ShellStencilConfig cfg, Real3 const& stretches, real perturbationScale) {
  for (int n = 0; n < kBendingStencilNodes; ++n) {
    cfg.displacements[n] =
        (stretches - 1_r) * cfg.refPositions[n] + perturbationScale * cfg.displacements[n];
  }
  return cfg;
}

// Flat interior config in the xy-plane with in-plane principal stretches (sx, sy). The central
// triangle has reference metric A = I and reference area 0.5.
static ShellStencilConfig
MakeStretchedInteriorConfig(real sx, real sy, real perturbationScale = 0_r) {
  return MakeStretchedConfig(MakeInteriorConfig(), Real3{sx, sy, 1_r}, perturbationScale);
}

// Flat equilateral config in the xz-plane with in-plane principal stretches (sx, sz). The central
// triangle has a non-identity reference metric A and reference area √3/4.
static ShellStencilConfig
MakeStretchedEquilateralConfig(real sx, real sz, real perturbationScale = 0_r) {
  return MakeStretchedConfig(MakeEquilateralConfig(), Real3{sx, 1_r, sz}, perturbationScale);
}

static std::array<ShellStencilConfig, 5> MakeCompressedConfigs() {
  return {
      MakeStretchedInteriorConfig(0.7_r, 0.8_r),
      MakeStretchedInteriorConfig(0.6_r, 1.1_r),
      MakeStretchedInteriorConfig(0.7_r, 0.8_r, 0.1_r),
      MakeStretchedEquilateralConfig(0.7_r, 0.8_r),
      MakeStretchedEquilateralConfig(0.6_r, 1.1_r, 0.1_r)};
}

// Contribution of an optional membrane term: outputs with its stiffness option set to `stiffness`
// minus outputs with it zero, all else equal.
template <int kBS>
static ShellOutputs<kBS> MembraneTermContribution(
    real ShellFdOptions::* term,
    real stiffness,
    std::array<ShellStencilConfig const*, kBS> const& cfgs,
    ShellFdOptions options,
    bool projectPsd) {
  OutputConfig constexpr kAll{.energy = true, .residual = true, .dresidual = true};
  options.*term = stiffness;
  auto const with = RunShell(ShellHarness<kBS>(cfgs, options), kAll, projectPsd);
  options.*term = 0_r;
  auto const without = RunShell(ShellHarness<kBS>(cfgs, options), kAll, projectPsd);
  return {with.energy - without.energy, with.res - without.res, with.dres - without.dres};
}

template <int kBS>
static ShellOutputs<kBS> CollapseBarrierContribution(
    std::array<ShellStencilConfig const*, kBS> const& cfgs,
    ShellFdOptions options = {},
    bool projectPsd = false) {
  return MembraneTermContribution<kBS>(
      &ShellFdOptions::membraneCollapseBarrierStiffness,
      kCollapseBarrierStiffness,
      cfgs,
      options,
      projectPsd);
}

template <int kBS>
static void ExpectZeroLane(ShellOutputs<kBS> const& out, int b) {
  using LaneVector = NdArray<real, kBendingStencilDofs>;
  using LaneMatrix = NdArray<real, kBendingStencilDofs * kBendingStencilDofs>;
  EXPECT_EQ(0.0, out.energy[b]);
  EXPECT_SPAN_EQ(LaneVector{}, GetLane(out.res, b));
  EXPECT_SPAN_EQ(LaneMatrix{}, GetLane(out.dres, b));
}

template <size_t kNumConfigs>
static void ExpectMembraneTermIndependentOfStiffnessDamping(
    real ShellFdOptions::* term,
    real stiffness,
    std::array<ShellStencilConfig, kNumConfigs> const& cfgs) {
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-4_r;
  ShellFdOptions damped;
  damped.stiffnessDampingFactor = 3_r;
  damped.stageStartDispScale = 0.5_r;
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    for (auto const& cfg : cfgs) {
      auto const lanes = BroadcastConfig<kBS>(cfg);
      auto const undampedTerm = MembraneTermContribution<kBS>(term, stiffness, lanes, {}, false);
      auto const dampedTerm = MembraneTermContribution<kBS>(term, stiffness, lanes, damped, false);
      for (int b = 0; b < kBS; ++b) {
        EXPECT_NEAR(
            undampedTerm.energy[b],
            dampedTerm.energy[b],
            static_cast<double>(kTol) * undampedTerm.energy[b]);
        ExpectNearL2(GetLane(undampedTerm.res, b), GetLane(dampedTerm.res, b), kTol);
        ExpectNearL2(GetLane(undampedTerm.dres, b), GetLane(dampedTerm.dres, b), kTol);
      }
    }
  });
}

// Homogeneous in-plane principal stretches (s₀, s₁) give g = s₀⁻² + s₁⁻² − 2, independent of the
// reference triangle. Anisotropic cases guard against formulas that agree only when s₀ = s₁, and
// cases with g = 0 must leave every output unchanged. (2⁻⁸, 1) lies just past the single-precision
// onset of the energy's linear continuation, which must still match.
TEST(ShellWork, CollapseBarrierEnergyClosedForm) {
  double constexpr kEnergyRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9 : 1e-4;
  std::array<Real2, 15> constexpr kStretches = {
      Real2{0.01_r, 0.01_r},
      Real2{0.3_r, 0.3_r},
      Real2{0.5_r, 0.5_r},
      Real2{0.8_r, 0.8_r},
      Real2{0.95_r, 0.95_r},
      Real2{1_r, 1_r},
      Real2{1.2_r, 1.2_r},
      Real2{2_r, 2_r},
      Real2{0.6_r, 1.1_r},
      Real2{0.5_r, 0.9_r},
      Real2{1.3_r, 0.7_r},
      Real2{0.05_r, 1_r},
      Real2{1_r, 0.01_r},
      Real2{0x1p-8_r, 1_r},
      Real2{1.5_r, 0.95_r}};
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    for (Real2 const& s : kStretches) {
      auto const s0 = static_cast<double>(s[0]);
      auto const s1 = static_cast<double>(s[1]);
      double const g = Max(1.0 / (s0 * s0) + 1.0 / (s1 * s1) - 2.0, 0.0);
      for (auto const& [cfg, refArea] :
           {std::pair{MakeStretchedInteriorConfig(s[0], s[1]), 0.5},
            std::pair{MakeStretchedEquilateralConfig(s[0], s[1]), std::numbers::sqrt3 / 4.0}}) {
        auto const barrier = CollapseBarrierContribution<kBS>(BroadcastConfig<kBS>(cfg));
        double const expected =
            static_cast<double>(kCollapseBarrierStiffness) * refArea * g * g * g;
        for (int b = 0; b < kBS; ++b) {
          if (g == 0.0) {
            ExpectZeroLane(barrier, b);
          } else {
            EXPECT_NEAR(expected, barrier.energy[b], kEnergyRelTol * expected);
          }
        }
      }
    }
  });
}

TEST(ShellWork, CollapseBarrierOutputModes) {
  ShellFdOptions options;
  options.membraneCollapseBarrierStiffness = kCollapseBarrierStiffness;
  for (auto const& cfg : MakeCompressedConfigs()) {
    RunSupportedFemShellRodBatchSizes(
        [&]<int kBS>() { VerifyShellOutputModes<kBS>(BroadcastConfig<kBS>(cfg), options); });
  }
}

// Inactive lanes must be unaffected by active neighbors.
TEST(ShellWork, CollapseBarrierInactiveUnderTensionBesideCompressedLanes) {
  auto const biaxial = MakeStretchedInteriorConfig(1.3_r, 1.1_r);
  auto const compressed = MakeStretchedInteriorConfig(0.7_r, 0.8_r);
  std::array<ShellStencilConfig const*, kDefaultFemBatchSize> mixed{};
  for (int b = 0; b < kDefaultFemBatchSize; ++b) {
    mixed[b] = (b % 2 == 0) ? &compressed : &biaxial;
  }
  auto const barrier = CollapseBarrierContribution<kDefaultFemBatchSize>(mixed);
  for (int b = 0; b < kDefaultFemBatchSize; ++b) {
    if (b % 2 == 0) {
      EXPECT_GT(barrier.energy[b], 0.0);
    } else {
      ExpectZeroLane(barrier, b);
    }
  }
}

// The near-collapse cases (smallest stretch 2⁻⁷ or 2⁻¹⁰) straddle the onset of saturation in single
// precision. They need steps far below their collapsed extent (powers of two keep the perturbed
// near-unit displacements exact) and a small stiffness, so that the squared norms of the compared
// gradients can't overflow.
TEST(ShellWork, CollapseBarrierEnergyResidualFd) {
  std::array<ShellStencilConfig, 3> const nearCollapse = {
      MakeStretchedInteriorConfig(1_r, 0x1p-7_r),
      MakeStretchedInteriorConfig(0.5_r, 0x1p-10_r, 0x1p-13_r),
      MakeStretchedEquilateralConfig(0x1p-10_r, 0.5_r)};
  for (real const dampingFactor : {0_r, 3_r}) {
    ShellFdOptions options;
    options.checkResidualDResidual = false;
    options.membraneCollapseBarrierStiffness = kCollapseBarrierStiffness;
    options.stiffnessDampingFactor = dampingFactor;
    options.stageStartDispScale = 0.5_r;
    for (auto const& cfg : MakeCompressedConfigs()) {
      RunSupportedFemShellRodBatchSizes(
          [&]<int kBS>() { VerifyShellFdBroadcast<kBS>(cfg, options); });
    }
    options.fdStep = MOCHI_USE_DOUBLE_PRECISION ? 0x1p-20_r : 0x1p-17_r;
    options.membraneCollapseBarrierStiffness = 0x1p-12_r;
    for (auto const& cfg : nearCollapse) {
      RunSupportedFemShellRodBatchSizes(
          [&]<int kBS>() { VerifyShellFdBroadcast<kBS>(cfg, options); });
    }
  }
}

// The DRes contribution is the Gauss–Newton tangent 6kAg·∇g∇gᵀ. With barrier energy E = kAg³ and
// residual r = 3kAg²·∇g, this equals 2·r rᵀ / (3E).
TEST(ShellWork, CollapseBarrierGaussNewtonDResidual) {
  constexpr int kN = kBendingStencilDofs;
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-4_r;
  auto const compressed = MakeCompressedConfigs();
  DynamicArray<ShellStencilConfig> cfgs(compressed.begin(), compressed.end());
  cfgs.push_back(MakeStretchedInteriorConfig(0.03_r, 0.03_r));
  for (bool const projectPsd : {false, true}) {
    RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
      for (auto const& cfg : cfgs) {
        auto const barrier =
            CollapseBarrierContribution<kBS>(BroadcastConfig<kBS>(cfg), {}, projectPsd);
        for (int b = 0; b < kBS; ++b) {
          ASSERT_GT(barrier.energy[b], 0.0);
          auto const r = GetLane(barrier.res, b);
          real const scale = static_cast<real>(2.0 / (3.0 * barrier.energy[b]));
          NdArray<real, kN * kN> expected MOCHI_NO_INIT;
          for (int i = 0; i < kN; ++i) {
            for (int j = 0; j < kN; ++j) {
              expected[i * kN + j] = scale * r[i] * r[j];
            }
          }
          ExpectNearL2(expected, GetLane(barrier.dres, b), kTol);
        }
      }
    });
  }
}

// Ordered by increasing g. The residual and tangent saturate near collapse (J ≤ 1e-10) in single
// precision only; exact collapse onto a line or a point (J = 0) saturates every term in both
// precisions. The energy keeps growing past force saturation, and while not fully collapsed, the
// saturated barrier must still push back. At J = 0, the residual vanishes regardless: a is
// quadratic in x, so ∂a/∂x has no component along the collapsed direction.
TEST(ShellWork, CollapseBarrierNearCollapse) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    double previousEnergy = 0.0;
    for (auto const& stretches :
         {Real2{1e-2_r, 1e-2_r},
          Real2{1e-3_r, 1e-3_r},
          Real2{1e-4_r, 1e-4_r},
          Real2{1e-5_r, 1e-5_r},
          Real2{1e-3_r, 1e-7_r},
          Real2{0_r, 1_r},
          Real2{0_r, 1e-3_r},
          Real2{0_r, 0_r}}) {
      bool const isFullyCollapsed = stretches[0] == 0_r;
      auto const cfg = MakeStretchedInteriorConfig(stretches[0], stretches[1]);
      auto const barrier = CollapseBarrierContribution<kBS>(BroadcastConfig<kBS>(cfg));
      for (int b = 0; b < kBS; ++b) {
        EXPECT_TRUE(IsFinite(barrier.energy[b]));
        for (real const v : GetLane(barrier.res, b)) {
          EXPECT_TRUE(IsFinite(v));
        }
        for (real const v : GetLane(barrier.dres, b)) {
          EXPECT_TRUE(IsFinite(v));
        }
        if (isFullyCollapsed) {
          EXPECT_GE(barrier.energy[b], previousEnergy);
        } else {
          EXPECT_GT(barrier.energy[b], previousEnergy);
          EXPECT_GT(Norm(GetLane(barrier.res, b)), 0_r);
          EXPECT_GT(Norm(GetLane(barrier.dres, b)), 0_r);
        }
      }
      if (!isFullyCollapsed) {
        previousEnergy = barrier.energy[kBS - 1];
      }
    }
  });
}

TEST(ShellWork, CollapseBarrierIndependentOfStiffnessDamping) {
  ExpectMembraneTermIndependentOfStiffnessDamping(
      &ShellFdOptions::membraneCollapseBarrierStiffness,
      kCollapseBarrierStiffness,
      MakeCompressedConfigs());
}

// ---------------------------------------------------------------------------
// Membrane strain stiffening k·Σᵢ⟨εᵢ⟩³
// ---------------------------------------------------------------------------

static constexpr real kStrainStiffeningStiffness = 200_r;

template <int kBS>
static ShellOutputs<kBS> StrainStiffeningContribution(
    std::array<ShellStencilConfig const*, kBS> const& cfgs,
    ShellFdOptions options = {},
    bool projectPsd = false) {
  return MembraneTermContribution<kBS>(
      &ShellFdOptions::membraneStrainStiffeningStiffness,
      kStrainStiffeningStiffness,
      cfgs,
      options,
      projectPsd);
}

static std::array<ShellStencilConfig, 5> MakeStretchedConfigs() {
  return {
      MakeStretchedInteriorConfig(1.2_r, 1.1_r),
      MakeStretchedInteriorConfig(1.3_r, 0.75_r),
      MakeStretchedInteriorConfig(1.2_r, 0.9_r, 0.1_r),
      MakeStretchedEquilateralConfig(1.2_r, 1.2_r),
      MakeStretchedEquilateralConfig(1.3_r, 0.8_r, 0.1_r)};
}

// Homogeneous in-plane principal stretches (s₀, s₁) give the energy k·A·Σᵢ⟨εᵢ⟩³ for reference area
// A and principal Green–Lagrange strains εᵢ = ½(sᵢ² − 1). Cases without stretch must leave every
// output unchanged.
TEST(ShellWork, StrainStiffeningEnergyClosedForm) {
  double constexpr kEnergyRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9 : 1e-4;
  std::array<Real2, 10> constexpr kStretches = {
      Real2{1.2_r, 0.7_r},
      Real2{1.3_r, 0.75_r},
      Real2{1.5_r, 1.5_r},
      Real2{2_r, 0.4_r},
      Real2{0.7_r, 1.1_r},
      Real2{1.05_r, 1_r},
      Real2{1.0001_r, 0.9999_r},
      Real2{0.9_r, 0.9_r},
      Real2{0.95_r, 0.7_r},
      Real2{1_r, 1_r}};
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    for (Real2 const& s : kStretches) {
      double strainSum = 0.0;
      for (real const si : s) {
        double const e = Max(0.5 * (Sqr(static_cast<double>(si)) - 1.0), 0.0);
        strainSum += e * e * e;
      }
      for (auto const& [cfg, refArea] :
           {std::pair{MakeStretchedInteriorConfig(s[0], s[1]), 0.5},
            std::pair{MakeStretchedEquilateralConfig(s[0], s[1]), std::numbers::sqrt3 / 4.0}}) {
        auto const stiffening = StrainStiffeningContribution<kBS>(BroadcastConfig<kBS>(cfg));
        double const expected =
            static_cast<double>(kStrainStiffeningStiffness) * refArea * strainSum;
        for (int b = 0; b < kBS; ++b) {
          if (expected == 0.0) {
            ExpectZeroLane(stiffening, b);
          } else {
            EXPECT_NEAR(expected, stiffening.energy[b], kEnergyRelTol * expected);
          }
        }
      }
    }
  });
}

// Permuting the nodes of a sheared triangle stretched along oblique axes changes the covariant
// components of its metric and strain, but not the energy. Its residual and stiffness follow, since
// the FD tests check that they are its exact derivatives.
TEST(ShellWork, StrainStiffeningEnergyInvariantUnderCentralNodePermutations) {
  auto cfg = MakeInteriorConfig();
  Real3 const axis0{0.8_r, 0.6_r, 0_r};
  Real3 const axis1{-0.6_r, 0.8_r, 0_r};
  Real2 const stretches{1.3_r, 0.75_r};
  for (int n = 0; n < kBendingStencilNodes; ++n) {
    auto& position = cfg.refPositions[n];
    position = Real3{1.3_r * position[0] + 0.4_r * position[1], 0.7_r * position[1], position[2]};
    cfg.displacements[n] = (stretches[0] - 1_r) * Dot(axis0, position) * axis0 +
        (stretches[1] - 1_r) * Dot(axis1, position) * axis1;
  }

  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-4_r;
  OutputConfig constexpr kEnergyOnly{.energy = true, .residual = false, .dresidual = false};
  ShellFdOptions options;
  options.membraneLambda = 0_r;
  options.membraneMu = 0_r;
  options.bendingAlpha = 0_r;
  options.bendingBeta = 0_r;
  options.membraneStrainStiffeningStiffness = kStrainStiffeningStiffness;
  double const referenceEnergy =
      RunShell(ShellHarness<1>(BroadcastConfig<1>(cfg), options), kEnergyOnly).energy[0];
  EXPECT_NE(referenceEnergy, 0.0);

  std::array<int, kTriangleNodes> permutation = {0, 1, 2};
  while (std::ranges::next_permutation(permutation).found) {
    SCOPED_TRACE(
        testing::Message() << "permutation = {" << permutation[0] << ", " << permutation[1] << ", "
                           << permutation[2] << "}");
    auto const permutedCfg = PermuteCentralNodes(cfg, permutation);
    ShellHarness<1> const permutedHarness(BroadcastConfig<1>(permutedCfg), options);
    ExpectNearEnergy(referenceEnergy, RunShell(permutedHarness, kEnergyOnly).energy[0], kTol);
  }
}

TEST(ShellWork, StrainStiffeningOutputModes) {
  ShellFdOptions options;
  options.membraneStrainStiffeningStiffness = kStrainStiffeningStiffness;
  for (auto const& cfg : MakeStretchedConfigs()) {
    RunSupportedFemShellRodBatchSizes(
        [&]<int kBS>() { VerifyShellOutputModes<kBS>(BroadcastConfig<kBS>(cfg), options); });
  }
}

// Inactive lanes must be unaffected by active neighbors.
TEST(ShellWork, StrainStiffeningInactiveUnderCompressionBesideStretchedLanes) {
  auto const compressed = MakeStretchedInteriorConfig(0.7_r, 0.8_r);
  auto const stretched = MakeStretchedInteriorConfig(1.3_r, 0.8_r);
  std::array<ShellStencilConfig const*, kDefaultFemBatchSize> mixed{};
  for (int b = 0; b < kDefaultFemBatchSize; ++b) {
    mixed[b] = (b % 2 == 0) ? &stretched : &compressed;
  }
  auto const stiffening = StrainStiffeningContribution<kDefaultFemBatchSize>(mixed);
  for (int b = 0; b < kDefaultFemBatchSize; ++b) {
    if (b % 2 == 0) {
      EXPECT_GT(stiffening.energy[b], 0.0);
    } else {
      ExpectZeroLane(stiffening, b);
    }
  }
}

// The stiffness is exact, so both finite-difference checks apply, including with compressed lanes
// beside stretched ones. Bending is disabled because its stiffness omits the geometric part.
TEST(ShellWork, StrainStiffeningFd) {
  auto const stretched = MakeStretchedConfigs();
  auto const compressed = MakeStretchedInteriorConfig(0.7_r, 0.8_r);
  for (real const svkScale : {0_r, 1_r}) {
    for (real const dampingFactor : {0_r, 3_r}) {
      SCOPED_TRACE(
          testing::Message() << "SVK scale = " << svkScale << ", damping = " << dampingFactor);
      ShellFdOptions options;
      options.membraneLambda = svkScale * kMembraneLambda;
      options.membraneMu = svkScale * kMembraneMu;
      options.bendingAlpha = 0_r;
      options.bendingBeta = 0_r;
      options.membraneStrainStiffeningStiffness = kStrainStiffeningStiffness;
      options.stiffnessDampingFactor = dampingFactor;
      options.stageStartDispScale = 0.5_r;
      RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
        for (auto const& cfg : stretched) {
          VerifyShellFdBroadcast<kBS>(cfg, options);
        }
        if constexpr (kBS > 1) {
          std::array<ShellStencilConfig const*, kBS> mixed{};
          for (int b = 0; b < kBS; ++b) {
            mixed[b] = (b % 2 == 0) ? &stretched[(b / 2) % isize(stretched)] : &compressed;
          }
          VerifyShellFd<kBS>(mixed, options);
        }
      });
    }
  }
}

// The collapse barrier, which is on by default, adds its stress after the stiffening stress.
// Stretching one direction and compressing the other activates both. Only the residual is checked,
// because the barrier's stiffness is approximate.
TEST(ShellWork, StrainStiffeningWithCollapseBarrierEnergyResidualFd) {
  ShellFdOptions options;
  options.checkResidualDResidual = false;
  options.membraneCollapseBarrierStiffness = kCollapseBarrierStiffness;
  options.membraneStrainStiffeningStiffness = kStrainStiffeningStiffness;
  for (auto const& cfg :
       {MakeStretchedInteriorConfig(1.3_r, 0.6_r),
        MakeStretchedEquilateralConfig(0.6_r, 1.3_r, 0.1_r)}) {
    RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
      auto const lanes = BroadcastConfig<kBS>(cfg);
      auto const barrier = CollapseBarrierContribution<kBS>(lanes);
      auto const stiffening = StrainStiffeningContribution<kBS>(lanes);
      for (int b = 0; b < kBS; ++b) {
        EXPECT_GT(barrier.energy[b], 0.0);
        EXPECT_GT(stiffening.energy[b], 0.0);
      }
      VerifyShellFd<kBS>(lanes, options);
    });
  }
}

TEST(ShellWork, StrainStiffeningIndependentOfStiffnessDamping) {
  ExpectMembraneTermIndependentOfStiffnessDamping(
      &ShellFdOptions::membraneStrainStiffeningStiffness,
      kStrainStiffeningStiffness,
      MakeStretchedConfigs());
}

// The exact stiffness is PSD at any stretch, including where it dominates the SVK stiffness, and so
// is the full stiffness with the projected geometric part.
TEST(ShellWork, StrainStiffeningDResidualPsd) {
  auto const stretched = MakeStretchedConfigs();
  DynamicArray<ShellStencilConfig> cfgs(stretched.begin(), stretched.end());
  cfgs.push_back(MakeStretchedInteriorConfig(2_r, 0.4_r, 0.1_r));
  cfgs.push_back(MakeStretchedEquilateralConfig(1.8_r, 1.6_r, 0.1_r));
  auto const expectPsd = [](auto const& dres) {
    ASSERT_GT(Norm(dres), 0_r);
    ExpectSymmetricPsd(dres);
  };
  ShellFdOptions options;
  options.membraneStrainStiffeningStiffness = kStrainStiffeningStiffness;
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() {
    for (auto const& cfg : cfgs) {
      auto const stiffening = StrainStiffeningContribution<kBS>(BroadcastConfig<kBS>(cfg));
      auto const projected = RunShell(
          ShellHarness<kBS>(BroadcastConfig<kBS>(cfg), options),
          {.energy = false, .residual = false, .dresidual = true},
          /*projectPsd*/ true);
      for (int b = 0; b < kBS; ++b) {
        expectPsd(GetLane(stiffening.dres, b));
        expectPsd(GetLane(projected.dres, b));
      }
    }
  });
}
