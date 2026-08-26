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
#include <mochi_core/element_operations/fem_stress.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/materials/batched_st_venant_kirchhoff.h>
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
#include <optional>

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
  real membraneLambda = kMembraneLambda;
  real membraneMu = kMembraneMu;
  real bendingAlpha = kBendingAlpha;
  real bendingBeta = kBendingBeta;
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
      da_dx, dpsi_da, AInv, referenceArea, V{1_r}, V{2_r}, /*projectPsd*/ false, dres);
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

static void ExpectSymmetricPsd(NdArray<real, kBendingStencilDofs * kBendingStencilDofs> const& m) {
  constexpr int kN = kBendingStencilDofs;
  real constexpr kSymAbsTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-9_r : 1e-5_r;
  real constexpr kSymRelTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-6_r;
  double constexpr kPsdTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8 : 1e-4;

  for (int i = 0; i < kN; ++i) {
    for (int j = 0; j < kN; ++j) {
      real const a = m[i * kN + j];
      real const b = m[j * kN + i];
      EXPECT_NEAR_TOL(a, b, Max(kSymAbsTol, kSymRelTol * Max(Abs(a), Abs(b))));
    }
  }

  for (int i = 0; i < kN; ++i) {
    NdArray<real, kN> v{};
    v[i] = 1_r;
    EXPECT_GE(QuadraticForm<kN>(m, v), -kPsdTol);
  }
  for (int i = 0; i < kN; ++i) {
    for (int j = i + 1; j < kN; ++j) {
      NdArray<real, kN> plus{};
      plus[i] = 1_r;
      plus[j] = 1_r;
      EXPECT_GE(QuadraticForm<kN>(m, plus), -kPsdTol);

      NdArray<real, kN> minus{};
      minus[i] = 1_r;
      minus[j] = -1_r;
      EXPECT_GE(QuadraticForm<kN>(m, minus), -kPsdTol);
    }
  }
}

template <int kBS>
static void VerifyShellFd(
    std::array<ShellStencilConfig const*, kBS> const& cfgs,
    ShellFdOptions options = {}) {
  using V = BatchReal<kBS>;
  constexpr int kN = kBendingStencilDofs;
  ShellHarness<kBS> h(cfgs, options);

  if (options.checkEnergyResidual || options.checkResidualDResidual) {
    real constexpr kEps = MOCHI_USE_DOUBLE_PRECISION ? 1e-6_r : 1e-3_r;
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
      dispP[j] = dispP[j] + V{kEps};
      dispM[j] = dispM[j] - V{kEps};
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
          resFd[b][j] = static_cast<real>((ep[b] - em[b]) / (2.0 * static_cast<double>(kEps)));
        }
        if (options.checkResidualDResidual) {
          for (int i = 0; i < kN; ++i) {
            dresFd[b][i * kN + j] = (rp[i][b] - rm[i][b]) / (2_r * kEps);
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
static void VerifyShellOutputModes(std::array<ShellStencilConfig const*, kBS> const& cfgs) {
  ShellHarness<kBS> h(cfgs);
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

static void VerifyMembraneMatchesStressWork() {
  real constexpr kYoungsModulus = 1e6_r;
  real constexpr kPoissonsRatio = 0.3_r;
  real const membraneLambda =
      (kYoungsModulus * kPoissonsRatio) / ((1_r + kPoissonsRatio) * (1_r - 2_r * kPoissonsRatio));
  real const membraneMu = kYoungsModulus / (2_r * (1_r + kPoissonsRatio));
  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-8_r : 1e-3_r;

  for (auto const& cfg : MakeHelperConfigs()) {
    DynamicArray<Real3> coordinates;
    coordinates.reserve(kBendingStencilNodes);
    for (int n = 0; n < kBendingStencilNodes; ++n) {
      coordinates.push_back(cfg.refPositions[n]);
    }

    DynamicArray<Int3> connectivity = {{0, 1, 2}};
    using ElementT = triangular::Pk2DElement<1, 1>;
    ElementT element(0, coordinates, connectivity);
    DynamicArray<ElementT> elements = {element};

    BatchDouble<1> shellEnergy{};
    ShellBatchVector<1> shellRes{};
    bool success = ShellWork<1>(
        BroadcastStencilNodes<1>(cfg.globalNodeIndices),
        MakeConstSpan(coordinates),
        BroadcastNodeValues<1>(cfg.displacements),
        &shellEnergy,
        &shellRes,
        nullptr,
        membraneLambda,
        membraneMu,
        0_r,
        0_r,
        false);
    EXPECT_TRUE(success);

    BatchElementVector<1, ElementT> stressDisp{};
    for (int dof = 0; dof < ElementT::kNumDofs * ElementT::kSpaceDim; ++dof) {
      stressDisp[dof] = BatchReal<1>{cfg.displacements[dof / kSpaceDim3][dof % kSpaceDim3]};
    }

    StVenantKirchhoffMaterialParams materialParams;
    materialParams.youngsModulus = kYoungsModulus;
    materialParams.poissonRatio = kPoissonsRatio;
    materialParams.psdStrategy = MaterialPsdStrategy::None;
    auto const perElementParams = materials::BuildPerElementParams(materialParams);
    auto const constitutiveResponse =
        materials::MakeBatchedConstitutiveResponse<StVenantKirchhoffMaterialParams, 1>(
            perElementParams);

    BatchDouble<1> stressEnergy{};
    BatchElementVector<1, ElementT> stressRes{};
    success = StressWork<1>(
        NdArray<int, 1>{0},
        MakeConstSpan(elements),
        stressDisp,
        &stressEnergy,
        &stressRes,
        nullptr,
        false,
        constitutiveResponse);
    EXPECT_TRUE(success);

    EXPECT_NEAR_TOL(
        static_cast<real>(shellEnergy[0]),
        static_cast<real>(stressEnergy[0]),
        Max(kTol, kTol * Abs(static_cast<real>(stressEnergy[0]))));
    for (int dof = 0; dof < ElementT::kNumDofs * ElementT::kSpaceDim; ++dof) {
      real const ref = stressRes[dof][0];
      EXPECT_NEAR_TOL(shellRes[dof][0], ref, Max(kTol, kTol * Abs(ref)));
    }
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

TEST(ShellHelpers, KelvinInversion) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyKelvinInversionHelpers<kBS>(); });
}

TEST(ShellHelpers, MembraneResiduals) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyMembraneResidualHelpers<kBS>(); });
}

TEST(ShellHelpers, BendingResiduals) {
  RunSupportedFemShellRodBatchSizes([&]<int kBS>() { VerifyBendingResidualHelpers<kBS>(); });
}

TEST(ShellWork, MembraneMatchesStressWork) {
  VerifyMembraneMatchesStressWork();
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
