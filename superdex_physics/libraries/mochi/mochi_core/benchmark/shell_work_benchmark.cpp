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

#include "config.h"

#include <mochi_core/element_operations/fem_shell.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <functional>
#include <memory>

using namespace mochi;
using namespace mochi::fem;

namespace mochi_benchmark {

// -------------------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------------------

static constexpr int kBatchSize = Min(2 * Simd<real>::kSize, 8);

enum class BenchmarkMode { Objective, Residual, DresidualWithoutPsd, DresidualWithPsd, All };

// Interior: every batch lane is an interior triangle (complete 6-node bending stencil).
// Boundary: every batch lane is a boundary triangle.
enum class BatchKind { Interior, Boundary };

// -------------------------------------------------------------------------------------
// Shared benchmark data: real-world sized triangular mesh with bending stencils
// -------------------------------------------------------------------------------------

using ElementT = triangular::Pk2DElement<1, 1>;
static constexpr int kStencilDofs = kBendingStencilNodes * kSpaceDim3;

namespace {

struct ShellWorkBenchmarkData {
  std::unique_ptr<TriangularMesh> mesh = {};
  DynamicArray<ElementT> elements;

  // Per-element bending stencil: 6 global node indices (kSentinelIndex for boundary).
  DynamicArray<NdArray<int, kBendingStencilNodes>> stencils;

  int numElements = 0;

  // Per-batch-lane data.
  NdArray<real, kBatchSize, kStencilDofs> displacements = {};

  // All-interior and all-boundary batches.
  NdArray<int, kBatchSize> interiorIndices = {};
  NdArray<int, kBatchSize> boundaryIndices = {};

  // Material parameters.
  real membraneLambda = 0_r;
  real membraneMu = 0_r;
  real bendingAlpha = 0_r;
  real bendingBeta = 0_r;

  ShellWorkBenchmarkData() {
    // Generate a 64x64 uniform square mesh (~8192 triangles, ~4225 nodes).
    // Large enough to exercise realistic cache/memory behavior.
    auto [coordinates, connectivity] =
        UniformSquareTriangularMeshData(Int2{64, 64}, Real2{1_r, 1_r}, 2);
    mesh = std::make_unique<TriangularMesh>(MakeSpan(coordinates), MakeSpan(connectivity));

    numElements = mesh->GetNumElements();
    elements.reserve(numElements);
    for (int i = 0; i < numElements; ++i) {
      elements.emplace_back(i, mesh->GetNodeCoordinates(), mesh->GetElementConnectivity());
    }

    // Build bending stencils (6-node: 3 triangle + 3 neighbors).
    auto bendingConnAndStencil = mesh->GenerateBendingConnectivityAndStencil();
    auto const& bendingConn = bendingConnAndStencil.first;
    auto const& stencilIndicesAll = bendingConnAndStencil.second;

    stencils.resize(numElements);
    for (int e = 0; e < numElements; ++e) {
      auto const& eleNodes = bendingConn[e];
      auto const& eleStencil = stencilIndicesAll[e];
      int const numLocalNodes = isize(eleStencil);

      for (int n = 0; n < kBendingStencilNodes; ++n) {
        stencils[e][n] = kSentinelIndex;
      }
      for (int i = 0; i < numLocalNodes; ++i) {
        int const stencilPos = eleStencil[i];
        stencils[e][stencilPos] = eleNodes[i];
      }
    }

    // Random displacements for each batch lane.
    auto gen = RandomGenerator(42);
    for (int b = 0; b < kBatchSize; ++b) {
      SetRandom(gen, -0.01_r, 0.01_r, displacements[b]);
    }

    // Partition elements into interior and boundary, then build an all-interior batch and an
    // all-boundary batch.
    DynamicArray<int> interiorList;
    DynamicArray<int> boundaryList;
    for (int e = 0; e < numElements; ++e) {
      bool complete = true;
      for (int n = 0; n < kBendingStencilNodes; ++n) {
        if (stencils[e][n] == kSentinelIndex) {
          complete = false;
          break;
        }
      }
      if (complete) {
        interiorList.push_back(e);
      } else {
        boundaryList.push_back(e);
      }
    }

    MOCHI_ASSERT(
        !interiorList.empty() && !boundaryList.empty(), "No interior or boundary elements.");
    for (int b = 0; b < kBatchSize; ++b) {
      interiorIndices[b] = interiorList[b % interiorList.size()];
      boundaryIndices[b] = boundaryList[b % boundaryList.size()];
    }

    // Material parameters (plane-stress SVK).
    constexpr real kYoungsModulus = 1e3_r;
    constexpr real kPoissonRatio = 0.3_r;
    membraneLambda =
        (kYoungsModulus * kPoissonRatio) / ((1_r + kPoissonRatio) * (1_r - 2_r * kPoissonRatio));
    membraneMu = kYoungsModulus / (2_r * (1_r + kPoissonRatio));
    bendingAlpha = 1e-6_r;
    bendingBeta = 1e-6_r;
  }
};

} // namespace

static auto const& GetSharedData() {
  static ShellWorkBenchmarkData const data;
  return data;
}

// -------------------------------------------------------------------------------------
// ShellWork benchmark
// -------------------------------------------------------------------------------------

static void BenchmarkShellWork(benchmark::State& state, BenchmarkMode mode, BatchKind batchKind) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;

  auto const& data = GetSharedData();

  bool const evalObj = (mode == BenchmarkMode::Objective || mode == BenchmarkMode::All);
  bool const evalRes = (mode == BenchmarkMode::Residual || mode == BenchmarkMode::All);
  bool const evalDRes =
      (mode == BenchmarkMode::DresidualWithPsd || mode == BenchmarkMode::DresidualWithoutPsd ||
       mode == BenchmarkMode::All);
  bool const projectPsd = (mode == BenchmarkMode::DresidualWithPsd || mode == BenchmarkMode::All);

  // Pre-pack stencil global nodes per lane (interior or boundary batch).
  auto const& elementIndices =
      (batchKind == BatchKind::Boundary) ? data.boundaryIndices : data.interiorIndices;
  NdArray<int, kBendingStencilNodes, kBatchSize> stencilGlobalNodes;
  for (int b = 0; b < kBatchSize; ++b) {
    auto const& stencil = data.stencils[elementIndices[b]];
    for (int n = 0; n < kBendingStencilNodes; ++n) {
      stencilGlobalNodes[n][b] = stencil[n];
    }
  }

  // Pre-pack displacements into SIMD (AoS → SoA).
  alignas(alignof(V)) real staging[V::kSize]{};
  NdArray<V, kStencilDofs> batchedDisp;
  for (int d = 0; d < kStencilDofs; ++d) {
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = data.displacements[b][d];
    }
    batchedDisp[d] = Load<V>(staging);
  }

  Vd energy{};
  NdArray<V, kStencilDofs> res{};
  NdArray<V, kStencilDofs * kStencilDofs> dres{};

  std::function<void()> fn = [&]() {
    energy = {};
    res = {};
    dres = {};
    fem::ShellWork<kBatchSize>(
        stencilGlobalNodes,
        data.mesh->GetNodeCoordinates(),
        batchedDisp,
        evalObj ? &energy : nullptr,
        evalRes ? &res : nullptr,
        evalDRes ? &dres : nullptr,
        data.membraneLambda,
        data.membraneMu,
        data.bendingAlpha,
        data.bendingBeta,
        projectPsd);
  };

  for (auto _ : state) {
    CallNoInline(fn);
  }
  state.counters["elements/s"] =
      benchmark::Counter(state.iterations() * kBatchSize, benchmark::Counter::kIsRate);
}

// -------------------------------------------------------------------------------------
// Benchmark registration
// -------------------------------------------------------------------------------------

// clang-format off
#define MOCHI_SHELL_WORK_MODES(Prefix, Kind)                                                                     \
  BENCHMARK_CAPTURE(BenchmarkShellWork, Prefix##_Objective,           BenchmarkMode::Objective,           Kind); \
  BENCHMARK_CAPTURE(BenchmarkShellWork, Prefix##_Residual,            BenchmarkMode::Residual,            Kind); \
  BENCHMARK_CAPTURE(BenchmarkShellWork, Prefix##_DresidualWithoutPsd, BenchmarkMode::DresidualWithoutPsd, Kind); \
  BENCHMARK_CAPTURE(BenchmarkShellWork, Prefix##_DresidualWithPsd,    BenchmarkMode::DresidualWithPsd,    Kind); \
  BENCHMARK_CAPTURE(BenchmarkShellWork, Prefix##_All,                 BenchmarkMode::All,                 Kind);
// clang-format on

MOCHI_SHELL_WORK_MODES(ShellWorkInterior, BatchKind::Interior)
MOCHI_SHELL_WORK_MODES(ShellWorkBoundary, BatchKind::Boundary)

} // namespace mochi_benchmark
