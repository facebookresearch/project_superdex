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

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <functional>

using namespace mochi;
using namespace mochi::fem;

namespace mochi_benchmark {

// -------------------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------------------

static constexpr int kBatchSize = Min(2 * Simd<real>::kSize, 8);
static constexpr int kDofs = kNumRodStencilDofs;

enum class BenchmarkMode { Objective, Residual, DresidualWithoutPsd, DresidualWithPsd, All };

// -------------------------------------------------------------------------------------
// Shared benchmark data: rod stencil with realistic parameters
// -------------------------------------------------------------------------------------

namespace {

struct RodWorkBenchmarkData {
  DynamicArray<Real3> meshNodes;
  DynamicArray<real> nodalMasses;
  DynamicArray<Real3> frameAxes;
  DynamicArray<Real3> referenceAxes;
  DynamicArray<int> l2gFlat;
  static constexpr real kAxialStiffness = 1e4_r;
  static constexpr Real2 kFlexuralStiffness = {1e-2_r, 1e-2_r};
  static constexpr real kTorsionalStiffness = 5e-3_r;
  static constexpr Real3 kGravity = {0_r, -9.81_r, 0_r};

  int numStencils;

  NdArray<NdArray<real, kDofs>, kBatchSize> displacements;
  NdArray<int, kBatchSize> elementIndices;

  RodWorkBenchmarkData() {
    // Create a straight rod with 200 nodes (199 elements, 200 stencils).
    int const numNodes = 200;
    numStencils = numNodes;
    real const segmentLength = 0.01_r;

    meshNodes.resize(numNodes);
    nodalMasses.resize(numNodes);
    frameAxes.resize(numNodes - 1);
    referenceAxes.resize(numNodes - 1);
    l2gFlat.resize(numStencils * kDofs);

    for (int i = 0; i < numNodes; ++i) {
      meshNodes[i] = Real3{StaticCast<real>(i) * segmentLength, 0_r, 0_r};
      nodalMasses[i] = 0.001_r;
    }
    for (int i = 0; i < numNodes - 1; ++i) {
      frameAxes[i] = Real3{0_r, 1_r, 0_r};
      referenceAxes[i] = Real3{0_r, 1_r, 0_r};
    }

    // Build padded L2G: interior stencils have 3 distinct nodes; boundary stencils are padded.
    for (int s = 0; s < numStencils; ++s) {
      for (int n = 0; n < kNumRodStencilNodes; ++n) {
        int const nodeIdx = Min(s + n, numNodes - 1);
        for (int d = 0; d < kNumRodFields; ++d) {
          l2gFlat[s * kDofs + n * kNumRodFields + d] = nodeIdx * kNumRodFields + d;
        }
      }
    }

    // Random displacements for each batch lane, evenly spaced element indices.
    auto gen = RandomGenerator(42);
    for (int b = 0; b < kBatchSize; ++b) {
      SetRandom(gen, -0.001_r, 0.001_r, displacements[b]);
      elementIndices[b] = 1 + b * ((numStencils - 2) / kBatchSize); // skip boundary stencils
    }
  }
};

} // namespace

static auto const& GetSharedData() {
  static RodWorkBenchmarkData const data;
  return data;
}

// -------------------------------------------------------------------------------------
// Rod work benchmark
// -------------------------------------------------------------------------------------

static void BenchmarkRodWork(benchmark::State& state, BenchmarkMode mode) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;

  auto const& data = GetSharedData();

  bool const evalObj = (mode == BenchmarkMode::Objective || mode == BenchmarkMode::All);
  bool const evalRes = (mode == BenchmarkMode::Residual || mode == BenchmarkMode::All);
  bool const evalDRes =
      (mode == BenchmarkMode::DresidualWithPsd || mode == BenchmarkMode::DresidualWithoutPsd ||
       mode == BenchmarkMode::All);
  bool const projectPsd = (mode == BenchmarkMode::DresidualWithPsd || mode == BenchmarkMode::All);

  // Pre-pack displacements into SIMD (AoS → SoA).
  alignas(alignof(V)) real staging[V::kSize]{};
  NdArray<V, kDofs> batchedDisp;
  for (int d = 0; d < kDofs; ++d) {
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = data.displacements[b][d];
    }
    batchedDisp[d] = Load<V>(staging);
  }

  Vd energy{};
  NdArray<V, kDofs> batchRes{};
  NdArray<V, kDofs * kDofs> batchDRes{};

  std::function<void()> fn = [&]() {
    energy = {};
    batchRes = {};
    batchDRes = {};

    RodGravity<kBatchSize>(
        MakeConstSpan(data.nodalMasses),
        batchedDisp,
        data.elementIndices,
        MakeConstSpan(data.l2gFlat),
        evalObj ? &energy : nullptr,
        evalRes ? &batchRes : nullptr,
        RodWorkBenchmarkData::kGravity);

    RodAxialStress<kBatchSize>(
        MakeConstSpan(data.meshNodes),
        batchedDisp,
        data.elementIndices,
        MakeConstSpan(data.l2gFlat),
        evalObj ? &energy : nullptr,
        evalRes ? &batchRes : nullptr,
        evalDRes ? &batchDRes : nullptr,
        RodWorkBenchmarkData::kAxialStiffness,
        projectPsd);

    RodBendTwistStress<kBatchSize>(
        MakeConstSpan(data.meshNodes),
        MakeConstSpan(data.frameAxes),
        MakeConstSpan(data.referenceAxes),
        batchedDisp,
        data.elementIndices,
        MakeConstSpan(data.l2gFlat),
        evalObj ? &energy : nullptr,
        evalRes ? &batchRes : nullptr,
        evalDRes ? &batchDRes : nullptr,
        RodWorkBenchmarkData::kFlexuralStiffness,
        RodWorkBenchmarkData::kTorsionalStiffness);
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

#define MOCHI_ROD_WORK_MODES(Prefix, Func)                                                   \
  BENCHMARK_CAPTURE(Func, Prefix##_Objective, BenchmarkMode::Objective);                     \
  BENCHMARK_CAPTURE(Func, Prefix##_Residual, BenchmarkMode::Residual);                       \
  BENCHMARK_CAPTURE(Func, Prefix##_DresidualWithoutPsd, BenchmarkMode::DresidualWithoutPsd); \
  BENCHMARK_CAPTURE(Func, Prefix##_DresidualWithPsd, BenchmarkMode::DresidualWithPsd);       \
  BENCHMARK_CAPTURE(Func, Prefix##_All, BenchmarkMode::All);

MOCHI_ROD_WORK_MODES(RodWork, BenchmarkRodWork)

} // namespace mochi_benchmark
