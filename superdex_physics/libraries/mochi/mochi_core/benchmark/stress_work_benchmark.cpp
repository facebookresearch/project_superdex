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

#include <mochi_core/element_operations/fem_stress.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/materials/batched_linear_elastic.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <functional>

using namespace mochi;
using namespace mochi::materials;

namespace mochi_benchmark {

static constexpr int kBatchSize = kDefaultFemBatchSize;
static constexpr int kNumElements = 1024;
static constexpr int kNumQuadPoints = 1; // Default for soft actors

using ElementT = tetrahedral::Pk3DElement<1, kNumQuadPoints>;
static constexpr int kDim = ElementT::kSpaceDim * ElementT::kNumDofs;

enum class BenchmarkMode { Objective, Residual, DresidualWithoutPsd, DresidualWithPsd, All };

template <typename BatchedMaterialFn>
static void BenchmarkStressWork(
    benchmark::State& state,
    BatchedMaterialFn batchedMaterialFn,
    BenchmarkMode mode) {
  using V = BatchReal<kBatchSize>;

  constexpr Real3 kCoordinates[] = {
      Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}};
  constexpr Int4 kConnectivity[] = {Int4{0, 1, 2, 3}};
  TetrahedralMesh mesh(kCoordinates, kConnectivity);

  DynamicArray<ElementT> elements;
  elements.reserve(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    elements.emplace_back(
        0,
        mesh.GetNodeCoordinates(),
        mesh.GetElementConnectivity(),
        tetrahedral::kTetrahedralQuadrature1);
  }

  NdArray<NdArray<real, kDim>, kBatchSize> displacements{};
  NdArray<int, kBatchSize> elementIndices{};
  auto gen = RandomGenerator(42);
  for (int b = 0; b < kBatchSize; ++b) {
    SetRandom(gen, -0.3_r, 0.3_r, displacements[b]);
    elementIndices[b] = b * (kNumElements / kBatchSize);
  }

  bool const evalObj = (mode == BenchmarkMode::Objective || mode == BenchmarkMode::All);
  bool const evalRes = (mode == BenchmarkMode::Residual || mode == BenchmarkMode::All);
  bool const evalDRes =
      (mode == BenchmarkMode::DresidualWithPsd || mode == BenchmarkMode::DresidualWithoutPsd ||
       mode == BenchmarkMode::All);
  bool const projectPsd = (mode == BenchmarkMode::DresidualWithPsd || mode == BenchmarkMode::All);

  alignas(alignof(V)) real staging[V::kSize]{};
  NdArray<V, kDim> batchedDisp{};
  for (int d = 0; d < kDim; ++d) {
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = displacements[b][d];
    }
    batchedDisp[d] = Load<V>(staging);
  }

  BatchDouble<kBatchSize> energy{0.0};
  NdArray<V, kDim> res{};
  NdArray<V, kDim * kDim> dres{};
  std::function<void()> fn = [&]() {
    energy = 0.0;
    res = {};
    dres = {};
    fem::StressWork<kBatchSize>(
        elementIndices,
        MakeConstSpan(elements),
        batchedDisp,
        evalObj ? &energy : nullptr,
        evalRes ? &res : nullptr,
        evalDRes ? &dres : nullptr,
        projectPsd,
        batchedMaterialFn);
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

static void LinearElastic_StressWork(benchmark::State& state, BenchmarkMode mode) {
  auto const params = materials::BuildPerElementParams(LinearElasticMaterialParams{});
  BenchmarkStressWork(
      state,
      materials::MakeBatchedConstitutiveResponse<LinearElasticMaterialParams, kBatchSize>(params),
      mode);
}

static void SmithNeoHookean_StressWork(benchmark::State& state, BenchmarkMode mode) {
  auto const params = materials::BuildPerElementParams(SmithNeoHookeanMaterialParams{});
  BenchmarkStressWork(
      state,
      materials::MakeBatchedConstitutiveResponse<SmithNeoHookeanMaterialParams, kBatchSize>(
          params, /* Ensure projectPsd is respected */ MaterialPsdOracle::None),
      mode);
}

// clang-format off
#define MOCHI_STRESS_WORK_BENCHMARK_MODES(Name)                                                         \
  BENCHMARK_CAPTURE(Name##_StressWork, Name##_Objective,           BenchmarkMode::Objective);           \
  BENCHMARK_CAPTURE(Name##_StressWork, Name##_Residual,            BenchmarkMode::Residual);            \
  BENCHMARK_CAPTURE(Name##_StressWork, Name##_DresidualWithoutPsd, BenchmarkMode::DresidualWithoutPsd); \
  BENCHMARK_CAPTURE(Name##_StressWork, Name##_DresidualWithPsd,    BenchmarkMode::DresidualWithPsd);    \
  BENCHMARK_CAPTURE(Name##_StressWork, Name##_All,                 BenchmarkMode::All);
// clang-format on

MOCHI_STRESS_WORK_BENCHMARK_MODES(LinearElastic)
MOCHI_STRESS_WORK_BENCHMARK_MODES(SmithNeoHookean)

} // namespace mochi_benchmark
