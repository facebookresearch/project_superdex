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

#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>

#include <functional>

using namespace mochi;
using namespace mochi::materials;

namespace mochi_benchmark {

enum class BenchmarkMode { Objective, Residual, DresidualWithoutPsd, DresidualWithPsd, All };

static constexpr int kBatchSize = kDefaultFemBatchSize;

template <typename EvalFn>
static void BenchmarkMaterial(benchmark::State& state, EvalFn eval, BenchmarkMode mode) {
  // Arbitrary deformation gradient to ensure SVD-backed materials take a non-trivial path.
  Matrix3x3r const scalarF{
      Real3{1.1_r, 0.05_r, 0.02_r}, Real3{0.03_r, -1.15_r, 0.04_r}, Real3{0.01_r, 0.06_r, 1.08_r}};
  BatchReal3x3<kBatchSize> F{};
  for (int lane = 0; lane < kBatchSize; ++lane) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        F[r][c] = Set(F[r][c], lane, scalarF[r][c]);
      }
    }
  }

  BatchDouble<kBatchSize> energy{0.0};
  BatchReal3x3<kBatchSize> pk1{};
  NdArray<BatchReal3x3<kBatchSize>, 3, 3> tangent{};
  auto* energyPtr =
      (mode == BenchmarkMode::Objective || mode == BenchmarkMode::All) ? &energy : nullptr;
  auto* pk1Ptr = (mode == BenchmarkMode::Residual || mode == BenchmarkMode::All) ? &pk1 : nullptr;
  auto* tangentPtr = (mode == BenchmarkMode::DresidualWithPsd ||
                      mode == BenchmarkMode::DresidualWithoutPsd || mode == BenchmarkMode::All)
      ? &tangent
      : nullptr;
  bool const projectPsd = (mode == BenchmarkMode::DresidualWithPsd || mode == BenchmarkMode::All);

  std::function<void()> fn = [&]() { eval(F, energyPtr, pk1Ptr, tangentPtr, projectPsd); };
  for (auto _ : state) {
    CallNoInline(fn);
  }
  state.counters["elements/s"] =
      benchmark::Counter(state.iterations() * kBatchSize, benchmark::Counter::kIsRate);
}

// clang-format off
#define MOCHI_MATERIAL_BENCHMARK_MODE(MaterialName, ModeName)                                       \
  BENCHMARK_CAPTURE(MaterialName##Benchmark, MaterialName##_##ModeName, BenchmarkMode::ModeName)    \
      ->Name("Material/" #MaterialName "/" #ModeName);

#define MOCHI_MATERIAL_BENCHMARK_MODES(MaterialName)                                                \
  MOCHI_MATERIAL_BENCHMARK_MODE(MaterialName, Objective)                                            \
  MOCHI_MATERIAL_BENCHMARK_MODE(MaterialName, Residual)                                             \
  MOCHI_MATERIAL_BENCHMARK_MODE(MaterialName, DresidualWithoutPsd)                                  \
  MOCHI_MATERIAL_BENCHMARK_MODE(MaterialName, DresidualWithPsd)                                     \
  MOCHI_MATERIAL_BENCHMARK_MODE(MaterialName, All)

#if MOCHI_HAS_VA_OPT
#define MOCHI_MATERIAL_BENCHMARK_OPTIONAL_ARGS(...) __VA_OPT__(, ) __VA_ARGS__
#else
#define MOCHI_MATERIAL_BENCHMARK_OPTIONAL_ARGS(...) , ##__VA_ARGS__
#endif // MOCHI_HAS_VA_OPT

#define MOCHI_MATERIAL_BENCHMARK(Name, ParamsType, BatchedFunc, ...)                                 \
  static void Name##Benchmark(benchmark::State& state, BenchmarkMode mode) {                         \
    auto const params = materials::BuildBatchParams<kBatchSize>(ParamsType{});                       \
    BenchmarkMaterial(                                                                               \
        state,                                                                                       \
        [&](auto const& F, auto* energy, auto* pk1, auto* tangent, bool projectPsd) {                \
          BatchedFunc<kBatchSize>(                                                                   \
              params,                                                                                \
              F,                                                                                     \
              energy,                                                                                \
              pk1,                                                                                   \
              tangent,                                                                               \
              projectPsd MOCHI_MATERIAL_BENCHMARK_OPTIONAL_ARGS(__VA_ARGS__));                       \
        },                                                                                           \
        mode);                                                                                       \
  }                                                                                                  \
  MOCHI_MATERIAL_BENCHMARK_MODES(Name)

MOCHI_MATERIAL_BENCHMARK(LinearElastic, LinearElasticMaterialParams, BatchedLinearElasticConstitutiveResponse)
MOCHI_MATERIAL_BENCHMARK(StVenantKirchhoff, StVenantKirchhoffMaterialParams, BatchedStVenantKirchhoffConstitutiveResponse)
MOCHI_MATERIAL_BENCHMARK(
    SmithNeoHookean,
    SmithNeoHookeanMaterialParams,
    BatchedSmithNeoHookeanConstitutiveResponse,
    MaterialPsdOracle::None) // Ensure projectPsd is respected
MOCHI_MATERIAL_BENCHMARK(KimNeoHookean, KimNeoHookeanMaterialParams, BatchedKimNeoHookeanConstitutiveResponse)
MOCHI_MATERIAL_BENCHMARK(Arap, ArapMaterialParams, BatchedArapConstitutiveResponse)
MOCHI_MATERIAL_BENCHMARK(ActiveAnisoArap, ActiveAnisoArapMaterialParams, BatchedActiveAnisoArapConstitutiveResponse)
MOCHI_MATERIAL_BENCHMARK(ActiveShapeTargetingArap, ActiveShapeTargetingArapMaterialParams, BatchedActiveShapeTargetingArapConstitutiveResponse)
MOCHI_MATERIAL_BENCHMARK(ActiveNeoHookean, ActiveNeoHookeanMaterialParams, BatchedActiveNeoHookeanConstitutiveResponse)
// clang-format on

} // namespace mochi_benchmark
