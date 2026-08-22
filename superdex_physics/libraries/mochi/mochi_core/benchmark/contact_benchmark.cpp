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

#include <mochi_core/contact/contact_utils.h>

using namespace mochi;

namespace mochi_benchmark {

template <int kBatchSize>
static void ComputeBatchCollisionForceDForce(benchmark::State& state) {
  auto const assemEnergy = static_cast<bool>(state.range(0));
  auto const assemForce = static_cast<bool>(state.range(1));
  auto const assemDForce = static_cast<bool>(state.range(2));
  auto const isSdfGradUnitary = static_cast<bool>(state.range(3));
  auto const explicitNormals = static_cast<bool>(state.range(4));
  auto const fullDissipation = static_cast<bool>(state.range(5));

  DynamicArray<Real3> posColliding;
  DynamicArray<Real3> posCollidingStageStart;
  SdfInfo sdfInfo;
  SdfInfo sdfInfoStageStart;
  DynamicArray<Real3> normalColliding;
  ContactParams params = {}; // Default contact params.
  ContactEvalConfig config = {.explicitNormals = explicitNormals};
  if (fullDissipation) {
    params.viscousFrictionCoefficient = 1_r;
    params.normalViscousDampingCoefficient = 1_r;
    config.useFittedHessian = false;
  }

  posColliding.resize(kBatchSize);
  posCollidingStageStart.resize(kBatchSize);
  sdfInfo.resize(kBatchSize);
  sdfInfoStageStart.resize(kBatchSize);
  normalColliding.resize(kBatchSize);

  double energyBuffer[kBatchSize];
  Real3 forceBuffer[kBatchSize];
  VMatrix3x3r dforceBuffer[kBatchSize];
  Real3 normal = {1_r, -1_r, 1_r};

  auto outEnergy = assemEnergy ? MakeSpan(energyBuffer) : Span<double>{};
  auto outForce = assemForce ? MakeSpan(forceBuffer) : Span<Real3>{};
  auto outDForce = assemDForce ? MakeSpan(dforceBuffer) : Span<VMatrix3x3r>{};

  for (int i = 0; i < kBatchSize; ++i) {
    normalColliding[i] = -Normalize(normal);
    sdfInfo.grad[i] = isSdfGradUnitary ? -normalColliding[i] : normal;
    sdfInfoStageStart.grad[i] = sdfInfo.grad[i];
  }

  for (auto x : state) {
    CallNoInline([&]() {
      ComputeBatchCollisionForceDForce<kBatchSize, GradTarget::Current>(
          outEnergy,
          outForce,
          outDForce,
          MakeConstSpan(sdfInfo.val),
          MakeConstSpan(sdfInfo.grad),
          MakeConstSpan(sdfInfoStageStart.val),
          MakeConstSpan(sdfInfoStageStart.grad),
          MakeConstSpan(normalColliding),
          MakeConstSpan(posColliding),
          MakeConstSpan(posCollidingStageStart),
          params,
          config,
          /*dtStage*/ 0.01_r,
          assemEnergy,
          assemForce,
          assemDForce,
          isSdfGradUnitary);
    });
  }

  state.counters["Points/second"] =
      benchmark::Counter(state.iterations() * kBatchSize, benchmark::Counter::kIsRate);
}

static constexpr int kBatchSize = Simd<real>::kSize;

#define MOCHI_CONTACT_BENCHMARK(benchmarkName, ...)                \
  BENCHMARK_TEMPLATE(ComputeBatchCollisionForceDForce, kBatchSize) \
      ->Name("Contact/BatchCollisionForceDForce/" benchmarkName)   \
      ->ArgNames(                                                  \
          {"objective",                                            \
           "residual",                                             \
           "dresidual",                                            \
           "unitSdfGradient",                                      \
           "explicitNormals",                                      \
           "fullDissipation"})                                     \
      ->Args({__VA_ARGS__})

MOCHI_CONTACT_BENCHMARK("Objective", true, false, false, true, false, false);
MOCHI_CONTACT_BENCHMARK("Residual", false, true, false, true, false, false);
MOCHI_CONTACT_BENCHMARK("DResidual", false, false, true, true, false, false);
MOCHI_CONTACT_BENCHMARK("ResidualDResidual", false, true, true, true, false, false);
MOCHI_CONTACT_BENCHMARK("All", true, true, true, true, false, false);
MOCHI_CONTACT_BENCHMARK("All/NonUnitSdfGradient", true, true, true, false, false, false);
MOCHI_CONTACT_BENCHMARK("All/ExplicitNormals", true, true, true, true, true, false);
MOCHI_CONTACT_BENCHMARK(
    "All/NonUnitSdfGradient/ExplicitNormals",
    true,
    true,
    true,
    false,
    true,
    false);
MOCHI_CONTACT_BENCHMARK(
    "All/NonUnitSdfGradient/ExplicitNormals/FullDissipation",
    true,
    true,
    true,
    false,
    true,
    true);

#undef MOCHI_CONTACT_BENCHMARK

} // namespace mochi_benchmark
