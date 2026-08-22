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

#include <mochi_core/element_operations/fem_inertia.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <functional>

using namespace mochi;

namespace mochi_benchmark {

static constexpr int kBatchSize = kDefaultFemBatchSize;
static constexpr int kNumElements = 1024;
// Soft actors always integrate inertia with 4-point (high) volume quadrature.
static constexpr int kNumQuadPoints = 4;
static constexpr real kDensity = 1000_r;
static constexpr real kDt = 0.01_r;
static constexpr real kInvDtSqr = 1_r / Sqr(kDt);

using ElementT = tetrahedral::Pk3DElement<1, kNumQuadPoints>;
static constexpr int kDim = ElementT::kSpaceDim * ElementT::kNumDofs;

enum class InertiaWorkMode { Objective, Residual, All };

static void BenchmarkInertiaWork(benchmark::State& state, InertiaWorkMode mode) {
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
        tetrahedral::kTetrahedralQuadrature4);
  }

  NdArray<NdArray<real, kDim>, kBatchSize> displacements{};
  NdArray<int, kBatchSize> elementIndices{};
  auto gen = RandomGenerator(123);
  for (int b = 0; b < kBatchSize; ++b) {
    SetRandom(gen, -0.3_r, 0.3_r, displacements[b]);
    elementIndices[b] = b * (kNumElements / kBatchSize);
  }

  NdArray<real, kDim> stageStartDisp{};
  NdArray<real, kDim> stageStartVel{};
  SetRandom(gen, -1_r, 1_r, stageStartDisp);
  SetRandom(gen, -1_r, 1_r, stageStartVel);

  bool const evalObj = (mode == InertiaWorkMode::Objective || mode == InertiaWorkMode::All);
  bool const evalRes = (mode == InertiaWorkMode::Residual || mode == InertiaWorkMode::All);

  alignas(alignof(V)) real staging[V::kSize]{};
  NdArray<V, kDim> batchedDisp{};
  NdArray<V, kDim> batchedStageStartTarget{};
  for (int d = 0; d < kDim; ++d) {
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = displacements[b][d];
    }
    batchedDisp[d] = Load<V>(staging);
    batchedStageStartTarget[d] = V{stageStartDisp[d] + kDt * stageStartVel[d]};
  }

  BatchDouble<kBatchSize> energy{0.0};
  NdArray<V, kDim> res{};
  std::function<void()> fn = [&]() {
    energy = 0.0;
    res = {};
    fem::InertiaWork<kBatchSize>(
        elementIndices,
        MakeConstSpan(elements),
        batchedDisp,
        batchedStageStartTarget,
        evalObj ? &energy : nullptr,
        evalRes ? &res : nullptr,
        kDensity,
        kInvDtSqr);
  };

  for (auto _ : state) {
    CallNoInline(fn);
  }
  state.counters["elements/s"] =
      benchmark::Counter(state.iterations() * kBatchSize, benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(BenchmarkInertiaWork, Inertia_Objective, InertiaWorkMode::Objective);
BENCHMARK_CAPTURE(BenchmarkInertiaWork, Inertia_Residual, InertiaWorkMode::Residual);
BENCHMARK_CAPTURE(BenchmarkInertiaWork, Inertia_All, InertiaWorkMode::All);

} // namespace mochi_benchmark
