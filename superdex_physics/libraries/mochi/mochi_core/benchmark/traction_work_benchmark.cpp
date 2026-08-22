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

#include <mochi_core/element_operations/fem_traction.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/finite_element_trace.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>

#include <functional>

using namespace mochi;

namespace mochi_benchmark {

static constexpr int kBatchSize = kDefaultFemBatchSize;
static constexpr int kNumTraceElements = 1024;
static constexpr int kNumFields = 3;
static constexpr real kTractionStiffness = 10_r;

// Soft actors always build the boundary discretization from the low (P1Q1) volume element.
using VolumeElement = tetrahedral::Pk3DElement<1, 1>;

enum class TractionWorkMode { Objective, Residual, Dresidual, All };

// Synthetic energy, force, and diagonal stiffness.
static auto MakeTractionCallback() {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  return [](NdArray<int, kBatchSize> const& /*elementIndices*/,
            int /*quadPointIndex*/,
            BatchDouble<kBatchSize>* outEnergy,
            BatchReal3<kBatchSize>* outForce,
            NdArray<BatchReal3<kBatchSize>, 3>* outDForce,
            NdArray<bool, kBatchSize>& outHasForce) {
    for (int b = 0; b < kBatchSize; ++b) {
      outHasForce[b] = true;
    }
    if (outEnergy) {
      *outEnergy = Vd{0.5};
    }
    if (outForce) {
      (*outForce)[0] = V{1_r};
      (*outForce)[1] = V{0_r};
      (*outForce)[2] = V{-1_r};
    }
    if (outDForce) {
      V const zero{0_r};
      V const k{kTractionStiffness};
      (*outDForce)[0] = {k, zero, zero};
      (*outDForce)[1] = {zero, k, zero};
      (*outDForce)[2] = {zero, zero, k};
    }
  };
}

template <int kNumQuadPoints, TractionWorkMode kMode>
static void BenchmarkTractionWork(benchmark::State& state) {
  using V = BatchReal<kBatchSize>;
  using TraceElement = tetrahedral::Pk3DElementTrace<VolumeElement, kNumQuadPoints>;
  constexpr int kDim = TraceElement::kNumDofs * kNumFields;

  constexpr Real3 kCoordinates[] = {
      Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}};
  constexpr Int4 kConnectivity[] = {Int4{0, 1, 2, 3}};
  TetrahedralMesh mesh(kCoordinates, kConnectivity);
  VolumeElement volumeElement(
      0,
      mesh.GetNodeCoordinates(),
      mesh.GetElementConnectivity(),
      tetrahedral::kTetrahedralQuadrature1);

  DynamicArray<TraceElement> traces;
  traces.reserve(kNumTraceElements);
  constexpr auto kTraceQuadrature = tetrahedral::details::MakeTetrahedralTraceQuadrature<
      triangular::TriangleQuadrature<kNumQuadPoints>>();
  for (int i = 0; i < kNumTraceElements; ++i) {
    int const face = i % isize(kTraceQuadrature);
    traces.emplace_back(volumeElement, face, kTraceQuadrature[face]);
  }

  NdArray<int, kBatchSize> elementIndices{};
  for (int b = 0; b < kBatchSize; ++b) {
    elementIndices[b] = b * (kNumTraceElements / kBatchSize);
  }

  bool const evalObj = (kMode == TractionWorkMode::Objective || kMode == TractionWorkMode::All);
  bool const evalRes = (kMode == TractionWorkMode::Residual || kMode == TractionWorkMode::All);
  bool const evalDRes = (kMode == TractionWorkMode::Dresidual || kMode == TractionWorkMode::All);

  auto traction = MakeTractionCallback();

  BatchDouble<kBatchSize> energy{0.0};
  NdArray<V, kDim> res{};
  NdArray<V, kDim * kDim> dres{};
  std::function<void()> fn = [&]() {
    energy = 0.0;
    res = {};
    dres = {};
    fem::TractionWork<kBatchSize, TraceElement, kNumFields>(
        elementIndices,
        MakeConstSpan(traces),
        evalObj ? &energy : nullptr,
        evalRes ? &res : nullptr,
        evalDRes ? &dres : nullptr,
        traction);
  };

  for (auto _ : state) {
    CallNoInline(fn);
  }
  state.counters["elements/s"] =
      benchmark::Counter(state.iterations() * kBatchSize, benchmark::Counter::kIsRate);
}

// Sweep the boundary quadrature configs: P1Q1, P1Q3, and P1Q6. Experimental P1Q7/Q12/Q16 are
// intentionally omitted.
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 1, TractionWorkMode::Objective)
    ->Name("TractionWork/P1Q1/Objective");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 1, TractionWorkMode::Residual)
    ->Name("TractionWork/P1Q1/Residual");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 1, TractionWorkMode::Dresidual)
    ->Name("TractionWork/P1Q1/DResidual");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 1, TractionWorkMode::All)->Name("TractionWork/P1Q1/All");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 3, TractionWorkMode::Objective)
    ->Name("TractionWork/P1Q3/Objective");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 3, TractionWorkMode::Residual)
    ->Name("TractionWork/P1Q3/Residual");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 3, TractionWorkMode::Dresidual)
    ->Name("TractionWork/P1Q3/DResidual");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 3, TractionWorkMode::All)->Name("TractionWork/P1Q3/All");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 6, TractionWorkMode::Objective)
    ->Name("TractionWork/P1Q6/Objective");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 6, TractionWorkMode::Residual)
    ->Name("TractionWork/P1Q6/Residual");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 6, TractionWorkMode::Dresidual)
    ->Name("TractionWork/P1Q6/DResidual");
BENCHMARK_TEMPLATE2(BenchmarkTractionWork, 6, TractionWorkMode::All)->Name("TractionWork/P1Q6/All");

} // namespace mochi_benchmark
