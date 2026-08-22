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

#include <mochi_core/element_operations/fem_stress_damping.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/simplex_quadrature.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/materials/batched_linear_elastic.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/materials/reference_material_stiffness.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <functional>

using namespace mochi;
using namespace mochi::materials;

namespace mochi_benchmark {

// Mirrors stress_work_benchmark.cpp (same mesh, element count, batch size, mode axis and
// "elements/s" counter) so StressDampingWork is directly comparable to the elastic StressWork.
static constexpr int kBatchSize = kDefaultFemBatchSize;
static constexpr int kNumElements = 1024;
static constexpr int kNumQuadPoints = 1; // Default for soft actors

using ElementT = tetrahedral::Pk3DElement<1, kNumQuadPoints>;
static constexpr int kDim = ElementT::kSpaceDim * ElementT::kNumDofs;

enum class BenchmarkMode { Objective, Residual, DresidualWithoutPsd, DresidualWithPsd, All };

template <typename BatchedMaterialFn>
static void BenchmarkStressDampingWork(
    benchmark::State& state,
    BatchedMaterialFn batchedMaterialFn,
    BenchmarkMode mode,
    bool includeGeometricStiffness,
    bool materialIsIsotropic) {
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

  // Current and stage-start displacements use distinct seeds so ΔE = E(F) − E(F_ss) ≠ 0 and the
  // residual/tangent paths do real work.
  NdArray<NdArray<real, kDim>, kBatchSize> displacements{};
  NdArray<NdArray<real, kDim>, kBatchSize> stageStartDisplacements{};
  NdArray<int, kBatchSize> elementIndices{};
  auto gen = RandomGenerator(42);
  auto stageStartGen = RandomGenerator(43);
  for (int b = 0; b < kBatchSize; ++b) {
    SetRandom(gen, -0.3_r, 0.3_r, displacements[b]);
    SetRandom(stageStartGen, -0.3_r, 0.3_r, stageStartDisplacements[b]);
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
  NdArray<V, kDim> batchedStageStartDisp{};
  for (int d = 0; d < kDim; ++d) {
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = displacements[b][d];
    }
    batchedDisp[d] = Load<V>(staging);
    for (int b = 0; b < kBatchSize; ++b) {
      staging[b] = stageStartDisplacements[b][d];
    }
    batchedStageStartDisp[d] = Load<V>(staging);
  }

  // Reference (zero-deformation) stiffness C₀, precomputed once from the material response (as done
  // at material-set time in production) and reused every iteration.
  NdArray<V, 6, 6> const referenceMaterialStiffnessVoigt =
      ComputeReferenceMaterialStiffnessVoigt<kBatchSize>(elementIndices, batchedMaterialFn);

  // Representative κ = β/dtStage. The exact value is perf-neutral: StressDampingWork runs whenever
  // κ > 0, so any positive value exercises the same code path.
  constexpr real kBeta = 1e-3_r; // stiffness-damping coefficient [s]
  constexpr real kDtStage = 1_r / 60_r; // stage step [s]
  constexpr real kStiffnessDampingFactor = kBeta / kDtStage;

  BatchDouble<kBatchSize> energy{0.0};
  NdArray<V, kDim> res{};
  NdArray<V, kDim * kDim> dres{};
  std::function<void()> fn = [&]() {
    energy = 0.0;
    res = {};
    dres = {};
    fem::StressDampingWork<kBatchSize>(
        elementIndices,
        MakeConstSpan(elements),
        batchedDisp,
        batchedStageStartDisp,
        evalObj ? &energy : nullptr,
        evalRes ? &res : nullptr,
        evalDRes ? &dres : nullptr,
        projectPsd,
        includeGeometricStiffness,
        kStiffnessDampingFactor,
        referenceMaterialStiffnessVoigt,
        materialIsIsotropic);
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

static void LinearElastic_StressDampingWork(
    benchmark::State& state,
    BenchmarkMode mode,
    bool includeGeometricStiffness) {
  auto const params = materials::BuildPerElementParams(LinearElasticMaterialParams{});
  BenchmarkStressDampingWork(
      state,
      materials::MakeBatchedConstitutiveResponse<LinearElasticMaterialParams, kBatchSize>(params),
      mode,
      includeGeometricStiffness,
      materials::kIsotropicReferenceStiffness<LinearElasticMaterialParams>);
}

static void SmithNeoHookean_StressDampingWork(
    benchmark::State& state,
    BenchmarkMode mode,
    bool includeGeometricStiffness) {
  auto const params = materials::BuildPerElementParams(SmithNeoHookeanMaterialParams{});
  BenchmarkStressDampingWork(
      state,
      materials::MakeBatchedConstitutiveResponse<SmithNeoHookeanMaterialParams, kBatchSize>(
          params, /* Ensure projectPsd is respected */ MaterialPsdOracle::None),
      mode,
      includeGeometricStiffness,
      materials::kIsotropicReferenceStiffness<SmithNeoHookeanMaterialParams>);
}

// clang-format off
// The non-dresidual modes are tangent-free, so the geometric flag is irrelevant there; they run
// with the geometric term on (the exact tangent). The PSD dresidual mode additionally gets an
// explicit geometric-off variant so the geometric-block cost (dominated by the skipped PSD
// projection) is directly measurable as the on/off delta.
//
// Both benchmarked materials are isotropic, so they exercise the production isotropic C₀ᵥ path
// (BenchmarkStressDampingWork passes kIsotropicReferenceStiffness<ParamsT>). The isotropic speedup
// is measured by comparing this benchmark across commits, not by an in-benchmark dense/iso axis.
#define MOCHI_STRESS_DAMPING_WORK_BENCHMARK_MODES(Name)                                                                                    \
  BENCHMARK_CAPTURE(Name##_StressDampingWork, Name##_Objective,               BenchmarkMode::Objective,           /*geo*/ true);           \
  BENCHMARK_CAPTURE(Name##_StressDampingWork, Name##_Residual,                BenchmarkMode::Residual,            /*geo*/ true);           \
  BENCHMARK_CAPTURE(Name##_StressDampingWork, Name##_DresidualWithoutPsd_Geo, BenchmarkMode::DresidualWithoutPsd, /*geo*/ true);           \
  BENCHMARK_CAPTURE(Name##_StressDampingWork, Name##_DresidualWithPsd_Geo,    BenchmarkMode::DresidualWithPsd,    /*geo*/ true);           \
  BENCHMARK_CAPTURE(Name##_StressDampingWork, Name##_DresidualWithPsd_NoGeo,  BenchmarkMode::DresidualWithPsd,    /*geo*/ false);          \
  BENCHMARK_CAPTURE(Name##_StressDampingWork, Name##_All_Geo,                 BenchmarkMode::All,                 /*geo*/ true);           \
  BENCHMARK_CAPTURE(Name##_StressDampingWork, Name##_All_NoGeo,               BenchmarkMode::All,                 /*geo*/ false);
// clang-format on

MOCHI_STRESS_DAMPING_WORK_BENCHMARK_MODES(LinearElastic)
MOCHI_STRESS_DAMPING_WORK_BENCHMARK_MODES(SmithNeoHookean)

} // namespace mochi_benchmark
