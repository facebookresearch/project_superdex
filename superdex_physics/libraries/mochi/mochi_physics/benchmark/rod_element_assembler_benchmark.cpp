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

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_physics/src/mochi_deformable.h>
#include <mochi_physics/src/mochi_rod.h>

#include <utility>

using namespace mochi;

namespace {

// Element operations to toggle. Inertia and stress (axial + bend-twist) always run in the full
// assembly; only gravity and PSD projection are toggleable. An elopFlags value of 0 selects the
// assembler-overhead path (a trivial no-op element operator).
enum RodElOpFlags {
  kRodElOpGravity = 1 << 0, // fem::RodGravity
  kRodElOpPsdProjection = 1 << 1, // PSD projection of the stress dresidual
};

// Results to compute for each element
enum ResultFlags {
  kResultObj = 1 << 1,
  kResultRes = 1 << 2,
  kResultDRes = 1 << 3,
  kResultObjResDRes = kResultObj | kResultRes | kResultDRes,
};

constexpr Real3 kGravity{0_r, -9.8_r, 0_r};

} // namespace

// This function measures rod element operations on a uniform polyline mesh of the unit interval.
static void RodElementAssemblerBenchmark(
    benchmark::State& state,
    int numElements,
    int elopFlags,
    int resultFlags,
    int numThreads) {
  // If the requested number of threads exceeds the capabilities of the hardware, then skip this
  // benchmark.
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    MOCHI_SKIP_BENCHMARK(state);
    return;
  }

  // Use this thread and (numThreads - 1) additional worker threads.
  int const numWorkerThreads = numThreads - 1;
  int const numNodes = numElements + 1;
  constexpr int kNumDofsPerNode = fem::kNumRodFields; // 4 (3 displacement + 1 twist)

  // Generate a uniform polyline mesh on the unit interval [0, 1] along the x-axis
  DynamicArray<Real3> meshNodes(numNodes);
  for (int i = 0; i < numNodes; ++i) {
    real const t = static_cast<real>(i) / static_cast<real>(numElements);
    meshNodes[i] = Real3{t, 0_r, 0_r};
  }

  // Generate connectivity and stencil
  auto connectivityAndStencil = GenerateRodConnectivityAndStencil(numNodes, /*isClosedLoop=*/false);

  // Nodal based structure for parallel assembly. The FEM assembler requires a uniform
  // kNumRodStencilNodes-per-element NBS, so pad each element's node list (boundary elements have
  // fewer), reusing the original N-to-N graph for the correct dresidual sparsity (mirrors
  // mochi_rod).
  auto const& rodConnectivity = connectivityAndStencil.first;
  auto const& rodStencil = connectivityAndStencil.second;
  auto nbs = BuildPaddedNodalBasedStructure<fem::kNumRodStencilNodes>(rodConnectivity, rodStencil);

  // Local-to-global map
  Local2GlobalMap l2g;
  l2g.InitializeFromElementNodeConnectivity(connectivityAndStencil.first, kNumDofsPerNode);
  l2g.InitializeStencilIndices(connectivityAndStencil.second);
  // The FEM assembler runs over the uniform rod stencil.
  l2g.InitializePaddedIndices(fem::kNumRodStencilNodes * kNumDofsPerNode);

  // SNLE data
  int const numGlobalDofs = l2g.GetGlobalRange().Size();

  // Use axial compression so axial stress/PSD paths run away from rest while the straight-rod frame
  // axes remain orthogonal to the deformed tangent.
  ColumnVector<real> sol(numGlobalDofs);
  MOCHI_ASSERT(sol.Rows() == kNumDofsPerNode * numNodes, "Unexpected number of DoFs.");
  for (int n = 0; n < numNodes; ++n) {
    sol[kNumDofsPerNode * n + 0] = -0.10_r * meshNodes[n][0];
    sol[kNumDofsPerNode * n + 1] = 0_r;
    sol[kNumDofsPerNode * n + 2] = 0_r;
    sol[kNumDofsPerNode * n + 3] = 0_r;
  }

  double merit = 0.0;
  auto res = ColumnVector<real>::Zero(numGlobalDofs);
  auto const sparsityGraph = MakeSparsityGraph(l2g, numGlobalDofs);
  int const numCols = isize(sparsityGraph.GetPointers()) - 1;
  DynamicArray<real> values(sparsityGraph.NumTargets(), 0_r);
  auto blockStructure = BlockedStructure<kNumDofsPerNode>(SparseMatrixView<real const>(
      numCols, sparsityGraph.GetPointers(), sparsityGraph.GetTargets(), MakeSpan(values)));
  BlockSparseMatrix<real, kNumDofsPerNode> dres(
      blockStructure.nBlockCols,
      std::move(blockStructure.ptr),
      std::move(blockStructure.ndIndices),
      std::move(values));
  MOCHI_ASSERT(res.Rows() == sol.Rows());
  MOCHI_ASSERT(dres.Rows() == sol.Rows());

  // Optionally compute any combination of these results
  bool const assemObj = resultFlags & kResultObj;
  bool const assemRes = resultFlags & kResultRes;
  bool const assemDRes = resultFlags & kResultDRes;

  // Lumped nodal masses and element rotational inertias (same logic as InitRodActor)
  constexpr real kLinearDensity = 1_r;
  constexpr real kLinearRotationalInertia = 0.01_r;
  DynamicArray<real> nodalMasses(numNodes, 0_r);
  DynamicArray<real> elementRotationalInertias(numElements);
  for (int i = 0; i < numElements; ++i) {
    real const elementLength = Norm(meshNodes[i + 1] - meshNodes[i]);
    real const halfMass = 0.5_r * kLinearDensity * elementLength;
    nodalMasses[i] += halfMass;
    nodalMasses[i + 1] += halfMass;
    elementRotationalInertias[i] = kLinearRotationalInertia * elementLength;
  }

  // Predicted displacements and velocities for inertia (arbitrary small values)
  ColumnVector<real> predDisp(numGlobalDofs);
  ColumnVector<real> predVelo(numGlobalDofs);
  predDisp.SetRandom(111, -0.01_r, 0.01_r);
  predVelo.SetRandom(222, -0.01_r, 0.01_r);

  // Frame axes: for a straight rod along the x-axis, all element frame axes are (0, 1, 0)
  DynamicArray<Real3> referenceAxes(numElements, Real3{0_r, 1_r, 0_r});
  DynamicArray<Real3> currentAxes(numElements, Real3{0_r, 1_r, 0_r});

  // Material parameters (arbitrary but reasonable values)
  constexpr real kAxialStiffness = 1e3_r;
  constexpr real kTorsionalStiffness = 1e1_r;
  constexpr Real2 kFlexuralStiffness{1e1_r, 1e1_r};

  constexpr real kDt = 0.01_r; // Arbitrary

  // Element operation toggles.
  bool const hasGravity = elopFlags & kRodElOpGravity;
  bool const psdProject = elopFlags & kRodElOpPsdProjection;

  constexpr int kBatchSize = kDefaultFemBatchSize;

  ElOpFnType<fem::RodStencilElement, fem::kNumRodFields> batchedOp;
  if (elopFlags == 0) {
    batchedOp = [](NdArray<int, kBatchSize> const&,
                   Span<int const>,
                   fem::BatchRodVector<kBatchSize> const&,
                   BatchDouble<kBatchSize>*,
                   fem::BatchRodVector<kBatchSize>*,
                   fem::BatchRodMatrix<kBatchSize>*,
                   bool) -> bool { return true; };
  } else {
    experimental::RodMaterialParams materialParams;
    materialParams.axialStiffness = kAxialStiffness;
    materialParams.flexuralStiffness = kFlexuralStiffness;
    materialParams.torsionalStiffness = kTorsionalStiffness;

    batchedOp = rod::MakeBatchedBodyOp(
        MakeConstSpan(meshNodes),
        hasGravity,
        MakeConstSpan(nodalMasses),
        MakeConstSpan(elementRotationalInertias),
        MakeConstSpan(currentAxes),
        MakeConstSpan(referenceAxes),
        materialParams,
        kGravity,
        predDisp.GetConstSpan(),
        predVelo.GetConstSpan(),
        MakeConstSpan(currentAxes),
        kDt);
  }

  // Assembly
  TaskScheduler scheduler(numWorkerThreads);

  AssemblyParams params{
      .assemObj = assemObj, .assemRes = assemRes, .assemDRes = assemDRes, .psdDRes = psdProject};

  auto assembleOnce = [&]() {
    // Accumulate merit, residual and dresidual across benchmark iterations to exclude clearing
    // overhead.
    AssembleObjResDRes<fem::RodStencilElement, fem::kNumRodFields, kBatchSize>(
        l2g,
        nbs,
        batchedOp,
        AsConstView(sol),
        AssemblyResults<real>{
            .outObj = assemObj ? &merit : nullptr,
            .outRes = assemRes ? AsView(res) : ColumnVectorView<real>{},
            .outDRes = assemDRes ? AnyMatrixView<real>{AsView(dres)} : AnyMatrixView<real>{},
            .params = params});
  };

  if (ProfilerIsConnected()) {
    // Run it exactly once for inspecting in the profiling tool
    assembleOnce();
    for (auto _ : state) {
    }
  } else {
    // Let the benchmark library decide how many times to run it
    for (auto _ : state) {
      assembleOnce();
    }
  }

  // Make sure the optimizer keeps all the calculations
  auto resNorm = res.Norm();
  auto dresNorm = dres.Norm();
  benchmark::DoNotOptimize(merit);
  benchmark::DoNotOptimize(resNorm);
  benchmark::DoNotOptimize(dresNorm);
}

// ---------------------------------------------------------------------------
// Rod Element Assembler Benchmark Cases
// ---------------------------------------------------------------------------

// Defines benchmarks with various numbers of threads
// clang-format off
#define MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER_FOR_THREADS(name, benchmarkName, numElements, elopFlags, resultFlags) \
  BENCHMARK_CAPTURE(RodElementAssemblerBenchmark, name##_1, numElements, elopFlags, resultFlags, 1)                 \
      ->Name("RodAssembler/" benchmarkName "/Threads1");                                                            \
  BENCHMARK_CAPTURE(RodElementAssemblerBenchmark, name##_2, numElements, elopFlags, resultFlags, 2)                 \
      ->Name("RodAssembler/" benchmarkName "/Threads2");                                                            \
  BENCHMARK_CAPTURE(RodElementAssemblerBenchmark, name##_4, numElements, elopFlags, resultFlags, 4)                 \
      ->Name("RodAssembler/" benchmarkName "/Threads4");                                                            \
  BENCHMARK_CAPTURE(RodElementAssemblerBenchmark, name##_8, numElements, elopFlags, resultFlags, 8)                 \
      ->Name("RodAssembler/" benchmarkName "/Threads8");                                                            \
  BENCHMARK_CAPTURE(RodElementAssemblerBenchmark, name##_16, numElements, elopFlags, resultFlags, 16)               \
      ->Name("RodAssembler/" benchmarkName "/Threads16");                                                           \
  BENCHMARK_CAPTURE(RodElementAssemblerBenchmark, name##_32, numElements, elopFlags, resultFlags, 32)               \
      ->Name("RodAssembler/" benchmarkName "/Threads32");                                                           \
  BENCHMARK_CAPTURE(RodElementAssemblerBenchmark, name##_64, numElements, elopFlags, resultFlags, 64)               \
      ->Name("RodAssembler/" benchmarkName "/Threads64");
// clang-format on

// Defines benchmarks with various element operations for rods. Inertia and stress always run in
// the full assembly; the AssemblerOverhead cases isolate the assembler's gather/scatter cost.
// clang-format off
#define MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER(name, numElements)                                                                                                                                                          \
  MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER_FOR_THREADS(name##_AssemblerOverhead_DRes, #name "/AssemblerOverhead/DResidual", numElements, 0, kResultDRes);                                                                    \
  MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER_FOR_THREADS(name##_AssemblerOverhead_All, #name "/AssemblerOverhead/All", numElements, 0, kResultObjResDRes);                                                                     \
  MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_Obj, #name "/Full/Objective", numElements, kRodElOpGravity, kResultObj);                                                \
  MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_Res, #name "/Full/Residual", numElements, kRodElOpGravity, kResultRes);                                                 \
  MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_Dres_NoPsd, #name "/Full/DResidual/NoPsd", numElements, kRodElOpGravity, kResultDRes);                                  \
  MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_Dres_Psd, #name "/Full/DResidual/Psd", numElements, kRodElOpGravity | kRodElOpPsdProjection, kResultDRes);              \
  MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_All_NoPsd, #name "/Full/All/NoPsd", numElements, kRodElOpGravity, kResultObjResDRes);                                   \
  MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_All_Psd, #name "/Full/All/Psd", numElements, kRodElOpGravity | kRodElOpPsdProjection, kResultObjResDRes);
// clang-format on

// Unit interval polyline meshes with various numbers of elements
// clang-format off
MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER(Interval_16, 16);      // 17 nodes, 68 DoFs
MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER(Interval_64, 64);      // 65 nodes, 260 DoFs
MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER(Interval_256, 256);    // 257 nodes, 1028 DoFs
MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER(Interval_1024, 1024);  // 1025 nodes, 4100 DoFs
MOCHI_BENCHMARK_ROD_ELEMENT_ASSEMBLER(Interval_4096, 4096);  // 4097 nodes, 16388 DoFs
// clang-format on
