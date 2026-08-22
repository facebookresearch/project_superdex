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
#include <mochi_core/element_operations/fem_shell.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_deformable.h>
#include <mochi_physics/src/mochi_shell.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::shell;

namespace {

// Element operations to toggle. Inertia and stress (membrane + bending) always run in the full
// assembly; only gravity and PSD projection are toggleable. An elopFlags value of 0 selects the
// assembler-overhead path (a trivial no-op element operator).
enum ShellElOpFlags {
  kShellElOpGravity = 1 << 0, // fem::GravityWork
  kShellElOpPsdProjection = 1 << 1, // PSD projection of the stress dresidual
};

// Results to compute for each element
enum ResultFlags {
  kResultObj = 1 << 1,
  kResultRes = 1 << 2,
  kResultDRes = 1 << 3,
  kResultObjResDRes = kResultObj | kResultRes | kResultDRes,
};

} // namespace

// This function is used to measure shell element operations on triangular meshes.
// If a mesh file is specified, then the ElementAssembler will be used to assemble over the given
// mesh. Otherwise, a programmatically-generated uniform mesh of a unit square will be used.
static void ShellElementAssemblerBenchmark(
    benchmark::State& state,
    std::string const& meshFilePath,
    int meshSize, // For generated square meshes: meshSize x meshSize cells
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

  // Determine what type of mesh to create
  bool const isGeneratedMesh = meshFilePath.empty() && meshSize > 0;

  // Must have a nonzero mesh size if generating a uniform mesh.
  MOCHI_ASSERT(!meshFilePath.empty() || meshSize > 0, "Must specify a mesh file or size");

  // Load or create a triangular mesh
  std::unique_ptr<TriangularMesh> triMesh;
  if (isGeneratedMesh) {
    // Generate a uniform square triangular mesh
    auto [coordinates, connectivity] =
        UniformSquareTriangularMeshData(Int2{meshSize, meshSize}, Real2{1_r, 1_r}, 2);
    triMesh = std::make_unique<TriangularMesh>(MakeSpan(coordinates), MakeSpan(connectivity));
  } else {
    // Load mesh from file
    // Create a MochiPhysics instance just so it can load the mesh for us.
    Context* mochiContext = mochi::CreateContext(numWorkerThreads);
    auto shape =
        mochiContext->LoadShapeFromFile(mochi_benchmark::GetAssetPath(meshFilePath), ErrorAssert{});
    auto meshData = mochiContext->GetShapeMesh(shape, ErrorAssert{});
    auto coordinates = Unflatten<Real3 const>(meshData.coordinates);
    auto connectivity = Unflatten<Int3 const>(meshData.connectivity);
    triMesh = std::make_unique<TriangularMesh>(coordinates, connectivity);
    mochi::DestroyContext(mochiContext);
  }

  // FEM Discretizations for triangular elements
  using SurfaceElementLow = triangular::Pk2DElement<1, 1>;
  using SurfaceElementHigh = triangular::Pk2DElement<1, 3>; // Higher order quadrature
  int const numElements = triMesh->GetNumElements();
  std::vector<SurfaceElementLow> femLowSurfDisc;
  std::vector<SurfaceElementHigh> femHighSurfDisc;
  femLowSurfDisc.reserve(numElements);
  femHighSurfDisc.reserve(numElements);
  for (int i = 0; i < numElements; ++i) {
    femLowSurfDisc.emplace_back(
        i, triMesh->GetNodeCoordinates(), triMesh->GetElementConnectivity());
    femHighSurfDisc.emplace_back(
        i, triMesh->GetNodeCoordinates(), triMesh->GetElementConnectivity());
  }

  // Nodal based structure. The FEM assembler requires a uniform kNumStencilNodes-per-element
  // NBS, so pad each element's node list to the 6-node bending stencil (boundary triangles have
  // fewer), reusing the original N-to-N graph for the correct dresidual sparsity (mirrors
  // mochi_shell_init).
  auto bendingConnectivityAndStencil = triMesh->GenerateBendingConnectivityAndStencil();
  auto const& bendingEToN = bendingConnectivityAndStencil.first;
  auto const& bendingStencil = bendingConnectivityAndStencil.second;
  auto nbs = BuildPaddedNodalBasedStructure<fem::kBendingStencilNodes>(bendingEToN, bendingStencil);

  // Local2GlobalMap for triangular elements
  Local2GlobalMap l2g;
  l2g.InitializeFromElementNodeConnectivity(bendingEToN, 3);
  l2g.InitializeStencilIndices(bendingStencil);
  // The FEM assembler runs over the uniform 6-node bending stencil (stride 18).
  l2g.InitializePaddedIndices(fem::kBendingStencilDofs);

  // SNLE data
  int numGlobalDofs = l2g.GetGlobalRange().Size();

  // Affine displacement so that PSD benchmarks do not run at the zero-strain rest state.
  ColumnVector<real> sol(numGlobalDofs);
  auto const nodes = triMesh->GetNodeCoordinates();
  MOCHI_ASSERT(sol.Rows() == 3 * isize(nodes), "Unexpected number of DoFs.");
  for (int n = 0; n < isize(nodes); ++n) {
    Real3 const& x = nodes[n];
    sol[3 * n + 0] = -0.08_r * x[0] + 0.03_r * x[1] + 0.01_r * x[2];
    sol[3 * n + 1] = 0.02_r * x[0] - 0.06_r * x[1] + 0.02_r * x[2];
    sol[3 * n + 2] = 0.04_r * x[0] - 0.03_r * x[1] + 0.02_r * x[2];
  }

  double merit = 0.0;
  auto res = ColumnVector<real>::Zero(numGlobalDofs);
  auto sparsityGraph = MakeSparsityGraph(l2g, numGlobalDofs);
  int const numRows = isize(sparsityGraph.GetPointers()) - 1;
  int const numCols = numRows; // Symmetrical
  DynamicArray<real> values(sparsityGraph.NumTargets(), 0_r);
  auto actorDResFullSparse = SparseMatrixView<real const>(
      numCols, sparsityGraph.GetPointers(), sparsityGraph.GetTargets(), MakeSpan(values));
  auto blockStructure = BlockedStructure<3>(actorDResFullSparse);
  BlockSparseMatrix<real, 3> dres(
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

  // Other stuff element operations might need (arbitrary values)
  ColumnVector<real> predGlobalDisp(numGlobalDofs);
  ColumnVector<real> predGlobalVelo(numGlobalDofs);
  predGlobalDisp.SetRandom(111, -1_r, 1_r);
  predGlobalVelo.SetRandom(222, -1_r, 1_r);

  // Material / physical constants (arbitrary values).
  constexpr int kBatchSize = kDefaultFemBatchSize;
  constexpr real kShellThickness = 0.001_r; // typical shell thickness
  constexpr real kShellDensity = 1000_r * kShellThickness; // surface density
  constexpr real kDt = 0.1_r; // Arbitrary
  constexpr real kOneOverDt2 = 1_r / Sqr(kDt);

  // Shell elasticity material (arbitrary values).
  constexpr real kBendingAlpha = 1e-6_r;
  constexpr real kBendingBeta = 1e-6_r;
  constexpr real kYoungsModulus = 1e3_r;
  constexpr real kPoissonRatio = 0.3_r;
  real const kMembraneLambda =
      (kYoungsModulus * kPoissonRatio) / ((1_r + kPoissonRatio) * (1_r - 2_r * kPoissonRatio));
  real const kMembraneMu = kYoungsModulus / (2_r * (1_r + kPoissonRatio));

  bool const hasGravity = elopFlags & kShellElOpGravity;
  bool const psdProject = elopFlags & kShellElOpPsdProjection;

  ElOpFnType<ShellStencilElement> batchedOp;
  if (elopFlags == 0) {
    batchedOp = [](NdArray<int, kBatchSize> const&,
                   Span<int const>,
                   fem::BatchElementVector<kBatchSize, ShellStencilElement> const&,
                   BatchDouble<kBatchSize>*,
                   fem::BatchElementVector<kBatchSize, ShellStencilElement>*,
                   fem::BatchElementMatrix<kBatchSize, ShellStencilElement>*,
                   bool) -> bool { return true; };
  } else {
    batchedOp = shell::MakeBatchedBodyOp(
        l2g,
        MakeConstSpan(femLowSurfDisc),
        MakeConstSpan(femHighSurfDisc),
        hasGravity,
        kMembraneLambda,
        kMembraneMu,
        kBendingAlpha,
        kBendingBeta,
        kDefaultGravity,
        kShellDensity,
        kOneOverDt2,
        predGlobalDisp.GetConstSpan(),
        predGlobalVelo.GetConstSpan(),
        kDt,
        0_r,
        0_r);
  }

  // Assembly
  // Create a task scheduler and bind it to this thread
  TaskScheduler scheduler(numWorkerThreads);

  AssemblyParams params{
      .assemObj = assemObj, .assemRes = assemRes, .assemDRes = assemDRes, .psdDRes = psdProject};

  auto assembleOnce = [&]() {
    // Accumulate merit, residual and dresidual across benchmark iterations to exclude clearing
    // overhead.
    AssembleObjResDRes<ShellStencilElement, ShellStencilElement::kSpaceDim, kBatchSize>(
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
// Shell Element Assembler Benchmark Cases
// ---------------------------------------------------------------------------

// Defines benchmarks with various numbers of threads
// clang-format off
#define MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER_FOR_THREADS(name, benchmarkName, file, meshSize, elopFlags, resultFlags) \
  BENCHMARK_CAPTURE(ShellElementAssemblerBenchmark, name##_1, file, meshSize, elopFlags, resultFlags, 1)                 \
      ->Name("ShellAssembler/" benchmarkName "/Threads1");                                                               \
  BENCHMARK_CAPTURE(ShellElementAssemblerBenchmark, name##_2, file, meshSize, elopFlags, resultFlags, 2)                 \
      ->Name("ShellAssembler/" benchmarkName "/Threads2");                                                               \
  BENCHMARK_CAPTURE(ShellElementAssemblerBenchmark, name##_4, file, meshSize, elopFlags, resultFlags, 4)                 \
      ->Name("ShellAssembler/" benchmarkName "/Threads4");                                                               \
  BENCHMARK_CAPTURE(ShellElementAssemblerBenchmark, name##_8, file, meshSize, elopFlags, resultFlags, 8)                 \
      ->Name("ShellAssembler/" benchmarkName "/Threads8");                                                               \
  BENCHMARK_CAPTURE(ShellElementAssemblerBenchmark, name##_16, file, meshSize, elopFlags, resultFlags, 16)               \
      ->Name("ShellAssembler/" benchmarkName "/Threads16");                                                              \
  BENCHMARK_CAPTURE(ShellElementAssemblerBenchmark, name##_32, file, meshSize, elopFlags, resultFlags, 32)               \
      ->Name("ShellAssembler/" benchmarkName "/Threads32");                                                              \
  BENCHMARK_CAPTURE(ShellElementAssemblerBenchmark, name##_64, file, meshSize, elopFlags, resultFlags, 64)               \
      ->Name("ShellAssembler/" benchmarkName "/Threads64");
// clang-format on

// Defines benchmarks with various element operations for shell. Inertia and stress always run in
// the full assembly; the AssemblerOverhead cases isolate the assembler's gather/scatter cost.
// clang-format off
#define MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER(name, file, meshSize)                                                                                                                                                    \
  MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER_FOR_THREADS(name##_AssemblerOverhead_DRes, #name "/AssemblerOverhead/DResidual", file, meshSize, 0, kResultDRes);                                                              \
  MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER_FOR_THREADS(name##_AssemblerOverhead_All, #name "/AssemblerOverhead/All", file, meshSize, 0, kResultObjResDRes);                                                               \
  MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_Obj, #name "/Full/Objective", file, meshSize, kShellElOpGravity, kResultObj);                                          \
  MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_Res, #name "/Full/Residual", file, meshSize, kShellElOpGravity, kResultRes);                                           \
  MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_Dres_NoPsd, #name "/Full/DResidual/NoPsd", file, meshSize, kShellElOpGravity, kResultDRes);                            \
  MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_Dres_Psd, #name "/Full/DResidual/Psd", file, meshSize, kShellElOpGravity | kShellElOpPsdProjection, kResultDRes);      \
  MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_All_NoPsd, #name "/Full/All/NoPsd", file, meshSize, kShellElOpGravity, kResultObjResDRes);                             \
  MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER_FOR_THREADS(name##_Full_All_Psd, #name "/Full/All/Psd", file, meshSize, kShellElOpGravity | kShellElOpPsdProjection, kResultObjResDRes);
// clang-format on

// Generated square mesh benchmarks (various sizes)
// clang-format off
MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER(Square_8x8, "", 8);        // 128 triangles, 81 nodes
MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER(Square_16x16, "", 16);     // 512 triangles, 289 nodes
MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER(Square_32x32, "", 32);     // 2048 triangles, 1089 nodes
MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER(Square_64x64, "", 64);     // 8192 triangles, 4225 nodes
MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER(Square_128x128, "", 128);  // 32768 triangles, 16641 nodes
// clang-format on

// Asset-based mesh benchmarks
#if MOCHI_INTERNAL // The Duck surface mesh is not shipped externally.
// clang-format off
MOCHI_BENCHMARK_SHELL_ELEMENT_ASSEMBLER(Duck_Surface_13542, "duck/duck_surface_mesh_13542.mochi.h5", 0);
// clang-format on
#endif
