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
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/materials/material_params.h>
#include <mochi_core/materials/reference_material_stiffness.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_soft.h>

#include <memory>
#include <string>
#include <vector>

using namespace mochi;

// The Armadillo and Duck meshes used below are not shipped externally.
#if MOCHI_INTERNAL
namespace {

// Element operations to perform (any bitwise combination)
enum ElOpFlags {
  // Volume operations
  kElOpGravity = 1 << 0, // Gravity volume operation
  kElOpInertia = 1 << 1, // Inertia volume operation
  kElOpStress = 1 << 2, // Stress volume operation
  kElOpMaterialNone = 1 << 3, // No-op stress material for assembler overhead benchmarks
  kElOpMaterialNeoHookean = 1 << 4, // NeoHookean material for the stress operation
  kElOpPsdNone = 1 << 6, // PSDStrategy::None for the stress operation
  kElOpPsdFast = 1 << 7, // PSDStrategy::Fast for the stress operation
  kElOpPsdProjection = 1 << 8, // PSDStrategy::Projection for the stress operation. Also a good
                               // proxy for performance with PSDStrategy::AbsEigenProjection
  kElOpStiffnessDamping = 1 << 9, // Stiffness (viscous) damping. Requires kElOpStress and a real
                                  // material (kElOpMaterialNeoHookean).
  kElOpStiffnessDampingGeometric = 1 << 10, // Include the geometric term in the
                                            // stiffness-damping tangent. Requires
                                            // kElOpStiffnessDamping.
  kElOpStressDefault =
      kElOpStress | kElOpMaterialNeoHookean | kElOpPsdProjection, // Typical combination
  kElOpVolumeDefault = kElOpGravity | kElOpInertia | kElOpStressDefault, // Typical combination
};

// Results to compute for each element
enum ResultFlags {
  kResultObj = 1 << 1,
  kResultRes = 1 << 2,
  kResultDRes = 1 << 3,
  kResultObjResDRes = kResultObj | kResultRes | kResultDRes,
};

} // namespace

// This function measures the FEM mesh assembler path for the specified mesh asset.
static void ElementAssemblerBenchmark(
    benchmark::State& state,
    std::string const& meshFilePath,
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

  // Load the volume mesh. Create a MochiPhysics instance just so it can load the mesh for us.
  Context* mochiContext = mochi::CreateContext(numWorkerThreads);
  auto shape =
      mochiContext->LoadShapeFromFile(mochi_benchmark::GetAssetPath(meshFilePath), ErrorAssert{});
  auto const& mesh = mochiContext->GetShapeMesh(shape, ErrorAssert{});
  auto coordinates = Unflatten<Real3 const>(mesh.coordinates);
  auto connectivity = Unflatten<Int4 const>(mesh.connectivity);
  std::unique_ptr<TetrahedralMesh> tetMesh =
      std::make_unique<TetrahedralMesh>(coordinates, connectivity);
  mochi::DestroyContext(mochiContext);

  // FEM Discretizations
  using VolumeElementLow = tetrahedral::Pk3DElement<1, 1>;
  using VolumeElementHigh = tetrahedral::Pk3DElement<1, 4>;
  int const numElements = tetMesh->GetNumElements();
  std::vector<VolumeElementLow> femLowVolDisc;
  std::vector<VolumeElementHigh> femHighVolDisc;
  femLowVolDisc.reserve(numElements);
  femHighVolDisc.reserve(numElements);
  for (int i = 0; i < numElements; ++i) {
    femLowVolDisc.emplace_back(
        i,
        tetMesh->GetNodeCoordinates(),
        tetMesh->GetElementConnectivity(),
        tetrahedral::kTetrahedralQuadrature1);
    femHighVolDisc.emplace_back(
        i,
        tetMesh->GetNodeCoordinates(),
        tetMesh->GetElementConnectivity(),
        tetrahedral::kTetrahedralQuadrature4);
  }

  // Nodal based structure.
  NodalBasedStructure nbs(tetMesh->GetElementConnectivity());

  // Local2GlobalMap
  Local2GlobalMap l2g;
  tetrahedral::BarycentricBasisTetrahedra<1> basis;
  l2g.InitializeFromMeshAndBasis(tetMesh.get(), basis, 3);

  // SNLE data
  int numGlobalDofs = l2g.GetGlobalRange().Size();

  // Affine displacement so that PSD benchmarks do not run at the identity-deformation special
  // cases.
  ColumnVector<real> sol(numGlobalDofs);
  auto const nodes = tetMesh->GetNodeCoordinates();
  MOCHI_ASSERT(sol.Rows() == 3 * isize(nodes), "Unexpected number of DoFs.");
  for (int n = 0; n < isize(nodes); ++n) {
    Real3 const& x = nodes[n];
    sol[3 * n + 0] = -0.24_r * x[0] + 0.07_r * x[1] + 0.02_r * x[2];
    sol[3 * n + 1] = 0.03_r * x[0] - 0.15_r * x[1] + 0.04_r * x[2];
    sol[3 * n + 2] = -0.02_r * x[0] + 0.05_r * x[1] + 0.12_r * x[2];
  }

  double merit = 0.0;
  auto res = ColumnVector<real>::Zero(numGlobalDofs);
  auto dres = ToBlockSparseMatrix<3>(MakeSparseMatrix(l2g));
  dres.SetZero();
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

  constexpr real kDt = 0.1_r; // Arbitrary

  // Stress (optional): select the material mode and PSD strategy.
  auto psdStrategy = MaterialPsdStrategy::None;
  bool const useNoMaterialStress = (elopFlags & kElOpStress) && (elopFlags & kElOpMaterialNone);
  bool const useNeoHookeanStress =
      (elopFlags & kElOpStress) && (elopFlags & kElOpMaterialNeoHookean);
  if (elopFlags & kElOpStress) {
    if (useNoMaterialStress == useNeoHookeanStress) {
      MOCHI_ASSERT(false, "Must specify exactly one stress material benchmark mode");
    }
    if (elopFlags & kElOpPsdNone) {
      psdStrategy = MaterialPsdStrategy::None;
    } else if (elopFlags & kElOpPsdFast) {
      psdStrategy = MaterialPsdStrategy::Fast;
    } else if (elopFlags & kElOpPsdProjection) {
      psdStrategy = MaterialPsdStrategy::Projection;
    } else {
      MOCHI_ASSERT(false, "Must specify a PSDStrategy for StressWork");
    }
  } else {
    MOCHI_ASSERT(
        (elopFlags &
         (kElOpMaterialNone | kElOpMaterialNeoHookean | kElOpPsdNone | kElOpPsdFast |
          kElOpPsdProjection)) == 0,
        "Material and PSD flags are only valid with kElOpStress.");
  }

  // Stiffness (viscous) damping is gated on stress and a real material (its reference stiffness C₀
  // is derived from the constitutive response), matching the production
  // hasStiffnessDamping = hasStress && factor > 0 gating.
  bool const useStiffnessDamping = (elopFlags & kElOpStiffnessDamping) != 0;
  if (useStiffnessDamping) {
    MOCHI_ASSERT(
        useNeoHookeanStress,
        "kElOpStiffnessDamping requires kElOpStress with kElOpMaterialNeoHookean.");
  }
  bool const useStiffnessDampingGeometric = (elopFlags & kElOpStiffnessDampingGeometric) != 0;
  if (useStiffnessDampingGeometric) {
    MOCHI_ASSERT(
        useStiffnessDamping, "kElOpStiffnessDampingGeometric requires kElOpStiffnessDamping.");
  }

  // Assembly (multi-threaded over the mesh).
  {
    // Create a task scheduler and bind it to this thread
    TaskScheduler scheduler(numWorkerThreads);

    constexpr int kBatchSize = kDefaultFemBatchSize;
    bool const psdProject = (psdStrategy != MaterialPsdStrategy::None);

    // Neo-Hookean constitutive for the real-material stress benchmarks.
    NeoHookeanMaterialParams neoHookeanParams;
    neoHookeanParams.psdStrategy = psdStrategy;
    auto const lame = materials::BuildBatchParams<kBatchSize>(neoHookeanParams);
    auto batchedConstitutive =
        [&](auto const&, auto const& F, auto* e, auto* pk1, auto* t, bool psd) {
          // Force PSD benchmark cases to execute the requested projection path by using psdOracle =
          // None.
          materials::BatchedSmithNeoHookeanConstitutiveResponse<kBatchSize>(
              lame, F, e, pk1, t, psd, materials::MaterialPsdOracle::None);
        };
    auto noOpConstitutive =
        [](auto const&, auto const& /*F*/, auto* /*e*/, auto* /*pk1*/, auto* /*t*/, bool /*psd*/) {
        };

    constexpr real kDensity = 0.5_r; // any value will do

    // Stiffness-damping inputs. When the flag is unset the store stays empty and the factor is 0,
    // exactly reproducing the pre-existing damping-off path. Both are named so they outlive the op
    // returned by MakeBatchedBodyOp (which captures them by reference).
    materials::PerElementReferenceMaterialStiffness referenceStiffness;
    // Representative κ = β/dtStage. The exact value is perf-neutral: StressDampingWork runs
    // whenever κ > 0, so any positive value exercises the same code path.
    constexpr real kStiffnessDampingBeta = 1e-3_r; // stiffness-damping coefficient [s]
    real const stiffnessDampingFactor = useStiffnessDamping ? kStiffnessDampingBeta / kDt : 0_r;
    // Build a homogeneous reference stiffness C₀ from a batch-1 NeoHookean response, mirroring
    // RebuildReferenceMaterialStiffness.
    if (useStiffnessDamping) {
      auto const perElemNeoHookean = materials::BuildPerElementParams(neoHookeanParams);
      auto const referenceResponse =
          materials::MakeBatchedConstitutiveResponse<SmithNeoHookeanMaterialParams, 1>(
              perElemNeoHookean, materials::MaterialPsdOracle::None);
      referenceStiffness = materials::BuildPerElementReferenceMaterialStiffness(
          referenceResponse,
          /*numEntries=*/1,
          materials::kIsotropicReferenceStiffness<SmithNeoHookeanMaterialParams>);
    }

    ElOpFnType<soft::SoftStencilElement> batchedOp;
    if (elopFlags == 0) {
      batchedOp = [](NdArray<int, kBatchSize> const&,
                     Span<int const>,
                     fem::BatchElementVector<kBatchSize, soft::SoftStencilElement> const&,
                     BatchDouble<kBatchSize>*,
                     fem::BatchElementVector<kBatchSize, soft::SoftStencilElement>*,
                     fem::BatchElementMatrix<kBatchSize, soft::SoftStencilElement>*,
                     bool) -> bool { return true; };
    } else if (useNoMaterialStress) {
      batchedOp = soft::MakeBatchedBodyOp(
          MakeConstSpan(femLowVolDisc),
          MakeConstSpan(femHighVolDisc),
          noOpConstitutive,
          referenceStiffness,
          /*hasStress*/ true,
          (elopFlags & kElOpGravity) != 0,
          (elopFlags & kElOpInertia) != 0,
          kDefaultGravity,
          kDensity,
          predGlobalDisp.GetConstSpan(),
          predGlobalVelo.GetConstSpan(),
          kDt,
          /*massDampingScale*/ 0_r,
          stiffnessDampingFactor,
          /*includeStiffnessDampingGeometricTerm*/ false,
          /*activeVolWeights*/ Span<real const>{});
    } else {
      batchedOp = soft::MakeBatchedBodyOp(
          MakeConstSpan(femLowVolDisc),
          MakeConstSpan(femHighVolDisc),
          batchedConstitutive,
          referenceStiffness,
          (elopFlags & kElOpStress) != 0,
          (elopFlags & kElOpGravity) != 0,
          (elopFlags & kElOpInertia) != 0,
          kDefaultGravity,
          kDensity,
          predGlobalDisp.GetConstSpan(),
          predGlobalVelo.GetConstSpan(),
          kDt,
          /*massDampingScale*/ 0_r,
          stiffnessDampingFactor,
          useStiffnessDampingGeometric,
          /*activeVolWeights*/ Span<real const>{});
    }

    AssemblyParams params{
        .assemObj = assemObj, .assemRes = assemRes, .assemDRes = assemDRes, .psdDRes = psdProject};

    auto assembleOnce = [&]() {
      // Accumulate merit, residual and dresidual across benchmark iterations to exclude clearing
      // overhead.
      AssembleObjResDRes<soft::SoftStencilElement, soft::SoftStencilElement::kSpaceDim, kBatchSize>(
          l2g,
          nbs,
          batchedOp,
          sol,
          AssemblyResults<real>{
              .outObj = assemObj ? &merit : nullptr,
              .outRes = assemRes ? AsView(res) : ColumnVectorView<real>{},
              .outDRes = assemDRes ? AnyMatrixView<real>{AsView(dres)} : AnyMatrixView<real>{},
              .params = params},
          AssemblyActiveSubset{});
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
  }

  // Make sure the optimizer keeps all the calculations
  auto resNorm = res.Norm();
  auto dresNorm = dres.Norm();
  benchmark::DoNotOptimize(merit);
  benchmark::DoNotOptimize(resNorm);
  benchmark::DoNotOptimize(dresNorm);
}

// ---------------------------------------------------------------------------
// Soft Element Assembler Benchmark Cases
// ---------------------------------------------------------------------------

// Defines benchmarks with various numbers of threads
// clang-format off
#define MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, numThreads) \
  BENCHMARK_CAPTURE(ElementAssemblerBenchmark, name##_##numThreads, file, elopFlags, resultFlags, numThreads)   \
      ->Name("SoftAssembler/" benchmarkName "/Threads" #numThreads)

#define MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS(name, benchmarkName, file, elopFlags, resultFlags) \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 1);        \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 2);        \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 4);        \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 8);        \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 16);       \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 32);       \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 64);
// clang-format on

// Defines benchmarks with various element operations.
// Other combinations could be added (at the cost of slow benchmark runtime).
// clang-format off
#define MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(name, file)                                                                                                              \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS(name##_AssemblerOverhead_Obj, #name "/AssemblerOverhead/Objective", file, 0, kResultObj);                          \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS(name##_AssemblerOverhead_Res, #name "/AssemblerOverhead/Residual", file, 0, kResultRes);                           \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS(name##_AssemblerOverhead_DRes, #name "/AssemblerOverhead/DResidual", file, 0, kResultDRes);                        \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS(name##_AssemblerOverhead_All, #name "/AssemblerOverhead/All", file, 0, kResultObjResDRes);                         \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS(name##_GravityInertiaStress_Obj, #name "/GravityInertiaStress/Objective", file, kElOpVolumeDefault, kResultObj);   \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS(name##_GravityInertiaStress_Res, #name "/GravityInertiaStress/Residual", file, kElOpVolumeDefault, kResultRes);    \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS(name##_GravityInertiaStress_DRes, #name "/GravityInertiaStress/DResidual", file, kElOpVolumeDefault, kResultDRes); \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS(name##_GravityInertiaStress_All_Psd, #name "/GravityInertiaStress/All/Psd", file, kElOpVolumeDefault, kResultObjResDRes);
// clang-format on

// Meshes to assemble
// clang-format off
MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Armadillo_451, "armadillo/armadillo_451.mochi.h5"); // 451 nodes, 1165 elements
MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Armadillo_7717, "armadillo/armadillo_7717.mochi.h5"); // 7717 nodes, 35584 elements
MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Duck_359, "duck/duck_359.mochi.h5"); // 359 nodes, 1101 elements
MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Duck_13051, "duck/duck_13051.mochi.h5"); // 13051 nodes, 67889 elements
// clang-format on

// If you're looking for more mesh sizes...
// MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Armadillo_851, "armadillo/armadillo_851.mochi.h5");
// MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Armadillo_1744, "armadillo/armadillo_1744.mochi.h5");
// MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Duck_730, "duck/duck_730.mochi.h5");
// MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Duck_1899, "duck/duck_1899.mochi.h5");
// MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Duck_3504, "duck/duck_3504.mochi.h5");
// MOCHI_BENCHMARK_ELEMENT_ASSEMBLER(Duck_7756, "duck/duck_7756.mochi.h5");

// ---------------------------------------------------------------------------
// Stiffness-Damping A/B Benchmark Cases
// ---------------------------------------------------------------------------

// The end-to-end "enable stiffness damping" cost is the delta between GravityInertiaStress and
// GravityInertiaStressDamping for the same mesh/threads. The elastic baseline is already registered
// at full fan-out above, so only the damping-enabled counterpart is added here. It is deliberately
// restricted to one small + one large mesh and a reduced thread set to keep total runtime bounded.
// clang-format off
#define MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS_REDUCED(name, benchmarkName, file, elopFlags, resultFlags) \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 1);                \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 8);                \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_THREAD(name, benchmarkName, file, elopFlags, resultFlags, 32);

#define MOCHI_BENCHMARK_DAMPING_AB(name, file) \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS_REDUCED(name##_GravityInertiaStressDamping_All_Psd_NoGeo, #name "/GravityInertiaStressDamping/All/Psd/NoGeo", file, kElOpVolumeDefault | kElOpStiffnessDamping, kResultObjResDRes); \
  MOCHI_BENCHMARK_ELEMENT_ASSEMBLER_FOR_THREADS_REDUCED(name##_GravityInertiaStressDamping_All_Psd_Geo, #name "/GravityInertiaStressDamping/All/Psd/Geo", file, kElOpVolumeDefault | kElOpStiffnessDamping | kElOpStiffnessDampingGeometric, kResultObjResDRes);

MOCHI_BENCHMARK_DAMPING_AB(Duck_359, "duck/duck_359.mochi.h5"); // 359 nodes, 1101 elements
MOCHI_BENCHMARK_DAMPING_AB(Duck_13051, "duck/duck_13051.mochi.h5"); // 13051 nodes, 67889 elements
// clang-format on

// ---------------------------------------------------------------------------
// Stress-Only Mesh Assembler Benchmark Cases
// ---------------------------------------------------------------------------

// clang-format off
#define MOCHI_BENCHMARK_STRESS_CASE(name, benchmarkName, elopFlag, resultFlag)                          \
  BENCHMARK_CAPTURE(ElementAssemblerBenchmark, name, "duck/duck_359.mochi.h5", elopFlag, resultFlag, 1) \
      ->Name("SoftAssembler/Duck_359/" benchmarkName "/Threads1");

MOCHI_BENCHMARK_STRESS_CASE(Stress_NoMaterial_Obj, "Stress/NoMaterial/Objective", kElOpStress | kElOpMaterialNone | kElOpPsdNone, kResultObj);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NoMaterial_Res, "Stress/NoMaterial/Residual", kElOpStress | kElOpMaterialNone | kElOpPsdNone, kResultRes);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NoMaterial_Dres, "Stress/NoMaterial/DResidual", kElOpStress | kElOpMaterialNone | kElOpPsdNone, kResultDRes);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NoMaterial_All, "Stress/NoMaterial/All", kElOpStress | kElOpMaterialNone | kElOpPsdNone, kResultObjResDRes);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NeoHookean_Obj, "Stress/NeoHookean/Objective", kElOpStress | kElOpMaterialNeoHookean | kElOpPsdNone, kResultObj);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NeoHookean_Res, "Stress/NeoHookean/Residual", kElOpStress | kElOpMaterialNeoHookean | kElOpPsdNone, kResultRes);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NeoHookean_Dres_NoPsd, "Stress/NeoHookean/DResidual/NoPsd", kElOpStress | kElOpMaterialNeoHookean | kElOpPsdNone, kResultDRes);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NeoHookean_All_NoPsd, "Stress/NeoHookean/All/NoPsd", kElOpStress | kElOpMaterialNeoHookean | kElOpPsdNone, kResultObjResDRes);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NeoHookean_Dres_PsdFast, "Stress/NeoHookean/DResidual/PsdFast", kElOpStress | kElOpMaterialNeoHookean | kElOpPsdFast, kResultDRes);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NeoHookean_All_PsdFast, "Stress/NeoHookean/All/PsdFast", kElOpStress | kElOpMaterialNeoHookean | kElOpPsdFast, kResultObjResDRes);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NeoHookean_Dres_Psd, "Stress/NeoHookean/DResidual/Psd", kElOpStress | kElOpMaterialNeoHookean | kElOpPsdProjection, kResultDRes);
MOCHI_BENCHMARK_STRESS_CASE(Stress_NeoHookean_All_Psd, "Stress/NeoHookean/All/Psd", kElOpStress | kElOpMaterialNeoHookean | kElOpPsdProjection, kResultObjResDRes);
// clang-format on

// ---------------------------------------------------------------------------
// Fine-grained Stiffness-Damping Benchmark Cases (single-thread, Duck_359)
// ---------------------------------------------------------------------------

// These run stress + stiffness damping (damping requires stress); the delta against the matching
// Stress/NeoHookean/... cases above isolates the damping op's obj/res/dres/PSD contributions inside
// the real assembler (threading, scatter, block-sparse writes), without gravity/inertia noise. The
// PSD dresidual cases come in geometric-off (default modified-Newton tangent) and geometric-on
// (exact tangent) variants; the on/off delta measures the geometric-block cost end-to-end
// (dominated by the skipped PSD projection).
// clang-format off
MOCHI_BENCHMARK_STRESS_CASE(StressDamping_NeoHookean_Obj, "StressDamping/NeoHookean/Objective", kElOpStress | kElOpMaterialNeoHookean | kElOpStiffnessDamping | kElOpPsdNone, kResultObj);
MOCHI_BENCHMARK_STRESS_CASE(StressDamping_NeoHookean_Res, "StressDamping/NeoHookean/Residual", kElOpStress | kElOpMaterialNeoHookean | kElOpStiffnessDamping | kElOpPsdNone, kResultRes);
MOCHI_BENCHMARK_STRESS_CASE(StressDamping_NeoHookean_Dres_NoPsd, "StressDamping/NeoHookean/DResidual/NoPsd", kElOpStress | kElOpMaterialNeoHookean | kElOpStiffnessDamping | kElOpPsdNone, kResultDRes);
MOCHI_BENCHMARK_STRESS_CASE(StressDamping_NeoHookean_All_NoPsd, "StressDamping/NeoHookean/All/NoPsd", kElOpStress | kElOpMaterialNeoHookean | kElOpStiffnessDamping | kElOpPsdNone, kResultObjResDRes);
MOCHI_BENCHMARK_STRESS_CASE(StressDamping_NeoHookean_Dres_Psd_NoGeo, "StressDamping/NeoHookean/DResidual/Psd/NoGeo", kElOpStress | kElOpMaterialNeoHookean | kElOpStiffnessDamping | kElOpPsdProjection, kResultDRes);
MOCHI_BENCHMARK_STRESS_CASE(StressDamping_NeoHookean_All_Psd_NoGeo, "StressDamping/NeoHookean/All/Psd/NoGeo", kElOpStress | kElOpMaterialNeoHookean | kElOpStiffnessDamping | kElOpPsdProjection, kResultObjResDRes);
MOCHI_BENCHMARK_STRESS_CASE(StressDamping_NeoHookean_Dres_Psd_Geo, "StressDamping/NeoHookean/DResidual/Psd/Geo", kElOpStress | kElOpMaterialNeoHookean | kElOpStiffnessDamping | kElOpStiffnessDampingGeometric | kElOpPsdProjection, kResultDRes);
MOCHI_BENCHMARK_STRESS_CASE(StressDamping_NeoHookean_All_Psd_Geo, "StressDamping/NeoHookean/All/Psd/Geo", kElOpStress | kElOpMaterialNeoHookean | kElOpStiffnessDamping | kElOpStiffnessDampingGeometric | kElOpPsdProjection, kResultObjResDRes);
// clang-format on

#endif // MOCHI_INTERNAL
