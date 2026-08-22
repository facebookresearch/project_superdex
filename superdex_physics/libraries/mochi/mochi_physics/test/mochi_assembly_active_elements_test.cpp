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

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/element_operations/fem_gravity.h>
#include <mochi_core/element_operations/fem_stress.h>
#include <mochi_core/element_operations/fem_traction.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/finite_element_trace.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/materials/material_params.h>
#include <mochi_core/rom/rom_hyper_reduction.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_core/utils/subset_map.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_discretization_components.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace mochi;

// The mesh assets used by these tests are not shipped externally.
#if MOCHI_USE_HDF5 && MOCHI_INTERNAL
#define MOCHI_HDF5_AND_INTERNAL 1
#else
#define MOCHI_HDF5_AND_INTERNAL 0
#endif

namespace {

constexpr int kNumFields = 3;
constexpr int kBatchSize = kDefaultFemBatchSize;
constexpr real kEnergyResTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1.5e-5_r;
constexpr real kDResTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-3_r;
using VolumeElementLow = tetrahedral::Pk3DElement<1, 1>;
using VolumeElementHigh = tetrahedral::Pk3DElement<1, 4>;
using ElementTraceT = tetrahedral::Pk3DElementTrace<VolumeElementLow, 3>;

template <class T>
std::vector<T> MergeVectorsIntoNewOne(std::vector<T> const& a, std::vector<T> const& b) {
  std::vector<T> r;
  r.reserve(a.size() + b.size());
  for (T v : a) {
    r.push_back(v);
  }
  for (T v : b) {
    r.push_back(v);
  }
  return r;
}

TetrahedralMesh LoadMeshFromAsset(std::string const& meshFilePath) {
  // Create a MochiPhysics instance just so it can load the mesh for us.
  Context* mochiContext = mochi::CreateContext(0);
  auto duckPath = test::GetAssetPath(meshFilePath);
  auto shape = mochiContext->LoadShapeFromFile(duckPath, ErrorAssert{});
  auto const& mesh = mochiContext->GetShapeMesh(shape, ErrorAssert{});
  auto coordinates = Unflatten<Real3 const>(mesh.coordinates);
  auto connectivity = Unflatten<Int4 const>(mesh.connectivity);
  TetrahedralMesh tetMesh(coordinates, connectivity);
  mochi::DestroyContext(mochiContext);
  return tetMesh;
}

auto CreateFemDiscretizations(TetrahedralMesh const& tetMesh) {
  int const numElements = tetMesh.GetNumElements();
  std::vector<VolumeElementLow> femLowVolDisc;
  std::vector<VolumeElementHigh> femHighVolDisc;
  femLowVolDisc.reserve(numElements);
  femHighVolDisc.reserve(numElements);
  for (int i = 0; i < numElements; ++i) {
    femLowVolDisc.emplace_back(
        i,
        tetMesh.GetNodeCoordinates(),
        tetMesh.GetElementConnectivity(),
        tetrahedral::kTetrahedralQuadrature1);
    femHighVolDisc.emplace_back(
        i,
        tetMesh.GetNodeCoordinates(),
        tetMesh.GetElementConnectivity(),
        tetrahedral::kTetrahedralQuadrature4);
  }

  return std::make_tuple(femLowVolDisc, femHighVolDisc);
}

std::vector<ElementTraceT> CreateElementTraces(
    TetrahedralMesh const& tetMesh,
    std::vector<VolumeElementLow> const& femLowVolDisc) {
  std::vector<ElementTraceT> elementsTraces;
  elementsTraces.reserve(tetMesh.GetNumBoundaryFaces());
  for (auto const& bdface : tetMesh.GetBoundaryFaces()) {
    elementsTraces.emplace_back(
        femLowVolDisc[bdface.element],
        static_cast<int>(bdface.faceNum),
        tetrahedral::kTetrahedralTraceQuadrature3[bdface.faceNum]);
  }
  return elementsTraces;
}

struct SpecificElements {
  std::vector<int> activeBdTraceIndices;
  std::vector<int> selectedVolumeElements;
};

struct StepSizes {
  int bdSampleStepSize = {};
  int innSampleStepSize = {};
};

struct Toggles {
  bool doGravity = true;
  bool doStress = true;
  bool doTraction = true;
};

constexpr Toggles kOnlyVolumeWork =
    Toggles{.doGravity = true, .doStress = true, .doTraction = false};
constexpr Toggles kOnlySurfaceWork =
    Toggles{.doGravity = false, .doStress = false, .doTraction = true};

std::vector<Toggles> CreateAllTogglesPermutations() {
  std::vector<Toggles> result;
  for (auto gravityB : {true, false}) {
    for (auto stressB : {true, false}) {
      for (auto tractionB : {true, false}) {
        result.emplace_back(
            Toggles{.doGravity = gravityB, .doStress = stressB, .doTraction = tractionB});
      }
    }
  }
  return result;
}

template <typename T>
void RemoveDuplicates(std::vector<T>& v) {
  std::sort(v.begin(), v.end());
  v.erase(unique(v.begin(), v.end()), v.end());
  auto rng = RandomGenerator(42);
  std::shuffle(v.begin(), v.end(), rng);
}

template <typename ElementTraceT>
std::vector<int> MapBdTraceIndicesToVolumeElements(
    std::vector<ElementTraceT> const& elementsTraces,
    std::vector<int> const& bdTraceIndices) {
  std::vector<int> result;
  result.reserve(bdTraceIndices.size());
  for (auto ti : bdTraceIndices) {
    result.push_back(elementsTraces[ti].GetElementIndex());
  }
  return result;
};

// Evaluates a batched element op for a single element (all lanes set to elemIndex). This builds an
// independently scattered reference from the same batched kernels that drive the assembler, so the
// comparison validates the assembler's gather / active-subset / scatter and lane independence, with
// kernel physics covered by per-kernel tests.
template <int kDof>
struct ElementEval {
  double energy = 0.0;
  std::array<real, kDof> residual{};
  std::array<real, kDof * kDof> dresidual{};
};

template <class ElementT, class OpT>
ElementEval<ElementT::kNumDofs * ElementT::kSpaceDim> EvalElementViaBatched(
    OpT const& op,
    int elemIndex,
    Span<int const> globalIndices,
    ColumnVector<real> const& sol,
    bool assemDRes = false) {
  constexpr int kDof = ElementT::kNumDofs * ElementT::kSpaceDim;
  using V = BatchReal<kBatchSize>;
  using DispT = fem::BatchElementVector<kBatchSize, ElementT>;
  using DResT = fem::BatchElementMatrix<kBatchSize, ElementT>;
  NdArray<int, kBatchSize> idx;
  for (int b = 0; b < kBatchSize; ++b) {
    idx[b] = elemIndex;
  }
  BatchDouble<kBatchSize> energy{};
  DispT res{};
  DResT dres{};
  bool const hasOutput = [&]() {
    if constexpr (std::is_invocable_v<
                      OpT const&,
                      NdArray<int, kBatchSize> const&,
                      Span<int const>,
                      DispT const&,
                      BatchDouble<kBatchSize>*,
                      DispT*,
                      DResT*,
                      bool>) {
      DispT disp{};
      for (int k = 0; k < kDof; ++k) {
        disp[k] = V{sol[globalIndices[k]]};
      }
      return op(idx, Span<int const>{}, disp, &energy, &res, assemDRes ? &dres : nullptr, false);
    } else {
      return op(idx, Span<int const>{}, &energy, &res, assemDRes ? &dres : nullptr, false);
    }
  }();
  EXPECT_TRUE(hasOutput);

  ElementEval<kDof> out;
  out.energy = energy[0];
  for (int k = 0; k < kDof; ++k) {
    out.residual[k] = res[k][0];
  }
  if (assemDRes) {
    for (int k = 0; k < kDof * kDof; ++k) {
      out.dresidual[k] = dres[k][0];
    }
  }
  return out;
}

} // namespace

void TestImpl(
    Toggles toggles,
    int numThreads,
    std::string const& meshFilePath,
    std::optional<SpecificElements> specificIds = {},
    std::optional<StepSizes> stepSz = {}) {
  MOCHI_ASSERT(specificIds || stepSz);

  int const numWorkerThreads = numThreads - 1;

  bool const doGravity = toggles.doGravity;
  bool const doStress = toggles.doStress;
  bool const doTraction = toggles.doTraction;
  constexpr bool kAssemObj = true;
  constexpr bool kAssemRes = true;
  bool const assemVolumeDRes = doStress;

  auto tetMesh = LoadMeshFromAsset(meshFilePath);

  // Create the material object
  auto generator = RandomGenerator(42);
  real E = RandomUniformValue(generator, 10_r, 10000_r);
  real nu = RandomUniformValue(generator, 0.1_r, 0.4_r);
  NeoHookeanMaterialParams materialParams;
  materialParams.youngsModulus = E;
  materialParams.poissonRatio = nu;

  auto [femLowVolDisc, femHighVolDisc] = CreateFemDiscretizations(tetMesh);

  // Note that the element traces take references to base elements, so femLowVolDisc must outlive
  // elementsTraces.
  auto elementsTraces = CreateElementTraces(tetMesh, femLowVolDisc);

  NodalBasedStructure nbs(tetMesh.GetElementConnectivity());

  Local2GlobalMap l2gTetra;
  tetrahedral::BarycentricBasisTetrahedra<1> basis;
  l2gTetra.InitializeFromMeshAndBasis(&tetMesh, basis, kNumFields);

  // SNLE data
  int numGlobalDofs = l2gTetra.GetGlobalRange().Size();
  ColumnVector<real> sol(numGlobalDofs);
  sol.SetRandom(111, -0.02_r, 0.02_r);

  SparseMatrix<real> spDres = MakeSparseMatrix(l2gTetra);
  auto dres = ToBlockSparseMatrix<kNumFields>(spDres);
  MOCHI_ASSERT(dres.Rows() == sol.Rows());
  dres.SetZero();

  // Physics parameters shared by the batched element operations below.
  Real3 const gravityVec = {0_r, -9.8_r, 0_r}; // any value will do
  constexpr real kDensity = 1000_r; // any value will do

  // Batched element operations, used to drive the FEM assembler. Both volume terms assemble into
  // the parent tet's 4 nodes, so the low-order element drives the assembler; the boundary traction
  // assembles through the trace element (also 4 parent nodes).
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  auto const lowElems = MakeConstSpan(femLowVolDisc);
  auto const highElems = MakeConstSpan(femHighVolDisc);
  auto const traceElems = MakeConstSpan(elementsTraces);

  auto const lame = materials::BuildBatchParams<kBatchSize>(materialParams);
  auto batchedConstitutive =
      [&](auto const&, auto const& F, auto* energy, auto* pk1, auto* tangent, bool psd) {
        materials::BatchedSmithNeoHookeanConstitutiveResponse<kBatchSize>(
            lame, F, energy, pk1, tangent, psd);
      };

  auto volumeOp = [&](NdArray<int, kBatchSize> const& batchElemIndices,
                      Span<int const> /*indicesFlat*/,
                      fem::BatchElementVector<kBatchSize, VolumeElementLow> const& batchDispl,
                      Vd* outEnergy,
                      fem::BatchElementVector<kBatchSize, VolumeElementLow>* outRes,
                      fem::BatchElementMatrix<kBatchSize, VolumeElementLow>* outDRes,
                      bool projectPsd) -> bool {
    bool out = false;
    if (doGravity) {
      out |= fem::GravityWork<kBatchSize>(
          batchElemIndices, lowElems, batchDispl, outEnergy, outRes, gravityVec, kDensity);
    }
    if (doStress) {
      out |= fem::StressWork<kBatchSize>(
          batchElemIndices,
          highElems,
          batchDispl,
          outEnergy,
          outRes,
          outDRes,
          projectPsd,
          batchedConstitutive);
    }
    return out;
  };

  auto tractionOp = [&](NdArray<int, kBatchSize> const& batchElemIndices,
                        Span<int const> /*indicesFlat*/,
                        Vd* outEnergy,
                        fem::BatchElementVector<kBatchSize, ElementTraceT>* outRes,
                        fem::BatchElementMatrix<kBatchSize, ElementTraceT>* outDRes,
                        bool /*projectPsd*/) -> bool {
    // Deterministic force/tangent. This test validates active-subset scatter, not traction physics.
    auto batchedTraction = [&traceElems](
                               NdArray<int, kBatchSize> const& eleIndices,
                               int q,
                               Vd* cbEnergy,
                               BatchReal3<kBatchSize>* cbForce,
                               NdArray<BatchReal3<kBatchSize>, 3>* cbDForce,
                               NdArray<bool, kBatchSize>& hasForce) {
      double energy[Vd::kSize]{};
      real sx[V::kSize]{};
      real sy[V::kSize]{};
      real sz[V::kSize]{};
      for (int b = 0; b < isize(eleIndices); ++b) {
        hasForce[b] = true;
        auto const& normal = traceElems[eleIndices[b]].normals[q];
        energy[b] = static_cast<double>(eleIndices[b]) + static_cast<double>(q) * 0.25;
        sx[b] = normal[0];
        sy[b] = normal[1];
        sz[b] = normal[2];
      }
      if (cbEnergy) {
        *cbEnergy = Load<Vd>(energy);
      }
      if (cbForce) {
        (*cbForce)[0] = Load<V>(sx);
        (*cbForce)[1] = Load<V>(sy);
        (*cbForce)[2] = Load<V>(sz);
      }
      if (cbDForce) {
        V const zero{0_r};
        V const stiffness{0.25_r};
        (*cbDForce)[0] = {stiffness, zero, zero};
        (*cbDForce)[1] = {zero, stiffness, zero};
        (*cbDForce)[2] = {zero, zero, stiffness};
      }
    };
    return fem::TractionWork<kBatchSize, ElementTraceT, kNumFields>(
        batchElemIndices, traceElems, outEnergy, outRes, outDRes, batchedTraction);
  };

  // Create a task scheduler and bind it to this thread
  TaskScheduler scheduler(numWorkerThreads);

  // ************************************************************************
  //
  // figure out active elements
  //
  // ************************************************************************

  std::vector<int> activeBdTraceIndices;
  std::vector<int> activeVolumeElements;
  if (specificIds) {
    activeBdTraceIndices = specificIds.value().activeBdTraceIndices;
    activeVolumeElements = specificIds.value().selectedVolumeElements;

  } else {
    // collect the subset of elements traces
    int const stepSizeForBoundaryTracesSelection = stepSz.value().bdSampleStepSize;
    int const numBdFaces = tetMesh.GetNumBoundaryFaces();
    for (int i = 0; i < numBdFaces; i += stepSizeForBoundaryTracesSelection) {
      activeBdTraceIndices.emplace_back(i);
    }

    // collect the subset of tet vol elements
    int const stepSizeForInteriorElementsSelection = stepSz.value().innSampleStepSize;
    int const numElements = tetMesh.GetNumElements();
    for (int i = 0; i < numElements; i += stepSizeForInteriorElementsSelection) {
      activeVolumeElements.emplace_back(i);
    }
  }
  RemoveDuplicates(activeBdTraceIndices);
  RemoveDuplicates(activeVolumeElements);

  // check the inputs are admissible for the given discretization
  auto maxBdIndex = std::ranges::max_element(activeBdTraceIndices);
  if (maxBdIndex != activeBdTraceIndices.end()) {
    MOCHI_ASSERT(*maxBdIndex < tetMesh.GetNumBoundaryFaces());
  }
  auto maxInnIndex = std::ranges::max_element(activeVolumeElements);
  if (maxInnIndex != activeVolumeElements.end()) {
    MOCHI_ASSERT(*maxInnIndex < tetMesh.GetNumElements());
  }

  auto activeBdElements = MapBdTraceIndicesToVolumeElements(elementsTraces, activeBdTraceIndices);
  RemoveDuplicates(activeBdElements);
  auto allActiveVolElements = MergeVectorsIntoNewOne(activeBdElements, activeVolumeElements);
  RemoveDuplicates(allActiveVolElements);

  int const numTotalElements = tetMesh.GetNumElements();
  DynamicArray<bool> isActiveVolElement;
  if (!allActiveVolElements.empty()) {
    isActiveVolElement.resize(numTotalElements, false);
    for (int e : allActiveVolElements) {
      isActiveVolElement[e] = true;
    }
  }

  // Build boundary-face L2G, NBS, and active bitmap for traction assembly
  BoundaryAssemblyData bdData(
      MakeConstSpan(elementsTraces), tetMesh.GetElementConnectivity(), nbs.GetNToN(), kNumFields);

  DynamicArray<bool> isActiveBdTrace(elementsTraces.size(), false);
  for (int i : activeBdTraceIndices) {
    isActiveBdTrace[i] = true;
  }

  if (doGravity || doStress) {
    auto assembledResidualStressGravity = ColumnVector<real>::Zero(numGlobalDofs);
    double assembledEnergyStressGravity = 0.0;
    AssemblyParams params{
        .assemObj = kAssemObj,
        .assemRes = kAssemRes,
        .assemDRes = assemVolumeDRes,
        .psdDRes = false};
    AssembleObjResDRes<VolumeElementLow, kNumFields, kBatchSize>(
        l2gTetra,
        nbs,
        volumeOp,
        sol,
        AssemblyResults<real>{
            .outObj = &assembledEnergyStressGravity,
            .outRes = AsView(assembledResidualStressGravity),
            .outDRes = AsView(dres),
            .params = params},
        AssemblyActiveSubset{allActiveVolElements, isActiveVolElement});

    // Compare against a hand-scattered reference for exactly the active volume subset.
    auto refResidualStressGravity = ColumnVector<real>::Zero(numGlobalDofs);
    auto refDResStressGravity = Matrix<real>::Zero(numGlobalDofs, numGlobalDofs);
    double refEnergyStressGravity = 0.0;
    constexpr int kVolumeDof = VolumeElementLow::kNumDofs * kNumFields;
    for (int e : allActiveVolElements) {
      auto globalIndices = l2gTetra.GetGlobalIndices(e);
      auto const elementEval =
          EvalElementViaBatched<VolumeElementLow>(volumeOp, e, globalIndices, sol, assemVolumeDRes);
      refEnergyStressGravity += elementEval.energy;
      for (int d = 0; d < isize(globalIndices); ++d) {
        refResidualStressGravity[globalIndices[d]] += elementEval.residual[d];
      }
      if (assemVolumeDRes) {
        for (int i = 0; i < isize(globalIndices); ++i) {
          for (int j = 0; j < isize(globalIndices); ++j) {
            refDResStressGravity(globalIndices[i], globalIndices[j]) +=
                elementEval.dresidual[i * kVolumeDof + j];
          }
        }
      }
    }

    EXPECT_NEAR(
        refEnergyStressGravity, assembledEnergyStressGravity, static_cast<double>(kEnergyResTol));
    EXPECT_TRUE(
        test::NearEqualMatrices(
            refResidualStressGravity, assembledResidualStressGravity, kEnergyResTol));
    if (assemVolumeDRes) {
      EXPECT_TRUE(test::NearEqualMatrices(refDResStressGravity, dres, kDResTol));
    }
  }

  if (doTraction) {
    auto assembledResidualTraction = ColumnVector<real>::Zero(numGlobalDofs);
    double assembledEnergyTraction = 0.0;
    dres.SetZero();
    AssemblyParams params{
        .assemObj = kAssemObj, .assemRes = kAssemRes, .assemDRes = true, .psdDRes = false};
    AssembleObjResDRes<ElementTraceT, kNumFields, kBatchSize>(
        bdData.l2g,
        bdData.nbs,
        tractionOp,
        AssemblyResults<real>{
            .outObj = &assembledEnergyTraction,
            .outRes = AsView(assembledResidualTraction),
            .outDRes = AsView(dres),
            .params = params},
        AssemblyActiveSubset{activeBdTraceIndices, isActiveBdTrace});

    // Compare against a hand-scattered reference for exactly the active boundary-trace subset.
    auto refResidualTraction = ColumnVector<real>::Zero(numGlobalDofs);
    auto refDResTraction = Matrix<real>::Zero(numGlobalDofs, numGlobalDofs);
    double refEnergyTraction = 0.0;
    constexpr int kTractionDof = ElementTraceT::kNumDofs * kNumFields;
    for (int traceIndex : activeBdTraceIndices) {
      int const volumeIndex = elementsTraces[traceIndex].GetElementIndex();
      auto globalIndices = l2gTetra.GetGlobalIndices(volumeIndex);
      auto const elementEval =
          EvalElementViaBatched<ElementTraceT>(tractionOp, traceIndex, globalIndices, sol, true);
      refEnergyTraction += elementEval.energy;
      for (int d = 0; d < isize(globalIndices); ++d) {
        refResidualTraction[globalIndices[d]] += elementEval.residual[d];
      }
      for (int i = 0; i < isize(globalIndices); ++i) {
        for (int j = 0; j < isize(globalIndices); ++j) {
          refDResTraction(globalIndices[i], globalIndices[j]) +=
              elementEval.dresidual[i * kTractionDof + j];
        }
      }
    }

    EXPECT_NEAR(refEnergyTraction, assembledEnergyTraction, static_cast<double>(kEnergyResTol));
    EXPECT_TRUE(
        test::NearEqualMatrices(refResidualTraction, assembledResidualTraction, kEnergyResTol));
    EXPECT_TRUE(test::NearEqualMatrices(refDResTraction, dres, kDResTol));
  }
}

//===================================================================
//===================================================================
//
// tests
//
//===================================================================
//===================================================================

TEST_IF(MOCHI_HDF5_AND_INTERNAL, AssemblerWithActiveElements, TestActiveElements0) {
  std::string const meshFilePath = "duck/duck_359.mochi.h5";

  /*no elements, because we want only volume work here*/
  std::vector<int> activeBdTraceIndices;

  // the inner active elements indices are based on the tetrahedra indexing
  std::vector<int> activeVolumeElements{5, 122, 943};

  int const numThreads = 1;
  TestImpl(
      kOnlyVolumeWork,
      numThreads,
      meshFilePath,
      SpecificElements{
          .activeBdTraceIndices = activeBdTraceIndices,
          .selectedVolumeElements = activeVolumeElements});
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, AssemblerWithActiveElements, TestActiveElements1) {
  std::string const meshFilePath = "duck/duck_359.mochi.h5";

  // the boundary active elements indices are based on the boundary traces indexing
  std::vector<int> activeBdTraceIndices{0, 255, 576};

  /*no volume elements, because we want only traction here*/
  std::vector<int> activeVolumeElements = {};

  int const numThreads = 1;
  TestImpl(
      kOnlySurfaceWork,
      numThreads,
      meshFilePath,
      SpecificElements{
          .activeBdTraceIndices = activeBdTraceIndices,
          .selectedVolumeElements = activeVolumeElements});
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, AssemblerWithActiveElements, TestActiveElements2) {
  std::string const meshFilePath = "duck/duck_359.mochi.h5";

  // the boundary active elements indices are based on the boundary traces indexing
  std::vector<int> activeBdTraceIndices{0, 255, 576};

  // the inner active elements indices are based on the tetrahedra indexing
  std::vector<int> activeVolumeElements{5, 122, 943};

  int const numThreads = 1;
  for (auto const& t : CreateAllTogglesPermutations()) {
    TestImpl(
        t,
        numThreads,
        meshFilePath,
        SpecificElements{
            .activeBdTraceIndices = activeBdTraceIndices,
            .selectedVolumeElements = activeVolumeElements});
  }
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, AssemblerWithActiveElements, TestActiveElements3) {
  std::string const meshFilePath = "duck/duck_359.mochi.h5";

  for (int numThreads : {1, 2, 3, 5}) {
    for (auto const& t : CreateAllTogglesPermutations()) {
      TestImpl(
          t,
          numThreads,
          meshFilePath,
          {},
          StepSizes{.bdSampleStepSize = 4, .innSampleStepSize = 7});
    }
  }
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, AssemblerWithActiveElements, TestActiveElements4) {
  std::string const meshFilePath = "duck/duck_359.mochi.h5";
  for (int numThreads : {1, 2, 3}) {
    for (auto const& t : CreateAllTogglesPermutations()) {
      TestImpl(
          t,
          numThreads,
          meshFilePath,
          {},
          StepSizes{.bdSampleStepSize = 14, .innSampleStepSize = 27});
    }
  }
}

TEST_IF(MOCHI_HDF5_AND_INTERNAL, AssemblerWithActiveElements, TestActiveElements5) {
  std::string const meshFilePath = "duck/duck_359.mochi.h5";
  for (int numThreads : {1, 2}) {
    for (auto const& t : CreateAllTogglesPermutations()) {
      TestImpl(
          t,
          numThreads,
          meshFilePath,
          {},
          StepSizes{.bdSampleStepSize = 22, .innSampleStepSize = 57});
    }
  }
}
