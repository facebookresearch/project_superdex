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
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/tetrahedral/finite_element_trace.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/rom/rom_hyper_reduction.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace mochi;

// Utilities to create finite element discretizations
using LowVolEleT = tetrahedral::Pk3DElement<1, 1>;
constexpr auto kLowVolQuad = tetrahedral::kTetrahedralQuadrature1;

using HighVolEleT = tetrahedral::Pk3DElement<1, 4>;
constexpr auto kHighVolQuad = tetrahedral::kTetrahedralQuadrature4;

using BoundaryEleT = tetrahedral::Pk3DElementTrace<tetrahedral::Pk3DElement<1, 1>, 6>;
constexpr auto kBoundaryQuad = tetrahedral::kTetrahedralTraceQuadrature6;

// Minimal assembler "shape" type: every discretization in these tests assembles into the parent
// tet's 4 nodes x 3 fields (L2G stride 12), so a single 4-node/3-field shape drives the batched
// assembler. The objective-only op below ignores the displacement, so only kNumDofs/kSpaceDim
// matter.
struct Q1ShapeElement {
  static constexpr int kNumDofs = 4;
  static constexpr int kSpaceDim = 3;
};

constexpr int kBatchSize = kDefaultFemBatchSize;

template <typename ElementT, typename QuadratureT>
std::vector<ElementT> MakeElements(TetrahedralMesh const& mesh, QuadratureT const& quadrature) {
  std::vector<ElementT> elements;
  elements.reserve(mesh.GetNumElements());
  for (int i = 0; i < mesh.GetNumElements(); ++i) {
    elements.emplace_back(i, mesh.GetNodeCoordinates(), mesh.GetElementConnectivity(), quadrature);
  }
  return elements;
}

template <typename ElementT, typename BaseElementT, typename QuadratureT>
std::vector<ElementT> MakeBoundaryElements(
    TetrahedralMesh const& mesh,
    Span<BaseElementT const> baseElements,
    QuadratureT const& faceQuadrature) {
  std::vector<ElementT> elements;
  for (int i = 0; i < mesh.GetNumBoundaryFaces(); ++i) {
    auto const& info = mesh.GetBoundaryFaces()[i];
    elements.emplace_back(baseElements[info.element], info.faceNum, faceQuadrature[info.faceNum]);
  }
  return elements;
}

template <typename ElementT>
std::vector<ElementT> ModifyQuadratureWeights(
    std::vector<ElementT> elements,
    Span<real const> weights) {
  MOCHI_ASSERT(weights.size() == elements.size());

  for (int i = 0; i < elements.size(); ++i) {
    elements[i].quadWeights *= weights[i];
  }

  return elements;
}

// Batched objective-only op integrating the constant 1 over each element (i.e. its measure),
// accumulating per batch lane into the batched energy. Drives the FEM assembler.
template <typename ElementT>
auto MakeOneBatchedObjOp(Span<ElementT const> elements) {
  return [elements](
             NdArray<int, kBatchSize> const& batchElemIndices,
             Span<int const> /*indicesFlat*/,
             fem::BatchElementVector<kBatchSize, Q1ShapeElement> const& /*disp*/,
             BatchDouble<kBatchSize>* outEnergy,
             fem::BatchElementVector<kBatchSize, Q1ShapeElement>* /*outRes*/,
             fem::BatchElementMatrix<kBatchSize, Q1ShapeElement>* /*outDRes*/,
             bool /*projectPsd*/) -> bool {
    if (outEnergy == nullptr) {
      return false;
    }
    for (int b = 0; b < kBatchSize; ++b) {
      auto const measure = static_cast<double>(Sum(elements[batchElemIndices[b]].quadWeights));
      *outEnergy = Set(*outEnergy, b, (*outEnergy)[b] + measure);
    }
    return true;
  };
}

/*
    Tests that integrating a constant on a sample mesh gives the same result
    as integrating the constant on a full mesh.
*/
static void TestSubsamplingWeightConsistency(std::string const& meshSrc) {
  auto tetMeshSrc = test::GetAssetPath(meshSrc);
  auto mesh = LoadTetrahedralMesh(tetMeshSrc, ErrorAssert{});

  // Create sample mesh
  rom::hyper::BoundaryAndInternalElementsSubsamplingParameters params;
  params.stepSizeForBoundaryElementsSelection = 41;
  params.stepSizeForInteriorElementsSelection = 15;
  auto sampleMeshData = rom::hyper::CreateSampleMeshAndWeights(*mesh, params, ErrorAssert{});

  // Create finite elements
  auto volLowFull = MakeElements<LowVolEleT>(*mesh, kLowVolQuad);
  auto volLowSample = ModifyQuadratureWeights(
      MakeElements<LowVolEleT>(*sampleMeshData.mesh, kLowVolQuad),
      sampleMeshData.weighting.volumeElements);
  auto volHighFull = MakeElements<HighVolEleT>(*mesh, kHighVolQuad);
  auto volHighSample = ModifyQuadratureWeights(
      MakeElements<HighVolEleT>(*sampleMeshData.mesh, kHighVolQuad),
      sampleMeshData.weighting.volumeElements);
  auto boundaryFull =
      MakeBoundaryElements<BoundaryEleT, LowVolEleT>(*mesh, volLowFull, kBoundaryQuad);
  auto boundarySample = ModifyQuadratureWeights(
      MakeBoundaryElements<BoundaryEleT, LowVolEleT>(
          *sampleMeshData.mesh, volLowSample, kBoundaryQuad),
      sampleMeshData.weighting.boundaryFaceElements);

  // Create batched element operations for integrating the constant one
  auto batchedObjOps = std::vector<ElOpFnType<Q1ShapeElement, 3, kBatchSize>>{
      MakeOneBatchedObjOp<LowVolEleT>(volLowFull),
      MakeOneBatchedObjOp<LowVolEleT>(volLowSample),
      MakeOneBatchedObjOp<HighVolEleT>(volHighFull),
      MakeOneBatchedObjOp<HighVolEleT>(volHighSample),
      MakeOneBatchedObjOp<BoundaryEleT>(boundaryFull),
      MakeOneBatchedObjOp<BoundaryEleT>(boundarySample)};

  // Create the appropriate assemblers for each element operation
  Local2GlobalMap l2gFull;
  l2gFull.InitializeFromMesh(mesh.get(), 3);
  Local2GlobalMap l2gSample;
  l2gSample.InitializeFromMesh(sampleMeshData.mesh.get(), 3);

  // Create a task scheduler and bind it to this thread
  TaskScheduler scheduler;

  auto zerosFull = ColumnVector<real>::Zero(mesh->GetNumNodes() * 3);
  auto zerosSample = ColumnVector<real>::Zero(sampleMeshData.mesh->GetNumNodes() * 3);

  // Everything else needed to do assembly
  struct LocalAssemblyParams {
    Local2GlobalMap const& l2g;
    ColumnVectorView<real const> input;
    NodalBasedStructure const& nbs;
    AssemblyActiveSubset activeSubset = {};

    LocalAssemblyParams(
        Local2GlobalMap const& l2gIn,
        ColumnVector<real> const& input,
        NodalBasedStructure const& nbsIn)
        : l2g(l2gIn), input(input), nbs(nbsIn) {}

    LocalAssemblyParams(
        Local2GlobalMap const& l2gIn,
        ColumnVector<real> const& input,
        NodalBasedStructure const& nbsIn,
        AssemblyActiveSubset activeSubsetIn)
        : l2g(l2gIn), input(input), nbs(nbsIn), activeSubset(activeSubsetIn) {}
  };

  NodalBasedStructure nbsFull(mesh->GetElementConnectivity());
  NodalBasedStructure nbsSample(sampleMeshData.mesh->GetElementConnectivity());

  // Build boundary-face L2G and NBS for full mesh
  BoundaryAssemblyData bdDataFull(
      MakeConstSpan(boundaryFull), mesh->GetElementConnectivity(), nbsFull.GetNToN(), 3);

  // Build boundary-face L2G and NBS for sample mesh
  BoundaryAssemblyData bdDataSample(
      MakeConstSpan(boundarySample),
      sampleMeshData.mesh->GetElementConnectivity(),
      nbsSample.GetNToN(),
      3);

  // Build shuffled boundary face indices for testing order-independence and subset filtering
  auto rng = RandomGenerator(42);
  std::vector<int> bdFaceIndicesFull(boundaryFull.size());
  std::iota(bdFaceIndicesFull.begin(), bdFaceIndicesFull.end(), 0);
  std::shuffle(bdFaceIndicesFull.begin(), bdFaceIndicesFull.end(), rng);
  DynamicArray<bool> isBdFaceActiveFull(boundaryFull.size(), true);

  std::vector<int> bdFaceIndicesSample(boundarySample.size());
  std::iota(bdFaceIndicesSample.begin(), bdFaceIndicesSample.end(), 0);
  std::shuffle(bdFaceIndicesSample.begin(), bdFaceIndicesSample.end(), rng);
  DynamicArray<bool> isBdFaceActiveSample(boundarySample.size(), true);

  // Mirrors the terms vector
  auto paramsPerFunc = std::vector<LocalAssemblyParams>{
      {l2gFull, zerosFull, nbsFull},
      {l2gSample, zerosSample, nbsSample},
      {l2gFull, zerosFull, nbsFull},
      {l2gSample, zerosSample, nbsSample},
      {bdDataFull.l2g,
       zerosFull,
       bdDataFull.nbs,
       AssemblyActiveSubset{bdFaceIndicesFull, isBdFaceActiveFull}},
      {bdDataSample.l2g,
       zerosSample,
       bdDataSample.nbs,
       AssemblyActiveSubset{bdFaceIndicesSample, isBdFaceActiveSample}}};

  // Actually do the integration
  AssemblyParams assemblyParams{.assemObj = true, .assemRes = false, .assemDRes = false};

  std::vector<real> integrationResults;
  std::transform(
      batchedObjOps.begin(),
      batchedObjOps.end(),
      paramsPerFunc.begin(),
      std::back_inserter(integrationResults),
      [&](ElOpFnType<Q1ShapeElement, 3, kBatchSize> const& op,
          LocalAssemblyParams localParams) -> real {
        double result = 0.0;

        AssembleObjResDRes<Q1ShapeElement, 3, kBatchSize>(
            localParams.l2g,
            localParams.nbs,
            op,
            localParams.input,
            AssemblyResults<real>{
                .outObj = &result, .outRes = {}, .outDRes = {}, .params = assemblyParams},
            localParams.activeSubset);

        return static_cast<real>(result);
      });

  EXPECT_TRUE(NearEqualRel(integrationResults[0], integrationResults[1], 1E-4_r));
  EXPECT_TRUE(NearEqualRel(integrationResults[2], integrationResults[3], 1E-4_r));
  EXPECT_TRUE(NearEqualRel(integrationResults[4], integrationResults[5], 1E-4_r));
}

/*
    Tests that integrating a constant on a sample mesh gives the same result
    as integrating the constant on a full mesh.
*/
TEST(HyperReduction, SubsamplingWeightConsistency) {
  std::vector<std::string> meshSrcs = {"sphere/icosphere_4subdiv.1.mochi.json"};
  // These meshes are not shipped externally.
#if MOCHI_INTERNAL
  meshSrcs.emplace_back("duck/duck_fine_mesh.mochi.json");
  meshSrcs.emplace_back("duck/duck_coarse_mesh.mochi.json");
  meshSrcs.emplace_back("dragon/dragon_mesh.mochi.json");
#endif

  for (auto const& src : meshSrcs) {
    TestSubsamplingWeightConsistency(src);
  }
}

TEST(TetrahedralMesh, SampleMeshCorrectNumFaces) {
  // A solid unit cube with one corner at (0,0,0)
  //
  //         6 ------- 7
  //       / |       / |
  //      /  |      /  |
  //     2 ------- 3   |
  //     |   4 ----|-- 5
  //     |  /      |  /
  //     | /       | /
  //     0 ------- 1
  //
  std::vector<Real3> const coordinates = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{1.0_r, 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, 1.0_r, 0.0_r}, // 2
      Real3{1.0_r, 1.0_r, 0.0_r}, // 3
      Real3{0.0_r, 0.0_r, 1.0_r}, // 4
      Real3{1.0_r, 0.0_r, 1.0_r}, // 5
      Real3{0.0_r, 1.0_r, 1.0_r}, // 6
      Real3{1.0_r, 1.0_r, 1.0_r}, // 7
  };
  std::vector<Int4> const connectivity = {
      Int4{2, 6, 3, 0}, // corner vert 2
      Int4{7, 3, 6, 5}, // corner vert 7
      Int4{1, 3, 5, 0}, // corner vert 1
      Int4{4, 0, 5, 6}, // corner vert 4
      Int4{6, 0, 3, 5}, // the one fully interior tetrahedron
  };

  TetrahedralMesh mesh(coordinates, connectivity);

  // the tet mesh has 4 boundary elements and 1 interior
  // pass 5,5 so that we sample only 1 boundary element and the only interior one

  auto meshSubset = rom::hyper::SampleMeshSubset(
      mesh, rom::hyper::BoundaryAndInternalElementsSubsamplingParameters{5, 5}, ErrorAssert{});
  auto sampleMeshUnweighted =
      rom::hyper::CreateUnweightedSampleMeshFromVolumeElements(mesh, meshSubset.sampleElements);
  auto& sampleMesh = *sampleMeshUnweighted.mesh;

  EXPECT_EQ(sampleMesh.GetNumNodes(), 5);
  EXPECT_EQ(sampleMesh.GetNumFaces(), 3);
}

template <typename MeshType>
static auto ComputeSampleMeshWeights(MeshType const& mesh, std::vector<int> const& activeList) {
  real fullSum = 0_r;
  for (int i = 0; i < mesh.GetNumElements(); ++i) {
    fullSum += mesh.GetElementMeasure(i);
  }

  real activeSum = 0_r;
  for (auto indexIt : activeList) {
    activeSum += mesh.GetElementMeasure(indexIt);
  }

  real const faceReweightFactor = fullSum / activeSum;
  return std::vector<real>(activeList.size(), faceReweightFactor);
}

TEST(HyperReduction, SubsamplingWeightConsistencyCube) {
  // A solid cube with one corner at (0,0,0)
  //
  //         2 ------- 3
  //       / |       / |
  //      /  |      /  |
  //     6 ------- 7   |
  //     |   0 ----|-- 1
  //     |  /      |  /
  //     | /       | /
  //     4 ------- 5

  real scaleX = 0.5_r;
  real scaleY = 1.0_r;
  real scaleZ = 2.5_r;
  Real3 scale = Real3{scaleX, scaleY, scaleZ};
  TetrahedralMesh mesh = test::CreateMinimalTetMeshUnitCube(scale);
  NodalBasedStructure volumeNbs(mesh.GetElementConnectivity());

  TaskScheduler scheduler;

  auto zerosFull = ColumnVector<real>::Zero(mesh.GetNumNodes() * 3);

  double resultA1 = 0.0;
  double resultA2 = 0.0;
  double resultB1 = 0.0;
  double resultB2 = 0.0;
  AssemblyParams params{.assemObj = true, .assemRes = false, .assemDRes = false};
  {
    auto volLowElems = MakeElements<LowVolEleT>(mesh, kLowVolQuad);
    auto bdTraces =
        MakeBoundaryElements<BoundaryEleT, LowVolEleT>(mesh, volLowElems, kBoundaryQuad);

    // Build boundary-face L2G and NBS
    BoundaryAssemblyData bdData(
        MakeConstSpan(bdTraces), mesh.GetElementConnectivity(), volumeNbs.GetNToN(), 3);

    // Shuffle all boundary face indices to test order-independence and subset filtering
    std::vector<int> allBdFaceIndices(bdTraces.size());
    std::iota(allBdFaceIndices.begin(), allBdFaceIndices.end(), 0);
    auto rng = RandomGenerator(42);
    std::shuffle(allBdFaceIndices.begin(), allBdFaceIndices.end(), rng);
    DynamicArray<bool> isAllBdFaceActive(bdTraces.size(), true);

    AssembleObjResDRes<Q1ShapeElement, 3, kBatchSize>(
        bdData.l2g,
        bdData.nbs,
        MakeOneBatchedObjOp<BoundaryEleT>(bdTraces),
        zerosFull,
        AssemblyResults<real>{.outObj = &resultA1, .outRes = {}, .outDRes = {}, .params = params},
        AssemblyActiveSubset{allBdFaceIndices, isAllBdFaceActive});

    // do it manually now
    for (auto const& trace : bdTraces) {
      resultA2 += Sum(trace.quadWeights);
    }
  }

  {
    auto volLowElems = MakeElements<LowVolEleT>(mesh, kLowVolQuad);
    auto bdTraces =
        MakeBoundaryElements<BoundaryEleT, LowVolEleT>(mesh, volLowElems, kBoundaryQuad);

    std::vector<int> activeTraceIndex{1};

    // Build boundary-face L2G and NBS
    BoundaryAssemblyData bdData(
        MakeConstSpan(bdTraces), mesh.GetElementConnectivity(), volumeNbs.GetNToN(), 3);

    // Active subset: boundary face indices
    DynamicArray<bool> isActiveTrace(bdTraces.size(), false);
    for (int i : activeTraceIndex) {
      isActiveTrace[i] = true;
    }

    auto const& bdMesh = *mesh.GetBoundaryMesh();
    auto const faceW = ComputeSampleMeshWeights(bdMesh, activeTraceIndex);
    bdTraces[activeTraceIndex[0]].quadWeights *= faceW[0];

    AssembleObjResDRes<Q1ShapeElement, 3, kBatchSize>(
        bdData.l2g,
        bdData.nbs,
        MakeOneBatchedObjOp<BoundaryEleT>(bdTraces),
        zerosFull,
        AssemblyResults<real>{.outObj = &resultB1, .outRes = {}, .outDRes = {}, .params = params},
        AssemblyActiveSubset{activeTraceIndex, isActiveTrace});

    resultB2 = Sum(bdTraces[activeTraceIndex[0]].quadWeights);
  }

  EXPECT_NEAR(resultA1, resultA2, 1E-4_r);
  EXPECT_NEAR(resultB1, resultB2, 1E-4_r);
  EXPECT_NEAR(resultA1, resultB1, 1E-4_r);
}
