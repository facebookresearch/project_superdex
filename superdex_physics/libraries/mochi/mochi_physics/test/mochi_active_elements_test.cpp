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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/subset_map.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_contact.h>
#include <mochi_physics/src/mochi_discretization_components.h>
#include <mochi_physics/src/mochi_discretization_functions.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mochi;

static std::shared_ptr<TetrahedralMesh const> LoadMeshFromAsset(std::string const& meshFilePath) {
  // Create a MochiPhysics instance just so it can load the mesh for us.
  Context* mochiContext = mochi::CreateContext(0);
  auto duckPath = test::GetAssetPath(meshFilePath);
  auto shape = mochiContext->LoadShapeFromFile(duckPath, ErrorAssert{});
  auto const& mesh = mochiContext->GetShapeMesh(shape, ErrorAssert{});
  auto coordinates = Unflatten<Real3 const>(mesh.coordinates);
  auto connectivity = Unflatten<Int4 const>(mesh.connectivity);
  auto tetMesh = std::make_shared<TetrahedralMesh>(coordinates, connectivity);
  mochi::DestroyContext(mochiContext);
  return tetMesh;
}

static CFemVolumeDiscretizationP1Q1 CreateFemVolDiscretization(TetrahedralMesh const& mesh) {
  auto const meshCoords = mesh.GetNodeCoordinates();
  auto const meshConnec = mesh.GetElementConnectivity();
  int const meshNumEle = mesh.GetNumElements();

  CFemVolumeDiscretizationP1Q1 femVolDisc;
  femVolDisc.femElements.reserve(meshNumEle);
  for (int i = 0; i < meshNumEle; ++i) {
    femVolDisc.femElements.emplace_back(i, meshCoords, meshConnec);
  }
  return femVolDisc;
}

static auto CreateTraces(
    TetrahedralMesh const& mesh,
    CFemVolumeDiscretizationP1Q1 const& femVolDisc) {
  using ET = tetrahedral::Pk3DElementTrace<tetrahedral::Pk3DElement<1, 1>, 1>;
  std::vector<ET> result;
  for (auto const& bdface : mesh.GetBoundaryFaces()) {
    result.emplace_back(femVolDisc.femElements[bdface.element], bdface.faceNum);
  }

  return result;
}

static std::vector<int> CollectGoldNodesFromSelectedVolumes(
    TetrahedralMesh const& mesh,
    Span<int const> selectedVolumes) {
  auto connec = mesh.GetElementConnectivity();

  std::vector<int> result;
  for (auto e : selectedVolumes) {
    auto const thisElemConnec = connec[e];
    for (int v : thisElemConnec) {
      result.push_back(v);
    }
  }
  SortAndRemoveDuplicates(result);
  return result;
}

template <class ET>
static std::vector<int> CollectGoldNodesFromSelectedTraces(
    TetrahedralMesh const& mesh,
    std::vector<ET> const& traceVec,
    Span<int const> selectedIndices) {
  auto connec = mesh.GetElementConnectivity();

  std::vector<int> result;
  for (auto traceIndex : selectedIndices) {
    auto const traceElement = traceVec[traceIndex];
    auto const elemIndex = traceElement.GetElementIndex();
    auto const thisElemConnec = connec[elemIndex];
    for (int v : thisElemConnec) {
      result.push_back(v);
    }
  }
  SortAndRemoveDuplicates(result);
  return result;
}

template <class ET>
static std::unordered_map<int, std::vector<int>> CreateParentVolElemToTraceIndexMap(
    std::vector<ET> const& traceVec,
    Span<int const> selectedIndices) {
  std::unordered_map<int, std::vector<int>> map;

  for (auto traceIndex : selectedIndices) {
    auto const traceElement = traceVec[traceIndex];
    auto const elemIndex = traceElement.GetElementIndex();
    map[elemIndex].push_back(traceIndex);
  }
  return map;
}

static void CheckNodes(Span<int const> nodes, Span<int const> goldNodes) {
  std::vector<int> nodesVec(nodes.begin(), nodes.end());
  sort(nodesVec);
  EXPECT_SPAN_EQ(nodesVec, goldNodes);
}

static std::vector<int> MergeVectors(Span<int const> v1, Span<int const> v2) {
  std::vector<int> r;
  r.reserve(v1.size() + v2.size());
  for (auto val : v1) {
    r.push_back(val);
  }
  for (auto val : v2) {
    r.push_back(val);
  }
  SortAndRemoveDuplicates(r);
  return r;
}

static void CheckContainsMethodForActiveVol(
    TetrahedralMesh const& mesh,
    CActiveVolumeElements const& actEl,
    Span<int const> elementsThatShouldReturnTrue) {
  std::set s(elementsThatShouldReturnTrue.begin(), elementsThatShouldReturnTrue.end());
  for (int iEl = 0; iEl < mesh.GetNumElements(); ++iEl) {
    if (s.contains(iEl)) {
      ASSERT_TRUE(actEl.Contains(iEl));
    } else {
      ASSERT_FALSE(actEl.Contains(iEl));
    }
  }
}

// ===================================================================================
//
// test active vol elements
//
// ===================================================================================

TEST_IF(MOCHI_USE_HDF5, MochiDiscretization, ActiveVolumeElements0) {
  auto mesh = LoadMeshFromAsset("duck/duck_730.mochi.h5");

  std::vector<int> activeElems = {};
  CActiveVolumeElements actEl(mesh, activeElems);

  // check active elements
  auto ai = actEl.ViewIndices();
  EXPECT_TRUE(ai.size() == 0);

  EXPECT_TRUE(actEl.empty());
  EXPECT_TRUE(actEl.ViewWeights().empty());

  // check active nodes
  auto uniqueNodes = actEl.ViewUniqueNodes();
  std::vector<int> goldActiveNodes{};
  EXPECT_SPAN_EQ(uniqueNodes, goldActiveNodes);

  CheckContainsMethodForActiveVol(*mesh, actEl, activeElems);
}

TEST_IF(MOCHI_USE_HDF5, MochiDiscretization, ActiveVolumeElements1) {
  auto mesh = LoadMeshFromAsset("duck/duck_730.mochi.h5");

  // note that the element IDs are chosen so that we have duplicates in the node IDs
  // that should be hanlded automatically internally
  std::vector<int> activeElems = {4, 81, 113};
  CActiveVolumeElements actEl(mesh, activeElems);

  EXPECT_FALSE(actEl.empty());
  EXPECT_TRUE(isize(actEl.ViewWeights()) == mesh->GetNumElements());

  // check active elements
  auto ai = actEl.ViewIndices();
  EXPECT_TRUE(ai.size() == 3);
  EXPECT_TRUE(ai[0] == 4);
  EXPECT_TRUE(ai[1] == 81);
  EXPECT_TRUE(ai[2] == 113);

  // check active nodes
  auto uniqueNodes = actEl.ViewUniqueNodes();
  std::vector<int> goldActiveNodes{480, 245, 569, 716, 308, 587, 127, 634, 196, 262, 471};
  EXPECT_SPAN_EQ(uniqueNodes, goldActiveNodes);

  CheckContainsMethodForActiveVol(*mesh, actEl, activeElems);
}

TEST_IF(MOCHI_USE_HDF5, MochiDiscretization, ActiveVolumeElements2) {
  auto mesh = LoadMeshFromAsset("duck/duck_730.mochi.h5");

  // now, we construct it with activeElemsA
  std::vector<int> activeElemsA = {4, 81, 113};
  CActiveVolumeElements actEl(mesh, activeElemsA);

  {
    EXPECT_FALSE(actEl.empty());
    EXPECT_TRUE(isize(actEl.ViewWeights()) == mesh->GetNumElements());

    // check active elements
    auto ai = actEl.ViewIndices();
    EXPECT_TRUE(ai.size() == 3);
    EXPECT_TRUE(ai[0] == 4);
    EXPECT_TRUE(ai[1] == 81);
    EXPECT_TRUE(ai[2] == 113);

    // check active nodes
    auto uniqueNodes = actEl.ViewUniqueNodes();
    std::vector<int> goldActiveNodes{480, 245, 569, 716, 308, 587, 127, 634, 196, 262, 471};
    EXPECT_SPAN_EQ(uniqueNodes, goldActiveNodes);

    CheckContainsMethodForActiveVol(*mesh, actEl, activeElemsA);
  }

  // and we replace the active elements activeElemsB which are FEWER than before
  std::vector<int> activeElemsB = {67, 657};
  actEl.Recompute(activeElemsB);

  {
    EXPECT_FALSE(actEl.empty());
    EXPECT_TRUE(isize(actEl.ViewWeights()) == mesh->GetNumElements());

    // check active elements
    auto ai = actEl.ViewIndices();
    EXPECT_TRUE(ai.size() == 2);
    EXPECT_TRUE(ai[0] == 67);
    EXPECT_TRUE(ai[1] == 657);

    // check active nodes
    auto uniqueNodes = actEl.ViewUniqueNodes();
    std::vector<int> goldActiveNodes{189, 375, 139, 610, 191, 186};
    EXPECT_SPAN_EQ(uniqueNodes, goldActiveNodes);

    CheckContainsMethodForActiveVol(*mesh, actEl, activeElemsB);
  }

  // and we replace the active elements activeElemsC which are MORE than before
  std::vector<int> activeElemsC = {1145, 2033, 33, 1911};
  actEl.Recompute(activeElemsC);

  {
    EXPECT_FALSE(actEl.empty());
    EXPECT_TRUE(isize(actEl.ViewWeights()) == mesh->GetNumElements());

    // check active elements
    auto ai = actEl.ViewIndices();
    EXPECT_TRUE(ai.size() == 4);
    EXPECT_TRUE(ai[0] == 1145);
    EXPECT_TRUE(ai[1] == 2033);
    EXPECT_TRUE(ai[2] == 33);
    EXPECT_TRUE(ai[3] == 1911);

    // check active nodes
    auto uniqueNodes = actEl.ViewUniqueNodes();
    std::vector<int> goldActiveNodes{
        182, 26, 176, 465, 383, 301, 531, 126, 414, 437, 710, 9, 421, 20, 618, 204};
    EXPECT_SPAN_EQ(uniqueNodes, goldActiveNodes);

    CheckContainsMethodForActiveVol(*mesh, actEl, activeElemsC);
  }

  // then make it empty
  std::vector<int> activeElemsD = {};
  actEl.Recompute(activeElemsD);

  {
    EXPECT_TRUE(actEl.empty());
    EXPECT_TRUE(isize(actEl.ViewWeights()) == 0);

    auto ai = actEl.ViewIndices();
    EXPECT_TRUE(ai.size() == 0);

    // check active nodes
    auto uniqueNodes = actEl.ViewUniqueNodes();
    std::vector<int> goldActiveNodes{};
    EXPECT_SPAN_EQ(uniqueNodes, goldActiveNodes);
  }

  // and then set again non-trivial active elements
  std::vector<int> activeElemsE = {67, 657};
  actEl.Recompute(activeElemsE);

  {
    EXPECT_FALSE(actEl.empty());
    EXPECT_TRUE(isize(actEl.ViewWeights()) == mesh->GetNumElements());

    // check active elements
    auto ai = actEl.ViewIndices();
    EXPECT_TRUE(ai.size() == 2);
    EXPECT_TRUE(ai[0] == 67);
    EXPECT_TRUE(ai[1] == 657);

    // check active nodes
    auto uniqueNodes = actEl.ViewUniqueNodes();
    std::vector<int> goldActiveNodes{189, 375, 139, 610, 191, 186};
    EXPECT_SPAN_EQ(uniqueNodes, goldActiveNodes);

    CheckContainsMethodForActiveVol(*mesh, actEl, activeElemsE);
  }
}

TEST(MochiDiscretization, ActiveVolumeElements3) {
  auto mesh =
      std::make_shared<TetrahedralMesh>(test::CreateMinimalTetMeshUnitCube(Real3{1_r, 1_r, 1_r}));

  std::vector<int> activeElems = {0, 1, 4};
  CActiveVolumeElements actEl(mesh, activeElems);

  EXPECT_FALSE(actEl.empty());
  EXPECT_TRUE(isize(actEl.ViewWeights()) == mesh->GetNumElements());

  // check active elements
  auto ai = actEl.ViewIndices();
  EXPECT_TRUE(ai.size() == 3);
  EXPECT_TRUE(ai[0] == 0);
  EXPECT_TRUE(ai[1] == 1);
  EXPECT_TRUE(ai[2] == 4);

  // check active nodes
  auto uniqueNodes = actEl.ViewUniqueNodes();
  std::vector<int> goldActiveNodes{0, 1, 2, 4, 6, 7};
  EXPECT_SPAN_EQ(uniqueNodes, goldActiveNodes);

  CheckContainsMethodForActiveVol(*mesh, actEl, activeElems);
}

// With every element active, the default weights are exactly 1: the active measure is summed like
// the mesh's total measure.
TEST(MochiDiscretization, AllActiveElementsHaveUnitWeights) {
  auto mesh = std::make_shared<TetrahedralMesh>(
      test::CreateMinimalTetMeshUnitGrid(Real3{0.3_r, 0.5_r, 0.7_r}, Int3{8, 8, 8}));
  DynamicArray<int> allVolumes(mesh->GetNumElements());
  std::iota(allVolumes.begin(), allVolumes.end(), 0);
  CActiveVolumeElements const actVolElems(mesh, allVolumes);

  auto femVolD = CreateFemVolDiscretization(*mesh);
  auto tracesVec = CreateTraces(*mesh, femVolD);
  DynamicArray<int> allTraces(tracesVec.size());
  std::iota(allTraces.begin(), allTraces.end(), 0);
  CActiveBoundaryFaces const actBdFaces(mesh, allTraces, MakeConstSpan(tracesVec));

  auto isOne = [](real weight) { return weight == 1_r; };
  EXPECT_TRUE(std::ranges::all_of(actVolElems.ViewWeights(), isOne));
  EXPECT_TRUE(std::ranges::all_of(actBdFaces.ViewWeights(), isOne));
}

// ===================================================================================
//
// test active traces
//
// ===================================================================================

TEST_IF(MOCHI_USE_HDF5, MochiDiscretization, ActiveBoundaryFaces0) {
  auto mesh = LoadMeshFromAsset("duck/duck_730.mochi.h5");
  auto femVolD = CreateFemVolDiscretization(*mesh);
  auto tracesVec = CreateTraces(*mesh, femVolD);

  //
  // do test
  std::vector<int> active = {};
  CActiveBoundaryFaces actBdFaces(mesh, active, MakeConstSpan(tracesVec));

  EXPECT_TRUE(actBdFaces.empty());
  EXPECT_TRUE(isize(actBdFaces.ViewWeights()) == 0);

  // check active elements
  auto ai = actBdFaces.ViewIndices();
  EXPECT_TRUE(ai.size() == 0);

  for (int i = 0; i < tracesVec.size(); ++i) {
    EXPECT_FALSE(actBdFaces.Contains(i));
  }

  // check active nodes
  auto uniqueNodes = actBdFaces.ViewUniqueVolumeNodes();
  std::vector<int> goldActiveNodes{};
  EXPECT_SPAN_EQ(uniqueNodes, goldActiveNodes);
}

TEST_IF(MOCHI_USE_HDF5, MochiDiscretization, ActiveBoundaryFaces1) {
  auto mesh = LoadMeshFromAsset("duck/duck_730.mochi.h5");
  auto femVolD = CreateFemVolDiscretization(*mesh);
  auto tracesVec = CreateTraces(*mesh, femVolD);

  std::vector<int> active = {5, 58};
  CActiveBoundaryFaces actBdFaces(mesh, active, MakeConstSpan(tracesVec));

  EXPECT_FALSE(actBdFaces.empty());
  EXPECT_TRUE(isize(actBdFaces.ViewWeights()) == mesh->GetBoundaryMesh()->GetNumElements());

  // check active elements
  auto ai = actBdFaces.ViewIndices();
  EXPECT_TRUE(ai.size() == 2);
  EXPECT_TRUE(ai[0] == 5);
  EXPECT_TRUE(ai[1] == 58);

  for (int i = 0; i < tracesVec.size(); ++i) {
    if (i != 5 && i != 58) {
      EXPECT_FALSE(actBdFaces.Contains(i));
    } else {
      EXPECT_TRUE(actBdFaces.Contains(i));
    }
  }

  // check active nodes
  auto uniqueNodes = actBdFaces.ViewUniqueVolumeNodes();
  auto goldActiveNodes = CollectGoldNodesFromSelectedTraces(*mesh, tracesVec, active);
  CheckNodes(uniqueNodes, goldActiveNodes);
}

TEST_IF(MOCHI_USE_HDF5, MochiDiscretization, ActiveBoundaryFaces2) {
  auto mesh = LoadMeshFromAsset("duck/duck_730.mochi.h5");
  auto femVolD = CreateFemVolDiscretization(*mesh);
  auto tracesVec = CreateTraces(*mesh, femVolD);

  auto testLambda = [&mesh, &tracesVec = std::as_const(tracesVec)](
                        Span<int const> activeVec,
                        CActiveBoundaryFaces const& actBdFaces,
                        int goldCount,
                        bool goldIsEmpty) {
    EXPECT_TRUE(actBdFaces.empty() == goldIsEmpty);
    EXPECT_TRUE(isize(actBdFaces.ViewWeights()) == mesh->GetBoundaryMesh()->GetNumElements());

    // check active elements
    auto ai = actBdFaces.ViewIndices();
    EXPECT_TRUE(ai.size() == goldCount);
    EXPECT_SPAN_EQ(ai, activeVec);

    std::set<int> activeSet(activeVec.begin(), activeVec.end());
    for (int i = 0; i < tracesVec.size(); ++i) {
      if (activeSet.count(i) == 1) {
        EXPECT_TRUE(actBdFaces.Contains(i));
      } else {
        EXPECT_FALSE(actBdFaces.Contains(i));
      }
    }

    // check active nodes
    auto uniqueNodes = actBdFaces.ViewUniqueVolumeNodes();
    auto goldActiveNodes = CollectGoldNodesFromSelectedTraces(*mesh, tracesVec, activeVec);
    CheckNodes(uniqueNodes, goldActiveNodes);
  };

  // now, we construct it with active set A
  std::vector<int> activeA = {0, 1, 366, 377, 178};
  CActiveBoundaryFaces actBdFaces(mesh, activeA, MakeConstSpan(tracesVec));
  testLambda(activeA, actBdFaces, /*goldCount = */ 5, /*goldIsEmpty = */ false);

  // and we replace the active elements active set B which are FEWER than before
  std::vector<int> activeB = {7, 713, 680};
  actBdFaces.Recompute(activeB, MakeConstSpan(tracesVec));
  testLambda(activeB, actBdFaces, /*goldCount = */ 3, /*goldIsEmpty = */ false);

  // and we replace the active elements active set C which are MORE than set B
  std::vector<int> activeC = {33, 801, 45, 902};
  actBdFaces.Recompute(activeC, MakeConstSpan(tracesVec));
  testLambda(activeC, actBdFaces, /*goldCount = */ 4, /*goldIsEmpty = */ false);

  // then set empty
  std::vector<int> activeD = {};
  actBdFaces.Recompute(activeD, MakeConstSpan(tracesVec));
  {
    EXPECT_TRUE(actBdFaces.empty());
    EXPECT_TRUE(isize(actBdFaces.ViewWeights()) == 0);

    // check active elements
    auto ai = actBdFaces.ViewIndices();
    EXPECT_TRUE(ai.size() == 0);

    // check active nodes
    auto uniqueNodes = actBdFaces.ViewUniqueVolumeNodes();
    EXPECT_TRUE(uniqueNodes.empty());
  }

  // and then set again non-trivial active elements
  std::vector<int> activeE = {7, 713, 680};
  actBdFaces.Recompute(activeE, MakeConstSpan(tracesVec));
  testLambda(activeE, actBdFaces, /*goldCount = */ 3, /*goldIsEmpty = */ false);
}

TEST(MochiDiscretization, ActiveBoundaryFaces3) {
  auto mesh =
      std::make_shared<TetrahedralMesh>(test::CreateMinimalTetMeshUnitCube(Real3{1_r, 1_r, 1_r}));
  auto femVolD = CreateFemVolDiscretization(*mesh);
  auto tracesVec = CreateTraces(*mesh, femVolD);

  std::vector<int> active = {0, 1, 6, 7, 8};
  CActiveBoundaryFaces actBdFaces(mesh, active, MakeConstSpan(tracesVec));

  EXPECT_FALSE(actBdFaces.empty());
  EXPECT_TRUE(isize(actBdFaces.ViewWeights()) == mesh->GetBoundaryMesh()->GetNumElements());

  // check active elements
  auto ai = actBdFaces.ViewIndices();
  EXPECT_TRUE(ai.size() == 5);
  EXPECT_SPAN_EQ(ai, active);

  std::set<int> activeSet(active.begin(), active.end());
  for (int i = 0; i < tracesVec.size(); ++i) {
    if (activeSet.count(i) == 1) {
      EXPECT_TRUE(actBdFaces.Contains(i));
    } else {
      EXPECT_FALSE(actBdFaces.Contains(i));
    }
  }

  // check active nodes
  auto uniqueNodes = actBdFaces.ViewUniqueVolumeNodes();
  auto goldActiveNodes = CollectGoldNodesFromSelectedTraces(*mesh, tracesVec, active);
  CheckNodes(uniqueNodes, goldActiveNodes);

  // Without displacements, the active samples are the reference samples of the active faces, in the
  // order of the active faces, with the quadrature weights scaled by the face weights.
  auto const boundaryDisc =
      CFemBoundaryDiscretization::Create(*mesh, femVolD, ActorBoundaryElementType::P1Q6);
  ColumnVector<real> const displacements =
      ColumnVector<real>::Zero(kSpaceDim3 * mesh->GetNumNodes());
  ContactSamples samples(boundaryDisc.GetNumQuadPoints());
  std::array constexpr kSampledFaces = {8, 1, 6};
  std::array constexpr kFaceWeights = {0.5_r, 2_r, 3_r};
  boundaryDisc.Visit([&](auto const& disc) {
    CActiveBoundaryFaces const sampledFaces(
        mesh,
        MakeConstSpan(kSampledFaces),
        MakeConstSpan(disc.femElements),
        MakeConstSpan(kFaceWeights));
    UpdateCollisionSamplePositionsFromNodeDisplacements<
        /*kUpdateOnlyActiveFaces*/ true,
        CFemBoundaryDiscretization,
        kSpaceDim3>(AsConstView(displacements), boundaryDisc, &sampledFaces, samples);

    int constexpr kNumQuads = std::decay_t<decltype(disc)>::ElementT::kNumQuadPoints;
    ASSERT_EQ(kNumQuads * isize(kSampledFaces), isize(samples.activePositions));
    ASSERT_EQ(kNumQuads * isize(kSampledFaces), isize(samples.activeIndices));
    for (int j = 0; j < isize(kSampledFaces); ++j) {
      int const face = kSampledFaces[j];
      for (int q = 0; q < kNumQuads; ++q) {
        int const sample = kNumQuads * face + q;
        EXPECT_EQ(sample, samples.activeIndices[kNumQuads * j + q]);
        EXPECT_EQ(disc.femElements[face].mapEvaluated[q], samples.positions[sample]);
        EXPECT_EQ(samples.positions[sample], samples.activePositions[kNumQuads * j + q]);
        EXPECT_EQ(disc.femElements[face].quadWeights[q] * kFaceWeights[j], samples.weights[sample]);
      }
    }
  });
}

// Expects picks to hold one index in each of the numPicks near-equal windows that partition
// [0, numItems), the windows of PickOneIndexPerWindow.
static void ExpectOnePickPerWindow(Span<int const> picks, int numItems, int numPicks) {
  ASSERT_EQ(numPicks, isize(picks));
  std::set<int> offsets;
  for (int i = 0; i < numPicks; ++i) {
    auto const windowBegin = static_cast<int>(int64_t(i) * numItems / numPicks);
    auto const windowEnd = static_cast<int>(int64_t(i + 1) * numItems / numPicks);
    EXPECT_LE(windowBegin, picks[i]);
    EXPECT_LT(picks[i], windowEnd);
    offsets.insert(picks[i] - windowBegin);
  }

  // With equal windows, a stride would put every pick at the same offset in its window
  if (1 < numPicks && numPicks < numItems) {
    EXPECT_LT(1, isize(offsets));
  }
}

// Uniform boundary subsampling and ROM sampling pick one face or element at random in each window
// of consecutive indices. On a grid stored row by row with two triangles per cell, like a cylinder
// stored ring by ring, a stride of 4 faces would sample only even columns. With 32 rows, a column
// goes unsampled with probability 2^-32 at a density of 0.25.
TEST(MochiDiscretization, SubsamplingPicksOnePerWindow) {
  int constexpr kNumColumns = 8;
  int constexpr kNumRows = 32;
  auto const triMesh = std::make_shared<TriangularMesh const>(
      UniformSquareTriangularMeshData(Int2{kNumColumns, kNumRows}));
  auto const surfaceDisc =
      CFemSurfaceDiscretization::Create(ActorBoundaryElementType::P1Q1, *triMesh);
  int const numTriangles = triMesh->GetNumElements();

  // A density of 0.3 rounds the number of picks up and partitions the faces into unequal windows
  for (real const density : {0.25_r, 0.3_r}) {
    BoundarySubsamplingParams const params{
        .subsamplingDensity = density, .strategy = BoundarySubsamplingStrategy::UniformProbability};
    auto const actBdFaces = CreateActiveBoundaryFaces(params, triMesh, surfaceDisc);
    auto const faces = actBdFaces.ViewIndices();
    ExpectOnePickPerWindow(faces, numTriangles, static_cast<int>(Ceil(density * numTriangles)));
    EXPECT_SPAN_EQ(faces, CreateActiveBoundaryFaces(params, triMesh, surfaceDisc).ViewIndices());

    std::set<int> sampledColumns;
    for (int const face : faces) {
      sampledColumns.insert(face / 2 % kNumColumns);
    }
    EXPECT_EQ(kNumColumns, isize(sampledColumns));
  }

  // ROM sampling on the 48 faces and 40 tets of a 2 x 2 x 2 tet grid, whose faces are listed by
  // element and whose tets are listed five per cell. Each pair gives faces and tets different
  // steps, so that mixing up the two fields fails: 3 for faces and 5 for tets partition both into
  // equal windows, 5 and 3 into unequal ones, and 64 exceeds both counts.
  auto const tetMesh = std::make_shared<TetrahedralMesh>(
      test::CreateMinimalTetMeshUnitGrid(Real3{1_r, 1_r, 1_r}, Int3{2, 2, 2}));
  TetrahedralMeshShape const tetShape(tetMesh);
  auto const femVolD = CreateFemVolDiscretization(*tetMesh);
  auto const traces = CreateTraces(*tetMesh, femVolD);
  int const numFaces = isize(traces);
  int const numTets = tetMesh->GetNumElements();

  for (auto const& [faceStep, tetStep] :
       {std::pair{rom::hyper::kAllElements, 64},
        std::pair{3, 5},
        std::pair{5, 3},
        std::pair{64, rom::hyper::kAllElements}}) {
    experimental::SampleMeshInitRandomSampling const params{
        .stepSizeForBoundaryElementsSelection = faceStep,
        .stepSizeForInteriorElementsSelection = tetStep};
    CActiveBoundaryFaces const romFaces = CreateActiveBoundaryFaces(params, tetMesh, traces);
    ExpectOnePickPerWindow(romFaces.ViewIndices(), numFaces, (numFaces + faceStep - 1) / faceStep);
    CActiveVolumeElements const romTets = CreateActiveVolumeElements(params, tetShape);
    ExpectOnePickPerWindow(romTets.ViewIndices(), numTets, (numTets + tetStep - 1) / tetStep);
  }

  // Each field alone keeps none at rom::hyper::kNoElements, while the other one keeps all
  EXPECT_TRUE(CreateActiveBoundaryFaces(
                  experimental::SampleMeshInitRandomSampling{
                      .stepSizeForBoundaryElementsSelection = rom::hyper::kNoElements},
                  tetMesh,
                  traces)
                  .empty());
  EXPECT_TRUE(CreateActiveVolumeElements(
                  experimental::SampleMeshInitRandomSampling{
                      .stepSizeForInteriorElementsSelection = rom::hyper::kNoElements},
                  tetShape)
                  .empty());
}

// ===================================================================================
//
// test active nodes
//
// ===================================================================================

TEST(MochiDiscretization, ActiveUniqueNodes0) {
  auto mesh =
      std::make_shared<TetrahedralMesh>(test::CreateMinimalTetMeshUnitCube(Real3{1_r, 1_r, 1_r}));
  auto femVolD = CreateFemVolDiscretization(*mesh);
  auto tracesVec = CreateTraces(*mesh, femVolD);

  std::vector<int> activeVols = {1, 4};
  CActiveVolumeElements actVolElems(mesh, activeVols);

  std::vector<int> activeTraces = {0, 7, 8};
  CActiveBoundaryFaces actBdFaces(mesh, activeTraces, MakeConstSpan(tracesVec));

  CActiveUniqueNodes actNodes(mesh, actVolElems, actBdFaces);
  auto nodes = actNodes.ViewIds();

  auto goldActiveNodesTraces = CollectGoldNodesFromSelectedTraces(*mesh, tracesVec, activeTraces);
  auto goldActiveNodesVols = CollectGoldNodesFromSelectedVolumes(*mesh, activeVols);
  auto goldNodes = MergeVectors(goldActiveNodesVols, goldActiveNodesTraces);
  CheckNodes(nodes, goldNodes);
}

TEST(MochiDiscretization, ActiveUniqueNodes1) {
  auto mesh =
      std::make_shared<TetrahedralMesh>(test::CreateMinimalTetMeshUnitCube(Real3{1_r, 1_r, 1_r}));
  auto femVolD = CreateFemVolDiscretization(*mesh);
  auto tracesVec = CreateTraces(*mesh, femVolD);

  std::vector<int> activeVols = {};
  CActiveVolumeElements actVolElems(mesh, activeVols);

  std::vector<int> activeTraces = {0, 7, 8};
  CActiveBoundaryFaces actBdFaces(mesh, activeTraces, MakeConstSpan(tracesVec));

  CActiveUniqueNodes actNodes(mesh, actVolElems, actBdFaces);
  auto nodes = actNodes.ViewIds();

  auto goldActiveNodesTraces = CollectGoldNodesFromSelectedTraces(*mesh, tracesVec, activeTraces);
  auto goldActiveNodesVols = CollectGoldNodesFromSelectedVolumes(*mesh, activeVols);
  auto goldNodes = MergeVectors(goldActiveNodesVols, goldActiveNodesTraces);
  CheckNodes(nodes, goldNodes);
}

TEST(MochiDiscretization, ActiveUniqueNodes2) {
  auto mesh =
      std::make_shared<TetrahedralMesh>(test::CreateMinimalTetMeshUnitCube(Real3{1_r, 1_r, 1_r}));
  auto femVolD = CreateFemVolDiscretization(*mesh);
  auto tracesVec = CreateTraces(*mesh, femVolD);

  std::vector<int> activeVols = {1, 4};
  CActiveVolumeElements actVolElems(mesh, activeVols);

  std::vector<int> activeTraces = {};
  CActiveBoundaryFaces actBdFaces(mesh, activeTraces, MakeConstSpan(tracesVec));

  CActiveUniqueNodes actNodes(mesh, actVolElems, actBdFaces);
  auto nodes = actNodes.ViewIds();

  auto goldActiveNodesTraces = CollectGoldNodesFromSelectedTraces(*mesh, tracesVec, activeTraces);
  auto goldActiveNodesVols = CollectGoldNodesFromSelectedVolumes(*mesh, activeVols);
  auto goldNodes = MergeVectors(goldActiveNodesVols, goldActiveNodesTraces);
  CheckNodes(nodes, goldNodes);
}

TEST(MochiDiscretization, ActiveUniqueNodes3) {
  auto mesh =
      std::make_shared<TetrahedralMesh>(test::CreateMinimalTetMeshUnitCube(Real3{1_r, 1_r, 1_r}));
  auto femVolD = CreateFemVolDiscretization(*mesh);
  auto tracesVec = CreateTraces(*mesh, femVolD);

  std::vector<int> activeVols = {};
  CActiveVolumeElements actVolElems(mesh, activeVols);
  std::vector<int> activeTraces = {};
  CActiveBoundaryFaces actBdFaces(mesh, activeTraces, MakeConstSpan(tracesVec));

  CActiveUniqueNodes actNodes(mesh, actVolElems, actBdFaces);
  auto nodes = actNodes.ViewIds();
  EXPECT_TRUE(nodes.empty());
}
