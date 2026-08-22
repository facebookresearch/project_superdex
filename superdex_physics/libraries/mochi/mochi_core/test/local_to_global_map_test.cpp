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

#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/local_to_global_map.h>
#include <mochi_core/utils/span.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::test;

namespace {
struct PaddedConnectivityTestData {
  Graph<int, int> connectivity;
  Graph<int, int> stencilGraph;
};
} // namespace

// Builds a 2-element Local2GlobalMap from per-element node lists with @p numFields fields per node.
static Local2GlobalMap MakeMap(
    int numFields,
    DynamicArray<DynamicArray<int>> const& elementNodeConnectivity) {
  Local2GlobalMap map;
  map.InitializeFromElementNodeConnectivity(elementNodeConnectivity, numFields);
  return map;
}

static void ExpectGraphEq(Graph<int, int> const& expected, Graph<int, int> const& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_SPAN_EQ(expected[i], actual[i]);
  }
}

static PaddedConnectivityTestData MakePaddedConnectivityTestData() {
  DynamicArray<DynamicArray<int>> connectivity;
  connectivity.push_back(DynamicArray<int>{0, 1, 2, 3});
  connectivity.push_back(DynamicArray<int>{1, 4});

  DynamicArray<DynamicArray<int>> stencil;
  stencil.push_back(DynamicArray<int>{0, 1, 2, 3});
  stencil.push_back(DynamicArray<int>{0, 3});

  return {
      .connectivity = GraphFromRangeOfRanges<int, int>(connectivity),
      .stencilGraph = GraphFromRangeOfRanges<int, int>(stencil)};
}

TEST(AssemblyTest, BuildPaddedConnectivity) {
  auto const data = MakePaddedConnectivityTestData();
  auto const paddedConnectivity = BuildPaddedConnectivity<4>(data.connectivity, data.stencilGraph);

  ASSERT_EQ(2, isize(paddedConnectivity));
  Int4 const expected0{0, 1, 2, 3};
  Int4 const expected1{1, 1, 1, 4};
  EXPECT_SPAN_EQ(expected0, paddedConnectivity[0]);
  EXPECT_SPAN_EQ(expected1, paddedConnectivity[1]);
}

TEST(AssemblyTest, BuildPaddedNodalBasedStructure) {
  auto const data = MakePaddedConnectivityTestData();
  auto const nbs = BuildPaddedNodalBasedStructure<4>(data.connectivity, data.stencilGraph);

  Int4 const expected0{0, 1, 2, 3};
  Int4 const expected1{1, 1, 1, 4};
  EXPECT_SPAN_EQ(expected0, nbs.GetEleNodes(0));
  EXPECT_SPAN_EQ(expected1, nbs.GetEleNodes(1));

  NodalBasedStructure const originalNbs(Graph<int, int>{data.connectivity});
  ExpectGraphEq(originalNbs.GetNToN(), nbs.GetNToN());
}

TEST(AssemblyTest, PaddedConnectivityMatchesPaddedLocalToGlobalMap) {
  int constexpr kNumFields = 3;
  int constexpr kNumStencilNodes = 4;
  auto const data = MakePaddedConnectivityTestData();
  auto const paddedConnectivity =
      BuildPaddedConnectivity<kNumStencilNodes>(data.connectivity, data.stencilGraph);

  Local2GlobalMap l2g;
  l2g.InitializeFromElementNodeConnectivity(data.connectivity, kNumFields);
  l2g.InitializeStencilIndices(data.stencilGraph);
  l2g.InitializePaddedIndices(kNumStencilNodes * kNumFields);

  for (int e = 0; e < isize(paddedConnectivity); ++e) {
    auto const paddedDofs = l2g.GetPaddedGlobalIndices().subspan(
        e * kNumStencilNodes * kNumFields, kNumStencilNodes * kNumFields);
    for (int n = 0; n < kNumStencilNodes; ++n) {
      for (int f = 0; f < kNumFields; ++f) {
        EXPECT_EQ(paddedConnectivity[e][n] * kNumFields + f, paddedDofs[n * kNumFields + f]);
      }
    }
  }
}

TEST(Local2GlobalMapTest, PaddedIndices) {
  constexpr int kNumFields = 3;
  constexpr int kStride = 12;

  // ---- Case 1: stride == eleSize, no padding actually written ----
  // Two 4-node elements; flat indices fully populate every padded slot.
  {
    DynamicArray<DynamicArray<int>> connectivity;
    connectivity.push_back(DynamicArray<int>{0, 1, 2, 3});
    connectivity.push_back(DynamicArray<int>{4, 5, 6, 7});
    auto map = MakeMap(kNumFields, connectivity);

    EXPECT_FALSE(map.HasPaddedIndices());
    map.InitializePaddedIndices(kStride);
    EXPECT_TRUE(map.HasPaddedIndices());
    EXPECT_EQ(map.GetPaddedStride(), kStride);
    // Padded == raw flat indices when stride == eleSize for every element.
    EXPECT_SPAN_EQ(map.GetPaddedGlobalIndices(), map.GetGlobalIndices());
  }

  // ---- Case 2: variable stride, no stencil — sequential placement ----
  // Element 0 is 3-node (9 DoFs) padded with node-0's per-field pattern in the trailing slots.
  // Element 1 is 4-node (12 DoFs) and needs no padding.
  {
    DynamicArray<DynamicArray<int>> connectivity;
    connectivity.push_back(DynamicArray<int>{0, 1, 2});
    connectivity.push_back(DynamicArray<int>{3, 4, 5, 6});
    auto map = MakeMap(kNumFields, connectivity);

    map.InitializePaddedIndices(kStride);
    EXPECT_EQ(map.GetPaddedStride(), kStride);

    // clang-format off
    int const expected[] = {
        // Element 0: real DoFs, then node-0's per-field pattern (g0, g1, g2) for the 1 padded node.
        0,  1,  2,  3,  4,  5,  6,  7,  8,  /* pad */ 0,  1,  2,
        // Element 1: fully populated, no padding.
        9, 10, 11, 12, 13, 14, 15, 16, 17,           18, 19, 20};
    // clang-format on
    EXPECT_SPAN_EQ(map.GetPaddedGlobalIndices(), MakeConstSpan(expected));
  }

  // ---- Case 3: variable stride, with stencil — stencil-aware placement ----
  // Element 0's 3 nodes occupy stencil slots [0, 2, 3] of a 4-slot stencil; slot 1 stays as the
  // node-0 padding pattern. Element 1's 4 nodes occupy all stencil slots [0, 1, 2, 3].
  {
    DynamicArray<DynamicArray<int>> connectivity;
    connectivity.push_back(DynamicArray<int>{0, 1, 2});
    connectivity.push_back(DynamicArray<int>{3, 4, 5, 6});
    auto map = MakeMap(kNumFields, connectivity);

    DynamicArray<DynamicArray<int>> nodeStencil;
    nodeStencil.push_back(DynamicArray<int>{0, 2, 3});
    nodeStencil.push_back(DynamicArray<int>{0, 1, 2, 3});
    map.InitializeStencilIndices(GraphFromRangeOfRanges<int, int>(nodeStencil));

    map.InitializePaddedIndices(kStride);

    // clang-format off
    int const expected[] = {
        // Element 0: node0 at slot 0, padding at slot 1, node1 at slot 2, node2 at slot 3.
        0,  1,  2,  /* pad */ 0,  1,  2,  3,  4,  5,  6,  7,  8,
        // Element 1: dense, no padding (stencil = identity).
        9, 10, 11,           12, 13, 14, 15, 16, 17, 18, 19, 20};
    // clang-format on
    EXPECT_SPAN_EQ(map.GetPaddedGlobalIndices(), MakeConstSpan(expected));
  }
}
