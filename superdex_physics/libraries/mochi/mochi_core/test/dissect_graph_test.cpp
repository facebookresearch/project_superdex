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

#include <mochi_core/numbering/dissect_graph.h>
#include <mochi_core/numbering/weighted_graph.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace mochi {
namespace {

/// @brief Helper to build a symmetric weighted graph from an edge list.
///
/// @param n Number of vertices.
/// @param edges Triplets of (u, v, weight). Both (u,v) and (v,u) are inserted,
///              plus a self-loop (v,v) with weight 0 for each vertex.
/// @return A WeightedGraph with the given edges.
WeightedGraph MakeSymmetricGraph(int n, DynamicArray<std::tuple<int, int, int>> const& edges) {
  // Count edges per vertex: one self-loop each, plus two entries per undirected edge.
  DynamicArray<int> pointers(n + 1, 0);
  for (int v = 0; v < n; ++v) {
    pointers[v + 1] = 1; // self-loop
  }
  for (auto const& [u, v, w] : edges) {
    ++pointers[u + 1];
    ++pointers[v + 1];
  }
  std::partial_sum(pointers.begin(), pointers.end(), pointers.begin());

  DynamicArray<WeightedEdge> targets(pointers.back());
  DynamicArray<int> cursor(n);
  for (int v = 0; v < n; ++v) {
    cursor[v] = pointers[v];
    targets[cursor[v]++] = {v, 0}; // self-loop
  }
  for (auto const& [u, v, w] : edges) {
    targets[cursor[u]++] = {v, w};
    targets[cursor[v]++] = {u, w};
  }

  return WeightedGraph{std::move(pointers), std::move(targets)};
}

/// @brief Compute the actual edge cut from the side assignment.
///
/// Counts each crossing edge once (for undirected graphs stored as symmetric directed).
int ComputeEdgeCut(WeightedGraph const& graph, Span<int const> side) {
  int cut = 0;
  for (int v = 0; v < graph.size(); ++v) {
    for (WeightedEdge const& e : graph[v]) {
      if (e.vertex != v && side[v] != side[e.vertex]) {
        cut += e.weight;
      }
    }
  }
  return cut / 2; // Each crossing edge is counted twice in a symmetric graph.
}

/// @brief Verifies the fundamental separator invariant: no direct edge connects a side-0
/// vertex to a side-1 vertex.
bool IsValidSeparator(WeightedGraph const& graph, Span<int const> side) {
  for (int v = 0; v < graph.size(); ++v) {
    if (side[v] != kSide0) {
      continue;
    }
    for (WeightedEdge const& e : graph[v]) {
      if (e.vertex != v && side[e.vertex] == kSide1) {
        return false;
      }
    }
  }
  return true;
}

/// @brief Sum the masses of vertices currently on a given side.
int MassOnSide(Span<int const> side, Span<int const> vertexMass, int whichSide) {
  int total = 0;
  for (int v = 0; v < isize(side); ++v) {
    if (side[v] == whichSide) {
      total += vertexMass[v];
    }
  }
  return total;
}

/// @brief Verifies that a set of separator vertex indices forms a valid separator
/// for the given initial partition.
///
/// @param separatorIndices The vertex indices returned by BuildOptimizedSeparator.
/// @param initialSide The initial partition assignment (kSide0 or kSide1 for each vertex).
/// @param graph The input graph.
/// @return True if the separator is valid (no edge directly connects side-0 to side-1
///         after removing separator vertices), false otherwise.
bool IsValidSeparatorIndices(
    Span<int const> separatorIndices,
    Span<int const> initialSide,
    WeightedGraph const& graph) {
  // Create a copy of the initial side assignment and mark separator vertices.
  DynamicArray<int> side(initialSide.begin(), initialSide.end());
  for (int v : separatorIndices) {
    side[v] = kSeparatorSide;
  }
  return IsValidSeparator(graph, side);
}

/// @brief Checks if a vertex is a boundary vertex in the initial partition
/// (has at least one neighbor on the opposite side).
bool IsBoundaryVertex(int v, Span<int const> side, WeightedGraph const& graph) {
  for (WeightedEdge const& e : graph[v]) {
    if (e.vertex != v && side[e.vertex] != side[v]) {
      return true;
    }
  }
  return false;
}

// ============================================================================
// Tests via DissectGraph (public API), which calls GreedyGraphGrowingPartition.
// ============================================================================

class DissectGraphTest : public ::testing::Test {};

/// Two disconnected components of equal mass. The optimal cut is 0.
TEST_F(DissectGraphTest, TwoComponents) {
  // 4 vertices: {0,1} connected, {2,3} connected, no edges between groups.
  //   0 --1-- 1
  //   2 --1-- 3
  auto graph = MakeSymmetricGraph(4, {{0, 1, 1}, {2, 3, 1}});
  DynamicArray<int> mass{1, 1, 1, 1};
  int totalMass = 4;
  int maxUnbalance = 0;

  auto result = DissectGraph(graph, mass, totalMass, maxUnbalance);

  EXPECT_EQ(isize(result.side), 4);

  // Each side should have total mass 2.
  int mass0 = 0, mass1 = 0;
  for (int v = 0; v < 4; ++v) {
    if (result.side[v] == 0) {
      mass0 += mass[v];
    } else {
      mass1 += mass[v];
    }
  }
  EXPECT_EQ(mass0, 2);
  EXPECT_EQ(mass1, 2);

  // Optimal cut should be 0 (the two components are on different sides).
  EXPECT_EQ(result.cutWeight, 0);
  EXPECT_EQ(ComputeEdgeCut(graph, result.side), result.cutWeight);
  EXPECT_TRUE(result.balanced);
}

/// Path graph: 0 --1-- 1 --1-- 2 --1-- 3.
/// With equal mass, the optimal bisection cuts one edge (cut = 1).
TEST_F(DissectGraphTest, PathGraph) {
  auto graph = MakeSymmetricGraph(4, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}});
  DynamicArray<int> mass{1, 1, 1, 1};
  int totalMass = 4;
  int maxUnbalance = 0;

  auto result = DissectGraph(graph, mass, totalMass, maxUnbalance);

  int mass0 = 0, mass1 = 0;
  for (int v = 0; v < 4; ++v) {
    (result.side[v] == 0 ? mass0 : mass1) += mass[v];
  }
  EXPECT_EQ(mass0, 2);
  EXPECT_EQ(mass1, 2);

  // Best possible cut for a path of 4 is 1 (cut between vertex 1 and 2).
  EXPECT_EQ(result.cutWeight, 1);
  EXPECT_EQ(ComputeEdgeCut(graph, result.side), result.cutWeight);
  EXPECT_TRUE(result.balanced);
}

/// Complete graph K4 with uniform weights. Any balanced partition cuts the same number of edges.
TEST_F(DissectGraphTest, CompleteGraphK4) {
  // K4: all pairs connected with weight 1.
  auto graph =
      MakeSymmetricGraph(4, {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {1, 2, 1}, {1, 3, 1}, {2, 3, 1}});
  DynamicArray<int> mass{1, 1, 1, 1};
  int totalMass = 4;
  int maxUnbalance = 0;

  auto result = DissectGraph(graph, mass, totalMass, maxUnbalance);

  int mass0 = 0, mass1 = 0;
  for (int v = 0; v < 4; ++v) {
    (result.side[v] == 0 ? mass0 : mass1) += mass[v];
  }
  EXPECT_EQ(mass0, 2);
  EXPECT_EQ(mass1, 2);

  // For K4 with 2-2 split, the cut is always 4 (2*2 crossing edges, each weight 1).
  EXPECT_EQ(result.cutWeight, 4);
  EXPECT_EQ(ComputeEdgeCut(graph, result.side), result.cutWeight);
  EXPECT_TRUE(result.balanced);
}

/// Weighted star graph: vertex 0 connected to all others with varying weights.
/// The algorithm should partition to minimize the cut.
TEST_F(DissectGraphTest, WeightedStarGraph) {
  // Star: 0 is the center, connected to 1..4.
  auto graph = MakeSymmetricGraph(5, {{0, 1, 10}, {0, 2, 1}, {0, 3, 1}, {0, 4, 1}});
  DynamicArray<int> mass{1, 1, 1, 1, 1};
  int totalMass = 5;
  int maxUnbalance = 1;

  auto result = DissectGraph(graph, mass, totalMass, maxUnbalance);

  int mass0 = 0, mass1 = 0;
  for (int v = 0; v < 5; ++v) {
    (result.side[v] == 0 ? mass0 : mass1) += mass[v];
  }
  // With maxUnbalance = 1, each side has mass 2 or 3.
  EXPECT_GE(mass0, 2);
  EXPECT_LE(mass0, 3);
  EXPECT_GE(mass1, 2);
  EXPECT_LE(mass1, 3);

  // The heavy edge (weight 10) should not be cut if possible.
  // Best is to keep {0, 1} together and cut the three light edges (cut = 3).
  EXPECT_LE(result.cutWeight, 3);
  EXPECT_EQ(ComputeEdgeCut(graph, result.side), result.cutWeight);
  EXPECT_TRUE(result.balanced);
}

/// Two vertices connected by a single edge.
TEST_F(DissectGraphTest, TwoVertices) {
  auto graph = MakeSymmetricGraph(2, {{0, 1, 5}});
  DynamicArray<int> mass{1, 1};
  int totalMass = 2;
  int maxUnbalance = 0;

  auto result = DissectGraph(graph, mass, totalMass, maxUnbalance);

  EXPECT_EQ(isize(result.side), 2);
  EXPECT_NE(result.side[0], result.side[1]);
  EXPECT_EQ(result.cutWeight, 5);
  EXPECT_EQ(ComputeEdgeCut(graph, result.side), result.cutWeight);
  EXPECT_TRUE(result.balanced);
}

/// Balance constraint: with unequal masses, verify that the partition respects maxUnbalance.
TEST_F(DissectGraphTest, UnequalMassBalance) {
  // 6-vertex path: 0--1--2--3--4--5, all weight 1.
  auto graph = MakeSymmetricGraph(6, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}, {4, 5, 1}});
  DynamicArray<int> mass{3, 1, 1, 1, 1, 3};
  int totalMass = 10;
  int maxUnbalance = 2;

  auto result = DissectGraph(graph, mass, totalMass, maxUnbalance);

  int mass0 = 0, mass1 = 0;
  for (int v = 0; v < 6; ++v) {
    (result.side[v] == 0 ? mass0 : mass1) += mass[v];
  }
  // lowerMass = (10 - 2 + 1)/2 = 4, upperMass = (10 + 2)/2 = 6.
  EXPECT_GE(mass0, 4);
  EXPECT_LE(mass0, 6);
  EXPECT_GE(mass1, 4);
  EXPECT_LE(mass1, 6);
  EXPECT_EQ(ComputeEdgeCut(graph, result.side), result.cutWeight);
  EXPECT_TRUE(result.balanced);
}

/// 2x3 grid graph. Tests a slightly larger structured graph.
///
///   0 --1-- 1 --1-- 2
///   |       |       |
///   1       1       1
///   |       |       |
///   3 --1-- 4 --1-- 5
TEST_F(DissectGraphTest, GridGraph2x3) {
  auto graph = MakeSymmetricGraph(
      6, {{0, 1, 1}, {1, 2, 1}, {3, 4, 1}, {4, 5, 1}, {0, 3, 1}, {1, 4, 1}, {2, 5, 1}});
  DynamicArray<int> mass{1, 1, 1, 1, 1, 1};
  int totalMass = 6;
  int maxUnbalance = 0;

  auto result = DissectGraph(graph, mass, totalMass, maxUnbalance);

  int mass0 = 0, mass1 = 0;
  for (int v = 0; v < 6; ++v) {
    (result.side[v] == 0 ? mass0 : mass1) += mass[v];
  }
  EXPECT_EQ(mass0, 3);
  EXPECT_EQ(mass1, 3);

  // GGGP is a heuristic; optimal bisection cuts 2 edges but 3 is acceptable.
  EXPECT_LE(result.cutWeight, 3);
  EXPECT_EQ(ComputeEdgeCut(graph, result.side), result.cutWeight);
  EXPECT_TRUE(result.balanced);
}

/// Single vertex graph.
TEST_F(DissectGraphTest, SingleVertex) {
  DynamicArray<int> pointers{0, 1};
  DynamicArray<WeightedEdge> targets{{0, 0}};
  WeightedGraph graph{std::move(pointers), std::move(targets)};
  DynamicArray<int> mass{1};

  auto result = DissectGraph(graph, mass, 1, 0);

  EXPECT_EQ(isize(result.side), 1);
  EXPECT_EQ(result.side[0], 0);
  EXPECT_EQ(result.cutWeight, 0);
  EXPECT_TRUE(result.balanced);
}

/// Empty graph.
TEST_F(DissectGraphTest, EmptyGraph) {
  WeightedGraph graph{DynamicArray<int>{0}, DynamicArray<WeightedEdge>{}};
  auto result = DissectGraph(graph, {}, 0, 0);
  EXPECT_EQ(isize(result.side), 0);
  EXPECT_EQ(result.cutWeight, 0);
  EXPECT_TRUE(result.balanced);
}

/// Balance constraint cannot be satisfied: a single vertex's mass exceeds upperMass.
/// The algorithm reports `balanced = false`.
TEST_F(DissectGraphTest, ImpossibleBalance) {
  // Two vertices: one heavy (10), one light (1). totalMass = 11, maxUnbalance = 0.
  // lowerMass = upperMass = ceil(11/2) = 6 / floor(11/2) = 5 — no single-vertex side
  // can satisfy [5, 5] (sides have mass 1 or 10).
  auto graph = MakeSymmetricGraph(2, {{0, 1, 1}});
  DynamicArray<int> mass{10, 1};
  auto result = DissectGraph(graph, mass, 11, 0);

  EXPECT_EQ(isize(result.side), 2);
  EXPECT_FALSE(result.balanced);
}

TEST_F(DissectGraphTest, OptimizeSeparatorPath) {
  // Path: 0 -- 1 -- 2 -- 3 -- 4
  auto graph = MakeSymmetricGraph(5, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}});
  DynamicArray<int> mass{1, 1, 1, 1, 1};

  // Initial thick separator
  DynamicArray<int> side{0, 2, 2, 2, 1};

  // mass0 = 1, mass1 = 1. maxUnbalance = 2.
  int const reportedMass = OptimizeSeparator(side, graph, mass, 2);

  int const mass0 = MassOnSide(side, mass, kSide0);
  int const mass1 = MassOnSide(side, mass, kSide1);
  int const mass2 = MassOnSide(side, mass, kSeparatorSide);

  // The reported separator mass must match the actual sum.
  EXPECT_EQ(reportedMass, mass2);

  // The separator should be thinned down to 1 vertex.
  // E.g., {0, 1} in 0, {2} in 2, {3, 4} in 1.
  // Or {0} in 0, {1} in 2, {2, 3, 4} in 1 (which is valid for maxUnbalance=2).
  EXPECT_EQ(mass2, 1);
  EXPECT_NEAR(mass0, mass1, 2);
  EXPECT_EQ(mass0 + mass1 + mass2, 5);
  EXPECT_TRUE(IsValidSeparator(graph, side));
}

TEST_F(DissectGraphTest, OptimizeSeparatorGrid) {
  // 3x3 Grid
  // 0 -- 1 -- 2
  // |    |    |
  // 3 -- 4 -- 5
  // |    |    |
  // 6 -- 7 -- 8
  auto graph = MakeSymmetricGraph(
      9,
      {{0, 1, 1},
       {1, 2, 1},
       {3, 4, 1},
       {4, 5, 1},
       {6, 7, 1},
       {7, 8, 1},
       {0, 3, 1},
       {3, 6, 1},
       {1, 4, 1},
       {4, 7, 1},
       {2, 5, 1},
       {5, 8, 1}});
  DynamicArray<int> mass(9, 1);

  // Initial thick separator covering the middle row and middle column
  DynamicArray<int> side{0, 2, 1, 2, 2, 2, 0, 2, 1};

  // mass0 = 2, mass1 = 2, maxUnbalance = 1
  int const reportedMass = OptimizeSeparator(side, graph, mass, 1);

  int const mass2 = MassOnSide(side, mass, kSeparatorSide);

  // The reported separator mass must match the actual sum.
  EXPECT_EQ(reportedMass, mass2);

  // Optimal separator for 3x3 grid separating corners is 3 vertices
  // (e.g. the diagonal or middle row).
  // The algorithm is heuristic but should definitely reduce the separator mass from 5.
  EXPECT_LE(mass2, 3);
  EXPECT_TRUE(IsValidSeparator(graph, side));
}

/// Seeds OptimizeSeparator with an unbalanced initial configuration where
/// |mass0 - mass1| > maxUnbalance. The loosened balance check
/// (std::max(maxUnbalance, |mass0 - mass1|)) must allow the optimizer to thin the
/// separator; with the original strict check every candidate move would be rejected
/// and the separator mass would be unchanged.
TEST_F(DissectGraphTest, OptimizeSeparatorThinsFromUnbalancedStart) {
  // Path: 0--1--2--3--4--5--6--7--8.
  auto graph = MakeSymmetricGraph(
      9, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}, {4, 5, 1}, {5, 6, 1}, {6, 7, 1}, {7, 8, 1}});
  DynamicArray<int> mass(9, 1);

  // Initial: mass0 = 5 (vertices 0..4), mass1 = 1 (vertex 8),
  // separator mass = 3 (vertices 5, 6, 7).
  DynamicArray<int> side{
      kSide0,
      kSide0,
      kSide0,
      kSide0,
      kSide0,
      kSeparatorSide,
      kSeparatorSide,
      kSeparatorSide,
      kSide1};
  ASSERT_TRUE(IsValidSeparator(graph, side));

  int const initialSeparatorMass = MassOnSide(side, mass, kSeparatorSide);
  ASSERT_EQ(initialSeparatorMass, 3);
  ASSERT_EQ(MassOnSide(side, mass, kSide0), 5);
  ASSERT_EQ(MassOnSide(side, mass, kSide1), 1);

  // |mass0 - mass1| = 4, but maxUnbalance is only 1. The strict check would
  // reject every move; the loosened check must allow improvement.
  int const reportedMass = OptimizeSeparator(side, graph, mass, /*maxUnbalance=*/1);

  int const finalSeparatorMass = MassOnSide(side, mass, kSeparatorSide);
  EXPECT_EQ(reportedMass, finalSeparatorMass);
  EXPECT_LT(finalSeparatorMass, initialSeparatorMass);
  EXPECT_TRUE(IsValidSeparator(graph, side));
}

// ============================================================================
// Direct tests for MarkTrivialSeparatorBothSides.
// ============================================================================

/// Path 0-1-2-3-4 with initial bisection {0,0,0,1,1}: vertex 2 sits on the boundary and
/// is promoted. Vertex 3's only opposite-side neighbor (vertex 2) is now in the separator,
/// so it stays on side 1. This guards against the cascading-promotion bug that would also
/// promote vertex 4.
TEST_F(DissectGraphTest, MarkTrivialSeparatorBothSidesPathBoundaryOnly) {
  auto graph = MakeSymmetricGraph(5, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}});
  DynamicArray<int> side{kSide0, kSide0, kSide0, kSide1, kSide1};

  MarkTrivialSeparatorBothSides(side, graph);

  DynamicArray<int> const expected{kSide0, kSide0, kSeparatorSide, kSide1, kSide1};
  EXPECT_EQ(side, expected);
  EXPECT_TRUE(IsValidSeparator(graph, side));
}

/// A graph where both sides have independent boundary vertices: BothSides promotes vertices
/// from both sides, unlike OneSide which promotes only from one specified side.
///
///   3-vertex path: 0 -- 1 -- 2  with initial bisection {1, 0, 1}.
///   Both vertex 0 (side 1) and vertex 1 (side 0) are boundary vertices and are promoted.
TEST_F(DissectGraphTest, MarkTrivialSeparatorBothSidesPromotesBothSides) {
  auto graph = MakeSymmetricGraph(3, {{0, 1, 1}, {1, 2, 1}});
  DynamicArray<int> side{kSide1, kSide0, kSide1};

  MarkTrivialSeparatorBothSides(side, graph);

  // Vertex 0 sees neighbor 1 on the opposite side: promoted.
  // Vertex 1 sees neighbor 0 (now sep, skip) and neighbor 2 on the opposite side: promoted.
  // Vertex 2 sees neighbor 1 (now sep, skip): not promoted.
  DynamicArray<int> const expected{kSeparatorSide, kSeparatorSide, kSide1};
  EXPECT_EQ(side, expected);
  EXPECT_TRUE(IsValidSeparator(graph, side));
}

/// On a graph with no cross-side edges, the function must not promote any vertex.
TEST_F(DissectGraphTest, MarkTrivialSeparatorBothSidesDisconnected) {
  // {0,1} on side 0, {2,3} on side 1, no edges between groups.
  auto graph = MakeSymmetricGraph(4, {{0, 1, 1}, {2, 3, 1}});
  DynamicArray<int> side{kSide0, kSide0, kSide1, kSide1};

  MarkTrivialSeparatorBothSides(side, graph);

  DynamicArray<int> const expected{kSide0, kSide0, kSide1, kSide1};
  EXPECT_EQ(side, expected);
}

// ============================================================================
// Direct tests for MarkTrivialSeparatorOneSide.
// ============================================================================

/// On the 0-1-2-3-4 path with bisection {0,0,0,1,1}, picking from side 0 promotes only
/// vertex 2 (its sole side-1 neighbor is 3); vertex 3 stays on side 1. The reported mass
/// must equal the actual mass of vertices marked separator.
TEST_F(DissectGraphTest, MarkTrivialSeparatorOneSidePromotesSide0Boundary) {
  auto graph = MakeSymmetricGraph(5, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}});
  DynamicArray<int> mass{1, 1, 2, 1, 1};
  DynamicArray<int> side{kSide0, kSide0, kSide0, kSide1, kSide1};

  int const separatorMass = MarkTrivialSeparatorOneSide(side, graph, mass, kSide0);

  DynamicArray<int> const expected{kSide0, kSide0, kSeparatorSide, kSide1, kSide1};
  EXPECT_EQ(side, expected);
  EXPECT_EQ(separatorMass, MassOnSide(side, mass, kSeparatorSide));
  EXPECT_EQ(separatorMass, mass[2]);
  EXPECT_TRUE(IsValidSeparator(graph, side));
}

/// Picking from side 1 instead promotes vertex 3.
TEST_F(DissectGraphTest, MarkTrivialSeparatorOneSidePromotesSide1Boundary) {
  auto graph = MakeSymmetricGraph(5, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}});
  DynamicArray<int> mass{1, 1, 1, 4, 1};
  DynamicArray<int> side{kSide0, kSide0, kSide0, kSide1, kSide1};

  int const separatorMass = MarkTrivialSeparatorOneSide(side, graph, mass, kSide1);

  DynamicArray<int> const expected{kSide0, kSide0, kSide0, kSeparatorSide, kSide1};
  EXPECT_EQ(side, expected);
  EXPECT_EQ(separatorMass, MassOnSide(side, mass, kSeparatorSide));
  EXPECT_EQ(separatorMass, mass[3]);
  EXPECT_TRUE(IsValidSeparator(graph, side));
}

// ============================================================================
// Tests for BuildOptimizedSeparator.
// ============================================================================

/// On a path graph, the optimal separator is a single vertex. The returned indices must
/// match the vertices marked as separator after optimization.
TEST_F(DissectGraphTest, BuildOptimizedSeparatorPath) {
  auto graph = MakeSymmetricGraph(5, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}});
  DynamicArray<int> mass{1, 1, 1, 1, 1};
  DynamicArray<int> side{kSide0, kSide0, kSide0, kSide1, kSide1};

  DynamicArray<int> const separator = BuildOptimizedSeparator(side, graph, mass, 2);

  // The optimal separator on this path is exactly one vertex.
  EXPECT_EQ(isize(separator), 1);

  // Verify that the returned indices form a valid separator.
  EXPECT_TRUE(IsValidSeparatorIndices(separator, side, graph));

  // Verify that each returned index is a boundary vertex in the initial partition
  // (has at least one neighbor on the opposite side).
  for (int v : separator) {
    EXPECT_TRUE(IsBoundaryVertex(v, side, graph))
        << "Vertex " << v << " is not a boundary vertex in the initial partition";
  }
}

/// 3x3 grid: middle row {3,4,5} is one of several minimum separators.
/// Using a thick initial bisection forces the optimizer to engage; the result must be
/// a small valid separator.
TEST_F(DissectGraphTest, BuildOptimizedSeparatorGrid) {
  // 3x3 grid (same topology as OptimizeSeparatorGrid).
  auto graph = MakeSymmetricGraph(
      9,
      {{0, 1, 1},
       {1, 2, 1},
       {3, 4, 1},
       {4, 5, 1},
       {6, 7, 1},
       {7, 8, 1},
       {0, 3, 1},
       {3, 6, 1},
       {1, 4, 1},
       {4, 7, 1},
       {2, 5, 1},
       {5, 8, 1}});
  DynamicArray<int> mass(9, 1);
  // Top row on side 0, bottom row on side 1, middle row split.
  DynamicArray<int> side{kSide0, kSide0, kSide0, kSide0, kSide1, kSide1, kSide1, kSide1, kSide1};

  DynamicArray<int> const separator = BuildOptimizedSeparator(side, graph, mass, 2);

  // Verify the separator size is reasonable.
  EXPECT_LE(isize(separator), 3);
  EXPECT_GE(isize(separator), 1);

  // Verify that the returned indices form a valid separator.
  EXPECT_TRUE(IsValidSeparatorIndices(separator, side, graph));

  // Verify that each returned index is a boundary vertex in the initial partition
  // (has at least one neighbor on the opposite side).
  for (int v : separator) {
    EXPECT_TRUE(IsBoundaryVertex(v, side, graph))
        << "Vertex " << v << " is not a boundary vertex in the initial partition";
  }
}

/// Verify that BuildOptimizedSeparator selects the better of the three trivial-separator
/// strategies. Use a 6-vertex "barbell" with a very heavy side-1 vertex on the boundary:
/// strategy B (promote only side-0 boundary, mass 1) wins over A (both sides) and
/// C (promote heavy side-1 boundary, mass 100).
TEST_F(DissectGraphTest, BuildOptimizedSeparatorPicksMinStrategy) {
  // 0-1-2 (side 0) -- 3-4-5 (side 1), with edge 2-3 between boundaries.
  auto graph = MakeSymmetricGraph(6, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}, {4, 5, 1}});
  // Vertex 3 is heavy: separator on side 1 would cost 100, separator on side 0 costs 1.
  DynamicArray<int> mass{1, 1, 1, 100, 1, 1};
  DynamicArray<int> side{kSide0, kSide0, kSide0, kSide1, kSide1, kSide1};

  DynamicArray<int> const separator = BuildOptimizedSeparator(side, graph, mass, 100);

  // The minimum separator mass here is 1 (vertex 2). Verify the heavy vertex is not picked.
  ASSERT_EQ(isize(separator), 1);
  EXPECT_EQ(separator[0], 2);

  // Verify that the returned indices form a valid separator.
  EXPECT_TRUE(IsValidSeparatorIndices(separator, side, graph));

  // Verify that each returned index is a boundary vertex in the initial partition
  // (has at least one neighbor on the opposite side).
  for (int v : separator) {
    EXPECT_TRUE(IsBoundaryVertex(v, side, graph))
        << "Vertex " << v << " is not a boundary vertex in the initial partition";
  }
}

// ============================================================================
// Trisect tests
// ============================================================================

/// @brief Builds an n x n weighted grid graph with unit edge weights.
WeightedGraph MakeUnitWeightGrid(int n) {
  DynamicArray<std::tuple<int, int, int>> edges;
  edges.reserve(2 * n * (n - 1));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      int const v = i + j * n;
      if (i + 1 < n) {
        edges.push_back({v, v + 1, 1});
      }
      if (j + 1 < n) {
        edges.push_back({v, v + n, 1});
      }
    }
  }
  return MakeSymmetricGraph(n * n, edges);
}

/// @brief Validates that @p side and @p separator together form a consistent trisection
/// of @p graph: the separator is exactly {v : side[v] == 2}, sides 0 and 1 are mass-balanced
/// within @p maxUnbalance, and no edge directly connects sides 0 and 1.
void ExpectValidTrisection(
    WeightedGraph const& graph,
    Span<int const> mass,
    Span<int const> side,
    Span<int const> separator,
    int maxUnbalance) {
  int const n = graph.size();
  ASSERT_EQ(isize(side), n);

  // Each side index is 0, 1, or 2.
  for (int v = 0; v < n; ++v) {
    EXPECT_GE(side[v], 0);
    EXPECT_LE(side[v], kSeparatorSide);
  }

  // Separator matches the vertices with side == 2 (in order).
  DynamicArray<int> expectedSeparator;
  for (int v = 0; v < n; ++v) {
    if (side[v] == kSeparatorSide) {
      expectedSeparator.push_back(v);
    }
  }
  EXPECT_SPAN_EQ(MakeConstSpan(expectedSeparator), separator);

  // No direct edge between sides 0 and 1.
  for (int v = 0; v < n; ++v) {
    for (WeightedEdge const& e : graph[v]) {
      if (e.vertex == v) {
        continue;
      }
      EXPECT_FALSE(
          (side[v] == kSide0 && side[e.vertex] == kSide1) ||
          (side[v] == kSide1 && side[e.vertex] == kSide0))
          << "Direct edge between sides 0 and 1 at v=" << v << ", u=" << e.vertex;
    }
  }

  // Sides 0 and 1 are mass-balanced within maxUnbalance.
  int mass0 = 0;
  int mass1 = 0;
  for (int v = 0; v < n; ++v) {
    if (side[v] == kSide0) {
      mass0 += mass[v];
    } else if (side[v] == kSide1) {
      mass1 += mass[v];
    }
  }
  EXPECT_NEAR(mass0, mass1, maxUnbalance);
}

/// Trisection on a small graph: no coarsening triggered (graph already ≤ 100 vertices).
TEST_F(DissectGraphTest, TrisectSmallPath) {
  // Path 0 -- 1 -- 2 -- 3 -- 4
  auto graph = MakeSymmetricGraph(5, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}});
  DynamicArray<int> mass{1, 1, 1, 1, 1};

  auto result = Trisect(graph, mass, /*maxUnbalance=*/2);

  ExpectValidTrisection(
      graph, mass, MakeConstSpan(result.side), MakeConstSpan(result.separator), 2);

  // For a 5-vertex path the minimum separator is a single vertex.
  EXPECT_EQ(isize(result.separator), 1);
}

/// Trisection on a graph large enough to require multiple coarsening levels.
/// This exercises Phase 3 of the algorithm (recursive projection with periodic optimization).
TEST_F(DissectGraphTest, TrisectLargeGridTriggersCoarsening) {
  int constexpr kSide = 24; // 576 vertices > 100, requires multiple coarsening levels.
  auto graph = MakeUnitWeightGrid(kSide);
  int const n = kSide * kSide;
  DynamicArray<int> mass(n, 1);

  // The multilevel algorithm is heuristic; we use a generous maxUnbalance because the
  // constraint is enforced only at the coarsest level (where vertex masses are non-unit)
  // and via local moves during refinement.
  int constexpr kMaxUnbalance = n / 4;
  auto result = Trisect(graph, mass, kMaxUnbalance);

  ExpectValidTrisection(
      graph, mass, MakeConstSpan(result.side), MakeConstSpan(result.separator), kMaxUnbalance);

  // For a square n x n grid the minimum balanced vertex separator is a single
  // row or column of length kSide. The multilevel heuristic should produce a
  // result close to this optimum; allow a 50% margin on the upper side to
  // absorb heuristic variation. The lower bound is also tied to kSide: any
  // separator that meets the balance constraint here must isolate at least
  // (n - maxUnbalance - separator) / 2 vertices on each side, which for a
  // square grid forces the separator to span roughly one grid dimension.
  EXPECT_LE(isize(result.separator), 3 * kSide / 2);
  EXPECT_GE(isize(result.separator), kSide);
}

/// Single-vertex graph: trivially trisected with no separator.
TEST_F(DissectGraphTest, TrisectSingleVertex) {
  DynamicArray<int> pointers{0, 1};
  DynamicArray<WeightedEdge> targets{{0, 0}};
  WeightedGraph graph{std::move(pointers), std::move(targets)};
  DynamicArray<int> mass{1};

  auto result = Trisect(graph, mass, 0);

  ExpectValidTrisection(
      graph, mass, MakeConstSpan(result.side), MakeConstSpan(result.separator), 1);
}

/// Empty graph: trisection of nothing is empty.
TEST_F(DissectGraphTest, TrisectEmptyGraph) {
  WeightedGraph graph{DynamicArray<int>{0}, DynamicArray<WeightedEdge>{}};
  auto result = Trisect(graph, {}, 0);

  EXPECT_TRUE(result.side.empty());
  EXPECT_TRUE(result.separator.empty());
}

// ============================================================================
// ExtractSideSubgraphs tests
// ============================================================================

/// @brief Validates the structural correctness of one side subgraph extracted from a
/// trisection: oldVertex picks exactly the vertices with side == whichSide, vertexMass is
/// carried over correctly, and edges match the original graph (only edges within the side,
/// with target indices remapped, edge weights preserved, self-loops kept).
void ExpectValidSideSubgraph(
    SideSubgraph const& sub,
    WeightedGraph const& fullGraph,
    Span<int const> fullMass,
    Span<int const> side,
    int whichSide) {
  // oldVertex matches the vertices on the requested side, in increasing index order.
  DynamicArray<int> expectedOld;
  for (int v = 0; v < fullGraph.size(); ++v) {
    if (side[v] == whichSide) {
      expectedOld.push_back(v);
    }
  }
  EXPECT_SPAN_EQ(MakeConstSpan(expectedOld), MakeConstSpan(sub.oldVertex));
  EXPECT_EQ(sub.graph.size(), isize(expectedOld));
  ASSERT_EQ(isize(sub.vertexMass), isize(expectedOld));

  // vertexMass is carried over from the original mass at the original vertex index.
  for (int newV = 0; newV < isize(expectedOld); ++newV) {
    EXPECT_EQ(sub.vertexMass[newV], fullMass[sub.oldVertex[newV]]);
  }

  // Build expected edges in the new ordering by filtering the original adjacency.
  DynamicArray<int> oldToNew(fullGraph.size(), -1);
  for (int newV = 0; newV < isize(sub.oldVertex); ++newV) {
    oldToNew[sub.oldVertex[newV]] = newV;
  }
  for (int newV = 0; newV < sub.graph.size(); ++newV) {
    int const oldV = sub.oldVertex[newV];
    DynamicArray<std::pair<int, int>> expectedEdges;
    for (WeightedEdge const& e : fullGraph[oldV]) {
      if (side[e.vertex] == whichSide) {
        expectedEdges.push_back({oldToNew[e.vertex], e.weight});
      }
    }
    DynamicArray<std::pair<int, int>> actualEdges;
    for (WeightedEdge const& e : sub.graph[newV]) {
      actualEdges.push_back({e.vertex, e.weight});
    }
    EXPECT_EQ(actualEdges, expectedEdges) << "Mismatched edges at newV=" << newV;
  }
}

/// Path graph 0--1--2--3--4 with separator at vertex 2: side 0 = {0,1}, side 1 = {3,4}.
TEST_F(DissectGraphTest, ExtractSideSubgraphsPath) {
  auto graph = MakeSymmetricGraph(5, {{0, 1, 1}, {1, 2, 2}, {2, 3, 3}, {3, 4, 4}});
  DynamicArray<int> mass{10, 20, 30, 40, 50};
  DynamicArray<int> side{0, 0, 2, 1, 1};

  auto subs = ExtractSideSubgraphs(graph, mass, side);

  ExpectValidSideSubgraph(subs[0], graph, mass, side, 0);
  ExpectValidSideSubgraph(subs[1], graph, mass, side, 1);
}

/// One side empty: when side 1 has no vertices, the side-1 subgraph is empty and the side-0
/// subgraph still extracts cleanly.
TEST_F(DissectGraphTest, ExtractSideSubgraphsEmptySide) {
  auto graph = MakeSymmetricGraph(3, {{0, 1, 1}, {1, 2, 1}});
  DynamicArray<int> mass{1, 1, 1};
  DynamicArray<int> side{0, 0, 2}; // No vertex on side 1.

  auto subs = ExtractSideSubgraphs(graph, mass, side);

  ExpectValidSideSubgraph(subs[0], graph, mass, side, 0);
  ExpectValidSideSubgraph(subs[1], graph, mass, side, 1);

  EXPECT_TRUE(subs[1].vertexMass.empty());
  EXPECT_TRUE(subs[1].oldVertex.empty());
}

/// End-to-end: extract side subgraphs from an actual Trisect result on a larger graph,
/// and validate structural consistency.
TEST_F(DissectGraphTest, ExtractSideSubgraphsAfterTrisect) {
  auto graph = MakeUnitWeightGrid(8); // 64 vertices, exercises multilevel coarsening lightly.
  int const n = graph.size();
  DynamicArray<int> mass(n, 1);

  auto trisection = Trisect(graph, mass, /*maxUnbalance=*/n / 4);
  auto subs = ExtractSideSubgraphs(graph, mass, trisection.side);

  ExpectValidSideSubgraph(subs[0], graph, mass, trisection.side, 0);
  ExpectValidSideSubgraph(subs[1], graph, mass, trisection.side, 1);

  // Together the two sides plus the separator account for every vertex exactly once.
  EXPECT_EQ(subs[0].graph.size() + subs[1].graph.size() + isize(trisection.separator), n);
}

// ============================================================================
// RenumberByDissection tests
// ============================================================================

/// @brief Asserts that @p perm is a permutation of [0, n).
void ExpectValidPermutation(Span<int const> perm, int n) {
  ASSERT_EQ(isize(perm), n);
  DynamicArray<int> seen(n, 0);
  for (int p : perm) {
    ASSERT_GE(p, 0);
    ASSERT_LT(p, n);
    EXPECT_EQ(seen[p], 0) << "Vertex " << p << " appears twice in perm";
    seen[p] = 1;
  }
}

/// @brief Recursively validates the structural invariants of a nested-dissection
/// ordering: at every non-leaf level the trailing positions of @p perm hold exactly the
/// separator vertices produced by Trisect on that subgraph, and the two sides occupy
/// contiguous prefixes whose entries map (through @c oldVertex) into the corresponding
/// side subgraph. Calls @ref Trisect with the same per-level @c levelUnbalance scaling as
/// @ref RenumberByDissection so the comparison is meaningful.
void ExpectValidNestedDissectStructure(
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    Span<int const> perm,
    int topMaxUnbalance,
    int topMass,
    int leafSize) {
  int const n = graph.size();
  ASSERT_EQ(isize(perm), n);
  if (n <= 1 || n < leafSize) {
    // Leaf: identity permutation.
    DynamicArray<int> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_SPAN_EQ(MakeConstSpan(expected), perm);
    return;
  }

  // Recompute the trisection that RenumberByDissection used at this level.
  int const subMass = std::accumulate(vertexMass.begin(), vertexMass.end(), 0);
  int const levelUnbalance =
      static_cast<int>(static_cast<int64_t>(topMaxUnbalance) * subMass / topMass);
  auto trisection = Trisect(graph, vertexMass, levelUnbalance);
  auto subs = ExtractSideSubgraphs(graph, vertexMass, trisection.side);
  int const numSep = isize(trisection.separator);
  int const n0 = subs[0].graph.size();
  int const n1 = subs[1].graph.size();
  ASSERT_EQ(n0 + n1 + numSep, n);

  // Trailing numSep entries must be exactly the separator vertices (as a set).
  DynamicArray<int> tail(perm.end() - numSep, perm.end());
  std::ranges::sort(tail);
  DynamicArray<int> expectedSep(trisection.separator.begin(), trisection.separator.end());
  std::ranges::sort(expectedSep);
  EXPECT_SPAN_EQ(MakeConstSpan(expectedSep), MakeConstSpan(tail));

  // Map each side's prefix back into the subgraph's local index space and recurse.
  DynamicArray<int> oldToNew(n, -1);
  for (int s = 0; s < 2; ++s) {
    std::ranges::fill(oldToNew, -1);
    for (int newIdx = 0; newIdx < isize(subs[s].oldVertex); ++newIdx) {
      oldToNew[subs[s].oldVertex[newIdx]] = newIdx;
    }
    int const offset = (s == 0) ? 0 : n0;
    int const subN = (s == 0) ? n0 : n1;
    DynamicArray<int> subPerm(subN);
    for (int i = 0; i < subN; ++i) {
      int const orig = perm[offset + i];
      ASSERT_GE(orig, 0);
      ASSERT_LT(orig, n);
      EXPECT_NE(oldToNew[orig], -1)
          << "perm[" << (offset + i) << "]=" << orig << " is not on side " << s;
      subPerm[i] = oldToNew[orig];
    }
    ExpectValidNestedDissectStructure(
        subs[s].graph,
        MakeConstSpan(subs[s].vertexMass),
        MakeConstSpan(subPerm),
        topMaxUnbalance,
        topMass,
        leafSize);
  }
}

/// Small graph below the leaf threshold: identity permutation.
TEST_F(DissectGraphTest, RenumberByDissectionSmallGraphIdentity) {
  auto graph = MakeSymmetricGraph(5, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}});
  DynamicArray<int> mass{1, 1, 1, 1, 1};

  // leafSize=10 > n=5, so the function should return identity without trisecting.
  auto perm = RenumberByDissection(graph, mass, /*maxUnbalance=*/2, /*leafSize=*/10);

  DynamicArray<int> expected{0, 1, 2, 3, 4};
  EXPECT_SPAN_EQ(MakeConstSpan(expected), MakeConstSpan(perm));
}

/// Single-trisection: graph just above leafSize so exactly one Trisect call happens. Verifies
/// permutation validity and that the separator vertices land at the very end.
TEST_F(DissectGraphTest, RenumberByDissectionOneLevel) {
  // 6x6 grid = 36 vertices. With leafSize=20, the top trisection runs but the resulting
  // sides (each ~17 vertices) fall below leafSize, so no further recursion happens.
  int constexpr kSide = 6;
  auto graph = MakeUnitWeightGrid(kSide);
  int const n = kSide * kSide;
  DynamicArray<int> mass(n, 1);

  int constexpr kLeafSize = 20;
  auto perm = RenumberByDissection(graph, mass, /*maxUnbalance=*/n / 4, /*leafSize=*/kLeafSize);
  ExpectValidPermutation(MakeConstSpan(perm), n);

  // Recompute the same trisection to identify which vertices are separator.
  auto trisection = Trisect(graph, mass, n / 4);
  int const numSep = isize(trisection.separator);

  // The last numSep entries of perm must be exactly the separator vertices (as a set).
  DynamicArray<int> tail(perm.end() - numSep, perm.end());
  std::ranges::sort(tail);
  DynamicArray<int> expectedSep(trisection.separator.begin(), trisection.separator.end());
  std::ranges::sort(expectedSep);
  EXPECT_SPAN_EQ(MakeConstSpan(expectedSep), MakeConstSpan(tail));
}

/// Multi-level recursion on a larger grid: verifies the result is a valid permutation
/// AND that at every recursion level the separator vertices land at the trailing
/// positions of their subrange. Without the structural check, an identity permutation
/// that skipped recursion entirely would still pass.
TEST_F(DissectGraphTest, RenumberByDissectionMultiLevel) {
  int constexpr kSide = 16;
  auto graph = MakeUnitWeightGrid(kSide);
  int const n = kSide * kSide;
  DynamicArray<int> mass(n, 1);

  int constexpr kMaxUnbalance = (kSide * kSide) / 4;
  int constexpr kLeafSize = 40;
  auto perm = RenumberByDissection(graph, mass, kMaxUnbalance, kLeafSize);
  ExpectValidPermutation(MakeConstSpan(perm), n);

  int const totalMass = std::accumulate(mass.begin(), mass.end(), 0);
  ExpectValidNestedDissectStructure(
      graph, MakeConstSpan(mass), MakeConstSpan(perm), kMaxUnbalance, totalMass, kLeafSize);
}

/// Non-uniform vertex masses exercise the per-level scaling
/// `levelUnbalance = topMaxUnbalance * subMass / topMass`. With uniform mass and
/// roughly equal sides, the scaling collapses to ~1/2 at each level and bugs in the
/// int64 cast or integer truncation would not be caught.
TEST_F(DissectGraphTest, RenumberByDissectionNonUniformMass) {
  int constexpr kSide = 10;
  auto graph = MakeUnitWeightGrid(kSide);
  int const n = kSide * kSide;

  // Mostly light vertices with a handful of heavy ones scattered through the graph.
  DynamicArray<int> mass(n, 1);
  mass[0] = 50;
  mass[n - 1] = 50;
  mass[n / 2] = 30;
  mass[n / 4] = 20;

  int const totalMass = std::accumulate(mass.begin(), mass.end(), 0);
  int const maxUnbalance = totalMass / 4;
  int constexpr kLeafSize = 15;
  auto perm = RenumberByDissection(graph, mass, maxUnbalance, kLeafSize);

  ExpectValidPermutation(MakeConstSpan(perm), n);
  ExpectValidNestedDissectStructure(
      graph, MakeConstSpan(mass), MakeConstSpan(perm), maxUnbalance, totalMass, kLeafSize);
}

/// Non-empty graph with all-zero vertex masses: exercises the `totalMass == 0` early
/// return on a non-empty graph (the empty-graph case is covered separately). A future
/// refactor that dropped this guard would silently divide by zero in the recursive impl.
TEST_F(DissectGraphTest, RenumberByDissectionAllZeroMassNonEmpty) {
  auto graph = MakeSymmetricGraph(4, {{0, 1, 1}, {2, 3, 1}});
  DynamicArray<int> const mass{0, 0, 0, 0};

  auto perm = RenumberByDissection(graph, mass, /*maxUnbalance=*/0, /*leafSize=*/2);

  // The zero-mass early return produces the identity permutation regardless of leafSize.
  DynamicArray<int> const expected{0, 1, 2, 3};
  EXPECT_SPAN_EQ(MakeConstSpan(expected), MakeConstSpan(perm));
}

/// Empty graph: empty permutation.
TEST_F(DissectGraphTest, RenumberByDissectionEmptyGraph) {
  WeightedGraph graph{DynamicArray<int>{0}, DynamicArray<WeightedEdge>{}};
  auto perm = RenumberByDissection(graph, {}, /*maxUnbalance=*/0, /*leafSize=*/100);
  EXPECT_TRUE(perm.empty());
}

} // namespace
} // namespace mochi
