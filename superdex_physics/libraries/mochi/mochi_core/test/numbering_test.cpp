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

#include <mochi_core/numbering/graph_coarsening.h>
#include <mochi_core/numbering/weighted_graph.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/graph_utils.h>

#include <gtest/gtest.h>

using namespace mochi;

namespace {

/// @brief Creates a star graph: vertex 0 connected to every other vertex.
/// @param n Total number of vertices (must be >= 1).
/// @param edgeWeight Weight for each edge.
WeightedGraph MakeStarGraph(int n, int edgeWeight) {
  GraphBuilder<WeightedEdge, int> builder(n, 2 * (n - 1));
  // Center vertex 0 connects to every other vertex.
  builder.StartSet();
  for (int i = 1; i < n; ++i) {
    builder.InsertTarget({.vertex = i, .weight = edgeWeight});
  }
  // Each leaf connects back to the center.
  for (int i = 1; i < n; ++i) {
    builder.StartSet();
    builder.InsertTarget({.vertex = 0, .weight = edgeWeight});
  }
  return builder.Build();
}

/// @brief Creates a 2D grid weighted graph with optional self-loops.
/// @param nx Number of grid columns.
/// @param ny Number of grid rows.
/// @param edgeWeight Weight for neighbor edges.
/// @param selfLoopWeight Weight for self-loops (0 to omit self-loops).
WeightedGraph MakeGridGraph(int nx, int ny, int edgeWeight, int selfLoopWeight = 0) {
  int const n = nx * ny;
  GraphBuilder<WeightedEdge, int> builder(n, 5 * n);
  for (int jy = 0; jy < ny; ++jy) {
    for (int ix = 0; ix < nx; ++ix) {
      int node = ix + jy * nx;
      builder.StartSet();
      if (selfLoopWeight != 0) {
        builder.InsertTarget({.vertex = node, .weight = selfLoopWeight});
      }
      if (jy > 0) {
        builder.InsertTarget({.vertex = node - nx, .weight = edgeWeight});
      }
      if (ix > 0) {
        builder.InsertTarget({.vertex = node - 1, .weight = edgeWeight});
      }
      if (ix + 1 < nx) {
        builder.InsertTarget({.vertex = node + 1, .weight = edgeWeight});
      }
      if (jy + 1 < ny) {
        builder.InsertTarget({.vertex = node + nx, .weight = edgeWeight});
      }
    }
  }
  return builder.Build();
}

} // namespace

// =============================================================================
// WeightedGraph Tests
// =============================================================================

TEST(WeightedGraph, MakeWeightedGraph_UniformWeight) {
  // Create a simple triangle graph: 0 -> 1 -> 2 -> 0
  DynamicArray<int> pointers = {0, 1, 2, 3};
  DynamicArray<int> targets = {1, 2, 0};
  Graph<int, int> graph(pointers, targets);

  int const weight = 5;
  auto weighted = MakeWeightedGraph(graph, weight);

  EXPECT_EQ(3, weighted.size());
  EXPECT_EQ(3, weighted.NumTargets());

  // Check that all edges have the uniform weight
  for (auto const m : weighted) {
    for (auto const& edge : m.targets) {
      EXPECT_EQ(weight, edge.weight);
    }
  }

  // Check that the structure is preserved
  EXPECT_EQ(1, weighted[0][0].vertex);
  EXPECT_EQ(2, weighted[1][0].vertex);
  EXPECT_EQ(0, weighted[2][0].vertex);
}

// =============================================================================
// HeavyEdgeMatch Tests
// =============================================================================

TEST(HeavyEdgeMatch, SimpleGraph) {
  // Create a simple path graph: 0 -- 1 -- 2 -- 3
  // With weights: 0-1: 10, 1-2: 5, 2-3: 10
  // Expected: vertices 0,1 match and 2,3 match (heavy edges)
  DynamicArray<int> pointers = {0, 1, 3, 5, 6};
  DynamicArray<WeightedEdge> targets = {
      {.vertex = 1, .weight = 10}, // 0 -> 1
      {.vertex = 0, .weight = 10}, // 1 -> 0
      {.vertex = 2, .weight = 5}, // 1 -> 2
      {.vertex = 1, .weight = 5}, // 2 -> 1
      {.vertex = 3, .weight = 10}, // 2 -> 3
      {.vertex = 2, .weight = 10}, // 3 -> 2
  };
  WeightedGraph graph(pointers, targets);
  DynamicArray<int> vertexMass = {1, 1, 1, 1};

  auto result = HeavyEdgeMatch(graph, vertexMass);

  // Should have 2 coarse vertices (two pairs).
  EXPECT_EQ(2, result.match.size());

  // Verify heavy edges were selected: vertices 0,1 should be matched together,
  // and vertices 2,3 should be matched together.
  EXPECT_EQ(result.matchedIn[0], result.matchedIn[1])
      << "Vertices 0 and 1 should be in the same match group (heavy edge weight 10)";
  EXPECT_EQ(result.matchedIn[2], result.matchedIn[3])
      << "Vertices 2 and 3 should be in the same match group (heavy edge weight 10)";
  EXPECT_NE(result.matchedIn[0], result.matchedIn[2])
      << "Groups {0,1} and {2,3} should be different";
}

TEST(HeavyEdgeMatch, AllVerticesMatched) {
  // Create a 3x3 grid graph (9 vertices)
  int const nx = 3, ny = 3;
  int const n = nx * ny;
  auto graph = MakeGridGraph(nx, ny, 1);
  DynamicArray<int> vertexMass(n, 1);

  auto result = HeavyEdgeMatch(graph, vertexMass);

  // Each vertex should be in a valid match group.
  for (int v = 0; v < n; ++v) {
    EXPECT_GE(result.matchedIn[v], 0) << "Vertex " << v << " should be in a valid match group";
    EXPECT_LT(result.matchedIn[v], result.match.size())
        << "Match group of vertex " << v << " should be valid";
  }

  // Verify that vertices in the same match group are actually matched.
  for (int g = 0; g < result.match.size(); ++g) {
    auto const& group = result.match[g];
    EXPECT_GE(group.size(), 1) << "Match group " << g << " should have at least 1 vertex";
    EXPECT_LE(group.size(), 2) << "Match group " << g << " should have at most 2 vertices";
    for (int v : group) {
      EXPECT_EQ(result.matchedIn[v], g) << "Vertex " << v << " should be in group " << g;
    }
  }
}

TEST(HeavyEdgeMatch, MatchedInConsistency) {
  // Create a simple graph
  DynamicArray<int> pointers = {0, 2, 4, 6, 8};
  DynamicArray<WeightedEdge> targets = {
      {.vertex = 1, .weight = 1},
      {.vertex = 2, .weight = 1},
      {.vertex = 0, .weight = 1},
      {.vertex = 3, .weight = 1},
      {.vertex = 0, .weight = 1},
      {.vertex = 3, .weight = 1},
      {.vertex = 1, .weight = 1},
      {.vertex = 2, .weight = 1},
  };
  WeightedGraph graph(pointers, targets);
  DynamicArray<int> vertexMass = {1, 1, 1, 1};

  auto result = HeavyEdgeMatch(graph, vertexMass);

  // For each vertex, check that matchedIn points to a group containing that vertex.
  for (int v = 0; v < graph.size(); ++v) {
    int group = result.matchedIn[v];
    EXPECT_GE(group, 0);
    EXPECT_LT(group, result.match.size());

    // Check that vertex v is in the group.
    bool found = false;
    for (int member : result.match[group]) {
      if (member == v) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Vertex " << v << " not found in group " << group;
  }
}

TEST(HeavyEdgeMatch, SingleVertex) {
  DynamicArray<int> pointers = {0, 0};
  DynamicArray<WeightedEdge> targets = {};
  WeightedGraph graph(pointers, targets);
  int vertexMasses[1] = {1};
  auto result = HeavyEdgeMatch(graph, vertexMasses);
  EXPECT_EQ(1, result.match.size());
  EXPECT_EQ(0, result.matchedIn[0]);
}

// =============================================================================
// BuildCoarsenedGraph Tests
// =============================================================================

TEST(BuildCoarsenedGraph, SizeReduction) {
  // Create a 4-vertex path graph
  DynamicArray<int> pointers = {0, 1, 3, 5, 6};
  DynamicArray<WeightedEdge> targets = {
      {.vertex = 1, .weight = 1},
      {.vertex = 0, .weight = 1},
      {.vertex = 2, .weight = 1},
      {.vertex = 1, .weight = 1},
      {.vertex = 3, .weight = 1},
      {.vertex = 2, .weight = 1},
  };
  WeightedGraph graph(pointers, targets);
  DynamicArray<int> vertexMass = {1, 1, 1, 1};

  auto matchResult = HeavyEdgeMatch(graph, vertexMass);
  auto coarse = CoarsenGraph(graph, matchResult);

  // Coarse graph should have same number of vertices as match groups.
  EXPECT_EQ(matchResult.match.size(), coarse.size());
  // Should be strictly smaller than original (unless all singletons).
  EXPECT_LT(coarse.size(), graph.size());
}

TEST(BuildCoarsenedGraph, WeightAccumulation) {
  // Create a graph where we can verify weight accumulation:
  // Vertices 0,1 will be matched, vertices 2,3 will be matched
  // Edges: 0->2 (w=3), 0->3 (w=4), 1->2 (w=5), 1->3 (w=6)
  // After coarsening: edge from group{0,1} to group{2,3} should have weight 3+4+5+6=18
  DynamicArray<int> pointers = {0, 3, 6, 9, 12};
  DynamicArray<WeightedEdge> targets = {
      // Vertex 0
      {.vertex = 1, .weight = 100}, // Heavy edge to force matching
      {.vertex = 2, .weight = 3},
      {.vertex = 3, .weight = 4},
      // Vertex 1
      {.vertex = 0, .weight = 100},
      {.vertex = 2, .weight = 5},
      {.vertex = 3, .weight = 6},
      // Vertex 2
      {.vertex = 0, .weight = 3},
      {.vertex = 1, .weight = 5},
      {.vertex = 3, .weight = 100},
      // Vertex 3
      {.vertex = 0, .weight = 4},
      {.vertex = 1, .weight = 6},
      {.vertex = 2, .weight = 100},
  };
  WeightedGraph graph(pointers, targets);
  DynamicArray<int> vertexMass = {1, 1, 1, 1};

  auto matchResult = HeavyEdgeMatch(graph, vertexMass);
  auto coarse = CoarsenGraph(graph, matchResult);

  // Should have 2 groups.
  EXPECT_EQ(2, coarse.size());

  // Find the edge between the two groups and verify weight.
  int group01 = matchResult.matchedIn[0];
  int group23 = matchResult.matchedIn[2];
  EXPECT_NE(group01, group23);

  // Find edge from group01 to group23.
  int totalWeight = 0;
  for (auto const& edge : coarse[group01]) {
    if (edge.vertex == group23) {
      totalWeight += edge.weight;
    }
  }
  // Weight should be 3+4+5+6 = 18.
  EXPECT_EQ(18, totalWeight);
}

TEST(BuildCoarsenedGraph, SelfLoopAccumulation) {
  // Create a graph with self-loops (diagonal elements)
  // Vertices 0,1 will be matched
  // Self-loop weights: 0->0 (w=10), 1->1 (w=20)
  // Edges between them: 0->1 (w=100, w=5), 1->0 (w=100, w=5)
  // Coarse self-loop should have weight: 10 + 100 + 5 + 100 + 5 + 20 = 240
  DynamicArray<int> pointers = {0, 3, 6};
  DynamicArray<WeightedEdge> targets = {
      // Vertex 0
      {.vertex = 0, .weight = 10}, // Self-loop
      {.vertex = 1, .weight = 100}, // Heavy edge to force matching
      {.vertex = 1, .weight = 5}, // Additional edge
      // Vertex 1
      {.vertex = 0, .weight = 100},
      {.vertex = 0, .weight = 5},
      {.vertex = 1, .weight = 20}, // Self-loop
  };
  WeightedGraph graph(pointers, targets);
  DynamicArray<int> vertexMass = {1, 1};

  auto matchResult = HeavyEdgeMatch(graph, vertexMass);
  auto coarse = CoarsenGraph(graph, matchResult);

  // Should have 1 group (both vertices matched together).
  EXPECT_EQ(1, coarse.size());

  // Find self-loop weight.
  int selfLoopWeight = 0;
  for (auto const& edge : coarse[0]) {
    if (edge.vertex == 0) {
      selfLoopWeight += edge.weight;
    }
  }
  // Self-loop should accumulate: 10 + 20 + 100 + 100 + 5 + 5 = 240
  // (all edges become self-loops when both vertices are in the same group).
  EXPECT_EQ(240, selfLoopWeight);
}

TEST(BuildCoarsenedGraph, CoarseMassAccumulation) {
  // Create a 4-vertex path graph where pairs will be matched.
  // Vertices 0,1 will be matched, vertices 2,3 will be matched.
  // Assign non-uniform masses to verify accumulation.
  DynamicArray<int> pointers = {0, 1, 3, 5, 6};
  DynamicArray<WeightedEdge> targets = {
      {.vertex = 1, .weight = 100}, // 0->1 heavy edge
      {.vertex = 0, .weight = 100}, // 1->0
      {.vertex = 2, .weight = 1}, // 1->2
      {.vertex = 1, .weight = 1}, // 2->1
      {.vertex = 3, .weight = 100}, // 2->3 heavy edge
      {.vertex = 2, .weight = 100}, // 3->2
  };
  WeightedGraph graph(pointers, targets);
  DynamicArray<int> vertexMass = {3, 5, 7, 11}; // Distinct primes for easy verification

  auto result = BuildCoarsenedGraph(graph, vertexMass);

  // Should have 2 coarse vertices.
  EXPECT_EQ(2, result.graph.size());
  EXPECT_EQ(2, isize(result.coarseMass));

  // Verify coarseMass equals sum of fine vertex masses in each match group.
  for (int g = 0; g < result.groups.match.size(); ++g) {
    int expectedMass = 0;
    for (int fv : result.groups.match[g]) {
      expectedMass += vertexMass[fv];
    }
    EXPECT_EQ(expectedMass, result.coarseMass[g])
        << "CoarseMass[" << g << "] should equal sum of fine vertex masses";
  }

  // Verify total fine mass equals total coarse mass.
  int totalFineMass = 0;
  for (int m : vertexMass) {
    totalFineMass += m;
  }
  int totalCoarseMass = 0;
  for (int m : result.coarseMass) {
    totalCoarseMass += m;
  }
  EXPECT_EQ(totalFineMass, totalCoarseMass) << "Total mass should be preserved";
  EXPECT_EQ(3 + 5 + 7 + 11, totalCoarseMass); // 26
}

TEST(BuildCoarsenedGraph, SymmetryPreservation) {
  // Create a symmetric graph (undirected)
  DynamicArray<int> pointers = {0, 2, 4, 6, 8, 10, 12};
  DynamicArray<WeightedEdge> targets = {
      // Vertex 0: neighbors 1, 3
      {.vertex = 1, .weight = 1},
      {.vertex = 3, .weight = 2},
      // Vertex 1: neighbors 0, 2
      {.vertex = 0, .weight = 1},
      {.vertex = 2, .weight = 3},
      // Vertex 2: neighbors 1, 5
      {.vertex = 1, .weight = 3},
      {.vertex = 5, .weight = 4},
      // Vertex 3: neighbors 0, 4
      {.vertex = 0, .weight = 2},
      {.vertex = 4, .weight = 5},
      // Vertex 4: neighbors 3, 5
      {.vertex = 3, .weight = 5},
      {.vertex = 5, .weight = 6},
      // Vertex 5: neighbors 2, 4
      {.vertex = 2, .weight = 4},
      {.vertex = 4, .weight = 6},
  };
  WeightedGraph graph(pointers, targets);
  DynamicArray<int> vertexMass = {1, 1, 1, 1, 1, 1};

  auto matchResult = HeavyEdgeMatch(graph, vertexMass);
  auto coarse = CoarsenGraph(graph, matchResult);

  // Check symmetry: for each edge (i,j,w), there should be an edge (j,i,w).
  for (int i = 0; i < coarse.size(); ++i) {
    for (auto const& edge : coarse[i]) {
      int j = edge.vertex;
      int wij = edge.weight;

      // Find reverse edge (should be exactly one).
      int wji = 0;
      for (auto const& revEdge : coarse[j]) {
        if (revEdge.vertex == i) {
          wji = revEdge.weight;
        }
      }
      EXPECT_EQ(wij, wji) << "Asymmetric edge between " << i << " and " << j;
    }
  }
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(GraphCoarsening, 2DGridGraph) {
  // Create a 4x4 grid graph with self-loops
  int const nx = 4, ny = 4;
  int const n = nx * ny;
  auto fineGraph = MakeGridGraph(nx, ny, 1, 4);
  DynamicArray<int> vertexMass(n, 1);

  // Apply coarsening
  auto matchResult = HeavyEdgeMatch(fineGraph, vertexMass);
  auto coarseGraph = CoarsenGraph(fineGraph, matchResult);

  // Coarse graph should have fewer vertices.
  EXPECT_LT(coarseGraph.size(), fineGraph.size());
  // But at least half (since matching pairs at most 2 vertices).
  EXPECT_GE(coarseGraph.size(), (fineGraph.size() + 1) / 2);

  // Every fine vertex should map to a valid coarse vertex.
  for (int v = 0; v < fineGraph.size(); ++v) {
    EXPECT_GE(matchResult.matchedIn[v], 0);
    EXPECT_LT(matchResult.matchedIn[v], coarseGraph.size());
  }
}

TEST(GraphCoarsening, CoarsenToLargeGrid_ReachesTarget) {
  // A 128x128 grid (16384 vertices) halves cleanly at each coarsening level, so
  // CoarsenTo should terminate by reaching the target size (not by stalling).
  int const nx = 128, ny = 128;
  int const n = nx * ny;
  auto fineGraph = MakeGridGraph(nx, ny, 1, 4);
  DynamicArray<int> vertexMass(n, 1);

  int const maxVtx = 100;
  auto levels = CoarsenTo(fineGraph, vertexMass, maxVtx);

  EXPECT_GT(levels.size(), 0) << "Should have at least one coarsening level";

  // Verify each level reduces the graph size.
  int prevSize = n;
  for (int lvl = 0; lvl < isize(levels); ++lvl) {
    int const currSize = levels[lvl].graph.size();
    EXPECT_LT(currSize, prevSize) << "Level " << lvl << " should be smaller than previous";
    EXPECT_GE(currSize, (prevSize + 1) / 2)
        << "Level " << lvl << " should have at least half the vertices";

    // Verify coarseMass sums to the original fine graph vertex count.
    // Each coarse vertex's mass represents how many original fine vertices it contains.
    int totalMass = 0;
    for (int m : levels[lvl].coarseMass) {
      totalMass += m;
    }
    EXPECT_EQ(n, totalMass) << "Total mass at level " << lvl
                            << " should equal original fine vertex count";
    prevSize = currSize;
  }

  // Grid graphs reduce well, so the target should be reached.
  EXPECT_LE(levels.back().graph.size(), maxVtx)
      << "Grid graph coarsening should reach the target vertex count";
}

TEST(GraphCoarsening, CoarsenTo_StallsOnStarGraph) {
  // In a star graph, heavy edge matching pairs the center with exactly one leaf;
  // every other leaf becomes a singleton because its only neighbor (the center)
  // is already matched. Each coarsening level therefore reduces the size by just 1,
  // which triggers the stall-termination path (reduction < 15%) in CoarsenTo.
  int const n = 100;
  auto fineGraph = MakeStarGraph(n, 1);
  DynamicArray<int> vertexMass(n, 1);

  // Request a much smaller graph than any single coarsening step can produce.
  int const maxVtx = 5;
  auto levels = CoarsenTo(fineGraph, vertexMass, maxVtx);

  ASSERT_GT(levels.size(), 0) << "Should have attempted at least one coarsening level";

  // The target size cannot be reached on a star: termination must be via the stall path.
  int const finalSize = levels.back().graph.size();
  EXPECT_GT(finalSize, maxVtx) << "Star graph cannot reach target size via coarsening";

  // The last coarsening step must have reduced by less than 15% (the stall threshold).
  int const prevSize = levels.size() >= 2 ? levels[levels.size() - 2].graph.size() : n;
  EXPECT_GE(finalSize, static_cast<int>(0.85 * prevSize))
      << "Final level should have failed the 15%-reduction threshold";
}

TEST(GraphCoarsening, CoarsenToGraphAlreadySmallEnough) {
  // When the graph already has at most maxVtx vertices, CoarsenTo should return
  // an empty result without performing any coarsening.
  int const nx = 2, ny = 2;
  int const n = nx * ny;
  auto fineGraph = MakeGridGraph(nx, ny, 1);
  DynamicArray<int> vertexMass(n, 1);

  auto levels = CoarsenTo(fineGraph, vertexMass, /*maxVtx=*/10);

  EXPECT_TRUE(levels.empty()) << "CoarsenTo should return empty result when graph size <= maxVtx";
}

// =============================================================================
// ProjectSide Tests
// =============================================================================

TEST(ProjectSide, AssignsSideToEachFineVertex) {
  // Coarse graph has 4 vertices, fine graph has 12 vertices grouped as:
  //   coarse 0: fine {0, 4, 8}
  //   coarse 1: fine {1, 5, 9, 11}
  //   coarse 2: fine {2, 6, 10}
  //   coarse 3: fine {3, 7}
  DynamicArray<int> matchedIn = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 1};
  DynamicArray<int> coarseSide = {0, 1, 0, 2};

  auto fineSide = ProjectSide(coarseSide, matchedIn);

  DynamicArray<int> const expected = {0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1};
  EXPECT_SPAN_EQ(MakeConstSpan(expected), MakeConstSpan(fineSide));
}
