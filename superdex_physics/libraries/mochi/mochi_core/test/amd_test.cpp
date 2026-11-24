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

#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/numbering/amd.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <set>

namespace mochi {

/// @brief Symmetric graph on @p n vertices with the given edges, diagonal included.
static Graph<int, int> MakeGraph(int n, DynamicArray<std::array<int, 2>> const& edges) {
  DynamicArray<DynamicArray<int>> rows(n);
  for (int v = 0; v < n; ++v) {
    rows[v].push_back(v);
  }
  for (auto const [a, b] : edges) {
    rows[a].push_back(b);
    rows[b].push_back(a);
  }

  GraphBuilder<int, int> gb(n, n + 2 * isize(edges));
  for (auto& row : rows) {
    std::ranges::sort(row);
    gb.append(row);
  }
  return gb.Build();
}

/// @brief Graph of an @p m x @p m grid numbered row by row: each vertex is coupled to its 4 nearest
/// neighbors (5-point stencil), or to all 8 with @p diagonals (node graph of a quad mesh).
static Graph<int, int> MakeGridGraph(int m, bool diagonals) {
  DynamicArray<std::array<int, 2>> edges;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      int const v = i * m + j;
      if (j + 1 < m) {
        edges.push_back({v, v + 1});
      }
      if (i + 1 < m) {
        edges.push_back({v, v + m});
      }
      if (diagonals && i + 1 < m && j + 1 < m) {
        edges.push_back({v, v + m + 1});
        edges.push_back({v + 1, v + m});
      }
    }
  }
  return MakeGraph(m * m, edges);
}

/// @brief Random tree on @p n vertices.
static Graph<int, int> MakeRandomTree(int n, unsigned seed) {
  std::mt19937 rng(seed);
  DynamicArray<std::array<int, 2>> edges;
  for (int v = 1; v < n; ++v) {
    edges.push_back({v, std::uniform_int_distribution<int>(0, v - 1)(rng)});
  }
  return MakeGraph(n, edges);
}

/// @brief Graph on @p n vertices with @p numEdges edges between uniformly drawn vertices.
static Graph<int, int> MakeRandomGraph(int n, int numEdges, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> vertex(0, n - 1);
  DynamicArray<std::array<int, 2>> edges;
  for (int k = 0; k < numEdges; ++k) {
    edges.push_back({vertex(rng), vertex(rng)});
  }
  return MakeGraph(n, edges);
}

/// @brief Cycle on @p n vertices.
static Graph<int, int> MakeCycle(int n) {
  DynamicArray<std::array<int, 2>> edges;
  for (int v = 0; v < n; ++v) {
    edges.push_back({v, (v + 1) % n});
  }
  return MakeGraph(n, edges);
}

static bool IsPermutation(Span<int const> perm, int n) {
  DynamicArray<int> sorted(perm.begin(), perm.end());
  std::ranges::sort(sorted);
  DynamicArray<int> identity(n);
  std::iota(identity.begin(), identity.end(), 0);
  return sorted == identity;
}

/// @brief Factorization metrics of @p graph eliminated in the order @p perm.
static FactorMetrics Metrics(Graph<int, int> const& graph, Span<int const> perm) {
  DynamicArray<int> order(perm.begin(), perm.end());
  DynamicArray<int> position = InversePermutation(order);
  return EliminationTree(graph, order, position).ComputeFactorMetrics(/*dofsPerNode=*/1);
}

/// @brief Exact minimum degree order of @p graph, lowest vertex first among ties.
static DynamicArray<int> ExactMinimumDegreeOrder(Graph<int, int> const& graph) {
  int const n = graph.size();
  DynamicArray<std::set<int>> adjacency(n);
  for (int v = 0; v < n; ++v) {
    adjacency[v].insert(graph[v].begin(), graph[v].end());
    adjacency[v].erase(v);
  }

  DynamicArray<int> order;
  DynamicArray<bool> eliminated(n, false);
  for (int k = 0; k < n; ++k) {
    int pivot = -1;
    for (int v = 0; v < n; ++v) {
      if (!eliminated[v] && (pivot < 0 || adjacency[v].size() < adjacency[pivot].size())) {
        pivot = v;
      }
    }
    order.push_back(pivot);
    eliminated[pivot] = true;

    // The neighbors of pivot become a clique.
    for (int const u : adjacency[pivot]) {
      adjacency[u].erase(pivot);
      adjacency[u].insert(adjacency[pivot].begin(), adjacency[pivot].end());
      adjacency[u].erase(u);
    }
  }
  return order;
}

TEST(Amd, OrdersGridsAsWellAsExactMinimumDegree) {
  // Ties make the fill of smaller grids vary by up to 20% with any minimum degree variant.
  for (bool const diagonals : {false, true}) {
    for (int const m : {13, 17, 25, 33}) {
      auto const graph = MakeGridGraph(m, diagonals);

      auto const perm = RenumberByAMD(graph);

      ASSERT_TRUE(IsPermutation(perm, graph.size()));
      EXPECT_LE(
          Metrics(graph, perm).opCount,
          1.05 * Metrics(graph, ExactMinimumDegreeOrder(graph)).opCount);
    }
  }
}

TEST(Amd, MatchesExactMinimumDegreeFill) {
  // Every elimination order fills a cycle C_n with n - 3 entries, so the cycles check validity,
  // and the long ones storage compaction, not ordering quality. The star and trees check ordering
  // (zero fill) and reach aggressive absorption. The seven-vertex graph's fill depends on degree
  // ties: its numbering was picked so that AMD needs aggressive absorption to reach the minimum
  // fill of 2, so a tie-breaking change must re-check it.
  DynamicArray<Graph<int, int>> graphs;
  graphs.push_back(MakeGraph(4, {{0, 1}, {0, 2}, {0, 3}}));
  graphs.push_back(MakeGraph(
      7, {{0, 1}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {1, 2}, {1, 6}, {2, 3}, {2, 4}, {3, 6}, {4, 6}}));
  for (int const n : {4, 50, 500}) {
    graphs.push_back(MakeCycle(n));
    for (unsigned const seed : {1u, 2u, 3u}) {
      graphs.push_back(MakeRandomTree(n, seed));
    }
  }

  for (auto const& graph : graphs) {
    auto const perm = RenumberByAMD(graph);

    ASSERT_TRUE(IsPermutation(perm, graph.size()));
    EXPECT_EQ(Metrics(graph, perm).storage, Metrics(graph, ExactMinimumDegreeOrder(graph)).storage);
  }
}

TEST(Amd, ProducesValidPermutations) {
  EXPECT_TRUE(RenumberByAMD(Graph<int, int>{}).empty());

  // An isolated vertex, two pairs and a triangle. In K2,4 and the next graph, a pivot without and
  // with elements, respectively, still lists a merged variable. The random graph compacts storage
  // twice.
  DynamicArray<Graph<int, int>> graphs;
  graphs.push_back(MakeGraph(8, {{1, 2}, {3, 4}, {5, 6}, {5, 7}, {6, 7}}));
  graphs.push_back(MakeGraph(6, {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {5, 1}, {5, 2}, {5, 3}, {5, 4}}));
  graphs.push_back(MakeGraph(
      8, {{0, 4}, {1, 5}, {1, 6}, {2, 5}, {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {6, 7}}));
  graphs.push_back(MakeRandomGraph(/*n=*/300, /*numEdges=*/1000, /*seed=*/1));

  for (auto const& graph : graphs) {
    EXPECT_TRUE(IsPermutation(RenumberByAMD(graph), graph.size()));
  }
}

TEST(Amd, IgnoresDuplicateEntries) {
  // Storing every entry of a grid twice, as an assembled matrix may be, doesn't change its
  // ordering. Each row repeats its entries, so rows are unsorted and duplicates aren't adjacent.
  auto const grid = MakeGridGraph(/*m=*/9, /*diagonals=*/true);
  GraphBuilder<int, int> gb(grid.size(), 2 * grid.NumTargets());
  for (auto const vertex : grid) {
    DynamicArray<int> row;
    Append(row, vertex.targets);
    Append(row, vertex.targets);
    gb.append(row);
  }
  EXPECT_EQ(RenumberByAMD(gb.Build()), RenumberByAMD(grid));
}

} // namespace mochi
