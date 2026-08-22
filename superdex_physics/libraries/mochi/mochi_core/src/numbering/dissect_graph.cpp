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

#include <mochi_core/numbering/graph_coarsening.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/indexed_heap.h>
#include <algorithm>
#include <array>
#include <numeric>

namespace mochi {
namespace {

/// @brief Target number of vertices in the coarsest graph used by @ref Trisect.
constexpr int kCoarseTargetSize = 100;

/// @brief Number of projection steps between successive @ref OptimizeSeparator calls in the
/// uncoarsening phase of @ref Trisect.
constexpr int kOptLevels = 2;

/// @brief Collect the indices of the separator vertices (those with side == kSeparatorSide).
[[nodiscard]] DynamicArray<int> CollectSeparator(Span<int const> side) {
  DynamicArray<int> separator;
  separator.reserve(std::ranges::count(side, kSeparatorSide));
  for (int v = 0; v < isize(side); ++v) {
    if (side[v] == kSeparatorSide) {
      separator.push_back(v);
    }
  }
  return separator;
}
/// @brief Performs Greedy Graph Growing Partition starting from a seed vertex.
///
/// @details The algorithm grows a partition by iteratively selecting the boundary vertex that
/// minimizes the increase in edge cut (total weight of edges between selected and non-selected
/// vertices). All vertices start as non-selected. The algorithm maintains:
/// - A boundary heap of non-selected vertices adjacent to at least one selected vertex
/// - For each boundary vertex, the cost of selecting it: totalEdgeWeight - 2 * edgesToSelected
///
/// When a vertex is selected, the edge cut changes per neighbor: +weight for each non-selected
/// neighbor (edge becomes a cut edge) and -weight for each selected neighbor (edge leaves the cut).
///
/// @note There may not be a solution where the maxUnbalance constraint is satisfied. This is due to
/// the fact that the coarsening may have produced vertices with masses larger than the maximum
/// unbalance. We silently accept such results.
///
/// @param graph The input weighted graph.
/// @param vertexMass The mass associated with each vertex.
/// @param totalMass The sum of all vertex masses.
/// @param maxUnbalance Maximum allowed difference in mass between the two sides.
/// @param seed The starting vertex index for the partition.
///
/// @return A dissection result containing the partition assignment and cut weight.
[[nodiscard]] DissectionResult GreedyGraphGrowingPartition(
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int totalMass,
    int maxUnbalance,
    int seed) {
  MOCHI_ASSERT_VERBOSE(
      totalMass == std::accumulate(vertexMass.begin(), vertexMass.end(), 0), "Inconsistent masses");
  int const n = graph.size();
  int const lowerMass = (totalMass - maxUnbalance + 1) / 2;
  int const upperMass = (totalMass + maxUnbalance) / 2;

  // Track which side each vertex is on: kSide1 = selected, kSide0 = not selected.
  DynamicArray<int> side(n, kSide0);

  // Boundary heap: non-selected vertices adjacent to at least one selected vertex.
  // Cost = change in total edge cut if vertex is selected
  //      = sum(weight(edge) to non-selected vertices) - sum(weight(edge) to selected
  //      vertices).
  // The cost is computed on-the-fly when a vertex first enters the boundary, then updated
  // incrementally by - 2 * edge.weight each time an additional neighbor is selected.
  IndexedHeap<int> heap(n, 0);

  // Lambda computing the boundary cost for a vertex by scanning its edges.
  auto const computeCost = [&](int v) {
    int cost = 0;
    for (WeightedEdge const& edge : graph[v]) {
      if (edge.vertex != v) {
        cost += (side[edge.vertex] == kSide1) ? -edge.weight : edge.weight;
      }
    }
    return cost;
  };

  // Insert the seed into the boundary heap. The main loop will select it first and
  // process its neighbors, bootstrapping the partition.
  auto seedCost = computeCost(seed);
  heap.Insert(seed, seedCost);
  int selectedMass = 0;
  int cutWeight = 0;

  // Grow to upperMass, recording each candidate solution between lowerMass and upperMass.
  // We pick the candidate with the minimum cut weight.
  struct Candidate {
    int vertex;
    int cutWeight;
    int mass;
  };

  DynamicArray<Candidate> candidates;
  // 32 should be enough for most cases since the coarsened graph is usually small.
  candidates.reserve(32);
  // Tracks the next vertex to consider when re-seeding for disconnected components,
  // so we don't rescan already-visited vertices.
  int nextUnselected = 0;
  while (!heap.IsEmpty()) {
    auto const [bestVertex, bestCost] = heap.ExtractMin();
    auto const addedMass = vertexMass[bestVertex];
    auto const nextMass = selectedMass + addedMass;
    // Exit if bestVertex would exceed the upper constraint and the previous
    // situation is closer to equilibrium.
    if (nextMass > upperMass && Abs(totalMass - 2 * selectedMass) < Abs(totalMass - 2 * nextMass)) {
      break;
    }
    cutWeight += bestCost;
    // The vertex may be set back to kSide0 later
    side[bestVertex] = kSide1;
    selectedMass = nextMass;

    // Update neighbors of the newly selected vertex.
    for (WeightedEdge const& edge : graph[bestVertex]) {
      int const u = edge.vertex;
      if (u == bestVertex || side[u] == kSide1) {
        continue;
      }

      // Neighbor is not selected: this edge now crosses the cut.
      if (!heap.Contains(u)) {
        heap.Insert(u, computeCost(u));
      } else {
        heap.UpdateCost(u, heap.GetCost(u) - 2 * edge.weight);
      }
    }
    // In order to properly handle the case where the mass jumps from below the lower mass to
    // above the upper mass, we need to keep track of the candidates.
    if (selectedMass >= lowerMass) {
      candidates.push_back({bestVertex, cutWeight, selectedMass});
    }

    // If the heap was emptied (e.g. the graph is disconnected) before reaching the lower
    // mass bound, seed the heap with the first unselected vertex so the partition can
    // continue growing into another component.
    // NOTE: If the graph is guaranteed to be connected, this is not needed.
    if (heap.IsEmpty() && selectedMass < lowerMass) {
      for (; nextUnselected < n; ++nextUnselected) {
        if (side[nextUnselected] == kSide0) {
          heap.Insert(nextUnselected, computeCost(nextUnselected));
          ++nextUnselected;
          break;
        }
      }
    }
  }

  if (candidates.empty()) {
    return {std::move(side), cutWeight, false};
  }
  // Find the candidate with minimum cut weight. Tie-break by unbalance.
  // We do not assume that the iterator type is a pointer. This keeps it
  // flexible to changes.
  // NOLINTNEXTLINE(readability-qualified-auto)
  auto it =
      std::ranges::min_element(candidates, [totalMass](Candidate const& a, Candidate const& b) {
        return a.cutWeight == b.cutWeight
            ? Abs(totalMass - 2 * a.mass) < Abs(totalMass - 2 * b.mass)
            : a.cutWeight < b.cutWeight;
      });
  int const bestCutWeight = it->cutWeight;
  bool const balanced = it->mass >= lowerMass && it->mass <= upperMass;

  // Deselect vertices added after the best candidate.
  for (++it; it != candidates.end(); ++it) {
    side[it->vertex] = kSide0;
  }

  return {std::move(side), bestCutWeight, balanced};
}

} // namespace

DissectionResult DissectGraph(
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int totalMass,
    int maxUnbalance) {
  int const n = graph.size();
  MOCHI_ASSERT_VERBOSE(isize(vertexMass) == n, "Inconsistent graph and vertexMass");

  if (n == 0) {
    return {{}, 0, true};
  }

  if (n == 1) {
    return {{kSide0}, 0, true};
  }

  // Try multiple seeds and pick the best result.
  // For now, try a few vertices with different characteristics.
  std::array<int, 4> seeds{};
  int seedCount = 0;

  // Seed 0: vertex with smallest mass.
  int minMassVertex = 0;
  for (int v = 1; v < n; ++v) {
    if (vertexMass[v] < vertexMass[minMassVertex]) {
      minMassVertex = v;
    }
  }
  seeds[seedCount++] = minMassVertex;

  // Seed 1: vertex with highest degree (most connections).
  int maxDegreeVertex = 0;
  int maxDegree = graph.EdgeCount(0);
  for (int v = 1; v < n; ++v) {
    int const degree = graph.EdgeCount(v);
    if (degree > maxDegree) {
      maxDegree = degree;
      maxDegreeVertex = v;
    }
  }
  if (maxDegreeVertex != minMassVertex) {
    seeds[seedCount++] = maxDegreeVertex;
  }

  // Seed 2: first vertex (index 0) if not already included.
  if (minMassVertex != 0 && maxDegreeVertex != 0) {
    seeds[seedCount++] = 0;
  }

  // Seed 3: last vertex if not already included.
  if (minMassVertex != n - 1 && maxDegreeVertex != n - 1) {
    seeds[seedCount++] = n - 1;
  }

  // Run GGGP from each seed and keep the best result.
  DissectionResult bestResult =
      GreedyGraphGrowingPartition(graph, vertexMass, totalMass, maxUnbalance, seeds[0]);

  for (int i = 1; i < seedCount; ++i) {
    DissectionResult result =
        GreedyGraphGrowingPartition(graph, vertexMass, totalMass, maxUnbalance, seeds[i]);
    if (result.balanced > bestResult.balanced ||
        (result.balanced == bestResult.balanced && result.cutWeight < bestResult.cutWeight)) {
      bestResult = std::move(result);
    }
  }

  return bestResult;
}

void MarkTrivialSeparatorBothSides(Span<int> side, WeightedGraph const& graph) {
  int const n = graph.size();
  for (int v = 0; v < n; ++v) {
    int const mySide = side[v];
    for (WeightedEdge const& edge : graph[v]) {
      // There is no need to check edge.vertex != v, since side[v] != side[v] is false.
      // Adding the check would potentially create additional branch mispredictions.
      // Skip neighbors already in the separator: a vertex whose only opposite-side neighbors
      // have already been promoted is no longer on a boundary.
      int const neighborSide = side[edge.vertex];
      if (neighborSide != mySide && neighborSide != kSeparatorSide) {
        side[v] = kSeparatorSide;
        break;
      }
    }
  }
}

int MarkTrivialSeparatorOneSide(
    Span<int> side,
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int whichSide) {
  int const n = graph.size();
  int separatorMass = 0;
  for (int v = 0; v < n; ++v) {
    if (side[v] != whichSide) {
      continue;
    }
    for (WeightedEdge const& edge : graph[v]) {
      // There is no need to check edge.vertex != v, since side[v] != side[v] is false.
      // Adding the check would potentially create additional branch mispredictions.
      if (side[edge.vertex] != whichSide && side[edge.vertex] != kSeparatorSide) {
        side[v] = kSeparatorSide;
        separatorMass += vertexMass[v];
        break;
      }
    }
  }
  return separatorMass;
}

namespace {

struct SeparatorOptimizer {
  Span<int> side;
  WeightedGraph const& graph;
  Span<int const> vertexMass;
  int maxUnbalance;

  int mass0;
  int mass1;
  int separatorMass;
  DynamicArray<int> degree0;
  DynamicArray<int> degree1;
  IndexedHeap<int> heap0;
  IndexedHeap<int> heap1;

  struct StashedMove {
    int v;
    int targetSide;
    int cost;
  };
  DynamicArray<StashedMove> stashed;

  SeparatorOptimizer(Span<int> s, WeightedGraph const& g, Span<int const> vm, int mu)
      : side(s),
        graph(g),
        vertexMass(vm),
        maxUnbalance(mu),
        mass0(0),
        mass1(0),
        separatorMass(0),
        degree0(graph.size(), 0),
        degree1(graph.size(), 0),
        heap0(graph.size()),
        heap1(graph.size()) {
    // The list of stashed vertices should generally be small.
    // 32 is just a reasonable default value.
    stashed.reserve(32);
  }

  void Initialize() {
    int const n = graph.size();
    for (int v = 0; v < n; ++v) {
      if (side[v] == kSide0) {
        mass0 += vertexMass[v];
      } else if (side[v] == kSide1) {
        mass1 += vertexMass[v];
      } else {
        separatorMass += vertexMass[v];
      }

      for (WeightedEdge const& e : graph[v]) {
        if (e.vertex == v) {
          continue;
        }
        if (side[e.vertex] == kSide0) {
          ++degree0[v];
        } else if (side[e.vertex] == kSide1) {
          ++degree1[v];
        }
      }
    }

    for (int v = 0; v < n; ++v) {
      if (side[v] == kSeparatorSide) {
        if (degree0[v] > 0) {
          heap0.Insert(v, ComputeCost(v, kSide0));
        }
        if (degree1[v] > 0) {
          heap1.Insert(v, ComputeCost(v, kSide1));
        }
      }
    }
  }

  int ComputeCost(int v, int targetSide) const {
    int const otherSide = (targetSide == kSide0) ? kSide1 : kSide0;
    int cost = -vertexMass[v];
    for (WeightedEdge const& e : graph[v]) {
      if (e.vertex == v) {
        continue;
      }
      if (side[e.vertex] == otherSide) {
        cost += vertexMass[e.vertex];
      }
    }
    return cost;
  }

  void RestoreStashed() {
    for (auto const& m : stashed) {
      if (m.targetSide == kSide0) {
        heap0.Insert(m.v, m.cost);
      } else {
        heap1.Insert(m.v, m.cost);
      }
    }
    stashed.clear();
  }

  void MoveVertexToSide(int v, int x, int newMass0, int newMass1) {
    int const y = (x == kSide0) ? kSide1 : kSide0;
    auto& degreeX = (x == kSide0) ? degree0 : degree1;
    auto& heapX = (x == kSide0) ? heap0 : heap1;
    auto& heapY = (y == kSide0) ? heap0 : heap1;

    side[v] = x;
    mass0 = newMass0;
    mass1 = newMass1;
    separatorMass -= vertexMass[v];

    if (degree0[v] > 0) {
      heap0.Delete(v);
    }
    if (degree1[v] > 0) {
      heap1.Delete(v);
    }

    for (WeightedEdge const& edge : graph[v]) {
      int const w = edge.vertex;
      if (w == v) {
        continue;
      }
      ++degreeX[w];

      if (side[w] == kSeparatorSide) {
        if (degreeX[w] == 1) {
          heapX.Insert(w, ComputeCost(w, x));
        }
        if (heapY.Contains(w)) {
          heapY.UpdateCost(w, heapY.GetCost(w) + vertexMass[v]);
        }
      }
    }
  }

  void PullNeighborsIntoSeparator(int v, int y) {
    int const x = (y == kSide0) ? kSide1 : kSide0;
    auto& degreeY = (y == kSide0) ? degree0 : degree1;
    auto& heapY = (y == kSide0) ? heap0 : heap1;
    auto& heapX = (x == kSide0) ? heap0 : heap1;

    for (WeightedEdge const& edge : graph[v]) {
      int const u = edge.vertex;
      if (u == v) {
        continue;
      }
      if (side[u] == y) {
        side[u] = kSeparatorSide;
        separatorMass += vertexMass[u];

        if (degree0[u] > 0) {
          heap0.Insert(u, ComputeCost(u, kSide0));
        }
        if (degree1[u] > 0) {
          heap1.Insert(u, ComputeCost(u, kSide1));
        }

        for (WeightedEdge const& uEdge : graph[u]) {
          int const w = uEdge.vertex;
          if (w == u) {
            continue;
          }
          --degreeY[w];

          if (side[w] == kSeparatorSide) {
            // Always update heap_x cost: u is no longer a side-y neighbor that
            // would be pulled into the separator if w were moved to side x.
            if (heapX.Contains(w)) {
              heapX.UpdateCost(w, heapX.GetCost(w) - vertexMass[u]);
            }
            // Separately, remove w from heap_y if it no longer has any side-y
            // neighbors (so moving it to side y is no longer meaningful).
            if (degreeY[w] == 0 && heapY.Contains(w)) {
              heapY.Delete(w);
            }
          }
        }
      }
    }
  }

  int Optimize() {
    Initialize();

    while (!heap0.IsEmpty() || !heap1.IsEmpty()) {
      int const c0 = heap0.IsEmpty() ? std::numeric_limits<int>::max() : heap0.FindMin().cost;
      int const c1 = heap1.IsEmpty() ? std::numeric_limits<int>::max() : heap1.FindMin().cost;

      if (c0 >= 0 && c1 >= 0) {
        break; // No valid improving moves
      }

      int const x = (c0 < c1) ? kSide0 : kSide1;
      auto& pickHeap = (x == kSide0) ? heap0 : heap1;
      int const v = pickHeap.FindMin().key;
      int const cost = pickHeap.FindMin().cost;

      int const y = (x == kSide0) ? kSide1 : kSide0;
      int const deltaMassX = vertexMass[v];
      int deltaMassY = 0;
      for (WeightedEdge const& edge : graph[v]) {
        int const u = edge.vertex;
        if (u == v) {
          continue;
        }
        if (side[u] == y) {
          deltaMassY -= vertexMass[u];
        }
      }
      int const newMass0 = mass0 + ((x == kSide0) ? deltaMassX : deltaMassY);
      int const newMass1 = mass1 + ((x == kSide1) ? deltaMassX : deltaMassY);

      if (std::abs(newMass0 - newMass1) <= std::max(maxUnbalance, std::abs(mass0 - mass1))) {
        RestoreStashed();
        MoveVertexToSide(v, x, newMass0, newMass1);
        PullNeighborsIntoSeparator(v, y);
      } else {
        pickHeap.ExtractMin();
        stashed.push_back({v, x, cost});
      }
    }
    return separatorMass;
  }
};

} // namespace

int OptimizeSeparator(
    Span<int> side,
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int maxUnbalance) {
  SeparatorOptimizer optimizer(side, graph, vertexMass, maxUnbalance);
  return optimizer.Optimize();
}

namespace {

/// Picks the best of three trivial initial separators (both-sides, side-0-only, side-1-only),
/// optimizes each via @ref OptimizeSeparator, and writes the winning side assignment back into
/// @p side. Returns the separator mass of the winner. Does not collect separator vertex indices.
int OptimizeBestInitialSeparator(
    Span<int> side,
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int maxUnbalance) {
  MOCHI_ASSERT_VERBOSE(isize(side) == graph.size(), "Inconsistent graph and side");
  MOCHI_ASSERT_VERBOSE(isize(vertexMass) == graph.size(), "Inconsistent graph and vertexMass");

  // Try the three possible trivial initial separators and keep the one with the smallest
  // separator mass after optimization:
  //   A: boundary vertices from both sides.
  //   B: boundary vertices from kSide0 only.
  //   C: boundary vertices from kSide1 only.
  DynamicArray<int> sideA{side.begin(), side.end()};
  MarkTrivialSeparatorBothSides(sideA, graph);
  int const massA = OptimizeSeparator(sideA, graph, vertexMass, maxUnbalance);

  DynamicArray<int> sideB{side.begin(), side.end()};
  (void)MarkTrivialSeparatorOneSide(sideB, graph, vertexMass, kSide0);
  int const massB = OptimizeSeparator(sideB, graph, vertexMass, maxUnbalance);

  DynamicArray<int> sideC{side.begin(), side.end()};
  (void)MarkTrivialSeparatorOneSide(sideC, graph, vertexMass, kSide1);
  int const massC = OptimizeSeparator(sideC, graph, vertexMass, maxUnbalance);

  DynamicArray<int>* best = &sideA;
  int bestMass = massA;
  if (massB < bestMass) {
    best = &sideB;
    bestMass = massB;
  }
  if (massC < bestMass) {
    best = &sideC;
    bestMass = massC;
  }

  // Copy the winning side assignment back into the caller's buffer.
  std::ranges::copy(*best, side.begin());
  return bestMass;
}

} // namespace

DynamicArray<int> BuildOptimizedSeparator(
    Span<int> side,
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int maxUnbalance) {
  (void)OptimizeBestInitialSeparator(side, graph, vertexMass, maxUnbalance);
  return CollectSeparator(MakeConstSpan(side));
}

TrisectionResult Trisect(WeightedGraph const& graph, Span<int const> vertexMass, int maxUnbalance) {
  MOCHI_ASSERT_VERBOSE(isize(vertexMass) == isize(graph), "Inconsistent graph and vertexMass");

  int const totalMass = std::accumulate(vertexMass.begin(), vertexMass.end(), 0);

  // Phase 1: coarsen the graph to at most kCoarseTargetSize vertices.
  DynamicArray<CoarseningResult> const levels = CoarsenTo(graph, vertexMass, kCoarseTargetSize);
  int const numLevels = isize(levels);

  // Helpers to access the graph and vertex mass at a given level (0 == original/finest graph).
  auto const graphAtLevel = [&](int level) -> WeightedGraph const& {
    return level == 0 ? graph : levels[level - 1].graph;
  };
  auto const massAtLevel = [&](int level) -> Span<int const> {
    return level == 0 ? vertexMass : MakeConstSpan(levels[level - 1].coarseMass);
  };

  // Phase 2a: bisect the coarsest graph.
  DissectionResult dissection =
      DissectGraph(graphAtLevel(numLevels), massAtLevel(numLevels), totalMass, maxUnbalance);
  DynamicArray<int> side = std::move(dissection.side);

  // No coarsening was needed: build the separator on the original graph and return.
  if (numLevels == 0) {
    DynamicArray<int> separator =
        BuildOptimizedSeparator(MakeSpan(side), graph, vertexMass, maxUnbalance);
    return {std::move(side), std::move(separator)};
  }

  // Phase 2b: project the bisection one level up, then refine it. The separator vertex
  // indices are not needed at intermediate levels; we collect them only after the final
  // projection to the finest graph below.
  side = ProjectSide(MakeConstSpan(side), MakeConstSpan(levels[numLevels - 1].groups.matchedIn));
  int currentLevel = numLevels - 1;
  (void)OptimizeBestInitialSeparator(
      MakeSpan(side), graphAtLevel(currentLevel), massAtLevel(currentLevel), maxUnbalance);

  // Phase 3: project to finer levels, optimizing every kOptLevels projections, and always at
  // the finest level.
  for (int projectionsSinceOpt = 0; currentLevel-- > 0;) {
    side = ProjectSide(MakeConstSpan(side), MakeConstSpan(levels[currentLevel].groups.matchedIn));
    ++projectionsSinceOpt;

    if (projectionsSinceOpt == kOptLevels || currentLevel == 0) {
      (void)OptimizeSeparator(
          MakeSpan(side), graphAtLevel(currentLevel), massAtLevel(currentLevel), maxUnbalance);
      projectionsSinceOpt = 0;
    }
  }

  DynamicArray<int> separator = CollectSeparator(MakeConstSpan(side));
  return {std::move(side), std::move(separator)};
}

Array<SideSubgraph, 2>
ExtractSideSubgraphs(WeightedGraph const& graph, Span<int const> vertexMass, Span<int const> side) {
  int const n = graph.size();
  MOCHI_ASSERT_VERBOSE(isize(vertexMass) == n, "Inconsistent graph and vertexMass");
  MOCHI_ASSERT_VERBOSE(isize(side) == n, "Inconsistent graph and side");

  // Pass 1: count vertices and retained edges per side, and assign new vertex indices.
  DynamicArray<int> newIndex(n, -1);
  Array<int, 2> nVertices{0, 0};
  Array<int, 2> nEdges{0, 0};
  for (auto [v, neighbors] : graph) {
    int const s = side[v];
    if (s != 0 && s != 1) {
      continue;
    }
    newIndex[v] = nVertices[s]++;
    for (WeightedEdge const& e : neighbors) {
      if (side[e.vertex] == s) {
        ++nEdges[s];
      }
    }
  }

  // Allocate per-side storage with exact sizes.
  Array<GraphBuilder<WeightedEdge, int>, 2> builders{
      GraphBuilder<WeightedEdge, int>(nVertices[0], nEdges[0]),
      GraphBuilder<WeightedEdge, int>(nVertices[1], nEdges[1]),
  };
  Array<DynamicArray<int>, 2> oldVertex;
  Array<DynamicArray<int>, 2> newMass;
  for (int s = 0; s < 2; ++s) {
    oldVertex[s].reserve(nVertices[s]);
    newMass[s].reserve(nVertices[s]);
  }

  // Pass 2: emit each side's vertices in original-index order, with edges remapped.
  for (auto [v, neighbors] : graph) {
    int const s = side[v];
    if (s != 0 && s != 1) {
      continue;
    }
    oldVertex[s].push_back(v);
    newMass[s].push_back(vertexMass[v]);
    builders[s].StartSet();
    for (WeightedEdge const& e : neighbors) {
      if (side[e.vertex] == s) {
        builders[s].InsertTarget({newIndex[e.vertex], e.weight});
      }
    }
  }

  return {
      SideSubgraph{builders[0].Build(), std::move(newMass[0]), std::move(oldVertex[0])},
      SideSubgraph{builders[1].Build(), std::move(newMass[1]), std::move(oldVertex[1])},
  };
}

namespace {

/// @brief Recursive helper for @ref RenumberByDissection. The maximum unbalance for each
/// level is scaled by @p subMass / @p topMass so the relative imbalance constraint is preserved.
[[nodiscard]] DynamicArray<int> RenumberByDissectionImpl(
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int topMaxUnbalance,
    int topMass,
    int leafSize) {
  int const n = graph.size();
  MOCHI_ASSERT_VERBOSE(leafSize > 1);
  // Leaf: emit vertices in identity order. The recursive caller will remap through the
  // subgraph's oldVertex array to original indices.
  if (n < leafSize) {
    DynamicArray<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    return perm;
  }

  // Scale the unbalance budget proportionally to this level's mass.
  int const subMass = std::accumulate(vertexMass.begin(), vertexMass.end(), 0);
  int const levelUnbalance =
      static_cast<int>(static_cast<int64_t>(topMaxUnbalance) * subMass / topMass);

  TrisectionResult trisection = Trisect(graph, vertexMass, levelUnbalance);
  Array<SideSubgraph, 2> subs = ExtractSideSubgraphs(graph, vertexMass, trisection.side);

  DynamicArray<int> perm;
  perm.reserve(n);
  for (int s = 0; s < 2; ++s) {
    DynamicArray<int> const subPerm = RenumberByDissectionImpl(
        subs[s].graph, MakeConstSpan(subs[s].vertexMass), topMaxUnbalance, topMass, leafSize);
    for (int newIdx : subPerm) {
      perm.push_back(subs[s].oldVertex[newIdx]);
    }
  }
  for (int sepV : trisection.separator) {
    perm.push_back(sepV);
  }
  return perm;
}

} // namespace

DynamicArray<int> RenumberByDissection(
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int maxUnbalance,
    int leafSize) {
  MOCHI_ASSERT_VERBOSE(isize(vertexMass) == graph.size(), "Inconsistent graph and vertexMass");
  MOCHI_ASSERT_VERBOSE(leafSize > 1, "leafSize must be larger than 1");

  int const totalMass = std::accumulate(vertexMass.begin(), vertexMass.end(), 0);
  // Avoid division by zero when totalMass is 0 (empty graph).
  if (totalMass == 0) {
    DynamicArray<int> perm(graph.size());
    std::iota(perm.begin(), perm.end(), 0);
    return perm;
  }
  return RenumberByDissectionImpl(graph, vertexMass, maxUnbalance, totalMass, leafSize);
}

} // namespace mochi
