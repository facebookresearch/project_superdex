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

#include <mochi_core/utils/graph_utils.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace mochi {

MatchResult HeavyEdgeMatch(WeightedGraph const& graph, Span<int const> vertexMass) {
  int const n = graph.size();
  MOCHI_ASSERT_VERBOSE(isize(vertexMass) == n, "Inconsistent graph and vertexMass");
  // We use a vector of bools for performance reasons. For large graphs, the std::vector<bool>
  // can contain 8x more values in cache than DynamicArray<bool>. Since this vector is accessed in a
  // very irregular pattern, this can significantly improve performance.
  std::vector<bool> matched(n, false);

  DynamicArray<int> order;
  order.resize_noinit(n);
  std::iota(order.begin(), order.end(), 0);
  // The METIS paper states that the matching is done with a randomized order.
  // However, the actual implementation for Heavy Edge Match sorts the vertices by associated vertex
  // mass. The reason is that at the end we would like to have fairly homogenous masses and starting
  // with lower mass vertices will give them a better chance to get matched.
  std::ranges::sort(order, [&](int a, int b) { return vertexMass[a] < vertexMass[b]; });

  GraphBuilder<int, int> builder(n, n);
  DynamicArray<int> matchedIn;
  matchedIn.resize_noinit(n);
  for (int v : order) {
    if (matched[v]) {
      continue;
    }
    // Add match group.
    builder.StartSet();
    builder.InsertTarget(v);
    matched[v] = true;
    matchedIn[v] = builder.CurrentIndex();
    // Find unmatched neighbor with highest weight.
    int bestNeighbor = -1;
    int bestWeight = -1;
    for (WeightedEdge const& edge : graph[v]) {
      if (!matched[edge.vertex] && edge.weight > bestWeight) {
        bestWeight = edge.weight;
        bestNeighbor = edge.vertex;
      }
    }

    if (bestNeighbor >= 0) {
      matchedIn[bestNeighbor] = builder.CurrentIndex();
      builder.InsertTarget(bestNeighbor);
      matched[bestNeighbor] = true;
    }
  }
  return {builder.Build(), std::move(matchedIn)};
}

WeightedGraph CoarsenGraph(WeightedGraph const& fineGraph, MatchResult const& matchResult) {
  auto const& match = matchResult.match;
  auto const& matchedIn = matchResult.matchedIn;
  int const coarseSize = match.size();

  // Temporary array to accumulate weights for each coarse vertex.
  DynamicArray<int> weights(coarseSize, 0);

  GraphBuilder<WeightedEdge, int> builder(coarseSize, fineGraph.NumTargets());
  for (int v = 0; v < coarseSize; ++v) {
    builder.StartSet();

    // First pass: accumulate weights for each target coarse vertex.
    for (int fineVertex : match[v]) {
      for (WeightedEdge const& edge : fineGraph[fineVertex]) {
        int const coarseTarget = matchedIn[edge.vertex];
        weights[coarseTarget] += edge.weight;
      }
    }

    // Second pass: emit edges and reset weights.
    for (int fineVertex : match[v]) {
      for (WeightedEdge const& edge : fineGraph[fineVertex]) {
        int const coarseTarget = matchedIn[edge.vertex];
        if (weights[coarseTarget] != 0) {
          builder.InsertTarget({.vertex = coarseTarget, .weight = weights[coarseTarget]});
          weights[coarseTarget] = 0;
        }
      }
    }
  }
  return builder.Build();
}

CoarseningResult BuildCoarsenedGraph(WeightedGraph const& graph, Span<int const> vertexMass) {
  MOCHI_ASSERT_VERBOSE(isize(vertexMass) == isize(graph), "Inconsistent graph and vertexMass");
  // Perform heavy edge matching.
  MatchResult matchResult = HeavyEdgeMatch(graph, vertexMass);

  // Build coarsened graph.
  WeightedGraph coarsenedGraph = CoarsenGraph(graph, matchResult);

  DynamicArray<int> coarseMasses(matchResult.match.size(), 0);
  for (int fv = 0; fv < isize(graph); ++fv) {
    coarseMasses[matchResult.matchedIn[fv]] += vertexMass[fv];
  }

  // Return result.
  return {std::move(matchResult), std::move(coarsenedGraph), std::move(coarseMasses)};
}

DynamicArray<CoarseningResult>
CoarsenTo(WeightedGraph const& graph, Span<int const> vertexMass, int maxVtx) {
  // If a coarsening does not reduce the number of vertices by more than 15%, we likely
  // have a graph that looks like a star and there is no point in continuing the coarsening.
  // There is no reason to use double precision for this constant. The integers are 32 bits.
  constexpr float kStopCoarseningRatio = 0.85f;
  DynamicArray<CoarseningResult> result;
  if (isize(graph) <= maxVtx) {
    return result;
  }
  WeightedGraph const* currentGraph = &graph;
  Span<int const> currentMass = vertexMass;
  int coarseSize = 0;
  int fineSize = 0;
  do {
    fineSize = isize(*currentGraph);
    result.emplace_back(BuildCoarsenedGraph(*currentGraph, currentMass));
    coarseSize = isize(result.back().graph);
    currentGraph = &result.back().graph;
    currentMass = MakeConstSpan(result.back().coarseMass);
  } while (coarseSize < kStopCoarseningRatio * fineSize && coarseSize > maxVtx);
  return result;
}

DynamicArray<int> ProjectSide(Span<int const> coarseSide, Span<int const> matchedIn) {
  int const n = isize(matchedIn);
  DynamicArray<int> fineSide;
  fineSide.resize_noinit(n);
  for (int fv = 0; fv < n; ++fv) {
    MOCHI_ASSERT_VERBOSE(
        matchedIn[fv] >= 0 && matchedIn[fv] < isize(coarseSide),
        "matchedIn[fv] out of range of coarseSide");
    fineSide[fv] = coarseSide[matchedIn[fv]];
  }
  return fineSide;
}

} // namespace mochi
