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

#pragma once

#include <mochi_core/utils/graph.h>

namespace mochi {

/// @brief Edge to a target vertex with an associated weight.
struct WeightedEdge {
  int vertex;
  int weight;
};

/// @brief Directed graph with weighted edges.
using WeightedGraph = Graph<WeightedEdge, int>;

/// @brief Non-owning view of a weighted graph.
using WeightedGraphView = Graph<WeightedEdge, int, Span>;

/// @brief Creates a weighted graph from an unweighted graph with uniform edge weights.
///
/// @param graph The input unweighted graph.
/// @param weight The weight to assign to all edges.
/// @return A weighted graph with the same structure and uniform weights.
[[nodiscard]] WeightedGraph MakeWeightedGraph(Graph<int, int> const& graph, int weight);

} // namespace mochi
