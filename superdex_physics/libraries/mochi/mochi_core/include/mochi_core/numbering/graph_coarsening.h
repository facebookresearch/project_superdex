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

#include <mochi_core/numbering/weighted_graph.h>
#include <mochi_core/utils/graph_utils.h>

namespace mochi {

/// @brief Result of a matching operation.
struct MatchResult {
  /// @brief Graph where each match group (source) maps to its constituent vertices (targets).
  Graph<int, int> match;
  /// @brief For each vertex, the index of the match group it belongs to.
  DynamicArray<int> matchedIn;
};

/// @brief Computes a heavy edge matching of a weighted graph.
///
/// @details Vertices are visited in order of increasing vertex mass. For each unmatched vertex,
/// the unmatched neighbor with the highest edge weight is selected for matching. Both vertices are
/// assigned to the same match group. Vertices with no unmatched neighbors form singleton groups.
///
/// @param graph The input weighted graph.
/// @param vertexMass The mass (number of fine vertices) associated with each vertex.
///
/// @return The matching result containing both the match graph and the vertex-to-group mapping.
[[nodiscard]] MatchResult HeavyEdgeMatch(WeightedGraph const& graph, Span<int const> vertexMass);

/// @brief Coarsens a graph by collapsing vertices according to a matching.
///
/// @details Edge weights between coarse vertices are the sum of edge weights between their
/// constituent fine vertices.
///
/// @param fineGraph The original weighted graph.
/// @param matchResult The matching result from @ref HeavyEdgeMatch applied to fineGraph.
///
/// @return A coarsened weighted graph where each vertex corresponds to a match group.
[[nodiscard]] WeightedGraph CoarsenGraph(
    WeightedGraph const& fineGraph,
    MatchResult const& matchResult);

/// @brief Result of a full graph coarsening operation (matching + graph reduction + mass
/// computation).
struct CoarseningResult {
  /// @brief Information about the match from which the coarsened graph was built.
  MatchResult groups;
  /// @brief The coarsened graph.
  WeightedGraph graph;
  /// @brief For each vertex, the number of vertices in the original graph that it represents.
  DynamicArray<int> coarseMass;
};

/// @brief Builds a coarsened graph by collapsing vertices according to a heavy edge matching.
///
/// @details Edge weights between coarse vertices are the sum of edge weights between their
/// constituent fine vertices.
///
/// @param graph The input weighted graph.
/// @param vertexMass The number of fine vertices in the input graph vertices.
///
/// @return The coarsening result containing the match, the coarsened graph, and the accumulated
/// coarse masses.
[[nodiscard]] CoarseningResult BuildCoarsenedGraph(
    WeightedGraph const& graph,
    Span<int const> vertexMass);

/// @brief Iteratively coarsens a graph until it reaches a target size.
///
/// @details Repeatedly applies heavy edge matching and graph coarsening until the graph has at most
/// @p maxVtx vertices, or until coarsening progress stalls (less than 15% reduction per level).
/// Each level's coarseMass accumulates from the original fine graph, so the sum of coarseMass
/// at any level equals the original vertex count.
///
/// @param graph The input weighted graph.
/// @param vertexMass The mass (number of fine vertices) associated with each vertex in @p graph.
/// @param maxVtx Target maximum number of vertices in the coarsest graph.
///
/// @return A sequence of coarsening levels from finest to coarsest. Empty if graph already has
/// at most @p maxVtx vertices.
[[nodiscard]] DynamicArray<CoarseningResult>
CoarsenTo(WeightedGraph const& graph, Span<int const> vertexMass, int maxVtx);

/// @brief Projects a per-vertex side assignment from a coarse graph back to its finer graph.
///
/// @details Each fine vertex inherits the side of the coarse vertex that contains it.
///
/// @param coarseSide The side assigned to each vertex of the coarse graph.
/// @param matchedIn For each fine vertex, the index of the coarse vertex it was matched into
/// (as produced by @ref HeavyEdgeMatch).
///
/// @return The side assigned to each vertex of the fine graph.
[[nodiscard]] DynamicArray<int> ProjectSide(Span<int const> coarseSide, Span<int const> matchedIn);

} // namespace mochi
