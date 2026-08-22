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
#include <mochi_core/utils/array.h>
#include <mochi_core/utils/span.h>

namespace mochi {

/// @brief Side index for vertices on the first side of a bisection.
constexpr int kSide0 = 0;
/// @brief Side index for vertices on the second side of a bisection.
constexpr int kSide1 = 1;
/// @brief Side index for vertices belonging to the separator.
constexpr int kSeparatorSide = 2;

/// @brief Result of a graph dissection.
struct DissectionResult {
  /// @brief For each vertex, which side of the partition it belongs to (@ref kSide0 or @ref
  /// kSide1).
  DynamicArray<int> side;
  /// @brief Total weight of edges crossing the partition (cut weight).
  int cutWeight;
  /// @brief Whether the partition satisfies the balance constraint (selectedMass in [lowerMass,
  /// upperMass]).
  bool balanced;
};

/// @brief Partitions a weighted graph into two sets while minimizing cut edge weight.
///
/// @details Uses the Greedy Graph Growing Partition (GGGP) heuristic to produce a balanced
/// partition with a low (not necessarily optimal) cut edge weight. The algorithm grows a
/// partition from multiple seed vertices and selects the best result.
///
/// @note A separator optimization is expected to refine the result.
/// @note The algorithm is not guaranteed to find a solution satisfying the maximum imbalance
/// constraint.
///
/// @param graph The input weighted graph.
/// @param vertexMass The mass associated with each vertex.
/// @param totalMass The sum of all vertex masses.
/// @param maxUnbalance Maximum allowed difference in mass between the two sides.
///
/// @return A dissection result containing the partition assignment and cut weight.
[[nodiscard]] DissectionResult DissectGraph(
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int totalMass,
    int maxUnbalance);

/// @brief Constructs a trivial vertex separator from a bisection by promoting boundary vertices
/// from both sides.
///
/// @details Scans all vertices and promotes boundary vertices (those with at least one neighbor
/// on the opposite side) to the separator by setting their side to @ref kSeparatorSide.
///
/// @param side The partition assignment for each vertex (@ref kSide0 or @ref kSide1). Modified in
/// place to mark separator vertices as @ref kSeparatorSide.
/// @param graph The input weighted graph.
void MarkTrivialSeparatorBothSides(Span<int> side, WeightedGraph const& graph);

/// @brief Constructs a trivial vertex separator from one side of a bisection.
///
/// @details Scans vertices on the specified side and promotes boundary vertices (those with at
/// least one neighbor on the opposite side) to the separator by setting their side to
/// @ref kSeparatorSide.
///
/// @param side The partition assignment for each vertex (@ref kSide0 or @ref kSide1). Modified in
/// place to mark separator vertices as @ref kSeparatorSide.
/// @param graph The input weighted graph.
/// @param vertexMass The mass associated with each vertex.
/// @param whichSide Which side (@ref kSide0 or @ref kSide1) to pick separator vertices from.
///
/// @return The total mass of the vertices moved to the separator.
[[nodiscard]] int MarkTrivialSeparatorOneSide(
    Span<int> side,
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int whichSide);

/// @brief Optimizes a valid separator to minimize its total vertex mass.
///
/// @details Takes an initial valid partition with a separator and refines it by moving
/// vertices to minimize the separator mass, while respecting the maximum allowed unbalance,
/// or moving toward a balanced partition if the initial partition is already valid.
/// The optimization picks the best vertex to move to either side, recursively pulling
/// necessary neighbors into the separator to maintain validity.
///
/// @param side The current partition assignment (@ref kSide0, @ref kSide1, or @ref
/// kSeparatorSide). Modified in place.
/// @param graph The input weighted graph.
/// @param vertexMass The mass associated with each vertex.
/// @param maxUnbalance Maximum allowed difference in mass between sides @ref kSide0 and @ref
/// kSide1. If the input partition is already imbalanced beyond this bound, the function only
/// guarantees that the resulting imbalance does not increase (it is not forced down to
/// @p maxUnbalance).
///
/// @return The total mass of the vertices in the separator after optimization.
[[nodiscard]] int OptimizeSeparator(
    Span<int> side,
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int maxUnbalance);

/// @brief Builds an optimized separator from a given partition.
///
/// @details Creates a minimum vertex-mass separator for the graph, using an initial partition.
/// The output separator is a subset of vertices whose removal splits the graph into two
/// balanced components. The algorithm iteratively improves the separator by moving vertices
/// between the separator and the two components, while maintaining the balance constraint.
///
/// @param side The initial partition assignment for each vertex (@ref kSide0 or @ref kSide1). On
/// return, contains the final partition with separator vertices marked as @ref kSeparatorSide.
/// @param graph The input weighted graph.
/// @param vertexMass The mass associated with each vertex.
/// @param maxUnbalance Maximum allowed difference in mass between the two sides.
///
/// @return A dynamic array containing the indices of the vertices in the separator.
[[nodiscard]] DynamicArray<int> BuildOptimizedSeparator(
    Span<int> side,
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int maxUnbalance);

/// @brief Result of a graph trisection (two-way partition with a vertex separator).
struct TrisectionResult {
  /// @brief Partition assignment for each vertex of the input graph ( @ref kSide0, @ref kSide1, or
  /// @ref kSeparatorSide).
  DynamicArray<int> side;
  /// @brief Indices of the vertices in the separator (i.e. vertices with @c side ==
  /// @ref kSeparatorSide).
  DynamicArray<int> separator;
};

/// @brief Partitions a weighted graph into two balanced sets and a vertex separator using
/// multilevel coarsening.
///
/// @details The algorithm proceeds in three phases:
/// 1. Coarsen the graph until it has at most @c kCoarseTargetSize vertices (currently 100).
/// 2. Bisect the coarsest graph with @ref DissectGraph, project the bisection one level up,
///    and turn it into a vertex separator with @ref BuildOptimizedSeparator.
/// 3. Recursively project the trisection back to the finest level. The separator is refined
///    via @ref OptimizeSeparator every @c kOptLevels projections, and always at the finest level.
///
/// @note The algorithm is not guaranteed to find a solution satisfying the maximum imbalance.
///
/// @param graph The input weighted graph.
/// @param vertexMass The mass associated with each vertex of @p graph.
/// @param maxUnbalance Maximum allowed difference in mass between sides 0 and 1.
///
/// @return A trisection result containing the side assignment and the separator vertex indices.
[[nodiscard]] TrisectionResult
Trisect(WeightedGraph const& graph, Span<int const> vertexMass, int maxUnbalance);

/// @brief One side of a trisection, expressed as an induced weighted subgraph.
struct SideSubgraph {
  /// @brief Subgraph induced by the vertices on this side. Edges to vertices on the other
  /// side or in the separator are dropped. Self-loops are preserved.
  WeightedGraph graph;
  /// @brief Vertex masses in the new (subgraph) ordering: @c vertexMass[newV] is the mass of
  /// @c oldVertex[newV] in the original graph.
  DynamicArray<int> vertexMass;
  /// @brief Mapping from new vertex index to original vertex index. Has size
  /// @c graph.size(); strictly increasing.
  DynamicArray<int> oldVertex;
};

/// @brief Extracts the induced weighted subgraphs for sides 0 and 1 of a trisection.
///
/// @details Builds, for each side, the subgraph induced by the vertices assigned to that
/// side. Edges from a side-N vertex to a vertex in the separator (side 2) or on the opposite
/// side are dropped; edge weights on retained edges are preserved unchanged. Self-loops are
/// kept.
///
/// @param graph The original weighted graph.
/// @param vertexMass Vertex masses for @p graph, indexed by original vertex.
/// @param side Side assignment for each vertex of @p graph (0, 1, or 2 for separator), as
/// produced by @ref Trisect.
///
/// @return An array of two @ref SideSubgraph values for sides 0 and 1, respectively.
[[nodiscard]] Array<SideSubgraph, 2>
ExtractSideSubgraphs(WeightedGraph const& graph, Span<int const> vertexMass, Span<int const> side);

/// @brief Recursive nested-dissection ordering of a weighted graph.
///
/// @details Recursively applies @ref Trisect to the input graph and to each of its two side
/// subgraphs (built via @ref ExtractSideSubgraphs) until the subgraph has fewer than @p
/// leafSize vertices, at which point the leaf's vertices are emitted in their current order.
/// At every recursive level, the returned permutation places the two sides first (each in
/// its own recursive nested-dissection order) followed by the separator vertices. This is
/// the classical nested-dissection ordering that minimizes fill-in for sparse direct solvers.
///
/// The maximum allowed mass imbalance is scaled at each level: a subgraph with total mass
/// @c subMass uses @c maxUnbalance * subMass / totalMass, where @c totalMass is the total
/// mass of the top-level graph. This keeps the relative imbalance constraint constant
/// throughout the recursion.
/// The vertex masses should not be zero. The notion of mass balance looses meaning if they
/// are.
///
/// @param graph The input weighted graph.
/// @param vertexMass The mass associated with each vertex of @p graph.
/// @param maxUnbalance Maximum allowed mass difference between sides 0 and 1 at the top
/// level. Scaled proportionally for each recursive call.
/// @param leafSize Recursion threshold. Subgraphs with strictly fewer vertices are emitted
/// in identity order. Must be larger than 1.
///
/// @return A permutation @c perm of length @c graph.size() such that @c perm[i] is the
/// original vertex that occupies position @c i in the nested-dissection ordering.
[[nodiscard]] DynamicArray<int> RenumberByDissection(
    WeightedGraph const& graph,
    Span<int const> vertexMass,
    int maxUnbalance,
    int leafSize);

} // namespace mochi
