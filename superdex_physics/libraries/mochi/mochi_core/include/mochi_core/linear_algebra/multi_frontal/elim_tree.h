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
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/interval.h>
#include <mochi_core/utils/range_by_iterators.h>

#include <utility>

namespace mochi {

/** @brief Metrics for a Cholesky-type factorization computed on the Elimination Tree.
 */
struct FactorMetrics {
  double opCount; ///< Total floating-point operation count
  size_t storage; ///< Total Cholesky factor L storage count (number of non-zero scalars)
};

/** @brief Elimination Tree object representing structural properties for a Cholesky-type
 * factorization.
 *
 */
class EliminationTree {
 public:
  /** @brief Build a post-ordered elimination tree for a given matrix graph.
   * @details The algorithm modifies the equation order to be a post-ordered
   * numbering. That is, for any branch of the tree, the nodes are given
   * a range of gapless consecutive numbers.
   *
   * @param graph [in] The matrix nodal graph in original numbering.
   * @param order [inout] The order of the nodes in the renumbered matrix, updated on output.
   * @param position [inout] The position of nodes in the renumbering, updated on output.
   */
  EliminationTree(
      Graph<int const, int const, Span> const& graph,
      Span<int> const& order,
      Span<int> const& position);

  [[nodiscard]] int NumSuperNodes() const {
    return isize(_snParents);
  }

  [[nodiscard]] int NumNodes() const {
    return _snBounds.back();
  }

  /// @brief Get the parent of a super-node.
  [[nodiscard]] int SuperParent(int superNode) const {
    return _snParents[superNode];
  }

  [[nodiscard]] auto const& SuperBounds() const {
    return _snBounds;
  }

  /// @brief Get the number of nodes in a super-node.
  [[nodiscard]] auto SuperSize(int superNode) const {
    return _snBounds[superNode + 1] - _snBounds[superNode];
  }

  /// @brief Get the number of nodes in the column of L for a super-node.
  /// @details This is the number of rows in L below (and including) the supernode's
  /// diagonal block, i.e. the supernode's own size plus the off-diagonal coupling.
  [[nodiscard]] auto SuperColSize(int superNode) const {
    return _snColSizes[superNode];
  }

  /// @brief Get a range spanning the renumbered nodes included in a super-node.
  [[nodiscard]] auto SuperRange(int superNode) const {
    // The line below is really what we want but some CI compilers have bugs in their <ranges>
    // library and we will have to wait until more up-to-date compilers are available.
    // `return std::views::iota(_snBounds[superNode], _snBounds[superNode + 1]);`
    return std::pair{_snBounds[superNode], _snBounds[superNode + 1]};
  }
  /// @brief Get the array of parents.
  [[nodiscard]] auto const& SuperParents() const {
    return _snParents;
  }
  /// @brief Get the depth of the tree below a node, including the node itself.
  [[nodiscard]] auto SubtreeDepth(int superNode) const {
    return _leafDistance[superNode] + 1;
  }
  /// @brief Check if a super-node is a leaf.
  [[nodiscard]] bool IsLeaf(int superNode) const {
    return _treeGraph[superNode].empty();
  }
  /// @brief Get the graph representing the super-nodal elimination tree.
  /// @details This graph excludes the roots.
  [[nodiscard]] auto TreeGraph() const {
    auto numTg = _treeGraph.GetPointers()[NumSuperNodes()];
    return Graph<int const, int const, Span>{
        _treeGraph.GetPointers().subspan(0, NumSuperNodes() + 1),
        _treeGraph.GetTargets().subspan(0, numTg)};
  }
  /// @brief Get all the elimination tree roots.
  /// @details There is only one root if the graph is connected.
  /// If the graph represents disjoint matrices, there will be one root per matrix.
  [[nodiscard]] auto Roots() const {
    return _treeGraph[NumSuperNodes()];
  }

  /// @brief Get the children of a super-node.
  [[nodiscard]] auto Children(int superNode) const {
    return _treeGraph[superNode];
  }

  /// @brief A subtree has node in order thanks to post-ordering.
  [[nodiscard]] Interval<int> SubtreeRange(int superNode) const;

  /// @brief Compute both floating-point operation count and storage requirement.
  ///
  /// @details The cost is summed over all super-nodes. For a super-node with @c nInSuper
  /// DOFs (its own DOFs) and @c nCoupling DOFs in the rows below it (the off-diagonal
  /// coupling), the per-super-node cost is the sum of three classical multifrontal terms:
  ///   - Factoring the diagonal block:        @c (2/3) * nInSuper^3
  ///   - Computing L below the block diagonal: @c nInSuper^2 * nCoupling
  ///   - Updating the Schur complement:        @c nCoupling^2 * nInSuper
  ///
  /// @param dofsPerNode Number of degrees of freedom on each node (assumed uniform).
  ///
  /// @return The metrics including total operation count (as @c double so it can represent
  /// counts that exceed the 64-bit integer range for very large factorizations) and storage.
  [[nodiscard]] FactorMetrics ComputeFactorMetrics(int dofsPerNode) const;

 protected:
  DynamicArray<int> _snBounds;
  DynamicArray<int> _snParents;
  DynamicArray<int> _leafDistance;
  /// @brief Graph object representing the tree, with the last index returning the root supernodes.
  Graph<int, int> _treeGraph;
  /// @brief Storage for column sizes computed before symbolic factorization.
  DynamicArray<int> _snColSizes;
};

/** @brief Full Elimination Tree object containing structural properties and symbolic factorization
 * data.
 *
 */
class SymbolicEliminationTree : public EliminationTree {
 public:
  SymbolicEliminationTree(
      Graph<int const, int const, Span> const& graph,
      Span<int> const& order,
      Span<int> const& position);

  /// @brief Get the list of indices in L for the first node of a super-node.
  [[nodiscard]] auto SuperIndices(int superNode) const {
    return _snIndices[superNode];
  }
  /// @brief Get the indices in L below the diagonal block of a super-node, i.e. the
  /// off-diagonal coupling rows (the supernode's own DOFs are excluded).
  [[nodiscard]] auto LowerIndices(int superNode) const {
    return _snIndices[superNode].subspan(SuperSize(superNode));
  }
  /// @brief Get a graph with all the super-node indices of L.
  [[nodiscard]] auto const& SuperIndices() const {
    return _snIndices;
  }

 private:
  Graph<int, size_t, DynamicArray> _snIndices;
};

} // namespace mochi
