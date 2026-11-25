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
#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>

namespace mochi {

/// @brief A run of consecutive child overlap indices mapping to consecutive parent positions.
struct IndexRange {
  int childStart; ///< Start index in the child's overlap-index list.
  int parentStart; ///< Corresponding start index in the parent's index space.
  int length; ///< Number of consecutive indices in this run.
};

/**
 * @brief Organizes and optimizes the frontal matrix operations in a multi-frontal solver.
 *
 * FrontalOrganizer analyzes the elimination tree to determine computational costs
 * and organize the parent-child relationships between supernodes. It helps optimize
 * the assembly and factorization process by tracking how indices from child supernodes
 * map into their parent supernodes.
 */
class FrontalOrganizer {
 public:
  /**
   * @brief Cost metrics for a supernode operation.
   *
   * @details Times count operations on nodes, not scalars: multiply them by dofsPerNode^3 for
   * scalar work.
   */
  struct Costs {
    double time; ///< Estimated time to eliminate the super-node's subtree, without front assembly.
    /// Estimated time to assemble the super-node's children's fronts and eliminate the super-node.
    double ownTime;
    size_t frontSize; ///< Number of elements in the super-node's front.
    size_t frontStackSize; ///< Number of elements for the full front stack.
    int numPanels; ///< Number of panels, the blocks of the super-node's L columns and front.
  };

  /**
   * @brief Constructs a frontal organizer for the given elimination tree.
   *
   * Analyzes the tree structure to compute costs and index mappings for
   * efficient frontal matrix operations.
   *
   * @param tree The elimination data defining the supernode structure
   * @param blockSize The block size used for stair matrix organization
   * @param dofsPerNode Degrees of freedom per node in the mesh
   */
  explicit FrontalOrganizer(
      SymbolicEliminationTree const& tree,
      size_t blockSize,
      size_t dofsPerNode);

  [[nodiscard]] auto& GetCosts() const {
    return _costs;
  }

  [[nodiscard]] auto& GetCosts(int superNode) const {
    return _costs[superNode];
  }

  [[nodiscard]] auto GetStackSize(int superNode) const {
    return _costs[superNode].frontStackSize;
  }

  [[nodiscard]] auto GetIndicesInParent(int superNode) const {
    return _indicesInParent[superNode];
  }

  /// @brief Returns the compressed index ranges mapping @p superNode's overlap indices into its
  /// parent.
  [[nodiscard]] auto GetRangesInParent(int superNode) const {
    return _rangesInParent[superNode];
  }

  [[nodiscard]] auto GetConstants() const {
    return std::pair{_blockSize, _dofsPerNode};
  }

  /// @brief Number of branches @ref PickBranches targets per thread, so uneven branches still
  /// balance.
  static constexpr int kBranchesPerThread = 2;

  /**
   * @brief Number of tasks that factorize @p superNode's panels in parallel.
   *
   * @details One task per @p minTaskTime of the super-node's own time (@ref Costs::ownTime),
   * between 1 and the smaller of @p maxTasks and its number of panels.
   *
   * @param superNode The super-node.
   * @param minTaskTime Positive minimum estimated time of a task.
   * @param maxTasks Positive maximum number of tasks.
   * @return The number of tasks.
   */
  [[nodiscard]] int NumPanelTasks(int superNode, double minTaskTime, int maxTasks) const;

  /**
   * @brief Picks a set of super-nodes from the elimination tree for parallel factorization.
   *
   * @details Each picked super-node is a root of a branch in the elimination tree.
   * Each of these branches can be factorized in parallel by a single thread without
   * any communication or synchronization with other threads.
   * Traverses the tree from the root down. A super-node is picked if its estimated factorization
   * time is at most the root's estimated time divided by @ref kBranchesPerThread * @p numThreads,
   * or if its subtree has no parallelism: it is a chain, such as a leaf, whose non-leaf super-nodes
   * each get one task from @ref NumPanelTasks.
   *
   * @param tree The elimination tree.
   * @param rootNode The root node for which to pick branches.
   * @param numThreads Number of threads to balance the branches across and to cap panel tasks.
   * @param minTaskTime Positive minimum estimated time of a panel task.
   * @return std::pair<DynamicArray<int>, int> Picked supernode indices in increasing order and the
   * number of nodes in the truncated tree
   */
  [[nodiscard]] std::pair<DynamicArray<int>, int> PickBranches(
      SymbolicEliminationTree const& tree,
      int rootNode,
      int numThreads,
      double minTaskTime) const;

 private:
  size_t _blockSize;
  size_t _dofsPerNode;
  /// @brief Computational costs for each supernode
  DynamicArray<Costs> _costs;

  /// @brief Mapping of each supernode's indices into its parent's index space.
  /// For each supernode, stores the positions where its indices appear in the parent,
  /// excluding the indices that the child owns (only the overlap/shared indices).
  Graph<int, int> _indicesInParent;

  /// @brief Mapping of each supernode's indices into parent's index space as @ref IndexRange runs.
  Graph<IndexRange, int> _rangesInParent;
};

} // namespace mochi
