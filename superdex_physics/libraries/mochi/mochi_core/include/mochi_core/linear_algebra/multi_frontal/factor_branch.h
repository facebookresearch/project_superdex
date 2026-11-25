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
#include "truncated_tree.h"

#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/linear_algebra/multi_frontal/l_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/organizer.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/task_scheduler.h>

#include <atomic>

namespace mochi {

/** @brief Full set of data necessary to pass to each stage of multi-frontal factorization
 * functions.
 *
 * @details Utility methods that essentially format data to a suitable
 * representation are also in this structure.
 */
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
struct FactorData {
  SymbolicEliminationTree const& eTree;
  FrontalOrganizer const& organizer;
  AssemblyHelper& helper;
  BlockSparseMatrixView<Scalar const, kDofsPerNode> A;
  LMatrix<Scalar, kBlockSize>& L;
  int numThreads;

  /// Positive minimum estimated time of a panel task, in the organizer's time units.
  double minTaskTime;

  FactorData(
      SymbolicEliminationTree const& eTree,
      FrontalOrganizer const& organizer,
      AssemblyHelper& helper,
      BlockSparseMatrixView<Scalar const, kDofsPerNode> const& A,
      LMatrix<Scalar, kBlockSize>& L,
      int nThreads,
      double minTaskTime)
      : eTree(eTree),
        organizer(organizer),
        helper(helper),
        A(A),
        L(L),
        numThreads(nThreads),
        minTaskTime(minTaskTime) {
    MOCHI_ASSERT_VERBOSE(minTaskTime > 0, "The minimum time of a panel task must be positive.");
  }

  void FactorSubTree(int rootNode, Span<Scalar> rootSpace) {
    FactorSubtree<kDofsPerNode>(eTree, organizer, L, A, helper, rootNode, rootSpace);
  }

  /**
   * @brief Allocate a front buffer for a given root node.
   *
   * @param rootNode The root node for which to retrieve the buffer.
   * @return A dynamically allocated buffer for the front.
   */
  auto GetFrontBuffer(int rootNode) {
    auto const& costs = organizer.GetCosts();
    DynamicArray<Scalar> buffer;
    buffer.resize_noinit(costs[rootNode].frontSize);
    return std::move(buffer);
  }

  auto
  GetTrunkWork(int rootNode, Span<Span<Scalar>> childrenFrontBuffers, Span<Scalar> rootBuffer) {
    return TrunkWork<Scalar, kBlockSize>(
        rootNode, L, kDofsPerNode, helper, eTree, organizer, childrenFrontBuffers, rootBuffer);
  }

  [[nodiscard]] DynamicArray<Scalar> DispatchTrunkWork(
      int rootNode,
      Span<Span<Scalar>> childrenFrontBuffers,
      DynamicArray<Scalar>& scratch);
};

/**
 * @brief Eliminate rootNode's DOFs.
 * @details The elimination is parallelized over @ref FrontalOrganizer::NumPanelTasks tasks,
 * allowing threads to work on separate panels of the rootNode's L and of its frontal matrix.
 * @param rootNode Node being factored.
 * @param childrenFrontBuffers Array of front buffers for children nodes.
 * @param scratch The calling task's scratch matrix for @ref TrunkWork::EliminatePanel, which the
 * last panel task borrows and sizes on first use.
 * @return A buffer containing rootNode's front.
 */
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
DynamicArray<Scalar> FactorData<kDofsPerNode, Scalar, kBlockSize>::DispatchTrunkWork(
    int rootNode,
    Span<Span<Scalar>> childrenFrontBuffers,
    DynamicArray<Scalar>& scratch) {
  auto frontBuffer = GetFrontBuffer(rootNode);
  auto trunkWork = GetTrunkWork(rootNode, childrenFrontBuffers, frontBuffer);
  MOCHI_ASSERT_VERBOSE(
      trunkWork.GetNumPanels() == organizer.GetCosts(rootNode).numPanels,
      "The organizer must count the panels of the trunk work.");

  // Each panel task needs its own scratch matrix. The last one, which ParallelForN runs on the
  // calling thread, borrows the calling task's, so a long trunk of single-task supernodes allocates
  // one only once.
  int const numTasks = organizer.NumPanelTasks(rootNode, minTaskTime, numThreads);
  auto factorByPanel = [&](int task) {
    if (!trunkWork.EnterWork()) {
      return;
    }

    DynamicArray<Scalar> taskScratch;
    auto& uBuffer = task == numTasks - 1 ? scratch : taskScratch;
    uBuffer.resize_noinit(kBlockSize * kBlockSize);
    do {
      // Look for panels to work on. The panels in L complete sequentially, but after that
      // The remaining panels can complete in any order. The start of the iteration reflects this
      // point.
      for (int p =
               Min(trunkWork.completedPanelCount.load(std::memory_order_relaxed),
                   trunkWork.numPanelsInL);
           p < trunkWork.GetNumPanels();
           ++p) {
        trunkWork.template WorkOnPanel<kDofsPerNode>(p, A, MakeSpan(uBuffer));
      }
    } while (!trunkWork.LeaveWork());
  };

  ParallelForN("TrunkWork", numTasks, /*minPerTask=*/1, factorByPanel);
  return frontBuffer;
}

/**
 * @brief Factor @p root's tree: the branches in parallel, then each trunk supernode once its
 * children are factored.
 *
 * @details The last task to finish one of a trunk supernode's children factors the supernode, then
 * continues to the supernode's parent, so a chain of single-child supernodes runs in one task.
 * When other threads are available, tasks claim branches in the order of @p branchRoots, except
 * that the calling thread takes the last one first.
 *
 * @param data The factorization data of the matrix.
 *
 * @param root The root of the tree to factor.
 *
 * @param branchRoots The roots of the branches. Their subtrees are disjoint and, with the trunk,
 * their ancestors up to @p root, tile @p root's tree.
 */
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
void FactorTree(
    FactorData<kDofsPerNode, Scalar, kBlockSize>& data,
    int root,
    Span<int const> branchRoots) {
  auto const& eTree = data.eTree;
  // Post-ordering numbers root's subtree from firstNode to root.
  int const firstNode = eTree.SubtreeRange(root).Min();
  // A supernode's front is released once its parent is factored.
  DynamicArray<DynamicArray<Scalar>> fronts(root - firstNode + 1);
  DynamicArray<std::atomic<int>> numFactoredChildren(root - firstNode + 1);

  auto factorBranch = [&](int branchRoot) {
    auto& branchFront = fronts[branchRoot - firstNode];
    branchFront = data.GetFrontBuffer(branchRoot);
    data.FactorSubTree(branchRoot, branchFront);

    DynamicArray<Scalar> scratch;
    for (int node = branchRoot; node != root;) {
      node = eTree.SuperParent(node);
      auto const children = eTree.Children(node);
      // Acquire the other children's fronts and release this task's to the last child's task.
      if (numFactoredChildren[node - firstNode].fetch_add(1, std::memory_order_acq_rel) + 1 <
          isize(children)) {
        return;
      }

      DynamicArray<Span<Scalar>> childFronts;
      childFronts.reserve(children.size());
      for (int child : children) {
        childFronts.push_back(fronts[child - firstNode]);
      }
      fronts[node - firstNode] = data.DispatchTrunkWork(node, childFronts, scratch);
      for (int child : children) {
        fronts[child - firstNode].reset();
      }
    }
  };

  ParallelForRange(
      "FactorBranch",
      0,
      isize(branchRoots),
      /*minPerTask=*/1,
      /*maxPerTask=*/1,
      [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
          factorBranch(branchRoots[i]);
        }
      });
}

} // namespace mochi
