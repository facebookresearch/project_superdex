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
#include "trunk.h"

#include <mochi_core/async/executor.h>
#include <mochi_core/async/task.h>
#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/linear_algebra/multi_frontal/elim_tree.h>
#include <mochi_core/linear_algebra/multi_frontal/l_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/organizer.h>
#include <mochi_core/utils/debug.h>

#include <algorithm>

namespace mochi {

/** @brief Full set of data necessary to recursively pass to each stage
 * of multi-frontal factorization functions.
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
  DynamicArray<int> branchRoots;
  int numThreads;

  /// Positive minimum estimated time of a panel task, in the organizer's time units.
  double minTaskTime;

  FactorData(
      SymbolicEliminationTree const& eTree,
      FrontalOrganizer const& organizer,
      AssemblyHelper& helper,
      BlockSparseMatrixView<Scalar const, kDofsPerNode> const& A,
      LMatrix<Scalar, kBlockSize>& L,
      Span<int const> branchRoots,
      int nThreads,
      double minTaskTime)
      : eTree(eTree),
        organizer(organizer),
        helper(helper),
        A(A),
        L(L),
        branchRoots(branchRoots.begin(), branchRoots.end()),
        numThreads(nThreads),
        minTaskTime(minTaskTime) {
    MOCHI_ASSERT_VERBOSE(minTaskTime > 0, "The minimum time of a panel task must be positive.");
    std::ranges::sort(this->branchRoots);
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

  /**
   * @brief Check if a node is a root of a branch to be factored by a single
   * thread.
   *
   * @param node The node to check.
   * @return True if the node is a branch root, false otherwise.
   */
  bool IsBranchRoot(int node) {
    return std::ranges::binary_search(branchRoots, node);
  }

  auto
  GetTrunkWork(int rootNode, Span<Span<Scalar>> childrenFrontBuffers, Span<Scalar> rootBuffer) {
    return TrunkWork<Scalar, kBlockSize>(
        rootNode, L, kDofsPerNode, helper, eTree, organizer, childrenFrontBuffers, rootBuffer);
  }

  Task<DynamicArray<Scalar>>
  DispatchTrunkWork(auto& executor, int rootNode, Span<Span<Scalar>> childrenFrontBuffers);
};

/**
 * @brief Eliminate rootNode's DOFs.
 * @details The elimination is parallelized over @ref FrontalOrganizer::NumPanelTasks tasks,
 * allowing threads to work on separate panels of the rootNode's L and of its frontal matrix.
 * @param exec The executor to dispatch parallel panel work to.
 * @param rootNode Node being factored.
 * @param childrenFrontBuffers Array of front buffers for children nodes.
 * @return A buffer containing rootNode's front.
 */
template <int kDofsPerNode, typename Scalar, size_t kBlockSize>
Task<DynamicArray<Scalar>> FactorData<kDofsPerNode, Scalar, kBlockSize>::DispatchTrunkWork(
    auto& exec,
    int rootNode,
    Span<Span<Scalar>> childrenFrontBuffers) {
  auto frontBuffer = GetFrontBuffer(rootNode);
  auto trunkWork = GetTrunkWork(rootNode, childrenFrontBuffers, frontBuffer);
  MOCHI_ASSERT_VERBOSE(
      trunkWork.GetNumPanels() == organizer.GetCosts(rootNode).numPanels,
      "The organizer must count the panels of the trunk work.");

  auto factorByPanel = [&]() {
    if (!trunkWork.EnterWork()) {
      return false;
    }
    // Allocate buffer for EliminatePanel
    auto maxPanelCols = kBlockSize;
    DynamicArray<Scalar> uBuffer;
    uBuffer.resize_noinit(maxPanelCols * maxPanelCols);
    bool didCompleteWork = false;
    do {
      // Look for panels to work on. The panels in L complete sequentially, but after that
      // The remaining panels can complete in any order. The start of the iteration reflects this
      // point.
      for (int p =
               Min(trunkWork.completedPanelCount.load(std::memory_order_relaxed),
                   trunkWork.numPanelsInL);
           p < trunkWork.GetNumPanels();
           ++p) {
        didCompleteWork |= trunkWork.template WorkOnPanel<kDofsPerNode>(p, A, MakeSpan(uBuffer));
      }
    } while (!trunkWork.LeaveWork());
    return didCompleteWork;
  };

  auto tr = std::make_shared<Trunk>();
  co_await ParallelFactor(
      tr,
      exec,
      factorByPanel,
      trunkWork.GetNumPanels(),
      organizer.NumPanelTasks(rootNode, minTaskTime, numThreads));
  co_return std::move(frontBuffer);
}

/**
 * @brief Factor a branch of the elimination tree, potentially factoring
 * subbranches in parallel.
 * @details If superNode is in the set of picked roots to be factored by
 * a single thread, this function does not dispatch other work. Otherwise,
 * it walks down from superNode through single children that are not picked
 * roots, to the first node that branches or whose child is a picked root. It
 * factors that node's children in parallel, then the walked chain bottom-up,
 * without a FactorBranch task per chain node.
 * @param executor The executor to which to dispatch the children's
 * factorization tasks.
 * @param data The factorization data of the matrix.
 * @param superNode The root node of the branch to factor.
 * @return A buffer containing the resulting front of the factorization.
 */
template <typename Exec, int kDofsPerNode, typename Scalar, size_t kBlockSize>
Task<DynamicArray<Scalar>>
FactorBranch(Exec& executor, FactorData<kDofsPerNode, Scalar, kBlockSize>& data, int superNode) {
  // When the superNode was picked as the root of a branch to factorize sequentially.
  if (data.IsBranchRoot(superNode)) {
    auto frontBuffer = data.GetFrontBuffer(superNode);
    data.FactorSubTree(superNode, frontBuffer);
    co_return std::move(frontBuffer);
  }

  // Single-child trunk nodes have no sibling work to overlap, so factor a chain of them here rather
  // than with a task per node.
  auto& eTree = data.eTree;
  int chainBottom = superNode;
  auto children = eTree.Children(chainBottom);
  while (children.size() == 1 && !data.IsBranchRoot(children[0])) {
    chainBottom = children[0];
    children = eTree.Children(chainBottom);
  }

  // Recursively factorize the children in parallel. Their fronts are released before the walk up.
  DynamicArray<Scalar> front;
  {
    auto childrenFronts = co_await Execute(
        executor,
        [&](int child) -> Task<DynamicArray<Scalar>> {
          return FactorBranch(executor, data, child);
        },
        async::ForEach{children});
    auto chFrontsAsSpans = DynamicArray<Span<Scalar>>(childrenFronts.begin(), childrenFronts.end());
    front = co_await data.DispatchTrunkWork(executor, chainBottom, chFrontsAsSpans);
  }

  // Factor the chain bottom-up. Only chain nodes with enough work split their panels into tasks.
  for (int node = chainBottom; node != superNode;) {
    node = eTree.SuperParent(node);
    Span<Scalar> childFront = front;
    front = co_await data.DispatchTrunkWork(executor, node, MakeSingletonSpan(childFront));
  }
  co_return std::move(front);
}

} // namespace mochi
