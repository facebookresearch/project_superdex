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

#include <mochi_core/linear_algebra/block_one_d_view.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/assembly_helper.h>
#include <mochi_core/linear_algebra/multi_frontal/front_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/front_operations.h>
#include <mochi_core/linear_algebra/multi_frontal/l_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/organizer.h>
#include <mochi_core/linear_algebra/multi_frontal/stair_matrix.h>
#include <mochi_core/utils/dynamic_array.h>

#include <atomic>
#include <cstdint>

namespace mochi {

/// @brief A block of columns within a supernode's L column or Schur complement (front).
///
/// @details In the panel-based parallel factorization, a supernode's staircase matrix
/// and front are subdivided into panels. Each panel tracks its own assembly and
/// factorization progress via atomic state variables, enabling concurrent
/// processing by multiple tasks.
///
/// @tparam Scalar The scalar type of the matrix elements.
template <typename Scalar>
struct Panel {
  // One bit is used to indicate whether the panel is being worked on by a task.
  static constexpr uint32_t kAcquired = uint32_t{1} << 31;
  // One bit is used to indicate whether the panel has been assembled.
  static constexpr uint32_t kAssembled = uint32_t{1} << 30;
  // Mask for the remainder.
  static constexpr uint32_t kMaskLevel = ~(kAcquired | kAssembled);

  Panel() = default;
  Panel(Panel&& other) noexcept
      : indexInSuperNode(other.indexInSuperNode),
        updateState(other.updateState.load()),
        nodeIndices(other.nodeIndices),
        panel(other.panel) {}

  /// @brief Offset of this panel's first column node within the supernode's index list.
  int indexInSuperNode = 0;
  /// @brief The state of the panel update.
  /// @details Bit 31 (kAcquired): set while a task is actively working on this panel.
  ///          Bit 30 (kAssembled): set once input matrix and children fronts have been scattered.
  ///          Bits 0-29 (kMaskLevel): number of L-panel updates applied so far.
  std::atomic<uint32_t> updateState = 0;
  /// @brief The indices of the nodes in the panel rows.
  /// @details The first indices are some of the super-node's nodes, starting
  ///          with the nodes of the columns of the panel.
  Span<int const> nodeIndices;
  /// @brief The panel matrix.
  MatrixView<Scalar> panel;
};

/// @brief Represents a child supernode's Schur complement to be assembled into its parent.
///
/// @details After factoring a child supernode, the remaining Schur complement
/// (stored in a @ref Front) must be scattered into the parent supernode's panels.
/// This struct bundles the child's front with the index mappings needed to
/// locate where each child row/column maps in the parent's index space.
///
/// @tparam kColumnBlock The block size for the front matrix layout.
template <size_t kColumnBlock>
struct ChildFront {
  int superNode{}; ///< Index of the child supernode.
  Span<int const> nodeIndices; ///< Node indices of the child's front rows/columns.
  /// @brief The indices of the child's nodes in the parent's node indices.
  Span<int const> indicesInParent;
  Front<kColumnBlock> front; ///< The child's Schur complement front matrix.
};

/// @brief Manages the panel-by-panel LDLt factorization of a single supernode.
///
/// @details This class is used to factor the truncated tree in a parallel fashion.
/// A supernode's L column is decomposed into panels (blocks of the staircase matrix).
/// Each panel can be processed independently in a left-looking fashion:
///   1. AddInputMatrix: assemble input matrix entries into the panel.
///   2. AssembleChildrenFronts: scatter children's Schur complements into the panel.
///   3. EliminatePanel: apply left-looking updates from previously factored panels.
///   4. FactorPanel: perform the LDLt factorization of the panel's diagonal block
///      and solve for the sub-diagonal.
///
/// Panels beyond the L columns represent the supernode's Schur complement (front),
/// which receives updates from L panels but is not factored at this super-node level.
///
/// @tparam Scalar The scalar type of the matrix elements.
/// @tparam kColumnBlock The block size for the staircase matrix and front matrix.
template <typename Scalar, size_t kColumnBlock>
class TrunkWork {
 public:
  /// @brief Construct TrunkWork for a supernode.
  ///
  /// @param superNode Index of the supernode to factor.
  /// @param lMatrix The L matrix storage containing the staircase for this supernode.
  /// @param dofsPerNode Number of degrees of freedom per node.
  /// @param helper Assembly helper providing node-to-matrix mappings.
  /// @param eTree The elimination tree providing supernode structure.
  /// @param organizer The frontal organizer providing parent-child index mappings.
  /// @param childrenFrontBuffers Pre-allocated buffers for each child's Schur complement.
  /// @param frontBuffer Pre-allocated buffer for this supernode's own Schur complement.
  TrunkWork(
      int superNode,
      LMatrix<Scalar, kColumnBlock>& lMatrix,
      int dofsPerNode,
      AssemblyHelper const& helper,
      SymbolicEliminationTree const& eTree,
      FrontalOrganizer const& organizer,
      Span<Span<Scalar>> childrenFrontBuffers,
      Span<Scalar> frontBuffer);

  /// @brief Returns the total number of panels (L panels + front panels).
  [[nodiscard]] int GetNumPanels() const {
    return isize(panels);
  }
  [[nodiscard]] auto& operator[](int i) {
    return panels[i];
  }
  [[nodiscard]] auto const& operator[](int i) const {
    return panels[i];
  }
  [[nodiscard]] auto const& GetChildrenFronts() const {
    return childrenFronts;
  }

  /// @brief Work on a panel through all the necessary steps, as far as possible.
  /// @details This method is thread safe and can be called concurrently by multiple tasks.
  /// Only one task will do actual work on a panel at any given time.
  /// @param panelIndex Index of the panel to work on.
  /// @param A The block-sparse input matrix.
  /// @param uBuffer Scratch buffer of size >= source.Cols() * target.Cols().
  /// @return True if the last panel was completed.
  template <int kDofsPerNode>
  bool WorkOnPanel(
      int panelIndex,
      BlockSparseMatrixView<Scalar const, kDofsPerNode> const& A,
      Span<Scalar> uBuffer);

  /// @brief Assemble input matrix entries into a panel's L column.
  /// @param panelIndex Index of the panel to assemble into.
  /// @param A The block-sparse input matrix.
  template <int kDofsPerNode>
  void AddInputMatrix(int panelIndex, BlockSparseMatrixView<Scalar const, kDofsPerNode> const& A);

  /// @brief Scatter children's Schur complement contributions into a panel.
  /// @param panelIndex Index of the target panel.
  template <int kDofsPerNode>
  void AssembleChildrenFronts(int panelIndex);

  /// @brief Apply a left-looking update from a factored source panel to a target panel.
  /// @details Computes target -= L_source * D_source^{-1} * L_source^T, where L_source
  ///          is the sub-diagonal of the source panel aligned with the target's rows.
  /// @param sourcePanelIndex Index of the previously factored source panel.
  /// @param targetPanelIndex Index of the target panel to update.
  /// @param uBuffer Scratch buffer of size >= source.Cols() * target.Cols().
  void EliminatePanel(int sourcePanelIndex, int targetPanelIndex, Span<Scalar> uBuffer);

  /// @brief Factor the diagonal block of a panel and solve for its sub-diagonal.
  /// @details Performs in-place LDLt factorization of the panel's top square block,
  ///          then applies L^{-T} and scales by D^{-1} on the sub-diagonal portion.
  /// @param panelIndex Index of the panel to factor. Must be an L panel (< numPanelsInL).
  void FactorPanel(int panelIndex);

  int FactoredPanelCount() const {
    return std::min(completedPanelCount.load(std::memory_order_acquire), numPanelsInL);
  }
  /// @brief Attempt to enter this TrunkWork. Returns false if there is not
  /// enough work for another thread.
  bool EnterWork();
  /// @brief Attempt to leave this TrunkWork.
  /// @return whether this thread left the work.
  bool LeaveWork();

  int superNode; ///< Index of the supernode being factored.
  int numPanelsInL; ///< Number of panels belonging to the L factor (factored panels).
  AssemblyHelper const& helper; ///< Assembly helper for node-to-matrix mappings.
  DynamicArray<ChildFront<kColumnBlock>> childrenFronts; ///< Children's Schur complements.
  DynamicArray<Panel<Scalar>> panels; ///< All panels (L + front).
  std::atomic<int> completedPanelCount = 0; ///< Progress counter for parallel use.
  std::atomic<int> activeWorkerCount = 0; ///< Use to stop threads from looking for work.
};

// ============================================================================
// Template Implementation
// ============================================================================

template <typename Scalar, size_t kColumnBlock>
TrunkWork<Scalar, kColumnBlock>::TrunkWork(
    int superNode,
    LMatrix<Scalar, kColumnBlock>& lMatrix,
    int dofsPerNode,
    AssemblyHelper const& helper,
    SymbolicEliminationTree const& eTree,
    FrontalOrganizer const& organizer,
    Span<Span<Scalar>> childrenFrontBuffers,
    Span<Scalar> frontBuffer)
    : superNode(superNode), helper(helper) {
  auto treeGraph = eTree.TreeGraph();
  auto children = treeGraph[superNode];
  childrenFronts.reserve(isize(children));
  for (int i = 0; i < isize(children); ++i) {
    auto child = children[i];
    auto childL = lMatrix.LforSN(child);
    auto childFrontDOFs = childL.Rows() - childL.Cols();
    auto childIndices = eTree.SuperIndices(child);
    auto frontIndices = childIndices.subspan(eTree.SuperSize(child));
    auto relativeIndices = organizer.GetIndicesInParent(child);
    childrenFronts.push_back(
        ChildFront<kColumnBlock>{
            .superNode = child,
            .nodeIndices = frontIndices,
            .indicesInParent = relativeIndices,
            .front = Front<kColumnBlock>(
                childrenFrontBuffers[i].data() + childrenFrontBuffers[i].size(),
                childFrontDOFs,
                false),
        });
  }
  auto snL = lMatrix.LforSN(superNode);
  auto snIndices = helper.SuperNodeIndices(superNode);
  auto numLBlocks = snL.NumBlocks();
  numPanelsInL = static_cast<int>(numLBlocks);

  // Reserve and pre-size (Panel has std::atomic, can't be copied/moved)
  auto frontDOFs = snL.Rows() - snL.Cols();
  Front<kColumnBlock> front(
      frontDOFs > 0 ? frontBuffer.data() + frontBuffer.size() : nullptr, frontDOFs, false);
  panels.resize(numLBlocks + front.NumBlocks());
  int panelIdx = 0;

  // Create L panels from stair blocks
  for (size_t ib = 0; ib < numLBlocks; ++ib) {
    auto block = snL.Block(ib);
    auto nodeOffset = ib == 0
        ? 0
        : panels[panelIdx - 1].indexInSuperNode + panels[panelIdx - 1].panel.Cols() / dofsPerNode;
    auto& pnl = panels[panelIdx++];
    pnl.indexInSuperNode = nodeOffset;
    pnl.updateState.store(0);
    pnl.nodeIndices = snIndices.subspan(nodeOffset);
    pnl.panel.Reset(block.Data(), block.Rows(), block.Cols(), block.LeadDim());
  }

  // Create front panels
  if (frontDOFs > 0) {
    auto frontNodeOffset = static_cast<int>(snL.Cols() / dofsPerNode);
    for (auto block : front.template Blocks<Scalar>()) {
      auto nodeOffset =
          frontNodeOffset + static_cast<int>((front.Size() - block.Rows()) / dofsPerNode);
      auto& pnl = panels[panelIdx++];
      pnl.indexInSuperNode = nodeOffset;
      pnl.updateState.store(0);
      pnl.nodeIndices = snIndices.subspan(nodeOffset);
      pnl.panel.Reset(block.Data(), block.Rows(), block.Cols(), block.LeadDim());
    }
  }
}

template <typename Scalar, size_t kColumnBlock>
template <int kDofsPerNode>
void TrunkWork<Scalar, kColumnBlock>::AddInputMatrix(
    int panelIndex,
    BlockSparseMatrixView<Scalar const, kDofsPerNode> const& A) {
  auto& pnl = panels[panelIndex];
  auto nd = helper.FirstNode(superNode) + pnl.indexInSuperNode;

  for (int j = 0; j * kDofsPerNode < pnl.panel.Cols(); ++j, ++nd) {
    auto rowBlocks = helper.InputRow(A, nd);
    auto placements = helper.LPlacements(nd);
    auto numBlockRows = isize(pnl.nodeIndices) - j;
    BlockColView<Scalar, kDofsPerNode> lNodal(
        &pnl.panel(j * kDofsPerNode, j * kDofsPerNode), pnl.panel.LeadDim(), numBlockRows);
    for (auto [aCol, lRow] : placements) {
      lNodal[lRow] += rowBlocks[aCol].Transpose();
    }
  }
}

template <typename Scalar, size_t kColumnBlock>
template <int kDofsPerNode>
void TrunkWork<Scalar, kColumnBlock>::AssembleChildrenFronts(int panelIndex) {
  auto& pnl = panels[panelIndex];
  auto panelStartNode = pnl.indexInSuperNode;
  auto panelCols = pnl.panel.Cols() / kDofsPerNode;

  for (auto& child : childrenFronts) {
    auto childNodes = child.nodeIndices;
    auto indicesInParent = child.indicesInParent;

    for (auto [childColIdx, childCol] : child.front.template NodalColumns<Scalar, kDofsPerNode>()) {
      int pi = indicesInParent[childColIdx] - panelStartNode;
      if (pi < 0) {
        continue;
      }
      if (pi >= panelCols) {
        break;
      }
      // Scatter child rows into panel rows
      auto childPerNode = BlockColView<Scalar, kDofsPerNode>(
          childCol.data(), childCol.LeadDim(), isize(childNodes) - childColIdx);
      auto panelColStart = &pnl.panel(pi * kDofsPerNode, pi * kDofsPerNode);
      auto numPanelBlockRows = isize(pnl.nodeIndices) - pi;
      BlockColView<Scalar, kDofsPerNode> panelNodal(
          panelColStart, pnl.panel.LeadDim(), numPanelBlockRows);

      for (int cj = 0; cj < childPerNode.NumBlocks(); ++cj) {
        int pj = indicesInParent[childColIdx + cj] - panelStartNode - pi;
        panelNodal[pj] += childPerNode[cj];
      }
    }
  }
}

template <typename Scalar, size_t kColumnBlock>
void TrunkWork<Scalar, kColumnBlock>::FactorPanel(int panelIndex) {
  auto& pnl = panels[panelIndex];
  auto Dblock = pnl.panel.TopRows(pnl.panel.Cols());
  auto singularityCheck = [](int, Scalar, auto const&) { return false; };
  LDLtFactorize(Dblock, true, singularityCheck);
  auto Lpart = pnl.panel.BottomRows(pnl.panel.Rows() - pnl.panel.Cols());
  ApplyLmTOnRight<Scalar>(Dblock, Lpart);
  for (int c = 0; c < pnl.panel.Cols(); ++c) {
    Lpart.Col(c) *= Dblock(c, c);
  }
}

template <typename Scalar, size_t kColumnBlock>
bool TrunkWork<Scalar, kColumnBlock>::EnterWork() {
  auto activeWorkers = activeWorkerCount.load(std::memory_order::relaxed);
  auto activePanels = isize(panels) - completedPanelCount.load(std::memory_order::relaxed);
  // If there are more panels to work on than workers already working, we try to get in.
  while (activePanels > activeWorkers &&
         !activeWorkerCount.compare_exchange_strong(activeWorkers, activeWorkers + 1)) {
    // Some other worker joined before us. Get an up-to-date number of active panels.
    activePanels = isize(panels) - completedPanelCount.load(std::memory_order::relaxed);
  }
  return activePanels > activeWorkers;
}

template <typename Scalar, size_t kColumnBlock>
bool TrunkWork<Scalar, kColumnBlock>::LeaveWork() {
  auto activeWorkers = activeWorkerCount.load(std::memory_order::relaxed);
  auto pendingPanels = isize(panels) - completedPanelCount.load(std::memory_order::relaxed);
  // If there are more active workers than panels to work on, we try to leave.
  while (pendingPanels < activeWorkers &&
         !activeWorkerCount.compare_exchange_strong(activeWorkers, activeWorkers - 1)) {
    // Some other worker left before us. Get an up-to-date number of active panels.
    pendingPanels = isize(panels) - completedPanelCount.load(std::memory_order::relaxed);
  }
  return pendingPanels < activeWorkers;
}

template <typename Scalar, size_t kColumnBlock>
void TrunkWork<Scalar, kColumnBlock>::EliminatePanel(
    int sourcePanelIndex,
    int targetPanelIndex,
    Span<Scalar> uBuffer) {
  auto& source = panels[sourcePanelIndex];
  auto& target = panels[targetPanelIndex];
  auto sourceCols = source.panel.Cols();
  auto targetCols = target.panel.Cols();
  auto leftBlock = source.panel.BottomRows(target.panel.Rows());
  auto leftD = source.panel.TopRows(sourceCols);

  MOCHI_ASSERT_VERBOSE(isize(uBuffer) >= sourceCols * targetCols, "Buffer too small");
  MatrixView<Scalar> U(uBuffer.data(), sourceCols, targetCols);
  U = leftBlock.TopRows(targetCols).Transpose();
  for (int r = 0; r < sourceCols; ++r) {
    U.Row(r) *= (Scalar{1} / leftD(r, r));
  }
  target.panel -= leftBlock * U;
}

template <typename Scalar, size_t kColumnBlock>
template <int kDofsPerNode>
bool TrunkWork<Scalar, kColumnBlock>::WorkOnPanel(
    int panelIndex,
    BlockSparseMatrixView<Scalar const, kDofsPerNode> const& A,
    Span<Scalar> uBuffer) {
  auto& panel = panels[panelIndex];

  auto state = panel.updateState.fetch_or(Panel<Scalar>::kAcquired);
  auto level = state & Panel<Scalar>::kMaskLevel;
  if ((state & Panel<Scalar>::kAcquired) || (level > panelIndex)) {
    return false; // This panel is already being worked on or is done
  }
  if ((state & Panel<Scalar>::kAssembled) == 0) {
    if (panelIndex < numPanelsInL) {
      // Assemble input matrix entries into the panel
      AddInputMatrix<kDofsPerNode>(panelIndex, A);
    } else {
      panel.panel.SetZero();
    }
    // Assemble the children's Schur complements into the panel
    AssembleChildrenFronts<kDofsPerNode>(panelIndex);
    // Mark the panel as assembled
    panel.updateState.fetch_or(Panel<Scalar>::kAssembled);
  }

  while (level < FactoredPanelCount()) {
    EliminatePanel(level, panelIndex, uBuffer);
    ++level;
  }

  bool completesWork = false;
  if (level == panelIndex || level == numPanelsInL) {
    if (panelIndex < numPanelsInL) {
      FactorPanel(panelIndex);
    }
    auto completed = this->completedPanelCount.fetch_add(1) + 1;
    completesWork = completed == isize(panels);
    // Update the level if this panel is complete.
    level = panelIndex + 1;
  }
  // Record the new panel level and clear the acquired flag.
  panel.updateState.store(level | Panel<Scalar>::kAssembled, std::memory_order_release);
  return completesWork; // True if the last panel was completed
}

} // namespace mochi
