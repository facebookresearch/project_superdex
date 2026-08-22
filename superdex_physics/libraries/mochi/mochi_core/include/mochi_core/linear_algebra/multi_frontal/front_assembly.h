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
#include <mochi_core/linear_algebra/multi_frontal/front_matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/organizer.h>
#include <mochi_core/linear_algebra/multi_frontal/stair_matrix.h>
#include <algorithm>

namespace mochi {

/** @brief Compact the data in the block the iterator is into to the right.
 *
 * @details On entry, the iterator points inside a block of nodal leading dimension `fullNRows`.
 * However, only the data of the `currentNRows` lower rows are going to be used. This method
 * compacts the lower part of each nodal column into a matrix of nodal leading dimension of
 * `currentNRows`, with the same `blockEnd`.
 *
 * @tparam Scalar
 * @tparam kDofsPerNode
 * @tparam kBlockCols
 * @param iterator
 */
template <typename Scalar, size_t kDofsPerNode, size_t kBlockCols>
void CompactRight(StairNodalIterator<Scalar, kDofsPerNode, kBlockCols>& iterator) {
  if (iterator.currentNRows == iterator.fullNRows)
    MOCHI_UNLIKELY {
      return;
    }
  // Compute the pointers for the last block to be moved.
  auto destinationEnd = iterator.blockEnd - iterator.currentNRows;
  auto sourceEnd = iterator.blockEnd - iterator.fullNRows;
  auto sourceBegin = iterator.blockEnd - 2 * iterator.fullNRows;
  while (sourceEnd > iterator.v) {
    std::copy_backward(sourceBegin, sourceEnd, destinationEnd);
    sourceEnd -= iterator.fullNRows;
    sourceBegin -= iterator.fullNRows;
    destinationEnd -= iterator.currentNRows;
  }
  iterator.v = destinationEnd;
  iterator.fullNRows = iterator.currentNRows;
}

/// @brief Advances past expired ranges and maps a child column to its parent position.
inline int AdvanceAndMapChildToParent(int col, Span<IndexRange const> ranges, int& rangeIdx) {
  int const numRanges = isize(ranges);
  while (rangeIdx < numRanges && col >= ranges[rangeIdx].childStart + ranges[rangeIdx].length) {
    ++rangeIdx;
  }
  if (rangeIdx >= numRanges) {
    return kMinusOne<int>;
  }
  return ranges[rangeIdx].parentStart + (col - ranges[rangeIdx].childStart);
}

/// @brief Accumulates child blocks into parent blocks over the given index ranges.
template <int kDofsPerNode, typename Scalar>
void AccumulateOverRanges(
    BlockColView<Scalar, kDofsPerNode> childBlocks,
    BlockColView<Scalar, kDofsPerNode> parentBlocks,
    Span<IndexRange const> ranges,
    int startRangeIdx,
    int childColumn,
    int parentNodeIndex) {
  int const numRanges = isize(ranges);
  for (int ri = startRangeIdx; ri < numRanges; ++ri) {
    auto const& range = ranges[ri];
    int startC = std::max(childColumn, range.childStart);
    int startP = range.parentStart + (startC - range.childStart);
    int len = range.length - (startC - range.childStart);
    if (len > 0) {
      auto childSub = childBlocks.Underlying().MiddleRows(
          (startC - childColumn) * kDofsPerNode, len * kDofsPerNode);
      auto parentSub = parentBlocks.Underlying().MiddleRows(
          (startP - parentNodeIndex) * kDofsPerNode, len * kDofsPerNode);
      parentSub += childSub;
    }
  }
}

/** @brief Expand the front of a child supernode into its parent's L and front.
 *
 * All the children's fronts must be assembled into the parent's front.
 * For one child, the front memory overlaps the parent's memory. This trick allows reducing
 * the size of the front stack memory, improving cache performance. and, if kPackSmall is true,
 * reducing the amount of work for this assembly case.
 *
 * @note The assembly of the child's front into L is between separate memory areas and does
 * not need any special consideration. The assembly into the parent's front can involve overlapping
 * memory. This implementation currently works only when kPackSmall is true. In such a case the
 * process is fairly simple: the operation can proceed in strictly increasing parent column indices.
 * This is because the parent's front is always larger than its child's overlapping part.
 * Any contribution from a child's column goes into a column further left in the parent's
 * front until a perfect overlay is reached. Thus, writing from left to right cannot erase data
 * that has not yet been accounted for. Once a perfect overlay is reached, the work can stop.
 *
 *
 * @tparam Scalar
 * @tparam kStairColBlock
 * @tparam kColumnBlock
 * @tparam kPackSmall
 * @param childFront Front of the child to assemble into the parent.
 * @param parentL The columns of L associated with the parent supernode.
 * @param parentFront The parent supernode's front matrix.
 * @param ranges Ranges mapping child overlap indices to parent index positions.
 */
template <
    int kDofsPerNode,
    typename Scalar,
    size_t kStairColBlock,
    size_t kColumnBlock,
    bool kPackSmall>
void ExpandIntoParent(
    Front<kColumnBlock, kPackSmall>& childFront,
    StairMatrixView<Scalar, kStairColBlock> parentL,
    Front<kColumnBlock, kPackSmall>& parentFront,
    Span<IndexRange const> ranges) {
  static_assert(
      kColumnBlock % kDofsPerNode == 0,
      "Front column block size must be a multiple of DoFs per node");
  static_assert(
      kStairColBlock % kDofsPerNode == 0, "L stair block size must be a multiple of DoFs per node");
  static_assert(kPackSmall == true, "pack big is not implemented yet.");
  auto childNodalRange = childFront.template NodalColumnsRange<Scalar, kDofsPerNode>();
  auto cIt = childNodalRange.begin();
  auto cEnd = childNodalRange.end();

  int parentNodeIndex = 0;
  int parentNdCount = static_cast<int>(parentL.Rows()) / kDofsPerNode;
  int childColumn = 0;

  int rangeIdx = 0;
  int childInParent =
      cIt == cEnd ? kMinusOne<int> : AdvanceAndMapChildToParent(childColumn, ranges, rangeIdx);
  int numChildNodes = ranges.empty() ? 0 : (ranges.back().childStart + ranges.back().length);

  for (auto nodalL : parentL.template NodalColumns<kDofsPerNode>()) {
    nodalL.SetZero();
    if (parentNodeIndex == childInParent) {
      // Assemble
      auto lPerNode = BlockColView<Scalar, kDofsPerNode>(
          nodalL.data(), nodalL.LeadDim(), parentNdCount - parentNodeIndex);
      auto chNodeCols = *cIt;
      auto childPerNode = BlockColView<Scalar, kDofsPerNode>(
          chNodeCols.data(), chNodeCols.LeadDim(), numChildNodes - childColumn);

      AccumulateOverRanges<kDofsPerNode>(
          childPerNode, lPerNode, ranges, rangeIdx, childColumn, parentNodeIndex);

      ++childColumn;
      childInParent = ++cIt == cEnd ? kMinusOne<int>
                                    : AdvanceAndMapChildToParent(childColumn, ranges, rangeIdx);
    }
    ++parentNodeIndex;
  }
  // The expansion of the front is based on an assumption that a child's data is always at a higher
  // address than its destination. There are, however, cases when this is not the case. This can
  // only happen with the first block of the front (because the parent front always has more
  // indices than the child's). If the possibility is detected, we compact the child's first
  // block to the right.
  if (childInParent != kMinusOne<int> && cIt->LeadDim() > parentFront.Size()) {
    CompactRight(cIt);
  }

  int totalParentBlocks = parentNdCount - parentNodeIndex;
  for (auto& [nd, nodalFCol] : parentFront.template NodalColumns<Scalar, kDofsPerNode>()) {
    // if we have perfect overlay, we don't need to work further.
    if (childInParent != kMinusOne<int> && (*cIt).data() == nodalFCol.data()) {
      return;
    }
    if (parentNodeIndex == childInParent) {
      // Assemble
      auto chNodeCols = *cIt;

      // WARNING: Do NOT try to remove the loop over k (DoFs per node). Doing so will, in some
      // cases, overwrite not-yet-used data from the child when writing to the parent.
      int const numRanges = isize(ranges);
      for (int k = 0; k < kDofsPerNode; ++k) {
        auto* parentPtr = nodalFCol.Col(k).data();
        auto* childPtr = chNodeCols.Col(k).data();
        int pOffset = 0;
        int curRangeIdx = rangeIdx;

        while (curRangeIdx < numRanges) {
          auto const& range = ranges[curRangeIdx];
          int startC = std::max(childColumn, range.childStart);
          int startP = range.parentStart + (startC - range.childStart);
          int len = range.length - (startC - range.childStart);

          if (len > 0) {
            int targetOffset = startP - parentNodeIndex;
            if (targetOffset > pOffset) {
              int gapLen = targetOffset - pOffset;
              std::fill_n(parentPtr, gapLen * kDofsPerNode, Scalar(0));
              parentPtr += gapLen * kDofsPerNode;
              pOffset = targetOffset;
            }
            int childOffset = startC - childColumn;
            // Safe: parentPtr <= childPtr (dest <= src), so forward copy cannot clobber unread
            // data.
            std::copy_n(childPtr + childOffset * kDofsPerNode, len * kDofsPerNode, parentPtr);
            parentPtr += len * kDofsPerNode;
            pOffset += len;
          }
          ++curRangeIdx;
        }
        if (totalParentBlocks > pOffset) {
          int const gapLen = totalParentBlocks - pOffset;
          std::fill_n(parentPtr, gapLen * kDofsPerNode, Scalar(0));
        }
      }

      ++cIt;
      ++childColumn;
      childInParent =
          cIt == cEnd ? kMinusOne<int> : AdvanceAndMapChildToParent(childColumn, ranges, rangeIdx);
    } else {
      // Performance note: This zero'ing is expensive. Assess if it could be performed in a single
      // SetZero upfront or be optimized otherwise.
      nodalFCol.SetZero();
    }
    ++parentNodeIndex;
    --totalParentBlocks;
  }
}

/** @brief Assemble a front into an existing parent front. */
template <
    int kDofsPerNode,
    typename Scalar,
    size_t kStairColBlock,
    size_t kColumnBlock,
    bool kPackSmall>
void AssembleIntoParent(
    Front<kColumnBlock, kPackSmall>& childFront,
    StairMatrixView<Scalar, kStairColBlock> parentL,
    Front<kColumnBlock, kPackSmall>& parentFront,
    Span<IndexRange const> ranges) {
  static_assert(
      kColumnBlock % kDofsPerNode == 0,
      "Front column block size must be a multiple of DoFs per node");
  static_assert(
      kStairColBlock % kDofsPerNode == 0, "L stair block size must be a multiple of DoFs per node");
  static_assert(kPackSmall, "pack big is not implemented yet.");
  auto childNodalColumns = childFront.template NodalColumns<Scalar, kDofsPerNode>();

  auto childIt = childNodalColumns.begin();
  auto childEnd = childNodalColumns.end();

  int rangeIdx = 0;
  int childInParent = childIt == childEnd
      ? kMinusOne<int>
      : AdvanceAndMapChildToParent(childIt->first, ranges, rangeIdx);
  int parentNodeIndex = 0;
  int parentNdCount = static_cast<int>(parentL.Rows() / kDofsPerNode);
  int numChildNodes = ranges.empty() ? 0 : (ranges.back().childStart + ranges.back().length);

  for (auto nodalL : parentL.template NodalColumns<kDofsPerNode>()) {
    if (parentNodeIndex == childInParent) {
      // Assemble
      auto lPerNode = BlockColView<Scalar, kDofsPerNode>(
          nodalL.data(), nodalL.LeadDim(), parentNdCount - parentNodeIndex);
      auto& chNodeCols = childIt->second;
      auto childPerNode = BlockColView<Scalar, kDofsPerNode>(
          chNodeCols.data(), chNodeCols.LeadDim(), numChildNodes - childIt->first);

      AccumulateOverRanges<kDofsPerNode>(
          childPerNode, lPerNode, ranges, rangeIdx, childIt->first, parentNodeIndex);

      childInParent = ++childIt == childEnd
          ? kMinusOne<int>
          : AdvanceAndMapChildToParent(childIt->first, ranges, rangeIdx);
    }
    ++parentNodeIndex;
  }
  for (auto& [nd, nodalFCol] : parentFront.template NodalColumns<Scalar, kDofsPerNode>()) {
    if (parentNodeIndex == childInParent) {
      // Assemble
      auto lPerNode = BlockColView<Scalar, kDofsPerNode>(
          nodalFCol.data(), nodalFCol.LeadDim(), parentNdCount - parentNodeIndex);
      auto& chNodeCols = childIt->second;
      auto childPerNode = BlockColView<Scalar, kDofsPerNode>(
          chNodeCols.data(), chNodeCols.LeadDim(), numChildNodes - childIt->first);

      AccumulateOverRanges<kDofsPerNode>(
          childPerNode, lPerNode, ranges, rangeIdx, childIt->first, parentNodeIndex);

      childInParent = ++childIt == childEnd
          ? kMinusOne<int>
          : AdvanceAndMapChildToParent(childIt->first, ranges, rangeIdx);
    }
    ++parentNodeIndex;
  }
}

} // namespace mochi
