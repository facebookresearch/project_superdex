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

#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/utils/task_scheduler.h>

namespace mochi::details {

/** @brief Check that a series of interaction matrices can be added to a block sparse matrix.
 *
 * @tparam kBlockSize Block size
 * @tparam Scalar Type for the numerical values
 * @param[in] interactionMatrices Interaction matrices to be added
 * @param[in] ABsr Target block sparse matrix
 * @param[in] AOffset Global offset of the target block sparse matrix
 */
template <int kBlockSize, typename Scalar>
void CheckDimensionsOffsets(
    std::vector<AnyInteractionMatrixViewInfo<Scalar const>> const& interactionMatrices,
    BlockSparseMatrix<std::remove_const_t<Scalar>, kBlockSize, int, int> const& ABsr,
    int AOffset) {
  for (auto const& [rOffset, cOffset, interactionMatrix, _] : interactionMatrices) {
    // Check the overlap in the rows
    if ((rOffset + GetNumRows(interactionMatrix) <= AOffset) ||
        (AOffset + ABsr.Rows() <= rOffset)) {
      continue;
    }
    // Check the overlap in the columns
    if ((cOffset + GetNumCols(interactionMatrix) <= AOffset) ||
        (AOffset + ABsr.Cols() <= cOffset)) {
      continue;
    }

    // The interaction matrix overlaps the actor matrix. If it does, it must be square and have the
    // same row and col offset (see documentation in InteractionMatrixInfoImpl).
    MOCHI_ASSERT(
        rOffset == cOffset && GetNumRows(interactionMatrix) == GetNumCols(interactionMatrix),
        "Unsupported interaction matrix layout.");

    static_assert(
        std::variant_size_v<decltype(interactionMatrix)> == 4,
        "Please update the logic below if the interaction matrix types change");
    int constexpr kOtherBlockSize = (kBlockSize == 3) ? 4 : 3; // Only 3 and 4 are supported.
    if (auto const* sp = std::get_if<SparseMatrixView<Scalar const>>(&interactionMatrix)) {
      // Adding sparse matrix to block sparse matrix is always supported.
    } else if (
        auto const* bsp =
            std::get_if<BlockSparseMatrixView<Scalar const, kBlockSize>>(&interactionMatrix)) {
      // Adding block sparse matrix to block sparse matrix requires offsets to be consistent.
      MOCHI_ASSERT(
          rOffset % kBlockSize == AOffset % kBlockSize &&
              cOffset % kBlockSize == AOffset % kBlockSize,
          "Offset of interaction matrix incompatible with offset of block sparse actor matrix");
    } else if (
        auto const* otherBSp =
            std::get_if<BlockSparseMatrixView<Scalar const, kOtherBlockSize>>(&interactionMatrix)) {
      // Adding block sparse matrix to block sparse matrix of different block size is only supported
      // if there are no entries in the overlap region.
      int const brBegin = Max(0, (AOffset - rOffset) / kOtherBlockSize);
      int const brEnd =
          Min(otherBSp->BlockRows(),
              (ABsr.Rows() + AOffset - rOffset + kOtherBlockSize - 1) / kOtherBlockSize);
      MOCHI_ASSERT(
          otherBSp->NumNonZeroBlocksInBlockRowRange(brBegin, brEnd) == 0,
          "Cannot add block sparse interaction matrix to block sparse actor matrix with different block size.");
    } else {
      MOCHI_ASSERT(false, "Cannot add dense interaction matrix to block sparse actor matrix.");
    }
  }
}

/**
 * @brief Add interaction matrices to a block sparse matrix.
 *
 * @tparam kSkipMissingSparsityEntries If true, interaction matrix entries that fall outside the
 * target block sparse matrix's sparsity pattern are silently dropped. If false (default), it is
 * invalid to have such entries (asserts in debug builds; undefined behavior in optimized builds).
 * @tparam kBlockSize Block size
 * @tparam Scalar Type for the numerical values
 * @param[in] interactionMatrices Interaction matrices to be added
 * @param[in,out] ABsr Block sparse matrix
 * @param[in] AOffset Global offset of the block sparse matrix
 *
 * @note The constraints for the interaction matrices are documented in @ref
 * InteractionMatrixInfoImpl.
 */
template <bool kSkipMissingSparsityEntries = false, int kBlockSize, typename Scalar>
void AddInteractionToBlockSparseMatrix(
    std::vector<AnyInteractionMatrixViewInfo<Scalar const>> const& interactionMatrices,
    BlockSparseMatrix<std::remove_const_t<Scalar>, kBlockSize, int, int>& ABsr,
    int AOffset) {
  CheckDimensionsOffsets<kBlockSize>(interactionMatrices, ABsr, AOffset);

  auto const aRows = ABsr.Rows();
  auto const aCols = ABsr.Cols();

  for (auto const& info : interactionMatrices) {
    auto const rOffset = info.rowOffset;
    auto const cOffset = info.colOffset;
    auto const& interactionMatrix = info.matrix;
    auto const globalRowBegin = Max(AOffset, rOffset);
    auto const globalRowEnd = Min(AOffset + aRows, rOffset + GetNumRows(interactionMatrix));
    if (globalRowEnd <= globalRowBegin) {
      continue;
    }
    auto const globalColBegin = Max(AOffset, cOffset);
    auto const globalColEnd = Min(AOffset + aCols, cOffset + GetNumCols(interactionMatrix));
    if (globalColEnd <= globalColBegin) {
      continue;
    }
    MOCHI_ASSERT_VERBOSE(
        rOffset == cOffset && globalRowBegin == globalColBegin && globalRowEnd == globalColEnd,
        "Supported overlapping interaction matrices must be diagonal for this actor.");

    if (auto const* bsp =
            std::get_if<BlockSparseMatrixView<Scalar const, kBlockSize>>(&interactionMatrix)) {
      auto const bStart = (globalRowBegin - rOffset) / kBlockSize;
      auto const bEnd = (globalRowEnd - rOffset) / kBlockSize;
      auto const bNnz = bsp->NumNonZeroBlocksInBlockRowRange(bStart, bEnd);
      if (bNnz == 0) {
        continue;
      }
      auto workerTask = [&ABsr, &bsp, AOffset, rOffset, bStart, bEnd](int brInit, int brStop) {
        auto const& interactionBsr = *bsp;
        int const blockShift = (rOffset - AOffset) / kBlockSize;
        for (int br = brInit; br < brStop; ++br) {
          auto const blockColIdx = interactionBsr.Indices(br);
          if (blockColIdx.empty()) {
            continue;
          }
          MOCHI_ASSERT_VERBOSE(
              std::is_sorted(blockColIdx.begin(), blockColIdx.end()), "Row entries are not sorted");
          auto const blockValues = interactionBsr.Values(br);
          int const aBlock = blockShift + br;
          auto const aBlockColIdx = ABsr.Indices(aBlock);
          auto aBlockValues = ABsr.Values(aBlock);
          auto aBlockColIdxItr = aBlockColIdx.begin();
          for (int k = static_cast<int>(
                   std::ranges::lower_bound(blockColIdx, bStart) - blockColIdx.begin());
               (k < isize(blockColIdx)) && (blockColIdx[k] < bEnd);
               ++k) {
            auto const aLocalBlock = blockShift + blockColIdx[k];
            auto next = std::lower_bound(aBlockColIdxItr, aBlockColIdx.end(), aLocalBlock);
            if constexpr (kSkipMissingSparsityEntries) {
              if ((next == aBlockColIdx.end()) || (*next != aLocalBlock))
                MOCHI_UNLIKELY {
                  continue;
                }
            } else {
              MOCHI_ASSERT_VERBOSE(
                  (next != aBlockColIdx.end()) && (*next == aLocalBlock),
                  "Attempting to add values that are not supported by the sparsity pattern of the actor matrix.");
            }
            // Add values (kBlockSize x kBlockSize matrix) in this block
            auto const myIdx = int(next - aBlockColIdx.begin());
            aBlockValues[myIdx] += blockValues[k];
            aBlockColIdxItr = next + 1;
          }
        }
      };
      // TODO Generalize the heuristic '100' value for any block size
      constexpr long long kMinBlocksPerTask = 100;
      int const minBlockRowsPerTask =
          Max(1, static_cast<int>((kMinBlocksPerTask * (bEnd - bStart)) / bNnz));
      ParallelForRange(
          "AddBlockSparseInteraction", bStart, bEnd, minBlockRowsPerTask, INT_MAX, workerTask);
    } else if (auto const* sp = std::get_if<SparseMatrixView<Scalar const>>(&interactionMatrix)) {
      // Note: The sparsity pattern of a sparse interaction matrix that overlapps the actor matrix
      // may NOT be blockable.
      auto const localRowStart = globalRowBegin - rOffset;
      auto const localRowEnd = globalRowEnd - rOffset;
      auto const localColStart = globalColBegin - cOffset;
      auto const localColEnd = globalColEnd - cOffset;
      auto const localNnz = sp->NumNonZerosInRowRange(localRowStart, localRowEnd);
      if (localNnz == 0) {
        continue;
      }
      auto workerTask = [&ABsr, &sp, AOffset, rOffset, cOffset, localColStart, localColEnd](
                            int rStart, int rEnd) {
        auto const& interactionSparse = *sp;
        for (int r = rStart; r < rEnd; ++r) {
          auto const localColIdx = interactionSparse.Indices(r);
          if (localColIdx.empty()) {
            continue;
          }
          MOCHI_ASSERT_VERBOSE(
              std::is_sorted(localColIdx.begin(), localColIdx.end()), "Row entries are not sorted");

          // Performance note: The implementation could be optimized if (a) the sparse matrix is
          // blockable in the overlap region or (b) consecutive rows of the interaction matrix share
          // the same sparsity.
          auto const values = interactionSparse.Values(r);
          auto const aRow = r + rOffset - AOffset;
          auto const aBlockRow = aRow / kBlockSize;
          auto const aLocalRow = aRow % kBlockSize;

          auto const aBlockColIdx = ABsr.Indices(aBlockRow);
          auto aBlockValues = ABsr.Values(aBlockRow);

          // Track current block to avoid repeated binary searches for consecutive columns in same
          // block.
          int currentBlockCol = -1;
          int currentBlockIdx = -1;

          for (int k = static_cast<int>(
                   std::ranges::lower_bound(localColIdx, localColStart) - localColIdx.begin());
               (k < isize(localColIdx)) && (localColIdx[k] < localColEnd);
               ++k) {
            auto const globalCol = cOffset + localColIdx[k];
            auto const aCol = globalCol - AOffset;
            auto const aBlockCol = aCol / kBlockSize;
            auto const aLocalCol = aCol % kBlockSize;

            // Find the block index (reuse if same block as previous iteration).
            if (aBlockCol != currentBlockCol) {
              auto it = std::lower_bound(
                  aBlockColIdx.begin() + (currentBlockIdx + 1), aBlockColIdx.end(), aBlockCol);
              if constexpr (kSkipMissingSparsityEntries) {
                if (it == aBlockColIdx.end() || *it != aBlockCol)
                  MOCHI_UNLIKELY {
                    continue;
                  }
              } else {
                MOCHI_ASSERT_VERBOSE(
                    it != aBlockColIdx.end() && *it == aBlockCol,
                    "Attempting to add values that are not supported by the sparsity pattern of the actor matrix.");
              }
              currentBlockIdx = static_cast<int>(it - aBlockColIdx.begin());
              currentBlockCol = aBlockCol;
            }

            // Add the value to the appropriate position in the block.
            aBlockValues[currentBlockIdx](aLocalRow, aLocalCol) += values[k];
          }
        }
      };
      // TODO Generalize the heuristic '300' value for any block size
      constexpr long long kMinValuesPerTask = 300;
      int const minRowsPerTask =
          Max(1, static_cast<int>((kMinValuesPerTask * (localRowEnd - localRowStart)) / localNnz));
      ParallelForRange(
          "AddSparseInteraction", localRowStart, localRowEnd, minRowsPerTask, INT_MAX, workerTask);
    }
  }
}

} // namespace mochi::details

namespace mochi {

template <int kBlockSize, bool kSkipMissingSparsityEntries, typename Scalar>
BlockSparseMatrix<std::remove_const_t<Scalar>, kBlockSize, int, int> ToBlockSparseMatrix(
    ActorPseudoMatrix<Scalar> const& in) {
  // Verify that the actor matrix is block sparse with the correct block size.
  using NonConstScalar = std::remove_const_t<Scalar>;
  using BSpMatrixView = BlockSparseMatrixView<Scalar const, kBlockSize>;
  MOCHI_ASSERT(
      std::holds_alternative<BSpMatrixView>(in.actorMatrix),
      "Actor matrix type not supported for conversion to block sparse matrix.");
  MOCHI_ASSERT_VERBOSE(in.Rows() == in.Cols(), "Expected square actor matrix.");
  MOCHI_ASSERT(in.Rows() % kBlockSize == 0, "Number of rows must be a multiple of the block size.");

  auto const& bsr = std::get<BSpMatrixView>(in.actorMatrix);
  // Make a copy of the actor matrix in block sparse format
  BlockSparseMatrix<NonConstScalar, kBlockSize> ABsr(bsr);
  // Add interaction to block sparse matrix
  details::AddInteractionToBlockSparseMatrix<kSkipMissingSparsityEntries>(
      in.interactionMatrices, ABsr, in.offset);
  return ABsr;
}

} // namespace mochi
