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
#include <mochi_core/async/generator.h>
#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/linear_algebra/multi_frontal/stair_iterator.h>
#include <mochi_core/utils/range_by_iterators.h>

namespace mochi {

inline size_t StairMatrixSize(size_t rows, size_t cols, size_t kBlockSize) {
  auto fullBlocks = cols / kBlockSize;
  auto leftOverCols = cols % kBlockSize;
  return rows * cols - fullBlocks * (fullBlocks - 1) * kBlockSize * kBlockSize / 2 -
      leftOverCols * fullBlocks * kBlockSize;
}
/**
 * @brief Provides a stair-like block view of a matrix.
 *
 * StairMatrixView divides a rectangular matrix into blocks in a stair pattern,
 * where each subsequent  block starts further to the right and has fewer rows
 * than the previous block.
 *
 * The matrix structure looks like this:
 * ```
 * [Block0][      ][      ]
 * [Block0][Block1][      ]
 * [Block0][Block1][Block2]
 * [Block0][Block1][Block2]
 * ```
 *
 * This view is particularly useful for block-based matrix BLAS-3 type operations.
 *
 * @tparam Scalar The data type of the matrix elements
 * @tparam kBlockSize The size of blocks (must be even)
 */
template <typename Scalar, size_t kBlockSize>
class StairMatrixView {
  static_assert(kBlockSize != 0 && kBlockSize % 2 == 0, "kBlockSize must be non-zero even");

 public:
  using BlockType = MatrixView<Scalar>;

  /**
   * @brief Constructs a stair matrix view.
   *
   * @param v Pointer to the underlying matrix data
   * @param rows Number of rows in the matrix
   * @param cols Number of columns in the matrix
   */
  StairMatrixView(Scalar* v, size_t rows, size_t cols);

  /**
   * @brief Returns a view of the specified block.
   *
   * Block indices start from 0 and increase left-to-right, top-to-bottom.
   * Each block has dimensions determined by the stair pattern:
   * - Width: At most kBlockSize (may be smaller for the first block or at matrix edges)
   * - Height: Decreases for each subsequent block
   *
   * @param iBlock Index of the block to retrieve
   * @return BlockType View of the requested block
   */
  auto Block(size_t iBlock) const -> BlockType;

  /**
   * @brief Generates views for all blocks in the matrix.
   *
   * Yields blocks in order from left to right.
   * Each block represents a view into part of the original matrix data,
   * with dimensions following the stair pattern.
   *
   * @return Generator<BlockType> Generator yielding views of each block
   */
  auto Blocks() const -> Generator<BlockType>;

  template <size_t kDofsPerNode>
  RangeByIterators<StairNodalIterator<Scalar, kDofsPerNode, kBlockSize>> NodalColumns() const;

  auto Rows() const {
    return _rows;
  }

  auto Cols() const {
    return _cols;
  }

  /** @brief Returns the number of blocks in the stair matrix.
   *
   * Calculates the total number of blocks needed to cover all columns.
   * The first block may have fewer columns than kBlockSize (minimum columns),
   * and subsequent blocks have exactly kBlockSize columns (except possibly the last).
   *
   * @return Total number of blocks in the stair pattern
   */
  [[nodiscard]] size_t NumBlocks() const {
    if (_cols == 0) {
      return 0;
    }
    // First block can be smaller, remaining columns divided into kBlockSize blocks
    auto remainingCols = _cols - _firstCols;
    return 1 + (remainingCols + kBlockSize - 1) / kBlockSize;
  }

 private:
  /** @brief Number of elements in half a block */
  static constexpr auto kHalfArea = kBlockSize * kBlockSize / 2;

  Scalar* _v; ///< Pointer to the matrix data
  size_t _rows; ///< Total number of rows in the matrix
  size_t _cols; ///< Total number of columns in the matrix
  size_t _firstCols; ///< Number of columns in the first block
};

template <typename Scalar, size_t kBlockSize>
StairMatrixView<Scalar, kBlockSize>::StairMatrixView(Scalar* v, size_t rows, size_t cols)
    : _v(v), _rows(rows), _cols(cols) {
  MOCHI_ASSERT_VERBOSE(cols != 0, "Cols must be non-zero");
  _firstCols = std::min(kBlockSize, _cols);
}
template <typename Scalar, size_t kBlockSize>
auto StairMatrixView<Scalar, kBlockSize>::Block(size_t iBlock) const -> BlockType {
  auto downCount = iBlock == 0 ? 0 : _firstCols + (iBlock - 1) * kBlockSize;
  auto offset = _rows * downCount - iBlock * (iBlock - 1) * kHalfArea;
  auto colCount = iBlock == 0 ? _firstCols : std::min(kBlockSize, _cols - downCount);
  return {_v + offset, static_cast<int>(_rows - downCount), static_cast<int>(colCount)};
}

template <typename Scalar, size_t kBlockSize>
auto StairMatrixView<Scalar, kBlockSize>::Blocks() const -> Generator<BlockType> {
  auto ptr = _v;
  size_t blockStartCol = 0;
  auto rows = _rows;
  auto blockCols = _firstCols;
  while (blockStartCol < _cols) {
    co_yield BlockType{ptr, static_cast<int>(rows), static_cast<int>(blockCols)};
    ptr += blockCols * rows;
    rows -= blockCols;
    blockStartCol += blockCols;
    blockCols = std::min(kBlockSize, _cols - blockStartCol);
  }
}

template <typename Scalar, size_t kBlockSize>
template <size_t kDofsPerNode>
RangeByIterators<StairNodalIterator<Scalar, kDofsPerNode, kBlockSize>>
StairMatrixView<Scalar, kBlockSize>::NodalColumns() const {
  auto stairSize = StairMatrixSize(_rows, _cols, kBlockSize);
  auto endPtr = _v + stairSize;
  int rows = static_cast<int>(_rows);
  return {{_v, _v + _firstCols * _rows, endPtr, rows, rows}, {endPtr, endPtr, endPtr, rows, 0}};
}

} // namespace mochi
