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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/multi_frontal/stair_iterator.h>
#include <mochi_core/utils/range_by_iterators.h>

namespace mochi {

/** @brief A non-owning front matrix object.
 *
 * @details A front is a symmetric square matrix. In principle, it is sufficient to store the
 * lower triangular part of the matrix. However, such a storage is not conducive to efficient
 * SIMD operations. For this reason, we store a *staircase* version of the matrix.
 * The matrix is stored as a series of rectangular blocks with decreasing size as the block
 * index increases (descending steps). Each block has
 * a fixed width of kColumnBlock except for the first or last block. When kPackSmall
 * is true, the first block may be narrower, otherwise the last one may be narrower.
 * The blocks are stored contiguously in block order.
 *
 * kColumnBlock should be chosen to be a multiple of the GEMM kernel (6 in Mochi as well as MKL)
 * to avoid the loop spill-over treatment. It should also be chosen large enough so that the rank-N
 * update, where N is kColumnBlock in many of the loops is efficient for the architecture.
 * This number should ideally be tuned for the architecture/BLAS library used.
 *
 * @tparam kColumnBlock Maximum number of columns in a rectangular block of a front.
 * @tparam kPackSmall Whether to fill the smallest block first or last.
 */
template <size_t kColumnBlock = 6 * 16, bool kPackSmall = true>
class Front {
 public:
  static_assert(kColumnBlock != 0 && kColumnBlock % 2 == 0, "kColumnBlock must be non-zero even");

  static constexpr size_t StorageSize(size_t numDOFs);

  /** @param end One past the last element of the front's storage, the @ref StorageSize(numDOFs)
   * elements before it.
   * @param numDOFs Size of the front matrix in DOFs.
   */
  template <typename Scalar>
  Front(Scalar* end, size_t numDOFs);

  /// @brief Number of rows/columns of the front matrix.
  [[nodiscard]] constexpr size_t Size() const {
    return _numDOFs;
  }

  template <typename Scalar>
  MatrixView<Scalar> Block(size_t block) const;

  /// @brief One @p kDofsPerNode-wide view per node, in increasing node order, each from the node's
  /// diagonal down. The front's size and @p kColumnBlock must be multiples of @p kDofsPerNode.
  template <typename Scalar, size_t kDofsPerNode>
  RangeByIterators<StairNodalIterator<Scalar, kDofsPerNode, kColumnBlock>> NodalColumns() const;

  [[nodiscard]] size_t NumBlocks() const;

  [[nodiscard]] constexpr size_t StorageSize() const;

  template <typename Scalar>
  static MatrixView<Scalar> MakeBlock(Scalar* ptr, size_t rows, size_t cols) {
    return MatrixView<Scalar>(ptr, static_cast<int>(rows), static_cast<int>(cols));
  }

  /// @brief Pointer to the beginning of the front's storage.
  template <typename Scalar>
  Scalar* MemoryStart() const {
    return static_cast<Scalar*>(_end) - StorageSize();
  }

 private:
  /// @brief End of the front's storage.
  void* _end;
  /// @brief Number of DOFs in the front.
  size_t _numDOFs;
  /// @brief Number of blocks.
  size_t _numBlocks;
  /// @brief Number of columns for the edge block (first if kPackSmall, last otherwise)
  size_t _edgeCols;

  static constexpr auto kHalfSquare = kColumnBlock * kColumnBlock / 2;
};

inline constexpr size_t FrontStorageSize(size_t numDOFs, size_t columnBlock) {
  auto numFullBlocks = numDOFs / columnBlock;
  auto overflow = numDOFs % columnBlock;
  auto numSquares = (numFullBlocks * (numFullBlocks + 1));
  return numSquares * columnBlock * columnBlock / 2 + overflow * numDOFs;
}

/**
 * @brief Computes the required storage size for a front matrix given the number of degrees of
 * freedom (DOFs).
 *
 * @param numDOFs The number of degrees of freedom (size of the front matrix).
 * @return The total storage size needed to store the front matrix in staircase format.
 *
 * @details
 * The function calculates the number of full blocks (of size kColumnBlock), the overflow (remaining
 * DOFs), and the number of square blocks. The total storage is the sum of the storage for all full
 * square blocks and the storage for the overflow columns.
 */
template <size_t kColumnBlock, bool kPackSmall>
constexpr size_t Front<kColumnBlock, kPackSmall>::StorageSize(size_t numDOFs) {
  return FrontStorageSize(numDOFs, kColumnBlock);
}

template <size_t kColumnBlock, bool kPackSmall>
template <typename Scalar>
Front<kColumnBlock, kPackSmall>::Front(Scalar* end, size_t numDOFs) : _end(end), _numDOFs(numDOFs) {
  _numBlocks = (numDOFs + kColumnBlock - 1) / kColumnBlock;
  // This formula gives kColumnBlock when numDOFs is divisible by kColumnBlock.
  _edgeCols = ((numDOFs + kColumnBlock - 1) % kColumnBlock) + 1;
}

template <size_t kColumnBlock, bool kPackSmall>
constexpr size_t Front<kColumnBlock, kPackSmall>::StorageSize() const {
  return _numBlocks * (_numBlocks - 1) * kHalfSquare + _edgeCols * _numDOFs;
}

template <size_t kColumnBlock, bool kPackSmall>
template <typename Scalar>
MatrixView<Scalar> Front<kColumnBlock, kPackSmall>::Block(size_t block) const {
  if constexpr (kPackSmall) {
    // revBlock is between 1 and _numBlocks
    auto revBlock = _numBlocks - block;
    auto height = std::min(_numDOFs, revBlock * kColumnBlock);
    auto width = std::min(kColumnBlock, _numDOFs - (revBlock - 1) * kColumnBlock);
    auto* blockEnd = static_cast<Scalar*>(_end) - revBlock * (revBlock - 1) * kHalfSquare;
    return MakeBlock<Scalar>(blockEnd - height * width, height, width);
  } else {
    auto height = _numDOFs - block * kColumnBlock;
    auto width = std::min(kColumnBlock, height);
    auto offset = block * _numDOFs * kColumnBlock - block * (block - 1) * kHalfSquare;
    return MakeBlock<Scalar>(MemoryStart<Scalar>() + offset, height, width);
  }
}

template <size_t kColumnBlock, bool kPackSmall>
template <typename Scalar, size_t kDofsPerNode>
RangeByIterators<StairNodalIterator<Scalar, kDofsPerNode, kColumnBlock>>
Front<kColumnBlock, kPackSmall>::NodalColumns() const {
  static_assert(
      kColumnBlock % kDofsPerNode == 0,
      "Front column block size must be a multiple of DoFs per node");
  MOCHI_ASSERT_VERBOSE(
      _numDOFs % kDofsPerNode == 0, "Front size must be a multiple of DoFs per node");

  auto startPtr = MemoryStart<Scalar>();
  auto endPtr = static_cast<Scalar*>(_end);
  auto firstNumCols = kPackSmall ? _edgeCols : std::min(kColumnBlock, _numDOFs);
  auto blockEndPtr = startPtr + firstNumCols * _numDOFs;
  auto nDofs = static_cast<int>(_numDOFs);
  return {
      StairNodalIterator<Scalar, kDofsPerNode, kColumnBlock>{
          startPtr, blockEndPtr, endPtr, nDofs, nDofs},
      StairNodalIterator<Scalar, kDofsPerNode, kColumnBlock>{endPtr, endPtr, endPtr, nDofs, 0}};
}

template <size_t kColumnBlock, bool kPackSmall>
size_t Front<kColumnBlock, kPackSmall>::NumBlocks() const {
  return _numBlocks;
}

} // namespace mochi
