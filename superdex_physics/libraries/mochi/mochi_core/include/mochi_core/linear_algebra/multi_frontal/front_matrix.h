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
 *
 * Storage-wise, the *normal* order of storage in memory starts with the larger blocks.
 * The user of Front objects may decide that a *mirrored* storage, where the smaller
 * blocks occupy lower addresses, may be more efficient.
 * When _mirrored is true, offsets from one block to the next are
 * negatives, otherwise positive.
 * kColumnBlock should be chosen to be a multiple of the GEMM kernel (6 in Mochi as well as MKL)
 * to avoid the loop spill-over treatment. It should also be chosen large enough so that the rang-N
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

  /** @details If `isMirrored` is true, space points to the beginning of the front's
   * memory, otherwise is points to the end.
   *
   * @param space Pointer to the beginning or end of the front memory.
   * @param numDOFs Size of the front matrix in DOFs.
   * @param isMirrored Whether blocks are ordered from smallest to largest,
   *        or, largest to smallest.
   */
  template <typename Scalar>
  Front(Scalar* space, size_t numDOFs, bool isMirrored);

  /// @brief Number of rows/columns of the front matrix.
  [[nodiscard]] constexpr size_t Size() const {
    return _numDOFs;
  }

  template <typename Scalar>
  MatrixView<Scalar> Block(size_t block) const;

  template <typename Scalar>
  Generator<MatrixView<Scalar>> Blocks() const;

  template <typename Scalar, size_t kDofsPerNode>
  Generator<std::pair<
      int,
      MatrixView<
          Scalar,
          krylov::kDynamic,
          kDofsPerNode,
          krylov::Direction::ColMajor,
          krylov::kDynamic>>>
  NodalColumns() const;

  template <typename Scalar, size_t kDofsPerNode>
  RangeByIterators<StairNodalIterator<Scalar, kDofsPerNode, kColumnBlock>> NodalColumnsRange()
      const;

  [[nodiscard]] size_t NumBlocks() const;

  [[nodiscard]] constexpr size_t StorageSize() const;

  template <typename Scalar>
  static MatrixView<Scalar> MakeBlock(Scalar* ptr, size_t rows, size_t cols) {
    return MatrixView<Scalar>(ptr, static_cast<int>(rows), static_cast<int>(cols));
  }

  /**
   * @brief Pointer to the beginning of the front's allocation.
   *
   * @details `_refPtr` is the low address of the allocation when `kPackSmall == _mirrored`;
   * otherwise it points one past the last element, so `StorageSize()` is subtracted. This is the
   * same storage-layout axis as @ref BasePtr (which keys off `kPackSmall ^ _mirrored`).
   */
  template <typename Scalar>
  Scalar* MemoryStart() const {
    auto* v = reinterpret_cast<Scalar*>(_refPtr);
    if constexpr (kPackSmall) {
      return _mirrored ? v : v - StorageSize();
    } else {
      return _mirrored ? v - StorageSize() : v;
    }
  }

 private:
  bool _mirrored = false;
  /// @brief Reference pointer.
  void* _refPtr;
  /// @brief Number of DOFs in the front.
  size_t _numDOFs;
  /// @brief Number of blocks.
  size_t _numBlocks;
  /// @brief Number of columns for the edge block (first if kPackSmall, last otherwise)
  size_t _edgeCols;

  static constexpr auto kHalfSquare = kColumnBlock * kColumnBlock / 2;

  /** @brief Compute the base pointer for a block.
   * @details The base pointer is the pointer to the beginning of the block
   * when (kPackSmall ^ _mirrored), the end pointer of the block otherwise.
   * @tparam Scalar
   * @param block
   * @return
   */
  template <typename Scalar>
  Scalar* BasePtr(size_t block) const;
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
Front<kColumnBlock, kPackSmall>::Front(Scalar* space, size_t numDOFs, bool isMirrored)
    : _mirrored(isMirrored), _numDOFs(numDOFs) {
  _numBlocks = (numDOFs + kColumnBlock - 1) / kColumnBlock;
  // This formula gives kColumnBlock when numDOFs is divisible by kColumnBlock.
  _edgeCols = ((numDOFs + kColumnBlock - 1) % kColumnBlock) + 1;
  if constexpr (kPackSmall) {
    _refPtr = space;
  } else {
    auto s = static_cast<ptrdiff_t>(StorageSize());
    _refPtr = space + (_mirrored ? s : -s);
  }
}

template <size_t kColumnBlock, bool kPackSmall>
constexpr size_t Front<kColumnBlock, kPackSmall>::StorageSize() const {
  return _numBlocks * (_numBlocks - 1) * kHalfSquare + _edgeCols * _numDOFs;
}

template <size_t kColumnBlock, bool kPackSmall>
template <typename Scalar>
MatrixView<Scalar> Front<kColumnBlock, kPackSmall>::Block(size_t block) const {
  Scalar* v = BasePtr<Scalar>(block);
  if constexpr (kPackSmall) {
    auto revBlock = _numBlocks - block;
    auto height = std::min(_numDOFs, revBlock * kColumnBlock);
    auto width = std::min(kColumnBlock, _numDOFs - (revBlock - 1) * kColumnBlock);
    return MakeBlock<Scalar>((_mirrored != kPackSmall) ? v - height * width : v, height, width);
  } else {
    auto height = _numDOFs - block * kColumnBlock;
    auto width = std::min(kColumnBlock, height);
    return MakeBlock<Scalar>((_mirrored != kPackSmall) ? v - height * width : v, height, width);
  }
}

template <size_t kColumnBlock, bool kPackSmall>
template <typename Scalar>
Generator<MatrixView<Scalar>> Front<kColumnBlock, kPackSmall>::Blocks() const {
  if constexpr (kPackSmall) {
    // Get the pointer to the reference pointer of the first block.
    ptrdiff_t offset = _numBlocks * (_numBlocks - 1) * (_mirrored ? kHalfSquare : -kHalfSquare);
    auto values = static_cast<Scalar*>(_refPtr) + offset;
    auto cols = _edgeCols; // The first block has a unique width
    auto rows = _numDOFs;
    while (rows > 0) {
      co_yield MatrixView<Scalar>{
          !_mirrored ? values - rows * cols : values,
          static_cast<int>(rows),
          static_cast<int>(cols)};
      rows -= cols;
      cols = kColumnBlock;
      if (_mirrored) {
        values -= rows * cols;
      } else {
        values += rows * cols;
      }
    }
  } else {
    ptrdiff_t cols = std::min<ptrdiff_t>(kColumnBlock, _numDOFs);
    ptrdiff_t rows = _numDOFs;
    // When kPackSmall is false, the `_refPtr` pointer
    // points to the beginning (if !_mirrored) or end (otherwise) of block 0.
    // Get the pointer to the beginning of the first block.
    ptrdiff_t offset = (_mirrored ? -cols * rows : 0);
    auto values = static_cast<Scalar*>(_refPtr) + offset;
    while (rows > 0) {
      co_yield MatrixView<Scalar>{values, static_cast<int>(rows), static_cast<int>(cols)};
      if (!_mirrored) {
        values += rows * cols;
      }
      rows -= cols;
      cols = std::min<ptrdiff_t>(kColumnBlock, rows);
      if (_mirrored) {
        values -= rows * cols;
      }
    }
  }
}
template <size_t kColumnBlock, bool kPackSmall>
template <typename Scalar, size_t kDofsPerNode>
Generator<std::pair<
    int,
    MatrixView<
        Scalar,
        krylov::kDynamic,
        kDofsPerNode,
        krylov::Direction::ColMajor,
        krylov::kDynamic>>>
Front<kColumnBlock, kPackSmall>::NodalColumns() const {
  int nodeIndex = 0;
  auto rows = static_cast<int>(_numDOFs);
  if constexpr (kPackSmall) {
    // Get the pointer to the beginning of the first block.
    ptrdiff_t offset = _numBlocks * (_numBlocks - 1) * (_mirrored ? kHalfSquare : -kHalfSquare);
    auto values = static_cast<Scalar*>(_refPtr) + offset;
    auto cols = static_cast<int>(_edgeCols); // The first block has a unique width
    while (rows > 0) {
      auto blockStart = !_mirrored ? values - rows * cols : values;
      for (size_t c = 0; c < cols; c += kDofsPerNode) {
        co_yield {
            nodeIndex++,
            MatrixView<
                Scalar,
                krylov::kDynamic,
                kDofsPerNode,
                krylov::Direction::ColMajor,
                krylov::kDynamic>{
                blockStart + c * rows + c,
                static_cast<int>(rows - c),
                static_cast<int>(kDofsPerNode),
                rows}};
      }
      rows -= cols;
      cols = kColumnBlock;
      if (_mirrored) {
        values -= rows * cols;
      } else {
        values += rows * cols;
      }
    }
  } else {
    auto cols = std::min(kColumnBlock, _numDOFs);
    // When kPackSmall is false, the `_refPtr` pointer
    // points to the beginning (if !_mirrored) or end (otherwise) of block 0.
    // Get the pointer to the beginning of the first block.
    ptrdiff_t offset = (_mirrored ? -cols * rows : 0);
    auto values = static_cast<Scalar*>(_refPtr) + offset;
    while (rows > 0) {
      auto blockStart = values;
      for (size_t c = 0; c < cols; c += kDofsPerNode) {
        co_yield {
            nodeIndex++,
            MatrixView<
                Scalar,
                krylov::kDynamic,
                kDofsPerNode,
                krylov::Direction::ColMajor,
                krylov::kDynamic>{
                blockStart + c * rows + c,
                static_cast<int>(rows - c),
                static_cast<int>(kDofsPerNode),
                rows}};
      }
      if (!_mirrored) {
        values += rows * cols;
      }
      rows -= cols;
      cols = std::min<int>(kColumnBlock, rows);
      if (_mirrored) {
        values -= rows * cols;
      }
    }
  }
}
template <size_t kColumnBlock, bool kPackSmall>
template <typename Scalar, size_t kDofsPerNode>
RangeByIterators<StairNodalIterator<Scalar, kDofsPerNode, kColumnBlock>>
Front<kColumnBlock, kPackSmall>::NodalColumnsRange() const {
  MOCHI_ASSERT_VERBOSE(!_mirrored, "Mirrored front are not yet supported by StairNodalIterators");
  auto startPtr = static_cast<Scalar*>(_refPtr) - StorageSize();
  auto endPtr = static_cast<Scalar*>(_refPtr);
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

template <size_t kColumnBlock, bool kPackSmall>
template <typename Scalar>
Scalar* Front<kColumnBlock, kPackSmall>::BasePtr(size_t block) const {
  auto values = static_cast<Scalar*>(_refPtr);
  if constexpr (kPackSmall) {
    // revBlock is between 1 and _numBlocks
    auto revBlock = _numBlocks - block;
    ptrdiff_t offset = revBlock * (revBlock - 1) * kHalfSquare;
    return values + (_mirrored ? offset : -offset);
  } else {
    ptrdiff_t offset = block * _numDOFs * kColumnBlock - block * (block - 1) * kHalfSquare;
    return values + (_mirrored ? -offset : offset);
  }
}

} // namespace mochi
