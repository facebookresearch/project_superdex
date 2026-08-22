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
#include <mochi_core/utils/debug.h>

namespace mochi {

/**
 * @brief A view of a rectangular matrix into a one-dimensional array of square fixed-size blocks.
 *
 * BlockOneDView provides a convenient interface to work with arrays of fixed-size square
 * blocks stored as a rectangular matrix. It can operate in either row-major or column-major
 * layout, depending on the template parameter kDir.
 *
 * @tparam Scalar The scalar type of the matrix elements (e.g., float, double)
 * @tparam kBlockSize The size of each square block (kBlockSize x kBlockSize)
 * @tparam kDir The memory layout direction (RowMajor or ColMajor)
 * @tparam CRIdx The index type for dimensions and indices (default: int)
 *
 * Memory Layout (with underlying matrix size):
 * - RowMajor: Blocks are arranged horizontally (kBlockSize rows, numBlocks*kBlockSize columns)
 * - ColMajor: Blocks are arranged vertically (numBlocks*kBlockSize rows, kBlockSize columns)
 *
 * Example usage:
 * @code
 * float data[125]; // Storage for blocks
 * BlockRowView<float, 5> view(data, 25, 5); // 5 blocks of size 5x5 in row-major layout
 * auto block = view[0]; // Access first block
 * view(0, 10) = 1.0f; // Access element at row 0, column 10 of the underlying matrix.
 * @endcode
 */
template <typename Scalar, int kBlockSize, krylov::Direction kDir, typename CRIdx = int>
struct BlockOneDView {
  /** @brief Type alias for the block type returned by operator[] */
  using BlockType = MatrixView<Scalar, kBlockSize, kBlockSize, kDir, krylov::kDynamic>;

  /**
   * @brief Constructs a row-major BlockOneDView.
   * @param v_ Pointer to the underlying data array
   * @param ld_ Leading dimension of the underlying matrix
   * @param numBlocks_ Number of blocks in the view
   *
   * For row-major layout, blocks are arranged horizontally:
   * - The underlying matrix has kBlockSize rows and numBlocks*kBlockSize columns
   */
  BlockOneDView(Scalar* v_, CRIdx ld_, CRIdx numBlocks_)
    requires(kDir == krylov::Direction::RowMajor)
      : mat{v_, kBlockSize, numBlocks_ * kBlockSize, ld_}, numBlocks(numBlocks_) {}

  /**
   * @brief Constructs a column-major BlockOneDView.
   * @param v_ Pointer to the underlying data array
   * @param ld_ Leading dimension of the underlying matrix
   * @param numBlocks_ Number of blocks in the view
   *
   * For column-major layout, blocks are arranged vertically:
   * - The underlying matrix has numBlocks*kBlockSize rows and kBlockSize columns
   */
  BlockOneDView(Scalar* v_, CRIdx ld_, CRIdx numBlocks_)
    requires(kDir == krylov::Direction::ColMajor)
      : mat{v_, numBlocks_ * kBlockSize, kBlockSize, ld_}, numBlocks(numBlocks_) {}

  /** @brief Copy constructor */
  BlockOneDView(BlockOneDView const& b) = default;

  /**
   * @brief Returns the number of blocks in the view.
   * @return Number of blocks
   */
  [[nodiscard]] MOCHI_FORCE_INLINE CRIdx NumBlocks() const {
    return numBlocks;
  }

  /**
   * @brief Returns the leading dimension of the underlying matrix.
   * @return Leading dimension
   */
  [[nodiscard]] MOCHI_FORCE_INLINE CRIdx LeadDim() const {
    return static_cast<CRIdx>(mat.LeadDim());
  }

  /**
   * @brief Returns a pointer to the underlying data array.
   * @return Pointer to data.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE Scalar* Data() {
    return mat.Data();
  }

  /**
   * @brief Returns a const pointer to the underlying data array.
   * @return Const pointer to data.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE Scalar const* Data() const {
    return mat.Data();
  }

  /**
   * @brief Lower-case overload for compatibility with STL containers.
   * @return Const pointer to data.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE Scalar const* data() const {
    return mat.data();
  }

  /**
   * @brief Returns a view of the i-th block.
   * @param i Block index (0 <= i < NumBlocks())
   * @return MatrixView representing the i-th block
   */
  [[nodiscard]] MOCHI_FORCE_INLINE auto operator[](CRIdx i) {
    MOCHI_ASSERT_VERBOSE((i >= 0) && (i < numBlocks), "Index out of range");
    if constexpr (kDir == krylov::Direction::RowMajor) {
      return mat.template Block<kBlockSize, kBlockSize>(0, i * kBlockSize, kBlockSize, kBlockSize);
    } else {
      return mat.template Block<kBlockSize, kBlockSize>(i * kBlockSize, 0, kBlockSize, kBlockSize);
    }
  }

  /**
   * @brief Returns a const view of the i-th block.
   * @param i Block index (0 <= i < NumBlocks())
   * @return Const MatrixView representing the i-th block
   */
  [[nodiscard]] MOCHI_FORCE_INLINE auto operator[](CRIdx i) const {
    MOCHI_ASSERT_VERBOSE((i >= 0) && (i < numBlocks), "Index out of range");
    if constexpr (kDir == krylov::Direction::RowMajor) {
      return mat.template Block<kBlockSize, kBlockSize>(0, i * kBlockSize, kBlockSize, kBlockSize);
    } else {
      return mat.template Block<kBlockSize, kBlockSize>(i * kBlockSize, 0, kBlockSize, kBlockSize);
    }
  }

  /**
   * @brief Access element at global position (r, c).
   * @param r Row index (0 <= r < kBlockSize for RowMajor, 0 <= r < NumBlocks()*kBlockSize for
   * ColMajor)
   * @param c Column index (0 <= c < NumBlocks()*kBlockSize for RowMajor, 0 <= c < kBlockSize for
   * ColMajor)
   * @return Reference to the element
   */
  [[nodiscard]] MOCHI_FORCE_INLINE Scalar& operator()(int r, int c) {
    return mat(r, c);
  }

  /**
   * @brief Access element at global position (r, c) (const version).
   * @param r Row index
   * @param c Column index
   * @return Const reference to the element
   */
  [[nodiscard]] MOCHI_FORCE_INLINE Scalar const& operator()(int r, int c) const {
    return mat(r, c);
  }

  /**
   * @brief Extract the first i blocks as a new view.
   * @param i Number of blocks to extract from the beginning
   * @return BlockOneDView containing the first i blocks
   */
  [[nodiscard]] MOCHI_FORCE_INLINE BlockOneDView FirstBlocks(CRIdx i) {
    MOCHI_ASSERT_VERBOSE((i >= 0) && (i <= numBlocks), "Index out of range");
    return {mat.Data(), mat.LeadDim(), i};
  }

  /**
   * @brief Extract the first i blocks as a new const view.
   * @param i Number of blocks to extract from the beginning
   * @return Const BlockOneDView containing the first i blocks
   */
  [[nodiscard]] MOCHI_FORCE_INLINE BlockOneDView<Scalar const, kBlockSize, kDir, CRIdx> FirstBlocks(
      CRIdx i) const {
    MOCHI_ASSERT_VERBOSE((i >= 0) && (i <= numBlocks), "Index out of range");
    return {mat.Data(), mat.LeadDim(), i};
  }

  /**
   * @brief Extract the last i blocks as a new view.
   * @param i Number of blocks to extract from the end
   * @return BlockOneDView containing the last i blocks
   */
  [[nodiscard]] MOCHI_FORCE_INLINE BlockOneDView LastBlocks(CRIdx i) {
    MOCHI_ASSERT_VERBOSE((i >= 0) && (i <= numBlocks), "Index out of range");
    return {mat.Data() + (numBlocks - i) * kBlockSize, mat.LeadDim(), i};
  }

  /**
   * @brief Extract the last i blocks as a new const view.
   * @param i Number of blocks to extract from the end
   * @return Const BlockOneDView containing the last i blocks
   */
  [[nodiscard]] MOCHI_FORCE_INLINE BlockOneDView<Scalar const, kBlockSize, kDir, CRIdx> LastBlocks(
      CRIdx i) const {
    MOCHI_ASSERT_VERBOSE((i >= 0) && (i <= numBlocks), "Index out of range");
    return {mat.Data() + (numBlocks - i) * kBlockSize, mat.LeadDim(), i};
  }

  /**
   * @brief Returns a view of the i-th row in the underlying matrix.
   * @param i Row index
   * @return View of the specified row
   */
  [[nodiscard]] MOCHI_FORCE_INLINE auto Row(int i) {
    return mat.Row(i);
  }

  /**
   * @brief Returns a const view of the i-th row in the underlying matrix.
   * @param i Row index
   * @return Const view of the specified row
   */
  [[nodiscard]] MOCHI_FORCE_INLINE auto Row(int i) const {
    return mat.Row(i);
  }

  /**
   * @brief Returns a view of the i-th column in the underlying matrix.
   * @param i Column index
   * @return View of the specified column
   */
  [[nodiscard]] MOCHI_FORCE_INLINE auto Col(int i) {
    return mat.Col(i);
  }

  /**
   * @brief Returns a const view of the i-th column in the underlying matrix.
   * @param i Column index
   * @return Const view of the specified column
   */
  [[nodiscard]] MOCHI_FORCE_INLINE auto Col(int i) const {
    return mat.Col(i);
  }

  /**
   * @brief Sets all elements in the view to zero.
   */
  MOCHI_FORCE_INLINE void SetZero() {
    mat.SetZero();
  }

  /**
   * @brief Returns the underlying matrix view.
   * @return The underlying MatrixView object
   */
  MOCHI_FORCE_INLINE auto& Underlying() {
    return mat;
  }

  /**
   * @brief Returns the underlying matrix view, const version.
   * @return Const version of the underlying MatrixView object
   */
  MOCHI_FORCE_INLINE auto& Underlying() const {
    return mat;
  }

  static constexpr auto kMatRows =
      kDir == krylov::Direction::RowMajor ? kBlockSize : krylov::kDynamic;
  static constexpr auto kMatCols =
      kDir == krylov::Direction::ColMajor ? kBlockSize : krylov::kDynamic;
  MatrixView<Scalar, kMatRows, kMatCols, kDir, krylov::kDynamic> mat;
  CRIdx numBlocks = 0;
};

/**
 * @brief Type alias for row-major block view.
 *
 * BlockRowView arranges blocks horizontally organized, creating a view with an underlying matrix of
 * size:
 * - kBlockSize rows
 * - numBlocks * kBlockSize columns
 *
 * @tparam Scalar The scalar type of the matrix elements
 * @tparam kBlockSize The size of each square block
 * @tparam CRIdx The index type (default: int)
 */
template <typename Scalar, int kBlockSize, typename CRIdx = int>
using BlockRowView = BlockOneDView<Scalar, kBlockSize, krylov::Direction::RowMajor, CRIdx>;

/**
 * @brief Type alias for column-major block view.
 *
 * BlockColView arranges blocks vertically in memory, creating a view with an underlying matrix of
 * size:
 * - numBlocks * kBlockSize rows
 * - kBlockSize columns
 *
 * @tparam Scalar The scalar type of the matrix elements
 * @tparam kBlockSize The size of each square block
 * @tparam CRIdx The index type (default: int)
 */
template <typename Scalar, int kBlockSize, typename CRIdx = int>
using BlockColView = BlockOneDView<Scalar, kBlockSize, krylov::Direction::ColMajor, CRIdx>;

} // namespace mochi
