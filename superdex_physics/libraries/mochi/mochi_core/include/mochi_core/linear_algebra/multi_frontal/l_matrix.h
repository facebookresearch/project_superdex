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
#include <mochi_core/linear_algebra/multi_frontal/stair_matrix.h>
#include <mochi_core/utils/dynamic_array.h>

namespace mochi {

/**
 * @brief Manages the layout and storage organization of L factor blocks in a multi-frontal solver.
 *
 * LShape determines the memory layout for storing L factor matrices from a multi-frontal
 * factorization. Each supernode in the elimination tree gets its own stair-shaped block
 * in the L factor, and LShape tracks the offset and dimensions of each block.
 *
 * @tparam kBlockSize The block size used for stair matrix organization (must be even)
 */
template <size_t kBlockSize>
class LShape {
 public:
  /**
   * @brief Constructs an LShape organizer for the given elimination tree.
   *
   * Computes the memory layout for all supernodes in the elimination tree,
   * determining offsets and dimensions for each L factor block.
   *
   * @param tree The elimination data defining the supernode structure
   * @param dofsPerNode Degrees of freedom per node in the mesh
   */
  LShape(SymbolicEliminationTree const& tree, size_t dofsPerNode);

  /**
   * @brief Returns the total storage size required for all L factor blocks.
   *
   * @return Total number of scalar elements needed to store all L factors
   */
  [[nodiscard]] size_t StorageSize() const {
    return _snInfo.back().offset;
  }

  /**
   * @brief Returns a stair matrix view for the L factor of a specific supernode.
   *
   * Provides access to the L factor block associated with the given supernode,
   * positioned at the appropriate offset within the full L storage.
   *
   * @tparam Scalar The scalar type of the matrix elements
   * @param lMem Memory span containing all L factor data
   * @param superNode Index of the supernode whose L factor to retrieve
   * @return StairMatrixView providing block-based access to the L factor
   */
  template <typename Scalar>
  [[nodiscard]] StairMatrixView<Scalar, kBlockSize> LforSN(Span<Scalar> lMem, size_t superNode);

 private:
  /**
   * @brief Storage information for a single supernode's L factor block.
   */
  struct SNInfo {
    size_t offset; ///< Offset in the global L storage where this block starts
    size_t rows; ///< Number of rows in this L factor block
    size_t cols; ///< Number of columns in this L factor block
  };

  /// @brief Storage info for each supernode, plus one final entry with the total size
  DynamicArray<SNInfo> _snInfo;
};

template <typename Scalar, size_t kBlockSize>
class LMatrix {
 public:
  LMatrix(SymbolicEliminationTree const& tree, size_t dofsPerNode);
  StairMatrixView<Scalar, kBlockSize> LforSN(size_t superNode);
  void SetZero() {
    std::fill_n(_lMem.get(), _memSize, Scalar{0});
  }

  Span<Scalar const> ConstData() const {
    return {_lMem.get(), _memSize};
  }

 private:
  LShape<kBlockSize> _lShape;
  size_t _memSize;
  std::unique_ptr<Scalar[]> _lMem;
};

// ============================================================================
// Template Implementation
// ============================================================================

template <size_t kBlockSize>
LShape<kBlockSize>::LShape(SymbolicEliminationTree const& tree, size_t dofsPerNode) {
  _snInfo.reserve(tree.NumSuperNodes() + 1);
  size_t offset = 0;

  // Compute layout for each supernode
  for (int i = 0; i < tree.NumSuperNodes(); ++i) {
    auto superNumNodes = tree.SuperSize(i);
    auto superLRows = tree.SuperColSize(i);
    _snInfo.push_back({offset, superLRows * dofsPerNode, superNumNodes * dofsPerNode});
    offset += StairMatrixSize(superLRows * dofsPerNode, superNumNodes * dofsPerNode, kBlockSize);
  }
  // Add a final sentinel entry with total size
  _snInfo.push_back({offset, 0, 0});
}

template <size_t kBlockSize>
template <typename Scalar>
StairMatrixView<Scalar, kBlockSize> LShape<kBlockSize>::LforSN(
    Span<Scalar> lMem,
    size_t superNode) {
  auto& info = _snInfo[superNode];
  return StairMatrixView<Scalar, kBlockSize>(lMem.data() + info.offset, info.rows, info.cols);
}

template <typename Scalar, size_t kBlockSize>
LMatrix<Scalar, kBlockSize>::LMatrix(SymbolicEliminationTree const& tree, size_t dofsPerNode)
    : _lShape(tree, dofsPerNode), _memSize(_lShape.StorageSize()), _lMem(new Scalar[_memSize]) {}

template <typename Scalar, size_t kBlockSize>
StairMatrixView<Scalar, kBlockSize> LMatrix<Scalar, kBlockSize>::LforSN(size_t superNode) {
  return _lShape.template LforSN<Scalar>({_lMem.get(), _memSize}, superNode);
}

} // namespace mochi
