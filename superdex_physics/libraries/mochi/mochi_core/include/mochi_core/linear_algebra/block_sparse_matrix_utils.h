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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

namespace mochi {

/// @brief Cast-copy BlockSparseMatrix with scalar entries to a BlockSparseMatrix with a different
/// scalar type.
///
/// @tparam ToBlockSparseMatrix The target block sparse matrix type for the conversion (must satisfy
/// IsBlockSparseMatrix).
/// @tparam FromScalar The source scalar type from the input BlockSparseMatrix.
/// @tparam kBlockSize The block size of the input BlockSparseMatrix.
/// @tparam FromIdx The source index type from the input BlockSparseMatrix.
/// @tparam FromPtr The source pointer type from the input BlockSparseMatrix.
/// @tparam FromStorage The source storage type from the input BlockSparseMatrix.
/// @param M The input BlockSparseMatrix to be converted.
/// @return ToBlockSparseMatrix with the converted scalar type.
template <
    typename ToBlockSparseMatrix,
    typename FromScalar,
    int kBlockSize,
    typename FromIdx,
    typename FromPtr,
    template <typename, typename...> typename FromStorage,
    MOCHI_CONCEPT(IsBlockSparseMatrix<ToBlockSparseMatrix>)>
[[nodiscard]] MOCHI_FORCE_INLINE ToBlockSparseMatrix
StaticCast(BlockSparseMatrix<FromScalar, kBlockSize, FromIdx, FromPtr, FromStorage> const& M) {
  static_assert(ToBlockSparseMatrix::kBlockSize == kBlockSize, "Block sizes must match.");
  return static_cast<ToBlockSparseMatrix>(M);
}

/// @brief Create the transpose block sparse matrix
template <
    typename Scalar_,
    int kBlockSize,
    typename CRIdx,
    typename Ptr_,
    template <typename, typename...> typename Storage>
auto Transpose(BlockSparseMatrix<Scalar_, kBlockSize, CRIdx, Ptr_, Storage> const& A);

} // namespace mochi

namespace mochi::details {
/// @brief Routine to compute the numerical values of the product
///
/// @param[in] A Input block sparse matrix
/// @param[in] B Input block sparse matrix
/// @param[out] AB Output block sparse matrix for the product A * B
///
/// @note A, B, and AB must have the same blocksize.
/// @note The routine does not modify the sparsity pattern of AB.
/// It will ONLY compute entries for the allocated non-zero entries in AB.
///
template <
    int kBlockSize,
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA,
    typename ScalarB,
    typename CRIdxB,
    typename PtrB,
    template <typename, typename...> typename StorageB,
    typename ScalarAB,
    typename CRIdxAB,
    typename PtrAB,
    template <typename, typename...> typename StorageAB>
void SparseMatProduct(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A,
    BlockSparseMatrix<ScalarB, kBlockSize, CRIdxB, PtrB, StorageB> const& B,
    BlockSparseMatrix<ScalarAB, kBlockSize, CRIdxAB, PtrAB, StorageAB>& AB);

} // namespace mochi::details

#include "block_sparse_matrix_utils_inl.h"
