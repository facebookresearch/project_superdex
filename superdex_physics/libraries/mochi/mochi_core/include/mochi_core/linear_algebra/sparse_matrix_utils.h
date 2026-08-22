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

#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

namespace mochi {

/// @brief Cast-copy SparseMatrix with scalar entries to a SparseMatrix with a different scalar
/// type.
///
/// @tparam ToSparseMatrix The target sparse matrix type for the conversion (must satisfy
/// IsSparseMatrix).
/// @tparam FromScalar The source scalar type from the input SparseMatrix.
/// @tparam FromIdx The source index type from the input SparseMatrix.
/// @tparam FromPtr The source pointer type from the input SparseMatrix.
/// @tparam FromStorage The source storage type from the input SparseMatrix.
/// @param M The input SparseMatrix to be converted.
/// @return ToSparseMatrix with the converted scalar type.
template <
    typename ToSparseMatrix,
    typename FromScalar,
    typename FromIdx,
    typename FromPtr,
    template <typename, typename...> typename FromStorage,
    MOCHI_CONCEPT(IsSparseMatrix<ToSparseMatrix>)>
[[nodiscard]] MOCHI_FORCE_INLINE ToSparseMatrix
StaticCast(SparseMatrix<FromScalar, FromIdx, FromPtr, FromStorage> const& M) {
  return static_cast<ToSparseMatrix>(M);
}

template <
    typename Scalar_,
    typename CRIdx,
    typename Ptr_,
    template <typename, typename...> typename Storage>
auto Transpose(SparseMatrix<Scalar_, CRIdx, Ptr_, Storage> const& A);

template <
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA,
    typename ScalarB,
    typename CRIdxB,
    typename PtrB,
    template <typename, typename...> typename StorageB>
auto operator*(
    SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& A,
    SparseMatrix<ScalarB, CRIdxB, PtrB, StorageB> const& B);

} // namespace mochi

#include "sparse_matrix_utils_inl.h"
