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

#include "any_matrix_utils.h"

#include <mochi_core/utils/nd_array_utils.h>

namespace mochi {

template <typename ToAnyMatrix, typename Scalar, MOCHI_CONCEPT_DEF(IsAnyMatrixVariant<ToAnyMatrix>)>
ToAnyMatrix StaticCast(AnyMatrix<Scalar> const& anyMat) {
  using ToScalar = typename krylov::details::MatTraits<ToAnyMatrix>::Scalar;
  static_assert(!std::is_const_v<ToScalar>, "Destination scalar type must be non-const");
  static_assert(
      std::variant_size_v<AnyMatrix<Scalar>> == 4,
      "Please update the code below if AnyMatrix is updated");
  if (auto* bspMat3 = std::get_if<BlockSparseMatrix<Scalar, 3>>(&anyMat)) {
    return ToAnyMatrix(StaticCast<BlockSparseMatrix<ToScalar, 3>>(*bspMat3));
  }
  if (auto* bspMat4 = std::get_if<BlockSparseMatrix<Scalar, 4>>(&anyMat)) {
    return ToAnyMatrix(StaticCast<BlockSparseMatrix<ToScalar, 4>>(*bspMat4));
  }
  if (auto* spMat = std::get_if<SparseMatrix<Scalar>>(&anyMat)) {
    return ToAnyMatrix(StaticCast<SparseMatrix<ToScalar>>(*spMat));
  }
  auto* denseMat = std::get_if<Matrix<Scalar>>(&anyMat);
  MOCHI_ASSERT_VERBOSE(denseMat != nullptr, "Unsupported variant alternative type");
  return ToAnyMatrix(StaticCast<Matrix<ToScalar>>(*denseMat));
}

template <typename ToAnyMatrix, typename Scalar, MOCHI_CONCEPT_DEF(IsAnyMatrixVariant<ToAnyMatrix>)>
ToAnyMatrix StaticCast(AnyMatrixView<Scalar> const& anyMat) {
  using ToScalar = typename krylov::details::MatTraits<ToAnyMatrix>::Scalar;
  static_assert(!std::is_const_v<ToScalar>, "Destination scalar type must be non-const");
  static_assert(
      std::variant_size_v<ToAnyMatrix> == std::variant_size_v<AnyMatrixView<Scalar>>,
      "Input and output variant sizes must match");
  static_assert(
      std::variant_size_v<AnyMatrixView<Scalar>> == 4,
      "Please update the code below if AnyMatrixView is updated");
  if (auto* bspMat3 = std::get_if<BlockSparseMatrixView<Scalar, 3>>(&anyMat)) {
    return ToAnyMatrix(StaticCast<BlockSparseMatrix<ToScalar, 3>>(*bspMat3));
  }
  if (auto* bspMat4 = std::get_if<BlockSparseMatrixView<Scalar, 4>>(&anyMat)) {
    return ToAnyMatrix(StaticCast<BlockSparseMatrix<ToScalar, 4>>(*bspMat4));
  }
  if (auto* spMat = std::get_if<SparseMatrixView<Scalar>>(&anyMat)) {
    return ToAnyMatrix(StaticCast<SparseMatrix<ToScalar>>(*spMat));
  }
  auto* denseMat = std::get_if<MatrixView<Scalar>>(&anyMat);
  MOCHI_ASSERT_VERBOSE(denseMat != nullptr, "Unsupported variant alternative type");
  return ToAnyMatrix(StaticCast<Matrix<ToScalar>>(*denseMat));
}

} // namespace mochi
