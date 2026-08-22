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

#include <mochi_core/linear_algebra/base_enums.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/utils/span_utils.h>

namespace mochi {

/// @brief Cast-copy Matrix with scalar entries to a different scalar type.
///
/// @tparam ToMatrix The target matrix type for the conversion (must satisfy IsHostMatrix).
/// @tparam FromScalar The source scalar type from the input Matrix.
/// @tparam kRowsAtCompileTime The number of rows of the input Matrix.
/// @tparam kColsAtCompileTime The number of columns of the input Matrix.
/// @tparam kMajorDirection The major direction of the input Matrix.
/// @tparam kOwnership The ownership of the input Matrix.
/// @tparam kLeadingDim The leading dimension of the input Matrix.
/// @param M The input Matrix to be converted.
/// @return ToMatrix with the converted scalar type.
template <
    typename ToMatrix,
    typename FromScalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim,
    MOCHI_CONCEPT(IsHostMatrix<ToMatrix>)>
[[nodiscard]] ToMatrix StaticCast(
    Matrix<
        FromScalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& M) {
  using ToScalar = typename ToMatrix::Scalar;
  static_assert(!std::is_const_v<ToScalar>, "Destination type must be non-const");
  static_assert(
      krylov::details::MatTraits<ToMatrix>::kMajorDir == kMajorDirection,
      "Major direction must match");
  static_assert(ToMatrix::kIsOwner, "Destination type must own matrix");
  static_assert(
      krylov::details::MatTraits<ToMatrix>::kLeadDim == krylov::kAutomaticLeadDim,
      "Only automatic leading dimension is implemented");
  //
  ToMatrix out(M.Rows(), M.Cols());
  if ((kLeadingDim == krylov::kAutomaticLeadDim) || (M.StorageSize() == M.Rows() * M.Cols())) {
    auto const matSize = size_t(M.StorageSize());
    StaticCast<ToScalar>(Span{M.data(), matSize}, Span{out.data(), matSize});
  } else {
    if constexpr (kMajorDirection == krylov::Direction::RowMajor) {
      for (int i = 0; i < M.Rows(); ++i) {
        StaticCast<ToScalar>(Span{M.Row(i).data(), M.Cols()}, Span{out.Row(i).data(), out.Cols()});
      }
    } else {
      static_assert(kMajorDirection == krylov::Direction::ColMajor);
      for (int j = 0; j < M.Cols(); ++j) {
        StaticCast<ToScalar>(Span{M.Col(j).data(), M.Rows()}, Span{out.Col(j).data(), out.Rows()});
      }
    }
  }
  return out;
}

} // namespace mochi
