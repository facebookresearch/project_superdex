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

#include <mochi_core/linear_algebra/strided_matrix.h>

#include <type_traits>

namespace mochi {

MOCHI_ANY auto StridedCofactors3x3(IsLooselyMatrix auto& F) {
  using Scalar = std::remove_const_t<std::decay_t<decltype(F(0, 0))>>;
  return StridedMatrix<Scalar, 3, 3>{
      {F(1, 1) * F(2, 2) - F(2, 1) * F(1, 2),
       F(2, 1) * F(0, 2) - F(0, 1) * F(2, 2),
       F(0, 1) * F(1, 2) - F(1, 1) * F(0, 2)},
      {F(2, 0) * F(1, 2) - F(1, 0) * F(2, 2),
       F(0, 0) * F(2, 2) - F(2, 0) * F(0, 2),
       F(1, 0) * F(0, 2) - F(0, 0) * F(1, 2)},
      {F(1, 0) * F(2, 1) - F(2, 0) * F(1, 1),
       F(2, 0) * F(0, 1) - F(0, 0) * F(2, 1),
       F(0, 0) * F(1, 1) - F(1, 0) * F(0, 1)},
  };
}

template <typename T>
struct DetInverse {
  StridedMatrix<T, 3, 3> inverse;
  T determinant;
};

template <
    bool withJacobian = false,
    typename T,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim,
    int kStride>
MOCHI_ANY auto Invert(
    StridedMatrix<T, 3, 3, kStride, kMajorDirection, kOwnership, kLeadingDim> const& A) {
  auto cofactors = StridedCofactors3x3(A);
  auto determinant =
      A(0, 0) * cofactors(0, 0) + A(0, 1) * cofactors(0, 1) + A(0, 2) * cofactors(0, 2);
  auto ood = T(1.0) / determinant;
  if constexpr (withJacobian) {
    return DetInverse<std::decay_t<T>>{
        StridedMatrix<T, 3, 3>{ood * cofactors.Transpose()}, determinant};
  } else {
    return StridedMatrix<T, 3, 3>{ood * cofactors.Transpose()};
  }
}

template <IsStridedMatrix A, IsStridedMatrix B>
MOCHI_ANY auto Dot(A&& a, B&& b) {
  static_assert(
      std::is_same_v<decltype(a.CECols()), details::IntOrEmpty<1>> &&
      std::is_same_v<decltype(b.CECols()), details::IntOrEmpty<1>>);
  auto res = a(0, 0) * b(0, 0);
  for (int i = 1; i < a.Rows(); ++i) {
    res += a(i, 0) * b(i, 0);
  }
  return res;
}

} // namespace mochi
