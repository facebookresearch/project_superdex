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

#include <gtest/gtest.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace mochi::test {

template <typename S, typename MA, typename MB, typename MC>
void MultiplyMatrices(S alpha, MA const& A, MB const& B, MC& C) {
  using ResScalar = std::decay_t<decltype(C(0, 0))>;

  for (int r = 0; r < C.Rows(); ++r) {
    for (int c = 0; c < C.Cols(); ++c) {
      S res{0};
      for (int j = 0; j < A.Cols(); ++j) {
        res += A(r, j) * B(j, c);
      }
      C(r, c) = ResScalar(alpha * res);
    }
  }
}

template <typename S, typename MA, typename MB, typename MC>
void AddMatrices(S alpha, MA const& A, MB const& B, MC& C) {
  using ResScalar = std::decay_t<decltype(C(0, 0))>;

  for (int r = 0; r < C.Rows(); ++r) {
    for (int c = 0; c < C.Cols(); ++c) {
      C(r, c) = ResScalar(alpha * A(r, c) + B(r, c));
    }
  }
}

template <typename Scalar>
Scalar GetTol(int n) {
  return 2 * n * std::numeric_limits<Scalar>::epsilon();
}

template <typename MA, typename MB, typename Scalar>
[[nodiscard]] bool IsNear(MA const& A, MB const& B, Scalar tol) {
  if (A.Cols() != B.Cols() || A.Rows() != B.Rows()) {
    return false;
  }
  for (int r = 0; r < A.Rows(); ++r) {
    for (int c = 0; c < B.Cols(); ++c) {
      if (!(std::abs(A(r, c) - B(r, c)) <= tol)) {
        return false;
      }
    }
  }
  return true;
}

template <typename MA, typename MB, typename Scalar>
[[nodiscard]] bool IsNearRTol(MA const& A, MB const& B, Scalar tol) {
  if (A.Cols() != B.Cols() || A.Rows() != B.Rows()) {
    return false;
  }
  for (int r = 0; r < A.Rows(); ++r) {
    for (int c = 0; c < B.Cols(); ++c) {
      if (!(2 * std::abs(A(r, c) - B(r, c)) <= tol * (std::abs(A(r, c)) + std::abs(B(r, c))))) {
        return false;
      }
    }
  }
  return true;
}

} // namespace mochi::test
