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

#include <mochi_core/linear_algebra/lu.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/utils/nd_array.h>

namespace mochi {

/**
 Compute the combination number C(n, r) with r=0..N

 Args: integer n

 Returns:
   The combination number vector of C(n, r)
 */
template <int N>
consteval NdArray<int, N + 1> CombinationNumbers() {
  NdArray<int, N + 1> row{};
  row[0] = row[N] = 1;

  for (int i = 1; i < N; ++i) {
    row[i] = row[i - 1] * (N - i + 1) / i;
  }
  return row;
}

/**
 Evaluate the coefficients of a Bezier curve with respect to control points

 Args:
   combinations: the combination numbers
   t: the parameter value

 Returns:
   coeffs: the coefficients of the control points
 */
template <typename T, int N>
constexpr NdArray<T, N + 1> GetBezierCurveCoefficients(T t) {
  constexpr NdArray<int, N + 1> combinations = CombinationNumbers<N>();
  static_assert(N > 0, "N must be greater than 0");
  NdArray<T, N + 1> coeffs;
  NdArray<T, N + 1> tN; // Store t^n
  NdArray<T, N + 1> t_1N; // Store (1-t)^n
  // zero order
  tN[0] = 1_r;
  t_1N[0] = 1_r;
  // first order
  tN[1] = t;
  t_1N[1] = 1_r - t;
  // higher than second order
  for (int i = 2; i <= N; i++) {
    tN[i] = tN[i - 1] * tN[1];
    t_1N[i] = t_1N[i - 1] * t_1N[1];
  }
  for (int i = 0; i <= N; i++) {
    coeffs[i] = t_1N[N - i] * tN[i] * combinations[i];
  }
  return coeffs;
}

/**
 Evaluate the matrix for shifting the Bezier curve parameterization
 Specifically, suppose we have a curve p(t,c) and want to shift by dt,
 then we can call this function m=ShiftBezierCurveParameter(dt) to get m.
 Now suppose our control points is a vector c, then we have:
 p(t+dt,c)=p(t,m*c)

 Args:
  dt: time shifted

 Return:
  m: time shift matrix
 */
template <typename T, int N>
inline Matrix<T, N + 1, N + 1> ShiftBezierCurveParameter(T dt) {
  NdArray<real, N + 1> coeffs;
  Matrix<real, N + 1, N + 1> LHS, RHS;
  // Build basis matrix
  for (int i = 0; i <= N; i++) {
    coeffs = GetBezierCurveCoefficients<T, N>((real)i / (real)N);
    LHS.Row(i) = Transpose(AsConstView(coeffs));

    coeffs = GetBezierCurveCoefficients<T, N>((real)i / (real)N + dt);
    RHS.Row(i) = Transpose(AsConstView(coeffs));
  }
  LU<real, N + 1, N + 1>(LHS).Inverse(LHS);
  return LHS * RHS;
}

} // namespace mochi
