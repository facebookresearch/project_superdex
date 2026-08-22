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

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/bernstein_polynomials.h>

using namespace mochi;

TEST(BernsteinPolynomials, SumOfCombination) {
  {
    auto combinations = CombinationNumbers<1>();
    EXPECT_NEAR_EQ(Pow(2, 1), Sum(combinations));
  }
  {
    auto combinations = CombinationNumbers<2>();
    EXPECT_NEAR_EQ(Pow(2, 2), Sum(combinations));
  }
  {
    auto combinations = CombinationNumbers<3>();
    EXPECT_NEAR_EQ(Pow(2, 3), Sum(combinations));
  }
  {
    auto combinations = CombinationNumbers<4>();
    EXPECT_NEAR_EQ(Pow(2, 4), Sum(combinations));
  }
}

TEST(BernsteinPolynomials, SumToOne) {
  constexpr int N = 10;
  for (int i = 0; i < 100; i++) {
    real alpha = real(i) / 100_r;
    NdArray<real, N + 1> coeffs = GetBezierCurveCoefficients<real, N>(alpha);
    EXPECT_NEAR_EQ(1_r, Sum(coeffs));
  }
}

TEST(BernsteinPolynomials, Shift) {
  constexpr int N = 10;
  mochi_default_random_engine engine;
  for (int i = 0; i < 100; i++) {
    real alpha = real(i) / 100_r;
    real dt = real(i) / 100_r;

    NdArray<real, N + 1> params;
    for (int j = 0; j <= N; j++) {
      params[j] = RandomUniformValue(engine, -1_r, 1_r);
    }
    NdArray<real, N + 1> paramsShifted;
    AsView(paramsShifted) = ShiftBezierCurveParameter<real, N>(dt) * AsConstView(params);

    NdArray<real, N + 1> coeffs = GetBezierCurveCoefficients<real, N>(alpha);
    NdArray<real, N + 1> coeffsShifted = GetBezierCurveCoefficients<real, N>(alpha - dt);
    real value = Dot(coeffs, params);
    real valueShifted = Dot(coeffsShifted, paramsShifted);
    EXPECT_NEAR_EQ(value, valueShifted);
  }
}

TEST(BernsteinPolynomials, SpecialCase) {
  constexpr int N = 10;
  constexpr NdArray<real, N + 1> coeffsZero = GetBezierCurveCoefficients<real, N>(0_r);
  constexpr NdArray<real, N + 1> coeffsOne = GetBezierCurveCoefficients<real, N>(1_r);
  for (int i = 0; i < N + 1; i++) {
    if (i == 0) {
      EXPECT_NEAR_EQ(0_r, coeffsOne[i]);
      EXPECT_NEAR_EQ(1_r, coeffsZero[i]);
    } else if (i == N) {
      EXPECT_NEAR_EQ(1_r, coeffsOne[i]);
      EXPECT_NEAR_EQ(0_r, coeffsZero[i]);
    } else {
      EXPECT_NEAR_EQ(0_r, coeffsOne[i]);
      EXPECT_NEAR_EQ(0_r, coeffsZero[i]);
    }
  }
}

TEST(BernsteinPolynomials, Groundtruth) {
  // first order
  for (int i = 0; i < 100; i++) {
    real alpha = real(i) / 100_r;
    NdArray<real, 2> coeffs = GetBezierCurveCoefficients<real, 1>(alpha);
    EXPECT_NEAR_EQ(coeffs[0], 1_r - alpha);
    EXPECT_NEAR_EQ(coeffs[1], alpha);
  }
  // second order
  for (int i = 0; i < 100; i++) {
    real alpha = real(i) / 100_r;
    NdArray<real, 3> coeffs = GetBezierCurveCoefficients<real, 2>(alpha);
    EXPECT_NEAR_EQ(coeffs[0], Pow(1_r - alpha, 2));
    EXPECT_NEAR_EQ(coeffs[1], alpha * (1_r - alpha) * 2_r);
    EXPECT_NEAR_EQ(coeffs[2], Pow(alpha, 2));
  }
  // third order
  for (int i = 0; i < 100; i++) {
    real alpha = real(i) / 100_r;
    NdArray<real, 4> coeffs = GetBezierCurveCoefficients<real, 3>(alpha);
    EXPECT_NEAR_EQ(coeffs[0], Pow(1_r - alpha, 3));
    EXPECT_NEAR_EQ(coeffs[1], alpha * Pow(1_r - alpha, 2) * 3_r);
    EXPECT_NEAR_EQ(coeffs[2], Pow(alpha, 2) * (1_r - alpha) * 3_r);
    EXPECT_NEAR_EQ(coeffs[3], Pow(alpha, 3));
  }
}
