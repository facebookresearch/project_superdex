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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/vmatrix.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

using namespace mochi;

TEST(MathUtils, kFourthOrderEye) {
  // Compare values from pyMochi
  static_assert(NearEqual(Real3{1_r, 0_r, 0_r}, kFourthOrderEye[0][0][0]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[0][0][1]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[0][0][2]));
  static_assert(NearEqual(Real3{0_r, 1_r, 0_r}, kFourthOrderEye[0][1][0]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[0][1][1]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[0][1][2]));
  static_assert(NearEqual(Real3{0_r, 0_r, 1_r}, kFourthOrderEye[0][2][0]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[0][2][1]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[0][2][2]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[1][0][0]));
  static_assert(NearEqual(Real3{1_r, 0_r, 0_r}, kFourthOrderEye[1][0][1]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[1][0][2]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[1][1][0]));
  static_assert(NearEqual(Real3{0_r, 1_r, 0_r}, kFourthOrderEye[1][1][1]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[1][1][2]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[1][2][0]));
  static_assert(NearEqual(Real3{0_r, 0_r, 1_r}, kFourthOrderEye[1][2][1]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[1][2][2]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[2][0][0]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[2][0][1]));
  static_assert(NearEqual(Real3{1_r, 0_r, 0_r}, kFourthOrderEye[2][0][2]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[2][1][0]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[2][1][1]));
  static_assert(NearEqual(Real3{0_r, 1_r, 0_r}, kFourthOrderEye[2][1][2]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[2][2][0]));
  static_assert(NearEqual(Real3{0_r, 0_r, 0_r}, kFourthOrderEye[2][2][1]));
  static_assert(NearEqual(Real3{0_r, 0_r, 1_r}, kFourthOrderEye[2][2][2]));
}

TEST(MathUtils, kFourthOrderEyeSym) {
  // Compare values from pyMochi
  static_assert(NearEqual(Real3{1.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[0][0][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[0][0][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[0][0][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.5_r, 0.0_r}, kFourthOrderEyeSym[0][1][0]));
  static_assert(NearEqual(Real3{0.5_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[0][1][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[0][1][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.5_r}, kFourthOrderEyeSym[0][2][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[0][2][1]));
  static_assert(NearEqual(Real3{0.5_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[0][2][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.5_r, 0.0_r}, kFourthOrderEyeSym[1][0][0]));
  static_assert(NearEqual(Real3{0.5_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[1][0][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[1][0][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[1][1][0]));
  static_assert(NearEqual(Real3{0.0_r, 1.0_r, 0.0_r}, kFourthOrderEyeSym[1][1][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[1][1][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[1][2][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.5_r}, kFourthOrderEyeSym[1][2][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.5_r, 0.0_r}, kFourthOrderEyeSym[1][2][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.5_r}, kFourthOrderEyeSym[2][0][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[2][0][1]));
  static_assert(NearEqual(Real3{0.5_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[2][0][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[2][1][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.5_r}, kFourthOrderEyeSym[2][1][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.5_r, 0.0_r}, kFourthOrderEyeSym[2][1][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[2][2][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kFourthOrderEyeSym[2][2][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 1.0_r}, kFourthOrderEyeSym[2][2][2]));
}

TEST(MathUtils, kIouterI) {
  // Compare values from pyMochi
  static_assert(NearEqual(Real3{1.0_r, 0.0_r, 0.0_r}, kIouterI[0][0][0]));
  static_assert(NearEqual(Real3{0.0_r, 1.0_r, 0.0_r}, kIouterI[0][0][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 1.0_r}, kIouterI[0][0][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[0][1][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[0][1][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[0][1][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[0][2][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[0][2][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[0][2][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[1][0][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[1][0][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[1][0][2]));
  static_assert(NearEqual(Real3{1.0_r, 0.0_r, 0.0_r}, kIouterI[1][1][0]));
  static_assert(NearEqual(Real3{0.0_r, 1.0_r, 0.0_r}, kIouterI[1][1][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 1.0_r}, kIouterI[1][1][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[1][2][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[1][2][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[1][2][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[2][0][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[2][0][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[2][0][2]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[2][1][0]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[2][1][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 0.0_r}, kIouterI[2][1][2]));
  static_assert(NearEqual(Real3{1.0_r, 0.0_r, 0.0_r}, kIouterI[2][2][0]));
  static_assert(NearEqual(Real3{0.0_r, 1.0_r, 0.0_r}, kIouterI[2][2][1]));
  static_assert(NearEqual(Real3{0.0_r, 0.0_r, 1.0_r}, kIouterI[2][2][2]));
}

TEST(MathUtils, ArgMinMax) {
  constexpr Real1 kTestVector1{-1.0_r};
  constexpr Real2 kTestVector2{-1.0_r, 2.0_r};
  constexpr Real3 kTestVector3{-1.0_r, 2.0_r, -3.0_r};
  constexpr Real4 kTestVector4{-1.0_r, 2.0_r, -3.0_r, 4.0_r};

  static_assert(ArgMax(kTestVector1) == 0);
  static_assert(ArgMax(kTestVector2) == 1);
  static_assert(ArgMax(kTestVector3) == 1);
  static_assert(ArgMax(kTestVector4) == 3);

  static_assert(ArgMin(kTestVector1) == 0);
  static_assert(ArgMin(kTestVector2) == 0);
  static_assert(ArgMin(kTestVector3) == 2);
  static_assert(ArgMin(kTestVector4) == 2);
}

TEST(MathUtils, Lerp) {
  // Note: Lerp does not clamp the 't' parameter

  // Real2
  Real2 const a(10_r, 100_r);
  Real2 const b(20_r, 200_r);
  EXPECT_NEAR_EQ(Real2(0_r, 0_r), Lerp(a, b, -1_r));
  EXPECT_NEAR_EQ(Real2(5_r, 50_r), Lerp(a, b, -0.5_r));
  EXPECT_NEAR_EQ(Real2(10_r, 100_r), Lerp(a, b, 0.0_r));
  EXPECT_NEAR_EQ(Real2(15_r, 150_r), Lerp(a, b, 0.5_r));
  EXPECT_NEAR_EQ(Real2(20_r, 200_r), Lerp(a, b, 1.0_r));
  EXPECT_NEAR_EQ(Real2(25_r, 250_r), Lerp(a, b, 1.5_r));

  // Vec4r
  auto const va = Vec4r(10_r, 100_r, 1000_r, 10000_r);
  auto const vb = Vec4r(20_r, 200_r, 2000_r, 20000_r);
  EXPECT_NEAR_EQ(Vec4r(0_r, 0_r, 0_r, 0_r), Lerp(va, vb, -1_r));
  EXPECT_NEAR_EQ(Vec4r(5_r, 50_r, 500_r, 5000_r), Lerp(va, vb, -0.5_r));
  EXPECT_NEAR_EQ(Vec4r(10_r, 100_r, 1000_r, 10000_r), Lerp(va, vb, 0_r));
  EXPECT_NEAR_EQ(Vec4r(15_r, 150_r, 1500_r, 15000_r), Lerp(va, vb, 0.5_r));
  EXPECT_NEAR_EQ(Vec4r(20_r, 200_r, 2000_r, 20000_r), Lerp(va, vb, 1_r));
  EXPECT_NEAR_EQ(Vec4r(25_r, 250_r, 2500_r, 25000_r), Lerp(va, vb, 1.5_r));
}

TEST(MathUtils, NearEqual) {
  // Real2
  {
    constexpr Real2 a = {1_r, 2_r};
    constexpr Real2 b = {1_r, 2_r};
    constexpr Real2 c = {1_r, 2.0000001_r};
    constexpr Real2 d = {1_r, 2.000002_r};
    static_assert(NearEqual(a, a));
    static_assert(NearEqual(a, b, 0_r));
    static_assert(NearEqual(a, c, 1e-6_r));
    static_assert(!NearEqual(a, d, 1e-6_r));
  }

  // VMatrix3x3r with scalar tolerance
  {
    VMatrix3x3r const a = VEye<3>();
    EXPECT_TRUE(NearEqual(a, a, 0_r));
    EXPECT_NEAR_TOL(a, a, 0_r);

    VMatrix3x3r c = a;
    c[0] = Set<0>(c[0], Get<0>(c[0]) + 1e-8_r);
    EXPECT_TRUE(NearEqual(a, c, 1e-6_r));

    VMatrix3x3r d = a;
    d[0] = Set<0>(d[0], Get<0>(d[0]) + 1e-4_r);
    EXPECT_FALSE(NearEqual(a, d, 1e-6_r));
  }

  // VTensor3x3x3x3r with scalar tolerance
  {
    VTensor3x3x3x3r a = {};
    EXPECT_TRUE(NearEqual(a, a, 0_r));
    EXPECT_NEAR_TOL(a, a, 0_r);

    VTensor3x3x3x3r c = {};
    c[1][2][0] = Set<1>(c[1][2][0], 1e-8_r);
    EXPECT_TRUE(NearEqual(a, c, 1e-6_r));

    VTensor3x3x3x3r d = {};
    d[1][2][0] = Set<1>(d[1][2][0], 1e-4_r);
    EXPECT_FALSE(NearEqual(a, d, 1e-6_r));
  }
}

TEST(MathUtils, Dot) {
  // Dot(Real2, Real2)
  {
    constexpr Real2 a = {1_r, 2_r};
    constexpr Real2 b = {3_r, 4_r};
    static_assert(NearEqual(5_r, Dot(a, a)));
    static_assert(NearEqual(11_r, Dot(a, b)));
  }

  // Dot(Real3, Real3)
  {
    constexpr Real3 a = {1_r, 2_r, 3_r};
    constexpr Real3 b = {4_r, 5_r, 6_r};
    static_assert(NearEqual(14_r, Dot(a, a)));
    static_assert(NearEqual(32_r, Dot(a, b)));
  }
}

TEST(MathUtils, DotDynamic) {
  // Dot(span, NdArray)
  {
    constexpr NdArray<real, 5> a = {1_r, 2_r, 3_r, 4_r, 5_r};
    std::vector<real> const b = {6_r, 7_r, 8_r, 9_r, 10_r};
    EXPECT_NEAR_EQ(130_r, Dot(Span(b.data(), b.size()), a));
  }

  // Dot(span, span)
  {
    std::vector<real> const a = {1_r, 2_r, 3_r, 4_r, 5_r};
    std::vector<real> const b = {6_r, 7_r, 8_r, 9_r, 10_r};
    EXPECT_NEAR_EQ(130_r, Dot(Span(a.data(), a.size()), Span(b.data(), b.size())));
  }
}

TEST(MathUtils, Cross) {
  {
    constexpr Real3 a = {1_r, 0_r, 0_r};
    constexpr Real3 b = {0_r, 1_r, 0_r};
    static_assert(NearEqual(Real3{0_r, 0_r, 1_r}, Cross(a, b)));
  }

  {
    constexpr Real3 a = {1_r, 2_r, 3_r};
    constexpr Real3 b = {4_r, 5_r, 6_r};
    static_assert(NearEqual(Real3{-3_r, 6_r, -3_r}, Cross(a, b)));
  }
}

TEST(MathUtils, Norm) {
  // Real2
  {
    EXPECT_TRUE(NearEqual(0_r, Norm(Real2{0_r, 0_r})));
    EXPECT_TRUE(NearEqual(1_r, Norm(Real2{1_r, 0_r})));
    EXPECT_TRUE(NearEqual(1_r, Norm(Real2{0_r, 1_r})));
    EXPECT_TRUE(NearEqual(3.6055512_r, Norm(Real2{2_r, 3_r})));
  }

  // Real3
  {
    EXPECT_TRUE(NearEqual(0_r, Norm(Real3{0_r, 0_r, 0_r})));
    EXPECT_TRUE(NearEqual(1_r, Norm(Real3{1_r, 0_r, 0_r})));
    EXPECT_TRUE(NearEqual(1_r, Norm(Real3{0_r, 1_r, 0_r})));
    EXPECT_TRUE(NearEqual(5.3851648_r, Norm(Real3{2_r, 3_r, 4_r})));
  }
}

TEST(MathUtils, VNorm) {
  real const sqt2 = std::sqrt(2_r);
  real const sqt3 = std::sqrt(3_r);
  real const sqt4 = std::sqrt(4_r);

  // 2 component
  EXPECT_NEAR_EQ(Vec4r(0_r), VNorm<2>(Vec4r(0_r, 0_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r), VNorm<2>(Vec4r(1_r, 0_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r), VNorm<2>(Vec4r(0_r, 1_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(0_r), VNorm<2>(Vec4r(0_r, 0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(0_r), VNorm<2>(Vec4r(0_r, 0_r, 0_r, 1_r)));
  EXPECT_NEAR_EQ(Vec4r(sqt2), VNorm<2>(Vec4r(1_r, 1_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(sqt2), VNorm<2>(Vec4r(1_r, 1_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(sqt2), VNorm<2>(Vec4r(1_r, 1_r, 1_r, 1_r)));

  // 3 component
  EXPECT_NEAR_EQ(Vec4r(0_r), VNorm<3>(Vec4r(0_r, 0_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r), VNorm<3>(Vec4r(1_r, 0_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r), VNorm<3>(Vec4r(0_r, 1_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r), VNorm<3>(Vec4r(0_r, 0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(0_r), VNorm<3>(Vec4r(0_r, 0_r, 0_r, 1_r)));
  EXPECT_NEAR_EQ(Vec4r(sqt2), VNorm<3>(Vec4r(1_r, 1_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(sqt3), VNorm<3>(Vec4r(1_r, 1_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(sqt3), VNorm<3>(Vec4r(1_r, 1_r, 1_r, 1_r)));

  // 4 component
  EXPECT_NEAR_EQ(Vec4r(0_r), VNorm<4>(Vec4r(0_r, 0_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r), VNorm<4>(Vec4r(1_r, 0_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r), VNorm<4>(Vec4r(0_r, 1_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r), VNorm<4>(Vec4r(0_r, 0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r), VNorm<4>(Vec4r(0_r, 0_r, 0_r, 1_r)));
  EXPECT_NEAR_EQ(Vec4r(sqt2), VNorm<4>(Vec4r(1_r, 1_r, 0_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(sqt3), VNorm<4>(Vec4r(1_r, 1_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(sqt4), VNorm<4>(Vec4r(1_r, 1_r, 1_r, 1_r)));
}

TEST(MathUtils, Normalize) {
  // Note: If the vector is unit length, we return zeros (not divide by zero)
  // clang-format off

  // Real2
  {
    // Norm from vector.
    EXPECT_TRUE(NearEqual(Real2{0_r, 0_r}, Normalize(Real2{0_r, 0_r}))); // not div by zero
    EXPECT_TRUE(NearEqual(Real2{1_r, 0_r}, Normalize(Real2{1_r, 0_r})));
    EXPECT_TRUE(NearEqual(Real2{1_r, 0_r}, Normalize(Real2{10_r, 0_r})));
    EXPECT_TRUE(NearEqual(Real2{0_r, -1_r}, Normalize(Real2{0_r, -100_r})));
    EXPECT_TRUE(NearEqual(Real2{8.000000e-01_r, 6.000000e-01_r}, Normalize(Real2{4_r, 3_r})));
    EXPECT_TRUE(NearEqual(Real2{-2.425356e-01_r, -9.701425e-01_r}, Normalize(Real2{-1_r, -4_r})));
    EXPECT_TRUE(NearEqual(Real2{-8.741573e-01_r, -4.856429e-01_r}, Normalize(Real2{-9_r, -5_r})));
    EXPECT_TRUE(NearEqual(Real2{-5.299989e-01_r, 8.479983e-01_r}, Normalize(Real2{-5_r, 8_r})));
    EXPECT_TRUE(NearEqual(Real2{5.547002e-01_r, 8.320503e-01_r}, Normalize(Real2{4_r, 6_r})));

    // Explicitly providing the square norm.
    EXPECT_TRUE(NearEqual(Real2{0_r, 0_r}, Normalize(Real2{0_r, 0_r}, 0.0_r))); // not div by zero
    EXPECT_TRUE(NearEqual(Real2{8.000000e-01_r, 6.000000e-01_r}, Normalize(Real2{4_r, 3_r}, 25_r)));
    EXPECT_TRUE(NearEqual(Real2{-2.425356e-01_r, -9.701425e-01_r}, Normalize(Real2{-1_r, -4_r}, 17_r)));
    EXPECT_TRUE(NearEqual(Real2{-8.741573e-01_r, -4.856429e-01_r}, Normalize(Real2{-9_r, -5_r}, 106_r)));
    EXPECT_TRUE(NearEqual(Real2{-5.299989e-01_r, 8.479983e-01_r}, Normalize(Real2{-5_r, 8_r}, 89_r)));
    EXPECT_TRUE(NearEqual(Real2{5.547002e-01_r, 8.320503e-01_r}, Normalize(Real2{4_r, 6_r}, 52_r)));
  }

  // Real3
  {
    // Computing norm from vector.
    EXPECT_TRUE(NearEqual(Real3{0_r, 0_r, 0_r}, Normalize(Real3{0_r, 0_r, 0_r}))); // not div by zero
    EXPECT_TRUE(NearEqual(Real3{1_r, 0_r, 0_r}, Normalize(Real3{1_r, 0_r, 0_r})));
    EXPECT_TRUE(NearEqual(Real3{1_r, 0_r, 0_r}, Normalize(Real3{10_r, 0_r, 0_r})));
    EXPECT_TRUE(NearEqual(Real3{0_r, 0_r, -1_r}, Normalize(Real3{0_r, 0_r, -100_r})));
    EXPECT_TRUE(NearEqual(Real3{0.3713907_r, 0.5570860_r, 0.7427814_r}, Normalize(Real3{2_r, 3_r, 4_r})));
    EXPECT_TRUE(NearEqual(Real3{4.239992e-01_r, 3.179994e-01_r, -8.479983e-01_r}, Normalize(Real3{4_r, 3_r, -8_r})));
    EXPECT_TRUE(NearEqual(Real3{-1.961161e-01_r, -7.844645e-01_r, -5.883484e-01_r}, Normalize(Real3{-1_r, -4_r, -3_r})));
    EXPECT_TRUE(NearEqual(Real3{-8.700628e-01_r, -4.833682e-01_r, 9.667365e-02_r}, Normalize(Real3{-9_r, -5_r, 1_r})));
    EXPECT_TRUE(NearEqual(Real3{-5.299989e-01_r, 8.479983e-01_r, 0.000000e+00_r}, Normalize(Real3{-5_r, 8_r, 0_r})));
    EXPECT_TRUE(NearEqual(Real3{3.980149e-01_r, 5.970223e-01_r, -6.965260e-01_r}, Normalize(Real3{4_r, 6_r, -7_r})));

    // Explicitly providing squared norm.
    EXPECT_TRUE(NearEqual(Real3{0_r, 0_r, 0_r}, Normalize(Real3{0_r, 0_r, 0_r}, 0.0_r))); // not div by zero
    EXPECT_TRUE(NearEqual(Real3{4.239992e-01_r, 3.179994e-01_r, -8.479983e-01_r}, Normalize(Real3{4_r, 3_r, -8_r}, 89_r)));
    EXPECT_TRUE(NearEqual(Real3{-1.961161e-01_r, -7.844645e-01_r, -5.883484e-01_r}, Normalize(Real3{-1_r, -4_r, -3_r}, 26_r)));
    EXPECT_TRUE(NearEqual(Real3{-8.700628e-01_r, -4.833682e-01_r, 9.667365e-02_r}, Normalize(Real3{-9_r, -5_r, 1_r}, 107_r)));
    EXPECT_TRUE(NearEqual(Real3{-5.299989e-01_r, 8.479983e-01_r, 0.000000e+00_r}, Normalize(Real3{-5_r, 8_r, 0_r}, 89_r)));
    EXPECT_TRUE(NearEqual(Real3{3.980149e-01_r, 5.970223e-01_r, -6.965260e-01_r}, Normalize(Real3{4_r, 6_r, -7_r}, 101_r)));
  }

  // Real4
  {
    // Computing norm from vector.
    EXPECT_TRUE(NearEqual(Real4{0_r, 0_r, 0_r, 0_r}, Normalize(Real4{0_r, 0_r, 0_r, 0_r}))); // not div by zero
    EXPECT_TRUE(NearEqual(Real4{3.746343e-01_r, 2.809757e-01_r, -7.492686e-01_r, -4.682929e-01_r}, Normalize(Real4{4_r, 3_r, -8_r, -5_r})));
    EXPECT_TRUE(NearEqual(Real4{-1.154701e-01_r, -4.618802e-01_r, -3.464102e-01_r, -8.082904e-01_r}, Normalize(Real4{-1_r, -4_r, -3_r, -7_r})));
    EXPECT_TRUE(NearEqual(Real4{-7.833495e-01_r, -4.351941e-01_r, 8.703883e-02_r, 4.351941e-01_r}, Normalize(Real4{-9_r, -5_r, 1_r, 5_r})));
    EXPECT_TRUE(NearEqual(Real4{-5.184758e-01_r, 8.295614e-01_r, 0.000000e+00_r, -2.073903e-01_r}, Normalize(Real4{-5_r, 8_r, 0_r, -2_r})));
    EXPECT_TRUE(NearEqual(Real4{2.964997e-01_r, 4.447496e-01_r, -5.188745e-01_r, -6.671244e-01_r}, Normalize(Real4{4_r, 6_r, -7_r, -9_r})));

    // Explicitly providing squared norm.
    EXPECT_TRUE(NearEqual(Real4{0_r, 0_r, 0_r, 0_r}, Normalize(Real4{0_r, 0_r, 0_r, 0_r}, 0_r))); // not div by zero
    EXPECT_TRUE(NearEqual(Real4{3.746343e-01_r, 2.809757e-01_r, -7.492686e-01_r, -4.682929e-01_r}, Normalize(Real4{4_r, 3_r, -8_r, -5_r}, 114_r)));
    EXPECT_TRUE(NearEqual(Real4{-1.154701e-01_r, -4.618802e-01_r, -3.464102e-01_r, -8.082904e-01_r}, Normalize(Real4{-1_r, -4_r, -3_r, -7_r}, 75_r)));
    EXPECT_TRUE(NearEqual(Real4{-7.833495e-01_r, -4.351941e-01_r, 8.703883e-02_r, 4.351941e-01_r}, Normalize(Real4{-9_r, -5_r, 1_r, 5_r}, 132_r)));
    EXPECT_TRUE(NearEqual(Real4{-5.184758e-01_r, 8.295614e-01_r, 0.000000e+00_r, -2.073903e-01_r}, Normalize(Real4{-5_r, 8_r, 0_r, -2_r}, 93_r)));
    EXPECT_TRUE(NearEqual(Real4{2.964997e-01_r, 4.447496e-01_r, -5.188745e-01_r, -6.671244e-01_r}, Normalize(Real4{4_r, 6_r, -7_r, -9_r}, 182_r)));
  }
}

TEST(MathUtils, Mean) {
  // Real2
  {
    constexpr Real2 a = {10_r, 20_r};
    static_assert(NearEqual(15_r, Mean(a)));
  }

  // Real3
  {
    constexpr Real3 a = {10_r, 20_r, 30_r};
    static_assert(NearEqual(20_r, Mean(a)));
  }
}

TEST(MathUtils, Sum) {
  // Real2
  {
    constexpr Real2 a = {10_r, 20_r};
    static_assert(NearEqual(30_r, Sum(a)));
  }

  // Real3
  {
    constexpr Real3 a = {10_r, 20_r, 30_r};
    static_assert(NearEqual(60_r, Sum(a)));
  }
}

TEST(MathUtils, Prod) {
  // Real2
  {
    constexpr Real2 a = {10_r, 20_r};
    static_assert(NearEqual(200_r, Prod(a)));
  }

  // Real3
  {
    constexpr Real3 a = {10_r, 20_r, 30_r};
    static_assert(NearEqual(6000_r, Prod(a)));
  }
}

TEST(MathUtils, BasisVector) {
  static_assert(NearEqual(Real2{1_r, 0_r}, BasisVector<real, 2>(0)));
  static_assert(NearEqual(Real2{0_r, 1_r}, BasisVector<real, 2>(1)));
  static_assert(NearEqual(Real3{1_r, 0_r, 0_r}, BasisVector<real, 3>(0)));
  static_assert(NearEqual(Real3{0_r, 1_r, 0_r}, BasisVector<real, 3>(1)));
  static_assert(NearEqual(Real3{0_r, 0_r, 1_r}, BasisVector<real, 3>(2)));
  static_assert(NearEqual(Real4{1_r, 0_r, 0_r, 0_r}, BasisVector<real, 4>(0)));
  static_assert(NearEqual(Real4{0_r, 1_r, 0_r, 0_r}, BasisVector<real, 4>(1)));
  static_assert(NearEqual(Real4{0_r, 0_r, 1_r, 0_r}, BasisVector<real, 4>(2)));
  static_assert(NearEqual(Real4{0_r, 0_r, 0_r, 1_r}, BasisVector<real, 4>(3)));
}

TEST(MathUtils, OrthogonalVector3) {
  // clang-format off
  constexpr Real3 kTestVectors[] = {
    Real3{ 0.97627008_r,  4.30378733_r,  2.05526752_r},
    Real3{ 0.89766366_r, -1.52690401_r,  2.91788226_r},
    Real3{-1.24825577_r,  7.83546002_r,  9.27325521_r},
    Real3{-2.33116962_r,  5.83450076_r,  0.57789840_r},
    Real3{ 1.36089122_r,  8.51193277_r, -8.57927884_r},
    Real3{-8.25741401_r, -9.59563205_r,  6.65239691_r},
    Real3{ 5.56313502_r,  7.40024296_r,  9.57236684_r},
    Real3{ 5.98317128_r, -0.77041275_r,  5.61058353_r},
    Real3{-7.63451148_r,  2.79842043_r, -7.13293425_r},
    Real3{ 8.89337834_r,  0.43696644_r, -1.70676120_r}
  };
  // clang-format on

  static_assert(NearEqual(Dot(kTestVectors[0], OrthogonalVector(kTestVectors[0])), 0.0_r));
  static_assert(NearEqual(Dot(kTestVectors[1], OrthogonalVector(kTestVectors[1])), 0.0_r));
  static_assert(NearEqual(Dot(kTestVectors[2], OrthogonalVector(kTestVectors[2])), 0.0_r));
  static_assert(NearEqual(Dot(kTestVectors[3], OrthogonalVector(kTestVectors[3])), 0.0_r));
  static_assert(NearEqual(Dot(kTestVectors[4], OrthogonalVector(kTestVectors[4])), 0.0_r));
  static_assert(NearEqual(Dot(kTestVectors[5], OrthogonalVector(kTestVectors[5])), 0.0_r));
  static_assert(NearEqual(Dot(kTestVectors[6], OrthogonalVector(kTestVectors[6])), 0.0_r));
  static_assert(NearEqual(Dot(kTestVectors[7], OrthogonalVector(kTestVectors[7])), 0.0_r));
  static_assert(NearEqual(Dot(kTestVectors[8], OrthogonalVector(kTestVectors[8])), 0.0_r));
  static_assert(NearEqual(Dot(kTestVectors[9], OrthogonalVector(kTestVectors[9])), 0.0_r));
}

TEST(MathUtils, Skew) {
  // Initialize test vectors with irrelevant fourth component.
  Vec4r u = {-1.5_r, 2.2_r, 3.7_r, -1.4_r};
  Vec4r v = {4.2_r, -1.1_r, 2.3_r, 5.6_r};
  auto res1 = Cross3(u, v);
  auto res2 = DotMatVec3x3(Skew3(u), v);
  // Overwrite the fourth component to avoid irrelevant differences.
  res1 = Set<3>(res1, 0_r);
  res2 = Set<3>(res2, 0_r);
  EXPECT_NEAR_EQ(res1, res2);
}

TEST(MathUtils, InvSkew) {
  // Initialize test vector with zero fourth component.
  Vec4r u = {-1.5_r, 2.2_r, 3.7_r, 0_r};
  EXPECT_NEAR_EQ(u, InvSkew3(Skew3(u) + VDiagonalMatrix<3>(2_r)));
}

template <class ND>
static void TestNdArrayIsFinite() {
  // Test non-finite values in every position of the NdArray
  ND arr = {};
  EXPECT_TRUE(IsFinite(arr));
  for (auto& f : Flatten(arr)) {
    using T = std::remove_reference_t<decltype(f)>;
    f = std::numeric_limits<T>::infinity();
    EXPECT_FALSE(IsFinite(arr));
    f = -std::numeric_limits<T>::infinity();
    EXPECT_FALSE(IsFinite(arr));
    f = std::numeric_limits<T>::quiet_NaN();
    EXPECT_FALSE(IsFinite(arr));
    f = std::numeric_limits<T>::signaling_NaN();
    EXPECT_FALSE(IsFinite(arr));
    f = (T)123;
    EXPECT_TRUE(IsFinite(arr));
  }
}

template <class ND>
static void TestNdArrayIsFiniteSimd() {
  // Test non-finite values in every position of the NdArray of Simd
  ND arr = {};
  EXPECT_TRUE(IsFinite(arr));
  for (auto& v : Flatten(arr)) {
    using V = std::remove_reference_t<decltype(v)>;
    using T = typename V::Scalar;
    for (int i = 0; i < V::kSize; ++i) {
      v = Set(v, i, std::numeric_limits<T>::infinity());
      EXPECT_FALSE(IsFinite(arr));
      v = Set(v, i, -std::numeric_limits<T>::infinity());
      EXPECT_FALSE(IsFinite(arr));
      v = Set(v, i, std::numeric_limits<T>::quiet_NaN());
      EXPECT_FALSE(IsFinite(arr));
      v = Set(v, i, std::numeric_limits<T>::signaling_NaN());
      EXPECT_FALSE(IsFinite(arr));
      v = Set(v, i, (T)123);
      EXPECT_TRUE(IsFinite(arr));
    }
  }
}

TEST(MathUtils, IsFinite) {
  // NdArray of ints
  EXPECT_TRUE(IsFinite(NdArray<int, 3>{}));
  EXPECT_TRUE(IsFinite(NdArray<int, 4, 4>{}));
  EXPECT_TRUE(IsFinite(NdArray<int, 3, 4, 5>{}));
  EXPECT_TRUE(IsFinite(NdArray<Simd<int, 4>, 3>{}));
  EXPECT_TRUE(IsFinite(NdArray<Simd<int, 8>, 4, 4>{}));
  EXPECT_TRUE(IsFinite(NdArray<Simd<int, 8>, 3, 4, 5>{}));

  // NdArray of floats
  TestNdArrayIsFinite<NdArray<float, 3>>();
  TestNdArrayIsFinite<NdArray<float, 4, 4>>();
  TestNdArrayIsFinite<NdArray<float, 3, 4, 5>>();
  TestNdArrayIsFiniteSimd<NdArray<Simd<float, 4>, 3>>();
  TestNdArrayIsFiniteSimd<NdArray<Simd<float, 8>, 4, 4>>();
  TestNdArrayIsFiniteSimd<NdArray<Simd<float, 8>, 3, 4, 5>>();

  // NdArray of doubles
  TestNdArrayIsFinite<NdArray<double, 3>>();
  TestNdArrayIsFinite<NdArray<double, 4, 4>>();
  TestNdArrayIsFinite<NdArray<double, 3, 4, 5>>();
  TestNdArrayIsFiniteSimd<NdArray<Simd<double, 4>, 3>>();
  TestNdArrayIsFiniteSimd<NdArray<Simd<double, 8>, 4, 4>>();
  TestNdArrayIsFiniteSimd<NdArray<Simd<double, 8>, 3, 4, 5>>();
}
