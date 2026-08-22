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
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

#include <picojson/picojson.h>

using namespace mochi;

static_assert(alignof(Real3) == alignof(real), "Unexpected alignment");
static_assert(sizeof(Real3) == sizeof(real) * 3, "Unexpected padding");
static_assert(std::is_trivially_copyable_v<Real3>);

static_assert(alignof(Matrix3x3r) == alignof(real), "Unexpected alignment");
static_assert(sizeof(Matrix3x3r) == sizeof(real) * 9, "Unexpected padding");
static_assert(std::is_trivially_copyable_v<Matrix3x3r>);

TEST(BasicTypes, Int2) {
  // Compile-time checks
  constexpr Int2 k = {123, 456};
  static_assert(1 == k.num_dims);
  static_assert(2 == k.dims[0]);
  static_assert(2 == k.size());
  static_assert(2 == k.end() - k.begin());
  static_assert(123 == k[0]);
  static_assert(456 == k[1]);
  static_assert(123 == k.begin()[0]);
  static_assert(456 == k.begin()[1]);
  static_assert(123 == k.data()[0]);
  static_assert(456 == k.data()[1]);
  static_assert(Int2{123, 456} == k);
  static_assert(Int2{123, 457} != k);
  static_assert(sizeof(Int2) == sizeof(int) * 2);

  // Default
  Int2 x = {};
  EXPECT_EQ(0, x[0]);
  EXPECT_EQ(0, x[1]);

  // Non-default
  Int2 x2 = {111, 222};
  EXPECT_EQ(111, x2[0]);
  EXPECT_EQ(222, x2[1]);

  // Copy
  Int2 y = x2;
  EXPECT_EQ(111, y[0]);
  EXPECT_EQ(222, y[1]);

  // Assign
  y = {333, 444};
  EXPECT_EQ(333, y[0]);
  EXPECT_EQ(444, y[1]);

  // Ranged for
  int sum = 0;
  for (int a : y) {
    sum += a;
  }
  EXPECT_EQ(333 + 444, sum);

  // Flatten
  EXPECT_EQ(2, Flatten(y).size());
  EXPECT_EQ(333, Flatten(y)[0]);
  EXPECT_EQ(444, Flatten(y)[1]);

  // operator[]
  int& ref = y[0];
  ref = 555;
  EXPECT_EQ(555, y[0]);
  EXPECT_EQ(444, y[1]);

  // operator==
  EXPECT_EQ(false, (y == Int2{0, 444}));
  EXPECT_EQ(true, (y == Int2{555, 444}));

  // operator!=
  EXPECT_EQ(true, (y != Int2{0, 444}));
  EXPECT_EQ(false, (y != Int2{555, 444}));

  // Math operations (runtime)
  EXPECT_EQ((Int2{26, 38}), (Int2{22, 33} + Int2{4, 5}));
  EXPECT_EQ((Int2{18, 28}), (Int2{22, 33} - Int2{4, 5}));
  EXPECT_EQ((Int2{88, 165}), (Int2{22, 33} * Int2{4, 5}));
  EXPECT_EQ((Int2{5, 6}), (Int2{22, 33} / Int2{4, 5}));
  EXPECT_EQ((Int2{24, 35}), (Int2{22, 33} + 2));
  EXPECT_EQ((Int2{20, 31}), (Int2{22, 33} - 2));
  EXPECT_EQ((Int2{44, 66}), (Int2{22, 33} * 2));
  EXPECT_EQ((Int2{11, 16}), (Int2{22, 33} / 2));
  EXPECT_EQ((Int2{24, 35}), (2 + Int2{22, 33}));
  EXPECT_EQ((Int2{-20, -31}), (2 - Int2{22, 33}));
  EXPECT_EQ((Int2{44, 66}), (2 * Int2{22, 33}));
  EXPECT_EQ((Int2{0, 0}), (2 / Int2{22, 33}));
  EXPECT_EQ((Int2{-10, 20}), -(Int2{10, -20}));

  // Overloaded operators (runtime)
  EXPECT_EQ((Int2{26, 38}), (Int2{22, 33} + Int2{4, 5}));
  EXPECT_EQ((Int2{18, 28}), (Int2{22, 33} - Int2{4, 5}));
  EXPECT_EQ((Int2{88, 165}), (Int2{22, 33} * Int2{4, 5}));
  EXPECT_EQ((Int2{5, 6}), (Int2{22, 33} / Int2{4, 5}));
  EXPECT_EQ((Int2{24, 35}), (Int2{22, 33} + 2));
  EXPECT_EQ((Int2{20, 31}), (Int2{22, 33} - 2));
  EXPECT_EQ((Int2{44, 66}), (Int2{22, 33} * 2));
  EXPECT_EQ((Int2{11, 16}), (Int2{22, 33} / 2));
  EXPECT_EQ((Int2{24, 35}), (2 + Int2{22, 33}));
  EXPECT_EQ((Int2{-20, -31}), (2 - Int2{22, 33}));
  EXPECT_EQ((Int2{44, 66}), (2 * Int2{22, 33}));
  EXPECT_EQ((Int2{0, 0}), (2 / Int2{22, 33}));
  EXPECT_EQ((Int2{-10, 20}), (-Int2{10, -20}));

  // Math operators (compile-time)
  static_assert(Int2{26, 38} == Int2{22, 33} + Int2{4, 5});
  static_assert(Int2{18, 28} == Int2{22, 33} - Int2{4, 5});
  static_assert(Int2{88, 165} == Int2{22, 33} * Int2{4, 5});
  static_assert(Int2{5, 6} == Int2{22, 33} / Int2{4, 5});
  static_assert(Int2{24, 35} == Int2{22, 33} + 2);
  static_assert(Int2{20, 31} == Int2{22, 33} - 2);
  static_assert(Int2{44, 66} == Int2{22, 33} * 2);
  static_assert(Int2{11, 16} == Int2{22, 33} / 2);
  static_assert(Int2{-10, 20} == -Int2{10, -20});
}

TEST(BasicTypes, Real3) {
  // Compile-time checks
  constexpr Real3 k = {1.2_r, 3.4_r, 5.6_r};
  static_assert(1 == k.num_dims);
  static_assert(3 == k.dims[0]);
  static_assert(3 == k.size());
  static_assert(3 == k.end() - k.begin());
  static_assert(1.2_r == k[0]);
  static_assert(3.4_r == k[1]);
  static_assert(5.6_r == k[2]);
  static_assert(1.2_r == k.begin()[0]);
  static_assert(3.4_r == k.begin()[1]);
  static_assert(5.6_r == k.begin()[2]);
  static_assert(1.2_r == k.data()[0]);
  static_assert(3.4_r == k.data()[1]);
  static_assert(5.6_r == k.data()[2]);
  static_assert(Real3{1.2_r, 3.4_r, 5.6_r} == k);
  static_assert(Real3{1.2_r, 3.4_r, 5.7_r} != k);
  static_assert(sizeof(Real3) == sizeof(real) * 3);

  // Default
  Real3 x = {};
  EXPECT_EQ(0_r, x[0]);
  EXPECT_EQ(0_r, x[1]);
  EXPECT_EQ(0_r, x[2]);

  // Non-default
  Real3 x2 = {1_r, 2_r, 3_r};
  EXPECT_EQ(1_r, x2[0]);
  EXPECT_EQ(2_r, x2[1]);
  EXPECT_EQ(3_r, x2[2]);

  // Copy
  Real3 y = x2;
  EXPECT_EQ(1_r, y[0]);
  EXPECT_EQ(2_r, y[1]);
  EXPECT_EQ(3_r, y[2]);

  // Assign
  y = {4_r, 5_r, 6_r};
  EXPECT_EQ(4_r, y[0]);
  EXPECT_EQ(5_r, y[1]);
  EXPECT_EQ(6_r, y[2]);

  // Ranged for
  real sum = 0;
  for (real a : y) {
    sum += a;
  }
  EXPECT_TRUE(mochi::NearEqual(15_r, sum));

  // Flatten
  EXPECT_EQ(3, Flatten(y).size());
  EXPECT_EQ(4_r, Flatten(y)[0]);
  EXPECT_EQ(5_r, Flatten(y)[1]);
  EXPECT_EQ(6_r, Flatten(y)[2]);

  // operator[]
  real& ref = y[0];
  ref = 44_r;
  EXPECT_EQ(44_r, y[0]);
  EXPECT_EQ(5_r, y[1]);
  EXPECT_EQ(6_r, y[2]);

  // operator==
  EXPECT_EQ(false, (y == Real3{0_r, 5_r, 6_r}));
  EXPECT_EQ(true, (y == Real3{44_r, 5_r, 6_r}));

  // operator!=
  EXPECT_EQ(true, (y != Real3{0_r, 5_r, 6_r}));
  EXPECT_EQ(false, (y != Real3{44_r, 5_r, 6_r}));

  // Math operations (runtime)
  EXPECT_NEAR_EQ(Real3(15_r, 27_r, 39_r), (Real3(11_r, 22_r, 33_r) + Real3(4_r, 5_r, 6_r)));
  EXPECT_NEAR_EQ(Real3(7_r, 17_r, 27_r), (Real3(11_r, 22_r, 33_r) - Real3(4_r, 5_r, 6_r)));
  EXPECT_NEAR_EQ(Real3(44_r, 110_r, 198_r), (Real3(11_r, 22_r, 33_r) * Real3(4_r, 5_r, 6_r)));
  EXPECT_NEAR_EQ(Real3(2.75_r, 4.4_r, 5.5_r), (Real3(11_r, 22_r, 33_r) / Real3(4_r, 5_r, 6_r)));
  EXPECT_NEAR_EQ(Real3(13_r, 24_r, 35_r), (Real3(11_r, 22_r, 33_r) + 2_r));
  EXPECT_NEAR_EQ(Real3(9_r, 20_r, 31_r), (Real3(11_r, 22_r, 33_r) - 2_r));
  EXPECT_NEAR_EQ(Real3(22_r, 44_r, 66_r), (Real3(11_r, 22_r, 33_r) * 2_r));
  EXPECT_NEAR_EQ(Real3(5.5_r, 11_r, 16.5_r), (Real3(11_r, 22_r, 33_r) / 2_r));
  EXPECT_NEAR_EQ(Real3(10.1_r, -20.2_r, 30.3_r), -(Real3(-10.1_r, 20.2_r, -30.3_r)));

  // Overloaded operat(rs (runtime)
  EXPECT_NEAR_EQ(Real3(15_r, 27_r, 39_r), Real3(11_r, 22_r, 33_r) + Real3(4_r, 5_r, 6_r));
  EXPECT_NEAR_EQ(Real3(7_r, 17_r, 27_r), Real3(11_r, 22_r, 33_r) - Real3(4_r, 5_r, 6_r));
  EXPECT_NEAR_EQ(Real3(44_r, 110_r, 198_r), Real3(11_r, 22_r, 33_r) * Real3(4_r, 5_r, 6_r));
  EXPECT_NEAR_EQ(Real3(2.75_r, 4.4_r, 5.5_r), Real3(11_r, 22_r, 33_r) / Real3(4_r, 5_r, 6_r));
  EXPECT_NEAR_EQ(Real3(13_r, 24_r, 35_r), Real3(11_r, 22_r, 33_r) + 2_r);
  EXPECT_NEAR_EQ(Real3(9_r, 20_r, 31_r), Real3(11_r, 22_r, 33_r) - 2_r);
  EXPECT_NEAR_EQ(Real3(22_r, 44_r, 66_r), Real3(11_r, 22_r, 33_r) * 2_r);
  EXPECT_NEAR_EQ(Real3(5.5_r, 11_r, 16.5_r), Real3(11_r, 22_r, 33_r) / 2_r);
  EXPECT_NEAR_EQ(Real3(10.1_r, -20.2_r, 30.3_r), -Real3(-10.1_r, 20.2_r, -30.3_r));

  // Math operators (compile-time)
  static_assert(NearEqual(Real3{15_r, 27_r, 39_r}, Real3{11_r, 22_r, 33_r} + Real3{4_r, 5_r, 6_r}));
  static_assert(NearEqual(Real3{7_r, 17_r, 27_r}, Real3{11_r, 22_r, 33_r} - Real3{4_r, 5_r, 6_r}));
  static_assert(
      NearEqual(Real3{44_r, 110_r, 198_r}, Real3{11_r, 22_r, 33_r} * Real3{4_r, 5_r, 6_r}));
  static_assert(
      NearEqual(Real3{2.75_r, 4.4_r, 5.5_r}, Real3{11_r, 22_r, 33_r} / Real3{4_r, 5_r, 6_r}));
  static_assert(NearEqual(Real3{13_r, 24_r, 35_r}, Real3{11_r, 22_r, 33_r} + 2_r));
  static_assert(NearEqual(Real3{9_r, 20_r, 31_r}, Real3{11_r, 22_r, 33_r} - 2_r));
  static_assert(NearEqual(Real3{22_r, 44_r, 66_r}, Real3{11_r, 22_r, 33_r} * 2_r));
  static_assert(NearEqual(Real3{5.5_r, 11_r, 16.5_r}, Real3{11_r, 22_r, 33_r} / 2_r));
  static_assert(NearEqual(Real3{10.1_r, -20.2_r, 30.3_r}, -Real3{-10.1_r, 20.2_r, -30.3_r}));
}

TEST(BasicTypes, Matrix2x3r) {
  // Compile-time checks
  constexpr Matrix2x3r k = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  static_assert(2 == k.num_dims);
  static_assert(2 == k.dims[0]);
  static_assert(3 == k.dims[1]);
  static_assert(2 == k.size());
  static_assert(1_r == k[0][0]);
  static_assert(2_r == k[0][1]);
  static_assert(3_r == k[0][2]);
  static_assert(4_r == k[1][0]);
  static_assert(5_r == k[1][1]);
  static_assert(6_r == k[1][2]);
  static_assert(1_r == k.begin()[0][0]);
  static_assert(2_r == k.begin()[0][1]);
  static_assert(3_r == k.begin()[0][2]);
  static_assert(4_r == k.begin()[1][0]);
  static_assert(5_r == k.begin()[1][1]);
  static_assert(6_r == k.begin()[1][2]);
  static_assert(1_r == k.data()[0][0]);
  static_assert(2_r == k.data()[0][1]);
  static_assert(3_r == k.data()[0][2]);
  static_assert(4_r == k.data()[1][0]);
  static_assert(5_r == k.data()[1][1]);
  static_assert(6_r == k.data()[1][2]);
  static_assert(Matrix2x3r{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}} == k);
  static_assert(Matrix2x3r{Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 7_r}} != k);
  static_assert(sizeof(Matrix2x3r) == sizeof(real) * 2 * 3);

  // Default (zeros not identity)
  Matrix2x3r x = {};
  for (Real3 const& row : x) {
    for (real val : row) {
      EXPECT_EQ(0, val);
    }
  }

  // Non-default
  Matrix2x3r x2 = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}};
  EXPECT_EQ(1_r, x2[0][0]);
  EXPECT_EQ(2_r, x2[0][1]);
  EXPECT_EQ(3_r, x2[0][2]);
  EXPECT_EQ(4_r, x2[1][0]);
  EXPECT_EQ(5_r, x2[1][1]);
  EXPECT_EQ(6_r, x2[1][2]);
  EXPECT_EQ((Real3{1_r, 2_r, 3_r}), x2[0]);
  EXPECT_EQ((Real3{4_r, 5_r, 6_r}), x2[1]);

  // Copy
  Matrix2x3r y = x2;
  EXPECT_EQ((Real3{1_r, 2_r, 3_r}), y[0]);
  EXPECT_EQ((Real3{4_r, 5_r, 6_r}), y[1]);

  // Assign
  y = {Real3{2_r, 3_r, 4_r}, Real3{5_r, 6_r, 7_r}};
  EXPECT_EQ((Real3{2_r, 3_r, 4_r}), y[0]);
  EXPECT_EQ((Real3{5_r, 6_r, 7_r}), y[1]);

  // Ranged for
  Real3 sum = {};
  for (Real3 const& row : y) {
    sum += row;
  }
  EXPECT_TRUE(mochi::NearEqual(Real3{7_r, 9_r, 11_r}, sum));

  // Flatten
  EXPECT_EQ(6, Flatten(y).size());
  EXPECT_EQ(2_r, Flatten(y)[0]);
  EXPECT_EQ(3_r, Flatten(y)[1]);
  EXPECT_EQ(4_r, Flatten(y)[2]);
  EXPECT_EQ(5_r, Flatten(y)[3]);
  EXPECT_EQ(6_r, Flatten(y)[4]);
  EXPECT_EQ(7_r, Flatten(y)[5]);

  // operator[]
  Real3& ref = y[0];
  ref = Real3{11_r, 22_r, 33_r};
  EXPECT_EQ((Real3{11_r, 22_r, 33_r}), y[0]);
  EXPECT_EQ((Real3{5_r, 6_r, 7_r}), y[1]);

  // operator==
  EXPECT_EQ(true, (y == Matrix2x3r{Real3{11_r, 22_r, 33_r}, Real3{5_r, 6_r, 7_r}}));
  EXPECT_EQ(false, (y == Matrix2x3r{Real3{11_r, 22_r, 33_r}, Real3{5_r, 6_r, 888_r}}));

  // operator!=
  EXPECT_EQ(false, (y != Matrix2x3r{Real3{11_r, 22_r, 33_r}, Real3{5_r, 6_r, 7_r}}));
  EXPECT_EQ(true, (y != Matrix2x3r{Real3{11_r, 22_r, 33_r}, Real3{5_r, 6_r, 888_r}}));

  // clang-format off

  // Math operations (runtime)
  constexpr auto lhs = Matrix2x3r(Real3(1_r, 2_r, 3_r), Real3(4_r, 5_r, 6_r));
  constexpr auto rhs = Matrix2x3r(Real3(10_r, 20_r, 30_r), Real3(40_r, 50_r, 60_r));
  constexpr real scalar = 100.0_r;
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(11_r, 22_r, 33_r), Real3(44_r, 55_r, 66_r)), lhs + rhs);
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(-9_r, -18_r, -27_r), Real3(-36_r, -45_r, -54_r)), lhs - rhs);
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(10_r, 40_r, 90_r), Real3(160_r, 250_r, 360_r)), (lhs * rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(0.1_r, 0.1_r, 0.1_r), Real3(0.1_r, 0.1_r, 0.1_r)), (lhs / rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(101_r, 102_r, 103_r), Real3(104_r, 105_r, 106_r)), lhs + scalar);
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(-99_r, -98_r, -97_r), Real3(-96_r, -95_r, -94_r)), lhs - scalar);
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(100_r, 200_r, 300_r), Real3(400_r, 500_r, 600_r)), (lhs * scalar));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(0.01_r, 0.02_r, 0.03_r), Real3(0.04_r, 0.05_r, 0.06_r)), (lhs / scalar));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(110_r, 120_r, 130_r), Real3(140_r, 150_r, 160_r)), scalar + rhs);
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(90_r, 80_r, 70_r), Real3(60_r, 50_r, 40_r)), scalar - rhs);
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(1000_r, 2000_r, 3000_r), Real3(4000_r, 5000_r, 6000_r)), (scalar * rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(10_r, 5_r, (100_r / 30_r)), Real3(2.5_r, 2_r, (100_r / 60_r))), (scalar / rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(-1_r, -2_r, -3_r), Real3(-4_r, -5_r, -6_r)), -lhs);

  // Overloaded operators
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(11_r, 22_r, 33_r), Real3(44_r, 55_r, 66_r)), (lhs + rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(-9_r, -18_r, -27_r), Real3(-36_r, -45_r, -54_r)), (lhs - rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(10_r, 40_r, 90_r), Real3(160_r, 250_r, 360_r)), (lhs * rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(0.1_r, 0.1_r, 0.1_r), Real3(0.1_r, 0.1_r, 0.1_r)), (lhs / rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(101_r, 102_r, 103_r), Real3(104_r, 105_r, 106_r)), (lhs + scalar));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(-99_r, -98_r, -97_r), Real3(-96_r, -95_r, -94_r)), (lhs - scalar));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(100_r, 200_r, 300_r), Real3(400_r, 500_r, 600_r)), (lhs * scalar));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(0.01_r, 0.02_r, 0.03_r), Real3(0.04_r, 0.05_r, 0.06_r)), (lhs / scalar));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(110_r, 120_r, 130_r), Real3(140_r, 150_r, 160_r)), (scalar + rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(90_r, 80_r, 70_r), Real3(60_r, 50_r, 40_r)), (scalar - rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(1000_r, 2000_r, 3000_r), Real3(4000_r, 5000_r, 6000_r)), (scalar * rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(10_r, 5_r, (100_r / 30_r)), Real3(2.5_r, 2_r, (100_r / 60_r))), (scalar / rhs));
  EXPECT_NEAR_EQ(Matrix2x3r(Real3(-1_r, -2_r, -3_r), Real3(-4_r, -5_r, -6_r)), -lhs);

  // Math operators (compile-time)
  static_assert(NearEqual(Matrix2x3r{Real3{11_r, 22_r, 33_r}, Real3{44_r, 55_r, 66_r} }, (lhs + rhs)));
  static_assert(NearEqual(Matrix2x3r{Real3{-9_r, -18_r, -27_r}, Real3{-36_r, -45_r, -54_r} }, (lhs - rhs)));
  static_assert(NearEqual(Matrix2x3r{Real3{10_r, 40_r, 90_r}, Real3{160_r, 250_r, 360_r} }, (lhs * rhs)));
  static_assert(NearEqual(Matrix2x3r{Real3{0.1_r, 0.1_r, 0.1_r}, Real3{0.1_r, 0.1_r, 0.1_r} }, (lhs / rhs)));
  static_assert(NearEqual(Matrix2x3r{Real3{101_r, 102_r, 103_r}, Real3{104_r, 105_r, 106_r} }, (lhs + scalar)));
  static_assert(NearEqual(Matrix2x3r{Real3{-99_r, -98_r, -97_r}, Real3{-96_r, -95_r, -94_r} }, (lhs - scalar)));
  static_assert(NearEqual(Matrix2x3r{Real3{100_r, 200_r, 300_r}, Real3{400_r, 500_r, 600_r} }, (lhs * scalar)));
  static_assert(NearEqual(Matrix2x3r{Real3{0.01_r, 0.02_r, 0.03_r}, Real3{0.04_r, 0.05_r, 0.06_r} }, (lhs / scalar)));
  static_assert(NearEqual(Matrix2x3r{Real3{110_r, 120_r, 130_r}, Real3{140_r, 150_r, 160_r} }, (scalar + rhs)));
  static_assert(NearEqual(Matrix2x3r{Real3{90_r, 80_r, 70_r}, Real3{60_r, 50_r, 40_r} }, (scalar - rhs)));
  static_assert(NearEqual(Matrix2x3r{Real3{1000_r, 2000_r, 3000_r}, Real3{4000_r, 5000_r, 6000_r} }, (scalar * rhs)));
  static_assert(NearEqual(Matrix2x3r{Real3{10_r, 5_r, (100_r / 30_r)}, Real3{2.5_r, 2_r, (100_r / 60_r)} }, (scalar / rhs)));
  static_assert(NearEqual(Matrix2x3r{ Real3{-1_r, -2_r, -3_r}, Real3{-4_r, -5_r, -6_r} }, -lhs));

  // clang-format on
}

TEST(NdArray, Reflection) {
  // Int2
  {
    using ArrayType = NdArray<int, 2>;
    ArrayType nd{123, 456};
    auto const& typeInfo = SReflect::GetTypeInfo<ArrayType>();

    // Serialization
    EXPECT_STREQ("[123,456]", SReflect::ToJsonString(nd, false).c_str());
    EXPECT_EQ(nd, SReflect::FromJsonString<ArrayType>("[123,456]"));

    // Type Introspection
    EXPECT_STREQ("NdArray<int32,2>", typeInfo._name);
    EXPECT_STREQ("mochi::NdArray<int32,2>", typeInfo._nameWithNamespace);
    EXPECT_EQ(SReflect::CoreType::CT_array, typeInfo._coreType);
    EXPECT_EQ(sizeof(ArrayType), typeInfo._sizeInBytes);
    EXPECT_EQ(alignof(ArrayType), typeInfo._alignment);
    EXPECT_EQ(&SReflect::GetTypeInfo<int>(), typeInfo._innerTypeInfo);

    // Factor Creation (does not require compile-time access to ArrayType)
    void* newObj = typeInfo.New();
    picojson::value json = picojson::object();
    typeInfo.Serialize(newObj, json);
    EXPECT_STREQ("[0,0]", json.serialize(false).c_str());
    typeInfo.Delete(newObj);
  }

  // Real3
  {
    using ArrayType = NdArray<real, 3>;
    ArrayType nd{-1_r, 0.5_r, 1_r};
    auto const& typeInfo = SReflect::GetTypeInfo<ArrayType>();

    // Serialization
    EXPECT_STREQ("[-1,0.5,1]", SReflect::ToJsonString(nd, false).c_str());
    EXPECT_EQ(nd, SReflect::FromJsonString<ArrayType>("[-1,0.5,1]"));

    // Type Introspection
    if constexpr (MOCHI_USE_DOUBLE_PRECISION) {
      EXPECT_STREQ("NdArray<double,3>", typeInfo._name);
      EXPECT_STREQ("mochi::NdArray<double,3>", typeInfo._nameWithNamespace);
    } else {
      EXPECT_STREQ("NdArray<float,3>", typeInfo._name);
      EXPECT_STREQ("mochi::NdArray<float,3>", typeInfo._nameWithNamespace);
    }
    EXPECT_EQ(SReflect::CoreType::CT_array, typeInfo._coreType);
    EXPECT_EQ(sizeof(ArrayType), typeInfo._sizeInBytes);
    EXPECT_EQ(alignof(ArrayType), typeInfo._alignment);
    EXPECT_EQ(&SReflect::GetTypeInfo<real>(), typeInfo._innerTypeInfo);

    // Factor Creation (does not require compile-time access to ArrayType)
    void* newObj = typeInfo.New();
    picojson::value json = picojson::object();
    typeInfo.Serialize(newObj, json);
    EXPECT_STREQ("[0,0,0]", json.serialize(false).c_str());
    typeInfo.Delete(newObj);
  }

  // NdArray<int, 3, 3>
  {
    using ArrayType = NdArray<int, 3, 3>;
    ArrayType nd{Int3{1, 2, 3}, Int3{4, 5, 6}, Int3{7, 8, 9}};
    auto const& typeInfo = SReflect::GetTypeInfo<ArrayType>();

    // Serialization
    EXPECT_STREQ("[[1,2,3],[4,5,6],[7,8,9]]", SReflect::ToJsonString(nd, false).c_str());
    EXPECT_EQ(nd, SReflect::FromJsonString<ArrayType>("[[1,2,3],[4,5,6],[7,8,9]]"));

    // Type Introspection
    EXPECT_STREQ("NdArray<int32,3,3>", typeInfo._name);
    EXPECT_STREQ("mochi::NdArray<int32,3,3>", typeInfo._nameWithNamespace);
    EXPECT_EQ(SReflect::CoreType::CT_array, typeInfo._coreType);
    EXPECT_EQ(sizeof(ArrayType), typeInfo._sizeInBytes);
    EXPECT_EQ(alignof(ArrayType), typeInfo._alignment);
    EXPECT_EQ(&SReflect::GetTypeInfo<Int3>(), typeInfo._innerTypeInfo);

    // Factor Creation (does not require compile-time access to ArrayType)
    void* newObj = typeInfo.New();
    picojson::value json = picojson::object();
    typeInfo.Serialize(newObj, json);
    EXPECT_STREQ("[[0,0,0],[0,0,0],[0,0,0]]", json.serialize(false).c_str());
    typeInfo.Delete(newObj);
  }

  // NdArray<int, 2, 3, 4>
  {
    using ArrayType = NdArray<int, 2, 2, 2>;
    ArrayType nd{
        NdArray<int, 2, 2>{Int2{1, 2}, Int2{3, 4}}, NdArray<int, 2, 2>{Int2{5, 6}, Int2{7, 8}}};
    auto const& typeInfo = SReflect::GetTypeInfo<ArrayType>();

    // Serialization
    EXPECT_STREQ("[[[1,2],[3,4]],[[5,6],[7,8]]]", SReflect::ToJsonString(nd, false).c_str());
    EXPECT_EQ(nd, SReflect::FromJsonString<ArrayType>("[[[1,2],[3,4]],[[5,6],[7,8]]]"));

    // Type Introspection
    EXPECT_STREQ("NdArray<int32,2,2,2>", typeInfo._name);
    EXPECT_STREQ("mochi::NdArray<int32,2,2,2>", typeInfo._nameWithNamespace);
    EXPECT_EQ(SReflect::CoreType::CT_array, typeInfo._coreType);
    EXPECT_EQ(sizeof(ArrayType), typeInfo._sizeInBytes);
    EXPECT_EQ(alignof(ArrayType), typeInfo._alignment);
    EXPECT_EQ((&SReflect::GetTypeInfo<NdArray<int, 2, 2>>()), typeInfo._innerTypeInfo);

    // Factor Creation (does not require compile-time access to ArrayType)
    void* newObj = typeInfo.New();
    picojson::value json = picojson::object();
    typeInfo.Serialize(newObj, json);
    EXPECT_STREQ("[[[0,0],[0,0]],[[0,0],[0,0]]]", json.serialize(false).c_str());
    typeInfo.Delete(newObj);
  }

  // NdArray<Vec4d, 2>
  {
    using ArrayType = NdArray<Simd<double, 2>, 2>;
    ArrayType nd{Vec2d{1.0, 2.0}, Vec2d{3.0, 4.0}};
    auto const& typeInfo = SReflect::GetTypeInfo<ArrayType>();

    // Serialization
    EXPECT_STREQ("[[1,2],[3,4]]", SReflect::ToJsonString(nd, false).c_str());
    EXPECT_EQ(nd, SReflect::FromJsonString<ArrayType>("[[1,2],[3,4]]"));

    // Type Introspection
    EXPECT_STREQ("NdArray<Simd<double,2>,2>", typeInfo._name);
    EXPECT_STREQ("mochi::NdArray<mochi::Simd<double,2>,2>", typeInfo._nameWithNamespace);
    EXPECT_EQ(SReflect::CoreType::CT_array, typeInfo._coreType);
    EXPECT_EQ(sizeof(ArrayType), typeInfo._sizeInBytes);
    EXPECT_EQ(alignof(ArrayType), typeInfo._alignment);
    EXPECT_EQ((&SReflect::GetTypeInfo<Vec2d>()), typeInfo._innerTypeInfo);

    // Factor Creation (does not require compile-time access to ArrayType)
    void* newObj = typeInfo.New();
    picojson::value json = picojson::object();
    typeInfo.Serialize(newObj, json);
    EXPECT_STREQ("[[0,0],[0,0]]", json.serialize(false).c_str());
    typeInfo.Delete(newObj);
  }
}

TEST(NdArray, ScalarType) {
  static_assert(std::is_same_v<ScalarType<NdArray<float, 3>>, float>);
  static_assert(std::is_same_v<ScalarType<NdArray<double, 3, 3>>, double>);
  static_assert(std::is_same_v<ScalarType<NdArray<Span<int const>, 4>>, int>);
  static_assert(std::is_same_v<ScalarType<NdArray<Simd<int64_t>, 3>>, int64_t>);
  static_assert(std::is_same_v<ScalarType<NdArray<Simd<float>, 3, 3>>, float>);
  static_assert(
      std::is_same_v<ScalarType<NdArray<Span<NdArray<Simd<double>, 2> const>, 4, 4, 4>>, double>);
}
