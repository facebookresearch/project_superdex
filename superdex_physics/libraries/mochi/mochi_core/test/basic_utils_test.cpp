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

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

using namespace mochi;

template <typename T>
static void TestNextPowerOfTwo() {
  constexpr T kMaxPowerOfTwo = T(1) << (std::numeric_limits<T>::digits - 1);

  static_assert(NextPowerOfTwo(T(0)) == T(1));
  static_assert(NextPowerOfTwo(T(1)) == T(1));
  static_assert(NextPowerOfTwo(T(2)) == T(2));
  static_assert(NextPowerOfTwo(T(3)) == T(4));
  static_assert(NextPowerOfTwo(T(4)) == T(4));
  static_assert(NextPowerOfTwo(T(5)) == T(8));
  static_assert(NextPowerOfTwo(T(kMaxPowerOfTwo - T(1))) == kMaxPowerOfTwo);
  static_assert(NextPowerOfTwo(kMaxPowerOfTwo) == kMaxPowerOfTwo);
  if constexpr (std::is_signed_v<T>) {
    static_assert(NextPowerOfTwo(T(-1)) == T(1));
    static_assert(NextPowerOfTwo(std::numeric_limits<T>::min()) == T(1));
  }

  EXPECT_EQ(T(1), NextPowerOfTwo(T(0)));
  EXPECT_EQ(T(1), NextPowerOfTwo(T(1)));
  EXPECT_EQ(T(2), NextPowerOfTwo(T(2)));
  EXPECT_EQ(T(4), NextPowerOfTwo(T(3)));
  EXPECT_EQ(T(4), NextPowerOfTwo(T(4)));
  EXPECT_EQ(T(8), NextPowerOfTwo(T(5)));
  EXPECT_EQ(kMaxPowerOfTwo, NextPowerOfTwo(T(kMaxPowerOfTwo - T(1))));
  EXPECT_EQ(kMaxPowerOfTwo, NextPowerOfTwo(kMaxPowerOfTwo));
  if constexpr (std::is_signed_v<T>) {
    EXPECT_EQ(T(1), NextPowerOfTwo(T(-1)));
    EXPECT_EQ(T(1), NextPowerOfTwo(std::numeric_limits<T>::min()));
  }
}

TEST(BasicUtils, Abs) {
  EXPECT_EQ(0, Abs(0));
  EXPECT_EQ(123, Abs(123));
  EXPECT_EQ(456, Abs(-456));
  EXPECT_EQ(0_r, Abs(0_r));
  EXPECT_EQ(0_r, Abs(-0_r));
  EXPECT_EQ(123_r, Abs(123_r));
  EXPECT_EQ(456_r, Abs(-456_r));
}

TEST(BasicUtils, AllTrue) {
  EXPECT_FALSE(AllTrue(0));
  EXPECT_FALSE(AllTrue(-0_r)); // -0 is equal to 0 and evaluates to false (as usual)
  EXPECT_TRUE(AllTrue(1));
  EXPECT_TRUE(AllTrue(-2_r));
  EXPECT_TRUE(AllTrue(3UL));
}

TEST(BasicUtils, AnyTrue) {
  // Same as AllTrue when argument is a single scalar
  EXPECT_FALSE(AnyTrue(0));
  EXPECT_FALSE(AnyTrue(-0_r)); // -0 is equal to 0 and evaluates to false (as usual)
  EXPECT_TRUE(AnyTrue(1));
  EXPECT_TRUE(AnyTrue(-2_r));
  EXPECT_TRUE(AnyTrue(3UL));
}

TEST(BasicUtils, StaticCast) {
  EXPECT_EQ(123, StaticCast<int>(123.0f));
  EXPECT_EQ(123.0f, StaticCast<float>(123));
  EXPECT_EQ(123.0f, StaticCast<float>(123.0));
}

TEST(BasicUtils, ReinterpretCast) {
  EXPECT_EQ(123, ReinterpretCast<long>(123UL));
  EXPECT_EQ(123UL, ReinterpretCast<unsigned long>(123L));
  EXPECT_EQ(0, ReinterpretCast<int32_t>(0.0f));
  EXPECT_EQ(0.0f, ReinterpretCast<float>(0x00000000));
  EXPECT_EQ(-0.0f, ReinterpretCast<float>(0x80000000));
  EXPECT_EQ(1.0f, ReinterpretCast<float>(0x3F800000));
  EXPECT_EQ(-1.0f, ReinterpretCast<float>(0xBF800000));
  EXPECT_EQ(std::numeric_limits<float>::infinity(), ReinterpretCast<float>(0x7F800000));
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), ReinterpretCast<float>(0xFF800000));
  EXPECT_EQ(nullptr, ReinterpretCast<void*>(static_cast<float*>(nullptr)));
}

TEST(BasicUtils, MulAdd) {
  constexpr real kValues[] = {1.1_r, -2.2_r, 3.3_r};
  for (real a : kValues) {
    for (real b : kValues) {
      for (real c : kValues) {
        EXPECT_NEAR_EQ(a * b + c, MulAdd(a, b, c));
        EXPECT_NEAR_EQ(-a * b + c, NegMulAdd(a, b, c));
        EXPECT_NEAR_EQ(a * b - c, MulSub(a, b, c));
        EXPECT_NEAR_EQ(-a * b - c, NegMulSub(a, b, c));
      }
    }
  }
}

TEST(BasicUtils, IsFinite) {
  EXPECT_TRUE(IsFinite(123));
  EXPECT_TRUE(IsFinite(123_r));
  EXPECT_TRUE(IsFinite(std::numeric_limits<real>::max()));
  EXPECT_TRUE(IsFinite(-std::numeric_limits<real>::max()));
  EXPECT_FALSE(IsFinite(std::numeric_limits<real>::infinity()));
  EXPECT_FALSE(IsFinite(-std::numeric_limits<real>::infinity()));
  EXPECT_FALSE(IsFinite(NAN));
}

TEST(BasicUtils, Min) {
  constexpr int a = 1;
  constexpr int b = 2;
  constexpr int c = 2;
  EXPECT_EQ(1, Min(a, b));
  EXPECT_EQ(1, Min(b, a));

  // Min & Max return references.
  // When equal, they return a reference to the first param.
  EXPECT_EQ(&b, &Min(b, c));
  EXPECT_EQ(&c, &Min(c, b));
  EXPECT_EQ(&b, &Min((int const&)b, (int const&)c));
  EXPECT_EQ(&c, &Min((int const&)c, (int const&)b));

  // Variadac version works for more than two params.
  EXPECT_EQ(1, Min(1, 2, 3));
  EXPECT_EQ(-2, Min(1, -2, 3));
  EXPECT_EQ(-3, Min(1, 2, -3));
  EXPECT_EQ(1, Min(1, 2, 3, 4));
  EXPECT_EQ(-2, Min(1, -2, 3, 4));
  EXPECT_EQ(-3, Min(1, 2, -3, 4));
  EXPECT_EQ(-4, Min(1, 2, 3, -4));
}

TEST(BasicUtils, Max) {
  constexpr int a = 1;
  constexpr int b = 2;
  constexpr int c = 2;
  EXPECT_EQ(2, Max(a, b));
  EXPECT_EQ(2, Max(b, a));

  // Min & Max return references.
  // When equal, they return a reference to the first param.
  EXPECT_EQ(&b, &Max(b, c));
  EXPECT_EQ(&c, &Max(c, b));
  EXPECT_EQ(&b, &Max((int const&)b, (int const&)c));
  EXPECT_EQ(&c, &Max((int const&)c, (int const&)b));

  // Variadac version works for more than two params.
  EXPECT_EQ(3, Max(1, 2, 3));
  EXPECT_EQ(2, Max(1, 2, -3));
  EXPECT_EQ(1, Max(1, -2, -3));
  EXPECT_EQ(4, Max(1, 2, 3, 4));
  EXPECT_EQ(3, Max(1, 2, 3, -4));
  EXPECT_EQ(2, Max(1, 2, -3, -4));
  EXPECT_EQ(1, Max(1, -2, -3, -4));
}

TEST(BasicUtils, Clamp) {
  EXPECT_EQ(0, Clamp(0, 0, 0));
  EXPECT_EQ(5, Clamp(0, 5, 7));
  EXPECT_EQ(5, Clamp(0, 5, 7));
  EXPECT_EQ(6, Clamp(0, 6, 7));
  EXPECT_EQ(7, Clamp(0, 7, 7));
  EXPECT_EQ(7, Clamp(0, 8, 7));
  EXPECT_EQ(0_r, Clamp(0_r, 0_r, 0_r));
  EXPECT_EQ(5_r, Clamp(0_r, 5_r, 7_r));
  EXPECT_EQ(5_r, Clamp(0_r, 5_r, 7_r));
  EXPECT_EQ(6_r, Clamp(0_r, 6_r, 7_r));
  EXPECT_EQ(7_r, Clamp(0_r, 7_r, 7_r));
  EXPECT_EQ(7_r, Clamp(0_r, 8_r, 7_r));
}

TEST(BasicUtils, Equal) {
  EXPECT_TRUE(Equal(1, 1));
  EXPECT_TRUE(Equal(1_r, 1_r));
  EXPECT_FALSE(Equal(1, 2));
  EXPECT_FALSE(Equal(1_r, 1.1_r));
}

TEST(BasicUtils, NotEqual) {
  EXPECT_FALSE(NotEqual(1, 1));
  EXPECT_FALSE(NotEqual(1_r, 1_r));
  EXPECT_TRUE(NotEqual(1, 2));
  EXPECT_TRUE(NotEqual(1_r, 1.1_r));
}

TEST(BasicUtils, NearEqual) {
  EXPECT_TRUE(NearEqual(2_r, 2_r));
  EXPECT_TRUE(NearEqual(2_r, 2_r, 0_r));
  EXPECT_FALSE(NearEqual(2_r, 3_r));
  EXPECT_FALSE(NearEqual(2_r, 3_r, 0_r));
  EXPECT_TRUE(NearEqual(2_r, 2.0000001_r, 1e-6_r));
  EXPECT_FALSE(NearEqual(2_r, 2.00001_r, 1e-6_r));
  EXPECT_TRUE(NearEqual(200_r, 200.0001_r, 1e-3_r));
  EXPECT_FALSE(NearEqual(200_r, 200.01_r, 1e-3_r));
}

TEST(BasicUtils, NearEqualRel) {
  EXPECT_TRUE(NearEqualRel(2_r, 2_r));
  EXPECT_TRUE(NearEqualRel(2_r, 2_r, 0_r));
  EXPECT_FALSE(NearEqualRel(2_r, 3_r));
  EXPECT_FALSE(NearEqualRel(2_r, 3_r, 0_r));
  EXPECT_TRUE(NearEqualRel(2_r, 2.0000001_r, 1e-6_r));
  EXPECT_FALSE(NearEqualRel(2_r, 2.00001_r, 1e-6_r));
  EXPECT_TRUE(NearEqualRel(200_r, 200.0001_r, 1e-3_r));
  EXPECT_TRUE(NearEqualRel(200_r, 200.01_r, 1e-3_r)); // Within scaled tolerance
}

TEST(BasicUtils, NearZero) {
  EXPECT_TRUE(NearZero(0_r));
  EXPECT_TRUE(NearZero(0_r, 0_r));
  EXPECT_TRUE(NearZero(kDefaultNearEqualEpsilon<real>));
  EXPECT_TRUE(NearZero(-kDefaultNearEqualEpsilon<real>));
  EXPECT_TRUE(NearZero(1e-6_r, 1e-6_r));
  EXPECT_TRUE(NearZero(-1e-6_r, 1e-6_r));
  EXPECT_FALSE(NearZero(1e-5_r));
  EXPECT_FALSE(NearZero(-1e-5_r));
  EXPECT_TRUE(NearZero(1e-5_r, 1e-5_r));
  EXPECT_TRUE(NearZero(-1e-5_r, 1e-5_r));
}

TEST(BasicUtils, isize) {
  // c-style array
  {
    static constexpr int x[] = {1, 2, 3, 4, 5};
    static_assert(5 == isize(x));
    EXPECT_EQ(5, isize(x));
  }

  // std::vector
  {
    std::vector<double> x{3.0, 5.0, 7.0};
    EXPECT_EQ(3, isize(x));
  }

  // Arbitrary type with static size() method
  {
    struct MyType {
      static constexpr size_t size() {
        return 123;
      }
    };
    static_assert(123 == isize(MyType{}));
    EXPECT_EQ(123, isize(MyType{}));
  }

  // Arbitrary type with non-static size() method
  {
    struct MyType {
      size_t _sz;
      constexpr explicit MyType(size_t sz) : _sz(sz) {}
      constexpr size_t size() const {
        return _sz;
      }
    };
    static_assert(123 == isize(MyType{123}));
    EXPECT_EQ(123, isize(MyType{123}));
  }
}

TEST(BasicUtils, Lerp) {
  // Note: Lerp does not clamp the 't' parameter
  EXPECT_NEAR_EQ(0_r, Lerp(10_r, 20_r, -1_r));
  EXPECT_NEAR_EQ(5_r, Lerp(10_r, 20_r, -0.5_r));
  EXPECT_NEAR_EQ(10_r, Lerp(10_r, 20_r, 0.0_r));
  EXPECT_NEAR_EQ(15_r, Lerp(10_r, 20_r, 0.5_r));
  EXPECT_NEAR_EQ(20_r, Lerp(10_r, 20_r, 1.0_r));
  EXPECT_NEAR_EQ(25_r, Lerp(10_r, 20_r, 1.5_r));
}

TEST(BasicUtils, Remap) {
  // Remap(value, fromMin, fromMax, toMin, toMax)

  // Remap: [0, 10] --> [100, 200]
  EXPECT_NEAR_EQ(100_r, Remap(0_r, 0_r, 10_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(110_r, Remap(1_r, 0_r, 10_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(150_r, Remap(5_r, 0_r, 10_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(200_r, Remap(10_r, 0_r, 10_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(210_r, Remap(11_r, 0_r, 10_r, 100_r, 200_r)); // value outside input range
  EXPECT_NEAR_EQ(90_r, Remap(-1_r, 0_r, 10_r, 100_r, 200_r)); // value outside input range

  // Remap: [10, 0] --> [100, 200] (input range inverted)
  EXPECT_NEAR_EQ(200_r, Remap(0_r, 10_r, 0_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(190_r, Remap(1_r, 10_r, 0_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(150_r, Remap(5_r, 10_r, 0_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(100_r, Remap(10_r, 10_r, 0_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(90_r, Remap(11_r, 10_r, 0_r, 100_r, 200_r)); // value outside input range
  EXPECT_NEAR_EQ(210_r, Remap(-1_r, 10_r, 0_r, 100_r, 200_r)); // value outside input range

  constexpr real kTol = 10_r * kDefaultNearEqualEpsilon<real>;

  // Remap: [0, 10] --> [200, 100] (output range inverted)
  EXPECT_NEAR_EQ(200_r, Remap(0_r, 0_r, 10_r, 200_r, 100_r));
  EXPECT_NEAR_EQ(190_r, Remap(1_r, 0_r, 10_r, 200_r, 100_r));
  EXPECT_NEAR_EQ(150_r, Remap(5_r, 0_r, 10_r, 200_r, 100_r));
  EXPECT_NEAR_EQ(100_r, Remap(10_r, 0_r, 10_r, 200_r, 100_r));
  EXPECT_NEAR_TOL(90_r, Remap(11_r, 0_r, 10_r, 200_r, 100_r), kTol); // value outside input range
  EXPECT_NEAR_EQ(210_r, Remap(-1_r, 0_r, 10_r, 200_r, 100_r)); // value outside input range

  // Remap: [10, 0] --> [200, 100] (both ranges inverted)
  EXPECT_NEAR_TOL(100_r, Remap(0_r, 10_r, 0_r, 200_r, 100_r), kTol);
  EXPECT_NEAR_TOL(110_r, Remap(1_r, 10_r, 0_r, 200_r, 100_r), kTol);
  EXPECT_NEAR_TOL(150_r, Remap(5_r, 10_r, 0_r, 200_r, 100_r), kTol);
  EXPECT_NEAR_TOL(200_r, Remap(10_r, 10_r, 0_r, 200_r, 100_r), kTol);
  EXPECT_NEAR_TOL(210_r, Remap(11_r, 10_r, 0_r, 200_r, 100_r), kTol); // value outside input range
  EXPECT_NEAR_TOL(90_r, Remap(-1_r, 10_r, 0_r, 200_r, 100_r), kTol); // value outside input range
}

TEST(BasicUtils, RemapAndClamp) {
  // Remap(value, fromMin, fromMax, toMin, toMax)

  // Remap: [0, 10] --> [100, 200]
  EXPECT_NEAR_EQ(100_r, RemapAndClamp(0_r, 0_r, 10_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(110_r, RemapAndClamp(1_r, 0_r, 10_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(150_r, RemapAndClamp(5_r, 0_r, 10_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(200_r, RemapAndClamp(10_r, 0_r, 10_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(200_r, RemapAndClamp(11_r, 0_r, 10_r, 100_r, 200_r)); // clamp to output range
  EXPECT_NEAR_EQ(100_r, RemapAndClamp(-1_r, 0_r, 10_r, 100_r, 200_r)); // clamp to output range

  // Remap: [10, 0] --> [100, 200] (input range inverted)
  EXPECT_NEAR_EQ(200_r, RemapAndClamp(0_r, 10_r, 0_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(190_r, RemapAndClamp(1_r, 10_r, 0_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(150_r, RemapAndClamp(5_r, 10_r, 0_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(100_r, RemapAndClamp(10_r, 10_r, 0_r, 100_r, 200_r));
  EXPECT_NEAR_EQ(100_r, RemapAndClamp(11_r, 10_r, 0_r, 100_r, 200_r)); // clamp to output range
  EXPECT_NEAR_EQ(200_r, RemapAndClamp(-1_r, 10_r, 0_r, 100_r, 200_r)); // clamp to output range

  // Remap: [0, 10] --> [200, 100] (output range inverted)
  EXPECT_NEAR_EQ(200_r, RemapAndClamp(0_r, 0_r, 10_r, 200_r, 100_r));
  EXPECT_NEAR_EQ(190_r, RemapAndClamp(1_r, 0_r, 10_r, 200_r, 100_r));
  EXPECT_NEAR_EQ(150_r, RemapAndClamp(5_r, 0_r, 10_r, 200_r, 100_r));
  EXPECT_NEAR_EQ(100_r, RemapAndClamp(10_r, 0_r, 10_r, 200_r, 100_r));
  EXPECT_NEAR_EQ(100_r, RemapAndClamp(11_r, 0_r, 10_r, 200_r, 100_r)); // clamp to output range
  EXPECT_NEAR_EQ(200_r, RemapAndClamp(-1_r, 0_r, 10_r, 200_r, 100_r)); // clamp to output range

  // Remap: [10, 0] --> [200, 100] (both ranges inverted)
  constexpr real kTolerance = kDefaultNearEqualEpsilon<real> * 10_r;
  EXPECT_NEAR_TOL(100_r, RemapAndClamp(0_r, 10_r, 0_r, 200_r, 100_r), kTolerance);
  EXPECT_NEAR_TOL(110_r, RemapAndClamp(1_r, 10_r, 0_r, 200_r, 100_r), kTolerance);
  EXPECT_NEAR_TOL(150_r, RemapAndClamp(5_r, 10_r, 0_r, 200_r, 100_r), kTolerance);
  EXPECT_NEAR_TOL(200_r, RemapAndClamp(10_r, 10_r, 0_r, 200_r, 100_r), kTolerance);
  EXPECT_NEAR_TOL(
      200_r, RemapAndClamp(11_r, 10_r, 0_r, 200_r, 100_r), kTolerance); // clamp to output range
  EXPECT_NEAR_TOL(
      100_r, RemapAndClamp(-1_r, 10_r, 0_r, 200_r, 100_r), kTolerance); // clamp to output range
}

TEST(BasicUtils, Pow) {
  EXPECT_NEAR_EQ(1_r, Pow(-5_r, 0_r));
  EXPECT_NEAR_EQ(-5_r, Pow(-5_r, 1_r));
  EXPECT_NEAR_EQ(25_r, Pow(-5_r, 2_r));
  EXPECT_NEAR_EQ(-125_r, Pow(-5_r, 3_r));
}

TEST(BasicUtils, Sqr) {
  EXPECT_EQ(25, Sqr(5));
  EXPECT_EQ(25, Sqr(-5));
  EXPECT_NEAR_EQ(25_r, Sqr(5_r));
  EXPECT_NEAR_EQ(25_r, Sqr(-5_r));
}

TEST(BasicUtils, SignedSqr) {
  EXPECT_EQ(25, SignedSqr(5));
  EXPECT_EQ(-25, SignedSqr(-5));
  EXPECT_NEAR_EQ(25_r, SignedSqr(5_r));
  EXPECT_NEAR_EQ(-25_r, SignedSqr(-5_r));
}

TEST(BasicUtils, Sqrt) {
  EXPECT_NEAR_EQ(5_r, Sqrt(25_r));
  EXPECT_FALSE(IsFinite(Sqrt(-25_r)));
}

TEST(BasicUtils, SignedSqrt) {
  EXPECT_NEAR_EQ(5_r, SignedSqrt(25_r));
  EXPECT_NEAR_EQ(-5_r, SignedSqrt(-25_r));
}

TEST(BasicUtils, IntegralSqrt) {
  // Basic cases: perfect squares.
  static_assert(1 == IntegralSqrt(1));
  static_assert(2 == IntegralSqrt(4));
  static_assert(3 == IntegralSqrt(9));
  static_assert(4 == IntegralSqrt(16));
  static_assert(5 == IntegralSqrt(25));

  // Edge cases: zero and non-perfect squares (floor semantics).
  static_assert(0 == IntegralSqrt(0));
  static_assert(1 == IntegralSqrt(2));
  static_assert(1 == IntegralSqrt(3));
  static_assert(2 == IntegralSqrt(5));
  static_assert(2 == IntegralSqrt(8));
  static_assert(3 == IntegralSqrt(10));
  static_assert(3 == IntegralSqrt(15));

  // Overflow case: `md * md` would overflow int32_t when `md > 46340`, which the binary search
  // probes for large inputs. Inside constexpr, UB is a compile error, so these asserts doubly
  // verify correctness and UB-freedom. 46340^2 = 2'147'395'600 is the largest int32_t square.
  static_assert(46340 == IntegralSqrt(int32_t{2147395600}));
  static_assert(46339 == IntegralSqrt(int32_t{2147395599}));
  static_assert(46340 == IntegralSqrt(std::numeric_limits<int32_t>::max()));

  // Same for int64_t: overflows when `md > 3037000499`. 3037000499^2 = 9,223,372,030,926,249,001 is
  // the largest int64_t square.
  static_assert(int64_t{3037000499} == IntegralSqrt(int64_t{9223372030926249001}));
  static_assert(int64_t{3037000499} == IntegralSqrt(std::numeric_limits<int64_t>::max()));
}

TEST(BasicUtils, IsPowerOfTwo) {
  EXPECT_FALSE(IsPowerOfTwo(-1));
  EXPECT_FALSE(IsPowerOfTwo(0));
  EXPECT_FALSE(IsPowerOfTwo(3));
  EXPECT_FALSE(IsPowerOfTwo(5L));
  EXPECT_FALSE(IsPowerOfTwo(6UL));
  EXPECT_FALSE(IsPowerOfTwo(7LL));
  EXPECT_FALSE(IsPowerOfTwo(9ULL));
  EXPECT_TRUE(IsPowerOfTwo(1));
  EXPECT_TRUE(IsPowerOfTwo(2));
  EXPECT_TRUE(IsPowerOfTwo(4L));
  EXPECT_TRUE(IsPowerOfTwo(8UL));
  EXPECT_TRUE(IsPowerOfTwo(16LL));
  EXPECT_TRUE(IsPowerOfTwo(32ULL));
}

TEST(BasicUtils, NextPowerOfTwo) {
  TestNextPowerOfTwo<int>();
  TestNextPowerOfTwo<unsigned int>();
  TestNextPowerOfTwo<int8_t>();
  TestNextPowerOfTwo<uint8_t>();
  TestNextPowerOfTwo<int16_t>();
  TestNextPowerOfTwo<uint16_t>();
  TestNextPowerOfTwo<int32_t>();
  TestNextPowerOfTwo<uint32_t>();
  TestNextPowerOfTwo<int64_t>();
  TestNextPowerOfTwo<uint64_t>();
}

TEST(BasicUtils, Rcp) {
  EXPECT_NEAR_EQ(1_r, Rcp(1_r));
  EXPECT_NEAR_EQ(-1_r, Rcp(-1_r));
  EXPECT_NEAR_EQ(0.5_r, Rcp(2_r));
  EXPECT_NEAR_EQ(-0.5_r, Rcp(-2_r));
  EXPECT_FALSE(IsFinite(Rcp(0_r)));
  EXPECT_FALSE(IsFinite(Rcp(-0_r)));
}

TEST(BasicUtils, RcpApprox) {
  EXPECT_NEAR_EQ(1_r, RcpApprox(1_r));
  EXPECT_NEAR_EQ(-1_r, RcpApprox(-1_r));
  EXPECT_NEAR_EQ(0.5_r, RcpApprox(2_r));
  EXPECT_NEAR_EQ(-0.5_r, RcpApprox(-2_r));
  EXPECT_FALSE(IsFinite(RcpApprox(0_r)));
  EXPECT_FALSE(IsFinite(RcpApprox(-0_r)));
}

TEST(BasicUtils, Floor) {
  EXPECT_NEAR_EQ(-3_r, Floor(-2.1_r));
  EXPECT_NEAR_EQ(-1_r, Floor(-1_r));
  EXPECT_NEAR_EQ(-1_r, Floor(-0.9_r));
  EXPECT_NEAR_EQ(-1_r, Floor(-0.1_r));
  EXPECT_NEAR_EQ(0_r, Floor(0.0_r));
  EXPECT_NEAR_EQ(0_r, Floor(0.1_r));
  EXPECT_NEAR_EQ(0_r, Floor(0.9_r));
  EXPECT_NEAR_EQ(1_r, Floor(1.0_r));
  EXPECT_NEAR_EQ(2_r, Floor(2.1_r));
}

TEST(BasicUtils, Ceil) {
  EXPECT_NEAR_EQ(-2_r, Ceil(-2.1_r));
  EXPECT_NEAR_EQ(-1_r, Ceil(-1_r));
  EXPECT_NEAR_EQ(0_r, Ceil(-0.9_r));
  EXPECT_NEAR_EQ(0_r, Ceil(-0.1_r));
  EXPECT_NEAR_EQ(0_r, Ceil(0.0_r));
  EXPECT_NEAR_EQ(1_r, Ceil(0.1_r));
  EXPECT_NEAR_EQ(1_r, Ceil(0.9_r));
  EXPECT_NEAR_EQ(1_r, Ceil(1.0_r));
  EXPECT_NEAR_EQ(3_r, Ceil(2.1_r));
}

TEST(BasicUtils, Round) {
  EXPECT_NEAR_EQ(-2_r, Round(-2.1_r));
  EXPECT_NEAR_EQ(-1_r, Round(-1_r));
  EXPECT_NEAR_EQ(-1_r, Round(-0.9_r));
  EXPECT_NEAR_EQ(0_r, Round(-0.1_r));
  EXPECT_NEAR_EQ(0_r, Round(0.0_r));
  EXPECT_NEAR_EQ(0_r, Round(0.1_r));
  EXPECT_NEAR_EQ(1_r, Round(0.9_r));
  EXPECT_NEAR_EQ(1_r, Round(1.0_r));
  EXPECT_NEAR_EQ(2_r, Round(2.1_r));

  // 0.5 rounds away from zero
  EXPECT_NEAR_EQ(-2_r, Round(-1.5_r));
  EXPECT_NEAR_EQ(-1_r, Round(-0.5_r));
  EXPECT_NEAR_EQ(1_r, Round(0.5_r));
  EXPECT_NEAR_EQ(2_r, Round(1.5_r));
}

TEST(BasicUtils, RoundUp) {
  EXPECT_EQ(0, RoundUp(0, 1));
  EXPECT_EQ(1, RoundUp(1, 1));
  EXPECT_EQ(2, RoundUp(2, 1));
  EXPECT_EQ(0, RoundUp(0, 2));
  EXPECT_EQ(2, RoundUp(1, 2));
  EXPECT_EQ(2, RoundUp(2, 2));
  EXPECT_EQ(4, RoundUp(3, 2));
  EXPECT_EQ(0, RoundUp(0, 3));
  EXPECT_EQ(3, RoundUp(1, 3));
  EXPECT_EQ(3, RoundUp(2, 3));
  EXPECT_EQ(3, RoundUp(3, 3));
  EXPECT_EQ(6, RoundUp(4, 3));
}

TEST(BasicUtils, RoundDown) {
  EXPECT_EQ(0, RoundDown(0, 1));
  EXPECT_EQ(1, RoundDown(1, 1));
  EXPECT_EQ(2, RoundDown(2, 1));
  EXPECT_EQ(0, RoundDown(0, 2));
  EXPECT_EQ(0, RoundDown(1, 2));
  EXPECT_EQ(2, RoundDown(2, 2));
  EXPECT_EQ(2, RoundDown(3, 2));
  EXPECT_EQ(0, RoundDown(0, 3));
  EXPECT_EQ(0, RoundDown(1, 3));
  EXPECT_EQ(0, RoundDown(2, 3));
  EXPECT_EQ(3, RoundDown(3, 3));
  EXPECT_EQ(3, RoundDown(4, 3));
}

TEST(BasicUtils, Select) {
  // int
  EXPECT_EQ(1, Select(true, 1, 2));
  EXPECT_EQ(2, Select(false, 1, 2));

  // real
  EXPECT_EQ(1_r, Select(true, 1_r, 2_r));
  EXPECT_EQ(2_r, Select(false, 1_r, 2_r));
}

TEST(BasicUtils, Sign) {
  EXPECT_EQ(1, Sign(0));
  EXPECT_EQ(1, Sign(-0));
  EXPECT_EQ(1, Sign(1));
  EXPECT_EQ(-1, Sign(-1));

  EXPECT_EQ(1_r, Sign(0_r));
  EXPECT_EQ(1_r, Sign(-0_r)); // -0.0 and 0.0 are treated the same
  EXPECT_EQ(1_r, Sign(1_r));
  EXPECT_EQ(-1_r, Sign(-1_r));
}

TEST(BasicUtils, Sin_Cos_Tan_ATan) {
  constexpr real kValues[] = {0_r, 1_r, -2_r, 0.25_r * kPI, 0.5_r * kPI, kPI, -2_r * kPI};
  for (real x : kValues) {
    EXPECT_NEAR_EQ(std::sin(x), Sin(x));
    EXPECT_NEAR_EQ(std::cos(x), Cos(x));
    EXPECT_NEAR_EQ(std::tan(x), Tan(x));
    EXPECT_NEAR_EQ(std::atan(x), ATan(x));
    for (real y : kValues) {
      EXPECT_NEAR_EQ(std::atan2(y, x), ATan2(y, x));
    }
  }
}

TEST(BasicUtils, Sinc) {
  constexpr real kValues[] = {1_r, -2_r, 0.25_r * kPI, 0.5_r * kPI, kPI, -2_r * kPI};
  for (real x : kValues) {
    EXPECT_NEAR_EQ(std::sin(x) / x, Sinc(x));
  }
  // Returns 1 if x is near zero.
  EXPECT_NEAR_EQ(1_r, Sinc(0_r));
  EXPECT_NEAR_EQ(1_r, Sinc(-0_r));
  EXPECT_NEAR_EQ(1_r, Sinc(1e-9_r));
  EXPECT_NEAR_EQ(1_r, Sinc(-1e-9_r));
}

TEST(BasicUtils, ASin_ACos) {
  constexpr real kValues[] = {0_r, 0.5_r, -0.75_r, 1_r, -1_r};
  for (real x : kValues) {
    EXPECT_NEAR_EQ(std::asin(x), ASin(x));
    EXPECT_NEAR_EQ(std::acos(x), ACos(x));
  }
  EXPECT_TRUE(std::isnan(ASin(-1.1_r)));
  EXPECT_TRUE(std::isnan(ASin(1.1_r)));
  EXPECT_TRUE(std::isnan(ACos(-1.1_r)));
  EXPECT_TRUE(std::isnan(ACos(1.1_r)));
}
