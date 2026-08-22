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

#include "simd_test.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/half.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

using namespace mochi;
using namespace mochi::simd_test;

static_assert(std::is_trivially_copyable_v<Simd<double, 2>>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec2d);

#define EXPECT_VEC2D(e0, e1, actual)                                                \
  {                                                                                 \
    Vec2d vActual = actual;                                                         \
    /* Testing for exact equality handles values like +/- infinity */               \
    EXPECT_TRUE(((e0) == vActual[0]) || NearEqual((e0), vActual[0], double(kEps))); \
    EXPECT_TRUE(((e1) == vActual[1]) || NearEqual((e1), vActual[1], double(kEps))); \
  }

TEST(Vec2d, Class) {
  static_assert(Vec2d::kIsSupported);
  static_assert(!Vec2d::kIsComposite);
  static_assert(Vec2d::kIsEmulated == !(MOCHI_USE_SIMD));
  static_assert(sizeof(Vec2d) == sizeof(double) * 2);
  static_assert(alignof(Vec2d) == alignof(typename Vec2d::NativeType));
  static_assert(std::is_same_v<Vec2d::Scalar, double>);
  static_assert(Vec2d::kSize == 2);
  static_assert(Vec2d::size() == 2);
  // Construct from broadcast
  EXPECT_VEC2D(1.0, 1.0, Vec2d(1.0));

  // Construct from scalars
  EXPECT_VEC2D(1.0, 1.0, Vec2d(1.0));
  EXPECT_VEC2D(1.0, 2.0, Vec2d(1.0, 2.0));

  // Implicit conversion from scalar
  Vec2d a = 2.0;
  EXPECT_VEC2D(2.0, 2.0, a);
  a = 3.0;
  EXPECT_VEC2D(3.0, 3.0, a);

  // Copy construct
  Vec2d b{a};
  EXPECT_VEC2D(3.0, 3.0, b);

  // Copy assign
  a = Vec2d{4.0};
  b = a;
  EXPECT_VEC2D(4.0, 4.0, b);
  // Comparison
  EXPECT_EQ(true, (a == b));
  EXPECT_EQ(true, (a != Vec2d{}));
  EXPECT_EQ(false, (a != b));
  EXPECT_EQ(false, (a == Vec2d{}));

  // Unary operators
  Vec2d ones = ReinterpretCast<Vec2d>(Vec4i{-1}); // {-nan,-nan} must be compared using Vec4i
  Vec2d zeros = {};
  EXPECT_EQ(ReinterpretCast<Vec4i>(ones), ReinterpretCast<Vec4i>(~zeros));
  EXPECT_EQ(ReinterpretCast<Vec4i>(zeros), ReinterpretCast<Vec4i>(~ones));
  EXPECT_VEC2D(-1.0, -2.0, -Vec2d(1.0, 2.0));

  // Binary operators
  a = Vec2d{1.0, 2.0};
  b = Vec2d{5.0, 6.0};
  EXPECT_VEC2D(6.0, 8.0, a + b);
  EXPECT_VEC2D(4.0, 5.0, a + 3.0);
  EXPECT_VEC2D(4.0, 5.0, 3.0 + a);
  EXPECT_VEC2D(-4.0, -4.0, a - b);
  EXPECT_VEC2D(-2.0, -1.0, a - 3.0);
  EXPECT_VEC2D(2.0, 1.0, 3.0 - a);
  EXPECT_VEC2D(5.0, 12.0, a * b);
  EXPECT_VEC2D(3.0, 6.0, a * 3.0);
  EXPECT_VEC2D(3.0, 6.0, 3.0 * a);
  EXPECT_VEC2D(5.0, 3.0, b / a);
  EXPECT_VEC2D(1.0 / 3.0, 2.0 / 3.0, a / 3.0);
  EXPECT_VEC2D(3.0 / 1.0, 3.0 / 2.0, 3.0 / a);
  EXPECT_EQ(a, a & ones);
  EXPECT_EQ(zeros, a & zeros);
  EXPECT_EQ(ReinterpretCast<Vec4i>(ones), ReinterpretCast<Vec4i>(a | ones));
  EXPECT_EQ(a, a | zeros);
  EXPECT_EQ(~a, a ^ ones);
  EXPECT_EQ(a, a ^ zeros);

  // Update operators
  a = Vec2d{1.0, 2.0};
  a += b;
  EXPECT_VEC2D(6.0, 8.0, a);
  a = Vec2d{1.0, 2.0};
  a += 3.0;
  EXPECT_VEC2D(4.0, 5.0, a);
  a = Vec2d{1.0, 2.0};
  a -= b;
  EXPECT_VEC2D(-4.0, -4.0, a);
  a = Vec2d{1.0, 2.0};
  a -= 3.0;
  EXPECT_VEC2D(-2.0, -1.0, a);
  a = Vec2d{1.0, 2.0};
  a *= b;
  EXPECT_VEC2D(5.0, 12.0, a);
  a = Vec2d{1.0, 2.0};
  a *= 3.0;
  EXPECT_VEC2D(3.0, 6.0, a);
  a = Vec2d{1.0, 2.0};
  b = Vec2d{5.0, 6.0};
  b /= a;
  EXPECT_VEC2D(5.0, 3.0, b);
  a = Vec2d{1.0, 2.0};
  a /= 3.0;
  EXPECT_VEC2D(1.0 / 3.0, 2.0 / 3.0, a);
  b = a;
  b &= ones;
  EXPECT_EQ(a, b);
  a = Vec2d{1.0, 2.0};
  a &= zeros;
  EXPECT_EQ(zeros, a);
  a = Vec2d{1.0, 2.0};
  a |= ones;
  EXPECT_EQ(ReinterpretCast<Vec4i>(ones), ReinterpretCast<Vec4i>(a));
  a = b = Vec2d{1.0, 2.0};
  b |= zeros;
  EXPECT_EQ(a, b);
  a = b = Vec2d{1.0, 2.0};
  b ^= ones;
  EXPECT_EQ(~a, b);
  a = b = Vec2d{1.0, 2.0};
  b ^= zeros;
  EXPECT_EQ(a, b);
}

MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, Abs, std::abs, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec2d, ACos, std::acos, -0.9, 0.9, kEps);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec2d, Add, +, kEps);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec2d, BitwiseAND, &);
MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(Vec2d);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec2d, ASin, std::asin, -0.9, 0.9, kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, ATan, std::atan, kEps);

TEST(Vec2d, AllTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  double zer = 0.0;
  double one = 0.0;
  auto allOnes = static_cast<uint64_t>(-1);
  memcpy(&one, &allOnes, sizeof(double));
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      auto vec = Vec2d{a ? one : zer, b ? one : zer};
      EXPECT_EQ(!!a, AllTrue<1>(vec));
      EXPECT_EQ(a && b, AllTrue<2>(vec));
      EXPECT_EQ(a && b, AllTrue(vec));
    }
  }
}

TEST(Vec2d, AnyTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  double zer = 0.0;
  double one = 0.0;
  auto allOnes = static_cast<uint64_t>(-1);
  memcpy(&one, &allOnes, sizeof(double));
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      auto vec = Vec2d{a ? one : zer, b ? one : zer};
      EXPECT_EQ(!!a, AnyTrue<1>(vec));
      EXPECT_EQ(a || b, AnyTrue<2>(vec));
      EXPECT_EQ(a || b, AnyTrue(vec));
    }
  }
}

TEST(Vec2d, Blend) {
  auto a = Vec2d{1.0, 2.0};
  auto b = Vec2d{3.0, 4.0};
  EXPECT_VEC2D(1.0, 2.0, (Blend<0, 0>(a, b)));
  EXPECT_VEC2D(1.0, 4.0, (Blend<0, 1>(a, b)));
  EXPECT_VEC2D(3.0, 2.0, (Blend<1, 0>(a, b)));
  EXPECT_VEC2D(3.0, 4.0, (Blend<1, 1>(a, b)));
}

TEST(Vec2d, IsFinite) {
  constexpr double kInfd = std::numeric_limits<double>::infinity();
  constexpr double kQNaN = std::numeric_limits<double>::quiet_NaN();
  constexpr double kSNaN = std::numeric_limits<double>::signaling_NaN();
  EXPECT_TRUE(IsFinite(Vec2d{}));
  EXPECT_TRUE(IsFinite(Vec2d{1.0, 2.0}));
  EXPECT_TRUE(IsFinite(Vec2d{std::numeric_limits<double>::min()}));
  EXPECT_TRUE(IsFinite(Vec2d{-std::numeric_limits<double>::min()}));
  EXPECT_TRUE(IsFinite(Vec2d{std::numeric_limits<double>::lowest()}));
  EXPECT_TRUE(IsFinite(Vec2d{-std::numeric_limits<double>::lowest()}));
  EXPECT_TRUE(IsFinite(Vec2d{std::numeric_limits<double>::max()}));
  EXPECT_TRUE(IsFinite(Vec2d{-std::numeric_limits<double>::max()}));
  EXPECT_FALSE(IsFinite(Vec2d(kInfd, 0.0)));
  EXPECT_FALSE(IsFinite(Vec2d(0.0, -kInfd)));
  EXPECT_FALSE(IsFinite(Vec2d(kQNaN, 0.0)));
  EXPECT_FALSE(IsFinite(Vec2d(0.0, kSNaN)));
  EXPECT_VEC2D(0.0, 1.0, VIsFinite(Vec2d(kInfd, 0.0)) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, VIsFinite(Vec2d(0.0, -kInfd)) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 1.0, VIsFinite(Vec2d(kQNaN, 0.0)) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, VIsFinite(Vec2d(0.0, kQNaN)) & Vec2d{1.0});
}

TEST(Vec2d, IsTrue) {
  auto a = SimdMask<Vec2d>(true, false);
  EXPECT_TRUE(IsTrue<0>(a));
  EXPECT_FALSE(IsTrue<1>(a));
  EXPECT_FALSE(IsTrue<0>(~a));
  EXPECT_TRUE(IsTrue<1>(~a));
}

// Test both Equal and NotEqual
TEST(Vec2d, Equal) {
  auto a = Vec2d{1.0, 2.0};
  auto b = a;
  auto c = Vec2d{1.1, 2.0};
  auto d = Vec2d{1.0, 2.1};

  // Compare 1 component
  ExpectEqual<1>(true, a, a);
  ExpectEqual<1>(true, a, b);
  ExpectEqual<1>(false, a, c);
  ExpectEqual<1>(true, a, d);

  // Compare 2 components
  ExpectEqual<2>(true, a, a);
  ExpectEqual<2>(true, a, b);
  ExpectEqual<2>(false, a, c);
  ExpectEqual<2>(false, a, d);
}

TEST(Vec2d, NearEqual) {
  auto a = Vec2d{1.0, 2.0};
  auto b = Vec2d{1.1, 2.0};
  auto c = Vec2d{1.0, 2.1};

  // Compare 1 component
  EXPECT_TRUE(NearEqual<1>(a, a, 0.0));
  EXPECT_FALSE(NearEqual<1>(a, b, 0.0));
  EXPECT_TRUE(NearEqual<1>(a, b, 0.11));
  EXPECT_TRUE(NearEqual<1>(a, c, 0.0));
  EXPECT_TRUE(NearEqual<1>(a, c, 0.11));

  // Compare 2 components (default)
  EXPECT_TRUE(NearEqual(a, a, 0.0));
  EXPECT_FALSE(NearEqual(a, b, 0.0));
  EXPECT_TRUE(NearEqual(a, b, 0.11));
  EXPECT_FALSE(NearEqual(a, c, 0.0));
  EXPECT_TRUE(NearEqual(a, c, 0.11));
}

TEST(Vec2d, VNearEqual) {
  auto a = Vec2d{1.0, 2.0};
  auto b = Vec2d{1.1, 2.0};
  auto c = Vec2d{1.0, 2.1};
  EXPECT_VEC2D(1.0, 1.0, VNearEqual(a, a, Vec2d(0.0, 0.0)) & Vec2d(1.0));
  EXPECT_VEC2D(0.0, 1.0, VNearEqual(a, b, Vec2d(0.0, 0.0)) & Vec2d(1.0));
  EXPECT_VEC2D(1.0, 1.0, VNearEqual(a, b, Vec2d(0.11, 0.0)) & Vec2d(1.0));
  EXPECT_VEC2D(1.0, 0.0, VNearEqual(a, c, Vec2d(0.0, 0.0)) & Vec2d(1.0));
  EXPECT_VEC2D(1.0, 1.0, VNearEqual(a, c, Vec2d(0.0, 0.11)) & Vec2d(1.0));
}

TEST(Vec2d, NearZero) {
  auto a = Vec2d{0.1, -0.2};

  // Default tolerance
  EXPECT_TRUE(NearZero(Vec2d{}));
  EXPECT_TRUE(NearZero(Vec2d{kDefaultNearEqualEpsilon<double>}));
  EXPECT_TRUE(NearZero(-Vec2d{kDefaultNearEqualEpsilon<double>}));
  EXPECT_FALSE(NearZero(a));

  // Compare 1 component
  EXPECT_FALSE(NearZero<1>(a, 0.0));
  EXPECT_TRUE(NearZero<1>(a, 0.11));

  // Compare 2 component (default)
  EXPECT_FALSE(NearZero(a, 0.0));
  EXPECT_FALSE(NearZero(a, 0.11));
  EXPECT_TRUE(NearZero(a, 0.21));
}

TEST(Vec2d, VNearZero) {
  auto z = Vec2d{};
  auto a = Vec2d{0.1, -0.2};
  EXPECT_VEC2D(1.0, 1.0, VNearZero(z, Vec2d(0.0, 0.0)) & Vec2d(1.0));
  EXPECT_VEC2D(0.0, 0.0, VNearZero(a, Vec2d(0.0, 0.0)) & Vec2d(1.0));
  EXPECT_VEC2D(1.0, 0.0, VNearZero(a, Vec2d(0.11, 0.0)) & Vec2d(1.0));
  EXPECT_VEC2D(0.0, 0.0, VNearZero(a, Vec2d(0.0, 0.11)) & Vec2d(1.0));
  EXPECT_VEC2D(1.0, 0.0, VNearZero(a, Vec2d(0.21, 0.11)) & Vec2d(1.0));
  EXPECT_VEC2D(1.0, 1.0, VNearZero(a, Vec2d(0.21, 0.21)) & Vec2d(1.0));
}

TEST(Vec2d, Broadcast) {
  // Broadcast scalar
  EXPECT_VEC2D(1.0, 1.0, Broadcast<Vec2d>(1.0));

  // Broadcast from address
  double const s = 2.0;
  EXPECT_VEC2D(2.0, 2.0, Broadcast<Vec2d>(&s));

  // Broadcast ith element (template)
  auto v = Vec2d{3.0, 4.0};
  EXPECT_VEC2D(3.0, 3.0, Broadcast<0>(v));
  EXPECT_VEC2D(4.0, 4.0, Broadcast<1>(v));

  // Broadcast ith element (dynamic index)
  EXPECT_VEC2D(3.0, 3.0, Broadcast(v, 0));
  EXPECT_VEC2D(4.0, 4.0, Broadcast(v, 1));
}

MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec2d, Clamp, Clamp, kEps);

TEST(Vec2d, Cos) {
  TestSimdTrigFunction<Vec2d>(
      [](double x) { return std::cos(x); }, [](Vec2d x) { return Cos(x); }, GetTrigTestValues());
}

MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec2d, Div, /, kEps);

TEST(Vec2d, VDot) {
  auto a = Vec2d{1.0, 2.0};
  auto b = Vec2d{3.0, 4.0};
  EXPECT_VEC2D(11.0, 11.0, (VDot(a, b)));
}

// Test bot VEqual and VNotEqual
template <class V>
static void ExpectVEqual(bool e0, bool e1, V a, V b) {
  EXPECT_VEC2D(e0 ? 1.0 : 0.0, e1 ? 1.0 : 0.0, VEqual(a, b) & Vec2d{1.0});
  EXPECT_VEC2D(e0 ? 0.0 : 1.0, e1 ? 0.0 : 1.0, VNotEqual(a, b) & Vec2d{1.0});
}

TEST(Vec2d, VEqual) {
  auto a = Vec2d{1.0, 2.0};
  auto b = Vec2d{1.0, 9.0};
  auto c = Vec2d{9.0, 2.0};
  ExpectVEqual(1, 1, a, a);
  ExpectVEqual(1, 0, a, b);
  ExpectVEqual(0, 1, a, c);
}

TEST(Vec2d, Norm) {
  Vec2d const kValues[] = {Vec2d{0.0, 0.0}, Vec2d{2.0, 0.0}, Vec2d{0.0, -2.0}, Vec2d{-2.0, 3.0}};
  for (auto v : kValues) {
    // 2 components (default)
    auto nsqr = Sqr(Get<0>(v)) + Sqr(Get<1>(v));
    auto norm = std::sqrt(nsqr);
    EXPECT_NEAR_EQ(Vec2d(norm), VNorm(v));
    EXPECT_NEAR_EQ(Vec2d(nsqr), VNormSqr(v));
    EXPECT_NEAR_EQ(norm, Norm(v));
    EXPECT_NEAR_EQ(nsqr, NormSqr(v));
  }
}

TEST(Vec2d, Normalize) {
  EXPECT_NEAR_EQ(Vec2d(), Normalize(Vec2d())); // Zero in, zero out
  EXPECT_NEAR_EQ(Vec2d(1.0, 0.0), Normalize(Vec2d(1.0, 0.0)));
  EXPECT_NEAR_EQ(Vec2d(0.0, -1.0), Normalize(Vec2d(0.0, -2.0)));
  EXPECT_NEAR_EQ(Vec2d(1.0, 2.0) / Sqrt(5.0), Normalize(Vec2d(1.0, 2.0)));

  // Normalize(v, normSqr) overloads — scalar and Simd normSqr
  auto const v = Vec2d{1.0, 2.0};
  auto const expected = v / Sqrt(5.0);
  EXPECT_NEAR_EQ(expected, Normalize(v, 5.0)); // scalar normSqr
  EXPECT_NEAR_EQ(expected, Normalize(v, Vec2d{5.0, 5.0})); // Simd normSqr
}

TEST(Vec2d, Get) {
  auto a = Vec2d{1.0, 2.0};

  // Fast template version
  EXPECT_EQ(1.0, Get0(a));
  EXPECT_EQ(1.0, Get<0>(a));
  EXPECT_EQ(2.0, Get<1>(a));

  // Slower runtime version
  EXPECT_EQ(1.0, Get(a, 0));
  EXPECT_EQ(2.0, Get(a, 1));

  // Same but with operator[] (read only)
  EXPECT_EQ(1.0, a[0]);
  EXPECT_EQ(2.0, a[1]);
}

TEST(Vec2d, Set) {
  auto a = Vec2d{1.0, 2.0};
  EXPECT_VEC2D(9.0, 2.0, Set<0>(a, 9.0));
  EXPECT_VEC2D(1.0, 9.0, Set<1>(a, 9.0));

  EXPECT_VEC2D(9.0, 2.0, Set(a, 0, 9.0));
  EXPECT_VEC2D(1.0, 9.0, Set(a, 1, 9.0));
}

TEST(Vec2d, Greater) {
  auto a = Vec2d{1.0, 4.0};
  auto b = Vec2d{2.0, 3.0};
  EXPECT_VEC2D(0.0, 1.0, (a > b) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (b > a) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 0.0, (a > a) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 1.0, (a > Vec2d{3.0}) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 0.0, (a > Vec2d{4.0}) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (Vec2d{3.0} > a) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (Vec2d{4.0} > a) & Vec2d{1.0});
}

TEST(Vec2d, GreaterEqual) {
  auto a = Vec2d{1.0, 4.0};
  auto b = Vec2d{2.0, 3.0};
  EXPECT_VEC2D(0.0, 1.0, (a >= b) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (b >= a) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 1.0, (a >= a) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 1.0, (a >= Vec2d{3.0}) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 1.0, (a >= Vec2d{4.0}) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (Vec2d{3.0} >= a) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 1.0, (Vec2d{4.0} >= a) & Vec2d{1.0});
}

TEST(Vec2d, HMax) {
  auto a = Vec2d{1_r, 2_r};
  EXPECT_EQ(2_r, HMax(a));
}

TEST(Vec2d, HMin) {
  auto a = Vec2d{2_r, 1_r};
  EXPECT_EQ(1_r, HMin(a));
}

TEST(Vec2d, HProd) {
  auto a = Vec2d{2.0, 3.0};
  EXPECT_NEAR_EQ(6.0, HProd(a));
}

TEST(Vec2d, HSum) {
  auto a = Vec2d{1.0, 2.0};
  EXPECT_NEAR_EQ(3.0, HSum(a));
}

TEST(Vec2d, Lerp) {
  // Note: Lerp does not clamp the 't' parameter
  auto const a = Vec2d{10.0, 100.0};
  auto const b = Vec2d{20.0, 200.0};
  EXPECT_VEC2D(0.0, 0.0, Lerp(a, b, -1.0));
  EXPECT_VEC2D(5.0, 50.0, Lerp(a, b, -0.5));
  EXPECT_VEC2D(10.0, 100.0, Lerp(a, b, 0.0));
  EXPECT_VEC2D(15.0, 150.0, Lerp(a, b, 0.5));
  EXPECT_VEC2D(20.0, 200.0, Lerp(a, b, 1.0));
  EXPECT_VEC2D(25.0, 250.0, Lerp(a, b, 1.5));
}

TEST(Vec2d, Less) {
  auto a = Vec2d{1.0, 4.0};
  auto b = Vec2d{2.0, 3.0};
  EXPECT_VEC2D(0.0, 1.0, (b < a) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (a < b) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 0.0, (a < a) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (a < Vec2d{3.0}) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (a < Vec2d{4.0}) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 1.0, (Vec2d{3.0} < a) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 0.0, (Vec2d{4.0} < a) & Vec2d{1.0});
}

TEST(Vec2d, LessEqual) {
  auto a = Vec2d{1.0, 4.0};
  auto b = Vec2d{2.0, 3.0};
  EXPECT_VEC2D(0.0, 1.0, (b <= a) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (a <= b) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 1.0, (a <= a) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 0.0, (a <= Vec2d{3.0}) & Vec2d{1.0});
  EXPECT_VEC2D(1.0, 1.0, (a <= Vec2d{4.0}) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 1.0, (Vec2d{3.0} <= a) & Vec2d{1.0});
  EXPECT_VEC2D(0.0, 1.0, (Vec2d{4.0} <= a) & Vec2d{1.0});
}

TEST(Vec2d, Load) {
  alignas(alignof(Vec2d)) double const values[] = {0.0, 1.0, 2.0};
  EXPECT_VEC2D(0.0, 0.0, (Load<0, Vec2d>(nullptr)));
  EXPECT_VEC2D(1.0, 0.0, (Load<1, Vec2d>(values + 1)));
  EXPECT_VEC2D(1.0, 2.0, (Load<2, Vec2d>(values + 1)));
  EXPECT_VEC2D(1.0, 2.0, (Load<Vec2d>(values + 1)));

  EXPECT_VEC2D(1.0, 0.0, (Load<Vec2d>(values + 1, 1)));
  EXPECT_VEC2D(1.0, 2.0, (Load<Vec2d>(values + 1, 2)));
}

TEST(Vec2d, LoadIndexed) {
  [[maybe_unused]] alignas(alignof(Vec2d)) double const values[] = {0.0, 1.0, 2.0, 3.0, 4.0};
  EXPECT_VEC2D(2.0, 4.0, LoadIndexed<Vec2d>(&values[1], Vec2l(1, 3)));
}

TEST(Vec2d, LoadTransposed) {
  alignas(alignof(Vec2d)) double const values[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto const* ptr = values + 1; // Not an aligned address
  Vec2d loaded[3] = {};
  LoadTransposed<1>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC2D(1.0, 0.0, loaded[0]);
  EXPECT_VEC2D(2.0, 0.0, loaded[1]);
  EXPECT_VEC2D(3.0, 0.0, loaded[2]);
  LoadTransposed<2>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC2D(1.0, 4.0, loaded[0]);
  EXPECT_VEC2D(2.0, 5.0, loaded[1]);
  EXPECT_VEC2D(3.0, 6.0, loaded[2]);
  LoadTransposed(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC2D(1.0, 4.0, loaded[0]);
  EXPECT_VEC2D(2.0, 5.0, loaded[1]);
  EXPECT_VEC2D(3.0, 6.0, loaded[2]);
}

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, Floor, ([](auto a) { return std::floor(a); }), kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, FastRound, ([](auto a) { return std::round(a); }), kEps); // The standard test values have no exact ties
MOCHI_SIMD_TEST_BINARY_FN_NEAR(Vec2d, Max, ([](auto a, auto b) { return std::max(a, b); }), kEps);
MOCHI_SIMD_TEST_BINARY_FN_NEAR(Vec2d, Min, ([](auto a, auto b) { return std::min(a, b); }), kEps);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec2d, Mul, *, kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec2d, MulAdd, ([](auto a, auto b, auto c) { return a * b + c; }), kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec2d, MulSub, ([](auto a, auto b, auto c) { return a * b - c; }), kEps);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec2d, Neg, -);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec2d, NegMulAdd, ([](auto a, auto b, auto c) { return -(a * b) + c; }), kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec2d, NegMulSub, ([](auto a, auto b, auto c) { return -(a * b) - c; }), kEps);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec2d, BitwiseNOT, ~);
// clang-format on

TEST(Vec2d, VNotEqual) {
  auto a = Vec2d{1.0, 2.0};
  auto b = Vec2d{1.0, 3.0};
  EXPECT_VEC2D(0.0, 1.0, VNotEqual(a, b) & Broadcast<Vec2d>(1.0));
  EXPECT_VEC2D(0.0, 0.0, VNotEqual(a, a) & Broadcast<Vec2d>(1.0));
}

// clang-format off
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec2d, BitwiseOR, |);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec2d, LogicalAND, &&, &);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec2d, LogicalOR, ||, |);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, RcpApprox, ([](auto a) { return 1 / a; }), 1e-2); // Large tolerance because these are only approximations
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, RcpSqrtApprox, ([](auto a) { return 1 / std::sqrt(a); }), 5e-3);
// clang-format on

TEST(Vec2d, Select) {
  auto a = Vec2d{1.0, 2.0};
  auto b = Vec2d{3.0, 4.0};
  auto t = true;
  auto f = false;
  EXPECT_VEC2D(1.0, 2.0, Select(SimdMask<Vec2d>(t, t), a, b));
  EXPECT_VEC2D(1.0, 4.0, Select(SimdMask<Vec2d>(t, f), a, b));
  EXPECT_VEC2D(3.0, 2.0, Select(SimdMask<Vec2d>(f, t), a, b));
  EXPECT_VEC2D(3.0, 4.0, Select(SimdMask<Vec2d>(f, f), a, b));
  EXPECT_VEC2D(1.0, 2.0, Select(a < b, a, b)); // Min(a, b)
  EXPECT_VEC2D(3.0, 4.0, Select(a > b, a, b)); // Max(a, b)
  EXPECT_VEC2D(1.0, 4.0, Select(a <= Vec2d{1.0}, a, b));
  EXPECT_VEC2D(3.0, 2.0, Select(a > Vec2d{1.0}, a, b));
}

TEST(Vec2d, SimdMask) {
  auto v = Vec2d{1.0, 2.0};
  bool constexpr f = false;
  bool constexpr t = true;
  EXPECT_VEC2D(0.0, 0.0, v & SimdMask<Vec2d>(f, f));
  EXPECT_VEC2D(1.0, 0.0, v & SimdMask<Vec2d>(t, f));
  EXPECT_VEC2D(0.0, 2.0, v & SimdMask<Vec2d>(f, t));
  EXPECT_VEC2D(0.0, 0.0, v & SimdMask<Vec2d>(f, f));
  EXPECT_VEC2D(0.0, 0.0, v & SimdMask<Vec2d>(f, f));
  EXPECT_VEC2D(1.0, 0.0, v & SimdMask<Vec2d>(t, f));
  EXPECT_VEC2D(1.0, 2.0, v & SimdMask<Vec2d>(t, t));
}

TEST(Vec2d, SimdZero) {
  EXPECT_VEC2D(0.0, 0.0, SimdZero<Vec2d>());
}

TEST(Vec2d, Shuffle) {
  auto v = Vec2d{1.0, 2.0};
  EXPECT_VEC2D(1.0, 1.0, (Shuffle<0, 0>(v)));
  EXPECT_VEC2D(1.0, 2.0, (Shuffle<0, 1>(v)));
  EXPECT_VEC2D(2.0, 1.0, (Shuffle<1, 0>(v)));
  EXPECT_VEC2D(2.0, 2.0, (Shuffle<1, 1>(v)));
}

#if MOCHI_USE_SIMD // SignBitMask is a utility only implemented in native SIMD types
TEST(Vec2d, SignBitMask) {
  auto const signBit = Vec2d::SignBitMask(); // -0.0 (all bits set to 0 except the sign bit)
  auto const a = Vec2d{1.0, -2.0};
  EXPECT_EQ(Vec2d{}, signBit); // 0.0 == -0.0
  EXPECT_EQ(-a, a ^ signBit); // flip sign
  EXPECT_EQ(a, -a ^ signBit); // flip sign
}
#endif

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, Sign, ([](auto a) { return (a >= 0) ? 1 : -1; }), kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, SignedSqrt, ([](auto a) { return std::sqrt(std::abs(a)) * ((a >= 0) ? 1 : -1); }), kEps);
// clang-format on

TEST(Vec2d, Sin) {
  TestSimdTrigFunction<Vec2d>(
      [](double x) { return std::sin(x); }, [](Vec2d x) { return Sin(x); }, GetTrigTestValues());
}

TEST(Vec2d, SinCos) {
  auto values = GetTrigTestValues();
  TestSimdTrigFunction<Vec2d>(
      [](double x) { return std::sin(x); }, [](Vec2d x) { return SinCos(x).first; }, values);
  TestSimdTrigFunction<Vec2d>(
      [](double x) { return std::cos(x); }, [](Vec2d x) { return SinCos(x).second; }, values);
}

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, Sqr, ([](auto a) { return a * a; }), kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec2d, Sqrt, ([](auto a) { return std::sqrt(a); }), 0.0, 10.0, kEps);
// clang-format on

TEST(Vec2d, Store) {
  std::vector<double> result(
      3); // NOTE: Changed from an array on the stack to work around an MSVC optimizer bug.
  Store<0>((double*)nullptr, Vec2d(1.0, 2.0));
  Store<0>(&result[1], Vec2d(1.0, 2.0));
  EXPECT_SPAN_EQ((std::array<double, 2>{0.0, 0.0}), Span(&result[1], 2));
  Store<1>(&result[1], Vec2d(1.0, 2.0));
  EXPECT_SPAN_EQ((std::array<double, 2>{1.0, 0.0}), Span(&result[1], 2));
  Store<2>(&result[1], Vec2d(1.0, 2.0));
  EXPECT_SPAN_EQ((std::array<double, 2>{1.0, 2.0}), Span(&result[1], 2));
  result.clear();
  result.resize(3);
  Store(&result[1], Vec2d(1.0, 2.0));
  EXPECT_SPAN_EQ((std::array<double, 2>{1.0, 2.0}), Span(&result[1], 2));

  result.clear();
  result.resize(3);
  Store(&result[1], Vec2d(1.0, 2.0), 0);
  EXPECT_SPAN_EQ((std::array<double, 2>{0.0, 0.0}), Span(&result[1], 2));
  Store(&result[1], Vec2d(1.0, 2.0), 1);
  EXPECT_SPAN_EQ((std::array<double, 2>{1.0, 0.0}), Span(&result[1], 2));
  Store(&result[1], Vec2d(1.0, 2.0), 2);
  EXPECT_SPAN_EQ((std::array<double, 2>{1.0, 2.0}), Span(&result[1], 2));
}

TEST(Vec2d, StoreSelected) {
  Vec2d srcValues{1.0, 2.0};
  std::array<int64_t, 2> condition{};
  std::array<double, 2> expectedValues{};
  alignas(16) std::array<double, 4> dstBuffer{};
  dstBuffer[0] = 123;
  dstBuffer[3] = 456;
  Span<double> dstValues{&dstBuffer[1], 2}; // Not aligned
  for (int mask = 0; mask < 4; ++mask) { // For each possible set of conditions
    int expectedCount = 0;
    for (int i = 0; i < 2; ++i) {
      condition[i] = (mask & (1 << i)) ? -1ull : 0ull; // true = -1, false = 0
      if (condition[i]) {
        expectedValues[expectedCount++] = srcValues[i];
      }
    }
    auto countStored = StoreSelected(dstValues.data(), Load<Vec2l>(condition.data()), srcValues);
    EXPECT_EQ(expectedCount, countStored);
    for (int i = 0; i < countStored; ++i) {
      EXPECT_EQ(expectedValues[i], dstValues[i]);
    }
  }
  // Sentinel values should remain unchanged
  EXPECT_EQ(123, dstBuffer[0]);
  EXPECT_EQ(456, dstBuffer[3]);
}

TEST(Vec2d, StoreTransposed) {
  alignas(alignof(Vec2d)) double result[8] = {};
  result[7] = 911.0; // Sentinel value
  auto* ptr = result + 1; // Not an aligned address
  Vec2d data[] = {Vec2d{1.0, 4.0}, Vec2d{2.0, 5.0}, Vec2d{3.0, 6.0}};
  StoreTransposed<1>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<double, 6>{1.0, 2.0, 3.0, 0.0, 0.0, 0.0}), Span(ptr, 6));
  StoreTransposed<2>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<double, 6>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), Span(ptr, 6));
  StoreTransposed(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<double, 6>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), Span(ptr, 6));
  EXPECT_EQ(0.0, result[0]); // No change
  EXPECT_EQ(911.0, result[7]); // No change
}

// clang-format off
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec2d, Sub, -, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec2d, Tan, ([](auto a) { return std::tan(a); }), -1.0, 1.0, kEps);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec2d, BitwiseXOR, ^);

MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, Exp, ([](auto a) { return std::exp(a); }), kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec2d, Ln, ([](auto a) { return std::log(a); }), 0.1, 10.0, kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec2d, Tanh, ([](auto a) { return std::tanh(a); }), kEps);
// clang-format on

TEST(Vec2d, ExpExtreme) {
  // Keep non-negative entries in array 'x'
  std::array<double, 2> x({567.8, 700.1});
  Vec2d v(x[0], x[1]);
  auto expv = Exp(v), expmv = Exp(-v);
  auto tol = double(2.0) * std::numeric_limits<double>::epsilon();
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR_RTOL(Vec2d::Get(expv, i), std::exp(x[i]), tol);
    EXPECT_LE(
        Abs(Vec2d::Get(expmv, i) - std::exp(-x[i])),
        Max(Vec2d::Get(expmv, i), std::exp(-x[i])) * tol);
  }
}
