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

#include <array>
#include <type_traits>
#include <vector>

using namespace mochi;
using namespace mochi::simd_test;

static_assert(std::is_trivially_copyable_v<Simd<int64_t, 4>>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec4l);

#define EXPECT_VEC4L(e0, e1, e2, e3, actual) \
  {                                          \
    Simd<int64_t, 4> vActual = actual;       \
    EXPECT_EQ((e0), vActual[0]);             \
    EXPECT_EQ((e1), vActual[1]);             \
    EXPECT_EQ((e2), vActual[2]);             \
    EXPECT_EQ((e3), vActual[3]);             \
  }

/***********************************************************************************************
  Vec4l Class
*/

TEST(Vec4l, Class) {
  static_assert(Vec4l::kIsSupported, "Should be supported");

#if MOCHI_USE_SIMD
  static_assert(Vec4l::kIsComposite == MOCHI_ARCH_ARM_NEON);
  static_assert(!Vec4l::kIsEmulated);
#else
  static_assert(!Vec4l::kIsComposite);
  static_assert(Vec4l::kIsEmulated);
#endif

  static_assert(sizeof(Vec4l) == sizeof(int64_t) * 4);
  static_assert(alignof(Vec4l) == alignof(typename Vec4l::NativeType));
  static_assert(std::is_same_v<Vec4l::Scalar, int64_t>);
  static_assert(Vec4l::kSize == 4);
  static_assert(Vec4l::size() == 4);

  // Construct from broadcast
  EXPECT_VEC4L(1, 1, 1, 1, Vec4l(1));

  // Construct from scalars
  EXPECT_VEC4L(1, 1, 1, 1, Vec4l(1, 1, 1, 1));
  EXPECT_VEC4L(1, 2, 3, 4, Vec4l(1, 2, 3, 4));

  // Implicit conversion from scalar
  Vec4l a = 2;
  EXPECT_VEC4L(2, 2, 2, 2, a);
  a = 3;
  EXPECT_VEC4L(3, 3, 3, 3, a);

  // Copy construct
  Vec4l b{a};
  EXPECT_VEC4L(3, 3, 3, 3, b);

  // Copy assign
  a = Vec4l{4};
  b = a;
  EXPECT_VEC4L(4, 4, 4, 4, b);

  // Comparison
  EXPECT_EQ(true, (a == b));
  EXPECT_EQ(true, (a != Vec4l{}));
  EXPECT_EQ(false, (a != b));
  EXPECT_EQ(false, (a == Vec4l{}));

  // Unary operators
  Vec4l ones{-1};
  Vec4l zeros{0};
  EXPECT_EQ(ones, ~zeros);
  EXPECT_EQ(zeros, ~ones);
  EXPECT_VEC4L(-1, -2, -3, -4, -Vec4l(1, 2, 3, 4));

  // Binary operators
  a = Vec4l{1, 2, 3, 4};
  b = Vec4l{5, 6, 7, 8};
  EXPECT_VEC4L(6, 8, 10, 12, a + b);
  EXPECT_VEC4L(4, 5, 6, 7, a + int64_t(3));
  EXPECT_VEC4L(4, 5, 6, 7, int64_t(3) + a);
  EXPECT_VEC4L(-4, -4, -4, -4, a - b);
  EXPECT_VEC4L(-2, -1, 0, 1, a - 3);
  EXPECT_VEC4L(2, 1, 0, -1, int64_t(3) - a);
  EXPECT_VEC4L(5, 12, 21, 32, a * b);
  EXPECT_VEC4L(3, 6, 9, 12, a * 3);
  EXPECT_VEC4L(3, 6, 9, 12, int64_t(3) * a);
  EXPECT_VEC4L(5, 3, 2, 2, b / a);
  EXPECT_VEC4L(0, 0, 1, 1, a / 3);
  EXPECT_VEC4L(3, 1, 1, 0, int64_t(3) / a);
  EXPECT_EQ(a, a & ones);
  EXPECT_EQ(zeros, a & zeros);
  EXPECT_EQ(ones, Vec4l(a | ones));
  EXPECT_EQ(a, a | zeros);
  EXPECT_EQ(~a, a ^ ones);
  EXPECT_EQ(a, a ^ zeros);

  // Update operators
  a = Vec4l{1, 2, -3, -4};
  a += b;
  EXPECT_VEC4L(6, 8, 4, 4, a);
  a = Vec4l{1, 2, -4, -3};
  a += 3;
  EXPECT_VEC4L(4, 5, -1, 0, a);
  a = Vec4l{1, 2, -3, -4};
  a -= b;
  EXPECT_VEC4L(-4, -4, -10, -12, a);
  a = Vec4l{1, 2, -3, -4};
  a -= 3;
  EXPECT_VEC4L(-2, -1, -6, -7, a);
  a = Vec4l{1, 2, -3, -4};
  a *= b;
  EXPECT_VEC4L(5, 12, -21, -32, a);
  a = Vec4l{1, 2, -3, -4};
  a *= 3;
  EXPECT_VEC4L(3, 6, -9, -12, a);
  a = Vec4l{1, 2, -3, -4};
  b = Vec4l{5, 6, 7, 8};
  b /= a;
  EXPECT_VEC4L(5, 3, -2, -2, b);
  a = Vec4l{1, 2, -4, -3};
  a /= 3;
  EXPECT_VEC4L(0, 0, -1, -1, a);

  b = a;
  b &= ones;
  EXPECT_EQ(a, b);
  a = Vec4l{1, 2, 3, 4};
  a &= zeros;
  EXPECT_EQ(zeros, a);
  a = Vec4l{1, 2, 3, 4};
  a |= ones;
  EXPECT_EQ(ones, a);
  a = b = Vec4l{1, 2, 3, 4};
  b |= zeros;
  EXPECT_EQ(a, b);
  a = b = Vec4l{1, 2, 3, 4};
  b ^= ones;
  EXPECT_EQ(~a, b);
  a = b = Vec4l{1, 2, 3, 4};
  b ^= zeros;
  EXPECT_EQ(a, b);
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec4l, Add, +);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec4l, BitwiseAND, &);
MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(Vec4l);
MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec4l, Div, /);

TEST(Vec4l, AllTrue) {
  int64_t zeros = 0;
  int64_t ones = -1;
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          auto vec = Vec4l{a ? ones : zeros, b ? ones : zeros, c ? ones : zeros, d ? ones : zeros};
          EXPECT_EQ(!!a, AllTrue<1>(vec));
          EXPECT_EQ(a && b, AllTrue<2>(vec));
          EXPECT_EQ(a && b && c, AllTrue<3>(vec));
          EXPECT_EQ(a && b && c && d, AllTrue<4>(vec));
          EXPECT_EQ(a && b && c && d, AllTrue(vec));
        }
      }
    }
  }
}

TEST(Vec4l, Broadcast) {
  // Broadcast scalar
  EXPECT_VEC4L(1, 1, 1, 1, Broadcast<Vec4l>(1));

  // Broadcast from address
  int64_t const s = 2;
  EXPECT_VEC4L(2, 2, 2, 2, Broadcast<Vec4l>(&s));

  // Broadcast ith element
  auto v = Vec4l{1, 2, 3, 4};
  EXPECT_VEC4L(1, 1, 1, 1, Broadcast<0>(v));
  EXPECT_VEC4L(2, 2, 2, 2, Broadcast<1>(v));
  EXPECT_VEC4L(3, 3, 3, 3, Broadcast<2>(v));
  EXPECT_VEC4L(4, 4, 4, 4, Broadcast<3>(v));

  // Broadcast ith element
  EXPECT_VEC4L(1, 1, 1, 1, Broadcast(v, 0));
  EXPECT_VEC4L(2, 2, 2, 2, Broadcast(v, 1));
  EXPECT_VEC4L(3, 3, 3, 3, Broadcast(v, 2));
  EXPECT_VEC4L(4, 4, 4, 4, Broadcast(v, 3));
}

TEST(Vec4l, Get) {
  auto a = Vec4l{1, 2, 3, -4};

  // Fast template version
  EXPECT_EQ(1, Get0(a));
  EXPECT_EQ(1, Get<0>(a));
  EXPECT_EQ(2, Get<1>(a));
  EXPECT_EQ(3, Get<2>(a));
  EXPECT_EQ(-4, Get<3>(a));

  // Slower runtime version
  EXPECT_EQ(1, Get(a, 0));
  EXPECT_EQ(2, Get(a, 1));
  EXPECT_EQ(3, Get(a, 2));
  EXPECT_EQ(-4, Get(a, 3));

  // Same but with operator[] (read only)
  EXPECT_EQ(1, a[0]);
  EXPECT_EQ(2, a[1]);
  EXPECT_EQ(3, a[2]);
  EXPECT_EQ(-4, a[3]);
}

TEST(Vec4l, GetHalf) {
  auto a = Vec4l{1, 2, 3, 4};
  auto low = GetHalf<0>(a);
  auto high = GetHalf<1>(a);
  EXPECT_EQ(1, low[0]);
  EXPECT_EQ(2, low[1]);
  EXPECT_EQ(3, high[0]);
  EXPECT_EQ(4, high[1]);
}

TEST(Vec4l, Greater) {
  auto a = Vec4l{1, 4, 5, 9};
  auto b = Vec4l{2, 3, 7, 8};
  EXPECT_VEC4L(1, 0, 1, 0, (b > a) & Vec4l{1});
  EXPECT_VEC4L(0, 1, 0, 1, (a > b) & Vec4l{1});
  EXPECT_VEC4L(0, 0, 0, 0, (a > a) & Vec4l{1});
  EXPECT_VEC4L(0, 0, 1, 1, (a > Vec4l{4}) & Vec4l{1});
  EXPECT_VEC4L(0, 0, 0, 1, (a > Vec4l{5}) & Vec4l{1});
  EXPECT_VEC4L(1, 0, 0, 0, (Vec4l{4} > a) & Vec4l{1});
  EXPECT_VEC4L(1, 1, 0, 0, (Vec4l{5} > a) & Vec4l{1});
}

TEST(Vec4l, GreaterEqual) {
  auto a = Vec4l{1, 4, 5, 9};
  auto b = Vec4l{2, 3, 7, 8};
  EXPECT_VEC4L(1, 0, 1, 0, (b >= a) & Vec4l{1});
  EXPECT_VEC4L(0, 1, 0, 1, (a >= b) & Vec4l{1});
  EXPECT_VEC4L(1, 1, 1, 1, (a >= a) & Vec4l{1});
  EXPECT_VEC4L(0, 1, 1, 1, (a >= Vec4l{4}) & Vec4l{1});
  EXPECT_VEC4L(0, 0, 1, 1, (a >= Vec4l{5}) & Vec4l{1});
  EXPECT_VEC4L(1, 1, 0, 0, (Vec4l{4} >= a) & Vec4l{1});
  EXPECT_VEC4L(1, 1, 1, 0, (Vec4l{5} >= a) & Vec4l{1});
}

TEST(Vec4l, HMax) {
  auto a = Vec4l{1, 2, 3, 4};
  EXPECT_EQ(2, HMax<2>(a));
  EXPECT_EQ(3, HMax<3>(a));
  EXPECT_EQ(4, HMax(a));
}

TEST(Vec4l, HMin) {
  auto a = Vec4l{4, 3, 2, 1};
  EXPECT_EQ(3, HMin<2>(a));
  EXPECT_EQ(2, HMin<3>(a));
  EXPECT_EQ(1, HMin(a));
}

TEST(Vec4l, HSum) {
  auto a = Vec4l{1, 2, 3, 4};
  EXPECT_EQ(3, HSum<2>(a));
  EXPECT_EQ(6, HSum<3>(a));
  EXPECT_EQ(10, HSum(a));
}

TEST(Vec4l, Less) {
  auto a = Vec4l{1, 4, 5, 9};
  auto b = Vec4l{2, 3, 7, 8};
  EXPECT_VEC4L(0, 1, 0, 1, (b < a) & Vec4l{1});
  EXPECT_VEC4L(1, 0, 1, 0, (a < b) & Vec4l{1});
  EXPECT_VEC4L(0, 0, 0, 0, (a < a) & Vec4l{1});
  EXPECT_VEC4L(1, 0, 0, 0, (a < Vec4l{4}) & Vec4l{1});
  EXPECT_VEC4L(1, 1, 0, 0, (a < Vec4l{5}) & Vec4l{1});
  EXPECT_VEC4L(0, 0, 1, 1, (Vec4l{4} < a) & Vec4l{1});
  EXPECT_VEC4L(0, 0, 0, 1, (Vec4l{5} < a) & Vec4l{1});
}

TEST(Vec4l, LessEqual) {
  auto a = Vec4l{1, 4, 5, 9};
  auto b = Vec4l{2, 3, 7, 8};
  EXPECT_VEC4L(0, 1, 0, 1, (b <= a) & Vec4l{1});
  EXPECT_VEC4L(1, 0, 1, 0, (a <= b) & Vec4l{1});
  EXPECT_VEC4L(1, 1, 1, 1, (a <= a) & Vec4l{1});
  EXPECT_VEC4L(1, 1, 0, 0, (a <= Vec4l{4}) & Vec4l{1});
  EXPECT_VEC4L(1, 1, 1, 0, (a <= Vec4l{5}) & Vec4l{1});
  EXPECT_VEC4L(0, 1, 1, 1, (Vec4l{4} <= a) & Vec4l{1});
  EXPECT_VEC4L(0, 0, 1, 1, (Vec4l{5} <= a) & Vec4l{1});
}

// Test Equal. NotEqual<N> for N > 1 requires AnyTrue, which is not supported for Vec4l, so only
// NotEqual<1> can be tested via the helper.
TEST(Vec4l, Equal) {
  auto a = Vec4l{1, 2, 3, 4};
  auto b = a;
  auto c = Vec4l{9, 2, 3, 4};
  auto d = Vec4l{1, 9, 3, 4};
  auto e = Vec4l{1, 2, 9, 4};
  auto f = Vec4l{1, 2, 3, 9};

  // Compare 1 component
  ExpectEqual<1>(true, a, a);
  ExpectEqual<1>(true, a, b);
  ExpectEqual<1>(false, a, c);
  ExpectEqual<1>(true, a, d);
  ExpectEqual<1>(true, a, e);
  ExpectEqual<1>(true, a, f);

  // Compare 2 components
  ExpectEqual<2>(true, a, a);
  ExpectEqual<2>(true, a, b);
  ExpectEqual<2>(false, a, c);
  ExpectEqual<2>(false, a, d);
  ExpectEqual<2>(true, a, e);
  ExpectEqual<2>(true, a, f);

  // Compare 3 components
  ExpectEqual<3>(true, a, a);
  ExpectEqual<3>(true, a, b);
  ExpectEqual<3>(false, a, c);
  ExpectEqual<3>(false, a, d);
  ExpectEqual<3>(false, a, e);
  ExpectEqual<3>(true, a, f);

  // Compare 4 components
  ExpectEqual<4>(true, a, a);
  ExpectEqual<4>(true, a, b);
  ExpectEqual<4>(false, a, c);
  ExpectEqual<4>(false, a, d);
  ExpectEqual<4>(false, a, e);
  ExpectEqual<4>(false, a, f);
}

template <class V>
static void ExpectVEqual(int64_t e0, int64_t e1, int64_t e2, int64_t e3, V a, V b) {
  EXPECT_VEC4L(e0, e1, e2, e3, VEqual(a, b) & Vec4l{1});
  EXPECT_VEC4L(int64_t(!e0), int64_t(!e1), int64_t(!e2), int64_t(!e3), VNotEqual(a, b) & Vec4l{1});
}

TEST(Vec4l, VEqual) {
  auto a = Vec4l{1, 2, 3, 4};
  auto b = Vec4l{1, 9, 3, 9};
  auto c = Vec4l{9, 2, 9, 4};
  ExpectVEqual(1, 1, 1, 1, a, a);
  ExpectVEqual(1, 0, 1, 0, a, b);
  ExpectVEqual(0, 1, 0, 1, a, c);
}

TEST(Vec4l, Load) {
  alignas(alignof(Vec4l)) int64_t const values[] = {0, 1, 2, 3, 4};
  EXPECT_VEC4L(0, 0, 0, 0, (Load<0, Vec4l>(nullptr)));
  EXPECT_VEC4L(1, 0, 0, 0, (Load<1, Vec4l>(values + 1)));
  EXPECT_VEC4L(1, 2, 0, 0, (Load<2, Vec4l>(values + 1)));
  EXPECT_VEC4L(1, 2, 3, 0, (Load<3, Vec4l>(values + 1)));
  EXPECT_VEC4L(1, 2, 3, 4, (Load<4, Vec4l>(values + 1)));
  EXPECT_VEC4L(1, 2, 3, 4, (Load<Vec4l>(values + 1)));

  EXPECT_VEC4L(1, 0, 0, 0, (Load<Vec4l>(values + 1, 1)));
  EXPECT_VEC4L(1, 2, 0, 0, (Load<Vec4l>(values + 1, 2)));
  EXPECT_VEC4L(1, 2, 3, 0, (Load<Vec4l>(values + 1, 3)));
  EXPECT_VEC4L(1, 2, 3, 4, (Load<Vec4l>(values + 1, 4)));
}

TEST(Vec4l, LoadTransposed) {
  alignas(alignof(Vec4l)) int64_t const values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto const* ptr = values + 1; // Not an aligned address
  Vec4l loaded[3] = {};
  LoadTransposed<1>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4L(1, 0, 0, 0, loaded[0]);
  EXPECT_VEC4L(2, 0, 0, 0, loaded[1]);
  EXPECT_VEC4L(3, 0, 0, 0, loaded[2]);
  LoadTransposed<2>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4L(1, 4, 0, 0, loaded[0]);
  EXPECT_VEC4L(2, 5, 0, 0, loaded[1]);
  EXPECT_VEC4L(3, 6, 0, 0, loaded[2]);
  LoadTransposed<3>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4L(1, 4, 7, 0, loaded[0]);
  EXPECT_VEC4L(2, 5, 8, 0, loaded[1]);
  EXPECT_VEC4L(3, 6, 9, 0, loaded[2]);
  LoadTransposed<4>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4L(1, 4, 7, 10, loaded[0]);
  EXPECT_VEC4L(2, 5, 8, 11, loaded[1]);
  EXPECT_VEC4L(3, 6, 9, 12, loaded[2]);
  LoadTransposed(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4L(1, 4, 7, 10, loaded[0]);
  EXPECT_VEC4L(2, 5, 8, 11, loaded[1]);
  EXPECT_VEC4L(3, 6, 9, 12, loaded[2]);
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec4l, Mul, *);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec4l, Neg, -);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec4l, BitwiseNOT, ~);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec4l, BitwiseOR, |);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec4l, LogicalAND, &&, &);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec4l, LogicalOR, ||, |);

TEST(Vec4l, Select) {
  auto a = Vec4l{1, 2, 3, 4};
  auto b = Vec4l{5, 6, 7, 8};
  EXPECT_VEC4L(5, 6, 7, 8, Select(Vec4l{0, 0, 0, 0}, a, b));
  EXPECT_VEC4L(1, 6, 7, 8, Select(Vec4l{-1, 0, 0, 0}, a, b));
  EXPECT_VEC4L(5, 2, 7, 8, Select(Vec4l{0, -1, 0, 0}, a, b));
  EXPECT_VEC4L(1, 2, 7, 8, Select(Vec4l{-1, -1, 0, 0}, a, b));
  EXPECT_VEC4L(5, 6, 7, 4, Select(Vec4l{0, 0, 0, -1}, a, b));
  EXPECT_VEC4L(1, 6, 7, 4, Select(Vec4l{-1, 0, 0, -1}, a, b));
  EXPECT_VEC4L(5, 2, 7, 4, Select(Vec4l{0, -1, 0, -1}, a, b));
  EXPECT_VEC4L(1, 2, 7, 4, Select(Vec4l{-1, -1, 0, -1}, a, b));
  EXPECT_VEC4L(5, 6, 3, 8, Select(Vec4l{0, 0, -1, 0}, a, b));
  EXPECT_VEC4L(1, 6, 3, 8, Select(Vec4l{-1, 0, -1, 0}, a, b));
  EXPECT_VEC4L(5, 2, 3, 8, Select(Vec4l{0, -1, -1, 0}, a, b));
  EXPECT_VEC4L(1, 2, 3, 8, Select(Vec4l{-1, -1, -1, 0}, a, b));
  EXPECT_VEC4L(5, 6, 3, 4, Select(Vec4l{0, 0, -1, -1}, a, b));
  EXPECT_VEC4L(1, 6, 3, 4, Select(Vec4l{-1, 0, -1, -1}, a, b));
  EXPECT_VEC4L(5, 2, 3, 4, Select(Vec4l{0, -1, -1, -1}, a, b));
  EXPECT_VEC4L(1, 2, 3, 4, Select(Vec4l{-1, -1, -1, -1}, a, b));
}

TEST(Vec4l, Sequence) {
  EXPECT_VEC4L(0, 1, 2, 3, Sequence<Vec4l>());
}

TEST(Vec4l, Shuffle) {
  auto a = Vec4l{1, 2, 3, 4};
  auto b = Vec4l{5, 6, 7, 8};

  // Shuffle single vector
  EXPECT_VEC4L(1, 1, 1, 1, (Shuffle<0, 0, 0, 0>(a)));
  EXPECT_VEC4L(2, 2, 2, 2, (Shuffle<1, 1, 1, 1>(a)));
  EXPECT_VEC4L(3, 3, 3, 3, (Shuffle<2, 2, 2, 2>(a)));
  EXPECT_VEC4L(4, 4, 4, 4, (Shuffle<3, 3, 3, 3>(a)));
  EXPECT_VEC4L(1, 2, 3, 4, (Shuffle<0, 1, 2, 3>(a)));
  EXPECT_VEC4L(4, 3, 2, 1, (Shuffle<3, 2, 1, 0>(a)));
  EXPECT_VEC4L(1, 2, 1, 2, (Shuffle<0, 1, 0, 1>(a)));
  EXPECT_VEC4L(4, 3, 4, 3, (Shuffle<3, 2, 3, 2>(a)));

  // Shuffle two vectors
  EXPECT_VEC4L(1, 2, 5, 6, (Shuffle<0, 1, 0, 1>(a, b)));
  EXPECT_VEC4L(1, 2, 7, 8, (Shuffle<0, 1, 2, 3>(a, b)));
  EXPECT_VEC4L(1, 1, 5, 5, (Shuffle<0, 0, 0, 0>(a, b)));
  EXPECT_VEC4L(2, 2, 6, 6, (Shuffle<1, 1, 1, 1>(a, b)));
}

TEST(Vec4l, SimdZero) {
  EXPECT_VEC4L(0, 0, 0, 0, SimdZero<Vec4l>());
}

MOCHI_SIMD_TEST_UNARY_FN_EXACT(Vec4l, Sqr, ([](auto a) { return a * a; }));
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec4l, Max, ([](auto a, auto b) { return std::max(a, b); }));
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec4l, Min, ([](auto a, auto b) { return std::min(a, b); }));

TEST(Vec4l, Store) {
  std::vector<int64_t> result(
      5); // NOTE: Changed from an array on the stack to work around an MSVC optimizer bug.
  auto const v = Vec4l{1, 2, 3, 4};
  Store<0>((int64_t*)nullptr, v);
  Store<0>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 4>{0, 0, 0, 0}), Span(&result[1], 4));
  Store<1>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 4>{1, 0, 0, 0}), Span(&result[1], 4));
  Store<2>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 4>{1, 2, 0, 0}), Span(&result[1], 4));
  Store<3>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 4>{1, 2, 3, 0}), Span(&result[1], 4));
  Store<4>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 4>{1, 2, 3, 4}), Span(&result[1], 4));
  result.clear();
  result.resize(5);
  Store(&result[1], v, 0);
  EXPECT_SPAN_EQ((std::array<int64_t, 4>{0, 0, 0, 0}), Span(&result[1], 4));
  Store(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 4>{1, 2, 3, 4}), Span(&result[1], 4));
}

TEST(Vec4l, StoreSelected) {
  Vec4l srcValues{1, 2, 3, 4};
  std::array<int64_t, 4> condition{};
  std::array<int64_t, 4> expectedValues{};
  alignas(16) std::array<int64_t, 6> dstBuffer{};
  dstBuffer[0] = 123;
  dstBuffer[5] = 456;
  Span<int64_t> dstValues{&dstBuffer[1], 4}; // Not aligned
  for (int mask = 0; mask < 16; ++mask) { // For each possible set of conditions
    int expectedCount = 0;
    for (int i = 0; i < 4; ++i) {
      condition[i] = (mask & (1 << i)) ? int64_t(-1) : int64_t(0); // true = -1, false = 0
      if (condition[i]) {
        expectedValues[expectedCount++] = srcValues[i];
      }
    }
    auto countStored = StoreSelected(dstValues.data(), Load<Vec4l>(condition.data()), srcValues);
    EXPECT_EQ(expectedCount, countStored);
    for (int i = 0; i < countStored; ++i) {
      EXPECT_EQ(expectedValues[i], dstValues[i]);
    }
  }
  // Sentinel values should remain unchanged
  EXPECT_EQ(123, dstBuffer[0]);
  EXPECT_EQ(456, dstBuffer[5]);
}

TEST(Vec4l, StoreTransposed) {
  alignas(alignof(Vec4l)) int64_t result[14] = {};
  result[13] = 911; // Sentinel value
  auto* ptr = result + 1; // Not an aligned address
  Vec4l data[] = {Vec4l{1, 4, 7, 10}, Vec4l{2, 5, 8, 11}, Vec4l{3, 6, 9, 12}};
  StoreTransposed<1>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int64_t, 12>{1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}), Span(ptr, 12));
  StoreTransposed<2>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int64_t, 12>{1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}), Span(ptr, 12));
  StoreTransposed<3>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int64_t, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0}), Span(ptr, 12));
  StoreTransposed<4>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int64_t, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), Span(ptr, 12));
  StoreTransposed(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int64_t, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), Span(ptr, 12));
  EXPECT_EQ(0, result[0]); // No change
  EXPECT_EQ(911, result[13]); // No change
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec4l, Sub, -);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec4l, BitwiseXOR, ^);

TEST(Vec4l, ShiftLeft) {
  Simd<int64_t, 4> v(1, 5, 17, 23);
  EXPECT_VEC4L(8, 40, 136, 184, v << 3);
  EXPECT_EQ(v, v << 0);
}

TEST(Vec4l, ShiftRight) {
  Vec4l v(8, 40, 56, 88);
  EXPECT_VEC4L(1, 5, 7, 11, ShiftRight<3>(v));
  EXPECT_EQ(v, ShiftRight<0>(v));
}
