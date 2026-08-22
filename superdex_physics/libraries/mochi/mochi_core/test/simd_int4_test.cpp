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

static_assert(std::is_trivially_copyable_v<Simd<int, 4>>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec4i);

#define EXPECT_VEC4I(e0, e1, e2, e3, actual) \
  {                                          \
    Vec4i vActual = actual;                  \
    EXPECT_EQ((e0), vActual[0]);             \
    EXPECT_EQ((e1), vActual[1]);             \
    EXPECT_EQ((e2), vActual[2]);             \
    EXPECT_EQ((e3), vActual[3]);             \
  }

/***********************************************************************************************
  Vec4i Class
*/

TEST(Vec4i, Class) {
  static_assert(Vec4i::kIsSupported);
  static_assert(!Vec4i::kIsComposite);
  static_assert(Vec4i::kIsEmulated == !(MOCHI_USE_SIMD));
  static_assert(sizeof(Vec4i) == sizeof(int) * 4);
  static_assert(alignof(Vec4i) == alignof(typename Vec4i::NativeType));
  static_assert(std::is_same_v<Vec4i::Scalar, int>);
  static_assert(Vec4i::kSize == 4);
  static_assert(Vec4i::size() == 4);

  // Construct from broadcast
  EXPECT_VEC4I(1, 1, 1, 1, Vec4i(1));

  // Construct from scalars
  EXPECT_VEC4I(1, 1, 1, 1, Vec4i(1));
  EXPECT_VEC4I(1, 2, 0, 0, Vec4i(1, 2));
  EXPECT_VEC4I(1, 2, 3, 0, Vec4i(1, 2, 3));
  EXPECT_VEC4I(1, 2, 3, 4, Vec4i(1, 2, 3, 4));

  // Implicit conversion from scalar
  Vec4i a = 2;
  EXPECT_VEC4I(2, 2, 2, 2, a);
  a = 3;
  EXPECT_VEC4I(3, 3, 3, 3, a);

  // Copy construct
  Vec4i b{a};
  EXPECT_VEC4I(3, 3, 3, 3, b);

  // Copy assign
  a = Vec4i{4};
  b = a;
  EXPECT_VEC4I(4, 4, 4, 4, b);

  // Comparison
  EXPECT_EQ(true, (a == b));
  EXPECT_EQ(true, (a != Vec4i{}));
  EXPECT_EQ(false, (a != b));
  EXPECT_EQ(false, (a == Vec4i{}));

  // Unary operators
  Vec4i ones{-1};
  Vec4i zeros{0};
  EXPECT_EQ(ones, ~zeros);
  EXPECT_EQ(zeros, ~ones);
  EXPECT_VEC4I(-1, -2, -3, -4, -Vec4i(1, 2, 3, 4));

  // Binary operators
  a = Vec4i{1, 2, 3, 4};
  b = Vec4i{5, 6, 7, 8};
  EXPECT_VEC4I(6, 8, 10, 12, a + b);
  EXPECT_VEC4I(4, 5, 6, 7, a + 3);
  EXPECT_VEC4I(4, 5, 6, 7, 3 + a);
  EXPECT_VEC4I(-4, -4, -4, -4, a - b);
  EXPECT_VEC4I(-2, -1, 0, 1, a - 3);
  EXPECT_VEC4I(2, 1, 0, -1, 3 - a);
  EXPECT_VEC4I(5, 12, 21, 32, a * b);
  EXPECT_VEC4I(3, 6, 9, 12, a * 3);
  EXPECT_VEC4I(3, 6, 9, 12, 3 * a);
  EXPECT_VEC4I(5, 3, 2, 2, b / a);
  EXPECT_VEC4I(0, 0, 1, 1, a / 3);
  EXPECT_VEC4I(3, 1, 1, 0, 3 / a);
  EXPECT_EQ(a, a & ones);
  EXPECT_EQ(zeros, a & zeros);
  EXPECT_EQ(ones, Vec4i(a | ones));
  EXPECT_EQ(a, a | zeros);
  EXPECT_EQ(~a, a ^ ones);
  EXPECT_EQ(a, a ^ zeros);

  // Update operators
  a = Vec4i{1, 2, 3, 4};
  a += b;
  EXPECT_VEC4I(6, 8, 10, 12, a);
  a = Vec4i{1, 2, 3, 4};
  a += 3;
  EXPECT_VEC4I(4, 5, 6, 7, a);
  a = Vec4i{1, 2, 3, 4};
  a -= b;
  EXPECT_VEC4I(-4, -4, -4, -4, a);
  a = Vec4i{1, 2, 3, 4};
  a -= 3;
  EXPECT_VEC4I(-2, -1, 0, 1, a);
  a = Vec4i{1, 2, 3, 4};
  a *= b;
  EXPECT_VEC4I(5, 12, 21, 32, a);
  a = Vec4i{1, 2, 3, 4};
  a *= 3;
  EXPECT_VEC4I(3, 6, 9, 12, a);
  a = Vec4i{1, 2, 3, 4};
  b = Vec4i{5, 6, 7, 8};
  b /= a;
  EXPECT_VEC4I(5, 3, 2, 2, b);
  a = Vec4i{1, 2, 3, 4};
  a /= 3;
  EXPECT_VEC4I(0, 0, 1, 1, a);
  b = a;
  b &= ones;
  EXPECT_EQ(a, b);
  a = Vec4i{1, 2, 3, 4};
  a &= zeros;
  EXPECT_EQ(zeros, a);
  a = Vec4i{1, 2, 3, 4};
  a |= ones;
  EXPECT_EQ(ones, a);
  a = b = Vec4i{1, 2, 3, 4};
  b |= zeros;
  EXPECT_EQ(a, b);
  a = b = Vec4i{1, 2, 3, 4};
  b ^= ones;
  EXPECT_EQ(~a, b);
  a = b = Vec4i{1, 2, 3, 4};
  b ^= zeros;
  EXPECT_EQ(a, b);
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec4i, Add, +);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec4i, BitwiseAND, &);
MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(Vec4i);

TEST(Vec4i, Blend) {
  auto a = Vec4i{1, 2, 3, 4};
  auto b = Vec4i{5, 6, 7, 8};
  EXPECT_VEC4I(1, 2, 3, 4, (Blend<0, 0, 0, 0>(a, b)));
  EXPECT_VEC4I(1, 2, 3, 8, (Blend<0, 0, 0, 1>(a, b)));
  EXPECT_VEC4I(1, 2, 7, 4, (Blend<0, 0, 1, 0>(a, b)));
  EXPECT_VEC4I(1, 2, 7, 8, (Blend<0, 0, 1, 1>(a, b)));
  EXPECT_VEC4I(1, 6, 3, 4, (Blend<0, 1, 0, 0>(a, b)));
  EXPECT_VEC4I(1, 6, 3, 8, (Blend<0, 1, 0, 1>(a, b)));
  EXPECT_VEC4I(1, 6, 7, 4, (Blend<0, 1, 1, 0>(a, b)));
  EXPECT_VEC4I(1, 6, 7, 8, (Blend<0, 1, 1, 1>(a, b)));
  EXPECT_VEC4I(5, 2, 3, 4, (Blend<1, 0, 0, 0>(a, b)));
  EXPECT_VEC4I(5, 2, 3, 8, (Blend<1, 0, 0, 1>(a, b)));
  EXPECT_VEC4I(5, 2, 7, 4, (Blend<1, 0, 1, 0>(a, b)));
  EXPECT_VEC4I(5, 2, 7, 8, (Blend<1, 0, 1, 1>(a, b)));
  EXPECT_VEC4I(5, 6, 3, 4, (Blend<1, 1, 0, 0>(a, b)));
  EXPECT_VEC4I(5, 6, 3, 8, (Blend<1, 1, 0, 1>(a, b)));
  EXPECT_VEC4I(5, 6, 7, 4, (Blend<1, 1, 1, 0>(a, b)));
  EXPECT_VEC4I(5, 6, 7, 8, (Blend<1, 1, 1, 1>(a, b)));
}

TEST(Vec4i, AllTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  int zeros = 0;
  int ones = -1;
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          auto vec = Vec4i{a ? ones : zeros, b ? ones : zeros, c ? ones : zeros, d ? ones : zeros};
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

TEST(Vec4i, AnyTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  int zeros = 0;
  int ones = -1;
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          auto vec = Vec4i{a ? ones : zeros, b ? ones : zeros, c ? ones : zeros, d ? ones : zeros};
          EXPECT_EQ(!!a, AnyTrue<1>(vec));
          EXPECT_EQ(a || b, AnyTrue<2>(vec));
          EXPECT_EQ(a || b || c, AnyTrue<3>(vec));
          EXPECT_EQ(a || b || c || d, AnyTrue<4>(vec));
          EXPECT_EQ(a || b || c || d, AnyTrue(vec));
        }
      }
    }
  }
}

TEST(Vec4i, IsTrue) {
  auto a = SimdMask<Vec4i>(true, false, true, false);
  EXPECT_TRUE(IsTrue<0>(a));
  EXPECT_FALSE(IsTrue<1>(a));
  EXPECT_TRUE(IsTrue<2>(a));
  EXPECT_FALSE(IsTrue<3>(a));
  EXPECT_FALSE(IsTrue<0>(~a));
  EXPECT_TRUE(IsTrue<1>(~a));
  EXPECT_FALSE(IsTrue<2>(~a));
  EXPECT_TRUE(IsTrue<3>(~a));
}

TEST(Vec4i, Broadcast) {
  // Broadcast scalar
  EXPECT_VEC4I(1, 1, 1, 1, Broadcast<Vec4i>(1));

  // Broadcast from address
  int const s = 2;
  EXPECT_VEC4I(2, 2, 2, 2, Broadcast<Vec4i>(&s));

  // Broadcast ith element
  auto v = Vec4i{1, 2, 3, 4};
  EXPECT_VEC4I(1, 1, 1, 1, Broadcast<0>(v));
  EXPECT_VEC4I(2, 2, 2, 2, Broadcast<1>(v));
  EXPECT_VEC4I(3, 3, 3, 3, Broadcast<2>(v));
  EXPECT_VEC4I(4, 4, 4, 4, Broadcast<3>(v));

  // Broadcast ith element
  EXPECT_VEC4I(1, 1, 1, 1, Broadcast(v, 0));
  EXPECT_VEC4I(2, 2, 2, 2, Broadcast(v, 1));
  EXPECT_VEC4I(3, 3, 3, 3, Broadcast(v, 2));
  EXPECT_VEC4I(4, 4, 4, 4, Broadcast(v, 3));
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec4i, Div, /);

TEST(Vec4i, Get) {
  auto a = Vec4i{1, 2, 3, 4};

  // Fast template version
  EXPECT_EQ(1, Get0(a));
  EXPECT_EQ(1, Get<0>(a));
  EXPECT_EQ(2, Get<1>(a));
  EXPECT_EQ(3, Get<2>(a));
  EXPECT_EQ(4, Get<3>(a));

  // Slower runtime version
  EXPECT_EQ(1, Get(a, 0));
  EXPECT_EQ(2, Get(a, 1));
  EXPECT_EQ(3, Get(a, 2));
  EXPECT_EQ(4, Get(a, 3));

  // Same but with operator[] (read only)
  EXPECT_EQ(1, a[0]);
  EXPECT_EQ(2, a[1]);
  EXPECT_EQ(3, a[2]);
  EXPECT_EQ(4, a[3]);
}

TEST(Vec4i, Set) {
  auto a = Vec4i{1, 2, 3, 4};
  EXPECT_VEC4I(9, 2, 3, 4, Set<0>(a, 9));
  EXPECT_VEC4I(1, 9, 3, 4, Set<1>(a, 9));
  EXPECT_VEC4I(1, 2, 9, 4, Set<2>(a, 9));
  EXPECT_VEC4I(1, 2, 3, 9, Set<3>(a, 9));

  EXPECT_VEC4I(9, 2, 3, 4, Set(a, 0, 9));
  EXPECT_VEC4I(1, 9, 3, 4, Set(a, 1, 9));
  EXPECT_VEC4I(1, 2, 9, 4, Set(a, 2, 9));
  EXPECT_VEC4I(1, 2, 3, 9, Set(a, 3, 9));
}

TEST(Vec4i, Sequence) {
  EXPECT_VEC4I(0, 1, 2, 3, Sequence<Vec4i>());
}

TEST(Vec4i, Greater) {
  auto a = Vec4i{1, 4, 5, 8};
  auto b = Vec4i{2, 3, 6, 7};
  EXPECT_VEC4I(1, 0, 1, 0, (b > a) & Vec4i{1});
  EXPECT_VEC4I(0, 1, 0, 1, (a > b) & Vec4i{1});
  EXPECT_VEC4I(0, 0, 0, 0, (a > a) & Vec4i{1});
  EXPECT_VEC4I(0, 0, 1, 1, (a > Vec4i{4}) & Vec4i{1});
  EXPECT_VEC4I(0, 0, 0, 1, (a > Vec4i{5}) & Vec4i{1});
  EXPECT_VEC4I(1, 0, 0, 0, (Vec4i{4} > a) & Vec4i{1});
  EXPECT_VEC4I(1, 1, 0, 0, (Vec4i{5} > a) & Vec4i{1});
}

TEST(Vec4i, GreaterEqual) {
  auto a = Vec4i{1, 4, 5, 8};
  auto b = Vec4i{2, 3, 6, 7};
  EXPECT_VEC4I(1, 0, 1, 0, (b >= a) & Vec4i{1});
  EXPECT_VEC4I(0, 1, 0, 1, (a >= b) & Vec4i{1});
  EXPECT_VEC4I(1, 1, 1, 1, (a >= a) & Vec4i{1});
  EXPECT_VEC4I(0, 1, 1, 1, (a >= Vec4i{4}) & Vec4i{1});
  EXPECT_VEC4I(0, 0, 1, 1, (a >= Vec4i{5}) & Vec4i{1});
  EXPECT_VEC4I(1, 1, 0, 0, (Vec4i{4} >= a) & Vec4i{1});
  EXPECT_VEC4I(1, 1, 1, 0, (Vec4i{5} >= a) & Vec4i{1});
}

TEST(Vec4i, HMax) {
  auto a = Vec4i{1, 2, 3, 4};
  EXPECT_EQ(2, HMax<2>(a));
  EXPECT_EQ(3, HMax<3>(a));
  EXPECT_EQ(4, HMax(a));
}

TEST(Vec4i, HMin) {
  auto a = Vec4i{4, 3, 2, 1};
  EXPECT_EQ(3, HMin<2>(a));
  EXPECT_EQ(2, HMin<3>(a));
  EXPECT_EQ(1, HMin(a));
}

TEST(Vec4i, HSum) {
  auto a = Vec4i{1, 2, 3, 4};
  EXPECT_EQ(3, HSum<2>(a));
  EXPECT_EQ(6, HSum<3>(a));
  EXPECT_EQ(10, HSum(a));
}

TEST(Vec4i, Less) {
  auto a = Vec4i{1, 4, 5, 8};
  auto b = Vec4i{2, 3, 6, 7};
  EXPECT_VEC4I(0, 1, 0, 1, (b < a) & Vec4i{1});
  EXPECT_VEC4I(1, 0, 1, 0, (a < b) & Vec4i{1});
  EXPECT_VEC4I(0, 0, 0, 0, (a < a) & Vec4i{1});
  EXPECT_VEC4I(1, 0, 0, 0, (a < Vec4i{4}) & Vec4i{1});
  EXPECT_VEC4I(1, 1, 0, 0, (a < Vec4i{5}) & Vec4i{1});
  EXPECT_VEC4I(0, 0, 1, 1, (Vec4i{4} < a) & Vec4i{1});
  EXPECT_VEC4I(0, 0, 0, 1, (Vec4i{5} < a) & Vec4i{1});
}

TEST(Vec4i, LessEqual) {
  auto a = Vec4i{1, 4, 5, 8};
  auto b = Vec4i{2, 3, 6, 7};
  EXPECT_VEC4I(0, 1, 0, 1, (b <= a) & Vec4i{1});
  EXPECT_VEC4I(1, 0, 1, 0, (a <= b) & Vec4i{1});
  EXPECT_VEC4I(1, 1, 1, 1, (a <= a) & Vec4i{1});
  EXPECT_VEC4I(1, 1, 0, 0, (a <= Vec4i{4}) & Vec4i{1});
  EXPECT_VEC4I(1, 1, 1, 0, (a <= Vec4i{5}) & Vec4i{1});
  EXPECT_VEC4I(0, 1, 1, 1, (Vec4i{4} <= a) & Vec4i{1});
  EXPECT_VEC4I(0, 0, 1, 1, (Vec4i{5} <= a) & Vec4i{1});
}

TEST(Vec4i, Equal) {
  auto a = Vec4i{1, 2, 3, 4};
  auto b = a;
  auto c = Vec4i{9, 2, 3, 4};
  auto d = Vec4i{1, 9, 3, 4};
  auto e = Vec4i{1, 2, 9, 4};
  auto f = Vec4i{1, 2, 3, 9};

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

// Test bot VEqual and VNotEqual
template <class V>
static void ExpectVEqual(int e0, int e1, int e2, int e3, V a, V b) {
  EXPECT_VEC4I(e0, e1, e2, e3, VEqual(a, b) & Vec4i{1});
  EXPECT_VEC4I(int(!e0), int(!e1), int(!e2), int(!e3), VNotEqual(a, b) & Vec4i{1});
}

TEST(Vec4i, VEqual) {
  auto a = Vec4i{1, 2, 3, 4};
  auto b = Vec4i{1, 9, 3, 9};
  auto c = Vec4i{9, 2, 9, 4};
  ExpectVEqual(1, 1, 1, 1, a, a);
  ExpectVEqual(1, 0, 1, 0, a, b);
  ExpectVEqual(0, 1, 0, 1, a, c);
}

TEST(Vec4i, Load) {
  alignas(alignof(Vec4i)) int const values[] = {0, 1, 2, 3, 4};
  EXPECT_VEC4I(0, 0, 0, 0, (Load<0, Vec4i>(nullptr)));
  EXPECT_VEC4I(1, 0, 0, 0, (Load<1, Vec4i>(values + 1)));
  EXPECT_VEC4I(1, 2, 0, 0, (Load<2, Vec4i>(values + 1)));
  EXPECT_VEC4I(1, 2, 3, 0, (Load<3, Vec4i>(values + 1)));
  EXPECT_VEC4I(1, 2, 3, 4, (Load<4, Vec4i>(values + 1)));
  EXPECT_VEC4I(1, 2, 3, 4, (Load<Vec4i>(values + 1)));

  EXPECT_VEC4I(1, 0, 0, 0, (Load<Vec4i>(values + 1, 1)));
  EXPECT_VEC4I(1, 2, 0, 0, (Load<Vec4i>(values + 1, 2)));
  EXPECT_VEC4I(1, 2, 3, 0, (Load<Vec4i>(values + 1, 3)));
  EXPECT_VEC4I(1, 2, 3, 4, (Load<Vec4i>(values + 1, 4)));
}

TEST(Vec4i, LoadTransposed) {
  alignas(alignof(Vec4i)) int const values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto const* ptr = values + 1; // Not an aligned address
  Vec4i loaded[3] = {};
  LoadTransposed<1>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4I(1, 0, 0, 0, loaded[0]);
  EXPECT_VEC4I(2, 0, 0, 0, loaded[1]);
  EXPECT_VEC4I(3, 0, 0, 0, loaded[2]);
  LoadTransposed<2>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4I(1, 4, 0, 0, loaded[0]);
  EXPECT_VEC4I(2, 5, 0, 0, loaded[1]);
  EXPECT_VEC4I(3, 6, 0, 0, loaded[2]);
  LoadTransposed<3>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4I(1, 4, 7, 0, loaded[0]);
  EXPECT_VEC4I(2, 5, 8, 0, loaded[1]);
  EXPECT_VEC4I(3, 6, 9, 0, loaded[2]);
  LoadTransposed<4>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4I(1, 4, 7, 10, loaded[0]);
  EXPECT_VEC4I(2, 5, 8, 11, loaded[1]);
  EXPECT_VEC4I(3, 6, 9, 12, loaded[2]);
  LoadTransposed(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4I(1, 4, 7, 10, loaded[0]);
  EXPECT_VEC4I(2, 5, 8, 11, loaded[1]);
  EXPECT_VEC4I(3, 6, 9, 12, loaded[2]);
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec4i, Mul, *);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec4i, Neg, -);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec4i, BitwiseNOT, ~);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec4i, BitwiseOR, |);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec4i, LogicalAND, &&, &);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec4i, LogicalOR, ||, |);

TEST(Vec4i, Select) {
  auto a = Vec4i{1, 2, 3, 4};
  auto b = Vec4i{5, 6, 7, 8};
  auto constexpr t = true;
  auto constexpr f = false;
  EXPECT_VEC4I(1, 2, 3, 4, Select(SimdMask<Vec4i>(t, t, t, t), a, b));
  EXPECT_VEC4I(1, 2, 3, 8, Select(SimdMask<Vec4i>(t, t, t, f), a, b));
  EXPECT_VEC4I(1, 2, 7, 4, Select(SimdMask<Vec4i>(t, t, f, t), a, b));
  EXPECT_VEC4I(1, 2, 7, 8, Select(SimdMask<Vec4i>(t, t, f, f), a, b));
  EXPECT_VEC4I(1, 6, 3, 4, Select(SimdMask<Vec4i>(t, f, t, t), a, b));
  EXPECT_VEC4I(1, 6, 3, 8, Select(SimdMask<Vec4i>(t, f, t, f), a, b));
  EXPECT_VEC4I(1, 6, 7, 4, Select(SimdMask<Vec4i>(t, f, f, t), a, b));
  EXPECT_VEC4I(1, 6, 7, 8, Select(SimdMask<Vec4i>(t, f, f, f), a, b));
  EXPECT_VEC4I(5, 2, 3, 4, Select(SimdMask<Vec4i>(f, t, t, t), a, b));
  EXPECT_VEC4I(5, 2, 3, 8, Select(SimdMask<Vec4i>(f, t, t, f), a, b));
  EXPECT_VEC4I(5, 2, 7, 4, Select(SimdMask<Vec4i>(f, t, f, t), a, b));
  EXPECT_VEC4I(5, 2, 7, 8, Select(SimdMask<Vec4i>(f, t, f, f), a, b));
  EXPECT_VEC4I(5, 6, 3, 4, Select(SimdMask<Vec4i>(f, f, t, t), a, b));
  EXPECT_VEC4I(5, 6, 3, 8, Select(SimdMask<Vec4i>(f, f, t, f), a, b));
  EXPECT_VEC4I(5, 6, 7, 4, Select(SimdMask<Vec4i>(f, f, f, t), a, b));
  EXPECT_VEC4I(5, 6, 7, 8, Select(SimdMask<Vec4i>(f, f, f, f), a, b));
  EXPECT_VEC4I(1, 2, 3, 4, Select(a < b, a, b)); // Min(a, b)
  EXPECT_VEC4I(5, 6, 7, 8, Select(a > b, a, b)); // Max(a, b)
  EXPECT_VEC4I(1, 2, 7, 8, Select(a <= Vec4i{2}, a, b));
  EXPECT_VEC4I(5, 6, 3, 4, Select(a > Vec4i{2}, a, b));
}

TEST(Vec4i, SimdMask) {
  auto v = Vec4i{1, 2, 3, 4};
  bool constexpr f = false;
  bool constexpr t = true;
  EXPECT_VEC4I(0, 0, 0, 0, v & SimdMask<Vec4i>(f, f, f, f));
  EXPECT_VEC4I(1, 0, 0, 0, v & SimdMask<Vec4i>(t, f, f, f));
  EXPECT_VEC4I(0, 2, 0, 0, v & SimdMask<Vec4i>(f, t, f, f));
  EXPECT_VEC4I(0, 0, 3, 0, v & SimdMask<Vec4i>(f, f, t, f));
  EXPECT_VEC4I(0, 0, 0, 4, v & SimdMask<Vec4i>(f, f, f, t));
  EXPECT_VEC4I(1, 0, 3, 0, v & SimdMask<Vec4i>(t, f, t, f));
  EXPECT_VEC4I(1, 2, 3, 4, v & SimdMask<Vec4i>(t, t, t, t));
}

TEST(Vec4i, SimdZero) {
  EXPECT_VEC4I(0, 0, 0, 0, SimdZero<Vec4i>());
}

TEST(Vec4i, Shuffle) {
  auto a = Vec4i{1, 2, 3, 4};
  auto b = Vec4i{5, 6, 7, 8};

  // Shuffle single vector
  EXPECT_VEC4I(1, 1, 1, 1, (Shuffle<0, 0, 0, 0>(a)));
  EXPECT_VEC4I(2, 2, 2, 2, (Shuffle<1, 1, 1, 1>(a)));
  EXPECT_VEC4I(3, 3, 3, 3, (Shuffle<2, 2, 2, 2>(a)));
  EXPECT_VEC4I(4, 4, 4, 4, (Shuffle<3, 3, 3, 3>(a)));
  EXPECT_VEC4I(1, 2, 3, 4, (Shuffle<0, 1, 2, 3>(a)));
  EXPECT_VEC4I(4, 3, 2, 1, (Shuffle<3, 2, 1, 0>(a)));
  EXPECT_VEC4I(1, 2, 1, 2, (Shuffle<0, 1, 0, 1>(a)));
  EXPECT_VEC4I(4, 3, 4, 3, (Shuffle<3, 2, 3, 2>(a)));

  // Shuffle two vectors
  EXPECT_VEC4I(1, 2, 5, 6, (Shuffle<0, 1, 0, 1>(a, b)));
  EXPECT_VEC4I(1, 2, 7, 8, (Shuffle<0, 1, 2, 3>(a, b)));
  EXPECT_VEC4I(1, 1, 5, 5, (Shuffle<0, 0, 0, 0>(a, b)));
  EXPECT_VEC4I(2, 2, 6, 6, (Shuffle<1, 1, 1, 1>(a, b)));
}

MOCHI_SIMD_TEST_UNARY_FN_EXACT(Vec4i, Sqr, ([](auto a) { return a * a; }));

TEST(Vec4i, Store) {
  std::vector<int> result(
      5); // NOTE: Changed from an array on the stack to work around an MSVC optimizer bug.
  auto const v = Vec4i{1, 2, 3, 4};
  Store<0>((int*)nullptr, v);
  Store<0>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 4>{0, 0, 0, 0}), Span(&result[1], 4));
  Store<1>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 4>{1, 0, 0, 0}), Span(&result[1], 4));
  Store<2>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 4>{1, 2, 0, 0}), Span(&result[1], 4));
  Store<3>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 4>{1, 2, 3, 0}), Span(&result[1], 4));
  Store<4>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 4>{1, 2, 3, 4}), Span(&result[1], 4));
  result.clear();
  result.resize(5);
  Store(&result[1], v, 0);
  EXPECT_SPAN_EQ((std::array<int, 4>{0, 0, 0, 0}), Span(&result[1], 4));
  Store(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 4>{1, 2, 3, 4}), Span(&result[1], 4));
}

TEST(Vec4i, StoreSelected) {
  Vec4i srcValues{1, 2, 3, 4};
  std::array<int, 4> condition{};
  std::array<int, 4> expectedValues{};
  alignas(16) std::array<int, 6> dstBuffer{};
  dstBuffer[0] = 123;
  dstBuffer[5] = 456;
  Span<int> dstValues{&dstBuffer[1], 4}; // Not aligned
  for (int mask = 0; mask < 16; ++mask) { // For each possible set of conditions
    int expectedCount = 0;
    for (int i = 0; i < 4; ++i) {
      condition[i] = (mask & (1 << i)) ? -1 : 0; // true = -1, false = 0
      if (condition[i]) {
        expectedValues[expectedCount++] = srcValues[i];
      }
    }
    auto countStored = StoreSelected(dstValues.data(), Load<Vec4i>(condition.data()), srcValues);
    EXPECT_EQ(expectedCount, countStored);
    for (int i = 0; i < countStored; ++i) {
      EXPECT_EQ(expectedValues[i], dstValues[i]);
    }
  }
  // Sentinel values should remain unchanged
  EXPECT_EQ(123, dstBuffer[0]);
  EXPECT_EQ(456, dstBuffer[5]);
}

TEST(Vec4i, StoreSelected_LargerConditionType) {
  Vec4i srcValues{1, 2, 3, 4};
  std::array<int64_t, 4> condition{};
  std::array<int, 4> expectedValues{};
  alignas(16) std::array<int, 6> dstBuffer{};
  dstBuffer[0] = 123;
  dstBuffer[5] = 456;
  Span<int> dstValues{&dstBuffer[1], 4}; // Not aligned
  for (int64_t mask = 0; mask < 16; ++mask) { // For each possible set of conditions
    int expectedCount = 0;
    for (int i = 0; i < 4; ++i) {
      condition[i] = (mask & (1 << i)) ? -1 : 0; // true = -1, false = 0
      if (condition[i]) {
        expectedValues[expectedCount++] = srcValues[i];
      }
    }
    // Use a Vec4d condition and Vec4i srcValues.
    // In real code, this is mostly likely to occur when the condition is written as Vec4r.
    auto condition4d = ReinterpretCast<Vec4d>(Load<Vec4l>(condition.data()));
    auto countStored = StoreSelected(dstValues.data(), condition4d, srcValues);
    EXPECT_EQ(expectedCount, countStored);
    for (int i = 0; i < countStored; ++i) {
      EXPECT_EQ(expectedValues[i], dstValues[i]);
    }
  }
  // Sentinel values should remain unchanged
  EXPECT_EQ(123, dstBuffer[0]);
  EXPECT_EQ(456, dstBuffer[5]);
}

TEST(Vec4i, StoreTransposed) {
  alignas(alignof(Vec4i)) int result[14] = {};
  result[13] = 911; // Sentinel value
  auto* ptr = result + 1; // Not an aligned address
  Vec4i data[3] = {Vec4i{1, 4, 7, 10}, Vec4i{2, 5, 8, 11}, Vec4i{3, 6, 9, 12}};
  StoreTransposed<1>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int, 12>{1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}), Span(ptr, 12));
  StoreTransposed<2>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int, 12>{1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}), Span(ptr, 12));
  StoreTransposed<3>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0}), Span(ptr, 12));
  StoreTransposed<4>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), Span(ptr, 12));
  StoreTransposed(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), Span(ptr, 12));
  EXPECT_EQ(0, result[0]); // No change
  EXPECT_EQ(911, result[13]); // No change
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec4i, Sub, -);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec4i, BitwiseXOR, ^);
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec4i, Max, ([](auto a, auto b) { return std::max(a, b); }));
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec4i, Min, ([](auto a, auto b) { return std::min(a, b); }));

TEST(Vec4i, ShiftLeft) {
  Simd<int, 4> v(1, 5, 17, 23);
  EXPECT_VEC4I(8, 40, 136, 184, v << 3);
  EXPECT_EQ(v, v << 0);
}

TEST(Vec4i, ShiftRight) {
  Vec4i v(8, 40, 56, 88);
  EXPECT_VEC4I(1, 5, 7, 11, ShiftRight<3>(v));
  EXPECT_EQ(v, ShiftRight<0>(v));
}
