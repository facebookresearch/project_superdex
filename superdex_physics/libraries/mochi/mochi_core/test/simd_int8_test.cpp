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
#include <limits>
#include <type_traits>
#include <vector>

using namespace mochi;
using namespace mochi::simd_test;

static_assert(std::is_trivially_copyable_v<Simd<int, 8>>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec8i);

#define EXPECT_VEC8I(e0, e1, e2, e3, e4, e5, e6, e7, actual) \
  {                                                          \
    Vec8i vActual = actual;                                  \
    EXPECT_EQ((e0), vActual[0]);                             \
    EXPECT_EQ((e1), vActual[1]);                             \
    EXPECT_EQ((e2), vActual[2]);                             \
    EXPECT_EQ((e3), vActual[3]);                             \
    EXPECT_EQ((e4), vActual[4]);                             \
    EXPECT_EQ((e5), vActual[5]);                             \
    EXPECT_EQ((e6), vActual[6]);                             \
    EXPECT_EQ((e7), vActual[7]);                             \
  }

TEST(Vec8i, Class) {
  static_assert(Vec8i::kIsSupported, "Should be supported");

#if MOCHI_USE_SIMD
  static_assert(Vec8i::kIsComposite == MOCHI_ARCH_ARM_NEON);
  static_assert(!Vec8i::kIsEmulated);
#else
  static_assert(!Vec8i::kIsComposite);
  static_assert(Vec8i::kIsEmulated);
#endif

  static_assert(sizeof(Vec8i) == sizeof(int) * 8);
  static_assert(alignof(Vec8i) == alignof(typename Vec8i::NativeType));
  static_assert(std::is_same_v<Vec8i::Scalar, int>);
  static_assert(Vec8i::kSize == 8);
  static_assert(Vec8i::size() == 8);

  // Construct from broadcast
  EXPECT_VEC8I(1, 1, 1, 1, 1, 1, 1, 1, Vec8i(1));

  // Construct from scalars
  EXPECT_VEC8I(1, 2, 0, 0, 0, 0, 0, 0, Vec8i(1, 2));
  EXPECT_VEC8I(1, 2, 3, 0, 0, 0, 0, 0, Vec8i(1, 2, 3));
  EXPECT_VEC8I(1, 2, 3, 4, 0, 0, 0, 0, Vec8i(1, 2, 3, 4));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 0, 0, 0, Vec8i(1, 2, 3, 4, 5));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 0, 0, Vec8i(1, 2, 3, 4, 5, 6));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 0, Vec8i(1, 2, 3, 4, 5, 6, 7));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 8, Vec8i(1, 2, 3, 4, 5, 6, 7, 8));

  // Construct from two Vec4i
  Vec4i const low{1, 2, 3, 4};
  Vec4i const high{5, 6, 7, 8};
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 8, Vec8i(low, high));

  // Implicit conversion from scalar
  Vec8i a = 2;
  EXPECT_VEC8I(2, 2, 2, 2, 2, 2, 2, 2, a);
  a = 3;
  EXPECT_VEC8I(3, 3, 3, 3, 3, 3, 3, 3, a);

  // Copy construct
  Vec8i b{a};
  EXPECT_VEC8I(3, 3, 3, 3, 3, 3, 3, 3, b);

  // Copy assign
  a = Vec8i{4};
  b = a;
  EXPECT_VEC8I(4, 4, 4, 4, 4, 4, 4, 4, b);

  // Comparison
  EXPECT_EQ(true, (a == b));
  EXPECT_EQ(true, (a != Vec8i{}));
  EXPECT_EQ(false, (a != b));
  EXPECT_EQ(false, (a == Vec8i{}));

  // Unary operators
  Vec8i ones{-1};
  Vec8i zeros{0};
  EXPECT_EQ(ones, ~zeros);
  EXPECT_EQ(zeros, ~ones);
  EXPECT_VEC8I(-1, -2, -3, -4, -5, -6, -7, -8, -Vec8i(1, 2, 3, 4, 5, 6, 7, 8));

  // Binary operators
  a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  b = Vec8i{2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_VEC8I(3, 5, 7, 9, 11, 13, 15, 17, a + b);
  EXPECT_VEC8I(4, 5, 6, 7, 8, 9, 10, 11, a + 3);
  EXPECT_VEC8I(4, 5, 6, 7, 8, 9, 10, 11, 3 + a);
  EXPECT_VEC8I(-1, -1, -1, -1, -1, -1, -1, -1, a - b);
  EXPECT_VEC8I(-2, -1, 0, 1, 2, 3, 4, 5, a - 3);
  EXPECT_VEC8I(2, 1, 0, -1, -2, -3, -4, -5, 3 - a);
  EXPECT_VEC8I(2, 6, 12, 20, 30, 42, 56, 72, a * b);
  EXPECT_VEC8I(3, 6, 9, 12, 15, 18, 21, 24, a * 3);
  EXPECT_VEC8I(3, 6, 9, 12, 15, 18, 21, 24, 3 * a);
  EXPECT_VEC8I(2, 1, 1, 1, 1, 1, 1, 1, b / a);
  EXPECT_VEC8I(0, 0, 1, 1, 1, 2, 2, 2, a / 3);
  EXPECT_VEC8I(10, 5, 3, 2, 2, 1, 1, 1, 10 / a);
  EXPECT_EQ(a, a & ones);
  EXPECT_EQ(zeros, a & zeros);
  EXPECT_EQ(ones, Vec8i(a | ones));
  EXPECT_EQ(a, a | zeros);
  EXPECT_EQ(~a, a ^ ones);
  EXPECT_EQ(a, a ^ zeros);

  // clang-format off

  // Update operators
  a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  a += b;
  EXPECT_VEC8I(3, 5, 7, 9, 11, 13, 15, 17, a);
  a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  a += 3;
  EXPECT_VEC8I(4, 5, 6, 7, 8, 9, 10, 11, a);
  a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  a -= b;
  EXPECT_VEC8I(-1, -1, -1, -1, -1, -1, -1, -1, a);
  a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  a -= 3;
  EXPECT_VEC8I(-2, -1, 0, 1, 2, 3, 4, 5, a);
  a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  a *= b;
  EXPECT_VEC8I(2, 6, 12, 20, 30, 42, 56, 72, a);
  a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  a *= 3;
  EXPECT_VEC8I(3, 6, 9, 12, 15, 18, 21, 24, a);
  a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  b = Vec8i{2, 3, 4, 5, 6, 7, 8, 9};
  b /= a;
  EXPECT_VEC8I(2, 1, 1, 1, 1, 1, 1, 1, b);
  a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  a /= 3;
  EXPECT_VEC8I(0, 0, 1, 1, 1, 2, 2, 2, a);
  a = b = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  b &= ones;
  EXPECT_EQ(a, b);
  a = b = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  b &= zeros;
  EXPECT_EQ(zeros, b);
  a = b = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  b |= ones;
  EXPECT_EQ(ones, b);
  a = b = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  b |= zeros;
  EXPECT_EQ(a, b);
  a = b = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  b ^= ones;
  EXPECT_EQ(~a, b);
  a = b = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  b &= zeros;
  EXPECT_EQ(zeros, b);
  a = b = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  b &= ones;
  EXPECT_EQ(a, b);

  // clang-format on
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec8i, Add, +);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec8i, BitwiseAND, &);
MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(Vec8i);
MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec8i, Div, /);

TEST(Vec8i, Get) {
  auto a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};

  // Fast template version
  EXPECT_EQ(1, Get0(a));
  EXPECT_EQ(1, Get<0>(a));
  EXPECT_EQ(2, Get<1>(a));
  EXPECT_EQ(3, Get<2>(a));
  EXPECT_EQ(4, Get<3>(a));
  EXPECT_EQ(5, Get<4>(a));
  EXPECT_EQ(6, Get<5>(a));
  EXPECT_EQ(7, Get<6>(a));
  EXPECT_EQ(8, Get<7>(a));

  // Slower runtime version
  EXPECT_EQ(1, Get(a, 0));
  EXPECT_EQ(2, Get(a, 1));
  EXPECT_EQ(3, Get(a, 2));
  EXPECT_EQ(4, Get(a, 3));
  EXPECT_EQ(5, Get(a, 4));
  EXPECT_EQ(6, Get(a, 5));
  EXPECT_EQ(7, Get(a, 6));
  EXPECT_EQ(8, Get(a, 7));

  // Same but with operator[] (read only)
  EXPECT_EQ(1, a[0]);
  EXPECT_EQ(2, a[1]);
  EXPECT_EQ(3, a[2]);
  EXPECT_EQ(4, a[3]);
  EXPECT_EQ(5, a[4]);
  EXPECT_EQ(6, a[5]);
  EXPECT_EQ(7, a[6]);
  EXPECT_EQ(8, a[7]);
}

TEST(Vec8i, GetHalf) {
  auto a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  auto low = GetHalf<0>(a);
  auto high = GetHalf<1>(a);
  EXPECT_EQ(1, Get(low, 0));
  EXPECT_EQ(2, Get(low, 1));
  EXPECT_EQ(3, Get(low, 2));
  EXPECT_EQ(4, Get(low, 3));
  EXPECT_EQ(5, Get(high, 0));
  EXPECT_EQ(6, Get(high, 1));
  EXPECT_EQ(7, Get(high, 2));
  EXPECT_EQ(8, Get(high, 3));
}

TEST(Vec8i, Set) {
  auto a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_VEC8I(9, 2, 3, 4, 5, 6, 7, 8, Set<0>(a, 9));
  EXPECT_VEC8I(1, 9, 3, 4, 5, 6, 7, 8, Set<1>(a, 9));
  EXPECT_VEC8I(1, 2, 9, 4, 5, 6, 7, 8, Set<2>(a, 9));
  EXPECT_VEC8I(1, 2, 3, 9, 5, 6, 7, 8, Set<3>(a, 9));
  EXPECT_VEC8I(1, 2, 3, 4, 9, 6, 7, 8, Set<4>(a, 9));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 9, 7, 8, Set<5>(a, 9));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 9, 8, Set<6>(a, 9));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 9, Set<7>(a, 9));

  EXPECT_VEC8I(9, 2, 3, 4, 5, 6, 7, 8, Set(a, 0, 9));
  EXPECT_VEC8I(1, 9, 3, 4, 5, 6, 7, 8, Set(a, 1, 9));
  EXPECT_VEC8I(1, 2, 9, 4, 5, 6, 7, 8, Set(a, 2, 9));
  EXPECT_VEC8I(1, 2, 3, 9, 5, 6, 7, 8, Set(a, 3, 9));
  EXPECT_VEC8I(1, 2, 3, 4, 9, 6, 7, 8, Set(a, 4, 9));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 9, 7, 8, Set(a, 5, 9));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 9, 8, Set(a, 6, 9));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 9, Set(a, 7, 9));
}

TEST(Vec8i, Sequence) {
  EXPECT_VEC8I(0, 1, 2, 3, 4, 5, 6, 7, Sequence<Vec8i>());
}

// We don't have a separate file for testing Simd<int, 12>. This test is here because it covers
// additional code paths in the composite implementation. On ARM, Simd<int, 12>::Second will be
// another composite. On x64, it will be a native type that is not the same as Simd<int, 12>::First.
TEST(Vec12i, Sequence) {
  auto a = Sequence<Simd<int, 12>>();
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(i, a[i]);
  }
}

TEST(Vec8i, AllTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  int zeros = 0;
  int ones = -1;
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          for (int e = 0; e < 2; ++e) {
            for (int f = 0; f < 2; ++f) {
              for (int g = 0; g < 2; ++g) {
                for (int h = 0; h < 2; ++h) {
                  auto vec = Vec8i{
                      a ? ones : zeros,
                      b ? ones : zeros,
                      c ? ones : zeros,
                      d ? ones : zeros,
                      e ? ones : zeros,
                      f ? ones : zeros,
                      g ? ones : zeros,
                      h ? ones : zeros};
                  EXPECT_EQ(!!a, AllTrue<1>(vec));
                  EXPECT_EQ(a && b, AllTrue<2>(vec));
                  EXPECT_EQ(a && b && c, AllTrue<3>(vec));
                  EXPECT_EQ(a && b && c && d, AllTrue<4>(vec));
                  EXPECT_EQ(a && b && c && d && e, AllTrue<5>(vec));
                  EXPECT_EQ(a && b && c && d && e && f, AllTrue<6>(vec));
                  EXPECT_EQ(a && b && c && d && e && f && g, AllTrue<7>(vec));
                  EXPECT_EQ(a && b && c && d && e && f && g && h, AllTrue<8>(vec));
                  EXPECT_EQ(a && b && c && d && e && f && g && h, AllTrue(vec));
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(Vec8i, AnyTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  int zeros = 0;
  int ones = -1;
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          for (int e = 0; e < 2; ++e) {
            for (int f = 0; f < 2; ++f) {
              for (int g = 0; g < 2; ++g) {
                for (int h = 0; h < 2; ++h) {
                  auto vec = Vec8i{
                      a ? ones : zeros,
                      b ? ones : zeros,
                      c ? ones : zeros,
                      d ? ones : zeros,
                      e ? ones : zeros,
                      f ? ones : zeros,
                      g ? ones : zeros,
                      h ? ones : zeros};
                  EXPECT_EQ(!!a, AnyTrue<1>(vec));
                  EXPECT_EQ(a || b, AnyTrue<2>(vec));
                  EXPECT_EQ(a || b || c, AnyTrue<3>(vec));
                  EXPECT_EQ(a || b || c || d, AnyTrue<4>(vec));
                  EXPECT_EQ(a || b || c || d || e, AnyTrue<5>(vec));
                  EXPECT_EQ(a || b || c || d || e || f, AnyTrue<6>(vec));
                  EXPECT_EQ(a || b || c || d || e || f || g, AnyTrue<7>(vec));
                  EXPECT_EQ(a || b || c || d || e || f || g || h, AnyTrue<8>(vec));
                  EXPECT_EQ(a || b || c || d || e || f || g || h, AnyTrue(vec));
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(Vec8i, Broadcast) {
  // Broadcast scalar
  EXPECT_VEC8I(1, 1, 1, 1, 1, 1, 1, 1, Broadcast<Vec8i>(1));

  // Broadcast from address
  int const s = 2;
  EXPECT_VEC8I(2, 2, 2, 2, 2, 2, 2, 2, Broadcast<Vec8i>(&s));

  // Broadcast ith element
  auto v = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_VEC8I(1, 1, 1, 1, 1, 1, 1, 1, Broadcast<0>(v));
  EXPECT_VEC8I(2, 2, 2, 2, 2, 2, 2, 2, Broadcast<1>(v));
  EXPECT_VEC8I(3, 3, 3, 3, 3, 3, 3, 3, Broadcast<2>(v));
  EXPECT_VEC8I(4, 4, 4, 4, 4, 4, 4, 4, Broadcast<3>(v));
  EXPECT_VEC8I(5, 5, 5, 5, 5, 5, 5, 5, Broadcast<4>(v));
  EXPECT_VEC8I(6, 6, 6, 6, 6, 6, 6, 6, Broadcast<5>(v));
  EXPECT_VEC8I(7, 7, 7, 7, 7, 7, 7, 7, Broadcast<6>(v));
  EXPECT_VEC8I(8, 8, 8, 8, 8, 8, 8, 8, Broadcast<7>(v));

  // Broadcast ith element
  EXPECT_VEC8I(1, 1, 1, 1, 1, 1, 1, 1, Broadcast(v, 0));
  EXPECT_VEC8I(2, 2, 2, 2, 2, 2, 2, 2, Broadcast(v, 1));
  EXPECT_VEC8I(3, 3, 3, 3, 3, 3, 3, 3, Broadcast(v, 2));
  EXPECT_VEC8I(4, 4, 4, 4, 4, 4, 4, 4, Broadcast(v, 3));
  EXPECT_VEC8I(5, 5, 5, 5, 5, 5, 5, 5, Broadcast(v, 4));
  EXPECT_VEC8I(6, 6, 6, 6, 6, 6, 6, 6, Broadcast(v, 5));
  EXPECT_VEC8I(7, 7, 7, 7, 7, 7, 7, 7, Broadcast(v, 6));
  EXPECT_VEC8I(8, 8, 8, 8, 8, 8, 8, 8, Broadcast(v, 7));
}

TEST(Vec8i, IsTrue) {
  auto a = SimdMask<Vec8i>(true, false, true, false, true, false, true, false);
  EXPECT_TRUE(IsTrue<0>(a));
  EXPECT_FALSE(IsTrue<1>(a));
  EXPECT_TRUE(IsTrue<2>(a));
  EXPECT_FALSE(IsTrue<3>(a));
  EXPECT_TRUE(IsTrue<4>(a));
  EXPECT_FALSE(IsTrue<5>(a));
  EXPECT_TRUE(IsTrue<6>(a));
  EXPECT_FALSE(IsTrue<7>(a));
  EXPECT_FALSE(IsTrue<0>(~a));
  EXPECT_TRUE(IsTrue<1>(~a));
  EXPECT_FALSE(IsTrue<2>(~a));
  EXPECT_TRUE(IsTrue<3>(~a));
  EXPECT_FALSE(IsTrue<4>(~a));
  EXPECT_TRUE(IsTrue<5>(~a));
  EXPECT_FALSE(IsTrue<6>(~a));
  EXPECT_TRUE(IsTrue<7>(~a));
}

TEST(Vec8i, Greater) {
  auto a = Vec8i{1, 4, 5, 8, 9, 12, 13, 16};
  auto b = Vec8i{2, 3, 6, 7, 10, 11, 14, 15};
  EXPECT_VEC8I(0, 1, 0, 1, 0, 1, 0, 1, (a > b) & Vec8i{1});
  EXPECT_VEC8I(1, 0, 1, 0, 1, 0, 1, 0, (b > a) & Vec8i{1});
  EXPECT_VEC8I(0, 0, 0, 0, 0, 0, 0, 0, (a > a) & Vec8i{1});
  EXPECT_VEC8I(0, 0, 1, 1, 1, 1, 1, 1, (a > Vec8i{4}) & Vec8i{1});
  EXPECT_VEC8I(0, 0, 0, 1, 1, 1, 1, 1, (a > Vec8i{5}) & Vec8i{1});
  EXPECT_VEC8I(1, 0, 0, 0, 0, 0, 0, 0, (Vec8i{4} > a) & Vec8i{1});
  EXPECT_VEC8I(1, 1, 0, 0, 0, 0, 0, 0, (Vec8i{5} > a) & Vec8i{1});
}

TEST(Vec8i, GreaterEqual) {
  auto a = Vec8i{1, 4, 5, 8, 9, 12, 13, 16};
  auto b = Vec8i{2, 3, 6, 7, 10, 11, 14, 15};
  EXPECT_VEC8I(0, 1, 0, 1, 0, 1, 0, 1, (a >= b) & Vec8i{1});
  EXPECT_VEC8I(1, 0, 1, 0, 1, 0, 1, 0, (b >= a) & Vec8i{1});
  EXPECT_VEC8I(1, 1, 1, 1, 1, 1, 1, 1, (a >= a) & Vec8i{1});
  EXPECT_VEC8I(0, 1, 1, 1, 1, 1, 1, 1, (a >= Vec8i{4}) & Vec8i{1});
  EXPECT_VEC8I(0, 0, 1, 1, 1, 1, 1, 1, (a >= Vec8i{5}) & Vec8i{1});
  EXPECT_VEC8I(1, 1, 0, 0, 0, 0, 0, 0, (Vec8i{4} >= a) & Vec8i{1});
  EXPECT_VEC8I(1, 1, 1, 0, 0, 0, 0, 0, (Vec8i{5} >= a) & Vec8i{1});
}

TEST(Vec8i, HMax) {
  auto a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(2, HMax<2>(a));
  EXPECT_EQ(3, HMax<3>(a));
  EXPECT_EQ(4, HMax<4>(a));
  EXPECT_EQ(5, HMax<5>(a));
  EXPECT_EQ(6, HMax<6>(a));
  EXPECT_EQ(7, HMax<7>(a));
  EXPECT_EQ(8, HMax(a));
}

TEST(Vec8i, HMin) {
  auto a = Vec8i{8, 7, 6, 5, 4, 3, 2, 1};
  EXPECT_EQ(7, HMin<2>(a));
  EXPECT_EQ(6, HMin<3>(a));
  EXPECT_EQ(5, HMin<4>(a));
  EXPECT_EQ(4, HMin<5>(a));
  EXPECT_EQ(3, HMin<6>(a));
  EXPECT_EQ(2, HMin<7>(a));
  EXPECT_EQ(1, HMin(a));
}

TEST(Vec8i, HSum) {
  auto a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(3, HSum<2>(a));
  EXPECT_EQ(6, HSum<3>(a));
  EXPECT_EQ(10, HSum<4>(a));
  EXPECT_EQ(15, HSum<5>(a));
  EXPECT_EQ(21, HSum<6>(a));
  EXPECT_EQ(28, HSum<7>(a));
  EXPECT_EQ(36, HSum(a));
}

TEST(Vec8i, Less) {
  auto a = Vec8i{1, 4, 5, 8, 9, 12, 13, 16};
  auto b = Vec8i{2, 3, 6, 7, 10, 11, 14, 15};
  EXPECT_VEC8I(0, 1, 0, 1, 0, 1, 0, 1, (b < a) & Vec8i{1});
  EXPECT_VEC8I(1, 0, 1, 0, 1, 0, 1, 0, (a < b) & Vec8i{1});
  EXPECT_VEC8I(0, 0, 0, 0, 0, 0, 0, 0, (a < a) & Vec8i{1});
  EXPECT_VEC8I(1, 0, 0, 0, 0, 0, 0, 0, (a < Vec8i{4}) & Vec8i{1});
  EXPECT_VEC8I(1, 1, 0, 0, 0, 0, 0, 0, (a < Vec8i{5}) & Vec8i{1});
  EXPECT_VEC8I(0, 0, 1, 1, 1, 1, 1, 1, (Vec8i{4} < a) & Vec8i{1});
  EXPECT_VEC8I(0, 0, 0, 1, 1, 1, 1, 1, (Vec8i{5} < a) & Vec8i{1});
}

TEST(Vec8i, LessEqual) {
  auto a = Vec8i{1, 4, 5, 8, 9, 12, 13, 16};
  auto b = Vec8i{2, 3, 6, 7, 10, 11, 14, 15};
  EXPECT_VEC8I(0, 1, 0, 1, 0, 1, 0, 1, (b <= a) & Vec8i{1});
  EXPECT_VEC8I(1, 0, 1, 0, 1, 0, 1, 0, (a <= b) & Vec8i{1});
  EXPECT_VEC8I(1, 1, 1, 1, 1, 1, 1, 1, (a <= a) & Vec8i{1});
  EXPECT_VEC8I(1, 1, 0, 0, 0, 0, 0, 0, (a <= Vec8i{4}) & Vec8i{1});
  EXPECT_VEC8I(1, 1, 1, 0, 0, 0, 0, 0, (a <= Vec8i{5}) & Vec8i{1});
  EXPECT_VEC8I(0, 1, 1, 1, 1, 1, 1, 1, (Vec8i{4} <= a) & Vec8i{1});
  EXPECT_VEC8I(0, 0, 1, 1, 1, 1, 1, 1, (Vec8i{5} <= a) & Vec8i{1});
}

// Test both Equal and NotEqual
TEST(Vec8i, Equal) {
  auto a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  auto b = a;
  auto c = Vec8i{9, 2, 3, 4, 5, 6, 7, 8};
  auto d = Vec8i{1, 9, 3, 4, 5, 6, 7, 8};
  auto e = Vec8i{1, 2, 9, 4, 5, 6, 7, 8};
  auto f = Vec8i{1, 2, 3, 9, 5, 6, 7, 8};
  auto g = Vec8i{1, 2, 3, 4, 9, 6, 7, 8};
  auto h = Vec8i{1, 2, 3, 4, 5, 9, 7, 8};
  auto i = Vec8i{1, 2, 3, 4, 5, 6, 9, 8};
  auto j = Vec8i{1, 2, 3, 4, 5, 6, 7, 9};

  // Compare 1 components
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

  // Compare 5 components
  ExpectEqual<5>(true, a, b);
  ExpectEqual<5>(true, a, h);
  ExpectEqual<5>(false, a, g);

  // Compare 6 components
  ExpectEqual<6>(true, a, b);
  ExpectEqual<6>(true, a, i);
  ExpectEqual<6>(false, a, h);

  // Compare 7 components
  ExpectEqual<7>(true, a, b);
  ExpectEqual<7>(true, a, j);
  ExpectEqual<7>(false, a, i);

  // Compare 8 components
  ExpectEqual<8>(true, a, b);
  ExpectEqual<8>(false, a, j);
}

// Test bot VEqual and VNotEqual
template <class V>
static void ExpectVEqual(int e0, int e1, int e2, int e3, int e4, int e5, int e6, int e7, V a, V b) {
  EXPECT_VEC8I(e0, e1, e2, e3, e4, e5, e6, e7, VEqual(a, b) & Vec8i{1});
  EXPECT_VEC8I(
      int(!e0),
      int(!e1),
      int(!e2),
      int(!e3),
      int(!e4),
      int(!e5),
      int(!e6),
      int(!e7),
      VNotEqual(a, b) & Vec8i{1});
}

TEST(Vec8i, VEqual) {
  auto a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  auto b = Vec8i{1, 9, 3, 9, 5, 9, 7, 9};
  auto c = Vec8i{9, 2, 9, 4, 9, 6, 9, 8};
  ExpectVEqual(1, 1, 1, 1, 1, 1, 1, 1, a, a);
  ExpectVEqual(1, 0, 1, 0, 1, 0, 1, 0, a, b);
  ExpectVEqual(0, 1, 0, 1, 0, 1, 0, 1, a, c);
}

TEST(Vec8i, Load) {
  alignas(alignof(Vec8i)) int const values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_VEC8I(0, 0, 0, 0, 0, 0, 0, 0, (Load<0, Vec8i>(nullptr)));
  EXPECT_VEC8I(1, 0, 0, 0, 0, 0, 0, 0, (Load<1, Vec8i>(values + 1)));
  EXPECT_VEC8I(1, 2, 0, 0, 0, 0, 0, 0, (Load<2, Vec8i>(values + 1)));
  EXPECT_VEC8I(1, 2, 3, 0, 0, 0, 0, 0, (Load<3, Vec8i>(values + 1)));
  EXPECT_VEC8I(1, 2, 3, 4, 0, 0, 0, 0, (Load<4, Vec8i>(values + 1)));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 0, 0, 0, (Load<5, Vec8i>(values + 1)));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 0, 0, (Load<6, Vec8i>(values + 1)));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 0, (Load<7, Vec8i>(values + 1)));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 8, (Load<8, Vec8i>(values + 1)));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 8, (Load<Vec8i>(values + 1)));

  EXPECT_VEC8I(1, 0, 0, 0, 0, 0, 0, 0, (Load<Vec8i>(values + 1, 1)));
  EXPECT_VEC8I(1, 2, 0, 0, 0, 0, 0, 0, (Load<Vec8i>(values + 1, 2)));
  EXPECT_VEC8I(1, 2, 3, 0, 0, 0, 0, 0, (Load<Vec8i>(values + 1, 3)));
  EXPECT_VEC8I(1, 2, 3, 4, 0, 0, 0, 0, (Load<Vec8i>(values + 1, 4)));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 0, 0, 0, (Load<Vec8i>(values + 1, 5)));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 0, 0, (Load<Vec8i>(values + 1, 6)));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 0, (Load<Vec8i>(values + 1, 7)));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 8, (Load<Vec8i>(values + 1, 8)));
}

TEST(Vec8i, LoadTransposed) {
  alignas(alignof(Vec8i)) int const values[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  auto const* ptr = values + 1; // Not an aligned address
  Vec8i loaded[3] = {};
  LoadTransposed<1>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8I(1, 0, 0, 0, 0, 0, 0, 0, loaded[0]);
  EXPECT_VEC8I(2, 0, 0, 0, 0, 0, 0, 0, loaded[1]);
  EXPECT_VEC8I(3, 0, 0, 0, 0, 0, 0, 0, loaded[2]);
  LoadTransposed<2>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8I(1, 4, 0, 0, 0, 0, 0, 0, loaded[0]);
  EXPECT_VEC8I(2, 5, 0, 0, 0, 0, 0, 0, loaded[1]);
  EXPECT_VEC8I(3, 6, 0, 0, 0, 0, 0, 0, loaded[2]);
  LoadTransposed<3>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8I(1, 4, 7, 0, 0, 0, 0, 0, loaded[0]);
  EXPECT_VEC8I(2, 5, 8, 0, 0, 0, 0, 0, loaded[1]);
  EXPECT_VEC8I(3, 6, 9, 0, 0, 0, 0, 0, loaded[2]);
  LoadTransposed<4>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8I(1, 4, 7, 10, 0, 0, 0, 0, loaded[0]);
  EXPECT_VEC8I(2, 5, 8, 11, 0, 0, 0, 0, loaded[1]);
  EXPECT_VEC8I(3, 6, 9, 12, 0, 0, 0, 0, loaded[2]);
  LoadTransposed<5>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8I(1, 4, 7, 10, 13, 0, 0, 0, loaded[0]);
  EXPECT_VEC8I(2, 5, 8, 11, 14, 0, 0, 0, loaded[1]);
  EXPECT_VEC8I(3, 6, 9, 12, 15, 0, 0, 0, loaded[2]);
  LoadTransposed<6>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8I(1, 4, 7, 10, 13, 16, 0, 0, loaded[0]);
  EXPECT_VEC8I(2, 5, 8, 11, 14, 17, 0, 0, loaded[1]);
  EXPECT_VEC8I(3, 6, 9, 12, 15, 18, 0, 0, loaded[2]);
  LoadTransposed<7>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8I(1, 4, 7, 10, 13, 16, 19, 0, loaded[0]);
  EXPECT_VEC8I(2, 5, 8, 11, 14, 17, 20, 0, loaded[1]);
  EXPECT_VEC8I(3, 6, 9, 12, 15, 18, 21, 0, loaded[2]);
  LoadTransposed<8>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8I(1, 4, 7, 10, 13, 16, 19, 22, loaded[0]);
  EXPECT_VEC8I(2, 5, 8, 11, 14, 17, 20, 23, loaded[1]);
  EXPECT_VEC8I(3, 6, 9, 12, 15, 18, 21, 24, loaded[2]);
  LoadTransposed(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8I(1, 4, 7, 10, 13, 16, 19, 22, loaded[0]);
  EXPECT_VEC8I(2, 5, 8, 11, 14, 17, 20, 23, loaded[1]);
  EXPECT_VEC8I(3, 6, 9, 12, 15, 18, 21, 24, loaded[2]);
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec8i, Mul, *);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec8i, Neg, -);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec8i, BitwiseNOT, ~);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec8i, BitwiseOR, |);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec8i, LogicalAND, &&, &);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec8i, LogicalOR, ||, |);
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec8i, Max, ([](auto a, auto b) { return std::max(a, b); }));
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec8i, Min, ([](auto a, auto b) { return std::min(a, b); }));

TEST(Vec8i, Select) {
  auto a = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  auto b = a * 10;
  for (int b0 = 0; b0 < 2; ++b0) {
    for (int b1 = 0; b1 < 2; ++b1) {
      for (int b2 = 0; b2 < 2; ++b2) {
        for (int b3 = 0; b3 < 2; ++b3) {
          for (int b4 = 0; b4 < 2; ++b4) {
            for (int b5 = 0; b5 < 2; ++b5) {
              for (int b6 = 0; b6 < 2; ++b6) {
                for (int b7 = 0; b7 < 2; ++b7) {
                  auto expected = Vec8i{
                      b0 ? Get(a, 0) : Get(b, 0),
                      b1 ? Get(a, 1) : Get(b, 1),
                      b2 ? Get(a, 2) : Get(b, 2),
                      b3 ? Get(a, 3) : Get(b, 3),
                      b4 ? Get(a, 4) : Get(b, 4),
                      b5 ? Get(a, 5) : Get(b, 5),
                      b6 ? Get(a, 6) : Get(b, 6),
                      b7 ? Get(a, 7) : Get(b, 7)};
                  auto actual =
                      Select(SimdMask<Vec8i>(!!b0, !!b1, !!b2, !!b3, !!b4, !!b5, !!b6, !!b7), a, b);
                  EXPECT_EQ(expected, actual);
                }
              }
            }
          }
        }
      }
    }
  }
  // More typical usage
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 8, Select(a < b, a, b)); // Min(a, b)
  EXPECT_VEC8I(10, 20, 30, 40, 50, 60, 70, 80, Select(a > b, a, b)); // Max(a, b)
  EXPECT_VEC8I(1, 2, 3, 4, 50, 60, 70, 80, Select(a <= Vec8i{4}, a, b));
  EXPECT_VEC8I(10, 20, 30, 40, 5, 6, 7, 8, Select(a > Vec8i{4}, a, b));
}

TEST(Vec8i, SimdMask) {
  auto v = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  bool constexpr f = false;
  bool constexpr t = true;
  EXPECT_VEC8I(0, 0, 0, 0, 0, 0, 0, 0, v & SimdMask<Vec8i>(f, f, f, f, f, f, f, f));
  EXPECT_VEC8I(1, 0, 0, 0, 0, 0, 0, 0, v & SimdMask<Vec8i>(t, f, f, f, f, f, f, f));
  EXPECT_VEC8I(0, 2, 0, 0, 0, 0, 0, 0, v & SimdMask<Vec8i>(f, t, f, f, f, f, f, f));
  EXPECT_VEC8I(0, 0, 3, 0, 0, 0, 0, 0, v & SimdMask<Vec8i>(f, f, t, f, f, f, f, f));
  EXPECT_VEC8I(0, 0, 0, 4, 0, 0, 0, 0, v & SimdMask<Vec8i>(f, f, f, t, f, f, f, f));
  EXPECT_VEC8I(0, 0, 0, 0, 5, 0, 0, 0, v & SimdMask<Vec8i>(f, f, f, f, t, f, f, f));
  EXPECT_VEC8I(0, 0, 0, 0, 0, 6, 0, 0, v & SimdMask<Vec8i>(f, f, f, f, f, t, f, f));
  EXPECT_VEC8I(0, 0, 0, 0, 0, 0, 7, 0, v & SimdMask<Vec8i>(f, f, f, f, f, f, t, f));
  EXPECT_VEC8I(0, 0, 0, 0, 0, 0, 0, 8, v & SimdMask<Vec8i>(f, f, f, f, f, f, f, t));
  EXPECT_VEC8I(1, 0, 3, 0, 5, 0, 7, 0, v & SimdMask<Vec8i>(t, f, t, f, t, f, t, f));
  EXPECT_VEC8I(0, 2, 0, 4, 0, 6, 0, 8, v & SimdMask<Vec8i>(f, t, f, t, f, t, f, t));
  EXPECT_VEC8I(1, 2, 3, 4, 0, 0, 0, 0, v & SimdMask<Vec8i>(t, t, t, t, f, f, f, f));
  EXPECT_VEC8I(0, 0, 0, 0, 5, 6, 7, 8, v & SimdMask<Vec8i>(f, f, f, f, t, t, t, t));
  EXPECT_VEC8I(1, 2, 3, 4, 5, 6, 7, 8, v & SimdMask<Vec8i>(t, t, t, t, t, t, t, t));
}

TEST(Vec8i, SimdZero) {
  EXPECT_VEC8I(0, 0, 0, 0, 0, 0, 0, 0, SimdZero<Vec8i>());
}

MOCHI_SIMD_TEST_UNARY_FN_EXACT(Vec8i, Sqr, ([](auto a) { return a * a; }));

TEST(Vec8i, Store) {
  std::vector<int> result(9); // NOTE: Changed from an array on the stack to work around an MSVC
                              // optimizer bug.
  auto const v = Vec8i{1, 2, 3, 4, 5, 6, 7, 8};
  Store<0>((int*)nullptr, v);
  Store<0>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{0, 0, 0, 0, 0, 0, 0, 0}), Span(&result[1], 8));
  Store<1>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 0, 0, 0, 0, 0, 0, 0}), Span(&result[1], 8));
  Store<2>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 0, 0, 0, 0, 0, 0}), Span(&result[1], 8));
  Store<3>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 0, 0, 0, 0, 0}), Span(&result[1], 8));
  Store<4>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 0, 0, 0, 0}), Span(&result[1], 8));
  Store<5>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 5, 0, 0, 0}), Span(&result[1], 8));
  Store<6>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 5, 6, 0, 0}), Span(&result[1], 8));
  Store<7>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 5, 6, 7, 0}), Span(&result[1], 8));
  Store<8>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 5, 6, 7, 8}), Span(&result[1], 8));
  result.clear();
  result.resize(9);
  Store(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 5, 6, 7, 8}), Span(&result[1], 8));

  result.clear();
  result.resize(9);
  Store(&result[1], v, 0);
  EXPECT_SPAN_EQ((std::array<int, 8>{0, 0, 0, 0, 0, 0, 0, 0}), Span(&result[1], 8));
  Store(&result[1], v, 1);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 0, 0, 0, 0, 0, 0, 0}), Span(&result[1], 8));
  Store(&result[1], v, 2);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 0, 0, 0, 0, 0, 0}), Span(&result[1], 8));
  Store(&result[1], v, 3);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 0, 0, 0, 0, 0}), Span(&result[1], 8));
  Store(&result[1], v, 4);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 0, 0, 0, 0}), Span(&result[1], 8));
  Store(&result[1], v, 5);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 5, 0, 0, 0}), Span(&result[1], 8));
  Store(&result[1], v, 6);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 5, 6, 0, 0}), Span(&result[1], 8));
  Store(&result[1], v, 7);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 5, 6, 7, 0}), Span(&result[1], 8));
  Store(&result[1], v, 8);
  EXPECT_SPAN_EQ((std::array<int, 8>{1, 2, 3, 4, 5, 6, 7, 8}), Span(&result[1], 8));
}

TEST(Vec8i, StoreSelected) {
  Vec8i srcValues{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<int, 8> condition{};
  std::array<int, 8> expectedValues{};
  alignas(16) std::array<int, 10> dstBuffer{};
  dstBuffer[0] = 123;
  dstBuffer[9] = 456;
  Span<int> dstValues{&dstBuffer[1], 8}; // Not aligned
  for (int mask = 0; mask < 256; ++mask) { // For each possible set of conditions
    int expectedCount = 0;
    for (int i = 0; i < 8; ++i) {
      condition[i] = (mask & (1 << i)) ? -1 : 0; // true = -1, false = 0
      if (condition[i]) {
        expectedValues[expectedCount++] = srcValues[i];
      }
    }
    auto countStored = StoreSelected(dstValues.data(), Load<Vec8i>(condition.data()), srcValues);
    EXPECT_EQ(expectedCount, countStored);
    for (int i = 0; i < countStored; ++i) {
      EXPECT_EQ(expectedValues[i], dstValues[i]);
    }
  }
  // Sentinel values should remain unchanged
  EXPECT_EQ(123_r, dstBuffer[0]);
  EXPECT_EQ(456_r, dstBuffer[9]);
}

TEST(Vec8i, StoreSelected_LargerConditionType) {
  Vec8i srcValues{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<int64_t, 8> condition{};
  std::array<int, 8> expectedValues{};
  alignas(16) std::array<int, 10> dstBuffer{};
  dstBuffer[0] = 123;
  dstBuffer[9] = 456;
  Span<int> dstValues{&dstBuffer[1], 8}; // Not aligned
  for (int64_t mask = 0; mask < 256; ++mask) { // For each possible set of conditions
    int expectedCount = 0;
    for (int i = 0; i < 8; ++i) {
      condition[i] = (mask & (1 << i)) ? -1 : 0; // true = -1, false = 0
      if (condition[i]) {
        expectedValues[expectedCount++] = srcValues[i];
      }
    }
    // Use a Vec8d condition and Vec8i srcValues.
    // In real code, this is mostly likely to occur when the condition is written as Vec8r.
    auto condition8d = ReinterpretCast<Vec8d>(Load<Vec8l>(condition.data()));
    auto countStored = StoreSelected(dstValues.data(), condition8d, srcValues);
    EXPECT_EQ(expectedCount, countStored);
    for (int i = 0; i < countStored; ++i) {
      EXPECT_EQ(expectedValues[i], dstValues[i]);
    }
  }
  // Sentinel values should remain unchanged
  EXPECT_EQ(123_r, dstBuffer[0]);
  EXPECT_EQ(456_r, dstBuffer[9]);
}

TEST(Vec8i, StoreTransposed) {
  alignas(alignof(Vec8i)) int result[26] = {};
  result[25] = 911; // Sentinel value
  auto* ptr = result + 1; // Not an aligned address
  Vec8i data[] = {
      Vec8i{1, 4, 7, 10, 13, 16, 19, 22},
      Vec8i{2, 5, 8, 11, 14, 17, 20, 23},
      Vec8i{3, 6, 9, 12, 15, 18, 21, 24}};
  StoreTransposed<1>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<int, 24>{1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
      Span(ptr, 24));
  StoreTransposed<2>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<int, 24>{1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
      Span(ptr, 24));
  StoreTransposed<3>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<int, 24>{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
      Span(ptr, 24));
  StoreTransposed<4>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<int, 24>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0}),
      Span(ptr, 24));
  StoreTransposed<5>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<int, 24>{1,  2,  3,  4, 5, 6, 7, 8, 9, 10, 11, 12,
                           13, 14, 15, 0, 0, 0, 0, 0, 0, 0,  0,  0}),
      Span(ptr, 24));
  StoreTransposed<6>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<int, 24>{1,  2,  3,  4,  5,  6,  7, 8, 9, 10, 11, 12,
                           13, 14, 15, 16, 17, 18, 0, 0, 0, 0,  0,  0}),
      Span(ptr, 24));
  StoreTransposed<7>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<int, 24>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 0,  0,  0}),
      Span(ptr, 24));
  StoreTransposed<8>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<int, 24>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
      Span(ptr, 24));
  StoreTransposed(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<int, 24>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
      Span(ptr, 24));
  EXPECT_EQ(0, result[0]); // No change
  EXPECT_EQ(911, result[25]); // No change
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec8i, Sub, -);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec8i, BitwiseXOR, ^);

TEST(Vec8i, ShiftLeft) {
  Vec8i v(1, 5, 7, 11, 17, 23, 29, 31);
  EXPECT_VEC8I(8, 40, 56, 88, 136, 184, 232, 248, v << 3);
  EXPECT_EQ(v, v << 0);
}

TEST(Vec8i, ShiftRight) {
  static constexpr int kSingleBitShift = 1;
  static constexpr int kHalfWidthShift = 8 * sizeof(Vec8i::Scalar) / 2;
  static constexpr int kHalfWidthScale = 1 << kHalfWidthShift;
  static constexpr int kIntMin = std::numeric_limits<int>::min();
  static constexpr int kNegativeSample = -100;
  static constexpr int kPositiveSample = 100;

  Vec8i v(8, 40, 56, 88, 136, 184, 232, 248);
  EXPECT_VEC8I(1, 5, 7, 11, 17, 23, 29, 31, ShiftRight<3>(v));
  EXPECT_EQ(v, ShiftRight<0>(v));
  // Shift by one catches logical shifts; half-width catches multi-bit sign fill.
  // clang-format off
  EXPECT_VEC8I(-1, -2, -4, -8, 0, 1, 2, 4, ShiftRight<kSingleBitShift>(Vec8i{-1, -3, -7, -15, 1, 2, 4, 8}));
  EXPECT_VEC8I(-1, kIntMin / kHalfWidthScale, -1, 0, -1, kIntMin / kHalfWidthScale, -1, 0, ShiftRight<kHalfWidthShift>(Vec8i{-1, kIntMin, kNegativeSample, kPositiveSample, -1, kIntMin, kNegativeSample, kPositiveSample}));
  // clang-format on
}
