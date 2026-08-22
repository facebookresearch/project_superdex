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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

using namespace mochi;
using namespace mochi::simd_test;

static_assert(std::is_trivially_copyable_v<Simd<real, 8>>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec8r);

#define EXPECT_VEC8R(e0, e1, e2, e3, e4, e5, e6, e7, actual)                    \
  {                                                                             \
    auto vActual = actual;                                                      \
    /* Testing for exact equality handles values like +/- infinity */           \
    EXPECT_TRUE(((e0) == vActual[0]) || NearEqual(real(e0), vActual[0], kEps)); \
    EXPECT_TRUE(((e1) == vActual[1]) || NearEqual(real(e1), vActual[1], kEps)); \
    EXPECT_TRUE(((e2) == vActual[2]) || NearEqual(real(e2), vActual[2], kEps)); \
    EXPECT_TRUE(((e3) == vActual[3]) || NearEqual(real(e3), vActual[3], kEps)); \
    EXPECT_TRUE(((e4) == vActual[4]) || NearEqual(real(e4), vActual[4], kEps)); \
    EXPECT_TRUE(((e5) == vActual[5]) || NearEqual(real(e5), vActual[5], kEps)); \
    EXPECT_TRUE(((e6) == vActual[6]) || NearEqual(real(e6), vActual[6], kEps)); \
    EXPECT_TRUE(((e7) == vActual[7]) || NearEqual(real(e7), vActual[7], kEps)); \
  }

TEST(Vec8r, Class) {
  static_assert(Vec8r::kIsSupported, "Should be supported");

#if MOCHI_USE_SIMD
  bool constexpr kExpectNativeSize = (MOCHI_ARCH_X64_AVX2 && !MOCHI_USE_DOUBLE_PRECISION);
  static_assert(Vec8r::kIsComposite == !kExpectNativeSize);
  static_assert(!Vec8r::kIsEmulated);
#else
  static_assert(!Vec8r::kIsComposite);
  static_assert(Vec8r::kIsEmulated);
#endif

  static_assert(sizeof(Vec8r) == sizeof(real) * 8);
  static_assert(alignof(Vec8r) == alignof(typename Vec8r::NativeType));
  static_assert(std::is_same_v<Vec8r::Scalar, real>);
  static_assert(Vec8r::kSize == 8);
  static_assert(Vec8r::size() == 8);

  // Construct from scalars
  // clang-format off
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, Vec8r(1_r));
  EXPECT_VEC8R(1_r, 2_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, Vec8r(1_r, 2_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 0_r, 0_r, 0_r, 0_r, 0_r, Vec8r(1_r, 2_r, 3_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 0_r, 0_r, 0_r, 0_r, Vec8r(1_r, 2_r, 3_r, 4_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 0_r, 0_r, 0_r, Vec8r(1_r, 2_r, 3_r, 4_r, 5_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 0_r, 0_r, Vec8r(1_r, 2_r, 3_r, 4_r, 5_r, 6_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 0_r, Vec8r(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, Vec8r(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r));
  // clang-format on

  // Construct from two Vec4r
  Vec4r const low{1_r, 2_r, 3_r, 4_r};
  Vec4r const high{5_r, 6_r, 7_r, 8_r};
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, Vec8r(low, high));

  // Implicit conversion from scalar
  Vec8r a = 2_r;
  EXPECT_VEC8R(2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r, 2_r, a);
  a = 3_r;
  EXPECT_VEC8R(3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r, a);

  // Copy construct
  Vec8r b{a};
  EXPECT_VEC8R(3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r, 3_r, b);

  // Copy assign
  a = Vec8r{4_r};
  b = a;
  EXPECT_VEC8R(4_r, 4_r, 4_r, 4_r, 4_r, 4_r, 4_r, 4_r, b);

  // Comparison
  EXPECT_EQ(true, (a == b));
  EXPECT_EQ(true, (a != Vec8r{}));
  EXPECT_EQ(false, (a != b));
  EXPECT_EQ(false, (a == Vec8r{}));

  // Unary operators
  using IVec = Simd<int, sizeof(Vec8r) / sizeof(int)>;
  Vec8r ones = ReinterpretCast<Vec8r>(IVec{-1}); // -nan can only be compared using int
  Vec8r zeros{0};
  EXPECT_EQ(ReinterpretCast<IVec>(ones), ReinterpretCast<IVec>(~zeros));
  EXPECT_EQ(ReinterpretCast<IVec>(zeros), ReinterpretCast<IVec>(~ones));
  EXPECT_VEC8R(
      -1_r,
      -2_r,
      -3_r,
      -4_r,
      -5_r,
      -6_r,
      -7_r,
      -8_r,
      -Vec8r(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r));

  // clang-format off

  // Binary operators
  a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  b = Vec8r{2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r};
  EXPECT_VEC8R(3_r, 5_r, 7_r, 9_r, 11_r, 13_r, 15_r, 17_r, a + b);
  EXPECT_VEC8R(4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, a + 3_r);
  EXPECT_VEC8R(4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, 3_r + a);
  EXPECT_VEC8R(-1_r, -1_r, -1_r, -1_r, -1_r, -1_r, -1_r, -1_r, a - b);
  EXPECT_VEC8R(-2_r, -1_r, 0_r, 1_r, 2_r, 3_r, 4_r, 5_r, a - 3_r);
  EXPECT_VEC8R(2_r, 1_r, 0_r, -1_r, -2_r, -3_r, -4_r, -5_r, 3_r - a);
  EXPECT_VEC8R(2_r, 6_r, 12_r, 20_r, 30_r, 42_r, 56_r, 72_r, a * b);
  EXPECT_VEC8R(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 21_r, 24_r, a * 3_r);
  EXPECT_VEC8R(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 21_r, 24_r, 3_r * a);
  EXPECT_VEC8R(2_r, 3_r / 2_r, 4_r / 3_r, 5_r / 4_r, 6_r / 5_r, 7_r / 6_r, 8_r / 7_r, 9_r / 8_r, b / a);
  EXPECT_VEC8R(1_r / 3_r, 2_r / 3_r, 3_r / 3_r, 4_r / 3_r, 5_r / 3_r, 6_r / 3_r, 7_r / 3_r, 8_r / 3_r, a / 3_r);
  EXPECT_VEC8R(3_r / 1_r, 3_r / 2_r, 3_r / 3_r, 3_r / 4_r, 3_r / 5_r, 3_r / 6_r, 3_r / 7_r, 3_r / 8_r, 3_r / a);
  EXPECT_EQ(a, a & ones);
  EXPECT_EQ(zeros, a & zeros);
  EXPECT_EQ(ReinterpretCast<IVec>(ones), ReinterpretCast<IVec>(Vec8r(a | ones)));
  EXPECT_EQ(a, a | zeros);
  EXPECT_EQ(~a, a ^ ones);
  EXPECT_EQ(a, a ^ zeros);

  // Update operators
  a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  a += b;
  EXPECT_VEC8R(3_r, 5_r, 7_r, 9_r, 11_r, 13_r, 15_r, 17_r, a);
  a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  a += 3_r;
  EXPECT_VEC8R(4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, a);
  a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  a -= b;
  EXPECT_VEC8R(-1_r, -1_r, -1_r, -1_r, -1_r, -1_r, -1_r, -1_r, a);
  a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  a -= 3_r;
  EXPECT_VEC8R(-2_r, -1_r, 0_r, 1_r, 2_r, 3_r, 4_r, 5_r, a);
  a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  a *= b;
  EXPECT_VEC8R(2_r, 6_r, 12_r, 20_r, 30_r, 42_r, 56_r, 72_r, a);
  a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  a *= 3_r;
  EXPECT_VEC8R(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 21_r, 24_r, a);
  a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  b = Vec8r{2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r};
  b /= a;
  EXPECT_VEC8R(2_r, 3_r / 2_r, 4_r / 3_r, 5_r / 4_r, 6_r / 5_r, 7_r / 6_r, 8_r / 7_r, 9_r / 8_r, b);
  a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  a /= 3_r;
  EXPECT_VEC8R(1_r / 3_r, 2_r / 3_r, 3_r / 3_r, 4_r / 3_r, 5_r / 3_r, 6_r / 3_r, 7_r / 3_r, 8_r / 3_r, a);
  a = b = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  b &= ones;
  EXPECT_EQ(a, b);
  a = b = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  b &= zeros;
  EXPECT_EQ(zeros, b);
  a = b = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  b |= ones;
  EXPECT_EQ(ReinterpretCast<IVec>(ones), ReinterpretCast<IVec>(Vec8r(b)));
  a = b = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  b |= zeros;
  EXPECT_EQ(a, b);
  a = b = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  b ^= ones;
  EXPECT_EQ(~a, b);
  a = b = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  b &= zeros;
  EXPECT_EQ(zeros, b);
  a = b = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  b &= ones;
  EXPECT_EQ(a, b);

  // clang-format on
}

MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, Abs, std::abs, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec8r, ACos, std::acos, -0.9_r, 0.9_r, kEps);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec8r, Add, +, kEps);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec8r, BitwiseAND, &);
MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(Vec8r);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec8r, ASin, std::asin, -0.9_r, 0.9_r, kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, ATan, std::atan, kEps);

TEST(Vec8r, AllTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  real zer = 0_r;
  real one = 0_r;
  auto allOnes = static_cast<uint64_t>(-1);
  memcpy(&one, &allOnes, sizeof(real));
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          for (int e = 0; e < 2; ++e) {
            for (int f = 0; f < 2; ++f) {
              for (int g = 0; g < 2; ++g) {
                for (int h = 0; h < 2; ++h) {
                  auto vec = Vec8r{
                      a ? one : zer,
                      b ? one : zer,
                      c ? one : zer,
                      d ? one : zer,
                      e ? one : zer,
                      f ? one : zer,
                      g ? one : zer,
                      h ? one : zer,
                  };
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

TEST(Vec8r, AnyTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  real zer = 0_r;
  real one = 0_r;
  auto allOnes = static_cast<uint64_t>(-1);
  memcpy(&one, &allOnes, sizeof(real));
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          for (int e = 0; e < 2; ++e) {
            for (int f = 0; f < 2; ++f) {
              for (int g = 0; g < 2; ++g) {
                for (int h = 0; h < 2; ++h) {
                  auto vec = Vec8r{
                      a ? one : zer,
                      b ? one : zer,
                      c ? one : zer,
                      d ? one : zer,
                      e ? one : zer,
                      f ? one : zer,
                      g ? one : zer,
                      h ? one : zer,
                  };
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

TEST(Vec8r, IsFinite) {
  constexpr real kQNaN = std::numeric_limits<real>::quiet_NaN();
  constexpr real kSNaN = std::numeric_limits<real>::signaling_NaN();
  EXPECT_TRUE(IsFinite(Vec8r{}));
  EXPECT_TRUE(IsFinite(Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r}));
  EXPECT_TRUE(IsFinite(Vec8r{std::numeric_limits<real>::min()}));
  EXPECT_TRUE(IsFinite(Vec8r{-std::numeric_limits<real>::min()}));
  EXPECT_TRUE(IsFinite(Vec8r{std::numeric_limits<real>::lowest()}));
  EXPECT_TRUE(IsFinite(Vec8r{-std::numeric_limits<real>::lowest()}));
  EXPECT_TRUE(IsFinite(Vec8r{std::numeric_limits<real>::max()}));
  EXPECT_TRUE(IsFinite(Vec8r{-std::numeric_limits<real>::max()}));
  EXPECT_FALSE(IsFinite(Vec8r(kInf, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r)));
  EXPECT_FALSE(IsFinite(Vec8r(0_r, -kInf, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r)));
  EXPECT_FALSE(IsFinite(Vec8r(0_r, 0_r, kQNaN, 0_r, 0_r, 0_r, 0_r, 0_r)));
  EXPECT_FALSE(IsFinite(Vec8r(0_r, 0_r, 0_r, kSNaN, 0_r, 0_r, 0_r, 0_r)));
  EXPECT_FALSE(IsFinite(Vec8r(0_r, 0_r, 0_r, 0_r, kInf, 0_r, 0_r, 0_r)));
  EXPECT_FALSE(IsFinite(Vec8r(0_r, 0_r, 0_r, 0_r, 0_r, -kInf, 0_r, 0_r)));
  EXPECT_FALSE(IsFinite(Vec8r(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, kQNaN, 0_r)));
  EXPECT_FALSE(IsFinite(Vec8r(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, kSNaN)));

  // clang-format off
  EXPECT_VEC8R(0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, VIsFinite(Vec8r(kInf, 0_r, -kInf, 0_r, kInf, 0_r, -kInf, 0_r)) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, VIsFinite(Vec8r(0_r, kQNaN, 0_r, kSNaN, 0_r, kQNaN, 0_r, kSNaN)) & Vec8r{1_r});
  // clang-format on
}

TEST(Vec8r, IsTrue) {
  auto a = SimdMask<Vec8r>(true, false, true, false, true, false, true, false);
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

// Test both Equal and NotEqual
TEST(Vec8r, Equal) {
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto b = a;
  auto c = Vec8r{1.1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto d = Vec8r{1_r, 2.1_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto e = Vec8r{1_r, 2_r, 3.1_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto f = Vec8r{1_r, 2_r, 3_r, 4.1_r, 5_r, 6_r, 7_r, 8_r};
  auto g = Vec8r{1_r, 2_r, 3_r, 4_r, 5.1_r, 6_r, 7_r, 8_r};
  auto h = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6.1_r, 7_r, 8_r};
  auto i = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7.1_r, 8_r};
  auto j = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8.1_r};

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

TEST(Vec8r, NearEqual) {
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto b = Vec8r{1.1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto c = Vec8r{1_r, 2.1_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto d = Vec8r{1_r, 2_r, 3.1_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto e = Vec8r{1_r, 2_r, 3_r, 4.1_r, 5_r, 6_r, 7_r, 8_r};
  auto f = Vec8r{1_r, 2_r, 3_r, 4_r, 5.1_r, 6_r, 7_r, 8_r};
  auto g = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6.1_r, 7_r, 8_r};
  auto h = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7.1_r, 8_r};
  auto i = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8.1_r};

  // Compare 1 components
  EXPECT_TRUE(NearEqual<1>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<1>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, b, 0.11f));
  EXPECT_TRUE(NearEqual<1>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, e, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, f, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, g, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, h, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, i, 0_r));

  // Compare 2 components
  EXPECT_TRUE(NearEqual<2>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<2>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, b, 0.11f));
  EXPECT_FALSE(NearEqual<2>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, c, 0.11f));
  EXPECT_TRUE(NearEqual<2>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, e, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, f, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, g, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, h, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, i, 0_r));

  // Compare 3 components
  EXPECT_TRUE(NearEqual<3>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<3>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, b, 0.11f));
  EXPECT_FALSE(NearEqual<3>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, c, 0.11f));
  EXPECT_FALSE(NearEqual<3>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, d, 0.11f));
  EXPECT_TRUE(NearEqual<3>(a, e, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, f, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, g, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, h, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, i, 0_r));

  // Compare 4 components
  EXPECT_TRUE(NearEqual<4>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<4>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<4>(a, b, 0.11f));
  EXPECT_FALSE(NearEqual<4>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<4>(a, c, 0.11f));
  EXPECT_FALSE(NearEqual<4>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<4>(a, d, 0.11f));
  EXPECT_FALSE(NearEqual<4>(a, e, 0_r));
  EXPECT_TRUE(NearEqual<4>(a, e, 0.11f));
  EXPECT_TRUE(NearEqual<4>(a, f, 0_r));
  EXPECT_TRUE(NearEqual<4>(a, g, 0_r));
  EXPECT_TRUE(NearEqual<4>(a, h, 0_r));
  EXPECT_TRUE(NearEqual<4>(a, i, 0_r));

  // Compare 5 components
  EXPECT_TRUE(NearEqual<5>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<5>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<5>(a, b, 0.11f));
  EXPECT_FALSE(NearEqual<5>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<5>(a, c, 0.11f));
  EXPECT_FALSE(NearEqual<5>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<5>(a, d, 0.11f));
  EXPECT_FALSE(NearEqual<5>(a, e, 0_r));
  EXPECT_TRUE(NearEqual<5>(a, e, 0.11f));
  EXPECT_FALSE(NearEqual<5>(a, f, 0_r));
  EXPECT_TRUE(NearEqual<5>(a, f, 0.11f));
  EXPECT_TRUE(NearEqual<5>(a, g, 0_r));
  EXPECT_TRUE(NearEqual<5>(a, h, 0_r));
  EXPECT_TRUE(NearEqual<5>(a, i, 0_r));

  // Compare 6 components
  EXPECT_TRUE(NearEqual<6>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<6>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<6>(a, b, 0.11f));
  EXPECT_FALSE(NearEqual<6>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<6>(a, c, 0.11f));
  EXPECT_FALSE(NearEqual<6>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<6>(a, d, 0.11f));
  EXPECT_FALSE(NearEqual<6>(a, e, 0_r));
  EXPECT_TRUE(NearEqual<6>(a, e, 0.11f));
  EXPECT_FALSE(NearEqual<6>(a, f, 0_r));
  EXPECT_TRUE(NearEqual<6>(a, f, 0.11f));
  EXPECT_FALSE(NearEqual<6>(a, g, 0_r));
  EXPECT_TRUE(NearEqual<6>(a, g, 0.11f));
  EXPECT_TRUE(NearEqual<6>(a, h, 0_r));
  EXPECT_TRUE(NearEqual<6>(a, i, 0_r));

  // Compare 7 components
  EXPECT_TRUE(NearEqual<7>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<7>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<7>(a, b, 0.11f));
  EXPECT_FALSE(NearEqual<7>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<7>(a, c, 0.11f));
  EXPECT_FALSE(NearEqual<7>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<7>(a, d, 0.11f));
  EXPECT_FALSE(NearEqual<7>(a, e, 0_r));
  EXPECT_TRUE(NearEqual<7>(a, e, 0.11f));
  EXPECT_FALSE(NearEqual<7>(a, f, 0_r));
  EXPECT_TRUE(NearEqual<7>(a, f, 0.11f));
  EXPECT_FALSE(NearEqual<7>(a, g, 0_r));
  EXPECT_TRUE(NearEqual<7>(a, g, 0.11f));
  EXPECT_FALSE(NearEqual<7>(a, h, 0_r));
  EXPECT_TRUE(NearEqual<7>(a, h, 0.11f));
  EXPECT_TRUE(NearEqual<7>(a, i, 0_r));

  // Compare 8 components
  EXPECT_TRUE(NearEqual(a, a, 0_r));
  EXPECT_FALSE(NearEqual(a, b, 0_r));
  EXPECT_TRUE(NearEqual(a, b, 0.11f));
  EXPECT_FALSE(NearEqual(a, c, 0_r));
  EXPECT_TRUE(NearEqual(a, c, 0.11f));
  EXPECT_FALSE(NearEqual(a, d, 0_r));
  EXPECT_TRUE(NearEqual(a, d, 0.11f));
  EXPECT_FALSE(NearEqual(a, e, 0_r));
  EXPECT_TRUE(NearEqual(a, e, 0.11f));
  EXPECT_FALSE(NearEqual(a, f, 0_r));
  EXPECT_TRUE(NearEqual(a, f, 0.11f));
  EXPECT_FALSE(NearEqual(a, g, 0_r));
  EXPECT_TRUE(NearEqual(a, g, 0.11f));
  EXPECT_FALSE(NearEqual(a, h, 0_r));
  EXPECT_TRUE(NearEqual(a, h, 0.11f));
  EXPECT_FALSE(NearEqual(a, i, 0_r));
  EXPECT_TRUE(NearEqual(a, i, 0.11f));
}

TEST(Vec8r, VNearEqual) {
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto b = Vec8r{1.1_r, 2_r, 3.1_r, 4_r, 5.1_r, 6_r, 7.1_r, 8_r};
  auto c = Vec8r{1.1_r, 2.2_r, 3.1_r, 4.2_r, 5.1_r, 6.2_r, 7.1_r, 8.2_r};
  // clang-format off
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, VNearEqual(a, a, Vec8r(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r)) & Vec8r(1_r));
  EXPECT_VEC8R(0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, VNearEqual(a, b, Vec8r(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 0_r, 1_r, 1_r, 1_r, 0_r, 1_r, VNearEqual(a, b, Vec8r(0.11_r, 0_r, 0_r, 0_r, 0.11_r, 0_r, 0_r, 0_r)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, VNearEqual(a, b, Vec8r(0.11_r, 0.11_r, 0.11_r, 0.11_r, 0.11_r, 0.11_r, 0.11_r, 0.11_r)) & Vec8r(1_r));
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, VNearEqual(a, c, Vec8r(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r)) & Vec8r(1_r));
  EXPECT_VEC8R(0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r, VNearEqual(a, c, Vec8r(0_r, 0_r, 0.11_r, 0.11_r, 0_r, 0_r, 0.11_r, 0_r)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, VNearEqual(a, c, Vec8r(0.11_r, 0.11_r, 0.11_r, 0.11_r, 0.11_r, 0.11_r, 0.11_r, 0.11_r)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 0_r, 1_r, 0_r, VNearEqual(a, c, Vec8r(0.21_r, 0.21_r, 0.21_r, 0.21_r, 0.11_r, 0.11_r, 0.11_r, 0.11_r)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, VNearEqual(a, c, Vec8r(0.21_r, 0.21_r, 0.21_r, 0.21_r, 0.21_r, 0.21_r, 0.21_r, 0.21_r)) & Vec8r(1_r));
  // clang-format on
}

TEST(Vec8r, NearZero) {
  auto a = Vec8r{0.1_r, -0.2_r, 0.3_r, -0.4_r, 0.5_r, -0.6_r, 0.7_r, -0.8_r};

  // Compare 1 components
  EXPECT_FALSE(NearZero<1>(a, 0_r));
  EXPECT_TRUE(NearZero<1>(a, 0.11f));

  // Compare 2 components
  EXPECT_FALSE(NearZero<2>(a, 0_r));
  EXPECT_FALSE(NearZero<2>(a, 0.11f));
  EXPECT_TRUE(NearZero<2>(a, 0.21f));

  // Compare 3 components
  EXPECT_FALSE(NearZero<3>(a, 0_r));
  EXPECT_FALSE(NearZero<3>(a, 0.11f));
  EXPECT_FALSE(NearZero<3>(a, 0.21f));
  EXPECT_TRUE(NearZero<3>(a, 0.31f));

  // Compare 4 components
  EXPECT_FALSE(NearZero<4>(a, 0_r));
  EXPECT_FALSE(NearZero<4>(a, 0.11f));
  EXPECT_FALSE(NearZero<4>(a, 0.21f));
  EXPECT_FALSE(NearZero<4>(a, 0.31f));
  EXPECT_TRUE(NearZero<4>(a, 0.41f));

  // Compare 5 components
  EXPECT_FALSE(NearZero<5>(a, 0_r));
  EXPECT_FALSE(NearZero<5>(a, 0.11f));
  EXPECT_FALSE(NearZero<5>(a, 0.21f));
  EXPECT_FALSE(NearZero<5>(a, 0.31f));
  EXPECT_FALSE(NearZero<5>(a, 0.41f));
  EXPECT_TRUE(NearZero<5>(a, 0.51f));

  // Compare 6 components
  EXPECT_FALSE(NearZero<6>(a, 0_r));
  EXPECT_FALSE(NearZero<6>(a, 0.11f));
  EXPECT_FALSE(NearZero<6>(a, 0.21f));
  EXPECT_FALSE(NearZero<6>(a, 0.31f));
  EXPECT_FALSE(NearZero<6>(a, 0.41f));
  EXPECT_FALSE(NearZero<6>(a, 0.51f));
  EXPECT_TRUE(NearZero<6>(a, 0.61f));

  // Compare 7 components
  EXPECT_FALSE(NearZero<7>(a, 0_r));
  EXPECT_FALSE(NearZero<7>(a, 0.11f));
  EXPECT_FALSE(NearZero<7>(a, 0.21f));
  EXPECT_FALSE(NearZero<7>(a, 0.31f));
  EXPECT_FALSE(NearZero<7>(a, 0.41f));
  EXPECT_FALSE(NearZero<7>(a, 0.51f));
  EXPECT_FALSE(NearZero<7>(a, 0.61f));
  EXPECT_TRUE(NearZero<7>(a, 0.71f));

  // Compare 8 components
  EXPECT_FALSE(NearZero(a, 0_r));
  EXPECT_FALSE(NearZero(a, 0.11f));
  EXPECT_FALSE(NearZero(a, 0.21f));
  EXPECT_FALSE(NearZero(a, 0.31f));
  EXPECT_FALSE(NearZero(a, 0.41f));
  EXPECT_FALSE(NearZero(a, 0.51f));
  EXPECT_FALSE(NearZero(a, 0.61f));
  EXPECT_FALSE(NearZero(a, 0.71f));
  EXPECT_TRUE(NearZero(a, 0.81f));
}

TEST(Vec8r, VNearZero) {
  auto z = Vec8r{};
  auto a = Vec8r{0.1_r, -0.2_r, 0.3_r, -0.4_r, 0.5_r, -0.6_r, 0.7_r, -0.8_r};
  // clang-format off
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, VNearZero(z, Vec8r(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r)) & Vec8r(1_r));
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, VNearZero(a, Vec8r(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 0_r, 0_r, 1_r, 1_r, 0_r, 0_r, VNearZero(a, Vec8r(0.11_r, 0.21_r, 0_r, 0_r, 0.51_r, 0.61_r, 0_r, 0_r)) & Vec8r(1_r));
  EXPECT_VEC8R(0_r, 0_r, 1_r, 1_r, 0_r, 0_r, 1_r, 1_r, VNearZero(a, Vec8r(0_r, 0_r, 0.31_r, 0.41_r, 0_r, 0_r, 0.71_r, 0.81_r)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, VNearZero(a, Vec8r(0.11f)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, VNearZero(a, Vec8r(0.21f)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, VNearZero(a, Vec8r(0.31f)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 0_r, 0_r, 0_r, 0_r, VNearZero(a, Vec8r(0.41f)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 0_r, 0_r, 0_r, VNearZero(a, Vec8r(0.51f)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 0_r, 0_r, VNearZero(a, Vec8r(0.61f)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 0_r, VNearZero(a, Vec8r(0.71f)) & Vec8r(1_r));
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, VNearZero(a, Vec8r(0.81f)) & Vec8r(1_r));
  // clang-format on
}

TEST(Vec8r, Broadcast) {
  // Broadcast scalar
  auto v = Broadcast<Vec8r>(0_r);
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, v);
  v = Broadcast<Vec8r>(1.1_r);
  EXPECT_VEC8R(1.1_r, 1.1_r, 1.1_r, 1.1_r, 1.1_r, 1.1_r, 1.1_r, 1.1_r, v);

  // Broadcast from address
  real const s = 2.2_r;
  v = Broadcast<Vec8r>(&s);
  EXPECT_VEC8R(2.2_r, 2.2_r, 2.2_r, 2.2_r, 2.2_r, 2.2_r, 2.2_r, 2.2_r, v);

  // Broadcast ith element
  v = Vec8r{0.1_r, 0.2_r, 0.3_r, 0.4_r, 0.5_r, 0.6_r, 0.7_r, 0.8_r};
  EXPECT_VEC8R(0.1_r, 0.1_r, 0.1_r, 0.1_r, 0.1_r, 0.1_r, 0.1_r, 0.1_r, Broadcast<0>(v));
  EXPECT_VEC8R(0.2_r, 0.2_r, 0.2_r, 0.2_r, 0.2_r, 0.2_r, 0.2_r, 0.2_r, Broadcast<1>(v));
  EXPECT_VEC8R(0.3_r, 0.3_r, 0.3_r, 0.3_r, 0.3_r, 0.3_r, 0.3_r, 0.3_r, Broadcast<2>(v));
  EXPECT_VEC8R(0.4_r, 0.4_r, 0.4_r, 0.4_r, 0.4_r, 0.4_r, 0.4_r, 0.4_r, Broadcast<3>(v));
  EXPECT_VEC8R(0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r, Broadcast<4>(v));
  EXPECT_VEC8R(0.6_r, 0.6_r, 0.6_r, 0.6_r, 0.6_r, 0.6_r, 0.6_r, 0.6_r, Broadcast<5>(v));
  EXPECT_VEC8R(0.7_r, 0.7_r, 0.7_r, 0.7_r, 0.7_r, 0.7_r, 0.7_r, 0.7_r, Broadcast<6>(v));
  EXPECT_VEC8R(0.8_r, 0.8_r, 0.8_r, 0.8_r, 0.8_r, 0.8_r, 0.8_r, 0.8_r, Broadcast<7>(v));

  // Broadcast ith element
  EXPECT_VEC8R(0.1_r, 0.1_r, 0.1_r, 0.1_r, 0.1_r, 0.1_r, 0.1_r, 0.1_r, Broadcast(v, 0));
  EXPECT_VEC8R(0.2_r, 0.2_r, 0.2_r, 0.2_r, 0.2_r, 0.2_r, 0.2_r, 0.2_r, Broadcast(v, 1));
  EXPECT_VEC8R(0.3_r, 0.3_r, 0.3_r, 0.3_r, 0.3_r, 0.3_r, 0.3_r, 0.3_r, Broadcast(v, 2));
  EXPECT_VEC8R(0.4_r, 0.4_r, 0.4_r, 0.4_r, 0.4_r, 0.4_r, 0.4_r, 0.4_r, Broadcast(v, 3));
  EXPECT_VEC8R(0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r, 0.5_r, Broadcast(v, 4));
  EXPECT_VEC8R(0.6_r, 0.6_r, 0.6_r, 0.6_r, 0.6_r, 0.6_r, 0.6_r, 0.6_r, Broadcast(v, 5));
  EXPECT_VEC8R(0.7_r, 0.7_r, 0.7_r, 0.7_r, 0.7_r, 0.7_r, 0.7_r, 0.7_r, Broadcast(v, 6));
  EXPECT_VEC8R(0.8_r, 0.8_r, 0.8_r, 0.8_r, 0.8_r, 0.8_r, 0.8_r, 0.8_r, Broadcast(v, 7));
}

MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec8r, Clamp, Clamp, kEps);

TEST(Vec8r, Cos) {
  TestSimdTrigFunction<Vec8r>(
      [](real x) { return std::cos(x); }, [](Vec8r x) { return Cos(x); }, GetTrigTestValues());
}

MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec8r, Div, /, kEps);

TEST(Vec8r, VDot) {
  // Currently only supports 8-component dot product

  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto b = Vec8r{1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r};
  EXPECT_VEC8R(36_r, 36_r, 36_r, 36_r, 36_r, 36_r, 36_r, 36_r, (VDot(a, b)));

  a = Vec8r{1_r, -2_r, 3_r, -4_r, 4_r, -3_r, 2_r, -1_r};
  b = Vec8r{1_r, 0.5_r, 1_r, 0.75_r, 1_r, 1_r, 0.5_r, -3_r};
  EXPECT_VEC8R(5_r, 5_r, 5_r, 5_r, 5_r, 5_r, 5_r, 5_r, (VDot(a, b)));
}

// Test bot VEqual and VNotEqual
template <class V>
static void
ExpectVEqual(bool e0, bool e1, bool e2, bool e3, bool e4, bool e5, bool e6, bool e7, V a, V b) {
  EXPECT_VEC8R(
      e0 ? 1_r : 0_r,
      e1 ? 1_r : 0_r,
      e2 ? 1_r : 0_r,
      e3 ? 1_r : 0_r,
      e4 ? 1_r : 0_r,
      e5 ? 1_r : 0_r,
      e6 ? 1_r : 0_r,
      e7 ? 1_r : 0_r,
      VEqual(a, b) & Vec8r{1_r});
  EXPECT_VEC8R(
      e0 ? 0_r : 1_r,
      e1 ? 0_r : 1_r,
      e2 ? 0_r : 1_r,
      e3 ? 0_r : 1_r,
      e4 ? 0_r : 1_r,
      e5 ? 0_r : 1_r,
      e6 ? 0_r : 1_r,
      e7 ? 0_r : 1_r,
      VNotEqual(a, b) & Vec8r{1_r});
}

TEST(Vec8r, VEqual) {
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto b = Vec8r{1_r, 9_r, 3_r, 9_r, 5_r, 9_r, 7_r, 9_r};
  auto c = Vec8r{9_r, 2_r, 9_r, 4_r, 9_r, 6_r, 9_r, 8_r};
  ExpectVEqual(1, 1, 1, 1, 1, 1, 1, 1, a, a);
  ExpectVEqual(1, 0, 1, 0, 1, 0, 1, 0, a, b);
  ExpectVEqual(0, 1, 0, 1, 0, 1, 0, 1, a, c);
}

TEST(Vec8r, Norm) {
  Vec8r const kValues[] = {
      Vec8r{0_r},
      Vec8r{2_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r},
      Vec8r{0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 2_r},
      Vec8r{1_r, -2_r, 3_r, -4_r, 5_r, -6_r, 7_r, -8_r}};
  for (auto v : kValues) {
    // 8 components (only size currently supported for Vec8r)
    real nsqr = 0_r;
    for (int i = 0; i < 8; ++i) {
      nsqr += Sqr(Get(v, i));
    }
    auto norm = std::sqrt(nsqr);
    EXPECT_NEAR_EQ(Vec8r(norm), VNorm(v));
    EXPECT_NEAR_EQ(Vec8r(nsqr), VNormSqr(v));
    EXPECT_NEAR_EQ(norm, Norm(v));
    EXPECT_NEAR_EQ(nsqr, NormSqr(v));
  }
}

TEST(Vec8r, Normalize) {
  auto v = Vec8r{1_r, -2_r, 3_r, -4_r, 5_r, -6_r, 7_r, -8_r};
  EXPECT_TRUE(NearEqual(Vec8r(), Normalize(Vec8r{}))); // Zero in, zero out
  EXPECT_TRUE(NearEqual(v / Sqrt(204_r), Normalize(v))); // 4 components
  EXPECT_TRUE(NearEqual(v / Sqrt(204_r), Normalize(v, 204_r))); // 4 components
}

TEST(Vec8r, get) {
  Vec8r a = {1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};

  EXPECT_NEAR(1_r, Get0(a), kEps);
  EXPECT_NEAR(1_r, Get<0>(a), kEps);
  EXPECT_NEAR(2_r, Get<1>(a), kEps);
  EXPECT_NEAR(3_r, Get<2>(a), kEps);
  EXPECT_NEAR(4_r, Get<3>(a), kEps);
  EXPECT_NEAR(5_r, Get<4>(a), kEps);
  EXPECT_NEAR(6_r, Get<5>(a), kEps);
  EXPECT_NEAR(7_r, Get<6>(a), kEps);
  EXPECT_NEAR(8_r, Get<7>(a), kEps);

  EXPECT_NEAR(1_r, Get(a, 0), kEps);
  EXPECT_NEAR(2_r, Get(a, 1), kEps);
  EXPECT_NEAR(3_r, Get(a, 2), kEps);
  EXPECT_NEAR(4_r, Get(a, 3), kEps);
  EXPECT_NEAR(5_r, Get(a, 4), kEps);
  EXPECT_NEAR(6_r, Get(a, 5), kEps);
  EXPECT_NEAR(7_r, Get(a, 6), kEps);
  EXPECT_NEAR(8_r, Get(a, 7), kEps);
}

TEST(Vec8r, GetHalf) {
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto low = GetHalf<0>(a);
  auto high = GetHalf<1>(a);
  EXPECT_NEAR_EQ(1_r, Get(low, 0));
  EXPECT_NEAR_EQ(2_r, Get(low, 1));
  EXPECT_NEAR_EQ(3_r, Get(low, 2));
  EXPECT_NEAR_EQ(4_r, Get(low, 3));
  EXPECT_NEAR_EQ(5_r, Get(high, 0));
  EXPECT_NEAR_EQ(6_r, Get(high, 1));
  EXPECT_NEAR_EQ(7_r, Get(high, 2));
  EXPECT_NEAR_EQ(8_r, Get(high, 3));
}

TEST(Vec8r, Set) {
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  EXPECT_VEC8R(9_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, Set<0>(a, 9_r));
  EXPECT_VEC8R(1_r, 9_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, Set<1>(a, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 9_r, 4_r, 5_r, 6_r, 7_r, 8_r, Set<2>(a, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 9_r, 5_r, 6_r, 7_r, 8_r, Set<3>(a, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 9_r, 6_r, 7_r, 8_r, Set<4>(a, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 9_r, 7_r, 8_r, Set<5>(a, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 9_r, 8_r, Set<6>(a, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 9_r, Set<7>(a, 9_r));

  EXPECT_VEC8R(9_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, Set(a, 0, 9_r));
  EXPECT_VEC8R(1_r, 9_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, Set(a, 1, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 9_r, 4_r, 5_r, 6_r, 7_r, 8_r, Set(a, 2, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 9_r, 5_r, 6_r, 7_r, 8_r, Set(a, 3, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 9_r, 6_r, 7_r, 8_r, Set(a, 4, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 9_r, 7_r, 8_r, Set(a, 5, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 9_r, 8_r, Set(a, 6, 9_r));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 9_r, Set(a, 7, 9_r));
}

TEST(Vec8r, Greater) {
  auto a = Vec8r{1_r, 4_r, 5_r, 8_r, 9_r, 12_r, 13_r, 16_r};
  auto b = Vec8r{2_r, 3_r, 6_r, 7_r, 10_r, 11_r, 14_r, 15_r};
  EXPECT_VEC8R(0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, (a > b) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, (b > a) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (a > a) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, (a > Vec8r{4.5_r}) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 0_r, 1_r, 1_r, 1_r, 1_r, 1_r, (a > Vec8r{5_r}) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Vec8r{4.5_r} > a) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Vec8r{5_r} > a) & Vec8r{1_r});
}

TEST(Vec8r, GreaterEqual) {
  auto a = Vec8r{1_r, 4_r, 5_r, 8_r, 9_r, 12_r, 13_r, 16_r};
  auto b = Vec8r{2_r, 3_r, 6_r, 7_r, 10_r, 11_r, 14_r, 15_r};
  EXPECT_VEC8R(0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, (a >= b) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, (b >= a) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, (a >= a) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, (a >= Vec8r{4.5_r}) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, (a >= Vec8r{5_r}) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Vec8r{4.5_r} >= a) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Vec8r{5_r} >= a) & Vec8r{1_r});
}

TEST(Vec8r, HMax) {
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  EXPECT_NEAR_EQ(2_r, HMax<2>(a));
  EXPECT_NEAR_EQ(3_r, HMax<3>(a));
  EXPECT_NEAR_EQ(4_r, HMax<4>(a));
  EXPECT_NEAR_EQ(5_r, HMax<5>(a));
  EXPECT_NEAR_EQ(6_r, HMax<6>(a));
  EXPECT_NEAR_EQ(7_r, HMax<7>(a));
  EXPECT_NEAR_EQ(8_r, HMax(a));
}

TEST(Vec8r, HMin) {
  auto a = Vec8r{8_r, 7_r, 6_r, 5_r, 4_r, 3_r, 2_r, 1_r};
  EXPECT_NEAR_EQ(7_r, HMin<2>(a));
  EXPECT_NEAR_EQ(6_r, HMin<3>(a));
  EXPECT_NEAR_EQ(5_r, HMin<4>(a));
  EXPECT_NEAR_EQ(4_r, HMin<5>(a));
  EXPECT_NEAR_EQ(3_r, HMin<6>(a));
  EXPECT_NEAR_EQ(2_r, HMin<7>(a));
  EXPECT_NEAR_EQ(1_r, HMin(a));
}

TEST(Vec8r, HSum) {
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  EXPECT_NEAR_EQ(3_r, HSum<2>(a));
  EXPECT_NEAR_EQ(6_r, HSum<3>(a));
  EXPECT_NEAR_EQ(10_r, HSum<4>(a));
  EXPECT_NEAR_EQ(15_r, HSum<5>(a));
  EXPECT_NEAR_EQ(21_r, HSum<6>(a));
  EXPECT_NEAR_EQ(28_r, HSum<7>(a));
  EXPECT_NEAR_EQ(36_r, HSum(a));
}

// HProd is not implemented for Vec8f (only Vec2d/Vec4d/Vec4f, per simd.h support matrix).
#if MOCHI_USE_DOUBLE_PRECISION
TEST(Vec8r, HProd) {
  auto a = Vec8r{2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r};
  EXPECT_NEAR_EQ(6_r, HProd<2>(a));
  EXPECT_NEAR_EQ(24_r, HProd<3>(a));
  EXPECT_NEAR_EQ(120_r, HProd<4>(a));
  EXPECT_NEAR_EQ(720_r, HProd<5>(a));
  EXPECT_NEAR_EQ(5040_r, HProd<6>(a));
  EXPECT_NEAR_EQ(40320_r, HProd<7>(a));
  EXPECT_NEAR_EQ(362880_r, HProd(a));
}
#endif // MOCHI_USE_DOUBLE_PRECISION

TEST(Vec8r, Lerp) {
  // Note: Lerp does not clamp the 't' parameter
  auto const a = Vec8r{10_r, 20_r, 30_r, 40_r, 50_r, 60_r, 70_r, 80_r};
  auto const b = Vec8r{20_r, 40_r, 60_r, 80_r, 100_r, 120_r, 140_r, 160_r};
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, Lerp(a, b, -1_r));
  EXPECT_VEC8R(5_r, 10_r, 15_r, 20_r, 25_r, 30_r, 35_r, 40_r, Lerp(a, b, -0.5_r));
  EXPECT_VEC8R(10_r, 20_r, 30_r, 40_r, 50_r, 60_r, 70_r, 80_r, Lerp(a, b, 0_r));
  EXPECT_VEC8R(15_r, 30_r, 45_r, 60_r, 75_r, 90_r, 105_r, 120_r, Lerp(a, b, 0.5_r));
  EXPECT_VEC8R(20_r, 40_r, 60_r, 80_r, 100_r, 120_r, 140_r, 160_r, Lerp(a, b, 1_r));
  EXPECT_VEC8R(25_r, 50_r, 75_r, 100_r, 125_r, 150_r, 175_r, 200_r, Lerp(a, b, 1.5_r));
}

TEST(Vec8r, Less) {
  auto a = Vec8r{1_r, 4_r, 5_r, 8_r, 9_r, 12_r, 13_r, 16_r};
  auto b = Vec8r{2_r, 3_r, 6_r, 7_r, 10_r, 11_r, 14_r, 15_r};
  EXPECT_VEC8R(0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, (b < a) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, (a < b) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (a < a) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (a < Vec8r{4.5_r}) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (a < Vec8r{5_r}) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, (Vec8r{4.5_r} < a) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 0_r, 1_r, 1_r, 1_r, 1_r, 1_r, (Vec8r{5_r} < a) & Vec8r{1_r});
}

TEST(Vec8r, LessEqual) {
  auto a = Vec8r{1_r, 4_r, 5_r, 8_r, 9_r, 12_r, 13_r, 16_r};
  auto b = Vec8r{2_r, 3_r, 6_r, 7_r, 10_r, 11_r, 14_r, 15_r};
  EXPECT_VEC8R(0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, (b <= a) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, (a <= b) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, (a <= a) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (a <= Vec8r{4.5_r}) & Vec8r{1_r});
  EXPECT_VEC8R(1_r, 1_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, (a <= Vec8r{5_r}) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, (Vec8r{4.5_r} <= a) & Vec8r{1_r});
  EXPECT_VEC8R(0_r, 0_r, 1_r, 1_r, 1_r, 1_r, 1_r, 1_r, (Vec8r{5_r} <= a) & Vec8r{1_r});
}

TEST(Vec8r, Load) {
  alignas(alignof(Vec8r)) real const values[] = {0_r, 1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Load<0, Vec8r>(nullptr)));
  EXPECT_VEC8R(1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Load<1, Vec8r>(values + 1)));
  EXPECT_VEC8R(1_r, 2_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Load<2, Vec8r>(values + 1)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Load<3, Vec8r>(values + 1)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 0_r, 0_r, 0_r, 0_r, (Load<4, Vec8r>(values + 1)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 0_r, 0_r, 0_r, (Load<5, Vec8r>(values + 1)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 0_r, 0_r, (Load<6, Vec8r>(values + 1)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 0_r, (Load<7, Vec8r>(values + 1)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, (Load<8, Vec8r>(values + 1)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, (Load<Vec8r>(values + 1)));

  EXPECT_VEC8R(1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Load<Vec8r>(values + 1, 1)));
  EXPECT_VEC8R(1_r, 2_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Load<Vec8r>(values + 1, 2)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 0_r, 0_r, 0_r, 0_r, 0_r, (Load<Vec8r>(values + 1, 3)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 0_r, 0_r, 0_r, 0_r, (Load<Vec8r>(values + 1, 4)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 0_r, 0_r, 0_r, (Load<Vec8r>(values + 1, 5)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 0_r, 0_r, (Load<Vec8r>(values + 1, 6)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 0_r, (Load<Vec8r>(values + 1, 7)));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, (Load<Vec8r>(values + 1, 8)));
}

TEST(Vec8r, LoadIndexed) {
  // clang-format off
  alignas(alignof(Vec8r)) real const values[] = {0_r, 1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  EXPECT_VEC8R(2_r, 4_r, 6_r, 8_r, 1_r, 3_r, 5_r, 7_r, LoadIndexed<Vec8r>(&values[1], Vec8i(1, 3, 5, 7, 0, 2, 4, 6)));
  // clang-format on
}

TEST(Vec8r, LoadTransposed) {
  alignas(alignof(Vec8r)) real const values[] = {
      0_r,  1_r,  2_r,  3_r,  4_r,  5_r,  6_r,  7_r,  8_r,  9_r,  10_r, 11_r, 12_r,
      13_r, 14_r, 15_r, 16_r, 17_r, 18_r, 19_r, 20_r, 21_r, 22_r, 23_r, 24_r};
  auto const* ptr = values + 1; // Not an aligned address
  Vec8r loaded[3] = {};
  LoadTransposed<1>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8R(1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, loaded[0]);
  EXPECT_VEC8R(2_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, loaded[1]);
  EXPECT_VEC8R(3_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, loaded[2]);
  LoadTransposed<2>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8R(1_r, 4_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, loaded[0]);
  EXPECT_VEC8R(2_r, 5_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, loaded[1]);
  EXPECT_VEC8R(3_r, 6_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, loaded[2]);
  LoadTransposed<3>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8R(1_r, 4_r, 7_r, 0_r, 0_r, 0_r, 0_r, 0_r, loaded[0]);
  EXPECT_VEC8R(2_r, 5_r, 8_r, 0_r, 0_r, 0_r, 0_r, 0_r, loaded[1]);
  EXPECT_VEC8R(3_r, 6_r, 9_r, 0_r, 0_r, 0_r, 0_r, 0_r, loaded[2]);
  LoadTransposed<4>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8R(1_r, 4_r, 7_r, 10_r, 0_r, 0_r, 0_r, 0_r, loaded[0]);
  EXPECT_VEC8R(2_r, 5_r, 8_r, 11_r, 0_r, 0_r, 0_r, 0_r, loaded[1]);
  EXPECT_VEC8R(3_r, 6_r, 9_r, 12_r, 0_r, 0_r, 0_r, 0_r, loaded[2]);
  LoadTransposed<5>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8R(1_r, 4_r, 7_r, 10_r, 13_r, 0_r, 0_r, 0_r, loaded[0]);
  EXPECT_VEC8R(2_r, 5_r, 8_r, 11_r, 14_r, 0_r, 0_r, 0_r, loaded[1]);
  EXPECT_VEC8R(3_r, 6_r, 9_r, 12_r, 15_r, 0_r, 0_r, 0_r, loaded[2]);
  LoadTransposed<6>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8R(1_r, 4_r, 7_r, 10_r, 13_r, 16_r, 0_r, 0_r, loaded[0]);
  EXPECT_VEC8R(2_r, 5_r, 8_r, 11_r, 14_r, 17_r, 0_r, 0_r, loaded[1]);
  EXPECT_VEC8R(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 0_r, 0_r, loaded[2]);
  LoadTransposed<7>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8R(1_r, 4_r, 7_r, 10_r, 13_r, 16_r, 19_r, 0_r, loaded[0]);
  EXPECT_VEC8R(2_r, 5_r, 8_r, 11_r, 14_r, 17_r, 20_r, 0_r, loaded[1]);
  EXPECT_VEC8R(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 21_r, 0_r, loaded[2]);
  LoadTransposed<8>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8R(1_r, 4_r, 7_r, 10_r, 13_r, 16_r, 19_r, 22_r, loaded[0]);
  EXPECT_VEC8R(2_r, 5_r, 8_r, 11_r, 14_r, 17_r, 20_r, 23_r, loaded[1]);
  EXPECT_VEC8R(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 21_r, 24_r, loaded[2]);
  LoadTransposed(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC8R(1_r, 4_r, 7_r, 10_r, 13_r, 16_r, 19_r, 22_r, loaded[0]);
  EXPECT_VEC8R(2_r, 5_r, 8_r, 11_r, 14_r, 17_r, 20_r, 23_r, loaded[1]);
  EXPECT_VEC8R(3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 21_r, 24_r, loaded[2]);
}

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, Floor, ([](auto a) { return std::floor(a); }), kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, FastRound, ([](auto a) { return std::round(a); }), kEps); // The standard test values have no exact ties
MOCHI_SIMD_TEST_BINARY_FN_NEAR(Vec8r, Max, ([](auto a, auto b) { return std::max(a, b); }), kEps);
MOCHI_SIMD_TEST_BINARY_FN_NEAR(Vec8r, Min, ([](auto a, auto b) { return std::min(a, b); }), kEps);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec8r, Mul, *, kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec8r, MulAdd, ([](auto a, auto b, auto c) { return a * b + c; }), kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec8r, MulSub, ([](auto a, auto b, auto c) { return a * b - c; }), kEps);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec8r, Neg, -);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec8r, NegMulAdd, ([](auto a, auto b, auto c) { return -(a * b) + c; }), kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec8r, NegMulSub, ([](auto a, auto b, auto c) { return -(a * b) - c; }), kEps);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec8r, BitwiseNOT, ~);
// clang-format on

TEST(Vec8r, VNotEqual) {
  // clang-format off
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto b = Vec8r{1_r, 1_r, 3_r, 3_r, 5_r, 5_r, 7_r, 7_r};
  EXPECT_VEC8R(0_r, 1_r, 0_r, 1_r, 0_r, 1_r, 0_r, 1_r, VNotEqual(a, b) & Broadcast<Vec8r>(1_r));
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, VNotEqual(a, a) & Broadcast<Vec8r>(1_r));
  // clang-format on
}

#if MOCHI_USE_SIMD // SignBitMask is a utility only implemented in native SIMD types
TEST(Vec8r, SignBitMask) {
  auto const signBit = Vec8r::SignBitMask(); // -0_r (all bits set to 0 except the sign bit)
  auto const a = Vec8r{1_r, -2_r, 3_r, -4_r, 5_r, -6_r, 7_r, -8_r};
  EXPECT_EQ(Vec8r{}, signBit); // 0_r == -0_r
  EXPECT_EQ(-a, a ^ signBit); // flip sign
}
#endif

// clang-format off
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec8r, BitwiseOR, |);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec8r, LogicalAND, &&, &);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec8r, LogicalOR, ||, |);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, RcpApprox, ([](auto a) { return 1 / a; }), 1e-3f);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, RcpSqrtApprox, ([](auto a) { return 1 / std::sqrt(a); }), 1e-3f);
// clang-format on

TEST(Vec8r, Select) {
  auto a = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  auto b = a + 0.1_r;
  for (int b0 = 0; b0 < 2; ++b0) {
    for (int b1 = 0; b1 < 2; ++b1) {
      for (int b2 = 0; b2 < 2; ++b2) {
        for (int b3 = 0; b3 < 2; ++b3) {
          for (int b4 = 0; b4 < 2; ++b4) {
            for (int b5 = 0; b5 < 2; ++b5) {
              for (int b6 = 0; b6 < 2; ++b6) {
                for (int b7 = 0; b7 < 2; ++b7) {
                  auto expected = Vec8r{
                      b0 ? Get(a, 0) : Get(b, 0),
                      b1 ? Get(a, 1) : Get(b, 1),
                      b2 ? Get(a, 2) : Get(b, 2),
                      b3 ? Get(a, 3) : Get(b, 3),
                      b4 ? Get(a, 4) : Get(b, 4),
                      b5 ? Get(a, 5) : Get(b, 5),
                      b6 ? Get(a, 6) : Get(b, 6),
                      b7 ? Get(a, 7) : Get(b, 7)};
                  auto actual =
                      Select(SimdMask<Vec8r>(!!b0, !!b1, !!b2, !!b3, !!b4, !!b5, !!b6, !!b7), a, b);
                  EXPECT_NEAR_EQ(expected, actual);
                }
              }
            }
          }
        }
      }
    }
  }
  // More typical usage
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, Select(a < b, a, b)); // Min(a, b)
  EXPECT_VEC8R(
      1.1_r, 2.1_r, 3.1_r, 4.1_r, 5.1_r, 6.1_r, 7.1_r, 8.1_r, Select(a > b, a, b)); // Max(a, b)
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5.1_r, 6.1_r, 7.1_r, 8.1_r, Select(a <= Vec8r{4_r}, a, b));
  EXPECT_VEC8R(1.1_r, 2.1_r, 3.1_r, 4.1_r, 5_r, 6_r, 7_r, 8_r, Select(a > Vec8r{4_r}, a, b));
}

TEST(Vec8r, SimdMask) {
  // clang-format off
  auto v = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  bool constexpr f = false;
  bool constexpr t = true;
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, v & SimdMask<Vec8r>(f, f, f, f, f, f, f, f));
  EXPECT_VEC8R(1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, v & SimdMask<Vec8r>(t, f, f, f, f, f, f, f));
  EXPECT_VEC8R(0_r, 2_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, v & SimdMask<Vec8r>(f, t, f, f, f, f, f, f));
  EXPECT_VEC8R(0_r, 0_r, 3_r, 0_r, 0_r, 0_r, 0_r, 0_r, v & SimdMask<Vec8r>(f, f, t, f, f, f, f, f));
  EXPECT_VEC8R(0_r, 0_r, 0_r, 4_r, 0_r, 0_r, 0_r, 0_r, v & SimdMask<Vec8r>(f, f, f, t, f, f, f, f));
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 5_r, 0_r, 0_r, 0_r, v & SimdMask<Vec8r>(f, f, f, f, t, f, f, f));
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 6_r, 0_r, 0_r, v & SimdMask<Vec8r>(f, f, f, f, f, t, f, f));
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 7_r, 0_r, v & SimdMask<Vec8r>(f, f, f, f, f, f, t, f));
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 8_r, v & SimdMask<Vec8r>(f, f, f, f, f, f, f, t));
  EXPECT_VEC8R(1_r, 0_r, 3_r, 0_r, 5_r, 0_r, 7_r, 0_r, v & SimdMask<Vec8r>(t, f, t, f, t, f, t, f));
  EXPECT_VEC8R(0_r, 2_r, 0_r, 4_r, 0_r, 6_r, 0_r, 8_r, v & SimdMask<Vec8r>(f, t, f, t, f, t, f, t));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 0_r, 0_r, 0_r, 0_r, v & SimdMask<Vec8r>(t, t, t, t, f, f, f, f));
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 5_r, 6_r, 7_r, 8_r, v & SimdMask<Vec8r>(f, f, f, f, t, t, t, t));
  EXPECT_VEC8R(1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, v & SimdMask<Vec8r>(t, t, t, t, t, t, t, t));
  // clang-format on
}

TEST(Vec8r, SimdZero) {
  EXPECT_VEC8R(0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, SimdZero<Vec8r>());
}

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, Sign, ([](auto a) { return (a >= 0) ? 1_r : -1_r; }), kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, SignedSqrt, ([](auto a) { return std::sqrt(std::abs(a)) * ((a >= 0) ? 1 : -1); }), kEps);
// clang-format on

TEST(Vec8r, Sin) {
  TestSimdTrigFunction<Vec8r>(
      [](real x) { return std::sin(x); }, [](Vec8r x) { return Sin(x); }, GetTrigTestValues());
}

TEST(Vec8r, SinCos) {
  auto values = GetTrigTestValues();
  TestSimdTrigFunction<Vec8r>(
      [](real x) { return std::sin(x); }, [](Vec8r x) { return SinCos(x).first; }, values);
  TestSimdTrigFunction<Vec8r>(
      [](real x) { return std::cos(x); }, [](Vec8r x) { return SinCos(x).second; }, values);
}

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, Sqr, ([](auto a) { return a * a; }), kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec8r, Sqrt, ([](auto a) { return std::sqrt(a); }), 0_r, 10_r, kEps);
// clang-format on

TEST(Vec8r, Store) {
  // clang-format off
  std::vector<real> result(9); // NOTE: Changed from an array on the stack to work around an MSVC optimizer bug.
  auto const v = Vec8r{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  Store<0>((real*)nullptr, v);
  Store<0>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store<1>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store<2>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store<3>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 0_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store<4>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store<5>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 5_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store<6>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 0_r, 0_r}), Span(&result[1], 8));
  Store<7>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 0_r}), Span(&result[1], 8));
  Store<8>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r}), Span(&result[1], 8));
  result.clear();
  result.resize(9);
  Store(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r}), Span(&result[1], 8));

  result.clear();
  result.resize(9);
  Store(&result[1], v, 0);
  EXPECT_SPAN_EQ((std::array<real, 8>{0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store(&result[1], v, 1);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store(&result[1], v, 2);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store(&result[1], v, 3);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 0_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store(&result[1], v, 4);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 0_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store(&result[1], v, 5);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 5_r, 0_r, 0_r, 0_r}), Span(&result[1], 8));
  Store(&result[1], v, 6);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 0_r, 0_r}), Span(&result[1], 8));
  Store(&result[1], v, 7);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 0_r}), Span(&result[1], 8));
  Store(&result[1], v, 8);
  EXPECT_SPAN_EQ((std::array<real, 8>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r}), Span(&result[1], 8));
  // clang-format on
}

TEST(Vec8r, StoreSelected) {
  Vec8r srcValues{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  // Use int64_t for the condition type. This will be larger than type real in single-precision
  // builds, but that's OK.
  std::array<int64_t, 8> condition{};
  std::array<real, 8> expectedValues{};
  alignas(16) std::array<real, 10> dstBuffer{};
  dstBuffer[0] = 123_r;
  dstBuffer[9] = 456_r;
  Span<real> dstValues{&dstBuffer[1], 8}; // Not aligned
  for (int64_t mask = 0; mask < 256; ++mask) { // For each possible set of conditions
    int expectedCount = 0;
    for (int i = 0; i < 8; ++i) {
      condition[i] = (mask & (1 << i)) ? -1 : 0; // true = -1, false = 0
      if (condition[i]) {
        expectedValues[expectedCount++] = srcValues[i];
      }
    }
    auto countStored =
        StoreSelected(dstValues.data(), Load<Simd<int64_t, 8>>(condition.data()), srcValues);
    EXPECT_EQ(expectedCount, countStored);
    for (int i = 0; i < countStored; ++i) {
      EXPECT_EQ(expectedValues[i], dstValues[i]);
    }
  }
  // Sentinel values should remain unchanged
  EXPECT_EQ(123_r, dstBuffer[0]);
  EXPECT_EQ(456_r, dstBuffer[9]);
}

TEST(Vec8r, StoreTransposed) {
  alignas(alignof(Vec8r)) real result[26] = {};
  result[25] = 911_r; // Sentinel value
  auto* ptr = result + 1; // Not an aligned address
  Vec8r data[] = {
      Vec8r{1_r, 4_r, 7_r, 10_r, 13_r, 16_r, 19_r, 22_r},
      Vec8r{2_r, 5_r, 8_r, 11_r, 14_r, 17_r, 20_r, 23_r},
      Vec8r{3_r, 6_r, 9_r, 12_r, 15_r, 18_r, 21_r, 24_r}};
  StoreTransposed<1>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 24>{1_r, 2_r, 3_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r,
                            0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}),
      Span(ptr, 24));
  StoreTransposed<2>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 24>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r,
                            0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}),
      Span(ptr, 24));
  StoreTransposed<3>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 24>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 0_r, 0_r, 0_r,
                            0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}),
      Span(ptr, 24));
  StoreTransposed<4>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 24>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, 12_r,
                            0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r,  0_r,  0_r}),
      Span(ptr, 24));
  StoreTransposed<5>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 24>{1_r,  2_r,  3_r,  4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, 12_r,
                            13_r, 14_r, 15_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r,  0_r,  0_r}),
      Span(ptr, 24));
  StoreTransposed<6>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 24>{1_r,  2_r,  3_r,  4_r,  5_r,  6_r,  7_r, 8_r, 9_r, 10_r, 11_r, 12_r,
                            13_r, 14_r, 15_r, 16_r, 17_r, 18_r, 0_r, 0_r, 0_r, 0_r,  0_r,  0_r}),
      Span(ptr, 24));
  StoreTransposed<7>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 24>{1_r,  2_r,  3_r,  4_r,  5_r,  6_r,  7_r,  8_r,  9_r,  10_r, 11_r, 12_r,
                            13_r, 14_r, 15_r, 16_r, 17_r, 18_r, 19_r, 20_r, 21_r, 0_r,  0_r,  0_r}),
      Span(ptr, 24));
  StoreTransposed<8>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 24>{1_r,  2_r,  3_r,  4_r,  5_r,  6_r,  7_r,  8_r,
                            9_r,  10_r, 11_r, 12_r, 13_r, 14_r, 15_r, 16_r,
                            17_r, 18_r, 19_r, 20_r, 21_r, 22_r, 23_r, 24_r}),
      Span(ptr, 24));
  StoreTransposed(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 24>{1_r,  2_r,  3_r,  4_r,  5_r,  6_r,  7_r,  8_r,
                            9_r,  10_r, 11_r, 12_r, 13_r, 14_r, 15_r, 16_r,
                            17_r, 18_r, 19_r, 20_r, 21_r, 22_r, 23_r, 24_r}),
      Span(ptr, 24));
  EXPECT_EQ(0_r, result[0]); // No change
  EXPECT_EQ(911_r, result[25]); // No change
}

// clang-format off
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec8r, Sub, -, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec8r, Tan, ([](auto a) { return std::tan(a); }), -1_r, 1_r, kEps);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec8r, BitwiseXOR, ^);

MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec8r, Exp, ([](auto a) { return std::exp(a); }), -3_r, 3_r, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec8r, Ln, ([](auto a) { return std::log(a); }), 0.1_r, 10_r, kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec8r, Tanh, ([](auto a) { return std::tanh(a); }), kEps);
// clang-format on

TEST(Vec8r, ExpExtreme) {
  // Keep non-negative entries in array 'x'
  std::array<real, 8> x({60.0, 65.123, 70.5, 75.3, 80.0, 84.123, 85.678, 86.35});
  Vec8r v(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
  auto expv = Exp(v), expmv = Exp(-v);
  auto tol = real(2.0) * std::numeric_limits<real>::epsilon();
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR_RTOL(Vec8r::Get(expv, i), std::exp(x[i]), tol);
    EXPECT_LE(
        Abs(Vec8r::Get(expmv, i) - std::exp(-x[i])),
        Max(Vec8r::Get(expmv, i), std::exp(-x[i])) * tol);
  }
}
