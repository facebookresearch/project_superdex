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

static_assert(std::is_trivially_copyable_v<Simd<real, 4>>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec4r);

#define EXPECT_VEC4R(e0, e1, e2, e3, actual)                                    \
  {                                                                             \
    auto vActual = actual;                                                      \
    /* Testing for exact equality handles values like +/- infinity */           \
    EXPECT_TRUE(((e0) == vActual[0]) || NearEqual(real(e0), vActual[0], kEps)); \
    EXPECT_TRUE(((e1) == vActual[1]) || NearEqual(real(e1), vActual[1], kEps)); \
    EXPECT_TRUE(((e2) == vActual[2]) || NearEqual(real(e2), vActual[2], kEps)); \
    EXPECT_TRUE(((e3) == vActual[3]) || NearEqual(real(e3), vActual[3], kEps)); \
  }

TEST(Vec4r, Class) {
  static_assert(Vec4r::kIsSupported, "Should be supported");

#if MOCHI_USE_SIMD
  bool constexpr kExpectNativeSize =
      (MOCHI_ARCH_X64_AVX2 || (MOCHI_ARCH_ARM_NEON && !MOCHI_USE_DOUBLE_PRECISION));
  static_assert(Vec4r::kIsComposite == !kExpectNativeSize);
  static_assert(!Vec4r::kIsEmulated);
#else
  static_assert(!Vec4r::kIsComposite);
  static_assert(Vec4r::kIsEmulated);
#endif

  static_assert(sizeof(Vec4r) == sizeof(real) * 4);
  static_assert(alignof(Vec4r) == alignof(typename Vec4r::NativeType));
  static_assert(std::is_same_v<Vec4r::Scalar, real>);
  static_assert(Vec4r::kSize == 4);
  static_assert(Vec4r::size() == 4);

  // Construct from scalars
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, Vec4r(1_r));
  EXPECT_VEC4R(1_r, 2_r, 0_r, 0_r, Vec4r(1_r, 2_r));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 0_r, Vec4r(1_r, 2_r, 3_r));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, Vec4r(1_r, 2_r, 3_r, 4_r));

  // Implicit conversion from scalar
  Vec4r a = 2_r;
  EXPECT_VEC4R(2_r, 2_r, 2_r, 2_r, a);
  a = 3_r;
  EXPECT_VEC4R(3_r, 3_r, 3_r, 3_r, a);

  // Copy construct
  Vec4r b{a};
  EXPECT_VEC4R(3_r, 3_r, 3_r, 3_r, b);

  // Copy assign
  a = Vec4r{4_r};
  b = a;
  EXPECT_VEC4R(4_r, 4_r, 4_r, 4_r, b);

  // Comparison
  EXPECT_EQ(true, (a == b));
  EXPECT_EQ(true, (a != Vec4r{}));
  EXPECT_EQ(false, (a != b));
  EXPECT_EQ(false, (a == Vec4r{}));

  // Unary operators
  using IVec = Simd<int, sizeof(Vec4r) / sizeof(int)>;
  Vec4r ones = ReinterpretCast<Vec4r>(IVec{-1}); // -nan must be compared using int
  Vec4r zeros = {};
  EXPECT_EQ(ReinterpretCast<IVec>(ones), ReinterpretCast<IVec>(~zeros));
  EXPECT_EQ(ReinterpretCast<IVec>(zeros), ReinterpretCast<IVec>(~ones));
  EXPECT_VEC4R(-1_r, -2_r, -3_r, -4_r, -Vec4r(1_r, 2_r, 3_r, 4_r));

  // Binary operators
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  b = Vec4r{5_r, 6_r, 7_r, 8_r};
  EXPECT_VEC4R(6_r, 8_r, 10_r, 12_r, a + b);
  EXPECT_VEC4R(4_r, 5_r, 6_r, 7_r, a + 3_r);
  EXPECT_VEC4R(4_r, 5_r, 6_r, 7_r, 3_r + a);
  EXPECT_VEC4R(-4_r, -4_r, -4_r, -4_r, a - b);
  EXPECT_VEC4R(-2_r, -1_r, 0_r, 1_r, a - 3_r);
  EXPECT_VEC4R(2_r, 1_r, 0_r, -1_r, 3_r - a);
  EXPECT_VEC4R(5_r, 12_r, 21_r, 32_r, a * b);
  EXPECT_VEC4R(3_r, 6_r, 9_r, 12_r, a * 3_r);
  EXPECT_VEC4R(3_r, 6_r, 9_r, 12_r, 3_r * a);
  EXPECT_VEC4R(5_r, 3_r, 2.333333_r, 2_r, b / a);
  EXPECT_VEC4R(1_r / 3_r, 2_r / 3_r, 3_r / 3_r, 4_r / 3_r, a / 3_r);
  EXPECT_VEC4R(3_r / 1_r, 3_r / 2_r, 3_r / 3_r, 3_r / 4_r, 3_r / a);
  EXPECT_EQ(a, a & ones);
  EXPECT_EQ(zeros, a & zeros);
  EXPECT_EQ(ReinterpretCast<IVec>(ones), ReinterpretCast<IVec>(a | ones));
  EXPECT_EQ(a, a | zeros);
  EXPECT_EQ(~a, a ^ ones);
  EXPECT_EQ(a, a ^ zeros);

  // Update operators
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  a += b;
  EXPECT_VEC4R(6_r, 8_r, 10_r, 12_r, a);
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  a += 3_r;
  EXPECT_VEC4R(4_r, 5_r, 6_r, 7_r, a);
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  a -= b;
  EXPECT_VEC4R(-4_r, -4_r, -4_r, -4_r, a);
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  a -= 3_r;
  EXPECT_VEC4R(-2_r, -1_r, 0_r, 1_r, a);
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  a *= b;
  EXPECT_VEC4R(5_r, 12_r, 21_r, 32_r, a);
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  a *= 3_r;
  EXPECT_VEC4R(3_r, 6_r, 9_r, 12_r, a);
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  b = Vec4r{5_r, 6_r, 7_r, 8_r};
  b /= a;
  EXPECT_VEC4R(5_r, 3_r, 2.333333_r, 2_r, b);
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  a /= 3_r;
  EXPECT_VEC4R(1_r / 3_r, 2_r / 3_r, 3_r / 3_r, 4_r / 3_r, a);
  b = a;
  b &= ones;
  EXPECT_EQ(a, b);
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  a &= zeros;
  EXPECT_EQ(zeros, a);
  a = Vec4r{1_r, 2_r, 3_r, 4_r};
  a |= ones;
  EXPECT_EQ(ReinterpretCast<IVec>(ones), ReinterpretCast<IVec>(a));
  a = b = Vec4r{1_r, 2_r, 3_r, 4_r};
  b |= zeros;
  EXPECT_EQ(a, b);
  a = b = Vec4r{1_r, 2_r, 3_r, 4_r};
  b ^= ones;
  EXPECT_EQ(~a, b);
  a = b = Vec4r{1_r, 2_r, 3_r, 4_r};
  b ^= zeros;
  EXPECT_EQ(a, b);
}

MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, Abs, std::abs, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec4r, ACos, std::acos, -0.9_r, 0.9_r, kEps);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec4r, Add, +, kEps);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec4r, BitwiseAND, &);
MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(Vec4r);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec4r, ASin, std::asin, -0.9_r, 0.9_r, kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, ATan, std::atan, kEps);

TEST(Vec4r, Blend) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = Vec4r{5_r, 6_r, 7_r, 8_r};
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, (Blend<0, 0, 0, 0>(a, b)));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 8_r, (Blend<0, 0, 0, 1>(a, b)));
  EXPECT_VEC4R(1_r, 2_r, 7_r, 4_r, (Blend<0, 0, 1, 0>(a, b)));
  EXPECT_VEC4R(1_r, 2_r, 7_r, 8_r, (Blend<0, 0, 1, 1>(a, b)));
  EXPECT_VEC4R(1_r, 6_r, 3_r, 4_r, (Blend<0, 1, 0, 0>(a, b)));
  EXPECT_VEC4R(1_r, 6_r, 3_r, 8_r, (Blend<0, 1, 0, 1>(a, b)));
  EXPECT_VEC4R(1_r, 6_r, 7_r, 4_r, (Blend<0, 1, 1, 0>(a, b)));
  EXPECT_VEC4R(1_r, 6_r, 7_r, 8_r, (Blend<0, 1, 1, 1>(a, b)));
  EXPECT_VEC4R(5_r, 2_r, 3_r, 4_r, (Blend<1, 0, 0, 0>(a, b)));
  EXPECT_VEC4R(5_r, 2_r, 3_r, 8_r, (Blend<1, 0, 0, 1>(a, b)));
  EXPECT_VEC4R(5_r, 2_r, 7_r, 4_r, (Blend<1, 0, 1, 0>(a, b)));
  EXPECT_VEC4R(5_r, 2_r, 7_r, 8_r, (Blend<1, 0, 1, 1>(a, b)));
  EXPECT_VEC4R(5_r, 6_r, 3_r, 4_r, (Blend<1, 1, 0, 0>(a, b)));
  EXPECT_VEC4R(5_r, 6_r, 3_r, 8_r, (Blend<1, 1, 0, 1>(a, b)));
  EXPECT_VEC4R(5_r, 6_r, 7_r, 4_r, (Blend<1, 1, 1, 0>(a, b)));
  EXPECT_VEC4R(5_r, 6_r, 7_r, 8_r, (Blend<1, 1, 1, 1>(a, b)));
}

TEST(Vec4r, AllTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  real zer = 0_r;
  real one = 0_r;
  auto allOnes = static_cast<uint64_t>(-1);
  memcpy(&one, &allOnes, sizeof(real));
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          auto vec = Vec4r{a ? one : zer, b ? one : zer, c ? one : zer, d ? one : zer};
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

TEST(Vec4r, AnyTrue) {
  // These are the values for "true" and "false" that are returned by SIMD comparisons
  real zer = 0_r;
  real one = 0_r;
  auto allOnes = static_cast<uint64_t>(-1);
  memcpy(&one, &allOnes, sizeof(real));
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          auto vec = Vec4r{a ? one : zer, b ? one : zer, c ? one : zer, d ? one : zer};
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

TEST(Vec4r, IsFinite) {
  constexpr real kQNaN = std::numeric_limits<real>::quiet_NaN();
  constexpr real kSNaN = std::numeric_limits<real>::signaling_NaN();
  EXPECT_TRUE(IsFinite(Vec4r{}));
  EXPECT_TRUE(IsFinite(Vec4r{1_r, 2_r, 3_r, 4_r}));
  EXPECT_TRUE(IsFinite(Vec4r{std::numeric_limits<real>::min()}));
  EXPECT_TRUE(IsFinite(Vec4r{-std::numeric_limits<real>::min()}));
  EXPECT_TRUE(IsFinite(Vec4r{std::numeric_limits<real>::lowest()}));
  EXPECT_TRUE(IsFinite(Vec4r{-std::numeric_limits<real>::lowest()}));
  EXPECT_TRUE(IsFinite(Vec4r{std::numeric_limits<real>::max()}));
  EXPECT_TRUE(IsFinite(Vec4r{-std::numeric_limits<real>::max()}));
  EXPECT_FALSE(IsFinite(Vec4r(kInf, 0_r, 0_r, 0_r)));
  EXPECT_FALSE(IsFinite(Vec4r(0_r, -kInf, 0_r, 0_r)));
  EXPECT_FALSE(IsFinite(Vec4r(0_r, 0_r, kQNaN, 0_r)));
  EXPECT_FALSE(IsFinite(Vec4r(0_r, 0_r, 0_r, kSNaN)));
  EXPECT_VEC4R(0_r, 1_r, 0_r, 1_r, VIsFinite(Vec4r(kInf, 0_r, -kInf, 0_r)) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 0_r, 1_r, 0_r, VIsFinite(Vec4r(0_r, kQNaN, 0_r, kSNaN)) & Vec4r{1_r});
}

TEST(Vec4r, IsTrue) {
  auto a = SimdMask<Vec4r>(true, false, true, false);
  EXPECT_TRUE(IsTrue<0>(a));
  EXPECT_FALSE(IsTrue<1>(a));
  EXPECT_TRUE(IsTrue<2>(a));
  EXPECT_FALSE(IsTrue<3>(a));
  EXPECT_FALSE(IsTrue<0>(~a));
  EXPECT_TRUE(IsTrue<1>(~a));
  EXPECT_FALSE(IsTrue<2>(~a));
  EXPECT_TRUE(IsTrue<3>(~a));
}

// Test both Equal and NotEqual
TEST(Vec4r, Equal) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = a;
  auto c = Vec4r{1.1_r, 2_r, 3_r, 4_r};
  auto d = Vec4r{1_r, 2.1_r, 3_r, 4_r};
  auto e = Vec4r{1_r, 2_r, 3.1_r, 4_r};
  auto f = Vec4r{1_r, 2_r, 3_r, 4.1_r};

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

TEST(Vec4r, Norm) {
  Vec4r const kValues[] = {
      Vec4r{0_r},
      Vec4r{2_r, 0_r, 0_r, 0_r},
      Vec4r{0_r, -2_r, 0_r, 0_r},
      Vec4r{0_r, 0_r, 2_r, 0_r},
      Vec4r{0_r, 0_r, 0_r, -2_r},
      Vec4r{-2_r, 3_r, -4_r, 5_r}};
  for (auto v : kValues) {
    // 2 components
    {
      auto nsqr = Sqr(Get<0>(v)) + Sqr(Get<1>(v));
      auto norm = std::sqrt(nsqr);
      EXPECT_NEAR_EQ(Vec4r(norm), VNorm<2>(v));
      EXPECT_NEAR_EQ(Vec4r(nsqr), VNormSqr<2>(v));
      EXPECT_NEAR_EQ(norm, Norm<2>(v));
      EXPECT_NEAR_EQ(nsqr, NormSqr<2>(v));
    }

    // 3 components
    {
      auto nsqr = Sqr(Get<0>(v)) + Sqr(Get<1>(v)) + Sqr(Get<2>(v));
      auto norm = std::sqrt(nsqr);
      EXPECT_NEAR_EQ(Vec4r(norm), VNorm<3>(v));
      EXPECT_NEAR_EQ(Vec4r(nsqr), VNormSqr<3>(v));
      EXPECT_NEAR_EQ(norm, Norm<3>(v));
      EXPECT_NEAR_EQ(nsqr, NormSqr<3>(v));
    }

    // 4 components
    {
      auto nsqr = Sqr(Get<0>(v)) + Sqr(Get<1>(v)) + Sqr(Get<2>(v)) + Sqr(Get<3>(v));
      auto norm = std::sqrt(nsqr);
      EXPECT_NEAR_EQ(Vec4r(norm), VNorm(v));
      EXPECT_NEAR_EQ(Vec4r(nsqr), VNormSqr(v));
      EXPECT_NEAR_EQ(norm, Norm(v));
      EXPECT_NEAR_EQ(nsqr, NormSqr(v));
    }
  }
}

TEST(Vec4r, Normalize) {
  auto v = Vec4r{1_r, -2_r, 3_r, -4_r};
  EXPECT_TRUE(NearEqual<2>(Vec4r(), Normalize<2>(Vec4r{}))); // Zero in, zero out
  EXPECT_TRUE(NearEqual<2>(v / Sqrt(5_r), Normalize<2>(v))); // 2 components
  EXPECT_TRUE(NearEqual<3>(v / Sqrt(14_r), Normalize<3>(v))); // 3 components
  EXPECT_TRUE(NearEqual(v / Sqrt(30_r), Normalize(v))); // 4 components
  EXPECT_TRUE(NearEqual(v / Sqrt(30_r), Normalize(v, 30_r))); // 4 components
}

TEST(Vec4r, OrthogonalVector3) {
  // clang-format off
  Vec4r const kValues[] = {
    Vec4r{ 0.97627008_r,  4.30378733_r,  2.05526752_r, 0_r},
    Vec4r{ 0.89766366_r, -1.52690401_r,  2.91788226_r, 1_r},
    Vec4r{-1.24825577_r,  7.83546002_r,  9.27325521_r, 2_r},
    Vec4r{-2.33116962_r,  5.83450076_r,  0.57789840_r, 3_r}};
  // clang-format on
  for (auto const& v : kValues) {
    Vec4r ortho = OrthogonalVector3(v);
    EXPECT_FALSE(NearZero(ortho));
    EXPECT_TRUE(NearZero(Dot(v, ortho), 2_r * kDefaultNearEqualEpsilon<real>));
  }
  EXPECT_EQ(Vec4r{}, OrthogonalVector3(Vec4r{})); // Zero in, zero out
}

TEST(Vec4r, NearEqual) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = Vec4r{1.1_r, 2_r, 3_r, 4_r};
  auto c = Vec4r{1_r, 2.1_r, 3_r, 4_r};
  auto d = Vec4r{1_r, 2_r, 3.1_r, 4_r};
  auto e = Vec4r{1_r, 2_r, 3_r, 4.1_r};

  // Compare 1 components
  EXPECT_TRUE(NearEqual<1>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<1>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, b, 0.11f));
  EXPECT_TRUE(NearEqual<1>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<1>(a, e, 0_r));

  // Compare 2 components
  EXPECT_TRUE(NearEqual<2>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<2>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, b, 0.11f));
  EXPECT_FALSE(NearEqual<2>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, c, 0.11f));
  EXPECT_TRUE(NearEqual<2>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<2>(a, e, 0_r));

  // Compare 3 components
  EXPECT_TRUE(NearEqual<3>(a, a, 0_r));
  EXPECT_FALSE(NearEqual<3>(a, b, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, b, 0.11f));
  EXPECT_FALSE(NearEqual<3>(a, c, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, c, 0.11f));
  EXPECT_FALSE(NearEqual<3>(a, d, 0_r));
  EXPECT_TRUE(NearEqual<3>(a, d, 0.11f));
  EXPECT_TRUE(NearEqual<3>(a, e, 0_r));

  // Compare 4 components (default)
  EXPECT_TRUE(NearEqual(a, a, 0_r));
  EXPECT_FALSE(NearEqual(a, b, 0_r));
  EXPECT_TRUE(NearEqual(a, b, 0.11f));
  EXPECT_FALSE(NearEqual(a, c, 0_r));
  EXPECT_TRUE(NearEqual(a, c, 0.11f));
  EXPECT_FALSE(NearEqual(a, d, 0_r));
  EXPECT_TRUE(NearEqual(a, d, 0.11f));
  EXPECT_FALSE(NearEqual(a, e, 0_r));
  EXPECT_TRUE(NearEqual(a, e, 0.11f));
}

TEST(Vec4r, VNearEqual) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = Vec4r{1.1_r, 2_r, 3.1_r, 4_r};
  auto c = Vec4r{1.1_r, 2.2_r, 3.2_r, 4.1_r};
  // clang-format off
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, VNearEqual(a, a, Vec4r(0_r, 0_r, 0_r, 0_r)) & Vec4r(1_r));
  EXPECT_VEC4R(0_r, 1_r, 0_r, 1_r, VNearEqual(a, b, Vec4r(0_r, 0_r, 0_r, 0_r)) & Vec4r(1_r));
  EXPECT_VEC4R(1_r, 1_r, 0_r, 1_r, VNearEqual(a, b, Vec4r(0.11_r, 0_r, 0_r, 0_r)) & Vec4r(1_r));
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, VNearEqual(a, b, Vec4r(0.11_r, 0.11_r, 0.11_r, 0.11_r)) & Vec4r(1_r));
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, VNearEqual(a, c, Vec4r(0_r, 0_r, 0_r, 0_r)) & Vec4r(1_r));
  EXPECT_VEC4R(0_r, 0_r, 0_r, 1_r, VNearEqual(a, c, Vec4r(0_r, 0_r, 0_r, 0.11f)) & Vec4r(1_r));
  EXPECT_VEC4R(1_r, 0_r, 0_r, 1_r, VNearEqual(a, c, Vec4r(0.11_r, 0.11_r, 0.11_r, 0.11_r)) & Vec4r(1_r));
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, VNearEqual(a, c, Vec4r(0.21_r, 0.21_r, 0.21_r, 0.21_r)) & Vec4r(1_r));
  // clang-format on
}

TEST(Vec4r, Neg4Bools) { // Neg<bool, bool, bool, bool>(Simd<T, N>)
  // clang-format off
  auto const a = Vec4r{1_r, -2_r,  3_r, -4_r};
  EXPECT_VEC4R( 1_r, -2_r,  3_r, -4_r, (Neg<false, false, false, false>(a)));
  EXPECT_VEC4R( 1_r, -2_r,  3_r,  4_r, (Neg<false, false, false, true >(a)));
  EXPECT_VEC4R( 1_r, -2_r, -3_r, -4_r, (Neg<false, false, true,  false>(a)));
  EXPECT_VEC4R( 1_r, -2_r, -3_r,  4_r, (Neg<false, false, true,  true >(a)));
  EXPECT_VEC4R( 1_r,  2_r,  3_r, -4_r, (Neg<false, true,  false, false>(a)));
  EXPECT_VEC4R( 1_r,  2_r,  3_r,  4_r, (Neg<false, true,  false, true >(a)));
  EXPECT_VEC4R( 1_r,  2_r, -3_r, -4_r, (Neg<false, true,  true,  false>(a)));
  EXPECT_VEC4R( 1_r,  2_r, -3_r,  4_r, (Neg<false, true,  true,  true >(a)));
  EXPECT_VEC4R(-1_r, -2_r,  3_r, -4_r, (Neg<true,  false, false, false>(a)));
  EXPECT_VEC4R(-1_r, -2_r,  3_r,  4_r, (Neg<true,  false, false, true >(a)));
  EXPECT_VEC4R(-1_r, -2_r, -3_r, -4_r, (Neg<true,  false, true,  false>(a)));
  EXPECT_VEC4R(-1_r, -2_r, -3_r,  4_r, (Neg<true,  false, true,  true >(a)));
  EXPECT_VEC4R(-1_r,  2_r,  3_r, -4_r, (Neg<true,  true,  false, false>(a)));
  EXPECT_VEC4R(-1_r,  2_r,  3_r,  4_r, (Neg<true,  true,  false, true >(a)));
  EXPECT_VEC4R(-1_r,  2_r, -3_r, -4_r, (Neg<true,  true,  true,  false>(a)));
  EXPECT_VEC4R(-1_r,  2_r, -3_r,  4_r, (Neg<true,  true,  true,  true >(a)));
  // clang-format on
}

TEST(Vec4r, NearZero) {
  auto a = Vec4r{0.1_r, -0.2_r, 0.3_r, -0.4_r};

  // Default tolerance
  EXPECT_TRUE(NearZero(Vec4r{}));
  EXPECT_TRUE(NearZero(Vec4r{kDefaultNearEqualEpsilon<real>}));
  EXPECT_TRUE(NearZero(-Vec4r{kDefaultNearEqualEpsilon<real>}));
  EXPECT_FALSE(NearZero(a));

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

  // Compare 4 components (default)
  EXPECT_FALSE(NearZero(a, 0_r));
  EXPECT_FALSE(NearZero(a, 0.11f));
  EXPECT_FALSE(NearZero(a, 0.21f));
  EXPECT_FALSE(NearZero(a, 0.31f));
  EXPECT_TRUE(NearZero(a, 0.41f));
}

TEST(Vec4r, VNearZero) {
  auto z = Vec4r{};
  auto a = Vec4r{0.1_r, -0.2_r, 0.3_r, -0.4_r};
  // clang-format off
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, VNearZero(z, Vec4r(0_r, 0_r, 0_r, 0_r)) & Vec4r(1_r));
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, VNearZero(a, Vec4r(0_r, 0_r, 0_r, 0_r)) & Vec4r(1_r));
  EXPECT_VEC4R(1_r, 0_r, 0_r, 0_r, VNearZero(a, Vec4r(0.11_r, 0_r, 0_r, 0_r)) & Vec4r(1_r));
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, VNearZero(a, Vec4r(0_r, 0.11_r, 0.11_r, 0.11_r)) & Vec4r(1_r));
  EXPECT_VEC4R(1_r, 0_r, 0_r, 0_r, VNearZero(a, Vec4r(0.11_r, 0.11_r, 0.11_r, 0.11_r)) & Vec4r(1_r));
  EXPECT_VEC4R(1_r, 1_r, 0_r, 0_r, VNearZero(a, Vec4r(0.21_r, 0.21_r, 0.21_r, 0.21_r)) & Vec4r(1_r));
  EXPECT_VEC4R(1_r, 1_r, 1_r, 0_r, VNearZero(a, Vec4r(0.31_r, 0.31_r, 0.31_r, 0.31_r)) & Vec4r(1_r));
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, VNearZero(a, Vec4r(0.41_r, 0.41_r, 0.41_r, 0.41_r)) & Vec4r(1_r));
  // clang-format on
}

TEST(Vec4r, Broadcast) {
  Vec4r v;

  // Broadcast scalar
  v = Broadcast<Vec4r>(1_r);
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, v);

  // Broadcast from address
  real const f = 2_r;
  v = Broadcast<Vec4r>(&f);
  EXPECT_VEC4R(2_r, 2_r, 2_r, 2_r, v);

  // Broadcast ith element
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, Broadcast<0>(a));
  EXPECT_VEC4R(2_r, 2_r, 2_r, 2_r, Broadcast<1>(a));
  EXPECT_VEC4R(3_r, 3_r, 3_r, 3_r, Broadcast<2>(a));
  EXPECT_VEC4R(4_r, 4_r, 4_r, 4_r, Broadcast<3>(a));

  // Broadcast ith element
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, Broadcast(a, 0));
  EXPECT_VEC4R(2_r, 2_r, 2_r, 2_r, Broadcast(a, 1));
  EXPECT_VEC4R(3_r, 3_r, 3_r, 3_r, Broadcast(a, 2));
  EXPECT_VEC4R(4_r, 4_r, 4_r, 4_r, Broadcast(a, 3));
}

MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec4r, Clamp, Clamp, kEps);

TEST(Vec4r, Cos) {
  TestSimdTrigFunction<Vec4r>(
      [](real x) { return std::cos(x); }, [](Vec4r x) { return Cos(x); }, GetTrigTestValues());
}

TEST(Vec4r, Cross3) {
  // Orthogonal axes
  auto x = Vec4r{1_r, 0_r, 0_r};
  auto y = Vec4r{0_r, 1_r, 0_r};
  auto z = Vec4r{0_r, 0_r, 1_r};
  auto eps = Broadcast<Vec4r>(kEps);
  EXPECT_TRUE(NearEqual(z, Cross3(x, y), eps));
  EXPECT_TRUE(NearEqual(-z, Cross3(y, x), eps));
  EXPECT_TRUE(NearEqual(x, Cross3(y, z), eps));
  EXPECT_TRUE(NearEqual(-x, Cross3(z, y), eps));
  EXPECT_TRUE(NearEqual(y, Cross3(z, x), eps));
  EXPECT_TRUE(NearEqual(-y, Cross3(x, z), eps));

  // Arbitrary values
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = Vec4r{5_r, 6_r, 7_r, 8_r};
  auto result = Cross3(a, b);
  EXPECT_VEC4R(-4_r, 8_r, -4_r, 0_r, result);
}

TEST(Vec4r, ToSimdDirection) {
  EXPECT_VEC4R(1_r, 2_r, 3_r, 0_r, (ToSimdDirection(Vec4r(1_r, 2_r, 3_r, 4_r))));
}

MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec4r, Div, /, kEps);

TEST(Vec4r, VDot) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = Vec4r{5_r, 6_r, 7_r, 8_r};
  EXPECT_VEC4R(17_r, 17_r, 17_r, 17_r, (VDot<2>(a, b))); // 2 component
  EXPECT_VEC4R(38_r, 38_r, 38_r, 38_r, (VDot<3>(a, b))); // 3 component
  EXPECT_VEC4R(70_r, 70_r, 70_r, 70_r, (VDot<4>(a, b))); // 4 component
}

// Test bot VEqual and VNotEqual
template <class V>
static void ExpectVEqual(bool e0, bool e1, bool e2, bool e3, V a, V b) {
  EXPECT_VEC4R(
      e0 ? 1_r : 0_r, e1 ? 1_r : 0_r, e2 ? 1_r : 0_r, e3 ? 1_r : 0_r, VEqual(a, b) & Vec4r{1_r});
  EXPECT_VEC4R(
      e0 ? 0_r : 1_r, e1 ? 0_r : 1_r, e2 ? 0_r : 1_r, e3 ? 0_r : 1_r, VNotEqual(a, b) & Vec4r{1_r});
}

TEST(Vec4r, VEqual) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = Vec4r{1_r, 9_r, 3_r, 9_r};
  auto c = Vec4r{9_r, 2_r, 9_r, 4_r};
  ExpectVEqual(1, 1, 1, 1, a, a);
  ExpectVEqual(1, 0, 1, 0, a, b);
  ExpectVEqual(0, 1, 0, 1, a, c);
}

TEST(Vec4r, Get) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};

  // Fast template version
  EXPECT_EQ(1_r, Get0(a));
  EXPECT_EQ(1_r, Get<0>(a));
  EXPECT_EQ(2_r, Get<1>(a));
  EXPECT_EQ(3_r, Get<2>(a));
  EXPECT_EQ(4_r, Get<3>(a));

  // Slower runtime version
  EXPECT_EQ(1_r, Get(a, 0));
  EXPECT_EQ(2_r, Get(a, 1));
  EXPECT_EQ(3_r, Get(a, 2));
  EXPECT_EQ(4_r, Get(a, 3));

  // Same but with operator[] (read only)
  EXPECT_EQ(1_r, a[0]);
  EXPECT_EQ(2_r, a[1]);
  EXPECT_EQ(3_r, a[2]);
  EXPECT_EQ(4_r, a[3]);
}

#if MOCHI_USE_DOUBLE_PRECISION
TEST(Vec4r, GetHalf) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto low = GetHalf<0>(a);
  auto high = GetHalf<1>(a);
  EXPECT_EQ(1_r, low[0]);
  EXPECT_EQ(2_r, low[1]);
  EXPECT_EQ(3_r, high[0]);
  EXPECT_EQ(4_r, high[1]);
}
#endif // MOCHI_USE_DOUBLE_PRECISION

TEST(Vec4r, Set) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};

  EXPECT_VEC4R(9_r, 2_r, 3_r, 4_r, Set<0>(a, 9_r));
  EXPECT_VEC4R(1_r, 9_r, 3_r, 4_r, Set<1>(a, 9_r));
  EXPECT_VEC4R(1_r, 2_r, 9_r, 4_r, Set<2>(a, 9_r));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 9_r, Set<3>(a, 9_r));

  EXPECT_VEC4R(9_r, 2_r, 3_r, 4_r, Set(a, 0, 9_r));
  EXPECT_VEC4R(1_r, 9_r, 3_r, 4_r, Set(a, 1, 9_r));
  EXPECT_VEC4R(1_r, 2_r, 9_r, 4_r, Set(a, 2, 9_r));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 9_r, Set(a, 3, 9_r));
}

TEST(Vec4r, Greater) {
  auto a = Vec4r{1_r, 4_r, 5_r, 8_r};
  auto b = Vec4r{2_r, 3_r, 6_r, 7_r};
  EXPECT_VEC4R(0_r, 1_r, 0_r, 1_r, (a > b) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 0_r, 1_r, 0_r, (b > a) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, (a > a) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 1_r, 1_r, (a > Vec4r{4.5_r}) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 0_r, 1_r, (a > Vec4r{5_r}) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 0_r, 0_r, (Vec4r{4.5_r} > a) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 0_r, 0_r, (Vec4r{5_r} > a) & Vec4r{1_r});
}

TEST(Vec4r, GreaterEqual) {
  auto a = Vec4r{1_r, 4_r, 5_r, 8_r};
  auto b = Vec4r{2_r, 3_r, 6_r, 7_r};
  EXPECT_VEC4R(0_r, 1_r, 0_r, 1_r, (a >= b) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 0_r, 1_r, 0_r, (b >= a) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, (a >= a) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 1_r, 1_r, (a >= Vec4r{4.5_r}) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 1_r, 1_r, (a >= Vec4r{5_r}) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 0_r, 0_r, (Vec4r{4.5_r} >= a) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 1_r, 0_r, (Vec4r{5_r} >= a) & Vec4r{1_r});
}

TEST(Vec4r, HMax) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  EXPECT_NEAR_EQ(2_r, HMax<2>(a));
  EXPECT_NEAR_EQ(3_r, HMax<3>(a));
  EXPECT_NEAR_EQ(4_r, HMax(a));
}

TEST(Vec4r, HMin) {
  auto a = Vec4r{4_r, 3_r, 2_r, 1_r};
  EXPECT_NEAR_EQ(3_r, HMin<2>(a));
  EXPECT_NEAR_EQ(2_r, HMin<3>(a));
  EXPECT_NEAR_EQ(1_r, HMin(a));
}

TEST(Vec4r, HProd) {
  auto a = Vec4r{2_r, 3_r, 4_r, 5_r};
  EXPECT_NEAR_EQ(6_r, HProd<2>(a));
  EXPECT_NEAR_EQ(24_r, HProd<3>(a));
  EXPECT_NEAR_EQ(120_r, HProd(a));
}

TEST(Vec4r, Lerp) {
  // Note: Lerp does not clamp the 't' parameter
  auto const a = Vec4r{10_r, 100_r, 1000_r, 10000_r};
  auto const b = Vec4r{20_r, 200_r, 2000_r, 20000_r};
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, Lerp(a, b, -1_r));
  EXPECT_VEC4R(5_r, 50_r, 500_r, 5000_r, Lerp(a, b, -0.5_r));
  EXPECT_VEC4R(10_r, 100_r, 1000_r, 10000_r, Lerp(a, b, 0_r));
  EXPECT_VEC4R(15_r, 150_r, 1500_r, 15000_r, Lerp(a, b, 0.5_r));
  EXPECT_VEC4R(20_r, 200_r, 2000_r, 20000_r, Lerp(a, b, 1_r));
  EXPECT_VEC4R(25_r, 250_r, 2500_r, 25000_r, Lerp(a, b, 1.5_r));
}

TEST(Vec4r, HSum) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  EXPECT_NEAR_EQ(3_r, HSum<2>(a));
  EXPECT_NEAR_EQ(6_r, HSum<3>(a));
  EXPECT_NEAR_EQ(10_r, HSum(a));
}

TEST(Vec4r, Less) {
  auto a = Vec4r{1_r, 4_r, 5_r, 8_r};
  auto b = Vec4r{2_r, 3_r, 6_r, 7_r};
  EXPECT_VEC4R(0_r, 1_r, 0_r, 1_r, (b < a) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 0_r, 1_r, 0_r, (a < b) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, (a < a) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 0_r, 0_r, (a < Vec4r{4.5_r}) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 0_r, 0_r, (a < Vec4r{5_r}) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 1_r, 1_r, (Vec4r{4.5_r} < a) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 0_r, 1_r, (Vec4r{5_r} < a) & Vec4r{1_r});
}

TEST(Vec4r, LessEqual) {
  auto a = Vec4r{1_r, 4_r, 5_r, 8_r};
  auto b = Vec4r{2_r, 3_r, 6_r, 7_r};
  EXPECT_VEC4R(0_r, 1_r, 0_r, 1_r, (b <= a) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 0_r, 1_r, 0_r, (a <= b) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, (a <= a) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 0_r, 0_r, (a <= Vec4r{4.5_r}) & Vec4r{1_r});
  EXPECT_VEC4R(1_r, 1_r, 1_r, 0_r, (a <= Vec4r{5_r}) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 1_r, 1_r, (Vec4r{4.5_r} <= a) & Vec4r{1_r});
  EXPECT_VEC4R(0_r, 0_r, 1_r, 1_r, (Vec4r{5_r} <= a) & Vec4r{1_r});
}

TEST(Vec4r, Load) {
  alignas(alignof(Vec4r)) real const values[] = {0.0, 1_r, 2_r, 3_r, 4_r};
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, (Load<0, Vec4r>(nullptr)));
  EXPECT_VEC4R(1_r, 0_r, 0_r, 0_r, (Load<1, Vec4r>(values + 1)));
  EXPECT_VEC4R(1_r, 2_r, 0_r, 0_r, (Load<2, Vec4r>(values + 1)));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 0_r, (Load<3, Vec4r>(values + 1)));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, (Load<4, Vec4r>(values + 1)));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, (Load<Vec4r>(values + 1)));

  EXPECT_VEC4R(1_r, 0_r, 0_r, 0_r, (Load<Vec4r>(values + 1, 1)));
  EXPECT_VEC4R(1_r, 2_r, 0_r, 0_r, (Load<Vec4r>(values + 1, 2)));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 0_r, (Load<Vec4r>(values + 1, 3)));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, (Load<Vec4r>(values + 1, 4)));
}

TEST(Vec4r, LoadIndexed) {
  alignas(alignof(Vec4r)) real const values[] = {0_r, 1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r};
  EXPECT_VEC4R(2_r, 4_r, 6_r, 8_r, LoadIndexed<Vec4r>(&values[1], Vec4i(1, 3, 5, 7)));
}

TEST(Vec4r, LoadTransposed) {
  alignas(alignof(Vec4r))
      real const values[] = {0_r, 1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, 12_r};
  auto const* ptr = values + 1; // Not an aligned address
  Vec4r loaded[3] = {};
  LoadTransposed<1>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4R(1_r, 0_r, 0_r, 0_r, loaded[0]);
  EXPECT_VEC4R(2_r, 0_r, 0_r, 0_r, loaded[1]);
  EXPECT_VEC4R(3_r, 0_r, 0_r, 0_r, loaded[2]);
  LoadTransposed<2>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4R(1_r, 4_r, 0_r, 0_r, loaded[0]);
  EXPECT_VEC4R(2_r, 5_r, 0_r, 0_r, loaded[1]);
  EXPECT_VEC4R(3_r, 6_r, 0_r, 0_r, loaded[2]);
  LoadTransposed<3>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4R(1_r, 4_r, 7_r, 0_r, loaded[0]);
  EXPECT_VEC4R(2_r, 5_r, 8_r, 0_r, loaded[1]);
  EXPECT_VEC4R(3_r, 6_r, 9_r, 0_r, loaded[2]);
  LoadTransposed<4>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4R(1_r, 4_r, 7_r, 10_r, loaded[0]);
  EXPECT_VEC4R(2_r, 5_r, 8_r, 11_r, loaded[1]);
  EXPECT_VEC4R(3_r, 6_r, 9_r, 12_r, loaded[2]);
  LoadTransposed(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC4R(1_r, 4_r, 7_r, 10_r, loaded[0]);
  EXPECT_VEC4R(2_r, 5_r, 8_r, 11_r, loaded[1]);
  EXPECT_VEC4R(3_r, 6_r, 9_r, 12_r, loaded[2]);
}

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, Floor, ([](auto a) { return std::floor(a); }), kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, FastRound, ([](auto a) { return std::round(a); }), kEps); // The standard test values have no exact ties
MOCHI_SIMD_TEST_BINARY_FN_NEAR(Vec4r, Max, ([](auto a, auto b) { return std::max(a, b); }), kEps);
MOCHI_SIMD_TEST_BINARY_FN_NEAR(Vec4r, Min, ([](auto a, auto b) { return std::min(a, b); }), kEps);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec4r, Mul, *, kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec4r, MulAdd, ([](auto a, auto b, auto c) { return a * b + c; }), kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec4r, MulSub, ([](auto a, auto b, auto c) { return a * b - c; }), kEps);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec4r, Neg, -);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec4r, NegMulAdd, ([](auto a, auto b, auto c) { return -(a * b) + c; }), kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(Vec4r, NegMulSub, ([](auto a, auto b, auto c) { return -(a * b) - c; }), kEps);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec4r, BitwiseNOT, ~);
// clang-format on

TEST(Vec4r, VNotEqual) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = Vec4r{1_r, 3_r, 3_r, 5_r};
  EXPECT_VEC4R(0_r, 1_r, 0_r, 1_r, VNotEqual(a, b) & Broadcast<Vec4r>(1_r));
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, VNotEqual(a, a) & Broadcast<Vec4r>(1_r));
}

MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec4r, BitwiseOR, |);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec4r, LogicalAND, &&, &);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec4r, LogicalOR, ||, |);

TEST(Vec4r, ToSimdPoint) {
  EXPECT_VEC4R(1_r, 2_r, 3_r, 1_r, (ToSimdPoint(Vec4r(1_r, 2_r, 3_r, 4_r))));
}

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, RcpApprox, ([](auto a) { return 1 / a; }), 1e-2f); // Large tolerance because these are only approximations
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, RcpSqrtApprox, ([](auto a) { return 1 / std::sqrt(a); }), 5e-3f);
// clang-format on

TEST(Vec4r, Select) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = Vec4r{5_r, 6_r, 7_r, 8_r};
  auto constexpr t = true;
  auto constexpr f = false;
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, Select(SimdMask<Vec4r>(t, t, t, t), a, b));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 8_r, Select(SimdMask<Vec4r>(t, t, t, f), a, b));
  EXPECT_VEC4R(1_r, 2_r, 7_r, 4_r, Select(SimdMask<Vec4r>(t, t, f, t), a, b));
  EXPECT_VEC4R(1_r, 2_r, 7_r, 8_r, Select(SimdMask<Vec4r>(t, t, f, f), a, b));
  EXPECT_VEC4R(1_r, 6_r, 3_r, 4_r, Select(SimdMask<Vec4r>(t, f, t, t), a, b));
  EXPECT_VEC4R(1_r, 6_r, 3_r, 8_r, Select(SimdMask<Vec4r>(t, f, t, f), a, b));
  EXPECT_VEC4R(1_r, 6_r, 7_r, 4_r, Select(SimdMask<Vec4r>(t, f, f, t), a, b));
  EXPECT_VEC4R(1_r, 6_r, 7_r, 8_r, Select(SimdMask<Vec4r>(t, f, f, f), a, b));
  EXPECT_VEC4R(5_r, 2_r, 3_r, 4_r, Select(SimdMask<Vec4r>(f, t, t, t), a, b));
  EXPECT_VEC4R(5_r, 2_r, 3_r, 8_r, Select(SimdMask<Vec4r>(f, t, t, f), a, b));
  EXPECT_VEC4R(5_r, 2_r, 7_r, 4_r, Select(SimdMask<Vec4r>(f, t, f, t), a, b));
  EXPECT_VEC4R(5_r, 2_r, 7_r, 8_r, Select(SimdMask<Vec4r>(f, t, f, f), a, b));
  EXPECT_VEC4R(5_r, 6_r, 3_r, 4_r, Select(SimdMask<Vec4r>(f, f, t, t), a, b));
  EXPECT_VEC4R(5_r, 6_r, 3_r, 8_r, Select(SimdMask<Vec4r>(f, f, t, f), a, b));
  EXPECT_VEC4R(5_r, 6_r, 7_r, 4_r, Select(SimdMask<Vec4r>(f, f, f, t), a, b));
  EXPECT_VEC4R(5_r, 6_r, 7_r, 8_r, Select(SimdMask<Vec4r>(f, f, f, f), a, b));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, Select(a < b, a, b)); // Min(a, b)
  EXPECT_VEC4R(5_r, 6_r, 7_r, 8_r, Select(a > b, a, b)); // Max(a, b)
  EXPECT_VEC4R(1_r, 2_r, 7_r, 8_r, Select(a <= Vec4r{2_r}, a, b));
  EXPECT_VEC4R(5_r, 6_r, 3_r, 4_r, Select(a > Vec4r{2_r}, a, b));
}

TEST(Vec4r, SimdBasisVector) {
  EXPECT_VEC4R(1_r, 0_r, 0_r, 0_r, (SimdBasisVector<0, Vec4r>()));
  EXPECT_VEC4R(0_r, 1_r, 0_r, 0_r, (SimdBasisVector<1, Vec4r>()));
  EXPECT_VEC4R(0_r, 0_r, 1_r, 0_r, (SimdBasisVector<2, Vec4r>()));
  EXPECT_VEC4R(0_r, 0_r, 0_r, 1_r, (SimdBasisVector<3, Vec4r>()));

  // Runtime axis form
  EXPECT_VEC4R(1_r, 0_r, 0_r, 0_r, SimdBasisVector<Vec4r>(0));
  EXPECT_VEC4R(0_r, 1_r, 0_r, 0_r, SimdBasisVector<Vec4r>(1));
  EXPECT_VEC4R(0_r, 0_r, 1_r, 0_r, SimdBasisVector<Vec4r>(2));
  EXPECT_VEC4R(0_r, 0_r, 0_r, 1_r, SimdBasisVector<Vec4r>(3));
}

TEST(Vec4r, SimdMask) {
  auto v = Vec4r{1_r, 2_r, 3_r, 4_r};
  bool constexpr f = false;
  bool constexpr t = true;
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, v & SimdMask<Vec4r>(f, f, f, f));
  EXPECT_VEC4R(1_r, 0_r, 0_r, 0_r, v & SimdMask<Vec4r>(t, f, f, f));
  EXPECT_VEC4R(0_r, 2_r, 0_r, 0_r, v & SimdMask<Vec4r>(f, t, f, f));
  EXPECT_VEC4R(0_r, 0_r, 3_r, 0_r, v & SimdMask<Vec4r>(f, f, t, f));
  EXPECT_VEC4R(0_r, 0_r, 0_r, 4_r, v & SimdMask<Vec4r>(f, f, f, t));
  EXPECT_VEC4R(1_r, 0_r, 3_r, 0_r, v & SimdMask<Vec4r>(t, f, t, f));
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, v & SimdMask<Vec4r>(t, t, t, t));
}

TEST(Vec4r, SimdZero) {
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, SimdZero<Vec4r>());
}

TEST(Vec4r, Shuffle) {
  auto a = Vec4r{1_r, 2_r, 3_r, 4_r};
  auto b = Vec4r{5_r, 6_r, 7_r, 8_r};
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, a);
  EXPECT_VEC4R(5_r, 6_r, 7_r, 8_r, b);

  // Shuffle single Vec4r
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, (Shuffle<0, 1, 2, 3>(a))); // same
  EXPECT_VEC4R(2_r, 3_r, 4_r, 1_r, (Shuffle<1, 2, 3, 0>(a))); // left shift
  EXPECT_VEC4R(4_r, 1_r, 2_r, 3_r, (Shuffle<3, 0, 1, 2>(a))); // right shift
  EXPECT_VEC4R(2_r, 1_r, 4_r, 3_r, (Shuffle<1, 0, 3, 2>(a))); // flip pairs
  EXPECT_VEC4R(4_r, 3_r, 2_r, 1_r, (Shuffle<3, 2, 1, 0>(a))); // reverse
  EXPECT_VEC4R(1_r, 1_r, 1_r, 1_r, (Shuffle<0, 0, 0, 0>(a))); // broadcast 0
  EXPECT_VEC4R(2_r, 2_r, 2_r, 2_r, (Shuffle<1, 1, 1, 1>(a))); // broadcast 1

  // Shuffle two Vec4fs
  EXPECT_VEC4R(1_r, 2_r, 5_r, 6_r, (Shuffle<0, 1, 0, 1>(a, b)));
  EXPECT_VEC4R(1_r, 2_r, 7_r, 8_r, (Shuffle<0, 1, 2, 3>(a, b)));
  EXPECT_VEC4R(1_r, 1_r, 5_r, 5_r, (Shuffle<0, 0, 0, 0>(a, b)));
  EXPECT_VEC4R(2_r, 2_r, 6_r, 6_r, (Shuffle<1, 1, 1, 1>(a, b)));
}

#if MOCHI_USE_SIMD // SignBitMask is a utility only implemented in native SIMD types
TEST(Vec4r, SignBitMask) {
  auto const signBit = Vec4r::SignBitMask(); // -0_r (all bits set to 0 except the sign bit)
  auto const a = Vec4r{1_r, -2_r, 3_r, -4_r};
  EXPECT_EQ(Vec4r{}, signBit); // 0_r == -0_r
  EXPECT_EQ(-a, a ^ signBit); // flip sign
}
#endif

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, Sign, ([](auto a) { return (a >= 0) ? 1_r : -1_r; }), kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, SignedSqrt, ([](auto a) { return std::sqrt(std::abs(a)) * ((a >= 0) ? 1 : -1); }), kEps);
// clang-format on

TEST(Vec4r, Sin) {
  TestSimdTrigFunction<Vec4r>(
      [](real x) { return std::sin(x); }, [](Vec4r x) { return Sin(x); }, GetTrigTestValues());
}

TEST(Vec4r, SinCos) {
  auto values = GetTrigTestValues();
  TestSimdTrigFunction<Vec4r>(
      [](real x) { return std::sin(x); }, [](Vec4r x) { return SinCos(x).first; }, values);
  TestSimdTrigFunction<Vec4r>(
      [](real x) { return std::cos(x); }, [](Vec4r x) { return SinCos(x).second; }, values);
}

// clang-format off
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, Sqr, ([](auto a) { return a * a; }), kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec4r, Sqrt, ([](auto a) { return std::sqrt(a); }), 0_r, 10_r, kEps);
// clang-format on

TEST(Vec4r, Store) {
  std::vector<real> result(
      5); // NOTE: Changed from an array on the stack to work around an MSVC optimizer bug.
  auto const v = Vec4r{1_r, 2_r, 3_r, 4_r};
  Store<0>((real*)nullptr, v);
  Store<0>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 4>{0_r, 0_r, 0_r, 0_r}), Span(&result[1], 4));
  Store<1>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 4>{1_r, 0_r, 0_r, 0_r}), Span(&result[1], 4));
  Store<2>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 4>{1_r, 2_r, 0_r, 0_r}), Span(&result[1], 4));
  Store<3>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 4>{1_r, 2_r, 3_r, 0_r}), Span(&result[1], 4));
  Store<4>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 4>{1_r, 2_r, 3_r, 4_r}), Span(&result[1], 4));
  result.clear();
  result.resize(5);
  Store(&result[1], v);
  EXPECT_SPAN_EQ((std::array<real, 4>{1_r, 2_r, 3_r, 4_r}), Span(&result[1], 4));

  result.clear();
  result.resize(5);
  Store(&result[1], v, 0);
  EXPECT_SPAN_EQ((std::array<real, 4>{0_r, 0_r, 0_r, 0_r}), Span(&result[1], 4));
  Store(&result[1], v, 1);
  EXPECT_SPAN_EQ((std::array<real, 4>{1_r, 0_r, 0_r, 0_r}), Span(&result[1], 4));
  Store(&result[1], v, 2);
  EXPECT_SPAN_EQ((std::array<real, 4>{1_r, 2_r, 0_r, 0_r}), Span(&result[1], 4));
  Store(&result[1], v, 3);
  EXPECT_SPAN_EQ((std::array<real, 4>{1_r, 2_r, 3_r, 0_r}), Span(&result[1], 4));
  Store(&result[1], v, 4);
  EXPECT_SPAN_EQ((std::array<real, 4>{1_r, 2_r, 3_r, 4_r}), Span(&result[1], 4));
}

TEST(Vec4r, StoreSelected) {
  Vec4r srcValues{1_r, 2_r, 3_r, 4_r};
  // Use int64_t for the condition type. This will be larger than type real in single-precision
  // builds, but that's OK.
  std::array<int64_t, 4> condition{};
  std::array<real, 4> expectedValues{};
  alignas(16) std::array<real, 6> dstBuffer{};
  dstBuffer[0] = 123_r;
  dstBuffer[5] = 456_r;
  Span<real> dstValues{&dstBuffer[1], 4}; // Not aligned
  for (int64_t mask = 0; mask < 16; ++mask) { // For each possible set of conditions
    int expectedCount = 0;
    for (int i = 0; i < 4; ++i) {
      condition[i] = (mask & (1 << i)) ? -1 : 0; // true = -1, false = 0
      if (condition[i]) {
        expectedValues[expectedCount++] = srcValues[i];
      }
    }
    auto countStored =
        StoreSelected(dstValues.data(), Load<Simd<int64_t, 4>>(condition.data()), srcValues);
    EXPECT_EQ(expectedCount, countStored);
    for (int i = 0; i < countStored; ++i) {
      EXPECT_EQ(expectedValues[i], dstValues[i]);
    }
  }
  // Sentinel values should remain unchanged
  EXPECT_EQ(123_r, dstBuffer[0]);
  EXPECT_EQ(456_r, dstBuffer[5]);
}

TEST(Vec4r, StoreTransposed) {
  alignas(alignof(Vec4r)) real result[14] = {};
  result[13] = 911_r; // Sentinel value
  auto* ptr = result + 1; // Not an aligned address
  Vec4r data[] = {
      Vec4r{1_r, 4_r, 7_r, 10_r}, Vec4r{2_r, 5_r, 8_r, 11_r}, Vec4r{3_r, 6_r, 9_r, 12_r}};
  StoreTransposed<1>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 12>{1_r, 2_r, 3_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}),
      Span(ptr, 12));
  StoreTransposed<2>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 12>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}),
      Span(ptr, 12));
  StoreTransposed<3>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 12>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 0_r, 0_r, 0_r}),
      Span(ptr, 12));
  StoreTransposed<4>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 12>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, 12_r}),
      Span(ptr, 12));
  StoreTransposed(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ(
      (std::array<real, 12>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, 12_r}),
      Span(ptr, 12));
  EXPECT_EQ(0_r, result[0]); // No change
  EXPECT_EQ(911_r, result[13]); // No change
}

// clang-format off
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec4r, Sub, -, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec4r, Tan, ([](auto a) { return std::tan(a); }), -1_r, 1_r, kEps);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec4r, BitwiseXOR, ^);
// clang-format on

TEST(Vec4r, Mat2x4f) {
  using Mat2x4f = NdArray<Vec4r, 2>;

  // Compile-time checks
  static_assert(1 == Mat2x4f::num_dims);
  static_assert(2 == Mat2x4f::dims[0]);
  static_assert(2 == Mat2x4f::size());
  static_assert(sizeof(Mat2x4f) == sizeof(Vec4r) * 2);

  // Default (zeros not identity)
  Mat2x4f x = {};
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, x[0]);
  EXPECT_VEC4R(0_r, 0_r, 0_r, 0_r, x[1]);

  // Non-default
  Mat2x4f x2 = {Vec4r{1_r, 2_r, 3_r, 4_r}, Vec4r{5_r, 6_r, 7_r, 8_r}};
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, x2[0]);
  EXPECT_VEC4R(5_r, 6_r, 7_r, 8_r, x2[1]);

  // Copy
  Mat2x4f y = x2;
  EXPECT_VEC4R(1_r, 2_r, 3_r, 4_r, y[0]);
  EXPECT_VEC4R(5_r, 6_r, 7_r, 8_r, y[1]);

  // Assign
  y = {Vec4r{2_r, 3_r, 4_r, 5_r}, Vec4r{6_r, 7_r, 8_r, 9_r}};
  EXPECT_VEC4R(2_r, 3_r, 4_r, 5_r, y[0]);
  EXPECT_VEC4R(6_r, 7_r, 8_r, 9_r, y[1]);

  // Ranged for
  Vec4r sum = {};
  for (auto const& row : y) {
    sum += row;
  }
  EXPECT_VEC4R(8_r, 10_r, 12_r, 14_r, sum);

  // Flatten
  EXPECT_EQ(2, Flatten(y).size());
  EXPECT_VEC4R(2_r, 3_r, 4_r, 5_r, Flatten(y)[0]);
  EXPECT_VEC4R(6_r, 7_r, 8_r, 9_r, Flatten(y)[1]);

  // operator[]
  Vec4r& ref = y[0];
  ref = Vec4r{11_r, 22_r, 33_r, 44_r};
  EXPECT_VEC4R(11_r, 22_r, 33_r, 44_r, y[0]);
  EXPECT_VEC4R(6_r, 7_r, 8_r, 9_r, y[1]);

  // operator==
  EXPECT_EQ(true, (y == Mat2x4f{Vec4r{11_r, 22_r, 33_r, 44_r}, Vec4r{6_r, 7_r, 8_r, 9_r}}));
  EXPECT_EQ(false, (y == Mat2x4f{Vec4r{11_r, 22_r, 33_r}, Vec4r{6_r, 7_r, 8_r, 911_r}}));

  // operator!=
  EXPECT_EQ(false, (y != Mat2x4f{Vec4r{11_r, 22_r, 33_r, 44_r}, Vec4r{6_r, 7_r, 8_r, 9_r}}));
  EXPECT_EQ(true, (y != Mat2x4f{Vec4r{11_r, 22_r, 33_r}, Vec4r{6_r, 7_r, 8_r, 911_r}}));

  // Math operators
  Mat2x4f a = {Vec4r{1_r, 2_r, 3_r, 4_r}, Vec4r{5_r, 6_r, 7_r, 8_r}};
  Mat2x4f b = {Vec4r{10_r, 20_r, 30_r, 40_r}, Vec4r{50_r, 60_r, 70_r, 80_r}};

  // Addition (just one example for now)
  EXPECT_TRUE(
      NearEqual(Mat2x4f{Vec4r{11_r, 22_r, 33_r, 44_r}, Vec4r{55_r, 66_r, 77_r, 88_r}}, (a + b)));
}

TEST(Vec4r, NdArrayConversion) {
  // Real2 <-- Vec4r
  EXPECT_EQ(Real2(1_r, 2_r), ToReal2(Vec4r(1_r, 2_r, 3_r, 4_r)));

  // Real3 <-- Vec4r
  EXPECT_EQ(Real3(1_r, 2_r, 3_r), ToReal3(Vec4r(1_r, 2_r, 3_r, 4_r)));

  // Real4 <-- Vec4r
  EXPECT_EQ(Real4(1_r, 2_r, 3_r, 4_r), ToReal4(Vec4r(1_r, 2_r, 3_r, 4_r)));

  // Vec4r <-- Real2
  EXPECT_EQ(Vec4r(1_r, 2_r, 0_r, 0_r), ToSimd(Real2(1_r, 2_r)));

  // Vec4r <-- Real3
  EXPECT_EQ(Vec4r(1_r, 2_r, 3_r, 0_r), ToSimd(Real3(1_r, 2_r, 3_r)));

  // Vec4r <-- Real4
  EXPECT_EQ(Vec4r(1_r, 2_r, 3_r, 4_r), ToSimd(Real4(1_r, 2_r, 3_r, 4_r)));

  // Matrix2x2r <--> VMatrix2x2r
  {
    NdArray<real, 2, 2> M = {Real2{1_r, 2_r}, Real2{3_r, 4_r}};
    VMatrix2x2r V = ToSimdMatrix(M);
    EXPECT_EQ(M, ToNdArray2x2(V));
  }

  // Matrix2x2r <--> VSymMatrix2x2r
  {
    Matrix2x2r M = {Real2{1_r, 3_r}, Real2{3_r, 2_r}};
    VSymMatrix2x2r V = ToSimdSymMatrix(M);
    EXPECT_EQ(M, ToNdArraySym2x2(V));
  }

  // Matrix3x3r <--> VMatrix3x3r
  {
    Matrix3x3r M = {Real3{1_r, 2_r, 3_r}, Real3{4_r, 5_r, 6_r}, Real3{7_r, 8_r, 9_r}};
    VMatrix3x3r V = ToSimdMatrix(M);
    EXPECT_EQ(M, ToNdArray3x3(V));
  }

  // Matrix4x4r <--> VMatrix4x4r
  {
    Matrix4x4r M = {
        Real4{1_r, 2_r, 3_r, 4_r},
        Real4{5_r, 6_r, 7_r, 8_r},
        Real4{9_r, 10_r, 11_r, 12_r},
        Real4{13_r, 14_r, 15_r, 16_r}};
    VMatrix4x4r V = ToSimdMatrix(M);
    EXPECT_EQ(M, ToNdArray(V));
  }

  // SymMatrix3x3r <--> Matrix3x3r
  {
    Matrix3x3r M = {Real3{1_r, 4_r, 5_r}, Real3{4_r, 2_r, 6_r}, Real3{5_r, 6_r, 3_r}};
    VSymMatrix3x3r V = ToSimdSymMatrix(M);
    EXPECT_EQ(M, ToNdArraySym3x3(V));
  }
}

// clang-format off
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec4r, Exp, ([](auto a) { return std::exp(a); }), -3_r, 3_r, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec4r, Ln, ([](auto a) { return std::log(a); }), 0.1_r, 10_r, kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec4r, Tanh, ([](auto a) { return std::tanh(a); }), kEps);
// clang-format on

TEST(Vec4r, ExpExtreme) {
  // Keep non-negative entries in array 'x'
  std::array<real, 4> x({60.0, 70.5, 80.0, 85.678});
  Vec4r v(x[0], x[1], x[2], x[3]);
  auto expv = Exp(v), expmv = Exp(-v);
  auto tol = real(2.0) * std::numeric_limits<real>::epsilon();
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR_RTOL(Vec4r::Get(expv, i), std::exp(x[i]), tol);
    EXPECT_LE(
        Abs(Vec4r::Get(expmv, i) - std::exp(-x[i])),
        Max(Vec4r::Get(expmv, i), std::exp(-x[i])) * tol);
  }
}
