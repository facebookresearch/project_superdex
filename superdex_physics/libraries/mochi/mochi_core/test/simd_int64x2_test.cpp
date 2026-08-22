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

static_assert(std::is_trivially_copyable_v<Simd<int64_t, 2>>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec2l);

#define EXPECT_VEC2L(e0, e1, actual)   \
  {                                    \
    Simd<int64_t, 2> vActual = actual; \
    EXPECT_EQ((e0), vActual[0]);       \
    EXPECT_EQ((e1), vActual[1]);       \
  }

/***********************************************************************************************
  Vec2l Class
*/

TEST(Vec2l, Class) {
  static_assert(Vec2l::kIsSupported);
  static_assert(!Vec2l::kIsComposite);
  static_assert(Vec2l::kIsEmulated == !(MOCHI_USE_SIMD));
  static_assert(sizeof(Vec2l) == sizeof(int64_t) * 2);
  static_assert(alignof(Vec2l) == alignof(typename Vec2l::NativeType));
  static_assert(std::is_same_v<Vec2l::Scalar, int64_t>);
  static_assert(Vec2l::kSize == 2);
  static_assert(Vec2l::size() == 2);

  // Construct from broadcast
  EXPECT_VEC2L(1, 1, Vec2l(1));

  // Construct from scalars
  EXPECT_VEC2L(1, 1, Vec2l(1));
  EXPECT_VEC2L(1, 2, Vec2l(1, 2));

  // Implicit conversion from scalar
  Vec2l a = 2;
  EXPECT_VEC2L(2, 2, a);
  a = 3;
  EXPECT_VEC2L(3, 3, a);

  // Copy construct
  Vec2l b{a};
  EXPECT_VEC2L(3, 3, b);

  // Copy assign
  a = Vec2l{4};
  b = a;
  EXPECT_VEC2L(4, 4, b);

  // Comparison
  EXPECT_EQ(true, (a == b));
  EXPECT_EQ(true, (a != Vec2l{}));
  EXPECT_EQ(false, (a != b));
  EXPECT_EQ(false, (a == Vec2l{}));

  // Unary operators
  Vec2l ones{-1};
  Vec2l zeros{0};
  EXPECT_EQ(ones, ~zeros);
  EXPECT_EQ(zeros, ~ones);
  EXPECT_VEC2L(-1, -2, -Vec2l(1, 2));

  // Binary operators
  a = Vec2l{1, 2};
  b = Vec2l{5, 6};
  EXPECT_VEC2L(6, 8, a + b);
  EXPECT_VEC2L(4, 5, a + int64_t(3));
  EXPECT_VEC2L(4, 5, int64_t(3) + a);
  EXPECT_VEC2L(-4, -4, a - b);
  EXPECT_VEC2L(-2, -1, a - 3);
  EXPECT_VEC2L(2, 1, int64_t(3) - a);
  EXPECT_VEC2L(5, 12, a * b);
  EXPECT_VEC2L(3, 6, a * 3);
  EXPECT_VEC2L(3, 6, int64_t(3) * a);
  EXPECT_VEC2L(5, 3, b / a);
  EXPECT_VEC2L(0, 0, a / 3);
  EXPECT_VEC2L(3, 1, int64_t(3) / a);
  EXPECT_EQ(a, a & ones);
  EXPECT_EQ(zeros, a & zeros);
  EXPECT_EQ(ones, Vec2l(a | ones));
  EXPECT_EQ(a, a | zeros);
  EXPECT_EQ(~a, a ^ ones);
  EXPECT_EQ(a, a ^ zeros);

  // Update operators
  a = Vec2l{1, 2};
  a += b;
  EXPECT_VEC2L(6, 8, a);
  a = Vec2l{1, 2};
  a += 3;
  EXPECT_VEC2L(4, 5, a);
  a = Vec2l{1, 2};
  a -= b;
  EXPECT_VEC2L(-4, -4, a);
  a = Vec2l{1, 2};
  a -= 3;
  EXPECT_VEC2L(-2, -1, a);
  a = Vec2l{1, 2};
  a *= b;
  EXPECT_VEC2L(5, 12, a);
  a = Vec2l{1, 2};
  a *= 3;
  EXPECT_VEC2L(3, 6, a);
  a = Vec2l{1, 2};
  b = Vec2l{5, 6};
  b /= a;
  EXPECT_VEC2L(5, 3, b);
  a = Vec2l{1, 2};
  a /= 3;
  EXPECT_VEC2L(0, 0, a);

  b = a;
  b &= ones;
  EXPECT_EQ(a, b);
  a = Vec2l{1, 2};
  a &= zeros;
  EXPECT_EQ(zeros, a);
  a = Vec2l{1, 2};
  a |= ones;
  EXPECT_EQ(ones, a);
  a = b = Vec2l{1, 2};
  b |= zeros;
  EXPECT_EQ(a, b);
  a = b = Vec2l{1, 2};
  b ^= ones;
  EXPECT_EQ(~a, b);
  a = b = Vec2l{1, 2};
  b ^= zeros;
  EXPECT_EQ(a, b);
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec2l, Add, +);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec2l, BitwiseAND, &);
MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(Vec2l);

TEST(Vec2l, AllTrue) {
  int64_t zeros = 0;
  int64_t ones = -1;
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      auto vec = Vec2l{a ? ones : zeros, b ? ones : zeros};
      EXPECT_EQ(!!a, AllTrue<1>(vec));
      EXPECT_EQ(a && b, AllTrue<2>(vec));
      EXPECT_EQ(a && b, AllTrue(vec));
    }
  }
}

TEST(Vec2l, Blend) {
  auto a = Vec2l{1, 2};
  auto b = Vec2l{3, 4};
  EXPECT_VEC2L(1, 2, (Blend<0, 0>(a, b)));
  EXPECT_VEC2L(1, 4, (Blend<0, 1>(a, b)));
  EXPECT_VEC2L(3, 2, (Blend<1, 0>(a, b)));
  EXPECT_VEC2L(3, 4, (Blend<1, 1>(a, b)));
}

TEST(Vec2l, Broadcast) {
  // Broadcast scalar
  EXPECT_VEC2L(1, 1, Broadcast<Vec2l>(1));

  // Broadcast from address
  int64_t const s = 2;
  EXPECT_VEC2L(2, 2, Broadcast<Vec2l>(&s));

  // Broadcast ith element
  auto v = Vec2l{1, 2};
  EXPECT_VEC2L(1, 1, Broadcast<0>(v));
  EXPECT_VEC2L(2, 2, Broadcast<1>(v));

  // Broadcast ith element
  EXPECT_VEC2L(1, 1, Broadcast(v, 0));
  EXPECT_VEC2L(2, 2, Broadcast(v, 1));
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec2l, Div, /);

TEST(Vec2l, Get) {
  auto a = Vec2l{1, 2};

  // Fast template version
  EXPECT_EQ(1, Get0(a));
  EXPECT_EQ(1, Get<0>(a));
  EXPECT_EQ(2, Get<1>(a));

  // Slower runtime version
  EXPECT_EQ(1, Get(a, 0));
  EXPECT_EQ(2, Get(a, 1));

  // Same but with operator[] (read only)
  EXPECT_EQ(1, a[0]);
  EXPECT_EQ(2, a[1]);
}

TEST(Vec2l, Greater) {
  auto a = Vec2l{1, 4};
  auto b = Vec2l{2, 3};
  EXPECT_VEC2L(1, 0, (b > a) & Vec2l{1});
  EXPECT_VEC2L(0, 1, (a > b) & Vec2l{1});
  EXPECT_VEC2L(0, 0, (a > a) & Vec2l{1});
  EXPECT_VEC2L(0, 0, (a > Vec2l{4}) & Vec2l{1});
  EXPECT_VEC2L(0, 0, (a > Vec2l{5}) & Vec2l{1});
  EXPECT_VEC2L(1, 0, (Vec2l{4} > a) & Vec2l{1});
  EXPECT_VEC2L(1, 1, (Vec2l{5} > a) & Vec2l{1});
}

TEST(Vec2l, GreaterEqual) {
  auto a = Vec2l{1, 4};
  auto b = Vec2l{2, 3};
  EXPECT_VEC2L(1, 0, (b >= a) & Vec2l{1});
  EXPECT_VEC2L(0, 1, (a >= b) & Vec2l{1});
  EXPECT_VEC2L(1, 1, (a >= a) & Vec2l{1});
  EXPECT_VEC2L(0, 1, (a >= Vec2l{4}) & Vec2l{1});
  EXPECT_VEC2L(0, 0, (a >= Vec2l{5}) & Vec2l{1});
  EXPECT_VEC2L(1, 1, (Vec2l{4} >= a) & Vec2l{1});
  EXPECT_VEC2L(1, 1, (Vec2l{5} >= a) & Vec2l{1});
}

TEST(Vec2l, HMax) {
  auto a = Vec2l{1, 2};
  EXPECT_EQ(2, HMax(a));
}

TEST(Vec2l, HMin) {
  auto a = Vec2l{2, 1};
  EXPECT_EQ(1, HMin(a));
}

TEST(Vec2l, HSum) {
  auto a = Vec2l{1, 2};
  EXPECT_EQ(3, HSum(a));
}

TEST(Vec2l, Less) {
  auto a = Vec2l{1, 4};
  auto b = Vec2l{2, 3};
  EXPECT_VEC2L(0, 1, (b < a) & Vec2l{1});
  EXPECT_VEC2L(1, 0, (a < b) & Vec2l{1});
  EXPECT_VEC2L(0, 0, (a < a) & Vec2l{1});
  EXPECT_VEC2L(1, 0, (a < Vec2l{4}) & Vec2l{1});
  EXPECT_VEC2L(1, 1, (a < Vec2l{5}) & Vec2l{1});
  EXPECT_VEC2L(0, 0, (Vec2l{4} < a) & Vec2l{1});
  EXPECT_VEC2L(0, 0, (Vec2l{5} < a) & Vec2l{1});
}

TEST(Vec2l, LessEqual) {
  auto a = Vec2l{1, 4};
  auto b = Vec2l{2, 3};
  EXPECT_VEC2L(0, 1, (b <= a) & Vec2l{1});
  EXPECT_VEC2L(1, 0, (a <= b) & Vec2l{1});
  EXPECT_VEC2L(1, 1, (a <= a) & Vec2l{1});
  EXPECT_VEC2L(1, 1, (a <= Vec2l{4}) & Vec2l{1});
  EXPECT_VEC2L(1, 1, (a <= Vec2l{5}) & Vec2l{1});
  EXPECT_VEC2L(0, 1, (Vec2l{4} <= a) & Vec2l{1});
  EXPECT_VEC2L(0, 0, (Vec2l{5} <= a) & Vec2l{1});
}

// Test Equal. NotEqual<N> for N > 1 requires AnyTrue, which is not supported for Vec2l, so only
// NotEqual<1> can be tested via the helper.
TEST(Vec2l, Equal) {
  auto a = Vec2l{1, 2};
  auto b = a;
  auto c = Vec2l{9, 2};
  auto d = Vec2l{1, 9};

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

// Test both VEqual and VNotEqual
template <class V>
static void ExpectVEqual(int64_t e0, int64_t e1, V a, V b) {
  EXPECT_VEC2L(e0, e1, VEqual(a, b) & Vec2l{1});
  EXPECT_VEC2L(int64_t(!e0), int64_t(!e1), VNotEqual(a, b) & Vec2l{1});
}

TEST(Vec2l, VEqual) {
  auto a = Vec2l{1, 2};
  auto b = Vec2l{1, 9};
  auto c = Vec2l{9, 2};
  ExpectVEqual(1, 1, a, a);
  ExpectVEqual(1, 0, a, b);
  ExpectVEqual(0, 1, a, c);
}

TEST(Vec2l, Load) {
  alignas(alignof(Vec2l)) int64_t const values[] = {0, 1, 2, 3, 4};
  EXPECT_VEC2L(0, 0, (Load<0, Vec2l>(nullptr)));
  EXPECT_VEC2L(1, 0, (Load<1, Vec2l>(values + 1)));
  EXPECT_VEC2L(1, 2, (Load<2, Vec2l>(values + 1)));
  EXPECT_VEC2L(1, 2, (Load<Vec2l>(values + 1)));

  EXPECT_VEC2L(1, 0, (Load<Vec2l>(values + 1, 1)));
  EXPECT_VEC2L(1, 2, (Load<Vec2l>(values + 1, 2)));
}

TEST(Vec2l, LoadTransposed) {
  alignas(alignof(Vec2l)) int64_t const values[] = {0, 1, 2, 3, 4, 5, 6};
  auto const* ptr = values + 1; // Not an aligned address
  Vec2l loaded[3] = {};
  LoadTransposed<1>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC2L(1, 0, loaded[0]);
  EXPECT_VEC2L(2, 0, loaded[1]);
  EXPECT_VEC2L(3, 0, loaded[2]);
  LoadTransposed<2>(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC2L(1, 4, loaded[0]);
  EXPECT_VEC2L(2, 5, loaded[1]);
  EXPECT_VEC2L(3, 6, loaded[2]);
  LoadTransposed(ptr, loaded[0], loaded[1], loaded[2]);
  EXPECT_VEC2L(1, 4, loaded[0]);
  EXPECT_VEC2L(2, 5, loaded[1]);
  EXPECT_VEC2L(3, 6, loaded[2]);
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec2l, Mul, *);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec2l, Neg, -);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec2l, BitwiseNOT, ~);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec2l, BitwiseOR, |);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec2l, LogicalAND, &&, &);
MOCHI_SIMD_TEST_BINARY_LOGICAL_OP(Vec2l, LogicalOR, ||, |);

TEST(Vec2l, Select) {
  auto a = Vec2l{1, 2};
  auto b = Vec2l{3, 4};
  EXPECT_VEC2L(3, 4, Select(Vec2l{0, 0}, a, b));
  EXPECT_VEC2L(1, 4, Select(Vec2l{-1, 0}, a, b));
  EXPECT_VEC2L(3, 2, Select(Vec2l{0, -1}, a, b));
  EXPECT_VEC2L(1, 2, Select(Vec2l{-1, -1}, a, b));
}

TEST(Vec2l, Sequence) {
  EXPECT_VEC2L(0, 1, Sequence<Vec2l>());
}

TEST(Vec2l, Shuffle) {
  auto v = Vec2l{1, 2};
  EXPECT_VEC2L(1, 1, (Shuffle<0, 0>(v)));
  EXPECT_VEC2L(1, 2, (Shuffle<0, 1>(v)));
  EXPECT_VEC2L(2, 1, (Shuffle<1, 0>(v)));
  EXPECT_VEC2L(2, 2, (Shuffle<1, 1>(v)));
}

TEST(Vec2l, SimdZero) {
  EXPECT_VEC2L(0, 0, SimdZero<Vec2l>());
}

MOCHI_SIMD_TEST_UNARY_FN_EXACT(Vec2l, Sqr, ([](auto a) { return a * a; }));
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec2l, Max, ([](auto a, auto b) { return std::max(a, b); }));
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec2l, Min, ([](auto a, auto b) { return std::min(a, b); }));

TEST(Vec2l, Store) {
  std::vector<int64_t> result(
      5); // NOTE: Changed from an array on the stack to work around an MSVC optimizer bug.
  auto const v = Vec2l{1, 2};
  Store<0>((int64_t*)nullptr, v);
  Store<0>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 2>{0, 0}), Span(&result[1], 2));
  Store<1>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 2>{1, 0}), Span(&result[1], 2));
  Store<2>(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 2>{1, 2}), Span(&result[1], 2));
  result.clear();
  result.resize(5);
  Store(&result[1], v, 0);
  EXPECT_SPAN_EQ((std::array<int64_t, 2>{0, 0}), Span(&result[1], 2));
  Store(&result[1], v);
  EXPECT_SPAN_EQ((std::array<int64_t, 2>{1, 2}), Span(&result[1], 2));
}

TEST(Vec2l, StoreSelected) {
  Vec2l srcValues{1, 2};
  std::array<int64_t, 2> condition{};
  std::array<int64_t, 2> expectedValues{};
  alignas(16) std::array<int64_t, 4> dstBuffer{};
  dstBuffer[0] = 123;
  dstBuffer[3] = 456;
  Span<int64_t> dstValues{&dstBuffer[1], 2}; // Not aligned
  for (int mask = 0; mask < 4; ++mask) { // For each possible set of conditions
    int expectedCount = 0;
    for (int i = 0; i < 2; ++i) {
      condition[i] = (mask & (1 << i)) ? int64_t(-1) : int64_t(0); // true = -1, false = 0
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

TEST(Vec2l, StoreTransposed) {
  alignas(alignof(Vec2l)) int64_t result[8] = {};
  result[7] = 911; // Sentinel value
  auto* ptr = result + 1; // Not an aligned address
  Vec2l data[3] = {Vec2l{1, 4}, Vec2l{2, 5}, Vec2l{3, 6}};
  StoreTransposed<1>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int64_t, 6>{1, 2, 3, 0, 0, 0}), Span(ptr, 6));
  StoreTransposed<2>(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int64_t, 6>{1, 2, 3, 4, 5, 6}), Span(ptr, 6));
  StoreTransposed(ptr, data[0], data[1], data[2]);
  EXPECT_SPAN_EQ((std::array<int64_t, 6>{1, 2, 3, 4, 5, 6}), Span(ptr, 6));
  EXPECT_EQ(0, result[0]); // No change
  EXPECT_EQ(911, result[7]); // No change
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec2l, Sub, -);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec2l, BitwiseXOR, ^);

TEST(Vec2l, ShiftLeft) {
  Simd<int64_t, 2> v(1, 5);
  EXPECT_VEC2L(8, 40, v << 3);
  EXPECT_EQ(v, v << 0);
}

TEST(Vec2l, ShiftRight) {
  Vec2l v(8, 40);
  EXPECT_VEC2L(1, 5, ShiftRight<3>(v));
  EXPECT_EQ(v, ShiftRight<0>(v));
}
