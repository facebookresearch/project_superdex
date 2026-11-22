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
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

using namespace mochi;
using namespace mochi::simd_test;

static_assert(std::is_trivially_copyable_v<Vec16r>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec16r);

namespace {

template <int I = 0>
void ExpectCompileTimeBroadcastsAndSets(
    Vec16r source,
    std::array<real, Vec16r::kSize> const& values) {
  if constexpr (I < Vec16r::kSize) {
    auto const broadcast = Broadcast<I>(source);
    auto const changed = Set<I>(source, -values[I]);
    for (int lane = 0; lane < Vec16r::kSize; ++lane) {
      EXPECT_EQ(values[I], broadcast[lane]);
      EXPECT_EQ(lane == I ? -values[I] : values[lane], changed[lane]);
    }
    ExpectCompileTimeBroadcastsAndSets<I + 1>(source, values);
  }
}

template <int I = 0>
void ExpectIsTrueLanes(Vec16r mask, std::array<bool, Vec16r::kSize> const& expected) {
  if constexpr (I < Vec16r::kSize) {
    EXPECT_EQ(expected[I], IsTrue<I>(mask) != 0);
    ExpectIsTrueLanes<I + 1>(mask, expected);
  }
}

template <int N = 1>
void ExpectNearPrefixes(Vec16r a, Vec16r b) {
  bool expectedEqual = true;
  bool expectedZero = true;
  for (int i = 0; i < N; ++i) {
    expectedEqual = expectedEqual && Abs(a[i] - b[i]) <= real{0.125};
    expectedZero = expectedZero && Abs(a[i]) <= real{8};
  }
  EXPECT_EQ(expectedEqual, NearEqual<N>(a, b, real{0.125}));
  EXPECT_EQ(expectedZero, NearZero<N>(a, real{8}));
  if constexpr (N < Vec16r::kSize) {
    ExpectNearPrefixes<N + 1>(a, b);
  }
}

template <int N = 2>
void ExpectNormPrefixes(Vec16r values) {
  real expectedNormSqr = real{};
  for (int i = 0; i < N; ++i) {
    expectedNormSqr += values[i] * values[i];
  }
  auto const expectedNorm = Sqrt(expectedNormSqr);
  EXPECT_NEAR(expectedNormSqr, NormSqr<N>(values), Abs(expectedNormSqr) * kEps);
  EXPECT_NEAR(expectedNorm, Norm<N>(values), Abs(expectedNorm) * kEps);
  auto const vectorNormSqr = VNormSqr<N>(values);
  auto const vectorNorm = VNorm<N>(values);
  auto const normalized = Normalize<N>(values);
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_NEAR(expectedNormSqr, vectorNormSqr[i], Abs(expectedNormSqr) * kEps);
    EXPECT_NEAR(expectedNorm, vectorNorm[i], Abs(expectedNorm) * kEps);
    EXPECT_NEAR(values[i] / expectedNorm, normalized[i], Abs(values[i] / expectedNorm) * kEps);
  }
  if constexpr (N < Vec16r::kSize) {
    ExpectNormPrefixes<N + 1>(values);
  }
}

template <int N = 1>
void ExpectPoisonedTailReductions(Vec16r values) {
  real expectedMin = values[0];
  real expectedMax = values[0];
  real expectedSum = values[0];
  for (int i = 1; i < N; ++i) {
    expectedMin = Min(expectedMin, values[i]);
    expectedMax = Max(expectedMax, values[i]);
    expectedSum += values[i];
  }
  auto poisoned = values;
  for (int i = N; i < Vec16r::kSize; ++i) {
    poisoned = Set(poisoned, i, std::numeric_limits<real>::quiet_NaN());
  }
  EXPECT_EQ(expectedMin, HMin<N>(poisoned));
  EXPECT_EQ(expectedMax, HMax<N>(poisoned));
  EXPECT_NEAR(expectedSum, HSum<N>(poisoned), Abs(expectedSum) * kEps);
  EXPECT_NEAR(expectedSum, Dot<N>(poisoned, Vec16r{real{1}}), Abs(expectedSum) * kEps);
  if constexpr (N < Vec16r::kSize) {
    ExpectPoisonedTailReductions<N + 1>(values);
  }
}

template <int I = 0>
void ExpectCompileTimeLanes(Vec16r v, std::array<real, Vec16r::kSize> const& values) {
  if constexpr (I < Vec16r::kSize) {
    EXPECT_EQ(values[I], Get<I>(v));
    ExpectCompileTimeLanes<I + 1>(v, values);
  }
}

template <int N>
void TestCompileTimePartial(std::array<real, Vec16r::kSize> const& values) {
  auto const v = Load<N, Vec16r>(values.data());
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_EQ(i < N ? values[i] : real{}, v[i]);
  }

  constexpr real kSentinel = real{-911};
  std::array<real, Vec16r::kSize + 2> stored{};
  stored.fill(kSentinel);
  Store<N>(stored.data() + 1, Load<Vec16r>(values.data()));
  EXPECT_EQ(kSentinel, stored.front());
  EXPECT_EQ(kSentinel, stored.back());
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_EQ(i < N ? values[i] : kSentinel, stored[i + 1]);
  }
}

template <size_t... Is>
void TestAllCompileTimePartials(
    std::array<real, Vec16r::kSize> const& values,
    std::index_sequence<Is...>) {
  (TestCompileTimePartial<static_cast<int>(Is)>(values), ...);
}

void ExpectMaskLane(Vec16r mask, int lane, bool expected) {
  using MaskScalar = std::conditional_t<sizeof(real) == sizeof(int), int, int64_t>;
  real const value = mask[lane];
  MaskScalar bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  EXPECT_EQ(expected ? MaskScalar{-1} : MaskScalar{0}, bits);
}

template <int N = 1>
void ExpectPrefixMaskReductions(Vec16r mask, std::array<bool, Vec16r::kSize> const& expected) {
  bool expectedAll = true;
  bool expectedAny = false;
  for (int i = 0; i < N; ++i) {
    expectedAll = expectedAll && expected[i];
    expectedAny = expectedAny || expected[i];
  }
  EXPECT_EQ(expectedAll, AllTrue<N>(mask));
  EXPECT_EQ(expectedAny, AnyTrue<N>(mask));
  if constexpr (N < Vec16r::kSize) {
    ExpectPrefixMaskReductions<N + 1>(mask, expected);
  }
}

void ExpectEveryPrefixMaskBoundary() {
  std::array<real, Vec16r::kSize> values{};
  std::array<bool, Vec16r::kSize> expected{};
  for (int lane = 0; lane < Vec16r::kSize; ++lane) {
    values.fill(real{1});
    expected.fill(true);
    values[lane] = real{};
    expected[lane] = false;
    ExpectPrefixMaskReductions(Load<Vec16r>(values.data()) > Vec16r{}, expected);

    values.fill(real{});
    expected.fill(false);
    values[lane] = real{1};
    expected[lane] = true;
    ExpectPrefixMaskReductions(Load<Vec16r>(values.data()) > Vec16r{}, expected);
  }
}

template <int N = 1>
void ExpectReductions(Vec16r values) {
  real expectedMin = values[0];
  real expectedMax = values[0];
  real expectedSum = values[0];
  [[maybe_unused]] real expectedProduct = values[0];
  for (int i = 1; i < N; ++i) {
    expectedMin = std::min(expectedMin, values[i]);
    expectedMax = std::max(expectedMax, values[i]);
    expectedSum += values[i];
    expectedProduct *= values[i];
  }

  EXPECT_EQ(expectedMin, HMin<N>(values));
  EXPECT_EQ(expectedMax, HMax<N>(values));
  EXPECT_NEAR(expectedSum, HSum<N>(values), Abs(expectedSum) * kEps);
  EXPECT_NEAR(expectedSum, Dot<N>(values, Vec16r{real{1}}), Abs(expectedSum) * kEps);
  if constexpr (std::is_same_v<typename Vec16r::Scalar, double>) {
    EXPECT_NEAR(expectedProduct, HProd<N>(values), Abs(expectedProduct) * kEps);
  }

  if constexpr (N < Vec16r::kSize) {
    ExpectReductions<N + 1>(values);
  }
}

template <int N = 1>
void ExpectTransposedIO(std::array<real, 3 * Vec16r::kSize> const& tuples) {
  Vec16r x;
  Vec16r y;
  Vec16r z;
  LoadTransposed<N>(tuples.data(), x, y, z);
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_EQ(i < N ? tuples[3 * i] : real{}, x[i]);
    EXPECT_EQ(i < N ? tuples[3 * i + 1] : real{}, y[i]);
    EXPECT_EQ(i < N ? tuples[3 * i + 2] : real{}, z[i]);
  }

  constexpr real kSentinel = real{-911};
  std::array<real, 3 * Vec16r::kSize + 2> stored{};
  stored.fill(kSentinel);
  StoreTransposed<N>(stored.data() + 1, x, y, z);
  EXPECT_EQ(kSentinel, stored.front());
  EXPECT_EQ(kSentinel, stored.back());
  for (int i = 0; i < 3 * Vec16r::kSize; ++i) {
    EXPECT_EQ(i < 3 * N ? tuples[i] : kSentinel, stored[i + 1]);
  }

  if constexpr (N < Vec16r::kSize) {
    ExpectTransposedIO<N + 1>(tuples);
  }
}

} // namespace

TEST(Vec16r, TypeProperties) {
  static_assert(Vec16r::kIsSupported);
  static_assert(Vec16r::kSize == 16);
  static_assert(Vec16r::size() == 16);
  static_assert(sizeof(Vec16r) == sizeof(real) * Vec16r::kSize);
  static_assert(alignof(Vec16r) == alignof(typename Vec16r::NativeType));
  static_assert(std::is_same_v<Vec16r::Scalar, real>);

#if MOCHI_USE_SIMD
  constexpr bool kIsNative = MOCHI_ARCH_X64_AVX512 && !MOCHI_USE_DOUBLE_PRECISION;
  static_assert(Vec16r::kIsComposite == !kIsNative);
  static_assert(!Vec16r::kIsEmulated);
#else
  static_assert(!Vec16r::kIsComposite);
  static_assert(Vec16r::kIsEmulated);
#endif
}

TEST(Vec16r, HalfConstructorAndLaneAccess) {
  std::array<real, Vec16r::kSize> values{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    values[i] = static_cast<real>(i + 1);
  }
  auto const v = Load<Vec16r>(values.data());
  EXPECT_EQ(v, (Vec16r{GetHalf<0>(v), GetHalf<1>(v)}));
  EXPECT_EQ(values[0], Get0(v));
  ExpectCompileTimeLanes(v, values);
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_EQ(values[i], v[i]);
  }
}

TEST(Vec16r, PartialLoadStore) {
  std::array<real, Vec16r::kSize> values{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    values[i] = static_cast<real>(i + 1);
  }

  constexpr real kSentinel = real{-911};
  for (int count = 0; count <= Vec16r::kSize; ++count) {
    auto const v = Load<Vec16r>(values.data(), count);
    for (int i = 0; i < Vec16r::kSize; ++i) {
      EXPECT_EQ(i < count ? values[i] : real{}, v[i]);
    }

    std::array<real, Vec16r::kSize + 2> stored{};
    stored.fill(kSentinel);
    Store(stored.data() + 1, Load<Vec16r>(values.data()), count);
    EXPECT_EQ(kSentinel, stored.front());
    EXPECT_EQ(kSentinel, stored.back());
    for (int i = 0; i < Vec16r::kSize; ++i) {
      EXPECT_EQ(i < count ? values[i] : kSentinel, stored[i + 1]);
    }
  }

  TestAllCompileTimePartials(values, std::make_index_sequence<Vec16r::kSize + 1>{});
}

TEST(Vec16r, StoreSelectedExhaustive) {
  std::array<real, Vec16r::kSize> values{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    values[i] = static_cast<real>(i + 1);
  }
  auto const source = Load<Vec16r>(values.data());
  std::array<int64_t, Vec16r::kSize> conditions{};
  constexpr real kSentinel = real{-911};
  std::array<real, Vec16r::kSize + 2> destination{};

  for (uint32_t mask = 0; mask < (uint32_t{1} << Vec16r::kSize); ++mask) {
    destination.fill(kSentinel);
    for (int i = 0; i < Vec16r::kSize; ++i) {
      conditions[i] = (mask & (uint32_t{1} << i)) != 0 ? int64_t{-1} : int64_t{0};
    }

    int const stored = StoreSelected(
        destination.data() + 1, Load<Simd<int64_t, Vec16r::kSize>>(conditions.data()), source);
    EXPECT_EQ(std::popcount(mask), stored);
    int expectedIndex = 0;
    for (int i = 0; i < Vec16r::kSize; ++i) {
      if ((mask & (uint32_t{1} << i)) != 0) {
        EXPECT_EQ(values[i], destination[++expectedIndex]);
      }
    }
    EXPECT_EQ(kSentinel, destination.front());
    if constexpr (MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512) {
      for (int i = stored + 1; i < static_cast<int>(destination.size()); ++i) {
        EXPECT_EQ(kSentinel, destination[i]);
      }
    }
  }
}

TEST(Vec16r, BroadcastAndSet) {
  std::array<real, Vec16r::kSize> values{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    values[i] = static_cast<real>(i + 1);
  }
  auto const source = Load<Vec16r>(values.data());

  auto const pointerBroadcast = Broadcast<Vec16r>(values.data() + 5);
  auto const laneBroadcast = Broadcast<11>(source);
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_EQ(values[5], pointerBroadcast[i]);
    EXPECT_EQ(values[11], laneBroadcast[i]);
  }

  auto runtimeSet = source;
  for (int i = 0; i < Vec16r::kSize; ++i) {
    runtimeSet = Set(runtimeSet, i, -values[i]);
  }
  EXPECT_EQ(-values[7], Set<7>(source, -values[7])[7]);
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_EQ(-values[i], runtimeSet[i]);
  }
}

TEST(Vec16r, ComparisonsMasksAndSelect) {
  std::array<real, Vec16r::kSize> lhsValues{};
  std::array<real, Vec16r::kSize> rhsValues{};
  std::array<bool, Vec16r::kSize> expectedLess{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    lhsValues[i] = static_cast<real>(i);
    rhsValues[i] = static_cast<real>(i + (i % 3) - 1);
  }
  lhsValues[3] = std::numeric_limits<real>::quiet_NaN();
  rhsValues[12] = std::numeric_limits<real>::quiet_NaN();

  auto const lhs = Load<Vec16r>(lhsValues.data());
  auto const rhs = Load<Vec16r>(rhsValues.data());
  auto const less = lhs < rhs;
  auto const greater = lhs > rhs;
  auto const lessEqual = lhs <= rhs;
  auto const greaterEqual = lhs >= rhs;
  auto const equal = VEqual(lhs, rhs);
  auto const notEqual = VNotEqual(lhs, rhs);
  auto const selected = Select(less, lhs, rhs);
  for (int i = 0; i < Vec16r::kSize; ++i) {
    bool const unordered = i == 3 || i == 12;
    bool const expectedEqual = !unordered && lhsValues[i] == rhsValues[i];
    expectedLess[i] = !unordered && lhsValues[i] < rhsValues[i];
    ExpectMaskLane(less, i, expectedLess[i]);
    ExpectMaskLane(greater, i, !unordered && lhsValues[i] > rhsValues[i]);
    ExpectMaskLane(lessEqual, i, !unordered && lhsValues[i] <= rhsValues[i]);
    ExpectMaskLane(greaterEqual, i, !unordered && lhsValues[i] >= rhsValues[i]);
    ExpectMaskLane(equal, i, expectedEqual);
    ExpectMaskLane(notEqual, i, unordered || !expectedEqual);
    if (expectedLess[i]) {
      EXPECT_EQ(lhsValues[i], selected[i]);
    } else if (i == 12) {
      EXPECT_TRUE(std::isnan(selected[i]));
    } else {
      EXPECT_EQ(rhsValues[i], selected[i]);
    }
  }
  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
  Vec16r const finite{real{1}};
  EXPECT_TRUE(finite == finite);
  EXPECT_FALSE(finite != finite);
  ExpectPrefixMaskReductions(less, expectedLess);
  ExpectEveryPrefixMaskBoundary();
}

TEST(Vec16r, IndexedAndTransposedIO) {
  std::array<real, 2 * Vec16r::kSize> values{};
  using IndexScalar = std::conditional_t<std::is_same_v<real, double>, int64_t, int>;
  std::array<IndexScalar, Vec16r::kSize> indices{};
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = static_cast<real>(i + 1);
  }
  for (int i = 0; i < Vec16r::kSize; ++i) {
    indices[i] = 2 * (Vec16r::kSize - i) - 1;
  }
  auto const indexed =
      LoadIndexed<Vec16r>(values.data(), Load<Simd<IndexScalar, Vec16r::kSize>>(indices.data()));
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_EQ(values[indices[i]], indexed[i]);
  }

  std::array<real, 3 * Vec16r::kSize> tuples{};
  for (int i = 0; i < static_cast<int>(tuples.size()); ++i) {
    tuples[i] = static_cast<real>(i + 1);
  }
  ExpectTransposedIO(tuples);
}

TEST(Vec16r, Reductions) {
  std::array<real, Vec16r::kSize> values{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    values[i] = static_cast<real>(1.0 + 0.0625 * i);
  }
  ExpectReductions(Load<Vec16r>(values.data()));
}

TEST(Vec16r, HSumEach) {
  auto const a = StaticCast<Vec16r>(Sequence<Vec16i>() + 1);
  EXPECT_EQ(
      (Vec4r{136_r, 1360_r, 13600_r, 136000_r}), HSumEach(a, 10_r * a, 100_r * a, 1000_r * a));
}

TEST(Vec16r, Reciprocal) {
  std::array<real, Vec16r::kSize> values{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    values[i] = static_cast<real>(i + 3);
  }
  auto const reciprocal = RcpApprox(Load<Vec16r>(values.data()));
  for (int i = 0; i < Vec16r::kSize; ++i) {
    real const expected = real{1} / values[i];
    EXPECT_NEAR(expected, reciprocal[i], Abs(expected) * real{1e-2});
  }
}

TEST(Vec16r, IsFinite) {
  std::array<real, Vec16r::kSize> values{};
  values.fill(real{1});
  values[0] = std::numeric_limits<real>::infinity();
  values[5] = -std::numeric_limits<real>::infinity();
  values[10] = std::numeric_limits<real>::quiet_NaN();
  values[15] = std::numeric_limits<real>::signaling_NaN();

  EXPECT_TRUE(IsFinite(Vec16r{real{1}}));
  auto const vector = Load<Vec16r>(values.data());
  EXPECT_FALSE(IsFinite(vector));
  auto const mask = VIsFinite(vector);
  for (int i = 0; i < Vec16r::kSize; ++i) {
    bool const expected = i != 0 && i != 5 && i != 10 && i != 15;
    ExpectMaskLane(mask, i, expected);
  }
}

MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec16r, Abs, std::abs, kEps);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec16r, Add, +, kEps);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec16r, BitwiseAND, &);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec16r, Div, /, kEps);
MOCHI_SIMD_TEST_BINARY_FN_NEAR(Vec16r, Max, ([](auto a, auto b) { return std::max(a, b); }), kEps);
MOCHI_SIMD_TEST_BINARY_FN_NEAR(Vec16r, Min, ([](auto a, auto b) { return std::min(a, b); }), kEps);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec16r, Mul, *, kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(
    Vec16r,
    MulAdd,
    ([](auto a, auto b, auto c) { return a * b + c; }),
    kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(
    Vec16r,
    MulSub,
    ([](auto a, auto b, auto c) { return a * b - c; }),
    kEps);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec16r, Neg, -);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(
    Vec16r,
    NegMulAdd,
    ([](auto a, auto b, auto c) { return -(a * b) + c; }),
    kEps);
MOCHI_SIMD_TEST_TERNARY_FN_NEAR(
    Vec16r,
    NegMulSub,
    ([](auto a, auto b, auto c) { return -(a * b) - c; }),
    kEps);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec16r, BitwiseNOT, ~);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec16r, BitwiseOR, |);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec16r, Floor, std::floor, kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec16r, FastRound, std::nearbyint, kEps);
MOCHI_SIMD_TEST_UNARY_FN_NEAR(
    Vec16r,
    RcpSqrtApprox,
    ([](auto a) { return 1 / std::sqrt(a); }),
    real{1e-3});
MOCHI_SIMD_TEST_UNARY_FN_NEAR(Vec16r, Sqr, ([](auto a) { return a * a; }), kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(
    Vec16r,
    Sqrt,
    ([](auto a) { return std::sqrt(a); }),
    real{0},
    real{16},
    kEps);
MOCHI_SIMD_TEST_BINARY_OP_NEAR(Vec16r, Sub, -, kEps);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec16r, BitwiseXOR, ^);

TEST(Vec16r, ConstructorsAndAllLaneOperations) {
  Vec16r const scalar{3_r};
  Vec16r const values{
      1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r, 10_r, 11_r, 12_r, 13_r, 14_r, 15_r, 16_r};
  std::array<real, Vec16r::kSize> expected{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    expected[i] = static_cast<real>(i + 1);
    EXPECT_EQ(3_r, scalar[i]);
  }
  EXPECT_EQ(values, Load<Vec16r>(expected.data()));
  ExpectCompileTimeBroadcastsAndSets(values, expected);
  for (int lane = 0; lane < Vec16r::kSize; ++lane) {
    auto const broadcast = Broadcast(values, lane);
    for (int i = 0; i < Vec16r::kSize; ++i) {
      EXPECT_EQ(expected[lane], broadcast[i]);
    }
  }
}

TEST(Vec16r, ExhaustiveMasks) {
  std::array<real, Vec16r::kSize> lhsValues{};
  std::array<real, Vec16r::kSize> rhsValues{};
  Vec16r const selectedTrue{17_r};
  Vec16r const selectedFalse{-19_r};
  for (uint32_t bits = 0; bits < (uint32_t{1} << Vec16r::kSize); ++bits) {
    for (int i = 0; i < Vec16r::kSize; ++i) {
      rhsValues[i] = (bits & (uint32_t{1} << i)) != 0 ? 1_r : -1_r;
    }
    auto const mask = Load<Vec16r>(lhsValues.data()) < Load<Vec16r>(rhsValues.data());
    EXPECT_EQ(bits == ((uint32_t{1} << Vec16r::kSize) - 1), AllTrue(mask));
    EXPECT_EQ(bits != 0, AnyTrue(mask));
    auto const selected = Select(mask, selectedTrue, selectedFalse);
    for (int i = 0; i < Vec16r::kSize; ++i) {
      bool const expected = (bits & (uint32_t{1} << i)) != 0;
      ExpectMaskLane(mask, i, expected);
      EXPECT_EQ(expected ? 17_r : -19_r, selected[i]);
    }
  }
}

TEST(Vec16r, SimdMaskIsTrueAndZero) {
  auto const mask = SimdMask<Vec16r>(
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true);
  std::array<bool, Vec16r::kSize> const expected = {
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true};
  ExpectIsTrueLanes(mask, expected);
  EXPECT_EQ(Vec16r{}, SimdZero<Vec16r>());
}

TEST(Vec16r, NearComparisons) {
  std::array<real, Vec16r::kSize> aValues{};
  std::array<real, Vec16r::kSize> bValues{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    aValues[i] = static_cast<real>(i + 1);
    bValues[i] = aValues[i] + (i % 3 == 0 ? real{0.1} : real{0.2});
  }
  auto const a = Load<Vec16r>(aValues.data());
  auto const b = Load<Vec16r>(bValues.data());
  ExpectNearPrefixes(a, b);
  auto const nearMask = VNearEqual(a, b, Vec16r{real{0.125}});
  auto const nearZero = VNearZero(a - Vec16r{real{8}}, Vec16r{real{4.25}});
  for (int i = 0; i < Vec16r::kSize; ++i) {
    ExpectMaskLane(nearMask, i, i % 3 == 0);
    ExpectMaskLane(nearZero, i, i >= 3 && i <= 11);
  }
}

TEST(Vec16r, VectorReductionsAndNormalize) {
  std::array<real, Vec16r::kSize> values{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    values[i] = static_cast<real>(i + 1);
  }
  auto const v = Load<Vec16r>(values.data());
  ExpectNormPrefixes(v);
  ExpectPoisonedTailReductions(v);
  auto const dot = VDot(v, Vec16r{real{1}});
  auto const normalizedFromNormSqr = Normalize(v, VNormSqr(v));
  real expectedSum = real{};
  real expectedNormSqr = real{};
  for (real value : values) {
    expectedSum += value;
    expectedNormSqr += value * value;
  }
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_EQ(expectedSum, dot[i]);
    EXPECT_NEAR(
        values[i] / Sqrt(expectedNormSqr),
        normalizedFromNormSqr[i],
        Abs(values[i] / Sqrt(expectedNormSqr)) * kEps);
  }
}

TEST(Vec16r, ClampLerpSignAndSignedSqrt) {
  std::array<real, Vec16r::kSize> values{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    values[i] = static_cast<real>(i - 8);
  }
  auto const v = Load<Vec16r>(values.data());
  auto const clamped = Clamp(v, Vec16r{-3_r}, Vec16r{3_r});
  auto const lerped = Lerp(v, -v, real{0.25});
  auto const sign = Sign(v);
  auto const signedSqrt = SignedSqrt(v);
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_EQ(Clamp(values[i], -3_r, 3_r), clamped[i]);
    EXPECT_EQ(values[i] * real{0.5}, lerped[i]);
    EXPECT_EQ(values[i] >= real{} ? real{1} : real{-1}, sign[i]);
    EXPECT_NEAR(values[i] >= real{} ? Sqrt(values[i]) : -Sqrt(-values[i]), signedSqrt[i], kEps);
  }
}

TEST(Vec16r, SinCos) {
  std::array<real, Vec16r::kSize> values{};
  for (int i = 0; i < Vec16r::kSize; ++i) {
    values[i] = static_cast<real>(i - 8) / real{8};
  }
  auto const input = Load<Vec16r>(values.data());
  auto const [sin, cos] = SinCos(input);
  for (int i = 0; i < Vec16r::kSize; ++i) {
    EXPECT_NEAR(std::sin(values[i]), sin[i], kEps);
    EXPECT_NEAR(std::cos(values[i]), cos[i], kEps);
  }
}

// clang-format off
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec16r, ACos, std::acos, -1, 1, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec16r, ASin, std::asin, -1, 1, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec16r, ATan, std::atan, -16, 16, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec16r, Cos, std::cos, -1, 1, kEps);
TEST(Vec16r, Exp) { TestExp<Vec16r>(); }
TEST(Vec16r, Ln) { TestLn<Vec16r>(); }
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec16r, Sin, std::sin, -1, 1, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec16r, Tan, std::tan, -1, 1, kEps);
MOCHI_SIMD_TEST_LIMIT_RANGE_UNARY_FN_NEAR(Vec16r, Tanh, std::tanh, -5, 5, kEps);
// clang-format on
