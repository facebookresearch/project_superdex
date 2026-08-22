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

#pragma once

#include "simd_test.h"

#include <mochi_core/utils/half.h>
#include <mochi_core/utils/span.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#if MOCHI_HAS_SIMD_HALF

namespace mochi::simd_half_test {

inline Span<Half const> GetTestValues(int n) {
  static Half const kValues[] = {
      kHalfMin,
      kHalfMax,
      Half(1.5f),
      Half(-2.25f),
      Half(0.5f),
      Half(-0.125f),
      Half(3.75f),
      Half(-4.0f),
      Half(0.0625f),
      Half(-7.5f),
      Half(0.25f),
      Half(-1.0f),
      Half(6.5f),
      Half(-3.5f),
      Half(0.375f),
      Half(-8.0f),
  };
  MOCHI_ASSERT_VERBOSE(n >= 0 && n <= isize(kValues), "n out of range");
  return {kValues, static_cast<size_t>(n)};
}

inline uint16_t GetBits(Half h) {
  return ReinterpretCast<uint16_t>(h);
}

template <class V>
void TestGet(Span<Half const> values) {
  ASSERT_EQ(V::kSize, isize(values));
  auto v = Load<V>(values.data());
  for (int i = 0; i < V::kSize; ++i) {
    EXPECT_EQ(values[i], Get(v, i));
    EXPECT_EQ(values[i], v[i]);
  }
}

template <class V>
void TestLoadStore(Span<Half const> values) {
  ASSERT_EQ(V::kSize, isize(values));
  auto v = Load<V>(values.data());
  Half result[V::kSize] = {};
  Store(result, v);
  for (int i = 0; i < V::kSize; ++i) {
    EXPECT_EQ(values[i], result[i]);
  }
}

template <class V>
void TestLoadPartial(Span<Half const> values) {
  ASSERT_EQ(V::kSize, isize(values));
  for (int n = 0; n <= V::kSize; ++n) {
    auto v = Load<V>(values.data(), n);
    for (int i = 0; i < V::kSize; ++i) {
      EXPECT_EQ(i < n ? values[i] : Half{}, v[i]);
    }
  }
}

template <int N, class V>
void TestLoadPartialN(Span<Half const> values) {
  ASSERT_EQ(V::kSize, isize(values));
  auto v = Load<N, V>(values.data());
  for (int i = 0; i < V::kSize; ++i) {
    EXPECT_EQ(i < N ? values[i] : Half{}, v[i]);
  }
}

template <class V>
void TestStorePartial(Span<Half const> values) {
  ASSERT_EQ(V::kSize, isize(values));
  auto v = Load<V>(values.data());
  for (int n = 0; n <= V::kSize; ++n) {
    Half sentinel = StaticCast<Half>(911.0f);
    Half result[V::kSize] = {};
    for (int i = 0; i < V::kSize; ++i) {
      result[i] = sentinel;
    }
    Store(result, v, n);
    for (int i = 0; i < V::kSize; ++i) {
      EXPECT_EQ(i < n ? values[i] : sentinel, result[i]);
    }
  }
}

template <int N, class V>
void TestStorePartialN(Span<Half const> values) {
  ASSERT_EQ(V::kSize, isize(values));
  auto v = Load<V>(values.data());
  Half result[V::kSize] = {};
  Store<N>(result, v);
  for (int i = 0; i < V::kSize; ++i) {
    EXPECT_EQ(i < N ? values[i] : Half{}, result[i]);
  }
}

template <class V, class SimdOp, class ScalarOp>
void TestBitwiseOp(SimdOp simdOp, ScalarOp scalarOp) {
  static_assert(V::kSize <= 16);

  // clang-format off
  uint16_t const kBitsA[16] = {0xFFFF, 0xFF00, 0x0F0F, 0x1234, 0xAAAA, 0x5555, 0x0000, 0xFFFF, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888};
  uint16_t const kBitsB[16] = {0xFFFF, 0x00FF, 0xF0F0, 0x4321, 0x5555, 0xAAAA, 0xFFFF, 0x0000, 0xEEEE, 0xDDDD, 0xCCCC, 0xBBBB, 0xAAAA, 0x9999, 0x8888, 0x7777};
  // clang-format on

  Half aVals[V::kSize] = {};
  Half bVals[V::kSize] = {};
  for (int i = 0; i < V::kSize; ++i) {
    aVals[i] = ReinterpretCast<Half>(kBitsA[i]);
    bVals[i] = ReinterpretCast<Half>(kBitsB[i]);
  }
  auto r = simdOp(Load<V>(aVals), Load<V>(bVals));
  for (int i = 0; i < V::kSize; ++i) {
    EXPECT_EQ(scalarOp(kBitsA[i], kBitsB[i]), GetBits(r[i]));
  }
}

template <class V>
void TestBroadcastConstructor() {
  auto value = StaticCast<Half>(1.23f);
  V v{value};
  for (int i = 0; i < V::kSize; ++i) {
    EXPECT_EQ(value, v[i]);
  }
}

template <class V>
void TestEquality(Span<Half const> values) {
  auto a = Load<V>(values.data());

  // Identical vectors: operator== true
  auto b = a;
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
  {
    auto eq = V::Equal(a, b);
    auto neq = V::NotEqual(a, b);
    for (int i = 0; i < V::kSize; ++i) {
      EXPECT_EQ(0xFFFF, GetBits(eq[i]));
      EXPECT_EQ(0x0000, GetBits(neq[i]));
    }
  }

  // Per-lane mismatch: flip one lane at a time and verify all four operations detect it.
  auto const sentinel = StaticCast<Half>(911.0f);
  for (int lane = 0; lane < V::kSize; ++lane) {
    EXPECT_NE(sentinel, values[lane]);
    Half modified[V::kSize] = {};
    for (int i = 0; i < V::kSize; ++i) {
      modified[i] = (i == lane) ? sentinel : values[i];
    }
    auto m = Load<V>(modified);
    EXPECT_FALSE(a == m);
    EXPECT_TRUE(a != m);
    auto eq = V::Equal(a, m);
    auto neq = V::NotEqual(a, m);
    for (int i = 0; i < V::kSize; ++i) {
      EXPECT_EQ(i == lane ? 0x0000 : 0xFFFF, GetBits(eq[i]));
      EXPECT_EQ(i == lane ? 0xFFFF : 0x0000, GetBits(neq[i]));
    }
  }

  // IEEE 754: +0 == -0
  auto posZero = StaticCast<V>(Simd<float, V::kSize>{0.0f});
  auto negZero = StaticCast<V>(Simd<float, V::kSize>{-0.0f});
  EXPECT_TRUE(posZero == negZero);
  {
    auto eq = V::Equal(posZero, negZero);
    auto neq = V::NotEqual(posZero, negZero);
    for (int i = 0; i < V::kSize; ++i) {
      EXPECT_EQ(0xFFFF, GetBits(eq[i]));
      EXPECT_EQ(0x0000, GetBits(neq[i]));
    }
  }

  // IEEE 754: NaN != NaN
  auto nans = StaticCast<V>(Simd<float, V::kSize>{std::numeric_limits<float>::quiet_NaN()});
  EXPECT_FALSE(nans == nans);
  EXPECT_TRUE(nans != nans);
  {
    auto eq = V::Equal(nans, nans);
    auto neq = V::NotEqual(nans, nans);
    for (int i = 0; i < V::kSize; ++i) {
      EXPECT_EQ(0x0000, GetBits(eq[i]));
      EXPECT_EQ(0xFFFF, GetBits(neq[i]));
    }
  }
}

template <int N, class V>
void ExpectAllTrueForN(V const& v, int zeroLane) {
  if (N <= zeroLane) {
    EXPECT_TRUE((V::template AllTrue<N>(v)));
  } else {
    EXPECT_FALSE((V::template AllTrue<N>(v)));
  }
  if constexpr (N < V::kSize) {
    ExpectAllTrueForN<N + 1, V>(v, zeroLane);
  }
}

template <class V>
void TestAllTrue() {
  Half allTrue[V::kSize] = {};
  for (int i = 0; i < V::kSize; ++i) {
    allTrue[i] = ReinterpretCast<Half>(uint16_t{0xFFFF});
  }
  auto v = Load<V>(allTrue);
  EXPECT_TRUE(V::AllTrue(v));

  // Single false lane at each position.
  for (int i = 0; i < V::kSize; ++i) {
    Half buf[V::kSize] = {};
    for (int j = 0; j < V::kSize; ++j) {
      buf[j] = allTrue[j];
    }
    buf[i] = ReinterpretCast<Half>(uint16_t{0x0000});
    EXPECT_FALSE(V::AllTrue(Load<V>(buf)));
  }

  // AllTrue<N> for every N: all lanes true → all pass.
  ExpectAllTrueForN<1, V>(v, V::kSize);

  // Zero each lane k: AllTrue<N> passes for N <= k, fails for N > k.
  for (int k = 0; k < V::kSize; ++k) {
    Half buf[V::kSize] = {};
    for (int j = 0; j < V::kSize; ++j) {
      buf[j] = allTrue[j];
    }
    buf[k] = ReinterpretCast<Half>(uint16_t{0x0000});
    ExpectAllTrueForN<1, V>(Load<V>(buf), k);
  }
}

template <class V>
void TestStaticCastSpecialValues() {
  static_assert(V::kSize <= 16);

  // clang-format off
  float const kSpecials[16] = {
      0.0f, -0.0f,
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
      65504.0f, 5.96046e-8f, 6.103515625e-5f,
      std::numeric_limits<float>::quiet_NaN(),
      1.0f, -1.0f, 0.5f, -0.5f,
      65504.0f, 6.103515625e-5f, 0.0f,
      std::numeric_limits<float>::quiet_NaN(),
  };
  // clang-format on

  auto f = Load<Simd<float, V::kSize>>(kSpecials);
  auto h = StaticCast<V>(f);
  auto back = StaticCast<Simd<float, V::kSize>>(h);

  // Lanes 0–7: standard IEEE 754 special values.
  EXPECT_EQ(0.0f, back[0]);
  EXPECT_TRUE(std::signbit(back[1]));
  EXPECT_EQ(0.0f, back[1]);
  EXPECT_EQ(std::numeric_limits<float>::infinity(), back[2]);
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), back[3]);
  EXPECT_EQ(65504.0f, back[4]);
  EXPECT_GE(back[5], 0.0f);
  EXPECT_NEAR(6.103515625e-5f, back[6], 1e-8f);
  EXPECT_TRUE(std::isnan(back[7]));

  // Remaining lanes: NaN stays NaN, everything else round-trips.
  for (int i = 8; i < V::kSize; ++i) {
    if (std::isnan(kSpecials[i])) {
      EXPECT_TRUE(std::isnan(back[i]));
    } else {
      EXPECT_NEAR(kSpecials[i], back[i], 1e-8f);
    }
  }
}

template <class V>
void TestStaticCastRoundTrip(Span<Half const> values) {
  ASSERT_EQ(V::kSize, isize(values));
  auto h = Load<V>(values.data());
  auto r = StaticCast<Simd<real, V::kSize>>(h);
  auto back = StaticCast<V>(r);
  for (int i = 0; i < V::kSize; ++i) {
    EXPECT_EQ(values[i], back[i]); // lossless
  }
}

template <class V, int I = 0>
void TestGetNImpl(V v, Half const* values) {
  if constexpr (I < V::kSize) {
    EXPECT_EQ(values[I], Get<I>(v));
    TestGetNImpl<V, I + 1>(v, values);
  }
}

template <class V>
void TestGetN(Half const* values) {
  auto v = Load<V>(values);
  TestGetNImpl<V>(v, values);
}

template <class V>
void TestGet0(Half const* values) {
  auto v = Load<V>(values);
  EXPECT_EQ(values[0], Get0(v));
}

template <class V, size_t... Is>
void TestAllLoadPartialNImpl(Span<Half const> values, std::index_sequence<Is...>) {
  (TestLoadPartialN<static_cast<int>(Is), V>(values), ...);
}

template <class V>
void TestAllLoadPartialN(Span<Half const> values) {
  TestAllLoadPartialNImpl<V>(values, std::make_index_sequence<V::kSize + 1>{});
}

template <class V, size_t... Is>
void TestAllStorePartialNImpl(Span<Half const> values, std::index_sequence<Is...>) {
  (TestStorePartialN<static_cast<int>(Is), V>(values), ...);
}

template <class V>
void TestAllStorePartialN(Span<Half const> values) {
  TestAllStorePartialNImpl<V>(values, std::make_index_sequence<V::kSize + 1>{});
}

template <int N, class V>
void ExpectAnyTrueForN(V const& v, int trueLane) {
  if (trueLane < N) {
    EXPECT_TRUE((V::template AnyTrue<N>(v)));
  } else {
    EXPECT_FALSE((V::template AnyTrue<N>(v)));
  }
  if constexpr (N < V::kSize) {
    ExpectAnyTrueForN<N + 1, V>(v, trueLane);
  }
}

template <class V>
void TestAnyTrue() {
  // All false
  Half allFalseBuf[V::kSize] = {};
  for (int i = 0; i < V::kSize; ++i) {
    allFalseBuf[i] = ReinterpretCast<Half>(uint16_t{0x0000});
  }
  auto vFalse = Load<V>(allFalseBuf);
  EXPECT_FALSE(V::AnyTrue(vFalse));

  // All true
  Half allTrueBuf[V::kSize] = {};
  for (int i = 0; i < V::kSize; ++i) {
    allTrueBuf[i] = ReinterpretCast<Half>(uint16_t{0xFFFF});
  }
  auto vTrue = Load<V>(allTrueBuf);
  EXPECT_TRUE(V::AnyTrue(vTrue));

  // Single true lane at each position
  for (int i = 0; i < V::kSize; ++i) {
    Half buf[V::kSize] = {};
    for (int j = 0; j < V::kSize; ++j) {
      buf[j] = allFalseBuf[j];
    }
    buf[i] = ReinterpretCast<Half>(uint16_t{0xFFFF});
    EXPECT_TRUE(V::AnyTrue(Load<V>(buf)));
  }

  // AnyTrue<N>: all false → all should be false
  ExpectAnyTrueForN<1, V>(vFalse, V::kSize);

  // Single true lane at position k: AnyTrue<N> is true iff N > k
  for (int k = 0; k < V::kSize; ++k) {
    Half buf[V::kSize] = {};
    for (int j = 0; j < V::kSize; ++j) {
      buf[j] = allFalseBuf[j];
    }
    buf[k] = ReinterpretCast<Half>(uint16_t{0xFFFF});
    ExpectAnyTrueForN<1, V>(Load<V>(buf), k);
  }
}

// Compare StaticCast (Half --> float) using Simd vs single scalars.
template <class V>
void TestExhaustiveConversionHalfToFloat() {
  constexpr auto kMax = std::numeric_limits<uint16_t>::max();
  for (uint32_t base = 0; base <= kMax; base += V::kSize) {
    Half halves[V::kSize] = {};
    for (int i = 0; i < V::kSize && (base + i) < kMax; ++i) {
      halves[i] = ReinterpretCast<Half>(static_cast<uint16_t>(base + i));
    }
    auto v = Load<V>(halves);
    auto floats = StaticCast<Simd<float, V::kSize>>(v);
    float fout[V::kSize] = {};
    Store(fout, floats);
    for (int i = 0; i < V::kSize && (base + i) < kMax; ++i) {
      auto expected = StaticCast<float>(halves[i]);
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(fout[i]));
      } else {
        EXPECT_EQ(expected, fout[i]);
      }
    }
  }
}

// Compare StaticCast (float --> Half) using Simd vs single scalars.
// Tests all 2^32 float bit patterns to verify SIMD matches scalar conversion.
template <class V>
void TestExhaustiveConversionFloatToHalf() {
  constexpr uint64_t kNumFloats = uint64_t{1} << 32;
  for (uint64_t base = 0; base < kNumFloats; base += V::kSize) {
    float floats[V::kSize] = {};
    for (int i = 0; i < V::kSize; ++i) {
      floats[i] = ReinterpretCast<float>(static_cast<uint32_t>(base + i));
    }
    auto vf = Load<Simd<float, V::kSize>>(floats);
    auto vh = StaticCast<V>(vf);
    Half hout[V::kSize] = {};
    Store(hout, vh);
    for (int i = 0; i < V::kSize; ++i) {
      auto expected = StaticCast<Half>(floats[i]);
      if (std::isnan(floats[i])) {
        if (!std::isnan(static_cast<float>(hout[i]))) {
          EXPECT_TRUE(std::isnan(static_cast<float>(hout[i])));
        }
      } else if (GetBits(expected) != GetBits(hout[i])) {
        EXPECT_EQ(GetBits(expected), GetBits(hout[i]));
      }
    }
  }
}

template <class V>
void TestReinterpretCastRoundTrip(Span<Half const> values) {
  ASSERT_EQ(V::kSize, isize(values));
  auto h = Load<V>(values.data());
  EXPECT_TRUE(h == ReinterpretCast<V>(ReinterpretCast<Simd<float, V::kSize / 2>>(h)));
  EXPECT_TRUE(h == ReinterpretCast<V>(ReinterpretCast<Simd<int, V::kSize / 2>>(h)));
  EXPECT_TRUE(h == ReinterpretCast<V>(ReinterpretCast<Simd<int64_t, V::kSize / 4>>(h)));
  EXPECT_TRUE(h == ReinterpretCast<V>(ReinterpretCast<Simd<double, V::kSize / 4>>(h)));
}

} // namespace mochi::simd_half_test

/***********************************************************************************************
  Macro to generate common Simd<Half, N> tests for any vector width.
  Each .cpp file invokes this once with its vector type (e.g., Vec8h, Vec16h),
  then adds any type-specific tests below the invocation.
*/

#define MOCHI_DEFINE_SIMD_HALF_COMMON_TESTS(V)                                                     \
  TEST(V, Get) {                                                                                   \
    TestGet<V>(GetTestValues(V::kSize));                                                           \
  }                                                                                                \
  TEST(V, Get0) {                                                                                  \
    TestGet0<V>(GetTestValues(V::kSize).data());                                                   \
  }                                                                                                \
  TEST(V, GetN) {                                                                                  \
    TestGetN<V>(GetTestValues(V::kSize).data());                                                   \
  }                                                                                                \
  TEST(V, Zero) {                                                                                  \
    EXPECT_EQ(V{}, V::Zero());                                                                     \
  }                                                                                                \
  TEST(V, CopyConstruction) {                                                                      \
    auto a = Load<V>(GetTestValues(V::kSize).data());                                              \
    V b{a};                                                                                        \
    EXPECT_TRUE(a == b);                                                                           \
  }                                                                                                \
  TEST(V, LoadStore) {                                                                             \
    TestLoadStore<V>(GetTestValues(V::kSize));                                                     \
  }                                                                                                \
  TEST(V, LoadPartial) {                                                                           \
    TestLoadPartial<V>(GetTestValues(V::kSize));                                                   \
  }                                                                                                \
  TEST(V, StorePartial) {                                                                          \
    TestStorePartial<V>(GetTestValues(V::kSize));                                                  \
  }                                                                                                \
  TEST(V, LoadPartialCompileTime) {                                                                \
    TestAllLoadPartialN<V>(GetTestValues(V::kSize));                                               \
  }                                                                                                \
  TEST(V, StorePartialCompileTime) {                                                               \
    TestAllStorePartialN<V>(GetTestValues(V::kSize));                                              \
  }                                                                                                \
  TEST(V, AND) {                                                                                   \
    TestBitwiseOp<V>(                                                                              \
        [](V a, V b) { return a & b; }, [](uint16_t a, uint16_t b) -> uint16_t { return a & b; }); \
  }                                                                                                \
  TEST(V, OR) {                                                                                    \
    TestBitwiseOp<V>(                                                                              \
        [](V a, V b) { return a | b; }, [](uint16_t a, uint16_t b) -> uint16_t { return a | b; }); \
  }                                                                                                \
  TEST(V, XOR) {                                                                                   \
    TestBitwiseOp<V>(                                                                              \
        [](V a, V b) { return a ^ b; }, [](uint16_t a, uint16_t b) -> uint16_t { return a ^ b; }); \
  }                                                                                                \
  TEST(V, NOT) {                                                                                   \
    TestBitwiseOp<V>(                                                                              \
        [](V a, V) { return ~a; }, [](uint16_t a, uint16_t) -> uint16_t { return uint16_t(~a); }); \
  }                                                                                                \
  TEST(V, BroadcastConstructor) {                                                                  \
    TestBroadcastConstructor<V>();                                                                 \
  }                                                                                                \
  TEST(V, EqualityAndComparison) {                                                                 \
    TestEquality<V>(GetTestValues(V::kSize));                                                      \
  }                                                                                                \
  TEST(V, AllTrue) {                                                                               \
    TestAllTrue<V>();                                                                              \
  }                                                                                                \
  TEST(V, AnyTrue) {                                                                               \
    TestAnyTrue<V>();                                                                              \
  }                                                                                                \
  TEST(V, StaticCastRoundTrip) {                                                                   \
    TestStaticCastRoundTrip<V>(GetTestValues(V::kSize));                                           \
  }                                                                                                \
  TEST(V, StaticCastSpecialValues) {                                                               \
    TestStaticCastSpecialValues<V>();                                                              \
  }                                                                                                \
  TEST(V, ReinterpretCastRoundTrip) {                                                              \
    TestReinterpretCastRoundTrip<V>(GetTestValues(V::kSize));                                      \
  }                                                                                                \
  TEST_IF(MOCHI_OPTIMIZED, V, ExhaustiveConversionHalfToFloat) {                                   \
    TestExhaustiveConversionHalfToFloat<V>();                                                      \
  }                                                                                                \
  TEST_IF(MOCHI_OPTIMIZED, V, ExhaustiveConversionFloatToHalf) {                                   \
    TestExhaustiveConversionFloatToHalf<V>();                                                      \
  }

#endif // MOCHI_HAS_SIMD_HALF
