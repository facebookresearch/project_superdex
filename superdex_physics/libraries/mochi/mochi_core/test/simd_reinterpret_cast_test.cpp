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

#include <mochi_core/utils/half.h>

#include <iterator>
#include <limits>

using namespace mochi;
using namespace mochi::simd_test;

template <typename To, typename From, int FromN>
static void TestSimdReinterpretCastTo(Simd<From, FromN> const& from) {
  constexpr int ToN = FromN * sizeof(From) / sizeof(To);
  static_assert(sizeof(Simd<To, ToN>) == sizeof(Simd<From, FromN>));
  if constexpr (Simd<To, ToN>::kIsSupported) {
    // Expect bitwise equality
    auto result = ReinterpretCast<Simd<To, ToN>>(from);
    EXPECT_EQ(0, memcmp(&from, &result, sizeof(From) * FromN));
    auto roundTrip = ReinterpretCast<Simd<From, FromN>>(result);
    EXPECT_EQ(0, memcmp(&from, &roundTrip, sizeof(From) * FromN));
    EXPECT_EQ(from, roundTrip);
  }
}

template <typename From, int FromN>
static void TestSimdReinterpretCastToEach(Simd<From, FromN> const& from) {
  // Cast it to every other Simd type of the same size in bytes
  TestSimdReinterpretCastTo<double>(from);
  TestSimdReinterpretCastTo<float>(from);
  TestSimdReinterpretCastTo<int>(from);
  TestSimdReinterpretCastTo<int64_t>(from);
}

template <typename From, int N>
static void TestSimdReinterpretCastFrom() {
  static_assert(Simd<From, N>::kIsSupported);

  // Invent some values
  From kTestValues[N] = {};
  From kNegTestValues[N] = {};
  for (int i = 0; i < N; ++i) {
    kTestValues[i] = static_cast<From>(i + 1);
    kNegTestValues[i] = static_cast<From>(-i - 1);
  }

  auto from = Load<Simd<From, N>>(kTestValues);
  auto negFrom = Load<Simd<From, N>>(kNegTestValues);
  TestSimdReinterpretCastToEach(from); // Positive values
  TestSimdReinterpretCastToEach(negFrom); // Negative values

  if constexpr (IsHalf<From>) {
    from = Simd<From, N>{kHalfMin, kHalfMax};
  } else {
    from = Simd<From, N>{std::numeric_limits<From>::min(), std::numeric_limits<From>::max()};
  }
  TestSimdReinterpretCastToEach(from);
}

TEST(Vec2d, ReinterpretCast) {
  TestSimdReinterpretCastFrom<double, 2>();
}

TEST(Vec2l, ReinterpretCast) {
  TestSimdReinterpretCastFrom<int64_t, 2>();
}

TEST(Vec4d, ReinterpretCast) {
  TestSimdReinterpretCastFrom<double, 4>();
}

TEST(Vec4f, ReinterpretCast) {
  TestSimdReinterpretCastFrom<float, 4>();
}

TEST(Vec4i, ReinterpretCast) {
  TestSimdReinterpretCastFrom<int, 4>();
}

TEST(Vec4l, ReinterpretCast) {
  TestSimdReinterpretCastFrom<int64_t, 4>();
}

TEST(Vec8d, ReinterpretCast) {
  TestSimdReinterpretCastFrom<double, 8>();
}

TEST(Vec8f, ReinterpretCast) {
  TestSimdReinterpretCastFrom<float, 8>();
}

TEST(Vec8i, ReinterpretCast) {
  TestSimdReinterpretCastFrom<int, 8>();
}

TEST(Vec8l, ReinterpretCast) {
  TestSimdReinterpretCastFrom<int64_t, 8>();
}

TEST(Vec12d, ReinterpretCast) {
  TestSimdReinterpretCastFrom<double, 12>();
}

TEST(Vec12f, ReinterpretCast) {
  TestSimdReinterpretCastFrom<float, 12>();
}

TEST(Vec12i, ReinterpretCast) {
  TestSimdReinterpretCastFrom<int, 12>();
}

TEST(Vec12l, ReinterpretCast) {
  TestSimdReinterpretCastFrom<int64_t, 12>();
}

TEST(Vec16d, ReinterpretCast) {
  TestSimdReinterpretCastFrom<double, 16>();
}

TEST(Vec16f, ReinterpretCast) {
  TestSimdReinterpretCastFrom<float, 16>();
}

TEST(Vec16i, ReinterpretCast) {
  TestSimdReinterpretCastFrom<int, 16>();
}

TEST(Vec16l, ReinterpretCast) {
  TestSimdReinterpretCastFrom<int64_t, 16>();
}

#if MOCHI_HAS_SIMD_HALF

TEST(Vec8h, ReinterpretCast) {
  TestSimdReinterpretCastFrom<Half, 8>();
}

TEST(Vec16h, ReinterpretCast) {
  TestSimdReinterpretCastFrom<Half, 16>();
}

#endif // MOCHI_HAS_SIMD_HALF
