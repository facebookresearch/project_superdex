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

template <typename To, typename From, int N>
static void TestSimdStaticCastTo(Simd<From, N> const& from) {
  if constexpr (Simd<To, N>::kIsSupported) {
    // Expect the same results as scalar static_cast
    auto result = StaticCast<Simd<To, N>>(from);
    for (int i = 0; i < N; ++i) {
      EXPECT_EQ(static_cast<To>(from[i]), result[i]);
    }
  }
}

template <typename From, int N>
static void TestSimdStaticCastToEach(Simd<From, N> const& from) {
  // Cast it to every other Simd type of the same size N
  TestSimdStaticCastTo<double>(from);
  TestSimdStaticCastTo<float>(from);
  TestSimdStaticCastTo<int>(from);
  TestSimdStaticCastTo<int64_t>(from);
  if constexpr (Simd<Half, N>::kIsSupported) {
    TestSimdStaticCastTo<Half>(from);
  }
}

// Like TestSimdStaticCastToEach, but skips floating-point to integer conversions where
// numeric_limits<From>::max() exceeds the target integer range (undefined behavior in C++).
template <typename From, int N>
static void TestSimdStaticCastToEachLimits(Simd<From, N> const& from) {
  TestSimdStaticCastTo<double>(from);
  TestSimdStaticCastTo<float>(from);
  if constexpr (!std::is_floating_point_v<From>) {
    TestSimdStaticCastTo<int>(from);
    TestSimdStaticCastTo<int64_t>(from);
  }
}

template <typename From, int N>
static void TestSimdStaticCastFrom() {
  // Invent some values
  From kTestValues[16] = {};
  static_assert(N <= std::size(kTestValues));
  for (int i = 0; i < N; ++i) {
    kTestValues[i] = static_cast<From>(i + 1);
  }

  auto from = Load<Simd<From, N>>(kTestValues);
  TestSimdStaticCastToEach(from);

  // Repeat with numeric limits, skipping conversions that would be undefined behavior
  if constexpr (IsHalf<From>) {
    from = Simd<From, N>{kHalfMin, kHalfMax};
  } else {
    from = Simd<From, N>{std::numeric_limits<From>::min(), std::numeric_limits<From>::max()};
  }
  TestSimdStaticCastToEachLimits(from);
}

TEST(Vec2d, StaticCast) {
  TestSimdStaticCastFrom<double, 2>();
}

TEST(Vec2l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 2>();
}

TEST(Vec4d, StaticCast) {
  TestSimdStaticCastFrom<double, 4>();
}

TEST(Vec4f, StaticCast) {
  TestSimdStaticCastFrom<float, 4>();
}

TEST(Vec4i, StaticCast) {
  TestSimdStaticCastFrom<int, 4>();
}

TEST(Vec4l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 4>();
}

TEST(Vec8d, StaticCast) {
  TestSimdStaticCastFrom<double, 8>();
}

TEST(Vec8f, StaticCast) {
  TestSimdStaticCastFrom<float, 8>();
}

TEST(Vec8i, StaticCast) {
  TestSimdStaticCastFrom<int, 8>();
}

TEST(Vec8l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 8>();
}

TEST(Vec12d, StaticCast) {
  TestSimdStaticCastFrom<double, 12>();
}

TEST(Vec12f, StaticCast) {
  TestSimdStaticCastFrom<float, 12>();
}

TEST(Vec12i, StaticCast) {
  TestSimdStaticCastFrom<int, 12>();
}

TEST(Vec12l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 12>();
}

TEST(Vec16d, StaticCast) {
  TestSimdStaticCastFrom<double, 16>();
}

TEST(Vec16f, StaticCast) {
  TestSimdStaticCastFrom<float, 16>();
}

TEST(Vec16i, StaticCast) {
  TestSimdStaticCastFrom<int, 16>();
}

TEST(Vec16l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 16>();
}

#if MOCHI_HAS_SIMD_HALF
TEST(Vec8h, StaticCast) {
  TestSimdStaticCastFrom<Half, 8>();
}
TEST(Vec16h, StaticCast) {
  TestSimdStaticCastFrom<Half, 16>();
}
#endif // MOCHI_HAS_SIMD_HALF
