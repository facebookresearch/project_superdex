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

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

using namespace mochi;

// IsSimdSupportedType for supported types
static_assert(IsSimdSupportedType<float>);
static_assert(Simd<float>::kIsSupported);
static_assert(IsSimdSupportedType<double>);
static_assert(Simd<double>::kIsSupported);
#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
static_assert(kSimdDefaultSize<float> == 16);
static_assert(kSimdDefaultSize<double> == 8);
static_assert(std::is_same_v<Simd<double>, Vec8d>);
static_assert(!Vec8d::kIsComposite);
#endif
static_assert(IsSimdSupportedType<int>);
static_assert(Simd<int>::kIsSupported);
static_assert(IsSimdSupportedType<int64_t>);
static_assert(Simd<int64_t>::kIsSupported);

// IsSimdSupportedType for unsupported types
static_assert(!IsSimdSupportedType<float const>);
static_assert(!IsSimdSupportedType<double const>);
static_assert(!IsSimdSupportedType<int const>);
static_assert(!IsSimdSupportedType<int64_t const>);
static_assert(!IsSimdSupportedType<bool>);
static_assert(!Simd<bool>::kIsSupported);
static_assert(!IsSimdSupportedType<char>);
static_assert(!Simd<char>::kIsSupported);
static_assert(!IsSimdSupportedType<size_t>);
static_assert(!Simd<size_t>::kIsSupported);
static_assert(!IsSimdSupportedType<signed char>);
static_assert(!Simd<signed char>::kIsSupported);
static_assert(!IsSimdSupportedType<unsigned char>);
static_assert(!Simd<unsigned char>::kIsSupported);
static_assert(!IsSimdSupportedType<unsigned short>);
static_assert(!Simd<unsigned short>::kIsSupported);
static_assert(!IsSimdSupportedType<unsigned int>);
static_assert(!Simd<unsigned int>::kIsSupported);
static_assert(!IsSimdSupportedType<unsigned long long>);
static_assert(!Simd<unsigned long long>::kIsSupported);
static_assert(!IsSimdSupportedType<long double>);
static_assert(!Simd<long double>::kIsSupported);

static_assert(details::kNextSupportedSimdSize<float, 1> == 4);
static_assert(details::kNextSupportedSimdSize<float, 2> == 4);
static_assert(details::kNextSupportedSimdSize<float, 3> == 4);
static_assert(details::kNextSupportedSimdSize<float, 4> == 4);
static_assert(details::kNextSupportedSimdSize<float, 5> == 8);
static_assert(details::kNextSupportedSimdSize<float, 8> == 8);
static_assert(details::kNextSupportedSimdSize<float, 13> == 16);

static_assert(details::kNextSupportedSimdSize<double, 1> == 2);
static_assert(details::kNextSupportedSimdSize<double, 2> == 2);
static_assert(details::kNextSupportedSimdSize<double, 3> == 4);
static_assert(details::kNextSupportedSimdSize<double, 4> == 4);

static_assert(details::kNextSupportedSimdSize<int, 1> == 4);
static_assert(details::kNextSupportedSimdSize<int, 2> == 4);
static_assert(details::kNextSupportedSimdSize<int, 3> == 4);
static_assert(details::kNextSupportedSimdSize<int, 4> == 4);
static_assert(details::kNextSupportedSimdSize<int, 5> == 8);
static_assert(details::kNextSupportedSimdSize<int, 8> == 8);
static_assert(details::kNextSupportedSimdSize<int, 13> == 16);

static_assert(details::kNextSupportedSimdSize<int64_t, 1> == 2);
static_assert(details::kNextSupportedSimdSize<int64_t, 2> == 2);
static_assert(details::kNextSupportedSimdSize<int64_t, 3> == 4);
static_assert(details::kNextSupportedSimdSize<int64_t, 4> == 4);

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
static_assert(details::kNextSupportedSimdSize<float, 9> == 16);
static_assert(details::kNextSupportedSimdSize<float, 12> == 16);
static_assert(details::kNextSupportedSimdSize<float, 16> == 16);
static_assert(details::kNextSupportedSimdSize<float, 17> == 20);
static_assert(details::kNextSupportedSimdSize<float, 21> == 24);
static_assert(details::kNextSupportedSimdSize<float, 25> == 32);
static_assert(details::kNextSupportedSimdSize<float, 28> == 32);
static_assert(details::kNextSupportedSimdSize<double, 5> == 8);
static_assert(details::kNextSupportedSimdSize<double, 6> == 8);
static_assert(details::kNextSupportedSimdSize<int, 9> == 16);
static_assert(details::kNextSupportedSimdSize<int, 12> == 16);
static_assert(details::kNextSupportedSimdSize<int64_t, 5> == 8);
static_assert(details::kNextSupportedSimdSize<int64_t, 6> == 8);
#else
static_assert(details::kNextSupportedSimdSize<float, 9> == 12);
static_assert(details::kNextSupportedSimdSize<float, 12> == 12);
static_assert(details::kNextSupportedSimdSize<float, 28> == 28);
static_assert(details::kNextSupportedSimdSize<double, 5> == 6);
static_assert(details::kNextSupportedSimdSize<double, 6> == 6);
static_assert(details::kNextSupportedSimdSize<int, 9> == 12);
static_assert(details::kNextSupportedSimdSize<int, 12> == 12);
static_assert(details::kNextSupportedSimdSize<int64_t, 5> == 6);
static_assert(details::kNextSupportedSimdSize<int64_t, 6> == 6);
#endif
static_assert(Simd<float, 12>::kIsSupported);
static_assert(Simd<int, 12>::kIsSupported);
static_assert(Simd<double, 6>::kIsSupported);
static_assert(Simd<int64_t, 6>::kIsSupported);
static_assert(Simd<float, 28>::kIsSupported);
static_assert(Simd<int, 28>::kIsSupported);

static_assert(details::kHasOwnHSumEach<Vec8f> == (MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2));
static_assert(details::kHasOwnHSumEach<Vec8d> == (MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512));

#if MOCHI_USE_SIMD
static_assert(Simd<float, 12>::kIsComposite);
static_assert(Simd<int, 12>::kIsComposite);
static_assert(Simd<double, 6>::kIsComposite);
static_assert(Simd<int64_t, 6>::kIsComposite);
static_assert(Simd<float, 28>::kIsComposite);
static_assert(Simd<int, 28>::kIsComposite);
#if MOCHI_ARCH_X64_AVX512
static_assert(Simd<float, 12>::kSizeFirst == 8 && Simd<float, 12>::kSizeSecond == 4);
static_assert(Simd<int, 12>::kSizeFirst == 8 && Simd<int, 12>::kSizeSecond == 4);
static_assert(Simd<double, 6>::kSizeFirst == 4 && Simd<double, 6>::kSizeSecond == 2);
static_assert(Simd<int64_t, 6>::kSizeFirst == 4 && Simd<int64_t, 6>::kSizeSecond == 2);
static_assert(Simd<float, 28>::kSizeFirst == 16 && Simd<float, 28>::kSizeSecond == 12);
static_assert(Simd<float, 28>::Second::kSizeFirst == 8);
static_assert(Simd<float, 28>::Second::kSizeSecond == 4);
static_assert(Simd<int, 28>::kSizeFirst == 16 && Simd<int, 28>::kSizeSecond == 12);
static_assert(Simd<int, 28>::Second::kSizeFirst == 8);
static_assert(Simd<int, 28>::Second::kSizeSecond == 4);
#endif // MOCHI_ARCH_X64_AVX512
#endif // MOCHI_USE_SIMD

static_assert(std::is_same_v<ScalarType<Simd<float>>, float>);
static_assert(std::is_same_v<ScalarType<Simd<float, 8>>, float>);
static_assert(std::is_same_v<ScalarType<Simd<double>>, double>);
static_assert(std::is_same_v<ScalarType<Simd<double, 4>>, double>);
static_assert(std::is_same_v<ScalarType<Simd<int>>, int>);
static_assert(std::is_same_v<ScalarType<Simd<int, 4>>, int>);
static_assert(std::is_same_v<ScalarType<Simd<int64_t>>, int64_t>);
static_assert(std::is_same_v<ScalarType<Simd<int64_t, 2>>, int64_t>);
#if MOCHI_HAS_SIMD_HALF
static_assert(std::is_same_v<ScalarType<Simd<Half>>, Half>);
static_assert(std::is_same_v<ScalarType<Simd<Half, 16>>, Half>);
#endif // MOCHI_HAS_SIMD_HALF

static bool EqualOrBothNaN(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) || (a == b);
}

static bool NearOrBothNaN(float a, float b) {
  constexpr float kMaxSinCosError = 6.0e-8f; // Match the public Sin/Cos accuracy guarantee.
  return EqualOrBothNaN(a, b) ||
      (std::isfinite(a) && std::isfinite(b) && NearEqual(a, b, kMaxSinCosError));
}

template <typename V, size_t kNumValues>
static void ExpectFloatSinCosMatchesScalar(float const (&testValues)[kNumValues]) {
  static_assert(std::is_same_v<typename V::Scalar, float>);
  static_assert(kNumValues % V::kSize == 0);

  for (size_t offset = 0; offset < kNumValues; offset += V::kSize) {
    V const input = Load<V>(&testValues[offset]);
    V const sin = Sin(input);
    V const cos = Cos(input);
    auto const [sinPair, cosPair] = SinCos(input);

    alignas(alignof(V)) float sinOut[V::kSize];
    alignas(alignof(V)) float cosOut[V::kSize];
    alignas(alignof(V)) float sinPairOut[V::kSize];
    alignas(alignof(V)) float cosPairOut[V::kSize];
    Store(sinOut, sin);
    Store(cosOut, cos);
    Store(sinPairOut, sinPair);
    Store(cosPairOut, cosPair);

    for (int i = 0; i < V::kSize; ++i) {
      float const expectedSin = std::sin(testValues[offset + i]);
      float const expectedCos = std::cos(testValues[offset + i]);
      EXPECT_TRUE(NearOrBothNaN(expectedSin, sinOut[i]));
      EXPECT_TRUE(NearOrBothNaN(expectedCos, cosOut[i]));
      EXPECT_TRUE(NearOrBothNaN(expectedSin, sinPairOut[i]));
      EXPECT_TRUE(NearOrBothNaN(expectedCos, cosPairOut[i]));
    }
  }
}

// This test targets the custom range-reduction cutoff; SVML bypasses it.
#if !MOCHI_ARCH_X64_SVML
#define MOCHI_TEST_USES_CUSTOM_FLOAT_SIN_COS_BACKEND 1
#else
#define MOCHI_TEST_USES_CUSTOM_FLOAT_SIN_COS_BACKEND 0
#endif

TEST_IF(
    MOCHI_TEST_USES_CUSTOM_FLOAT_SIN_COS_BACKEND,
    Simd,
    FloatSinCosMatchesScalarNearFastPathLimit) {
  float constexpr kMaxFastInput = details::kMaxFastSinCosInput;
  float const belowLimit = std::nextafter(kMaxFastInput, 0.0f);
  float const aboveLimit = std::nextafter(kMaxFastInput, std::numeric_limits<float>::infinity());
  float const inRangeValues[] = {
      belowLimit, -belowLimit, kMaxFastInput, -kMaxFastInput, 1.0f, -1.0f, 123.0f, -123.0f};
  float const outOfRangeValues[] = {
      aboveLimit,
      -aboveLimit,
      1.0e20f,
      -1.0e20f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max()};
  float const mixedValues[] = {
      belowLimit,
      aboveLimit,
      -belowLimit,
      -aboveLimit,
      kMaxFastInput,
      1.0e20f,
      -kMaxFastInput,
      -1.0e20f};

  ExpectFloatSinCosMatchesScalar<Simd<float, 4>>(inRangeValues);
  ExpectFloatSinCosMatchesScalar<Simd<float, 4>>(outOfRangeValues);
  ExpectFloatSinCosMatchesScalar<Simd<float, 4>>(mixedValues);
  ExpectFloatSinCosMatchesScalar<Simd<float, 8>>(inRangeValues);
  ExpectFloatSinCosMatchesScalar<Simd<float, 8>>(outOfRangeValues);
  ExpectFloatSinCosMatchesScalar<Simd<float, 8>>(mixedValues);
}

#undef MOCHI_TEST_USES_CUSTOM_FLOAT_SIN_COS_BACKEND

// Expects LoadTransposed and StoreTransposed with a runtime count to read and write exactly
// (count * 3) scalars, with zeros in the unused lanes.
template <class T, int N>
static void ExpectTransposedIOWithRuntimeCount() {
  using V = Simd<T, N>;
  T constexpr kSentinel = static_cast<T>(-911);
  std::array<T, 3 * N + 2> tuples{};
  tuples.fill(kSentinel);
  T* const unaligned = tuples.data() + 1;
  for (int i = 0; i < 3 * N; ++i) {
    unaligned[i] = static_cast<T>(i + 1);
  }
  for (int count = 0; count <= N; ++count) {
    SCOPED_TRACE(testing::Message() << "N=" << N << " count=" << count);
    // `exact` ends after (count * 3) scalars, so ASan flags an over-read where loads are
    // instrumented (not AVX2 masked or NEON lane loads). `unaligned` continues with non-zero
    // data, so an over-read fails the zero-lane check on every backend.
    auto const source = std::make_unique<T[]>(3 * count + 1);
    T* const exact = source.get() + 1;
    std::copy_n(unaligned, 3 * count, exact);
    for (T const* src : {exact, unaligned}) {
      SCOPED_TRACE(src == exact ? "source=exact" : "source=unaligned");
      // Non-zero lanes before the load, so the zero check fails unless the load zeroes them.
      V x(kSentinel), y(kSentinel), z(kSentinel);
      LoadTransposed(src, x, y, z, count);
      for (int i = 0; i < N; ++i) {
        EXPECT_EQ(i < count ? unaligned[3 * i + 0] : T{}, x[i]);
        EXPECT_EQ(i < count ? unaligned[3 * i + 1] : T{}, y[i]);
        EXPECT_EQ(i < count ? unaligned[3 * i + 2] : T{}, z[i]);
      }

      std::array<T, 3 * N + 2> stored{};
      stored.fill(kSentinel);
      StoreTransposed(stored.data() + 1, x, y, z, count);
      for (int i = 0; i < isize(stored); ++i) {
        EXPECT_EQ(i >= 1 && i <= 3 * count ? tuples[i] : kSentinel, stored[i]);
      }
    }
  }
}

TEST(Simd, TransposedIOWithRuntimeCount) {
  ExpectTransposedIOWithRuntimeCount<float, 4>();
  ExpectTransposedIOWithRuntimeCount<float, 8>();
  ExpectTransposedIOWithRuntimeCount<float, 12>();
  ExpectTransposedIOWithRuntimeCount<float, 16>();
  ExpectTransposedIOWithRuntimeCount<double, 2>();
  ExpectTransposedIOWithRuntimeCount<double, 4>();
  ExpectTransposedIOWithRuntimeCount<double, 6>();
  ExpectTransposedIOWithRuntimeCount<double, 8>();
  ExpectTransposedIOWithRuntimeCount<double, 16>();
  ExpectTransposedIOWithRuntimeCount<int, 4>();
  ExpectTransposedIOWithRuntimeCount<int, 8>();
  ExpectTransposedIOWithRuntimeCount<int, 12>();
  ExpectTransposedIOWithRuntimeCount<int, 16>();
  ExpectTransposedIOWithRuntimeCount<int64_t, 2>();
  ExpectTransposedIOWithRuntimeCount<int64_t, 4>();
  ExpectTransposedIOWithRuntimeCount<int64_t, 6>();
  ExpectTransposedIOWithRuntimeCount<int64_t, 8>();
  ExpectTransposedIOWithRuntimeCount<int64_t, 16>();
}

// MulAdd and its variants must round once in every supported build, including SIMD emulation, so
// that an explicit MulAdd sequence rounds the same with and without SIMD.
template <class T, int N>
static void ExpectMulAddIsFused() {
  using V = Simd<T, N>;
  // (1 + eps) * (1 - eps) = 1 - eps^2 rounds to 1, so only a fused operation keeps the eps^2.
  T constexpr kEps = std::numeric_limits<T>::epsilon();
  V const a{T(1) + kEps};
  V const b{T(1) - kEps};
  V const one{T(1)};
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(-kEps * kEps, MulAdd(a, b, -one)[i]);
    EXPECT_EQ(-kEps * kEps, MulSub(a, b, one)[i]);
    EXPECT_EQ(kEps * kEps, NegMulAdd(a, b, one)[i]);
    EXPECT_EQ(kEps * kEps, NegMulSub(a, b, -one)[i]);
  }
}

TEST(Simd, MulAddIsFused) {
  ExpectMulAddIsFused<float, 4>();
  ExpectMulAddIsFused<float, 8>();
  ExpectMulAddIsFused<float, 16>();
  ExpectMulAddIsFused<double, 2>();
  ExpectMulAddIsFused<double, 4>();
  ExpectMulAddIsFused<double, 8>();
}

#if MOCHI_USE_SIMD

namespace {

template <class V, size_t... Is>
V MakeSequenceFromScalars(std::index_sequence<Is...>) {
  using T = typename V::Scalar;
  return V{static_cast<T>(Is + 1)...};
}

template <class V, size_t... Is>
void ExpectScalarConstructor(std::index_sequence<Is...>) {
  using T = typename V::Scalar;
  auto const value = V{static_cast<T>(Is + 1)...};
  for (int i = 0; i < V::kSize; ++i) {
    T const expected = i < sizeof...(Is) ? static_cast<T>(i + 1) : T{};
    EXPECT_EQ(expected, value[i]);
  }
}

template <class V, size_t N = 2>
void ExpectAllScalarConstructors() {
  ExpectScalarConstructor<V>(std::make_index_sequence<N>{});
  if constexpr (N < V::kSize) {
    ExpectAllScalarConstructors<V, N + 1>();
  }
}

template <class T, int N>
void ExpectCompositeLaneOperations() {
  using V = Simd<T, N>;
  ExpectAllScalarConstructors<V>();
  std::array<T, N> values{};
  for (int i = 0; i < N; ++i) {
    values[i] = static_cast<T>(static_cast<int64_t>(i) + 1);
  }
  auto const original = Load<V>(values.data());
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(values[i], original[i]);
  }

  auto const scalarConstructed = MakeSequenceFromScalars<V>(std::make_index_sequence<N>{});
  auto const sequence = V::Sequence();
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(values[i], scalarConstructed[i]);
    EXPECT_EQ(static_cast<T>(i), sequence[i]);
  }

  if constexpr (Simd<T, N / 2>::kIsSupported) {
    auto const reconstructed = V{GetHalf<0>(original), GetHalf<1>(original)};
    for (int i = 0; i < N; ++i) {
      EXPECT_EQ(values[i], reconstructed[i]);
    }
  }

  if constexpr (!std::is_same_v<T, int64_t> && !IsHalf<T>) {
    for (int index = 0; index < N; ++index) {
      T const replacement = static_cast<T>(-41 - index);
      auto const changed = Set(original, index, replacement);
      for (int lane = 0; lane < N; ++lane) {
        EXPECT_EQ(lane == index ? replacement : values[lane], changed[lane]);
      }
    }
  }
}

} // namespace

TEST(SimdComposite, NativeWidthBoundaries) {
  ExpectCompositeLaneOperations<float, 12>();
  ExpectCompositeLaneOperations<int, 12>();
  ExpectCompositeLaneOperations<double, 6>();
  ExpectCompositeLaneOperations<int64_t, 6>();
  ExpectCompositeLaneOperations<float, 28>();
  ExpectCompositeLaneOperations<int, 28>();
  ExpectCompositeLaneOperations<float, 24>();
  ExpectCompositeLaneOperations<int, 24>();
}

#if MOCHI_HAS_SIMD_HALF
static_assert(Simd<Half, 24>::kIsSupported);
static_assert(Simd<Half, 24>::kIsComposite);
#if MOCHI_ARCH_X64_AVX512
static_assert(Simd<Half, 24>::kSizeFirst == 16 && Simd<Half, 24>::kSizeSecond == 8);
#endif

TEST(SimdComposite, Half24NativeWidthBoundary) {
  ExpectCompositeLaneOperations<Half, 24>();
}
#endif // MOCHI_HAS_SIMD_HALF

#endif // MOCHI_USE_SIMD
