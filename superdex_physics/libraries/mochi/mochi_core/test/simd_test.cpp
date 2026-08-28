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

#include <cmath>
#include <limits>

using namespace mochi;

// IsSimdSupportedType for supported types
static_assert(IsSimdSupportedType<float>);
static_assert(Simd<float>::kIsSupported);
static_assert(IsSimdSupportedType<double>);
static_assert(Simd<double>::kIsSupported);
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
static_assert(details::kNextSupportedSimdSize<float, 9> == 12);
static_assert(details::kNextSupportedSimdSize<float, 13> == 16);

static_assert(details::kNextSupportedSimdSize<double, 1> == 2);
static_assert(details::kNextSupportedSimdSize<double, 2> == 2);
static_assert(details::kNextSupportedSimdSize<double, 3> == 4);
static_assert(details::kNextSupportedSimdSize<double, 4> == 4);
static_assert(details::kNextSupportedSimdSize<double, 5> == 6);

static_assert(details::kNextSupportedSimdSize<int, 1> == 4);
static_assert(details::kNextSupportedSimdSize<int, 2> == 4);
static_assert(details::kNextSupportedSimdSize<int, 3> == 4);
static_assert(details::kNextSupportedSimdSize<int, 4> == 4);
static_assert(details::kNextSupportedSimdSize<int, 5> == 8);
static_assert(details::kNextSupportedSimdSize<int, 8> == 8);
static_assert(details::kNextSupportedSimdSize<int, 9> == 12);
static_assert(details::kNextSupportedSimdSize<int, 13> == 16);

static_assert(details::kNextSupportedSimdSize<int64_t, 1> == 2);
static_assert(details::kNextSupportedSimdSize<int64_t, 2> == 2);
static_assert(details::kNextSupportedSimdSize<int64_t, 3> == 4);
static_assert(details::kNextSupportedSimdSize<int64_t, 4> == 4);
static_assert(details::kNextSupportedSimdSize<int64_t, 5> == 6);

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
