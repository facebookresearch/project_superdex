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
