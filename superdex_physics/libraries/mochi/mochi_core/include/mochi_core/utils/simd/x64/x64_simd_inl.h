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

/**
  This file contains Simd specializations for x64 CPUs with AVX2 support.
  Used for both Intel and AMD CPUs.
*/

#include "../../simd.h" // for IntelliSense

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2

#include <immintrin.h>
#include <xmmintrin.h>
#include <cstring>

/***********************************************************************************************
  Simd Specializations for x64 Architecture
*/

// NOTE: Order of these headers matters in some cases. Do not sort alphabetically.

#include "x64_simd_tables_inl.h"

#include "x64_simd_int_4_inl.h"

#include "x64_simd_int_8_inl.h"

#include "x64_simd_int64_2_inl.h"

#include "x64_simd_int64_4_inl.h"

#include "x64_simd_double_2_inl.h"

#include "x64_simd_double_4_inl.h"

#include "x64_simd_float_4_inl.h"

#include "x64_simd_float_8_inl.h"

/***********************************************************************************************
  Simd Utlities for x64 Architecture
*/

namespace mochi {

// clang-format off
template <> MOCHI_FORCE_INLINE Simd<double, 2> ReinterpretCast<Simd<double, 2>, Simd<int64_t, 2>>(Simd<int64_t, 2> const& a) { return _mm_castsi128_pd(a.raw); } // SSE2
template <> MOCHI_FORCE_INLINE Simd<int64_t, 2> ReinterpretCast<Simd<int64_t, 2>, Simd<double, 2>>(Simd<double, 2> const& a) { return _mm_castpd_si128(a.raw); } // SSE2
// clang-format on

template <>
MOCHI_FORCE_INLINE Simd<int64_t, 2> StaticCast<Simd<int64_t, 2>, Simd<double, 2>>(
    Simd<double, 2> const& a) {
  // Native support for _mm_cvttpd_epi64 requires AVX512
  return {_mm_cvttsd_si64(a.raw), _mm_cvttsd_si64(_mm_unpackhi_pd(a.raw, a.raw))};
} // SSE2
template <>
MOCHI_FORCE_INLINE Simd<double, 2> StaticCast<Simd<double, 2>, Simd<int64_t, 2>>(
    Simd<int64_t, 2> const& a) {
  // Native support for _mm_cvtepi64_pd requires AVX512
  return {
      static_cast<double>(_mm_cvtsi128_si64(a.raw)),
      static_cast<double>(_mm_cvtsi128_si64(_mm_unpackhi_epi64(a.raw, a.raw)))};
} // SSE2

// clang-format off
template <> MOCHI_FORCE_INLINE Simd<double, 2> ReinterpretCast<Simd<double, 2>, Simd<float, 4>>(Simd<float, 4> const& a) { return _mm_castps_pd(a.raw); } // SSE2
template <> MOCHI_FORCE_INLINE Simd<float, 4> ReinterpretCast<Simd<float, 4>, Simd<double, 2>>(Simd<double, 2> const& a) { return _mm_castpd_ps(a.raw); } // SSE2

template <> MOCHI_FORCE_INLINE Simd<double, 2> ReinterpretCast<Simd<double, 2>, Simd<int, 4>>(Simd<int, 4> const& a) { return _mm_castsi128_pd(a.raw); } // SSE2
template <> MOCHI_FORCE_INLINE Simd<int, 4> ReinterpretCast<Simd<int, 4>, Simd<double, 2>>(Simd<double, 2> const& a) { return _mm_castpd_si128(a.raw); } // SSE2

template <> MOCHI_FORCE_INLINE Simd<int64_t, 2> ReinterpretCast<Simd<int64_t, 2>, Simd<float, 4>>(Simd<float, 4> const& a) { return _mm_castps_si128(a.raw); } // SSE2
template <> MOCHI_FORCE_INLINE Simd<float, 4> ReinterpretCast<Simd<float, 4>, Simd<int64_t, 2>>(Simd<int64_t, 2> const& a) { return _mm_castsi128_ps(a.raw); } // SSE2

template <> MOCHI_FORCE_INLINE Simd<int64_t, 2> ReinterpretCast<Simd<int64_t, 2>, Simd<int, 4>>(Simd<int, 4> const& a) { return a.raw; }
template <> MOCHI_FORCE_INLINE Simd<int, 4> ReinterpretCast<Simd<int, 4>, Simd<int64_t, 2>>(Simd<int64_t, 2> const& a) { return a.raw; }

template <> MOCHI_FORCE_INLINE Simd<float, 4> StaticCast<Simd<float, 4>, Simd<double, 4>>(Simd<double, 4> const& a) { return {_mm256_cvtpd_ps(a.raw)}; } // AVX
template <> MOCHI_FORCE_INLINE Simd<double, 4> StaticCast<Simd<double, 4>, Simd<float, 4>>(Simd<float, 4> const& a) { return {_mm256_cvtps_pd(a.raw)}; } // AVX

template <> MOCHI_FORCE_INLINE Simd<int, 4> StaticCast<Simd<int, 4>, Simd<double, 4>>(Simd<double, 4> const& a) { return {_mm256_cvttpd_epi32(a.raw)}; } // AVX
template <> MOCHI_FORCE_INLINE Simd<double, 4> StaticCast<Simd<double, 4>, Simd<int, 4>>(Simd<int, 4> const& a) { return {_mm256_cvtepi32_pd(a.raw)}; } // AVX

template <> MOCHI_FORCE_INLINE Simd<double, 4> ReinterpretCast<Simd<double, 4>, Simd<int64_t, 4>>(Simd<int64_t, 4> const& a) { return _mm256_castsi256_pd(a.raw); } // AVX
template <> MOCHI_FORCE_INLINE Simd<int64_t, 4> ReinterpretCast<Simd<int64_t, 4>, Simd<double, 4>>(Simd<double, 4> const& a) { return _mm256_castpd_si256(a.raw); } // AVX
// clang-format on

template <>
MOCHI_FORCE_INLINE Simd<int64_t, 4> StaticCast<Simd<int64_t, 4>, Simd<double, 4>>(
    Simd<double, 4> const& a) {
  /// Native support is provided in AVX512 '_mm256_cvttpd_epi64'
  using VType = Simd<double, 4>;
  return {
      static_cast<int64_t>(VType::Get<0>(a)),
      static_cast<int64_t>(VType::Get<1>(a)),
      static_cast<int64_t>(VType::Get<2>(a)),
      static_cast<int64_t>(VType::Get<3>(a))};
}

template <>
MOCHI_FORCE_INLINE Simd<double, 4> StaticCast<Simd<double, 4>, Simd<int64_t, 4>>(
    Simd<int64_t, 4> const& a) {
  /// Native support is provided in AVX512 '_mm256_cvtepi64_pd'
  using VType = Simd<int64_t, 4>;
  return {
      static_cast<double>(VType::Get<0>(a)),
      static_cast<double>(VType::Get<1>(a)),
      static_cast<double>(VType::Get<2>(a)),
      static_cast<double>(VType::Get<3>(a))};
}

// clang-format off
template <> MOCHI_FORCE_INLINE Simd<double, 4> ReinterpretCast<Simd<double, 4>, Simd<float, 8>>(Simd<float, 8> const& a) { return _mm256_castps_pd(a.raw); } // AVX
template <> MOCHI_FORCE_INLINE Simd<float, 8> ReinterpretCast<Simd<float, 8>, Simd<double, 4>>(Simd<double, 4> const& a) { return _mm256_castpd_ps(a.raw); } // AVX

template <> MOCHI_FORCE_INLINE Simd<double, 4> ReinterpretCast<Simd<double, 4>, Simd<int, 8>>(Simd<int, 8> const& a) { return _mm256_castsi256_pd(a.raw); } // AVX
template <> MOCHI_FORCE_INLINE Simd<int, 8> ReinterpretCast<Simd<int, 8>, Simd<double, 4>>(Simd<double, 4> const& a) { return _mm256_castpd_si256(a.raw); } // AVX

template <> MOCHI_FORCE_INLINE Simd<float, 4> ReinterpretCast<Simd<float, 4>, Simd<int, 4>>(Simd<int, 4> const& a) { return _mm_castsi128_ps(a.raw); } // SSE2
template <> MOCHI_FORCE_INLINE Simd<int, 4> ReinterpretCast<Simd<int, 4>, Simd<float, 4>>(Simd<float, 4> const& a) { return _mm_castps_si128(a.raw); } // SSE2

template <> MOCHI_FORCE_INLINE Simd<int, 4> StaticCast<Simd<int, 4>, Simd<float, 4>>(Simd<float, 4> const& a) { return {_mm_cvttps_epi32(a.raw)}; } // SSE2
template <> MOCHI_FORCE_INLINE Simd<float, 4> StaticCast<Simd<float, 4>, Simd<int, 4>>(Simd<int, 4> const& a) { return {_mm_cvtepi32_ps(a.raw)}; } // SSE2
// clang-format on

template <>
MOCHI_FORCE_INLINE Simd<int64_t, 4> StaticCast<Simd<int64_t, 4>, Simd<float, 4>>(
    Simd<float, 4> const& a) {
  // Native support '_mm256_cvttps_epi64' requires AVX512
  using VType = Simd<float, 4>;
  return {
      static_cast<int64_t>(VType::Get<0>(a)),
      static_cast<int64_t>(VType::Get<1>(a)),
      static_cast<int64_t>(VType::Get<2>(a)),
      static_cast<int64_t>(VType::Get<3>(a))};
}

template <>
MOCHI_FORCE_INLINE Simd<float, 4> StaticCast<Simd<float, 4>, Simd<int64_t, 4>>(
    Simd<int64_t, 4> const& a) {
  // Native support '_mm256_cvtepi64_ps' requires AVX512
  using VType = Simd<int64_t, 4>;
  return {
      static_cast<float>(VType::Get<0>(a)),
      static_cast<float>(VType::Get<1>(a)),
      static_cast<float>(VType::Get<2>(a)),
      static_cast<float>(VType::Get<3>(a))};
}

template <>
MOCHI_FORCE_INLINE Simd<int, 4> StaticCast<Simd<int, 4>, Simd<int64_t, 4>>(
    Simd<int64_t, 4> const& a) {
  // Native support '_mm256_cvtepi64_epi32' requires AVX512
  using VType = Simd<int64_t, 4>;
  return {
      static_cast<int>(VType::Get<0>(a)),
      static_cast<int>(VType::Get<1>(a)),
      static_cast<int>(VType::Get<2>(a)),
      static_cast<int>(VType::Get<3>(a))};
}
template <>
MOCHI_FORCE_INLINE Simd<int64_t, 4> StaticCast<Simd<int64_t, 4>, Simd<int, 4>>(
    Simd<int, 4> const& a) {
  return _mm256_cvtepi32_epi64(a.raw); // AVX2
}

// clang-format off
template <> MOCHI_FORCE_INLINE Simd<float, 8> ReinterpretCast<Simd<float, 8>, Simd<int, 8>>(Simd<int, 8> const& a) { return _mm256_castsi256_ps(a.raw); } // AVX
template <> MOCHI_FORCE_INLINE Simd<int, 8> ReinterpretCast<Simd<int, 8>, Simd<float, 8>>(Simd<float, 8> const& a) { return _mm256_castps_si256(a.raw); } // AVX
template <> MOCHI_FORCE_INLINE Simd<int, 8> StaticCast<Simd<int, 8>, Simd<float, 8>>(Simd<float, 8> const& a) { return _mm256_cvttps_epi32(a.raw); } // AVX
template <> MOCHI_FORCE_INLINE Simd<float, 8> StaticCast<Simd<float, 8>, Simd<int, 8>>(Simd<int, 8> const& a) { return _mm256_cvtepi32_ps(a.raw); } // AVX

template <> MOCHI_FORCE_INLINE Simd<float, 8> ReinterpretCast<Simd<float, 8>, Simd<int64_t, 4>>(Simd<int64_t, 4> const& a) { return _mm256_castsi256_ps(a.raw); } // AVX
template <> MOCHI_FORCE_INLINE Simd<int64_t, 4> ReinterpretCast<Simd<int64_t, 4>, Simd<float, 8>>(Simd<float, 8> const& a) { return _mm256_castps_si256(a.raw); } // AVX

template <> MOCHI_FORCE_INLINE Simd<int, 8> ReinterpretCast<Simd<int, 8>, Simd<int64_t, 4>>(Simd<int64_t, 4> const& a) { return a.raw; } // AVX
template <> MOCHI_FORCE_INLINE Simd<int64_t, 4> ReinterpretCast<Simd<int64_t, 4>, Simd<int, 8>>(Simd<int, 8> const& a) { return a.raw; } // AVX
// clang-format on

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
