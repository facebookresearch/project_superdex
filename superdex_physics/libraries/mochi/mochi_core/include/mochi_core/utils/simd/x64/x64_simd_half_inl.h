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

#include "../../half.h" // for IntelliSense

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2

/***********************************************************************************************
  Simd<Half, N> Specializations for x64 Architecture
*/

#include "x64_simd_half_16_inl.h"
#include "x64_simd_half_8_inl.h"

/***********************************************************************************************
  x64 Simd<Half, N> Utilities for x64 Architecture
*/

namespace mochi {

// clang-format off
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<float, 8> StaticCast<Simd<float, 8>, Simd<Half, 8>>(Simd<Half, 8> const& a) { return _mm256_cvtph_ps(a.raw); } // F16C
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 8> StaticCast<Simd<Half, 8>, Simd<float, 8>>(Simd<float, 8> const& a) { return _mm256_cvtps_ph(a.raw, _MM_FROUND_TO_NEAREST_INT); } // F16C

template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<float, 16> StaticCast<Simd<float, 16>, Simd<Half, 16>>(Simd<Half, 16> const& a) { return {StaticCast<Simd<float, 8>>(Simd<Half, 16>::GetHalf<0>(a)), StaticCast<Simd<float, 8>>(Simd<Half, 16>::GetHalf<1>(a))}; }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 16> StaticCast<Simd<Half, 16>, Simd<float, 16>>(Simd<float, 16> const& a) { return {StaticCast<Simd<Half, 8>>(a.first), StaticCast<Simd<Half, 8>>(a.second)}; }

template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 8> ReinterpretCast<Simd<Half, 8>, Simd<float, 4>>(Simd<float, 4> const& a) { return _mm_castps_si128(a.raw); }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<float, 4> ReinterpretCast<Simd<float, 4>, Simd<Half, 8>>(Simd<Half, 8> const& a) { return _mm_castsi128_ps(a.raw); }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 8> ReinterpretCast<Simd<Half, 8>, Simd<int, 4>>(Simd<int, 4> const& a) { return a.raw; }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<int, 4> ReinterpretCast<Simd<int, 4>, Simd<Half, 8>>(Simd<Half, 8> const& a) { return a.raw; }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 8> ReinterpretCast<Simd<Half, 8>, Simd<int64_t, 2>>(Simd<int64_t, 2> const& a) { return a.raw; }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<int64_t, 2> ReinterpretCast<Simd<int64_t, 2>, Simd<Half, 8>>(Simd<Half, 8> const& a) { return a.raw; }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 8> ReinterpretCast<Simd<Half, 8>, Simd<double, 2>>(Simd<double, 2> const& a) { return _mm_castpd_si128(a.raw); }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<double, 2> ReinterpretCast<Simd<double, 2>, Simd<Half, 8>>(Simd<Half, 8> const& a) { return _mm_castsi128_pd(a.raw); }

template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 16> ReinterpretCast<Simd<Half, 16>, Simd<float, 8>>(Simd<float, 8> const& a) { return _mm256_castps_si256(a.raw); }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<float, 8> ReinterpretCast<Simd<float, 8>, Simd<Half, 16>>(Simd<Half, 16> const& a) { return _mm256_castsi256_ps(a.raw); }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 16> ReinterpretCast<Simd<Half, 16>, Simd<int, 8>>(Simd<int, 8> const& a) { return a.raw; }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<int, 8> ReinterpretCast<Simd<int, 8>, Simd<Half, 16>>(Simd<Half, 16> const& a) { return a.raw; }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 16> ReinterpretCast<Simd<Half, 16>, Simd<double, 4>>(Simd<double, 4> const& a) { return _mm256_castpd_si256(a.raw); }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<double, 4> ReinterpretCast<Simd<double, 4>, Simd<Half, 16>>(Simd<Half, 16> const& a) { return _mm256_castsi256_pd(a.raw); }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 16> ReinterpretCast<Simd<Half, 16>, Simd<int64_t, 4>>(Simd<int64_t, 4> const& a) { return a.raw; }
template <> [[nodiscard]] MOCHI_FORCE_INLINE Simd<int64_t, 4> ReinterpretCast<Simd<int64_t, 4>, Simd<Half, 16>>(Simd<Half, 16> const& a) { return a.raw; }
// clang-format on

namespace details {
template <>
inline constexpr bool IsSimdSupportedTypeDef<Half> = true;
} // namespace details

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
