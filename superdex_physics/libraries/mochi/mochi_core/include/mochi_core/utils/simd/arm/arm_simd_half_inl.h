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

#if MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON

/***********************************************************************************************
  Simd<Half, N> Specializations for ARM Architecture
*/

#include "arm_simd_half_8_inl.h"

/***********************************************************************************************
  ARM Simd<Half, N> Utilities for ARM Architecture
*/

namespace mochi {

template <>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<float, 8> StaticCast<Simd<float, 8>, Simd<Half, 8>>(
    Simd<Half, 8> const& a) {
  float16x4_t low_f16 = vget_low_f16(a.raw);
  float16x4_t high_f16 = vget_high_f16(a.raw);
  Simd<float, 4> low_f32 = vcvt_f32_f16(low_f16);
  Simd<float, 4> high_f32 = vcvt_f32_f16(high_f16);
  return {low_f32, high_f32};
}

template <>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<Half, 8> StaticCast<Simd<Half, 8>, Simd<float, 8>>(
    Simd<float, 8> const& a) {
  float16x4_t low_f16 = vcvt_f16_f32(a.first.raw);
  float16x4_t high_f16 = vcvt_f16_f32(a.second.raw);
  return vcombine_f16(low_f16, high_f16);
}

// clang-format off
template<> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half,    8> ReinterpretCast<Simd<Half,    8>, Simd<float,   4>>(Simd<float,   4> const& a) { return vreinterpretq_f16_f32(a.raw); }
template<> [[nodiscard]] MOCHI_FORCE_INLINE Simd<float,   4> ReinterpretCast<Simd<float,   4>, Simd<Half,    8>>(Simd<Half,    8> const& a) { return vreinterpretq_f32_f16(a.raw); }
template<> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half,    8> ReinterpretCast<Simd<Half,    8>, Simd<int,     4>>(Simd<int,     4> const& a) { return vreinterpretq_f16_s32(a.raw); }
template<> [[nodiscard]] MOCHI_FORCE_INLINE Simd<int,     4> ReinterpretCast<Simd<int,     4>, Simd<Half,    8>>(Simd<Half,    8> const& a) { return vreinterpretq_s32_f16(a.raw); }
template<> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half,    8> ReinterpretCast<Simd<Half,    8>, Simd<int64_t, 2>>(Simd<int64_t, 2> const& a) { return vreinterpretq_f16_s64(a.raw); }
template<> [[nodiscard]] MOCHI_FORCE_INLINE Simd<int64_t, 2> ReinterpretCast<Simd<int64_t, 2>, Simd<Half,    8>>(Simd<Half,    8> const& a) { return vreinterpretq_s64_f16(a.raw); }
template<> [[nodiscard]] MOCHI_FORCE_INLINE Simd<Half,    8> ReinterpretCast<Simd<Half,    8>, Simd<double,  2>>(Simd<double,  2> const& a) { return vreinterpretq_f16_f64(a.raw); }
template<> [[nodiscard]] MOCHI_FORCE_INLINE Simd<double,  2> ReinterpretCast<Simd<double,  2>, Simd<Half,    8>>(Simd<Half,    8> const& a) { return vreinterpretq_f64_f16(a.raw); }
// clang-format on

namespace details {
template <>
inline constexpr bool IsSimdSupportedTypeDef<Half> = true;
} // namespace details

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON
