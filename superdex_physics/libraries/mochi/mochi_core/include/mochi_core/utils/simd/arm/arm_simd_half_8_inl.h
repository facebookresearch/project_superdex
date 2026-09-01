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

namespace mochi {

/***********************************************************************************************
  Simd<Half, 8> — 128-bit NEON register holding 8 half-precision floating point values
*/
template <>
class Simd<Half, 8> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(Half, 8, float16x8_t);

  MOCHI_FORCE_INLINE explicit Simd(Scalar val)
      : raw(vreinterpretq_f16_u16(vdupq_n_u16(ReinterpretCast<uint16_t>(val)))) {}

  MOCHI_FORCE_INLINE Simd(
      Scalar a,
      Scalar b,
      Scalar c = Scalar{},
      Scalar d = Scalar{},
      Scalar e = Scalar{},
      Scalar f = Scalar{},
      Scalar g = Scalar{},
      Scalar h = Scalar{}) {
    Scalar const buf[] = {a, b, c, d, e, f, g, h};
    raw = vld1q_f16(reinterpret_cast<float16_t const*>(buf));
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return vreinterpretq_f16_u16(vdupq_n_u16(0));
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Half const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Zero();
    } else if constexpr (N == 4) {
      float16x4_t low = vld1_f16(reinterpret_cast<float16_t const*>(ptr));
      return vcombine_f16(low, vreinterpret_f16_u16(vdup_n_u16(0)));
    } else if constexpr (N < kSize) {
      Simd result = Zero();
      memcpy(&result.raw, ptr, N * sizeof(Half));
      return result;
    } else {
      return vld1q_f16(reinterpret_cast<float16_t const*>(ptr));
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Half const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    Simd result = Zero();
    memcpy(&result.raw, ptr, n * sizeof(Half));
    return result;
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Half* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N == 4) {
      vst1_f16(reinterpret_cast<float16_t*>(ptr), vget_low_f16(v.raw));
    } else if constexpr (N < kSize) {
      memcpy(ptr, &v.raw, N * sizeof(Half));
    } else {
      vst1q_f16(reinterpret_cast<float16_t*>(ptr), v.raw);
    }
  }

  static MOCHI_FORCE_INLINE void Store(Half* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    memcpy(ptr, &v.raw, n * sizeof(Half));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return vreinterpretq_f16_u16(
        vandq_u16(vreinterpretq_u16_f16(raw), vreinterpretq_u16_f16(rhs.raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return vreinterpretq_f16_u16(
        vorrq_u16(vreinterpretq_u16_f16(raw), vreinterpretq_u16_f16(rhs.raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return vreinterpretq_f16_u16(
        veorq_u16(vreinterpretq_u16_f16(raw), vreinterpretq_u16_f16(rhs.raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return vreinterpretq_f16_u16(vmvnq_u16(vreinterpretq_u16_f16(raw)));
  }

  // IEEE 754 float equality: +0 == -0, NaN != NaN (matches Simd<float> behavior)
  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    return AllTrue<kSize>(Equal(*this, rhs));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
#if MOCHI_ARCH_ARM_NEON_FP16_ARITHMETIC
    return vreinterpretq_f16_u16(vceqq_f16(a.raw, b.raw));
#else
    // Baseline ARMv8-A lacks FP16 vector comparison. Widen exactly to FP32, then narrow
    // the comparison masks back to the same 16-bit lane representation as vceqq_f16.
    uint16x4_t const low =
        vmovn_u32(vceqq_f32(vcvt_f32_f16(vget_low_f16(a.raw)), vcvt_f32_f16(vget_low_f16(b.raw))));
    uint16x4_t const high = vmovn_u32(
        vceqq_f32(vcvt_f32_f16(vget_high_f16(a.raw)), vcvt_f32_f16(vget_high_f16(b.raw))));
    return vreinterpretq_f16_u16(vcombine_u16(low, high));
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b);
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    // vmovn_u16 narrows each 16-bit lane to 8 bits, preserving 1:1 lane mapping
    uint16x8_t bits = vreinterpretq_u16_f16(v.raw);
    uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(bits)), 0);
    if constexpr (N == kSize) {
      return mask == 0xFFFFFFFFFFFFFFFFULL;
    } else {
      int constexpr kNumBits = N * 8;
      auto constexpr kMustBeSet = (uint64_t(1) << kNumBits) - 1;
      return (mask & kMustBeSet) == kMustBeSet;
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    uint16x8_t bits = vreinterpretq_u16_f16(v.raw);
    uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(bits)), 0);
    if constexpr (N == kSize) {
      return mask != 0;
    } else {
      int constexpr kNumBits = N * 8;
      auto constexpr kMustBeSet = (uint64_t(1) << kNumBits) - 1;
      return (mask & kMustBeSet) != 0;
    }
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return ReinterpretCast<Half>(
        static_cast<uint16_t>(vgetq_lane_u16(vreinterpretq_u16_f16(v.raw), i)));
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range.");
    switch (i) { // clang-format off
      case 0: return Get<0>(v);
      case 1: return Get<1>(v);
      case 2: return Get<2>(v);
      case 3: return Get<3>(v);
      case 4: return Get<4>(v);
      case 5: return Get<5>(v);
      case 6: return Get<6>(v);
      case 7: return Get<7>(v);
      MOCHI_UNLIKELY default: return Scalar{};
    } // clang-format on
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON
