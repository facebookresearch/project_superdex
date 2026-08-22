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

#include "arm_simd_inl.h" // for IntelliSense

#if MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON

namespace mochi {

/***********************************************************************************************
  Simd<int64_t, 2>
*/
template <>
class Simd<int64_t, 2> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(int64_t, 2, int64x2_t);

  MOCHI_FORCE_INLINE Simd(int64_t a, int64_t b) : raw{a, b} {}
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  MOCHI_FORCE_INLINE Simd(U a) : raw{vdupq_n_s64(a)} {}

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return vgetq_lane_s64(v.raw, i);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
    return v.raw[i];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
    auto result = v;
    result.raw[i] = value;
    return result;
  }

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Set(Simd v, Scalar value) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return Set(v, i, value);
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static bool AllTrue(Simd v) {
    static_assert(N == 1 || N == 2, "Invalid N");
    uint64_t mask =
        vget_lane_u64(vreinterpret_u64_u16(vqmovn_u32(vreinterpretq_u32_s64(v.raw))), 0);
    if constexpr (N == 1) {
      return (mask & 0x00000000FFFFFFFFULL) == 0x00000000FFFFFFFFULL;
    } else {
      return mask == 0xFFFFFFFFFFFFFFFFULL;
    }
  }

  template <int x, int y>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Blend(Simd a, Simd b) {
    static_assert(x >= 0 && x <= 1 && y >= 0 && y <= 1, "invalid blend index");
    if constexpr (x == 0 && y == 0) {
      return a;
    } else if constexpr (x == 1 && y == 1) {
      return b;
    } else {
      auto mask = int64x2_t{x ? (int64_t)0 : (int64_t)-1, y ? (int64_t)0 : (int64_t)-1};
      return Select(mask, a, b);
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Broadcast(Scalar const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Broadcast(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return vdupq_laneq_s64(v.raw, i);
  }

  template <int N = 2>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HMin(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return mochi::Min(a.raw[0], a.raw[1]);
  }

  template <int N = 2>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HMax(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return mochi::Max(a.raw[0], a.raw[1]);
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HSum(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return vaddvq_s64(a.raw);
  }

  template <int N = kSize>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{ptr[0], 0};
    } else {
      return vld1q_s64(ptr);
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    // clang-format off
    switch (n) {
      case 1: return Load<1>(ptr);
      case 2: return Load<2>(ptr);
      MOCHI_UNLIKELY default: return Zero();
    } // clang-format on
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(int64_t const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Unsupported kTupleCount");
    if constexpr (kTupleCount == 1) {
      out0.raw = int64x2_t{ptr[0], 0};
      out1.raw = int64x2_t{ptr[1], 0};
      out2.raw = int64x2_t{ptr[2], 0};
    } else {
      int64x2x3_t result = vld3q_s64(ptr);
      out0.raw = result.val[0];
      out1.raw = result.val[1];
      out2.raw = result.val[2];
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Min(Simd a, Simd b) {
    return vbslq_s64(vcltq_s64(a.raw, b.raw), a.raw, b.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Max(Simd a, Simd b) {
    return vbslq_s64(vcgtq_s64(a.raw, b.raw), a.raw, b.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Select(Simd mask, Simd a, Simd b) {
    return vbslq_s64(vreinterpretq_u64_s64(mask.raw), a.raw, b.raw);
  }

  // return Simd{v[x], v[y]}
  template <int x = 0, int y = 1>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Shuffle(Simd v) {
    static_assert(x >= 0 && x < 2, "Invalid index");
    static_assert(y >= 0 && y < 2, "Invalid index");
    if constexpr (x == 0 && y == 0) {
      return Broadcast<0>(v);
    } else if constexpr (x == 0 && y == 1) {
      return v; // no change
    } else if constexpr (x == 1 && y == 0) {
      return vcombine_s64(vget_high_s64(v.raw), vget_low_s64(v.raw));
    } else if constexpr (x == 1 && y == 1) {
      return Broadcast<1>(v);
    }
  }

  template <int N = kSize>
  MOCHI_FORCE_INLINE static void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N < kSize) {
      memcpy(ptr, &v, sizeof(Scalar) * N);
    } else {
      vst1q_s64(ptr, v.raw);
    }
  }

  MOCHI_FORCE_INLINE static void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    // clang-format off
    switch (n) {
      case 1: Store<1>(ptr, v); break;
      case 2: Store<2>(ptr, v); break;
      MOCHI_UNLIKELY default: break;
    } // clang-format on
  }

  MOCHI_FORCE_INLINE static int StoreSelected(int64_t* ptr, Simd condition, Simd values) {
    uint64x2_t shifted = vshrq_n_u64(vreinterpretq_u64_s64(condition.raw), 63);
    uint32_t bit0 = vgetq_lane_u64(shifted, 0);
    uint32_t bit1 = vgetq_lane_u64(shifted, 1);
    uint32_t mask = bit0 | (bit1 << 1);
    uint32_t count = bit0 + bit1;
    uint8x16_t pattern = vld1q_u8(arm_simd::kStoreSelectedShuffleTableD2[mask]);
    uint8x16_t packed = vqtbl1q_u8(vreinterpretq_u8_s64(values.raw), pattern);
    vst1q_s64(ptr, vreinterpretq_s64_u8(packed));
    return static_cast<int>(count);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(int64_t* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    if constexpr (kTupleCount == 1) {
      ptr[0] = a.raw[0];
      ptr[1] = b.raw[0];
      ptr[2] = c.raw[0];
    } else {
      vst3q_s64(ptr, int64x2x3_t({a.raw, b.raw, c.raw}));
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Zero() {
    return vdupq_n_s64(0);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return vreinterpretq_s64_u64(vcltq_s64(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return vreinterpretq_s64_u64(vcgtq_s64(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return vreinterpretq_s64_u64(vcleq_s64(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return vreinterpretq_s64_u64(vcgeq_s64(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Equal(Simd a, Simd b) {
    return vreinterpretq_s64_u64(vceqq_s64(a.raw, b.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return vnegq_s64(raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return vaddq_s64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return vsubq_s64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return Simd{
        // NEON does not implement long integer multiplication
        raw[0] * rhs.raw[0],
        raw[1] * rhs.raw[1]};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return Simd{
        // NEON does not implement long integer division
        raw[0] / rhs.raw[0],
        raw[1] / rhs.raw[1]};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    uint32x2_t t = vqmovn_u64(vceqq_s64(raw, rhs.raw));
    return vget_lane_u64(vreinterpret_u64_u32(t), 0) == uint64_t(-1);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return vreinterpretq_s64_u32(vmvnq_u32(vreinterpretq_u32_s64(raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return vandq_s64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return vorrq_s64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return veorq_s64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<<(int shift) const {
    // NOTE: If shift were a constexpr, then vshlq_n_s64 would be better because the shift amount
    // could be an immediate value. Fortunately, Clang appears to be smart enough to do the right
    // thing.
    auto vShift = Simd(shift);
    return vshlq_s64(raw, vShift.raw);
  }

  template <int kShift>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd ShiftRight(Simd a) {
    return vshrq_n_s64(a.raw, kShift);
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON
