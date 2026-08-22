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
  Simd<double, 2> (NEON Implementation)
*/
template <>
class Simd<double, 2> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(double, 2, float64x2_t);

  MOCHI_FORCE_INLINE Simd(double a, double b) : raw{a, b} {}
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  MOCHI_FORCE_INLINE Simd(U a) : raw{vdupq_n_f64(a)} {}

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar Get(Simd v) {
    static_assert(i >= 0 && i < 2, "Index out of range");
    return vgetq_lane_f64(v.raw, i);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < 2, "Index out of range");
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
        vget_lane_u64(vreinterpret_u64_u16(vqmovn_u32(vreinterpretq_u32_f64(v.raw))), 0);
    if constexpr (N == 1) {
      return (mask & 0x00000000FFFFFFFFULL) == 0x00000000FFFFFFFFULL;
    } else {
      return mask == 0xFFFFFFFFFFFFFFFFULL;
    }
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static bool AnyTrue(Simd v) {
    static_assert(N == 1 || N == 2, "Invalid N");
    uint64_t mask =
        vget_lane_u64(vreinterpret_u64_u16(vqmovn_u32(vreinterpretq_u32_f64(v.raw))), 0);
    if constexpr (N == 1) {
      return (mask & 0x00000000FFFFFFFFULL) == 0x00000000FFFFFFFFULL;
    } else {
      return mask != 0;
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
      auto mask = vreinterpretq_f64_s64(
          int64x2_t{x ? (int64_t)0 : (int64_t)-1, y ? (int64_t)0 : (int64_t)-1});
      return Select(mask, a, b);
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Broadcast(double const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Broadcast(Simd v) {
    static_assert(i >= 0 && i < 2, "Index out of range");
    return vdupq_laneq_f64(v.raw, i);
  }

  template <int N = kSize>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Load([[maybe_unused]] double const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{*ptr, 0.0};
    } else {
      return vld1q_f64(ptr);
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

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd LoadIndexed(
      double const* ptr,
      Simd<int64_t, 2> const& indices) {
    return Simd<double, 2>{ptr[indices.raw[0]], ptr[indices.raw[1]]};
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(double const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Unsupported kTupleCount");
    if constexpr (kTupleCount == 1) {
      out0.raw = float64x2_t{ptr[0], 0.0};
      out1.raw = float64x2_t{ptr[1], 0.0};
      out2.raw = float64x2_t{ptr[2], 0.0};
    } else {
      float64x2x3_t result = vld3q_f64(ptr);
      out0.raw = result.val[0];
      out1.raw = result.val[1];
      out2.raw = result.val[2];
    }
  }

  template <int N = kSize>
  static void Store([[maybe_unused]] double* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N == 1) {
      *ptr = Get<0>(v);
    } else {
      vst1q_f64(ptr, v.raw);
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

  MOCHI_FORCE_INLINE static int StoreSelected(double* ptr, Simd condition, Simd values) {
    uint64x2_t shifted = vshrq_n_u64(vreinterpretq_u64_f64(condition.raw), 63);
    uint32_t bit0 = vgetq_lane_u64(shifted, 0);
    uint32_t bit1 = vgetq_lane_u64(shifted, 1);
    uint32_t mask = bit0 | (bit1 << 1);
    uint32_t count = bit0 + bit1;
    uint8x16_t pattern = vld1q_u8(arm_simd::kStoreSelectedShuffleTableD2[mask]);
    uint8x16_t packed = vqtbl1q_u8(vreinterpretq_u8_f64(values.raw), pattern);
    vst1q_f64(ptr, vreinterpretq_f64_u8(packed));
    return static_cast<int>(count);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(double* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    if constexpr (kTupleCount == 1) {
      ptr[0] = a.raw[0];
      ptr[1] = b.raw[0];
      ptr[2] = c.raw[0];
    } else {
      vst3q_f64(ptr, float64x2x3_t({a.raw, b.raw, c.raw}));
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Select(Simd mask, Simd a, Simd b) {
    return vbslq_f64(vreinterpretq_u64_f64(mask.raw), a.raw, b.raw);
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
      return vcombine_f64(vget_high_f64(v.raw), vget_low_f64(v.raw));
    } else if constexpr (x == 1 && y == 1) {
      return Broadcast<1>(v);
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd SignBitMask() {
    return vreinterpretq_f64_s64(vdupq_n_s64((int64_t)0x8000000000000000LL));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Sqrt(Simd v) {
    return vsqrtq_f64(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd RcpApprox(Simd v) {
    return vrecpeq_f64(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd RcpSqrtApprox(Simd v) {
    return vrsqrteq_f64(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Abs(Simd v) {
    return vabsq_f64(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Min(Simd a, Simd b) {
    return vminq_f64(a.raw, b.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Max(Simd a, Simd b) {
    return vmaxq_f64(a.raw, b.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Floor(Simd a) {
    return vrndmq_f64(a.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd FastRound(Simd v) {
    return vrndaq_f64(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd MulAdd(Simd a, Simd b, Simd c) {
    return vfmaq_f64(c.raw, b.raw, a.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd MulSub(Simd a, Simd b, Simd c) {
    return MulAdd(a, b, -c);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd NegMulAdd(Simd a, Simd b, Simd c) {
    return vfmsq_f64(c.raw, b.raw, a.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd NegMulSub(Simd a, Simd b, Simd c) {
    return NegMulAdd(a, b, -c);
  }

  template <int N = 2>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HMin(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return vminvq_f64(a.raw);
  }

  template <int N = 2>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HMax(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return vmaxvq_f64(a.raw);
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HSum(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return vaddvq_f64(a.raw);
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HProd(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return Get<0>(a) * Get<1>(a);
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Dot(Simd a, Simd b) {
    static_assert(N == 2, "Unsupported N");
    return Simd{HSum<N>(a * b)};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return vreinterpretq_f64_u64(vcltq_f64(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return vreinterpretq_f64_u64(vcgtq_f64(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return vreinterpretq_f64_u64(vcleq_f64(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return vreinterpretq_f64_u64(vcgeq_f64(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Equal(Simd a, Simd b) {
    return vreinterpretq_f64_u64(vceqq_f64(a.raw, b.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Zero() {
    return vdupq_n_f64(0);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    return AllTrue<kSize>(Equal(*this, rhs));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_f64(raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return vnegq_f64(raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return vaddq_f64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return vsubq_f64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return vmulq_f64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return vdivq_f64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return vreinterpretq_f64_s64(
        vandq_s64(vreinterpretq_s64_f64(raw), vreinterpretq_s64_f64(rhs.raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return vreinterpretq_f64_s64(
        vorrq_s64(vreinterpretq_s64_f64(raw), vreinterpretq_s64_f64(rhs.raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return vreinterpretq_f64_s64(
        veorq_s64(vreinterpretq_s64_f64(raw), vreinterpretq_s64_f64(rhs.raw)));
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON
