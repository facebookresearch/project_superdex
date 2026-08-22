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
  Simd<float, 4>
*/
template <>
class Simd<float, 4> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(float, 4, float32x4_t);

  MOCHI_FORCE_INLINE Simd(float a, float b, float c = 0.0f, float d = 0.0f) : raw{a, b, c, d} {}
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  MOCHI_FORCE_INLINE Simd(U a) : raw{vdupq_n_f32(a)} {}

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar Get(Simd v) {
    static_assert(i >= 0 && i < 4, "Index out of range");
    return vgetq_lane_f32(v.raw, i);
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

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd AsPoint(Simd a) {
    return vsetq_lane_f32(1.0f, a.raw, 3);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd AsDirection(Simd a) {
    return vsetq_lane_f32(0.0f, a.raw, 3);
  }

  template <int x, int y, int z, int w>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Blend(Simd a, Simd b) {
    static_assert(
        x >= 0 && x <= 1 && y >= 0 && y <= 1 && z >= 0 && z <= 1 && w >= 0 && w <= 1,
        "invalid blend index");
    int constexpr kCount = x + y + z + w;
    if constexpr (kCount == 0) {
      return a;
    } else if constexpr (kCount == 4) {
      return b;
    } else if constexpr (kCount == 1) {
      // Replace one lane of a with the corresponding lane of b.
      int constexpr kLane = x ? 0 : (y ? 1 : (z ? 2 : 3));
      return vcopyq_laneq_f32(a.raw, kLane, b.raw, kLane);
    } else if constexpr (kCount == 3) {
      // Replace one lane of b with the corresponding lane of a.
      int constexpr kLane = !x ? 0 : (!y ? 1 : (!z ? 2 : 3));
      return vcopyq_laneq_f32(b.raw, kLane, a.raw, kLane);
    } else if constexpr (x == 1 && y == 1) { // <1,1,0,0>: low half from b, high from a
      return vcombine_f32(vget_low_f32(b.raw), vget_high_f32(a.raw));
    } else if constexpr (z == 1 && w == 1) { // <0,0,1,1>: low half from a, high from b
      return vcombine_f32(vget_low_f32(a.raw), vget_high_f32(b.raw));
    } else {
      // Remaining patterns: <1,0,1,0>, <0,1,0,1>, <1,0,0,1>, <0,1,1,0>.
      // mask lane = -1 -> select from a; mask lane = 0 -> select from b.
      auto mask = int32x4_t{x ? 0 : -1, y ? 0 : -1, z ? 0 : -1, w ? 0 : -1};
      return Select(vreinterpretq_f32_s32(mask), a, b);
    }
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= 4, "Invalid number of components");
    uint64_t mask =
        vget_lane_u64(vreinterpret_u64_u16(vqmovn_u32(vreinterpretq_u32_f32(v.raw))), 0);
    if constexpr (N == kSize) {
      return mask == 0xFFFFFFFFFFFFFFFFULL;
    } else {
      int constexpr kNumBits = N * 16; // 64-bit mask has 16 bits per lane
      auto constexpr kMustBeSet = (uint64_t(1) << kNumBits) - 1;
      return (mask & kMustBeSet) == kMustBeSet;
    }
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= 4, "Invalid number of components");
    uint64_t mask =
        vget_lane_u64(vreinterpret_u64_u16(vqmovn_u32(vreinterpretq_u32_f32(v.raw))), 0);
    if constexpr (N == kSize) {
      return mask != 0;
    } else {
      int constexpr kNumBits = N * 16; // 64-bit mask has 16 bits per lane
      auto constexpr kMayBeSet = (uint64_t(1) << kNumBits) - 1;
      return (mask & kMayBeSet) != 0;
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Broadcast(float const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Broadcast(Simd v) {
    return vdupq_laneq_f32(v.raw, i);
  }

  template <int N = kSize>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Load([[maybe_unused]] float const* ptr) {
    static_assert(N >= 0 && N <= 4);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{ptr[0], 0, 0, 0};
    } else if constexpr (N == 2) {
      return Simd{ptr[0], ptr[1], 0, 0};
    } else if constexpr (N == 3) {
      return Simd{ptr[0], ptr[1], ptr[2], 0};
    } else {
      return vld1q_f32(ptr);
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    // clang-format off
    switch (n) {
      case 1: return Load<1>(ptr);
      case 2: return Load<2>(ptr);
      case 3: return Load<3>(ptr);
      case 4: return Load<4>(ptr);
      MOCHI_UNLIKELY default: return Zero();
    } // clang-format on
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd LoadIndexed(
      float const* ptr,
      Simd<int, 4> const& indices) {
    return Simd<float, 4>{
        ptr[indices.raw[0]], ptr[indices.raw[1]], ptr[indices.raw[2]], ptr[indices.raw[3]]};
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(float const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Unsupported kTupleCount");
    if constexpr (kTupleCount == 1) {
      out0.raw = float32x4_t{ptr[0], 0.0f, 0.0f, 0.0f};
      out1.raw = float32x4_t{ptr[1], 0.0f, 0.0f, 0.0f};
      out2.raw = float32x4_t{ptr[2], 0.0f, 0.0f, 0.0f};
    } else if constexpr (kTupleCount == 2) {
      out0.raw = float32x4_t{ptr[0], ptr[3], 0.0f, 0.0f};
      out1.raw = float32x4_t{ptr[1], ptr[4], 0.0f, 0.0f};
      out2.raw = float32x4_t{ptr[2], ptr[5], 0.0f, 0.0f};
    } else if constexpr (kTupleCount == 3) {
      out0.raw = float32x4_t{ptr[0], ptr[3], ptr[6], 0.0f};
      out1.raw = float32x4_t{ptr[1], ptr[4], ptr[7], 0.0f};
      out2.raw = float32x4_t{ptr[2], ptr[5], ptr[8], 0.0f};
    } else { // kTupleCount == 4 (kSize)
      float32x4x3_t result = vld3q_f32(ptr);
      out0.raw = result.val[0];
      out1.raw = result.val[1];
      out2.raw = result.val[2];
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Select(Simd mask, Simd a, Simd b) {
    return vbslq_f32(vreinterpretq_u32_f32(mask.raw), a.raw, b.raw);
  }

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd SetBasisVector() {
    static_assert(i >= 0 && i <= 3, "Invalid component index");
    auto zeros = vdupq_n_f32(0);
    return vsetq_lane_f32(1.0f, zeros, i);
  }

  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Shuffle(Simd a, Simd b) {
    static_assert(
        x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4 && w >= 0 && w < 4, "Invalid index");
    return Simd{a.raw[x], a.raw[y], b.raw[z], b.raw[w]};
  }

  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Shuffle(Simd v) {
    return Shuffle<x, y, z, w>(v, v);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd SignBitMask() {
    return vreinterpretq_f32_s32(vdupq_n_s32((int32_t)0x80000000));
  }

  template <int N = kSize>
  MOCHI_FORCE_INLINE static void Store([[maybe_unused]] float* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N < kSize) {
      memcpy(ptr, &v, sizeof(float) * N);
    } else {
      vst1q_f32(ptr, v.raw);
    }
  }

  MOCHI_FORCE_INLINE static void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    // clang-format off
    switch (n) {
      case 1: Store<1>(ptr, v); break;
      case 2: Store<2>(ptr, v); break;
      case 3: Store<3>(ptr, v); break;
      case 4: Store<4>(ptr, v); break;
      MOCHI_UNLIKELY default: break;
    } // clang-format on
  }

  MOCHI_FORCE_INLINE static int StoreSelected(float* ptr, Simd condition, Simd values) {
    uint32x4_t shifted = vshrq_n_u32(vreinterpretq_u32_f32(condition.raw), 31);
    uint32x4_t const multipliers = {1, 2, 4, 8};
    uint32x4_t weighted = vmulq_u32(shifted, multipliers);
    uint32_t count = vaddvq_u32(shifted);
    uint32_t mask = vaddvq_u32(weighted);
    uint8x16_t pattern = vld1q_u8(arm_simd::kStoreSelectedShuffleTableS4[mask]);
    uint8x16_t packed = vqtbl1q_u8(vreinterpretq_u8_f32(values.raw), pattern);
    vst1q_f32(ptr, vreinterpretq_f32_u8(packed));
    return static_cast<int>(count);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(float* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    if constexpr (kTupleCount == 1) {
      ptr[0] = a.raw[0];
      ptr[1] = b.raw[0];
      ptr[2] = c.raw[0];
    } else if constexpr (kTupleCount == 2) {
      Simd::Store(ptr, Simd{a[0], b[0], c[0], a[1]});
      ptr[4] = b.raw[1];
      ptr[5] = c.raw[1];
    } else if constexpr (kTupleCount == 3) {
      Simd::Store(ptr + 0, Simd{a[0], b[0], c[0], a[1]});
      Simd::Store(ptr + 4, Simd{b[1], c[1], a[2], b[2]});
      ptr[8] = c.raw[2];
    } else {
      vst3q_f32(ptr, float32x4x3_t({a.raw, b.raw, c.raw}));
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Sqrt(Simd v) {
    return vsqrtq_f32(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd RcpApprox(Simd v) {
    return vrecpeq_f32(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd RcpSqrtApprox(Simd v) {
    return vrsqrteq_f32(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Abs(Simd v) {
    return vabsq_f32(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Min(Simd a, Simd b) {
    return vminq_f32(a.raw, b.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Max(Simd a, Simd b) {
    return vmaxq_f32(a.raw, b.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Floor(Simd a) {
    return vrndmq_f32(a.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd FastRound(Simd v) {
    return vrndaq_f32(v.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd MulAdd(Simd a, Simd b, Simd c) {
    return vfmaq_f32(c.raw, b.raw, a.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd MulSub(Simd a, Simd b, Simd c) {
    return MulAdd(a, b, -c);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd NegMulAdd(Simd a, Simd b, Simd c) {
    return vfmsq_f32(c.raw, b.raw, a.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd NegMulSub(Simd a, Simd b, Simd c) {
    return NegMulAdd(a, b, -c);
  }

  template <int N = 4>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HMin(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return mochi::Min(a.raw[0], a.raw[1]);
    } else if constexpr (N == 3) {
      return vminvq_f32(vsetq_lane_f32(std::numeric_limits<Scalar>::infinity(), a.raw, 3));
    } else {
      return vminvq_f32(a.raw);
    }
  }

  template <int N = 4>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HMax(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return mochi::Max(a.raw[0], a.raw[1]);
    } else if constexpr (N == 3) {
      // HMax({x, y, z, -infinity})
      return vmaxvq_f32(vsetq_lane_f32(-std::numeric_limits<Scalar>::infinity(), a.raw, 3));
    } else {
      return vmaxvq_f32(a.raw);
    }
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HSum(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return a.raw[0] + a.raw[1];
    } else if constexpr (N == 3) {
      return a.raw[0] + a.raw[1] + a.raw[2];
    } else if constexpr (N == 4) {
      return vaddvq_f32(a.raw);
    }
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HProd(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return a.raw[0] * a.raw[1];
    } else if constexpr (N == 3) {
      return a.raw[0] * a.raw[1] * a.raw[2];
    } else if constexpr (N == 4) {
      return a.raw[0] * a.raw[1] * a.raw[2] * a.raw[3];
    }
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Dot(Simd a, Simd b) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    return Simd{HSum<N>(a * b)};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return vreinterpretq_f32_u32(vcltq_f32(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return vreinterpretq_f32_u32(vcgtq_f32(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return vreinterpretq_f32_u32(vcleq_f32(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return vreinterpretq_f32_u32(vcgeq_f32(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Equal(Simd a, Simd b) {
    return vreinterpretq_f32_u32(vceqq_f32(a.raw, b.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Zero() {
    return vdupq_n_f32(0);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    return AllTrue<kSize>(Equal(*this, rhs));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return vnegq_f32(raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return vaddq_f32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return vsubq_f32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return vmulq_f32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return vdivq_f32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return vreinterpretq_f32_u32(
        vandq_u32(vreinterpretq_u32_f32(raw), vreinterpretq_u32_f32(rhs.raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return vreinterpretq_f32_u32(
        vorrq_u32(vreinterpretq_u32_f32(raw), vreinterpretq_u32_f32(rhs.raw)));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return vreinterpretq_f32_u32(
        veorq_u32(vreinterpretq_u32_f32(raw), vreinterpretq_u32_f32(rhs.raw)));
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON
