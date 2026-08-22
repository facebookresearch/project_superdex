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
  Simd<int, 4>
*/
template <>
class Simd<int, 4> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(int, 4, int32x4_t);

  MOCHI_FORCE_INLINE Simd(int a, int b, int c = 0, int d = 0) : raw{a, b, c, d} {}
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  MOCHI_FORCE_INLINE Simd(U a) : raw{vdupq_n_s32(a)} {}

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar Get(Simd v) {
    static_assert(i >= 0 && i < 4, "Index out of range");
    return vgetq_lane_s32(v.raw, i);
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

  // Set via 2 int64_t instead of 4 int
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd SetInt64(int64_t a, int64_t b) {
    return vreinterpretq_s32_s64(int64x2_t{a, b});
  }

  template <int N>
  [[nodiscard]] MOCHI_FORCE_INLINE static bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= 4, "Invalid number of components");
    uint64_t mask =
        vget_lane_u64(vreinterpret_u64_u16(vqmovn_u32(vreinterpretq_u32_s32(v.raw))), 0);
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
        vget_lane_u64(vreinterpret_u64_u16(vqmovn_u32(vreinterpretq_u32_s32(v.raw))), 0);
    if constexpr (N == kSize) {
      return mask != 0;
    } else {
      int constexpr kNumBits = N * 16; // 64-bit mask has 16 bits per lane
      auto constexpr kMayBeSet = (uint64_t(1) << kNumBits) - 1;
      return (mask & kMayBeSet) != 0;
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Broadcast(int const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Broadcast(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return vdupq_laneq_s32(v.raw, i);
  }

  template <int N = 4>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HMin(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return mochi::Min(a.raw[0], a.raw[1]);
    } else if constexpr (N == 3) {
      // HMin({x, y, z, max()})
      return vminvq_s32(vsetq_lane_s32(std::numeric_limits<Scalar>::max(), a.raw, 3));
    } else {
      return vminvq_s32(a.raw);
    }
  }

  template <int N = 4>
  [[nodiscard]] MOCHI_FORCE_INLINE static Scalar HMax(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return mochi::Max(a.raw[0], a.raw[1]);
    } else if constexpr (N == 3) {
      // HMax({x, y, z, lowest()})
      return vmaxvq_s32(vsetq_lane_s32(std::numeric_limits<Scalar>::lowest(), a.raw, 3));
    } else {
      return vmaxvq_s32(a.raw);
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
      return vaddvq_s32(a.raw);
    }
  }

  template <int N = kSize>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Load([[maybe_unused]] int const* ptr) {
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
      return vld1q_s32(ptr);
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

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(int const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    if constexpr (kTupleCount == 1) {
      out0.raw = int32x4_t{ptr[0], 0, 0, 0};
      out1.raw = int32x4_t{ptr[1], 0, 0, 0};
      out2.raw = int32x4_t{ptr[2], 0, 0, 0};
    } else if constexpr (kTupleCount == 2) {
      out0.raw = int32x4_t{ptr[0], ptr[3], 0, 0};
      out1.raw = int32x4_t{ptr[1], ptr[4], 0, 0};
      out2.raw = int32x4_t{ptr[2], ptr[5], 0, 0};
    } else if constexpr (kTupleCount == 3) {
      out0.raw = int32x4_t{ptr[0], ptr[3], ptr[6], 0};
      out1.raw = int32x4_t{ptr[1], ptr[4], ptr[7], 0};
      out2.raw = int32x4_t{ptr[2], ptr[5], ptr[8], 0};
    } else { // kTupleCount == 4 (kSize)
      int32x4x3_t result = vld3q_s32(ptr);
      out0.raw = result.val[0];
      out1.raw = result.val[1];
      out2.raw = result.val[2];
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Min(Simd a, Simd b) {
    return vminq_s32(a.raw, b.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Max(Simd a, Simd b) {
    return vmaxq_s32(a.raw, b.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Select(Simd mask, Simd a, Simd b) {
    return vbslq_s32(vreinterpretq_u32_s32(mask.raw), a.raw, b.raw);
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
      return vcopyq_laneq_s32(a.raw, kLane, b.raw, kLane);
    } else if constexpr (kCount == 3) {
      // Replace one lane of b with the corresponding lane of a.
      int constexpr kLane = !x ? 0 : (!y ? 1 : (!z ? 2 : 3));
      return vcopyq_laneq_s32(b.raw, kLane, a.raw, kLane);
    } else if constexpr (x == 1 && y == 1) { // <1,1,0,0>: low half from b, high from a
      return vcombine_s32(vget_low_s32(b.raw), vget_high_s32(a.raw));
    } else if constexpr (z == 1 && w == 1) { // <0,0,1,1>: low half from a, high from b
      return vcombine_s32(vget_low_s32(a.raw), vget_high_s32(b.raw));
    } else {
      // Remaining patterns: <1,0,1,0>, <0,1,0,1>, <1,0,0,1>, <0,1,1,0>.
      // mask lane = -1 -> select from a; mask lane = 0 -> select from b.
      Simd const mask{int32x4_t{x ? 0 : -1, y ? 0 : -1, z ? 0 : -1, w ? 0 : -1}};
      return Select(mask, a, b);
    }
  }

  // return Simd{v[x], v[y], v[z], v[w]}
  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Shuffle(Simd v) {
    static_assert(
        x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4 && w >= 0 && w < 4, "Invalid index");
    return Simd{v.raw[x], v.raw[y], v.raw[z], v.raw[w]};
  }

  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Shuffle(Simd a, Simd b) {
    static_assert(
        x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4 && w >= 0 && w < 4, "Invalid index");
    return Simd{a.raw[x], a.raw[y], b.raw[z], b.raw[w]};
  }

  template <int N = kSize>
  MOCHI_FORCE_INLINE static void Store([[maybe_unused]] int* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N < kSize) {
      memcpy(ptr, &v, sizeof(int) * N);
    } else {
      vst1q_s32(ptr, v.raw);
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

  MOCHI_FORCE_INLINE static int StoreSelected(int* ptr, Simd condition, Simd values) {
    uint32x4_t shifted = vshrq_n_u32(vreinterpretq_u32_s32(condition.raw), 31);
    uint32x4_t multipliers = {1, 2, 4, 8}; // Optimizer can pull this out of a loop when appropriate
    uint32x4_t weighted = vmulq_u32(shifted, multipliers);
    uint32_t count = vaddvq_u32(shifted);
    uint32_t mask = vaddvq_u32(weighted);
    uint8x16_t pattern = vld1q_u8(arm_simd::kStoreSelectedShuffleTableS4[mask]);
    uint8x16_t packed = vqtbl1q_u8(vreinterpretq_u8_s32(values.raw), pattern);
    vst1q_s32(ptr, vreinterpretq_s32_u8(packed));
    return static_cast<int>(count);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(int* ptr, Simd a, Simd b, Simd c) {
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
      vst3q_s32(ptr, int32x4x3_t({a.raw, b.raw, c.raw}));
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Zero() {
    return vdupq_n_s32(0);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return vreinterpretq_s32_u32(vcltq_s32(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return vreinterpretq_s32_u32(vcgtq_s32(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return vreinterpretq_s32_u32(vcleq_s32(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return vreinterpretq_s32_u32(vcgeq_s32(this->raw, rhs.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd Equal(Simd a, Simd b) {
    return vreinterpretq_s32_u32(vceqq_s32(a.raw, b.raw));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    uint16x4_t t = vqmovn_u32(vceqq_s32(raw, rhs.raw));
    return vget_lane_u64(vreinterpret_u64_u16(t), 0) == uint64_t(-1);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return vmvnq_s32(raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return vnegq_s32(raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return vaddq_s32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return vsubq_s32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return vmulq_s32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return Simd{
        // NEON does not implement integer division
        raw[0] / rhs.raw[0],
        raw[1] / rhs.raw[1],
        raw[2] / rhs.raw[2],
        raw[3] / rhs.raw[3]};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return vandq_s32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return vorrq_s32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return veorq_s32(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<<(int shift) const {
    // NOTE: If shift were a constexpr, then vshlq_n_s32 would be better because the shift amount
    // could be an immediate value. Fortunately, Clang appears to be smart enough to do the right
    // thing.
    auto vShift = Simd(shift);
    return vshlq_s32(raw, vShift.raw);
  }

  template <int kShift>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd ShiftRight(Simd a) {
    return vshrq_n_s32(a.raw, kShift);
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_ARM_NEON
