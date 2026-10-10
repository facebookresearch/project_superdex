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

#include "x64_simd_inl.h" // for IntelliSense

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2

namespace mochi {

/***********************************************************************************************
  Simd<int64_t, 2>
*/
template <>
class Simd<int64_t, 2> {
 public:
  static_assert(sizeof(int64_t) == sizeof(long long));

  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(int64_t, 2, __m128i);
  Simd(int64_t low, int64_t high)
      : raw(_mm_set_epi64x(static_cast<long long>(high), static_cast<long long>(low))) {} // SSE2
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm_set1_epi64x(static_cast<long long>(a))) {} // SSE2

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return v[i];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Scalar operator[](int i) const {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return raw.m128i_i64[i];
#else
    return static_cast<Scalar>(raw[i]);
#endif
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    int constexpr kLanes = (1 << N) - 1;
    return (ToMask(v) & kLanes) == kLanes;
  }

  template <int x, int y>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Blend(Simd a, Simd b) {
    static_assert(x >= 0 && x < 2 && y >= 0 && y < 2, "invalid blend index");
    if constexpr (x == 0 && y == 0) {
      return a;
    } else if constexpr (x == 1 && y == 1) {
      return b;
    } else {
      return _mm_castpd_si128(
          _mm_blend_pd(_mm_castsi128_pd(a.raw), _mm_castsi128_pd(b.raw), x | (y << 1))); // SSE4.1
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Scalar const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Simd v) {
    return Shuffle<i, i>(v);
  }

  template <int N = 2>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMin(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return Get<0>(Min(a, Broadcast<1>(a)));
  }

  template <int N = 2>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMax(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return Get<0>(Max(a, Broadcast<1>(a)));
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HSum(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return Get<0>(a) + Get<1>(a);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd HSumEach(Simd a, Simd b) {
    return _mm_add_epi64(
        _mm_unpacklo_epi64(a.raw, b.raw), _mm_unpackhi_epi64(a.raw, b.raw)); // SSE2
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{*ptr, 0};
    } else if constexpr (N == 2) {
      return _mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)); // SSE
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
#if MOCHI_ARCH_X64_AVX512
    return _mm_maskz_loadu_epi64(x64_simd::kLaneMasksS8[n], ptr); // AVX512VL
#else
    switch (n) { // clang-format off
      case 1: return Load<1>(ptr);
      case 2: return Load<2>(ptr);
      MOCHI_UNLIKELY default: return Zero();
    } // clang-format on
#endif
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 2, 0, 2);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 4, 0, 2);
    auto a = _mm_castsi128_pd(Simd::Load<2>(ptr).raw); // [0,1]
    auto b = _mm_castsi128_pd(Simd::Load<kCount1>(kCount1 == 0 ? ptr : ptr + 2).raw); // [2,3]
    auto c = _mm_castsi128_pd(Simd::Load<kCount2>(kCount2 == 0 ? ptr : ptr + 4).raw); // [4,5]
    Deinterleave3(a, b, c, out0, out1, out2);
  }

  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2, int count) {
    MOCHI_ASSERT_VERBOSE(count >= 0 && count <= kSize, "Invalid tuple count");
    int const n = count * 3;
    auto a = _mm_castsi128_pd(Simd::Load(ptr, mochi::Min(n, 2)).raw); // [0,1]
    auto b = _mm_castsi128_pd(Simd::Load(n > 2 ? ptr + 2 : ptr, Clamp(n - 2, 0, 2)).raw); // [2,3]
    auto c = _mm_castsi128_pd(Simd::Load(n > 4 ? ptr + 4 : ptr, Clamp(n - 4, 0, 2)).raw); // [4,5]
    Deinterleave3(a, b, c, out0, out1, out2);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
#if MOCHI_ARCH_X64_AVX512
    return _mm_min_epi64(a.raw, b.raw); // AVX512VL
#else
    return Simd{mochi::Min(Get<0>(a), Get<0>(b)), mochi::Min(Get<1>(a), Get<1>(b))};
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
#if MOCHI_ARCH_X64_AVX512
    return _mm_max_epi64(a.raw, b.raw); // AVX512VL
#else
    return Simd{mochi::Max(Get<0>(a), Get<0>(b)), mochi::Max(Get<1>(a), Get<1>(b))};
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm_blendv_epi8(b.raw, a.raw, mask.raw); // SSE4.1
  }

  template <int x = 0, int y = 1>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd v) {
    static_assert(x >= 0 && x < 2, "Invalid index");
    static_assert(y >= 0 && y < 2, "Invalid index");
    if constexpr (x == 0 && y == 1) {
      return v;
    } else {
      return _mm_castpd_si128(
          _mm_shuffle_pd(_mm_castsi128_pd(v.raw), _mm_castsi128_pd(v.raw), x | (y << 1))); // SSE2
    }
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    // Partial stores write the lanes straight from the register. With AVX2, masked stores are
    // several times slower on AMD and block store-to-load forwarding on Intel. With AVX-512, they
    // block forwarding on both. Measured on Zen 4 and Sapphire Rapids. A memcpy of the vector can
    // go through the stack.
    if constexpr (N == 0) {
    } else if constexpr (N == 1) {
      _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr), v.raw); // SSE2
    } else {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.raw); // SSE
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
#if MOCHI_ARCH_X64_AVX512
    _mm_mask_storeu_epi64(ptr, x64_simd::kLaneMasksS8[n], v.raw); // AVX512VL
#else
    // With AVX2, this is faster than masked store for a predictable value of n.
    // It is much slower for a random value of n.
    switch (n) { // clang-format off
      case 1: Store<1>(ptr, v); break;
      case 2: Store<2>(ptr, v); break;
      MOCHI_UNLIKELY default: break;
    } // clang-format on
#endif
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
#if MOCHI_ARCH_X64_AVX512
    auto const mask = static_cast<__mmask8>(ToMask(condition));
    _mm_mask_compressstoreu_epi64(ptr, mask, values.raw); // AVX512VL
    return _mm_popcnt_u32(mask);
#else
    auto mask = ToMask(condition);
    auto swapped = _mm_castpd_si128(_mm_shuffle_pd(
        _mm_castsi128_pd(values.raw), _mm_castsi128_pd(values.raw), 1)); // swap halves
    auto blendMask = _mm_set1_epi32((mask & 1) - 1); // swap first bit of mask is zero
    auto packed = _mm_blendv_epi8(values.raw, swapped, blendMask);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), packed);
    return _mm_popcnt_u32(mask);
#endif
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    Simd x0, x1, x2;
    Interleave3(a, b, c, x0, x1, x2);
    Simd::Store<2>(ptr, x0);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 2, 0, 2);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 4, 0, 2);
    if constexpr (kCount1 > 0) {
      Simd::Store<kCount1>(ptr + 2, x1);
    }
    if constexpr (kCount2 > 0) {
      Simd::Store<kCount2>(ptr + 4, x2);
    }
  }

  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c, int count) {
    MOCHI_ASSERT_VERBOSE(count >= 0 && count <= kSize, "Invalid tuple count");
    Simd x0, x1, x2;
    Interleave3(a, b, c, x0, x1, x2);
#if MOCHI_ARCH_X64_AVX512
    // Bit i selects scalar i. Masks come from shifts, not clamped counts, which clang may turn
    // back into branches. A zero mask stores nothing.
    uint32_t const bits = (1u << (count * 3)) - 1;
    _mm_mask_storeu_epi64(ptr, static_cast<__mmask8>(bits), x0.raw); // AVX512VL
    _mm_mask_storeu_epi64(ptr + (count > 0 ? 2 : 0), static_cast<__mmask8>(bits >> 2), x1.raw);
    _mm_mask_storeu_epi64(ptr + (count > 1 ? 4 : 0), static_cast<__mmask8>(bits >> 4), x2.raw);
#else
    // Scalar i is stored if i < count * 3. Comparing against scalar indices avoids clamped
    // counts, which clang may turn back into branches. A zero mask stores nothing.
    auto const n = _mm_set1_epi64x(count * 3);
    auto const mask0 = _mm_cmpgt_epi64(n, _mm_set_epi64x(1, 0)); // SSE4.2
    auto const mask1 = _mm_cmpgt_epi64(n, _mm_set_epi64x(3, 2));
    auto const mask2 = _mm_cmpgt_epi64(n, _mm_set_epi64x(5, 4));
    _mm_maskstore_epi64(reinterpret_cast<long long*>(ptr), mask0, x0.raw); // AVX2
    _mm_maskstore_epi64(reinterpret_cast<long long*>(ptr + (count > 0 ? 2 : 0)), mask1, x1.raw);
    _mm_maskstore_epi64(reinterpret_cast<long long*>(ptr + (count > 1 ? 4 : 0)), mask2, x2.raw);
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm_setzero_si128(); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return _mm_cmpgt_epi64(rhs.raw, this->raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return _mm_cmpgt_epi64(this->raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return ~(*this > rhs); // No native support until AVX512
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return ~(*this < rhs); // No native support until AVX512
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return _mm_cmpeq_epi64(a.raw, b.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b); // // No native support until AVX512
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    return ToMask(Equal(*this, rhs)) == 0x3; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return ToMask(NotEqual(*this, rhs)) != 0; // Any values not equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    __m128i ones = _mm_cmpeq_epi64(raw, raw); // SSE2
    return _mm_xor_si128(raw, ones); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return _mm_sub_epi64(_mm_setzero_si128(), raw); // SSE2, SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm_add_epi64(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm_sub_epi64(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
#if MOCHI_ARCH_X64_AVX512
    return _mm_mullo_epi64(raw, rhs.raw); // AVX512VL
#else
    return Simd{Get<0>(*this) * Get<0>(rhs), Get<1>(*this) * Get<1>(rhs)};
#endif
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
#if MOCHI_ARCH_X64_SVML
    return _mm_div_epi64(raw, rhs.raw); // SSE
#else
    // Fallback
    return Simd{Get<0>(*this) / Get<0>(rhs), Get<1>(*this) / Get<1>(rhs)};
#endif
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm_and_si128(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm_or_si128(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm_xor_si128(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<<(int rhs) const {
    return _mm_slli_epi64(raw, rhs); // SSE2
  }

  template <int kShift>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd ShiftRight(Simd a) {
    static_assert(kShift >= 0 && kShift < 64, "Shift amount out-of-range");
    if constexpr (kShift == 0) {
      return a;
    } else {
#if MOCHI_ARCH_X64_AVX512
      return _mm_srai_epi64(a.raw, kShift); // AVX512VL
#else
      auto shifted = _mm_srli_epi64(a.raw, kShift); // SSE2
      auto signMask = _mm_cmpgt_epi64(_mm_setzero_si128(), a.raw); // SSE4.2
      auto signFill = _mm_slli_epi64(signMask, 64 - kShift); // SSE2
      return _mm_or_si128(shifted, signFill); // SSE2
#endif
    }
  }

 private:
  // One bit per lane, from the lane's sign bit.
  [[nodiscard]] static MOCHI_FORCE_INLINE int ToMask(Simd a) {
#if MOCHI_ARCH_X64_AVX512
    return _mm_movepi64_mask(a.raw); // AVX512DQ, AVX512VL
#else
    return _mm_movemask_pd(_mm_castsi128_pd(a.raw)); // SSE2, SSE2
#endif
  }

  // Splits 6 consecutive scalars {a, b, c} into 3 vectors of every third scalar.
  MOCHI_FORCE_INLINE static void
  Deinterleave3(__m128d a, __m128d b, __m128d c, Simd& out0, Simd& out1, Simd& out2) {
    out0.raw = _mm_castpd_si128(_mm_shuffle_pd(a, b, 0b0010)); // [0,3]
    out1.raw = _mm_castpd_si128(_mm_shuffle_pd(a, c, 0b0001)); // [1,4]
    out2.raw = _mm_castpd_si128(_mm_shuffle_pd(b, c, 0b0010)); // [2,5]
  }

  // Inverse of Deinterleave3: out0, out1, out2 are the 6 interleaved scalars in memory order.
  MOCHI_FORCE_INLINE static void
  Interleave3(Simd a, Simd b, Simd c, Simd& out0, Simd& out1, Simd& out2) {
    // a = [0,3], b = [1,4], c = [2,5]
    auto const a_ = _mm_castsi128_pd(a.raw);
    auto const b_ = _mm_castsi128_pd(b.raw);
    auto const c_ = _mm_castsi128_pd(c.raw);
    out0.raw = _mm_castpd_si128(_mm_shuffle_pd(a_, b_, 0b00)); // [0,1]
    out1.raw = _mm_castpd_si128(_mm_shuffle_pd(c_, a_, 0b10)); // [2,3]
    out2.raw = _mm_castpd_si128(_mm_shuffle_pd(b_, c_, 0b11)); // [4,5]
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
