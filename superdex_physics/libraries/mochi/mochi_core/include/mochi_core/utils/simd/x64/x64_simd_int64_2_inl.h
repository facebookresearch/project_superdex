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
    if constexpr (i == 0) {
      return _mm_cvtsi128_si64(v.raw); // SSE2
    } else if constexpr (i == 1) {
      return _mm_cvtsi128_si64(_mm_unpackhi_epi64(v.raw, v.raw)); // SSE2, SSE2
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return v.raw.m128i_i64[i];
#else
    switch (i) { // clang-format off
                case 0: return Get<0>(v);
                case 1: return Get<1>(v);
                MOCHI_UNLIKELY default: return 0;
            } // clang-format on
#endif
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto mask = GetMSBitMask(v); // One bit for each byte in the vector
    if constexpr (N == kSize) {
      return mask == 0x0000FFFF;
    } else {
      return (mask & 0x000000FF) == 0x000000FF;
    }
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
    switch (n) { // clang-format off
      case 1: return Load<1>(ptr);
      case 2: return Load<2>(ptr);
      MOCHI_UNLIKELY default: return Zero();
    } // clang-format on
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
    out0.raw = _mm_castpd_si128(_mm_shuffle_pd(a, b, 0b0010)); // [0,3]
    out1.raw = _mm_castpd_si128(_mm_shuffle_pd(a, c, 0b0001)); // [1,4]
    out2.raw = _mm_castpd_si128(_mm_shuffle_pd(b, c, 0b0010)); // [2,5]
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    // TODO: Use _mm_min_epi64 for AVX512
    return Simd{mochi::Min(Get<0>(a), Get<0>(b)), mochi::Min(Get<1>(a), Get<1>(b))};
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    // TODO: Use _mm_max_epi64 for AVX512
    return Simd{mochi::Max(Get<0>(a), Get<0>(b)), mochi::Max(Get<1>(a), Get<1>(b))};
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
    if constexpr (N == 0) {
    } else if constexpr (N < kSize) {
      // About 3X faster than a masked store on AMD. About the same on Intel.
      memcpy(ptr, &v.raw, sizeof(Scalar) * N);
    } else {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.raw); // SSE
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    // Faster than masked store on AMD.
    // clang-format off
            switch (n) {
                case 1: Store<1>(ptr, v); break;
                case 2: Store<2>(ptr, v); break;
                MOCHI_UNLIKELY default: break;
            } // clang-format on
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    auto mask = _mm_movemask_pd(_mm_castsi128_pd(condition.raw));
    auto swapped = _mm_castpd_si128(_mm_shuffle_pd(
        _mm_castsi128_pd(values.raw), _mm_castsi128_pd(values.raw), 1)); // swap halves
    auto blendMask = _mm_set1_epi32((mask & 1) - 1); // swap first bit of mask is zero
    auto packed = _mm_blendv_epi8(values.raw, swapped, blendMask);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), packed);
    return _mm_popcnt_u32(mask);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    // a = [0,3], b = [1,4], c = [2,5]
    auto d = _mm_shuffle_pd(_mm_castsi128_pd(a.raw), _mm_castsi128_pd(b.raw), 0b00); // [0,1]
    auto e = _mm_shuffle_pd(_mm_castsi128_pd(c.raw), _mm_castsi128_pd(a.raw), 0b10); // [2,3]
    auto f = _mm_shuffle_pd(_mm_castsi128_pd(b.raw), _mm_castsi128_pd(c.raw), 0b11); // [4,5]
    Simd::Store<2>(ptr, _mm_castpd_si128(d));
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 2, 0, 2);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 4, 0, 2);
    if constexpr (kCount1 > 0) {
      Simd::Store<kCount1>(ptr + 2, _mm_castpd_si128(e));
    }
    if constexpr (kCount2 > 0) {
      Simd::Store<kCount2>(ptr + 4, _mm_castpd_si128(f));
    }
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
    auto mask = GetMSBitMask(Equal(*this, rhs));
    return mask == 0xFFFF; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    auto mask = GetMSBitMask(NotEqual(*this, rhs));
    return mask != 0; // Any values not equal
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
    // Fallback
    // Requires AVX512 _mm_mullo_epi64
    return Simd{Get<0>(*this) * Get<0>(rhs), Get<1>(*this) * Get<1>(rhs)};
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
    return _mm_srli_epi64(a.raw, kShift); // SSE2
  }

 private:
  // Integer mask with the most significant bit of each byte in the vector
  [[nodiscard]] static MOCHI_FORCE_INLINE int GetMSBitMask(Simd a) {
    return _mm_movemask_epi8(a.raw); // SSE2
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
