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

#include "x64_simd_half_8_inl.h" // Must come before the definition of Simd<Half, 16>

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2

namespace mochi {

/***********************************************************************************************
  Simd<Half, 16> — 256-bit AVX register holding 16 half-precision floating point values
*/
template <>
class Simd<Half, 16> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(Half, 16, __m256i);

  MOCHI_FORCE_INLINE explicit Simd(Scalar val)
      : raw(_mm256_set1_epi16(static_cast<short>(ReinterpretCast<uint16_t>(val)))) {}

  MOCHI_FORCE_INLINE Simd(
      Scalar a,
      Scalar b,
      Scalar c = Scalar{},
      Scalar d = Scalar{},
      Scalar e = Scalar{},
      Scalar f = Scalar{},
      Scalar g = Scalar{},
      Scalar h = Scalar{},
      Scalar i = Scalar{},
      Scalar j = Scalar{},
      Scalar k = Scalar{},
      Scalar l = Scalar{},
      Scalar m = Scalar{},
      Scalar n = Scalar{},
      Scalar o = Scalar{},
      Scalar p = Scalar{})
      : raw(_mm256_set_epi16(
            static_cast<short>(ReinterpretCast<uint16_t>(p)),
            static_cast<short>(ReinterpretCast<uint16_t>(o)),
            static_cast<short>(ReinterpretCast<uint16_t>(n)),
            static_cast<short>(ReinterpretCast<uint16_t>(m)),
            static_cast<short>(ReinterpretCast<uint16_t>(l)),
            static_cast<short>(ReinterpretCast<uint16_t>(k)),
            static_cast<short>(ReinterpretCast<uint16_t>(j)),
            static_cast<short>(ReinterpretCast<uint16_t>(i)),
            static_cast<short>(ReinterpretCast<uint16_t>(h)),
            static_cast<short>(ReinterpretCast<uint16_t>(g)),
            static_cast<short>(ReinterpretCast<uint16_t>(f)),
            static_cast<short>(ReinterpretCast<uint16_t>(e)),
            static_cast<short>(ReinterpretCast<uint16_t>(d)),
            static_cast<short>(ReinterpretCast<uint16_t>(c)),
            static_cast<short>(ReinterpretCast<uint16_t>(b)),
            static_cast<short>(ReinterpretCast<uint16_t>(a)))) {}

  MOCHI_FORCE_INLINE Simd(Simd<Half, 8> const& low, Simd<Half, 8> const& high)
      : raw(_mm256_set_m128i(high.raw, low.raw)) {}

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<Half, 8> GetHalf(Simd v) {
    static_assert(i == 0 || i == 1, "Index must be 0 or 1");
    if constexpr (i == 0) {
      return _mm256_castsi256_si128(v.raw);
    } else {
      return _mm256_extracti128_si256(v.raw, 1);
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm256_setzero_si256();
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Half const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Zero();
    } else if constexpr (N <= 8) {
      return {Simd<Half, 8>::Load<N>(ptr), Simd<Half, 8>::Zero()};
    } else if constexpr (N < kSize) {
      return {Simd<Half, 8>::Load<8>(ptr), Simd<Half, 8>::Load<N - 8>(ptr + 8)};
    } else {
      return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(ptr));
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Half const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    if (n <= 8) {
      return {Simd<Half, 8>::Load(ptr, n), Simd<Half, 8>::Zero()};
    } else {
      return {Simd<Half, 8>::Load(ptr), Simd<Half, 8>::Load(ptr + 8, n - 8)};
    }
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Half* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N <= 8) {
      Simd<Half, 8>::Store<N>(ptr, GetHalf<0>(v));
    } else if constexpr (N < kSize) {
      Simd<Half, 8>::Store<8>(ptr, GetHalf<0>(v));
      Simd<Half, 8>::Store<N - 8>(ptr + 8, GetHalf<1>(v));
    } else {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.raw);
    }
  }

  static MOCHI_FORCE_INLINE void Store(Half* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    if (n <= 8) {
      Simd<Half, 8>::Store(ptr, GetHalf<0>(v), n);
    } else {
      Simd<Half, 8>::Store(ptr, GetHalf<0>(v), 8);
      Simd<Half, 8>::Store(ptr + 8, GetHalf<1>(v), n - 8);
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm256_and_si256(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm256_or_si256(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm256_xor_si256(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return _mm256_xor_si256(raw, _mm256_set1_epi32(-1));
  }

  // IEEE 754 float equality: +0 == -0, NaN != NaN (matches Simd<float> behavior)
  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    __m256 cmpLo = _mm256_cmp_ps(
        _mm256_cvtph_ps(_mm256_castsi256_si128(raw)),
        _mm256_cvtph_ps(_mm256_castsi256_si128(rhs.raw)),
        _CMP_EQ_OQ);
    __m256 cmpHi = _mm256_cmp_ps(
        _mm256_cvtph_ps(_mm256_extracti128_si256(raw, 1)),
        _mm256_cvtph_ps(_mm256_extracti128_si256(rhs.raw, 1)),
        _CMP_EQ_OQ);
    return _mm256_movemask_ps(_mm256_and_ps(cmpLo, cmpHi)) == 0xFF;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    __m256i cmpLo = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_cvtph_ps(_mm256_castsi256_si128(a.raw)),
        _mm256_cvtph_ps(_mm256_castsi256_si128(b.raw)),
        _CMP_EQ_OQ));
    __m256i cmpHi = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_cvtph_ps(_mm256_extracti128_si256(a.raw, 1)),
        _mm256_cvtph_ps(_mm256_extracti128_si256(b.raw, 1)),
        _CMP_EQ_OQ));
    // Pack 32-bit comparison results to 16-bit, then fix cross-lane ordering
    return _mm256_permute4x64_epi64(_mm256_packs_epi32(cmpLo, cmpHi), 0xD8);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b);
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto mask = _mm256_movemask_epi8(v.raw);
    if constexpr (N == kSize) {
      return mask == -1;
    } else {
      int constexpr kBytesNeeded = N * 2;
      int constexpr kMask = (1 << kBytesNeeded) - 1;
      return (mask & kMask) == kMask;
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto mask = _mm256_movemask_epi8(v.raw);
    if constexpr (N == kSize) {
      return mask != 0;
    } else {
      int constexpr kBytesNeeded = N * 2;
      int constexpr kMask = (1 << kBytesNeeded) - 1;
      return (mask & kMask) != 0;
    }
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    if constexpr (i < 8) {
      return Simd<Half, 8>::Get<i>(GetHalf<0>(v));
    } else {
      return Simd<Half, 8>::Get<i - 8>(GetHalf<1>(v));
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range.");
    if (i < 8) {
      return Simd<Half, 8>::Get(GetHalf<0>(v), i);
    } else {
      return Simd<Half, 8>::Get(GetHalf<1>(v), i - 8);
    }
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
