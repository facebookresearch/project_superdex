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

#include "x64_simd_half_inl.h" // for IntelliSense

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2

namespace mochi {

/***********************************************************************************************
  Simd<Half, 8> — 128-bit SSE register holding 8 half-precision floating point values
*/
template <>
class Simd<Half, 8> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(Half, 8, __m128i);

  MOCHI_FORCE_INLINE explicit Simd(Scalar val)
      : raw(_mm_set1_epi16(static_cast<short>(ReinterpretCast<uint16_t>(val)))) {}

  MOCHI_FORCE_INLINE Simd(
      Scalar a,
      Scalar b,
      Scalar c = Scalar{},
      Scalar d = Scalar{},
      Scalar e = Scalar{},
      Scalar f = Scalar{},
      Scalar g = Scalar{},
      Scalar h = Scalar{})
      : raw(_mm_set_epi16(
            static_cast<short>(ReinterpretCast<uint16_t>(h)),
            static_cast<short>(ReinterpretCast<uint16_t>(g)),
            static_cast<short>(ReinterpretCast<uint16_t>(f)),
            static_cast<short>(ReinterpretCast<uint16_t>(e)),
            static_cast<short>(ReinterpretCast<uint16_t>(d)),
            static_cast<short>(ReinterpretCast<uint16_t>(c)),
            static_cast<short>(ReinterpretCast<uint16_t>(b)),
            static_cast<short>(ReinterpretCast<uint16_t>(a)))) {}

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm_setzero_si128();
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Half const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Zero();
    } else if constexpr (N == 4) {
      return _mm_loadl_epi64(reinterpret_cast<__m128i const*>(ptr));
    } else if constexpr (N < kSize) {
      Simd result = Zero();
      memcpy(&result.raw, ptr, N * sizeof(Half));
      return result;
    } else {
      return _mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr));
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
    } else if constexpr (N < kSize) {
      memcpy(ptr, &v.raw, N * sizeof(Half));
    } else {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.raw);
    }
  }

  static MOCHI_FORCE_INLINE void Store(Half* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    memcpy(ptr, &v.raw, n * sizeof(Half));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm_and_si128(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm_or_si128(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm_xor_si128(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return _mm_xor_si128(raw, _mm_set1_epi32(-1));
  }

  // IEEE 754 float equality: +0 == -0, NaN != NaN (matches Simd<float> behavior)
  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    __m256 fa = _mm256_cvtph_ps(raw);
    __m256 fb = _mm256_cvtph_ps(rhs.raw);
    return _mm256_movemask_ps(_mm256_cmp_ps(fa, fb, _CMP_EQ_OQ)) == 0xFF;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    __m256 fa = _mm256_cvtph_ps(a.raw);
    __m256 fb = _mm256_cvtph_ps(b.raw);
    __m256 cmp = _mm256_cmp_ps(fa, fb, _CMP_EQ_OQ);
    __m128i lo = _mm256_castsi256_si128(_mm256_castps_si256(cmp));
    __m128i hi = _mm256_extracti128_si256(_mm256_castps_si256(cmp), 1);
    return _mm_packs_epi32(lo, hi);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b);
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto mask = _mm_movemask_epi8(v.raw);
    if constexpr (N == kSize) {
      return mask == 0xFFFF;
    } else {
      int constexpr kBytesNeeded = N * 2;
      int constexpr kMask = (1 << kBytesNeeded) - 1;
      return (mask & kMask) == kMask;
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto mask = _mm_movemask_epi8(v.raw);
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
    return ReinterpretCast<Half>(static_cast<uint16_t>(_mm_extract_epi16(v.raw, i)));
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

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
