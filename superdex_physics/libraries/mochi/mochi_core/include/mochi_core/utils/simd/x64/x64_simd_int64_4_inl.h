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
  Simd<int64_t, 4>
*/
template <>
class Simd<int64_t, 4> {
 public:
  static_assert(sizeof(int64_t) == sizeof(long long));

  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(int64_t, 4, __m256i);
  Simd(int64_t a, int64_t b, int64_t c = 0, int64_t d = 0)
      : raw(_mm256_set_epi64x(
            static_cast<long long>(d),
            static_cast<long long>(c),
            static_cast<long long>(b),
            static_cast<long long>(a))) {} // AVX
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm256_set1_epi64x(static_cast<long long>(a))) {} // AVX

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    // Do not use _mm256_extract_epi64 for MSVC builds because of a bug in the optimizer. It tries
    // to use register XMM18 even though "/arch:AVX2" was specified (AVX2 only has 16 vector
    // registers). Microsoft claims that the behavior is "by design" when an AVX-512 intrinsic is
    // used. However, _mm256_extract_epi64 is an AVX intrinsic. Maybe MSVC misclassified it?
    if constexpr (i < 2) {
      return _mm_extract_epi64(_mm256_castsi256_si128(v.raw), i);
    } else {
      return _mm_extract_epi64(_mm256_extracti128_si256(v.raw, 1), i - 2);
    }
#else
    return _mm256_extract_epi64(v.raw, i);
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return v.raw.m256i_i64[i];
#else
    switch (i) { // clang-format off
                case 0: return Get<0>(v);
                case 1: return Get<1>(v);
                case 2: return Get<2>(v);
                case 3: return Get<3>(v);
                MOCHI_UNLIKELY default: return 0;
            } // clang-format on
#endif
  }

  template <int iHalf>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<int64_t, 2> GetHalf(Simd a) {
    static_assert(iHalf == 0 || iHalf == 1);
    return _mm256_extracti128_si256(a.raw, iHalf); // AVX2
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    int mask = GetMSBitMask(v); // One bit for each byte in the vector
    if constexpr (N == kSize) {
      return mask == 0xFFFFFFFF;
    } else {
      int constexpr kNumBits = N * sizeof(Scalar);
      auto constexpr kMustBeSet = (1UL << kNumBits) - 1;
      return (mask & kMustBeSet) == kMustBeSet;
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Scalar const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Simd v) {
    return Shuffle<i, i, i, i>(v);
  }

  template <int N = 4>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMin(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    using HalfT = Simd<Scalar, 2>;
    if constexpr (N == 2) {
      return Get<0>(Min(a, Broadcast<1>(a)));
    } else if constexpr (N == 3) {
      auto lo = GetHalf<0>(a);
      auto hi = GetHalf<1>(a);
      return HalfT::Get<0>(HalfT::Min(HalfT::Min(lo, HalfT::Broadcast<1>(lo)), hi));
    } else {
      return HalfT::HMin(HalfT::Min(GetHalf<0>(a), GetHalf<1>(a)));
    }
  }

  template <int N = 4>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMax(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    using HalfT = Simd<Scalar, 2>;
    if constexpr (N == 2) {
      return Get<0>(Max(a, Broadcast<1>(a)));
    } else if constexpr (N == 3) {
      auto lo = GetHalf<0>(a);
      auto hi = GetHalf<1>(a);
      return HalfT::Get<0>(HalfT::Max(HalfT::Max(lo, HalfT::Broadcast<1>(lo)), hi));
    } else {
      return HalfT::HMax(HalfT::Max(GetHalf<0>(a), GetHalf<1>(a)));
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HSum(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return Get<0>(a) + Get<1>(a);
    } else if constexpr (N == 3) {
      using HalfT = Simd<int64_t, 2>;
      HalfT tmp = GetHalf<0>(a) + GetHalf<1>(a);
      return HalfT::Get<0>(tmp) + Get<1>(a); // (a[0] + a[2]) + a[1]
    } else if constexpr (N == 4) {
      using HalfT = Simd<int64_t, 2>;
      HalfT tmp = GetHalf<0>(a) + GetHalf<1>(a);
      return HalfT::Get<0>(tmp) + HalfT::Get<1>(tmp); // (a[0] + a[2]) + (a[1] + a[3]);
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{*ptr, 0, 0, 0};
    } else if constexpr (N == 2) {
      __m256i mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1); // AVX
      return _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr), mask); // AVX2
    } else if constexpr (N == 3) {
      __m256i mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1); // AVX
      return _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr), mask); // AVX2
    } else {
      return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(ptr)); // AVX
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    switch (n) { // clang-format off
                case 1: return Load<1>(ptr);
                case 2: return Load<2>(ptr);
                case 3: return Load<3>(ptr);
                case 4: return Load<4>(ptr);
                MOCHI_UNLIKELY default: return Zero();
            } // clang-format on
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    constexpr int kCount0 = Clamp(kTupleCount * 3 - 0, 0, 4);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 4, 0, 4);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 8, 0, 4);
    auto a = _mm256_castsi256_pd(Simd::Load<kCount0>(ptr).raw); // [0,1,2,3]
    auto b =
        _mm256_castsi256_pd(Simd::Load<kCount1>(kCount1 == 0 ? ptr : ptr + 4).raw); // [4,5,6,7]
    auto c =
        _mm256_castsi256_pd(Simd::Load<kCount2>(kCount2 == 0 ? ptr : ptr + 8).raw); // [8,9,10,11]

    auto d = _mm256_blend_pd(a, b, 0b0100); // [0,_,6,3]
    d = _mm256_blend_pd(d, c, 0b0010); // [0,9,6,3]
    auto e = _mm256_permute2f128_pd(d, d, 0x01); // [6,3,0,9]
    out0.raw = _mm256_castpd_si256(_mm256_blend_pd(d, e, 0b1010)); // [0,3,6,9]

    d = _mm256_blend_pd(a, b, 0b1001); // [4,1,_,7]
    d = _mm256_blend_pd(d, c, 0b0100); // [4,1,10,7]
    out1.raw = _mm256_castpd_si256(_mm256_shuffle_pd(d, d, 0b0101)); // [1,4,7,10]

    d = _mm256_blend_pd(a, b, 0b0010); // [_,5,2,_]
    d = _mm256_blend_pd(d, c, 0b1001); // [8,5,2,11]
    e = _mm256_permute2f128_pd(d, d, 0x01); // [2,11,8,5]
    out2.raw = _mm256_castpd_si256(_mm256_blend_pd(d, e, 0b0101)); // [2,5,8,11]
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    // TODO: Use _mm256_min_epi64 for AVX512
    return Simd{
        mochi::Min(Get<0>(a), Get<0>(b)),
        mochi::Min(Get<1>(a), Get<1>(b)),
        mochi::Min(Get<2>(a), Get<2>(b)),
        mochi::Min(Get<3>(a), Get<3>(b))};
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    // TODO: Use _mm256_min_epi64 for AVX512
    return Simd{
        mochi::Max(Get<0>(a), Get<0>(b)),
        mochi::Max(Get<1>(a), Get<1>(b)),
        mochi::Max(Get<2>(a), Get<2>(b)),
        mochi::Max(Get<3>(a), Get<3>(b))};
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm256_blendv_epi8(b.raw, a.raw, mask.raw); // AVX2
  }

  template <int x, int y, int z, int w>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd a) {
    if constexpr (x == 0 && y == 1 && z == 2 && w == 3) {
      return a;
    } else {
      return _mm256_permute4x64_epi64(a.raw, _MM_SHUFFLE(w, z, y, x)); // AVX2
    }
  }

  template <int x, int y, int z, int w>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd a, Simd b) {
    return Simd{Get<x>(a), Get<y>(a), Get<z>(b), Get<w>(b)};
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N < kSize) {
      memcpy(ptr, &v.raw, sizeof(Scalar) * N);
    } else {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.raw); // AVX2
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    // Faster than masked store on AMD.
    // clang-format off
            switch (n) {
                case 1: Store<1>(ptr, v); break;
                case 2: Store<2>(ptr, v); break;
                case 3: Store<3>(ptr, v); break;
                case 4: Store<4>(ptr, v); break;
                MOCHI_UNLIKELY default: break;
            } // clang-format on
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    auto mask = _mm256_movemask_pd(_mm256_castsi256_pd(condition.raw));
    // Load 8 bytes from the table, then zero-exend to get the shuffle pattern.
    auto const* tableRow =
        reinterpret_cast<__m128i const*>(x64_simd::kStoreSelectedShuffleTableD4[mask]);
    auto pattern = _mm256_cvtepu8_epi32(_mm_loadl_epi64(tableRow));
    auto packed = _mm256_permutevar8x32_epi32(values.raw, pattern);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), packed);
    return _mm_popcnt_u32(mask);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    // a = [0,3,6,9], b = [1,4,7,10], c = [2,5,8,11]
    auto a_ = _mm256_castsi256_pd(a.raw);
    auto b_ = _mm256_castsi256_pd(b.raw);
    auto c_ = _mm256_castsi256_pd(c.raw);
    auto d = _mm256_shuffle_pd(b_, b_, 0b0101); // [4,1,10,7]
    auto e = _mm256_blend_pd(a_, c_, 0b0101); // [2,3,8,9]
    e = _mm256_permute2f128_pd(e, e, 0x01); // [8,9,2,3]
    auto f = _mm256_blend_pd(a_, d, 0b1010); // [0,1,6,7]
    auto g = _mm256_blend_pd(d, c_, 0b1010); // [4,5,10,11]
    constexpr int kCount0 = Clamp(kTupleCount * 3 - 0, 0, 4);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 4, 0, 4);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 8, 0, 4);
    Simd::Store<kCount0>(
        ptr, Simd{_mm256_castpd_si256(_mm256_blend_pd(f, e, 0b1100))}); // [0,1,2,3]
    if constexpr (kCount1 > 0) {
      Simd::Store<kCount1>(
          ptr + 4, Simd{_mm256_castpd_si256(_mm256_blend_pd(g, f, 0b1100))}); // [4,5,6,7]
    }
    if constexpr (kCount2 > 0) {
      Simd::Store<kCount2>(
          ptr + 8, Simd{_mm256_castpd_si256(_mm256_blend_pd(e, g, 0b1100))}); // [8,9,10,11]
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm256_setzero_si256(); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return _mm256_cmpgt_epi64(rhs.raw, this->raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return _mm256_cmpgt_epi64(this->raw, rhs.raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return ~(*this > rhs); // No native support
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return ~(*this < rhs); // No native support
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return _mm256_cmpeq_epi64(a.raw, b.raw); // AVX2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b); // // No native support
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    auto mask = GetMSBitMask(Equal(*this, rhs));
    return mask == 0xFFFFFFFF; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    auto mask = GetMSBitMask(NotEqual(*this, rhs));
    return mask != 0; // Any values not equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    auto ones = _mm256_cmpeq_epi64(raw, raw); // AVX2
    return _mm256_xor_si256(raw, ones); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return _mm256_sub_epi64(_mm256_setzero_si256(), raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm256_add_epi64(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm256_sub_epi64(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    // Fallback
    // Requires AVX512 _mm256_mullo_epi64
    return Simd{
        Get<0>(*this) * Get<0>(rhs),
        Get<1>(*this) * Get<1>(rhs),
        Get<2>(*this) * Get<2>(rhs),
        Get<3>(*this) * Get<3>(rhs)};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
#if MOCHI_ARCH_X64_SVML
    return _mm256_div_epi64(raw, rhs.raw); // SSE
#else
    // Fallback
    return Simd{
        Get<0>(*this) / Get<0>(rhs),
        Get<1>(*this) / Get<1>(rhs),
        Get<2>(*this) / Get<2>(rhs),
        Get<3>(*this) / Get<3>(rhs)};
#endif
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm256_and_si256(raw, rhs.raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm256_or_si256(raw, rhs.raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm256_xor_si256(raw, rhs.raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<<(int rhs) const {
    return _mm256_slli_epi64(raw, rhs); // AVX2
  }

  template <int kShift>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd ShiftRight(Simd a) {
    return _mm256_srli_epi64(a.raw, kShift); // AVX2
  }

 private:
  // Integer mask with the most significant bit of each byte in the vector
  [[nodiscard]] static MOCHI_FORCE_INLINE int GetMSBitMask(Simd a) {
    return _mm256_movemask_epi8(a.raw); // AVX2
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
