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
  Simd<int, 8>
*/
template <>
class Simd<int, 8> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(int, 8, __m256i);
  Simd(int a, int b, int c = 0, int d = 0, int e = 0, int f = 0, int g = 0, int h = 0)
      : raw(_mm256_set_epi32(h, g, f, e, d, c, b, a)) {} // AVX

  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm256_set1_epi32(a)) {} // AVX

  // Joint two Vec4i into one Vec8i
  Simd(Simd<int, 4> const& low, Simd<int, 4> const& high)
      : raw(_mm256_set_m128i(high.raw, low.raw)) {} // AVX

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE int Get(Simd v) {
    static_assert(i >= 0 && i < 8, "Index out of range");
    if constexpr (i == 0) {
      return _mm256_cvtsi256_si32(v.raw); // AVX
    } else if constexpr (i == 1) {
      return _mm256_cvtsi256_si32(_mm256_shuffle_epi32(v.raw, 0x01)); // AVX, AVX2
    } else if constexpr (i == 2) {
      return _mm256_cvtsi256_si32(_mm256_shuffle_epi32(v.raw, 0x02)); // AVX, AVX2
    } else if constexpr (i == 3) {
      return _mm256_cvtsi256_si32(_mm256_shuffle_epi32(v.raw, 0x03)); // AVX, AVX2
    } else if constexpr (i == 4) {
      return _mm256_cvtsi256_si32(_mm256_permute2f128_si256(v.raw, v.raw, 0x01)); // AVX, AVX
    } else if constexpr (i == 5) {
      auto tmp = _mm256_permute2f128_si256(v.raw, v.raw, 0x01); // AVX
      return _mm256_cvtsi256_si32(_mm256_shuffle_epi32(tmp, 0x01)); // AVX, AVX2
    } else if constexpr (i == 6) {
      auto tmp = _mm256_permute2f128_si256(v.raw, v.raw, 0x01); // AVX
      return _mm256_cvtsi256_si32(_mm256_shuffle_epi32(tmp, 0x02)); // AVX, AVX2
    } else if constexpr (i == 7) {
      auto tmp = _mm256_permute2f128_si256(v.raw, v.raw, 0x01); // AVX
      return _mm256_cvtsi256_si32(_mm256_shuffle_epi32(tmp, 0x03)); // AVX, AVX2
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Scalar operator[](int i) const {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return raw.m256i_i32[i];
#else
    switch (i) { // clang-format off
      case 0: return Get<0>(*this);
      case 1: return Get<1>(*this);
      case 2: return Get<2>(*this);
      case 3: return Get<3>(*this);
      case 4: return Get<4>(*this);
      case 5: return Get<5>(*this);
      case 6: return Get<6>(*this);
      case 7: return Get<7>(*this);
      MOCHI_UNLIKELY default: return 0;
    } // clang-format on
#endif
  }

  template <int iHalf>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<int, 4> GetHalf(Simd a) {
    static_assert(iHalf == 0 || iHalf == 1);
    return _mm256_extracti128_si256(a.raw, iHalf); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    auto result = v;
    result.raw.m256i_i32[i] = value;
    return result;
#else
    // clang-format off
    static constexpr __m256i kMasks[] = {
        {static_cast<long long>(0x00000000FFFFFFFFLL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0xFFFFFFFF00000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x00000000FFFFFFFFLL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0xFFFFFFFF00000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x00000000FFFFFFFFLL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0xFFFFFFFF00000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x00000000FFFFFFFFLL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0xFFFFFFFF00000000LL)}};
    // clang-format on
    return _mm256_blendv_epi8(v.raw, _mm256_set1_epi32(value), kMasks[i]); // AVX2
#endif
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, Scalar value) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return Set(v, i, value);
  }

  // Set via 4 int64_t instead of 8 int
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd
  SetInt64(int64_t a, int64_t b, int64_t c, int64_t d) {
    return _mm256_set_epi64x(d, c, b, a); // AVX
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    int constexpr kLanes = (1 << N) - 1;
    return (ToMask(v) & kLanes) == kLanes;
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    int constexpr kLanes = (1 << N) - 1;
    return (ToMask(v) & kLanes) != 0;
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Scalar const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Simd v) {
    return Simd{Get<i>(v)}; // TODO: There is probably a faster way
  }

  template <int N = 8>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMin(Simd a) {
    static_assert(N >= 2 && N <= 8, "Unsupported N");
    using HalfT = Simd<Scalar, 4>;
    if constexpr (N >= 2 && N <= 4) {
      return HalfT::HMin<N>(GetHalf<0>(a));
    } else {
      auto lo = GetHalf<0>(a);
      auto hi = GetHalf<1>(a);
      if constexpr (N != 8) {
        // Set the hi values we don't want to compare to the max int
        auto inf = HalfT{std::numeric_limits<Scalar>::max()};
        hi = HalfT::Blend<1, N >= 6, N >= 7, 0>(inf, hi);
      }
      return HalfT::HMin<4>(HalfT::Min(lo, hi));
    }
  }

  template <int N = 8>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMax(Simd a) {
    static_assert(N >= 2 && N <= 8, "Unsupported N");
    using HalfT = Simd<Scalar, 4>;
    if constexpr (N >= 2 && N <= 4) {
      return HalfT::HMax<N>(GetHalf<0>(a));
    } else {
      auto lo = GetHalf<0>(a);
      auto hi = GetHalf<1>(a);
      if constexpr (N != 8) {
        // Set the hi values we don't want to compare to the most negative int
        auto lowest = HalfT{std::numeric_limits<Scalar>::lowest()};
        hi = HalfT::Blend<1, N >= 6, N >= 7, 0>(lowest, hi);
      }
      return HalfT::HMax<4>(HalfT::Max(lo, hi));
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HSum(Simd a) {
    static_assert(N >= 2 && N <= 8, "Unsupported N");
    using HalfT = Simd<Scalar, kSize / 2>;
    if constexpr (N >= 2 && N <= 4) {
      return HalfT::HSum<N>(GetHalf<0>(a));
    } else if constexpr (N == 5) {
      auto tmp = GetHalf<0>(a) + GetHalf<1>(a);
      return HalfT::Get<0>(tmp) + Get<1>(a) + Get<2>(a) + Get<3>(a);
    } else if constexpr (N == 6) {
      auto tmp = GetHalf<0>(a) + GetHalf<1>(a);
      return HalfT::Get<0>(tmp) + HalfT::Get<1>(tmp) + Get<2>(a) + Get<3>(a);
    } else if constexpr (N == 7) {
      auto tmp = GetHalf<0>(a) + GetHalf<1>(a);
      return HalfT::Get<0>(tmp) + HalfT::Get<1>(tmp) + HalfT::Get<2>(tmp) + Get<3>(a);
    } else if constexpr (N == 8) {
      return HalfT::HSum<4>(GetHalf<0>(a) + GetHalf<1>(a));
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{*ptr, 0};
    } else if constexpr (N == 2) {
      auto const low = _mm_loadl_epi64(reinterpret_cast<__m128i const*>(ptr)); // SSE2
      return _mm256_zextsi128_si256(low); // AVX
    } else if constexpr (N == 3) {
      __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1); // AVX
      return _mm256_maskload_epi32(ptr, mask); // AVX2
    } else if constexpr (N == 4) {
      auto const low = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)); // SSE2
      return _mm256_zextsi128_si256(low); // AVX
    } else if constexpr (N == 5) {
      __m256i mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1); // AVX
      return _mm256_maskload_epi32(ptr, mask); // AVX2
    } else if constexpr (N == 6) {
      __m256i mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1); // AVX
      return _mm256_maskload_epi32(ptr, mask); // AVX2
    } else if constexpr (N == 7) {
      __m256i mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1); // AVX
      return _mm256_maskload_epi32(ptr, mask); // AVX2
    } else {
      return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(ptr)); // AVX
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
#if MOCHI_ARCH_X64_AVX512
    return _mm256_maskz_loadu_epi32(x64_simd::kLaneMasksS8[n], ptr); // AVX512VL
#else
    // Use the same masks as Simd<float, 8>
    return _mm256_maskload_epi32(ptr, x64_simd::kLoadMasksS8[n]); //  AVX2
#endif
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    constexpr int kCount0 = Clamp(kTupleCount * 3 - 0, 0, 8);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 8, 0, 8);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 16, 0, 8);

    //  [ 0, 1, 2, 3,   4, 5, 6, 7]
    auto a = Simd::Load<kCount0>(ptr).raw;
    //  [ 8, 9, 10, 11,  12, 13, 14, 15]
    auto b = Simd::Load<kCount1>(kCount1 == 0 ? ptr : ptr + 8).raw;
    // [16, 17, 18, 19,  20, 21, 22, 23]
    auto c = Simd::Load<kCount2>(kCount2 == 0 ? ptr : ptr + 16).raw;
    Deinterleave3(a, b, c, out0, out1, out2);
  }

  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2, int count) {
    MOCHI_ASSERT_VERBOSE(count >= 0 && count <= kSize, "Invalid tuple count");
    int const n = count * 3;
    auto a = Simd::Load(ptr, mochi::Min(n, 8)).raw;
    auto b = Simd::Load(n > 8 ? ptr + 8 : ptr, Clamp(n - 8, 0, 8)).raw;
    auto c = Simd::Load(n > 16 ? ptr + 16 : ptr, Clamp(n - 16, 0, 8)).raw;
    Deinterleave3(a, b, c, out0, out1, out2);
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v) {
    static_assert(N >= 0 && N <= kSize);
    // Partial stores write the lanes straight from the register. With AVX2, masked stores are
    // several times slower on AMD and block store-to-load forwarding on Intel. With AVX-512, they
    // block forwarding on both. Measured on Zen 4 and Sapphire Rapids. A memcpy of the vector can
    // go through the stack.
    using HalfT = Simd<Scalar, 4>;
    if constexpr (N == 0) {
    } else if constexpr (N <= 4) {
      HalfT::Store<N>(ptr, HalfT(_mm256_castsi256_si128(v.raw))); // AVX
    } else if constexpr (N < kSize) {
      HalfT::Store<N - 4>(ptr + 4, HalfT(_mm256_extracti128_si256(v.raw, 1))); // AVX2
      HalfT::Store<4>(ptr, HalfT(_mm256_castsi256_si128(v.raw))); // AVX
    } else {
      return _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.raw); // AVX
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
#if MOCHI_ARCH_X64_AVX512
    _mm256_mask_storeu_epi32(ptr, x64_simd::kLaneMasksS8[n], v.raw); // AVX512VL
#else
    // With AVX2, this is faster than masked store for a predictable value of n.
    // It is much slower for a random value of n.
    switch (n) { // clang-format off
      case 1: Store<1>(ptr, v); break;
      case 2: Store<2>(ptr, v); break;
      case 3: Store<3>(ptr, v); break;
      case 4: Store<4>(ptr, v); break;
      case 5: Store<5>(ptr, v); break;
      case 6: Store<6>(ptr, v); break;
      case 7: Store<7>(ptr, v); break;
      case 8: Store<8>(ptr, v); break;
      MOCHI_UNLIKELY default: break;
    } // clang-format on
#endif
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    auto mask = ToMask(condition);
    // Load 8 bytes from the table, then zero-exend to get the shuffle pattern.
    auto const* tableRow =
        reinterpret_cast<__m128i const*>(x64_simd::kStoreSelectedShuffleTableS8[mask]);
    auto pattern = _mm256_cvtepu8_epi32(_mm_loadl_epi64(tableRow));
    auto packed = _mm256_permutevar8x32_ps(_mm256_castsi256_ps(values.raw), pattern);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), _mm256_castps_si256(packed));
    return _mm_popcnt_u32(mask);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    Simd x0, x1, x2;
    Interleave3(a, b, c, x0, x1, x2);
    constexpr int kCount0 = Clamp(kTupleCount * 3 - 0, 0, 8);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 8, 0, 8);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 16, 0, 8);
    Simd::Store<kCount0>(ptr, x0);
    if constexpr (kCount1 > 0) {
      Simd::Store<kCount1>(ptr + 8, x1);
    }
    if constexpr (kCount2 > 0) {
      Simd::Store<kCount2>(ptr + 16, x2);
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
    _mm256_mask_storeu_epi32(ptr, static_cast<__mmask8>(bits), x0.raw); // AVX512VL
    _mm256_mask_storeu_epi32(ptr + (count > 2 ? 8 : 0), static_cast<__mmask8>(bits >> 8), x1.raw);
    _mm256_mask_storeu_epi32(ptr + (count > 5 ? 16 : 0), static_cast<__mmask8>(bits >> 16), x2.raw);
#else
    // Scalar i is stored if i < count * 3. Comparing against scalar indices avoids clamped
    // counts, which clang may turn back into branches. A zero mask stores nothing.
    auto const n = _mm256_set1_epi32(count * 3);
    auto const mask0 = _mm256_cmpgt_epi32(n, _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)); // AVX2
    auto const mask1 = _mm256_cmpgt_epi32(n, _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15));
    auto const mask2 = _mm256_cmpgt_epi32(n, _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23));
    _mm256_maskstore_epi32(ptr, mask0, x0.raw); // AVX2
    _mm256_maskstore_epi32(ptr + (count > 2 ? 8 : 0), mask1, x1.raw);
    _mm256_maskstore_epi32(ptr + (count > 5 ? 16 : 0), mask2, x2.raw);
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    return _mm256_min_epi32(a.raw, b.raw); // AVX2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    return _mm256_max_epi32(a.raw, b.raw); // AVX2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm256_blendv_epi8(b.raw, a.raw, mask.raw); // AVX2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm256_setzero_si256(); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return _mm256_cmpgt_epi32(rhs.raw, this->raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return _mm256_cmpgt_epi32(this->raw, rhs.raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return ~(*this > rhs);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return ~(*this < rhs);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return _mm256_cmpeq_epi32(a.raw, b.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b); // No native support until AVX512
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    return ToMask(Equal(*this, rhs)) == 0xFF; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return ToMask(NotEqual(*this, rhs)) != 0; // Any values not equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    auto ones = _mm256_set1_epi32(-1); // AVX
    return _mm256_xor_si256(raw, ones); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return Zero() - *this;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm256_add_epi32(raw, rhs.raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm256_sub_epi32(raw, rhs.raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return _mm256_mullo_epi32(raw, rhs.raw); // AVX2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
#if MOCHI_ARCH_X64_SVML
    return _mm256_div_epi32(raw, rhs.raw); // AVX2
#else
    // Fallback
    return Simd{
        Get<0>(*this) / Get<0>(rhs),
        Get<1>(*this) / Get<1>(rhs),
        Get<2>(*this) / Get<2>(rhs),
        Get<3>(*this) / Get<3>(rhs),
        Get<4>(*this) / Get<4>(rhs),
        Get<5>(*this) / Get<5>(rhs),
        Get<6>(*this) / Get<6>(rhs),
        Get<7>(*this) / Get<7>(rhs)};
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
    return _mm256_slli_epi32(raw, rhs); // SSE2
  }

  template <int kShift>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd ShiftRight(Simd a) {
    static_assert(kShift >= 0 && kShift < 32, "Shift amount out-of-range");
    return _mm256_srai_epi32(a.raw, kShift); // AVX2
  }

 private:
  // One bit per lane, from the lane's sign bit.
  [[nodiscard]] static MOCHI_FORCE_INLINE int ToMask(Simd a) {
    return _mm256_movemask_ps(_mm256_castsi256_ps(a.raw)); // AVX, AVX
  }

  // Splits 24 consecutive scalars {a, b, c} into 3 vectors of every third scalar.
  MOCHI_FORCE_INLINE static void
  Deinterleave3(__m256i a, __m256i b, __m256i c, Simd& out0, Simd& out1, Simd& out2) {
    auto d = _mm256_blend_epi32(a, b, 0b10010010); // [0,9,_,3,  12,_,6,15]
    auto e = _mm256_blend_epi32(d, c, 0b00100100); // [0,9,18,3, 12,21,6,15]
    auto f = _mm256_permute2x128_si256(e, e, 0x01); // [12,21,6,15,  0,9,18,3]
    auto g = _mm256_blend_epi32(e, f, 0b01000100); // [0,9,6,3, 12,21,18,15]
    out0.raw = _mm256_shuffle_epi32(g, _MM_SHUFFLE(1, 2, 3, 0)); // [0,3,6,9, 12,15,18,21]

    d = _mm256_blend_epi32(a, b, 0b00100100); // [_,1,10,_,  4,13,_,7]
    e = _mm256_blend_epi32(d, c, 0b01001001); // [16,1,10,19  4,13,22,7]
    f = _mm256_permute2x128_si256(e, e, 0x01); // [4,13,22,7  16,1,10,19]
    g = _mm256_blend_epi32(e, f, 0b10011001); // [4,1,10,7,  16,13,22,19]
    out1.raw = _mm256_shuffle_epi32(g, _MM_SHUFFLE(2, 3, 0, 1)); // [1,4,7,10,  13,16,19,22]

    d = _mm256_blend_epi32(a, b, 0b01001001); // [8,_,2,11,  _,5,14,_]
    e = _mm256_blend_epi32(d, c, 0b10010010); // [8,17,2,11,  20,5,14,23]
    f = _mm256_permute2x128_si256(e, e, 0x01); // [20,5,14,23,  8,17,2,11]
    g = _mm256_blend_epi32(e, f, 0b00100010); // [8,5,2,11,  20,17,14,23]
    out2.raw = _mm256_shuffle_epi32(g, _MM_SHUFFLE(3, 0, 1, 2)); // [2,5,8,11,  14,17,20,23]
  }

  // Inverse of Deinterleave3: out0, out1, out2 are the 24 interleaved scalars in memory order.
  MOCHI_FORCE_INLINE static void
  Interleave3(Simd a, Simd b, Simd c, Simd& out0, Simd& out1, Simd& out2) {
    // a = [0,3,6,9, 12,15,18,21]
    // b = [1,4,7,10, 13,16,19,22]
    // c = [2,5,8,11, 14,17,20,23]
    auto d = _mm256_shuffle_epi32(a.raw, _MM_SHUFFLE(1, 2, 3, 0)); // [0,9,6,3, 12,21,18,15]
    auto e = _mm256_shuffle_epi32(b.raw, _MM_SHUFFLE(2, 3, 0, 1)); // [4,1,10,7, 16,13,22,19]
    auto f = _mm256_shuffle_epi32(c.raw, _MM_SHUFFLE(3, 0, 1, 2)); // [8,5,2,11, 20,17,14,23]
    __m256i g = _mm256_blend_epi32(d, e, 0b00100010); // [0,1,_,3, 12,13,_,15]
    g = _mm256_blend_epi32(g, f, 0b01000100); // [0,1,2,3, 12,13,14,15]
    __m256i h = _mm256_blend_epi32(d, e, 0b10011001); // [4,_,6,7, 16,_,18,19]
    h = _mm256_blend_epi32(h, f, 0b00100010); // [4,5,6,7, 16,17,18,19]
    h = _mm256_permute2x128_si256(h, h, 0x01); // [16,17,18,19, 4,5,6,7]
    __m256i i = _mm256_blend_epi32(d, e, 0b01000100); // [_,9,10,_, _,21,22,_]
    i = _mm256_blend_epi32(i, f, 0b10011001); // [8,9,10,11, 20,21,22,23]
    out0 = _mm256_blend_epi32(g, h, 0b11110000); // [0,1,2,3, 4,5,6,7]
    out1 = _mm256_blend_epi32(i, g, 0b11110000); // [8,9,10,11, 12,13,14,15]
    out2 = _mm256_blend_epi32(h, i, 0b11110000); // [16,17,18,19, 20,21,22,23]
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
