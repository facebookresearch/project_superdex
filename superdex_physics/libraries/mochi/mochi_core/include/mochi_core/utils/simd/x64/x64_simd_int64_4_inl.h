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
    return v[i];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Scalar operator[](int i) const {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return raw.m256i_i64[i];
#else
    return static_cast<Scalar>(raw[i]);
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
    int constexpr kLanes = (1 << N) - 1;
    return (ToMask(v) & kLanes) == kLanes;
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

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd HSumEach(Simd a, Simd b, Simd c, Simd d) {
    // ac = {a[0] + a[2], a[1] + a[3], c[0] + c[2], c[1] + c[3]}, and bd likewise.
    auto const ac = _mm256_add_epi64(
        _mm256_blend_epi32(a.raw, c.raw, 0xF0),
        _mm256_permute2x128_si256(a.raw, c.raw, 0x21)); // AVX2
    auto const bd = _mm256_add_epi64(
        _mm256_blend_epi32(b.raw, d.raw, 0xF0),
        _mm256_permute2x128_si256(b.raw, d.raw, 0x21)); // AVX2
    // {ac[0], bd[0], ac[2], bd[2]} + {ac[1], bd[1], ac[3], bd[3]}.
    return _mm256_add_epi64(_mm256_unpacklo_epi64(ac, bd), _mm256_unpackhi_epi64(ac, bd)); // AVX2
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{*ptr, 0, 0, 0};
    } else if constexpr (N == 2) {
      auto const low = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)); // SSE2
      return _mm256_zextsi128_si256(low); // AVX
    } else if constexpr (N == 3) {
      __m256i mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1); // AVX
      return _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr), mask); // AVX2
    } else {
      return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(ptr)); // AVX
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
#if MOCHI_ARCH_X64_AVX512
    return _mm256_maskz_loadu_epi64(x64_simd::kLaneMasksS8[n], ptr); // AVX512VL
#else
    switch (n) { // clang-format off
                case 1: return Load<1>(ptr);
                case 2: return Load<2>(ptr);
                case 3: return Load<3>(ptr);
                case 4: return Load<4>(ptr);
                MOCHI_UNLIKELY default: return Zero();
            } // clang-format on
#endif
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
    Deinterleave3(a, b, c, out0, out1, out2);
  }

  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2, int count) {
    MOCHI_ASSERT_VERBOSE(count >= 0 && count <= kSize, "Invalid tuple count");
    int const n = count * 3;
    auto a = _mm256_castsi256_pd(Simd::Load(ptr, mochi::Min(n, 4)).raw); // [0,1,2,3]
    auto b =
        _mm256_castsi256_pd(Simd::Load(n > 4 ? ptr + 4 : ptr, Clamp(n - 4, 0, 4)).raw); // [4,5,6,7]
    auto c = _mm256_castsi256_pd(
        Simd::Load(n > 8 ? ptr + 8 : ptr, Clamp(n - 8, 0, 4)).raw); // [8,9,10,11]
    Deinterleave3(a, b, c, out0, out1, out2);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
#if MOCHI_ARCH_X64_AVX512
    return _mm256_min_epi64(a.raw, b.raw); // AVX512VL
#else
    return Simd{
        mochi::Min(Get<0>(a), Get<0>(b)),
        mochi::Min(Get<1>(a), Get<1>(b)),
        mochi::Min(Get<2>(a), Get<2>(b)),
        mochi::Min(Get<3>(a), Get<3>(b))};
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
#if MOCHI_ARCH_X64_AVX512
    return _mm256_max_epi64(a.raw, b.raw); // AVX512VL
#else
    return Simd{
        mochi::Max(Get<0>(a), Get<0>(b)),
        mochi::Max(Get<1>(a), Get<1>(b)),
        mochi::Max(Get<2>(a), Get<2>(b)),
        mochi::Max(Get<3>(a), Get<3>(b))};
#endif
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
    // Partial stores write the lanes straight from the register. With AVX2, masked stores are
    // several times slower on AMD and block store-to-load forwarding on Intel. With AVX-512, they
    // block forwarding on both. Measured on Zen 4 and Sapphire Rapids. A memcpy of the vector can
    // go through the stack.
    using HalfT = Simd<Scalar, 2>;
    if constexpr (N == 0) {
    } else if constexpr (N <= 2) {
      HalfT::Store<N>(ptr, HalfT(_mm256_castsi256_si128(v.raw))); // AVX
    } else if constexpr (N < kSize) {
      HalfT::Store<N - 2>(ptr + 2, HalfT(_mm256_extracti128_si256(v.raw, 1))); // AVX2
      HalfT::Store<2>(ptr, HalfT(_mm256_castsi256_si128(v.raw))); // AVX
    } else {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.raw); // AVX2
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
#if MOCHI_ARCH_X64_AVX512
    _mm256_mask_storeu_epi64(ptr, x64_simd::kLaneMasksS8[n], v.raw); // AVX512VL
#else
    // With AVX2, this is faster than masked store for a predictable value of n.
    // It is much slower for a random value of n.
    switch (n) { // clang-format off
      case 1: Store<1>(ptr, v); break;
      case 2: Store<2>(ptr, v); break;
      case 3: Store<3>(ptr, v); break;
      case 4: Store<4>(ptr, v); break;
      MOCHI_UNLIKELY default: break;
    } // clang-format on
#endif
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
#if MOCHI_ARCH_X64_AVX512
    auto const mask = static_cast<__mmask8>(ToMask(condition));
    _mm256_mask_compressstoreu_epi64(ptr, mask, values.raw); // AVX512VL
    return _mm_popcnt_u32(mask);
#else
    auto mask = ToMask(condition);
    // Load 8 bytes from the table, then zero-exend to get the shuffle pattern.
    auto const* tableRow =
        reinterpret_cast<__m128i const*>(x64_simd::kStoreSelectedShuffleTableD4[mask]);
    auto pattern = _mm256_cvtepu8_epi32(_mm_loadl_epi64(tableRow));
    auto packed = _mm256_permutevar8x32_epi32(values.raw, pattern);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), packed);
    return _mm_popcnt_u32(mask);
#endif
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    Simd x0, x1, x2;
    Interleave3(a, b, c, x0, x1, x2);
    constexpr int kCount0 = Clamp(kTupleCount * 3 - 0, 0, 4);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 4, 0, 4);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 8, 0, 4);
    Simd::Store<kCount0>(ptr, x0);
    if constexpr (kCount1 > 0) {
      Simd::Store<kCount1>(ptr + 4, x1);
    }
    if constexpr (kCount2 > 0) {
      Simd::Store<kCount2>(ptr + 8, x2);
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
    _mm256_mask_storeu_epi64(ptr, static_cast<__mmask8>(bits), x0.raw); // AVX512VL
    _mm256_mask_storeu_epi64(ptr + (count > 1 ? 4 : 0), static_cast<__mmask8>(bits >> 4), x1.raw);
    _mm256_mask_storeu_epi64(ptr + (count > 2 ? 8 : 0), static_cast<__mmask8>(bits >> 8), x2.raw);
#else
    // Scalar i is stored if i < count * 3. Comparing against scalar indices avoids clamped
    // counts, which clang may turn back into branches. A zero mask stores nothing.
    auto const n = _mm256_set1_epi64x(count * 3);
    auto const mask0 = _mm256_cmpgt_epi64(n, _mm256_setr_epi64x(0, 1, 2, 3)); // AVX2
    auto const mask1 = _mm256_cmpgt_epi64(n, _mm256_setr_epi64x(4, 5, 6, 7));
    auto const mask2 = _mm256_cmpgt_epi64(n, _mm256_setr_epi64x(8, 9, 10, 11));
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr), mask0, x0.raw); // AVX2
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr + (count > 1 ? 4 : 0)), mask1, x1.raw);
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr + (count > 2 ? 8 : 0)), mask2, x2.raw);
#endif
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
    return ToMask(Equal(*this, rhs)) == 0xF; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return ToMask(NotEqual(*this, rhs)) != 0; // Any values not equal
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
#if MOCHI_ARCH_X64_AVX512
    return _mm256_mullo_epi64(raw, rhs.raw); // AVX512VL
#else
    return Simd{
        Get<0>(*this) * Get<0>(rhs),
        Get<1>(*this) * Get<1>(rhs),
        Get<2>(*this) * Get<2>(rhs),
        Get<3>(*this) * Get<3>(rhs)};
#endif
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
    static_assert(kShift >= 0 && kShift < 64, "Shift amount out-of-range");
    if constexpr (kShift == 0) {
      return a;
    } else {
#if MOCHI_ARCH_X64_AVX512
      return _mm256_srai_epi64(a.raw, kShift); // AVX512VL
#else
      auto shifted = _mm256_srli_epi64(a.raw, kShift); // AVX2
      auto signMask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a.raw); // AVX2
      auto signFill = _mm256_slli_epi64(signMask, 64 - kShift); // AVX2
      return _mm256_or_si256(shifted, signFill); // AVX2
#endif
    }
  }

 private:
  // One bit per lane, from the lane's sign bit.
  [[nodiscard]] static MOCHI_FORCE_INLINE int ToMask(Simd a) {
#if MOCHI_ARCH_X64_AVX512
    return _mm256_movepi64_mask(a.raw); // AVX512DQ, AVX512VL
#else
    return _mm256_movemask_pd(_mm256_castsi256_pd(a.raw)); // AVX, AVX
#endif
  }

  // Splits 12 consecutive scalars {a, b, c} into 3 vectors of every third scalar.
  MOCHI_FORCE_INLINE static void
  Deinterleave3(__m256d a, __m256d b, __m256d c, Simd& out0, Simd& out1, Simd& out2) {
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

  // Inverse of Deinterleave3: out0, out1, out2 are the 12 interleaved scalars in memory order.
  MOCHI_FORCE_INLINE static void
  Interleave3(Simd a, Simd b, Simd c, Simd& out0, Simd& out1, Simd& out2) {
    // a = [0,3,6,9], b = [1,4,7,10], c = [2,5,8,11]
    auto a_ = _mm256_castsi256_pd(a.raw);
    auto b_ = _mm256_castsi256_pd(b.raw);
    auto c_ = _mm256_castsi256_pd(c.raw);
    auto d = _mm256_shuffle_pd(b_, b_, 0b0101); // [4,1,10,7]
    auto e = _mm256_blend_pd(a_, c_, 0b0101); // [2,3,8,9]
    e = _mm256_permute2f128_pd(e, e, 0x01); // [8,9,2,3]
    auto f = _mm256_blend_pd(a_, d, 0b1010); // [0,1,6,7]
    auto g = _mm256_blend_pd(d, c_, 0b1010); // [4,5,10,11]
    out0.raw = _mm256_castpd_si256(_mm256_blend_pd(f, e, 0b1100)); // [0,1,2,3]
    out1.raw = _mm256_castpd_si256(_mm256_blend_pd(g, f, 0b1100)); // [4,5,6,7]
    out2.raw = _mm256_castpd_si256(_mm256_blend_pd(e, g, 0b1100)); // [8,9,10,11]
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
