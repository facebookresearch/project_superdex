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
  Simd<int, 4>
*/
template <>
class Simd<int, 4> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(int, 4, __m128i);
  Simd(int a, int b, int c = 0, int d = 0) : raw(_mm_set_epi32(d, c, b, a)) {} // SSE2
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm_set1_epi32(a)) {} // SSE2

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE int Get(Simd v) {
    static_assert(i >= 0 && i < 4, "Index out of range");
    if constexpr (i == 0) {
      return _mm_cvtsi128_si32(v.raw); // SSE2
    } else if constexpr (i == 1) {
      return _mm_cvtsi128_si32(_mm_shuffle_epi32(v.raw, _MM_SHUFFLE(1, 1, 1, 1))); // SSE2, SSE2
    } else if constexpr (i == 2) {
      return _mm_cvtsi128_si32(_mm_shuffle_epi32(v.raw, _MM_SHUFFLE(2, 2, 2, 2))); // SSE2, SSE2
    } else if constexpr (i == 3) {
      return _mm_cvtsi128_si32(_mm_shuffle_epi32(v.raw, _MM_SHUFFLE(3, 3, 3, 3))); // SSE2, SSE2
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Scalar operator[](int i) const {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return raw.m128i_i32[i];
#else
    switch (i) { // clang-format off
      case 0: return Get<0>(*this);
      case 1: return Get<1>(*this);
      case 2: return Get<2>(*this);
      case 3: return Get<3>(*this);
      MOCHI_UNLIKELY default: return 0;
    } // clang-format on
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    auto result = v;
    result.raw.m128i_i32[i] = value;
    return result;
#else
    static constexpr __m128i kMasks[] = {
        // clang-format off
        {static_cast<long long>(0x00000000FFFFFFFFLL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0xFFFFFFFF00000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x00000000FFFFFFFFLL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0xFFFFFFFF00000000LL)}
    }; // clang-format on
    return _mm_blendv_epi8(v.raw, _mm_set1_epi32(value), kMasks[i]); // SSE4.1
#endif
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, Scalar value) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    // _mm_insert_epi32 exists, but it requires both i and value to be constexpr.
    // _mm_insert_ps moves the same 4 bytes as if they were floats. The casts are free.
    return _mm_castps_si128(_mm_insert_ps(
        _mm_castsi128_ps(v.raw), _mm_castsi128_ps(_mm_set1_epi32(value)), i << 4)); // SSE4.1
  }

  // Set via 2 int64_t instead of 4 int
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd SetInt64(int64_t a, int64_t b) {
    return _mm_set_epi64x(b, a); // SSE2
  }

  template <int x, int y, int z, int w>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Blend(Simd a, Simd b) {
    static_assert(
        x >= 0 && x < 2 && y >= 0 && y < 2 && z >= 0 && z < 2 && w >= 0 && w < 2,
        "invalid blend index");
    if constexpr (x == 0 && y == 0 && z == 0 && w == 0) {
      return a;
    } else if constexpr (x == 1 && y == 1 && z == 1 && w == 1) {
      return b;
    } else {
      return _mm_blend_epi32(a.raw, b.raw, x | (y << 1) | (z << 2) | (w << 3)); // AVX2
    }
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
    return Shuffle<i, i, i, i>(v);
  }

  template <int N = 4>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMin(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return Get<0>(Min(a, Broadcast<1>(a)));
    } else if constexpr (N == 3) {
      return Get<0>(Min(Min(a, Broadcast<1>(a)), Broadcast<2>(a)));
    } else {
      auto tmp = Min(a, Shuffle<1, 2, 3, 0>(a));
      return Get<0>(Min(tmp, Shuffle<2, 3, 0, 1>(tmp)));
    }
  }

  template <int N = 4>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMax(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return Get<0>(Max(a, Broadcast<1>(a)));
    } else if constexpr (N == 3) {
      return Get<0>(Max(Max(a, Broadcast<1>(a)), Broadcast<2>(a)));
    } else {
      auto tmp = Max(a, Shuffle<1, 2, 3, 0>(a));
      return Get<0>(Max(tmp, Shuffle<2, 3, 0, 1>(tmp)));
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HSum(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    if constexpr (N == 2) {
      return Get<0>(a) + Get<1>(a);
    } else if constexpr (N == 3) {
      return Get<0>(a) + Get<1>(a) + Get<2>(a);
    } else {
      return Get<0>(a) + Get<1>(a) + Get<2>(a) + Get<3>(a);
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd HSumEach(Simd a, Simd b, Simd c, Simd d) {
    // ab = {a[0] + a[2], b[0] + b[2], a[1] + a[3], b[1] + b[3]}, and cd likewise.
    auto const ab =
        _mm_add_epi32(_mm_unpacklo_epi32(a.raw, b.raw), _mm_unpackhi_epi32(a.raw, b.raw)); // SSE2
    auto const cd =
        _mm_add_epi32(_mm_unpacklo_epi32(c.raw, d.raw), _mm_unpackhi_epi32(c.raw, d.raw)); // SSE2
    // {ab[0], ab[1], cd[2], cd[3]} + {ab[2], ab[3], cd[0], cd[1]}.
    return _mm_add_epi32(
        _mm_blend_epi16(ab, cd, 0xF0), _mm_alignr_epi8(cd, ab, 8)); // SSE4.1, SSSE3
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= 4);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{*ptr, 0};
    } else if constexpr (N == 2) {
      return _mm_loadl_epi64(reinterpret_cast<__m128i const*>(ptr)); // SSE2
    } else if constexpr (N == 3) {
      __m128i mask = _mm_set_epi32(0, -1, -1, -1); // SSE2
      return _mm_maskload_epi32(ptr, mask); // AVX2
    } else if constexpr (N == 4) {
      return _mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)); // SSE
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
#if MOCHI_ARCH_X64_AVX512
    return _mm_maskz_loadu_epi32(x64_simd::kLaneMasksS8[n], ptr); // AVX512VL
#else
    // Use the same masks as Simd<float, 4>
    return _mm_maskload_epi32(ptr, x64_simd::kLoadMasksS4[n]); // AVX2
#endif
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    constexpr int kCount0 = Clamp(kTupleCount * 3 - 0, 0, 4);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 4, 0, 4);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 8, 0, 4);
    auto a = Simd::Load<kCount0>(ptr).raw; // [0,1,2,3]
    auto b = Simd::Load<kCount1>(kCount1 == 0 ? ptr : ptr + 4).raw; // [4,5,6,7]
    auto c = Simd::Load<kCount2>(kCount2 == 0 ? ptr : ptr + 8).raw; // [8,9,10,11]
    Deinterleave3(a, b, c, out0, out1, out2);
  }

  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2, int count) {
    MOCHI_ASSERT_VERBOSE(count >= 0 && count <= kSize, "Invalid tuple count");
    int const n = count * 3;
    auto a = Simd::Load(ptr, mochi::Min(n, 4)).raw; // [0,1,2,3]
    auto b = Simd::Load(n > 4 ? ptr + 4 : ptr, Clamp(n - 4, 0, 4)).raw; // [4,5,6,7]
    auto c = Simd::Load(n > 8 ? ptr + 8 : ptr, Clamp(n - 8, 0, 4)).raw; // [8,9,10,11]
    Deinterleave3(a, b, c, out0, out1, out2);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    return _mm_min_epi32(a.raw, b.raw); // SSE4.1
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    return _mm_max_epi32(a.raw, b.raw); // SSE4.1
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm_blendv_epi8(b.raw, a.raw, mask.raw); // SSE4.1
  }

  // return Simd{v[x], v[y], v[z], v[w]}
  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd v) {
    static_assert(
        x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4 && w >= 0 && w < 4, "Invalid index");
    if constexpr (x == 0 && y == 1 && z == 2 && w == 3) {
      return v;
    } else {
      return _mm_shuffle_epi32(v.raw, x | (y << 2) | (z << 4) | (w << 6)); // SSE2
    }
  }

  // return Simd{a[x], a[y], b[z], b[w]}
  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd a, Simd b) {
    static_assert(
        x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4 && w >= 0 && w < 4, "Invalid index");
    return _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(a.raw),
        _mm_castsi128_ps(b.raw),
        x | (y << 2) | (z << 4) | (w << 6))); // SSE2
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
      _mm_storeu_si32(ptr, v.raw); // SSE2
    } else if constexpr (N == 2) {
      _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr), v.raw); // SSE2
    } else if constexpr (N == 3) {
      _mm_storeu_si32(ptr + 2, _mm_unpackhi_epi64(v.raw, v.raw)); // SSE2
      _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr), v.raw); // SSE2
    } else {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.raw); // SSE
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
#if MOCHI_ARCH_X64_AVX512
    _mm_mask_storeu_epi32(ptr, x64_simd::kLaneMasksS8[n], v.raw); // AVX512VL
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
    auto mask = ToMask(condition);
    // Load the shuffle pattern from a lookup table.
    auto const* tableRow =
        reinterpret_cast<__m128i const*>(x64_simd::kStoreSelectedShuffleTableS4[mask]);
    auto pattern = _mm_load_si128(tableRow);
    auto packed = _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(values.raw), pattern));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), packed);
    return _mm_popcnt_u32(mask);
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
    _mm_mask_storeu_epi32(ptr, static_cast<__mmask8>(bits), x0.raw); // AVX512VL
    _mm_mask_storeu_epi32(ptr + (count > 1 ? 4 : 0), static_cast<__mmask8>(bits >> 4), x1.raw);
    _mm_mask_storeu_epi32(ptr + (count > 2 ? 8 : 0), static_cast<__mmask8>(bits >> 8), x2.raw);
#else
    // Scalar i is stored if i < count * 3. Comparing against scalar indices avoids clamped
    // counts, which clang may turn back into branches. A zero mask stores nothing.
    auto const n = _mm_set1_epi32(count * 3);
    auto const mask0 = _mm_cmpgt_epi32(n, _mm_setr_epi32(0, 1, 2, 3)); // SSE2
    auto const mask1 = _mm_cmpgt_epi32(n, _mm_setr_epi32(4, 5, 6, 7));
    auto const mask2 = _mm_cmpgt_epi32(n, _mm_setr_epi32(8, 9, 10, 11));
    _mm_maskstore_epi32(ptr, mask0, x0.raw); // AVX2
    _mm_maskstore_epi32(ptr + (count > 1 ? 4 : 0), mask1, x1.raw);
    _mm_maskstore_epi32(ptr + (count > 2 ? 8 : 0), mask2, x2.raw);
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm_setzero_si128(); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return _mm_cmplt_epi32(this->raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return _mm_cmpgt_epi32(this->raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return ~(*this > rhs); // No native support until AVX512
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return ~(*this < rhs); // No native support until AVX512
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return _mm_cmpeq_epi32(a.raw, b.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return ~Equal(a, b); // // No native support until AVX512
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    return ToMask(Equal(*this, rhs)) == 0xF; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return ToMask(NotEqual(*this, rhs)) != 0; // Any values not equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    // _mm_cmpeq_epi32 appears to be the fastest way to fill an SSE register with ones.
    __m128i ones = _mm_cmpeq_epi32(raw, raw); // SSE2
    return _mm_xor_si128(raw, ones); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return _mm_sub_epi32(_mm_setzero_si128(), raw); // SSE2, SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm_add_epi32(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm_sub_epi32(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return _mm_mullo_epi32(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
#if MOCHI_ARCH_X64_SVML
    return _mm_div_epi32(raw, rhs.raw); // SSE
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
    return _mm_and_si128(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm_or_si128(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm_xor_si128(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<<(int rhs) const {
    return _mm_slli_epi32(raw, rhs); // SSE2
  }

  template <int kShift>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd ShiftRight(Simd a) {
    static_assert(kShift >= 0 && kShift < 32, "Shift amount out-of-range");
    return _mm_srai_epi32(a.raw, kShift); // SSE2
  }

 private:
  // One bit per lane, from the lane's sign bit.
  [[nodiscard]] static MOCHI_FORCE_INLINE int ToMask(Simd a) {
    return _mm_movemask_ps(_mm_castsi128_ps(a.raw)); // SSE, SSE2
  }

  // Splits 12 consecutive scalars {a, b, c} into 3 vectors of every third scalar.
  MOCHI_FORCE_INLINE static void
  Deinterleave3(__m128i a, __m128i b, __m128i c, Simd& out0, Simd& out1, Simd& out2) {
    auto t0 = _mm_blend_epi32(a, b, 0b0100); // [0,_,6,3]
    auto t1 = _mm_blend_epi32(t0, c, 0b0010); // [0,9,6,3]
    out0 = _mm_shuffle_epi32(t1, _MM_SHUFFLE(1, 2, 3, 0)); // [0,3,6,9]

    t0 = _mm_blend_epi32(a, b, 0b1001); // [4,1,_,7]
    t1 = _mm_blend_epi32(t0, c, 0b0100); // [4,1,10,7]
    out1 = _mm_shuffle_epi32(t1, _MM_SHUFFLE(2, 3, 0, 1)); // [1,4,7,10]

    t0 = _mm_blend_epi32(a, b, 0b0010); // [_,5,2,_]
    t1 = _mm_blend_epi32(c, t0, 0b0110); // [8,5,2,11]
    out2 = _mm_shuffle_epi32(t1, _MM_SHUFFLE(3, 0, 1, 2)); // [2,5,8,11]
  }

  // Inverse of Deinterleave3: out0, out1, out2 are the 12 interleaved scalars in memory order.
  MOCHI_FORCE_INLINE static void
  Interleave3(Simd a, Simd b, Simd c, Simd& out0, Simd& out1, Simd& out2) {
    // a = [0,3,6,9], b = [1,4,7,10], c = [2,5,8,11]
    auto d = _mm_shuffle_epi32(a.raw, _MM_SHUFFLE(1, 2, 3, 0)); // [0,9,6,3]
    auto e = _mm_shuffle_epi32(b.raw, _MM_SHUFFLE(2, 3, 0, 1)); // [4,1,10,7]
    auto f = _mm_shuffle_epi32(c.raw, _MM_SHUFFLE(3, 0, 1, 2)); // [8,5,2,11]
    out0 = _mm_blend_epi32(_mm_blend_epi32(d, e, 0b0010), f, 0b0100); // [0,1,2,3]
    out1 = _mm_blend_epi32(_mm_blend_epi32(d, e, 0b1001), f, 0b0010); // [4,5,6,7]
    out2 = _mm_blend_epi32(_mm_blend_epi32(d, e, 0b0100), f, 0b1001); // [8,9,10,11]
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
