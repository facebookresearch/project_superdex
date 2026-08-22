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
  Simd<double, 2>
*/
template <>
class Simd<double, 2> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(double, 2, __m128d);
  Simd(double a, double b) : raw(_mm_set_pd(b, a)) {} // SSE2

  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm_set_pd1(a)) {} // SSE2

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE double Get(Simd v) {
    static_assert(i >= 0 && i < 2, "Index out of range");
    if constexpr (i == 0) {
      return _mm_cvtsd_f64(v.raw); // AVX
    } else if constexpr (i == 1) {
      return _mm_cvtsd_f64(_mm_shuffle_pd(v.raw, v.raw, 0x01)); // AVX, AVX
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE double Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < 2, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return v.raw.m128d_f64[i];
#else
    switch (i) { // clang-format off
                case 0: return Get<0>(v);
                case 1: return Get<1>(v);
                MOCHI_UNLIKELY default: return 0.0;
            } // clang-format on
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < 2, "Index out of range");
#if MOCHI_COMPILER_MSVC
    auto result = v;
    result.raw.m128d_f64[i] = value;
    return result;
#else
    static constexpr __m128i kMasks[] = {{-1LL, 0LL}, {0LL, -1LL}};
    return _mm_blendv_pd(v.raw, _mm_set_pd1(value), _mm_castsi128_pd(kMasks[i])); // SSE4.1
#endif
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, Scalar value) {
    static_assert(i >= 0 && i < 2, "Index out of range");
    if constexpr (i == 0) {
      return _mm_shuffle_pd(_mm_set1_pd(value), v.raw, 0x02); // SSE2
    } else {
      return _mm_shuffle_pd(v.raw, _mm_set1_pd(value), 0x02); // SSE2
    }
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

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    int mask = GetMSBitMask(v); // One bit for each byte in the vector
    if constexpr (N == kSize) {
      return mask != 0;
    } else {
      return (mask & 0x000000FF) != 0;
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
      return _mm_blend_pd(a.raw, b.raw, x | (y << 1)); // SSE4.1
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Scalar const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Simd v) {
    return Shuffle<i, i>(v);
  }

  [[nodiscard]] static Simd LoadIndexed(Scalar const* ptr, Simd<int64_t, 2> const& indices) {
    return _mm_i64gather_pd(ptr, indices.raw, sizeof(double)); // AVX2
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{*ptr, 0.0};
    } else {
      return _mm_loadu_pd(ptr); // SSE2
    }
  }

  static_assert(sizeof(long long) == 8);

#if MOCHI_COMPILER_MSVC
  // MSVC defines __m128i as a union. Byte arrays are required to initialize it this way.
  // clang-format off
  static constexpr __m128i kLoadMasks[] = {
      { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {-1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
  // clang-format on
#else
  // GCC and Clang define __m128i as 'long long' with special attributes.
  static constexpr __m128i kLoadMasks[] = {{0LL, 0LL}, {-1LL, 0LL}, {-1LL, -1LL}};
#endif

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    return _mm_maskload_pd(ptr, kLoadMasks[n]); // AVX
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 2, 0, 2);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 4, 0, 2);
    auto a = Simd::Load<2>(ptr).raw; // [0,1]
    auto b = Simd::Load<kCount1>(kCount1 == 0 ? ptr : ptr + 2).raw; // [2,3]
    auto c = Simd::Load<kCount2>(kCount2 == 0 ? ptr : ptr + 4).raw; // [4,5]
    out0.raw = _mm_shuffle_pd(a, b, 0b0010); // [0,3]
    out1.raw = _mm_shuffle_pd(a, c, 0b0001); // [1,4]
    out2.raw = _mm_shuffle_pd(b, c, 0b0010); // [2,5]
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N < kSize) {
      // About 3X faster than a masked store on AMD. About the same on Intel.
      memcpy(ptr, &v, sizeof(Scalar) * N);
    } else {
      _mm_storeu_pd(ptr, v.raw); // SSE2
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    // Faster than masked store on AMD.
    switch (n) { // clang-format off
                case 1: Store<1>(ptr, v); break;
                case 2: Store<2>(ptr, v); break;
                MOCHI_UNLIKELY default: break;
            } // clang-format on
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    auto mask = _mm_movemask_pd(condition.raw);
    auto swapped = _mm_shuffle_pd(values.raw, values.raw, 1); // swap halves
    auto blendMask =
        _mm_castsi128_pd(_mm_set1_epi32((mask & 1) - 1)); // swap first bit of mask is zero
    _mm_storeu_pd(ptr, _mm_blendv_pd(values.raw, swapped, blendMask));
    return _mm_popcnt_u32(mask);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    // a = [0,3], b = [1,4], c = [2,5]
    Simd::Store<2>(ptr, _mm_shuffle_pd(a.raw, b.raw, 0b00)); // [0,1]
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 2, 0, 2);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 4, 0, 2);
    if constexpr (kCount1 > 0) {
      Simd::Store<kCount1>(ptr + 2, _mm_shuffle_pd(c.raw, a.raw, 0b10)); // [2,3]
    }
    if constexpr (kCount2 > 0) {
      Simd::Store<kCount2>(ptr + 4, _mm_shuffle_pd(b.raw, c.raw, 0b11)); // [4,5]
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm_blendv_pd(b.raw, a.raw, mask.raw); // SSE4.1
  }

  // return Simd{v[x], v[y]}
  template <int x = 0, int y = 1>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd v) {
    static_assert(x >= 0 && x < 2, "Invalid index");
    static_assert(y >= 0 && y < 2, "Invalid index");
    if constexpr (x == 0 && y == 1) {
      return v;
    } else {
      return _mm_shuffle_pd(v.raw, v.raw, x | (y << 1)); // SSE2
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sqrt(Simd v) {
    return _mm_sqrt_pd(v.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpApprox(Simd v) {
    return Simd{1.0} / v;
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpSqrtApprox(Simd v) {
    return Simd{1.0} / Sqrt(v);
  }

  // Broadcast the value -0.0. Use this in bitwise operations to affect just the sign bit.
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd SignBitMask() {
    return _mm_castsi128_pd(_mm_set1_epi64x(0x8000000000000000LL)); // SSE2, SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Abs(Simd v) {
    return _mm_andnot_pd(SignBitMask().raw, v.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    return _mm_min_pd(a.raw, b.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    return _mm_max_pd(a.raw, b.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Floor(Simd a) {
    return _mm_floor_pd(a.raw); // SSE4.1
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FastRound(Simd v) {
    return _mm_round_pd(v.raw, _MM_FROUND_TO_NEAREST_INT); // SSE4.1
  }

#if MOCHI_ARCH_X64_SVML
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Cos(Simd a) {
    return _mm_cos_pd(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sin(Simd a) {
    return _mm_sin_pd(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tan(Simd a) {
    return _mm_tan_pd(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ACos(Simd a) {
    return _mm_acos_pd(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ASin(Simd a) {
    return _mm_asin_pd(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ATan(Simd a) {
    return _mm_atan_pd(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Exp(Simd a) {
    return _mm_exp_pd(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Ln(Simd a) {
    return _mm_log_pd(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tanh(Simd a) {
    return _mm_tanh_pd(a.raw); // SSE
  }
#endif // MOCHI_ARCH_X64_SVML

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulAdd(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return {_mm_fmadd_pd(a.raw, b.raw, c.raw)}; // FMA
#else
    return (a * b) + c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulSub(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm_fmsub_pd(a.raw, b.raw, c.raw); // FMA
#else
    return (a * b) - c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulAdd(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm_fnmadd_pd(a.raw, b.raw, c.raw); // FMA
#else
    return -(a * b) + c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulSub(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm_fnmsub_pd(a.raw, b.raw, c.raw); // FMA
#else
    return -(a * b) - c;
#endif
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

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HProd(Simd a) {
    static_assert(N == 2, "Unsupported N");
    return Get<0>(Broadcast<0>(a) * Broadcast<1>(a));
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Dot(Simd a, Simd b) {
    static_assert(N == 2, "Unsupported N");
#if MOCHI_COMPILER_CLANG
    return _mm_dp_pd(a.raw, b.raw, -1); // SSE4.1
#else
    return _mm_dp_pd(a.raw, b.raw, 0xFF); // SSE4.1
#endif
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return _mm_cmplt_pd(this->raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return _mm_cmpgt_pd(this->raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return _mm_cmple_pd(this->raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return _mm_cmpge_pd(this->raw, rhs.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return _mm_cmpeq_pd(a.raw, b.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return _mm_cmpneq_pd(a.raw, b.raw); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm_setzero_pd(); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    auto mask = GetMSBitMask(Equal(raw, rhs.raw));
    return mask == 0xFFFF; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    auto mask = GetMSBitMask(NotEqual(raw, rhs.raw));
    return mask != 0; // Any values not equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    // _mm_cmpeq_epi32 appears to be the fastest way to fill an SSE register with ones.
    __m128i dummy{};
    __m128d ones = _mm_castsi128_pd(_mm_cmpeq_epi32(dummy, dummy)); // SSE2, SSE2
    return _mm_xor_pd(raw, ones); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return _mm_xor_pd(raw, SignBitMask().raw); // SSE2, SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm_add_pd(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm_sub_pd(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return _mm_mul_pd(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return _mm_div_pd(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm_and_pd(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm_or_pd(raw, rhs.raw); // SSE2
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm_xor_pd(raw, rhs.raw); // SSE2
  }

 private:
  // Integer mask with with the most significant bit of each byte in the vector
  [[nodiscard]] static MOCHI_FORCE_INLINE int GetMSBitMask(Simd a) {
    return _mm_movemask_epi8(_mm_castpd_si128(a.raw)); // SSE2, SSE2
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
