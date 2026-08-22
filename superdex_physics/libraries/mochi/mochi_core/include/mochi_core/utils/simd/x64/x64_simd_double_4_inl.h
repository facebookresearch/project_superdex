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
  Simd<double, 4>
*/
template <>
class Simd<double, 4> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(double, 4, __m256d);
  Simd(double a, double b, double c = 0.0, double d = 0.0)
      : raw(_mm256_set_pd(d, c, b, a)) {} // AVX
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm256_set1_pd(a)) {} // AVX

  // Joint two Vec2d into a single Vec4d
  Simd(Simd<double, 2> const& low, Simd<double, 2> const& high)
      : raw(_mm256_set_m128d(high.raw, low.raw)) {} // AVX

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE double Get(Simd v) {
    static_assert(i >= 0 && i < 4, "Index out of range");
    if constexpr (i == 0) {
      return _mm256_cvtsd_f64(v.raw); // AVX
    } else if constexpr (i == 1) {
      return _mm256_cvtsd_f64(_mm256_shuffle_pd(v.raw, v.raw, 0x01)); // AVX, AVX
    } else if constexpr (i == 2) {
      return _mm256_cvtsd_f64(_mm256_permute2f128_pd(v.raw, v.raw, 0x01)); // AVX, AVX
    } else if constexpr (i == 3) {
      auto tmp = _mm256_permute2f128_pd(v.raw, v.raw, 0x01); // AVX
      return _mm256_cvtsd_f64(_mm256_shuffle_pd(tmp, tmp, 0x01)); // AVX, AVX
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE double Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return v.raw.m256d_f64[i];
#else
    switch (i) { // clang-format off
      case 0: return Get<0>(v);
      case 1: return Get<1>(v);
      case 2: return Get<2>(v);
      case 3: return Get<3>(v);
      MOCHI_UNLIKELY default: return 0.0;
    } // clang-format on
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    auto result = v;
    result.raw.m256d_f64[i] = value;
    return result;
#else
    static constexpr __m256i kMasks[] = {
        {-1LL, 0LL, 0LL, 0LL}, {0LL, -1LL, 0LL, 0LL}, {0LL, 0LL, -1LL, 0LL}, {0LL, 0LL, 0LL, -1LL}};
    return _mm256_blendv_pd(v.raw, _mm256_set1_pd(value), _mm256_castsi256_pd(kMasks[i])); // AVX
#endif
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, Scalar value) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return Set(v, i, value);
  }

  template <int iHalf>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<double, 2> GetHalf(Simd a) {
    static_assert(iHalf == 0 || iHalf == 1);
    return _mm256_extractf128_pd(a.raw, iHalf); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd AsPoint(Simd a) {
    // Replace the 3rd component with an integer that has the same bits as 1.0.
    auto araw = _mm256_castpd_si256(a.raw); // AVX
    auto v = _mm256_insert_epi64(araw, 0x3FF0000000000000LL, 3); // AVX
    return _mm256_castsi256_pd(v); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd AsDirection(Simd a) {
    // Replace the 3rd component with an integer that has the same bits as 0.0.
    auto araw = _mm256_castpd_si256(a.raw); // AVX
    auto v = _mm256_insert_epi64(araw, 0, 3); // AVX
    return _mm256_castsi256_pd(v); // AVX
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
      return _mm256_blend_pd(a.raw, b.raw, x | (y << 1) | (z << 2) | (w << 3)); // SSE4.1
    }
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

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    int mask = GetMSBitMask(v); // One bit for each byte in the vector
    if constexpr (N == kSize) {
      return mask != 0;
    } else {
      int constexpr kNumBits = N * sizeof(Scalar);
      auto constexpr kMayBeSet = (1UL << kNumBits) - 1;
      return (mask & kMayBeSet) != 0;
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Scalar const* p) {
    return _mm256_broadcast_sd(p); // AVX
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Simd v) {
    return Shuffle<i, i, i, i>(v);
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= 4);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{*ptr, 0.0};
    } else if constexpr (N == 2) {
      __m256i mask = _mm256_set_epi64x(0, 0, -1, -1); // AVX
      return _mm256_maskload_pd(ptr, mask); // AVX
    } else if constexpr (N == 3) {
      __m256i mask = _mm256_set_epi64x(0, -1, -1, -1); // AVX
      return _mm256_maskload_pd(ptr, mask); // AVX
    } else {
      return _mm256_loadu_pd(ptr); // AVX
    }
  }

#if MOCHI_COMPILER_MSVC
  // MSVC defines __m128i as a union. Byte arrays are required to initialize it this way.
  // clang-format off
  static constexpr __m256i kLoadMasks[] = {
      { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {-1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
  // clang-format on
#else
  // GCC and Clang define __m256i as 'long long' with special attributes.
  static constexpr __m256i kLoadMasks[] = {
      {0LL, 0LL, 0LL, 0LL},
      {-1LL, 0LL, 0LL, 0LL},
      {-1LL, -1LL, 0LL, 0LL},
      {-1LL, -1LL, -1LL, 0LL},
      {-1LL, -1LL, -1LL, -1LL}};
#endif

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    return _mm256_maskload_pd(ptr, kLoadMasks[n]); // AVX
  }

  [[nodiscard]] static Simd LoadIndexed(Scalar const* ptr, Simd<int, 4> const& indices) {
    return _mm256_i32gather_pd(ptr, indices.raw, sizeof(double)); // AVX2
  }

  [[nodiscard]] static Simd LoadIndexed(Scalar const* ptr, Simd<int64_t, 4> const& indices) {
    return _mm256_i64gather_pd(ptr, indices.raw, sizeof(double)); // AVX2
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

    auto d = _mm256_blend_pd(a, b, 0b0100); // [0,_,6,3]
    d = _mm256_blend_pd(d, c, 0b0010); // [0,9,6,3]
    auto e = _mm256_permute2f128_pd(d, d, 0x01); // [6,3,0,9]
    out0.raw = _mm256_blend_pd(d, e, 0b1010); // [0,3,6,9]

    d = _mm256_blend_pd(a, b, 0b1001); // [4,1,_,7]
    d = _mm256_blend_pd(d, c, 0b0100); // [4,1,10,7]
    out1.raw = _mm256_shuffle_pd(d, d, 0b0101); // [1,4,7,10]

    d = _mm256_blend_pd(a, b, 0b0010); // [_,5,2,_]
    d = _mm256_blend_pd(d, c, 0b1001); // [8,5,2,11]
    e = _mm256_permute2f128_pd(d, d, 0x01); // [2,11,8,5]
    out2.raw = _mm256_blend_pd(d, e, 0b0101); // [2,5,8,11]
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd SetBasisVector() {
    static_assert(i >= 0 && i <= 3, "Invalid component index");
    auto zero = _mm256_setzero_si256(); // AVX
    auto v = _mm256_insert_epi64(zero, 0x3FF0000000000000LL, i); // AVX
    return _mm256_castsi256_pd(v); // AVX
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N < kSize) {
      // About 3X faster than a masked store on AMD. About the same on Intel.
      memcpy(ptr, &v, sizeof(Scalar) * N);
    } else {
      _mm256_storeu_pd(ptr, v.raw); // AVX
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    // Faster than masked store on AMD.
    switch (n) { // clang-format off
      case 1: Store<1>(ptr, v); break;
      case 2: Store<2>(ptr, v); break;
      case 3: Store<3>(ptr, v); break;
      case 4: Store<4>(ptr, v); break;
      MOCHI_UNLIKELY default: break;
    } // clang-format on
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    auto mask = _mm256_movemask_pd(condition.raw);
    // Load 8 bytes from the table, then zero-exend to get the shuffle pattern.
    auto const* tableRow =
        reinterpret_cast<__m128i const*>(x64_simd::kStoreSelectedShuffleTableD4[mask]);
    __m256i pattern = _mm256_cvtepu8_epi32(_mm_loadl_epi64(tableRow));
    __m256i packed = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(values.raw), pattern);
    _mm256_storeu_pd(ptr, _mm256_castsi256_pd(packed));
    return _mm_popcnt_u32(mask);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    // a = [0,3,6,9], b = [1,4,7,10], c = [2,5,8,11]
    auto d = _mm256_shuffle_pd(b.raw, b.raw, 0b0101); // [4,1,10,7]
    auto e = _mm256_blend_pd(a.raw, c.raw, 0b0101); // [2,3,8,9]
    e = _mm256_permute2f128_pd(e, e, 0x01); // [8,9,2,3]
    auto f = _mm256_blend_pd(a.raw, d, 0b1010); // [0,1,6,7]
    auto g = _mm256_blend_pd(d, c.raw, 0b1010); // [4,5,10,11]
    constexpr int kCount0 = Clamp(kTupleCount * 3 - 0, 0, 4);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 4, 0, 4);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 8, 0, 4);
    Simd::Store<kCount0>(ptr, _mm256_blend_pd(f, e, 0b1100)); // [0,1,2,3]
    if constexpr (kCount1 > 0) {
      Simd::Store<kCount1>(ptr + 4, _mm256_blend_pd(g, f, 0b1100)); // [4,5,6,7]
    }
    if constexpr (kCount2 > 0) {
      Simd::Store<kCount2>(ptr + 8, _mm256_blend_pd(e, g, 0b1100)); // [8,9,10,11]
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm256_blendv_pd(b.raw, a.raw, mask.raw); // AVX
  }

  // return Simd{v[x], v[y], v[z], v[w]}
  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd v) {
    static_assert(x >= 0 && x < 4, "Invalid index");
    static_assert(y >= 0 && y < 4, "Invalid index");
    static_assert(z >= 0 && z < 4, "Invalid index");
    static_assert(w >= 0 && w < 4, "Invalid index");
    if constexpr (x == 0 && y == 1 && z == 2 && w == 3) {
      return v;
    } else {
      return _mm256_permute4x64_pd(v.raw, x | (y << 2) | (z << 4) | (w << 6)); // AVX2
    }
  }

  // return Simd{a[x], a[y], b[z], b[w]}
  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd a, Simd b) {
    static_assert(x >= 0 && x < 4, "Invalid index");
    static_assert(y >= 0 && y < 4, "Invalid index");
    static_assert(z >= 0 && z < 4, "Invalid index");
    static_assert(w >= 0 && w < 4, "Invalid index");
    return Simd{Get<x>(a), Get<y>(a), Get<z>(b), Get<w>(b)};
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sqrt(Simd v) {
    return _mm256_sqrt_pd(v.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpApprox(Simd v) {
    return Simd{1.0} / v;
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpSqrtApprox(Simd v) {
    return Simd{1.0} / Sqrt(v);
  }

  // Broadcast the value -0.0. Use this in bitwise operations to affect just the sign bit.
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd SignBitMask() {
    return _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000LL)); // AVX, AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Abs(Simd v) {
    return _mm256_andnot_pd(SignBitMask().raw, v.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    return _mm256_min_pd(a.raw, b.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    return _mm256_max_pd(a.raw, b.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Floor(Simd a) {
    return _mm256_floor_pd(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FastRound(Simd v) {
    return _mm256_round_pd(v.raw, _MM_FROUND_TO_NEAREST_INT); // AVX
  }

#if MOCHI_ARCH_X64_SVML
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Cos(Simd a) {
    return _mm256_cos_pd(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sin(Simd a) {
    return _mm256_sin_pd(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tan(Simd a) {
    return _mm256_tan_pd(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ACos(Simd a) {
    return _mm256_acos_pd(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ASin(Simd a) {
    return _mm256_asin_pd(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ATan(Simd a) {
    return _mm256_atan_pd(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Exp(Simd a) {
    return _mm256_exp_pd(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Ln(Simd a) {
    return _mm256_log_pd(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tanh(Simd a) {
    return _mm256_tanh_pd(a.raw); // AVX
  }
#endif // MOCHI_ARCH_X64_SVML

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulAdd(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return {_mm256_fmadd_pd(a.raw, b.raw, c.raw)}; // FMA
#else
    return (a * b) + c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulSub(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm256_fmsub_pd(a.raw, b.raw, c.raw); // FMA
#else
    return (a * b) - c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulAdd(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm256_fnmadd_pd(a.raw, b.raw, c.raw); // FMA
#else
    return -(a * b) + c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulSub(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm256_fnmsub_pd(a.raw, b.raw, c.raw); // FMA
#else
    return -(a * b) - c;
#endif
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
      using HalfT = Simd<double, 2>;
      HalfT tmp = GetHalf<0>(a) + GetHalf<1>(a);
      return HalfT::Get<0>(tmp) + Get<1>(a); // (a[0] + a[2]) + a[1]
    } else if constexpr (N == 4) {
      // PERF NOTE: An alternate implementation could be: Get<0>(a) + Get<1>(a) + Get<2>(a) +
      // Get<3>(a). In comparison, this implementation saves two instructions, while maintaining
      // identical throughput and latency, when benchmarked on an AMD CPU.
      using HalfT = Simd<double, 2>;
      HalfT tmp = GetHalf<0>(a) + GetHalf<1>(a);
      return HalfT::Get<0>(tmp) + HalfT::Get<1>(tmp); // (a[0] + a[2]) + (a[1] + a[3]);
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HProd(Simd a) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    alignas(alignof(Simd)) Scalar buf[4];
    Store(buf, a);
    if constexpr (N == 2) {
      return buf[0] * buf[1];
    } else if constexpr (N == 3) {
      return buf[0] * buf[1] * buf[2];
    } else if constexpr (N == 4) {
      return buf[0] * buf[1] * buf[2] * buf[3];
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Dot(Simd a, Simd b) {
    static_assert(N >= 2 && N <= 4, "Unsupported N");
    return Simd{HSum<N>(a * b)};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return _mm256_cmp_pd(this->raw, rhs.raw, _CMP_LT_OQ); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return _mm256_cmp_pd(this->raw, rhs.raw, _CMP_GT_OQ); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return _mm256_cmp_pd(this->raw, rhs.raw, _CMP_LE_OQ); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return _mm256_cmp_pd(this->raw, rhs.raw, _CMP_GE_OQ); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return _mm256_cmp_pd(a.raw, b.raw, _CMP_EQ_OQ); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return _mm256_cmp_pd(a.raw, b.raw, _CMP_NEQ_UQ); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm256_setzero_pd(); // AVX
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
    auto ones = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)); // AVX, AVX
    return _mm256_xor_pd(raw, ones); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return _mm256_xor_pd(raw, SignBitMask().raw); // AVX, AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm256_add_pd(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm256_sub_pd(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return _mm256_mul_pd(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return _mm256_div_pd(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm256_and_pd(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm256_or_pd(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm256_xor_pd(raw, rhs.raw); // AVX
  }

 private:
  // Integer mask with the most significant bit of each byte in the vector
  [[nodiscard]] static MOCHI_FORCE_INLINE int GetMSBitMask(Simd a) {
    return _mm256_movemask_epi8(_mm256_castpd_si256(a.raw)); // AVX2, AVX
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
