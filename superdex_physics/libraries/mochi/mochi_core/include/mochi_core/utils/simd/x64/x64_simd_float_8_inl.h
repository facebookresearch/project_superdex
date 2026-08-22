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
  Simd<float, 8>
*/
template <>
class Simd<float, 8> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(float, 8, __m256);
  Simd(
      float a,
      float b,
      float c = 0.0f,
      float d = 0.0f,
      float e = 0.0f,
      float f = 0.0f,
      float g = 0.0f,
      float h = 0.0f)
      : raw(_mm256_set_ps(h, g, f, e, d, c, b, a)) {} // AVX

  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm256_set1_ps(a)) {} // AVX

  // Joint two Vec4f into a single Vec8f
  Simd(Simd<float, 4> const& low, Simd<float, 4> const& high)
      : raw(_mm256_set_m128(high.raw, low.raw)) {} // AVX

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE float Get(Simd v) {
    static_assert(i >= 0 && i < 8, "Index out of range");
    if constexpr (i == 0) {
      return _mm256_cvtss_f32(v.raw); // AVX
    } else if constexpr (i == 1) {
      return _mm256_cvtss_f32(_mm256_shuffle_ps(v.raw, v.raw, 0x01)); // AVX, AVX
    } else if constexpr (i == 2) {
      return _mm256_cvtss_f32(_mm256_shuffle_ps(v.raw, v.raw, 0x02)); // AVX, AVX
    } else if constexpr (i == 3) {
      return _mm256_cvtss_f32(_mm256_shuffle_ps(v.raw, v.raw, 0x03)); // AVX, AVX
    } else if constexpr (i == 4) {
      return _mm256_cvtss_f32(_mm256_permute2f128_ps(v.raw, v.raw, 0x01)); // AVX, AVX
    } else if constexpr (i == 5) {
      auto tmp = _mm256_permute2f128_ps(v.raw, v.raw, 0x01); // AVX
      return _mm256_cvtss_f32(_mm256_shuffle_ps(tmp, tmp, 0x01)); // AVX, AVX
    } else if constexpr (i == 6) {
      auto tmp = _mm256_permute2f128_ps(v.raw, v.raw, 0x01); // AVX
      return _mm256_cvtss_f32(_mm256_shuffle_ps(tmp, tmp, 0x02)); // AVX, AVX
    } else if constexpr (i == 7) {
      auto tmp = _mm256_permute2f128_ps(v.raw, v.raw, 0x01); // AVX
      return _mm256_cvtss_f32(_mm256_shuffle_ps(tmp, tmp, 0x03)); // AVX, AVX
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE float Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return v.raw.m256_f32[i];
#else
    switch (i) { // clang-format off
      case 0: return Get<0>(v);
      case 1: return Get<1>(v);
      case 2: return Get<2>(v);
      case 3: return Get<3>(v);
      case 4: return Get<4>(v);
      case 5: return Get<5>(v);
      case 6: return Get<6>(v);
      case 7: return Get<7>(v);
      MOCHI_UNLIKELY default: return 0;
    } // clang-format on
#endif
  }

  template <int iHalf>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<float, 4> GetHalf(Simd a) {
    static_assert(iHalf == 0 || iHalf == 1);
    return _mm256_extractf128_ps(a.raw, iHalf); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    auto result = v;
    result.raw.m256_f32[i] = value;
    return result;
#else
    static constexpr __m256i kMasks[] = {
        // clang-format off
        {static_cast<long long>(0x00000000FFFFFFFFLL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0xFFFFFFFF00000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x00000000FFFFFFFFLL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0xFFFFFFFF00000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x00000000FFFFFFFFLL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0xFFFFFFFF00000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x00000000FFFFFFFFLL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0xFFFFFFFF00000000LL)}
    }; // clang-format on
    return _mm256_blendv_ps(v.raw, _mm256_set1_ps(value), _mm256_castsi256_ps(kMasks[i])); // AVX
#endif
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
    return Set(v, i, value);
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
    return _mm256_broadcast_ss(p); // AVX
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Simd v) {
    return Simd{Get<i>(v)}; // TODO: There is probably a faster way
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= 8);
    if constexpr (N == 0) {
      return Simd::Zero();
    } else if constexpr (N == 1) {
      return Simd{*ptr, 0.0f};
    } else if constexpr (N == 2) {
      __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1); // AVX
      return _mm256_maskload_ps(ptr, mask); // AVX2
    } else if constexpr (N == 3) {
      __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1); // AVX
      return _mm256_maskload_ps(ptr, mask); // AVX2
    } else if constexpr (N == 4) {
      __m256i mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1); // AVX
      return _mm256_maskload_ps(ptr, mask); // AVX2
    } else if constexpr (N == 5) {
      __m256i mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1); // AVX
      return _mm256_maskload_ps(ptr, mask); // AVX2
    } else if constexpr (N == 6) {
      __m256i mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1); // AVX
      return _mm256_maskload_ps(ptr, mask); // AVX2
    } else if constexpr (N == 7) {
      __m256i mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1); // AVX
      return _mm256_maskload_ps(ptr, mask); // AVX2
    } else if constexpr (N == 8) {
      return _mm256_loadu_ps(ptr); // AVX
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    return _mm256_maskload_ps(ptr, x64_simd::kLoadMasksS8[n]); // AVX
  }

  [[nodiscard]] static Simd LoadIndexed(Scalar const* ptr, Simd<int, 8> const& indices) {
    return _mm256_i32gather_ps(ptr, indices.raw, sizeof(float)); // AVX2
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N < kSize) {
      // About 3X faster than a masked store on AMD. About the same on Intel.
      memcpy(ptr, &v, sizeof(Scalar) * N);
    } else {
      _mm256_storeu_ps(ptr, v.raw); // AVX
    }
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

    auto d = _mm256_blend_ps(a, b, 0b10010010); // [0,9,_,3,  12,_,6,15]
    auto e = _mm256_blend_ps(d, c, 0b00100100); // [0,9,18,3, 12,21,6,15]
    auto f = _mm256_permute2f128_ps(e, e, 0x01); // [12,21,6,15,  0,9,18,3]
    auto g = _mm256_blend_ps(e, f, 0b01000100); // [0,9,6,3, 12,21,18,15]
    out0.raw = _mm256_shuffle_ps(g, g, _MM_SHUFFLE(1, 2, 3, 0)); // [0,3,6,9, 12,15,18,21]

    d = _mm256_blend_ps(a, b, 0b00100100); // [_,1,10,_,  4,13,_,7]
    e = _mm256_blend_ps(d, c, 0b01001001); // [16,1,10,19  4,13,22,7]
    f = _mm256_permute2f128_ps(e, e, 0x01); // [4,13,22,7  16,1,10,19]
    g = _mm256_blend_ps(e, f, 0b10011001); // [4,1,10,7,  16,13,22,19]
    out1.raw = _mm256_shuffle_ps(g, g, _MM_SHUFFLE(2, 3, 0, 1)); // [1,4,7,10,  13,16,19,22]

    d = _mm256_blend_ps(a, b, 0b01001001); // [8,_,2,11,  _,5,14,_]
    e = _mm256_blend_ps(d, c, 0b10010010); // [8,17,2,11,  20,5,14,23]
    f = _mm256_permute2f128_ps(e, e, 0x01); // [20,5,14,23,  8,17,2,11]
    g = _mm256_blend_ps(e, f, 0b00100010); // [8,5,2,11,  20,17,14,23]
    out2.raw = _mm256_shuffle_ps(g, g, _MM_SHUFFLE(3, 0, 1, 2)); // [2,5,8,11,  14,17,20,23]
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
      case 5: Store<5>(ptr, v); break;
      case 6: Store<6>(ptr, v); break;
      case 7: Store<7>(ptr, v); break;
      case 8: Store<8>(ptr, v); break;
      MOCHI_UNLIKELY default: break;
    }
    // clang-format on
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    auto mask = _mm256_movemask_ps(condition.raw);
    // Load 8 bytes from the table, then zero-exend to get the shuffle pattern.
    auto const* tableRow =
        reinterpret_cast<__m128i const*>(x64_simd::kStoreSelectedShuffleTableS8[mask]);
    auto pattern = _mm256_cvtepu8_epi32(_mm_loadl_epi64(tableRow));
    auto packed = _mm256_permutevar8x32_ps(values.raw, pattern);
    _mm256_storeu_ps(ptr, packed);
    return _mm_popcnt_u32(mask);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    // a = [0,3,6,9, 12,15,18,21]
    // b = [1,4,7,10, 13,16,19,22]
    // c = [2,5,8,11, 14,17,20,23]
    auto d = _mm256_shuffle_ps(a.raw, a.raw, _MM_SHUFFLE(1, 2, 3, 0)); // [0,9,6,3, 12,21,18,15]
    auto e = _mm256_shuffle_ps(b.raw, b.raw, _MM_SHUFFLE(2, 3, 0, 1)); // [4,1,10,7, 16,13,22,19]
    auto f = _mm256_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(3, 0, 1, 2)); // [8,5,2,11, 20,17,14,23]
    auto g = _mm256_blend_ps(d, e, 0b00100010); // [0,1,_,3, 12,13,_,15]
    g = _mm256_blend_ps(g, f, 0b01000100); // [0,1,2,3, 12,13,14,15]
    auto h = _mm256_blend_ps(d, e, 0b10011001); // [4,_,6,7, 16,_,18,19]
    h = _mm256_blend_ps(h, f, 0b00100010); // [4,5,6,7, 16,17,18,19]
    h = _mm256_permute2f128_ps(h, h, 0x01); // [16,17,18,19, 4,5,6,7]
    auto i = _mm256_blend_ps(d, e, 0b01000100); // [_,9,10,_, _,21,22,_]
    i = _mm256_blend_ps(i, f, 0b10011001); // [8,9,10,11, 20,21,22,23]
    d = _mm256_blend_ps(g, h, 0b11110000); // [0,1,2,3, 4,5,6,7]
    e = _mm256_blend_ps(i, g, 0b11110000); // [8,9,10,11, 12,13,14,15]
    f = _mm256_blend_ps(h, i, 0b11110000); // [16,17,18,19, 20,21,22,23]
    constexpr int kCount0 = Clamp(kTupleCount * 3 - 0, 0, 8);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 8, 0, 8);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 16, 0, 8);
    Simd::Store<kCount0>(ptr, d);
    if constexpr (kCount1 > 0) {
      Simd::Store<kCount1>(ptr + 8, e);
    }
    if constexpr (kCount2 > 0) {
      Simd::Store<kCount2>(ptr + 16, f);
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm256_blendv_ps(b.raw, a.raw, mask.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sqrt(Simd v) {
    return _mm256_sqrt_ps(v.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpApprox(Simd v) {
    return _mm256_rcp_ps(v.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpSqrtApprox(Simd v) {
    return _mm256_rsqrt_ps(v.raw); // AVX
  }

  // Broadcast the value -0.0. Use this in bitwise operations to affect just the sign bit.
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd SignBitMask() {
    return _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)); // AVX, AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Abs(Simd v) {
    return _mm256_andnot_ps(SignBitMask().raw, v.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    return _mm256_min_ps(a.raw, b.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    return _mm256_max_ps(a.raw, b.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Floor(Simd a) {
    return _mm256_floor_ps(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FastRound(Simd v) {
    return _mm256_round_ps(v.raw, _MM_FROUND_TO_NEAREST_INT); // AVX
  }

#if MOCHI_ARCH_X64_SVML
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Cos(Simd a) {
    return _mm256_cos_ps(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sin(Simd a) {
    return _mm256_sin_ps(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tan(Simd a) {
    return _mm256_tan_ps(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ACos(Simd a) {
    return _mm256_acos_ps(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ASin(Simd a) {
    return _mm256_asin_ps(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ATan(Simd a) {
    return _mm256_atan_ps(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Exp(Simd a) {
    return _mm256_exp_ps(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Ln(Simd a) {
    return _mm256_log_ps(a.raw); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tanh(Simd a) {
    return _mm256_tanh_ps(a.raw); // AVX
  }
#endif // MOCHI_ARCH_X64_SVML

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulAdd(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm256_fmadd_ps(a.raw, b.raw, c.raw); // FMA
#else
    return (a * b) + c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulSub(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm256_fmsub_ps(a.raw, b.raw, c.raw); // FMA
#else
    return (a * b) - c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulAdd(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm256_fnmadd_ps(a.raw, b.raw, c.raw); // FMA
#else
    return -(a * b) + c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulSub(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm256_fnmsub_ps(a.raw, b.raw, c.raw); // FMA
#else
    return -(a * b) - c;
#endif
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
        // Set the hi values we don't want to compare to infinity
        auto inf = HalfT{std::numeric_limits<Scalar>::infinity()};
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
        // Set the hi values we don't want to compare to -infinity
        auto inf = HalfT{std::numeric_limits<Scalar>::infinity()};
        hi = HalfT::Blend<1, N >= 6, N >= 7, 0>(-inf, hi);
      }
      return HalfT::HMax<4>(HalfT::Max(lo, hi));
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HSum(Simd a) {
    static_assert(N >= 2 && N <= 8, "Unsupported N");
    using HalfT = Simd<float, 4>;
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
      // PERF NOTE: Alternatively _mm256_dp_ps could be used to compute the dot product with
      // Simd{1.0f}. In comparison, this implementation takes 2 extra instructions, but it had ~29%
      // higher throughput and ~25% lower latency, when benchmarked on an AMD CPU.
      return HalfT::HSum<4>(GetHalf<0>(a) + GetHalf<1>(a));
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Dot(Simd a, Simd b) {
    static_assert(N == 8, "smaller dot products not yet supported on Vec8f");
#if MOCHI_COMPILER_CLANG
    auto ab_half = _mm256_dp_ps(a.raw, b.raw, -1); // AVX
#else
    auto ab_half = _mm256_dp_ps(a.raw, b.raw, 0xFF); // AVX
#endif
    auto lo_ab = _mm256_extractf128_ps(ab_half, 0); // AVX
    auto hi_ab = _mm256_extractf128_ps(ab_half, 1); // AVX
    auto my_dot = _mm_add_ps(lo_ab, hi_ab); // SSE
    return _mm256_set_m128(my_dot, my_dot); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return _mm256_cmp_ps(this->raw, rhs.raw, _CMP_LT_OQ); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return _mm256_cmp_ps(this->raw, rhs.raw, _CMP_GT_OQ); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return _mm256_cmp_ps(this->raw, rhs.raw, _CMP_LE_OQ); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return _mm256_cmp_ps(this->raw, rhs.raw, _CMP_GE_OQ); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return _mm256_cmp_ps(a.raw, b.raw, _CMP_EQ_OQ); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return _mm256_cmp_ps(a.raw, b.raw, _CMP_NEQ_UQ); // AVX
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm256_setzero_ps(); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    auto mask = GetMSBitMask(Equal(*this, rhs));
    return mask == 0xFFFFFFFF; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    auto mask = GetMSBitMask(NotEqual(*this, rhs));
    return mask != 0; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    auto ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1)); // AVX, AVX
    return _mm256_xor_ps(raw, ones); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return Zero() - *this;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm256_add_ps(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm256_sub_ps(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return _mm256_mul_ps(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return _mm256_div_ps(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm256_and_ps(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm256_or_ps(raw, rhs.raw); // AVX
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm256_xor_ps(raw, rhs.raw); // AVX
  }

 private:
  // Integer mask with with the most significant bit of each byte in the vector
  [[nodiscard]] static MOCHI_FORCE_INLINE int GetMSBitMask(Simd a) {
    return _mm256_movemask_epi8(_mm256_castps_si256(a.raw)); // AVX, AVX
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
