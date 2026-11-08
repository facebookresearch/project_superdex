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

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512

namespace mochi {

/***********************************************************************************************
  Simd<double, 8>
*/
template <>
class Simd<double, 8> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(double, 8, __m512d);
  Simd(
      double a,
      double b,
      double c = 0.0,
      double d = 0.0,
      double e = 0.0,
      double f = 0.0,
      double g = 0.0,
      double h = 0.0)
      : raw(_mm512_set_pd(h, g, f, e, d, c, b, a)) {} // AVX512F

  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm512_set1_pd(a)) {} // AVX512F

  Simd(Simd<double, 4> const& low, Simd<double, 4> const& high)
      : raw(_mm512_insertf64x4(_mm512_castpd256_pd512(low.raw), high.raw, 1)) {} // AVX512F

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE double Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return v[i];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Scalar operator[](int i) const {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return raw.m512d_f64[i];
#else
    return raw[i];
#endif
  }

  template <int iHalf>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<double, 4> GetHalf(Simd a) {
    static_assert(iHalf == 0 || iHalf == 1);
    if constexpr (iHalf == 0) {
      return _mm512_castpd512_pd256(a.raw); // AVX512F
    } else {
      return _mm512_extractf64x4_pd(a.raw, 1); // AVX512F
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
    auto const mask = static_cast<__mmask8>(1u << i);
    return _mm512_mask_broadcastsd_pd(v.raw, mask, _mm_set_sd(value));
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, Scalar value) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    constexpr auto kMask = static_cast<__mmask8>(1u << i);
    return _mm512_mask_broadcastsd_pd(v.raw, kMask, _mm_set_sd(value));
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto const mask = ToMask(v);
    if constexpr (N == kSize) {
      return _kortestc_mask8_u8(mask, mask) != 0;
    } else {
      constexpr auto kLanes = LaneMask<N>();
      return (mask & kLanes) == kLanes;
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto const mask = ToMask(v);
    if constexpr (N == kSize) {
      return _kortestz_mask8_u8(mask, mask) == 0;
    } else {
      return (mask & LaneMask<N>()) != 0;
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Scalar const* p) {
    return _mm512_set1_pd(*p); // AVX512F
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    if constexpr (i == 0) {
      return _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v.raw));
    } else {
      constexpr int kLane = i % 2;
      constexpr int kGroup = i / 2;
      auto const group =
          _mm512_shuffle_f64x2(v.raw, v.raw, _MM_SHUFFLE(kGroup, kGroup, kGroup, kGroup));
      return _mm512_permute_pd(group, kLane == 0 ? 0x00 : 0xFF);
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Zero();
    } else if constexpr (N == 1) {
      return _mm512_zextpd128_pd512(_mm_load_sd(ptr));
    } else if constexpr (N == 2) {
      return _mm512_zextpd128_pd512(_mm_loadu_pd(ptr));
    } else if constexpr (N == 3) {
      return _mm512_zextpd256_pd512(_mm256_maskz_loadu_pd(LaneMask<N>(), ptr));
    } else if constexpr (N == 4) {
      return _mm512_zextpd256_pd512(_mm256_loadu_pd(ptr));
    } else if constexpr (N < kSize) {
      return _mm512_maskz_loadu_pd(LaneMask<N>(), ptr); // AVX512F
    } else {
      return _mm512_loadu_pd(ptr); // AVX512F
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    return _mm512_maskz_loadu_pd(LaneMask(n), ptr); // AVX512F
  }

  [[nodiscard]] static Simd LoadIndexed(Scalar const* ptr, Simd<int, 8> const& indices) {
    return _mm512_i32gather_pd(indices.raw, ptr, sizeof(double)); // AVX512F
  }

  [[nodiscard]] static Simd LoadIndexed(Scalar const* ptr, Simd<int64_t, 8> const& indices) {
    return _mm512_i64gather_pd(indices.raw, ptr, sizeof(double)); // AVX512F
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    if constexpr (kTupleCount == 1) {
      out0 = Load<1>(ptr);
      out1 = Load<1>(ptr + 1);
      out2 = Load<1>(ptr + 2);
      return;
    }
    constexpr int kTotalCount = kTupleCount * 3;
    auto const x0 = Load<Clamp(kTotalCount, 0, kSize)>(ptr).raw;
    if constexpr (kTupleCount <= 5) {
      auto const index0 = LoadTransposeIndices<kTupleCount, 0>();
      auto const index1 = LoadTransposeIndices<kTupleCount, 1>();
      auto const index2 = LoadTransposeIndices<kTupleCount, 2>();
      if constexpr (kTupleCount <= 2) {
        out0.raw = _mm512_permutexvar_pd(index0, x0);
        out1.raw = _mm512_permutexvar_pd(index1, x0);
        out2.raw = _mm512_permutexvar_pd(index2, x0);
      } else {
        auto const x1 = Load<kTotalCount - kSize>(ptr + kSize).raw;
        out0.raw = _mm512_permutex2var_pd(x0, index0, x1);
        out1.raw = _mm512_permutex2var_pd(x0, index1, x1);
        out2.raw = _mm512_permutex2var_pd(x0, index2, x1);
      }
    } else {
      auto const x1 = Load<kSize>(ptr + kSize).raw;
      constexpr int kCount2 = kTotalCount - 2 * kSize;
      auto const x2 = Load<kCount2>(ptr + 2 * kSize).raw;
      auto const index0 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0);
      auto const index1 = _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0);
      auto const index2 = _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0);
      if constexpr (kTupleCount == 6) {
        out0.raw = _mm512_maskz_permutex2var_pd(LaneMask<kTupleCount>(), x0, index0, x1);
        constexpr int kZeroIndex = kSize + kCount2;
        auto const partial1 = _mm512_permutex2var_pd(x0, index1, x1);
        auto const finalIndex1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 8, kZeroIndex, kZeroIndex);
        out1.raw = _mm512_permutex2var_pd(partial1, finalIndex1, x2);
        auto const partial2 = _mm512_permutex2var_pd(x0, index2, x1);
        auto const finalIndex2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 9, kZeroIndex, kZeroIndex);
        out2.raw = _mm512_permutex2var_pd(partial2, finalIndex2, x2);
      } else {
        constexpr int kZeroIndex = kSize + kCount2;
        auto const partial0 = _mm512_permutex2var_pd(x0, index0, x1);
        auto const finalIndex0 = _mm512_setr_epi64(
            0, 1, 2, 3, 4, 5, kTupleCount > 6 ? 10 : kZeroIndex, kTupleCount > 7 ? 13 : kZeroIndex);
        out0.raw = _mm512_permutex2var_pd(partial0, finalIndex0, x2);
        auto const partial1 = _mm512_permutex2var_pd(x0, index1, x1);
        auto const finalIndex1 = _mm512_setr_epi64(
            0, 1, 2, 3, 4, 8, kTupleCount > 6 ? 11 : kZeroIndex, kTupleCount > 7 ? 14 : kZeroIndex);
        out1.raw = _mm512_permutex2var_pd(partial1, finalIndex1, x2);
        auto const partial2 = _mm512_permutex2var_pd(x0, index2, x1);
        auto const finalIndex2 = _mm512_setr_epi64(
            0, 1, 2, 3, 4, 9, kTupleCount > 6 ? 12 : kZeroIndex, kTupleCount > 7 ? 15 : kZeroIndex);
        out2.raw = _mm512_permutex2var_pd(partial2, finalIndex2, x2);
      }
    }
  }

  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2, int count) {
    MOCHI_ASSERT_VERBOSE(count >= 0 && count <= kSize, "Invalid tuple count");
    int const n = count * 3;
    auto const x0 = Load(ptr, mochi::Min(n, kSize)).raw;
    auto const x1 = Load(n > kSize ? ptr + kSize : ptr, Clamp(n - kSize, 0, kSize)).raw;
    auto const x2 = Load(n > 2 * kSize ? ptr + 2 * kSize : ptr, Clamp(n - 2 * kSize, 0, kSize)).raw;
    out0.raw = DeinterleaveComponent<0>(x0, x1, x2);
    out1.raw = DeinterleaveComponent<1>(x0, x1, x2);
    out2.raw = DeinterleaveComponent<2>(x0, x1, x2);
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N == 1) {
      _mm_store_sd(ptr, _mm512_castpd512_pd128(v.raw));
    } else if constexpr (N == 2) {
      _mm_storeu_pd(ptr, _mm512_castpd512_pd128(v.raw));
    } else if constexpr (N == 3) {
      _mm256_mask_storeu_pd(
          ptr, static_cast<__mmask8>((uint32_t{1} << N) - 1), _mm512_castpd512_pd256(v.raw));
    } else if constexpr (N == 4) {
      _mm256_storeu_pd(ptr, _mm512_castpd512_pd256(v.raw));
    } else if constexpr (N < kSize) {
      _mm512_mask_storeu_pd(ptr, LaneMask(N), v.raw); // AVX512F
    } else {
      _mm512_storeu_pd(ptr, v.raw); // AVX512F
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    _mm512_mask_storeu_pd(ptr, LaneMask(n), v.raw); // AVX512F
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    __mmask8 const mask = ToMask(condition);
    _mm512_mask_compressstoreu_pd(ptr, mask, values.raw); // AVX512F
    return _mm_popcnt_u32(mask);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    if constexpr (kTupleCount == 1) {
      Store<1>(ptr, a);
      Store<1>(ptr + 1, b);
      Store<1>(ptr + 2, c);
      return;
    }
    constexpr int kTotalCount = kTupleCount * 3;
    Simd x0, x1, x2;
    Interleave3(a, b, c, x0, x1, x2);
    Store<Clamp(kTotalCount, 0, kSize)>(ptr, x0);
    if constexpr (kTotalCount > kSize) {
      Store<Clamp(kTotalCount - kSize, 0, kSize)>(ptr + kSize, x1);
    }
    if constexpr (kTotalCount > 2 * kSize) {
      Store<kTotalCount - 2 * kSize>(ptr + 2 * kSize, x2);
    }
  }

  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c, int count) {
    MOCHI_ASSERT_VERBOSE(count >= 0 && count <= kSize, "Invalid tuple count");
    Simd x0, x1, x2;
    Interleave3(a, b, c, x0, x1, x2);
    // Bit i selects scalar i. Masks come from shifts, not clamped counts, which clang may turn
    // back into branches. A zero mask stores nothing.
    uint32_t const bits = (1u << (count * 3)) - 1;
    _mm512_mask_storeu_pd(ptr, static_cast<__mmask8>(bits), x0.raw); // AVX512F
    _mm512_mask_storeu_pd(ptr + (count > 2 ? 8 : 0), static_cast<__mmask8>(bits >> 8), x1.raw);
    _mm512_mask_storeu_pd(ptr + (count > 5 ? 16 : 0), static_cast<__mmask8>(bits >> 16), x2.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm512_mask_blend_pd(ToMask(mask), b.raw, a.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sqrt(Simd v) {
    return _mm512_sqrt_pd(v.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpApprox(Simd v) {
    return _mm512_rcp14_pd(v.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpSqrtApprox(Simd v) {
    return _mm512_rsqrt14_pd(v.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd SignBitMask() {
    return _mm512_castsi512_pd(
        _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Abs(Simd v) {
    return _mm512_abs_pd(v.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    return _mm512_min_pd(a.raw, b.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    return _mm512_max_pd(a.raw, b.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Floor(Simd a) {
    return _mm512_floor_pd(a.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FastRound(Simd v) {
    return _mm512_roundscale_pd(v.raw, _MM_FROUND_TO_NEAREST_INT); // AVX512F
  }

#if MOCHI_ARCH_X64_SVML
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Cos(Simd a) {
    return _mm512_cos_pd(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sin(Simd a) {
    return _mm512_sin_pd(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tan(Simd a) {
    return _mm512_tan_pd(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ACos(Simd a) {
    return _mm512_acos_pd(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ASin(Simd a) {
    return _mm512_asin_pd(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ATan(Simd a) {
    return _mm512_atan_pd(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Exp(Simd a) {
    return _mm512_exp_pd(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Ln(Simd a) {
    return _mm512_log_pd(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tanh(Simd a) {
    return _mm512_tanh_pd(a.raw);
  }
#endif // MOCHI_ARCH_X64_SVML

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulAdd(Simd a, Simd b, Simd c) {
    return _mm512_fmadd_pd(a.raw, b.raw, c.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulSub(Simd a, Simd b, Simd c) {
    return _mm512_fmsub_pd(a.raw, b.raw, c.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulAdd(Simd a, Simd b, Simd c) {
    return _mm512_fnmadd_pd(a.raw, b.raw, c.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulSub(Simd a, Simd b, Simd c) {
    return _mm512_fnmsub_pd(a.raw, b.raw, c.raw); // AVX512F
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMin(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 4>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 4) {
      return HalfT::template HMin<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 5) {
        return mochi::Min(HalfT::template HMin<4>(lo), HalfT::template Get<0>(hi));
      } else if constexpr (N == 8) {
        return HalfT::template HMin<4>(HalfT::Min(lo, hi));
      } else {
        return _mm512_mask_reduce_min_pd(LaneMask<N>(), a.raw);
      }
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMax(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 4>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 4) {
      return HalfT::template HMax<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 5) {
        return mochi::Max(HalfT::template HMax<4>(lo), HalfT::template Get<0>(hi));
      } else if constexpr (N == 8) {
        return HalfT::template HMax<4>(HalfT::Max(lo, hi));
      } else {
        return _mm512_mask_reduce_max_pd(LaneMask<N>(), a.raw);
      }
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HSum(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 4>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 4) {
      return HalfT::template HSum<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 5) {
        return HalfT::template HSum<4>(lo) + HalfT::template Get<0>(hi);
      } else if constexpr (N == kSize) {
        return HalfT::template HSum<4>(lo + hi);
      } else {
        return _mm512_mask_reduce_add_pd(LaneMask<N>(), a.raw);
      }
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<double, 4> HSumEach(Simd a, Simd b, Simd c, Simd d) {
    // In each 128-bit lane: ab = {a[0] + a[1], b[0] + b[1]}, and cd likewise.
    auto const ab = _mm512_add_pd(
        _mm512_unpacklo_pd(a.raw, b.raw), _mm512_unpackhi_pd(a.raw, b.raw)); // AVX512F
    auto const cd = _mm512_add_pd(
        _mm512_unpacklo_pd(c.raw, d.raw), _mm512_unpackhi_pd(c.raw, d.raw)); // AVX512F
    // Halves first: the lane-crossing shuffle below is cheaper in 256 bits than in 512.
    auto const ab4 =
        _mm256_add_pd(_mm512_castpd512_pd256(ab), _mm512_extractf64x4_pd(ab, 1)); // AVX512F
    auto const cd4 =
        _mm256_add_pd(_mm512_castpd512_pd256(cd), _mm512_extractf64x4_pd(cd, 1)); // AVX512F
    // {ab4[0], ab4[1], cd4[2], cd4[3]} + {ab4[2], ab4[3], cd4[0], cd4[1]}.
    return _mm256_add_pd(
        _mm256_blend_pd(ab4, cd4, 0b1100), _mm256_permute2f128_pd(ab4, cd4, 0x21)); // AVX
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HProd(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 4>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 4) {
      return HalfT::template HProd<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 5) {
        return HalfT::template HProd<4>(lo) * HalfT::template Get<0>(hi);
      } else if constexpr (N == kSize) {
        return HalfT::template HProd<4>(lo * hi);
      } else {
        return _mm512_mask_reduce_mul_pd(LaneMask<N>(), a.raw);
      }
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Dot(Simd a, Simd b) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    return Simd{HSum<N>(a * b)};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return FromMask(_mm512_cmp_pd_mask(raw, rhs.raw, _CMP_LT_OQ)); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return FromMask(_mm512_cmp_pd_mask(raw, rhs.raw, _CMP_GT_OQ)); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return FromMask(_mm512_cmp_pd_mask(raw, rhs.raw, _CMP_LE_OQ)); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return FromMask(_mm512_cmp_pd_mask(raw, rhs.raw, _CMP_GE_OQ)); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return FromMask(_mm512_cmp_pd_mask(a.raw, b.raw, _CMP_EQ_OQ)); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return FromMask(_mm512_cmp_pd_mask(a.raw, b.raw, _CMP_NEQ_UQ)); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm512_setzero_pd(); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    return _mm512_cmp_pd_mask(raw, rhs.raw, _CMP_EQ_OQ) == static_cast<__mmask8>(0xFFu);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return _mm512_cmp_pd_mask(raw, rhs.raw, _CMP_NEQ_UQ) != 0;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return _mm512_castsi512_pd(
        _mm512_xor_si512(_mm512_castpd_si512(raw), _mm512_set1_epi64(-1))); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return _mm512_xor_pd(raw, SignBitMask().raw); // AVX512DQ
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm512_add_pd(raw, rhs.raw); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm512_sub_pd(raw, rhs.raw); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return _mm512_mul_pd(raw, rhs.raw); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return _mm512_div_pd(raw, rhs.raw); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm512_and_pd(raw, rhs.raw); // AVX512DQ
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm512_or_pd(raw, rhs.raw); // AVX512DQ
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm512_xor_pd(raw, rhs.raw); // AVX512DQ
  }

 private:
  template <int kTupleCount, int kComponent>
  [[nodiscard]] static MOCHI_FORCE_INLINE __m512i LoadTransposeIndices() {
    constexpr int kZeroIndex = kTupleCount * 3;
    return _mm512_setr_epi64(
        kComponent,
        kTupleCount > 1 ? 3 + kComponent : kZeroIndex,
        kTupleCount > 2 ? 6 + kComponent : kZeroIndex,
        kTupleCount > 3 ? 9 + kComponent : kZeroIndex,
        kTupleCount > 4 ? 12 + kComponent : kZeroIndex,
        kTupleCount > 5 ? 15 + kComponent : kZeroIndex,
        kTupleCount > 6 ? 18 + kComponent : kZeroIndex,
        kTupleCount > 7 ? 21 + kComponent : kZeroIndex);
  }

  // Returns every third scalar, starting at kComponent, of the 3 * kSize consecutive scalars
  // {x0, x1, x2}.
  template <int kComponent>
  [[nodiscard]] static MOCHI_FORCE_INLINE __m512d
  DeinterleaveComponent(__m512d x0, __m512d x1, __m512d x2) {
    auto const index = LoadTransposeIndices<kSize, kComponent>();
    // Lanes whose index is at least 2 * kSize read from x2. permutexvar uses the index modulo
    // kSize.
    constexpr auto kLanesFromX2 =
        static_cast<__mmask8>(~LaneMask<(2 * kSize - kComponent + 2) / 3>());
    auto const fromX0X1 = _mm512_permutex2var_pd(x0, index, x1); // AVX512F
    return _mm512_mask_permutexvar_pd(fromX0X1, kLanesFromX2, index, x2); // AVX512F
  }

  // Inverse of DeinterleaveComponent<0..2>: out0, out1, out2 are the 24 interleaved scalars in
  // memory order.
  MOCHI_FORCE_INLINE static void
  Interleave3(Simd a, Simd b, Simd c, Simd& out0, Simd& out1, Simd& out2) {
    auto const ab0 =
        _mm512_permutex2var_pd(a.raw, _mm512_setr_epi64(0, 8, 0, 1, 9, 0, 2, 10), b.raw);
    out0.raw = _mm512_permutex2var_pd(ab0, _mm512_setr_epi64(0, 1, 8, 3, 4, 9, 6, 7), c.raw);
    auto const ab1 =
        _mm512_permutex2var_pd(a.raw, _mm512_setr_epi64(0, 3, 11, 0, 4, 12, 0, 5), b.raw);
    out1.raw = _mm512_permutex2var_pd(ab1, _mm512_setr_epi64(10, 1, 2, 11, 4, 5, 12, 7), c.raw);
    auto const ab2 =
        _mm512_permutex2var_pd(a.raw, _mm512_setr_epi64(13, 0, 6, 14, 0, 7, 15, 0), b.raw);
    out2.raw = _mm512_permutex2var_pd(ab2, _mm512_setr_epi64(0, 13, 2, 3, 14, 5, 6, 15), c.raw);
  }

  // Returns a mask selecting the lowest N lanes.
  template <int N>
  [[nodiscard]] static constexpr __mmask8 LaneMask() {
    static_assert(N >= 0 && N <= kSize);
    return LaneMask(N);
  }

  // Returns a mask selecting the lowest n lanes.
  [[nodiscard]] static constexpr __mmask8 LaneMask(int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid lane count");
    return static_cast<__mmask8>((uint32_t{1} << n) - 1);
  }

  // Converts a canonical logical vector (all-zero or all-one lanes) to a mask.
  [[nodiscard]] static MOCHI_FORCE_INLINE __mmask8 ToMask(Simd a) {
    auto const bits = _mm512_castpd_si512(a.raw);
    auto const mask = _mm512_movepi64_mask(bits); // AVX512DQ
    MOCHI_ASSERT_VERBOSE(
        _mm512_cmpeq_epi64_mask(bits, _mm512_movm_epi64(mask)) == LaneMask<kSize>(),
        "Expected a canonical logical mask");
    return mask;
  }

  // Expands a mask into a canonical logical vector (all-zero or all-one lanes).
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FromMask(__mmask8 mask) {
    return _mm512_castsi512_pd(_mm512_movm_epi64(mask)); // AVX512DQ
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
