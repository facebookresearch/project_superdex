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
  Simd<float, 4>
*/
template <>
class Simd<float, 4> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(float, 4, __m128);
  Simd(float a, float b, float c = 0.0f, float d = 0.0f) : raw(_mm_set_ps(d, c, b, a)) {} // SSE
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm_set_ps1(a)) {} // SSE

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE float Get(Simd v) {
    static_assert(i >= 0 && i < 4, "Index out of range");
    if constexpr (i == 0) {
      return _mm_cvtss_f32(v.raw); // SSE
    } else if constexpr (i == 1) {
      return _mm_cvtss_f32(_mm_shuffle_ps(v.raw, v.raw, _MM_SHUFFLE(1, 1, 1, 1))); // SSE, SSE
    } else if constexpr (i == 2) {
      return _mm_cvtss_f32(_mm_shuffle_ps(v.raw, v.raw, _MM_SHUFFLE(2, 2, 2, 2))); // SSE, SSE
    } else if constexpr (i == 3) {
      return _mm_cvtss_f32(_mm_shuffle_ps(v.raw, v.raw, _MM_SHUFFLE(3, 3, 3, 3))); // SSE, SSE
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE float Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return v.raw.m128_f32[i];
#else
    switch (i) { // clang-format off
      case 0: return Get<0>(v);
      case 1: return Get<1>(v);
      case 2: return Get<2>(v);
      case 3: return Get<3>(v);
      MOCHI_UNLIKELY default: return 0.0f;
    } // clang-format on
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    auto result = v;
    result.raw.m128_f32[i] = value;
    return result;
#else
    static constexpr __m128i kMasks[] = {
        // clang-format off
        {static_cast<long long>(0x00000000FFFFFFFFLL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0xFFFFFFFF00000000LL), static_cast<long long>(0x0000000000000000LL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0x00000000FFFFFFFFLL)},
        {static_cast<long long>(0x0000000000000000LL), static_cast<long long>(0xFFFFFFFF00000000LL)}
    }; // clang-format on
    return _mm_blendv_ps(v.raw, _mm_set1_ps(value), _mm_castsi128_ps(kMasks[i])); // SSE4.1
#endif
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, Scalar value) {
    return _mm_insert_ps(v.raw, _mm_set1_ps(value), i << 4); // SSE4.1
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd AsPoint(Simd a) {
    // Replace the 3rd component with an integer that has the same bits as 1.0f.
    auto araw = _mm_castps_si128(a.raw); // SSE2
    auto v = _mm_insert_epi32(araw, 0x3f800000, 3); // SSE4.1
    return _mm_castsi128_ps(v); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd AsDirection(Simd a) {
    // Replace the 3rd component with an integer that has the same bits as 0.0f.
    auto araw = _mm_castps_si128(a.raw); // SSE2
    auto v = _mm_insert_epi32(araw, 0x00000000, 3); // SSE4.1
    return _mm_castsi128_ps(v); // SSE2
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
      return _mm_blend_ps(a.raw, b.raw, x | (y << 1) | (z << 2) | (w << 3)); // SSE4.1
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    int mask = GetMSBitMask(v); // One bit for each byte in the vector
    if constexpr (N == kSize) {
      return mask == 0x0000FFFF;
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
    return _mm_broadcast_ss(p); // AVX
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
      return Simd{*ptr, 0.0f};
    } else if constexpr (N == 2) {
      __m128i mask = _mm_set_epi32(0, 0, -1, -1); // SSE2
      return _mm_maskload_ps(ptr, mask); // AVX
    } else if constexpr (N == 3) {
      __m128i mask = _mm_set_epi32(0, -1, -1, -1); // SSE2
      return _mm_maskload_ps(ptr, mask); // AVX
    } else if constexpr (N == 4) {
      return _mm_loadu_ps(ptr); // SSE
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    return _mm_maskload_ps(ptr, x64_simd::kLoadMasksS4[n]); // AVX
  }

  [[nodiscard]] static Simd LoadIndexed(Scalar const* ptr, Simd<int, 4> const& indices) {
    return _mm_i32gather_ps(ptr, indices.raw, sizeof(float)); // AVX2
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

    auto t0 = _mm_blend_ps(a, b, 0b0100); // [0,_,6,3]
    auto t1 = _mm_blend_ps(t0, c, 0b0010); // [0,9,6,3]
    out0 = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(1, 2, 3, 0)); // [0,3,6,9]

    t0 = _mm_blend_ps(a, b, 0b1001); // [4,1,_,7]
    t1 = _mm_blend_ps(t0, c, 0b0100); // [4,1,10,7]
    out1 = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 3, 0, 1)); // [1,4,7,10]

    t0 = _mm_blend_ps(a, b, 0b0010); // [_,5,2,_]
    t1 = _mm_blend_ps(c, t0, 0b0110); // [8,5,2,11]
    out2 = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(3, 0, 1, 2)); // [2,5,8,11]
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd SetBasisVector() {
    static_assert(i >= 0 && i <= 3, "Invalid component index");
    auto zero = _mm_setzero_si128(); // SSE2
    auto v = _mm_insert_epi32(zero, 0x3f800000, i); // SSE4.1
    return _mm_castsi128_ps(v); // SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm_blendv_ps(b.raw, a.raw, mask.raw); // SSE4.1
  }

  // return Simd{v[x], v[y], v[z], v[w]}
  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd v) {
    static_assert(
        x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4 && w >= 0 && w < 4, "Invalid index");
    if constexpr (x == 0 && y == 1 && z == 2 && w == 3) {
      return v;
    } else {
      return _mm_shuffle_ps(v.raw, v.raw, x | (y << 2) | (z << 4) | (w << 6)); // SSE
    }
  }
  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd a, Simd b) {
    static_assert(
        x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4 && w >= 0 && w < 4, "Invalid index");
    return _mm_shuffle_ps(a.raw, b.raw, x | (y << 2) | (z << 4) | (w << 6)); // SSE
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N < kSize) {
      // About 3X faster than a masked store on AMD. About the same on Intel.
      memcpy(ptr, &v, sizeof(Scalar) * N);
    } else {
      _mm_storeu_ps(ptr, v.raw); // SSE
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
    auto mask = _mm_movemask_ps(condition.raw);
    // Load the shuffle pattern from a lookup table.
    auto const* tableRow =
        reinterpret_cast<__m128i const*>(x64_simd::kStoreSelectedShuffleTableS4[mask]);
    auto pattern = _mm_load_si128(tableRow);
    auto packed = _mm_permutevar_ps(values.raw, pattern);
    _mm_storeu_ps(ptr, packed);
    return _mm_popcnt_u32(mask);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    // a = [0,3,6,9], b = [1,4,7,10], c = [2,5,8,11]
    auto d = _mm_shuffle_ps(a.raw, a.raw, _MM_SHUFFLE(1, 2, 3, 0)); // [0,9,6,3]
    auto e = _mm_shuffle_ps(b.raw, b.raw, _MM_SHUFFLE(2, 3, 0, 1)); // [4,1,10,7]
    auto f = _mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(3, 0, 1, 2)); // [8,5,2,11]
    constexpr int kCount0 = Clamp(kTupleCount * 3 - 0, 0, 4);
    constexpr int kCount1 = Clamp(kTupleCount * 3 - 4, 0, 4);
    constexpr int kCount2 = Clamp(kTupleCount * 3 - 8, 0, 4);
    Simd::Store<kCount0>(ptr, _mm_blend_ps(_mm_blend_ps(d, e, 0b0010), f, 0b0100)); // [0,1,2,3]
    if constexpr (kCount1 > 0) {
      Simd::Store<kCount1>(
          ptr + 4, _mm_blend_ps(_mm_blend_ps(d, e, 0b1001), f, 0b0010)); // [4,5,6,7]
    }
    if constexpr (kCount2 > 0) {
      Simd::Store<kCount2>(
          ptr + 8, _mm_blend_ps(_mm_blend_ps(d, e, 0b0100), f, 0b1001)); // [8,9,10,11]
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sqrt(Simd v) {
    return _mm_sqrt_ps(v.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpApprox(Simd v) {
    return _mm_rcp_ps(v.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpSqrtApprox(Simd v) {
    return _mm_rsqrt_ps(v.raw); // SSE
  }

  // Broadcast the value -0.0. Use this in bitwise operations to affect just the sign bit.
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd SignBitMask() {
    return _mm_castsi128_ps(_mm_set1_epi32(0x80000000)); // SSE2, SSE2
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Abs(Simd v) {
    return _mm_andnot_ps(SignBitMask().raw, v.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    return _mm_min_ps(a.raw, b.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    return _mm_max_ps(a.raw, b.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Floor(Simd a) {
    return _mm_floor_ps(a.raw); // SSE4.1
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FastRound(Simd v) {
    return _mm_round_ps(v.raw, _MM_FROUND_TO_NEAREST_INT); // SSE4.1
  }

#if MOCHI_ARCH_X64_SVML
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Cos(Simd a) {
    return _mm_cos_ps(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sin(Simd a) {
    return _mm_sin_ps(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tan(Simd a) {
    return _mm_tan_ps(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ACos(Simd a) {
    return _mm_acos_ps(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ASin(Simd a) {
    return _mm_asin_ps(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ATan(Simd a) {
    return _mm_atan_ps(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Exp(Simd a) {
    return _mm_exp_ps(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Ln(Simd a) {
    return _mm_log_ps(a.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tanh(Simd a) {
    return _mm_tanh_ps(a.raw); // SSE
  }
#endif // MOCHI_ARCH_X64_SVML

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulAdd(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return {_mm_fmadd_ps(a.raw, b.raw, c.raw)}; // FMA
#else
    return (a * b) + c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulSub(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm_fmsub_ps(a.raw, b.raw, c.raw); // FMA
#else
    return (a * b) - c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulAdd(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm_fnmadd_ps(a.raw, b.raw, c.raw); // FMA
#else
    return -(a * b) + c;
#endif
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulSub(Simd a, Simd b, Simd c) {
#if MOCHI_ARCH_X64_FMA
    return _mm_fnmsub_ps(a.raw, b.raw, c.raw); // FMA
#else
    return -(a * b) - c;
#endif
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
      return Get<0>(Min(tmp, Simd::Shuffle<2, 3, 0, 1>(tmp)));
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
    // Terms are added in the same order as Simd<double, 4>::HSum<N> for consistency.
    if constexpr (N == 2) {
      return Get<0>(a) + Get<1>(a); // a[0] + a[1]
    } else if constexpr (N == 3) {
      return (Get<0>(a) + Get<2>(a)) + Get<1>(a); // (a[0] + a[2]) + a[1]
    } else {
      // PERF NOTE: Alternatively _mm_dp_ps could be used to compute the dot product with
      // Simd{1.0f}. In comparison, this implementation takes 2 extra instructions, but it had ~50%
      // higher throughput and ~45% lower latency, when benchmarked on an AMD CPU.
      return (Get<0>(a) + Get<2>(a)) + (Get<1>(a) + Get<3>(a)); // (a[0] + a[2]) + (a[1] + a[3])
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
    if constexpr (N == 2) {
      return _mm_dp_ps(a.raw, b.raw, 0x3F); // SSE4.1
    } else if constexpr (N == 3) {
      return _mm_dp_ps(a.raw, b.raw, 0x7F); // SSE4.1
    } else if constexpr (N == 4) {
#if MOCHI_COMPILER_CLANG
      return _mm_dp_ps(a.raw, b.raw, -1); // SSE4.1
#else
      return _mm_dp_ps(a.raw, b.raw, 0xFF); // SSE4.1
#endif
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return _mm_cmplt_ps(this->raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return _mm_cmpgt_ps(this->raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return _mm_cmple_ps(this->raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return _mm_cmpge_ps(this->raw, rhs.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return _mm_cmpeq_ps(a.raw, b.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return _mm_cmpneq_ps(a.raw, b.raw); // SSE
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm_setzero_ps(); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    auto mask = GetMSBitMask(Equal(*this, rhs));
    return mask == 0x0000FFFF; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    auto mask = GetMSBitMask(NotEqual(*this, rhs));
    return mask != 0; // All values equal
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    // _mm_cmpeq_epi32 appears to be the fastest way to fill an SSE register with ones.
    __m128i dummy{};
    __m128 ones = _mm_castsi128_ps(_mm_cmpeq_epi32(dummy, dummy)); // SSE2, SSE2
    return _mm_xor_ps(raw, ones); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return _mm_xor_ps(raw, SignBitMask().raw); // SSE, SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm_add_ps(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm_sub_ps(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return _mm_mul_ps(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return _mm_div_ps(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm_and_ps(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm_or_ps(raw, rhs.raw); // SSE
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm_xor_ps(raw, rhs.raw); // SSE
  }

 private:
  // Integer mask with the most significant bit of each byte in the vector
  [[nodiscard]] static MOCHI_FORCE_INLINE int GetMSBitMask(Simd a) {
    return _mm_movemask_epi8(_mm_castps_si128(a.raw)); // SSE2, SSE2
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX2
