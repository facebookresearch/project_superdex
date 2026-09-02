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

#include <mochi_core/utils/debug.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

// Reverse include for intellisense
#include "../simd.h"

namespace mochi {

namespace details {
#if MOCHI_ARCH_X64_SVML
inline constexpr bool kUseSvml = true;
#else
inline constexpr bool kUseSvml = false;
#endif
} // namespace details

/***********************************************************************************************
  Overloaded Operators for Simd<T, N>
*/

// Defines operators like += for Simd types. The right hand side can be any type as long as
// the corresponding binary operator (e.g. operator+) exists.
#define MOCHI_DEFINE_SIMD_OP_EQ(OP_EQ, OP)                                            \
  template <class T, int N, class RHS>                                                \
  MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N>& operator OP_EQ(Simd<T, N>& lhs, RHS rhs) { \
    lhs = lhs OP rhs;                                                                 \
    return lhs;                                                                       \
  }

// Defines binary operators between Simd<T, N> and T (either order).
#define MOCHI_DEFINE_MIXED_SIMD_SCALAR_OP(OP)                            \
  template <class T, int N>                                              \
  MOCHI_ANY MOCHI_FORCE_INLINE auto operator OP(Simd<T, N> lhs, T rhs) { \
    return lhs OP Simd<T, N>{rhs};                                       \
  }                                                                      \
  template <class T, int N>                                              \
  MOCHI_ANY MOCHI_FORCE_INLINE auto operator OP(T lhs, Simd<T, N> rhs) { \
    return Simd<T, N>{lhs} OP rhs;                                       \
  }

MOCHI_DEFINE_SIMD_OP_EQ(+=, +)
MOCHI_DEFINE_SIMD_OP_EQ(-=, -)
MOCHI_DEFINE_SIMD_OP_EQ(*=, *)
MOCHI_DEFINE_SIMD_OP_EQ(/=, /)
MOCHI_DEFINE_SIMD_OP_EQ(&=, &)
MOCHI_DEFINE_SIMD_OP_EQ(|=, |)
MOCHI_DEFINE_SIMD_OP_EQ(^=, ^)
#undef MOCHI_DEFINE_SIMD_OP_EQ

MOCHI_DEFINE_MIXED_SIMD_SCALAR_OP(+);
MOCHI_DEFINE_MIXED_SIMD_SCALAR_OP(-);
MOCHI_DEFINE_MIXED_SIMD_SCALAR_OP(*);
MOCHI_DEFINE_MIXED_SIMD_SCALAR_OP(/);
#undef MOCHI_DEFINE_MIXED_SIMD_SCALAR_OP

namespace details {
// Returns true if every lane is either all-bits-0 (logical false) or all-bits-1 (logical true).
template <class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE bool IsValidLogicalMask(Simd<T, N> a) {
  static_assert(sizeof(T) == sizeof(int) || sizeof(T) == sizeof(int64_t));
  using I = std::conditional_t<sizeof(T) == sizeof(int), int, int64_t>;
  using IVec = Simd<I, N>;
  auto const ia = ReinterpretCast<IVec>(a);
  auto const zero = IVec{};
  auto const ones = ~zero;
  return AllTrue(VEqual(ia, zero) | VEqual(ia, ones));
}
} // namespace details

template <class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> operator&&(Simd<T, N> lhs, Simd<T, N> rhs) {
  MOCHI_ASSERT_VERBOSE(
      mochi::details::IsValidLogicalMask(lhs) && mochi::details::IsValidLogicalMask(rhs),
      "operator&& requires all lanes to have all-bits-0 or all-bits-1");
  return lhs & rhs;
}

template <class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> operator||(Simd<T, N> lhs, Simd<T, N> rhs) {
  MOCHI_ASSERT_VERBOSE(
      mochi::details::IsValidLogicalMask(lhs) && mochi::details::IsValidLogicalMask(rhs),
      "operator|| requires all lanes to have all-bits-0 or all-bits-1");
  return lhs | rhs;
}

/***********************************************************************************************
  Simd Function Definitions
*/

template <class V, MOCHI_CONCEPT_DEF(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Broadcast(typename V::Scalar a) {
  return V{a};
}

// Default implementation for most vector type
template <class V, MOCHI_CONCEPT_DEF(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Broadcast(typename V::Scalar const* p) {
  return V::Broadcast(p);
}

template <int i, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Broadcast(Simd<T, N> v) {
  return Simd<T, N>::template Broadcast<i>(v);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Broadcast(Simd<T, N> v, int i) {
  // TODO: Other implementations may be faster for specific vector sizes, but this covers the bases.
  return Simd<T, N>{Simd<T, N>::Get(v, i)};
}

template <class V, class... MoreBools>
MOCHI_ANY MOCHI_FORCE_INLINE V SimdMask(bool b0, bool b1, MoreBools... bs) {
  static_assert(V::kSize == sizeof...(bs) + 2, "Incorrect number of arguments");
  using I = std::conditional_t<sizeof(typename V::Scalar) == sizeof(int), int, int64_t>;
  using IVec = Simd<I, V::kSize>;
  constexpr I kSimdBool[2] = {0, -1}; // false, true
  return ReinterpretCast<V>(IVec{kSimdBool[b0], kSimdBool[b1], kSimdBool[bs]...});
}

template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V SimdZero() {
  return V::Zero();
}

template <int i, class V>
MOCHI_ANY MOCHI_FORCE_INLINE V SimdBasisVector() {
  return V::template SetBasisVector<i>();
}

template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V SimdBasisVector(int axis) {
  static_assert(V::kSize == 4, "Unsupported SIMD size");
  switch (axis) {
    case 0:
      return V::template SetBasisVector<0>();
    case 1:
      return V::template SetBasisVector<1>();
    case 2:
      return V::template SetBasisVector<2>();
    case 3:
      return V::template SetBasisVector<3>();
    default:
      MOCHI_ASSERT(axis >= 0 && axis <= 3, "Invalid component index");
      return {};
  }
}

template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V ToSimdPoint(V a) {
  return V::AsPoint(a);
}

template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V ToSimdDirection(V a) {
  return V::AsDirection(a);
}

template <class V, MOCHI_CONCEPT_DEF(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Load(typename V::Scalar const* ptr) {
  return V::template Load<V::kSize>(ptr);
}

template <int N, class V, MOCHI_CONCEPT_DEF(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Load(typename V::Scalar const* ptr) {
  return V::template Load<N>(ptr);
}

template <class V, MOCHI_CONCEPT_DEF(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Load(typename V::Scalar const* ptr, int n) {
  return V::Load(ptr, n);
}

template <class V, class I, MOCHI_CONCEPT_DEF(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V
LoadIndexed(typename V::Scalar const* ptr, Simd<I, V::kSize> indices) {
  static_assert(std::is_integral_v<I>, "Requires integral type");
  return V::LoadIndexed(ptr, indices);
}

template <int kTupleCount, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE void
LoadTransposed(T const* ptr, Simd<T, N>& out0, Simd<T, N>& out1, Simd<T, N>& out2) {
  constexpr int kTupleCount_ = (kTupleCount == -1) ? N : kTupleCount;
  Simd<T, N>::template LoadTransposed<kTupleCount_>(ptr, out0, out1, out2);
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE void Store(T* ptr, Simd<T, N> a) {
  constexpr int COUNT_ = (COUNT == -1) ? N : COUNT;
  Simd<T, N>::template Store<COUNT_>(ptr, a);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE void Store(T* ptr, Simd<T, N> a, int count) {
  Simd<T, N>::Store(ptr, a, count);
}

template <class T, int N, class MaskT>
MOCHI_ANY MOCHI_FORCE_INLINE int
StoreSelected(T* ptr, Simd<MaskT, N> condition, Simd<T, N> values) {
  MOCHI_ASSERT_VERBOSE(
      mochi::details::IsValidLogicalMask(condition),
      "Not a valid logical mask. Each lane must be all-bits-0 or all-bits-1.");
  if constexpr (sizeof(MaskT) == 8 && sizeof(T) == 4) { // Example: MaskT = double, T = int
    auto conditionI64 = ReinterpretCast<Simd<int64_t, N>>(condition);
    auto conditionI32 = StaticCast<Simd<int, N>>(conditionI64);
    MOCHI_ASSERT_VERBOSE(
        (conditionI32 == StaticCast<Simd<int, N>>(conditionI64 & int64_t(0x00000000FFFFFFFFLL))),
        "Expected StaticCast from int64_t to int32_t to return the lower 32 bits. This is not guaranteed by the C++ standard for static_cast, "
        "but it is guaranteed by the x64 and ARM implementations. Your new CPU architecture behaves differenty. Therefore, this code will "
        "need to perform a masking operation, or use Simd<uint64_t, N> and Simd<uint32_t, N> (not supported at the time of writing).");
    return Simd<T, N>::StoreSelected(ptr, ReinterpretCast<Simd<T, N>>(conditionI32), values);
  } else {
    return Simd<T, N>::StoreSelected(ptr, ReinterpretCast<Simd<T, N>>(condition), values);
  }
}

template <int kTupleCount, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE void
StoreTransposed(T* ptr, Simd<T, N> a, Simd<T, N> b, Simd<T, N> c) {
  constexpr int kTupleCount_ = (kTupleCount == -1) ? N : kTupleCount;
  Simd<T, N>::template StoreTransposed<kTupleCount_>(ptr, a, b, c);
}

template <class T, int N, class MaskT>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N>
Select(Simd<MaskT, N> conditionMask, Simd<T, N> a, Simd<T, N> b) {
  MOCHI_ASSERT_VERBOSE(
      mochi::details::IsValidLogicalMask(conditionMask),
      "Not a valid logical mask. Each lane must be all-bits-0 or all-bits-1.");
  return Simd<T, N>::Select(ReinterpretCast<Simd<T, N>>(conditionMask), a, b);
}

template <int kShift, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> ShiftRight(Simd<T, N> a) {
  static_assert(
      Simd<T, N>::kIsSupported && std::is_integral_v<T> && std::is_signed_v<T>,
      "ShiftRight requires a supported signed integer Simd type");
  static_assert(kShift >= 0 && kShift < (8 * sizeof(T)), "Shift amount out-of-range");
  if constexpr (kShift == 0) {
    return a;
  } else {
    return Simd<T, N>::template ShiftRight<kShift>(a);
  }
}

template <int x, int y, class T>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, 2> Shuffle(Simd<T, 2> a) {
  return Simd<T, 2>::template Shuffle<x, y>(a);
}

template <int x, int y, int z, int w, class T>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, 4> Shuffle(Simd<T, 4> a) {
  return Simd<T, 4>::template Shuffle<x, y, z, w>(a);
}

template <int x, int y, int z, int w, class T>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, 4> Shuffle(Simd<T, 4> a, Simd<T, 4> b) {
  return Simd<T, 4>::template Shuffle<x, y, z, w>(a, b);
}

template <int x, int y, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Blend(Simd<T, N> a, Simd<T, N> b) {
  return Simd<T, N>::template Blend<x, y>(a, b);
}

template <int x, int y, int z, int w, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Blend(Simd<T, N> a, Simd<T, N> b) {
  return Simd<T, N>::template Blend<x, y, z, w>(a, b);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T Get0(Simd<T, N> v) {
  return Simd<T, N>::template Get<0>(v);
}

template <int i, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T Get(Simd<T, N> v) {
  return Simd<T, N>::template Get<i>(v);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T Get(Simd<T, N> v, int i) {
  return Simd<T, N>::Get(v, i);
}

template <int iHalf, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N / 2> GetHalf(Simd<T, N> a) {
  return Simd<T, N>::template GetHalf<iHalf>(a);
}

template <int i, class T, int N>
[[nodiscard]] Simd<T, N> Set(Simd<T, N> a, T value) {
  return Simd<T, N>::template Set<i>(a, value);
}

template <class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Set(Simd<T, N> a, int i, T value) {
  return Simd<T, N>::Set(a, i, value);
}

template <class V>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE V Sequence() {
  using T = typename V::Scalar;
  static_assert(V::kIsSupported && std::is_integral_v<T>, "Must be a supported integral Simd type");
  if constexpr (V::kIsComposite) {
    return V::Sequence();
  } else {
    alignas(V) T constexpr kSequence[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    static_assert(
        std::size(kSequence) >= V::kSize,
        "Vector size is too large. Size of kSequence must be increased.");
    return V::template Load<V::kSize>(kSequence);
  }
}

template <bool x, bool y, bool z, bool w, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Neg(Simd<T, N> a) {
  static_assert(
      std::is_floating_point_v<T> && (N == 4),
      "This implementation is intended for Vec4f or Vec4d only");
  return Blend<x, y, z, w>(a, -a);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Sqrt(Simd<T, N> a) {
  static_assert(std::is_floating_point_v<T>, "Requires float or double");
  return Simd<T, N>::Sqrt(a);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> RcpApprox(Simd<T, N> a) {
  return Simd<T, N>::RcpApprox(a);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> RcpSqrtApprox(Simd<T, N> a) {
  return Simd<T, N>::RcpSqrtApprox(a);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Abs(Simd<T, N> a) {
  return Simd<T, N>::Abs(a);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Min(Simd<T, N> a, Simd<T, N> b) {
  return Simd<T, N>::Min(a, b);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Max(Simd<T, N> a, Simd<T, N> b) {
  return Simd<T, N>::Max(a, b);
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T HSum(Simd<T, N> a) {
  constexpr int COUNT_ = (COUNT == -1) ? N : COUNT;
  return Simd<T, N>::template HSum<COUNT_>(a);
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T HProd(Simd<T, N> a) {
  constexpr int COUNT_ = (COUNT == -1) ? N : COUNT;
  return Simd<T, N>::template HProd<COUNT_>(a);
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T HMin(Simd<T, N> a) {
  constexpr int COUNT_ = (COUNT == -1) ? N : COUNT;
  return Simd<T, N>::template HMin<COUNT_>(a);
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T HMax(Simd<T, N> a) {
  constexpr int COUNT_ = (COUNT == -1) ? N : COUNT;
  return Simd<T, N>::template HMax<COUNT_>(a);
}

#define MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, FN, inVec) \
  ([&]() {                                              \
    alignas(alignof(Simd<T, N>)) T buf[N];              \
    Store(buf, inVec);                                  \
    for (int i = 0; i < N; ++i) {                       \
      buf[i] = FN(buf[i]);                              \
    }                                                   \
    return Load<Simd<T, N>>(buf);                       \
  }())

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Floor(Simd<T, N> a) {
  return Simd<T, N>::Floor(a);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> FastRound(Simd<T, N> a) {
  return Simd<T, N>::FastRound(a);
}

namespace details {
// We have a custom SIMD implementation of single-precision sine and cosine. These are computed by
// summing the first few terms of the Taylor series. The problem is that precision decreases as
// abs(x) increases. Therefore, the first step is to reduce the range of x. Depending on the
// quadrant, we may return either the sine or the cosine, and we may need to flip the sign. All of
// this can be done quickly with branchless SIMD instructions.
//
// Some details of this implementation were inspired by code provided by ARM Software:
// https://github.com/ARM-software/optimized-routines
//
template <int N>
MOCHI_ANY MOCHI_FORCE_INLINE void
SinCosImpl(Simd<float, N> xf, Simd<float, N>& outSin, Simd<float, N>& outCos, Simd<int, N>& outN) {
  // Convert to double precision
  auto x = StaticCast<Simd<double, N>>(xf);

  // Reduce x to [-pi/4, pi/4] in quadrant n
  auto r = FastRound(x * 0x1.45f306dc9c883p-1); // round(x * (2/pi))
  outN = StaticCast<Simd<int, N>>(r);
  x = x - r * 0x1.921fb54442d18p0; // x - n * (pi/2)

  // Compute Taylor series for both sin and cos
  auto x2 = x * x; // x^2
  auto x3 = x * x2; // x^3
  auto x4 = x2 * x2; // x^4
  auto s1 = 0x1.1107605230bc4p-7 - x2 * 0x1.994eb3774cf24p-13; // 1/5! - (x^2)/7!
  auto c1 = 1.0 - x2 * 0.5;
  auto c2 = -0x1.6c087e89a359dp-10 + x2 * 0x1.99343027bf8c3p-16; // -1/6! + (x^2)/8!
  auto x5 = x3 * x2; // x^5
  auto x6 = x4 * x2; // x^6
  auto s0 = x - x3 * 0x1.555545995a603p-3; // x - (x^3)/3!
  auto c0 = c1 + x4 * 0x1.55553e1068f19p-5; // 1 - (x^2)/2! + (x^4)/4!

  // outSin = x - (x^3)/3! + (x^5)/5! - (x^7)/7!
  outSin = StaticCast<Simd<float, N>>(s0 + x5 * s1);

  // outCos = 1 - (x^2)/2! + (x^4)/4! - (x^6)/6! + (x^8)/8!
  outCos = StaticCast<Simd<float, N>>(c0 + x6 * c2);
}

// Preserve the documented accuracy bound and keep the quadrant index
// representable by falling back for larger inputs.
inline constexpr float kMaxFastSinCosInput = 1e6f;

template <int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool IsFastSinCosInput(Simd<float, N> a) {
  return AllTrue(Abs(a) <= Simd<float, N>{kMaxFastSinCosInput});
}
} // namespace details

template <class T, int N>
inline Simd<T, N> Cos(Simd<T, N> a) {
  // TODO: When C++20 syntax in this header is legal, gate based on whether Simd<T, N>::Cos() is
  // implemented. Same for all other functions with a SIMD memberwise fallback.
  if constexpr (details::kUseSvml || Simd<T, N>::kIsEmulated) {
    // There are x64 intrinsics if SVML extension is available
    return Simd<T, N>::Cos(a);
  } else if constexpr (std::is_same_v<T, float>) {
    if (!details::IsFastSinCosInput(a))
      MOCHI_UNLIKELY {
        return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::cos, a);
      }
    // Reduce x to [-pi/4, pi/4) in quadrant n and compute Taylor series
    Simd<float, N> sin, cos;
    Simd<int, N> n;
    details::SinCosImpl(a, sin, cos, n);
    // Use bitwise ops to select cos for (n == 0) || (n == 2). Else select sin.
    auto result = Select((n & 1) - 1, cos, sin);
    // Then flip the sign if (n == 1) || (n == 2).
    return result ^ ReinterpretCast<Simd<float, N>>((n ^ ShiftRight<1>(n)) << 31);
  } else {
    return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::cos, a); // Fallback
  }
}

template <class T, int N>
inline Simd<T, N> Sin(Simd<T, N> a) {
  if constexpr (details::kUseSvml || Simd<T, N>::kIsEmulated) {
    // There are x64 intrinsics if SVML extension is available
    return Simd<T, N>::Sin(a);
  } else if constexpr (std::is_same_v<T, float>) {
    if (!details::IsFastSinCosInput(a))
      MOCHI_UNLIKELY {
        return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::sin, a);
      }
    // Reduce x to [-pi/4, pi/4) in quadrant n and compute Taylor series
    Simd<T, N> sin, cos;
    Simd<int, N> n;
    details::SinCosImpl(a, sin, cos, n);
    // Use bitwise ops to select sin for (n == 0) || (n == 2). Else select cos.
    auto result = Select((n & 1) - 1, sin, cos);
    // Then flip the sign if (n == 2) || (n == 3).
    return result ^ ReinterpretCast<Simd<T, N>>(ShiftRight<1>(n) << 31);
  } else {
    return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::sin, a);
  }
}

template <class T, int N>
inline std::pair<Simd<T, N>, Simd<T, N>> SinCos(Simd<T, N> a) {
  // Our custom single-precision implementation can efficiently compute sin and cos at the same
  // time. However, we still use call Sin and Cos separately when they are implemented with SVML, so
  // that the results will be exactly the same.
  if constexpr (!MOCHI_ARCH_X64_SVML && std::is_same_v<T, float>) {
    if (!details::IsFastSinCosInput(a))
      MOCHI_UNLIKELY {
        return {Sin(a), Cos(a)};
      }
    Simd<float, N> sin, cos;
    Simd<int, N> n;
    details::SinCosImpl(a, sin, cos, n);
    auto mask = ReinterpretCast<Simd<float, N>>((n & 1) - 1);
    auto nr = ShiftRight<1>(n);
    auto sresult = Select(mask, sin, cos) ^ ReinterpretCast<Simd<float, N>>(nr << 31);
    auto cresult = Select(mask, cos, sin) ^ ReinterpretCast<Simd<float, N>>((n ^ nr) << 31);
    return {sresult, cresult};
  } else {
    return {Sin(a), Cos(a)};
  }
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Tan(Simd<T, N> a) {
  if constexpr (details::kUseSvml || Simd<T, N>::kIsEmulated) {
    // Simd<T, N>::Tan only implemented in this case.
    return Simd<T, N>::Tan(a);
  } else {
    return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::tan, a);
  }
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> ACos(Simd<T, N> a) {
  if constexpr (details::kUseSvml || Simd<T, N>::kIsEmulated) {
    // Simd<T, N>::ACos only implemented in this case.
    return Simd<T, N>::ACos(a);
  } else {
    return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::acos, a);
  }
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> ASin(Simd<T, N> a) {
  if constexpr (details::kUseSvml || Simd<T, N>::kIsEmulated) {
    // Simd<T, N>::ASin only implemented in this case.
    return Simd<T, N>::ASin(a);
  } else {
    return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::asin, a);
  }
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> ATan(Simd<T, N> a) {
  if constexpr (details::kUseSvml || Simd<T, N>::kIsEmulated) {
    // Simd<T, N>::ATan only implemented in this case.
    return Simd<T, N>::ATan(a);
  } else {
    return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::atan, a);
  }
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Exp(Simd<T, N> a) {
  if constexpr (details::kUseSvml || Simd<T, N>::kIsEmulated) {
    // Simd<T, N>::Exp only implemented in this case.
    return Simd<T, N>::Exp(a);
  } else {
    //
    // This code is mimicking the following implementations:
    // Vc (float):
    // Vc (double): https://github.com/VcDevel/Vc/blob/1.4/Vc/common/math.h
    // cephes: https://github.com/jeremybarnes/cephes/blob/master/cmath/exp.c
    // avx_mathfun: https://github.com/reyoung/avx_mathfun/blob/master/avx_mathfun.h
    // and the references therein.
    //
    //--- Treat the case of float and double
    Simd<T, N> const infinity(std::numeric_limits<T>::infinity());
    Simd<T, N> const log2_e(T(1.44269504088896341)); // = ln(e) / ln(2)
    Simd<T, N> const one(T(1.0));
    Simd<T, N> const half(T(0.5));
    Simd<T, N> const zero(T(0.0));
    //
    auto x = a;
    auto n = log2_e * x;
    n += half;
    n = Floor(n);
    if constexpr (std::is_same_v<T const, float const>) {
      Simd<T, N> const C1(T(0.693359375));
      Simd<T, N> const C2(T(-2.121944400547138e-04));
      x -= C1 * n;
      x -= C2 * n;
    } else {
      static_assert(std::is_same_v<T const, double const>);
      Simd<T, N> const C1(T(0.693145751953125));
      Simd<T, N> const C2(T(1.42860682030941723212e-06));
      x -= C1 * n;
      x -= C2 * n;
    }
    //--- Polynomial approximation
    Simd<T, N> y;
    if constexpr (std::is_same_v<T const, float const>) {
      float const P[] = {
          1.9875691500e-04,
          1.3982999507e-03,
          8.3334519073e-03,
          4.1665795894e-02,
          1.6666665459e-01,
          5.0000001201e-01};
      auto z = x * x;
      y = Simd<T, N>(P[0]);
      for (int i = 1; i <= 5; ++i) {
        y *= x;
        y += Simd<T, N>(P[i]);
      }
      y *= z;
      y += x;
      y += one;
      auto imm0 = StaticCast<Simd<int, N>>(n);
      imm0 += Simd<int, N>(127); // 127 <- 0x7f
      imm0 = imm0 << 23;
      x = y * ReinterpretCast<Simd<T, N>>(imm0); // x = y * Exp2(n);
      //
      Simd<T, N> const exp_lo(T(-88.37626226647949));
      auto const underFlow = (a < exp_lo);
      x = Simd<T, N>::Select(underFlow, zero, x);
      //
      Simd<T, N> const exp_hi(T(88.37626226647949));
      auto const overFlow = (a > exp_hi);
      x = Simd<T, N>::Select(overFlow, infinity, x);
      return x;
    }
    //
    // --- Treating the 'double' case
    //
    if constexpr (std::is_same_v<T const, double const>) {
      double const P[] = {
          1.26177193074810590878E-4,
          3.02994407707441961300E-2,
          9.99999999999999999910E-1,
      };
      double const Q[] = {
          3.00198505138664455042E-6,
          2.52448340349684104192E-3,
          2.27265548208155028766E-1,
          2.00000000000000000009E0,
      };
      auto z = x * x;
      Simd<T, N> pz(P[0]);
      for (int i = 1; i <= 2; ++i) {
        pz *= z;
        pz += Simd<T, N>(P[i]);
      }
      pz *= x;
      Simd<T, N> qz(Q[0]);
      for (int i = 1; i <= 3; ++i) {
        qz *= z;
        qz += Simd<T, N>(Q[i]);
      }
      y = pz / (qz - pz);
      y *= Simd<T, N>(T(2.0));
      y += one;
      auto jmm = StaticCast<Simd<int64_t, N>>(n);
      jmm += Simd<int64_t, N>(1023);
      jmm = jmm << 52;
      x = y * ReinterpretCast<Simd<T, N>>(jmm);
      //
      Simd<T, N> const exp_lo(T(-709.0));
      auto const underFlow = (a < exp_lo);
      x = Simd<T, N>::Select(underFlow, zero, x);
      //
      Simd<T, N> const exp_hi(T(709.0));
      auto const overFlow = (a > exp_hi);
      x = Simd<T, N>::Select(overFlow, infinity, x);
      return x;
    }
  }
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Ln(Simd<T, N> a) {
  if constexpr (details::kUseSvml || Simd<T, N>::kIsEmulated) {
    // Simd<T, N>::Ln only implemented in this case.
    return Simd<T, N>::Ln(a);
  } else {
    return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::log, a);
  }
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Tanh(Simd<T, N> a) {
  if constexpr (details::kUseSvml || Simd<T, N>::kIsEmulated) {
    // Simd<T, N>::Tanh only implemented in this case.
    return Simd<T, N>::Tanh(a);
  } else {
    return MOCHI_SIMD_MEMBERWISE_FALLBACK(T, N, std::tanh, a);
  }
}

#undef MOCHI_SIMD_MEMBERWISE_FALLBACK

template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VEqual(V a, V b) {
  return V::Equal(a, b);
}

template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VNotEqual(V a, V b) {
  return V::NotEqual(a, b);
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool Equal(Simd<T, N> a, Simd<T, N> b) {
  if constexpr (COUNT == 1) {
    return Get0(a) == Get0(b);
  } else {
    return AllTrue<COUNT>(VEqual(a, b));
  }
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool NotEqual(Simd<T, N> a, Simd<T, N> b) {
#if MOCHI_PLATFORM_MACOS && MOCHI_ARCH_X64 && MOCHI_OPTIMIZED
  // Work-around for a compiler bug in optimized builds on Intel macOS.
  return !Equal<COUNT>(a, b);
#else
  if constexpr (COUNT == 1) {
    return Get0(a) != Get0(b);
  } else {
    return AnyTrue<COUNT>(VNotEqual(a, b));
  }
#endif
}

template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VNearEqual(V a, V b, V epsilon) {
  return Abs(a - b) <= epsilon;
}

template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VNearZero(V a, V epsilon) {
  return Abs(a) <= epsilon;
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool AllTrue(Simd<T, N> a) {
  constexpr int COUNT_ = (COUNT == -1) ? N : COUNT; // -1 means "all"
  return Simd<T, N>::template AllTrue<COUNT_>(a);
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool AnyTrue(Simd<T, N> a) {
  constexpr int COUNT_ = (COUNT == -1) ? N : COUNT; // -1 means "all"
  return Simd<T, N>::template AnyTrue<COUNT_>(a);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> VIsFinite(Simd<T, N> a) {
  if constexpr (std::is_same_v<T, float>) {
    constexpr int kMask = 0x7F800000; // These bits set for inf and NaN variants
    auto mask = Simd<int, N>{kMask};
    return ReinterpretCast<Simd<T, N>>(VNotEqual(ReinterpretCast<Simd<int, N>>(a) & mask, mask));
  } else {
    static_assert(std::is_same_v<T, double>, "VIsFinite only supports float and double.");
    constexpr int kMask = 0x7FF00000; // These bits set for inf and NaN variants
    if constexpr (N == 2 && Simd<int, 4>::kIsSupported) {
      // Use an Int4 mask. Result comes from the 2 that correspond to the high bits of each
      // double.
      auto mask = Simd<int, 4>{kMask};
      auto temp4i32 = VNotEqual(ReinterpretCast<Simd<int, 4>>(a) & mask, mask);
      auto temp2i64 = Shuffle<1, 1, 3, 3>(temp4i32);
      return ReinterpretCast<Simd<double, 2>>(temp2i64);
    } else if constexpr (N == 4 && Simd<int, 8>::kIsSupported) {
      // Use an int8 mask. Result comes from the 4 that correspond to the high bits of each
      // double.
      auto mask = Simd<int, 8>{kMask};
      auto temp8i32 = VNotEqual(ReinterpretCast<Simd<int, 8>>(a) & mask, mask);
      auto low2i64 = Shuffle<1, 1, 3, 3>(GetHalf<0>(temp8i32)); // no 8-way shuffle currently
      auto high2i64 = Shuffle<1, 1, 3, 3>(GetHalf<1>(temp8i32));
      auto temp4i64 = Simd<int, 8>(low2i64, high2i64);
      return ReinterpretCast<Simd<double, 4>>(temp4i64);
    } else if constexpr (Simd<T, N>::kIsComposite) {
      return Simd<T, N>{VIsFinite(a.first), VIsFinite(a.second)};
    } else if constexpr (!MOCHI_USE_SIMD) {
      // Simd Emulated
      Simd<T, N> result;
      for (int i = 0; i < N; ++i) {
        uint64_t isFinite = -static_cast<uint64_t>(IsFinite(a.raw[i])); // true = -1, false = 0
        memcpy(&result.raw[i], &isFinite, sizeof(T));
      }
      return result;
    } else {
      static_assert(std::is_void_v<T>, "Unsupported type or size");
    }
  }
}

template <int i, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE int IsTrue(Simd<T, N> mask) {
  // The bits of mask[i] should be all zeros for "false", or all ones for "true". Therefore we can
  // test any byte(s) within mask[i]. We never return a floating-point type so that the caller
  // doesn't have to worry about comparisons with NaN.
  constexpr int kNumInts{(sizeof(T) * N) / sizeof(int)};
  constexpr int kStride = kNumInts / N;
  return Get<i * kStride>(ReinterpretCast<Simd<int, kNumInts>>(mask));
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> MulAdd(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c) {
  return Simd<T, N>::MulAdd(a, b, c);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> MulSub(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c) {
  return Simd<T, N>::MulSub(a, b, c);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> NegMulAdd(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c) {
  return Simd<T, N>::NegMulAdd(a, b, c);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> NegMulSub(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c) {
  return Simd<T, N>::NegMulSub(a, b, c);
}

template <int COUNT, class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VDot(V a, V b) {
  constexpr int COUNT_ = (COUNT == -1) ? V::kSize : COUNT;
  return V::template Dot<COUNT_>(a, b);
}

template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Cross3(Simd<T, N> a, Simd<T, N> b) {
  return Shuffle<1, 2, 0, 3>(MulSub(a, Shuffle<1, 2, 0, 3>(b), b * Shuffle<1, 2, 0, 3>(a)));
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> VNormSqr(Simd<T, N> a) {
  static_assert(COUNT == -1 || COUNT >= 2, "Unsupported COUNT");
  static_assert(std::is_floating_point_v<T>, "Requires float or double");
  return VDot<COUNT>(a, a);
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> VNorm(Simd<T, N> a) {
  return Sqrt(VNormSqr<COUNT>(a));
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T NormSqr(Simd<T, N> a) {
  return Get0(VNormSqr<COUNT>(a));
}

template <int COUNT, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T Norm(Simd<T, N> a) {
  return Sqrt(NormSqr<COUNT>(a));
}

template <int COUNT, class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Normalize(Simd<T, N> a) {
  // By adding the smallest possible scalar we prevent divide-by-zero and get a zero vector result
  // There is no change in result for any vector longer than.... something very very very small
  return a / (VNorm<COUNT>(a) + std::numeric_limits<T>::min());
}

template <class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Normalize(Simd<T, N> a, Simd<T, N> normSqr) {
  // By adding the smallest possible scalar we prevent divide-by-zero and get a zero vector result
  // There is no change in result for any vector longer than.... something very very very small
  return a / (Sqrt(normSqr) + std::numeric_limits<T>::min());
}

template <class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Normalize(Simd<T, N> a, T normSqr) {
  return Normalize(a, Simd<T, N>{normSqr});
}

template <class T>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, 4> OrthogonalVector3(Simd<T, 4> a) {
  static_assert(std::is_floating_point_v<T>, "Requires float or double");

  // Project the coordinate of the minimum absolute value
  // then build the orthogonal vector in that subspace

  // Compute mask selecting the minimum absolute value(s)
  auto abs = Abs(a);
  auto absMin = HMin<3>(abs);

  // Build possible orthogonal vectors and select them according to the mask.
  a = ToSimdDirection(a);
  if (absMin == Get<0>(abs)) { // abs(v[0]) was the smallest value (or tied for smallest)
    a = Neg<false, false, true, false>(Shuffle<3, 2, 1, 3>(a));
  } else if (absMin == Get<1>(abs)) { // abs(v[1]) was the smallest value (or tied for smallest)
    a = Neg<true, false, false, false>(Shuffle<2, 3, 0, 3>(a));
  } else { // abs(v[2]) was the smallest value
    a = Neg<false, true, false, false>(Shuffle<1, 0, 3, 3>(a));
  }
  return a;
}

namespace details {

/**
  Utilities to determine the smallest supported SIMD size that is greater than or equal to a given
  size.
*/
template <typename T, int kSize>
struct NextSupportedSimdSizeHelper {
  static_assert(Simd<T>::kIsSupported, "Type T is not supported for any size N.");
  static constexpr int value = std::conditional_t<
      Simd<T, kSize>::kIsSupported,
      std::integral_constant<int, kSize>,
      NextSupportedSimdSizeHelper<T, kSize + 1>>::value;
};

template <typename T, int kSize>
inline constexpr int kNextSupportedSimdSize = NextSupportedSimdSizeHelper<T, kSize>::value;

} // namespace details

} // namespace mochi

/************************************************************************************
  Reflection support for Simd<T, N>
  Serializes like std::array<T, N>.
*/
#if MOCHI_USE_REFLECTION
template <typename T, int N>
struct SReflectTypeTraits<mochi::Simd<T, N>> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_array;
  static SReflect::ArrayTypeInfo const& GetTypeInfo() {
    static auto* s_typeInfo =
        SReflect::MakeFixedArrayTypeInfo<mochi::Simd<T, N>, T, N>("mochi::Simd", true);
    return *s_typeInfo;
  }
};
#endif // MOCHI_USE_REFLECTION
