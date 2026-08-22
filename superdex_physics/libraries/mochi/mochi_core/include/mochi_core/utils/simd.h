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
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/reflection.h>

#include <type_traits>
#include <utility>

namespace mochi {

// Forward declaration for use in composite StaticCast (full definition is in half.h)
struct Half;

// Number of values of type T that fit in a native SIMD register
template <class T>
constexpr int kSimdDefaultSize = MOCHI_SIMD_REGISTER_SIZE_BYTES / sizeof(T);

// Type used with enable_if_t that can be used to select a Simd specialization based on type traits
struct SimdConcept {};

/***********************************************************************************************
  Simd<T, N>

  A SIMD vector storing N values of type T.

  This generic template must be fully specialized for supported sizes and types.
  There currently is no reference implementation.
*/
template <class T, int N = kSimdDefaultSize<T>, class Concept = SimdConcept>
class Simd;

// clang-format off
namespace details {
template <class T> constexpr bool IsSimdDef = false;
template <class T, int N> constexpr bool IsSimdDef<Simd<T, N>> = true;
template <class T> constexpr bool IsSimdSupportedTypeDef = false;
template <> inline constexpr bool IsSimdSupportedTypeDef<float> = true;
template <> inline constexpr bool IsSimdSupportedTypeDef<double> = true;
template <> inline constexpr bool IsSimdSupportedTypeDef<int> = true;
template <> inline constexpr bool IsSimdSupportedTypeDef<int64_t> = true;
template <class T> constexpr bool IsHalfDef = false; // Specialized in half.h
} // namespace details
// clang-format on

// Compile-time type traits:
//  - IsSimd<T> is true T is an instantiation of the Simd template class.
//  - IsSimdSupportedType<T> is true if Simd<T, N> is supported for some integer N.
//  - IsHalf<T> is true if T is Half (from half.h).
#if MOCHI_LANGUAGE_CPP20
template <class T>
concept IsSimd = details::IsSimdDef<std::decay_t<T>>;
template <class T>
concept IsSimdSupportedType = details::IsSimdSupportedTypeDef<T>;
template <class T>
concept IsHalf = details::IsHalfDef<std::decay_t<T>>;
#else
template <class T>
inline constexpr bool IsSimd = details::IsSimdDef<std::decay_t<T>>;
template <class T>
inline constexpr bool IsSimdSupportedType = details::IsSimdSupportedTypeDef<T>;
template <class T>
inline constexpr bool IsHalf = details::IsHalfDef<std::decay_t<T>>;
#endif

/***********************************************************************************************
  Simd<T, N>

  A SIMD vector storing N values of type T.

  This generic template must be fully specialized for supported sizes and types.
  There currently is no reference implementation.
*/
template <class T, int N, class Concept>
class Simd {
 public:
  struct NotSupported {};
  using NativeType = NotSupported;
  using Scalar = T;
  static constexpr int kSize = N;
  static constexpr bool kIsSupported = false; // Can this combination of T and N function
  static constexpr bool kIsComposite = false; // Is this type a composite of native sizes
  static constexpr bool kIsEmulated = false; // Whether it emulates SIMD instructions (if true) or
                                             // uses actual native SIMD instructions (if false).

  // Check to prevent misuse of Simd<T>::kIsSupported if queried with a const T.
  static_assert(!std::is_const_v<T>, "Scalar type must not be const");

  Simd() {
    static_assert(std::is_same_v<T, void>, "The requested SIMD size or type is not supported");
  }

  Simd(Simd const& rhs) = default;

  // Implicit conversion from native storage type.
  Simd(NativeType rhs) : raw(rhs) {};

  // Broadcast scalar
  Simd(Scalar a);

  // Init from 2 or more scalars. Remaining values will be zero.
  template <class... Ts>
  Simd(Scalar a, Scalar b, Ts... vals);

  // Lower-case for compatibility with std::size
  static constexpr size_t size() {
    return kSize;
  }

  // Assignment
  Simd& operator=(Simd rhs);
  Simd& operator=(Scalar rhs);

  // Comparison operators
  bool operator==(Simd rhs) const;
  bool operator!=(Simd rhs) const;
  Simd operator<(Simd rhs) const;
  Simd operator<=(Simd rhs) const;
  Simd operator>(Simd rhs) const;
  Simd operator>=(Simd rhs) const;

  // Math operators
  Simd operator-() const;
  Simd operator+(Simd rhs) const;
  Simd operator-(Simd rhs) const;
  Simd operator*(Simd rhs) const;
  Simd operator/(Simd rhs) const;
  Simd& operator+=(Simd rhs);
  Simd& operator-=(Simd rhs);
  Simd& operator*=(Simd rhs);
  Simd& operator/=(Simd rhs);
  Simd& operator+=(T rhs);
  Simd& operator-=(T rhs);
  Simd& operator*=(T rhs);
  Simd& operator/=(T rhs);

  // Bitwise operators
  Simd operator~() const;
  Simd operator&(Simd rhs) const;
  Simd operator|(Simd rhs) const;
  Simd operator^(Simd rhs) const;
  Simd& operator&=(Simd rhs);
  Simd& operator|=(Simd rhs);
  Simd& operator^=(Simd rhs);

  // Logical operators
  // Each lane must have all-bits-0 (logical false) or all-bits-1 (logical true).
  Simd operator&&(Simd rhs) const;
  Simd operator||(Simd rhs) const;

  // Left shift
  Simd operator<<(int shift) const;

  // Indexing (read only)
  Scalar operator[](int i) const;

  // Storage type is implementation specific (e.g. __m256)
  NativeType raw;
};

/// @brief Reinterpret the bits of one Simd type to another of the same binary size.
/// @note Example: ReinterpretCast<Vec2d>(myVec4i);
/// @note The generic version of ReinterpretCast is not valid, but specializations
/// can be provided for specific pairs of types.
template <class VecTo, class VecFrom, MOCHI_CONCEPT(IsSimd<VecTo>&& IsSimd<VecFrom>)>
MOCHI_ANY MOCHI_FORCE_INLINE VecTo ReinterpretCast(VecFrom const& a) {
  static_assert(
      std::is_same_v<std::decay_t<VecTo>, std::decay_t<VecFrom>>,
      "ReinterpretCast not supported for these types.");
  return a;
}

/// @brief StaticCast from one Simd type to another of the same size.
/// @note The generic version is a fallback.
/// Specializations are provided for specific pairs of types.
template <class VecTo, class VecFrom, MOCHI_CONCEPT(IsSimd<VecTo>&& IsSimd<VecFrom>)>
MOCHI_ANY MOCHI_FORCE_INLINE VecTo StaticCast(VecFrom const& a);

} // namespace mochi
#include "simd/simd_specializations_inl.h"
namespace mochi {

// Aliases
using Vec2d = Simd<double, 2>;
using Vec2l = Simd<int64_t, 2>;
using Vec4d = Simd<double, 4>;
using Vec4f = Simd<float, 4>;
using Vec4i = Simd<int, 4>;
using Vec4l = Simd<int64_t, 4>;
using Vec4r = Simd<real, 4>;
using Vec8d = Simd<double, 8>;
using Vec8f = Simd<float, 8>;
using Vec8i = Simd<int, 8>;
using Vec8l = Simd<int64_t, 8>;
using Vec8r = Simd<real, 8>;

// clang-format off
/***********************************************************************************************
                     Native SIMD Function Support Matrix (please keep up-to-date)
                     Functions supporting Simd<T, N> also support Simd<T, MultipleOfN>

                     For Simd<Half, N> (Vec8h, Vec16h), include half.h.

************************************************************************************************

                     | Vec2d | Vec2l | Vec4d | Vec4f | Vec4i | Vec4l | Vec8f | Vec8i | Vec8h | Vec16h |
                 Abs | x     |       | x     | x     |       |       | x     |       |       |        |
                ACos | x     |       | x     | x     |       |       | x     |       |       |        |
             AllTrue | x     | x     | x     | x     | x     | x     | x     | x     | x     | x      |
             AnyTrue | x     |       | x     | x     | x     |       | x     | x     | x     | x      |
                ASin | x     |       | x     | x     |       |       | x     |       |       |        |
                ATan | x     |       | x     | x     |       |       | x     |       |       |        |
               Blend | x     | x     | x     | x     | x     |       |       |       |       |        |
           Broadcast | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
               Clamp | x     |       | x     | x     |       |       | x     |       |       |        |
                 Cos | x     |       | x     | x     |       |       | x     |       |       |        |
              Cross3 |       |       | x     | x     |       |       |       |       |       |        |
                 Dot | x     |       | x     | x     |       |       | x     |       |       |        |
            (V)Equal | x     | x     | x     | x     | x     | x     | x     | x     | x     | x      |
                 Exp | x     |       | x     | x     |       |       | x     |       |       |        |
           FastRound | x     |       | x     | x     |       |       | x     |       |       |        |
               Floor | x     |       | x     | x     |       |       | x     |       |       |        |
                 Get | x     | x     | x     | x     | x     | x     | x     | x     | x     | x      |
                Get0 | x     | x     | x     | x     | x     | x     | x     | x     | x     | x      |
             GetHalf |       |       | x     |       |       | x     | x     | x     |       | x      |
                HMax | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
                HMin | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
               HProd | x     |       | x     | x     |       |       |       |       |       |        |
                HSum | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
         (V)IsFinite | x     |       | x     | x     |       |       | x     |       |       |        |
              IsTrue | x     |       | x     | x     | x     |       | x     | x     |       |        |
                Lerp | x     |       | x     | x     |       |       | x     |       |       |        |
                  Ln | x     |       | x     | x     |       |       | x     |       |       |        |
                Load | x     | x     | x     | x     | x     | x     | x     | x     | x     | x      |
         LoadIndexed | x     |       | x     | x     |       |       | x     |       |       |        |
      LoadTransposed | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
                 Max | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
                 Min | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
              MulAdd | x     |       | x     | x     |       |       | x     |       |       |        |
              MulSub | x     |       | x     | x     |       |       | x     |       |       |        |
        (V)NearEqual | x     |       | x     | x     |       |       | x     |       |       |        |
         (V)NearZero | x     |       | x     | x     |       |       | x     |       |       |        |
       Neg (4 bools) |       |       | x     | x     |       |       |       |       |       |        |
           NegMulAdd | x     |       | x     | x     |       |       | x     |       |       |        |
           NegMulSub | x     |       | x     | x     |       |       | x     |       |       |        |
             (V)Norm | x     |       | x     | x     |       |       | x     |       |       |        |
           Normalize | x     |       | x     | x     |       |       | x     |       |       |        |
         (V)NotEqual | x     | x     | x     | x     | x     | x     | x     | x     | x     | x      |
   OrthogonalVector3 |       |       | x     | x     |       |       |       |       |       |        |
           RcpApprox | x     |       | x     | x     |       |       | x     |       |       |        |
       RcpSqrtApprox | x     |       | x     | x     |       |       | x     |       |       |        |
              Select | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
                 Set | x     |       | x     | x     | x     |       | x     | x     |       |        |
            Sequence |       | x     |       |       | x     | x     |       | x     |       |        |
          ShiftRight |       | x     |       |       | x     | x     |       | x     |       |        |
     Shuffle (1 arg) | x     | x     | x     | x     | x     | x     |       |       |       |        |
     Shuffle (2 arg) |       |       | x     | x     | x     | x     |       |       |       |        |
                Sign | x     |       | x     | x     |       |       | x     |       |       |        |
          SignedSqrt | x     |       | x     | x     |       |       | x     |       |       |        |
     SimdBasisVector |       |       | x     | x     |       |       |       |       |       |        |
            SimdMask | x     |       | x     | x     | x     |       | x     | x     |       |        |
            SimdZero | x     | x     | x     | x     | x     | x     | x     | x     | x     | x      |
                 Sin | x     |       | x     | x     |       |       | x     |       |       |        |
              SinCos | x     |       | x     | x     |       |       | x     |       |       |        |
                 Sqr | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
                Sqrt | x     |       | x     | x     |       |       |       |       |       |        |
               Store | x     | x     | x     | x     | x     | x     | x     | x     | x     | x      |
       StoreSelected | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
     StoreTransposed | x     | x     | x     | x     | x     | x     | x     | x     |       |        |
                 Tan | x     |       | x     | x     |       |       | x     |       |       |        |
                Tanh | x     |       | x     | x     |       |       | x     |       |       |        |
              ToSimd | x     |       | x     | x     | x     |       | x     | x     |       |        |
     ToSimdDirection |       |       | x     | x     |       |       |       |       |       |        |
         ToSimdPoint |       |       | x     | x     |       |       |       |       |       |        |

***********************************************************************************************/
// clang-format on

// Make kDefaultNearEqualEpsilon<T> "just work" when T is a Simd type
template <class T, int N>
inline Simd<T, N> const kDefaultNearEqualEpsilon<Simd<T, N>>{kDefaultNearEqualEpsilon<T>};

// Broadcast a single value to all elements of the result
template <class V, MOCHI_CONCEPT(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Broadcast(typename V::Scalar a);

// Broadcast a single value at address ptr to all elements of the result
template <class V, MOCHI_CONCEPT(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Broadcast(typename V::Scalar const* ptr);

// Broadcast the ith element of a vector to all elements of the result
template <int i, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Broadcast(Simd<T, N> v);

// Broadcast the ith element of a vector to all elements of the result.
// Prefer Broadcast<i> if index is known at compile time.
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Broadcast(Simd<T, N> v, int i);

// Return V{b0 ? TRUE : FALSE, b1 ? TRUE : FALSE, ...} where TRUE is represented by all bits set
// to 1 (e.g. 0xFFFFFFFF) and FALSE is represented by all bits set to 0 (e.g. 0x00000000). This is
// the same type of mask returned by comparison operators like <, <=, >, and >=. It can be used with
// functions like AllTrue, AnyTrue, and Select.
// Example: SimdMask<Vec4r>(true, false, true, false)
template <class V, class... MoreBools>
MOCHI_ANY MOCHI_FORCE_INLINE V SimdMask(bool b0, bool b1, MoreBools... bs);

// Return 0 value
//
// The function is templated on the return type.
template <class V = Vec4r>
MOCHI_ANY MOCHI_FORCE_INLINE V SimdZero();

// Return a vector with a single component set to 1, and the rest to 0.
template <int i, class V = Vec4r>
MOCHI_ANY MOCHI_FORCE_INLINE V SimdBasisVector();

// Return a vector with a single component set to 1, and the rest to 0.
template <class V = Vec4r>
MOCHI_ANY MOCHI_FORCE_INLINE V SimdBasisVector(int axis);

// Return {a[0], a[1], a[2], 1}. Used for positions that can be both rotated & translated.
template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V ToSimdPoint(V a);

// Return {a[0], a[1], a[2], 0}. Used for direction vectors that can only be rotated.
template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V ToSimdDirection(V a);

// Read V::kSize scalars from the specified address into a vector
template <class V, MOCHI_CONCEPT(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Load(typename V::Scalar const* ptr);

// Read N scalars from the specified address into a vector.
template <int N, class V, MOCHI_CONCEPT(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Load(typename V::Scalar const* ptr);

// Read n scalars from the specified address into a vector.
// Prefer Load<N, V> if the number is a constexpr.
template <class V, MOCHI_CONCEPT(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V Load(typename V::Scalar const* ptr, int n);

// Read V::kSize scalars from an unaligned address offset by Simd<I, kSize>, where I is an integer
// type.
template <class V, class I, MOCHI_CONCEPT(IsSimd<V>)>
MOCHI_ANY MOCHI_FORCE_INLINE V
LoadIndexed(typename V::Scalar const* ptr, Simd<I, V::kSize> indices);

// Load (kTupleCount * 3) values from memory. Deinterleave into 3 output vectors.
// By default (kTupleCount == N).
//
// If the memory order was:
//    {x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3}
//
// ...then the output vectors will be:
//    {x0, x1, x2, x3}, {y0, y1, y2, y3}, {z0, z1, z2, z3}
//
template <int kTupleCount = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE void
LoadTransposed(T const* ptr, Simd<T, N>& out0, Simd<T, N>& out1, Simd<T, N>& out2);

// Return the first component of a vector (e.g. a[0])
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T Get0(Simd<T, N> v);

// Return the ith component of a vector (e.g a[i]), when the index is a constexpr
template <int i, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T Get(Simd<T, N> v);

// Return the ith component of a vector (e.g. a[i]).
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T Get(Simd<T, N> v, int i);

// Return the low or high half of a vector via GetHalf<0>(a) or GetHalf<1>(a)
template <int iHalf, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N / 2> GetHalf(Simd<T, N> a);

// Return a copy of the vector with the ith component replaced with a new value.
template <int i, class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Set(Simd<T, N> a, T value);

// Return a copy of the vector with the ith component replaced with a new value.
template <class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Set(Simd<T, N> a, int i, T value);

// Returns a vector with integer values [0, 1, 2, 3, ...].
// If you want a different starting value, then add an integer to the result.
// Example: (Sequence<V>() + 1) results in the sequence [1, 2, 3, ...]
template <class V>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE V Sequence();

// Write COUNT scalars from a vector to the specified address. A COUNT of -1 means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE void Store(T* ptr, Simd<T, N> a);

// Write count scalars from a vector to the specified address.
// Prefer Store<N> if the number is a constexpr.
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE void Store(T* ptr, Simd<T, N> a, int count);

// Write each value for which condition[i] is true. It may then write unspecified values for a total
// of N values written, so the destination buffer must be large enough. Returns the number of
// values for which condition[i] was true.
//
// Example - Store all the positive values (branchless):
//   int count = 0;
//   for (int i = 0; i + N <= isize(src); i += N) {
//     auto values = Load<Simd<T, N>>(&src[i]);
//     count += StoreSelected(&dst[count], values >= 0, values);
//   }
//
// Warning: If the condition is somewhat predictable (e.g. many false values in a row), then a
// traditional branching algorithm may be faster. You should always profile it.
//
template <class T, int N, class MaskT>
MOCHI_ANY MOCHI_FORCE_INLINE int StoreSelected(T* ptr, Simd<MaskT, N> condition, Simd<T, N> values);

// Interleave the values of 3 input vectors. Store those (kTupleCount * 3) values to memory.
// By default (kTupleCount == N).
//
// If the input vectors are:
//    {x0, x1, x2, x3}, {y0, y1, y2, y3}, {z0, z1, z2, z3}
//
// ...then the order written to memory will be:
//    {x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3}
//
template <int kTupleCount = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE void StoreTransposed(T* ptr, Simd<T, N> a, Simd<T, N> b, Simd<T, N> c);

// Return result[i] = conditionalMask[i] ? a[i] : b[i]
// where conditionMask[i] is 0x00000000 (false) or 0xFFFFFFFF (true).
// Example: Select(a <= b, a, b) is equivalent to Min(a, b)
template <class T, int N, class MaskT>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N>
Select(Simd<MaskT, N> conditionMask, Simd<T, N> a, Simd<T, N> b);

// Special bit shift operations
template <int kShift, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> ShiftRight(Simd<T, N> a); // return (a >> kShift)

// Shuffle elements in a vector. Returns: {a[i0], a[i1], ... }
template <int x = 0, int y = 1, class T>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, 2> Shuffle(Simd<T, 2> a);
template <int x = 0, int y = 1, int z = 2, int w = 3, class T>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, 4> Shuffle(Simd<T, 4> a);

// Shuffle elements between two vectors. Returns: { a[x], a[y], b[z], b[w] }
template <int x, int y, int z, int w, class T>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, 4> Shuffle(Simd<T, 4> a, Simd<T, 4> b);

// Blend returns { (x ? b[0] : a[0]), (y ? b[1] : a[1]) }
template <int x, int y, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Blend(Simd<T, N> a, Simd<T, N> b);

// Blend returns { (x ? b[0] : a[0]), (y ? b[1] : a[1]), (z ? b[2] : a[2]), (w ? b[3] : a[3]) }
template <int x, int y, int z, int w, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Blend(Simd<T, N> a, Simd<T, N> b);

// Math Operations:
template <bool x, bool y, bool z, bool w, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Neg(
    Simd<T, N> a); // { x ? -a[0] : a[0], y ? -a[1] : a[1], z ? -a[2] : a[2], w ? -a[3] : a[3] }
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Sqrt(Simd<T, N> a); // sqrt(a)
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> RcpApprox(Simd<T, N> a); // Fast approximation of 1/a
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> RcpSqrtApprox(
    Simd<T, N> a); // Fast approximation of 1/sqrt(a)
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Abs(Simd<T, N> a); // abs(a)

// NOTE: NaN behavior of Min/Max is undefined (platform-dependent and operand-order-dependent).
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Min(Simd<T, N> a, Simd<T, N> b); // min(a, b)
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Max(Simd<T, N> a, Simd<T, N> b); // max(a, b)
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Floor(Simd<T, N> a); // floor(a)

// Round to the nearest integer. On ARM, exact ties are rounded away from zero, just like with
// std::round. However, on x64 ties are rounded to the nearest even integer. Use only in situations
// where this discrepancy is not an issue.
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> FastRound(Simd<T, N> a);

// NOT FAST: Horizontal sum of first COUNT elements of a vector. COUNT of -1 means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T HSum(Simd<T, N> a);

// NOT FAST: Horizontal product of first COUNT elements of a vector. COUNT of -1 means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T HProd(Simd<T, N> a);

// NOT FAST: Horizontal minimum of first COUNT elements of a vector. COUNT of -1 means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T HMin(Simd<T, N> a);

// NOT FAST: Horizontal maximum of first COUNT elements of a vector. COUNT of -1 means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T HMax(Simd<T, N> a);

// We have a custom implementation of cosine for single-precision floats. It matches the precision
// of std::cos to within than 6.0e-8. Performance is a little slower than a single call to std::cos,
// but faster than 2 calls, and much faster than N calls. For double-precision, we fall back on SVML
// intrinsics (no faster than our implementation), or std::cos if SVML is not available.
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Cos(Simd<T, N> a);

// We have a custom implementation of sine for single-precision floats. Performance and precision is
// nearly identical to Cos (see notes above).
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Sin(Simd<T, N> a);

// If you need both the sine and the cosine of an angle, then this function can compute both at the
// same time. It is faster than calling std::sin + std::cos for a single float. In that time, it
// computes sine and cosine for all N vector components. Precision matches the standard library
// functions to within 6.0e-8 for single-precision floats. For double-precision, we fall back on
// SVML or scalar functions (more precision, but slower).
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE std::pair<Simd<T, N>, Simd<T, N>> SinCos(Simd<T, N> a);

// These trig functions are implemented with intrinsics with the SVML extension is available on x64
// platforms. Otherwise, these functions are quite slow because they simply call the corresponding
// std library functions N times.
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Tan(Simd<T, N> a);
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Tanh(Simd<T, N> a);
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> ACos(Simd<T, N> a);
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> ASin(Simd<T, N> a);
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> ATan(Simd<T, N> a);

// Math functions
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Exp(Simd<T, N> a);
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Ln(Simd<T, N> a);

// Simd Comparisons:
//  Comparisons are performed for each Simd elements. They return a Simd mask where result[i] has
//  all bits set to 0 to signify "false" or all bits set to 1 to signify "true". This is true for
//  Simd operators <, <=, >, and >= as well as Simd functions with a "V" prefix.
//
// Exception:
//  Simd operator== and operator!= return a bool to match typical user expectation. Specifically
//  operator== returns true if all Simd elements are equal, while operator!= returns true if any
//  Simd elements are not equal.
//
// Example:
//  auto a = VEqual(Vec4i{1,2,3,4}, Vec4i{1,0,3,0});
//  MOCHI_ASSERT(a == Vec4i{0xFFFFFFFF, 0, 0xFFFFFFFF, 0});
//

// Return true if ALL of the first COUNT elements have a non-zero bit pattern. COUNT of -1 means
// "all". Example: AllTrue<2>(a < b) returns true iff ((a[0] < b[0]) && (a[1] < b[1]))
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool AllTrue(Simd<T, N> a);

// Return true if ANY of the first COUNT elements have a non-zero bit pattern. COUNT of -1 means
// "all". Example: AnyTrue<2>(a < b) returns true iff ((a[0] < b[0]) || (a[1] < b[1]))
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool AnyTrue(Simd<T, N> a);

// Returns result[i] = (a[i] == b[i]) ? 0xFFFFFFFF : 0x00000000;
// See note above regarding Simd comparisons.
template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VEqual(V a, V b);

// Return true if (a[i] == b[i]) for ALL of the first COUNT elements. Count of -1 means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool Equal(Simd<T, N> a, Simd<T, N> b);

// Returns result[i] = (a[i] != b[i]) ? 0xFFFFFFFF : 0x00000000;
// See note above regarding Simd comparisons.
template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VNotEqual(V a, V b);

// Return true if (a[i] != b[i]) for ANY of the first COUNT lanes. Count of -1 means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool NotEqual(Simd<T, N> a, Simd<T, N> b);

// Like NearEqual except that it takes a Simd epsilon and returns a Simd bit mask.
// Returns result[i] = (Abs(a[i] - b[i]) <= epsilon[i]) ? 0xFFFFFFFF : 0x00000000;
// See note above regarding Simd comparisons.
template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VNearEqual(V a, V b, V epsilon);

// Return true if (Abs(a[i] - b[i]) <= epsilon) for ALL of the first N lanes.
// Count of -1 means "all".
template <int COUNT = -1, class T, int N, class Eps = T>
MOCHI_ANY MOCHI_FORCE_INLINE bool
NearEqual(Simd<T, N> a, Simd<T, N> b, Eps epsilon = kDefaultNearEqualEpsilon<Eps>) {
  return AllTrue<COUNT>(VNearEqual(a, b, Simd<T, N>{epsilon}));
}

// Like NearZero except that it takes a Simd epsilon and returns a Simd bit mask.
// Returns result[i] = (Abs(a[i]) <= epsilon[i]) ? 0xFFFFFFFF : 0x00000000;
// See note above regarding Simd comparisons.
template <class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VNearZero(V a, V epsilon);

// Return true if (Abs(a[i]) <= epsilon) for ALL of the first N lanes. Count of -1 means "all".
template <int COUNT = -1, class T, int N, class Eps = T>
MOCHI_ANY MOCHI_FORCE_INLINE bool NearZero(
    Simd<T, N> a,
    Eps epsilon = kDefaultNearEqualEpsilon<Eps>) {
  return AllTrue<COUNT>(VNearZero(a, Simd<T, N>{epsilon}));
}

// Returns result[i] = IsFinite(a[i]) ? 0xFFFFFFFF : 0x00000000;
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> VIsFinite(Simd<T, N> a);

// Return true if all elements of the Simd vector are finite (not +inf, -inf, nor NaN).
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE bool IsFinite(Simd<T, N> a) {
  return AllTrue(VIsFinite(a));
}

// Return a non-zero integer if component i of the vector is considered "true" (see AnyTrue,
// AllTrue). Example: "if (IsTrue<i>(a < b))" is like writing "if (a[i] < b[i])".
template <int i, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE int IsTrue(Simd<T, N> mask);

// Fused Math Operations:
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N>
MulAdd(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c); // (a * b) + c
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N>
MulSub(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c); // (a * b) - c
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N>
NegMulAdd(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c); // -(a * b) + c
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N>
NegMulSub(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c); // -(a * b) - c

// Dot product of the first COUNT elements. Count of -1 means "all". Broadcasts the result to all
// elements. Use "Dot" (not "VDot") if you want a single scalar result.
template <int COUNT = -1, class V>
MOCHI_ANY MOCHI_FORCE_INLINE V VDot(V a, V b);

// Dot product of the first COUNT elements. Count of -1 means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T Dot(Simd<T, N> a, Simd<T, N> b) {
  return Get0(VDot<COUNT>(a, b));
}

// Cross Product (3 component)
template <class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Cross3(Simd<T, N> a, Simd<T, N> b);

// Compute the square of the norm using the first COUNT elements of a Simd vector. A count of -1
// means "all". The result will be broadcasted to a new vector. Use NormSqr instead if you want a
// scalar result.
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> VNormSqr(Simd<T, N> a);

// Compute the norm using the first COUNT elements of a Simd vector. A count of -1 means "all". The
// result will be broadcasted to a new vector. Use Norm instead if you want a scalar result.
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> VNorm(Simd<T, N> a);

// Compute the square of the norm using the first COUNT elements of a Simd vector. A count of -1
// means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T NormSqr(Simd<T, N> a);

// Compute the norm using the first COUNT elements of a Simd vector. Count of -1 means "all".
template <int COUNT = -1, class T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE T Norm(Simd<T, N> a);

// Normalizes Simd<T, N> elements introducing an epsilon term (to avoid divide-by-zero).
// Returns (x / (√n₀ + ε), y / (√n₁ + ε), z / (√n₂ + ε), w / (√n₃ + ε))
template <int COUNT = -1, class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Normalize(Simd<T, N> a);

// Use this overload if you have the square of the norm as a Simd vector.
template <class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Normalize(Simd<T, N> a, Simd<T, N> normSqr);

// Use this overload if you have the square of the norm as a scalar.
template <class T, int N>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, N> Normalize(Simd<T, N> a, T normSqr);

// Builds an arbitrary vector that is orthogonal to the given one.
template <typename T, int N>
MOCHI_ANY MOCHI_FORCE_INLINE Simd<T, 4> OrthogonalVector3(Simd<T, 4> a);

} // namespace mochi

#include "simd/simd_inl.h"
