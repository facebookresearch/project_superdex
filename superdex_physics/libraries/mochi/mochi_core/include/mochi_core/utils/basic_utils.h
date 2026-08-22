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

/**********************************************************************

  Base Utils:

    This file contains generic template functions for scalar types (or other types with similar
    operators).

    Other overloads exist for types like NdArray, Matrix, and Simd. Such overloads are declared in
    other headers, with their corresponding types. They generally perform the same operation, but
    they do so member-wise.

**/

#pragma once

// PLEASE DO NOT INCLUDE ADDITIONAL MOCHI HEADERS HERE
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/debug.h>
// PLEASE DO NOT INCLUDE ADDITIONAL MOCHI HEADERS HERE

#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace mochi {

/**********************************************************************
  Absolute Value
*/

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Abs(T a) {
  return a >= T{0} ? a : -a;
}

/**********************************************************************
  AllTrue & AnyTrue convert to bool
*/

// Return true if (a != 0). Simd overloads return true if (a[i] != 0) for all i.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE bool AllTrue(T const& a) {
  return a != T{0};
}

// Return true if (a != 0). Simd overloads return true if (a[i] != 0) for ANY i.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE bool AnyTrue(T const& a) {
  return a != T{0};
}

/**********************************************************************
  Casting
*/

// StaticCast supports numeric conversions (e.g. truncate float to int)
template <
    typename To,
    typename From,
    MOCHI_CONCEPT(std::is_arithmetic_v<From>&& std::is_arithmetic_v<To>)>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr To StaticCast(From const& a) {
  return static_cast<To>(a);
}

// Returns a pointer with the same address as the input pointer, but with a different type.
// Equivalent to reinterpret_cast<From>(To).
template <
    typename To,
    typename From,
    MOCHI_CONCEPT(std::is_pointer_v<From>&& std::is_pointer_v<To>)>
[[nodiscard]] MOCHI_FORCE_INLINE To ReinterpretCast(From const& a) {
  static_assert(sizeof(To) == sizeof(From), "Invalid cast. Types must be the same size.");
  return reinterpret_cast<To>(a);
}

// Returns an arithmetic type with the same bits as the input variable.
// Example: auto var = ReinterpretCast<float>(-1); // var is a float with all bits set to 1
template <
    typename To,
    typename From,
    MOCHI_CONCEPT(std::is_arithmetic_v<From>&& std::is_arithmetic_v<To>)>
[[nodiscard]] MOCHI_FORCE_INLINE To ReinterpretCast(From const& a) {
  static_assert(sizeof(To) == sizeof(From), "Invalid cast. Types must be the same size.");
  To b;
  memcpy(&b, &a, sizeof(a));
  return b;
}

/**********************************************************************
  Fused Multiply-Add (and friends)
*/

template <typename A, typename B, typename C>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto MulAdd(A a, B b, C c) {
  return (a * b) + c;
}

template <typename A, typename B, typename C>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto MulSub(A a, B b, C c) {
  return (a * b) - c;
}

template <typename A, typename B, typename C>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto NegMulAdd(A a, B b, C c) {
  return -(a * b) + c;
}

template <typename A, typename B, typename C>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto NegMulSub(A a, B b, C c) {
  return -(a * b) - c;
}

/**********************************************************************
  IsFinite
*/

template <typename T, MOCHI_CONCEPT(std::is_arithmetic_v<T>)>
[[nodiscard]] MOCHI_FORCE_INLINE bool IsFinite(T a) {
  if constexpr (std::is_integral_v<T>) {
    return true;
  } else {
    return std::isfinite(a);
  }
}

/**********************************************************************
  Min, Max, Clamp, Lerp
*/

template <typename T>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T const& Min(T const& a, T const& b) {
  return (a <= b) ? a : b;
}

// For convenience, you can find the minimum of an arbitrary number of values.
template <typename T, typename... More>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Min(T a, T b, T c, More... args) {
  return Min(a, Min(b, c, args...));
}

template <typename T>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T const& Max(T const& a, T const& b) {
  return (a >= b) ? a : b;
}

// For convenience, you can find the maximum of an arbitrary number of values.
template <typename T, typename... More>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Max(T a, T b, T c, More... args) {
  return Max(a, Max(b, c, args...));
}

template <typename ValT, typename MinT, typename MaxT>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr ValT Clamp(ValT value, MinT min, MaxT max) {
  return Min(ValT(max), Max(ValT(min), value));
}

template <typename ValT, typename MinT, typename MaxT, bool kMinInclusive, bool kMaxInclusive>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr ValT Rect(
    ValT value,
    MinT min,
    MaxT max,
    std::integral_constant<bool, kMinInclusive> /*minInclusive*/ = std::true_type{},
    std::integral_constant<bool, kMaxInclusive> /*maxInclusive*/ = std::false_type{}) {
  bool test1 = false;
  if constexpr (kMinInclusive) {
    test1 = (value >= min);
  } else {
    test1 = (value > min);
  }
  bool test2 = false;
  if constexpr (kMaxInclusive) {
    test2 = (value <= max);
  } else {
    test2 = (value < max);
  }
  return static_cast<ValT>(test1 && test2);
}

/**********************************************************************
  Equality
 */

// clang-format off
template<typename T> inline constexpr T kDefaultNearEqualEpsilon{0};
template<> inline constexpr float kDefaultNearEqualEpsilon<float>{1.0e-6f};
template<> inline constexpr double kDefaultNearEqualEpsilon<double>{1.0e-6}; // Historical value. Could probably be much smaller.
// clang-format on

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto Equal(T const& a, T const& b) {
  return (a == b);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto NotEqual(T const& a, T const& b) {
  return !Equal(a, b);
}

template <typename T, MOCHI_CONCEPT(std::is_arithmetic_v<T>)>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto
NearEqual(T const& a, T const& b, T epsilon = kDefaultNearEqualEpsilon<T>) {
  static_assert(!std::is_same_v<std::remove_cv_t<T>, bool>, "NearEqual not supported for bool");
  return Abs(a - b) <= epsilon;
}

template <typename T, MOCHI_CONCEPT(std::is_arithmetic_v<T>)>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto
NearEqualRel(T const& a, T const& b, T epsilon = kDefaultNearEqualEpsilon<T>) {
  static_assert(!std::is_same_v<std::remove_cv_t<T>, bool>, "NearEqualRel not supported for bool");
  auto scaledEpsilon = epsilon * Max(Abs(a), Abs(b));
  auto tolerance = Max(epsilon, scaledEpsilon);
  return NearEqual(a, b, tolerance);
}

template <typename T, MOCHI_CONCEPT(std::is_arithmetic_v<T>)>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr auto NearZero(
    T const& a,
    T epsilon = kDefaultNearEqualEpsilon<T>) {
  static_assert(!std::is_same_v<std::remove_cv_t<T>, bool>, "NearZero not supported for bool");
  return Abs(a) <= epsilon;
}

/**********************************************************************
  isize - Return a container size as int. Inspired by std::ssize.
          http://en.cppreference.com/w/cpp/iterator/size
*/

// Generic version uses the size() member function.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr int isize(T const& a) {
  return static_cast<int>(a.size());
}

// Specialization for c-style arrays
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr int isize([[maybe_unused]] T const (&array)[N]) {
  return static_cast<int>(N);
}

/**********************************************************************
  Linear Interpolation
*/

template <typename T, typename Frac>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Lerp(T a, T b, Frac t) {
  return T(a * (Frac{1} - t) + b * t); // The faction t is not clamped
}

// Linearly remap a value from one number range to another. Does not clamp. If the value was outside
// the input range, it will be outside the output range as well. You also remap an ascending range
// onto a descending range or vice versa by setting (inA > inB) or (outA > outB).
// Examples:
//  Remap(val, 0.0, 1.0, 0.0, 255.0); // As value goes from 0->1, result goes from 0->255
//  Remap(val, 1.0, 0.0, 0.0, 255.0); // As value goes from 1->0, result goes from 0->255
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Remap(T value, T inA, T inB, T outA, T outB) {
  return Lerp<T>(outA, outB, (value - inA) / (inB - inA));
}

// Similar to Remap (see above), except that the result is guaranteed to be between outA and outB.
// Note that the output range might be ascending (outA < outB) or descending (outB < outA).
// Examples:
//  RemapAndClamp(1.1, 0.0, 1.0, 0.0, 255.0); // Returns 255.0
//  RemapAndClamp(1.1, 0.0, 1.0, 255.0, 0.0); // Returns 0.0, as if value was 1.0 (not 1.1)
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T RemapAndClamp(T value, T inA, T inB, T outA, T outB) {
  return Lerp<T>(outA, outB, Clamp<T>((value - inA) / (inB - inA), T{0}, T{1}));
}

/**********************************************************************
  Powers
*/

template <typename B, typename E>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr B Pow(B base, E exp) {
  return std::pow(base, exp);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Sqr(T a) {
  return a * a;
}

// Result as the same sign as the input
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T SignedSqr(T a) {
  return a * Abs(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Sqrt(T a) {
  return std::sqrt(a);
}

// Result as the same sign as the input
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T SignedSqrt(T a);

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T IntegralSqrt(T a);

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr bool IsPowerOfTwo(T a) {
  static_assert(
      std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>,
      "IsPowerOfTwo requires a non-bool integral type");
  return (a > T(0)) && ((a & (a - T(1))) == T(0));
}

// Returns the smallest power of two greater than or equal to a. Non-positive inputs return 1.
// Positive inputs must not exceed the largest representable power of two.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T NextPowerOfTwo(T a) {
  static_assert(
      std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>,
      "NextPowerOfTwo requires a non-bool integral type");
  [[maybe_unused]] constexpr T kMaxPowerOfTwo = T(1) << (std::numeric_limits<T>::digits - 1);
  MOCHI_ASSERT_VERBOSE(
      a <= kMaxPowerOfTwo, "Input exceeds the largest representable power of two.");
  if (a <= T(0)) {
    return T(1);
  }
  using UnsignedT = std::make_unsigned_t<T>;
  static_assert(
      std::numeric_limits<UnsignedT>::digits > 4 && std::numeric_limits<UnsignedT>::digits <= 64,
      "Unsupported integral type");
  auto result = static_cast<UnsignedT>(a - T(1));
  result |= result >> 1;
  result |= result >> 2;
  result |= result >> 4;
  if constexpr (std::numeric_limits<UnsignedT>::digits > 8) {
    result |= result >> 8;
  }
  if constexpr (std::numeric_limits<UnsignedT>::digits > 16) {
    result |= result >> 16;
  }
  if constexpr (std::numeric_limits<UnsignedT>::digits > 32) {
    result |= result >> 32;
  }
  return static_cast<T>(result + UnsignedT(1));
}

/**********************************************************************
  Reciprocal
*/

MOCHI_WARNING_PUSH()
MOCHI_WARNING_IGNORE_MSVC(4723) // warning C4723: potential divide by 0

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Rcp(T a) {
  return T(1) / a;
}

// SIMD overloads of RcpApprox may have less precision. Use with caution.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T RcpApprox(T a) {
  return T(1) / a;
}

MOCHI_WARNING_POP()

/**********************************************************************
  Rounding
*/

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Floor(T a) {
  return std::floor(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Ceil(T a) {
  return std::ceil(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Round(T a) {
  return std::round(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T RoundUp(T numToRound, T multiple) {
  static_assert(
      std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>, "Unsupported type");
  return ((numToRound + multiple - 1) / multiple) * multiple;
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T RoundDown(T numToRound, T multiple) {
  static_assert(
      std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>, "Unsupported type");
  return (numToRound / multiple) * multiple;
}

/**********************************************************************
  Select
*/

// Return (condition ? a : b).
// Only supported for arithmetic types to prevent misuse. For other types, callers should explicitly
// use a ternary expression if that's what they want. Other overloads do this per-member (e.g. for a
// Matrix or Simd type).
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Select(bool condition, T a, T b) {
  static_assert(std::is_arithmetic_v<T>, "Unsupported type");
  return condition ? a : b;
}

/**********************************************************************
  Sign
*/

// Return +1 or -1 matching the sign of the given value.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Sign(T x) {
  return Select(x >= T{}, T{1}, T{-1});
}

/**********************************************************************
  Trigonometry
*/

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T ASin(T a) {
  return std::asin(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Sin(T a) {
  return std::sin(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Sinc(T a);

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Cos(T a) {
  return std::cos(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Tan(T a) {
  return std::tan(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T ACos(T a) {
  return std::acos(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T ATan(T a) {
  return std::atan(a);
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T ATan2(T y, T x) {
  return std::atan2(y, x);
}

/**********************************************************************
  Exponential-based
*/

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Exp(T a) {
  return std::exp(a);
}

} // namespace mochi

#include "basic_utils_inl.h"
