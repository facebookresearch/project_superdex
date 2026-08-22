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

// Reverse include for intellisense
#include "math_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace mochi {

/**************************************************************************************************
  ArgMin, ArgMax
*/

// Reference implementation of std::min_element from:
// https://en.cppreference.com/w/cpp/algorithm/min_element
template <class ForwardIt>
inline constexpr ForwardIt MinElement(ForwardIt first, ForwardIt last) {
  if (first == last) {
    return last;
  }

  ForwardIt smallest = first;
  ++first;
  for (; first != last; ++first) {
    if (*first < *smallest) {
      smallest = first;
    }
  }
  return smallest;
}

// Reference implementation of std::max_element from:
// https://en.cppreference.com/w/cpp/algorithm/max_element
template <class ForwardIt>
inline constexpr ForwardIt MaxElement(ForwardIt first, ForwardIt last) {
  if (first == last) {
    return last;
  }

  ForwardIt largest = first;
  ++first;
  for (; first != last; ++first) {
    if (*largest < *first) {
      largest = first;
    }
  }
  return largest;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr size_t ArgMin(NdArray<T, N> const& a) {
  return (size_t)(MinElement(a.begin(), a.end()) - a.begin());
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr size_t ArgMax(NdArray<T, N> const& a) {
  return (size_t)(MaxElement(a.begin(), a.end()) - a.begin());
}

/**************************************************************************************************
  NearEqual: (abs(a-b) <= epsilon)
*/

template <typename T, typename Eps, size_t D0, size_t... DIMS>
MOCHI_FORCE_INLINE constexpr bool
NearEqual(NdArray<T, D0, DIMS...> const& a, NdArray<T, D0, DIMS...> const& b, Eps epsilon) {
  static_assert(D0 > 0, "NearEqual requires non-empty arrays");
  bool isNear = NearEqual(a[0], b[0], epsilon);
  for (size_t i = 1; i < D0; ++i) {
    isNear = isNear && NearEqual(a[i], b[i], epsilon);
  }
  return isNear;
}

template <typename T, size_t D0, int D1>
MOCHI_FORCE_INLINE Simd<T, D1> VNearEqual(
    NdArray<Simd<T, D1>, D0> const& a,
    NdArray<Simd<T, D1>, D0> const& b,
    Simd<T, D1> epsilon) {
  static_assert(D0 > 0, "VNearEqual requires non-empty arrays");
  auto isNear = VNearEqual(a[0], b[0], epsilon);
  for (size_t i = 1; i < D0; ++i) {
    isNear &= VNearEqual(a[i], b[i], epsilon);
  }
  return isNear;
}

template <typename T, size_t D0, int D1>
MOCHI_FORCE_INLINE bool
NearEqual(NdArray<Simd<T, D1>, D0> const& a, NdArray<Simd<T, D1>, D0> const& b, T epsilon) {
  return AllTrue(VNearEqual(a, b, Simd<T, D1>{epsilon}));
}

/**************************************************************************************************
  Euclidean Norm
*/

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr T NormSqr(NdArray<T, N> const& a) {
  return Dot(a, a);
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE T Norm(NdArray<T, N> const& a) {
  return Sqrt(NormSqr<T, N>(a));
}

/**************************************************************************************************
  Basis vector
*/

template <typename T, int N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> BasisVector(int axis) {
  NdArray<T, N> v = {};

  for (int i = 0; i < axis; ++i) {
    v[i] = (T)0;
  }

  v[axis] = (T)1;

  for (int i = axis + 1; i < N; ++i) {
    v[i] = (T)0;
  }

  return v;
}

/**************************************************************************************************
  Normalize (make unit length)
*/

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Normalize(NdArray<T, N> const& a) {
  // By adding the smallest possible scalar we prevent divide-by-zero and get a zero vector result
  // There is no change in result for any vector longer than.... something very very very small.
  constexpr auto kSmallestFloat = std::numeric_limits<ScalarType<T>>::min();
  static_assert(kSmallestFloat > 0); // Check numeric_limits has been correctly specialized.
  return a * (T(1) / (Norm(a) + T(kSmallestFloat)));
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Normalize(NdArray<T, N> const& a, T sqrNorm) {
  constexpr auto kSmallestFloat = std::numeric_limits<ScalarType<T>>::min();
  static_assert(kSmallestFloat > 0); // Check numeric_limits has been correctly specialized.
  return a * (T(1) / (Sqrt(sqrNorm) + T(kSmallestFloat)));
}

/**************************************************************************************************
  Sum, Prod and Mean
*/

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr T Sum(NdArray<T, N> const& a) {
  static_assert(std::is_arithmetic_v<T>, "Arithmetic type required");
  static_assert(N > 0, "Sum requires a non-empty array");
  T sum = a[0];
  for (size_t i = 1; i < N; ++i) {
    sum += a[i];
  }
  return sum;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr T Prod(NdArray<T, N> const& a) {
  static_assert(std::is_arithmetic_v<T>, "Arithmetic type required");
  static_assert(N > 0, "Prod requires a non-empty array");
  T prod = a[0];
  for (size_t i = 1; i < N; ++i) {
    prod *= a[i];
  }
  return prod;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr T Mean(NdArray<T, N> const& a) {
  static_assert(std::is_arithmetic_v<T>, "Arithmetic type required");
  static_assert(N > 0, "Mean requires a non-empty array");
  return Sum(a) / static_cast<T>(N);
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr T Max(NdArray<T, N> const& a) {
  static_assert(N > 0, "Max requires a non-empty array");
  static_assert(std::is_arithmetic_v<T>, "Arithmetic type required");
  T max = a[0];
  for (int i = 1; i < N; ++i) {
    if (a[i] > max) {
      max = a[i];
    }
  }
  return max;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr T Min(NdArray<T, N> const& a) {
  static_assert(N > 0, "Min requires a non-empty array");
  static_assert(std::is_arithmetic_v<T>, "Arithmetic type required");
  T min = a[0];
  for (int i = 1; i < N; ++i) {
    if (a[i] < min) {
      min = a[i];
    }
  }
  return min;
}

/**************************************************************************************************
  Floor, ceil, round
*/

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Floor(NdArray<T, N> const& a) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = T(Floor(a[i]));
  }
  return output;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Ceil(NdArray<T, N> const& a) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = T(Ceil(a[i]));
  }
  return output;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Round(NdArray<T, N> const& a) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = T(Round(a[i]));
  }
  return output;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Clamp(NdArray<T, N> const& a, T min, T max) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = Clamp(a[i], min, max);
  }
  return output;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Max(NdArray<T, N> const& a, T max) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = Max(a[i], max);
  }
  return output;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Min(NdArray<T, N> const& a, T min) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = Min(a[i], min);
  }
  return output;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N>
Clamp(NdArray<T, N> const& a, NdArray<T, N> const& min, NdArray<T, N> const& max) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = Clamp(a[i], min[i], max[i]);
  }
  return output;
}

template <typename T, size_t N, bool MinInclusiveT, bool MaxInclusiveT>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Rect(
    NdArray<T, N> const& a,
    NdArray<T, N> const& min,
    NdArray<T, N> const& max,
    std::integral_constant<bool, MinInclusiveT> minInclusive,
    std::integral_constant<bool, MaxInclusiveT> maxInclusive) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = Rect(a[i], min[i], max[i], minInclusive, maxInclusive);
  }
  return output;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Max(NdArray<T, N> const& a, NdArray<T, N> const& max) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = Max(a[i], max[i]);
  }
  return output;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr NdArray<T, N> Min(NdArray<T, N> const& a, NdArray<T, N> const& min) {
  NdArray<T, N> output = {};
  for (int i = 0; i < N; ++i) {
    output[i] = Min(a[i], min[i]);
  }
  return output;
}

/**************************************************************************************************
  Dot Product
*/

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr T Dot(NdArray<T, N> const& a, NdArray<T, N> const& b) {
  static_assert(N > 0, "Dot requires non-empty arrays");
  T result = a[0] * b[0];
  for (size_t i = 1; i < N; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

template <typename T, size_t N>
MOCHI_FORCE_INLINE constexpr T Dot(Span<T const> a, NdArray<T, N> const& b) {
  static_assert(N > 0, "Dot requires non-empty arrays");
  MOCHI_ASSERT_VERBOSE(a.size() == N, "Size mismatch");
  T result = a[0] * b[0];
  for (size_t i = 1; i < N; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

template <typename T>
MOCHI_FORCE_INLINE constexpr T Dot(Span<T const> a, Span<T const> b) {
  MOCHI_ASSERT_VERBOSE(a.size() == b.size(), "Size mismatch");
  T result = T{0};
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

/**************************************************************************************************
  Cross Product
*/

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 3> Cross(NdArray<T, 3> const& a, NdArray<T, 3> const& b) {
  return NdArray<T, 3>{
      (a[1] * b[2] - a[2] * b[1]), (a[2] * b[0] - a[0] * b[2]), (a[0] * b[1] - a[1] * b[0])};
}

/**************************************************************************************************
  Triple Product
*/

template <typename T>
MOCHI_FORCE_INLINE constexpr T
TripleProduct(NdArray<T, 3> const& a, NdArray<T, 3> const& b, NdArray<T, 3> const& c) {
  static_assert(std::is_arithmetic_v<T>, "Arithmetic type required");
  return Dot(a, Cross(b, c));
}

/**************************************************************************************************
  Other Geometric Utilities
*/

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 2> OrthogonalVector(NdArray<T, 2> const& vec) {
  return NdArray<T, 2>{-vec[1], vec[0]};
}

template <typename T>
MOCHI_FORCE_INLINE constexpr NdArray<T, 3> OrthogonalVector(NdArray<T, 3> const& vec) {
  // Project out the coordinate of the minimum absolute value, then build the orthogonal vector in
  // the remaining 2D subspace.
  // NOTE: We use <= so that ties prefer the lower-index axis, matching the SIMD implementation
  // OrthogonalVector3(Simd<T, 4>).
  T const absX = Abs(vec[0]);
  T const absY = Abs(vec[1]);
  T const absZ = Abs(vec[2]);

  auto const xSmallest = (absX <= absY) && (absX <= absZ);
  auto const ySmallest = (absY <= absX) && (absY <= absZ);
  T const zero{0};
  return {
      Select(xSmallest, zero, Select(ySmallest, -vec[2], vec[1])),
      Select(xSmallest, vec[2], Select(ySmallest, zero, -vec[0])),
      Select(xSmallest, -vec[1], Select(ySmallest, vec[0], zero))};
}

/**************************************************************************************************
  IsFinite
*/

template <class T, int DN, size_t D0, size_t... DIMS>
MOCHI_FORCE_INLINE Simd<T, DN> VIsFinite(NdArray<Simd<T, DN>, D0, DIMS...> const& a) {
  static_assert(std::is_floating_point_v<T>, "VIsFinite only supports float or double");
  static_assert(D0 > 0, "VIsFinite requires non-empty arrays");
  auto isFinite = VIsFinite(a[0]);
  for (size_t i = 1; i < D0; ++i) {
    isFinite &= VIsFinite(a[i]);
  }
  return isFinite;
}

template <class T, int DN, size_t D0, size_t... DIMS>
MOCHI_FORCE_INLINE bool IsFinite(NdArray<Simd<T, DN>, D0, DIMS...> const& a) {
  if constexpr (std::is_integral_v<T>) {
    return true;
  } else {
    return AllTrue(VIsFinite(a));
  }
}

template <class T, size_t D0, size_t... DIMS>
MOCHI_FORCE_INLINE bool IsFinite(NdArray<T, D0, DIMS...> const& a) {
  static_assert(D0 > 0, "IsFinite requires non-empty arrays");
  bool isFinite = IsFinite(a[0]);
  for (size_t i = 1; i < D0; ++i) {
    isFinite &= IsFinite(a[i]);
  }
  return isFinite;
}

} // namespace mochi
