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
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

#include <type_traits>

namespace mochi {

/**************************************************************************************************
  ArgMin, ArgMax
*/

// Returns the index of the lesser entry of an array.
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr size_t ArgMin(NdArray<T, N> const& a);

// Returns the index of the greater entry of an array.
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr size_t ArgMax(NdArray<T, N> const& a);

/**************************************************************************************************
  NearEqual: (abs(a-b) <= epsilon)
*/

template <typename T, typename Eps = T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr bool NearEqual(
    NdArray<T, D0, DIMS...> const& a,
    NdArray<T, D0, DIMS...> const& b,
    Eps epsilon = kDefaultNearEqualEpsilon<T>);

template <typename T, size_t D0, int D1>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, D1> VNearEqual(
    NdArray<Simd<T, D1>, D0> const& a,
    NdArray<Simd<T, D1>, D0> const& b,
    Simd<T, D1> epsilon = kDefaultNearEqualEpsilon<T>);

template <typename T, size_t D0, int D1>
[[nodiscard]] MOCHI_FORCE_INLINE bool NearEqual(
    NdArray<Simd<T, D1>, D0> const& a,
    NdArray<Simd<T, D1>, D0> const& b,
    T epsilon = kDefaultNearEqualEpsilon<T>);

/**************************************************************************************************
  Euclidean Norm (e.g. vector magnitude)
*/

template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T NormSqr(NdArray<T, N> const& a);

template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE T Norm(NdArray<T, N> const& a);

/**************************************************************************************************
  Basis vector
*/

template <typename T, int N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> BasisVector(int axis);

/**************************************************************************************************
  Normalize (make unit length)
*/

template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Normalize(NdArray<T, N> const& a);

template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Normalize(
    NdArray<T, N> const& a,
    T sqrNorm);

/**************************************************************************************************
  Sum, Prod and Mean
*/

// Sum of 1D array
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Sum(NdArray<T, N> const& a);

// Product of 1D array
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Prod(NdArray<T, N> const& a);

// Mean of 1D array
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Mean(NdArray<T, N> const& a);

// Max of 1D array
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Max(NdArray<T, N> const& a);

// Min of 1D array
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Min(NdArray<T, N> const& a);

/**************************************************************************************************
  Floor, ceil, round
*/

// Floor of 1D array
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Floor(NdArray<T, N> const& a);

// Ceil of 1D array
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Ceil(NdArray<T, N> const& a);

// Round of 1D array
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Round(NdArray<T, N> const& a);

// out[i] = Clamp(a[i], min, max)
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N>
Clamp(NdArray<T, N> const& a, T min, T max);

// out[i] = max(a[i], max)
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Max(NdArray<T, N> const& a, T max);

// out[i] = min(a[i], min)
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Min(NdArray<T, N> const& a, T min);

// out[i] = Clamp(a[i], min[i], max[i])
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N>
Clamp(NdArray<T, N> const& a, NdArray<T, N> const& min, NdArray<T, N> const& max);

// out[i] = Rect(a[i], min[i], max[i])
template <typename T, size_t N, bool MinInclusiveT, bool MaxInclusiveT>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Rect(
    NdArray<T, N> const& a,
    NdArray<T, N> const& min,
    NdArray<T, N> const& max,
    std::integral_constant<bool, MinInclusiveT> minInclusive = std::true_type{},
    std::integral_constant<bool, MaxInclusiveT> maxInclusive = std::false_type{});

// out[i] = max(a[i], max[i])
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Max(
    NdArray<T, N> const& a,
    NdArray<T, N> const& max);

// out[i] = min(a[i], min[i])
template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, N> Min(
    NdArray<T, N> const& a,
    NdArray<T, N> const& min);

/**************************************************************************************************
  Dot Product
*/

template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Dot(NdArray<T, N> const& a, NdArray<T, N> const& b);

template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Dot(Span<T const> a, NdArray<T, N> const& b);

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T Dot(Span<T const> a, Span<T const> b);

/**************************************************************************************************
  Cross Product
*/

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 3> Cross(
    NdArray<T, 3> const& a,
    NdArray<T, 3> const& b);

/**************************************************************************************************
  Triple Product
*/

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr T
TripleProduct(NdArray<T, 3> const& a, NdArray<T, 3> const& b, NdArray<T, 3> const& c);

/**************************************************************************************************
  Other Geometric Utilities
*/

// Builds an arbitrary vector that is orthogonal to the given one.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 2> OrthogonalVector(NdArray<T, 2> const& vec);

// Builds an arbitrary vector that is orthogonal to the given one.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE constexpr NdArray<T, 3> OrthogonalVector(NdArray<T, 3> const& vec);

/**************************************************************************************************
  IsFinite
*/

/// @brief Per-lane finiteness check on a SIMD-typed NdArray. Returns a per-lane SIMD mask.
///
/// @note Use when you need per-lane finiteness instead of the whole-array @ref IsFinite, which
/// collapses the result to a single bool.
template <class T, int DN, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE Simd<T, DN> VIsFinite(NdArray<Simd<T, DN>, D0, DIMS...> const& a);

// Return true if all elements of the NdArray are finite.
template <class T, int DN, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE bool IsFinite(NdArray<Simd<T, DN>, D0, DIMS...> const& a);

template <class T, size_t D0, size_t... DIMS>
[[nodiscard]] MOCHI_FORCE_INLINE bool IsFinite(NdArray<T, D0, DIMS...> const& a);

} // namespace mochi

#include "math_utils_inl.h"
