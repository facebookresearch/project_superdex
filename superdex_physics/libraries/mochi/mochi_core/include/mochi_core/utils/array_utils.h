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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_transform_rt.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/transform_srt.h>

#include <utility>
#include <vector>

namespace mochi {

/* dst[i] += valuesToAddToDst[i] */
template <typename T, typename SZ>
void ArrayPlusEquals(Span<T, SZ> dst, Span<T const, SZ> valuesToAddToDst);

/* dst[i] -= valuesToSubToDst[i] */
template <typename T, typename SZ>
void ArrayMinusEquals(Span<T, SZ> dst, Span<T const, SZ> valuesToSubToDst);

/* dst[i] *= valuesToMulToDst[i] */
template <typename T, typename SZ>
void ArrayMulEquals(Span<T, SZ> dst, Span<T const, SZ> valuesToMulToDst);

/* dst[i] *= alpha */
template <typename T, typename SZ>
void ArrayMulEquals(Span<T, SZ> dst, T alpha);

/* dst[i] = 1 / dst[i] */
template <typename T, typename SZ>
void ArrayInverts(Span<T, SZ> dst);

/* dst[i] = a[i] + b[i] */
template <typename T, typename SZ>
void ArrayAdd(Span<T, SZ> dst, Span<T const, SZ> a, Span<T const, SZ> b);

/* dst[i] = a[i] + b */
template <typename T>
void ArrayAdd(Span<NdArray<T, 3>> dst, Span<NdArray<T, 3> const> a, NdArray<T, 3> const& b);

/**
  For i in range 0 to (numValues - 1):
    dst[m[i]] += a[i]
*/
template <typename T>
inline void ArrayPlusEqualsIndexedDst(T* dst, T const* a, int const* m, int numValues) {
  for (int i = 0; i < numValues; ++i) {
    dst[m[i]] += a[i];
  }
}

/* dst[i] = a[i] - b[i] */
template <typename T, typename SZ>
void ArraySub(Span<T, SZ> dst, Span<T const, SZ> a, Span<T const, SZ> b);

/**
  ArrayGetTriples

  Args:
    src - The array to read from
    indices - The indices of the values to be copied (length == N)
    dst - The values to to be written (length == N)

  Requirements:
    * N is a multiple of 3
    * Every 3 indices are consecutive. E.g. {1, 2, 3, 12, 13, 14, 5, 6, 7}
*/
template <int N, typename T>
void ArrayGetTriples(T const* src, Span<int const> const& indices, T* dst);

/**
  ArrayAddTriples

  Args:
    vec - The array of values to be modified
    n - The number of values to modify
    indices - The indices of the values to be modified (length == n)
    values - The values to add with values in vec (length == n)

  Requirements:
    * n is a multiple of 3
    * Every 3 indices are consecutive. E.g. {1, 2, 3, 12, 13, 14, 5, 6, 7}
*/
template <typename T>
void ArrayAddTriples(T* vec, int n, int const* indices, T const* values);

/**
  ArrayAddTriplesN

  Like ArrayAddTriples except the number of values & indices is known at compile time.

  Performance for tetrahedral case (kNumValues == 12):
    About 20% faster than ArrayAddTriples
*/
template <int N, typename T>
void ArrayAddTriplesN(T* vec, Span<int const> const& indices, T const* values);

/**
  Transform an array of 3D positions using a TransformSRT such that:
    dst[i] = transform.TransformPoint(src[i])

  The src and dst arrays must have the same length.
  The src and dst arrays CAN be equal (modify in-place).
  If (kSingleThreaded == false), then the caller must link MochiCore for TaskScheduler support.

  See below for additional overloads
*/
template <bool kSingleThreaded = false, typename T>
void ArrayTransformPoints(
    Span<NdArray<T, 3>> dst,
    Span<NdArray<T, 3> const> src,
    TransformSRT const& transform);

/**
  Transform an array of 3D displacements, which are relative to an array of 3D reference
  coordinates, such that:
    dstDisplacements[i] = transform * (srcDisplacement[i] + refCoords[i]) - refCoords[i]

  All Spans must have the same length.
  The srcDisplacements and dstDisplacements arrays CAN be equal (modify in-place).
  If (kSingleThreaded == false), then the caller must link MochiCore for TaskScheduler support.

  See below for additional overloads
*/
template <bool kSingleThreaded = false, typename T>
void ArrayTransformDisplacements(
    Span<NdArray<T, 3>> dstDisplacements,
    Span<NdArray<T, 3> const> srcDisplacements,
    Span<NdArray<T, 3> const> refCoords,
    TransformRT const& transform);

// This overload takes a TRANSPOSED matrix
template <bool kSingleThreaded = false, typename T>
void ArrayTransformDisplacements_MatT(
    Span<NdArray<T, 3>> dstDisplacements,
    Span<NdArray<T, 3> const> srcDisplacements,
    Span<NdArray<T, 3> const> refCoords,
    NdArray<Simd<T, 4>, 4> const& matT);

// krylov vector overload of transposed version
template <bool kSingleThreaded = false, typename T>
void ArrayTransformDisplacements_MatT(
    ColumnVectorView<T> dstDisplacements,
    ColumnVectorView<T const> srcDisplacements,
    ColumnVectorView<T const> refCoordsFlat,
    NdArray<Simd<T, 4>, 4> const& matT);

/**
  Rotate an array of 3D vectors using a Quaternion such that:
    dst[i] = rotation * src[i];

  The src and dst arrays must have the same length.
  The src and dst arrays CAN be equal (modify in-place).
  If (kSingleThreaded == false), then the caller must link MochiCore for TaskScheduler support.

  See below for additional overloads
*/
template <bool kSingleThreaded = false, typename T>
void ArrayRotateVectors(
    Span<NdArray<T, 3>> dst,
    Span<NdArray<T, 3> const> src,
    Quaternion const& rotation);

// This overload takes a TRANSPOSED 3x3 rotation matrix.
// A 4x4 matrix is also allowed, but only the upper-left 3x3 will be used.
template <bool kSingleThreaded = false, typename T, size_t kNumRows>
  requires(kNumRows == 3 || kNumRows == 4)
void ArrayRotateVectors_MatT(
    Span<NdArray<T, 3>> dst,
    Span<NdArray<T, 3> const> src,
    NdArray<Simd<T, 4>, kNumRows> const& matT);

// This is the Vector overload with a transposed matrix.
// A 4x4 matrix is also allowed, but only the upper-left 3x3 will be used.
template <bool kSingleThreaded = false, typename T, size_t kNumRows>
  requires(kNumRows == 3 || kNumRows == 4)
void ArrayRotateVectors_MatT(
    ColumnVectorView<T> dst,
    ColumnVectorView<T const> src,
    NdArray<Simd<T, 4>, kNumRows> const& matT);

// Return the minimum value of a[i].
template <typename T, typename SZ>
[[nodiscard]] T Min(Span<T const, SZ> a);

// Return the maximum value of a[i].
template <typename T, typename SZ>
[[nodiscard]] T Max(Span<T const, SZ> a);

// Return the minimum and maximum value of a[i], as a pair.
template <typename T, typename SZ>
[[nodiscard]] std::pair<T, T> MinMax(Span<T const, SZ> a);

// Return the maximum value of abs(a[i])
template <typename T, typename SZ>
[[nodiscard]] T MaxAbs(Span<T const, SZ> a);

// Given two Span of equal length, return the max value of abs(a[i] - b[i]) for all i.
template <typename T, typename SZ>
[[nodiscard]] T MaxAbsDifference(Span<T const, SZ> a, Span<T const, SZ> b);

// Return the sum of a[i] (aka horizontal sum).
template <typename T, typename SZ>
[[nodiscard]] T HSum(Span<T, SZ> a);

// Sort and remove duplicates
template <typename ContainerT>
void SortAndRemoveDuplicates(ContainerT& v);

// Check if elements are unique
template <typename T, typename SZ>
[[nodiscard]] bool IsUnique(Span<T const, SZ> v);

/**
 * Return the permutation of indices that would sort the input array in ascending order.
 *
 * NOTE:
 * - Accepts a read-only Span to be "argsorted", and returns a DynamicArray of indices representing
 *   the sorted order of the array elements.
 * - The sort is not stable, i.e. the relative order of equal elements is not guaranteed to be
 *   preserved.
 * - The comparison is done using the default operator<.
 */
template <typename IndexType = size_t, typename T>
[[nodiscard]] DynamicArray<IndexType> ArgSort(Span<T const> v);

// Return true if all elements are finite.
template <class T, typename SizeT>
[[nodiscard]] bool IsFinite(Span<T, SizeT> const& a);

// Overwrite the destination array with copies of a single value.
// Faster than std::ranges::fill for trivially copyable types.
template <class T>
void Fill(Span<T> dst, T value);

} // namespace mochi

#include "array_utils_inl.h"
