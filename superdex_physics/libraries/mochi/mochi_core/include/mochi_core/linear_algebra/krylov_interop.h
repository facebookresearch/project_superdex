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
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rigid_body_size.h>
#include <mochi_core/utils/task_scheduler.h>

#include <vector>

namespace mochi {

/**
  RowMatrixView <-- NdArray
*/
template <typename T, int kRows, int kCols>
auto AsView(NdArray<T, kRows, kCols>& arr) {
  return RowMatrixView<T, kRows, kCols>(arr.data()->data());
}
template <typename T, int kRows, int kCols>
auto AsView(NdArray<T, kRows, kCols> const& arr) {
  return RowMatrixView<T const, kRows, kCols>(arr.data()->data());
}
template <typename T, int kRows, int kCols>
auto AsConstView(NdArray<T, kRows, kCols> const& arr) {
  return RowMatrixView<T const, kRows, kCols>(arr.data()->data());
}

/**
  ColumnVectorView <-- NdArray
*/
template <typename T, size_t N>
auto AsView(NdArray<T, N>& vec) {
  return ColumnVectorView<T, N>{vec.data()};
}
template <typename T, size_t N>
auto AsView(NdArray<T, N> const& vec) {
  return ColumnVectorView<T const, N>{vec.data()};
}
template <typename T, size_t N>
auto AsConstView(NdArray<T, N> const& vec) {
  return ColumnVectorView<T const, N>{vec.data()};
}

/**
  ColumnVectorView <-- Span
*/
template <typename T, typename S>
auto AsView(Span<T, S> span) {
  return ColumnVectorView<T>{span.data(), isize(span)};
}
template <typename T, typename S>
auto AsConstView(Span<T, S> span) {
  return ColumnVectorView<T const>(span.data(), isize(span));
}

/**
  ColumnVectorView <-- std::vector
*/
template <typename T, typename Allocator>
auto AsView(std::vector<T, Allocator>& vec) {
  return ColumnVectorView<T>{vec.data(), isize(vec)};
}

template <typename T, typename Allocator>
auto AsView(std::vector<T, Allocator> const& vec) {
  return ColumnVectorView<T const>{vec.data(), isize(vec)};
}

template <typename T, typename Allocator>
auto AsConstView(std::vector<T, Allocator> const& vec) {
  return ColumnVectorView<T const>{vec.data(), isize(vec)};
}

/**
  ColumnVectorView <-- DynamicArray
*/
template <typename T>
auto AsView(DynamicArray<T>& vec) {
  return ColumnVectorView<T>{vec.data(), isize(vec)};
}

template <typename T>
auto AsView(DynamicArray<T> const& vec) {
  return ColumnVectorView<T const>{vec.data(), isize(vec)};
}

template <typename T>
auto AsConstView(DynamicArray<T> const& vec) {
  return ColumnVectorView<T const>{vec.data(), isize(vec)};
}

/**
  ColumnVector of raw rigid pose (using quaternion) <--> TransformRT
*/
inline void TransformToRawPose(
    TransformRT const& transform,
    ColumnVectorView<real, RigidSize::kAll> output) {
  Store(output.data(), transform.VGetTranslation());
  Store<RigidSize::kRot>(&output(RigidSize::kTrans), transform.GetRotation().data);
}

inline TransformRT TransformFromRawPose(ColumnVectorView<real const, RigidSize::kAll> input) {
  auto t = Load<Vec4r>(input.data());
  auto r = Load<RigidSize::kRot, Vec4r>(&input(RigidSize::kTrans));
  return TransformRT(Quaternion(r), t);
}

/**
  ColumnVector of raw rigid dofs (using rotation vector) <--> TransformRT
*/
inline void TransformToRawDofs(
    TransformRT const& transform,
    ColumnVectorView<real, RigidSize::kDAll> output) {
  Store(output.data(), transform.VGetTranslation());
  Store<RigidSize::kDRot>(&output(RigidSize::kTrans), transform.GetRotation().VToRotationVector());
}

inline TransformRT TransformFromRawDofs(ColumnVectorView<real const, RigidSize::kDAll> input) {
  auto t = Load<Vec4r>(input.data());
  auto r = Load<RigidSize::kDRot, Vec4r>(&input(RigidSize::kTrans));
  return TransformRT(Quaternion::FromRotationVector(r), t);
}

// Matrix view of a SIMD 3x3 matrix. Note that the SIMD matrix is an alias for NdArray<Vec4r, 3>,
// i.e. it is actually a 3x4 matrix where we use the first 3 values of each row and the last value
// of each row is unused padding. Also note that NdArray is row-major.
inline auto AsMatrixView(VMatrix3x3r const& v) {
  return RowMatrixView<real const, 3, 3, 4>{reinterpret_cast<real const*>(&v)};
}
inline auto AsMatrixView(VMatrix3x3r& v) {
  return RowMatrixView<real, 3, 3, 4>{reinterpret_cast<real*>(&v)};
}

// Matrix view of a SIMD 4x4 matrix. Note that SIMD 4x4 matrices are row-major.
inline auto AsMatrixView(VMatrix4x4r const& v) {
  return RowMatrixView<real const, 4, 4>{reinterpret_cast<real const*>(&v)};
}
inline auto AsMatrixView(VMatrix4x4r& v) {
  return RowMatrixView<real, 4, 4>{reinterpret_cast<real*>(&v)};
}

// Column vector view of a SIMD vector. The number of entries in the SIMD vector to take a view of
// is an optional template parameter. The default is to take a view of all the entries.
template <int kSize0 = -1, typename T, int N>
inline auto AsColumnVectorView(Simd<T, N> const& v) {
  constexpr int kSize = (kSize0 == -1) ? N : kSize0;
  static_assert(kSize >= 1 && kSize <= N, "Unsupported size");
  return ColumnVectorView<T const, kSize>{reinterpret_cast<T const*>(&v)};
}

template <int kSize0 = -1, typename T, int N>
inline auto AsColumnVectorView(Simd<T, N>& v) {
  constexpr int kSize = (kSize0 == -1) ? N : kSize0;
  static_assert(kSize >= 1 && kSize <= N, "Unsupported size");
  return ColumnVectorView<T, kSize>{reinterpret_cast<T*>(&v)};
}

} // namespace mochi
