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

#include "coordinate_space_converter.h" // Reverse include for Intellisense

#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/vmatrix.h>

#include <type_traits>

namespace mochi {

namespace details {

/**
 * @brief Extract the semantic direction (eg. @ref CoordinateSpaceAxes::Right), as an integer.
 *
 * @param[in] axes @ref CoordinateSpaceAxes value
 * @param[in] axisIndex 0 for X, 1 for Y, 2 for Z.
 */
[[nodiscard]] constexpr int GetAxisDirection(CoordinateSpaceAxes axes, int axisIndex) {
  return static_cast<int>((static_cast<uint16_t>(axes) >> (3 * axisIndex)) & 0x7);
}

} // namespace details

inline CoordinateSpaceConverter::CoordinateSpaceConverter()
    : CoordinateSpaceConverter(CoordinateSpace::Default(), CoordinateSpace::Default()) {}

inline CoordinateSpaceConverter::CoordinateSpaceConverter(
    CoordinateSpace const& fromSpace,
    CoordinateSpace const& toSpace) {
  fromSpace.Validate(ErrorAssert{});
  toSpace.Validate(ErrorAssert{});

  auto basisMatrix = [](CoordinateSpaceAxes axes) {
    constexpr Real3 kDirectionVectors[] = {
        {1_r, 0_r, 0_r}, // Right
        {-1_r, 0_r, 0_r}, // Left
        {0_r, 1_r, 0_r}, // Up
        {0_r, -1_r, 0_r}, // Down
        {0_r, 0_r, 1_r}, // Forward
        {0_r, 0_r, -1_r}, // Backward
    };
    Matrix3x3r basis{};
    for (int col = 0; col < 3; ++col) {
      int const dir = details::GetAxisDirection(axes, col);
      MOCHI_ASSERT(dir >= 0 && dir < isize(kDirectionVectors));
      Real3 const& v = kDirectionVectors[dir];
      for (int row = 0; row < 3; ++row) {
        basis[row][col] = v[row];
      }
    }
    return basis;
  };

  _directionMatrix = Dot(Transpose(basisMatrix(toSpace.axes)), basisMatrix(fromSpace.axes));
  _scale = toSpace.unitsPerMeter / fromSpace.unitsPerMeter;
  _flipsHandedness = Det(_directionMatrix) < 0_r;

  // The direction matrix is a signed permutation, so its transpose is its exact inverse.
  real const invScale = 1_r / _scale;
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      _transformMatrix[row][col] = _scale * _directionMatrix[row][col];
      _inverseTransformMatrix[row][col] = invScale * _directionMatrix[col][row];
    }
  }
  _transformMatrix[3][3] = 1_r;
  _inverseTransformMatrix[3][3] = 1_r;
}

inline bool CoordinateSpaceConverter::FlipsHandedness() const {
  return _flipsHandedness;
}

inline real CoordinateSpaceConverter::GetScale() const {
  return _scale;
}

inline Matrix3x3r const& CoordinateSpaceConverter::GetDirectionMatrix() const {
  return _directionMatrix;
}

inline Matrix4x4r const& CoordinateSpaceConverter::GetTransformMatrix() const {
  return _transformMatrix;
}

template <typename T>
NdArray<T, 3> CoordinateSpaceConverter::TranslationToOutput(NdArray<T, 3> const& value) const {
  static_assert(std::is_floating_point_v<T>);
  return DirectionToOutput(value) * static_cast<T>(_scale);
}

template <typename T>
NdArray<T, 3> CoordinateSpaceConverter::DirectionToOutput(NdArray<T, 3> const& value) const {
  static_assert(std::is_floating_point_v<T>);
  return DotMatVec(StaticCast<NdArray<T, 3, 3>>(_directionMatrix), value);
}

template <typename T>
void CoordinateSpaceConverter::TranslationsToOutput(Span<T> values, Error& error) const {
  static_assert(std::is_floating_point_v<T>);
  MOCHI_ERROR_IF(values.size() % 3 != 0, error, "Expected 3 values per input vector (XYZ).");
  MOCHI_ERROR_RETURN(error);

  ArrayTransformPoints_MatT(
      Unflatten<NdArray<T, 3>>(values),
      Unflatten<NdArray<T, 3> const>(values),
      StaticCast<NdArray<Simd<T, 4>, 4>>(Transpose4x4(ToSimdMatrix(_transformMatrix))));
}

template <typename T>
void CoordinateSpaceConverter::DirectionsToOutput(Span<T> values, Error& error) const {
  static_assert(std::is_floating_point_v<T>);
  MOCHI_ERROR_IF(values.size() % 3 != 0, error, "Expected 3 values per input vector (XYZ).");
  MOCHI_ERROR_RETURN(error);

  ArrayRotateVectors_MatT(
      Unflatten<NdArray<T, 3>>(values),
      Unflatten<NdArray<T, 3> const>(values),
      StaticCast<NdArray<Simd<T, 4>, 3>>(Transpose3x3(ToSimdMatrix(_directionMatrix))));
}

inline Quaternion CoordinateSpaceConverter::RotationToOutput(Quaternion const& value) const {
  Real4 const q = value.ToReal4();
  Real3 axis{q[0], q[1], q[2]};
  if (_flipsHandedness) {
    axis = -axis;
  }
  Real3 const converted = DirectionToOutput(axis);
  return Quaternion{converted[0], converted[1], converted[2], q[3]};
}

inline TransformRT CoordinateSpaceConverter::TransformToOutput(TransformRT const& value) const {
  return TransformRT{
      RotationToOutput(value.GetRotation()), TranslationToOutput(value.GetTranslation())};
}

template <typename T>
NdArray<T, 4, 4> CoordinateSpaceConverter::TransformToOutput(NdArray<T, 4, 4> const& value) const {
  static_assert(std::is_floating_point_v<T>);
  return Dot(
      Dot(StaticCast<NdArray<T, 4, 4>>(_transformMatrix), value),
      StaticCast<NdArray<T, 4, 4>>(_inverseTransformMatrix));
}

} // namespace mochi
