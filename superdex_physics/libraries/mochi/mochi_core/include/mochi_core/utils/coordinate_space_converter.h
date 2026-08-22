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

#include <mochi_core/utils/coordinate_space.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/transform_rt.h>

namespace mochi {

/** @brief Convert values between coordinate-space conventions and unit scales. */
class CoordinateSpaceConverter {
 public:
  /** @brief Construct an identity Default-to-Default converter. */
  CoordinateSpaceConverter();

  /**
   * @brief Construct a converter from @p fromSpace to @p toSpace.
   *
   * @pre Both spaces pass @ref CoordinateSpace::Validate.
   */
  CoordinateSpaceConverter(CoordinateSpace const& fromSpace, CoordinateSpace const& toSpace);

  /** @brief Return true when the conversion changes handedness. */
  [[nodiscard]] bool FlipsHandedness() const;

  /** @brief Return the output-units/input-units scale factor. */
  [[nodiscard]] real GetScale() const;

  /** @brief Return the unscaled direction change-of-basis matrix. */
  [[nodiscard]] Matrix3x3r const& GetDirectionMatrix() const;

  /** @brief Return the homogeneous point conversion matrix, including scale. */
  [[nodiscard]] Matrix4x4r const& GetTransformMatrix() const;

  /** @brief Convert one 3D translation vector to output space (includes scale). */
  template <typename T>
  [[nodiscard]] NdArray<T, 3> TranslationToOutput(NdArray<T, 3> const& value) const;

  /** @brief Convert one 3D direction vector to output space (no scale). */
  template <typename T>
  [[nodiscard]] NdArray<T, 3> DirectionToOutput(NdArray<T, 3> const& value) const;

  /**
   * @brief Convert flat, packed XYZ translation vectors in place (includes scale)
   *
   * @param[in,out] values Values to convert. Size must be a multiple of 3.
   * @param[in,out] error Error status.
   */
  template <typename T>
  void TranslationsToOutput(Span<T> values, Error& error) const;

  /**
   * @brief Convert flat, packed XYZ direction vectors in place (no scale).
   *
   * @param[in,out] values Values to convert. Its size must be a multiple of 3.
   * @param[in,out] error Error status.
   */
  template <typename T>
  void DirectionsToOutput(Span<T> values, Error& error) const;

  /** @brief Convert a rotation between coordinate-space conventions. */
  [[nodiscard]] Quaternion RotationToOutput(Quaternion const& value) const;

  /** @brief Convert a rigid transform, including unit scaling of translation. */
  [[nodiscard]] TransformRT TransformToOutput(TransformRT const& value) const;

  /**
   * @brief Convert a homogeneous transform, preserving arbitrary SRT content.
   *
   * @details Computes `C * value * inverse(C)`, where `C` is the point
   * conversion matrix including unit scale.
   */
  template <typename T>
  [[nodiscard]] NdArray<T, 4, 4> TransformToOutput(NdArray<T, 4, 4> const& value) const;

 private:
  Matrix3x3r _directionMatrix{};
  Matrix4x4r _transformMatrix{};
  Matrix4x4r _inverseTransformMatrix{};
  real _scale = 1_r;
  bool _flipsHandedness = false;
};

} // namespace mochi

#include "coordinate_space_converter_inl.h"
