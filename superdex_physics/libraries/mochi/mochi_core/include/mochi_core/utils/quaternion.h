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
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/simd.h>

namespace mochi {

class Quaternion;

/**************************************************************************************************
  Quaternion - Used for 3D rotations
*/
class Quaternion {
 public:
  // Public data. Stored in (X,Y,Z,W) order.
  Vec4r data;

  // Default constructor is identity quaternion
  Quaternion() : data{0.0_r, 0.0_r, 0.0_r, 1.0_r} {}
  explicit Quaternion(Vec4r v) : data(v) {}
  explicit Quaternion(Real4 const& v) : data(v[0], v[1], v[2], v[3]) {}
  explicit Quaternion(real i, real j, real k, real r) : data(i, j, k, r) {}

  static Quaternion Identity();
  static Quaternion Zero();
  static Quaternion RotationX(real a);
  static Quaternion RotationY(real a);
  static Quaternion RotationZ(real a);

  // Construct a quaternion from an axis and angle.
  [[nodiscard]] static Quaternion FromUnitAxisAngle(
      Vec4r normalizedAxis,
      real angleRadians); // Axis must be unit length.
  [[nodiscard]] static Quaternion FromAxisAngle(Vec4r axis, real angleRadians);
  [[nodiscard]] static Quaternion FromAxisAngle(Real3 const& axis, real angleRadians);

  // Construct a quaternion from a rotation vector (Euler vector), r = axis * angle
  [[nodiscard]] static Quaternion FromRotationVector(Vec4r rotVector);
  [[nodiscard]] static Quaternion FromRotationVector(Real3 const& rotVector);

  // Get the rotation axis and angle for this Quaternion. The axis is unit length, and the angle is
  // in the range [-pi, pi]. Assumes unit length quaternion.
  void ToAxisAngle(Real3* outAxis, real* outAngleRad) const;
  void ToAxisAngle(Vec4r* outAxis, real* outAngleRad) const;

  // Gets the angle of this Quaternion. The angle is in the range [-pi, pi]. Assumes unit length
  // quaternion.
  [[nodiscard]] real GetAngle() const; // NOTE: Assumes quaternion is normalized

  // Get the rotation vector of this Quaternion. Assumes unit length quaternion.
  // The norm of the rotation vector is bounded by pi, which corresponds to an axis-angle with angle
  // in the range [-pi, pi]
  [[nodiscard]] Vec4r VToRotationVector() const;
  [[nodiscard]] Real3 ToRotationVector() const;

  // Get the data as a Real4
  [[nodiscard]] Real4 ToReal4() const;

  // Exact Equality
  [[nodiscard]] bool operator==(Quaternion const& rhs) const;
  [[nodiscard]] bool operator!=(Quaternion const& rhs) const;

  // Addition and subtraction
  [[nodiscard]] Quaternion operator+(Quaternion const& a) const;
  [[nodiscard]] Quaternion operator-(Quaternion const& a) const;

  // Unary negation (negates all 4 components)
  [[nodiscard]] Quaternion operator-() const;

  // Concatenate rotations in right-to-left order.
  // Quaternion q3 = q1 * q2; // is equivalent to "rotate by q2, then by q1"
  [[nodiscard]] Quaternion operator*(Quaternion const& rhs) const;

  // Rotate a vector by this quaternion. Requires that this is a unit quaternion.
  [[nodiscard]] Vec4r operator*(
      Vec4r v) const; // v[3] can be 0 (like a direction vector) or 1 (like a point)
  [[nodiscard]] Real3 operator*(Real3 const& v) const;

  // Get the quaternion conjugate
  [[nodiscard]] Quaternion GetConjugate() const;

 private:
  // Private helper functions:
  [[nodiscard]] real GetAngleImpl(Quaternion& qPositiveNormalized, real& vectorMag) const;
};

} // namespace mochi

#include "quaternion_inl.h"
