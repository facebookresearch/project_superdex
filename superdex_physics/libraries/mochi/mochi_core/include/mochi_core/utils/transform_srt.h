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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/transform_rt.h>

namespace mochi {

/**************************************************************************************************
  Affine Transform (scale, rotation, and translation)
*/
class TransformSRT {
  // Tolerance for unitary quaternion checks.
  [[maybe_unused]] static constexpr real kQuaternionTol = 20_r * kDefaultNearEqualEpsilon<real>;

 public:
  // Constructors
  TransformSRT() = default;
  TransformSRT(TransformRT const& rt); // Implicit conversion adds unit scale
  explicit TransformSRT(real scale);
  explicit TransformSRT(real scale, TransformRT const& rt);
  explicit TransformSRT(real scale, Quaternion const& rotation, Vec4r translation);
  explicit TransformSRT(real scale, Quaternion const& rotation, Real3 const& translation);

  [[nodiscard]] static TransformSRT Identity();

  // Scale
  [[nodiscard]] Vec4r VGetScale() const;
  [[nodiscard]] real GetScale() const;
  void SetScale(real scale);

  // Rotation
  [[nodiscard]] Quaternion const& GetRotation() const;
  void SetRotation(Quaternion const& rotation); // should be unit length

  // Translation
  [[nodiscard]] Vec4r VGetTranslation() const; // (x,y,z,1)
  [[nodiscard]] Real3 GetTranslation() const;
  void SetTranslation(Vec4r translation);
  void SetTranslation(Real3 translation);

  // Conversion to TransformRT (explicit because it discards scale)
  [[nodiscard]] TransformRT GetTransformRT() const;

  // Packed Translation & Scale as (x, y, z, s)
  [[nodiscard]] Vec4r VGetPackedTranslationAndScale() const;

  // Exact Equality
  [[nodiscard]] bool operator==(TransformSRT const& rhs) const;
  [[nodiscard]] bool operator!=(TransformSRT const& rhs) const;

  // Multiplication with another TransformRT concatenates transforms
  [[nodiscard]] TransformSRT operator*(TransformSRT const& rhs) const;
  TransformSRT& operator*=(TransformSRT const& rhs);

  // Multiplication with a 3-component vector must specify point or direction
  [[nodiscard]] Vec4r TransformPoint(Vec4r const& pt) const; // rotates & translates
  [[nodiscard]] Vec4r TransformDirection(Vec4r const& dir) const; // only rotates

  // Multiplication with a 3-component vector must specify point or direction
  [[nodiscard]] Real3 TransformPoint(Real3 const& pt) const; // rotates & translates
  [[nodiscard]] Real3 TransformDirection(Real3 const& dir) const; // only rotates

  // Jacobian of the transform wrt. the input point
  [[nodiscard]] Matrix3x3r Jacobian3x3() const;

 private:
  void WarnIfRotationNotNormalized() const;

  Quaternion _rotation = {};
  Real3 _translation = {};
  real _scale = 1_r;

 public:
  MOCHI_STRUCT_BEGIN(mochi::TransformSRT)
  MOCHI_FIELD_NAME(_scale, "scale")
  MOCHI_FIELD_NAME(_rotation, "rotation")
  MOCHI_FIELD_NAME(_translation, "translation")
  MOCHI_STRUCT_END()
};

/************************************************************************************
  Utilities
*/

// Compute a VMatrix4x4r from TransformRT. The resulting matrix can be used to transform vectors by
// calling DotMatVec4x4(VMatrix4x4r, Vec4r), which is equivalent to (TransformRT * Vec4r). However,
// if you have to transform multiple points, consider using ToVMatrix4x4Transpose to get the
// transpose, so that you can then use DotVecMat4x4 instead. DotVecMat4x4 is >40% faster with SIMD
// than DotMatVec4x4.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4(TransformSRT const& a);

// Equivalent to Transpose4x4(ToVMatrix4x4(a)) only faster.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4Transpose(TransformSRT const& a);

// NearEqual: (abs(a-b) <= epsilon)
[[nodiscard]] MOCHI_FORCE_INLINE bool NearEqual(
    TransformSRT const& a,
    TransformSRT const& b,
    real epsilon = kDefaultNearEqualEpsilon<real>);

// Return true if all values are finite
[[nodiscard]] MOCHI_FORCE_INLINE bool IsFinite(TransformSRT const& a);

// Invert TransformSRT (opposite scale, rotation and translation)
[[nodiscard]] MOCHI_FORCE_INLINE TransformSRT Invert(TransformSRT const& a);

// Interpolate
[[nodiscard]] TransformSRT Interpolate(TransformSRT const& a, TransformSRT const& b, real t);

// Translate
[[nodiscard]] MOCHI_FORCE_INLINE TransformSRT TranslateSRT(Vec4r translation);

} // namespace mochi

#include "transform_srt_inl.h"
