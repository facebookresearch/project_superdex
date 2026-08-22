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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <utility>

namespace mochi {

/**************************************************************************************************
  Affine Transform (rotation & translation, but no scale)
*/
class TransformRT {
  // Tolerance for unitary quaternion checks.
  [[maybe_unused]] static constexpr real kQuaternionTol = 50_r * kDefaultNearEqualEpsilon<real>;

 public:
  // Constructors
  TransformRT() = default;
  explicit TransformRT(Quaternion const& rotation, Vec4r translation);
  explicit TransformRT(Quaternion const& rotation, Real3 const& translation);
  explicit TransformRT(Quaternion const& rotation);
  explicit TransformRT(Vec4r translation);
  explicit TransformRT(Real3 const& translation);

  [[nodiscard]] static TransformRT Identity();
  [[nodiscard]] static TransformRT FromOrthoNormal(VMatrix4x4r const& m);
  [[nodiscard]] static TransformRT FromOrthoNormalTranspose(VMatrix4x4r const& mT);

  // Rotation
  [[nodiscard]] Quaternion const& GetRotation() const;
  void SetRotation(Quaternion const& rotation); // should be unit length

  // Translation
  [[nodiscard]] Real3 GetTranslation() const;
  [[nodiscard]] Vec4r VGetTranslation() const; // (x,y,z,1)
  void SetTranslation(Real3 translation);
  void SetTranslation(Vec4r translation);

  // Exact Equality
  [[nodiscard]] bool operator==(TransformRT const& rhs) const;
  [[nodiscard]] bool operator!=(TransformRT const& rhs) const;

  // Multiplication with another TransformRT concatenates transforms
  [[nodiscard]] TransformRT operator*(TransformRT const& rhs) const;
  TransformRT& operator*=(TransformRT const& rhs);

  // Utilities to apply the transform to a point or a direction.
  [[nodiscard]] Real3 TransformPoint(Real3 const& pt) const; // rotates & translates
  [[nodiscard]] Real3 TransformDirection(Real3 const& dir) const; // only rotates

  [[nodiscard]] Vec4r TransformPoint(Vec4r const& pt) const; // rotates & translates
  [[nodiscard]] Vec4r TransformDirection(Vec4r const& dir) const; // only rotates

  // Utilities to apply the inverse transform to a point or a direction.
  // WARNING: These utilities are slow. To invert multiple points or directions, consider
  // precomputing the inverse transform as VMatrix and applying it to all points or directions.
  [[nodiscard]] Real3 TransformPointInverse(Real3 const& pt) const; // rotates & translates
  [[nodiscard]] Real3 TransformDirectionInverse(Real3 const& dir) const; // only rotates

  [[nodiscard]] Vec4r TransformPointInverse(Vec4r const& pt) const; // rotates & translates
  [[nodiscard]] Vec4r TransformDirectionInverse(Vec4r const& dir) const; // only rotates

 private:
  void WarnIfRotationNotNormalized() const;

  Quaternion _rotation = {};
  Real3 _translation{0_r, 0_r, 0_r};

 public:
  MOCHI_STRUCT_BEGIN(mochi::TransformRT)
  MOCHI_FIELD_NAME(_rotation, "rotation")
  MOCHI_FIELD_NAME(_translation, "translation")
  MOCHI_STRUCT_END()
};

/************************************************************************************
  Utilities
*/

// Compute a Matrix4x4r from TransformRT. The resulting matrix can be used to transform vectors by
// calling DotMatVec4x4(VMatrix4x4r, Vec4r), which is equivalent to (TransformRT * Vec4r). However,
// if you have to transform multiple points, consider using ToVMatrix4x4Transpose to get the
// transpose, so that you can then use DotVecMat4x4 instead. DotVecMat4x4 is >40% faster with SIMD
// than DotMatVec4x4.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4(TransformRT const& a);

// Equivalent to Transpose4x4(ToVMatrix4x4(a)) only faster.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4Transpose(TransformRT const& a);

// Take a 4x4 matrix consisting of scale (possibly non-uniform, possibly negative), rotation, and
// translation. Return the scale and TransformRT that could be re-combined to form the same matrix.
std::pair<Real3, TransformRT> DecomposeMatrixTransform(VMatrix4x4r const& matrixTransform);

// NearEqual: (abs(a-b) <= epsilon)
[[nodiscard]] MOCHI_FORCE_INLINE bool NearEqual(
    TransformRT const& a,
    TransformRT const& b,
    real epsilon = kDefaultNearEqualEpsilon<real>);

// IsFinite returns true if all values are finite
[[nodiscard]] MOCHI_FORCE_INLINE bool IsFinite(TransformRT const& a);

// Invert TransformRT (opposite rotation & translation)
[[nodiscard]] MOCHI_FORCE_INLINE TransformRT Invert(TransformRT const& a);

// Return a TransformRT with a normalized (unit length) rotation Quaternion.
[[nodiscard]] MOCHI_FORCE_INLINE TransformRT NormalizeRotation(TransformRT const& a);

// Repivot a transform such that calling the returned transform with the given pivot produces the
// same results.
[[nodiscard]] MOCHI_FORCE_INLINE TransformRT
Repivot(TransformRT const& transform, Real3 const& pivot);

// Interpolate
[[nodiscard]] TransformRT Interpolate(TransformRT const& a, TransformRT const& b, real t);

} // namespace mochi

#include "transform_rt_inl.h"
