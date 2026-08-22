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

#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_transform_rt.h> // for intellisense
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/transform_rt_utils.h>

namespace mochi {

/************************************************************************************
  MatrixTransformRT class inlines
*/

inline MatrixTransformRT::MatrixTransformRT(VMatrix3x3r const& rotation, Vec4r translation)
    : _rotation(
          ToSimdDirection(rotation[0]),
          ToSimdDirection(rotation[1]),
          ToSimdDirection(rotation[2])),
      _translation(ToSimdPoint(translation)) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  // Check for illegal scale or skew in the matrix
  VMatrix3x3r basis = Transpose3x3(_rotation);
  Real3 scale{Norm<3>(basis[0]), Norm<3>(basis[1]), Norm<3>(basis[2])};
  Real3 dot{Dot<3>(basis[0], basis[1]), Dot<3>(basis[1], basis[2]), Dot<3>(basis[2], basis[0])};
  real constexpr kCloseEnough = 1e-4_r;
  MOCHI_ASSERT_VERBOSE(
      NearEqual(Real3{1_r, 1_r, 1_r}, scale, kCloseEnough),
      "Non-unit scale detected in 3x3 rotation matrix");
  MOCHI_ASSERT_VERBOSE(
      NearEqual(Real3{0_r, 0_r, 0_r}, dot, kCloseEnough), "Skew detected in 3x3 rotation matrix");
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
}

MOCHI_FORCE_INLINE MatrixTransformRT::MatrixTransformRT(
    Matrix3x3r const& rotation,
    Real3 const& translation)
    : MatrixTransformRT(ToSimdMatrix(rotation), ToSimd(translation)) {}

MOCHI_FORCE_INLINE Vec4r MatrixTransformRT::TransformPoint(Vec4r pt) const {
  return ToSimdPoint(DotMatVec3x3(_rotation, pt) + _translation);
}

MOCHI_FORCE_INLINE Vec4r MatrixTransformRT::TransformDirection(Vec4r pt) const {
  return ToSimdDirection(DotMatVec3x3(_rotation, pt));
}

MOCHI_FORCE_INLINE Vec4r MatrixTransformRT::InverseTransformPoint(Vec4r pt) const {
  return ToSimdPoint(DotVecMat3x3(pt - _translation, _rotation));
}

MOCHI_FORCE_INLINE Vec4r MatrixTransformRT::InverseTransformDirection(Vec4r pt) const {
  return ToSimdDirection(DotVecMat3x3(pt, _rotation));
}

MOCHI_FORCE_INLINE Real3 MatrixTransformRT::TransformPoint(Real3 const& pt) const {
  return ToReal3(TransformPoint(ToSimd(pt)));
}

MOCHI_FORCE_INLINE Real3 MatrixTransformRT::TransformDirection(Real3 const& dir) const {
  return ToReal3(TransformDirection(ToSimd(dir)));
}

MOCHI_FORCE_INLINE Real3 MatrixTransformRT::InverseTransformPoint(Real3 const& pt) const {
  return ToReal3(InverseTransformPoint(ToSimd(pt)));
}

MOCHI_FORCE_INLINE Real3 MatrixTransformRT::InverseTransformDirection(Real3 const& dir) const {
  return ToReal3(InverseTransformDirection(ToSimd(dir)));
}

MOCHI_FORCE_INLINE MatrixTransformRT
MatrixTransformRT::operator*(MatrixTransformRT const& rhs) const {
  return MatrixTransformRT{
      Dot3x3(VGetRotation(), rhs.VGetRotation()), TransformPoint(rhs.VGetTranslation())};
}

/************************************************************************************
  MatrixTransformRT Utilities
*/

MOCHI_FORCE_INLINE MatrixTransformRT ToMatrixTransformRT(TransformRT const& a) {
  return MatrixTransformRT(VGetRotationMatrix(a), a.VGetTranslation());
}

MOCHI_FORCE_INLINE TransformRT ToTransformRT(MatrixTransformRT const& a) {
  return TransformRT(QuaternionFromMatrix(a.VGetRotation()), a.VGetTranslation());
}

MOCHI_FORCE_INLINE MatrixTransformRT
ToMatrixTransformRT(Quaternion const& rot, Real3 const& trans) {
  return ToMatrixTransformRT(TransformRT{rot, trans});
}

MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4Transpose(MatrixTransformRT const& a) {
  VMatrix3x3r const& rot = Transpose3x3(a.VGetRotation());
  return VMatrix4x4r{rot[0], rot[1], rot[2], a.VGetTranslation()};
}

MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4(MatrixTransformRT const& a) {
  // Pack the translation into the 4th column
  VMatrix3x3r const& r = a.VGetRotation();
  Vec4r t = a.VGetTranslation();
  return VMatrix4x4r{
      Blend<0, 0, 0, 1>(r[0], Broadcast<0>(t)),
      Blend<0, 0, 0, 1>(r[1], Broadcast<1>(t)),
      Blend<0, 0, 0, 1>(r[2], Broadcast<2>(t)),
      Vec4r(0_r, 0_r, 0_r, 1_r)};
}

MOCHI_FORCE_INLINE MatrixTransformRT Invert(MatrixTransformRT const& a) {
  // This assumes that the 3x3 rotation is an orthonormal matrix (as required by MatrixTransformRT).
  auto rot = a.VGetRotation();
  auto trans = a.VGetTranslation();
  return MatrixTransformRT{Transpose3x3(rot), DotVecMat3x3(-trans, rot)};
}

MOCHI_FORCE_INLINE bool
NearEqual(MatrixTransformRT const& a, MatrixTransformRT const& b, Vec4r epsilon) {
  return AllTrue<3>(
      VNearEqual<real, 3, 4>(a.VGetRotation(), b.VGetRotation(), epsilon) &
      VNearEqual(a.VGetTranslation(), b.VGetTranslation(), epsilon));
}

MOCHI_FORCE_INLINE bool
NearEqual(MatrixTransformRT const& a, MatrixTransformRT const& b, real epsilon) {
  return NearEqual(a, b, Vec4r{epsilon});
}

} // namespace mochi
