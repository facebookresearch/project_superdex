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

#include <mochi_core/utils/transform_rt.h> // for intellisense

#include <limits>

namespace mochi {

/************************************************************************************
  TransformRT Inlines
*/

inline void TransformRT::WarnIfRotationNotNormalized() const {
  // Log a warning in debug builds if the rotation is not unit length. This does not use
  // MOCHI_ASSERT_VERBOSE because we don't want it to be fatal, especially in external code.
#if MOCHI_ASSERT_VERBOSE_ENABLED && MOCHI_LOG_ENABLED
  auto norm = Norm(_rotation);
  if (!NearEqual(norm, 1_r, kQuaternionTol)) MOCHI_UNLIKELY {
    MOCHI_LOG_WARNING(
        "TransformRT expects a unit length quaternion, but got [%g, %g, %g, %g] (magnitude %g).",
        _rotation.data[0],
        _rotation.data[1],
        _rotation.data[2],
        _rotation.data[3],
        norm);
  }
#endif
}

MOCHI_FORCE_INLINE TransformRT::TransformRT(Quaternion const& rotation, Vec4r translation)
    : _rotation(rotation), _translation(ToReal3(translation)) {
  WarnIfRotationNotNormalized();
}

MOCHI_FORCE_INLINE TransformRT::TransformRT(Quaternion const& rotation, Real3 const& translation)
    : _rotation(rotation), _translation(translation) {
  WarnIfRotationNotNormalized();
}

MOCHI_FORCE_INLINE TransformRT::TransformRT(Quaternion const& rotation) : _rotation(rotation) {
  WarnIfRotationNotNormalized();
}

MOCHI_FORCE_INLINE TransformRT::TransformRT(Vec4r translation)
    : _translation(ToReal3(translation)) {}

MOCHI_FORCE_INLINE TransformRT::TransformRT(Real3 const& translation) : _translation(translation) {}

MOCHI_FORCE_INLINE TransformRT TransformRT::Identity() {
  return TransformRT{};
}

inline TransformRT TransformRT::FromOrthoNormalTranspose(VMatrix4x4r const& mT) {
  return TransformRT{QuaternionFromMatrix(Transpose3x3(mT)), mT[3]};
}

MOCHI_FORCE_INLINE TransformRT TransformRT::FromOrthoNormal(VMatrix4x4r const& mat) {
  return FromOrthoNormalTranspose(Transpose4x4(mat));
}

MOCHI_FORCE_INLINE Quaternion const& TransformRT::GetRotation() const {
  return _rotation;
}

MOCHI_FORCE_INLINE void TransformRT::SetRotation(Quaternion const& rotation) {
  _rotation = rotation;
  WarnIfRotationNotNormalized();
}

MOCHI_FORCE_INLINE Vec4r TransformRT::VGetTranslation() const {
  return ToSimd(_translation, 1_r);
}

MOCHI_FORCE_INLINE Real3 TransformRT::GetTranslation() const {
  return _translation;
}

MOCHI_FORCE_INLINE void TransformRT::SetTranslation(Vec4r translation) {
  _translation = ToReal3(translation);
}

MOCHI_FORCE_INLINE void TransformRT::SetTranslation(Real3 translation) {
  _translation = translation;
}

MOCHI_FORCE_INLINE bool TransformRT::operator==(TransformRT const& rhs) const {
  return _rotation == rhs._rotation && _translation == rhs._translation;
}

MOCHI_FORCE_INLINE bool TransformRT::operator!=(TransformRT const& rhs) const {
  return !(*this == rhs);
}

MOCHI_FORCE_INLINE TransformRT TransformRT::operator*(TransformRT const& rhs) const {
  return TransformRT{
      this->GetRotation() * rhs.GetRotation(),
      this->VGetTranslation() + this->GetRotation() * rhs.VGetTranslation()};
}

MOCHI_FORCE_INLINE TransformRT& TransformRT::operator*=(TransformRT const& rhs) {
  *this = (*this) * rhs;
  return *this;
}

MOCHI_FORCE_INLINE Real3 TransformRT::TransformPoint(Real3 const& pt) const {
  return ToReal3(_rotation * ToSimd(pt, 0_r) + VGetTranslation());
}

MOCHI_FORCE_INLINE Real3 TransformRT::TransformDirection(Real3 const& dir) const {
  return ToReal3(_rotation * ToSimd(dir, 0_r));
}

MOCHI_FORCE_INLINE Real3 TransformRT::TransformPointInverse(Real3 const& pt) const {
  return ToReal3(Conjugate(_rotation) * (ToSimd(pt, 0_r) - VGetTranslation()));
}

MOCHI_FORCE_INLINE Real3 TransformRT::TransformDirectionInverse(Real3 const& dir) const {
  return ToReal3(Conjugate(_rotation) * ToSimd(dir, 0_r));
}

MOCHI_FORCE_INLINE Vec4r TransformRT::TransformPoint(Vec4r const& pt) const {
  return _rotation * ToSimdDirection(pt) + VGetTranslation();
}

MOCHI_FORCE_INLINE Vec4r TransformRT::TransformDirection(Vec4r const& dir) const {
  return _rotation * ToSimdDirection(dir);
}

MOCHI_FORCE_INLINE Vec4r TransformRT::TransformPointInverse(Vec4r const& pt) const {
  return ToSimdPoint(Conjugate(_rotation) * (pt - VGetTranslation()));
}

MOCHI_FORCE_INLINE Vec4r TransformRT::TransformDirectionInverse(Vec4r const& dir) const {
  return Conjugate(_rotation) * ToSimdDirection(dir);
}

/************************************************************************************
  Utilities
*/

MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4Transpose(TransformRT const& a) {
  Quaternion q = a.GetRotation();
  return VMatrix4x4r{
      q * Vec4r(1.0_r, 0.0_r, 0.0_r),
      q * Vec4r(0.0_r, 1.0_r, 0.0_r),
      q * Vec4r(0.0_r, 0.0_r, 1.0_r),
      a.VGetTranslation()};
}

MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4(TransformRT const& a) {
  return Transpose4x4(ToVMatrix4x4Transpose(a));
}

MOCHI_FORCE_INLINE bool NearEqual(TransformRT const& a, TransformRT const& b, Vec4r epsilon) {
  return NearEqual(a.GetRotation(), b.GetRotation(), epsilon) &&
      NearEqual(a.VGetTranslation(), b.VGetTranslation(), epsilon);
}

MOCHI_FORCE_INLINE bool NearEqual(TransformRT const& a, TransformRT const& b, real epsilon) {
  return NearEqual(a, b, Vec4r{epsilon});
}

MOCHI_FORCE_INLINE Vec4r VIsFinite(TransformRT const& a) {
  return VIsFinite(a.GetRotation().data) & VIsFinite(a.VGetTranslation());
}

MOCHI_FORCE_INLINE bool IsFinite(TransformRT const& a) {
  return AllTrue(VIsFinite(a));
}

MOCHI_FORCE_INLINE TransformRT Invert(TransformRT const& a) {
  Quaternion rotation = Conjugate(a.GetRotation());
  Vec4r translation = rotation * ToSimdDirection(-a.VGetTranslation());
  return TransformRT{rotation, translation};
}

MOCHI_FORCE_INLINE TransformRT NormalizeRotation(TransformRT const& a) {
  return TransformRT{Normalize(a.GetRotation()), a.VGetTranslation()};
}

MOCHI_FORCE_INLINE TransformRT Repivot(TransformRT const& transform, Real3 const& pivot) {
  return TransformRT(
      transform.GetRotation(),
      transform.GetTranslation() - pivot + transform.GetRotation() * pivot);
}

inline TransformRT Interpolate(TransformRT const& a, TransformRT const& b, real t) {
  auto trans = a.GetTranslation() * (1.0_r - t) + b.GetTranslation() * t;
  auto rot = Slerp(a.GetRotation(), b.GetRotation(), t);
  return TransformRT(rot, trans);
}

inline std::pair<Real3, TransformRT> DecomposeMatrixTransform(VMatrix4x4r const& matrixTransform) {
  // Extract translation and scale using transpose to access columns
  auto matrixT = Transpose4x4(matrixTransform);
  auto translation = ToReal3(matrixT[3]);
  auto scale = Real3{
      Norm<3>(matrixT[0]), // Column 0
      Norm<3>(matrixT[1]), // Column 1
      Norm<3>(matrixT[2]) // Column 2
  };

  // Calculate 3x3 determinant to detect negative scaling (reflection)
  auto determinant = Det3x3(matrixT);
  if (determinant < 0_r) {
    scale[0] = -scale[0]; // Flip the sign of any one axis.
  }

  // Normalize columns to extract rotation
  matrixT[0] /= scale[0] + std::numeric_limits<real>::min();
  matrixT[1] /= scale[1] + std::numeric_limits<real>::min();
  matrixT[2] /= scale[2] + std::numeric_limits<real>::min();

  // Zero scales leave their columns zero. Complete the rotation from the remaining columns.
  for (int i = 0; i < 3; ++i) {
    int const j = (i + 1) % 3;
    int const k = (i + 2) % 3;
    if (scale[i] == 0_r && scale[j] != 0_r && scale[k] != 0_r) {
      matrixT[i] = Cross3(matrixT[j], matrixT[k]);
    } else if (scale[i] != 0_r && scale[j] == 0_r && scale[k] == 0_r) {
      matrixT[j] = Normalize<3>(OrthogonalVector3(matrixT[i]));
      matrixT[k] = Cross3(matrixT[i], matrixT[j]);
    }
  }
  auto rotation = QuaternionFromMatrix(Transpose3x3(matrixT));

  return std::make_pair(scale, TransformRT{rotation, translation});
}

} // namespace mochi
