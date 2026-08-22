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

#include <mochi_core/utils/transform_srt.h> // for intellisense

namespace mochi {

/************************************************************************************
  TransformRT Inlines
*/

inline void TransformSRT::WarnIfRotationNotNormalized() const {
  // Log a warning in debug builds if the rotation is not unit length. This does not use
  // MOCHI_ASSERT_VERBOSE because we don't want it to be fatal, especially in external code.
#if MOCHI_ASSERT_VERBOSE_ENABLED && MOCHI_LOG_ENABLED
  auto norm = Norm(_rotation);
  if (!NearEqual(norm, 1_r, kQuaternionTol))
    MOCHI_UNLIKELY {
      MOCHI_LOG_WARNING(
          "TransformSRT expects a unit length quaternion, but got [%g, %g, %g, %g] (magnitude %g).",
          _rotation.data[0],
          _rotation.data[1],
          _rotation.data[2],
          _rotation.data[3],
          norm);
    }
#endif
}

MOCHI_FORCE_INLINE TransformSRT::TransformSRT(real scale)
    : TransformSRT(scale, TransformRT::Identity()) {}

MOCHI_FORCE_INLINE TransformSRT::TransformSRT(TransformRT const& rt)
    : TransformSRT(1_r, rt.GetRotation(), rt.VGetTranslation()) {}

MOCHI_FORCE_INLINE TransformSRT::TransformSRT(real scale, TransformRT const& rt)
    : TransformSRT(scale, rt.GetRotation(), rt.VGetTranslation()) {}

MOCHI_FORCE_INLINE
TransformSRT::TransformSRT(real scale, Quaternion const& rotation, Vec4r translation)
    : TransformSRT(scale, rotation, ToReal3(translation)) {}

MOCHI_FORCE_INLINE
TransformSRT::TransformSRT(real scale, Quaternion const& rotation, Real3 const& translation)
    : _rotation(rotation), _translation(translation), _scale(scale) {
  WarnIfRotationNotNormalized();
}

MOCHI_FORCE_INLINE TransformSRT TransformSRT::Identity() {
  return TransformSRT{};
}

MOCHI_FORCE_INLINE Vec4r TransformSRT::VGetScale() const {
  return Vec4r{_scale};
}

MOCHI_FORCE_INLINE real TransformSRT::GetScale() const {
  return _scale;
}

MOCHI_FORCE_INLINE void TransformSRT::SetScale(real scale) {
  _scale = scale;
}

MOCHI_FORCE_INLINE Quaternion const& TransformSRT::GetRotation() const {
  return _rotation;
}

MOCHI_FORCE_INLINE void TransformSRT::SetRotation(Quaternion const& rotation) {
  _rotation = rotation;
  WarnIfRotationNotNormalized();
}

MOCHI_FORCE_INLINE Vec4r TransformSRT::VGetTranslation() const {
  return ToSimd(_translation, 1_r);
}

MOCHI_FORCE_INLINE Real3 TransformSRT::GetTranslation() const {
  return _translation;
}

MOCHI_FORCE_INLINE void TransformSRT::SetTranslation(Vec4r translation) {
  _translation = ToReal3(translation);
}

MOCHI_FORCE_INLINE void TransformSRT::SetTranslation(Real3 translation) {
  _translation = translation;
}

MOCHI_FORCE_INLINE TransformRT TransformSRT::GetTransformRT() const {
  return TransformRT{GetRotation(), VGetTranslation()};
}

MOCHI_FORCE_INLINE Vec4r TransformSRT::VGetPackedTranslationAndScale() const {
  return ToSimd(_translation, _scale);
}

MOCHI_FORCE_INLINE bool TransformSRT::operator==(TransformSRT const& rhs) const {
  return (_scale == rhs._scale) && (_rotation == rhs._rotation) &&
      (_translation == rhs._translation);
}

MOCHI_FORCE_INLINE bool TransformSRT::operator!=(TransformSRT const& rhs) const {
  return !(*this == rhs);
}

MOCHI_FORCE_INLINE TransformSRT TransformSRT::operator*(TransformSRT const& rhs) const {
  return TransformSRT{
      this->GetScale() * rhs.GetScale(),
      this->GetRotation() * rhs.GetRotation(),
      this->TransformPoint(rhs.VGetTranslation())};
}

MOCHI_FORCE_INLINE TransformSRT& TransformSRT::operator*=(TransformSRT const& rhs) {
  *this = (*this) * rhs;
  return *this;
}

MOCHI_FORCE_INLINE Vec4r TransformSRT::TransformDirection(Vec4r const& dir) const {
  return _rotation * ToSimdDirection(dir);
}

MOCHI_FORCE_INLINE Real3 TransformSRT::TransformDirection(Real3 const& dir) const {
  return ToReal3(_rotation * ToSimd(dir, 0_r));
}

MOCHI_FORCE_INLINE Vec4r TransformSRT::TransformPoint(Vec4r const& pt) const {
  return ToSimdPoint(_rotation * ToSimdDirection(pt * _scale) + VGetTranslation());
}

MOCHI_FORCE_INLINE Real3 TransformSRT::TransformPoint(Real3 const& pt) const {
  return ToReal3(TransformPoint(ToSimd(pt)));
}

MOCHI_FORCE_INLINE Matrix3x3r TransformSRT::Jacobian3x3() const {
  auto mat = ToMatrix3x3(GetRotation());
  return GetScale() * mat;
}

/************************************************************************************
  Utilities
*/

MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4Transpose(TransformSRT const& a) {
  real s = a.GetScale();
  Quaternion q = a.GetRotation();
  return VMatrix4x4r{
      q * Vec4r(s, 0.0_r, 0.0_r),
      q * Vec4r(0.0_r, s, 0.0_r),
      q * Vec4r(0.0_r, 0.0_r, s),
      a.VGetTranslation()};
}

MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4(TransformSRT const& a) {
  return Transpose4x4(ToVMatrix4x4Transpose(a));
}

MOCHI_FORCE_INLINE bool NearEqual(TransformSRT const& a, TransformSRT const& b, Vec4r epsilon) {
  return NearEqual(a.GetRotation(), b.GetRotation(), epsilon) &&
      NearEqual(a.VGetPackedTranslationAndScale(), b.VGetPackedTranslationAndScale(), epsilon);
}

MOCHI_FORCE_INLINE bool NearEqual(TransformSRT const& a, TransformSRT const& b, real epsilon) {
  return NearEqual(a, b, Vec4r{epsilon});
}

MOCHI_FORCE_INLINE Vec4r VIsFinite(TransformSRT const& a) {
  return VIsFinite(a.GetRotation()) & VIsFinite(a.VGetPackedTranslationAndScale());
}

MOCHI_FORCE_INLINE bool IsFinite(TransformSRT const& a) {
  return AllTrue(VIsFinite(a));
}

MOCHI_FORCE_INLINE TransformSRT Invert(TransformSRT const& a) {
  real scale = 1_r / a.GetScale();
  Quaternion rotation = a.GetRotation().GetConjugate();
  Vec4r translation = rotation * ToSimdDirection(-scale * a.VGetTranslation());
  return TransformSRT{scale, rotation, translation};
}

inline TransformSRT Interpolate(TransformSRT const& a, TransformSRT const& b, real t) {
  auto trans = a.GetTranslation() * (1.0_r - t) + b.GetTranslation() * t;
  auto rot = Slerp(a.GetRotation(), b.GetRotation(), t);
  auto s = a.GetScale() * (1.0_r - t) + b.GetScale() * t;
  return TransformSRT(s, rot, trans);
}

MOCHI_FORCE_INLINE TransformSRT TranslateSRT(Vec4r translation) {
  return TransformSRT(1.0_r, Quaternion::Identity(), translation);
}

} // namespace mochi
