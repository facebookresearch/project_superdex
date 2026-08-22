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

#include "quaternion.h" // For IntelliSense

namespace mochi {

/************************************************************************************
  Quaternion Inlines
*/
MOCHI_FORCE_INLINE Quaternion Quaternion::Identity() {
  return Quaternion{};
}

MOCHI_FORCE_INLINE Quaternion Quaternion::Zero() {
  return Quaternion{SimdZero()};
}

MOCHI_FORCE_INLINE Quaternion Quaternion::RotationX(real a) {
  return Quaternion{Sin(a * 0.5_r), 0_r, 0_r, Cos(a * 0.5_r)};
}

MOCHI_FORCE_INLINE Quaternion Quaternion::RotationY(real a) {
  return Quaternion{0_r, Sin(a * 0.5_r), 0_r, Cos(a * 0.5_r)};
}

MOCHI_FORCE_INLINE Quaternion Quaternion::RotationZ(real a) {
  return Quaternion{0_r, 0_r, Sin(a * 0.5_r), Cos(a * 0.5_r)};
}

MOCHI_FORCE_INLINE Real4 Quaternion::ToReal4() const {
  return Real4{data[0], data[1], data[2], data[3]};
}

inline Quaternion Quaternion::FromUnitAxisAngle(Vec4r normalizedAxis, real angleRadians) {
  // Log a warning in debug builds if the axis is not unit length (non-fatal unlike an assert).
#if MOCHI_ASSERT_VERBOSE_ENABLED && MOCHI_LOG_ENABLED
  auto norm = Norm<3>(normalizedAxis);
  if (!NearEqual(norm, 1_r, 20_r * kDefaultNearEqualEpsilon<real>))
    MOCHI_UNLIKELY {
      MOCHI_LOG_WARNING(
          "Quaternion::FromUnitAxisAngle expects a unit length vector, but got [%g, %g, %g] (magnitude %g).",
          normalizedAxis[0],
          normalizedAxis[1],
          normalizedAxis[2],
          norm);
    }
#endif
  // Use double precision to avoid rounding error in the trig functions.
  double c = Cos(static_cast<double>(angleRadians) * 0.5);
  double s = Sin(static_cast<double>(angleRadians) * 0.5);
  Vec4d v = Set<3>(StaticCast<Vec4d>(normalizedAxis) * s, c);
  return Quaternion{StaticCast<Vec4r>(v)};
}

inline Quaternion Quaternion::FromAxisAngle(Vec4r axis, real angleRadians) {
  return FromUnitAxisAngle(Normalize<3>(axis), angleRadians);
}

inline Quaternion Quaternion::FromAxisAngle(Real3 const& axis, real angleRadians) {
  return FromAxisAngle(Vec4r{axis[0], axis[1], axis[2], 0_r}, angleRadians);
}

inline Quaternion Quaternion::FromRotationVector(Vec4r rotVector) {
  Vec4r angle = VNorm<3>(rotVector);
  real rangle = Get0(angle);
  if (rangle > 1e-9_r)
    MOCHI_LIKELY {
      Vec4r axis = ToSimdDirection(rotVector) / angle;
      return FromUnitAxisAngle(axis, rangle);
    }
  else {
    // q = (xyz, w)
    // xyz = v/|v| * sin(|v|/2) ≈ v/|v| * |v|/2 = 0.5 * v
    // w = cos(|v|/2) ≈ 1 - |v|^2/8 ≈ 1
    return Quaternion(ToSimdPoint(0.5_r * rotVector));
  }
}

inline Quaternion Quaternion::FromRotationVector(Real3 const& rotVector) {
  Vec4r vrotVector(rotVector[0], rotVector[1], rotVector[2], 0.0_r);
  return FromRotationVector(vrotVector);
}

inline real Quaternion::GetAngleImpl(Quaternion& qPositiveNormalized, real& vectorMag) const {
  // We compute the angle using the part of the quaternion (the vector part (x, y, z) or the scalar
  // part w) that is smaller, as this provides higher precision.
  qPositiveNormalized = Quaternion{Normalize<4>(this->data)};
  // For simplicity, use always the quaternion with positive scalar part.
  if (qPositiveNormalized.data[3] < 0_r) {
    qPositiveNormalized.data = -qPositiveNormalized.data;
  }
  real w = qPositiveNormalized.data[3]; // Scalar part (always >= 0)
  vectorMag = Norm<3>(qPositiveNormalized.data); // Norm of the vector part
  // Note that vectorMag^2 + w^2 = 1. However, we do not compute vectorMag = sqrt(1 - w^2) to avoid
  // catastrophic cancellation when vectorMag is small but w = 1. This can often happen with single
  // precision for small angles.
  return 2_r * (w < vectorMag ? ACos(w) : ASin(vectorMag));
}

inline real Quaternion::GetAngle() const {
  Quaternion qPositiveNormalized MOCHI_NO_INIT; // discarded
  real vectorMag MOCHI_NO_INIT; // discarded
  return GetAngleImpl(qPositiveNormalized, vectorMag);
}

inline void Quaternion::ToAxisAngle(Real3* outAxis, real* outAngleRad) const {
  Quaternion qPositiveNormalized MOCHI_NO_INIT;
  real vectorMag MOCHI_NO_INIT;
  *outAngleRad = GetAngleImpl(qPositiveNormalized, vectorMag);
  auto axis =
      vectorMag > 1e-9_r ? (qPositiveNormalized.data / vectorMag) : Vec4r{1_r, 0_r, 0_r, 0_r};
  *outAxis = Real3{axis[0], axis[1], axis[2]};
}

inline void Quaternion::ToAxisAngle(Vec4r* outAxis, real* outAngleRad) const {
  Real3 outAxisTemp MOCHI_NO_INIT;
  this->ToAxisAngle(&outAxisTemp, outAngleRad);
  *outAxis = Vec4r{outAxisTemp[0], outAxisTemp[1], outAxisTemp[2], 0_r};
}

inline Vec4r Quaternion::VToRotationVector() const {
  Vec4r outAxis MOCHI_NO_INIT; // NOLINT(cppcoreguidelines-init-variables)
  real outAngle MOCHI_NO_INIT; // NOLINT(cppcoreguidelines-init-variables)
  this->ToAxisAngle(&outAxis, &outAngle);
  return outAxis * outAngle;
}

MOCHI_FORCE_INLINE Real3 Quaternion::ToRotationVector() const {
  auto r = VToRotationVector();
  return Real3{r[0], r[1], r[2]};
}

MOCHI_FORCE_INLINE bool Quaternion::operator==(Quaternion const& rhs) const {
  return data == rhs.data;
}

MOCHI_FORCE_INLINE bool Quaternion::operator!=(Quaternion const& rhs) const {
  return data != rhs.data;
}

MOCHI_FORCE_INLINE Quaternion Quaternion::operator*(Quaternion const& rhs) const {
  return Quaternion{MulAdd(
      Shuffle<0, 0, 0, 0>(this->data),
      Neg<false, true, false, true>(Shuffle<3, 2, 1, 0>(rhs.data)),
      MulAdd(
          Shuffle<1, 1, 1, 1>(this->data),
          Neg<false, false, true, true>(Shuffle<2, 3, 0, 1>(rhs.data)),
          MulAdd(
              Shuffle<2, 2, 2, 2>(this->data),
              Neg<true, false, false, true>(Shuffle<1, 0, 3, 2>(rhs.data)),
              Shuffle<3, 3, 3, 3>(this->data) * rhs.data)))};
}

MOCHI_FORCE_INLINE Quaternion Quaternion::operator-() const {
  return Quaternion{-this->data};
}

MOCHI_FORCE_INLINE Vec4r Quaternion::operator*(Vec4r v) const {
  // The canonical way to multiply this quaternion q by vector v is:
  //
  //    v' = q * v * conjugate(q)
  //
  // where the vector v is treated as a quaternion with w=0, but that is rather expensive since
  // quaternion multiplication is not very SIMD friendly (unless you do multiple pairs of
  // quaternions at once). Fortunately, there is a faster way:
  //
  //    t = 2 * cross(q.xyz, v)
  //    v' = v + q.w * t + cross(q.xyz, t)
  //
  // With the MSVC x64 compiler, this results in 27 total instructions vs 55 instructions for the
  // canonical approach. A derivation of the formula can be found here:
  // https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
  //
  Vec4r c = Cross3(this->data, v);
  Vec4r t = c + c;
  Vec4r qw = Broadcast<3>(this->data);
  return MulAdd(qw, t, v) + Cross3(this->data, t);
}

MOCHI_FORCE_INLINE Real3 Quaternion::operator*(Real3 const& v) const {
  auto r = *this * Vec4r{v[0], v[1], v[2], 0_r};
  return Real3{r[0], r[1], r[2]};
}

MOCHI_FORCE_INLINE Quaternion Quaternion::operator+(Quaternion const& a) const {
  return Quaternion{this->data + a.data};
}

MOCHI_FORCE_INLINE Quaternion Quaternion::operator-(Quaternion const& a) const {
  return Quaternion{this->data - a.data};
}

MOCHI_FORCE_INLINE Quaternion Quaternion::GetConjugate() const {
  return Quaternion{Neg<true, true, true, false>(data)};
}

} // namespace mochi

/************************************************************************************
  Reflection support for Quaternion.
  Serializes like std::array<real, 4>.
*/
#if MOCHI_USE_REFLECTION
template <>
struct SReflectTypeTraits<mochi::Quaternion> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_array;
  static SReflect::ArrayTypeInfo const& GetTypeInfo() {
    static auto const* s_typeInfo =
        SReflect::MakeFixedArrayTypeInfo<mochi::Quaternion, mochi::real, 4>("mochi::Quaternion");
    return *s_typeInfo;
  }
};
#endif // MOCHI_USE_REFLECTION
