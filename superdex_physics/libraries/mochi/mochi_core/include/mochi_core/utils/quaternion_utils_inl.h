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

#include "quaternion_utils.h" // For IntelliSense

namespace mochi {

/************************************************************************************
  Quaternion Utilities
*/

MOCHI_FORCE_INLINE bool NearEqual(Quaternion const& a, Quaternion const& b, Vec4r epsilon) {
  return NearEqual(a.data, b.data, epsilon);
}

MOCHI_FORCE_INLINE bool NearEqual(Quaternion const& a, Quaternion const& b, real epsilon) {
  return NearEqual(a, b, Vec4r{epsilon});
}

MOCHI_FORCE_INLINE bool
EquivalentRotation(Quaternion const& a, Quaternion const& b, Vec4r epsilon) {
  return NearEqual(a, b, epsilon) || NearEqual(a, -b, epsilon);
}

MOCHI_FORCE_INLINE bool EquivalentRotation(Quaternion const& a, Quaternion const& b, real epsilon) {
  return EquivalentRotation(a, b, Vec4r{epsilon});
}

MOCHI_FORCE_INLINE Quaternion Lerp(Quaternion a, Quaternion b, real t) {
  return Quaternion{Lerp(a.data, b.data, t)};
}

inline Quaternion Slerp(Quaternion a, Quaternion b, real t) {
  constexpr real kDotThreshold = 1e-6_r;
  real dot = Dot<4>(a.data, b.data);

  // If the dot product is negative, then they have opposite handedness
  // Invert it so it gives the shortest rotation direction
  if (dot < 0_r) {
    dot = -dot;
    b = -b;
  }

  // Use linear interpolation and normalize if the angle is small enough
  if (dot < 1_r - kDotThreshold)
    MOCHI_LIKELY {
      dot = Clamp(dot, -1.0_r, 1.0_r);
      real theta = ACos(dot) * (real)t;
      b.data = Normalize<4>(b.data - a.data * dot);
      return Quaternion{a.data * std::cos(theta) + b.data * std::sin(theta)};
    }
  else {
    return Lerp(a, b, t);
  }
}

MOCHI_FORCE_INLINE real NormSqr(Quaternion a) {
  return NormSqr<4>(a.data);
}
MOCHI_FORCE_INLINE real Norm(Quaternion a) {
  return Norm<4>(a.data);
}

MOCHI_FORCE_INLINE Quaternion Normalize(Quaternion a) {
  return Quaternion{Normalize<4>(a.data)};
}

MOCHI_FORCE_INLINE Quaternion Conjugate(Quaternion const& a) {
  return a.GetConjugate();
}

inline Quaternion QuaternionFromMatrix(VMatrix3x3r const& matrix, real eps) {
  return QuaternionFromMatrix(ToNdArray3x3(matrix), eps);
}

namespace details {
inline Real4 MatrixToAxisAngleImpl(Matrix3x3r const& matrix, real eps) {
  // Find axis and angle of rotation. Adapted from:
  //  https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/
  real t = 0.5_r * (matrix[0][0] + matrix[1][1] + matrix[2][2] - 1_r);
  real theta = ACos(Clamp(t, -1_r, 1_r));
  Real3 axis{};
  if (NearEqual(theta, 0_r))
    MOCHI_UNLIKELY {
      // Singularity at zero degree rotation. Any axis will do.
      theta = 0_r;
      axis = {1_r, 0_r, 0_r};
    }
  else if (NearEqual(Abs(theta), kPI, eps))
    MOCHI_UNLIKELY {
      // Singularity at +/- 180 degree rotation (sign does not matter). Compute the axis.
      theta = kPI;
      real xx = (matrix[0][0] + 1_r) * 0.5_r;
      real yy = (matrix[1][1] + 1_r) * 0.5_r;
      real zz = (matrix[2][2] + 1_r) * 0.5_r;
      real xy = (matrix[0][1] + matrix[1][0]) * 0.25_r;
      real xz = (matrix[0][2] + matrix[2][0]) * 0.25_r;
      real yz = (matrix[1][2] + matrix[2][1]) * 0.25_r;
      real constexpr kSqrt2Over2 = kSqrt2 * 0.5_r;
      if ((xx > yy) && (xx > zz)) { // matrix[0][0] is the largest diagonal term
        if (xx < eps) {
          axis = Real3{0_r, kSqrt2Over2, kSqrt2Over2};
        } else {
          real x = Sqrt(Clamp(xx, 0_r, 1_r));
          axis = Real3{x, xy / x, xz / x};
        }
      } else if (yy > zz) { // matrix[1][1] is the largest diagonal term
        if (yy < eps) {
          axis = Real3{kSqrt2Over2, 0_r, kSqrt2Over2};
        } else {
          real y = Sqrt(Clamp(yy, 0_r, 1_r));
          axis = Real3{xy / y, y, yz / y};
        }
      } else { // matrix[2][2] is the largest diagonal term so base result on this
        if (zz < eps) {
          axis = Real3{kSqrt2Over2, kSqrt2Over2, 0_r};
        } else {
          real z = Sqrt(Clamp(zz, 0_r, 1_r));
          axis = Real3{xz / z, yz / z, z};
        }
      }
    }
  else {
    // No singularity. This is the normal case.
    real scale = 0.5_r / Sin(theta);
    axis = {
        (matrix[2][1] - matrix[1][2]) * scale,
        (matrix[0][2] - matrix[2][0]) * scale,
        (matrix[1][0] - matrix[0][1]) * scale};
  }

  // store axis+angle as Real4 instead of as rotation vector for smaller numerical errors
  return Real4{axis[0], axis[1], axis[2], theta};
}
} // namespace details

inline Quaternion QuaternionFromMatrix(Matrix3x3r const& matrix, real eps) {
  Real4 const axisAngle = mochi::details::MatrixToAxisAngleImpl(matrix, eps);
  Real3 const axis{axisAngle[0], axisAngle[1], axisAngle[2]};
  real const angle = axisAngle[3];
  return Quaternion::FromAxisAngle(axis, angle);
}

MOCHI_FORCE_INLINE VMatrix3x3r ToVMatrix3x3Transpose(Quaternion const& q) {
  return VMatrix3x3r{q * Vec4r(1_r, 0_r, 0_r), q * Vec4r(0_r, 1_r, 0_r), q * Vec4r(0_r, 0_r, 1_r)};
}

MOCHI_FORCE_INLINE VMatrix3x3r ToVMatrix3x3(Quaternion const& q) {
  return Transpose3x3(ToVMatrix3x3Transpose(q));
}

MOCHI_FORCE_INLINE Matrix3x3r ToMatrix3x3(Quaternion const& q) {
  return ToNdArray3x3(ToVMatrix3x3(q));
}

MOCHI_FORCE_INLINE std::pair<VMatrix3x3r, VMatrix3x3r> ToVMatrix3x3_WithTranspose(
    Quaternion const& q) {
  auto matT = ToVMatrix3x3Transpose(q);
  return std::make_pair(Transpose3x3(matT), matT);
}

MOCHI_FORCE_INLINE Quaternion operator*(real a, Quaternion b) {
  return Quaternion{a * b.data};
}

MOCHI_FORCE_INLINE Quaternion operator*(Quaternion a, real b) {
  return Quaternion{a.data * b};
}

MOCHI_FORCE_INLINE Quaternion operator/(Quaternion a, real b) {
  return Quaternion{a.data / b};
}

MOCHI_FORCE_INLINE Vec4r VIsFinite(Quaternion const& q) {
  return VIsFinite(q.data);
}

MOCHI_FORCE_INLINE bool IsFinite(Quaternion const& q) {
  return AllTrue(VIsFinite(q));
}

} // namespace mochi
