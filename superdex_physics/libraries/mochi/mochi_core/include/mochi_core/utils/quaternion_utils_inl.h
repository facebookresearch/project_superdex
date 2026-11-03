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
  if (dot < 1_r - kDotThreshold) MOCHI_LIKELY {
    dot = Clamp(dot, -1.0_r, 1.0_r);
    real theta = ACos(dot) * (real)t;
    b.data = Normalize<4>(b.data - a.data * dot);
    return Quaternion{a.data * std::cos(theta) + b.data * std::sin(theta)};
  } else {
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

inline Quaternion QuaternionFromMatrix(VMatrix3x3r const& matrix) {
  return QuaternionFromMatrix(ToNdArray3x3(matrix));
}

inline Quaternion QuaternionFromMatrix(Matrix3x3r const& matrix) {
  // Shepperd's branches below agree only for an orthonormal matrix, so a matrix that is slightly
  // off, e.g. from rounding, would make the result jump where the branch changes. Gram-Schmidt on
  // the columns, keeping the first column's direction, removes that.
  Matrix3x3r columns = Transpose(matrix);
  columns[0] = Normalize(columns[0]);
  columns[1] = Normalize(columns[1] - Dot(columns[1], columns[0]) * columns[0]);
  columns[2] = Cross(columns[0], columns[1]);
  Matrix3x3r const rotation = Transpose(columns);

  // Shepperd's method. 4 w^2 = 1 + trace and 4 x^2 = 1 + 2 m00 - trace (likewise y and z), and the
  // off-diagonal pairs give the other products, e.g. m21 - m12 = 4 x w and m01 + m10 = 4 x y. Each
  // branch builds 4 c q for the component c with the largest 4 c^2, which is at least 1, and the
  // normalization below divides out 4 c.
  real const trace = rotation[0][0] + rotation[1][1] + rotation[2][2];
  real const diagMax = Max(rotation[0][0], rotation[1][1], rotation[2][2]);
  Real4 xyzw MOCHI_NO_INIT;
  if (trace >= diagMax) { // 4 w q
    xyzw = {
        rotation[2][1] - rotation[1][2],
        rotation[0][2] - rotation[2][0],
        rotation[1][0] - rotation[0][1],
        1_r + trace};
  } else if (rotation[0][0] == diagMax) { // 4 x q
    xyzw = {
        1_r + 2_r * rotation[0][0] - trace,
        rotation[0][1] + rotation[1][0],
        rotation[0][2] + rotation[2][0],
        rotation[2][1] - rotation[1][2]};
  } else if (rotation[1][1] == diagMax) { // 4 y q
    xyzw = {
        rotation[0][1] + rotation[1][0],
        1_r + 2_r * rotation[1][1] - trace,
        rotation[1][2] + rotation[2][1],
        rotation[0][2] - rotation[2][0]};
  } else { // 4 z q
    xyzw = {
        rotation[0][2] + rotation[2][0],
        rotation[1][2] + rotation[2][1],
        1_r + 2_r * rotation[2][2] - trace,
        rotation[1][0] - rotation[0][1]};
  }

  // w >= 0 as for an angle in [0, pi]
  Quaternion const q = Normalize(Quaternion{xyzw});
  return xyzw[3] < 0_r ? -q : q;
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
