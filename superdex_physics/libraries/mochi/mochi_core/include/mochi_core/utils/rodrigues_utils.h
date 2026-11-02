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
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <limits>
#include <type_traits>

namespace mochi {

// Computes the Rodrigues rotation for some rotation vector p = axis * angle.
// Rodrigues(p) = exp(skew(p))
template <typename T>
[[nodiscard]] NdArray<Simd<T, 4>, 3> Rodrigues(Simd<T, 4> p) {
  // [p] skew-symmetric: [p][p]^T = -[p][p]
  // R = I + sinc(a)*[p] - .5*sinc^2(a/2)[p][p]^T

  NdArray<Simd<T, 4>, 3> S = Skew3(p);
  NdArray<Simd<T, 4>, 3> S2neg = Dot3x3(S, Transpose3x3(S));
  T angle = Get0(VNorm<3>(p));
  T sinchalf = Sinc(angle / T(2));

  return VEye<3, T>() + S * Sinc(angle) - T(0.5) * sinchalf * sinchalf * S2neg;
}

// Computes the rotation vector corresponding to a rotation matrix.
// InvRodrigues(R) = skewInv(log(R))
[[nodiscard]] Vec4r InvRodrigues(VMatrix3x3r const& R);

// Cap a rotation vector to a norm <= pi.
[[nodiscard]] Vec4r RotVectorPiCap(Vec4r rotVec);

[[nodiscard]] Real3 RotVectorPiCap(Real3 rotVec);

// Given a rotation r in rotation vector representation, an incremental change of the rotation
// can be expressed as R(theta) * R(r) = R(r + dr), for a differential rotation vector theta -> 0.
// The following functions compute the partial derivatives dtheta/dr, the left Jacobian J_l(r) of
// SO(3), and its inverse dr/dtheta. Both have the form eye + a sk(r) + b sk(r)^2, with
// sk(r)^2 = r rT - |r|^2 eye.

namespace drotvector {
// Below this |r|^2, a and b use truncated series instead of trigonometric functions. Scaled by |r|
// for sk(r) and |r|^2 for sk(r)^2, the truncation of the series and the cancellation in the closed
// forms are below rounding.
template <typename T>
inline constexpr T kSeriesSwitchNormSqr = std::is_same_v<T, float> ? T(0.04) : T(1e-3);
} // namespace drotvector

// dtheta/dr, with a = (1 - cos|r|) / |r|^2 and b = (|r| - sin|r|) / |r|^3.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DRotIncrementDRotVector(Simd<T, 4> r) {
  T const angleSqr = NormSqr<3>(r);
  T a MOCHI_NO_INIT;
  T b MOCHI_NO_INIT;
  if (angleSqr < drotvector::kSeriesSwitchNormSqr<T>) {
    a = T(1) / T(2) -
        angleSqr * (T(1) / T(24) - angleSqr * (T(1) / T(720) - angleSqr * (T(1) / T(40320))));
    b = T(1) / T(6) - angleSqr * (T(1) / T(120) - angleSqr * (T(1) / T(5040)));
  } else {
    T const angle = Sqrt(angleSqr);
    // 1 - cos|r| = 2 sin^2(|r| / 2), without cancellation.
    a = T(2) * Sqr(Sin(T(0.5) * angle)) / angleSqr;
    b = (angle - Sin(angle)) / (angleSqr * angle);
  }
  return VDiagonalMatrix<3>(T(1) - b * angleSqr) + a * Skew3(r) + b * Outer3(r, r);
}

// dr/dtheta, the inverse of dtheta/dr, with a = -1/2 and b = (1 - (|r| / 2) cot(|r| / 2)) / |r|^2.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DRotVectorDRotIncrement(Simd<T, 4> r) {
  T const angleSqr = NormSqr<3>(r);
  T b MOCHI_NO_INIT;
  if (angleSqr < drotvector::kSeriesSwitchNormSqr<T>) {
    b = T(1) / T(12) + angleSqr * (T(1) / T(720) + angleSqr * (T(1) / T(30240)));
  } else {
    T const halfAngle = T(0.5) * Sqrt(angleSqr);
    b = (T(1) - halfAngle / Tan(halfAngle)) / angleSqr;
  }
  return VDiagonalMatrix<3>(T(1) - b * angleSqr) - T(0.5) * Skew3(r) + b * Outer3(r, r);
}

// Given two rotations qa, qb, and a reference frame q0, in quaternion representation, computes the
// relative rotation between them, Rab, in the R0 frame. This results in Rb * R0 = Ra * R0 * Rab
// => Rab = R0' * Ra' * Rb * R0.
[[nodiscard]] Quaternion RelativeRotation_Reference(Quaternion qa, Quaternion qb, Quaternion q0);

} // namespace mochi
