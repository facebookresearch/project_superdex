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

#include <mochi_core/utils/rodrigues_utils.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <cmath>

namespace mochi {

static MOCHI_FORCE_INLINE Vec4r InvSkewR(VMatrix3x3r const& R) {
  // invSkewR = InvSkew3(R - Transpose3x3(R))
  return Vec4r{R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]};
}

Vec4r InvRodrigues(VMatrix3x3r const& R) {
  real const trace = Trace3x3(R);
  real const cosTheta = Clamp(0.5_r * (trace - 1_r), -1_r, 1_r);
  Vec4r const v = InvSkewR(R);

  // Threshold on cos(θ) for switching to the near-zero Taylor approximation used below
  // (0.5 + |v|^2 / 48 ≈ θ / (2·sin(θ))). Derivation: the approximation's Taylor expansion is
  // 0.5 + sin^2(θ) / 12 = 0.5 + θ^2 / 12 - θ^4 / 36 + O(θ^6); the true value is
  // 0.5 + θ^2 / 12 + 7θ^4 / 720 + O(θ^6); their difference is 3θ^4 / 80 + O(θ^6). Setting
  // 3θ^4 / 80 ≤ ε and using 1 - cos(θ) ≈ θ^2 / 2 gives cos(θ) ≥ 1 - Sqrt((80/3)·ε) / 2.
  constexpr real kNearZeroThreshold = MOCHI_USE_DOUBLE_PRECISION ? 0.99999997_r : 0.9991_r;

  if (cosTheta > kNearZeroThreshold) {
    // Angle is near 0 (trace is close to 3).
    //
    // To avoid 0/0 and catastrophic cancellation, use the Taylor expansion of f(θ) = θ / (2 *
    // sin(θ)) around θ = 0, i.e., f(θ) ≈ 0.5 + θ^2 / 12. Since |v|^2 = 4 * sin^2(θ) ≈ 4θ^2, this
    // yields f(θ) ≈ 0.5 + |v|^2 / 48.
    return (0.5_r + NormSqr<3>(v) / 48_r) * v;

  } else if (cosTheta < -kNearZeroThreshold) {
    // Angle is near π (trace is close to -1).
    //
    // θ ≈ π: R = 2·n·n^T − I. Pick the pivot p with the largest diagonal (= largest |n_p|²) and
    // recover the other two components from R[p][k] + R[k][p] = 4·n_p·n_k. A fixed pivot would lose
    // sign information when the axis is orthogonal to it (e.g. n = (0, 1/√2, −1/√2) with a pivot on
    // x).
    int p = 0;
    if (R[1][1] > R[p][p]) {
      p = 1;
    }
    if (R[2][2] > R[p][p]) {
      p = 2;
    }

    // The maximum diagonal element of a rotation matrix is bounded below by -1/3. Allowing a
    // generous tolerance for FP drift.
    MOCHI_ASSERT_VERBOSE(R[p][p] > -0.35_r, "Input matrix is severely outside SO(3).");

    // The symmetric part of R is R + R^T = 2cI + 2(1-c)n*n^T. We can extract an unnormalized axis
    // directly from the p-th column of R + R^T - 2cI. This avoids square roots, divisions, and is
    // exactly parallel to n for ANY c.
    real n[3];
    for (int k = 0; k < 3; ++k) {
      n[k] = (k == p) ? 2_r * (R[p][p] - cosTheta) : R[p][k] + R[k][p];
    }
    Vec4r axis{n[0], n[1], n[2]};

    // Sign resolution to match the regular branch's limit for θ → π⁻, preserving orientation
    // continuity.
    if (Dot<3>(axis, v) < 0_r) {
      axis = -axis;
    }

    // Norm + ATan2 is more expensive but more numerically stable than ACos, whose condition number
    // becomes unbounded as θ → π.
    real const sinTheta = 0.5_r * Norm<3>(v);
    real const theta = ATan2(sinTheta, cosTheta);
    MOCHI_ASSERT_VERBOSE(!NearZero(Norm<3>(axis)), "Input matrix is severely outside SO(3).");
    axis *= theta / Norm<3>(axis);
    return axis;

  } else {
    // Regular range.
    real const theta = ACos(cosTheta);
    real const twoSinTheta = 2_r * Sqrt(1_r - Sqr(cosTheta));
    return (theta / twoSinTheta) * v;
  }
}

Vec4r RotVectorPiCap(Vec4r rotVec) {
  real angle = Norm<3>(rotVec);
  if (angle > kPI) {
    real newAngle = std::fmod(angle + kPI, 2_r * kPI) - kPI;
    rotVec *= newAngle / angle;
  }
  return rotVec;
}

Real3 RotVectorPiCap(Real3 rotVec) {
  return ToReal3(RotVectorPiCap(ToSimd(rotVec)));
}

Quaternion RelativeRotation_Reference(Quaternion qa, Quaternion qb, Quaternion q0) {
  return Conjugate(q0) * Conjugate(qa) * qb * q0;
}

} // namespace mochi
