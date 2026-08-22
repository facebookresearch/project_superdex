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

#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/quaternion.h>

/**************************************************************************************************
 * Functions of Lie derivatives with respect to rotations.
 */

namespace mochi::lie {
// d(R * v)/dR = sk(- R * v).
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r DMultRotVecDRot(Vec4r const& multRotVec /* R * v */) {
  return Skew3(-multRotVec);
}

// d(R^T * v)/dR = R^T * sk(v) = sk(R^T * v) * R^T.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
DMultRotTVecDRot(VMatrix3x3r const& rotT /* R^T */, Vec4r const& multRotTVec /* R^T * v */) {
  return Dot3x3(Skew3(multRotTVec), rotT);
}

// va^T * d(R * vb)/dR = ((R * vb) x va)^T.
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r
MultVecaTDMultRotVecbDRot(Vec4r const& multRotVecb /* R * vb */, Vec4r const& veca /* va */) {
  return Cross3(multRotVecb, veca);
}

// d(M * R * v)/dR = M * sk(- R * v).
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
DMultMatRotVecDRot(VMatrix3x3r const& mat /* M */, Vec4r const& multRotVec /* R * v */) {
  return Dot3x3(mat, Skew3(-multRotVec));
}

// d(M * R^T * v)/dR = M * R^T * sk(v).
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
DMultMatRotTVecDRot(VMatrix3x3r const& multMatRotT /* M * R^T */, Vec4r const& vec /* v */) {
  return Dot3x3(multMatRotT, Skew3(vec));
}

// d(Ra * R * Rb)/dR = Ra.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
DMultRotaRotRotbDRot(VMatrix3x3r const& rota /* Ra */) {
  return rota;
}

// d(Ra * R^T * Rb)/dR = - Ra * R^T.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
DMultRotaRotTRotbDRot(VMatrix3x3r const& multRotaRotT /* Ra * R^T */) {
  return -multRotaRotT;
}

// d(tr(R * M))/dR = - 2 * sk^-1(R * M).
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r
DTrMultRotMatDRot(VMatrix3x3r const& multRotMat /* R * M */) {
  return (-2_r) * InvSkew3(multRotMat);
}

// d2(tr(R * M))/dR2 = sym(R * M) - tr(R * M) * eye.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
D2TrMultRotMatDRot2(VMatrix3x3r const& multRotMat /* R * M */) {
  auto multRotMatSym = 0.5_r * (multRotMat + Transpose3x3(multRotMat));
  return multRotMatSym - VDiagonalMatrix<3>(Trace3x3(multRotMat));
}

// d2(tr(Ra * M * Rb^T))/dRb/dRa = tr(Ra * M * Rb^T) * eye - (Ra * M * Rb^T)
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
D2TrMultRotaMatRotbTDRotbDRota(VMatrix3x3r const& multRotaMatRobT /* Ra * M * Rb^T */) {
  return VDiagonalMatrix<3>(Trace3x3(multRotaMatRobT)) - multRotaMatRobT;
}

// v^T * d2(R * u)/dR2 * w = 1/2 * v x (w x (R * u)) + 1/2 * v x (w x (R * u))
// The contractions with v and w are along the 2nd and 3rd dimensions of the tensor (the
// derivatives).
// This is also the result for d2(R(s, t) * u)/dvsdt, by chain-rule of d2(R * u)/dR2 with dR/ds = v
// and dR/dt = w, as long as R is linear in s and t.
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r MultVecbTD2MultRotVecaDRot2MultVecc(
    Vec4r const& multRotVeca /* R * u */,
    Vec4r const& vecb /* v */,
    Vec4r const& vecc) {
  return 0.5_r *
      (Cross3(vecb, Cross3(vecc, multRotVeca)) + Cross3(vecc, Cross3(vecb, multRotVeca)));
}

// axisa^T * d2(R * u)/dR2 * axisb = 1/2 * (axisa x (axisb x (R * u)) + axisb x (axisa x (R * u)))
// The contractions with axisa and axisb are along the 2nd and 3rd dimensions of the tensor (the
// derivatives).
// This is also the result for d2(R(axis) * u)/daxisadaxisb, by chain-rule of d2(R * u)/dR2 with
// dR/daxis = eye.
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r MultAxisaTD2MultRotVecDRot2MultAxisb(
    Vec4r const& multRotVec /* R * u */,
    int i /* axisa */,
    int j /* axisb */) {
  MOCHI_ASSERT_VERBOSE(i >= 0 && i < 3 && j >= 0 && j < 3);
  Real3 result{0_r, 0_r, 0_r};
  if (i != j) {
    result[i] = multRotVec[j];
    result[j] = multRotVec[i];
    return 0.5_r * ToSimd(result);
  } else {
    result[(i + 1) % 3] = -multRotVec[(i + 1) % 3];
    result[(i + 2) % 3] = -multRotVec[(i + 2) % 3];
    return ToSimd(result);
  }
}

//             | qw,  qz, -qy|
//             |-qz,  qw,  qx|
// dq/dR = 1/2 | qy, -qx,  qw|
//             |-qx, -qy, -qz|
// Given q = [qx, qy, qz, qw] stored in q.data (indices 0, 1, 2, 3)
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix4x3r DQuatDRot(Quaternion const& q) {
  Vec4r const& d = q.data;
  return VMatrix4x3r{
      0.5_r * Neg<false, false, true, false>(Shuffle<3, 2, 1, 0>(d)), // [ qw,  qz, -qy, ?]
      0.5_r * Neg<true, false, false, false>(Shuffle<2, 3, 0, 1>(d)), // [-qz,  qw,  qx, ?]
      0.5_r * Neg<false, true, false, false>(Shuffle<1, 0, 3, 2>(d)), // [ qy, -qx,  qw, ?]
      -0.5_r * d // [-qx, -qy, -qz, ?]
  };
}

//           | qw, -qz,  qy, -qx|
// dR/dq = 2 | qz,  qw, -qx, -qy|
//           |-qy,  qx,  qw, -qz|
// Given q = [qx, qy, qz, qw] stored in q.data (indices 0, 1, 2, 3)
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x4r DRotDQuat(Quaternion const& q) {
  Vec4r const& d = q.data;
  return VMatrix3x4r{
      2_r * Neg<false, true, false, true>(Shuffle<3, 2, 1, 0>(d)), // [ qw, -qz,  qy, -qx]
      2_r * Neg<false, false, true, true>(Shuffle<2, 3, 0, 1>(d)), // [ qz,  qw, -qx, -qy]
      2_r * Neg<true, false, false, true>(Shuffle<1, 0, 3, 2>(d)), // [-qy,  qx,  qw, -qz]
  };
}

/**************************************************************************************************
 * Functions for merit and derivatives of rotation differences. All derivatives wrt rotations are
 * expressed as Lie derivatives (i.e., derivatives in the local Lie algebra).
 */

// Returns Psi = 1/2 * tr((R - Q) * W * (R - Q)T) = 1/2 tr(W) + 1/2 tr(Q W QT) - tr(R * W * QT).
// It produces the same derivatives as tr(- R * W * QT), but it is more numerically robust,
// as it tends to zero when R = Q.
// Three overloads, with W a weight matrix, the diagonal weight vector, or a uniform weight.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE T WeightedRotationDifferenceMerit(
    NdArray<Simd<T, 4>, 3> const& rot,
    NdArray<Simd<T, 4>, 3> const& mat,
    NdArray<Simd<T, 4>, 3> const& w) {
  auto diff = rot - mat;
  return T(0.5) * Trace3x3(Dot3x3(Dot3x3(diff, w), Transpose3x3(diff)));
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE T WeightedRotationDifferenceMerit(
    NdArray<Simd<T, 4>, 3> const& rot,
    NdArray<Simd<T, 4>, 3> const& mat,
    Simd<T, 4> const& w) {
  auto diff = rot - mat;
  auto diff2 = diff * diff;
  return T(0.5) * HSum<3>(w * (diff2[0] + diff2[1] + diff2[2]));
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE T WeightedRotationDifferenceMerit(
    NdArray<Simd<T, 4>, 3> const& rot,
    NdArray<Simd<T, 4>, 3> const& mat,
    T w) {
  auto diff = rot - mat;
  auto diff2 = diff * diff;
  return T(0.5) * w * HSum<3>(diff2[0] + diff2[1] + diff2[2]);
}

// Returns M = -R * W * QT, useful for gradients and Hessians.
// Three overloads, with W a weight matrix, the diagonal weight vector, or a uniform weight.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r WeightedRotationDifferenceMatrix(
    VMatrix3x3r const& rot,
    VMatrix3x3r const& mat,
    VMatrix3x3r const& w) {
  return -Dot3x3(Dot3x3(rot, w), Transpose3x3(mat));
}

[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
WeightedRotationDifferenceMatrix(VMatrix3x3r const& rot, VMatrix3x3r const& mat, Vec4r const& w) {
  return Dot3x3(rot, Transpose3x3(mat * (-w)));
}

[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
WeightedRotationDifferenceMatrix(VMatrix3x3r const& rot, VMatrix3x3r const& mat, real w) {
  return (-w) * Dot3x3(rot, Transpose3x3(mat));
}

// Returns dPsi/dR, for Psi = tr(M) and M = R * M'.
// This function computes the gradient for WeightedRotationDifferenceMerit(), with R a rotation
// matrix. The argument mat can be built using WeightedRotationDifferenceMatrix().
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r WeightedRotationDifferenceGradient(VMatrix3x3r const& mat) {
  return DTrMultRotMatDRot(mat);
}

// Returns d2Psi/dR2, for Psi = tr(M) and M = R * M'.
// This function computes the Hessian for WeightedRotationDifferenceMerit(), with R a rotation
// matrix. The argument mat can be built using WeightedRotationDifferenceMatrix().
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
WeightedRotationDifferenceHessian(VMatrix3x3r const& mat) {
  return D2TrMultRotMatDRot2(mat);
}

// Returns d2Psi/dRbdRa, for Psi = tr(M) and M = Ra * M' * RbT.
// This function computes the Hessian for WeightedRotationDifferenceMerit(), when both Ra and Rb are
// rotation matrices. The argument mat can be built using WeightedRotationDifferenceMatrix().
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r
WeightedRotationDifferenceHessianMixed(VMatrix3x3r const& mat) {
  return D2TrMultRotaMatRotbTDRotbDRota(mat);
}
} // namespace mochi::lie
