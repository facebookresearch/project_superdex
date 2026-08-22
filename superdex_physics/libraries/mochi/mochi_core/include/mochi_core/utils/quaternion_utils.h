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
#include <mochi_core/utils/vmatrix.h>

#include <utility>

namespace mochi {

/************************************************************************************
  Quaternion Utilities
*/

// (abs(a-b) <= epsilon)
[[nodiscard]] MOCHI_FORCE_INLINE bool
NearEqual(Quaternion const& a, Quaternion const& b, Vec4r epsilon);
[[nodiscard]] MOCHI_FORCE_INLINE bool
NearEqual(Quaternion const& a, Quaternion const& b, real epsilon = kDefaultNearEqualEpsilon<real>);

// Similar to NearEqual but EquivalentRotation(q, -q) is also true
[[nodiscard]] MOCHI_FORCE_INLINE bool
EquivalentRotation(Quaternion const& a, Quaternion const& b, Vec4r epsilon);
[[nodiscard]] MOCHI_FORCE_INLINE bool EquivalentRotation(
    Quaternion const& a,
    Quaternion const& b,
    real epsilon = kDefaultNearEqualEpsilon<real>);

// Scalar multiplication
[[nodiscard]] MOCHI_FORCE_INLINE Quaternion operator*(real a, Quaternion b);
[[nodiscard]] MOCHI_FORCE_INLINE Quaternion operator*(Quaternion a, real b);
[[nodiscard]] MOCHI_FORCE_INLINE Quaternion operator/(Quaternion a, real b);

// Linear interpolation. The faction t is not clamped.
[[nodiscard]] MOCHI_FORCE_INLINE Quaternion Lerp(Quaternion a, Quaternion b, real t);

// Spherical linear interpolation of two Quaternions.
// Assumes a and b are normalized. The fraction t is not clamped
[[nodiscard]] inline Quaternion Slerp(Quaternion a, Quaternion b, real t);

// Squared magnitude
[[nodiscard]] MOCHI_FORCE_INLINE real NormSqr(Quaternion a);

// Vector magnitude
[[nodiscard]] MOCHI_FORCE_INLINE real Norm(Quaternion a);

// Make unit length
[[nodiscard]] MOCHI_FORCE_INLINE Quaternion Normalize(Quaternion a);

// Conjugate Quaternion (opposite rotation)
[[nodiscard]] MOCHI_FORCE_INLINE Quaternion Conjugate(Quaternion const& a);

// Construct a quaternion from a rotation matrix. Must be orthonormal.
// eps is used to determine whether the angle is close to +/- 180deg
[[nodiscard]] inline Quaternion QuaternionFromMatrix(VMatrix3x3r const& matrix, real eps = 1e-3_r);
[[nodiscard]] inline Quaternion QuaternionFromMatrix(Matrix3x3r const& matrix, real eps = 1e-3_r);

// Compute a Matrix3x3r from Quaternion. The resulting matrix can be used to rotate vectors by
// calling DotMatVec3x3(VMatrix3x3r, Vec4r), which is equivalent to (Quaternion * Real3). However,
// if you have to transform multiple points, consider using ToVMatrix3x3Transpose to get the
// transpose, so that you can then use DotVecMat3x3 instead (faster).
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r ToVMatrix3x3(Quaternion const& q);

// Equivalent to Transpose3x3(ToVMatrix3x3(q)) only faster.
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r ToVMatrix3x3Transpose(Quaternion const& q);

// Compute a Matrix3x3r from Quaternion
[[nodiscard]] MOCHI_FORCE_INLINE Matrix3x3r ToMatrix3x3(Quaternion const& q);

// One stop shopping in case you need both the matrix and its transpose.
// Example:
//   auto [R, RT] = ToVMatrix3x3_WithTranspose(q);
[[nodiscard]] MOCHI_FORCE_INLINE std::pair<VMatrix3x3r, VMatrix3x3r> ToVMatrix3x3_WithTranspose(
    Quaternion const& q);

// Return true if all values are finite
[[nodiscard]] MOCHI_FORCE_INLINE Vec4r VIsFinite(Quaternion const& q);
[[nodiscard]] MOCHI_FORCE_INLINE bool IsFinite(Quaternion const& q);

} // namespace mochi

#include "quaternion_utils_inl.h"
