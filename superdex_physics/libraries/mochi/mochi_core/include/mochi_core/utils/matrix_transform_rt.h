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
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi {

/************************************************************************************
  Rotation and translation in Matrix form
*/
class MatrixTransformRT {
 public:
  MatrixTransformRT() = default;
  explicit MatrixTransformRT(Matrix3x3r const& rotation, Real3 const& translation);
  explicit MatrixTransformRT(VMatrix3x3r const& rotation, Vec4r translation);

  // clang-format off
  [[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r const& VGetRotation() const { return _rotation; }
  [[nodiscard]] MOCHI_FORCE_INLINE Vec4r VGetTranslation() const { return _translation; }
  [[nodiscard]] MOCHI_FORCE_INLINE Matrix3x3r GetRotation() const { return ToNdArray3x3(_rotation); }
  [[nodiscard]] MOCHI_FORCE_INLINE Real3 GetTranslation() const { return ToReal3(_translation); }
  // clang-format on

  // Multiplication with a 3-component vector must specify point or direction
  [[nodiscard]] Vec4r TransformPoint(Vec4r pt) const;
  [[nodiscard]] Vec4r TransformDirection(Vec4r pt) const;
  [[nodiscard]] Vec4r InverseTransformPoint(Vec4r pt) const;
  [[nodiscard]] Vec4r InverseTransformDirection(Vec4r pt) const;
  [[nodiscard]] Real3 TransformPoint(Real3 const& pt) const;
  [[nodiscard]] Real3 TransformDirection(Real3 const& dir) const;
  [[nodiscard]] Real3 InverseTransformPoint(Real3 const& pt) const;
  [[nodiscard]] Real3 InverseTransformDirection(Real3 const& dir) const;

  // Multiplication with another TransformRT concatenates transforms
  [[nodiscard]] MatrixTransformRT operator*(MatrixTransformRT const& rhs) const;

 private:
  VMatrix3x3r _rotation = VEye<3>(); // 4th SIMD column is always 0
  Vec4r _translation{0_r, 0_r, 0_r, 1_r}; // 4th SIMD components is always 1

 public:
  MOCHI_STRUCT_BEGIN(mochi::MatrixTransformRT)
  MOCHI_FIELD_NAME(_rotation, "rotation")
  MOCHI_FIELD_NAME(_translation, "translation")
  MOCHI_STRUCT_END()
};

/************************************************************************************
  Utilities
*/

// Converting representations
[[nodiscard]] MOCHI_FORCE_INLINE MatrixTransformRT ToMatrixTransformRT(TransformRT const& a);
[[nodiscard]] MOCHI_FORCE_INLINE MatrixTransformRT
ToMatrixTransformRT(Quaternion const& rot, Real3 const& trans);
[[nodiscard]] MOCHI_FORCE_INLINE TransformRT ToTransformRT(MatrixTransformRT const& a);
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4(MatrixTransformRT const& a);
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix4x4r ToVMatrix4x4Transpose(MatrixTransformRT const& a);

// Compute the inverse transform
[[nodiscard]] MOCHI_FORCE_INLINE MatrixTransformRT Invert(MatrixTransformRT const& a);

// NearEqual: (abs(a-b) <= epsilon)
[[nodiscard]] MOCHI_FORCE_INLINE bool NearEqual(
    MatrixTransformRT const& a,
    MatrixTransformRT const& b,
    real epsilon = kDefaultNearEqualEpsilon<real>);

} // namespace mochi

#include "matrix_transform_rt_inl.h"
