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

#include <mochi_core/utils/transform_rt.h>

namespace mochi {

// Get/set rotation in converted formats
[[nodiscard]] MOCHI_FORCE_INLINE Matrix3x3r GetRotationMatrix(TransformRT const& transform) {
  return ToMatrix3x3(transform.GetRotation());
}

[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r VGetRotationMatrix(TransformRT const& transform) {
  return ToVMatrix3x3(transform.GetRotation());
}

[[nodiscard]] MOCHI_FORCE_INLINE Vec4r GetRotationVector(TransformRT const& transform) {
  return transform.GetRotation().VToRotationVector();
}

MOCHI_FORCE_INLINE void SetRotationVector(Vec4r const& rotVec, TransformRT& outTransform) {
  outTransform.SetRotation(Quaternion::FromRotationVector(rotVec));
}

MOCHI_FORCE_INLINE void SetRotationVector(Real3 const& rotVec, TransformRT& outTransform) {
  SetRotationVector(ToSimd(rotVec), outTransform);
}

} // namespace mochi
