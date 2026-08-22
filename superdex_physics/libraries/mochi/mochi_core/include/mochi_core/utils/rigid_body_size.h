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

#include <mochi_core/utils/nd_array.h>

namespace mochi {

/* Struct to store size definitions */
struct RigidSize {
  static constexpr int kDim = 3;
  static constexpr int kRot = 4; // Rotation state: quaternion
  static constexpr int kDRot = 3; // Lie derivative of rotation
  static constexpr int kTrans = 3; // Translation state: 3D vector
  static constexpr int kDTrans = 3; // Derivative of translation
  static constexpr int kAll = kTrans + kRot; // Full state
  static constexpr int kDAll = kDTrans + kDRot; // Full derivative
};

/* Useful types */
using RigidGradient = NdArray<real, RigidSize::kDAll>;
using RigidHessian = NdArray<real, RigidSize::kDAll, RigidSize::kDAll>;

} // namespace mochi
