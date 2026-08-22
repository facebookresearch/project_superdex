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

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/coordinate_space.h>
#include <mochi_core/utils/nd_array.h>

namespace mochi::dbg {

// A camera posed in the simulation's coordinate space.
struct Camera {
  using Mat4x4 = Matrix<float, 4, 4>;
  enum class Mode { Perspective, OrthoSide, OrthoTop, OrthoFront };

  // The initial pose, in meters: the eye sits right, up and back of a target that is itself
  // slightly above the origin. ResetToDefault turns these into the simulation's axes and units.
  static constexpr float kDefaultEyeRightMeters = 0.25f;
  static constexpr float kDefaultEyeUpMeters = 0.75f;
  static constexpr float kDefaultEyeBackMeters = 1.0f;
  static constexpr float kDefaultTargetUpMeters = 0.25f;
  static constexpr float kDefaultNearZMeters = 0.01f;
  static constexpr float kDefaultFarZMeters = 50.0f;
  static constexpr float kDefaultOrthoHeightMeters = 1.0f;

  Mode mode = Mode::Perspective;
  Float3 position = {}; // simulation units

  // Yaw turns about the simulation's up axis, pitch tilts about the camera's own right axis, and
  // roll spins about its own forward axis. All zero looks along the simulation's forward axis.
  // Positive yaw turns left, which is why mouse-look subtracts the horizontal delta.
  Float3 rotationEulerYPR = {}; // radians

  float nearZ = kDefaultNearZMeters;
  float farZ = kDefaultFarZMeters;
  float verticalFov = 50.0f * kRadiansPerDegree; // radians
  float orthoHeight = kDefaultOrthoHeightMeters;

  // The simulation's coordinate space. Updated on connect.
  CoordinateSpace simSpace = CoordinateSpace::Filament();

  // Restore the initial pose in @p space, so it frames the same physical volume from the same
  // angle whatever the simulation's convention and units are.
  void ResetToDefault(CoordinateSpace const& space);

  void LookAt(Float3 const& target);

  // Frame the world Aabb, so it approximately fills the given percentage of the viewport.
  void FocusOnBounds(Aabb const& bounds, float viewportAspect, float fillPercent);

  // The camera's own axes, in simulation coordinates.
  [[nodiscard]] Float3 GetRight() const;
  [[nodiscard]] Float3 GetUp() const;
  [[nodiscard]] Float3 GetForward() const;

  // The direction at the given yaw and pitch, in @p space coordinates -- what a camera with those
  // angles looks along. Exposed because the directional light is aimed the same way.
  [[nodiscard]] static Float3
  DirectionFromYawPitch(CoordinateSpace const& space, float yaw, float pitch);
};

} // namespace mochi::dbg
