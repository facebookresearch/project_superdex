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

#include "camera.h"

#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <cmath>

namespace mochi::dbg {

namespace {

// The simulation's own axes as floats, for composing poses.
struct SimAxes {
  Float3 right;
  Float3 up;
  Float3 forward;
};

SimAxes GetSimAxes(CoordinateSpace const& space) {
  return {
      StaticCast<Float3>(space.GetRight()),
      StaticCast<Float3>(space.GetUp()),
      StaticCast<Float3>(space.GetForward())};
}

// The camera's three axes at the given angles, built by summing the simulation's axes rather than
// by rotating about them. Axis-angle rotation follows the numeric right-hand rule, so it would
// turn the opposite way in a left-handed simulation space; weighting the axes directly keeps the
// controls feeling the same in every space.
struct CameraAxes {
  Float3 right;
  Float3 up;
  Float3 forward;
};

CameraAxes GetCameraAxes(CoordinateSpace const& space, Float3 const& eulerYPR) {
  SimAxes const sim = GetSimAxes(space);
  float const cy = Cos(eulerYPR[0]);
  float const sy = Sin(eulerYPR[0]);
  float const cp = Cos(eulerYPR[1]);
  float const sp = Sin(eulerYPR[1]);

  // Yaw about the simulation's up axis, then pitch about the yawed right axis.
  CameraAxes axes;
  axes.forward = sim.forward * (cp * cy) - sim.right * (cp * sy) + sim.up * sp;
  axes.right = sim.right * cy + sim.forward * sy;
  axes.up = sim.right * (sp * sy) + sim.up * cp - sim.forward * (sp * cy);

  // Roll spins right and up about forward, which is unaffected.
  float const cr = Cos(eulerYPR[2]);
  float const sr = Sin(eulerYPR[2]);
  Float3 const rolledRight = axes.right * cr + axes.up * sr;
  axes.up = axes.up * cr - axes.right * sr;
  axes.right = rolledRight;
  return axes;
}

} // namespace

Float3 Camera::DirectionFromYawPitch(CoordinateSpace const& space, float yaw, float pitch) {
  return GetCameraAxes(space, {yaw, pitch, 0.0f}).forward;
}

Float3 Camera::GetRight() const {
  return GetCameraAxes(simSpace, rotationEulerYPR).right;
}

Float3 Camera::GetUp() const {
  return GetCameraAxes(simSpace, rotationEulerYPR).up;
}

Float3 Camera::GetForward() const {
  return GetCameraAxes(simSpace, rotationEulerYPR).forward;
}

void Camera::ResetToDefault(CoordinateSpace const& space) {
  // The distance defaults are authored in meters, so a default-constructed camera only has to be
  // scaled into the simulation's units. This keeps them next to the fields they belong to.
  *this = Camera{};
  simSpace = space;

  auto const unitsPerMeter = static_cast<float>(space.unitsPerMeter);
  nearZ *= unitsPerMeter;
  farZ *= unitsPerMeter;
  orthoHeight *= unitsPerMeter;

  SimAxes const sim = GetSimAxes(space);
  position = (sim.right * kDefaultEyeRightMeters + sim.up * kDefaultEyeUpMeters -
              sim.forward * kDefaultEyeBackMeters) *
      unitsPerMeter;
  LookAt(sim.up * (kDefaultTargetUpMeters * unitsPerMeter));
}

void Camera::LookAt(Float3 const& target) {
  switch (mode) {
    case Mode::Perspective: {
      SimAxes const sim = GetSimAxes(simSpace);
      Float3 const toTarget = target - position;
      // How far the target lies along each of the simulation's axes. Yaw turns left for positive
      // angles, so its rightward component is negated.
      float const forwardness = Dot(toTarget, sim.forward);
      float const rightness = Dot(toTarget, sim.right);
      float const upness = Dot(toTarget, sim.up);
      rotationEulerYPR[0] = std::atan2(-rightness, forwardness);
      rotationEulerYPR[1] =
          std::atan2(upness, std::sqrt(forwardness * forwardness + rightness * rightness));
    } break;
    case Mode::OrthoSide:
      position = target;
      rotationEulerYPR = {0.5f * static_cast<float>(kPI), 0.0f, 0.0f}; // look along -right
      break;
    case Mode::OrthoTop:
      position = target;
      rotationEulerYPR = {0.0f, -0.5f * static_cast<float>(kPI), 0.0f}; // look along -up
      break;
    case Mode::OrthoFront:
      position = target;
      rotationEulerYPR = {}; // look along forward
      break;
  }
}

void Camera::FocusOnBounds(Aabb const& bounds, float viewportAspect, float fillPercent) {
  auto const center = StaticCast<Float3>(bounds.GetCenter());
  auto const halfExtent = StaticCast<Float3>(bounds.GetHalfExtents());
  float const fill = Clamp(fillPercent, 0.05f, 1.0f);
  auto const unitsPerMeter = static_cast<float>(simSpace.unitsPerMeter);

  if (mode == Mode::Perspective) {
    float const radius = Max(Norm(halfExtent), 1e-3f * unitsPerMeter);
    float const dist = radius / (fill * Tan(verticalFov * 0.5f));
    Float3 const toEye = position - center;
    float const len = Norm(toEye);
    // Standing on the center leaves no look direction to preserve, so back straight off.
    Float3 const backward = -StaticCast<Float3>(simSpace.GetForward());
    Float3 const dir = (len > 1e-4f * unitsPerMeter) ? (toEye / len) : backward;
    position = center + dir * dist;
    LookAt(center);
  } else {
    // Orthographic: aim first, then size so the AABB fills.
    LookAt(center);
    CameraAxes const axes = GetCameraAxes(simSpace, rotationEulerYPR);
    float const sceneW = 2.0f * Dot(Abs(axes.right), halfExtent);
    float const sceneH = 2.0f * Dot(Abs(axes.up), halfExtent);
    float const safeAspect = Max(0.01f, viewportAspect);
    orthoHeight = Max(0.01f * unitsPerMeter, Max(sceneH, sceneW / safeAspect) / fill);
  }
}

} // namespace mochi::dbg
