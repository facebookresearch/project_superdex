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
#include <imgui.h>
#include <math/quat.h>
#include <math/vec3.h>
#include <array>
#include <memory>
#include <optional>

namespace mochi_renderer {
class Scene;
} // namespace mochi_renderer

namespace superdex::studio {

class CameraController {
 public:
  static std::unique_ptr<CameraController> Create(mochi_renderer::Scene* scene);
  void Update(ImGuiIO& io, bool allowControl);
  void DoFreeLookControl(
      ImVec2 mouseDelta,
      bool W,
      bool A,
      bool S,
      bool D,
      bool Q,
      bool E,
      bool shift,
      double deltaTime,
      double sensitivity,
      double moveSpeed);
  void DoMoveControl(ImVec2 mouseDelta, double sensitivity);
  void DoPanControl(ImVec2 mouseDelta, double sensitivity);
  void DoOrbitControl(ImVec2 mouseDelta, double sensitivity);
  bool IsActive() const;
  void LerpPositionTo(
      filament::math::double3 position,
      double durationSeconds = 0.2f,
      std::optional<float> orthoHeight = std::nullopt);
  void LerpYawPitchTo(double yawDeg, double pitchDeg, double durationSeconds = 0.2f);
  void LerpOrbitTo(
      filament::math::double3 orbitPos,
      double orbitDist,
      double yawDeg,
      double pitchDeg,
      double durationSeconds = 0.2f,
      std::optional<float> orthoHeight = std::nullopt);
  void SetOrbitPosition(filament::math::double3 orbitPosition);
  filament::math::double3 GetOrbitPosition() const;
  double GetYaw() const;
  double GetPitch() const;

  // Free-fly movement speed, also adjustable via scroll wheel while flying. The setter clamps to
  // the same bounds as the scroll-wheel adjustment.
  double GetMoveSpeed() const;
  void SetMoveSpeed(double speed);

  void DebugWindow();

 private:
  CameraController(mochi_renderer::Scene* scene);

  mochi_renderer::Scene* _scene = nullptr;

  double _cameraMoveSpeed = 1.0f;
  double _cameraMouseSensitivity = 0.15f;

  filament::math::double3 _cameraPosition{0.0f};
  filament::math::quat _cameraRotation{1.0f, 0.0f, 0.0f, 0.0f};
  filament::math::double3 _orbitPos = {0.0f, 0.0f, 0.0f};
  double _orbitDist = 1.0f;
  double _yaw = 0.0f;
  double _pitch = 0.0f;
  ImVec2 _lastMousePos{0, 0};
  bool _cameraControlActive = false;

  double _lerpElapsed = 0.0f;
  double _lerpDuration = 0.1f;

  bool _lerpingPos = false;
  std::array<filament::math::double3, 2> _lerpPos;
  std::array<filament::math::quat, 2> _lerpRot;
  // Optional orthographic height lerp (only used when in ortho mode)
  bool _lerpingOrthoHeight = false;
  std::array<float, 2> _lerpOrthoHeight = {0.0f, 0.0f};

  bool _lerpingYawPitch = false;
  std::array<double, 2> _lerpYaw = {0.0f, 0.0f};
  std::array<double, 2> _lerpPitch = {0.0f, 0.0f};

  bool _lerpingOrbit = false;
  std::array<double, 2> _lerpOrbitDist = {0.0f, 0.0f};
};

} // namespace superdex::studio
