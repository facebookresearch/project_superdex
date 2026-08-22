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

#include "rendering/camera_controller.h"

#include <numbers>
#include <optional>

#include <mochi_renderer/scene.h>

namespace superdex::studio {

namespace {

constexpr double k_pi = std::numbers::pi;
constexpr double k_deg2rad = k_pi / 180.0;
constexpr double k_rad2deg = 180.0 / k_pi;

constexpr filament::math::double3 kWorldUp{0.0f, 1.0f, 0.0f};
constexpr filament::math::double3 kLocalForward{0.0f, 0.0f, -1.0f};
constexpr filament::math::double3 kLocalRight{1.0f, 0.0f, 0.0f};

// Bounds for the free-fly movement speed, shared by the scroll-wheel adjustment and the
// viewport speed field so both stay in sync.
constexpr double kMinCameraMoveSpeed = 0.01;
constexpr double kMaxCameraMoveSpeed = 1000.0;

double WrapAngle(double angle) {
  while (angle > k_pi) {
    angle -= 2.0 * k_pi;
  }
  while (angle < -k_pi) {
    angle += 2.0 * k_pi;
  }
  return angle;
}

double ShortestAngleTo(double from, double to) {
  double delta = WrapAngle(to - from);
  return from + delta;
}

filament::math::double3 ForwardFromQuat(filament::math::quat q) {
  return normalize(q * kLocalForward);
}

filament::math::double3 RightFromQuat(filament::math::quat q) {
  return normalize(q * kLocalRight);
}

filament::math::quat BuildRotation(double yaw, double pitch) {
  auto qYaw = filament::math::quat::fromAxisAngle(kWorldUp, yaw);
  auto qPitch = filament::math::quat::fromAxisAngle(kLocalRight, pitch);
  return normalize(qYaw * qPitch);
}

void ExtractYawPitch(filament::math::quat q, double& yaw, double& pitch) {
  auto fwd = normalize(q * kLocalForward);
  pitch = std::asin(std::clamp(fwd.y, -1.0, 1.0));
  yaw = std::atan2(-fwd.x, -fwd.z);
}

bool QuaternionsNearlyEqual(filament::math::quat a, filament::math::quat b, double epsilon = 1e-5) {
  double dotVal = dot(a, b);
  return std::abs(std::abs(dotVal) - 1.0) < epsilon;
}

} // namespace

CameraController::CameraController(mochi_renderer::Scene* scene) : _scene(scene) {
  _cameraPosition = _scene->GetCameraPosition();
  ExtractYawPitch(_scene->GetCameraRotation(), _yaw, _pitch);
  _cameraRotation = BuildRotation(_yaw, _pitch);
  _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
  auto forward = ForwardFromQuat(_cameraRotation);
  _orbitDist = length(_cameraPosition);
  _orbitPos = _cameraPosition + forward * _orbitDist;
}

std::unique_ptr<CameraController> CameraController::Create(mochi_renderer::Scene* scene) {
  return std::unique_ptr<CameraController>(new CameraController(scene));
}

void CameraController::Update(ImGuiIO& io, bool allowControl) {
  double dt = io.DeltaTime;
  // Handle lerp animation
  if (_lerpingPos || _lerpingYawPitch || _lerpingOrbit || _lerpingOrthoHeight) {
    _lerpElapsed += dt;
    double t = std::clamp(_lerpElapsed / _lerpDuration, 0.0, 1.0);
    t = t * t * (3.0f - 2.0f * t);

    // Handle orthographic height lerp (runs in parallel with position lerp)
    if (_lerpingOrthoHeight) {
      float currentHeight =
          _lerpOrthoHeight[0] + (_lerpOrthoHeight[1] - _lerpOrthoHeight[0]) * static_cast<float>(t);
      _scene->SetOrthographicHeight(currentHeight);
      if (_lerpElapsed >= _lerpDuration) {
        _lerpingOrthoHeight = false;
      }
    }

    if (_lerpingOrbit) {
      auto UpdateLerp = [this](double t) {
        _orbitPos = _lerpPos[0] + (_lerpPos[1] - _lerpPos[0]) * t;
        _orbitDist = _lerpOrbitDist[0] + (_lerpOrbitDist[1] - _lerpOrbitDist[0]) * t;
        _yaw = WrapAngle(_lerpYaw[0] + (_lerpYaw[1] - _lerpYaw[0]) * t);
        _pitch = WrapAngle(_lerpPitch[0] + (_lerpPitch[1] - _lerpPitch[0]) * t);
        _cameraRotation = BuildRotation(_yaw, _pitch);
        auto forward = ForwardFromQuat(_cameraRotation);
        _cameraPosition = _orbitPos - forward * _orbitDist;
        _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
      };
      if (_lerpElapsed >= _lerpDuration) {
        UpdateLerp(1.0);
        _lerpingOrbit = false;
      } else {
        UpdateLerp(t);
      }
    } else if (_lerpingPos) {
      auto UpdateLerp = [this](double t) {
        _cameraPosition = _lerpPos[0] + (_lerpPos[1] - _lerpPos[0]) * t;
        _cameraRotation = slerp(_lerpRot[0], _lerpRot[1], t);
        ExtractYawPitch(_cameraRotation, _yaw, _pitch);
        _orbitDist = length(_cameraPosition - _orbitPos);
        _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
      };
      if (_lerpElapsed >= _lerpDuration) {
        UpdateLerp(1.0);
        _lerpingPos = false;
      } else {
        UpdateLerp(t);
      }
    } else if (_lerpingYawPitch) {
      auto UpdateLerp = [this](double t) {
        _yaw = WrapAngle(_lerpYaw[0] + (_lerpYaw[1] - _lerpYaw[0]) * t);
        _pitch = WrapAngle(_lerpPitch[0] + (_lerpPitch[1] - _lerpPitch[0]) * t);
        _cameraRotation = BuildRotation(_yaw, _pitch);
        auto forward = ForwardFromQuat(_cameraRotation);
        _cameraPosition = _orbitPos - forward * _orbitDist;
        _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
      };
      if (_lerpElapsed >= _lerpDuration) {
        UpdateLerp(1.0);
        _lerpingYawPitch = false;
      } else {
        UpdateLerp(t);
      }
    }
    _lastMousePos = ImGui::GetMousePos();
    return;
  }

  // Gather all boolean states
  bool rightMouseDown = ImGui::IsMouseDown(ImGuiMouseButton_Right);
  bool middleMouseDown = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
  bool leftMouseDown = ImGui::IsMouseDown(ImGuiMouseButton_Left);
  bool altDown = io.KeyAlt;
  double scroll = io.MouseWheel;
  bool isOrthographic = (_scene->GetCameraMode() == mochi_renderer::CameraMode::Orthographic);

  if (allowControl) {
    // Left is reserved for the viewport (selection, gizmo, object drag) except with Alt held, which
    // orbits the camera. Detect Alt+left on its rising edge (not just the click frame) so pressing
    // Alt mid object-drag starts orbit immediately; the !_cameraControlActive guard keeps it a
    // one-shot latch (re-syncing every frame would fight DoOrbitControl).
    bool const altLeftOrbit = altDown && leftMouseDown && !rightMouseDown && !_cameraControlActive;
    if (altLeftOrbit || ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
        ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
      _cameraControlActive = true;
      _cameraPosition = _scene->GetCameraPosition();
      auto sceneRotation = filament::math::quat(_scene->GetCameraRotation());
      if (!QuaternionsNearlyEqual(sceneRotation, _cameraRotation)) {
        _cameraRotation = sceneRotation;
        ExtractYawPitch(_cameraRotation, _yaw, _pitch);
      }
      _orbitDist = length(_cameraPosition - _orbitPos);
    }
  }
  if ((!rightMouseDown && !leftMouseDown && !middleMouseDown)) {
    _cameraControlActive = false;
  }

  // Select mode. Middle-drag pans, Alt+left-drag orbits, right-drag free-looks. Plain left is left
  // to the viewport (selection, gizmo, object drag); right-drag still works while left is held so
  // the camera can be moved mid object-drag.
  bool const panControlActive = _cameraControlActive && middleMouseDown;
  bool const orbitControlActive =
      _cameraControlActive && leftMouseDown && altDown && !rightMouseDown;
  bool const rightMouseControlActive = _cameraControlActive && rightMouseDown;

  ImVec2 const mousePos = ImGui::GetMousePos();
  ImVec2 const mouseDelta{mousePos.x - _lastMousePos.x, mousePos.y - _lastMousePos.y};

  // Handle scroll for both modes - check hover independently of _cameraControlActive
  // so scroll works when just hovering over the viewport
  if ((ImGui::IsWindowHovered() || _cameraControlActive) && scroll != 0.0f) {
    if (_cameraControlActive) {
      // Adjust movement speed (original behavior) - conditioned on mouse down
      _cameraMoveSpeed *= (scroll > 0.0f) ? 1.1f : 1.0f / 1.1f;
      _cameraMoveSpeed = std::clamp(_cameraMoveSpeed, kMinCameraMoveSpeed, kMaxCameraMoveSpeed);
    } else if (isOrthographic) {
      // Orthographic mode: adjust orthographic height for zoom
      float currentHeight = _scene->GetOrthographicHeight();
      float zoomFactor = (scroll > 0.0f) ? 0.9f : 1.1f; // Zoom in = smaller height
      float newHeight = currentHeight * zoomFactor;
      newHeight = std::clamp(newHeight, 0.1f, 1000.0f);
      _scene->SetOrthographicHeight(newHeight);
    } else {
      // Move camera forward/backward along view direction
      auto forward = ForwardFromQuat(_cameraRotation);
      double moveDistance = scroll * 0.1;
      _cameraPosition += forward * moveDistance;
      _orbitDist = length(_orbitPos - _cameraPosition);
      _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
    }
  }

  if (_cameraControlActive) {
    if (isOrthographic) {
      // Orthographic mode controls (pan, orbit, etc.)
      if (orbitControlActive) {
        DoOrbitControl(mouseDelta, _cameraMouseSensitivity);
      }
      if (panControlActive) {
        DoPanControl(mouseDelta, _cameraMouseSensitivity);
      }
      if (rightMouseControlActive) {
        DoFreeLookControl(
            mouseDelta,
            ImGui::IsKeyDown(ImGuiKey_W),
            ImGui::IsKeyDown(ImGuiKey_A),
            ImGui::IsKeyDown(ImGuiKey_S),
            ImGui::IsKeyDown(ImGuiKey_D),
            ImGui::IsKeyDown(ImGuiKey_Q),
            ImGui::IsKeyDown(ImGuiKey_E),
            ImGui::IsKeyDown(ImGuiKey_LeftShift),
            dt,
            _cameraMouseSensitivity,
            _cameraMoveSpeed);
      }
    } else {
      // Perspective mode controls
      // Right-click: free-look + WASD
      if (rightMouseControlActive) {
        DoFreeLookControl(
            mouseDelta,
            ImGui::IsKeyDown(ImGuiKey_W),
            ImGui::IsKeyDown(ImGuiKey_A),
            ImGui::IsKeyDown(ImGuiKey_S),
            ImGui::IsKeyDown(ImGuiKey_D),
            ImGui::IsKeyDown(ImGuiKey_Q),
            ImGui::IsKeyDown(ImGuiKey_E),
            ImGui::IsKeyDown(ImGuiKey_LeftShift),
            dt,
            _cameraMouseSensitivity,
            _cameraMoveSpeed);
      }
      // Middle mouse: pan
      if (panControlActive) {
        DoPanControl(mouseDelta, _cameraMouseSensitivity);
      }
      // Alt + left mouse: orbit around _orbitPosition
      if (orbitControlActive) {
        DoOrbitControl(mouseDelta, _cameraMouseSensitivity);
      }
    }
  }
  _lastMousePos = mousePos;
}

void CameraController::DoFreeLookControl(
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
    double moveSpeed) {
  double yawAngle = -mouseDelta.x * sensitivity * k_deg2rad;
  double pitchAngle = -mouseDelta.y * sensitivity * k_deg2rad;
  _yaw = WrapAngle(_yaw + yawAngle);
  _pitch = WrapAngle(_pitch + pitchAngle);
  _cameraRotation = BuildRotation(_yaw, _pitch);
  auto const forward = ForwardFromQuat(_cameraRotation);
  auto const right = RightFromQuat(_cameraRotation);
  double speed = moveSpeed * deltaTime;
  if (shift) {
    speed *= 4.0f;
  }
  _cameraPosition += forward * speed * W;
  _cameraPosition -= forward * speed * S;
  _cameraPosition += right * speed * D;
  _cameraPosition -= right * speed * A;
  _cameraPosition.y -= speed * Q;
  _cameraPosition.y += speed * E;
  _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
  _orbitPos = _cameraPosition + forward * _orbitDist;
}

void CameraController::DoMoveControl(ImVec2 mouseDelta, double sensitivity) {
  double yawAngle = -mouseDelta.x * sensitivity * k_deg2rad;
  _yaw = WrapAngle(_yaw + yawAngle);
  _cameraRotation = BuildRotation(_yaw, _pitch);
  auto forward = ForwardFromQuat(_cameraRotation);
  filament::math::double3 worldForward{forward.x, 0.0f, forward.z};
  double worldForwardLen = length(worldForward);
  if (worldForwardLen > 1e-5f) {
    worldForward /= worldForwardLen;
  }
  _cameraPosition -= worldForward * mouseDelta.y * sensitivity * 0.025f;
  _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
  _orbitPos = _cameraPosition + forward * _orbitDist;
}

void CameraController::DoPanControl(ImVec2 mouseDelta, double sensitivity) {
  auto const right = RightFromQuat(_cameraRotation);
  // Compute camera up vector from rotation (for camera-relative panning)
  constexpr filament::math::double3 kLocalUp{0.0f, 1.0f, 0.0f};
  auto const up = normalize(_cameraRotation * kLocalUp);
  double panSpeed = sensitivity * 0.025f;
  // Inverted so dragging moves the scene with the cursor (grab-the-world feel).
  _cameraPosition -= right * mouseDelta.x * panSpeed;
  _cameraPosition += up * mouseDelta.y * panSpeed;
  _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
  auto forward = ForwardFromQuat(_cameraRotation);
  _orbitPos = _cameraPosition + forward * _orbitDist;
}

void CameraController::DoOrbitControl(ImVec2 mouseDelta, double sensitivity) {
  double yawAngle = -mouseDelta.x * sensitivity * k_deg2rad;
  double pitchAngle = -mouseDelta.y * sensitivity * k_deg2rad;
  _yaw = WrapAngle(_yaw + yawAngle);
  _pitch = WrapAngle(_pitch + pitchAngle);
  _cameraRotation = BuildRotation(_yaw, _pitch);
  auto forward = ForwardFromQuat(_cameraRotation);
  _cameraPosition = _orbitPos - forward * _orbitDist;
  _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
}

bool CameraController::IsActive() const {
  return _cameraControlActive || _lerpingPos || _lerpingYawPitch || _lerpingOrbit ||
      _lerpingOrthoHeight;
}

void CameraController::LerpPositionTo(
    filament::math::double3 position,
    double durationSeconds,
    std::optional<float> orthoHeight) {
  if (IsActive()) {
    return;
  }
  _lerpPos[0] = _scene->GetCameraPosition();
  _lerpPos[1] = position;
  _lerpRot[0] = filament::math::quat(_scene->GetCameraRotation());
  _lerpRot[1] = _lerpRot[0];
  _lerpDuration = durationSeconds;
  _lerpElapsed = 0.0f;
  _lerpingPos = true;
  // If orthographic height is provided, set up height lerp as well
  if (orthoHeight.has_value()) {
    _lerpOrthoHeight[0] = _scene->GetOrthographicHeight();
    _lerpOrthoHeight[1] = orthoHeight.value();
    _lerpingOrthoHeight = true;
  }
}

void CameraController::LerpYawPitchTo(double yawDeg, double pitchDeg, double durationSeconds) {
  if (IsActive()) {
    return;
  }
  _lerpYaw[0] = _yaw;
  _lerpYaw[1] = ShortestAngleTo(_yaw, yawDeg * k_deg2rad);
  _lerpPitch[0] = _pitch;
  _lerpPitch[1] = ShortestAngleTo(_pitch, pitchDeg * k_deg2rad);
  _lerpDuration = durationSeconds;
  _lerpElapsed = 0.0f;
  _lerpingYawPitch = true;
}

void CameraController::LerpOrbitTo(
    filament::math::double3 orbitPos,
    double orbitDist,
    double yawDeg,
    double pitchDeg,
    double durationSeconds,
    std::optional<float> orthoHeight) {
  if (IsActive()) {
    return;
  }
  _lerpPos[0] = _orbitPos;
  _lerpPos[1] = orbitPos;
  _lerpOrbitDist[0] = _orbitDist;
  _lerpOrbitDist[1] = orbitDist;
  _lerpYaw[0] = _yaw;
  _lerpYaw[1] = ShortestAngleTo(_yaw, yawDeg * k_deg2rad);
  _lerpPitch[0] = _pitch;
  _lerpPitch[1] = ShortestAngleTo(_pitch, pitchDeg * k_deg2rad);
  _lerpDuration = durationSeconds;
  _lerpElapsed = 0.0f;
  _lerpingOrbit = true;
  // If orthographic height is provided, set up height lerp as well
  if (orthoHeight.has_value()) {
    _lerpOrthoHeight[0] = _scene->GetOrthographicHeight();
    _lerpOrthoHeight[1] = orthoHeight.value();
    _lerpingOrthoHeight = true;
  }
}

void CameraController::SetOrbitPosition(filament::math::double3 orbitPosition) {
  _orbitPos = orbitPosition;
}

filament::math::double3 CameraController::GetOrbitPosition() const {
  return _orbitPos;
}

double CameraController::GetYaw() const {
  return _yaw;
}

double CameraController::GetPitch() const {
  return _pitch;
}

double CameraController::GetMoveSpeed() const {
  return _cameraMoveSpeed;
}

void CameraController::SetMoveSpeed(double speed) {
  _cameraMoveSpeed = std::clamp(speed, kMinCameraMoveSpeed, kMaxCameraMoveSpeed);
}

void CameraController::DebugWindow() {
  ImGui::Begin("Camera Debug");
  bool needUpdate = false;
  double yaw = _yaw * k_rad2deg;
  if (ImGui::DragScalarN("Position", ImGuiDataType_Double, _cameraPosition.v, 3)) {
    needUpdate = true;
  }
  if (ImGui::DragScalarN("Rotation", ImGuiDataType_Double, &_cameraRotation.x, 4)) {
    needUpdate = true;
  }
  if (ImGui::DragScalar("Yaw", ImGuiDataType_Double, &yaw)) {
    _yaw = yaw * k_deg2rad;
    needUpdate = true;
  }
  double pitch = _pitch * k_rad2deg;
  if (ImGui::DragScalar("Pitch", ImGuiDataType_Double, &pitch)) {
    _pitch = pitch * k_deg2rad;
    needUpdate = true;
  }
  if (ImGui::DragScalarN("Orbit Position", ImGuiDataType_Double, _orbitPos.v, 3)) {
    needUpdate = true;
  }
  if (ImGui::DragScalar("Orbit Distance", ImGuiDataType_Double, &_orbitDist)) {
    needUpdate = true;
  }
  if (needUpdate) {
    _cameraRotation = BuildRotation(_yaw, _pitch);
    auto forward = ForwardFromQuat(_cameraRotation);
    _cameraPosition = _orbitPos - forward * _orbitDist;
    _scene->CameraSetTransform(_cameraPosition, _cameraRotation);
  }
  ImGui::End();
}

} // namespace superdex::studio
