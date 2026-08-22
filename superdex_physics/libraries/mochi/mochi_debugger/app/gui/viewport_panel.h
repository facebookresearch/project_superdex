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

#include "../viewport/camera.h"

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/utils/nd_array.h>

#include <cstdint>

namespace mochi::dbg {

// Forwards
struct UiState;
class RenderScene;

struct ViewportPanelState {
  uint32_t width = 0;
  uint32_t height = 0;
  RenderScene* renderScene = nullptr; // owned by the app; used to resize + draw the viewport

  // Camera
  Camera camera;
  Aabb focusBounds = {};

  // Keyboard Movement Input
  float moveSpeedSlow = 0.4f; // m/s
  float moveSpeedFast = 2.0f; // m/s
  float orthoZoomSpeedSlow = 0.2f; // m/s
  float orthoZoomSpeedFast = 1.0f; // m/s

  // Mouse Look Input
  bool isMouseAiming = false;
  Float2 prevMousePos = {};
  float mouseLookSensitivity = 0.005f; // radians/mouse_delta
};

// Viewport panel: camera toolbar, offscreen render-target image, and camera controls.
void BuildViewportPanel(UiState& state);

} // namespace mochi::dbg
