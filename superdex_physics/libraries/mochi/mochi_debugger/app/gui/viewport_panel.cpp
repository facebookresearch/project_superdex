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

#include "viewport_panel.h"

#include "gui.h"
#include "ui_helpers.h"
#include "viewport/imgui_render_target.h"
#include "viewport/render_scene.h"

#include <imguios/imguios.h>
#include <mochi_core/utils/defer.h>
#include <mochi_debugger/lib/debug_client.h>

#include <algorithm>
#include <cstddef>

// Forward from GLFW (imguios pulls in the real declarations).
struct GLFWwindow;

using namespace mochi;

namespace mochi::dbg {

// Called while building the Viewport panel.
static void UpdateCameraControls(ViewportPanelState& state, double timeStep) {
  Camera& camera = state.camera;

  // Keyboard movement input
  if (ImGui::IsWindowFocused()) {
    auto isKeyDown = [](ImGuiKey key) { return ImGui::IsKeyDown(key) ? 1.0f : 0.0f; };
    bool isOrtho = (camera.mode != Camera::Mode::Perspective);
    Float3 moveDir = {};
    moveDir[0] = isKeyDown(ImGuiKey_D) - isKeyDown(ImGuiKey_A); // right - left
    moveDir[1] = isKeyDown(ImGuiKey_E) - isKeyDown(ImGuiKey_Q); // up - down
    moveDir[2] = isKeyDown(ImGuiKey_S) - isKeyDown(ImGuiKey_W); // back - forward
    bool moveFaster = ImGui::IsKeyDown(ImGuiKey_ModShift);
    // Speeds are authored in m/s; the camera moves in simulation units.
    auto const unitsPerMeter = static_cast<float>(camera.simSpace.unitsPerMeter);
    float moveSpeed = (moveFaster ? state.moveSpeedFast : state.moveSpeedSlow) * unitsPerMeter;
    Float3 moveDeltaRelative = moveDir * (moveSpeed * static_cast<float>(timeStep));
    if (isOrtho) {
      moveDeltaRelative[2] = 0.0f; // We will adjust the orthoHeight instead
    }
    camera.position += camera.GetRight() * moveDeltaRelative[0] +
        camera.GetUp() * moveDeltaRelative[1] - camera.GetForward() * moveDeltaRelative[2];
    if (isOrtho) {
      float zoomSpeed =
          (moveFaster ? state.orthoZoomSpeedFast : state.orthoZoomSpeedSlow) * unitsPerMeter;
      camera.orthoHeight += moveDir[2] * (zoomSpeed * static_cast<float>(timeStep));
      camera.orthoHeight = std::max(0.01f * unitsPerMeter, camera.orthoHeight); // Don't invert
    }
  }

  // Mouse look by dragging the right mouse button
  Float2 mousePos = {ImGui::GetMousePos().x, ImGui::GetMousePos().y};
  if (ImGui::IsWindowHovered() || state.isMouseAiming) {
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      state.isMouseAiming = true;
      Float2 mouseLookDelta = mousePos - state.prevMousePos;
      state.camera.rotationEulerYPR[0] -= mouseLookDelta[0] * state.mouseLookSensitivity; // yaw
      state.camera.rotationEulerYPR[1] -= mouseLookDelta[1] * state.mouseLookSensitivity; // pitch

    } else {
      state.isMouseAiming = false;
    }
  }
  state.prevMousePos = mousePos;

  // Capture or release the mouse from the corresponding GLFW window.
  // May be nullptr while the window is being dragged.
  auto* platformWindow = reinterpret_cast<GLFWwindow*>(ImGui::GetWindowViewport()->PlatformHandle);
  if (platformWindow) {
    glfwSetInputMode(
        platformWindow,
        GLFW_CURSOR,
        state.isMouseAiming ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
  }
}

namespace {
enum class TimelineControl {
  RestoreInitialState,
  FastBackward,
  StepBackward,
  PlayPause,
  StepForward,
  FastForward,
};

struct TimelineControlButton {
  TimelineControl control = TimelineControl::PlayPause;
  char const* label = "";
  char const* tooltip = "";
  bool enabled = false;
  bool active = false;
};
} // namespace

static float GetButtonWidth(char const* label) {
  auto const& style = ImGui::GetStyle();
  return ImGui::CalcTextSize(label).x + style.FramePadding.x * 2.0f;
}

template <std::size_t kNumButtons>
static float GetTimelineControlsWidth(TimelineControlButton const (&buttons)[kNumButtons]) {
  auto const& style = ImGui::GetStyle();
  float width = 0.0f;
  for (std::size_t i = 0; i < kNumButtons; ++i) {
    width += GetButtonWidth(buttons[i].label);
    if (i + 1 < kNumButtons) {
      width += style.ItemSpacing.x;
    }
  }
  return width;
}

template <std::size_t kNumButtons>
static void BuildTimelineControls(
    UiState& state,
    TimelineControlButton const (&buttons)[kNumButtons]) {
  for (std::size_t i = 0; i < kNumButtons; ++i) {
    TimelineControlButton const& button = buttons[i];
    if (i > 0) {
      ImGui::SameLine();
    }

    if (button.active) {
      auto const activeColor = ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive);
      ImGui::PushStyleColor(ImGuiCol_Button, activeColor);
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, activeColor);
    }
    bool const clicked = UiButton(button.label, button.tooltip, button.enabled);
    if (button.active) {
      ImGui::PopStyleColor(2);
    }

    if (!clicked) {
      continue;
    }

    switch (button.control) {
      case TimelineControl::RestoreInitialState:
        state.client->RestoreSceneState();
        break;
      case TimelineControl::FastBackward:
        // TODO
        break;
      case TimelineControl::StepBackward:
        // TODO
        break;
      case TimelineControl::PlayPause: {
        auto const mode = state.client->GetSceneStepMode();
        state.client->SetSceneStepMode(mode == StepMode::Pause ? StepMode::Play : StepMode::Pause);
        break;
      }
      case TimelineControl::StepForward:
        state.client->StepScene();
        break;
      case TimelineControl::FastForward: {
        auto const mode = state.client->GetSceneStepMode();
        state.client->SetSceneStepMode(
            mode == StepMode::FastForward ? StepMode::Play : StepMode::FastForward);
        break;
      }
    }
  }
}

static void BuildViewportToolbar(UiState& state) {
  ViewportPanelState& viewport = state.viewport;

  // Aspect of the viewport panel, used to frame the scene for the camera.
  float const viewportAspect = viewport.height > 0
      ? static_cast<float>(viewport.width) / static_cast<float>(viewport.height)
      : 1.0f;

  static constexpr float kFocusOnBoundsFillPercent = 0.5f;
  float const toolbarLeftX = ImGui::GetCursorPosX();
  float const toolbarWidth = ImGui::GetContentRegionAvail().x;

  // Camera Mode
  {
    ImGui::PushItemWidth(120);
    MOCHI_DEFER(ImGui::PopItemWidth());
    int camMode = static_cast<int>(viewport.camera.mode);
    char const* itemNames[] = {"Perspective", "Side", "Top", "Front"};
    if (ImGui::Combo("##View", &camMode, itemNames, isize(itemNames))) {
      auto newMode = static_cast<Camera::Mode>(camMode);
      if (newMode == Camera::Mode::Perspective) {
        // When switching back to perspective mode, back up the camera some so that
        // it is not exactly at the center of the focus volume. That way, the FocusOnBounds
        // function will be able to preserve the look direction.
        viewport.camera.position -= viewport.camera.GetForward() *
            static_cast<float>(viewport.camera.simSpace.unitsPerMeter);
      }
      viewport.camera.mode = newMode;
      viewport.camera.FocusOnBounds(
          viewport.focusBounds, viewportAspect, kFocusOnBoundsFillPercent);
    }
  }

  // Focus On Meshes
  ImGui::SameLine();
  if (UiButton(ICON_FA_MAP_MARKER_ALT, "Focus (F)") ||
      (ImGui::IsWindowFocused() && ImGui::IsKeyPressed(ImGuiKey_F))) {
    viewport.camera.FocusOnBounds(viewport.focusBounds, viewportAspect, kFocusOnBoundsFillPercent);
  }

  bool const isConnected = (state.client->GetStatus() == net::SocketStatus::Connected);
  bool const hasScene = state.client->GetSelectedScene().IsValid();
  StepMode const stepMode = isConnected ? state.client->GetSceneStepMode() : StepMode::Pause;
  bool const isScenePaused = stepMode == StepMode::Pause;
  bool const isFastForwarding = stepMode == StepMode::FastForward;
  TimelineControlButton const timelineControls[] = {
      {TimelineControl::RestoreInitialState,
       ICON_FA_SYNC_ALT,
       "Restore initial state",
       isConnected && hasScene},
      // TODO: Hook up the rewind controls using a history of captured state buffers
      // {TimelineControl::FastBackward, ICON_FA_FAST_BACKWARD, "Rewind to start"},
      // {TimelineControl::StepBackward, ICON_FA_STEP_BACKWARD, "Rewind single step"},
      {TimelineControl::PlayPause,
       isScenePaused ? ICON_FA_PLAY : ICON_FA_PAUSE,
       isScenePaused ? "Play" : "Pause",
       isConnected},
      {TimelineControl::StepForward, ICON_FA_STEP_FORWARD, "Single step", isConnected && hasScene},
      {TimelineControl::FastForward,
       ICON_FA_FAST_FORWARD,
       "Run as fast as possible",
       isConnected,
       isFastForwarding},
  };

  auto const& style = ImGui::GetStyle();
  float const controlsX =
      toolbarLeftX + (toolbarWidth - GetTimelineControlsWidth(timelineControls)) * 0.5f;
  float const minControlsX = ImGui::GetCursorPosX() + style.ItemSpacing.x;
  ImGui::SameLine(std::max(controlsX, minControlsX));
  BuildTimelineControls(state, timelineControls);
}

void BuildViewportPanel(UiState& state) {
  ViewportPanelState& viewport = state.viewport;
  BuildViewportToolbar(state);

  ImVec2 windowSize = ImGui::GetContentRegionAvail();
  int const logicalWidth = std::max(1, static_cast<int>(windowSize.x));
  int const logicalHeight = std::max(1, static_cast<int>(windowSize.y));
  viewport.width = static_cast<uint32_t>(logicalWidth);
  viewport.height = static_cast<uint32_t>(logicalHeight);

  // Resize the offscreen target (before drawing it) and draw it. The actual scene render happens
  // afterwards in the app's render step, into this same target.
  if (viewport.renderScene) {
    float const fbScale = ImGui::GetIO().DisplayFramebufferScale.x;
    viewport.renderScene->Resize(logicalWidth, logicalHeight, fbScale);
    ImGui::RenderTargetImage(viewport.renderScene->GetTextureId(), logicalWidth, logicalHeight);
  }

  UpdateCameraControls(viewport, ImGui::GetIO().DeltaTime);
}

} // namespace mochi::dbg
