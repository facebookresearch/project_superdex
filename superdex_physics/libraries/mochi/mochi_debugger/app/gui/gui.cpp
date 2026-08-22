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

#include "gui.h"

#include "connect_dialog.h"
#include "properties_panel.h"
#include "scene_panel.h"
#include "terminal_panel.h"
#include "viewport_panel.h"

#include <mochi_core/utils/defer.h>
#include <mochi_debugger/lib/debug_client.h>

#include <imgui_internal.h> // DockBuilder
#include <imguios/imguios.h>

#include <string>

using namespace mochi;
using namespace mochi::dbg;

/*****************************************************************************
  Connection status indicator
*/

// Draws a right-aligned 4-bar signal-strength meter on the menu bar.
//   When Disconnected: No bars / grey
//   When Connection Pending: Cycle animation from 1 to 4 bars in amber
//   When Connected: Show all 4 bars in green
static void BuildConnectionStatusIcon(UiState& state) {
  net::SocketStatus const status =
      state.client ? state.client->GetStatus() : net::SocketStatus::None;

  int constexpr kNumBars = 4;
  int level = 0;
  ImU32 fillColor = IM_COL32(0x9E, 0x9E, 0x9E, 0xFF);
  std::string tooltip;
  switch (status) {
    case net::SocketStatus::Connected:
      level = kNumBars;
      fillColor = IM_COL32(0x4C, 0xAF, 0x50, 0xFF); // green
      tooltip = "Connected to " + state.connection.FormatAddress();
      break;
    case net::SocketStatus::Pending:
      // Cycle 1..kNumBars over time to convey "searching / connecting".
      level = 1 + (static_cast<int>(ImGui::GetTime() * 2.0) % kNumBars);
      fillColor = IM_COL32(0xE5, 0xC0, 0x7B, 0xFF); // amber
      tooltip = "Waiting for " + state.connection.FormatAddress();
      break;
    case net::SocketStatus::None:
    case net::SocketStatus::Lost:
      level = 0;
      tooltip = "No Connection";
      break;
  }
  ImU32 constexpr kEmptyColor = IM_COL32(0x55, 0x55, 0x55, 0xFF);

  float constexpr kBarWidth = 3.0f;
  float constexpr kBarGap = 2.0f;
  float const totalWidth = kNumBars * kBarWidth + (kNumBars - 1) * kBarGap;
  float const barsHeight = ImGui::GetTextLineHeight();
  float const itemHeight = ImGui::GetFrameHeight(); // full menu-bar row height

  auto const& style = ImGui::GetStyle();
  // Nudge the meter a few pixels left of the far edge so it isn't flush against the window border.
  float constexpr kRightMargin = 8.0f;
  float const offset =
      Max(0.0f,
          ImGui::GetWindowWidth() - totalWidth - style.FramePadding.x - style.ItemSpacing.x -
              kRightMargin);
  ImGui::SameLine(offset);

  ImVec2 const itemTopLeft = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##connStatus", ImVec2(totalWidth, itemHeight));
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("%s", tooltip.c_str());
  }

  // Center the bars vertically within the menu-bar row.
  float const barsTop = itemTopLeft.y + (itemHeight - barsHeight) * 0.5f;
  auto* drawList = ImGui::GetWindowDrawList();
  for (int i = 0; i < kNumBars; ++i) {
    float const barHeight = barsHeight * static_cast<float>(i + 1) / static_cast<float>(kNumBars);
    ImVec2 const topLeft(
        itemTopLeft.x + i * (kBarWidth + kBarGap), barsTop + (barsHeight - barHeight));
    ImVec2 const bottomRight(topLeft.x + kBarWidth, barsTop + barsHeight);
    drawList->AddRectFilled(topLeft, bottomRight, (i < level) ? fillColor : kEmptyColor, 1.0f);
  }
}

/*****************************************************************************
  Menus
*/

static void BuildMainMenuBar(UiState& state) {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  MOCHI_DEFER(ImGui::PopStyleVar());

  auto sepColor = ImGui::GetStyleColorVec4(ImGuiCol_Text);
  sepColor.w = 0.1f;
  ImGui::PushStyleColor(ImGuiCol_Separator, sepColor);
  MOCHI_DEFER(ImGui::PopStyleColor());

  if (ImGui::BeginMainMenuBar()) {
    MOCHI_DEFER(ImGui::EndMainMenuBar());

    // File Menu
    if (ImGui::BeginMenu("File")) {
      MOCHI_DEFER(ImGui::EndMenu());
      if (ImGui::MenuItem("Connect...")) {
        state.connectDialog.shouldOpen = true;
      }
      if (ImGui::MenuItem("Disconnect")) {
        if (state.client) {
          state.client->Disconnect();
        }
        state.connection.address.clear();
        state.connection.port = 0;
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Exit", MOCHI_PLATFORM_WINDOWS ? "Alt + F4" : "")) {
        state.uiCommands.push_back(UiCommand{CommandId::Exit});
      }
    }

    // Windows Menu
    if (ImGui::BeginMenu("Windows")) {
      MOCHI_DEFER(ImGui::EndMenu());
      if (ImGui::MenuItem("Reset Layout", nullptr)) {
        state.uiCommands.push_back(UiCommand{CommandId::ResetLayout});
      }
      ImGui::Separator();

      for (auto&& [name, panel] : state.panels) {
        ImGui::MenuItem(panel.label.c_str(), nullptr, &panel.isVisible);
      }
    }

    BuildConnectionStatusIcon(state);
  }
}

/*****************************************************************************
  Dockable Panels
*/

// Build a sensible default dock layout: Scene on the left, Viewport in the center, and Properties
// tabbed on the right, and the Terminal along the bottom.
static void BuildDefaultLayout(UiState& state, ImGuiID dockspaceId) {
  ImGui::DockBuilderRemoveNode(dockspaceId);
  ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockspaceId, ImGui::GetMainViewport()->WorkSize);

  ImGuiID center = dockspaceId;
  ImGuiID const left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.18f, nullptr, &center);
  ImGuiID const right =
      ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.24f, nullptr, &center);
  ImGuiID const bottom =
      ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.28f, nullptr, &center);

  auto dock = [&](char const* key, ImGuiID node) {
    auto it = state.panels.find(key);
    if (it != state.panels.end()) {
      ImGui::DockBuilderDockWindow(it->second.label.c_str(), node);
    }
  };
  dock(kPanelNameScene, left);
  dock(kPanelNameViewport, center);
  dock(kPanelNameProperties, right);
  dock(kPanelNameTerminal, bottom);

  ImGui::DockBuilderFinish(dockspaceId);
}

static void BuildDockablePanels(UiState& state) {
  // Always show the terminal if there are new errors or warnings.
  if (state.logView.HasNewErrorsOrWarnings()) {
    state.panels[kPanelNameTerminal].isVisible = true;
  }

  for (auto&& [name, panel] : state.panels) {
    if (panel.isVisible || state.firstFrame) {
      ImGui::PushStyleVar(
          ImGuiStyleVar_WindowPadding, ImVec2{panel.windowPadding, panel.windowPadding});
      MOCHI_DEFER(ImGui::PopStyleVar());

      if (ImGui::Begin(panel.label.c_str(), &panel.isVisible)) {
        // Set focus on right-click (as well as left-click). This is helpful for the Viewport
        // panel where people often right-click to aim the camera and then use keys to move.
        if (ImGui::IsWindowHovered() && !ImGui::IsWindowFocused() &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
          ImGui::SetWindowFocus();
        }

        panel.buildUiFn(state);
      }
      ImGui::End();
    }
  }
}

/*****************************************************************************
  UI Initialization
*/

void dbg::InitializeUi(UiState& state) {
  state.panels.try_emplace(kPanelNameScene, ICON_FA_SITEMAP, kPanelNameScene, &BuildScenePanel);
  state.panels.try_emplace(
      kPanelNameViewport, ICON_FA_CUBES, kPanelNameViewport, &BuildViewportPanel);
  state.panels.try_emplace(
      kPanelNameProperties, ICON_FA_SLIDERS_H, kPanelNameProperties, &BuildPropertiesPanel);
  state.panels.try_emplace(
      kPanelNameTerminal, ICON_FA_TERMINAL, kPanelNameTerminal, &BuildTerminalPanel);
}

/*****************************************************************************
  BuildUi
*/

void dbg::BuildUi(UiState& state) {
  ImGui::StyleColorsDark();
  ImGuiID const dockspaceId = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport()->ID);

  if (state.rebuildLayout) {
    state.rebuildLayout = false;
    BuildDefaultLayout(state, dockspaceId);
  }

  BuildMainMenuBar(state);
  BuildDockablePanels(state);
  BuildConnectDialog(state);

  state.firstFrame = false;
}
