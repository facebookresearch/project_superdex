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

#include "connect_dialog.h"

#include "gui.h"
#include "ui_helpers.h"

#include <mochi_core/utils/defer.h>
#include <mochi_debugger/lib/address.h>
#include <mochi_debugger/lib/debug_client.h>
#include <mochi_physics/dbg/protocol.h> // For mochi::dbg::kDefaultDebugServerPort

#include <imguios/imguios.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace mochi::dbg {

// Only digits and dots are allowed in the address field.
static int FilterAddressChar(ImGuiInputTextCallbackData* data) {
  ImWchar const c = data->EventChar;
  return ((c >= '0' && c <= '9') || c == '.') ? 0 : 1;
}

// Only digits are allowed in the port field.
static int FilterPortChar(ImGuiInputTextCallbackData* data) {
  ImWchar const c = data->EventChar;
  return (c >= '0' && c <= '9') ? 0 : 1;
}

// Sort by address then by port
static void SortServers(DynamicArray<net::ServerInfo>& servers) {
  std::ranges::sort(servers, [](auto const& a, auto const& b) {
    int const addrCmp = a.address.compare(b.address);
    if (addrCmp != 0) {
      return addrCmp < 0;
    }
    return a.port < b.port;
  });
}

void ConnectTo(UiState& state, std::string_view address, uint16_t port) {
  // Store endpoint
  state.connection.address = std::string{address};
  state.connection.port = port;

  // Clear scene data immediately, in case we were previously connected.
  state.scene = {};

  // Clear meshes and debug draw from the previous scene immediately.
  state.viewport.renderScene->ClearMeshes();
  state.viewport.renderScene->ClearDebugDraw();

  state.client->Connect(state.connection.address, port);
}

void BuildConnectDialog(UiState& state) {
  auto& dialog = state.connectDialog;

  // Cancel is a one-time action.
  bool const shouldCancel = dialog.shouldCancel;
  dialog.shouldCancel = false;

  // If cancel and open are both queued, cancel takes precedence.
  if (shouldCancel && dialog.shouldOpen) {
    dialog.shouldOpen = false;
  }

  if (dialog.shouldOpen) {
    dialog.shouldOpen = false;
    // Reset the fields to their defaults each time the dialog opens.
    dialog.startPaused = true;
    MOCHI_ASSERT_VERBOSE(kLocalHost.size() < dialog.address.size());
    std::copy_n(kLocalHost.data(), kLocalHost.size(), dialog.address.data());
    dialog.address[kLocalHost.size()] = '\0';
    std::snprintf(dialog.port.data(), dialog.port.size(), "%d", dbg::kDefaultDebugServerPort);
    dialog.serverList.Refresh();
    ImGui::OpenPopup("Connect");
  }

  ImVec2 const center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(540, 400), ImGuiCond_Appearing);

  // Escape cancels the dialog (handled explicitly below).
  if (!ImGui::BeginPopupModal("Connect", nullptr, ImGuiWindowFlags_NoSavedSettings)) {
    return;
  }
  MOCHI_DEFER(ImGui::EndPopup());

  if (shouldCancel) {
    ImGui::CloseCurrentPopup();
    return;
  }

  auto connectTo = [&state](std::string_view address, uint16_t port, bool startPaused) {
    // Set the pause state before we connect.
    state.client->SetSceneStepMode(startPaused ? StepMode::Pause : StepMode::Play);

    ConnectTo(state, address, port);
    ImGui::CloseCurrentPopup();
  };

  // Address + port inputs. Enter in either field acts as Connect.
  bool submit = false;
  ImGui::TextUnformatted("Address");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(160);
  if (ImGui::InputText(
          "##address",
          dialog.address.data(),
          dialog.address.size(),
          ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_EnterReturnsTrue,
          &FilterAddressChar)) {
    submit = true;
  }

  ImGui::SameLine();
  ImGui::TextUnformatted("Port");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(60); // wide enough for 5 digits
  if (ImGui::InputText(
          "##port",
          dialog.port.data(),
          dialog.port.size(),
          ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_EnterReturnsTrue,
          &FilterPortChar)) {
    submit = true;
  }

  ImGui::SameLine();
  if (UiButton(ICON_FA_SYNC, "Refresh server list")) {
    dialog.serverList.Refresh();
  }

  // Poll discovery results (may grow across frames as responses arrive).
  dialog.serverList.GetServers(dialog.servers);
  SortServers(dialog.servers);

  Error validateError;
  dialog.address[dialog.address.size() - 1] = '\0'; // Paranoid null-terminator
  dialog.port[dialog.port.size() - 1] = '\0';
  std::string_view const currentAddress{dialog.address.data()};
  ValidateAddress(currentAddress, validateError);
  uint16_t const currentPort = ParsePort(dialog.port.data(), validateError);
  bool const currentEndpointValid = validateError.IsOK();

  // Scrollable list of discovered servers, formatted like the mdb "servers" command.
  ImGui::BeginChild("##servers", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), true);
  {
    MOCHI_DEFER(ImGui::EndChild());
    if (dialog.servers.empty()) {
      ImGui::TextDisabled("No servers found");
    }
    for (int i = 0; i < isize(dialog.servers); ++i) {
      auto const& server = dialog.servers[i];
      std::string const addrPort = std::string(server.address) + ":" + std::to_string(server.port);
      std::string suffix;
      if (server.version != state.client->GetVersion()) {
        suffix = " (incompatible)";
      } else if (server.numClients == server.maxClients) {
        suffix = " (busy)";
      }
      char line[160];
      std::snprintf(
          line,
          sizeof(line),
          "%-21s  %s%s",
          addrPort.c_str(),
          server.label.c_str(),
          suffix.c_str());

      // A row is selected exactly when it matches the current address+port fields.
      bool const selected = currentEndpointValid && (server.port == currentPort) &&
          (currentAddress == server.address);

      ImGui::PushID(i);
      MOCHI_DEFER(ImGui::PopID());
      if (ImGui::Selectable(line, selected, ImGuiSelectableFlags_AllowDoubleClick)) {
        // Single click fills the fields; double click also connects.
        std::snprintf(dialog.address.data(), dialog.address.size(), "%s", server.address.c_str());
        std::snprintf(
            dialog.port.data(), dialog.port.size(), "%u", static_cast<unsigned>(server.port));
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          connectTo(server.address, server.port, dialog.startPaused);
        }
      }
    }
  }

  // Enter anywhere in the dialog acts as Connect.
  if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
      (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter))) {
    submit = true;
  }

  // Esc anywhere in the dialog acts as Cancel.
  if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
      ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    ImGui::CloseCurrentPopup();
  }

  // Bottom row: Options on the left, Connect / Cancel right-justified.
  UiCheckbox(
      "Auto Reconnect",
      &state.connection.autoReconnect,
      "Automatically reconnect to this endpoint if the connection is lost.");
  ImGui::SameLine(0.0f, 15.0f);
  UiCheckbox(
      "Start Paused",
      &dialog.startPaused,
      "Pause immediately so you can inspect the initial state.");

  auto const& style = ImGui::GetStyle();
  float const connectWidth = ImGui::CalcTextSize("Connect").x + style.FramePadding.x * 2.0f;
  float const cancelWidth = ImGui::CalcTextSize("Cancel").x + style.FramePadding.x * 2.0f;
  float const buttonsWidth = connectWidth + style.ItemSpacing.x + cancelWidth;
  ImGui::SameLine(ImGui::GetContentRegionMax().x - buttonsWidth);
  if (UiButton("Connect", "Connect to the specified address", currentEndpointValid)) {
    submit = true;
  }
  if (submit && currentEndpointValid) {
    connectTo(currentAddress, currentPort, dialog.startPaused);
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) {
    ImGui::CloseCurrentPopup();
  }
}

} // namespace mochi::dbg
