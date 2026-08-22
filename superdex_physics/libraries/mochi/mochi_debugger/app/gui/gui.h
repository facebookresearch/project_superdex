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

#include "../viewport/render_scene.h"
#include "connect_dialog.h"
#include "log_view.h"
#include "scene_panel.h"
#include "terminal_panel.h"
#include "viewport_panel.h"

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/no_copy.h>
#include <mochi_core/utils/string_utils.h>

#include <functional>
#include <map>
#include <string>
#include <utility>

namespace mochi::dbg {

// Forwards
class DebugClient;
struct UiState;

// Panel names to help prevent typos
constexpr char const* kPanelNameScene = "Scene";
constexpr char const* kPanelNameViewport = "Viewport";
constexpr char const* kPanelNameProperties = "Properties";
constexpr char const* kPanelNameTerminal = "Terminal";

enum class CommandId {
  Exit,
  ResetLayout,
  SelectScene,
};

// A queued UI command. The scene payload is only used by SelectScene.
struct UiCommand {
  CommandId id = {};
  uint64_t value = {}; // Depends on the command
};

struct PanelState : NoCopy {
  using BuildUiFn = std::function<void(UiState&)>;

  PanelState() = default;
  PanelState(char const* icon, char const* name, BuildUiFn fn)
      : icon(icon), name(name), buildUiFn(std::move(fn)) {}

  std::string icon;
  std::string name;
  std::string label = icon + " " + name;
  BuildUiFn buildUiFn;
  float windowPadding = 5.0f;
  bool isVisible = true;
};

struct ConnectionState {
  std::string address;
  uint16_t port = 0;
  bool autoReconnect = true;

  std::string FormatAddress() const { // Formatted as "ip:port"
    return address.empty() ? "" : Format("%s:%u", address.c_str(), port);
  }
};

struct UiState : NoCopy {
  bool firstFrame = true;
  bool rebuildLayout = false; // request a default dock layout on the next frame
  std::map<std::string, PanelState> panels;
  DynamicArray<UiCommand> uiCommands;
  DebugClient* client = nullptr; // Owned by the app

  // Panel + dialog state
  ViewportPanelState viewport;
  RenderSceneParams rendering;
  ConnectionState connection;
  LogView logView;
  ConnectDialogState connectDialog;
  TerminalPanelState terminal;
  ScenePanelState scene;
};

void InitializeUi(UiState& state);
void BuildUi(UiState& state);

} // namespace mochi::dbg
