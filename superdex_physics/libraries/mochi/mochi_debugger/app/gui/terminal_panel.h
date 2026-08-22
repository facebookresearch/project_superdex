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

#include <mochi_core/utils/dynamic_array.h>

#include <string>

// Forward
class ImFont;

namespace mochi::dbg {

struct UiState;

struct TerminalPanelState {
  ImFont* font = nullptr; // font to use for the terminal (nullptr means default)
  std::string input; // resizable so recalled commands can be edited to any length
  DynamicArray<std::string> history; // submitted commands, oldest first
  std::string stashed; // live line saved while browsing history
  int historyPos = 0; // browse cursor; == history.size() means the live line
  bool refocus = false; // re-assert keyboard focus on the input next frame (after Enter)
};

void BuildTerminalPanel(UiState& state);

// Returns the font to use for the terminal.
// Must be called after ImGui has been initialized.
ImFont* GetTerminalFont();

} // namespace mochi::dbg
