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
#include <mochi_core/utils/log.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

// Forward
class ImFont;

namespace mochi::dbg {

struct UiState;

struct TerminalPanelState {
  struct LogTextEntry {
    uint64_t sequence = 0;
    size_t offset = 0;
    LogChannel channel = {};
  };

  ImFont* font = nullptr; // font to use for the terminal (nullptr means default)
  std::string logText;
  DynamicArray<LogTextEntry> logEntries;
  int logSelectionStart = 0;
  int logSelectionEnd = 0;
  std::string input; // resizable so recalled commands can be edited to any length
  DynamicArray<std::string> history; // submitted commands, oldest first
  std::string stashed; // live line saved while browsing history
  int historyPos = 0; // browse cursor; == history.size() means the live line
  bool refocus = false; // re-assert keyboard focus on the input next frame
  bool scrollToBottom = false;
  // Tab completion
  std::optional<std::string> pendingInput; // replaces the input when the box next runs its callback
  std::string reportedInput; // the input and cursor last reported to the completer
  int reportedCursor = -1;
  bool isMenuOpen = false;
  bool isMenuScrollNeeded = false; // scroll the selected candidate into view
  double menuEmptyTime = -1.0; // when the open menu last ran out of candidates, or -1
  DynamicArray<std::string> menuCandidates; // what the menu shows
  size_t menuSelected = 0;
  size_t menuOffset = 0; // byte offset in the input where the candidates start
  bool bringToForegroundOnErrorOrWarning = true;
};

void BuildTerminalPanel(UiState& state);

// Returns the font to use for the terminal.
// Must be called after ImGui has been initialized.
ImFont* GetTerminalFont();

} // namespace mochi::dbg
