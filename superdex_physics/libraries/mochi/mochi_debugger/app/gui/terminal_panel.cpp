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

#include "terminal_panel.h"

#include "command_input.h"
#include "gui.h"
#include "log_text.h"
#include "ui_helpers.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/log.h>
#include <mochi_debugger/lib/debug_client.h>

#include <imguios/imguios.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <string>
#include <utility>

using namespace mochi;
using namespace mochi::dbg;

// Returns the monospace font loaded by ImGuios (Roboto Mono), falling back to default.
ImFont* dbg::GetTerminalFont() {
  ImGuiIO& io = ImGui::GetIO();
  for (ImFont* font : io.Fonts->Fonts) {
    if (font == nullptr) {
      continue;
    }
    char const* name = font->GetDebugName();
    if (name != nullptr &&
        (std::strstr(name, "Mono") != nullptr || std::strstr(name, "mono") != nullptr)) {
      return font;
    }
  }
  return io.Fonts->Fonts.empty() ? nullptr : io.Fonts->Fonts[0];
}

// Draws the scrolling log output. Caller is responsible for pushing the desired terminal font
// before invoking (BuildTerminalPanel does so for the whole panel).
static void BuildLogOutput(
    LogView& logView,
    TerminalPanelState& terminal,
    float reservedFooterHeight,
    DebugClient* client) {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
  MOCHI_DEFER(ImGui::PopStyleVar(2));

  ImGui::BeginChild(
      "##termLog", ImVec2(0, -reservedFooterHeight), true, ImGuiWindowFlags_HorizontalScrollbar);
  MOCHI_DEFER(ImGui::EndChild());

  if (client) {
    auto const& style = ImGui::GetStyle();
    float const scrollbarWidth = ImGui::GetScrollMaxY() > 0.0f ? style.ScrollbarSize : 0.0f;
    float const width = ImGui::GetWindowWidth() - 2.0f * style.WindowPadding.x - scrollbarWidth;
    float const characterWidth = ImGui::CalcTextSize("M").x;
    client->SetConsoleLineWidth(
        characterWidth > 0.0f ? static_cast<size_t>(std::max(1.0f, width / characterWidth)) : 0);
  }

  bool const wasAtBottom =
      ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - ImGui::GetTextLineHeight();
  bool const hadEntries = !terminal.logEntries.empty();
  uint64_t const previousSequence = hadEntries ? terminal.logEntries.back().sequence : 0;
  bool const logActive = BuildSelectableLog(logView, terminal);
  bool const newEntries = !terminal.logEntries.empty() &&
      (!hadEntries || terminal.logEntries.back().sequence != previousSequence);
  bool const dragging = logActive && ImGui::IsMouseDown(ImGuiMouseButton_Left);

  if (ImGui::BeginPopupContextItem("##logContext")) {
    MOCHI_DEFER(ImGui::EndPopup());
    bool const hasSelection = terminal.logSelectionStart != terminal.logSelectionEnd;
    if (ImGui::MenuItem("Copy", nullptr, false, hasSelection)) {
      auto const start =
          static_cast<size_t>(std::min(terminal.logSelectionStart, terminal.logSelectionEnd));
      size_t const length =
          static_cast<size_t>(std::max(terminal.logSelectionStart, terminal.logSelectionEnd)) -
          start;
      std::string const clipboardText = terminal.logText.substr(start, length);
      ImGui::SetClipboardText(clipboardText.c_str());
    }
    if (ImGui::MenuItem("Copy All")) {
      ImGui::SetClipboardText(terminal.logText.c_str());
    }
    if (ImGui::MenuItem("Clear")) {
      logView.Clear();
    }
    ImGui::Separator();
    ImGui::MenuItem(
        "Bring to Foreground on Error/Warning",
        nullptr,
        &terminal.bringToForegroundOnErrorOrWarning);
  }

  if (std::exchange(terminal.scrollToBottom, false) || (newEntries && wasAtBottom && !dragging)) {
    ImGui::SetScrollHereY(1.0f);
  }
}

static std::string FormatPrompt(DebugClient const* client) {
  return (client ? client->GetCommandPrompt() : std::string{}) + "> ";
}

void dbg::BuildTerminalPanel(UiState& state) {
  // Use fixed-width terminal font for the entire panel so tables and prompts align.
  auto* font = state.terminal.font;
  if (font) {
    ImGui::PushFont(font);
  }
  MOCHI_DEFER(if (font) { ImGui::PopFont(); });

  // Reserve space at the bottom for the single command input row.
  float const footerHeight = ImGui::GetFrameHeightWithSpacing();

  BuildLogOutput(state.logView, state.terminal, footerHeight, state.client);

  bool const submit = BuildCommandInput(state.terminal, FormatPrompt(state.client));
  if (submit) {
    std::string const command = state.terminal.input;
    state.terminal.scrollToBottom = true;

    // Queued, so the echo and cls come after the output of earlier commands
    auto const run = [&](std::function<void()> fn) {
      if (state.client) {
        state.client->QueueCommand(std::move(fn));
      } else {
        fn();
      }
    };

    // Echo the submitted command. An empty command prints a blank line, like a shell.
    run([&logView = state.logView, client = state.client, command] {
      logView.AddLine(
          LogChannel::Info,
          command.empty() ? std::string{} : FormatPrompt(client) + command,
          "",
          0);
    });

    if (!command.empty()) {
      if (command == "cls") {
        // We handle this command ourselves
        run([&logView = state.logView] { logView.Clear(); });
      } else if (state.client) {
        state.client->QueueCommand(command);
      }
      // Append to history (skip consecutive duplicates), like a normal shell.
      if (state.terminal.history.empty() || state.terminal.history.back() != command) {
        state.terminal.history.push_back(command);
      }
    }

    state.terminal.historyPos = isize(state.terminal.history);
    state.terminal.stashed.clear();
    state.terminal.input.clear();
    // Keep focus in the box after Enter so the user can immediately type another command.
    state.terminal.refocus = true;
  }
}
