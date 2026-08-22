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

#include "gui.h"
#include "ui_helpers.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/log.h>
#include <mochi_debugger/lib/debug_client.h>

#include <imguios/imguios.h>
#include <misc/cpp/imgui_stdlib.h>

#include <cstring>
#include <string>

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
static void BuildLogOutput(LogView& logView, float reservedFooterHeight) {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
  MOCHI_DEFER(ImGui::PopStyleVar(2));

  ImGui::BeginChild(
      "##termLog", ImVec2(0, -reservedFooterHeight), true, ImGuiWindowFlags_HorizontalScrollbar);
  MOCHI_DEFER(ImGui::EndChild());

  if (ImGui::BeginPopupContextWindow()) {
    MOCHI_DEFER(ImGui::EndPopup());
    if (ImGui::MenuItem("Copy")) {
      std::string clipboardText;
      logView.ReadEntries([&](LogView::Entry const& entry) { clipboardText += entry.message; });
      ImGui::SetClipboardText(clipboardText.c_str());
    }
    if (ImGui::MenuItem("Clear")) {
      logView.Clear();
    }
  }

  // The store is bounded (see LogView::kMaxEntries), so simply render every retained entry.
  logView.ReadEntries([](LogView::Entry const& entry) {
    ImVec4 color(1.0f, 1.0f, 1.0f, 1.0f); // White
    switch (entry.channel) {
      case LogChannel::Warning:
        color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow
        break;
      case LogChannel::Error:
        color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f); // Red
        break;
      case LogChannel::Verbose:
      case LogChannel::Info:
      case LogChannel::Count:
        break;
    }
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    MOCHI_DEFER(ImGui::PopStyleColor());
    ImGui::TextUnformatted(entry.message.c_str());
  });

  // If new messages have arrived since the last frame, scroll to the bottom.
  if (logView.ConsumeNewEntries()) {
    ImGui::SetScrollHereY(1.0f);
  }
}

// Handles Up/Down history browsing inside the command input, shell-style.
static int TerminalInputCallback(ImGuiInputTextCallbackData* data) {
  auto* term = static_cast<TerminalPanelState*>(data->UserData);
  if (data->EventFlag != ImGuiInputTextFlags_CallbackHistory) {
    return 0;
  }

  int const historySize = isize(term->history);
  int const prevPos = term->historyPos;
  if (data->EventKey == ImGuiKey_UpArrow) {
    if (term->historyPos == historySize) {
      term->stashed = data->Buf; // save the live line before browsing into history
    }
    if (term->historyPos > 0) {
      --term->historyPos;
    }
  } else if (data->EventKey == ImGuiKey_DownArrow) {
    if (term->historyPos < historySize) {
      ++term->historyPos;
    }
  }

  if (term->historyPos != prevPos) {
    std::string const& line =
        (term->historyPos == historySize) ? term->stashed : term->history[term->historyPos];
    data->DeleteChars(0, data->BufTextLen);
    data->InsertChars(0, line.c_str());
    // Replace the box contents, put the caret at the end, and clear any selection so the user can
    // start typing to append immediately (a lingering selection would swallow the first keystroke).
    data->CursorPos = data->BufTextLen;
    data->SelectionStart = data->BufTextLen;
    data->SelectionEnd = data->BufTextLen;
  }
  return 0;
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

  BuildLogOutput(state.logView, footerHeight);

  std::string const prompt = state.client ? state.client->GetCommandPrompt() : std::string{};
  std::string const promptText = prompt + "> ";

  // After Enter the input is cleared and loses focus, so re-assert it here (the box is empty, so
  // this cannot cause a select-all). History recall keeps the input active on its own.
  if (state.terminal.refocus) {
    state.terminal.refocus = false;
    ImGui::SetKeyboardFocusHere();
  }

  auto const& style = ImGui::GetStyle();
  float const promptWidth = ImGui::CalcTextSize(promptText.c_str()).x;

  // Draw the prompt *inside* the input box: inflate the left frame padding to reserve room for it,
  // then render the prompt text into that reserved area. The prompt is not part of the editable
  // buffer, so the caret sits to its right and it can never be selected or backspaced.
  ImVec2 const inputPos = ImGui::GetCursorScreenPos();
  ImGui::PushStyleVar(
      ImGuiStyleVar_FramePadding, ImVec2(style.FramePadding.x + promptWidth, style.FramePadding.y));
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  // std::string overload (imgui_stdlib) enables ImGuiInputTextFlags_CallbackResize under the hood,
  // which is required so that a recalled command can be grown/edited to any length.
  bool submit = ImGui::InputText(
      "##cmd",
      &state.terminal.input,
      ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory,
      &TerminalInputCallback,
      &state.terminal);
  ImGui::PopStyleVar();
  ImGui::GetWindowDrawList()->AddText(
      ImVec2(inputPos.x + style.FramePadding.x, inputPos.y + style.FramePadding.y),
      ImGui::GetColorU32(ImGuiCol_Text),
      promptText.c_str());

  if (submit) {
    std::string const command = state.terminal.input;

    // Echo the submitted command. An empty command prints a blank line, like a shell.
    state.logView.AddLine(
        LogChannel::Info, command.empty() ? std::string{} : (promptText + command), "", 0);

    if (!command.empty()) {
      if (command == "cls") {
        // We handle this command ourselves
        state.logView.Clear();
      } else if (state.client) {
        // TODO: Queue this command for execution on another thread, to avoid blocking the UI.
        state.client->ExecuteCommand(command);
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
