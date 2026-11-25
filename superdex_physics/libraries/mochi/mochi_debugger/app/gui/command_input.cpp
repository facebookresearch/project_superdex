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

#include "command_input.h"

#include "terminal_panel.h"

#include <mochi_core/utils/basic_utils.h>

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

using namespace mochi;
using namespace mochi::dbg;

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

bool dbg::BuildCommandInput(TerminalPanelState& term, std::string const& prompt) {
  // After Enter the input is cleared and loses focus, so re-assert it here (the box is empty, so
  // this cannot cause a select-all). History recall keeps the input active on its own.
  if (term.refocus) {
    term.refocus = false;
    ImGui::SetKeyboardFocusHere();
  }

  auto const& style = ImGui::GetStyle();
  float const promptWidth = ImGui::CalcTextSize(prompt.c_str()).x;

  // Draw the prompt *inside* the input box: inflate the left frame padding to reserve room for it,
  // then render the prompt text into that reserved area. The prompt is not part of the editable
  // buffer, so the caret sits to its right and it can never be selected or backspaced.
  ImVec2 const inputPos = ImGui::GetCursorScreenPos();
  ImGui::PushStyleVar(
      ImGuiStyleVar_FramePadding, ImVec2(style.FramePadding.x + promptWidth, style.FramePadding.y));
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  // std::string overload (imgui_stdlib) enables ImGuiInputTextFlags_CallbackResize under the hood,
  // which is required so that a recalled command can be grown/edited to any length.
  bool const submit = ImGui::InputText(
      "##cmd",
      &term.input,
      ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory,
      &TerminalInputCallback,
      &term);
  ImGui::PopStyleVar();
  ImGui::GetWindowDrawList()->AddText(
      ImVec2(inputPos.x + style.FramePadding.x, inputPos.y + style.FramePadding.y),
      ImGui::GetColorU32(ImGuiCol_Text),
      prompt.c_str());

  return submit;
}
