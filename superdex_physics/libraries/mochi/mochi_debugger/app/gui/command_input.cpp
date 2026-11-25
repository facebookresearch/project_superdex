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
#include <mochi_debugger/lib/auto_completer.h>
#include <mochi_debugger/lib/terminal_text.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <cfloat>
#include <optional>
#include <string_view>
#include <utility>

using namespace mochi;
using namespace mochi::dbg;

namespace {
// How the completion menu closes, and whether it opens without Tab. Tab and Shift+Tab always open
// it. Closing as soon as nothing completes is the simplest to understand. The other modes trade
// that for a steadier menu while typing through several steps.
enum class CompletionMenuMode {
  CloseWhenEmpty, // Closes as soon as nothing completes
  LingerWhenEmpty, // Stays open briefly while nothing completes, so the next step's reply can
                   // refill it
  OpenWhileTyping, // Lingers, and also opens whenever typed text completes
};

struct InputCallbackContext {
  TerminalPanelState* terminal;
  AutoCompleter* completer; // nullptr without a client
};
} // namespace

static constexpr CompletionMenuMode kCompletionMenuMode = CompletionMenuMode::CloseWhenEmpty;
static constexpr double kMenuLingerSeconds = 0.1;
static constexpr int kMenuMaxRows = 10;
static constexpr char const* kMenuName = "##completionMenu";

// Replace the box contents and put the caret at the end without a selection, so the user can type
// to append immediately (a lingering selection would swallow the first keystroke).
static void ReplaceInput(ImGuiInputTextCallbackData* data, std::string const& text) {
  data->DeleteChars(0, data->BufTextLen);
  data->InsertChars(0, text.c_str());
  data->CursorPos = data->BufTextLen;
  data->SelectionStart = data->BufTextLen;
  data->SelectionEnd = data->BufTextLen;
}

// Handles Up/Down history browsing inside the command input, shell-style.
static void BrowseHistory(TerminalPanelState& term, ImGuiInputTextCallbackData* data) {
  int const historySize = isize(term.history);
  int const prevPos = term.historyPos;
  if (data->EventKey == ImGuiKey_UpArrow) {
    if (term.historyPos == historySize) {
      term.stashed = data->Buf; // save the live line before browsing into history
    }
    if (term.historyPos > 0) {
      --term.historyPos;
    }
  } else if (data->EventKey == ImGuiKey_DownArrow) {
    if (term.historyPos < historySize) {
      ++term.historyPos;
    }
  }

  if (term.historyPos != prevPos) {
    ReplaceInput(
        data, term.historyPos == historySize ? term.stashed : term.history[term.historyPos]);
  }
}

// Whether the mouse is over the menu, as of the start of the frame
static bool IsMenuHovered() {
  ImGuiWindow const* const hovered = ImGui::GetCurrentContext()->HoveredWindow;
  return hovered != nullptr && hovered == ImGui::FindWindowByName(kMenuName);
}

static void OpenMenu(TerminalPanelState& term) {
  term.isMenuOpen = true;
  term.isMenuScrollNeeded = true;
}

static void CloseMenu(TerminalPanelState& term) {
  term.isMenuOpen = false;
  term.menuEmptyTime = -1.0;
  term.menuCandidates.clear();
}

// Up and Down select the previous and next candidate in the menu, stopping at the ends.
static void MoveMenuSelection(
    TerminalPanelState& term,
    AutoCompleter& completer,
    ImGuiInputTextCallbackData* data) {
  auto const view = completer.GetView();
  if (view.candidates.empty()) {
    return;
  }
  size_t const index = data->EventKey == ImGuiKey_UpArrow
      ? (view.selected == 0 ? 0 : view.selected - 1)
      : std::min(view.selected + 1, view.candidates.size() - 1);
  if (auto const text = completer.Select(index)) {
    ReplaceInput(data, *text);
    term.isMenuScrollNeeded = true;
  }
}

static int TerminalInputCallback(ImGuiInputTextCallbackData* data) {
  auto const& context = *static_cast<InputCallbackContext*>(data->UserData);
  auto& term = *context.terminal;
  AutoCompleter* const completer = context.completer;
  switch (data->EventFlag) {
    case ImGuiInputTextFlags_CallbackCompletion:
      if (auto const text = completer ? completer->Cycle(/*backward*/ false) : std::nullopt) {
        ReplaceInput(data, *text);
        OpenMenu(term);
      }
      break;
    case ImGuiInputTextFlags_CallbackHistory:
      if (term.isMenuOpen && completer) {
        MoveMenuSelection(term, *completer, data);
      } else {
        BrowseHistory(term, data);
      }
      break;
    case ImGuiInputTextFlags_CallbackAlways: {
      if (term.pendingInput) {
        ReplaceInput(data, *term.pendingInput);
        term.pendingInput.reset();
      }
      // The completer sees every edit and caret move, which also ends a Tab cycle.
      std::string_view const text{data->Buf, static_cast<size_t>(data->BufTextLen)};
      if (completer && (text != term.reportedInput || data->CursorPos != term.reportedCursor)) {
        term.reportedInput = text;
        term.reportedCursor = data->CursorPos;
        completer->SetInput(text, data->CursorPos == data->BufTextLen);
      }
      break;
    }
    default:
      break;
  }
  return 0;
}

// Open or close the menu for the current completion, and remember what it shows.
static void UpdateCompletionMenu(TerminalPanelState& term, AutoCompleter::View const& view) {
  bool const hasCandidates = !view.candidates.empty();
  if constexpr (kCompletionMenuMode == CompletionMenuMode::OpenWhileTyping) {
    if (!term.isMenuOpen && hasCandidates && (!view.preview.empty() || view.isCycling)) {
      OpenMenu(term);
    }
  }
  if (!term.isMenuOpen) {
    return;
  }
  if (hasCandidates) {
    term.menuCandidates = view.candidates;
    term.menuSelected = view.selected;
    term.menuOffset = view.offset;
    term.menuEmptyTime = -1.0;
    return;
  }
  if constexpr (kCompletionMenuMode == CompletionMenuMode::CloseWhenEmpty) {
    CloseMenu(term);
    return;
  }
  double const now = ImGui::GetTime();
  if (term.menuEmptyTime < 0.0) {
    term.menuEmptyTime = now;
  } else if (now - term.menuEmptyTime > kMenuLingerSeconds) {
    CloseMenu(term);
  }
}

// Draws the menu with its bottom-left corner at anchor. It takes no keyboard focus when it appears.
// Returns the candidate the user clicked.
static std::optional<size_t> BuildCompletionMenu(TerminalPanelState& term, ImVec2 anchor) {
  auto const& style = ImGui::GetStyle();
  // In the terminal's viewport, so the menu never becomes a separate platform window
  ImGui::SetNextWindowViewport(ImGui::GetWindowViewport()->ID);
  ImGui::SetNextWindowPos(anchor, ImGuiCond_Always, ImVec2(0.0f, 1.0f));
  float const maxHeight =
      kMenuMaxRows * ImGui::GetTextLineHeightWithSpacing() + 2.0f * style.WindowPadding.y;
  ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(FLT_MAX, maxHeight));
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking;
  std::optional<size_t> clicked;
  if (ImGui::Begin(kMenuName, nullptr, kFlags)) {
    ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
    for (size_t i = 0; i < term.menuCandidates.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      bool const isSelected = i == term.menuSelected;
      // An empty label, so candidates containing "##" display in full
      if (ImGui::Selectable("", isSelected)) {
        clicked = i;
      }
      ImGui::SameLine(0.0f, 0.0f);
      ImGui::TextUnformatted(term.menuCandidates[i].c_str());
      if (isSelected && term.isMenuScrollNeeded) {
        ImGui::SetScrollHereY();
      }
      ImGui::PopID();
    }
    term.isMenuScrollNeeded = false;
    // A click elsewhere closes the menu.
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !IsMenuHovered()) {
      CloseMenu(term);
    }
  }
  ImGui::End();
  return clicked;
}

bool dbg::BuildCommandInput(
    TerminalPanelState& term,
    AutoCompleter* completer,
    std::string const& prompt) {
  // After Enter or a click in the completion menu the input loses focus, so re-assert it here. The
  // select-all that focusing causes is cleared by the pending input after a click, and the box is
  // empty after Enter. History recall keeps the input active on its own.
  if (term.refocus) {
    term.refocus = false;
    ImGui::SetKeyboardFocusHere();
  }

  auto const& style = ImGui::GetStyle();
  float const promptWidth = ImGui::CalcTextSize(prompt.c_str()).x;

  // Keys handled before the input runs. Esc closes the menu instead of reverting the input, and
  // Shift+Tab cycles backward.
  ImGuiID const inputId = ImGui::GetID("##cmd");
  bool const wasInputActive = ImGui::GetActiveID() == inputId;
  if (wasInputActive && term.isMenuOpen && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    CloseMenu(term);
    ImGui::SetKeyOwner(ImGuiKey_Escape, ImGui::GetID(kMenuName), ImGuiInputFlags_LockThisFrame);
  }
  if (completer && wasInputActive &&
      ImGui::IsKeyChordPressed(ImGuiMod_Shift | ImGuiKey_Tab, ImGuiInputFlags_None, inputId)) {
    if (auto text = completer->Cycle(/*backward*/ true)) {
      term.pendingInput = std::move(text);
      OpenMenu(term);
    }
  }

  // Draw the prompt *inside* the input box: inflate the left frame padding to reserve room for it,
  // then render the prompt text into that reserved area. The prompt is not part of the editable
  // buffer, so the caret sits to its right and it can never be selected or backspaced.
  ImVec2 const inputPos = ImGui::GetCursorScreenPos();
  float const inputWidth = ImGui::GetContentRegionAvail().x;
  ImGui::PushStyleVar(
      ImGuiStyleVar_FramePadding, ImVec2(style.FramePadding.x + promptWidth, style.FramePadding.y));
  ImGui::SetNextItemWidth(inputWidth);
  // std::string overload (imgui_stdlib) enables ImGuiInputTextFlags_CallbackResize under the hood,
  // which is required so that a recalled command can be grown/edited to any length.
  InputCallbackContext callbackContext{.terminal = &term, .completer = completer};
  bool const submit = ImGui::InputTextWithHint(
      "##cmd",
      "Type 'help' for commands",
      &term.input,
      ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory |
          ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackAlways,
      &TerminalInputCallback,
      &callbackContext);
  bool const isInputActive = ImGui::IsItemActive();
  auto const& io = ImGui::GetIO();
  if (isInputActive && !io.KeyCtrl && !io.KeySuper) {
    // Keyboard navigation reads Tab before the input runs, so claim it a frame ahead. Otherwise
    // Shift+Tab would move focus to the previous widget. Ctrl+Tab still switches windows.
    ImGui::SetKeyOwner(ImGuiKey_Tab, inputId);
  }
  ImGui::PopStyleVar();
  float const textX = inputPos.x + style.FramePadding.x + promptWidth;
  float const textY = inputPos.y + style.FramePadding.y;
  ImGui::GetWindowDrawList()->AddText(
      ImVec2(inputPos.x + style.FramePadding.x, textY),
      ImGui::GetColorU32(ImGuiCol_Text),
      prompt.c_str());

  if (completer) {
    if (submit) {
      // The submitted line keeps no suggestion while the input is cleared and refocused.
      completer->SetInput("", true);
      term.reportedInput.clear();
      term.reportedCursor = 0;
      CloseMenu(term);
    } else if (!isInputActive && !IsMenuHovered() && !term.pendingInput) {
      CloseMenu(term); // Focus moved elsewhere
    }

    // The completion shows in faint text after the input, as in mdb, while the input has focus and
    // does not scroll.
    auto const view = completer->GetView();
    float const textEnd = textX + ImGui::CalcTextSize(term.input.c_str()).x;
    float const boxEnd = inputPos.x + inputWidth - style.FramePadding.x;
    float const cellWidth = ImGui::CalcTextSize("M").x;
    if (isInputActive && textEnd < boxEnd && cellWidth > 0.0f) {
      std::string const suggestion = FitSuggestion(
          term.input,
          view.preview,
          view.annotations,
          static_cast<size_t>((boxEnd - textEnd) / cellWidth));
      ImGui::GetWindowDrawList()->AddText(
          ImVec2(textEnd, textY), ImGui::GetColorU32(ImGuiCol_TextDisabled), suggestion.c_str());
    }

    UpdateCompletionMenu(term, view);
    if (term.isMenuOpen && !submit) {
      size_t const offset = std::min(term.menuOffset, term.input.size());
      float const anchorX =
          textX + ImGui::CalcTextSize(term.input.data(), term.input.data() + offset).x;
      if (auto const clicked = BuildCompletionMenu(term, ImVec2(anchorX, inputPos.y))) {
        term.pendingInput = completer->Select(*clicked);
        term.isMenuScrollNeeded = true;
        // The click moved focus to the menu. Focusing the input selects all of it, which the
        // pending input replaces.
        term.refocus = term.pendingInput.has_value();
      }
    }
  }

  return submit;
}
