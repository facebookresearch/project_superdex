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

#include "log_text.h"
#include "log_view.h"
#include "terminal_panel.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <string>
#include <utility>

namespace mochi::dbg {

static int LogTextCallback(ImGuiInputTextCallbackData* data) {
  auto& terminal = *static_cast<TerminalPanelState*>(data->UserData);
  terminal.logSelectionStart = data->SelectionStart;
  terminal.logSelectionEnd = data->SelectionEnd;
  return 0;
}

bool BuildSelectableLog(LogView& logView, TerminalPanelState& terminal) {
  std::string text;
  DynamicArray<TerminalPanelState::LogTextEntry> entries;
  logView.ReadEntries([&](LogView::Entry const& entry) {
    entries.push_back(
        {.sequence = entry.sequence, .offset = text.size(), .channel = entry.channel});
    text += entry.message;
    if (entry.message.empty() || entry.message.back() != '\n') {
      text += '\n';
    }
  });

  ImGuiID const id = ImGui::GetID("##logText");
  bool const textChanged = text != terminal.logText;
  size_t dropped = 0;
  bool resetSelection = entries.empty();
  if (!entries.empty() && !terminal.logEntries.empty() &&
      entries.front().sequence != terminal.logEntries.front().sequence) {
    auto* const first = std::ranges::find_if(terminal.logEntries, [&](auto const& entry) {
      return entry.sequence == entries.front().sequence;
    });
    resetSelection = first == terminal.logEntries.end();
    if (!resetSelection) {
      dropped = first->offset;
    }
  }
  if (resetSelection || dropped != 0) {
    auto const adjust = [&](int position) {
      return resetSelection ? 0 : position - static_cast<int>(std::min(dropped, size_t(position)));
    };
    terminal.logSelectionStart = adjust(terminal.logSelectionStart);
    terminal.logSelectionEnd = adjust(terminal.logSelectionEnd);
  }
  if (textChanged || resetSelection || dropped != 0) {
    if (auto* edit = ImGui::GetInputTextState(id)) {
      edit->ReloadUserBufAndKeepSelection();
      edit->ReloadSelectionStart = terminal.logSelectionStart;
      edit->ReloadSelectionEnd = terminal.logSelectionEnd;
    }
  }
  terminal.logText = std::move(text);
  terminal.logEntries = std::move(entries);

  auto const padding = ImGui::GetStyle().FramePadding;
  ImVec2 const available = ImGui::GetContentRegionAvail();
  ImVec2 const textSize = ImGui::CalcTextSize(terminal.logText.c_str());
  ImVec2 const size{
      std::max(available.x, textSize.x + 2.0f * padding.x + 1.0f),
      std::max(available.y, textSize.y + 2.0f * padding.y + ImGui::GetTextLineHeight())};
  ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_ChildBg));
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  ImGui::InputTextMultiline(
      "##logText",
      &terminal.logText,
      size,
      ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_CallbackAlways,
      &LogTextCallback,
      &terminal);
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(2);
  bool const active = ImGui::IsItemActive();

  // InputText owns selection and copying; draw channel colors over its selection background.
  for (auto* child : ImGui::GetCurrentWindow()->DC.ChildWindows) {
    if (child->ChildId != id) {
      continue;
    }
    ImVec2 position = child->DC.CursorStartPos;
    position.x += padding.x;
    position.y += padding.y;
    if (auto const* edit = ImGui::GetInputTextState(id)) {
      position.x -= edit->Scroll.x;
    }
    for (size_t i = 0; i < terminal.logEntries.size(); ++i) {
      auto const& entry = terminal.logEntries[i];
      ImVec4 color(1.0f, 1.0f, 1.0f, 1.0f);
      if (entry.channel == LogChannel::Warning) {
        color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
      } else if (entry.channel == LogChannel::Error) {
        color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
      }
      char const* const begin = terminal.logText.c_str() + entry.offset;
      char const* const end = terminal.logText.c_str() +
          (i + 1 < terminal.logEntries.size() ? terminal.logEntries[i + 1].offset
                                              : terminal.logText.size());
      child->DrawList->AddText(position, ImGui::ColorConvertFloat4ToU32(color), begin, end);
      position.y +=
          static_cast<float>(std::ranges::count(begin, end, '\n')) * ImGui::GetTextLineHeight();
    }
    break;
  }

  if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    auto const& rect = ImGui::GetCurrentWindow()->InnerRect;
    auto const& io = ImGui::GetIO();
    float const speed = 10.0f * io.DeltaTime;
    ImGui::SetScrollY(
        ImGui::GetScrollY() +
        (io.MousePos.y - std::clamp(io.MousePos.y, rect.Min.y, rect.Max.y)) * speed);
    ImGui::SetScrollX(
        ImGui::GetScrollX() +
        (io.MousePos.x - std::clamp(io.MousePos.x, rect.Min.x, rect.Max.x)) * speed);
  }
  return active;
}

} // namespace mochi::dbg
