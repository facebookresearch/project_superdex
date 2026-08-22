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

#include "ui/log_console.h"

#include <mochi_physics/cpp_api/mochi_context.h>

#include "mochi_core/mochi_platform.h"
#include "mochi_core/utils/defer.h"

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#if MOCHI_PLATFORM_WINDOWS
#include <Windows.h>
#endif

using namespace mochi;

namespace superdex::studio {

//------------------------------------------------------------------------------------------------
// LogHook
//------------------------------------------------------------------------------------------------

LogHook::LogHook(std::streambuf* outputBuf, Callback callback)
    : _outputBuf(outputBuf), _callback(std::move(callback)) {}

std::streambuf::int_type LogHook::overflow(int_type c) {
  if (c != EOF) {
    _line.push_back(static_cast<char>(c));
    if (c == '\n') {
      _callback(std::move(_line));
      _line.clear();
    }
  }
  return _outputBuf->sputc(c);
}

//------------------------------------------------------------------------------------------------
// LogView
//------------------------------------------------------------------------------------------------

LogView::LogView() {
  // Intercept cout
  _prevCoutBuf = std::cout.rdbuf();
  _coutHook = std::make_unique<LogHook>(_prevCoutBuf, [this](std::string message) {
    this->AddEntry(LogChannel::Info, std::move(message), nullptr, 0);
  });
  std::cout.rdbuf(_coutHook.get());

  // Intercept cerr
  _prevCerrBuf = std::cerr.rdbuf();
  _cerrHook = std::make_unique<LogHook>(_prevCerrBuf, [this](std::string message) {
    this->AddEntry(LogChannel::Error, std::move(message), nullptr, 0);
  });
  std::cerr.rdbuf(_cerrHook.get());

  // Intercept mochi logging
  _prevMochiFn = mochi::Context::GetLogCallback();
  mochi::Context::SetLogCallback(
      [this](LogChannel channel, char const* message, char const* sourceFile, int sourceLine) {
        this->AddEntry(channel, std::string(message), sourceFile, sourceLine);

        // Forward Mochi messages to std::out (NOT to our LogHook which intercepts std::cout).
        auto messageForDebugger = mochi::Format("%s(%d): %s", sourceFile, sourceLine, message);
        _prevCoutBuf->sputn(messageForDebugger.c_str(), messageForDebugger.length());

#if MOCHI_PLATFORM_WINDOWS
        // On Windows, debuggers like VSCode and Visual Studio Pro do not show std::out in the debug
        // output window. There is a special function which puts text there.
        // OutputDebugStringA has a hard limit on string length (~4KB); send in chunks.
        {
          constexpr size_t kMaxChunk = 4095;
          char const* ptr = messageForDebugger.c_str();
          size_t remaining = messageForDebugger.length();
          while (remaining > kMaxChunk) {
            char buf[kMaxChunk + 1];
            memcpy(buf, ptr, kMaxChunk);
            buf[kMaxChunk] = '\0';
            ::OutputDebugStringA(buf);
            ptr += kMaxChunk;
            remaining -= kMaxChunk;
          }
          ::OutputDebugStringA(ptr);
        }
#endif // MOCHI_PLATFORM_WINDOWS
      });
}

LogView::~LogView() {
  // Restore global state
  std::cout.rdbuf(_prevCoutBuf);
  std::cerr.rdbuf(_prevCerrBuf);
  mochi::Context::SetLogCallback(_prevMochiFn);
}

void LogView::AddEntry(
    mochi::LogChannel channel,
    std::string message,
    char const* sourceFile,
    int sourceLine) {
  // Guard against re-entry: if logging is triggered during this callback
  // (e.g. from mochi::Format or sputn), skip to avoid deadlocking the mutex.
  thread_local bool insideAddEntry = false;
  if (insideAddEntry) {
    return;
  }
  insideAddEntry = true;
  MOCHI_DEFER(insideAddEntry = false);
  std::lock_guard lock(mutex);
  // Normalize to a single clean line: drop the trailing newline/carriage-return so the console's
  // Selectable rows are not double-height and copied text gets exactly one newline per entry.
  while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
    message.pop_back();
  }
  entries.emplace_back(Entry{channel, std::move(message), sourceFile, sourceLine, _nextEntryId++});
  hasNewEntries = true;
  if (channel == LogChannel::Error) {
    hasNewErrors.store(true, std::memory_order_relaxed);
  }
}

//--------------------------------------------------------------------------------------------------
// LogConsole
//--------------------------------------------------------------------------------------------------

LogConsole::LogConsole() : _logView(std::make_unique<LogView>()) {}

void LogConsole::ApplyVerboseCapture(LogConsoleSettings const& settings) {
  mochi::EnableLogChannel(mochi::LogChannel::Verbose, settings.showVerbose);
}

bool LogConsole::ShowWindow(bool* open, LogConsoleSettings& settings) {
  bool settingsChanged = false;
  if (!ImGui::Begin("Log Console", open)) {
    ImGui::End();
    return settingsChanged;
  }

  // Base text colors (drawn on the dark log background and on the selection highlight).
  static constexpr ImVec4 kVerboseColor{0.80f, 0.80f, 0.80f, 1.0f};
  static constexpr ImVec4 kInfoColor{46.0f / 255.0f, 134.0f / 255.0f, 233.0f / 255.0f, 1.0f};
  static constexpr ImVec4 kWarningColor{1.0f, 1.0f, 0.0f, 1.0f};
  static constexpr ImVec4 kErrorColor{233.0f / 255.0f, 55.0f / 255.0f, 81.0f / 255.0f, 1.0f};

  // Dark text variants drawn when a row is hovered, so text stays readable on the light hover
  // highlight (brightness inverted relative to the base colors above).
  static constexpr ImVec4 kVerboseColorHover{0.22f, 0.22f, 0.22f, 1.0f};
  static constexpr ImVec4 kInfoColorHover{0.06f, 0.16f, 0.34f, 1.0f};
  static constexpr ImVec4 kWarningColorHover{0.42f, 0.38f, 0.0f, 1.0f};
  static constexpr ImVec4 kErrorColorHover{0.42f, 0.09f, 0.13f, 1.0f};

  // Row highlight backgrounds. The render loop draws these itself rather than via ImGui's
  // Header/HeaderHovered colors (whose state mapping coupled the selection's look to the hover
  // color), so the two are independent. A hovered row uses kHoverBg whether or not it is selected.
  static constexpr ImVec4 kHoverBg{0.86f, 0.86f, 0.88f, 1.0f}; // any hovered row (light)
  static constexpr ImVec4 kSelectedBg{0.07f, 0.16f, 0.30f, 0.85f}; // selected, not hovered

  // Toolbar
  static constexpr ImVec4 kBlackTextColor{0.0f, 0.0f, 0.0f, 1.0f};
  auto ColoredToggle = [](char const* label,
                          bool* value,
                          ImVec4 const& color,
                          ImVec4 const* activeTextColor = nullptr) -> bool {
    bool const isActive = *value;
    if (!isActive) {
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.25f);
    } else if (activeTextColor) {
      ImGui::PushStyleColor(ImGuiCol_Text, *activeTextColor);
    }
    bool const toggled = ImGuios::ButtonColored(label, color);
    if (toggled) {
      *value = !*value;
    }
    if (!isActive) {
      ImGui::PopStyleVar();
    } else if (activeTextColor) {
      ImGui::PopStyleColor();
    }
    return toggled;
  };
  // Verbose is the one channel that is off at the source by default, so its toggle enables/disables
  // capture (only on change, to avoid per-frame overhead) rather than merely filtering the view.
  if (ColoredToggle("Verbose", &settings.showVerbose, kVerboseColor, &kBlackTextColor)) {
    ApplyVerboseCapture(settings);
    settingsChanged = true;
  }
  ImGui::SameLine();
  settingsChanged |= ColoredToggle("Info", &settings.showInfo, kInfoColor);
  ImGui::SameLine();
  settingsChanged |=
      ColoredToggle("Warning", &settings.showWarning, kWarningColor, &kBlackTextColor);
  ImGui::SameLine();
  settingsChanged |= ColoredToggle("Error", &settings.showError, kErrorColor);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(200);
  if (ImGui::InputTextWithHint(
          "##Filter", "Filter", _textFilter.InputBuf, IM_ARRAYSIZE(_textFilter.InputBuf))) {
    _textFilter.Build();
  }
  ImGui::SameLine();

  if (ImGui::Button("Clear Logs")) {
    std::lock_guard lock(_logView->mutex);
    _logView->entries.clear();
    _logSelection.Clear();
  }

  // Log area
  ImGui::BeginChild(
      "ConsoleLog", ImVec2(0, 0), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
  {
    using Entry = LogView::Entry;
    std::lock_guard lock(_logView->mutex);
    // Limit to last 1000 entries for performance.
    constexpr size_t kMaxEntries = 1000;
    size_t const totalEntries = _logView->entries.size();
    size_t const startIdx = totalEntries > kMaxEntries ? totalEntries - kMaxEntries : 0;

    auto passesFilters = [&](Entry const& entry) -> bool {
      bool levelEnabled = false;
      switch (entry.channel) {
        case mochi::LogChannel::Verbose:
          levelEnabled = settings.showVerbose;
          break;
        case mochi::LogChannel::Info:
          levelEnabled = settings.showInfo;
          break;
        case mochi::LogChannel::Warning:
          levelEnabled = settings.showWarning;
          break;
        case mochi::LogChannel::Error:
          levelEnabled = settings.showError;
          break;
        case mochi::LogChannel::Count:
          return false;
      }
      return levelEnabled && _textFilter.PassFilter(entry.message.c_str());
    };

    auto channelColor = [&](mochi::LogChannel channel) -> ImVec4 {
      switch (channel) {
        case mochi::LogChannel::Verbose:
          return kVerboseColor;
        case mochi::LogChannel::Info:
          return kInfoColor;
        case mochi::LogChannel::Warning:
          return kWarningColor;
        case mochi::LogChannel::Error:
          return kErrorColor;
        case mochi::LogChannel::Count:
          break;
      }
      return kInfoColor;
    };

    auto channelColorHover = [&](mochi::LogChannel channel) -> ImVec4 {
      switch (channel) {
        case mochi::LogChannel::Verbose:
          return kVerboseColorHover;
        case mochi::LogChannel::Info:
          return kInfoColorHover;
        case mochi::LogChannel::Warning:
          return kWarningColorHover;
        case mochi::LogChannel::Error:
          return kErrorColorHover;
        case mochi::LogChannel::Count:
          break;
      }
      return kInfoColorHover;
    };

    // The filtered lines in display order; drives rendering, the selection<->id mapping, and copy.
    std::vector<Entry const*> visible;
    visible.reserve(totalEntries - startIdx);
    for (size_t i = startIdx; i < totalEntries; ++i) {
      Entry const& entry = _logView->entries[i];
      if (passesFilters(entry)) {
        visible.push_back(&entry);
      }
    }

    // Map each item's selection user data (its display index) to the entry's stable id, so range
    // and box selections cover exactly the visible lines and survive new entries / filtering.
    _logSelection.UserData = &visible;
    _logSelection.AdapterIndexToStorageId = [](ImGuiSelectionBasicStorage* self,
                                               int idx) -> ImGuiID {
      auto const* items = static_cast<std::vector<Entry const*>*>(self->UserData);
      return static_cast<ImGuiID>((*items)[static_cast<size_t>(idx)]->id);
    };

    ImGuiMultiSelectFlags const msFlags =
        ImGuiMultiSelectFlags_ClearOnEscape | ImGuiMultiSelectFlags_BoxSelect1d;
    ImGuiMultiSelectIO* ms =
        ImGui::BeginMultiSelect(msFlags, _logSelection.Size, static_cast<int>(visible.size()));
    _logSelection.ApplyRequests(ms);

    // Draw each row's background ourselves so the four states have fully independent colors. ImGui
    // renders a selected row (and a selected+hovered row) with its Header/HeaderHovered colors in a
    // way that coupled the selection's look to the hover color, so Selectable's own backgrounds are
    // disabled (Header* pushed transparent); it still handles interaction + sizing, and we paint
    // the background and re-draw the text on top.
    static constexpr ImVec4 kTransparent{0.0f, 0.0f, 0.0f, 0.0f};
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, kTransparent);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, kTransparent);
    ImGui::PushStyleColor(ImGuiCol_Header, kTransparent);
    ImDrawList* const drawList = ImGui::GetWindowDrawList();
    for (size_t i = 0; i < visible.size(); ++i) {
      Entry const* entry = visible[i];
      bool const selected = _logSelection.Contains(static_cast<ImGuiID>(entry->id));
      ImVec2 const rowMin = ImGui::GetCursorScreenPos();
      ImGui::SetNextItemSelectionUserData(static_cast<ImGuiSelectionUserData>(i));
      // Push the base text color so Selectable's own (transparent-background) text matches what we
      // re-draw below, then let Selectable add the item + handle sizing/horizontal scrolling.
      ImGui::PushStyleColor(ImGuiCol_Text, channelColor(entry->channel));
      ImGui::PushID(static_cast<int>(entry->id));
      ImGui::Selectable(entry->message.c_str(), selected);
      // Hover is read after drawing (the row's true state); suppressed while the mouse is held so a
      // box-select drag does not recolor rows mid-drag.
      bool const hovered = ImGui::IsItemHovered() && !ImGui::IsMouseDown(ImGuiMouseButton_Left);
      ImVec4 background = kTransparent;
      ImVec4 textColor = channelColor(entry->channel);
      if (hovered) {
        // A hovered row (selected or not) uses the light hover highlight with dark text.
        background = kHoverBg;
        textColor = channelColorHover(entry->channel);
      } else if (selected) {
        background = kSelectedBg;
      }
      if (background.w > 0.0f) {
        drawList->AddRectFilled(
            ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), ImGui::GetColorU32(background));
      }
      drawList->AddText(rowMin, ImGui::GetColorU32(textColor), entry->message.c_str());
      ImGui::PopID();
      ImGui::PopStyleColor();
    }
    ImGui::PopStyleColor(3);

    ms = ImGui::EndMultiSelect();
    _logSelection.ApplyRequests(ms);

    // Copies the filtered, visible lines (in display order) to the clipboard; the selection-only
    // variant copies just the highlighted lines.
    auto copyText = [&](bool selectedOnly) {
      std::string out;
      for (Entry const* entry : visible) {
        if (selectedOnly && !_logSelection.Contains(static_cast<ImGuiID>(entry->id))) {
          continue;
        }
        out += entry->message;
        out.push_back('\n');
      }
      if (!out.empty()) {
        ImGui::SetClipboardText(out.c_str());
      }
    };

    // Ctrl+C copies the current selection when the log area is focused.
    if (ImGui::IsWindowFocused() && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_C)) {
      copyText(/*selectedOnly=*/true);
    }

    // Right-click menu (does not alter the current selection).
    if (ImGui::BeginPopupContextWindow("LogContextMenu")) {
      if (ImGui::MenuItem("Copy selected", "Ctrl+C", false, _logSelection.Size > 0)) {
        copyText(/*selectedOnly=*/true);
      }
      if (ImGui::MenuItem("Copy all (filtered)")) {
        copyText(/*selectedOnly=*/false);
      }
      ImGui::EndPopup();
    }

    // Auto-scroll only when already at the bottom.
    if (_logView->hasNewEntries && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
      ImGui::SetScrollHereY(1.0f);
    }
    _logView->hasNewEntries = false;
  }
  ImGui::EndChild();
  ImGui::End();
  return settingsChanged;
}

bool LogConsole::HasNewErrors() {
  if (!_logView) {
    return false;
  }
  return _logView->hasNewErrors.exchange(false, std::memory_order_relaxed);
}

} // namespace superdex::studio
