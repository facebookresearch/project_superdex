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

#include "core/settings.h"

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/log.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <streambuf>
#include <string>

#include <imguios/imguios.h>

namespace superdex::studio {

// Class to monitor std::cout or std::cerr.
class LogHook : public std::streambuf {
 public:
  using Callback = std::function<void(std::string line)>;
  using int_type = std::streambuf::int_type;

  // In addition to being captured, text will also be forwarded to a stream buffer (e.g. the the
  // original cout or cerr). Callback will be fired for each line collected.
  LogHook(std::streambuf* outputBuf, Callback callback);

  // Override overflow to capture text one character at a time
  int_type overflow(int_type c) override;

 private:
  std::streambuf* _outputBuf = nullptr;
  Callback _callback;
  std::string _line;
};

// Class to collect logging data. Used for the console window UI.
class LogView : public std::streambuf {
 public:
  LogView();
  LogView(LogView const&) = delete;
  LogView& operator=(LogView const&) = delete;
  ~LogView() override;

  void
  AddEntry(mochi::LogChannel channel, std::string message, char const* sourceFile, int sourceLine);

  struct Entry {
    mochi::LogChannel channel = {};
    std::string message;
    char const* sourceFile = nullptr; // string literal
    int sourceLine = 0;
    std::uint64_t id = 0; // stable, monotonic; used as the selection key in the console
  };

  // Public data:
  std::mutex mutex;
  mochi::DynamicArray<Entry> entries;
  bool hasNewEntries = false;
  std::atomic<bool> hasNewErrors = false;

 private:
  std::streambuf* _prevCoutBuf = nullptr;
  std::streambuf* _prevCerrBuf = nullptr;
  mochi::LogFn _prevMochiFn = nullptr;
  std::unique_ptr<LogHook> _coutHook;
  std::unique_ptr<LogHook> _cerrHook;
  std::uint64_t _nextEntryId = 0; // assigns Entry::id under `mutex`
};

class LogConsole {
 public:
  LogConsole();
  // Returns true when a channel toggle changed, so the caller can persist `settings`.
  bool ShowWindow(bool* open, LogConsoleSettings& settings);
  // Returns whether new errors arrived since the last call, clearing the flag on read.
  bool HasNewErrors();
  // Applies the persisted verbose setting to the logger. Verbose is gated at the source, so this
  // must run once after settings are loaded or a restored "on" state would capture nothing.
  static void ApplyVerboseCapture(LogConsoleSettings const& settings);

 private:
  std::unique_ptr<LogView> _logView;
  ImGuiTextFilter _textFilter;
  // Selection state for the log lines (keyed by Entry::id). Drives multi-select + copy.
  ImGuiSelectionBasicStorage _logSelection;
};

} // namespace superdex::studio
