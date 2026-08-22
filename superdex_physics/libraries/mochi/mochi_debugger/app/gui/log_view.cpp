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

#include "log_view.h"

#include <mochi_core/utils/guarded.h>

#include <functional>
#include <iostream>
#include <memory>
#include <streambuf>
#include <string>
#include <string_view>
#include <utility>

namespace mochi::dbg {

//------------------------------------------------------------------------------------------------
// LogHook
//------------------------------------------------------------------------------------------------

// Class to monitor std::cout or std::cerr.
class LogHook : public std::streambuf {
 public:
  using Callback = std::function<void(std::string line)>;
  using int_type = std::streambuf::int_type;

  // In addition to being captured, text will also be forwarded to a stream buffer (e.g. the
  // original cout or cerr). Callback will be fired for each line collected.
  LogHook(std::streambuf* outputBuf, Callback callback);

  // Override overflow to capture text one character at a time.
  int_type overflow(int_type c) override;

  // Forwards text directly to the wrapped output buffer without capturing it. Serialized on the
  // same lock as @ref overflow, so writes from other threads cannot interleave. Thread-safe.
  void ForwardRaw(std::string_view text);

 private:
  std::streambuf* _outputBuf = nullptr;
  Callback _callback;
  Guarded<std::string> _line; // accumulates characters until a newline completes a line
};

LogHook::LogHook(std::streambuf* outputBuf, Callback callback)
    : _outputBuf(outputBuf), _callback(std::move(callback)) {}

std::streambuf::int_type LogHook::overflow(int_type c) {
  // Accumulate the character and forward it under the lock; fire the callback afterwards (outside
  // the lock) to avoid holding it across arbitrary user code.
  std::string completedLine;
  bool lineComplete = false;
  int_type const result = _line.Mutate([&](auto& line) {
    if (c != EOF) {
      line.push_back(static_cast<char>(c));
      if (c == '\n') {
        completedLine = std::move(line);
        line.clear();
        lineComplete = true;
      }
    }
    // Forward real characters only; signal success for EOF without writing it as a data byte.
    return (c == EOF) ? traits_type::not_eof(c) : _outputBuf->sputc(c);
  });

  if (lineComplete) {
    _callback(std::move(completedLine));
  }
  return result;
}

void LogHook::ForwardRaw(std::string_view text) {
  _line.Mutate(
      [&](auto&) { _outputBuf->sputn(text.data(), static_cast<std::streamsize>(text.size())); });
}

//------------------------------------------------------------------------------------------------
// LogView
//------------------------------------------------------------------------------------------------

void LogView::AppendEntry(
    SharedState& shared,
    mochi::LogChannel channel,
    std::string message,
    std::string sourceFile,
    int sourceLine) {
  shared.state.Mutate([&](State& s) {
    s.entries.push_back(Entry{channel, std::move(message), std::move(sourceFile), sourceLine});
    if (s.entries.size() > kMaxEntries) {
      // Cap reached: drop the oldest entry.
      s.entries.pop_front();
    }

    s.hasNewEntries = true;
    if (channel == LogChannel::Error) {
      s.hasNewErrors = true;
    } else if (channel == LogChannel::Warning) {
      s.hasNewWarnings = true;
    }
  });
}

LogView::LogView() : _shared(std::make_shared<SharedState>()) {
  // Callbacks capture a std::shared_ptr<SharedState> by value (never `this`). If a callback is
  // invoked on a background thread after ~LogView begins, its captured shared_ptr keeps
  // SharedState alive until the callback returns.

  // Intercept cout
  _prevCoutBuf = std::cout.rdbuf();
  _shared->coutHook =
      std::make_unique<LogHook>(_prevCoutBuf, [shared = _shared](std::string message) {
        AppendEntry(*shared, LogChannel::Info, std::move(message), "", 0);
      });
  std::cout.rdbuf(_shared->coutHook.get());

  // Intercept cerr
  _prevCerrBuf = std::cerr.rdbuf();
  _shared->cerrHook =
      std::make_unique<LogHook>(_prevCerrBuf, [shared = _shared](std::string message) {
        AppendEntry(*shared, LogChannel::Error, std::move(message), "", 0);
      });
  std::cerr.rdbuf(_shared->cerrHook.get());

  // Intercept mochi logging
  _prevMochiFn = mochi::GetLogCallback();
  mochi::SetLogCallback(
      [shared = _shared](
          LogChannel channel, char const* message, char const* sourceFile, int sourceLine) {
        if (!message) {
          message = "";
        }
        if (!sourceFile) {
          sourceFile = "";
        }
        AppendEntry(*shared, channel, std::string(message), std::string(sourceFile), sourceLine);

        // Forward Mochi messages to the original std::cout buffer (NOT through our LogHook, which
        // would re-capture them). ForwardRaw serializes with the hook's own writes.
        auto messageForDebugger = mochi::Format("%s(%d): %s", sourceFile, sourceLine, message);
        shared->coutHook->ForwardRaw(messageForDebugger);

#if MOCHI_PLATFORM_WINDOWS
        // On Windows, debuggers like VSCode and Visual Studio Pro do not show std::out in the debug
        // output window. There is a special function which puts text there. It is thread-safe.
        //
        // OutputDebugStringA truncates long strings (the limit is debugger-dependent, historically
        // ~4 KB). Therefore, split it into chunks if it is too long.
        constexpr size_t kMaxDebugChunk = 4000;
        if (messageForDebugger.size() <= kMaxDebugChunk) {
          ::OutputDebugStringA(messageForDebugger.c_str());
        } else {
          std::string_view remaining = messageForDebugger;
          while (!remaining.empty()) {
            std::string_view const chunk = remaining.substr(0, kMaxDebugChunk);
            ::OutputDebugStringA(std::string(chunk).c_str());
            remaining.remove_prefix(chunk.size());
          }
        }
#endif // MOCHI_PLATFORM_WINDOWS
      });
}

LogView::~LogView() {
  // Restore global state so no NEW callback invocations start against our SharedState.
  std::cout.rdbuf(_prevCoutBuf);
  std::cerr.rdbuf(_prevCerrBuf);
  mochi::SetLogCallback(_prevMochiFn);
  // Any callback that has already been dispatched holds its own shared_ptr<SharedState> copy,
  // keeping SharedState alive until it returns. Releasing our own reference here is safe.
}

void LogView::AddLine(
    mochi::LogChannel channel,
    std::string message,
    std::string sourceFile,
    int sourceLine) {
  AppendEntry(*_shared, channel, std::move(message), std::move(sourceFile), sourceLine);
}

void LogView::Clear() {
  _shared->state.Mutate([](State& state) {
    state.entries.clear();
    state.hasNewEntries = false;
    state.hasNewErrors = false;
    state.hasNewWarnings = false;
  });
}

bool LogView::HasNewErrorsOrWarnings() const {
  return _shared->state.Read(
      [](State const& state) { return state.hasNewErrors || state.hasNewWarnings; });
}

bool LogView::ConsumeNewEntries() {
  return _shared->state.Mutate([](State& state) {
    bool const hadNewEntries = state.hasNewEntries;
    state.hasNewEntries = false;
    state.hasNewErrors = false;
    state.hasNewWarnings = false;
    return hadNewEntries;
  });
}

} // namespace mochi::dbg
