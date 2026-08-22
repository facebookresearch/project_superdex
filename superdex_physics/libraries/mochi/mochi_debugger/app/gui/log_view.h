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

#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>

#include <cstddef>
#include <deque>
#include <iosfwd>
#include <memory>
#include <string>

namespace mochi::dbg {

// Defined in log_view.cpp
class LogHook;

// Class to collect logging data. Used for the log panel UI.
class LogView {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(LogView);

 public:
  LogView();
  ~LogView();

  struct Entry {
    mochi::LogChannel channel = {};
    std::string message;
    std::string sourceFile;
    int sourceLine = 0;
  };

  // Maximum number of retained log entries. Once full, the oldest entry is dropped.
  static constexpr size_t kMaxEntries = 1000;

  // Appends a log entry, setting the appropriate "new entries" flags. Thread-safe.
  void
  AddLine(mochi::LogChannel channel, std::string message, std::string sourceFile, int sourceLine);

  // Removes all log entries and resets the "new entries" flags. Thread-safe.
  void Clear();

  // Returns whether new errors or warnings have arrived since the flags were last consumed.
  // Thread-safe.
  bool HasNewErrorsOrWarnings() const;

  // Resets all "new entries" flags (including the new-error and new-warning flags read by
  // @ref HasNewErrorsOrWarnings) and returns whether new entries had arrived since they were last
  // consumed (drives scroll-to-bottom). Thread-safe.
  bool ConsumeNewEntries();

  // Invokes visitor(Entry const&) for each retained entry, oldest to newest, under the read lock.
  template <typename Visitor>
  void ReadEntries(Visitor visitor) const {
    _shared->state.Read([&](State const& state) {
      for (Entry const& entry : state.entries) {
        visitor(entry);
      }
    });
  }

 private:
  // All mutable log state, guarded by @ref SharedState::state. Entries are held
  // oldest-to-newest; once the count exceeds @ref kMaxEntries the oldest entries are dropped
  // from the front.
  struct State {
    std::deque<Entry> entries;
    bool hasNewEntries = false;
    bool hasNewErrors = false;
    bool hasNewWarnings = false;
  };

  // State shared with the std::cout/std::cerr and mochi log callbacks. The callbacks capture a
  // std::shared_ptr<SharedState> (not a raw @ref LogView pointer), so any callback that is
  // in-flight when @ref ~LogView begins keeps this object alive via its captured shared_ptr and
  // can safely complete before it is destroyed.
  struct SharedState {
    Guarded<State> state;
    // The stream hooks live here so the mochi callback's ForwardRaw stays valid for the same
    // lifetime as the callbacks that reference it.
    std::unique_ptr<LogHook> coutHook;
    std::unique_ptr<LogHook> cerrHook;
  };

  // Appends an entry under the state lock and updates the "new entries" flags. Shared by all
  // three log callbacks (cout, cerr, mochi) so they cannot drift out of sync.
  static void AppendEntry(
      SharedState& shared,
      mochi::LogChannel channel,
      std::string message,
      std::string sourceFile,
      int sourceLine);

  std::shared_ptr<SharedState> _shared;
  std::streambuf* _prevCoutBuf = nullptr;
  std::streambuf* _prevCerrBuf = nullptr;
  mochi::LogFn _prevMochiFn = nullptr;
};

} // namespace mochi::dbg
