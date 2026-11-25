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
#include <cstdint>
#include <deque>
#include <iosfwd>
#include <memory>
#include <string>

namespace mochi::dbg {

// Defined in log_view.cpp
class LogHook;

// Collect log-panel data and intercept process-wide streams. Construct before logging threads
// start and destroy only after those threads have stopped.
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
    uint64_t sequence = 0;
  };

  // Maximum number of retained log entries. Once full, the oldest entry is dropped.
  static constexpr size_t kMaxEntries = 1000;

  // Appends a log entry, setting the appropriate "new entries" flags. Thread-safe.
  void
  AddLine(mochi::LogChannel channel, std::string message, std::string sourceFile, int sourceLine);

  // Removes all log entries and resets the "new entries" flags. Thread-safe.
  void Clear();

  // Resets the new-error and new-warning flags and returns whether either was set (drives raising
  // the log panel). Thread-safe.
  bool ConsumeNewErrorsOrWarnings();

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
    uint64_t nextSequence = 0;
    bool hasNewErrors = false;
    bool hasNewWarnings = false;
  };

  // State shared with the std::cout/std::cerr and mochi log callbacks. The callbacks capture a
  // std::shared_ptr<SharedState> (not a raw @ref LogView pointer), so any callback that is
  // in-flight when @ref ~LogView begins keeps this object alive via its captured shared_ptr and
  // can safely complete before it is destroyed.
  struct SharedState {
    Guarded<State> state;
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
  // Also retained by the Mochi callback so in-flight forwarding can complete during teardown.
  std::shared_ptr<LogHook> _coutHook;
  std::unique_ptr<LogHook> _cerrHook;
  std::streambuf* _prevCoutBuf = nullptr;
  std::streambuf* _prevCerrBuf = nullptr;
  mochi::LogFn _prevMochiFn = nullptr;
};

} // namespace mochi::dbg
