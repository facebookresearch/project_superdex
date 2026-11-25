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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_debugger/lib/debug_client.h>
#include <mochi_physics/dbg/protocol.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace mochi::dbg {

/// @brief What an @ref AutoCompleter needs from the debug client. Each function must be
/// thread-safe.
struct AutoCompleterHost {
  std::function<void(protocol::AutoCompleteRequest const&)> send;
  std::function<std::string()> workingDirectory;
  std::function<ConsoleCommandKind(std::string_view word)> commandKind;
  std::function<uint64_t()> connectionGeneration; ///< See @ref DebugClient::GetConnectionGeneration
  std::function<std::chrono::nanoseconds()> now; ///< Any monotonic clock
};

/**
 * @brief Tab completion for a console input line, shared by the debugger terminals.
 *
 * @details The terminal reports every edit and cursor movement with @ref SetInput, Tab and
 * Shift+Tab with @ref Cycle, and server replies with @ref OnReply. It shows @ref GetView.
 *
 * - A request goes to the server at the start of each path step or argument. Typing within it
 *   filters the reply's candidates locally, so what is shown shrinks as the user types instead of
 *   waiting for the server.
 * - A reply counts only for the step it was requested for, and only while the text typed in that
 *   step still extends the text it was requested with. The newest such reply wins.
 * - While no reply covers the input, an input change sends a new request once the previous one is
 *   answered or older than @ref kReplyTimeout, so an unresponsive server gets at most one request
 *   per timeout. A repeated request for the same step includes the typed text.
 * - A blank line, such as a new prompt, forgets all replies, and so does connecting or
 *   disconnecting. Replies to requests sent before that are ignored.
 *
 * @note All public methods are thread-safe. Requests are sent without holding the completer's
 * lock, so a host may deliver the reply before its send function returns.
 */
class AutoCompleter {
 public:
  static constexpr std::chrono::milliseconds kReplyTimeout{500};

  /** @brief What a terminal shows for the current input. */
  struct View {
    /// The input line this view is for
    std::string text;
    /// Text that Tab inserts at the cursor. Empty until something is typed in the step or argument,
    /// and while cycling.
    std::string preview;
    /// Shown after the preview: "(+N)" for the other matching candidates, then "# " and the note.
    std::string annotation;
    /// Candidates that match the text typed so far, or the candidates being cycled through
    DynamicArray<std::string> candidates;
    /// The candidate Tab inserts, or the one cycling inserted. Ignored if there are no candidates.
    size_t selected = 0;
    /// Byte offset in the input where the candidates start
    size_t offset = 0;
    bool isCycling = false;
  };

  explicit AutoCompleter(AutoCompleterHost host);

  /** @brief A host that sends requests through client and reads its state. */
  static AutoCompleterHost MakeHost(DebugClient& client);

  /**
   * @brief Report the input line after any edit or cursor movement, which ends a Tab cycle unless
   * it leaves the text that @ref Cycle or @ref Select produced unchanged, with the cursor at the
   * end. Sends a request if the step or argument at the end needs one.
   */
  void SetInput(std::string_view text, bool isCursorAtEnd);

  /**
   * @brief Store a server reply. Call with every @ref protocol::AutoCompleteReply.
   *
   * @return True if the view changed.
   */
  bool OnReply(protocol::AutoCompleteReply&& reply);

  /**
   * @brief Complete the input for Tab, or Shift+Tab when backward is true.
   *
   * @details The first call inserts the candidate Tab would insert, or the last candidate when
   * backward. Calls without input changes between them replace it with the next or previous
   * candidate that matches the text typed before the first call, wrapping around. Nothing completes
   * while the cursor is not at the end of the input.
   *
   * @return The new input line, or nullopt if nothing completes.
   */
  std::optional<std::string> Cycle(bool backward);

  /**
   * @brief Insert @ref View::candidates[index], as cycling does.
   *
   * @return The new input line, or nullopt if index is out of range.
   */
  std::optional<std::string> Select(size_t index);

  [[nodiscard]] View GetView() const;

 private:
  // What the end of an input line completes
  struct Step {
    std::string prefix; // Text before the step or argument
    std::string typed; // Text typed in it so far
    uint32_t pathTokenIndex = 0;
    bool directoriesOnly = false;
  };

  struct Reply {
    protocol::AutoCompleteReply message;
    Step step; // What the request completed
    std::string note; // The result's note as the terminals show it
  };

  struct Cycling {
    std::string prefix; // Text before the step or argument being completed
    DynamicArray<std::string> candidates;
    size_t index = 0;
    std::string text; // The input that the last Cycle or Select produced
  };

  struct State {
    std::string text;
    bool isCursorAtEnd = true;
    std::string workingDirectory;
    uint64_t connectionGeneration = 0;
    std::optional<Step> step;
    std::optional<Reply> reply; // The newest reply that counted for the input when it arrived
    std::chrono::nanoseconds resetTime{}; // Replies to requests sent by then are ignored
    std::string sentWorkingDirectory; // Of the last request
    std::string sentPrefix; // Step::prefix of the last request
    std::chrono::nanoseconds sentTime{}; // The last request's timestamp, which identifies its reply
    bool isSentAnswered = false; // A reply to the last request arrived in time
    std::optional<Cycling> cycling;
  };

  using Request = std::optional<protocol::AutoCompleteRequest>;

  std::optional<Step> FindStep(std::string_view text) const;
  // Whether reply is the server's answer for the input's step and typed text
  static bool Covers(State const& state, Reply const& reply);
  static bool HasCoveringReply(State const& state);
  // The covering reply's candidates that match the typed text, and which of them is the default
  static DynamicArray<std::string> MatchingCandidates(State const& state, size_t& outDefault);
  // Forget all replies, and ignore replies to requests sent so far
  void Reset(State& state) const;
  // Read the client's state, forgetting replies from an earlier connection
  void Refresh(State& state) const;
  Request MakeRequest(State& state) const;
  Request RequestIfNeeded(State& state) const;
  std::optional<Cycling> StartCycling(State& state, Request& outRequest) const;
  std::string Insert(State& state, Cycling cycling) const;
  void Send(Request const& request) const;

  AutoCompleterHost const _host;
  Guarded<State> _state;
};

} // namespace mochi::dbg
