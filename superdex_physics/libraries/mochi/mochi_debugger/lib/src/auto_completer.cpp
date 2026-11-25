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

#include <mochi_debugger/lib/auto_completer.h>

#include <mochi_core/utils/container_utils.h>
#include <mochi_debugger/lib/console_utils.h>
#include <mochi_debugger/lib/debug_client.h>
#include <simple_reflection/simple_reflection_console.h>

#include <algorithm>
#include <utility>

namespace mochi::dbg {

AutoCompleter::AutoCompleter(AutoCompleterHost host) : _host(std::move(host)) {}

AutoCompleterHost AutoCompleter::MakeHost(DebugClient& client) {
  return {
      .send = [&client](protocol::AutoCompleteRequest const& request) { client.Send(request); },
      .workingDirectory = [&client] { return client.GetWorkingDirectory(); },
      .commandKind = [&client](std::string_view word) { return client.GetCommandKind(word); },
      .connectionGeneration = [&client] { return client.GetConnectionGeneration(); },
      .now = [] { return std::chrono::steady_clock::now().time_since_epoch(); }};
}

std::optional<AutoCompleter::Step> AutoCompleter::FindStep(std::string_view text) const {
  std::string error;
  auto const tokens = SReflect::detail::ConsoleTokenize(text, error);
  bool const isTyping =
      !tokens.empty() && tokens.back().offset + tokens.back().length == text.size();
  Step step;
  if ((isTyping ? tokens.size() - 1 : tokens.size()) > 0) {
    switch (_host.commandKind(tokens.front().text)) {
      case ConsoleCommandKind::Remote:
        break;
      case ConsoleCommandKind::LocalWithPath:
        step.pathTokenIndex = 1;
        step.directoriesOnly = true;
        break;
      case ConsoleCommandKind::Local:
        return std::nullopt;
    }
  }
  auto const completion = SReflect::detail::FindConsoleCompletion(text, step.pathTokenIndex);
  if (!completion ||
      (completion->target == SReflect::detail::ConsoleCompletionTarget::Argument &&
       step.pathTokenIndex != 0)) {
    return std::nullopt;
  }
  step.prefix = text.substr(0, completion->offset);
  step.typed = text.substr(completion->offset);
  return step;
}

bool AutoCompleter::Covers(State const& state, Reply const& reply) {
  if (!state.step || reply.step.prefix != state.step->prefix ||
      std::string_view{reply.message.workingDirectory} != state.workingDirectory) {
    return false;
  }
  // A truncated reply leaves out candidates that a longer typed text could still match.
  std::string_view const typed = state.step->typed;
  return reply.message.result._isTruncated ? typed == reply.step.typed
                                           : typed.starts_with(reply.step.typed);
}

bool AutoCompleter::HasCoveringReply(State const& state) {
  return state.reply && Covers(state, *state.reply);
}

DynamicArray<std::string> AutoCompleter::MatchingCandidates(
    State const& state,
    size_t& outDefault) {
  auto const& result = state.reply->message.result;
  std::string_view const typed = state.step->typed;
  DynamicArray<std::string> matches;
  outDefault = 0;
  for (size_t i = 0; i < result._candidates.size(); ++i) {
    if (result._candidates[i].starts_with(typed)) {
      if (i == result._defaultIndex) {
        outDefault = matches.size();
      }
      matches.push_back(result._candidates[i]);
    }
  }
  return matches;
}

void AutoCompleter::Reset(State& state) const {
  state.reply.reset();
  state.cycling.reset();
  state.sentWorkingDirectory.clear();
  state.sentPrefix.clear();
  // Later requests get later timestamps, so their replies are kept.
  state.resetTime = state.sentTime = std::max(_host.now(), state.sentTime);
}

void AutoCompleter::Refresh(State& state) const {
  state.workingDirectory = _host.workingDirectory();
  uint64_t const generation = _host.connectionGeneration();
  if (generation != state.connectionGeneration) {
    state.connectionGeneration = generation;
    Reset(state);
  }
}

AutoCompleter::Request AutoCompleter::MakeRequest(State& state) const {
  auto const& step = *state.step;
  protocol::AutoCompleteRequest request;
  // Ask for the whole step, so later typing can be matched locally. A repeated request for the
  // same step includes the typed text, because only that gets the candidates a truncated reply
  // left out.
  bool const isRepeat =
      state.sentPrefix == step.prefix && state.sentWorkingDirectory == state.workingDirectory;
  request.command = isRepeat ? state.text : step.prefix;
  request.workingDirectory = state.workingDirectory;
  request.pathTokenIndex = step.pathTokenIndex;
  request.directoriesOnly = step.directoriesOnly;
  // Timestamps increase strictly, so each one identifies its request.
  state.sentTime = std::max(_host.now(), state.sentTime + std::chrono::nanoseconds{1});
  request.timestamp = static_cast<uint64_t>(state.sentTime.count());
  state.sentWorkingDirectory = state.workingDirectory;
  state.sentPrefix = step.prefix;
  state.isSentAnswered = false;
  return request;
}

AutoCompleter::Request AutoCompleter::RequestIfNeeded(State& state) const {
  if (!state.step) {
    return std::nullopt;
  }
  if (state.sentWorkingDirectory != state.workingDirectory ||
      state.sentPrefix != state.step->prefix) {
    return MakeRequest(state); // A new step or argument
  }
  if (!HasCoveringReply(state) &&
      (state.isSentAnswered || _host.now() - state.sentTime >= kReplyTimeout)) {
    return MakeRequest(state); // The last reply does not cover the input, or never came
  }
  return std::nullopt;
}

void AutoCompleter::Send(Request const& request) const {
  if (request) {
    _host.send(*request);
  }
}

void AutoCompleter::SetInput(std::string_view text, bool isCursorAtEnd) {
  Request const request = _state.Mutate([&](State& state) {
    if (state.cycling && (text != state.cycling->text || !isCursorAtEnd)) {
      state.cycling.reset();
    }
    state.text = text;
    state.isCursorAtEnd = isCursorAtEnd;
    Refresh(state);
    // A blank line, such as a new prompt, sends no request and starts the next command fresh. A
    // request for the first step of a relative path still has an empty command.
    if (text.find_first_not_of(" \t\n\v\f\r") == std::string_view::npos) {
      state.step.reset();
      Reset(state);
      return Request{};
    }
    state.step = FindStep(text);
    return RequestIfNeeded(state);
  });
  Send(request);
}

bool AutoCompleter::OnReply(protocol::AutoCompleteReply&& reply) {
  auto step = FindStep(reply.command);
  if (!step) {
    return false;
  }
  DynamicArray<std::string> notes;
  if (!reply.result._note.empty()) {
    auto const placeholder = [](std::string_view note) {
      return "<" + FormatConsoleTypeName(note) + ">";
    };
    notes.push_back(placeholder(reply.result._note));
    for (auto const& note : reply.result._laterArgumentNotes) {
      notes.push_back(placeholder(note));
    }
  }
  Reply received{.message = std::move(reply), .step = std::move(*step), .notes = std::move(notes)};
  auto const sent = std::chrono::nanoseconds{static_cast<int64_t>(received.message.timestamp)};
  return _state.Mutate([&](State& state) {
    if (sent <= state.resetTime) {
      return false; // From before the last reset
    }
    if (sent == state.sentTime) {
      state.isSentAnswered = true;
    }
    if (!Covers(state, received) ||
        (HasCoveringReply(state) && state.reply->message.timestamp >= received.message.timestamp)) {
      return false;
    }
    state.reply = std::move(received);
    return true;
  });
}

std::optional<AutoCompleter::Cycling> AutoCompleter::StartCycling(State& state, Request& outRequest)
    const {
  if (!state.isCursorAtEnd || !state.step) {
    return std::nullopt;
  }
  if (!HasCoveringReply(state)) {
    outRequest = RequestIfNeeded(state);
    return std::nullopt;
  }
  Cycling cycling;
  cycling.prefix = state.step->prefix;
  cycling.candidates = MatchingCandidates(state, cycling.index);
  if (cycling.candidates.empty()) {
    return std::nullopt;
  }
  return cycling;
}

std::string AutoCompleter::Insert(State& state, Cycling cycling) const {
  cycling.text = cycling.prefix + cycling.candidates[cycling.index];
  state.text = cycling.text;
  state.isCursorAtEnd = true;
  state.step = FindStep(state.text);
  state.cycling = std::move(cycling);
  return state.cycling->text;
}

std::optional<std::string> AutoCompleter::Cycle(bool backward) {
  Request request;
  auto text = _state.Mutate([&](State& state) -> std::optional<std::string> {
    Refresh(state);
    std::optional<Cycling> cycling = state.cycling;
    if (cycling) {
      size_t const count = cycling->candidates.size();
      cycling->index = (cycling->index + (backward ? count - 1 : 1)) % count;
    } else {
      cycling = StartCycling(state, request);
      if (!cycling) {
        return std::nullopt;
      }
      if (backward) {
        cycling->index = cycling->candidates.size() - 1;
      }
    }
    return Insert(state, std::move(*cycling));
  });
  Send(request);
  return text;
}

std::optional<std::string> AutoCompleter::Select(size_t index) {
  Request request;
  auto text = _state.Mutate([&](State& state) -> std::optional<std::string> {
    Refresh(state);
    std::optional<Cycling> cycling = state.cycling ? state.cycling : StartCycling(state, request);
    if (!cycling || index >= cycling->candidates.size()) {
      return std::nullopt;
    }
    cycling->index = index;
    return Insert(state, std::move(*cycling));
  });
  Send(request);
  return text;
}

AutoCompleter::View AutoCompleter::GetView() const {
  uint64_t const generation = _host.connectionGeneration();
  return _state.Read([&](State const& state) {
    View view;
    view.text = state.text;
    if (generation != state.connectionGeneration) {
      return view; // Everything stored is from an earlier connection
    }
    DynamicArray<std::string> notes;
    if (state.reply) {
      notes = state.reply->notes;
    }
    if (state.cycling) {
      view.candidates = state.cycling->candidates;
      view.selected = state.cycling->index;
      view.offset = state.cycling->prefix.size();
      view.isCycling = true;
      view.annotations = std::move(notes);
      return view;
    }
    if (!state.isCursorAtEnd || !HasCoveringReply(state)) {
      return view;
    }
    view.candidates = MatchingCandidates(state, view.selected);
    view.offset = state.step->prefix.size();
    std::string_view const typed = state.step->typed;
    if (!typed.empty() && !view.candidates.empty()) {
      view.preview = view.candidates[view.selected].substr(typed.size());
    }
    if (!view.preview.empty() && view.candidates.size() > 1) {
      view.annotations.push_back("(+" + std::to_string(view.candidates.size() - 1) + ")");
    }
    Append(view.annotations, notes);
    return view;
  });
}

} // namespace mochi::dbg
