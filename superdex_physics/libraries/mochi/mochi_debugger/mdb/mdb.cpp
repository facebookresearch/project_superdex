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

#include "terminal.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/log.h>
#include <mochi_debugger/lib/address.h>
#include <mochi_debugger/lib/auto_completer.h>
#include <mochi_debugger/lib/debug_client.h>
#include <mochi_physics/dbg/protocol.h>

#include <mutex>
#include <string>
#include <string_view>
#include <utility>

using namespace mochi;

static constexpr Color kErrorColor = MakeColor(0xE06C75FF);
static constexpr Color kWarningColor = MakeColor(0xE5C07BFF);

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  // Create terminal
  dbg::Terminal terminal;
  terminal.Print("Mochi Debugger CLI\n");
  terminal.Print("Type 'help' for more information\n\n");

  // Hook logging
  SetLogCallback([&](LogChannel channel, char const* message, char const* file, int line) {
    MOCHI_ASSERT_VERBOSE(message != nullptr);
    std::optional<Color> color;
    if (channel == LogChannel::Error) {
      color = kErrorColor;
    } else if (channel == LogChannel::Warning) {
      color = kWarningColor;
    }
    if (file && *file) {
      terminal.Print(Format("%s:%d: %s", file, line, message), color);
    } else {
      terminal.Print(message, color); // Without file and line
    }
  });
  MOCHI_DEFER(SetLogCallback(nullptr)); // Restore default logging

  // Create client
  dbg::DebugClient client;
  client.SetSettings({.autoSelectScene = false});

  // Tab completion. Replies arrive on a network thread, so they update the suggestion directly.
  // Reading the view and showing it happen under one lock, so a view read before a reply arrived
  // cannot replace the newer view that the reply shows.
  dbg::AutoCompleter completer(dbg::AutoCompleter::MakeHost(client));
  std::mutex suggestionMutex;
  auto const showSuggestion = [&] {
    std::lock_guard const lock(suggestionMutex);
    auto const view = completer.GetView();
    terminal.SetSuggestion(view.text, view.preview, view.annotation);
  };
  client.Register<dbg::protocol::AutoCompleteReply>([&](dbg::protocol::AutoCompleteReply&& reply) {
    if (completer.OnReply(std::move(reply))) {
      showSuggestion();
    }
  });
  MOCHI_DEFER(client.ShutDown());
  terminal.SetCompletion(
      [&](std::string_view text, bool isCursorAtEnd) {
        completer.SetInput(text, isCursorAtEnd);
        showSuggestion();
      },
      [&](bool backward) { return completer.Cycle(backward); });

  client.Connect(dbg::kLocalHost, dbg::kDefaultDebugServerPort);

  // Main loop
  std::string input;
  for (;;) {
    terminal.SetPrompt(client.GetCommandPrompt());
    if (!terminal.ReadLine(input)) {
      break;
    }
    client.SetConsoleLineWidth(terminal.GetColumnCount());
    client.QueueCommand(input);
    client.WaitForQueuedCommands();
    if (client.WasExitRequested()) {
      break;
    }
  }

  return 0;
}
