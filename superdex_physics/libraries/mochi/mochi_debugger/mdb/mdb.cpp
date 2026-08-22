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
#include <mochi_debugger/lib/debug_client.h>

#include <string>

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

  // Main loop
  std::string input;
  for (;;) {
    terminal.SetPrompt(client.GetCommandPrompt());
    if (!terminal.ReadLine(input)) {
      break;
    }
    client.ExecuteCommand(input);
    if (client.WasExitRequested()) {
      break;
    }
  }

  return 0;
}
