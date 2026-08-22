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

#include <mochi_debugger/lib/command_line.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_debugger/lib/address.h>
#include <utility>

using namespace mochi;
using namespace mochi::dbg;

std::pair<CommandLine, std::string> CommandLine::Parse(Span<std::string const> args) {
  CommandLine cli;
  for (int i = 0; i < isize(args); ++i) {
    if (args[i] == kConnectArg) {
      std::string nextTok;
      if (i + 1 < isize(args) && !args[i + 1].starts_with("--")) {
        nextTok = args[i + 1];
        ++i; // consume the target
      }
      Error error;
      auto [address, port] = ParseAddressAndPort(std::move(nextTok), error);
      if (!error.IsOK()) {
        return {{}, error.GetDescription()};
      }
      cli.address = std::move(address);
      cli.port = port;
    } else if (args[i] == kSingletonArg) {
      cli.singleton = true;
    } else {
      return {{}, Format("Unknown argument: %s", args[i].c_str())};
    }
  }
  return {std::move(cli), std::string{}};
}

DynamicArray<std::string> CommandLine::ArgsFromMain(int argc, char** argv) {
  // Skip first arg (the program path)
  if (argc > 1) {
    return DynamicArray<std::string>{argv + 1, argv + argc};
  } else {
    return {};
  }
}
