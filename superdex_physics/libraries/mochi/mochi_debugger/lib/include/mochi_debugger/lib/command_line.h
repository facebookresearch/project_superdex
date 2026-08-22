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
#include <mochi_core/utils/span.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace mochi::dbg {

/**
 * @brief Arguments parsed from the command line.
 */
struct CommandLine {
  static constexpr std::string_view kConnectArg = "--connect";
  static constexpr std::string_view kSingletonArg = "--singleton";

  /// Empty means no connection was requested.
  std::string address;
  uint16_t port = 0;

  /// Forward this launch to an existing mochi_debugger instance, if one exists.
  bool singleton = false;

  /**
   * @brief Parse a list of command line arguments.
   *
   * @details If a valid argument is listed more than once, then the last occurrence wins.
   *
   * @param[in] args Incoming arguments
   * @return Pair of CommandLine and error string (empty means "OK").
   */
  static std::pair<CommandLine, std::string> Parse(Span<std::string const> args);

  /// Skips the first argument (presumed to be the application path).
  static DynamicArray<std::string> ArgsFromMain(int argc, char** argv);

  bool operator==(CommandLine const& rhs) const = default;
};

} // namespace mochi::dbg
