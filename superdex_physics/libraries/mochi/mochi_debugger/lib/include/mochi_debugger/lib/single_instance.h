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

#include <mochi_core/net/message_server.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_debugger/lib/command_line.h>

#include <cstdint>
#include <deque>
#include <optional>
#include <string>

namespace mochi::dbg {

struct CommandLineRequest;
struct CommandLineReply;

/**
 * @brief Owns a hidden client server pair to make mochi_debugger act as a single instance
 * application. If another instance is already running, this instance will forward its command line
 * arguments to the primary instance and exit.
 */
class SingleInstanceHelper final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(SingleInstanceHelper);

 public:
  /**
   * @brief Construct helper
   *
   * @details if @ref CommandLine::singleton is set, then check for another instance of the app that
   * is already running. If found, forward the string arguments and set the exit flag.
   *
   * @param[in] cli Command line arguments that have been parsed for this process.
   * @param[in] args String command line arguments to potentially forward.
   */
  explicit SingleInstanceHelper(CommandLine const& cli, Span<std::string const> args);
  ~SingleInstanceHelper();

  /**
   * @brief Return true if arguments were forwarded to another instance of the app.
   * In that case, the current process should exit.
   */
  [[nodiscard]] bool ShouldExit() const;

  /** @brief Poll for new command line arguments from another instance. */
  std::optional<CommandLine> PollCommandLine();

 private:
  void OnRequest(net::ClientId client, CommandLineRequest const& request);
  static bool ForwardArgs(Span<std::string const> args);

  Guarded<std::deque<CommandLine>> _inbox;
  net::MessageServer _server;
  bool _shouldExit = false;
};

} // namespace mochi::dbg
