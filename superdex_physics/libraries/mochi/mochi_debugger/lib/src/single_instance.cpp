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

#include <mochi_debugger/lib/single_instance.h>

#include <mochi_core/net/message_client.h>
#include <mochi_core/utils/log.h>
#include <mochi_physics/dbg/protocol.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <thread>
#include <utility>

namespace mochi::dbg {

static constexpr int kSingletonMaxClients = 2;
static constexpr double kForwardTimeoutSeconds = 1.0;

struct CommandLineRequest : net::RequestMessage {
  using Reply = CommandLineReply;
  DynamicArray<std::string> args;

  MOCHI_STRUCT_BEGIN(mochi::dbg::CommandLineRequest)
  MOCHI_BASE_CLASS(net::RequestMessage)
  MOCHI_FIELD(args)
  MOCHI_STRUCT_END()
};

struct CommandLineReply : net::ReplyMessage {
  using net::ReplyMessage::ReplyMessage;
  std::string error; // Empty means success

  MOCHI_STRUCT_BEGIN(mochi::dbg::CommandLineReply)
  MOCHI_BASE_CLASS(net::ReplyMessage)
  MOCHI_FIELD(error);
  MOCHI_STRUCT_END()
};

SingleInstanceHelper::SingleInstanceHelper(CommandLine const& cli, Span<std::string const> args) {
  // Init Server
  _server.Register<CommandLineRequest>(
      [this](net::ClientId client, auto const& request) { OnRequest(client, request); });
  _server.Register<CommandLineReply>();
  _server.SetVersion(_server.CalcProtocolVersionHash());

  // Try to start the server using kSingletonPort
  _server.Start(kSingletonPort, kSingletonMaxClients, "mochi_debugger_singleton");

  // If we successfully bound kSingletonPort, then we must be the only instance running on this
  // machine.
  bool const isFirstInstance = (_server.GetPort() == kSingletonPort);
  if (!isFirstInstance) {
    _server.Stop();
  }

  // Optionally forward our arguments to the first instance. If successful, then this process should
  // exit. It is possible that ForwardArgs could fail, despite the existence of another instance. In
  // that case, this process will not exit. It's better to open a 2nd window than to have nothing
  // visible happen.
  if (cli.singleton && !isFirstInstance) {
    _shouldExit = ForwardArgs(args);
  }
}

SingleInstanceHelper::~SingleInstanceHelper() = default;

bool SingleInstanceHelper::ShouldExit() const {
  return _shouldExit;
}

std::optional<CommandLine> SingleInstanceHelper::PollCommandLine() {
  std::optional<CommandLine> cli;
  _inbox.Mutate([&](auto& inbox) {
    if (!inbox.empty()) {
      cli = std::move(inbox.front());
      inbox.pop_front();
    }
  });
  return cli;
}

void SingleInstanceHelper::OnRequest(net::ClientId client, CommandLineRequest const& request) {
  CommandLineReply reply{request};
  auto parsed = CommandLine::Parse(request.args);
  if (parsed.second.empty()) {
    _inbox.Mutate([&](auto& inbox) { inbox.emplace_back(std::move(parsed.first)); });
  } else {
    reply.error = std::move(parsed.second);
  }
  _server.SendTo(client, reply);
}

// This function attempts to forward command line arguments to another instance of the app.
// It returns true if arguments were successfully delivered.
bool SingleInstanceHelper::ForwardArgs(Span<std::string const> args) {
  // Init client
  net::MessageClient client;
  client.Register<CommandLineRequest>();
  client.Register<CommandLineReply>();
  client.SetVersion(client.CalcProtocolVersionHash());

  // Attempt to connect to kSingletonPort on this machine.
  client.Connect("127.0.0.1", kSingletonPort);

  // Wait for connection or failure
  using clock = std::chrono::steady_clock;
  auto const start = clock::now();
  for (;;) {
    auto const status = client.GetStatus();
    if (status == net::SocketStatus::Connected) {
      break;
    } else if (status == net::SocketStatus::Lost) {
      return false; // Connect failed
    }

    double const elapsed = std::chrono::duration<double>(clock::now() - start).count();
    if (elapsed > kForwardTimeoutSeconds) {
      return false; // Give up
    }
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Send command line arguments (except for "--singleton")
  Error error;
  CommandLineRequest request;
  request.args = args;
  request.args.erase(
      std::remove(
          request.args.begin(), request.args.end(), std::string{CommandLine::kSingletonArg}),
      request.args.end());
  auto reply = client.SendAndAwaitReply(request, kForwardTimeoutSeconds, error);
  return error.IsOK() && reply.error.empty();
}

} // namespace mochi::dbg
