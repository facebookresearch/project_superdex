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

#include "mochi_debugger_test.h"

#include <mochi_core/net/client_socket.h> // for net::SocketStatus
#include <mochi_core/test/wait_until.h>
#include <mochi_core/utils/defer.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>

using namespace mochi;
using namespace mochi::dbg;

TEST_F(MochiDebuggerTest, ServerStartStop) {
  EXPECT_EQ(_server->HasStarted(), false);
  EXPECT_FALSE(_server->HasConnection());
  EXPECT_EQ(_server->GetPort(), 0);
  StartServer();
  EXPECT_EQ(_server->HasStarted(), true);
  EXPECT_EQ(_server->GetPort(), 0); // Still zero because we are using an in-process connection.
  StopServer();
  EXPECT_EQ(_server->HasStarted(), false);
  EXPECT_FALSE(_server->HasConnection());
  EXPECT_EQ(_server->GetPort(), 0);
}

TEST_F(MochiDebuggerTest, ClientConnectDisconnect) {
  StartServer();
  constexpr int kNumReconnects = 5;
  for (int i = 0; i < kNumReconnects; ++i) {
    EXPECT_EQ(_client->GetStatus(), net::SocketStatus::None);
    EXPECT_FALSE(_server->HasConnection());
    ConnectClient();
    EXPECT_EQ(_client->GetStatus(), net::SocketStatus::Connected);
    EXPECT_TRUE(_server->HasConnection());
    DisconnectClient();
    EXPECT_EQ(_client->GetStatus(), net::SocketStatus::None);
    EXPECT_FALSE(_server->HasConnection());
  }
}

TEST_F(MochiDebuggerTest, ServerStopWithClientConnected) {
  StartServer();
  ConnectClient();
  EXPECT_EQ(_client->GetStatus(), net::SocketStatus::Connected);
  EXPECT_TRUE(_server->HasConnection());
  StopServer();
  EXPECT_EQ(_client->GetStatus(), net::SocketStatus::Lost);
  DisconnectClient();
  EXPECT_EQ(_client->GetStatus(), net::SocketStatus::None);
  EXPECT_FALSE(_server->HasConnection());

  // Restart server and reconnect client
  StartServer();
  ConnectClient();
  EXPECT_EQ(_client->GetStatus(), net::SocketStatus::Connected);
  EXPECT_TRUE(_server->HasConnection());
}

TEST_F(MochiDebuggerTest, ShutDownWaitsForCallbacks) {
  std::atomic<int> callbackCount = 0;
  std::promise<void> entered;
  auto hasEntered = entered.get_future();
  std::promise<void> release;
  auto mayReturn = release.get_future().share();
  std::future<void> delivery;
  std::future<void> shutdown;
  bool released = false;
  MOCHI_DEFER({
    if (!released) {
      release.set_value();
    }
    _client->ShutDown();
  });
  _client->Register<protocol::CommandReply>([&](protocol::CommandReply&&) {
    if (callbackCount.fetch_add(1) == 0) {
      entered.set_value();
    }
    mayReturn.wait();
  });
  StartServer();
  ConnectClient();
  auto& server = _server->GetMessageServer_ForTestingOnly();
  delivery = std::async(std::launch::async, [&] { server.Broadcast(protocol::CommandReply{}); });
  ASSERT_EQ(hasEntered.wait_for(std::chrono::seconds(60)), std::future_status::ready);
  shutdown = std::async(std::launch::async, [&] { _client->ShutDown(); });
  // Without waiting for the blocked callback, ShutDown would return well within this time.
  EXPECT_EQ(shutdown.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  released = true;
  release.set_value();
  delivery.get();
  shutdown.get();

  _client->ShutDown();
  server.Broadcast(protocol::CommandReply{});
  EXPECT_EQ(callbackCount.load(), 1);
  EXPECT_FALSE(_client->Send(protocol::CommandReply{}));
  _client->ConnectInProc(server);
  EXPECT_EQ(_client->GetStatus(), net::SocketStatus::None);
  bool commandRan = false;
  _client->QueueCommand([&] { commandRan = true; });
  _client->WaitForQueuedCommands();
  EXPECT_FALSE(commandRan);
}

// A command that is already running when ShutDown starts may still call Disconnect. That call must
// wait until ShutDown has disconnected the socket, because the socket is not thread safe.
TEST_F(MochiDebuggerTest, ShutDownSerializesARunningDisconnectCommand) {
  std::promise<void> commandStarted;
  auto hasCommandStarted = commandStarted.get_future();
  std::promise<void> teardownStarted;
  auto hasTeardownStarted = teardownStarted.get_future();
  // Relaxed, so that TSAN also reports the race if the calls overlap.
  std::atomic<bool> commandDisconnected = false;
  bool disconnectedDuringTeardown = false;
  net::MessageServer server;
  server.SetVersion(_client->GetVersion());
  // The in-process server reports the disconnect on the thread inside ClientSocket::Disconnect.
  server.SetClientCallback([&](net::ClientId, net::ClientEvent event) {
    if (event == net::ClientEvent::Disconnected) {
      teardownStarted.set_value();
      disconnectedDuringTeardown = test::WaitUntil(
          [&] { return commandDisconnected.load(std::memory_order_relaxed); }, 0.5f, false);
    }
  });
  server.StartInProc(1, "test");
  _client->ConnectInProc(server);
  test::WaitUntil([&] { return server.GetClients().size() == 1; });
  _client->QueueCommand([&] {
    commandStarted.set_value();
    hasTeardownStarted.wait_for(std::chrono::seconds(60));
    _client->Disconnect();
    commandDisconnected.store(true, std::memory_order_relaxed);
  });
  ASSERT_EQ(hasCommandStarted.wait_for(std::chrono::seconds(60)), std::future_status::ready);
  _client->ShutDown();
  EXPECT_FALSE(disconnectedDuringTeardown);
  EXPECT_TRUE(commandDisconnected.load());
}
