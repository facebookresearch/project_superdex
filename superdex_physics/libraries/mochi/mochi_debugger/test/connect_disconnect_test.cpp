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

#include <gtest/gtest.h>

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
