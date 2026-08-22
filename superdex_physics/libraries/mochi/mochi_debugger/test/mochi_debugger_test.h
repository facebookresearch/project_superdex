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

#include <mochi_debugger/lib/debug_client.h>

#include <mochi_core/net/message_server.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>
#include <mochi_physics/dbg/debug_server_internal.h>
#include <mochi_physics/dbg/protocol.h>
#include <mochi_physics/mochi_physics.h>

namespace mochi::dbg {

// Test fixture that owns a mochi::Context and a DebugClient. Connects using in-process
// communication (no TCP sockets required).
class MochiDebuggerTest : public ::testing::Test {
 public:
  void SetUp() override;
  void TearDown() override;

  // Return true if the DebugClient's cached scene list contains the given handle.
  bool ClientHasScene(SceneHandle handle) const;

  // Call DebugClient::SelectScene. Waits for the scene list if called immediately after connect.
  void ClientSelectScene(SceneHandle handle);

 protected:
  void StartServer();
  void StopServer();
  void ConnectClient();
  void DisconnectClient();

  Context* _context = nullptr;
  DebugServerInternal* _server = nullptr;
  std::unique_ptr<DebugClient> _client;
  Guarded<DynamicArray<protocol::LogMessage>> _clientLogs;
  LogFn _prevLogCallback = {};
};

} // namespace mochi::dbg
