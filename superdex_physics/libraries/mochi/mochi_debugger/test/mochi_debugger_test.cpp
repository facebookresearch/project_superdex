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

using namespace mochi;
using namespace mochi::dbg;

void MochiDebuggerTest::SetUp() {
  // Save the test harness's global log callback (installed by InitUnitTest), because the
  // DebugServer will override it.
  _prevLogCallback = GetLogCallback();
  _context = mochi::CreateContext();

  // Store a pointer to the server. It is owned by the Context.
  _server = assert_cast<DebugServerInternal*>(&_context->GetDebugServer());

  // Capture DebugClient logging.
  _client = std::make_unique<DebugClient>();
  _client->SetPrintFunction(
      [this](LogChannel channel, char const* message, char const* file, int line) {
        _clientLogs.Mutate([&](auto& list) {
          list.emplace_back(protocol::LogMessage{channel, message, file, line});
        });
      });
}

void MochiDebuggerTest::TearDown() {
  DisconnectClient();
  _client->SetPrintFunction({}); // No more callbacks please
  mochi::DestroyContext(_context);

  // Restore the test harness's global log callback.
  SetLogCallback(_prevLogCallback);
}

void MochiDebuggerTest::StartServer() {
  _server->StartInProc();
}

void MochiDebuggerTest::StopServer() {
  _server->Stop();
  test::WaitUntil([this] {
    return _client->GetStatus() == net::SocketStatus::Lost ||
        _client->GetStatus() == net::SocketStatus::None;
  });
}

void MochiDebuggerTest::ConnectClient() {
  _client->ConnectInProc(_server->GetMessageServer_ForTestingOnly());
  test::WaitUntil([this] { return _client->GetStatus() == net::SocketStatus::Connected; });
}

void MochiDebuggerTest::DisconnectClient() {
  _client->Disconnect();
  test::WaitUntil([this] { return _client->GetStatus() == net::SocketStatus::None; });
}

bool MochiDebuggerTest::ClientHasScene(SceneHandle handle) const {
  DynamicArray<SceneInfo> scenes;
  _client->GetSceneList(scenes);
  for (auto const& s : scenes) {
    if (s.handle == handle) {
      return true;
    }
  }
  return false;
}

void MochiDebuggerTest::ClientSelectScene(SceneHandle handle) {
  if (handle.IsValid()) {
    test::WaitUntil([&] { return ClientHasScene(handle); });
  }
  _client->SelectScene(handle);
  EXPECT_EQ(handle, _client->GetSelectedScene());
}
