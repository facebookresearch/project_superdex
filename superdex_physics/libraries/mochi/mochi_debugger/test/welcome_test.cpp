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

#include <mochi_core/net/client_socket.h> // net::SocketStatus
#include <mochi_core/test/wait_until.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/string_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string_view>

using namespace mochi;
using namespace mochi::dbg;

namespace {

class WelcomeTest : public MochiDebuggerTest {
 protected:
  int GetNumClientScenes() const {
    DynamicArray<SceneInfo> scenes;
    _client->GetSceneList(scenes);
    return isize(scenes);
  }
};

} // namespace

// The status only reaches Connected once the welcome message has been processed, so the scene list
// is already complete at that moment - no additional waiting required.
TEST_F(WelcomeTest, ConnectedImpliesSceneListIsPopulated) {
  int constexpr kNumScenes = 3;
  for (int i = 0; i < kNumScenes; ++i) {
    [[maybe_unused]] Scene* scene = _context->CreateScene(Format("s%d", i));
  }
  StartServer();

  // Connect without the fixture's helper, so we observe the raw status progression.
  _client->ConnectInProc(_server->GetMessageServer_ForTestingOnly());
  test::WaitUntil([&] { return _client->GetStatus() != net::SocketStatus::None; });
  test::WaitUntil([&] { return _client->GetStatus() == net::SocketStatus::Connected; });
  EXPECT_EQ(kNumScenes, GetNumClientScenes());
}

// Losing the connection must clear the handshake, so a reconnect performs it again.
TEST_F(WelcomeTest, ReconnectRepeatsTheHandshake) {
  [[maybe_unused]] Scene* scene = _context->CreateScene("MyScene");
  StartServer();
  ConnectClient();
  ASSERT_EQ(1, GetNumClientScenes());

  DisconnectClient();
  EXPECT_EQ(net::SocketStatus::None, _client->GetStatus());
  EXPECT_EQ(0, GetNumClientScenes());

  ConnectClient();
  EXPECT_EQ(net::SocketStatus::Connected, _client->GetStatus());
  EXPECT_EQ(1, GetNumClientScenes());
}

TEST_F(WelcomeTest, DefaultCoordinateSpace) {
  StartServer();
  ConnectClient();
  // Default matches Filament and OpenGL (Mochi's historical coordinate space).
  EXPECT_EQ(CoordinateSpace::Filament(), _client->GetCoordinateSpace());
}

TEST_F(WelcomeTest, ServerCoordinateSpaceReachesTheClient) {
  _server->SetCoordinateSpace(CoordinateSpace::Unreal());
  StartServer();
  ConnectClient();
  EXPECT_EQ(CoordinateSpace::Unreal(), _client->GetCoordinateSpace());

  // It is re-sent on reconnect, not just remembered from the first handshake.
  DisconnectClient();
  EXPECT_EQ(CoordinateSpace::Filament(), _client->GetCoordinateSpace());
  ConnectClient();
  EXPECT_EQ(CoordinateSpace::Unreal(), _client->GetCoordinateSpace());
}
