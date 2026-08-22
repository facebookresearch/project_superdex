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
#include <string>

using namespace mochi;
using namespace mochi::dbg;

namespace {
class SceneListTest : public MochiDebuggerTest {
 public:
  // Return the client's cached scene names, as a sorted, comma-separated string.
  std::string GetClientSceneNames() const {
    DynamicArray<SceneInfo> scenes;
    _client->GetSceneList(scenes);
    DynamicArray<std::string> names;
    for (auto& s : scenes) {
      names.emplace_back(std::move(s.name));
    }
    std::ranges::sort(names);
    return Join(names, ", ");
  }

  static void WaitForSelectedScene(DebugClient const& client, SceneHandle handle) {
    test::WaitUntil([&] { return client.GetSelectedScene() == handle; });
  }
};
} // namespace

TEST_F(SceneListTest, ScenesPresentBeforeConnect) {
  // Create multiple scenes before connect
  [[maybe_unused]] Scene* alpha = _context->CreateScene("Alpha");
  [[maybe_unused]] Scene* beta = _context->CreateScene("Beta");
  [[maybe_unused]] Scene* hidden = _context->CreateScene("_Hidden");

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return !GetClientSceneNames().empty(); });
  EXPECT_STREQ("Alpha, Beta, _Hidden", GetClientSceneNames().c_str());

  // Client auto-selects the most recently created scene that is not named with a
  // leading underscore.
  EXPECT_EQ(beta->GetHandle(), _client->GetSelectedScene());
}

TEST_F(SceneListTest, EmptySceneNameBeforeConnect) {
  Scene* scene = _context->CreateScene("");
  [[maybe_unused]] Scene* temp = _context->CreateScene("_Temp");

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return !GetClientSceneNames().empty(); });
  EXPECT_STREQ(", _Temp", GetClientSceneNames().c_str());
  EXPECT_EQ(scene->GetHandle(), _client->GetSelectedScene());
}

TEST_F(SceneListTest, OnlyHiddenScenesBeforeConnect) {
  [[maybe_unused]] Scene* stage = _context->CreateScene("_Stage");
  [[maybe_unused]] Scene* temp = _context->CreateScene("_Temp");

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "_Stage, _Temp"; });
  EXPECT_FALSE(_client->GetSelectedScene().IsValid()); // No selection
}

TEST_F(SceneListTest, PrivateScenesAfterConnect) {
  StartServer();
  ConnectClient();

  // Add a scene named with "_" after connect
  [[maybe_unused]] Scene* stage = _context->CreateScene("_Stage");
  test::WaitUntil([&] { return GetClientSceneNames() == "_Stage"; });

  // Client did not select it.
  EXPECT_FALSE(_client->GetSelectedScene().IsValid());
}

TEST_F(SceneListTest, AutoSelectInitialScene) {
  // By convention, scene names that start with an underscore should not be auto-selected.
  [[maybe_unused]] Scene* stage = _context->CreateScene("_Stage");
  Scene* scene = _context->CreateScene("Scene");

  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "Scene, _Stage"; });

  // The DebugClient made a selection as soon as it learned about the scenes (no wait required).
  EXPECT_EQ(scene->GetHandle(), _client->GetSelectedScene());
}

TEST_F(SceneListTest, AddingNormalSceneAfterOnlyPrivateScenesSelectsIt) {
  [[maybe_unused]] Scene* stage = _context->CreateScene("_Stage");
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "_Stage"; });
  EXPECT_FALSE(_client->GetSelectedScene().IsValid());

  Scene* scene = _context->CreateScene("Scene");
  test::WaitUntil([&] { return GetClientSceneNames() == "Scene, _Stage"; });
  WaitForSelectedScene(*_client, scene->GetHandle());
}

TEST_F(SceneListTest, ExistingNormalSelectionIsNotChangedByLaterSceneAdditions) {
  Scene* alpha = _context->CreateScene("Alpha");
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "Alpha"; });
  WaitForSelectedScene(*_client, alpha->GetHandle());

  [[maybe_unused]] Scene* beta = _context->CreateScene("Beta");
  [[maybe_unused]] Scene* stage = _context->CreateScene("_Stage");
  test::WaitUntil([&] { return GetClientSceneNames() == "Alpha, Beta, _Stage"; });
  EXPECT_EQ(alpha->GetHandle(), _client->GetSelectedScene());
}

TEST_F(SceneListTest, ManuallySelectedPrivateSceneStaysSelected) {
  Scene* alpha = _context->CreateScene("Alpha");
  Scene* stage = _context->CreateScene("_Stage");
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "Alpha, _Stage"; });
  WaitForSelectedScene(*_client, alpha->GetHandle());

  ClientSelectScene(stage->GetHandle());
  [[maybe_unused]] Scene* beta = _context->CreateScene("Beta");
  test::WaitUntil([&] { return GetClientSceneNames() == "Alpha, Beta, _Stage"; });
  EXPECT_EQ(stage->GetHandle(), _client->GetSelectedScene());
}

TEST_F(SceneListTest, DeliberateNoneSelectionSurvivesLaterSceneAdditions) {
  Scene* alpha = _context->CreateScene("Alpha");
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "Alpha"; });
  WaitForSelectedScene(*_client, alpha->GetHandle());

  // Deliberately select nothing. Adding a scene must not override that choice.
  ClientSelectScene(SceneHandle{});

  [[maybe_unused]] Scene* beta = _context->CreateScene("Beta");
  test::WaitUntil([&] { return GetClientSceneNames() == "Alpha, Beta"; });
  EXPECT_FALSE(_client->GetSelectedScene().IsValid());
}

TEST_F(SceneListTest, RemovingSelectedSceneFallsBackToHighestNormalScene) {
  Scene* alpha = _context->CreateScene("Alpha");
  Scene* beta = _context->CreateScene("Beta");
  [[maybe_unused]] Scene* stage = _context->CreateScene("_Stage");
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "Alpha, Beta, _Stage"; });
  WaitForSelectedScene(*_client, beta->GetHandle());

  _context->DestroyScene(beta);
  test::WaitUntil([&] { return GetClientSceneNames() == "Alpha, _Stage"; });
  WaitForSelectedScene(*_client, alpha->GetHandle());
}

TEST_F(SceneListTest, RemovingSelectedSceneClearsWhenOnlyPrivateScenesRemain) {
  Scene* alpha = _context->CreateScene("Alpha");
  [[maybe_unused]] Scene* stage = _context->CreateScene("_Stage");
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "Alpha, _Stage"; });
  WaitForSelectedScene(*_client, alpha->GetHandle());

  _context->DestroyScene(alpha);
  test::WaitUntil([&] { return GetClientSceneNames() == "_Stage"; });
  WaitForSelectedScene(*_client, SceneHandle{});
}

TEST_F(SceneListTest, SceneAddedAfterConnectAppears) {
  // Start with one scene so we can observe the authoritative snapshot before testing the delta.
  [[maybe_unused]] Scene* base = _context->CreateScene("Base");
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "Base"; });

  [[maybe_unused]] Scene* gamma = _context->CreateScene("Gamma");
  test::WaitUntil([&] { return GetClientSceneNames() == "Base, Gamma"; });
}

TEST_F(SceneListTest, SceneRemovedDropsFromList) {
  int constexpr kNumScenes = 6;
  DynamicArray<Scene*> scenes;
  scenes.reserve(kNumScenes);
  for (int i = 0; i < kNumScenes; ++i) {
    scenes.push_back(_context->CreateScene(Format("s%d", i)));
  }
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "s0, s1, s2, s3, s4, s5"; });

  _context->DestroyScene(scenes[1]);
  test::WaitUntil([&] { return GetClientSceneNames() == "s0, s2, s3, s4, s5"; });

  _context->DestroyScene(scenes[3]);
  _context->DestroyScene(scenes[5]);
  test::WaitUntil([&] { return GetClientSceneNames() == "s0, s2, s4"; });
}

TEST_F(SceneListTest, DisconnectClearsSceneList) {
  [[maybe_unused]] Scene* scene = _context->CreateScene("MyScene");
  StartServer();
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "MyScene"; });

  // Disconnect clears the cached list synchronously.
  DisconnectClient();
  EXPECT_TRUE(GetClientSceneNames().empty());

  // Reconnect
  ConnectClient();
  test::WaitUntil([&] { return GetClientSceneNames() == "MyScene"; });

  // Stopping the server also disconnects the client and clears the list.
  StopServer();
  test::WaitUntil([&] { return GetClientSceneNames().empty(); });
}
