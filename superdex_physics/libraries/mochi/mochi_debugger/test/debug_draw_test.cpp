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

#include <mochi_core/test/log_suppression.h>

#include <mochi_core/test/wait_until.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/dbg/protocol.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <string_view>

using namespace mochi;
using namespace mochi::dbg;

namespace {

class DebugDrawTest : public MochiDebuggerTest {
 protected:
  // A feature that every build registers, used as the subject of the toggle tests.
  static constexpr std::string_view kFeature = "Actor Root Transform";

  // Names of the features the client learned about, in catalog order.
  DynamicArray<std::string> GetClientFeatureNames() const {
    DynamicArray<std::string> names;
    for (auto const& feature : _client->GetDebugDrawFeatures()) {
      names.emplace_back(feature.name);
    }
    return names;
  }

  // Names of the features a real scene's DebugDraw reports, in its own order.
  static DynamicArray<std::string> GetSceneFeatureNames(Scene* scene) {
    auto const& debugDraw = scene->GetDebugDraw();
    DynamicArray<std::string> names;
    for (int i = 0; i < debugDraw.GetNumFeatures(); ++i) {
      names.emplace_back(debugDraw.GetFeatureName(i));
    }
    return names;
  }

  int FindClientFeature(std::string_view name) const {
    auto const features = _client->GetDebugDrawFeatures();
    for (int i = 0; i < isize(features); ++i) {
      if (features[i].name == name) {
        return i;
      }
    }
    return -1;
  }

  // Pump the (paused) scene until the server has applied the expected debug draw state.
  static bool
  WaitForSceneState(Scene* scene, int featureIndex, bool masterEnabled, bool featureEnabled) {
    return test::WaitUntil([&] {
      scene->UpdateDebugger();
      auto const& debugDraw = scene->GetDebugDraw();
      return debugDraw.IsEnabled() == masterEnabled &&
          debugDraw.IsFeatureEnabled(featureIndex) == featureEnabled;
    });
  }

  uint64_t GetSyncCounter() const {
    uint64_t counter = 0;
    _client->GetSceneSyncData([&](auto const& data) { counter = data.counter; });
    return counter;
  }
};

} // namespace

TEST_F(DebugDrawTest, WelcomeCarriesTheSameCatalogAsAScene) {
  Scene* scene = _context->CreateScene("DebugDrawTest");
  StartServer();
  ConnectClient();

  auto const clientNames = GetClientFeatureNames();
  EXPECT_FALSE(clientNames.empty());
  EXPECT_EQ(GetSceneFeatureNames(scene), clientNames);

  // Everything starts disabled, matching the server.
  EXPECT_FALSE(_client->IsDebugDrawEnabled());
  EXPECT_FALSE(scene->GetDebugDraw().IsEnabled());
  auto const features = _client->GetDebugDrawFeatures();
  EXPECT_TRUE(std::ranges::none_of(features, &protocol::DbgDrawFeature::enabled));
}

TEST_F(DebugDrawTest, TogglesReachTheSelectedScene) {
  Scene* scene = _context->CreateScene("DebugDrawTest");
  StartServer();
  ConnectClient();
  ClientSelectScene(scene->GetHandle());

  int const index = FindClientFeature(kFeature);
  ASSERT_LE(0, index);

  _client->EnableDebugDrawFeature(kFeature, true);
  EXPECT_TRUE(_client->IsDebugDrawEnabled());
  EXPECT_TRUE(WaitForSceneState(scene, index, /*masterEnabled*/ true, /*featureEnabled*/ true));

  _client->EnableDebugDraw(false);
  EXPECT_TRUE(WaitForSceneState(scene, index, /*masterEnabled*/ false, /*featureEnabled*/ true));

  _client->EnableDebugDraw(true);
  _client->EnableDebugDrawFeature(kFeature, false);
  EXPECT_TRUE(WaitForSceneState(scene, index, /*masterEnabled*/ true, /*featureEnabled*/ false));
}

TEST_F(DebugDrawTest, InvalidFeatureNameIsIgnored) {
  [[maybe_unused]] Scene* scene = _context->CreateScene("DebugDrawTest");
  StartServer();
  ConnectClient();

  _client->EnableDebugDrawFeature("NoSuchThing", true);

  auto const features = _client->GetDebugDrawFeatures();
  EXPECT_TRUE(std::ranges::none_of(features, &protocol::DbgDrawFeature::enabled));
}

TEST_F(DebugDrawTest, FeatureIndicesOutsideRange) {
  auto* scene = _context->CreateScene("DebugDrawTest");
  auto& draw = scene->GetDebugDraw();
  auto suppressErrors = test::SuppressLogError();
  for (int const index : {-1, draw.GetNumFeatures(), draw.GetNumFeatures() + 1}) {
    SCOPED_TRACE(index);
    draw.EnableFeature(index, true);
    EXPECT_TRUE(draw.GetFeatureName(index).empty());
    EXPECT_TRUE(draw.GetFeatureDescription(index).empty());
    EXPECT_FALSE(draw.IsFeatureEnabled(index));
  }
}

TEST_F(DebugDrawTest, SelectingSceneAppliesTheDisabledState) {
  Scene* scene = _context->CreateScene("DebugDrawTest");
  auto& debugDraw = scene->GetDebugDraw();
  int const index = debugDraw.FindFeature(kFeature);
  ASSERT_LE(0, index);
  debugDraw.EnableFeature(index, true);
  debugDraw.Enable(true);

  StartServer();
  ConnectClient();

  EXPECT_TRUE(WaitForSceneState(scene, index, /*masterEnabled*/ false, /*featureEnabled*/ false));
}

TEST_F(DebugDrawTest, SelectingAnotherSceneTransfersTheEnabledSet) {
  Scene* sceneA = _context->CreateScene("A");
  Scene* sceneB = _context->CreateScene("B");
  StartServer();
  ConnectClient();
  ClientSelectScene(sceneA->GetHandle());

  int const index = FindClientFeature(kFeature);
  ASSERT_LE(0, index);
  _client->EnableDebugDraw(true);
  _client->EnableDebugDrawFeature(kFeature, true);
  EXPECT_TRUE(WaitForSceneState(sceneA, index, /*masterEnabled*/ true, /*featureEnabled*/ true));

  // The newly selected scene takes on the state, and the old one stops doing the work.
  ClientSelectScene(sceneB->GetHandle());
  EXPECT_TRUE(WaitForSceneState(sceneB, index, /*masterEnabled*/ true, /*featureEnabled*/ true));
  EXPECT_TRUE(WaitForSceneState(sceneA, index, /*masterEnabled*/ false, /*featureEnabled*/ false));
}

TEST_F(DebugDrawTest, DisconnectDisablesSceneAndReconnectRestoresClientState) {
  Scene* scene = _context->CreateScene("DebugDrawTest");
  StartServer();
  ConnectClient();
  ClientSelectScene(scene->GetHandle());

  int const index = FindClientFeature(kFeature);
  ASSERT_LE(0, index);
  _client->EnableDebugDrawFeature(kFeature, true);
  ASSERT_TRUE(WaitForSceneState(scene, index, /*masterEnabled*/ true, /*featureEnabled*/ true));

  DisconnectClient();
  EXPECT_TRUE(_client->IsDebugDrawEnabled());
  EXPECT_TRUE(_client->GetDebugDrawFeatures()[index].enabled);
  EXPECT_TRUE(WaitForSceneState(scene, index, /*masterEnabled*/ false, /*featureEnabled*/ false));

  ConnectClient();
  int const reconnectedIndex = FindClientFeature(kFeature);
  ASSERT_LE(0, reconnectedIndex);
  EXPECT_TRUE(
      WaitForSceneState(scene, reconnectedIndex, /*masterEnabled*/ true, /*featureEnabled*/ true));
}

TEST_F(DebugDrawTest, ChangingTheStateProducesFreshSyncDataWhilePaused) {
  Scene* scene = _context->CreateScene("DebugDrawTest");
  StartServer();
  ConnectClient();

  DebugClientSettings settings;
  settings.sync.enabled = true;
  settings.sync.syncDebugDraw = true;
  _client->SetSettings(settings);
  ClientSelectScene(scene->GetHandle());

  // The immediate reply from selecting the scene arrives first.
  uint64_t const baseCounter = GetSyncCounter();
  test::WaitUntil([&] {
    scene->UpdateDebugger();
    return GetSyncCounter() > baseCounter;
  });

  // The scene is never stepped, so pumping alone produces no further syncs.
  uint64_t const idleCounter = GetSyncCounter();
  test::WaitUntil(
      [&] {
        scene->UpdateDebugger();
        return GetSyncCounter() != idleCounter;
      },
      /*timeoutSec*/ 0.01f,
      /*failOnTimeout*/ false);
  ASSERT_EQ(idleCounter, GetSyncCounter());

  // Changing the debug draw state changes what the scene renders, so the server must transmit
  // fresh data even though the simulation has not advanced.
  int const index = FindClientFeature(kFeature);
  ASSERT_LE(0, index);
  _client->EnableDebugDraw(true);
  _client->EnableDebugDrawFeature(kFeature, true);
  test::WaitUntil([&] {
    scene->UpdateDebugger();
    return GetSyncCounter() > idleCounter;
  });

  // A redundant toggle changes nothing, so no request is sent and no new data arrives.
  uint64_t const settledCounter = GetSyncCounter();
  _client->EnableDebugDraw(true);
  _client->EnableDebugDrawFeature(kFeature, true);
  test::WaitUntil(
      [&] {
        scene->UpdateDebugger();
        return GetSyncCounter() != settledCounter;
      },
      /*timeoutSec*/ 0.01f,
      /*failOnTimeout*/ false);
  EXPECT_EQ(settledCounter, GetSyncCounter());
}
