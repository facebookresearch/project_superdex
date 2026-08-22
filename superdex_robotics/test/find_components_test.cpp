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

#include "mochi_bots_test_helpers.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/superdex_robotics.h>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

class FindComponentsTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    ASSERT_NE(_mochiContext, nullptr);
    _scene = _mochiContext->CreateScene("FindComponentsTest");
    ASSERT_NE(_scene, nullptr);
    _botsCtx = CreateRoboticsContext();
    ASSERT_NE(_botsCtx, nullptr);
  }

  void TearDown() override {
    if (_botsCtx != nullptr) {
      DestroyRoboticsContext(_botsCtx);
      _botsCtx = nullptr;
    }
    if (_scene != nullptr) {
      _mochiContext->DestroyScene(_scene);
      _scene = nullptr;
    }
    if (_mochiContext != nullptr) {
      mochi::DestroyContext(_mochiContext);
      _mochiContext = nullptr;
    }
  }

  static BotPrefab LoadFR3() {
    return LoadBotPrefabFromFile(GetAssetPath("bots/arms/fr3/fr3.superdex_bot"), ExpectOK{});
  }

  Context* _mochiContext = nullptr;
  Scene* _scene = nullptr;
  RoboticsContext* _botsCtx = nullptr;
};

// FindControllersByType returns every controller of that type on the bot, not just the first.
TEST_F(FindComponentsTest, FindControllersByTypeReturnsAllMatchesOnBot) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  ASSERT_NE(bot->CreateController("BASIC_OSC_PD", "osc_a", ExpectOK{}), nullptr);
  ASSERT_NE(bot->CreateController("BASIC_OSC_PD", "osc_b", ExpectOK{}), nullptr);
  ASSERT_NE(bot->CreateController("BASIC_JSC_PD", "jsc", ExpectOK{}), nullptr);

  EXPECT_EQ(isize(_botsCtx->FindControllersByType("BASIC_OSC_PD", bot)), 2);
  EXPECT_EQ(isize(_botsCtx->FindControllersByType("BASIC_JSC_PD", bot)), 1);
  EXPECT_TRUE(_botsCtx->FindControllersByType("MOCHI_ARTICULATED_POSE", bot).empty());
}

// FindControllersByName returns every controller sharing the (non-unique) name.
TEST_F(FindComponentsTest, FindControllersByNameReturnsAllMatches) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  ASSERT_NE(bot->CreateController("BASIC_OSC_PD", "grip", ExpectOK{}), nullptr);
  ASSERT_NE(bot->CreateController("BASIC_JSC_PD", "grip", ExpectOK{}), nullptr);
  ASSERT_NE(bot->CreateController("MOCHI_ARTICULATED_POSE", "pose", ExpectOK{}), nullptr);

  EXPECT_EQ(isize(_botsCtx->FindControllersByName("grip", bot)), 2);
  EXPECT_EQ(isize(_botsCtx->FindControllersByName("pose", bot)), 1);
  EXPECT_TRUE(_botsCtx->FindControllersByName("missing", bot).empty());
}

// A per-bot scope only returns that bot's components; a null scope spans the whole context.
TEST_F(FindComponentsTest, PerBotScopeIsolatesFromContextWide) {
  BotPrefab const prefab = LoadFR3();
  Bot* const botA = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  Bot* const botB = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(botA, nullptr);
  ASSERT_NE(botB, nullptr);

  ASSERT_NE(botA->CreateController("BASIC_OSC_PD", "osc", ExpectOK{}), nullptr);
  ASSERT_NE(botB->CreateController("BASIC_OSC_PD", "osc", ExpectOK{}), nullptr);

  EXPECT_EQ(isize(_botsCtx->FindControllersByType("BASIC_OSC_PD", botA)), 1);
  EXPECT_EQ(isize(_botsCtx->FindControllersByType("BASIC_OSC_PD", botB)), 1);
  EXPECT_EQ(isize(_botsCtx->FindControllersByType("BASIC_OSC_PD", /*scope*/ nullptr)), 2);
}

// Sensors are findable by name and by type across all of a bot's links (no link needed).
TEST_F(FindComponentsTest, FindSensorsByNameAndTypeOnBot) {
  BotPrefab prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());

  BotSensorPrefab headCam;
  headCam.type = DynamicString("SENSOR_CAMERA");
  headCam.name = DynamicString("head_cam");
  prefab.links[0].sensors.push_back(headCam);
  BotSensorPrefab wristCam;
  wristCam.type = DynamicString("SENSOR_CAMERA");
  wristCam.name = DynamicString("wrist_cam");
  prefab.links[0].sensors.push_back(wristCam);

  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  EXPECT_EQ(isize(_botsCtx->FindSensorsByType("SENSOR_CAMERA", bot)), 2);
  EXPECT_EQ(isize(_botsCtx->FindSensorsByName("head_cam", bot)), 1);
  EXPECT_EQ(isize(_botsCtx->FindSensorsByName("wrist_cam", bot)), 1);
}

// Matches come back in creation order. The slot maps are unordered, so without an explicit order
// the same query could answer differently between runs, which matters as soon as a caller indexes
// the result — names are not unique, so multiple matches are the normal case.
TEST_F(FindComponentsTest, FindReturnsMatchesInCreationOrder) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  ControllerBase* const first = bot->CreateController("BASIC_OSC_PD", "grip", ExpectOK{});
  ControllerBase* const second = bot->CreateController("BASIC_JSC_PD", "grip", ExpectOK{});
  ControllerBase* const third = bot->CreateController("MOCHI_ARTICULATED_POSE", "grip", ExpectOK{});
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(third, nullptr);

  DynamicArray<ControllerHandle> const found = _botsCtx->FindControllersByName("grip", bot);
  ASSERT_EQ(isize(found), 3);
  EXPECT_EQ(_botsCtx->GetController(found[0]), first);
  EXPECT_EQ(_botsCtx->GetController(found[1]), second);
  EXPECT_EQ(_botsCtx->GetController(found[2]), third);
}

// A scene-level sensor — actor-less, so belonging to no bot — is found by a context-wide search
// but excluded from a per-bot search. Attaching it to the bot's actor instead would make it the
// bot's, since the owning bot is inferred from the actor.
TEST_F(FindComponentsTest, SceneLevelSensorFoundContextWideButNotPerBot) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  _botsCtx->CreateSensor(
      "SENSOR_CAMERA", /*linkActor*/ nullptr, /*name*/ "", /*paramArgs*/ "", ExpectOK{});

  EXPECT_EQ(isize(_botsCtx->FindSensorsByType("SENSOR_CAMERA", /*scope*/ nullptr)), 1);
  EXPECT_TRUE(_botsCtx->FindSensorsByType("SENSOR_CAMERA", bot).empty());
}

// The per-bot Bot::FindX forwarders scope to that bot and return the same results as the
// bot-scoped RoboticsContext finders.
TEST_F(FindComponentsTest, BotForwardersScopeToThatBot) {
  BotPrefab prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());
  BotSensorPrefab cam;
  cam.type = DynamicString("SENSOR_CAMERA");
  cam.name = DynamicString("eye");
  prefab.links[0].sensors.push_back(cam);

  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  ASSERT_NE(bot->CreateController("BASIC_OSC_PD", "osc_a", ExpectOK{}), nullptr);
  ASSERT_NE(bot->CreateController("BASIC_OSC_PD", "osc_b", ExpectOK{}), nullptr);

  EXPECT_EQ(isize(bot->FindControllersByType("BASIC_OSC_PD")), 2);
  EXPECT_EQ(isize(bot->FindSensorsByType("SENSOR_CAMERA")), 1);
  EXPECT_EQ(isize(bot->FindSensorsByName("eye")), 1);
  EXPECT_TRUE(bot->FindActuatorsByType("ANY").empty());
}

// FindBotsByName returns every bot sharing the (non-unique) name; it is context-wide (bots have no
// per-bot scope).
TEST_F(FindComponentsTest, FindBotsByNameReturnsAllMatches) {
  BotPrefab prefab = LoadFR3();

  prefab.name = DynamicString("arm_left");
  Bot* const leftA = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  Bot* const leftB = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(leftA, nullptr);
  ASSERT_NE(leftB, nullptr);

  prefab.name = DynamicString("arm_right");
  Bot* const right = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(right, nullptr);

  // Two bots share the name "arm_left" — both handles are returned.
  EXPECT_EQ(isize(_botsCtx->FindBotsByName("arm_left")), 2);

  DynamicArray<BotHandle> const rights = _botsCtx->FindBotsByName("arm_right");
  ASSERT_EQ(isize(rights), 1);
  EXPECT_EQ(_botsCtx->GetBot(rights[0]), right);

  EXPECT_TRUE(_botsCtx->FindBotsByName("missing").empty());
}

} // namespace
