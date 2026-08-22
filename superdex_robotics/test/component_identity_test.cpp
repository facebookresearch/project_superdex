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

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/superdex_robotics.h>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

class ComponentIdentityTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    ASSERT_NE(_mochiContext, nullptr);
    _scene = _mochiContext->CreateScene("ComponentIdentityTest");
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

// A controller created on a bot records that bot as its owner, so it can later be found by
// bot-scoped queries and cleaned up when the bot is destroyed.
TEST_F(ComponentIdentityTest, ControllerCreatedOnBotReportsOwningBot) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  ControllerBase* const controller = bot->CreateController("BASIC_OSC_PD", "arm_osc", ExpectOK{});
  ASSERT_NE(controller, nullptr);

  EXPECT_EQ(controller->GetOwningBot(), bot);
}

// Removing the one-controller-per-type limit: a bot may hold multiple controllers of the same
// registered type, each a distinct instance owned by that bot.
TEST_F(ComponentIdentityTest, BotAllowsMultipleControllersOfSameType) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  ControllerBase* const first = bot->CreateController("BASIC_OSC_PD", "left_osc", ExpectOK{});
  ASSERT_NE(first, nullptr);
  ControllerBase* const second = bot->CreateController("BASIC_OSC_PD", "right_osc", ExpectOK{});
  ASSERT_NE(second, nullptr);

  EXPECT_NE(first, second);
  EXPECT_EQ(first->GetOwningBot(), bot);
  EXPECT_EQ(second->GetOwningBot(), bot);
}

// Creating a controller straight off a bot's articulation actor, bypassing the bot API, still
// attributes it to that bot: the owner is inferred from the actor rather than the entry point used.
TEST_F(ComponentIdentityTest, ControllerCreatedOnBotArticulationActorInfersOwningBot) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  Actor* const actor = bot->GetArticulatedActor();
  ASSERT_NE(actor, nullptr);

  ControllerHandle const handle = _botsCtx->CreateController(
      "BASIC_OSC_PD", /*prefab*/ nullptr, actor, /*name*/ "", ExpectOK{});
  ASSERT_TRUE(handle.IsValid());
  ControllerBase* const controller = _botsCtx->GetController(handle);
  ASSERT_NE(controller, nullptr);

  EXPECT_EQ(controller->GetOwningBot(), bot);
  EXPECT_EQ(_botsCtx->GetBotContainingActor(actor), bot);
}

// A sensor created straight off a bot's link actor is listed by the bot as well as owned by it:
// GetSensorHandles and the bot-scoped finders describe the same set, however the sensor was made.
TEST_F(ComponentIdentityTest, SensorCreatedOnBotLinkActorIsListedByTheBot) {
  BotPrefab const prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());
  std::string_view const linkName(prefab.links[0].name.c_str(), prefab.links[0].name.size());

  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  Actor* const articulation = bot->GetArticulatedActor();
  ASSERT_NE(articulation, nullptr);
  Span<ActorHandle const> const links = articulation->GetNestedLinkActors(ExpectOK{});
  ASSERT_FALSE(links.empty());
  Actor* const linkActor = _scene->GetActor(links[0]);
  ASSERT_NE(linkActor, nullptr);

  SensorHandle const handle =
      _botsCtx->CreateSensor("SENSOR_CAMERA", linkActor, "wrist_cam", /*paramArgs*/ "", ExpectOK{});

  ASSERT_EQ(isize(bot->GetSensorHandles()), 1);
  EXPECT_EQ(bot->GetSensorHandles()[0], handle);
  EXPECT_EQ(isize(bot->FindSensorsByName("wrist_cam")), 1);
  EXPECT_NE(bot->GetSensor(handle, ExpectOK{}), nullptr);
  EXPECT_EQ(bot->GetSensorLinkName(handle, ExpectOK{}), linkName);
}

// A sensor on the bot's articulation actor is owned and listed, but sits on no prefab link, so
// asking for its link name is an error rather than a wrong answer.
TEST_F(ComponentIdentityTest, SensorOnArticulationActorHasNoLinkName) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  SensorHandle const handle = _botsCtx->CreateSensor(
      "SENSOR_CAMERA", bot->GetArticulatedActor(), "body_cam", /*paramArgs*/ "", ExpectOK{});

  ASSERT_EQ(isize(bot->GetSensorHandles()), 1);
  EXPECT_EQ(bot->GetSensorHandles()[0], handle);

  Error error;
  EXPECT_EQ(bot->GetSensorLinkName(handle, error), nullptr);
  EXPECT_FALSE(error.IsOK());
}

// Controllers answer the same way sensors and actuators do: the bot lists every controller it
// owns, whether it created them itself or the context attributed them from the actor.
TEST_F(ComponentIdentityTest, BotListsEveryControllerItOwns) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  EXPECT_TRUE(bot->GetControllerHandles().empty());

  ControllerBase* const viaBot = bot->CreateController("BASIC_OSC_PD", "arm_osc", ExpectOK{});
  ASSERT_NE(viaBot, nullptr);
  ControllerHandle const viaContext = _botsCtx->CreateController(
      "BASIC_JSC_PD", /*prefab*/ nullptr, bot->GetArticulatedActor(), "arm_jsc", ExpectOK{});

  DynamicArray<ControllerHandle> const handles = bot->GetControllerHandles();
  ASSERT_EQ(isize(handles), 2);
  EXPECT_EQ(_botsCtx->GetController(handles[0]), viaBot);
  EXPECT_EQ(handles[1], viaContext);
}

// Bot::GetController resolves only handles the bot owns, so a handle from another bot is an error
// rather than being silently resolved the way the context-level getter would.
TEST_F(ComponentIdentityTest, BotGetControllerRejectsAnotherBotsHandle) {
  BotPrefab const prefab = LoadFR3();
  Bot* const botA = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  Bot* const botB = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(botA, nullptr);
  ASSERT_NE(botB, nullptr);

  ControllerBase* const owned = botA->CreateController("BASIC_OSC_PD", "osc", ExpectOK{});
  ASSERT_NE(owned, nullptr);
  ASSERT_EQ(isize(botA->GetControllerHandles()), 1);
  ControllerHandle const handle = botA->GetControllerHandles()[0];

  EXPECT_EQ(botA->GetController(handle, ExpectOK{}), owned);

  Error error;
  EXPECT_EQ(botB->GetController(handle, error), nullptr);
  EXPECT_FALSE(error.IsOK());
  // The context-level getter has no owner to check against, so it still resolves it.
  EXPECT_EQ(_botsCtx->GetController(handle), owned);
}

// A controller on a standalone articulation belongs to no bot, so no bot lists it.
TEST_F(ComponentIdentityTest, BotDoesNotListControllersItDoesNotOwn) {
  BotPrefab const prefab = LoadFR3();
  Bot* const botA = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  Bot* const botB = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(botA, nullptr);
  ASSERT_NE(botB, nullptr);

  ASSERT_NE(botA->CreateController("BASIC_OSC_PD", "osc", ExpectOK{}), nullptr);

  EXPECT_EQ(isize(botA->GetControllerHandles()), 1);
  EXPECT_TRUE(botB->GetControllerHandles().empty());
}

// Owner inference reaches the bot's link actors too, not just its articulation actor.
TEST_F(ComponentIdentityTest, SensorCreatedOnBotLinkActorInfersOwningBot) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  Actor* const articulation = bot->GetArticulatedActor();
  ASSERT_NE(articulation, nullptr);

  Span<ActorHandle const> const links = articulation->GetNestedLinkActors(ExpectOK{});
  ASSERT_FALSE(links.empty());
  Actor* const linkActor = _scene->GetActor(links[0]);
  ASSERT_NE(linkActor, nullptr);

  SensorHandle const handle =
      _botsCtx->CreateSensor("SENSOR_CAMERA", linkActor, /*name*/ "", /*paramArgs*/ "", ExpectOK{});
  SensorBase* const sensor = _botsCtx->GetSensor(handle);
  ASSERT_NE(sensor, nullptr);

  EXPECT_EQ(sensor->GetOwningBot(), bot);
  EXPECT_EQ(_botsCtx->GetBotContainingActor(linkActor), bot);
}

// Destroying the bot also destroys a component created against its actor outside the bot API,
// because the cascade keys off the inferred owner rather than a bot-side list.
TEST_F(ComponentIdentityTest, DestroyBotDestroysControllerCreatedOnItsActor) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  BotHandle const botHandle = bot->GetHandle();

  ControllerHandle const handle = _botsCtx->CreateController(
      "BASIC_OSC_PD", /*prefab*/ nullptr, bot->GetArticulatedActor(), /*name*/ "", ExpectOK{});
  ASSERT_TRUE(_botsCtx->IsValidController(handle));

  _botsCtx->DestroyBot(_scene, botHandle);

  EXPECT_FALSE(_botsCtx->IsValidController(handle));
}

// A controller records the instance name it was created with.
TEST_F(ComponentIdentityTest, ControllerRecordsNameFromCreation) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  ControllerBase* const controller =
      bot->CreateController("BASIC_OSC_PD", "left_arm_osc", ExpectOK{});
  ASSERT_NE(controller, nullptr);

  EXPECT_EQ(controller->GetName(), "left_arm_osc");
}

// A bot-owned sensor records both its declared instance name and its owning bot.
TEST_F(ComponentIdentityTest, BotSensorRecordsNameAndOwningBot) {
  BotPrefab prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());

  BotSensorPrefab cameraSensor;
  cameraSensor.type = DynamicString("SENSOR_CAMERA");
  cameraSensor.name = DynamicString("head_cam");
  prefab.links[0].sensors.push_back(cameraSensor);

  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  ASSERT_FALSE(bot->GetSensorHandles().empty());

  SensorBase* const sensor = _botsCtx->GetSensor(bot->GetSensorHandles()[0]);
  ASSERT_NE(sensor, nullptr);
  EXPECT_EQ(sensor->GetName(), "head_cam");
  EXPECT_EQ(sensor->GetOwningBot(), bot);
}

// A prefab naming a sensor type this build does not have still loads: the unknown sensor is
// skipped and everything else on the bot is built. This is what lets the open-source build load
// bot assets whose sensors are Meta-internal.
TEST_F(ComponentIdentityTest, BotSkipsPrefabSensorsOfUnknownType) {
  BotPrefab prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());
  ASSERT_FALSE(_botsCtx->IsSensorTypeRegistered("SENSOR_NOT_IN_THIS_BUILD"));

  BotSensorPrefab unknownSensor;
  unknownSensor.type = DynamicString("SENSOR_NOT_IN_THIS_BUILD");
  unknownSensor.name = DynamicString("unknown_sensor");
  prefab.links[0].sensors.push_back(unknownSensor);

  BotSensorPrefab cameraSensor;
  cameraSensor.type = DynamicString("SENSOR_CAMERA");
  cameraSensor.name = DynamicString("head_cam");
  prefab.links[0].sensors.push_back(cameraSensor);

  // Skipping is deliberately noisy: the warning is the only signal the bot is missing a sensor.
  Bot* bot = nullptr;
  {
    auto const suppressWarning = SuppressLogWarning();
    bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  }
  ASSERT_NE(bot, nullptr);

  // Only the known sensor was built; the unknown one left no trace on the bot.
  DynamicArray<SensorHandle> const handles = bot->GetSensorHandles();
  ASSERT_EQ(handles.size(), 1);
  SensorBase* const sensor = _botsCtx->GetSensor(handles[0]);
  ASSERT_NE(sensor, nullptr);
  EXPECT_EQ(sensor->GetName(), "head_cam");
  EXPECT_TRUE(bot->FindSensorsByName("unknown_sensor").empty());
}

// Skipping an unknown type must not soften genuine failures: a registered type whose construction
// fails still fails the whole bot, because that means the asset is wrong rather than merely richer
// than this build.
TEST_F(ComponentIdentityTest, BotStillFailsWhenAKnownSensorTypeCannotBeBuilt) {
  BotPrefab prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());

  _botsCtx->RegisterSensorType(
      "SENSOR_ALWAYS_FAILS", [](Actor*, std::string_view, Error& error) -> SensorBase* {
        MOCHI_ERROR_SET(error, "sensor construction failed on purpose");
        return nullptr;
      });

  BotSensorPrefab failing;
  failing.type = DynamicString("SENSOR_ALWAYS_FAILS");
  failing.name = DynamicString("broken");
  prefab.links[0].sensors.push_back(failing);

  Error error;
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, error);
  EXPECT_EQ(bot, nullptr);
  EXPECT_FALSE(error.IsOK());
}

// A sensor created through the bot API at runtime is indistinguishable from one declared in the
// prefab: attached to the named link, owned by the bot, and listed among its sensors.
TEST_F(ComponentIdentityTest, BotCreateSensorAttachesToLinkAndBot) {
  BotPrefab const prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());
  std::string_view const linkName(prefab.links[0].name.c_str(), prefab.links[0].name.size());

  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  SensorBase* const sensor =
      bot->CreateSensor("SENSOR_CAMERA", linkName, "wrist_cam", /*paramArgs*/ "", ExpectOK{});
  ASSERT_NE(sensor, nullptr);

  EXPECT_EQ(sensor->GetName(), "wrist_cam");
  EXPECT_EQ(sensor->GetOwningBot(), bot);
  ASSERT_EQ(isize(bot->GetSensorHandles()), 1);
  EXPECT_EQ(bot->GetSensorLinkName(bot->GetSensorHandles()[0], ExpectOK{}), linkName);
  EXPECT_EQ(isize(bot->FindSensorsByName("wrist_cam")), 1);
}

// A sensor created through the bot API is torn down with the bot, like a prefab-declared one.
TEST_F(ComponentIdentityTest, DestroyBotDestroysSensorCreatedViaBotApi) {
  BotPrefab const prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());
  std::string_view const linkName(prefab.links[0].name.c_str(), prefab.links[0].name.size());

  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  BotHandle const botHandle = bot->GetHandle();

  ASSERT_NE(
      bot->CreateSensor("SENSOR_CAMERA", linkName, "wrist_cam", /*paramArgs*/ "", ExpectOK{}),
      nullptr);
  SensorHandle const sensorHandle = bot->GetSensorHandles()[0];
  ASSERT_TRUE(_botsCtx->IsValidSensor(sensorHandle));

  _botsCtx->DestroyBot(_scene, botHandle);

  EXPECT_FALSE(_botsCtx->IsValidSensor(sensorHandle));
}

// Naming a link the bot does not have is an error rather than a silent attach to the wrong link.
TEST_F(ComponentIdentityTest, BotCreateSensorRejectsUnknownLink) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  Error error;
  EXPECT_EQ(
      bot->CreateSensor("SENSOR_CAMERA", "no_such_link", "cam", /*paramArgs*/ "", error), nullptr);
  EXPECT_FALSE(error.IsOK());
  EXPECT_TRUE(bot->GetSensorHandles().empty());
}

// The actuator path resolves its link the same way. No concrete actuator type is registered yet, so
// only link resolution — which runs before the type lookup — is exercisable here.
TEST_F(ComponentIdentityTest, BotCreateActuatorRejectsUnknownLink) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  Error error;
  EXPECT_EQ(
      bot->CreateActuator("ANY_ACTUATOR", "no_such_link", "act", /*paramArgs*/ "", error), nullptr);
  EXPECT_FALSE(error.IsOK());
  EXPECT_TRUE(bot->GetActuatorHandles().empty());
}

} // namespace
