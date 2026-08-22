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
#include <superdex_robotics/controllers/controller_basic_osc_pd.h>
#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/superdex_robotics.h>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

class BotLifecycleTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    ASSERT_NE(_mochiContext, nullptr);
    _scene = _mochiContext->CreateScene("BotLifecycleTest");
    ASSERT_NE(_scene, nullptr);
    _botsCtx = CreateRoboticsContext();
    ASSERT_NE(_botsCtx, nullptr);
  }

  void TearDown() override {
    // Guarded so a test that intentionally tears these down in a different order (e.g. destroying
    // the scene before the RoboticsContext) can null them out and not be double-freed here.
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

// End-to-end: load FR3, create a Bot, attach BASIC_OSC_PD, destroy the bot — and verify the
// Bot interface and the RoboticsContext interface agree all the way through.
TEST_F(BotLifecycleTest, CreateBotAndControllerThenDestroyBot) {
  BotPrefab const prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());

  Bot* bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  // Bot interface and context interface return the same Bot for the bot's handle.
  BotHandle const botHandle = bot->GetHandle();
  EXPECT_TRUE(botHandle.IsValid());
  EXPECT_TRUE(_botsCtx->IsValidBot(botHandle));
  EXPECT_EQ(_botsCtx->GetBot(botHandle), bot);

  // Underlying articulated actor is non-null and lives in our scene.
  Actor* const actor = bot->GetArticulatedActor();
  ASSERT_NE(actor, nullptr);
  EXPECT_EQ(_scene->GetActor(actor->GetHandle()), actor);

  // Bot's accessors agree with what we passed in.
  EXPECT_EQ(bot->GetScene(), _scene);
  EXPECT_EQ(bot->GetBotContext(), _botsCtx);
  EXPECT_EQ(bot->GetMochiContext(), _mochiContext);

  // Create an BASIC_OSC_PD controller via the Bot interface.
  ControllerBase* const controller = bot->CreateController("BASIC_OSC_PD", "arm_osc", ExpectOK{});
  ASSERT_NE(controller, nullptr);

  // The bot- and context-scoped finders each return the single controller of this type, whose
  // handle resolves back to that same controller.
  DynamicArray<ControllerHandle> const found = bot->FindControllersByType("BASIC_OSC_PD");
  ASSERT_EQ(isize(found), 1);
  ControllerHandle const controllerHandle = found[0];
  ASSERT_TRUE(controllerHandle.IsValid());
  EXPECT_EQ(_botsCtx->GetController(controllerHandle), controller);
  EXPECT_EQ(isize(_botsCtx->FindControllersByType("BASIC_OSC_PD", bot)), 1);

  // The controller is downcastable to ControllerBasicOscPd and reports the bot's actor.
  auto* osc = dynamic_cast<ControllerBasicOscPd*>(controller);
  ASSERT_NE(osc, nullptr);
  EXPECT_EQ(osc->GetActor(), actor);
  EXPECT_TRUE(controller->IsValid());
  EXPECT_TRUE(_botsCtx->IsValidController(controllerHandle));

  // Destroy the bot — this should also destroy its BASIC_OSC_PD controller and tear down the
  // articulated actor in the scene.
  ActorHandle const actorHandle = actor->GetHandle();
  DestroyBot(_scene, bot);

  // Bot is gone from the context.
  EXPECT_FALSE(_botsCtx->IsValidBot(botHandle));
  EXPECT_EQ(_botsCtx->GetBot(botHandle), nullptr);

  // Controller is gone from the context too.
  EXPECT_FALSE(_botsCtx->IsValidController(controllerHandle));
  EXPECT_EQ(_botsCtx->GetController(controllerHandle), nullptr);

  // The articulated actor is gone from the scene.
  EXPECT_EQ(_scene->GetActor(actorHandle), nullptr);
}

// Destroying the controller via the context should leave the bot's finders returning no match.
TEST_F(BotLifecycleTest, DestroyControllerViaContextClearsBotCache) {
  BotPrefab const prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());

  Bot* bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  ControllerBase* const controller = bot->CreateController("BASIC_OSC_PD", "arm_osc", ExpectOK{});
  ASSERT_NE(controller, nullptr);

  // The bot finds the single controller of this type; its handle resolves to that controller.
  DynamicArray<ControllerHandle> const found = bot->FindControllersByType("BASIC_OSC_PD");
  ASSERT_EQ(isize(found), 1);
  ControllerHandle const controllerHandle = found[0];
  EXPECT_EQ(_botsCtx->GetController(controllerHandle), controller);

  // Destroy via the context.
  _botsCtx->DestroyController(controllerHandle);
  EXPECT_FALSE(_botsCtx->IsValidController(controllerHandle));

  // The bot no longer finds a controller of this type.
  EXPECT_TRUE(bot->FindControllersByType("BASIC_OSC_PD").empty());

  // After removal we can create a new BASIC_OSC_PD controller on the same bot without error.
  ControllerBase* const newController =
      bot->CreateController("BASIC_OSC_PD", "arm_osc", ExpectOK{});
  ASSERT_NE(newController, nullptr);

  DestroyBot(_scene, bot);
}

// Regression: destroying the mochi Scene BEFORE the RoboticsContext must not leave the bot context
// dereferencing dangling Scene*/Actor* pointers. Before the handle-based lifetime fix, the bot
// cached a raw Scene* and its controllers/sensors cached a raw Actor*, so tearing the scene down
// first turned RoboticsContext teardown (and Bot::GetArticulatedActor) into a use-after-free. The
// bot here carries an auto-instantiated camera sensor and an initialized MOCHI_ARTICULATED_POSE
// controller so the teardown path exercises the bot's scene pointer, a sensor, and a controller
// whose destructor releases an actor-held resource.
TEST_F(BotLifecycleTest, DestroySceneBeforeRoboticsContextDoesNotDangle) {
  BotPrefab prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());

  // Declare a camera sensor on the root link so CreateBot auto-instantiates it with the bot.
  BotSensorPrefab cameraSensor;
  cameraSensor.type = DynamicString("SENSOR_CAMERA");
  cameraSensor.name = DynamicString("test_camera");
  prefab.links[0].sensors.push_back(cameraSensor);

  Bot* bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  BotHandle const botHandle = bot->GetHandle();

  // The camera sensor was auto-created and belongs to the bot.
  ASSERT_FALSE(bot->GetSensorHandles().empty());
  SensorHandle const sensorHandle = bot->GetSensorHandles()[0];
  EXPECT_NE(_botsCtx->GetSensor(sensorHandle), nullptr);

  // Add an articulated-pose controller and initialize it so it actually installs a pose controller
  // on the actor — its destructor must touch the actor, which is what the fix makes safe.
  ControllerBase* const controller =
      bot->CreateController("MOCHI_ARTICULATED_POSE", "arm_pose", ExpectOK{});
  ASSERT_NE(controller, nullptr);
  auto* const pose = dynamic_cast<ControllerMochiArticulatedPose*>(controller);
  ASSERT_NE(pose, nullptr);
  pose->Initialize(/*removeExisting=*/true, ExpectOK{});

  // While the scene is alive, everything resolves.
  EXPECT_EQ(bot->GetScene(), _scene);
  EXPECT_NE(bot->GetArticulatedActor(), nullptr);

  // Destroy the mochi scene first — the ordering that previously dangled the bot context.
  _mochiContext->DestroyScene(_scene);
  _scene = nullptr;

  // The bot now reports its scene/actor as gone instead of dereferencing freed memory; the context
  // (the outermost lifetime) is still valid.
  EXPECT_EQ(bot->GetScene(), nullptr);
  EXPECT_EQ(bot->GetArticulatedActor(), nullptr);
  EXPECT_EQ(bot->GetMochiContext(), _mochiContext);

  // The bot itself is still owned by (and valid in) the RoboticsContext — only its scene resources
  // died.
  EXPECT_TRUE(_botsCtx->IsValidBot(botHandle));

  // Tearing down the RoboticsContext after the scene is gone must complete without a crash: it
  // skips the now-defunct actor teardown and safely destroys the bot's camera sensor and pose
  // controller.
  DestroyRoboticsContext(_botsCtx);
  _botsCtx = nullptr;
}

} // namespace
