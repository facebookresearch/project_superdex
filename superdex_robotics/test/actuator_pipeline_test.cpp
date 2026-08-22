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

// Exercises the actuator resource pipeline at the plumbing level. No concrete ActuatorBase
// subclass exists yet, so create->find round-trips cannot be tested end-to-end; these cover the
// handle validity, registry lookup, unknown-type error, and empty-find behavior that stand on their
// own until the first concrete actuator lands.

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

// An ActuatorHandle round-trips its raw value and reports validity. Distinctness from the other
// component kinds is now a compile-time property (ActuatorHandle is its own C++ type), so it no
// longer needs a runtime tag check.
TEST(ActuatorHandleTest, ActuatorHandleRoundTrips) {
  ActuatorHandle const h(42);
  EXPECT_TRUE(h.IsValid());
  EXPECT_EQ(h.value, RoboticsHandle::ValueType{42});
  EXPECT_FALSE(ActuatorHandle{}.IsValid());
}

class ActuatorPipelineTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    ASSERT_NE(_mochiContext, nullptr);
    _scene = _mochiContext->CreateScene("ActuatorPipelineTest");
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

// No concrete actuator types are registered by default.
TEST_F(ActuatorPipelineTest, NoActuatorTypesRegisteredByDefault) {
  EXPECT_FALSE(_botsCtx->IsActuatorTypeRegistered("ANY_ACTUATOR"));
}

// Creating an unregistered actuator type reports an error and yields an invalid handle.
TEST_F(ActuatorPipelineTest, CreateActuatorUnknownTypeErrors) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  Actor* const actor = bot->GetArticulatedActor();
  ASSERT_NE(actor, nullptr);

  Error error;
  ActuatorHandle const handle =
      _botsCtx->CreateActuator("NO_SUCH_ACTUATOR", actor, /*name*/ "", /*paramArgs*/ "", error);
  EXPECT_FALSE(error.IsOK());
  EXPECT_FALSE(handle.IsValid());
}

// With no actuators created, the finders return empty rather than erroring.
TEST_F(ActuatorPipelineTest, FindActuatorsEmptyWhenNoneExist) {
  EXPECT_TRUE(_botsCtx->FindActuatorsByType("ANY_ACTUATOR", /*scope*/ nullptr).empty());
  EXPECT_TRUE(_botsCtx->FindActuatorsByName("any", /*scope*/ nullptr).empty());
}

// Invalid handles resolve to null / invalid for the actuator accessors. Cross-kind confusion (e.g.
// passing a controller handle) is now rejected at compile time by the typed accessor signature, so
// what remains to check at runtime is that a well-formed but never-issued ActuatorHandle value does
// not resolve.
TEST_F(ActuatorPipelineTest, InvalidActuatorHandleQueries) {
  EXPECT_EQ(_botsCtx->GetActuator(ActuatorHandle{}), nullptr);
  EXPECT_FALSE(_botsCtx->IsValidActuator(ActuatorHandle{}));
  EXPECT_EQ(_botsCtx->GetActuator(ActuatorHandle{1}), nullptr);
  EXPECT_FALSE(_botsCtx->IsValidActuator(ActuatorHandle{1}));
}

// A bot with no declared actuators reports no actuator handles.
TEST_F(ActuatorPipelineTest, BotWithoutActuatorsHasNoActuatorHandles) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  EXPECT_TRUE(bot->GetActuatorHandles().empty());
}

// An actuator of an unregistered type on a link is skipped with a warning and the rest of the bot
// still loads, mirroring how CreateSensors treats unknown sensor types. This is what lets a build
// load bot assets whose actuators are registered only elsewhere (e.g. by an example script).
// Construction failures of a type that IS registered still fail the bot -- see CreateActuators.
TEST_F(ActuatorPipelineTest, BotSkipsPrefabActuatorsOfUnknownType) {
  BotPrefab prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());
  ASSERT_FALSE(_botsCtx->IsActuatorTypeRegistered("NO_SUCH_ACTUATOR"));

  BotActuatorPrefab actuator;
  actuator.type = DynamicString("NO_SUCH_ACTUATOR");
  actuator.name = DynamicString("a1");
  prefab.links[0].actuators.push_back(actuator);

  // Skipping is deliberately noisy: the warning is the only signal the bot is missing an actuator.
  Bot* bot = nullptr;
  {
    auto const suppressWarning = SuppressLogWarning();
    bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  }
  ASSERT_NE(bot, nullptr);

  // The unknown actuator left no trace on the bot.
  EXPECT_TRUE(bot->GetActuatorHandles().empty());
}

// Skipping an unknown type must not soften genuine failures: a registered type whose construction
// fails still fails the whole bot, because that means the asset is wrong rather than merely richer
// than this build.
TEST_F(ActuatorPipelineTest, BotStillFailsWhenAKnownActuatorTypeCannotBeBuilt) {
  BotPrefab prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());

  _botsCtx->RegisterActuatorType(
      "ACTUATOR_ALWAYS_FAILS", [](Actor*, std::string_view, Error& error) -> ActuatorBase* {
        MOCHI_ERROR_SET(error, "actuator construction failed on purpose");
        return nullptr;
      });

  BotActuatorPrefab failing;
  failing.type = DynamicString("ACTUATOR_ALWAYS_FAILS");
  failing.name = DynamicString("broken");
  prefab.links[0].actuators.push_back(failing);

  Error error;
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, error);
  EXPECT_EQ(bot, nullptr);
  EXPECT_FALSE(error.IsOK());
}

} // namespace
