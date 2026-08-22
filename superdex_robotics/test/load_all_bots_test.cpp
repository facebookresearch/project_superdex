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

#include <mochi_core/utils/defer.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <filesystem>
#include <string>
#include <variant>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

// Parameterized so each bot is a separate test case — loading shapes from disk is expensive,
// and this allows the test runner to shard bots across processes for parallel execution.
class LoadAllBotsTest : public ::testing::TestWithParam<std::filesystem::path> {};

TEST_P(LoadAllBotsTest, LoadAndAddToScene) {
  Context* context = CreateContext(0);
  ASSERT_NE(context, nullptr);
  Scene* scene = context->CreateScene("LoadAllBotsTest");
  ASSERT_NE(scene, nullptr);
  RoboticsContext* botsContext = CreateRoboticsContext();
  ASSERT_NE(botsContext, nullptr);

  BotPrefab botPrefab = LoadBotPrefabFromFile(GetParam().string(), ExpectOK{});
  if (!botPrefab.links.empty()) {
    // Many bots lack pre-computed SDFs. Computing them here would take so long that tests would
    // time out on CI. Therefore, we disable all colliders to prevent SDF generation. That is OK
    // because SDF generation is thoroughly tested elsewhere.
    for (auto& link : botPrefab.links) {
      link.colliderType = ColliderType::None;
    }

    /* A bot may declare a sensor or actuator type this build does not have -- some are
     * internal-only and stripped from the open-source export, and some (e.g. custom example
     * components) are registered only by an example script -- while the bots naming them still
     * ship. Creation skips such a component and warns, and the bot itself must still load, which is
     * what this asserts. Detected by asking the context what it has rather than by naming a type,
     * so no shipped test source has to know which components a given build includes. */
    bool expectSkipWarnings = false;
    for (auto const& link : botPrefab.links) {
      for (auto const& sensor : link.sensors) {
        if (!botsContext->IsSensorTypeRegistered(
                std::string_view(sensor.type.c_str(), sensor.type.size()))) {
          expectSkipWarnings = true;
        }
      }
      for (auto const& actuator : link.actuators) {
        if (!botsContext->IsActuatorTypeRegistered(
                std::string_view(actuator.type.c_str(), actuator.type.size()))) {
          expectSkipWarnings = true;
        }
      }
    }
    bool const warningsWereEnabled = IsLogChannelEnabled(LogChannel::Warning);
    if (expectSkipWarnings) {
      EnableLogChannel(LogChannel::Warning, false);
    }
    MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, warningsWereEnabled));

    Bot* bot = CreateBot(scene, botPrefab, botsContext, ExpectOK{});
    EXPECT_NE(bot, nullptr);
    if (bot != nullptr) {
      EXPECT_NE(bot->GetArticulatedActor(), nullptr);
      DestroyBot(scene, bot);
    }
  }

  DestroyRoboticsContext(botsContext);
  DestroyContext(context);
}

TEST(LoadAllBotsTest, BotFilesExist) {
  ASSERT_FALSE(FindAllBotFiles().empty());
}

// A bot prefab may declare inline sensors on a link (BotLinkPrefab::sensors / BotSensorPrefab).
// The BotSensorPrefab `name` field was added after some assets were authored, so existing assets
// omit it. Deserialization uses DeserializeFlags::Default (WarnIfExtraneousFields only -- it does
// NOT warn or fail on missing fields), so a sensor entry without a `name` loads fine and defaults
// `name` to empty. dg5f_seed is the one landed asset with an inline sensor, authored without a
// sensor name; this guards that loading such assets keeps working (no hard error on the missing
// field) as BotSensorPrefab gains fields.
TEST(LoadBotWithSensorTest, InlineSensorMissingName_LoadsWithEmptyDefaultName) {
  std::string const botPath = GetAssetsDir() + "bots/sensors/dg5f_seed/dg5f_seed.superdex_bot";
  BotPrefab const prefab = LoadBotPrefabFromFile(botPath, ExpectOK{});

  int sensorCount = 0;
  for (auto const& link : prefab.links) {
    for (auto const& sensor : link.sensors) {
      ++sensorCount;
      // dg5f_seed uses the legacy typeName/paramsFile keys; the loader folds them into type/params.
      EXPECT_FALSE(sensor.type.empty()); // folded from the legacy "typeName" key
      EXPECT_FALSE(sensor.params.empty()); // folded from legacy "paramsFile"
      EXPECT_TRUE(sensor._legacyTypeName.empty()); // legacy fields cleared after the fold
      EXPECT_TRUE(sensor.name.empty()); // omitted in the asset -> defaulted, not a load error
    }
  }
  EXPECT_EQ(sensorCount, 1);
}

// LoadModBotPrefab must apply the same legacy sensor-field fold as LoadBotPrefab: a mod bot whose
// AttachLink or ReplaceLink carries a link with legacy typeName/paramsFile sensor keys must fold
// them into type/params, or CreateSensor later fails with an unknown sensor type and the whole
// bot fails to load. This unit-tests the ModBotPrefab overload the loader now applies.
TEST(LoadBotWithSensorTest, ModBotInlineSensorLegacyFieldsFolded) {
  auto makeLegacySensor = []() {
    BotSensorPrefab sensor;
    sensor._legacyTypeName = "TEST_SENSOR";
    sensor._legacyParamsFile = "test.superdex_sensor";
    return sensor;
  };

  ModBotPrefab modBot;
  modBot.base = "base.superdex_bot";
  AttachLink attach;
  attach.link.sensors.push_back(makeLegacySensor());
  modBot.modifications.push_back(attach);
  ReplaceLink replace;
  replace.link.sensors.push_back(makeLegacySensor());
  modBot.modifications.push_back(replace);

  ApplyLegacyBotSensorFields(modBot);

  auto const& attachSensor = std::get<AttachLink>(modBot.modifications[0]).link.sensors[0];
  EXPECT_EQ(std::string(attachSensor.type.data(), attachSensor.type.size()), "TEST_SENSOR");
  EXPECT_EQ(
      std::string(attachSensor.params.data(), attachSensor.params.size()), "test.superdex_sensor");
  EXPECT_TRUE(attachSensor._legacyTypeName.empty());
  EXPECT_TRUE(attachSensor._legacyParamsFile.empty());

  auto const& replaceSensor = std::get<ReplaceLink>(modBot.modifications[1]).link.sensors[0];
  EXPECT_EQ(std::string(replaceSensor.type.data(), replaceSensor.type.size()), "TEST_SENSOR");
  EXPECT_TRUE(replaceSensor._legacyParamsFile.empty());
}

// LoadModBotPrefab must apply the same legacy joint-type default as LoadBotPrefab: a mod bot whose
// AttachLink/AttachBot connecting joint omitted "type" deserializes as
// ArticulatedJointType::Invalid (the old default was Hard). BuildBot pushes these joints into the
// assembled bot as-is, so without the fold the assembled bot would carry an Invalid joint. Joints
// with an explicit type are left untouched. ReplaceLink carries no joint, so it must be ignored.
TEST(LoadBotWithSensorTest, ModBotInlineJointTypeLegacyDefaulted) {
  ModBotPrefab modBot;
  modBot.base = "base.superdex_bot";

  AttachLink attachLegacy; // joint.type defaults to Invalid (legacy: omitted "type")
  modBot.modifications.push_back(attachLegacy);

  AttachBot attachBotExplicit;
  attachBotExplicit.joint.type = ArticulatedJointType::Revolute; // explicit type must be preserved
  modBot.modifications.push_back(attachBotExplicit);

  ReplaceLink replace; // no joint; must not throw
  modBot.modifications.push_back(replace);

  ApplyLegacyBotJointTypes(modBot);

  EXPECT_EQ(std::get<AttachLink>(modBot.modifications[0]).joint.type, ArticulatedJointType::Hard);
  EXPECT_EQ(
      std::get<AttachBot>(modBot.modifications[1]).joint.type, ArticulatedJointType::Revolute);
}

INSTANTIATE_TEST_SUITE_P(
    AllBots,
    LoadAllBotsTest,
    ::testing::ValuesIn(FindAllBotFiles()),
    [](::testing::TestParamInfo<std::filesystem::path> const& info) {
      std::string name = info.param.stem().string();
      for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
          c = '_';
        }
      }
      return name;
    });
