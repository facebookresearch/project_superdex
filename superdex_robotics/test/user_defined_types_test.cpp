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

/* Exercises the contract a downstream user of the library relies on: define new controller,
 * sensor and actuator types outside the library, register them, and have them instantiate both
 * when created directly and when declared in a bot prefab. The types below deliberately use only
 * the public headers and the documented registration archetypes -- nothing here is reachable only
 * from inside the library. */

#include "mochi_bots_test_helpers.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/actuators/actuator_base.h>
#include <superdex_robotics/controllers/controller_base.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/sensors/sensor_base.h>
#include <superdex_robotics/superdex_robotics.h>

#include <string>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

class UserController final : public ControllerBase {
 public:
  static constexpr std::string_view TypeName() {
    return "USER_CONTROLLER";
  }

  UserController(BotPrefab const* prefab, Actor* actor, superdex::Error& error)
      : ControllerBase(prefab, actor, error), _sawPrefab(prefab != nullptr) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  void Reset() override {
    ++_resetCount;
  }

  /* Controllers -- unlike sensors and actuators -- are configured after construction, because the
   * scene entry that names them also carries their params. */
  void ConfigureFromSceneEntry(
      std::string_view paramArgs,
      std::string_view initArgs,
      superdex::Error& error) override {
    MOCHI_ERROR_RETURN(error);
    _paramArgs = paramArgs;
    _initArgs = initArgs;
  }

  [[nodiscard]] bool SawPrefab() const {
    return _sawPrefab;
  }
  [[nodiscard]] int ResetCount() const {
    return _resetCount;
  }
  [[nodiscard]] std::string const& ParamArgs() const {
    return _paramArgs;
  }
  [[nodiscard]] std::string const& InitArgs() const {
    return _initArgs;
  }

 private:
  bool _sawPrefab = false;
  int _resetCount = 0;
  std::string _paramArgs;
  std::string _initArgs;
};

class UserSensor final : public SensorBase {
 public:
  static constexpr std::string_view TypeName() {
    return "USER_SENSOR";
  }

  UserSensor(Actor* actor, std::string_view paramArgs, superdex::Error& error)
      : SensorBase(actor, error), _paramArgs(paramArgs) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  void Reset() override {
    ++_resetCount;
  }

  [[nodiscard]] std::string const& ParamArgs() const {
    return _paramArgs;
  }
  [[nodiscard]] int ResetCount() const {
    return _resetCount;
  }

 private:
  /* Copied, not referenced: paramArgs is a view onto the caller's (often temporary) string. */
  std::string _paramArgs;
  int _resetCount = 0;
};

/* A second user sensor type, to confirm several independently defined types of the same kind can
 * be registered and told apart. */
class OtherUserSensor final : public SensorBase {
 public:
  static constexpr std::string_view TypeName() {
    return "OTHER_USER_SENSOR";
  }

  OtherUserSensor(Actor* actor, std::string_view /*paramArgs*/, superdex::Error& error)
      : SensorBase(actor, error) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  void Reset() override {}
};

class UserActuator final : public ActuatorBase {
 public:
  static constexpr std::string_view TypeName() {
    return "USER_ACTUATOR";
  }

  UserActuator(Actor* actor, std::string_view paramArgs, superdex::Error& error)
      : ActuatorBase(actor, error), _paramArgs(paramArgs) {}

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  void Reset() override {}

  [[nodiscard]] std::string const& ParamArgs() const {
    return _paramArgs;
  }

 private:
  std::string _paramArgs;
};

class UserDefinedTypesTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    ASSERT_NE(_mochiContext, nullptr);
    _scene = _mochiContext->CreateScene("UserDefinedTypesTest");
    ASSERT_NE(_scene, nullptr);
    _botsCtx = CreateRoboticsContext();
    ASSERT_NE(_botsCtx, nullptr);

    _botsCtx->RegisterController<UserController>();
    _botsCtx->RegisterSensor<UserSensor>();
    _botsCtx->RegisterSensor<OtherUserSensor>();
    _botsCtx->RegisterActuator<UserActuator>();
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

TEST_F(UserDefinedTypesTest, UserTypesReportAsRegistered) {
  EXPECT_TRUE(_botsCtx->IsControllerTypeRegistered(UserController::TypeName()));
  EXPECT_TRUE(_botsCtx->IsSensorTypeRegistered(UserSensor::TypeName()));
  EXPECT_TRUE(_botsCtx->IsSensorTypeRegistered(OtherUserSensor::TypeName()));
  EXPECT_TRUE(_botsCtx->IsActuatorTypeRegistered(UserActuator::TypeName()));
  EXPECT_FALSE(_botsCtx->IsSensorTypeRegistered("NEVER_REGISTERED"));
}

// A user controller and sensor can be created straight off the context with no bot and no actor
// (the real-robot / fusion-sensor case), carrying the name they were created with.
TEST_F(UserDefinedTypesTest, CreatesUserControllerAndSensorWithoutAnActor) {
  ControllerHandle const controllerHandle = _botsCtx->CreateController(
      UserController::TypeName(), nullptr, nullptr, "user_ctrl", ExpectOK{});
  auto* const controller = dynamic_cast<UserController*>(_botsCtx->GetController(controllerHandle));
  ASSERT_NE(controller, nullptr);
  EXPECT_EQ(controller->GetTypeName(), UserController::TypeName());
  EXPECT_EQ(controller->GetName(), "user_ctrl");
  EXPECT_EQ(controller->GetActor(), nullptr);
  EXPECT_FALSE(controller->SawPrefab());

  SensorHandle const sensorHandle = _botsCtx->CreateSensor(
      UserSensor::TypeName(), nullptr, "user_sensor", R"({"rate": 30})", ExpectOK{});
  auto* const sensor = dynamic_cast<UserSensor*>(_botsCtx->GetSensor(sensorHandle));
  ASSERT_NE(sensor, nullptr);
  EXPECT_EQ(sensor->GetTypeName(), UserSensor::TypeName());
  EXPECT_EQ(sensor->GetName(), "user_sensor");
  /* The paramArgs the caller passed reach the user constructor verbatim -- this is the only
   * configuration channel a sensor or actuator gets. */
  EXPECT_EQ(sensor->ParamArgs(), R"({"rate": 30})");

  // Framework Reset() dispatches to the user override through the base pointer.
  _botsCtx->GetSensor(sensorHandle)->Reset();
  EXPECT_EQ(sensor->ResetCount(), 1);
}

// An actuator requires an actor (ActuatorBase enforces it), so it is created against a bot link.
TEST_F(UserDefinedTypesTest, CreatesUserActuatorOnAnActor) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  ActuatorHandle const handle = _botsCtx->CreateActuator(
      UserActuator::TypeName(),
      bot->GetArticulatedActor(),
      "user_actuator",
      "params.json",
      ExpectOK{});
  auto* const actuator = dynamic_cast<UserActuator*>(_botsCtx->GetActuator(handle));
  ASSERT_NE(actuator, nullptr);
  EXPECT_EQ(actuator->GetTypeName(), UserActuator::TypeName());
  EXPECT_EQ(actuator->GetName(), "user_actuator");
  EXPECT_EQ(actuator->ParamArgs(), "params.json");
  /* Created on an actor that belongs to a bot, so the context infers that bot as the owner. */
  EXPECT_EQ(actuator->GetOwningBot(), bot);
}

// Independently defined types of the same kind coexist and stay distinguishable by type name.
TEST_F(UserDefinedTypesTest, MultipleUserSensorTypesCoexist) {
  SensorHandle const first =
      _botsCtx->CreateSensor(UserSensor::TypeName(), nullptr, "a", "", ExpectOK{});
  SensorHandle const second =
      _botsCtx->CreateSensor(OtherUserSensor::TypeName(), nullptr, "b", "", ExpectOK{});
  ASSERT_TRUE(first.IsValid());
  ASSERT_TRUE(second.IsValid());

  DynamicArray<SensorHandle> const found = _botsCtx->FindSensorsByType(OtherUserSensor::TypeName());
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].value, second.value);
  EXPECT_NE(dynamic_cast<OtherUserSensor*>(_botsCtx->GetSensor(found[0])), nullptr);
}

// The bot-file path: a user sensor and actuator declared on a link of a bot prefab are
// instantiated by CreateBot and end up owned by, and reachable from, the bot.
TEST_F(UserDefinedTypesTest, BotPrefabDeclaredUserComponentsAreInstantiated) {
  BotPrefab prefab = LoadFR3();
  ASSERT_FALSE(prefab.links.empty());
  std::string const linkName(prefab.links[0].name.c_str(), prefab.links[0].name.size());

  BotSensorPrefab sensorPrefab;
  sensorPrefab.type = UserSensor::TypeName();
  sensorPrefab.name = "prefab_sensor";
  sensorPrefab.params = R"({"from": "prefab"})";
  prefab.links[0].sensors.push_back(sensorPrefab);

  BotActuatorPrefab actuatorPrefab;
  actuatorPrefab.type = UserActuator::TypeName();
  actuatorPrefab.name = "prefab_actuator";
  prefab.links[0].actuators.push_back(actuatorPrefab);

  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);

  DynamicArray<SensorHandle> const sensorHandles = bot->GetSensorHandles();
  ASSERT_EQ(sensorHandles.size(), 1u);
  auto* const sensor = dynamic_cast<UserSensor*>(bot->GetSensor(sensorHandles[0], ExpectOK{}));
  ASSERT_NE(sensor, nullptr);
  EXPECT_EQ(sensor->GetName(), "prefab_sensor");
  EXPECT_EQ(sensor->ParamArgs(), R"({"from": "prefab"})");
  EXPECT_EQ(sensor->GetOwningBot(), bot);
  EXPECT_STREQ(bot->GetSensorLinkName(sensorHandles[0], ExpectOK{}), linkName.c_str());

  DynamicArray<ActuatorHandle> const actuatorHandles = bot->GetActuatorHandles();
  ASSERT_EQ(actuatorHandles.size(), 1u);
  auto* const actuator =
      dynamic_cast<UserActuator*>(bot->GetActuator(actuatorHandles[0], ExpectOK{}));
  ASSERT_NE(actuator, nullptr);
  EXPECT_EQ(actuator->GetName(), "prefab_actuator");
  EXPECT_EQ(actuator->GetOwningBot(), bot);
  EXPECT_STREQ(bot->GetActuatorLinkName(actuatorHandles[0], ExpectOK{}), linkName.c_str());
}

// A user type can also be registered on the bot itself by type string, the path a scene entry or
// interactive tooling takes.
TEST_F(UserDefinedTypesTest, BotCreatesUserComponentsByTypeName) {
  BotPrefab const prefab = LoadFR3();
  Bot* const bot = CreateBot(_scene, prefab, _botsCtx, ExpectOK{});
  ASSERT_NE(bot, nullptr);
  std::string const linkName(prefab.links[0].name.c_str(), prefab.links[0].name.size());

  ControllerBase* const controller =
      bot->CreateController(UserController::TypeName(), "user_ctrl", ExpectOK{});
  ASSERT_NE(dynamic_cast<UserController*>(controller), nullptr);
  /* The bot passes its own prefab down, so a user controller can read the model it drives. */
  EXPECT_TRUE(dynamic_cast<UserController*>(controller)->SawPrefab());

  SensorBase* const sensor =
      bot->CreateSensor(UserSensor::TypeName(), linkName, "s", "", ExpectOK{});
  ASSERT_NE(dynamic_cast<UserSensor*>(sensor), nullptr);

  ActuatorBase* const actuator =
      bot->CreateActuator(UserActuator::TypeName(), linkName, "a", "", ExpectOK{});
  ASSERT_NE(dynamic_cast<UserActuator*>(actuator), nullptr);
}

// Registration accepts a stateful (capturing) factory, not just a plain function. The Python
// bindings depend on this to hold the Python callable that builds each instance.
TEST_F(UserDefinedTypesTest, RegistrationAcceptsCapturingFactories) {
  int sensorsBuilt = 0;
  _botsCtx->RegisterSensorType(
      "CAPTURING_SENSOR",
      [&sensorsBuilt](
          Actor* actor, std::string_view paramArgs, superdex::Error& error) -> SensorBase* {
        ++sensorsBuilt;
        return new UserSensor(actor, paramArgs, error);
      });

  SensorHandle const first =
      _botsCtx->CreateSensor("CAPTURING_SENSOR", nullptr, "one", "", ExpectOK{});
  SensorHandle const second =
      _botsCtx->CreateSensor("CAPTURING_SENSOR", nullptr, "two", "", ExpectOK{});
  EXPECT_TRUE(first.IsValid());
  EXPECT_TRUE(second.IsValid());
  EXPECT_EQ(sensorsBuilt, 2);
}

} // namespace
