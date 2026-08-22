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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/controllers/controller_basic_jsc_pd.h>
#include <superdex_robotics/controllers/controller_basic_osc_pd.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/sensors/camera_sensor.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

// Minimal loader that returns empty shapes; used to exercise BuildArticulatedActorParams
// for prefabs whose links carry no shape files.
struct NoOpBotLoader : IBotLoader {
  BotFileType GetBotFileType(std::string_view, Error&) const override {
    return BotFileType::BotPrefab;
  }
  BotPrefab LoadBotPrefab(std::string_view, Error&) const override {
    return {};
  }
  ModBotPrefab LoadModBotPrefab(std::string_view, Error&) const override {
    return {};
  }
  ShapeHandle LoadShape(std::string_view, Real3 const&, TransformRT const&, Context*, Error&)
      const override {
    return {};
  }
};

/* Test fixture that creates a Mochi Context + Scene for controller integration tests.
 * Each test gets a fresh RoboticsContext via CreateRoboticsContext() / DestroyRoboticsContext(). */
class RoboticsContextTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    ASSERT_NE(_mochiContext, nullptr);
    _scene = _mochiContext->CreateScene("RoboticsContextTest");
    ASSERT_NE(_scene, nullptr);
    _scene->SetGravity(Real3(0_r, 0_r, 0_r));
    _botsCtx = CreateRoboticsContext();
    ASSERT_NE(_botsCtx, nullptr);
  }

  void TearDown() override {
    /* Null-guarded so a test may pre-destroy the scene (e.g. to exercise scene-destroyed-while-a-
     * controller-is-still-alive) without TearDown double-destroying it. */
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

  RoboticsContext& GetCtx() {
    return *_botsCtx;
  }

  /* Create a minimal articulated actor with the given number of revolute joints. */
  Actor* CreateChainRobot(int numLinks) {
    /* Tet-mesh cube shape for each link (same pattern as mochi_physics constraint tests) */
    auto&& [coordinates, connectivity] = CreateMinimalTetMeshUnitCube();
    auto linkShape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ExpectOK{});

    ArticulatedActorParams actorParams;
    actorParams.name = "TestRobot";
    actorParams.joints.resize(numLinks);
    actorParams.links.resize(numLinks);
    for (int i = 0; i < numLinks; ++i) {
      auto& joint = actorParams.joints[i];
      joint.name = DynamicString("joint") + DynamicString(std::to_string(i));
      joint.type = ArticulatedJointType::Revolute;
      joint.axis = Real3{0_r, 0_r, 1_r};
      // Root joint stays at the identity default; subsequent joints offset one unit along +X.
      if (i > 0) {
        joint.parentLinkFromJoint = TransformRT{Real3{1_r, 0_r, 0_r}};
      }

      auto& link = actorParams.links[i];
      link.name = (i == 0) ? DynamicString("base_link")
                           : DynamicString("link") + DynamicString(std::to_string(i));
      link.parentLink = i - 1;
      link.shape = linkShape;
      link.density = 1000_r;
    }
    return _scene->CreateArticulatedActor(actorParams, ExpectOK{});
  }

  Context* _mochiContext = nullptr;
  Scene* _scene = nullptr;
  RoboticsContext* _botsCtx = nullptr;
};

/* ---------- Basic handle lifecycle ---------- */

TEST_F(RoboticsContextTest, DefaultHandleIsInvalid) {
  ControllerHandle handle;
  EXPECT_FALSE(handle.IsValid());
  EXPECT_FALSE(GetCtx().IsValidController(handle));
  EXPECT_EQ(GetCtx().GetController(handle), nullptr);
}

TEST_F(RoboticsContextTest, CreateAndDestroyJSC) {
  Actor* robot = CreateChainRobot(4);
  Error error;
  auto handle =
      GetCtx().CreateController("BASIC_JSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  EXPECT_TRUE(error.IsOK());
  EXPECT_TRUE(handle.IsValid());
  EXPECT_TRUE(GetCtx().IsValidController(handle));

  auto* controller = GetCtx().GetController(handle);
  ASSERT_NE(controller, nullptr);

  GetCtx().DestroyController(handle);
  EXPECT_FALSE(GetCtx().IsValidController(handle));
  EXPECT_EQ(GetCtx().GetController(handle), nullptr);
}

// Owner inference must not claim standalone articulations: a scene may hold actors that belong to
// no bot, and components created on them stay unowned.
TEST_F(RoboticsContextTest, ControllerOnNonBotActorHasNoOwningBot) {
  Actor* robot = CreateChainRobot(4);
  Error error;
  auto handle =
      GetCtx().CreateController("BASIC_JSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  ASSERT_TRUE(error.IsOK());

  auto* controller = GetCtx().GetController(handle);
  ASSERT_NE(controller, nullptr);
  EXPECT_EQ(controller->GetOwningBot(), nullptr);
  EXPECT_EQ(GetCtx().GetBotContainingActor(robot), nullptr);
}

TEST_F(RoboticsContextTest, CreateAndDestroyOSC) {
  Actor* robot = CreateChainRobot(6);
  Error error;
  auto handle =
      GetCtx().CreateController("BASIC_OSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  EXPECT_TRUE(error.IsOK());
  EXPECT_TRUE(handle.IsValid());

  auto* osc = static_cast<ControllerBasicOscPd*>(GetCtx().GetController(handle));
  ASSERT_NE(osc, nullptr);
  osc->Initialize("TestRobot/base_link", "TestRobot/link5", error);
  EXPECT_TRUE(error.IsOK());

  GetCtx().DestroyController(handle);
  EXPECT_FALSE(GetCtx().IsValidController(handle));
}

// Regression: an OSC-family controller caches its owning scene and end-effector link as stale-safe
// handles (ComponentBase::GetScene / ControllerBasicOscPd::_eeLinkHandle) rather than raw pointers.
// If the mochi Scene is destroyed while the controller is still alive, the observation path must
// resolve those handles to nullptr and return an error instead of dereferencing a dangling
// Scene*/Actor*. Before the handle migration this cached a raw Scene*/Actor* and crashed here.
TEST_F(RoboticsContextTest, OscControllerToleratesSceneDestroyedWhileAlive) {
  Actor* robot = CreateChainRobot(6);
  Error error;
  auto handle =
      GetCtx().CreateController("BASIC_OSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  ASSERT_TRUE(error.IsOK());

  auto* osc = static_cast<ControllerBasicOscPd*>(GetCtx().GetController(handle));
  ASSERT_NE(osc, nullptr);
  osc->Initialize("TestRobot/base_link", "TestRobot/link5", error);
  ASSERT_TRUE(error.IsOK());

  // While the scene is alive, the observation path resolves the EE link and succeeds.
  Error aliveError;
  osc->GetCurrentObservationsFromMochi(aliveError);
  EXPECT_TRUE(aliveError.IsOK());

  // Destroy the mochi Scene out from under the still-alive controller.
  _mochiContext->DestroyScene(_scene);
  _scene = nullptr;

  // GetScene()/GetActor() now resolve to nullptr, so the controller reports an error rather than
  // dereferencing a freed actor.
  Error deadError;
  osc->GetCurrentObservationsFromMochi(deadError);
  EXPECT_FALSE(deadError.IsOK());

  // The direct ComputeOutput(obsv, target) path reads the cached actor rather than going through
  // GetCurrentObservationsFromMochi; it re-resolves the actor up front (RefreshActor) and likewise
  // returns an error instead of dereferencing it. An empty obsv is fine — the dead-actor guard
  // fires before any observation validation.
  Error computeError;
  osc->ComputeOutput(ControllerBasicOscPd::Obsv{}, ControllerBasicOscPd::Target{}, computeError);
  EXPECT_FALSE(computeError.IsOK());

  // Destroying the controller after its scene is gone is safe (ComponentBase::Destroy re-resolves
  // the cached actor to nullptr first); the RoboticsContext is torn down in TearDown.
  GetCtx().DestroyController(handle);
  EXPECT_FALSE(GetCtx().IsValidController(handle));
}

TEST_F(RoboticsContextTest, UnknownTypeReturnsError) {
  Actor* robot = CreateChainRobot(4);
  Error error;
  auto handle =
      GetCtx().CreateController("NONEXISTENT_TYPE", /*prefab*/ nullptr, robot, /*name*/ "", error);
  EXPECT_FALSE(error.IsOK());
  EXPECT_FALSE(handle.IsValid());
}

// A controller may be created with no actor: that is how one is run against a real robot, with
// observations pushed in externally instead of read off a Mochi articulation.
TEST_F(RoboticsContextTest, CreateControllerAcceptsNullRobot) {
  Error error;
  auto handle = GetCtx().CreateController(
      "BASIC_JSC_PD", /*prefab*/ nullptr, static_cast<Actor*>(nullptr), /*name*/ "", error);
  EXPECT_TRUE(error.IsOK());
  EXPECT_TRUE(handle.IsValid());

  auto* controller = GetCtx().GetController(handle);
  ASSERT_NE(controller, nullptr);
  EXPECT_EQ(controller->GetActor(), nullptr);
}

// The exception is MOCHI_ARTICULATED_POSE, which drives Mochi's built-in implicit PD controller and
// so cannot run without a simulation. It rejects the null actor itself rather than relying on a
// blanket check in the creation path.
TEST_F(RoboticsContextTest, MochiArticulatedPoseRejectsNullRobot) {
  Error error;
  auto handle = GetCtx().CreateController(
      "MOCHI_ARTICULATED_POSE",
      /*prefab*/ nullptr,
      static_cast<Actor*>(nullptr),
      /*name*/ "",
      error);
  EXPECT_FALSE(error.IsOK());
  EXPECT_FALSE(handle.IsValid());
}

TEST_F(RoboticsContextTest, DestroyInvalidHandleIsNoOp) {
  // Should not crash
  GetCtx().DestroyController(ControllerHandle{});
  GetCtx().DestroyController(ControllerHandle{999});
}

TEST_F(RoboticsContextTest, MultipleControllers) {
  Actor* robot = CreateChainRobot(6);
  Error error;

  auto oscHandle =
      GetCtx().CreateController("BASIC_OSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  EXPECT_TRUE(error.IsOK());

  auto jscHandle =
      GetCtx().CreateController("BASIC_JSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  EXPECT_TRUE(error.IsOK());

  EXPECT_TRUE(GetCtx().IsValidController(oscHandle));
  EXPECT_TRUE(GetCtx().IsValidController(jscHandle));
  EXPECT_NE(oscHandle, jscHandle);

  // Destroy one, the other should still be valid
  GetCtx().DestroyController(oscHandle);
  EXPECT_FALSE(GetCtx().IsValidController(oscHandle));
  EXPECT_TRUE(GetCtx().IsValidController(jscHandle));

  GetCtx().DestroyController(jscHandle);
}

/* ---------- C5: Bad link names produce graceful error ---------- */

TEST_F(RoboticsContextTest, BadLinkNamesGracefulError) {
  Actor* robot = CreateChainRobot(6);
  Error error;

  auto handle =
      GetCtx().CreateController("BASIC_OSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  EXPECT_TRUE(error.IsOK());

  auto* osc = static_cast<ControllerBasicOscPd*>(GetCtx().GetController(handle));
  ASSERT_NE(osc, nullptr);

  /* Suppress expected warnings from the OSC controller's diagnostics log */
  auto prevCallback = Context::GetLogCallback();
  Context::SetLogCallback([](LogChannel, char const*, char const*, int) {});

  Error initError;
  osc->Initialize("nonexistent_base", "nonexistent_ee", initError);
  EXPECT_FALSE(initError.IsOK());

  Context::SetLogCallback(prevCallback);
  GetCtx().DestroyController(handle);
}

/* ---------- DestroyRoboticsContext cleans up ---------- */

TEST_F(RoboticsContextTest, DestroyAndRecreate) {
  Actor* robot = CreateChainRobot(4);
  Error error;
  auto handle =
      GetCtx().CreateController("BASIC_JSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  EXPECT_TRUE(error.IsOK());
  EXPECT_TRUE(GetCtx().IsValidController(handle));

  /* Destroy and recreate — old handles must be invalid in the new context. */
  DestroyRoboticsContext(_botsCtx);
  _botsCtx = CreateRoboticsContext();

  EXPECT_FALSE(GetCtx().IsValidController(handle));
}

/* ---------- Typed handles ---------- */

// Component kinds are now distinct C++ types (ControllerHandle/SensorHandle/...), so misuse is a
// compile-time error rather than a runtime tag check. Each still carries and round-trips its raw
// value from the shared RoboticsHandle base, and a default-constructed handle is invalid.
TEST_F(RoboticsContextTest, TypedHandlesCarryValue) {
  ControllerHandle const ctrl(42);
  SensorHandle const sensor(7);

  EXPECT_TRUE(ctrl.IsValid());
  EXPECT_TRUE(sensor.IsValid());
  EXPECT_EQ(ctrl.value, RoboticsHandle::ValueType{42});
  EXPECT_EQ(sensor.value, RoboticsHandle::ValueType{7});
  EXPECT_FALSE(ControllerHandle{}.IsValid());
  EXPECT_FALSE(SensorHandle{}.IsValid());
}

TEST_F(RoboticsContextTest, DoubleDestroyIsNoOp) {
  Actor* robot = CreateChainRobot(4);
  Error error;
  auto handle =
      GetCtx().CreateController("BASIC_JSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  EXPECT_TRUE(error.IsOK());
  EXPECT_TRUE(GetCtx().IsValidController(handle));

  GetCtx().DestroyController(handle);
  EXPECT_FALSE(GetCtx().IsValidController(handle));

  /* Second destroy of the same handle should be a no-op, not crash. */
  GetCtx().DestroyController(handle);
  EXPECT_FALSE(GetCtx().IsValidController(handle));
}

TEST_F(RoboticsContextTest, UnknownControllerHandleRejectedByControllerCRUD) {
  /* Passing a foreign component kind is now a compile-time error, so what remains to check is that
   * a well-formed ControllerHandle whose value was never issued (e.g. one that belongs to another
   * slot map's domain) is rejected by the controller accessors and is a no-op to destroy. */
  ControllerHandle const bogus{1};
  EXPECT_FALSE(GetCtx().IsValidController(bogus));
  EXPECT_EQ(GetCtx().GetController(bogus), nullptr);
  // Should not crash
  GetCtx().DestroyController(bogus);
}

/* ---------- Sensor lifecycle ---------- */

/* Exercise RoboticsContext lifecycle and error paths with the public camera sensor. */

TEST_F(RoboticsContextTest, CreateAndDestroySensor) {
  /* Use a rigid body for proper contact query support. */
  auto&& [coordinates, connectivity] = CreateMinimalTetMeshUnitCube();
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ExpectOK{});
  RigidActorParams rigidParams;
  rigidParams.shape = shape;
  rigidParams.name = "sensor_link";
  rigidParams.density = 1000_r;
  auto* linkActor = _scene->CreateRigidActor(rigidParams, ExpectOK{});

  auto handle =
      GetCtx().CreateSensor(CameraSensor::TypeName(), linkActor, /*name*/ {}, {}, ExpectOK{});
  EXPECT_TRUE(handle.IsValid());
  EXPECT_TRUE(GetCtx().IsValidSensor(handle));

  auto* sensor = GetCtx().GetSensor(handle);
  ASSERT_NE(sensor, nullptr);
  EXPECT_TRUE(sensor->IsValid());

  GetCtx().DestroySensor(handle);
  EXPECT_FALSE(GetCtx().IsValidSensor(handle));
  EXPECT_EQ(GetCtx().GetSensor(handle), nullptr);
}

TEST_F(RoboticsContextTest, UnknownSensorTypeReturnsError) {
  Actor* robot = CreateChainRobot(4);
  Error error;
  auto handle = GetCtx().CreateSensor("NONEXISTENT_SENSOR", robot, /*name*/ {}, {}, error);
  EXPECT_FALSE(error.IsOK());
  EXPECT_FALSE(handle.IsValid());
}

// The legacy sensor aliases "fixed_camera" and "wrist_camera" resolve to the same sensor as
// SENSOR_CAMERA (CameraSensor), so old (internal) assets keep loading. Registered by
// RegisterLegacySensorTypes, which is internal-only, so this test is gated to internal builds.
#if MOCHI_INTERNAL
TEST_F(RoboticsContextTest, LegacyCameraAliasesResolveToCameraSensor) {
  Actor* robot = CreateChainRobot(4);
  for (char const* legacyName : {"fixed_camera", "wrist_camera"}) {
    auto handle = GetCtx().CreateSensor(legacyName, robot, {}, /*name*/ {}, ExpectOK{});
    EXPECT_TRUE(handle.IsValid());
    EXPECT_NE(dynamic_cast<CameraSensor*>(GetCtx().GetSensor(handle)), nullptr);
    GetCtx().DestroySensor(handle);
  }
}
#endif // MOCHI_INTERNAL

TEST_F(RoboticsContextTest, DestroyInvalidSensorHandleIsNoOp) {
  SensorHandle defaultHandle{};
  SensorHandle bogusHandle{999};

  GetCtx().DestroySensor(defaultHandle);
  GetCtx().DestroySensor(bogusHandle);

  // Verify state is unchanged after destroying invalid handles
  EXPECT_FALSE(GetCtx().IsValidSensor(defaultHandle));
  EXPECT_FALSE(GetCtx().IsValidSensor(bogusHandle));
  EXPECT_EQ(GetCtx().GetSensor(defaultHandle), nullptr);
  EXPECT_EQ(GetCtx().GetSensor(bogusHandle), nullptr);
}

// A sensor may be created with no link actor: that is how a scene-level sensor (e.g. an egocentric
// camera, posed relative to the scene root) is expressed. It belongs to no bot.
TEST_F(RoboticsContextTest, CreateSensorAcceptsNullLinkActor) {
  Error error;
  auto handle = GetCtx().CreateSensor(
      "SENSOR_CAMERA", /*linkActor*/ nullptr, /*name*/ "", /*paramArgs*/ "", error);
  ASSERT_TRUE(error.IsOK());
  ASSERT_TRUE(handle.IsValid());

  auto* sensor = GetCtx().GetSensor(handle);
  ASSERT_NE(sensor, nullptr);
  EXPECT_EQ(sensor->GetOwningBot(), nullptr);
}

// An actuator always drives a body, so it has no scene-level form: a null link actor is a caller
// error rather than the "not attached to a bot" case it means for a sensor.
TEST_F(RoboticsContextTest, CreateActuatorRejectsNullLinkActor) {
  Error error;
  auto handle = GetCtx().CreateActuator(
      "ANY_ACTUATOR", /*linkActor*/ nullptr, /*name*/ "", /*paramArgs*/ "", error);
  EXPECT_FALSE(error.IsOK());
  EXPECT_FALSE(handle.IsValid());
  /* Assert on the message, not just the failure: no actuator type is registered yet, so an
   * unattributed EXPECT_FALSE would also pass on the unknown-type branch -- and would keep passing
   * if the null-actor check were deleted. The null check runs first, so this pins that branch. */
  EXPECT_STREQ(error.GetDescription(), "RoboticsContext::CreateActuator: link actor is null");
}

// Each component kind can report whether a type name is registered.
TEST_F(RoboticsContextTest, TypeRegistrationQueries) {
  EXPECT_TRUE(GetCtx().IsControllerTypeRegistered("BASIC_OSC_PD"));
  EXPECT_FALSE(GetCtx().IsControllerTypeRegistered("NOT_A_CONTROLLER"));
  EXPECT_TRUE(GetCtx().IsSensorTypeRegistered("SENSOR_CAMERA"));
  EXPECT_FALSE(GetCtx().IsSensorTypeRegistered("NOT_A_SENSOR"));
  // No concrete actuator type is registered yet, so only the negative case is assertable.
  EXPECT_FALSE(GetCtx().IsActuatorTypeRegistered("NOT_AN_ACTUATOR"));
}

TEST_F(RoboticsContextTest, ControllerAndSensorHandlesAreDistinct) {
  Actor* robot = CreateChainRobot(6);
  Error error;

  auto ctrlHandle =
      GetCtx().CreateController("BASIC_OSC_PD", /*prefab*/ nullptr, robot, /*name*/ "", error);
  EXPECT_TRUE(error.IsOK());

  auto&& [coordinates, connectivity] = CreateMinimalTetMeshUnitCube();
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ExpectOK{});
  RigidActorParams rigidParams;
  rigidParams.shape = shape;
  rigidParams.name = "sensor_link";
  rigidParams.density = 1000_r;
  auto* linkActor = _scene->CreateRigidActor(rigidParams, ExpectOK{});

  auto sensorHandle =
      GetCtx().CreateSensor(CameraSensor::TypeName(), linkActor, /*name*/ {}, {}, ExpectOK{});
  EXPECT_TRUE(sensorHandle.IsValid());

  /* Controllers and sensors are now distinct C++ types (ControllerHandle vs SensorHandle), so
   * cross-kind lookups such as GetSensor(ctrlHandle) no longer compile — the confusion this test
   * once caught at runtime is prevented at compile time. The shared monotonic allocator still gives
   * every live handle a unique value, so the two never collide. */
  EXPECT_NE(ctrlHandle.value, sensorHandle.value);

  GetCtx().DestroyController(ctrlHandle);
  GetCtx().DestroySensor(sensorHandle);
}

TEST_F(RoboticsContextTest, ContextDestroyCleansSensors) {
  auto&& [coordinates, connectivity] = CreateMinimalTetMeshUnitCube();
  auto shape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ExpectOK{});
  RigidActorParams rigidParams;
  rigidParams.shape = shape;
  rigidParams.name = "sensor_link";
  rigidParams.density = 1000_r;
  auto* linkActor = _scene->CreateRigidActor(rigidParams, ExpectOK{});

  auto handle =
      GetCtx().CreateSensor(CameraSensor::TypeName(), linkActor, /*name*/ {}, {}, ExpectOK{});
  EXPECT_TRUE(handle.IsValid());
  EXPECT_TRUE(GetCtx().IsValidSensor(handle));

  /* Destroy and recreate — should destroy all sensors without crashing. */
  DestroyRoboticsContext(_botsCtx);
  _botsCtx = CreateRoboticsContext();

  /* New context — old handle should be invalid. */
  EXPECT_FALSE(GetCtx().IsValidSensor(handle));
}

/* ---------- Cycle joints ---------- */

// BuildArticulatedActorParams copies BotPrefab::cycles into the actor params unchanged.
TEST_F(RoboticsContextTest, BuildArticulatedActorParamsCopiesCycles) {
  BotPrefab bp;
  bp.name = "cycle_bot";

  BotLinkPrefab root;
  root.name = "root";
  root.parentLink = kIndexNone;
  BotLinkPrefab a;
  a.name = "a_link";
  a.parentLink = 0;
  BotLinkPrefab b;
  b.name = "b_link";
  b.parentLink = 0;
  bp.links = {root, a, b};

  BotJointPrefab rootJoint;
  rootJoint.name = "root_joint";
  rootJoint.type = ArticulatedJointType::Hard;
  BotJointPrefab aJoint;
  aJoint.name = "a_joint";
  aJoint.type = ArticulatedJointType::Revolute;
  aJoint.axis = {0_r, 0_r, 1_r};
  BotJointPrefab bJoint;
  bJoint.name = "b_joint";
  bJoint.type = ArticulatedJointType::Revolute;
  bJoint.axis = {0_r, 0_r, 1_r};
  bp.joints = {rootJoint, aJoint, bJoint};

  ArticulatedCycleJointParams cycle;
  cycle.parentLink = 1;
  cycle.childLink = 2;
  cycle.stiffness = 4242_r;
  bp.cycles.push_back(cycle);

  NoOpBotLoader loader;
  auto const params = BuildArticulatedActorParams(bp, loader, _mochiContext, ExpectOK{});

  ASSERT_EQ(isize(params.cycles), 1);
  EXPECT_EQ(params.cycles[0].parentLink, 1);
  EXPECT_EQ(params.cycles[0].childLink, 2);
  EXPECT_NEAR(static_cast<double>(params.cycles[0].stiffness), 4242.0, 1e-2);
}

} // namespace
