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
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/sensors/sensor_joint_encoder.h>
#include <superdex_robotics/superdex_robotics.h>

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

/* The FR3 arm's seven revolute joints, in its pose's order. */
std::array<std::string_view, 7> constexpr kFr3Joints = {
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7"};

class SensorJointEncoderTest : public testing::Test {
 protected:
  void SetUp() override {
    _mochiContext = mochi::CreateContext(0);
    ASSERT_NE(_mochiContext, nullptr);
    _scene = _mochiContext->CreateScene("SensorJointEncoderTest");
    ASSERT_NE(_scene, nullptr);
    _robotics = CreateRoboticsContext();
    ASSERT_NE(_robotics, nullptr);
    _prefab = LoadBotPrefabFromFile(GetAssetPath("bots/arms/fr3/fr3.superdex_bot"), ExpectOK{});
  }

  void TearDown() override {
    _graph.Clear();
    DestroyRoboticsContext(_robotics);
    _mochiContext->DestroyScene(_scene);
    mochi::DestroyContext(_mochiContext);
  }

  /* Creates an FR3 from _prefab whose last link declares an encoder with @p params, and graphs the
   * encoder. */
  SensorJointEncoder* CreateEncoder(std::string_view params, Error& error) {
    BotPrefab prefab = _prefab;
    BotSensorPrefab encoder;
    encoder.type = DynamicString(std::string(SensorJointEncoder::TypeName()));
    encoder.name = DynamicString("encoder");
    encoder.params = DynamicString(std::string(params));
    prefab.links.back().sensors.push_back(encoder);
    _defaultPose = prefab.defaultPose;
    Bot* const bot = CreateBot(_scene, prefab, _robotics, ExpectOK{});
    _arm = bot->GetArticulatedActor();
    DynamicArray<SensorHandle> const found = bot->FindSensorsByName("encoder");
    EXPECT_EQ(found.size(), 1);
    auto* const sensor = static_cast<SensorJointEncoder*>(_robotics->GetSensor(found[0]));
    _graph.Add("encoder", sensor, ExpectOK{});
    _graph.SetOutputEnabled("encoder", true, ExpectOK{});
    _graph.Build({.scene = _scene, .robotics = _robotics}, error);
    _graph.Reset(error);
    return sensor;
  }

  Context* _mochiContext = nullptr;
  Scene* _scene = nullptr;
  RoboticsContext* _robotics = nullptr;
  SensorGraph _graph;
  /* The FR3, which CreateEncoder instantiates. */
  BotPrefab _prefab;
  /* The last created bot's default pose, where it starts, and articulated actor. */
  DynamicArray<real> _defaultPose;
  Actor* _arm = nullptr;
};

DynamicArray<float> Values(SensorBase const& sensor) {
  return {sensor.Latest().Floats().begin(), sensor.Latest().Floats().end()};
}

/* @p arm's positions and velocities of @p joints as floats: the positions, then the velocities. */
DynamicArray<float> JointState(Actor const& arm, Span<std::string_view const> joints) {
  ArticulatedShapeInfo const shape = arm.GetArticulatedShapeInfo(ExpectOK{});
  DynamicArray<real> pose(static_cast<std::size_t>(arm.GetNumDofs()));
  DynamicArray<real> velocities(pose.size());
  arm.GetArticulatedPose(MakeSpan(pose), ExpectOK{});
  arm.GetArticulatedJointVelocities(MakeSpan(velocities), ExpectOK{});
  DynamicArray<float> state(2 * joints.size());
  for (std::size_t i = 0; i < joints.size(); ++i) {
    auto const joint = static_cast<std::size_t>(
        std::find(shape.jointNames.begin(), shape.jointNames.end(), joints[i]) -
        shape.jointNames.begin());
    auto const dof = static_cast<std::size_t>(shape.dofInfo[joint].offset);
    state[i] = static_cast<float>(pose[dof]);
    state[joints.size() + i] = static_cast<float>(velocities[dof]);
  }
  return state;
}

// With no joints listed, a bot link's encoder reads every one-DOF joint but the root in the bot's
// order, one without effort too, at the reset step too: a row of positions and a row of velocities,
// labelled by quantity and joint name.
TEST_F(SensorJointEncoderTest, ReadsEveryJointByDefault) {
  for (BotJointPrefab& joint : _prefab.joints) {
    if (joint.name == "fr3_joint7") {
      joint.effortLimit = 0_r;
    }
  }
  SensorJointEncoder* const encoder = CreateEncoder("", ExpectOK{});

  _graph.Step({.t = 0.0, .dt = 0.0, .step = 0}, ExpectOK{});

  SensorFormat const& format = encoder->GetFormat();
  EXPECT_EQ(format.kind, "JointState");
  EXPECT_EQ(format.shape, (DynamicArray<SensorDimension>{2, 7}));
  ASSERT_EQ(format.axes.size(), 2);
  EXPECT_EQ(
      format.axes[0],
      (SensorAxis{
          .name = "quantity", .labels = {"position", "velocity"}, .units = {"rad", "rad/s"}}));
  EXPECT_EQ(format.axes[1].name, "joint");
  EXPECT_EQ(
      format.axes[1].labels, (DynamicArray<DynamicString>(kFr3Joints.begin(), kFr3Joints.end())));
  EXPECT_TRUE(format.axes[1].units.empty());
  EXPECT_EQ(encoder->Latest().publishedStep, 0);
  ASSERT_EQ(_defaultPose.size(), kFr3Joints.size());
  DynamicArray<float> const values = Values(*encoder);
  ASSERT_EQ(values.size(), 2 * kFr3Joints.size());
  for (std::size_t i = 0; i < kFr3Joints.size(); ++i) {
    EXPECT_NEAR(values[i], _defaultPose[i], 1e-5F) << kFr3Joints[i];
  }
}

// The encoder's positions and velocities are bitwise the arm's, read at each joint's DOF and cast
// to float, while the arm moves.
TEST_F(SensorJointEncoderTest, ReadsTheArmsPositionsAndVelocities) {
  SensorJointEncoder* const encoder = CreateEncoder("", ExpectOK{});
  for (int64_t step = 1; step <= 20; ++step) {
    _scene->Step(0.005_r);
    _graph.Step({.t = 0.005 * step, .dt = 0.005, .step = step}, ExpectOK{});
  }

  DynamicArray<float> const expected = JointState(*_arm, {kFr3Joints.data(), kFr3Joints.size()});
  EXPECT_EQ(Values(*encoder), expected);
  EXPECT_TRUE(std::any_of(expected.begin() + kFr3Joints.size(), expected.end(), [](float v) {
    return v != 0.0F;
  })) << "the arm should move under gravity";
}

// Listed joints read in the listed order with their position offsets, which SetParams changes at
// runtime; velocities take no offset.
TEST_F(SensorJointEncoderTest, ReadsListedJointsWithOffsets) {
  SensorJointEncoder* const encoder = CreateEncoder(
      R"({"joints": ["fr3_joint4", "fr3_joint2"], "offset": [0.1, -0.2]})", ExpectOK{});
  _graph.Step({.t = 0.01, .dt = 0.01, .step = 1}, ExpectOK{});
  DynamicArray<float> const offsetValues = Values(*encoder);

  SensorJointEncoder::Params params = encoder->GetParams();
  params.offset = {0.0_r, 0.0_r};
  encoder->SetParams(params, ExpectOK{});
  _graph.Step({.t = 0.02, .dt = 0.01, .step = 2}, ExpectOK{});

  std::array<std::string_view, 2> constexpr kListed = {"fr3_joint4", "fr3_joint2"};
  DynamicArray<float> const truth = JointState(*_arm, {kListed.data(), kListed.size()});
  EXPECT_EQ(
      encoder->GetFormat().axes[1].labels,
      (DynamicArray<DynamicString>(kListed.begin(), kListed.end())));
  EXPECT_EQ(Values(*encoder), truth);
  ASSERT_EQ(offsetValues.size(), 4);
  EXPECT_NEAR(offsetValues[0], truth[0] + 0.1F, 1e-5F);
  EXPECT_NEAR(offsetValues[1], truth[1] - 0.2F, 1e-5F);
  EXPECT_EQ(offsetValues[2], truth[2]);
  EXPECT_EQ(offsetValues[3], truth[3]);
  params.joints = {DynamicString("fr3_joint1"), DynamicString("fr3_joint2")};
  encoder->SetParams(params, ExpectNotOK{});
}

// Listed joints must be one-DOF joints of the bot, with one offset each; the log names a refused
// joint.
TEST_F(SensorJointEncoderTest, RefusesJointsItCannotRead) {
  auto const suppressError = SuppressLogError();
  for (std::string_view const params :
       {R"({"joints": ["missing"]})",
        R"({"joints": ["world_joint"]})",
        R"({"joints": ["fr3_joint8"]})",
        R"({"offset": [0.1, 0.2]})"}) {
    CreateEncoder(params, ExpectNotOK{});
    _graph.Clear();
  }
}

TEST_F(SensorJointEncoderTest, NeedsAnOwningBot) {
  SensorJointEncoder encoder(nullptr, SensorJointEncoder::Params{}, ExpectOK{});
  _graph.Add("encoder", &encoder, ExpectOK{});
  _graph.SetOutputEnabled("encoder", true, ExpectOK{});
  _graph.Build({.scene = _scene, .robotics = _robotics}, ExpectNotOK{});
  _graph.Clear();
}

} // namespace
