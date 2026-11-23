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

#include <superdex_robotics/sensors/sensor_link_contact_force.h>

#include <superdex_robotics/sensors/interaction_wrench_metadata.h>
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/sensors/tactile_observation.h>
#include <superdex_robotics/superdex_robotics.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <numbers>

using namespace mochi;
using namespace mochi::test;
using namespace superdex::robotics;

namespace {

real constexpr kBoxSide = 0.01_r;
real constexpr kBoxDensity = 1e5_r;

class SensorLinkContactForceTest : public ::testing::Test {
 protected:
  // A heavy, gravity-free link with a box resting on its top face. Gravity is -y.
  void SetUp() override {
    context = CreateContext(0);
    scene = context->CreateScene("LinkContactForce");
    auto&& [linkNodes, linkTets] = CreateMinimalTetMeshUnitCube(Real3{0.02_r, 0.02_r, 0.02_r});
    RigidActorParams linkParams;
    linkParams.shape = context->CreateTetMeshShape(
        Flatten(MakeSpan(linkNodes)), Flatten(MakeSpan(linkTets)), ExpectOK{});
    linkParams.name = "link";
    linkParams.hasGravity = false;
    linkParams.density = 1e9_r;
    link = scene->CreateRigidActor(linkParams, ExpectOK{});
    auto&& [boxNodes, boxTets] = CreateMinimalTetMeshUnitCube(Real3{kBoxSide, kBoxSide, kBoxSide});
    RigidActorParams boxParams;
    boxParams.shape = context->CreateTetMeshShape(
        Flatten(MakeSpan(boxNodes)), Flatten(MakeSpan(boxTets)), ExpectOK{});
    boxParams.name = "box";
    boxParams.density = kBoxDensity;
    boxParams.worldFromLocal = TransformRT(Real3{0.005_r, 0.0201_r, 0.005_r});
    scene->CreateRigidActor(boxParams, ExpectOK{});
  }

  void TearDown() override {
    context->DestroyScene(scene);
    DestroyContext(context);
  }

  // Steps the scene and the graph together at 1 kHz.
  void Run(SensorGraph& graph, int64_t steps) {
    for (int64_t step = 1; step <= steps; ++step) {
      scene->Step(0.001_r);
      graph.Step({.t = 0.001 * static_cast<double>(step), .dt = 0.001, .step = step}, ExpectOK{});
    }
  }

  Context* context = nullptr;
  Scene* scene = nullptr;
  Actor* link = nullptr;
};

} // namespace

TEST_F(SensorLinkContactForceTest, ABoxOnTheLinkReadsItsWeightInTheSensorFrame) {
  // The sensor frame's z axis is world +y, so the weight pressing down reads as -z.
  SensorLinkContactForce sensor(link, SensorLinkContactForce::Params{}, ExpectOK{});
  sensor.SetParentFromSensor(TransformRT(Quaternion::RotationX(-std::numbers::pi_v<real> / 2_r)));
  SensorGraph graph;
  graph.Add("force", &sensor, ExpectOK{});
  graph.SetOutputEnabled("force", true, ExpectOK{});
  graph.Build({.scene = scene}, ExpectOK{});
  graph.Reset(ExpectOK{});
  Run(graph, 1000);

  EXPECT_EQ(kSensorFormatKindForce3D, sensor.GetFormat().kind);
  auto const* const interaction = TryGetSensorMetadata<InteractionWrenchMeta>(sensor);
  ASSERT_NE(nullptr, interaction);
  EXPECT_EQ(InteractionWrenchSense::EnvironmentOnSensor, interaction->sense);
  real const weight = kBoxDensity * kBoxSide * kBoxSide * kBoxSide * Norm(scene->GetGravity());
  Span<float const> const force = sensor.Latest().Floats();
  EXPECT_NEAR(0.0F, force[0], 0.02_r * weight);
  EXPECT_NEAR(0.0F, force[1], 0.02_r * weight);
  EXPECT_NEAR(-weight, force[2], 0.02_r * weight);
}

TEST_F(SensorLinkContactForceTest, SamplesOnItsSchedule) {
  SensorLinkContactForce sensor(link, {.sampleRateHz = 100_r}, ExpectOK{});
  SensorGraph graph;
  graph.Add("force", &sensor, ExpectOK{});
  graph.SetOutputEnabled("force", true, ExpectOK{});
  graph.Build({.scene = scene}, ExpectOK{});
  graph.Reset(ExpectOK{});
  Run(graph, 25);

  EXPECT_EQ(21, sensor.Latest().captureStep);
}

TEST_F(SensorLinkContactForceTest, RejectsANegativeRateAndAMissingLink) {
  SensorLinkContactForce negativeRate(link, {.sampleRateHz = -1_r}, ExpectNotOK{});
  SensorLinkContactForce noLink(nullptr, SensorLinkContactForce::Params{}, ExpectNotOK{});
}

TEST_F(SensorLinkContactForceTest, ConstructsThroughThePublicFactory) {
  RoboticsContext* const robotics = CreateRoboticsContext();
  MOCHI_DEFER(DestroyRoboticsContext(robotics));
  SensorHandle const handle = robotics->CreateSensor(
      SensorLinkContactForce::TypeName(), link, "force", R"({"sampleRateHz": 100})", ExpectOK{});
  ASSERT_TRUE(handle.IsValid());
  SensorBase* const sensor = robotics->GetSensor(handle);
  ASSERT_NE(nullptr, sensor);
  EXPECT_EQ(SensorLinkContactForce::TypeName(), sensor->GetTypeName());
  EXPECT_EQ(100.0, sensor->GetPublicationFrequencyHzEstimate());
  SensorGraph graph;
  graph.Add("force", handle, sensor, {}, ExpectOK{});
  graph.SetOutputEnabled("force", true, ExpectOK{});
  graph.Build({.scene = scene}, ExpectOK{});
  graph.Reset(ExpectOK{});
  Run(graph, 25);
  EXPECT_EQ(21, sensor->Latest().captureStep);
}

TEST_F(SensorLinkContactForceTest, RejectsFrameConversionAtAnIntermediateIntegrationStage) {
  SolverParams params = scene->GetSolverParams();
  params.integrationMethod = IntegrationMethod::SymplecticDIRK22;
  scene->SetSolverParams(params, ExpectOK{});
  SensorLinkContactForce sensor(link, SensorLinkContactForce::Params{}, ExpectOK{});
  SensorGraph graph;
  graph.Add("force", &sensor, ExpectOK{});
  graph.SetOutputEnabled("force", true, ExpectOK{});
  graph.Build({.scene = scene}, ExpectOK{});
  graph.Reset(ExpectOK{});
  graph.Step({.t = 0.001, .dt = 0.001, .step = 1}, ExpectNotOK{});
  EXPECT_EQ(-1, sensor.Latest().captureStep);
}
