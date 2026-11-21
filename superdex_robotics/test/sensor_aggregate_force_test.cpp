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

#include <superdex_robotics/sensors/sensor_aggregate_force.h>
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/sensors/sensor_link_vertex_force.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <array>
#include <numbers>

using namespace mochi;
using namespace mochi::test;
using namespace superdex::robotics;

namespace {

class SensorAggregateForceTest : public testing::Test {
 protected:
  void SetUp() override {
    context = CreateContext(0);
    scene = context->CreateScene("AggregateForce");
    auto [points, tets] = CreateMinimalTetMeshUnitCube(Real3{0.02_r, 0.02_r, 0.02_r});
    RigidActorParams params;
    params.name = "pad";
    params.shape =
        context->CreateTetMeshShape(Flatten(MakeSpan(points)), Flatten(MakeSpan(tets)), ExpectOK{});
    params.colliderType = ColliderType::Box;
    params.hasGravity = false;
    params.density = 1e9_r;
    params.contact.penaltyCoefficient = 1e8_r;
    params.contact.penaltyThresholdDefault = 0.005_r;
    pad = scene->CreateRigidActor(params, ExpectOK{});
    auto [loadPoints, loadTets] = CreateMinimalTetMeshUnitCube(Real3{0.01_r, 0.01_r, 0.01_r});
    params.name = "load";
    params.shape = context->CreateTetMeshShape(
        Flatten(MakeSpan(loadPoints)), Flatten(MakeSpan(loadTets)), ExpectOK{});
    params.hasGravity = true;
    params.density = 1e5_r;
    params.worldFromLocal = TransformRT{Real3{0.005_r, 0.0201_r, 0.005_r}};
    scene->CreateRigidActor(params, ExpectOK{});
    scene->SetGravity(Real3{0_r, -10_r, 0_r});
  }

  void TearDown() override {
    context->DestroyScene(scene);
    DestroyContext(context);
  }

  Context* context = nullptr;
  Scene* scene = nullptr;
  Actor* pad = nullptr;
};

} // namespace

TEST_F(SensorAggregateForceTest, ReducesBorrowedVerticesInTheSensorFrame) {
  TransformRT const frame(Quaternion::RotationX(-std::numbers::pi_v<real> / 2_r));
  SensorLinkVertexForce vertices(pad, SensorLinkVertexForce::Params{}, ExpectOK{});
  vertices.SetParentFromSensor(frame);
  SensorAggregateForce composed(SensorAggregateForce::Params{}, ExpectOK{});
  SensorGraph graph;
  graph.Add("vertices", &vertices, ExpectOK{});
  graph.Add("composed", &composed, DynamicArray<DynamicString>{"vertices"}, ExpectOK{});
  graph.SetOutputEnabled("composed", true, ExpectOK{});
  graph.Build({.scene = scene}, ExpectOK{});
  graph.Reset(ExpectOK{});
  for (int64_t step = 1; step <= 1000; ++step) {
    scene->Step(0.001_r);
    graph.Step({.t = 0.001 * static_cast<double>(step), .dt = 0.001, .step = step}, ExpectOK{});
  }
  auto const output = composed.Latest();
  ASSERT_EQ(3, output.Floats().size());
  EXPECT_LT(output.Floats()[2], -0.1F);
  std::array<double, 3> expected{};
  auto const input = vertices.Latest();
  for (size_t i = 0; i < input.Floats().size(); ++i) {
    expected[i % 3] += input.Floats()[i];
  }
  for (int component = 0; component < 3; ++component) {
    EXPECT_NEAR(expected[component], output.Floats()[component], 1e-6);
  }
  EXPECT_EQ(input.captureStep, output.captureStep);
  EXPECT_DOUBLE_EQ(input.captureTime, output.captureTime);
  EXPECT_EQ(
      SensorStepResult::NoUpdate, composed.Step({.t = 1, .dt = 0.001, .step = 1000}, ExpectOK{}));
}

TEST_F(SensorAggregateForceTest, MountedActorDoesNotReplaceAnInput) {
  SensorAggregateForce sensor(pad, "", ExpectOK{});
  SensorGraph graph;
  graph.Add("force", &sensor, ExpectOK{});
  graph.SetOutputEnabled("force", true, ExpectOK{});
  graph.Build({.scene = scene}, ExpectNotOK{});
  EXPECT_TRUE(pad->GetNodeContactForcesWorld(ExpectNotOK{}).empty());
}

TEST(SensorAggregateForceStandaloneTest, RequiresAnInput) {
  SensorAggregateForce sensor(SensorAggregateForce::Params{}, ExpectOK{});
  SensorGraph graph;
  graph.Add("force", &sensor, ExpectOK{});
  graph.SetOutputEnabled("force", true, ExpectOK{});
  graph.Build({}, ExpectNotOK{});
}
