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
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/sensors/sensor_link_vertex_force.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

// A 100 Hz source on 200 Hz physics captures every other step all episode on a clock that sums its
// steps, as BotScene's does, in either precision.
TEST(SensorLinkVertexForceTest, SampleRateHoldsOnAnAccumulatedClock) {
  Context* const context = mochi::CreateContext(0);
  Scene* const scene = context->CreateScene("SensorLinkVertexForceTest");
  auto&& [coordinates, connectivity] = CreateMinimalTetMeshUnitCube();
  RigidActorParams params;
  params.shape = context->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), ExpectOK{});
  params.name = "link";
  params.density = 1000_r;
  Actor* const link = scene->CreateRigidActor(params, ExpectOK{});
  {
    SensorLinkVertexForce forces(
        link, SensorLinkVertexForce::Params{.sampleRateHz = 100_r}, ExpectOK{});
    SensorGraph graph;
    graph.Add("forces", &forces, ExpectOK{});
    graph.SetOutputEnabled("forces", true, ExpectOK{});
    graph.Build({.scene = scene}, ExpectOK{});
    graph.Reset(ExpectOK{});

    double constexpr kDt = 0.005;
    double t = 0.0;
    int64_t late = 0;
    for (int64_t step = 1; step <= 300; ++step) {
      scene->Step(static_cast<real>(kDt));
      t += kDt;
      graph.Step({.t = t, .dt = kDt, .step = step}, ExpectOK{});
      int64_t const expected = step % 2 == 1 ? step : step - 1;
      late += forces.Latest().captureStep != expected ? 1 : 0;
    }
    EXPECT_EQ(late, 0);
  }
  context->DestroyScene(scene);
  mochi::DestroyContext(context);
}

} // namespace
