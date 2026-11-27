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

#include <superdex_lab/controllers/lab_joint_pd.h>
#include <superdex_lab/lab_kinds.h>
#include <superdex_lab/task_env.h>
#include <superdex_lab/task_env_config.h>
#include <superdex_lab/tasks/locomotion.h>

#include "test_helpers.h"

#include <mochi_core/utils/error.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace superdex::lab;

namespace {

LabKinds const& Kinds() {
  static LabKinds const kinds = [] {
    LabKinds all;
    RegisterCoreKinds(all);
    RegisterLocomotionKinds(all);
    all.Register(&RegisterLabControllers);
    return all;
  }();
  return kinds;
}

// g1_push's record layout: pelvis linear velocity 0-2 and angular 3-5 (pelvis axes), its rotation
// (x, y, z, w) 6-9, the last push 10-15.
constexpr int kLinear = 0;
constexpr int kAngular = 3;
constexpr int kRotation = 6;
constexpr int kPush = 10;

/* `v` (pelvis axes) in world axes. */
std::array<double, 3> ToWorld(std::vector<float> const& r, int at) {
  double const x = r[kRotation], y = r[kRotation + 1], z = r[kRotation + 2], w = r[kRotation + 3];
  double const m[3][3] = {
      {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
      {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
      {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
  std::array<double, 3> out{};
  for (int i = 0; i < 3; ++i) {
    out[i] = m[i][0] * r[at] + m[i][1] * r[at + 1] + m[i][2] * r[at + 2];
  }
  return out;
}

/* g1_push (the G1 in the air, turned 90° about z, pushed after its first step) with the push's
   angular range set to `angular` (world axes) and its linear range to `linear`. */
std::vector<float> PushedOnce(
    std::array<double, 3> const& linear,
    std::array<double, 3> const& angular) {
  LabConfig config;
  std::string error;
  EXPECT_TRUE(LoadLabFile(
      test::Assets("tasks/g1_velocity_flat/test/g1_push.superdex_lab").string(), config, error))
      << error;
  auto const range = [](std::array<double, 3> const& v) {
    std::string out = "[";
    for (int i = 0; i < 3; ++i) {
      out += (i ? ", [" : "[") + std::to_string(v[i]) + ", " + std::to_string(v[i]) + "]";
    }
    return out + "]";
  };
  config.events.at(2).params = R"({"intervalS": [0.02, 0.02], "linear": )" + range(linear) +
      R"(, "angular": )" + range(angular) + "}";
  mochi::Context* const context = mochi::CreateContext(0);
  std::vector<float> before, after;
  {
    std::unique_ptr<TaskEnv> env = TaskEnv::Create(context, config, Kinds(), mochi::ErrorAssert{});
    env->Reset(0, mochi::ErrorAssert{});
    std::vector<float> const zero(env->Spec().action.Dim(), 0.0f);
    env->Step({zero.data(), zero.size()}, mochi::ErrorAssert{}); // the push follows this step
    before = test::Record(*env);
    env->Step({zero.data(), zero.size()}, mochi::ErrorAssert{});
    after = test::Record(*env);
  }
  mochi::DestroyContext(context);
  std::vector<float> both = before;
  both.insert(both.end(), after.begin(), after.end());
  return both;
}

/* A linear push adds its world velocity to the floating pelvis: one step later (in the air, so
   only gravity acts on it), the pelvis's world velocity is the push plus g dt down. */
TEST(PushRobot, AddsWorldLinearVelocity) {
  std::vector<float> const both = PushedOnce({2.0, -1.0, 0.5}, {0.0, 0.0, 0.0});
  std::vector<float> const before(both.begin(), both.begin() + 16);
  std::vector<float> const after(both.begin() + 16, both.end());
  std::array<double, 3> const v0 = ToWorld(before, kLinear);
  std::array<double, 3> const v1 = ToWorld(after, kLinear);
  EXPECT_NEAR(v1[0] - v0[0], 2.0, 0.02);
  EXPECT_NEAR(v1[1] - v0[1], -1.0, 0.02);
  EXPECT_NEAR(v1[2] - v0[2], 0.5 - 9.81 * 0.02, 0.02);
  EXPECT_FLOAT_EQ(after[kPush], 2.0f);
}

/* An angular push about world x turns the pelvis, turned 90° about z, about its own -y axis: the
   push is in world axes. */
TEST(PushRobot, AddsWorldAngularVelocity) {
  std::vector<float> const both = PushedOnce({0.0, 0.0, 0.0}, {0.5, 0.0, 0.0});
  std::vector<float> const before(both.begin(), both.begin() + 16);
  std::vector<float> const after(both.begin() + 16, both.end());
  EXPECT_NEAR(after[kAngular] - before[kAngular], 0.0, 0.05);
  EXPECT_NEAR(after[kAngular + 1] - before[kAngular + 1], -0.5, 0.05);
  EXPECT_NEAR(after[kAngular + 2] - before[kAngular + 2], 0.0, 0.05);
}

/* After the root has yawed since its placement (spun about z from 0.6 s, a quarter turn by the
   push at 1.0 s), a world push still adds its world velocity: the push is turned into the free
   joint's frame (the placement), not the root link's current axes. */
TEST(PushRobot, AddsWorldVelocityAfterTheRootYawed) {
  LabConfig config;
  std::string error;
  ASSERT_TRUE(LoadLabFile(
      test::Assets("tasks/g1_velocity_flat/test/g1_push_yawed.superdex_lab").string(),
      config,
      error))
      << error;
  mochi::Context* const context = mochi::CreateContext(0);
  {
    std::unique_ptr<TaskEnv> env = TaskEnv::Create(context, config, Kinds(), mochi::ErrorAssert{});
    env->Reset(0, mochi::ErrorAssert{});
    std::vector<float> const zero(env->Spec().action.Dim(), 0.0f);
    auto const step = [&] { env->Step({zero.data(), zero.size()}, mochi::ErrorAssert{}); };
    for (int i = 0; i < 50; ++i) {
      step(); // the push follows the 50th step (1.0 s)
    }
    std::vector<float> const before = test::Record(*env);
    step();
    std::vector<float> const after = test::Record(*env);
    double const x = before[kRotation], y = before[kRotation + 1], z = before[kRotation + 2],
                 w = before[kRotation + 3];
    double const yaw = std::atan2(2.0 * (x * y + z * w), 1.0 - 2.0 * (y * y + z * z));
    // Placed at +90° and spun about a quarter turn more: facing about -x.
    EXPECT_GT(std::abs(yaw), 2.5) << "the root should have yawed since its placement";
    std::array<double, 3> const v0 = ToWorld(before, kLinear);
    std::array<double, 3> const v1 = ToWorld(after, kLinear);
    EXPECT_NEAR(v1[0] - v0[0], 2.0, 0.05);
    EXPECT_NEAR(v1[1] - v0[1], -1.0, 0.05);
    EXPECT_NEAR(v1[2] - v0[2], -9.81 * 0.02, 0.05);
  }
  mochi::DestroyContext(context);
}

/* The whole robot's angular momentum (`links:*.angular_momentum`) is 0 at rest and, in the air
   after a push about world z, holds still while the robot spins, gravity having no moment about
   its center of mass: about I_zz x 1 rad/s about z (with x and y parts, the bent posture's
   products of inertia). */
TEST(PushRobot, AngularMomentumIsConservedInTheAir) {
  LabConfig config;
  std::string error;
  ASSERT_TRUE(LoadLabFile(
      test::Assets("tasks/g1_velocity_flat/test/g1_push.superdex_lab").string(), config, error))
      << error;
  config.events.at(2).params =
      R"({"intervalS": [0.02, 0.02], "linear": [[0, 0], [0, 0], [0, 0]],
          "angular": [[0, 0], [0, 0], [1.0, 1.0]]})";
  config.events.at(2).name = "spin";
  config.record = {"links:*.angular_momentum"};
  mochi::Context* const context = mochi::CreateContext(0);
  {
    std::unique_ptr<TaskEnv> env = TaskEnv::Create(context, config, Kinds(), mochi::ErrorAssert{});
    env->Reset(0, mochi::ErrorAssert{});
    std::vector<float> const zero(env->Spec().action.Dim(), 0.0f);
    auto const step = [&] { env->Step({zero.data(), zero.size()}, mochi::ErrorAssert{}); };
    step(); // the first push follows this step
    std::vector<float> const rest = test::Record(*env);
    EXPECT_LT(std::hypot(rest[0], rest[1], rest[2]), 1e-3);
    // The push repeats every decision, each adding the same I ω: in the air the momentum holds
    // between pushes, so two pushes carry twice one.
    step();
    std::vector<float> const one = test::Record(*env);
    step();
    std::vector<float> const two = test::Record(*env);
    EXPECT_GT(one[2], 0.3f) << "a z spin of 1 rad/s carries I_zz (kg m^2)";
    EXPECT_NEAR(two[2], 2.0f * one[2], 0.02f * one[2]);
  }
  mochi::DestroyContext(context);
}

} // namespace
