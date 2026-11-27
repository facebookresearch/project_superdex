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

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace superdex::lab;

namespace {

constexpr double kWeight = 33.341142 * 9.81;

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

/* g1_feet: the statue G1 with a fixed command (0.5, 0, 0.2), foot_contact "feet" and the terms.
   With `dropM`, each reset raises the robot by that much, so it falls onto its feet. */
LabConfig FeetConfig(double dropM = 0.0) {
  LabConfig config;
  std::string error;
  EXPECT_TRUE(LoadLabFile(
      test::Assets("tasks/g1_velocity_flat/test/g1_feet.superdex_lab").string(), config, error))
      << error;
  if (dropM > 0.0) {
    CommandSpec drop;
    drop.name = "drop";
    drop.kind = "randomize_field";
    drop.params =
        R"({"field": "bot.root_position", "target": "bot/g1", "distribution": ["uniform", )" +
        std::to_string(dropM) + ", " + std::to_string(dropM) +
        R"(], "operation": "add", "axes": [2], "temporal": "reset"})";
    config.events.push_back(drop);
  }
  return config;
}

class FeetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    _context = mochi::CreateContext(0);
  }
  void TearDown() override {
    _env.reset();
    mochi::DestroyContext(_context);
  }
  TaskEnv& Make(LabConfig const& config) {
    _env = TaskEnv::Create(_context, config, Kinds(), mochi::ErrorAssert{});
    return *_env;
  }
  void Step() {
    std::vector<float> const zero(_env->Spec().action.Dim(), 0.0f);
    _env->Step({zero.data(), zero.size()}, mochi::ErrorAssert{});
  }
  /* Output "feet.<name>" as last written. */
  std::vector<float> Feet(std::string const& name) const {
    Outputs const& outputs = _env->ComponentOutputs();
    mochi::Span<float const> const values = outputs.Read(outputs.Find("feet." + name));
    return {values.begin(), values.end()};
  }
  std::vector<float> Rewards() const {
    std::vector<float> rewards(_env->Spec().rewards.size()), metrics(_env->Spec().metrics.size()),
        curricula(_env->Spec().curricula.size());
    _env->WriteTerms(
        {rewards.data(), rewards.size()},
        {metrics.data(), metrics.size()},
        {curricula.data(), curricula.size()},
        mochi::ErrorAssert{});
    return rewards;
  }

 private:
  mochi::Context* _context = nullptr;
  std::unique_ptr<TaskEnv> _env;
};

/* Standing, both feet are found from their landing (the robot starts a few mm up, so within the
   first steps) on: their contact time then grows one control step a step and their air time stays
   0, they land (first contact) once, the ground's force on them is the robot's weight, and the
   sites sit at the ground at rest. */
TEST_F(FeetTest, StandingFeetAreInContact) {
  TaskEnv& env = Make(FeetConfig());
  env.Reset(0, mochi::ErrorAssert{});
  double const dt = env.Spec().controlDt;
  int landed = -1;
  for (int step = 1; step <= 100; ++step) {
    Step();
    std::vector<float> const found = Feet("found");
    if (landed < 0 && found[0] == 1.0f && found[1] == 1.0f) {
      landed = step;
    }
    if (landed < 0) {
      continue;
    }
    std::vector<float> const force = Feet("force");
    std::vector<float> const contact = Feet("contact_time");
    std::vector<float> const air = Feet("air_time");
    std::vector<float> const first = Feet("first_contact");
    for (int foot = 0; foot < 2; ++foot) {
      EXPECT_EQ(found[foot], 1.0f) << "step " << step;
      EXPECT_NEAR(contact[foot], (step - landed + 1) * dt, 1e-5) << "step " << step;
      EXPECT_EQ(air[foot], 0.0f);
      EXPECT_EQ(first[foot], step == landed ? 1.0f : 0.0f) << "step " << step;
    }
    if (step > 50) {
      EXPECT_NEAR(force[0] + force[1], kWeight, 0.02 * kWeight) << "step " << step;
    }
  }
  ASSERT_GT(landed, 0);
  EXPECT_LE(landed, 3);
  std::vector<float> const height = Feet("height");
  std::vector<float> const speed = Feet("speed_xy");
  for (int foot = 0; foot < 2; ++foot) {
    // The site is 1 cm below the capsules' axis, at the spheres' bottom: about the ground.
    EXPECT_NEAR(height[foot], 0.0, 0.005);
    EXPECT_LT(speed[foot], 0.005);
  }
}

/* Dropped from 10 cm, the feet are in the air until they land, about sqrt(2 x 0.1 / 9.81) = 0.14 s
   later: air time grows meanwhile, the landing decision has first contact and records the air time
   it ended, and soft_landing then charges the feet's landing force. */
TEST_F(FeetTest, DroppedFeetLandOnce) {
  TaskEnv& env = Make(FeetConfig(0.1));
  env.Reset(0, mochi::ErrorAssert{});
  double const dt = env.Spec().controlDt;
  int landed = -1;
  for (int step = 1; step <= 25 && landed < 0; ++step) {
    Step();
    std::vector<float> const first = Feet("first_contact");
    if (first[0] == 1.0f || first[1] == 1.0f) {
      landed = step;
      std::vector<float> const lastAir = Feet("last_air_time");
      std::vector<float> const force = Feet("force");
      for (int foot = 0; foot < 2; ++foot) {
        if (first[foot] == 1.0f) {
          EXPECT_NEAR(lastAir[foot], (step - 1) * dt, 1e-5);
        }
      }
      // soft_landing: the sum of the landing feet's forces; reward = weight x value x dt.
      double const landing = first[0] * force[0] + first[1] * force[1];
      EXPECT_GT(landing, 0.0);
      EXPECT_NEAR(Rewards()[3], landing * dt, 1e-4 * std::max(1.0, landing));
    } else {
      std::vector<float> const air = Feet("air_time");
      EXPECT_NEAR(air[0], step * dt, 1e-5) << "step " << step;
      EXPECT_EQ(Feet("found")[0], 0.0f);
    }
  }
  ASSERT_GT(landed, 0) << "the feet never landed";
  EXPECT_NEAR(landed * dt, std::sqrt(2.0 * 0.1 / 9.81), 2.0 * dt);
}

/* The terms are mjlab's functions of the outputs, the command the step was taken under (its
   record) and the joints: rewards = weight (1) x value x dt. */
TEST_F(FeetTest, TermsAreMjlabsFunctions) {
  TaskEnv& env = Make(FeetConfig());
  env.Reset(0, mochi::ErrorAssert{});
  double const dt = env.Spec().controlDt;
  ASSERT_EQ(env.Spec().rewards.size(), 10u);
  for (int step = 1; step <= 60; ++step) {
    Step();
    std::vector<float> const r = test::Record(env); // command 0-2, joints 3-31, torso quat 32-35
    std::vector<float> const reward = Rewards();
    std::vector<float> const found = Feet("found");
    std::vector<float> const height = Feet("height");
    std::vector<float> const speed = Feet("speed_xy");
    double const command = std::hypot(r[0], r[1]) + std::abs(r[2]);
    ASSERT_NEAR(command, 0.7, 1e-5);
    // feet_gait: stance while (t / 0.6 + offset) mod 1 < 0.56.
    double matches = 0.0;
    for (int foot = 0; foot < 2; ++foot) {
      double const leg = std::fmod(step * dt / 0.6 + (foot == 0 ? 0.0 : 0.5), 1.0);
      matches += ((leg < 0.56) == (found[foot] > 0.0f)) ? 1.0 : 0.0;
    }
    EXPECT_NEAR(reward[0], matches / 2.0 * dt, 1e-6) << "step " << step;
    double clearance = 0.0, slip = 0.0;
    for (int foot = 0; foot < 2; ++foot) {
      clearance += std::abs(height[foot] - 0.1) * speed[foot];
      slip += found[foot] * speed[foot] * speed[foot];
    }
    EXPECT_NEAR(reward[1], clearance * dt, 1e-6);
    EXPECT_NEAR(reward[2], slip * dt, 1e-6);
    // variable_posture at walking speed (0.7 in [0.1, 1.5)): std 0.3 for every joint.
    double posture = 0.0;
    for (int j = 0; j < 29; ++j) {
      posture += (r[3 + j] / 0.3) * (r[3 + j] / 0.3);
    }
    EXPECT_NEAR(reward[4], std::exp(-posture / 29.0) * dt, 1e-6);
    EXPECT_EQ(reward[5], 0.0f) << "stand_still is off while the command moves";
    EXPECT_EQ(reward[6], 0.0f) << "no failure";
    double const x = r[32], y = r[33], z = r[34], w = r[35];
    double const gx = 2 * (x * z - y * w), gy = 2 * (y * z + x * w);
    EXPECT_NEAR(reward[7], (gx * gx + gy * gy) * dt, 1e-6);
    EXPECT_GE(reward[8], 0.0f);
    EXPECT_GE(reward[9], 0.0f);
    if (step == 1) {
      EXPECT_EQ(reward[9], 0.0f) << "joint_acc_l2 starts after a decision";
    }
  }
}

} // namespace
