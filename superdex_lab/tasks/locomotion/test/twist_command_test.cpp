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
#include <numbers>
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

// g1_command's record layout: command 0-2, phase 3-4, linear velocity 5-7, angular 8-10, pelvis
// rotation (x, y, z, w) 11-14. A step's record is read right after its physics, before the
// commands' update: it holds the command the step was taken under. A reset's holds the command
// after the reset's update (which advances no time).
constexpr int kCommand = 0;
constexpr int kPhase = 3;
constexpr int kLinear = 5;
constexpr int kAngular = 8;
constexpr int kRotation = 11;

/* twist_command's params, each a JSON value as text; the defaults are a fixed command of 0. */
struct TwistParams {
  std::string linVelX{"[0.0, 0.0]"};
  std::string linVelY{"[0.0, 0.0]"};
  std::string angVelZ{"[0.0, 0.0]"};
  std::string heading{"[-3.141592653589793, 3.141592653589793]"};
  std::string resamplingS{"[3.0, 8.0]"};
  std::string standingShare{"0.0"};
  std::string headingShare{"1.0"};
  std::string headingCommand{"false"};
  std::string headingStiffness{"1.0"};
  std::string minNorm{"0.1"};
  std::string phasePeriodS{"0.6"};
  std::string schedule;
};

/* `p` as twist_command's params on the pelvis. */
std::string Json(TwistParams const& p) {
  std::string json = R"({"base": "pelvis", "linVelX": )" + p.linVelX + R"(, "linVelY": )" +
      p.linVelY + R"(, "angVelZ": )" + p.angVelZ + R"(, "heading": )" + p.heading +
      R"(, "resamplingS": )" + p.resamplingS + R"(, "standingShare": )" + p.standingShare +
      R"(, "headingShare": )" + p.headingShare + R"(, "headingCommand": )" + p.headingCommand +
      R"(, "headingStiffness": )" + p.headingStiffness + R"(, "minNorm": )" + p.minNorm +
      R"(, "phasePeriodS": )" + p.phasePeriodS;
  if (!p.schedule.empty()) {
    json += R"(, "schedule": )" + p.schedule;
  }
  return json + "}";
}

/* g1_command with the twist command's params replaced by `params`, when given. */
LabConfig CommandConfig(std::string const& params = {}) {
  LabConfig config;
  std::string error;
  EXPECT_TRUE(LoadLabFile(
      test::Assets("tasks/g1_velocity_flat/test/g1_command.superdex_lab").string(), config, error))
      << error;
  if (!params.empty()) {
    config.commands.at(0).params = params;
  }
  return config;
}

class TwistCommandTest : public ::testing::Test {
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

 private:
  mochi::Context* _context = nullptr;
  std::unique_ptr<TaskEnv> _env;
};

/* Each episode's first command lies in mjlab G1's ranges, and is all zeros when its norm is at
   most 0.1; over 50 episodes the draws spread over the ranges. */
TEST_F(TwistCommandTest, DrawsWithinTheRanges) {
  TaskEnv& env = Make(CommandConfig());
  double lowX = 1e9, highX = -1e9;
  for (int episode = 0; episode < 50; ++episode) {
    env.Reset(episode, mochi::ErrorAssert{});
    std::vector<float> const r = test::Record(env);
    double const vx = r[kCommand], vy = r[kCommand + 1], wz = r[kCommand + 2];
    EXPECT_GE(vx, -1.0);
    EXPECT_LE(vx, 2.0);
    EXPECT_GE(vy, -1.0);
    EXPECT_LE(vy, 1.0);
    EXPECT_GE(wz, -1.0);
    EXPECT_LE(wz, 1.0);
    double const norm = std::hypot(vx, vy, wz);
    EXPECT_TRUE(norm > 0.1 || norm == 0.0) << norm;
    lowX = std::min(lowX, vx);
    highX = std::max(highX, vx);
  }
  EXPECT_LT(lowX, -0.5);
  EXPECT_GT(highX, 1.5);
}

/* A command holds for a period drawn from [3, 8] s, then a new one is drawn. */
TEST_F(TwistCommandTest, ResamplesAfterItsPeriodAndClocksThePhase) {
  TaskEnv& env = Make(CommandConfig());
  env.Reset(3, mochi::ErrorAssert{});
  std::vector<float> const first = test::Record(env);
  double const dt = env.Spec().controlDt;
  int changedAt = -1;
  for (int step = 1; step <= 450 && changedAt < 0; ++step) {
    Step();
    std::vector<float> const r = test::Record(env);
    if (r[kCommand] != first[kCommand] || r[kCommand + 1] != first[kCommand + 1]) {
      changedAt = step;
    }
  }
  ASSERT_GT(changedAt, 0) << "no new command in 9 s";
  EXPECT_GE(changedAt * dt, 3.0 - 1e-6);
  EXPECT_LE(changedAt * dt, 8.0 + 2.0 * dt + 1e-6);
}

/* The phase after k decisions is (sin, cos) of 2 pi (k dt mod 0.6) / 0.6. */
TEST_F(TwistCommandTest, PhaseFollowsTheEpisodeClock) {
  TaskEnv& env =
      Make(CommandConfig(Json({.linVelX = "[1.0, 1.0]", .resamplingS = "[100.0, 100.0]"})));
  env.Reset(0, mochi::ErrorAssert{});
  double const dt = env.Spec().controlDt;
  for (int k = 1; k <= 40; ++k) {
    Step();
    std::vector<float> const r = test::Record(env);
    double const cycle = std::fmod((k - 1) * dt, 0.6) / 0.6; // the phase step k was taken under
    EXPECT_NEAR(r[kPhase], std::sin(2.0 * std::numbers::pi * cycle), 1e-5) << "k " << k;
    EXPECT_NEAR(r[kPhase + 1], std::cos(2.0 * std::numbers::pi * cycle), 1e-5) << "k " << k;
  }
}

/* A standing env's command is zero from the reset's update on, and so is its phase. */
TEST_F(TwistCommandTest, StandingEnvsCommandZero) {
  TaskEnv& env = Make(CommandConfig(Json(
      {.linVelX = "[1.0, 1.0]",
       .linVelY = "[0.5, 0.5]",
       .angVelZ = "[0.5, 0.5]",
       .standingShare = "1.0"})));
  env.Reset(0, mochi::ErrorAssert{});
  std::vector<float> const r = test::Record(env);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(r[kCommand + i], 0.0f) << i;
  }
}

/* A heading env's wz is 0.5 times the heading error, clipped to angVelZ: with the heading target
   fixed at 1 rad, wz is 0.5 (1 - h), h the yaw of the pelvis's x axis when the update ran. */
TEST_F(TwistCommandTest, HeadingControlSetsTheYawRate) {
  TaskEnv& env = Make(CommandConfig(Json(
      {.linVelX = "[0.5, 0.5]",
       .angVelZ = "[-1.0, 1.0]",
       .heading = "[1.0, 1.0]",
       .headingCommand = "true",
       .headingStiffness = "0.5"})));
  env.Reset(0, mochi::ErrorAssert{});
  Step();
  std::vector<float> const record = test::Record(env);
  std::vector<float> const q(record.begin() + kRotation, record.end());
  Step(); // this step's record holds the command the first step's update set
  double const heading =
      std::atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
  EXPECT_NEAR(test::Record(env)[kCommand + 2], 0.5 * (1.0 - heading), 1e-4);
}

/* With a schedule, the command is the last row whose time has come: stand, then 0.5 m/s forward
   from 0.1 s, then a 0.5 rad/s turn from 0.2 s, with no draws, heading control or standing. */
TEST_F(TwistCommandTest, FollowsASchedule) {
  TaskEnv& env = Make(CommandConfig(Json(
      {.standingShare = "1.0",
       .headingCommand = "true",
       .schedule = "[[0.0, 0.0, 0.0, 0.0], [0.1, 0.5, 0.0, 0.0], [0.2, 0.0, 0.0, 0.5]]"})));
  env.Reset(0, mochi::ErrorAssert{});
  double const dt = env.Spec().controlDt;
  for (int k = 1; k <= 15; ++k) {
    Step();
    // The step's record holds the command the update after k - 1 decisions set.
    double const t = (k - 1) * dt;
    std::vector<float> const r = test::Record(env);
    EXPECT_FLOAT_EQ(r[kCommand], t >= 0.1 - 1e-9 && t < 0.2 - 1e-9 ? 0.5f : 0.0f) << "k " << k;
    EXPECT_FLOAT_EQ(r[kCommand + 1], 0.0f);
    EXPECT_FLOAT_EQ(r[kCommand + 2], t >= 0.2 - 1e-9 ? 0.5f : 0.0f) << "k " << k;
  }
}

/* The two tracking rewards are mjlab's kernels of the step's base velocities and the command the
   step was taken under (terms run before the command's update, as mjlab's rewards before its
   command compute), both of which the step's record holds: reward = weight x value x dt. Over
   10 s, commands resample in between. */
TEST_F(TwistCommandTest, TrackingRewardsAreMjlabsKernels) {
  TaskEnv& env = Make(CommandConfig());
  env.Reset(1, mochi::ErrorAssert{});
  double const dt = env.Spec().controlDt;
  ASSERT_EQ(env.Spec().rewards.size(), 2u);
  for (int step = 0; step < 500 && !env.Done(); ++step) {
    Step();
    std::vector<float> const r = test::Record(env);
    std::vector<float> rewards(2), metrics(env.Spec().metrics.size()),
        curricula(env.Spec().curricula.size());
    env.WriteTerms(
        {rewards.data(), rewards.size()},
        {metrics.data(), metrics.size()},
        {curricula.data(), curricula.size()},
        mochi::ErrorAssert{});
    auto const sq = [](double x) { return x * x; };
    double const lin = sq(r[kCommand] - r[kLinear]) + sq(r[kCommand + 1] - r[kLinear + 1]) +
        2.0 * sq(r[kLinear + 2]);
    double const ang =
        sq(r[kCommand + 2] - r[kAngular + 2]) + 0.05 * (sq(r[kAngular]) + sq(r[kAngular + 1]));
    EXPECT_NEAR(rewards[0], std::exp(-lin / 0.25) * dt, 1e-5) << "step " << step;
    EXPECT_NEAR(rewards[1], std::exp(-ang / 0.5) * dt, 1e-5) << "step " << step;
  }
}

/* Each of twist_command's constants is required: a lab that leaves out phasePeriodS is refused,
   naming the key. */
TEST(TwistCommandParamsTest, AMissingConstantRefusesTheLab) {
  std::string const full = Json({});
  std::string const key = R"(, "phasePeriodS": 0.6)";
  std::string const missing =
      full.substr(0, full.find(key)) + full.substr(full.find(key) + key.size());
  std::string error;
  mochi::LogFn const previous = mochi::GetLogCallback();
  mochi::SetLogCallback([&error](mochi::LogChannel channel, char const* message, char const*, int) {
    if (channel == mochi::LogChannel::Error) {
      error += message;
    }
  });
  mochi::Context* const context = mochi::CreateContext(0);
  mochi::Error status;
  std::unique_ptr<TaskEnv> env = TaskEnv::Create(context, CommandConfig(missing), Kinds(), status);
  mochi::SetLogCallback(previous);
  EXPECT_EQ(env, nullptr);
  EXPECT_NE(error.find("params: missing key 'phasePeriodS'"), std::string::npos) << error;
  env.reset();
  mochi::DestroyContext(context);
}

} // namespace
