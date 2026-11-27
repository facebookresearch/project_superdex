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

#include "test_helpers.h"

#include <mochi_core/utils/error.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <numeric>
#include <string>
#include <vector>

using namespace superdex::lab;

namespace {

constexpr double kMass = 33.341142; // g1.xml's bodies
constexpr double kGravity = 9.81;
// Sphere radius and box half height: a level collider's bottom is this far below its center.
constexpr double kHalfHeight = 0.01;

LabKinds const& Kinds() {
  static LabKinds const kinds = [] {
    LabKinds all;
    RegisterCoreKinds(all);
    all.Register(&RegisterLabControllers);
    return all;
  }();
  return kinds;
}

struct Feet {
  char const* bot;
  char const* colliders; // the feet's collider links
  int count;
};

constexpr Feet kSphereFeet{.bot = "g1", .colliders = "*_foot?_s?", .count = 28};
constexpr Feet kBoxFeet{.bot = "g1_box_feet", .colliders = "*_foot_box", .count = 2};

/* One decision's state: the pelvis's position (world), tilt, speeds (pelvis axes), the feet's
   ground load and the deepest foot collider bottom below the ground. */
struct Sample {
  double t = 0.0;
  mochi::Real3 pelvis{};
  double tilt = 0.0; // degrees from upright
  double linSpeed = 0.0;
  double angSpeed = 0.0;
  double load = 0.0; // N
  double depth = 0.0; // m
};

/* Runs `lab` on `feet.bot` for `seconds` with zero actions (the home pose held), one sample per
   decision; an episode that ends early fails the test. */
std::vector<Sample> Stand(char const* lab, Feet const& feet, double seconds) {
  LabConfig config;
  std::string error;
  EXPECT_TRUE(LoadLabFile(
      test::Assets(std::string{"tasks/g1_velocity_flat/test/"} + lab).string(), config, error))
      << error;
  config.slots.at("robot").bot = feet.bot;
  std::string const links = std::string{"links:"} + feet.colliders;
  config.record = {
      "pelvis.translation_in_task_root",
      "pelvis.linear_velocity_from_base",
      "pelvis.angular_velocity_from_base",
      "gravity_from_base",
      links + ".contact_force.StaticPlane",
      links + ".translation_in_task_root",
  };
  mochi::Context* const context = mochi::CreateContext(0);
  std::vector<Sample> samples;
  {
    std::unique_ptr<TaskEnv> env = TaskEnv::Create(context, config, Kinds(), mochi::ErrorAssert{});
    env->Reset(0, mochi::ErrorAssert{});
    int const n = feet.count;
    int const force = 12;
    int const position = force + n;
    EXPECT_EQ(env->Spec().record.Dim(), position + 3 * n);
    std::vector<float> const zero(env->Spec().action.Dim(), 0.0f);
    int const steps = static_cast<int>(std::lround(seconds / env->Spec().controlDt));
    for (int step = 0; step < steps; ++step) {
      env->Step({zero.data(), zero.size()}, mochi::ErrorAssert{});
      if (env->Done()) {
        ADD_FAILURE() << "episode ended at step " << step << ": " << env->Result().failure << " "
                      << env->Result().nonFinite;
        break;
      }
      std::vector<float> const r = test::Record(*env);
      Sample s{.t = (step + 1) * env->Spec().controlDt, .pelvis = {r[0], r[1], r[2]}};
      double const g = std::hypot<double>(r[9], r[10], r[11]);
      s.tilt = std::acos(std::clamp(-r[11] / g, -1.0, 1.0)) * 180.0 / std::numbers::pi;
      s.linSpeed = std::hypot<double>(r[3], r[4], r[5]);
      s.angSpeed = std::hypot<double>(r[6], r[7], r[8]);
      s.load = std::accumulate(r.begin() + force, r.begin() + position, 0.0);
      for (int i = 0; i < n; ++i) {
        s.depth = std::max(s.depth, kHalfHeight - r[position + 3 * i + 2]);
      }
      samples.push_back(s);
    }
  }
  mochi::DestroyContext(context);
  return samples;
}

Sample At(std::vector<Sample> const& samples, double t) {
  auto const at = std::ranges::find_if(samples, [t](Sample const& s) { return s.t >= t - 1e-9; });
  EXPECT_NE(at, samples.end()) << "no sample at " << t << " s";
  return at == samples.end() ? Sample{} : *at;
}

double Rms(std::vector<double> const& values) {
  double const sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
  return values.empty() ? 0.0 : std::sqrt(sum / static_cast<double>(values.size()));
}

class G1Stand : public ::testing::TestWithParam<Feet> {};

/* mjlab's gains do not hold the G1 up: at the home pose its PD is softer than gravity's pull on the
   body, in MuJoCo as here. The zero-action fall follows MuJoCo's (tools/mujoco_stand.py: g1.xml
   with g1_constants.py's actuators, feet-only collision, friction 0.6, 200 Hz), whose pelvis x, z
   and tilt from upright were, at 0.2 s 0.019 m, 0.779 m, 2.6°; at 0.6 s 0.136, 0.758, 1.6°; at
   1.0 s 0.349, 0.675, 18.1°; at 1.2 s 0.539, 0.548, 42.6°. While both feet are flat (to 0.6 s)
   the ground carries the robot's weight. */
TEST_P(G1Stand, FallsAsInMujocoWithMjlabGains) {
  std::vector<Sample> const samples = Stand("g1_stand.superdex_lab", GetParam(), 1.2);
  struct Reference {
    double t, x, z, tilt, tiltTolerance;
  };
  for (Reference const& ref : {
           Reference{0.2, 0.019, 0.779, 2.6, 2.0},
           Reference{0.6, 0.136, 0.758, 1.6, 2.0},
           Reference{1.0, 0.349, 0.675, 18.1, 5.0},
           Reference{1.2, 0.539, 0.548, 42.6, 8.0},
       }) {
    Sample const s = At(samples, ref.t);
    std::printf(
        "[g1 fall %s] t %.1f pelvis x %.3f z %.3f tilt %.1f load %.1f (mujoco %.3f %.3f %.1f)\n",
        GetParam().bot,
        ref.t,
        s.pelvis[0],
        s.pelvis[2],
        s.tilt,
        s.load,
        ref.x,
        ref.z,
        ref.tilt);
    EXPECT_NEAR(s.pelvis[0], ref.x, 0.04) << "at " << ref.t << " s";
    EXPECT_NEAR(s.pelvis[2], ref.z, 0.025) << "at " << ref.t << " s";
    EXPECT_NEAR(s.tilt, ref.tilt, ref.tiltTolerance) << "at " << ref.t << " s";
  }
  for (double const t : {0.2, 0.3, 0.4, 0.5, 0.6}) {
    EXPECT_NEAR(At(samples, t).load, kMass * kGravity, 0.05 * kMass * kGravity) << "at " << t;
  }
}

/* With every actuator 30x stiffer and more damped (g1_stand_stiff), the G1 stands as a statue on
   its feet for 10 s: no fault or failure, the pelvis within 1 cm of where it settled (after 1 s)
   and within 2° of upright, no jitter, the ground carrying its weight, and the feet sinking under
   2 mm. */
TEST_P(G1Stand, StandsTenSecondsWithStiffJoints) {
  std::vector<Sample> const samples = Stand("g1_stand_stiff.superdex_lab", GetParam(), 10.0);
  ASSERT_FALSE(samples.empty());
  Sample const settled = At(samples, 1.0);
  double drift = 0.0;
  double tilt = 0.0;
  double depth = 0.0;
  std::vector<double> lin, ang, load;
  for (Sample const& s : samples) {
    if (s.t < 1.0) {
      continue;
    }
    drift = std::max<double>(
        drift,
        std::hypot(
            s.pelvis[0] - settled.pelvis[0],
            s.pelvis[1] - settled.pelvis[1],
            s.pelvis[2] - settled.pelvis[2]));
    tilt = std::max(tilt, s.tilt);
    depth = std::max(depth, s.depth);
    lin.push_back(s.linSpeed);
    ang.push_back(s.angSpeed);
    load.push_back(s.load);
  }
  double const weight = kMass * kGravity;
  double const meanLoad = std::accumulate(load.begin(), load.end(), 0.0) / load.size();
  double const loadSpread = *std::ranges::max_element(load) - *std::ranges::min_element(load);
  std::printf(
      "[g1 stand %s] %.1f s, pelvis z %.4f m, drift %.4f m, tilt %.2f deg, lin RMS %.5f m/s, ang "
      "RMS %.5f rad/s, load %.1f N (mg %.1f, spread %.1f), deepest %.2f mm\n",
      GetParam().bot,
      samples.back().t,
      settled.pelvis[2],
      drift,
      tilt,
      Rms(lin),
      Rms(ang),
      meanLoad,
      weight,
      loadSpread,
      depth * 1000.0);
  EXPECT_NEAR(samples.back().t, 10.0, 1e-6);
  EXPECT_LT(drift, 0.01);
  EXPECT_LT(tilt, 2.0);
  EXPECT_LT(Rms(lin), 0.005);
  EXPECT_LT(Rms(ang), 0.02);
  EXPECT_NEAR(meanLoad, weight, 0.02 * weight);
  EXPECT_LT(loadSpread, 0.1 * weight);
  EXPECT_LT(depth, 0.002);
}

/* A reset after a fall puts the free pelvis back where the scene places it, upright and at rest,
   and the joints back at the home pose: the floating base needs nothing of the slot beyond what a
   fixed base gets. */
TEST(G1Reset, RestoresTheFloatingBaseAfterAFall) {
  LabConfig config;
  std::string error;
  ASSERT_TRUE(LoadLabFile(
      test::Assets("tasks/g1_velocity_flat/test/g1_stand.superdex_lab").string(), config, error))
      << error;
  config.record = {
      "pelvis.translation_in_task_root",
      "pelvis.linear_velocity_from_base",
      "pelvis.angular_velocity_from_base",
      "gravity_from_base",
  };
  mochi::Context* const context = mochi::CreateContext(0);
  {
    std::unique_ptr<TaskEnv> env = TaskEnv::Create(context, config, Kinds(), mochi::ErrorAssert{});
    env->Reset(0, mochi::ErrorAssert{});
    std::vector<float> const start = test::Record(*env);
    std::vector<float> const startObs = test::Observe(*env);
    std::vector<float> const zero(env->Spec().action.Dim(), 0.0f);
    for (int step = 0; step < 100 && !env->Done(); ++step) {
      env->Step({zero.data(), zero.size()}, mochi::ErrorAssert{});
    }
    ASSERT_LT(test::Record(*env)[2], 0.3f) << "the G1 should have fallen in 2 s";
    env->Reset(1, mochi::ErrorAssert{});
    std::vector<float> const again = test::Record(*env);
    std::vector<float> const againObs = test::Observe(*env);
    for (int i = 0; i < 12; ++i) {
      EXPECT_NEAR(again[i], start[i], 1e-4) << "record value " << i;
    }
    EXPECT_NEAR(start[2], 0.8f, 1e-4);
    EXPECT_NEAR(start[11], -kGravity, 1e-4);
    for (std::size_t i = 0; i < againObs.size(); ++i) {
      EXPECT_NEAR(againObs[i], startObs[i], 1e-4) << "observation value " << i;
    }
  }
  mochi::DestroyContext(context);
}

INSTANTIATE_TEST_SUITE_P(
    Feet,
    G1Stand,
    ::testing::Values(kSphereFeet, kBoxFeet),
    [](::testing::TestParamInfo<Feet> const& info) { return std::string{info.param.bot}; });

} // namespace
