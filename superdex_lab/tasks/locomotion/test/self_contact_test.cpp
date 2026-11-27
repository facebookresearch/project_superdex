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
#include <superdex_lab/tasks/lift.h>
#include <superdex_lab/tasks/locomotion.h>
#include <superdex_lab/tasks/reorient.h>

#include "test_helpers.h"

#include <mochi_core/utils/error.h>
#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <algorithm>
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
    RegisterLiftKinds(all); // reward_curriculum
    RegisterLocomotionKinds(all);
    RegisterReorientKinds(all); // squared_norm
    all.Register(&RegisterLabControllers);
    return all;
  }();
  return kinds;
}

constexpr char const* kSelf = "links:*.contact_force_other_than.StaticPlane";
// N: the zero-action fall's self contact peaks near mjlab's 10 N, so the test reads above 1 N.
constexpr float kThreshold = 1.0f;
constexpr double kPhysicsHz = 200.0; // g1_velocity.superdex_task

LabConfig G1Lab(char const* lab) {
  LabConfig config;
  std::string error;
  EXPECT_TRUE(LoadLabFile(
      test::Assets(std::string{"tasks/g1_velocity_flat/"} + lab).string(), config, error))
      << error;
  config.record = {
      std::string{"peak:"} + kSelf,
      "substeps_above:" + std::to_string(kThreshold) + ":" + kSelf,
  };
  return config;
}

/* Per decision of a zero-action episode on `lab` (the G1 falls): the largest self-contact force
   on any collider link over the decision's physics steps, and the substeps_above count. */
struct Fall {
  std::vector<float> largest;
  std::vector<float> count;
  int substeps = 0;
};

Fall FallOn(char const* lab) {
  LabConfig const config = G1Lab(lab);
  mochi::Context* const context = mochi::CreateContext(0);
  Fall fall;
  {
    std::unique_ptr<TaskEnv> env = TaskEnv::Create(context, config, Kinds(), mochi::ErrorAssert{});
    env->Reset(0, mochi::ErrorAssert{});
    fall.substeps = static_cast<int>(std::lround(env->Spec().controlDt * kPhysicsHz));
    std::vector<float> const zero(env->Spec().action.Dim(), 0.0f);
    for (int step = 0; step < 200 && !env->Done(); ++step) {
      env->Step({zero.data(), zero.size()}, mochi::ErrorAssert{});
      std::vector<float> const r = test::Record(*env);
      fall.largest.push_back(*std::max_element(r.begin(), r.end() - 1));
      fall.count.push_back(r.back());
    }
  }
  mochi::DestroyContext(context);
  return fall;
}

} // namespace

// On the full-collision G1 (FULL_COLLISION), the falling robot's links touch each other: some
// decision's self-contact force passes the threshold, and the substeps_above count is a whole
// number of the decision's physics steps, above 0 exactly when the per-link peak passed it.
TEST(SelfContactTest, TheFullCollisionG1TouchesItselfAndCountsSubsteps) {
  Fall const fall = FallOn("g1_velocity_flat_mjlab_full_collision.superdex_lab");

  ASSERT_FALSE(fall.count.empty());
  EXPECT_EQ(fall.substeps, 4);
  for (std::size_t i = 0; i < fall.count.size(); ++i) {
    float const count = fall.count[i];
    EXPECT_EQ(count, std::round(count)) << i;
    EXPECT_GE(count, 0.0f) << i;
    EXPECT_LE(count, static_cast<float>(fall.substeps)) << i;
    EXPECT_EQ(count > 0.0f, fall.largest[i] > kThreshold) << i;
  }
  EXPECT_GT(*std::max_element(fall.largest.begin(), fall.largest.end()), kThreshold);
}

// The feet-only G1 has no self contact: its scene disables the G1 layers against each other, so
// only the ground touches the feet.
TEST(SelfContactTest, TheFeetOnlyG1NeverTouchesItself) {
  Fall const fall = FallOn("g1_velocity_flat_mjlab.superdex_lab");

  ASSERT_FALSE(fall.count.empty());
  EXPECT_EQ(*std::max_element(fall.largest.begin(), fall.largest.end()), 0.0f);
  EXPECT_EQ(*std::max_element(fall.count.begin(), fall.count.end()), 0.0f);
}

TEST(SelfContactTest, SubstepsAboveTakesAFiniteThreshold) {
  LabConfig config = G1Lab("g1_velocity_flat_mjlab_full_collision.superdex_lab");
  config.record = {std::string{"substeps_above:ten:"} + kSelf};
  mochi::Context* const context = mochi::CreateContext(0);
  mochi::Error error;
  EXPECT_EQ(TaskEnv::Create(context, config, Kinds(), error), nullptr);
  EXPECT_FALSE(error.IsOK());
  mochi::DestroyContext(context);
}

TEST(SelfContactTest, ContactForceOtherThanRefusesAnActorNamedTwice) {
  LabConfig config = G1Lab("g1_velocity_flat_mjlab_full_collision.superdex_lab");
  config.record = {"links:*.contact_force_other_than.StaticPlane|StaticPlane"};
  mochi::Context* const context = mochi::CreateContext(0);
  mochi::Error error;
  EXPECT_EQ(TaskEnv::Create(context, config, Kinds(), error), nullptr);
  EXPECT_FALSE(error.IsOK());
  mochi::DestroyContext(context);
}
