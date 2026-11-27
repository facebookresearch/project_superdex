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

// g1.xml's torso_link center of mass, in the link's axes.
constexpr std::array<double, 3> kTorsoCom{0.00203158, 0.000339683, 0.184568};

/* At reset, the distance from the pelvis's origin to the torso's center of mass, and the one the
   torso's pose predicts for a center of mass at kTorsoCom + `offset` in its axes; with a startup
   body.com_offset of `offset` on the torso when `apply`. */
std::array<double, 2> TorsoComDistance(std::array<double, 3> const& offset, bool apply) {
  LabConfig config;
  std::string error;
  EXPECT_TRUE(LoadLabFile(
      test::Assets("tasks/g1_velocity_flat/test/g1_stand.superdex_lab").string(), config, error))
      << error;
  if (apply) {
    std::string const o = std::to_string(offset[0]) + ", " + std::to_string(offset[1]) + ", " +
        std::to_string(offset[2]);
    CommandSpec com;
    com.name = "torso_com";
    com.kind = "randomize_field";
    com.params = R"({"field": "body.com_offset", "target": "bot/g1/links/torso_link",
        "distribution": ["uniform_box", [)" +
        o + "], [" + o + R"(]], "operation": "add", "temporal": "startup"})";
    config.events.push_back(com);
  }
  config.record = {
      "pelvis.translation_in_task_root",
      "torso_link.translation_in_task_root",
      "torso_link.rotation_in_task_root.quat",
      "links:torso_link.center_of_mass.mean_distance_from_pelvis",
  };
  mochi::Context* const context = mochi::CreateContext(0);
  std::array<double, 2> out{};
  {
    std::unique_ptr<TaskEnv> env = TaskEnv::Create(context, config, Kinds(), mochi::ErrorAssert{});
    env->Reset(0, mochi::ErrorAssert{});
    std::vector<float> const r = test::Record(*env);
    double const x = r[6], y = r[7], z = r[8], w = r[9];
    double const m[3][3] = {
        {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
        {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
        {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
    double squared = 0.0;
    for (int i = 0; i < 3; ++i) {
      double c = r[3 + i] - r[i];
      for (int k = 0; k < 3; ++k) {
        c += m[i][k] * (kTorsoCom[k] + offset[k]);
      }
      squared += c * c;
    }
    out = {r[10], std::sqrt(squared)};
  }
  mochi::DestroyContext(context);
  return out;
}

/* body.com_offset moves the torso's center of mass by its offset in the torso's axes: the measured
   distance to it matches the one the torso's pose predicts, with the offset and, as a check of the
   prediction, without. */
TEST(ComOffset, MovesTheTorsoCenterOfMass) {
  std::array<double, 3> const offset{0.05, -0.03, 0.02};
  std::array<double, 2> const nominal = TorsoComDistance({0.0, 0.0, 0.0}, false);
  EXPECT_NEAR(nominal[0], nominal[1], 1e-4);
  std::array<double, 2> const moved = TorsoComDistance(offset, true);
  EXPECT_NEAR(moved[0], moved[1], 1e-4);
  EXPECT_GT(std::abs(moved[0] - nominal[0]), 0.01);
}

} // namespace
