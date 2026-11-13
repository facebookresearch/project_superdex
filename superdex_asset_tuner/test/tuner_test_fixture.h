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

#pragma once

#include "superdex_asset_tuner/scene_targets.h"
#include "superdex_asset_tuner/tuner_runner.h"
#include "superdex_asset_tuner/tuner_spec.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace superdex::tuner::test {

// Unit cube spanning [0, 1]^3 in its local frame, so its COM is offset from the actor origin.
inline constexpr char kCubeShape[] = "cube/cube_fine_mesh.mochi.h5";

// A 0.1 m, 1 kg rigid cube (unit cube scaled by 0.1 at density 1000).
inline constexpr char kRigidCubePrefab[] = R"({
  "actors": { "rigid": [ {
    "name": "cube", "shape": "cube/cube_fine_mesh.mochi.h5", "scale": [0.1, 0.1, 0.1],
    "colliderType": "Box", "density": 1000 } ] }
})";

// A single free-floating link with the same geometry as kRigidCubePrefab.
inline constexpr char kFreeLinkPrefab[] = R"({
  "actors": { "articulated": [ {
    "name": "arm",
    "joints": [ { "name": "root", "type": "Free" } ],
    "links": [ { "name": "body", "parentLink": -1, "shape": "cube/cube_fine_mesh.mochi.h5",
                 "shapeScale": [0.1, 0.1, 0.1], "colliderType": "Box", "density": 1000 } ] } ] }
})";

// A world-welded base with a 1 kg link on a frictionless x-axis prismatic joint.
inline constexpr char kSliderPrefab[] = R"({
  "actors": { "articulated": [ {
    "name": "slider",
    "joints": [
      { "name": "weld", "type": "Hard" },
      { "name": "slide", "type": "Prismatic", "axis": [1, 0, 0] } ],
    "links": [
      { "name": "base", "parentLink": -1, "shape": "cube/cube_fine_mesh.mochi.h5",
        "shapeScale": [0.1, 0.1, 0.1], "colliderType": "None", "density": 1000 },
      { "name": "carriage", "parentLink": 0, "shape": "cube/cube_fine_mesh.mochi.h5",
        "shapeScale": [0.1, 0.1, 0.1], "colliderType": "None", "density": 1000 } ] } ] }
})";

struct RunCapture {
  std::vector<std::string> channels;
  std::vector<Sample> samples;
  RunSummary summary;

  [[nodiscard]] real Value(std::string_view channel, size_t sampleIndex) const {
    for (size_t i = 0; i < channels.size(); ++i) {
      if (channels[i] == channel) {
        return samples.at(sampleIndex).values[i];
      }
    }
    ADD_FAILURE() << "No channel '" << channel << "'";
    return 0;
  }
  [[nodiscard]] real Final(std::string_view channel) const {
    return Value(channel, samples.size() - 1);
  }
};

class TunerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto const* info = ::testing::UnitTest::GetInstance()->current_test_info();
    // Unique per process: the float and double test binaries run the same tests concurrently.
    _dir = std::filesystem::temp_directory_path() / "superdex_asset_tuner_test" /
        (std::string(info->test_suite_name()) + "." + info->name() + "." +
         std::to_string(std::random_device{}()));
    std::filesystem::remove_all(_dir);
    std::filesystem::create_directories(_dir);
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(_dir, error);
  }

  [[nodiscard]] std::filesystem::path const& Dir() const {
    return _dir;
  }
  [[nodiscard]] std::filesystem::path TunerPath() const {
    return _dir / "test.superdex_tuner";
  }

  // A spec for `prefabJson` in a zero-gravity world without a ground plane.
  [[nodiscard]] TunerSpec MakeSpec(char const* prefabJson) const {
    std::ofstream(_dir / "scene.mochi_scene") << prefabJson;
    TunerSpec spec;
    spec.prefab = "./scene.mochi_scene";
    char const* assets = std::getenv("MOCHI_ASSETS_PATH");
    EXPECT_NE(assets, nullptr) << "MOCHI_ASSETS_PATH must point at the superdex assets";
    spec.assetsRoot = assets != nullptr ? std::filesystem::absolute(assets).string() : "";
    spec.run.gravity = Real3{0.0f, 0.0f, 0.0f};
    spec.run.groundPlane = false;
    spec.run.dt = 0.01;
    spec.run.duration = 1.0;
    return spec;
  }

  [[nodiscard]] RunCapture Run(TunerSpec const& spec, RunOptions options = {}) const {
    RunCapture capture;
    options.onSample = [&](Span<DynamicString const> names, Sample const& sample) {
      if (capture.channels.empty()) {
        for (auto const& name : names) {
          capture.channels.emplace_back(std::string_view(name));
        }
      }
      capture.samples.push_back(sample);
    };
    std::string error;
    EXPECT_TRUE(RunTuner(spec, TunerPath(), options, capture.summary, error)) << error;
    return capture;
  }

  [[nodiscard]] SceneTargets Describe(TunerSpec const& spec) const {
    SceneTargets targets;
    std::string error;
    EXPECT_TRUE(DescribeTuner(spec, TunerPath(), targets, error)) << error;
    return targets;
  }

 private:
  std::filesystem::path _dir;
};

[[nodiscard]] inline LoadSpec MakeLoad(
    std::string_view name,
    LoadType type,
    std::string_view target,
    Real3 const& direction,
    real magnitude) {
  LoadSpec load;
  load.name = DynamicString(name);
  load.type = type;
  load.target = DynamicString(target);
  load.direction = direction;
  load.profile.magnitude = magnitude;
  return load;
}

[[nodiscard]] inline ProbeSpec MakeProbe(ProbeKind kind, std::string_view target) {
  ProbeSpec probe;
  probe.kind = kind;
  probe.target = DynamicString(target);
  return probe;
}

// Displacement from rest under constant acceleration. Implicit integrators overshoot this by
// O(dt / t), so compare with a relative tolerance of a few percent.
[[nodiscard]] inline double ConstantAccelerationDisplacement(double acceleration, double t) {
  return 0.5 * acceleration * t * t;
}

} // namespace superdex::tuner::test
