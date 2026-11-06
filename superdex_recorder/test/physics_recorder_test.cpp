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

#include <superdex_recorder/physics_recorder.h>

#include <superdex_physics.h>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace superdex::recorder {
namespace {

constexpr std::array<real, 24> kCoordinates = {
    -0.1_r, -0.1_r, -0.1_r, +0.1_r, -0.1_r, -0.1_r, -0.1_r, +0.1_r, -0.1_r, +0.1_r, +0.1_r, -0.1_r,
    -0.1_r, -0.1_r, +0.1_r, +0.1_r, -0.1_r, +0.1_r, -0.1_r, +0.1_r, +0.1_r, +0.1_r, +0.1_r, +0.1_r,
};

constexpr std::array<int, 36> kConnectivity = {
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 5, 0, 5, 4,
    2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
};

constexpr std::array<real, 12> kTetCoordinates =
    {0_r, 0_r, 0_r, 0.1_r, 0_r, 0_r, 0_r, 0.1_r, 0_r, 0_r, 0_r, 0.1_r};

constexpr std::array<int, 4> kTetConnectivity = {0, 1, 2, 3};

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    auto const suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
        ("superdex_recorder_physics_test_" + std::to_string(suffix) + "_" +
         std::to_string(std::random_device{}()));
    std::filesystem::create_directories(_path);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }

  [[nodiscard]] std::filesystem::path const& Path() const {
    return _path;
  }

 private:
  std::filesystem::path _path;
};

class PhysicsRecorderTest : public testing::Test {
 protected:
  void SetUp() override {
    context = CreateContext(0);
    ASSERT_NE(context, nullptr);
    scene = context->CreateScene("recorder physics test");
    ASSERT_NE(scene, nullptr);
  }

  void TearDown() override {
    if (scene != nullptr) {
      for (Actor* actor : actors) {
        scene->DestroyActor(actor);
      }
      context->DestroyScene(scene);
    }
    if (context != nullptr) {
      DestroyContext(context);
    }
  }

  Actor* CreateRigidActor(std::string_view name, ShapeHandle shape, bool isStatic = false) {
    RigidActorParams params;
    params.name = name;
    params.shape = std::move(shape);
    params.density = 1000_r;
    params.isStatic = isStatic;
    Actor* const actor = scene->CreateRigidActor(params, error);
    if (actor != nullptr) {
      actors.push_back(actor);
    }
    return actor;
  }

  Actor* CreateSoftActor(std::string_view name) {
    SoftActorParams params;
    params.name = name;
    params.shape = context->CreateTetMeshShape(kTetCoordinates, kTetConnectivity, error);
    Actor* const actor = scene->CreateSoftActor(params, error);
    if (actor != nullptr) {
      actors.push_back(actor);
    }
    return actor;
  }

  Context* context = nullptr;
  Scene* scene = nullptr;
  std::vector<Actor*> actors;
  Error error;
};

TEST_F(PhysicsRecorderTest, CommitsWholeSceneWithoutExplicitActorRegistration) {
  ShapeHandle const shape = context->CreateTriMeshShape(kCoordinates, kConnectivity, error);
  Actor* const cube = CreateRigidActor("cube", shape);
  ShapeHandle const plane = context->CreatePlaneShape({0_r, 0_r, 1_r}, 0_r, error);
  Actor* const ground = CreateRigidActor("ground", plane, true);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(cube, nullptr);
  ASSERT_NE(ground, nullptr);

  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  PhysicsRecorder physics{*episode, *scene, error};
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  physics.Record({.frame = 0, .simulationTimeSec = 0.0}, error);
  scene->Step(0.01);
  physics.Record({.frame = 1, .simulationTimeSec = scene->GetTotalSimulationTime()}, error);
  episode->Commit(error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(std::filesystem::exists(episode->GetOutputPath()));
}

TEST_F(PhysicsRecorderTest, RecordFailsAfterRecordedActorIsDestroyed) {
  ShapeHandle const shape = context->CreateTriMeshShape(kCoordinates, kConnectivity, error);
  Actor* const cube = CreateRigidActor("cube", shape);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(cube, nullptr);

  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  PhysicsRecorder physics{*episode, *scene, {.recordContactForces = true}, error};
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  scene->DestroyActor(cube);
  std::erase(actors, cube);
  physics.Record({.frame = 0, .simulationTimeSec = 0.0}, error);

  EXPECT_FALSE(error.IsOK());
}

TEST_F(PhysicsRecorderTest, RecordsSoftActors) {
  ShapeHandle const shape = context->CreateTriMeshShape(kCoordinates, kConnectivity, error);
  ASSERT_NE(CreateRigidActor("cube", shape), nullptr);
  ASSERT_NE(CreateSoftActor("soft"), nullptr);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_NE(episode, nullptr);
  PhysicsRecorder physics{*episode, *scene, error};
  physics.Record({.frame = 0, .simulationTimeSec = 0.0}, error);
  scene->Step(0.01);
  physics.Record({.frame = 1, .simulationTimeSec = scene->GetTotalSimulationTime()}, error);
  episode->Commit(error);

  EXPECT_TRUE(error.IsOK()) << error.ToString();
}

TEST_F(PhysicsRecorderTest, ContactRecordingRejectsActorsOfOtherTypes) {
  ShapeHandle const shape = context->CreateTriMeshShape(kCoordinates, kConnectivity, error);
  ASSERT_NE(CreateRigidActor("cube", shape), nullptr);
  ASSERT_NE(CreateSoftActor("soft"), nullptr);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  PhysicsRecorder physics{*episode, *scene, {.recordContactForces = true}, error};

  EXPECT_NE(
      error.ToString().find("Contact recording supports only rigid and articulated actors."),
      std::string::npos)
      << error.ToString();
}

} // namespace
} // namespace superdex::recorder
