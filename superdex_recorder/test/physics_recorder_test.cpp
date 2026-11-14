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

#include <superdex_recorder/episode_reader.h>
#include <superdex_recorder/physics_recorder.h>

#include <mochi_physics/mochi_physics_experimental.h>
#include <superdex_physics.h>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <random>
#include <set>
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

  // A 30 cm rod along x with a tubular contact skin.
  Actor* CreateRodActor(std::string_view name, Real3 const& position) {
    std::array<Real3, 7> nodes{};
    for (size_t i = 0; i < nodes.size(); ++i) {
      nodes[i] = {0.05_r * static_cast<real>(i), 0_r, 0_r};
    }
    ModelData const model =
        experimental::GenerateTubularRodModelData(nodes, {}, 0.01_r, 8, false, error);
    RodActorParams params;
    params.name = name;
    params.shape = context->CreateModelShape(model, error);
    params.worldFromLocal = TransformRT{Quaternion{}, position};
    params.colliderType = ColliderType::PointCloud;
    params.useContactSkin = true;
    Actor* const actor = scene->CreateRodActor(params, error);
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

TEST_F(PhysicsRecorderTest, RecordsContactsWithRodActors) {
  ShapeHandle const plane = context->CreatePlaneShape({0_r, 1_r, 0_r}, 0_r, error);
  ASSERT_NE(CreateRigidActor("ground", plane, true), nullptr);
  // Touching the ground with its contact skin.
  ASSERT_NE(CreateRodActor("rod", {0_r, 0.01_r, 0_r}), nullptr);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_NE(episode, nullptr);
  PhysicsRecorder physics{*episode, *scene, {.recordContactForces = true}, error};
  scene->Step(0.01);
  physics.Record({.frame = 1, .simulationTimeSec = scene->GetTotalSimulationTime()}, error);
  episode->Commit(error);
  auto const recording = EpisodeReader::Open(episode->GetOutputPath(), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  EXPECT_EQ(
      recording->ReadStaticTexts(
          "data/physics/contacts", "superdex.physics.contact_body_paths", error),
      (std::vector<std::string>{"world/ground", "world/rod"}));
  ScalarRows const bodyA = recording->ReadScalars(
      "data/physics/contacts", "superdex.physics.contact_body_a_indices", error);
  ScalarRows const bodyB = recording->ReadScalars(
      "data/physics/contacts", "superdex.physics.contact_body_b_indices", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(bodyA.GetNumRows(), 1);
  size_t const numContacts = bodyA.GetRow(0).size();
  EXPECT_GT(numContacts, 0);
  // Every contact is between the ground (0) and the rod (1).
  EXPECT_EQ(
      std::vector<double>(bodyA.GetRow(0).begin(), bodyA.GetRow(0).end()),
      std::vector<double>(numContacts, 0.0));
  EXPECT_EQ(
      std::vector<double>(bodyB.GetRow(0).begin(), bodyB.GetRow(0).end()),
      std::vector<double>(numContacts, 1.0));
}

TEST_F(PhysicsRecorderTest, RecordsEachRodSelfContactOnce) {
  // A rod folded into a hairpin, whose legs press into each other.
  constexpr int kNumNodes = 21;
  constexpr int kTurn = kNumNodes / 2;
  constexpr real kSegment = 0.02_r;
  constexpr real kRadius = 0.005_r;
  std::array<Real3, kNumNodes> nodes{};
  std::array<real, 3 * kNumNodes> displacements{};
  for (int i = 0; i < kNumNodes; ++i) {
    nodes[i] = {kSegment * static_cast<real>(i), 0_r, 0_r};
    Real3 const folded = i <= kTurn
        ? nodes[i]
        : Real3{kSegment * (static_cast<real>(kNumNodes - 1 - i) + 0.5_r), 1.8_r * kRadius, 0_r};
    for (int k = 0; k < 3; ++k) {
      displacements[3 * i + k] = folded[k] - nodes[i][k];
    }
  }
  std::array<real, 3 * (kNumNodes - 1)> axes{};
  for (int element = 0; element + 1 < kNumNodes; ++element) {
    axes[3 * element + 2] = 1_r;
  }
  RodActorParams params;
  params.name = "rod";
  params.shape = context->CreateModelShape(
      experimental::GenerateTubularRodModelData(nodes, {}, kRadius, 8, false, error), error);
  params.colliderType = ColliderType::PointCloud;
  params.pointCloudCollider.radius = kRadius;
  params.pointCloudCollider.selfContact = true;
  params.useContactSkin = true;
  Actor* const rod = scene->CreateRodActor(params, error);
  ASSERT_NE(rod, nullptr);
  actors.push_back(rod);
  rod->SetRodPose(displacements, axes, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_NE(episode, nullptr);
  PhysicsRecorder physics{*episode, *scene, {.recordContactForces = true}, error};
  scene->Step(1e-3_r);
  physics.Record({.frame = 1, .simulationTimeSec = scene->GetTotalSimulationTime()}, error);
  size_t const numMochiContacts = rod->GetContactPointsWorld(error).size();
  episode->Commit(error);
  auto const recording = EpisodeReader::Open(episode->GetOutputPath(), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto const read = [&](char const* field) {
    ScalarRows const rows = recording->ReadScalars("data/physics/contacts", field, error);
    EXPECT_EQ(rows.GetNumRows(), 1);
    return std::vector<double>(rows.GetRow(0).begin(), rows.GetRow(0).end());
  };
  std::vector<double> const bodyA = read("superdex.physics.contact_body_a_indices");
  std::vector<double> const bodyB = read("superdex.physics.contact_body_b_indices");
  std::vector<double> const positionA = read("superdex.physics.contact_position_a_world_xyz");
  std::vector<double> const positionB = read("superdex.physics.contact_position_b_world_xyz");
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  EXPECT_GT(bodyA.size(), 0);
  EXPECT_EQ(bodyA.size(), numMochiContacts);
  EXPECT_EQ(bodyA, std::vector<double>(bodyA.size(), 0.0));
  EXPECT_EQ(bodyB, std::vector<double>(bodyA.size(), 0.0));
  // Each self-contact is recorded once.
  ASSERT_EQ(positionA.size(), 3 * bodyA.size());
  ASSERT_EQ(positionB.size(), 3 * bodyA.size());
  std::set<std::array<double, 6>> distinct;
  for (size_t i = 0; i < bodyA.size(); ++i) {
    distinct.insert(
        {positionA[3 * i],
         positionA[3 * i + 1],
         positionA[3 * i + 2],
         positionB[3 * i],
         positionB[3 * i + 1],
         positionB[3 * i + 2]});
  }
  EXPECT_EQ(distinct.size(), bodyA.size());
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
      error.ToString().find("Contact recording supports only rigid, articulated and rod actors."),
      std::string::npos)
      << error.ToString();
}

} // namespace
} // namespace superdex::recorder
