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
#include <superdex_recorder/episode_restorer.h>
#include <superdex_recorder/frame_bank.h>
#include <superdex_recorder/physics_recorder.h>
#include <superdex_recorder/series.h>

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace superdex::recorder {
namespace {

using mochi::operator""_r;
using mochi::real;

// Every recorded value is a short binary fraction and every rotation exactly unit, so float and
// double scenes hold the same values and restore the same states.
constexpr size_t kNumFrames = 6;
constexpr std::array<uint64_t, kNumFrames> kFrames = {0, 2, 4, 6, 8, 10};
constexpr double kStepSec = 0.125;
constexpr std::array<std::array<real, 4>, kNumFrames> kCubeRotations = {{
    {0_r, 0_r, 0_r, 1_r},
    {0.5_r, 0.5_r, 0.5_r, 0.5_r},
    {0_r, 1_r, 0_r, 0_r},
    {0.5_r, -0.5_r, 0.5_r, 0.5_r},
    {0_r, 0_r, -1_r, 0_r},
    {-0.5_r, 0.5_r, 0.5_r, 0.5_r},
}};
// The joint values of frame k are these plus k/8.
constexpr std::array<real, 2> kJointPositions = {0.25_r, -0.5_r};
constexpr std::array<real, 2> kJointVelocities = {1.5_r, -0.75_r};
constexpr std::array<real, 2> kJointTargets = {0.375_r, -0.125_r};
// Efforts are recorded at these frame indices only, and hold until the next.
constexpr std::array<size_t, 3> kEffortFrames = {2, 3, 5};
constexpr std::array<size_t, kNumFrames> kRestoreOrder = {3, 0, 5, 1, 4, 2};

// A box around (0.25, 0, 0): its center of mass is off its actor origin.
constexpr std::array<real, 24> kOffsetBox = {
    0.125_r,  -0.125_r, -0.125_r, 0.375_r,  -0.125_r, -0.125_r, 0.125_r, 0.125_r,
    -0.125_r, 0.375_r,  0.125_r,  -0.125_r, 0.125_r,  -0.125_r, 0.125_r, 0.375_r,
    -0.125_r, 0.125_r,  0.125_r,  0.125_r,  0.125_r,  0.375_r,  0.125_r, 0.125_r,
};
constexpr std::array<real, 24> kBox = {
    -0.125_r, -0.125_r, -0.125_r, 0.125_r,  -0.125_r, -0.125_r, -0.125_r, 0.125_r,
    -0.125_r, 0.125_r,  0.125_r,  -0.125_r, -0.125_r, -0.125_r, 0.125_r,  0.125_r,
    -0.125_r, 0.125_r,  -0.125_r, 0.125_r,  0.125_r,  0.125_r,  0.125_r,  0.125_r,
};
constexpr std::array<int, 36> kBoxConnectivity = {
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 5, 0, 5, 4,
    2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
};

mochi::Span<real const> AsSpan(std::array<real, 2> const& values) {
  return {values.data(), values.size()};
}

real FrameOffset(size_t frameIndex) {
  return static_cast<real>(frameIndex) * 0.125_r;
}

std::array<real, 2> Shifted(std::array<real, 2> values, size_t frameIndex) {
  for (real& value : values) {
    value += FrameOffset(frameIndex);
  }
  return values;
}

std::array<real, 2> Effort(size_t frameIndex) {
  return {2_r + FrameOffset(frameIndex), -4_r};
}

/* A static floor, a dynamic cube whose center of mass is off its origin, and a two-joint arm with
 * a pose controller. */
struct GoldenScene {
  static constexpr int kNumLinks = 3;

  GoldenScene(mochi::Context& context, char const* name, mochi::Error& error)
      : context{context}, scene{context.CreateScene(name)} {
    mochi::ShapeHandle const box = context.CreateTriMeshShape(kBox, kBoxConnectivity, error);
    mochi::RigidActorParams floorParams;
    floorParams.name = "floor";
    floorParams.shape = box;
    floorParams.isStatic = true;
    floorParams.worldFromLocal = mochi::TransformRT{mochi::Real3{0_r, 0_r, -2_r}};
    floor = scene->CreateRigidActor(floorParams, error);

    mochi::RigidActorParams cubeParams;
    cubeParams.name = "cube";
    cubeParams.shape = context.CreateTriMeshShape(kOffsetBox, kBoxConnectivity, error);
    cubeParams.density = 1000_r;
    cube = scene->CreateRigidActor(cubeParams, error);

    mochi::ArticulatedActorParams armParams;
    armParams.name = "arm";
    armParams.worldFromRoot = mochi::TransformRT{mochi::Real3{0_r, 2_r, 0_r}};
    armParams.joints = {
        {.name = "base", .type = mochi::ArticulatedJointType::Hard},
        {.name = "shoulder",
         .type = mochi::ArticulatedJointType::Revolute,
         .parentLinkFromJoint = mochi::TransformRT{mochi::Real3{0.5_r, 0_r, 0_r}},
         .axis = {0_r, 0_r, 1_r}},
        {.name = "elbow",
         .type = mochi::ArticulatedJointType::Revolute,
         .parentLinkFromJoint = mochi::TransformRT{mochi::Real3{0.5_r, 0_r, 0_r}},
         .axis = {0_r, 1_r, 0_r}},
    };
    for (int link = 0; link < kNumLinks; ++link) {
      armParams.links.push_back({
          .name = std::array{"link0", "link1", "link2"}[link],
          .parentLink = link - 1,
          .shape = box,
          .colliderType = mochi::ColliderType::Box,
          .density = 500_r,
      });
    }
    arm = scene->CreateArticulatedActor(armParams, error);
    if (arm != nullptr) {
      arm->AddArticulatedPoseController(mochi::PoseControllerParams{kNumLinks}, error);
    }
  }

  ~GoldenScene() {
    context.DestroyScene(scene);
  }

  mochi::Context& context;
  mochi::Scene* scene = nullptr;
  mochi::Actor* floor = nullptr;
  mochi::Actor* cube = nullptr;
  mochi::Actor* arm = nullptr;
};

/* Records each frame's state, set directly rather than simulated, and each effort frame's effort
 * the way a robot recorder does. */
std::filesystem::path RecordGoldenEpisode(
    GoldenScene& live,
    std::filesystem::path const& directory,
    mochi::Error& error) {
  auto episode = Recorder{{.outputDirectory = directory}}.BeginEpisode({}, error);
  if (episode == nullptr) {
    return {};
  }
  std::string const robotPath = "data/robotics/robots/arm";
  ComponentField const effortField{
      .archetype = "superdex.archetypes.RobotCommand",
      .component = "superdex.robotics.applied_joint_effort"};
  episode->WriteConstantText(
      robotPath,
      {.archetype = "superdex.archetypes.RobotCommand", .component = "superdex.robotics.robot_id"},
      "arm",
      error);
  ScalarSeries* const efforts =
      episode->CreateScalarSeries(robotPath + "/applied_command", {effortField}, error);
  {
    PhysicsRecorder physics{*episode, *live.scene, error};
    std::array<int, 2> const dofs = {0, 1};
    for (size_t i = 0; i < kNumFrames && error.IsOK(); ++i) {
      double const timeSec = static_cast<double>(kFrames[i]) * kStepSec * 0.5;
      mochi::experimental::SetSimulationTime(live.scene, timeSec, kStepSec, error);
      auto const& q = kCubeRotations[i];
      live.cube->SetCenterOfMassTransform(
          mochi::TransformRT{
              mochi::Quaternion{q[0], q[1], q[2], q[3]},
              mochi::Real3{FrameOffset(i), 1_r - FrameOffset(i), 0.5_r}},
          error);
      live.cube->SetVelocity(
          {0.5_r, FrameOffset(i), -1_r}, {FrameOffset(i), 0_r, 0.25_r - FrameOffset(i)}, error);
      auto const& r = kCubeRotations[kNumFrames - 1 - i];
      live.arm->SetRootTransform(
          mochi::TransformRT{
              mochi::Quaternion{r[0], r[1], r[2], r[3]}, mochi::Real3{0_r, 2_r, FrameOffset(i)}},
          error);
      auto const positions = Shifted(kJointPositions, i);
      auto const velocities = Shifted(kJointVelocities, i);
      auto const targets = Shifted(kJointTargets, i);
      live.arm->SetArticulatedPoseFromJoints(AsSpan(positions), error);
      live.arm->SetArticulatedJointVelocities(AsSpan(velocities), error);
      live.arm->SetArticulatedTargetPose(AsSpan(targets), error);
      SampleTime const time{.frame = kFrames[i], .simulationTimeSec = timeSec};
      if (std::find(kEffortFrames.begin(), kEffortFrames.end(), i) != kEffortFrames.end()) {
        auto const effort = Effort(i);
        live.arm->SetExternalForcesOnDofs(
            mochi::Span<int const>{dofs.data(), dofs.size()}, AsSpan(effort), error);
        std::array<mochi::Span<real const>, 1> const fields{AsSpan(effort)};
        efforts->Append(time, fields, error);
      }
      physics.Record(time, error);
    }
  }
  episode->CommitSync(error);
  return episode->GetOutputPath();
}

void AppendTransform(mochi::TransformRT const& transform, std::vector<double>& state) {
  for (int i = 0; i < 4; ++i) {
    state.push_back(transform.GetRotation().data[i]);
  }
  for (int i = 0; i < 3; ++i) {
    state.push_back(transform.GetTranslation()[i]);
  }
}

template <size_t N>
void AppendValues(std::array<real, N> const& values, std::vector<double>& state) {
  state.insert(state.end(), values.begin(), values.end());
}

/* The restored state the golden values hold: the clock, the cube's center of mass and
 * velocities, the arm's root pose, joint positions, velocities, targets and external forces, and
 * the floor's pose. */
std::vector<double> ReadState(GoldenScene& scene, mochi::Error& error) {
  std::vector<double> state{scene.scene->GetTotalSimulationTime(), scene.scene->GetLastTimeStep()};
  AppendTransform(scene.cube->GetCenterOfMassTransform(error), state);
  auto const linear = scene.cube->GetLinearVelocity(error);
  auto const angular = scene.cube->GetAngularVelocity(error);
  state.insert(state.end(), linear.begin(), linear.end());
  state.insert(state.end(), angular.begin(), angular.end());
  AppendTransform(scene.arm->GetRootTransform(), state);
  std::array<real, 2> values{};
  mochi::Span<real> const output{values.data(), values.size()};
  scene.arm->GetArticulatedPose(output, error);
  AppendValues(values, state);
  scene.arm->GetArticulatedJointVelocities(output, error);
  AppendValues(values, state);
  scene.arm->GetArticulatedTargetPose(output, error);
  AppendValues(values, state);
  scene.arm->GetExternalForces(output, error);
  AppendValues(values, state);
  AppendTransform(scene.floor->GetRootTransform(), state);
  return state;
}

constexpr size_t kStateSize = 37;

// Restored states in kRestoreOrder, captured from EpisodeRestorer before it restored through
// frame banks. Set SUPERDEX_RECORDER_PRINT_GOLDEN=1 to print the current ones.
constexpr std::array<std::array<double, kStateSize>, kNumFrames> kGolden = {{
    {0.375, 0.125,  0.5,   -0.5, 0.5, 0.5, 0.375, 0.625, 0.5,   0.5,   0.375,  -1,    0.375,
     0,     -0.125, 0,     1,    0,   0,   0,     2,     0.375, 0.625, -0.125, 1.875, -0.375,
     0.75,  0.25,   2.375, -4,   0,   0,   0,     1,     0,     0,     -2},
    {0,     0.125,  0,    0,   0,   1,   0, 1, 0.5, 0.5,  0,    -1,  0,
     0,     0.25,   -0.5, 0.5, 0.5, 0.5, 0, 2, 0,   0.25, -0.5, 1.5, -0.75,
     0.375, -0.125, 0,    0,   0,   0,   0, 1, 0,   0,    -2},
    {0.625, 0.125,  -0.5,  0.5, 0.5, 0.5, 0.625, 0.375, 0.5,   0.5,   0.625, -1,    0.625,
     0,     -0.375, 0,     0,   0,   1,   0,     2,     0.625, 0.875, 0.125, 2.125, -0.125,
     1,     0.5,    2.625, -4,  0,   0,   0,     1,     0,     0,     -2},
    {0.125, 0.125, 0.5, 0.5, 0.5, 0.5, 0.125, 0.875, 0.5,   0.5,   0.125,  -1,    0.125,
     0,     0.125, 0,   0,   -1,  0,   0,     2,     0.125, 0.375, -0.375, 1.625, -0.625,
     0.5,   0,     0,   0,   0,   0,   0,     1,     0,     0,     -2},
    {0.5,   0.125, 0,     0,   -1,  0,   0.5, 0.5, 0.5, 0.5,  0.5, -1, 0.5,
     0,     -0.25, 0.5,   0.5, 0.5, 0.5, 0,   2,   0.5, 0.75, 0,   2,  -0.25,
     0.875, 0.375, 2.375, -4,  0,   0,   0,   1,   0,   0,    -2},
    {0.25,  0.125, 0,    1,    0,   0,   0.25, 0.75, 0.5,  0.5, 0.25,  -1,   0.25,
     0,     0,     0.5,  -0.5, 0.5, 0.5, 0,    2,    0.25, 0.5, -0.25, 1.75, -0.5,
     0.625, 0.125, 2.25, -4,   0,   0,   0,    1,    0,    0,   -2},
}};

FrameBank ReadBank(EpisodeReader const& recording, mochi::Error& error) {
  FrameBank bank;
  bank.episodes.push_back(ReadBankEpisode(recording, error));
  return bank;
}

class EpisodeRestorerGoldenTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NE(context, nullptr);
    path = RecordGoldenEpisode(live, directory, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_EQ(live.arm->GetNumDofs(), 2);
  }

  void TearDown() override {
    std::filesystem::remove_all(directory);
  }

  /* Restores each frame in kRestoreOrder and checks its state against the golden values. */
  void ExpectGoldenStates(EpisodeRestorer& restorer, GoldenScene& scene) {
    bool const print = std::getenv("SUPERDEX_RECORDER_PRINT_GOLDEN") != nullptr;
    for (size_t i = 0; i < kNumFrames; ++i) {
      restorer.RestoreFrame(kRestoreOrder[i], error);
      std::vector<double> const state = ReadState(scene, error);
      ASSERT_TRUE(error.IsOK()) << error.ToString();
      ASSERT_EQ(state.size(), kStateSize);
      if (print) {
        std::printf("    {");
        for (double const value : state) {
          std::printf("%.17g, ", value);
        }
        std::printf("},\n");
        continue;
      }
      for (size_t j = 0; j < kStateSize; ++j) {
        EXPECT_EQ(state[j], kGolden[i][j]) << "restore " << i << ", value " << j;
      }
    }
  }

  std::unique_ptr<mochi::Context, decltype(&mochi::DestroyContext)> context{
      mochi::CreateContext(0),
      &mochi::DestroyContext};
  mochi::Error error;
  GoldenScene live{*context, "live", error};
  std::filesystem::path const directory = std::filesystem::temp_directory_path() /
      ("superdex_episode_restorer_golden_test_" + std::to_string(std::random_device{}()));
  std::filesystem::path path;
};

TEST_F(EpisodeRestorerGoldenTest, RestoresTheGoldenStatesFromAnEpisodeFile) {
  auto const recording = EpisodeReader::Open(path, error);
  GoldenScene restored{*context, "restored", error};
  auto const restorer = EpisodeRestorer::Create(*restored.scene, *recording, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(restorer->GetFrames(), (std::vector<int64_t>{kFrames.begin(), kFrames.end()}));

  ExpectGoldenStates(*restorer, restored);
}

TEST_F(EpisodeRestorerGoldenTest, RestoresTheGoldenStatesFromASavedBank) {
  auto const recording = EpisodeReader::Open(path, error);
  SaveFrameBank(ReadBank(*recording, error), directory / "golden.superdex_framebank", error);
  auto const bank = LoadFrameBank(directory / "golden.superdex_framebank", error);
  GoldenScene restored{*context, "restored", error};
  auto const restorer = EpisodeRestorer::Create(*restored.scene, bank, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(restorer->GetFrames(), (std::vector<int64_t>{kFrames.begin(), kFrames.end()}));
  EXPECT_EQ(restorer->GetActors().size(), 3);

  ExpectGoldenStates(*restorer, restored);
}

TEST_F(EpisodeRestorerGoldenTest, BankAndEpisodeRestoresStepTheSame) {
  auto const recording = EpisodeReader::Open(path, error);
  GoldenScene fromEpisode{*context, "from episode", error};
  GoldenScene fromBank{*context, "from bank", error};
  auto const episodeRestorer = EpisodeRestorer::Create(*fromEpisode.scene, *recording, error);
  auto const bankRestorer = EpisodeRestorer::Create(
      *fromBank.scene, std::make_shared<FrameBank const>(ReadBank(*recording, error)), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  for (size_t const frameIndex : kRestoreOrder) {
    episodeRestorer->RestoreFrame(frameIndex, error);
    bankRestorer->RestoreFrame(frameIndex, error);
    fromEpisode.scene->Step(0.0625_r);
    fromBank.scene->Step(0.0625_r);

    ASSERT_TRUE(error.IsOK()) << error.ToString();
    EXPECT_EQ(ReadState(fromEpisode, error), ReadState(fromBank, error)) << "frame " << frameIndex;
  }
}

} // namespace
} // namespace superdex::recorder
