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
#include <superdex_recorder/physics_recorder.h>

#include <mochi_physics/mochi_physics_experimental.h>
#include <superdex_physics.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <numbers>
#include <random>
#include <string>
#include <vector>

namespace superdex::recorder {
namespace {

constexpr std::array<real, 24> kCubeCoordinates = {
    -0.1_r, -0.1_r, -0.1_r, +0.1_r, -0.1_r, -0.1_r, -0.1_r, +0.1_r, -0.1_r, +0.1_r, +0.1_r, -0.1_r,
    -0.1_r, -0.1_r, +0.1_r, +0.1_r, -0.1_r, +0.1_r, -0.1_r, +0.1_r, +0.1_r, +0.1_r, +0.1_r, +0.1_r,
};

constexpr std::array<int, 36> kCubeConnectivity = {
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 5, 0, 5, 4,
    2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
};

TEST(EpisodeRestorerTest, RestoresRecordedFramesFromAnEpisodeFile) {
  Error error;
  Context* const context = CreateContext(0);
  // A spinning cube in flight.
  auto const createCube = [&](Scene& scene) {
    RigidActorParams params;
    params.name = "cube";
    params.shape = context->CreateTriMeshShape(kCubeCoordinates, kCubeConnectivity, error);
    params.density = 1000_r;
    Actor* const cube = scene.CreateRigidActor(params, error);
    cube->SetVelocity({0.5_r, 0_r, 0_r}, {0_r, 0.5_r, 2_r}, error);
    return cube;
  };
  Scene* const live = context->CreateScene("live");
  Actor* const liveCube = createCube(*live);
  auto const directory = std::filesystem::temp_directory_path() /
      ("superdex_episode_restorer_test_" + std::to_string(std::random_device{}()));
  auto episode = Recorder{{.outputDirectory = directory}}.BeginEpisode({}, error);
  std::vector<Real3> recordedTranslations;
  {
    PhysicsRecorder physics{*episode, *live, error};
    for (uint64_t frame = 0; frame < 5; ++frame) {
      if (frame > 0) {
        live->Step(0.01_r);
      }
      physics.Record({.frame = frame, .simulationTimeSec = live->GetTotalSimulationTime()}, error);
      recordedTranslations.push_back(liveCube->GetRootTransform().GetTranslation());
    }
  }
  episode->CommitSync(error);
  auto const recording = EpisodeReader::Open(episode->GetOutputPath(), error);
  Scene* const restored = context->CreateScene("restored");
  Actor* const restoredCube = createCube(*restored);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  auto const restorer = EpisodeRestorer::Create(*restored, *recording, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(restorer->GetFrames(), (std::vector<int64_t>{0, 1, 2, 3, 4}));
  // Out of order, so that each restored state comes from the recording.
  for (size_t const frameIndex : {4, 1, 3}) {
    restorer->RestoreFrame(frameIndex, error);
    Real3 const translation = restoredCube->GetRootTransform().GetTranslation();
    for (size_t axis = 0; axis < 3; ++axis) {
      EXPECT_NEAR(translation[axis], recordedTranslations[frameIndex][axis], 1e-6_r);
    }
  }
  EXPECT_TRUE(error.IsOK()) << error.ToString();
  context->DestroyScene(live);
  context->DestroyScene(restored);
  DestroyContext(context);
  std::filesystem::remove_all(directory);
}

// Records an actor in flight for five frames, restores frames out of order into a second scene, and
// expects its root position and `getState` to match the recording at each restored frame and after
// the next step. `createActor` creates the actor with its initial velocities.
void ExpectRestoresActorInFlight(
    std::function<Actor*(Context&, Scene&, Error&)> const& createActor,
    std::function<std::vector<real>(Actor const&, Error&)> const& getState) {
  Error error;
  Context* const context = CreateContext(0);
  Scene* const live = context->CreateScene("live");
  Actor* const liveActor = createActor(*context, *live, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto const directory = std::filesystem::temp_directory_path() /
      ("superdex_episode_restorer_test_" + std::to_string(std::random_device{}()));
  auto episode = Recorder{{.outputDirectory = directory}}.BeginEpisode({}, error);
  auto const getFullState = [&](Actor const& actor) {
    Real3 const root = actor.GetRootTransform().GetTranslation();
    std::vector<real> state{root[0], root[1], root[2]};
    std::vector<real> const actorState = getState(actor, error);
    state.insert(state.end(), actorState.begin(), actorState.end());
    return state;
  };
  std::vector<std::vector<real>> recordedStates;
  {
    PhysicsRecorder physics{*episode, *live, error};
    for (uint64_t frame = 0; frame < 5; ++frame) {
      if (frame > 0) {
        live->Step(0.01_r);
      }
      physics.Record({.frame = frame, .simulationTimeSec = live->GetTotalSimulationTime()}, error);
      recordedStates.push_back(getFullState(*liveActor));
    }
  }
  episode->CommitSync(error);
  auto const recording = EpisodeReader::Open(episode->GetOutputPath(), error);
  Scene* const restored = context->CreateScene("restored");
  Actor* const restoredActor = createActor(*context, *restored, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  auto const restorer = EpisodeRestorer::Create(*restored, *recording, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto const expectStateAt = [&](size_t frameIndex) {
    std::vector<real> const state = getFullState(*restoredActor);
    ASSERT_EQ(state.size(), recordedStates[frameIndex].size());
    for (size_t i = 0; i < state.size(); ++i) {
      EXPECT_NEAR(state[i], recordedStates[frameIndex][i], 1e-6_r);
    }
  };
  // Out of order, so that each restored state comes from the recording.
  for (size_t const frameIndex : {4, 1, 3}) {
    restorer->RestoreFrame(frameIndex, error);
    expectStateAt(frameIndex);
  }
  // The restored velocities carry the next step.
  restored->Step(0.01_r);
  expectStateAt(4);
  EXPECT_TRUE(error.IsOK()) << error.ToString();
  context->DestroyScene(live);
  context->DestroyScene(restored);
  DestroyContext(context);
  std::filesystem::remove_all(directory);
}

TEST(EpisodeRestorerTest, RestoresSoftActorsFromAnEpisodeFile) {
  // A tetrahedron in flight, deforming from one moving corner.
  ExpectRestoresActorInFlight(
      [](Context& context, Scene& scene, Error& error) {
        constexpr std::array<real, 12> kCoordinates = {
            0_r, 0_r, 0_r, 0.1_r, 0_r, 0_r, 0_r, 0.1_r, 0_r, 0_r, 0_r, 0.1_r};
        constexpr std::array<int, 4> kConnectivity = {0, 1, 2, 3};
        constexpr std::array<real, 12> kVelocities = {
            0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r};
        SoftActorParams params;
        params.name = "tetrahedron";
        params.shape = context.CreateTetMeshShape(kCoordinates, kConnectivity, error);
        Actor* const tetrahedron = scene.CreateSoftActor(params, error);
        if (tetrahedron != nullptr) {
          tetrahedron->SetNodeVelocitiesLocal(kVelocities, error);
        }
        return tetrahedron;
      },
      [](Actor const& tetrahedron, Error& error) {
        auto const displacements = tetrahedron.GetDisplacements(error);
        return std::vector<real>(displacements.begin(), displacements.end());
      });
}

// A rod in flight, bending from one moving node and twisting: a straight one, or a ring.
void ExpectRestoresRodInFlight(bool isClosedLoop) {
  constexpr size_t kNumNodes = 8;
  ExpectRestoresActorInFlight(
      [isClosedLoop](Context& context, Scene& scene, Error& error) {
        std::array<Real3, kNumNodes> nodes{};
        for (size_t i = 0; i < nodes.size(); ++i) {
          real const angle = 2_r * std::numbers::pi_v<real> * static_cast<real>(i) / kNumNodes;
          nodes[i] = isClosedLoop ? Real3{0.05_r * std::cos(angle), 0.05_r * std::sin(angle), 0_r}
                                  : Real3{0.05_r * static_cast<real>(i), 0_r, 0_r};
        }
        RodActorParams params;
        params.name = "rod";
        params.shape = experimental::CreatePolylineShape(&context, nodes, {}, isClosedLoop, error);
        Actor* const rod = scene.CreateRodActor(params, error);
        if (rod != nullptr) {
          // Per node: velocity xyz, then the twist rate of the element that starts at it, which an
          // open rod's last node lacks.
          std::array<real, 4 * kNumNodes> velocities{};
          velocities[2] = 1_r;
          size_t const numElements = isClosedLoop ? kNumNodes : kNumNodes - 1;
          for (size_t element = 0; element < numElements; ++element) {
            velocities[4 * element + 3] = 2_r;
          }
          rod->SetNodeVelocitiesLocal(velocities, error);
        }
        return rod;
      },
      [](Actor const& rod, Error& error) {
        auto const mesh = rod.GetMesh();
        size_t const numDisplacements = mesh.coordinates.size();
        std::vector<real> pose(numDisplacements + 3 * static_cast<size_t>(mesh.GetNumElements()));
        rod.GetRodPose(
            Span<real>(pose.data(), numDisplacements),
            Span<real>(pose.data() + numDisplacements, pose.size() - numDisplacements),
            error);
        return pose;
      });
}

TEST(EpisodeRestorerTest, RestoresOpenRodActorsFromAnEpisodeFile) {
  ExpectRestoresRodInFlight(false);
}

TEST(EpisodeRestorerTest, RestoresClosedRodActorsFromAnEpisodeFile) {
  ExpectRestoresRodInFlight(true);
}

} // namespace
} // namespace superdex::recorder
