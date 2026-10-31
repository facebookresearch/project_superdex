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

#include <superdex_physics.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
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
  episode->Commit(error);
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

} // namespace
} // namespace superdex::recorder
