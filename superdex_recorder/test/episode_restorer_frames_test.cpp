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

#include <superdex_recorder/episode_restorer.h>

#include <mochi_physics/mochi_physics.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace superdex::recorder {
namespace {

using mochi::operator""_r;

// A cube centered at (kComOffsetX, 0, 0): its center of mass is off its actor origin, so restoring
// the center of mass and restoring the actor origin to the same pose differ.
constexpr mochi::real kComOffsetX = 0.3_r;

constexpr std::array<mochi::real, 24> kCubeCoordinates = {
    0.2_r, -0.1_r, -0.1_r, 0.4_r, -0.1_r, -0.1_r, 0.2_r, +0.1_r, -0.1_r, 0.4_r, +0.1_r, -0.1_r,
    0.2_r, -0.1_r, +0.1_r, 0.4_r, -0.1_r, +0.1_r, 0.2_r, +0.1_r, +0.1_r, 0.4_r, +0.1_r, +0.1_r,
};

constexpr std::array<int, 36> kCubeConnectivity = {
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 5, 0, 5, 4,
    2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
};

void ExpectNear(mochi::Real3 const& actual, std::array<double, 3> const& expected) {
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-5) << "component " << i;
  }
}

class EpisodeRestorerTest : public testing::Test {
 protected:
  void SetUp() override {
    context = mochi::CreateContext(0);
    ASSERT_NE(context, nullptr);
    scene = context->CreateScene("episode restorer test");
    ASSERT_NE(scene, nullptr);
    mochi::RigidActorParams params;
    params.name = "cube";
    params.shape = context->CreateTriMeshShape(kCubeCoordinates, kCubeConnectivity, error);
    params.density = 1000_r;
    cube = scene->CreateRigidActor(params, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_NE(cube, nullptr);
    mochi::Real3 const com = cube->GetRigidCenterOfMassLocal(error);
    ASSERT_NEAR(com[0], kComOffsetX, 1e-4);
    ASSERT_NEAR(com[1], 0.0, 1e-4);
    ASSERT_NEAR(com[2], 0.0, 1e-4);
  }

  void TearDown() override {
    if (scene != nullptr) {
      context->DestroyScene(scene);
    }
    if (context != nullptr) {
      mochi::DestroyContext(context);
    }
  }

  // Two frames of the cube: at rest at the origin, then moved, turned a quarter about Z and moving.
  [[nodiscard]] static std::shared_ptr<FrameBank const> TwoFramesOfCubeBank() {
    BankActor actor;
    actor.path = "data/physics/actors/cube";
    actor.runtimeName = "cube";
    actor.type = "rigid";
    actor.rootRotations = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0.70710678_r, 0.70710678_r};
    actor.comTranslations = {0_r, 0_r, 0_r, 0.5_r, -0.25_r, 1_r};
    actor.linearVelocities = {0_r, 0_r, 0_r, 1_r, 2_r, 3_r};
    actor.angularVelocities = {0_r, 0_r, 0_r, 0_r, 0_r, 0.5_r};
    FrameBank bank;
    bank.episodes.push_back({
        .frames = {10, 11},
        .simulationTimesSec = {0.0, 0.02},
        .stepDtsSec = {0.01, 0.01},
        .actors = {actor},
    });
    return std::make_shared<FrameBank const>(std::move(bank));
  }

  void ExpectSecondFrameOfCube() {
    EXPECT_DOUBLE_EQ(scene->GetTotalSimulationTime(), 0.02);
    mochi::TransformRT const worldFromCom = cube->GetCenterOfMassTransform(error);
    mochi::Real4 const rotation = worldFromCom.GetRotation().ToReal4();
    ExpectNear(worldFromCom.GetTranslation(), {0.5, -0.25, 1.0});
    // The quarter turn carries the center of mass's +X offset to +Y, so the origin sits below it.
    ExpectNear(cube->GetRootTransform().GetTranslation(), {0.5, -0.25 - kComOffsetX, 1.0});
    // A unit quaternion's dot product with the expected one is +-1 only for the same rotation.
    std::array<double, 4> const expectedRotation{0.0, 0.0, 0.70710678, 0.70710678};
    double dot = 0.0;
    for (int i = 0; i < 4; ++i) {
      dot += rotation[i] * expectedRotation[i];
    }
    EXPECT_NEAR(std::abs(dot), 1.0, 1e-5);
    ExpectNear(cube->GetLinearVelocity(error), {1.0, 2.0, 3.0});
    ExpectNear(cube->GetAngularVelocity(error), {0.0, 0.0, 0.5});
    EXPECT_TRUE(error.IsOK()) << error.ToString();
  }

  mochi::Context* context = nullptr;
  mochi::Scene* scene = nullptr;
  mochi::Actor* cube = nullptr;
  mochi::Error error;
};

TEST_F(EpisodeRestorerTest, RestoresTheStateAndClockOfABankFrame) {
  auto const restorer = EpisodeRestorer::Create(*scene, TwoFramesOfCubeBank(), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(restorer->GetFrames(), (std::vector<int64_t>{10, 11}));
  EXPECT_EQ(restorer->GetActors().size(), 1);

  restorer->RestoreFrame(1, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ExpectSecondFrameOfCube();
}

TEST_F(EpisodeRestorerTest, RefusesABankThatDoesNotFitTheScene) {
  FrameBank bank = *TwoFramesOfCubeBank();
  bank.episodes[0].actors[0].runtimeName = "ball";

  auto const restorer =
      EpisodeRestorer::Create(*scene, std::make_shared<FrameBank const>(std::move(bank)), error);

  EXPECT_FALSE(error.IsOK());
  EXPECT_EQ(restorer, nullptr);
}

TEST_F(EpisodeRestorerTest, RefusesABankEpisodeOrFrameOutOfRange) {
  auto const restorer = EpisodeRestorer::Create(*scene, TwoFramesOfCubeBank(), error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  restorer->RestoreFrame(2, error);
  EXPECT_FALSE(error.IsOK());
  mochi::Error episodeError;
  restorer->RestoreFrame(1, 0, episodeError);
  EXPECT_FALSE(episodeError.IsOK());
}

} // namespace
} // namespace superdex::recorder
