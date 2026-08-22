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

#include "mochi_debugger_test.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/test/wait_until.h>
#include <mochi_core/utils/span_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/src/mochi_scene_debugger.h>

#include <algorithm>
#include <memory>
#include <string_view>

using namespace mochi;
using namespace mochi::dbg;

namespace {

// Test fixture for DebugClient::RestoreSceneState behavior.
class SceneRestoreTest : public MochiDebuggerTest {
 protected:
  static constexpr double kTimeStep = 0.01;

  static std::shared_ptr<SceneDebugger> GetDebugger(Scene* scene) {
    return assert_cast<SceneImpl*>(scene)->GetDebugger();
  }

  static void WaitForServerMode(Scene* scene, StepMode mode) {
    auto sceneDebugger = GetDebugger(scene);
    ASSERT_NE(nullptr, sceneDebugger);
    test::WaitUntil([&]() {
      scene->UpdateDebugger();
      return sceneDebugger->GetStepMode() == mode;
    });
  }

  void ConnectSelectedPlayingScene(Scene* scene) {
    StartServer();
    ConnectClient();
    ClientSelectScene(scene->GetHandle());
    WaitForServerMode(scene, StepMode::Play);
    EXPECT_EQ(StepMode::Play, _client->GetSceneStepMode());
  }

  ShapeHandle CreateUnitCube() const {
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    return _context->CreateTetMeshShape(
        Flatten(MakeConstSpan(coordinates)),
        Flatten(MakeConstSpan(connectivity)),
        test::ExpectOK{});
  }

  Actor* CreateRigidActor(Scene* scene, Real3 position = {}) const {
    RigidActorParams params;
    params.shape = CreateUnitCube();
    params.colliderType = ColliderType::None;
    params.worldFromLocal.SetTranslation(position);
    return scene->CreateRigidActor(params, test::ExpectOK{});
  }

  bool HasClientWarning(std::string_view needle) const {
    return _clientLogs.Read([&](auto const& logs) {
      return std::any_of(logs.begin(), logs.end(), [&](auto const& log) {
        return log.channel == LogChannel::Warning &&
            std::string_view{log.message}.find(needle) != std::string_view::npos;
      });
    });
  }
};

} // namespace

TEST_F(SceneRestoreTest, RestoreSceneStateRestoresInitialStateWithoutChangingMode) {
  _client->SetSceneStepMode(StepMode::Play);
  Scene* scene = _context->CreateScene("RestoreSceneStateSuccess");
  Actor* actor = CreateRigidActor(scene, Real3{1_r, 2_r, 3_r});
  TransformRT const initialTransform = actor->GetRootTransform();
  ConnectSelectedPlayingScene(scene);

  // The debugger captures the initial state snapshot just before the first step.
  scene->Step(kTimeStep);

  actor->SetRootTransform(TransformRT{Real3{4_r, 5_r, 6_r}}, test::ExpectOK{});
  EXPECT_EQ((Real3{4_r, 5_r, 6_r}), actor->GetRootTransform().GetTranslation());

  _client->RestoreSceneState();
  test::WaitUntil([&]() {
    scene->UpdateDebugger();
    return actor->GetRootTransform().GetTranslation() == initialTransform.GetTranslation();
  });

  EXPECT_EQ(StepMode::Play, _client->GetSceneStepMode());
  EXPECT_EQ(StepMode::Play, GetDebugger(scene)->GetStepMode());
}

TEST_F(SceneRestoreTest, RestoreSceneStateFailureLogsWarning) {
  _client->SetSceneStepMode(StepMode::Play);
  Scene* scene = _context->CreateScene("RestoreSceneStateFailure");
  Actor* actor = CreateRigidActor(scene);
  ActorHandle const actorHandle = actor->GetHandle();
  ConnectSelectedPlayingScene(scene);

  // The debugger captures the initial state snapshot just before the first step.
  scene->Step(kTimeStep);

  scene->DestroyActor(actorHandle);
  EXPECT_EQ(nullptr, scene->GetActor(actorHandle));

  _client->RestoreSceneState();
  test::WaitUntil([&]() {
    scene->UpdateDebugger();
    return HasClientWarning("Failed to restore initial scene state");
  });

  EXPECT_EQ(nullptr, scene->GetActor(actorHandle));
}

TEST_F(SceneRestoreTest, RestoreSceneStateWithoutInitialStateLogsWarning) {
  _client->SetSceneStepMode(StepMode::Play);
  Scene* scene = _context->CreateScene("RestoreSceneStateNoInitialState");
  CreateRigidActor(scene);
  ConnectSelectedPlayingScene(scene);

  // No simulation step has run, so the debugger never captured an initial state snapshot.
  _client->RestoreSceneState();
  test::WaitUntil([&]() {
    scene->UpdateDebugger();
    return HasClientWarning("Initial scene state is not available");
  });
}
