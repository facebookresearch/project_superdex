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

#include <mochi_core/test/wait_until.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_physics/src/mochi_island.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/src/mochi_scene_debugger.h>

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>

using namespace mochi;
using namespace mochi::dbg;

namespace {

class SceneSettingsTest : public MochiDebuggerTest {
 protected:
  static constexpr double kTimeStep = 0.01;
  static constexpr double kTimeout = 60.0; // Generous to prevent spurious CI errors [seconds]
  static constexpr int kNumActors = 3;
  static constexpr SceneSettings kAllForced{.forceSingleThreaded = true, .forceSingleIsland = true};

  int _numThreads = 0;

  void SetUp() override {
    MochiDebuggerTest::SetUp(); // Call down
    _numThreads = _context->GetNumThreads();
    ASSERT_GT(_numThreads, 0) << "These tests require at least one worker thread";
    _client->SetSceneStepMode(StepMode::FastForward);
  }

  // A scene of separated dynamic actors that do not collide, so each one is its own island.
  Scene* CreateSceneWithSeparateIslands(std::string_view name) const {
    Scene* scene = _context->CreateScene(name);
    scene->SetGravity({});
    auto [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    ShapeHandle const cube = _context->CreateTetMeshShape(
        Flatten(MakeConstSpan(coordinates)),
        Flatten(MakeConstSpan(connectivity)),
        test::ExpectOK{});
    for (int i = 0; i < kNumActors; ++i) {
      RigidActorParams params;
      params.shape = cube;
      params.colliderType = ColliderType::None;
      params.worldFromLocal.SetTranslation(Real3{3_r * i, 0_r, 0_r});
      scene->CreateRigidActor(params, test::ExpectOK{});
    }
    return scene;
  }

  void SetClientSettings(SceneSettings const& scene, bool syncPerformance = false) {
    auto settings = _client->GetSettings();
    settings.scene = scene;
    settings.sync.enabled = syncPerformance;
    settings.sync.syncPerformance = syncPerformance;
    _client->SetSettings(settings);
  }

  // Send a request and return the debugger's settings from the reply. Every request sent to the
  // scene before it has been applied once this returns.
  SceneSettings SendSettingsRequest(Scene* scene, protocol::SceneSettingsRequest request = {}) {
    request.scene = scene->GetHandle();
    protocol::SceneSettingsReply reply;
    Error error;
    std::atomic<bool> requestFinished = false;
    std::thread requestThread([&] {
      reply = _client->SendAndAwaitReply(request, kTimeout, error);
      requestFinished = true;
    });
    test::WaitUntil([&] {
      scene->UpdateDebugger();
      return requestFinished.load();
    });
    requestThread.join();
    EXPECT_TRUE(error.IsOK()) << error.GetDescription();
    EXPECT_TRUE(reply.error.empty()) << reply.error.c_str();
    return {
        .forceSingleThreaded = reply.forceSingleThreaded,
        .forceSingleIsland = reply.forceSingleIsland};
  }

  // Step the scene and return the profile the client receives for it.
  protocol::StepProfileData StepAndWaitForProfile(Scene* scene) const {
    uint64_t endSequence = 0;
    _client->ReadProfileHistory(
        [&](auto const& history) { endSequence = history.GetEndSequence(); });
    scene->Step(kTimeStep);
    protocol::StepProfileData profile;
    test::WaitUntil([&] {
      scene->UpdateDebugger();
      bool received = false;
      _client->ReadProfileHistory([&](auto const& history) {
        if (history.GetEndSequence() > endSequence) {
          profile = history.back();
          received = true;
        }
      });
      return received;
    });
    return profile;
  }

  // Step the scene and return the worker thread count seen during the step. Works whether or not
  // the scene is selected.
  int StepAndGetNumThreads(Scene* scene) const {
    int numThreads = -1;
    CallbackHandle const callback = scene->RegisterPreStepCallback(
        "GetNumThreads", [&](auto const&) { numThreads = _context->GetNumThreads(); });
    scene->Step(kTimeStep);
    scene->CancelCallback(callback);
    return numThreads;
  }

  static entt::registry& GetRegistry(Scene* scene) {
    return assert_cast<SceneImpl*>(scene)->GetRegistry();
  }

  static int GetNumIslands(Scene* scene) {
    return static_cast<int>(GetRegistry(scene).view<TagIsland const>().size());
  }

  static bool IsSingleIslandForcedByDebugger(Scene* scene) {
    return island::GetForceSingleIslandByDebugger(GetRegistry(scene));
  }

  static std::shared_ptr<SceneDebugger> GetDebugger(Scene* scene) {
    return assert_cast<SceneImpl*>(scene)->GetDebugger();
  }
};

enum class StepThreadMode { Unbound, Bound, BoundSingleThreaded };

class SceneSettingsCallbackExceptionTest
    : public SceneSettingsTest,
      public testing::WithParamInterface<std::tuple<StepThreadMode, bool>> {};

} // namespace

TEST_F(SceneSettingsTest, ProfilesReportEffectiveThreadCount) {
  Scene* scene = CreateSceneWithSeparateIslands("Threads");
  SetClientSettings({}, /*syncPerformance*/ true);
  StartServer();
  ConnectClient();
  SendSettingsRequest(scene);

  EXPECT_EQ(_numThreads, StepAndWaitForProfile(scene).numWorkerThreads);

  _context->SetIsSingleThreaded(true);
  MOCHI_DEFER(_context->SetIsSingleThreaded(false));
  EXPECT_EQ(0, StepAndWaitForProfile(scene).numWorkerThreads);
}

TEST_F(SceneSettingsTest, ForceSingleThreaded) {
  Scene* scene = CreateSceneWithSeparateIslands("Threads");
  SetClientSettings({.forceSingleThreaded = true}, /*syncPerformance*/ true);
  StartServer();
  ConnectClient();
  SendSettingsRequest(scene);

  EXPECT_EQ(0, StepAndWaitForProfile(scene).numWorkerThreads);
  EXPECT_EQ(_numThreads, _context->GetNumThreads()); // The application's mode is unchanged

  SetClientSettings({}, /*syncPerformance*/ true);
  SendSettingsRequest(scene);
  EXPECT_EQ(_numThreads, StepAndWaitForProfile(scene).numWorkerThreads);
}

// The step thread stays bound across steps, so a leaked push or a lost application push would be
// visible here.
TEST_F(SceneSettingsTest, ForceSingleThreadedPreservesApplicationLocalMode) {
  Scene* scene = CreateSceneWithSeparateIslands("Threads");
  SetClientSettings({.forceSingleThreaded = true});
  StartServer();
  ConnectClient();
  SendSettingsRequest(scene);
  _context->BindThisThread();
  MOCHI_DEFER(_context->UnbindThisThread());

  EXPECT_EQ(0, StepAndGetNumThreads(scene));
  EXPECT_FALSE(TaskScheduler::IsLocalSingleThreaded());

  TaskScheduler::PushLocalSingleThreadedMode();
  EXPECT_EQ(0, StepAndGetNumThreads(scene));
  SetClientSettings({});
  SendSettingsRequest(scene);
  EXPECT_EQ(0, StepAndGetNumThreads(scene)); // The application's push still applies
  TaskScheduler::PopLocalSingleThreadedMode();
  EXPECT_FALSE(TaskScheduler::IsLocalSingleThreaded());

  EXPECT_EQ(_numThreads, StepAndGetNumThreads(scene));
}

TEST_P(SceneSettingsCallbackExceptionTest, ForceSingleThreadedReleasedAfterCallbackThrows) {
  auto const [threadMode, postStep] = GetParam();
  Scene* scene = CreateSceneWithSeparateIslands("Exceptions");
  SetClientSettings({.forceSingleThreaded = true});
  StartServer();
  ConnectClient();
  SendSettingsRequest(scene);

  bool const bindThread = threadMode != StepThreadMode::Unbound;
  if (bindThread) {
    _context->BindThisThread();
  }
  MOCHI_DEFER(if (bindThread) { _context->UnbindThisThread(); });

  bool const applicationSingleThreaded = threadMode == StepThreadMode::BoundSingleThreaded;
  if (applicationSingleThreaded) {
    TaskScheduler::PushLocalSingleThreadedMode();
  }
  MOCHI_DEFER(if (applicationSingleThreaded) { TaskScheduler::PopLocalSingleThreadedMode(); });

  auto throwDuringStep = [&] {
    auto callback = [&](auto const&) {
      EXPECT_EQ(0, _context->GetNumThreads());
      throw std::runtime_error("Step callback failed");
    };
    CallbackHandle const handle = postStep ? scene->RegisterPostStepCallback("Throw", callback)
                                           : scene->RegisterPreStepCallback("Throw", callback);
    EXPECT_THROW(scene->Step(kTimeStep), std::runtime_error);
    scene->CancelCallback(handle);
    EXPECT_EQ(applicationSingleThreaded, TaskScheduler::IsLocalSingleThreaded());
  };

  int const expectedNumThreads = applicationSingleThreaded ? 0 : _numThreads;
  throwDuringStep();
  SetClientSettings({});
  SendSettingsRequest(scene);
  EXPECT_EQ(expectedNumThreads, StepAndGetNumThreads(scene));

  SetClientSettings({.forceSingleThreaded = true});
  SendSettingsRequest(scene);
  throwDuringStep();
  DisconnectClient();
  test::WaitUntil([&] { return GetDebugger(scene) == nullptr; });
  EXPECT_EQ(applicationSingleThreaded, TaskScheduler::IsLocalSingleThreaded());
  EXPECT_EQ(expectedNumThreads, StepAndGetNumThreads(scene));
}

INSTANTIATE_TEST_SUITE_P(
    ThreadModes,
    SceneSettingsCallbackExceptionTest,
    testing::Combine(
        testing::Values(
            StepThreadMode::Unbound,
            StepThreadMode::Bound,
            StepThreadMode::BoundSingleThreaded),
        testing::Bool()));

TEST_F(SceneSettingsTest, ForceSingleIsland) {
  Scene* scene = CreateSceneWithSeparateIslands("Islands");
  SetClientSettings({}, /*syncPerformance*/ true);
  StartServer();
  ConnectClient();
  SendSettingsRequest(scene);
  EXPECT_EQ(kNumActors, StepAndWaitForProfile(scene).activeIslands);

  SetClientSettings({.forceSingleIsland = true}, /*syncPerformance*/ true);
  SendSettingsRequest(scene);
  EXPECT_EQ(1, StepAndWaitForProfile(scene).activeIslands);
  EXPECT_FALSE(scene->GetForceSingleIsland()); // The application's value is unchanged

  SetClientSettings({}, /*syncPerformance*/ true);
  SendSettingsRequest(scene);
  EXPECT_EQ(kNumActors, StepAndWaitForProfile(scene).activeIslands);
}

TEST_F(SceneSettingsTest, RequestAppliesOnlyFieldsWithValues) {
  Scene* scene = CreateSceneWithSeparateIslands("Request");
  StartServer();
  ConnectClient();
  EXPECT_EQ(SceneSettings{}, SendSettingsRequest(scene));

  protocol::SceneSettingsRequest islandRequest;
  islandRequest.forceSingleIsland = true;
  EXPECT_EQ(SceneSettings{.forceSingleIsland = true}, SendSettingsRequest(scene, islandRequest));

  protocol::SceneSettingsRequest threadedRequest;
  threadedRequest.forceSingleThreaded = true;
  EXPECT_EQ(kAllForced, SendSettingsRequest(scene, threadedRequest));
  EXPECT_EQ(kAllForced, SendSettingsRequest(scene));
}

TEST_F(SceneSettingsTest, SelectionChangeReleasesPreviousScene) {
  Scene* sceneA = CreateSceneWithSeparateIslands("A");
  Scene* sceneB = CreateSceneWithSeparateIslands("B");
  SetClientSettings(kAllForced);
  StartServer();
  ConnectClient();
  ClientSelectScene(sceneA->GetHandle());
  ClientSelectScene(sceneB->GetHandle());

  EXPECT_EQ(SceneSettings{}, SendSettingsRequest(sceneA));
  EXPECT_EQ(_numThreads, StepAndGetNumThreads(sceneA));
  EXPECT_EQ(kNumActors, GetNumIslands(sceneA));

  EXPECT_EQ(kAllForced, SendSettingsRequest(sceneB));
  EXPECT_EQ(0, StepAndGetNumThreads(sceneB));
  EXPECT_EQ(1, GetNumIslands(sceneB));
}

TEST_F(SceneSettingsTest, DisconnectReleasesScene) {
  Scene* scene = CreateSceneWithSeparateIslands("Disconnect");
  SetClientSettings(kAllForced);
  StartServer();
  ConnectClient();
  ASSERT_EQ(kAllForced, SendSettingsRequest(scene));
  EXPECT_EQ(0, StepAndGetNumThreads(scene));
  EXPECT_EQ(1, GetNumIslands(scene));

  DisconnectClient();
  test::WaitUntil([&] { return GetDebugger(scene) == nullptr; });
  EXPECT_EQ(_numThreads, StepAndGetNumThreads(scene));
  EXPECT_FALSE(IsSingleIslandForcedByDebugger(scene));
  EXPECT_EQ(kNumActors, GetNumIslands(scene));
}

// The old debugger releases the scene on the scene's thread only after the new one has replaced it.
TEST_F(SceneSettingsTest, QuickReconnectReappliesSettings) {
  Scene* scene = CreateSceneWithSeparateIslands("Reconnect");
  SetClientSettings(kAllForced);
  StartServer();
  ConnectClient();
  ASSERT_EQ(kAllForced, SendSettingsRequest(scene));
  auto const oldDebugger = GetDebugger(scene);

  DisconnectClient();
  ConnectClient();
  test::WaitUntil([&] {
    auto const debugger = GetDebugger(scene);
    return debugger != nullptr && debugger != oldDebugger;
  });
  EXPECT_EQ(kAllForced, SendSettingsRequest(scene));
  EXPECT_EQ(0, StepAndGetNumThreads(scene));
  EXPECT_EQ(1, GetNumIslands(scene));
}
