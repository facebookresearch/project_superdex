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
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_scene.h>
#include <mochi_physics/src/mochi_scene_debugger.h>

#include <atomic>
#include <memory>
#include <thread>

using namespace mochi;
using namespace mochi::dbg;

namespace {

// Test fixture used in this file
class ScenePauseTest : public MochiDebuggerTest {
 protected:
  static constexpr double kTimeStep = 0.01;
  static constexpr float kShortTimeout = 0.05f;

  void ConnectSelectedPlayingScene(Scene* scene) {
    StartServer();
    ConnectClient();
    ClientSelectScene(scene->GetHandle());
    WaitForServerMode(scene, StepMode::Play);
    EXPECT_EQ(StepMode::Play, _client->GetSceneStepMode());
  }

  Scene* CreateConnectedSelectedScene() {
    _client->SetSceneStepMode(StepMode::Play);

    Scene* scene = _context->CreateScene("ScenePauseTest");
    ConnectSelectedPlayingScene(scene);
    return scene;
  }

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

  void SetModeAndWait(Scene* scene, StepMode mode) {
    _client->SetSceneStepMode(mode);
    EXPECT_EQ(mode, _client->GetSceneStepMode());
    WaitForServerMode(scene, mode);
  }
};

} // namespace

TEST_F(ScenePauseTest, AllScenesPauseOnConnect) {
  Scene* sceneA = _context->CreateScene("A");
  Scene* sceneB = _context->CreateScene("B");
  StartServer();

  // Client not connected. Scenes not paused (or Step would block).
  sceneA->Step(kTimeStep);
  sceneB->Step(kTimeStep);

  ConnectClient();
  auto sceneDebuggerA = assert_cast<SceneImpl*>(sceneA)->GetDebugger();
  auto sceneDebuggerB = assert_cast<SceneImpl*>(sceneB)->GetDebugger();
  ASSERT_NE(nullptr, sceneDebuggerA);
  ASSERT_NE(nullptr, sceneDebuggerB);
  EXPECT_TRUE(sceneDebuggerA->IsPaused()); // Paused automatically
  EXPECT_TRUE(sceneDebuggerB->IsPaused()); // Paused automatically
}

TEST_F(ScenePauseTest, NewScenePauseOnStart) {
  StartServer();
  ConnectClient();

  // Create a new scene. It should start paused.
  Scene* sceneA = _context->CreateScene("A");
  auto sceneDebuggerA = assert_cast<SceneImpl*>(sceneA)->GetDebugger();
  ASSERT_NE(nullptr, sceneDebuggerA);
  EXPECT_TRUE(sceneDebuggerA->IsPaused()); // Paused automatically
}

TEST_F(ScenePauseTest, PlayModeBeforeConnect) {
  _client->SetSceneStepMode(StepMode::Play);

  // Some scenes exist before connect
  Scene* sceneA = _context->CreateScene("A");
  Scene* sceneB = _context->CreateScene("B");
  StartServer();
  ConnectClient();

  // Existing scenes may start paused, but the client should unpause them immediately.
  auto sceneDebuggerA = assert_cast<SceneImpl*>(sceneA)->GetDebugger();
  auto sceneDebuggerB = assert_cast<SceneImpl*>(sceneB)->GetDebugger();
  ASSERT_NE(nullptr, sceneDebuggerA);
  ASSERT_NE(nullptr, sceneDebuggerB);
  test::WaitUntil([&]() {
    sceneA->UpdateDebugger();
    sceneB->UpdateDebugger();
    return !sceneDebuggerA->IsPaused() && !sceneDebuggerB->IsPaused();
  });

  // Create a new scene after connect.
  Scene* sceneC = _context->CreateScene("C");
  auto sceneDebuggerC = assert_cast<SceneImpl*>(sceneC)->GetDebugger();
  ASSERT_NE(nullptr, sceneDebuggerC);

  // The new scene may start paused, but the client should unpause it immediately.
  test::WaitUntil([&]() {
    sceneC->UpdateDebugger();
    return !sceneDebuggerC->IsPaused();
  });
}

TEST_F(ScenePauseTest, StepModeAppliesToAllScenes) {
  Scene* sceneA = _context->CreateScene("A");
  Scene* sceneB = _context->CreateScene("B");
  StartServer();
  ConnectClient();
  ASSERT_NE(nullptr, GetDebugger(sceneA));
  ASSERT_NE(nullptr, GetDebugger(sceneB));

  // Unpause every scene, even though none is selected.
  _client->SelectScene({});
  _client->SetSceneStepMode(StepMode::Play);
  EXPECT_EQ(StepMode::Play, _client->GetSceneStepMode());
  WaitForServerMode(sceneA, StepMode::Play);
  WaitForServerMode(sceneB, StepMode::Play);

  // The step mode is global, so selecting a scene does not change it.
  ClientSelectScene(sceneA->GetHandle());
  EXPECT_EQ(StepMode::Play, _client->GetSceneStepMode());
  ClientSelectScene(sceneB->GetHandle());
  EXPECT_EQ(StepMode::Play, _client->GetSceneStepMode());

  // Pause every scene.
  _client->SetSceneStepMode(StepMode::Pause);
  EXPECT_EQ(StepMode::Pause, _client->GetSceneStepMode());
  WaitForServerMode(sceneA, StepMode::Pause);
  WaitForServerMode(sceneB, StepMode::Pause);
}

TEST_F(ScenePauseTest, SingleStepPausesAllScenes) {
  Scene* sceneA = _context->CreateScene("A");
  Scene* sceneB = _context->CreateScene("B");
  StartServer();
  ConnectClient();
  ClientSelectScene(sceneA->GetHandle());
  SetModeAndWait(sceneA, StepMode::Play);
  WaitForServerMode(sceneB, StepMode::Play);

  // Stepping the selected scene pauses every scene.
  _client->StepScene();
  EXPECT_EQ(StepMode::Pause, _client->GetSceneStepMode());
  WaitForServerMode(sceneA, StepMode::Pause);
  WaitForServerMode(sceneB, StepMode::Pause);
}

TEST_F(ScenePauseTest, StepBlocksWhenPaused) {
  Scene* scene = _context->CreateScene("StepCommandTest");
  StartServer();
  ConnectClient();
  ClientSelectScene(scene->GetHandle());
  auto sceneDebugger = assert_cast<SceneImpl*>(scene)->GetDebugger();
  ASSERT_NE(nullptr, sceneDebugger);
  EXPECT_TRUE(sceneDebugger->IsPaused()); // Starts paused

  // Since the debugger is paused, the next call to Step will block.
  // Therefore, use another thread to call Step.
  std::atomic<bool> ready{false};
  std::atomic<int> stepCounter{0};
  auto thread = std::thread{[&]() {
    ready.store(true);
    for (int i = 0; i < 10; ++i) {
      scene->Step(kTimeStep);
      stepCounter++;
    }
  }};

  // Wait for the thread to start
  test::WaitUntil([&]() { return ready.load(); });

  // Debugger is paused. First step has not completed.
  EXPECT_FALSE(
      test::WaitUntil(
          [&]() { return stepCounter > 0; }, /*timeoutSec*/ 0.05f, /*failOnTimeout*/ false));
  EXPECT_EQ(0, stepCounter.load());

  // Step once
  _client->StepScene();
  test::WaitUntil([&]() { return stepCounter > 0; });
  EXPECT_EQ(1, stepCounter.load());

  // Scene is paused and waiting again.
  EXPECT_TRUE(sceneDebugger->IsPaused());
  EXPECT_FALSE(
      test::WaitUntil(
          [&]() { return stepCounter > 1; },
          /*timeoutSec*/ 0.05f,
          /*failOnTimeout*/ false));
  EXPECT_EQ(1, stepCounter.load()); // no change

  // Step again
  _client->StepScene();
  test::WaitUntil([&]() { return stepCounter > 1; });
  EXPECT_EQ(2, stepCounter.load());

  // Scene is paused and waiting again.
  EXPECT_TRUE(sceneDebugger->IsPaused());
  EXPECT_FALSE(
      test::WaitUntil(
          [&]() { return stepCounter > 2; },
          /*timeoutSec*/ 0.05f,
          /*failOnTimeout*/ false));
  EXPECT_EQ(2, stepCounter.load()); // no change

  // Unpause
  _client->SetSceneStepMode(StepMode::Play);
  thread.join();
  EXPECT_EQ(10, stepCounter.load()); // all 10 steps
}

TEST_F(ScenePauseTest, SingleStepPausesScene) {
  Scene* scene = _context->CreateScene("StepCommandTest");
  StartServer();
  ConnectClient();
  ClientSelectScene(scene->GetHandle());
  auto sceneDebugger = assert_cast<SceneImpl*>(scene)->GetDebugger();
  ASSERT_NE(nullptr, sceneDebugger);
  EXPECT_EQ(StepMode::Pause, sceneDebugger->GetStepMode());

  // Pause --> Play
  _client->SetSceneStepMode(StepMode::Play);
  test::WaitUntil([&]() {
    scene->UpdateDebugger();
    return sceneDebugger->GetStepMode() == StepMode::Play;
  });

  // Running normally on the server
  scene->Step(kTimeStep);
  scene->Step(kTimeStep);
  scene->Step(kTimeStep);

  // SceneStep results in: Play --> Pause
  _client->StepScene();
  test::WaitUntil([&]() {
    scene->UpdateDebugger();
    return sceneDebugger->GetStepMode() == StepMode::Pause;
  });

  // This step will not block, but a 2nd step would block.
  scene->Step(kTimeStep);

  // Pause --> FastForward
  _client->SetSceneStepMode(StepMode::FastForward);
  test::WaitUntil([&]() {
    scene->UpdateDebugger();
    return sceneDebugger->GetStepMode() == StepMode::FastForward;
  });

  // Running normally on the server
  scene->Step(kTimeStep);
  scene->Step(kTimeStep);
  scene->Step(kTimeStep);

  // SceneStep results in: FastForward --> Pause
  _client->StepScene();
  test::WaitUntil([&]() {
    scene->UpdateDebugger();
    return sceneDebugger->GetStepMode() == StepMode::Pause;
  });

  // This step will not block, but a 2nd step would block.
  scene->Step(kTimeStep);
}

TEST_F(ScenePauseTest, UnpauseWhenClientDisconnects) {
  Scene* scene = _context->CreateScene("MyScene");
  StartServer();
  ConnectClient();

  std::atomic<bool> ready{false};
  std::atomic<int> stepCounter{0};
  auto thread = std::thread{[&]() {
    ready.store(true);
    for (int i = 0; i < 10; ++i) {
      scene->Step(kTimeStep);
      stepCounter++;
    }
  }};

  // Wait for the thread to start
  test::WaitUntil([&]() { return ready.load(); });

  // Since the scene debugger is paused, the first call to Step has not returned yet.
  EXPECT_FALSE(
      test::WaitUntil(
          [&]() { return stepCounter > 0; }, /*timeoutSec*/ 0.05f, /*failOnTimeout*/ false));
  EXPECT_EQ(0, stepCounter.load());

  DisconnectClient();
  EXPECT_EQ(
      nullptr,
      assert_cast<SceneImpl*>(scene)->GetDebugger()); // Scene debugger no longer associated
  thread.join();
  EXPECT_EQ(10, stepCounter.load()); // All 10 steps completed
}

TEST_F(ScenePauseTest, UnpauseWhenServerStops) {
  Scene* scene = _context->CreateScene("MyScene");
  StartServer();
  ConnectClient();

  std::atomic<bool> ready{false};
  std::atomic<int> stepCounter{0};
  auto thread = std::thread{[&]() {
    ready.store(true);
    for (int i = 0; i < 10; ++i) {
      scene->Step(kTimeStep);
      stepCounter++;
    }
  }};

  // Wait for the thread to start
  test::WaitUntil([&]() { return ready.load(); });

  // Since the scene debugger is paused, the first call to Step has not returned yet.
  EXPECT_FALSE(
      test::WaitUntil(
          [&]() { return stepCounter > 0; }, /*timeoutSec*/ 0.05f, /*failOnTimeout*/ false));
  EXPECT_EQ(0, stepCounter.load());

  _server->Stop();
  EXPECT_EQ(
      nullptr,
      assert_cast<SceneImpl*>(scene)->GetDebugger()); // Scene debugger no longer associated
  thread.join();
  EXPECT_EQ(10, stepCounter.load()); // All 10 steps completed
}

TEST_F(ScenePauseTest, SetSceneStepModeUpdatesClientAndServer) {
  Scene* scene = CreateConnectedSelectedScene();

  SetModeAndWait(scene, StepMode::FastForward);
  SetModeAndWait(scene, StepMode::Pause);
  SetModeAndWait(scene, StepMode::Play);
}

TEST_F(ScenePauseTest, PlayWaitsForRealTimeClock) {
  Scene* scene = CreateConnectedSelectedScene();
  auto sceneDebugger = GetDebugger(scene);
  ASSERT_NE(nullptr, sceneDebugger);

  std::atomic<double> clock{0.0};
  sceneDebugger->SetClock([&clock]() { return clock.load(); });

  scene->Step(kTimeStep);

  std::atomic<bool> stepStarted{false};
  std::atomic<bool> stepFinished{false};
  std::thread thread{[&]() {
    stepStarted.store(true);
    scene->Step(kTimeStep);
    stepFinished.store(true);
  }};

  test::WaitUntil([&]() { return stepStarted.load(); });
  EXPECT_FALSE(
      test::WaitUntil(
          [&]() { return stepFinished.load(); },
          kShortTimeout,
          /*failOnTimeout*/ false));

  clock.store(kTimeStep);
  test::WaitUntil([&]() { return stepFinished.load(); });
  thread.join();
}

TEST_F(ScenePauseTest, PlayThrottlesEveryStepWithoutSpuriousReset) {
  Scene* scene = CreateConnectedSelectedScene();
  auto sceneDebugger = GetDebugger(scene);
  ASSERT_NE(nullptr, sceneDebugger);

  std::atomic<double> clock{0.0};
  sceneDebugger->SetClock([&clock]() { return clock.load(); });

  scene->Step(kTimeStep);

  int constexpr kNumSteps = 4;
  std::atomic<int> stepCounter{0};
  std::atomic<bool> done{false};
  std::thread thread{[&]() {
    for (int i = 0; i < kNumSteps; ++i) {
      scene->Step(kTimeStep);
      ++stepCounter;
    }
    done.store(true);
  }};

  double constexpr kJitter = kTimeStep * 0.01;
  for (int i = 0; i < kNumSteps; ++i) {
    EXPECT_FALSE(
        test::WaitUntil(
            [&]() { return stepCounter.load() > i; },
            kShortTimeout,
            /*failOnTimeout*/ false));
    clock.store((i + 1) * kTimeStep + kJitter);
    test::WaitUntil([&]() { return stepCounter.load() > i; });
  }

  test::WaitUntil([&]() { return done.load(); });
  thread.join();
  EXPECT_EQ(kNumSteps, stepCounter.load());
}

TEST_F(ScenePauseTest, PlayResetsThrottleAfterExternalStall) {
  Scene* scene = CreateConnectedSelectedScene();
  auto sceneDebugger = GetDebugger(scene);
  ASSERT_NE(nullptr, sceneDebugger);

  std::atomic<double> clock{0.0};
  sceneDebugger->SetClock([&clock]() { return clock.load(); });

  scene->Step(kTimeStep);

  {
    std::atomic<bool> finished{false};
    std::thread thread{[&]() {
      scene->Step(kTimeStep);
      finished.store(true);
    }};
    EXPECT_FALSE(
        test::WaitUntil(
            [&]() { return finished.load(); },
            kShortTimeout,
            /*failOnTimeout*/ false));
    clock.store(kTimeStep);
    test::WaitUntil([&]() { return finished.load(); });
    thread.join();
  }

  clock.store(kTimeStep + 10.0 * kTimeStep);
  std::atomic<bool> finished{false};
  std::thread thread{[&]() {
    scene->Step(kTimeStep);
    finished.store(true);
  }};

  test::WaitUntil([&]() { return finished.load(); });
  ASSERT_TRUE(finished.load());
  thread.join();
}

TEST_F(ScenePauseTest, FastForwardDoesNotWaitForRealTimeClock) {
  Scene* scene = CreateConnectedSelectedScene();
  auto sceneDebugger = GetDebugger(scene);
  ASSERT_NE(nullptr, sceneDebugger);
  SetModeAndWait(scene, StepMode::FastForward);

  std::atomic<double> clock{0.0};
  sceneDebugger->SetClock([&clock]() { return clock.load(); });

  std::atomic<bool> stepsStarted{false};
  std::atomic<bool> stepsFinished{false};
  std::thread thread{[&]() {
    stepsStarted.store(true);
    scene->Step(kTimeStep);
    scene->Step(kTimeStep);
    scene->Step(kTimeStep);
    stepsFinished.store(true);
  }};

  test::WaitUntil([&]() { return stepsStarted.load(); });
  test::WaitUntil([&]() { return stepsFinished.load(); });
  ASSERT_TRUE(stepsFinished.load());
  thread.join();
}
