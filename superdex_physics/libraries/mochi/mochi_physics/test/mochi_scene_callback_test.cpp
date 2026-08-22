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

#include "mochi_physics_test_fixture.h"

#include <string>
#include <vector>

using namespace mochi;

class SceneCallback : public mochi::test::MochiSceneTestBase {};

TEST_F(SceneCallback, PrePostStepCallbacks) {
  std::string log;
  double timeStep = 0.0;

  // Default handle is not "valid"
  EXPECT_FALSE(CallbackHandle{}.IsValid());

  // Register some post-step and post-step callbacks
  std::vector<CallbackHandle> handles;
  handles.reserve(6);
  handles.push_back(_scene->RegisterPostStepCallback("Callback A", [&](auto const& step) {
    EXPECT_EQ(_scene, step.scene);
    EXPECT_EQ(timeStep, step.timeStepSec);
    log += "A";
  }));
  handles.push_back(_scene->RegisterPostStepCallback("Callback B", [&](auto const& step) {
    EXPECT_EQ(_scene, step.scene);
    EXPECT_EQ(timeStep, step.timeStepSec);
    log += "B";
  }));
  handles.push_back(_scene->RegisterPostStepCallback("Callback C", [&](auto const& step) {
    EXPECT_EQ(_scene, step.scene);
    EXPECT_EQ(timeStep, step.timeStepSec);
    log += "C";
  }));
  handles.push_back(_scene->RegisterPreStepCallback("Callback D", [&](auto const& step) {
    EXPECT_EQ(_scene, step.scene);
    EXPECT_EQ(timeStep, step.timeStepSec);
    log += "D";
  }));
  handles.push_back(_scene->RegisterPreStepCallback("Callback E", [&](auto const& step) {
    EXPECT_EQ(_scene, step.scene);
    EXPECT_EQ(timeStep, step.timeStepSec);
    log += "E";
  }));
  handles.push_back(_scene->RegisterPreStepCallback("Callback F", [&](auto const& step) {
    EXPECT_EQ(_scene, step.scene);
    EXPECT_EQ(timeStep, step.timeStepSec);
    log += "F";
  }));

  // All handles are valid an unique.
  for (int i = 0; i < isize(handles); ++i) {
    EXPECT_TRUE(handles[i].IsValid());
    for (int j = i + 1; j < isize(handles); ++j) {
      EXPECT_NE(handles[i], handles[j]);
    }
  }

  // Callbacks should be fired even when the time step is 0.0
  _scene->Step(timeStep);

  // The order between like callbacks is not specified, but all pre-step callbacks should happen
  // before post-step callbacks.
  std::sort(log.begin(), log.begin() + 3);
  std::sort(log.begin() + 3, log.end());
  EXPECT_STREQ("DEFABC", log.c_str());
  log.clear();

  // Canceling an invalid callback is allowed
  _scene->CancelCallback(CallbackHandle{});
  _scene->Step(timeStep);
  std::sort(log.begin(), log.begin() + 3);
  std::sort(log.begin() + 3, log.end());
  EXPECT_STREQ("DEFABC", log.c_str()); // no change
  log.clear();

  // Now cancel one of each callback.
  _scene->CancelCallback(handles[1]);
  _scene->CancelCallback(handles[4]);

  // Non-zero time step, just to mix it up.
  timeStep = 0.01;
  _scene->Step(timeStep);
  std::sort(log.begin(), log.begin() + 2);
  std::sort(log.begin() + 2, log.end());
  EXPECT_STREQ("DFAC", log.c_str());
  log.clear();

  // Attempting to cancel redundantly is allowed
  _scene->CancelCallback(handles[1]);
  _scene->CancelCallback(handles[4]);
  _scene->Step(timeStep);
  std::sort(log.begin(), log.begin() + 2);
  std::sort(log.begin() + 2, log.end());
  EXPECT_STREQ("DFAC", log.c_str()); // no change
  log.clear();

  // Register another callback of each type
  handles.push_back(_scene->RegisterPreStepCallback("Callback G", [&](auto const& step) {
    EXPECT_EQ(_scene, step.scene);
    EXPECT_EQ(timeStep, step.timeStepSec);
    log += "G";
  }));
  handles.push_back(_scene->RegisterPostStepCallback("Callback H", [&](auto const& step) {
    EXPECT_EQ(_scene, step.scene);
    EXPECT_EQ(timeStep, step.timeStepSec);
    log += "H";
  }));
  _scene->Step(timeStep);
  std::sort(log.begin(), log.begin() + 3);
  std::sort(log.begin() + 3, log.end());
  EXPECT_STREQ("DFGACH", log.c_str());
  log.clear();

  // Cancel all
  for (auto h : handles) {
    _scene->CancelCallback(h);
  }
  _scene->Step(timeStep);
  EXPECT_STREQ("", log.c_str());
}

static void TestCallbackPriorityOrder(
    Scene* scene,
    std::function<CallbackHandle(std::function<void(StepInfo const&)> fn, int priority)> const&
        registerCallback) {
  std::string log;

  // Register several callbacks with various priority settings
  std::vector<CallbackHandle> handles;
  handles.reserve(6);
  handles.push_back(registerCallback([&](StepInfo const&) { log += "A"; }, 100));
  handles.push_back(registerCallback([&](StepInfo const&) { log += "B"; }, 10));
  handles.push_back(registerCallback([&](StepInfo const&) { log += "C"; }, -10));
  handles.push_back(registerCallback([&](StepInfo const&) { log += "D"; }, 20));
  handles.push_back(registerCallback([&](StepInfo const&) { log += "E"; }, -30));
  handles.push_back(registerCallback([&](StepInfo const&) { log += "F"; }, 0));

  // Expect priority order (lower priority values get called first).
  scene->Step(0.0);
  EXPECT_STREQ("ECFBDA", log.c_str());
}

TEST_F(SceneCallback, PreStepPriorityOrder) {
  TestCallbackPriorityOrder(_scene, [this](std::function<void(StepInfo const&)> fn, int priority) {
    return _scene->RegisterPreStepCallback("", fn, priority);
  });
}

TEST_F(SceneCallback, PostStepPriorityOrder) {
  TestCallbackPriorityOrder(_scene, [this](std::function<void(StepInfo const&)> fn, int priority) {
    return _scene->RegisterPostStepCallback("", fn, priority);
  });
}

TEST_F(SceneCallback, HandleFromAnotherScene) {
  std::string log;

  // Register for a callback from each scene
  auto* otherScene = _mochiContext->CreateScene("other scene");
  auto hA = _scene->RegisterPreStepCallback("", [&](auto const&) { log += "A"; });
  auto hB = otherScene->RegisterPreStepCallback("", [&](auto const&) { log += "B"; });

  // Step each
  _scene->Step(0.0);
  EXPECT_STREQ("A", log.c_str());
  log.clear();
  otherScene->Step(0.0);
  EXPECT_STREQ("B", log.c_str());
  log.clear();

  // Try to cancel the wrong handle
  {
    // Expect log warnings in this scope
    int warningCount = 0;
    auto prevLoggingHook = mochi::GetLogCallback();
    _mochiContext->SetLogCallback(
        [&](LogChannel channel, char const* /*message*/, char const* /*file*/, int /*line*/) {
          if (channel == LogChannel::Warning) {
            ++warningCount;
          }
        });

    _scene->CancelCallback(hB);
    EXPECT_EQ(1, warningCount);
    otherScene->CancelCallback(hA);
    EXPECT_EQ(2, warningCount);

    // Restore normal logging
    mochi::SetLogCallback(prevLoggingHook);
  }

  // No change. Callbacks are still registered.
  _scene->Step(0.0);
  EXPECT_STREQ("A", log.c_str());
  log.clear();
  otherScene->Step(0.0);
  EXPECT_STREQ("B", log.c_str());
  log.clear();
}
