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

#pragma once

#include "Async/Future.h"
#include "SuperDexBotComponent.h"
#include "SuperDexBotController.h"

#include "SuperDexBotComponentTestTypes.generated.h"

// Reflected fixtures; instantiated only by dev automation tests.

// Runs the Mochi simulation thread's control callback, as the bot's articulated actor does each
// step, from any thread.
UCLASS()
class USuperDexBotComponentTestComponent : public USuperDexBotComponent {
  GENERATED_BODY()
 public:
  bool RunControlCallback() const {
    FMochiArticulatedActorPoseVariant PoseTarget;
    return OnArticulatedActorControl(PoseTarget, nullptr, nullptr, 0.001);
  }
};

// A controller whose Initialize and Shutdown wait for the control callback to run on another
// thread, as a controller does that waits for the Mochi simulation thread (to create or destroy a
// Mochi controller, for example) while that thread steps the bot.
UCLASS()
class USuperDexBotControllerWaitsForControlCallback : public USuperDexBotController {
  GENERATED_BODY()
 public:
  // Whether the control callback ran within the wait, in Initialize and in Shutdown.
  bool bCallbackRanDuringInitialize = false;
  bool bCallbackRanDuringShutdown = false;
  TFuture<bool> Callback;

 protected:
  virtual bool Initialize(const USuperDexBotComponent* Component) override {
    bCallbackRanDuringInitialize = RunCallbackOnAnotherThread(Component);
    return true;
  }
  virtual void Shutdown(const USuperDexBotComponent* Component) override {
    bCallbackRanDuringShutdown = RunCallbackOnAnotherThread(Component);
  }

 private:
  bool RunCallbackOnAnotherThread(const USuperDexBotComponent* Component) {
    const auto* TestComponent = Cast<USuperDexBotComponentTestComponent>(Component);
    if (TestComponent == nullptr) {
      return false;
    }
    if (Callback.IsValid()) {
      Callback.Wait();
    }
    Callback = Async(
        EAsyncExecution::Thread, [TestComponent] { return TestComponent->RunControlCallback(); });
    return Callback.WaitFor(FTimespan::FromSeconds(2.0));
  }
};
