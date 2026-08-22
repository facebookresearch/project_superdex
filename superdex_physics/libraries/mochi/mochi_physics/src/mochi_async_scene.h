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

#include <mochi_physics/mochi_physics.h>

#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/task_scheduler.h>

#include <marl/conditionvariable.h> // for mochi::TaskConditionVariable
#include <marl/mutex.h> // for mochi::TaskMutex
#include <marl/tsa.h> // for thread safety annotations

#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mochi {

// Forwards:
class ContextImpl;
class SceneImpl;

class AsyncSceneImpl final : public AsyncScene {
 public:
  using CommandFn = std::function<void(Scene*)>; // QueueCommand callbacks
  using OnStepFn = std::function<void(StepInfo const&)>; // Pre/post step callbacks

  explicit AsyncSceneImpl(ContextImpl* _context, std::string_view name, bool startPaused);
  ~AsyncSceneImpl() override;

  // AsyncScene API:
  Context* GetContext() override;
  void QueueCommand(CommandFn callback) override REQUIRES(!_mutex);
  void QueueActorCommand(ActorHandle actor, std::function<void(Actor*)> callback) override
      REQUIRES(!_mutex);
  void WaitForQueuedCommands() override REQUIRES(!_mutex, !_commandQueueWaitMutex);
  CallbackHandle RegisterPreStepCallback(
      std::string_view debugName,
      OnStepFn callback,
      int priority) override REQUIRES(!_mutex);
  CallbackHandle RegisterPostStepCallback(
      std::string_view debugName,
      OnStepFn callback,
      int priority) override REQUIRES(!_mutex);
  void CancelCallback(CallbackHandle handle) override REQUIRES(!_mutex);
  QueryHandle RegisterActorQuery(ActorHandle actor, QueryType type, bool forceCompute) override
      REQUIRES(!_mutex);
  void CancelActorQuery(ActorHandle actor, QueryHandle handle) override REQUIRES(!_mutex);
  AsyncStepParams GetAsyncStepParams() const override REQUIRES(!_mutex);
  void SetAsyncStepParams(AsyncStepParams const& params, Error& error) override REQUIRES(!_mutex);
  void Pause(bool shouldPause) override REQUIRES(!_mutex);
  bool IsPaused() const override REQUIRES(!_mutex);
  void RequestStepThenPause() override REQUIRES(!_mutex);

  // Internal API:
  SceneImpl* GetSceneImplUnsafe(); // Unsafe because you could use it from the wrong thread

 private:
  void StartSimLoopTask();
  void StopSimLoopTask() REQUIRES(!_mutex);
  bool IsSimThread() const REQUIRES(!_mutex);
  void SimulationLoop() REQUIRES(!_mutex);

  ContextImpl* _context = nullptr;
  std::string _name;

  // Simulation state (sim thread only!)
  SceneImpl* _scene = nullptr;
  bool _isPaused = false;

  // Simulation thread
  Guarded<std::thread::id> _simThreadId;
  mutable TaskMutex _mutex;
  TaskSemaphore _simLoopTaskComplete;
  TaskConditionVariable _wakeCondition;

  // State requested by the API thread (synchronized by _mutex)
  std::vector<CommandFn> _commandQueue;
  AsyncStepParams _stepParams;
  bool _wasWakeRequested = false; // Set back to false on wake
  bool _wasStopRequested = false; // Set back to false on stop
  bool _wasStepRequested = false; // Set back to false on step
  bool _wasPauseAfterStepRequested = false; // Set back to false on step
  bool _shouldPause = false; // Most recently requested pause state

  // Support for WaitForQueuedCommands()
  TaskMutex _commandQueueWaitMutex;
  TaskConditionVariable _commandQueueWaitCondition;
};

} // namespace mochi
