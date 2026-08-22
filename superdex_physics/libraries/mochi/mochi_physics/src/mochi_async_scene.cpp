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

#include "mochi_async_scene.h"

#include "mochi_context.h"
#include "mochi_query.h"
#include "mochi_scene.h"

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/time.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace mochi {

AsyncSceneImpl::AsyncSceneImpl(ContextImpl* context, std::string_view name, bool startPaused)
    : _context(context), _name(name), _shouldPause(startPaused) {
  MOCHI_ASSERT(context != nullptr);
  StartSimLoopTask();
}

AsyncSceneImpl::~AsyncSceneImpl() {
  MOCHI_ASSERT(_scene != nullptr);
  StopSimLoopTask();
  MOCHI_ASSERT(
      _scene == nullptr,
      "The Scene should have been destroyed on the simulation thread prior to exit");
}

Context* AsyncSceneImpl::GetContext() {
  return _context;
}

void AsyncSceneImpl::StartSimLoopTask() {
  auto& scheduler = _context->GetTaskScheduler();
  MOCHI_ASSERT(
      !TaskScheduler::IsCurrentThreadAWorker(),
      "AsyncScene must not be created from a Mochi worker thread. We should have caught this earlier.");
  MOCHI_ASSERT(
      scheduler.GetNumOtherThreads() > 0,
      "AsyncScene requires at least one worker thread. We should have caught this earlier.");

  // Increment the TaskSemaphore since we're starting the async task
  _simLoopTaskComplete.Add();

  // Start a task which will create the Scene, call SimulationLoop(), and finally destroy the scene.
  // This task will be woken up many times, but it will always be the same task on the same worker
  // thread. Copy TaskSemaphores to the lambda by value as always (see explanation comment in
  // task_scheduler.h).
  TaskSemaphore signalReady(1);
  scheduler.AddTask(
      "SimulationLoop", [this, signalDone = this->_simLoopTaskComplete, signalReady]() {
        MOCHI_DEFER(signalDone.Done());

        MOCHI_ASSERT(
            TaskScheduler::IsCurrentThreadAWorker(),
            "AsyncScene simulation task must run on a Mochi worker thread.");

        // Create and destroy the scene on the thread that will use it
        this->_scene = assert_cast<SceneImpl*>(_context->CreateScene(_name));

        // Initialize before signalReady so IsSimThread() is valid when CreateAsyncScene() returns.
        _simThreadId.Store(std::this_thread::get_id());
        signalReady.Done();
        MOCHI_DEFER(_context->DestroyScene(_scene); _scene = nullptr;);

        SimulationLoop();
      });

  // Wait for the scene to be created
  signalReady.Wait();
  MOCHI_ASSERT(_scene != nullptr);
}

void AsyncSceneImpl::StopSimLoopTask() {
  // Tell SimulationLoop() to return
  {
    TaskLock lock(_mutex);
    _wasStopRequested = true;
    _wasWakeRequested = true;
  }
  _wakeCondition.notify_one();

  // Wait until it does
  _simLoopTaskComplete.Wait();
}

void AsyncSceneImpl::SimulationLoop() {
  // This function is the body of the simulation thread.
  // It does not return until the AsyncScene is shut down.

  // Give this worker thread a special name in the debugger
  marl::Thread::setName("Mochi AsyncScene");

  bool wasPauseAfterStepRequested = false;
  bool wasStopRequested = false;
  std::vector<CommandFn> commands;

  // The system_clock used by marl::Scheduler, and by extension TaskConditionVariable::wait_until.
  // It should have plenty of accuracy for measuring simulation steps.
  using Clock = std::chrono::system_clock;
  Clock::time_point lastStepTime = Clock::now();
  Clock::time_point nextWakeTime = lastStepTime;

  while (!wasStopRequested) {
    MOCHI_PROFILE_SCOPE_N("SimulationLoop");

    int numStepsRemaining = 0;
    double timeStepSec = 0.0;

    // Synchronization Block:
    {
      MOCHI_PROFILE_SCOPE_N("WaitForWake");

      // Sleep until there is something to do
      TaskLock lock(_mutex);

      // Sleep until it is time for the next step, or when explicitly
      // requested.
      _wakeCondition.wait_until(
          lock, nextWakeTime, [&] { return _wasWakeRequested || (Clock::now() >= nextWakeTime); });

      // Determine if we should step (and how many times)
      bool willPause = _shouldPause && !_wasPauseAfterStepRequested;
      if (!_wasStopRequested && !willPause) {
        if ((_wasStepRequested && _wasPauseAfterStepRequested) ||
            (!_shouldPause && (Clock::now() >= nextWakeTime))) {
          numStepsRemaining = 1;
        }
      }

      // Calculate the time step
      if (numStepsRemaining != 0) {
        double callbackStepSec = 0.0;
        if (_stepParams.timeStepCallback) {
          callbackStepSec = _stepParams.timeStepCallback();
          numStepsRemaining = 1;
        }

        if (_stepParams.timeStepCallback && callbackStepSec >= 0.0) {
          timeStepSec = callbackStepSec;
        } else if (_stepParams.useFixedTimeStep || _wasPauseAfterStepRequested) {
          timeStepSec = _stepParams.fixedTimeStepSeconds / (double)numStepsRemaining;
        } else {
          Clock::time_point now = Clock::now();
          Clock::duration maxDeltaTime =
              mochi::TimeSpanFromSeconds<Clock>(_stepParams.dynamicTimeStepMaxSeconds);
          Clock::duration deltaTime = std::min(now - lastStepTime, maxDeltaTime);
          lastStepTime = now;
          nextWakeTime =
              now + mochi::TimeSpanFromSeconds<Clock>(_stepParams.dynamicTimeStepMinSeconds);
          timeStepSec = mochi::ToSeconds<Clock>(deltaTime) / (double)numStepsRemaining;
        }
      } else if (willPause) {
        // When paused, set nextWakeTime to avoid spinning on wait_until with a past time
        nextWakeTime =
            Clock::now() + mochi::TimeSpanFromSeconds<Clock>(_stepParams.dynamicTimeStepMaxSeconds);
      }

      // Capture state changes while _mutex is still locked
      _isPaused = willPause;
      wasPauseAfterStepRequested = _wasPauseAfterStepRequested;
      wasStopRequested = _wasStopRequested;
      commands.swap(_commandQueue);

      // Reset request flags
      _wasWakeRequested = false;
      _wasStepRequested = false;
      _wasStopRequested = false;
      _wasPauseAfterStepRequested = false;
    }

    // Execute queued commands in order
    for (auto& fn : commands) {
      MOCHI_PROFILE_SCOPE_N("ExecuteQueuedCommand");
      fn(_scene);
    }
    commands.clear();

    // Advance the simulation and fire any callbacks registered with the Scene
    if (numStepsRemaining > 0) {
      for (; numStepsRemaining > 0; --numStepsRemaining) {
        _scene->Step(timeStepSec);
      }
    } else {
      // If a debugger is attached and we are not stepping the scene, then update the debugger
      // manually so it can respond to network messages and interact with the scene on this thread.
      _scene->UpdateDebugger();
    }

    if (wasPauseAfterStepRequested) {
      // The next step will be paused.
      Pause(true);
    }
  }
}

// TODO: Detect whether the current Marl task/fiber is this AsyncScene's simulation task. Comparing
// OS worker-thread IDs may give false positives for any other task/fiber running on the same
// worker. This can make IsPaused() return the actual state instead of the requested state and make
// WaitForQueuedCommands() skip a safe cross-scene wait.
bool AsyncSceneImpl::IsSimThread() const {
  return std::this_thread::get_id() == _simThreadId.Load();
}

void AsyncSceneImpl::QueueCommand(CommandFn callback) {
  {
    TaskLock lock(_mutex);
    _commandQueue.push_back(std::move(callback));
    _wasWakeRequested = true;
  }
  _wakeCondition.notify_one();
}

void AsyncSceneImpl::QueueActorCommand(
    ActorHandle actorHandle,
    std::function<void(Actor*)> callback) {
  if (actorHandle.IsValid()) {
    QueueCommand([actorHandle, callback = std::move(callback)](Scene* scene) {
      Actor* actor = scene->GetActor(actorHandle);
      if (actor) {
        callback(actor);
      } else {
        MOCHI_LOG_WARNING(
            "QueueActorCommand called with invalid ActorHandle (%llu).", actorHandle.value);
      }
    });
  }
}

void AsyncSceneImpl::WaitForQueuedCommands() {
  if (IsSimThread())
    MOCHI_UNLIKELY {
      MOCHI_LOG_ERROR_ONCE(
          "AsyncScene::WaitForQueuedCommands must not be called from the simulation thread (would self-deadlock). "
          "Returning without waiting. Previously queued commands may not have executed yet. Do not rely on their effects.");
      return;
    }

  // Queue a new command that will set the stack variable 'done' to true.
  // Commands are executed in order, so any previously queued commands will
  // execute first.
  bool done = false;
  QueueCommand([&](Scene*) {
    {
      // For correct use of condition_variable, done has to be modified while
      // holding the mutex lock. It cannot simply be an atomic<bool>.
      TaskLock lock(_commandQueueWaitMutex);
      done = true;
    }
    // Notify the waiting thread(s) AFTER releasing the mutex lock, so that the
    // thread that wakes can immediately acquire the lock.
    _commandQueueWaitCondition.notify_all();
  });
  {
    // WARNING: This call to condition_variable::wait only waits for 'done'. It
    // does not guarantee that the condition_variable::notify_all() was called.
    // That is why _commandQueueWaitCondition is a member variable and not a
    // temporary variable on the stack. It needs to live long enough for the
    // worker thread to finish calling notify_all(), which might happen after
    // the return of this function (which is OK).
    TaskLock lock(_commandQueueWaitMutex);
    _commandQueueWaitCondition.wait(lock, [&]() { return done; });
  }
}

CallbackHandle AsyncSceneImpl::RegisterPreStepCallback(
    std::string_view debugName,
    OnStepFn callback,
    int priority) {
  return _scene->RegisterPreStepCallback(debugName, std::move(callback), priority); // thread-safe
}

CallbackHandle AsyncSceneImpl::RegisterPostStepCallback(
    std::string_view debugName,
    OnStepFn callback,
    int priority) {
  return _scene->RegisterPostStepCallback(debugName, std::move(callback), priority); // thread-safe
}

void AsyncSceneImpl::CancelCallback(CallbackHandle handle) {
  _scene->CancelCallback(handle);
}

QueryHandle
AsyncSceneImpl::RegisterActorQuery(ActorHandle actorHandle, QueryType type, bool forceCompute) {
  if (!actorHandle.IsValid()) {
    return QueryHandle{};
  }

  // Allocate a new QueryHandle immediately (thread-safe)
  QueryHandle newQueryHandle = _scene->NewQueryHandle(type);

  // Queue registration using this new handle
  QueueCommand([actorHandle, newQueryHandle, forceCompute](Scene* scene) {
    Error error;
    assert_cast<SceneImpl*>(scene)->RegisterActorQuery(
        scene->GetActor(actorHandle), newQueryHandle, forceCompute, error);
    if (!error.IsOK()) {
      MOCHI_LOG_WARNING(
          "RegisterActorQuery failed for QueryType %d. Reason: %s",
          static_cast<int>(GetQueryType(newQueryHandle)),
          error.ToString().c_str());
    }
  });

  return newQueryHandle;
}

void AsyncSceneImpl::CancelActorQuery(ActorHandle actor, QueryHandle handle) {
  QueueActorCommand(actor, [handle](Actor* actor) { actor->CancelQuery(handle); });
}

AsyncStepParams AsyncSceneImpl::GetAsyncStepParams() const {
  TaskLock lock(_mutex);
  return _stepParams; // Return a copy
}

void AsyncSceneImpl::SetAsyncStepParams(AsyncStepParams const& params, Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.dynamicTimeStepMinSeconds) && IsFinite(params.dynamicTimeStepMaxSeconds),
      error,
      "AsyncStepParams dynamic time-step bounds must be finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.fixedTimeStepSeconds) && params.fixedTimeStepSeconds >= 0.0,
      error,
      "AsyncStepParams::fixedTimeStepSeconds must be finite and non-negative.");
  MOCHI_ERROR_RETURN(error);

  AsyncStepParams newParams = params;

  // Clamp values for consistency
  if (newParams.dynamicTimeStepMaxSeconds < 0.0) {
    newParams.dynamicTimeStepMaxSeconds = 0.0;
  }
  if (newParams.dynamicTimeStepMinSeconds < 0.0) {
    newParams.dynamicTimeStepMinSeconds = 0.0;
  }
  if (newParams.dynamicTimeStepMinSeconds > newParams.dynamicTimeStepMaxSeconds) {
    newParams.dynamicTimeStepMinSeconds = newParams.dynamicTimeStepMaxSeconds;
  }

  // Synchronize
  TaskLock lock(_mutex);
  _stepParams = newParams;
}

void AsyncSceneImpl::Pause(bool shouldPause) {
  {
    TaskLock lock(_mutex);
    _shouldPause = shouldPause;
  }
  if (!shouldPause) {
    // Make sure the sim thread wakes up to observe this change in pause state.
    _wakeCondition.notify_one();
  }
}

bool AsyncSceneImpl::IsPaused() const {
  if (IsSimThread()) {
    // If called on the simulation thread, return the actual pause state. This
    // ensures that the state of IsPaused() does not appear to change during the
    // simulation step.
    return _isPaused;
  } else {
    // If called on any other thread, then return the most recently requested
    // pause state as so to present a consistent Pause/IsPaused state to the
    // calling thread.
    TaskLock lock(_mutex);
    return _shouldPause;
  }
}

void AsyncSceneImpl::RequestStepThenPause() {
  {
    TaskLock lock(_mutex);

    // We set a flag to indicate the request. If someone requests steps faster
    // than we can actually simulate, then steps will be skipped (not queued up
    // indefinitely)
    _wasPauseAfterStepRequested = true;
    _wasStepRequested = true;
    _wasWakeRequested = true;
  }
  _wakeCondition.notify_one();
}

SceneImpl* AsyncSceneImpl::GetSceneImplUnsafe() {
  return _scene;
}

} // namespace mochi
