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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/cpp_api/mochi_handle.h>
#include <mochi_physics/dbg/protocol.h>

#include <functional>
#include <memory>

namespace mochi {
class SceneImpl;
} // namespace mochi

namespace mochi::net {
using ClientId = uint32_t;
struct Message;
class MessageServer;
} // namespace mochi::net

namespace mochi::dbg {

class SceneDebugger {
 public:
  using ClockFn = std::function<double()>;

  // Call this periodically on the scene's controlling thread, when it is safe to
  // access and modify the scene contents.
  virtual void UpdateOnSceneThread(SceneImpl* scene) = 0;

  // Check the current step mode (modified by queued messages). Thread-safe.
  virtual StepMode GetStepMode() const = 0;

  // Check the current pause state (modified by queued messages). Thread-safe.
  bool IsPaused() const {
    return GetStepMode() == StepMode::Pause;
  }

  // Override the clock used by debugger timing. Clock values are elapsed seconds. Pass an empty
  // function to restore the default clock.
  virtual void SetClock(ClockFn clock) = 0;

  // Called by DebugServer when messages for this scene arrive. Thread-safe.
  virtual void OnReceiveAsync(std::unique_ptr<net::Message> msg) = 0;

  // Called on the DebugServer's thread before the SceneDebugger is destroyed.
  // It will no longer function after this point.
  virtual void PreShutdownAsync(SceneImpl* scene) = 0;

  // Finish shutting down this SceneDebugger on the scene's controlling thread. Must come after
  // PreShutdown. If the scene is being destroyed, then it is OK to skip this.
  virtual void ShutdownOnSceneThread(SceneImpl* scene) = 0;

  virtual ~SceneDebugger() = default;
};

std::shared_ptr<dbg::SceneDebugger>
CreateSceneDebugger(net::ClientId client, SceneImpl* scene, net::MessageServer& server);

} // namespace mochi::dbg
