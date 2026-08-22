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

#include "mochi_enums.h"
#include "mochi_handle.h"
#include "mochi_scene.h"
#include "mochi_structs.h"

/********************************************************************************
 IMPORTANT: PLEASE KEEP HEADER INCLUDES TO A MINIMUM.
    If you must include a mochi_core header, then please make sure that it only
    declares the data types (not containing other implementation details).
*********************************************************************************/
#include <mochi_core/utils/error.h>

#include <functional>
#include <string_view>

namespace mochi {

class Actor;
class Context;

class AsyncScene {
 public:
  [[nodiscard]] virtual Context* GetContext() = 0;

  virtual void QueueCommand(std::function<void(Scene*)> callback) = 0;

  virtual void QueueActorCommand(ActorHandle actor, std::function<void(Actor*)> callback) = 0;

  virtual void WaitForQueuedCommands() = 0;

  [[nodiscard]] virtual CallbackHandle RegisterPreStepCallback(
      std::string_view debugName,
      std::function<void(StepInfo const&)> callback,
      int priority = Scene::kDefaultCallbackPriority) = 0;

  [[nodiscard]] virtual CallbackHandle RegisterPostStepCallback(
      std::string_view debugName,
      std::function<void(StepInfo const&)> callback,
      int priority = Scene::kDefaultCallbackPriority) = 0;

  virtual void CancelCallback(CallbackHandle handle) = 0;

  virtual QueryHandle
  RegisterActorQuery(ActorHandle actor, QueryType type, bool forceCompute = false) = 0;

  virtual void CancelActorQuery(ActorHandle actor, QueryHandle handle) = 0;

  [[nodiscard]] virtual AsyncStepParams GetAsyncStepParams() const = 0;

  virtual void SetAsyncStepParams(AsyncStepParams const& params, Error& error) = 0;

  virtual void Pause(bool shouldPause) = 0;

  [[nodiscard]] virtual bool IsPaused() const = 0;

  virtual void RequestStepThenPause() = 0;

 protected:
  /// Don't delete the AsyncScene pointer. Call @ref Context::DestroyAsyncScene.
  virtual ~AsyncScene() = default;
};

} // namespace mochi
