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

#include <mochi_physics/cpp_api/mochi_debug_server.h>

#include <memory>
#include <string_view>

// Forwards
namespace mochi {
class Context;
class Scene;
} // namespace mochi

// Forwards
namespace mochi::net {
class MessageServer;
} // namespace mochi::net

namespace mochi::dbg {

/** @brief Extends DebugServer API for internal use within mochi_physics and tests.  */
class DebugServerInternal : public DebugServer {
 public:
  virtual ~DebugServerInternal() = default;

  /** @brief Start accepting in-process connections (for unit tests) */
  virtual void StartInProc() = 0;

  /**
   * @brief Called by @ref Context when a new scene is being added to the context.
   *
   * @note See @ref protocol::SceneAddRemove for callback restrictions.
   */
  virtual void OnAddScene(Scene* scene) = 0;

  /**
   * @brief Called by @ref Context when a scene is being removed from the context.
   *
   * @note See @ref protocol::SceneAddRemove for callback restrictions.
   */
  virtual void OnRemoveScene(Scene* scene) = 0;

  /**
   * @brief FOR UNIT TESTS ONLY: Get the internal @ref net::MessageServer so that clients can
   * connect to this server after @ref StartInProc.
   *
   * @warning In-process callbacks may execute synchronously on the sending thread. See
   * @ref protocol::SceneAddRemove for restrictions on scene-add/remove notifications.
   */
  virtual net::MessageServer& GetMessageServer_ForTestingOnly() = 0;
};

/**
 * @brief Create a DebugServer implementation.
 *
 * @param[in] context Pointer to the context. Must remain valid for the lifetime of the debug
 * server.
 * @return A new instance of DebugServerInternal.
 */
std::unique_ptr<DebugServerInternal> CreateDebugServer(Context* context);

} // namespace mochi::dbg
