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
#include <mochi_physics/dbg/protocol.h>

#include <string>

namespace mochi {
class SceneImpl;
} // namespace mochi

namespace mochi::dbg {

// The arguments of a CommandRequest as std::string, which the SReflect console functions take.
DynamicArray<std::string> GetCommandArgs(protocol::CommandRequest const& request);

// A path inside a scene can go through the built-in directories actors/<actor name> and, in an
// actor with nested actors, children/<child name>. They hide reflected members with the same name.

// List, read, set or call the member at the request's path inside the scene, which starts at
// request.pathStart. Call on the scene's thread. It runs inside the scene's step hooks, so console
// members must not step the scene, pump the debugger, or destroy the scene or its actors.
protocol::CommandReply ExecuteSceneCommand(
    SceneImpl& scene,
    protocol::CommandRequest const& request);

} // namespace mochi::dbg
