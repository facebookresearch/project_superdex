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
#include <string_view>
#include <unordered_map>

namespace mochi {
class SceneImpl;
} // namespace mochi

namespace mochi::dbg {

// The arguments of a CommandRequest as std::string, which the SReflect console functions take.
DynamicArray<std::string> GetCommandArgs(protocol::CommandRequest const& request);

// Return name with each character that would need quotes in a console path or in the terminal
// replaced by '_': '/', quotes, spaces and control characters, and a leading '[' or '{', which
// starts a bracket group in the terminal. Return an empty string for "." and "..".
std::string SanitizeConsoleName(std::string_view name);

// The next number to try for each base name that MakeUniqueDebugName has found taken.
using DebugNameSuffixes = std::unordered_map<std::string, int>;

// Return base if it is free. Otherwise return base followed by the first free number from
// nextSuffixes[base] up, which starts at 2, and store the number after it, so the 100th "box"
// starts at box100 instead of trying box2 to box99.
template <class IsTakenFn>
std::string MakeUniqueDebugName(
    std::string const& base,
    DebugNameSuffixes& nextSuffixes,
    IsTakenFn const& isTaken) {
  if (!isTaken(base)) {
    return base;
  }
  int& suffix = nextSuffixes.try_emplace(base, 2).first->second;
  std::string name;
  do {
    name = base + std::to_string(suffix++);
  } while (isTaken(name));
  return name;
}

// DebugNameSuffixes for each directory of actors, keyed by the parent's handle (invalid for the
// top-level actors).
using ActorDebugNameSuffixes = std::unordered_map<ActorHandle, DebugNameSuffixes>;

// Give each actor without a CDebugName one, unique within its directory (the top-level actors, or
// the children of one parent). The base is the sanitized name, only its local part after the last
// '/' for a child, or the actor type, such as Rigid, if that is empty. A taken base gets a number
// from MakeUniqueDebugName. Names never change, and the scene debugger removes them and clears
// nextSuffixes when it shuts down. Call on the scene's thread before reading CDebugName.
void AssignDebugNames(SceneImpl& scene, ActorDebugNameSuffixes& nextSuffixes);

// A path inside a scene can go through the built-in directories actors/<debug name> and, in an
// actor with nested actors, children/<debug name>. They hide reflected members with the same
// name.

// List, read, set or call the member at the request's path inside the scene, which starts at
// request.pathStart. Call on the scene's thread. It runs inside the scene's step hooks, so console
// members must not step the scene, pump the debugger, or destroy the scene or its actors.
protocol::CommandReply ExecuteSceneCommand(
    SceneImpl& scene,
    ActorDebugNameSuffixes& nextSuffixes,
    protocol::CommandRequest const& request);

} // namespace mochi::dbg
