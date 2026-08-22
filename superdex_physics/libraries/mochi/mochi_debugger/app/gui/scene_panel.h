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

#include "../viewport/render_scene.h" // MeshId

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_debugger/lib/debug_client.h>
#include <mochi_physics/cpp_api/mochi_handle.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace mochi::dbg {

struct UiState;

// Everything the debugger knows about one actor.
struct ActorInfo {
  ActorHandle parent; // Parent articulated actor; invalid if top-level.
  ActorType type = {};
  std::string name; // Actual actor name on server
  std::string displayName; // Modified for unnamed actors and for child actors.
  bool hasMesh = false; // Whether the actor has a renderable surface mesh.

  // Child actors in display order
  DynamicArray<ActorHandle> children;

  // User state. Automatic parent-skin visibility adapts until the user changes it explicitly.
  bool isVisible = true;
  bool visibilityOverridden = false;

  // Render mesh owned by this record. Mesh version and counts detect topology changes
  // (which require recreating a dynamic mesh); isRigid selects the static vs dynamic path.
  MeshId meshId = kInvalidMeshId;
  size_t nodeCount = 0;
  uint64_t meshVersionCounter = 0;
  bool isRigid = false;

  // Mark/sweep bookkeeping; only meaningful during a sync import.
  bool visited = false;
};

struct ScenePanelState {
  // Scene Selection
  DynamicArray<SceneInfo> scenes; // Current list; index 0 is always the invalid "None" entry.
  int selectedSceneIndex = 0; // Index into `scenes` (0 = the None entry) for the UI.

  // Scene Contents
  uint64_t lastSyncCounter = 0; // Last observed sync counter (change detection).
  bool hasSyncData = false; // True once sync data has been imported at least once.
  ActorHandle selectedActor; // Currently selected actor.

  // Every actor in the selected scene
  std::unordered_map<ActorHandle, ActorInfo> actors;

  // Top-level actors in display order (does not include nested children)
  DynamicArray<ActorHandle> sortedRootActors;

  // Set this after adding or removing an actor; the panel rebuilds the ordering on the next frame.
  bool treeDirty = false;

  // Clear all information about the current scene.
  void ResetSceneContents() {
    actors.clear();
    sortedRootActors.clear();
    selectedActor = {};
    treeDirty = false;
    hasSyncData = false;
    lastSyncCounter = 0;
  }
};

// Per-frame update of scene selection and cached sync data has moved to main.cpp (SyncSceneData),
// so that DebugClient::GetSceneSyncData is called from exactly one place per frame.

void BuildScenePanel(UiState& state);

} // namespace mochi::dbg
