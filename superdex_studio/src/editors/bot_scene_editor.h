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

#include "editors/asset_editor.h"
#include "rendering/scene_stage.h"
#include "rendering/stats.h"
#include "simulation/mochi_async_scene.h"
#include "simulation/physics_drag_controller.h"
#include "simulation/skin_sim_sync.h"

#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/scene/bot_task.h>

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace superdex::studio {

class BotSceneAsset;

// Editor for .superdex_scene and .superdex_scene_archive files.
// Visualizes and edits the base scene, spawnable prefabs, and placed bots, and can simulate them
// (play/step/stop). Edits are undoable and restage the viewport live.
class BotSceneEditor : public AssetEditor {
 public:
  //------------------------------------------------------------------------------------------------
  // AssetEditor
  //------------------------------------------------------------------------------------------------

  BotSceneEditor(SuperDexStudio* studio, BotSceneAsset* asset);
  void Initialize() override;
  void OnHandleInputs() override;
  void OnRender(Renderer const* renderer) override;
  void Shutdown() override;
  void OnActivate() override;
  void OnDeactivate() override;
  void Refresh() override;
  void ShowTabContents() override;
  static std::vector<WindowDeclaration> GetDefaultWindows();
  std::vector<WindowDeclaration> GetAuxiliaryWindows() const override;
  void ShowAuxiliaryWindows() override;
  void ShowMainMenuItems() override;
  bool CanUndoRedo() const override;
  void ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) override;
  void OnAppSettingsChanged(AppSettings const& settings) override;
  Viewport* GetViewport() override {
    return _viewport.get();
  }

  // Loads @p path as the previewed task, as "Load Task..." does, and brings up the Task Preview
  // window. Refused while simulating, when that button is disabled too.
  bool OpenTask(mochi::Path const& path);

 private:
  //------------------------------------------------------------------------------------------------
  // Undo/Redo
  //------------------------------------------------------------------------------------------------

  std::string TakeUndoSnapshot() const;
  void RestoreUndoSnapshot(std::string const& json, int selectionIndex);

  //------------------------------------------------------------------------------------------------
  // Staging
  //------------------------------------------------------------------------------------------------

  void RestageBotScene();
  // The viewport Stats overlay rows for what RestageBotScene stages.
  StatsText BuildStats() const;
  // Resolves @p task against the current scene and loads its prefabs. On failure, @p spawns is
  // empty and @p error says why.
  bool BindTaskSpawns(
      superdex::robotics::BotTaskPrefab const& task,
      std::vector<ResolvedTaskSpawn>& spawns,
      std::string& error);
  bool LoadTask(mochi::Path const& path);
  void ClearTask();
  // Re-reads the loaded task from the scene's task cache, whose values undo restores.
  void SyncTaskFromCache();
  // Writes the edited task to the cache and restages, marks dirty and records undo like a scene
  // edit. Structural edits (add/remove/reorder) get their own undo entry.
  void OnTaskEdited(bool structural);
  // Asks for a new file, copies the task there with its unsaved edits, and switches to the copy.
  void SaveTaskAs();

  //------------------------------------------------------------------------------------------------
  // Mochi Scene
  //------------------------------------------------------------------------------------------------

  bool CanSimulate() const;
  AssetSceneOverrides GetAssetSceneOverrides() const;
  void CreatePhysicsActors(mochi::Scene* scene);
  void DestroyPhysicsActors(mochi::Scene* scene);
  mochi::CallbackHandle RegisterPostStepCallback(mochi::AsyncScene* scene);
  void OnStartPhysics();
  // Why @p path cannot be saved as a reference from this scene, or empty if it can.
  std::string const& GetReferenceProblem(std::string_view path);
  void OnStopPhysics();
  void SyncFromPhysics();

  //------------------------------------------------------------------------------------------------
  // ImGui
  //------------------------------------------------------------------------------------------------

  void ShowInfoWindow(bool* open);
  void ShowTaskWindow(bool* open);
  // The editable spawn list, shown while not simulating.
  void ShowTaskEditor();
  struct SlotEditResult;
  // The Slots section of the Bot Scene Info window.
  SlotEditResult ShowSlots(superdex::robotics::BotScenePrefab& prefab);

 private:
  // target asset
  BotSceneAsset* _sceneAsset = nullptr;
  // render scene state
  std::unique_ptr<Viewport> _viewport;
  SceneStage _stage;
  // The stage's render and collision stats, taken after each staging at the rest pose.
  StageStats _stageStats;
  StageType _stageType = StageType::RenderModelFallbackToMochiModel;
  // physics scene state
  MochiAsyncScene _mochiScene;
  std::optional<superdex::robotics::BotScene> _botScene;
  std::vector<mochi::ActorHandle> _physicsActors;
  // Bot prefabs rebuilt and copied from the AssetManager by OnStartPhysics, keyed by bot path, for
  // CreatePhysicsActors. Rebuilding mutates the asset, so it must happen on the UI thread.
  std::unordered_map<std::string, superdex::robotics::BotPrefab> _physicsBotPrefabs;
  // GetReferenceProblem results by path; cleared on activation in case a .superdex_root changed.
  std::unordered_map<std::string, std::string> _referenceProblems;
  // The scene copied by OnStartPhysics, with unsaved file-backed controller params inlined.
  superdex::robotics::BotScenePrefab _physicsScenePrefab;
  struct TaskRuntimeSpawn {
    std::optional<mochi::ActorHandle> primaryActor;
    mochi::TransformRT spawnFromPrimary = {};
  };
  std::vector<TaskRuntimeSpawn> _taskRuntimeSpawns;
  std::vector<superdex::robotics::Bot*> _taskBots;
  std::vector<mochi::ActorHandle> _taskPrefabActors;
  std::vector<mochi::ConstraintHandle> _taskPrefabConstraints;
  std::unique_ptr<PhysicsDragController> _dragController;
  // Whether each staged actor is a skin, captured by OnStartPhysics for CreatePhysicsActors.
  std::vector<bool> _stagedSkinRows;
  // One per simulated articulated actor whose skin is staged, so the staged skin follows it.
  std::vector<SkinSimSync> _skinSyncs;
  struct SimData {
    // In staged-actor order: a skin's row follows its articulation's links.
    std::vector<std::string> actorNames;
    std::vector<mochi::TransformRT> actorTransforms;
    std::vector<mochi::TransformRT> taskTransforms;
    std::vector<SoftMeshUpdate> skinSurfaces;
    std::vector<SkinnedPoseUpdate> skinPoses;
  };
  mochi_renderer::ProducerConsumerBuffer<SimData> _simData;
  mochi::Path _taskPath;
  std::optional<superdex::robotics::BotTaskPrefab> _taskPrefab;
  std::vector<ResolvedTaskSpawn> _taskSpawns;
  // Why _taskPrefab no longer binds to the edited scene; its spawns are not staged meanwhile.
  std::string _taskBindError;
  // Why the most recent Load/Replace Task attempt failed; the previous task, if any, stays loaded.
  std::string _taskLoadError;
  // Why the most recent Save Task failed.
  std::string _taskSaveError;
  bool _showCurrentTaskTransforms = false;
  // The spawn whose Name field is being edited: its name and the indices of its children when
  // editing started.
  struct SpawnRename {
    std::string name;
    std::vector<int> children;
  };
  SpawnRename _spawnRename;
  // The bot whose Name field is being edited: its name and the (slot, entry) indices that named it
  // when editing started.
  struct BotRename {
    std::string name;
    std::vector<std::pair<int, int>> slotEntries;
    // Slots in _activeSlotBots whose active bot it was.
    std::vector<std::string> activeSlots;
  };
  BotRename _botRename;
  // The bot each slot includes in the preview and in simulation, by slot name; a slot without one,
  // or whose bot it no longer lists, includes its first bot. A view setting: not saved or undone.
  std::unordered_map<std::string, std::string> _activeSlotBots;
  // The slot whose Name field is being edited: its name and active bot, and _activeSlotBots, when
  // editing started.
  struct SlotRename {
    std::string name;
    std::string activeBot;
    std::unordered_map<std::string, std::string> activeSlotBots;
  };
  SlotRename _slotRename;
  // Set by OpenTask: brings the Task Preview tab forward the next time it is drawn.
  bool _focusTaskWindow = false;
};

} // namespace superdex::studio
