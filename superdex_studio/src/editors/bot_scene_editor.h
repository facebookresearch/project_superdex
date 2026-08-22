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

#if MOCHI_INTERNAL

#include "editors/asset_editor.h"
#include "rendering/scene_stage.h"
#include "simulation/mochi_async_scene.h"
#include "simulation/physics_drag_controller.h"

#include <superdex_robotics/internal/bot_scene.h>

#include <optional>
#include <string>
#include <vector>

namespace superdex::studio {

class BotSceneAsset;

// Editor for .mochi_bot_scene files (and a read-only viewer for .mochi_bot_scene_archive files).
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
  bool CanUndoRedo() const override;
  void ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) override;
  void OnAppSettingsChanged(AppSettings const& settings) override;
  Viewport* GetViewport() override {
    return _viewport.get();
  }

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

  //------------------------------------------------------------------------------------------------
  // Mochi Scene
  //------------------------------------------------------------------------------------------------

  bool CanSimulate() const;
  AssetSceneOverrides GetAssetSceneOverrides() const;
  void CreatePhysicsActors(mochi::Scene* scene);
  void DestroyPhysicsActors(mochi::Scene* scene);
  mochi::CallbackHandle RegisterPostStepCallback(mochi::AsyncScene* scene);
  void OnStopPhysics();
  void SyncFromPhysics();

  //------------------------------------------------------------------------------------------------
  // ImGui
  //------------------------------------------------------------------------------------------------

  void ShowInfoWindow(bool* open);

 private:
  // target asset
  BotSceneAsset* _sceneAsset = nullptr;
  // render scene state
  std::unique_ptr<Viewport> _viewport;
  SceneStage _stage;
  StageType _stageType = StageType::RenderModelFallbackToMochiModel;
  // physics scene state
  MochiAsyncScene _mochiScene;
  std::optional<superdex::robotics::BotScene> _botScene;
  std::vector<mochi::ActorHandle> _physicsActors;
  std::unique_ptr<PhysicsDragController> _dragController;
  struct SimData {
    std::vector<std::string> actorNames;
    std::vector<mochi::TransformRT> actorTransforms;
  };
  mochi_renderer::ProducerConsumerBuffer<SimData> _simData;
};

} // namespace superdex::studio

#endif // MOCHI_INTERNAL
