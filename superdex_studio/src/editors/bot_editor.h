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
#include "editors/bot_editor_contact.h"
#include "editors/bot_editor_control.h"
#include "rendering/bot_visualization.h"
#include "rendering/scene_stage.h"
#include "rendering/viewport.h"
#include "simulation/mochi_async_scene.h"
#include "simulation/physics_drag_controller.h"
#include "ui/imgui_widgets.h"

#include <superdex_robotics/superdex_robotics.h>

#include <mochi_renderer/buffer.h>

namespace superdex::studio {

class BotAsset;
class SuperDexStudio;

class BotEditor : public AssetEditor {
 public:
  //------------------------------------------------------------------------------------------------
  // AssetEditor
  //------------------------------------------------------------------------------------------------

  BotEditor(SuperDexStudio* studio, BotAsset* asset);
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

 private:
  //------------------------------------------------------------------------------------------------
  // Undo/Redo
  //------------------------------------------------------------------------------------------------

  std::string TakeUndoSnapshot() const;
  void RestoreUndoSnapshot(std::string const& json, int selectionIndex);
  int GetUndoSelectionIndex() const;

  //------------------------------------------------------------------------------------------------
  // Bot Instance
  //------------------------------------------------------------------------------------------------

  void RestageBot();
  void AddBotLink(int iLink);
  void DeleteBotLink(int iLink);
  void ReparentBotLink(int linkIdx, int newParentIdx, bool preserveWorldTransform = true);

  //------------------------------------------------------------------------------------------------
  // Viewport
  //------------------------------------------------------------------------------------------------

  void DrawBotVisualizations(mochi_renderer::Scene* renderScene) const;
  void OnSceneSelectionChanged(std::vector<mochi_renderer::SceneObject*> const& objects);

  //------------------------------------------------------------------------------------------------
  // Mochi Scene
  //------------------------------------------------------------------------------------------------

  bool CanSimulate() const;
  void CreatePhysicsActors(mochi::Scene* scene);
  void DestroyPhysicsActors(mochi::Scene* scene);
  mochi::CallbackHandle RegisterPreStepCallback(mochi::AsyncScene* scene);
  mochi::CallbackHandle RegisterPostStepCallback(mochi::AsyncScene* scene);
  void OnStopPhysics();
  void SyncFromPhysics();

  //------------------------------------------------------------------------------------------------
  // ImGui
  //------------------------------------------------------------------------------------------------

  void ShowBotHierarchyWindow(bool* open);
  void ShowBotTreeSettingsCog();
  ImGuiWindowFlags GetBotWindowFlags() const;
  void ShowBotDetailsWindow(bool* open);
  void ShowBotLinkDetailsWindow(bool* open);
  void ApplyBotParamsEdit();
  bool ShowBotJointEditorWidgets(superdex::robotics::BotJointPrefab& joint, bool isRoot);
  bool ShowBotLinkEditorWidgets(
      superdex::robotics::BotLinkPrefab& link,
      AssetManager const& assetManager,
      bool acceptDragDropPayload,
      bool& modelChanged);
  bool ShowBotPrefabEditorWidgets(superdex::robotics::BotPrefab& botPrefab);
  bool ShowModBotPrefabEditorWidgets(
      superdex::robotics::ModBotPrefab& modBotPrefab,
      AssetManager const& assetManager);
  // Mod bot recipe build status. ComputeFirstFailingModIndex sets outIsBuildError
  // to distinguish a hard build failure (chain stops; mods below have no effect)
  // from a validation failure (chain completes but the built bot is invalid).
  int ComputeFirstFailingModIndex(
      superdex::robotics::ModBotPrefab const& params,
      bool& outIsBuildError);
  bool RecomputeModBuildStatus();
  // Bot Contact
  void ShowBotContactWindow(bool* open);
  // Bot Transmissions
  void ShowBotTransmissionsWindow(bool* open);

  //------------------------------------------------------------------------------------------------
  // Batch Rename Links & Joints
  //------------------------------------------------------------------------------------------------

  // Nested batch-rename UI state, specialized for renaming a bot's link and joint names. The
  // transform params, input controls, and preview table are shared via <ui/imgui_widgets.h>; the
  // entry gathering, name-uniqueness validation, and data-model apply stay here.
  struct BatchRenameState {
    bool open = false;
    // Single input set shared by both tabs, so switching tabs preserves the settings.
    BatchRenameParams inputs;
    // Snapshot of the renameable names, refreshed on open and after each apply.
    std::vector<std::string> linkNames;
    std::vector<std::string> jointNames;
  };
  // Snapshot the current renameable link/joint names and request the modal to open.
  void OpenBatchRenameLinksJointsModal();
  // Refresh the link/joint name snapshots from the live bot/mod-bot data.
  void SnapshotBatchRenameNames();
  // Pumped once per frame; renders the modal, computes previews, validates, and applies.
  void ShowBatchRenameLinksJointsModal();
  // Renders the active tab's inputs and its live old->new preview table.
  void ShowBatchRenameTabContents(
      BatchRenameParams& inputs,
      std::vector<std::string> const& originalNames);
  // Applies the active category (links or joints) to the live bot/mod-bot data, then rebuilds.
  void ApplyBatchRename(bool applyLinks);
  // Computes new names for every entry and flags invalid rows (empty or duplicate within the
  // category). Returns the number of names that changed.
  static int ComputeBatchRenamePreview(
      std::vector<std::string> const& originalNames,
      BatchRenameParams const& inputs,
      std::vector<std::string>& outNewNames,
      std::vector<bool>& outRowInvalid);

 private:
  // target asset
  BotAsset* _botAsset = nullptr;
  // render scene state
  std::unique_ptr<Viewport> _viewport;
  SceneStage _stage;
  StageType _stageType = StageType::RenderModelOnly;
  // physics scene state
  MochiAsyncScene _mochiScene;
  superdex::robotics::Bot* _bot = nullptr;
  std::unique_ptr<PhysicsDragController> _dragController;
  struct SimData {
    std::vector<std::string> linkNames;
    std::vector<mochi::TransformRT> linkTransforms;
    mochi::DynamicArray<float> transmissionDisplacements;
  };
  mochi_renderer::ProducerConsumerBuffer<SimData> _simData;
  // UI state
  int _selectedBotLinkIndex = -1;
  bool _forceLinkFocus = false;
  // Index of the first mod bot modification that fails to build, or
  // superdex::robotics::kIndexNone when the recipe builds and validates cleanly. Cached;
  // recomputed only when the recipe is edited (never per-frame).
  int _firstFailingModIndex = superdex::robotics::kIndexNone;
  // Whether the failure at _firstFailingModIndex is a hard build failure (which
  // stops the modification chain, so mods below are dimmed). False for a
  // validation failure, where subsequent mods still take effect.
  bool _firstFailingIsBuildError = false;
  // Error/issue message from the failing modification (for hover tooltip).
  mochi::DynamicString _modBuildError;
  // bot visualization visibility, toggled from the viewport Show menu. Appearance lives in
  // AppSettings::botVisualization.
  BotVisualizationFlags _botVizFlags;
  mochi::DynamicArray<float> _transmissionDisplacements;
  mochi::DynamicArray<bool> _linearTransmissionExpanded;
  mochi::DynamicArray<bool> _spatialTendonExpanded;
  // bot control window data
  BotControl _botControl;
  // bot contact window data
  BotContactEstimator _contactEstimator;
  // batch rename dialog state
  BatchRenameState _batchRename;
};

} // namespace superdex::studio
