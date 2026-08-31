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

#include "editors/bot_editor.h"
#include "app/app.h"
#include "ui/asset_browser.h"
#include "ui/imgui_widgets.h"

#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/math_utils.h>
#include "superdex_robotics/utils/archive_utils.h"

#include <mochi_physics/utils/mochi_prefab.h>

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/file_utils.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// Utils
//--------------------------------------------------------------------------------------------------

// Forward declarations (defined below).
static bool ValidateResultsHaveIssues(superdex::robotics::ValidateResults const& results);
static mochi::DynamicString FirstValidateIssue(superdex::robotics::ValidateResults const& results);

template <typename T>
bool CopyPasteContextMenu(char const* id, T& value) {
  bool changed = false;
  if (ImGui::BeginPopupContextItem(id, ImGuiPopupFlags_MouseButtonRight)) {
    if (ImGui::Selectable("Copy")) {
      ImGui::SetClipboardText(SReflect::ToJsonString(value, false).c_str());
    }
    if (ImGui::Selectable("Paste")) {
      char const* clipboard = ImGui::GetClipboardText();
      if (clipboard && clipboard[0] != '\0') {
        T pasted{};
        if (SReflect::FromJsonString(pasted, std::string(clipboard))) {
          value = std::move(pasted);
          changed = true;
        }
      }
    }
    ImGui::EndPopup();
  }
  return changed;
}

// Exports the current bot as a standalone Mochi prefab under <destDir>/<botName>/.
//
// The runtime exporter (mochi::prefab::ExportScene) populates shapes, joints, and contact filters
// but never emits render-model associations and rewrites physics meshes into its own
// generated_assets/ folder. To keep the prefab self-contained and readable, this ignores the
// generated meshes and instead copies the bot's original render and physics source files (with
// their original names) into _render and _mochi, matched to the exported links by name.
static void ExportMochiPrefab(
    SuperDexStudio* studio,
    BotAsset* botAsset,
    std::filesystem::path const& destDir,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  namespace fs = std::filesystem;

  std::string const exportName = botAsset->GetName();
  superdex::robotics::BotPrefab const& botPrefab = botAsset->GetBotPrefab();
  mochi::Context* context = studio->GetMochiContext();

  // Export the bot into a temp staging dir (auto-deleted on scope exit) so the runtime exporter
  // populates shapes, joints, and contact filters for us.
  auto tempDir = mochi::CreateTempDirectory("ExportMochiPrefab", error);
  MOCHI_ERROR_RETURN(error);
  mochi::Scene* scene = context->CreateScene("ExportPrefab");
  superdex::robotics::Bot* exportBot =
      superdex::robotics::CreateBot(scene, botPrefab, studio->GetRoboticsContext(), error);
  mochi::prefab::ExportScene(scene, exportName, tempDir.Path().string(), error);
  superdex::robotics::DestroyBot(scene, exportBot);
  context->DestroyScene(scene);
  MOCHI_ERROR_RETURN(error);

  // Reload the scene prefab that ExportScene just wrote into the temp staging directory.
  fs::path const tempScenePath = tempDir.Path() / exportName / (exportName + ".mochi_scene");
  mochi::prefab::ScenePrefab prefab =
      mochi::prefab::LoadFromFile(tempScenePath.string(), tempDir.Path().string(), context, error);
  MOCHI_ERROR_RETURN(error);

  prefab.scene = std::nullopt;
  prefab.comment = mochi::DynamicString(
      "Exported from " + botAsset->GetPath().GetFilename() + " using SuperDex Studio");

  // Everything is written into a single <destDir>/<exportName>/ folder.
  std::error_code ec;
  fs::path const prefabDir = destDir / exportName;
  fs::path const renderDir = prefabDir / fs::path(superdex::robotics::kRenderSubdir);
  fs::path const collisionDir = prefabDir / fs::path(superdex::robotics::kCollisionSubdir);
  fs::create_directories(renderDir, ec);
  MOCHI_ERROR_IF(ec, error, "Failed to create render directory.");
  fs::create_directories(collisionDir, ec);
  MOCHI_ERROR_IF(ec, error, "Failed to create collision directory.");
  MOCHI_ERROR_RETURN(error);

  // Map bot link name -> bot link, to recover the original source files.
  std::unordered_map<mochi::DynamicString, superdex::robotics::BotLinkPrefab const*> botLinksByName;
  for (auto const& link : botPrefab.links) {
    botLinksByName.emplace(link.name, &link);
  }

  // Copy each link's original render and physics source files (with their original names) into
  // _render and _mochi, and point the exported prefab at those copies.
  for (auto& actor : prefab.actors.articulated) {
    for (auto& link : actor.links) {
      auto const it = botLinksByName.find(link.name);
      if (it == botLinksByName.end()) {
        continue;
      }
      superdex::robotics::BotLinkPrefab const& botLink = *it->second;
      if (!botLink.renderModelFile.empty()) {
        fs::path const fileName = fs::path(botLink.renderModelFile.c_str()).filename();
        fs::copy_file(
            botLink.renderModelFile.c_str(),
            renderDir / fileName,
            fs::copy_options::overwrite_existing,
            ec);
        MOCHI_ERROR_IF(ec, error, "Failed to copy render model into render.");
        MOCHI_ERROR_RETURN(error);
        link.renderModelFile = mochi::DynamicString(
            "./" + std::string(superdex::robotics::kRenderSubdir) + "/" + fileName.string());
        link.renderModelScale = botLink.renderModelScale;
        link.renderModelRotation = botLink.renderModelRotation;
        link.renderModelTranslation = botLink.renderModelTranslation;
      }
      if (!botLink.shapeFile.empty()) {
        fs::path const fileName = fs::path(botLink.shapeFile.c_str()).filename();
        fs::copy_file(
            botLink.shapeFile.c_str(),
            collisionDir / fileName,
            fs::copy_options::overwrite_existing,
            ec);
        MOCHI_ERROR_IF(ec, error, "Failed to copy physics shape into collision.");
        MOCHI_ERROR_RETURN(error);
        link.shapeFile = mochi::DynamicString(
            "./" + std::string(superdex::robotics::kCollisionSubdir) + "/" + fileName.string());
        link.shapeScale = botLink.shapeScale;
        link.shapeRotation = botLink.shapeRotation;
        link.shapeTranslation = botLink.shapeTranslation;
      }
    }
  }

  fs::path const outPath = prefabDir / (exportName + ".mochi_prefab");
  mochi::prefab::SaveToJsonFile(prefab, outPath.string(), error);
  MOCHI_ERROR_RETURN(error);
}

//--------------------------------------------------------------------------------------------------
// AssetEditor
//--------------------------------------------------------------------------------------------------

BotEditor::BotEditor(SuperDexStudio* studio, BotAsset* asset)
    : AssetEditor(studio, asset), _botAsset(asset), _stage(studio, "BotEditorStage") {}

void BotEditor::Initialize() {
  // Initialize _undoStack
  if (!_botAsset->IsReadOnly()) {
    _undoStack.Initialize(
        [this] { return TakeUndoSnapshot(); },
        [this](std::string const& json, int selIdx) { RestoreUndoSnapshot(json, selIdx); });
    GetUndoStack().SetSelectionFn([this] { return GetUndoSelectionIndex(); });
  }
  // Initialize Viewport
  _viewport = Viewport::Create(_studio, _studio->GetViewSettings());
  _viewport->onSceneSelectionChanged =
      [this](std::vector<mochi_renderer::SceneObject*> const& objects) {
        OnSceneSelectionChanged(objects);
      };
  // Register viewport "Show" toggle commands (keyboard shortcuts + top-left dropdown). The Grid
  // toggle is registered by the viewport itself, for every editor.
  _viewport->RegisterShowCommand(
      {.name = "Joint Limits",
       .onToggle = [this] { _botVizFlags.showJointLimits ^= true; },
       .getState = [this] { return _botVizFlags.showJointLimits; },
       .shortcut = ImGuiKey_J});
  _viewport->RegisterShowCommand(
      {.name = "Joint Limits: Revolute",
       .onToggle = [this] { _botVizFlags.showRevoluteLimits ^= true; },
       .getState = [this] { return _botVizFlags.showRevoluteLimits; }});
  _viewport->RegisterShowCommand(
      {.name = "Joint Limits: Prismatic",
       .onToggle = [this] { _botVizFlags.showPrismaticLimits ^= true; },
       .getState = [this] { return _botVizFlags.showPrismaticLimits; }});
  _viewport->RegisterShowCommand(
      {.name = "Joint Limits: Spherical",
       .onToggle = [this] { _botVizFlags.showSphericalLimits ^= true; },
       .getState = [this] { return _botVizFlags.showSphericalLimits; }});
  _viewport->RegisterShowCommand(
      {.name = "Joint Limits: Cycle",
       .onToggle = [this] { _botVizFlags.showCycles ^= true; },
       .getState = [this] { return _botVizFlags.showCycles; }});
  _viewport->RegisterShowCommand(
      {.name = "Linear Transmissions",
       .onToggle = [this] { _botVizFlags.showLinearTransmissions ^= true; },
       .getState = [this] { return _botVizFlags.showLinearTransmissions; }});
  _viewport->RegisterShowCommand(
      {.name = "Spatial Tendons",
       .onToggle = [this] { _botVizFlags.showSpatialTendons ^= true; },
       .getState = [this] { return _botVizFlags.showSpatialTendons; }});
  _viewport->RegisterShowCommand(
      {.name = "Link Transform",
       .onToggle = [this] { _botVizFlags.showLocalTransform ^= true; },
       .getState = [this] { return _botVizFlags.showLocalTransform; },
       .shortcut = ImGuiKey_T});
  _viewport->RegisterShowCommand(
      {.name = "Link Inertia",
       .onToggle = [this] { _botVizFlags.showInertiaBox ^= true; },
       .getState = [this] { return _botVizFlags.showInertiaBox; },
       .shortcut = ImGuiKey_I});
  _viewport->RegisterShowCommand(
      {.name = "Center of Mass",
       .onToggle = [this] { _botVizFlags.showCenterOfMass ^= true; },
       .getState = [this] { return _botVizFlags.showCenterOfMass; },
       .shortcut = ImGuiKey_M});
  _viewport->RegisterShowCommand(
      {.name = "Physics Debug Draw",
       .onToggle = [this] { _mochiScene.ToggleDebugDraw(); },
       .getState = [this] { return _mochiScene.IsDebugDrawEnabled(); },
       .shortcut = ImGuiKey_P});
  _viewport->RegisterShowCommand(
      {.name = "Render Only",
       .onToggle =
           [this] {
             _stageType = StageType::RenderModelOnly;
             RestageBot();
           },
       .getState = [this] { return _stageType == StageType::RenderModelOnly; },
       .shortcut = ImGuiKey_1});
  _viewport->RegisterShowCommand(
      {.name = "Collision Only",
       .onToggle =
           [this] {
             _stageType = StageType::MochiModelOnly;
             RestageBot();
           },
       .getState = [this] { return _stageType == StageType::MochiModelOnly; },
       .shortcut = ImGuiKey_2});
  _viewport->RegisterShowCommand(
      {.name = "Render (Collision Fallback)",
       .onToggle =
           [this] {
             _stageType = StageType::RenderModelFallbackToMochiModel;
             RestageBot();
           },
       .getState = [this] { return _stageType == StageType::RenderModelFallbackToMochiModel; },
       .shortcut = ImGuiKey_3});
  // Initialize Mochi scene and callbacks
  _mochiScene.Initialize(_studio->GetMochiContext(), "BotEditor");
  _mochiScene.SetSettings(_studio->GetAppSettings().physics);
  _mochiScene.createPhysicsActors = [this](mochi::Scene* s) { CreatePhysicsActors(s); };
  _mochiScene.destroyPhysicsActors = [this](mochi::Scene* s) { DestroyPhysicsActors(s); };
  _mochiScene.registerPreStepCallback = [this](mochi::AsyncScene* a) {
    return RegisterPreStepCallback(a);
  };
  _mochiScene.registerPostStepCallback = [this](mochi::AsyncScene* a) {
    return RegisterPostStepCallback(a);
  };
  _mochiScene.onStopPhysics = [this]() { OnStopPhysics(); };
  // Force-drag (left-drag) of simulated links. The controller only exists while a session is
  // running (created in RegisterPreStepCallback, destroyed in OnStopPhysics), so the hooks read the
  // slot on each event.
  BindSceneObjectDragHooks(*_viewport, _dragController);
  // Bind the stage to the viewport's render scene, and give the viewport the stage so it can route
  // selection/hover highlights to it (the stage owns the per-link highlight clones).
  _stage.BindRenderScene(_viewport->GetRenderScene());
  _viewport->SetSceneStage(&_stage);
  // Stage the bot
  RecomputeModBuildStatus();
  RestageBot(); // also positions the ground plane at the bot's lowest point
  _viewport->SetSelectedSceneObjects({});
  _forceLinkFocus = false;
  _viewport->FocusCameraOnScene();
}

void BotEditor::OnHandleInputs() {
  if (CanSimulate()) {
    _mochiScene.HandleHotkeys();
  }
  // Delete/Backspace Key = Delete Select Link
  if (!_mochiScene.IsSimulating()) {
    if ((ImGui::IsKeyChordPressed(ImGuiKey_Delete) ||
         ImGui::IsKeyChordPressed(ImGuiKey_Backspace)) &&
        _viewport->HasSelection()) {
      DeleteBotLink(_selectedBotLinkIndex);
    }
  }
}

void BotEditor::OnRender(Renderer const* renderer) {
  // Sync data from physics simulation
  if (_mochiScene.IsSimulating()) {
    _mochiScene.UpdateStats();
    SyncFromPhysics();
  }
  // Draw mochi scene debug
  _mochiScene.DrawDebug(
      _viewport->GetRenderScene()->GetDebugDraw(), _studio->GetEditorToRendererSpaceConverter());
  // Draw force-drag visualization (grab point, target, connecting line).
  if (_dragController) {
    _dragController->DrawDebug(
        _viewport->GetRenderScene()->GetDebugDraw(),
        _viewport->GetDebugText(),
        _studio->GetEditorToRendererSpaceConverter());
  }
  // Draw visualization overlay/gizmos
  DrawBotVisualizations(_viewport->GetRenderScene());
  // Render the scene.
  _viewport->RenderScene(renderer);
}

void BotEditor::Shutdown() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  _contactEstimator.Stop();
  _stage.Clear();
  _viewport.reset();
}

void BotEditor::OnActivate() {
  // Rebuild when we are reactivated because the user may have changed a dependency (e.g. base bot)
  if (!_mochiScene.IsSimulating() && RecomputeModBuildStatus()) {
    _botAsset->MarkThumbnailDirty();
    RestageBot();
  }
}

void BotEditor::OnDeactivate() {
  // Stop simulation when the user changes to another editor.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
}

void BotEditor::Refresh() {
  // A referenced model was replaced/renamed elsewhere in the app. Stop any running simulation
  // (restaging mid-sim is unsafe), then refresh the mod build status and restage.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  RecomputeModBuildStatus();
  _botAsset->MarkThumbnailDirty();
  RestageBot();
}

void BotEditor::ShowTabContents() {
  ImGui::BeginChild("Viewport_Child", ImVec2(0, 0), 0, ImGuiWindowFlags_NoMove);
  // Force-drag only applies while the sim runs.
  _viewport->enableSceneObjectDrag = _mochiScene.IsSimulating();
  _viewport->ShowViewportContents(true);
  _viewport->ShowStatsOverlay(_mochiScene.GetStepsPerSecond());
  ImGui::BeginDisabled(!CanSimulate());
  _mochiScene.ShowPlayToolbarOverViewport();
  ImGui::EndDisabled();
  ImGui::EndChild(); // Viewport_Child
  ShowBatchRenameLinksJointsModal();
}

std::vector<AssetEditor::WindowDeclaration> BotEditor::GetDefaultWindows() {
  using Dock = AssetEditor::DockRegion;
  return {
      // main windows
      {"Bot Hierarchy", true, Dock::SidePanelTop, false},
      {"Bot Details", true, Dock::SidePanelBottom, false},
      {"Bot Link Details", true, Dock::SidePanelBottom, false},
      {"Bot Control", false, Dock::SidePanelBottom, false},
      {"Bot Contact", false, Dock::SidePanelBottom, false},
      {"Bot Transmissions", false, Dock::SidePanelBottom, false},
      {"Physics Settings", false, Dock::SidePanelBottom, false},
      // debug windows
      {"Render Scene Hierarchy", false, Dock::SidePanelTop, true},
      {"Render Scene Details", false, Dock::SidePanelBottom, true},
      {"Render Scene Stage", false, Dock::SidePanelTop, true}};
}

std::vector<AssetEditor::WindowDeclaration> BotEditor::GetAuxiliaryWindows() const {
  return GetDefaultWindows();
}

void BotEditor::ShowAuxiliaryWindows() {
  // main windows
  if (bool& open = _studio->GetWindowVisible("Bot Hierarchy")) {
    ImGui::BeginDisabled(_mochiScene.IsSimulating());
    ShowBotHierarchyWindow(&open);
    ImGui::EndDisabled();
  }
  if (bool& open = _studio->GetWindowVisible("Bot Details")) {
    ImGui::BeginDisabled(_mochiScene.IsSimulating());
    ShowBotDetailsWindow(&open);
    ImGui::EndDisabled();
  }
  if (bool& open = _studio->GetWindowVisible("Bot Link Details")) {
    ImGui::BeginDisabled(_mochiScene.IsSimulating());
    ShowBotLinkDetailsWindow(&open);
    ImGui::EndDisabled();
  }
  if (bool& open = _studio->GetWindowVisible("Bot Control")) {
    // Not wrapped in BeginDisabled: the enable/bandwidth knobs stay live while stopped, and
    // BotControl disables the slider sections itself.
    _botControl.ShowWindow(
        &open,
        _botAsset->GetBotPrefab(),
        _mochiScene.IsSimulating(),
        _mochiScene.GetAsyncScene(),
        _botAsset->GetPath());
  }
  if (bool& open = _studio->GetWindowVisible("Bot Contact")) {
    ImGui::BeginDisabled(_mochiScene.IsSimulating());
    ShowBotContactWindow(&open);
    ImGui::EndDisabled();
  }
  if (bool& open = _studio->GetWindowVisible("Bot Transmissions")) {
    ImGui::BeginDisabled(_mochiScene.IsSimulating());
    ShowBotTransmissionsWindow(&open);
    ImGui::EndDisabled();
  }
  if (bool& open = _studio->GetWindowVisible("Physics Settings")) {
    // A BotPrefab carries no scene settings, so the editor's override always applies.
    _mochiScene.ShowPhysicsSettingsWindow("Physics Settings", &open);
  }
  // debug windows
  if (bool& open = _studio->GetWindowVisible("Render Scene Hierarchy")) {
    _viewport->ShowSceneHierarchyWindow("Render Scene Hierarchy", &open);
  }
  if (bool& open = _studio->GetWindowVisible("Render Scene Details")) {
    _viewport->ShowSelectedObjectDetailsWindow("Render Scene Details", &open);
  }
  if (bool& open = _studio->GetWindowVisible("Render Scene Stage")) {
    auto* simNames = _mochiScene.IsSimulating() ? &_simData.GetConsumerData().linkNames : nullptr;
    _stage.ShowSceneStageWindow("Render Scene Stage", &open, simNames);
  }
}

void BotEditor::ShowMainMenuItems() {
  if (ImGui::BeginMenu("Bot")) {
    mochi::ErrorLog error;
    // Archive bot
    if (ImGui::MenuItem("Archive")) {
      mochi::Path const botPath = _botAsset->GetPath();
      mochi::Path defaultPath = botPath;
      defaultPath.ReplaceExtension(".superdex_bot_archive");
      constexpr auto filters = std::to_array<char const*>({"*.superdex_bot_archive"});
      auto outputPath = SuperDexStudio::GetFileDialogPath(
          "Archive Bot",
          filters.data(),
          1,
          "SuperDex Bot Archive (*.superdex_bot_archive)",
          true,
          defaultPath);
      if (!outputPath.IsEmpty()) {
        superdex::robotics::ArchiveParams params;
        params.src = botPath.ToString();
        params.dst = outputPath.ToString();
        params.comment = "Archived from SuperDex Studio";
        superdex::robotics::ArchiveBot(params, error);
      }
    }
    // Export Prefab
    if (ImGui::MenuItem("Export Prefab")) {
      auto outputDir =
          SuperDexStudio::GetFolderDialogPath("Export Prefab — Select Output Directory");
      if (!outputDir.IsEmpty()) {
        ExportMochiPrefab(_studio, _botAsset, outputDir.ToString(), error);
      }
    }
    // Export URDF
    if (ImGui::MenuItem("Export URDF")) {
      constexpr auto filters = std::to_array<char const*>({"*.urdf"});
      auto path = SuperDexStudio::GetFileDialogPath(
          "Export URDF", filters.data(), 1, "URDF (*.urdf)", true);
      if (!path.IsEmpty()) {
        superdex::robotics::SaveToUrdfFile(
            _botAsset->GetBotPrefab(), path.ToString().c_str(), error);
      }
    }
    ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    // Batch Rename Links & Joints: mirror of the asset browser's batch-rename dialog, operating on
    // a bot's link and joint names (with reference fixups) instead of files.
    if (ImGui::MenuItem("Batch Rename Links & Joints...")) {
      OpenBatchRenameLinksJointsModal();
    }
    // Bake Inertia: compute centerOfMass and momentOfInertia from mesh for links with mass/density.
    // For mod bots, bake the links introduced by AttachLink / ReplaceLink mods instead of the
    // (read-only, derived) built prefab.
    if (ImGui::MenuItem("Bake Mass & Inertial Properties")) {
      if (_botAsset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab) {
        superdex::robotics::BakeMassProperties(
            _botAsset->GetModBotPrefab(),
            _studio->GetMochiContext(),
            _studio->GetBotLoader(),
            error);
      } else {
        superdex::robotics::BakeMassProperties(
            _botAsset->GetBotPrefab(), _studio->GetMochiContext(), _studio->GetBotLoader(), error);
      }
      if (error.IsOK()) {
        _botAsset->Rebuild(_studio->GetBotLoader());
        _botAsset->SetDirty(true);
        _undoStack.PushNow();
      }
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip(
          "Converts applicable link inertias from a mass or density parameterization to a mass+CoM+MoI parameterization assuming uniform density of the body mesh.");
    }
    ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    bool const isModBot =
        _botAsset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab;
    // Always copies the (built) BotPrefab. For a Mod Bot that is the compiled result, so it is
    // labeled "(Flattened)".
    if (ImGui::MenuItem(isModBot ? "Copy BotPrefab (Flattened)" : "Copy BotPrefab")) {
      _botAsset->Rebuild(_studio->GetBotLoader());
      auto& params = _botAsset->GetBotPrefab();
      auto json = superdex::robotics::WriteToJsonString(params, mochi::ErrorLog{});
      ImGui::SetClipboardText(json.c_str());
    }
    // The ModBotPrefab recipe only exists for Mod Bots.
    ImGui::BeginDisabled(!isModBot);
    if (ImGui::MenuItem("Copy ModBotPrefab")) {
      _botAsset->Rebuild(_studio->GetBotLoader());
      auto& params = _botAsset->GetModBotPrefab();
      auto json = superdex::robotics::WriteToJsonString(params, mochi::ErrorLog{});
      ImGui::SetClipboardText(json.c_str());
    }
    ImGui::EndDisabled();
    ImGui::EndMenu();
  }
}

//--------------------------------------------------------------------------------------------------
// Batch Rename Links & Joints
//--------------------------------------------------------------------------------------------------

int BotEditor::ComputeBatchRenamePreview(
    std::vector<std::string> const& originalNames,
    BatchRenameParams const& inputs,
    std::vector<std::string>& outNewNames,
    std::vector<bool>& outRowInvalid) {
  outNewNames.clear();
  outNewNames.reserve(originalNames.size());
  outRowInvalid.assign(originalNames.size(), false);
  int changedCount = 0;
  for (auto const& name : originalNames) {
    std::string newName = ComputeBatchRenamedName(name, inputs);
    if (newName != name) {
      ++changedCount;
    }
    outNewNames.push_back(std::move(newName));
  }
  // A row is invalid if its new name is empty, or if it collides with any other
  // entry's new name (names must stay unique within the category).
  for (size_t i = 0; i < outNewNames.size(); ++i) {
    if (outNewNames[i].empty()) {
      outRowInvalid[i] = true;
      continue;
    }
    for (size_t j = i + 1; j < outNewNames.size(); ++j) {
      if (outNewNames[i] == outNewNames[j]) {
        outRowInvalid[i] = true;
        outRowInvalid[j] = true;
      }
    }
  }
  return changedCount;
}

void BotEditor::SnapshotBatchRenameNames() {
  using namespace superdex::robotics;
  _batchRename.linkNames.clear();
  _batchRename.jointNames.clear();
  if (_botAsset->GetBotFileType() == BotFileType::ModBotPrefab) {
    // Only names introduced by the recipe are renameable; base and externally-imported
    // (prefixed) names are read-only and excluded.
    ModBotPrefab const& modBotPrefab = _botAsset->GetModBotPrefab();
    for (auto const& mod : modBotPrefab.modifications) {
      std::visit(
          [&](auto&& m) {
            using T = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<T, AttachLink>) {
              _batchRename.linkNames.push_back(m.link.name.c_str());
              _batchRename.jointNames.push_back(m.joint.name.c_str());
            } else if constexpr (std::is_same_v<T, ReplaceLink>) {
              _batchRename.linkNames.push_back(m.link.name.c_str());
            } else if constexpr (std::is_same_v<T, AttachBot>) {
              _batchRename.jointNames.push_back(m.joint.name.c_str());
            }
          },
          mod);
    }
  } else {
    BotPrefab const& botPrefab = _botAsset->GetBotPrefab();
    for (auto const& link : botPrefab.links) {
      _batchRename.linkNames.push_back(link.name.c_str());
    }
    for (auto const& joint : botPrefab.joints) {
      _batchRename.jointNames.push_back(joint.name.c_str());
    }
  }
}

void BotEditor::OpenBatchRenameLinksJointsModal() {
  _batchRename = BatchRenameState{};
  SnapshotBatchRenameNames();
  _batchRename.open = true;
}

void BotEditor::ShowBatchRenameTabContents(
    BatchRenameParams& inputs,
    std::vector<std::string> const& originalNames) {
  int trimMax = 0;
  for (auto const& name : originalNames) {
    trimMax = std::max(trimMax, static_cast<int>(name.size()));
  }
  ImGui::BatchRenameInputs(inputs, trimMax);
  ImGui::Spacing();

  std::vector<std::string> newNames;
  std::vector<bool> rowInvalid;
  ComputeBatchRenamePreview(originalNames, inputs, newNames, rowInvalid);

  if (originalNames.empty()) {
    ImGui::TextDisabled("No renameable names.");
  } else {
    ImGui::BatchRenamePreviewTable(originalNames, newNames, rowInvalid);
  }
}

void BotEditor::ApplyBatchRename(bool applyLinks) {
  using namespace superdex::robotics;
  // Build the old->new map of changed names for the active category only.
  std::vector<std::string> const& originalNames =
      applyLinks ? _batchRename.linkNames : _batchRename.jointNames;
  std::vector<std::string> newNames;
  std::vector<bool> rowInvalid;
  ComputeBatchRenamePreview(originalNames, _batchRename.inputs, newNames, rowInvalid);

  std::map<std::string, std::string> nameMap;
  for (size_t i = 0; i < originalNames.size(); ++i) {
    if (newNames[i] != originalNames[i]) {
      nameMap[originalNames[i]] = newNames[i];
    }
  }
  if (nameMap.empty()) {
    return;
  }

  // Remap a name-valued field through the map if it matches a changed (old) name. Uses assign() to
  // preserve the DynamicString's allocator.
  auto remap = [&nameMap](mochi::DynamicString& value) {
    auto const it = nameMap.find(std::string(value.c_str()));
    if (it != nameMap.end()) {
      value.assign(it->second.c_str(), it->second.size());
    }
  };

  if (_botAsset->GetBotFileType() == BotFileType::ModBotPrefab) {
    ModBotPrefab& modBotPrefab = _botAsset->GetModBotPrefab();
    // Each field independently holds an old name, so a single pass that renames introduced
    // names and remaps link references is unambiguous (valid bots have unique names).
    for (auto& mod : modBotPrefab.modifications) {
      std::visit(
          [&](auto&& m) {
            using T = std::decay_t<decltype(m)>;
            if (applyLinks) {
              if constexpr (std::is_same_v<T, AttachLink>) {
                remap(m.link.name);
                remap(m.parentLinkName);
              } else if constexpr (std::is_same_v<T, AttachBot>) {
                remap(m.parentLinkName);
              } else if constexpr (std::is_same_v<T, ReplaceLink>) {
                remap(m.link.name);
                remap(m.linkToReplace);
              } else if constexpr (std::is_same_v<T, ReplaceLinkWithBot>) {
                remap(m.linkToReplace);
              }
            } else {
              if constexpr (std::is_same_v<T, AttachLink> || std::is_same_v<T, AttachBot>) {
                remap(m.joint.name);
              }
            }
          },
          mod);
    }
  } else {
    BotPrefab& botPrefab = _botAsset->GetBotPrefab();
    if (applyLinks) {
      for (auto& link : botPrefab.links) {
        remap(link.name);
      }
      for (auto& contactOverride : botPrefab.contactOverrides) {
        remap(contactOverride.linkA);
        remap(contactOverride.linkB);
      }
    } else {
      for (auto& joint : botPrefab.joints) {
        remap(joint.name);
      }
    }
  }

  _botAsset->Rebuild(_studio->GetBotLoader());
  _botAsset->SetDirty(true);
  _undoStack.PushNow();
  // Refresh snapshots so the preview reflects the applied names and further edits map correctly.
  SnapshotBatchRenameNames();
}

void BotEditor::ShowBatchRenameLinksJointsModal() {
  if (_batchRename.open) {
    ImGui::OpenPopup("Batch Rename Links & Joints");
    _batchRename.open = false;
  }
  ImVec2 const center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(800, 0), ImGuiCond_Appearing);
  // A non-null p_open gives the window an ImGui close (X) button in the top-right.
  bool popupOpen = true;
  if (ImGui::BeginPopupModal(
          "Batch Rename Links & Joints",
          &popupOpen,
          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
              ImGuiWindowFlags_NoSavedSettings)) {
    // The active tab determines which category Apply operates on. Both tabs share the same input
    // set, so switching tabs preserves the settings.
    std::vector<std::string> const* activeNames = nullptr;
    bool activeIsLinks = false;
    if (ImGui::BeginTabBar("##BatchRenameTabs")) {
      if (ImGui::BeginTabItem("Links")) {
        ShowBatchRenameTabContents(_batchRename.inputs, _batchRename.linkNames);
        activeNames = &_batchRename.linkNames;
        activeIsLinks = true;
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Joints")) {
        ShowBatchRenameTabContents(_batchRename.inputs, _batchRename.jointNames);
        activeNames = &_batchRename.jointNames;
        activeIsLinks = false;
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }

    // Gate Apply on the active tab: enabled only when at least one name changes and no row is
    // invalid in that category.
    bool canApply = false;
    if (activeNames != nullptr) {
      std::vector<std::string> newNames;
      std::vector<bool> rowInvalid;
      int const changed =
          ComputeBatchRenamePreview(*activeNames, _batchRename.inputs, newNames, rowInvalid);
      int const invalidCount =
          static_cast<int>(std::count(rowInvalid.begin(), rowInvalid.end(), true));
      canApply = invalidCount == 0 && changed > 0;
    }

    ImGui::Separator();
    ImGui::BeginDisabled(!canApply);
    if (ImGui::Button("Apply", ImVec2(120, 0))) {
      ApplyBatchRename(activeIsLinks);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Close", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

bool BotEditor::CanUndoRedo() const {
  return !_mochiScene.IsSimulating();
}

void BotEditor::ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) {
  if (_viewport && _viewport->GetRenderScene()) {
    _viewport->GetRenderScene()->ApplyViewSettings(viewSettings);
  }
}

void BotEditor::OnAppSettingsChanged(AppSettings const& settings) {
  // The drag controller snapshots its tuning per session, so a live one has to be told.
  if (_dragController) {
    _dragController->SetSettings(settings.physicsDrag);
  }
}
//--------------------------------------------------------------------------------------------------
// Undo/Redo
//--------------------------------------------------------------------------------------------------

std::string BotEditor::TakeUndoSnapshot() const {
  if (_botAsset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab) {
    return SReflect::ToJsonString(_botAsset->GetModBotPrefab(), false);
  }
  return SReflect::ToJsonString(_botAsset->GetBotPrefab(), false);
}

void BotEditor::RestoreUndoSnapshot(std::string const& json, int selectionIndex) {
  if (!_viewport) {
    return;
  }
  _viewport->SetSelectedSceneObjects({});
  if (_botAsset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab) {
    _botAsset->GetModBotPrefab() = superdex::robotics::ModBotPrefab{};
    SReflect::FromJsonString(
        _botAsset->GetModBotPrefab(), json, SReflect::DeserializeFlags::Default);
  } else {
    _botAsset->GetBotPrefab() = superdex::robotics::BotPrefab{};
    SReflect::FromJsonString(_botAsset->GetBotPrefab(), json, SReflect::DeserializeFlags::Default);
  }
  RecomputeModBuildStatus();
  _selectedBotLinkIndex = selectionIndex;
  RestageBot();
  _botAsset->MarkThumbnailDirty();
  _botAsset->SetDirty(!GetUndoStack().IsAtSavedState());
}

int BotEditor::GetUndoSelectionIndex() const {
  return _selectedBotLinkIndex;
}

//--------------------------------------------------------------------------------------------------
// Bot Instance
//--------------------------------------------------------------------------------------------------

void BotEditor::RestageBot() {
  int const selected = _selectedBotLinkIndex;
  bool const hadSelection = _viewport->HasSelection();
  bool const rebuilt = _stage.StageBot(_botAsset->GetBotPrefab(), _stageType);
  if (rebuilt && hadSelection) {
    _viewport->SetSelectedSceneObjects({});
    _selectedBotLinkIndex = selected;
    if (selected >= 0 && selected < _stage.GetNumActors()) {
      if (auto* object = _stage.GetActors()[selected].sceneObject) {
        _viewport->SetSelectedSceneObjects({object});
      }
    }
  }
  // Reposition the drop-shadow ground plane and the grid at the bot's lowest point (in the rest
  // pose staged above); the same height positions the studio physics ground plane. Safe here:
  // RestageBot rebuilds the rest pose and is never called mid-sim.
  _mochiScene.SetGroundPlaneHeight(_viewport->UpdateGroundPlane());
}

void BotEditor::AddBotLink(int iLink) {
  if (_stage.IsEmpty() ||
      _botAsset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab) {
    return;
  }
  auto& botPrefab = _botAsset->GetBotPrefab();
  if (iLink < 0 || iLink >= static_cast<int>(botPrefab.links.size())) {
    return;
  }
  int newIdx = superdex::robotics::AddLink(botPrefab, iLink, "new_link", mochi::ErrorLog{});
  _selectedBotLinkIndex = newIdx;
  bool buildOk = false;
  _botAsset->Rebuild(_studio->GetBotLoader(), &buildOk);
  if (buildOk) {
    _botAsset->SetDirty(true);
    _botAsset->MarkThumbnailDirty();
    RestageBot();
    GetUndoStack().PushNow();
  }
}

void BotEditor::DeleteBotLink(int iLink) {
  if (_stage.IsEmpty() ||
      _botAsset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab) {
    return;
  }
  if (_mochiScene.IsSimulating()) {
    return;
  }
  superdex::robotics::RemoveLinkAndDescendants(_botAsset->GetBotPrefab(), iLink, mochi::ErrorLog{});
  _selectedBotLinkIndex = -1;
  _viewport->SetSelectedSceneObjects({});
  bool buildOk = false;
  _botAsset->Rebuild(_studio->GetBotLoader(), &buildOk);
  if (buildOk) {
    _botAsset->SetDirty(true);
    _botAsset->MarkThumbnailDirty();
    RestageBot();
    GetUndoStack().PushNow();
  }
}

void BotEditor::ReparentBotLink(int linkIdx, int newParentIdx, bool preserveWorldTransform) {
  if (_stage.IsEmpty() ||
      _botAsset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab) {
    return;
  }
  if (_mochiScene.IsSimulating()) {
    return;
  }
  auto& botPrefab = _botAsset->GetBotPrefab();
  int const numLinks = static_cast<int>(botPrefab.links.size());
  if (linkIdx <= 0 || linkIdx >= numLinks || newParentIdx < 0 || newParentIdx >= numLinks ||
      newParentIdx == linkIdx) {
    return;
  }
  mochi::DynamicString const savedName = botPrefab.links[linkIdx].name;
  int const prevParentLink = botPrefab.links[linkIdx].parentLink;
  mochi::TransformRT const prevParentLinkFromJoint = botPrefab.joints[linkIdx].parentLinkFromJoint;
  if (preserveWorldTransform) {
    mochi::DynamicArray<mochi::TransformRT> rootFromLink;
    superdex::robotics::ComputeLinkTransformsFromPose(
        botPrefab,
        mochi::MakeConstSpan(botPrefab.defaultPose),
        rootFromLink,
        superdex::robotics::LinkTransformSpace::RootFromParent,
        mochi::ErrorLog{});
    if (static_cast<int>(rootFromLink.size()) == numLinks) {
      botPrefab.joints[linkIdx].parentLinkFromJoint = mochi::Invert(rootFromLink[newParentIdx]) *
          rootFromLink[prevParentLink] * prevParentLinkFromJoint;
    }
  }
  botPrefab.links[linkIdx].parentLink = newParentIdx;
  bool buildOk = false;
  _botAsset->Rebuild(_studio->GetBotLoader(), &buildOk);
  if (buildOk) {
    _selectedBotLinkIndex = superdex::robotics::FindLinkIndexByName(botPrefab, savedName);
    _botAsset->SetDirty(true);
    _botAsset->MarkThumbnailDirty();
    RestageBot();
    GetUndoStack().PushNow();
  } else {
    botPrefab.links[linkIdx].parentLink = prevParentLink;
    botPrefab.joints[linkIdx].parentLinkFromJoint = prevParentLinkFromJoint;
    _botAsset->Rebuild(_studio->GetBotLoader());
  }
}

//--------------------------------------------------------------------------------------------------
// Viewport
//--------------------------------------------------------------------------------------------------

void BotEditor::DrawBotVisualizations(mochi_renderer::Scene* renderScene) const {
  if (_stage.IsEmpty()) {
    return;
  }
  auto* dd = renderScene->GetDebugDraw();
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  auto const& botPrefab = _botAsset->GetBotPrefab();
  auto const& viz = _studio->GetAppSettings().botVisualization;
  if (_botVizFlags.showJointLimits) {
    if (_selectedBotLinkIndex > 0) {
      DrawJointLimitVisualization(
          dd, _stage, botPrefab, _selectedBotLinkIndex, converter, _botVizFlags, viz.joint);
    } else {
      DrawAllJointLimits(dd, _stage, botPrefab, converter, _botVizFlags, viz.joint);
    }
    // Cycle joints reference two links, so they are not tied to a single selection; draw all.
    DrawAllCycleJointVisualizations(dd, _stage, botPrefab, converter, _botVizFlags, viz.joint);
  }
  if (_botVizFlags.showInertiaBox || _botVizFlags.showCenterOfMass ||
      _botVizFlags.showLocalTransform) {
    if (_selectedBotLinkIndex >= 0) {
      DrawLinkVisualization(
          dd, _stage, botPrefab, _selectedBotLinkIndex, converter, _botVizFlags, viz.link);
    } else {
      DrawAllLinkVisualizations(dd, _stage, botPrefab, converter, _botVizFlags, viz.link);
    }
  }
  // Determine which displacements to use for drawing transmissions:
  // physics during simulation, UI expanded state otherwise.
  bool const simulating = _mochiScene.IsSimulating();
  mochi::DynamicArray<float> const* displacementsPtr = &_transmissionDisplacements;
  mochi::DynamicArray<float> uiDisplacements;
  if (!simulating) {
    size_t const numLinear = botPrefab.linearTransmissions.size();
    size_t const numSpatial = botPrefab.spatialTendons.size();
    uiDisplacements.resize(numLinear + numSpatial);
    for (size_t i = 0; i < numLinear; ++i) {
      bool expanded = i < _linearTransmissionExpanded.size() && _linearTransmissionExpanded[i];
      uiDisplacements[i] = expanded ? 1.0f : -1.0f;
    }
    for (size_t i = 0; i < numSpatial; ++i) {
      bool expanded = i < _spatialTendonExpanded.size() && _spatialTendonExpanded[i];
      uiDisplacements[numLinear + i] = expanded ? 1.0f : -1.0f;
    }
    displacementsPtr = &uiDisplacements;
  }
  if (_botVizFlags.showLinearTransmissions) {
    // Pass transmission displacements: simulation values during simulation, UI expanded state
    // otherwise
    DrawAllLinearTransmissionVisualizations(
        dd, _stage, botPrefab, converter, _botVizFlags, viz.transmission, displacementsPtr);
  }
  if (_botVizFlags.showSpatialTendons) {
    // Pass transmission displacements with offset for spatial tendons
    // LinearTransmissions occupy indices 0..N-1, SpatialTendons occupy N..N+M-1
    size_t const numLinearTransmissions = botPrefab.linearTransmissions.size();
    DrawAllSpatialTendonVisualizations(
        dd,
        _stage,
        botPrefab,
        converter,
        _botVizFlags,
        viz.transmission,
        displacementsPtr,
        numLinearTransmissions);
  }
}

void BotEditor::OnSceneSelectionChanged(std::vector<mochi_renderer::SceneObject*> const& objects) {
  // The bot editor is single-select (it doesn't opt into multi-select), so derive the selected link
  // from the primary (last) object, or clear it when the selection is empty.
  if (objects.empty()) {
    _selectedBotLinkIndex = -1;
    return;
  }
  _selectedBotLinkIndex = _stage.GetSceneObjectIndex(objects.back());
  if (_selectedBotLinkIndex >= 0) {
    _forceLinkFocus = true;
  }
}

//--------------------------------------------------------------------------------------------------
// Mochi Scene
//--------------------------------------------------------------------------------------------------

bool BotEditor::CanSimulate() const {
  return !_stage.IsEmpty();
}

void BotEditor::CreatePhysicsActors(mochi::Scene* scene) {
  if (_stage.IsEmpty()) {
    return;
  }
  auto* botsContext = _studio->GetRoboticsContext();
  mochi::ErrorLog e;
  // Mochi cannot toggle gravity on a live actor, so gravity compensation is baked into this
  // spawn-time copy of the prefab -- never into the asset.
  superdex::robotics::BotPrefab botPrefab = _botAsset->GetBotPrefab();
  if (_botControl.IsGravityCompensated()) {
    for (auto& link : botPrefab.links) {
      link.hasGravity = false;
    }
  }
  _bot = botsContext->CreateBot(scene, botPrefab, _studio->GetBotLoader(), e);
  _botControl.SetBot(_bot);
}

void BotEditor::DestroyPhysicsActors(mochi::Scene* scene) {
  if (!_bot) {
    return;
  }
  _botControl.SetBot(nullptr);
  superdex::robotics::DestroyBot(scene, _bot);
  _bot = nullptr;
}

mochi::CallbackHandle BotEditor::RegisterPreStepCallback(mochi::AsyncScene* scene) {
  // Per-session force-drag controller; torn down in OnStopPhysics so its lifetime matches the
  // session.
  _dragController = std::make_unique<PhysicsDragController>(
      _mochiScene,
      &_stage,
      _studio->GetAppSettings().physicsDrag,
      _studio->GetRendererToEditorSpaceConverter());
  return _botControl.RegisterPreStepCallback(scene);
}

mochi::CallbackHandle BotEditor::RegisterPostStepCallback(mochi::AsyncScene* scene) {
  if (!_bot) {
    return {};
  }
  mochi::CallbackHandle handle =
      scene->RegisterPostStepCallback("MochiBots::ExtractBot", [this](mochi::StepInfo const& info) {
        mochi::ErrorLog e;
        if (_bot == nullptr) {
          return;
        }
        // Query link actor names and world transforms
        auto* actor = _bot->GetArticulatedActor();
        auto linkActors = actor->GetNestedLinkActors(e);
        auto const numLinks = static_cast<int>(linkActors.size());
        SimData& data = _simData.GetProducerData();
        data.linkNames.resize(numLinks);
        data.linkTransforms.resize(numLinks);
        for (int i = 0; i < numLinks; ++i) {
          auto linkActor = info.scene->GetActor(linkActors[i]);
          data.linkNames[i] = linkActor->GetName();
          data.linkTransforms[i] = linkActor->GetRootTransform();
        }
        // Query transmission displacements for all linear transmissions and spatial tendons
        // Both types share the same transmission index namespace:
        // - LinearTransmissions occupy indices 0..numLinearTransmissions-1
        // - SpatialTendons occupy indices numLinearTransmissions..total-1
        auto const& botPrefab = _botAsset->GetBotPrefab();
        auto const numLinearTransmissions = botPrefab.linearTransmissions.size();
        auto const numSpatialTendons = botPrefab.spatialTendons.size();
        auto const totalTransmissions = numLinearTransmissions + numSpatialTendons;
        data.transmissionDisplacements.resize(totalTransmissions);
        // Query LinearTransmissions (indices 0..N-1)
        for (size_t i = 0; i < numLinearTransmissions; ++i) {
          data.transmissionDisplacements[i] = static_cast<float>(
              mochi::experimental::GetTransmissionDisplacement(actor, static_cast<int>(i), e));
        }
        // Query SpatialTendons (indices N..N+M-1)
        for (size_t i = 0; i < numSpatialTendons; ++i) {
          size_t const transmissionIndex = numLinearTransmissions + i;
          data.transmissionDisplacements[transmissionIndex] =
              static_cast<float>(mochi::experimental::GetTransmissionDisplacement(
                  actor, static_cast<int>(transmissionIndex), e));
        }
        _simData.Produce();
      });
  return handle;
}

void BotEditor::OnStopPhysics() {
  // Tear down the per-session force-drag controller. Runs after the async scene (and its callbacks)
  // are destroyed, so the controller's callback is already gone.
  _dragController.reset();
  _stage.ResetWorldTransforms(_studio->GetEditorToRendererSpaceConverter());
  _simData.Consume();
  // Hide and reset the Control window so the next sim starts at zero effort.
  _botControl.OnStopPhysics();
}

void BotEditor::SyncFromPhysics() {
  if (_simData.Consume()) {
    auto const& data = _simData.GetConsumerData();
    _transmissionDisplacements = data.transmissionDisplacements;
    _stage.ApplyWorldTransforms(
        mochi::MakeConstSpan(data.linkTransforms), _studio->GetEditorToRendererSpaceConverter());
  }
}

//--------------------------------------------------------------------------------------------------
// ImGui
//--------------------------------------------------------------------------------------------------

void BotEditor::ShowBotHierarchyWindow(bool* open) {
  using namespace superdex::robotics;
  ImGui::Begin("Bot Hierarchy", open, GetBotWindowFlags());

  auto BuildIssuesText = [](mochi::DynamicArray<mochi::DynamicString> const& issues) {
    mochi::DynamicString text;
    for (auto& issue : issues) {
      text += issue + "\n";
    }
    return text;
  };

  bool const isBuiltBot = _botAsset->GetBotFileType() == BotFileType::ModBotPrefab;
  bool const isReadOnly = _botAsset->IsReadOnly();
  auto const& validateResults = _botAsset->GetValidateResults();
  ImVec4 defaultColor = ImGui::GetStyleColorVec4(ImGuiCol_Text);
  ImVec4 warningColor = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
  bool const botHasIssues = !_botAsset->IsBuildOk() || ValidateResultsHaveIssues(validateResults);
  ImGui::PushStyleColor(ImGuiCol_Text, botHasIssues ? warningColor : defaultColor);
  float const iconWidth = ImGui::GetTextLineHeight();
  float const spacing = ImGui::GetStyle().ItemSpacing.x;
  float const avail = ImGui::GetContentRegionAvail().x;
  float const selWidth = avail - 2 * (iconWidth + spacing);
  ImGui::PushFont(_studio->GetFont("Roboto Bold Large"));
  ImGui::SetNextItemAllowOverlap();
  // Not a selection: the bot row only raises the Bot Details window. Link selection is owned by
  // the link tree below and is deliberately left untouched so both details windows can coexist.
  if (ImGui::Selectable(_botAsset->GetBotName().c_str(), false, 0, ImVec2(selWidth, 0.0f))) {
    FocusWindowIfOpen("Bot Details");
  }
  ImGui::PopFont();

  if (botHasIssues) {
    ImGui::SameLine();
    ImGui::TextColored(warningColor, ICON_FA_EXCLAMATION_TRIANGLE);
    if (ImGui::IsItemHovered()) {
      mochi::DynamicString text = BuildIssuesText(validateResults.botIssues);
      if (text.empty()) {
        text = FirstValidateIssue(validateResults);
      }
      if (text.empty() && !_modBuildError.empty()) {
        text = _modBuildError;
      }
      if (text.empty()) {
        text = "The bot has errors. See highlighted modifications, links, and joints.";
      }
      ImGui::SetTooltip("%s", text.c_str());
    }
  }
  ImGui::PopStyleColor();

  ImGui::SetNextItemAllowOverlap();
  ShowBotTreeSettingsCog();

  ImGui::Separator();
  ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));

  defaultColor =
      ImGui::GetStyleColorVec4(isBuiltBot || isReadOnly ? ImGuiCol_TextDisabled : ImGuiCol_Text);
  ImVec4 warningColorLinks =
      ImVec4(1.0f, 1.0f, 0.0f, isBuiltBot ? ImGui::GetStyle().DisabledAlpha : 1.0f);

  // Inline warning icon for a row that has validation issues, with the issues as its tooltip.
  auto ShowIssuesWarning = [&](mochi::DynamicArray<mochi::DynamicString> const& issues) {
    if (issues.empty()) {
      return;
    }
    ImGui::SameLine();
    ImGui::TextColored(warningColorLinks, ICON_FA_EXCLAMATION_TRIANGLE);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("%s", BuildIssuesText(issues).c_str());
    }
  };

  ImGui::BeginChild("LinkTreeChild");
  auto tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable |
      ImGuiTableRowFlags_Headers;
  int linkToAdd = -1;
  int linkToDelete = -1;
  if (ImGui::BeginTable("##LinkTree", 2, tableFlags)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Link");
    ImGui::TableSetupColumn("Joint");
    ImGui::TableHeadersRow();

    bool showlinkTreeHierarchical = _studio->GetAppSettings().botEditor.showlinkTreeHierarchical;
    if (showlinkTreeHierarchical) {
      // Compute ancestors that need to be open for the selected link
      std::unordered_set<int> ancestorsToOpen;
      if (_forceLinkFocus && _selectedBotLinkIndex >= 0) {
        int current = _botAsset->GetBotPrefab().links[_selectedBotLinkIndex].parentLink;
        while (current != kIndexNone) {
          ancestorsToOpen.insert(current);
          current = _botAsset->GetBotPrefab().links[current].parentLink;
        }
      }

      std::function<void(int)> RenderLinkTree;
      RenderLinkTree = [&](int linkIndex) {
        BotLinkPrefab const& link = _botAsset->GetBotPrefab().links[linkIndex];
        BotJointPrefab const& joint = _botAsset->GetBotPrefab().joints[linkIndex];
        bool hasChildren = !link._childrenIndices.empty();
        bool const selected = _selectedBotLinkIndex == linkIndex;
        bool const linkHasIssues = !validateResults.linkIssues[linkIndex].empty();
        bool const jointHasIssues = !validateResults.jointIssues[linkIndex].empty();

        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        ImGui::PushID(linkIndex);

        // Force open ancestors of the selected link
        if (_forceLinkFocus && ancestorsToOpen.contains(linkIndex)) {
          ImGui::SetNextItemOpen(true, ImGuiCond_Always);
        }

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
            ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_DefaultOpen |
            ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAllColumns;
        if (!hasChildren) {
          flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
              ImGuiTreeNodeFlags_Bullet;
        }
        if (selected) {
          flags |= ImGuiTreeNodeFlags_Selected;
        }

        ImGui::PushStyleColor(ImGuiCol_Text, linkHasIssues ? warningColorLinks : defaultColor);
        bool nodeOpen = ImGui::TreeNodeEx(link.name.c_str(), flags);
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) {
          if (linkIndex >= 0 && linkIndex < _stage.GetNumActors()) {
            _viewport->HighlightSceneObject(
                _stage.GetActors()[linkIndex].sceneObject,
                {46.f / 255.f, 134.f / 255.f, 233.f / 255.f});
          }
        }
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
          _selectedBotLinkIndex = linkIndex;
          if (linkIndex >= 0 && linkIndex < _stage.GetNumActors()) {
            _viewport->SetSelectedSceneObjects({_stage.GetActors()[linkIndex].sceneObject}, false);
          }
          FocusWindowIfOpen("Bot Link Details");
        }
        if (!isBuiltBot && !isReadOnly) {
          if (ImGui::BeginPopupContextItem("Popup", ImGuiPopupFlags_MouseButtonRight)) {
            if (ImGui::Selectable("Add Child Link")) {
              linkToAdd = linkIndex;
            }
            ImGui::BeginDisabled(linkIndex == 0);
            if (ImGui::Selectable("Delete Link")) {
              linkToDelete = linkIndex;
            }
            ImGui::EndDisabled();
            ImGui::EndPopup();
          }
        }
        // Last, so every query above reports on the tree node rather than on this icon.
        ShowIssuesWarning(validateResults.linkIssues[linkIndex]);

        ImGui::TableNextColumn();

        ImGui::PushStyleColor(ImGuiCol_Text, jointHasIssues ? warningColorLinks : defaultColor);
        ImGui::TextUnformatted(joint.name.c_str());
        ImGui::PopStyleColor();
        ShowIssuesWarning(validateResults.jointIssues[linkIndex]);

        // Scroll to selected item and center it
        if (_forceLinkFocus && selected) {
          ImGui::SetScrollHereY(0.5f);
          _forceLinkFocus = false;
        }

        if (hasChildren && nodeOpen) {
          for (int childIndex : link._childrenIndices) {
            RenderLinkTree(childIndex);
          }
          ImGui::TreePop();
        }

        ImGui::PopID();
      };

      float originalIndent = ImGui::GetStyle().IndentSpacing;
      ImGui::GetStyle().IndentSpacing = originalIndent * 0.4f;

      for (size_t i = 0; i < _botAsset->GetBotPrefab().links.size(); ++i) {
        if (_botAsset->GetBotPrefab().links[i].parentLink == kIndexNone) {
          RenderLinkTree(static_cast<int>(i));
          break;
        }
      }

      ImGui::GetStyle().IndentSpacing = originalIndent;
    } else {
      for (int i = 0; i < static_cast<int>(_botAsset->GetBotPrefab().links.size()); ++i) {
        BotLinkPrefab const& link = _botAsset->GetBotPrefab().links[i];
        BotJointPrefab* const joint = &_botAsset->GetBotPrefab().joints[i];
        bool const selected = _selectedBotLinkIndex == i;
        bool const linkHasIssues = !validateResults.linkIssues[i].empty();
        bool const jointHasIssues = !validateResults.jointIssues[i].empty();

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushID(i);

        ImGuiSelectableFlags selFlags = ImGuiSelectableFlags_SpanAllColumns;
        ImGui::PushStyleColor(ImGuiCol_Text, linkHasIssues ? warningColorLinks : defaultColor);
        if (ImGui::Selectable(link.name.c_str(), selected, selFlags)) {
          _selectedBotLinkIndex = i;
          FocusWindowIfOpen("Bot Link Details");
        }
        ImGui::PopStyleColor();

        if (!isBuiltBot && !isReadOnly &&
            ImGui::BeginPopupContextItem("Popup", ImGuiPopupFlags_MouseButtonRight)) {
          if (ImGui::Selectable("Add Child Link")) {
            AddBotLink(i);
          }
          ImGui::BeginDisabled(i == 0);
          if (ImGui::Selectable("Delete Link")) {
            linkToDelete = i;
          }
          ImGui::EndDisabled();
          ImGui::EndPopup();
        }
        // Last, so the context menu above targets the row rather than this icon.
        ShowIssuesWarning(validateResults.linkIssues[i]);

        ImGui::TableNextColumn();

        ImGui::PushStyleColor(ImGuiCol_Text, jointHasIssues ? warningColorLinks : defaultColor);
        ImGui::TextUnformatted(joint->name.c_str());
        ImGui::PopStyleColor();
        ShowIssuesWarning(validateResults.jointIssues[i]);

        if (_forceLinkFocus && selected) {
          ImGui::SetScrollHereY(0.5f);
          _forceLinkFocus = false;
        }

        ImGui::PopID();
      }
    }
    ImGui::EndTable();
  }
  ImGui::PopStyleColor(); // ImGuiCol_Header
  ImGui::EndChild();

  // handle added links
  if (linkToAdd != -1) {
    AddBotLink(linkToAdd);
  }
  // handle deleted links
  if (linkToDelete != -1) {
    DeleteBotLink(linkToDelete);
  }

  ImGui::End();
}

void BotEditor::ShowBotTreeSettingsCog() {
  auto backup = ImGui::GetCursorPos();
  float const gearWidth = ImGui::CalcTextSize(ICON_FA_COG).x;
  ImGui::SetCursorPos(ImGui::GetCursorStartPos());
  ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - gearWidth);
  if (ImGui::TextButton(ICON_FA_COG)) {
    ImGui::OpenPopup("##BotTreeSettings");
  }
  ImGui::SetCursorPos(backup);
  if (ImGui::BeginPopup("##BotTreeSettings")) {
    bool& showlinkTreeHierarchical = _studio->GetAppSettings().botEditor.showlinkTreeHierarchical;
    if (ImGui::MenuItem("Show Links in Data Order", nullptr, !showlinkTreeHierarchical)) {
      showlinkTreeHierarchical = false;
    }
    if (ImGui::MenuItem("Show Links in Hierarchical Order", nullptr, showlinkTreeHierarchical)) {
      showlinkTreeHierarchical = true;
    }
    ImGui::EndPopup();
  }
}

ImGuiWindowFlags BotEditor::GetBotWindowFlags() const {
  return !_stage.IsEmpty() && _botAsset->IsDirty() ? ImGuiWindowFlags_UnsavedDocument
                                                   : ImGuiWindowFlags_None;
}

void BotEditor::ApplyBotParamsEdit() {
  bool buildOk = false;
  _botAsset->Rebuild(_studio->GetBotLoader(), &buildOk);
  if (buildOk) {
    _botAsset->SetDirty(!_botAsset->IsReadOnly());
    GetUndoStack().MarkEdited();
    _botAsset->MarkThumbnailDirty();
    RestageBot();
  }
}

void BotEditor::ShowBotDetailsWindow(bool* open) {
  using namespace superdex::robotics;
  ImGui::Begin("Bot Details", open, GetBotWindowFlags());
  bool const isBuiltBot = _botAsset->GetBotFileType() == BotFileType::ModBotPrefab;
  bool const isReadOnly = _botAsset->IsReadOnly();
  bool modsChanged = false;
  bool paramsChanged =
      ImGui::InputText("Name", &_botAsset->GetBotName(), ImGuiInputTextFlags_CharsNoBlank);
  if (isBuiltBot) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32_BLACK_TRANS);
    ImGui::BeginChild("Bot Details", ImVec2(0, 0));
    modsChanged |=
        ShowModBotPrefabEditorWidgets(_botAsset->GetModBotPrefab(), _studio->GetAssetManager());
    ImGui::EndChild();
    ImGui::PopStyleColor(); // ImGuiCol_ChildBg
  } else {
    paramsChanged |= ShowBotPrefabEditorWidgets(_botAsset->GetBotPrefab());
  }
  if (paramsChanged) {
    ApplyBotParamsEdit();
  }
  if (modsChanged) {
    // Persist the recipe edit (dirty + undoable) regardless of build success.
    _botAsset->SetDirty(!isReadOnly);
    GetUndoStack().MarkEdited();
    if (RecomputeModBuildStatus()) {
      // Only restage/refresh the thumbnail on success; on failure the last good
      // prefab stays staged so the hierarchy/details remain populated.
      _botAsset->MarkThumbnailDirty();
      RestageBot();
    }
  }
  ImGui::End(); // Bot Details
}

void BotEditor::ShowBotLinkDetailsWindow(bool* open) {
  using namespace superdex::robotics;
  ImGui::Begin("Bot Link Details", open, GetBotWindowFlags());
  if (_selectedBotLinkIndex < 0 ||
      _selectedBotLinkIndex >= static_cast<int>(_botAsset->GetBotPrefab().links.size())) {
    ImGui::TextDisabled("Select a link to view details");
    ImGui::End(); // Bot Link Details
    return;
  }
  bool const isBuiltBot = _botAsset->GetBotFileType() == BotFileType::ModBotPrefab;
  bool const isReadOnly = _botAsset->IsReadOnly();
  bool modelChanged = false;
  bool paramsChanged = false;
  ImGui::PushFont(_studio->GetFont("Roboto Bold Large"));
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(_botAsset->GetBotPrefab().links[_selectedBotLinkIndex].name.c_str());
  ImGui::PopFont();
  ImGui::Separator();
  ImGui::PushID(_selectedBotLinkIndex);
  ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32_BLACK_TRANS);
  ImGui::BeginChild("Bot Link Details", ImVec2(0, 0));
  BotLinkPrefab& link = _botAsset->GetBotPrefab().links[_selectedBotLinkIndex];
  int reparentNewParent = kIndexNone;
  char const* jointLabel = _selectedBotLinkIndex == 0 ? "World Joint" : "Joint";
  // Parent-link selector, shown above the Joint/Link sections. Only non-root links of an unbuilt
  // bot can be reparented.
  if (!isBuiltBot && _selectedBotLinkIndex > 0) {
    auto& botPrefab = _botAsset->GetBotPrefab();
    int const numLinks = static_cast<int>(botPrefab.links.size());
    std::set<int> excluded;
    {
      mochi::DynamicArray<int> stack;
      stack.push_back(_selectedBotLinkIndex);
      while (!stack.empty()) {
        int const idx = stack.back();
        stack.pop_back();
        excluded.insert(idx);
        for (int child : botPrefab.links[idx]._childrenIndices) {
          stack.push_back(child);
        }
      }
    }
    int const currentParent = botPrefab.links[_selectedBotLinkIndex].parentLink;
    char const* const previewName = (currentParent >= 0 && currentParent < numLinks)
        ? botPrefab.links[currentParent].name.c_str()
        : "";
    ImGui::BeginDisabled(isReadOnly);
    if (ImGui::BeginCombo("Parent", previewName)) {
      for (int j = 0; j < numLinks; ++j) {
        if (excluded.count(j) > 0) {
          continue;
        }
        bool const isSelected = (j == currentParent);
        if (ImGui::Selectable(botPrefab.links[j].name.c_str(), isSelected) && j != currentParent) {
          reparentNewParent = j;
        }
        if (isSelected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    ImGui::EndDisabled();
  }
  if (ImGui::CollapsingHeader(jointLabel, ImGuiTreeNodeFlags_DefaultOpen)) {
    BotJointPrefab& joint = _botAsset->GetBotPrefab().joints[_selectedBotLinkIndex];
    ImGui::BeginDisabled(isBuiltBot || isReadOnly);
    paramsChanged |= ShowBotJointEditorWidgets(joint, _selectedBotLinkIndex == 0);
    ImGui::EndDisabled();
  }
  if (ImGui::CollapsingHeader("Link", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::BeginDisabled(isBuiltBot || isReadOnly);
    paramsChanged |=
        ShowBotLinkEditorWidgets(link, _studio->GetAssetManager(), !isBuiltBot, modelChanged);
    ImGui::EndDisabled(); // isBuiltBot
  }
  ImGui::EndChild();
  ImGui::PopStyleColor(); // ImGuiCol_ChildBg
  ImGui::PopID();
  if (reparentNewParent != kIndexNone) {
    ReparentBotLink(_selectedBotLinkIndex, reparentNewParent);
  }
  if (modelChanged) {
    paramsChanged = true;
    _studio->GetAssetManager().ResyncReferencer(_botAsset);
    RestageBot();
  }
  if (paramsChanged) {
    ApplyBotParamsEdit();
  }
  ImGui::End(); // Bot Link Details
}

bool BotEditor::ShowBotJointEditorWidgets(superdex::robotics::BotJointPrefab& joint, bool isRoot) {
  using namespace mochi;
  bool changed = false;
  ImGui::PushID("Joint");
  changed |= ImGui::ArticulatedJointEditor(joint, isRoot);
  // Bot-specific: effort limit, appended under the shared editor's "Dynamics" section. A leading
  // glyph signals the regime of the current value (< 0 unbounded, 0 non-actuated/free-moving,
  // > 0 finite); the value is entered directly since it is normally a specific number. A Hard joint
  // has no degrees of freedom, so its dynamics (including effort limit) are disabled.
  ImGui::BeginDisabled(joint.type == ArticulatedJointType::Hard);
  char const* effortIcon = ICON_FA_LESS_THAN_EQUAL;
  char const* effortTip = "Finite effort limit: applied effort is clamped to this magnitude.";
  if (joint.effortLimit < 0.0f) {
    effortIcon = ICON_FA_INFINITY;
    effortTip = "Unbounded: no effort limit is applied to this joint.";
  } else if (joint.effortLimit == 0.0f) {
    effortIcon = ICON_FA_ARROWS_ALT_H;
    effortTip =
        "Non-actuated: no effort can be applied, but the joint still moves freely within its range.";
  }
  ImGui::IconInputPrefix(effortIcon, effortTip);
  // DragReal (not InputFloat) so the value text is center-aligned like every other field in the
  // panel. v_min == v_max == 0 leaves it unclamped so the negative unbounded sentinel is enterable.
  if (ImGui::DragReal(
          "Effort Limit",
          &joint.effortLimit,
          0.5f,
          0.0f,
          0.0f,
          GetUnitFormat(UnitFormat::Effort, joint.type, static_cast<float>(joint.effortLimit)))) {
    changed = true;
  }
  ImGui::EndDisabled();
  ImGui::PopID();
  return changed;
}

bool BotEditor::ShowBotLinkEditorWidgets(
    superdex::robotics::BotLinkPrefab& link,
    AssetManager const& assetManager,
    bool acceptDragDropPayload,
    bool& modelChanged) {
  bool changed = false;
  ImGui::PushID("Link");
  changed |= ImGui::ArticulatedLinkEditor(
      link, _studio, assetManager, acceptDragDropPayload, modelChanged);
  ImGui::PopID(); // Link
  return changed;
}

bool BotEditor::ShowBotPrefabEditorWidgets(superdex::robotics::BotPrefab& botPrefab) {
  using namespace mochi; // for _r literals
  bool changed = false;
  ImGui::SeparatorText("World Transform (World From Root)");
  if (ImGui::DragTransformRT("World Transform", botPrefab.worldFromRoot)) {
    changed |= true;
  }
  // Cycle Joints. Soft spherical-joint constraints that close kinematic loops
  // (topologies beyond a simple tree). Parent/child links are chosen by name; valid
  // stored indices are remapped during sorting, while malformed cycles remain for validation.
  ImGui::HoverableSeparatorText("Cycle Joints");
  changed |= CopyPasteContextMenu("CycleJointsPopup", botPrefab.cycles);
  {
    // Combo to pick a link by name, storing the resolved index into linkIndex.
    auto linkCombo = [&](char const* label, int& linkIndex) {
      char const* const currentName =
          (linkIndex >= 0 && linkIndex < static_cast<int>(botPrefab.links.size()))
          ? botPrefab.links[linkIndex].name.c_str()
          : "<invalid>";
      bool comboChanged = false;
      if (ImGui::BeginCombo(label, currentName)) {
        for (int j = 0; j < static_cast<int>(botPrefab.links.size()); ++j) {
          bool const isSelected = (j == linkIndex);
          ImGui::PushID(j);
          if (ImGui::Selectable(botPrefab.links[j].name.c_str(), isSelected)) {
            linkIndex = j;
            comboChanged = true;
          }
          if (isSelected) {
            ImGui::SetItemDefaultFocus();
          }
          ImGui::PopID();
        }
        ImGui::EndCombo();
      }
      return comboChanged;
    };

    int cycleToDelete = -1;
    for (size_t i = 0; i < botPrefab.cycles.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      auto& cycle = botPrefab.cycles[i];

      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
      if (ImGui::Button(ICON_FA_TRASH)) {
        cycleToDelete = static_cast<int>(i);
      }
      ImGui::PopStyleVar();
      ImGui::SameLine();

      std::array<char, 32> label{};
      snprintf(label.data(), label.size(), "Cycle %zu###Cycle%zu", i, i);
      if (ImGui::CollapsingHeader(label.data())) {
        changed |= linkCombo("Parent Link", cycle.parentLink);
        changed |= linkCombo("Child Link", cycle.childLink);
        if (ImGui::DragTransformRT("Joint From Child Link", cycle.jointFromChildLink)) {
          changed = true;
        }
        if (ImGui::DragReal("Stiffness", &cycle.stiffness, 1.0f, 0.0f, 0.0f, "%.1f")) {
          changed = true;
        }
      }
      ImGui::PopID();
    }

    if (cycleToDelete >= 0) {
      botPrefab.cycles.erase(botPrefab.cycles.begin() + cycleToDelete);
      changed = true;
    }

    if (ImGui::Button("Add Cycle Joint")) {
      // Seed with two distinct valid link indices so the new cycle is immediately
      // editable and valid (requires at least two links).
      if (botPrefab.links.size() >= 2) {
        ArticulatedCycleJointParams newCycle;
        newCycle.parentLink = 0;
        newCycle.childLink = 1;
        botPrefab.cycles.push_back(newCycle);
        changed = true;
      }
    }
    if (botPrefab.links.size() < 2) {
      ImGui::TextDisabled("Add at least two links to create a cycle joint.");
    }
  }

  // Default Joint Position Sliders
  if (!botPrefab._dofIndices.empty()) {
    ImGui::HoverableSeparatorText("Default Pose");
    changed |= CopyPasteContextMenu("DefaultPosePopup", botPrefab.defaultPose);
    changed |= ImGui::JointPoseEditor(botPrefab, botPrefab.defaultPose);
  }

  return changed;
}

// Returns the name of the first leaf link of the current mod bot recipe, or empty if the recipe
// cannot be built (e.g. empty/invalid base). Best-effort: errors are logged and treated as no leaf.
static mochi::DynamicString FirstLeafLinkName(
    superdex::robotics::ModBotPrefab const& modBotPrefab,
    superdex::robotics::IBotLoader const& loader) {
  using namespace superdex::robotics;
  BotPrefab const built = BuildBot(modBotPrefab, loader, /*validate*/ false, mochi::ErrorLog{});
  mochi::DynamicArray<int> const leaves = FindLeafLinkIndices(built);
  return leaves.empty() ? mochi::DynamicString{} : built.links[leaves.front()].name;
}

// Prepopulates a freshly created modification with a unique name, a sensible parent/target leaf
// link, derived joint/link names, and a Hard joint (each field applied only where it exists).
static void PrepopulateModification(
    superdex::robotics::ModBotPrefab const& modBotPrefab,
    superdex::robotics::IBotLoader const& loader,
    superdex::robotics::BotMod& newMod) {
  using namespace superdex::robotics;
  mochi::DynamicString const firstLeafName = FirstLeafLinkName(modBotPrefab, loader);
  char const* const baseName = std::visit(
      [](auto&& m) -> char const* {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, AttachBot>) {
          return "AttachBot";
        } else if constexpr (std::is_same_v<T, AttachLink>) {
          return "AttachLink";
        } else if constexpr (std::is_same_v<T, ReplaceLink>) {
          return "ReplaceLink";
        } else {
          return "ReplaceLinkWithBot";
        }
      },
      newMod);
  // Generate a unique modification name of the form "<TypeName><N>".
  std::set<mochi::DynamicString> existingNames;
  for (auto const& mod : modBotPrefab.modifications) {
    std::visit([&](auto&& m) { existingNames.insert(m.name); }, mod);
  }
  mochi::DynamicString uniqueName;
  int suffix = 1;
  do {
    uniqueName = mochi::DynamicString(baseName) + mochi::DynamicString(std::to_string(suffix++));
  } while (existingNames.count(uniqueName) > 0);
  // Populate fields where they exist.
  std::visit(
      [&](auto&& m) {
        using T = std::decay_t<decltype(m)>;
        m.name = uniqueName;
        if constexpr (std::is_same_v<T, AttachBot>) {
          m.parentLinkName = firstLeafName;
          m.joint.name = uniqueName + "_joint";
          m.joint.type = mochi::ArticulatedJointType::Hard;
        } else if constexpr (std::is_same_v<T, AttachLink>) {
          m.parentLinkName = firstLeafName;
          m.joint.name = uniqueName + "_joint";
          m.joint.type = mochi::ArticulatedJointType::Hard;
          m.link.name = uniqueName + "_link";
        } else if constexpr (std::is_same_v<T, ReplaceLink>) {
          m.linkToReplace = firstLeafName;
          m.link.name = uniqueName + "_link";
        } else { // ReplaceLinkWithBot
          m.linkToReplace = firstLeafName;
        }
      },
      newMod);
}

// Returns the valid link names present in the mod bot after applying modifications [0, modIndex).
// Used to populate parent/target link dropdowns for the modification at position modIndex.
static mochi::DynamicArray<mochi::DynamicString> LinkNamesUpToMod(
    superdex::robotics::ModBotPrefab const& modBotPrefab,
    superdex::robotics::IBotLoader const& loader,
    int modIndex) {
  using namespace superdex::robotics;
  ModBotPrefab truncated = modBotPrefab;
  if (modIndex >= 0 && modIndex < isize(truncated.modifications)) {
    truncated.modifications.erase(
        truncated.modifications.begin() + modIndex, truncated.modifications.end());
  }
  BotPrefab const built = BuildBot(truncated, loader, /*validate*/ false, mochi::ErrorLog{});
  mochi::DynamicArray<mochi::DynamicString> names;
  names.reserve(isize(built.links));
  for (auto const& link : built.links) {
    names.push_back(link.name);
  }
  return names;
}

// True if any bot/link/joint validation issue was recorded.
static bool ValidateResultsHaveIssues(superdex::robotics::ValidateResults const& results) {
  if (!results.botIssues.empty()) {
    return true;
  }
  for (auto const& issues : results.linkIssues) {
    if (!issues.empty()) {
      return true;
    }
  }
  for (auto const& issues : results.jointIssues) {
    if (!issues.empty()) {
      return true;
    }
  }
  return false;
}

// First recorded validation issue message, for use as a tooltip.
static mochi::DynamicString FirstValidateIssue(superdex::robotics::ValidateResults const& results) {
  if (!results.botIssues.empty()) {
    return results.botIssues[0];
  }
  for (auto const& issues : results.linkIssues) {
    if (!issues.empty()) {
      return issues[0];
    }
  }
  for (auto const& issues : results.jointIssues) {
    if (!issues.empty()) {
      return issues[0];
    }
  }
  return {};
}

// Find the first modification that makes the recipe fail to build or produces an
// invalid bot. Builds recipe prefixes of increasing length (base only, then base
// + first mod, etc.); the first prefix that fails to build or fails validation
// pinpoints the offending modification. Returns kIndexNone if the whole recipe
// builds and validates cleanly, or if the base bot alone is already bad (the base
// is the culprit, not a modification). outIsBuildError is set to true for a hard
// build failure (chain stops) and false for a validation failure (chain completes
// but is invalid). Disabled mods are skipped by BuildBot, so a prefix ending on a
// disabled mod matches its predecessor and never becomes the culprit.
int BotEditor::ComputeFirstFailingModIndex(
    superdex::robotics::ModBotPrefab const& params,
    bool& outIsBuildError) {
  using namespace superdex::robotics;
  IBotLoader const& loader = _studio->GetBotLoader();
  int const numMods = mochi::isize(params.modifications);
  for (int prefixLen = 0; prefixLen <= numMods; ++prefixLen) {
    ModBotPrefab truncated = params;
    // Keep only the first prefixLen modifications.
    truncated.modifications.erase(
        truncated.modifications.begin() + prefixLen, truncated.modifications.end());
    mochi::ErrorLog buildError;
    BotPrefab const built = BuildBot(truncated, loader, /*validate*/ false, buildError);
    if (!buildError.IsOK()) {
      _modBuildError = static_cast<mochi::Error&>(buildError).GetDescription();
      outIsBuildError = true;
      // prefixLen == 0 means the base bot itself fails; not a modification.
      return prefixLen == 0 ? kIndexNone : prefixLen - 1;
    }
    // This probe validates prefixes that are expected to fail, and reports the first issue
    // through the UI (_modBuildError), so nothing here is logged.
    ValidateResults results;
    results.suppressWarnings = true;
    mochi::Error validateError;
    Validate(built, &results, validateError);
    if (ValidateResultsHaveIssues(results)) {
      _modBuildError = FirstValidateIssue(results);
      outIsBuildError = false;
      return prefixLen == 0 ? kIndexNone : prefixLen - 1;
    }
  }
  _modBuildError = {};
  outIsBuildError = false;
  return kIndexNone;
}

// Rebuild the asset and refresh the cached failing-mod status. When the recipe
// builds and validates cleanly the status is cleared; otherwise the first
// offending modification is located (the last good prefab is retained by
// BotAsset::Rebuild). Returns whether the build itself succeeded (validation
// failures still build a displayable bot).
bool BotEditor::RecomputeModBuildStatus() {
  bool buildOk = false;
  _botAsset->Rebuild(_studio->GetBotLoader(), &buildOk);
  // On a failed build the retained (last good) prefab is validated instead, so
  // rely on buildOk to detect build failures and the validate results otherwise.
  bool const isModBot =
      _botAsset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab;
  bool const isBad = !buildOk || ValidateResultsHaveIssues(_botAsset->GetValidateResults());
  if (isModBot && isBad) {
    _firstFailingModIndex =
        ComputeFirstFailingModIndex(_botAsset->GetModBotPrefab(), _firstFailingIsBuildError);
  } else {
    _firstFailingModIndex = superdex::robotics::kIndexNone;
    _firstFailingIsBuildError = false;
    _modBuildError = {};
  }
  return buildOk;
}

bool BotEditor::ShowModBotPrefabEditorWidgets(
    superdex::robotics::ModBotPrefab& modBotPrefab,
    AssetManager const& assetManager) {
  bool changed = false;

  ImGui::SeparatorText("Base Bot");
  changed |=
      ImGui::AssetSlot("Base Bot", modBotPrefab.base, assetManager, _studio, AssetType::Bot, true);

  // Modifications stack
  ImGui::HoverableSeparatorText("Modifications");
  changed |= CopyPasteContextMenu("ModificationsPopup", modBotPrefab.modifications);

  int indexToDelete = -1;
  int indexToMoveUp = -1;
  int indexToMoveDown = -1;

  auto linkNameCombo = [&](char const* label, mochi::DynamicString& value, int modIndex) -> bool {
    bool comboChanged = false;
    if (ImGui::BeginCombo(label, value.c_str())) {
      auto const names = LinkNamesUpToMod(modBotPrefab, _studio->GetBotLoader(), modIndex);
      for (auto const& name : names) {
        bool const isSelected = (name == value);
        if (ImGui::Selectable(name.c_str(), isSelected) && value != name) {
          value = name;
          comboChanged = true;
        }
        if (isSelected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    return comboChanged;
  };
  for (size_t i = 0; i < modBotPrefab.modifications.size(); ++i) {
    ImGui::PushID(static_cast<int>(i));

    // A failing mod is highlighted; mods below it have no effect yet (BuildBot
    // stops at the failure) so they are dimmed. State is read from the cached
    // index — no building happens here.
    bool const isFailing = (static_cast<int>(i) == _firstFailingModIndex);
    bool const isBelowFailure = _firstFailingModIndex != superdex::robotics::kIndexNone &&
        _firstFailingIsBuildError && static_cast<int>(i) > _firstFailingModIndex;
    ImGuiStyle const& style = ImGui::GetStyle();
    if (isBelowFailure) {
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, style.Alpha * style.DisabledAlpha);
    }

    auto& mod = modBotPrefab.modifications[i];
    changed |= std::visit(
        [&](auto&& m) -> bool {
          using T = std::decay_t<decltype(m)>;
          bool modChanged = false;

          char const* typeName = std::is_same_v<T, superdex::robotics::AttachBot> ? "Attach Bot"
              : std::is_same_v<T, superdex::robotics::AttachLink>                 ? "Attach Link"
              : std::is_same_v<T, superdex::robotics::ReplaceLink>                ? "Replace Link"
                                                                   : "Replace Link With Bot";
          char const* warnIcon = isFailing ? " " ICON_FA_EXCLAMATION_TRIANGLE : "";
          std::array<char, 128> label{};
          if (m.name.empty()) {
            snprintf(label.data(), label.size(), "%s #%zu%s###Mod%zu", typeName, i, warnIcon, i);
          } else {
            snprintf(
                label.data(),
                label.size(),
                "%s (%s)%s###Mod%zu",
                typeName,
                m.name.c_str(),
                warnIcon,
                i);
          }

          ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
          if (ImGui::Button(ICON_FA_TRASH)) {
            indexToDelete = static_cast<int>(i);
          }
          ImGui::SameLine();
          if (ImGui::Checkbox("##Enabled", &m.enabled)) {
            modChanged = true;
            changed = true;
          }
          ImGui::SameLine();
          ImGui::BeginDisabled(i == 0);
          if (ImGui::Button(ICON_FA_CARET_UP)) {
            indexToMoveUp = static_cast<int>(i);
          }
          ImGui::EndDisabled();
          ImGui::SameLine();
          ImGui::BeginDisabled(i == modBotPrefab.modifications.size() - 1);
          if (ImGui::Button(ICON_FA_CARET_DOWN)) {
            indexToMoveDown = static_cast<int>(i);
          }
          ImGui::EndDisabled();
          ImGui::SameLine();
          ImGui::PopStyleVar();

          if (isFailing) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
          }
          bool const headerOpen = ImGui::CollapsingHeader(label.data());
          if (isFailing && ImGui::IsItemHovered() && !_modBuildError.empty()) {
            ImGui::SetTooltip("%s", _modBuildError.c_str());
          }
          if (isFailing) {
            ImGui::PopStyleColor();
          }
          if (headerOpen) {
            if (ImGui::InputText("Name", &m.name, ImGuiInputTextFlags_CharsNoBlank)) {
              modChanged = true;
              changed = true;
            }
            if constexpr (std::is_same_v<T, superdex::robotics::AttachBot>) {
              modChanged |= ImGui::AssetSlot(
                  "Attach Bot", m.path, assetManager, _studio, AssetType::Bot, true);
              modChanged |= ImGui::InputText("Prefix", &m.prefix);
              modChanged |= linkNameCombo("Parent Link", m.parentLinkName, static_cast<int>(i));
              if (ImGui::CollapsingHeader("Joint", ImGuiTreeNodeFlags_DefaultOpen)) {
                modChanged |= ShowBotJointEditorWidgets(m.joint, false);
              }
            } else if constexpr (std::is_same_v<T, superdex::robotics::ReplaceLinkWithBot>) {
              modChanged |= ImGui::AssetSlot(
                  "Replace With", m.path, assetManager, _studio, AssetType::Bot, true);
              modChanged |= ImGui::InputText("Prefix", &m.prefix);
              modChanged |= linkNameCombo("Link to Replace", m.linkToReplace, static_cast<int>(i));
            } else if constexpr (std::is_same_v<T, superdex::robotics::ReplaceLink>) {
              modChanged |= linkNameCombo("Link to Replace", m.linkToReplace, static_cast<int>(i));
              bool modelChanged = false;
              modChanged |= ShowBotLinkEditorWidgets(m.link, assetManager, true, modelChanged);
            } else {
              modChanged |= linkNameCombo("Parent Link", m.parentLinkName, static_cast<int>(i));
              if (ImGui::CollapsingHeader("Joint", ImGuiTreeNodeFlags_DefaultOpen)) {
                modChanged |= ShowBotJointEditorWidgets(m.joint, false);
              }
              if (ImGui::CollapsingHeader("Link", ImGuiTreeNodeFlags_DefaultOpen)) {
                bool modelChanged = false;
                modChanged |= ShowBotLinkEditorWidgets(m.link, assetManager, true, modelChanged);
              }
            }
          }
          return modChanged;
        },
        mod);

    if (isBelowFailure) {
      ImGui::PopStyleVar();
    }
    ImGui::PopID();
  }

  // Handle reordering
  if (indexToMoveUp > 0) {
    std::swap(
        modBotPrefab.modifications[indexToMoveUp], modBotPrefab.modifications[indexToMoveUp - 1]);
    changed = true;
  }
  if (indexToMoveDown >= 0 &&
      indexToMoveDown < static_cast<int>(modBotPrefab.modifications.size()) - 1) {
    std::swap(
        modBotPrefab.modifications[indexToMoveDown],
        modBotPrefab.modifications[indexToMoveDown + 1]);
    changed = true;
  }

  // Handle deletion
  if (indexToDelete >= 0) {
    modBotPrefab.modifications.erase(modBotPrefab.modifications.begin() + indexToDelete);
    changed = true;
  }

  // Add new mod UI
  ImGui::Separator();
  static int selectedModType = 0;
  ImGui::Combo(
      "New Mod Type", &selectedModType, "AttachBot\0AttachLink\0ReplaceLink\0ReplaceLinkWithBot\0");
  if (ImGui::Button("Add Modification")) {
    superdex::robotics::BotMod newMod;
    if (selectedModType == 0) {
      newMod = superdex::robotics::AttachBot{};
    } else if (selectedModType == 1) {
      newMod = superdex::robotics::AttachLink{};
    } else if (selectedModType == 2) {
      newMod = superdex::robotics::ReplaceLink{};
    } else {
      newMod = superdex::robotics::ReplaceLinkWithBot{};
    }
    PrepopulateModification(modBotPrefab, _studio->GetBotLoader(), newMod);
    modBotPrefab.modifications.push_back(std::move(newMod));
    changed = true;
  }

  return changed;
}

void BotEditor::ShowBotContactWindow(bool* open) {
  if (_stage.IsEmpty()) {
    return;
  }
  if (!ImGui::Begin("Bot Contact", open)) {
    ImGui::End();
    return;
  }
  auto& prefab = _botAsset->GetBotPrefab();
  auto& filters = prefab.contactOverrides;
  bool isReadOnly = _botAsset->IsReadOnly() ||
      _botAsset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab;

  int const numLinks = static_cast<int>(prefab.links.size());
  if (numLinks <= 1) {
    ImGui::TextUnformatted("Not enough links to define contact filters.");
    ImGui::End();
    return;
  }

  // Encapsulates the implicit-disable mask and the override-editing rules.
  BotContactFilterBuilder builder(prefab);
  bool modified = false;

  float const avail = ImGui::GetContentRegionAvail().x;
  float const btnSpacing = ImGui::GetStyle().ItemSpacing.x;
  float const btnW = (avail - 3 * btnSpacing) / 4.0f;
  // Read-only/mod bots can't edit filters, but the grid must stay scrollable, so disable
  // only the interactive controls (buttons + checkboxes) instead of the whole window.
  ImGui::BeginDisabled(isReadOnly);
  if (ImGui::Button("Disable Colliding", ImVec2(btnW, 0))) {
    mochi::ErrorLog err;
    BotContactProbe probe(prefab, _studio->GetMochiContext(), _studio->GetBotLoader(), err);
    auto pairs = probe.DetectCollidingLinks(prefab.defaultPose, err);
    if (err.IsOK()) {
      for (auto const& pr : pairs) {
        builder.SetFilter(pr.linkA, pr.linkB, false);
      }
      modified = true;
    }
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Step the bot at its default pose and disable contact for any link\n"
        "pairs found in collision. Pairs that the physics engine already implicitly disables\n"
        "have any user override removed instead of writing a redundant entry.");
  }
  ImGui::SameLine();
  if (ImGui::Button("Disable All", ImVec2(btnW, 0))) {
    builder.SetAll(false);
    modified = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Enable All", ImVec2(btnW, 0))) {
    builder.SetAll(true);
    modified = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Reset", ImVec2(btnW, 0))) {
    if (!filters.empty()) {
      filters.clear();
      modified = true;
    }
  }

  static int estimateIterations = 10000;
  {
    bool const running = _contactEstimator.IsRunning();
    if (running) {
      if (ImGui::Button("Cancel", ImVec2(btnW, 0))) {
        _contactEstimator.RequestCancel();
      }
    } else {
      if (ImGui::Button("Estimate Non-Colliding", ImVec2(btnW, 0))) {
        _contactEstimator.Start(
            _studio->GetMochiContext(), _studio->GetBotLoader(), prefab, estimateIterations);
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Sample random poses and disable link pairs that never collided.\n"
            "Implicitly tests Min, Max, Mid, and Zero poses first.\n"
            "All filters are CLEARED before the result is applied, so run this before any manual operations.\n"
            "Heuristic based; rare collisions may be missed. Sanity check the results.");
      }
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(running);
    ImGui::SetNextItemWidth(btnW);
    ImGui::DragInt("Iterations", &estimateIterations, 1.0f, 1, 1000000, "%d");
    ImGui::EndDisabled();
    BotContactEstimator::Progress const progress = _contactEstimator.GetProgress();
    if (running) {
      ImGui::SameLine();
      float const frac = progress.total > 0
          ? static_cast<float>(progress.completed) / static_cast<float>(progress.total)
          : 0.0f;
      std::array<char, 64> overlay{};
      snprintf(overlay.data(), overlay.size(), "%d / %d", progress.completed, progress.total);
      ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0), overlay.data());
    }

    // Each frame while running: snapshot the worker's last sampled pose and
    // animate the bot so the viewport reflects what's being tested.
    if (running) {
      mochi::DynamicArray<mochi::real> poseCopy = _contactEstimator.GetLatestPose();
      if (!poseCopy.empty() && poseCopy.size() == prefab.defaultPose.size()) {
        prefab.defaultPose = std::move(poseCopy);
        _botAsset->Rebuild(_studio->GetBotLoader());
        _botAsset->SetDirty(true);
        RestageBot();
      }
    }
    // Worker thread finished? Take the result (joins the worker) and apply it.
    if (_contactEstimator.IsFinished()) {
      BotContactEstimator::Result result = _contactEstimator.TakeResult();
      // Restore the pose the bot was at before the run started.
      if (!result.originalPose.empty() && result.originalPose.size() == prefab.defaultPose.size()) {
        prefab.defaultPose = std::move(result.originalPose);
        _botAsset->Rebuild(_studio->GetBotLoader());
        _botAsset->SetDirty(true);
        RestageBot();
      }
      // Only apply results on natural completion, not on cancellation.
      if (!result.cancelled) {
        // Clear all existing filters first so stale entries from earlier runs
        // (or manual edits) can't shadow the result of this run.
        filters.clear();
        for (auto const& p : result.survivors) {
          builder.SetFilter(p.first, p.second, false);
        }
        modified = true;
      }
    }
  }
  ImGui::EndDisabled();

  // The grid is an N-by-N upper triangle that can grow past the window; host it in a scrolling
  // child (filling the remaining space, both scrollbars) so no cells become unreachable when the
  // window is small. Coordinates below are relative to this child's (scrolled) content region.
  ImGui::BeginChild("##ContactGrid", ImVec2(0.0f, 0.0f), 0, ImGuiWindowFlags_HorizontalScrollbar);

  // Layout: a strict upper-triangle matrix (cells where i < j). Column labels
  // are drawn vertically on top; row labels are drawn on the right.
  float const cellSize = ImGui::GetFrameHeight();
  float const fontSize = ImGui::GetFontSize();
  float const spacing = ImGui::GetStyle().ItemSpacing.x;
  float maxLabelW = 0.0f;
  for (auto const& link : prefab.links) {
    float const w = ImGui::CalcTextSize(link.name.c_str()).x;
    if (w > maxLabelW) {
      maxLabelW = w;
    }
  }
  // Rotated text: its on-screen height equals the (horizontal) text width.
  float const colHeaderH = maxLabelW + spacing;

  ImVec2 const origin = ImGui::GetCursorScreenPos();
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  ImU32 const textColor = ImGui::GetColorU32(ImGuiCol_Text);

  // Strict upper triangle (i < j): skip j=0 (no cells) and the last row (no cells).
  int const numCols = numLinks - 1;
  int const numRows = numLinks - 1;

  // Reserve top header area; column labels are drawn after cells so that
  // hover state is known (deferred draw).
  ImGui::Dummy(ImVec2(numCols * cellSize, colHeaderH));
  ImVec2 const cellsOrigin = ImGui::GetCursorScreenPos();

  // Render cells; track which row/col is hovered.
  int hoveredI = -1;
  int hoveredJ = -1;
  ImGui::BeginDisabled(isReadOnly || _contactEstimator.IsRunning());
  for (int i = 0; i < numRows; ++i) {
    for (int j = i + 1; j < numLinks; ++j) {
      ImGui::SetCursorScreenPos(
          ImVec2(cellsOrigin.x + (j - 1) * cellSize, cellsOrigin.y + i * cellSize));
      bool const implicitOff = builder.IsImplicitlyDisabled(i, j);
      bool const hasFilter = builder.HasOverride(i, j);
      bool checked = builder.IsEnabled(i, j);
      ImGui::PushID(i * numLinks + j);
      // Dim cells that aren't user-overridden (i.e. still showing the implicit default).
      if (!hasFilter) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
      }
      if (ImGui::Checkbox("##c", &checked)) {
        builder.SetFilter(i, j, checked);
        modified = true;
      }
      if (!hasFilter) {
        ImGui::PopStyleVar();
      }
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        hoveredI = i;
        hoveredJ = j;
        ImGui::SetTooltip(
            "%s " ICON_FA_ARROWS_ALT_H " %s%s",
            prefab.links[i].name.c_str(),
            prefab.links[j].name.c_str(),
            implicitOff ? "  (implicitly disabled)" : "");
      }
      ImGui::PopID();
    }
  }

  // Reserve cell grid + row label area so the window sizes correctly.
  ImGui::EndDisabled();
  ImGui::SetCursorScreenPos(cellsOrigin);
  ImGui::Dummy(ImVec2(numCols * cellSize + spacing + maxLabelW, numRows * cellSize));

  // Rows are red, columns are blue.
  constexpr ImU32 kRowHoverColor = IM_COL32(233, 55, 81, 255);
  constexpr ImU32 kColHoverColor = IM_COL32(46, 134, 233, 255);
  ImU32 const disabledColor = ImGui::GetColorU32(ImGuiCol_TextDisabled);

  // Hover-driven highlight state, tracked separately for rows and columns
  mochi::DynamicArray<bool> rowHot(numLinks, false);
  mochi::DynamicArray<bool> colHot(numLinks, false);

  // Hovering a cell highlights its row link (red) and its column link (blue).
  if (hoveredI >= 0) {
    rowHot[hoveredI] = true;
  }
  if (hoveredJ >= 0) {
    colHot[hoveredJ] = true;
  }

  // Hovering a row/column label highlights that link plus every link it can still
  // collide with (enabled pair) in the opposite color: a hovered row highlights the
  // collidable columns, and a hovered column highlights the collidable rows.
  if (!_contactEstimator.IsRunning() && ImGui::IsWindowHovered()) {
    // Column labels (links 1..numLinks-1) sit in the vertical header above each column.
    for (int j = 1; j < numLinks; ++j) {
      float const cellLeft = cellsOrigin.x + (j - 1) * cellSize;
      if (ImGui::IsMouseHoveringRect(
              ImVec2(cellLeft, origin.y), ImVec2(cellLeft + cellSize, origin.y + colHeaderH))) {
        colHot[j] = true;
        for (int k = 0; k < numLinks; ++k) {
          if (k != j && builder.IsEnabled(j, k)) {
            rowHot[k] = true;
          }
        }
      }
    }
    // Row labels (links 0..numRows-1) sit in the strip to the right of the grid.
    float const rowLabelX = cellsOrigin.x + numCols * cellSize + spacing;
    for (int i = 0; i < numRows; ++i) {
      float const rowTop = cellsOrigin.y + i * cellSize;
      if (ImGui::IsMouseHoveringRect(
              ImVec2(rowLabelX, rowTop), ImVec2(rowLabelX + maxLabelW, rowTop + cellSize))) {
        rowHot[i] = true;
        for (int k = 0; k < numLinks; ++k) {
          if (k != i && builder.IsEnabled(i, k)) {
            colHot[k] = true;
          }
        }
      }
    }
  }

  auto labelColor = [&](int linkIdx, bool hot, ImU32 hotColor) -> ImU32 {
    if (hot) {
      return hotColor;
    }
    return prefab.links[linkIdx].renderModelFile.empty() ? disabledColor : textColor;
  };

  // Deferred draw of column labels (hover-aware).
  for (int j = 1; j < numLinks; ++j) {
    char const* name = prefab.links[j].name.c_str();
    float const cellLeft = cellsOrigin.x + (j - 1) * cellSize;
    float const textX = cellLeft + (cellSize - fontSize) * 0.5f;
    float const textY = origin.y + colHeaderH;
    ImPlot::AddTextVertical(
        drawList, ImVec2(textX, textY), labelColor(j, colHot[j], kColHoverColor), name);
  }
  // Deferred draw of row labels (hover-aware).
  for (int i = 0; i < numRows; ++i) {
    char const* name = prefab.links[i].name.c_str();
    float const labelX = cellsOrigin.x + numCols * cellSize + spacing;
    float const labelY = cellsOrigin.y + i * cellSize + (cellSize - fontSize) * 0.5f;
    drawList->AddText(ImVec2(labelX, labelY), labelColor(i, rowHot[i], kRowHoverColor), name);
  }

  ImGui::EndChild();

  // Per-frame highlight: tint the corresponding link objects in the viewport.
  // Colors match the row/col label colors (red/blue). A link is never both.
  constexpr filament::math::float3 kRowHighlight = {233.f / 255.f, 55.f / 255.f, 81.f / 255.f};
  constexpr filament::math::float3 kColHighlight = {46.f / 255.f, 134.f / 255.f, 233.f / 255.f};
  auto highlightLink = [&](int linkIdx, filament::math::float3 color) {
    if (linkIdx < 0 || linkIdx >= _stage.GetNumActors()) {
      return;
    }
    if (mochi_renderer::SceneObject* obj = _stage.GetActors()[linkIdx].sceneObject) {
      _viewport->HighlightSceneObject(obj, color);
    }
  };
  for (int k = 0; k < numLinks; ++k) {
    if (rowHot[k]) {
      highlightLink(k, kRowHighlight);
    } else if (colHot[k]) {
      highlightLink(k, kColHighlight);
    }
  }

  if (modified) {
    _botAsset->Rebuild(_studio->GetBotLoader());
    _botAsset->SetDirty(true);
    GetUndoStack().PushNow();
  }

  ImGui::End();
}

void BotEditor::ShowBotTransmissionsWindow(bool* open) {
  if (!ImGui::Begin("Bot Transmissions", open)) {
    ImGui::End();
    return;
  }

  if (_stage.IsEmpty()) {
    ImGui::TextDisabled("No bot loaded");
    ImGui::End();
    return;
  }

  auto& botPrefab = _botAsset->GetBotPrefab();
  using namespace mochi; // for _r literals
  bool changed = false;

  // Local helper to render displacement-control actuator UI shared by Linear Transmissions and
  // Spatial Tendons. Using a lambda avoids introducing a new top-level member while keeping both
  // sections synchronized.
  auto showDisplacementControlActuator = [&](auto& actuator) -> bool {
    bool localChanged = false;
    // Displacement-control actuator parameters (consumed at bot instantiation time by
    // RoboticsContext). The stiffness and damping have linear (length-based) units, so we borrow
    // the Prismatic-joint formats from GetUnitFormat. Displayed in 2x2 grid to save vertical space.
    ImGui::SeparatorText("Displacement Control Actuator");
    if (ImGui::BeginTable("##DisplacementControlActuator", 2, ImGuiTableFlags_SizingStretchSame)) {
      ImGui::TableSetupColumn("##Col0", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      // Leave room on right for label by not using -FLT_MIN; default width lets ImGui draw label
      // to the right of the drag widget as in the original vertical layout.
      if (ImGui::DragReal(
              "Target Displacement",
              &actuator.targetDisplacement,
              0.0001f,
              0.0f,
              0.0f,
              GetUnitFormat(UnitFormat::Length, actuator.targetDisplacement))) {
        localChanged = true;
      }
      ImGui::TableNextColumn();
      ImGui::AlignTextToFramePadding();
      if (ImGui::Checkbox("Allow Compressive Force", &actuator.allowCompressiveForce)) {
        localChanged = true;
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      if (ImGui::DragReal(
              "Stiffness",
              &actuator.stiffness,
              1.0f,
              0.0f,
              std::numeric_limits<float>::max(),
              GetUnitFormat(
                  UnitFormat::Stiffness,
                  mochi::ArticulatedJointType::Prismatic,
                  actuator.stiffness))) {
        localChanged = true;
      }
      ImGui::TableNextColumn();
      if (ImGui::DragReal(
              "Damping",
              &actuator.damping,
              0.01f,
              0.0f,
              std::numeric_limits<float>::max(),
              GetUnitFormat(
                  UnitFormat::Damping, mochi::ArticulatedJointType::Prismatic, actuator.damping))) {
        localChanged = true;
      }
      ImGui::EndTable();
    }
    return localChanged;
  };

  /* @brief Result from rendering a per-item collapsible header with reordering controls. */
  struct ItemHeaderResult {
    bool isOpen = false;
    bool moveUp = false;
    bool moveDown = false;
    bool deleted = false;
  };

  // Local helper to render a collapsible header with right-aligned up/down/delete buttons.
  // Shared between Fixed Transmissions and Spatial Tendons to keep UI scaffolding in sync.
  // Returns ItemHeaderResult indicating header open state and requested actions.
  auto showItemHeader = [&](size_t index,
                            size_t totalCount,
                            auto const& name,
                            char const* defaultPrefix,
                            char const* idPrefix) -> ItemHeaderResult {
    ItemHeaderResult result;

    std::array<char, 128> headerLabel{};
    if (name.empty()) {
      snprintf(
          headerLabel.data(),
          headerLabel.size(),
          "%s #%zu###%s%zu",
          defaultPrefix,
          index,
          idPrefix,
          index);
    } else {
      snprintf(headerLabel.data(), headerLabel.size(), "%s###%s%zu", name.c_str(), idPrefix, index);
    }

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
    ImGui::SetNextItemAllowOverlap();
    result.isOpen = ImGui::CollapsingHeader(headerLabel.data());

    // Calculate total width of the three buttons to right-align them.
    ImGuiStyle const& style = ImGui::GetStyle();
    float const buttonWidth = ImGui::CalcTextSize(ICON_FA_CARET_UP).x + style.FramePadding.x * 2.0f;
    // All three icons have similar width, use the same for simplicity.
    float const totalButtonWidth = buttonWidth * 3.0f + style.ItemSpacing.x * 2.0f;

    // Position buttons at the right edge, on the same line as the header.
    // GetWindowContentRegionMax().x and SameLine(x) are both window-local and
    // independent of the current indent, so no indent compensation is needed.
    ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - totalButtonWidth);
    ImGui::BeginDisabled(index == 0);
    if (ImGui::Button(ICON_FA_CARET_UP)) {
      result.moveUp = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(index == totalCount - 1);
    if (ImGui::Button(ICON_FA_CARET_DOWN)) {
      result.moveDown = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_TRASH)) {
      result.deleted = true;
    }
    ImGui::PopStyleVar();

    return result;
  };

  // Local helper to apply move-up/move-down/delete bookkeeping to a vector and keep the
  // parallel expanded-state array in sync so the 3D highlight follows the item.
  auto applyItemReorder =
      [&](auto& items, auto& expanded, int toDelete, int toMoveUp, int toMoveDown) -> bool {
    bool localChanged = false;
    if (toMoveUp > 0) {
      std::swap(items[toMoveUp], items[toMoveUp - 1]);
      if (toMoveUp < static_cast<int>(expanded.size()) &&
          toMoveUp - 1 < static_cast<int>(expanded.size())) {
        std::swap(expanded[toMoveUp], expanded[toMoveUp - 1]);
      }
      localChanged = true;
    }
    if (toMoveDown >= 0 && toMoveDown < static_cast<int>(items.size()) - 1) {
      std::swap(items[toMoveDown], items[toMoveDown + 1]);
      if (toMoveDown < static_cast<int>(expanded.size()) &&
          toMoveDown + 1 < static_cast<int>(expanded.size())) {
        std::swap(expanded[toMoveDown], expanded[toMoveDown + 1]);
      }
      localChanged = true;
    }
    if (toDelete >= 0) {
      items.erase(items.begin() + toDelete);
      if (toDelete >= 0 && toDelete < static_cast<int>(expanded.size())) {
        expanded.erase(expanded.begin() + toDelete);
      }
      localChanged = true;
    }
    return localChanged;
  };

  // ========================================================================
  // Linear Transmissions Section
  // ========================================================================
  if (ImGui::CollapsingHeader("Linear Transmissions", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Indent();
    if (CopyPasteContextMenu("LinearTransmissionsPopup", botPrefab.linearTransmissions)) {
      changed = true;
      _linearTransmissionExpanded.resize(botPrefab.linearTransmissions.size());
      for (bool& expanded : _linearTransmissionExpanded) {
        expanded = false;
      }
    } else {
      _linearTransmissionExpanded.resize(botPrefab.linearTransmissions.size());
    }

    int transmissionToDelete = -1;
    int transmissionToMoveUp = -1;
    int transmissionToMoveDown = -1;
    for (size_t iTransmission = 0; iTransmission < botPrefab.linearTransmissions.size();
         ++iTransmission) {
      ImGui::PushID(static_cast<int>(iTransmission));
      auto& transmission = botPrefab.linearTransmissions[iTransmission];

      // Defensive: keep the three parallel arrays the same length in case of malformed JSON.
      size_t const maxLen = std::max(
          {transmission.jointIndices.size(),
           transmission.jointCoefficients.size(),
           transmission.jointAxisDisps.size()});
      if (transmission.jointIndices.size() != maxLen) {
        transmission.jointIndices.resize(maxLen, 0);
        changed = true;
      }
      if (transmission.jointCoefficients.size() != maxLen) {
        transmission.jointCoefficients.resize(maxLen, 0_r);
        changed = true;
      }
      if (transmission.jointAxisDisps.size() != maxLen) {
        transmission.jointAxisDisps.resize(maxLen, 0_r);
        changed = true;
      }

      ItemHeaderResult headerResult = showItemHeader(
          iTransmission,
          botPrefab.linearTransmissions.size(),
          transmission.name,
          "Transmission",
          "Transmission");
      if (headerResult.moveUp) {
        transmissionToMoveUp = static_cast<int>(iTransmission);
      }
      if (headerResult.moveDown) {
        transmissionToMoveDown = static_cast<int>(iTransmission);
      }
      if (headerResult.deleted) {
        transmissionToDelete = static_cast<int>(iTransmission);
      }
      if (iTransmission < _linearTransmissionExpanded.size()) {
        _linearTransmissionExpanded[iTransmission] = headerResult.isOpen;
      }

      if (headerResult.isOpen) {
        ImGui::Indent();
        if (ImGui::InputText("Name", &transmission.name, ImGuiInputTextFlags_CharsNoBlank)) {
          changed = true;
        }

        int rowToDelete = -1;
        auto tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
            ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp;
        if (ImGui::BeginTable("##TransmissionJoints", 4, tableFlags)) {
          ImGui::TableSetupColumn("Joint", ImGuiTableColumnFlags_WidthStretch, 3.0f);
          ImGui::TableSetupColumn("Coefficient", ImGuiTableColumnFlags_WidthStretch, 2.0f);
          ImGui::TableSetupColumn("Axis Disp", ImGuiTableColumnFlags_WidthStretch, 2.0f);
          ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 28.0f);
          ImGui::TableHeadersRow();

          for (size_t iRow = 0; iRow < transmission.jointIndices.size(); ++iRow) {
            ImGui::PushID(static_cast<int>(iRow));
            ImGui::TableNextRow();

            // Joint dropdown — filtered to single-DOF (actuated) joints.
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            int const currentJointIndex = transmission.jointIndices[iRow];
            char const* currentName =
                (currentJointIndex >= 0 &&
                 currentJointIndex < static_cast<int>(botPrefab.joints.size()))
                ? botPrefab.joints[currentJointIndex].name.c_str()
                : "<invalid>";
            if (ImGui::BeginCombo("##Joint", currentName)) {
              for (int dofIdx : botPrefab._dofIndices) {
                if (dofIdx < 0 || dofIdx >= static_cast<int>(botPrefab.joints.size())) {
                  continue;
                }
                auto const& dofJoint = botPrefab.joints[dofIdx];
                bool const isSelected = (dofIdx == currentJointIndex);
                ImGui::PushID(dofIdx);
                if (ImGui::Selectable(dofJoint.name.c_str(), isSelected)) {
                  transmission.jointIndices[iRow] = dofIdx;
                  changed = true;
                }
                if (isSelected) {
                  ImGui::SetItemDefaultFocus();
                }
                ImGui::PopID();
              }
              ImGui::EndCombo();
            }

            // Coefficient
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::DragReal(
                    "##Coefficient",
                    &transmission.jointCoefficients[iRow],
                    0.001f,
                    0.0f,
                    0.0f,
                    GetUnitFormat(UnitFormat::Length, transmission.jointCoefficients[iRow]))) {
              changed = true;
            }

            // Axis Displacement
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::DragReal(
                    "##AxisDisp",
                    &transmission.jointAxisDisps[iRow],
                    0.0001f,
                    0.0f,
                    0.0f,
                    GetUnitFormat(UnitFormat::Length, transmission.jointAxisDisps[iRow]))) {
              changed = true;
            }

            // Delete row
            ImGui::TableNextColumn();
            if (ImGui::Button(ICON_FA_TRASH)) {
              rowToDelete = static_cast<int>(iRow);
            }

            ImGui::PopID();
          }
          ImGui::EndTable();
        }

        if (rowToDelete >= 0) {
          transmission.jointIndices.erase(transmission.jointIndices.begin() + rowToDelete);
          transmission.jointCoefficients.erase(
              transmission.jointCoefficients.begin() + rowToDelete);
          transmission.jointAxisDisps.erase(transmission.jointAxisDisps.begin() + rowToDelete);
          changed = true;
        }

        if (ImGui::Button("Add Joint")) {
          // Default to the first actuated DOF joint if available, else 0.
          int const defaultJointIndex =
              botPrefab._dofIndices.empty() ? 0 : botPrefab._dofIndices[0];
          transmission.jointIndices.push_back(defaultJointIndex);
          transmission.jointCoefficients.push_back(0_r);
          transmission.jointAxisDisps.push_back(0_r);
          changed = true;
        }

        // Displacement-control actuator parameters shared via local lambda to keep UI synchronized
        // across transmission types.
        changed |= showDisplacementControlActuator(transmission);
        ImGui::Unindent();
      }

      ImGui::PopID();
    }

    // Handle transmission reordering / deletion via shared helper to keep UI in sync.
    changed |= applyItemReorder(
        botPrefab.linearTransmissions,
        _linearTransmissionExpanded,
        transmissionToDelete,
        transmissionToMoveUp,
        transmissionToMoveDown);

    if (ImGui::Button("Add Transmission")) {
      superdex::robotics::BotLinearTransmissionPrefab newTransmission;
      std::array<char, 32> defaultName{};
      snprintf(
          defaultName.data(),
          defaultName.size(),
          "transmission_%zu",
          botPrefab.linearTransmissions.size());
      newTransmission.name = defaultName.data();
      botPrefab.linearTransmissions.push_back(std::move(newTransmission));
      _linearTransmissionExpanded.push_back(false);
      changed = true;
    }
    ImGui::Unindent();
  } // Linear Transmissions section

  // ========================================================================
  // Spatial Tendons Section
  // ========================================================================
  if (ImGui::CollapsingHeader("Spatial Tendons", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Indent();
    if (CopyPasteContextMenu("SpatialTendonsPopup", botPrefab.spatialTendons)) {
      changed = true;
      _spatialTendonExpanded.resize(botPrefab.spatialTendons.size());
      for (bool& expanded : _spatialTendonExpanded) {
        expanded = false;
      }
    } else {
      _spatialTendonExpanded.resize(botPrefab.spatialTendons.size());
    }

    int tendonToDelete = -1;
    int tendonToMoveUp = -1;
    int tendonToMoveDown = -1;
    for (size_t iTendon = 0; iTendon < botPrefab.spatialTendons.size(); ++iTendon) {
      ImGui::PushID(static_cast<int>(iTendon));
      auto& tendon = botPrefab.spatialTendons[iTendon];

      ItemHeaderResult headerResult =
          showItemHeader(iTendon, botPrefab.spatialTendons.size(), tendon.name, "Tendon", "Tendon");
      if (headerResult.moveUp) {
        tendonToMoveUp = static_cast<int>(iTendon);
      }
      if (headerResult.moveDown) {
        tendonToMoveDown = static_cast<int>(iTendon);
      }
      if (headerResult.deleted) {
        tendonToDelete = static_cast<int>(iTendon);
      }
      if (iTendon < _spatialTendonExpanded.size()) {
        _spatialTendonExpanded[iTendon] = headerResult.isOpen;
      }

      if (headerResult.isOpen) {
        ImGui::Indent();
        if (ImGui::InputText("Name", &tendon.name, ImGuiInputTextFlags_CharsNoBlank)) {
          changed = true;
        }
        ImGui::SeparatorText("Routing Elements");
        int rowToDelete = -1;
        auto tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
            ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp;
        if (ImGui::BeginTable("##TendonRoutingElements", 4, tableFlags)) {
          ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 1.0f);
          ImGui::TableSetupColumn("Link/Joint", ImGuiTableColumnFlags_WidthStretch, 1.5f);
          ImGui::TableSetupColumn(
              "Position / Coefficient", ImGuiTableColumnFlags_WidthStretch, 3.5f);
          ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 28.0f);
          ImGui::TableHeadersRow();

          for (size_t iRow = 0; iRow < tendon.routingElements.size(); ++iRow) {
            ImGui::PushID(static_cast<int>(iRow));
            auto& element = tendon.routingElements[iRow];

            ImGui::TableNextRow();

            // Type column
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            char const* typeName =
                (element.type == mochi::RoutingElementType::Waypoint) ? "Waypoint" : "Linear Joint";
            if (ImGui::BeginCombo("##Type", typeName)) {
              if (ImGui::Selectable(
                      "Waypoint", element.type == mochi::RoutingElementType::Waypoint)) {
                if (element.type != mochi::RoutingElementType::Waypoint) {
                  element.type = mochi::RoutingElementType::Waypoint;
                  // Reset index to valid link index when switching to Waypoint
                  element.index = 0;
                  changed = true;
                }
              }
              if (ImGui::Selectable(
                      "Linear Joint", element.type == mochi::RoutingElementType::LinearJoint)) {
                if (element.type != mochi::RoutingElementType::LinearJoint) {
                  element.type = mochi::RoutingElementType::LinearJoint;
                  // Reset index to first single-DoF joint when switching to LinearJoint.
                  // Leave as -1 (surfaced as "<invalid>" in the dropdown) if none exist,
                  // rather than silently picking a non-single-DoF joint.
                  element.index = -1;
                  for (int dofIdx : botPrefab._dofIndices) {
                    if (dofIdx >= 0 && dofIdx < static_cast<int>(botPrefab.joints.size())) {
                      auto const& joint = botPrefab.joints[dofIdx];
                      if (joint.type == mochi::ArticulatedJointType::Revolute ||
                          joint.type == mochi::ArticulatedJointType::Prismatic) {
                        element.index = dofIdx;
                        break;
                      }
                    }
                  }
                  changed = true;
                }
              }
              ImGui::EndCombo();
            }

            // Link/Joint column
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (element.type == mochi::RoutingElementType::Waypoint) {
              // Link index dropdown
              char const* currentName =
                  (element.index >= 0 && element.index < static_cast<int>(botPrefab.links.size()))
                  ? botPrefab.links[element.index].name.c_str()
                  : "<invalid>";
              if (ImGui::BeginCombo("##Link", currentName)) {
                for (size_t linkIdx = 0; linkIdx < botPrefab.links.size(); ++linkIdx) {
                  auto const& link = botPrefab.links[linkIdx];
                  bool const isSelected = (static_cast<int>(linkIdx) == element.index);
                  ImGui::PushID(static_cast<int>(linkIdx));
                  if (ImGui::Selectable(link.name.c_str(), isSelected)) {
                    element.index = static_cast<int>(linkIdx);
                    changed = true;
                  }
                  if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                  }
                  ImGui::PopID();
                }
                ImGui::EndCombo();
              }
            } else {
              // Joint index dropdown (filtered to single-DOF joints only)
              // LinearJoint elements require single-DoF joints (revolute/prismatic)
              char const* currentName =
                  (element.index >= 0 && element.index < static_cast<int>(botPrefab.joints.size()))
                  ? botPrefab.joints[element.index].name.c_str()
                  : "<invalid>";
              // Check if there are any single-DoF joints available
              bool hasSingleDofJoints = false;
              for (int dofIdx : botPrefab._dofIndices) {
                if (dofIdx >= 0 && dofIdx < static_cast<int>(botPrefab.joints.size())) {
                  auto const& joint = botPrefab.joints[dofIdx];
                  // Single-DoF joints are revolute or prismatic (not spherical, free, hard, or
                  // cycle)
                  if (joint.type == mochi::ArticulatedJointType::Revolute ||
                      joint.type == mochi::ArticulatedJointType::Prismatic) {
                    hasSingleDofJoints = true;
                    break;
                  }
                }
              }
              ImGui::BeginDisabled(!hasSingleDofJoints);
              if (ImGui::BeginCombo(
                      "##Joint", hasSingleDofJoints ? currentName : "<no single-DoF joints>")) {
                for (int dofIdx : botPrefab._dofIndices) {
                  if (dofIdx < 0 || dofIdx >= static_cast<int>(botPrefab.joints.size())) {
                    continue;
                  }
                  auto const& dofJoint = botPrefab.joints[dofIdx];
                  // Only show single-DoF joints (revolute or prismatic)
                  if (dofJoint.type != mochi::ArticulatedJointType::Revolute &&
                      dofJoint.type != mochi::ArticulatedJointType::Prismatic) {
                    continue;
                  }
                  bool const isSelected = (dofIdx == element.index);
                  ImGui::PushID(dofIdx);
                  if (ImGui::Selectable(dofJoint.name.c_str(), isSelected)) {
                    element.index = dofIdx;
                    changed = true;
                  }
                  if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                  }
                  ImGui::PopID();
                }
                ImGui::EndCombo();
              }
              ImGui::EndDisabled();
              if (!hasSingleDofJoints &&
                  ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip(
                    "Linear Joint elements require single-DoF joints (revolute or prismatic).\nThis bot has no single-DoF joints.");
              }
            }

            // Position / Coefficient column
            ImGui::TableNextColumn();
            if (element.type == mochi::RoutingElementType::Waypoint) {
              // For Waypoint, provide option to select child link or enter manual position
              // Check if current link has child links with Hard/Fixed joints
              bool hasChildLinks = false;
              if (element.index >= 0 && element.index < static_cast<int>(botPrefab.links.size())) {
                auto const& parentLink = botPrefab.links[element.index];
                for (int childIdx : parentLink._childrenIndices) {
                  if (childIdx >= 0 && childIdx < static_cast<int>(botPrefab.joints.size()) &&
                      childIdx < static_cast<int>(botPrefab.links.size())) {
                    auto const& joint = botPrefab.joints[childIdx];
                    if (joint.type == mochi::ArticulatedJointType::Hard) {
                      hasChildLinks = true;
                      break;
                    }
                  }
                }
              }

              if (hasChildLinks) {
                // Show child link selector
                ImGui::SetNextItemWidth(-FLT_MIN);
                // Build list of child links with Hard joints
                struct ChildLinkInfo {
                  int index;
                  std::string name;
                  mochi::Real3 position;
                };
                mochi::DynamicArray<ChildLinkInfo> childLinks;
                // Add "Manual" option
                childLinks.push_back(
                    ChildLinkInfo{-1, "Manual (enter coordinates)", {0_r, 0_r, 0_r}});

                auto const& parentLink = botPrefab.links[element.index];
                for (int childIdx : parentLink._childrenIndices) {
                  if (childIdx >= 0 && childIdx < static_cast<int>(botPrefab.joints.size()) &&
                      childIdx < static_cast<int>(botPrefab.links.size())) {
                    auto const& joint = botPrefab.joints[childIdx];
                    if (joint.type == mochi::ArticulatedJointType::Hard) {
                      auto const& childLink = botPrefab.links[childIdx];
                      mochi::Real3 pos = childLink._parentFromLink.GetTranslation();
                      childLinks.push_back(
                          ChildLinkInfo{childIdx, std::string(childLink.name), pos});
                    }
                  }
                }

                // Use ImGui state storage to remember manual mode selection per element to allow
                // explicit Manual toggle even when position matches a child link. Scope the ID to
                // the element's address rather than the surrounding (iTendon, iRow) ID stack so
                // the state follows the storage of this element instead of its slot position.
                ImGuiStorage* storage = ImGui::GetStateStorage();
                ImGuiID manualModeId = ImGui::GetID(static_cast<void const*>(&element));
                bool forceManual = storage->GetBool(manualModeId, false);

                // Find current selection based on position, unless forced manual
                int currentChildIdx = 0; // 0 means Manual by default
                if (!forceManual) {
                  for (size_t i = 0; i < childLinks.size(); ++i) {
                    if (childLinks[i].index >= 0) {
                      // Check if current localPosition matches this child link's position
                      auto const& pos = childLinks[i].position;
                      if (mochi::NearEqual(element.localPosition[0], pos[0], 1e-6_r) &&
                          mochi::NearEqual(element.localPosition[1], pos[1], 1e-6_r) &&
                          mochi::NearEqual(element.localPosition[2], pos[2], 1e-6_r)) {
                        currentChildIdx = static_cast<int>(i);
                        break;
                      }
                    }
                  }
                }
                // If no match found, it's Manual
                if (currentChildIdx == 0) {
                  forceManual = true;
                  storage->SetBool(manualModeId, true);
                }

                if (ImGui::BeginCombo("##Preset", childLinks[currentChildIdx].name.c_str())) {
                  for (size_t i = 0; i < childLinks.size(); ++i) {
                    bool isSelected = (static_cast<int>(i) == currentChildIdx);
                    if (ImGui::Selectable(childLinks[i].name.c_str(), isSelected)) {
                      if (childLinks[i].index >= 0) {
                        // Set position to child link origin
                        element.localPosition = childLinks[i].position;
                        storage->SetBool(manualModeId, false);
                        forceManual = false;
                        changed = true;
                      } else {
                        // Manual selected - keep current position but force manual mode
                        storage->SetBool(manualModeId, true);
                        forceManual = true;
                        changed = true;
                      }
                    }
                    if (isSelected) {
                      ImGui::SetItemDefaultFocus();
                    }
                  }
                  ImGui::EndCombo();
                }

                // Show manual coordinate editor - disabled when a preset child link is selected
                ImGui::BeginDisabled(!forceManual);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::DragRealXYZ(
                        "##LocalPos",
                        element.localPosition,
                        0.001f,
                        0.0f,
                        0.0f,
                        GetUnitFormat(UnitFormat::Length))) {
                  changed = true;
                  storage->SetBool(manualModeId, true);
                  forceManual = true;
                }
                ImGui::EndDisabled();
              } else {
                // No child links, just show manual coordinate editor
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::DragRealXYZ(
                        "##LocalPos",
                        element.localPosition,
                        0.001f,
                        0.0f,
                        0.0f,
                        GetUnitFormat(UnitFormat::Length))) {
                  changed = true;
                }
              }
            } else {
              // Coefficient for LinearJoint
              ImGui::SetNextItemWidth(-FLT_MIN);
              if (ImGui::DragReal(
                      "##Coefficient",
                      &element.coefficient,
                      0.001f,
                      0.0f,
                      0.0f,
                      GetUnitFormat(UnitFormat::Length, element.coefficient))) {
                changed = true;
              }
            }

            // Action column with delete button
            ImGui::TableNextColumn();
            if (ImGui::Button(ICON_FA_TRASH)) {
              rowToDelete = static_cast<int>(iRow);
            }

            ImGui::PopID();
          }
          ImGui::EndTable();
        }

        if (rowToDelete >= 0) {
          tendon.routingElements.erase(tendon.routingElements.begin() + rowToDelete);
          changed = true;
        }

        if (ImGui::Button("Add Routing Element")) {
          mochi::RoutingElement newElement;
          newElement.type = mochi::RoutingElementType::Waypoint;
          newElement.index = 0; // Default to first link
          newElement.localPosition = {0_r, 0_r, 0_r};
          newElement.coefficient = 0_r;
          tendon.routingElements.push_back(newElement);
          changed = true;
        }

        // Displacement-control actuator parameters shared via local lambda to keep UI synchronized
        // across transmission types.
        changed |= showDisplacementControlActuator(tendon);
        ImGui::Unindent();
      }

      ImGui::PopID();
    }

    // Handle tendon reordering / deletion via shared helper to keep UI in sync.
    changed |= applyItemReorder(
        botPrefab.spatialTendons,
        _spatialTendonExpanded,
        tendonToDelete,
        tendonToMoveUp,
        tendonToMoveDown);

    if (ImGui::Button("Add Tendon")) {
      superdex::robotics::BotSpatialTendonPrefab newTendon;
      std::array<char, 32> defaultName{};
      snprintf(
          defaultName.data(), defaultName.size(), "tendon_%zu", botPrefab.spatialTendons.size());
      newTendon.name = defaultName.data();
      // Add two default waypoint elements to make the tendon valid
      // (a single waypoint would be isolated and fail validation)
      for (int i = 0; i < 2; ++i) {
        mochi::RoutingElement defaultElement;
        defaultElement.type = mochi::RoutingElementType::Waypoint;
        defaultElement.index = 0; // Default to first link
        defaultElement.localPosition = {0_r, 0_r, 0_r};
        newTendon.routingElements.push_back(defaultElement);
      }
      botPrefab.spatialTendons.push_back(std::move(newTendon));
      _spatialTendonExpanded.push_back(false);
      changed = true;
    }
    ImGui::Unindent();
  } // Spatial Tendons section

  if (changed) {
    bool buildOk = false;
    _botAsset->Rebuild(_studio->GetBotLoader(), &buildOk);
    if (buildOk) {
      _botAsset->SetDirty(!_botAsset->IsReadOnly());
      GetUndoStack().MarkEdited();
      _botAsset->MarkThumbnailDirty();
      RestageBot();
    }
  }

  ImGui::End();
}

} // namespace superdex::studio
