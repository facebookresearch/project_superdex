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

#if MOCHI_INTERNAL

#include "editors/bot_scene_editor.h"
#include "app/app.h"
#include "assets/asset_manager.h"
#include "assets/bot_asset.h"
#include "assets/bot_scene_asset.h"
#include "assets/mochi_prefab_asset.h"
#include "ui/imgui_widgets.h"

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/path.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>

#include <imguios/fonts/icons_font_awesome5.h>

#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <string_view>

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// AssetEditor
//--------------------------------------------------------------------------------------------------

namespace {

// Returns a name of the form "<stem><N>" (N starting at 1) that is not in @p existing. Trailing
// digits of @p base are stripped to form the stem so numbering continues from it. Mirrors
// MochiPrefabEditor::MakeUniquePrefabActorName so new entries get sensible non-conflicting names.
std::string MakeUniqueName(std::string_view base, std::set<std::string> const& existing) {
  std::string stem(base);
  while (!stem.empty() && std::isdigit(static_cast<unsigned char>(stem.back()))) {
    stem.pop_back();
  }
  if (stem.empty()) {
    stem = std::string(base);
  }
  for (int i = 1;; ++i) {
    std::string candidate = stem + std::to_string(i);
    if (existing.count(candidate) == 0) {
      return candidate;
    }
  }
}

} // namespace

BotSceneEditor::BotSceneEditor(SuperDexStudio* studio, BotSceneAsset* asset)
    : AssetEditor(studio, asset), _sceneAsset(asset), _stage(studio, "BotSceneEditorStage") {}

void BotSceneEditor::Initialize() {
  // Initialize _undoStack (edits are only possible on non-archive scenes).
  if (!_sceneAsset->IsReadOnly()) {
    _undoStack.Initialize(
        [this] { return TakeUndoSnapshot(); },
        [this](std::string const& json, int selIdx) { RestoreUndoSnapshot(json, selIdx); });
  }
  // Initialize Viewport (read-only: no gizmo).
  _viewport = Viewport::Create(_studio, _studio->GetViewSettings());
  _viewport->showTransformGizmoTarget = []() { return false; };
  // Register viewport "Show" toggle commands (keyboard shortcuts + top-left dropdown).
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
             RestageBotScene();
           },
       .getState = [this] { return _stageType == StageType::RenderModelOnly; },
       .shortcut = ImGuiKey_1});
  _viewport->RegisterShowCommand(
      {.name = "Collision Only",
       .onToggle =
           [this] {
             _stageType = StageType::MochiModelOnly;
             RestageBotScene();
           },
       .getState = [this] { return _stageType == StageType::MochiModelOnly; },
       .shortcut = ImGuiKey_2});
  _viewport->RegisterShowCommand(
      {.name = "Render (Collision Fallback)",
       .onToggle =
           [this] {
             _stageType = StageType::RenderModelFallbackToMochiModel;
             RestageBotScene();
           },
       .getState = [this] { return _stageType == StageType::RenderModelFallbackToMochiModel; },
       .shortcut = ImGuiKey_3});
  // Initialize Mochi scene and callbacks.
  _mochiScene.Initialize(_studio->GetMochiContext(), "BotSceneEditor");
  _mochiScene.SetSettings(_studio->GetAppSettings().physics);
  _mochiScene.createPhysicsActors = [this](mochi::Scene* s) { CreatePhysicsActors(s); };
  _mochiScene.destroyPhysicsActors = [this](mochi::Scene* s) { DestroyPhysicsActors(s); };
  _mochiScene.registerPostStepCallback = [this](mochi::AsyncScene* a) {
    return RegisterPostStepCallback(a);
  };
  // Create the per-session force-drag controller (this hook fires once per CreateScene); it is torn
  // down in OnStopPhysics so its lifetime matches the session.
  _mochiScene.registerPreStepCallback = [this](mochi::AsyncScene*) {
    _dragController = std::make_unique<PhysicsDragController>(
        _mochiScene,
        &_stage,
        _studio->GetAppSettings().physicsDrag,
        _studio->GetRendererToEditorSpaceConverter());
    return mochi::CallbackHandle{};
  };
  _mochiScene.onStopPhysics = [this]() { OnStopPhysics(); };
  // Force-drag (left-drag) of simulated rigid bodies / articulated links. The controller only
  // exists while a session is running, so the hooks read the slot on each event.
  BindSceneObjectDragHooks(*_viewport, _dragController);
  // Bind the stage to the viewport's render scene and stage the bot scene.
  _stage.BindRenderScene(_viewport->GetRenderScene());
  RestageBotScene();
  _viewport->FocusCameraOnScene();
}

void BotSceneEditor::OnHandleInputs() {
  if (CanSimulate()) {
    _mochiScene.HandleHotkeys();
  }
}

void BotSceneEditor::OnRender(Renderer const* renderer) {
  // Sync data from physics simulation.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.UpdateStats();
    SyncFromPhysics();
  }
  // Draw mochi scene debug.
  _mochiScene.DrawDebug(
      _viewport->GetRenderScene()->GetDebugDraw(), _studio->GetEditorToRendererSpaceConverter());
  // Draw force-drag visualization (grab point, target, connecting line).
  if (_dragController) {
    _dragController->DrawDebug(
        _viewport->GetRenderScene()->GetDebugDraw(),
        _viewport->GetDebugText(),
        _studio->GetEditorToRendererSpaceConverter());
  }
  // Render the scene.
  _viewport->RenderScene(renderer);
}

void BotSceneEditor::Shutdown() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  _stage.Clear();
  _viewport.reset();
}

void BotSceneEditor::OnActivate() {
  // A referenced base scene, spawnable prefab, or bot may have been edited (and saved) in another
  // tab while we were inactive. RestageBotScene reloads referenced assets and re-stages, so their
  // changes appear automatically when returning to this editor.
  if (!_mochiScene.IsSimulating()) {
    RestageBotScene();
  }
}

void BotSceneEditor::Refresh() {
  // A referenced base scene, spawnable prefab, or bot was replaced/renamed elsewhere in the app.
  // Stop any running simulation (restaging mid-sim is unsafe), then restage to reflect it.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  RestageBotScene();
}

void BotSceneEditor::OnDeactivate() {
  // Stop simulation when the user changes to another editor.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
}

void BotSceneEditor::ShowTabContents() {
  ImGui::BeginChild("Viewport_Child", ImVec2(0, 0), 0, ImGuiWindowFlags_NoMove);
  // Force-drag only applies while the sim runs.
  _viewport->enableSceneObjectDrag = _mochiScene.IsSimulating();
  _viewport->ShowViewportContents(true);
  _viewport->ShowStatsOverlay(_mochiScene.GetStepsPerSecond());
  ImGui::BeginDisabled(!CanSimulate());
  _mochiScene.ShowPlayToolbarOverViewport();
  ImGui::EndDisabled();
  ImGui::EndChild(); // Viewport_Child
}

std::vector<AssetEditor::WindowDeclaration> BotSceneEditor::GetDefaultWindows() {
  using Dock = AssetEditor::DockRegion;
  return {
      {"Bot Scene Info", true, Dock::SidePanelTop},
      {"Physics Settings", false, Dock::SidePanelBottom},
      {"Scene Stage Debug", false, Dock::SidePanelTop, true}};
}

AssetSceneOverrides BotSceneEditor::GetAssetSceneOverrides() const {
  // A base scene is the authoritative source for every simulation physics parameter, so it supplies
  // both gravity and the solver.
  bool const hasBaseScene = !_sceneAsset->GetPrefab().scene.baseScene.empty();
  return {hasBaseScene, hasBaseScene};
}

std::vector<AssetEditor::WindowDeclaration> BotSceneEditor::GetAuxiliaryWindows() const {
  return GetDefaultWindows();
}

void BotSceneEditor::ShowAuxiliaryWindows() {
  if (bool& open = _studio->GetWindowVisible("Bot Scene Info")) {
    ShowInfoWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Physics Settings")) {
    _mochiScene.ShowPhysicsSettingsWindow("Physics Settings", &open, GetAssetSceneOverrides());
  }
  if (bool& open = _studio->GetWindowVisible("Scene Stage Debug")) {
    auto* simNames = _mochiScene.IsSimulating() ? &_simData.GetConsumerData().actorNames : nullptr;
    _stage.ShowSceneStageWindow("Scene Stage Debug", &open, simNames);
  }
}

bool BotSceneEditor::CanUndoRedo() const {
  return !_mochiScene.IsSimulating();
}

void BotSceneEditor::ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) {
  if (_viewport && _viewport->GetRenderScene()) {
    _viewport->GetRenderScene()->ApplyViewSettings(viewSettings);
  }
}

void BotSceneEditor::OnAppSettingsChanged(AppSettings const& settings) {
  // The drag controller snapshots its tuning per session, so a live one has to be told.
  if (_dragController) {
    _dragController->SetSettings(settings.physicsDrag);
  }
}
//--------------------------------------------------------------------------------------------------
// Undo/Redo
//--------------------------------------------------------------------------------------------------

std::string BotSceneEditor::TakeUndoSnapshot() const {
  return SReflect::ToJsonString(_sceneAsset->GetPrefab(), false);
}

void BotSceneEditor::RestoreUndoSnapshot(std::string const& json, int /*selectionIndex*/) {
  // Reset to defaults before deserializing — NoSerializeDefaults omits default-valued fields from
  // the JSON, so without a reset those fields would retain the current (edited) values instead of
  // reverting to defaults.
  _sceneAsset->GetPrefab() = superdex::robotics::BotScenePrefab{};
  SReflect::FromJsonString(_sceneAsset->GetPrefab(), json, SReflect::DeserializeFlags::Default);

  // Reload referenced assets and refresh reference tracking, then re-stage the viewport.
  _studio->GetAssetManager().ResyncReferencer(_sceneAsset);
  RestageBotScene();

  _sceneAsset->SetDirty(!GetUndoStack().IsAtSavedState());
}

//--------------------------------------------------------------------------------------------------
// Staging
//--------------------------------------------------------------------------------------------------

void BotSceneEditor::RestageBotScene() {
  _stage.StageBotScene(_sceneAsset->GetPrefab(), _stageType);
  // Position the drop-shadow ground plane at the scene's lowest point (rest pose staged above; not
  // called mid-sim). The same height positions the studio physics ground plane, when the settings
  // ask for one.
  _mochiScene.SetGroundPlaneHeight(_viewport->UpdateGroundPlane());
}

//--------------------------------------------------------------------------------------------------
// Mochi Scene
//--------------------------------------------------------------------------------------------------

bool BotSceneEditor::CanSimulate() const {
  return !_stage.IsEmpty();
}

void BotSceneEditor::CreatePhysicsActors(mochi::Scene* scene) {
  auto* mochiContext = _studio->GetMochiContext();
  auto* botsContext = _studio->GetRoboticsContext();

  mochi::ErrorLog e;
  _botScene = superdex::robotics::LoadBotScene(
      scene,
      _sceneAsset->GetPrefab(),
      _sceneAsset->GetBotsRootPath(),
      mochiContext,
      botsContext,
      e);
  if (!e.IsOK()) {
    MOCHI_LOG_ERROR("Failed to load bot scene physics");
    _botScene.reset();
    return;
  }

  auto const& prefab = _sceneAsset->GetPrefab();

  // Load all spawnable prefabs by default so they are simulated alongside the base scene and bots.
  for (auto const& prefabEntry : prefab.scene.spawnablePrefabs) {
    mochi::ErrorLog spawnError;
    _botScene->LoadSpawnablePrefab(std::string(prefabEntry.name), spawnError);
  }

  // Physics actor order must match SceneStage::StageBotScene: base-scene actors first (in
  // LoadBotScene / prefab::AddToScene order), then spawnable prefabs in declaration order, then
  // each bot's articulated actor in declaration order.
  _physicsActors.clear();
  for (auto const& handle : _botScene->GetBaseSceneActorHandles()) {
    _physicsActors.push_back(handle);
  }
  for (auto const& prefabEntry : prefab.scene.spawnablePrefabs) {
    for (auto const& handle : _botScene->GetSpawnedPrefabActors(std::string(prefabEntry.name))) {
      _physicsActors.push_back(handle);
    }
  }
  for (auto const& botEntry : prefab.bots) {
    auto* bot = _botScene->GetBot(std::string(botEntry.name));
    if (bot && bot->GetArticulatedActor()) {
      _physicsActors.push_back(bot->GetArticulatedActor()->GetHandle());
    }
  }
}

void BotSceneEditor::DestroyPhysicsActors(mochi::Scene*) {
  // Destroy bots/controllers before the async scene destroys the scene. The BotScene is non-owning,
  // so the scene itself is left intact for MochiAsyncScene to destroy.
  _botScene.reset();
  _physicsActors.clear();
}

mochi::CallbackHandle BotSceneEditor::RegisterPostStepCallback(mochi::AsyncScene* scene) {
  return scene->RegisterPostStepCallback(
      "BotSceneEditor::ExtractActors", [this](mochi::StepInfo const& info) {
        mochi::ErrorLog e;
        auto& data = _simData.GetProducerData();
        data.actorTransforms.clear();
        data.actorNames.clear();
        // Build the ordered transform list
        // actors expand to their nested link transforms, rigid actors push their root transform,
        // and everything else (e.g. soft) is skipped.
        for (auto const& handle : _physicsActors) {
          auto* actor = info.scene->GetActor(handle);
          if (!actor) {
            continue;
          }
          if (actor->GetType() == mochi::ActorType::Articulated) {
            auto const links = actor->GetNestedLinkActors(e);
            std::vector<mochi::TransformRT> linkTransforms(links.size());
            actor->GetArticulatedLinkTransforms(mochi::MakeSpan(linkTransforms), e);
            for (int i = 0; i < mochi::isize(linkTransforms); ++i) {
              data.actorTransforms.push_back(linkTransforms[i]);
              char const* const linkName = info.scene->GetActor(links[i])->GetName();
              data.actorNames.emplace_back(linkName ? linkName : "");
            }
          } else if (actor->GetType() == mochi::ActorType::Rigid) {
            data.actorTransforms.push_back(actor->GetRootTransform());
            char const* const rigidName = actor->GetName();
            data.actorNames.emplace_back(rigidName ? rigidName : "");
          }
        }
        _simData.Produce();
      });
}

void BotSceneEditor::OnStopPhysics() {
  // Tear down the per-session force-drag controller (its callback is already gone with the scene).
  _dragController.reset();
  _stage.ResetWorldTransforms(_studio->GetEditorToRendererSpaceConverter());
  _simData.Consume();
  _physicsActors.clear();
}

void BotSceneEditor::SyncFromPhysics() {
  if (_simData.Consume()) {
    _stage.ApplyWorldTransforms(
        mochi::MakeConstSpan(_simData.GetConsumerData().actorTransforms),
        _studio->GetEditorToRendererSpaceConverter());
  }
}

//--------------------------------------------------------------------------------------------------
// ImGui
//--------------------------------------------------------------------------------------------------

void BotSceneEditor::ShowInfoWindow(bool* open) {
  auto const windowFlags =
      _sceneAsset->IsDirty() ? ImGuiWindowFlags_UnsavedDocument : ImGuiWindowFlags_None;
  ImGui::Begin("Bot Scene Info", open, windowFlags);

  auto& prefab = _sceneAsset->GetPrefab();
  auto& assetManager = _studio->GetAssetManager();

  // Big title (fixed, outside the scroll child): the asset/file name.
  ImGui::PushFont(_studio->GetFont("Roboto Bold Large"));
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(_sceneAsset->GetName().c_str());
  ImGui::PopFont();
  ImGui::Separator();

  // Continuous edits (debounced into a single undo entry) vs. discrete structural edits
  // (add/remove/reorder — pushed immediately). Applied together in the epilogue below.
  bool changed = false;
  bool structural = false;

  ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32_BLACK_TRANS);
  ImGui::BeginChild("BotSceneInfoChild", ImVec2(0, 0));

  // ---- Metadata ----
  if (ImGui::CollapsingHeader("Metadata", ImGuiTreeNodeFlags_DefaultOpen)) {
    changed |= ImGui::SimpleReflectionStruct(prefab.metadata);
  }

  // ---- Scene ----
  if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
    // Base scene.
    if (ImGui::AssetSlot(
            "Base Scene",
            prefab.scene.baseScene,
            assetManager,
            _studio,
            AssetType::MochiPrefab,
            true)) {
      if (!prefab.scene.baseScene.empty()) {
        assetManager.LoadMochiPrefabAsset(prefab.scene.baseScene);
      }
      assetManager.ResyncReferencer(_sceneAsset);
      changed = true;
    }

    // Spawnable prefabs (add / remove / reorder).
    ImGui::HoverableSeparatorText("Spawnable Prefabs");
    auto& spawnables = prefab.scene.spawnablePrefabs;
    int prefabToDelete = -1;
    int prefabToMoveUp = -1;
    int prefabToMoveDown = -1;
    // Section-level ID scope so per-item widget IDs don't collide with the Bots section (both
    // loops push the same integer index).
    ImGui::PushID("Spawnables");
    for (int i = 0; i < static_cast<int>(spawnables.size()); ++i) {
      auto& entry = spawnables[i];
      ImGui::PushID(i);
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
      if (ImGui::Button(ICON_FA_TRASH)) {
        prefabToDelete = i;
      }
      ImGui::SameLine();
      ImGui::BeginDisabled(i == 0);
      if (ImGui::Button(ICON_FA_CARET_UP)) {
        prefabToMoveUp = i;
      }
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(i == static_cast<int>(spawnables.size()) - 1);
      if (ImGui::Button(ICON_FA_CARET_DOWN)) {
        prefabToMoveDown = i;
      }
      ImGui::EndDisabled();
      ImGui::PopStyleVar();
      ImGui::SameLine();
      // "###" keeps the header ID stable (per PushID) as the visible name is edited.
      std::string const label =
          (entry.name.empty() ? std::string("(unnamed)") : std::string(entry.name)) + "###entry";
      if (ImGui::CollapsingHeader(label.c_str())) {
        changed |= ImGui::InputText("Name", &entry.name);
        if (ImGui::AssetSlot(
                "Prefab", entry.path, assetManager, _studio, AssetType::MochiPrefab, true)) {
          if (!entry.path.empty()) {
            assetManager.LoadMochiPrefabAsset(entry.path);
          }
          assetManager.ResyncReferencer(_sceneAsset);
          changed = true;
        }
      }
      ImGui::PopID();
    }
    ImGui::PopID(); // Spawnables
    if (prefabToMoveUp > 0) {
      std::swap(spawnables[prefabToMoveUp], spawnables[prefabToMoveUp - 1]);
      structural = true;
    }
    if (prefabToMoveDown >= 0 && prefabToMoveDown < static_cast<int>(spawnables.size()) - 1) {
      std::swap(spawnables[prefabToMoveDown], spawnables[prefabToMoveDown + 1]);
      structural = true;
    }
    if (prefabToDelete >= 0) {
      spawnables.erase(spawnables.begin() + prefabToDelete);
      assetManager.ResyncReferencer(_sceneAsset);
      structural = true;
    }
    if (ImGui::Button(ICON_FA_PLUS "###AddSpawnable")) {
      std::set<std::string> existingNames;
      for (auto const& e : spawnables) {
        existingNames.insert(std::string(e.name));
      }
      superdex::robotics::PrefabEntry newEntry;
      newEntry.name = mochi::DynamicString{MakeUniqueName("Prefab", existingNames)};
      spawnables.push_back(std::move(newEntry));
      structural = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Add Spawnable Prefab");
  }

  // ---- Bots ----
  if (ImGui::CollapsingHeader("Bots", ImGuiTreeNodeFlags_DefaultOpen)) {
    auto& bots = prefab.bots;
    int botToDelete = -1;
    int botToMoveUp = -1;
    int botToMoveDown = -1;
    // Section-level ID scope so per-item widget IDs don't collide with the Spawnable Prefabs
    // section (both loops push the same integer index).
    ImGui::PushID("Bots");
    for (int i = 0; i < static_cast<int>(bots.size()); ++i) {
      auto& bot = bots[i];
      ImGui::PushID(i);
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
      if (ImGui::Button(ICON_FA_TRASH)) {
        botToDelete = i;
      }
      ImGui::SameLine();
      ImGui::BeginDisabled(i == 0);
      if (ImGui::Button(ICON_FA_CARET_UP)) {
        botToMoveUp = i;
      }
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(i == static_cast<int>(bots.size()) - 1);
      if (ImGui::Button(ICON_FA_CARET_DOWN)) {
        botToMoveDown = i;
      }
      ImGui::EndDisabled();
      ImGui::PopStyleVar();
      ImGui::SameLine();
      std::string const label =
          (bot.name.empty() ? std::string("(unnamed)") : std::string(bot.name)) + "###bot";
      if (ImGui::CollapsingHeader(label.c_str())) {
        changed |= ImGui::InputText("Name", &bot.name);
        if (ImGui::AssetSlot("Bot", bot.path, assetManager, _studio, AssetType::Bot, true)) {
          // Reset the initial pose to the newly-assigned bot's default pose (the previous pose
          // belonged to a different bot with a different DOF layout).
          bot.initialPose.clear();
          if (!bot.path.empty()) {
            if (auto* newBotAsset = assetManager.LoadBotAsset(bot.path)) {
              bot.initialPose = newBotAsset->GetBotPrefab().defaultPose;
            }
          }
          assetManager.ResyncReferencer(_sceneAsset);
          changed = true;
        }
        changed |= ImGui::DragTransformRT("Spawn Transform", bot.parentFromBot);

        // Initial pose: limit-aware sliders identical to the bot editor's "Default Pose".
        auto* botAsset = assetManager.FindAssetByPath<BotAsset>(mochi::Path{bot.path.c_str()});
        if (botAsset != nullptr && !botAsset->GetBotPrefab()._dofIndices.empty()) {
          ImGui::HoverableSeparatorText("Initial Pose");
          changed |= ImGui::JointPoseEditor(botAsset->GetBotPrefab(), bot.initialPose);
        } else if (!bot.path.empty()) {
          ImGui::TextDisabled("Referenced bot not loaded");
        }

        // Controllers: read-only summary ("name (type)").
        if (!bot.controllers.empty()) {
          ImGui::HoverableSeparatorText("Controllers");
          ImGui::BeginDisabled(true);
          for (auto const& ctrl : bot.controllers) {
            ImGui::BulletText("%s (%s)", ctrl.name.c_str(), ctrl.type.c_str());
          }
          ImGui::EndDisabled();
        }
      }
      ImGui::PopID();
    }
    ImGui::PopID(); // Bots
    if (botToMoveUp > 0) {
      std::swap(bots[botToMoveUp], bots[botToMoveUp - 1]);
      structural = true;
    }
    if (botToMoveDown >= 0 && botToMoveDown < static_cast<int>(bots.size()) - 1) {
      std::swap(bots[botToMoveDown], bots[botToMoveDown + 1]);
      structural = true;
    }
    if (botToDelete >= 0) {
      bots.erase(bots.begin() + botToDelete);
      assetManager.ResyncReferencer(_sceneAsset);
      structural = true;
    }
    if (ImGui::Button(ICON_FA_PLUS "###AddBot")) {
      std::set<std::string> existingNames;
      for (auto const& b : bots) {
        existingNames.insert(std::string(b.name));
      }
      superdex::robotics::BotEntry newBot;
      newBot.name = mochi::DynamicString{MakeUniqueName("Bot", existingNames)};
      bots.push_back(std::move(newBot));
      structural = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Add Bot");
  }

  ImGui::EndChild();
  ImGui::PopStyleColor(); // ImGuiCol_ChildBg

  // Edit epilogue: reflect edits in the viewport, mark dirty, and record undo state. A bot scene
  // needs no derived "build" step — references resolve live during staging.
  if (changed || structural) {
    RestageBotScene();
    _sceneAsset->SetDirty(true);
    _sceneAsset->MarkThumbnailDirty();
    if (structural) {
      GetUndoStack().PushNow(); // discrete op → immediate undo entry
    } else {
      GetUndoStack().MarkEdited(); // continuous → debounced by the app loop
    }
  }

  ImGui::End();
}

} // namespace superdex::studio

#endif // MOCHI_INTERNAL
