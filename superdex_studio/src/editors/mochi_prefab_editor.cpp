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

#include "editors/mochi_prefab_editor.h"
#include "app/app.h"
#include "assets/asset.h"
#include "assets/mochi_prefab_asset.h"
#include "ui/imgui_widgets.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/reflection.h>

#include <imguios/fonts/icons_font_awesome5.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

using mochi::operator""_r;
using namespace mochi_renderer;

namespace superdex::studio {

// Screen-space distance (pixels) the pointer must travel during an alt+drag before the selection is
// duplicated. Prevents an accidental alt+click (with little to no drag) from creating a stray copy.
constexpr float kAltDuplicateDragThresholdPx = 4.0f;

// Fallback-name prefixes for unnamed top-level actors / links.
// rows, display names, and collision checks all resolve identical names.
constexpr std::string_view kRigidPrefix = "Rigid_";
constexpr std::string_view kSoftPrefix = "Soft_";
constexpr std::string_view kArticulatedPrefix = "Articulated_";
constexpr std::string_view kLinkPrefix = "Link_";
constexpr std::string_view kNestedPrefix = "Nested_";

// An actor's display name: its own name, or "<prefix><index>" when unnamed.
static std::string
FallbackName(std::string_view prefix, mochi::DynamicString const& name, int index) {
  return name.empty() ? (std::string(prefix) + std::to_string(index)) : std::string(name);
}

//--------------------------------------------------------------------------------------------------
// AssetEditor
//--------------------------------------------------------------------------------------------------

MochiPrefabEditor::MochiPrefabEditor(SuperDexStudio* studio, MochiPrefabAsset* asset)
    : AssetEditor(studio, asset), _prefabAsset(asset), _stage(studio, "MochiPrefabEditorStage") {}

void MochiPrefabEditor::Initialize() {
  // Initialize _undoStack
  if (!_prefabAsset->IsReadOnly()) {
    _undoStack.Initialize(
        [this] { return TakeUndoSnapshot(); },
        [this](std::string const& json, int selIdx) { RestoreUndoSnapshot(json, selIdx); });
    GetUndoStack().SetSelectionFn([this] { return GetUndoSelectionIndex(); });
  }
  // Initialize Viewport
  _viewport = Viewport::Create(_studio, _studio->GetViewSettings());
  _viewport->onSceneSelectionChanged = [this](std::vector<SceneObject*> const& objects) {
    OnSceneSelectionChanged(objects);
  };
  // The prefab editor is the only viewport consumer that opts into multi-selection (Ctrl/Shift-
  // click).
  _viewport->allowMultiSelect = true;
  _viewport->showTransformGizmoTarget = [this]() { return !_mochiScene.IsSimulating(); };
  _viewport->getTransformGizmoTarget = [this]() { return GetTransformGizmoTarget(); };
  _viewport->onTransformGizmoDelta = [this](mochi::TransformRT const& d, mochi::Real3 const& s) {
    OnGizmoTransformDelta(d, s);
  };
  // Alt+drag duplicates the selection and manipulates the copies (once per drag). Alt (not Ctrl) so
  // Ctrl stays free as the transform-gizmo snap modifier. Gate the gizmo start behind a small drag
  // threshold when Alt is held so a stray alt+click doesn't create a copy; a plain drag starts
  // immediately. DuplicateSelection() retargets the gizmo onto the new copies, so the drag
  // continues on them.
  _viewport->transformGizmoStartThresholdPx = []() {
    return ImGui::GetIO().KeyAlt ? kAltDuplicateDragThresholdPx : 0.0f;
  };
  _viewport->onTransformGizmoStarted = [this]() {
    if (ImGui::GetIO().KeyAlt) {
      DuplicateSelection(/*commitUndo=*/false);
    }
  };
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
             RestagePrefab();
           },
       .getState = [this] { return _stageType == StageType::RenderModelOnly; },
       .shortcut = ImGuiKey_1});
  _viewport->RegisterShowCommand(
      {.name = "Collision Only",
       .onToggle =
           [this] {
             _stageType = StageType::MochiModelOnly;
             RestagePrefab();
           },
       .getState = [this] { return _stageType == StageType::MochiModelOnly; },
       .shortcut = ImGuiKey_2});
  _viewport->RegisterShowCommand(
      {.name = "Render (Collision Fallback)",
       .onToggle =
           [this] {
             _stageType = StageType::RenderModelFallbackToMochiModel;
             RestagePrefab();
           },
       .getState = [this] { return _stageType == StageType::RenderModelFallbackToMochiModel; },
       .shortcut = ImGuiKey_3});
  // Initialize Mochi scene and callbacks
  _mochiScene.Initialize(_studio->GetMochiContext(), "MochiPrefabEditor");
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
  // Bind the stage to the viewport's render scene, and give the viewport the stage so selection/
  // hover highlights route to it (the stage owns the per-link highlight clones).
  _stage.BindRenderScene(_viewport->GetRenderScene());
  _viewport->SetSceneStage(&_stage);
  RestagePrefab();
  _viewport->FocusCameraOnScene();
}

void MochiPrefabEditor::OnHandleInputs() {
  // Sim-control shortcuts must run even while simulating (e.g. Esc to stop), so handle them before
  // the early-out below that ignores editing shortcuts during simulation.
  if (CanSimulate()) {
    _mochiScene.HandleHotkeys();
  }
  if (_mochiScene.IsSimulating()) {
    return;
  }
  // Don't steal shortcuts while the user is typing in a text field (let ImGui handle them).
  if (ImGui::GetIO().WantTextInput) {
    return;
  }
  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_C)) {
    CopySelectionToClipboard();
    return;
  }
  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_D)) {
    DuplicateSelection();
    return;
  }
  if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_V)) {
    PasteFromClipboard();
    return;
  }
  if (!ImGui::IsKeyChordPressed(ImGuiKey_Delete) && !ImGui::IsKeyChordPressed(ImGuiKey_Backspace)) {
    return;
  }
  DeleteSelection();
}

void MochiPrefabEditor::OnRender(Renderer const* renderer) {
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
  // Render the scene.
  _viewport->RenderScene(renderer);
}

void MochiPrefabEditor::Shutdown() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  _stage.Clear();
  _viewport.reset();
}

void MochiPrefabEditor::OnActivate() {
  // A nested prefab may have been edited (and saved) in another tab while we were inactive.
  // RestagePrefab reloads nested prefabs from disk (LoadNestedPrefabs) and re-stages, so their
  // changes appear automatically when returning to this editor.
  if (!_mochiScene.IsSimulating()) {
    RestagePrefab();
  }
}

void MochiPrefabEditor::Refresh() {
  // A referenced shape/model was replaced/renamed elsewhere in the app. Stop any running simulation
  // (restaging mid-sim is unsafe), then restage to reflect it.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  RestagePrefab();
}

void MochiPrefabEditor::OnDeactivate() {
  // Stop simulation when the user changes to another editor.
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
}

void MochiPrefabEditor::ShowTabContents() {
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

std::vector<AssetEditor::WindowDeclaration> MochiPrefabEditor::GetDefaultWindows() {
  using Dock = AssetEditor::DockRegion;
  return {
      // main windows
      {"Prefab Hierarchy", true, Dock::SidePanelTop, false},
      {"Prefab Details", true, Dock::SidePanelBottom, false},
      {"Actor Details", true, Dock::SidePanelBottom, false},
      {"Contact Filter", false, Dock::SidePanelBottom, false},
      {"Physics Settings", false, Dock::SidePanelBottom, false},
      // debug windows
      {"Render Scene Hierarchy", false, Dock::SidePanelTop, true},
      {"Render Scene Details", false, Dock::SidePanelBottom, true},
      {"Render Scene Stage", false, Dock::SidePanelTop, true}};
}

AssetSceneOverrides MochiPrefabEditor::GetAssetSceneOverrides() const {
  auto const& scene = _prefabAsset->GetPrefab().scene;
  if (!scene.has_value()) {
    return {};
  }
  return {scene->gravity.has_value(), scene->solver.has_value()};
}

std::vector<AssetEditor::WindowDeclaration> MochiPrefabEditor::GetAuxiliaryWindows() const {
  return GetDefaultWindows();
}

void MochiPrefabEditor::ShowAuxiliaryWindows() {
  // main windows
  if (bool& open = _studio->GetWindowVisible("Prefab Hierarchy")) {
    ShowPrefabHierarchyWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Prefab Details")) {
    ShowPrefabDetailsWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Actor Details")) {
    ShowActorDetailsWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Contact Filter")) {
    ShowContactFilterWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Physics Settings")) {
    _mochiScene.ShowPhysicsSettingsWindow("Physics Settings", &open, GetAssetSceneOverrides());
  }
  // debug windows
  if (bool& open = _studio->GetWindowVisible("Render Scene Hierarchy")) {
    _viewport->ShowSceneHierarchyWindow("Render Scene Hierarchy", &open);
  }
  if (bool& open = _studio->GetWindowVisible("Render Scene Details")) {
    _viewport->ShowSelectedObjectDetailsWindow("Render Scene Details", &open);
  }
  if (bool& open = _studio->GetWindowVisible("Render Scene Stage")) {
    auto* simNames = _mochiScene.IsSimulating() ? &_simData.GetConsumerData().actorNames : nullptr;
    _stage.ShowSceneStageWindow("Render Scene Stage", &open, simNames);
  }
}

void MochiPrefabEditor::ShowMainMenuItems() {}

bool MochiPrefabEditor::CanUndoRedo() const {
  return !_mochiScene.IsSimulating();
}

void MochiPrefabEditor::ApplySceneViewSettings(
    mochi_renderer::SceneViewSettings const& viewSettings) {
  if (_viewport && _viewport->GetRenderScene()) {
    _viewport->GetRenderScene()->ApplyViewSettings(viewSettings);
  }
}

void MochiPrefabEditor::OnAppSettingsChanged(AppSettings const& settings) {
  // The drag controller snapshots its tuning per session, so a live one has to be told.
  if (_dragController) {
    _dragController->SetSettings(settings.physicsDrag);
  }
}
//--------------------------------------------------------------------------------------------------
// Undo/Redo
//--------------------------------------------------------------------------------------------------

// Undo payload. The stashed scene settings ride along with the prefab: the "Scene Prefab" toggle
// moves data between the two, so restoring only the prefab would drop what the toggle put aside.
struct PrefabUndoSnapshot {
  mochi::prefab::ScenePrefab prefab;
  std::optional<mochi::prefab::SceneParams> stashedScene;

  MOCHI_STRUCT_BEGIN(superdex::studio::PrefabUndoSnapshot)
  MOCHI_FIELD(prefab)
  MOCHI_FIELD(stashedScene)
  MOCHI_STRUCT_END()
};

std::string MochiPrefabEditor::TakeUndoSnapshot() const {
  PrefabUndoSnapshot const snapshot{
      _prefabAsset->GetPrefab(), _prefabAsset->GetStashedSceneParams()};
  return SReflect::ToJsonString(snapshot, false);
}

void MochiPrefabEditor::RestoreUndoSnapshot(std::string const& json, int selectionIndex) {
  // Clear highlights before the scene objects are rebuilt below.
  _viewport->SetSelectedSceneObjects({});

  // Reset to defaults before deserializing — NoSerializeDefaults omits default-valued
  // fields from the JSON, so without a reset those fields would retain the current
  // (edited) values instead of reverting to defaults.
  PrefabUndoSnapshot snapshot;
  SReflect::FromJsonString(snapshot, json, SReflect::DeserializeFlags::Default);
  _prefabAsset->GetPrefab() = std::move(snapshot.prefab);
  _prefabAsset->GetStashedSceneParams() = std::move(snapshot.stashedScene);

  // sourceFilePath is not serialized (runtime-only), so restore it for prefab-relative path
  // resolution.
  _prefabAsset->GetPrefab().sourceFilePath =
      mochi::DynamicString{_prefabAsset->GetPath().ToString()};

  // Restore selection before restaging so RestagePrefab rebinds the viewport objects. The undo
  // selection channel is a single int, so only a primary rigid selection is round-tripped; other
  // selections reset after undo/redo (minor UX tradeoff).
  _selections.clear();
  _selectionAnchorRow = -1;
  if (selectionIndex >= 0) {
    _selections.push_back({Selection::Kind::Rigid, selectionIndex, -1});
  }

  // Rebuild staged render objects (and rebind the restored selection).
  RestagePrefab();

  // Sync dirty state.
  _prefabAsset->SetDirty(!GetUndoStack().IsAtSavedState());
}

int MochiPrefabEditor::GetUndoSelectionIndex() const {
  Selection const* const primary = PrimarySelection();
  return (primary && primary->kind == Selection::Kind::Rigid) ? primary->actorIndex : -1;
}

//------------------------------------------------------------------------------------------------
// Prefab Staging
//------------------------------------------------------------------------------------------------

// Forward declaration; defined below near the gizmo handlers. Clears an articulated actor's cached
// link/skin shapes so they re-bake at the current scale on the next physics load.
static void InvalidateArticulatedActorShapes(mochi::prefab::ArticulatedActorPrefab& actor);

// Clear cached shape handles on all actors of a prefab so they re-bake from their shape files on
// the next physics load. Nested prefab scale is baked into these shapes, and EnsureFullyLoaded
// skips already-loaded shapes, so a copied-from-memory nested prefab (whose shapes may have been
// baked in its own editor) must be cleared to pick up the parent reference's scale.
static void ClearPrefabShapes(mochi::prefab::ScenePrefab& prefab) {
  for (auto& actor : prefab.actors.rigid) {
    actor.shape = {};
  }
  for (auto& actor : prefab.actors.soft) {
    actor.shape = {};
  }
  for (auto& actor : prefab.actors.articulated) {
    InvalidateArticulatedActorShapes(actor);
  }
}

// Populate each nested prefab reference's loaded data, preferring the live in-memory prefab of a
// loaded MochiPrefabAsset (so unsaved edits made in a nested prefab's own editor propagate here
// without saving) and falling back to a fresh disk load otherwise. Recurses so deeper nested edits
// propagate too.
static void ResolveNestedPrefabsFromMemory(
    mochi::prefab::ScenePrefab& prefab,
    AssetManager& manager,
    std::string const& rootPath) {
  std::string_view const prefabFilePath = prefab.sourceFilePath.has_value()
      ? std::string_view(*prefab.sourceFilePath)
      : std::string_view();
  for (auto& nested : prefab.prefabs) {
    if (nested.path.empty()) {
      continue;
    }
    if (auto* asset =
            manager.FindAssetByPath<MochiPrefabAsset>(mochi::Path{std::string(nested.path)})) {
      nested.prefab = std::make_shared<mochi::prefab::ScenePrefab>(asset->GetPrefab());
    } else {
      mochi::ErrorLog error;
      auto const fullPath = mochi::prefab::GetPrefabFullPath(nested.path, rootPath, prefabFilePath);
      nested.prefab = std::make_shared<mochi::prefab::ScenePrefab>(
          mochi::prefab::ShallowLoadFromFile(fullPath, error));
    }
    // Clear cached shapes so the nested reference's scale re-bakes correctly on the next sim.
    ClearPrefabShapes(*nested.prefab);
    ResolveNestedPrefabsFromMemory(*nested.prefab, manager, rootPath);
  }
}

void MochiPrefabEditor::RestagePrefab() {
  std::vector<Selection> const saved = _selections;
  auto& prefab = _prefabAsset->GetPrefab();
  // Refresh nested prefab data from memory (or disk) so nested edits — even unsaved ones from a
  // nested prefab's editor — are reflected here.
  if (!prefab.prefabs.empty()) {
    ResolveNestedPrefabsFromMemory(
        prefab, _studio->GetAssetManager(), _prefabAsset->GetAssetsRoot());
  }
  bool const rebuilt = _stage.StagePrefab(prefab, _prefabAsset->GetAssetsRoot(), _stageType);
  if (rebuilt) {
    // StagePrefab destroyed the previous scene objects, which cleared the viewport's selection (and
    // via onSceneSelectionChanged, our _selections). The prefab arrays are unchanged, so restore
    // the saved selection and rebind it to the freshly staged objects.
    _selections = saved;
    SyncViewportSelection();
  }
  // Position the drop-shadow ground plane at the prefab's lowest point (rest pose staged above; not
  // called mid-sim). The same height positions the studio physics ground plane, when the settings
  // ask for one.
  _mochiScene.SetGroundPlaneHeight(_viewport->UpdateGroundPlane());

  // The staged scene changed, so the asset's cached thumbnail is stale.
  _prefabAsset->MarkThumbnailDirty();
}

SceneObject* MochiPrefabEditor::StagedObjectForSource(
    StagedActorSource source,
    int actorIndex,
    int linkIndex) const {
  if (actorIndex < 0) {
    return nullptr;
  }
  for (auto const& staged : _stage.GetActors()) {
    if (staged.source == source && staged.sourceActorIndex == actorIndex &&
        (linkIndex < 0 || staged.sourceLinkIndex == linkIndex)) {
      return staged.sceneObject;
    }
  }
  return nullptr;
}

//------------------------------------------------------------------------------------------------
// Mochi Scene
//------------------------------------------------------------------------------------------------

bool MochiPrefabEditor::CanSimulate() const {
  return !_stage.IsEmpty();
}

void MochiPrefabEditor::CreatePhysicsActors(mochi::Scene* scene) {
  auto* mochiContext = _studio->GetMochiContext();
  std::string const rootPath = _prefabAsset->GetAssetsRoot();

  mochi::ErrorLog e;
  mochi::prefab::EnsureFullyLoaded(_prefabAsset->GetPrefab(), rootPath, mochiContext, e);
  if (!e.IsOK()) {
    MOCHI_LOG_ERROR("Failed to load shapes for prefab physics");
    return;
  }

  _physicsActors.clear();
  mochi::prefab::PrefabParams prefabParams;
  auto const result = mochi::prefab::AddToScene(_prefabAsset->GetPrefab(), scene, prefabParams, e);
  if (!e.IsOK()) {
    MOCHI_LOG_ERROR("Failed to add prefab to physics");
    return;
  }
  for (auto* actor : result.actors) {
    _physicsActors.push_back(actor->GetHandle());
  }

  // Register per-soft-actor surface queries so the deforming mesh can be read each step. Kept until
  // the actors are torn down (cancelled in DestroyPhysicsActors).
  _softQueryState.clear();
  for (auto* actor : result.actors) {
    if (actor->GetType() != mochi::ActorType::Soft) {
      continue;
    }
    SoftQueryState state;
    state.actor = actor->GetHandle();
    state.positions = actor->RegisterQuery(mochi::QueryType::SurfaceNodePositions, e);
    state.normals = actor->RegisterQuery(mochi::QueryType::SurfaceNodeNormals, e);
    if (!e.IsOK()) {
      // Consistent with the EnsureFullyLoaded / AddToScene error handling above: abort setup rather
      // than storing a half-registered soft actor whose per-frame mesh update would read nothing.
      MOCHI_LOG_ERROR("Failed to register soft actor surface queries");
      return;
    }
    char const* const name = actor->GetName();
    state.name = name ? name : "";
    _softQueryState.push_back(std::move(state));
  }
}

void MochiPrefabEditor::DestroyPhysicsActors(mochi::Scene* scene) {
  // Release the soft-actor surface queries while the actors are still alive (this callback runs
  // before the scene is destroyed).
  for (auto const& state : _softQueryState) {
    if (auto* actor = scene->GetActor(state.actor)) {
      actor->CancelQuery(state.positions);
      actor->CancelQuery(state.normals);
    }
  }
}

mochi::CallbackHandle MochiPrefabEditor::RegisterPostStepCallback(mochi::AsyncScene* scene) {
  return scene->RegisterPostStepCallback(
      "MochiPrefabEditor::ExtractActors", [this](mochi::StepInfo const& info) {
        mochi::ErrorLog e;
        auto& data = _simData.GetProducerData();
        data.actorTransforms.clear();
        data.actorNames.clear();
        // Build the ordered transform list to match StagePrefab's staged-actor order: articulated
        // actors expand to their nested link transforms; rigid and soft actors push their root
        // transform. Soft actors are included here so their world transform is applied by name and
        // their row appears in the Render Scene Stage debug view; their deformed surface mesh is
        // additionally applied via the per-vertex mesh-update channel below (ApplySoftMeshUpdates).
        for (auto const& handle : _physicsActors) {
          auto* actor = info.scene->GetActor(handle);
          if (!actor) {
            continue;
          }
          if (actor->GetType() == mochi::ActorType::Articulated) {
            auto const links = actor->GetNestedLinkActors(e);
            std::vector<mochi::TransformRT> linkTransforms(links.size());
            actor->GetArticulatedLinkTransforms(mochi::MakeSpan(linkTransforms), e);
            for (int i = 0; i < isize(linkTransforms); ++i) {
              data.actorTransforms.push_back(linkTransforms[i]);
              char const* const linkName = info.scene->GetActor(links[i])->GetName();
              data.actorNames.emplace_back(linkName ? linkName : "");
            }
          } else if (
              actor->GetType() == mochi::ActorType::Rigid ||
              actor->GetType() == mochi::ActorType::Soft) {
            data.actorTransforms.push_back(actor->GetRootTransform());
            char const* const name = actor->GetName();
            data.actorNames.emplace_back(name ? name : "");
          }
        }
        // Soft actors: overwrite per-vertex surface deformations (matched to staged meshes by
        // name), converting local positions/normals to renderer space. The SoftMeshUpdate entries
        // and their position/normal buffers are kept resident across steps and resized/overwritten
        // in place to avoid per-step reallocation; the vector is trimmed to the number of live soft
        // actors so the consumer sees exactly the valid updates.
        auto const& converter = _studio->GetEditorToRendererSpaceConverter();
        data.softMeshUpdates.resize(_softQueryState.size());
        int updateCount = 0;
        for (auto const& state : _softQueryState) {
          auto* actor = info.scene->GetActor(state.actor);
          if (!actor) {
            continue;
          }
          auto const positions = actor->GetSurfaceMeshNodePositionsLocal(e);
          auto const normals = actor->GetSurfaceMeshNodeNormalsLocal(e);
          int const numNodes = static_cast<int>(positions.size() / 3);
          int const numNormals = static_cast<int>(normals.size() / 3);
          SoftMeshUpdate& update = data.softMeshUpdates[updateCount++];
          update.name = state.name;
          update.worldTransform = actor->GetRootTransform();
          update.positions.resize(static_cast<size_t>(numNodes) * 3);
          update.normals.resize(static_cast<size_t>(numNodes) * 3);
          for (int n = 0; n < numNodes; ++n) {
            auto const p = converter.TranslationToOutput(
                StaticCast<Float3>(mochi::Real3{
                    positions[3 * n + 0], positions[3 * n + 1], positions[3 * n + 2]}));
            update.positions[3 * n + 0] = p[0];
            update.positions[3 * n + 1] = p[1];
            update.positions[3 * n + 2] = p[2];
            mochi::Float3 nm{0.0f, 1.0f, 0.0f};
            if (n < numNormals) {
              nm = {
                  static_cast<float>(normals[3 * n + 0]),
                  static_cast<float>(normals[3 * n + 1]),
                  static_cast<float>(normals[3 * n + 2])};
            }
            nm = converter.DirectionToOutput(nm);
            update.normals[3 * n + 0] = nm[0];
            update.normals[3 * n + 1] = nm[1];
            update.normals[3 * n + 2] = nm[2];
          }
        }
        // Trim any trailing entries left over from soft actors that were missing this step (rare;
        // reuse keeps the common all-present case allocation-free).
        data.softMeshUpdates.resize(updateCount);
        _simData.Produce();
      });
}

void MochiPrefabEditor::OnStopPhysics() {
  // Tear down the per-session force-drag controller (its callback is already gone with the scene).
  _dragController.reset();
  _softQueryState.clear();
  _stage.ResetWorldTransforms(_studio->GetEditorToRendererSpaceConverter());
  _simData.Consume();
  _physicsActors.clear();
}

void MochiPrefabEditor::SyncFromPhysics() {
  if (_simData.Consume()) {
    auto const& data = _simData.GetConsumerData();
    auto const& converter = _studio->GetEditorToRendererSpaceConverter();
    // Rigid, soft, and articulated actors are matched by name here. Soft actors additionally have
    // their deformed surface mesh applied by ApplySoftMeshUpdates below.
    _stage.ApplyWorldTransforms(
        mochi::MakeConstSpan(data.actorNames),
        mochi::MakeConstSpan(data.actorTransforms),
        converter);
    _stage.ApplySoftMeshUpdates(mochi::MakeConstSpan(data.softMeshUpdates), converter);
  }
}

//--------------------------------------------------------------------------------------------------
// Viewport
//--------------------------------------------------------------------------------------------------

MochiPrefabEditor::Selection MochiPrefabEditor::SelectionForRow(HierarchyRow const& row) {
  switch (row.category) {
    case RowCategory::Rigid:
      return {Selection::Kind::Rigid, row.index, -1};
    case RowCategory::Soft:
      return {Selection::Kind::Soft, row.index, -1};
    case RowCategory::Articulated:
      return {Selection::Kind::ArticulatedActor, row.index, -1};
    case RowCategory::NestedPrefab:
      return {Selection::Kind::NestedPrefab, row.index, -1};
  }
  return {};
}

MochiPrefabEditor::Selection MochiPrefabEditor::SelectionForObject(SceneObject* object) const {
  if (!object) {
    return {};
  }
  int const idx = _stage.GetSceneObjectIndex(object);
  if (idx < 0 || idx >= _stage.GetNumActors()) {
    return {};
  }
  auto const& staged = _stage.GetActors()[idx];
  switch (staged.source) {
    case StagedActorSource::Rigid:
      return {Selection::Kind::Rigid, staged.sourceActorIndex, -1};
    case StagedActorSource::Soft:
      return {Selection::Kind::Soft, staged.sourceActorIndex, -1};
    case StagedActorSource::Articulated:
      return {Selection::Kind::ArticulatedLink, staged.sourceActorIndex, staged.sourceLinkIndex};
    case StagedActorSource::NestedPrefab:
      // Picking any actor inside a nested prefab selects that nested reference (so the gizmo
      // shows).
      if (staged.sourceActorIndex >= 0) {
        return {Selection::Kind::NestedPrefab, staged.sourceActorIndex, -1};
      }
      return {};
  }
  return {};
}

void MochiPrefabEditor::OnSceneSelectionChanged(std::vector<SceneObject*> const& objects) {
  _selections.clear();
  _selectionAnchorRow = -1;
  for (SceneObject* object : objects) {
    Selection const sel = SelectionForObject(object);
    if (sel.kind == Selection::Kind::None) {
      continue;
    }
    // A nested prefab spans several scene objects; keep only one selection entry for it.
    bool const already =
        std::any_of(_selections.begin(), _selections.end(), [&](Selection const& s) {
          return s.kind == sel.kind && s.actorIndex == sel.actorIndex &&
              s.linkIndex == sel.linkIndex;
        });
    if (!already) {
      _selections.push_back(sel);
    }
  }
  // Expand-and-scroll to a link only when a single link was picked in the viewport.
  if (_selections.size() == 1 && _selections.back().kind == Selection::Kind::ArticulatedLink) {
    _forceLinkFocus = true;
  }
  // A viewport pick highlights only the clicked object, but a nested prefab or articulated actor is
  // a single unit spanning several staged objects. Re-sync so the whole unit reads as selected.
  SyncViewportSelection();
}

MochiPrefabEditor::Selection const* MochiPrefabEditor::PrimarySelection() const {
  return _selections.empty() ? nullptr : &_selections.back();
}

bool MochiPrefabEditor::IsRowSelected(Selection::Kind kind, int actorIndex) const {
  for (Selection const& sel : _selections) {
    // An articulated actor row is highlighted whether the actor itself or one of its links is
    // selected.
    bool const kindMatches = sel.kind == kind ||
        (kind == Selection::Kind::ArticulatedActor && sel.kind == Selection::Kind::ArticulatedLink);
    if (kindMatches && sel.actorIndex == actorIndex) {
      return true;
    }
  }
  return false;
}

void MochiPrefabEditor::SyncViewportSelection() {
  std::vector<SceneObject*> objects;
  for (Selection const& sel : _selections) {
    switch (sel.kind) {
      case Selection::Kind::Rigid:
        if (auto* o = StagedObjectForSource(StagedActorSource::Rigid, sel.actorIndex)) {
          objects.push_back(o);
        }
        break;
      case Selection::Kind::Soft:
        if (auto* o = StagedObjectForSource(StagedActorSource::Soft, sel.actorIndex)) {
          objects.push_back(o);
        }
        break;
      case Selection::Kind::ArticulatedActor:
      case Selection::Kind::ArticulatedLink:
      case Selection::Kind::NestedPrefab:
        // An articulated actor (whether the actor itself or one of its links is selected) and a
        // nested prefab each span multiple staged objects; highlight all of them so the whole unit
        // reads as selected in the viewport.
        for (auto const& staged : _stage.GetActors()) {
          StagedActorSource const wanted = sel.kind == Selection::Kind::NestedPrefab
              ? StagedActorSource::NestedPrefab
              : StagedActorSource::Articulated;
          if (staged.source == wanted && staged.sourceActorIndex == sel.actorIndex &&
              staged.sceneObject) {
            objects.push_back(staged.sceneObject);
          }
        }
        break;
      case Selection::Kind::None:
        break;
    }
  }
  // Rebind without re-entering OnSceneSelectionChanged (which would rebuild _selections and drop
  // the multi-object mappings above).
  _viewport->SetSelectedSceneObjects(std::move(objects), /*invokeCallback=*/false);
}

void MochiPrefabEditor::SetSingleSelection(Selection selection) {
  if (selection.kind == Selection::Kind::None) {
    _selections.clear();
  } else {
    _selections = {selection};
  }
  _selectionAnchorRow = -1;
  SyncViewportSelection();
}

void MochiPrefabEditor::HandleTopLevelRowClick(int rowIndex, Selection const& sel) {
  ImGuiIO const& io = ImGui::GetIO();
  if (io.KeyCtrl) {
    // Toggle this row in/out of the selection.
    auto it = std::find_if(_selections.begin(), _selections.end(), [&](Selection const& s) {
      return s.kind == sel.kind && s.actorIndex == sel.actorIndex;
    });
    if (it != _selections.end()) {
      _selections.erase(it);
    } else {
      _selections.push_back(sel);
    }
    _selectionAnchorRow = rowIndex;
    SyncViewportSelection();
  } else if (
      io.KeyShift && _selectionAnchorRow >= 0 &&
      _selectionAnchorRow < static_cast<int>(_hierarchyRows.size())) {
    // Select the contiguous range of top-level rows (in the current sorted order) from the anchor
    // to this row.
    int const lo = std::min(_selectionAnchorRow, rowIndex);
    int const hi = std::max(_selectionAnchorRow, rowIndex);
    _selections.clear();
    for (int r = lo; r <= hi && r < static_cast<int>(_hierarchyRows.size()); ++r) {
      _selections.push_back(SelectionForRow(_hierarchyRows[r]));
    }
    SyncViewportSelection();
  } else {
    // Plain click replaces the selection and sets the range anchor.
    _selections = {sel};
    _selectionAnchorRow = rowIndex;
    SyncViewportSelection();
  }
  FocusWindowIfOpen("Actor Details");
}

std::optional<std::pair<mochi::TransformRT, mochi::Real3>>
MochiPrefabEditor::GetTransformGizmoTarget() const {
  auto const& prefab = _prefabAsset->GetPrefab();
  // Per-axis-scale target (rigid/soft): authored transform + the actor's Real3 scale.
  auto perAxis = [](mochi::Quaternion const& rotation,
                    mochi::Real3 const& translation,
                    mochi::Real3 const& scale) {
    return std::make_pair(mochi::TransformRT(mochi::Normalize(rotation), translation), scale);
  };
  // Scalar-scale target (articulated/nested): authored transform + a uniform scale broadcast.
  auto scalar =
      [](mochi::Quaternion const& rotation, mochi::Real3 const& translation, mochi::real s) {
        return std::make_pair(
            mochi::TransformRT(mochi::Normalize(rotation), translation), mochi::Real3{s, s, s});
      };
  // The gizmo target (transform + scale) for a single selection.
  auto targetForSelection =
      [&](Selection const& sel) -> std::optional<std::pair<mochi::TransformRT, mochi::Real3>> {
    switch (sel.kind) {
      case Selection::Kind::Rigid: {
        auto const& rigid = prefab.actors.rigid;
        if (sel.actorIndex < 0 || sel.actorIndex >= static_cast<int>(rigid.size())) {
          return std::nullopt;
        }
        auto const& actor = rigid[sel.actorIndex];
        return perAxis(actor.rotation, actor.translation, actor.scale);
      }
      // Soft actors expose their authored transform and per-axis scale (like rigid).
      case Selection::Kind::Soft: {
        auto const& soft = prefab.actors.soft;
        if (sel.actorIndex < 0 || sel.actorIndex >= static_cast<int>(soft.size())) {
          return std::nullopt;
        }
        auto const& actor = soft[sel.actorIndex];
        return perAxis(actor.rotation, actor.translation, actor.scale);
      }
      // Articulated actors expose only their root transform for gizmo editing, whether the actor
      // itself or one of its links is selected. The actor's scale is scalar.
      case Selection::Kind::ArticulatedActor:
      case Selection::Kind::ArticulatedLink: {
        auto const& articulated = prefab.actors.articulated;
        if (sel.actorIndex < 0 || sel.actorIndex >= static_cast<int>(articulated.size())) {
          return std::nullopt;
        }
        auto const& actor = articulated[sel.actorIndex];
        return scalar(actor.rotation, actor.translation, actor.scale);
      }
      // Nested prefab references expose their reference transform + uniform scale.
      case Selection::Kind::NestedPrefab: {
        auto const& prefabs = prefab.prefabs;
        if (sel.actorIndex < 0 || sel.actorIndex >= static_cast<int>(prefabs.size())) {
          return std::nullopt;
        }
        auto const& ref = prefabs[sel.actorIndex];
        return scalar(ref.rotation, ref.translation, ref.scale);
      }
      case Selection::Kind::None:
        return std::nullopt;
    }
    return std::nullopt;
  };

  if (_selections.empty()) {
    return std::nullopt;
  }
  // The pivot for both single- and multi-selection is the first selected item's transform; for a
  // multi-selection the delta path then applies the gizmo's incremental motion about that pivot to
  // every selected item.
  return targetForSelection(_selections.front());
}

// The actor scale is baked into the link/skin shapes at instantiation, so cached shape handles must
// be invalidated when the scale changes to force a re-bake on the next physics load.
static void InvalidateArticulatedActorShapes(mochi::prefab::ArticulatedActorPrefab& actor) {
  for (auto& link : actor.links) {
    link.shape = {};
  }
  if (actor.skin) {
    actor.skin->shape = {};
  }
}

void MochiPrefabEditor::OnGizmoTransformDelta(
    mochi::TransformRT const& delta,
    mochi::Real3 const& scaleMul) {
  auto& prefab = _prefabAsset->GetPrefab();

  // The gizmo scales in the pivot's (first selected item's) local frame, so `scaleMul` is a
  // per-frame per-axis multiplier expressed in that frame. A group scale is applied by (a)
  // spreading each member's position about the pivot along those axes and (b) scaling each member's
  // own scale by that same stretch re-expressed in the member's local frame. With a single
  // selection the member is the pivot, so this reduces to a plain per-axis scale of that object (no
  // position change).
  auto const& pivotSel = _selections.front();
  mochi::Quaternion pivotRotation{0.0_r, 0.0_r, 0.0_r, 1.0_r};
  mochi::Real3 pivotPosition{0.0_r, 0.0_r, 0.0_r};
  switch (pivotSel.kind) {
    case Selection::Kind::Rigid:
      if (pivotSel.actorIndex >= 0 &&
          pivotSel.actorIndex < static_cast<int>(prefab.actors.rigid.size())) {
        auto const& actor = prefab.actors.rigid[pivotSel.actorIndex];
        pivotRotation = actor.rotation;
        pivotPosition = actor.translation;
      }
      break;
    case Selection::Kind::Soft:
      if (pivotSel.actorIndex >= 0 &&
          pivotSel.actorIndex < static_cast<int>(prefab.actors.soft.size())) {
        auto const& actor = prefab.actors.soft[pivotSel.actorIndex];
        pivotRotation = actor.rotation;
        pivotPosition = actor.translation;
      }
      break;
    case Selection::Kind::ArticulatedActor:
    case Selection::Kind::ArticulatedLink:
      if (pivotSel.actorIndex >= 0 &&
          pivotSel.actorIndex < static_cast<int>(prefab.actors.articulated.size())) {
        auto const& actor = prefab.actors.articulated[pivotSel.actorIndex];
        pivotRotation = actor.rotation;
        pivotPosition = actor.translation;
      }
      break;
    case Selection::Kind::NestedPrefab:
      if (pivotSel.actorIndex >= 0 &&
          pivotSel.actorIndex < static_cast<int>(prefab.prefabs.size())) {
        auto const& ref = prefab.prefabs[pivotSel.actorIndex];
        pivotRotation = ref.rotation;
        pivotPosition = ref.translation;
      }
      break;
    case Selection::Kind::None:
      break;
  }
  pivotRotation = mochi::Normalize(pivotRotation);
  mochi::Quaternion const worldToPivot = pivotRotation.GetConjugate();

  // Scalar multiplier for scalar-scale actors (articulated/nested): the axis the user actually
  // dragged (the component of scaleMul farthest from 1). Untouched axes stay at 1, so averaging
  // would dilute the drag.
  mochi::real scalarMul = 1.0_r;
  {
    mochi::real maxDev = 0.0_r;
    for (int i = 0; i < 3; ++i) {
      mochi::real const dev = mochi::Abs(scaleMul[i] - 1.0_r);
      if (dev > maxDev) {
        maxDev = dev;
        scalarMul = scaleMul[i];
      }
    }
  }

  // Move a member's position toward/away from the pivot along the pivot's local axes. Identity when
  // scaleMul is all ones (translate/rotate frames) or the member sits at the pivot (single-select).
  auto spreadPosition = [&](mochi::Real3& translation) {
    mochi::Real3 local = worldToPivot * (translation - pivotPosition);
    local = {local[0] * scaleMul[0], local[1] * scaleMul[1], local[2] * scaleMul[2]};
    translation = pivotPosition + pivotRotation * local;
  };

  // Apply the incremental rotation/translation delta about the pivot, then the group position
  // spread. For a single selection `delta` reconstructs the object's absolute transform and the
  // spread is a no-op.
  auto applyRT = [&](mochi::Quaternion& rotation, mochi::Real3& translation) {
    mochi::TransformRT const updated =
        delta * mochi::TransformRT(mochi::Normalize(rotation), translation);
    rotation = updated.GetRotation();
    translation = updated.GetTranslation();
    spreadPosition(translation);
  };
  // Per-axis scale of a Real3-scale actor. The pivot-frame per-axis stretch is re-expressed in the
  // member's local frame and its diagonal taken as the per-axis factor: exact when the member
  // shares the pivot's orientation, a shear-free approximation otherwise. Column k of the
  // member->pivot rotation is `pivotFromMember * e_k`, and the diagonal entry is a convex
  // combination of scaleMul (so it stays within [min, max] of the dragged factors).
  auto scaledReal3 = [&](mochi::Real3 const& scale, mochi::Quaternion const& rotation) {
    mochi::Quaternion const pivotFromMember = worldToPivot * mochi::Normalize(rotation);
    mochi::Real3 factor{1.0_r, 1.0_r, 1.0_r};
    for (int k = 0; k < 3; ++k) {
      mochi::Real3 axis{0.0_r, 0.0_r, 0.0_r};
      axis[k] = 1.0_r;
      mochi::Real3 const column = pivotFromMember * axis;
      factor[k] = scaleMul[0] * column[0] * column[0] + scaleMul[1] * column[1] * column[1] +
          scaleMul[2] * column[2] * column[2];
    }
    return mochi::Real3{scale[0] * factor[0], scale[1] * factor[1], scale[2] * factor[2]};
  };

  // Multiple links of the same articulated actor can be co-selected, but the gizmo delta applies to
  // the actor root, so each articulated actor must be transformed only once.
  std::unordered_set<int> transformedArticulated;
  for (Selection const& sel : _selections) {
    switch (sel.kind) {
      case Selection::Kind::Rigid: {
        auto& rigid = prefab.actors.rigid;
        if (sel.actorIndex < 0 || sel.actorIndex >= static_cast<int>(rigid.size())) {
          break;
        }
        auto& actor = rigid[sel.actorIndex];
        mochi::Quaternion const originalRotation = mochi::Normalize(actor.rotation);
        applyRT(actor.rotation, actor.translation);
        mochi::Real3 const newScale = scaledReal3(actor.scale, originalRotation);
        if (actor.scale != newScale) {
          actor.scale = newScale;
          actor.shape = {};
        }
        break;
      }
      case Selection::Kind::Soft: {
        auto& soft = prefab.actors.soft;
        if (sel.actorIndex < 0 || sel.actorIndex >= static_cast<int>(soft.size())) {
          break;
        }
        auto& actor = soft[sel.actorIndex];
        mochi::Quaternion const originalRotation = mochi::Normalize(actor.rotation);
        applyRT(actor.rotation, actor.translation);
        mochi::Real3 const newScale = scaledReal3(actor.scale, originalRotation);
        if (actor.scale != newScale) {
          actor.scale = newScale;
          actor.shape = {};
        }
        break;
      }
      case Selection::Kind::ArticulatedActor:
      case Selection::Kind::ArticulatedLink: {
        auto& articulated = prefab.actors.articulated;
        if (sel.actorIndex < 0 || sel.actorIndex >= static_cast<int>(articulated.size())) {
          break;
        }
        if (!transformedArticulated.insert(sel.actorIndex).second) {
          break;
        }
        auto& actor = articulated[sel.actorIndex];
        applyRT(actor.rotation, actor.translation);
        mochi::real const newScale = actor.scale * scalarMul;
        if (newScale != actor.scale) {
          actor.scale = newScale;
          InvalidateArticulatedActorShapes(actor);
        }
        break;
      }
      case Selection::Kind::NestedPrefab: {
        auto& prefabs = prefab.prefabs;
        if (sel.actorIndex < 0 || sel.actorIndex >= static_cast<int>(prefabs.size())) {
          break;
        }
        auto& ref = prefabs[sel.actorIndex];
        applyRT(ref.rotation, ref.translation);
        ref.scale *= scalarMul;
        break;
      }
      case Selection::Kind::None:
        break;
    }
  }
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  GetUndoStack().MarkEdited();
}

//--------------------------------------------------------------------------------------------------
// Structural edits
//--------------------------------------------------------------------------------------------------

int MochiPrefabEditor::AddArticulatedLink(
    mochi::prefab::ArticulatedActorPrefab& actor,
    int parentIdx,
    std::string_view baseName) {
  using namespace mochi;
  if (parentIdx < 0 || parentIdx >= static_cast<int>(actor.links.size())) {
    return -1;
  }
  // Generate a unique link name.
  // Note: std::unordered_set<DynamicString> does not compile on some platforms.
  std::set<DynamicString> existingNames;
  for (auto const& link : actor.links) {
    existingNames.insert(link.name);
  }
  DynamicString const base(baseName);
  DynamicString name = base;
  int suffix = 1;
  while (existingNames.count(name) > 0) {
    name = base + "_" + DynamicString(std::to_string(suffix++));
  }
  // Append the new link/joint. The parent already exists at a lower index, so appending at the end
  // keeps parent-before-child topological order without reindexing existing links.
  mochi::prefab::ArticulatedLinkPrefab newLink;
  newLink.name = name;
  newLink.parentLink = parentIdx;
  mochi::prefab::ArticulatedJointPrefab newJoint;
  newJoint.type = ArticulatedJointType::Hard;
  newJoint.name = name + "_joint";
  actor.links.push_back(std::move(newLink));
  actor.joints.push_back(std::move(newJoint));
  return static_cast<int>(actor.links.size()) - 1;
}

void MochiPrefabEditor::RemoveArticulatedLinkAndDescendants(
    mochi::prefab::ArticulatedActorPrefab& actor,
    int linkIdx) {
  using namespace mochi;
  int const numLinks = static_cast<int>(actor.links.size());
  // The root link (no parent) cannot be removed.
  if (linkIdx <= 0 || linkIdx >= numLinks ||
      actor.links[linkIdx].parentLink == superdex::robotics::kIndexNone) {
    return;
  }
  // Mark linkIdx and all its descendants. A single forward pass suffices because appends preserve
  // topological order (a link's parent always has a smaller index).
  std::vector<bool> toRemove(numLinks, false);
  toRemove[linkIdx] = true;
  for (int i = linkIdx + 1; i < numLinks; ++i) {
    int const parent = actor.links[i].parentLink;
    if (parent >= 0 && parent < numLinks && toRemove[parent]) {
      toRemove[i] = true;
    }
  }
  // Rebuild the links/joints arrays with remapped parent indices (joint i connects link i).
  std::vector<int> indexMap(numLinks, superdex::robotics::kIndexNone);
  DynamicArray<mochi::prefab::ArticulatedLinkPrefab> newLinks;
  DynamicArray<mochi::prefab::ArticulatedJointPrefab> newJoints;
  for (int i = 0; i < numLinks; ++i) {
    if (toRemove[i]) {
      continue;
    }
    indexMap[i] = static_cast<int>(newLinks.size());
    mochi::prefab::ArticulatedLinkPrefab link = actor.links[i];
    if (link.parentLink != superdex::robotics::kIndexNone) {
      link.parentLink = indexMap[link.parentLink];
    }
    newLinks.push_back(std::move(link));
    if (i < static_cast<int>(actor.joints.size())) {
      newJoints.push_back(actor.joints[i]);
    }
  }
  actor.links = std::move(newLinks);
  actor.joints = std::move(newJoints);
  // cycles/jointVelocities encode DOF/joint indices this editor does not maintain; drop them
  // best-effort so the actor stays self-consistent after a structural edit (known limitation).
  actor.cycles.clear();
  actor.jointVelocities.reset();
}

// Find an articulated link by name; returns its index or kIndexNone if absent.
static int FindArticulatedLinkIndexByName(
    mochi::prefab::ArticulatedActorPrefab const& actor,
    mochi::DynamicString const& name) {
  for (int i = 0; i < static_cast<int>(actor.links.size()); ++i) {
    if (actor.links[i].name == name) {
      return i;
    }
  }
  return superdex::robotics::kIndexNone;
}

// Compute each link's rest (zero-pose) transform relative to the actor root. Requires the
// parent-before-child ordering the actor maintains. Mirrors mochi_shape.cpp:
//   rootFromLink[i] = rootFromLink[parent] * parentLinkFromJoint[i] * parentJointFromLink[i].
static mochi::DynamicArray<mochi::TransformRT> ComputeArticulatedRestRootTransforms(
    mochi::prefab::ArticulatedActorPrefab const& actor) {
  using namespace mochi;
  int const numLinks = static_cast<int>(actor.links.size());
  int const numJoints = static_cast<int>(actor.joints.size());
  DynamicArray<TransformRT> rootFrom(numLinks, TransformRT::Identity());
  for (int i = 0; i < numLinks; ++i) {
    auto const& link = actor.links[i];
    int const parent = link.parentLink;
    if (parent < 0 || parent >= numLinks) {
      continue; // Root link keeps identity.
    }
    TransformRT const parentLinkFromJoint =
        i < numJoints ? actor.joints[i].parentLinkFromJoint : TransformRT::Identity();
    rootFrom[i] = rootFrom[parent] * parentLinkFromJoint * link.parentJointFromLink;
  }
  return rootFrom;
}

// Reorder an articulated actor's links/joints into parent-before-child order (required by the
// articulated actor) via a stable DFS pre-order from the root, remapping parentLink indices.
// Sibling order is preserved, so the hierarchy view is unchanged. Assumes a single valid (acyclic)
// tree.
static void ReorderArticulatedActorTopological(mochi::prefab::ArticulatedActorPrefab& actor) {
  using namespace mochi;
  int const numLinks = static_cast<int>(actor.links.size());
  if (numLinks <= 1) {
    return;
  }
  DynamicArray<DynamicArray<int>> children(numLinks);
  int root = 0;
  for (int i = 0; i < numLinks; ++i) {
    int const p = actor.links[i].parentLink;
    if (p == superdex::robotics::kIndexNone) {
      root = i;
    } else if (p >= 0 && p < numLinks) {
      children[p].push_back(i);
    }
  }
  DynamicArray<int> newOrder;
  newOrder.reserve(numLinks);
  DynamicArray<int> stack;
  stack.push_back(root);
  while (!stack.empty()) {
    int const idx = stack.back();
    stack.pop_back();
    newOrder.push_back(idx);
    // Push children in reverse so they are visited in ascending (current) order, preserving
    // siblings.
    for (int c = static_cast<int>(children[idx].size()) - 1; c >= 0; --c) {
      stack.push_back(children[idx][c]);
    }
  }
  if (static_cast<int>(newOrder.size()) != numLinks) {
    return; // Disconnected/cyclic: leave untouched (caller guards against this).
  }
  DynamicArray<int> oldToNew(numLinks, 0);
  for (int newIdx = 0; newIdx < numLinks; ++newIdx) {
    oldToNew[newOrder[newIdx]] = newIdx;
  }
  int const numJoints = static_cast<int>(actor.joints.size());
  DynamicArray<mochi::prefab::ArticulatedLinkPrefab> newLinks;
  DynamicArray<mochi::prefab::ArticulatedJointPrefab> newJoints;
  newLinks.reserve(numLinks);
  newJoints.reserve(numJoints);
  for (int newIdx = 0; newIdx < numLinks; ++newIdx) {
    int const oldIdx = newOrder[newIdx];
    newLinks.push_back(actor.links[oldIdx]);
    if (oldIdx < numJoints) {
      newJoints.push_back(actor.joints[oldIdx]);
    }
  }
  for (auto& link : newLinks) {
    if (link.parentLink != superdex::robotics::kIndexNone) {
      link.parentLink = oldToNew[link.parentLink];
    }
  }
  actor.links = std::move(newLinks);
  actor.joints = std::move(newJoints);
}

void MochiPrefabEditor::ReparentArticulatedLink(
    int actorIndex,
    int linkIdx,
    int newParentIdx,
    bool preserveWorldTransform) {
  using namespace mochi;
  if (_mochiScene.IsSimulating()) {
    return;
  }
  auto& articulated = _prefabAsset->GetPrefab().actors.articulated;
  if (actorIndex < 0 || actorIndex >= static_cast<int>(articulated.size())) {
    return;
  }
  auto& actor = articulated[actorIndex];
  int const numLinks = static_cast<int>(actor.links.size());
  // The root link (parentLink == kIndexNone) has no parent and cannot be reparented.
  if (linkIdx <= 0 || linkIdx >= numLinks || newParentIdx < 0 || newParentIdx >= numLinks ||
      newParentIdx == linkIdx ||
      actor.links[linkIdx].parentLink == superdex::robotics::kIndexNone) {
    return;
  }
  // Reject reparenting to the link itself or any descendant (would create a cycle). The
  // parent-before-child invariant lets us find descendants in a single forward pass.
  std::vector<bool> isDescendant(numLinks, false);
  isDescendant[linkIdx] = true;
  for (int i = linkIdx + 1; i < numLinks; ++i) {
    int const p = actor.links[i].parentLink;
    if (p >= 0 && p < numLinks && isDescendant[p]) {
      isDescendant[i] = true;
    }
  }
  if (isDescendant[newParentIdx]) {
    return;
  }
  // Capture the link's name so the selection can be restored after any reindexing.
  DynamicString const savedName = actor.links[linkIdx].name;
  int const prevParent = actor.links[linkIdx].parentLink;
  if (preserveWorldTransform) {
    // Keep the link (and its subtree) at the same world placement. Prefab articulations have no
    // overridable pose, so the rest transform is the displayed transform. Holding the suffix (this
    // joint's parentJointFromLink and the whole subtree) fixed and swapping the parent gives:
    //   parentLinkFromJoint_new = Invert(rootFrom[newParent]) * rootFrom[oldParent] *
    //                             parentLinkFromJoint_old
    // rootFrom[newParent] is unchanged by the reparent (the new parent is not a descendant).
    DynamicArray<TransformRT> const rootFrom = ComputeArticulatedRestRootTransforms(actor);
    if (static_cast<int>(rootFrom.size()) == numLinks &&
        linkIdx < static_cast<int>(actor.joints.size())) {
      actor.joints[linkIdx].parentLinkFromJoint = Invert(rootFrom[newParentIdx]) *
          rootFrom[prevParent] * actor.joints[linkIdx].parentLinkFromJoint;
    }
  }
  actor.links[linkIdx].parentLink = newParentIdx;
  // The articulated actor requires parent-before-child ordering. Reparenting under a higher-index
  // link violates it, so re-establish topological order. Reindexing invalidates cycle/velocity data
  // that references joint/DOF indices, so drop it best-effort (as link removal does).
  if (newParentIdx > linkIdx) {
    ReorderArticulatedActorTopological(actor);
    actor.cycles.clear();
    actor.jointVelocities.reset();
  }
  SetSingleSelection(
      {Selection::Kind::ArticulatedLink,
       actorIndex,
       FindArticulatedLinkIndexByName(actor, savedName)});
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
}

//--------------------------------------------------------------------------------------------------
// ImGui
//--------------------------------------------------------------------------------------------------

void MochiPrefabEditor::HighlightStagedObject(SceneObject* object) {
  if (object) {
    _viewport->HighlightSceneObject(object, {46.f / 255.f, 134.f / 255.f, 233.f / 255.f});
  }
}

void MochiPrefabEditor::RenderArticulatedLinkTree(
    int actorIndex,
    mochi::prefab::ArticulatedActorPrefab const& actor,
    std::vector<std::vector<int>> const& children,
    std::unordered_set<int> const& ancestorsToOpen,
    int linkIndex,
    bool simulating,
    HierarchyEdits& edits) {
  auto const& link = actor.links[linkIndex];
  bool const hasChildren = !children[linkIndex].empty();
  bool const isRoot = link.parentLink == superdex::robotics::kIndexNone;
  Selection const* const primary = PrimarySelection();
  bool const selected = primary && primary->kind == Selection::Kind::ArticulatedLink &&
      primary->actorIndex == actorIndex && primary->linkIndex == linkIndex;

  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::PushID(linkIndex);

  if (_forceLinkFocus && ancestorsToOpen.count(linkIndex) > 0) {
    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
  }

  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanFullWidth |
      ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnDoubleClick |
      ImGuiTreeNodeFlags_SpanAllColumns;
  if (!hasChildren) {
    flags |=
        ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_Bullet;
  }
  if (selected) {
    flags |= ImGuiTreeNodeFlags_Selected;
  }

  std::string const linkName = FallbackName(kLinkPrefix, link.name, linkIndex);
  bool const nodeOpen = ImGui::TreeNodeEx(linkName.c_str(), flags);
  if (ImGui::IsItemHovered()) {
    HighlightStagedObject(
        StagedObjectForSource(StagedActorSource::Articulated, actorIndex, linkIndex));
  }
  if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
    // Articulated links are single-select only (they clear any multi-selection).
    SetSingleSelection({Selection::Kind::ArticulatedLink, actorIndex, linkIndex});
    FocusWindowIfOpen("Actor Details");
  }
  if (!simulating && ImGui::BeginPopupContextItem("LinkPopup", ImGuiPopupFlags_MouseButtonRight)) {
    if (ImGui::Selectable("Add Child Link")) {
      edits.addLinkActor = actorIndex;
      edits.addLinkParent = linkIndex;
    }
    ImGui::BeginDisabled(isRoot);
    if (ImGui::Selectable("Delete Link")) {
      edits.delLinkActor = actorIndex;
      edits.delLinkIndex = linkIndex;
    }
    ImGui::EndDisabled();
    ImGui::EndPopup();
  }

  ImGui::TableNextColumn();
  ImGui::TextDisabled("Articulated Link");

  if (_forceLinkFocus && selected) {
    ImGui::SetScrollHereY(0.5f);
    _forceLinkFocus = false;
  }

  if (hasChildren && nodeOpen) {
    for (int child : children[linkIndex]) {
      RenderArticulatedLinkTree(
          actorIndex, actor, children, ancestorsToOpen, child, simulating, edits);
    }
    ImGui::TreePop();
  }
  ImGui::PopID();
}

void MochiPrefabEditor::ShowPrefabHierarchyWindow(bool* open) {
  auto& prefab = _prefabAsset->GetPrefab();

  ImGuiWindowFlags scenePrefabFlags =
      _prefabAsset->IsDirty() ? ImGuiWindowFlags_UnsavedDocument : ImGuiWindowFlags_None;
  ImGui::Begin("Prefab Hierarchy", open, scenePrefabFlags);

  // Top-level prefab entry (like the Bot Editor's bot-name row): it only raises the Prefab Details
  // window. It is not a selection, so the actor selection is deliberately left untouched.
  ImGui::PushFont(_studio->GetFont("Roboto Bold Large"));
  if (ImGui::Selectable(_prefabAsset->GetName().c_str(), false)) {
    FocusWindowIfOpen("Prefab Details");
  }
  ImGui::PopFont();
  ImGui::Separator();

  // Flatten every actor kind into a single row list. Each row carries its source category and index
  // so selection/edit/delete map back to the underlying prefab arrays after sorting. The row buffer
  // is a persistent member, cleared and reused to avoid a per-frame allocation.
  auto& rows = _hierarchyRows;
  rows.clear();
  rows.reserve(
      prefab.actors.rigid.size() + prefab.actors.soft.size() + prefab.actors.articulated.size() +
      prefab.prefabs.size());
  for (int r = 0; r < static_cast<int>(prefab.actors.rigid.size()); ++r) {
    rows.push_back(
        {FallbackName(kRigidPrefix, prefab.actors.rigid[r].name, r),
         "Rigid",
         RowCategory::Rigid,
         r});
  }
  for (int s = 0; s < static_cast<int>(prefab.actors.soft.size()); ++s) {
    rows.push_back(
        {FallbackName(kSoftPrefix, prefab.actors.soft[s].name, s), "Soft", RowCategory::Soft, s});
  }
  for (int a = 0; a < static_cast<int>(prefab.actors.articulated.size()); ++a) {
    rows.push_back(
        {FallbackName(kArticulatedPrefix, prefab.actors.articulated[a].name, a),
         "Articulated",
         RowCategory::Articulated,
         a});
  }
  for (int p = 0; p < static_cast<int>(prefab.prefabs.size()); ++p) {
    rows.push_back(
        {FallbackName(kNestedPrefix, prefab.prefabs[p].name, p),
         "Nested Prefab",
         RowCategory::NestedPrefab,
         p});
  }

  // Structural edits are deferred until after the table is rendered (mutating the prefab arrays
  // mid-iteration would invalidate indices).
  HierarchyEdits edits;

  // When at least one articulated actor is present, indent leaf rows' first column so their labels
  // align with the articulated tree-node labels (which sit to the right of the collapse arrow).
  bool const hasArticulated = !prefab.actors.articulated.empty();
  float const leafIndent = ImGui::GetTreeNodeToLabelSpacing();

  // Structural edits (create/delete/duplicate/paste) mutate the prefab and restage, which conflicts
  // with the running simulation, so the right-click context menus are suppressed while simulating
  // (the keyboard shortcuts in OnHandleInputs and the gizmo ctrl-drag are already gated the same
  // way). Left-click selection still works so actors can be inspected during playback.
  bool const simulating = _mochiScene.IsSimulating();

  // Render one Rigid/Soft/Nested leaf row: an indented span-all-columns Selectable that selects the
  // staged object, a hover highlight, and a right-click Copy/Duplicate/Delete popup. Parameterized
  // by selection kind, the staged-object source, the popup id, and the deferred-delete target.
  auto leafRow = [&](int rowIndex,
                     HierarchyRow const& row,
                     Selection::Kind kind,
                     StagedActorSource source,
                     char const* popupId,
                     int& deleteTarget) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    if (hasArticulated) {
      ImGui::Indent(leafIndent);
    }
    bool const isSelected = IsRowSelected(kind, row.index);
    if (ImGui::Selectable(row.name.c_str(), isSelected, ImGuiSelectableFlags_SpanAllColumns)) {
      HandleTopLevelRowClick(rowIndex, {kind, row.index, -1});
    }
    if (ImGui::IsItemHovered()) {
      if (kind == Selection::Kind::NestedPrefab) {
        // Hovering the nested prefab row highlights all of its staged scene objects.
        for (auto const& staged : _stage.GetActors()) {
          if (staged.source == StagedActorSource::NestedPrefab &&
              staged.sourceActorIndex == row.index) {
            HighlightStagedObject(staged.sceneObject);
          }
        }
      } else {
        HighlightStagedObject(StagedObjectForSource(source, row.index));
      }
    }
    if (!simulating && ImGui::BeginPopupContextItem(popupId, ImGuiPopupFlags_MouseButtonRight)) {
      bool const rowInSelection = IsRowSelected(kind, row.index);
      if (ImGui::Selectable("Copy")) {
        if (rowInSelection && _selections.size() > 1) {
          CopySelectionToClipboard();
        } else {
          CopyActorToClipboard(kind, row.index);
        }
      }
      if (ImGui::Selectable("Duplicate")) {
        if (rowInSelection && _selections.size() > 1) {
          edits.duplicateSelection = true;
        } else {
          edits.duplicateKind = kind;
          edits.duplicateIndex = row.index;
        }
      }
      if (ImGui::Selectable("Delete")) {
        if (rowInSelection && _selections.size() > 1) {
          edits.deleteSelection = true;
        } else {
          deleteTarget = row.index;
        }
      }
      ImGui::EndPopup();
    }
    if (hasArticulated) {
      ImGui::Unindent(leafIndent);
    }
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(row.type);
  };

  // Match the Bot Hierarchy table styling.
  ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
  ImGui::BeginChild("PrefabActorsChild");
  auto tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable |
      ImGuiTableFlags_Sortable;
  if (ImGui::BeginTable("##PrefabActors", 2, tableFlags)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_DefaultSort);
    ImGui::TableSetupColumn("Type");
    ImGui::TableHeadersRow();

    // Sort by the active column/direction; Name is the stable tiebreaker.
    if (ImGuiTableSortSpecs* sortSpecs = ImGui::TableGetSortSpecs()) {
      if (sortSpecs->SpecsCount > 0) {
        ImGuiTableColumnSortSpecs const& spec = sortSpecs->Specs[0];
        bool const ascending = spec.SortDirection == ImGuiSortDirection_Ascending;
        std::stable_sort(
            rows.begin(), rows.end(), [&](HierarchyRow const& lhs, HierarchyRow const& rhs) {
              int cmp = 0;
              if (spec.ColumnIndex == 1) {
                cmp = std::strcmp(lhs.type, rhs.type);
              }
              if (cmp == 0) {
                cmp = lhs.name.compare(rhs.name);
              }
              return ascending ? (cmp < 0) : (cmp > 0);
            });
      }
    }

    auto const& articulatedActors = prefab.actors.articulated;

    for (int rowIndex = 0; rowIndex < static_cast<int>(rows.size()); ++rowIndex) {
      HierarchyRow const& row = rows[rowIndex];
      // Two-part ID scope (category, then index) so rows of different kinds sharing an index don't
      // collide.
      ImGui::PushID(static_cast<int>(row.category));
      ImGui::PushID(row.index);

      if (row.category == RowCategory::Rigid) {
        leafRow(
            rowIndex,
            row,
            Selection::Kind::Rigid,
            StagedActorSource::Rigid,
            "ActorPopup",
            edits.rigidToDelete);
      } else if (row.category == RowCategory::Soft) {
        leafRow(
            rowIndex,
            row,
            Selection::Kind::Soft,
            StagedActorSource::Soft,
            "SoftPopup",
            edits.softToDelete);
      } else if (row.category == RowCategory::Articulated) {
        int const a = row.index;
        auto const& actor = articulatedActors[a];
        int const numLinks = static_cast<int>(actor.links.size());
        // Compute each link's children on the fly from parentLink (no persisted child lists),
        // reusing the persistent scratch buffer: grow if needed, then clear the entries we use.
        if (static_cast<int>(_linkChildrenScratch.size()) < numLinks) {
          _linkChildrenScratch.resize(numLinks);
        }
        for (int j = 0; j < numLinks; ++j) {
          _linkChildrenScratch[j].clear();
        }
        for (int j = 0; j < numLinks; ++j) {
          int const parent = actor.links[j].parentLink;
          if (parent >= 0 && parent < numLinks) {
            _linkChildrenScratch[parent].push_back(j);
          }
        }
        // Ancestors of the focused link (within this actor) that must be force-opened.
        std::unordered_set<int> ancestorsToOpen;
        Selection const* const primary = PrimarySelection();
        bool const focusThisActor = _forceLinkFocus && primary &&
            primary->kind == Selection::Kind::ArticulatedLink && primary->actorIndex == a;
        if (focusThisActor && primary->linkIndex >= 0 && primary->linkIndex < numLinks) {
          int current = actor.links[primary->linkIndex].parentLink;
          while (current >= 0 && current < numLinks) {
            ancestorsToOpen.insert(current);
            current = actor.links[current].parentLink;
          }
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        bool const actorSelected = IsRowSelected(Selection::Kind::ArticulatedActor, a);
        if (focusThisActor) {
          ImGui::SetNextItemOpen(true, ImGuiCond_Always);
        }
        ImGuiTreeNodeFlags actorFlags = ImGuiTreeNodeFlags_OpenOnArrow |
            ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAllColumns;
        if (actorSelected) {
          actorFlags |= ImGuiTreeNodeFlags_Selected;
        }
        float const originalIndent = ImGui::GetStyle().IndentSpacing;
        ImGui::GetStyle().IndentSpacing = originalIndent * 0.4f;
        bool const actorOpen = ImGui::TreeNodeEx(row.name.c_str(), actorFlags);
        if (ImGui::IsItemHovered()) {
          // Hovering the actor row highlights all of its links.
          for (int j = 0; j < numLinks; ++j) {
            HighlightStagedObject(StagedObjectForSource(StagedActorSource::Articulated, a, j));
          }
        }
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
          HandleTopLevelRowClick(rowIndex, {Selection::Kind::ArticulatedActor, a, -1});
        }
        if (!simulating &&
            ImGui::BeginPopupContextItem("ArticulatedPopup", ImGuiPopupFlags_MouseButtonRight)) {
          bool const rowInSelection = IsRowSelected(Selection::Kind::ArticulatedActor, a);
          if (ImGui::Selectable("Copy")) {
            if (rowInSelection && _selections.size() > 1) {
              CopySelectionToClipboard();
            } else {
              CopyActorToClipboard(Selection::Kind::ArticulatedActor, a);
            }
          }
          if (ImGui::Selectable("Duplicate")) {
            if (rowInSelection && _selections.size() > 1) {
              edits.duplicateSelection = true;
            } else {
              edits.duplicateKind = Selection::Kind::ArticulatedActor;
              edits.duplicateIndex = a;
            }
          }
          if (ImGui::Selectable("Delete")) {
            if (rowInSelection && _selections.size() > 1) {
              edits.deleteSelection = true;
            } else {
              edits.articulatedActorToDelete = a;
            }
          }
          ImGui::EndPopup();
        }
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(row.type);
        if (actorOpen) {
          for (int j = 0; j < numLinks; ++j) {
            if (actor.links[j].parentLink == superdex::robotics::kIndexNone) {
              RenderArticulatedLinkTree(
                  a, actor, _linkChildrenScratch, ancestorsToOpen, j, simulating, edits);
            }
          }
          ImGui::TreePop();
        }
        ImGui::GetStyle().IndentSpacing = originalIndent;
      } else {
        leafRow(
            rowIndex,
            row,
            Selection::Kind::NestedPrefab,
            StagedActorSource::NestedPrefab,
            "NestedPopup",
            edits.nestedToDelete);
      }

      ImGui::PopID();
      ImGui::PopID();
    }
    ImGui::EndTable();
  }
  // Right-click empty table space to create new top-level actors. The table (ScrollY) has its own
  // inner child window, so detect the hover via ChildWindows and open the popup manually rather
  // than using BeginPopupContextWindow (which only tests the outer child).
  if (!simulating && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !ImGui::IsAnyItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
    ImGui::OpenPopup("PrefabTableContext");
  }
  if (ImGui::BeginPopup("PrefabTableContext")) {
    if (ImGui::Selectable("Create Rigid")) {
      edits.createRigid = true;
    }
    if (ImGui::Selectable("Create Soft")) {
      edits.createSoft = true;
    }
    if (ImGui::Selectable("Create Articulated")) {
      edits.createArticulated = true;
    }
    if (ImGui::Selectable("Create Nested Prefab")) {
      edits.createNested = true;
    }
    ImGui::Separator();
    if (ImGui::Selectable("Paste")) {
      edits.pasteRequested = true;
    }
    ImGui::EndPopup();
  }
  ImGui::EndChild();
  ImGui::PopStyleColor(); // ImGuiCol_Header
  ImGui::End();

  // Common epilogue for the deferred structural edits below: clear selection, restage, mark dirty
  // and push undo. Each branch performs its array erase, then calls this (which also clears any
  // viewport selection pointing at the now-destroyed objects).
  auto applyDeferredEdit = [this]() {
    _viewport->SetSelectedSceneObjects({});
    _selections.clear();
    _selectionAnchorRow = -1;
    RestagePrefab();
    _prefabAsset->SetDirty(true);
    GetUndoStack().PushNow();
  };

  // Apply deferred structural edits (at most one per frame is reachable).
  if (edits.rigidToDelete >= 0 &&
      edits.rigidToDelete < static_cast<int>(prefab.actors.rigid.size())) {
    prefab.actors.rigid.erase(prefab.actors.rigid.begin() + edits.rigidToDelete);
    applyDeferredEdit();
  } else if (
      edits.softToDelete >= 0 && edits.softToDelete < static_cast<int>(prefab.actors.soft.size())) {
    prefab.actors.soft.erase(prefab.actors.soft.begin() + edits.softToDelete);
    applyDeferredEdit();
  } else if (
      edits.articulatedActorToDelete >= 0 &&
      edits.articulatedActorToDelete < static_cast<int>(prefab.actors.articulated.size())) {
    prefab.actors.articulated.erase(
        prefab.actors.articulated.begin() + edits.articulatedActorToDelete);
    applyDeferredEdit();
  } else if (
      edits.addLinkActor >= 0 &&
      edits.addLinkActor < static_cast<int>(prefab.actors.articulated.size())) {
    auto& actor = prefab.actors.articulated[edits.addLinkActor];
    int const newLink = AddArticulatedLink(actor, edits.addLinkParent, "new_link");
    if (newLink >= 0) {
      // Select the newly-created link (rather than clearing selection), so this branch does not use
      // the shared epilogue.
      _selections = {{Selection::Kind::ArticulatedLink, edits.addLinkActor, newLink}};
      _selectionAnchorRow = -1;
      RestagePrefab();
      _prefabAsset->SetDirty(true);
      GetUndoStack().PushNow();
    }
  } else if (
      edits.delLinkActor >= 0 &&
      edits.delLinkActor < static_cast<int>(prefab.actors.articulated.size())) {
    RemoveArticulatedLinkAndDescendants(
        prefab.actors.articulated[edits.delLinkActor], edits.delLinkIndex);
    applyDeferredEdit();
  } else if (edits.createRigid) {
    CreateRigidActor();
  } else if (edits.createSoft) {
    CreateSoftActor();
  } else if (edits.createArticulated) {
    CreateArticulatedActor();
  } else if (edits.createNested) {
    CreateNestedPrefab();
  } else if (
      edits.nestedToDelete >= 0 && edits.nestedToDelete < static_cast<int>(prefab.prefabs.size())) {
    prefab.prefabs.erase(prefab.prefabs.begin() + edits.nestedToDelete);
    applyDeferredEdit();
  } else if (edits.duplicateSelection) {
    DuplicateSelection();
  } else if (edits.deleteSelection) {
    DeleteSelection();
  } else if (edits.duplicateKind != Selection::Kind::None) {
    DuplicateActor(edits.duplicateKind, edits.duplicateIndex);
  } else if (edits.pasteRequested) {
    PasteFromClipboard();
  }
}

std::vector<std::string> MochiPrefabEditor::CollectTopLevelNames() const {
  auto const& prefab = _prefabAsset->GetPrefab();
  std::vector<std::string> names;
  names.reserve(
      prefab.actors.rigid.size() + prefab.actors.soft.size() + prefab.actors.articulated.size() +
      prefab.prefabs.size());
  for (int i = 0; i < static_cast<int>(prefab.actors.rigid.size()); ++i) {
    names.push_back(FallbackName(kRigidPrefix, prefab.actors.rigid[i].name, i));
  }
  for (int i = 0; i < static_cast<int>(prefab.actors.soft.size()); ++i) {
    names.push_back(FallbackName(kSoftPrefix, prefab.actors.soft[i].name, i));
  }
  for (int i = 0; i < static_cast<int>(prefab.actors.articulated.size()); ++i) {
    names.push_back(FallbackName(kArticulatedPrefix, prefab.actors.articulated[i].name, i));
  }
  for (int i = 0; i < static_cast<int>(prefab.prefabs.size()); ++i) {
    names.push_back(FallbackName(kNestedPrefix, prefab.prefabs[i].name, i));
  }
  return names;
}

std::string MochiPrefabEditor::MakeUniquePrefabActorName(std::string_view base) const {
  auto const resolved = CollectTopLevelNames();
  std::set<std::string> existing(resolved.begin(), resolved.end());
  // Strip any trailing digits so the numbering continues from the stem (e.g. duplicating "actor1"
  // yields "actor2", not "actor11"). A purely numeric base keeps its digits as the stem.
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

void MochiPrefabEditor::CreateRigidActor() {
  auto& prefab = _prefabAsset->GetPrefab();
  mochi::prefab::RigidActorPrefab actor;
  actor.name = mochi::DynamicString{MakeUniquePrefabActorName("Rigid")};
  prefab.actors.rigid.push_back(std::move(actor));
  _selections = {{Selection::Kind::Rigid, static_cast<int>(prefab.actors.rigid.size()) - 1, -1}};
  _selectionAnchorRow = -1;
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
}

void MochiPrefabEditor::CreateSoftActor() {
  auto& prefab = _prefabAsset->GetPrefab();
  mochi::prefab::SoftActorPrefab actor;
  actor.name = mochi::DynamicString{MakeUniquePrefabActorName("Soft")};
  prefab.actors.soft.push_back(std::move(actor));
  _selections = {{Selection::Kind::Soft, static_cast<int>(prefab.actors.soft.size()) - 1, -1}};
  _selectionAnchorRow = -1;
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
}

void MochiPrefabEditor::CreateArticulatedActor() {
  auto& prefab = _prefabAsset->GetPrefab();
  mochi::prefab::ArticulatedActorPrefab actor;
  actor.name = mochi::DynamicString{MakeUniquePrefabActorName("Articulated")};
  // Seed a fixed root link + joint so the actor is a valid minimal articulation.
  mochi::prefab::ArticulatedLinkPrefab root;
  root.name = mochi::DynamicString{"root"};
  root.parentLink = superdex::robotics::kIndexNone;
  mochi::prefab::ArticulatedJointPrefab rootJoint;
  rootJoint.type = mochi::ArticulatedJointType::Hard;
  rootJoint.name = mochi::DynamicString{"root_joint"};
  actor.links.push_back(std::move(root));
  actor.joints.push_back(std::move(rootJoint));
  prefab.actors.articulated.push_back(std::move(actor));
  _selections = {
      {Selection::Kind::ArticulatedActor,
       static_cast<int>(prefab.actors.articulated.size()) - 1,
       -1}};
  _selectionAnchorRow = -1;
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
}

void MochiPrefabEditor::CreateNestedPrefab() {
  auto& prefab = _prefabAsset->GetPrefab();
  mochi::prefab::PrefabReference nested;
  nested.name = mochi::DynamicString{MakeUniquePrefabActorName("Nested")};
  prefab.prefabs.push_back(std::move(nested));
  _selections = {{Selection::Kind::NestedPrefab, static_cast<int>(prefab.prefabs.size()) - 1, -1}};
  _selectionAnchorRow = -1;
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
}

// Tagged text envelope for a single actor on the system clipboard: "<prefix><kind>\n<json>". The
// text is plain reflected JSON so it round-trips across editor instances and external text editors.
static constexpr std::string_view kClipboardPrefix = "MochiPrefabActor:";
// Separator joining multiple actor blocks in a group copy. Chosen so it cannot appear inside the
// reflected JSON payloads, letting PasteFromClipboard split a multi-actor clipboard back into
// individual blocks.
static constexpr std::string_view kClipboardSeparator = "\n@@MOCHI_PREFAB_ACTOR@@\n";

std::string MochiPrefabEditor::BuildActorClipboardBlock(Selection::Kind kind, int actorIndex)
    const {
  auto const& prefab = _prefabAsset->GetPrefab();
  char const* tag = nullptr;
  std::string json;
  switch (kind) {
    case Selection::Kind::Rigid:
      if (actorIndex >= 0 && actorIndex < static_cast<int>(prefab.actors.rigid.size())) {
        tag = "rigid";
        json = SReflect::ToJsonString(prefab.actors.rigid[actorIndex], true);
      }
      break;
    case Selection::Kind::Soft:
      if (actorIndex >= 0 && actorIndex < static_cast<int>(prefab.actors.soft.size())) {
        tag = "soft";
        json = SReflect::ToJsonString(prefab.actors.soft[actorIndex], true);
      }
      break;
    case Selection::Kind::ArticulatedActor:
    case Selection::Kind::ArticulatedLink:
      if (actorIndex >= 0 && actorIndex < static_cast<int>(prefab.actors.articulated.size())) {
        tag = "articulated";
        json = SReflect::ToJsonString(prefab.actors.articulated[actorIndex], true);
      }
      break;
    case Selection::Kind::NestedPrefab:
      if (actorIndex >= 0 && actorIndex < static_cast<int>(prefab.prefabs.size())) {
        tag = "nested";
        json = SReflect::ToJsonString(prefab.prefabs[actorIndex], true);
      }
      break;
    case Selection::Kind::None:
      break;
  }
  if (!tag) {
    return {};
  }
  return std::string(kClipboardPrefix) + tag + "\n" + json;
}

void MochiPrefabEditor::CopyActorToClipboard(Selection::Kind kind, int actorIndex) const {
  std::string const block = BuildActorClipboardBlock(kind, actorIndex);
  if (block.empty()) {
    return;
  }
  ImGui::SetClipboardText(block.c_str());
}

void MochiPrefabEditor::CopySelectionToClipboard() const {
  std::string clip;
  for (Selection const& sel : _selections) {
    std::string const block = BuildActorClipboardBlock(sel.kind, sel.actorIndex);
    if (block.empty()) {
      continue;
    }
    if (!clip.empty()) {
      clip += kClipboardSeparator;
    }
    clip += block;
  }
  if (!clip.empty()) {
    ImGui::SetClipboardText(clip.c_str());
  }
}

std::optional<MochiPrefabEditor::Selection> MochiPrefabEditor::DuplicateActorNoCommit(
    Selection::Kind kind,
    int actorIndex) {
  auto& prefab = _prefabAsset->GetPrefab();
  switch (kind) {
    case Selection::Kind::Rigid: {
      auto& rigid = prefab.actors.rigid;
      if (actorIndex < 0 || actorIndex >= static_cast<int>(rigid.size())) {
        return std::nullopt;
      }
      mochi::prefab::RigidActorPrefab copy = rigid[actorIndex];
      copy.name = mochi::DynamicString{
          MakeUniquePrefabActorName(copy.name.empty() ? "Rigid" : std::string(copy.name))};
      rigid.push_back(std::move(copy));
      return Selection{Selection::Kind::Rigid, static_cast<int>(rigid.size()) - 1, -1};
    }
    case Selection::Kind::Soft: {
      auto& soft = prefab.actors.soft;
      if (actorIndex < 0 || actorIndex >= static_cast<int>(soft.size())) {
        return std::nullopt;
      }
      mochi::prefab::SoftActorPrefab copy = soft[actorIndex];
      copy.name = mochi::DynamicString{
          MakeUniquePrefabActorName(copy.name.empty() ? "Soft" : std::string(copy.name))};
      soft.push_back(std::move(copy));
      return Selection{Selection::Kind::Soft, static_cast<int>(soft.size()) - 1, -1};
    }
    case Selection::Kind::ArticulatedActor:
    case Selection::Kind::ArticulatedLink: {
      auto& articulated = prefab.actors.articulated;
      if (actorIndex < 0 || actorIndex >= static_cast<int>(articulated.size())) {
        return std::nullopt;
      }
      mochi::prefab::ArticulatedActorPrefab copy = articulated[actorIndex];
      copy.name = mochi::DynamicString{
          MakeUniquePrefabActorName(copy.name.empty() ? "Articulated" : std::string(copy.name))};
      articulated.push_back(std::move(copy));
      return Selection{
          Selection::Kind::ArticulatedActor, static_cast<int>(articulated.size()) - 1, -1};
    }
    case Selection::Kind::NestedPrefab: {
      auto& prefabs = prefab.prefabs;
      if (actorIndex < 0 || actorIndex >= static_cast<int>(prefabs.size())) {
        return std::nullopt;
      }
      mochi::prefab::PrefabReference copy = prefabs[actorIndex];
      copy.name = mochi::DynamicString{
          MakeUniquePrefabActorName(copy.name.empty() ? "Nested" : std::string(copy.name))};
      prefabs.push_back(std::move(copy));
      _studio->GetAssetManager().ResyncReferencer(_prefabAsset);
      return Selection{Selection::Kind::NestedPrefab, static_cast<int>(prefabs.size()) - 1, -1};
    }
    case Selection::Kind::None:
      return std::nullopt;
  }
  return std::nullopt;
}

void MochiPrefabEditor::DuplicateActor(Selection::Kind kind, int actorIndex) {
  auto const newSelection = DuplicateActorNoCommit(kind, actorIndex);
  if (!newSelection) {
    return;
  }
  _selections = {*newSelection};
  _selectionAnchorRow = -1;
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
}

void MochiPrefabEditor::DuplicateSelection(bool commitUndo) {
  // Duplicating appends to the end of each array, so earlier indices stay valid as we iterate
  // _selections directly (DuplicateActorNoCommit only appends to the prefab arrays; _selections is
  // reassigned only after the loop). Multiple links of one articulated actor map to the same actor,
  // so duplicate each (kind, actor) at most once.
  std::set<std::pair<Selection::Kind, int>> duplicated;
  std::vector<Selection> copies;
  for (Selection const& sel : _selections) {
    Selection::Kind const key =
        sel.kind == Selection::Kind::ArticulatedLink ? Selection::Kind::ArticulatedActor : sel.kind;
    if (!duplicated.insert({key, sel.actorIndex}).second) {
      continue;
    }
    if (auto const copy = DuplicateActorNoCommit(sel.kind, sel.actorIndex)) {
      copies.push_back(*copy);
    }
  }
  if (copies.empty()) {
    return;
  }
  _selections = std::move(copies);
  _selectionAnchorRow = -1;
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  if (commitUndo) {
    GetUndoStack().PushNow();
  } else {
    GetUndoStack().MarkEdited();
  }
}

bool MochiPrefabEditor::DeleteSelection() {
  if (_selections.empty()) {
    return false;
  }
  auto& prefab = _prefabAsset->GetPrefab();

  // Single articulated-link selection: delete just that link (and its descendants), or the whole
  // articulation when the root link is selected. Mirrors the per-link delete in the hierarchy.
  if (_selections.size() == 1 && _selections.back().kind == Selection::Kind::ArticulatedLink) {
    Selection const sel = _selections.back();
    auto& articulated = prefab.actors.articulated;
    bool deleted = false;
    if (sel.actorIndex >= 0 && sel.actorIndex < static_cast<int>(articulated.size())) {
      auto& actor = articulated[sel.actorIndex];
      if (sel.linkIndex > 0 && sel.linkIndex < static_cast<int>(actor.links.size()) &&
          actor.links[sel.linkIndex].parentLink != superdex::robotics::kIndexNone) {
        RemoveArticulatedLinkAndDescendants(actor, sel.linkIndex);
        deleted = true;
      } else if (sel.linkIndex >= 0 && sel.linkIndex < static_cast<int>(actor.links.size())) {
        articulated.erase(articulated.begin() + sel.actorIndex);
        deleted = true;
      }
    }
    if (!deleted) {
      return false;
    }
    _viewport->SetSelectedSceneObjects({});
    _selections.clear();
    _selectionAnchorRow = -1;
    RestagePrefab();
    _prefabAsset->SetDirty(true);
    GetUndoStack().PushNow();
    return true;
  }

  // Otherwise delete every selected top-level actor. Gather per-array indices and erase in
  // descending order so earlier erases don't invalidate later indices.
  std::vector<int> rigidIdx, softIdx, articulatedIdx, nestedIdx;
  for (Selection const& sel : _selections) {
    switch (sel.kind) {
      case Selection::Kind::Rigid:
        rigidIdx.push_back(sel.actorIndex);
        break;
      case Selection::Kind::Soft:
        softIdx.push_back(sel.actorIndex);
        break;
      case Selection::Kind::ArticulatedActor:
      case Selection::Kind::ArticulatedLink:
        articulatedIdx.push_back(sel.actorIndex);
        break;
      case Selection::Kind::NestedPrefab:
        nestedIdx.push_back(sel.actorIndex);
        break;
      case Selection::Kind::None:
        break;
    }
  }
  if (rigidIdx.empty() && softIdx.empty() && articulatedIdx.empty() && nestedIdx.empty()) {
    return false;
  }
  auto eraseDescending = [](auto& array, std::vector<int>& indices) {
    std::sort(indices.begin(), indices.end(), std::greater<int>());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    for (int index : indices) {
      if (index >= 0 && index < static_cast<int>(array.size())) {
        array.erase(array.begin() + index);
      }
    }
  };
  eraseDescending(prefab.actors.rigid, rigidIdx);
  eraseDescending(prefab.actors.soft, softIdx);
  eraseDescending(prefab.actors.articulated, articulatedIdx);
  eraseDescending(prefab.prefabs, nestedIdx);

  _viewport->SetSelectedSceneObjects({});
  _selections.clear();
  _selectionAnchorRow = -1;
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
  return true;
}

void MochiPrefabEditor::PasteFromClipboard() {
  char const* text = ImGui::GetClipboardText();
  if (!text) {
    return;
  }
  std::string const clip = text;
  if (clip.rfind(kClipboardPrefix, 0) != 0) {
    return; // Not our clipboard format.
  }
  auto& prefab = _prefabAsset->GetPrefab();
  auto const flags = SReflect::DeserializeFlags::WarnIfExtraneousFields;

  // Parse one tagged block, append the actor, and return its new selection (None if invalid).
  auto parseBlock = [&](std::string const& block) -> Selection {
    if (block.rfind(kClipboardPrefix, 0) != 0) {
      return {};
    }
    auto const nl = block.find('\n');
    if (nl == std::string::npos) {
      return {};
    }
    std::string const tag = block.substr(kClipboardPrefix.size(), nl - kClipboardPrefix.size());
    std::string const json = block.substr(nl + 1);
    if (tag == "rigid") {
      mochi::prefab::RigidActorPrefab actor;
      if (!SReflect::FromJsonString(actor, json, flags)) {
        return {};
      }
      actor.name = mochi::DynamicString{
          MakeUniquePrefabActorName(actor.name.empty() ? "Rigid" : std::string(actor.name))};
      prefab.actors.rigid.push_back(std::move(actor));
      return {Selection::Kind::Rigid, static_cast<int>(prefab.actors.rigid.size()) - 1, -1};
    }
    if (tag == "soft") {
      mochi::prefab::SoftActorPrefab actor;
      if (!SReflect::FromJsonString(actor, json, flags)) {
        return {};
      }
      actor.name = mochi::DynamicString{
          MakeUniquePrefabActorName(actor.name.empty() ? "Soft" : std::string(actor.name))};
      prefab.actors.soft.push_back(std::move(actor));
      return {Selection::Kind::Soft, static_cast<int>(prefab.actors.soft.size()) - 1, -1};
    }
    if (tag == "articulated") {
      mochi::prefab::ArticulatedActorPrefab actor;
      if (!SReflect::FromJsonString(actor, json, flags)) {
        return {};
      }
      actor.name = mochi::DynamicString{
          MakeUniquePrefabActorName(actor.name.empty() ? "Articulated" : std::string(actor.name))};
      prefab.actors.articulated.push_back(std::move(actor));
      return {
          Selection::Kind::ArticulatedActor,
          static_cast<int>(prefab.actors.articulated.size()) - 1,
          -1};
    }
    if (tag == "nested") {
      mochi::prefab::PrefabReference ref;
      if (!SReflect::FromJsonString(ref, json, flags)) {
        return {};
      }
      ref.name = mochi::DynamicString{
          MakeUniquePrefabActorName(ref.name.empty() ? "Nested" : std::string(ref.name))};
      ref.prefab = nullptr;
      prefab.prefabs.push_back(std::move(ref));
      int const p = static_cast<int>(prefab.prefabs.size()) - 1;
      if (!prefab.prefabs[p].path.empty()) {
        // Preload the referenced prefab (and its models) so it renders immediately.
        _studio->GetAssetManager().LoadMochiPrefabAsset(
            mochi::Path{std::string(prefab.prefabs[p].path)});
      }
      return {Selection::Kind::NestedPrefab, p, -1};
    }
    return {};
  };

  // Split the clipboard into blocks (a single copy has one block, a group copy several) and paste
  // each, selecting all successfully pasted actors.
  std::vector<Selection> pasted;
  size_t start = 0;
  while (start <= clip.size()) {
    size_t const sep = clip.find(kClipboardSeparator, start);
    std::string const block =
        clip.substr(start, sep == std::string::npos ? std::string::npos : sep - start);
    Selection const sel = parseBlock(block);
    if (sel.kind != Selection::Kind::None) {
      pasted.push_back(sel);
    }
    if (sep == std::string::npos) {
      break;
    }
    start = sep + kClipboardSeparator.size();
  }
  if (pasted.empty()) {
    return;
  }
  _selections = std::move(pasted);
  _selectionAnchorRow = -1;
  // Keep newly-referenced assets (nested prefab / models) loaded.
  _studio->GetAssetManager().ResyncReferencer(_prefabAsset);
  RestagePrefab();
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
}

void MochiPrefabEditor::ShowPrefabDetailsWindow(bool* open) {
  ImGui::Begin("Prefab Details", open);

  // Sticky large name header: rendered before the scrolling child so it stays fixed at the top.
  ImGui::PushFont(_studio->GetFont("Roboto Bold Large"));
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(_prefabAsset->GetName().c_str());
  ImGui::PopFont();
  ImGui::Separator();

  ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32_BLACK_TRANS);
  ImGui::BeginChild("PrefabDetailsScroll");
  // Editing a property mutates the prefab and restages, which conflicts with the running
  // simulation; keep the details viewable but read-only while simulating.
  ImGui::BeginDisabled(_mochiScene.IsSimulating());
  ShowScenePrefabDetails();
  ImGui::EndDisabled();
  ImGui::EndChild();
  ImGui::PopStyleColor(); // ImGuiCol_ChildBg
  ImGui::End();
}

void MochiPrefabEditor::ShowActorDetailsWindow(bool* open) {
  auto const& prefab = _prefabAsset->GetPrefab();

  ImGui::Begin("Actor Details", open);

  // Sticky large name header: rendered before the scrolling child so it stays fixed at the top.
  std::string const headerName = SelectedActorDisplayName();
  if (!headerName.empty()) {
    ImGui::PushFont(_studio->GetFont("Roboto Bold Large"));
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(headerName.c_str());
    ImGui::PopFont();
    ImGui::Separator();
  }

  ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32_BLACK_TRANS);
  ImGui::BeginChild("ActorDetailsScroll");
  // Editing a property mutates the prefab and restages, which conflicts with the running
  // simulation; keep the details viewable but read-only while simulating.
  ImGui::BeginDisabled(_mochiScene.IsSimulating());
  if (_selections.size() > 1) {
    // Multi-actor property editing is not supported yet; the gizmo still transforms the group.
    ImGui::TextDisabled("Multiple Actors Selected");
    ImGui::TextDisabled("Per-actor property editing is unavailable for multiple selections.");
  } else if (Selection const* const primary = PrimarySelection()) {
    switch (primary->kind) {
      case Selection::Kind::Rigid:
        if (primary->actorIndex >= 0 &&
            primary->actorIndex < static_cast<int>(prefab.actors.rigid.size())) {
          ShowRigidActorDetails(primary->actorIndex);
        }
        break;
      case Selection::Kind::Soft:
        if (primary->actorIndex >= 0 &&
            primary->actorIndex < static_cast<int>(prefab.actors.soft.size())) {
          ShowSoftActorDetails(primary->actorIndex);
        }
        break;
      case Selection::Kind::ArticulatedActor:
        if (primary->actorIndex >= 0 &&
            primary->actorIndex < static_cast<int>(prefab.actors.articulated.size())) {
          ShowArticulatedActorDetails(primary->actorIndex);
        }
        break;
      case Selection::Kind::ArticulatedLink:
        if (primary->actorIndex >= 0 &&
            primary->actorIndex < static_cast<int>(prefab.actors.articulated.size())) {
          ShowArticulatedLinkDetails(primary->actorIndex, primary->linkIndex);
        }
        break;
      case Selection::Kind::NestedPrefab:
        if (primary->actorIndex >= 0 &&
            primary->actorIndex < static_cast<int>(prefab.prefabs.size())) {
          ShowNestedPrefabDetails(primary->actorIndex);
        }
        break;
      case Selection::Kind::None:
        ImGui::TextDisabled("Select an actor to edit its properties.");
        break;
    }
  } else {
    ImGui::TextDisabled("Select an actor to edit its properties.");
  }
  ImGui::EndDisabled();
  ImGui::EndChild();
  ImGui::PopStyleColor(); // ImGuiCol_ChildBg
  ImGui::End();
}

std::string MochiPrefabEditor::SelectedActorDisplayName() const {
  if (_selections.size() > 1) {
    return "Multiple Actors Selected";
  }
  Selection const* const primary = PrimarySelection();
  if (!primary) {
    return {};
  }
  auto const& prefab = _prefabAsset->GetPrefab();
  auto const& rigid = prefab.actors.rigid;
  auto const& soft = prefab.actors.soft;
  auto const& articulated = prefab.actors.articulated;
  switch (primary->kind) {
    case Selection::Kind::None:
      return {};
    case Selection::Kind::Rigid:
      if (primary->actorIndex < 0 || primary->actorIndex >= static_cast<int>(rigid.size())) {
        return {};
      }
      return FallbackName(kRigidPrefix, rigid[primary->actorIndex].name, primary->actorIndex);
    case Selection::Kind::Soft:
      if (primary->actorIndex < 0 || primary->actorIndex >= static_cast<int>(soft.size())) {
        return {};
      }
      return FallbackName(kSoftPrefix, soft[primary->actorIndex].name, primary->actorIndex);
    case Selection::Kind::ArticulatedActor:
    case Selection::Kind::ArticulatedLink: {
      if (primary->actorIndex < 0 || primary->actorIndex >= static_cast<int>(articulated.size())) {
        return {};
      }
      auto const& actor = articulated[primary->actorIndex];
      std::string actorName = FallbackName(kArticulatedPrefix, actor.name, primary->actorIndex);
      if (primary->kind == Selection::Kind::ArticulatedActor) {
        return actorName;
      }
      if (primary->linkIndex < 0 || primary->linkIndex >= static_cast<int>(actor.links.size())) {
        return actorName;
      }
      std::string const linkName =
          FallbackName(kLinkPrefix, actor.links[primary->linkIndex].name, primary->linkIndex);
      return actorName + "/" + linkName;
    }
    case Selection::Kind::NestedPrefab: {
      auto const& prefabs = prefab.prefabs;
      if (primary->actorIndex < 0 || primary->actorIndex >= static_cast<int>(prefabs.size())) {
        return {};
      }
      return FallbackName(kNestedPrefix, prefabs[primary->actorIndex].name, primary->actorIndex);
    }
  }
  return {};
}

void MochiPrefabEditor::ShowScenePrefabDetails() {
  auto& prefab = _prefabAsset->GetPrefab();
  bool changed = false;

  // A prefab is a "scene prefab" when it carries scene settings. Toggling the checkbox populates or
  // clears the optional `scene` field. Gravity and solver params are modelled as optional so a
  // prefab can omit them, but a scene prefab authored here always defines both -- there is no
  // reason to declare a scene section and leave it empty -- so they are shown as plain fields.
  bool isScenePrefab = prefab.scene.has_value();
  if (ImGui::Checkbox("Scene Prefab", &isScenePrefab)) {
    // The stash lives on the asset, so it survives closing this editor for as long as the unsaved
    // edit it belongs to does.
    auto& stashed = _prefabAsset->GetStashedSceneParams();
    if (!isScenePrefab) {
      stashed = std::exchange(prefab.scene, std::nullopt);
    } else if (stashed.has_value()) {
      prefab.scene = std::exchange(stashed, std::nullopt);
    } else {
      mochi::prefab::SceneParams params;
      auto const& physics = _studio->GetAppSettings().physics;
      params.gravity = physics.scene.gravity;
      params.solver = physics.solver;
      prefab.scene = std::move(params);
    }
    changed = true;
  }
  if (prefab.scene.has_value()) {
    // Scene and Solver are siblings here, in the app Settings window and in each editor's Physics
    // Settings window; keep the three in step. A field a prefab authored elsewhere left out shows
    // the value that would be used, and is only written once edited.
    auto& scene = *prefab.scene;
    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Indent();
      changed |= ImGui::InputText("Description", &scene.description);
      mochi::Real3 gravity =
          scene.gravity.value_or(_studio->GetAppSettings().physics.scene.gravity);
      if (ImGui::DragRealXYZ("Gravity", gravity, 0.01f, 0.0f, 0.0f, "%.4f m/s^2")) {
        scene.gravity = gravity;
        changed = true;
      }
      ImGui::Unindent();
    }
    if (ImGui::CollapsingHeader("Solver")) {
      ImGui::Indent();
      mochi::SolverParams solver = scene.solver.value_or(_studio->GetAppSettings().physics.solver);
      if (ImGui::SimpleReflectionStruct(solver)) {
        scene.solver = solver;
        changed = true;
      }
      ImGui::Unindent();
    }
  }

  if (changed) {
    // Scene settings do not affect the staged render objects, so no restage is needed.
    _prefabAsset->SetDirty(true);
    GetUndoStack().MarkEdited();
  }
}

// Draw a combo over the full-path staged actor names for a contact-filter member. Returns true if
// `value` changed.
static bool ContactActorCombo(
    char const* id,
    mochi::DynamicString& value,
    std::vector<std::string> const& actorNames) {
  bool changed = false;
  // Flag entries that reference an actor no longer present in the prefab, using the same red
  // FrameBg as name-conflict highlighting.
  std::string_view const current(value);
  bool const missing = !current.empty() &&
      std::find(actorNames.begin(), actorNames.end(), current) == actorNames.end();
  if (missing) {
    // The combo frame uses FrameBg; its arrow button uses Button. Push both so the whole widget is
    // red (same highlight as a name conflict).
    ImGui::PushStyleColor(ImGuiCol_FrameBg, kNameConflictColor);
    ImGui::PushStyleColor(ImGuiCol_Button, kNameConflictColor);
  }
  ImGui::SetNextItemWidth(-1.0f);
  bool const open = ImGui::BeginCombo(id, value.c_str());
  if (missing) {
    ImGui::PopStyleColor(2);
  }
  if (open) {
    for (auto const& name : actorNames) {
      bool const selected = std::string_view(value) == name;
      if (ImGui::Selectable(name.c_str(), selected)) {
        value = mochi::DynamicString{name};
        changed = true;
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  return changed;
}

// Placeholder used whenever a layer name would otherwise be empty. Empty layer names are invalid
// input to the physics contact filter, so they are never allowed.
static constexpr char const* kDefaultLayerName = "Layer";

// Draw a free-text input for a contact-filter layer member. Empty values are disallowed: if the
// user commits an empty string (Enter or focus loss), it is reset to a generic placeholder. Returns
// true if `value` changed.
static bool ContactLayerInput(char const* id, mochi::DynamicString& value) {
  ImGui::SetNextItemWidth(-1.0f);
  bool changed = ImGui::InputText(id, &value, ImGuiInputTextFlags_CharsNoBlank);
  if (ImGui::IsItemDeactivatedAfterEdit() && value.empty()) {
    value = mochi::DynamicString{kDefaultLayerName};
    changed = true;
  }
  return changed;
}

// Render an editable contact-filter table for one category (actors or layers). `EntryT` is
// ActorContactEntry or LayerContactEntry; `getMembers` returns the entry's 2-element name array and
// `renderCell(id, member)` draws the A/B editor widget for one member. New rows are pre-populated
// with `defaultA`/`defaultB` so entries are never created with empty (invalid) names. Symmetric and
// asymmetric entries share one table; the Type column moves an entry between the two arrays.
// Structural edits (remove / retype / add) are deferred until after the table is rendered. Returns
// true if the underlying filter data changed.
template <typename EntryT, typename GetMembers, typename RenderCell>
bool ShowContactEntryTable(
    char const* tableId,
    char const* colAName,
    char const* colBName,
    std::string_view defaultA,
    std::string_view defaultB,
    std::optional<mochi::DynamicArray<EntryT>>& symmetric,
    std::optional<mochi::DynamicArray<EntryT>>& asymmetric,
    GetMembers getMembers,
    RenderCell renderCell) {
  using mochi::DynamicArray;
  using mochi::DynamicString;
  bool changed = false;

  enum class Op { None, Remove, ToSymmetric, ToAsymmetric };
  Op op = Op::None;
  bool opFromSymmetric = false;
  int opIndex = -1;

  constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders;
  if (ImGui::BeginTable(tableId, 5, kFlags)) {
    // A/B use fixed-weight stretch (not content-proportional) so the fill-width combos/inputs don't
    // feed back into the column sizing and shrink it every frame.
    ImGui::TableSetupColumn(colAName, ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn(colBName, ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 110.0f);
    ImGui::TableSetupColumn("Enable", ImGuiTableColumnFlags_WidthFixed, 55.0f);
    ImGui::TableSetupColumn("##remove", ImGuiTableColumnFlags_WidthFixed, 30.0f);
    ImGui::TableHeadersRow();

    // Blend the cell widgets (combos, inputs, checkbox, icon button) into the table, matching the
    // URDF importer.
    ImGui::PushFramelessWidgetStyle();

    auto renderGroup = [&](std::optional<DynamicArray<EntryT>>& group, bool isSymmetric) {
      if (!group.has_value()) {
        return;
      }
      for (int i = 0; i < static_cast<int>(group->size()); ++i) {
        EntryT& entry = (*group)[i];
        auto& members = getMembers(entry);
        while (members.size() < 2) {
          members.push_back(DynamicString{});
        }
        ImGui::TableNextRow();
        ImGui::PushID((isSymmetric ? 1 : -1) * (i + 1));

        ImGui::TableNextColumn();
        changed |= renderCell("##a", members[0]);
        ImGui::TableNextColumn();
        changed |= renderCell("##b", members[1]);

        ImGui::TableNextColumn();
        int typeIdx = isSymmetric ? 1 : 0;
        char const* const types[] = {"Asymmetric", "Symmetric"};
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::Combo("##type", &typeIdx, types, 2) && (typeIdx == 1) != isSymmetric) {
          op = isSymmetric ? Op::ToAsymmetric : Op::ToSymmetric;
          opFromSymmetric = isSymmetric;
          opIndex = i;
        }

        ImGui::TableNextColumn();
        // Center the checkbox within the fixed-width Enable column.
        float const enableCellWidth = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX(
            ImGui::GetCursorPosX() + (enableCellWidth - ImGui::GetFrameHeight()) * 0.5f);
        changed |= ImGui::Checkbox("##enable", &entry.enable);

        ImGui::TableNextColumn();
        if (ImGui::Button(ICON_FA_TRASH)) {
          op = Op::Remove;
          opFromSymmetric = isSymmetric;
          opIndex = i;
        }
        ImGui::PopID();
      }
    };
    renderGroup(symmetric, true);
    renderGroup(asymmetric, false);
    ImGui::PopFramelessWidgetStyle();
    ImGui::EndTable();
  }

  // Apply the (single) deferred structural edit.
  std::optional<DynamicArray<EntryT>>& fromArr = opFromSymmetric ? symmetric : asymmetric;
  bool const validIndex =
      fromArr.has_value() && opIndex >= 0 && opIndex < static_cast<int>(fromArr->size());
  if (op == Op::Remove && validIndex) {
    fromArr->erase(fromArr->begin() + opIndex);
    changed = true;
  } else if ((op == Op::ToSymmetric || op == Op::ToAsymmetric) && validIndex) {
    std::optional<DynamicArray<EntryT>>& toArr = (op == Op::ToSymmetric) ? symmetric : asymmetric;
    if (!toArr.has_value()) {
      toArr = DynamicArray<EntryT>{};
    }
    toArr->push_back(std::move((*fromArr)[opIndex]));
    fromArr->erase(fromArr->begin() + opIndex);
    changed = true;
  }

  // New rows default to asymmetric, pre-populated with member names so entries are never created
  // empty (empty names are invalid input to the physics contact filter).
  if (ImGui::Button("Add", ImVec2(120.0f, 0.0f))) {
    if (!asymmetric.has_value()) {
      asymmetric = DynamicArray<EntryT>{};
    }
    EntryT entry{};
    auto& members = getMembers(entry);
    members.clear();
    members.push_back(DynamicString{defaultA});
    members.push_back(DynamicString{defaultB});
    asymmetric->push_back(std::move(entry));
    changed = true;
  }

  return changed;
}

void MochiPrefabEditor::ShowContactFilterWindow(bool* open) {
  ImGui::Begin("Contact Filter", open);
  auto& prefab = _prefabAsset->GetPrefab();

  // A ContactFilter is optional; offer to create one when absent.
  if (!prefab.contactFilter.has_value()) {
    ImGui::TextDisabled("This prefab has no contact filter.");
    if (ImGui::Button("Add Contact Filter")) {
      prefab.contactFilter = mochi::prefab::ContactFilter{};
      _prefabAsset->SetDirty(true);
      GetUndoStack().PushNow();
    }
    ImGui::End();
    return;
  }
  auto& filter = *prefab.contactFilter;

  // Full-path staged actor names (same list as the Render Scene Stage) for the actor combos.
  std::vector<std::string> actorNames;
  actorNames.reserve(_stage.GetActors().size());
  for (auto const& staged : _stage.GetActors()) {
    actorNames.push_back(staged.name);
  }

  bool changed = false;
  if (ImGui::BeginTabBar("##ContactFilterTabs")) {
    if (ImGui::BeginTabItem("Actors")) {
      // Default new rows to real staged actors so they aren't created empty/invalid.
      std::string_view const defaultActorA =
          actorNames.empty() ? std::string_view{} : std::string_view(actorNames.front());
      std::string_view const defaultActorB =
          actorNames.size() > 1 ? std::string_view(actorNames[1]) : defaultActorA;
      changed |= ShowContactEntryTable(
          "##ActorContacts",
          "Actor A",
          "Actor B",
          defaultActorA,
          defaultActorB,
          filter.actorContactSymmetric,
          filter.actorContactAsymmetric,
          [](mochi::prefab::ActorContactEntry& e) -> mochi::DynamicArray<mochi::DynamicString>& {
            return e.actors;
          },
          [&](char const* id, mochi::DynamicString& v) {
            return ContactActorCombo(id, v, actorNames);
          });
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Layers")) {
      changed |= ShowContactEntryTable(
          "##LayerContacts",
          "Layer A",
          "Layer B",
          kDefaultLayerName,
          kDefaultLayerName,
          filter.layerContactSymmetric,
          filter.layerContactAsymmetric,
          [](mochi::prefab::LayerContactEntry& e) -> mochi::DynamicArray<mochi::DynamicString>& {
            return e.layers;
          },
          [](char const* id, mochi::DynamicString& v) { return ContactLayerInput(id, v); });
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }

  // Contact-filter edits do not affect staged render objects, so no restage is needed.
  if (changed) {
    _prefabAsset->SetDirty(true);
    GetUndoStack().MarkEdited();
  }
  ImGui::End();
}

// Custom material editor for a soft actor: a constitutive-model dropdown that reveals only the
// selected model's parameters, plus the shared density. Returns true if anything changed.
static bool ShowSoftMaterialEditor(mochi::SoftMaterialParams& material) {
  bool changed = false;
  ImGui::SeparatorText("Material");
  changed |= ImGui::SimpleReflectionEnum("Model", material.type);
  switch (material.type) {
    case mochi::SoftMaterialType::NeoHookean:
      changed |= ImGui::SimpleReflectionStruct(material.neoHookean);
      break;
    case mochi::SoftMaterialType::StVenantKirchhoff:
      changed |= ImGui::SimpleReflectionStruct(material.stVenantKirchhoff);
      break;
    case mochi::SoftMaterialType::LinearElastic:
      changed |= ImGui::SimpleReflectionStruct(material.linearElastic);
      break;
    case mochi::SoftMaterialType::ActiveNeoHookean:
      changed |= ImGui::SimpleReflectionStruct(material.activeNeoHookean);
      break;
    case mochi::SoftMaterialType::ActiveShapeTargetingArap:
      changed |= ImGui::SimpleReflectionStruct(material.activeShapeTargetingArap);
      break;
    case mochi::SoftMaterialType::Arap:
      changed |= ImGui::SimpleReflectionStruct(material.arap);
      break;
    case mochi::SoftMaterialType::Count:
      break;
  }
  changed |= ImGui::DragReal("Density", &material.density, 1.0f, 0.0f, 0.0f, "%.2f");
  return changed;
}

void MochiPrefabEditor::ShowRigidActorDetails(int rigidIndex) {
  auto* actor = &_prefabAsset->GetPrefab().actors.rigid[rigidIndex];
  AssetManager const& assetManager = _studio->GetAssetManager();

  // Track geometry/pose/model edits separately from cheap edits (layer, flags, inertia, contact):
  // only the former need a restage, so dragging e.g. a friction slider does not restage every
  // frame.
  bool changed = false;
  bool restage = false;

  ImGui::SeparatorText("General");
  ImGui::LabelText("Type", "%s", "Rigid");
  if (ImGui::NameInputWithCollisionCheck(
          "Name",
          actor->name,
          TopLevelNameCollides(FallbackName(kRigidPrefix, actor->name, rigidIndex)))) {
    changed = true;
    // The staged actor is keyed by name (used for simulation sync), so a rename must restage.
    restage = true;
  }
  changed |= ImGui::InputText("Layer", &actor->layer, ImGuiInputTextFlags_CharsNoBlank);
  changed |= ImGui::Checkbox("Static", &actor->isStatic);
  ImGui::SameLine();
  changed |= ImGui::Checkbox("Has Gravity", &actor->hasGravity);

  ImGui::SeparatorText("Transform");
  if (ImGui::DragTransformRT("World From Local", actor->rotation, actor->translation)) {
    changed = true;
    restage = true;
  }

  changed |= ImGui::InertialProperties(
      actor->density, actor->mass, actor->centerOfMass, actor->momentOfInertia);

  // Both model editors edit geometry-affecting fields: the Mochi Model editor edits the shape
  // path/scale/offset directly, and the Render Model editor is cross-wired (link transforms) onto
  // the physics scale/offset. The ModelEditor `modelChanged` out-param only fires on a path change,
  // so a scale/transform drag would otherwise leave the baked physics shape stale; treat a true
  // return from either editor as requiring shape invalidation + restage.
  bool modelChanged = false;
  bool const mochiModelEdited = ImGui::ModelEditor(
      "Mochi Model",
      _studio,
      AssetType::MochiModel,
      actor->shapeFile,
      actor->scale,
      actor->shapeRotation,
      actor->shapeTranslation,
      &actor->renderModelScale,
      &actor->renderModelRotation,
      &actor->renderModelTranslation,
      assetManager,
      true,
      modelChanged);

  bool const renderModelEdited = ImGui::ModelEditor(
      "Render Model",
      _studio,
      AssetType::RenderModel,
      actor->renderModelFile,
      actor->renderModelScale,
      actor->renderModelRotation,
      actor->renderModelTranslation,
      &actor->scale,
      &actor->shapeRotation,
      &actor->shapeTranslation,
      assetManager,
      true,
      modelChanged);

  changed |= ImGui::CollisionContact(
      "Collision / Contact",
      actor->colliderType,
      actor->contact,
      actor->boundaryElementType,
      &actor->boundarySubsampling);

  if (mochiModelEdited || renderModelEdited) {
    // A model/scale/transform edit changes the staged geometry and the baked physics shape.
    actor->shape = {};
    changed = true;
    restage = true;
  }

  if (changed) {
    if (restage) {
      RestagePrefab();
    }
    _prefabAsset->SetDirty(true);
    GetUndoStack().MarkEdited();
  }
}

void MochiPrefabEditor::ShowSoftActorDetails(int softIndex) {
  auto* actor = &_prefabAsset->GetPrefab().actors.soft[softIndex];
  AssetManager const& assetManager = _studio->GetAssetManager();

  // Track geometry/pose edits separately from cheap edits (material, contact, flags): only the
  // former need a (relatively expensive) restage that rebuilds the soft surface mesh.
  bool changed = false;
  bool restage = false;
  bool modelChanged = false;

  ImGui::SeparatorText("General");
  ImGui::LabelText("Type", "%s", "Soft");
  if (ImGui::NameInputWithCollisionCheck(
          "Name",
          actor->name,
          TopLevelNameCollides(FallbackName(kSoftPrefix, actor->name, softIndex)))) {
    changed = true;
    // The staged actor is keyed by name (used for simulation sync), so a rename must restage.
    restage = true;
  }
  changed |= ImGui::InputText("Layer", &actor->layer, ImGuiInputTextFlags_CharsNoBlank);
  changed |= ImGui::Checkbox("Has Gravity", &actor->hasGravity);
  ImGui::SameLine();
  changed |= ImGui::Checkbox("Has Inertia", &actor->hasInertia);
  ImGui::SameLine();
  changed |= ImGui::Checkbox("Has Stress", &actor->hasStress);
  changed |= ImGui::Checkbox("Use Recentering", &actor->useRecentering);

  ImGui::SeparatorText("Transform");
  if (ImGui::DragTransformRT("World From Local", actor->rotation, actor->translation)) {
    changed = true;
    restage = true;
  }

  // Shape (Mochi model). The actor scale and shape offset are baked into the surface geometry at
  // load, so any change here invalidates the cached shape and rebuilds the mesh.
  ImGui::SeparatorText("Shape");
  if (ImGui::AssetSlot(
          "Mochi Model", actor->shapeFile, assetManager, _studio, AssetType::MochiModel, true)) {
    modelChanged = true;
  }
  if (ImGui::DragRealXYZ("Scale", actor->scale, 0.01f, 0.0f, 0.0f, "%.4f")) {
    modelChanged = true;
  }
  if (ImGui::DragTransformRT("Shape Offset", actor->shapeRotation, actor->shapeTranslation)) {
    modelChanged = true;
  }

  changed |= ShowSoftMaterialEditor(actor->material);

  ImGui::SeparatorText("Collision / Contact");
  changed |= ImGui::SimpleReflectionEnum("Collider Type", actor->colliderType);
  changed |= ImGui::SimpleReflectionStruct(actor->contact);
  changed |= ImGui::SimpleReflectionEnum("Boundary Element Type", actor->boundaryElementType);

  if (modelChanged) {
    // Invalidate the cached shape so the stage rebuilds geometry with the new model/scale/offset.
    actor->shape = {};
    changed = true;
    restage = true;
  }

  if (changed) {
    if (restage) {
      RestagePrefab();
    }
    _prefabAsset->SetDirty(true);
    GetUndoStack().MarkEdited();
  }
}

void MochiPrefabEditor::ShowArticulatedActorDetails(int actorIndex) {
  auto& actor = _prefabAsset->GetPrefab().actors.articulated[actorIndex];
  AssetManager const& assetManager = _studio->GetAssetManager();

  // Every edit in this panel affects staged geometry/pose/model, so `restage` is set alongside
  // `changed`; the split is kept for consistency with the other detail panels.
  bool changed = false;
  bool restage = false;

  ImGui::SeparatorText("General");
  ImGui::LabelText("Type", "%s", "Articulated");
  if (ImGui::NameInputWithCollisionCheck(
          "Name",
          actor.name,
          TopLevelNameCollides(FallbackName(kArticulatedPrefix, actor.name, actorIndex)))) {
    changed = true;
    // The staged actor is keyed by name (used for simulation sync), so a rename must restage.
    restage = true;
  }

  ImGui::SeparatorText("Transform");
  if (ImGui::DragTransformRT("World From Root", actor.rotation, actor.translation)) {
    changed = true;
    restage = true;
  }
  if (ImGui::DragReal("Scale", &actor.scale, 0.01f, 0.0f, 0.0f, "%.4f")) {
    changed = true;
    restage = true;
    InvalidateArticulatedActorShapes(actor);
  }

  // Skin models (optional). The skin's Mochi shape has no per-shape transform (only a file), while
  // its render model carries its own scale/rotation/translation.
  if (actor.skin.has_value()) {
    auto& skin = *actor.skin;
    bool modelChanged = false;
    ImGui::SeparatorText("Skin");
    if (ImGui::AssetSlot(
            "Mochi Model", skin.shapeFile, assetManager, _studio, AssetType::MochiModel, true)) {
      modelChanged = true;
    }
    // The render model editor changes the staged skin render object (scale/offset/path), so any
    // true return restages; a Mochi-model path change additionally invalidates the baked skin
    // shape.
    if (ImGui::ModelEditor(
            "Render Model",
            _studio,
            AssetType::RenderModel,
            skin.renderModelFile,
            skin.renderModelScale,
            skin.renderModelRotation,
            skin.renderModelTranslation,
            nullptr,
            nullptr,
            nullptr,
            assetManager,
            true,
            modelChanged)) {
      changed = true;
      restage = true;
    }
    if (modelChanged) {
      // Invalidate the cached skin shape so the stage reloads it.
      skin.shape = {};
      changed = true;
      restage = true;
    }
  }

  if (changed) {
    if (restage) {
      RestagePrefab();
    }
    _prefabAsset->SetDirty(true);
    GetUndoStack().MarkEdited();
  }
}

void MochiPrefabEditor::ShowArticulatedLinkDetails(int actorIndex, int linkIndex) {
  auto& actor = _prefabAsset->GetPrefab().actors.articulated[actorIndex];
  if (linkIndex < 0 || linkIndex >= static_cast<int>(actor.links.size())) {
    return;
  }
  AssetManager const& assetManager = _studio->GetAssetManager();

  bool changed = false;
  bool modelChanged = false;
  int reparentNewParent = superdex::robotics::kIndexNone;

  bool const isRoot = actor.links[linkIndex].parentLink == superdex::robotics::kIndexNone;
  if (linkIndex < static_cast<int>(actor.joints.size())) {
    char const* jointLabel = isRoot ? "World Joint" : "Joint";
    if (ImGui::CollapsingHeader(jointLabel, ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::PushID("Joint");
      changed |= ImGui::ArticulatedJointEditor(actor.joints[linkIndex], isRoot);
      ImGui::PopID();
    }
  }
  if (!isRoot) {
    int const numLinks = static_cast<int>(actor.links.size());
    std::vector<bool> isDescendant(numLinks, false);
    isDescendant[linkIndex] = true;
    for (int i = linkIndex + 1; i < numLinks; ++i) {
      int const p = actor.links[i].parentLink;
      if (p >= 0 && p < numLinks && isDescendant[p]) {
        isDescendant[i] = true;
      }
    }
    int const currentParent = actor.links[linkIndex].parentLink;
    char const* const previewName = (currentParent >= 0 && currentParent < numLinks)
        ? actor.links[currentParent].name.c_str()
        : "";
    if (ImGui::BeginCombo("Parent", previewName)) {
      for (int j = 0; j < numLinks; ++j) {
        if (isDescendant[j]) {
          continue;
        }
        bool const isSelected = (j == currentParent);
        if (ImGui::Selectable(actor.links[j].name.c_str(), isSelected) && j != currentParent) {
          reparentNewParent = j;
        }
        if (isSelected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
  }
  if (ImGui::CollapsingHeader("Link", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::PushID("Link");
    changed |= ImGui::ArticulatedLinkEditor(
        actor.links[linkIndex], _studio, assetManager, true, modelChanged);
    ImGui::PopID();
  }

  if (modelChanged) {
    // Invalidate the cached link shape so the stage reloads geometry with the new model/scale.
    actor.links[linkIndex].shape = {};
    changed = true;
  }

  if (changed) {
    RestagePrefab();
    _prefabAsset->SetDirty(true);
    GetUndoStack().MarkEdited();
  }
  if (reparentNewParent != superdex::robotics::kIndexNone) {
    ReparentArticulatedLink(actorIndex, linkIndex, reparentNewParent);
  }
}

void MochiPrefabEditor::ShowNestedPrefabDetails(int prefabIndex) {
  auto& ref = _prefabAsset->GetPrefab().prefabs[prefabIndex];
  AssetManager const& assetManager = _studio->GetAssetManager();

  // Every edit in this panel affects the staged reference (name map, referenced prefab, pose), so
  // `restage` is set alongside `changed`; the split is kept for consistency with the other panels.
  bool changed = false;
  bool restage = false;

  ImGui::SeparatorText("General");
  ImGui::LabelText("Type", "%s", "Nested Prefab");
  if (ImGui::NameInputWithCollisionCheck(
          "Name",
          ref.name,
          TopLevelNameCollides(FallbackName(kNestedPrefix, ref.name, prefabIndex)))) {
    changed = true;
    // The staged actor is keyed by name (used for simulation sync), so a rename must restage.
    restage = true;
  }

  ImGui::SeparatorText("Prefab");
  bool const pathChanged =
      ImGui::AssetSlot("Prefab", ref.path, assetManager, _studio, AssetType::MochiPrefab, true);
  if (pathChanged) {
    // Drop the cached nested prefab so RestagePrefab reloads the new reference from disk.
    ref.prefab = nullptr;
    if (!ref.path.empty()) {
      // Load the referenced prefab (and, transitively, its models) so staging can render it
      // immediately, mirroring the preload MochiPrefabAsset::Create performs for nested references.
      _studio->GetAssetManager().LoadMochiPrefabAsset(mochi::Path{std::string(ref.path)});
    }
    // Re-register our references so the newly-referenced prefab (and its models) stay loaded;
    // otherwise the nested reference renders nothing until the prefab is reloaded manually.
    _studio->GetAssetManager().ResyncReferencer(_prefabAsset);
    changed = true;
    restage = true;
  }

  ImGui::SeparatorText("Transform");
  if (ImGui::DragTransformRT("Transform", ref.rotation, ref.translation)) {
    changed = true;
    restage = true;
  }
  if (ImGui::DragReal("Scale", &ref.scale, 0.01f, 0.0f, 0.0f, "%.4f")) {
    changed = true;
    restage = true;
  }

  if (changed) {
    if (restage) {
      RestagePrefab();
    }
    _prefabAsset->SetDirty(true);
    GetUndoStack().MarkEdited();
  }
}

bool MochiPrefabEditor::TopLevelNameCollides(std::string const& name) const {
  // Top-level actor names share a namespace (a duplicate name collides in the staged scene's name
  // map). More than one top-level actor resolving to `name` means a conflict.
  auto const names = CollectTopLevelNames();
  return std::count(names.begin(), names.end(), name) > 1;
}

} // namespace superdex::studio
