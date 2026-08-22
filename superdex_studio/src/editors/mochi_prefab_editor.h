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
#include "simulation/mochi_async_scene.h"
#include "simulation/physics_drag_controller.h"

#include <mochi_physics/utils/mochi_prefab.h>

#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace superdex::studio {

class MochiPrefabAsset;

class MochiPrefabEditor : public AssetEditor {
 public:
  //------------------------------------------------------------------------------------------------
  // AssetEditor
  //------------------------------------------------------------------------------------------------

  MochiPrefabEditor(SuperDexStudio* studio, MochiPrefabAsset* asset);
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
  // Identifies the currently selected hierarchy item.
  struct Selection {
    enum class Kind { None, Rigid, ArticulatedActor, ArticulatedLink, NestedPrefab, Soft };
    Kind kind = Kind::None;
    int actorIndex = -1;
    int linkIndex = -1; // articulated link index (ArticulatedLink only)
  };

  // Flattened hierarchy row categories (top-level actor kinds).
  enum class RowCategory { Rigid, Soft, Articulated, NestedPrefab };

  // One flattened hierarchy row. `type` is a static display label; `category`/`index` map the row
  // back to its underlying prefab array after sorting.
  struct HierarchyRow {
    std::string name;
    char const* type;
    RowCategory category;
    int index;
  };

  // Deferred structural edits collected while rendering the hierarchy, applied after the table is
  // built (mutating the prefab arrays mid-iteration would invalidate indices).
  struct HierarchyEdits {
    int rigidToDelete = -1;
    int softToDelete = -1;
    int articulatedActorToDelete = -1;
    int nestedToDelete = -1;
    int addLinkActor = -1;
    int addLinkParent = -1;
    int delLinkActor = -1;
    int delLinkIndex = -1;
    Selection::Kind duplicateKind = Selection::Kind::None;
    int duplicateIndex = -1;
    // Group operations on the whole current multi-selection (deferred like the single-item edits).
    bool duplicateSelection = false;
    bool deleteSelection = false;
    bool createRigid = false;
    bool createSoft = false;
    bool createArticulated = false;
    bool createNested = false;
    bool pasteRequested = false;
  };

  //------------------------------------------------------------------------------------------------
  // Undo/Redo
  //------------------------------------------------------------------------------------------------

  std::string TakeUndoSnapshot() const;
  void RestoreUndoSnapshot(std::string const& json, int selectionIndex);
  int GetUndoSelectionIndex() const;

  //------------------------------------------------------------------------------------------------
  // Prefab Staging
  //------------------------------------------------------------------------------------------------

  void RestagePrefab();
  // Find the staged render object for a top-level prefab actor by source + index (and, for
  // articulated links, link index). Pass linkIndex < 0 to match any link.
  mochi_renderer::SceneObject*
  StagedObjectForSource(StagedActorSource source, int actorIndex, int linkIndex = -1) const;

  //------------------------------------------------------------------------------------------------
  // Viewport
  //------------------------------------------------------------------------------------------------

  void OnSceneSelectionChanged(std::vector<mochi_renderer::SceneObject*> const& objects);
  std::optional<std::pair<mochi::TransformRT, mochi::Real3>> GetTransformGizmoTarget() const;
  // Apply an incremental transform delta (rotation/translation about the gizmo target) and per-axis
  // scale multiplier to every selected actor. Drives both single- and multi-selection: with one
  // selection the target is the actor itself, so the delta reproduces an absolute edit.
  void OnGizmoTransformDelta(mochi::TransformRT const& delta, mochi::Real3 const& scaleMul);
  // Map a staged scene object back to the hierarchy selection it belongs to (None if unmapped).
  Selection SelectionForObject(mochi_renderer::SceneObject* object) const;
  // Map a flattened hierarchy row to its top-level actor selection.
  static Selection SelectionForRow(HierarchyRow const& row);
  // The primary (last) selection, or nullptr when nothing is selected.
  Selection const* PrimarySelection() const;
  // Whether a top-level actor row of the given kind/index is part of the current selection.
  bool IsRowSelected(Selection::Kind kind, int actorIndex) const;
  // Replace the whole selection with a single item (or clear it for Kind::None), reset the range
  // anchor, and rebind the viewport's highlighted objects.
  void SetSingleSelection(Selection selection);
  // Rebind the viewport's selected/highlighted scene objects to the current `_selections`, without
  // re-entering the selection-changed callback.
  void SyncViewportSelection();
  // Handle a click on the flattened top-level row `rowIndex` selecting `sel`, honoring Ctrl
  // (toggle) and Shift (contiguous range from the anchor) modifiers; plain click replaces.
  void HandleTopLevelRowClick(int rowIndex, Selection const& sel);

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

  void ShowPrefabHierarchyWindow(bool* open);
  // Recursively render an articulated actor's link/joint sub-tree. `children` maps each link to its
  // child link indices; `ancestorsToOpen` holds links to force-open to reveal the focused link.
  // Deferred structural edits (add/delete link) are written into `edits`.
  void RenderArticulatedLinkTree(
      int actorIndex,
      mochi::prefab::ArticulatedActorPrefab const& actor,
      std::vector<std::vector<int>> const& children,
      std::unordered_set<int> const& ancestorsToOpen,
      int linkIndex,
      bool simulating,
      HierarchyEdits& edits);
  // Highlight a staged render object (if any) with the hierarchy hover color.
  void HighlightStagedObject(mochi_renderer::SceneObject* object);
  void ShowPrefabDetailsWindow(bool* open);
  void ShowActorDetailsWindow(bool* open);
  void ShowContactFilterWindow(bool* open);
  void ShowScenePrefabDetails();
  void ShowRigidActorDetails(int rigidIndex);
  void ShowSoftActorDetails(int softIndex);
  void ShowArticulatedActorDetails(int actorIndex);
  void ShowArticulatedLinkDetails(int actorIndex, int linkIndex);
  void ShowNestedPrefabDetails(int prefabIndex);

  //------------------------------------------------------------------------------------------------
  // Structural edits
  //------------------------------------------------------------------------------------------------

  static int AddArticulatedLink(
      mochi::prefab::ArticulatedActorPrefab& actor,
      int parentIdx,
      std::string_view baseName);
  static void RemoveArticulatedLinkAndDescendants(
      mochi::prefab::ArticulatedActorPrefab& actor,
      int linkIdx);
  void ReparentArticulatedLink(
      int actorIndex,
      int linkIdx,
      int newParentIdx,
      bool preserveWorldTransform = true);

  void CreateRigidActor();
  void CreateSoftActor();
  void CreateArticulatedActor();
  void CreateNestedPrefab();

  //------------------------------------------------------------------------------------------------
  // Clipboard (copy / paste / duplicate)
  //------------------------------------------------------------------------------------------------

  // Build the tagged-JSON clipboard block for one top-level actor ("<prefix><tag>\n<json>"), or an
  // empty string for an invalid/uncopyable selection. Shared by the single and group copy paths.
  std::string BuildActorClipboardBlock(Selection::Kind kind, int actorIndex) const;
  // Serialize the given top-level actor to the system clipboard as tagged reflected JSON. An
  // ArticulatedLink selection copies its owning articulated actor. No-op for None.
  void CopyActorToClipboard(Selection::Kind kind, int actorIndex) const;
  // Serialize every currently-selected top-level actor to the system clipboard (multi-actor
  // envelope; a single selection uses the same format as CopyActorToClipboard so it round-trips).
  void CopySelectionToClipboard() const;
  // Append a deep copy of the given top-level actor (with a unique name) and select it.
  void DuplicateActor(Selection::Kind kind, int actorIndex);
  // Append a deep copy of the given top-level actor (with a unique name) without restaging or
  // pushing undo; returns the new copy's selection, or None if the source was invalid. Shared by
  // DuplicateActor and DuplicateSelection.
  std::optional<Selection> DuplicateActorNoCommit(Selection::Kind kind, int actorIndex);
  // Duplicate every currently-selected top-level actor and select the copies. Pushes one undo entry
  // when commitUndo is true (discrete Ctrl+D / context-menu duplicate). Pass false for the alt+drag
  // path: it only marks the prefab edited so the drag's release-time snapshot merges the duplicate
  // and the move into a single undo entry.
  void DuplicateSelection(bool commitUndo = true);
  // Delete every currently-selected item (top-level actors, or a single articulated link) in one
  // undo entry. Returns true if anything was deleted.
  bool DeleteSelection();
  // Append a top-level actor parsed from clipboard JSON (if it holds a valid tagged actor).
  void PasteFromClipboard();

  //------------------------------------------------------------------------------------------------
  // Utils
  //------------------------------------------------------------------------------------------------

  std::string SelectedActorDisplayName() const;
  std::string MakeUniquePrefabActorName(std::string_view base) const;
  // Resolve every top-level actor's display name (its own name, or a "<Kind>_<index>" fallback when
  // unnamed) in hierarchy order. Shared so name-uniqueness and collision checks agree.
  std::vector<std::string> CollectTopLevelNames() const;
  bool TopLevelNameCollides(std::string const& name) const;

 private:
  // target asset
  MochiPrefabAsset* _prefabAsset = nullptr;
  // render scene state
  std::unique_ptr<Viewport> _viewport;
  SceneStage _stage;
  StageType _stageType = StageType::RenderModelFallbackToMochiModel;
  // physics scene state
  MochiAsyncScene _mochiScene;
  std::vector<mochi::ActorHandle> _physicsActors;
  std::unique_ptr<PhysicsDragController> _dragController;
  struct SimData {
    std::vector<std::string> actorNames;
    std::vector<mochi::TransformRT> actorTransforms;
    std::vector<SoftMeshUpdate> softMeshUpdates;
  };
  mochi_renderer::ProducerConsumerBuffer<SimData> _simData;
  // UI state
  // Ordered multi-selection of hierarchy items; back() is the primary/active item. Top-level actor
  // rows (Rigid, Soft, Articulated actor, Nested prefab) support multi-select; articulated links
  // are always single-select.
  std::vector<Selection> _selections;
  // Flattened-row index (into `_hierarchyRows`) anchoring Shift-range selection; -1 when unset.
  int _selectionAnchorRow = -1;
  // Persistent scratch buffers for the hierarchy window, cleared and reused each frame to avoid
  // per-frame heap allocation: the flattened row list, and (per articulated actor) each link's
  // child indices.
  std::vector<HierarchyRow> _hierarchyRows;
  std::vector<std::vector<int>> _linkChildrenScratch;
  // Per-soft-actor query registration held while simulating. The two surface queries are registered
  // once in CreatePhysicsActors and cancelled on stop; `name` matches the staged actor for the
  // per-frame mesh update.
  struct SoftQueryState {
    mochi::ActorHandle actor;
    mochi::QueryHandle positions;
    mochi::QueryHandle normals;
    std::string name;
  };
  std::vector<SoftQueryState> _softQueryState;
  // One-shot flag: when set, the hierarchy expands ancestors of and scrolls to the selected
  // articulated link (set on viewport pick, consumed while rendering the tree).
  bool _forceLinkFocus = false;
};

} // namespace superdex::studio
