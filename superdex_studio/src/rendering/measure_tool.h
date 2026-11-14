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

#include "rendering/measure_geometry.h"

#include <mochi_renderer/path.h>
#include <mochi_renderer/utils.h> // mochi_renderer::MeshSection

#include <math/vec3.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mochi_renderer {
class DebugDraw;
class SceneObject;
} // namespace mochi_renderer

namespace superdex::studio {

class DebugText;
class SuperDexStudio;
class Viewport;
struct SceneStage;

// A size that stays the same on screen: `fraction` of `point`'s distance to the camera, clamped to
// [minSize, maxSize] (renderer space, meters). Sizes debug-draw markers and lines.
float ScreenScaledSize(
    filament::math::float3 point,
    filament::math::float3 cameraPosition,
    float fraction,
    float minSize,
    float maxSize);

//--------------------------------------------------------------------------------------------------
// TARGETS
//--------------------------------------------------------------------------------------------------

// One object the Measure tool may pick on: the scene object that supplies its world transform,
// bounds and visibility, plus a way to obtain the cached triangle geometry in that object's local
// space.
//
// The geometry is fetched through a callable rather than held directly so that a provider can list
// every candidate cheaply. Reading and analysing a scene's worth of meshes takes long
// and the tool only needs the ones whose world bounds the cursor ray actually crosses.
struct MeasureTarget {
  std::string label;
  MeasureTargetKind kind = MeasureTargetKind::Render;
  mochi_renderer::SceneObject* object = nullptr;
  std::function<std::shared_ptr<MeasureMesh const>()> getMesh;
};

// Builds and caches the @ref MeasureMesh for each measurable object, keyed either by the model file
// it was loaded from or by the in-memory buffer it was generated into. Owned by the tool and
// dropped when the tool is switched off, so nothing is held for a tool the user is not using.
class MeasureGeometryCache {
 public:
  explicit MeasureGeometryCache(MeasureMeshParams params) : _params(params) {}

  // Geometry for a model file, read in renderer space so it lines up with the SceneObject that was
  // staged from the same file. `.mochi.h5` collision models go through the mochi-model loader, mesh
  // files through the render-model reader; both honor the per-format space convention in
  // processing_mesh_utils.h. Returns an empty mesh (never null) for an unsupported or unreadable
  // file.
  std::shared_ptr<MeasureMesh const> GetForModelFile(mochi::Path const& path);

  // Geometry for sections an editor already holds in memory, keyed by @p owner (the buffer's
  // address). The entry is rebuilt when the sections change; see the change-stamp note in .cpp.
  std::shared_ptr<MeasureMesh const> GetForSections(
      void const* owner,
      std::vector<mochi_renderer::MeshSection> const& sections);

  // Rebuilds everything on the next request. Call when the facet tolerance changes.
  void SetParams(MeasureMeshParams params);
  void Clear();

 private:
  struct SectionsEntry {
    uint64_t stamp = 0;
    std::shared_ptr<MeasureMesh const> mesh;
  };

  MeasureMeshParams _params;
  std::map<mochi::Path, std::shared_ptr<MeasureMesh const>> _byFile;
  std::map<void const*, SectionsEntry> _bySections;
};

// Emits a target for each visible representation of each staged actor: the render model and the
// collision (shape) model are separate targets, so a link can be measured in either. Shared by
// every editor that stages through a @ref SceneStage.
//
// @p excludeDeformable drops the actors whose surface deforms per-vertex while simulating (soft
// actors and articulated skins). Their geometry is cached from the rest-pose model file, so once a
// run has deformed them a pick would report the undeformed shape while the viewport shows the
// deformed one. Set it for the duration of a simulation session, paused or not.
void CollectSceneStageMeasureTargets(
    SceneStage const& stage,
    MeasureGeometryCache& cache,
    bool excludeDeformable,
    std::vector<MeasureTarget>& out);

// Wires @p viewport's Measure tool to @p stage, so it picks on the visible render and collision
// representations of every staged actor. @p stage must outlive @p viewport. Mirrors @ref
// BindSceneObjectDragHooks.
//
// The tool is locked out while the simulation is running, but not when paused
void BindSceneStageMeasureTargets(
    Viewport& viewport,
    SceneStage const& stage,
    std::function<bool()> const& isSimulating,
    std::function<bool()> const& isPaused);

//--------------------------------------------------------------------------------------------------
// TOOL
//--------------------------------------------------------------------------------------------------

// A recorded measurement: the selection and its report,
// not serialized and does not survive between sessions
struct MeasureRecord {
  std::string label;
  MeasurementReport report;
  std::vector<MeasureEntity> entities;
};

// Modal viewport tool for measuring geometry: the user picks vertices or faces on any visible
// render or collision mesh and the panel reports the distances between them.
//
// Owned by @ref Viewport, which registers its Ctrl+M toggle, feeds it the cursor ray, and submits
// its highlights. Editors supply the pickable geometry through @ref SetTargetProvider.
//
// Picking is a CPU ray cast against @ref MeasureGeometryCache rather than the viewport's GPU pick:
// the GPU pick is asynchronous and resolves only to an object plus a surface point, with no
// triangle or vertex identity. Casting here also makes the hover preview free, since a pick
// resolves in the frame it was issued.
class MeasureTool {
 public:
  // Name of the tool's panel, and the key its visibility is persisted under. Owned here rather than
  // by the editors because the tool opens its own panel when it is switched on.
  static char const* GetWindowName();

  // @p viewport must outlive the tool (the viewport owns it).
  MeasureTool(SuperDexStudio* studio, Viewport* viewport);
  ~MeasureTool();

  MeasureTool(MeasureTool const&) = delete;
  MeasureTool& operator=(MeasureTool const&) = delete;
  MeasureTool(MeasureTool&&) = delete;
  MeasureTool& operator=(MeasureTool&&) = delete;

  // Fills the list of objects that may be picked, called on every pick and hover. Editors set this
  // in Initialize(). Without one the tool activates but finds nothing.
  using TargetProvider = std::function<void(std::vector<MeasureTarget>&)>;
  void SetTargetProvider(TargetProvider provider);

  // Consulted before the tool can be switched on, so an editor can lock it out (e.g. while a
  // simulation is running, when the staged transforms move under the cached geometry every step).
  // An unset predicate means always available.
  void SetAvailablePredicate(std::function<bool()> predicate);
  [[nodiscard]] bool IsAvailable() const;

  // Supplies a stamp that changes whenever the staged geometry moves. A selection bakes its points
  // at pick time, so it describes where things were then; @ref DiscardSelectionIfPoseChanged drops
  // it once the stamp moves on. An unset provider disables that check.
  void SetPoseStampProvider(std::function<uint64_t()> provider);
  // Drops a selection picked against an older staged pose. Called once per frame by the viewport,
  // which is what catches a single step: that re-pauses within the step, so the tool never observes
  // an unpaused frame and cannot notice the move from the pause state alone.
  void DiscardSelectionIfPoseChanged();

  // The cache a target provider builds its @ref MeasureTarget geometry through.
  MeasureGeometryCache& GetGeometryCache() {
    return _cache;
  }

  bool IsActive() const {
    return _active;
  }
  // Turning the tool on opens and raises its panel, and takes over left-click in the viewport;
  // turning it off restores the viewport's own picking and drag gates and clears the live selection
  // (records are kept).
  void SetActive(bool active);
  // What the Ctrl+M shortcut and the Show-menu item do. Normally a plain toggle, but when the tool
  // is already on with its panel closed it reopens the panel instead of switching off
  void ToggleFromShortcut();

  // Resolves the cursor ray against every target and, when @p commit is set, adds the hit to the
  // selection (replacing it unless Ctrl or Shift is held). Called by the viewport each frame it is
  // hovered while the tool is active; @p rayOrigin / @p rayDirection are in render space.
  void ProcessCursorRay(
      filament::math::float3 rayOrigin,
      filament::math::float3 rayDirection,
      bool commit,
      bool additive);
  // Drops the hover preview, e.g. when the cursor leaves the viewport.
  void ClearHover();

  // Draws the selection highlight, the hover preview, and the measurement lines and labels. Called
  // once per frame from Viewport::DrawDebug; a no-op when the tool is off.
  void DrawDebug(mochi_renderer::DebugDraw* debugDraw, DebugText* debugText) const;

  // The Measure panel. Pass the studio's visibility flag so the window's close button works.
  void ShowWindow(char const* name, bool* open);

 private:
  void ClearSelection();
  void RecomputeReport();
  // The current staged-pose stamp, or 0 without a provider.
  uint64_t CurrentPoseStamp() const;
  // Opens the panel (if closed) and queues a request to raise it to the front of its dock node.
  void RevealWindow();
  // Applies the current picking options to the cache, rebuilding geometry if the facet tolerance
  // changed.
  void ApplyMeshParams();
  bool IsTargetPickable(MeasureTarget const& target) const;
  void ShowOptions();
  void ShowSelectionList();
  void ShowRecords();
  // Adds the current selection to @ref _records.
  void RecordSelection();

  SuperDexStudio* _studio = nullptr;
  Viewport* _viewport = nullptr;

  TargetProvider _targetProvider;
  std::function<bool()> _availablePredicate;
  std::function<uint64_t()> _poseStampProvider;
  // Staged-pose stamp the live selection was picked against.
  uint64_t _selectionPoseStamp = 0;
  // Scratch list refilled by the provider on each pick/hover, kept as a member to avoid
  // reallocating every frame.
  std::vector<MeasureTarget> _targets;

  bool _active = false;
  bool _savedEnableViewportPicking = true;
  bool _savedEnableSceneObjectDrag = true;

  MeasurePickMode _pickMode = MeasurePickMode::Facet;
  // Which representations may be picked. Both by default.
  bool _pickRender = true;
  bool _pickCollision = true;
  // Whether the delta's axis legs carry their X/Y/Z labels. The legs are drawn either way.
  bool _showAxisLabels = true;
  MeasureMeshParams _meshParams;
  MeasureReportParams _reportParams;

  MeasureGeometryCache _cache;
  std::vector<MeasureEntity> _selection;
  MeasurementReport _report;
  std::vector<MeasureEntity> _hover;

  std::vector<MeasureRecord> _records;
  // Index of the record whose selection was last restored, so the panel can mark it. -1 when the
  // selection has been edited since.
  int _activeRecord = -1;
  int _nextRecordNumber = 1;

  // What the last hover resolved to, so an unchanged hover is not rebuilt every frame. A facet can
  // run to thousands of triangles at a loose tolerance, which is too much to re-extract per frame.
  struct HoverKey {
    mochi_renderer::SceneObject const* object = nullptr;
    int triangle = -1;
    int facet = -1;
    // The welded corner nearest the cursor. Part of the key because in Vertex mode it alone decides
    // the entity: without it, sliding the cursor between the corners of one large triangle would
    // keep showing the first corner's marker.
    int vertex = -1;
    MeasurePickMode pickMode = MeasurePickMode::Facet;

    bool operator==(HoverKey const& other) const = default;
  };
  HoverKey _hoverKey;

  bool _pendingWindowFocus = false;
};

} // namespace superdex::studio
