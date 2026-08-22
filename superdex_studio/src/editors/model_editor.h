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

#include "core/async_task.h"
#include "editors/asset_editor.h"
#include "meshing/processing_modifiers/processing_modifier.h"
#include "rendering/viewport.h"

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/error.h>

#include <mochi_mesh/surface_remeshing.h>

#include <mochi_renderer/utils.h>

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace mochi_renderer {
class MaterialInstance;
class Mesh;
class WireframeMesh;
} // namespace mochi_renderer

namespace superdex::studio {

class MochiModelAsset;
class RenderModelAsset;
class CadModelAsset;
class SuperDexStudio;

// One user-added "Additional Reference Model": a viewer-only reference asset shown alongside the
// fixed CAD/render/mochi slots, with its own type, transform, and visualization toggles. Its
// persisted form is ReferenceModelState (processing_serialization.h).
//
// IMPORTANT: reference models are visualization context ONLY. They are intentionally NOT reachable
// by the mesh-processing modifiers (no entry in ModifierRunContext, not offered as a source). If
// you want to feed a mesh that is not part of this model's filename set into the pipeline, add a
// Source modifier with "From File" instead. Do not wire these into the modifiers without revisiting
// that design decision.
struct ReferenceModel {
  AssetType type = AssetType::RenderModel;
  mochi::DynamicString path;
  // Extra transform, applied the same way as the CAD Model transform regardless of slot type.
  mochi::Real3 scale = {1.0f, 1.0f, 1.0f};
  mochi::Quaternion rotation = {};
  mochi::Real3 translation = {};
  // Flat surface color (hashed from the asset path when assigned, same scheme as the fixed slots,
  // so a given model has a stable color everywhere). For a render mesh it applies only when
  // `overrideColor` is set (or opacity < 1); otherwise the mesh keeps its real PBR
  // materials/textures. CAD/mochi references are always flat-shaded with this color.
  mochi::Real3 color = {0.75f, 0.55f, 0.85f};
  bool overrideColor = false;
  // Untransformed geometry as loaded (STL/render/mochi read, or CAD tessellation); the displayed,
  // transform-applied copy plus its scene objects and viz toggles live in `buffer`.
  std::vector<mochi_renderer::MeshSection> sectionsOriginal;
  StageBuffer buffer;
  uint64_t uiId = 0; // stable id for ImGui scoping and async re-lookup
  // A CAD STEP reference tessellates asynchronously; this marks it as needing (re)tessellation. A
  // per-frame pump (PumpReferenceCadTessellations) starts pending references one at a time as the
  // async runner frees up, so a drop / disk change always regenerates without a manual button.
  bool needsCadTessellation = false;
  // On-disk modification time of `path`, for hot-reload of the reference (see PollSlotFileChanges).
  std::filesystem::file_time_type mtime = {};
};

// Persistent drag state for one drag-to-reorder "bubble" list (see DrawReorderableBubbleList in
// model_editor.cpp). While a drag is active `draggingId` is the stable id of the grabbed row (0 =
// not dragging) and `dropSlot` is the highlighted insertion gap (the final array index on release).
struct DragReorderState {
  uint64_t draggingId = 0;
  int dropSlot = -1;
};

class ModelEditor : public AssetEditor {
 public:
  ModelEditor(SuperDexStudio* studio, MochiModelAsset* asset);
  ModelEditor(SuperDexStudio* studio, RenderModelAsset* asset);
  ModelEditor(SuperDexStudio* studio, CadModelAsset* asset);
  void Initialize() override;
  void OnRender(Renderer const* renderer) override;
  void Shutdown() override;
  void ShowTabContents() override;
  static std::vector<WindowDeclaration> GetDefaultWindows();
  std::vector<WindowDeclaration> GetAuxiliaryWindows() const override;
  void ShowAuxiliaryWindows() override;
  void ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) override;
  Viewport* GetViewport() override {
    return _viewport.get();
  }
  void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const override;
  // A Model Editor represents all of its slotted models (CAD / render / mochi), so opening any of
  // them focuses this editor rather than opening a second instance.
  bool RepresentsAsset(Asset const* asset) const override;
  // Reopening one of the slotted models focuses its visualization: show that slot, hide the other
  // two (and the SDF sub-viz), matching a fresh open.
  void OnReopenedFor(Asset* asset) override;
  // Re-resolve the slotted models from their current paths and rebuild their visualizations after
  // an external edit (e.g. a slot's model was replaced/renamed elsewhere in the app).
  void Refresh() override;

 private:
  // Model Viewer window: the CAD / render / mochi model slots and their visualization controls.
  void ShowModelViewerWindow(bool* open);
  // "Additional Reference Models" section of the Model Viewer: a user-managed array of viewer-only
  // reference slots (see ReferenceModel). Draws the add button and each element's type dropdown,
  // asset slot, visualization toggles, transform, and remove button.
  void ShowReferenceModelsSection();
  // Inserts a new (empty, Render Model) reference slot and persists. @p addToTop inserts at the
  // front of the array; otherwise it is appended to the back.
  void AddReferenceModel(bool addToTop);
  // Destroys the reference model at @p index (and its scene objects) and persists.
  void RemoveReferenceModel(std::size_t index);
  // (Re)loads a reference model's geometry for its current type/path (STL/render/mochi read
  // synchronously; a CAD STEP is tessellated asynchronously via RegenerateReferenceCadModel), then
  // applies its transform and rebuilds its meshes. Loads the asset so the slot renders a thumbnail.
  // Pass @p assignDefaultColor = true when the asset is newly assigned (drop / type change) so a
  // mochi reference adopts the model's own surface color; false on load / hot-reload keeps the
  // persisted (possibly user-customized) color.
  void UpdateReferenceModel(ReferenceModel& rm, bool assignDefaultColor = false);
  // Re-derives a reference model's displayed buffer from its untransformed geometry by applying its
  // transform, then rebuilds its scene meshes. Cheap enough to run live on transform edits.
  void ApplyReferenceModelTransform(ReferenceModel& rm);
  // (Re)builds a reference model's surface + wireframe scene objects from its transformed sections.
  // A render reference keeps its real PBR materials/textures (via GetRenderModelInstance, posed
  // with SetLocalTransform) unless its color override is on or it is translucent; CAD/mochi
  // references (and overridden render references) are flat-shaded with the reference's color.
  void RebuildReferenceMeshes(ReferenceModel& rm);
  // Tessellates a CAD-type reference model's slotted STEP file asynchronously (progress modal), the
  // same standalone preview tessellation used by the main CAD slot (including SetTessellation so
  // the slot gets a thumbnail). STL CAD references are read directly in UpdateReferenceModel.
  void RegenerateReferenceCadModel(ReferenceModel& rm);
  // Once per frame: if the async runner is idle, start the tessellation of the first reference
  // marked needsCadTessellation (dropped / disk-changed CAD STEP references), one at a time. This
  // replaces a manual Generate button -- references regenerate automatically as the runner frees
  // up.
  void PumpReferenceCadTessellations();
  // Destroys a reference model's surface + wireframe scene objects.
  void DestroyReferenceModelMeshes(ReferenceModel& rm);
  // Model Processing window: the mesh processing modifier stack (mirrors the Bot Hierarchy / Bot
  // Details split — Viewer docks top, Processing docks bottom).
  void ShowModelProcessingWindow(bool* open);
  // Shows the mesh processing modifier stack: an ordered, user-editable array of modifiers (add /
  // remove / reorder / enable). Each transform modifies the output of the nearest enabled modifier
  // before it; the array starts with a source modifier. Generate on any modifier auto-cascades,
  // rebuilding stale upstream modifiers first (see GenerateModifier).
  void ShowMeshModifierStack();
  // Fill empty slots by looking for same-named CAD/render/mochi files next to the originating asset
  // (see Form 1 in the multi-file loading design): the matching sibling folder up one level
  // (intermediates/render/collision), then the asset's own folder, else left blank.
  void DiscoverSiblingSlots();
  // Polls the slot files' modification times (throttled to ~10 Hz) and reloads any slot whose file
  // changed on disk (e.g. after an export modifier overwrites it), rebuilding that slot's
  // visualization. Reloading drops and re-establishes this editor's asset references so the changed
  // asset can be unloaded and re-read.
  void PollSlotFileChanges();
  // Whether @p path's on-disk modification time differs from @p stored (false for an empty or
  // missing path).
  static bool SlotFileChanged(
      mochi::DynamicString const& path,
      std::filesystem::file_time_type stored);
  void UpdateCadModel();
  // (Re)builds the CAD surface + wireframe scene objects from _cadSections (the current opacity and
  // visibility flags apply). Does not re-tessellate.
  void RebuildCadModelMeshes();
  // Tessellates the slotted CAD file asynchronously (progress modal) into _cadSections and the
  // asset's thumbnail model, then rebuilds the CAD meshes. This is the standalone CAD-preview
  // tessellation (fixed default deflections); the modifier stack's source is where tessellation is
  // tuned. Invoked on open and by the Generate button.
  void RegenerateCadModel();
  // Re-derives the displayed CAD buffer (_cadSections) from the untransformed tessellation
  // (_cadSectionsOriginal) by applying the CAD transform, then rebuilds the CAD meshes. Cheap
  // enough to run live on every transform edit (no re-tessellation).
  void ApplyCadModelTransform();
  // Applies the staged CAD transform (_cadScale/_cadRotation/_cadTranslation) in place to a set of
  // mesh sections (positions scaled, then rotated, then translated; stored normals rotated). Shared
  // by the viewing buffer and the processing stages.
  void ApplyCadTransform(std::vector<mochi_renderer::MeshSection>& sections) const;
  // The seed shared by all of a stage's role colors: the opened model's file path (first populated
  // slot) plus the stage's header label (modifier + method), so each stage's colors are stable and
  // differ between stages and models.
  std::string StageColorSeed(MeshProcessingModifier const& modifier) const;
  // Stable, distinct display color for a modifier stage's output, derived by hashing
  // StageColorSeed. Same scheme as the model / reference default colors. A method that pins
  // PreferredDisplayColor overrides the hash.
  mochi::Real3 StageOutputColor(MeshProcessingModifier const& modifier) const;
  // Display color for a stage mesh in a given role (e.g. the input surface vs. the reconstructed
  // SDF of an Export Mochi Model modifier): hashes StageColorSeed + @p role so paired meshes get
  // distinct but stable colors. A method that pins PreferredDisplayColor overrides the hash.
  mochi::Real3 StageRoleColor(MeshProcessingModifier const& modifier, std::string_view role) const;
  // (Re)builds a processing stage's flat-lit surface + wireframe scene objects from its sections,
  // honoring the stage's visualization toggles and opacity. The wireframe color is derived from
  // @p surfaceColor via WireframeColorForSurface (the shared app-wide scheme).
  void RebuildStageMeshes(StageBuffer& stage, mochi::Real3 const& surfaceColor);
  // Destroys a stage buffer's surface + wireframe scene objects (and nulls the pointers). Shared by
  // the reset / remove paths for both a modifier's output and its optional input-view buffer.
  void DestroyStageBufferMeshes(StageBuffer& stage);
  // Recolors an existing wireframe's edges to match @p surfaceColor (via WireframeColorForSurface),
  // without rebuilding the mesh -- used when a color override changes a surface color live.
  void RecolorWireframe(mochi_renderer::WireframeMesh* wireframe, mochi::Real3 const& surfaceColor);
  // Destroys every processing stage's scene objects and clears its sections (e.g. when the CAD slot
  // changes). The stage buffers are per-stage manual, so they are not recomputed here.
  void ResetStageBuffers();
  // Generates the modifier at @p index (Generate button): builds its chain and regenerates only the
  // stale stages down to it; logs "already up to date" when nothing is stale. One serial async
  // batch.
  void GenerateModifier(std::size_t index);
  // Generates the modifier at @p index (stale-aware) and then exports it to its configured file.
  // Used by an export modifier's Export button.
  void GenerateAndExportModifier(std::size_t index);
  // Builds every enabled modifier top-to-bottom (stale-aware) then exports every export modifier
  // that has a configured path -- one serial async batch. Backs the Build/Export All button.
  void BuildAndExportAll();
  // Shared driver: regenerates the stale stages of @p chain (source-first), and exports the
  // modifiers at @p exportChainPositions (positions within @p chain) as it reaches them. Early-outs
  // (with a log) when nothing is stale and there is nothing to export.
  void GenerateChainWithExports(
      std::vector<std::size_t> const& chain,
      std::vector<std::size_t> const& exportChainPositions);
  // The source-first list of enabled modifier indices feeding @p index (with @p index last), or
  // empty when no source is reachable upstream. Stops at the nearest source -- one modifier's
  // Generate rebuilds its own segment, not the segments feeding it through files.
  [[nodiscard]] std::vector<std::size_t> BuildGenerationChain(std::size_t index) const;
  // The whole enabled stack in order, from its first enabled source -- what Build/Export All
  // builds, spanning every source->...->export segment. Empty when no enabled source exists.
  [[nodiscard]] std::vector<std::size_t> BuildFullGenerationChain() const;
  // The modifier index whose output an edge-swap at @p index references (its explicit upstream
  // target, else the nearest preceding source), or -1 if none.
  int ReferenceModifierIndex(std::size_t index) const;
  // Whether Generate is available for @p index: a source is reachable upstream and has its input
  // file/slot, and no async task is running. Deliberately NOT gated on intermediate buffers being
  // up to date (the cascade rebuilds stale ones).
  bool CanGenerateModifier(std::size_t index) const;
  // Runs the stale stages of @p chain (regen[p] marks a stale stage), each modifier at
  // @p exportChainPositions exporting right after its own stage -- one serial async batch --
  // committing generation ids / signatures + meshes on completion.
  void RunGenerationCascade(
      std::vector<std::size_t> const& chain,
      std::vector<bool> const& regen,
      ModifierRunContext const& ctx,
      std::vector<std::size_t> const& exportChainPositions);
  // Saves the modifier at @p index's current output mesh to a .glb/.obj file chosen via a dialog
  // that defaults to the asset's intermediates folder. Async write.
  void SaveModifierOutput(std::size_t index);
  // Default output-save path for the modifier at @p index:
  // <intermediates>/<openedBaseName>_<TypeName><index><ext>.
  std::string DefaultModifierSavePath(std::size_t index, char const* ext) const;
  // Builds the run context shared by generate + readiness: the editor's current slot paths and CAD
  // transform (the reference mesh is filled in by GenerateModifier when needed).
  ModifierRunContext MakeRunContext() const;
  // Path of the model this editor was opened for: the first populated slot, CAD then render then
  // mochi. Empty when no slot is populated.
  mochi::DynamicString OriginModelPath() const;
  // Directory holding OriginModelPath, or empty when there is no origin. Only used to give an
  // export Browse dialog somewhere sensible to open when there is no source to suggest from.
  std::string OriginModelFolder() const;
  // The file the stack's source (first) modifier reads, which every export modifier derives its
  // default output path from. Empty when the stack is empty or its source has no file yet.
  std::string StackSourceFilePath() const;
  // Re-derives every export modifier's path from @p sourceFilePath while its Auto toggle is on (see
  // MeshProcessingMethod::RefreshAutoExportPath), so an export writes its file on Build/Export All
  // without the user opening Browse, and an Auto path always matches what Browse would offer.
  // Called once per frame from the stack UI, ahead of everything that reads an export path. Safe
  // to run that often because an Auto path is never serialized, so re-deriving it cannot dirty the
  // saved snapshot however much it moves.
  void RefreshAutoExportPaths(std::string const& sourceFilePath);
  // Per-modifier flag marking export modifiers that share an output file with another enabled
  // export modifier, so the UI can say so. Two exports aimed at the same path silently overwrite
  // each other; the Auto path is deliberately not disambiguated (it has to keep the model's own
  // name to stay associated with it), so the fix is to turn Auto off on one of them. Compares
  // resolved paths, so hand-picked duplicates are caught as well.
  std::vector<bool> FindCollidingExportPaths() const;
  // Appends a new modifier of registry type @p name to the stack.
  void AddModifier(std::string_view name);
  // Inserts a new modifier of registry type @p name at @p at (clamped to [0, size]), updating
  // existing modifiers' upstream references so they keep pointing at the same elements. AddModifier
  // appends via this. Logs and no-ops if @p name is not registered.
  void InsertModifier(std::string_view name, std::size_t at);
  // Destroys the modifier at @p index (and its scene objects) and removes it from the stack.
  void RemoveModifier(std::size_t index);
  // Moves the modifier at @p from to @p to. Sources may move anywhere (including index 0); other
  // modifiers cannot become the first element.
  void MoveModifier(std::size_t from, std::size_t to);
  // Applies @p oldToNew (old array index -> new index, or -1 if removed) to every modifier's stored
  // upstream reference. Called by insert / remove / move so references follow their targets.
  void RemapModifierReferences(std::vector<int> const& oldToNew);
  // Populates the stack with the default STEP->mesh pipeline (the original stages 1-8), collapsed.
  void AddDefaultStepPipeline();
  // Path of this editor's pipeline JSON (<intermediates>/<originBaseName>.StudioProcessing.json),
  // or empty when there is no origin model.
  std::string ComputeProcessingJsonPath() const;
  // On open, loads the saved pipeline JSON into _modifiers (if it exists), applying the saved CAD
  // transform. On a read/parse failure it starts empty and blocks auto-save so the file is not
  // clobbered.
  void LoadProcessingPipelineOnOpen();
  // Persists the current pipeline (+ CAD transform) to _processingJsonPath. No-op when there is no
  // path, the on-open load failed, or the serialized content is unchanged since the last save/load
  // (so repeated generates / an unedited open+close never touch the file). When the pipeline is
  // back to the default state (no modifiers, no reference models, identity CAD transform), any
  // existing file is deleted instead of written, so a reset state never leaves a redundant JSON
  // behind. Called on generate / build / close and on discrete edits.
  void SaveProcessingPipelineToDisk();
  // Serializes the editor's current pipeline state (modifiers + reference models + CAD transform)
  // into the persisted JSON snapshot string. Used to detect changes (vs _lastSavedPipelineJson) and
  // reusable as an in-memory snapshot for a future undo/redo history.
  std::string SerializeCurrentPipeline() const;
  // Replaces the modifier stack with a shipped preset pipeline loaded from @p path (confirming
  // first when the stack is non-empty). Saving still targets this model's own intermediates JSON.
  void PopulateFromPreset(std::string const& path);
  void UpdateRenderModel();
  // (Re)builds the render model's glTF surface instance and applies the current opacity (opaque at
  // 1.0 to preserve PBR; a translucent override below 1.0).
  void RebuildRenderModelSurface();
  void UpdateMochiModel();
  void RebuildMochiModelSurfaceMesh();
  void RebuildMochiModelWireframeMesh();
  void DestroyMochiModelWireframeMesh();
  // Reconstructs the SDF surface once (a CLI subprocess) and builds both the SDF surface and
  // wireframe meshes from that shared geometry.
  void RebuildSdfMeshes();
  void DestroySdfMesh();
  void DestroySdfWireframeMesh();
  bool BuildSdfSurfaceGeometry(std::vector<float>& positions, std::vector<int>& indices) const;
  void ApplyModelTransform();
  // Launches a hidden background task computing the approximate Hausdorff distance between a
  // modifier's @p input and @p output meshes (via the mesh CLI), tagged with the modifier's id +
  // generation so a stale result is discarded. Drained by PumpHausdorffResults.
  // Enqueues a hidden serial background job computing the approximate Hausdorff distance between a
  // modifier's @p input and @p output meshes, keyed by the modifier id; when it returns,
  // PumpHausdorffResults stores it on the still-current modifier's MeshStats. A later regenerate /
  // remove / reset supersedes or cancels it through the queue (independently of other modifiers).
  void KickoffModifierHausdorff(
      MeshProcessingModifier const& mod,
      mochi::MeshData input,
      mochi::MeshData output);
  // Applies finished background Hausdorff results (main thread): stores each on its still-current
  // modifier's MeshStats (guarded by outputGenId) and recomposites the display. Called once per
  // frame.
  void PumpHausdorffResults();

 private:
  std::unique_ptr<Viewport> _viewport;
  // target asset(s)
  CadModelAsset* _cadModelAsset = nullptr;
  RenderModelAsset* _renderModelAsset = nullptr;
  MochiModelAsset* _mochiModelAsset = nullptr;
  // The model the editor was opened for (the double-clicked asset). Only this model's visualization
  // is enabled by default; discovered sibling slots stay hidden until toggled on.
  AssetType _originType = AssetType::Unknown;
  mochi::DynamicString _cadModelPath;
  mochi::DynamicString _renderModelPath;
  mochi::DynamicString _mochiModelPath;
  // Visibility state. All default off; Initialize() turns on only the origin model's visualization
  // (see _originType), so sibling slots that get auto-populated open hidden.
  bool _showCadModelSurface = false;
  bool _showCadModelWireframe = false;
  bool _showRenderModelSurface = false;
  bool _showRenderModelWireframe = false;
  bool _showMochiModelSurface = false;
  bool _showMochiModelWireframe = false;
  bool _showSdfSurface = false;
  bool _showSdfWireframe = false;
  // Show the read-only mesh statistics block for each fixed slot (stages/reference models keep this
  // in their StageVisualization).
  bool _showCadModelStats = false;
  bool _showRenderModelStats = false;
  bool _showMochiModelStats = false;
  bool _showSdfStats = false;
  // per-surface opacity [0,1]
  float _cadModelOpacity = 1.0f;
  float _renderModelOpacity = 1.0f;
  float _mochiModelOpacity = 1.0f;
  float _sdfOpacity = 1.0f;
  // Mesh statistics per fixed slot (base summary + composited annotations), recomputed when that
  // slot's geometry changes (drop / reload / regenerate) and shown under its viz controls when its
  // _show*Stats flag is on.
  MeshStats _cadStats;
  MeshStats _renderStats;
  MeshStats _mochiStats;
  MeshStats _sdfStats;
  // Per-fixed-slot flat surface color (editable via each slot's color picker). Defaults to the
  // color the slot loaded with (CAD/render use fixed defaults; mochi picks up the asset's surface
  // color in UpdateMochiModel). Session-only -- not persisted, so each open starts from the loaded
  // color. The CAD and mochi surfaces are always flat-shaded with their color; the render surface
  // keeps its PBR materials/textures unless _renderOverrideColor is set (or it is translucent).
  mochi::Real3 _cadColor = {0.60f, 0.60f, 0.62f};
  mochi::Real3 _renderColor = {0.70f, 0.70f, 0.72f};
  bool _renderOverrideColor = false;
  mochi::Real3 _mochiColor = {0.5f, 0.7f, 1.0f};
  // SDF reconstruction surface color, derived (like _mochiColor) by hashing the mochi model path
  // with an "output SDF" role so the mochi surface and its SDF get distinct but stable colors. Set
  // in UpdateMochiModel; the wireframe uses a darker shade.
  mochi::Real3 _sdfColor = {1.0f, 0.0f, 0.0f};
  // scene objects
  mochi_renderer::SceneObject* _cadModelSurfaceMesh = nullptr;
  mochi_renderer::WireframeMesh* _cadModelWireframeMesh = nullptr;
  mochi_renderer::SceneObject* _renderModelSurfaceMesh = nullptr;
  mochi_renderer::WireframeMesh* _renderModelWireframeMesh = nullptr;
  mochi_renderer::SceneObject* _mochiModelSurfaceMesh = nullptr;
  mochi_renderer::WireframeMesh* _mochiModelWireframeMesh = nullptr;
  mochi_renderer::Mesh* _sdfSurfaceMesh = nullptr;
  mochi_renderer::WireframeMesh* _sdfWireframeMesh = nullptr;
  // working geometry backing per slot (created on open; drives wireframe and future mesh
  // processing)
  // The CAD buffer is kept in two forms: the raw tessellation as returned by the helper
  // (_cadSectionsOriginal) and the transform-applied copy that is actually displayed and fed to the
  // processing stages (_cadSections). Editing the CAD transform only re-applies it to the original,
  // avoiding a re-tessellation.
  std::vector<mochi_renderer::MeshSection> _cadSectionsOriginal;
  std::vector<mochi_renderer::MeshSection> _cadSections;
  std::vector<mochi_renderer::MeshSection> _renderSections;
  std::vector<mochi_renderer::MeshSection> _mochiSections;
  // mochi model edit param
  // Kept at identity: the Mochi Model section no longer edits a transform, but ApplyModelTransform
  // still uses these to position the visualization meshes.
  mochi::Real3 _modelScale = {1.0f, 1.0f, 1.0f};
  mochi::Quaternion _modelRotation = {};
  mochi::Real3 _modelTranslation = {};

  // Full transform applied to the imported CAD geometry (replaces cad_mesher's axis-swap flag).
  // Edited in the CAD Model section and applied live to _cadSections via ApplyCadModelTransform; it
  // bakes into the displayed buffer so it also flows into every downstream processing stage.
  mochi::Real3 _cadScale = {1.0f, 1.0f, 1.0f};
  mochi::Quaternion _cadRotation = {};
  mochi::Real3 _cadTranslation = {};

  // User-managed, viewer-only reference models (the "Additional Reference Models" section).
  // Persisted to / restored from the .StudioProcessing.json. Not exposed to the mesh-processing
  // modifiers (see the note on ReferenceModel).
  std::vector<ReferenceModel> _referenceModels;
  uint64_t _nextReferenceModelId = 1; // stable ids for ImGui scoping / async re-lookup
  // Set when a live drag (transform / opacity) edits a reference model; flushed to the JSON once no
  // widget is active, so a drag persists on release without writing every frame. Discrete edits
  // (add / remove / type / asset / visibility toggle) save immediately instead.
  bool _referenceModelsDirty = false;

  // The mesh processing modifier stack (a source followed by transforms), each owning its
  // params, GUI, and output buffer. Edited via ShowMeshModifierStack; generated per-modifier on
  // demand. The buffers are stale after a reorder/toggle until re-generated (cascading build
  // later).
  std::vector<std::unique_ptr<MeshProcessingModifier>> _modifiers;
  uint64_t _nextModifierId =
      1; // stable ids for ImGui / drag-drop, incremented per created modifier
  // Globally-unique, monotonic id stamped on each modifier output as it is generated. A modifier
  // detects that its input changed when its stored input id no longer equals the input's current
  // output id; global uniqueness makes this robust to reorders with no special invalidation. The
  // sentinel 0 means "never generated".
  int _nextGenerationId = 1;
  // Drag-to-reorder state for the modifier stack and the reference models list (see
  // DrawReorderableBubbleList).
  DragReorderState _modifierDrag;
  DragReorderState _referenceDrag;

  // Set by DiscoverSiblingSlots when the opened model is not the one its own base name resolves to
  // -- a copy in `render/internal/` outranked by one in `render/`. Such a model is a set of one: no
  // sibling slots are filled, and its generated files are keyed apart from the canonical model's so
  // its pipeline is neither loaded from nor written over that model's (see AssetGeneratedFilePath).
  bool _originIsShadowed = false;
  // Pipeline persistence: the JSON this editor loads on open and saves to on generate / build /
  // close. Empty when there is no origin model.
  std::string _processingJsonPath;
  bool _pipelineLoadFailed = false; // on-open load failed -> block auto-save (do not clobber)
  // The serialized pipeline snapshot as of the last successful save or the on-open load. A save
  // writes only when the current snapshot differs from this, so nothing is written unless the
  // modifiers, reference models, or CAD transform actually changed.
  std::string _lastSavedPipelineJson;

  // Hot-reload of slot files overwritten on disk. Each Update*Model records its slot file's
  // modification time; PollSlotFileChanges compares against these at ~10 Hz and reloads on a
  // change.
  std::chrono::steady_clock::time_point _lastSlotPollTime = {};
  std::filesystem::file_time_type _cadModelMtime = {};
  std::filesystem::file_time_type _renderModelMtime = {};
  std::filesystem::file_time_type _mochiModelMtime = {};

  // A finished background Hausdorff computation for one modifier. `genId` is the modifier's
  // outputGenId at kickoff; on drain the result is applied only if the modifier still exists and
  // has not regenerated since (so a stale distance is never shown).
  struct HausdorffResult {
    uint64_t modifierId = 0;
    int genId = 0;
    double distance = -1.0;
  };
  // Mailbox the background worker posts finished results into; drained on the main thread by
  // PumpHausdorffResults. Held by shared_ptr so an in-flight job can safely post even if this
  // editor is torn down first (the job keeps the mailbox alive via its own copy).
  struct HausdorffInbox {
    std::mutex mutex;
    std::vector<HausdorffResult> results;
  };
  std::shared_ptr<HausdorffInbox> _hausdorffInbox = std::make_shared<HausdorffInbox>();
  // Serial background queue running one modifier Hausdorff at a time, keyed by modifier id so a
  // regenerated / removed / reset modifier's job is superseded or cancelled independently of
  // others. Declared last so it is destroyed (worker joined) before the mailbox it posts to.
  BackgroundTaskQueue _hausdorffQueue;
};

} // namespace superdex::studio
