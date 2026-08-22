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
#include <superdex_robotics/internal/bot_scene.h>
#endif
#include <superdex_robotics/superdex_robotics.h>

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_renderer/path.h>

#include <math/vec3.h>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mochi {
class Scene;
} // namespace mochi

namespace mochi::prefab {
struct ScenePrefab;
} // namespace mochi::prefab

namespace mochi_renderer {
class Scene;
class SceneObject;
class Mesh;
class WireframeMesh;
} // namespace mochi_renderer

namespace superdex::studio {

class SuperDexStudio;
class MochiModelAsset;

enum class StageType {
  RenderModelFallbackToMochiModel,
  RenderModelOnly,
  MochiModelOnly,
};

// Identifies which prefab actor a staged render object originated from, so the editor can map a
// picked scene object back to the underlying prefab. Only populated for top-level prefab actors;
// nested-prefab actors keep the default (NestedPrefab) and are treated as read-only.
enum class StagedActorSource {
  NestedPrefab,
  Rigid,
  Articulated,
  Soft,
};

struct StagedActor {
  // One rendered representation of the actor. A link may carry both a render-model and a
  // mochi/shape representation; StageType chooses which is visible while both stay staged.
  // `sceneObject` is null when the actor has no model of this kind. Non-owning pointer (owned by
  // the target render Scene).
  struct Instance {
    mochi_renderer::SceneObject* sceneObject = nullptr;
    mochi::Path modelFile;
    // Representation-specific model offset and scale (render-model and shape offsets/scales
    // differ).
    mochi::TransformRT modelTransform = {};
    mochi::Real3 modelScale = {mochi::real(1), mochi::real(1), mochi::real(1)};
    // Whether this representation is shown by the current StageType. Tracked separately from the
    // SceneObject's own visibility because the highlight swap below temporarily hides the base
    // while a tinted clone stands in for it; this records the StageType intent to restore.
    bool visible = false;
    // True if this representation renders with flat (faceted) shading: shape/mochi models use a
    // flat-lit material, render models keep their (smooth) glb materials. The highlight clone picks
    // a matching flat/smooth material so it shades like the base it stands in for.
    bool flatShaded = false;
    // Tinted stand-in shown while this representation is highlighted. Created lazily on first
    // highlight and then cached (shown/hidden by UpdateHighlights, not recreated) so a rapidly
    // changing hover doesn't churn scene objects; destroyed only with its base on rebuild/Clear. A
    // non-owning pointer into the render Scene (the Scene owns it, the stage manages its lifetime).
    mochi_renderer::SceneObject* highlightClone = nullptr;
    // Color the cached highlightClone was built with (to detect selection<->hover color changes).
    filament::math::float3 highlightCloneColor = {};
  };
  // Name of the staged physics actor. Used for name-based lookup and reuse matching.
  std::string name;
  // Both representations are staged (one hidden); switching StageType toggles their visibility
  // instead of rebuilding, and the hidden one still contributes to the scene AABB / ground plane.
  Instance render;
  Instance shape;
  // The currently-visible (selectable) representation's object -- aliases `render.sceneObject` or
  // `shape.sceneObject` (null if neither is shown). Selection/picking treat a link as one object,
  // so callers use this.
  mochi_renderer::SceneObject* sceneObject = nullptr;
  // Last-applied world transform (default pose, or simulation pose while playing).
  mochi::TransformRT worldTransform = {};
  // Default-pose world transform captured during staging (used to reset after sim).
  mochi::TransformRT defaultWorldTransform = {};
  // Actor-frame scale applied between worldTransform's rotation and the model offset.
  mochi::Real3 worldScale = {mochi::real(1), mochi::real(1), mochi::real(1)};
  // Per-frame highlight request (set by selection/hover via SceneStage::RequestHighlight, consumed
  // and cleared each frame by UpdateHighlights). A link is highlighted as one identity: the swap is
  // applied to every currently-visible representation.
  bool highlightRequested = false;
  filament::math::float3 highlightColor = {};
  // Source identity mapping this staged object back to its prefab actor (top-level actors only).
  StagedActorSource source = StagedActorSource::NestedPrefab;
  int sourceActorIndex = -1;
  int sourceLinkIndex = -1;
  // Soft actors only: two deforming dynamic meshes -- a solid `mochi_renderer::Mesh` (render slot,
  // shown in StageType 1/2) and a `mochi_renderer::WireframeMesh` (shape slot, shown in StageType
  // 3). Their SceneObjects live in `render.sceneObject` / `shape.sceneObject` (so
  // picking/highlighting/visibility reuse the existing machinery); the pointers here are non-owning
  // back-pointers used to push per-frame vertex updates during simulation. Null for non-soft
  // actors. `softAsset` is the (non-owning) MochiModelAsset that re-bakes the rest surface for idle
  // edits and rest-pose restore after simulation, using the per-instance `softBakeScale` /
  // `softShapeTransform`. `softMeshVertexCount` is the surface node count (used by the reuse check
  // to force a rebuild when the topology changes).
  mochi_renderer::Mesh* dynamicSolidMesh = nullptr;
  mochi_renderer::WireframeMesh* dynamicWireframeMesh = nullptr;
  MochiModelAsset* softAsset = nullptr;
  int softMeshVertexCount = 0;
  mochi::Real3 softBakeScale = {mochi::real(1), mochi::real(1), mochi::real(1)};
  mochi::TransformRT softShapeTransform = {};
};

using StagedActors = std::vector<StagedActor>;

// A per-frame deformation update for one soft actor's dynamic mesh: produced by the simulation
// post-step callback and consumed by SceneStage::ApplySoftMeshUpdates. Position/normal buffers are
// flat (3 floats per vertex) in renderer space; `worldTransform` is the soft actor's root transform
// (Mochi space, converted on apply). Matched to a staged actor by `name`.
struct SoftMeshUpdate {
  std::string name;
  std::vector<float> positions;
  std::vector<float> normals;
  mochi::TransformRT worldTransform = {};
};

// A single resolved staging instruction produced by traversing a prefab tree. Carries both the
// render and shape representations (either file may be empty) plus which is visible for the
// StageType. Decouples request building from the reuse/rebuild bookkeeping in SceneStage.
struct StageRequest {
  std::string name;
  mochi::Path renderModelFile;
  mochi::Path shapeModelFile;
  mochi::TransformRT renderModelTransform = {};
  mochi::Real3 renderModelScale = {mochi::real(1), mochi::real(1), mochi::real(1)};
  mochi::TransformRT shapeModelTransform = {};
  mochi::Real3 shapeModelScale = {mochi::real(1), mochi::real(1), mochi::real(1)};
  bool renderVisible = false;
  bool shapeVisible = false;
  mochi::TransformRT worldTransform = {};
  mochi::Real3 worldScale = {mochi::real(1), mochi::real(1), mochi::real(1)};
  // Source identity (see StagedActor). Only set for top-level rigid/articulated/soft actors.
  StagedActorSource source = StagedActorSource::NestedPrefab;
  int sourceActorIndex = -1;
  int sourceLinkIndex = -1;
  // Soft actors only: `isSoft` marks this request as a soft actor whose solid + wireframe deforming
  // meshes are created/colored by the MochiModelAsset at `shapeModelFile` (which also forms part of
  // the reuse key). `softBakeScale` / `softShapeTransform` are the per-instance bake parameters the
  // asset uses to (re)generate the rest surface; `softVertexCount` is the surface node count used
  // to detect topology changes for the reuse check. No ordinary shape/render instance is loaded.
  bool isSoft = false;
  mochi::Real3 softBakeScale = {mochi::real(1), mochi::real(1), mochi::real(1)};
  mochi::TransformRT softShapeTransform = {};
  int softVertexCount = 0;
};

// Stages render objects for physics actors into a target render Scene and provides mechanisms to
// keep them in sync with the actors' world transforms.
struct SceneStage {
  SceneStage() = default;
  SceneStage(SuperDexStudio* studio, char const* sceneName);
  ~SceneStage();
  SceneStage(SceneStage const&) = delete;
  SceneStage& operator=(SceneStage const&) = delete;
  SceneStage(SceneStage&&) = delete;
  SceneStage& operator=(SceneStage&&) = delete;

  // Bind the target render scene that staged objects are added to / removed from. Call once before
  // staging. Re-binding to a different scene clears any objects staged into the previous one.
  void BindRenderScene(mochi_renderer::Scene* renderScene);

  // Stage (or re-stage) a bot prefab's render objects into the bound render scene at its default
  // pose, using Mochi forward kinematics. Reuses existing objects when the actor set and active
  // model files are unchanged (pure pose/param edits); otherwise rebuilds. No-op if no scene is
  // bound.
  // Returns true if the staged objects were rebuilt (pointers invalidated), false if reused.
  bool StageBot(superdex::robotics::BotPrefab const& prefab, StageType stageType);

  // Stage (or re-stage) a scene prefab's render objects into the bound render scene at its default
  // pose. Rigid (and nested-rigid) poses are computed analytically; articulated actors are expanded
  // to their nested link actors, whose world transforms come from Mochi forward kinematics on a
  // scratch scene. Staging follows prefab::AddToScene order (nested prefabs, then articulated, then
  // rigid; soft skipped) so the staged list aligns with the physics actors created during
  // simulation. Reuses existing objects when the actor set and active model files are unchanged;
  // otherwise rebuilds. No-op if no scene is bound. `rootPath` resolves non-"./" model file
  // references.
  // Returns true if the staged objects were rebuilt (pointers invalidated), false if existing
  // objects were reused.
  bool StagePrefab(
      mochi::prefab::ScenePrefab const& prefab,
      std::string const& rootPath,
      StageType stageType);

#if MOCHI_INTERNAL
  // Stage (or re-stage) a complete bot scene (base ScenePrefab + all placed bots) into the bound
  // render scene at rest pose. The base scene and each bot are resolved internally via the
  // AssetManager. The staged-actor order is the base scene (prefab::AddToScene order) followed by
  // each bot's links in declaration order — matching superdex::robotics::LoadBotScene so that
  // index-based ApplyWorldTransforms aligns with the simulated actors. No-op if no scene is bound.
  // Returns true if the staged objects were rebuilt (pointers invalidated), false if reused.
  bool StageBotScene(superdex::robotics::BotScenePrefab const& scene, StageType stageType);
#endif

  // Apply per-actor world transforms (e.g. from simulation), in staged-actor order. Warns and does
  // nothing if the incoming array length does not match the number of staged actors.
  void ApplyWorldTransforms(
      mochi::Span<mochi::TransformRT const> actorWorldTransforms,
      mochi::CoordinateSpaceConverter const& converter);

  // Apply world transforms to a subset of staged actors, matched by name. Only the named actors are
  // updated; unknown names are ignored. Errors out if the two arrays differ in length.
  void ApplyWorldTransforms(
      mochi::Span<std::string const> names,
      mochi::Span<mochi::TransformRT const> actorWorldTransforms,
      mochi::CoordinateSpaceConverter const& converter);

  // Apply a single world transform to the staged actor with the given name (no-op if not found).
  void ApplyWorldTransform(
      std::string_view name,
      mochi::TransformRT const& actorWorldTransform,
      mochi::CoordinateSpaceConverter const& converter);

  // Re-apply the default-pose transforms captured during staging.
  void ResetWorldTransforms(mochi::CoordinateSpaceConverter const& converter);

  // Apply per-soft-actor deforming-mesh updates (from simulation), matched by name. For each update
  // both dynamic meshes (solid + wireframe) are deformed and the actor's world transform is set.
  // Unknown names and non-soft actors are ignored; a vertex-count mismatch skips that update.
  void ApplySoftMeshUpdates(
      mochi::Span<SoftMeshUpdate const> updates,
      mochi::CoordinateSpaceConverter const& converter);

  // Get the ordered list of staged actors.
  StagedActors const& GetActors() const;

  // The number of staged actors.
  int GetNumActors() const;

  // Get a staged actor's index by name. Returns -1 if not found.
  int FindIndexByName(std::string_view name) const;

  // Get a staged actor's index by associated SceneObject*. Returns -1 if not found. Matches either
  // representation, so a pick/selection on either the render or shape instance resolves to the
  // link.
  int GetSceneObjectIndex(mochi_renderer::SceneObject const* object) const;

  // Request that staged actor `index` be highlighted this frame with `color` (no-op for
  // out-of-range indices). Per-frame request cleared by UpdateHighlights; the first request per
  // actor per frame wins, so callers declaring hover before selection give hover color precedence.
  // Selection and hover UIs flag the link (by index); the stage owns the mesh-level highlight swap.
  void RequestHighlight(int index, filament::math::float3 color);

  // Reconcile highlight clones against this frame's requests: for each highlighted actor, show a
  // tinted stand-in clone over every currently-visible representation (hiding that base), and hide
  // the stand-ins for actors/representations no longer highlighted (restoring the base to its
  // StageType visibility). Clones are cached (shown/hidden, not destroyed) so a rapidly changing
  // hover doesn't churn scene objects. Clears the per-frame requests. Call once per frame before
  // render.
  void UpdateHighlights();

  // True if any highlight clone is currently shown (drives the viewport's see-through overlay
  // pass).
  bool HasActiveHighlights() const;

  // Returns true if the staged scene is empty (no staged actors).
  bool IsEmpty() const;

  // Destroy the staged actors from the target render scene and forget them.
  void Clear();

  // Debug helper window the staged actor order/names, optionally against the per-step simulation
  // extraction order.
  void ShowSceneStageWindow(char const* name, bool* open, std::vector<std::string> const* simNames)
      const;

 private:
  // Apply a list of staging requests: reuse existing staged objects when the actor set (by name)
  // and active model files are unchanged, otherwise rebuild. Shared by StageBot, StagePrefab, and
  // StageBotScene.
  // Returns true if the staged objects were rebuilt (existing SceneObjects destroyed and recreated,
  // invalidating external pointers); false if existing objects were reused.
  bool ApplyStageRequests(std::vector<StageRequest> const& requests);

  // Create the opaque, tinted, shadowless stand-in clone of `base` (parented to it, tagged into the
  // highlight overlay pass). Returns nullptr if no instance could be allocated.
  mochi_renderer::SceneObject* CreateHighlightClone(
      mochi_renderer::SceneObject* base,
      filament::math::float3 color,
      bool flatShaded) const;

  SuperDexStudio* _studio = nullptr;
  mochi::Scene* _scene = nullptr;
  StagedActors _actors;
  std::unordered_map<std::string, int> _nameToIndex;
  mochi_renderer::Scene* _renderScene = nullptr;
  // Whether any highlight clone was shown by the last UpdateHighlights (see HasActiveHighlights).
  bool _hasActiveHighlights = false;
};

} // namespace superdex::studio
