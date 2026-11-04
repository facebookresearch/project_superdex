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

#include "rendering/soft_surface.h"

#if MOCHI_INTERNAL
#include <superdex_robotics/internal/bot_scene.h>
#endif
#include <superdex_robotics/superdex_robotics.h>

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_renderer/path.h>

#include <math/mat4.h>
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
class SkinnedModelInstance;
} // namespace mochi_renderer

namespace superdex::studio {

class SuperDexStudio;
class MochiModelAsset;

enum class StageType {
  RenderModelFallbackToMochiModel,
  RenderModelOnly,
  MochiModelOnly,
  // The collision model's SDF surface where it has one, else the collision model itself.
  SdfFallbackToMochiModel,
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
  // instead of rebuilding. Only the visible one contributes to the scene AABB / ground plane, so
  // framing follows what is actually drawn.
  Instance render;
  Instance shape;
  // The shape's SDF surface: an instance of the collision model's shared SDF mesh, sharing the
  // shape's model file, offset and scale. Created the first time StageType::SdfFallbackToMochiModel
  // finds the surface built, then kept like the other two. Never set for deforming actors.
  Instance sdf;
  // The currently-visible (selectable) representation's object -- aliases `render.sceneObject`,
  // `shape.sceneObject` or `sdf.sceneObject` (null if none is shown). Selection/picking treat a
  // link as one object, so callers use this.
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
  // shown in RenderModelOnly / RenderModelFallbackToMochiModel) of `softSurfaces.rendered` and a
  // `mochi_renderer::WireframeMesh` (shape slot, shown in MochiModelOnly) of
  // `softSurfaces.collision`. Their SceneObjects live in
  // `render.sceneObject` / `shape.sceneObject` (so picking/highlighting/visibility reuse the
  // existing machinery); the pointers here are non-owning back-pointers used to push per-frame
  // vertex updates during simulation. Null for non-soft actors. `softAsset` is the (non-owning)
  // MochiModelAsset that re-bakes the rest surfaces for idle edits and rest-pose restore after
  // simulation, using the per-instance `softBakeScale` / `softShapeTransform`.
  // `softMeshVertexCount` / `softCollisionVertexCount` are the two meshes' node counts (used by the
  // reuse check to force a rebuild when the topology changes).
  mochi_renderer::Mesh* dynamicSolidMesh = nullptr;
  mochi_renderer::WireframeMesh* dynamicWireframeMesh = nullptr;
  // Skinned articulated skin only: the shape slot has two interchangeable surfaces. A GPU-skinned
  // wireframe previews the skin while editing (linear blend skinning off the link transforms);
  // during simulation Mochi reports the skin's true deformed surface per vertex, which blend
  // skinning can only approximate, so a dynamic wireframe (`dynamicWireframeMesh`) stands in.
  // Exactly one is installed in `shape.sceneObject` at a time -- see SetSimSurfaceActive -- so
  // picking, highlighting, visibility and transforms all keep working through the existing slot,
  // and only the installed one is ever visible. Both are baked from the same surface by the same
  // asset, so their vertex ordering matches and a drag grabbed on one hits the same nodes on the
  // other. `shapeSkinnedSurface` is null for everything except a skinned articulated skin.
  //
  // `shapeSimSurface` exists only while simulating: it is built on the first per-vertex update and
  // torn down on reset. It holds the surface Mochi last reported, which outside a run means the
  // undeformed rest surface sitting at the skin's bind pose -- and since a hidden object still
  // contributes to the scene AABB, keeping one around at edit time would drag the ground plane,
  // the camera framing and the asset thumbnail toward wherever the bind pose happens to be.
  mochi_renderer::SceneObject* shapeSkinnedSurface = nullptr;
  mochi_renderer::SceneObject* shapeSimSurface = nullptr;
  MochiModelAsset* softAsset = nullptr;
  SoftSurfaces softSurfaces;
  int softMeshVertexCount = 0;
  int softCollisionVertexCount = 0;
  mochi::Real3 softBakeScale = {mochi::real(1), mochi::real(1), mochi::real(1)};
  mochi::TransformRT softShapeTransform = {};
  // Skinned articulated skin only: the GPU-skinned render model in `render.sceneObject`, kept as a
  // typed non-owning back-pointer so its skin can be posed each frame (SetBoneMatrices). Null
  // unless the skin has a skinned render GLB; when set, the render slot is the GLB and the
  // deforming collision mesh (dynamicSolidMesh) occupies the shape slot instead of the render slot.
  mochi_renderer::SkinnedModelInstance* skinnedRenderObject = nullptr;
  // Skinned skin only: the GPU-skinned collision wireframe in `shape.sceneObject` (Collision view),
  // driven by the same articulation link transforms as the render GLB via SetBoneMatrices.
  // Non-owning back-pointer; null unless the skin has a skinned render GLB. Replaces the
  // CPU-updated dynamicWireframeMesh for skinned skins.
  mochi_renderer::WireframeMesh* skinnedCollisionMesh = nullptr;
  // Skinned skin only: the rest/default-pose per-link world transforms (Mochi space, GLB joint
  // order) captured at stage time. Used to restore the skin's joint pose when simulation stops
  // (ResetWorldTransforms), mirroring how the collision mesh and world transforms are reset.
  std::vector<mochi::TransformRT> skinRestLinkWorldTransforms;
  // Skinned skin only: per-link inverse of the rest root-relative transform in renderer space
  // (the inverse-bind matrix for the collision wireframe), cached at stage time so bone matrices
  // can be computed as boneMatrix_j = ToFilament(rootRelCurrent_j) * skinRestBoneInverse[j].
  std::vector<filament::math::mat4f> skinRestBoneInverse;
};

using StagedActors = std::vector<StagedActor>;

#if MOCHI_INTERNAL
// One task-file spawn resolved against a bot scene and into world space. The scene editor owns the
// loaded task; SceneStage only needs this render-staging view of it.
struct ResolvedTaskSpawn {
  std::string name;
  std::string prefabName;
  mochi::Path prefabPath;
  mochi::TransformRT worldFromSpawn = {};
};
#endif

// A per-frame deformation update for one soft actor's dynamic meshes: produced by the simulation
// post-step callback and consumed by SceneStage::ApplySoftMeshUpdates. Position/normal buffers are
// flat (3 floats per vertex) in renderer space: `positions`/`normals` for the solid mesh
// (StagedActor::softSurfaces.rendered) and `collisionPositions`/`collisionNormals` for the
// wireframe (softSurfaces.collision), left empty when the two are the same mesh.
// `worldTransform` is the soft actor's root transform (Mochi space, converted on apply). Matched to
// the staged actor at `stagedIndex`, or by `name` when it is negative.
struct SoftMeshUpdate {
  std::string name;
  int stagedIndex = -1;
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<float> collisionPositions;
  std::vector<float> collisionNormals;
  mochi::TransformRT worldTransform = {};
};

// A per-frame pose update for one skinned articulated skin's render model: produced by the editor
// (rest FK) and the simulation post-step callback, consumed by SceneStage::ApplySkinnedPose.
// `linkWorldTransforms` are per-link WORLD transforms (Mochi space) in the GLB's joint order
// (== nested link order); SceneStage makes them root-relative against the staged actor's current
// world transform and converts them to renderer space before driving the GPU bones. Matched to the
// staged actor at `stagedIndex`, or by `name` (the SkinStagedName of the articulated actor) when it
// is negative.
struct SkinnedPoseUpdate {
  std::string name;
  int stagedIndex = -1;
  std::vector<mochi::TransformRT> linkWorldTransforms;
};

// The staged/render name for an articulated actor's skin. The skin is folded into the articulated
// compound entity (it shares the actor's handle and is not a nested link), so it is identified by
// the top-level actor's name plus a " (Skin)" suffix. This is the single source of truth that skin
// staging (SceneStage), the per-step surface query (editors), and force-drag resolution
// (PhysicsDragController) must all agree on.
[[nodiscard]] std::string SkinStagedName(std::string_view articulatedActorName);
// Whether `stagedName` is a SkinStagedName.
[[nodiscard]] bool IsSkinStagedName(std::string_view stagedName);

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
  // meshes (of `softSurfaces`) are created/colored by the MochiModelAsset at `shapeModelFile`
  // (which also forms part of the reuse key). `softBakeScale` / `softShapeTransform` are the
  // per-instance bake parameters the asset uses to (re)generate the rest surfaces;
  // `softVertexCount` / `softCollisionVertexCount` are the two meshes' node counts, used to detect
  // topology changes for the reuse check. No ordinary shape/render instance is loaded.
  bool isSoft = false;
  mochi::Real3 softBakeScale = {mochi::real(1), mochi::real(1), mochi::real(1)};
  mochi::TransformRT softShapeTransform = {};
  SoftSurfaces softSurfaces;
  int softVertexCount = 0;
  int softCollisionVertexCount = 0;
  // Skinned skin only: per-link WORLD transforms (Mochi space) in the GLB's joint order (== nested
  // link / link-declaration order), captured at build time. Populated for ANY articulated skin,
  // with a collision shape (then also `isSoft`), a render GLB, or both, so either can be posed.
  // SceneStage converts them to renderer space and makes them root-relative against
  // `worldTransform` to drive both the render model's joints and the collision wireframe's bones.
  // Simulation supplies fresh transforms via ApplySkinnedPose.
  std::vector<mochi::TransformRT> skinLinkWorldTransforms;
  // Skinned skin only: per-link WORLD transforms at the articulation's ZERO/rest (bind) joint
  // configuration, in the SAME order/frame as skinLinkWorldTransforms. This is the pose the skin
  // surface (.mochi.h5 / GLB) was authored at, so it is the bind reference used to build the
  // collision bones (bone_i = current_i * inverse(bind_i)). It is pose-independent (does NOT latch
  // to the current editor pose), mirroring the GLB's baked inverse-bind matrices.
  std::vector<mochi::TransformRT> skinBindLinkWorldTransforms;
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
  // bound. With hideSkinCoveredLinks, the links the bot's skin stands in for are staged hidden:
  // where the skin's render model is drawn, the links it hides (see
  // superdex::robotics::ComputeSkinHiddenLinks); where only its collision surface is drawn, the
  // links whose collision it covers (see superdex::robotics::ComputeSkinCoveredLinks).
  // Returns true if the staged objects were rebuilt (pointers invalidated), false if reused.
  bool StageBot(
      superdex::robotics::BotPrefab const& prefab,
      StageType stageType,
      bool hideSkinCoveredLinks);

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
  // render scene at rest pose. Spawnable-prefab declarations are not instances and are omitted
  // until StageBotTask is called. The base scene and each bot are resolved internally via the
  // AssetManager. The staged-actor order is the base scene (prefab::AddToScene order) followed by
  // each bot's links, then its skin if it has one, in declaration order — matching
  // superdex::robotics::LoadBotScene so that index-based ApplyWorldTransforms aligns with the
  // simulated actors (a skin's row carries its articulation's root transform). The links each skin
  // stands in for are staged hidden, as StageBot does with hideSkinCoveredLinks. No-op if no scene
  // is bound.
  // Returns true if the staged objects were rebuilt (pointers invalidated), false if reused.
  bool StageBotScene(superdex::robotics::BotScenePrefab const& scene, StageType stageType);

  // Stage a bot scene with the declared spawnable-prefab previews replaced by the concrete object
  // instances from a loaded task. An empty task intentionally stages no spawnables.
  bool StageBotTask(
      superdex::robotics::BotScenePrefab const& scene,
      mochi::Span<ResolvedTaskSpawn const> taskSpawns,
      StageType stageType);
#endif

  // Change which already-staged representation is visible without rebuilding or restoring actors
  // to their default pose. Used for view changes during simulation.
  void SetRepresentationMode(StageType stageType);

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

  // Apply per-soft-actor deforming-mesh updates (from simulation), matched by staged index or name
  // (see SoftMeshUpdate). For each update
  // both dynamic meshes (solid + wireframe) are deformed and the actor's world transform is set.
  // Unknown names and non-soft actors are ignored; a vertex-count mismatch skips that mesh.
  void ApplySoftMeshUpdates(
      mochi::Span<SoftMeshUpdate const> updates,
      mochi::CoordinateSpaceConverter const& converter);

  // Apply per-skin joint poses (from simulation or editor FK), matched by staged index or name (see
  // SkinnedPoseUpdate). For each update the
  // named skinned actor's render model has its joints posed (making the link world transforms
  // root-relative against the actor's current world transform, then converting to renderer space).
  // Unknown names and actors without a skinned render model are ignored. Call after
  // ApplyWorldTransforms / ApplySoftMeshUpdates so the actor's world transform (the skin root) is
  // current.
  void ApplySkinnedPose(
      mochi::Span<SkinnedPoseUpdate const> updates,
      mochi::CoordinateSpaceConverter const& converter);

  // Get the ordered list of staged actors.
  StagedActors const& GetActors() const;

  // The number of staged actors.
  int GetNumActors() const;

  // Install either the simulation-driven or the GPU-skinned surface in a skinned skin's shape
  // slot, keeping the other hidden. No-op for actors that do not have both.
  void SetSimSurfaceActive(StagedActor& actor, bool simActive);
  // Build a skinned skin's simulation surface if it does not have one yet. No-op for other actors.
  void EnsureSimSurface(StagedActor& actor, mochi::CoordinateSpaceConverter const& converter);
  // Reinstall the GPU-skinned surface and tear the simulation one down. No-op for other actors.
  void DestroySimSurface(StagedActor& actor);
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
  // The staged actor a per-frame update targets: the one at `stagedIndex` if it still has `name`,
  // else -1; by name when `stagedIndex` is negative.
  int FindUpdateTarget(std::string_view name, int stagedIndex) const;

  // Apply a list of staging requests: reuse existing staged objects when the actor set (by name)
  // and active model files are unchanged, otherwise rebuild. Shared by StageBot, StagePrefab, and
  // StageBotScene.
  // Returns true if the staged objects were rebuilt (existing SceneObjects destroyed and recreated,
  // invalidating external pointers); false if existing objects were reused.
  bool ApplyStageRequests(std::vector<StageRequest> const& requests, StageType stageType);

  // Shows `actor`'s SDF surface in place of its collision model when the stage type asks for it
  // and the surface is built, creating the instance on first use; otherwise hides it. Updates the
  // selectable alias; the caller applies the transforms.
  void ApplySdfVisibility(StagedActor& actor);

  // Under StageType::SdfFallbackToMochiModel, reconstructs the SDF surfaces the staged collision
  // models have but have not built yet, as a background task batch with a progress modal, then
  // shows them in place of their collision models (no restage needed). A no-op when nothing is
  // missing or another batch is running.
  void BuildMissingSdfSurfaces();

#if MOCHI_INTERNAL
  bool StageBotSceneImpl(
      superdex::robotics::BotScenePrefab const& scene,
      mochi::Span<ResolvedTaskSpawn const> const* taskSpawns,
      StageType stageType);
#endif

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
  // The stage type of the last staging call.
  StageType _stageType = StageType::RenderModelFallbackToMochiModel;
  // Expires with the stage, so a background SDF batch finishing after it is gone skips it.
  std::shared_ptr<bool> _lifetime = std::make_shared<bool>(true);
};

} // namespace superdex::studio
