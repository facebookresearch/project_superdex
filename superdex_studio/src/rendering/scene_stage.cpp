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

#include "rendering/scene_stage.h"

#include "rendering/soft_render_link.h"

#include <mochi_renderer/deformable_model_instance.h>
#include <mochi_renderer/mesh.h>
#include <mochi_renderer/resource.h>
#include <mochi_renderer/resource_manager.h>
#include <mochi_renderer/scene.h>
#include <mochi_renderer/scene_object.h>
#include <mochi_renderer/type_conversions.h>
#include <mochi_renderer/utils.h>

#include "app/app.h"

#include "assets/asset_manager.h"
#include "assets/bot_asset.h"
#include "assets/mochi_model_asset.h"
#include "assets/mochi_prefab_asset.h"
#include "assets/render_model_asset.h"
#include "core/async_task.h"
#include "ui/imgui_widgets.h" // HashStringToColor

#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/vmatrix.h>

#include <mochi_physics/cpp_api/mochi_actor.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>

using mochi::operator""_r;
using namespace mochi_renderer;

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// HELPERS
//--------------------------------------------------------------------------------------------------

// Per-articulated-actor forward-kinematics result. `transforms` and `names` are parallel arrays in
// internal nested-link-actor order (i.e GetNestedLinkActors / GetArticulatedLinkTransforms order)
struct ArticulatedData {
  std::vector<std::string> names;
  std::vector<mochi::TransformRT> transforms;
  // Link world transforms at the ZERO/rest (bind) joint configuration, same order as `transforms`.
  std::vector<mochi::TransformRT> bindTransforms;
};

// True if the prefab (or any nested prefab) contains at least one articulated actor.
static bool PrefabHasArticulated(mochi::prefab::ScenePrefab const& prefab) {
  if (!prefab.actors.articulated.empty()) {
    return true;
  }
  for (auto const& nested : prefab.prefabs) {
    if (nested.prefab && PrefabHasArticulated(*nested.prefab)) {
      return true;
    }
  }
  return false;
}

// Build a copy of the prefab suitable for cheap articulated forward kinematics: rigid/soft actors
// and the contact settings that may name them are dropped, and articulated link shapes are cleared
// (turning links into shapeless dummy links), while articulated transforms, joints, and
// nested-prefab structure are preserved. Nested prefabs
// are deep-cloned so the original (shared) nested prefabs are not mutated.
static mochi::prefab::ScenePrefab StripArticulated(mochi::prefab::ScenePrefab const& src) {
  mochi::prefab::ScenePrefab out = src;
  out.actors.rigid.clear();
  out.actors.soft.clear();
  out.actors.softSkinned.clear();
  out.contactFilter.reset();
  out.contactPairParamsOverrides.reset();
  for (auto& art : out.actors.articulated) {
    for (auto& link : art.links) {
      link.shapeFile.clear();
      // Also clear the loaded shape handle: after a simulation run EnsureFullyLoaded populates it
      // on the live prefab, and a populated handle would make AddToScene build a real collider
      // (BVH) instead of a cheap shapeless dummy link.
      link.shape = {};
      link.colliderType = mochi::ColliderType::None;
      // Mass properties only apply with a shape (Mochi warns about partial ones without one).
      link.mass.reset();
      link.density.reset();
      link.centerOfMass.reset();
      link.momentOfInertia.reset();
    }
    // Drop the skin outright rather than blanking its shape: this scratch scene exists only to read
    // link transforms, and skin params without a shape are rejected at actor creation.
    art.skin.reset();
  }
  for (auto& nested : out.prefabs) {
    if (nested.prefab) {
      nested.prefab =
          std::make_shared<mochi::prefab::ScenePrefab>(StripArticulated(*nested.prefab));
    }
  }
  return out;
}

// Both the render-model and shape representations are staged whenever they exist; StageType only
// decides which is visible. In fallback mode the shape shows only for links lacking a render model;
// the "Only" modes force one representation visible and the other hidden. Staging both (hiding one)
// keeps StageType switches to visibility toggles and lets the hidden representation still
// contribute to the scene AABB / ground plane.
static void ComputeStageVisibility(
    bool hasRender,
    bool hasShape,
    StageType stageType,
    bool& outRenderVisible,
    bool& outShapeVisible) {
  outRenderVisible = false;
  outShapeVisible = false;
  switch (stageType) {
    case StageType::RenderModelOnly:
      outRenderVisible = hasRender;
      break;
    case StageType::MochiModelOnly:
    case StageType::SdfFallbackToMochiModel:
      // SDF mode stages the collision model as the fallback; ApplySdfVisibility swaps in the SDF
      // surface where there is one.
      outShapeVisible = hasShape;
      break;
    case StageType::RenderModelFallbackToMochiModel:
      outRenderVisible = hasRender;
      outShapeVisible = hasShape && !hasRender;
      break;
  }
}

static void ApplyStageVisibility(StagedActor& actor, bool renderVisible, bool shapeVisible) {
  actor.render.visible = renderVisible;
  actor.shape.visible = shapeVisible;
  actor.sdf.visible = false;
  if (actor.render.sceneObject) {
    actor.render.sceneObject->SetVisible(renderVisible);
  }
  if (actor.shape.sceneObject) {
    actor.shape.sceneObject->SetVisible(shapeVisible);
  }
  actor.sceneObject = (renderVisible && actor.render.sceneObject) ? actor.render.sceneObject
      : (shapeVisible && actor.shape.sceneObject)                 ? actor.shape.sceneObject
                                                                  : nullptr;
}

// Resolve a prefab-relative model reference to a full path; empty when the reference is empty or
// does not resolve.
static mochi::Path ResolvePrefabModelPath(
    mochi::DynamicString const& modelFile,
    std::string const& rootPath,
    std::string_view prefabSource) {
  if (modelFile.empty()) {
    return {};
  }
  auto resolved = mochi::prefab::GetPrefabFullPath(modelFile, rootPath, prefabSource);
  return resolved.empty() ? mochi::Path{} : mochi::Path(resolved);
}

// Whether a soft actor of @p asset can draw the render model at @p renderModelFile deformed by its
// visual mesh, given where the two are placed in the actor's frame.
static RenderModelLink LinkSoftRenderModel(
    SuperDexStudio& studio,
    MochiModelAsset const& asset,
    mochi::Path const& renderModelFile,
    ModelPlacement const& render,
    ModelPlacement const& shape) {
  auto* const renderAsset = studio.GetAssetManager().LoadRenderModelAsset(renderModelFile);
  if (renderAsset == nullptr) {
    return {.reason = "The render model could not be loaded."};
  }
  mochi::Error error;
  std::optional<mochi_renderer::DeformableMeshView> const renderMesh =
      renderAsset->GetDeformableMesh(error);
  if (!renderMesh.has_value()) {
    return {.reason = error.GetDescription()};
  }
  std::optional<mochi::MeshData> const& visualMesh = asset.GetModelData().visualMesh;
  return CheckRenderModelLink(
      *renderMesh,
      visualMesh.has_value() ? mochi::MeshDataView(*visualMesh) : mochi::MeshDataView{},
      render,
      shape);
}

// Records why a soft actor's render model is not drawn (empty when it is or there is none). Warns
// once as the actor starts falling back, not on every restage: an offset being dragged restages
// repeatedly, and the reason then names a changing distance.
static void
SetSoftRenderModelNotice(StagedActor& actor, std::string const& name, std::string notice) {
  if (actor.softRenderModelNotice.empty() && !notice.empty()) {
    MOCHI_LOG_WARNING(
        "Soft actor '%s' draws its Mochi model instead of its render model. %s",
        name.c_str(),
        notice.c_str());
  }
  actor.softRenderModelNotice = std::move(notice);
}

// Calls @p fn on a soft actor's linked render model, if any, and on its highlight stand-in: a
// separate instance, which would otherwise freeze at the pose it was cloned in.
template <typename Fn>
static void ForEachSoftRenderModel(StagedActor const& actor, Fn&& fn) {
  if (!actor.softRenderModel) {
    return;
  }
  fn(*actor.softRenderModel);
  if (actor.render.highlightClone) {
    if (auto* clone = actor.render.highlightClone->AsDeformableModelInstance()) {
      fn(*clone);
    }
  }
}

// Combine a prefab path and a name the same way prefab::AddToScene does, so staged actor names
// match the physics actor names (empty segments are collapsed rather than producing stray slashes).
static std::string CombineNames(std::string const& path, std::string const& name) {
  if (path.empty()) {
    return name;
  }
  if (name.empty()) {
    return path;
  }
  return path + "/" + name;
}

static constexpr std::string_view kSkinStagedNameSuffix = " (Skin)";

std::string SkinStagedName(std::string_view articulatedActorName) {
  return std::string(articulatedActorName) + std::string(kSkinStagedNameSuffix);
}

bool IsSkinStagedName(std::string_view stagedName) {
  return stagedName.ends_with(kSkinStagedNameSuffix);
}

// Recursively append staging requests for a prefab level, mirroring prefab::AddToScene's creation
// order: nested prefabs (recursively), then articulated actors (expanded to nested link actors),
// then rigid actors, then soft actors. `worldFromPrefab` is the composed transform of the enclosing
// prefab chain; `artFk`/`artCursor` supply per-articulated-actor link world transforms (FK results)
// in the same traversal order. Soft actor rest meshes are built on demand from their (baked)
// shapes.
static void BuildPrefabRequests(
    SuperDexStudio* studio,
    mochi::prefab::ScenePrefab const& prefab,
    std::string const& rootPath,
    std::string_view prefabSource,
    mochi::TransformSRT const& worldFromPrefab,
    StageType stageType,
    std::vector<ArticulatedData> const& artFk,
    int& artCursor,
    std::string const& prefabName,
    int topLevelNestedIndex,
    std::vector<StageRequest>& out) {
  using namespace mochi;

  // 1) Nested prefabs first.
  for (int p = 0; p < isize(prefab.prefabs); ++p) {
    auto const& nested = prefab.prefabs[p];
    if (!nested.prefab) {
      continue;
    }
    TransformSRT const prefabFromNested(
        nested.scale, Normalize(nested.rotation), nested.translation);
    TransformSRT const worldFromNested = worldFromPrefab * prefabFromNested;
    std::string_view const nestedSource = nested.prefab->sourceFilePath.has_value()
        ? std::string_view(*nested.prefab->sourceFilePath)
        : std::string_view();
    std::string const nestedName = CombineNames(prefabName, std::string(nested.name));
    // Attribute every actor in this subtree to the top-level nested reference so viewport picks and
    // hierarchy hovers map back to it. Deeper nesting keeps the top-level ancestor's index.
    int const childNestedIndex = topLevelNestedIndex < 0 ? p : topLevelNestedIndex;
    BuildPrefabRequests(
        studio,
        *nested.prefab,
        rootPath,
        nestedSource,
        worldFromNested,
        stageType,
        artFk,
        artCursor,
        nestedName,
        childNestedIndex,
        out);
  }

  real const prefabScale = worldFromPrefab.GetScale();

  // Tag staged objects with a source identity so the prefab editor can map them back: top-level
  // rigid/articulated actors to themselves, and any actor inside a nested prefab to that top-level
  // nested reference.
  bool const topLevel = topLevelNestedIndex < 0;

  // 2) Articulated actors of this level, expanded to nested link actors (the actor itself has no
  // transform to stage). FK link transforms are already fully composed (computed on a scratch scene
  // with the prefab placed at its world transform), so they are used directly as world transforms.
  // They come in internal nested-link-actor order, which may differ from the prefab's link
  // declaration order, so each link is paired with its declaration entry by name to fetch the right
  // model/offset/scale.
  for (int a = 0; a < isize(prefab.actors.articulated); ++a) {
    auto const& art = prefab.actors.articulated[a];
    ArticulatedData const& fk = artFk[artCursor++];
    real const effectiveScale = art.scale * prefabScale;
    std::unordered_map<std::string, int> declIndexByName;
    for (int j = 0; j < isize(art.links); ++j) {
      declIndexByName[std::string(art.links[j].name)] = j;
    }
    for (int i = 0; i < isize(fk.transforms); ++i) {
      StageRequest req;
      req.name = fk.names[i];
      req.worldTransform = fk.transforms[i];
      // The nested link actor name is "<articulatedActor>/<linkName>"; match its bare link name.
      std::string const& fullName = fk.names[i];
      auto const slash = fullName.find_last_of('/');
      std::string const bareName =
          slash == std::string::npos ? fullName : fullName.substr(slash + 1);
      auto const it = declIndexByName.find(bareName);
      if (it != declIndexByName.end()) {
        auto const& link = art.links[it->second];
        // Stage both representations (either may be absent); the articulated actor's scale folds
        // into each model scale (uniform, so model-frame vs actor-frame is equivalent here).
        req.renderModelFile = ResolvePrefabModelPath(link.renderModelFile, rootPath, prefabSource);
        if (!req.renderModelFile.IsEmpty()) {
          req.renderModelTransform =
              TransformRT(link.renderModelRotation, link.renderModelTranslation);
          req.renderModelScale = effectiveScale * link.renderModelScale;
        }
        req.shapeModelFile = ResolvePrefabModelPath(link.shapeFile, rootPath, prefabSource);
        if (!req.shapeModelFile.IsEmpty()) {
          req.shapeModelTransform = TransformRT(link.shapeRotation, link.shapeTranslation);
          req.shapeModelScale = effectiveScale * link.shapeScale;
        }
        if (topLevel) {
          req.source = StagedActorSource::Articulated;
          req.sourceActorIndex = a;
          req.sourceLinkIndex = it->second;
        } else {
          req.source = StagedActorSource::NestedPrefab;
          req.sourceActorIndex = topLevelNestedIndex;
        }
      }
      ComputeStageVisibility(
          !req.renderModelFile.IsEmpty(),
          !req.shapeModelFile.IsEmpty(),
          stageType,
          req.renderVisible,
          req.shapeVisible);
      out.push_back(std::move(req));
    }

    // Articulated skin: folded into the compound entity (it shares the actor's handle and is not a
    // nested link), so it is staged as a standalone deforming mesh named
    // SkinStagedName(<articulated actor name>). The per-step surface query (editors) and force-drag
    // resolution (PhysicsDragController) find it by that name; its deformed surface is pushed each
    // step via ApplySoftMeshUpdates just like a soft actor. The articulated actor name is the
    // shared prefix of its nested link names (fk.names), which matches the runtime actor->GetName()
    // the query uses.
    if (art.skin.has_value() && !art.skin->shapeFile.empty() && !fk.names.empty()) {
      mochi::Path const skinShapePath =
          ResolvePrefabModelPath(art.skin->shapeFile, rootPath, prefabSource);
      auto* asset = studio->GetAssetManager().LoadMochiModelAsset(skinShapePath);
      int const skinVertexCount = asset ? asset->GetSurfaceVertexCount(SoftSurface::Surface) : 0;
      if (skinVertexCount > 0) {
        std::string const& firstLink = fk.names.front();
        auto const slash = firstLink.find_last_of('/');
        std::string const artName =
            slash == std::string::npos ? firstLink : firstLink.substr(0, slash);
        StageRequest req;
        req.name = SkinStagedName(artName);
        // The rest surface is authored in the frame of the skin's origin link (bone 0) at the
        // zero/bind pose -- the same convention BuildBotRequests roots a composed skin at -- so
        // place it there rather than at the articulation root, which is generally somewhere else
        // entirely. The bind root also cancels out of the bone matrices (see ApplySkinnedJointPose:
        // bone_j = bindRoot^-1 * current_j * inverse(bindRoot^-1 * bind_j)), so the surface is
        // undeformed at rest and correctly posed otherwise. Falls back to the actor's world
        // transform if forward kinematics produced no bind pose. Overwritten each step by
        // ApplySoftMeshUpdates once simulating.
        TransformSRT const prefabFromActor(1_r, Normalize(art.rotation), art.translation);
        req.worldTransform = fk.bindTransforms.empty()
            ? (worldFromPrefab * prefabFromActor).GetTransformRT()
            : fk.bindTransforms.front();
        req.worldScale = Real3{1_r, 1_r, 1_r};
        req.shapeModelFile = skinShapePath;
        req.isSoft = true;
        req.softVertexCount = skinVertexCount;
        req.softCollisionVertexCount = skinVertexCount;
        // The articulated actor's (uniform) scale bakes into the rest surface; no shape offset is
        // supported for skins yet (see ArticulatedSkinPrefab).
        req.softBakeScale = Real3{effectiveScale, effectiveScale, effectiveScale};
        req.softShapeTransform = TransformRT{};
        // Populate link transforms for ANY articulated skin so the collision skin can be posed
        // independently of the render model. fk.transforms are the per-link world transforms in
        // nested-link (== GLB joint) order; bindTransforms are the same at the zero/rest pose.
        req.skinLinkWorldTransforms.assign(fk.transforms.begin(), fk.transforms.end());
        req.skinBindLinkWorldTransforms.assign(fk.bindTransforms.begin(), fk.bindTransforms.end());
        // Optional GPU-skinned render model: when present it renders in the Render-Model view (the
        // collision surface still stages as the shape rep) and is posed each frame from the link
        // FK transforms. The GLB is authored in the articulation root frame, so its root is placed
        // at the actor world transform with no model offset (offsets would break the skin's
        // inverse-bind math) and only the actor's uniform scale (see ApplySkinnedJointPose).
        mochi::Path const skinRenderPath =
            ResolvePrefabModelPath(art.skin->renderModelFile, rootPath, prefabSource);
        if (!skinRenderPath.IsEmpty()) {
          req.renderModelFile = skinRenderPath;
          req.renderModelScale = Real3{effectiveScale, effectiveScale, effectiveScale};
        }
        // The collision wireframe (shape slot) always exists; the render GLB is optional. Route
        // visibility on the real render presence so fallback mode shows the collision when no GLB.
        ComputeStageVisibility(
            /*hasRender=*/!req.renderModelFile.IsEmpty(),
            /*hasShape=*/true,
            stageType,
            req.renderVisible,
            req.shapeVisible);
        if (topLevel) {
          req.source = StagedActorSource::Articulated;
          req.sourceActorIndex = a;
        } else {
          req.source = StagedActorSource::NestedPrefab;
          req.sourceActorIndex = topLevelNestedIndex;
        }
        out.push_back(std::move(req));
      }
    }
  }

  // 3) Rigid actors of this level. A rigid actor's rest pose is just its authored transform, so it
  // is computed analytically.
  for (int i = 0; i < isize(prefab.actors.rigid); ++i) {
    auto const& actor = prefab.actors.rigid[i];
    TransformSRT const prefabFromActor(1_r, Normalize(actor.rotation), actor.translation);
    TransformSRT const worldFromActor = worldFromPrefab * prefabFromActor;

    StageRequest req;
    req.name = CombineNames(
        prefabName, actor.name.empty() ? ("Rigid_" + std::to_string(i)) : std::string(actor.name));
    req.worldTransform = worldFromActor.GetTransformRT();
    // Stage both representations (either may be absent). The actor's own (possibly non-uniform)
    // scale is applied in the actor frame (worldScale); each model's own scale in its model frame.
    // Keeping them separate avoids axis swizzle when a non-uniform actor scale meets a rotated
    // model offset.
    req.worldScale = actor.scale * prefabScale;
    req.renderModelFile = ResolvePrefabModelPath(actor.renderModelFile, rootPath, prefabSource);
    if (!req.renderModelFile.IsEmpty()) {
      req.renderModelTransform =
          TransformRT(actor.renderModelRotation, actor.renderModelTranslation);
      req.renderModelScale = actor.renderModelScale;
    }
    req.shapeModelFile = ResolvePrefabModelPath(actor.shapeFile, rootPath, prefabSource);
    if (!req.shapeModelFile.IsEmpty()) {
      // RigidActorPrefab has no per-shape scale; only the shape offset applies.
      req.shapeModelTransform = TransformRT(actor.shapeRotation, actor.shapeTranslation);
    }
    ComputeStageVisibility(
        !req.renderModelFile.IsEmpty(),
        !req.shapeModelFile.IsEmpty(),
        stageType,
        req.renderVisible,
        req.shapeVisible);
    if (topLevel) {
      req.source = StagedActorSource::Rigid;
      req.sourceActorIndex = i;
    } else {
      req.source = StagedActorSource::NestedPrefab;
      req.sourceActorIndex = topLevelNestedIndex;
    }
    out.push_back(std::move(req));
  }

  // 4) Soft actors of this level. A soft actor's rest pose is its authored transform; its scale and
  // shape offset are baked into the surface geometry (via the shape), so worldScale stays identity.
  // The rest-pose surface is generated by the MochiModelAsset from the per-instance bake scale +
  // shape offset (deferred to stage time). Both a solid and a wireframe dynamic mesh are staged
  // (solid in the render slot, wireframe in the shape slot), so the 1/2/3 StageType toggle switches
  // between them like it does for rigid actors. ChooseSoftSurfaces picks each one's mesh.
  real const softShapeScale = worldFromPrefab.GetScale();
  for (int i = 0; i < isize(prefab.actors.soft); ++i) {
    auto const& actor = prefab.actors.soft[i];
    // Soft actors without a shape produce no geometry; skip them.
    if (actor.shapeFile.empty()) {
      continue;
    }
    mochi::Path const shapePath = ResolvePrefabModelPath(actor.shapeFile, rootPath, prefabSource);
    auto* asset = studio->GetAssetManager().LoadMochiModelAsset(shapePath);
    if (!asset) {
      continue;
    }
    // Skip non-tetrahedral models (no soft surface).
    if (asset->GetSoftSurfaceVertexCount() <= 0) {
      continue;
    }
    TransformSRT const prefabFromActor(1_r, Normalize(actor.rotation), actor.translation);
    TransformSRT const worldFromActor = worldFromPrefab * prefabFromActor;

    StageRequest req;
    req.name = CombineNames(
        prefabName, actor.name.empty() ? ("Soft_" + std::to_string(i)) : std::string(actor.name));
    req.worldTransform = worldFromActor.GetTransformRT();
    // Scale is baked into the mesh geometry, so the actor-frame scale is identity.
    req.worldScale = Real3{1_r, 1_r, 1_r};
    // Carry the resolved shape path as part of the reuse key and to look up the asset that creates
    // and colors the dynamic meshes.
    req.shapeModelFile = shapePath;
    req.isSoft = true;
    // The node counts seed the reuse check.
    req.softSurfaces = asset->GetSoftSurfaces(actor.useContactSkin);
    req.softVertexCount = asset->GetSurfaceVertexCount(req.softSurfaces.rendered);
    req.softCollisionVertexCount = asset->GetSurfaceVertexCount(req.softSurfaces.collision);
    // Per-instance bake: scale then the shape offset, matching mochi::model_utils::BakeTransform
    // (rt * diag(scale)). The asset re-bakes from these on create/refresh/reset.
    req.softBakeScale = actor.scale * softShapeScale;
    req.softShapeTransform =
        TransformRT{actor.shapeRotation, actor.shapeTranslation * softShapeScale};
    // A render model is only drawn deformed by the visual mesh it is embedded as; otherwise the
    // solid stands in and the reason is shown in the actor's details.
    req.renderModelFile = ResolvePrefabModelPath(actor.renderModelFile, rootPath, prefabSource);
    if (!req.renderModelFile.IsEmpty()) {
      RenderModelLink const link = LinkSoftRenderModel(
          *studio,
          *asset,
          req.renderModelFile,
          ModelPlacement{
              .scale = actor.renderModelScale * softShapeScale,
              .transform =
                  TransformRT{
                      actor.renderModelRotation, actor.renderModelTranslation * softShapeScale}},
          ModelPlacement{.scale = req.softBakeScale, .transform = req.softShapeTransform});
      req.softRenderModelLinked = link.linked;
      req.softRenderModelNotice = link.reason;
    }
    // Solid (render slot) and wireframe (shape slot) are both staged; StageType picks which shows.
    ComputeStageVisibility(
        /*hasRender=*/true, /*hasShape=*/true, stageType, req.renderVisible, req.shapeVisible);
    if (topLevel) {
      req.source = StagedActorSource::Soft;
      req.sourceActorIndex = i;
    } else {
      req.source = StagedActorSource::NestedPrefab;
      req.sourceActorIndex = topLevelNestedIndex;
    }
    out.push_back(std::move(req));
  }

  // Soft-skinned actors are not yet supported and will not render.
  if (!prefab.actors.softSkinned.empty()) {
    MOCHI_LOG_ERROR_ONCE(
        "SceneStage: soft-skinned actors are not yet supported and will not render.");
  }
}

// Compute per-articulated-actor nested-link world transforms (and link actor names) via a stripped
// copy of the prefab added to `scene` (a scratch scene) with `params`; the transforms are in the
// world frame `params` places the prefab in (identity by default). Output is ordered to match the
// depth-first articulated-actor traversal (nested prefabs first, then each level's articulated
// actors). No-op (leaves `out` empty) when the prefab contains no articulated actors.
static void ComputeArticulatedLinkTransforms(
    mochi::Scene* scene,
    mochi::prefab::ScenePrefab const& prefab,
    std::vector<ArticulatedData>& out,
    mochi::prefab::PrefabParams const& params = {}) {
  using namespace mochi;
  out.clear();
  if (!PrefabHasArticulated(prefab)) {
    return;
  }
  mochi::prefab::ScenePrefab const stripped = StripArticulated(prefab);
  ErrorLog e;
  auto const result = mochi::prefab::AddToScene(stripped, scene, params, e);
  if (!e.IsOK()) {
    MOCHI_LOG_ERROR("StagePrefab: failed to compute articulated forward kinematics.");
    return;
  }
  auto const articulatedActors = result.Filter(ActorType::Articulated);
  out.resize(articulatedActors.size());
  for (int i = 0; i < isize(articulatedActors); ++i) {
    Actor* actor = articulatedActors[i];
    auto const links = actor->GetNestedLinkActors(e);
    out[i].transforms.resize(links.size());
    actor->GetArticulatedLinkTransforms(MakeSpan(out[i].transforms), e);
    // Capture the ZERO/rest (bind) link transforms: zero all joint DoFs, re-read, then restore.
    // This is pose-independent and matches the pose the skin surface was authored at. Captured for
    // every articulated actor: these are throwaway scratch actors, and the skin was stripped from
    // this scene, so there is nothing here to tell a skinned actor from an unskinned one. Left
    // empty when the pose cannot be read, in which case the skin staging path degrades to no bind
    // reference rather than using garbage.
    {
      int const numDofs = actor->GetNumDofs();
      std::vector<real> savedPose(numDofs);
      ErrorLog poseError;
      actor->GetArticulatedPose(MakeSpan(savedPose), poseError);
      if (poseError.IsOK()) {
        out[i].bindTransforms.resize(links.size());
        std::vector<real> const zeros(numDofs, 0_r);
        actor->SetArticulatedPoseFromJoints(MakeConstSpan(zeros), e);
        actor->GetArticulatedLinkTransforms(MakeSpan(out[i].bindTransforms), e);
        // Restore only what was actually read; otherwise savedPose is uninitialized.
        actor->SetArticulatedPoseFromJoints(MakeConstSpan(savedPose), e);
      } else {
        MOCHI_LOG_ERROR_ONCE(
            "SceneStage: could not read an articulated actor's pose to capture its bind "
            "transforms; any skin on it will not deform.");
      }
    }
    out[i].names.resize(links.size());
    for (int j = 0; j < isize(links); ++j) {
      Actor const* linkActor = scene->GetActor(links[j]);
      out[i].names[j] = linkActor ? linkActor->GetName() : "";
    }
  }
  // Remove the temporary FK actors so the scratch scene does not accumulate state across restages.
  for (auto* actor : result.actors) {
    scene->DestroyActor(actor->GetHandle());
  }
}

// Load the asset for `modelFile` and create a fresh render instance for it. `isRenderModel` selects
// between a RenderModelAsset (true) and a MochiModelAsset (false). Returns null when the file is
// empty or the asset fails to load. Shared by StageBot and StagePrefab.
static std::unique_ptr<SceneObject>
CreateRenderInstance(AssetManager& manager, mochi::Path const& modelFile, bool isRenderModel) {
  if (modelFile.IsEmpty()) {
    return nullptr;
  }
  if (isRenderModel) {
    if (auto* asset = manager.LoadRenderModelAsset(modelFile)) {
      return asset->GetRenderModelInstance();
    }
  } else {
    if (auto* asset = manager.LoadMochiModelAsset(modelFile)) {
      return asset->GetRenderModelInstance();
    }
  }
  return nullptr;
}

// Write a world transform into an actor's cache and push it to its render object (if any).
static void SetStagedActorWorldTransform(
    StagedActor& actor,
    mochi::TransformRT const& worldTransform,
    mochi::CoordinateSpaceConverter const& converter) {
  using namespace mochi;
  actor.worldTransform = worldTransform;
  // Both representations share the actor's world transform and actor-frame scale but keep their own
  // model offset/scale. The rendered transform is worldTransform * actorScale * modelOffset *
  // modelScale: the actor's (possibly non-uniform) scale must sit between the actor and model
  // rotations so it scales in the actor frame rather than the rotated model frame (which would
  // swizzle axes). That product is not a single rotation/translation/scale, so compose it as a
  // matrix and decompose it back into the (transform, scale) pair the renderer accepts.
  auto localTransformFor = [&](StagedActor::Instance const& instance) {
    VMatrix4x4r const local = Dot4x4(
        ToVMatrix4x4(worldTransform),
        Dot4x4(
            VDiagonalMatrix<4>(ToSimd(actor.worldScale, 1_r)),
            Dot4x4(
                ToVMatrix4x4(instance.modelTransform),
                VDiagonalMatrix<4>(ToSimd(instance.modelScale, 1_r)))));
    return DecomposeMatrixTransform(local);
  };
  auto applyTo = [&](StagedActor::Instance& instance) {
    if (!instance.sceneObject) {
      return;
    }
    auto const [scale, rt] = localTransformFor(instance);
    instance.sceneObject->SetLocalTransform(rt, scale, &converter);
  };
  applyTo(actor.render);
  applyTo(actor.shape);
  applyTo(actor.sdf);
  // A skinned skin's other surface is not held in a slot, so it would keep the identity transform
  // and sit at the origin. A hidden object still contributes to the scene AABB, so that stray
  // surface would drag the ground plane, the camera framing and the asset thumbnail. It is the same
  // baked surface as the installed one and shares the shape slot's model offset/scale.
  if (actor.shapeSkinnedSurface != nullptr || actor.shapeSimSurface != nullptr) {
    auto const [scale, rt] = localTransformFor(actor.shape);
    for (auto* surface : {actor.shapeSkinnedSurface, actor.shapeSimSurface}) {
      if (surface != nullptr && surface != actor.shape.sceneObject) {
        surface->SetLocalTransform(rt, scale, &converter);
      }
    }
  }
}

// Convert a Mochi TransformRT (already in renderer space) to a Filament mat4f, composing
// translation * rotation to match the bone matrices SetBoneMatrices expects.
static filament::math::mat4f ToFilamentMat4(mochi::TransformRT const& t) {
  return filament::math::mat4f::translation(mochi_renderer::ToFilament(t.GetTranslation())) *
      filament::math::mat4f{mochi_renderer::ToFilament(t.GetRotation())};
}

// Pose a skinned skin's render-model joints from per-link WORLD transforms (Mochi space, in GLB
// joint order): make them root-relative against the actor's current world transform (the skin
// root), convert to renderer space, and push them to the GPU-skinned render instance AND the
// GPU-skinned collision wireframe. No-op unless the actor has a skinned render model and transforms
// were provided. `actor.worldTransform` must already be set to the frame the joints should be
// relative to.
static void ApplySkinnedJointPose(
    StagedActor& actor,
    mochi::Span<mochi::TransformRT const> linkWorldTransforms,
    mochi::CoordinateSpaceConverter const& converter) {
  using namespace mochi;
  if ((!actor.skinnedRenderObject && !actor.skinnedCollisionMesh) || linkWorldTransforms.empty()) {
    return;
  }
  TransformRT const rootInv = Invert(actor.worldTransform);
  std::vector<TransformRT> jointTransforms(linkWorldTransforms.size());
  for (int j = 0; j < isize(linkWorldTransforms); ++j) {
    jointTransforms[j] = converter.TransformToOutput(rootInv * linkWorldTransforms[j]);
  }
  // Build renderable-local bone matrices from the ARTICULATION's own current + rest link frames:
  // boneMatrix_j = currentRootRel_j * inverse(restRootRel_j). At rest they are identity (the baked
  // rest surface is shown). These never reference the render GLB's baked skeleton / inverse-bind
  // matrices, so a render GLB authored in a different bind convention than the articulation (e.g. a
  // DCC/Unreal export whose joint frames are rotated from the link frames) still deforms correctly.
  // The same matrices drive the GPU-skinned render GLB, its highlight clone, and the collision
  // wireframe.
  if (actor.skinRestBoneInverse.empty()) {
    return;
  }
  int const boneCount = std::min(isize(jointTransforms), isize(actor.skinRestBoneInverse));
  std::vector<filament::math::mat4f> boneMatrices(boneCount);
  for (int j = 0; j < boneCount; ++j) {
    boneMatrices[j] = ToFilamentMat4(jointTransforms[j]) * actor.skinRestBoneInverse[j];
  }
  auto const boneSpan = mochi::MakeConstSpan(boneMatrices);
  if (actor.skinnedRenderObject) {
    // A scaled articulation's link frames carry its uniform scale s, which the render GLB (authored
    // at scale 1) gets as its model scale, outside the bones. Its bones are then S^-1 * bone * S:
    // the same rotations, translations divided by s. The collision surface has s baked in, so it
    // keeps the bones as they are.
    real const renderScale = actor.render.modelScale[0];
    std::vector<filament::math::mat4f> renderBoneMatrices = boneMatrices;
    if (renderScale != 1_r) {
      auto const inverseScale = static_cast<float>(1_r / renderScale);
      for (auto& bone : renderBoneMatrices) {
        bone[3].xyz *= inverseScale;
      }
    }
    auto const renderBoneSpan = mochi::MakeConstSpan(renderBoneMatrices);
    actor.skinnedRenderObject->SetBoneMatrices(renderBoneSpan);
    // The selection/hover highlight clone is a separate skinned instance, so it must be driven too
    // or it freezes at the pose it was cloned in while the base animates.
    if (actor.render.highlightClone) {
      if (auto* clone = actor.render.highlightClone->AsSkinnedModelInstance()) {
        clone->SetBoneMatrices(renderBoneSpan);
      }
    }
  }
  if (actor.skinnedCollisionMesh) {
    actor.skinnedCollisionMesh->SetBoneMatrices(boneSpan);
  }
}

// Instantiate `prefab` as a shapeless temporary bot in `scene` (a scratch scene), read its
// default-pose link world transforms via Mochi forward kinematics, and append one staging request
// per link to `out`. Mirrors BuildPrefabRequests but for a single bot's articulated links. The link
// ordering follows GetNestedLinkActors, matching the per-step simulation extraction order.
static void BuildBotRequests(
    mochi::Scene* scene,
    SuperDexStudio* studio,
    superdex::robotics::BotPrefab const& prefab,
    StageType stageType,
    bool hideSkinCoveredLinks,
    std::vector<StageRequest>& out) {
  using namespace mochi;

  // A bot with no links (e.g. a mod bot whose base is not set yet) has nothing to stage, and
  // instantiating it would fail.
  if (prefab.links.empty()) {
    return;
  }

  // Strip shapes/sensors/colliders for cheap FK-only instantiation, and the mass properties that
  // only apply with a shape (Mochi warns about partial ones on a shapeless link).
  auto prefabCopy = prefab;
  for (auto& link : prefabCopy.links) {
    link.colliderType = mochi::ColliderType::None;
    link.shapeFile.clear();
    link.sensors.clear();
    link.mass.reset();
    link.density.reset();
    link.centerOfMass.reset();
    link.momentOfInertia.reset();
  }
  // The skin is not part of forward kinematics; drop it so the cheap FK-only bot does not load the
  // skin shape. It is staged separately below as a deforming dynamic mesh.
  prefabCopy.skin.reset();
  // Contact overrides are irrelevant to these shapeless FK-only links and a skin<->link override
  // still references the skin we just dropped (its party is the bot name), so it would fail to
  // resolve in CreateBot and log a spurious warning on every restage. Drop them.
  prefabCopy.contactOverrides.clear();
  ErrorLog e;
  auto* botsCtx = studio->GetRoboticsContext();
  // Staging re-runs on every edit, so an invalid intermediate bot (e.g. a duplicate name partway
  // through a rename) is expected here. The editor surfaces validation issues in its own UI, and
  // simulation reports them through its own error channel, so this stays out of the log console.
  Error createError;
  auto* bot = botsCtx->CreateBot(scene, prefabCopy, studio->GetBotLoader(), createError);
  if (!createError.IsOK() || !bot) {
    MOCHI_LOG_VERBOSE("StageBot: failed to create bot: %s", createError.GetDescription());
    return;
  }
  MOCHI_DEFER(superdex::robotics::DestroyBot(scene, bot));

  auto linkActors = bot->GetArticulatedActor()->GetNestedLinkActors(e);
  if (linkActors.size() != prefab.links.size()) {
    MOCHI_LOG_ERROR("Bot prefab and articulated actor link sizes do not match");
    return;
  }

  int const firstLinkRequest = isize(out);
  out.reserve(out.size() + prefab.links.size());
  // Per-link world transforms in nested-link order (== GLB joint order), captured for posing an
  // optional skinned render model below.
  std::vector<mochi::TransformRT> linkWorldTransforms;
  linkWorldTransforms.reserve(prefab.links.size());
  for (int i = 0; i < isize(prefab.links); ++i) {
    auto const& link = prefab.links[i];
    auto* linkActor = scene->GetActor(linkActors[i]);

    StageRequest req;
    req.name = linkActor->GetName();
    req.worldTransform = linkActor->GetRootTransform();
    linkWorldTransforms.push_back(linkActor->GetRootTransform());
    // Stage both representations (either may be absent); bot links reference model files directly.
    // The link world transforms already carry the bot scale; the link-local models scale with it.
    real const scale = prefab.scale;
    if (!link.renderModelFile.empty()) {
      req.renderModelFile = mochi::Path(link.renderModelFile);
      req.renderModelTransform =
          TransformRT(link.renderModelRotation, link.renderModelTranslation * scale);
      req.renderModelScale = link.renderModelScale * scale;
    }
    if (!link.shapeFile.empty()) {
      req.shapeModelFile = mochi::Path(link.shapeFile);
      req.shapeModelTransform = TransformRT(link.shapeRotation, link.shapeTranslation * scale);
      req.shapeModelScale = link.shapeScale * scale;
    }
    ComputeStageVisibility(
        !req.renderModelFile.IsEmpty(),
        !req.shapeModelFile.IsEmpty(),
        stageType,
        req.renderVisible,
        req.shapeVisible);
    out.push_back(std::move(req));
  }

  // Capture the ZERO/rest (bind) link world transforms in the SAME order/frame as
  // linkWorldTransforms: zero all joint DoFs and re-read each link. The bot is a throwaway created
  // for staging (destroyed by the MOCHI_DEFER above), so the pose need not be restored. This bind
  // reference is pose-independent, matching the pose the skin surface was authored at.
  std::vector<mochi::TransformRT> bindLinkWorldTransforms;
  bindLinkWorldTransforms.reserve(prefab.links.size());
  {
    auto* art = bot->GetArticulatedActor();
    std::vector<mochi::real> const zeros(art->GetNumDofs(), 0_r);
    art->SetArticulatedPoseFromJoints(mochi::MakeConstSpan(zeros), e);
    for (int i = 0; i < isize(prefab.links); ++i) {
      auto* linkActor = scene->GetActor(linkActors[i]);
      bindLinkWorldTransforms.push_back(linkActor->GetRootTransform());
    }
  }

  // Skin (optional). The skin is a deformable surface skinned over the links; it is staged through
  // the same deforming dynamic-mesh path as soft actors (a solid mesh in the render slot, a
  // wireframe in the shape slot), and updated each simulation step -- matched by staged index or
  // name -- via SceneStage::ApplySoftMeshUpdates from the articulated actor's skinned surface. The
  // staged name is the top-level articulated actor's name plus " (Skin)": the skin is that actor's
  // own surface (not a nested link), so SkinSimSync derives its name identically. It is staged
  // right after the bot's links, so editors that apply simulated transforms by index give it a row
  // of its own there (the articulation root, which its bone matrices cancel). Its rest surface is
  // rooted at skin bone 0's bind-pose frame with the bot scale baked in (see below). A skin may
  // have only a collision shape, only a render model, or both.
  if (prefab.skin.has_value()) {
    mochi::Path const skinShapePath(prefab.skin->shapeFile);
    auto* asset = skinShapePath.IsEmpty()
        ? nullptr
        : studio->GetAssetManager().LoadMochiModelAsset(skinShapePath);
    int const skinVertexCount = asset ? asset->GetSurfaceVertexCount(SoftSurface::Surface) : 0;
    bool const hasSkinShape = skinVertexCount > 0;
    if (hasSkinShape || !prefab.skin->renderModelFile.empty()) {
      StageRequest req;
      char const* const skinActorName = bot->GetArticulatedActor()->GetName();
      req.name = SkinStagedName(skinActorName ? skinActorName : "");
      // Root the rest surface at the skin's origin link (bone 0) at the zero/bind pose, the frame
      // the surface is authored in -- matching the composed-skin branch below and the prefab path.
      // For a natively skinned bot bone i == link i, so bone 0 is link 0. Every skinned bot
      // authored so far has its root link at the articulation root, making this identical to
      // worldFromRoot; it diverges only for a future skin whose origin link is offset from the
      // root, which is exactly the case worldFromRoot places wrongly.
      req.worldTransform =
          bindLinkWorldTransforms.empty() ? prefab.worldFromRoot : bindLinkWorldTransforms.front();
      req.worldScale = Real3{1_r, 1_r, 1_r};
      if (hasSkinShape) {
        req.shapeModelFile = skinShapePath;
        req.isSoft = true;
        req.softVertexCount = skinVertexCount;
        req.softCollisionVertexCount = skinVertexCount;
      }
      req.softBakeScale = Real3{prefab.scale, prefab.scale, prefab.scale};
      req.softShapeTransform = TransformRT{};
      // Populate link transforms for ANY articulated skin so the collision skin can be posed
      // independently of the render model (see the prefab path). Current + bind (zero/rest) poses.
      req.skinLinkWorldTransforms = linkWorldTransforms;
      req.skinBindLinkWorldTransforms = bindLinkWorldTransforms;
      // A composed (mod) bot adopts a sub-bot's skin whose bone indices (== GLB joint order) index
      // the SUB-bot's links, not the merged articulation's link order. The GPU-skinned collision
      // wireframe and render GLB carry those child-local bone indices / inverse-bind, so pose them
      // with per-link transforms reordered into skin-bone order, and root the rest surface at the
      // skin's origin link (skin bone 0) captured at the bind/zero pose -- the frame the skin
      // surface and inverse-bind were authored in. `_skinBoneLinks` is empty for a natively skinned
      // bot (bone i == link i), so that path keeps the identity ordering above.
      if (!prefab._skinBoneLinks.empty()) {
        std::unordered_map<std::string_view, int> linkIndexByName;
        linkIndexByName.reserve(prefab.links.size());
        for (int i = 0; i < isize(prefab.links); ++i) {
          linkIndexByName.emplace(std::string_view(prefab.links[i].name), i);
        }
        std::vector<mochi::TransformRT> boneCurrent;
        std::vector<mochi::TransformRT> boneBind;
        boneCurrent.reserve(prefab._skinBoneLinks.size());
        boneBind.reserve(prefab._skinBoneLinks.size());
        bool resolved = true;
        for (auto const& linkName : prefab._skinBoneLinks) {
          auto const it = linkIndexByName.find(std::string_view(linkName));
          if (it == linkIndexByName.end()) {
            resolved = false;
            break;
          }
          boneCurrent.push_back(linkWorldTransforms[it->second]);
          boneBind.push_back(bindLinkWorldTransforms[it->second]);
        }
        if (resolved && !boneBind.empty()) {
          req.worldTransform = boneBind.front();
          req.skinLinkWorldTransforms = std::move(boneCurrent);
          req.skinBindLinkWorldTransforms = std::move(boneBind);
        } else {
          MOCHI_LOG_ERROR_ONCE(
              "SceneStage: composed skin bone link names do not resolve to bot links; the skin "
              "preview will be misposed.");
        }
      }
      // Optional GPU-skinned render model (see the prefab path for the rationale): the GLB renders
      // in the Render-Model view and is posed from the link world transforms captured above (in GLB
      // joint order). No model offset is applied so its inverse-bind math stays valid; the bot
      // scale is its model scale (see ApplySkinnedJointPose).
      if (!prefab.skin->renderModelFile.empty()) {
        req.renderModelFile = mochi::Path(prefab.skin->renderModelFile);
        req.renderModelScale = Real3{prefab.scale, prefab.scale, prefab.scale};
      }
      // Route visibility on the real render and collision presence so fallback mode shows the
      // collision when no GLB.
      ComputeStageVisibility(
          /*hasRender=*/!req.renderModelFile.IsEmpty(),
          hasSkinShape,
          stageType,
          req.renderVisible,
          req.shapeVisible);
      // The skin stands in for some links, so hide them where the skin is drawn, as the skeletal
      // GLB export and the Unreal importer leave them out: where its render model is drawn, the
      // links it hides (ComputeSkinHiddenLinks); where only its collision surface is, the links
      // whose collision it covers (ComputeSkinCoveredLinks). A view that draws no skin keeps them.
      // They stay staged and still bound the scene, but cannot be picked while hidden.
      if (hideSkinCoveredLinks && (req.renderVisible || req.shapeVisible)) {
        DynamicArray<DynamicString> linkNames;
        linkNames.reserve(prefab.links.size());
        for (auto const& link : prefab.links) {
          linkNames.push_back(link.name);
        }
        DynamicArray<bool> standInLinks;
        if (req.renderVisible) {
          standInLinks =
              superdex::robotics::ComputeSkinHiddenLinks(prefab, MakeConstSpan(linkNames));
        } else {
          auto const& skinMesh = asset->GetModelData().mesh;
          standInLinks = superdex::robotics::ComputeSkinCoveredLinks(
              prefab,
              MakeConstSpan(linkNames),
              skinMesh.has_value() && skinMesh->skinning.has_value() ? &*skinMesh->skinning
                                                                     : nullptr);
        }
        for (int i = 0; i < isize(standInLinks); ++i) {
          if (standInLinks[i]) {
            out[firstLinkRequest + i].renderVisible = false;
            out[firstLinkRequest + i].shapeVisible = false;
          }
        }
      }
      out.push_back(std::move(req));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// SCENE STAGE
//--------------------------------------------------------------------------------------------------

SceneStage::SceneStage(SuperDexStudio* studio, char const* sceneName) : _studio(studio) {
  auto* context = _studio->GetMochiContext();
  if (context) {
    // Add a leading underscore to the scene name.
    // By convention, the remote debugger will not auto-select such scenes.
    _scene = context->CreateScene(Format("_%s", sceneName));
  }
}

SceneStage::~SceneStage() {
  if (_scene) {
    _scene->GetContext()->DestroyScene(_scene);
  }
}

void SceneStage::BindRenderScene(mochi_renderer::Scene* renderScene) {
  if (_renderScene == renderScene) {
    return;
  }
  // Drop anything staged into the previously bound scene before switching.
  Clear();
  _renderScene = renderScene;
}

bool SceneStage::StageBot(
    superdex::robotics::BotPrefab const& prefab,
    StageType stageType,
    bool hideSkinCoveredLinks) {
  if (!_studio || !_scene || !_renderScene) {
    return false;
  }
  std::vector<StageRequest> requests;
  BuildBotRequests(_scene, _studio, prefab, stageType, hideSkinCoveredLinks, requests);
  return ApplyStageRequests(requests, stageType);
}

bool SceneStage::StagePrefab(
    mochi::prefab::ScenePrefab const& prefab,
    std::string const& rootPath,
    StageType stageType) {
  using namespace mochi;
  if (!_studio || !_scene || !_renderScene) {
    return false;
  }

  // Compute articulated link forward kinematics up front; cheap no-op for rigid-only prefabs.
  std::vector<ArticulatedData> articulatedFk;
  ComputeArticulatedLinkTransforms(_scene, prefab, articulatedFk);

  // Build the ordered staging requests by traversing the prefab tree in AddToScene order.
  std::vector<StageRequest> requests;
  std::string_view const topSource = prefab.sourceFilePath.has_value()
      ? std::string_view(*prefab.sourceFilePath)
      : std::string_view();
  int artCursor = 0;
  BuildPrefabRequests(
      _studio,
      prefab,
      rootPath,
      topSource,
      TransformSRT::Identity(),
      stageType,
      articulatedFk,
      artCursor,
      std::string(),
      -1,
      requests);

  return ApplyStageRequests(requests, stageType);
}

#if MOCHI_INTERNAL
bool SceneStage::StageBotScene(
    superdex::robotics::BotScenePrefab const& scene,
    StageType stageType) {
  return StageBotSceneImpl(scene, nullptr, stageType);
}

bool SceneStage::StageBotTask(
    superdex::robotics::BotScenePrefab const& scene,
    mochi::Span<ResolvedTaskSpawn const> taskSpawns,
    StageType stageType) {
  return StageBotSceneImpl(scene, &taskSpawns, stageType);
}

bool SceneStage::StageBotSceneImpl(
    superdex::robotics::BotScenePrefab const& scene,
    mochi::Span<ResolvedTaskSpawn const> const* taskSpawns,
    StageType stageType) {
  using namespace mochi;
  if (!_studio || !_scene || !_renderScene) {
    return false;
  }
  auto& manager = _studio->GetAssetManager();

  std::vector<StageRequest> requests;

  // 1. Base scene (resolved from the AssetManager). Build requests in prefab::AddToScene order and
  // under the same "scene" prefab name that superdex::robotics::LoadBotScene uses, so the staged
  // actor names (and order) align with the base-scene physics actors.
  std::string const baseScenePath{scene.scene.baseScene};
  if (auto* baseAsset = manager.FindAssetByPath<MochiPrefabAsset>(Path{baseScenePath})) {
    auto const& basePrefab = baseAsset->GetPrefab();
    std::vector<ArticulatedData> articulatedFk;
    ComputeArticulatedLinkTransforms(_scene, basePrefab, articulatedFk);
    std::string_view const topSource = basePrefab.sourceFilePath.has_value()
        ? std::string_view(*basePrefab.sourceFilePath)
        : std::string_view();
    int artCursor = 0;
    BuildPrefabRequests(
        _studio,
        basePrefab,
        baseAsset->GetAssetsRoot(),
        topSource,
        TransformSRT::Identity(),
        stageType,
        articulatedFk,
        artCursor,
        std::string("scene"),
        -1,
        requests);
  }

  auto appendSpawnable =
      [&](std::string const& name, mochi::Path const& path, TransformRT const& worldFromSpawn) {
        if (path.IsEmpty()) {
          return;
        }

        if (superdex::robotics::IsBotArchivePath(path.ToString()) ||
            superdex::robotics::IsBotPath(path.ToString())) {
          auto* botAsset = manager.FindAssetByPath<BotAsset>(path);
          if (!botAsset) {
            MOCHI_LOG_ERROR(
                "SceneStage: bot '%s' for spawn '%s' is not loaded; not staged",
                path.ToString().c_str(),
                name.c_str());
            return;
          }
          superdex::robotics::BotPrefab botPrefab = botAsset->GetBotPrefab();
          botPrefab.name = name;
          botPrefab.worldFromRoot = worldFromSpawn;
          BuildBotRequests(
              _scene, _studio, botPrefab, stageType, /*hideSkinCoveredLinks=*/true, requests);
          return;
        }

        auto* prefabAsset = manager.FindAssetByPath<MochiPrefabAsset>(path);
        if (!prefabAsset) {
          MOCHI_LOG_ERROR(
              "SceneStage: prefab '%s' for spawn '%s' is not loaded; not staged",
              path.ToString().c_str(),
              name.c_str());
          return;
        }
        auto const& spawnPrefab = prefabAsset->GetPrefab();
        mochi::prefab::PrefabParams params;
        params.name = name;
        params.rotation = worldFromSpawn.GetRotation();
        params.translation = worldFromSpawn.GetTranslation();
        params.applySceneSettings = false;
        std::vector<ArticulatedData> articulatedFk;
        ComputeArticulatedLinkTransforms(_scene, spawnPrefab, articulatedFk, params);
        std::string_view const source = spawnPrefab.sourceFilePath.has_value()
            ? std::string_view(*spawnPrefab.sourceFilePath)
            : std::string_view();
        int artCursor = 0;
        BuildPrefabRequests(
            _studio,
            spawnPrefab,
            prefabAsset->GetAssetsRoot(),
            source,
            TransformSRT{1_r, worldFromSpawn.GetRotation(), worldFromSpawn.GetTranslation()},
            stageType,
            articulatedFk,
            artCursor,
            name,
            -1,
            requests);
      };

  // 2. Spawnable declarations are templates, not scene instances. Only a loaded task contributes
  // concrete spawnable objects to the preview.
  if (taskSpawns != nullptr) {
    for (auto const& spawn : *taskSpawns) {
      appendSpawnable(spawn.name, spawn.prefabPath, spawn.worldFromSpawn);
    }
  }

  // 3. Each placed bot, in declaration order (matching LoadBotScene's bot creation order).
  for (auto const& botEntry : scene.bots) {
    if (botEntry.path.empty()) {
      continue;
    }
    auto* botAsset = manager.FindAssetByPath<BotAsset>(Path{std::string(botEntry.path)});
    if (!botAsset) {
      continue;
    }
    superdex::robotics::BotPrefab botPrefab = botAsset->GetBotPrefab();
    botPrefab.worldFromRoot = botEntry.parentFromBot;
    if (!botEntry.initialPose.empty()) {
      botPrefab.defaultPose = botEntry.initialPose;
    }
    BuildBotRequests(
        _scene, _studio, botPrefab, stageType, /*hideSkinCoveredLinks=*/true, requests);
  }

  return ApplyStageRequests(requests, stageType);
}
#endif // MOCHI_INTERNAL

bool SceneStage::ApplyStageRequests(
    std::vector<StageRequest> const& requests,
    StageType stageType) {
  using namespace mochi;
  _stageType = stageType;
  auto& manager = _studio->GetAssetManager();
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();

  // Reuse the existing staged objects when the actor set (by name) and BOTH model files are
  // unchanged (the common pose/param-edit case, and -- because the reuse key is independent of
  // StageType -- a StageType switch too, which then only re-applies visibility below). Otherwise
  // rebuild.
  bool reuse = _actors.size() == requests.size();
  if (reuse) {
    for (size_t i = 0; i < requests.size(); ++i) {
      bool const reqIsSoft = requests[i].isSoft;
      // A staged actor is "soft" if it has any deforming representation.
      bool const actorIsSoft = _actors[i].dynamicSolidMesh != nullptr ||
          _actors[i].dynamicWireframeMesh != nullptr || _actors[i].softRenderModel != nullptr ||
          _actors[i].skinnedCollisionMesh != nullptr;
      // A soft/non-soft mismatch, or a change in which soft meshes are shown or in their vertex
      // counts (topology change), forces a rebuild.
      if (_actors[i].name != requests[i].name ||
          _actors[i].render.modelFile != requests[i].renderModelFile ||
          _actors[i].shape.modelFile != requests[i].shapeModelFile || reqIsSoft != actorIsSoft ||
          (reqIsSoft &&
           (_actors[i].softSurfaces != requests[i].softSurfaces ||
            _actors[i].softMeshVertexCount != requests[i].softVertexCount ||
            _actors[i].softCollisionVertexCount != requests[i].softCollisionVertexCount ||
            (_actors[i].softRenderModel != nullptr) != requests[i].softRenderModelLinked))) {
        reuse = false;
        break;
      }
      // A skinned skin bakes scale/shape-offset into its GPU-skinned collision geometry, which has
      // no in-place vertex update, so a change to those params forces a full rebuild (pose edits,
      // the common case, keep the same params and reuse).
      if ((_actors[i].skinnedCollisionMesh != nullptr ||
           _actors[i].skinnedRenderObject != nullptr) &&
          (_actors[i].softBakeScale[0] != requests[i].softBakeScale[0] ||
           _actors[i].softBakeScale[1] != requests[i].softBakeScale[1] ||
           _actors[i].softBakeScale[2] != requests[i].softBakeScale[2] ||
           !(_actors[i].softShapeTransform == requests[i].softShapeTransform))) {
        reuse = false;
        break;
      }
    }
  }
  if (!reuse) {
    Clear();
    _actors.resize(requests.size());
  }

  for (int i = 0; i < isize(requests); ++i) {
    auto& actor = _actors[i];
    auto const& req = requests[i];

    // Refresh transforms/offsets/scales every stage so pure pose/offset/scale edits (which do not
    // invalidate reuse) are reflected. Object (re)creation only happens on the rebuild path.
    actor.worldTransform = req.worldTransform;
    actor.defaultWorldTransform = req.worldTransform;
    actor.worldScale = req.worldScale;
    actor.render.modelTransform = req.renderModelTransform;
    actor.render.modelScale = req.renderModelScale;
    actor.shape.modelTransform = req.shapeModelTransform;
    actor.shape.modelScale = req.shapeModelScale;
    // Per-instance soft bake params are refreshed every stage (harmless defaults for non-soft), so
    // rest-pose restore and idle refresh re-bake with the current scale/offset.
    actor.softBakeScale = req.softBakeScale;
    actor.softShapeTransform = req.softShapeTransform;
    SetSoftRenderModelNotice(actor, req.name, req.softRenderModelNotice);

    // Soft actors bake scale/shape-offset into their geometry, so an edit that changes the rest
    // geometry but not the vertex count (e.g. scaling a nested prefab) reuses the meshes. Re-bake
    // the reused solid + wireframe meshes via the asset so the idle pose stays correct (the rebuild
    // path below handles vertex-count changes).
    if (reuse && req.isSoft &&
        (actor.dynamicSolidMesh || actor.dynamicWireframeMesh || actor.softRenderModel)) {
      // Re-fetch the asset from the AssetManager instead of trusting the cached raw pointer: the
      // manager owns assets as unique_ptr and may have evicted/reloaded this one since it was
      // staged, leaving actor.softAsset dangling. Refreshing here keeps it valid for this and later
      // dereferences (e.g. ResetWorldTransforms).
      actor.softAsset = manager.LoadMochiModelAsset(req.shapeModelFile);
      if (actor.softAsset) {
        actor.softAsset->UpdateSoftDynamicMeshes(
            actor.softSurfaces,
            actor.dynamicSolidMesh,
            actor.dynamicWireframeMesh,
            req.softBakeScale,
            req.softShapeTransform);
        ForEachSoftRenderModel(actor, [&](mochi_renderer::DeformableModelInstance& renderModel) {
          actor.softAsset->UpdateSoftRenderModel(
              renderModel, req.softBakeScale, req.softShapeTransform);
        });
      }
    }

    if (!reuse) {
      actor.name = req.name;
      actor.render.modelFile = req.renderModelFile;
      actor.shape.modelFile = req.shapeModelFile;
      actor.source = req.source;
      actor.sourceActorIndex = req.sourceActorIndex;
      actor.sourceLinkIndex = req.sourceLinkIndex;
      // Render models keep their smooth glb materials; shape/mochi models are flat-lit. Record it
      // so the highlight clone can match each base's shading.
      actor.render.flatShaded = false;
      actor.shape.flatShaded = true;
      if (!req.skinLinkWorldTransforms.empty()) {
        // Skinned articulated skin: an optional GPU-skinned collision wireframe occupies the shape
        // slot (the Collision view), GPU skinned by the articulation link transforms (via
        // SetBoneMatrices) so it deforms at editor time and tracks physics at sim time without CPU
        // re-baking. An optional GPU-skinned render GLB (posed each frame via SetBoneMatrices)
        // occupies the render slot. The two are fully independent of each other.
        bool const hasCollisionSkin = !req.shapeModelFile.IsEmpty();
        auto* asset = hasCollisionSkin ? manager.LoadMochiModelAsset(req.shapeModelFile) : nullptr;
        std::unique_ptr<mochi_renderer::WireframeMesh> wireframe;
        int const boneCount = isize(req.skinLinkWorldTransforms);
        if (asset) {
          asset->CreateSkinnedCollisionWireframe(
              req.softBakeScale, req.softShapeTransform, boneCount, wireframe);
        }
        actor.softAsset = asset;
        actor.dynamicSolidMesh = nullptr;
        // The simulation surface is built on demand by EnsureSimSurface once Mochi actually reports
        // vertices; at edit time the blend-skinned preview is the only surface staged.
        actor.dynamicWireframeMesh = nullptr;
        actor.shapeSimSurface = nullptr;
        actor.skinnedCollisionMesh = wireframe.get();
        if (wireframe) {
          wireframe->SetName(req.name);
        }
        actor.shapeSkinnedSurface =
            wireframe ? _renderScene->AddSceneObjectToScene(std::move(wireframe)) : nullptr;
        actor.shape.sceneObject = actor.shapeSkinnedSurface;
        actor.shape.flatShaded = true;
        actor.softSurfaces = req.softSurfaces;
        actor.softMeshVertexCount = req.softVertexCount;
        actor.softCollisionVertexCount = req.softCollisionVertexCount;
        if (hasCollisionSkin && !actor.skinnedCollisionMesh) {
          MOCHI_LOG_ERROR_ONCE(
              "SceneStage: skin collision surface has no skinning; the Collision view will not "
              "deform with the pose.");
        }

        // Cache the collision wireframe's inverse-bind bone matrices from the ZERO/rest (bind) link
        // transforms (renderer space, root-relative). This is pose-independent -- it does NOT latch
        // to the current editor pose -- so bone_j = current_j * inverse(bind_j) reproduces the
        // authored rest surface at rest and the correct deformation when posed, matching the GLB's
        // baked inverse-bind. `actor.worldTransform` is the articulation root (set above from
        // req.worldTransform); it cancels between current and bind so only joint motion remains.
        TransformRT const restRootInv = Invert(actor.worldTransform);
        int const bindCount = std::min(boneCount, isize(req.skinBindLinkWorldTransforms));
        actor.skinRestBoneInverse.assign(bindCount, filament::math::mat4f{});
        for (int j = 0; j < bindCount; ++j) {
          TransformRT const restJoint =
              converter.TransformToOutput(restRootInv * req.skinBindLinkWorldTransforms[j]);
          actor.skinRestBoneInverse[j] = inverse(ToFilamentMat4(restJoint));
        }

        // Optional GPU-skinned render GLB (independent of the collision skin above).
        if (!req.renderModelFile.IsEmpty()) {
          auto renderInstance =
              CreateRenderInstance(manager, req.renderModelFile, /*isRenderModel=*/true);
          actor.skinnedRenderObject =
              renderInstance ? renderInstance->AsSkinnedModelInstance() : nullptr;
          if (renderInstance) {
            renderInstance->SetName(req.name);
          }
          actor.render.sceneObject = renderInstance
              ? _renderScene->AddSceneObjectToScene(std::move(renderInstance))
              : nullptr;
          actor.render.flatShaded = false;
          if (!actor.skinnedRenderObject) {
            MOCHI_LOG_ERROR_ONCE(
                "SceneStage: skin render model has no skin data; it will render undeformed.");
          }
        }
      } else if (req.isSoft) {
        // Soft actors stage a deforming wireframe in the shape slot, plus either a linked render
        // model or a solid Mochi mesh in the render slot. The solid is smooth-lit only when it
        // shows the visual mesh; recording that lets the highlight clone match its shading. The
        // asset is kept so idle refresh / simulation-stop can re-bake the rest pose.
        actor.render.flatShaded = req.softSurfaces.rendered != SoftSurface::Visual;
        auto* asset = manager.LoadMochiModelAsset(req.shapeModelFile);
        actor.softAsset = asset;
        // A render model linked to the visual mesh takes the solid's place, keeping its own
        // materials and deformed by the same visual-mesh updates the solid would get.
        std::unique_ptr<mochi_renderer::SceneObject> render;
        if (asset && req.softRenderModelLinked) {
          if (auto* renderAsset = manager.LoadRenderModelAsset(req.renderModelFile)) {
            mochi::Error error;
            std::unique_ptr<DeformableModelInstance> instance =
                renderAsset->GetDeformableModelInstance(error);
            if (instance) {
              asset->UpdateSoftRenderModel(*instance, req.softBakeScale, req.softShapeTransform);
              actor.softRenderModel = instance.get();
              render = std::move(instance);
            } else {
              // A linked actor's notice was cleared above, so this failure is always warned.
              MOCHI_ASSERT_VERBOSE(actor.softRenderModelNotice.empty());
              SetSoftRenderModelNotice(actor, req.name, error.GetDescription());
            }
          }
        }
        std::unique_ptr<mochi_renderer::Mesh> solid;
        std::unique_ptr<mochi_renderer::WireframeMesh> wireframe;
        if (asset) {
          asset->CreateSoftDynamicMeshes(
              req.softSurfaces,
              req.softBakeScale,
              req.softShapeTransform,
              /*createSolid=*/!render,
              solid,
              wireframe);
        }
        actor.dynamicSolidMesh = solid.get();
        actor.dynamicWireframeMesh = wireframe.get();
        if (!render) {
          render = std::move(solid);
        }
        if (render) {
          render->SetName(req.name);
        }
        if (wireframe) {
          wireframe->SetName(req.name);
        }
        actor.render.sceneObject =
            render ? _renderScene->AddSceneObjectToScene(std::move(render)) : nullptr;
        actor.shape.sceneObject =
            wireframe ? _renderScene->AddSceneObjectToScene(std::move(wireframe)) : nullptr;
        actor.softSurfaces = req.softSurfaces;
        actor.softMeshVertexCount = req.softVertexCount;
        actor.softCollisionVertexCount = req.softCollisionVertexCount;
      } else {
        auto renderInstance =
            CreateRenderInstance(manager, req.renderModelFile, /*isRenderModel=*/true);
        auto shapeInstance =
            CreateRenderInstance(manager, req.shapeModelFile, /*isRenderModel=*/false);
        if (renderInstance) {
          renderInstance->SetName(req.name);
        }
        if (shapeInstance) {
          shapeInstance->SetName(req.name);
        }
        actor.render.sceneObject = renderInstance
            ? _renderScene->AddSceneObjectToScene(std::move(renderInstance))
            : nullptr;
        actor.shape.sceneObject =
            shapeInstance ? _renderScene->AddSceneObjectToScene(std::move(shapeInstance)) : nullptr;
      }
    }

    // Record StageType visibility and apply it every stage so a mode switch (which reuses) is just
    // a toggle. This is the base's baseline visibility; UpdateHighlights overrides it to hide a
    // base while its highlight clone stands in. The selectable object aliases whichever
    // representation is visible (render preferred).
    ApplyStageVisibility(actor, req.renderVisible, req.shapeVisible);
    ApplySdfVisibility(actor);

    SetStagedActorWorldTransform(actor, actor.worldTransform, converter);

    // Editor-time skin pose: drive the skinned render model's joints from the captured rest/default
    // FK link transforms. Runs every stage (rebuild and reuse) so default-pose edits -- which reuse
    // objects -- still repose the skin. During simulation the editor overrides this via
    // ApplySkinnedPose. No-op unless this actor has a skinned render model or collision skin. The
    // transforms are also cached so simulation-stop can restore this rest pose (see
    // ResetWorldTransforms).
    if (actor.skinnedRenderObject || actor.skinnedCollisionMesh) {
      actor.skinRestLinkWorldTransforms = req.skinLinkWorldTransforms;
      ApplySkinnedJointPose(actor, mochi::MakeConstSpan(req.skinLinkWorldTransforms), converter);
    }
  }

  if (!reuse) {
    _nameToIndex.clear();
    for (int i = 0; i < isize(_actors); ++i) {
      _nameToIndex[_actors[i].name] = i;
    }
  }
  BuildMissingSdfSurfaces();
  return !reuse;
}

// True for an actor whose collision surface deforms (a soft actor or an articulated skin). Its
// SDF, if any, does not follow the deformation, so it is not shown in SDF mode.
static bool HasDeformingSurface(StagedActor const& actor) {
  return actor.dynamicSolidMesh != nullptr || actor.dynamicWireframeMesh != nullptr ||
      actor.shapeSkinnedSurface != nullptr;
}

void SceneStage::ApplySdfVisibility(StagedActor& actor) {
  // The SDF stands in for a shown shape: shape.visible as staged, or sdf.visible once the SDF has
  // replaced it (ApplyStageVisibility clears sdf.visible before each stage).
  bool const wanted = _stageType == StageType::SdfFallbackToMochiModel &&
      actor.shape.sceneObject != nullptr && (actor.shape.visible || actor.sdf.visible) &&
      !HasDeformingSurface(actor);
  if (wanted && actor.sdf.sceneObject == nullptr) {
    auto* asset = _studio->GetAssetManager().LoadMochiModelAsset(actor.shape.modelFile);
    if (auto instance = asset ? asset->GetSdfRenderModelInstance() : nullptr) {
      instance->SetName(actor.name);
      actor.sdf.sceneObject = _renderScene->AddSceneObjectToScene(std::move(instance));
    }
  }
  actor.sdf.modelFile = actor.shape.modelFile;
  actor.sdf.modelTransform = actor.shape.modelTransform;
  actor.sdf.modelScale = actor.shape.modelScale;
  actor.sdf.flatShaded = true;
  actor.sdf.visible = wanted && actor.sdf.sceneObject != nullptr;
  if (actor.sdf.sceneObject == nullptr) {
    return;
  }
  actor.sdf.sceneObject->SetVisible(actor.sdf.visible);
  if (actor.sdf.visible) {
    actor.shape.visible = false;
    actor.shape.sceneObject->SetVisible(false);
    actor.sceneObject = actor.sdf.sceneObject;
  }
}

void SceneStage::BuildMissingSdfSurfaces() {
  if (_stageType != StageType::SdfFallbackToMochiModel || !_studio ||
      _studio->IsAsyncTasksRunning()) {
    return;
  }
  // One job per collision model: the SDF is copied so the workers never touch the asset.
  struct Job {
    mochi::Path path;
    int generation = 0;
    mochi::GridSdfData sdf;
    std::optional<MochiModelAsset::SdfSurface> surface;
    // False if the task was skipped or cancelled, so the next request tries again.
    bool finished = false;
  };
  auto jobs = std::make_shared<std::vector<Job>>();
  auto& manager = _studio->GetAssetManager();
  std::set<mochi::Path> seen;
  for (StagedActor const& actor : _actors) {
    if (actor.sdf.sceneObject != nullptr || actor.shape.sceneObject == nullptr ||
        HasDeformingSurface(actor) || !seen.insert(actor.shape.modelFile).second) {
      continue;
    }
    auto* asset = manager.LoadMochiModelAsset(actor.shape.modelFile);
    if (asset && asset->HasSdf() && !asset->IsSdfSurfaceBuilt()) {
      jobs->push_back(
          {.path = actor.shape.modelFile,
           .generation = asset->GetSdfGeneration(),
           .sdf = *asset->GetModelData().sdf});
    }
  }
  if (jobs->empty()) {
    return;
  }

  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  std::vector<AsyncTask> tasks;
  std::set<std::string> labels;
  for (int i = 0; i < mochi::isize(*jobs); ++i) {
    // Task labels identify tasks (e.g. to skip one), so disambiguate same-named files.
    std::string const fileName =
        std::filesystem::path((*jobs)[i].path.ToString()).filename().string();
    std::string label = fileName;
    for (int n = 2; !labels.insert(label).second; ++n) {
      label = fileName + " (" + std::to_string(n) + ")";
    }
    tasks.push_back(
        {.label = std::move(label), .work = [jobs, i, &converter](AsyncCancelToken const& cancel) {
           Job& job = (*jobs)[i];
           job.surface = MochiModelAsset::ReconstructSdfSurface(job.sdf, converter);
           job.finished = !cancel.IsCancelRequested();
           return job.surface.has_value();
         }});
  }
  _studio->BeginAsyncTasks(
      "Building SDF Surfaces",
      std::move(tasks),
      [this, studio = _studio, jobs, lifetime = std::weak_ptr<bool>(_lifetime)](bool) {
        // Install into the assets even if this stage is gone: other viewports share the cache.
        for (Job& job : *jobs) {
          auto* asset = studio->GetAssetManager().FindAssetByPath<MochiModelAsset>(job.path);
          if (asset != nullptr && job.finished) {
            asset->SetSdfSurface(job.generation, std::move(job.surface));
          }
        }
        if (lifetime.expired()) {
          return;
        }
        auto const& converter = studio->GetEditorToRendererSpaceConverter();
        for (StagedActor& actor : _actors) {
          ApplySdfVisibility(actor);
          if (actor.sdf.sceneObject != nullptr) {
            SetStagedActorWorldTransform(actor, actor.worldTransform, converter);
          }
        }
      });
}

void SceneStage::SetRepresentationMode(StageType stageType) {
  _stageType = stageType;
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  for (auto& actor : _actors) {
    bool renderVisible = false;
    bool shapeVisible = false;
    ComputeStageVisibility(
        actor.render.sceneObject != nullptr,
        actor.shape.sceneObject != nullptr,
        stageType,
        renderVisible,
        shapeVisible);
    ApplyStageVisibility(actor, renderVisible, shapeVisible);
    ApplySdfVisibility(actor);
    // An SDF surface created just now has not been placed yet.
    if (actor.sdf.sceneObject != nullptr) {
      SetStagedActorWorldTransform(actor, actor.worldTransform, converter);
    }
  }
  BuildMissingSdfSurfaces();
}

void SceneStage::Clear() {
  if (_renderScene) {
    for (auto& actor : _actors) {
      // Highlight clones are separate scene objects the stage owns; destroy them before their bases
      // so a rebuild/clear does not orphan them.
      if (actor.render.highlightClone) {
        _renderScene->DestroySceneObject(actor.render.highlightClone);
      }
      if (actor.shape.highlightClone) {
        _renderScene->DestroySceneObject(actor.shape.highlightClone);
      }
      if (actor.sdf.highlightClone) {
        _renderScene->DestroySceneObject(actor.sdf.highlightClone);
      }
      if (actor.sdf.sceneObject) {
        _renderScene->DestroySceneObject(actor.sdf.sceneObject);
      }
      if (actor.render.sceneObject) {
        _renderScene->DestroySceneObject(actor.render.sceneObject);
      }
      if (actor.shape.sceneObject) {
        _renderScene->DestroySceneObject(actor.shape.sceneObject);
      }
      // A skinned skin owns a second surface that is not currently installed in the slot above.
      for (auto* surface : {actor.shapeSkinnedSurface, actor.shapeSimSurface}) {
        if (surface && surface != actor.shape.sceneObject) {
          _renderScene->DestroySceneObject(surface);
        }
      }
    }
  }
  _actors.clear();
  _nameToIndex.clear();
}

void SceneStage::ApplyWorldTransforms(
    mochi::Span<mochi::TransformRT const> actorWorldTransforms,
    mochi::CoordinateSpaceConverter const& converter) {
  if (static_cast<size_t>(actorWorldTransforms.size()) != _actors.size()) {
    MOCHI_LOG_ERROR(
        "ApplyWorldTransforms: transform count does not match the number of staged actors");
    return;
  }
  for (int i = 0; i < static_cast<int>(_actors.size()); ++i) {
    SetStagedActorWorldTransform(_actors[i], actorWorldTransforms[i], converter);
  }
}

void SceneStage::ApplyWorldTransforms(
    mochi::Span<std::string const> names,
    mochi::Span<mochi::TransformRT const> actorWorldTransforms,
    mochi::CoordinateSpaceConverter const& converter) {
  if (names.size() != actorWorldTransforms.size()) {
    MOCHI_LOG_ERROR("ApplyWorldTransforms: names and transforms arrays differ in length");
    return;
  }
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    ApplyWorldTransform(names[i], actorWorldTransforms[i], converter);
  }
}

void SceneStage::ApplyWorldTransform(
    std::string_view name,
    mochi::TransformRT const& actorWorldTransform,
    mochi::CoordinateSpaceConverter const& converter) {
  int const index = FindIndexByName(name);
  if (index < 0) {
    return;
  }
  SetStagedActorWorldTransform(_actors[index], actorWorldTransform, converter);
}

void SceneStage::ResetWorldTransforms(mochi::CoordinateSpaceConverter const& converter) {
  for (auto& actor : _actors) {
    // Soft actors also restore their rest-pose surface geometry (deformed during simulation) by
    // re-baking it via the asset into both the solid and wireframe meshes. Re-fetch the asset from
    // the AssetManager instead of trusting the cached raw pointer, which the manager may have
    // evicted/reloaded (leaving actor.softAsset dangling); this only runs for staged soft actors,
    // so _studio is valid.
    if (actor.dynamicSolidMesh || actor.dynamicWireframeMesh || actor.softRenderModel) {
      actor.softAsset = _studio->GetAssetManager().LoadMochiModelAsset(actor.shape.modelFile);
      if (actor.softAsset) {
        actor.softAsset->UpdateSoftDynamicMeshes(
            actor.softSurfaces,
            actor.dynamicSolidMesh,
            actor.dynamicWireframeMesh,
            actor.softBakeScale,
            actor.softShapeTransform);
        ForEachSoftRenderModel(actor, [&](mochi_renderer::DeformableModelInstance& renderModel) {
          actor.softAsset->UpdateSoftRenderModel(
              renderModel, actor.softBakeScale, actor.softShapeTransform);
        });
      }
    }
    // Simulation is over: hand the shape slot back to the blend-skinned surface, which tracks
    // editor-time pose changes, and drop the simulation surface entirely so its rest geometry does
    // not sit in the scene bounds until the next run.
    DestroySimSurface(actor);
    SetStagedActorWorldTransform(actor, actor.defaultWorldTransform, converter);
    // Restore the skin's render-model joint pose to its captured rest pose (worldTransform was just
    // reset to the default above, so joints become root-relative against the default root).
    ApplySkinnedJointPose(
        actor, mochi::MakeConstSpan(actor.skinRestLinkWorldTransforms), converter);
  }
}

void SceneStage::ApplySoftMeshUpdates(
    mochi::Span<SoftMeshUpdate const> updates,
    mochi::CoordinateSpaceConverter const& converter) {
  for (auto const& update : updates) {
    int const index = FindUpdateTarget(update.name, update.stagedIndex);
    if (index < 0) {
      continue;
    }
    auto& actor = _actors[index];
    // Mochi is reporting this skin's surface, so a run is under way and the simulation surface is
    // needed; it is not staged at edit time.
    EnsureSimSurface(actor, converter);
    if (!actor.dynamicSolidMesh && !actor.dynamicWireframeMesh && !actor.softRenderModel) {
      continue;
    }
    // A vertex-count mismatch means the meshes were rebuilt underneath this update; skip that mesh.
    auto const vertexCount = [](std::vector<float> const& positions) {
      return static_cast<int>(positions.size() / 3);
    };
    if (vertexCount(update.positions) == actor.softMeshVertexCount) {
      if (actor.dynamicSolidMesh) {
        actor.dynamicSolidMesh->Update(
            mochi::MakeConstSpan(update.positions), mochi::MakeConstSpan(update.normals));
      }
      ForEachSoftRenderModel(actor, [&](mochi_renderer::DeformableModelInstance& renderModel) {
        renderModel.SetVertices(
            mochi::MakeConstSpan(update.positions), mochi::MakeConstSpan(update.normals));
      });
    }
    bool const sameMesh = update.collisionPositions.empty();
    std::vector<float> const& wirePositions =
        sameMesh ? update.positions : update.collisionPositions;
    std::vector<float> const& wireNormals = sameMesh ? update.normals : update.collisionNormals;
    bool wireframeUpdated = false;
    if (actor.dynamicWireframeMesh &&
        vertexCount(wirePositions) == actor.softCollisionVertexCount) {
      wireframeUpdated = actor.dynamicWireframeMesh->Update(
          mochi::MakeConstSpan(wirePositions), mochi::MakeConstSpan(wireNormals));
    }
    if (wireframeUpdated) {
      // Mochi is now reporting this surface directly, so show it in place of the blend-skinned
      // approximation. No-op for plain soft actors, and after the first step of a run.
      SetSimSurfaceActive(actor, true);
    }
    SetStagedActorWorldTransform(actor, update.worldTransform, converter);
  }
}

void SceneStage::ApplySkinnedPose(
    mochi::Span<SkinnedPoseUpdate const> updates,
    mochi::CoordinateSpaceConverter const& converter) {
  using namespace mochi;
  for (auto const& update : updates) {
    int const index = FindUpdateTarget(update.name, update.stagedIndex);
    if (index < 0) {
      continue;
    }
    auto& actor = _actors[index];
    if (!actor.skinnedRenderObject && !actor.skinnedCollisionMesh) {
      continue;
    }
    // The bind (inverse-rest) matrices cancel these at rest, so the joints drive the deformation
    // delta while the SceneObject transform (kept in sync by ApplyWorldTransforms /
    // ApplySoftMeshUpdates) places the whole skin. Drives both the render GLB and the collision
    // skin.
    ApplySkinnedJointPose(actor, mochi::MakeConstSpan(update.linkWorldTransforms), converter);
  }
}

StagedActors const& SceneStage::GetActors() const {
  return _actors;
}

int SceneStage::GetNumActors() const {
  return static_cast<int>(_actors.size());
}

int SceneStage::FindIndexByName(std::string_view name) const {
  auto it = _nameToIndex.find(std::string(name));
  return it == _nameToIndex.end() ? -1 : it->second;
}

int SceneStage::FindUpdateTarget(std::string_view name, int stagedIndex) const {
  if (stagedIndex < 0) {
    return FindIndexByName(name);
  }
  // A staged index is only meaningful for the actor set it was taken from.
  return stagedIndex < GetNumActors() && _actors[stagedIndex].name == name ? stagedIndex : -1;
}

void SceneStage::EnsureSimSurface(
    StagedActor& actor,
    mochi::CoordinateSpaceConverter const& converter) {
  if (!actor.shapeSkinnedSurface || actor.shapeSimSurface || !_renderScene || !_studio) {
    return;
  }
  // Re-fetch rather than trusting the cached pointer, which the manager may have evicted since
  // staging (same reasoning as ResetWorldTransforms).
  actor.softAsset = _studio->GetAssetManager().LoadMochiModelAsset(actor.shape.modelFile);
  if (!actor.softAsset) {
    return;
  }
  // Only the wireframe is wanted (a skin has no solid shape representation).
  std::unique_ptr<mochi_renderer::Mesh> unusedSolid;
  std::unique_ptr<mochi_renderer::WireframeMesh> simWireframe;
  actor.softAsset->CreateSoftDynamicMeshes(
      actor.softSurfaces,
      actor.softBakeScale,
      actor.softShapeTransform,
      /*createSolid=*/false,
      unusedSolid,
      simWireframe);
  if (!simWireframe) {
    return;
  }
  simWireframe->SetName(actor.name);
  actor.dynamicWireframeMesh = simWireframe.get();
  actor.shapeSimSurface = _renderScene->AddSceneObjectToScene(std::move(simWireframe));
  if (actor.shapeSimSurface) {
    actor.shapeSimSurface->SetVisible(false);
  }
  SetStagedActorWorldTransform(actor, actor.worldTransform, converter);
}

void SceneStage::DestroySimSurface(StagedActor& actor) {
  if (!actor.shapeSkinnedSurface || !actor.shapeSimSurface) {
    return;
  }
  SetSimSurfaceActive(actor, false);
  if (_renderScene) {
    _renderScene->DestroySceneObject(actor.shapeSimSurface);
  }
  actor.shapeSimSurface = nullptr;
  actor.dynamicWireframeMesh = nullptr;
}

void SceneStage::SetSimSurfaceActive(StagedActor& actor, bool simActive) {
  if (!actor.shapeSimSurface || !actor.shapeSkinnedSurface) {
    return;
  }
  auto* const wanted = simActive ? actor.shapeSimSurface : actor.shapeSkinnedSurface;
  if (actor.shape.sceneObject == wanted) {
    return;
  }
  bool const wasSelectable = actor.sceneObject == actor.shape.sceneObject;
  actor.shape.sceneObject = wanted;
  // The tinted stand-in was cloned from the outgoing surface; drop it so UpdateHighlights rebuilds
  // it from the incoming one rather than highlighting a hidden mesh.
  if (actor.shape.highlightClone) {
    _renderScene->DestroySceneObject(actor.shape.highlightClone);
    actor.shape.highlightClone = nullptr;
  }
  // Only the installed surface follows the slot's StageType visibility.
  actor.shapeSkinnedSurface->SetVisible(!simActive && actor.shape.visible);
  actor.shapeSimSurface->SetVisible(simActive && actor.shape.visible);
  // Keep the selection/picking alias pointing at whichever surface is now installed.
  if (wasSelectable) {
    actor.sceneObject = actor.shape.visible ? wanted : nullptr;
  }
}

int SceneStage::GetSceneObjectIndex(SceneObject const* object) const {
  if (!object) {
    return -1;
  }
  for (int i = 0; i < static_cast<int>(_actors.size()); ++i) {
    if (_actors[i].render.sceneObject == object || _actors[i].shape.sceneObject == object ||
        _actors[i].sdf.sceneObject == object || _actors[i].shapeSkinnedSurface == object ||
        _actors[i].shapeSimSurface == object) {
      return i;
    }
  }
  return -1;
}

bool SceneStage::IsEmpty() const {
  return _actors.empty();
}

void SceneStage::RequestHighlight(int index, filament::math::float3 color) {
  if (index < 0 || index >= static_cast<int>(_actors.size())) {
    return;
  }
  auto& actor = _actors[index];
  // First request this frame wins: callers declare hover before selection, so hover color takes
  // precedence when both apply to one link.
  if (actor.highlightRequested) {
    return;
  }
  actor.highlightRequested = true;
  actor.highlightColor = color;
}

void SceneStage::UpdateHighlights() {
  if (!_renderScene) {
    return;
  }
  auto sameColor = [](filament::math::float3 const& a, filament::math::float3 const& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  };
  _hasActiveHighlights = false;
  for (auto& actor : _actors) {
    bool const highlighted = actor.highlightRequested;
    for (auto* inst : {&actor.render, &actor.shape, &actor.sdf}) {
      if (!inst->sceneObject) {
        continue;
      }
      // Highlight a representation only when the actor is highlighted AND that representation is
      // the one the current StageType is showing. So if the render mode hides this model, its base
      // is hidden and the highlight is suppressed too -- the highlight follows the shown model(s),
      // and both render and shape can be highlighted at once when both are shown.
      bool const showHighlight = highlighted && inst->visible;
      if (showHighlight) {
        // Create the stand-in clone once, then cache it and just toggle its visibility below -- so
        // a rapidly changing hover (e.g. the contact matrix) never churns scene objects. Only the
        // material is swapped when the highlight color changes (selection <-> hover).
        if (!inst->highlightClone) {
          inst->highlightClone =
              CreateHighlightClone(inst->sceneObject, actor.highlightColor, inst->flatShaded);
          inst->highlightCloneColor = actor.highlightColor;
        } else if (!sameColor(inst->highlightCloneColor, actor.highlightColor)) {
          auto& rm = _studio->GetResourceManager();
          inst->highlightClone->SetMaterial(
              inst->flatShaded ? rm.CreateFlatLitOpaqueMaterial(actor.highlightColor)
                               : rm.CreateLitOpaqueMaterial(actor.highlightColor));
          inst->highlightCloneColor = actor.highlightColor;
        }
        _hasActiveHighlights = true;
      }
      // Show/hide the cached clone in both the main view (0x01) and the overlay pass
      // (kHighlightOverlayLayer): SetVisible only toggles the main-view bit, so the overlay bit
      // must be cleared too or a hidden clone would keep drawing the see-through highlight. The
      // base is hidden exactly while its clone stands in for it; otherwise it follows StageType.
      if (inst->highlightClone) {
        inst->highlightClone->SetVisible(showHighlight);
        inst->highlightClone->SetHighlightOverlay(showHighlight);
      }
      inst->sceneObject->SetVisible(inst->visible && !showHighlight);
    }
    // Per-frame request consumed; UIs re-declare it next frame.
    actor.highlightRequested = false;
  }
}

bool SceneStage::HasActiveHighlights() const {
  return _hasActiveHighlights;
}

mochi_renderer::SceneObject* SceneStage::CreateHighlightClone(
    SceneObject* base,
    filament::math::float3 color,
    bool flatShaded) const {
  auto instance = base->GetInstanceable()->GetInstance();
  if (!instance) {
    return nullptr;
  }
  // Opaque, lit, shadowless clone. The base mesh is hidden while highlighted, so this stands in for
  // it: it writes depth and is occluded by nearer links, never receives shadows (the highlight
  // color stays uniform), and still casts a shadow (the ground shadow is preserved). Tagging it
  // into the highlight overlay pass also re-renders it in isolation for the clean see-through
  // composite. Flat vs smooth is chosen to match the base's shading so the clone doesn't look
  // smoother (or more faceted) than the model it replaces.
  auto& rm = _studio->GetResourceManager();
  instance->SetMaterial(
      flatShaded ? rm.CreateFlatLitOpaqueMaterial(color) : rm.CreateLitOpaqueMaterial(color));
  instance->SetShadows(true, false);
  instance->SetSortPriority(7);
  instance->_internal = true;
  // Picking the clone (which stands in for the hidden base while highlighted) resolves back to the
  // base, so a selected object stays pickable (e.g. to Ctrl-click it out of a multi-selection).
  instance->_pickProxy = base;
  instance->SetParent(base);
  instance->SetHighlightOverlay(true);
  return _renderScene->AddSceneObjectToScene(std::move(instance));
}

void SceneStage::ShowSceneStageWindow(
    char const* name,
    bool* open,
    std::vector<std::string> const* simNames) const {
  ImGui::Begin(name, open);
  auto const& actors = GetActors();
  bool const simulating = simNames != nullptr;
  // The sim names (and equality coloring) are only meaningful while the physics is producing
  // per-step data; otherwise just list the staged actor names in a single column.
  int const numStaged = static_cast<int>(actors.size());
  int const numSim = static_cast<int>(simulating ? simNames->size() : 0);
  int const rowCount = (simulating && numSim > numStaged) ? numSim : numStaged;
  int const columns = simulating ? 2 : 1;
  if (ImGui::BeginTable("##StageVsSim", columns, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Stage Name");
    if (simulating) {
      ImGui::TableSetupColumn("Sim Name");
    }
    ImGui::TableHeadersRow();
    for (int i = 0; i < rowCount; ++i) {
      char const* const stageName = i < numStaged ? actors[i].name.c_str() : "";
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      if (simulating) {
        static std::string const kEmptyName;
        std::string const& simName = i < numSim ? (*simNames)[i] : kEmptyName;
        bool const equal = i < numStaged && i < numSim && actors[i].name == simName;
        ImVec4 const color =
            equal ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
        ImGui::TextColored(color, "%s", stageName);
        ImGui::TableNextColumn();
        ImGui::TextColored(color, "%s", simName.c_str());
      } else {
        ImGui::TextUnformatted(stageName);
      }
    }
    ImGui::EndTable();
  }
  ImGui::End();
}

} // namespace superdex::studio
