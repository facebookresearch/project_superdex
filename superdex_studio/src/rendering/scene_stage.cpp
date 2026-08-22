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

#include <mochi_renderer/mesh.h>
#include <mochi_renderer/resource_manager.h>
#include <mochi_renderer/scene.h>
#include <mochi_renderer/scene_object.h>
#include <mochi_renderer/utils.h>

#include "app/app.h"

#include "assets/asset_manager.h"
#include "assets/bot_asset.h"
#include "assets/mochi_model_asset.h"
#include "assets/mochi_prefab_asset.h"
#include "ui/imgui_widgets.h" // HashStringToColor

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/vmatrix.h>

#include <mochi_physics/cpp_api/mochi_actor.h>

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
// are dropped and articulated link shapes are cleared (turning links into shapeless dummy links),
// while articulated transforms, joints, and nested-prefab structure are preserved. Nested prefabs
// are deep-cloned so the original (shared) nested prefabs are not mutated.
static mochi::prefab::ScenePrefab StripArticulated(mochi::prefab::ScenePrefab const& src) {
  mochi::prefab::ScenePrefab out = src;
  out.actors.rigid.clear();
  out.actors.soft.clear();
  out.actors.softSkinned.clear();
  for (auto& art : out.actors.articulated) {
    for (auto& link : art.links) {
      link.shapeFile.clear();
      // Also clear the loaded shape handle: after a simulation run EnsureFullyLoaded populates it
      // on the live prefab, and a populated handle would make AddToScene build a real collider
      // (BVH) instead of a cheap shapeless dummy link.
      link.shape = {};
      link.colliderType = mochi::ColliderType::None;
    }
    if (art.skin.has_value()) {
      art.skin->shapeFile.clear();
      art.skin->shape = {};
    }
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
      outShapeVisible = hasShape;
      break;
    case StageType::RenderModelFallbackToMochiModel:
      outRenderVisible = hasRender;
      outShapeVisible = hasShape && !hasRender;
      break;
  }
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
  // rooted at identity), so they are used directly as world transforms. They come in internal
  // nested-link-actor order, which may differ from the prefab's link declaration order, so each
  // link is paired with its declaration entry by name to fetch the right model/offset/scale.
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
  // between them like it does for rigid actors.
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
    // Skip non-tetrahedral models (no soft surface); the vertex count also seeds the reuse check.
    int const softVertexCount = asset->GetSoftSurfaceVertexCount();
    if (softVertexCount <= 0) {
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
    req.softVertexCount = softVertexCount;
    // Per-instance bake: scale then the shape offset, matching mochi::model_utils::BakeTransform
    // (rt * diag(scale)). The asset re-bakes from these on create/refresh/reset.
    req.softBakeScale = actor.scale * softShapeScale;
    req.softShapeTransform =
        TransformRT{actor.shapeRotation, actor.shapeTranslation * softShapeScale};
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
// copy of the prefab added to `scene` (a scratch scene). Output is ordered to match the depth-first
// articulated-actor traversal (nested prefabs first, then each level's articulated actors). No-op
// (leaves `out` empty) when the prefab contains no articulated actors.
static void ComputeArticulatedLinkTransforms(
    mochi::Scene* scene,
    mochi::prefab::ScenePrefab const& prefab,
    std::vector<ArticulatedData>& out) {
  using namespace mochi;
  out.clear();
  if (!PrefabHasArticulated(prefab)) {
    return;
  }
  mochi::prefab::ScenePrefab const stripped = StripArticulated(prefab);
  ErrorLog e;
  auto const result = mochi::prefab::AddToScene(stripped, scene, {}, e);
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
  auto applyTo = [&](StagedActor::Instance& instance) {
    if (!instance.sceneObject) {
      return;
    }
    VMatrix4x4r const local = Dot4x4(
        ToVMatrix4x4(worldTransform),
        Dot4x4(
            VDiagonalMatrix<4>(ToSimd(actor.worldScale, 1_r)),
            Dot4x4(
                ToVMatrix4x4(instance.modelTransform),
                VDiagonalMatrix<4>(ToSimd(instance.modelScale, 1_r)))));
    auto const [scale, rt] = DecomposeMatrixTransform(local);
    instance.sceneObject->SetLocalTransform(rt, scale, &converter);
  };
  applyTo(actor.render);
  applyTo(actor.shape);
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
    std::vector<StageRequest>& out) {
  using namespace mochi;

  // A bot with no links (e.g. a mod bot whose base is not set yet) has nothing to stage, and
  // instantiating it would fail.
  if (prefab.links.empty()) {
    return;
  }

  // Strip shapes/sensors/colliders for cheap FK-only instantiation.
  auto prefabCopy = prefab;
  for (auto& link : prefabCopy.links) {
    link.colliderType = mochi::ColliderType::None;
    link.shapeFile.clear();
    link.sensors.clear();
  }
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

  out.reserve(out.size() + prefab.links.size());
  for (int i = 0; i < isize(prefab.links); ++i) {
    auto const& link = prefab.links[i];
    auto* linkActor = scene->GetActor(linkActors[i]);

    StageRequest req;
    req.name = linkActor->GetName();
    req.worldTransform = linkActor->GetRootTransform();
    // Stage both representations (either may be absent); bot links reference model files directly.
    if (!link.renderModelFile.empty()) {
      req.renderModelFile = mochi::Path(link.renderModelFile);
      req.renderModelTransform = TransformRT(link.renderModelRotation, link.renderModelTranslation);
      req.renderModelScale = link.renderModelScale;
    }
    if (!link.shapeFile.empty()) {
      req.shapeModelFile = mochi::Path(link.shapeFile);
      req.shapeModelTransform = TransformRT(link.shapeRotation, link.shapeTranslation);
      req.shapeModelScale = link.shapeScale;
    }
    ComputeStageVisibility(
        !req.renderModelFile.IsEmpty(),
        !req.shapeModelFile.IsEmpty(),
        stageType,
        req.renderVisible,
        req.shapeVisible);
    out.push_back(std::move(req));
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

bool SceneStage::StageBot(superdex::robotics::BotPrefab const& prefab, StageType stageType) {
  if (!_studio || !_scene || !_renderScene) {
    return false;
  }
  std::vector<StageRequest> requests;
  BuildBotRequests(_scene, _studio, prefab, stageType, requests);
  return ApplyStageRequests(requests);
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

  return ApplyStageRequests(requests);
}

#if MOCHI_INTERNAL
bool SceneStage::StageBotScene(
    superdex::robotics::BotScenePrefab const& scene,
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

  // 2. Spawnable prefabs, in declaration order. Loaded into the simulation by default (see
  // BotSceneEditor::CreatePhysicsActors); stage them under their entry name so the staged actor
  // names/order align with the physics actors that prefab::AddToScene creates.
  for (auto const& prefabEntry : scene.scene.spawnablePrefabs) {
    if (prefabEntry.path.empty()) {
      continue;
    }
    auto* prefabAsset =
        manager.FindAssetByPath<MochiPrefabAsset>(Path{std::string(prefabEntry.path)});
    if (!prefabAsset) {
      continue;
    }
    auto const& spawnPrefab = prefabAsset->GetPrefab();
    std::vector<ArticulatedData> articulatedFk;
    ComputeArticulatedLinkTransforms(_scene, spawnPrefab, articulatedFk);
    std::string_view const source = spawnPrefab.sourceFilePath.has_value()
        ? std::string_view(*spawnPrefab.sourceFilePath)
        : std::string_view();
    int artCursor = 0;
    BuildPrefabRequests(
        _studio,
        spawnPrefab,
        prefabAsset->GetAssetsRoot(),
        source,
        TransformSRT::Identity(),
        stageType,
        articulatedFk,
        artCursor,
        std::string(prefabEntry.name),
        -1,
        requests);
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
    BuildBotRequests(_scene, _studio, botPrefab, stageType, requests);
  }

  return ApplyStageRequests(requests);
}
#endif // MOCHI_INTERNAL

bool SceneStage::ApplyStageRequests(std::vector<StageRequest> const& requests) {
  using namespace mochi;
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
      bool const actorIsSoft = _actors[i].dynamicSolidMesh != nullptr;
      // A soft/non-soft mismatch, or a soft vertex-count change (topology change), forces a
      // rebuild.
      if (_actors[i].name != requests[i].name ||
          _actors[i].render.modelFile != requests[i].renderModelFile ||
          _actors[i].shape.modelFile != requests[i].shapeModelFile || reqIsSoft != actorIsSoft ||
          (reqIsSoft && _actors[i].softMeshVertexCount != requests[i].softVertexCount)) {
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

    // Soft actors bake scale/shape-offset into their geometry, so an edit that changes the rest
    // geometry but not the vertex count (e.g. scaling a nested prefab) reuses the meshes. Re-bake
    // the reused solid + wireframe meshes via the asset so the idle pose stays correct (the rebuild
    // path below handles vertex-count changes).
    if (reuse && req.isSoft && actor.dynamicSolidMesh) {
      // Re-fetch the asset from the AssetManager instead of trusting the cached raw pointer: the
      // manager owns assets as unique_ptr and may have evicted/reloaded this one since it was
      // staged, leaving actor.softAsset dangling. Refreshing here keeps it valid for this and later
      // dereferences (e.g. ResetWorldTransforms).
      actor.softAsset = manager.LoadMochiModelAsset(req.shapeModelFile);
      if (actor.softAsset) {
        actor.softAsset->UpdateSoftDynamicMeshes(
            actor.dynamicSolidMesh,
            actor.dynamicWireframeMesh,
            req.softBakeScale,
            req.softShapeTransform);
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
      if (req.isSoft) {
        // Soft actors stage two deforming dynamic meshes created + colored by the MochiModelAsset:
        // a solid mesh in the `render` slot (StageType 1/2) and a wireframe mesh in the `shape`
        // slot (StageType 3). Both are flat-lit, so the highlight clone matches their shading. The
        // asset is kept so idle refresh / simulation-stop can re-bake the rest pose.
        actor.render.flatShaded = true;
        auto* asset = manager.LoadMochiModelAsset(req.shapeModelFile);
        std::unique_ptr<mochi_renderer::Mesh> solid;
        std::unique_ptr<mochi_renderer::WireframeMesh> wireframe;
        if (asset) {
          asset->CreateSoftDynamicMeshes(
              req.softBakeScale, req.softShapeTransform, solid, wireframe);
        }
        actor.softAsset = asset;
        actor.dynamicSolidMesh = solid.get();
        actor.dynamicWireframeMesh = wireframe.get();
        if (solid) {
          solid->SetName(req.name);
        }
        if (wireframe) {
          wireframe->SetName(req.name);
        }
        actor.render.sceneObject =
            solid ? _renderScene->AddSceneObjectToScene(std::move(solid)) : nullptr;
        actor.shape.sceneObject =
            wireframe ? _renderScene->AddSceneObjectToScene(std::move(wireframe)) : nullptr;
        actor.softMeshVertexCount = req.softVertexCount;
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
    actor.render.visible = req.renderVisible;
    actor.shape.visible = req.shapeVisible;
    if (actor.render.sceneObject) {
      actor.render.sceneObject->SetVisible(req.renderVisible);
    }
    if (actor.shape.sceneObject) {
      actor.shape.sceneObject->SetVisible(req.shapeVisible);
    }
    actor.sceneObject = (req.renderVisible && actor.render.sceneObject) ? actor.render.sceneObject
        : (req.shapeVisible && actor.shape.sceneObject)                 ? actor.shape.sceneObject
                                                                        : nullptr;

    SetStagedActorWorldTransform(actor, actor.worldTransform, converter);
  }

  if (!reuse) {
    _nameToIndex.clear();
    for (int i = 0; i < isize(_actors); ++i) {
      _nameToIndex[_actors[i].name] = i;
    }
  }
  return !reuse;
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
      if (actor.render.sceneObject) {
        _renderScene->DestroySceneObject(actor.render.sceneObject);
      }
      if (actor.shape.sceneObject) {
        _renderScene->DestroySceneObject(actor.shape.sceneObject);
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
    if (actor.dynamicSolidMesh) {
      actor.softAsset = _studio->GetAssetManager().LoadMochiModelAsset(actor.shape.modelFile);
      if (actor.softAsset) {
        actor.softAsset->UpdateSoftDynamicMeshes(
            actor.dynamicSolidMesh,
            actor.dynamicWireframeMesh,
            actor.softBakeScale,
            actor.softShapeTransform);
      }
    }
    SetStagedActorWorldTransform(actor, actor.defaultWorldTransform, converter);
  }
}

void SceneStage::ApplySoftMeshUpdates(
    mochi::Span<SoftMeshUpdate const> updates,
    mochi::CoordinateSpaceConverter const& converter) {
  for (auto const& update : updates) {
    int const index = FindIndexByName(update.name);
    if (index < 0) {
      continue;
    }
    auto& actor = _actors[index];
    if (!actor.dynamicSolidMesh) {
      continue;
    }
    // A vertex-count mismatch means the meshes were rebuilt underneath this update; skip it.
    if (static_cast<int>(update.positions.size() / 3) != actor.softMeshVertexCount) {
      continue;
    }
    actor.dynamicSolidMesh->Update(
        mochi::MakeConstSpan(update.positions), mochi::MakeConstSpan(update.normals));
    if (actor.dynamicWireframeMesh) {
      actor.dynamicWireframeMesh->Update(
          mochi::MakeConstSpan(update.positions), mochi::MakeConstSpan(update.normals));
    }
    SetStagedActorWorldTransform(actor, update.worldTransform, converter);
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

int SceneStage::GetSceneObjectIndex(SceneObject const* object) const {
  if (!object) {
    return -1;
  }
  for (int i = 0; i < static_cast<int>(_actors.size()); ++i) {
    if (_actors[i].render.sceneObject == object || _actors[i].shape.sceneObject == object) {
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
    for (auto* inst : {&actor.render, &actor.shape}) {
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
