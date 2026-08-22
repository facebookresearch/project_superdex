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

#include "mochi_articulated_actor_params.h"
#include "mochi_ecs_utils.h"
#include "mochi_scene.h"

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/path.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace mochi;
using namespace mochi::experimental;
using namespace mochi::prefab;

namespace {

// Maps a prefab-local actor name to the resulting ActorHandle. Used to resolve actors referenced by
// name by constraints, pose controllers, and contact filters. Names need not be unique within the
// prefab (nor within the scene). A name claimed by more than one actor is tracked as ambiguous (see
// ActorNameRegistry) so resolving it fails loudly instead of silently binding to one of them.
using ActorNameToHandleMap = std::unordered_map<std::string, ActorHandle>;

// Actor-name resolution state for one prefab subtree: the name -> handle map plus the set of names
// claimed by more than one actor. A duplicated name is permitted as long as nothing references it
// by name. Referencing an ambiguous name is an error. Each AddToSceneImpl builds a registry local
// to its own subtree and merges its nested prefabs' registries into it (see
// MergeActorNameRegistry).
struct ActorNameRegistry {
  ActorNameToHandleMap handles;
  std::unordered_set<std::string> ambiguousNames;
};

class ActivePrefabGuard {
 public:
  ActivePrefabGuard(
      DynamicArray<ScenePrefab const*>& activePrefabs,
      ScenePrefab const* prefab,
      Error& error)
      : _activePrefabs(activePrefabs) {
    MOCHI_ERROR_IF(
        Contains(activePrefabs, prefab),
        error,
        "A prefab references itself, directly or indirectly.");
    MOCHI_ERROR_RETURN(error);
    activePrefabs.push_back(prefab);
    _pushed = true;
  }

  ~ActivePrefabGuard() {
    if (_pushed) {
      _activePrefabs.pop_back();
    }
  }

  MOCHI_DECLARE_NO_COPY_NO_MOVE(ActivePrefabGuard);

 private:
  DynamicArray<ScenePrefab const*>& _activePrefabs;
  bool _pushed = false;
};

} // namespace

// Resolve a path referenced inside a prefab to a full path. See the DSL
// (mochi_physics_prefab.mochi_gen) for documentation.
MOCHI_API DynamicString prefab::GetPrefabFullPath(
    std::string_view inputPath,
    std::string_view rootForRelativePath,
    std::string_view prefabFilePath) {
  if (!inputPath.empty() && inputPath.starts_with("./") && !prefabFilePath.empty()) {
    auto prefabDir = std::filesystem::path(prefabFilePath).parent_path();
    std::string_view relativePart = inputPath.substr(2); // Skip "./"
    auto resolvedPathObj = prefabDir / std::filesystem::path(relativePart);
    return DynamicString(resolvedPathObj.string());
  }
  return DynamicString(path::GetFullPath(inputPath, rootForRelativePath));
}

[[nodiscard]] static DynamicString GetPrefabPathKey(std::string_view path) {
  // Canonicalize so different spellings of one file (symlinks, "./a/../b") share a cycle key. Case
  // isn't folded on purpose: weakly_canonical resolves existing files (the only ones that can
  // recurse) to their on-disk casing anyway, and folding would false-positive on case-sensitive
  // filesystems.
  std::filesystem::path const inputPath(path);
  std::error_code ec;
  auto const canonicalPath = std::filesystem::weakly_canonical(inputPath, ec);
  return DynamicString((ec ? inputPath.lexically_normal() : canonicalPath).string());
}

MOCHI_API ScenePrefab prefab::ShallowLoadFromJsonString(std::string_view json, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  ScenePrefab prefab;
  bool success = SReflect::FromJsonString(
      prefab, std::string(json), SReflect::DeserializeFlags::WarnIfExtraneousFields);
  MOCHI_ERROR_IF(
      !success, error, "Failed to parse prefab from JSON string. See log (above) for details.");
  MOCHI_ERROR_RETURN(error, {});

  return prefab;
}

MOCHI_API ScenePrefab prefab::ShallowLoadFromFile(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  ScenePrefab prefab;
  int numIssues = 0;
  bool success = SReflect::LoadFromJsonFile(
      prefab,
      DynamicString(path).c_str(),
      SReflect::DeserializeFlags::WarnIfExtraneousFields,
      numIssues);
  MOCHI_ERROR_IF(
      !success, error, "Failed to open prefab file for reading. See log (above) for details.");
  MOCHI_ERROR_RETURN(error, {}); // No partial results

  // Store where this prefab was loaded from so we can resolve prefab-relative paths
  prefab.sourceFilePath = DynamicString{path};

  return prefab;
}

static void LoadNestedPrefabsWithCycleCheck(
    ScenePrefab& prefab,
    std::string_view rootPath,
    bool skipLoaded,
    DynamicArray<DynamicString>& activePrefabPaths,
    DynamicArray<ScenePrefab const*>& activePrefabs,
    Error& error) {
  ActivePrefabGuard activePrefabGuard(activePrefabs, &prefab, error);
  MOCHI_ERROR_RETURN(error);

  // Use the prefab's own sourceFilePath for resolving prefab-relative paths
  std::string_view prefabFilePath = prefab.sourceFilePath.has_value()
      ? std::string_view(*prefab.sourceFilePath)
      : std::string_view();

  for (auto& nested : prefab.prefabs) {
    auto const fullPath = GetPrefabFullPath(nested.path, rootPath, prefabFilePath);
    // Track only file-backed references for cycle detection, keyed on the resolved path but gated
    // on the reference's own path: an in-memory (pathless) reference resolves to rootPath, which is
    // non-empty and identical across siblings, so gating on fullPath would falsely flag two
    // pathless references on one branch as a cycle.
    bool const tracksActivePath = !nested.path.empty();
    if (tracksActivePath) {
      auto const pathKey = GetPrefabPathKey(fullPath);
      MOCHI_ERROR_IF(
          Contains(activePrefabPaths, pathKey),
          error,
          "A prefab references itself, directly or indirectly.");
      MOCHI_ERROR_RETURN(error);
      activePrefabPaths.push_back(pathKey);
    }
    MOCHI_DEFER(if (tracksActivePath) { activePrefabPaths.pop_back(); });
    if (!skipLoaded || !nested.prefab) {
      nested.prefab = std::make_shared<ScenePrefab>(ShallowLoadFromFile(fullPath, error));
      MOCHI_ERROR_RETURN(error);
    }
    LoadNestedPrefabsWithCycleCheck(
        *nested.prefab, rootPath, skipLoaded, activePrefabPaths, activePrefabs, error);
    MOCHI_ERROR_RETURN(error);
  }
}

// Seeds fresh active stacks for LoadNestedPrefabsWithCycleCheck. Ownership stays in this file
// so no caller can accidentally share or reuse a stack and break cycle detection.
static void LoadNestedPrefabsImpl(
    ScenePrefab& prefab,
    std::string_view rootPath,
    bool skipLoaded,
    Error& error) {
  DynamicArray<DynamicString> activePrefabPaths;
  DynamicArray<ScenePrefab const*> activePrefabs;
  LoadNestedPrefabsWithCycleCheck(
      prefab, rootPath, skipLoaded, activePrefabPaths, activePrefabs, error);
}

MOCHI_API void
prefab::LoadNestedPrefabs(ScenePrefab& prefab, std::string_view rootPath, Error& error) {
  LoadNestedPrefabsImpl(prefab, rootPath, false /*skipLoaded*/, error);
}

// This struct gets converted to string.
// That string is used as the key in a cache of loaded shapes.
namespace {

struct MochiShapeKey {
  DynamicString fullPath;
  Real3 shapeScale{};
  Quaternion shapeRotation{};
  Real3 shapeTranslation{};
  MOCHI_STRUCT_BEGIN(MochiShapeKey)
  MOCHI_FIELD(fullPath)
  MOCHI_FIELD(shapeScale)
  MOCHI_FIELD(shapeRotation)
  MOCHI_FIELD(shapeTranslation)
  MOCHI_STRUCT_END()
};
using MochiShapeCache = std::unordered_map<std::string, ShapeHandle>;
} // namespace

static void LoadPrefabActorShape(
    Context* context,
    MochiShapeCache& shapeCache,
    std::string_view shapePath,
    std::string_view rootPath,
    std::string_view prefabFilePath,
    Real3 const& shapeScale,
    Quaternion const& shapeRotation,
    Real3 const& shapeTranslation,
    ShapeHandle& outShapeHandle,
    bool skipLoaded,
    Error& error) {
  if (skipLoaded && outShapeHandle.IsValid()) {
    return;
  }
  MOCHI_ERROR_IF(shapePath.empty(), error, "Empty shape file path.");
  MOCHI_ERROR_RETURN(error);

  // Create key
  MochiShapeKey key;
  key.fullPath = GetPrefabFullPath(shapePath, rootPath, prefabFilePath);
  key.shapeScale = shapeScale;
  key.shapeRotation = shapeRotation;
  key.shapeTranslation = shapeTranslation;
  std::string keyStr = SReflect::ToJsonString(key, false);

  // Check cache
  auto& cachedHandleRef = shapeCache[keyStr];
  if (cachedHandleRef.IsValid()) {
    // Already loaded
    outShapeHandle = cachedHandleRef;
    return;
  }

  MOCHI_ERROR_IF(!std::filesystem::exists(key.fullPath), error, "File not found.");
  MOCHI_ERROR_RETURN(error);

  // Load it
  outShapeHandle = context->LoadShapeFromFile(
      key.fullPath, key.shapeScale, TransformRT{shapeRotation, shapeTranslation}, error);
  MOCHI_ERROR_RETURN(error);

  // Cache it
  cachedHandleRef = outShapeHandle;
}

static void LoadActorShapes(
    ArticulatedActorPrefab& actor,
    Context* context,
    MochiShapeCache& shapeCache,
    std::string_view rootPath,
    std::string_view prefabFilePath,
    real scaleModifier,
    bool skipLoaded,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Load link shapes
  for (auto& link : actor.links) {
    if (link.shapeFile.empty()) {
      continue; // Dummy links have no shape file
    }
    real const actorScale = actor.scale * scaleModifier;
    LoadPrefabActorShape(
        context,
        shapeCache,
        link.shapeFile,
        rootPath,
        prefabFilePath,
        link.shapeScale * actorScale, // Bake scale
        link.shapeRotation, // Bake rotation
        link.shapeTranslation * actorScale, // Bake translation
        link.shape,
        skipLoaded,
        error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR(
          "Prefab failed to load articulated link shape \"%s\", referenced by articulated actor \"%s\"",
          GetPrefabFullPath(link.shapeFile, rootPath, prefabFilePath).c_str(),
          actor.name.c_str());
      return;
    }
  }

  // Skinned shape is optional
  if (actor.skin && !actor.skin->shapeFile.empty()) {
    LoadPrefabActorShape(
        context,
        shapeCache,
        actor.skin->shapeFile,
        rootPath,
        prefabFilePath,
        scaleModifier * Real3{actor.scale, actor.scale, actor.scale}, // bake uniform scale
        Quaternion::Identity(), // bake rotation
        Real3{}, // bake translation
        actor.skin->shape,
        skipLoaded,
        error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR(
          "Prefab failed to load skin shape \"%s\", referenced by articulated actor \"%s\"",
          GetPrefabFullPath(actor.skin->shapeFile, rootPath, prefabFilePath).c_str(),
          actor.name.c_str());
      return;
    }
  }
}

static void LoadActorShapes(
    RigidActorPrefab& actor,
    Context* context,
    MochiShapeCache& shapeCache,
    std::string_view rootPath,
    std::string_view prefabFilePath,
    real scaleModifier,
    bool skipLoaded,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  LoadPrefabActorShape(
      context,
      shapeCache,
      actor.shapeFile,
      rootPath,
      prefabFilePath,
      actor.scale * Real3{scaleModifier, scaleModifier, scaleModifier},
      actor.shapeRotation,
      actor.shapeTranslation * scaleModifier,
      actor.shape,
      skipLoaded,
      error);
  if (!error.IsOK()) {
    MOCHI_LOG_ERROR(
        "Prefab failed to load shape \"%s\", referenced by rigid actor \"%s\"",
        GetPrefabFullPath(actor.shapeFile, rootPath, prefabFilePath).c_str(),
        actor.name.c_str());
    return;
  }
}

static void LoadActorShapes(
    SoftActorPrefab& actor,
    Context* context,
    MochiShapeCache& shapeCache,
    std::string_view rootPath,
    std::string_view prefabFilePath,
    real scaleModifier,
    bool skipLoaded,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  LoadPrefabActorShape(
      context,
      shapeCache,
      actor.shapeFile,
      rootPath,
      prefabFilePath,
      actor.scale * scaleModifier,
      actor.shapeRotation,
      actor.shapeTranslation * scaleModifier,
      actor.shape,
      skipLoaded,
      error);
  if (!error.IsOK()) {
    MOCHI_LOG_ERROR(
        "Prefab failed to load shape \"%s\", referenced by soft actor \"%s\"",
        GetPrefabFullPath(actor.shapeFile, rootPath, prefabFilePath).c_str(),
        actor.name.c_str());
    return;
  }

  // Flow is optional
  if (!actor.flowFile.empty()) {
    LoadPrefabActorShape(
        context,
        shapeCache,
        actor.flowFile,
        rootPath,
        prefabFilePath,
        Real3{scaleModifier, scaleModifier, scaleModifier},
        Quaternion::Identity(),
        Real3{},
        actor.flow,
        skipLoaded,
        error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR(
          "Prefab failed to load Deep Flow file \"%s\", referenced by soft actor \"%s\"",
          GetPrefabFullPath(actor.flowFile, rootPath, prefabFilePath).c_str(),
          actor.name.c_str());
      return;
    }
  }
}

static void LoadActorShapes(
    SoftSkinnedActorPrefab& actor,
    Context* context,
    MochiShapeCache& shapeCache,
    std::string_view rootPath,
    std::string_view prefabFilePath,
    real scaleModifier,
    bool skipLoaded,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  LoadActorShapes(
      actor.skeletonParams,
      context,
      shapeCache,
      rootPath,
      prefabFilePath,
      scaleModifier,
      skipLoaded,
      error);
  for (auto& soft : actor.softParams) {
    LoadActorShapes(
        soft, context, shapeCache, rootPath, prefabFilePath, scaleModifier, skipLoaded, error);
  }
}

template <class ParamsArrayT>
static void LoadActorListShapes(
    Context* context,
    MochiShapeCache& shapeCache,
    std::string_view rootPath,
    std::string_view prefabFilePath,
    real scaleModifier,
    ParamsArrayT& actorList,
    bool skipLoaded,
    Error& error) {
  for (auto& actor : actorList) {
    LoadActorShapes(
        actor, context, shapeCache, rootPath, prefabFilePath, scaleModifier, skipLoaded, error);
  }
}

static void LoadShapesWithCycleCheck(
    ScenePrefab& prefab,
    std::string_view rootPath,
    Context* context,
    real scaleModifier,
    bool skipLoaded,
    DynamicArray<ScenePrefab const*>& activePrefabs,
    Error& error) {
  ActivePrefabGuard activePrefabGuard(activePrefabs, &prefab, error);
  MOCHI_ERROR_RETURN(error);

  // Use the prefab's own sourceFilePath for resolving prefab-relative paths
  std::string_view prefabFilePath = prefab.sourceFilePath.has_value()
      ? std::string_view(*prefab.sourceFilePath)
      : std::string_view();

  // Shape cache
  MochiShapeCache shapeCache;

  // Actor Types
  LoadActorListShapes(
      context,
      shapeCache,
      rootPath,
      prefabFilePath,
      scaleModifier,
      prefab.actors.articulated,
      skipLoaded,
      error);
  LoadActorListShapes(
      context,
      shapeCache,
      rootPath,
      prefabFilePath,
      scaleModifier,
      prefab.actors.rigid,
      skipLoaded,
      error);
  LoadActorListShapes(
      context,
      shapeCache,
      rootPath,
      prefabFilePath,
      scaleModifier,
      prefab.actors.soft,
      skipLoaded,
      error);
  LoadActorListShapes(
      context,
      shapeCache,
      rootPath,
      prefabFilePath,
      scaleModifier,
      prefab.actors.softSkinned,
      skipLoaded,
      error);

  // Nested Prefabs (recursive)
  for (auto& nested : prefab.prefabs) {
    if (nested.prefab) {
      LoadShapesWithCycleCheck(
          *nested.prefab,
          rootPath,
          context,
          nested.scale * scaleModifier,
          skipLoaded,
          activePrefabs,
          error);
      if (!error.IsOK()) {
        std::string_view nestedPrefabPath = nested.prefab->sourceFilePath.has_value()
            ? *nested.prefab->sourceFilePath
            : nested.path;
        MOCHI_LOG_ERROR(
            "Prefab failed to load shapes for nested prefab \"%s\"", nestedPrefabPath.data());
        return;
      }
    }
  }
}

static void LoadShapesImpl(
    ScenePrefab& prefab,
    std::string_view rootPath,
    Context* context,
    real scaleModifier,
    bool skipLoaded,
    Error& error) {
  DynamicArray<ScenePrefab const*> activePrefabs;
  LoadShapesWithCycleCheck(
      prefab, rootPath, context, scaleModifier, skipLoaded, activePrefabs, error);
}

MOCHI_API void
prefab::LoadShapes(ScenePrefab& prefab, std::string_view rootPath, Context* context, Error& error) {
  return LoadShapesImpl(
      prefab, rootPath, context, 1_r /*scaleModifier*/, false /*skipLoaded*/, error);
}

MOCHI_API void prefab::EnsureFullyLoaded(
    ScenePrefab& prefab,
    std::string_view rootPath,
    Context* context,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  LoadNestedPrefabsImpl(prefab, rootPath, true /*skipLoaded*/, error);
  LoadShapesImpl(prefab, rootPath, context, 1_r /*scaleModifier*/, true /*skipLoaded*/, error);
}

// Loads a prefab from file, baking `scaleModifier` into its geometry (see LoadShapesImpl). The
// public LoadFromFile passes 1 (geometry at authored scale); the prefabPath-based AddToScene passes
// PrefabParams::scale so a non-identity instance scale is baked into geometry, matching the scale
// it applies to poses, constraints, and inertia. The ScenePrefab AddToScene overload cannot do this
// -- its geometry is already baked before it runs.
static ScenePrefab LoadFromFileImpl(
    std::string_view prefabPath,
    std::string_view rootPath,
    Context* context,
    real scaleModifier,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ScenePrefab prefab = ShallowLoadFromFile(prefabPath, error);
  LoadNestedPrefabsImpl(prefab, rootPath, false /*skipLoaded*/, error);
  LoadShapesImpl(prefab, rootPath, context, scaleModifier, false /*skipLoaded*/, error);
  return error.IsOK() ? prefab : ScenePrefab{};
}

MOCHI_API ScenePrefab prefab::LoadFromFile(
    std::string_view prefabPath,
    std::string_view rootPath,
    Context* context,
    Error& error) {
  return LoadFromFileImpl(prefabPath, rootPath, context, 1_r /*scaleModifier*/, error);
}

MOCHI_API ScenePrefab prefab::LoadFromJsonString(
    std::string_view json,
    std::string_view rootPath,
    Context* context,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ScenePrefab prefab = ShallowLoadFromJsonString(json, error);
  LoadNestedPrefabs(prefab, rootPath, error);
  LoadShapes(prefab, rootPath, context, error);
  return error.IsOK() ? prefab : ScenePrefab{};
}

MOCHI_API void
prefab::SaveToJsonFile(ScenePrefab const& prefab, std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto contents = SaveToJsonString(prefab, error);
  WriteFile(path, contents, error);
}

MOCHI_API DynamicString prefab::SaveToJsonString(ScenePrefab const& prefab, Error& error) {
  MOCHI_ERROR_RETURN(error, "");
  auto json = SReflect::ToJsonString(prefab, true);
  return DynamicString{json.c_str(), json.size()};
}

static DynamicString CombineNames(DynamicString const& path, DynamicString const& name) {
  if (path.empty()) {
    return name;
  } else if (name.empty()) {
    return path;
  } else {
    return path + "/" + name;
  }
}

// Records actorName -> handle. A second actor claiming an existing name marks that name ambiguous.
// Empty names are ignored: unnamed actors cannot be referenced by name.
static void RegisterActorName(
    ActorNameRegistry& actorNames,
    DynamicString const& actorName,
    ActorHandle handle) {
  if (actorName.empty()) {
    return;
  }
  auto const [it, inserted] = actorNames.handles.try_emplace(std::string(actorName), handle);
  if (!inserted) {
    actorNames.ambiguousNames.insert(it->first);
  }
}

// Merges a nested subtree's name registry into this level's. A name claimed in both (or already
// ambiguous in the source) becomes ambiguous here, so a reference at this level or above to a name
// reused across the subtree fails. References already resolved inside the source subtree are
// unaffected.
static void MergeActorNameRegistry(ActorNameRegistry& dst, ActorNameRegistry const& src) {
  for (auto const& [name, handle] : src.handles) {
    if (!dst.handles.try_emplace(name, handle).second) {
      dst.ambiguousNames.insert(name);
    }
  }
  dst.ambiguousNames.insert(src.ambiguousNames.begin(), src.ambiguousNames.end());
}

// Resolves fullName to its handle iterator in actorNames. Returns end() if error is already set,
// or -- with error set -- if the name is ambiguous (claimed by more than one actor), since an
// ambiguous name cannot identify a single actor. Otherwise returns the lookup result; the caller
// compares it against actorNames.handles.end() for its own not-found handling. The name is
// converted to the std::string map key once and reused for the ambiguity check and the lookup.
[[nodiscard]] static ActorNameToHandleMap::const_iterator
FindActorHandle(ActorNameRegistry const& actorNames, DynamicString const& fullName, Error& error) {
  MOCHI_ERROR_RETURN(error, actorNames.handles.end());
  std::string const key(fullName);
  if (actorNames.ambiguousNames.contains(key)) {
    MOCHI_LOG_ERROR(
        "Prefab actor name \"%s\" is used by multiple actors and cannot be referenced by a "
        "constraint, pose controller, or contact filter.",
        fullName.c_str());
    MOCHI_ERROR_SET(
        error,
        "Ambiguous prefab actor name referenced by a constraint, pose controller, or contact filter.");
    return actorNames.handles.end();
  }
  return actorNames.handles.find(key);
}

// Cumulative prefab metric scale, keyed by the concrete actor handle referenced by prefab logic.
// This can differ from the constraint owner's prefab scale when a parent prefab references an actor
// inside a scaled nested prefab. Standalone rigid/soft shape-bake scale fields are not included;
// articulated actor scale is included because it scales authored joint/link distances below.
using ActorPrefabMetricScaleMap = std::unordered_map<ActorHandle, real>;

// Applies actor-vs-actor contact entries: resolves each named actor pair and invokes enableContact.
static void ApplyActorContactEntries(
    DynamicArray<ActorContactEntry> const& entries,
    DynamicString const& baseName,
    ActorNameRegistry const& actorNames,
    Scene* scene,
    void (Scene::*enableContact)(ActorHandle, ActorHandle, bool, IncludeNestedActors, Error&),
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  for (auto const& entry : entries) {
    MOCHI_ERROR_IF(
        entry.actors.size() != 2, error, "ActorContactEntry must have exactly 2 actors.");
    MOCHI_ERROR_RETURN(error);
    auto nameToFindA = CombineNames(baseName, entry.actors[0]);
    auto nameToFindB = CombineNames(baseName, entry.actors[1]);
    auto const itA = FindActorHandle(actorNames, nameToFindA, error);
    auto const itB = FindActorHandle(actorNames, nameToFindB, error);
    MOCHI_ERROR_RETURN(error);

    if (itA == actorNames.handles.end()) {
      MOCHI_LOG_ERROR(
          "Actor contact filter references actor \"%s\" which was not found.", nameToFindA.c_str());
      MOCHI_ERROR_SET(error, "Failed to find actor referenced by contact filter.");
      return;
    }
    if (itB == actorNames.handles.end()) {
      MOCHI_LOG_ERROR(
          "Actor contact filter references actor \"%s\" which was not found.", nameToFindB.c_str());
      MOCHI_ERROR_SET(error, "Failed to find actor referenced by contact filter.");
      return;
    }

    auto const includeNestedActors =
        entry.includeNestedActors ? IncludeNestedActors::Yes : IncludeNestedActors::No;
    (scene->*enableContact)(itA->second, itB->second, entry.enable, includeNestedActors, error);
    MOCHI_ERROR_RETURN(error);
  }
}

static void ApplyContactFilter(
    ContactFilter const& filter,
    DynamicString const& baseName,
    ActorNameRegistry const& actorNames,
    Scene* scene,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Apply asymmetric layer contact filters.
  if (filter.layerContactAsymmetric.has_value()) {
    for (auto const& entry : *filter.layerContactAsymmetric) {
      MOCHI_ERROR_IF(
          entry.layers.size() != 2, error, "LayerContactEntry must have exactly 2 layers.");
      MOCHI_ERROR_RETURN(error);
      scene->EnableLayerContactAsymmetric(entry.layers[0], entry.layers[1], entry.enable, error);
      MOCHI_ERROR_RETURN(error);
    }
  }

  // Apply symmetric layer contact filters.
  if (filter.layerContactSymmetric.has_value()) {
    for (auto const& entry : *filter.layerContactSymmetric) {
      MOCHI_ERROR_IF(
          entry.layers.size() != 2, error, "LayerContactEntry must have exactly 2 layers.");
      MOCHI_ERROR_RETURN(error);
      scene->EnableLayerContactSymmetric(entry.layers[0], entry.layers[1], entry.enable, error);
      MOCHI_ERROR_RETURN(error);
    }
  }

  // Apply asymmetric actor contact filters.
  if (filter.actorContactAsymmetric.has_value()) {
    ApplyActorContactEntries(
        *filter.actorContactAsymmetric,
        baseName,
        actorNames,
        scene,
        &Scene::EnableActorContactAsymmetric,
        error);
    MOCHI_ERROR_RETURN(error);
  }

  // Apply symmetric actor contact filters.
  if (filter.actorContactSymmetric.has_value()) {
    ApplyActorContactEntries(
        *filter.actorContactSymmetric,
        baseName,
        actorNames,
        scene,
        &Scene::EnableActorContactSymmetric,
        error);
    MOCHI_ERROR_RETURN(error);
  }
}

static ActorHandle FindActorHandleForConstraint(
    std::string_view basePath,
    std::string_view relPath,
    ActorNameRegistry const& actorNames,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto actorNameToFind = CombineNames(DynamicString(basePath), DynamicString(relPath));
  auto const it = FindActorHandle(actorNames, actorNameToFind, error);
  MOCHI_ERROR_RETURN(error, {});
  if (it == actorNames.handles.end()) {
    MOCHI_LOG_ERROR(
        "Constraint references actor \"%s\" which was not found.", actorNameToFind.c_str());
    MOCHI_ERROR_SET(error, "Failed to find actor referenced by constraint.");
    return {};
  }
  return it->second;
}

// clang-format off
static Constraint* AddConstraintImpl(Scene* scene, ArticulatedSingleDofRangeConstraintParams const& params, Error& error) { return scene->CreateArticulatedSingleDofRangeConstraint(params, error); }
static Constraint* AddConstraintImpl(Scene* scene, Articulated3dRotationRangeConstraintParams const& params, Error& error) { return scene->CreateArticulated3dRotationRangeConstraint(params, error); }
static Constraint* AddConstraintImpl(Scene* scene, ArticulatedSingleDofTargetConstraintParams const& params, Error& error) { return scene->CreateArticulatedSingleDofTargetConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, Articulated3dRotationTargetConstraintParams const& params, Error& error) { return scene->CreateArticulated3dRotationTargetConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, JointRotationRangeConstraintParams const& params, Error& error) { return scene->CreateJointRotationRangeConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, JointRotationTrackingConstraintParams const& params, Error& error) { return scene->CreateJointRotationTrackingConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, RigidPivotPositionConstraintParams const& params, Error& error) { return scene->CreateRigidPivotPositionConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, RigidPivotToRigidTargetConstraintParams const& params, Error& error) { return scene->CreateRigidPivotToRigidTargetConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, RigidPivotRotationConstraintParams const& params, Error& error) { return scene->CreateRigidPivotRotationConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, RigidPrismaticJointConstraintParams const& params, Error& error) { return scene->CreateRigidPrismaticJointConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, RigidSphericalJointConstraintParams const& params, Error& error) { return scene->CreateRigidSphericalJointConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, DeformableNodePositionConstraintParams const& params, Error& error) { return scene->CreateDeformableNodePositionConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, DeformableNodeToDeformableNodeConstraintParams const& params, Error& error) { return scene->CreateDeformableNodeToDeformableNodeConstraint(params, error);}
static Constraint* AddConstraintImpl(Scene* scene, DeformableNodeToRigidConstraintParams const& params, Error& error) { return scene->CreateDeformableNodeToRigidConstraint(params, error);}
// clang-format on

[[nodiscard]] static Real3 TransformRotationVector(
    TransformSRT const& worldFromPrefab,
    Real3 const& rotationVector) {
  // Compose the prefab rotation onto the rotation vector, with the prefab rotation applied in the
  // parent frame.
  return (worldFromPrefab.GetRotation() * Quaternion::FromRotationVector(rotationVector))
      .ToRotationVector();
}

[[nodiscard]] static real GetActorPrefabMetricScale(
    ActorPrefabMetricScaleMap const& actorPrefabMetricScale,
    ActorHandle actor) {
  auto const it = actorPrefabMetricScale.find(actor);
  MOCHI_ASSERT(
      it != actorPrefabMetricScale.end(),
      "A resolved actor handle is missing its prefab metric scale.");
  return it->second;
}

static void ApplyPrefabTransformToConstraint(
    TransformSRT const& worldFromPrefab,
    ActorPrefabMetricScaleMap const&,
    RigidPrismaticJointConstraintPrefab& params) {
  params.freeAxis = worldFromPrefab.TransformDirection(params.freeAxis);
  real const scale = worldFromPrefab.GetScale();
  if (params.max.has_value()) {
    *params.max *= scale;
  }
  if (params.min.has_value()) {
    *params.min *= scale;
  }
}

static void ApplyPrefabTransformToConstraint(
    TransformSRT const& worldFromPrefab,
    ActorPrefabMetricScaleMap const&,
    JointRotationRangeConstraintPrefab& params) {
  params.refFrameRotVec = TransformRotationVector(worldFromPrefab, params.refFrameRotVec);
}

static void ApplyPrefabTransformToConstraint(
    TransformSRT const& worldFromPrefab,
    ActorPrefabMetricScaleMap const& actorPrefabMetricScale,
    RigidPivotPositionConstraintPrefab& params) {
  params.targetPosition = worldFromPrefab.TransformPoint(params.targetPosition);
  params.localPosition *= GetActorPrefabMetricScale(actorPrefabMetricScale, params.actor);
}

static void ApplyPrefabTransformToConstraint(
    TransformSRT const& worldFromPrefab,
    ActorPrefabMetricScaleMap const& actorPrefabMetricScale,
    RigidPivotToRigidTargetConstraintPrefab& params) {
  params.targetTransform =
      (worldFromPrefab * TransformSRT(params.targetTransform)).GetTransformRT();
  params.localPosition *= GetActorPrefabMetricScale(actorPrefabMetricScale, params.actor);
}

static void ApplyPrefabTransformToConstraint(
    TransformSRT const&,
    ActorPrefabMetricScaleMap const& actorPrefabMetricScale,
    RigidSphericalJointConstraintPrefab& params) {
  params.localPosA *= GetActorPrefabMetricScale(actorPrefabMetricScale, params.actorA);
  params.localPosB *= GetActorPrefabMetricScale(actorPrefabMetricScale, params.actorB);
}

static void ApplyPrefabTransformToConstraint(
    TransformSRT const& worldFromPrefab,
    ActorPrefabMetricScaleMap const&,
    RigidPivotRotationConstraintPrefab& params) {
  params.targetRotation = TransformRotationVector(worldFromPrefab, params.targetRotation);
}

static void ApplyPrefabTransformToConstraint(
    TransformSRT const& worldFromPrefab,
    ActorPrefabMetricScaleMap const&,
    DeformableNodePositionConstraintPrefab& params) {
  params.position = worldFromPrefab.TransformPoint(params.position);
}

static void ApplyPrefabTransformToConstraint(
    TransformSRT const&,
    ActorPrefabMetricScaleMap const& actorPrefabMetricScale,
    DeformableNodeToRigidConstraintPrefab& params) {
  // rigidLocalPos follows the referenced rigid actor's prefab metric scale: it is the attachment
  // point when fixToDeformablePos is false, and the search point when findClosest is true. It is
  // inert only when fixToDeformablePos is true and findClosest is false.
  params.rigidLocalPos *= GetActorPrefabMetricScale(actorPrefabMetricScale, params.rigidActor);
}

static void ApplyPrefabTransformToConstraint(
    TransformSRT const& worldFromPrefab,
    ActorPrefabMetricScaleMap const&,
    JointRotationTrackingConstraintPrefab& params) {
  params.refFrameRotVec = TransformRotationVector(worldFromPrefab, params.refFrameRotVec);
}

// Remaining constraint types: no field that the prefab transform changes here. Rotation targets and
// ranges are scale-invariant, and DeformableNodeToDeformableNode references only node indices.
// ArticulatedSingleDof target/range values are also left unchanged here: a translational (prismatic
// or free-translation) DoF's value is a distance that must scale, but that needs the referenced
// actor's effective scale and per-DoF type, which are only available in AddConstraintsImpl (see
// ScaleArticulatedSingleDofValue), so it is applied there rather than in this params-only pass.
template <class ParamsT>
static constexpr bool kPrefabTransformNoOpConstraint =
    std::is_same_v<ParamsT, DeformableNodeToDeformableNodeConstraintPrefab> ||
    std::is_same_v<ParamsT, ArticulatedSingleDofTargetConstraintPrefab> ||
    std::is_same_v<ParamsT, Articulated3dRotationTargetConstraintPrefab> ||
    std::is_same_v<ParamsT, ArticulatedSingleDofRangeConstraintPrefab> ||
    std::is_same_v<ParamsT, Articulated3dRotationRangeConstraintPrefab>;

template <class ParamsT>
static void
ApplyPrefabTransformToConstraint(TransformSRT const&, ActorPrefabMetricScaleMap const&, ParamsT&) {
  static_assert(
      kPrefabTransformNoOpConstraint<ParamsT>,
      "Add an ApplyPrefabTransformToConstraint overload for this constraint type, or add it to "
      "this allowlist if this params-only transform pass has nothing to do.");
}

// Scale factor for an articulated single-DoF constraint value (targetValue / min / max). Returns
// the referenced actor's prefab metric scale for a translational DoF (the value is a distance
// [m], like the joint translations and prismatic limits AddToSceneImpl scales), or 1 for a
// rotational DoF (the value is an angle [rad]). Non-articulated actors and out-of-range jointIndex
// values are left for creation-time validation to report.
[[nodiscard]] static real ScaleArticulatedSingleDofValue(
    Scene* scene,
    ActorPrefabMetricScaleMap const& metricScaleByActor,
    ActorHandle actor,
    int jointIndex,
    int dofIndex) {
  auto const& reg = assert_cast<SceneImpl const*>(scene)->GetRegistry();
  entt::entity const entity = GetEntity(reg, actor, ErrorAssert{});
  auto const* articulatedShape = reg.try_get<CArticulatedBodyShape const>(entity);
  if (articulatedShape == nullptr) {
    return 1_r;
  }

  auto const* joints = articulatedShape->shape->GetJointsData();
  if (jointIndex < 0 || jointIndex >= isize(joints->dofInfo)) {
    return 1_r;
  }
  bool const isTranslational = dofIndex >= 0 && dofIndex < joints->dofInfo[jointIndex].transSize;
  if (!isTranslational) {
    return 1_r;
  }
  return GetActorPrefabMetricScale(metricScaleByActor, actor);
}

template <class ParamsT>
static void AddConstraintsImpl(
    Scene* scene,
    DynamicString const& path,
    ActorNameRegistry const& actorNames,
    TransformSRT const& worldFromPrefab,
    ActorPrefabMetricScaleMap const& actorPrefabMetricScale,
    DynamicArray<ParamsT> const& paramsList,
    DynamicArray<Constraint*>& outConstraints,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  for (auto const& paramsIn : paramsList) {
    auto params = paramsIn; // mutable copy

    // First look up the ActorHandle value for each actor.
    // The field names and number of actors depend on the type of constraint
    if constexpr (
        std::is_same_v<ParamsT, RigidSphericalJointConstraintPrefab> ||
        std::is_same_v<ParamsT, RigidPrismaticJointConstraintPrefab> ||
        std::is_same_v<ParamsT, DeformableNodeToDeformableNodeConstraintPrefab> ||
        std::is_same_v<ParamsT, JointRotationRangeConstraintPrefab> ||
        std::is_same_v<ParamsT, JointRotationTrackingConstraintPrefab>) {
      params.actorA = FindActorHandleForConstraint(path, params.actorNameA, actorNames, error);
      params.actorB = FindActorHandleForConstraint(path, params.actorNameB, actorNames, error);
    } else if constexpr (std::is_same_v<ParamsT, DeformableNodeToRigidConstraintPrefab>) {
      params.rigidActor =
          FindActorHandleForConstraint(path, params.rigidActorName, actorNames, error);
      params.deformableActor =
          FindActorHandleForConstraint(path, params.deformableActorName, actorNames, error);
    } else if constexpr (
        std::is_same_v<ParamsT, RigidPivotPositionConstraintPrefab> ||
        std::is_same_v<ParamsT, RigidPivotToRigidTargetConstraintPrefab> ||
        std::is_same_v<ParamsT, RigidPivotRotationConstraintPrefab> ||
        std::is_same_v<ParamsT, DeformableNodePositionConstraintPrefab> ||
        std::is_same_v<ParamsT, ArticulatedSingleDofTargetConstraintPrefab> ||
        std::is_same_v<ParamsT, Articulated3dRotationTargetConstraintPrefab> ||
        std::is_same_v<ParamsT, ArticulatedSingleDofRangeConstraintPrefab> ||
        std::is_same_v<ParamsT, Articulated3dRotationRangeConstraintPrefab>) {
      params.actor = FindActorHandleForConstraint(path, params.actorName, actorNames, error);
    } else {
      static_assert(
          std::is_same_v<ParamsT, RigidSphericalJointConstraintPrefab>,
          "New constraint type: add its actor-name resolution to this if constexpr chain.");
    }
    MOCHI_ERROR_RETURN(error);

    ApplyPrefabTransformToConstraint(worldFromPrefab, actorPrefabMetricScale, params);

    // Translational articulated single-DoF target/range values are distances that scale with the
    // referenced actor's effective prefab scale (rotational DoFs are angles and stay unchanged).
    if constexpr (std::is_same_v<ParamsT, ArticulatedSingleDofTargetConstraintPrefab>) {
      params.targetValue *= ScaleArticulatedSingleDofValue(
          scene, actorPrefabMetricScale, params.actor, params.jointIndex, params.dofIndex);
    } else if constexpr (std::is_same_v<ParamsT, ArticulatedSingleDofRangeConstraintPrefab>) {
      real const dofScale = ScaleArticulatedSingleDofValue(
          scene, actorPrefabMetricScale, params.actor, params.jointIndex, params.dofIndex);
      params.minValue *= dofScale;
      params.maxValue *= dofScale;
    }

    // Add the constraint to the scene
    Constraint* newConstraint = AddConstraintImpl(scene, params, error);
    MOCHI_ERROR_RETURN(error);
    outConstraints.push_back(newConstraint);
  }
}

[[nodiscard]] static bool IsPositiveScale(real scale) {
  return IsFinite(scale) && scale > 0_r && !NearZero(scale);
}

[[nodiscard]] static bool IsPositiveScale(Real3 const& scale) {
  return IsPositiveScale(scale[0]) && IsPositiveScale(scale[1]) && IsPositiveScale(scale[2]);
}

[[nodiscard]] static bool IsNonZeroScale(Real3 const& scale) {
  return IsFinite(scale) && !NearZero(scale[0]) && !NearZero(scale[1]) && !NearZero(scale[2]);
}

// Validate that a scale is strictly positive and finite.
#define MOCHI_VALIDATE_POSITIVE_SCALE(scale, scaleName, error) \
  MOCHI_ERROR_IF_NOT(                                          \
      ::IsPositiveScale(scale),                                \
      error,                                                   \
      scaleName " must be strictly positive and finite. Negative or zero scale is not supported.")

// Validate that a pose has a finite, non-zero quaternion and a finite translation.
#define MOCHI_VALIDATE_POSE(rotation, translation, poseName, error)                       \
  MOCHI_ERROR_IF_NOT(                                                                     \
      ::IsFinite(rotation) && !NearEqual(Norm(rotation), 0_r) && ::IsFinite(translation), \
      error,                                                                              \
      poseName                                                                            \
      " rotation quaternion must be finite and non-zero, and translation must be finite.")

// Bake a diagonal scale into optional rigid-body inertia overrides. The inertia tensor I is
// converted to the second moment matrix C, scaled as C' = |det(S)| S C S, and converted back via
// I' = tr(C') Identity - C'.
static void ApplyDiagonalScaleToInertia(
    std::optional<Real3>& centerOfMass,
    std::optional<real>& mass,
    std::optional<Real6>& momentOfInertia,
    Real3 const& scale) {
  if (scale == kReal3Ones) {
    // Preserve bitwise identity for exported prefabs reloaded with default (identity) scale. The
    // inertia transform is algebraically identity, but the I -> C -> I roundtrip can change
    // serialized floating-point values.
    return;
  }
  if (centerOfMass.has_value()) {
    *centerOfMass *= scale;
  }
  real const volumeScale = Abs(scale[0] * scale[1] * scale[2]);
  if (mass.has_value()) {
    *mass *= volumeScale;
  }
  if (momentOfInertia.has_value()) {
    Real6 const inertia = *momentOfInertia;
    real const cxx = 0.5_r * (-inertia[0] + inertia[3] + inertia[5]);
    real const cyy = 0.5_r * (inertia[0] - inertia[3] + inertia[5]);
    real const czz = 0.5_r * (inertia[0] + inertia[3] - inertia[5]);
    real const cxy = -inertia[1];
    real const cxz = -inertia[2];
    real const cyz = -inertia[4];

    real const cxxScaled = volumeScale * scale[0] * scale[0] * cxx;
    real const cyyScaled = volumeScale * scale[1] * scale[1] * cyy;
    real const czzScaled = volumeScale * scale[2] * scale[2] * czz;
    real const cxyScaled = volumeScale * scale[0] * scale[1] * cxy;
    real const cxzScaled = volumeScale * scale[0] * scale[2] * cxz;
    real const cyzScaled = volumeScale * scale[1] * scale[2] * cyz;

    *momentOfInertia = Real6{
        cyyScaled + czzScaled,
        -cxyScaled,
        -cxzScaled,
        cxxScaled + czzScaled,
        -cyzScaled,
        cxxScaled + cyyScaled};
  }
}

static void AddToSceneImpl(
    ScenePrefab const& prefab,
    Scene* scene,
    PrefabParams const& params,
    ActorNameRegistry& outActorNames,
    ActorPrefabMetricScaleMap& actorPrefabMetricScale,
    AddToSceneResult& outResult,
    DynamicArray<ScenePrefab const*>& activePrefabs,
    Error& error) {
  ActivePrefabGuard activePrefabGuard(activePrefabs, &prefab, error);
  MOCHI_ERROR_RETURN(error);

  // Helper
  auto onActorCreated = [&](Actor* newActor, DynamicString const& actorName) {
    outResult.actors.push_back(newActor);
    RegisterActorName(outActorNames, actorName, newActor->GetHandle());
  };

  MOCHI_VALIDATE_POSITIVE_SCALE(params.scale, "PrefabParams::scale", error);
  MOCHI_VALIDATE_POSE(params.rotation, params.translation, "PrefabParams", error);
  MOCHI_ERROR_RETURN(error);

  if (params.applySceneSettings && prefab.scene.has_value()) {
    if (prefab.scene->gravity.has_value()) {
      scene->SetGravity(*prefab.scene->gravity);
    }
    if (prefab.scene->solver.has_value()) {
      scene->SetSolverParams(*prefab.scene->solver, error);
    }
  }
  MOCHI_ERROR_RETURN(error);

  TransformSRT worldFromPrefab(params.scale, Normalize(params.rotation), params.translation);

  // Nested Prefabs: each nested prefab is instantiated into its own subtree name registry, which is
  // then merged into this level's registry. A nested prefab's own references resolve against its
  // subtree registry during its recursion (prefab references only ever point downward -- an own
  // actor or a "nested/child" path, never upward), so a name reused across sibling instances stays
  // unambiguous for each instance's internal references and becomes ambiguous only for references
  // at this level or above (via the merge). Scale maps and outResult stay accumulated across the
  // whole recursion; only the name registry is subtree-local.
  for (auto const& nested : prefab.prefabs) {
    if (nested.prefab) {
      MOCHI_VALIDATE_POSITIVE_SCALE(nested.scale, "PrefabReference::scale", error);
      MOCHI_VALIDATE_POSE(nested.rotation, nested.translation, "PrefabReference", error);
      MOCHI_ERROR_RETURN(error);
      TransformSRT prefabFromNested(nested.scale, Normalize(nested.rotation), nested.translation);
      TransformSRT worldFromNested = worldFromPrefab * prefabFromNested;

      auto nestedParams = params;
      nestedParams.name = CombineNames(params.name, nested.name);
      nestedParams.applySceneSettings = false; // Scene settings only come from the top-level prefab
      nestedParams.scale = worldFromNested.GetScale();
      nestedParams.rotation = worldFromNested.GetRotation();
      nestedParams.translation = worldFromNested.GetTranslation();
      ActorNameRegistry nestedActorNames;
      AddToSceneImpl(
          *nested.prefab,
          scene,
          nestedParams,
          nestedActorNames,
          actorPrefabMetricScale,
          outResult,
          activePrefabs,
          error);
      MOCHI_ERROR_RETURN(error);
      MergeActorNameRegistry(outActorNames, nestedActorNames);
    }
  }

  // Articulated Actors:
  auto getArticulatedActorParams = [&](ArticulatedActorPrefab const& actor) {
    ArticulatedActorParams actorParams;

    MOCHI_VALIDATE_POSITIVE_SCALE(actor.scale, "ArticulatedActorPrefab::scale", error);
    MOCHI_VALIDATE_POSE(actor.rotation, actor.translation, "ArticulatedActorPrefab", error);
    for (auto const& link : actor.links) {
      if (!link.shapeFile.empty()) {
        MOCHI_ERROR_IF_NOT(
            IsNonZeroScale(link.shapeScale),
            error,
            "ArticulatedLinkPrefab::shapeScale must be non-zero and finite on all 3 axes.");
      }
    }
    MOCHI_ERROR_RETURN(error, actorParams);

    TransformSRT prefabFromActor(1_r, Normalize(actor.rotation), actor.translation);
    TransformSRT worldFromActor = worldFromPrefab * prefabFromActor;

    actorParams.name = CombineNames(params.name, actor.name);
    actorParams.worldFromRoot = worldFromActor.GetTransformRT();
    actorParams.cycles = actor.cycles;
    actorParams.joints.assign(actor.joints.begin(), actor.joints.end());
    actorParams.links.assign(actor.links.begin(), actor.links.end());

    // The articulated effective scale composes the per-actor uniform scale
    // with the cumulative outer prefab scale.
    real const effectiveScale = actor.scale * params.scale;
    auto scaleTranslation = [effectiveScale](TransformRT const& t) {
      return TransformRT(t.GetRotation(), t.GetTranslation() * effectiveScale);
    };
    for (auto& joint : actorParams.joints) {
      joint.parentLinkFromJoint = scaleTranslation(joint.parentLinkFromJoint);
      // Prismatic joint limits are translational (meters) and must be scaled.
      if (joint.type == ArticulatedJointType::Prismatic) {
        if (joint.minLimit.has_value()) {
          *joint.minLimit *= effectiveScale;
        }
        if (joint.maxLimit.has_value()) {
          *joint.maxLimit *= effectiveScale;
        }
      }
    }
    for (int i = 0; i < isize(actorParams.links); ++i) {
      auto& link = actorParams.links[i];
      link.parentJointFromLink = scaleTranslation(link.parentJointFromLink);

      Real3 linkEffectiveScale{effectiveScale, effectiveScale, effectiveScale};
      if (!actor.links[i].shapeFile.empty()) {
        linkEffectiveScale *= actor.links[i].shapeScale;
      }
      ApplyDiagonalScaleToInertia(
          link.centerOfMass, link.mass, link.momentOfInertia, linkEffectiveScale);
    }
    for (auto& cycle : actorParams.cycles) {
      cycle.jointFromChildLink = scaleTranslation(cycle.jointFromChildLink);
    }

    if (actor.skin.has_value()) {
      actorParams.skin.emplace(*actor.skin);
    }
    actorParams.jointVelocities = actor.jointVelocities;
    return actorParams;
  };
  auto recordActorPrefabMetricScale = [&](ActorHandle handle, real scale) {
    actorPrefabMetricScale[handle] = scale;
  };
  auto addNestedLinksActorsToMap =
      [&](Actor const* newActor, DynamicString const& name, real localDistanceScale) {
        auto linkActors = newActor->GetNestedLinkActors(ErrorAssert{});
        for (auto linkActor : linkActors) {
          recordActorPrefabMetricScale(linkActor, localDistanceScale);
        }

        // Keep track of created link actors so constraints can reference them by name.
        if (!name.empty()) {
          auto artInfo = newActor->GetArticulatedShapeInfo(ErrorAssert{});
          if (isize(artInfo.linkNames) == isize(linkActors)) {
            for (int i = 0; i < isize(artInfo.linkNames); ++i) {
              auto const& linkName = artInfo.linkNames[i];
              if (!linkName.empty()) {
                RegisterActorName(outActorNames, CombineNames(name, linkName), linkActors[i]);
              }
            }
          }
        }
      };
  for (auto const& actor : prefab.actors.articulated) {
    auto actorParams = getArticulatedActorParams(actor);
    MOCHI_ERROR_RETURN(error);
    Actor* newActor = scene->CreateArticulatedActor(actorParams, error);
    MOCHI_ERROR_RETURN(error);
    onActorCreated(newActor, actorParams.name);
    real const effectiveScale = actor.scale * params.scale;
    recordActorPrefabMetricScale(newActor->GetHandle(), effectiveScale);
    addNestedLinksActorsToMap(newActor, actorParams.name, effectiveScale);
  }

  // Rigid Actors:
  for (auto const& actor : prefab.actors.rigid) {
    MOCHI_ERROR_IF_NOT(
        IsNonZeroScale(actor.scale),
        error,
        "RigidActorPrefab::scale must be non-zero and finite on all 3 axes.");
    MOCHI_VALIDATE_POSE(actor.rotation, actor.translation, "RigidActorPrefab", error);
    MOCHI_ERROR_RETURN(error);

    TransformSRT prefabFromActor(1_r, Normalize(actor.rotation), actor.translation);
    TransformSRT worldFromActor = worldFromPrefab * prefabFromActor;

    RigidActorParams actorParams = actor;
    actorParams.name = CombineNames(params.name, actor.name);
    actorParams.worldFromLocal = worldFromActor.GetTransformRT();
    bool const hasInertiaOverride = actor.mass.has_value() || actor.centerOfMass.has_value() ||
        actor.momentOfInertia.has_value();

    if (hasInertiaOverride) {
      Real3 const effectiveScale = actor.scale * params.scale;
      ApplyDiagonalScaleToInertia(
          actorParams.centerOfMass, actorParams.mass, actorParams.momentOfInertia, effectiveScale);
    }

    Actor* newActor = scene->CreateRigidActor(actorParams, error);
    MOCHI_ERROR_RETURN(error);
    onActorCreated(newActor, actorParams.name);
    recordActorPrefabMetricScale(newActor->GetHandle(), params.scale);
  }

  // Soft Actors:
  for (auto const& actor : prefab.actors.soft) {
    MOCHI_VALIDATE_POSITIVE_SCALE(actor.scale, "SoftActorPrefab::scale", error);
    MOCHI_VALIDATE_POSE(actor.rotation, actor.translation, "SoftActorPrefab", error);
    MOCHI_ERROR_IF(
        !actor.flowFile.empty() && !NearEqual(actor.scale, Real3{1_r, 1_r, 1_r}),
        error,
        "SoftActorPrefab::scale must be (1, 1, 1) when SoftActorPrefab::flowFile is set.");
    MOCHI_ERROR_RETURN(error);

    TransformSRT prefabFromActor(1_r, Normalize(actor.rotation), actor.translation);
    TransformSRT worldFromActor = worldFromPrefab * prefabFromActor;

    SoftActorParams actorParams = actor;
    actorParams.name = CombineNames(params.name, actor.name);
    actorParams.worldFromLocal = worldFromActor.GetTransformRT();

    ExperimentalSoftActorParams experimentalParams;
    experimentalParams.colliderType = actor.colliderType;
    experimentalParams.sdf = actor.sdf;
    experimentalParams.flow = actor.flow;
    experimentalParams.useRecentering = actor.useRecentering;

    Actor* newActor = CreateSoftActor(scene, actorParams, experimentalParams, error);
    MOCHI_ERROR_RETURN(error);
    onActorCreated(newActor, actorParams.name);
    recordActorPrefabMetricScale(newActor->GetHandle(), params.scale);
  }

  // Soft Skinned Actors:
  for (auto const& actor : prefab.actors.softSkinned) {
    for (auto const& soft : actor.softParams) {
      MOCHI_VALIDATE_POSITIVE_SCALE(
          soft.scale, "SoftSkinnedActorPrefab::softParams[].scale", error);
      MOCHI_ERROR_IF(
          soft.scale[0] != soft.scale[1] || soft.scale[1] != soft.scale[2],
          error,
          "SoftSkinnedActorPrefab::softParams[].scale must be uniform.");
      MOCHI_ERROR_IF(
          !soft.flowFile.empty() && !NearEqual(soft.scale, Real3{1_r, 1_r, 1_r}),
          error,
          "SoftSkinnedActorPrefab::softParams[].scale must be (1, 1, 1) when SoftSkinnedActorPrefab::softParams[].flowFile is set.");
    }
    MOCHI_ERROR_RETURN(error);

    int numSoft = isize(actor.softParams);

    SoftSkinnedActorParams actorParams;
    actorParams.skeletonParams = getArticulatedActorParams(actor.skeletonParams);
    MOCHI_ERROR_RETURN(error);
    actorParams.softParams.resize(numSoft);
    ExperimentalSoftSkinnedActorParams experimentalParams;
    experimentalParams.softParams.resize(numSoft);
    for (int i = 0; i < numSoft; ++i) {
      // Copy SoftActorPrefab --> SoftActorParams
      actorParams.softParams[i] = actor.softParams[i];

      // Copy experimental fields from prefab
      experimentalParams.softParams[i].colliderType = actor.softParams[i].colliderType;
      experimentalParams.softParams[i].sdf = actor.softParams[i].sdf;
      experimentalParams.softParams[i].flow = actor.softParams[i].flow;
      experimentalParams.softParams[i].useRecentering = actor.softParams[i].useRecentering;
    }
    actorParams.softAttachLinks = actor.softAttachLinks;
    actorParams.enableCollidingLinks = actor.enableCollidingLinks;
    actorParams.hasGravity = actor.hasGravity;
    actorParams.hasInertia = actor.hasInertia;
    actorParams.hasStress = actor.hasStress;

    Actor* newActor = CreateSoftSkinnedActor(scene, actorParams, experimentalParams, error);
    MOCHI_ERROR_RETURN(error);
    auto const& actorName = actorParams.skeletonParams.name;
    onActorCreated(newActor, actorName);
    real const skeletonEffectiveScale = actor.skeletonParams.scale * params.scale;
    recordActorPrefabMetricScale(newActor->GetHandle(), skeletonEffectiveScale);
    addNestedLinksActorsToMap(newActor, actorName, skeletonEffectiveScale);
    MOCHI_ERROR_RETURN(error);

    auto softActors = newActor->GetNestedSoftActors(error);
    MOCHI_ERROR_RETURN(error);
    MOCHI_ASSERT(
        numSoft == isize(softActors),
        "Expected one nested soft actor per SoftSkinnedActorParams::softParams entry.");
    for (int i = 0; i < numSoft; ++i) {
      recordActorPrefabMetricScale(softActors[i], params.scale);

      // Also keep track of named nested soft actors so constraints can reference them.
      if (!actorName.empty()) {
        Actor const* softActor = scene->GetActor(softActors[i]);
        MOCHI_ASSERT(softActor != nullptr, "GetNestedSoftActors should provide valid handles.");
        DynamicString const softName = GetNestedActorLocalName(softActor->GetName());
        MOCHI_ASSERT(!softName.empty(), "Nested soft local name must be non-empty.");
        RegisterActorName(outActorNames, CombineNames(actorName, softName), softActors[i]);
      }
    }
  }

  // Constraints:
  // If you add a ConstraintType, decide whether prefabs must (de)serialize it. If so, add a prefab
  // constraint list, an AddConstraintsImpl(...) call below, an ApplyPrefabTransformToConstraint
  // overload, and (if it references actors) an actor-name resolution branch above -- then bump this
  // count. This guards against a new constraint type being silently dropped by a forgotten dispatch
  // call; the per-type ApplyPrefabTransformToConstraint / if-constexpr static_asserts only fire
  // once a type is actually dispatched, so they cannot catch an omitted call on their own.
  // TODO(T280114669): RodElementRotationToRigid is the only real ConstraintType without prefab
  // support. Add a RodElementRotationToRigidConstraintPrefab, a ConstraintLists entry,
  // actor-name resolution, transform handling for refFrameRotVec, and dispatch here if
  // rod-to-rigid rotation constraints need to be authorable from prefab JSON.
  static_assert(
      static_cast<int>(ConstraintType::Count) == 16,
      "ConstraintType changed: review the prefab constraint dispatch below.");
  // clang-format off
  auto const& con = prefab.constraints;
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.rigidSphericalJoint, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.rigidPrismaticJoint, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.deformableNodeToDeformableNode, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.deformableNodeToRigid, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.jointRotationRange, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.rigidPivotPosition, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.rigidPivotToRigidTarget, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.rigidPivotRotation, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.deformableNodePosition, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.jointRotationTracking, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.articulatedSingleDofTarget, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.articulated3dRotationTarget, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.articulatedSingleDofRange, outResult.constraints, error);
  AddConstraintsImpl(scene, params.name, outActorNames, worldFromPrefab, actorPrefabMetricScale, con.articulated3dRotationRange, outResult.constraints, error);
  MOCHI_ERROR_RETURN(error);
  // clang-format on

  // Pose Controllers:
  for (auto const& controller : prefab.controllers) {
    // The controller must reference an articulated actor within this prefab or a nested prefab.
    // The name of that actor will be interpreted as a hierarchy path, starting from this prefab.
    MOCHI_ERROR_IF(
        controller.articulatedActor.empty(),
        error,
        "Pose controller must specify the name of an articulated actor to control.");
    MOCHI_ERROR_RETURN(error);
    auto actorNameToFind = CombineNames(params.name, controller.articulatedActor);
    auto const it = FindActorHandle(outActorNames, actorNameToFind, error);
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        it == outActorNames.handles.end(),
        error,
        "Failed to find articulated actor referenced by pose controller.");
    MOCHI_ERROR_RETURN(error);

    Actor* articulatedActor = scene->GetActor(it->second);
    MOCHI_ASSERT(
        articulatedActor != nullptr, "We just created this actor. It should still be valid.");
    MOCHI_ERROR_IF(
        articulatedActor->GetType() != ActorType::Articulated,
        error,
        "Failed to attach pose controller. The referenced actor must be articulated.");
    MOCHI_ERROR_RETURN(error);

    PoseControllerParams controllerParams;
    controllerParams.linkPosTracking = controller.linkPosTracking;
    controllerParams.linkRotTracking = controller.linkRotTracking;
    controllerParams.jointTracking = controller.jointTracking;
    articulatedActor->AddArticulatedPoseController(controllerParams, error);
    MOCHI_ERROR_RETURN(error);
  }

  // Apply actor-vs-actor and layer-vs-layer contact settings (must be after all actors are created)
  if (prefab.contactFilter.has_value()) {
    ApplyContactFilter(*prefab.contactFilter, params.name, outActorNames, scene, error);
  }
  MOCHI_ERROR_RETURN(error);
}

template <class T, class TypeEnum>
static DynamicArray<T*> FilterImpl(DynamicArray<T*> const& list, TypeEnum type) {
  DynamicArray<T*> result;
  result.reserve(isize(list));
  for (int i = 0; i < isize(list); ++i) {
    if (list[i]->GetType() == type) {
      result.push_back(list[i]);
    }
  }
  return result;
}

MOCHI_API DynamicArray<Actor*> AddToSceneResult::Filter(ActorType type) const {
  return FilterImpl<Actor>(actors, type);
}

MOCHI_API DynamicArray<Constraint*> AddToSceneResult::Filter(ConstraintType type) const {
  return FilterImpl<Constraint>(constraints, type);
}

// Owns the result/actor-map/AddToSceneImpl tail shared by both AddToScene overloads, which differ
// only in how they obtain and validate the prefab before instantiating it.
// TODO: Decide whether prefab::AddToScene should be transactional on error. Today failures after
// instantiation has begun, including constraint, pose-controller, and contact-filter errors, can
// leave actors, constraints, scene settings, or contact settings already applied to the Scene.
static AddToSceneResult AddToSceneFromLoaded(
    ScenePrefab const& prefab,
    Scene* scene,
    PrefabParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  AddToSceneResult result;
  ActorNameRegistry actorNames;
  ActorPrefabMetricScaleMap actorPrefabMetricScale;
  DynamicArray<ScenePrefab const*> activePrefabs;
  AddToSceneImpl(
      prefab, scene, params, actorNames, actorPrefabMetricScale, result, activePrefabs, error);
  return result;
}

MOCHI_API AddToSceneResult prefab::AddToScene(
    ScenePrefab const& prefab,
    Scene* scene,
    PrefabParams const& params,
    Error& error) {
  // This overload receives an already-loaded prefab whose geometry was baked (at LoadShapes time)
  // before params.scale is known. A non-identity instance scale here would scale poses, constraint
  // targets, and inertia but NOT mesh geometry.
  MOCHI_ERROR_IF_NOT(
      NearEqual(params.scale, 1_r),
      error,
      "This AddToScene overload is only supported for PrefabParams::scale = 1. To instantiate with a "
      "different scale, use the prefabPath-based AddToScene overload or a nested PrefabReference::scale.");
  MOCHI_ERROR_RETURN(error, {});

  // Treat tolerated round-trip noise as identity so scale cannot affect poses/inertia after
  // geometry has already been baked.
  auto identityScaleParams = params;
  identityScaleParams.scale = 1_r;
  return AddToSceneFromLoaded(prefab, scene, identityScaleParams, error);
}

MOCHI_API AddToSceneResult prefab::AddToScene(
    std::string_view prefabPath,
    std::string_view rootPath,
    Scene* scene,
    PrefabParams const& params,
    Error& error) {
  MOCHI_VALIDATE_POSITIVE_SCALE(params.scale, "PrefabParams::scale", error);
  MOCHI_ERROR_RETURN(error, {});
  auto* context = scene->GetContext();
  auto prefab = LoadFromFileImpl(prefabPath, rootPath, context, params.scale, error);
  MOCHI_ERROR_RETURN(error, {});
  return AddToSceneFromLoaded(prefab, scene, params, error);
}
