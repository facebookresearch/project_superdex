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

#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/utils/mochi_prefab.h>
#include "mochi_actor.h"
#include "mochi_articulated_actor_params.h"
#include "mochi_articulated_body.h"
#include "mochi_constraint.h"
#include "mochi_contact_filter.h"
#include "mochi_ecs_utils.h"
#include "mochi_scene.h"
#include "mochi_simulation.h"
#include "mochi_soft.h"
#include "mochi_soft_skinned.h"

#include <mochi_core/utils/file_utils.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace mochi;
using namespace mochi::prefab;

namespace {
struct ExportActorNameEntry {
  entt::entity entity;
  DynamicString name;
};
} // namespace

// Helper function to make a name unique by appending a number if needed
static void MakeNameUnique(DynamicString& name, std::unordered_map<std::string, int>& usedNames) {
  auto [it, inserted] = usedNames.try_emplace(std::string(name), 0);
  if (inserted) {
    // Name is unique, no modification needed
    return;
  }

  // Name already exists, find a unique variant by incrementing counter
  std::string baseName(name);
  int counter = it->second + 1;
  do {
    name = Format("%s%d", baseName.c_str(), counter);
    auto [candidateIt, candidateInserted] = usedNames.try_emplace(std::string(name), 0);
    if (candidateInserted) {
      usedNames[baseName] = counter;
      return;
    }
    counter++;
  } while (true);
}

// Helper function to generate shape file paths for export
static std::pair<DynamicString, std::filesystem::path> GenerateShapeFilePaths(
    std::string_view actorName,
    std::string_view suffix,
    std::string_view relativePrefix,
    std::string_view assetsDir,
    std::unordered_map<std::string, int>& usedFileNames) {
  auto toLowercase = [](std::string_view str) {
    DynamicString result{str};
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
      return std::tolower(c);
    });
    return result;
  };

  DynamicString safeName = toLowercase(actorName);
  std::replace(safeName.begin(), safeName.end(), '/', '_');
  std::replace(safeName.begin(), safeName.end(), '\\', '_');
  std::replace(safeName.begin(), safeName.end(), ' ', '_');

  // Generate base filename
  DynamicString stem = suffix.empty()
      ? safeName
      : DynamicString(Format("%s_%s", safeName.c_str(), DynamicString(suffix).c_str()));

  // Handle filename collisions — uniquify the stem before attaching the extension
  MakeNameUnique(stem, usedFileNames);
  DynamicString fileName = DynamicString(Format("%s.mochi.h5", stem.c_str()));

  DynamicString relativePath = DynamicString(relativePrefix) + fileName;
  std::filesystem::path fullPath = std::filesystem::path(assetsDir) / fileName;

  std::filesystem::create_directories(fullPath.parent_path());

  return {relativePath, fullPath};
}

[[nodiscard]] static DynamicString MakeNestedActorName(
    DynamicString const& parentName,
    DynamicString const& localName) {
  DynamicString nestedName = parentName;
  nestedName += "/";
  nestedName += localName;
  return nestedName;
}

// Actor types prefab export can pre-name and serialize. Must stay in sync with the types
// ExportActorToPrefab actually dispatches: a type marked supported here but not handled there (or
// vice versa) would be pre-named yet skipped on export, reintroducing dangling contact-filter
// names.
[[nodiscard]] static constexpr bool IsActorTypeSupportedForPrefabExport(ActorType type) {
  static_assert(
      static_cast<int>(ActorType::Count) == 6,
      "If you add a new ActorType, please update the switch statement below");

  switch (type) {
    case ActorType::Rigid:
    case ActorType::Soft:
    case ActorType::Articulated:
      return true;
    case ActorType::None:
    case ActorType::Shell:
    case ActorType::Rod:
    case ActorType::Count:
      return false;
  }

  MOCHI_ASSERT(false, "Unexpected actor type.");
  return false;
}

// Helper function to get a meaningful base name for an actor
[[nodiscard]] static DynamicString GetActorBaseName(Actor const* actor) {
  DynamicString name = actor->GetName();

  if (!name.empty()) {
    return name;
  }

  auto handle = actor->GetHandle();
  DynamicString typeString;
  switch (actor->GetType()) {
    case ActorType::Rigid:
      typeString = "rigid";
      break;
    case ActorType::Soft:
      typeString = "soft";
      break;
    case ActorType::Articulated:
      typeString = "articulated";
      break;
    case ActorType::Shell:
      typeString = "shell";
      break;
    case ActorType::Rod:
      typeString = "rod";
      break;
    default:
      typeString = "unknown";
      break;
  }
  static_assert(
      static_cast<int>(ActorType::Count) == 6,
      "If you add a new ActorType, consider adding it to the switch statement above");

  DynamicString generatedName(Format("%s_%u", typeString.c_str(), handle.value));
  return generatedName;
}

[[nodiscard]] static DynamicString MakeNameCandidate(DynamicString const& baseName, int suffix) {
  return suffix == 0 ? baseName : DynamicString(Format("%s%d", baseName.c_str(), suffix));
}

// Converts an actor contact mesh to .h5 file
static DynamicString SaveActorMeshToFile(
    Actor const* actor,
    std::string_view filePath,
    std::string_view relativeFilePath,
    std::unordered_map<Shape const*, DynamicString>& exportedShapeCache,
    Error& error) {
  MOCHI_ERROR_RETURN(error, "");

  // Get ECS registry and entity for this actor
  auto const* scene = actor->GetScene();
  auto const* sceneImpl = assert_cast<SceneImpl const*>(scene);
  auto const& registry = sceneImpl->GetRegistry();
  auto entity = GetEntity(registry, actor->GetHandle(), ErrorAssert{});

  // Get the actor's Shape from the registry
  auto const* shapeComponent = registry.try_get<CShape const>(entity);
  MOCHI_ERROR_IF(!shapeComponent, error, "Could not determine the model shape for export");
  MOCHI_ERROR_RETURN(error, "");

  Shape const* shapePtr = shapeComponent->shape.get();
  MOCHI_ASSERT(shapePtr, "A null shape pointer should not be in the registry");

  // Check cache
  auto it = exportedShapeCache.find(shapePtr);
  if (it != exportedShapeCache.end()) {
    // Return the previously exported file path
    return it->second;
  }

  // Serialize the model data to an H5 file.
  ModelData data = shapePtr->GetModelData(error);
  model::SaveToFile(data, filePath, FileFormat::H5, error);
  MOCHI_ERROR_RETURN(error, "");

  // Cache the result
  exportedShapeCache[shapePtr] = DynamicString(relativeFilePath);

  return DynamicString(relativeFilePath);
}

// Helper function to extract RigidActorParams for standalone rigids and articulated links
static RigidActorParams ExtractRigidActorParamsHelper(
    Actor const* actor,
    entt::registry const& registry,
    entt::entity entity) {
  MOCHI_ASSERT(actor->GetType() == ActorType::Rigid, "Expected a rigid actor");

  RigidActorParams params;

  // Static rigid actors must be tagged "static"; static articulated links must not.
  // Static actors and dummy articulated links with no mesh do not have mass-related properties.
  bool const actorIsStatic = actor->IsStatic();
  bool const shouldExtractPhysics = !actorIsStatic && !actor->GetSurfaceMesh().IsEmpty();
  params.isStatic = actorIsStatic && !actor->IsNestedLinkActor();

  if (shouldExtractPhysics) {
    Error densityError;
    Error massError;
    Error centerOfMassError;
    Error momentOfInertiaError;

    auto density = actor->GetDensity(densityError);
    (void)actor->GetMass(massError); // Check mass is available but don't use it
    auto centerOfMass = actor->GetRigidCenterOfMassLocal(centerOfMassError);
    auto momentOfInertia = actor->GetRigidMomentOfInertiaLocal(momentOfInertiaError);

    // For dynamic actors with shapes, all physics properties must be available
    MOCHI_ASSERT(
        densityError.IsOK() && massError.IsOK() && centerOfMassError.IsOK() &&
            momentOfInertiaError.IsOK(),
        "Dynamic actors with shapes should have all physics properties available");

    // Use the extracted values
    params.density = density;
    params.centerOfMass = centerOfMass;
    params.momentOfInertia = momentOfInertia;
    params.hasGravity = registry.all_of<TagUseGravity>(entity);
  }

  // Extract contact parameters
  Error contactError;
  auto contactParams = actor->GetContactParams(contactError);
  if (contactError.IsOK()) {
    params.contact = contactParams;
  }

  // Extract layer
  params.layer = actor->GetContactLayer();

  // Extract collider type from ECS registry
  auto const* colliderInfo = registry.try_get<CColliderInfo const>(entity);
  params.colliderType = colliderInfo ? colliderInfo->type : ColliderType::None;

  MOCHI_ASSERT(
      registry.all_of<CRigidExportParams>(entity), "Missing CRigidExportParams on a rigid actor.");
  auto const& exportParams = registry.get<CRigidExportParams const>(entity);
  params.boundaryElementType = exportParams.boundaryElementType;
  params.boundarySubsampling = exportParams.boundarySubsampling;

  return params;
}

static RigidActorPrefab ExportRigidActor(
    Actor const* actor,
    entt::registry const& registry,
    entt::entity entity,
    std::unordered_map<Shape const*, DynamicString>& shapeCache,
    std::unordered_map<entt::entity, DynamicString> const& exportActorNames,
    std::unordered_map<std::string, int>& usedFileNames,
    std::string_view assetsDir,
    std::string_view relativePrefix,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  RigidActorPrefab prefab;

  // Look up pre-assigned name from the map
  auto it = exportActorNames.find(entity);
  MOCHI_ERROR_IF(it == exportActorNames.end(), error, "Actor name not found in export map");
  MOCHI_ERROR_RETURN(error, {});
  prefab.name = it->second;
  prefab.layer = actor->GetContactLayer();
  prefab.isStatic = actor->IsStatic();

  // Transform
  auto transform = actor->GetRootTransform();
  prefab.translation = transform.GetTranslation();
  prefab.rotation = transform.GetRotation();
  prefab.scale = Real3{1_r, 1_r, 1_r}; // Default scale

  // Generate shape file paths
  auto [relativePath, fullShapePath] =
      GenerateShapeFilePaths(prefab.name, "rigid", relativePrefix, assetsDir, usedFileNames);

  // Save mesh data from the actor
  DynamicString shapeFile =
      SaveActorMeshToFile(actor, fullShapePath.string(), relativePath, shapeCache, error);
  MOCHI_ERROR_RETURN(error, {}); // Return if export failed
  prefab.shapeFile = shapeFile;

  // Extract rigid actor parameters
  auto params = ExtractRigidActorParamsHelper(actor, registry, entity);

  // Copy extracted parameters to prefab structure
  prefab.contact = params.contact;
  prefab.colliderType = params.colliderType;
  prefab.boundaryElementType = params.boundaryElementType;
  prefab.boundarySubsampling = params.boundarySubsampling;

  // For dynamic actors, export physics properties and gravity flag
  if (!prefab.isStatic) {
    prefab.hasGravity = params.hasGravity;
    prefab.density = params.density;
    prefab.centerOfMass = params.centerOfMass;
    prefab.momentOfInertia = params.momentOfInertia;
  }

  MOCHI_LOG("Exported rigid actor '%s'", prefab.name.c_str());
  return prefab;
}

static SoftActorPrefab ExportSoftActor(
    Actor const* actor,
    entt::registry const& registry,
    entt::entity entity,
    std::unordered_map<Shape const*, DynamicString>& shapeCache,
    std::unordered_map<entt::entity, DynamicString> const& exportActorNames,
    std::unordered_map<std::string, int>& usedFileNames,
    std::string_view assetsDir,
    std::string_view relativePrefix,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  SoftActorPrefab prefab;

  // Look up pre-assigned name from the map
  auto it = exportActorNames.find(entity);
  MOCHI_ERROR_IF(it == exportActorNames.end(), error, "Actor name not found in export map");
  MOCHI_ERROR_RETURN(error, {});
  prefab.name = it->second;
  prefab.layer = actor->GetContactLayer();

  // Transform
  auto transform = actor->GetRootTransform();
  prefab.translation = transform.GetTranslation();
  prefab.rotation = transform.GetRotation();
  prefab.scale = Real3{1_r, 1_r, 1_r}; // Default scale

  // Generate shape file paths
  auto [relativePath, fullShapePath] =
      GenerateShapeFilePaths(prefab.name, "soft", relativePrefix, assetsDir, usedFileNames);

  // Save mesh data from the actor
  DynamicString shapeFile =
      SaveActorMeshToFile(actor, fullShapePath.string(), relativePath, shapeCache, error);
  MOCHI_ERROR_IF(shapeFile.empty(), error, "Soft actor does not have a supported mesh shape.");
  MOCHI_ERROR_RETURN(error, {});

  // Export path to mesh
  prefab.shapeFile = shapeFile;

  // Physics properties - query ECS tags
  prefab.hasGravity = registry.all_of<TagUseGravity>(entity);
  prefab.hasInertia = registry.all_of<TagUseInertia>(entity);
  prefab.hasStress = registry.all_of<TagUseStress>(entity);

  if (auto const* recentering = registry.try_get<CRecenteringParams const>(entity)) {
    prefab.useRecentering = recentering->useRecentering;
  } else {
    prefab.useRecentering = false;
  }

  // Export collider type from ECS
  if (auto const* colliderInfo = registry.try_get<CColliderInfo>(entity)) {
    prefab.colliderType = colliderInfo->type;
  }

  // Export boundary element type
  MOCHI_ASSERT(
      registry.all_of<CSoftExportParams>(entity), "Missing CSoftExportParams on a soft actor.");
  prefab.boundaryElementType = registry.get<CSoftExportParams const>(entity).boundaryElementType;

  // Export physics properties
  prefab.contact = actor->GetContactParams(error);

  // Export material properties
  prefab.material = actor->GetSoftMaterialParams(error);
  MOCHI_ERROR_RETURN(error, {});

  MOCHI_LOG("Exported soft actor '%s'", prefab.name.c_str());
  return prefab;
}

// Helper function to export contact filter settings from the scene
static std::optional<ContactFilter> ExportContactFilter(
    entt::registry const& registry,
    std::unordered_map<entt::entity, DynamicString> const& exportedActorNames,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  ContactFilter contactFilter;
  contactFilter.actorContactAsymmetric = DynamicArray<ActorContactEntry>{};
  contactFilter.actorContactSymmetric = DynamicArray<ActorContactEntry>{};
  contactFilter.layerContactAsymmetric = DynamicArray<LayerContactEntry>{};
  contactFilter.layerContactSymmetric = DynamicArray<LayerContactEntry>{};

  // Access the contact filter data from the registry
  auto const& contactTable = registry.ctx<CContactFilterTable const>();

  // Build reverse map: ContactLayerId -> layer name
  std::unordered_map<ContactLayerId, DynamicString> layerIdToName;
  for (auto const& [name, id] : contactTable.layerNameToId) {
    layerIdToName[id] = name;
  }

  // Directly iterate over disabled layer pairs
  for (auto const& [layerAId, layerBId] : contactTable.layersWithNoContact) {
    // Look up layer names
    auto itA = layerIdToName.find(layerAId);
    auto itB = layerIdToName.find(layerBId);
    MOCHI_ASSERT(
        itA != layerIdToName.end() && itB != layerIdToName.end(),
        "Layer IDs should be in layerIdToName");
    auto const& nameA = itA->second;
    auto const& nameB = itB->second;

    // Check if reverse pair also disabled to determine symmetry
    if (contactTable.layersWithNoContact.contains({layerBId, layerAId})) {
      // Both directions disabled - export as symmetric, only when A <= B to avoid duplicates
      if (layerAId <= layerBId) {
        LayerContactEntry entry;
        entry.enable = false;
        entry.layers = {nameA, nameB};
        contactFilter.layerContactSymmetric->push_back(entry);
      }
    } else {
      // Only one direction disabled - export as asymmetric
      LayerContactEntry entry;
      entry.enable = false;
      entry.layers = {nameA, nameB};
      contactFilter.layerContactAsymmetric->push_back(entry);
    }
  }

  // Directly iterate over disabled entity pairs
  for (auto const& [entityA, entityB] : contactTable.entitiesWithNoContact) {
    // Look up actor names — skip entries referencing actors not in the export
    // (e.g. actors excluded via ExportSceneExcluding).
    auto itA = exportedActorNames.find(entityA);
    auto itB = exportedActorNames.find(entityB);
    if (itA == exportedActorNames.end() || itB == exportedActorNames.end()) {
      continue;
    }
    auto const& nameA = itA->second;
    auto const& nameB = itB->second;

    bool const containsExpandableParent =
        actor::CanOwnNestedActors(registry.get<CActorInfo const>(entityA).type) ||
        actor::CanOwnNestedActors(registry.get<CActorInfo const>(entityB).type);

    ActorContactEntry entry;
    entry.enable = false;
    entry.actors = {nameA, nameB};
    if (containsExpandableParent) {
      // Each exported entry represents one concrete disabled pair. Prevent a parent name from
      // expanding to additional pairs when the exported prefab is loaded.
      entry.includeNestedActors = false;
    }

    // Check if reverse pair also disabled to determine symmetry
    if (contactTable.entitiesWithNoContact.contains({entityB, entityA})) {
      // Both directions disabled - export as symmetric, only when A <= B to avoid duplicates
      if (entityA <= entityB) {
        contactFilter.actorContactSymmetric->push_back(entry);
      }
    } else {
      // Only one direction disabled - export as asymmetric
      contactFilter.actorContactAsymmetric->push_back(entry);
    }
  }

  // Prune empty optional arrays
  if (contactFilter.actorContactAsymmetric->empty()) {
    contactFilter.actorContactAsymmetric = std::nullopt;
  }
  if (contactFilter.actorContactSymmetric->empty()) {
    contactFilter.actorContactSymmetric = std::nullopt;
  }
  if (contactFilter.layerContactAsymmetric->empty()) {
    contactFilter.layerContactAsymmetric = std::nullopt;
  }
  if (contactFilter.layerContactSymmetric->empty()) {
    contactFilter.layerContactSymmetric = std::nullopt;
  }

  return contactFilter;
}

static ArticulatedActorPrefab ExportArticulatedActorImpl(
    Actor const* actor,
    entt::registry const& registry,
    entt::entity entity,
    std::unordered_map<Shape const*, DynamicString>& shapeCache,
    std::unordered_map<entt::entity, DynamicString> const& exportActorNames,
    std::unordered_map<std::string, int>& usedFileNames,
    std::string_view assetsDir,
    std::string_view relativePrefix,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ASSERT(actor != nullptr);

  // Look up pre-assigned name from the map
  auto it = exportActorNames.find(entity);
  MOCHI_ERROR_IF(it == exportActorNames.end(), error, "Actor name not found in export map");
  MOCHI_ERROR_RETURN(error, {});
  DynamicString const& actorName = it->second;

  // Get the articulated shape info and nested link actors
  auto artInfo = actor->GetArticulatedShapeInfo(error);
  auto linkActorHandles = actor->GetNestedLinkActors(error);
  MOCHI_ERROR_RETURN(error, {});

  int const numLinks = isize(linkActorHandles);
  int const numJointsAndCycles = isize(artInfo.jointTypes);
  int const numCycles = isize(artInfo.cycles);
  int const numJoints = numJointsAndCycles - numCycles;

  // More required components
  auto const& inertiaParams = registry.get<CArticulatedInertiaParams const>(entity);
  auto const& frictionParams = registry.get<CArticulatedJointFrictionParams const>(entity);
  auto const* jointLimitConstraints = registry.try_get<CArticulatedJointLimits const>(entity);
  auto const* cycleConstraints =
      registry.try_get<CArticulatedCycleJoints const>(entity); // Required if there are cycles
  MOCHI_ERROR_IF(
      !jointLimitConstraints || (numCycles && !cycleConstraints),
      error,
      "Missing internal component. Cannot export without it.");
  MOCHI_ERROR_RETURN(error, {});

  // Array size sanity checks
  MOCHI_ERROR_IF(numLinks != numJoints, error, "Array size mismatch");
  MOCHI_ERROR_IF(isize(artInfo.rootFromLinksAtRest) != numLinks, error, "Array size mismatch");
  MOCHI_ERROR_IF(isize(artInfo.linkNames) != numLinks, error, "Array size mismatch");
  MOCHI_ERROR_IF(isize(artInfo.parents) != numLinks, error, "Array size mismatch");
  MOCHI_ERROR_IF(isize(artInfo.jointAxes) != numJointsAndCycles, error, "Array size mismatch");
  MOCHI_ERROR_IF(
      isize(artInfo.jointFromChildLink) != numJointsAndCycles, error, "Array size mismatch");
  MOCHI_ERROR_IF(
      isize(artInfo.parentLinkFromJoint) != numJointsAndCycles, error, "Array size mismatch");
  MOCHI_ERROR_IF(isize(artInfo.jointMinLimits) != numJointsAndCycles, error, "Array size mismatch");
  MOCHI_ERROR_IF(isize(artInfo.jointMaxLimits) != numJointsAndCycles, error, "Array size mismatch");
  MOCHI_ERROR_IF(isize(artInfo.jointNames) != numJointsAndCycles, error, "Array size mismatch");
  MOCHI_ERROR_IF(isize(inertiaParams) != numJointsAndCycles, error, "Array size mismatch");
  MOCHI_ERROR_IF(isize(frictionParams) != numJointsAndCycles, error, "Array size mismatch");
  MOCHI_ERROR_IF(
      numCycles != 0 && numCycles != isize(*cycleConstraints), error, "Array size mismatch");
  MOCHI_ERROR_RETURN(error, {});

  Scene const* scene = actor->GetScene();

  // Links
  DynamicArray<ArticulatedLinkPrefab> links(numLinks);
  for (int i = 0; i < numLinks; ++i) {
    Actor const* linkActor = scene->GetActor(linkActorHandles[i]);
    MOCHI_ASSERT(linkActor != nullptr, "GetNestedLinkActors should not give us invalid handles");
    auto linkEntity = GetEntity(registry, linkActorHandles[i], ErrorAssert{});
    auto rigidParams = ExtractRigidActorParamsHelper(linkActor, registry, linkEntity);
    MOCHI_ASSERT(!artInfo.linkNames[i].empty(), "Articulated link names must be non-empty.");

    // Export the link shape (if any)
    DynamicString linkShapeFilePath;
    if (!linkActor->GetSurfaceMesh().IsEmpty()) {
      auto linkFileBaseName =
          DynamicString(Format("%s_%s", actorName.c_str(), artInfo.linkNames[i].c_str()));
      auto [linkRelativePath, linkFullPath] =
          GenerateShapeFilePaths(linkFileBaseName, "", relativePrefix, assetsDir, usedFileNames);
      linkShapeFilePath = SaveActorMeshToFile(
          linkActor, linkFullPath.string(), linkRelativePath, shapeCache, error);
      MOCHI_ERROR_RETURN(error, {});
    }

    // Fill out ArticulatedLinkParams
    auto& link = links[i];
    link.name = artInfo.linkNames[i];
    link.parentLink = artInfo.parents[i];
    link.parentJointFromLink = artInfo.jointFromChildLink[i];
    link.shapeFile = std::move(linkShapeFilePath);
    link.layer = std::move(rigidParams.layer);
    link.colliderType = rigidParams.colliderType;
    link.contact = rigidParams.contact;
    link.hasGravity = rigidParams.hasGravity;
    link.density = rigidParams.density;
    link.centerOfMass = rigidParams.centerOfMass;
    link.momentOfInertia = rigidParams.momentOfInertia;
    link.boundaryElementType = rigidParams.boundaryElementType;
    link.boundarySubsampling = rigidParams.boundarySubsampling;

    // Apply automatic fixups to the base class (ArticulatedLinkParams)
    AutoCorrect(link, error);
    MOCHI_ERROR_RETURN(error, {});
  }

  // Joints
  DynamicArray<ArticulatedJointPrefab> joints(numJoints);
  for (int i = 0; i < numJoints; ++i) {
    auto& joint = joints[i];
    joint.name = artInfo.jointNames[i];
    joint.type = artInfo.jointTypes[i];
    joint.parentLinkFromJoint = artInfo.parentLinkFromJoint[i];
    joint.axis = artInfo.jointAxes[i];

    joint.friction = frictionParams[i];
    if (inertiaParams[i] != 0_r) {
      joint.inertia = inertiaParams[i];
    }

    if (artInfo.jointMinLimits[i] != -kInf3) {
      joint.minLimit = artInfo.jointMinLimits[i];
    }

    if (artInfo.jointMaxLimits[i] != kInf3) {
      joint.maxLimit = artInfo.jointMaxLimits[i];
    }
  }

  // Get stiffness and damping from the limit constraints (if any)
  for (auto const* constraint : *jointLimitConstraints) {
    MOCHI_ASSERT(constraint != nullptr, "List should not contain null pointers");
    auto constraintEntity = GetEntity(registry, constraint->GetHandle(), ErrorAssert{});
    auto const* data3d =
        registry.try_get<CConstraintData<ConstraintType::Articulated3dRotationRange> const>(
            constraintEntity);
    auto const* data1d =
        registry.try_get<CConstraintData<ConstraintType::ArticulatedSingleDofRange> const>(
            constraintEntity);
    if (!data3d && !data1d) {
      continue; // Skip non-articulated-joint constraints (e.g. cycle joint rotation range)
    }
    int const jointIdx = data3d ? data3d->jointIdx : data1d->jointIdx;
    MOCHI_ASSERT(jointIdx < numJoints, "Invalid joint index");

    joints[jointIdx].limitStiffness = constraint->GetStiffness();
    joints[jointIdx].limitDamping = constraint->GetDamping();
  }

  // Apply automatic fixups to the base class (ArticulatedJointParams)
  for (auto& joint : joints) {
    AutoCorrect(joint, error);
    MOCHI_ERROR_RETURN(error, {});
  }

  // Cycles
  DynamicArray<ArticulatedCycleJointParams> cycles(numCycles);
  for (int i = 0; i < numCycles; ++i) {
    auto const* constraint = scene->GetConstraint((*cycleConstraints)[i]);
    MOCHI_ERROR_IF(!constraint, error, "Cycle joint constraint not found");
    MOCHI_ERROR_RETURN(error, {});
    cycles[i].parentLink = artInfo.cycles[i].parent;
    cycles[i].childLink = artInfo.cycles[i].child;
    cycles[i].jointFromChildLink = artInfo.jointFromChildLink[numLinks + i];
    cycles[i].stiffness = constraint->GetStiffness();

    // Apply automatic fixups
    AutoCorrect(cycles[i], error);
    MOCHI_ERROR_RETURN(error, {});
  }

  // Skin
  std::optional<ArticulatedSkinPrefab> skin;
  if (!actor->GetSurfaceMesh().IsEmpty()) {
    auto skinFileBaseName = DynamicString(Format("%s_skin", actorName.c_str()));
    auto [skinRelativePath, skinFullPath] =
        GenerateShapeFilePaths(skinFileBaseName, "", relativePrefix, assetsDir, usedFileNames);
    DynamicString skinShapeFile =
        SaveActorMeshToFile(actor, skinFullPath.string(), skinRelativePath, shapeCache, error);
    MOCHI_ERROR_RETURN(error, {});

    ArticulatedSkinPrefab skinPrefab;
    skinPrefab.shapeFile = std::move(skinShapeFile);

    // Recover skin contact params from the ECS (stored on the articulated actor entity)
    skinPrefab.layer = actor->GetContactLayer();
    skinPrefab.contact = actor->GetContactParams(error);
    MOCHI_ERROR_RETURN(error, {});

    // Recover boundary params from the export component (stored during creation via the New API)
    auto const* skinExportParams = registry.try_get<CArticulatedSkinExportParams const>(entity);
    if (skinExportParams) {
      skinPrefab.boundaryElementType = skinExportParams->boundaryElementType;
      skinPrefab.boundarySubsampling = skinExportParams->boundarySubsampling;
    }

    skin = std::move(skinPrefab);
  }

  // Put it all together
  ArticulatedActorPrefab prefabActor;
  prefabActor.name = actorName;
  auto const rootTransform = actor->GetRootTransform();
  prefabActor.translation = rootTransform.GetTranslation();
  prefabActor.rotation = Normalize(rootTransform.GetRotation());
  prefabActor.cycles = std::move(cycles);
  prefabActor.joints = std::move(joints);
  prefabActor.links = std::move(links);
  prefabActor.skin = std::move(skin);

  // Joint velocities are not currently exported. If we ever want this, then we will probably also
  // want the current pose (not necessarily the rest pose).
  prefabActor.jointVelocities = std::nullopt;

  MOCHI_LOG("Exported articulated actor '%s'", prefabActor.name.c_str());

  return prefabActor;
}

static SoftSkinnedActorPrefab ExportSoftSkinnedActor(
    Actor const* actor,
    entt::registry const& registry,
    entt::entity entity,
    std::unordered_map<Shape const*, DynamicString>& shapeCache,
    std::unordered_map<entt::entity, DynamicString> const& exportActorNames,
    std::unordered_map<std::string, int>& usedFileNames,
    std::string_view assetsDir,
    std::string_view relativePrefix,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  SoftSkinnedActorPrefab prefab;

  // Export the articulated skeleton - this is just the underlying articulated actor
  prefab.skeletonParams = ExportArticulatedActorImpl(
      actor,
      registry,
      entity,
      shapeCache,
      exportActorNames,
      usedFileNames,
      assetsDir,
      relativePrefix,
      error);
  MOCHI_ERROR_RETURN(error, {});

  // Get nested soft actors to export
  auto nestedSoftActors = actor->GetNestedSoftActors(error);
  MOCHI_ERROR_RETURN(error, {});

  // Export each nested soft actor
  prefab.softParams.reserve(nestedSoftActors.size());
  for (auto softHandle : nestedSoftActors) {
    Actor const* softActor = actor->GetScene()->GetActor(softHandle);
    auto softEntity = GetEntity(registry, softHandle, ErrorAssert{});
    auto softPrefab = ExportSoftActor(
        softActor,
        registry,
        softEntity,
        shapeCache,
        exportActorNames,
        usedFileNames,
        assetsDir,
        relativePrefix,
        error);
    MOCHI_ERROR_RETURN(error, {});

    // During loading the parent name is prepended again, so store only the effective local name.
    softPrefab.name = GetNestedActorLocalName(softPrefab.name);
    MOCHI_ASSERT(!softPrefab.name.empty(), "Nested soft local name must be non-empty.");

    // Nested soft actors cannot use recentering or unposed gravity.
    softPrefab.useRecentering = false;
    softPrefab.hasGravity = false;

    prefab.softParams.push_back(std::move(softPrefab));
  }

  // Extract soft-specific physics flags for the soft skinned system
  prefab.hasGravity = registry.all_of<TagUseGravity>(entity);
  prefab.hasInertia = registry.all_of<TagUseInertia>(entity);
  prefab.hasStress = registry.all_of<TagUseStress>(entity);

  // Determine enableCollidingLinks by checking if any link has TagUseContact
  auto linkActors = actor->GetNestedLinkActors(ErrorAssert{});
  prefab.enableCollidingLinks = false; // Default to false
  for (auto linkHandle : linkActors) {
    auto linkEntity = GetEntity(registry, linkHandle, ErrorAssert{});
    if (registry.all_of<TagUseContact>(linkEntity)) {
      prefab.enableCollidingLinks = true;
      break; // At least one link has TagUseContact, so we're done
    }
  }

  // Recover soft attachment links from the stored ECS component
  auto const* attachmentLinksComp = registry.try_get<CSoftAttachmentLinks const>(entity);

  // Only emit softAttachLinks when external attachments exist. For internally skinned soft-skinned
  // actors it must stay empty. The loader treats a non-empty softAttachLinks array as one link name
  // per soft actor, so unconditionally resizing it would serialize empty link names that fail to
  // resolve on re-import.
  if (attachmentLinksComp && !attachmentLinksComp->softAttachLinks.empty()) {
    MOCHI_ASSERT(
        isize(attachmentLinksComp->softAttachLinks) == isize(prefab.softParams),
        "Soft attachment link count does not match soft actor count.");

    // Resize to match the number of soft actors
    prefab.softAttachLinks.resize(prefab.softParams.size());

    // Get articulated shape info to convert link indices to names
    auto artInfo = actor->GetArticulatedShapeInfo(error);
    MOCHI_ERROR_RETURN(error, {});

    // Convert from link indices back to link names for the prefab format
    for (int softIndex = 0; softIndex < isize(attachmentLinksComp->softAttachLinks); ++softIndex) {
      int linkIndex = attachmentLinksComp->softAttachLinks[softIndex];

      MOCHI_ASSERT(
          linkIndex >= 0 && linkIndex < isize(artInfo.linkNames),
          "Invalid link index %d for soft actor %d. Link indices should be validated during soft skinned actor creation.",
          linkIndex,
          softIndex);
      prefab.softAttachLinks[softIndex] = DynamicString(artInfo.linkNames[linkIndex]);
    }
  }

  return prefab;
}

// Creates the prefab export directory layout:
//   <outputDir>/<exportName>/
//   <outputDir>/<exportName>/generated_assets/
//
// Returns the (exportDir, assetsDir) pair. Sets `error` and returns empty paths on failure.
static std::pair<std::filesystem::path, std::filesystem::path> CreatePrefabExportDirectories(
    std::string_view outputDir,
    std::string_view exportName,
    Error& error) {
  std::filesystem::path const outputPath = std::filesystem::path(outputDir).lexically_normal();
  std::filesystem::path const exportDir = outputPath / exportName;
  std::filesystem::path const assetsDir = exportDir / "generated_assets";

  try {
    std::filesystem::create_directories(assetsDir);
  } catch (std::exception const& e) {
    MOCHI_LOG_ERROR("Failed to create export directories: %s", e.what());
    MOCHI_ERROR_SET(error, "Failed to create export directories");
    return {};
  }
  return {exportDir, assetsDir};
}

// Serializes `prefab` to <exportDir>/<exportName>.mochi_scene and logs the export directory.
static void WritePrefabToDisk(
    ScenePrefab const& prefab,
    std::filesystem::path const& exportDir,
    std::string_view exportName,
    Error& error) {
  std::filesystem::path const prefabFilePath =
      exportDir / (DynamicString(exportName) + ".mochi_scene");
  auto const json = SReflect::ToJsonString(prefab, true);
  WriteFile(prefabFilePath, json, error);
  if (error.IsOK()) {
    MOCHI_LOG("Export directory: %s", exportDir.string().c_str());
  }
}

// Dispatches a single actor to the appropriate type-specific exporter and appends the
// resulting prefab to `prefab`. Returns false (without setting `error`) when the actor type
// is unsupported, leaving the unsupported-type policy to the caller.
static bool ExportActorToPrefab(
    Actor const* actor,
    entt::registry const& registry,
    entt::entity entity,
    std::unordered_map<Shape const*, DynamicString>& shapeCache,
    std::unordered_map<entt::entity, DynamicString> const& exportActorNames,
    std::unordered_map<std::string, int>& usedFileNames,
    std::string_view assetsDir,
    std::string_view relativePrefix,
    ScenePrefab& prefab,
    Error& error) {
  MOCHI_ERROR_RETURN(error, false);

  switch (actor->GetType()) {
    case ActorType::Rigid: {
      auto rigidPrefab = ExportRigidActor(
          actor,
          registry,
          entity,
          shapeCache,
          exportActorNames,
          usedFileNames,
          assetsDir,
          relativePrefix,
          error);
      MOCHI_ERROR_RETURN(error, false);
      prefab.actors.rigid.push_back(std::move(rigidPrefab));
      break;
    }
    case ActorType::Soft: {
      auto softPrefab = ExportSoftActor(
          actor,
          registry,
          entity,
          shapeCache,
          exportActorNames,
          usedFileNames,
          assetsDir,
          relativePrefix,
          error);
      MOCHI_ERROR_RETURN(error, false);
      prefab.actors.soft.push_back(std::move(softPrefab));
      break;
    }
    case ActorType::Articulated: {
      // A soft skinned actor is an articulated actor that owns nested soft actors.
      auto nestedSoftActors = actor->GetNestedSoftActors(ErrorAssert{});
      if (!nestedSoftActors.empty()) {
        auto softSkinnedPrefab = ExportSoftSkinnedActor(
            actor,
            registry,
            entity,
            shapeCache,
            exportActorNames,
            usedFileNames,
            assetsDir,
            relativePrefix,
            error);
        MOCHI_ERROR_RETURN(error, false);
        prefab.actors.softSkinned.push_back(std::move(softSkinnedPrefab));
      } else {
        auto articulatedPrefab = ExportArticulatedActorImpl(
            actor,
            registry,
            entity,
            shapeCache,
            exportActorNames,
            usedFileNames,
            assetsDir,
            relativePrefix,
            error);
        MOCHI_ERROR_RETURN(error, false);
        prefab.actors.articulated.push_back(std::move(articulatedPrefab));
      }
      break;
    }
    case ActorType::None:
    case ActorType::Shell:
    case ActorType::Rod:
    case ActorType::Count:
      // Unsupported for prefab export.
      MOCHI_ASSERT_VERBOSE(
          !IsActorTypeSupportedForPrefabExport(actor->GetType()),
          "IsActorTypeSupportedForPrefabExport must match ExportActorToPrefab.");
      return false;
  }

  MOCHI_ASSERT_VERBOSE(
      IsActorTypeSupportedForPrefabExport(actor->GetType()),
      "IsActorTypeSupportedForPrefabExport must match ExportActorToPrefab.");
  return true;
}

MOCHI_API void prefab::ExportScene(
    Scene const* scene,
    std::string_view exportName,
    std::string_view outputDir,
    Error& error) {
  ExportSceneExcluding(scene, exportName, outputDir, {}, error);
}

MOCHI_API void prefab::ExportSceneExcluding(
    Scene const* scene,
    std::string_view exportName,
    std::string_view outputDir,
    Span<ActorHandle const> excludeActors,
    Error& error) {
  MOCHI_ERROR_IF(scene == nullptr, error, "Scene must not be null.");
  MOCHI_ERROR_RETURN(error);

  // Build exclusion set for O(1) lookup.
  std::unordered_set<ActorHandle> excludeSet(excludeActors.begin(), excludeActors.end());

  // Nested link/soft actors are exported through their top-level parent, so a nested handle cannot
  // be excluded on its own.
  for (auto handle : excludeActors) {
    Actor const* actor = scene->GetActor(handle);
    MOCHI_ERROR_IF(
        actor != nullptr && (actor->IsNestedLinkActor() || actor->IsNestedSoftActor()),
        error,
        "ExportSceneExcluding does not support nested actor handles. Pass the top-level "
        "articulated or soft-skinned actor handle instead.");
    MOCHI_ERROR_RETURN(error);
  }

  // Track naming collisions across actors and exported asset files.
  std::unordered_set<std::string> usedActorNames;
  std::unordered_map<std::string, int> usedFileNames;

  // Get ECS registry
  auto const* sceneImpl = static_cast<SceneImpl const*>(scene);
  auto const& registry = sceneImpl->GetRegistry();
  std::unordered_map<entt::entity, DynamicString> exportActorNames;

  auto isExcluded = [&](Actor const* actor) {
    if (excludeSet.contains(actor->GetHandle())) {
      return true;
    }
    if (actor->IsNestedLinkActor()) {
      auto parent = actor->GetArticulatedActor(ErrorAssert{});
      if (excludeSet.contains(parent)) {
        return true;
      }
    }
    if (actor->IsNestedSoftActor()) {
      auto entity = GetEntity(registry, actor->GetHandle(), ErrorAssert{});
      MOCHI_ASSERT(registry.all_of<CSkinnedComposition>(entity), "Missing skinned composition.");
      auto const& composition = registry.get<CSkinnedComposition const>(entity);
      if (excludeSet.contains(composition.articulatedHandle)) {
        return true;
      }
    }
    return false;
  };

  auto buildExportNameBundle = [&](Actor const* actor,
                                   entt::entity entity,
                                   DynamicString const& exportActorName) {
    DynamicArray<ExportActorNameEntry> bundle;
    bundle.push_back({entity, exportActorName});

    if (actor->GetType() != ActorType::Articulated) {
      return bundle;
    }

    auto linkActors = actor->GetNestedLinkActors(ErrorAssert{});
    auto artInfo = actor->GetArticulatedShapeInfo(ErrorAssert{});
    MOCHI_ASSERT(
        isize(artInfo.linkNames) == isize(linkActors),
        "Articulated link name/actor count mismatch during prefab export.");
    // Creation guarantees nested local actor names are non-empty and unique within the parent.
    // Export keeps that invariant explicit because prefab actor references share this flat bundle.
    std::unordered_set<std::string> usedLocalNames;
    for (int i = 0; i < isize(artInfo.linkNames); ++i) {
      auto linkEntity = GetEntity(registry, linkActors[i], ErrorAssert{});

      MOCHI_ASSERT(!artInfo.linkNames[i].empty(), "Articulated link names must be non-empty.");
      [[maybe_unused]] bool const insertedLink =
          usedLocalNames.insert(std::string(artInfo.linkNames[i])).second;
      MOCHI_ASSERT(insertedLink, "Nested link actor local names must be unique.");
      bundle.push_back({linkEntity, MakeNestedActorName(exportActorName, artInfo.linkNames[i])});
    }

    auto softActors = actor->GetNestedSoftActors(ErrorAssert{});
    for (auto softHandle : softActors) {
      auto softEntity = GetEntity(registry, softHandle, ErrorAssert{});
      auto const* softActor = scene->GetActor(softHandle);
      MOCHI_ASSERT(softActor, "GetNestedSoftActors should provide valid handles.");

      DynamicString const localName = GetNestedActorLocalName(softActor->GetName());
      MOCHI_ASSERT(!localName.empty(), "Nested soft local name must be non-empty.");
      [[maybe_unused]] bool const insertedSoft =
          usedLocalNames.insert(std::string(localName)).second;
      MOCHI_ASSERT(insertedSoft, "Nested soft actor local names must be unique.");
      bundle.push_back({softEntity, MakeNestedActorName(exportActorName, localName)});
    }

    return bundle;
  };

  // Pre-assign unique names to all scene actors before export begins.
  scene->ForEachActor([&](Actor const* actor) {
    // Nested actors are exported through their parent actor. Skip them in the top-level pass so
    // they are not serialized as standalone actors or assigned names independent of the parent's
    // export name.
    if (isExcluded(actor) || actor->IsNestedLinkActor() || actor->IsNestedSoftActor() ||
        !IsActorTypeSupportedForPrefabExport(actor->GetType())) {
      return;
    }

    auto entity = GetEntity(registry, actor->GetHandle(), ErrorAssert{});
    DynamicString const baseName = GetActorBaseName(actor);

    // Parent and nested actor names share one flat namespace on import, so reserve the whole
    // hierarchy atomically for a candidate parent name before committing any of it. Try successive
    // suffixes until every name in the bundle is free. A free candidate always exists because
    // usedActorNames is finite and each suffix yields a distinct parent name.
    for (int suffix = 0;; ++suffix) {
      auto bundle = buildExportNameBundle(actor, entity, MakeNameCandidate(baseName, suffix));

      std::unordered_set<std::string> bundleNames;
      for (auto const& entry : bundle) {
        // buildExportNameBundle guarantees unique names within a bundle.
        [[maybe_unused]] bool const inserted = bundleNames.insert(std::string(entry.name)).second;
        MOCHI_ASSERT(inserted, "Prefab export generated duplicate actor name.");
      }

      // Retry with the next suffix if any name is already claimed by another actor.
      bool anyNameTaken = false;
      for (auto const& name : bundleNames) {
        if (usedActorNames.contains(name)) {
          anyNameTaken = true;
          break;
        }
      }
      if (anyNameTaken) {
        continue;
      }

      for (auto const& name : bundleNames) {
        usedActorNames.emplace(name);
      }
      for (auto& entry : bundle) {
        MOCHI_ASSERT(
            !exportActorNames.contains(entry.entity),
            "Prefab export generated duplicate actor entity.");
        exportActorNames.emplace(entry.entity, std::move(entry.name));
      }
      return;
    }
  });

  // Create export directories only after name assignment has succeeded, so a name error does not
  // leave stray output directories behind.
  auto const dirs = CreatePrefabExportDirectories(outputDir, exportName, error);
  MOCHI_ERROR_RETURN(error);
  auto const& exportDir = dirs.first;
  auto const& assetsDir = dirs.second;

  // Use prefab-relative paths
  DynamicString relativePrefix = "./generated_assets/";

  // Cache for mesh exports
  std::unordered_map<Shape const*, DynamicString> exportedShapeCache;

  ScenePrefab prefab;

  // Export scene settings
  SceneParams sceneParams;
  sceneParams.description = "Exported scene: " + DynamicString(scene->GetName());
  sceneParams.gravity = scene->GetGravity();
  sceneParams.solver = scene->GetSolverParams();
  prefab.scene = sceneParams;

  // Export all top-level actors.
  scene->ForEachActor([&](Actor const* actor) {
    if (!error.IsOK()) {
      return;
    }

    if (isExcluded(actor)) {
      return;
    }

    // Skip link actors - they are not top-level actors and will be handled
    // as part of their parent articulated actor
    if (actor->IsNestedLinkActor()) {
      return;
    }

    // Skip nested soft actors - they are not top-level actors and will be handled
    // as part of their parent soft skinned actor
    if (actor->IsNestedSoftActor()) {
      return;
    }

    auto entity = GetEntity(registry, actor->GetHandle(), ErrorAssert{});

    bool const handled = ExportActorToPrefab(
        actor,
        registry,
        entity,
        exportedShapeCache,
        exportActorNames,
        usedFileNames,
        assetsDir.string(),
        relativePrefix,
        prefab,
        error);
    MOCHI_ERROR_RETURN(error);
    if (!handled) {
      MOCHI_LOG_WARNING("Skipping actor '%s' with unsupported type for export", actor->GetName());
    }
  });

  // TODO: Export constraints (iterate through scene constraints and convert to prefab format)
  // TODO: Export controllers (access articulated actor controllers)
  // TODO: Export implicit shapes (generate shape files for non-infinite shapes like
  // boxes/spheres)
  // TODO: Fix implicit shape friction inconsistencies (export friction params or generate
  // explicit geometry)

  MOCHI_ERROR_RETURN(error);

  // Sort all actor arrays alphabetically
  auto sortByName = [](auto& actorContainer) {
    std::ranges::sort(
        actorContainer, {}, &std::remove_reference_t<decltype(actorContainer[0])>::name);
  };

  sortByName(prefab.actors.rigid);
  sortByName(prefab.actors.soft);
  sortByName(prefab.actors.articulated);
  std::ranges::sort(prefab.actors.softSkinned, {}, [](auto const& a) -> auto const& {
    return a.skeletonParams.name;
  });

  // Export contact filter settings after all actors are named
  prefab.contactFilter = ExportContactFilter(registry, exportActorNames, error);
  MOCHI_ERROR_RETURN(error);

  WritePrefabToDisk(prefab, exportDir, exportName, error);
}

MOCHI_API void prefab::ExportActor(
    Actor const* actor,
    std::string_view exportName,
    std::string_view outputDir,
    Error& error) {
  MOCHI_ERROR_IF(actor == nullptr, error, "Actor must not be null");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      actor->IsNestedLinkActor(), error, "ExportActor does not support nested link actors.");
  MOCHI_ERROR_IF(
      actor->IsNestedSoftActor(), error, "ExportActor does not support nested soft actors.");
  MOCHI_ERROR_RETURN(error);
  if (actor->GetType() == ActorType::Articulated) {
    auto const nestedSoftActors = actor->GetNestedSoftActors(ErrorAssert{});
    MOCHI_ERROR_IF(
        !nestedSoftActors.empty(),
        error,
        "ExportActor does not support soft-skinned actors. Use ExportScene instead.");
    MOCHI_ERROR_RETURN(error);
  }

  // Reject unsupported actor types before creating any output directory, so a failed export leaves
  // nothing on disk.
  MOCHI_ERROR_IF_NOT(
      IsActorTypeSupportedForPrefabExport(actor->GetType()),
      error,
      "Actor type is unsupported for prefab export.");
  MOCHI_ERROR_RETURN(error);

  auto const dirs = CreatePrefabExportDirectories(outputDir, exportName, error);
  MOCHI_ERROR_RETURN(error);
  auto const& exportDir = dirs.first;
  auto const& assetsDir = dirs.second;

  // Set up internal bookkeeping
  auto const* sceneImpl = assert_cast<SceneImpl const*>(actor->GetScene());
  auto const& registry = sceneImpl->GetRegistry();
  auto const entity = GetEntity(registry, actor->GetHandle(), ErrorAssert{});

  std::unordered_map<Shape const*, DynamicString> shapeCache;
  std::unordered_map<std::string, int> usedFileNames;
  std::unordered_map<entt::entity, DynamicString> exportActorNames;
  exportActorNames[entity] = DynamicString(exportName);
  DynamicString const relativePrefix = "./generated_assets/";

  // Build the prefab in memory (writes mesh files to disk as a side effect).
  ScenePrefab prefab;
  bool const handled = ExportActorToPrefab(
      actor,
      registry,
      entity,
      shapeCache,
      exportActorNames,
      usedFileNames,
      assetsDir.string(),
      relativePrefix,
      prefab,
      error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ASSERT(handled, "ExportActor type support precheck must match ExportActorToPrefab.");

  WritePrefabToDisk(prefab, exportDir, exportName, error);
}
