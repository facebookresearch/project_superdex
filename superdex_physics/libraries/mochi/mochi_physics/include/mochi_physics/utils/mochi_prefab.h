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

// Documentation for all public types and functions in this file lives:
//   mochi_physics/mochi_physics_prefab.mochi_gen

#pragma once

#include "mochi_physics_macros.h"

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/span.h>
#include <mochi_physics/mochi_physics.h>

#include <optional>
#include <string_view>

namespace mochi::prefab {

/***********************************************************************************************
  Prefab Types:
*/

struct SceneParams {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicString description{};
  std::optional<Real3> gravity = std::nullopt;
  std::optional<SolverParams> solver = std::nullopt;

  bool operator==(SceneParams const&) const = default;
};

struct ActorContactEntry {
  bool enable = true;
  DynamicArray<DynamicString> actors;
  bool includeNestedActors = true;

  bool operator==(ActorContactEntry const&) const = default;
};

struct LayerContactEntry {
  bool enable = true;
  DynamicArray<DynamicString> layers;

  bool operator==(LayerContactEntry const&) const = default;
};

struct ContactFilter {
  std::optional<DynamicString> comment = std::nullopt;
  std::optional<DynamicArray<ActorContactEntry>> actorContactAsymmetric{};
  std::optional<DynamicArray<ActorContactEntry>> actorContactSymmetric{};
  std::optional<DynamicArray<LayerContactEntry>> layerContactAsymmetric{};
  std::optional<DynamicArray<LayerContactEntry>> layerContactSymmetric{};

  bool operator==(ContactFilter const&) const = default;
};

struct ContactPairParamsOverrideEntry {
  DynamicArray<DynamicString> actors;
  ContactPairParamsOverride paramsOverride;

  bool operator==(ContactPairParamsOverrideEntry const&) const = default;
};

struct RigidActorPrefab : public RigidActorParams {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicString shapeFile{};
  Real3 scale{1_r, 1_r, 1_r};
  Quaternion shapeRotation{};
  Real3 shapeTranslation{};
  Quaternion rotation{};
  Real3 translation{};
  DynamicString renderModelFile;
  Real3 renderModelScale = {1_r, 1_r, 1_r};
  Quaternion renderModelRotation;
  Real3 renderModelTranslation = {};
};

struct SoftActorPrefab : public SoftActorParams {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicString shapeFile{};
  ColliderType colliderType = ColliderType::None;
  GridSdfParams sdf;
  DynamicString flowFile{};
  ShapeHandle flow;
  bool useRecentering = true;
  Real3 scale{1_r, 1_r, 1_r};
  Quaternion shapeRotation{};
  Real3 shapeTranslation{};
  Quaternion rotation{};
  Real3 translation{};
  DynamicString renderModelFile;
  Real3 renderModelScale = {1_r, 1_r, 1_r};
  Quaternion renderModelRotation;
  Real3 renderModelTranslation = {};
};

struct ArticulatedJointPrefab : ArticulatedJointParams {
  bool operator==(ArticulatedJointPrefab const&) const = default;
};

struct ArticulatedLinkPrefab : ArticulatedLinkParams {
  DynamicString shapeFile;
  Real3 shapeScale = {1_r, 1_r, 1_r};
  Quaternion shapeRotation;
  Real3 shapeTranslation = {};
  DynamicString renderModelFile;
  Real3 renderModelScale = {1_r, 1_r, 1_r};
  Quaternion renderModelRotation;
  Real3 renderModelTranslation = {};

  bool operator==(ArticulatedLinkPrefab const&) const = default;
};

struct ArticulatedSkinPrefab : ArticulatedSkinParams {
  DynamicString shapeFile;

  // TODO[T265138451]: Add support for baked-in transform
  // Real3 shapeScale = {1_r, 1_r, 1_r};
  // Quaternion shapeRotation;
  // Real3 shapeTranslation = {};

  DynamicString renderModelFile;
  Real3 renderModelScale = {1_r, 1_r, 1_r};
  Quaternion renderModelRotation;
  Real3 renderModelTranslation = {};

  bool operator==(ArticulatedSkinPrefab const&) const = default;
};

struct ArticulatedActorPrefab {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicString name;
  real scale = 1_r;
  Quaternion rotation;
  Real3 translation = {};
  DynamicArray<ArticulatedCycleJointParams> cycles;
  DynamicArray<ArticulatedJointPrefab> joints;
  DynamicArray<ArticulatedLinkPrefab> links;
  std::optional<ArticulatedSkinPrefab> skin;
  std::optional<DynamicArray<real>> jointVelocities;

  bool operator==(ArticulatedActorPrefab const&) const = default;
};

struct SoftSkinnedActorPrefab {
  std::optional<DynamicString> comment = std::nullopt;
  ArticulatedActorPrefab skeletonParams{};
  DynamicArray<SoftActorPrefab> softParams{};
  DynamicArray<DynamicString> softAttachLinks{};
  bool enableCollidingLinks = false;
  bool hasGravity = false;
  bool hasInertia = false;
  bool hasStress = false;

  bool operator==(SoftSkinnedActorPrefab const&) const = default;
};

struct ActorLists {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicArray<ArticulatedActorPrefab> articulated{};
  DynamicArray<RigidActorPrefab> rigid{};
  DynamicArray<SoftActorPrefab> soft{};
  DynamicArray<SoftSkinnedActorPrefab> softSkinned{};

  bool operator==(ActorLists const&) const = default;
};

struct RigidSphericalJointConstraintPrefab : public RigidSphericalJointConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};
};

struct RigidPrismaticJointConstraintPrefab : public RigidPrismaticJointConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};
};

struct DeformableNodeToDeformableNodeConstraintPrefab
    : public DeformableNodeToDeformableNodeConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};
};

struct DeformableNodeToRigidConstraintPrefab : public DeformableNodeToRigidConstraintParams {
  DynamicString rigidActorName{};
  DynamicString deformableActorName{};
};

struct JointRotationRangeConstraintPrefab : public JointRotationRangeConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};
};

struct RigidPivotPositionConstraintPrefab : public RigidPivotPositionConstraintParams {
  DynamicString actorName{};
};

struct RigidPivotToRigidTargetConstraintPrefab : public RigidPivotToRigidTargetConstraintParams {
  DynamicString actorName{};
};

struct RigidPivotRotationConstraintPrefab : public RigidPivotRotationConstraintParams {
  DynamicString actorName{};
};

struct DeformableNodePositionConstraintPrefab : public DeformableNodePositionConstraintParams {
  DynamicString actorName{};

  bool operator==(DeformableNodePositionConstraintPrefab const&) const = default;
};

struct JointRotationTrackingConstraintPrefab : public JointRotationTrackingConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};
};

struct ArticulatedSingleDofTargetConstraintPrefab
    : public ArticulatedSingleDofTargetConstraintParams {
  DynamicString actorName;
};

struct Articulated3dRotationTargetConstraintPrefab
    : public Articulated3dRotationTargetConstraintParams {
  DynamicString actorName{};
};

struct ArticulatedSingleDofRangeConstraintPrefab
    : public ArticulatedSingleDofRangeConstraintParams {
  DynamicString actorName{};
};

struct Articulated3dRotationRangeConstraintPrefab
    : public Articulated3dRotationRangeConstraintParams {
  DynamicString actorName{};
};

/////////////////

struct ConstraintLists {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicArray<Articulated3dRotationRangeConstraintPrefab> articulated3dRotationRange{};
  DynamicArray<ArticulatedSingleDofRangeConstraintPrefab> articulatedSingleDofRange{};
  DynamicArray<Articulated3dRotationTargetConstraintPrefab> articulated3dRotationTarget{};
  DynamicArray<ArticulatedSingleDofTargetConstraintPrefab> articulatedSingleDofTarget{};
  DynamicArray<JointRotationRangeConstraintPrefab> jointRotationRange{};
  DynamicArray<JointRotationTrackingConstraintPrefab> jointRotationTracking{};
  DynamicArray<RigidPivotPositionConstraintPrefab> rigidPivotPosition{};
  DynamicArray<RigidPivotToRigidTargetConstraintPrefab> rigidPivotToRigidTarget{};
  DynamicArray<RigidPivotRotationConstraintPrefab> rigidPivotRotation{};
  DynamicArray<RigidPrismaticJointConstraintPrefab> rigidPrismaticJoint{};
  DynamicArray<RigidSphericalJointConstraintPrefab> rigidSphericalJoint{};
  DynamicArray<DeformableNodePositionConstraintPrefab> deformableNodePosition{};
  DynamicArray<DeformableNodeToRigidConstraintPrefab> deformableNodeToRigid{};
  DynamicArray<DeformableNodeToDeformableNodeConstraintPrefab> deformableNodeToDeformableNode{};

  bool operator==(ConstraintLists const&) const = default;
};

struct PoseControllerPrefab {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicString articulatedActor{};
  DynamicArray<PoseTrackingParams> linkPosTracking{};
  DynamicArray<PoseTrackingParams> linkRotTracking{};
  DynamicArray<PoseTrackingParams> jointTracking{};

  bool operator==(PoseControllerPrefab const&) const = default;
};

struct ScenePrefab;

struct PrefabReference {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicString name{};
  DynamicString path{};
  real scale = 1_r;
  Quaternion rotation{};
  Real3 translation{};

  // Loaded prefab. Runtime state, not serialized.
  PrefabHandle prefab{};

  bool operator==(PrefabReference const&) const = default;
};

struct ScenePrefab {
  std::optional<DynamicString> comment = std::nullopt;
  ActorLists actors{};
  ConstraintLists constraints{};
  DynamicArray<PoseControllerPrefab> controllers{};
  DynamicArray<PrefabReference> prefabs{};
  std::optional<SceneParams> scene = std::nullopt;
  std::optional<DynamicString> sourceFilePath = std::nullopt;
  std::optional<ContactFilter> contactFilter = std::nullopt;
  std::optional<DynamicArray<ContactPairParamsOverrideEntry>> contactPairParamsOverrides =
      std::nullopt;

  bool operator==(ScenePrefab const&) const = default;
};

/***********************************************************************************************
  Prefab Serialization:
*/

MOCHI_API ScenePrefab LoadFromFile(
    std::string_view prefabPath,
    std::string_view rootPath,
    Context* context,
    Error& error);

MOCHI_API ScenePrefab LoadFromJsonString(
    std::string_view json,
    std::string_view rootPath,
    Context* context,
    Error& error);

MOCHI_API void SaveToJsonFile(ScenePrefab const& prefab, std::string_view path, Error& error);

MOCHI_API DynamicString SaveToJsonString(ScenePrefab const& prefab, Error& error);

/***********************************************************************************************
  Prefab Instantiation:
*/

struct PrefabParams {
  DynamicString name{};
  real scale = 1_r;
  Quaternion rotation{};
  Real3 translation{};
  bool applySceneSettings = true;
};

struct AddToSceneResult {
  DynamicArray<Actor*> actors;
  DynamicArray<Constraint*> constraints;

  MOCHI_API DynamicArray<Actor*> Filter(ActorType type) const;
  MOCHI_API DynamicArray<Constraint*> Filter(ConstraintType type) const;
};

MOCHI_API AddToSceneResult
AddToScene(ScenePrefab const& prefab, Scene* scene, PrefabParams const& params, Error& error);

MOCHI_API AddToSceneResult AddToScene(ScenePrefab const& prefab, Scene* scene, Error& error);

MOCHI_API AddToSceneResult AddToScene(
    std::string_view prefabPath,
    std::string_view rootPath,
    Scene* scene,
    PrefabParams const& params,
    Error& error);

MOCHI_API AddToSceneResult
AddToScene(std::string_view prefabPath, std::string_view rootPath, Scene* scene, Error& error);

/***********************************************************************************************
  Prefab Utilities:
*/

MOCHI_API DynamicString GetPrefabFullPath(
    std::string_view inputPath,
    std::string_view rootForRelativePath,
    std::string_view prefabFilePath);

MOCHI_API ScenePrefab ShallowLoadFromFile(std::string_view path, Error& error);

MOCHI_API ScenePrefab ShallowLoadFromJsonString(std::string_view json, Error& error);

MOCHI_API void LoadNestedPrefabs(ScenePrefab& prefab, std::string_view rootPath, Error& error);

MOCHI_API void
LoadShapes(ScenePrefab& prefab, std::string_view rootPath, Context* context, Error& error);

MOCHI_API void
EnsureFullyLoaded(ScenePrefab& prefab, std::string_view rootPath, Context* context, Error& error);

MOCHI_API void ExportScene(
    Scene const* scene,
    std::string_view exportName,
    std::string_view outputDir,
    Error& error);

MOCHI_API void ExportSceneExcluding(
    Scene const* scene,
    std::string_view exportName,
    std::string_view outputDir,
    Span<ActorHandle const> excludeActors,
    Error& error);

MOCHI_API void ExportActor(
    Actor const* actor,
    std::string_view exportName,
    std::string_view outputDir,
    Error& error);

} // namespace mochi::prefab

#include <mochi_physics/utils/mochi_prefab_inl.h>
