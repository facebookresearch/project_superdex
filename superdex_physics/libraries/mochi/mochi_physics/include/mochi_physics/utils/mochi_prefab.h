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

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SceneParams const&) const = default;
#endif
};

struct ActorContactEntry {
  bool enable = true;
  DynamicArray<DynamicString> actors;
  bool includeNestedActors = true;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ActorContactEntry const&) const = default;
#endif
};

struct LayerContactEntry {
  bool enable = true;
  DynamicArray<DynamicString> layers;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(LayerContactEntry const&) const = default;
#endif
};

struct ContactFilter {
  std::optional<DynamicString> comment = std::nullopt;
  std::optional<DynamicArray<ActorContactEntry>> actorContactAsymmetric{};
  std::optional<DynamicArray<ActorContactEntry>> actorContactSymmetric{};
  std::optional<DynamicArray<LayerContactEntry>> layerContactAsymmetric{};
  std::optional<DynamicArray<LayerContactEntry>> layerContactSymmetric{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ContactFilter const&) const = default;
#endif
};

struct ContactPairParamsOverrideEntry {
  DynamicArray<DynamicString> actors;
  ContactPairParamsOverride paramsOverride;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ContactPairParamsOverrideEntry const&) const = default;
#endif
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

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RigidActorPrefab const&) const = default;
#endif
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
  std::optional<DynamicArray<real>> nodeDisplacements;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SoftActorPrefab const&) const = default;
#endif
};

struct RodPose {
  DynamicArray<real> nodeDisplacements{};
  DynamicArray<real> elementFrameAxes{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RodPose const&) const = default;
#endif
};

struct RodActorPrefab : public RodActorParams {
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
  std::optional<RodPose> pose = std::nullopt;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RodActorPrefab const&) const = default;
#endif
};

struct ArticulatedJointPrefab : ArticulatedJointParams {
#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedJointPrefab const&) const = default;
#endif
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

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedLinkPrefab const&) const = default;
#endif
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
  std::optional<DynamicArray<DynamicString>> hiddenLinks;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedSkinPrefab const&) const = default;
#endif
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
  std::optional<DynamicArray<real>> jointPose;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedActorPrefab const&) const = default;
#endif
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

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SoftSkinnedActorPrefab const&) const = default;
#endif
};

struct ActorLists {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicArray<ArticulatedActorPrefab> articulated{};
  DynamicArray<RigidActorPrefab> rigid{};
  DynamicArray<SoftActorPrefab> soft{};
  DynamicArray<SoftSkinnedActorPrefab> softSkinned{};
  DynamicArray<RodActorPrefab> rod{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ActorLists const&) const = default;
#endif
};

struct RigidSphericalJointConstraintPrefab : public RigidSphericalJointConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RigidSphericalJointConstraintPrefab const&) const = default;
#endif
};

struct RigidPrismaticJointConstraintPrefab : public RigidPrismaticJointConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RigidPrismaticJointConstraintPrefab const&) const = default;
#endif
};

struct DeformableNodeToDeformableNodeConstraintPrefab
    : public DeformableNodeToDeformableNodeConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(DeformableNodeToDeformableNodeConstraintPrefab const&) const = default;
#endif
};

struct DeformableNodeToRigidConstraintPrefab : public DeformableNodeToRigidConstraintParams {
  DynamicString rigidActorName{};
  DynamicString deformableActorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(DeformableNodeToRigidConstraintPrefab const&) const = default;
#endif
};

struct JointRotationRangeConstraintPrefab : public JointRotationRangeConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(JointRotationRangeConstraintPrefab const&) const = default;
#endif
};

struct RigidPivotPositionConstraintPrefab : public RigidPivotPositionConstraintParams {
  DynamicString actorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RigidPivotPositionConstraintPrefab const&) const = default;
#endif
};

struct RigidPivotToRigidTargetConstraintPrefab : public RigidPivotToRigidTargetConstraintParams {
  DynamicString actorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RigidPivotToRigidTargetConstraintPrefab const&) const = default;
#endif
};

struct RigidPivotRotationConstraintPrefab : public RigidPivotRotationConstraintParams {
  DynamicString actorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RigidPivotRotationConstraintPrefab const&) const = default;
#endif
};

struct DeformableNodePositionConstraintPrefab : public DeformableNodePositionConstraintParams {
  DynamicString actorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(DeformableNodePositionConstraintPrefab const&) const = default;
#endif
};

struct JointRotationTrackingConstraintPrefab : public JointRotationTrackingConstraintParams {
  DynamicString actorNameA{};
  DynamicString actorNameB{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(JointRotationTrackingConstraintPrefab const&) const = default;
#endif
};

struct ArticulatedSingleDofTargetConstraintPrefab
    : public ArticulatedSingleDofTargetConstraintParams {
  DynamicString actorName;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedSingleDofTargetConstraintPrefab const&) const = default;
#endif
};

struct Articulated3dRotationTargetConstraintPrefab
    : public Articulated3dRotationTargetConstraintParams {
  DynamicString actorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(Articulated3dRotationTargetConstraintPrefab const&) const = default;
#endif
};

struct ArticulatedSingleDofRangeConstraintPrefab
    : public ArticulatedSingleDofRangeConstraintParams {
  DynamicString actorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedSingleDofRangeConstraintPrefab const&) const = default;
#endif
};

struct Articulated3dRotationRangeConstraintPrefab
    : public Articulated3dRotationRangeConstraintParams {
  DynamicString actorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(Articulated3dRotationRangeConstraintPrefab const&) const = default;
#endif
};

struct RodElementRotationConstraintPrefab : public RodElementRotationConstraintParams {
  DynamicString actorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RodElementRotationConstraintPrefab const&) const = default;
#endif
};

struct RodElementRotationToRigidConstraintPrefab
    : public RodElementRotationToRigidConstraintParams {
  DynamicString rigidActorName{};
  DynamicString rodActorName{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RodElementRotationToRigidConstraintPrefab const&) const = default;
#endif
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
  DynamicArray<RodElementRotationConstraintPrefab> rodElementRotation{};
  DynamicArray<RodElementRotationToRigidConstraintPrefab> rodElementRotationToRigid{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ConstraintLists const&) const = default;
#endif
};

struct PoseControllerPrefab {
  std::optional<DynamicString> comment = std::nullopt;
  DynamicString articulatedActor{};
  DynamicArray<PoseTrackingParams> linkPosTracking{};
  DynamicArray<PoseTrackingParams> linkRotTracking{};
  DynamicArray<PoseTrackingParams> jointTracking{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(PoseControllerPrefab const&) const = default;
#endif
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

#if MOCHI_LANGUAGE_CPP20
  bool operator==(PrefabReference const&) const = default;
#endif
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

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ScenePrefab const&) const = default;
#endif
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

MOCHI_API void
ScaleArticulatedActorParams(ArticulatedActorParams& params, real scale, Error& error);

MOCHI_API ScenePrefab ShallowLoadFromFile(std::string_view path, Error& error);

MOCHI_API ScenePrefab ShallowLoadFromJsonString(std::string_view json, Error& error);

MOCHI_API void LoadNestedPrefabs(ScenePrefab& prefab, std::string_view rootPath, Error& error);

MOCHI_API void
LoadShapes(ScenePrefab& prefab, std::string_view rootPath, Context* context, Error& error);

MOCHI_API void
EnsureFullyLoaded(ScenePrefab& prefab, std::string_view rootPath, Context* context, Error& error);

struct ExportParams {
  DynamicArray<ActorHandle> excludeActors{};
  bool exportCurrentPose = false;
};

MOCHI_API void ExportScene(
    Scene const* scene,
    std::string_view exportName,
    std::string_view outputDir,
    ExportParams const& params,
    Error& error);

MOCHI_API void ExportScene(
    Scene const* scene,
    std::string_view exportName,
    std::string_view outputDir,
    Error& error);

MOCHI_API void ExportActor(
    Actor const* actor,
    std::string_view exportName,
    std::string_view outputDir,
    ExportParams const& params,
    Error& error);

MOCHI_API void ExportActor(
    Actor const* actor,
    std::string_view exportName,
    std::string_view outputDir,
    Error& error);

} // namespace mochi::prefab

#include <mochi_physics/utils/mochi_prefab_inl.h>

#if MOCHI_USE_REFLECTION
#include <mochi_physics/utils/mochi_prefab_reflection.generated.h>
#endif
