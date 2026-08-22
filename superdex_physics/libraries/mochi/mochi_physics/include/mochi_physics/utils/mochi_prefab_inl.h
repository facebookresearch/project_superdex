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

#include <mochi_physics/utils/mochi_prefab.h>

#include <mochi_core/utils/reflection.h>

namespace mochi::prefab {

// Convenience overloads of @ref AddToScene that delegate to the canonical overloads using default
// @ref PrefabParams. Declared in mochi_prefab.h.
inline AddToSceneResult AddToScene(ScenePrefab const& prefab, Scene* scene, Error& error) {
  return AddToScene(prefab, scene, PrefabParams{}, error);
}

inline AddToSceneResult
AddToScene(std::string_view prefabPath, std::string_view rootPath, Scene* scene, Error& error) {
  return AddToScene(prefabPath, rootPath, scene, PrefabParams{}, error);
}

} // namespace mochi::prefab

// clang-format off

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::SceneParams)
MOCHI_FIELD_NAME(comment, "_comment") // _ comes first in JSON
MOCHI_FIELD(description)
MOCHI_FIELD(gravity) MOCHI_ATTRIBUTE(Units("m/s^2"))
MOCHI_FIELD(solver)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ActorContactEntry)
MOCHI_FIELD(enable)
MOCHI_FIELD(actors)
MOCHI_FIELD(includeNestedActors) MOCHI_ATTRIBUTE(NoSerializeDefaults)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::LayerContactEntry)
MOCHI_FIELD(enable)
MOCHI_FIELD(layers)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ContactFilter)
MOCHI_FIELD_NAME(comment, "_comment")
MOCHI_FIELD(actorContactAsymmetric)
MOCHI_FIELD(actorContactSymmetric)
MOCHI_FIELD(layerContactAsymmetric)
MOCHI_FIELD(layerContactSymmetric)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::RigidActorPrefab)
MOCHI_BASE_CLASS(mochi::RigidActorParams)
MOCHI_REMOVE_FIELD("worldFromLocal") // Hide the inherited TransformRT field
MOCHI_FIELD_NAME(comment, "_comment") // _ comes first in JSON
MOCHI_REPLACE_FIELD_NAME(shapeFile, "shape") // Replace inherited field
MOCHI_FIELD(shapeRotation)
MOCHI_FIELD(shapeTranslation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_FIELD(scale)
MOCHI_FIELD(rotation)
MOCHI_FIELD(translation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_FIELD_NAME(renderModelFile, "renderModel")
MOCHI_FIELD(renderModelScale)
MOCHI_FIELD(renderModelRotation)
MOCHI_FIELD(renderModelTranslation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::SoftActorPrefab)
MOCHI_BASE_CLASS(mochi::SoftActorParams)
MOCHI_REMOVE_FIELD("worldFromLocal") // Hide the inherited TransformRT field
MOCHI_FIELD_NAME(comment, "_comment") // _ comes first in JSON
MOCHI_REPLACE_FIELD_NAME(shapeFile, "shape") // Replace inherited field
MOCHI_FIELD(colliderType)
MOCHI_FIELD(sdf)
MOCHI_FIELD_NAME(flowFile, "flow")
MOCHI_FIELD(useRecentering)
MOCHI_FIELD(shapeRotation)
MOCHI_FIELD(shapeTranslation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_FIELD(scale)
MOCHI_FIELD(rotation)
MOCHI_FIELD(translation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_FIELD_NAME(renderModelFile, "renderModel")
MOCHI_FIELD(renderModelScale)
MOCHI_FIELD(renderModelRotation)
MOCHI_FIELD(renderModelTranslation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ArticulatedJointPrefab)
MOCHI_BASE_CLASS(mochi::ArticulatedJointParams)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ArticulatedLinkPrefab)
MOCHI_BASE_CLASS(mochi::ArticulatedLinkParams)
MOCHI_REMOVE_FIELD("shape") // The prefab serializes a file path instead
MOCHI_FIELD_NAME(shapeFile, "shape")
MOCHI_FIELD(shapeScale)
MOCHI_FIELD(shapeRotation)
MOCHI_FIELD(shapeTranslation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_FIELD_NAME(renderModelFile, "renderModel")
MOCHI_FIELD(renderModelScale)
MOCHI_FIELD(renderModelRotation)
MOCHI_FIELD(renderModelTranslation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ArticulatedSkinPrefab)
MOCHI_BASE_CLASS(mochi::ArticulatedSkinParams)
MOCHI_REMOVE_FIELD("shape") // The prefab serializes a file path instead
MOCHI_FIELD_NAME(shapeFile, "shape")

// TODO[T265138451]: Add support for baked-in transform
// MOCHI_FIELD(shapeScale)
// MOCHI_FIELD(shapeRotation)
// MOCHI_FIELD(shapeTranslation) MOCHI_ATTRIBUTE(Units("m"))

MOCHI_FIELD_NAME(renderModelFile, "renderModel")
MOCHI_FIELD(renderModelScale)
MOCHI_FIELD(renderModelRotation)
MOCHI_FIELD(renderModelTranslation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ArticulatedActorPrefab)
MOCHI_FIELD_NAME(comment, "_comment") // _ comes first in JSON
MOCHI_FIELD(name)
MOCHI_FIELD(scale)
MOCHI_FIELD(rotation)
MOCHI_FIELD(translation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_FIELD(cycles)
MOCHI_FIELD(joints)
MOCHI_FIELD(links)
MOCHI_FIELD(skin)
MOCHI_FIELD(jointVelocities)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::SoftSkinnedActorPrefab)
MOCHI_FIELD_NAME(comment, "_comment")
MOCHI_FIELD(skeletonParams)
MOCHI_FIELD(softParams)
MOCHI_FIELD(softAttachLinks)
MOCHI_FIELD(enableCollidingLinks)
MOCHI_FIELD(hasGravity)
MOCHI_FIELD(hasInertia)
MOCHI_FIELD(hasStress)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ActorLists)
MOCHI_FIELD_NAME(comment, "_comment") // _ comes first in JSON
MOCHI_FIELD(articulated)
MOCHI_FIELD(rigid)
MOCHI_FIELD(soft)
MOCHI_FIELD(softSkinned)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::RigidSphericalJointConstraintPrefab)
MOCHI_BASE_CLASS(mochi::RigidSphericalJointConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorNameA, "actorA")
MOCHI_REPLACE_FIELD_NAME(actorNameB, "actorB")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::RigidPrismaticJointConstraintPrefab)
MOCHI_BASE_CLASS(mochi::RigidPrismaticJointConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorNameA, "actorA")
MOCHI_REPLACE_FIELD_NAME(actorNameB, "actorB")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::DeformableNodeToDeformableNodeConstraintPrefab)
MOCHI_BASE_CLASS(mochi::DeformableNodeToDeformableNodeConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorNameA, "actorA")
MOCHI_REPLACE_FIELD_NAME(actorNameB, "actorB")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::DeformableNodeToRigidConstraintPrefab)
MOCHI_BASE_CLASS(mochi::DeformableNodeToRigidConstraintParams)
MOCHI_REPLACE_FIELD_NAME(rigidActorName, "rigidActor")
MOCHI_REPLACE_FIELD_NAME(deformableActorName, "deformableActor")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::JointRotationRangeConstraintPrefab)
MOCHI_BASE_CLASS(mochi::JointRotationRangeConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorNameA, "actorA")
MOCHI_REPLACE_FIELD_NAME(actorNameB, "actorB")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::RigidPivotPositionConstraintPrefab)
MOCHI_BASE_CLASS(mochi::RigidPivotPositionConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorName, "actor")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::RigidPivotToRigidTargetConstraintPrefab)
MOCHI_BASE_CLASS(mochi::RigidPivotToRigidTargetConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorName, "actor")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::RigidPivotRotationConstraintPrefab)
MOCHI_BASE_CLASS(mochi::RigidPivotRotationConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorName, "actor")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::DeformableNodePositionConstraintPrefab)
MOCHI_BASE_CLASS(mochi::DeformableNodePositionConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorName, "actor")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::JointRotationTrackingConstraintPrefab)
MOCHI_BASE_CLASS(mochi::JointRotationTrackingConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorNameA, "actorA")
MOCHI_REPLACE_FIELD_NAME(actorNameB, "actorB")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ArticulatedSingleDofTargetConstraintPrefab)
MOCHI_BASE_CLASS(mochi::ArticulatedSingleDofTargetConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorName, "actor")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::Articulated3dRotationTargetConstraintPrefab)
MOCHI_BASE_CLASS(mochi::Articulated3dRotationTargetConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorName, "actor")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ArticulatedSingleDofRangeConstraintPrefab)
MOCHI_BASE_CLASS(mochi::ArticulatedSingleDofRangeConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorName, "actor")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::Articulated3dRotationRangeConstraintPrefab)
MOCHI_BASE_CLASS(mochi::Articulated3dRotationRangeConstraintParams)
MOCHI_REPLACE_FIELD_NAME(actorName, "actor")
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ConstraintLists)
MOCHI_FIELD_NAME(comment, "_comment") // _ comes first in JSON
MOCHI_FIELD(rigidSphericalJoint)
MOCHI_FIELD(rigidPrismaticJoint)
MOCHI_FIELD(deformableNodeToDeformableNode)
MOCHI_FIELD(deformableNodeToRigid)
MOCHI_FIELD(jointRotationRange)
MOCHI_FIELD(rigidPivotPosition)
MOCHI_FIELD(rigidPivotToRigidTarget)
MOCHI_FIELD(rigidPivotRotation)
MOCHI_FIELD(deformableNodePosition)
MOCHI_FIELD(jointRotationTracking)
MOCHI_FIELD(articulatedSingleDofTarget)
MOCHI_FIELD(articulated3dRotationTarget)
MOCHI_FIELD(articulatedSingleDofRange)
MOCHI_FIELD(articulated3dRotationRange)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::PoseControllerPrefab)
MOCHI_FIELD_NAME(comment, "_comment") // _ comes first in JSON
MOCHI_FIELD(articulatedActor)
MOCHI_FIELD(linkPosTracking)
MOCHI_FIELD(linkRotTracking)
MOCHI_FIELD(jointTracking)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::PrefabReference)
MOCHI_FIELD_NAME(comment, "_comment") // _ comes first in JSON
MOCHI_FIELD(name)
MOCHI_FIELD(path)
MOCHI_FIELD(scale)
MOCHI_FIELD(rotation)
MOCHI_FIELD(translation) MOCHI_ATTRIBUTE(Units("m"))
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(mochi::prefab::ScenePrefab)
MOCHI_FIELD_NAME(comment, "_comment") // _ comes first in JSON
MOCHI_FIELD(actors) MOCHI_ATTRIBUTE(NoSerializeDefaults)
MOCHI_FIELD(constraints) MOCHI_ATTRIBUTE(NoSerializeDefaults)
MOCHI_FIELD(controllers) MOCHI_ATTRIBUTE(NoSerializeDefaults)
MOCHI_FIELD(prefabs) MOCHI_ATTRIBUTE(NoSerializeDefaults)
MOCHI_FIELD(scene) MOCHI_ATTRIBUTE(NoSerializeDefaults)
MOCHI_FIELD(contactFilter) // No NoSerializeDefaults - we want enable field always serialized
MOCHI_STRUCT_END_EX()

    // clang-format on
