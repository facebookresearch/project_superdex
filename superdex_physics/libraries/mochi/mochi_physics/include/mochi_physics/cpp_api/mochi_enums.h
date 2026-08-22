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

/********************************************************************************
 IMPORTANT: Please do not add any #includes to this file.
*********************************************************************************/

namespace mochi {

enum class ActorType {
  None,

  Rigid,

  Soft,

  Articulated,

  Shell,

  Rod,

  Count
};

enum class BoundarySubsamplingStrategy {
  UniformProbability = 0,

  AreaProportional = 1,

  Count = 2
};

enum class ActorBoundaryElementType {
  P1Q1,

  P1Q3,

  P1Q6,

  ExperimentalP1Q7,

  ExperimentalP1Q12,

  ExperimentalP1Q16,

  Count,

  Default = P1Q3,
};

enum class ActorSegmentElementType {
  P1Q1,

  P1Q2,

  P1Q3,

  Count,

  Default = P1Q3,
};

enum class ConstraintType {
  None,

  ArticulatedSingleDofRange,

  Articulated3dRotationRange,

  ArticulatedSingleDofTarget,

  Articulated3dRotationTarget,

  JointRotationRange,

  JointRotationTracking,

  RigidPivotPosition,

  RigidPivotToRigidTarget,

  RigidPivotRotation,

  RigidPrismaticJoint,

  RigidSphericalJoint,

  DeformableNodePosition,

  DeformableNodeToRigid,

  DeformableNodeToDeformableNode,

  RodElementRotationToRigid,

  Count
};

// WARNING: This enum is replicated in the UE Plugin. Changes must be transferred.
enum class PoseConstraintType {
  Joint = 0,

  LinkTranslation = 1,

  LinkRotation = 2,

  Count,

  Invalid = 255,
};

enum class QueryType {
  NodePositions,

  ElementsDeformationGradient,

  SurfaceNodePositions,

  SurfaceNodeNormals,

  VisualNodePositions,

  VisualNodeNormals,

  ContactPoints,

  ElasticEnergy,

  NodeContactForces,

  ConstraintForce,

  ArticulatedControllerForce,

  SdfDistances,

  TotalContactForce,

  Count
};

enum class IncludeNestedActors {
  No,

  Yes,

  Count
};

} // namespace mochi
