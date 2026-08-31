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

#include "mochi_enums.h"
#include "mochi_handle.h"

/********************************************************************************
 IMPORTANT: PLEASE KEEP HEADER INCLUDES TO A MINIMUM.
    If you must include a mochi_core header, then please make sure that it only
    declares the data types (not containing other implementation details).
*********************************************************************************/
#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/contact/contact_params.h>
#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/grid_sdf_params.h>
#include <mochi_core/integration/integration_params.h>
#include <mochi_core/materials/material_params.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/solvers/linear_solver_params.h>
#include <mochi_core/solvers/nonlinear_solver_params.h>
#include <mochi_core/utils/color.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/eval_params.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/verbosity_params.h>

#include <functional>
#include <optional>

namespace mochi {

class Scene;

struct ContactPairParamsOverride {
  std::optional<real> penaltyCoefficient;

  std::optional<real> frictionFalloffVel;

  std::optional<real> viscousFrictionCoefficient;

  std::optional<real> coulombFrictionCoefficient;

  std::optional<real> normalViscousDampingCoefficient;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ContactPairParamsOverride const&) const = default;
#endif
};

struct SolverParams {
  NonLinearSolverParams nonLinearSolver = {};

  LinearSolverParams linearSolver = {};

  IntegrationMethod integrationMethod = IntegrationMethod::Default;

  ExperimentalEvalParams experimentalEval = {};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SolverParams const&) const = default;
#endif
};

struct RecenteringParams {
  bool useRecentering = true;

  real rotationEpsilonDeg = 0_r;

  real translationEpsilon = 0_r;
};

struct BoundarySubsamplingParams {
  real subsamplingDensity = 1_r;

  BoundarySubsamplingStrategy strategy = BoundarySubsamplingStrategy::UniformProbability;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(BoundarySubsamplingParams const&) const = default;
#endif
};

struct ArticulatedShapeInfo {
  Span<TransformRT const> rootFromLinksAtRest;

  Span<DynamicString const> linkNames;

  Span<int const> parents;

  Span<ArticulatedJointType const> jointTypes;

  Span<ArticulatedCycleJoint const> cycles;

  Span<Real3 const> jointAxes;

  Span<ArticulatedDofInfo const> dofInfo;

  Span<TransformRT const> jointFromChildLink;

  Span<TransformRT const> parentLinkFromJoint;

  Span<Real3 const> jointMinLimits;

  Span<Real3 const> jointMaxLimits;

  Span<DynamicString const> jointNames;
};

struct StepInfo {
  Scene* scene = nullptr;

  double timeStepSec = 0.0;
};

struct RigidActorParams {
  DynamicString name;

  DynamicString layer;

  ShapeHandle shape;

  TransformRT worldFromLocal;

  ColliderType colliderType = ColliderType::Auto;

  bool isStatic = false;

  ContactParams contact;

  GridSdfParams sdf;

  bool hasGravity = true;

  std::optional<real> density;

  std::optional<real> mass;

  std::optional<Real3> centerOfMass;

  std::optional<Real6> momentOfInertia;

  ActorBoundaryElementType boundaryElementType = ActorBoundaryElementType::Default;

  std::optional<BoundarySubsamplingParams> boundarySubsampling;

  std::optional<Real3> linearVelocity;

  std::optional<Real3> angularVelocity;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RigidActorParams const&) const = default;
#endif
};

struct SoftActorParams {
  DynamicString name;

  DynamicString layer;

  TransformRT worldFromLocal;

  ShapeHandle shape;

  SoftMaterialParams material = {};

  ContactParams contact;

  bool hasGravity = true;

  bool hasInertia = true;

  bool hasStress = true;

  ActorBoundaryElementType boundaryElementType = ActorBoundaryElementType::Default;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SoftActorParams const&) const = default;
#endif
};

struct ArticulatedJointParams {
  DynamicString name;
  ArticulatedJointType type = ArticulatedJointType::Invalid;
  TransformRT parentLinkFromJoint;
  Real3 axis = {};
  ArticulatedJointFrictionParams friction;
  std::optional<real> inertia;
  std::optional<Real3> minLimit;
  std::optional<Real3> maxLimit;
  real limitStiffness = 100_r;
  real limitDamping = 0_r;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedJointParams const&) const = default;
#endif
};

struct ArticulatedLinkParams {
  DynamicString name;
  int parentLink = -1;
  TransformRT parentJointFromLink;
  ShapeHandle shape;
  DynamicString layer;
  ColliderType colliderType = ColliderType::Auto;
  ContactParams contact;
  bool hasGravity = true;
  std::optional<real> density;
  std::optional<real> mass;
  std::optional<Real3> centerOfMass;
  std::optional<Real6> momentOfInertia;
  ActorBoundaryElementType boundaryElementType = ActorBoundaryElementType::Default;
  std::optional<BoundarySubsamplingParams> boundarySubsampling;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedLinkParams const&) const = default;
#endif
};

struct ArticulatedSkinParams {
  ShapeHandle shape;
  DynamicString layer;
  ContactParams contact;
  ActorBoundaryElementType boundaryElementType = ActorBoundaryElementType::Default;
  std::optional<BoundarySubsamplingParams> boundarySubsampling;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedSkinParams const&) const = default;
#endif
};

struct ArticulatedCycleJointParams {
  int parentLink = -1;
  int childLink = -1;
  TransformRT jointFromChildLink = {};
  real stiffness = 50000_r;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedCycleJointParams const&) const = default;
#endif
};

struct ArticulatedActorParams {
  DynamicString name;
  TransformRT worldFromRoot;
  DynamicArray<ArticulatedCycleJointParams> cycles;
  DynamicArray<ArticulatedJointParams> joints;
  DynamicArray<ArticulatedLinkParams> links;
  std::optional<ArticulatedSkinParams> skin;
  std::optional<DynamicArray<real>> jointVelocities;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedActorParams const&) const = default;
#endif
};

struct SoftSkinnedActorParams {
  ArticulatedActorParams skeletonParams;
  DynamicArray<SoftActorParams> softParams;
  DynamicArray<DynamicString> softAttachLinks;
  bool enableCollidingLinks = false;
  bool hasGravity = false;
  bool hasInertia = false;
  bool hasStress = false;
};

struct ConstraintParams {
  real stiffness = 1e6_r;

  real damping = 0_r;

  real saturation = -1_r;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ConstraintParams const&) const = default;
#endif
};

struct RigidSphericalJointConstraintParams : public ConstraintParams {
  Real3 localPosA{};

  Real3 localPosB{};

  ActorHandle actorA{};

  ActorHandle actorB{};
};

struct RigidPrismaticJointConstraintParams : public ConstraintParams {
  Real3 freeAxis{};

  ActorHandle actorA{};

  ActorHandle actorB{};

  std::optional<real> max{};

  std::optional<real> min{};
};

struct DeformableNodeToDeformableNodeConstraintParams : public ConstraintParams {
  int nodeIndexA = -1;

  int nodeIndexB = -1;

  ActorHandle actorA{};

  ActorHandle actorB{};

  bool findClosest = false;
};

struct RodElementRotationToRigidConstraintParams : public ConstraintParams {
  Real3 refFrameRotVec{};

  ActorHandle rigidActor{};

  ActorHandle rodActor{};

  int elementIndex = 0;
};

struct DeformableNodeToRigidConstraintParams : public ConstraintParams {
  Real3 rigidLocalPos{};

  int deformableNodeIndex = -1;

  ActorHandle rigidActor{};

  ActorHandle deformableActor{};

  bool findClosest = false;

  bool fixToDeformablePos = true;
};

struct JointRotationRangeConstraintParams : public ConstraintParams {
  Real3 refFrameRotVec{};

  Real2 angleRangeX = kMinus2PiPlus2Pi;

  Real2 angleRangeY = kMinus2PiPlus2Pi;

  Real2 angleRangeZ = kMinus2PiPlus2Pi;

  ActorHandle actorA{};

  ActorHandle actorB{};

  bool rangeAroundRest = true;
};

struct RigidPivotPositionConstraintParams : public ConstraintParams {
  Real3 targetPosition{};

  Real3 localPosition{};

  ActorHandle actor{};
};

struct RigidPivotToRigidTargetConstraintParams : public ConstraintParams {
  TransformRT targetTransform{};

  Real3 localPosition{};

  ActorHandle actor{};
};

struct RigidPivotRotationConstraintParams : public ConstraintParams {
  Real3 targetRotation{};

  Real3 localRotation{};

  ActorHandle actor{};
};

struct DeformableNodePositionConstraintParams : public ConstraintParams {
  int nodeIndex = -1;

  Real3 position{};

  ActorHandle actor{};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(DeformableNodePositionConstraintParams const&) const = default;
#endif
};

struct JointRotationTrackingConstraintParams : public ConstraintParams {
  Real3 refFrameRotVec{};

  ActorHandle actorA{};

  ActorHandle actorB{};
};

struct ArticulatedSingleDofTargetConstraintParams : public ConstraintParams {
  ActorHandle actor{};

  int jointIndex = 0;

  int dofIndex = 0;

  real targetValue = 0_r;
};

struct Articulated3dRotationTargetConstraintParams : public ConstraintParams {
  ActorHandle actor{};

  int jointIndex = 0;

  Quaternion target{};
};

struct ArticulatedSingleDofRangeConstraintParams : public ConstraintParams {
  ActorHandle actor{};

  int jointIndex = 0;

  int dofIndex = {};

  real minValue = 0_r;

  real maxValue = 0_r;
};

struct Articulated3dRotationRangeConstraintParams : public ConstraintParams {
  ActorHandle actor{};

  int jointIndex = 0;

  Real3 minValues{};

  Real3 maxValues{};
};

struct ContactPoint {
  ActorHandle actorA = {};

  ActorHandle actorB = {};

  real distance = 0_r;

  Real3 posA = {};

  Real3 posB = {};

  Real3 normal = {};

  Real3 force = {};

  Real3 pointVelocityA = {};

  Real3 pointVelocityB = {};

  int sampleIndex = 0;

  real intWeight = 0_r;

  int elementIndex = -1;

  Real3 parametricCoords = {};
};

struct NodeContactForce {
  int index = 0;

  Real3 force = {};
};

struct SdfDistances {
  Span<int const> sampleIndices;

  Span<Real3 const> worldPositions;

  Span<real const> distances;

  Span<Real3 const> distanceGrads;

  real maxSdfFarDistanceEvaluation = 0_r;
};

struct AsyncStepParams {
  bool useFixedTimeStep = false;

  double fixedTimeStepSeconds = 0.01;

  double dynamicTimeStepMinSeconds = 0.001;

  double dynamicTimeStepMaxSeconds = 0.01;

  std::function<double()> timeStepCallback;
};

struct PerformanceStats {
  double totalStepDurationSec = 0.0;

  double solveStepDurationSec = 0.0;

  double preStepCallbacksDurationSec = 0.0;

  double postStepCallbacksDurationSec = 0.0;

  double recordingStepDurationSec = 0.0;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(PerformanceStats const&) const = default;
#endif
};

struct SolverStats {
  int maxNonLinearIters = 0;

  double residualNorm = 0.0;

  int maxLineSearchIters = 0;

  ConvergenceStatus convergenceStatus = ConvergenceStatus::None;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SolverStats const&) const = default;
#endif
};

struct RecordingParams {
  // ------------------------------------------------------------------------------
  // Per-Actor Information
  // ------------------------------------------------------------------------------

  bool recordActorMeshes = true;

  bool recordActorLocalToGlobalMap = false;

  bool recordActorMassMatrix = false;

  // ------------------------------------------------------------------------------
  // Per-Step Information
  // ------------------------------------------------------------------------------

  bool recordTargetState = true;

  bool recordDynamicActorState = true;

  bool recordStaticActorState = false;

  bool recordContactPoints = false;

  bool recordNodeContactForces = false;

  bool recordSdfDistances = false;

  // ------------------------------------------------------------------------------
  // Utils
  // ------------------------------------------------------------------------------

  static RecordingParams All(bool enabled = true) {
    RecordingParams params;
    params.recordActorMeshes = enabled;
    params.recordActorLocalToGlobalMap = enabled;
    params.recordActorMassMatrix = enabled;
    params.recordTargetState = enabled;
    params.recordDynamicActorState = enabled;
    params.recordStaticActorState = enabled;
    params.recordContactPoints = enabled;
    params.recordNodeContactForces = enabled;
    params.recordSdfDistances = enabled;
    return params;
  }
};

struct PoseTrackingParams {
  real stiffness = 0_r;

  real damping = 0_r;

  real saturation = -1_r;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(PoseTrackingParams const&) const = default;
#endif
};

struct PoseControllerParams {
  PoseControllerParams() = default;

  // Construct with all three tracking arrays pre-sized to numLinks default-constructed
  // PoseTrackingParams (zero gains). Convenient for Get/SetArticulatedPoseControllerParams, which
  // operate on link-indexed arrays of size numLinks.
  explicit PoseControllerParams(int numLinks)
      : linkPosTracking(numLinks), linkRotTracking(numLinks), jointTracking(numLinks) {}

  DynamicArray<PoseTrackingParams> linkPosTracking;

  DynamicArray<PoseTrackingParams> linkRotTracking;

  DynamicArray<PoseTrackingParams> jointTracking;
};

struct PoseConstraintInfo {
  ConstraintHandle handle;

  PoseConstraintType type = PoseConstraintType::Invalid;

  int link = -1;

  int parent = -1;
};

struct DebugDrawSpheres {
  Span<Real3 const> positions;

  Span<real const> radii;

  Span<Color const> colors;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(DebugDrawSpheres const&) const = default;
#endif
};

struct DebugDrawLineVertices {
  Span<Real3 const> positions;

  Span<Color const> colors;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(DebugDrawLineVertices const&) const = default;
#endif
};

struct DebugDrawData {
  DebugDrawLineVertices lineVertices;

  DebugDrawSpheres spheres;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(DebugDrawData const&) const = default;
#endif
};

} // namespace mochi
