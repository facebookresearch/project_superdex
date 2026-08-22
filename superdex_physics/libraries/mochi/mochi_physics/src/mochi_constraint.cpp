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

#include "mochi_constraint.h"

#include "mochi_articulated_body.h"
#include "mochi_ecs_utils.h"
#include "mochi_group.h"
#include "mochi_query.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"
#include "mochi_soft.h"

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/utils/constants.h>

#include <memory>
#include <vector>

using namespace mochi;

#define MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error)                        \
  MOCHI_ERROR_IF_NOT(                                                                 \
      ((params).stiffness >= 0_r && IsFinite((params).stiffness)),                    \
      error,                                                                          \
      "Constraint stiffness must be non-negative and finite.");                       \
  MOCHI_ERROR_IF_NOT(                                                                 \
      ((params).damping >= 0_r && IsFinite((params).damping)),                        \
      error,                                                                          \
      "Constraint damping must be non-negative and finite.");                         \
  MOCHI_ERROR_IF_NOT(                                                                 \
      IsFinite((params).saturation), error, "Constraint saturation must be finite."); \
  MOCHI_ERROR_IF(                                                                     \
      (params).saturation == 0_r,                                                     \
      error,                                                                          \
      "Constraint saturation must not be zero. Use any negative value "               \
      "to disable saturation or a positive value to enable it.");                     \
  MOCHI_ERROR_RETURN(error);

#define MOCHI_VALIDATE_DEFORMABLE_ACTOR(reg, e, error)              \
  MOCHI_ERROR_IF(                                                   \
      !((reg).any_of<TagSoftActor, TagShellActor, TagRodActor>(e)), \
      error,                                                        \
      "Invalid actor type. Expected a soft, shell, or rod actor.");

static int DofsPerNode(entt::registry const& reg, entt::entity e) {
  if (reg.any_of<TagSoftActor, TagShellActor>(e)) {
    // These actors only have displacement DoFs
    return 3;
  } else if (reg.all_of<TagRodActor>(e)) {
    // Rod actors have an additional twist angle per node
    return 4;
  } else {
    // This branch should never become accessible via the user-facing API, so it asserts instead of
    // error-ing.
    MOCHI_ASSERT(false, "Expected a soft, shell, or rod actor");
    return 0;
  }
}

entt::entity mochi::TryGetParentArticulatedActor(
    ecs::PartialRegistry<CArticulatedEntity const> reg,
    entt::entity actor) {
  if (auto const* articulatedEntity = reg.try_get<CArticulatedEntity const>(actor)) {
    return articulatedEntity->entity;
  }
  return entt::null;
}

void mochi::InitConstraint_RigidSphericalJoint(
    entt::registry& reg,
    entt::entity e,
    RigidSphericalJointConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(IsFinite(params.localPosA), error, "Local position A must be finite.");
  MOCHI_ERROR_IF_NOT(IsFinite(params.localPosB), error, "Local position B must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entities
  entt::entity objA = GetEntity(reg, ActorHandle(params.actorA), error);
  entt::entity objB = GetEntity(reg, ActorHandle(params.actorB), error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(objA == objB, error, "Constrained objects cannot be the same");
  MOCHI_ERROR_RETURN(error);

  // Make sure they are rigid bodies
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(objA), error, "Invalid actor A type");
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(objB), error, "Invalid actor B type");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {objA, objB};
  std::vector<std::vector<int>> actorDofs = {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::RigidSphericalJoint,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(2),
      params);
  info.name = "RigidSphericalJointConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::RigidSphericalJoint>>(e);
  data.posLocalA =
      params.localPosA - ToReal3(reg.get<CRigidBodyInertia>(objA).GetCenterOfMassLocal());
  data.posLocalB =
      params.localPosB - ToReal3(reg.get<CRigidBodyInertia>(objB).GetCenterOfMassLocal());
}

void mochi::InitConstraint_RigidPrismaticJoint(
    entt::registry& reg,
    entt::entity e,
    RigidPrismaticJointConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF(
      !IsFinite(params.freeAxis) || NearEqual(Norm(params.freeAxis), 0_r),
      error,
      "Free axis must be non-zero and finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  if (params.max && params.min) {
    MOCHI_ERROR_IF(
        params.max.value() < params.min.value(), error, "Wrong limits for prismatic joint");
    MOCHI_ERROR_RETURN(error);
  }

  // Get actor entities
  entt::entity objA = GetEntity(reg, ActorHandle(params.actorA), error);
  entt::entity objB = GetEntity(reg, ActorHandle(params.actorB), error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(objA == objB, error, "Constrained objects cannot be the same");
  MOCHI_ERROR_RETURN(error);

  // Make sure they are rigid bodies
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(objA), error, "Invalid actor A type");
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(objB), error, "Invalid actor B type");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {objA, objB};
  std::vector<std::vector<int>> actorDofs = {{0, 1, 2, 3, 4, 5}, {0, 1, 2}};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::RigidPrismaticJoint,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(2),
      params);
  info.name = "RigidPrismaticJointConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::RigidPrismaticJoint>>(e);
  // Compute local axis
  auto const& stateA = reg.get<CRigidState<TimeStep::Current> const>(objA).value;
  auto localAxis = stateA.GetRotation().GetConjugate() * params.freeAxis;
  localAxis = Normalize(localAxis);
  // Compute local frame
  Real3 axisZ = {0_r, 0_r, 1_r};
  Real3 axis = Cross(localAxis, axisZ);
  real normSqr = NormSqr(axis);
  if (normSqr > 1e-6_r) {
    real const angle = ACos(Clamp(Dot(localAxis, axisZ), -1_r, 1_r));
    axis *= angle / Sqrt(normSqr);
  } else if (Dot(localAxis, axisZ) < 0_r) {
    // localAxis ≈ -Z: rotate π around X (any perpendicular axis works)
    axis = {kPI, 0_r, 0_r};
  }
  data.localFrame = Quaternion::FromRotationVector(axis);
  // Compute reference translation
  auto const& stateB = reg.get<CRigidState<TimeStep::Current> const>(objB).value;
  Real3 trel =
      stateA.GetRotation().GetConjugate() * (stateB.GetTranslation() - stateA.GetTranslation());
  data.tRef = data.localFrame * trel;
  // Add limits (optional)
  data.min = params.min;
  data.max = params.max;
}

static int
GetNodeOnClosestFace(entt::registry const& reg, entt::entity e, Real3 const& point, Error& error) {
  MOCHI_ERROR_IF(
      !reg.all_of<TagSoftActor>(e), error, "Find closest only supported for soft actors.");
  MOCHI_ERROR_RETURN(error, {});
  auto const& mesh = reg.get<CTetrahedralMesh>(e).mesh;
  auto const& softTrans = reg.get<CRootTransform const>(e).worldFromLocal;
  auto nodes = mesh->GetNodeCoordinates();
  auto faces = mesh->GetBoundaryFacesConnectivity();
  auto pointMeshLocal = softTrans.TransformPointInverse(point);
  int nearestFace = mochi::ClosestFacePointTriangularMesh(pointMeshLocal, nodes, faces);
  return faces[nearestFace][0];
}

// This function assumes that displacement DoFs are the first 3 DoFs of each node
static std::vector<int> NodeIndexToDisplacementDofs(int nodeIndex, int dofsPerNode) {
  auto offset = nodeIndex * dofsPerNode;
  return std::vector<int>{offset + 0, offset + 1, offset + 2};
}

static Real3
GetNodeCoordinates(entt::registry const& reg, entt::entity e, int index, Error& error) {
  auto allCoords = [&]() {
    if (auto const* tetMesh = reg.try_get<CTetrahedralMesh const>(e)) {
      // Soft actor
      return tetMesh->mesh->GetNodeCoordinates();
    } else if (auto const* surfMesh = reg.try_get<CSurfaceMesh const>(e)) {
      // Shell actor
      MOCHI_ASSERT_VERBOSE(reg.all_of<TagShellActor>(e), "Expected a shell actor");
      return surfMesh->mesh->GetNodeCoordinates();
    } else if (auto const* polylineMesh = reg.try_get<CPolylineMesh const>(e)) {
      // Rod actor
      MOCHI_ASSERT_VERBOSE(reg.all_of<TagRodActor>(e), "Expected a rod actor");
      return polylineMesh->nodes;
    } else {
      MOCHI_ERROR_SET(error, "Expected a soft, shell, or rod actor");
      return Span<Real3 const>{};
    }
  }();
  MOCHI_ERROR_IF((index < 0) || (index >= isize(allCoords)), error, "Node index out-of-bounds");
  MOCHI_ERROR_RETURN(error, {});
  return allCoords[index];
}

void mochi::InitConstraint_DeformableNodeToDeformableNode(
    entt::registry& reg,
    entt::entity e,
    DeformableNodeToDeformableNodeConstraintParams const& params,
    Error& error) {
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entities
  entt::entity objA = GetEntity(reg, ActorHandle(params.actorA), error);
  entt::entity objB = GetEntity(reg, ActorHandle(params.actorB), error);
  MOCHI_ERROR_RETURN(error);

  // Make sure they are deformable bodies
  MOCHI_VALIDATE_DEFORMABLE_ACTOR(reg, objA, error);
  MOCHI_VALIDATE_DEFORMABLE_ACTOR(reg, objB, error);
  MOCHI_ERROR_RETURN(error);

  // If requested, define constraint based on closest nodes to the mid-point of the transforms
  DeformableNodeToDeformableNodeConstraintParams paramsIn = params;
  if (params.findClosest) {
    auto const* softTransA = &reg.get<CRootTransform>(objA).worldFromLocal;
    auto const* softTransB = &reg.get<CRootTransform>(objB).worldFromLocal;
    Real3 midPoint = 0.5_r * (softTransA->GetTranslation() + softTransB->GetTranslation());
    paramsIn.nodeIndexA = GetNodeOnClosestFace(reg, objA, midPoint, error);
    paramsIn.nodeIndexB = GetNodeOnClosestFace(reg, objB, midPoint, error);
    MOCHI_ERROR_RETURN(error);
  }

  // Get the node coordinates
  Real3 restA = GetNodeCoordinates(reg, objA, paramsIn.nodeIndexA, error);
  Real3 restB = GetNodeCoordinates(reg, objB, paramsIn.nodeIndexB, error);
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {objA, objB};
  int const dofsPerNodeA = DofsPerNode(reg, objA);
  int const dofsPerNodeB = DofsPerNode(reg, objB);
  std::vector<std::vector<int>> actorDofs = {
      NodeIndexToDisplacementDofs(paramsIn.nodeIndexA, dofsPerNodeA),
      NodeIndexToDisplacementDofs(paramsIn.nodeIndexB, dofsPerNodeB)};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::DeformableNodeToDeformableNode,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(2),
      paramsIn);
  info.name = "DeformableNodeToDeformableNodeConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::DeformableNodeToDeformableNode>>(e);
  data.restA = restA;
  data.restB = restB;
}

// Helper to get displacement values from a deformable actor, which may be a rod (CRodPose)
// or a soft/shell actor (CDisplacementSlice).
template <TimeStep kTimeStep, typename RegT>
static Real3 GetDeformableDisplacement(RegT const& reg, entt::entity actor, Span<int const> dofs) {
  if (auto const* rodPose = reg.template try_get<CRodPose<kTimeStep> const>(actor)) {
    auto const& d = rodPose->value.displacements;
    return Real3{d[dofs[0]], d[dofs[1]], d[dofs[2]]};
  }
  auto const& d = reg.template get<CDisplacementSlice<real, kTimeStep> const>(actor).value;
  return Real3{d[dofs[0]], d[dofs[1]], d[dofs[2]]};
}

void mochi::InitConstraint_DeformableNodeToRigid(
    entt::registry& reg,
    entt::entity e,
    DeformableNodeToRigidConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(IsFinite(params.rigidLocalPos), error, "Rigid local position must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entities
  entt::entity objRigid = GetEntity(reg, ActorHandle(params.rigidActor), error);
  entt::entity objDeformable = GetEntity(reg, ActorHandle(params.deformableActor), error);
  MOCHI_ERROR_RETURN(error);

  // Make sure they are rigid and deformable bodies
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(objRigid), error, "Invalid rigid actor type");
  MOCHI_VALIDATE_DEFORMABLE_ACTOR(reg, objDeformable, error);
  MOCHI_ERROR_RETURN(error);

  TransformRT const& rbState = reg.get<CRigidState<TimeStep::Current> const>(objRigid).value;
  auto rbLocalCom = reg.get<CRigidBodyInertia const>(objRigid).GetCenterOfMassLocal();

  // If requested, define constraint based on the closest node to the rigid actor's point
  DeformableNodeToRigidConstraintParams paramsIn = params;
  if (params.findClosest) {
    Real3 point = rbState.TransformPoint(params.rigidLocalPos - ToReal3(rbLocalCom));
    paramsIn.deformableNodeIndex = GetNodeOnClosestFace(reg, objDeformable, point, error);
    MOCHI_ERROR_RETURN(error);
  }

  // Get the node coordinates
  Real3 restDeformable =
      GetNodeCoordinates(reg, objDeformable, paramsIn.deformableNodeIndex, error);
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {objRigid, objDeformable};
  int const dofsPerNode = DofsPerNode(reg, objDeformable);
  std::vector<std::vector<int>> actorDofs = {
      {0, 1, 2, 3, 4, 5}, NodeIndexToDisplacementDofs(paramsIn.deformableNodeIndex, dofsPerNode)};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::DeformableNodeToRigid,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(2),
      paramsIn);
  info.name = "DeformableNodeToRigidConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::DeformableNodeToRigid>>(e);
  data.restDeformable = restDeformable;
  if (params.fixToDeformablePos) {
    auto const dofs = NodeIndexToDisplacementDofs(paramsIn.deformableNodeIndex, dofsPerNode);
    Real3 dispDeformable =
        GetDeformableDisplacement<TimeStep::Current>(reg, objDeformable, MakeConstSpan(dofs));
    auto const& deformableTrans = reg.get<CRootTransform const>(objDeformable).worldFromLocal;
    auto posWorldDeformable = deformableTrans.TransformPoint(data.restDeformable + dispDeformable);
    data.posLocalRigid = rbState.TransformPointInverse(posWorldDeformable);
  } else {
    data.posLocalRigid = params.rigidLocalPos - ToReal3(rbLocalCom);
  }
}

void mochi::InitConstraint_JointRotationRange(
    entt::registry& reg,
    entt::entity e,
    JointRotationRangeConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.refFrameRotVec), error, "Reference frame rotation vector must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entities
  entt::entity objA = GetEntity(reg, ActorHandle(params.actorA), error);
  entt::entity objB = GetEntity(reg, ActorHandle(params.actorB), error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(objA == objB, error, "Constrained objects cannot be the same");
  MOCHI_ERROR_RETURN(error);

  // Make sure they are rigid bodies
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(objA), error, "Invalid actor A type");
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(objB), error, "Invalid actor B type");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {objA, objB};
  std::vector<std::vector<int>> actorDofs = {{3, 4, 5}, {3, 4, 5}};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::JointRotationRange,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(2),
      params);
  info.name = "JointRotationRangeConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::JointRotationRange>>(e);
  // Compute joint frame in rest configuration
  auto const& rotA = reg.get<CRigidState<TimeStep::Current> const>(objA).value.GetRotation();
  Quaternion qR = Quaternion::FromRotationVector(params.refFrameRotVec);
  data.q0 = Normalize(rotA.GetConjugate() * qR);
  // Set reference rotation
  if (params.rangeAroundRest) {
    auto const& rotB = reg.get<CRigidState<TimeStep::Current> const>(objB).value.GetRotation();
    data.qr = Normalize(RelativeRotation_Reference(rotA, rotB, data.q0));
  } else {
    data.qr = Quaternion::Identity();
  }
  // Set ranges
  data.minRotVec = {params.angleRangeX[0], params.angleRangeY[0], params.angleRangeZ[0]};
  data.maxRotVec = {params.angleRangeX[1], params.angleRangeY[1], params.angleRangeZ[1]};
}

void mochi::InitConstraint_JointRotationTracking(
    entt::registry& reg,
    entt::entity e,
    JointRotationTrackingConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.refFrameRotVec), error, "Reference frame rotation vector must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entities
  entt::entity objA = GetEntity(reg, ActorHandle(params.actorA), error);
  entt::entity objB = GetEntity(reg, ActorHandle(params.actorB), error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(objA == objB, error, "Constrained objects cannot be the same");
  MOCHI_ERROR_RETURN(error);

  // Make sure they are rigid bodies
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(objA), error, "Invalid actor A type");
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(objB), error, "Invalid actor B type");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {objA, objB};
  std::vector<std::vector<int>> actorDofs = {{3, 4, 5}, {3, 4, 5}};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::JointRotationTracking,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(2),
      params);
  info.name = "JointRotationTrackingConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::JointRotationTracking>>(e);
  // Compute joint frame in rest configuration
  auto const& rotA = reg.get<CRigidState<TimeStep::Current> const>(objA).value.GetRotation();
  auto const& rotB = reg.get<CRigidState<TimeStep::Current> const>(objB).value.GetRotation();
  Quaternion qR = Quaternion::FromRotationVector(params.refFrameRotVec);
  data.q0 = Normalize(rotA.GetConjugate() * qR);

  // Create targets. Initialize the target rotation as the current relative rotation
  auto& target = reg.emplace<CConstraintTarget<Quaternion, TimeStep::Current>>(e);
  target.value = Normalize(rotA.GetConjugate() * rotB);
  auto& targetOld = reg.emplace<CConstraintTarget<Quaternion, TimeStep::StageStart>>(e);
  targetOld.value = target.value;
}

void mochi::InitConstraint_RodElementRotationToRigid(
    entt::registry& reg,
    entt::entity e,
    RodElementRotationToRigidConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.refFrameRotVec), error, "Reference frame rotation vector must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entities
  entt::entity rigidActor = GetEntity(reg, ActorHandle(params.rigidActor), error);
  entt::entity rodActor = GetEntity(reg, ActorHandle(params.rodActor), error);
  MOCHI_ERROR_RETURN(error);

  // Make sure they are rigid and rod actors
  MOCHI_ERROR_IF(!reg.all_of<TagRigidActor>(rigidActor), error, "Invalid rigid actor type");
  MOCHI_ERROR_IF(!reg.all_of<TagRodActor>(rodActor), error, "Invalid rod actor type");
  MOCHI_ERROR_RETURN(error);

  // Validate element index
  auto const& polylineMesh = reg.get<CPolylineMesh const>(rodActor);
  int const numElements = polylineMesh.NumElements();
  MOCHI_ERROR_IF(
      params.elementIndex < 0 || params.elementIndex >= numElements,
      error,
      "Element index out of bounds");
  MOCHI_ERROR_RETURN(error);

  // Get the two nodes of the element
  Int2 const en = polylineMesh.ElementNodes(params.elementIndex);
  int const node0 = en[0];
  int const node1 = en[1];

  // Create constraint info
  // Rigid: rotation DoFs (3, 4, 5)
  // Rod: DoFs for both nodes of the element (8 DoFs total: 4 per node)
  std::vector<entt::entity> actors = {rigidActor, rodActor};
  std::vector<std::vector<int>> actorDofs = {
      {3, 4, 5}, // Rigid body rotation DoFs
      {node0 * 4 + 0,
       node0 * 4 + 1,
       node0 * 4 + 2,
       node0 * 4 + 3, // Node 0: x, y, z, twist
       node1 * 4 + 0,
       node1 * 4 + 1,
       node1 * 4 + 2,
       node1 * 4 + 3} // Node 1: x, y, z, twist
  };
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::RodElementRotationToRigid,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(2),
      params);
  info.name = "RodElementRotationToRigidConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::RodElementRotationToRigid>>(e);
  data.elementIndex = params.elementIndex;
  // Compute joint frame in rest configuration
  auto const& rotRigid =
      reg.get<CRigidState<TimeStep::Current> const>(rigidActor).value.GetRotation();
  Quaternion qR = Quaternion::FromRotationVector(params.refFrameRotVec);
  data.q0 = Normalize(rotRigid.GetConjugate() * qR);

  // Get the rotation of the rod element in the rest configuration
  VMatrix3x3r rotRodMat{};
  Int2 const initEn = polylineMesh.ElementNodes(params.elementIndex);
  Vec4r const X0 = ToSimd(polylineMesh.nodes[initEn[0]]);
  Vec4r const X1 = ToSimd(polylineMesh.nodes[initEn[1]]);
  auto const& currPose = reg.get<CRodPose<TimeStep::Current> const>(rodActor);
  Vec4r const frameAxis = Normalize<3>(ToSimd(currPose.value.frameAxes[params.elementIndex]));
  NdArray<real, 2 * fem::kNumRodFields> initDofsRod MOCHI_NO_INIT;
  Store(
      &(initDofsRod[0]),
      Load<Vec4r>(&(currPose.value.displacements[fem::kNumRodFields * initEn[0]])));
  Store(
      &(initDofsRod[fem::kNumRodFields]),
      Load<Vec4r>(&(currPose.value.displacements[fem::kNumRodFields * initEn[1]])));
  fem::ComputeRodElementRotationLocal(
      X0, X1, frameAxis, MakeConstSpan(initDofsRod), rotRodMat, nullptr);

  // Apply the world-from-local rotation to the rod element rotation
  auto const& worldFromLocalRod = reg.get<CRootTransform const>(rodActor).worldFromLocal;
  auto rotRodLocal = Normalize(QuaternionFromMatrix(rotRodMat));
  auto rotRod = worldFromLocalRod.GetRotation() * rotRodLocal;

  // Create targets. Initialize the target rotation as the current relative rotation
  auto& target = reg.emplace<CConstraintTarget<Quaternion, TimeStep::Current>>(e);
  target.value = Normalize(rotRigid.GetConjugate() * rotRod);
  auto& targetOld = reg.emplace<CConstraintTarget<Quaternion, TimeStep::StageStart>>(e);
  targetOld.value = target.value;
}

void mochi::InitConstraint_RigidPivotPosition(
    entt::registry& reg,
    entt::entity e,
    RigidPivotPositionConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(IsFinite(params.targetPosition), error, "Target position must be finite.");
  MOCHI_ERROR_IF_NOT(IsFinite(params.localPosition), error, "Local position must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  ActorHandle handle(params.actor);
  auto actore = GetEntity(reg, handle, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !reg.all_of<CRigidState<TimeStep::Current>>(actore),
      error,
      "Invalid actor type for this handle");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {actore};
  std::vector<std::vector<int>> actorDofs = {{0, 1, 2, 3, 4, 5}};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::RigidPivotPosition,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(1),
      params);
  info.name = "RigidPivotPositionConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::RigidPivotPosition>>(e);
  data.posLocal = params.localPosition;

  // Create targets
  auto& target = reg.emplace<CConstraintTarget<Real3, TimeStep::Current>>(e);
  target.value = params.targetPosition;
  auto& targetOld = reg.emplace<CConstraintTarget<Real3, TimeStep::StageStart>>(e);
  targetOld.value = target.value;
}

void mochi::InitConstraint_RigidPivotToRigidTarget(
    entt::registry& reg,
    entt::entity e,
    RigidPivotToRigidTargetConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(IsFinite(params.localPosition), error, "Local position must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  ActorHandle handle(params.actor);
  auto actore = GetEntity(reg, handle, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !reg.all_of<CRigidState<TimeStep::Current>>(actore),
      error,
      "Invalid actor type for this handle");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {actore};
  std::vector<std::vector<int>> actorDofs = {{0, 1, 2, 3, 4, 5}};
  std::vector<std::vector<int>> actorTargets = {{0, 1, 2, 3, 4, 5}};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::RigidPivotToRigidTarget,
      std::move(actors),
      std::move(actorDofs),
      std::move(actorTargets),
      params);
  info.name = "RigidPivotToRigidTargetConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::RigidPivotToRigidTarget>>(e);
  data.posLocal = params.localPosition;

  // Create targets
  auto& target = reg.emplace<CConstraintTarget<TransformRT, TimeStep::Current>>(e);
  target.value = params.targetTransform;
  auto& targetOld = reg.emplace<CConstraintTarget<TransformRT, TimeStep::StageStart>>(e);
  targetOld.value = target.value;
}

void mochi::InitConstraint_RigidPivotRotation(
    entt::registry& reg,
    entt::entity e,
    RigidPivotRotationConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(IsFinite(params.targetRotation), error, "Target rotation must be finite.");
  MOCHI_ERROR_IF_NOT(IsFinite(params.localRotation), error, "Local rotation must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  ActorHandle handle(params.actor);
  auto actore = GetEntity(reg, handle, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !reg.all_of<CRigidState<TimeStep::Current>>(actore),
      error,
      "Invalid actor type for this handle");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {actore};
  std::vector<std::vector<int>> actorDofs = {{3, 4, 5}};
  std::vector<std::vector<int>> actorTargets = {{3, 4, 5}};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::RigidPivotRotation,
      std::move(actors),
      std::move(actorDofs),
      std::move(actorTargets),
      params);
  info.name = "RigidPivotRotationConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::RigidPivotRotation>>(e);
  data.rotLocal = Quaternion::FromRotationVector(params.localRotation);

  // Create targets
  auto& target = reg.emplace<CConstraintTarget<Quaternion, TimeStep::Current>>(e);
  target.value = Quaternion::FromRotationVector(params.targetRotation);
  auto& targetOld = reg.emplace<CConstraintTarget<Quaternion, TimeStep::StageStart>>(e);
  targetOld.value = target.value;
}

void mochi::InitConstraint_DeformableNodePosition(
    entt::registry& reg,
    entt::entity e,
    DeformableNodePositionConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(IsFinite(params.position), error, "Deformable node position must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entity
  entt::entity obj = GetEntity(reg, ActorHandle(params.actor), error);
  MOCHI_ERROR_RETURN(error);

  // Make sure it is a deformable actor
  MOCHI_VALIDATE_DEFORMABLE_ACTOR(reg, obj, error);
  MOCHI_ERROR_RETURN(error);

  // Get the node coordinates
  Real3 rest = GetNodeCoordinates(reg, obj, params.nodeIndex, error);
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {obj};
  int const dofsPerNode = DofsPerNode(reg, obj);
  std::vector<std::vector<int>> actorDofs = {
      NodeIndexToDisplacementDofs(params.nodeIndex, dofsPerNode)};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::DeformableNodePosition,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(1),
      params);
  info.name = "DeformableNodePositionConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::DeformableNodePosition>>(e);
  data.rest = rest;

  // Create targets
  auto& target = reg.emplace<CConstraintTarget<Real3, TimeStep::Current>>(e);
  target.value = params.position;
  auto& targetOld = reg.emplace<CConstraintTarget<Real3, TimeStep::StageStart>>(e);
  targetOld.value = target.value;
}

void mochi::InitConstraint_ArticulatedSingleDofTarget(
    entt::registry& reg,
    entt::entity e,
    ArticulatedSingleDofTargetConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(IsFinite(params.targetValue), error, "Target DoF must be finite.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entity
  entt::entity obj = GetEntity(reg, ActorHandle(params.actor), error);
  MOCHI_ERROR_RETURN(error);

  // Make sure the actor is an articulated body
  MOCHI_ERROR_IF_NOT(reg.all_of<TagArticulatedActor>(obj), error, "Invalid actor type");
  MOCHI_ERROR_RETURN(error);

  // Ensure the joint is in the valid range
  auto const* joints = reg.get<CArticulatedBodyShape const>(obj).shape->GetJointsData();
  MOCHI_ERROR_IF(
      params.jointIndex < 0 || params.jointIndex >= isize(joints->dofInfo),
      error,
      "The joint id is not valid");
  MOCHI_ERROR_RETURN(error);

  // Ensure the joint is a translation or 1D joint
  auto const& dofInfo = joints->dofInfo[params.jointIndex];
  MOCHI_ERROR_IF_NOT(
      dofInfo.transSize > 0 || dofInfo.rotSize == 1,
      error,
      "ArticulatedSingleDofTargetConstraint must affect a translation or 1D joint");
  MOCHI_ERROR_RETURN(error);

  // Ensure the target dof is valid
  MOCHI_ERROR_IF(
      (params.dofIndex < 0) || (params.dofIndex >= dofInfo.GetSize()) ||
          (dofInfo.rotSize == 3 && params.dofIndex >= dofInfo.transSize),
      error,
      "Wrong dof index for ArticulatedSingleDofTargetConstraint");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {obj};
  std::vector<std::vector<int>> actorDofs = {{dofInfo.offset + params.dofIndex}};
  std::vector<std::vector<int>> actorTargets = {{dofInfo.offset + params.dofIndex}};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::ArticulatedSingleDofTarget,
      std::move(actors),
      std::move(actorDofs),
      std::move(actorTargets),
      params);
  info.name = "ArticulatedSingleDofTargetConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::ArticulatedSingleDofTarget>>(e);
  data.jointIdx = params.jointIndex;
  data.dofIdx = params.dofIndex;

  // Create targets
  auto& target = reg.emplace<CConstraintTarget<real, TimeStep::Current>>(e);
  target.value = params.targetValue;
  auto& targetOld = reg.emplace<CConstraintTarget<real, TimeStep::StageStart>>(e);
  targetOld.value = target.value;
}

void mochi::InitConstraint_Articulated3dRotationTarget(
    entt::registry& reg,
    entt::entity e,
    Articulated3dRotationTargetConstraintParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(IsFinite(params.target), error, "Target quaternion must be finite.");
  MOCHI_ERROR_IF(NearEqual(Norm(params.target), 0_r), error, "Target quaternion must be non-zero.");
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entity
  entt::entity obj = GetEntity(reg, ActorHandle(params.actor), error);
  MOCHI_ERROR_RETURN(error);

  // Make sure the actor is an articulated body
  MOCHI_ERROR_IF_NOT(reg.all_of<TagArticulatedActor>(obj), error, "Invalid actor type");
  MOCHI_ERROR_RETURN(error);

  // Ensure the joint is in the valid range
  auto const* joints = reg.get<CArticulatedBodyShape const>(obj).shape->GetJointsData();
  MOCHI_ERROR_IF(
      params.jointIndex < 0 || params.jointIndex >= isize(joints->dofInfo),
      error,
      "The joint id is not valid");
  MOCHI_ERROR_RETURN(error);

  // Ensure the joint is a 3D rotation joint (spherical or free)
  auto const& dofInfo = joints->dofInfo[params.jointIndex];
  MOCHI_ERROR_IF(
      dofInfo.rotSize != 3,
      error,
      "Articulated3dRotationTargetConstraint must affect a joint with 3D rotation (spherical or free)");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {obj};
  std::vector<std::vector<int>> actorDofs(1);
  actorDofs[0].resize(3);
  std::iota(actorDofs[0].begin(), actorDofs[0].end(), dofInfo.GetRotOffset());
  std::vector<std::vector<int>> actorTargets = {actorDofs[0]};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::Articulated3dRotationTarget,
      std::move(actors),
      std::move(actorDofs),
      std::move(actorTargets),
      params);
  info.name = "Articulated3dRotationTargetConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::Articulated3dRotationTarget>>(e);
  data.jointIdx = params.jointIndex;

  // Create targets
  auto targetRot = Normalize(params.target);
  auto& target = reg.emplace<CConstraintTarget<Quaternion, TimeStep::Current>>(e);
  target.value = targetRot;
  auto& targetOld = reg.emplace<CConstraintTarget<Quaternion, TimeStep::StageStart>>(e);
  targetOld.value = targetRot;
}

void mochi::InitConstraint_ArticulatedSingleDofRange(
    entt::registry& reg,
    entt::entity e,
    ArticulatedSingleDofRangeConstraintParams const& params,
    Error& error) {
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entity
  entt::entity obj = GetEntity(reg, ActorHandle(params.actor), error);
  MOCHI_ERROR_RETURN(error);

  // Make sure the actor is an articulated body
  MOCHI_ERROR_IF_NOT(reg.all_of<TagArticulatedActor>(obj), error, "Invalid actor type");
  MOCHI_ERROR_RETURN(error);

  // Ensure the joint is in the valid range
  auto const* joints = reg.get<CArticulatedBodyShape const>(obj).shape->GetJointsData();
  MOCHI_ERROR_IF(
      params.jointIndex < 0 || params.jointIndex >= isize(joints->dofInfo),
      error,
      "The joint id is not valid");
  MOCHI_ERROR_RETURN(error);

  // Ensure the joint is a translation or 1D joint
  auto const& dofInfo = joints->dofInfo[params.jointIndex];
  MOCHI_ERROR_IF_NOT(
      dofInfo.transSize > 0 || dofInfo.rotSize == 1,
      error,
      "ArticulatedSingleDofRangeConstraint must affect a translation or 1D joint");
  MOCHI_ERROR_RETURN(error);

  // Ensure the target dof is valid
  MOCHI_ERROR_IF(
      (params.dofIndex < 0) || (params.dofIndex >= dofInfo.GetSize()) ||
          (dofInfo.rotSize == 3 && params.dofIndex >= dofInfo.transSize),
      error,
      "Wrong dof index for ArticulatedSingleDofRangeConstraint");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {obj};
  std::vector<std::vector<int>> actorDofs(1);
  actorDofs[0] = {dofInfo.offset + params.dofIndex};
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::ArticulatedSingleDofRange,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(1),
      params);
  info.name = "ArticulatedSingleDofRangeConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::ArticulatedSingleDofRange>>(e);
  data.jointIdx = params.jointIndex;
  data.dofIdx = params.dofIndex;
  data.minValue = params.minValue;
  data.maxValue = params.maxValue;
}

void mochi::InitConstraint_Articulated3dRotationRange(
    entt::registry& reg,
    entt::entity e,
    Articulated3dRotationRangeConstraintParams const& params,
    Error& error) {
  MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS(params, error);
  MOCHI_PROFILE_SCOPE();

  // Get actor entity
  entt::entity obj = GetEntity(reg, ActorHandle(params.actor), error);
  MOCHI_ERROR_RETURN(error);

  // Make sure the actor is an articulated body
  MOCHI_ERROR_IF_NOT(reg.all_of<TagArticulatedActor>(obj), error, "Invalid actor type");
  MOCHI_ERROR_RETURN(error);

  // Ensure the joint is in the valid range
  auto const* joints = reg.get<CArticulatedBodyShape const>(obj).shape->GetJointsData();
  MOCHI_ERROR_IF(
      params.jointIndex < 0 || params.jointIndex >= isize(joints->dofInfo),
      error,
      "The joint id is not valid");
  MOCHI_ERROR_RETURN(error);

  // Ensure the joint is a 3D rotation joint (spherical or free)
  auto const& dofInfo = joints->dofInfo[params.jointIndex];
  MOCHI_ERROR_IF(
      dofInfo.rotSize != 3,
      error,
      "Articulated3dRotationTargetConstraint must affect a joint with 3D rotation (spherical or free)");
  MOCHI_ERROR_RETURN(error);

  // Create constraint info
  std::vector<entt::entity> actors = {obj};
  std::vector<std::vector<int>> actorDofs(1);
  actorDofs[0].resize(3);
  std::iota(actorDofs[0].begin(), actorDofs[0].end(), dofInfo.GetRotOffset());
  auto& info = reg.emplace<CConstraintInfo>(
      e,
      reg,
      ConstraintType::Articulated3dRotationRange,
      std::move(actors),
      std::move(actorDofs),
      std::vector<std::vector<int>>(1),
      params);
  info.name = "Articulated3dRotationRangeConstraint";

  // Create specific constraint data
  auto& data = reg.emplace<CConstraintData<ConstraintType::Articulated3dRotationRange>>(e);
  data.jointIdx = params.jointIndex;
  data.minValues = params.minValues;
  data.maxValues = params.maxValues;
}

#undef MOCHI_VALIDATE_COMMON_CONSTRAINT_PARAMS
#undef MOCHI_VALIDATE_DEFORMABLE_ACTOR

namespace {
// Partial registries for the evaluation of constraints and their Jacobians.
using RigidConstraintReg = ecs::PartialRegistry<
    CRigidState<TimeStep::Current> const,
    CRigidState<TimeStep::StageStart> const,
    CTimeIntegratorState const>;

using DeformableConstraintReg = ecs::PartialRegistry<
    CDisplacementSlice<real, TimeStep::Current> const,
    CDisplacementSlice<real, TimeStep::StageStart> const,
    CRodPose<TimeStep::Current> const,
    CRodPose<TimeStep::StageStart> const,
    CRootTransform const,
    CTimeIntegratorState const>;

using DeformableAndRigidConstraintReg = ecs::PartialRegistry<
    CDisplacementSlice<real, TimeStep::Current> const,
    CDisplacementSlice<real, TimeStep::StageStart> const,
    CRodPose<TimeStep::Current> const,
    CRodPose<TimeStep::StageStart> const,
    CRootTransform const,
    CRigidState<TimeStep::Current> const,
    CRigidState<TimeStep::StageStart> const,
    CTimeIntegratorState const>;

using PivotConstraintReg = ecs::PartialRegistry<
    CRigidState<TimeStep::Current> const,
    CRigidState<TimeStep::StageStart> const,
    CRigidBodyInertia const,
    CMeshPivot const,
    CTimeIntegratorState const>;

using ArticulatedConstraintReg = ecs::PartialRegistry<
    CArticulatedReducedPose<TimeStep::Current> const,
    CArticulatedReducedPose<TimeStep::StageStart> const,
    CArticulatedBodyShape const,
    CArticulatedJointPoseInfo const,
    CTimeIntegratorState const>;

using RodAndRigidConstraintReg = ecs::PartialRegistry<
    CRigidState<TimeStep::Current> const,
    CRigidState<TimeStep::StageStart> const,
    CRodPose<TimeStep::Current> const,
    CRodPose<TimeStep::StageStart> const,
    CPolylineMesh const,
    CRootTransform const,
    CTimeIntegratorState const>;

// Partial registry for global assembly that requires link actor Jacobian lookups
using LinkActorAssemblyReg =
    ecs::PartialRegistry<CArticulatedEntity const, CArticulatedRigidJacobian const>;

// Empty target type, used for constraints that do not have a target value.
struct NoTarget {};

// Helper to get the parent articulated actor from a partial registry
entt::entity TryGetParentArticulatedActorLocal(
    LinkActorAssemblyReg const& reg,
    entt::entity actor) {
  if (auto const* articulatedEntity = reg.try_get<CArticulatedEntity const>(actor)) {
    return articulatedEntity->entity;
  }
  return entt::null;
}

// Traits to map ConstraintType to its PartialRegistry type and target type
template <ConstraintType kType>
struct ConstraintTraits {};

#define MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(kType, kReg, kTargetType) \
  template <>                                                        \
  struct ConstraintTraits<ConstraintType::kType> {                   \
    using RegistryType = kReg;                                       \
    using TargetType = kTargetType;                                  \
  }

MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(RigidSphericalJoint, RigidConstraintReg, NoTarget);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(RigidPrismaticJoint, RigidConstraintReg, NoTarget);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(
    DeformableNodeToDeformableNode,
    DeformableConstraintReg,
    NoTarget);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(
    DeformableNodeToRigid,
    DeformableAndRigidConstraintReg,
    NoTarget);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(DeformableNodePosition, DeformableConstraintReg, Real3);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(JointRotationRange, RigidConstraintReg, NoTarget);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(JointRotationTracking, RigidConstraintReg, Quaternion);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(RodElementRotationToRigid, RodAndRigidConstraintReg, Quaternion);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(RigidPivotPosition, PivotConstraintReg, Real3);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(RigidPivotToRigidTarget, PivotConstraintReg, TransformRT);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(RigidPivotRotation, RigidConstraintReg, Quaternion);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(ArticulatedSingleDofTarget, ArticulatedConstraintReg, real);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(
    Articulated3dRotationTarget,
    ArticulatedConstraintReg,
    Quaternion);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(ArticulatedSingleDofRange, ArticulatedConstraintReg, NoTarget);
MOCHI_SPECIALIZE_CONSTRAINT_TRAITS(Articulated3dRotationRange, ArticulatedConstraintReg, NoTarget);

#undef MOCHI_SPECIALIZE_CONSTRAINT_TRAITS
} // namespace

// Overloaded constraint evaluation functions
template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    RigidConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::RigidSphericalJoint> const& data,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 12>* outJac,
    bool& outActive) {
  auto const& stateA = reg.get<CRigidState<kTimeStep> const>(info.actors[0]).value;
  auto const& stateB = reg.get<CRigidState<kTimeStep> const>(info.actors[1]).value;
  EvalRigidSphericalJointConstraint(stateA, stateB, data.posLocalA, data.posLocalB, outC, outJac);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    RigidConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::RigidPrismaticJoint> const& data,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 9>* outJac,
    bool& outActive) {
  auto const& stateA = reg.get<CRigidState<kTimeStep> const>(info.actors[0]).value;
  auto const& posB = reg.get<CRigidState<kTimeStep> const>(info.actors[1]).value.GetTranslation();
  EvalRigidPrismaticJointConstraint(
      stateA, posB, data.localFrame, data.tRef, data.max, data.min, outC, outJac);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    DeformableConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::DeformableNodeToDeformableNode> const& data,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 6>* outJac,
    bool& outActive) {
  auto const& transformA = reg.get<CRootTransform const>(info.actors[0]).worldFromLocal;
  auto const& transformB = reg.get<CRootTransform const>(info.actors[1]).worldFromLocal;
  Real3 dispA =
      GetDeformableDisplacement<kTimeStep>(reg, info.actors[0], MakeConstSpan(info.actorDofs[0]));
  Real3 dispB =
      GetDeformableDisplacement<kTimeStep>(reg, info.actors[1], MakeConstSpan(info.actorDofs[1]));
  EvalDeformableNodeToDeformableNodeConstraint(
      transformA, transformB, data.restA + dispA, data.restB + dispB, outC, outJac);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    DeformableAndRigidConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::DeformableNodeToRigid> const& data,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 9>* outJac,
    bool& outActive) {
  auto const& stateRigid = reg.get<CRigidState<kTimeStep> const>(info.actors[0]).value;
  auto const& transformDeformable = reg.get<CRootTransform const>(info.actors[1]).worldFromLocal;
  Real3 dispDeformable =
      GetDeformableDisplacement<kTimeStep>(reg, info.actors[1], MakeConstSpan(info.actorDofs[1]));
  EvalDeformableNodeToRigidConstraint(
      stateRigid,
      transformDeformable,
      data.restDeformable + dispDeformable,
      data.posLocalRigid,
      outC,
      outJac);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    DeformableConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::DeformableNodePosition> const& data,
    CConstraintTarget<Real3, kTimeStep> const& target,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 3>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget,
    bool& outActive) {
  auto const& transform = reg.get<CRootTransform const>(info.actors[0]).worldFromLocal;
  Real3 disp =
      GetDeformableDisplacement<kTimeStep>(reg, info.actors[0], MakeConstSpan(info.actorDofs[0]));
  EvalDeformableNodeFixedConstraint(
      transform, data.rest + disp, target.value, outC, outJac, outJacTarget);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    RigidConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::JointRotationRange> const& data,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 6>* outJac,
    bool& outActive) {
  auto const& rotA = reg.get<CRigidState<kTimeStep> const>(info.actors[0]).value.GetRotation();
  auto const& rotB = reg.get<CRigidState<kTimeStep> const>(info.actors[1]).value.GetRotation();
  EvalJointRotationRangeConstraint(
      rotA, rotB, data.q0, data.qr, data.minRotVec, data.maxRotVec, outC, outJac, outActive);
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    RigidConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::JointRotationTracking> const& data,
    CConstraintTarget<Quaternion, kTimeStep> const& target,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 6>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget,
    bool& outActive) {
  auto const& rotA = reg.get<CRigidState<kTimeStep> const>(info.actors[0]).value.GetRotation();
  auto const& rotB = reg.get<CRigidState<kTimeStep> const>(info.actors[1]).value.GetRotation();
  EvalJointRotationTargetConstraint(rotA, rotB, data.q0, target.value, outC, outJac, outJacTarget);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    RodAndRigidConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::RodElementRotationToRigid> const& data,
    CConstraintTarget<Quaternion, kTimeStep> const& target,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 11>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget,
    bool& outActive) {
  // Get rigid rotation
  auto const& rotRigid = reg.get<CRigidState<kTimeStep> const>(info.actors[0]).value.GetRotation();

  // Get rod local DoFs (8 DoFs: two nodes with 4 DoFs each)
  auto const& rodActor = info.actors[1];
  auto const& rodDisp = reg.get<CRodPose<kTimeStep> const>(rodActor).value.displacements;
  NdArray<real, 8> localDofsRod MOCHI_NO_INIT;
  Store(&(localDofsRod[0]), Load<Vec4r>(&(rodDisp[info.actorDofs[1][0]])));
  Store(&(localDofsRod[4]), Load<Vec4r>(&(rodDisp[info.actorDofs[1][4]])));

  // Reference quantities and element frame axes from rod element:
  int const elementIndex = data.elementIndex;
  auto const& polylineMesh = reg.get<CPolylineMesh const>(rodActor);
  Int2 const evalEn = polylineMesh.ElementNodes(elementIndex);
  Vec4r const X0 = ToSimd(polylineMesh.nodes[evalEn[0]]);
  Vec4r const X1 = ToSimd(polylineMesh.nodes[evalEn[1]]);
  auto const& rodPose = reg.get<CRodPose<kTimeStep> const>(rodActor);
  Vec4r const frameAxis = Normalize<3>(ToSimd(rodPose.value.frameAxes[elementIndex]));

  // Get rod world-from-local transform
  auto const& worldFromLocalRod = reg.get<CRootTransform const>(rodActor).worldFromLocal;

  EvalRodElementRotationToRigidConstraint(
      rotRigid,
      worldFromLocalRod,
      X0,
      X1,
      frameAxis,
      localDofsRod,
      data.q0,
      target.value,
      outC,
      outJac,
      outJacTarget);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    PivotConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::RigidPivotPosition> const& data,
    CConstraintTarget<Real3, kTimeStep> const& target,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 6>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget,
    bool& outActive) {
  auto const& state = reg.get<CRigidState<kTimeStep> const>(info.actors[0]).value;
  auto [inertia, pivot] = reg.try_get<CRigidBodyInertia const, CMeshPivot const>(info.actors[0]);
  Real3 pivotPos = pivot ? pivot->position : Real3{};
  if (inertia) {
    EvalRigidPositionFixedConstraint(
        state,
        data.posLocal - ToReal3(inertia->GetCenterOfMassLocal()),
        target.value,
        outC,
        outJac,
        outJacTarget);
  } else {
    EvalRigidPositionFixedConstraint(
        state, data.posLocal - pivotPos, target.value, outC, outJac, outJacTarget);
  }
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    PivotConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::RigidPivotToRigidTarget> const& data,
    CConstraintTarget<TransformRT, kTimeStep> const& target,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 6>* outJac,
    RowMatrix<real, 3, 6>* outJacTarget,
    bool& outActive) {
  auto const& state = reg.get<CRigidState<kTimeStep> const>(info.actors[0]).value;
  auto const com = ToReal3(reg.get<CRigidBodyInertia const>(info.actors[0]).GetCenterOfMassLocal());
  EvalRigidPositionToRigidTargetConstraint(
      state, target.value, data.posLocal - com, outC, outJac, outJacTarget);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    RigidConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::RigidPivotRotation> const& data,
    CConstraintTarget<Quaternion, kTimeStep> const& target,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 3>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget,
    bool& outActive) {
  auto const& rot = reg.get<CRigidState<kTimeStep> const>(info.actors[0]).value.GetRotation();
  EvalRotationFixedConstraint(rot, data.rotLocal, target.value, outC, outJac, outJacTarget);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    ArticulatedConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::ArticulatedSingleDofTarget> const& data,
    CConstraintTarget<real, kTimeStep> const& target,
    ColumnVector<real, 1>* outC,
    RowMatrix<real, 1, 1>* outJac,
    RowMatrix<real, 1, 1>* outJacTarget,
    bool& outActive) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  auto const* joints = reg.get<CArticulatedBodyShape const>(info.actors[0]).shape->GetJointsData();
  MOCHI_ASSERT_VERBOSE(
      data.dofIdx >= 0 && data.dofIdx < joints->dofInfo[data.jointIdx].GetSize(),
      "DoF index out of bounds.");
#endif
  auto const& reducedPose = reg.get<CArticulatedReducedPose<kTimeStep> const>(info.actors[0]).value;
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(info.actors[0])[data.jointIdx];
  auto dofValue = reducedPose(poseInfo.offset + data.dofIdx);
  real* outValPtr = outC ? outC->data() : nullptr;
  real* outJacPtr = outJac ? outJac->data() : nullptr;
  real* outJacTargetPtr = outJacTarget ? outJacTarget->data() : nullptr;
  EvalSingleDofTargetConstraint(dofValue, target.value, outValPtr, outJacPtr, outJacTargetPtr);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    ArticulatedConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::Articulated3dRotationTarget> const& data,
    CConstraintTarget<Quaternion, kTimeStep> const& target,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 3>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget,
    bool& outActive) {
  auto const& reducedPose = reg.get<CArticulatedReducedPose<kTimeStep> const>(info.actors[0]).value;
  auto const* joints = reg.get<CArticulatedBodyShape const>(info.actors[0]).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(info.actors[0])[data.jointIdx];
  Quaternion rot = mochi::articulated::ComputeJointTransform(
                       reducedPose,
                       joints->jointTypes[data.jointIdx],
                       joints->jointAxes[data.jointIdx],
                       poseInfo)
                       .GetRotation();
  EvalRotationFixedConstraint(
      rot, Quaternion::Identity(), target.value, outC, outJac, outJacTarget);
  outActive = true;
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    ArticulatedConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::ArticulatedSingleDofRange> const& data,
    ColumnVector<real, 1>* outC,
    RowMatrix<real, 1, 1>* outJac,
    bool& outActive) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  auto const* joints = reg.get<CArticulatedBodyShape const>(info.actors[0]).shape->GetJointsData();
  MOCHI_ASSERT_VERBOSE(
      data.dofIdx >= 0 && data.dofIdx < joints->dofInfo[data.jointIdx].GetSize(),
      "DoF index out of bounds.");
#endif
  auto const& reducedPose = reg.get<CArticulatedReducedPose<kTimeStep> const>(info.actors[0]).value;
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(info.actors[0])[data.jointIdx];
  auto dofValue = reducedPose(poseInfo.offset + data.dofIdx);
  real* outValPtr = outC ? outC->data() : nullptr;
  real* outJacPtr = outJac ? outJac->data() : nullptr;
  EvalSingleDofRangeConstraint(
      dofValue, data.minValue, data.maxValue, outValPtr, outJacPtr, outActive);
}

template <TimeStep kTimeStep>
static void EvalConstraintImpl(
    ArticulatedConstraintReg reg,
    CConstraintInfo const& info,
    CConstraintData<ConstraintType::Articulated3dRotationRange> const& data,
    ColumnVector<real, 3>* outC,
    RowMatrix<real, 3, 3>* outJac,
    bool& outActive) {
  auto const& reducedPose = reg.get<CArticulatedReducedPose<kTimeStep> const>(info.actors[0]).value;
  auto const* joints = reg.get<CArticulatedBodyShape const>(info.actors[0]).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(info.actors[0])[data.jointIdx];
  Quaternion rot = mochi::articulated::ComputeJointTransform(
                       reducedPose,
                       joints->jointTypes[data.jointIdx],
                       joints->jointAxes[data.jointIdx],
                       poseInfo)
                       .GetRotation();
  EvalRotationRangeConstraint(
      rot.ToRotationVector(), data.minValues, data.maxValues, outC, outJac, outActive);
}

template <ConstraintType kType, TimeStep kTimeStep>
static void EvalConstraintT(
    entt::registry& reg,
    entt::entity e,
    Span<real> outC,
    Span<real> outJac,
    Span<real> outJacTarget,
    bool& outActive) {
  // Create containers with compile-time sizes
  static int constexpr kCSize = GetConstraintSize(kType);
  static int constexpr kDofsSize = GetNumConstrainedDofs(kType);
  static int constexpr kTargetSize = GetNumConstrainedTargets(kType);
  MOCHI_ASSERT_VERBOSE(
      !outC || isize(outC) == kCSize, "EvalConstraint output does not match constraint size");
  MOCHI_ASSERT_VERBOSE(
      !outJac || isize(outJac) == kCSize * kDofsSize,
      "EvalConstraint output does not match constraint Jacobian size");
  MOCHI_ASSERT_VERBOSE(
      !outJacTarget || isize(outJacTarget) == kCSize * kTargetSize,
      "EvalConstraint output does not match constraint target Jacobian size");
  ColumnVector<real, kCSize> c;
  RowMatrix<real, kCSize, kDofsSize> jac;
  auto* cPtr = outC ? &c : nullptr;
  auto* jacPtr = outJac ? &jac : nullptr;

  // Fetch constraint data and call evaluation
  auto const& info = reg.get<CConstraintInfo const>(e);
  auto const& data = reg.get<CConstraintData<kType> const>(e);
  if constexpr (std::is_same_v<typename ConstraintTraits<kType>::TargetType, NoTarget>) {
    EvalConstraintImpl<kTimeStep>(reg, info, data, cPtr, jacPtr, outActive);
  } else {
    using CTarget = CConstraintTarget<typename ConstraintTraits<kType>::TargetType, kTimeStep>;
    auto const& target = reg.get<CTarget const>(e);
    RowMatrix<real, kCSize, kTargetSize> jacTarget;
    auto* jacTargetPtr = outJacTarget ? &jacTarget : nullptr;
    EvalConstraintImpl<kTimeStep>(reg, info, data, target, cPtr, jacPtr, jacTargetPtr, outActive);

    // Copy target Jacobian to the output container
    if (outJacTarget) {
      RowMatrixView<real, kCSize, kTargetSize>(outJacTarget.data()) = jacTarget;
    }
  }

  // Copy results to the output containers
  if (outC) {
    AsView(outC) = c;
  }
  if (outJac) {
    RowMatrixView<real, kCSize, kDofsSize>(outJac.data()) = jac;
  }
}

template <TimeStep kTimeStep>
void mochi::EvalConstraint(
    entt::registry& reg,
    entt::entity e,
    Span<real> outC,
    Span<real> outJac,
    Span<real> outJacTarget,
    bool& outActive) {
  static_assert(kTimeStep == TimeStep::Current || kTimeStep == TimeStep::StageStart);

  auto const& type = reg.get<CConstraintInfo const>(e).type;

  static_assert(
      static_cast<int>(ConstraintType::Count) == 16,
      "Please update the switch statement below if ConstraintType enum changes");

#define MOCHI_TRY_CALL_EVAL_CONSTRAINT(kType)                                         \
  case kType:                                                                         \
    EvalConstraintT<kType, kTimeStep>(reg, e, outC, outJac, outJacTarget, outActive); \
    break;

  switch (type) {
    case ConstraintType::None:
      MOCHI_ASSERT(false, "Invalid constraint type");
      break;
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::ArticulatedSingleDofRange);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::Articulated3dRotationRange);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::ArticulatedSingleDofTarget);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::Articulated3dRotationTarget);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::JointRotationRange);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::JointRotationTracking);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::RodElementRotationToRigid);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::RigidPivotPosition);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::RigidPivotToRigidTarget);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::RigidPivotRotation);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::RigidPrismaticJoint);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::RigidSphericalJoint);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::DeformableNodePosition);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::DeformableNodeToRigid);
      MOCHI_TRY_CALL_EVAL_CONSTRAINT(ConstraintType::DeformableNodeToDeformableNode);
    default:
      MOCHI_ASSERT(false, "Unhandled constraint type");
      break;
  }

#undef MOCHI_TRY_CALL_EVAL_CONSTRAINT
}

template MOCHI_API void mochi::EvalConstraint<TimeStep::Current>(
    entt::registry&,
    entt::entity,
    Span<real>,
    Span<real>,
    Span<real>,
    bool&);
template MOCHI_API void mochi::EvalConstraint<TimeStep::StageStart>(
    entt::registry&,
    entt::entity,
    Span<real>,
    Span<real>,
    Span<real>,
    bool&);

template <GradTarget kGradTarget, int kCSize, int kDofsSize>
static void ComputeLocalObjResDRes(
    CConstraintInfo const& info,
    real dtStage,
    bool useFittedSaturationHessian,
    ColumnVectorView<real const, kCSize> c,
    ColumnVectorView<real const, kCSize> cOld,
    RowMatrixView<real const, kCSize, kDofsSize> jac, // jac or jacOld, depending on kGradTarget
    double* outObj,
    ColumnVector<real, kDofsSize>* outRes,
    Matrix<real, kDofsSize, kDofsSize>* outDRes) {
  MOCHI_PROFILE_SCOPE();

  static_assert(
      kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous ||
          kGradTarget == GradTarget::CurrentInput || kGradTarget == GradTarget::PreviousInput,
      "Unexpected grad target");

  real const stiffness = info.stiffness;
  real const dampingFactor = info.damping / dtStage;
  real const saturation = info.saturation;

  // Compute constraint objective, residual and dresidual and store in result structures.
  // With GradTarget::Previous or GradTarget::PreviousInput saturation is irrelevant, because we
  // only support residual computation and the residual is due only to the damping term.
  if (saturation < 0_r || kGradTarget == GradTarget::Previous ||
      kGradTarget == GradTarget::PreviousInput) {
    // Add objective and residual
    if (outObj || outRes) {
      ColumnVector<real, kCSize> dC = c - cOld;
      if (outObj) {
        *outObj = 0.5 * (stiffness * c.NormSqr() + dampingFactor * dC.NormSqr());
      }
      if (outRes) {
        if constexpr (
            kGradTarget == GradTarget::Previous || kGradTarget == GradTarget::PreviousInput) {
          *outRes = jac.Transpose() * ColumnVector<real, kCSize>((-dampingFactor) * dC);
        } else {
          *outRes =
              jac.Transpose() * ColumnVector<real, kCSize>(stiffness * c + dampingFactor * dC);
        }
      }
    }

    // Add dresidual. We ignore the term that depends on the second derivative of the constraint
    if (outDRes) {
      *outDRes =
          Matrix<real, kDofsSize, kCSize>((stiffness + dampingFactor) * jac.Transpose()) * jac;
    }
  } else {
    real const valNorm = c.Norm();
    real const maxForce = stiffness * saturation;

    // Compute smooth force clamping f(valNorm) -> [0,1]
    real smoother = 0_r;
    real dSmoother = 0_r;
    real ddSmoother = 0_r;
    real dSmoother_valNorm = 0_r;
    IPCstepC1(valNorm, saturation, smoother, dSmoother, ddSmoother, dSmoother_valNorm);

    // Add objective and residual
    if (outObj || outRes) {
      ColumnVector<real, kCSize> dC = c - cOld;
      if (outObj) {
        *outObj = maxForce * smoother + 0.5 * dampingFactor * dC.NormSqr();
      }
      if (outRes) {
        *outRes = jac.Transpose() *
            ColumnVector<real, kCSize>((maxForce * dSmoother_valNorm) * c + dampingFactor * dC);
      }
    }

    // Add dresidual
    if (outDRes) {
      if (useFittedSaturationHessian) {
        // The exact Hessian in the direction of cUnit is problematic, as it gives a bad global
        // quadratic approximation of the merit. Instead, use the same Hessian as in other
        // directions, which amounts to fitting a global quadratic approximation based on the local
        // gradient. This approach is the same as the one used for static IPC friction in
        // contact_utils.h

        // NOTE: we ignore the term that depends on the second derivative of the constraint
        Matrix<real, kDofsSize, kCSize> tmp =
            (dSmoother_valNorm * maxForce + dampingFactor) * jac.Transpose();
        *outDRes = tmp * jac;
      } else {
        // Used only for consistency tests
        Matrix<real, kCSize, kCSize> eye;
        eye.SetIdentity();
        Matrix<real, kCSize, kCSize> d2c;
        if (valNorm < 1e-9_r) {
          d2c = dSmoother_valNorm * eye;
        } else {
          ColumnVector<real, kCSize> cUnit = (1_r / valNorm) * c;
          Matrix<real, kCSize, kCSize> cUnitCUnitT = cUnit * cUnit.Transpose();
          d2c = dSmoother_valNorm * eye + (ddSmoother - dSmoother_valNorm) * cUnitCUnitT;
        }

        // NOTE: we ignore the term that depends on the second derivative of the constraint
        Matrix<real, kCSize, kCSize> tmp = maxForce * d2c + dampingFactor * eye;
        *outDRes = Matrix<real, kDofsSize, kCSize>(jac.Transpose() * tmp) * jac;
      }
    }
  }
}

template <int kDofsSize>
static void AssembleGlobalObjResDRes(
    ConstraintGlobalSparsityCache const& sparsity,
    CConstraintInfo const& info,
    LinkActorAssemblyReg const& reg,
    double const* obj,
    ColumnVector<real, kDofsSize> const* res,
    Matrix<real, kDofsSize, kDofsSize> const* dres,
    CCompoundConstraintSnle& outConstraintSnle) {
  MOCHI_PROFILE_SCOPE();

  // Add energy (objective is independent of DoF transformation)
  if (obj) {
    outConstraintSnle.objective += *obj;
  }

  if (!info.hasMixedLinks) {
    // Simple case: no transformation needed, use direct assembly
    if (res) {
      auto const& globalResIndices = sparsity.resIndices;
      MOCHI_ASSERT_VERBOSE(isize(globalResIndices) == kDofsSize);
      auto& outRes = outConstraintSnle.residuals[0].second;
      ArrayPlusEqualsIndexedDst(outRes.data(), res->data(), globalResIndices.data(), kDofsSize);
    }

    if (dres) {
      auto const& globalDResIndices = sparsity.dresIndices;
      MOCHI_ASSERT_VERBOSE(isize(globalDResIndices) == kDofsSize * kDofsSize);
      auto& outDRes = std::get<SparseMatrix<real>>(outConstraintSnle.dresiduals[0].matrix);
      ArrayPlusEqualsIndexedDst(
          outDRes.Values().data(), dres->data(), globalDResIndices.data(), kDofsSize * kDofsSize);
    }
  } else if (res || dres) {
    // If some actors involved in the constraint are links of an articulation, then the constraint
    // Jacobian used for the residual and (approximate) dresidual is
    //
    // d(constraint value) / d(compound DoFs)
    //  = d(constraint value) / d(constraint DoFs) * d(constraint DoFs) / d(compound DoFs)
    //
    // where "constraint DoFs" includes position and/or rotation DoFs of the links, and "compound
    // DoFs" instead uses reduced-pose DoFs of the parent articulation. In this code, function, we
    // use the notation
    //
    //  J := d(constraint DoFs) / d(compound DoFs)
    //
    // for brevity. In the special case of a constraint on all rigid-body DoFs of a link actor, this
    // would be the link's ArticulatedRigidJacobian.

    auto const& allCompoundDofs = sparsity.resIndices;
    int const compoundDofsSize = isize(allCompoundDofs);

    // Build the Jacobian matrix on the fly using current link Jacobian values
    // J has shape (compoundDofsSize x kDofsSize) and maps from constraint DoFs to compound DoFs
    RowMatrix<real> J = RowMatrix<real>::Zero(kDofsSize, compoundDofsSize);

    int constraintDofOffset = 0;
    int compoundDofOffset = 0;
    for (int o = 0; o < isize(info.actors); ++o) {
      entt::entity actor = info.actors[o];
      auto const& actorDofs = info.actorDofs[o];
      int numActorDofs = isize(actorDofs);
      entt::entity parentArticulated = TryGetParentArticulatedActorLocal(reg, actor);

      if (parentArticulated != entt::null) {
        // This actor is a link. Map through the link Jacobian.
        auto const& linkJacobian = reg.get<CArticulatedRigidJacobian const>(actor);
        int const numLinkReducedDofs = isize(linkJacobian.dofs);

        // For each actor DoF used by the constraint, find the corresponding row
        // in the link Jacobian and add its contributions
        for (int localIdx = 0; localIdx < numActorDofs; ++localIdx) {
          int const actorDof = actorDofs[localIdx];
          J.Row(constraintDofOffset + localIdx).MiddleCols(compoundDofOffset, numLinkReducedDofs) =
              linkJacobian.value.Row(actorDof);
        }
        compoundDofOffset += numLinkReducedDofs;
      } else {
        // Regular actor, identity mapping for these DoFs
        for (int localIdx = 0; localIdx < numActorDofs; ++localIdx) {
          int const compoundDofIdx = compoundDofOffset + localIdx;
          J(constraintDofOffset + localIdx, compoundDofIdx) = 1_r;
        }
        compoundDofOffset += numActorDofs;
      }
      constraintDofOffset += numActorDofs;
    }

    // Used to form residual and/or dresidual
    auto JTranspose = Transpose(J);

    if (res) {
      // Compute transformed residual: res_compound = J^T * res_constraint
      ColumnVector<real> const resTransformed = JTranspose * (*res);

      // Add to global residual using compound DoF indices
      auto const& globalResIndices = sparsity.resIndices;
      MOCHI_ASSERT_VERBOSE(isize(globalResIndices) == compoundDofsSize);
      auto& outRes = outConstraintSnle.residuals[0].second;
      ArrayPlusEqualsIndexedDst(
          outRes.data(), resTransformed.data(), globalResIndices.data(), compoundDofsSize);
    }

    if (dres) {
      // Compute transformed dresidual: dres_compound = J^T * dres_constraint * J
      Matrix<real> const dresTransformed = JTranspose * (*dres) * J;

      // Add to global dresidual
      auto const& globalDResIndices = sparsity.dresIndices;
      MOCHI_ASSERT_VERBOSE(isize(globalDResIndices) == compoundDofsSize * compoundDofsSize);
      auto& outDRes = std::get<SparseMatrix<real>>(outConstraintSnle.dresiduals[0].matrix);
      ArrayPlusEqualsIndexedDst(
          outDRes.Values().data(),
          dresTransformed.data(),
          globalDResIndices.data(),
          compoundDofsSize * compoundDofsSize);
    }
  }
}

template <ConstraintType kType, GradTarget kGradTarget>
static void AssembleConstraintT(
    AssemblyParams const& params,
    CCompoundConstraintSnle& outConstraintSnle,
    LinkActorAssemblyReg linkActorReg,
    typename ConstraintTraits<kType>::RegistryType reg,
    CConstraintInfo const& info,
    CConstraintData<kType> const& data,
    CConstraintTarget<typename ConstraintTraits<kType>::TargetType, TimeStep::Current> const*
        target,
    CConstraintTarget<typename ConstraintTraits<kType>::TargetType, TimeStep::StageStart> const*
        targetOld,
    CConstraintGlobalSparsityCache const& sparsity,
    CQueryConstraintForce* outQuery) {
  MOCHI_PROFILE_SCOPE();

  static_assert(
      kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous,
      "Unexpected grad target");

  static int constexpr kCSize = GetConstraintSize(kType);
  static int constexpr kDofsSize = GetNumConstrainedDofs(kType);

  auto handleQuery = [&](ColumnVectorView<real const, kDofsSize> force) {
    if (outQuery && params.assemRes && kGradTarget == GradTarget::Current) {
      outQuery->force.Resize(kDofsSize);
      outQuery->force = force;
    }
  };

  if ((info.stiffness == 0_r || kGradTarget == GradTarget::Previous) && info.damping == 0_r) {
    // Handle force queries and early exit
    handleQuery(ColumnVector<real, kDofsSize>::Zero());
    return;
  }

  // Create containers for constraint evaluation
  bool const evalC = params.assemObj || params.assemRes || info.saturation > 0;
  bool const evalCOld = evalC && info.damping > 0_r;
  bool const evalJac = kGradTarget == GradTarget::Current && (params.assemRes || params.assemDRes);
  bool const evalJacOld = kGradTarget == GradTarget::Previous && params.assemRes;
  ColumnVector<real, kCSize> c MOCHI_NO_INIT;
  ColumnVector<real, kCSize> cOld = {};
  RowMatrix<real, kCSize, kDofsSize> jac MOCHI_NO_INIT;
  auto* cPtr = evalC ? &c : nullptr;
  auto* cOldPtr = evalCOld ? &cOld : nullptr;
  auto* jacPtr = evalJac ? &jac : nullptr;
  auto* jacOldPtr = evalJacOld ? &jac : nullptr;

  // Evaluate current constraint value and Jacobian
  bool isActive{};
  if constexpr (std::is_same_v<typename ConstraintTraits<kType>::TargetType, NoTarget>) {
    EvalConstraintImpl<TimeStep::Current>(reg, info, data, cPtr, jacPtr, isActive);
  } else {
    MOCHI_ASSERT_VERBOSE(target, "Missing target");
    EvalConstraintImpl<TimeStep::Current>(reg, info, data, *target, cPtr, jacPtr, {}, isActive);
  }

  // Possibly evaluate old constraint value and Jacobian
  MOCHI_ASSERT_VERBOSE(
      evalCOld || !evalJacOld, "Should not evaluate old Jacobian if old value is not needed");
  if (evalCOld) {
    bool isActiveOld{};
    if constexpr (std::is_same_v<typename ConstraintTraits<kType>::TargetType, NoTarget>) {
      EvalConstraintImpl<TimeStep::StageStart>(reg, info, data, cOldPtr, jacOldPtr, isActiveOld);
    } else {
      MOCHI_ASSERT_VERBOSE(targetOld, "Missing target");
      EvalConstraintImpl<TimeStep::StageStart>(
          reg, info, data, *targetOld, cOldPtr, jacOldPtr, {}, isActiveOld);
    }
    isActive |= isActiveOld;
  }

  if (!isActive) {
    // Handle force queries and early exit
    handleQuery(ColumnVector<real, kDofsSize>::Zero());
    return;
  }

  // Compute local terms
  real const dtStage = reg.template get<CTimeIntegratorState const>(info.actors[0]).dtStage;
  double obj = 0.0;
  ColumnVector<real, kDofsSize> res{}; // Zero-initialized to avoid "-res" UB in handleQuery
  Matrix<real, kDofsSize, kDofsSize> dres MOCHI_NO_INIT;
  ComputeLocalObjResDRes<kGradTarget>(
      info,
      dtStage,
      params.fittedSaturationHessian.constraintSaturation,
      AsConstView(c),
      AsConstView(cOld),
      AsConstView(jac),
      params.assemObj ? &obj : nullptr,
      params.assemRes ? &res : nullptr,
      params.assemDRes ? &dres : nullptr);

  // Global assembly (with optional Jacobian transformation for constraints involving link actors)
  AssembleGlobalObjResDRes(
      sparsity,
      info,
      linkActorReg,
      params.assemObj ? &obj : nullptr,
      params.assemRes ? &res : nullptr,
      params.assemDRes ? &dres : nullptr,
      outConstraintSnle);

  // Handle force queries
  handleQuery(ColumnVector<real, kDofsSize>(-res));
}

template <ConstraintType kType, GradTarget kGradTarget>
static void AssembleConstraintWrtInputT(
    AssemblyParams const& params,
    CCompoundConstraintSnle& outConstraintSnle,
    LinkActorAssemblyReg linkActorReg,
    typename ConstraintTraits<kType>::RegistryType reg,
    CConstraintInfo const& info,
    CConstraintData<kType> const& data,
    CConstraintTarget<typename ConstraintTraits<kType>::TargetType, TimeStep::Current> const&
        target,
    CConstraintTarget<typename ConstraintTraits<kType>::TargetType, TimeStep::StageStart> const&
        targetOld,
    CConstraintGlobalInputSparsityCache const& sparsity) {
  MOCHI_PROFILE_SCOPE();

  static_assert(
      kGradTarget == GradTarget::CurrentInput || kGradTarget == GradTarget::PreviousInput,
      "Unexpected grad target");
  MOCHI_ASSERT_VERBOSE(
      params.assemObj == false && params.assemRes == true && params.assemDRes == false,
      "Invalid params");
  MOCHI_ASSERT(
      !info.hasMixedLinks,
      "Differentiability not supported for constraints between links of an articulated body "
      "and external actors.");

  static int constexpr kCSize = GetConstraintSize(kType);
  static int constexpr kTargetSize = GetNumConstrainedTargets(kType);

  if ((info.stiffness == 0_r || kGradTarget == GradTarget::PreviousInput) && info.damping == 0_r) {
    return;
  }

  // Create containers for constraint evaluation
  ColumnVector<real, kCSize> c MOCHI_NO_INIT;
  ColumnVector<real, kCSize> cOld = {};
  RowMatrix<real, kCSize, kTargetSize> jac MOCHI_NO_INIT;
  auto* jacPtr = kGradTarget == GradTarget::CurrentInput ? &jac : nullptr;
  auto* jacOldPtr = kGradTarget == GradTarget::PreviousInput ? &jac : nullptr;

  // Evaluate current constraint value and Jacobian
  bool isActive = false;
  EvalConstraintImpl<TimeStep::Current>(reg, info, data, target, &c, {}, jacPtr, isActive);

  // Possibly evaluate old constraint value and Jacobian
  bool isActiveOld = false;
  if (info.damping > 0_r) {
    EvalConstraintImpl<TimeStep::StageStart>(
        reg, info, data, targetOld, &cOld, {}, jacOldPtr, isActiveOld);
  }

  if (!isActive && !isActiveOld) {
    // Early exit
    return;
  }

  // Compute local gradient
  real const dtStage = reg.template get<CTimeIntegratorState const>(info.actors[0]).dtStage;
  ColumnVector<real, kTargetSize> res MOCHI_NO_INIT;
  ComputeLocalObjResDRes<kGradTarget, kCSize, kTargetSize>(
      info,
      dtStage,
      params.fittedSaturationHessian.constraintSaturation,
      AsConstView(c),
      AsConstView(cOld),
      AsConstView(jac),
      nullptr,
      &res,
      nullptr);

  // Global assembly
  AssembleGlobalObjResDRes<kTargetSize>(
      sparsity, info, linkActorReg, nullptr, &res, nullptr, outConstraintSnle);
}

template <GradTarget kGradTarget>
void mochi::AssembleConstraint(
    entt::registry& reg,
    entt::entity e,
    AssemblyParams const& params,
    CCompoundConstraintSnle& outConstraintSnle) {
  static_assert(
      kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous ||
          kGradTarget == GradTarget::CurrentInput || kGradTarget == GradTarget::PreviousInput,
      "Constraint assembly not needed for this target");
  MOCHI_ASSERT_VERBOSE(kGradTarget == params.gradTarget, "Inconsistent grad target");
  MOCHI_ASSERT_VERBOSE(
      kGradTarget == GradTarget::Current || (!params.assemObj && !params.assemDRes),
      "Unsupported assembly params for this grad target");

  // Early exit if the request is an input gradient and the constraint has no differentiable input
  if ((kGradTarget == GradTarget::CurrentInput || kGradTarget == GradTarget::PreviousInput) &&
      !reg.all_of<TagConstraintWithDifferentiableInput>(e)) {
    return;
  }

  auto const& type = reg.get<CConstraintInfo const>(e).type;

  static_assert(
      static_cast<int>(ConstraintType::Count) == 16,
      "Please update the switch statements below if ConstraintType enum changes");

// Macro for the assembly of each constraint type.
// If GradTarget::Current or GradTarget::Previous, invoke AssembleConstraintT.
// If GradTarget::CurrentInput or GradTarget::PreviousInput and the constraint type has target,
// invoke AssembleConstraintWrtInputT.
#define MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(kType)                                               \
  case kType:                                                                                     \
    if constexpr (kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous) {    \
      ecs::InvokeOnEntity<ecs::policy::AllowMutableExternalParams>(                               \
          AssembleConstraintT<kType, kGradTarget>,                                                \
          reg,                                                                                    \
          e,                                                                                      \
          std::cref(params),                                                                      \
          std::ref(outConstraintSnle));                                                           \
    } else if constexpr (!std::                                                                   \
                             is_same_v<typename ConstraintTraits<kType>::TargetType, NoTarget>) { \
      ecs::InvokeOnEntity<ecs::policy::AllowMutableExternalParams>(                               \
          AssembleConstraintWrtInputT<kType, kGradTarget>,                                        \
          reg,                                                                                    \
          e,                                                                                      \
          std::cref(params),                                                                      \
          std::ref(outConstraintSnle));                                                           \
    } else {                                                                                      \
      MOCHI_ASSERT_VERBOSE(false, "Constraint with differentiable input must have target");       \
    }                                                                                             \
    break;

  switch (type) {
    case ConstraintType::None:
      MOCHI_ASSERT(false, "Invalid constraint type");
      break;
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::ArticulatedSingleDofRange);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::Articulated3dRotationRange);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::ArticulatedSingleDofTarget);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::Articulated3dRotationTarget);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::JointRotationRange);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::JointRotationTracking);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::RodElementRotationToRigid);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::RigidPivotPosition);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::RigidPivotToRigidTarget);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::RigidPivotRotation);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::RigidPrismaticJoint);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::RigidSphericalJoint);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::DeformableNodePosition);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::DeformableNodeToRigid);
      MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT(ConstraintType::DeformableNodeToDeformableNode);
    default:
      MOCHI_ASSERT(false, "Unhandled constraint type");
      break;
  }
#undef MOCHI_TRY_INVOKE_ASSEMBLE_CONSTRAINT
}

template MOCHI_API void mochi::AssembleConstraint<GradTarget::Current>(
    entt::registry&,
    entt::entity,
    AssemblyParams const&,
    CCompoundConstraintSnle&);
template MOCHI_API void mochi::AssembleConstraint<GradTarget::Previous>(
    entt::registry&,
    entt::entity,
    AssemblyParams const&,
    CCompoundConstraintSnle&);
template MOCHI_API void mochi::AssembleConstraint<GradTarget::CurrentInput>(
    entt::registry&,
    entt::entity,
    AssemblyParams const&,
    CCompoundConstraintSnle&);
template MOCHI_API void mochi::AssembleConstraint<GradTarget::PreviousInput>(
    entt::registry&,
    entt::entity,
    AssemblyParams const&,
    CCompoundConstraintSnle&);

namespace mochi::constraint {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CConstraintGlobalSparsityCache>(reg);
  ecs::RegisterComponent<CConstraintGlobalInputSparsityCache>(reg);
  ecs::RegisterComponent<CConstraintInfo>(reg);
  ecs::RegisterComponent<CConstraintMemberInfo>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::RigidSphericalJoint>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::RigidPrismaticJoint>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::DeformableNodeToDeformableNode>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::DeformableNodeToRigid>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::DeformableNodePosition>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::JointRotationRange>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::JointRotationTracking>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::RodElementRotationToRigid>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::RigidPivotPosition>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::RigidPivotToRigidTarget>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::RigidPivotRotation>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::ArticulatedSingleDofTarget>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::Articulated3dRotationTarget>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::ArticulatedSingleDofRange>>(reg);
  ecs::RegisterComponent<CConstraintData<ConstraintType::Articulated3dRotationRange>>(reg);
  ecs::RegisterComponent<CConstraintTarget<real, TimeStep::Current>>(reg);
  ecs::RegisterComponent<CConstraintTarget<real, TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CConstraintTarget<Real3, TimeStep::Current>>(reg);
  ecs::RegisterComponent<CConstraintTarget<Real3, TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CConstraintTarget<Quaternion, TimeStep::Current>>(reg);
  ecs::RegisterComponent<CConstraintTarget<Quaternion, TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CConstraintTarget<TransformRT, TimeStep::Current>>(reg);
  ecs::RegisterComponent<CConstraintTarget<TransformRT, TimeStep::StageStart>>(reg);
}
} // namespace mochi::constraint
