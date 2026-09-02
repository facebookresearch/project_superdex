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

#include "mochi_common_components.h"
#include "mochi_contact_partition.h"
#include "mochi_ecs.h"
#include "mochi_pose_controller.h"
#include "mochi_rigid.h"
#include "mochi_shape.h"
#include "mochi_skinning.h"
#include "mochi_snle.h"

#include <mochi_physics/cpp_api/mochi_structs.h> // ArticulatedActorParams, ArticulatedSkinParams

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/articulated_body/transmission.h>
#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/utils/graph.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mochi {

// Forwards
struct CDofOffset;
struct CGroupMembers;
struct CIslandMemberInfo;

/*
Base data structures used to define articulated-body and articulated rigid components
*/

// Stores the actor handles of rigid bodies in the articulated body compound
struct CBoneHandles : NoCopy {
  std::vector<ActorHandle> bones;
};

// Data structure that stores the pose for an articulated body or articulated rigid. Used later to
// define components for articulated bodies and articulated rigids
struct ArticulatedPose {
  ArticulatedPose() = default;
  explicit ArticulatedPose(int size) : value(ColumnVector<real>::Zero(size)) {}
  explicit ArticulatedPose(ColumnVector<real> const& value) : value(value) {};
  explicit ArticulatedPose(ColumnVector<real>&& value) : value(std::move(value)) {};

  ColumnVector<real> value;

  MOCHI_STRUCT_BEGIN(mochi::ArticulatedPose);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

// Component to store a reference to the full pose of an articulated body
struct CArticulatedFullPoseRef : public VectorComponentRef {
  using VectorComponentRef::VectorComponentRef;
};

struct ArticulatedSkinningData {
  ColumnVector<real> restCoords;
  SkinningData skinningData;
  DSkinningTransform skinningTransform;
  SparseMatrix<real> jacobianDBones;
  RowMatrix<real> jacobianDJoints;
};

// Components registered to an articulated body actor
template <TimeStep kRelTime>
struct CArticulatedReducedPose : public ArticulatedPose, NoCopy {
  using ArticulatedPose::ArticulatedPose; // Inherit base class' constructors

  MOCHI_TEMPLATE_BEGIN(mochi::CArticulatedReducedPose, kRelTime);
  MOCHI_ATTRIBUTE_IF(kRelTime == TimeStep::Current, CaptureState);
  MOCHI_BASE_CLASS(ArticulatedPose);
  MOCHI_TEMPLATE_END();
};
struct CArticulatedFullPose : public ArticulatedPose, NoCopy {
  using ArticulatedPose::ArticulatedPose; // Inherit base class' constructors
};
template <TimeStep kRelTime>
struct CArticulatedJointTransforms : public std::vector<TransformRT>, NoCopy {
  using std::vector<TransformRT>::vector; // Inherit base class' constructors
};
template <TimeStep kRelTime>
struct CArticulatedLinkTransforms : public std::vector<TransformRT>, NoCopy {
  using std::vector<TransformRT>::vector; // Inherit base class' constructors
};
template <TimeStep kRelTime>
struct CArticulatedJointVels : NoCopy {
  CArticulatedJointVels() = default;
  explicit CArticulatedJointVels(int size) : value(size) {}
  std::vector<CRigidVel<kRelTime>> value;

  MOCHI_TEMPLATE_BEGIN(mochi::CArticulatedJointVels, kRelTime);
  MOCHI_ATTRIBUTE_IF(kRelTime == TimeStep::Current, CaptureState);
  MOCHI_FIELD(value);
  MOCHI_TEMPLATE_END();
};

/// @brief Component for time integration of articulated reduced pose.
struct CIntegrationArticulatedReducedPose : public IntegrationBundle<ArticulatedPose>, NoCopy {
  using IntegrationBundle<ArticulatedPose>::IntegrationBundle;

  MOCHI_STRUCT_BEGIN(mochi::CIntegrationArticulatedReducedPose);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_BASE_CLASS(IntegrationBundle<ArticulatedPose>);
  MOCHI_STRUCT_END();
};

/// @brief Component for time integration of articulated joint velocities.
struct CIntegrationArticulatedJointVels : public NoCopy {
  CIntegrationArticulatedJointVels() = default;
  explicit CIntegrationArticulatedJointVels(int numLinks) : value(numLinks) {}
  DynamicArray<IntegrationRigidVels> value;

  MOCHI_STRUCT_BEGIN(mochi::CIntegrationArticulatedJointVels);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

struct CArticulatedSkinningData : public ArticulatedSkinningData, NoCopy {};

struct CArticulatedRestTransforms : public articulated::RestTransformArray, NoCopy {};

// ECS component storing skin creation params that are consumed during InitSkinMesh and cannot
// be recovered from the ECS afterward. Used for lossless prefab export.
struct CArticulatedSkinExportParams : NoCopy {
  ActorBoundaryElementType boundaryElementType = ActorBoundaryElementType::Default;
  std::optional<BoundarySubsamplingParams> boundarySubsampling;
};

struct CArticulatedParents : public articulated::ParentIndexArray, NoCopy {};

// Component to store the pose size and offset of each joint. Dofs sizes and offsets are stored in
// CArticulatedBodyShape::shape->GetJointsData()->dofInfo
struct CArticulatedJointPoseInfo : public DynamicArray<ArticulatedPoseInfo>, NoCopy {};

struct CArticulatedProps : public ArticulatedProperties, NoCopy {};

struct CArticulatedInertiaParams : public std::vector<real>, NoCopy {};

struct CArticulatedJointFrictionParams : public std::vector<ArticulatedJointFrictionParams>,
                                         NoCopy {};

// Jacobian of link transforms wrt joint dofs. It is expressed in Lie algebra for 3D rotations.
struct CArticulatedJacobian : public NoCopy {
  RowMatrix<real> value;

  MOCHI_DECLARE_MOVE(CArticulatedJacobian);
  explicit CArticulatedJacobian(RowMatrix<real>&& valueIn) : value(std::move(valueIn)) {}
};

// Stores pointers to joint-limit constraints
struct CArticulatedJointLimits : public std::vector<Constraint*>, NoCopy {};

// Store cycle joint constraint handles
struct CArticulatedCycleJoints : public std::vector<ConstraintHandle>, NoCopy {};

// Components registered to an articulated rigid actor
struct CArticulatedEntity : public NoCopy {
  CArticulatedEntity(entt::entity e) : entity(e) {}
  entt::entity entity;
};

// Jacobian of link transform wrt joint dofs. It stores the rows of CArticulatedJacobian
// corresponding to a link, and only the non-zero columns (as values and dofs).
struct CArticulatedRigidJacobian : public NoCopy {
  Matrix<real> value;
  std::vector<int> dofs;

  MOCHI_DECLARE_MOVE(CArticulatedRigidJacobian);
  CArticulatedRigidJacobian(Matrix<real>&& valueIn, Span<int const> dofsIn)
      : value(std::move(valueIn)), dofs(dofsIn.begin(), dofsIn.end()) {}
};

struct CTransmissions : public NoCopy {
  DynamicArray<std::unique_ptr<Transmission>> transmissions;
};

// Emplace data to process a skinned mesh.
void InitSkinnedMesh(
    entt::registry& reg,
    entt::entity entity,
    entt::entity articulated,
    std::shared_ptr<Shape const> shape,
    Error& error);

// Emplace boundary discretization of a skinned mesh. Returns the number of quad points.
int InitDiscretizationSkinMesh(
    entt::registry& reg,
    entt::entity e,
    Shape const& shape,
    std::optional<BoundarySubsamplingParams> boundarySubsampling,
    ActorBoundaryElementType elementType,
    Error& error);

// Emplace data to process a colliding skinned mesh.
void InitCollidingSkinMesh(
    entt::registry& reg,
    entt::entity e,
    entt::entity articulated,
    Shape const& shape,
    Span<ContactPartitionStrategy const> strategies,
    ContactParams const& contactParams,
    int numCollidingSamples);

/*
  Systems for an articulated body
*/
namespace articulated::compound {

// General method to initialize an articulated actor.
void InitArticulatedBodyActor(
    entt::registry& reg,
    entt::entity e,
    ArticulatedActorParams const& params,
    bool useContact,
    std::shared_ptr<ArticulatedBodyShape const> articulatedShapePtr,
    std::shared_ptr<Shape const> skinMeshShape,
    Span<ActorHandle const> bones,
    Error& error);

// Validate that an articulated actor supports differentiability.
void ValidateDifferentiabilitySupport(entt::registry const& reg, entt::entity e, Error& error);

// Initialization specific to a differentiable scene
void InitDifferentiableActor(entt::registry& reg, entt::entity e, Error& error);

// Method to initialize the full-dof problem, necessary after an update to constraints on the bones
void InitFullDofProblem(entt::registry& reg, entt::entity e);

/*
 * Pipeline to set values of internal full-space and reduced pose from the state of the bones, and
 * trigger other derived updates. Snaps the links onto the joint manifold so they stay consistent
 * with the projected reduced pose.
 */
void SetArticulatedPoseFromLinks(entt::registry& reg, entt::entity e);

// Set the articulated body's dofs
void SetArticulatedBodyPose(
    entt::registry& reg,
    entt::entity e,
    Span<real const> pose,
    Error& error);

// Set the articulated body's root transform (world-from-root) and recompute derived state (the link
// world transforms depend on the root via forward kinematics).
void SetArticulatedRootTransform(
    entt::registry& reg,
    entt::entity e,
    TransformRT const& worldFromRoot);

// Get the link transforms
void GetLinkTransforms(
    entt::registry& reg,
    entt::entity e,
    Span<TransformRT> outWorldFromLinks,
    Error& error);

// Get the time derivatives of the articulated body's dofs. Velocity is implicitly represented by
// current and previous state, so it's computed by finite differences.
void GetArticulatedJointVelocities(
    entt::registry const& reg,
    entt::entity e,
    Span<real> outVel,
    Error& error);

// Set the time derivatives of the articulated body's dofs. The function sets the previous state of
// the actor's joints and the velocity of the bones.
void SetArticulatedJointVelocities(
    entt::registry& reg,
    entt::entity e,
    Span<real const> vel,
    Error& error);

// Add a pose controller (unless one already exists)
void AddPoseController(
    entt::registry& reg,
    entt::entity e,
    Scene* scene,
    PoseControllerParams const& params,
    Error& error);

// Initialize the pose controller for differentiability
void InitializeDifferentiablePoseController(entt::registry& reg, entt::entity e);

// Remove the pose controller with differentiability
void RemoveDifferentiablePoseController(entt::registry& reg, entt::entity e);

// Get the last target world-from-local link transforms set to the pose controller
void GetTargetLinkTransforms(
    entt::registry const& reg,
    entt::entity e,
    Span<TransformRT> outWorldFromTargets,
    Error& error);

// Set the target world-from-local link transforms for the pose controller
void SetTargetLinkTransforms(
    entt::registry& reg,
    entt::entity e,
    Span<TransformRT const> worldFromTargets,
    Error& error);

// Set the target pose (as joint dofs) for the pose controller
void SetTargetPose(entt::registry& reg, entt::entity e, Span<real const> pose, Error& error);

// Combine joint-level and link-level control targets into a single set of target link transforms.
// linkTransforms must contain root-frame transforms with 3D controls already applied.
// pose must contain the modified joint DOFs (with single-DOF controls already applied).
// Per-link bool spans indicate which links have position/rotation/joint controls.
// Position and rotation are merged independently per link: each is taken from the link target when
// controlled, and from forward kinematics otherwise. Descendants compose from the merged parent
// transform, so a partially controlled parent influences its whole subtree.
// On return, linkTransforms contains the combined root-frame target link transforms.
void CombinePoseAndLinkTargets(
    entt::registry& reg,
    entt::entity e,
    Span<real const> pose,
    Span<bool const> isLinkPosControlled,
    Span<bool const> isLinkRotControlled,
    Span<TransformRT> outLinkTransforms);

// Reset the target world-from-local link transforms for the pose controller
void ResetTargetLinkTransforms(
    entt::registry& reg,
    entt::entity e,
    Span<TransformRT const> worldFromTargets,
    Error& error);

// Reset the target pose (as joint dofs) for the pose controller
void ResetTargetPose(entt::registry& reg, entt::entity e, Span<real const> pose, Error& error);

// Set the target velocity (as time-derivatives of joint dofs) for the pose controller
void SetTargetJointVelocities(
    entt::registry& reg,
    entt::entity e,
    Span<real const> vel,
    Error& error);

// System to set the targets of the controller if the target velocity was imposed externally
void SetOldControllerTargets(
    entt::registry& reg,
    entt::entity e,
    ecs::CtxGlobal<CSceneTime const> time,
    CControllerTarget<TimeStep::Current> const& target,
    CControllerTargetVelocity& vel,
    CControllerTarget<TimeStep::Previous>& outTargetOld);

// Get the parameters of the pose-controller constraints.
// All three arrays are link-indexed and must be pre-sized to numLinks (the arrays are not resized,
// as they may be owned by another module). JointTracking params for links without a controllable
// joint are set to zero gains.
void GetPoseControllerParams(
    entt::registry& reg,
    entt::entity e,
    PoseControllerParams& outParams,
    Error& error);

// Set the parameters of the pose-controller constraints.
// Each array must be empty, size 1 (broadcast to all links), or size numLinks. An empty array will
// broadcast using a default-constructed PoseTrackingParams (zero gains) to all links.
void SetPoseControllerParams(
    entt::registry& reg,
    entt::entity e,
    PoseControllerParams const& params,
    Error& error);

// Query the force of the pose controller
Span<real const>
GetPoseControllerForce(entt::registry& reg, entt::entity e, Scene const* scene, Error& error);

// To be run pre-simulation step for all articulated body actors
void PreStepArticulatedBodyActorAsync(entt::registry& reg, entt::entity e);

/*
 * Pipeline to update quantities that are a function of the state (aka derived state) of the
 * articulated body actor and make them consistent with the state.
 */
void UpdateDerivedStatePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Pipelines to update articulated Jacobians, based on state or target input.
 */
template <TimeStep kTimeStep>
void UpdateJacobiansStatePipeline(entt::registry& reg, Span<entt::entity const> entities);

template <TimeStep kTimeStep>
void UpdateJacobiansInputPipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Update the Jacobian from joint DoFs to full-space DoFs (DBones/DJoints), evaluated on the state
 */
template <TimeStep kTimeStep>
void UpdateJacobianState(
    ecs::Included<TagArticulatedActor>,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointTransforms<kTimeStep> const& jointTransforms,
    CArticulatedLinkTransforms<kTimeStep> const& linkTransforms,
    CArticulatedJacobian& outJacobian) {
  MOCHI_PROFILE_SCOPE();
  auto const* joints = bodyShape.shape->GetJointsData();
  Jacobian(
      joints->jointTypes,
      parents,
      joints->jointAxes,
      joints->dofInfo,
      restTransforms,
      rootTransform.worldFromLocal,
      jointTransforms,
      linkTransforms,
      outJacobian.value);
}

/*
 * System to update the displacements of the skinning. Templatized according to time step type.
 */
template <TimeStep kStep, bool kForceUseAllNodes = false>
void ResolveSkinning(
    CArticulatedLinkTransforms<kStep> const& linkTransforms,
    CArticulatedSkinningData const& skinningData,
    CActiveUniqueNodes const* activeNodes,
    CDisplacementSlice<real, kStep, DisplacementLayer::Skinned>& outDisplacements);

/*
 * Function to update the Jacobian of some skinned data w.r.t. the bone dofs (if one exists).
 * It is called from the method computing the Jacobian w.r.t. to the joint dofs.
 * Note that in this case template does not require time step type or mesh type since Jacobian is
 * always computed for the current step and stored internally in the skinning data structure.
 */
void ResolveSkinningJacobianDBones(
    Span<TransformRT const> linkTransforms,
    ColumnVectorView<real const> unposedCoords,
    ArticulatedSkinningData& skinningData,
    CActiveUniqueNodes const* activeNodes = nullptr);

/*
 * System to update the Jacobian of some skinned data w.r.t. the joint dofs (if one exists).
 * It is called within each nonlinear solver iteration.
 * Note that in this case template does not require time step type or mesh type since Jacobian is
 * always computed for the current step and stored internally in the skinning data structure.
 */
void ResolveSkinningJacobianDJoints(
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    CArticulatedJacobian const& articulatedJacobian,
    CActiveUniqueNodes const* activeNodes,
    CArticulatedSkinningData& skinningData);

/*
 * System to compute contact Jacobians as colliding actor. It is called after the collision
 * detection pipeline, within each nonlinear solver iteration.
 */
template <typename DiscretizationT>
void SetupCollidingJacobians(
    ecs::Included<TagArticulatedActor>,
    ecs::Excluded<TagBlendedActor>,
    DiscretizationT const& discretization,
    CDofOffset const& dofOffset,
    CArticulatedSkinningData const& skinningData,
    CContactPartitions const& contactPartitions,
    CCollJacs<CollRole::Colliding>& outJacobians);

/**
 * System to assemble the objective, residual and dresidual of the damping forces acting on joints
 * of the articulated body. Internally called by EntityAssemble.
 */
void AssembleDampingForces(
    AssemblyParams const& params,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CTimeIntegratorState const& intState,
    CActorSnle& outCompoundSnle);

/**
 * System to assemble the objective, residual, and dresidual of forces from transmissions attached
 * to the articulated body.
 */
void AssembleTransmissionForces(
    AssemblyParams const& params,
    CArticulatedLinkTransforms<TimeStep::Current> const& currLinkTxs,
    CArticulatedLinkTransforms<TimeStep::StageStart> const& stageStartLinkTxs,
    CArticulatedJacobian const& jacobian,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CTimeIntegratorState const& intState,
    CTransmissions const& transmissions,
    CActorSnle& outCompoundSnle);

/**
 * Compute the per-link world-from-CoM transforms for the given reduced pose by running forward
 * kinematics. Reads the entity's joint info / parents / rest transforms / root transform from the
 * registry. `outLinkTransformsCom.size()` must equal the number of links.
 */
void GetLinkTransformsComFromPose(
    entt::registry const& reg,
    entt::entity e,
    Span<real const> pose,
    Span<TransformRT> outLinkTransformsCom);

/**
 * System to assemble the objective, residual and dresidual of the inertia forces acting on joints
 * of the articulated body. Internally called by EntityAssemble.
 */
void AssembleInertiaForces(
    AssemblyParams const& params,
    ecs::OptionalTag<TagUseNewtonEulerInertia> useNewtonEulerInertia,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedInertiaParams const& inertiaParams,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedJointVels<TimeStep::StageStart> const& stageStartJointVels,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CTimeIntegratorState const& intState,
    CActorSnle& outCompoundSnle);

/**
 * System to assemble the objective, residual and dresidual of the friction forces acting on joints
 * of the articulated body. Internally called by EntityAssemble.
 */
void AssembleFrictionForces(
    AssemblyParams const& params,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CTimeIntegratorState const& intState,
    CActorSnle& outCompoundSnle);

/**
 * System to assemble the objective, residual and dresidual of external forces acting on joint dofs.
 * Internally called by EntityAssemble.
 */
void AssembleExternalForces(
    AssemblyParams const& params,
    CArticulatedProps const& props,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CExternalForces const& externalForces,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CActorSnle& outCompoundSnle);

/*
 * System to copy a span of reals to the position state of the articulated compound actor (i.e.
 * CArticulatedReducedPose<TimeStep::Current>). It also updates CArticulatedFullPose (derived
 * state). The span of reals is usually the components of the non-linear problem solution vector
 * corresponding to the actor, thereby the name.
 */
void EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::RequiredTag<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CArticulatedReducedPose<TimeStep::Current>& outReducedPose,
    CArticulatedJointTransforms<TimeStep::Current>& outJointTransforms,
    CArticulatedLinkTransforms<TimeStep::Current>& outLinkTransforms,
    CArticulatedFullPose& outFullPose);

/*
 * System to copy the position state of the articulated compound actor (i.e.
 * CArticulatedReducedPose<TimeStep::Current>) to a span of reals. The span of reals is usually the
 * components of the non-linear problem solution vector corresponding to the actor, thereby the
 * name.
 */
void EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::RequiredTag<TagArticulatedActor>,
    CArticulatedReducedPose<TimeStep::Current> const& reducedPose);

/*
 * System executed after the solution of the non-linear problem is updated. It MUST update the
 * position components of the articulated actor state (aka position state) to make it
 * consistent with the new solution. It may OPTIONALLY update other quantities that are a function
 * of the state (aka derived state). In particular, it updates:
 * 1. CArticulatedReducedPose<TimeStep::Current> (position state)
 * 2. CArticulatedFullPose (derived state)
 * 3. CArticulatedJointTransforms<TimeStep::Current> (derived state)
 * 4. CArticulatedLinkTransforms<TimeStep::Current> (derived state)
 * Any derived state that is required for assembly and not updated in this system MUST be
 * updated in EntityAssemble or in mochi_solve's UpdateDerivedStateBeforeAssembly.
 */
void EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::RequiredTag<TagArticulatedActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& dofInfo,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CArticulatedJointTransforms<TimeStep::Current>& jointTransforms,
    CArticulatedLinkTransforms<TimeStep::Current>& linkTransforms,
    CArticulatedFullPose& fullPose,
    CArticulatedReducedPose<TimeStep::Current>& reducedPose);

/*
 * System executed after the solution increment of the non-linear problem is updated. It MUST update
 * the position components of the articulated actor state (aka position state) to make it consistent
 * with the new solution. It may OPTIONALLY update other quantities that are a function of the state
 * (aka derived state). In particular, it updates:
 * 1. CArticulatedReducedPose<TimeStep::Current> (position state)
 * 2. CArticulatedFullPose (derived state)
 * 3. CArticulatedJointTransforms<TimeStep::Current> (derived state)
 * 4. CArticulatedLinkTransforms<TimeStep::Current> (derived state)
 * Any derived state that is required for assembly and not updated in this system MUST be
 * updated in EntityAssemble or in mochi_solve's UpdateDerivedStateBeforeAssembly.
 */
void EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::RequiredTag<TagArticulatedActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& dofInfo,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CDirichletBC<real> const& dirichlet,
    CArticulatedJointTransforms<TimeStep::Current>& jointTransforms,
    CArticulatedLinkTransforms<TimeStep::Current>& linkTransforms,
    CArticulatedFullPose& fullPose,
    CArticulatedReducedPose<TimeStep::Current>& reducedPose);

// mochi::articulated::compound
void EntityAssemble(
    AssemblyParams const& params,
    ecs::RequiredTag<TagArticulatedActor>,
    ecs::PartialRegistry<CActorSnle const, CDofOffset const> reg,
    ecs::OptionalTag<TagUseNewtonEulerInertia> useNewtonEulerInertia,
    CArticulatedProps const& props,
    CGroupMembers const& groupMembers,
    CArticulatedJacobian const& jacobian,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CTimeIntegratorState const& intState,
    CArticulatedInertiaParams const& inertiaParams,
    CArticulatedJointFrictionParams const& frictionParams,
    CCompoundConstraintSnle const* constraintFullSnle,
    CExternalForces const& externalForces,
    CTransmissions const& transmissions,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedLinkTransforms<TimeStep::StageStart> const& stageStartLinkTxs,
    CArticulatedLinkTransforms<TimeStep::Current> const& currLinkTxs,
    CArticulatedJointVels<TimeStep::StageStart> const& stageStartJointVels,
    CActorDerivedStateInfo const* derivedStateInfo,
    CActorDiffInputInfo const* diffInputInfo,
    CActorSnle& outCompoundSnle);

/*
 * Pipeline executed before each time step.
 */
void PreStepPipeline(entt::registry& reg);

/*
 * Executed before the first time integration stage of the time step.
 */
void EntityPreFirstStage(
    ecs::Included<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CTimeIntegratorState const& intState,
    CArticulatedReducedPose<TimeStep::Previous> const& prevPose,
    CArticulatedJointVels<TimeStep::Previous> const& prevJointVels,
    CIntegrationArticulatedReducedPose& outIntPose,
    CIntegrationArticulatedJointVels& outIntJointVels);

/*
 * Pipeline executed before each time integration stage.
 */
void PreStagePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Pipeline executed after each time integration stage.
 */
void PostStagePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Pipeline executed after the last time integration stage of the time step.
 */
void PostLastStagePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * ECS system to record the target state of an articulated pose controller.
 */
void RecordTargetState(
    CArticulatedProps const& articulatedProps,
    CControllerTarget<TimeStep::Current> const& controllerTarget,
    CControllerConstraints const& controllerConstraints,
    entt::registry& reg,
    entt::entity e,
    CRecordingData& outData);

/*
 * ECS system to record the current state of an articulated actor.
 */
void RecordState(
    CArticulatedReducedPose<TimeStep::Current> const& reducedPose,
    CArticulatedJointVels<TimeStep::Current> const& jointVels,
    CControllerTarget<TimeStep::Previous> const* targetOld,
    CRecordingData& outData);

/*
 * ECS system to possibly update vsym of joint velocities at the beginning of a time step.
 */
void UpdateVSym(
    ecs::Included<TagArticulatedActor>,
    ecs::CtxGlobal<CSceneTime const> time,
    CArticulatedJointVels<TimeStep::Current>& outJointVels);

/*
 * Helper function to produce a SkinningParams data structure for an articulated body
 */
SkinningParams
CreateSkinningParams(entt::registry& reg, entt::entity e, bool allowUnusedBones, Error& error);

/*
 * [Differentiability] System to project a derived state gradient to a state gradient.
 * It computes dg/dq = dg/dx * dx/dq.
 */
void ProjectDerivedStateGradient(
    ecs::Included<TagArticulatedActor>,
    CArticulatedJacobian const& jacobian,
    CDiffContainerDerivedState const& derivedStateGrad,
    CDiffContainerState& outStateGrad);

/*
 * [Differentiability] System to shift a derived state gradient.
 * It computes dg/dxold = dg/dDeltax * dDeltax/dxold
 */
void ShiftDerivedStateGradient(
    ecs::PartialRegistry<
        CRigidState<TimeStep::Current> const,
        CRigidState<TimeStep::Previous> const> reg,
    ecs::Included<TagArticulatedActor>,
    CArticulatedProps const& props,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CGroupMembers const& members,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::Previous> const& prevPose,
    CDiffContainerDerivedState& outDerivedStateGrad);

/*
 * System to transport a gradient according to Lie derivatives.
 */
void TransportGradient(
    ColumnVectorView<real const> delta,
    ColumnVectorView<real> outGradient,
    ecs::Included<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CDofOffset const& dofOffset,
    CActorDofInfo const& dofInfo);

/*
 * System to project contact-force adjoints from the links to the articulated actor.
 */
template <GradTarget kGradTarget>
void ProjectContactForceAdjoints(
    ecs::PartialRegistry<CDofOffset const, CDiffContactGrad<kGradTarget> const> reg,
    ecs::RequiredTag<TagArticulatedActor>,
    CGroupMembers const& groupMembers,
    CArticulatedJacobian const& jacobian,
    CDiffContactGrad<kGradTarget>& outGrad);

// Update CBoundingVolume<TimeStep::Current>.localShape from the deformation of a triangular skin
// mesh on a compound (articulated/blended) actor.
template <TimeStep kStep>
void UpdateBounds(
    ecs::RequiredTag<TagCompoundActor>,
    CTriangularMesh const& meshComponent,
    CFinalDisplacementRef<kStep> const& solComponent,
    CBoundingVolume<TimeStep::Current>& outBounds);

} // namespace articulated::compound

/*
Systems for an articulated rigid
*/
namespace articulated::rigid {

/*
 * System to update Jacobian
 */
void EntityUpdateJacobian(
    ecs::Included<TagArticulatedLinkActor>,
    ecs::PartialRegistry<CArticulatedJacobian const> reg,
    CArticulatedEntity const& entArticulated,
    CDofOffset const& dofOffset,
    CActorDofInfo const& numDofs,
    CArticulatedRigidJacobian& outJacobian);

// Compute the contact Jacobians as colliding actor
template <TimeStep kTimeStep>
MOCHI_FORCE_INLINE void SetupCollidingJacobians(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CRigidState<kTimeStep> const& state,
    CRootTransform const& transform,
    ecs::PartialRegistry<CDofOffset const> reg,
    CArticulatedEntity const& entArticulated,
    CArticulatedRigidJacobian const& jacobianBody,
    CContactSamples<TimeStep::Current> const& samples, // Not templatized, they are constant
    CCollJacs<CollRole::Colliding>& outJacobians) {
  static_assert(kTimeStep == TimeStep::Current || kTimeStep == TimeStep::StageStart);
  mochi::rigid::SetupCollidingJacobiansImpl(
      state.value,
      kTimeStep == TimeStep::Current ? transform.worldFromLocal
                                     : transform.worldFromLocalStageStart,
      reg.get<CDofOffset const>(entArticulated.entity),
      samples,
      outJacobians,
      jacobianBody.value,
      jacobianBody.dofs);
}

// Compute the contact Jacobians as collider actor
template <TimeStep kTimeStep>
MOCHI_FORCE_INLINE void SetupColliderJacobians(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CRigidState<kTimeStep> const& state,
    ecs::PartialRegistry<CDofOffset const> reg,
    CArticulatedEntity const& entArticulated,
    CArticulatedRigidJacobian const& jacobianBody,
    CRigidBodyInertia const& rigidInertia,
    CCollJacs<CollRole::Collider>& outJacobians) {
  mochi::rigid::SetupColliderJacobiansImpl(
      state.value,
      reg.get<CDofOffset const>(entArticulated.entity),
      rigidInertia,
      outJacobians,
      jacobianBody.value,
      jacobianBody.dofs);
}

/*
 * Copies the position state of the articulated actor to the position state of the underlying rigid
 * actors.
 */
void EntitySetSolution(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CDofOffset const& rigidDofOffset,
    CArticulatedFullPoseRef const& fullPoseRef,
    CRigidState<TimeStep::Current>& outCurrPose);

void EntityPreStep(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CRigidState<TimeStep::Previous>& prevPose,
    CRigidVel<TimeStep::Previous>& prevVel);

void EntityPreFirstStage(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CTimeIntegratorState const& intState,
    CRigidVel<TimeStep::Previous> const& prevVel,
    CIntegrationRigidVels& intVels);

void EntityPreStage(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CDofOffset const& rigidDofOffset,
    CRigidBodyInertia const& rigidInertia,
    CArticulatedFullPoseRef const& fullPoseRef,
    CTimeIntegratorState const& intState,
    CIntegrationRigidVels& intVels,
    CRigidState<TimeStep::StageStart>& stageStartPose,
    CRigidVel<TimeStep::StageStart>& stageStartVel,
    CRootTransform& rootTransform);

void EntityPostStage(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CConvergenceStatus const& convergence,
    CDofOffset const& rigidDofOffset,
    CTimeIntegratorState const& intState,
    CArticulatedFullPoseRef const& fullPoseRef,
    CRigidState<TimeStep::StageStart> const& stageStartPose,
    CRigidState<TimeStep::Current>& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CIntegrationRigidVels& intRigidVels);

void EntityPostLastStage(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CRigidBodyInertia const& rigidInertia,
    CDofOffset const& rigidDofOffset,
    CTimeIntegratorState const& intState,
    CArticulatedFullPoseRef const& fullPoseRef,
    CRigidState<TimeStep::Current>& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CIntegrationRigidVels& intVels,
    CRootTransform& rootTransform);

/*
 * System executed after the solution of the non-linear problem is updated. It MUST update the
 * pose of the articulated rigid actor (aka position state) to make it consistent with the new
 * solution. It may OPTIONALLY update other quantities that are a function of the state (aka derived
 * state). In particular, it updates:
 * 1. CRigidState (position state)
 * 2. CRootTransform (derived state)
 * Any derived state that is required for assembly and not updated in this system MUST be
 * updated in EntityAssemble or in mochi_solve's UpdateDerivedStateBeforeAssembly.
 */
void EntityPostNewSolution(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    ecs::Excluded<TagStaticActor>,
    CRigidBodyInertia const& rigidInertia,
    CDofOffset const& rigidDofOffset,
    CArticulatedFullPoseRef& fullPoseRef,
    CRigidState<TimeStep::Current>& rigidPose,
    CRootTransform& rootTransform);

} // namespace articulated::rigid

namespace articulated {
void InitializeOnce(entt::registry& reg);

// Get the mass of an articulated actor.
[[nodiscard]] real GetActorMass(entt::registry const& reg, entt::entity actor);
} // namespace articulated

} // namespace mochi
