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
#include "mochi_contact.h"
#include "mochi_differentiable.h"
#include "mochi_ecs.h"
#include "mochi_shape.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/utils/rigid_body_utils.h>

#include <memory>
#include <optional>

namespace mochi {

/**************************************************************************
  ECS components
*/

/*
  Stores a velocity slice
*/
struct RigidBodyVelContainer {
  RigidBodyVelContainer() = default;
  explicit RigidBodyVelContainer(RigidBodyVel const& valueIn) : value(valueIn) {}

  RigidBodyVel value;

  MOCHI_STRUCT_BEGIN(mochi::RigidBodyVelContainer);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

template <TimeStep kRelTime>
struct CRigidVel : public RigidBodyVelContainer {
  MOCHI_TEMPLATE_BEGIN(mochi::CRigidVel, kRelTime);
  MOCHI_ATTRIBUTE_IF(kRelTime == TimeStep::Current, CaptureState)
  MOCHI_BASE_CLASS(mochi::RigidBodyVelContainer);
  MOCHI_TEMPLATE_END();
};

/// @brief Component for time integration of rigid velocity.
using IntegrationRigidVels = IntegrationBundle<RigidBodyVelContainer>;
struct CIntegrationRigidVels : public IntegrationRigidVels, NoCopy {
  MOCHI_STRUCT_BEGIN(mochi::CIntegrationRigidVels);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_BASE_CLASS(IntegrationRigidVels);
  MOCHI_STRUCT_END();
};

// ECS component storing rigid actor creation params that are consumed during InitRigidActor
// and cannot be recovered from the ECS afterward. Used for lossless prefab export.
struct CRigidExportParams : NoCopy {
  ActorBoundaryElementType boundaryElementType = ActorBoundaryElementType::Default;
  std::optional<BoundarySubsamplingParams> boundarySubsampling;
};

/**************************************************************************
  ECS Rigid-body Actor Utils
*/

// Initialize a rigid-body actor (static or dynamic)
void InitRigidActor(
    entt::registry& reg,
    entt::entity e,
    RigidActorParams const& params,
    bool useContact,
    std::shared_ptr<Shape const> shapePtr,
    Error& error);

// Initialization specific to a differentiable scene
void InitDifferentiableRigidActor(entt::registry& reg, entt::entity e, bool isArticulatedLink);

// To be run pre-simulation step for all rigid actors
void PreStepRigidActorAsync(entt::registry& reg, entt::entity e);

namespace rigid {

void SetRootTransform(
    TransformRT const& worldFromLocal,
    CRigidBodyInertia const& rbInertia,
    CRootTransform& outRootTransform,
    CRigidState<TimeStep::Current>& outCurrState);

// Update CRootTransform from rigid body state
MOCHI_API void RigidStateToRootTransform(
    Vec4r const& comLocal,
    TransformRT const& state,
    TransformRT& worldFromLocal);

void ComputeRootTransformAtStageStart(
    CRigidBodyInertia const& rigidInertia,
    CRigidState<TimeStep::StageStart> const& stageStartPose,
    CRootTransform& rootTransform);

void ComputeRootTransformCurrent(
    CRigidBodyInertia const& rigidInertia,
    CRigidState<TimeStep::Current> const& currPose,
    CRootTransform& rootTransform);

void ComputeVelocityAtStepStart(
    CTimeIntegratorState const& intState,
    CRigidVel<TimeStep::Previous> const& prevVel,
    CIntegrationRigidVels& intVels);

void ComputeVelocityAtStageStart(
    CTimeIntegratorState const& intState,
    CIntegrationRigidVels& intVels,
    CRigidVel<TimeStep::StageStart>& stageStartVel);

void ComputeVelocityAtStageEnd(
    CTimeIntegratorState const& intState,
    CRigidState<TimeStep::StageStart> const& stageStartPose,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidVel<TimeStep::Current>& currVel);

void ComputeVelocityAtTimeStepEnd(
    CTimeIntegratorState const& intState,
    CIntegrationRigidVels& intVels,
    CRigidVel<TimeStep::Current>& currVel);

// Call UpdateRigidVelocity for static rigid actors
inline void UpdateRigidVelocity_Static(
    ecs::RequiredTag<TagStaticActor>,
    ecs::CtxGlobal<CSceneTime const> time,
    CRootTransform const& root,
    CPrevRigidVelocity& outRigidVelocity) {
  ComputeRigidVelocityWorldSpace(
      static_cast<real>(time->DeltaTime()),
      root.worldFromLocal,
      root.worldFromLocalPrev,
      outRigidVelocity.centerOfMassLocal,
      outRigidVelocity.linearVelocityWorld,
      outRigidVelocity.angularVelocityWorld);
}

// Call UpdateRigidVelocity for dynamic rigid actors
inline void UpdateRigidVelocity_Dynamic(
    ecs::RequiredTag<TagRigidActor>,
    ecs::Excluded<TagStaticActor>,
    CRigidVel<TimeStep::Current> const& currVel,
    CPrevRigidVelocity& outRigidVelocity) {
  outRigidVelocity.linearVelocityWorld = currVel.value.GetVCom();
  outRigidVelocity.angularVelocityWorld = currVel.value.GetOmegaAndVSym().first;
}

// System to initialize state (position and velocity) for a new step
MOCHI_API void EntityIncrementStep(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CRigidState<TimeStep::Previous>& prevPose,
    CRigidVel<TimeStep::Previous>& prevVel);

// Pre-first-stage callback.
MOCHI_API void EntityPreFirstStage(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CTimeIntegratorState const& intState,
    CRigidState<TimeStep::Previous> const& prevPose,
    CRigidVel<TimeStep::Previous> const& prevVel,
    CIntegrationRigidStates& intPoses,
    CIntegrationRigidVels& intVels);

// Pre-stage callback.
MOCHI_API void EntityPreStage(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidBodyInertia const& rigidInertia,
    CTimeIntegratorState const& intState,
    CIntegrationRigidStates& intPoses,
    CIntegrationRigidVels& intVels,
    CRigidState<TimeStep::StageStart>& stageStartPose,
    CRigidVel<TimeStep::StageStart>& stageStartVel,
    CRootTransform& rootTransform);

// Post-stage callback.
MOCHI_API void EntityPostStage(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CConvergenceStatus const& convergence,
    CTimeIntegratorState const& intState,
    CRigidBodyInertia const& rigidInertia,
    CRigidState<TimeStep::Previous> const& prevPose,
    CRigidState<TimeStep::StageStart> const& stageStartPose,
    CRootTransform& rootTransform,
    CRigidState<TimeStep::Current>& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CIntegrationRigidStates& intPoses,
    CIntegrationRigidVels& intVels);

// Post-last-stage callback.
MOCHI_API void EntityPostLastStage(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidBodyInertia const& rigidInertia,
    CTimeIntegratorState const& intState,
    CIntegrationRigidStates& intPoses,
    CIntegrationRigidVels& intVels,
    CRigidState<TimeStep::Current>& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CRootTransform& rootTransform);

// Set up kinematic data of all active collision points; specific for rigid actors
template <ContactType kContactType>
void SetupActiveCollisionNormals(
    ecs::RequiredTag<TagRigidActor>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CContactSamples<TimeStep::Current> const& samples,
    CRootTransform const& rootTransform,
    CActiveCollisions<kContactType, TimeStep::Current>& activeCollisions);

// Compute the contact Jacobians as colliding actor
void SetupCollidingJacobiansImpl(
    TransformRT const& state,
    TransformRT const& transform,
    CDofOffset const& dofOffset,
    CContactSamples<TimeStep::Current> const& samples,
    CCollJacs<CollRole::Colliding>& outJacobians,
    MatrixView<real const> jacAux = {},
    Span<int const> dofsAux = {});

template <TimeStep kTimeStep>
MOCHI_FORCE_INLINE void SetupCollidingJacobians(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidState<kTimeStep> const& state,
    CRootTransform const& transform,
    CDofOffset const& dofOffset,
    CContactSamples<TimeStep::Current> const& samples, // Not templatized, they are constant
    CCollJacs<CollRole::Colliding>& outJacobians) {
  static_assert(kTimeStep == TimeStep::Current || kTimeStep == TimeStep::StageStart);
  SetupCollidingJacobiansImpl(
      state.value,
      kTimeStep == TimeStep::Current ? transform.worldFromLocal
                                     : transform.worldFromLocalStageStart,
      dofOffset,
      samples,
      outJacobians);
}

// Compute the contact Jacobians as collider actor
void SetupColliderJacobiansImpl(
    TransformRT const& state,
    CDofOffset const& dofOffset,
    CRigidBodyInertia const& rigidInertia,
    CCollJacs<CollRole::Collider>& outJacobians,
    MatrixView<real const> jacAux = {},
    Span<int const> dofsAux = {});

template <TimeStep kTimeStep>
MOCHI_FORCE_INLINE void SetupColliderJacobians(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidState<kTimeStep> const& state,
    CDofOffset const& dofOffset,
    CRigidBodyInertia const& rigidInertia,
    CCollJacs<CollRole::Collider>& outJacobians) {
  SetupColliderJacobiansImpl(state.value, dofOffset, rigidInertia, outJacobians);
}

/*
 * System executed when the solution of the non-linear problem is set. It MUST update the
 * position components of the rigid actor's internal state to make them consistent with the new
 * solution. It may OPTIONALLY update other quantities that are a function of the state (aka derived
 * state). The current implementation updates:
 * 1. CRigidState (state)
 * 2. CRootTransform (derived state)
 * Any derived state that is required for assembly and not updated in this system MUST be
 * updated in EntityAssemble or in mochi_solve's UpdateDerivedStateBeforeAssembly.
 */
MOCHI_API void EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CDofOffset const& dofOffset,
    CRigidBodyInertia const& rigidInertia,
    CRigidState<TimeStep::Current>& rigidState,
    CRootTransform& rootTransform);

/*
 * System executed after the solution increment of the non-linear problem is updated. It MUST update
 * the position components of the rigid actor's internal state to make them consistent with the new
 * solution. It may OPTIONALLY update other quantities that are a function of the state (aka derived
 * state). The current implementation updates:
 * 1. CRigidState (state)
 * 2. CRootTransform (derived state)
 * Any derived state that is required for assembly and not updated in this system MUST be
 * updated in EntityAssemble or in mochi_solve's UpdateDerivedStateBeforeAssembly.
 */
void EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CDofOffset const& dofOffset,
    CRigidBodyInertia const& rigidInertia,
    CDirichletBC<real> const& dirichlet,
    CRigidState<TimeStep::Current>& rigidState,
    CRootTransform& rootTransform);

// Assemble the full DOFs of the rigid body. Includes gravity, inertia, and contact.
void EntityAssemble(
    AssemblyParams const& params,
    entt::entity e,
    CActorSnle& outActorSnle,
    ecs::Included<TagRigidActor>,
    ContactAssemblyReg reg,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    ecs::OptionalTag<TagUseInertia> hasInertiaTag,
    ecs::OptionalTag<TagUseContact> hasContactTag,
    ecs::OptionalTag<TagUseNewtonEulerInertia> useNewtonEulerInertia,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CTimeIntegratorState const& intState,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidState<TimeStep::StageStart> const& stageStartPose,
    CRigidVel<TimeStep::StageStart> const& stageStartVel,
    CRigidBodyInertia const& rigidInertia,
    CExternalForces const& externalForces,
    CColliderInfo const& colliderInfo,
    CContactSamples<TimeStep::Current> const* contactSample,
    CActiveCollisions<ContactType::Async, TimeStep::Current>* activeCollisions);

/*
 * System to copy a column vector with the position state of the rigid actor to its CRigidState
 * component. The column vector usually contains the components of the non-linear problem solution
 * vector corresponding to the actor, thereby the name.
 */
template <TimeStep kTimeStep>
void EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidState<kTimeStep>& outPose) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(solution.size() == RigidSize::kAll, "Invalid data size");
  outPose.value = TransformFromRawPose(solution.TopRows<RigidSize::kAll>(RigidSize::kAll));
}

/*
 * System to copy the position state of the rigid actor (i.e. CRigidState) to a column vector. The
 * column vector usually represents the components of the non-linear problem solution vector
 * corresponding to the actor, thereby the name.
 */
void EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidState<TimeStep::Current> const& state);

/*
 * ECS system to record the current state of a rigid actor.
 */
void RecordState(
    ecs::Included<TagRigidActor>,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidVel<TimeStep::Current> const& currVel,
    CRecordingData& outData);

/*
 * ECS system to possibly update vsym of the rigid velocity at the beginning of a time step.
 */
void UpdateVSym(
    ecs::Included<TagRigidActor>,
    ecs::CtxGlobal<CSceneTime const> time,
    CRigidVel<TimeStep::Current>& outVel);

/*
 * [Differentiability] System to project a derived state gradient to a state gradient.
 * It computes dg/dq = dg/dx * dx/dq.
 * This is trivial for a rigid actor, where the derived state matches the state.
 */
MOCHI_FORCE_INLINE void ProjectDerivedStateGradient(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CDiffContainerDerivedState const& derivedStateGrad,
    CDiffContainerState& outStateGrad) {
  AsView(outStateGrad) = derivedStateGrad;
}

/*
 * [Differentiability] System to shift a derived state gradient.
 * It computes dg/dxold = dg/dDeltax * dDeltax/dxold
 */
MOCHI_FORCE_INLINE void ShiftDerivedStateGradient(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidState<TimeStep::Previous> const& prevPose,
    CDiffContainerDerivedState& outDerivedStateGrad) {
  ChainRigidGradientDDeltaDOld(
      currPose.value,
      prevPose.value,
      outDerivedStateGrad.TopRows<RigidSize::kDAll>(RigidSize::kDAll));
}

/*
 * System to transport a gradient according to Lie derivatives.
 */
void TransportGradient(
    ColumnVectorView<real const> delta,
    ColumnVectorView<real> outGradient,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CDofOffset const& dofOffset);

void InitializeOnce(entt::registry& reg);

// Get the mass of a rigid actor.
[[nodiscard]] real GetActorMass(entt::registry const& reg, entt::entity actor);

} // namespace rigid

} // namespace mochi
