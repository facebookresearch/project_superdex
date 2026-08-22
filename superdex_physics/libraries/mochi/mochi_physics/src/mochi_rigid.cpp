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

#include "mochi_rigid.h"

#include "mochi_actor_convergence.h"
#include "mochi_common_components.h"
#include "mochi_contact.h"
#include "mochi_contact_filter.h"
#include "mochi_differentiable.h"
#include "mochi_discretization_components.h"
#include "mochi_discretization_functions.h"
#include "mochi_ecs_utils.h"
#include "mochi_integration.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/contact/dmap.h>
#include <mochi_core/geometry/any_shape.h>
#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/utils/differentiability.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/rigid_body_assembly.h>
#include <mochi_core/utils/rigid_body_utils.h>
#include <mochi_core/utils/spmat_utils.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::dmap;

/*****************************************************************************************************
  Utilities
*/

void rigid::EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidState<TimeStep::Current> const& state) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(outSolution.size() == RigidSize::kAll, "Invalid data size");
  TransformToRawPose(state.value, outSolution.TopRows<RigidSize::kAll>(RigidSize::kAll));
}

MOCHI_API void rigid::RigidStateToRootTransform(
    Vec4r const& comLocal,
    TransformRT const& state,
    TransformRT& worldFromLocal) {
  auto const& rot = state.GetRotation();
  worldFromLocal.SetRotation(rot);
  worldFromLocal.SetTranslation(state.GetTranslation() - ToReal3(rot * comLocal));
}

void rigid::SetRootTransform(
    TransformRT const& worldFromLocal,
    CRigidBodyInertia const& rbInertia,
    CRootTransform& outRootTransform,
    CRigidState<TimeStep::Current>& outCurrState) {
  outRootTransform.worldFromLocal = worldFromLocal;
  outCurrState.value.SetTranslation(
      worldFromLocal.TransformPoint(rbInertia.GetCenterOfMassLocal()));
  outCurrState.value.SetRotation(worldFromLocal.GetRotation());
}

template <GradTarget kGradTarget>
static void AssembleRigidBodyAsyncContactResponse(
    ContactAssemblyReg reg,
    entt::entity e,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    ContactEvalConfig const& config,
    ContactSamples const& sample,
    real dtStage,
    TransformRT const& pose,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& collisions,
    double* outEnergy,
    RigidGradient* outGradient,
    RigidHessian* outHessian) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(
      (!outEnergy || *outEnergy == 0.0) && (!outGradient || *outGradient == RigidGradient{}) &&
          (!outHessian || *outHessian == RigidHessian{}),
      "Expected this to be the first assembly term, with zero energy");

  bool const assemEnergy = (outEnergy != nullptr);
  bool const assemGradient = (outGradient != nullptr);
  bool const assemHessian = (outHessian != nullptr);

  // Gradient and Hessian accumulation terms across all colliders
  Vec4r outGradientCom = {};
  Vec4r outGradientRot = {};
  VMatrix3x3r outHessianCom = {};
  VMatrix3x3r outHessianRot = {};
  VMatrix3x3r outHessianMix = {};

  // Preallocate memory for collision response. Use stack memory if possible. 32 KiB is enough in
  // most cases.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 32 * 1024);
  CollisionResponseResult collisionResponse(&allocator);
  collisionResponse.Reserve(collisions, assemEnergy, assemGradient, assemHessian);

  // Get the com translation
  auto const com = pose.VGetTranslation();

  // For each actor with which we have async contact
  for (auto& collision : collisions) {
    auto contactParams = GetContactPairParams(reg, e, collision.colliderEntity);

    ContactDetectionResult& contactQuery = collision.collisionResult;
    int const numPoints = isize(contactQuery.sampleIndices);
    collisionResponse.ResizeNoInit(numPoints, assemEnergy, assemGradient, assemHessian);

    // Compute collision response
    ComputeCollisionResponse<kGradTarget>(
        contactQuery,
        contactParams,
        config,
        dtStage,
        assemEnergy,
        assemGradient,
        assemHessian,
        collisionResponse);

    // WARNING: If this implementation is modified, consider modifying also
    // AccumulateAsyncContactForceAdjoints, which is its dual for differentiability.

    // Contact res and dres are expressed in the collider's local frame.
    // Notation: (trans, rot) is the collider's root transform; p is the contact point in world
    // frame; p' is the contact point in collider's local frame:
    //   p = rot * p' + trans
    // com is the actor's COM in world frame; com' is the actor's COM in collider's local frame:
    //   com = rot * com' + trans
    // jVec is the lever arm from the actor's COM to the contact point, in collider's local frame:
    //   jVec = p' - com'
    // Trans jac: rotT
    // Rot jac: rotT * sk(com - p) = -sk(jVec) * rotT
    // Trans jac transpose: rot
    // Rot jac transpose: rot * sk(jVec)
    //
    // The per-contact contribution to the actor's energy Hessian (world frame) factors as
    //   J^T * (-df_world) * J,   J = [I, -sk(r)],   r = p - com = rot * jVec,
    // giving the three blocks:
    //   H_trans-trans = -df_world
    //   H_trans-rot   = df_world * sk(r)
    //   H_rot-rot     = sk(r) * df_world * sk(r)
    // In local frame, sk(r) = rot * sk(jVec) * rotT, so each block reduces to
    // R * (local sandwich) * R^T, which we accumulate per sample and apply once after the loop.

    // Fetch the collider's world-from-local rotation and precompute the actor COM in the collider's
    // local frame.
    auto const& colliderTransform =
        reg.get<CRootTransform const>(collision.colliderEntity).worldFromLocalPrev;
    auto const trans = colliderTransform.VGetTranslation();
    auto const [rot, rotT] = ToVMatrix3x3_WithTranspose(colliderTransform.GetRotation());
    auto const comColliderSpace = DotVecMat3x3(com - trans, rot);

    // Gradient and Hessian accumulation terms for this collider
    Vec4r res = {};
    Vec4r skJRes = {};
    VMatrix3x3r dres = {};
    VMatrix3x3r dresSkJ = {};
    VMatrix3x3r skJDresSkJ = {};

    for (int i = 0; i < numPoints; ++i) {
      // Get weight of the contact sample
      int const sampleIndex = contactQuery.sampleIndices[i];
      real const weight = sample.weights[sampleIndex];

      // Get contact position in collider's local frame
      auto posColliding = ToSimd(GetCollidingPosition<GetTimeStep<kGradTarget>()>(contactQuery, i));

      // Accumulate energy
      if (assemEnergy) {
        *outEnergy += collisionResponse.energy[i] * weight;
      }

      // Compute local lever arm from COM
      auto const jVec = posColliding - comColliderSpace;

      // Accumulate gradient terms
      if (assemGradient) {
        Vec4r collRes = weight * ToSimd(collisionResponse.force[i]);
        res += collRes;
        skJRes += Cross3(jVec, collRes);
      }

      // Accumulate hessian terms
      if (assemHessian) {
        VMatrix3x3r collDres = weight * collisionResponse.dforce[i];
        dres += collDres;
        VMatrix3x3r collSkJ = Skew3(jVec);
        VMatrix3x3r collDresSkJ = Dot3x3(collDres, collSkJ);
        dresSkJ += collDresSkJ;
        skJDresSkJ += Dot3x3(collSkJ, collDresSkJ);
      }
    }

    // Rotate the per-collider local-frame accumulators into world frame (R * _ * R^T for
    // matrices, R * _ for vectors) and accumulate into the actor's gradient/Hessian.
    if (assemGradient) {
      outGradientCom -= DotVecMat3x3(res, rotT);
      outGradientRot -= DotVecMat3x3(skJRes, rotT);
    }

    if (assemHessian) {
      outHessianCom -= Dot3x3(rot, Dot3x3(dres, rotT));
      outHessianRot += Dot3x3(rot, Dot3x3(skJDresSkJ, rotT));
      outHessianMix += Dot3x3(rot, Dot3x3(dresSkJ, rotT));
    }

    // Optionally store data for queries
    if (assemGradient && queryActiveContacts) {
      contactQuery.forcePerUnitArea = collisionResponse.force;
    }
  }

  // Store outputs
  if (assemGradient) {
    Store(outGradient->data(), outGradientCom);
    Store<RigidSize::kDRot>(outGradient->data() + RigidSize::kDTrans, outGradientRot);
  }
  if (assemHessian) {
    StoreSubmatrix<RigidSize::kDTrans, RigidSize::kDTrans, RigidSize::kDAll, RigidSize::kDAll>(
        *outHessian, Int2{0, 0}, outHessianCom);
    StoreSubmatrix<RigidSize::kDRot, RigidSize::kDRot, RigidSize::kDAll, RigidSize::kDAll>(
        *outHessian, Int2{RigidSize::kDTrans, RigidSize::kDTrans}, outHessianRot);
    StoreSubmatrix<RigidSize::kDTrans, RigidSize::kDRot, RigidSize::kDAll, RigidSize::kDAll>(
        *outHessian, Int2{0, RigidSize::kDTrans}, outHessianMix);
    StoreSubmatrix<RigidSize::kDRot, RigidSize::kDTrans, RigidSize::kDAll, RigidSize::kDAll>(
        *outHessian, Int2{RigidSize::kDTrans, 0}, Transpose3x3(outHessianMix));
  }
}

/*****************************************************************************************************
  ECS Dynamic Rigid-body Actor Utils
*/

static void ComputePoseAtStepStart(
    CTimeIntegratorState const& intState,
    CRigidState<TimeStep::Previous> const& prevPose,
    CIntegrationRigidStates& intPoses) {
  // Pose is a differential variable. Use integration utilities to compute its value at the
  // beginning of the step.
  integration::ApplyTimeIntegrationStepStart(intState, intPoses, prevPose, intPoses.stepStart);
}

void mochi::rigid::ComputeVelocityAtStepStart(
    CTimeIntegratorState const& intState,
    CRigidVel<TimeStep::Previous> const& prevVel,
    CIntegrationRigidVels& intVels) {
  // Velocity is a differential variable. Use integration utilities to compute its value at the
  // beginning of the step.
  integration::ApplyTimeIntegrationStepStart(intState, intVels, prevVel, intVels.stepStart);
}

static void ComputePoseAtStageStart(
    CTimeIntegratorState const& intState,
    CIntegrationRigidStates& intPoses,
    CRigidState<TimeStep::StageStart>& stageStartPose) {
  // Pose is a differential variable. Use integration utilities to compute its value at the
  // beginning of the stage.
  integration::ApplyTimeIntegration<TimeTarget::StageStart>(intState, intPoses, stageStartPose);
}

void mochi::rigid::ComputeRootTransformAtStageStart(
    CRigidBodyInertia const& rigidInertia,
    CRigidState<TimeStep::StageStart> const& stageStartPose,
    CRootTransform& rootTransform) {
  RigidStateToRootTransform(
      rigidInertia.GetCenterOfMassLocal(),
      stageStartPose.value,
      rootTransform.worldFromLocalStageStart);
}

void mochi::rigid::ComputeRootTransformCurrent(
    CRigidBodyInertia const& rigidInertia,
    CRigidState<TimeStep::Current> const& currPose,
    CRootTransform& rootTransform) {
  RigidStateToRootTransform(
      rigidInertia.GetCenterOfMassLocal(), currPose.value, rootTransform.worldFromLocal);
}

void mochi::rigid::ComputeVelocityAtStageStart(
    CTimeIntegratorState const& intState,
    CIntegrationRigidVels& intVels,
    CRigidVel<TimeStep::StageStart>& stageStartVel) {
  // Velocity is a differential variable. Use integration utilities to compute its value at the
  // beginning of the stage.
  integration::ApplyTimeIntegration<TimeTarget::StageStart>(intState, intVels, stageStartVel);
}

void mochi::rigid::ComputeVelocityAtStageEnd(
    CTimeIntegratorState const& intState,
    CRigidState<TimeStep::StageStart> const& stageStartPose,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidVel<TimeStep::Current>& currVel) {
  // Velocity is a differential variable but it's not explicitly solved for to reduce the number of
  // DoFs in the solver. Velocity at the end of the stage is recovered via finite differences of the
  // poses at the beginning and at the end of the stage.
  currVel.value.SetFromFiniteDifferencePose(stageStartPose.value, currPose.value, intState.dtStage);
}

static void ComputePoseAtTimeStepEnd(
    CTimeIntegratorState const& intState,
    CIntegrationRigidStates& intPoses,
    CRigidState<TimeStep::Current>& currPose) {
  // Pose is a differential variable. Use integration utilities to compute its value at the end of
  // the step.
  integration::ApplyTimeIntegration<TimeTarget::StepEnd>(intState, intPoses, currPose);
}

void mochi::rigid::ComputeVelocityAtTimeStepEnd(
    CTimeIntegratorState const& intState,
    CIntegrationRigidVels& intVels,
    CRigidVel<TimeStep::Current>& currVel) {
  // Velocity is a differential variable. Use integration utilities to compute its value at the end
  // of the step.
  integration::ApplyTimeIntegration<TimeTarget::StepEnd>(intState, intVels, currVel);
}

static void HandleSolverDivergence(
    CConvergenceStatus const& convergence,
    CRigidBodyInertia const& rigidInertia,
    CRigidState<TimeStep::Previous> const& prevPose,
    CRigidState<TimeStep::Current>& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CRootTransform& rootTransform) {
  if (convergence.stageStatus == ConvergenceStatus::Diverged) {
    // If diverged, reset velocity to zero and pose to that at the previous time step. Do not reset
    // pose at the beginning of the stage since there is no guarantee that's a safe state.
    currPose.value = prevPose.value;
    currVel.value.SetZero();
    rigid::RigidStateToRootTransform(
        rigidInertia.GetCenterOfMassLocal(), currPose.value, rootTransform.worldFromLocal);
  }
}

MOCHI_API void mochi::rigid::EntityIncrementStep(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CRigidState<TimeStep::Previous>& prevPose,
    CRigidVel<TimeStep::Previous>& prevVel) {
  MOCHI_PROFILE_SCOPE();
  // Copy pose and velocity from previous to current time step.
  prevPose.value = currPose.value;
  prevVel.value = currVel.value;

  // Reset current velocity to zero.
  currVel.value.SetZero();
}

MOCHI_API void mochi::rigid::EntityPreFirstStage(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CTimeIntegratorState const& intState,
    CRigidState<TimeStep::Previous> const& prevPose,
    CRigidVel<TimeStep::Previous> const& prevVel,
    CIntegrationRigidStates& intPoses,
    CIntegrationRigidVels& intVels) {
  MOCHI_PROFILE_SCOPE();
  ComputePoseAtStepStart(intState, prevPose, intPoses);
  ComputeVelocityAtStepStart(intState, prevVel, intVels);
}

MOCHI_API void mochi::rigid::EntityPreStage(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidBodyInertia const& rigidInertia,
    CTimeIntegratorState const& intState,
    CIntegrationRigidStates& intPoses,
    CIntegrationRigidVels& intVels,
    CRigidState<TimeStep::StageStart>& stageStartPose,
    CRigidVel<TimeStep::StageStart>& stageStartVel,
    CRootTransform& rootTransform) {
  MOCHI_PROFILE_SCOPE();
  // Compute pose, root transform, and velocity at the beginning of the stage.
  ComputePoseAtStageStart(intState, intPoses, stageStartPose);
  ComputeRootTransformAtStageStart(rigidInertia, stageStartPose, rootTransform);
  ComputeVelocityAtStageStart(intState, intVels, stageStartVel);
}

MOCHI_API void mochi::rigid::EntityPostStage(
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
    CIntegrationRigidVels& intVels) {
  MOCHI_PROFILE_SCOPE();
  // Compute the velocity at the end of the stage.
  ComputeVelocityAtStageEnd(intState, stageStartPose, currPose, currVel);

  // Reset pose and velocity if the solver diverged.
  HandleSolverDivergence(convergence, rigidInertia, prevPose, currPose, currVel, rootTransform);

  // Pose and velocity are differential variables. Push them to the vectors containing the poses and
  // velocities at the end of each time integration stage.
  intPoses.stages[intState.currentStage].value = currPose.value;
  intVels.stages[intState.currentStage].value = currVel.value;
}

MOCHI_API void mochi::rigid::EntityPostLastStage(
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CRigidBodyInertia const& rigidInertia,
    CTimeIntegratorState const& intState,
    CIntegrationRigidStates& intPoses,
    CIntegrationRigidVels& intVels,
    CRigidState<TimeStep::Current>& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CRootTransform& rootTransform) {
  MOCHI_PROFILE_SCOPE();
  // Compute pose and velocity at the end of the time step.
  ComputePoseAtTimeStepEnd(intState, intPoses, currPose);
  RigidStateToRootTransform(
      rigidInertia.GetCenterOfMassLocal(), currPose.value, rootTransform.worldFromLocal);
  ComputeVelocityAtTimeStepEnd(intState, intVels, currVel);
}

void rigid::SetupCollidingJacobiansImpl(
    TransformRT const& state,
    TransformRT const& transform,
    CDofOffset const& dofOffset,
    CContactSamples<TimeStep::Current> const& samples,
    CCollJacs<CollRole::Colliding>& outJacobians,
    MatrixView<real const> jacAux,
    Span<int const> dofsAux) {
  MOCHI_PROFILE_SCOPE();

  // Create sync-rigid differentiable map, shared by all Jacobians
  DMapSyncRigid dsyncRigid(0, dofOffset.dofsOffset, jacAux, dofsAux);
  DMap<DMapSyncRigid> dmapSyncRigid(&dsyncRigid);

  // Compute Jacobians
  for (auto& jac : outJacobians) {
    if (jac.type == ContactType::Sync) {
      auto& jacs = *jac.jacs;
      if (jac.bothRigid) {
        dmapSyncRigid.GetJac(jac.query->sampleIndices, jacs);
      } else {
        // Create regular differentiable map
        DMapRTInput dtransform(
            0,
            state,
            transform,
            dofOffset.dofsOffset,
            samples.positions,
            jac.query->jacColliderFromWorld,
            jacAux,
            dofsAux);
        DMap<DMapRTInput> dmap(&dtransform);
        dmap.GetJac(jac.query->sampleIndices, jacs);
      }
      jacs[0].CompressIndices();
    }
  }
}

void rigid::SetupColliderJacobiansImpl(
    TransformRT const& state,
    CDofOffset const& dofOffset,
    CRigidBodyInertia const& rigidInertia,
    CCollJacs<CollRole::Collider>& outJacobians,
    MatrixView<real const> jacAux,
    Span<int const> dofsAux) {
  MOCHI_PROFILE_SCOPE();

  // Create differentiable maps
  DMapRTOutput dtransform(
      0, state, dofOffset.dofsOffset, rigidInertia.GetCenterOfMassLocal(), jacAux, dofsAux);
  DMap<DMapRTOutput> dmap(&dtransform);
  DMapSyncRigid dsyncRigid(0, dofOffset.dofsOffset, jacAux, dofsAux);
  DMap<DMapSyncRigid> dmapSyncRigid(&dsyncRigid);

  // Compute Jacobians
  for (auto& jac : outJacobians) {
    auto& jacs = *jac.jacs;
    if (jac.bothRigid) {
      dmapSyncRigid.GetJac(jac.query->sampleIndices, jacs);
    } else {
      dtransform.SetData(jac.query);
      dmap.GetJac({}, jacs);
    }
    jacs[0].CompressIndices();
  }
}

template <ContactType kContactType>
void mochi::rigid::SetupActiveCollisionNormals(
    ecs::RequiredTag<TagRigidActor>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CContactSamples<TimeStep::Current> const& samples,
    CRootTransform const& transform,
    CActiveCollisions<kContactType, TimeStep::Current>& activeCollisions) {
  MOCHI_PROFILE_SCOPE();

  MOCHI_ASSERT_VERBOSE(
      samples.normals.has_value() && samples.normals->size() == samples.positions.size(),
      "Contact sample normals should be pre-computed for rigid actors");
  auto const& normals = *samples.normals;

  auto const explicitNormals = simParams->experimentalEval.explicitNormals;
  auto const rotWorldFromCollidingT = ToVMatrix3x3Transpose(
      explicitNormals ? transform.worldFromLocalStageStart.GetRotation()
                      : transform.worldFromLocal.GetRotation());

  for (auto& activeCollision : activeCollisions) {
    auto& collisionResult = activeCollision.collisionResult;
    if (collisionResult.sampleIndices.empty()) {
      continue;
    }
    collisionResult.normalColliding.resize_noinit(collisionResult.sampleIndices.size());

    // Initialize with the transform of the first contact. For rigid colliders, it is shared by all.
    auto const& jacColliderFromWorld = explicitNormals
        ? collisionResult.jacColliderFromWorldStageStart
        : collisionResult.jacColliderFromWorld;
    auto jacColliderFromCollidingT =
        Dot3x3(rotWorldFromCollidingT, Transpose3x3(jacColliderFromWorld[0]));

    for (size_t i = 0; i < collisionResult.sampleIndices.size(); i++) {
      Vec4r normalColliding = ToSimd(normals[collisionResult.sampleIndices[i]]);
      if (jacColliderFromWorld.size() == 1) {
        collisionResult.normalColliding[i] =
            ToReal3(DotVecMat3x3(normalColliding, jacColliderFromCollidingT));
      } else {
        normalColliding = DotVecMat3x3(normalColliding, rotWorldFromCollidingT);
        collisionResult.normalColliding[i] =
            ToReal3(DotMatVec3x3(jacColliderFromWorld[i], normalColliding));
      }
    }
  }
}

#define MOCHI_SPECIALIZE_SETUP_ACTIVE_COLLISIONS_KINEMATICS(contactType) \
  template void mochi::rigid::SetupActiveCollisionNormals<contactType>(  \
      ecs::RequiredTag<TagRigidActor>,                                   \
      ecs::CtxGlobal<CSimulationParams const>,                           \
      CContactSamples<TimeStep::Current> const&,                         \
      CRootTransform const&,                                             \
      CActiveCollisions<contactType, TimeStep::Current>&);
MOCHI_SPECIALIZE_SETUP_ACTIVE_COLLISIONS_KINEMATICS(ContactType::Sync);
MOCHI_SPECIALIZE_SETUP_ACTIVE_COLLISIONS_KINEMATICS(ContactType::Async);
#undef MOCHI_SPECIALIZE_SETUP_ACTIVE_COLLISIONS_KINEMATICS

static void EntityPostNewSolutionImpl(
    ColumnVectorView<real const, RigidSize::kAll> solution,
    CRigidBodyInertia const& rigidInertia,
    CRigidState<TimeStep::Current>& rigidState,
    CRootTransform& rootTransform) {
  MOCHI_PROFILE_SCOPE();

  // Update internal state
  rigid::EntitySetSolution(solution, {}, {}, rigidState);

  // Update actor root transform
  rigid::RigidStateToRootTransform(
      rigidInertia.GetCenterOfMassLocal(), rigidState.value, rootTransform.worldFromLocal);
}

void mochi::rigid::EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CDofOffset const& dofOffset,
    CRigidBodyInertia const& rigidInertia,
    CRigidState<TimeStep::Current>& rigidState,
    CRootTransform& rootTransform) {
  auto solData = solution.MiddleRows<RigidSize::kAll>(dofOffset.poseOffset, RigidSize::kAll);
  EntityPostNewSolutionImpl(solData, rigidInertia, rigidState, rootTransform);
}

void mochi::rigid::EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CDofOffset const& dofOffset,
    CRigidBodyInertia const& rigidInertia,
    CDirichletBC<real> const& dirichlet,
    CRigidState<TimeStep::Current>& rigidState,
    CRootTransform& rootTransform) {
  auto ref = reference.MiddleRows<RigidSize::kAll>(dofOffset.poseOffset, RigidSize::kAll);
  auto inc = increment.MiddleRows<RigidSize::kDAll>(dofOffset.dofsOffset, RigidSize::kDAll);
  ColumnVector<real, RigidSize::kAll> sol;
  // Add the increment translation to the reference translation
  Store(sol.data(), Load<Vec4r>(ref.data()) + Load<Vec4r>(inc.data()));
  // Compose the increment rotation with the reference rotation. Normalize to avoid drift.
  auto rot =
      Quaternion::FromRotationVector(Load<RigidSize::kDRot, Vec4r>(&inc[RigidSize::kDTrans])) *
      Quaternion(Load<RigidSize::kRot, Vec4r>(&ref[RigidSize::kTrans]));
  rot = Normalize(rot);
  Store<RigidSize::kRot>(&sol[RigidSize::kTrans], rot.data);

  // Apply Dirichlet boundary conditions explicitly, because the rotation composition may add noise.
  SetDirichletBCs(dirichlet, sol);

  EntityPostNewSolutionImpl(sol, rigidInertia, rigidState, rootTransform);
}

void mochi::rigid::EntityAssemble(
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
    CActiveCollisions<ContactType::Async, TimeStep::Current>* activeCollisions) {
  MOCHI_PROFILE_SCOPE();

  // If this is an input grad target, all we need to do is resize the residual to 0
  auto const gradTarget = params.gradTarget;
  if (gradTarget == GradTarget::CurrentInput || gradTarget == GradTarget::PreviousInput) {
    MOCHI_ASSERT_VERBOSE(
        !params.assemObj && params.assemRes && !params.assemDRes, "Invalid request");
    outActorSnle.fullResidual.Resize(0);
    return;
  }

  // Prepare energy call arguments
  double tempEnergy{};
  double* energy = params.assemObj ? &tempEnergy : nullptr;
  RigidGradient tempGradient{};
  RigidGradient* gradient = params.assemRes ? &tempGradient : nullptr;
  RigidHessian tempHessian{};
  RigidHessian* hessian = params.assemDRes ? &tempHessian : nullptr;

  // Compute async contact term first. It assumes that energy starts at zero.
  bool bContactEnabled = hasContactTag && activeCollisions && !activeCollisions->empty() &&
      IsAssemblyNeeded(StateDependency::FirstOrder, false /*inputDependency*/, gradTarget);
  if (bContactEnabled) {
    MOCHI_ASSERT_VERBOSE(contactSample, "Contact sample must exist for contact.");
    ContactEvalConfig config{
        .psdDRes = params.psdDRes,
        .addPadding = ShouldAddPenaltyPadding(colliderInfo.type),
        .validCollidingNormals = ValidCollidingNormals(reg, e),
        .explicitNormals = simParams->experimentalEval.explicitNormals,
        .fadeFriction = simParams->experimentalEval.fadeFriction,
        .implicitNormalForceForDissipation =
            simParams->experimentalEval.implicitNormalForceForDissipation,
        .useFittedHessian = params.fittedSaturationHessian.contactFriction,
        .frictionModel = simParams->experimentalEval.frictionModel};
    auto const& pose = gradTarget == GradTarget::Current ? currPose.value : stageStartPose.value;
    // Dispatch to the appropriate templatized implementation
    auto assembleRigidBodyAsyncContactResponseFn = gradTarget == GradTarget::Previous
        ? AssembleRigidBodyAsyncContactResponse<GradTarget::Previous>
        : AssembleRigidBodyAsyncContactResponse<GradTarget::Current>;
    assembleRigidBodyAsyncContactResponseFn(
        reg,
        e,
        queryActiveContacts,
        config,
        *contactSample,
        intState.dtStage,
        pose,
        *activeCollisions,
        energy,
        gradient,
        hessian);
  }

  // Compute gravitational terms
  if (hasGravityTag &&
      IsAssemblyNeeded(StateDependency::ZeroOrder, false /*inputDependency*/, gradTarget)) {
    AddRigidBodyGravity(
        currPose.value, rigidInertia, ToReal3(sceneGravity->accel), energy, gradient);
  }

  // Compute inertial terms
  if (hasInertiaTag && IsFinite(intState.dtStage) &&
      IsAssemblyNeeded(StateDependency::SecondOrder, false /*inputDependency*/, gradTarget)) {
    auto inertiaFunc = GetRigidInertiaFn<VMatrix3x3r>(gradTarget, useNewtonEulerInertia);

    // We need a separate hessian storage to project the rotational part to its PSD cone
    RigidHessian tempInertiaHessian{};
    RigidHessian* inertiaHessian =
        (params.assemDRes && params.psdDRes) ? &tempInertiaHessian : hessian;
    inertiaFunc(
        rigidInertia.GetMass(),
        useNewtonEulerInertia ? rigidInertia.GetMomentOfInertiaLocal()
                              : rigidInertia.GetSecondMomentLocal(),
        intState.dtStage,
        currPose.value,
        stageStartPose.value,
        stageStartVel.value,
        energy,
        gradient,
        inertiaHessian);

    if (params.assemDRes && params.psdDRes) {
      VMatrix3x3r outHessian, outInertiaHessian;
      // add translational part
      LoadSubmatrix<3, 3, 6, 6>(outHessian, Int2{0, 0}, tempHessian);
      LoadSubmatrix<3, 3, 6, 6>(outInertiaHessian, Int2{0, 0}, tempInertiaHessian);
      outHessian += outInertiaHessian;
      StoreSubmatrix<3, 3, 6, 6>(tempHessian, Int2{0, 0}, outHessian);
      // project rotational part
      LoadSubmatrix<3, 3, 6, 6>(outHessian, Int2{3, 3}, tempHessian);
      LoadSubmatrix<3, 3, 6, 6>(outInertiaHessian, Int2{3, 3}, tempInertiaHessian);
      ProjectSymPsd(outInertiaHessian, std::numeric_limits<real>::epsilon());
      outHessian += outInertiaHessian;
      StoreSubmatrix<3, 3, 6, 6>(tempHessian, Int2{3, 3}, outHessian);
    }
  }

  // Compute contact terms
  if (!externalForces.Empty() &&
      IsAssemblyNeeded(StateDependency::ZeroOrder, false /*inputDependency*/, gradTarget)) {
    AddRigidBodyExternalForces(
        currPose.value,
        stageStartPose.value,
        externalForces.dofs,
        externalForces.forces,
        energy,
        gradient);
  }

  // Sum energy
  if (params.assemObj) {
    outActorSnle.objective = tempEnergy;
  }

  // Assemble residual at offset.
  if (params.assemRes) {
    outActorSnle.fullResidual.Resize(RigidSize::kDAll);
    outActorSnle.fullResidual = AsConstView(tempGradient);
  }

  // Assemble dresidual at offset
  if (params.assemDRes) {
    auto localDRes = Flatten(tempHessian);
    MOCHI_ASSERT(
        std::holds_alternative<Matrix<real>>(outActorSnle.fullDResidual),
        "Expected dense storage.");
    auto& actorDRes = std::get<Matrix<real>>(outActorSnle.fullDResidual);
    MOCHI_ASSERT(
        isize(actorDRes) == isize(localDRes) &&
        isize(actorDRes) == RigidSize::kDAll * RigidSize::kDAll);
    actorDRes = RowMatrixView<real const, RigidSize::kDAll, RigidSize::kDAll>(localDRes.begin());
  }
}

// Called at the end of InitRigidActor if the actor is dynamic (not static)
static void InitRigidActor_Dynamic(
    entt::registry& reg,
    entt::entity e,
    RigidActorParams const& params,
    bool useContact,
    Shape const* shapePtr,
    Error& error) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ERROR_RETURN(error);
  MOCHI_ASSERT(reg.valid(e), "Expected the caller to create the entity");
  MOCHI_ASSERT(!params.isStatic, "This function is for dynamic rigid actors only");

  auto const* meshComponent = reg.try_get<CSurfaceMesh const>(e);
  MOCHI_ERROR_IF(
      useContact && !meshComponent,
      error,
      "Unable to create dynamic rigid actor. The shape must have a surface mesh.");
  MOCHI_ERROR_RETURN(error);

  // Basic components
  reg.emplace_or_replace<CDofOffset>(e);
  reg.emplace<CConvergenceStatus>(e);

  // Surface discretization
  std::unique_ptr<CFemSurfaceDiscretization> surfDiscr =
      nullptr; // Temporary for initialization. Not stored as component to reduce memory footprint.
  std::shared_ptr<TriangularMesh const> boundaryMesh;
  if (meshComponent) {
    boundaryMesh = meshComponent->mesh;
    reg.emplace<CFemSurfaceDiscretizationLite>(
        e, CFemSurfaceDiscretizationLite::Create(params.boundaryElementType, *boundaryMesh));
    surfDiscr = std::make_unique<CFemSurfaceDiscretization>(
        CFemSurfaceDiscretization::Create(params.boundaryElementType, *boundaryMesh));
  }

  // Center-of-mass (if there is mass)
  std::optional<Vec4r> comLocal;

  // Inertia
  if (params.centerOfMass.has_value() != params.momentOfInertia.has_value()) {
    MOCHI_LOG_WARNING(
        "Creating rigid actor \"%s\" with a user-specified center-of-mass but no specified moment-of-inertia, or "
        "visa versa. Please set both of these parameters, or neither. Mochi will compute the missing data, but it "
        "will assume uniform density, which may not be correct.",
        params.name.c_str());
  }
  if (shapePtr) {
    // Get or compute volume
    std::optional<real> volume = shapePtr->GetVolume();
    if (!volume && surfDiscr) {
      // Compute volume by computing the mass with a density of 1.
      // If we do this, then store the center-of-mass for later use.
      surfDiscr->Visit([&](auto const& discrImpl) {
        comLocal = Vec4r{};
        volume = 0_r;
        ComputeCenterOfMassFem(
            MakeConstSpan(discrImpl.femElements), /*density*/ 1_r, *comLocal, *volume);
      });
    }
    MOCHI_ERROR_IF(
        !volume.has_value() || (*volume <= 0_r),
        error,
        "Unable to create dynamic rigid actor. The shape has no volume.");
    MOCHI_ERROR_IF(
        volume.has_value() && !IsFinite(*volume),
        error,
        "Unable to create dynamic rigid actor. The shape volume is not finite.");
    MOCHI_ERROR_RETURN(error);

    // The caller is allowed to specify mass or density, but not both.
    std::optional<real> mass = params.mass;
    std::optional<real> density = params.density;
    if (mass && density) {
      MOCHI_LOG_WARNING(
          "Attempting to set both mass and density when creating rigid actor \"%s\". Please specify one or the other. Using the mass value. Ignoring the density value.",
          std::string{params.name}.c_str());
      density = std::nullopt;
    } else if (!mass && !density) {
      // Use default density
      density = kDefaultDensity;
    }

    // Compute mass or density, whichever is not provided by the caller.
    if (density) {
      mass = (*density) * (*volume);
    } else {
      density = (*mass) / (*volume);
    }

    MOCHI_ERROR_IF_NOT(
        *density > 0_r && IsFinite(*density),
        error,
        "Unable to create dynamic rigid actor. Density must be positive and finite.");
    MOCHI_ERROR_IF_NOT(
        *mass > 0_r && IsFinite(*mass),
        error,
        "Unable to create dynamic rigid actor. Mass must be positive and finite.");
    MOCHI_ERROR_RETURN(error);

    // The coller can provide center-of-mass. If not, we compute it.
    if (params.centerOfMass) {
      MOCHI_ERROR_IF_NOT(
          IsFinite(*params.centerOfMass),
          error,
          "Unable to create dynamic rigid actor. Center of mass must be finite.");
      MOCHI_ERROR_RETURN(error);
      comLocal = ToSimd(*params.centerOfMass);
    } else if (comLocal) {
      // Already computed (see above)
    } else if (auto center = shapePtr->GetCentroid()) {
      // Already computed by the Shape
      comLocal = ToSimd(*center);
    } else if (surfDiscr) {
      // Compute the center-of-mass (density does not matter for this).
      surfDiscr->Visit([&](auto const& discrImpl) {
        comLocal = Vec4r{};
        real massIgnored = 0_r;
        ComputeCenterOfMassFem(
            MakeConstSpan(discrImpl.femElements), /*density*/ 1_r, *comLocal, massIgnored);
      });
    } else {
      MOCHI_ERROR_SET(error, "Cannot create dynamic rigid actor. Center-of-mass unknown.");
      return;
    }

    // Compute moment of inertia.
    VMatrix3x3r moi;
    if (params.momentOfInertia) {
      Real6 const& inI = *params.momentOfInertia;
      MOCHI_ERROR_IF_NOT(
          IsFinite(inI),
          error,
          "Unable to create dynamic rigid actor. Moment-of-inertia tensor must be finite.");
      MOCHI_ERROR_RETURN(error);
      if (!IsMomentOfInertiaValid(inI)) {
        MOCHI_LOG_WARNING(
            "Moment-of-inertia tensor of rigid actor \"%s\" is not physically valid: principal moments must be non-negative and satisfy the triangle inequality.",
            params.name.c_str());
      }
      moi = VMatrix3x3r{
          Vec4r{inI[0], inI[1], inI[2]}, // ixx, ixy, ixz
          Vec4r{inI[1], inI[3], inI[4]}, // ixy, iyy, iyz
          Vec4r{inI[2], inI[4], inI[5]}, // ixz, iyz, izz
      };
    } else if (surfDiscr) {
      surfDiscr->Visit([&](auto const& discrImpl) {
        auto moiTwo = ComputeSecondMomentOfInertiaFem(
            MakeConstSpan(discrImpl.femElements), *density, *comLocal);
        moi = SecondMomentToMomentOfInertia(moiTwo);
      });
    } else {
      MOCHI_ERROR_SET(error, "Cannot create dynamic rigid actor. Moment-of-inertia unknown.");
      return;
    }

    // Store the inertia properties.
    reg.emplace<CRigidBodyInertia>(e, *comLocal, *volume, moi, *density);

    // Energy terms used
    reg.emplace<TagUseInertia>(e);
    if (params.hasGravity) {
      reg.emplace<TagUseGravity>(e);
    }
  } else {
    // This actor has no shape. It must be a dummy link in an articulated actor. Such actors have no
    // volume, nor mass. However, systems elsewhere still expect it to have CRigidBodyInertia, even
    // though all the values will be zero. TODO: Remove this requirement.
    reg.emplace<CRigidBodyInertia>(e);
  }

  // Scaled time step
  reg.emplace<CTimeIntegratorState>(e);

  // Initialize from root transform
  auto const& rootTransform = reg.get<CRootTransform const>(e);
  auto const& currState = reg.emplace<CRigidState<TimeStep::Current>>(e).value;
  ecs::InvokeOnEntity(&rigid::SetRootTransform, reg, e, std::cref(rootTransform.worldFromLocal));
  reg.emplace<CRigidState<TimeStep::Previous>>(e, currState);
  reg.emplace<CRigidState<TimeStep::StageStart>>(e, currState);
  reg.emplace<CIntegrationRigidStates>(e);

  // Velocity components
  reg.emplace<CRigidVel<TimeStep::Previous>>(e);
  auto& vel = reg.emplace<CRigidVel<TimeStep::Current>>(e).value;
  reg.emplace<CRigidVel<TimeStep::StageStart>>(e);
  reg.emplace<CIntegrationRigidVels>(e);
  if (params.linearVelocity) {
    MOCHI_ERROR_IF_NOT(
        IsFinite(*params.linearVelocity),
        error,
        "Unable to create dynamic rigid actor. Linear velocity must be finite.");
    MOCHI_ERROR_RETURN(error);
    vel.SetVCom(ToSimd(*params.linearVelocity));
    reg.get<CPrevRigidVelocity>(e).linearVelocityWorld = ToSimd(*params.linearVelocity);
  }
  if (params.angularVelocity) {
    MOCHI_ERROR_IF_NOT(
        IsFinite(*params.angularVelocity),
        error,
        "Unable to create dynamic rigid actor. Angular velocity must be finite.");
    MOCHI_ERROR_RETURN(error);
    vel.SetOmega(ToSimd(*params.angularVelocity));
    reg.get<CPrevRigidVelocity>(e).angularVelocityWorld = ToSimd(*params.angularVelocity);
  }

  // Boundary conditions
  reg.emplace<CDirichletBC<real>>(e);

  // Storage for external forces
  reg.emplace<CExternalForces>(e);

  // Center of mass
  auto& rigidVelocity = reg.get<CPrevRigidVelocity>(e);
  if (comLocal) {
    rigidVelocity.centerOfMassLocal = *comLocal;
  }

  // Components to detect and compute contact against other actors
  if (useContact) {
    MOCHI_ASSERT(surfDiscr, "Surface discretization is missing.");
    reg.emplace<TagUseContact>(e);

    // Active boundary faces.
    CActiveBoundaryFaces* activeBoundaryFaces = nullptr;
    if (params.boundarySubsampling) {
      MOCHI_ERROR_IF_NOT(boundaryMesh, error, "A boundary mesh is required for hyper-reduction.");
      MOCHI_ERROR_IF_NOT(
          params.boundarySubsampling->subsamplingDensity >= 0_r &&
              params.boundarySubsampling->subsamplingDensity <= 1_r,
          error,
          "Boundary subsampling density must be between 0 and 1.");
      MOCHI_ERROR_RETURN(error);
      if (params.boundarySubsampling->subsamplingDensity < 1_r) {
        activeBoundaryFaces = &reg.emplace<CActiveBoundaryFaces>(
            e, CreateActiveBoundaryFaces(*params.boundarySubsampling, boundaryMesh, *surfDiscr));
      }
    }

    // Initialize contact sample with respect to the root transform.
    int const numSamples = surfDiscr->GetNumQuadPoints();
    auto& contactSamples = reg.emplace<CContactSamples<TimeStep::Current>>(e, numSamples);
    surfDiscr->Visit([&](auto& discrImpl) {
      constexpr int kNumQuadsPerFace = std::decay_t<decltype(discrImpl)>::kNumQuads;
      int const numBoundFaces = boundaryMesh->GetNumElements();

      contactSamples.normals = std::vector<Real3>(numSamples);
      if (activeBoundaryFaces) {
        contactSamples.activePositions.reserve(numSamples);
        contactSamples.activeIndices.reserve(numSamples);
      }

      // Add trace samples
      int count = 0;
      for (int i = 0; i < numBoundFaces; ++i) {
        bool const storePointAsActive = activeBoundaryFaces && activeBoundaryFaces->Contains(i);
        auto& posLocalQuads = discrImpl.femElements[i].mapEvaluated;
        auto& normalLocalQuads = discrImpl.femElements[i].normals;
        for (int j = 0; j < kNumQuadsPerFace; ++j) {
          contactSamples.positions[count] = posLocalQuads[j];
          contactSamples.weights[count] = discrImpl.femElements[i].quadWeights[j];
          (*contactSamples.normals)[count] = normalLocalQuads[j];
          if (storePointAsActive) {
            contactSamples.activePositions.push_back(posLocalQuads[j]);
            contactSamples.activeIndices.push_back(count);
          }
          count++;
        }
      }

      // Create BSH tree to accelerate collision detection.
      contactSamples.bsh = activeBoundaryFaces ? MakeConstSpan(contactSamples.activePositions)
                                               : MakeConstSpan(contactSamples.positions);
    });

    // Emplace helper component to map stage-start and current contact results
    reg.emplace<CContactCorrespondence<ContactType::Async>>(e, numSamples);
    reg.emplace<CContactCorrespondence<ContactType::Sync>>(e, numSamples);

    reg.emplace<CConservativePotentialColliders<ContactType::Async>>(e);
    reg.emplace<CConservativePotentialColliders<ContactType::Sync>>(e);
    reg.emplace<CPotentialColliders<ContactType::Async>>(e);
    reg.emplace<CPotentialColliders<ContactType::Sync>>(e);
    reg.emplace<CActiveCollisions<ContactType::Async, TimeStep::Current>>(e);
    reg.emplace<CActiveCollisions<ContactType::Async, TimeStep::StageStart>>(e);
    reg.emplace<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e);
    reg.emplace<CActiveCollisions<ContactType::Sync, TimeStep::StageStart>>(e);
    reg.emplace<CCollJacs<CollRole::Colliding>>(e);
  }

  if (reg.get<CColliderInfo const>(e).type != ColliderType::None) {
    reg.emplace<CCollJacs<CollRole::Collider>>(e);
  }

  // CActorDofInfo
  auto& dofInfo = reg.emplace<CActorDofInfo>(e);
  dofInfo.poseSize = RigidSize::kAll;
  dofInfo.dofsSize = RigidSize::kDAll;

  // Sparsity pattern
  reg.emplace<CFullSparsityPattern>(e, MakeDenseSparsityGraph(dofInfo.dofsSize, dofInfo.dofsSize));

  // Per-actor SNLE data
  reg.emplace<CActorSnle>(
      e,
      Matrix<real>(dofInfo.dofsSize, dofInfo.dofsSize),
      // Use symmetric inverse actor preconditioner (cheap and exact).
      PreconditionerType::SymInverse);

  // Non-linear solver convergence weights (lazily initialized).
  reg.emplace<CActorConvergenceWeights>(e);

  // Bounding volume used for island grouping
  reg.emplace<CConservativeStepBounds>(e);
}

// Used for both static and dynamic rigid actors
void mochi::InitRigidActor(
    entt::registry& reg,
    entt::entity e,
    RigidActorParams const& params,
    bool useContact,
    std::shared_ptr<Shape const> shapePtr,
    Error& error) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(reg.valid(e), "Expected the caller to create the entity");
  ValidateContactParams(params.contact, error);
  auto const colliderTypeValue = static_cast<int>(params.colliderType);
  MOCHI_ERROR_IF(
      colliderTypeValue < 0 || colliderTypeValue >= static_cast<int>(ColliderType::Count),
      error,
      "Invalid collider type.");
  auto const boundaryElementType = static_cast<int>(params.boundaryElementType);
  MOCHI_ERROR_IF(
      boundaryElementType < 0 ||
          boundaryElementType >= static_cast<int>(ActorBoundaryElementType::Count),
      error,
      "Invalid actor boundary element type.");
  if (params.boundarySubsampling) {
    auto const subsamplingStrategy = static_cast<int>(params.boundarySubsampling->strategy);
    MOCHI_ERROR_IF(
        subsamplingStrategy < 0 ||
            subsamplingStrategy >= static_cast<int>(BoundarySubsamplingStrategy::Count),
        error,
        "Invalid boundary subsampling strategy.");
  }
  MOCHI_ERROR_RETURN(error);

  // Basic components
  reg.emplace<CActorInfo>(e, std::string(params.name), ActorType::Rigid);
  reg.emplace<CRootTransform>(e, params.worldFromLocal);

  // Disable contact if there's no shape
  useContact &= shapePtr != nullptr;

  // Collider Type (may be modified below)
  ColliderType colliderType = params.colliderType;

  // Rigid-body shape
  CSurfaceMesh* meshComponent = nullptr;
  if (shapePtr) {
    reg.emplace<CShape>(e, shapePtr);
    auto const* tetMeshShape = dynamic_cast<TetrahedralMeshShape const*>(shapePtr.get());
    auto const* triMeshShape = dynamic_cast<TriangularMeshShape const*>(shapePtr.get());
    if (tetMeshShape || triMeshShape) {
      auto const& surfMesh =
          tetMeshShape ? tetMeshShape->GetMesh()->GetBoundaryMesh() : triMeshShape->GetMesh();
      MOCHI_ASSERT(surfMesh != nullptr, "Missing surface mesh.");
      meshComponent = &reg.emplace<CSurfaceMesh>(e, surfMesh);
      if (auto const& visualMesh =
              tetMeshShape ? tetMeshShape->GetVisualMesh() : triMeshShape->GetVisualMesh();
          visualMesh) {
        reg.emplace<CVisualMesh>(e, visualMesh, nullptr);
      }

      // Auto-select Sdf for mesh shapes
      if (colliderType == ColliderType::Auto) {
        colliderType = ColliderType::Sdf;
      }
    } else if (
        auto const* implicitShape = dynamic_cast<ImplicitRigidShape const*>(shapePtr.get())) {
      if (std::get_if<Sphere>(&implicitShape->shape)) {
        colliderType = ColliderType::Sphere; // Must collide as a sphere
      } else if (std::get_if<Plane>(&implicitShape->shape)) {
        colliderType = ColliderType::Plane; // Must collide as a plane
      } else {
        MOCHI_ERROR_SET(error, "Not a supported ImplicitRigidShape type");
      }
    } else {
      MOCHI_ERROR_SET(error, "Not a supported shape type for rigid actors");
    }
    MOCHI_ERROR_RETURN(error);
  } else {
    colliderType = ColliderType::None; // No collider if there's no shape
  }

  // Collider
  auto& colliderInfo = reg.emplace<CColliderInfo>(e);
  colliderInfo.type = colliderType;
  AnyShape bv = shapePtr ? shapePtr->GetBoundingVolume(error) : AnyShape{Sphere()};
  MOCHI_ERROR_RETURN(error);
  if (shapePtr) {
    auto const& contactParams = reg.emplace<CContactParams>(e, params.contact);

    EmplaceContactLayer(reg, e, params.layer);
    static_assert(
        static_cast<int>(ColliderType::Count) == 8,
        "Please update the following switch statement if the ColliderType enum changes.");
    switch (colliderType) {
      case ColliderType::Sphere: {
        auto& sphereColl = reg.emplace<CSphereCollider>(e);
        if (auto* bvSphere = std::get_if<Sphere>(&bv)) {
          sphereColl.shape = *bvSphere; // already a sphere
        } else if (auto* bvObb = std::get_if<Obb>(&bv)) {
          // Use the sphere that inscribes the box. It should be a good match for the mesh, if the
          // mesh really is in the approximate shape of a sphere
          sphereColl.shape = Sphere{bvObb->GetCenter(), bvObb->GetHalfExtents()[0]};
        } else if (auto* bvAabb = std::get_if<Aabb>(&bv)) {
          // Similarly, use the sphere that inscribes the AABB.
          sphereColl.shape = Sphere{bvAabb->GetCenter(), bvAabb->GetHalfExtents()[0]};
        } else {
          // Fallback: Just wrap the shape in a sphere. Expand bv to make sure it is large enough
          // to contain both.
          sphereColl.shape = GetBoundingSphere(bv);
          bv = GetObb(EncloseShapes(GetAabb(sphereColl.shape), GetAabb(bv)));
        }
      } break;
      case ColliderType::Box: {
        auto& boxColl = reg.emplace<CBoxCollider>(e);
        boxColl.shape = mochi::GetObb(bv);
        // If the bv is something like a sphere, then our box collider would be larger than the
        // original shape. Let's expand the BV to contain both just in case.
        bv = GetObb(EncloseShapes(GetAabb(boxColl.shape), GetAabb(bv)));
      } break;
      case ColliderType::Mesh: {
        if (meshComponent) {
          auto& collider = reg.emplace<CMeshCollider>(e, meshComponent->mesh);
          collider.Initialize();
          LogMeshColliderDiagnostics(collider, std::string(params.name));
        } else {
          MOCHI_ERROR_SET(error, "ColliderType::Mesh requires a surface mesh");
        }
      } break;
      case ColliderType::Plane: {
        if (auto const* implicitShape = dynamic_cast<ImplicitRigidShape const*>(shapePtr.get())) {
          auto const* plane = std::get_if<Plane>(&implicitShape->shape);
          MOCHI_ASSERT(
              plane != nullptr,
              "If it were not a plane, then the previous check for ImplicitRigidShape should have changed the ColliderType or returned an error (see above).");
          reg.emplace<CPlaneCollider>(e, *plane);
        } else {
          MOCHI_ERROR_SET(
              error, "ColliderType::Plane requires a ShapeHandle representing an infinite plane.");
        }
      } break;
      case ColliderType::Sdf: {
        if (auto gridSdfShape = std::dynamic_pointer_cast<GridSdfShape const>(shapePtr)) {
          auto& sdfCollider = reg.emplace<CSdfCollider>(e);

          // Define the padding based on the most conservative penalty threshold distance.
          GridSdfParams gridSdfParams = params.sdf;
          gridSdfParams.boundaryPaddingDist =
              contactParams.GetPenaltyThresholdDist(/* addPadding */ true);

          // Get or request the GridSdf.
          bool isPending = false;
          sdfCollider.shape = gridSdfShape->RequestGridSdf(gridSdfParams, &isPending);

          // If the GridSdf is being computed asynchronously, then add a component.
          // We will wait for completion before the next simulation step.
          if (isPending) {
            auto& sdfCollPending = reg.emplace<CSdfColliderPending>(e);
            sdfCollPending.gridSdfShape = gridSdfShape;
          } else {
            LogSdfColliderDiagnostics(sdfCollider, std::string(params.name));
          }
        } else {
          MOCHI_ERROR_SET(error, "Not a supported shape type for ColliderType::Sdf");
        }
      } break;
      case ColliderType::PointCloud: {
        MOCHI_ERROR_SET(error, "ColliderType::PointCloud is not supported for rigid actors");
      } break;
      case ColliderType::Auto: {
        MOCHI_ASSERT(false, "ColliderType::Auto should have been resolved before reaching here");
      } break;
      case ColliderType::None: {
        // Other actors won't be able to collide with this one. That is legal. Contact might still
        // be possible via sync coupling, as long as the other actor has a valid ColliderType.
      } break;
      default:
        MOCHI_ASSERT(false, "Invalid ColliderType");
        break;
    }
    MOCHI_ERROR_RETURN(error);
  }

  // This bounding volume contains all nodes of the mesh and the shape of the collider.
  reg.emplace<CBoundingVolume<TimeStep::Current>>(e, bv);
  reg.emplace<CBoundingVolume<TimeStep::Previous>>(e, bv);

  // Rigid velocity component
  auto& rigidVelocity = reg.emplace<CPrevRigidVelocity>(e);

  auto& exportParams = reg.emplace<CRigidExportParams>(e);
  exportParams.boundaryElementType = params.boundaryElementType;
  exportParams.boundarySubsampling = params.boundarySubsampling;

  if (params.isStatic) {
    reg.emplace<TagStaticActor>(e);
    rigidVelocity.centerOfMassLocal = GetBoundingSphere(bv).VGetCenter();
  } else {
    reg.emplace<TagRigidActor>(e); // currently implies dynamic
    InitRigidActor_Dynamic(reg, e, params, useContact, shapePtr.get(), error);
  }
}

void mochi::InitDifferentiableRigidActor(
    entt::registry& reg,
    entt::entity e,
    bool isArticulatedLink) {
  MOCHI_ASSERT_VERBOSE(!reg.any_of<TagStaticActor>(e), "Do not call on static actors");

  // Emplace general components needed for differentiability. Stand-alone rigid actors only.
  if (!isArticulatedLink) {
    ecs::InvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
        &EmplaceDifferentiabilityComponents, reg, e, RigidSize::kDAll);
  }

  // Emplace components needed for differentiable contact forces. Rigid and link actors.
  ecs::InvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      &EmplaceDifferentiableContactComponents, reg, e);
}

static void ApplyRigidBoundaryConditions(
    ecs::RequiredTag<TagRigidActor>,
    CDofPositionsBC const* inWorldBC,
    CDirichletBC<real>& outWorldBC) {
  // Copy boundary conditions. They are already in world-space.
  if (inWorldBC) {
    outWorldBC.poseIndices = inWorldBC->poseIndices;
    outWorldBC.dofIndices = inWorldBC->dofIndices;
    outWorldBC.poseValues = inWorldBC->poseValues;
    outWorldBC.colValueIndices = inWorldBC->colValueIndices;
  } else {
    outWorldBC.Clear();
  }
}

void mochi::PreStepRigidActorAsync(entt::registry& reg, entt::entity e) {
  MOCHI_PROFILE_SCOPE();

  // Apply boundary conditions
  ecs::TryInvokeOnEntity(&ApplyRigidBoundaryConditions, reg, e);
}

void mochi::rigid::RecordState(
    ecs::Included<TagRigidActor>,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidVel<TimeStep::Current> const& currVel,
    CRecordingData& outData) {
  // Record rigid state and velocity
  auto pos = currPose.value.GetTranslation();
  auto rot = currPose.value.GetRotation().ToReal4();
  auto vcom = ToReal3(currVel.value.GetVCom());
  auto omega = ToReal3(currVel.value.GetOmegaAndVSym().first);
  auto vsym = ToNdArraySym3x3(currVel.value.GetOmegaAndVSym().second);
  RecordAttribute<real>("translationCom", pos, outData);
  RecordAttribute<real>("rotationCom", rot, outData);
  RecordAttribute<real>("vcom", vcom, outData);
  RecordAttribute<real>("omega", omega, outData);
  RecordAttribute<real>("vsym", Flatten(vsym), outData);
}

void mochi::rigid::UpdateVSym(
    ecs::Included<TagRigidActor>,
    ecs::CtxGlobal<CSceneTime const> time,
    CRigidVel<TimeStep::Current>& outVel) {
  outVel.value.UpdateVSymIfDirty(static_cast<real>(time->DeltaTime()));
}

void mochi::rigid::TransportGradient(
    ColumnVectorView<real const> delta,
    ColumnVectorView<real> outGradient,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagArticulatedLinkActor>,
    CDofOffset const& dofOffset) {
  TransportInputOfLieJacobian(
      delta.MiddleRows<RigidSize::kDAll>(dofOffset.dofsOffset, RigidSize::kDAll),
      outGradient.MiddleRows<RigidSize::kDAll>(dofOffset.dofsOffset, RigidSize::kDAll).Transpose());
}

namespace mochi::rigid {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CRigidBodyInertia>(reg);
  ecs::RegisterComponent<CRigidVel<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CRigidVel<TimeStep::Previous>>(reg);
  ecs::RegisterComponent<CRigidVel<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CIntegrationRigidVels>(reg);
  ecs::RegisterComponent<CRigidExportParams>(reg);
}

real GetActorMass(entt::registry const& reg, entt::entity actor) {
  MOCHI_ASSERT(reg.all_of<TagRigidActor>(actor), "Expected rigid actor.");
  MOCHI_ASSERT(!reg.any_of<TagStaticActor>(actor), "Cannot get mass of static actor.");
  return reg.get<CRigidBodyInertia>(actor).GetMass();
}
} // namespace mochi::rigid
