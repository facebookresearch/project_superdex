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

#include "mochi_differentiable.h"

#include "mochi_articulated_body.h"
#include "mochi_contact.h"
#include "mochi_rigid.h"

#include <mochi_core/memory/filo_allocator.h>

using namespace mochi;

static ContactEvalConfig InitializeContactForceAdjointEvalConfig(
    CSimulationParams const& simParams) {
  // Match the contact force law used by the prepared query data. Do not apply solver-stabilization
  // approximations here: PSD projection and fitted Hessians are useful for nonlinear solves, but
  // this VJP needs the derivative of the actual force returned by the query.
  return ContactEvalConfig{
      .psdDRes = false,
      .explicitNormals = simParams.experimentalEval.explicitNormals,
      .fadeFriction = simParams.experimentalEval.fadeFriction,
      .implicitNormalForceForDissipation =
          simParams.experimentalEval.implicitNormalForceForDissipation,
      .useFittedHessian = false,
      .frictionModel = simParams.experimentalEval.frictionModel};
}

static ContactEvalConfig RefineContactForceAdjointEvalConfig(
    ContactAssemblyReg const& reg,
    entt::entity colliding,
    ContactEvalConfig const& base) {
  ContactEvalConfig config = base;
  config.addPadding = ShouldAddPenaltyPadding(reg.get<CColliderInfo const>(colliding).type);
  config.validCollidingNormals = ValidCollidingNormals(reg, colliding);
  return config;
}

template <GradTarget kGradTarget>
static void AccumulateAsyncContactForceAdjoints(
    ContactAssemblyReg reg,
    entt::entity e,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagStaticActor>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CTimeIntegratorState const& intState,
    CRigidState<GetTimeStep<kGradTarget>()> const& pose,
    CQueryActorContactForces const& /*queryActorContactForces*/,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& activeCollisions,
    CDiffContactGrad<kGradTarget>& outGrad) {
  if (activeCollisions.empty()) {
    return;
  }

  ContactEvalConfig config = InitializeContactForceAdjointEvalConfig(simParams.value);
  config = RefineContactForceAdjointEvalConfig(reg, e, config);

  // Gradient accumulators.
  Vec4r outGradientCom = {};
  Vec4r outGradientRot = {};

  // Match the forward async contact assembly path: use local FILO memory for the temporary
  // response.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 32 * 1024);
  CollisionResponseResult collisionResponse(&allocator);
  collisionResponse.Reserve(activeCollisions, false, true, true);

  auto const com = pose.value.VGetTranslation();
  for (auto& collision : activeCollisions) {
    ContactDetectionResult& contactQuery = collision.collisionResult;
    int const numContacts = isize(contactQuery.forcePerUnitArea);
    if (numContacts <= 0) {
      continue;
    }
    MOCHI_ASSERT_VERBOSE(isize(contactQuery.sampleIndices) == numContacts, "Unexpected size");

    // Compute the Jacobians of contact forces wrt contact positions.
    auto contactParams = GetContactPairParams(reg, e, collision.colliderEntity);
    collisionResponse.ResizeNoInit(numContacts, false, true, true);
    ComputeCollisionResponse<kGradTarget>(
        contactQuery,
        contactParams,
        config,
        intState.dtStage,
        false,
        false,
        true,
        collisionResponse);

    // Compute the gradient wrt contact positions. Reuse `force` for storage.
    for (int i = 0; i < numContacts; ++i) {
      collisionResponse.force[i] = ToReal3(
          DotVecMat3x3(ToSimd(contactQuery.forcePerUnitArea[i]), collisionResponse.dforce[i]));
    }

    // Accumulation of gradient terms for all contact points.
    // The implementation matches AssembleRigidBodyAsyncContactResponse.
    auto const& colliderTransform =
        reg.get<CRootTransform const>(collision.colliderEntity).worldFromLocalPrev;
    auto const trans = colliderTransform.VGetTranslation();
    auto const [rot, rotT] = ToVMatrix3x3_WithTranspose(colliderTransform.GetRotation());
    auto const comColliderSpace = DotVecMat3x3(com - trans, rot);

    Vec4r res = {};
    Vec4r skJRes = {};
    for (int i = 0; i < numContacts; ++i) {
      auto const posColliding =
          ToSimd(GetCollidingPosition<GetTimeStep<kGradTarget>()>(contactQuery, i));
      auto const jVec = posColliding - comColliderSpace;
      Vec4r const collRes = ToSimd(collisionResponse.force[i]);
      res += collRes;
      skJRes += Cross3(jVec, collRes);
    }

    outGradientCom += DotVecMat3x3(res, rotT);
    outGradientRot += DotVecMat3x3(skJRes, rotT);
  }

  ColumnVector<real, RigidSize::kDAll> grad;
  Store(grad.data(), outGradientCom);
  Store<RigidSize::kDRot>(grad.data() + RigidSize::kDTrans, outGradientRot);
  outGrad += grad;
}

template <GradTarget kGradTarget>
static void AccumulateAllSyncRigidContactForceAdjoints(
    entt::registry& reg,
    Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();

  ContactEvalConfig const configAllPairs =
      InitializeContactForceAdjointEvalConfig(reg.ctx<CSimulationParams const>());

  MOCHI_FILO_STACK_ALLOCATOR(allocator, 32 * 1024);

  for (auto e : actors) {
    // Only rigid-actor contact is supported in differentiability.
    if (!reg.all_of<TagRigidActor>(e)) {
      continue;
    }
    auto* activeCollisions =
        reg.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e);
    if (!activeCollisions) {
      continue;
    }

    // Params for this colliding actor
    real const dtStage = reg.get<CTimeIntegratorState const>(e).dtStage;
    ContactEvalConfig const configPair =
        RefineContactForceAdjointEvalConfig(reg, e, configAllPairs);

    // Target component for this colliding actor
    auto outGradA = AsView(reg.get<CDiffContactGrad<kGradTarget>>(e));

    // Reserve for the largest collision up-front so the per-collision ResizeNoInit below never
    // reallocates, which the FILO allocator requires.
    CollisionResponseResult response(&allocator);
    response.Reserve(*activeCollisions, false, true, true);

    // Traverse all its active collisions
    for (auto& coll : *activeCollisions) {
      // `forcePerUnitArea` is the container for contact-force adjoints, and it is allocated only if
      // contact queries were enabled for some actor in the contact pair.
      auto& query = coll.collisionResult;
      int const numContacts = isize(query.forcePerUnitArea);
      if (numContacts <= 0) {
        continue;
      }
      MOCHI_ASSERT_VERBOSE(isize(query.sampleIndices) == numContacts, "Unexpected size");

      // Only rigid-actor contact is supported in differentiability.
      auto const e2 = coll.colliderEntity;
      if (!reg.all_of<TagRigidActor>(e2)) {
        continue;
      }

      // Compute the Jacobians of contact forces wrt contact positions.
      auto const contactParams = GetContactPairParams(reg, e, e2);
      response.ResizeNoInit(numContacts, false, true, true);
      ComputeCollisionResponseRange<kGradTarget>(
          {0, numContacts},
          query,
          contactParams,
          configPair,
          dtStage,
          false,
          false,
          true,
          response);

      // Compute the gradient wrt contact positions. Reuse `force` for storage.
      for (int i = 0; i < numContacts; ++i) {
        response.force[i] =
            ToReal3(DotVecMat3x3(ToSimd(query.forcePerUnitArea[i]), response.dforce[i]));
      }

      // Accumulation of gradient terms for all contact points.
      // The implementation matches AssembleCollisionResponseRange_SyncRigid.
      TimeStep constexpr kTimeStep = GetTimeStep<kGradTarget>();
      Vec4r comA = reg.template get<CRigidState<kTimeStep> const>(e).value.VGetTranslation();
      auto const& stateB = reg.template get<CRigidState<kTimeStep> const>(e2).value;
      auto [rotB, rotBT] = ToVMatrix3x3_WithTranspose(stateB.GetRotation());
      Vec4r comB = stateB.VGetTranslation();
      Vec4r comBLocal = reg.template get<CRigidBodyInertia const>(e2).GetCenterOfMassLocal();

      Vec4r res = {};
      Vec4r skPRes = {};
      for (int s = 0; s < numContacts; ++s) {
        Vec4r const collRes = ToSimd(response.force[s]);
        res += collRes;
        auto posColliding = ToSimd(GetCollidingPosition<kTimeStep>(query, s));
        skPRes += Cross3(posColliding - comBLocal, collRes);
      }
      res = DotVecMat3x3(res, rotBT);
      skPRes = DotVecMat3x3(skPRes, rotBT);

      // Target component for the collider actor
      auto outGradB = AsView(reg.get<CDiffContactGrad<kGradTarget>>(e2));

      ColumnVector<real, RigidSize::kDAll> gradA;
      ColumnVector<real, RigidSize::kDAll> gradB;
      Store(gradA.data(), res);
      Store<RigidSize::kDRot>(gradA.data() + RigidSize::kDTrans, skPRes - Cross3(comA - comB, res));
      Store(gradB.data(), -res);
      Store<RigidSize::kDRot>(gradB.data() + RigidSize::kDTrans, -skPRes);

      outGradA += gradA;
      outGradB += gradB;
    }
  }
}

template <GradTarget kGradTarget>
static void AccumulateIslandContactForceAdjoints(
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  // Handle async contact per actor
  ecs::InvokeForEach(
      &AccumulateAsyncContactForceAdjoints<kGradTarget>, reg, descendants.rigidActors);

  // Handle sync contact per island
  AccumulateAllSyncRigidContactForceAdjoints<kGradTarget>(reg, descendants.rigidActors);
}

void mochi::AccumulateContactForceAdjoints(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  reg.view<CDiffContactGrad<GradTarget::Current>, CDiffContactGrad<GradTarget::Previous>>().each(
      [](CDiffContactGrad<GradTarget::Current>& outGradCurr,
         CDiffContactGrad<GradTarget::Previous>& outGradPrev) {
        outGradCurr.SetZero();
        outGradPrev.SetZero();
      });

  reg.view<CIslandDescendants const>().each([&](CIslandDescendants const& descendants) {
    AccumulateIslandContactForceAdjoints<GradTarget::Current>(reg, descendants);
    AccumulateIslandContactForceAdjoints<GradTarget::Previous>(reg, descendants);
  });
}
