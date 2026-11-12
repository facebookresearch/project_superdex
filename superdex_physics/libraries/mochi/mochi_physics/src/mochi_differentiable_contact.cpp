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
#include "mochi_contact_anchors.h"
#include "mochi_rigid.h"

#include <mochi_core/memory/filo_allocator.h>

#include <algorithm>
#include <tuple>
#include <utility>

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
      .frictionModel = simParams.experimentalEval.frictionModel,
      .elastoPlasticFriction = simParams.experimentalEval.elastoPlasticFriction};
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

// Whether the contacts of a collision read friction anchors of the colliding actor, whose force
// then depends on the anchor offsets.
static bool ReadsFrictionAnchors(
    FrictionAnchorReg reg,
    entt::entity collider,
    ContactParams const& params,
    ContactEvalConfig const& config) {
  return config.elastoPlasticFriction && params.coulombFrictionCoefficient > 0_r &&
      HasFrictionAnchors(reg, collider);
}

namespace {

// Geometry of a rigid contact pair at one time slice, to map per-contact vectors in the collider's
// local frame to rigid DoFs (translation of the COM, world-frame rotation) and back. Torques are
// taken about a pivot: the collider's COM for sync contact, which matches
// AssembleSyncRigidContactSums, and the collider's origin for async contact, whose collider is
// static.
struct RigidContactFrame {
  VMatrix3x3r rot; // World from collider rotation
  VMatrix3x3r rotT;
  Vec4r pivot{}; // In collider frame
  Vec4r collidingFromPivot{}; // Colliding actor COM minus pivot, in world frame
};

template <TimeStep kTimeStep, bool kIsSync>
RigidContactFrame
GetRigidContactFrame(ContactAssemblyReg reg, entt::entity colliding, entt::entity collider) {
  RigidContactFrame frame;
  Vec4r const comColliding =
      reg.get<CRigidState<kTimeStep> const>(colliding).value.VGetTranslation();
  if constexpr (kIsSync) {
    auto const& state = reg.get<CRigidState<kTimeStep> const>(collider).value;
    std::tie(frame.rot, frame.rotT) = ToVMatrix3x3_WithTranspose(state.GetRotation());
    frame.pivot = reg.get<CRigidBodyInertia const>(collider).GetCenterOfMassLocal();
    frame.collidingFromPivot = comColliding - state.VGetTranslation();
  } else {
    auto const& transform =
        GetRootTransform<kTimeStep>(reg, collider, reg.get<CRootTransform const>(collider));
    std::tie(frame.rot, frame.rotT) = ToVMatrix3x3_WithTranspose(transform.GetRotation());
    frame.collidingFromPivot = comColliding - transform.VGetTranslation();
  }
  return frame;
}

// Sums over the contacts of a collision of per-contact gradients g wrt the contact positions p, and
// of their torques about the pivot, in the collider frame.
struct RigidContactAdjointSums {
  Vec4r force{};
  Vec4r torque{};
};

MOCHI_FORCE_INLINE void AddContactAdjoint(
    RigidContactFrame const& frame,
    Vec4r const& p,
    Vec4r const& g,
    RigidContactAdjointSums& inOutSums) {
  inOutSums.force += g;
  inOutSums.torque += Cross3(p - frame.pivot, g);
}

// Adds J^T g, summed over the contacts of a collision, to the rigid DoF gradients of the colliding
// actor and, for sync contact, of the collider (nullptr for async contact), where J maps their
// rigid DoFs to the contact positions. Rotates the sums to world once.
void AddRigidContactAdjointSums(
    RigidContactFrame const& frame,
    RigidContactAdjointSums const& sums,
    ColumnVector<real>& inOutCollidingGrad,
    ColumnVector<real>* inOutColliderGrad) {
  auto const addToRigidDofs = [](ColumnVector<real>& inOutGrad, Vec4r const& f, Vec4r const& t) {
    real* const dofs = inOutGrad.data();
    // The translation reads and writes all 4 lanes; the rotation then overwrites the 4th.
    Vec4r const trans = Load<Vec4r>(dofs) + f;
    Vec4r const rot = Load<RigidSize::kDRot, Vec4r>(dofs + RigidSize::kDTrans) + t;
    Store(dofs, trans);
    Store<RigidSize::kDRot>(dofs + RigidSize::kDTrans, rot);
  };
  Vec4r const force = DotVecMat3x3(sums.force, frame.rotT);
  Vec4r const torque = DotVecMat3x3(sums.torque, frame.rotT);
  addToRigidDofs(inOutCollidingGrad, force, torque - Cross3(frame.collidingFromPivot, force));
  if (inOutColliderGrad) {
    addToRigidDofs(*inOutColliderGrad, -force, -torque);
  }
}

} // namespace

// Registry access of the back-propagation of the queried contact forces: contact assembly and,
// for sync contact, the contact gradients of the dynamic colliders. Async colliders are static, and
// per-entity systems, which run async contact, cannot write through a registry.
template <GradTarget kGradTarget, bool kIsSync>
using ContactForceAdjointReg = std::conditional_t<
    kIsSync,
    ContactAssemblyRegWith<CDiffContactGrad<kGradTarget>>,
    ContactAssemblyReg>;

// Actors whose queried contact forces are back-propagated. Async contacts are reported only by the
// colliding actor's own query, so only queried actors carry their adjoints. Sync contacts are also
// reported by the collider's query, so the adjoints of a colliding actor's collision may come from
// its collider.
template <bool kIsSync>
using ContactForceAdjointFilter = std::conditional_t<
    kIsSync,
    ecs::Included<TagRigidActor>,
    ecs::Included<TagRigidActor, CQueryActorContactForces>>;

// System for async contact, called directly for sync contact (see
// AccumulateAllSyncRigidContactForceAdjoints): back-propagates the queried contact forces of a
// rigid colliding actor, whose adjoints fBar are stored in forcePerUnitArea. Adds fBar^T df/dp,
// mapped to rigid DoFs, to the contact gradients of the colliding actor and, for sync contact, of
// the collider, and, with anchor adjoints, fBar^T df/dz to the adjoints of the anchor offsets that
// the step read.
template <GradTarget kGradTarget, bool kIsSync>
static void AccumulateRigidContactForceAdjoints(
    ContactForceAdjointReg<kGradTarget, kIsSync> reg,
    entt::entity e,
    ContactForceAdjointFilter<kIsSync>,
    ecs::Excluded<TagStaticActor>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CTimeIntegratorState const& intState,
    CActiveCollisions<kIsSync, TimeStep::Current> const& activeCollisions,
    CFrictionAnchors<AnchorSet::Active> const* activeAnchors,
    CFrictionAnchorIndex<AnchorSet::Active> const* anchorIndex,
    CDiffFrictionAnchorGrad<AnchorSet::Active>* outAnchorGrad,
    CDiffContactGrad<kGradTarget>& outGrad) {
  if (activeCollisions.empty()) {
    return;
  }
  ContactEvalConfig const config = RefineContactForceAdjointEvalConfig(
      reg, e, InitializeContactForceAdjointEvalConfig(simParams.value));
  if (kGradTarget != GradTarget::Current || (activeAnchors && activeAnchors->anchors.empty())) {
    outAnchorGrad = nullptr;
  }
  MOCHI_ASSERT_VERBOSE(
      !outAnchorGrad || (activeAnchors && anchorIndex), "Anchor adjoints require the anchors.");
  TimeStep constexpr kTimeStep = GetTimeStep<kGradTarget>();

  // Reserve for the largest collision up-front so the per-collision ResizeNoInit below never
  // reallocates, which the FILO allocator requires.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 32 * 1024);
  CollisionResponseResult response(&allocator);
  response.Reserve(activeCollisions, false, false, true, outAnchorGrad != nullptr);

  int anchorHint = 0;
  for (auto const& collision : activeCollisions) {
    // `forcePerUnitArea` is the container for contact-force adjoints, and it is allocated only if
    // contact queries were enabled for some actor in the contact pair.
    ContactDetectionResult const& query = collision.collisionResult;
    int const numContacts = isize(query.forcePerUnitArea);
    if (numContacts <= 0) {
      continue;
    }
    MOCHI_ASSERT_VERBOSE(isize(query.sampleIndices) == numContacts, "Unexpected size");
    entt::entity const collider = collision.colliderEntity;
    CDiffContactGrad<kGradTarget>* outColliderGrad = nullptr;
    if constexpr (kIsSync) {
      outColliderGrad = &reg.template get<CDiffContactGrad<kGradTarget>>(collider);
    }

    // Jacobians of the contact forces wrt the contact positions and, with anchor adjoints, wrt the
    // anchor offsets.
    ContactParams const params = GetContactPairParams(reg, e, collider);
    bool const readsAnchors = outAnchorGrad && ReadsFrictionAnchors(reg, collider, params, config);
    response.ResizeNoInit(numContacts, false, false, true, readsAnchors);
    ComputeCollisionResponse<kGradTarget>(
        query, params, config, intState.dtStage, false, false, true, response, readsAnchors);

    // The queried force also depends on the anchor offset z_n that the step read, so
    // zBar_n += (df/dz)^T fBar. New contacts and contacts cleared by a placement read a zero offset
    // and have no input-anchor adjoint.
    if (readsAnchors) {
      auto const colliderId = entt::to_integral(collider);
      for (int i = 0; i < numContacts; ++i) {
        int const anchor = FindFrictionAnchorFromHint(
            *anchorIndex,
            MakeConstSpan(activeAnchors->anchors),
            query.sampleIndices[i],
            colliderId,
            anchorHint);
        if (anchor != kSentinelIndex) {
          outAnchorGrad->grads[anchor] += ToReal3(DotVecMat3x3(
              ToSimd(query.forcePerUnitArea[i]), SimdSymToFull(response.dforceAnchor[i])));
        }
      }
    }

    // Gradient wrt the contact positions, mapped to the rigid DoFs.
    RigidContactFrame const frame = GetRigidContactFrame<kTimeStep, kIsSync>(reg, e, collider);
    RigidContactAdjointSums sums;
    for (int i = 0; i < numContacts; ++i) {
      AddContactAdjoint(
          frame,
          ToSimd(GetCollidingPosition<kTimeStep>(query, i)),
          DotVecMat3x3(ToSimd(query.forcePerUnitArea[i]), response.dforce[i]),
          sums);
    }
    AddRigidContactAdjointSums(frame, sums, outGrad, outColliderGrad);
  }
}

// Sync contact also writes the contact gradients of the collider, so the actors are processed one
// after the other.
template <GradTarget kGradTarget>
static void AccumulateAllSyncRigidContactForceAdjoints(
    entt::registry& reg,
    Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();
  auto const& simParams = reg.ctx<CSimulationParams const>();
  for (auto e : actors) {
    auto const* activeCollisions =
        reg.try_get<CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const>(e);
    if (!activeCollisions) {
      continue;
    }
    AccumulateRigidContactForceAdjoints<kGradTarget, /*kIsSync*/ true>(
        reg,
        e,
        {},
        {},
        simParams,
        reg.get<CTimeIntegratorState const>(e),
        *activeCollisions,
        reg.try_get<CFrictionAnchors<AnchorSet::Active> const>(e),
        reg.try_get<CFrictionAnchorIndex<AnchorSet::Active> const>(e),
        reg.try_get<CDiffFrictionAnchorGrad<AnchorSet::Active>>(e),
        reg.get<CDiffContactGrad<kGradTarget>>(e));
  }
}

template <GradTarget kGradTarget>
static void AccumulateIslandContactForceAdjoints(
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  // Handle async contact per actor
  ecs::InvokeForEach(
      &AccumulateRigidContactForceAdjoints<kGradTarget, /*kIsSync*/ false>,
      reg,
      descendants.rigidActors);

  // Handle sync contact per island
  AccumulateAllSyncRigidContactForceAdjoints<kGradTarget>(reg, descendants.rigidActors);
}

/**************************************************************************
  Friction anchor adjoints (elasto-plastic friction). A step reads the anchor offsets z_n and
  advances them to z_{n+1} = R(z_n + p - p0) (AdvanceFrictionAnchorOffsets). On top of the existing
  solve and products, back-propagating it adds:
  - Through the anchor advance, before the solve (AccumulateAnchorAdvanceAdjoints): the adjoint of
    z_{n+1} back through R, to the right-hand side of the solve through p
    (CDiffContactGrad<Current>), to the old state through p0 (CDiffContactGrad<Previous>), and to
    z_n.
  - Through the solve, after it (AccumulateSolveAnchorAdjoints): (dr/dz_n)^T lambda, the dependence
    of the solve on z_n.
  The adjoint of z_n is stored per anchor (CDiffFrictionAnchorGrad) for the step before.
*/

namespace {

// Registry access of the back-propagation of the anchor advance: contact assembly and, for sync
// contact, the contact-force adjoints of the dynamic colliders. Async colliders are static, and
// per-entity systems, which run async contact, cannot write through a registry.
template <bool kIsSync>
using AnchorAdvanceAdjointReg = std::conditional_t<
    kIsSync,
    ContactAssemblyRegWith<
        CDiffContactGrad<GradTarget::Current>,
        CDiffContactGrad<GradTarget::Previous>>,
    ContactAssemblyReg>;

// System for async contact, called directly for sync contact (see
// AccumulateAllSyncAnchorAdvanceAdjoints): back-propagates the anchor advance of one contact set
// (AdvanceFrictionAnchorOffsets), z_{n+1} = R(w) with w = z_n + p - p0, using its evaluation
// configuration.
template <bool kIsSync>
void AccumulateAnchorAdvanceAdjoints(
    AnchorAdvanceAdjointReg<kIsSync> reg,
    entt::entity e,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CActiveCollisions<kIsSync, TimeStep::Current> const& activeCollisions,
    CFrictionAnchors<AnchorSet::Active> const& active,
    CFrictionAnchorIndex<AnchorSet::Active> const& activeIndex,
    CFrictionAnchors<AnchorSet::Pending> const& pending,
    CFrictionAnchorIndex<AnchorSet::Pending> const& pendingIndex,
    CDiffFrictionAnchorGrad<AnchorSet::Pending> const& pendingGrad,
    CDiffFrictionAnchorGrad<AnchorSet::Active>& activeGrad,
    CDiffContactGrad<GradTarget::Current>& outGradCurr,
    CDiffContactGrad<GradTarget::Previous>& outGradPrev) {
  // Empty Pending gradients: no later step was back-propagated.
  if (pendingGrad.grads.empty()) {
    return;
  }
  ContactEvalConfig const advanceConfig =
      MakeAnchorAdvanceEvalConfig(simParams.value.experimentalEval, reg, e);
  auto const& activeAnchors = active.anchors;
  auto const& pendingAnchors = pending.anchors;
  MOCHI_ASSERT_VERBOSE(pendingGrad.grads.size() == pendingAnchors.size());
  MOCHI_ASSERT_VERBOSE(activeGrad.grads.size() == activeAnchors.size());

  // Reserve for the largest collision up-front so the per-collision resize below never
  // reallocates, which the FILO allocator requires.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 8 * 1024);
  DynamicArray<Real3> wBar(&allocator);
  int maxContacts = 0;
  for (auto const& collision : activeCollisions) {
    maxContacts = Max(maxContacts, isize(collision.collisionResult.sampleIndices));
  }
  wBar.reserve(maxContacts);
  int pendingHint = 0;
  int activeHint = 0;
  for (auto const& collision : activeCollisions) {
    entt::entity const collider = collision.colliderEntity;
    auto const colliderId = entt::to_integral(collider);
    auto const& result = collision.collisionResult;
    int const numContacts = isize(result.sampleIndices);
    if (numContacts == 0 || !HasFrictionAnchors(reg, collider)) {
      continue;
    }

    // wBar = (dz_{n+1}/dw)^T zBar_{n+1}. A sample has at most one contact per collider with
    // friction anchors, so each contact advances its own anchor.
    wBar.resize_noinit(numContacts);
    bool hasAdjoint = false;
    for (int i = 0; i < numContacts; ++i) {
      int const anchor = FindFrictionAnchorFromHint(
          pendingIndex,
          MakeConstSpan(pendingAnchors),
          result.sampleIndices[i],
          colliderId,
          pendingHint);
      wBar[i] = anchor == kSentinelIndex ? Real3{} : pendingGrad.grads[anchor];
      hasAdjoint |= wBar[i] != Real3{};
    }
    if (!hasAdjoint) {
      continue;
    }
    AdvanceAnchorOffsetsVjp(
        result, GetContactPairParams(reg, e, collider), advanceConfig, MakeSpan(wBar));

    // dw/dp = I with p at the end of the step (q_{n+1}), dw/dp0 = -I with p0 at its stage start
    // (q_n), and dw/dz_n = I.
    RigidContactFrame const frameCurr =
        GetRigidContactFrame<TimeStep::Current, kIsSync>(reg, e, collider);
    RigidContactFrame const frameStageStart =
        GetRigidContactFrame<TimeStep::StageStart, kIsSync>(reg, e, collider);
    RigidContactAdjointSums sumsCurr;
    RigidContactAdjointSums sumsStageStart;
    for (int i = 0; i < numContacts; ++i) {
      if (wBar[i] == Real3{}) {
        continue;
      }
      Vec4r const g = ToSimd(wBar[i]);
      AddContactAdjoint(frameCurr, ToSimd(result.posColliding[i]), g, sumsCurr);
      AddContactAdjoint(
          frameStageStart, ToSimd(result.posCollidingStageStart[i]), -g, sumsStageStart);
      int const anchor = FindFrictionAnchorFromHint(
          activeIndex,
          MakeConstSpan(activeAnchors),
          result.sampleIndices[i],
          colliderId,
          activeHint);
      if (anchor != kSentinelIndex) {
        activeGrad.grads[anchor] += wBar[i];
      }
    }
    CDiffContactGrad<GradTarget::Current>* colliderGradCurr = nullptr;
    CDiffContactGrad<GradTarget::Previous>* colliderGradPrev = nullptr;
    if constexpr (kIsSync) {
      colliderGradCurr = &reg.template get<CDiffContactGrad<GradTarget::Current>>(collider);
      colliderGradPrev = &reg.template get<CDiffContactGrad<GradTarget::Previous>>(collider);
    }
    AddRigidContactAdjointSums(frameCurr, sumsCurr, outGradCurr, colliderGradCurr);
    AddRigidContactAdjointSums(frameStageStart, sumsStageStart, outGradPrev, colliderGradPrev);
  }
}

// Back-propagates the anchor advance of sync contact. It also writes the adjoints of the dynamic
// colliders, so the actors are processed one after the other, as in
// AccumulateAllSyncRigidContactForceAdjoints.
void AccumulateAllSyncAnchorAdvanceAdjoints(entt::registry& reg, Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();
  auto const& simParams = reg.ctx<CSimulationParams const>();
  for (auto e : actors) {
    auto const* activeCollisions =
        reg.try_get<CActiveCollisions</*kIsSync*/ true, TimeStep::Current> const>(e);
    auto* activeGrad = reg.try_get<CDiffFrictionAnchorGrad<AnchorSet::Active>>(e);
    if (!activeCollisions || !activeGrad) {
      continue;
    }
    AccumulateAnchorAdvanceAdjoints</*kIsSync*/ true>(
        reg,
        e,
        simParams,
        *activeCollisions,
        reg.get<CFrictionAnchors<AnchorSet::Active> const>(e),
        reg.get<CFrictionAnchorIndex<AnchorSet::Active> const>(e),
        reg.get<CFrictionAnchors<AnchorSet::Pending> const>(e),
        reg.get<CFrictionAnchorIndex<AnchorSet::Pending> const>(e),
        reg.get<CDiffFrictionAnchorGrad<AnchorSet::Pending> const>(e),
        *activeGrad,
        reg.get<CDiffContactGrad<GradTarget::Current>>(e),
        reg.get<CDiffContactGrad<GradTarget::Previous>>(e));
  }
}

// System: back-propagates the solve to the anchor offsets that the step read,
// zBar_n += (dr/dz_n)^T lambda = -w (df/dz)^T J lambda, for contacts that read an input anchor. New
// contacts and contacts cleared by a placement read a zero offset and have no input-anchor adjoint.
// The solve adjoint is lambda = -CDiffForceGrad, in the rigid DoFs of rigid actors and articulated
// links alike; forceGradReg reads it for sync colliders. It writes only the actor's own anchor
// adjoints, for async and sync contact.
template <bool kIsSync>
void AccumulateSolveAnchorAdjoints(
    ContactAssemblyReg reg,
    [[maybe_unused]] ecs::PartialRegistry<CDiffForceGrad const> forceGradReg,
    entt::entity e,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CActiveCollisions<kIsSync, TimeStep::Current> const& activeCollisions,
    CFrictionAnchors<AnchorSet::Active> const& activeAnchors,
    CFrictionAnchorIndex<AnchorSet::Active> const& index,
    CContactSamples<TimeStep::Current> const& samples,
    CTimeIntegratorState const& intState,
    CDiffForceGrad const& forceGrad,
    CDiffFrictionAnchorGrad<AnchorSet::Active>& outActiveGrad) {
  auto const& anchors = activeAnchors.anchors;
  if (activeCollisions.empty() || anchors.empty()) {
    return;
  }
  ContactEvalConfig const config = RefineContactForceAdjointEvalConfig(
      reg, e, InitializeContactForceAdjointEvalConfig(simParams.value));

  // Reserve for the largest collision up-front so the per-collision ResizeNoInit below never
  // reallocates, which the FILO allocator requires.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 32 * 1024);
  CollisionResponseResult response(&allocator);
  response.Reserve(activeCollisions, false, false, false, true);
  int hint = 0;
  for (auto const& collision : activeCollisions) {
    auto const& result = collision.collisionResult;
    entt::entity const collider = collision.colliderEntity;
    int const numContacts = isize(result.sampleIndices);
    if (numContacts == 0) {
      continue;
    }
    ContactParams const params = GetContactPairParams(reg, e, collider);
    if (!ReadsFrictionAnchors(reg, collider, params, config)) {
      continue;
    }

    // df/dz_n, the derivative of the per-unit-area force with respect to the anchor offset.
    response.ResizeNoInit(numContacts, false, false, false, true);
    ComputeCollisionResponse<GradTarget::Current>(
        result, params, config, intState.dtStage, false, false, false, response, true);

    // J (-lambda), the contact displacement induced by the solve adjoint, in the collider frame:
    // the rigid displacement of the colliding actor minus that of the collider (static for async
    // contact), dp = t + r x (p - pivot), with the translation shifted from the colliding COM to
    // the pivot.
    RigidContactFrame const frame =
        GetRigidContactFrame<TimeStep::Current, kIsSync>(reg, e, collider);
    int constexpr kRot = RigidSize::kDTrans;
    Vec4r rotation = Load<RigidSize::kDRot, Vec4r>(forceGrad.data() + kRot);
    Vec4r translation = Load<RigidSize::kDTrans, Vec4r>(forceGrad.data()) -
        Cross3(rotation, frame.collidingFromPivot);
    if constexpr (kIsSync) {
      real const* const forceGradCollider = forceGradReg.get<CDiffForceGrad const>(collider).data();
      translation -= Load<RigidSize::kDTrans, Vec4r>(forceGradCollider);
      rotation -= Load<RigidSize::kDRot, Vec4r>(forceGradCollider + kRot);
    }
    translation = DotVecMat3x3(translation, frame.rot);
    rotation = DotVecMat3x3(rotation, frame.rot);
    auto const colliderId = entt::to_integral(collider);
    for (int i = 0; i < numContacts; ++i) {
      int const sample = result.sampleIndices[i];
      int const anchor =
          FindFrictionAnchorFromHint(index, MakeConstSpan(anchors), sample, colliderId, hint);
      if (anchor == kSentinelIndex) {
        continue;
      }
      Vec4r const dp = translation + Cross3(rotation, ToSimd(result.posColliding[i]) - frame.pivot);
      outActiveGrad.grads[anchor] +=
          ToReal3(DotVecMat3x3(dp, SimdSymToFull(response.dforceAnchor[i]))) *
          samples.weights[sample];
    }
  }
}

} // namespace

void mochi::ReverseFrictionAnchorDropAdjoints(
    CDiffFrictionAnchorGrad<AnchorSet::Active>& activeGrad,
    CDiffFrictionAnchorGrad<AnchorSet::Pending>& pendingGrad,
    CFrictionAnchors<AnchorSet::Active> const& activeAnchors,
    CFrictionAnchors<AnchorSet::Pending> const& pendingAnchors) {
  auto const& active = activeAnchors.anchors;
  auto const& pending = pendingAnchors.anchors;

  MOCHI_ASSERT_VERBOSE(activeGrad.grads.size() == active.size());

  auto const sameAnchor = [](FrictionAnchor const& a, FrictionAnchor const& b) {
    return a.sample == b.sample && a.collider == b.collider;
  };
  if (active.size() == pending.size()) {
    MOCHI_ASSERT_VERBOSE(std::ranges::equal(active, pending, sameAnchor));
    std::swap(activeGrad.grads, pendingGrad.grads);
  } else {
    pendingGrad.grads.resize_noinit(pending.size());

    int activeIndex = 0;
    for (int pendingIndex = 0; pendingIndex < isize(pending); ++pendingIndex) {
      if (activeIndex < isize(active) && sameAnchor(active[activeIndex], pending[pendingIndex])) {
        pendingGrad.grads[pendingIndex] = activeGrad.grads[activeIndex++];
      } else {
        pendingGrad.grads[pendingIndex] = Real3{};
      }
    }
    MOCHI_ASSERT_VERBOSE(
        activeIndex == isize(active), "Active anchors must be a subsequence of Pending anchors.");
  }

  activeGrad.grads.clear();
}

void mochi::AccumulateFrictionAnchorAdjoints(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  ecs::InvokeForEachGlobal(&AccumulateAnchorAdvanceAdjoints</*kIsSync*/ false>, reg);
  reg.view<CIslandDescendants const>().each([&](CIslandDescendants const& descendants) {
    AccumulateAllSyncAnchorAdvanceAdjoints(reg, descendants.rigidActors);
  });
}

void mochi::PropagateFrictionAnchorAdjoints(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  ecs::InvokeForEachGlobal(&AccumulateSolveAnchorAdjoints</*kIsSync*/ false>, reg);
  ecs::InvokeForEachGlobal(&AccumulateSolveAnchorAdjoints</*kIsSync*/ true>, reg);
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
