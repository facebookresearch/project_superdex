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

#include "mochi_contact.h"
#include "mochi_rod_pose.h"
#include "mochi_simulation.h"

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/element_operations/fem_traction.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/vmatrix.h>

#include <optional>
#include <type_traits>

namespace mochi {

// Forwards
struct CRigidTransformEval;

// To be run pre-simulation step for all deformable actors
void PreStepDeformableActorAsync(entt::registry& reg, entt::entity e);

// This namespace is intended to contain code that is shared between different types of deformable
// actors, including (volume) soft actors, shell actors and rod actors.
namespace deformable {

// Update the maximum world-space nodal speed for an ordinary soft or shell actor.
void UpdateMaxGeometrySpeed(
    ecs::Included<TagDeformableActor>,
    ecs::Excluded<TagNestedSoftActor, TagRodActor>,
    CVelocitySlice<real, TimeStep::Current> const& velocity,
    CColliderInfo const& collider,
    CContactSkinningData const* contactSkinning,
    CConservativeStepBounds& outStepBounds);

// Index of the reference node closest to the centroid of all reference nodes.
[[nodiscard]] int FindRecenteringPivotNode(Span<Real3 const> referenceNodes);

// Translate the root transform so that a pivot node with local displacement `pivotDisplacement`
// returns to its reference position, moving the previous-step root transform and the local bounds
// (if any) along with the local frame. Returns the translation the caller must add to all stored
// local positions, including those of previous steps, or nullopt if recentering is disabled or the
// pivot has not moved beyond `params.translationEpsilon`. Velocities are unaffected by a
// translation of the local frame.
[[nodiscard]] std::optional<Real3> TranslateRootForRecentering(
    CRecenteringParams const& params,
    Real3 const& pivotDisplacement,
    CRootTransform& root,
    CBoundingVolume* bounds);

// (Actors with a deforming surface) Set up normals of all active collision points.
template <bool kIsSync, typename DiscretizationT>
void SetupActiveCollisionNormals(
    ecs::Excluded<TagShellActor, TagRodActor, TagUseDeformableContactSkin>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    DiscretizationT const& femDisc,
    CFinalDisplacementRef<TimeStep::Current> const& currentDispl,
    CFinalDisplacementRef<TimeStep::StageStart> const& stageStartDispl,
    CRootTransform const& transform,
    CContactSamples<TimeStep::Current> const& contactPositions,
    CActiveCollisions<kIsSync, TimeStep::Current>& activeCollisions);

template <typename DiscretizationType, int kNumFields>
void ComputeAsyncContactResponse(
    ContactAssemblyReg reg,
    entt::entity e,
    ExperimentalEvalParams const& experimentalEval,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    DiscretizationType const& femBoundaryDisc,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current>& collisions,
    CTimeIntegratorState const& intState,
    CRootTransform const& rootTransform,
    AssemblyParams const& params,
    CDeformablePointAsyncCollisionsResponse& outResponse,
    Span<bool const> allowedContactElementMask = {});

/**
 * @brief Creates a batched boundary (traction/contact) assembly operator.
 */
template <
    class ElementT,
    int kNumFields = ElementT::kSpaceDim,
    int kBatchSize = kDefaultFemBatchSize>
[[nodiscard]] NoDispElOpFnType<ElementT, kNumFields, kBatchSize> MakeBatchedBoundaryOp(
    Span<ElementT const> elements,
    CDeformablePointAsyncCollisionsResponse const& response,
    Span<real const> bdFaceWeights) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  return [&response, elements, bdFaceWeights](
             NdArray<int, kBatchSize> const& elemIndices,
             Span<int const> /*indicesFlat*/,
             Vd* outEnergy,
             fem::BatchElementVector<kBatchSize, ElementT, kNumFields>* outRes,
             fem::BatchElementMatrix<kBatchSize, ElementT, kNumFields>* outDRes,
             bool /*projectPsd*/) -> bool {
    auto batchedContactFn = [&response](
                                NdArray<int, kBatchSize> const& eleIndices,
                                int q,
                                Vd* cbEnergy,
                                V3* cbForce,
                                NdArray<V3, 3>* cbDForce,
                                NdArray<bool, kBatchSize>& hasForce) {
      // sf is only accessed when cbForce is non-null, so it only needs zeroing then.
      Real3 sf[V::kSize] MOCHI_NO_INIT;
      if (cbForce) {
        for (auto& force : sf) {
          force = {};
        }
      }
      int respIdxByLane[kBatchSize] MOCHI_NO_INIT;
      for (int b = 0; b < kBatchSize; ++b) {
        int const sampleIdx = eleIndices[b] * ElementT::kNumQuadPoints + q;
        int const respIdx = response.GetResponseIndexFromSampleIndex(sampleIdx);
        respIdxByLane[b] = respIdx;
        hasForce[b] = (respIdx != CDeformablePointAsyncCollisionsResponse::kInvalidIndex);
        if (!hasForce[b]) {
          continue;
        }
        if (cbEnergy) {
          *cbEnergy = Set(*cbEnergy, b, response.GetEnergy(respIdx));
        }
        if (cbForce) {
          sf[b] = response.GetGradient(respIdx);
        }
      }
      if (cbForce) {
        LoadTransposed(&sf[0][0], *cbForce);
      }
      if (cbDForce) {
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) {
            alignas(alignof(V)) real sd[V::kSize]{};
            for (int b = 0; b < kBatchSize; ++b) {
              if (hasForce[b]) {
                sd[b] = response.GetHessian(respIdxByLane[b])[r][c];
              }
            }
            (*cbDForce)[r][c] = Load<V>(sd);
          }
        }
      }
    };

    return fem::TractionWork<kBatchSize, ElementT, kNumFields>(
        elemIndices, elements, outEnergy, outRes, outDRes, batchedContactFn, bdFaceWeights);
  };
}

// Compute the contact Jacobians as colliding actor
template <typename ActorTag, typename DiscretizationType>
void SetupCollidingJacobians(
    ecs::Included<ActorTag>,
    ecs::Excluded<TagRomActor, TagNestedSoftActor, TagUseDeformableContactSkin>,
    DiscretizationType const& discretization,
    CRootTransform const& transform,
    CDofOffset const& dofOffset,
    CCollJacs<CollRole::Colliding>& outJacobians);

// Compute the contact Jacobians as collider actor
void SetupColliderJacobians(
    [[maybe_unused]] ecs::OptionalTag<TagSoftActor> isSoftActor,
    [[maybe_unused]] ecs::OptionalTag<TagShellActor> isShellActor,
    [[maybe_unused]] ecs::OptionalTag<TagRodActor> isRodActor,
    ecs::OptionalTag<TagRomActor> isRomActor,
    CDofOffset const& dofOffset,
    CCollJacs<CollRole::Collider>& outJacobians);

// Emplace components needed for contact of all deformable actors. Sub-types (soft volume, shell,
// articulated skin, etc.) may need additional components.
void EmplaceContactComponents(
    entt::registry& reg,
    entt::entity e,
    int numCollidingSamples,
    int numContactPartitions);

void RecordState(
    CDisplacementSlice<real, TimeStep::Current> const* disp,
    CVelocitySlice<real, TimeStep::Current> const& vel,
    CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const* dispSkinned,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned> const* velSkinned,
    CIntegrationVelocitySlices<DisplacementLayer::Skinned> const* integrationVelSkinned,
    CRodPose<TimeStep::Current> const* rodPose,
    ecs::OptionalTag<TagSoftActor> isSoft,
    ecs::OptionalTag<TagShellActor> isShell,
    ecs::OptionalTag<TagRodActor> isRod,
    CRecordingData& outData);

void RecordRigidTransformEval(CRigidTransformEval const& eval, CRecordingData& outData);

void RecordingPipeline(entt::registry& reg, Span<entt::entity const> entities);

namespace details {

// Writes each active collision's colliding normals in the collider frame. Normals of P1 elements
// are constant, so elementNormal(discretizationImpl, elementIndex, quadPointIndex) is evaluated
// once per element and must return the normal in the colliding actor's local frame.
template <typename DiscretizationT, typename ActiveCollisionsT, typename ElementNormalFn>
void SetupActiveCollisionNormalsFromElements(
    bool explicitNormals,
    CRootTransform const& transform,
    DiscretizationT const& femDisc,
    ActiveCollisionsT& activeCollisions,
    ElementNormalFn const& elementNormal) {
  auto const rotWorldFromCollidingT = ToVMatrix3x3Transpose(transform.worldFromLocal.GetRotation());

  ParallelForEach("SetupCollisionResultNormals", activeCollisions, 1, [&](auto& activeCollision) {
    auto& collisionResult = activeCollision.collisionResult;
    if (collisionResult.sampleIndices.empty()) {
      return;
    }

    collisionResult.normalColliding.resize_noinit(collisionResult.sampleIndices.size());

    // Initialize with the transform of the first contact. For rigid colliders, it is shared by all.
    auto const& jacColliderFromWorld = explicitNormals
        ? collisionResult.jacColliderFromWorldStageStart
        : collisionResult.jacColliderFromWorld;
    auto const jacColliderFromCollidingT =
        Dot3x3(rotWorldFromCollidingT, Transpose3x3(jacColliderFromWorld[0]));

    femDisc.Visit([&](auto const& discretizationImpl) {
      using DiscretizationImplT = std::decay_t<decltype(discretizationImpl)>;
      using BaseElementT =
          std::decay_t<decltype(discretizationImpl.femElements[0].GetBaseElement())>;
      static_assert(
          BaseElementT::kPolyOrder == 1, "Collision-normal caching requires P1 elements.");
      static int constexpr kNumQuads = DiscretizationImplT::kNumQuads;
      int prevElementIndex = -1;
      Vec4r normalColliding;

      for (int i = 0; i < isize(collisionResult.sampleIndices); ++i) {
        int const sampleIndex = collisionResult.sampleIndices[i];
        int const elementIndex = sampleIndex / kNumQuads;
        if (elementIndex != prevElementIndex) {
          normalColliding =
              elementNormal(discretizationImpl, elementIndex, sampleIndex % kNumQuads);
          prevElementIndex = elementIndex;
        }

        // Transform to the collider's local frame.
        if (jacColliderFromWorld.size() == 1) {
          collisionResult.normalColliding[i] =
              ToReal3(DotVecMat3x3(normalColliding, jacColliderFromCollidingT));
        } else {
          auto const normalWorld = DotVecMat3x3(normalColliding, rotWorldFromCollidingT);
          collisionResult.normalColliding[i] =
              ToReal3(DotMatVec3x3(jacColliderFromWorld[i], normalWorld));
        }
      }
    });
  });
}

/**
 * @brief Assemble one translational external-force entry (3 displacement DoFs) into the
 * residual and/or objective. Used by both deformable and rod external-force assembly.
 *
 * On the fast path (three consecutive in-order DoFs at `nodeStartIndex`, `nodeStartIndex+1`,
 * `nodeStartIndex+2`), returns 2 so the caller's outer loop can skip the already-consumed
 * components. Otherwise treats the single component as a sparse vector, rotates it, and
 * returns 0.
 *
 * @param[in] externalForces Sparse external-force component.
 * @param[in] i Index into `externalForces.dofs`/`forces` of the entry being assembled.
 * @param[in] component Translational component index in [0, 3) of `externalForces.dofs[i]`
 * relative to its node.
 * @param[in] nodeStartIndex DoF index of the first component of the node containing
 * `externalForces.dofs[i]`.
 * @param[in] worldFromLocalR Rotation from local to world (used to bring force into local frame).
 * @param[in] displacements Local-frame displacement DoFs of the actor.
 * @param[in] stageStartDisplacements Local-frame stage-start displacement DoFs of the actor.
 * @param[in,out] outObj Optional objective accumulator. The energy is relative to the stage start.
 * @param[in,out] outRes Optional residual accumulator.
 * @return Number of additional `externalForces` entries consumed beyond `i` (0 or 2). The caller
 * should advance its loop index by this amount in addition to its own `++i`.
 */
[[nodiscard]] MOCHI_FORCE_INLINE int AddTranslationalExternalForceEntry(
    CExternalForces const& externalForces,
    int i,
    int component,
    int nodeStartIndex,
    VMatrix3x3r const& worldFromLocalR,
    ColumnVectorView<real const> displacements,
    ColumnVectorView<real const> stageStartDisplacements,
    double* outObj,
    ColumnVectorView<real>* outRes) {
  int const numForces = isize(externalForces.dofs);

  // Treat external forces as vectors in world coordinates and transform at assembly time.
  //
  // Branching hint: full nodal forces are typically provided with components in order, but the
  // out-of-order/partial case is still handled correctly because rotation is linear and forces can
  // be superposed.
  Vec4r forceWorld MOCHI_NO_INIT;
  int extraConsumed = 0;
  if ((component == 0) && ((i + 2) < numForces) &&
      (externalForces.dofs[i + 1] == nodeStartIndex + 1) &&
      (externalForces.dofs[i + 2] == nodeStartIndex + 2)) MOCHI_LIKELY {
    forceWorld = Load<3, Vec4r>(&(externalForces.forces[i]));
    extraConsumed = 2;
  } else {
    forceWorld = Set(Vec4r{}, component, externalForces.forces[i]);
  }
  Vec4r const forceLocal =
      DotVecMat3x3(forceWorld, worldFromLocalR); // worldFromLocalR^T * forceWorld
  if (outObj) {
    Vec4r const dispLocal = Load<3, Vec4r>(&displacements[nodeStartIndex]) -
        Load<3, Vec4r>(&stageStartDisplacements[nodeStartIndex]);
    *outObj -= StaticCast<double>(Dot<3>(dispLocal, forceLocal));
  }
  if (outRes) {
    // Note that forces are subtracted from the residual.
    outRes->MiddleRows<3>(nodeStartIndex, 3) -= AsColumnVectorView<3>(forceLocal);
  }
  return extraConsumed;
}

} // namespace details

} // namespace deformable
} // namespace mochi
