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
#include <mochi_core/element_operations/fem_inertia.h>
#include <mochi_core/element_operations/fem_traction.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/linear_algebra/utils/assembly.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi {

// Forwards
struct CRigidTransformEval;

// To be run pre-simulation step for all deformable actors
void PreStepDeformableActorAsync(entt::registry& reg, entt::entity e);

// This namespace is intended to contain code that is shared between different types of deformable
// actors, including (volume) soft actors, shell actors and rod actors.
namespace deformable {

template <typename BlockSparseMatViewT, typename ElemMassMatT>
void SetZeroMassMatrix(
    BlockSparseMatViewT& outMassMatrix,
    Span<ElemMassMatT> outPerElemMassMatrix) {
  outMassMatrix.SetZero();
  Fill(outPerElemMassMatrix, ElemMassMatT{});
}

/**
 * @brief Computes the row-sum lumped mass matrix from a sparse mass matrix.
 *
 * @param[in] sparsity The sparsity pattern of the mass matrix.
 * @param[in] massMatrix The full sparse mass matrix values (must match sparsity pattern).
 * @param[out] outLumpedMassMatrix The output lumped mass matrix values.
 */
void ComputeLumpedMassMatrix(
    CFullSparsityPattern const& sparsity,
    CMassMatrix const& massMatrix,
    CLumpedMassMatrix& outLumpedMassMatrix);

// (Actors with a deforming surface) Set up normals of all active collision points.
template <ContactType kContactType, typename DiscretizationT>
void SetupActiveCollisionNormals(
    ecs::Excluded<TagShellActor, TagRodActor>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    DiscretizationT const& femDisc,
    CFinalDisplacementRef<TimeStep::Current> const& currentDispl,
    CFinalDisplacementRef<TimeStep::StageStart> const& stageStartDispl,
    CRootTransform const& transform,
    CContactSamples<TimeStep::Current> const& contactPositions,
    CActiveCollisions<kContactType, TimeStep::Current>& activeCollisions);

template <typename DiscretizationType, int kNumFields>
void ComputeAsyncContactResponse(
    ContactAssemblyReg reg,
    entt::entity e,
    ExperimentalEvalParams const& experimentalEval,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    DiscretizationType const& femBoundaryDisc,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& collisions,
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
    CDeformablePointAsyncCollisionsResponse const& outResponse,
    Span<real const> bdFaceWeights) {
  using V = BatchReal<kBatchSize>;
  using Vd = BatchDouble<kBatchSize>;
  using V3 = BatchReal3<kBatchSize>;
  return [&outResponse, elements, bdFaceWeights](
             NdArray<int, kBatchSize> const& elemIndices,
             Span<int const> /*indicesFlat*/,
             Vd* outEnergy,
             fem::BatchElementVector<kBatchSize, ElementT, kNumFields>* outRes,
             fem::BatchElementMatrix<kBatchSize, ElementT, kNumFields>* outDRes,
             bool /*projectPsd*/) -> bool {
    auto batchedContactFn = [&outResponse](
                                NdArray<int, kBatchSize> const& eleIndices,
                                int q,
                                Vd* cbEnergy,
                                V3* cbForce,
                                NdArray<V3, 3>* cbDForce,
                                NdArray<bool, kBatchSize>& hasForce) {
      Real3 sf[V::kSize]{};
      int respIdxByLane[kBatchSize] MOCHI_NO_INIT;
      for (int b = 0; b < kBatchSize; ++b) {
        int const sampleIdx = eleIndices[b] * ElementT::kNumQuadPoints + q;
        int const respIdx = outResponse.GetResponseIndexFromSampleIndex(sampleIdx);
        respIdxByLane[b] = respIdx;
        hasForce[b] = (respIdx != CDeformablePointAsyncCollisionsResponse::kInvalidIndex);
        if (!hasForce[b]) {
          continue;
        }
        if (cbEnergy) {
          *cbEnergy = Set(*cbEnergy, b, outResponse.GetEnergy(respIdx));
        }
        if (cbForce) {
          sf[b] = outResponse.GetGradient(respIdx);
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
                sd[b] = outResponse.GetHessian(respIdxByLane[b])[r][c];
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

/**
 * @brief Creates a batched op that adds precomputed per-element mass matrices into the dresidual.
 */
template <class ElementT, int kBatchSize = kDefaultFemBatchSize, size_t kMassDof>
[[nodiscard]] auto MakeAddMassMatrixToDResOp(
    Span<NdArray<real, kMassDof, kMassDof> const> mmPerElem) {
  return [mmPerElem](
             NdArray<int, kBatchSize> const& elemIndices,
             Span<int const> /*indicesFlat*/,
             BatchDouble<kBatchSize>* /*outEnergy*/,
             fem::BatchElementVector<kBatchSize, ElementT>* /*outRes*/,
             fem::BatchElementMatrix<kBatchSize, ElementT>* outDRes,
             bool /*projectPsd*/) -> bool {
    MOCHI_ASSERT_VERBOSE(outDRes != nullptr, "outDRes must be non-null.");
    return fem::AddMassMatrixToDRes<kBatchSize, ElementT>(elemIndices, mmPerElem, *outDRes);
  };
}

// Compute the contact Jacobians as colliding actor
template <typename ActorTag, typename DiscretizationType>
void SetupCollidingJacobians(
    ecs::Included<ActorTag>,
    ecs::Excluded<TagRomActor, TagSoftSkinnedActor, TagUseVisualMeshContact>,
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
void EmplaceContactComponents(entt::registry& reg, entt::entity e, int numCollidingSamples);

void RecordState(
    CDisplacementSlice<real, TimeStep::Current> const* disp,
    CVelocitySlice<real, TimeStep::Current> const& vel,
    CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const* dispSkinned,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned> const* velSkinned,
    CRodPose<TimeStep::Current> const* rodPose,
    ecs::OptionalTag<TagSoftActor> isSoft,
    ecs::OptionalTag<TagShellActor> isShell,
    ecs::OptionalTag<TagRodActor> isRod,
    CRecordingData& outData);

void RecordRigidTransformEval(CRigidTransformEval const& eval, CRecordingData& outData);

void RecordingPipeline(entt::registry& reg, Span<entt::entity const> entities);

namespace details {

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
 * @param[in,out] outObj Optional objective accumulator.
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
      (externalForces.dofs[i + 2] == nodeStartIndex + 2))
    MOCHI_LIKELY {
      forceWorld = Load<3, Vec4r>(&(externalForces.forces[i]));
      extraConsumed = 2;
    }
  else {
    forceWorld = Set(Vec4r{}, component, externalForces.forces[i]);
  }
  Vec4r const forceLocal =
      DotVecMat3x3(forceWorld, worldFromLocalR); // worldFromLocalR^T * forceWorld
  if (outObj) {
    Vec4r const dispLocal = Load<3, Vec4r>(&displacements[nodeStartIndex]);
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
