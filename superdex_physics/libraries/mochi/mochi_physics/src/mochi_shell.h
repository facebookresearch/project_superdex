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
#include "mochi_discretization_components.h"
#include "mochi_ecs.h"
#include "mochi_ecs_utils.h"
#include "mochi_materials.h"
#include "mochi_shape.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/element_operations/fem_gravity.h>
#include <mochi_core/element_operations/fem_inertia.h>
#include <mochi_core/element_operations/fem_shell.h>
#include <mochi_core/elements/eval_point.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/utils/transform_rt.h>

namespace mochi::shell {

int constexpr kNumStencilNodes = fem::kBendingStencilNodes;

/** @brief Alias for the 6-node shell bending assembly stencil tag. */
using ShellStencilElement = fem::ShellStencilElement;

/** @brief Alias for the 3-node shell triangle sub-stencil tag. */
using ShellTriangleElement = fem::ShellTriangleElement;

// Discretization info
using FemInfo = CFemSurfaceDiscretizationP1Q1;

/**
 * @brief Creates a batched shell assembly operator composing stress (membrane + bending), gravity,
 * and inertia into a single ElOpFnType over the 6-node bending stencil.
 *
 * @warning Inertia contributes to the objective and residual only. This operator does NOT assemble
 * the inertia dresidual. When assembling the dresidual, the caller must first initialize it with
 * the scaled mass matrix (see @ref AssembleBody). Otherwise the inertia dresidual is missing.
 *
 * @note Gravity and inertia operate on the 3-node triangle sub-stencil (nodes 0–2, DoFs 0–8). Their
 * 9-DoF contributions are extracted from / accumulated into the 18-DoF stencil vectors. Stress
 * (membrane + bending) operates on the full 6-node stencil.
 */
template <typename TriElementLow, typename TriElementHigh>
[[nodiscard]] ElOpFnType<ShellStencilElement> MakeBatchedBodyOp(
    Local2GlobalMap const& l2g,
    Span<TriElementLow const> lowElements,
    Span<TriElementHigh const> highElements,
    bool hasGravity,
    real membraneLambda,
    real membraneMu,
    real bendingAlpha,
    real bendingBeta,
    Real3 gravity,
    real density,
    real dtfi2,
    Span<real const> stageStartDispl,
    Span<real const> stageStartVel,
    real dtStage,
    real massDampingScale,
    real stiffnessDampingFactor) {
  static_assert(TriElementLow::kSpaceDim == ShellTriangleElement::kSpaceDim);
  static_assert(TriElementHigh::kSpaceDim == ShellTriangleElement::kSpaceDim);
  static_assert(TriElementLow::kNumDofs == ShellTriangleElement::kNumDofs);
  static_assert(TriElementHigh::kNumDofs == ShellTriangleElement::kNumDofs);
  MOCHI_ASSERT_VERBOSE(!lowElements.empty(), "Empty shell discretization.");
  MOCHI_ASSERT_VERBOSE(
      lowElements.size() == highElements.size(),
      "Low/high shell discretizations must have identical element counts.");
  MOCHI_ASSERT_VERBOSE(
      l2g.GetNumElements() == lowElements.size(),
      "L2G element count must match low discretization.");
  constexpr int kBatchSize = kDefaultFemBatchSize;
  constexpr int kStencilDofs = fem::kBendingStencilDofs;

  return [= /*All copies are inexpensive*/, &l2g](
             NdArray<int, kBatchSize> const& elemIndices,
             Span<int const> indicesFlat,
             fem::BatchElementVector<kBatchSize, ShellStencilElement> const& displ,
             BatchDouble<kBatchSize>* outEnergy,
             fem::BatchElementVector<kBatchSize, ShellStencilElement>* outRes,
             fem::BatchElementMatrix<kBatchSize, ShellStencilElement>* outDRes,
             bool projectPsd) -> bool {
    bool out = false;

    constexpr int kTriangleDofs = ShellTriangleElement::kNumDofs * ShellTriangleElement::kSpaceDim;
    bool const hasMassDamping = massDampingScale > 0_r;
    bool const hasStiffnessDamping = stiffnessDampingFactor > 0_r;

    // --- Gather displacements once for stress and mass damping ---
    //
    // We always gather the full 6-node bending stencil (18 DoFs) for stress; the first 9 DoFs
    // (central triangle) double as the mass-damping stage-start. Stencil global nodes are also
    // needed by the stress block for the shell assembly.
    using V = BatchReal<kBatchSize>;
    NdArray<V, kStencilDofs> stageStartDispBatch MOCHI_NO_INIT;
    NdArray<int, fem::kBendingStencilNodes, kBatchSize> stencilGlobalNodes MOCHI_NO_INIT;
    // Stage-start gather depth: full 18 DoFs for stiffness damping (also used as the central
    // triangle sub-span for mass damping), otherwise just 9 DoFs for mass damping alone.
    int const numStageStartDofs =
        hasStiffnessDamping ? kStencilDofs : (hasMassDamping ? kTriangleDofs : 0);

    for (int b = 0; b < kBatchSize; ++b) {
      int const eleIdx = elemIndices[b];
      auto const stencil =
          fem::GlobalStencilIndices(l2g.GetGlobalIndices(eleIdx), l2g.GetStencilIndices(eleIdx));
      for (int n = 0; n < fem::kBendingStencilNodes; ++n) {
        stencilGlobalNodes[n][b] = stencil[n];
      }
    }
    if (numStageStartDofs > 0) {
      for (int dof = 0; dof < numStageStartDofs; ++dof) {
        int const n = dof / fem::kSpaceDim3;
        int const d = dof % fem::kSpaceDim3;
        alignas(alignof(V)) real staging[V::kSize]{};
        for (int b = 0; b < kBatchSize; ++b) {
          int const globalNode = stencilGlobalNodes[n][b];
          if (globalNode != kSentinelIndex) {
            staging[b] = stageStartDispl[globalNode * fem::kSpaceDim3 + d];
          }
        }
        stageStartDispBatch[dof] = Load<V>(staging);
      }
    }

    // --- Gravity and inertia on the 3-node triangle sub-stencil (DoFs 0-8) ---
    //
    // NOTE: Gravity has no DRes contribution. Inertia DRes is handled globally via the precomputed
    // mass matrix. Mass damping's DRes is also handled globally.
    if (outEnergy || outRes) {
      // Nodes 0-2 (DoFs 0-8) of the bending stencil are the triangle vertices.
      auto const triDisp = MakeConstSpan(displ).subspan(0, kTriangleDofs);
      Span<BatchReal<kBatchSize>> triRes =
          outRes ? MakeSpan(*outRes).subspan(0, kTriangleDofs) : Span<BatchReal<kBatchSize>>{};

      if (hasGravity) {
        out |= fem::GravityWork<kBatchSize>(
            elemIndices, lowElements, triDisp, outEnergy, triRes, gravity, density);
      }

      // Gather predicted target using the stride-18 padded L2G. GatherPredTarget reads only the
      // first 3 nodes at the correct stride-18 offsets.
      fem::BatchElementVector<kBatchSize, ShellTriangleElement> stageStartTarget;
      fem::GatherPredTarget<
          kBatchSize,
          ShellTriangleElement,
          /*kGatherVelocity*/ true,
          fem::kSpaceDim3,
          kStencilDofs>(
          indicesFlat, elemIndices, stageStartDispl, stageStartVel, dtStage, stageStartTarget);
      out |= fem::InertiaWork<kBatchSize>(
          elemIndices,
          highElements,
          triDisp,
          MakeConstSpan(stageStartTarget),
          outEnergy,
          triRes,
          density,
          dtfi2);

      if (hasMassDamping) {
        // Mass damping: α/dt · M · (d - d_stageStart). Reuse the first 9 DoFs of the gathered
        // stage-start displacement above.
        out |= fem::InertiaWork<kBatchSize>(
            elemIndices,
            highElements,
            triDisp,
            MakeConstSpan(stageStartDispBatch).subspan(0, kTriangleDofs),
            outEnergy,
            triRes,
            density,
            massDampingScale);
      }
    }

    // --- Stress (membrane + bending) on the full 6-node stencil ---
    Span<Real3 const> meshNodes = lowElements[0].coordinates;
    out |= fem::ShellWork<kBatchSize>(
        stencilGlobalNodes,
        meshNodes,
        displ,
        outEnergy,
        outRes,
        outDRes,
        membraneLambda,
        membraneMu,
        bendingAlpha,
        bendingBeta,
        projectPsd,
        stiffnessDampingFactor,
        hasStiffnessDamping ? &stageStartDispBatch : nullptr);

    return out;
  };
}

// Assemble just the volume term into CActorSnle.
void AssembleBody(
    AssemblyParams const& params, // external parameter
    ecs::Included<TagShellActor>,
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CFemSurfaceDiscretizationP1Q1 const& femLowVolDisc,
    CFemSurfaceDiscretizationP1Q3 const& femHighVolDisc,
    CRootTransform const& rootTransform,
    CShellMaterialParams const& materialParams,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CDisplacementSlice<real, TimeStep::StageStart> const& stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart> const& stageStartVel,
    CMassMatrix const& massMatrix,
    CExternalForces const& externalForces,
    CActorSnle& outActorSnle);

/**
 * @brief Add per-DoF external forces to the residual and objective for a shell actor with
 * 3 translational DoFs per node.
 *
 * @param[in] externalForces Sparse external-force component (DoF indices and force values in
 * world coordinates).
 * @param[in] worldFromLocal Rigid transform from the actor's local frame to world.
 * @param[in] displacements Current displacement DoFs in the actor's local frame.
 * @param[in,out] outObj Objective accumulator. If non-null, `dot(displacement, force)` (in local
 * coordinates) is subtracted from it.
 * @param[in,out] outRes Residual accumulator. If non-null, the local-frame force is subtracted
 * from the residual entries at the affected DoFs.
 *
 * @note Force values are interpreted as world-frame Cartesian forces and are rotated into the
 * actor's local frame before being applied.
 */
void AddExternalForces(
    CExternalForces const& externalForces,
    TransformRT const& worldFromLocal,
    ColumnVectorView<real const> displacements,
    double* outObj,
    ColumnVectorView<real>* outRes);

//////////////// TODO: The following may be combine-able with soft-FEM counterparts. //////////////

void EntityIncrementStep(
    ecs::Included<TagShellActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CDisplacementSlice<real, TimeStep::Previous>& prevDispl,
    CVelocitySlice<real, TimeStep::Previous>& prevVel);

void EntityPreFirstStage(
    ecs::Included<TagShellActor>,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Previous> const& prevDispl,
    CVelocitySlice<real, TimeStep::Previous> const& prevVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels);

void EntityPreStage(
    ecs::Included<TagShellActor>,
    CTimeIntegratorState const& intState,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CDisplacementSlice<real, TimeStep::StageStart>& stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart>& stageStartVel);

void EntityPostStage(
    ecs::Included<TagShellActor>,
    CConvergenceStatus const& convergence,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::StageStart> const& stageStartDispl,
    CDisplacementSlice<real, TimeStep::Current>& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels);

void EntityPostLastStage(
    ecs::Included<TagShellActor>,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Current>& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels);

void EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagShellActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CDisplacementSlice<real, TimeStep::Current>& currSol);

void EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagShellActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CDisplacementSlice<real, TimeStep::Current>& currSol);

void EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::Included<TagShellActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currSol);

void EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagShellActor>,
    CDisplacementSlice<real, TimeStep::Current>& currSol);

// TODO: Consider adding this as a case in ActorInterfaceImpl::SetMaterialParams once the shell
// implementation matures.
inline void SetMaterialParams(
    experimental::ShellMaterialParams const& inParams,
    CShellMaterialParams& outMaterial) {
  outMaterial = inParams;
}

void AssembleAsyncContact(
    AssemblyParams const& params, // external parameter
    entt::entity e,
    ecs::Included<TagShellActor, TagUseContact>,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CTimeIntegratorState const& intState,
    ContactAssemblyReg reg,
    CFemSurfaceDiscretization const& femDisc,
    CContactLocal2GlobalMap const& contactL2g,
    CContactNodalBasedStructure const& contactNbs,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& collisions,
    CRootTransform const& rootTransform,
    CDeformablePointAsyncCollisionsResponse& outResponse,
    CActorSnle& outActorSnle);

// Update CBoundingVolume<TimeStep::Current>.localShape based on the deformation of the shell. kStep
// defines the data to be used in the update, not the component storing the result. There's no
// CBoundingVolume<TimeStep::StageStart>, as it's not needed. We do bound checks in stage-start
// collision detection, but we can use CBoundingVolume<TimeStep::Current> for this.
template <TimeStep kStep>
void UpdateBounds(
    ecs::Included<TagShellActor>,
    CTriangularMesh const& meshComponent,
    CFinalDisplacementRef<kStep> const& solComponent,
    CPointCloudColliderParams const* pointCloudColliderParams,
    CBoundingVolume<TimeStep::Current>& outBounds) {
  static_assert(kStep == TimeStep::Current || kStep == TimeStep::StageStart);
  MOCHI_PROFILE_SCOPE();
  auto const& sol = solComponent.value;
  auto nodeCoordinates = meshComponent.mesh->GetNodeCoordinates();
  auto nodeDisplacements = Unflatten<Real3 const>(sol.GetConstSpan());
  Obb bounds = GetObb(CalcAabbWithDisplacements(nodeCoordinates, nodeDisplacements));
  // NOTE: Contact padding will be automatically added elsewhere if shell-actors become first-class
  // colliders without separate contact collision detection and parameters.
  if (pointCloudColliderParams) {
    bounds = ExpandShape(bounds, pointCloudColliderParams->radius);
  }
  outBounds.localShape = bounds;
}

void InitializeOnce(entt::registry& reg);

// Get the mass of a shell actor.
[[nodiscard]] real GetActorMass(entt::registry const& reg, entt::entity actor);

// Validate ShellMaterialParams for physical admissibility:
//  - All seven fields finite.
//  - membraneMu > 0 and membraneLambda > -membraneMu (2D plane-strain bulk-modulus positivity).
//  - bendingBeta > 0 and bendingAlpha > -bendingBeta / 2 (analogous bound for bending
//    coefficients).
//  - density > 0.
//  - massDampingCoefficient >= 0.
//  - stiffnessDampingCoefficient >= 0.
void ValidateShellMaterialParams(experimental::ShellMaterialParams const& params, Error& error);

} // namespace mochi::shell
