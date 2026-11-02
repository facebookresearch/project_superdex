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
#include "mochi_discretization_functions.h"
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
 * inertia and damping into a single ElOpFnType over the 6-node bending stencil.
 *
 * @note Inertia and mass damping use each triangle's mass matrix in closed form; their dresidual is
 * that matrix scaled by 1/dt² + @p massDampingScale.
 *
 * @note Gravity, inertia and mass damping operate on the 3-node triangle sub-stencil (nodes 0–2,
 * DoFs 0–8). Their 9-DoF contributions are extracted from / accumulated into the 18-DoF stencil
 * vectors. Stress (membrane + bending) operates on the full 6-node stencil.
 *
 * @param[in] predTarget Predicted target u₀ + dt·v₀ of each DoF. Read only for a finite @p dtStage,
 * for the objective and residual.
 */
template <typename TriElementT>
[[nodiscard]] ElOpFnType<ShellStencilElement> MakeBatchedBodyOp(
    Local2GlobalMap const& l2g,
    Span<TriElementT const> triElements,
    bool hasGravity,
    real membraneLambda,
    real membraneMu,
    real bendingAlpha,
    real bendingBeta,
    real membraneCollapseBarrierStiffness,
    Real3 gravity,
    real density,
    Span<real const> predTarget,
    Span<real const> stageStartDispl,
    real dtStage,
    real massDampingScale,
    real stiffnessDampingFactor) {
  static_assert(TriElementT::kSpaceDim == ShellTriangleElement::kSpaceDim);
  static_assert(TriElementT::kNumDofs == ShellTriangleElement::kNumDofs);
  MOCHI_ASSERT_VERBOSE(!triElements.empty(), "Empty shell discretization.");
  MOCHI_ASSERT_VERBOSE(
      l2g.GetNumElements() == isize(triElements),
      "L2G element count must match the discretization.");
  constexpr int kBatchSize = kDefaultFemBatchSize;
  constexpr int kStencilDofs = fem::kBendingStencilDofs;
  real const dtfi2 = 1_r / Sqr(dtStage);

  // Inertia vanishes in quasistatic steps, whose time step is infinite.
  bool const hasInertia = IsFinite(dtStage);

  real const massScale = dtfi2 + massDampingScale;

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
    bool const hasMassDamping = hasInertia && massDampingScale > 0_r;
    bool const hasStiffnessDamping = stiffnessDampingFactor > 0_r;

    // --- Gather the stencil global nodes and the stage-start displacements once ---
    //
    // Stress uses the global nodes of the full 6-node bending stencil. Stiffness damping reads the
    // stage-start displacement of the full stencil (18 DoFs) for its energy and residual. The
    // gravity energy and mass damping read that of the central triangle only (its first 9 DoFs), so
    // at most those are gathered when stiffness damping is off.
    using V = BatchReal<kBatchSize>;
    NdArray<V, kStencilDofs> stageStartDispBatch MOCHI_NO_INIT;
    NdArray<int, fem::kBendingStencilNodes, kBatchSize> stencilGlobalNodes MOCHI_NO_INIT;
    int const numStageStartDofs = hasStiffnessDamping
        ? kStencilDofs
        : (((hasGravity && outEnergy) || hasMassDamping) ? kTriangleDofs : 0);

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

    // --- Gravity, inertia and mass damping on the 3-node triangle sub-stencil (DoFs 0-8) ---
    if (outEnergy || outRes) {
      // Nodes 0-2 (DoFs 0-8) of the bending stencil are the triangle vertices.
      auto const triDisp = MakeConstSpan(displ).subspan(0, kTriangleDofs);
      auto const triStageStartDisp = MakeConstSpan(stageStartDispBatch).subspan(0, kTriangleDofs);
      Span<BatchReal<kBatchSize>> triRes =
          outRes ? MakeSpan(*outRes).subspan(0, kTriangleDofs) : Span<BatchReal<kBatchSize>>{};

      if (hasGravity) {
        out |= fem::GravityWork<kBatchSize>(
            elemIndices,
            triElements,
            triDisp,
            triStageStartDisp,
            outEnergy,
            triRes,
            gravity,
            density);
      }

      if (hasInertia) {
        // Gather the predicted target with the stride-18 padded L2G. GatherBatchElementVector reads
        // only the first 3 nodes at the stride-18 offsets.
        fem::BatchElementVector<kBatchSize, ShellTriangleElement> predTargetBatch MOCHI_NO_INIT;
        fem::GatherBatchElementVector<
            kBatchSize,
            ShellTriangleElement,
            fem::kSpaceDim3,
            kStencilDofs>(indicesFlat, elemIndices, predTarget, predTargetBatch);
        out |= fem::InertiaWork<kBatchSize>(
            elemIndices,
            triElements,
            triDisp,
            MakeConstSpan(predTargetBatch),
            outEnergy,
            triRes,
            density,
            dtfi2);
      }

      if (hasMassDamping) {
        // Mass damping: α/dt · M · (d - d_stageStart). Reuse the first 9 DoFs of the gathered
        // stage-start displacement above.
        out |= fem::InertiaWork<kBatchSize>(
            elemIndices,
            triElements,
            triDisp,
            triStageStartDisp,
            outEnergy,
            triRes,
            density,
            massDampingScale);
      }
    }

    // --- Stress (membrane + bending) on the full 6-node stencil ---
    Span<Real3 const> meshNodes = triElements[0].coordinates;
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
        membraneCollapseBarrierStiffness,
        projectPsd,
        stiffnessDampingFactor,
        hasStiffnessDamping ? &stageStartDispBatch : nullptr);

    // --- Inertia and mass-damping dresidual on the 3-node triangle sub-stencil ---
    if (hasInertia && outDRes) {
      out |= fem::AddMassToDRes<kBatchSize, ShellStencilElement>(
          elemIndices, triElements, density * massScale, *outDRes);
    }

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
    CFemSurfaceDiscretizationP1Q1 const& femSurfDisc,
    CRootTransform const& rootTransform,
    CShellMaterialParams const& materialParams,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CDisplacementSlice<real, TimeStep::StageStart> const& stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart> const& stageStartVel,
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
 * @param[in] stageStartDisplacements Stage-start displacement DoFs in the actor's local frame.
 * @param[in,out] outObj Objective accumulator. If non-null, `dot(displacement -
 * stageStartDisplacement, force)` (in local coordinates) is subtracted from it.
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
    ColumnVectorView<real const> stageStartDisplacements,
    double* outObj,
    ColumnVectorView<real>* outRes);

//////////////// TODO: The following may be combine-able with soft-FEM counterparts. //////////////

void EntityIncrementStep(
    ecs::Included<TagShellActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CDisplacementSlice<real, TimeStep::Previous>& prevDispl,
    CVelocitySlice<real, TimeStep::Previous>& prevVel);

/*
 * Reconstructs the begin-of-step @ref TimeStep::Previous state from the integration history if
 * history is present and the integration method allows, otherwise it estimates the state from
 * constant velocity.
 */
void EntityReconstructPreviousStep(
    ecs::Included<TagShellActor>,
    ecs::CtxGlobal<CSceneTime const> time,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CVelocitySlice<real, TimeStep::Current> const& currVel,
    CIntegrationDisplacementSlices const& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default> const& intVels,
    CDisplacementSlice<real, TimeStep::Previous>& outPrevDispl,
    CVelocitySlice<real, TimeStep::Previous>& outPrevVel);

// Recenter the local frame on the pivot node. The pivot carries no rotation, so only translation is
// recentered.
void RecenterSolutionUsingPivotNode(
    ecs::Included<TagShellActor>,
    CRecenteringParams const& params,
    CRecenteringPivotNode const& pivot,
    CRootTransform& root,
    CDisplacementSlice<real, TimeStep::Current>& currDispl,
    CDisplacementSlice<real, TimeStep::Previous>& prevDispl,
    CIntegrationDisplacementSlices& intDispls,
    CBoundingVolume* bounds);

void EntityPreFirstStage(
    ecs::Included<TagShellActor>,
    ecs::OptionalCtxGlobal<TagAssemblyPreserveHistory const> preserveHistory,
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
    ecs::Excluded<TagUseDeformableContactSkin>,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CTimeIntegratorState const& intState,
    ContactAssemblyReg reg,
    CFemSurfaceDiscretization const& femDisc,
    CContactLocal2GlobalMap const& contactL2g,
    CContactNodalBasedStructure const& contactNbs,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CActiveCollisions</*kIsSync*/ false, TimeStep::Current>& collisions,
    CRootTransform const& rootTransform,
    CDeformablePointAsyncCollisionsResponse& outResponse,
    CActorSnle& outActorSnle);

// Update CBoundingVolume.localShape based on the deformation of the shell. kStep selects the
// displacement state used to compute the bounds.
template <TimeStep kStep>
void UpdateBounds(
    ecs::Included<TagShellActor>,
    ecs::Excluded<TagUseDeformableContactSkin>,
    CTriangularMesh const& meshComponent,
    CFinalDisplacementRef<kStep> const& solComponent,
    CPointCloudColliderParams const* pointCloudColliderParams,
    CBoundingVolume& outBounds) {
  static_assert(kStep == TimeStep::Current || kStep == TimeStep::StageStart);
  MOCHI_PROFILE_SCOPE();

  auto const nodeDisplacements = Unflatten<Real3 const>(solComponent.value.GetConstSpan());
  Aabb bounds =
      CalcAabbWithDisplacements(meshComponent.mesh->GetNodeCoordinates(), nodeDisplacements);
  if (pointCloudColliderParams) {
    bounds = GetAabb(ExpandShape(GetObb(bounds), pointCloudColliderParams->radius));
  }
  outBounds.localShape = GetObb(bounds);
}

void InitializeOnce(entt::registry& reg);

// Get the mass of a shell actor.
[[nodiscard]] real GetActorMass(entt::registry const& reg, entt::entity actor);

// Validate ShellMaterialParams for physical admissibility:
//  - All fields finite.
//  - membraneMu > 0 and membraneLambda > -membraneMu (positive 2D bulk modulus
//    membraneLambda + membraneMu).
//  - bendingBeta > 0 and bendingAlpha > -bendingBeta / 2 (analogous bound for bending
//    coefficients).
//  - density > 0.
//  - massDampingCoefficient >= 0.
//  - stiffnessDampingCoefficient >= 0.
//  - membraneCollapseBarrierScale >= 0.
void ValidateShellMaterialParams(experimental::ShellMaterialParams const& params, Error& error);

} // namespace mochi::shell
