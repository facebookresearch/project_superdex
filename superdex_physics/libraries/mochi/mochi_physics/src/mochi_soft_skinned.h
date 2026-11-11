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

#include "mochi_articulated_body.h"
#include "mochi_common_components.h"
#include "mochi_ecs.h"
#include "mochi_rom_jacobian.h"
#include "mochi_simulation.h"

#include <mochi_core/utils/variant_jacobian.h>

#include <type_traits>

namespace mochi {

// Forwards
struct CIslandMemberInfo;
struct CRomProjectionStrategy;

// Composition linking a nested soft actor to its articulated parent.
struct CSkinnedComposition : public NoCopy {
  entt::entity articulated;
  ActorHandle articulatedHandle;
};

// Component to store node positions including soft displacements, prior to skinning.
struct CNodePositions : public NoCopy {
  explicit CNodePositions(int size) : value(size) {}
  ColumnVector<real> value;
};

// Jacobian of the skinned mesh wrt soft-actor displacements, at rest. It is a block diagonal
// matrix, so the data structure stores only the diagonal blocks.
struct CSoftSkinnedJacobianRest {
  VariantJacobian::MatrixBlockDiag value;
};

// Jacobian of the skinned mesh wrt soft-actor DoFs. It admits two implementations:
// - For FOM soft actors, it is a block diagonal matrix, and only the diagonal blocks are stored.
// - For ROM soft actors, it is a dense matrix.
struct CSoftSkinnedJacobianDynamic : public VariantJacobian {
  CSoftSkinnedJacobianDynamic(bool isBlockDiag, int numRows, int numCols)
      : VariantJacobian(isBlockDiag, numRows, numCols) {}
};

// Jacobians of the skinned mesh wrt bone parameters and wrt articulated DoFs.
struct CSoftSkinnedJacobianArticulated : public NoCopy {
  SparseMatrix<real> dBones;
  RowMatrix<real> dJoints;
};

// Jacobian of an articulated actor with soft skinned actors, in dense form, for their
// dJoints = dBones * value. skinned::UpdateJacobiansPipeline expands it once for all of them.
struct CDenseArticulatedJacobian : public NoCopy {
  RowMatrix<real> value;
};

// Reusable memory used during skinned::EntityAssembleBody
struct CSoftSkinnedAssemblyScratchMemory {
  RowMatrix<real> A_Jsoft; // Only needed for ROMs. Will be resized on the fly.
  RowMatrix<real> A_Jart; // Will be resized on the fly.
};

// Component to store soft attachment links for soft skinned actors (for export purposes)
// Note: This only stores external attachments (softAttachLinks parameter).
// Internal skinning data (whole-hand skins) is stored in the shape file, not here.
struct CSoftAttachmentLinks {
  DynamicArray<int> softAttachLinks; // softAttachLinks[softIndex] = linkIndex
};

/*
Nested soft actor systems
*/
namespace skinned {

void EntityAssembleBody(
    AssemblyParams const& params, // external parameter
    TaskSemaphore softAssemblySem,
    TaskSemaphore artAssemblySem,
    entt::registry& reg,
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    ecs::OptionalTag<TagUseInertia> hasInertiaTag,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    ecs::OptionalTag<TagRomActor> isRom,
    CSkinnedEnergy const& skinnedEnergy,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CFemVolumeDiscretizationP1Q1 const& femVolDisc,
    CSoftMaterialParams const& materialParams,
    CMassMatrix const* massMatrix,
    CDofOffset const& dofOffset,
    CActiveVolumeElements const* activeVolElems,
    CSkinnedComposition const& composition,
    CTimeIntegratorState const& intState,
    CSoftSkinnedJacobianArticulated const& jacobianArticulated,
    CSoftSkinnedJacobianDynamic const& jacobianSoft,
    CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const& currDispl,
    CDisplacementSlice<real, TimeStep::StageStart, DisplacementLayer::Skinned> const&
        stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart, DisplacementLayer::Skinned> const* stageStartVel,
    CRomProjectionStrategy const* romProjectionStrategy,
    CSoftSkinnedAssemblyScratchMemory& scratchMemory,
    CActorSnle& outSoftSnle,
    CSoftSkinnedUnposedSnle& outSoftSkinnedSnle,
    CArticulatedSkinnedBodyLinkResidual& outLinkResidual,
    CSkinnedInteractionSnle& outInteractionSnle);

void InitSkinnedActor(
    entt::registry& reg,
    entt::entity e,
    SoftSkinnedActorParams const& params,
    bool useContact,
    ActorHandle articulatedHandle,
    Error& error);

void EntityIncrementStep(
    ecs::RequiredTag<TagNestedSoftActor>,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& currVel,
    CVelocitySlice<real, TimeStep::Previous, DisplacementLayer::Skinned>& prevVel);

/*
 * Reconstructs the begin-of-step @ref TimeStep::Previous state from the integration history if
 * history is present and the integration method allows, otherwise it estimates the state from
 * constant velocity.
 */
void EntityReconstructPreviousStep(
    ecs::RequiredTag<TagNestedSoftActor>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned> const& currVel,
    CIntegrationVelocitySlices<DisplacementLayer::Skinned> const& intVels,
    CVelocitySlice<real, TimeStep::Previous, DisplacementLayer::Skinned>& outPrevVel);

/*
 * Executed before the first time integration stage of the time step.
 */
void EntityPreFirstStage(
    ecs::RequiredTag<TagNestedSoftActor>,
    ecs::OptionalCtxGlobal<TagAssemblyPreserveHistory const> preserveHistory,
    CTimeIntegratorState const& intState,
    CVelocitySlice<real, TimeStep::Previous, DisplacementLayer::Skinned> const& prevVel,
    CIntegrationVelocitySlices<DisplacementLayer::Skinned>& intVels);

/*
 * Pipeline executed before each time integration stage.
 */
void PreStagePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * System executed after each time integration stage. It must be called after
 * articulated::compound::PostStagePipeline.
 */
void EntityPostStage(
    ecs::RequiredTag<TagNestedSoftActor>,
    CConvergenceStatus const& convergence,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::StageStart, DisplacementLayer::Skinned> const&
        stageStartDispl,
    CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned>& currDispl,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& currVel,
    CIntegrationVelocitySlices<DisplacementLayer::Skinned>& intVels);

/*
 * Pipeline executed after the last time integration stage of the time step. It must be called after
 * articulated::compound::PostLastStagePipeline.
 */
void PostLastStagePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Pipeline to resolve current skinning displacements for all nodes, including inactive nodes when
 * subsampling is enabled.
 */
void ResolveAllNodeSkinningDisplacementsPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities);

/*
 * Synchronize a nested soft actor's current skinned displacement and velocity after an external
 * displacement and velocity change. If it belongs to a blended actor, republishes the parent's
 * final displacement and invalidates the parent's step history.
 */
void SynchronizeAfterExternalChange(entt::registry& reg, entt::entity e);

// Compute world-space skinning velocity for nested soft actors. The template argument indicates if
// the system must run on state or non-state velocities.
template <bool kIsState>
void UpdateSkinningVelocity(
    std::conditional_t<
        kIsState,
        ecs::Included<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>,
        ecs::Excluded<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>>,
    ecs::CtxGlobal<CSceneTime const> time,
    ecs::PartialRegistry<
        CArticulatedLinkTransforms<TimeStep::Current> const,
        CArticulatedLinkVels const> reg,
    CSkinnedComposition const& composition,
    CVelocitySlice<real, TimeStep::Current> const& softVelocity,
    CArticulatedSkinningData const& skinningData,
    CNodePositions const& positions,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& outVelocity);

/*
 * Pipeline to update quantities that are a function of the state (aka derived state) of the
 * nested soft actor and make them consistent with the state. Must be called after
 * articulated::compound::UpdateDerivedStatePipeline()
 */
void UpdateDerivedStatePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Pipeline to update Jacobians for assembly. Must be called after
 * articulated::compound::UpdateJacobiansStatePipeline<TimeStep::Current>(). compoundActors must
 * include the articulated actors of the entities.
 */
void UpdateJacobiansPipeline(
    entt::registry& reg,
    Span<entt::entity const> compoundActors,
    Span<entt::entity const> entities);

// Call once on startup
void InitializeOnce(entt::registry& reg);

} // namespace skinned

} // namespace mochi
