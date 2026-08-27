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
#include "mochi_ecs.h"
#include "mochi_rom_jacobian.h"
#include "mochi_simulation.h"

#include <mochi_core/utils/variant_jacobian.h>

namespace mochi {

// Forwards
struct CIslandMemberInfo;
struct CRomProjectionStrategy;

// Actor composition of a soft skinned actor
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
Systems for a soft skinned body
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
    CFemVolumeDiscretizationP1Q1 const& femLowVolDisc,
    CFemVolumeDiscretizationP1Q4 const& femHighVolDisc,
    CSoftMaterialParams const& materialParams,
    CMassMatrix const& massMatrix,
    CDofOffset const& dofOffset,
    CActiveVolumeElements const* activeVolElems,
    CSkinnedComposition const& composition,
    CTimeIntegratorState const& intState,
    CArticulatedSkinningData const& skinningData,
    CSoftSkinnedJacobianDynamic const& jacobianSoft,
    CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const& currDispl,
    CDisplacementSlice<real, TimeStep::StageStart, DisplacementLayer::Skinned> const&
        stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart, DisplacementLayer::Skinned> const* stageStartVel,
    CRomProjectionStrategy const* romProjectionStrategy,
    CSoftSkinnedAssemblyScratchMemory& scratchMemory,
    CActorSnle& outSoftSnle,
    CSoftSkinnedUnposedSnle& outSoftSkinnedSnle,
    CSkinnedInteractionSnle& outInteractionSnle);

// Method to initialize a soft skinned actor.
void InitSkinnedActor(
    entt::registry& reg,
    entt::entity e,
    SoftSkinnedActorParams const& params,
    bool useContact,
    ActorHandle articulatedHandle,
    Error& error);

void EntityIncrementStep(
    ecs::RequiredTag<TagSoftSkinnedActor>,
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>& currVel,
    CVelocitySlice<real, TimeStep::Previous, DisplacementLayer::Skinned>& prevVel);

/*
 * Executed before the first time integration stage of the time step.
 */
void EntityPreFirstStage(
    ecs::RequiredTag<TagSoftSkinnedActor>,
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
    ecs::RequiredTag<TagSoftSkinnedActor>,
    CConvergenceStatus const& convergence,
    CTimeIntegratorState const& intState,
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
 * Pipeline to update quantities that are a function of the state (aka derived state) of the
 * soft skinned actor and make them consistent with the state. Must be called after
 * articulated::compound::UpdateDerivedStatePipeline()
 */
void UpdateDerivedStatePipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * Pipeline to update Jacobians for contact and assembly. Must be called after
 * articulated::compound::UpdateJacobiansPipeline()
 */
void UpdateJacobiansPipeline(entt::registry& reg, Span<entt::entity const> entities);

/*
 * System to compute contact Jacobians as colliding actor. It is called after the collision
 * detection pipeline, within each nonlinear solver iteration.
 */
void SetupCollidingJacobians(
    ecs::Included<TagSoftSkinnedActor>,
    ecs::PartialRegistry<CDofOffset const, CArticulatedLinkTransforms<TimeStep::Current> const> reg,
    CSkinnedComposition const& composition,
    CDofOffset const& dofOffset,
    CContactPartitions const& contactPartitions,
    CFemBoundaryDiscretization const& discretization,
    CArticulatedSkinningData const& skinningInfo,
    rom::CRomJacobian const* romJacobian,
    CCollJacs<CollRole::Colliding>& outJacobians);

// Call once on startup
void InitializeOnce(entt::registry& reg);

} // namespace skinned

} // namespace mochi
