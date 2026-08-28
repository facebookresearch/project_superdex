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

#include "mochi_discretization_components.h"
#include "mochi_ecs.h"
#include "mochi_rom_jacobian.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"
#include "mochi_soft.h"
#include "mochi_soft_rom_components.h"

#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/rom/rom.h>
#include <mochi_core/rom/rom_pivot.h>

#include <optional>

namespace mochi::rom {

// Update CSdfMapping based on the ROM state. Must be called for all ROM actors on each assembly
// (with kTimeStep = TimeStep::Current), and at the beginning of each integration stage (with
// kTimeStep = TimeStep::StageStart) when ExperimentalEvalParams.explicitNormals = true.
template <TimeStep kTimeStep>
void UpdateMap(
    ecs::RequiredTag<TagRomActor>,
    CRomModeAmplitudes const& amplitudes,
    CSdfMapping<kTimeStep>& mapSdf) {
  mapSdf->UpdateMap(amplitudes.value.GetConstSpan());
}

/*
 * System to copy a span of reals to the position state of the soft ROM actor. The span of reals is
 * usually the components of the non-linear problem solution vector corresponding to the actor,
 * thereby the name.
 */
inline void EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::OptionalTag<TagRomActorFixRigidTransformInSolve> isRigidTransformFixedInSolve,
    CRomModeAmplitudes& currAmplitudes,
    CRigidState<TimeStep::Current>& rigidTransform) {
  MOCHI_PROFILE_SCOPE();

  FromRawFunc baseSetter = [&](ColumnVectorView<real const> pose) { currAmplitudes.value = pose; };

  FromRawFunc modifiedSetter = baseSetter;
  if (!isRigidTransformFixedInSolve) {
    modifiedSetter = [&](ColumnVectorView<real const> pose) {
      pivoted::FromRawPose(pose, baseSetter, rigidTransform.value);
    };
  }

  modifiedSetter(solution);
}

/*
 * System to set the state of a ROM actor from a reference state plus an increment expressed in
 * local tangent space.
 */
template <TimeStep kStep>
void EntitySetIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::OptionalTag<TagRomActorFixRigidTransformInSolve> isRigidTransformFixedInSolve,
    CRomModeAmplitudes& currAmplitudes,
    CRigidState<kStep>& rigidTransform) {
  MOCHI_PROFILE_SCOPE();

  FromIncrementFunc baseSetter = [&](ColumnVectorView<real const> ref,
                                     ColumnVectorView<real const> inc) {
    currAmplitudes.value = ref + inc;
  };

  FromIncrementFunc modifiedSetter = baseSetter;
  if (!isRigidTransformFixedInSolve) {
    modifiedSetter = [&](ColumnVectorView<real const> ref, ColumnVectorView<real const> inc) {
      pivoted::FromIncrement(ref, inc, baseSetter, rigidTransform.value);
    };
  }

  modifiedSetter(reference, increment);
}

/*
 * System to copy the position state of the soft ROM actor to a span of reals. The span of reals is
 * usually the components of the non-linear problem solution vector corresponding to the actor,
 * thereby the name.
 */
inline void EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::OptionalTag<TagRomActorFixRigidTransformInSolve> isRigidTransformFixedInSolve,
    CRomModeAmplitudes const& currAmplitudes,
    CRigidState<TimeStep::Current> const* rigidTransform) {
  MOCHI_PROFILE_SCOPE();

  ToRawFunc baseGetter = [&](ColumnVectorView<real> pose) { pose = currAmplitudes.value; };

  ToRawFunc modifiedGetter = baseGetter;
  if (!isRigidTransformFixedInSolve) {
    MOCHI_ASSERT(rigidTransform, "Missing rigid transform.");
    modifiedGetter = [&](ColumnVectorView<real> pose) {
      pivoted::ToRawPose(baseGetter, rigidTransform->value, pose);
    };
  }

  modifiedGetter(outSolution);
}

// Assemble a soft ROM by reading the full DOF CActorSnle, projecting to the reduced DOFs.
// Must be called AFTER the regular soft actor assembly.
MOCHI_API void AssembleFullToReduced(
    AssemblyParams const& params,
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::OptionalTag<TagNestedSoftActor> isNestedSoft,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    ecs::OptionalTag<TagUseInertia> hasInertiaTag,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    ecs::OptionalTag<TagUseContact> hasContactTag,
    CSkinnedEnergy const& skinnedEnergy,
    CRomProjectionStrategy const& projectionStrategy,
    CRomJacobian& jacobian, // non-const because of caching
    CActorSnle& actorSnle);

// Assemble and project the volume terms.
MOCHI_API void AssembleAndProjectBody(
    AssemblyParams const& params, // External parameter
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    ecs::OptionalTag<TagUseInertia> hasInertiaTag,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    ecs::OptionalTag<TagNestedSoftActor> isNestedSoft,
    CSkinnedEnergy const& skinnedEnergy,
    CRomProjectionStrategy const& projectionStrategy,
    CLocal2GlobalMap const& l2g,
    CFemVolumeDiscretizationP1Q1 const& femLowVolDisc,
    CFemVolumeDiscretizationP1Q4 const& femHighVolDisc,
    CRootTransform const& rootTransform,
    CSoftMaterialParams const& materialParams,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CDisplacementSlice<real, TimeStep::StageStart> const& stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart> const& stageStartVel,
    CPerElementMassMatrix<CFemVolumeDiscretizationP1Q4> const& perElemMass,
    CRomJacobian const& jacobian,
    CActorSnle& outSnle,
    CActiveVolumeElements const* activeVolElems = nullptr);

// Assemble and project async contact.
void AssembleAndProjectAsyncContact(
    AssemblyParams const& params, // External parameter
    entt::entity e,
    ecs::Included<TagSoftActor, TagRomActor, TagUseContact>,
    ecs::Excluded<TagNestedSoftActor>,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    ContactAssemblyReg reg,
    CRomProjectionStrategy const& projectionStrategy,
    CFemBoundaryDiscretization const& femBoundaryDisc,
    CBoundaryLocal2GlobalMap const& bdL2g,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& collisions,
    CRootTransform const& rootTransform,
    CTimeIntegratorState const& intState,
    CRomJacobian const& jacobian,
    CDeformablePointAsyncCollisionsResponse& outResponse,
    CActorSnle& outSnle,
    CActiveBoundaryFaces const* activeBoundaryFaces = nullptr);

/***************************************************************************
  Related to rigid transform
*/

using RigidTransformRegistryExp = ecs::PartialRegistry<
    TagRomActor const,
    CRootTransform const,
    CRigidState<TimeStep::Current> const,
    CRigidState<TimeStep::StageStart> const,
    CRigidState<TimeStep::Previous> const,
    CMeshPivot const>;

// Get the rigid transform if a transform layer is present.
template <TimeStep kTimeStep>
std::optional<TransformRT> GetRigidTransform(RigidTransformRegistryExp reg, entt::entity entity);

// Modify the contact Jacobians stored in outContact.mapInfo to account for the rigid transform
// layer.
void AddRigidContactJacobians(
    RigidTransformRegistryExp reg,
    entt::entity entity,
    ContactDetectionResult& outContact);

/***************************************************************************
  Recording Systems
*/

void RecordRomAmplitudes(
    ecs::RequiredTag<TagRomActor>,
    CRomModeAmplitudes const& amplitudes,
    CRecordingData& outData);

void RecordRomRigidTransform(
    ecs::RequiredTag<TagRomActor>,
    CRigidState<TimeStep::Current> const& rigid,
    CRecordingData& outData);

void RecordingPipeline(entt::registry& reg, Span<entt::entity const> entities);

/**************************************************************************
  ROM Pipelines
*/

void RomFomSwitchingPipeline(
    entt::registry& reg,
    entt::entity e,
    CRomFomSwitchingParams const& params);

void PreStepPipeline(entt::registry& reg);

void EntityPreFirstStage(
    ecs::RequiredTag<TagRomActor>,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Previous> const& prevDispl,
    CVelocitySlice<real, TimeStep::Previous> const& prevFomVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intFomVels);

void PreStagePipeline(entt::registry& reg, Span<entt::entity const> entities);

// Updates velocity in ROM space by taking finite differences between the current and stage-start
// ROM state as raw vectors. Called every time the solution is set.
MOCHI_API void UpdateCurrentRomVelocity(
    ecs::Included<TagSoftActor, TagRomActor>,
    ecs::OptionalTag<TagRomActorFixRigidTransformInSolve> isRigidTransformFixedInSolve,
    CRomCommonProperties const& props,
    CTimeIntegratorState const& intState,
    CRomModeAmplitudes const& currAmplitudes,
    CRigidState<TimeStep::Current> const* currRigidTransform,
    CRigidState<TimeStep::StageStart> const* stageStartRigidTransform,
    CRomVelocity<real, TimeStep::Current>& outVelocity);

void PostNewSolutionPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities,
    ColumnVectorView<real const> solution);

void PostNewIncrementPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities,
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment);

void PostStagePipeline(entt::registry& reg, Span<entt::entity const> entities);

void EntityPostLastStage(
    ecs::RequiredTag<TagRomActor>,
    CTimeIntegratorState const& intState,
    CIntegrationDisplacementSlices& intFomDispls,
    CDisplacementSlice<real, TimeStep::Current>& currFomDispl,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intFomVels,
    CVelocitySlice<real, TimeStep::Current>& currFomVel);

/**************************************************************************
  Swapping active elements
*/

void SwapActiveElements(entt::registry& reg, entt::entity e);

} // namespace mochi::rom
