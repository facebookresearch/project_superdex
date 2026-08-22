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

#include "mochi_soft.h"

#include "mochi_actor_convergence.h"
#include "mochi_contact.h"
#include "mochi_deformable.h"
#include "mochi_discretization_components.h"
#include "mochi_ecs_utils.h"
#include "mochi_integration.h"
#include "mochi_simulation.h"
#include "mochi_soft_rom_components.h"

#include <mochi_core/contact/dmap.h>
#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/element_operations/fem_gravity.h>
#include <mochi_core/element_operations/fem_inertia.h>
#include <mochi_core/element_operations/fem_stress.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/tetrahedral_map.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/materials/batched_materials.h>
#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/solvers/snle_problem.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/rigid_body_utils.h>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <entt/entity/view.hpp>

using namespace mochi;
using namespace mochi::dmap;

/**
 * @brief Creates a batched op that accumulates per-element stress energy.
 *
 * @warning @p batchedConstitutive must outlive the returned op.
 */
template <class ElementT, class ConstitutiveFn, int kBatchSize = kDefaultFemBatchSize>
[[nodiscard]] static auto MakeBatchedStressEnergyOp(
    Span<ElementT const> elements,
    ConstitutiveFn const& batchedConstitutive,
    Span<real const> activeVolWeights) {
  return [elements, &batchedConstitutive, activeVolWeights](
             NdArray<int, kBatchSize> const& elementIndices,
             Span<int const> /*indicesFlat*/,
             fem::BatchElementVector<kBatchSize, ElementT> const& disp,
             BatchDouble<kBatchSize>* outEnergy,
             fem::BatchElementVector<kBatchSize, ElementT>* /*outRes*/,
             fem::BatchElementMatrix<kBatchSize, ElementT>* /*outDRes*/,
             bool projectPsd) -> bool {
    MOCHI_ASSERT_VERBOSE(outEnergy != nullptr, "Output energy must not be nullptr.");
    return fem::StressWork<kBatchSize>(
        elementIndices,
        elements,
        disp,
        outEnergy,
        /*outRes*/ nullptr,
        /*outDRes*/ nullptr,
        projectPsd,
        batchedConstitutive,
        activeVolWeights);
  };
}

/*************************************************************************************************/

void mochi::soft::EntityIncrementStep(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CDisplacementSlice<real, TimeStep::Previous>& prevDispl,
    CVelocitySlice<real, TimeStep::Previous>& prevVel) {
  prevDispl.CopyFrom(currDispl); // Copy previous displacement
  prevVel.CopyFrom(currVel); // Copy previous velocity
  currVel.value.SetZero(); // Reset current velocity to zero
}

void mochi::soft::EntityPreFirstStage(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Previous> const& prevDispl,
    CVelocitySlice<real, TimeStep::Previous> const& prevVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels) {
  // Displacement and velocity are differential variables. Compute their values at the beginning
  // of the step using integration utilities.
  integration::ApplyTimeIntegrationStepStart(intState, intDispls, prevDispl, intDispls.stepStart);
  integration::ApplyTimeIntegrationStepStart(intState, intVels, prevVel, intVels.stepStart);
}

void mochi::soft::EntityPreStage(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CTimeIntegratorState const& intState,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CDisplacementSlice<real, TimeStep::StageStart>& stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart>& stageStartVel) {
  MOCHI_PROFILE_SCOPE();
  // Displacement and velocity are differential variables. Compute their values at the beginning
  // of the stage using integration utilities.
  integration::ApplyTimeIntegration<TimeTarget::StageStart>(intState, intDispls, stageStartDispl);
  integration::ApplyTimeIntegration<TimeTarget::StageStart>(intState, intVels, stageStartVel);
}

void mochi::soft::EntityPostStage(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CConvergenceStatus const& convergence,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::StageStart> const& stageStartDispl,
    CDisplacementSlice<real, TimeStep::Current>& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels) {
  MOCHI_PROFILE_SCOPE();
  // Velocity is a differential variable but it's not explicitly solved for to reduce the number
  // of DoFs in the solver. Velocity at the end of the stage is recovered via finite differences
  // of the displacements at the beginning and at the end of the stage.
  currVel.value = (currDispl.value - stageStartDispl.value) * (1_r / intState.dtStage);

  // If the solver diverged, reset the displacements and velocities to zero.
  if (convergence.stageStatus == ConvergenceStatus::Diverged) {
    currDispl.value.SetZero();
    currVel.value.SetZero();
  }

  // Displacement and velocity are differential variables. Push them to the vectors containing the
  // displacements and velocities at the end of each time integration stage.
  intDispls.stages[intState.currentStage].value = currDispl.value;
  intVels.stages[intState.currentStage].value = currVel.value;
}

void mochi::soft::EntityPostLastStage(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Current>& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels) {
  // Displacement and velocity are differential variables. Compute their values at the end of
  // the time step using integration utilities.
  integration::ApplyTimeIntegration<TimeTarget::StepEnd>(intState, intDispls, currDispl);
  integration::ApplyTimeIntegration<TimeTarget::StepEnd>(intState, intVels, currVel);
}

void mochi::soft::EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CDisplacementSlice<real, TimeStep::Current>& currSol) {
  MOCHI_PROFILE_SCOPE();
  currSol.value = solution.MiddleRows(dofOffset.poseOffset, actorDofInfo.poseSize);
}

void mochi::soft::EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CDisplacementSlice<real, TimeStep::Current>& currSol) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(actorDofInfo.dofsSize == actorDofInfo.poseSize, "Unexpected DoF info");
  currSol.value = reference.MiddleRows(dofOffset.poseOffset, actorDofInfo.poseSize) +
      increment.MiddleRows(dofOffset.dofsOffset, actorDofInfo.dofsSize);
}

void mochi::soft::AssembleBodyImpl(
    AssemblyParams const& params, // external parameter
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    bool hasGravity,
    bool hasInertia,
    bool hasStress,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CFemVolumeDiscretizationP1Q1 const& femLowVolDisc,
    CFemVolumeDiscretizationP1Q4 const& femHighVolDisc,
    TransformRT const& worldFromLocal,
    CSoftMaterialParams const& materialParams,
    CTimeIntegratorState const& intState,
    ColumnVectorView<real const> currDispl,
    ColumnVectorView<real const> stageStartDispl,
    ColumnVectorView<real const> stageStartVel,
    CMassMatrix const& massMatrix,
    ActorSnle& outSnle,
    CActiveVolumeElements const* activeVolElems) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(
      !activeVolElems || !activeVolElems->empty(), "Active volume elements must not be empty.");
  MOCHI_ASSERT_VERBOSE(
      hasGravity || hasInertia || hasStress,
      "AssembleBodyImpl expects at least one body term. No-term cases must be filtered by the caller.");

  // Damping scales. Mass damping requires inertia; stiffness damping requires stress. Mass damping
  // is folded into the mass-matrix dresidual scale (alongside inertia), so it needs no per-element
  // dresidual assembly.
  auto const dampingScales =
      soft::ComputeSoftDampingScales(materialParams, intState.dtStage, hasInertia, hasStress);
  real const massScale = dampingScales.massScale;
  real const massDampingScale = dampingScales.massDampingScale;
  real const stiffnessDampingFactor = dampingScales.stiffnessDampingFactor;
  bool const includeStiffnessDampingGeometricTerm =
      dampingScales.includeStiffnessDampingGeometricTerm;

  // Clear SNLE data
  if (params.assemObj) {
    outSnle.objective = 0.0;
  }
  if (params.assemRes) {
    outSnle.fullResidual.SetZero();
  }
  if (params.assemDRes) {
    MOCHI_PROFILE_SCOPE_N("InitSoftDResidual");
    // Initialization of the soft dresidual is memory-bounded. Parallelization relies on the
    // assumption that multi-threaded memory bandwidth is larger than single-threaded memory
    // bandwidth, which is true in most architectures.
    // TODO: Limit max threads to the number of threads that saturate memory bandwidth.
    auto const numValues = GetNumValues(outSnle.fullDResidual);
    constexpr int kMinValuesPerTask = 150000;
    ParallelForRange(
        "InitSoftDResidual", 0, numValues, kMinValuesPerTask, INT_MAX, [&](int rBegin, int rEnd) {
          MOCHI_ASSERT_VERBOSE(rBegin >= 0 && rBegin <= rEnd && rEnd <= numValues);
          ColumnVectorView<real> dresValues = AsView(GetValues(outSnle.fullDResidual));
          if (massScale != 0_r) {
            // Initialize the dresidual matrix by scaling the mass matrix (inertia + mass damping),
            // instead of clearing it to zero. This eliminates the need to assemble the dresidual of
            // the inertia and mass damping terms for each element.
            MOCHI_ASSERT(dresValues.size() == massMatrix.values.size(), "Size mismatch.");
            dresValues.MiddleRows(rBegin, rEnd - rBegin) =
                massScale * AsView(MakeSpan(massMatrix.values)).MiddleRows(rBegin, rEnd - rBegin);
          } else {
            dresValues.MiddleRows(rBegin, rEnd - rBegin).SetZero();
          }
        });
  }

  AssemblyParams bodyParams = params;
  bodyParams.assemDRes &= hasStress; // Only stress contributes to DRes
  if (!bodyParams.assemObj && !bodyParams.assemRes && !bodyParams.assemDRes) {
    return;
  }

  AssemblyActiveSubset activeSubset = activeVolElems
      ? AssemblyActiveSubset{activeVolElems->ViewIndices(), activeVolElems->ViewIsActive()}
      : AssemblyActiveSubset{};

  Span<real const> activeVolWeights =
      activeVolElems ? activeVolElems->ViewWeights() : Span<real const>{};
  auto const gravity = ToReal3(worldFromLocal.TransformDirectionInverse(sceneGravity->accel));

  details::VisitPerElementMaterialParams(
      materialParams, [&]<typename ParamsT>(auto const& perElem) {
        auto batchedConstitutive = materials::MakeBatchedConstitutiveResponse<ParamsT>(perElem);
        auto bodyOp = MakeBatchedBodyOp(
            MakeConstSpan(femLowVolDisc.femElements),
            MakeConstSpan(femHighVolDisc.femElements),
            batchedConstitutive,
            materialParams.referenceMaterialStiffness,
            hasStress,
            hasGravity,
            hasInertia,
            gravity,
            materialParams.density,
            stageStartDispl.GetConstSpan(),
            stageStartVel.GetConstSpan(),
            intState.dtStage,
            massDampingScale,
            stiffnessDampingFactor,
            includeStiffnessDampingGeometricTerm,
            activeVolWeights);

        AssembleObjResDRes<SoftStencilElement>(
            l2g,
            nbs,
            bodyOp,
            currDispl,
            AssemblyResults<real>{
                .outObj = &outSnle.objective,
                .outRes = AsView(outSnle.fullResidual),
                .outDRes = AsView(outSnle.fullDResidual),
                .params = bodyParams},
            activeSubset);
      });
}

void mochi::soft::AssembleBody(
    AssemblyParams const& params, // external parameter
    ecs::Included<TagSoftActor>,
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    ecs::OptionalTag<TagUseInertia> hasInertiaTag,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    ecs::OptionalTag<TagRomActor> isRom,
    [[maybe_unused]] ecs::OptionalTag<TagSoftSkinnedActor> isSkinned,
    CSkinnedEnergy const& skinnedEnergy,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CFemVolumeDiscretizationP1Q1 const& femLowVolDisc,
    CFemVolumeDiscretizationP1Q4 const& femHighVolDisc,
    CRootTransform const& rootTransform,
    CSoftMaterialParams const& materialParams,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CDisplacementSlice<real, TimeStep::StageStart> const& stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart> const& stageStartVel,
    CMassMatrix const& massMatrix,
    CRomProjectionStrategy const* romProjectionStrategy,
    CActorSnle& outActorSnle,
    CActiveVolumeElements const* activeVolElems) {
  MOCHI_ASSERT_VERBOSE(!isRom || romProjectionStrategy, "Missing ROM projection strategy.");
  bool const hasGravity = hasGravityTag && !skinnedEnergy.gravity;
  bool const hasInertia = hasInertiaTag && !skinnedEnergy.inertia;
  bool const hasStress = hasStressTag && !skinnedEnergy.stress;
  static_assert(
      static_cast<int>(experimental::RomProjectionStrategy::Count) == 2,
      "Please update logic below if RomProjectionStrategy enum changes");

  if (!hasGravity && !hasInertia && !hasStress) {
    // Only soft-skinned actors may reach here. Note their Dresidual in this case must NOT be
    // modified in AssembleBodyImpl to prevent race conditions with skinned::EntityAssembleBody.
    MOCHI_ASSERT_VERBOSE(
        isSkinned, "Only soft-skinned actors can assemble with no unposed body terms.");
    return;
  } else if (
      isRom &&
      romProjectionStrategy->value == experimental::RomProjectionStrategy::ElementLevelProjection) {
    // ROMs with element-level projection do not assemble the FOM.
    return;
  }

  AssembleBodyImpl(
      params,
      sceneGravity,
      hasGravity,
      hasInertia,
      hasStress,
      l2g,
      nbs,
      femLowVolDisc,
      femHighVolDisc,
      rootTransform.worldFromLocal,
      materialParams,
      intState,
      currDispl.value,
      stageStartDispl.value,
      stageStartVel.value,
      massMatrix,
      outActorSnle,
      activeVolElems);
}

void mochi::soft::AssembleAsyncContact(
    AssemblyParams const& params,
    entt::entity e,
    ecs::Included<TagSoftActor, TagUseContact>,
    ecs::OptionalTag<TagRomActor> isRom,
    ecs::Excluded<TagSoftSkinnedActor>,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CTimeIntegratorState const& intState,
    ContactAssemblyReg reg,
    CFemBoundaryDiscretization const& femBoundaryDisc,
    CBoundaryLocal2GlobalMap const& bdL2g,
    CBoundaryNodalBasedStructure const& bdNbs,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& collisions,
    CRootTransform const& rootTransform,
    CRomProjectionStrategy const* romProjectionStrategy,
    CDeformablePointAsyncCollisionsResponse& outResponse,
    CActorSnle& outActorSnle,
    CActiveBoundaryFaces const* activeBoundaryFaces) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(params.assemObj || params.assemRes || params.assemDRes, "Must assemble something");
  MOCHI_ASSERT_VERBOSE(!isRom || romProjectionStrategy, "Missing ROM projection strategy.");

  static_assert(
      static_cast<int>(experimental::RomProjectionStrategy::Count) == 2,
      "Please update logic below if RomProjectionStrategy enum changes");
  if (isRom &&
      romProjectionStrategy->value == experimental::RomProjectionStrategy::ElementLevelProjection) {
    // Ensure ROMs are not assembled twice.
    return;
  } else {
    MOCHI_ASSERT_VERBOSE(
        !isRom ||
            romProjectionStrategy->value ==
                experimental::RomProjectionStrategy::ActorLevelProjection,
        "Unexpected ROM projection strategy.");
  }

  if (collisions.empty() || (activeBoundaryFaces && activeBoundaryFaces->empty())) {
    // No contacts or no active boundary faces.
    outResponse.Clear();
    return;
  }

  deformable::ComputeAsyncContactResponse<CFemBoundaryDiscretization, kSpaceDim3>(
      reg,
      e,
      simParams->experimentalEval,
      queryActiveContacts,
      femBoundaryDisc,
      samples,
      colliderInfo,
      collisions,
      intState,
      rootTransform,
      params,
      outResponse,
      activeBoundaryFaces ? activeBoundaryFaces->ViewIsActive() : Span<bool const>{});

  if (outResponse.Empty()) {
    // No contacts.
    return;
  }

  // TODO[T162644296] - This function assembles values for all 4 nodes of each boundary
  // tetrahedron. The internal node will have zero contribution. Instead, it should only compute
  // the 3 boundary terms, and it should only write those to the residual/dresidual.
  AssemblyResults<real> results{
      .outObj = &outActorSnle.objective,
      .outRes = AsView(outActorSnle.fullResidual),
      .outDRes = AsView(outActorSnle.fullDResidual),
      .params = params};

  femBoundaryDisc.Visit([&](auto const& disc) {
    using DiscT = std::decay_t<decltype(disc)>;
    using ElementT = typename DiscT::ElementT;

    AssemblyActiveSubset const activeSubset = outResponse.ViewActiveContactElementSubset();

    Span<real const> bdFaceWeights =
        activeBoundaryFaces ? activeBoundaryFaces->ViewWeights() : Span<real const>{};
    auto boundaryOp = deformable::MakeBatchedBoundaryOp(
        MakeConstSpan(disc.femElements), outResponse, bdFaceWeights);

    AssembleObjResDRes<ElementT>(bdL2g, bdNbs, boundaryOp, results, activeSubset);
  });
}

void soft::EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currSol) {
  MOCHI_PROFILE_SCOPE();
  outSolution = currSol.value; // copy values
}

void soft::EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDisplacementSlice<real, TimeStep::Current>& currSol) {
  MOCHI_PROFILE_SCOPE();
  currSol.value = solution; // copy values
}

static void GetSoftMaterialParamsHelper(
    materials::AnyMaterialParams const& params,
    SoftMaterialParams& outParams) {
  static_assert(
      static_cast<int>(SoftMaterialType::Count) == 6,
      "Please update the if statement below if SoftMaterialType enum changes");
  outParams.type = GetSoftMaterialType(params);
  if (auto const* p0 = std::get_if<NeoHookeanMaterialParams>(&params)) {
    outParams.neoHookean = *p0;
  } else if (auto const* p1 = std::get_if<StVenantKirchhoffMaterialParams>(&params)) {
    outParams.stVenantKirchhoff = *p1;
  } else if (auto const* p2 = std::get_if<LinearElasticMaterialParams>(&params)) {
    outParams.linearElastic = *p2;
  } else if (auto const* p3 = std::get_if<ActiveNeoHookeanMaterialParams>(&params)) {
    outParams.activeNeoHookean = *p3;
  } else if (auto const* p4 = std::get_if<ActiveShapeTargetingArapMaterialParams>(&params)) {
    outParams.activeShapeTargetingArap = *p4;
  } else if (auto const* p5 = std::get_if<ArapMaterialParams>(&params)) {
    outParams.arap = *p5;
  }
}

template <typename ParamsT>
[[nodiscard]] static constexpr bool HasExplicitPsdStrategy(ParamsT const& params) {
  if constexpr (requires { params.psdStrategy; }) {
    return params.psdStrategy != MaterialPsdStrategy::MaterialDefault;
  } else {
    return false;
  }
}

// Material field params own per-element properties, not global PSD strategy. MaterialDefault
// inherits the base actor material PSD strategy. An explicit material field PSD strategy must match
// an explicit base strategy.
template <typename ParamsT>
[[nodiscard]] static MaterialPsdStrategy ResolvePerElementFieldPsdStrategy(
    ParamsT const& actorParams,
    MaterialPsdStrategy fieldPsdStrategy,
    Error& error) {
  auto const basePsdStrategy = materials::utils::ResolvePsdStrategy(actorParams);
  if (fieldPsdStrategy == MaterialPsdStrategy::MaterialDefault) {
    return basePsdStrategy;
  }

  MOCHI_ERROR_IF(
      HasExplicitPsdStrategy(actorParams) && fieldPsdStrategy != basePsdStrategy,
      error,
      "Per-element PSD strategy must match SoftActorParams::material PSD strategy when both are explicit.");
  MOCHI_ERROR_RETURN(error, basePsdStrategy);

  return fieldPsdStrategy;
}

void mochi::soft::GetMaterialParams(
    CSoftMaterialParams const& material,
    SoftMaterialParams& outParams) {
  auto params = material.params;
  GetSoftMaterialParamsHelper(params, outParams);
  outParams.density = material.density;
  outParams.massDampingCoefficient = material.massDampingCoefficient;
  outParams.stiffnessDampingCoefficient = material.stiffnessDampingCoefficient;
  outParams.stiffnessDampingIncludeGeometricTerm = material.stiffnessDampingIncludeGeometricTerm;
}

void mochi::soft::GetMaterialParamsField(
    CSoftMaterialParams const& material,
    int elementIndex,
    SoftMaterialParams& outParams) {
  auto params = material.params;
  std::visit(
      [&](auto& p) {
        using ParamsT = std::decay_t<decltype(p)>;
        p = materials::GetElementParams<ParamsT>(
            soft::details::GetMatchingPerElementParams<ParamsT>(material), elementIndex);
      },
      params);
  GetSoftMaterialParamsHelper(params, outParams);
  outParams.density = material.density;
  outParams.massDampingCoefficient = material.massDampingCoefficient;
  outParams.stiffnessDampingCoefficient = material.stiffnessDampingCoefficient;
  outParams.stiffnessDampingIncludeGeometricTerm = material.stiffnessDampingIncludeGeometricTerm;
}

// Rebuild the reference material stiffness store (consumed by stiffness damping) over all entries.
// numEntries is 1 for homogeneous materials and the element count for heterogeneous materials.
// A no-op-equivalent (empty store) results when stiffness damping is disabled.
static void RebuildReferenceMaterialStiffness(CSoftMaterialParams& mat, int numEntries) {
  if (mat.stiffnessDampingCoefficient <= 0_r) {
    mat.referenceMaterialStiffness = {};
    return;
  }
  soft::details::VisitPerElementMaterialParams(mat, [&]<typename ParamsT>(auto const& perElem) {
    auto resp = materials::MakeBatchedConstitutiveResponse<ParamsT, 1>(perElem);
    mat.referenceMaterialStiffness = materials::BuildPerElementReferenceMaterialStiffness(
        resp, numEntries, materials::kIsotropicReferenceStiffness<ParamsT>);
  });
}

// O(1) update of a single element's reference material stiffness, mirroring the per-element λ/μ
// update. Promotes the store to heterogeneous size when needed, filling new entries with the prior
// homogeneous tensor. Assumes the per-element material params have already been updated.
static void UpdateReferenceMaterialStiffnessElement(
    CSoftMaterialParams& mat,
    int elementIndex,
    int numElements) {
  if (mat.stiffnessDampingCoefficient <= 0_r) {
    mat.referenceMaterialStiffness = {};
    return;
  }
  auto& store = mat.referenceMaterialStiffness;
  if (store.empty()) {
    RebuildReferenceMaterialStiffness(mat, numElements);
    return;
  }
  if (isize(store.data) < numElements) {
    auto const homogeneous = store.data[0]; // copy before resize (avoid self-reference)
    store.data.resize(numElements, homogeneous);
  }
  soft::details::VisitPerElementMaterialParams(mat, [&]<typename ParamsT>(auto const& perElem) {
    auto resp = materials::MakeBatchedConstitutiveResponse<ParamsT, 1>(perElem);
    NdArray<int, 1> idx MOCHI_NO_INIT;
    idx[0] = elementIndex;
    auto const c0v = materials::ComputeReferenceMaterialStiffnessVoigt<1>(idx, resp);
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) {
        store.data[elementIndex][a][b] = c0v[a][b][0];
      }
    }
  });
}

void mochi::soft::SetMaterialParams(
    SoftMaterialParams const& inParams,
    CSoftMaterialParams& outMaterial) {
  static_assert(
      static_cast<int>(SoftMaterialType::Count) == 6,
      "Please update the switch statement below if SoftMaterialType enum changes");
  switch (inParams.type) {
    case SoftMaterialType::NeoHookean: {
      outMaterial.params = inParams.neoHookean;
      break;
    }
    case SoftMaterialType::StVenantKirchhoff: {
      outMaterial.params = inParams.stVenantKirchhoff;
      break;
    }
    case SoftMaterialType::LinearElastic: {
      outMaterial.params = inParams.linearElastic;
      break;
    }
    case SoftMaterialType::ActiveNeoHookean: {
      outMaterial.params = inParams.activeNeoHookean;
      break;
    }
    case SoftMaterialType::ActiveShapeTargetingArap: {
      outMaterial.params = inParams.activeShapeTargetingArap;
      break;
    }
    case SoftMaterialType::Arap: {
      outMaterial.params = inParams.arap;
      break;
    }
    default: {
      MOCHI_ASSERT(false, "Unexpected soft material type.");
      break;
    }
  }
  outMaterial.density = inParams.density;
  outMaterial.massDampingCoefficient = inParams.massDampingCoefficient;
  outMaterial.stiffnessDampingCoefficient = inParams.stiffnessDampingCoefficient;
  outMaterial.stiffnessDampingIncludeGeometricTerm = inParams.stiffnessDampingIncludeGeometricTerm;

  // Build per-element constants for FEM assembler (homogeneous: size-1).
  std::visit(
      [&](auto const& p) { outMaterial.perElementParams = materials::BuildPerElementParams(p); },
      outMaterial.params);

  // Precompute the reference material stiffness for stiffness damping (homogeneous: size-1).
  RebuildReferenceMaterialStiffness(outMaterial, /*numEntries*/ 1);
}

void mochi::soft::SetMaterialParamsField(
    PerElementSoftMaterialData const* materialField,
    int numElements,
    CSoftMaterialParams& outMaterial,
    Error& error) {
  MOCHI_ERROR_IF(!materialField, error, "Material field must not be null.");
  MOCHI_ERROR_RETURN(error);
  mochi::ValidateSoftMaterialParams(
      PerElementSoftMaterialDataView{*materialField}, numElements, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      materialField->type == GetSoftMaterialType(outMaterial.params),
      error,
      "Per-element material type must match actor material type.");
  MOCHI_ERROR_RETURN(error);

  // Build per-element constants for FEM assembler (heterogeneous: size-N).
  std::visit(
      [&](auto const& baseP) {
        using ParamsT = std::decay_t<decltype(baseP)>;
        if constexpr (materials::kIsLameMaterial<ParamsT>) {
          auto const psd =
              ResolvePerElementFieldPsdStrategy(baseP, materialField->psdStrategy, error);
          MOCHI_ERROR_RETURN(error);
          outMaterial.perElementParams = materials::BuildPerElementLameParams(
              MakeConstSpan(materialField->youngsModulus),
              MakeConstSpan(materialField->poissonRatio),
              psd);
        } else if constexpr (std::is_same_v<ParamsT, ArapMaterialParams>) {
          auto const psd =
              ResolvePerElementFieldPsdStrategy(baseP, materialField->psdStrategy, error);
          MOCHI_ERROR_RETURN(error);
          materials::PerElementArapParams out;
          out.stiffness.assign(
              materialField->arapStiffness.begin(), materialField->arapStiffness.end());
          out.psdStrategy = psd;
          outMaterial.perElementParams = std::move(out);
        } else if constexpr (std::is_same_v<ParamsT, ActiveShapeTargetingArapMaterialParams>) {
          auto const psd =
              ResolvePerElementFieldPsdStrategy(baseP, materialField->psdStrategy, error);
          MOCHI_ERROR_RETURN(error);
          materials::PerElementActiveShapeTargetingArapParams out;
          out.stiffness.assign(
              materialField->arapStiffness.begin(), materialField->arapStiffness.end());
          out.shapeTargetTensor.assign(
              materialField->shapeTargetTensor.begin(), materialField->shapeTargetTensor.end());
          out.psdStrategy = psd;
          outMaterial.perElementParams = std::move(out);
        } else {
          static_assert(std::is_same_v<ParamsT, ActiveNeoHookeanMaterialParams>);
          auto const lamePsd = ResolvePerElementFieldPsdStrategy(
              baseP.passiveIsotropic, materialField->psdStrategy, error);
          auto const anisoPsd = ResolvePerElementFieldPsdStrategy(
              baseP.activeAnisotropic, materialField->psdStrategy, error);
          MOCHI_ERROR_RETURN(error);
          materials::PerElementActiveNeoHookeanParams out;
          out.lame = materials::BuildPerElementLameParams(
              MakeConstSpan(materialField->youngsModulus),
              MakeConstSpan(materialField->poissonRatio),
              lamePsd);
          out.aniso.alpha.assign(
              materialField->anisoAlpha.begin(), materialField->anisoAlpha.end());
          out.aniso.length.assign(
              materialField->anisoLength.begin(), materialField->anisoLength.end());
          out.aniso.anisoDir.resize_noinit(numElements);
          for (int j = 0; j < numElements; ++j) {
            out.aniso.anisoDir[j] = ActiveAnisoArapMaterialParams::ComputeFiberDirection(
                materialField->anisoTheta[j], materialField->anisoPhi[j]);
          }
          out.aniso.psdStrategy = anisoPsd;
          outMaterial.perElementParams = std::move(out);
        }
      },
      outMaterial.params);

  // Precompute the reference material stiffness for stiffness damping (heterogeneous: size-N).
  RebuildReferenceMaterialStiffness(outMaterial, numElements);
}

void mochi::soft::SetMaterialParamsField(
    SoftMaterialParams const& materialParams,
    int elementIndex,
    int numElements,
    CSoftMaterialParams& outMaterial,
    Error& error) {
  // Per-element updates do not apply density. Validate against the actor-wide density.
  auto paramsForValidation = materialParams;
  paramsForValidation.density = kDefaultDensity;
  mochi::ValidateSoftMaterialParams(paramsForValidation, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      elementIndex < 0 || elementIndex >= numElements,
      error,
      "Element index should be greater than or equal to zero and smaller than the number of elements.");
  MOCHI_ERROR_IF_NOT(
      materialParams.type == GetSoftMaterialType(outMaterial.params),
      error,
      "Element material type must match the existing actor material type.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      IsMaterialParamsFieldCompatible(outMaterial, materialParams),
      error,
      "Element material PSD strategy must match the existing actor material PSD strategy.");
  MOCHI_ERROR_RETURN(error);

  static_assert(
      static_cast<int>(SoftMaterialType::Count) == 6,
      "Please update the logic below if SoftMaterialType enum changes");
  std::visit(
      [&](auto const& baseP) {
        using P = std::decay_t<decltype(baseP)>;
        auto const homogeneous = materials::BuildPerElementParams(baseP);
        auto& c = soft::details::GetMatchingPerElementParams<P>(outMaterial);
        auto const& p = soft::details::GetTypedMaterialParams<P>(materialParams);

        if constexpr (materials::kIsLameMaterial<P>) {
          if (isize(c) < numElements) {
            c.mu.resize(numElements, homogeneous.mu[0]);
            c.lambda.resize(numElements, homogeneous.lambda[0]);
          }
          auto const [lam, mu] =
              materials::utils::ComputeLameConstants(p.youngsModulus, p.poissonRatio);
          c.mu[elementIndex] = mu;
          c.lambda[elementIndex] = lam;
        } else if constexpr (std::is_same_v<P, ArapMaterialParams>) {
          if (isize(c) < numElements) {
            c.stiffness.resize(numElements, homogeneous.stiffness[0]);
          }
          c.stiffness[elementIndex] = p.stiffness;
        } else if constexpr (std::is_same_v<P, ActiveShapeTargetingArapMaterialParams>) {
          if (isize(c) < numElements) {
            int const oldSize = isize(c);
            c.stiffness.resize(numElements, homogeneous.stiffness[0]);
            c.shapeTargetTensor.resize_noinit(numElements * 6);
            for (int i = oldSize; i < numElements; ++i) {
              std::copy_n(
                  homogeneous.shapeTargetTensor.data(), 6, c.shapeTargetTensor.data() + i * 6);
            }
          }
          c.stiffness[elementIndex] = p.stiffness;
          std::copy_n(p.shapeTargetTensor.data(), 6, c.shapeTargetTensor.data() + elementIndex * 6);
        } else {
          static_assert(std::is_same_v<P, ActiveNeoHookeanMaterialParams>);
          if (isize(c) < numElements) {
            c.lame.mu.resize(numElements, homogeneous.lame.mu[0]);
            c.lame.lambda.resize(numElements, homogeneous.lame.lambda[0]);
            c.aniso.alpha.resize(numElements, homogeneous.aniso.alpha[0]);
            c.aniso.length.resize(numElements, homogeneous.aniso.length[0]);
            c.aniso.anisoDir.resize(numElements, homogeneous.aniso.anisoDir[0]);
          }
          auto const [lam, mu] = materials::utils::ComputeLameConstants(
              p.passiveIsotropic.youngsModulus, p.passiveIsotropic.poissonRatio);
          c.lame.mu[elementIndex] = mu;
          c.lame.lambda[elementIndex] = lam;
          c.aniso.alpha[elementIndex] = p.activeAnisotropic.alpha;
          c.aniso.length[elementIndex] = p.activeAnisotropic.length;
          c.aniso.anisoDir[elementIndex] = p.activeAnisotropic.anisoDir;
        }
      },
      outMaterial.params);

  // Keep the reference material stiffness (stiffness damping) in sync with the updated element.
  UpdateReferenceMaterialStiffnessElement(outMaterial, elementIndex, numElements);
}

bool mochi::soft::IsMaterialParamsFieldCompatible(
    CSoftMaterialParams const& actorParams,
    SoftMaterialParams const& elementParams) {
  if (elementParams.type != GetSoftMaterialType(actorParams.params)) {
    return false;
  }

  static_assert(
      static_cast<int>(SoftMaterialType::Count) == 6,
      "Please update the logic below if SoftMaterialType enum changes");
  switch (elementParams.type) {
    case SoftMaterialType::NeoHookean:
      return soft::details::GetMatchingPerElementParams<NeoHookeanMaterialParams>(actorParams)
                 .psdStrategy == materials::utils::ResolvePsdStrategy(elementParams.neoHookean);
    case SoftMaterialType::StVenantKirchhoff:
      return soft::details::GetMatchingPerElementParams<StVenantKirchhoffMaterialParams>(
                 actorParams)
                 .psdStrategy ==
          materials::utils::ResolvePsdStrategy(elementParams.stVenantKirchhoff);
    case SoftMaterialType::LinearElastic:
      return true;
    case SoftMaterialType::ActiveNeoHookean: {
      auto const& perElementParams =
          soft::details::GetMatchingPerElementParams<ActiveNeoHookeanMaterialParams>(actorParams);
      return perElementParams.lame.psdStrategy ==
          materials::utils::ResolvePsdStrategy(elementParams.activeNeoHookean.passiveIsotropic) &&
          perElementParams.aniso.psdStrategy ==
          materials::utils::ResolvePsdStrategy(elementParams.activeNeoHookean.activeAnisotropic);
    }
    case SoftMaterialType::ActiveShapeTargetingArap:
      return soft::details::GetMatchingPerElementParams<ActiveShapeTargetingArapMaterialParams>(
                 actorParams)
                 .psdStrategy ==
          materials::utils::ResolvePsdStrategy(elementParams.activeShapeTargetingArap);
    case SoftMaterialType::Arap:
      return soft::details::GetMatchingPerElementParams<ArapMaterialParams>(actorParams)
                 .psdStrategy == materials::utils::ResolvePsdStrategy(elementParams.arap);
    default:
      MOCHI_ASSERT(false, "Unexpected soft material type.");
      return false;
  }
}

void mochi::soft::UpdateSoftMass(
    CActorSnle const& actorSnle,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CSoftMaterialParams const& material,
    CFemVolumeDiscretizationP1Q4 const& femHighVolDisc,
    CFullSparsityPattern const& sparsity,
    CMassMatrix& outMassMatrix,
    CPerElementMassMatrix<CFemVolumeDiscretizationP1Q4>& outPerElemMass,
    CLumpedMassMatrix& outLumpedMass,
    CActiveVolumeElements const* activeVolElems) {
  MOCHI_PROFILE_SCOPE();

  // Wrap the mass matrix values in a BlockSparseMatrixView.
  MOCHI_ASSERT_VERBOSE(
      (std::holds_alternative<BlockSparseMatrix<real, 3>>(actorSnle.fullDResidual)),
      "Expected block sparse actor matrix.");
  auto const& dresidual = std::get<BlockSparseMatrix<real, 3>>(actorSnle.fullDResidual);
  BlockSparseMatrixView<real, 3> outMassMatrixBSp(
      dresidual.BlockCols(), dresidual.Pointers(), dresidual.Indices(), outMassMatrix.values);

  // Initialize to zero.
  outPerElemMass.values.resize_noinit(femHighVolDisc.femElements.size());
  deformable::SetZeroMassMatrix(outMassMatrixBSp, MakeSpan(outPerElemMass.values));

  MOCHI_ASSERT(
      !activeVolElems || !activeVolElems->empty(), "Active volume elements must not be empty.");
  Span<int const> activeVolIndices =
      activeVolElems ? activeVolElems->ViewIndices() : Span<int const>{};
  Span<real const> activeVolWeights =
      activeVolElems ? activeVolElems->ViewWeights() : Span<real const>{};

  // we need to first recompute the per-element mass matrices since this is used
  // below for assembling the global mass matrix
  fem::ComputeMassMatrixPerElement(
      MakeConstSpan(femHighVolDisc.femElements),
      material.density,
      MakeSpan(outPerElemMass.values),
      activeVolIndices,
      activeVolWeights);

  //
  // here we only request to assemble the DRes because we want
  // to assemble the mass matrix
  //
  AssemblyParams params{.assemObj = false, .assemRes = false, .assemDRes = true, .psdDRes = false};

  AssemblyActiveSubset activeSubset = activeVolElems
      ? AssemblyActiveSubset{activeVolIndices, activeVolElems->ViewIsActive()}
      : AssemblyActiveSubset{};

  // Assemble the global mass matrix from precomputed per-element mass matrices.
  auto const massMatrixOp = deformable::MakeAddMassMatrixToDResOp<SoftStencilElement>(
      MakeConstSpan(outPerElemMass.values));
  AssembleObjResDRes<SoftStencilElement>(
      l2g,
      nbs,
      massMatrixOp,
      AssemblyResults<real>{
          .outObj = nullptr, .outRes = {}, .outDRes = outMassMatrixBSp, .params = params},
      activeSubset);

  // Compute lumped mass matrix.
  deformable::ComputeLumpedMassMatrix(sparsity, outMassMatrix, outLumpedMass);
}

MOCHI_API void mochi::soft::RecenterSolutionUsingRigidTransformEval(
    ecs::RequiredTag<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CRecenteringParams const& params,
    CSimplicialMesh const& simplicial,
    CRigidTransformEvalPoint const& pivot,
    CRootTransform& root,
    CDisplacementSlice<real, TimeStep::Current>& currDispl,
    CDisplacementSlice<real, TimeStep::Previous>& prevDispl,
    CIntegrationDisplacementSlices& intDispls,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CVelocitySlice<real, TimeStep::Previous>& prevVel,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CRigidTransformEval& pivotEval,
    CBoundingVolume<TimeStep::Current>* currBounds,
    CBoundingVolume<TimeStep::Previous>* prevBounds) {
  MOCHI_PROFILE_SCOPE();
  if (!params.useRecentering) {
    return;
  }

  // Get the evaluation point (current and reference)
  Real3 const& refPivotPos = pivot.GetPositionReference();
  TransformRT refPivot = TransformRT{Quaternion::Identity(), refPivotPos};
  TransformRT newPivot = pivotEval.value;
  TransformRT const deltaPivot = newPivot * Invert(refPivot);

  // Early out if the evaluation point hasn't rotated or translated far enough
  bool hasEnoughTranslation =
      !NearEqual(Real3{}, deltaPivot.GetTranslation(), params.translationEpsilon);
  Real3 rotationAxis = {};
  real rotationAngle = 0_r; // radians
  deltaPivot.GetRotation().ToAxisAngle(&rotationAxis, &rotationAngle);
  real rotationEps = kRadiansPerDegree * params.rotationEpsilonDeg;
  bool hasEnoughRotation = !NearEqual(0_r, rotationAngle, rotationEps);
  if (!hasEnoughRotation && !hasEnoughTranslation) {
    return;
  }

  // Move CRootTransform so that the pivot remains fixed in world space.
  // In other words: (newWorldFromLocal * refPivot) == (oldWorldFromLocal * newPivot)
  TransformRT oldWorldFromLocal = root.worldFromLocal;
  TransformRT newWorldFromLocal = oldWorldFromLocal * deltaPivot;
  newWorldFromLocal.SetRotation(Normalize(
      newWorldFromLocal.GetRotation())); // Normalize the quaternion to avoid drift over time.
  MOCHI_ASSERT_VERBOSE(
      NearEqual(newWorldFromLocal * refPivot, oldWorldFromLocal * newPivot, 1e-5_r));
  root.worldFromLocal = newWorldFromLocal;

  // Transform the pivot in the opposite direction (inverse of deltaPivot) to compensate
  MOCHI_ASSERT_VERBOSE(NearEqual(refPivot, Invert(deltaPivot) * pivotEval.value, 1e-4_r));

  // Reset the pivot eval so that it is correct in the new reference frame.
  pivotEval.value = refPivot;

  // Transform kinematics to compensate for the new root transform.
  TransformRT const newLocalFromOldLocal = Invert(newWorldFromLocal) * oldWorldFromLocal;
  Span<real const> refCoords = Flatten(simplicial.mesh->GetNodeCoordinates());

  currDispl.ApplyTransform(newLocalFromOldLocal, refCoords);
  prevDispl.ApplyTransform(newLocalFromOldLocal, refCoords);
  for (auto& prev : intDispls.prevSteps) {
    prev.ApplyTransform(newLocalFromOldLocal, refCoords);
  }
  currVel.ApplyTransform(newLocalFromOldLocal, refCoords);
  prevVel.ApplyTransform(newLocalFromOldLocal, refCoords);
  for (auto& prev : intVels.prevSteps) {
    prev.ApplyTransform(newLocalFromOldLocal, refCoords);
  }

  // Update the local bounding volume (an Obb for soft actors)
  if (currBounds) {
    currBounds->localShape = TransformShape(newLocalFromOldLocal, currBounds->localShape);
  }
  if (prevBounds) {
    prevBounds->localShape = TransformShape(newLocalFromOldLocal, prevBounds->localShape);
  }
}

template <class EvalPointT>
static void EvaluateDisplacements(
    EvalPointT const& x,
    CLocal2GlobalMap const& l2g,
    ColumnVectorView<real const> sol,
    Real3* valueOut,
    Matrix3x3r* gradientOut) {
  MOCHI_PROFILE_SCOPE();

  // Get the DoFs in order [kSpaceDim][kNumEleNodes] because that is the order that EvalPoint
  // expects.
  NdArray<real, EvalPointT::kSpaceDim, EvalPointT::kNumEleNodes> dofValues;
  for (int dof = 0; dof < EvalPointT::kNumEleNodes; ++dof) {
    for (int field = 0; field < EvalPointT::kSpaceDim; ++field) {
      dofValues[field][dof] =
          sol[l2g.GetGlobalIndex(x.GetElementIndex(), EvalPointT::kSpaceDim * dof + field)];
    }
  }

  // Interpolate at x
  *valueOut = x.Eval(dofValues);
  *gradientOut = x.DEval(dofValues);
}

void mochi::SetZeroDisplacements(entt::registry& reg, entt::entity e, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  auto* currDispl = reg.try_get<CDisplacementSlice<real, TimeStep::Current>>(e);
  MOCHI_ERROR_IF(
      currDispl == nullptr, error, "CDisplacementSlice<real, TimeStep::Current> required.");
  auto* prevDispl = reg.try_get<CDisplacementSlice<real, TimeStep::Previous>>(e);
  MOCHI_ERROR_IF(
      prevDispl == nullptr, error, "CDisplacementSlice<real, TimeStep::Previous> required.");
  MOCHI_ERROR_RETURN(error);

  currDispl->value.SetZero();
  prevDispl->value.SetZero();

  // External state changes invalidate step history.
  InvalidateActorStepHistory(reg, e);
}

void mochi::SetNodePositionsLocal(
    entt::registry& reg,
    entt::entity e,
    Span<real const> inPositionsLocal,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  MOCHI_ERROR_IF(
      reg.any_of<TagSoftSkinnedActor>(e),
      error,
      "SetNodePositionsLocal is not supported for soft-skinned actors.");
  auto* currDispl = reg.try_get<CDisplacementSlice<real, TimeStep::Current>>(e);
  MOCHI_ERROR_IF(
      currDispl == nullptr, error, "CDisplacementSlice<real, TimeStep::Current> required.");
  auto* prevDispl = reg.try_get<CDisplacementSlice<real, TimeStep::Previous>>(e);
  MOCHI_ERROR_IF(
      prevDispl == nullptr, error, "CDisplacementSlice<real, TimeStep::Previous> required.");
  auto const* cmesh = MOCHI_TRY_GET(CSimplicialMesh const, reg, e, error);
  MOCHI_ERROR_RETURN(error);

  int const numValues = cmesh->mesh->GetNumNodes() * kSpaceDim3;
  MOCHI_ASSERT(currDispl->value.size() == numValues, "Array size mismatch");
  MOCHI_ASSERT(prevDispl->value.size() == numValues, "Array size mismatch");
  MOCHI_ERROR_IF(isize(inPositionsLocal) != numValues, error, "Array size mismatch");
  MOCHI_ERROR_IF_NOT(IsFinite(inPositionsLocal), error, "Node positions must be finite.");
  MOCHI_ERROR_RETURN(error);

  // Subtract each position from the corresponding reference position because the
  // current solution vector stores just the displacements.
  ArraySub(
      MakeSpan(currDispl->value), inPositionsLocal, Flatten(cmesh->mesh->GetNodeCoordinates()));

  prevDispl->CopyFrom(*currDispl);

  // External state changes invalidate step history.
  InvalidateActorStepHistory(reg, e);
}

void mochi::SetZeroVelocities(entt::registry& reg, entt::entity e, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();
  auto* prevVel = reg.try_get<CVelocitySlice<real, TimeStep::Previous>>(e);
  MOCHI_ERROR_IF(prevVel == nullptr, error, "Requires CVelocitySlice<real, TimeStep::Previous>.");
  MOCHI_ERROR_RETURN(error);
  prevVel->value.SetZero();
  auto* currVel = reg.try_get<CVelocitySlice<real, TimeStep::Current>>(e);
  MOCHI_ERROR_IF(currVel == nullptr, error, "Requires CVelocitySlice<real, TimeStep::Current>.");
  MOCHI_ERROR_RETURN(error);
  currVel->value.SetZero();

  // Zero skinned velocity layers if they exist (soft-skinned actors).
  if (auto* prevVelSkinned =
          reg.try_get<CVelocitySlice<real, TimeStep::Previous, DisplacementLayer::Skinned>>(e)) {
    prevVelSkinned->value.SetZero();
  }
  if (auto* currVelSkinned =
          reg.try_get<CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>>(e)) {
    currVelSkinned->value.SetZero();
  }

  // External state changes invalidate step history.
  InvalidateActorStepHistory(reg, e);
}

void mochi::SetNodeVelocitiesLocal(
    entt::registry& reg,
    entt::entity e,
    Span<real const> inVelocitiesLocal,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  MOCHI_ERROR_IF(
      reg.any_of<TagSoftSkinnedActor>(e),
      error,
      "SetNodeVelocitiesLocal is not supported for soft-skinned actors.");
  auto* prevVel = reg.try_get<CVelocitySlice<real, TimeStep::Previous>>(e);
  MOCHI_ERROR_IF(prevVel == nullptr, error, "Requires CVelocitySlice<real, TimeStep::Previous>.");
  MOCHI_ERROR_IF(
      prevVel && prevVel->value.size() != inVelocitiesLocal.size(), error, "Array size mismatch");
  MOCHI_ERROR_IF_NOT(IsFinite(inVelocitiesLocal), error, "Node velocities must be finite.");
  MOCHI_ERROR_RETURN(error);

  prevVel->value = AsConstView(inVelocitiesLocal); // copy values
  auto* currVel = reg.try_get<CVelocitySlice<real, TimeStep::Current>>(e);
  MOCHI_ERROR_IF(currVel == nullptr, error, "Requires CVelocitySlice<real, TimeStep::Current>.");
  MOCHI_ERROR_IF(
      currVel && currVel->value.size() != inVelocitiesLocal.size(), error, "Array size mismatch");
  MOCHI_ERROR_RETURN(error);
  currVel->value = AsConstView(inVelocitiesLocal); // copy values

  // External state changes invalidate step history.
  InvalidateActorStepHistory(reg, e);
}

static void AssembleElasticEnergy(
    ColumnVectorView<real const> currSol,
    soft::FemInfo const& femVolLow,
    CSoftMaterialParams const& materialParams,
    Local2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    real& outEnergy,
    CActiveVolumeElements const* activeVolElems) {
  MOCHI_PROFILE_SCOPE();

  MOCHI_ASSERT(
      !activeVolElems || !activeVolElems->empty(), "Active volume elements must not be empty.");
  Span<real const> activeVolWeights =
      activeVolElems ? activeVolElems->ViewWeights() : Span<real const>{};

  double energy = 0;
  AssemblyParams params{.assemObj = true, .assemRes = false, .assemDRes = false};

  AssemblyActiveSubset activeSubset = activeVolElems
      ? AssemblyActiveSubset{activeVolElems->ViewIndices(), activeVolElems->ViewIsActive()}
      : AssemblyActiveSubset{};

  soft::details::VisitPerElementMaterialParams(
      materialParams, [&]<typename ParamsT>(auto const& perElem) {
        auto batchedConstitutive = materials::MakeBatchedConstitutiveResponse<ParamsT>(perElem);

        auto const stressEnergyOp = MakeBatchedStressEnergyOp(
            MakeConstSpan(femVolLow.femElements), batchedConstitutive, activeVolWeights);

        AssembleObjResDRes<soft::SoftStencilElement>(
            l2g,
            nbs,
            stressEnergyOp,
            currSol,
            AssemblyResults<real>{.outObj = &energy, .outRes = {}, .outDRes = {}, .params = params},
            activeSubset);
      });
  outEnergy = static_cast<real>(energy);
}

void soft::UpdateQueryElasticEnergy(
    ecs::RequiredTag<TagSoftActor>,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    CStressDisplacementRef const& currDisp,
    FemInfo const& femVolLow,
    CSoftMaterialParams const& material,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CQueryElasticEnergy& outQuery,
    CActiveVolumeElements const* activeVolElems) {
  if (!hasStressTag) {
    outQuery.isEnergyAtRestInitialized = true;
    outQuery.energy = outQuery.energyAtRest = 0_r;
    return;
  }
  if (!outQuery.isEnergyAtRestInitialized) {
    auto zero = ColumnVector<real>::Zero(currDisp.value.Rows());
    AssembleElasticEnergy(
        zero, femVolLow, material, l2g, nbs, outQuery.energyAtRest, activeVolElems);
    outQuery.isEnergyAtRestInitialized = true;
  }
  AssembleElasticEnergy(
      currDisp.value, femVolLow, material, l2g, nbs, outQuery.energy, activeVolElems);
}

void soft::UpdateQueryElementsDeformationGradient(
    ecs::RequiredTag<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CFinalDisplacementRef<TimeStep::Current> const& currDisp,
    FemInfo const& femVolumeDisc,
    CLocal2GlobalMap const& l2g,
    CQueryElementsDeformationGradient& outQuery) {
  MOCHI_PROFILE_SCOPE();

  auto const& elements = femVolumeDisc.femElements;
  int const numElements = femVolumeDisc.femElements.size();
  {
    constexpr int kDefGradientSize = 9;
    int const vecSize = numElements * kDefGradientSize;
    outQuery.elementsDeformationGradient.resize(vecSize);
  }

  int constexpr kMinPerTask = 256;
  ParallelForN("Evaluate elements deformation gradients", numElements, kMinPerTask, [&](int iElt) {
    constexpr int kNumFields = kSpaceDim3; // One field for each direction of displacements
    constexpr int kNumEleDofs = FemInfo::ElementT::kNumDofs * kNumFields;
    constexpr int kSpaceDim = FemInfo::ElementT::kSpaceDim;
    constexpr int kDefGradientSize = 9;
    NdArray<real, kNumEleDofs> displacementsSlice;
    auto const* element = &(elements[iElt]);
    auto globalDofs = l2g.GetGlobalIndices(iElt);

    // Get the slice of element dofs values
    ArrayGetTriples<kNumEleDofs>(currDisp.value.data(), globalDofs, displacementsSlice.data());

    constexpr int kNumDofsPerField = FemInfo::ElementT::kNumDofs;
    auto const* displacementsSliceUnflattened =
        reinterpret_cast<NdArray<real, kNumDofsPerField, kNumFields> const*>(
            (displacementsSlice.data()));

    int const indx = iElt * kDefGradientSize;

    auto* elementDeformationGradient =
        reinterpret_cast<NdArray<real, 3, 3>*>(&(outQuery.elementsDeformationGradient[indx]));

    EvaluateFieldGradient<kNumFields, kNumDofsPerField, kSpaceDim>(
        element->dBasisEvaluated[0], *displacementsSliceUnflattened, elementDeformationGradient);

    *elementDeformationGradient += kSecondOrderIdentity;
  });
}

void soft::UpdateQueryQuadraturePointsPosition(
    ecs::RequiredTag<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CFinalDisplacementRef<TimeStep::Current> const& currDisp,
    FemInfo const& femVolumeDisc,
    CLocal2GlobalMap const& l2g,
    CQueryQuadraturePointsPosition& outQuery) {
  MOCHI_PROFILE_SCOPE();

  int const numElements = femVolumeDisc.femElements.size();
  {
    constexpr int kNumQuadsPerElement = FemInfo::ElementT::kNumQuadPoints;
    constexpr int kSpaceDim = FemInfo::ElementT::kSpaceDim;
    int const numQuads = numElements * kNumQuadsPerElement;
    int const vecSize = numQuads * kSpaceDim;

    // populate reference positions if not existent
    if (outQuery.quadraturePointsReferencePosition.size() != vecSize) {
      outQuery.quadraturePointsReferencePosition.resize(vecSize);
      // loop over all elements and all quadrature points per element
      for (int e = 0; e < numElements; ++e) {
        for (int q = 0; q < kNumQuadsPerElement; ++q) {
          auto const& quadPoint = femVolumeDisc.femElements[e].mapEvaluated[q];
          for (int i = 0; i < kSpaceDim; ++i) {
            outQuery.quadraturePointsReferencePosition
                [e * kNumQuadsPerElement * kSpaceDim + q * kSpaceDim + i] = quadPoint[i];
          }
        }
      }
    }

    // The outQuer.quadraturePointsWorldPosition is the vector containing
    // flattened world positions of all quadrature points
    outQuery.quadraturePointsWorldPosition.resize(vecSize);
  }
  int constexpr kMinPerTask = 256;
  ParallelForN("Map quadrature points", numElements, kMinPerTask, [&](int e) {
    constexpr int kNumQuadsPerElement = FemInfo::ElementT::kNumQuadPoints;
    constexpr int kSpaceDim = FemInfo::ElementT::kSpaceDim;
    constexpr int kNumFields = kSpaceDim;
    constexpr int kNumDofsPerField = FemInfo::ElementT::kNumDofs;
    constexpr int kNumEleDofs = kNumDofsPerField * kNumFields;

    auto const* element = &(femVolumeDisc.femElements[e]);
    NdArray<real, kNumQuadsPerElement, kNumDofsPerField> const basisEvaluated =
        element->basisEvaluated;
    auto const globalDofs = l2g.GetGlobalIndices(e);

    // The element vertex displacements
    NdArray<real, kNumEleDofs> displacementsSlice;

    // Get the slice of element dofs values
    ArrayGetTriples<kNumEleDofs>(currDisp.value.data(), globalDofs, displacementsSlice.data());

    auto const* displacementsSliceUnflattened =
        reinterpret_cast<NdArray<real, kNumDofsPerField, kNumFields> const*>(
            (displacementsSlice.data()));

    for (int q = 0; q < element->kNumQuadPoints; ++q) {
      int const indx = e * kNumQuadsPerElement * kSpaceDim + q * kSpaceDim;

      auto* worldPosition =
          reinterpret_cast<Real3*>(&(outQuery.quadraturePointsWorldPosition[indx]));

      EvaluateField<kNumFields, kNumDofsPerField>(
          basisEvaluated[q], *displacementsSliceUnflattened, worldPosition);

      (*worldPosition) +=
          *(reinterpret_cast<Real3*>(&(outQuery.quadraturePointsReferencePosition[indx])));
    }
  });
}

void mochi::soft::ComputeTransformAtEvalPoint(
    ColumnVectorView<real const> displ,
    CLocal2GlobalMap const& l2g,
    CRigidTransformEvalPoint const& evalPoint,
    TransformRT& outTransform) {
  MOCHI_PROFILE_SCOPE();

  auto prevTransform = outTransform;

  // Evaluate the pivot displacements
  Real3 pivotDisplacement;
  Matrix3x3r pivotDisplacementGradient;
  EvaluateDisplacements(evalPoint, l2g, displ, &pivotDisplacement, &pivotDisplacementGradient);

  /**
    Note[Nate] This comment block was adapted from Philip's original rigid pivot code...

    For a soft body, we want to approximate the local-space configuration of the body by a rigid
    transform from the reference configuration. The Taylor approximation to the world space
    configuration map Phi at the pivot X is given by:

    Phi(X + dX) = Phi(X) + DPhi * dX

    The returned rigid body transform with rotation R and translation b will try to match the map
    Phi exactly at X and approximately at the tangent space around X. R should therefore be chosen
    such that when the inverse is applied, we get:

    R^{-1} DPhi ~ I

    Hence, we take R = U where U is the othogonal factor in the polar decomposition DPhi = UP.
    Note that the inversion result R^{-1} DPhi = P ~ I  will therefore be a pure stretch (PSD
    matrix). The affine translation from the actor's local space origin is simply:

    b = X + dX
  */

  Real3 X = evalPoint.GetPositionReference();
  Real3 b = pivotDisplacement + X;
  VMatrix3x3r DPhiX = ToSimdMatrix(pivotDisplacementGradient) + VEye<3, real>();
  VMatrix3x3r R, P;
  LeftPolarDecomposition3x3(DPhiX, R, P);

  // R is the local-space rotation matrix, with basis vectors stored as columns.
  // RT is the transpose, with basis vectors stored as rows.
  VMatrix3x3r RT = Transpose3x3(R);

  Vec4r cross12 = Normalize<3>(Cross3(RT[1], RT[2]));
  if (Dot<3>(RT[0], cross12) < 1e-3_r) {
    // The evaluated element is inverted (or nearly so), resulting in negative scale on one of the
    // basis vectors. Do not try to update the pivot's quaternion rotation until the element
    // recovers.
    outTransform.SetRotation(prevTransform.GetRotation());
    outTransform.SetTranslation(b);
  } else {
    // Build an orthonormal frame before converting to a quaternion
    // (FromOrthoNormalTranspose assumes an exactly orthonormal input).
    Vec4r const row2 = Normalize<3>(RT[2]);
    Vec4r const row0 = cross12; // = Normalize<3>(Cross3(RT[1], RT[2])), orthogonal to RT[2]
    Vec4r const row1 = Cross3(row2, row0);
    VMatrix4x4r const matT = {row0, row1, row2, ToSimd(b, 1_r)};

    // Use Transpose(matT) to compute an equivalent TransformRT (quaternion form)
    outTransform = TransformRT::FromOrthoNormalTranspose(matT);
  }
}

MOCHI_API void soft::UpdateRigidVelocity(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    ecs::CtxGlobal<CSceneTime const> time,
    CRootTransform const& root,
    CBoundingVolume<TimeStep::Current> const& bounds,
    CPrevRigidVelocity& outRigidVel) {
  // Approximate the center-of-mass using the center-of-volume.
  // Then compute velocity of that point based on the change in root transform.
  //
  // TODO[Nate]: This only works if recentering is enabled with zero tolerance.
  //             It may be better to track the movement of CRigidTransformEval instead.
  outRigidVel.centerOfMassLocal = GetAabb(bounds.localShape).VGetCenter();
  ComputeRigidVelocityWorldSpace(
      static_cast<real>(time->DeltaTime()),
      root.worldFromLocal,
      root.worldFromLocalPrev,
      outRigidVel.centerOfMassLocal,
      outRigidVel.linearVelocityWorld,
      outRigidVel.angularVelocityWorld);
}

void soft::InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CActorAsyncContactSemaphore>(reg);
  ecs::RegisterComponent<CRigidTransformEval>(reg);
  ecs::RegisterComponent<CRigidTransformEvalPoint>(reg);
  ecs::RegisterComponent<CSkinnedEnergy>(reg);
  ecs::RegisterComponent<CStressDisplacementRef>(reg);
  ecs::RegisterComponent<CSoftExportParams>(reg);
}

real soft::GetActorMass(entt::registry const& reg, entt::entity actor) {
  MOCHI_ASSERT(reg.all_of<TagSoftActor>(actor), "Expected soft actor.");
  auto const& material = reg.get<CSoftMaterialParams>(actor);
  auto const& mesh = reg.get<CTetrahedralMesh>(actor);
  real const volume = mesh.mesh->GetTotalMeasure();
  return material.density * volume;
}
