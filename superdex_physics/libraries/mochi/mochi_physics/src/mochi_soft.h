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
#include <mochi_core/element_operations/fem_stress.h>
#include <mochi_core/element_operations/fem_stress_damping.h>
#include <mochi_core/elements/eval_point.h>
#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/utils/batch_config.h>
#include <mochi_core/utils/batch_types.h>

#include <type_traits>
#include <utility>

namespace mochi {

// Forwards
struct CRecordingData;
struct CRomProjectionStrategy;
struct TractionWorkArgs;

/**************************************************************************
  ECS Components for Soft Actors
*/

// Stores the EvalPoint which is used to determine the "rigid pivot" of a soft mesh. This is a
// specific location within a particular element of the discretization. By default, it will be
// located at the center-of-mass of the reference mesh, however, a different point might be chosen
// (e.g. if the COM is not within an element).
template <class ElementT>
struct CRigidTransformEvalPointT : public EvalPoint<ElementT> {
  using EvalPoint<ElementT>::EvalPoint;
};
using CRigidTransformEvalPoint = CRigidTransformEvalPointT<tetrahedral::Pk3DElement<1, 1>>;

// Evaluation performed at the CRigidTransformEvalPoint are spit out here.
// Rotation & translation of the eval point in the actor's local space
struct CRigidTransformEval {
  TransformRT value;

  MOCHI_STRUCT_BEGIN(mochi::CRigidTransformEval);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

// Component to indicate if soft-actor energies are evaluated on skinned positions.
struct CSkinnedEnergy : public NoCopy {
  bool gravity = false;
  bool inertia = false;
  bool stress = false;
};

// Component to store a reference to the displacement on which stress is evaluated.
struct CStressDisplacementRef : public VectorComponentRef {
  using VectorComponentRef::VectorComponentRef;
};

// ECS component storing soft actor creation params that are consumed during InitSoftActor
// and cannot be recovered from the ECS afterward. Used for lossless prefab export.
struct CSoftExportParams : NoCopy {
  ActorBoundaryElementType boundaryElementType = ActorBoundaryElementType::Default;
};

/**************************************************************************
  Soft Actor Utils
*/

// Set the displacements to zero. Note this is the local displacement thus does not zero out
// recentering transform.
void SetZeroDisplacements(entt::registry& reg, entt::entity e, Error& error);

// Set the position of all nodes by replacing the contents of the solution vector.
// Size of inPositionsLocal == numNodes * kSpaceDim.
void SetNodePositionsLocal(
    entt::registry& reg,
    entt::entity e,
    Span<real const> inPositionsLocal,
    Error& error);

// Zero out the velocities.
void SetZeroVelocities(entt::registry& reg, entt::entity e, Error& error);

// Set the velocity of all nodes in the CGlobalKinematics component.
// Size of inVelocitiesLocal == numNodes * kSpaceDim.
void SetNodeVelocitiesLocal(
    entt::registry& reg,
    entt::entity e,
    Span<real const> inVelocitiesLocal,
    Error& error);

namespace soft {

// Discretization info
using FemInfo = CFemVolumeDiscretizationP1Q1;

/**
 * @brief Compile-time element tag for the 4-node soft volume assembly stencil.
 *
 * @details Used as the @p ElementT parameter for FEM assembly and batched element vectors/matrices.
 * This matches the P1 tetrahedral connectivity used by @ref CFemVolumeDiscretizationP1Q1.
 */
struct SoftStencilElement {
  static constexpr int kSpaceDim = FemInfo::kSpaceDim;
  static constexpr int kNumDofs = FemInfo::kNumEleNodes; // 4 (P1 tet)
};

void UpdateQueryElasticEnergy(
    ecs::RequiredTag<TagSoftActor>,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    CStressDisplacementRef const& currDisp,
    FemInfo const& femVolLow,
    CSoftMaterialParams const& material,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CQueryElasticEnergy& outQuery,
    CActiveVolumeElements const* activeVolElems);

void UpdateQueryElementsDeformationGradient(
    ecs::RequiredTag<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CFinalDisplacementRef<TimeStep::Current> const& currDisp,
    FemInfo const& femVolumeDisc,
    CLocal2GlobalMap const& l2g,
    CQueryElementsDeformationGradient& outQuery);

void UpdateQueryQuadraturePointsPosition(
    ecs::RequiredTag<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CFinalDisplacementRef<TimeStep::Current> const& currDisp,
    FemInfo const& femVolumeDisc,
    CLocal2GlobalMap const& l2g,
    CQueryQuadraturePointsPosition& outQuery);

// Compute the TransformRT at the eval point.
void ComputeTransformAtEvalPoint(
    ColumnVectorView<real const> displ,
    CLocal2GlobalMap const& l2g,
    CRigidTransformEvalPoint const& evalPoint,
    TransformRT& outTransform);

// Update CRigidTransformEval based on the current solution.
inline void UpdateRigidTransformEval(
    CDisplacementSlice<real, TimeStep::Current> const& currSol,
    CLocal2GlobalMap const& l2g,
    CRigidTransformEvalPoint const& evalPoint,
    CRigidTransformEval& outTransformEval) {
  ComputeTransformAtEvalPoint(currSol.value, l2g, evalPoint, outTransformEval.value);
}

// Use the information in CRigidTransformEval to perform a recentering of data.
void RecenterSolutionUsingRigidTransformEval(
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
    CBoundingVolume<TimeStep::Previous>* prevBounds);

// Get SoftMaterialParams from CSoftMaterialParams
void GetMaterialParams(CSoftMaterialParams const& material, SoftMaterialParams& outParams);

// Get SoftMaterialParams from CSoftMaterialParams
void GetMaterialParamsField(
    CSoftMaterialParams const& material,
    int elementIndex,
    SoftMaterialParams& outParams);

// Set material params
void SetMaterialParams(SoftMaterialParams const& inParams, CSoftMaterialParams& outMaterial);

// Set material params field
void SetMaterialParamsField(
    PerElementSoftMaterialData const* materialField,
    int numElements,
    CSoftMaterialParams& outMaterial,
    Error& error);

// Set material params field
void SetMaterialParamsField(
    SoftMaterialParams const& materialParams,
    int elementIndex,
    int numElements,
    CSoftMaterialParams& outMaterial,
    Error& error);

// Returns true if the element material params are compatible with the actor material params.
[[nodiscard]] bool IsMaterialParamsFieldCompatible(
    CSoftMaterialParams const& actorParams,
    SoftMaterialParams const& elementParams);

// Sets an error if there is anything wrong with the given SoftMaterialParams.
void ValidateSoftMaterialParams(SoftMaterialParams const& param, Error& error);

// Update inertia
void UpdateSoftMass(
    CActorSnle const& actorSnle,
    CLocal2GlobalMap const& l2g,
    CNodalBasedStructure const& nbs,
    CSoftMaterialParams const& material,
    CFemVolumeDiscretizationP1Q4 const& femHighVolDisc,
    CFullSparsityPattern const& sparsity,
    CMassMatrix& outMassMatrix,
    CPerElementMassMatrix<CFemVolumeDiscretizationP1Q4>& outPerElemMass,
    CLumpedMassMatrix& outLumpedMass,
    CActiveVolumeElements const* activeVolElems = nullptr);

// Update CSdfMapping based on the deformation of the soft mesh. Must be called for all soft actors
// on each assembly (with kTimeStep = TimeStep::Current), and at the beginning of each integration
// stage (with kTimeStep = TimeStep::StageStart) when ExperimentalEvalParams.explicitNormals = true.
template <TimeStep kTimeStep>
void UpdateMap(
    ecs::RequiredTag<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDisplacementSlice<real, kTimeStep> const& sol,
    CSdfMapping<kTimeStep>& mapSdf) {
  mapSdf->UpdateMap(sol.value.GetConstSpan());
}

/**
 * @brief Per-stage damping and inertia scale factors derived from the material and time step.
 *
 * @details Computed once per assembly. Mass damping requires inertia; stiffness damping requires
 * stress. @ref massScale folds the inertia term (1/dt²) and the mass-damping term together for the
 * mass-matrix-scaled dresidual initialization. @ref massDampingScale [1/s²] (mass-damping
 * coefficient / dtStage) and @ref stiffnessDampingFactor [dimensionless] (stiffness-damping
 * coefficient / dtStage) are passed to the body assembly operator.
 */
struct SoftDampingScales {
  real massScale;
  real massDampingScale;
  real stiffnessDampingFactor;
  bool includeStiffnessDampingGeometricTerm;
};

/**
 * @brief Computes the @ref SoftDampingScales for a body assembly stage.
 *
 * @param[in] materialParams Material params carrying the damping coefficients.
 * @param[in] dtStage Time-integration stage step [s].
 * @param[in] hasInertia Whether the inertia term (and hence mass damping) is active.
 * @param[in] hasStress Whether the stress term (and hence stiffness damping) is active.
 */
[[nodiscard]] inline SoftDampingScales ComputeSoftDampingScales(
    CSoftMaterialParams const& materialParams,
    real dtStage,
    bool hasInertia,
    bool hasStress) {
  real const dtfi2 = 1_r / Sqr(dtStage);
  bool const hasMassDamping = hasInertia && materialParams.massDampingCoefficient > 0_r;
  bool const hasStiffnessDamping = hasStress && materialParams.stiffnessDampingCoefficient > 0_r;
  real const massDampingScale =
      hasMassDamping ? materialParams.massDampingCoefficient / dtStage : 0_r;
  return SoftDampingScales{
      .massScale = (hasInertia ? dtfi2 : 0_r) + massDampingScale,
      .massDampingScale = massDampingScale,
      .stiffnessDampingFactor =
          hasStiffnessDamping ? materialParams.stiffnessDampingCoefficient / dtStage : 0_r,
      .includeStiffnessDampingGeometricTerm = materialParams.stiffnessDampingIncludeGeometricTerm};
}

/**
 * @brief Creates a batched volume assembly operator composing stress, gravity, and inertia work.
 *
 * @warning @p batchedConstitutive and @p referenceMaterialStiffness are captured by reference. They
 * must outlive the returned op and must not be mutated until all invocations of the op have
 * completed.
 * @warning Inertia and mass damping contribute to the objective and residual only. This operator
 * does NOT assemble their dresidual. When assembling the dresidual with @p hasInertia set, the
 * caller must first initialize it with the mass matrix scaled by @ref SoftDampingScales::massScale,
 * which folds together the inertia (1/dt^2) and mass-damping terms (see @ref AssembleBodyImpl).
 * Otherwise the inertia and mass-damping dresidual is missing.
 *
 * @note Mass damping (residual/objective only) is active only when @p hasInertia is true and @p
 * massDampingScale > 0; its dresidual is folded into the mass-matrix-scaled initialization by the
 * caller. Stiffness damping (viscous stress) is active only when @p hasStress is true and @p
 * stiffnessDampingFactor > 0, and contributes to the objective, residual, and dresidual.
 */
template <
    class ElementLow,
    class ElementHigh,
    class ConstitutiveFn,
    int kBatchSize = kDefaultFemBatchSize>
[[nodiscard]] ElOpFnType<SoftStencilElement> MakeBatchedBodyOp(
    Span<ElementLow const> lowVolElements,
    Span<ElementHigh const> highVolElements,
    ConstitutiveFn const& batchedConstitutive,
    materials::PerElementReferenceMaterialStiffness const& referenceMaterialStiffness,
    bool hasStress,
    bool hasGravity,
    bool hasInertia,
    Real3 gravity,
    real density,
    Span<real const> stageStartDispl,
    Span<real const> stageStartVel,
    real dtStage,
    real massDampingScale,
    real stiffnessDampingFactor,
    bool includeStiffnessDampingGeometricTerm,
    Span<real const> activeVolWeights) {
  real const dtfi2 = 1_r / Sqr(dtStage);

  // Hoist the homogeneous reference-stiffness broadcast out of the per-batch lambda. When the store
  // is homogeneous (size()==1), the Voigt tensor gathered by GatherReferenceMaterialStiffnessVoigt
  // is identical for every batch, so build the broadcast once here and capture it by value instead
  // of rebuilding it (36 broadcasts) each batch. Heterogeneous stores still gather per batch.
  bool const hasStiffnessDamping = hasStress && stiffnessDampingFactor > 0_r;
  bool const homogeneousStiffness = hasStiffnessDamping && referenceMaterialStiffness.size() == 1;
  [[maybe_unused]] auto const referenceMaterialStiffnessSize = referenceMaterialStiffness.size();
  NdArray<BatchReal<kBatchSize>, 6, 6> homogeneousStiffnessVoigt{};
  if (homogeneousStiffness) {
    homogeneousStiffnessVoigt = materials::GatherReferenceMaterialStiffnessVoigt<kBatchSize>(
        referenceMaterialStiffness, NdArray<int, kBatchSize>{});
  }

  return [= /*All copies are inexpensive*/, &batchedConstitutive, &referenceMaterialStiffness](
             NdArray<int, kBatchSize> const& elemIndices,
             Span<int const> indicesFlat,
             fem::BatchElementVector<kBatchSize, SoftStencilElement> const& displ,
             BatchDouble<kBatchSize>* outEnergy,
             fem::BatchElementVector<kBatchSize, SoftStencilElement>* outRes,
             fem::BatchElementMatrix<kBatchSize, SoftStencilElement>* outDRes,
             bool projectPsd) -> bool {
    bool out = false;
    bool const hasMassDamping = hasInertia && massDampingScale > 0_r;

    if (hasStress) {
      out |= fem::StressWork<kBatchSize>(
          elemIndices,
          lowVolElements,
          displ,
          outEnergy,
          outRes,
          outDRes,
          projectPsd,
          batchedConstitutive,
          activeVolWeights);
    }
    if (hasGravity && (outEnergy || outRes)) {
      out |= fem::GravityWork<kBatchSize>(
          elemIndices,
          lowVolElements,
          displ,
          outEnergy,
          outRes,
          gravity,
          density,
          activeVolWeights);
    }
    if (hasInertia && (outEnergy || outRes)) {
      fem::BatchElementVector<kBatchSize, SoftStencilElement> stageStartTarget MOCHI_NO_INIT;
      fem::GatherPredTarget<kBatchSize, SoftStencilElement>(
          indicesFlat, elemIndices, stageStartDispl, stageStartVel, dtStage, stageStartTarget);
      out |= fem::InertiaWork<kBatchSize>(
          elemIndices,
          highVolElements,
          displ,
          stageStartTarget,
          outEnergy,
          outRes,
          density,
          dtfi2,
          activeVolWeights);
    }

    // Stage-start displacement only (no velocity), shared by mass and stiffness damping. Their
    // strain-rate dependence is encoded as a displacement difference, so the velocity span is not
    // read (it may be empty, e.g. a posed-stress skinned actor without posed inertia).
    //
    // Only gather it when a consumer will actually read it: mass damping contributes to the
    // objective/residual only, and stiffness damping reads the stage-start strain only for the
    // objective, residual, or geometric tangent term (see @ref fem::StressDampingWork). A
    // material-tangent-only stiffness-damping assembly (dresidual-only with the geometric term off)
    // leaves it unread, so the gather is skipped in that common modified-Newton case; the condition
    // below must stay in sync with StressDampingWork's needStrainWork.
    bool const wantObjOrRes = (outEnergy != nullptr) || (outRes != nullptr);
    bool const needStageStartDispl = (hasMassDamping && wantObjOrRes) ||
        (hasStiffnessDamping && (wantObjOrRes || includeStiffnessDampingGeometricTerm));
    fem::BatchElementVector<kBatchSize, SoftStencilElement> stageStartDisplBatch MOCHI_NO_INIT;
    if (needStageStartDispl) {
      fem::GatherPredTarget<kBatchSize, SoftStencilElement, /*kGatherVelocity*/ false>(
          indicesFlat,
          elemIndices,
          stageStartDispl,
          stageStartVel,
          /*dtStage*/ 0_r,
          stageStartDisplBatch);
    }

    // Mass damping (residual/objective only): α/dt · M · (displ − stageStartDispl). Reuses
    // InertiaWork with massDampingScale as the time-scale. DRes handled via mass-matrix init.
    if (hasMassDamping && (outEnergy || outRes)) {
      out |= fem::InertiaWork<kBatchSize>(
          elemIndices,
          highVolElements,
          displ,
          stageStartDisplBatch,
          outEnergy,
          outRes,
          density,
          massDampingScale,
          activeVolWeights);
    }

    // Stiffness damping (viscous stress): objective, residual, and dresidual.
    if (hasStiffnessDamping) {
      MOCHI_ASSERT_VERBOSE(
          referenceMaterialStiffness.size() == referenceMaterialStiffnessSize,
          "Reference material stiffness cardinality must not change while the body op is in use.");
      MOCHI_ASSERT_VERBOSE(
          !referenceMaterialStiffness.empty(),
          "Reference material stiffness must be built when stiffness damping is enabled.");
      // Reuse the hoisted broadcast for the homogeneous store; gather per batch otherwise.
      NdArray<BatchReal<kBatchSize>, 6, 6> gatheredStiffnessVoigt MOCHI_NO_INIT;
      NdArray<BatchReal<kBatchSize>, 6, 6> const* referenceMaterialStiffnessVoigt =
          &homogeneousStiffnessVoigt;
      if (!homogeneousStiffness) {
        gatheredStiffnessVoigt = materials::GatherReferenceMaterialStiffnessVoigt<kBatchSize>(
            referenceMaterialStiffness, elemIndices);
        referenceMaterialStiffnessVoigt = &gatheredStiffnessVoigt;
      }
      out |= fem::StressDampingWork<kBatchSize>(
          elemIndices,
          lowVolElements,
          displ,
          stageStartDisplBatch,
          outEnergy,
          outRes,
          outDRes,
          projectPsd,
          includeStiffnessDampingGeometricTerm,
          stiffnessDampingFactor,
          *referenceMaterialStiffnessVoigt,
          referenceMaterialStiffness.isIsotropic,
          activeVolWeights);
    }
    return out;
  };
}

namespace details {

template <typename ParamsT>
struct PerElementParamsFor;

template <typename ParamsT>
  requires(materials::kIsLameMaterial<ParamsT>)
struct PerElementParamsFor<ParamsT> {
  using Type = materials::PerElementLameParams;
};

template <>
struct PerElementParamsFor<ArapMaterialParams> {
  using Type = materials::PerElementArapParams;
};

template <>
struct PerElementParamsFor<ActiveShapeTargetingArapMaterialParams> {
  using Type = materials::PerElementActiveShapeTargetingArapParams;
};

template <>
struct PerElementParamsFor<ActiveNeoHookeanMaterialParams> {
  using Type = materials::PerElementActiveNeoHookeanParams;
};

template <typename ParamsT>
using PerElementParamsForT = typename PerElementParamsFor<ParamsT>::Type;

template <typename ParamsT>
[[nodiscard]] PerElementParamsForT<ParamsT> const& GetMatchingPerElementParams(
    CSoftMaterialParams const& params) {
  using PerElemParamsT = PerElementParamsForT<ParamsT>;
  auto const& perElem = params.perElementParams;
  MOCHI_ASSERT_VERBOSE(
      std::holds_alternative<PerElemParamsT>(perElem),
      "Material params and per-element params must hold consistent alternatives.");
  return std::get<PerElemParamsT>(perElem);
}

template <typename ParamsT>
[[nodiscard]] PerElementParamsForT<ParamsT>& GetMatchingPerElementParams(
    CSoftMaterialParams& params) {
  // Delegate to the const overload. params is non-const, so casting away const on the result is
  // well-defined.
  return const_cast<PerElementParamsForT<ParamsT>&>(
      GetMatchingPerElementParams<ParamsT>(std::as_const(params)));
}

// Returns the typed material params field of SoftMaterialParams matching ParamsT.
template <typename ParamsT>
[[nodiscard]] ParamsT const& GetTypedMaterialParams(SoftMaterialParams const& materialParams) {
  if constexpr (std::is_same_v<ParamsT, NeoHookeanMaterialParams>) {
    return materialParams.neoHookean;
  } else if constexpr (std::is_same_v<ParamsT, StVenantKirchhoffMaterialParams>) {
    return materialParams.stVenantKirchhoff;
  } else if constexpr (std::is_same_v<ParamsT, LinearElasticMaterialParams>) {
    return materialParams.linearElastic;
  } else if constexpr (std::is_same_v<ParamsT, ArapMaterialParams>) {
    return materialParams.arap;
  } else if constexpr (std::is_same_v<ParamsT, ActiveShapeTargetingArapMaterialParams>) {
    return materialParams.activeShapeTargetingArap;
  } else {
    static_assert(std::is_same_v<ParamsT, ActiveNeoHookeanMaterialParams>);
    return materialParams.activeNeoHookean;
  }
}

/** @brief Visit the per-element material constants matching the active material type, instantiating
 * only the N valid (ParamsT, PerElemT) combinations instead of N² from nested std::visit. */
template <typename Fn>
void VisitPerElementMaterialParams(CSoftMaterialParams const& material, Fn&& fn) {
  std::visit(
      [&](auto const& p) {
        using ParamsT = std::decay_t<decltype(p)>;
        fn.template operator()<ParamsT>(GetMatchingPerElementParams<ParamsT>(material));
      },
      material.params);
}

} // namespace details

// System to initialize state (position and velocity) for a new step.
void EntityIncrementStep(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CDisplacementSlice<real, TimeStep::Previous>& prevDispl,
    CVelocitySlice<real, TimeStep::Previous>& prevVel);

/*
 * System executed after the solution of the non-linear problem is updated. It MUST update the
 * position components of the soft actor state (aka position state) to make it
 * consistent with the new solution. It may OPTIONALLY update other quantities that are a function
 * of the state (aka derived state). In particular, it updates:
 * - CDisplacementSlice (position state)
 * Any derived state that is required for assembly and not updated in this system MUST be
 * updated in EntityAssemble or in mochi_solve's UpdateDerivedStateBeforeAssembly.
 */
void EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CDisplacementSlice<real, TimeStep::Current>& currSol);

/*
 * System executed after the increment of the non-linear problem is updated. It MUST update the
 * position components of the soft actor state (aka position state) to make it
 * consistent with the new solution. It may OPTIONALLY update other quantities that are a function
 * of the state (aka derived state). In particular, it updates:
 * - CDisplacementSlice (position state)
 * Any derived state that is required for assembly and not updated in this system MUST be
 * updated in EntityAssemble or in mochi_solve's UpdateDerivedStateBeforeAssembly.
 */
void EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CDisplacementSlice<real, TimeStep::Current>& currSol);

/*
 * System executed before the first time integration stage of the time step. It computes the
 * differential variables (displacements and velocities) at the beginning of the step.
 */
void EntityPreFirstStage(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Previous> const& prevDispl,
    CVelocitySlice<real, TimeStep::Previous> const& prevVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels);

/*
 * System executed before each time integration stage. It computes the displacements and velocities
 * at the beginning of the stage.
 */
void EntityPreStage(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CTimeIntegratorState const& intState,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels,
    CDisplacementSlice<real, TimeStep::StageStart>& stageStartDispl,
    CVelocitySlice<real, TimeStep::StageStart>& stageStartVel);

/*
 * System executed after each time integration stage:
 * - It computes the velocities at the end of the stage.
 * - If the solver diverged, it resets the displacements ane velocities to zero.
 * - It pushes the displacements and velocities at the end of the stage to the vectors containing
 *   the displacements and velocities at the end of each time integration stage.
 */
void EntityPostStage(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CConvergenceStatus const& convergence,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::StageStart> const& stageStartDispl,
    CDisplacementSlice<real, TimeStep::Current>& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels);

/*
 * System executed after the last time integration stage in the time step. It computes the
 * displacements and velocities at the end of the step.
 */
void EntityPostLastStage(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Current>& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels);

// Assemble just the volume term into CActorSnle.
void AssembleBody(
    AssemblyParams const& params, // external parameter
    ecs::Included<TagSoftActor>,
    ecs::CtxGlobal<CSceneGravity const> sceneGravity,
    ecs::OptionalTag<TagUseGravity> hasGravityTag,
    ecs::OptionalTag<TagUseInertia> hasInertiaTag,
    ecs::OptionalTag<TagUseStress> hasStressTag,
    ecs::OptionalTag<TagRomActor> isRom,
    ecs::OptionalTag<TagNestedSoftActor> isNestedSoft,
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
    CActiveVolumeElements const* activeVolElems = nullptr);

// Implementation of AssembleBody, which admits different components for the result Snle.
void AssembleBodyImpl(
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
    CActiveVolumeElements const* activeVolElems = nullptr);

void AssembleAsyncContact(
    AssemblyParams const& params, // external parameter
    entt::entity e,
    ecs::Included<TagSoftActor, TagUseContact>,
    ecs::OptionalTag<TagRomActor> isRom,
    ecs::Excluded<TagNestedSoftActor>,
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
    CActiveBoundaryFaces const* activeBoundaryFaces = nullptr);

/*
 * System to copy the position state of the soft actor (i.e. CDisplacementSlice) to a span of reals.
 * The span of reals is usually the components of the non-linear problem solution vector
 * corresponding to the actor, thereby the name.
 */
void EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currSol);

/*
 * System to copy a span of reals to the position state of the soft actor (i.e. CDisplacementSlice).
 * The span of reals is usually the components of the non-linear problem solution vector
 * corresponding to the actor, thereby the name.
 */
void EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    CDisplacementSlice<real, TimeStep::Current>& currSol);

// Update CRigidVelocityLocal by approximating the movement of the center-of-mass
void UpdateRigidVelocity(
    ecs::Included<TagSoftActor>,
    ecs::Excluded<TagRomActor>,
    ecs::CtxGlobal<CSceneTime const> time,
    CRootTransform const& root,
    CBoundingVolume<TimeStep::Current> const& bounds,
    CPrevRigidVelocity& outRigidVelWorld);

// Update CBoundingVolume<TimeStep::Current>.localShape based on the deformation of the mesh. kStep
// defines the data to be used in the update, not the component storing the result. There's no
// CBoundingVolume<TimeStep::StageStart>, as it's not needed. We do bound checks in stage-start
// collision detection, but we can use CBoundingVolume<TimeStep::Current> for this.
template <TimeStep kStep>
void UpdateBounds(
    CColliderInfo const& /*collider*/, // TODO: Is this actually required for the ECS system?
    CTetrahedralMesh const& meshSolver,
    CFinalDisplacementRef<kStep> const& solSolver,
    CBoundingVolume<TimeStep::Current>& outBounds) {
  static_assert(kStep == TimeStep::Current || kStep == TimeStep::StageStart);
  MOCHI_PROFILE_SCOPE();
  auto const& sol = solSolver.value;
  auto boundaryIndices = meshSolver.mesh->GetBoundaryNodes();
  auto nodeCoordinates = meshSolver.mesh->GetNodeCoordinates();
  auto nodeDisplacements = Unflatten<Real3 const>(sol.GetConstSpan());
  outBounds.localShape =
      GetObb(CalcAabbWithSortedIndices(nodeCoordinates, nodeDisplacements, boundaryIndices));
}

// Call once on startup
void InitializeOnce(entt::registry& reg);

// Get the mass of a soft actor.
[[nodiscard]] real GetActorMass(entt::registry const& reg, entt::entity actor);

} // namespace soft
} // namespace mochi
