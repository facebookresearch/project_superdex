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

#include "mochi_shell.h"

#include "mochi_contact.h"
#include "mochi_core/contact/dmap.h"
#include "mochi_core/element_operations/element_assembler.h"
#include "mochi_core/utils/quaternion_utils.h"
#include "mochi_deformable.h"
#include "mochi_integration.h"

#include <optional>
#include <type_traits>

using namespace mochi;

// Call once on startup
namespace mochi::shell {
void InitializeOnce(entt::registry& /*reg*/) {}

real GetActorMass(entt::registry const& reg, entt::entity actor) {
  MOCHI_ASSERT(reg.all_of<TagShellActor>(actor), "Expected shell actor.");
  auto const& material = reg.get<CShellMaterialParams>(actor);
  auto const& mesh = reg.get<CTriangularMesh>(actor);
  real const area = mesh.mesh->GetTotalMeasure();
  return material.density * area;
}
} // namespace mochi::shell

experimental::ShellMaterialParams mochi::experimental::ShellMaterialParamsFrom3dIsotropic(
    real youngsModulus3d,
    real poissonsRatio3d,
    real density3d,
    real thickness,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(youngsModulus3d) && (youngsModulus3d > 0_r),
      error,
      "Invalid shell material params (3D isotropic inputs): youngsModulus3d must be finite and "
      "positive.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(poissonsRatio3d) && (poissonsRatio3d > -1_r) && (poissonsRatio3d < 0.5_r),
      error,
      "Invalid shell material params (3D isotropic inputs): poissonsRatio3d must be finite and in "
      "(-1, 0.5).");
  MOCHI_ERROR_IF_NOT(
      IsFinite(density3d) && (density3d > 0_r),
      error,
      "Invalid shell material params (3D isotropic inputs): density3d must be finite and "
      "positive.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(thickness) && (thickness > 0_r),
      error,
      "Invalid shell material params (3D isotropic inputs): thickness must be finite and "
      "positive.");
  MOCHI_ERROR_RETURN(error, {});

  // Compute bending stiffness factors from plate bending theory.
  real const bendingFactor =
      (Pow(thickness, 3_r) / 12_r) * youngsModulus3d / (1_r - poissonsRatio3d * poissonsRatio3d);
  // Membrane Lamé parameters from the plane-stress reduction of 3D isotropic linear elasticity,
  // integrated through the thickness (units: [Pa*m] = [N/m]).
  real const membraneMu = thickness * youngsModulus3d / (2_r * (1_r + poissonsRatio3d));
  real const membraneLambda =
      thickness * youngsModulus3d * poissonsRatio3d / (1_r - Sqr(poissonsRatio3d));
  return experimental::ShellMaterialParams{
      .membraneLambda = membraneLambda,
      .membraneMu = membraneMu,
      .bendingAlpha = bendingFactor * poissonsRatio3d,
      .bendingBeta = bendingFactor * (1_r - poissonsRatio3d),
      .density = thickness * density3d};
}

void mochi::shell::ValidateShellMaterialParams(
    experimental::ShellMaterialParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.membraneLambda) && IsFinite(params.membraneMu) &&
          IsFinite(params.bendingAlpha) && IsFinite(params.bendingBeta) &&
          IsFinite(params.density) && IsFinite(params.massDampingCoefficient) &&
          IsFinite(params.stiffnessDampingCoefficient) &&
          IsFinite(params.membraneCollapseBarrierScale),
      error,
      "Invalid shell material params: All fields must be finite.");
  MOCHI_ERROR_IF_NOT(
      params.membraneMu > 0_r,
      error,
      "Invalid shell material params: membraneMu must be positive.");
  MOCHI_ERROR_IF_NOT(
      params.membraneLambda > -params.membraneMu,
      error,
      "Invalid shell material params: membraneLambda must be greater than -membraneMu "
      "(positive 2D bulk modulus membraneLambda + membraneMu).");
  MOCHI_ERROR_IF_NOT(
      params.bendingBeta > 0_r,
      error,
      "Invalid shell material params: bendingBeta must be positive.");
  MOCHI_ERROR_IF_NOT(
      params.bendingAlpha > -params.bendingBeta / 2_r,
      error,
      "Invalid shell material params: bendingAlpha must be greater than -bendingBeta / 2.");
  MOCHI_ERROR_IF_NOT(
      params.density > 0_r, error, "Invalid shell material params: density must be positive.");
  MOCHI_ERROR_IF_NOT(
      params.massDampingCoefficient >= 0_r,
      error,
      "Invalid shell material params: massDampingCoefficient must be non-negative.");
  MOCHI_ERROR_IF_NOT(
      params.stiffnessDampingCoefficient >= 0_r,
      error,
      "Invalid shell material params: stiffnessDampingCoefficient must be non-negative.");
  MOCHI_ERROR_IF_NOT(
      params.membraneCollapseBarrierScale >= 0_r,
      error,
      "Invalid shell material params: membraneCollapseBarrierScale must be non-negative.");
  MOCHI_ERROR_RETURN(error);
}

void mochi::shell::EntityIncrementStep(
    ecs::Included<TagShellActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CVelocitySlice<real, TimeStep::Current>& currVel,
    CDisplacementSlice<real, TimeStep::Previous>& prevDispl,
    CVelocitySlice<real, TimeStep::Previous>& prevVel) {
  prevDispl.CopyFrom(currDispl); // Copy previous displacement
  prevVel.CopyFrom(currVel); // Copy previous velocity
  currVel.value.SetZero(); // Reset current velocity to zero
}

void mochi::shell::RecenterSolutionUsingPivotNode(
    ecs::Included<TagShellActor>,
    CRecenteringParams const& params,
    CRecenteringPivotNode const& pivot,
    CRootTransform& root,
    CDisplacementSlice<real, TimeStep::Current>& currDispl,
    CDisplacementSlice<real, TimeStep::Previous>& prevDispl,
    CIntegrationDisplacementSlices& intDispls,
    CBoundingVolume* bounds) {
  MOCHI_PROFILE_SCOPE();
  std::optional<Real3> const shift = deformable::TranslateRootForRecentering(
      params, Unflatten<Real3 const>(currDispl.value.GetConstSpan())[pivot.index], root, bounds);
  if (!shift) {
    return;
  }
  currDispl.ApplyTranslation(*shift);
  prevDispl.ApplyTranslation(*shift);
  for (auto& prev : intDispls.prevSteps) {
    prev.ApplyTranslation(*shift);
  }
}

void mochi::shell::AddExternalForces(
    CExternalForces const& externalForces,
    TransformRT const& worldFromLocal,
    ColumnVectorView<real const> displacements,
    ColumnVectorView<real const> stageStartDisplacements,
    double* outObj,
    ColumnVectorView<real>* outRes) {
  if (!outObj && !outRes) {
    return;
  }
  constexpr int kNumFields = 3;
  int const numForces = isize(externalForces.dofs);
  VMatrix3x3r const worldFromLocalR = ToVMatrix3x3(worldFromLocal.GetRotation());
  for (int i = 0; i < numForces; ++i) {
    int const dof = externalForces.dofs[i];
    int const component = dof % kNumFields;
    int const nodeStartIndex = dof - component;
    i += deformable::details::AddTranslationalExternalForceEntry(
        externalForces,
        i,
        component,
        nodeStartIndex,
        worldFromLocalR,
        displacements,
        stageStartDisplacements,
        outObj,
        outRes);
  }
}

void mochi::shell::AssembleBody(
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
    CActorSnle& outActorSnle) {
  bool const hasGravity = hasGravityTag;

  // Inertia vanishes in quasistatic steps, whose time step is infinite.
  bool const hasInertia = IsFinite(intState.dtStage);

  bool const hasMassDamping = materialParams.massDampingCoefficient > 0_r;
  bool const hasStiffnessDamping = materialParams.stiffnessDampingCoefficient > 0_r;
  real const massDampingScale =
      hasMassDamping ? materialParams.massDampingCoefficient / intState.dtStage : 0_r;
  real const stiffnessDampingFactor =
      hasStiffnessDamping ? materialParams.stiffnessDampingCoefficient / intState.dtStage : 0_r;

  // Clear SNLE data
  if (params.assemObj) {
    outActorSnle.objective = 0.0;
  }
  if (params.assemRes) {
    outActorSnle.fullResidual.SetZero();
  }
  if (params.assemDRes) {
    MOCHI_PROFILE_SCOPE_N("InitShellDResidual");
    auto const numValues = GetNumValues(outActorSnle.fullDResidual);
    constexpr int kMinValuesPerTask = 150000;
    ParallelForRange(
        "InitShellDResidual", 0, numValues, kMinValuesPerTask, INT_MAX, [&](int rBegin, int rEnd) {
          MOCHI_ASSERT_VERBOSE(rBegin >= 0 && rBegin <= rEnd && rEnd <= numValues);
          AsView(GetValues(outActorSnle.fullDResidual)).MiddleRows(rBegin, rEnd - rBegin).SetZero();
        });
  }

  if (params.assemObj || params.assemRes || params.assemDRes) {
    auto const gravity =
        ToReal3(rootTransform.worldFromLocal.TransformDirectionInverse(sceneGravity->accel));

    // The elements gather the predicted target u₀ + dt·v₀, computed once per assembly.
    ColumnVector<real> predTarget;
    if (hasInertia && (params.assemObj || params.assemRes)) {
      predTarget = stageStartDispl.value + intState.dtStage * stageStartVel.value;
    }

    auto bodyOp = MakeBatchedBodyOp(
        l2g,
        MakeConstSpan(femSurfDisc.femElements),
        hasGravity,
        materialParams.membraneLambda,
        materialParams.membraneMu,
        materialParams.bendingAlpha,
        materialParams.bendingBeta,
        2_r * materialParams.membraneCollapseBarrierScale * materialParams.membraneMu,
        gravity,
        materialParams.density,
        predTarget.GetConstSpan(),
        stageStartDispl.value.GetConstSpan(),
        intState.dtStage,
        massDampingScale,
        stiffnessDampingFactor);

    AssembleObjResDRes<ShellStencilElement>(
        l2g,
        nbs,
        bodyOp,
        currDispl.value,
        AssemblyResults<real>{
            .outObj = &outActorSnle.objective,
            .outRes = AsView(outActorSnle.fullResidual),
            .outDRes = AsView(outActorSnle.fullDResidual),
            .params = params});
  }

  if (!externalForces.Empty()) {
    ColumnVectorView<real> resView = AsView(outActorSnle.fullResidual);
    AddExternalForces(
        externalForces,
        rootTransform.worldFromLocal,
        AsConstView(currDispl.value),
        AsConstView(stageStartDispl.value),
        params.assemObj ? &outActorSnle.objective : nullptr,
        params.assemRes ? &resView : nullptr);
  }
}

void mochi::shell::EntityReconstructPreviousStep(
    ecs::Included<TagShellActor>,
    ecs::CtxGlobal<CSceneTime const> time,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CDisplacementSlice<real, TimeStep::Current> const& currDispl,
    CVelocitySlice<real, TimeStep::Current> const& currVel,
    CIntegrationDisplacementSlices const& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default> const& intVels,
    CDisplacementSlice<real, TimeStep::Previous>& outPrevDispl,
    CVelocitySlice<real, TimeStep::Previous>& outPrevVel) {
  // Displacements and velocities are both differential variables.
  auto const method = simParams->integrationMethod;
  integration::ReconstructPreviousPoseFromHistoryOrExtrapolate(
      method, time.value, intDispls, currDispl, currVel, outPrevDispl);
  integration::ReconstructPreviousVelocityFromHistoryOrCopy(method, intVels, currVel, outPrevVel);
}

void mochi::shell::EntityPreFirstStage(
    ecs::Included<TagShellActor>,
    ecs::OptionalCtxGlobal<TagAssemblyPreserveHistory const> preserveHistory,
    CTimeIntegratorState const& intState,
    CDisplacementSlice<real, TimeStep::Previous> const& prevDispl,
    CVelocitySlice<real, TimeStep::Previous> const& prevVel,
    CIntegrationDisplacementSlices& intDispls,
    CIntegrationVelocitySlices<DisplacementLayer::Default>& intVels) {
  // Displacement and velocity are differential variables. Compute their values at the beginning
  // of the step using integration utilities.
  integration::ApplyTimeIntegrationStepStart(
      !preserveHistory, intState, intDispls, prevDispl, intDispls.stepStart);
  integration::ApplyTimeIntegrationStepStart(
      !preserveHistory, intState, intVels, prevVel, intVels.stepStart);
}

void mochi::shell::EntityPreStage(
    ecs::Included<TagShellActor>,
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

void mochi::shell::EntityPostStage(
    ecs::Included<TagShellActor>,
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

void mochi::shell::EntityPostLastStage(
    ecs::Included<TagShellActor>,
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

void mochi::shell::EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagShellActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CDisplacementSlice<real, TimeStep::Current>& currSol) {
  MOCHI_PROFILE_SCOPE();
  currSol.value = solution.MiddleRows(dofOffset.poseOffset, actorDofInfo.poseSize);
}

void mochi::shell::EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::Included<TagShellActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& actorDofInfo,
    CDisplacementSlice<real, TimeStep::Current>& currSol) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(actorDofInfo.dofsSize == actorDofInfo.poseSize, "Unexpected DoF info");
  currSol.value = reference.MiddleRows(dofOffset.poseOffset, actorDofInfo.poseSize) +
      increment.MiddleRows(dofOffset.dofsOffset, actorDofInfo.dofsSize);
}

void mochi::shell::EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::Included<TagShellActor>,
    CDisplacementSlice<real, TimeStep::Current> const& currSol) {
  MOCHI_PROFILE_SCOPE();
  outSolution = currSol.value; // copy values
}

void mochi::shell::EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::Included<TagShellActor>,
    CDisplacementSlice<real, TimeStep::Current>& currSol) {
  MOCHI_PROFILE_SCOPE();
  currSol.value = solution; // copy values
}

void mochi::shell::AssembleAsyncContact(
    AssemblyParams const& params,
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
    CActorSnle& outActorSnle) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(params.assemObj || params.assemRes || params.assemDRes, "Must assemble something");

  if (collisions.empty()) {
    // No contacts.
    outResponse.Clear();
    return;
  }

  deformable::ComputeAsyncContactResponse<CFemSurfaceDiscretization, kSpaceDim3>(
      reg,
      e,
      simParams->experimentalEval,
      queryActiveContacts,
      femDisc,
      samples,
      colliderInfo,
      collisions,
      intState,
      rootTransform,
      params,
      outResponse);

  if (outResponse.Empty()) {
    // No contacts.
    return;
  }

  AssemblyResults<real> results{
      .outObj = &outActorSnle.objective,
      .outRes = AsView(outActorSnle.fullResidual),
      .outDRes = AsView(outActorSnle.fullDResidual),
      .params = params};

  femDisc.Visit([&](auto const& disc) {
    using DiscT = std::decay_t<decltype(disc)>;
    using ElementT = typename DiscT::ElementT;

    AssemblyActiveSubset const activeSubset = outResponse.ViewActiveContactElementSubset();

    auto boundaryOp = deformable::MakeBatchedBoundaryOp(
        MakeConstSpan(disc.femElements), outResponse, Span<real const>{});

    AssembleObjResDRes<ElementT>(contactL2g, contactNbs, boundaryOp, results, activeSubset);
  });
}
