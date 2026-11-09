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

#include "mochi_simulation.h"

#include "mochi_actor_convergence.h"
#include "mochi_articulated_body.h"
#include "mochi_contact_anchors.h"
#include "mochi_ecs_utils.h"
#include "mochi_integration.h"
#include "mochi_island.h"
#include "mochi_rigid.h"
#include "mochi_soft.h"
#include "mochi_solve.h"

#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/solvers/newton_solver_params.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/time.h>

using namespace mochi;

void mochi::GetIslandNewtonParams(
    entt::registry const& reg,
    entt::entity island,
    NewtonSolverParams& outNewtonParams) {
  MOCHI_PROFILE_SCOPE();
  int const numDofs = reg.get<CIslandDofInfo>(island).dofsSize;
  auto const& descendants = reg.get<CIslandDescendants>(island);
  auto const& simParams = reg.ctx<CSimulationParams const>();

  // Solver type
  outNewtonParams.solverType = simParams.nonLinearSolver.solverType;
  outNewtonParams.dResidualAssemblyPeriod = simParams.nonLinearSolver.dResidualAssemblyPeriod;

  // Stop criteria
  outNewtonParams.maxIter = simParams.nonLinearSolver.maxIter;
  outNewtonParams.maxElapsedTime =
      TimeSpanFromSeconds(simParams.nonLinearSolver.maxElapsedTimeSeconds);
  outNewtonParams.convergenceMode = simParams.nonLinearSolver.convergenceMode;
  outNewtonParams.absTolRes = simParams.nonLinearSolver.absTol;
  outNewtonParams.relTolRes = simParams.nonLinearSolver.relTol;
  outNewtonParams.solRelTol = simParams.nonLinearSolver.relStepTol;
  outNewtonParams.stopIfNoImprovement = simParams.nonLinearSolver.stopIfNoImprovement;

  // Retries
  outNewtonParams.psdProjMode = simParams.nonLinearSolver.psdProjMode;
  outNewtonParams.fittedSaturationHessian = simParams.experimentalEval.fittedSaturationHessian;
  if (!articulated::ReadsJointFrictionHessianChoice(reg, descendants.compoundActors)) {
    // Mark the joint-friction pathway fitted where nothing reads it, so it doesn't trigger
    // Newton retries.
    outNewtonParams.fittedSaturationHessian.jointFriction = true;
  }

  // Explosion control
  outNewtonParams.explosionControl = simParams.nonLinearSolver.explosionControl;
  outNewtonParams.absDivTol = simParams.nonLinearSolver.absDivTol;
  outNewtonParams.relDivTol = simParams.nonLinearSolver.relDivTol;

  // Line search
  outNewtonParams.lineSearch.type = simParams.nonLinearSolver.lineSearchType;
  outNewtonParams.lineSearch.alpha = simParams.nonLinearSolver.lineSearchAlpha;
  outNewtonParams.lineSearch.wolfe1 = simParams.nonLinearSolver.lineSearchWolfe1;
  outNewtonParams.lineSearch.wolfe2 = simParams.nonLinearSolver.lineSearchWolfe2;
  outNewtonParams.lineSearch.maxIter = simParams.nonLinearSolver.lineSearchMaxIter;
  outNewtonParams.lineSearch.maxRelIncrease = simParams.nonLinearSolver.lineSearchMaxRelIncrease;

  // Linear tolerance strategy
  outNewtonParams.linearToleranceStrategy = simParams.nonLinearSolver.linearToleranceStrategy;

  // Other
  outNewtonParams.verbosity = simParams.nonLinearSolver.verbosity;
  outNewtonParams.consistencyResNorm = simParams.experimentalEval.consistencyResNorm;
  outNewtonParams.consistencyResNormStep = simParams.experimentalEval.consistencyResNormStep;

  // Linear solver and preconditioner type
  outNewtonParams.lParams.preconditionerType = simParams.linearSolver.preconditionerType;
  if (simParams.linearSolver.solverType == LinearSolverType::Auto) {
    // NOTE: The heuristic to select the linear solver type has not been tuned yet. The optimal
    // threshold to use a direct solver is likely larger than 50 DoFs.
    static_assert(
        /* Safety check to avoid defaulting to sparse LDLt, which has not been optimized yet.*/
        50 < LinearSolver<real>::kSparseLdltDofThreshold); //
    if (numDofs <= 50) {
      outNewtonParams.lParams.solverType = LinearSolverType::LDLT;
      outNewtonParams.lParams.preconditionerType = PreconditionerType::None;
    } else {
      outNewtonParams.lParams.solverType = LinearSolverType::CG;
    }
  } else {
    outNewtonParams.lParams.solverType = simParams.linearSolver.solverType;
  }

  // Linear solver stop criteria
  outNewtonParams.lParams.normType = simParams.linearSolver.normType;
  if (simParams.linearSolver.maxIter == kAutoLinearSolverMaxIter) {
    // TODO: Tune the criteria to select the maximum number of iterations.
    outNewtonParams.lParams.maxIter = Clamp(numDofs, 1, kDefaultLinearSolverMaxIter);
  } else {
    outNewtonParams.lParams.maxIter = simParams.linearSolver.maxIter;
  }
  MOCHI_ASSERT(
      outNewtonParams.lParams.maxIter > 0,
      "Maximum number of iterations for iterative linear solvers must be positive.");
  outNewtonParams.lParams.absTol = simParams.linearSolver.absTol;
  outNewtonParams.lParams.relTol = simParams.linearSolver.relTol;
  outNewtonParams.lParams.relDivTol = simParams.linearSolver.relDivTol;
  outNewtonParams.lParams.restartSize = simParams.linearSolver.restartSize;
  outNewtonParams.lParams.abortIfNotSpd = simParams.linearSolver.abortIfNotSpd;
  outNewtonParams.lParams.verbosity = simParams.linearSolver.verbosity;
}

void mochi::SetDirichletBCs(
    CDirichletBC<real> const& dirichlet,
    ColumnVectorView<real> outSolution) {
  MOCHI_ASSERT_VERBOSE(isize(dirichlet.poseIndices) == isize(dirichlet.poseValues));
  for (int i = 0; i < isize(dirichlet.poseIndices); ++i) {
    MOCHI_ASSERT_VERBOSE(dirichlet.poseIndices[i] < outSolution.Rows());
    outSolution[dirichlet.poseIndices[i]] = dirichlet.poseValues[i];
  }
}

void mochi::InvalidateActorStepHistory(entt::registry& reg, entt::entity actor) {
  // For multi-step integrators, clear previous step data.
  integration::ClearMultiStepIntegrationData(reg, actor);

  // The next conservative-step-bounds prediction cannot trust pre-reset state/history. Enable its
  // one-step relaxation.
  RelaxConservativeStepBoundsOnNextStep(reg, actor);

  // The overwritten state is no longer the output of the last solve.
  if (auto* convergence = reg.try_get<CConvergenceStatus>(actor)) {
    convergence->stageStatus = ConvergenceStatus::None;
    convergence->stepStatus = ConvergenceStatus::None;
  }

  // Clear friction preload (anchors) on the next step.
  RequestFrictionAnchorClear(reg, actor);
}

namespace mochi::simulation {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CActorConvergenceWeights>(reg);
  ecs::RegisterComponent<CActorDofInfo>(reg);
  ecs::RegisterComponent<CDirichletBC<real>>(reg);
  ecs::RegisterComponent<CDofOffset>(reg);
  ecs::RegisterComponent<CDofPositionsBC>(reg);
  ecs::RegisterComponent<CExternalForces>(reg);
  ecs::RegisterComponent<CMassMatrix>(reg);
  ecs::RegisterComponent<CLumpedMassMatrix>(reg);
  ecs::RegisterComponent<CSimulationParams>(reg);
  ecs::RegisterComponent<TagGlobalDofsChanged>(reg);
  ecs::RegisterComponent<TagUseContact>(reg);
  ecs::RegisterComponent<TagUseGravity>(reg);
  ecs::RegisterComponent<TagUseInertia>(reg);
  ecs::RegisterComponent<TagUseNewtonEulerInertia>(reg);
  ecs::RegisterComponent<TagUseStress>(reg);
}
} // namespace mochi::simulation
