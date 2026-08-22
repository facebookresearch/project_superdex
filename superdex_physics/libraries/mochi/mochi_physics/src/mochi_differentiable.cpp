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

#include "mochi_differentiable.h"

#include "mochi_actor_convergence.h"
#include "mochi_articulated_body.h"
#include "mochi_constraint.h"
#include "mochi_rigid.h"
#include "mochi_simulation.h"
#include "mochi_solve.h"
#include "mochi_step.h"

#include <mochi_physics/diffsim/mochi_diffsim_types.h>

#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/solvers/newton_solver.h>
#include <mochi_core/utils/assembly_params.h>
#include <mochi_core/utils/task_scheduler.h>

#include <limits>
#include <unordered_set>

using namespace mochi;

namespace {
// Traits to map GradTarget to container, offset and size components
template <GradTarget kGradTarget>
struct GradTargetTraits {};

#define MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS(kGradTarget, kContainer, kSize, kOffset) \
  template <>                                                                        \
  struct GradTargetTraits<GradTarget::kGradTarget> {                                 \
    using ContainerT = kContainer;                                                   \
    using SizeT = kSize;                                                             \
    using OffsetT = kOffset;                                                         \
  }
// clang-format off
MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS(Current, CDiffContainerState, CActorDofInfo, CDofOffset);
MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS(Previous, CDiffContainerState, CActorDofInfo, CDofOffset);
MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS(PreviousDelta, CDiffContainerDerivedState, CActorDerivedStateInfo, CDerivedStateOffset);
// clang-format on
#undef MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS
} // namespace

template <GradTarget kGradTarget>
static void GetContainer(
    ColumnVectorView<real> outData,
    typename GradTargetTraits<kGradTarget>::SizeT const& size,
    typename GradTargetTraits<kGradTarget>::OffsetT const& offset,
    typename GradTargetTraits<kGradTarget>::ContainerT const& container) {
  outData.MiddleRows(offset.dofsOffset, size.dofsSize) = container;
}

template <GradTarget kGradTarget>
static void SetContainer(
    ColumnVectorView<real const> data,
    typename GradTargetTraits<kGradTarget>::SizeT const& size,
    typename GradTargetTraits<kGradTarget>::OffsetT const& offset,
    typename GradTargetTraits<kGradTarget>::ContainerT& outContainer) {
  outContainer = data.MiddleRows(offset.dofsOffset, size.dofsSize);
}

// Update all target pose gradients: accumulate current, set previous, compute propagated.
static void UpdateTargetPoseGrad(
    ColumnVectorView<real const> currentGrad,
    ColumnVectorView<real const> previousGrad,
    uint64_t stepCounter,
    CActorDiffInputInfo const& size,
    CDiffInputOffset const& offset,
    CTargetOwners const& owner,
    CDiffTargetPoseGrad& outTargetPoseGrad) {
  MOCHI_ASSERT_VERBOSE(owner.oldPoseStep < stepCounter, "Inconsistent step counter");
  MOCHI_ASSERT_VERBOSE(owner.velStep < stepCounter, "Inconsistent step counter");
  MOCHI_ASSERT_VERBOSE(owner.newPoseStep < stepCounter, "Inconsistent step counter");
  if (owner.velStep > owner.oldPoseStep) {
    MOCHI_LOG_WARNING_ONCE(
        "The target velocity of the pose controller was set without setting the target pose. Differentiability gradients may be incorrect.");
  }

  // Accumulate current input gradient (adds propagated from previous step).
  outTargetPoseGrad.current =
      outTargetPoseGrad.propagated + currentGrad.MiddleRows(offset.dofsOffset, size.dofsSize);

  // Set previous input gradient.
  outTargetPoseGrad.previous = previousGrad.MiddleRows(offset.dofsOffset, size.dofsSize);

  // Compute what should propagate to the previous step.
  // Propagate oldPose gradient only if oldPose was inherited (not set at this step).
  if (owner.oldPoseStep < stepCounter - 1) {
    outTargetPoseGrad.propagated = outTargetPoseGrad.previous;
  } else {
    outTargetPoseGrad.propagated.SetZero();
  }
  // Propagate newPose gradient only if newPose was inherited (not set at this step).
  if (owner.newPoseStep < stepCounter - 1) {
    outTargetPoseGrad.propagated += outTargetPoseGrad.current;
  }
}

// Implement d2merit/dtargetdq * vector as a finite difference approximation (0.5 / eps) *
// (dmerit/dtarget(q0 + eps * vector) - dmerit/dtarget(q0 - eps * vector)).
// For kGradTarget = GradTarget::Current, dmerit/dtarget must be transported to q0.
static void GetHessianVectorProduct(
    entt::registry& reg,
    entt::entity island,
    GradTarget gradTarget,
    SnleProblem<real>& problemForward,
    ColumnVectorView<real const> vector,
    ColumnVectorView<real> outHvp) {
  if (IsZero(vector)) {
    outHvp.SetZero();
    return;
  }

  // Stack memory for local vector data (4 x 256 elements)
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 4 * 256 * sizeof(real));

  // Set up the gradient assembly function
  AssemblyParams params = {
      .assemObj = false, .assemRes = true, .assemDRes = false, .gradTarget = gradTarget};
  problemForward.SetAssemblyFunction(
      [&](SnleProblem<real>& problem, AssemblyParams const& /* params */) {
        solver::AssembleIslandPipeline(reg, island, params, problem);
      });

  // Size eps such that each pose component changes by epsFiniteDiff.
  auto const& solverParams = reg.ctx<CBackPropagationSolverParams const>();
  int const numDofs = problemForward.GetDofsSize();
  auto const eps = Sqrt(static_cast<real>(numDofs)) * solverParams.epsFiniteDiff /
      (vector.Norm() + std::numeric_limits<real>::min());
  ColumnVector<real> delta(numDofs, &allocator);

  // Lambda for gradient evaluation
  int const solutionSize = problemForward.GetSolutionSize();
  ColumnVector<real> refSolution(solutionSize, &allocator);
  refSolution = problemForward.solution;
  auto evalGradient = [&](ColumnVectorView<real> outGradientLocal) {
    solver::PostNewIncrementLocalPipeline(reg, island, delta, problemForward.solution);
    problemForward.InvalidateCachedData();
    problemForward.UpdateResidual();
    outGradientLocal = problemForward.GetResidual();
    if (gradTarget == GradTarget::Current) {
      // Gradients computed with Lie derivatives use local parameterizations of 3D rotations. When
      // the target is GradTarget::Current, the gradient is evaluated at slightly offset rotations,
      // hence the local parameterizations are slightly incorrect. The gradient must be transported
      // to account for the correct local parameterization of rotations.
      auto const& descendants = reg.get<CIslandDescendants const>(island);
      ecs::InvokeForEach(
          rigid::TransportGradient,
          reg,
          descendants.rigidActors,
          AsConstView(delta),
          outGradientLocal);
      ecs::InvokeForEach(
          articulated::compound::TransportGradient,
          reg,
          descendants.compoundActors,
          AsConstView(delta),
          outGradientLocal);
    }
    problemForward.solution = refSolution;
  };

  // Approximate Hessian-vector product with central finite differences of gradient
  ColumnVector<real> auxGrad(outHvp.Rows(), &allocator);
  auto evalHvpAtEps = [&](real epsScale, ColumnVectorView<real> outGrad) {
    delta = (epsScale * eps) * vector;
    evalGradient(outGrad);
    delta *= -1_r;
    evalGradient(auxGrad);
    outGrad -= auxGrad;
    outGrad *= (0.5_r / (epsScale * eps));
  };
  evalHvpAtEps(1_r, outHvp);

  // Validate finite-difference robustness by comparing Hvp at different
  // epsilon scales. Always run when validateFiniteDiff is set; record any
  // failure in the per-island stats (aggregated to scene stats via logical
  // AND), and emit a warning only when verbosity admits it.
  if (solverParams.validateFiniteDiff) {
    real const norm = outHvp.Norm() + std::numeric_limits<real>::min();
    real constexpr kFiniteDiffTol = 1e-2_r;
    ColumnVector<real> auxHvp(outHvp.Rows(), &allocator);
    auto& islandBackPropStats = reg.get<CIslandBackPropSolverStats>(island);

    auto checkError = [&](real epsScale) {
      evalHvpAtEps(epsScale, auxHvp);
      auxHvp -= outHvp;
      real const err = auxHvp.Norm() / norm;
      if (err > kFiniteDiffTol) {
        islandBackPropStats.finiteDiffValid = false;
        if (solverParams.verbosity >= VerbosityLevel::Verbose) {
          MOCHI_LOG(
              "Finite diff unstable: err(%f x eps)=%f (tol=%f)", epsScale, err, kFiniteDiffTol);
        }
      }
    };

    checkError(2_r);
    checkError(0.5_r);
  }
}

static void WriteToActorResidual(
    ColumnVectorView<real const> islandResidual,
    SnleProblem<real>& outProblem,
    ecs::Excluded<TagArticulatedLinkActor>,
    CActorConvergenceWeights const& weights,
    CActorDofInfo const& dofInfo,
    CDofOffset const& dofOffset,
    CActorSnle& outSnle) {
  auto& outResidual = outSnle.UseReduced() ? outSnle.reducedResidual : outSnle.fullResidual;
  outResidual = islandResidual.MiddleRows(dofOffset.dofsOffset, dofInfo.dofsSize);
  outProblem.actorResiduals.emplace_back(dofOffset.dofsOffset, &outResidual);
  outProblem.actorConvergenceWeights.emplace_back(dofOffset.dofsOffset, &weights.values);
}

// Use a Krylov solver for the linear problem dres * z = rhs, where dres is approximated.
// The approximate Hessian hat(dres) is used as preconditioner, and dres * v products
// are computed via finite differences in GetHessianVectorProduct.
static void KrylovSolveZ(
    entt::registry& reg,
    entt::entity island,
    SnleProblem<real>& problemForward,
    ColumnVectorView<real const> rhs,
    ColumnVectorView<real> outZ) {
  auto const& islandDofInfo = reg.get<CIslandDofInfo>(island);
  int const numDofs = islandDofInfo.dofsSize;

  // Retrieve backpropagation solver stats component.
  auto& islandBackPropSolverStats = reg.get<CIslandBackPropSolverStats>(island);

  // Handle trivial case
  if (IsZero(rhs)) {
    outZ.SetZero();
    islandBackPropSolverStats.stats = StageSolverStats{};
    return;
  }

  // Get outer and inner solver parameters
  auto const& backpropParams = reg.ctx<CBackPropagationSolverParams const>();
  auto const& simParams = reg.ctx<CSimulationParams const>();
  NewtonSolverParams newtonParamsForward;
  GetIslandNewtonParams(numDofs, simParams, newtonParamsForward);
  KrylovSolverParams& innerLParams = newtonParamsForward.lParams;
  innerLParams.absTol = backpropParams.innerSolverAbsTol;

  // Assemble approximate Hessian hat(dres) for preconditioning
  AssemblyParams paramsDRes = {
      .assemObj = false, .assemRes = false, .assemDRes = true, .psdDRes = true};
  solver::AssembleIslandPipeline(reg, island, paramsDRes, problemForward);

  // Get the approximate Hessian matrix
  auto const& approxHessian = ToMatrix(problemForward.GetDResidual());

  // Create linear solver for the preconditioner (inner solver) using scene's linear solver settings
  auto& preconditionerRecyclingMgr = reg.get<CIslandPreconditioner>(island);
  LinearSolver<real> precLinearSolver(innerLParams, preconditionerRecyclingMgr);

  // Create callable operator for matrix-vector product via finite differences
  // This is used by krylov::Apply which calls A(v, Av) for non-matrix types
  auto hessianOp = [&](ColumnVectorView<real const> in, ColumnVectorView<real> out) {
    GetHessianVectorProduct(reg, island, GradTarget::Current, problemForward, in, out);
  };

  // Create callable preconditioner: solves hat(dres) * z = r using the inner linear solver
  auto precOp = [&](ColumnVectorView<real const> in, ColumnVectorView<real> out) {
    out.SetZero();
    precLinearSolver.Solve(approxHessian, in, out, /*hasOperatorChanged*/ false);
  };

  // Initialize solution
  outZ.SetZero();

  // Try PCG first (faster for SPD systems), fall back to MINRES if non-SPD detected
  // PCG stopping criterion
  krylov::StatusResidualL2<krylov::UsualDot, real> pcgStatusCheck(
      backpropParams.outerSolverRelTol, // relTol: convergence if ||r|| <= relTol * ||rhs||
      backpropParams.outerSolverAbsTol, // absTol: convergence if ||r|| <= absTol
      static_cast<real>(newtonParamsForward.relDivTol));

  auto outerResult = krylov::PCG(
      hessianOp,
      rhs,
      outZ,
      precOp,
      backpropParams.outerSolverMaxIter,
      pcgStatusCheck,
      /*abortIfNotSpd*/ true,
      backpropParams.verbosity);

  // If PCG aborted due to non-SPD, fall back to MINRES
  if (!outerResult.converged && outerResult.numIterDone < backpropParams.outerSolverMaxIter) {
    if (backpropParams.verbosity >= VerbosityLevel::Warning) {
      MOCHI_LOG_WARNING(
          "PCG aborted at iteration %d (likely non-SPD). Falling back to MINRES.",
          outerResult.numIterDone);
    }

    // Reset solution for MINRES.
    // Note: An alternative could be to reuse the PCG solution as a warm start, by solving dres dz =
    // rhs_new, with rhs_new = rhs - dres z_pcg, and then z = z_pcg + dz. However, this could be a
    // bad idea if PCG progressed in a wrong direction.
    outZ.SetZero();

    // MINRES stopping criterion
    krylov::StatusImplicitResidualNorm<real> minresStatusCheck(
        backpropParams.outerSolverRelTol,
        backpropParams.outerSolverAbsTol,
        static_cast<real>(newtonParamsForward.relDivTol));

    outerResult = krylov::MinRes(
        hessianOp,
        rhs,
        outZ,
        precOp,
        backpropParams.outerSolverMaxIter,
        minresStatusCheck,
        backpropParams.verbosity);

    if (backpropParams.verbosity >= VerbosityLevel::Verbose) {
      MOCHI_LOG(
          "MINRES: Finished after %d iterations, final resNorm = %f, converged = %d",
          outerResult.numIterDone,
          outerResult.residualNorm,
          outerResult.converged);
    }

    if (backpropParams.verbosity >= VerbosityLevel::Warning && backpropParams.validateFiniteDiff) {
      // Diagnostic: compare MINRES's implicit residual (tracked internally via Givens rotations,
      // in the P^-1-norm) against the true residual computed from a fresh Hv product. A large
      // gap between them means MINRES's claimed convergence cannot be trusted. Two common
      // causes in this setting:
      //   1. FD noise on H*v corrupts the Lanczos recursion (residual-gap phenomenon). One bad
      //      Hv product can drive the implicit |eta| to a spurious near-zero via cancellations
      //      in the Givens rotations, causing MINRES to exit with a bad iterate.
      //   2. H is rank-deficient and rhs has a non-trivial component in null(H). Then Hz = rhs
      //      has no solution and MINRES wanders along null directions; the implicit norm drops
      //      but the true residual is bounded below by ||rhs_null||.
      MOCHI_FILO_STACK_ALLOCATOR(allocator, 2 * 256 * sizeof(real));
      ColumnVector<real> trueRes(rhs.Rows(), &allocator);
      hessianOp(AsConstView(outZ), AsView(trueRes)); // trueRes = H*outZ
      trueRes -= rhs; // trueRes = H*outZ - rhs (sign doesn't affect L2/P^-1 norms below)
      ColumnVector<real> pInvTrueRes(rhs.Rows(), &allocator);
      precOp(AsConstView(trueRes), AsView(pInvTrueRes));
      real const trueResNormPInv = Sqrt(Max(0_r, trueRes.Dot(pInvTrueRes)));
      real constexpr kRatioThreshold = 1e2_r;
      if (trueResNormPInv > kRatioThreshold * outerResult.residualNorm) {
        MOCHI_LOG_WARNING(
            "MINRES integrity check: implicit resNorm (P^-1) = %e, true resNorm (P^-1) = %e.",
            outerResult.residualNorm,
            static_cast<double>(trueResNormPInv));
      }
    }
  } else if (backpropParams.verbosity >= VerbosityLevel::Verbose) {
    MOCHI_LOG(
        "PCG: Finished after %d iterations, final resNorm = %f, converged = %d",
        outerResult.numIterDone,
        outerResult.residualNorm,
        outerResult.converged);
  }

  // Record statistics
  StageSolverStats stats;
  stats.numIterDone = outerResult.numIterDone;
  stats.resNorm = static_cast<real>(outerResult.residualNorm);
  stats.resNormError = 0_r;
  stats.numLSIterDone = 0;
  islandBackPropSolverStats.stats = stats;
}

// Use a NR solver for the linear problem dres * z = rhs, where dres is approximated.
static void NewtonSolveZ(
    entt::registry& reg,
    entt::entity island,
    SnleProblem<real>& problemForward,
    ColumnVectorView<real const> rhs,
    ColumnVectorView<real> outZ) {
  auto const& islandDofInfo = reg.get<CIslandDofInfo>(island);
  int const numDofs = islandDofInfo.dofsSize;

  // Retrieve backpropagation solver stats component.
  auto& islandBackPropSolverStats = reg.get<CIslandBackPropSolverStats>(island);

  // Configure NR solver
  // ------------------------------------------------------------------------
  auto const& simParams = reg.ctx<CSimulationParams const>();
  NewtonSolverParams newtonParams;
  GetIslandNewtonParams(numDofs, simParams, newtonParams);
  // Apply general back-propagation parameters
  // - Currently, the back-propagation solver uses regular residual-based line search, but in the
  // test examples line-search never acted in practice.
  // - Allow reuse of the dresidual on all Newton iterations.
  newtonParams.dResidualAssemblyPeriod = std::numeric_limits<int>::max();
  // - Disable explosion control for two reasons: (1) The state was generated in a forward solve,
  // where possible explosions were already handled. (2) In the remote chance of an explosion, there
  // is no explosion handling mechanism, as the solver doesn't go through the actors.
  newtonParams.explosionControl = false;
  // Apply user-defined parameters for outer solver (Newton)
  auto const& solverParams = reg.ctx<CBackPropagationSolverParams const>();
  newtonParams.verbosity = solverParams.verbosity;
  newtonParams.maxIter = solverParams.outerSolverMaxIter;
  newtonParams.absTolRes = solverParams.outerSolverAbsTol;
  newtonParams.relTolRes = solverParams.outerSolverRelTol;
  newtonParams.convergenceMode = solverParams.outerSolverConvergenceMode;
  // Apply user-defined parameters for inner solver (linear)
  newtonParams.lParams.absTol = solverParams.innerSolverAbsTol;
  auto& preconditioner = reg.get<CIslandPreconditioner>(island);
  NewtonSolver<real> solver(newtonParams, preconditioner);

  // SNLE Callback Functions
  // -----------------------------------------------------------------------
  SnleProblemFunctions<real> functions;
  functions.onPostNewSolution = [](auto& /*problem*/) {}; // Not currently used.
  functions.onPostNewIncrement = [](auto& problem) { problem.solution += problem.increment; };
  functions.assemble = [&](SnleProblem<real>& problem, AssemblyParams const& params) {
    // dresidual assembly (only once)
    if (params.assemDRes) {
      // Assemble approx dresidual the first time
      AssemblyParams paramsDRes = {
          .assemObj = false,
          .assemRes = false,
          .assemDRes = true,
          .psdDRes = params.psdDRes,
          .fittedSaturationHessian = params.fittedSaturationHessian};
      solver::AssembleIslandPipeline(reg, island, paramsDRes, problem);
    }

    // residual or objective assembly
    if (params.assemObj || params.assemRes) {
      // Stack memory for 100 dofs
      MOCHI_FILO_STACK_ALLOCATOR(allocator, 100 * sizeof(real));

      // Finite-difference approximation of dres * z
      ColumnVector<real> dresTimesZ(problem.GetDofsSize(), &allocator);
      GetHessianVectorProduct(
          reg, island, GradTarget::Current, problemForward, problem.GetSolution(), dresTimesZ);

      if (params.assemObj) {
        // obj = 1/2 * zT * dres * z - zT * rhs
        auto const& z = problem.GetSolution();
        problem.objective = 0.5_r * z.Dot(dresTimesZ) - z.Dot(rhs);
      }

      if (params.assemRes) {
        // res = dres * z - rhs
        ColumnVector<real> res = std::move(dresTimesZ); // Reuse memory
        res -= rhs;

        // Write the actor residuals and convergence weights.
        auto descendants = reg.get<CIslandDescendants const>(island).actors;
        problem.actorResiduals.clear();
        problem.actorConvergenceWeights.clear();
        problem.actorResiduals.reserve(descendants.size());
        problem.actorConvergenceWeights.reserve(descendants.size());
        ecs::InvokeForEach<ecs::policy::AllowMutableExternalParams>(
            WriteToActorResidual, reg, descendants, AsConstView(res), std::ref(problem));
      }
    }
  };

  // SNLE Problem
  // -----------------------------------------------------------------------
  SnleProblem<real> problem(numDofs, numDofs, std::move(functions));
  problem.solution.SetZero();

  // Solve
  auto status = solver.Solve(problem);
  islandBackPropSolverStats.stats = StageSolverStats::FromNewtonSolverStatus(status);
  outZ = problem.GetSolution();
}

static void SetForceContainer(
    ColumnVectorView<real const> data,
    CActorDofInfo const& size,
    CDofOffset const& offset,
    CDiffForceGrad& outContainer) {
  outContainer = data.MiddleRows(offset.dofsOffset, size.dofsSize);
}

static void PrepareBackPropagationIslandAsync(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();

  // The island pre-step operation is also needed for back-propagation
  PreStepIslandAsync(reg, descendants);

  // Define integration settings for this island
  auto const& simParams = reg.ctx<CSimulationParams const>();
  MOCHI_ASSERT(
      simParams.integrationMethod == IntegrationMethod::BackwardEuler,
      "Only Backward Euler is supported")
  auto const integrationParams =
      solver::CreateIslandTimeIntegrationParams(reg, descendants, simParams.integrationMethod);
  MOCHI_ASSERT(integrationParams.numStages == 1);
  solver::SetTimeIntegratorState(reg, descendants.actors, integrationParams, /*iStage*/ 0);

  // General settings and SNLE forward problem for contact assembly
  auto const& islandDofInfo = reg.get<CIslandDofInfo const>(island);
  SnleProblemFunctions<real> functions;
  SnleProblem<real> problem(islandDofInfo.dofsSize, islandDofInfo.poseSize, std::move(functions));

  // Set the stage-start state
  solver::PreFirstStageLocalPipeline(reg, descendants);
  solver::PreStageLocalPipeline(reg, descendants, problem);

  // Run collision detection if some actor has queries enabled
  if (std::any_of(descendants.actors.begin(), descendants.actors.end(), [&](auto const& e) {
        return reg.any_of<CQueryActorContactForces>(e);
      })) {
    // We must visit the full island, to account for cases where the queried actors act as
    // colliders.
    solver::UpdateDerivedStateBeforeAssembly(reg, GradTarget::Current, descendants);
  }
}

void mochi::PrepareBackPropagation(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  TaskSemaphore eachTask;
  reg.view<CIslandDescendants const>().each(
      [&](entt::entity island, CIslandDescendants const& descendants) {
        Schedule(eachTask, "PrepareBackPropagationIslandAsync", [&, island]() {
          PrepareBackPropagationIslandAsync(reg, island, descendants);
        });
      });
  eachTask.Wait();

  ecs::InvokeForEachGlobal(&PrepareContactForceAdjoints, reg);
}

static void BackPropagationSolveIslandAsync(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();

  // General settings and SNLE forward problem for gradient assembly
  auto const& islandDofInfo = reg.get<CIslandDofInfo const>(island);
  int const solutionSize = islandDofInfo.poseSize;
  int const dofsSize = islandDofInfo.dofsSize;
  int const derivedDofsSize = reg.get<CIslandDerivedStateInfo const>(island).dofsSize;
  SnleProblemFunctions<real> functions; // dummy functions
  SnleProblem<real> problemForward(dofsSize, solutionSize, std::move(functions));

  // The state pair and contact data were prepared by PrepareBackPropagation(). Initialize
  // the local solution vector from the prepared current actor states without rerunning preparation.
  GetSolutions(problemForward.solution, reg, descendants.actors, /*baseOffset*/ 0);

  // Stack memory for 3 vectors of max size. The filo stack allocator requires that vectors are
  // deallocated in reverse order, so we allocate all at max size upfront.
  int const maxVecSize = Max(dofsSize, derivedDofsSize);
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 3 * 256 * sizeof(real));
  ColumnVector<real> vec0(maxVecSize, &allocator);
  ColumnVector<real> vec1(maxVecSize, &allocator);
  ColumnVector<real> vec2(maxVecSize, &allocator);

  // 1. Collect rhs from CDiffContainerState
  auto rhs = vec0.TopRows(dofsSize); // use vec0 as rhs
  ecs::InvokeForEach<ecs::policy::AllowMutableExternalParams>(
      &GetContainer<GradTarget::Current>, reg, descendants.actors, rhs);

  // 2. Solve linear problem
  rhs *= -1_r;
  auto zSolve = vec1.TopRows(dofsSize); // use vec1 as zSolve
  auto const& solverParams = reg.ctx<CBackPropagationSolverParams const>();
  if (solverParams.useNewtonOuterSolver) {
    // Use Newton-Raphson outer solver
    NewtonSolveZ(reg, island, problemForward, rhs, zSolve);
  } else {
    // Use Krylov outer solver (PCG with MINRES fallback)
    KrylovSolveZ(reg, island, problemForward, rhs, zSolve);
  }

  // 3. Update actor gradient containers.
  auto forceGrad = vec0.TopRows(dofsSize); // use vec0 as grad; rhs is no longer needed
  forceGrad = -1_r * zSolve;
  ecs::InvokeForEach(&SetForceContainer, reg, descendants.actors, AsConstView(forceGrad));
  auto oldStateGrad =
      vec0.TopRows(dofsSize); // use vec0 as oldStateGrad; forceGrad is no longer needed
  GetHessianVectorProduct(reg, island, GradTarget::Previous, problemForward, zSolve, oldStateGrad);
  ecs::InvokeForEach(
      &SetContainer<GradTarget::Previous>, reg, descendants.actors, AsConstView(oldStateGrad));
  auto derivedStepGrad = vec0.TopRows(
      derivedDofsSize); // use vec0 as derivedStepGrad; oldStateGrad is no longer needed
  GetHessianVectorProduct(
      reg, island, GradTarget::PreviousDelta, problemForward, zSolve, derivedStepGrad);
  ecs::InvokeForEach(
      &SetContainer<GradTarget::PreviousDelta>,
      reg,
      descendants.actors,
      AsConstView(derivedStepGrad));

  // 4. Update input gradient containers if needed
  int const diffInputSize = reg.get<CIslandDiffInputInfo const>(island).size;
  if (diffInputSize > 0) {
    auto currentGrad =
        vec0.TopRows(diffInputSize); // use vec0 as currentGrad; derivedStepGrad is no longer needed
    auto previousGrad = vec2.TopRows(diffInputSize); // use vec2 as previousGrad
    GetHessianVectorProduct(
        reg, island, GradTarget::CurrentInput, problemForward, zSolve, currentGrad);
    GetHessianVectorProduct(
        reg, island, GradTarget::PreviousInput, problemForward, zSolve, previousGrad);
    auto const stepCounter = reg.ctx<CSceneStepCounter const>().value;
    ecs::InvokeForEach(
        UpdateTargetPoseGrad,
        reg,
        descendants.actors,
        AsConstView(currentGrad),
        AsConstView(previousGrad),
        stepCounter);
  }
}

void mochi::BackPropagationSolve(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  // Emplace CIslandBackPropSolverStats for all islands.
  reg.view<TagIsland>().each(
      [&](entt::entity island) { reg.emplace_or_replace<CIslandBackPropSolverStats>(island); });

  TaskSemaphore eachTask;
  reg.view<CIslandDescendants const>().each(
      [&](entt::entity island, CIslandDescendants const& descendants) {
        Schedule(eachTask, "BackPropagateIsland", [&, island]() {
          BackPropagationSolveIslandAsync(reg, island, descendants);
        });
      });

  eachTask.Wait();
}

// Assign/Acquire to global Jacobian matrices, using CSceneStateOffset to determine block offset
static void AssignDResBlk(
    entt::registry& reg,
    Span<entt::entity const> actors,
    MatrixView<real> dRes,
    MatrixView<real const> dResBlk) {
  for (auto const& r : actors) {
    int rOffGlobal = reg.get<CSceneStateOffset const>(r).dofsOffset;
    int rOff = reg.get<CDofOffset const>(r).dofsOffset;
    int rDofs = reg.get<CActorDofInfo const>(r).dofsSize;
    for (auto const& c : actors) {
      int cOffGlobal = reg.get<CSceneStateOffset const>(c).dofsOffset;
      int cOff = reg.get<CDofOffset const>(c).dofsOffset;
      int cDofs = reg.get<CActorDofInfo const>(c).dofsSize;
      dRes.Block(rOffGlobal, cOffGlobal, rDofs, cDofs) = dResBlk.Block(rOff, cOff, rDofs, cDofs);
    }
  }
}

static void AcquireDResBlk(
    entt::registry& reg,
    Span<entt::entity const> actors,
    MatrixView<real const> dRes,
    MatrixView<real> dResBlk) {
  for (auto const& r : actors) {
    int rOffGlobal = reg.get<CSceneStateOffset const>(r).dofsOffset;
    int rOff = reg.get<CDofOffset const>(r).dofsOffset;
    int rDofs = reg.get<CActorDofInfo const>(r).dofsSize;
    for (auto const& c : actors) {
      int cOffGlobal = reg.get<CSceneStateOffset const>(c).dofsOffset;
      int cOff = reg.get<CDofOffset const>(c).dofsOffset;
      int cDofs = reg.get<CActorDofInfo const>(c).dofsSize;
      dResBlk.Block(rOff, cOff, rDofs, cDofs) = dRes.Block(rOffGlobal, cOffGlobal, rDofs, cDofs);
    }
  }
}

void mochi::ComputeHqx(
    int numIslandDofs,
    Span<entt::entity const> actors,
    CActorSnle const& actorSnle,
    CDofOffset const& dofOffset,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac& outDerivedState) {
  int cols = derivedStateInfo.dofsSize;
  outDerivedState.actors = actors;
  outDerivedState.numIslandDofs = numIslandDofs;
  outDerivedState.data.Resize(numIslandDofs, cols);

  // Fill the mixed Hessian H_{qx} = [df/dδ]
  // Note that [df/dδ] is a block diagonal matrix because this term is due to inertial term.
  // This is why we can store the corresponding columns of the derived state matrix in the per-actor
  // struct: CForwardPropContainerDerivedStateJac
  outDerivedState.data.SetZero();
  outDerivedState.data.MiddleRows(dofOffset.dofsOffset, derivedStateInfo.dofsSize) =
      actorSnle.UseReduced() ? ToMatrix(AsConstView(actorSnle.reducedDResidual))
                             : ToMatrix(AsConstView(actorSnle.fullDResidual));
}

void mochi::ComputeDqDDerivedState(
    int numIslandDofs,
    LU<real> const& invDRes,
    Span<entt::entity const> actors,
    CActorSnle const& actorSnle,
    CDofOffset const& dofOffset,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac& outDerivedState) {
  // Compute H_{qx} = [df/dδ]
  ComputeHqx(numIslandDofs, actors, actorSnle, dofOffset, derivedStateInfo, outDerivedState);

  // Solve for dqk/dδ = -[df/dqk]^{-1} * H_{qx}
  outDerivedState.data *= -1_r;
  invDRes.LeftSolveInPlace(outDerivedState.data);
}

static void ShiftDqDDerivedState(
    int numIslandDofs,
    MatrixView<real> outJacCurr,
    MatrixView<real> outJacOld,
    entt::entity e,
    entt::registry& reg,
    CDofOffset const& dofOffset,
    CActorDofInfo const& dofInfo,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac const& derivedState,
    CDiffContainerDerivedState& outDerivedState) {
  int cols = derivedStateInfo.dofsSize;
  MatrixView<real const> derivedStateMat(derivedState.data.data(), numIslandDofs, cols);

  auto jacCurrActor = outJacCurr.MiddleCols(dofOffset.dofsOffset, dofInfo.dofsSize);
  auto jacOldActor = outJacOld.MiddleCols(dofOffset.dofsOffset, dofInfo.dofsSize);
  // The data structure CForwardPropContainerDerivedStateJac stores the sub-matrix dqk/dδ of size:
  //   |numIslandDofs| x |#derived state of the actor|
  // The matrix is computed via:
  //   dqk/dδ = -[df/dqk]^{-1} [df/dδ]
  for (int row = 0; row < numIslandDofs; row++) {
    // CDiffContainerDerivedState = dqk_row/dδ
    outDerivedState = derivedStateMat.Row(row).Transpose();
    // CDiffContainerState = dqk_row/dq_k-1 = dqk_row/dδ * dδ/dq_k-1
    ecs::TryInvokeOnEntity(rigid::ProjectDerivedStateGradient, reg, e);
    ecs::TryInvokeOnEntity(rigid::ShiftDerivedStateGradient, reg, e);
    // dqk_row/dq_k-1 = dqk_row/dq_k-1 + dqk_row/dδ * dδ/dq_k-1
    jacCurrActor.Row(row) += reg.get<CDiffContainerState const>(e).Transpose();
    // CDiffContainerState = dqk_row/dq_k-2 = dqk_row/dδ * dδ/dq_k-2
    // In the case of rigid body, this is just an assignment
    ecs::TryInvokeOnEntity(rigid::ProjectDerivedStateGradient, reg, e);
    // dqk_row/dq_k-2 = dqk_row/dδ * dδ/dq_k-2
    jacOldActor.Row(row) = reg.get<CDiffContainerState const>(e).Transpose();
  }
}

static void StepJacobianSolveIslandAsync(
    entt::registry& reg,
    entt::entity island,
    CIslandMembers const& members,
    CIslandDescendants const& descendants,
    MatrixView<real> outJacCurr) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(!descendants.actors.empty(), "Empty islands should have been pruned");

  // General settings and SNLE forward problem for mixed Hessian assembly
  auto const& islandDofInfo = reg.get<CIslandDofInfo>(island);
  int const solutionSize = islandDofInfo.poseSize;
  int const dofsSize = islandDofInfo.dofsSize;

  // Initialize island assemble problem
  SnleProblemFunctions<real> functions;
  functions.onPostNewSolution = [](auto& /*problem*/) {}; // Not currently used.
  functions.onPostNewIncrement = [](auto& /*problem*/) {}; // Not currently used.
  functions.assemble = [&](SnleProblem<real>& problem, AssemblyParams const& params) {
    solver::AssembleIslandPipeline(reg, island, params, problem);
  };
  SnleProblem<real> problem(dofsSize, solutionSize, std::move(functions));

  // Set the stage-start state
  solver::PreFirstStageLocalPipeline(reg, descendants);
  solver::PreStageLocalPipeline(reg, descendants, problem);

  // Assemble problem with psdDRes=false, notifying exact Hessian
  AssemblyParams paramsExactDRes = {
      .assemObj = false, .assemRes = false, .assemDRes = true, .psdDRes = false};
  problem.UpdateObjResDRes(paramsExactDRes);

  // Direct factorization using LU
  LU<real> invDRes(ToMatrix(problem.GetDResidual()));

  // Derivative with respect to previous state
  paramsExactDRes.gradTarget = GradTarget::Previous;
  problem.SetAssemblyFunction([&](SnleProblem<real>& problem, AssemblyParams const& params) {
    mochi::solver::AssembleIslandPipeline(reg, island, params, problem);
  });
  problem.InvalidateCachedData();
  problem.UpdateObjResDRes(paramsExactDRes);
  // dqk/dqk-1 = -[d2f/dqk2]^{-1} * d2f/dqk/dqk-1
  auto jacCurrBlk = ToMatrix(problem.GetDResidual());
  jacCurrBlk *= -1_r;
  invDRes.LeftSolveInPlace(jacCurrBlk);
  AssignDResBlk(reg, members.actors, outJacCurr, jacCurrBlk);

  // Derivative with respect to delta
  paramsExactDRes.gradTarget = GradTarget::PreviousDelta;
  problem.SetAssemblyFunction([&](SnleProblem<real>& problem, AssemblyParams const& params) {
    mochi::solver::AssembleIslandPipeline(reg, island, params, problem);
  });
  problem.InvalidateCachedData();
  problem.UpdateObjResDRes(paramsExactDRes);
  // dqk/dδ = -[d2f/dqk2]^{-1} * d2f/dqk/dδ
  ecs::InvokeForEach(
      ComputeDqDDerivedState,
      reg,
      members.actors,
      dofsSize,
      std::cref(invDRes),
      MakeConstSpan(members.actors));
}

static void StepJacobianShiftActor(
    MatrixView<real> jacCurr,
    MatrixView<real> jacOld,
    entt::registry& reg,
    CForwardPropContainerDerivedStateJac const& derivedState) {
  MOCHI_PROFILE_SCOPE();

  // Initialize jacobian blocks to zero
  Matrix<real> jacCurrBlk(derivedState.numIslandDofs, derivedState.numIslandDofs);
  Matrix<real> jacOldBlk(derivedState.numIslandDofs, derivedState.numIslandDofs);
  AcquireDResBlk(reg, derivedState.actors, jacCurr, jacCurrBlk);

  // Assuming that we have computed CForwardPropContainerDerivedStateJac, which stores dqk/dδ
  // The following call propagates Jacobian to compute:
  //   dqk/dqk-1 += dqk/dδ * dδ/dqk-1
  //   dqk/dqk-2  = dqk/dδ * dδ/dqk-2
  // This is same computation as BackPropagation, but applied row-by-row, i.e.:
  // We extract dqk_i/dδ and store this vector into CDiffContainerDerivedState
  // We can then use the backpropagation functionality to compute:
  //   dqk_i/dqk-1 += dqk_i/dδ * dδ/dqk-1
  //   dqk_i/dqk-2  = dqk_i/dδ * dδ/dqk-2
  ecs::InvokeForEach<ecs::policy::AllowFullRegistryAccess>(
      ShiftDqDDerivedState,
      reg,
      derivedState.actors,
      derivedState.numIslandDofs,
      AsView(jacCurrBlk),
      AsView(jacOldBlk));

  // assign to scene-wise global jacobian matrix
  AssignDResBlk(reg, derivedState.actors, jacCurr, jacCurrBlk);
  AssignDResBlk(reg, derivedState.actors, jacOld, jacOldBlk);
}

void mochi::StepJacobianSolve(entt::registry& reg, MatrixView<real> jacCurr) {
  MOCHI_PROFILE_SCOPE();

  // Initialize Jacobian matrix to zero
  jacCurr.SetZero();

  reg.view<CIslandMembers const, CIslandDescendants const>().each(
      [&](entt::entity island,
          CIslandMembers const& members,
          CIslandDescendants const& descendants) {
        // The island pre-step operation is also needed for forward-propagation
        PreStepIslandAsync(reg, descendants);
        // Now forward-propagate each island
        StepJacobianSolveIslandAsync(reg, island, members, descendants, jacCurr);
        // The island post-step operation is not needed for forward-propagation
      });
}

void mochi::StepJacobianShiftAndProject(
    entt::registry& reg,
    MatrixView<real> outJacCurr,
    MatrixView<real> outJacOld) {
  MOCHI_PROFILE_SCOPE();
  // Initialize Jacobian matrix to zero
  outJacOld.SetZero();

  // Shift each stateNew island exactly once, keyed on the stored grouping (the actors list each
  // actor recorded in ComputeHqx), not the current CIslandMembers: GetStepJacobian has since
  // restored stateCurr/stateOld, so current islanding can differ from stateNew's. De-duplicate by
  // the island's first member so a split/merged current island can't double- or under-apply it.
  std::unordered_set<entt::id_type> processedIslands;
  reg.view<CForwardPropContainerDerivedStateJac const>().each(
      [&](entt::entity /*e*/, CForwardPropContainerDerivedStateJac const& derivedState) {
        if (derivedState.actors.empty()) {
          return;
        }
        if (processedIslands.insert(static_cast<entt::id_type>(derivedState.actors[0])).second) {
          StepJacobianShiftActor(AsView(outJacCurr), AsView(outJacOld), reg, derivedState);
        }
      });
}

void mochi::EmplaceDifferentiabilityComponents(
    int numDerivedStateDofs,
    entt::registry& reg,
    entt::entity e,
    CActorDofInfo const& dofInfo) {
  // Emplace components for derived-state vector indexing
  auto& derivedStateInfo = reg.emplace_or_replace<CActorDerivedStateInfo>(e);
  derivedStateInfo.dofsSize = numDerivedStateDofs;
  reg.emplace_or_replace<CDerivedStateOffset>(e);

  // Emplace components for differentiable-input vector indexing
  auto& diffInputInfo = reg.emplace_or_replace<CActorDiffInputInfo>(e);
  diffInputInfo.dofsSize = 0;
  reg.emplace_or_replace<CDiffInputOffset>(e);

  // Emplace components to store temporary gradient data during the forward/back-propagation step.
  reg.emplace<CDiffContainerState>(e, dofInfo.dofsSize);
  reg.emplace<CDiffContainerDerivedState>(e, numDerivedStateDofs);
  reg.emplace<CDiffForceGrad>(e, dofInfo.dofsSize);
  reg.emplace<CForwardPropContainerDerivedStateJac>(e);

  // Emplace components to store adjoints per-entity across BackPropagate calls.
  reg.emplace<CDiffStateGrad>(e, dofInfo.dofsSize);
  reg.emplace<CDiffDerivedStepGrad>(e, numDerivedStateDofs);

  // Emplace actor data
  reg.emplace<CSceneStateOffset>(e);
}

void mochi::EmplaceDifferentiableContactComponents(
    entt::registry& reg,
    entt::entity e,
    CActorDofInfo const& dofInfo) {
  // Emplace components to store contact-force adjoints per-entity.
  reg.emplace<CDiffContactGrad<GradTarget::Current>>(e, dofInfo.dofsSize);
  reg.emplace<CDiffContactGrad<GradTarget::Previous>>(e, dofInfo.dofsSize);
}

void mochi::EmplaceConstraintDifferentiabilityComponents(entt::registry& reg, entt::entity e) {
  auto const& info = reg.get<CConstraintInfo const>(e);
  MOCHI_ASSERT(
      GetNumConstrainedTargets(info.type) > 0,
      "A constraint with no target does not require differentiability components");
  MOCHI_ASSERT(
      info.GetNumTargets() == GetNumConstrainedTargets(info.type),
      "This constraint does not support differentiability of targets");
  MOCHI_ASSERT(
      !info.hasMixedLinks,
      "Differentiability not supported for constraints between links of an articulated body "
      "and external actors.");
  reg.emplace<TagConstraintWithDifferentiableInput>(e);
  reg.emplace<CConstraintGlobalInputSparsityCache>(e);
}

void mochi::ResetBackPropagationContainers(
    CDiffStateGrad& outGradState,
    CDiffDerivedStepGrad& outGradDerivedStep,
    CDiffTargetPoseGrad* outTargetPoseGrad) {
  outGradState.value.SetZero();
  outGradDerivedStep.value.SetZero();
  if (outTargetPoseGrad) {
    outTargetPoseGrad->propagated.SetZero();
  }
}

void mochi::PrepareContactForceAdjoints(
    CQueryActorContactForces const& /*queryActorContactForces*/,
    [[maybe_unused]] CRequiresFarSdfEvaluation const* farSdfEval,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& outActiveCollisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current>& outActiveCollisionsSync,
    CCollJacs<CollRole::Collider>* outColliderJacs) {
  MOCHI_ASSERT_VERBOSE(
      !farSdfEval, "Far SDF evaluation is not compatible with contact-force queries.");

  auto prepareAdjoints = [](ContactDetectionResult& collisionResult) {
    auto& adjoints = collisionResult.forcePerUnitArea;
    adjoints.resize_noinit(collisionResult.sampleIndices.size());
    std::fill(adjoints.begin(), adjoints.end(), Real3{});
  };

  for (auto& collision : outActiveCollisionsAsync) {
    prepareAdjoints(collision.collisionResult);
  }
  for (auto& collision : outActiveCollisionsSync) {
    prepareAdjoints(collision.collisionResult);
  }
  if (outColliderJacs) {
    for (auto& jac : *outColliderJacs) {
      prepareAdjoints(*jac.query);
    }
  }
}

namespace mochi::differentiable {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<TagDifferentiableScene>(reg);
  ecs::RegisterComponent<TagBackPropagationPrepared>(reg);
  ecs::RegisterComponent<CStatePair>(reg);
  ecs::RegisterComponent<TagConstraintWithDifferentiableInput>(reg);
  ecs::RegisterComponent<CBackPropagationSolverParams>(reg);
  ecs::RegisterComponent<CBackPropagationSceneStats>(reg);
  ecs::RegisterComponent<CIslandDerivedStateInfo>(reg);
  ecs::RegisterComponent<CActorDerivedStateInfo>(reg);
  ecs::RegisterComponent<CDerivedStateOffset>(reg);
  ecs::RegisterComponent<CIslandDiffInputInfo>(reg);
  ecs::RegisterComponent<CActorDiffInputInfo>(reg);
  ecs::RegisterComponent<CDiffInputOffset>(reg);
  ecs::RegisterComponent<CDiffContainerState>(reg);
  ecs::RegisterComponent<CDiffContainerDerivedState>(reg);
  ecs::RegisterComponent<CForwardPropContainerDerivedStateJac>(reg);
  ecs::RegisterComponent<CSceneStateOffset>(reg);
  ecs::RegisterComponent<CDiffForceGrad>(reg);
  ecs::RegisterComponent<CDiffTargetPoseGrad>(reg);
  ecs::RegisterComponent<CDiffStateGrad>(reg);
  ecs::RegisterComponent<CDiffDerivedStepGrad>(reg);
  ecs::RegisterComponent<CDiffContactGrad<GradTarget::Current>>(reg);
  ecs::RegisterComponent<CDiffContactGrad<GradTarget::Previous>>(reg);
  ecs::RegisterComponent<CIslandBackPropSolverStats>(reg);
}
} // namespace mochi::differentiable
