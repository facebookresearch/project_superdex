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

#include "mochi_solve.h"

#include "mochi_actor_convergence.h"
#include "mochi_articulated_body.h"
#include "mochi_blended.h"
#include "mochi_common_components.h"
#include "mochi_compound.h"
#include "mochi_contact.h"
#include "mochi_contact_filter.h"
#include "mochi_deformable.h"
#include "mochi_group.h"
#include "mochi_island.h"
#include "mochi_point_cloud_contact.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"
#include "mochi_rom_jacobian.h"
#include "mochi_scene_recorder.h"
#include "mochi_shell.h"
#include "mochi_snle.h"
#include "mochi_soft.h"
#include "mochi_soft_rom_systems.h"
#include "mochi_soft_skinned.h"

#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix_utils.h>
#include <mochi_core/solvers/newton_solver.h>
#include <mochi_core/utils/profile.h>

#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

using namespace mochi;

void mochi::GetSolutions(
    ColumnVectorView<real> outSolutions,
    entt::registry& reg,
    Span<entt::entity const> entities,
    int baseOffset) {
  MOCHI_PROFILE_SCOPE();
  for (auto e : entities) {
    if (reg.any_of<TagArticulatedLinkActor>(e)) {
      // Skip articulated links. The articulated actor will get the reduced solution.
      continue;
    }
    auto const* dofOffset = reg.try_get<CDofOffset const>(e);
    auto const* dofInfo = reg.try_get<CActorDofInfo const>(e);
    if (dofOffset && dofInfo) {
      auto outActorSln =
          outSolutions.MiddleRows(dofOffset->poseOffset - baseOffset, dofInfo->poseSize);
      ecs::TryInvokeOnEntity(soft::EntityGetSolution, reg, e, outActorSln);
      ecs::TryInvokeOnEntity(shell::EntityGetSolution, reg, e, outActorSln);
      ecs::TryInvokeOnEntity(rigid::EntityGetSolution, reg, e, outActorSln);
      ecs::TryInvokeOnEntity(rom::EntityGetSolution, reg, e, outActorSln);
      ecs::TryInvokeOnEntity(rod::EntityGetSolution, reg, e, outActorSln);
      ecs::TryInvokeOnEntity(articulated::compound::EntityGetSolution, reg, e, outActorSln);
    }
  }
}

void mochi::SetSolutions(
    entt::registry& reg,
    Span<entt::entity const> entities,
    ColumnVectorView<real const> solution) {
  MOCHI_PROFILE_SCOPE();
  for (auto e : entities) {
    if (reg.any_of<TagArticulatedLinkActor>(e)) {
      // Skip articulated links. The articulated actor will set the reduced solution.
      continue;
    }
    auto const* dofOffset = reg.try_get<CDofOffset const>(e);
    auto const* dofInfo = reg.try_get<CActorDofInfo const>(e);
    if (dofOffset && dofInfo) {
      auto const actorSln = solution.MiddleRows(dofOffset->poseOffset, dofInfo->poseSize);
      ecs::TryInvokeOnEntity(soft::EntitySetSolution, reg, e, actorSln);
      ecs::TryInvokeOnEntity(shell::EntitySetSolution, reg, e, actorSln);
      ecs::TryInvokeOnEntity(rigid::EntitySetSolution<TimeStep::Current>, reg, e, actorSln);
      ecs::TryInvokeOnEntity(rom::EntitySetSolution, reg, e, actorSln);
      ecs::TryInvokeOnEntity(rod::EntitySetSolution, reg, e, actorSln);
      ecs::TryInvokeOnEntity(articulated::compound::EntitySetSolution, reg, e, actorSln);
    }
  }

  // Update articulated rigid actors AFTER articulated::compound::EntitySetSolution
  ecs::InvokeForEach(&articulated::rigid::EntitySetSolution, reg, entities);
}

/**
 * @brief Distribute convergence status to all actors in the island.
 *
 * @note When using @ref NonLinearSolverConvergenceMode::PerActorWeighted, solver-participating
 * actors get their individual convergence status. All other actors (or all actors when using @ref
 * NonLinearSolverConvergenceMode::Global, or when the solver diverged) get the global island
 * convergence status.
 */
static void DistributeConvergenceStatus(
    entt::registry& reg,
    Span<entt::entity const> actors,
    SnleProblem<real> const& problem,
    NewtonSolverStatus<real> const& result) {
  MOCHI_ASSERT(
      result.convergence != ConvergenceStatus::None, "Convergence status has not been set.");

  // Note: Divergence is a global condition. If the solver diverged, all actors receive Diverged
  // status.
  bool const usePerActor =
      !result.actorConvergence.empty() && (result.convergence != ConvergenceStatus::Diverged);

  for (auto e : actors) {
    auto* convergence = reg.try_get<CConvergenceStatus>(e);
    if (!convergence) {
      continue;
    }

    // Default to the global result.
    ConvergenceStatus stageStatus = result.convergence;

    if (usePerActor) {
      // Look up this actor's per-actor convergence. Only solver-participating actors (useInSolver)
      // are available in actorResiduals.
      auto const* dofOffset = reg.try_get<CDofOffset const>(e);
      auto const* snle = reg.try_get<CActorSnle const>(e);
      if (dofOffset && snle && snle->useInSolver) {
        int i = 0;
        for (; i < isize(problem.actorResiduals); ++i) {
          if (problem.actorResiduals[i].first == dofOffset->dofsOffset) {
            stageStatus = result.actorConvergence[i];
            break;
          }
        }
#if MOCHI_ASSERT_VERBOSE_ENABLED
        MOCHI_ASSERT_VERBOSE(
            i < isize(problem.actorResiduals), "Could not find actor in actorResiduals.");
        for (++i; i < isize(problem.actorResiduals); ++i) {
          MOCHI_ASSERT_VERBOSE(
              problem.actorResiduals[i].first != dofOffset->dofsOffset, "Duplicated actor offset.");
        }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
      }
    }

    convergence->stageStatus = stageStatus;
    convergence->stepStatus = Max(convergence->stepStatus, stageStatus);
  }
}

void mochi::solver::PostNewSolutionLocalPipeline(
    entt::registry& reg,
    entt::entity island,
    ColumnVectorView<real> outSolution) {
  MOCHI_PROFILE_SCOPE();

  ColumnVectorView<real const> solution = AsConstView(outSolution);
  auto const& descendants = reg.get<CIslandDescendants const>(island);
  ecs::InvokeForEach(rigid::EntityPostNewSolution, reg, descendants.rigidActors, solution);
  ecs::InvokeForEach(soft::EntityPostNewSolution, reg, descendants.softActors, solution);
  ecs::InvokeForEach(shell::EntityPostNewSolution, reg, descendants.shellActors, solution);
  ecs::InvokeForEach(rod::EntityPostNewSolution, reg, descendants.rodActors, solution);
  ecs::InvokeForEach(
      articulated::compound::EntityPostNewSolution, reg, descendants.compoundActors, solution);

  // articulated::rigid::EntityPostNewSolution must be called AFTER
  // articulated::compound::EntityPostNewSolution
  ecs::InvokeForEach(articulated::rigid::EntityPostNewSolution, reg, descendants.rigidActors);

  rom::PostNewSolutionPipeline(reg, descendants.softActors, solution);

  // Actors may cap or recenter the state. Copy it back to the solution vector to ensure
  // consistency.
  GetSolutions(outSolution, reg, descendants.actors, /*baseOffset*/ 0);
}

void mochi::solver::PostNewIncrementLocalPipeline(
    entt::registry& reg,
    entt::entity island,
    ColumnVectorView<real const> increment,
    ColumnVectorView<real> outSolution) {
  MOCHI_PROFILE_SCOPE();

  // If the increment is zero, just set the previous solution
  if (IsZero(increment)) {
    PostNewSolutionLocalPipeline(reg, island, outSolution);
    return;
  }

  ColumnVectorView<real const> solution = AsConstView(outSolution);

  auto const& descendants = reg.get<CIslandDescendants const>(island);
  ecs::InvokeForEach(
      rigid::EntityPostNewIncrement, reg, descendants.rigidActors, solution, increment);
  ecs::InvokeForEach(
      soft::EntityPostNewIncrement, reg, descendants.softActors, solution, increment);
  ecs::InvokeForEach(
      shell::EntityPostNewIncrement, reg, descendants.shellActors, solution, increment);
  ecs::InvokeForEach(rod::EntityPostNewIncrement, reg, descendants.rodActors, solution, increment);
  ecs::InvokeForEach(
      articulated::compound::EntityPostNewIncrement,
      reg,
      descendants.compoundActors,
      solution,
      increment);
  rom::PostNewIncrementPipeline(reg, descendants.softActors, solution, increment);

  // articulated::rigid::EntityPostNewSolution must be called AFTER
  // articulated::compound::EntityPostNewIncrement
  ecs::InvokeForEach(articulated::rigid::EntityPostNewSolution, reg, descendants.rigidActors);

  // Now that the actors have updated their state, copy it back to the solution vector to ensure
  // consistency.
  GetSolutions(outSolution, reg, descendants.actors, /*baseOffset*/ 0);
}

/*
 * Subpipeline to update derived state necessary for collision detection. Called from
 * UpdateDerivedStateBeforeAssembly().
 */
static void UpdateDerivedStateSubpipeline(
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();
  // Order requirements:
  // - 'articulated::compound' must be called before 'skinned'.
  // - 'skinned' must be called before 'blended'.
  articulated::compound::UpdateDerivedStatePipeline(reg, descendants.compoundActors);
  skinned::UpdateDerivedStatePipeline(reg, descendants.nestedSoftActors);
  blended::UpdateDerivedStatePipeline(reg, descendants.blendedActors);
}

void solver::UpdateJacobiansSubpipeline(
    TaskSemaphore& sem,
    entt::registry& reg,
    GradTarget gradTarget,
    CIslandDescendants const& descendants) {
  // None of the systems below apply to rigid actors. Early out if that's all we have.
  // Note that if there were articulated rigid actors, then there would also have to be an
  // articulated compound.
  if (descendants.rigidActors.size() == descendants.actors.size()) {
    return;
  }

  MOCHI_PROFILE_SCOPE();

  // Schedule a task if any of these actor types are in the island.
  Schedule(sem, "UpdateJacobiansPipeline", [&reg, &descendants, gradTarget]() {
    // Update Jacobians for articulated compounds, according to the gradient target.
    auto updateArticulatedJacobiansFn =
        articulated::compound::UpdateJacobiansStatePipeline<TimeStep::Current>;
    if (gradTarget == GradTarget::Previous) {
      updateArticulatedJacobiansFn =
          articulated::compound::UpdateJacobiansStatePipeline<TimeStep::StageStart>;
    } else if (gradTarget == GradTarget::CurrentInput) {
      updateArticulatedJacobiansFn =
          articulated::compound::UpdateJacobiansInputPipeline<TimeStep::Current>;
    } else if (gradTarget == GradTarget::PreviousInput) {
      updateArticulatedJacobiansFn =
          articulated::compound::UpdateJacobiansInputPipeline<TimeStep::Previous>;
    } else {
      MOCHI_ASSERT_VERBOSE(gradTarget == GradTarget::Current, "Unexpected gradient target");
    }
    updateArticulatedJacobiansFn(reg, descendants.compoundActors);
    {
      MOCHI_PROFILE_SCOPE_N("InvokeForEach articulated::rigid::EntityUpdateJacobian");
      // Update Jacobians for rigid actors in articulated compounds.
      // This must be called after the update of Jacobians of articulated compounds.
      ecs::InvokeForEach(&articulated::rigid::EntityUpdateJacobian, reg, descendants.rigidActors);
    }

    // The remaining terms are only needed for GradTarget::Current, as they're not supported with
    // differentiability
    if (gradTarget == GradTarget::Current) {
      // Update Jacobians for nested soft actors.
      // This must be called after the update of Jacobians of articulated
      skinned::UpdateJacobiansPipeline(reg, descendants.nestedSoftActors);
      // Update Jacobians for blended actors.
      // This must be called after the update of Jacobians of nested soft actors.
      blended::UpdateJacobiansPipeline(reg, descendants.blendedActors);
      // Update skinning Jacobians for rod contact skins.
      ecs::InvokeForEach(&rod::ResolveContactSkinningJacobian, reg, descendants.rodActors);
      // @TODO: ROMs use Jacobians also for velocity computations, so we can't move their Jacobian
      // computation here yet.
    }
  });
}

void solver::UpdateDerivedStateBeforeAssembly(
    entt::registry& reg,
    GradTarget gradTarget,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();

  // Update derived state necessary for both collision detection and contact Jacobians
  // For now, just skip unless gradTarget == GradTarget::Current, because the updated components
  // (e.g. skinning) are not supported with differentiability
  if (gradTarget == GradTarget::Current) {
    UpdateDerivedStateSubpipeline(reg, descendants);
  }

  // Jacobians are needed for first-order or input terms.
  TaskSemaphore updateJacobianSem;
  if (IsAssemblyNeeded(StateDependency::FirstOrder, true /*inputDependency*/, gradTarget)) {
    // Start update of Jacobians. Can run in parallel with most of CollisionDetectionPipeline().
    solver::UpdateJacobiansSubpipeline(updateJacobianSem, reg, gradTarget, descendants);
  }

  // Collision detection is needed only for first-order terms.
  if (IsAssemblyNeeded(StateDependency::FirstOrder, false /*inputDependency*/, gradTarget)) {
    // Perform collision detection.
    CollisionDetectionPipeline<TimeStep::Current>(reg, descendants);

    // Set up contact Jacobians. Also calls updateJacobianSem.Wait().
    ContactJacobiansPipeline(reg, gradTarget, descendants, updateJacobianSem);
  } else {
    // Wait for any tasks scheduled by UpdateJacobiansSubpipeline.
    updateJacobianSem.Wait();
  }
}

static void ApplyBoundaryConditionsToProblem(
    entt::registry& reg,
    Span<entt::entity const> actors,
    AssemblyParams const& params,
    SnleProblem<real>& problem) {
  MOCHI_PROFILE_SCOPE();
  // TODO: Schedule SetZeroOnRows as non-blocking tasks if the number of zeros to set is large.
  for (auto e : actors) {
    auto const* dirichlet = reg.try_get<CDirichletBC<real> const>(e);
    if (dirichlet && !dirichlet->dofIndices.empty()) {
      int dofOffset = reg.get<CDofOffset const>(e).dofsOffset;

      // Apply BCs to actors and interactions
      if (params.assemRes) {
        for (auto&& [actorOffset, actorRes] : problem.actorResiduals) {
          if (actorOffset == dofOffset) {
            SetZeroOnRows(*actorRes, dirichlet->dofIndices, 0); // No DoF offset
            break;
          }
        }
        for (auto&& [rOffset, interactionRes] : problem.interactionResiduals) {
          SetZeroOnRows(*interactionRes, dirichlet->dofIndices, dofOffset - rOffset);
        }
      }
      if (params.assemDRes) {
        for (auto&& [actorOffset, actorMat] : problem.actorMatrices) {
          if (actorOffset == dofOffset) {
            MOCHI_ASSERT(
                std::holds_alternative<Matrix<real>>(*actorMat) ||
                    !dirichlet->colValueIndices.empty(),
                "Non-dense actors must cache column value indices for fast column zeroing of Dirichlet BCs.");
            SetZeroOnCols(
                AsView(*actorMat),
                dirichlet->dofIndices,
                /* No DoF offset */ 0,
                AsConstView(*actorMat),
                MakeConstSpan(dirichlet->colValueIndices));
            SetZeroOnRows(AsView(*actorMat), dirichlet->dofIndices, /* No DoF offset */ 0, 1_r);
            break;
          }
        }
        for (auto&& [rOffset, cOffset, interactionMat, symmetricPair] :
             problem.interactionMatrices) {
          MOCHI_ASSERT(
              (rOffset == cOffset) || symmetricPair.has_value(),
              "Off-diagonal interaction matrices must have a symmetric pair.");
          // Performance note: The sparsity pattern of some interaction matrices is fixed. The
          // column value indices could be precomputed and cached in such cases.
          SetZeroOnCols(
              AsView(*interactionMat),
              dirichlet->dofIndices,
              dofOffset - cOffset,
              symmetricPair.has_value() ? *symmetricPair : AsConstView(*interactionMat));
          // Even the diagonal should be zero in this case.
          SetZeroOnRows(AsView(*interactionMat), dirichlet->dofIndices, dofOffset - rOffset, 0_r);
        }
      }
    }
  }
}

static bool ApplyBoundaryConditionsToSolution(
    entt::registry& reg,
    Span<entt::entity const> actors,
    ColumnVectorView<real> outSolutions) {
  bool bcActive = false;

  // Set boundary conditions to solution (if any)
  for (auto actor : actors) {
    auto const* dirichlet = reg.try_get<CDirichletBC<real> const>(actor);
    if (dirichlet && !dirichlet->poseIndices.empty()) {
      MOCHI_ASSERT((!reg.any_of<TagRomActor>(actor)), "ROM does not support Dirichlet bc yet!");
      int bcSize = isize(dirichlet->poseIndices);
      MOCHI_ASSERT(isize(dirichlet->poseValues) == bcSize);
      if (bcSize > 0) {
        bcActive = true;
        int poseOffset = reg.get<CDofOffset const>(actor).poseOffset;
        for (size_t i = 0; i < bcSize; ++i) {
          outSolutions[dirichlet->poseIndices[i] + poseOffset] = dirichlet->poseValues[i];
        }
      }
    }
  }

  return bcActive;
}

static void GatherSnleDataFromActors(
    SnleProblem<real>& outProblem,
    AssemblyParams const& params,
    entt::registry& reg,
    entt::entity island,
    Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();

  auto isInputTarget = params.gradTarget == GradTarget::CurrentInput ||
      params.gradTarget == GradTarget::PreviousInput;

  // Helper: Adds to actorResiduals and/or actorMatrices
  auto addActorSnleToProblem = [&](int dofOffset, ActorSnle* actorSnle, entt::entity actor) {
    if (actorSnle && actorSnle->useInSolver) {
      bool const useReduced = actorSnle->UseReduced();
      auto& res = useReduced ? actorSnle->reducedResidual : actorSnle->fullResidual;
      auto& dRes = useReduced ? actorSnle->reducedDResidual : actorSnle->fullDResidual;

      if (params.assemObj) {
        outProblem.objective += actorSnle->objective;
      }
      if (params.assemRes) {
        MOCHI_ASSERT(isInputTarget || !res.empty(), "Residual must not be empty.");
        outProblem.actorResiduals.emplace_back(dofOffset, &res);
        auto const& weights = reg.get<CActorConvergenceWeights const>(actor);
        MOCHI_ASSERT(weights.isValid, "Invalid actor convergence weights.");
        outProblem.actorConvergenceWeights.emplace_back(dofOffset, &weights.values);
      }
      if (params.assemDRes) {
        MOCHI_ASSERT(GetNumRows(dRes) > 0);
        outProblem.actorMatrices.emplace_back(dofOffset, &dRes);
        outProblem.actorPreconditioners.emplace_back(
            dofOffset,
            useReduced ? actorSnle->reducedPreconditioner : actorSnle->fullPreconditioner,
            useReduced ? actorSnle->reducedPreconditionerType : actorSnle->fullPreconditionerType);
      }
    }
  };

  // Helper: Adds to interactionResiduals and/or interactionMatrices
  auto addInteractionSnleToProblem = [&](InteractionSnle* interSnle) {
    if (interSnle && interSnle->useInSolver) {
      if (params.assemObj) {
        outProblem.objective += interSnle->objective;
      }
      if (params.assemRes) {
        for (auto& [rowOffset, residual] : interSnle->residuals) {
          MOCHI_ASSERT(isInputTarget || !residual.empty(), "Residual must not be empty.");
          outProblem.interactionResiduals.emplace_back(rowOffset, &residual);
        }
      }
      if (params.assemDRes) {
        for (auto& [rowOffset, colOffset, matrix, symmetricPair] : interSnle->dresiduals) {
          MOCHI_ASSERT(GetNumValues(matrix) > 0, "DResidual must not be empty.");
          outProblem.interactionMatrices.emplace_back(rowOffset, colOffset, &matrix, symmetricPair);
        }
      }
    }
  };

  // Clear previous per-actor data
  if (params.assemObj) {
    outProblem.objective = 0.0;
  }
  if (params.assemRes) {
    outProblem.actorResiduals.clear();
    outProblem.interactionResiduals.clear();
    outProblem.actorConvergenceWeights.clear();
  }
  if (params.assemDRes) {
    outProblem.actorMatrices.clear();
    outProblem.interactionMatrices.clear();
    outProblem.actorPreconditioners.clear();
  }

  for (auto e : actors) {
    ActorSnle* actorSnle = reg.try_get<CActorSnle>(e);
    if (actorSnle && actorSnle->useInSolver) {
      // Determine the dof offset for this actor based on the grad target.
      int dofOffset{};
      switch (params.gradTarget) {
        case GradTarget::Current:
        case GradTarget::Previous:
          dofOffset = reg.get<CDofOffset const>(e).dofsOffset;
          break;
        case GradTarget::PreviousDelta:
          dofOffset = reg.get<CDerivedStateOffset const>(e).dofsOffset;
          break;
        case GradTarget::CurrentInput:
        case GradTarget::PreviousInput:
          dofOffset = reg.get<CDiffInputOffset const>(e).dofsOffset;
          break;
        default:
          MOCHI_ASSERT_VERBOSE(false, "Unexpected grad target");
          break;
      }
      addActorSnleToProblem(dofOffset, actorSnle, e);
    }
    // Add any interactions stored on this actor.
    addInteractionSnleToProblem(reg.try_get<CCompoundConstraintSnle>(e));
    addInteractionSnleToProblem(reg.try_get<CSkinnedContactSnle>(e));
    addInteractionSnleToProblem(reg.try_get<CSkinnedInteractionSnle>(e));
  }

  addInteractionSnleToProblem(reg.try_get<CIslandContactSnle>(island));
}

// Ordered sequence of systems to assemble a single soft actor (includes soft ROM)
static void
AssembleSoftActorPipeline(AssemblyParams const& params, entt::registry& reg, entt::entity e) {
  auto* semComp = reg.try_get<CActorAsyncContactSemaphore>(e);
  bool useContact = reg.all_of<TagUseContact>(e);

  ecs::InvokeOnEntity(soft::AssembleBody, reg, e, std::cref(params));
  ecs::TryInvokeOnEntity(rom::AssembleAndProjectBody, reg, e, std::cref(params));

  if (semComp && useContact) {
    // Wait for async contact to complete before we continue
    semComp->asyncContactUpToDate->Wait();
  }

  ecs::TryInvokeOnEntity(soft::AssembleAsyncContact, reg, e, std::cref(params));
  ecs::TryInvokeOnEntity(rom::AssembleFullToReduced, reg, e, std::cref(params));
  ecs::TryInvokeOnEntity(rom::AssembleAndProjectAsyncContact, reg, e, std::cref(params));
}

static void
AssembleShellActorPipeline(AssemblyParams const& params, entt::registry& reg, entt::entity e) {
  auto* semComp = reg.try_get<CActorAsyncContactSemaphore>(e);
  bool useContact = reg.all_of<TagUseContact>(e);

  ecs::InvokeOnEntity(shell::AssembleBody, reg, e, std::cref(params));

  if (semComp && useContact) {
    // Wait for async contact to complete before we continue
    semComp->asyncContactUpToDate->Wait();
  }

  ecs::TryInvokeOnEntity(shell::AssembleAsyncContact, reg, e, std::cref(params));
}

static void
AssembleRodActorPipeline(AssemblyParams const& params, entt::registry& reg, entt::entity e) {
  auto* semComp = reg.try_get<CActorAsyncContactSemaphore>(e);
  bool const useContact = reg.any_of<TagUseContact>(e);

  ecs::InvokeOnEntity(rod::AssembleBody, reg, e, std::cref(params));

  if (semComp && useContact) {
    // Wait for async contact to complete before we continue
    semComp->asyncContactUpToDate->Wait();
  }

  // Centerline async contact (uses CFemSegmentDiscretization). Contact-skin async contact
  // is assembled at the island level (AssembleIslandRodAsyncContact) to avoid merging
  // contact sparsity into the rod's incompatible pentadiagonal body matrix.
  if (!reg.all_of<TagRodSurfaceContact>(e)) {
    ecs::TryInvokeOnEntity(rod::AssembleAsyncContact, reg, e, std::cref(params));
  }
}

/*
 * Pipeline to assemble the objective, residual and/or residual derivative. Note:
 * - The solution vector of the non-linear problem and the position components of the state (aka
 *   position state) of all actors must have been updated and be in sync before the pipeline is
 *   executed.
 * - The velocity components of the state (aka velocity state) and other quantities that are a
 *   function of the state (aka derived state) do NOT need to be in sync with the solution and
 *   the position state. But if they are not in sync and are required for the assembly, they MUST
 *   be updated in UpdateDerivedStateBeforeAssembly, which is executed during this pipeline.
 */
void mochi::solver::AssembleIslandPipeline(
    entt::registry& reg,
    entt::entity island,
    AssemblyParams const& params,
    SnleProblem<real>& problem) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_PROFILE_DESCRIPTION_F(
      "assemObj: %d\nassemRes: %d\nassemDres: %d",
      static_cast<int>(params.assemObj),
      static_cast<int>(params.assemRes),
      static_cast<int>(params.assemDRes));

  auto const& descendants = reg.get<CIslandDescendants const>(island);

  // Used to wait for async tasks
  TaskSemaphore masterSemaphore, softCompletionSem, artCompletionSem;

  // Used to signal start of the rest of the assembly
  TaskSemaphore startOtherAssemblySemaphore;
  bool hasSolutionChanged = problem.HasSolutionChangedSinceLastAssembly();
  if (hasSolutionChanged) {
    for (auto e : descendants.actors) {
      if (reg.all_of<TagUseContact>(e)) {
        if (auto* comp = reg.try_get<CActorAsyncContactSemaphore>(e)) {
          MOCHI_ASSERT(
              comp->asyncContactUpToDate->IsDone(),
              "Should be incremented and decremented exactly once per assembly");
          comp->asyncContactUpToDate->Add(1);
        }
      }
    }

    // Update derived state and perform collision detection.
    startOtherAssemblySemaphore.Add(1);
    Schedule(masterSemaphore, "UpdateDerivedStateBeforeAssembly", [&]() {
      UpdateDerivedStateBeforeAssembly(reg, params.gradTarget, descendants);
      startOtherAssemblySemaphore.Done();
    });
  }

  // Start async tasks that can begin before contact. If they need contact data to complete
  // their work, then they can wait for the startSoftActorAssemblySemaphore and resume when it
  // is ready.

  for (auto e : descendants.softActors) {
    softCompletionSem.Add(1);
    Schedule(masterSemaphore, "AssembleSoftActorPipeline", [&, e, softCompletionSem]() {
      AssembleSoftActorPipeline(params, reg, e);
      softCompletionSem.Done();
    });
  }

  for (auto e : descendants.shellActors) {
    Schedule(masterSemaphore, "AssembleShellActorPipeline", [&, e]() {
      AssembleShellActorPipeline(params, reg, e);
    });
  }

  for (auto e : descendants.rodActors) {
    Schedule(masterSemaphore, "AssembleRodActorPipeline", [&, e]() {
      AssembleRodActorPipeline(params, reg, e);
    });
  }

  // If all actors in the island have a multiple of 3 DoFs and are not BlockSparseMatrix<real, 4>,
  // then BlockSparseMatrix<real, 3> can be used to represent any dresidual, including interaction
  // matrices.
  static_assert(
      std::variant_size_v<decltype(CActorSnle::fullDResidual)> == 4,
      "Please update the logic below if the actor matrix types change");
  bool useBlockSparse3x3ForInteractions = true;
  if (params.assemDRes) {
    for (auto e : descendants.actors) {
      if (auto* actorSnle = reg.try_get<CActorSnle>(e)) {
        auto& dRes =
            actorSnle->UseReduced() ? actorSnle->reducedDResidual : actorSnle->fullDResidual;
        bool const isBlockable3x3 = (std::holds_alternative<BlockSparseMatrix<real, 3>>(dRes) ||
                                     std::holds_alternative<Matrix<real>>(dRes) ||
                                     std::holds_alternative<SparseMatrix<real>>(dRes)) &&
            (reg.get<CActorDofInfo>(e).dofsSize % 3 == 0);
        if (!isBlockable3x3) {
          useBlockSparse3x3ForInteractions = false;
          break;
        }
      }
    }
  }

  // Start async tasks that require contact or Jacobian data.
  startOtherAssemblySemaphore.Wait();

  // Assemble sync contact within the island. Contact is first-order; assemble only if needed.
  if (IsAssemblyNeeded(StateDependency::FirstOrder, true /*inputDependency*/, params.gradTarget)) {
    ecs::TryScheduleInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
        masterSemaphore,
        "AssembleIslandSyncContact",
        &AssembleIslandSyncContact,
        reg,
        island,
        std::cref(params),
        useBlockSparse3x3ForInteractions);
  }

  // Start a task for each compound with constraints (including articulated compounds). Constraints
  // are first-order and rely on input; assemble only if needed.
  TaskSemaphore constraintSemaphore;
  if (IsAssemblyNeeded(StateDependency::FirstOrder, true /*inputDependency*/, params.gradTarget)) {
    ecs::ScheduleInvokeForEach<ecs::policy::AllowFullRegistryAccess>(
        constraintSemaphore,
        "compound::AssembleConstraints",
        compound::AssembleConstraints,
        reg,
        descendants.compoundActors,
        std::cref(params));
  }

  // Start a task for each actor with async skinned contact.
  std::array<Span<entt::entity const>, 3> skinnedActors = {
      descendants.compoundActors, descendants.nestedSoftActors, descendants.rodActors};
  for (auto actors : skinnedActors) {
    ecs::ScheduleInvokeForEach<ecs::policy::AllowFullRegistryAccess>(
        masterSemaphore,
        "AssembleAsyncSkinnedContact",
        AssembleAsyncSkinnedContact,
        reg,
        actors,
        std::cref(params),
        useBlockSparse3x3ForInteractions);
  }

  // rigid::EntityAssemble is called for both (normal) rigid bodies and for articulated rigid
  // bodies.
  if (IsAssemblyNeeded(StateDependency::SecondOrder, true /*inputDependency*/, params.gradTarget)) {
    ecs::InvokeForEach(rigid::EntityAssemble, reg, descendants.rigidActors, std::cref(params));
  }

  // Wait for constraint assembly to finish (but don't wait for anything else). Then, schedule a
  // task for each articulated compound. They will read the full DoF values assembled from their
  // rigid bodies and constraints (already assembled by this point).
  constraintSemaphore.Wait();
  for (auto e : descendants.compoundActors) {
    if (reg.all_of<TagArticulatedActor, CActorSnle>(e)) {
      artCompletionSem.Add(1);
      Schedule(
          masterSemaphore, "articulated::compound::EntityAssemble", [&, e, artCompletionSem]() {
            ecs::InvokeOnEntity<ecs::policy::AllowReadWriteSameComponent>(
                articulated::compound::EntityAssemble, reg, e, std::cref(params));
            artCompletionSem.Done();
          });
    }
  }

  // Schedule assembly of each nested soft actor's posed energy terms. This may need to wait for
  // soft and/or articulated assembly to complete.
  ecs::ScheduleInvokeForEach<ecs::policy::AllowFullRegistryAccess>(
      masterSemaphore,
      "skinned::EntityAssembleBody",
      skinned::EntityAssembleBody,
      reg,
      descendants.nestedSoftActors,
      std::cref(params),
      softCompletionSem,
      artCompletionSem);

  // Wait for all of the above tasks
  masterSemaphore.Wait();

  // Update convergence weights. Must run after per-actor assembly (dependencies updated) and
  // before GatherSnleDataFromActors (weights consumed).
  ecs::InvokeForEach(&UpdateActorConvergenceWeights, reg, descendants.actors, std::cref(reg));

  // Collect SNLE data from the individual actors.
  GatherSnleDataFromActors(problem, params, reg, island, descendants.actors);

  // Apply Dirichlet Boundary Conditions
  ApplyBoundaryConditionsToProblem(reg, descendants.actors, params, problem);
}

void mochi::solver::SetTimeIntegratorState(
    entt::registry& reg,
    Span<entt::entity const> actors,
    TimeIntegratorParams const& params,
    int iStage) {
  MOCHI_PROFILE_SCOPE();
  auto dt = static_cast<real>(reg.ctx<CSceneTime const>().DeltaTime());
  for (auto e : actors) {
    if (auto* intState = reg.try_get<CTimeIntegratorState>(e)) {
      intState->Set(params, iStage, dt);
    }
  }
}

void mochi::solver::PreFirstStageLocalPipeline(
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();
  // Invalidate convergence weights for configuration-dependent actors. Invalid weights are
  // recomputed during the assembly, where all dependencies are up-to-date.
  for (auto e : descendants.actors) {
    InvalidateConfigDependentActorConvergenceWeights(reg, e);
  }

  // Order requirements: None
  ecs::InvokeForEach(&soft::EntityPreFirstStage, reg, descendants.softActors);
  ecs::InvokeForEach(&shell::EntityPreFirstStage, reg, descendants.shellActors);
  ecs::InvokeForEach(&rod::EntityPreFirstStage, reg, descendants.rodActors);
  ecs::InvokeForEach(&rigid::EntityPreFirstStage, reg, descendants.rigidActors);
  ecs::InvokeForEach(&articulated::compound::EntityPreFirstStage, reg, descendants.compoundActors);
  ecs::InvokeForEach(&articulated::rigid::EntityPreFirstStage, reg, descendants.rigidActors);
  ecs::InvokeForEach(&rom::EntityPreFirstStage, reg, descendants.softActors);
  ecs::InvokeForEach(&skinned::EntityPreFirstStage, reg, descendants.nestedSoftActors);
}

void mochi::solver::PreStageLocalPipeline(
    entt::registry& reg,
    CIslandDescendants const& descendants,
    SnleProblem<real>& problem) {
  MOCHI_PROFILE_SCOPE();
  // Order requirements:
  // - 'rom' must be called after 'soft'. This was required for GMMs. Still required?
  // - 'skinned' must be called after 'soft' and 'articulated::compound'.
  // - 'blended' must be called after 'skinned'.
  ecs::InvokeForEach(&rigid::EntityPreStage, reg, descendants.rigidActors);
  ecs::InvokeForEach(&soft::EntityPreStage, reg, descendants.softActors);
  ecs::InvokeForEach(&shell::EntityPreStage, reg, descendants.shellActors);
  ecs::InvokeForEach(&rod::EntityPreStage, reg, descendants.rodActors);
  rom::PreStagePipeline(reg, descendants.softActors);
  if (!descendants.compoundActors.empty()) {
    // Use the full list of actors because this pipeline interleaves operations for articulated
    // compounds and articulated rigid bodies.
    articulated::compound::PreStagePipeline(reg, descendants.actors);
  }
  skinned::PreStagePipeline(reg, descendants.nestedSoftActors);
  blended::PreStagePipeline(reg, descendants.blendedActors);

  // Once actors have computed their states, copy them to the solution vector.
  GetSolutions(problem.solution, reg, descendants.actors, /*baseOffset*/ 0);

  bool const explicitNormals = reg.ctx<CSimulationParams const>().experimentalEval.explicitNormals;
  if (explicitNormals) {
    // Run collision detection pipeline with stage-start configuration.
    // This populates CActiveCollisions<*, TimeStep::StageStart> for all actors.
    CollisionDetectionPipeline<TimeStep::StageStart>(reg, descendants);
  } else {
    // Update colliding samples and mapped colliders at stage start.
    UpdateStageStartDataPipeline(reg, descendants);
  }
}

/*
 * Pipeline executed after each time integration stage. Each actor is responsible for:
 * - Updating its position and velocity state at the end of the time integration stage.
 * - If position and/or velocity are differential variables, push them to the vector containing the
 *   differential variables at the end of each time integration stage.
 */
void mochi::solver::PostStageLocalPipeline(
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();
  ecs::InvokeForEach(&soft::EntityPostStage, reg, descendants.softActors);
  ecs::InvokeForEach(&shell::EntityPostStage, reg, descendants.shellActors);
  ecs::InvokeForEach(&rod::EntityPostStage, reg, descendants.rodActors);
  ecs::InvokeForEach(&rigid::EntityPostStage, reg, descendants.rigidActors);
  if (!descendants.compoundActors.empty()) {
    // Use the full list of actors because this pipeline interleaves operations for articulated
    // compounds and articulated rigid bodies.
    articulated::compound::PostStagePipeline(reg, descendants.actors);
  }
  rom::PostStagePipeline(reg, descendants.softActors);

  // Must be called after articulated::compound::PostStagePipeline.
  ecs::InvokeForEach(&skinned::EntityPostStage, reg, descendants.nestedSoftActors);
}

/*
 * Pipeline executed after the last time integration stage of the time step. Each actor is
 * responsible for computing its position and velocity state at the end of the time step.
 */
static void PostLastStageLocalPipeline(entt::registry& reg, CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();

  // Order requirements:
  // - 'skinned' must be called after 'soft' and 'articulated::compound'.
  // - 'blended' must be called after 'skinned'.
  // - 'soft::UpdateRigidTransformEval' must be called after the actors' PostLastStage systems.
  // - Recording pipelines must be called after 'soft::UpdateRigidTransformEval'.
  ecs::InvokeForEach(&soft::EntityPostLastStage, reg, descendants.softActors);
  ecs::InvokeForEach(&shell::EntityPostLastStage, reg, descendants.shellActors);
  ecs::InvokeForEach(&rod::EntityPostLastStage, reg, descendants.rodActors);
  ecs::InvokeForEach(&rigid::EntityPostLastStage, reg, descendants.rigidActors);
  if (!descendants.compoundActors.empty()) {
    // Use the full list of actors because this pipeline interleaves operations for articulated
    // compounds and articulated rigid bodies.
    articulated::compound::PostLastStagePipeline(reg, descendants.actors);
  }
  ecs::InvokeForEach(&rom::EntityPostLastStage, reg, descendants.softActors);
  skinned::PostLastStagePipeline(reg, descendants.nestedSoftActors);
  blended::PostLastStagePipeline(reg, descendants.blendedActors);

  // Update the rigid transform at the specified eval point.
  // Used for soft actors (both FOMs and ROMs).
  ecs::InvokeForEach(&soft::UpdateRigidTransformEval, reg, descendants.actors);

  // Perform a final collision detection pass with far SDF evaluation
  // States of objects haven't changed since last collision detection evaluation, so we
  // can easily skip pre collision detection
  ecs::InvokeForEach<ecs::policy::AllowFullRegistryAccess>(
      &FarSdfCollisionDetection, reg, descendants.actors);

  // Optionally record state and queries
  if (reg.try_ctx<TagSceneRecordingEnabled>()) {
    auto const& params = reg.ctx<CRecordingParams const>();
    if (params.recordTargetState) {
      ecs::InvokeForEach<ecs::policy::AllowFullRegistryAccess>(
          articulated::compound::RecordTargetState, reg, descendants.compoundActors);
    }
    if (params.recordDynamicActorState) {
      ecs::InvokeForEach(rigid::RecordState, reg, descendants.rigidActors);
      deformable::RecordingPipeline(reg, descendants.actors);
      rom::RecordingPipeline(reg, descendants.actors);
      ecs::InvokeForEach(articulated::compound::RecordState, reg, descendants.compoundActors);
    }
    if (params.recordContactPoints) {
      ecs::InvokeForEach(RecordQueryContactPoints, reg, descendants.actors);
    }
    if (params.recordNodeContactForces) {
      ecs::InvokeForEach(RecordQueryNodeContactForces, reg, descendants.actors);
    }
    if (params.recordSdfDistances) {
      ecs::InvokeForEach(RecordQuerySdfDistances, reg, descendants.actors);
    }
  }
}

// NOTE: Multi-step integrators require using a different integrator if there are not enough
// previous steps available (cold start), e.g. BDF2 requires using backward Euler in the first step.
// The number of previous steps available is determined by the minimum across all actors in the
// island.
TimeIntegratorParams mochi::solver::CreateIslandTimeIntegrationParams(
    entt::registry const& reg,
    CIslandDescendants const& descendants,
    IntegrationMethod const& targetMethod) {
  auto getNumPrevStepsAvailable = [&]() {
    int numPrevSteps = std::numeric_limits<int>::max();
    for (auto const& actor : descendants.actors) {
      int numPrevStepsActor = -1;

      // WARNING: Order of branches matters.
      if (reg.any_of<TagStaticActor>(actor)) {
        // Static actors do not participate in time integration and therefore don't limit the number
        // of previous steps available.
        numPrevStepsActor = kMaxIntegrationSteps;
      } else if (reg.any_of<TagRigidActor>(actor)) {
        numPrevStepsActor = isize(reg.get<CIntegrationRigidVels>(actor).prevSteps);
      } else if (reg.any_of<TagSoftActor, TagShellActor>(actor)) {
        numPrevStepsActor = isize(reg.get<CIntegrationDisplacementSlices>(actor).prevSteps);
      } else if (reg.any_of<TagRodActor>(actor)) {
        numPrevStepsActor = isize(reg.get<CIntegrationRodPoses>(actor).prevSteps);
      } else if (reg.any_of<TagArticulatedActor>(actor)) {
        numPrevStepsActor = isize(reg.get<CIntegrationArticulatedReducedPose>(actor).prevSteps);
      } else {
        MOCHI_ASSERT_VERBOSE(reg.any_of<TagCompoundActor>(actor), "Unexpected actor type.");
        // Compound actors that are not articulated actors do not participate in time integration
        // and therefore don't limit the number of previous steps available.
        numPrevStepsActor = kMaxIntegrationSteps;
      }

#if MOCHI_ASSERT_VERBOSE_ENABLED
      // Consistency checks.
      if (auto const* intRigidStates = reg.try_get<CIntegrationRigidStates>(actor)) {
        MOCHI_ASSERT_VERBOSE(numPrevStepsActor == isize(intRigidStates->prevSteps));
      }
      if (auto const* intVels =
              reg.try_get<CIntegrationVelocitySlices<DisplacementLayer::Default>>(actor)) {
        MOCHI_ASSERT_VERBOSE(numPrevStepsActor == isize(intVels->prevSteps));
      }
      if (auto const* intSkinnedVels =
              reg.try_get<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>(actor)) {
        MOCHI_ASSERT_VERBOSE(numPrevStepsActor == isize(intSkinnedVels->prevSteps));
      }
      if (auto const* intJointVels = reg.try_get<CIntegrationArticulatedJointVels>(actor)) {
        for (auto const& jointVel : intJointVels->value) {
          MOCHI_ASSERT_VERBOSE(numPrevStepsActor == isize(jointVel.prevSteps));
        }
      }
      if (auto const* intRodPoses = reg.try_get<CIntegrationRodPoses>(actor)) {
        MOCHI_ASSERT_VERBOSE(numPrevStepsActor == isize(intRodPoses->prevSteps));
      }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

      numPrevSteps = Min(numPrevSteps, numPrevStepsActor);
    }

    MOCHI_ASSERT_VERBOSE(numPrevSteps >= 0 && numPrevSteps <= kMaxIntegrationSteps);
    return numPrevSteps + 1; // "Previous" is always available and not included in prevSteps yet.
  };

  IntegrationMethod method = targetMethod;
  if (targetMethod == IntegrationMethod::BDF2) {
    method = (getNumPrevStepsAvailable() >= 2) ? IntegrationMethod::BDF2
                                               : IntegrationMethod::BackwardEuler;
  } else if (targetMethod == IntegrationMethod::BDF3) {
    int const numPrevSteps = getNumPrevStepsAvailable();
    method = (numPrevSteps >= 3) ? IntegrationMethod::BDF3
        : (numPrevSteps >= 2)    ? IntegrationMethod::BDF2
                                 : IntegrationMethod::BackwardEuler;
  } else {
    MOCHI_ASSERT_VERBOSE(GetNumSteps(targetMethod) == 1);
  }
  static_assert(
      static_cast<int>(IntegrationMethod::Count) == 8,
      "Please update the if statement above if IntegrationMethod enumerator changes");

  return CreateTimeIntegratorParams(method);
}

// Builds the island's SnleProblem, gathers the initial solution from the actors, and applies
// Dirichlet boundary conditions. The assembly callback is always set. When @p withSolve is true,
// the solve-time callbacks (onPostNewSolution, onPostNewIncrement) are also added.
static SnleProblem<real> InitIslandProblem(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants,
    bool withSolve) {
  CIslandDofInfo const& islandDofInfo = reg.get<CIslandDofInfo>(island);
  int const solutionSize = islandDofInfo.poseSize;
  int const numDofs = islandDofInfo.dofsSize;

  // SNLE Callback Functions. Always assemble; add solve-time callbacks only when solving.
  // -----------------------------------------------------------------------
  SnleProblemFunctions<real> functions;
  functions.assemble = [&reg, island](SnleProblem<real>& problem, AssemblyParams const& params) {
    solver::AssembleIslandPipeline(reg, island, params, problem);
  };
  if (withSolve) {
    functions.onPostNewSolution = [&reg, island](auto& problem) {
      solver::PostNewSolutionLocalPipeline(reg, island, problem.solution);
    };
    functions.onPostNewIncrement = [&reg, island](auto& problem) {
      solver::PostNewIncrementLocalPipeline(reg, island, problem.increment, problem.solution);
    };
  }

  // SNLE Problem
  // -----------------------------------------------------------------------
  SnleProblem<real> problem(numDofs, solutionSize, std::move(functions));

  // Gather initial solution
  GetSolutions(problem.solution, reg, descendants.actors, 0);

  // Apply boundary conditions (if any) to solution and actors
  if (ApplyBoundaryConditionsToSolution(reg, descendants.actors, problem.solution)) {
    SetSolutions(reg, descendants.actors, problem.solution);
    solver::PostNewSolutionLocalPipeline(reg, island, problem.solution);
  }
  return problem;
}

void mochi::solver::AssembleIsland(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants,
    ColumnVectorView<real> outRes,
    MatrixView<real> outDRes) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(isize(descendants.actors) > 0, "Empty islands should have been pruned");

  SnleProblem<real> problem = InitIslandProblem(reg, island, descendants, /*withSolve*/ false);

  // Multi-stage assembly is not supported
  auto const& simParams = reg.ctx<CSimulationParams const>();
  TimeIntegratorParams const integrationParams =
      CreateIslandTimeIntegrationParams(reg, descendants, simParams.integrationMethod);
  MOCHI_ASSERT(integrationParams.numStages == 1, "Only one stage is supported for assembly");
  SetTimeIntegratorState(reg, descendants.actors, integrationParams, 0);

  PreFirstStageLocalPipeline(reg, descendants);

  PreStageLocalPipeline(reg, descendants, problem);

  problem.UpdateObjResDRes(
      {.assemObj = false, // Can be set to false if the merit isn't needed
       .assemRes = !outRes.empty(),
       .assemDRes = !outDRes.empty(),
       .psdDRes = false}); // Set to true if you need the SPD projection of the Hessian

  // Output residual
  if (!outRes.empty()) {
    outRes = problem.GetResidual();
  }

  // Output dresidual
  if (!outDRes.empty()) {
    outDRes = ToMatrix(problem.GetDResidual());
  }
}

bool mochi::solver::StepIslandNewtonAsync(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(descendants.actors.size() > 0, "Empty islands should have been pruned");

  auto const& islandDofInfo = reg.get<CIslandDofInfo>(island);
  auto& islandSolverStats = reg.get<CIslandSolverStats>(island);
  int const numDofs = islandDofInfo.dofsSize;

  // Clear island solver stats.
  islandSolverStats.stages.clear();

  // Reset step-level convergence status. DistributeConvergenceStatus accumulates via
  // stepStatus = Max(stepStatus, stageStatus) across stages, so the initial value must be None.
  for (auto e : descendants.actors) {
    if (auto* convergence = reg.try_get<CConvergenceStatus>(e)) {
      convergence->stepStatus = ConvergenceStatus::None;
    }
  }

  // Configure SNLE solver
  // ------------------------------------------------------------------------
  auto const& simParams = reg.ctx<CSimulationParams const>();
  NewtonSolverParams newtonParams;
  GetIslandNewtonParams(numDofs, simParams, newtonParams);
  auto& preconditioner = reg.get<CIslandPreconditioner>(island);
  NewtonSolver<real> snleSolver(newtonParams, preconditioner);

  // SNLE Problem
  // -----------------------------------------------------------------------
  SnleProblem<real> problem = InitIslandProblem(reg, island, descendants, /*withSolve*/ true);

  // Perform time integration.
  bool success = true;
  auto const integrationParams =
      CreateIslandTimeIntegrationParams(reg, descendants, simParams.integrationMethod);

  for (int iStage = 0; iStage < integrationParams.numStages; ++iStage) {
    bool const isImplicitStage = (integrationParams.A(iStage, iStage) != 0_r);
    MOCHI_ASSERT(isImplicitStage, "Explicit time integration stages not supported yet.");
    SetTimeIntegratorState(reg, descendants.actors, integrationParams, iStage);

    if (iStage == 0) {
      PreFirstStageLocalPipeline(reg, descendants);
    }

    PreStageLocalPipeline(reg, descendants, problem);

    // Stage solve: the non-linear problem is solved. Islands have no knowledge on the physical
    // problem being solved here, they only serve as a brigde between solver and actors.
    NewtonSolverStatus<real> result = snleSolver.Solve(problem);
    success &= (result.convergence == ConvergenceStatus::Converged);
    islandSolverStats.stages.emplace_back(StageSolverStats::FromNewtonSolverStatus(result));

    // Distribute solution to actors. NOTE: This call might be redundant in some cases,
    // actors most probably already implement the CSnleProblemPostNewSolutionCallback,
    // which is called after each line-search iteration
    SetSolutions(reg, descendants.actors, problem.solution);

    // Distribute convergence status to actors. Must be called before PostStageLocalPipeline.
    DistributeConvergenceStatus(reg, descendants.actors, problem, result);

    PostStageLocalPipeline(reg, descendants);
  }

  PostLastStageLocalPipeline(reg, descendants);

  return success;
}
