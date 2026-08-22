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
#include "mochi_ecs.h"
#include "mochi_group.h"
#include "mochi_island.h"
#include "mochi_simulation.h"

#include <mochi_core/solvers/snle_problem.h>
#include <mochi_core/utils/differentiability.h>
#include <mochi_physics/mochi_physics.h>

namespace mochi {

/*
 * Copies the position components of the state (aka position state) of all actors to the solution
 * vector of the non-linear problem.
 */
void GetSolutions(
    ColumnVectorView<real> outSolutions,
    entt::registry& reg,
    Span<entt::entity const> entities,
    int baseOffset);

/*
 * Copy a solution vector to the position components of the state of all actors.
 */
void SetSolutions(
    entt::registry& reg,
    Span<entt::entity const> entities,
    ColumnVectorView<real const> solution);
} // namespace mochi

namespace mochi::solver {
MOCHI_API void SetTimeIntegratorState(
    entt::registry& reg,
    Span<entt::entity const> actors,
    TimeIntegratorParams const& params,
    int iStage);

/*
 * Pipeline executed before the first time integration stage of the time step. Each actor is
 * responsible for computing its differential variables (if any) at the beginning of the time step.
 */
void PreFirstStageLocalPipeline(entt::registry& reg, CIslandDescendants const& descendants);

/*
 * Pipeline executed before each time integration stage. Each actor is responsible for computing its
 * position and velocity state at the beginning of the time integration stage.
 */
void PreStageLocalPipeline(
    entt::registry& reg,
    CIslandDescendants const& descendants,
    SnleProblem<real>& problem);

MOCHI_API void PostStageLocalPipeline(entt::registry& reg, CIslandDescendants const& descendants);

MOCHI_API TimeIntegratorParams CreateIslandTimeIntegrationParams(
    entt::registry const& reg,
    CIslandDescendants const& descendants,
    IntegrationMethod const& targetMethod);

/*
 * Auxiliary pipeline to update quantities that are a function of the state (aka derived state) that
 * (a) are required for the assembly and (2) are not updated in PostNewSolutionLocalPipeline for
 * performance reasons. These are often expensive-to-compute quantities. Also for performance
 * reasons, this pipeline is only executed if the solution has changed since the previous assembly.
 * Derived state that is a function of the state of only one actor is updated before derived state
 * that is a function of the state of multiple actors, such as collision detection.
 */
void UpdateDerivedStateBeforeAssembly(
    entt::registry& reg,
    GradTarget gradTarget,
    CIslandDescendants const& descendants);

/*
 * Assembles the system of equations for a simulation island at a given state and computes the
 * residual and/or its derivative (Jacobian).
 *
 * This function sets up a non-linear equation (SNLE) problem for the given island, configures
 * time integration (single-stage only), runs the pre-assembly pipelines, and then performs the
 * assembly to produce the requested outputs.
 *
 * @param island     The entity handle identifying the simulation island to assemble.
 * @param descendants The pre-computed list of actors and typed entities within the island.
 * @param outRes     Output view for the assembled residual vector. If empty, residual assembly
 *                   is skipped.
 * @param outDRes    Output view for the assembled residual derivative (Jacobian) matrix. If
 *                   empty, derivative assembly is skipped.
 */
void AssembleIsland(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants,
    ColumnVectorView<real> outRes,
    MatrixView<real> outDRes);

// Step in time all the actors in a simulation island.
bool StepIslandNewtonAsync(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants);

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
void AssembleIslandPipeline(
    entt::registry& reg,
    entt::entity island,
    AssemblyParams const& params,
    SnleProblem<real>& problem);

/*
 * Pipeline executed after updating the solution vector of the non-linear problem. It MUST update
 * the position components of the state (aka position state) of all actors to make it consistent
 * with the new solution. It may OPTIONALLY update the velocity components of the state (aka
 * velocity state) and other quantities that are a function of the state (aka derived state). Any
 * velocity state and derived state that is required for assembly and not updated in
 * PostNewSolutionLocalPipeline MUST be updated in UpdateDerivedStateBeforeAssembly.
 */
void PostNewSolutionLocalPipeline(
    entt::registry& reg,
    entt::entity island,
    ColumnVectorView<real> outSolution);

/*
 * Pipeline executed after updating the increment vector of the non-linear problem. It MUST update
 * the position components of the state (aka position state) of all actors to make it consistent
 * with the new solution. It may OPTIONALLY update the velocity components of the state (aka
 * velocity state) and other quantities that are a function of the state (aka derived state). Any
 * velocity state and derived state that is required for assembly and not updated in
 * PostNewIncrementLocalPipeline MUST be updated in UpdateDerivedStateBeforeAssembly.
 */
void PostNewIncrementLocalPipeline(
    entt::registry& reg,
    entt::entity island,
    ColumnVectorView<real const> increment,
    ColumnVectorView<real> outSolution);

/*
 * Subpipeline to update Jacobians necessary for contact Jacobians. Called from
 * UpdateDerivedStateBeforeAssembly().
 */
MOCHI_API void UpdateJacobiansSubpipeline(
    TaskSemaphore& sem,
    entt::registry& reg,
    GradTarget gradTarget,
    CIslandDescendants const& descendants);
} // namespace mochi::solver
