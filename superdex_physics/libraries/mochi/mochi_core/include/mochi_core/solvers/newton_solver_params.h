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

#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/solvers/nonlinear_solver_params.h>
#include <mochi_core/utils/eval_params.h>
#include <mochi_core/utils/time.h>

namespace mochi {

/**
 * Struct storing all parameters affecting the line search.
 */
// clang-format off
struct LineSearchParams {
  LineSearchType type = NonLinearSolverParams{}.lineSearchType;           // Type of line search to use.
  int maxIter = NonLinearSolverParams{}.lineSearchMaxIter;                // Maximum number of line search iterations. Must be >= 1, except if 'type' is 'kNone'.
  real alpha = NonLinearSolverParams{}.lineSearchAlpha;                   // Step reduction factor between line search iterations.
  real wolfe1 = NonLinearSolverParams{}.lineSearchWolfe1;                 // Wolfe first parameter. Only applies to Wolfe line search types.
  real wolfe2 = NonLinearSolverParams{}.lineSearchWolfe2;                 // Wolfe second parameter. Only applies to Wolfe line search types.
  real maxRelIncrease = NonLinearSolverParams{}.lineSearchMaxRelIncrease; // Maximum relative increase in the merit to accept the step. Only applies to Simple line search type.
};
// clang-format on

/**
 * Struct storing all parameters affecting a Newton solve.
 */
// clang-format off
struct NewtonSolverParams {
  NonLinearSolverType solverType = NonLinearSolverParams{}.solverType;
  int dResidualAssemblyPeriod = NonLinearSolverParams{}.dResidualAssemblyPeriod; // Every how many iterations to assemble the DResidual. It may be >1 in quasi-Newton methods such as BFGS and SR1.

  // Stop criteria. The solver terminates when any of the criteria is satisfied.
  int maxIter = NonLinearSolverParams{}.maxIter;                    // Maximum number of iterations
  TimeSpan maxElapsedTime = TimeSpanFromSeconds(NonLinearSolverParams{}.maxElapsedTimeSeconds); // Maximum elapsed time. 0 = "no max elapsed time limit".
  NonLinearSolverConvergenceMode convergenceMode = NonLinearSolverParams{}.convergenceMode; // Convergence monitoring mode.
  real absTolRes = NonLinearSolverParams{}.absTol;                  // Absolute tolerance for convergence criterion.
  real relTolRes = NonLinearSolverParams{}.relTol;                  // Relative tolerance for convergence criterion.
  real solRelTol = NonLinearSolverParams{}.relStepTol;              // Relative tolerance on solution step. See @ref NonLinearSolverParams::relStepTol.
  bool stopIfNoImprovement = NonLinearSolverParams{}.stopIfNoImprovement; // Boolean for whether to stop if there is no improvement w.r.t. the previous Newton iteration in the figure of merit monitored by the line search

  // Retries and fallbacks
  PsdProjectionMode psdProjMode = NonLinearSolverParams{}.psdProjMode; // PSD projection mode
  bool gradientDescentFallback = NonLinearSolverParams{}.gradientDescentFallback; // Whether to fall back to the gradient descent direction if the Newton direction fails
  SaturationHessianParams fittedSaturationHessian = ExperimentalEvalParams{}.fittedSaturationHessian; // Per-pathway control of fitted Hessian use in force saturation terms

  // Explosion control
  bool explosionControl = NonLinearSolverParams{}.explosionControl; // Heuristically prevents explosions
  real absDivTol = NonLinearSolverParams{}.absDivTol;               // Absolute divergence tolerance
  real relDivTol = NonLinearSolverParams{}.relDivTol;               // Relative (wrt to initial iters Res) divergence tolerance

  // Line search
  LineSearchParams lineSearch;                                      // Line search parameters

  // Linear tolerance strategy
  LinearToleranceStrategy linearToleranceStrategy = NonLinearSolverParams{}.linearToleranceStrategy; // Strategy to select the relative tolerance of the linear solver (aka forcing term)

  // Verbosity level for logging output.
  VerbosityLevel verbosity = NonLinearSolverParams{}.verbosity;

  // Consistency check of the residual and dresidual, executed by finite-difference validation (for debugging only).
  bool consistencyResDRes = false;
  int consistencyResDResLogEntries = 10;                            // Number of residual/dresidual entries to print during consistency checks.
  real consistencyResDResStep = 1e-4_r;                             // Finite difference step size for consistency checks.

  // Consistency check of the residual norm, executed by finite-difference validation (for debugging only).
  bool consistencyResNorm = false;
  real consistencyResNormStep = ExperimentalEvalParams{}.consistencyResNormStep;

  KrylovSolverParams lParams;                                       // Linear solver params
};
// clang-format on

} // namespace mochi
