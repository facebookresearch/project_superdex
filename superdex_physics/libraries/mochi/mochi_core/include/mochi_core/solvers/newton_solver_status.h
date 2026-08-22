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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/nonlinear_solver_params.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/time.h>

#include <string>

namespace mochi {

/** @brief Struct storing information on the status of @ref NewtonSolver::Solve. */
template <typename T>
struct NewtonSolverStatus {
  /// @brief Initial merit.
  double merit0 = 0.0;

  /// @brief Current merit.
  double merit = 0.0;

  /// @brief Initial residual norm (global, unweighted).
  T resNorm0 = T(0);

  /// @brief Residual norm with the solution at the previous iteration (global, unweighted).
  T prevResNorm = T(0);

  /// @brief Residual norm with the current solution (global, unweighted).
  T resNorm = T(0);

  /// @brief Per-actor initial weighted residual norms |r0|_W. Only populated if @ref
  /// NewtonSolverParams::convergenceMode is @ref NonLinearSolverConvergenceMode::PerActorWeighted.
  DynamicArray<T> actorResidualWeightedNorm0;

  /// @brief Per-actor convergence status. Only populated if @ref
  /// NewtonSolverParams::convergenceMode is @ref NonLinearSolverConvergenceMode::PerActorWeighted.
  DynamicArray<ConvergenceStatus> actorConvergence;

  ColumnVector<T> dxSolve;

  /// @brief Number of iterations done.
  int numIterDone = 0;

  /// @brief Iterations in the last line search.
  int numLastLSIterDone = 0;

  /// @brief Total number of line-search iterations across all Newton iterations.
  int totalNumLSIterDone = 0;

  /// @brief Total number of iterations of the linear solver across all Newton iterations.
  int totalNumLinearIterDone = 0;

  /// @brief Whether the most recent Newton iteration improved the figure of merit monitored by the
  /// line search.
  bool improvedInLastIter = false;

  /// @brief Number of iterations since the last DResidual assembly.
  int itersSinceDResidualAssembly = 0;

  /// @brief Reason why the solver stopped.
  std::string stopReasonStr;

  /// @brief Convergence status after the solve.
  ConvergenceStatus convergence = ConvergenceStatus::None;

  /// @brief Maximum error in the residual norm across Newton iterations. Computed only if
  /// @ref NewtonSolverParams::consistencyResNorm is true.
  T resNormError = T(0);

  /// @brief Solver timer.
  Timer timer;
};

} // namespace mochi
