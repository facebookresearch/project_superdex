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
#include <mochi_core/solvers/linear_solver_params.h>

#include <type_traits>

namespace mochi {

/// @brief Enum for choosing which directions to keep in recycling subspace iterative solvers.
enum class RecyclingAlgorithm : uint8_t {
  LiFo = 0, // Last In First Out - keeps the initial directions
};

/**
 * Linear solver parameters.
 */
inline constexpr int kDefaultLinearSolverMaxIter = 500;
struct KrylovSolverParams {
  // Absolute convergence tolerance for the (possibly preconditioned) residual norm. Only used for
  // iterative solvers.
  double absTol = static_cast<double>(LinearSolverParams{}.absTol);
  // Relative convergence tolerance for the decrease in the (possibly preconditioned) residual norm.
  // Only used for iterative solvers.
  double relTol = static_cast<double>(LinearSolverParams{}.relTol);
  // Relative divergence tolerance, i.e. how much the (possibly preconditioned) residual norm can
  // increase before the solver concludes that the method is diverging. Only used for iterative
  // solvers.
  double relDivTol = static_cast<double>(LinearSolverParams{}.relDivTol);
  // Maximum number of iterations. Only used for iterative solvers.
  int maxIter = kDefaultLinearSolverMaxIter; // LinearSolverParams defaults to "Auto".
  // Krylov subspace size triggering a restarting (used in GMRes)
  int restartSize = LinearSolverParams{}.restartSize;
  // Solver type.
  LinearSolverType solverType = LinearSolverType::CG; // LinearSolverParams defaults to Auto
  // Preconditioner type. Only used for iterative solvers. The default is not inherited from
  // LinearSolverParams since some preconditioners (e.g. PerActor) are not supported for all
  // operator types.
  PreconditionerType preconditionerType = PreconditionerType::Jacobi;
  // Norm to monitor convergence and divergence. Only used for CG-like iterative solvers. Unused for
  // GMRES (in which the implicit L2-norm is always used) and direct solvers.
  LinearSolverConvergenceNorm normType = LinearSolverParams{}.normType;
  // Maximum size (dimensionality) of the recycling subspace. Only used for augmented solvers.
  int maxSubspaceSize = 32;
  // Target number of directions to recycle per solve. Only used for augmented solvers.
  int numRecyclingDir = 8;
  // Algorithm to select the recycling directions. Only used for augmented solvers.
  RecyclingAlgorithm recyclingAlgo = RecyclingAlgorithm::LiFo;
  // Whether to abort the solve if the matrix is detected not to be symmetric positive definite.
  // Only applies to symmetric positive-definite solvers such as CG and variations.
  bool abortIfNotSpd = LinearSolverParams{}.abortIfNotSpd;
  // Number of solves after which the preconditioner is updated during preconditioner recycling. A
  // value of 1 means the preconditioner is updated every solve. Higher values allow reusing the
  // same preconditioner across multiple solves. Recycling is not supported with CUDA solvers.
  int preconditionerLifespan = 1;
  // Verbosity level for logging output.
  VerbosityLevel verbosity = LinearSolverParams{}.verbosity;
};

/**
 * Output structure of the linear solver.
 */
struct LinearSolverStatus {
  // Number of iterations done by the solver. "maxIter+1" is used to indicate that the maximum
  // number of iterations was reached without convergence. Only populated with iterative solvers.
  int numIterDone = 0;
  // Norm of the residual at the end of the solve. Only populated with iterative solvers.
  double residualNorm = {};
  // Relative norm of the residual at the end vs. at the beginning of the solve. Only populated with
  // iterative solvers.
  double relativeResidualNorm = {};
  // Did the solver converge?
  bool converged = false;
};

/**
 * Recycling parameters structure. Used in recycling subspace methods.
 */
struct RecyclingParams {
  explicit RecyclingParams(KrylovSolverParams const& params)
      : maxSubspaceSize(params.maxSubspaceSize),
        incrDirections(params.numRecyclingDir),
        algorithm(params.recyclingAlgo) {
    MOCHI_ASSERT_VERBOSE(
        (incrDirections >= 0) && (maxSubspaceSize >= incrDirections),
        "Inconsistent recycling parameters.");
  }

  explicit RecyclingParams(
      int maxSubspaceSizeIn,
      int incrDirectionsIn,
      RecyclingAlgorithm algorithmIn)
      : maxSubspaceSize(maxSubspaceSizeIn),
        incrDirections(incrDirectionsIn),
        algorithm(algorithmIn) {
    MOCHI_ASSERT_VERBOSE(
        (incrDirections >= 0) && (maxSubspaceSize >= incrDirections),
        "Inconsistent recycling parameters.");
  }

  /// @brief Maximum dimensionality of the recycling subspace.
  int const maxSubspaceSize = {};
  /// @brief Number of search directions to retain per solve.
  int const incrDirections = {};
  /// @brief Strategy for retaining search directions during one solve.
  RecyclingAlgorithm const algorithm = {};
};

/**
 * Subspace recycling status structure. Used in recycling subspace methods to communicate the
 * recycling status across linear solves.
 */
template <typename T>
struct SubspaceRecyclingStatus {
  using Scalar = std::remove_const_t<T>;
  /// @brief Matrix whose column vectors span the current recycling subspace. The column vectors are
  /// not necessarily orthogonal.
  Matrix<Scalar> V;
  /// @brief Matrix whose column vectors span the image of the current recycling subspace under the
  /// linear operator. The column vectors are not necessarily orthogonal. The column vectors may be
  /// obsolete (and will be recomputed in the next solve) if the linear operator has changed since
  /// the previous solve.
  Matrix<Scalar> AV;
  /// @brief Current dimensionality of the recycling subspace.
  int subspaceSize = 0;
  /// @brief Flag to indicate whether the linear operator has changed since the previous solve. If
  /// the operator hasn't changed, some compute is saved by setting this flag to false.
  bool hasOperatorChanged = true;
};

} // namespace mochi
