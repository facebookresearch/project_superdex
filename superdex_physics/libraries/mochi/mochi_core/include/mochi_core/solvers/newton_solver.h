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

#include <mochi_core/linear_algebra/low_rank_augmented_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/solvers/line_search.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/solvers/newton_solver_status.h>

#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace mochi {

/** @brief Newton-Raphson solver. */
template <typename T>
class NewtonSolver {
 public:
  using Problem = SnleProblem<T>;
  using Status = NewtonSolverStatus<T>;
  using Params = NewtonSolverParams;

  /** @brief Functor defining the signature of all possible stop criteria. */
  using StopCriterion =
      std::function<bool(Problem const& problem, Status& status, Params const& params)>;

  /** @brief Structure containing stopping criteria. */
  struct StopCriterionType {
    /** @brief Check residual norm convergence. */
    static bool StopCriterion_ResNorm(Problem const& problem, Status& status, Params const& params);

    /** @brief Check if the number of iterations done is >= params.maxIters. */
    static bool
    StopCriterion_MaxIters(Problem const& problem, Status& status, Params const& params);

    /** @brief Stop criterion for elapsed time exceeding params.maxElapsedTime. */
    static bool
    StopCriterion_MaxElapsedTime(Problem const& problem, Status& status, Params const& params);

    /** @brief Check if the last step increment is <= params.dxRelTol. */
    static bool
    StopCriterion_RelStepNorm(Problem const& problem, Status& status, Params const& params);

    /**
     * @brief Stop criterion based on lack of improvement.
     *
     * @note Only enabled if params.stopIfNoImprovement is 'true'. If enabled, the solver stops if
     * there is no improvement with respect to the previous Newton iteration in the figure of merit
     * monitored by the line search.
     */
    static bool
    StopCriterion_NoImprovement(Problem const& problem, Status& status, Params const& params);
  };

  /** @brief Constructor. */
  NewtonSolver(
      Params const& params = NewtonSolverParams(),
      std::shared_ptr<PreconditionerRecyclingManager<T>> preconditionerRecyclingMgr = nullptr);

  /** @brief Sets the solver parameters. */
  void SetParams(Params const& params);

  /** @brief Gets the solver parameters. */
  Params const& GetParams() const;

  /** @brief Solves the specified system of non-linear equations. */
  Status Solve(Problem& problem);

 protected:
  /**
   * @brief Workspace with dynamic memory for the solve. It minimizes dynamic memory allocation
   * based on the solver parameters.
   */
  struct Workspace {
    /**
     * @brief Resize the workspace to the requested size.
     *
     * @note For performance reasons, only the variables that are required for the given solver
     * parameters are resized.
     * @note Matrix::Resize is no-op if the matrix already has the requested size.
     */
    void Resize(size_t nIn, Params const& params) {
      if (n != nIn) {
        // Reset the workspace to avoid non-empty variables with incorrect size.
        this->~Workspace();
        new (this) Workspace;
      }
      n = nIn;
      auto iN = static_cast<int>(nIn);
      if (params.gradientDescentFallback) {
        residual.Resize(iN);
      }
      if (params.solverType == NonLinearSolverType::BFGS ||
          params.solverType == NonLinearSolverType::SR1) {
        residual.Resize(iN);
        prevIterResidual.Resize(iN);
        deltaRes.Resize(iN);
        deltaSol.Resize(iN);
        y.Resize(iN);
        if (params.solverType == NonLinearSolverType::BFGS) {
          U.Resize(iN, 2);
          V.Resize(iN, 2);
        } else if (params.solverType == NonLinearSolverType::SR1) {
          U.Resize(iN, 1);
          V.Resize(iN, 1);
        }
      } else {
        MOCHI_ASSERT(params.solverType == NonLinearSolverType::Newton, "Unexpected solver type.");
      }
    }

    void SaveDeltaSol(Problem const& problem) {
      if (!deltaSol.empty()) { // Only if needed.
        deltaSol = problem.GetIncrement();
      }
    }

    void AdvanceResidual(Problem const& problem, bool savePrevious) {
      if (!residual.empty()) { // Only if needed.
        if (savePrevious && !prevIterResidual.empty()) { // Only if needed.
          prevIterResidual = residual;
        }
        residual = problem.GetResidual();
      }
    }

    size_t n = 0;
    ColumnVector<T> residual; // Residual in the current iteration.
    ColumnVector<T> prevIterResidual; // Residual in the previous iteration.
    ColumnVector<T> deltaRes; // deltaRes = residual - prevIterResidual
    ColumnVector<T> deltaSol; // deltaSol = solution - prevIterSolution
    ColumnVector<T> y; // Action of the linear operator on 'deltaSol'.
    Matrix<T> U; // Left vectors for the next low-rank update.
    Matrix<T> V; // Right vectors for the next low-rank update.
  };

  /**
   * @brief Matrix and linear operator types that can be used for the linear solve. All of them are
   * views.
   */
  using AnyLinearOperator = std::variant<
      BlockSparseMatrixView<T const, 3>,
      BlockSparseMatrixView<T const, 4>,
      SparseMatrixView<T const>,
      MatrixView<T const>,
      IslandOperators<T>,
      LowRankAugmentedMatrix<BlockSparseMatrixView<T const, 3>>,
      LowRankAugmentedMatrix<BlockSparseMatrixView<T const, 4>>,
      LowRankAugmentedMatrix<SparseMatrixView<T const>>,
      LowRankAugmentedMatrix<MatrixView<T const>>,
      LowRankAugmentedMatrix<IslandOperators<T>>>;

  /**
   * @brief Prepare the linear operator for the linear solve.
   *
   * @details Resets the operator from the assembled DResidual, or applies a quasi-Newton
   * low-rank correction when reusing a previous DResidual.
   */
  void PrepareLinearOperator(
      Problem const& problem,
      Status const& status,
      bool useIslandOperators,
      bool projectPsd,
      AnyLinearOperator& outLinOp);

  /** @brief Reset the linear operator from the assembled DResidual. */
  void
  ResetLinearOperator(Problem const& problem, bool useIslandOperators, AnyLinearOperator& outLinOp);

  /** @brief Apply a quasi-Newton low-rank correction to the linear operator. */
  void AddLowRankUpdate(bool projectPsd, AnyLinearOperator& outLinOp);

  /** @brief Takes a step in the direction of the current solution increment depending on step
   * parameters. */
  bool TakeStep(Problem& problem, Status& status, bool takeFinalLsStepIfNoImprovement) const;

  /** @brief Test exit of the iterative solver. */
  bool IsSolverDone(
      std::vector<StopCriterion> const& stopCriteria,
      ColumnVectorView<T const> initialSolution,
      Problem const& problem,
      Status& status) const;

  /**
   * @brief Sets the relative tolerance of the linear solver (aka forcing term).
   *
   * @see [Choosing the forcing terms in an inexact Newton method (Eisenstat and Walker,
   * 1996)](https://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR94463.pdf)
   */
  void SetLinearSolverRelativeTolerance(Status const& status, T linearResidualNorm);

  NewtonSolverParams _params;
  std::unique_ptr<LinearSolver<T>> _linearSolver;
  std::vector<StopCriterion> _stopCriteria;
  std::function<bool(Problem& problem, Status& status, LineSearchParams const& lsParams)>
      _lineSearch;
  Workspace _workspace = {};
  std::shared_ptr<PreconditionerRecyclingManager<T>> _preconditionerRecyclingMgr = nullptr;
};

#if MOCHI_USE_EXTERN_TEMPLATE
extern template class NewtonSolver<real>;
#endif // MOCHI_USE_EXTERN_TEMPLATE

} // namespace mochi
