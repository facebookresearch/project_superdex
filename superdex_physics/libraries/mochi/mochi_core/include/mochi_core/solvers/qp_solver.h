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

#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/linear_solver_params.h>
#include <mochi_core/solvers/newton_solver.h>
#include <mochi_core/solvers/nonlinear_solver_params.h>
#include <mochi_core/solvers/snle_problem.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/verbosity_params.h>

#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

namespace mochi {
enum class QPSolverConvergenceStatus {
  InnerSolverFailed,
  MaxIterationsReached,
  FeasibleSolutionFound,
  InfeasibilityDetected,
  Count,
};

struct QPSolverParams {
  // The following two conditions hold simultaneously in feasible problem convergence:
  // The primal feasible stopping criteria of absolute constraint violation
  // ||(Cx - d)_-||_∞ < ε_c (must be strictly positive)
  real absTolConstraint = 1e-4_r;

  // The tolerance for checking the KKT condition:
  // The primal feasible stopping criteria of lagrangian gradient
  // ||Hx + g + C^T lambda||_∞ < ε_L (this condition is tested only when positive)
  real absTolLagrangianGradient = 0_r;

  // The tolerance for checking the KKT condition:
  // The primal feasibility stopping criteria of complementarity condition violation
  // ||lambda^T (Cx-d)||_∞ < ε_C (this condition is tested only when positive)
  real absTolComplementarity = 0_r;

  // The primal feasible stopping criteria of relative or absolute lambda change,
  // whichever hits first
  // ||lambda-lambda_prev||_∞ < ε_l (must be strictly positive)
  real tolLambda = 1e-4_r;

  // According to the Farka's lemma,
  // the following two conditions hold simultaneously in infeasible problems:
  //   lim ||C^T (lambda/||lambda||)||_∞ = 0
  //   lim d^T (lambda/||lambda||) = ∞
  // Therefore, we check the following two conditions to detect infeasibility:
  //   ||C^T (lambda/||lambda||)||_∞ < ε_d (absTolDuality)
  //   d^T (lambda/||lambda||) > δ_infeasibility (absTolInfeasibility)

  // The dual convergence stopping criteria (must be strictly positive)
  real absTolDuality = 1e-2_r;

  // The dual infeasibility criteria (must be strictly positive)
  real absTolInfeasibility = 1e-1_r;

  // Whether to normalize all constraint rows to have unit norm
  bool normalizeConstraint = true;

  // Maximal outer iterations
  int maxIterations = 1e3;

  // Verbosity level for logging output.
  VerbosityLevel verbosity = VerbosityLevel::Warning;
};

// Small-scale Quadratic Programming (QP) Solver using the Augmented Lagrangian Method (ALM).
// The objective Hessian and general linear constraint matrix must use matching storage: either both
// dense or both sparse. When using sparse storage, the Hessian sparsity pattern must include all
// entries needed by the constraint outer products added during assembly.
template <typename T>
class QPSolver {
 public:
  using Problem = SnleProblem<T>;

  // Interface for a general affine constraint:
  //    coeff * x + bias >= 0
  // An optional positive scale factor is applied for numerical conditioning without changing the
  // constraint's feasible set.
  struct AffineConstraint {
    AffineConstraint(
        int row,
        AnyMatrix<T> const* ATDenseOrASparse,
        int index,
        T bias,
        bool isGE,
        T scale = T(1))
        : rowIndex(row), _bias(bias), _isGE(isGE), _scale(scale) {
      MOCHI_ASSERT_VERBOSE(IsFinite(_scale) && _scale > T(0), "Scale must be positive and finite.");
      if (ATDenseOrASparse != nullptr) {
        if (std::holds_alternative<Matrix<T>>(*ATDenseOrASparse)) {
          auto const& ATDense = std::get<Matrix<T>>(*ATDenseOrASparse);
          _coeff.template emplace<MatrixView<T const>>(
              ATDense.Col(index).data(), ATDense.Rows(), 1);
        } else {
          MOCHI_ASSERT_VERBOSE(
              std::holds_alternative<SparseMatrix<T>>(*ATDenseOrASparse),
              "Invalid constraint type.");
          auto const& ASparse = std::get<SparseMatrix<T>>(*ATDenseOrASparse);
          auto ptrs = ASparse.Pointers();
          auto rowStart = ptrs[index];
          auto nnzInRow = ptrs[index + 1] - rowStart;
          _rowPtr = {0, static_cast<int>(nnzInRow)};
          _coeff.template emplace<SparseMatrixView<T const>>(
              ASparse.Cols(),
              Span<int const>{_rowPtr.data(), 2},
              ASparse.Indices(index),
              ASparse.Values(index));
        }
      } else {
        _colIndex = index;
      }
    }

    T ComputeLhs(ColumnVectorView<T const> x) const {
      T coeffDotX{};
      if (_colIndex >= 0) {
        coeffDotX = x[_colIndex];
      } else if (std::holds_alternative<MatrixView<T const>>(_coeff)) {
        coeffDotX = std::get<MatrixView<T const>>(_coeff).Dot(x);
      } else {
        MOCHI_ASSERT_VERBOSE(
            std::holds_alternative<SparseMatrixView<T const>>(_coeff), "Invalid constraint type.");
        std::get<SparseMatrixView<T const>>(_coeff).Apply(x, ColumnVectorView<T>(&coeffDotX, 1));
      }
      return (coeffDotX + _bias) * _scale * (_isGE ? T(1) : T(-1));
    }

    void AddScaledCoeffTransposed(T coef, ColumnVectorView<T> outRes) const {
      if (coef == T(0)) {
        // Prevent computation for constraints that are never active
        return;
      }
      T const s = coef * _scale * (_isGE ? T(1) : T(-1));
      if (_colIndex >= 0) {
        outRes[_colIndex] += s;
      } else if (std::holds_alternative<MatrixView<T const>>(_coeff)) {
        outRes += std::get<MatrixView<T const>>(_coeff) * s;
      } else {
        MOCHI_ASSERT_VERBOSE(
            std::holds_alternative<SparseMatrixView<T const>>(_coeff), "Invalid constraint type.");
        auto const& coeffSparse = std::get<SparseMatrixView<T const>>(_coeff);
        auto indices = coeffSparse.Indices(0);
        auto values = coeffSparse.Values(0);
        for (size_t i = 0; i < indices.size(); ++i) {
          outRes[indices[i]] += values[i] * s;
        }
      }
    }

    void AddScaledHessian(T coef, AnyMatrixView<T> outDRes) const {
      T const s = coef * _scale * _scale;
      if (std::holds_alternative<MatrixView<T>>(outDRes)) {
        auto& outDResDense = std::get<MatrixView<T>>(outDRes);
        if (_colIndex >= 0) {
          outDResDense(_colIndex, _colIndex) += s;
        } else {
          MOCHI_ASSERT_VERBOSE(
              std::holds_alternative<MatrixView<T const>>(_coeff), "Invalid constraint type.");
          auto const& coeffDense = std::get<MatrixView<T const>>(_coeff);
          outDResDense += coeffDense * coeffDense.Transpose() * s;
        }
      } else {
        MOCHI_ASSERT_VERBOSE(
            std::holds_alternative<SparseMatrixView<T>>(outDRes), "Invalid constraint type.");
        auto& outDResSparse = std::get<SparseMatrixView<T>>(outDRes);
        if (_colIndex >= 0) {
          outDResSparse.AddValue(_colIndex, _colIndex, s);
        } else {
          MOCHI_ASSERT_VERBOSE(
              std::holds_alternative<SparseMatrixView<T const>>(_coeff),
              "Invalid constraint type.");
          auto const& coeffSparse = std::get<SparseMatrixView<T const>>(_coeff);
          auto indices = coeffSparse.Indices(0);
          auto values = coeffSparse.Values(0);
          for (size_t i = 0; i < indices.size(); ++i) {
            for (size_t j = 0; j < indices.size(); ++j) {
              outDResSparse.AddValue(indices[i], indices[j], values[i] * values[j] * s);
            }
          }
        }
      }
    }

    T Bias() const {
      return _bias * _scale * (_isGE ? T(1) : T(-1));
    }

    int rowIndex = -1;

   private:
    std::array<int, 2> _rowPtr{};
    AnyMatrixView<T const> _coeff{};
    int _colIndex = -1;
    T _bias{};
    bool _isGE{};
    T _scale = T(1);
  };

  // Create a QP with n variables and m constraints
  QPSolver(int n, int m) {
    // Objective
    _gradient.Resize(n);

    // Initialize box constraints to be unbounded
    _lb.resize(n, -std::numeric_limits<T>::infinity());
    _ub.resize(n, std::numeric_limits<T>::infinity());

    // Initialize constraints
    _Alb.resize(m, -std::numeric_limits<T>::infinity());
    _Aub.resize(m, std::numeric_limits<T>::infinity());

    // Initialize lambda: altogether, we have 2 * m + 2 * n constraints
    //    the first  m are general linear constraints Ax-Alb>=0
    //    the second m are general linear constraints Aub-Ax>=0
    //    the next n are lower box constraints
    //    the last n are upper box constraints
    // These constraints are stacked together into:
    //    Cx-d>=0
    // With C = [A -A, I, -I]^T and d = [Alb, -Aub, lb, -ub]
    // We only store matrices A,b and the matrices C and d are evaluated on the fly
    _ctLambda.Resize(n);
    _lambda.Resize((m + n) * 2);
    _prevLambda.Resize((m + n) * 2);
    _hx.Resize(n);
    _res.Resize(n);

    // Default inner solver params
    auto params = _innerSolver.GetParams();
    params.convergenceMode = NonLinearSolverConvergenceMode::Global;
    params.absTolRes = 1e-4_r;
    params.relTolRes = 1e-8_r;
    params.psdProjMode = PsdProjectionMode::Never;
    params.maxIter = 1e3;
    params.lParams.preconditionerType = PreconditionerType::None;
    params.lParams.solverType = LinearSolverType::LDLT;
    params.lineSearch.type = LineSearchType::WolfeWeak;
    SetNewtonSolverParams(params);
  }

  int GetNumVariables() const {
    return isize(_lb);
  }

  int GetNumConstraints() const {
    return isize(_Alb);
  }

  // Get and Set the parameters for the inner Newton's solver
  NewtonSolverParams const& GetNewtonSolverParams() const {
    return _innerSolver.GetParams();
  }

  void SetNewtonSolverParams(NewtonSolverParams const& params) {
    auto p = params;
    if (params.convergenceMode != NonLinearSolverConvergenceMode::Global) {
      // QPSolver assembles a single residual. It requires using global convergence mode.
      MOCHI_LOG_WARNING(
          "QPSolver only supports global convergence mode (NewtonSolverParams::convergenceMode = NonLinearSolverConvergenceMode::Global). "
          "The provided convergence mode will be overridden.");
      p.convergenceMode = NonLinearSolverConvergenceMode::Global;
    }
    _innerSolver.SetParams(p);
  }

  // Get and Set the parameters for the outer ALM solver
  QPSolverParams const& GetQPSolverParams() const {
    return _params;
  }

  void SetQPSolverParams(QPSolverParams const& params) {
    _params = params;
  }

  // Set the box constraint bounds for the ith variables, return error if index out of range
  void SetBounds(int i, T lb, T ub, Error& error) {
    MOCHI_ERROR_RETURN(error);
    int n = GetNumVariables();
    MOCHI_ERROR_IF(i < 0 || i >= n, error, "Index out of range in SetBounds.");
    MOCHI_ERROR_IF(lb > ub, error, "Infeasible bounds.");
    MOCHI_ERROR_RETURN(error);

    _lb[i] = lb;
    _ub[i] = ub;
  }

  // Set the general linear constraint bounds for the ith row, return error if index out of range
  void SetABounds(int i, T lb, T ub, Error& error) {
    MOCHI_ERROR_RETURN(error);
    int m = GetNumConstraints();
    MOCHI_ERROR_IF(i < 0 || i >= m, error, "Index out of range in SetABounds.");
    MOCHI_ERROR_IF(lb > ub, error, "Infeasible bounds.");
    MOCHI_ERROR_RETURN(error);

    _Alb[i] = lb;
    _Aub[i] = ub;
  }

  // Set dense constraint coefficient matrix in row major format. Returns an error on incorrect size
  // or if a sparse Hessian has already been set.
  void SetA(MatrixView<T const> A, Error& error) {
    int n = GetNumVariables();
    int m = GetNumConstraints();
    MOCHI_ERROR_IF(A.Rows() != m || A.Cols() != n, error, "Incorrect matrix size.");
    MOCHI_ERROR_IF(
        _hessian.has_value() && !std::holds_alternative<Matrix<T>>(*_hessian),
        error,
        "Constraint matrix and Hessian storage types must match.");
    MOCHI_ERROR_RETURN(error);
    _ATDenseOrASparse = A.Transpose();
  }

  // Set sparse constraint coefficient matrix in row major format. Returns an error on incorrect
  // size or if a dense Hessian has already been set.
  void SetA(SparseMatrixView<T const> A, Error& error) {
    int n = GetNumVariables();
    int m = GetNumConstraints();
    MOCHI_ERROR_IF(A.Rows() != m || A.Cols() != n, error, "Incorrect matrix size.");
    MOCHI_ERROR_IF(
        _hessian.has_value() && !std::holds_alternative<SparseMatrix<T>>(*_hessian),
        error,
        "Constraint matrix and Hessian storage types must match.");
    MOCHI_ERROR_RETURN(error);
    _ATDenseOrASparse = A;
  }

  // Set the objective function. The Hessian storage type must match the storage type of A, if A has
  // been set.
  void SetObjective(AnyMatrixView<T const> h, ColumnVectorView<T const> g, Error& error) {
    int n = GetNumVariables();
    MOCHI_ERROR_IF(GetNumRows(h) != n || GetNumCols(h) != n, error, "Incorrect Hessian size.");
    MOCHI_ERROR_IF(g.Rows() != n, error, "Incorrect gradient size.");
    MOCHI_ERROR_RETURN(error);
    if (std::holds_alternative<MatrixView<T const>>(h)) {
      MOCHI_ERROR_IF(
          _ATDenseOrASparse.has_value() && !std::holds_alternative<Matrix<T>>(*_ATDenseOrASparse),
          error,
          "Constraint matrix and Hessian storage types must match.");
      MOCHI_ERROR_RETURN(error);
      _hessian = std::get<MatrixView<T const>>(h);
    } else if (std::holds_alternative<SparseMatrixView<T const>>(h)) {
      MOCHI_ERROR_IF(
          _ATDenseOrASparse.has_value() &&
              !std::holds_alternative<SparseMatrix<T>>(*_ATDenseOrASparse),
          error,
          "Constraint matrix and Hessian storage types must match.");
      MOCHI_ERROR_RETURN(error);
      _hessian = std::get<SparseMatrixView<T const>>(h);
    } else {
      MOCHI_ERROR_SET(error, "Unknown matrix type.");
      MOCHI_ERROR_RETURN(error);
    }
    _gradient = g;
  }

  // ALM works by repeatedly solving inner problem of the following using Gauss-Newton method:
  //    argmin_x f(x) - lambda^T (Cx - d) + rho/2 ||(Cx - d)_-||^2
  // We do not need to update the penalty parameters when f(x) is convex, which is assumed
  // and when the inner solve is sufficiently accurate.
  // The solution is input as initial guess and output.
  QPSolverConvergenceStatus Solve(Span<T> outSolution) {
    MOCHI_ASSERT(outSolution.size() == GetNumVariables(), "Incorrect solution size.");

    // Initialize data structure
    Initialize();

    // Main loop
    for (int iter = 0; iter < _params.maxIterations; iter++) {
      // Solve the inner problem
      _innerProblem.SetSolution(AsConstView(outSolution));
      NewtonSolverStatus<T> status = _innerSolver.Solve(_innerProblem);
      AsView(outSolution) = _innerProblem.GetSolution();
      if (status.convergence != ConvergenceStatus::Converged) {
        return QPSolverConvergenceStatus::InnerSolverFailed;
      }

      // Update the Lagrangian
      UpdateLagrangian();

      // Check for convergence, feasibility
      T maxConsVio{}, maxLGrad{}, maxCompVio{}, maxLChange{};
      bool converged = PrimalConverged(maxConsVio, maxLGrad, maxCompVio, maxLChange);
      bool feasible = !IsInfeasible();

      // Profile
      if (_params.verbosity >= VerbosityLevel::Verbose) {
        MOCHI_LOG(
            "o-Iter=%3d i-Iter=%3d ||min(0,C)||_inf=%.7f, ||gradL||=_inf=%.7f ||L^TC||_inf=%.7f ||dL||_inf=%.7f conv=%d feas=%d",
            iter,
            status.numIterDone,
            maxConsVio,
            maxLGrad,
            maxCompVio,
            maxLChange,
            converged,
            feasible);
      }

      // Termination
      if (converged) {
        return QPSolverConvergenceStatus::FeasibleSolutionFound;
      }
      if (!feasible) {
        return QPSolverConvergenceStatus::InfeasibilityDetected;
      }
    }

    return QPSolverConvergenceStatus::MaxIterationsReached;
  }

 private:
  // Compute _hx = H * x, dispatching on dense vs sparse hessian
  void ApplyHessian(ColumnVectorView<T const> x) {
    MOCHI_ASSERT_VERBOSE(_hessian.has_value(), "Objective has not been set.");
    if (std::holds_alternative<Matrix<T>>(*_hessian)) {
      _hx = std::get<Matrix<T>>(*_hessian) * x;
    } else {
      MOCHI_ASSERT_VERBOSE(
          std::holds_alternative<SparseMatrix<T>>(*_hessian), "Invalid hessian type.");
      std::get<SparseMatrix<T>>(*_hessian).Apply(x, _hx);
    }
  }

  // Iterate over all constraints
  template <typename Fn>
  void ForEachConstraint(Fn const& callback) const {
    int n = GetNumVariables();
    int m = GetNumConstraints();

    MOCHI_ASSERT_VERBOSE(
        m == 0 || _ATDenseOrASparse.has_value(), "Constraint matrix has not been set.");
    for (int i = 0; i < m; ++i) {
      T const scale = _params.normalizeConstraint ? _constraintRowInvNorms[i] : T(1);
      // Ax - Alb >= 0
      if (IsFinite(_Alb[i])) {
        callback(AffineConstraint(i, &*_ATDenseOrASparse, i, -_Alb[i], true, scale));
      }
      // Ax - Aub <= 0
      if (IsFinite(_Aub[i])) {
        callback(AffineConstraint(m + i, &*_ATDenseOrASparse, i, -_Aub[i], false, scale));
      }
    }

    for (int i = 0; i < n; i++) {
      // x - lb >= 0
      if (IsFinite(_lb[i])) {
        callback(AffineConstraint(m * 2 + i, nullptr, i, -_lb[i], true));
      }
      // x - ub <= 0
      if (IsFinite(_ub[i])) {
        callback(AffineConstraint(m * 2 + n + i, nullptr, i, -_ub[i], false));
      }
    }
  }

  // Initialize penalty to be 1 and zero out Lagrangian multiplier.
  // Optionally compute constraint row scaling to have unit norm.
  void Initialize() {
    int n = GetNumVariables();
    int m = GetNumConstraints();

    // Initialize penalty and Lagrangian multipliers
    _penalty = T(1);
    _lambda.SetZero();
    _prevLambda.SetZero();

    // Optionally precompute inverse row norms for constraint normalization.
    if (_params.normalizeConstraint) {
      MOCHI_ASSERT_VERBOSE(
          m == 0 || _ATDenseOrASparse.has_value(), "Constraint matrix has not been set.");
      _constraintRowInvNorms.resize_noinit(m);
      for (int i = 0; i < m; ++i) {
        constexpr T kMin = std::numeric_limits<T>::min();
        if (std::holds_alternative<Matrix<T>>(*_ATDenseOrASparse)) {
          _constraintRowInvNorms[i] =
              T(1) / (std::get<Matrix<T>>(*_ATDenseOrASparse).Col(i).Norm() + kMin);
        } else {
          MOCHI_ASSERT_VERBOSE(
              std::holds_alternative<SparseMatrix<T>>(*_ATDenseOrASparse),
              "Invalid constraint type.");
          auto const& ASparse = std::get<SparseMatrix<T>>(*_ATDenseOrASparse);
          _constraintRowInvNorms[i] = T(1) / (AsConstView(ASparse.Values(i)).Norm() + kMin);
        }
      }
    }

    // Set up the inner problem assembly function for the augmented Lagrangian
    _innerProblem = SnleProblem<T>(
        n,
        n,
        SnleProblemFunctions<T>{
            .assemble =
                [this](SnleProblem<T>& problem, AssemblyParams const& params) {
                  AssembleInnerProblem(problem, params);
                },
            .onPostNewIncrement = [](auto& problem) { problem.solution += problem.increment; }});
  }

  // Update Lagrangian multiplier after each inner solve according to:
  //    lambda_prev = lambda
  //    lambda = max(0, lambda - rho * (Cx - d))
  void UpdateLagrangian() {
    // Save previous lambda
    _prevLambda = _lambda;

    // Get current solution
    auto x = _innerProblem.GetSolution();

    // Compute constraint violations and update lambda
    // lambda = max(0, lambda - rho * (Cx-d))
    ForEachConstraint([&](AffineConstraint const& c) {
      _lambda[c.rowIndex] = Max(T(0), _lambda[c.rowIndex] - _penalty * c.ComputeLhs(x));
    });
  }

  // Primal feasible convergence check:
  //    ||(Cx - d)_-||_∞ < ε_c
  //    ||Hx + g - C^T * lambda||_∞ < ε_L
  //    ||lambda^T (Cx - d)||_∞ < ε_C
  //    ||lambda-lambda_prev||_∞ < ε_l
  bool PrimalConverged(
      T& maxConstraintViolation,
      T& maxLagrangianGradient,
      T& maxComplementViolation,
      T& maxLambdaChange) {
    auto x = _innerProblem.GetSolution();

    // Compute max constraint violation ||(Cx - d)_-||_∞
    maxConstraintViolation = T(0);
    // Compute max component of Lagrangian gradient ||Hx + g + C^T * lambda||_∞
    // We reuse the memory of _hx
    ApplyHessian(x);
    _hx += _gradient;
    // Compute complementarity violation ||lambda^T (Cx - d)||_∞
    maxComplementViolation = T(0);
    // Check lambda change: ||lambda - lambda_prev||_∞
    maxLambdaChange = T(0);
    ForEachConstraint([&](AffineConstraint const& c) {
      auto lhs = c.ComputeLhs(x);
      maxConstraintViolation = Max(maxConstraintViolation, -Min(T(0), lhs));
      c.AddScaledCoeffTransposed(-_lambda[c.rowIndex], _hx);
      maxComplementViolation = Max(maxComplementViolation, Abs(lhs * _lambda(c.rowIndex)));
      maxLambdaChange = Max(maxLambdaChange, Abs(_lambda(c.rowIndex) - _prevLambda(c.rowIndex)));
    });
    maxLagrangianGradient = MaxAbs(_hx.GetConstSpan());

    return (maxConstraintViolation < _params.absTolConstraint) &&
        (_params.absTolLagrangianGradient <= 0_r ||
         maxLagrangianGradient < _params.absTolLagrangianGradient) &&
        (_params.absTolComplementarity <= 0_r ||
         maxComplementViolation < _params.absTolComplementarity) &&
        (maxLambdaChange < _params.tolLambda * Max(T(1), _lambda.Norm()));
  }

  // Farkas certificate test to check for primal infeasibility:
  //    ||C^T (lambda/||lambda||)||_∞ < ε_d (absTolDuality)
  //    d^T (lambda/||lambda||) > δ_infeasibility (absTolInfeasibility)
  bool IsInfeasible() {
    T lambdaNorm = _lambda.Norm();
    if (lambdaNorm < std::numeric_limits<T>::epsilon()) {
      return false; // Cannot determine infeasibility with zero lambda
    }

    // Compute C^T * (lambda / ||lambda||)
    _ctLambda.SetZero();
    // Compute d^T * (lambda / ||lambda||)
    T dtLambdaNormalized = T(0);
    ForEachConstraint([&](AffineConstraint const& c) {
      c.AddScaledCoeffTransposed(_lambda[c.rowIndex], _ctLambda);
      dtLambdaNormalized -= c.Bias() * _lambda[c.rowIndex];
    });
    _ctLambda *= T(1) / lambdaNorm;
    dtLambdaNormalized *= T(1) / lambdaNorm;

    // Compute ||C^T (lambda/||lambda||)||_∞
    T dualNorm = MaxAbs(_ctLambda.GetConstSpan());

    return (dualNorm < _params.absTolDuality) && (dtLambdaNormalized > _params.absTolInfeasibility);
  }

  // Assemble the inner problem for the augmented Lagrangian
  void AssembleInnerProblem(SnleProblem<T>& problem, AssemblyParams const& params) {
    auto x = problem.GetSolution();

    // The augmented Lagrangian objective is:
    // L(x) = 0.5 * x^T H x + g^T x - lambda^T (Cx - d) + (rho/2) ||(Cx - d)_-||^2
    //
    // The gradient is:
    // grad L = H x + g - C^T lambda + rho * C^T (Cx - d)_-
    //
    // The Hessian is:
    // Hess L = H + rho * C^T diag(mask) C
    // where mask[i] = 1 if (Cx - d)_i < 0, else 0

    // Compute constraint violations
    T obj(0);
    if (params.assemObj || params.assemRes) {
      ApplyHessian(x);
    }
    if (params.assemObj) {
      obj = T(0.5) * x.Dot(_hx) + _gradient.Dot(x);
    }
    if (params.assemRes) {
      _res = _hx + _gradient;
    }
    if (params.assemDRes) {
      MOCHI_ASSERT_VERBOSE(_hessian.has_value(), "Objective has not been set.");
      _dRes = *_hessian;
    }
    ForEachConstraint([&](AffineConstraint const& c) {
      T lhs = c.ComputeLhs(x);
      T vio = Min(T(0), lhs);
      if (params.assemObj) {
        obj -= _lambda[c.rowIndex] * lhs;
        obj += T(0.5) * _penalty * vio * vio;
      }
      if (params.assemRes) {
        c.AddScaledCoeffTransposed(_penalty * vio - _lambda[c.rowIndex], _res);
      }
      if (params.assemDRes && vio < T(0)) {
        c.AddScaledHessian(_penalty, AsView(_dRes));
      }
    });
    if (params.assemObj) {
      problem.objective = static_cast<double>(obj);
    }
    if (params.assemRes) {
      problem.actorResiduals.resize(1);
      problem.actorResiduals[0] = std::make_pair(0, &_res);
    }
    if (params.assemDRes) {
      problem.actorMatrices.resize(1);
      problem.actorMatrices[0] = std::make_pair(0, &_dRes);
    }
  }

 private:
  // The objective hessian and gradient at zero
  std::optional<AnyMatrix<T>> _hessian;
  ColumnVector<T> _gradient;

  // The lagrangian multiplier and penalty coefficient
  ColumnVector<T> _lambda, _prevLambda;
  T _penalty = T(1);

  // The general linear constraint
  std::optional<AnyMatrix<T>> _ATDenseOrASparse;
  DynamicArray<T> _Alb, _Aub;
  DynamicArray<T> _constraintRowInvNorms;

  // The box constraint, can be +/- infinity
  DynamicArray<T> _lb, _ub;

  // Helper variable: C^T lambda
  ColumnVector<T> _ctLambda;

  // Handle to the inner problem solver
  NewtonSolver<T> _innerSolver;
  SnleProblem<T> _innerProblem;
  ColumnVector<T> _res, _hx;
  AnyMatrix<T> _dRes;

  // QP solver parameters
  QPSolverParams _params;
};
} // namespace mochi
