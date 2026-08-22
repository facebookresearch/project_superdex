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

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_bsr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_csr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_gmres.h>
#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_pcg.h>
#include <mochi_core/linear_algebra/cuda/cuda_sparse_factorization.h>
#include <mochi_core/linear_algebra/krylov/async_pcg.h>
#include <mochi_core/linear_algebra/krylov/augmented_pcg.h>
#include <mochi_core/linear_algebra/krylov/gmres.h>
#include <mochi_core/linear_algebra/krylov/minres.h>
#include <mochi_core/linear_algebra/krylov/parallel_pcg.h>
#include <mochi_core/linear_algebra/krylov/pcg.h>
#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/krylov/preconditioner_utils.h>
#include <mochi_core/linear_algebra/krylov/sparse_ldlt.h>
#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>
#include <mochi_core/linear_algebra/krylov/tools/custom_matrix_traits.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/ldlt.h>
#include <mochi_core/linear_algebra/low_rank_augmented_matrix.h>
#include <mochi_core/linear_algebra/lu.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/solvers/linear_solver_params.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/task_scheduler.h>

#include <functional>
#include <memory>
#include <type_traits>
#include <variant>

namespace mochi::details {

template <typename T>
struct PrecApplyer {
  std::reference_wrapper<Preconditioner<T>> prec;

  void operator()(ColumnVectorView<T const> x, ColumnVectorView<T> Px) const {
    prec.get().Solve(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<T const> x,
      ColumnVectorView<T> Px,
      ParallelWorkerInfo const& data) const {
    prec.get().ConcurrentSolve(x, Px, data);
  }
};

inline constexpr bool IsCudaSolver(LinearSolverType const& solverType) {
  switch (solverType) {
    case LinearSolverType::CudaCG:
    case LinearSolverType::CudaGMRES:
    case LinearSolverType::ExperimentalCudaSparseCholesky:
    case LinearSolverType::ExperimentalCudaSparseLDLT:
    case LinearSolverType::ExperimentalCudaSparseLU:
      return true;
    case LinearSolverType::CG:
    case LinearSolverType::GMRES:
    case LinearSolverType::AugmentedCG:
    case LinearSolverType::AsyncCG:
    case LinearSolverType::ParallelCG:
    case LinearSolverType::MINRES:
    case LinearSolverType::LDLT:
    case LinearSolverType::LU:
      return false;
    [[unlikely]] default:
      MOCHI_ASSERT(false, "Unsupported solver type.");
      return {};
  }
  static_assert(
      static_cast<int>(LinearSolverType::Count) == 14,
      "Please update the switch statement above if LinearSolverType enumerator changes");
}

inline constexpr bool IsIterativeSolver(LinearSolverType const& solverType) {
  switch (solverType) {
    case LinearSolverType::CG:
    case LinearSolverType::GMRES:
    case LinearSolverType::CudaCG:
    case LinearSolverType::CudaGMRES:
    case LinearSolverType::AugmentedCG:
    case LinearSolverType::AsyncCG:
    case LinearSolverType::ParallelCG:
    case LinearSolverType::MINRES:
      return true;
    case LinearSolverType::LDLT:
    case LinearSolverType::LU:
    case LinearSolverType::ExperimentalCudaSparseCholesky:
    case LinearSolverType::ExperimentalCudaSparseLDLT:
    case LinearSolverType::ExperimentalCudaSparseLU:
      return false;
    [[unlikely]] default:
      MOCHI_ASSERT(false, "Unsupported solver type.");
      return {};
  }
  static_assert(
      static_cast<int>(LinearSolverType::Count) == 14,
      "Please update the switch statement above if LinearSolverType enumerator changes");
}

/// @brief Dispatches a runtime LinearSolverConvergenceNorm to the corresponding compile-time
/// stop criterion type and invokes the given callable with it. The callable must accept a reference
/// to a stop criterion object (use a generic lambda) and return a LinearSolverStatus.
template <typename T, typename Fn>
LinearSolverStatus DispatchConvergenceNorm(KrylovSolverParams const& params, Fn&& fn) {
  static_assert(
      static_cast<int>(LinearSolverConvergenceNorm::Count) == 3,
      "Please update DispatchConvergenceNorm if LinearSolverConvergenceNorm enumerator changes");
  auto const relTol = static_cast<T>(params.relTol);
  auto const absTol = static_cast<T>(params.absTol);
  auto const relDivTol = static_cast<T>(params.relDivTol);
  switch (params.normType) {
    case LinearSolverConvergenceNorm::ResidualL2: {
      krylov::StatusResidualL2<krylov::UsualDot, T> stopCriterion(relTol, absTol, relDivTol);
      return fn(stopCriterion);
    }
    case LinearSolverConvergenceNorm::PreconditionedResidualL2: {
      krylov::StatusPreconditionedResidualL2<krylov::UsualDot, T> stopCriterion(
          relTol, absTol, relDivTol);
      return fn(stopCriterion);
    }
    case LinearSolverConvergenceNorm::ResidualPreconditionerInduced: {
      krylov::StatusResidualPreconditionerInduced<krylov::UsualDot, T> stopCriterion(
          relTol, absTol, relDivTol);
      return fn(stopCriterion);
    }
    default: {
      MOCHI_ASSERT(
          false, "Convergence norm type (%i) not supported.", static_cast<int>(params.normType));
      return {};
    }
  }
}

} // namespace mochi::details

namespace mochi {

/// @brief Linear solver class supporting both direct and iterative methods.
///
/// @tparam T Scalar type (typically float or double).
///
/// @note The input matrix must be a supported matrix or linear operator type. For the particular
/// case of an IslandOperators, the preconditioner is constructed (or updated) in the linear solver
/// for consistency with the case in which the input is a matrix type. That is, the caller is NOT
/// responsible for calling @ref IslandOperators::MakePerActorPrec before @ref LinearSolver::Solve.
/// Doing so would be inefficient (the preconditioner would be computed or updated twice) but not
/// incorrect.
/// @note It provides a unified interface for solving a linear system with various solver and
/// preconditioner types.
/// @note It manages internal state for preconditioner recycling and Krylov subspace recycling.
/// @note All solvers are also available as standalone functions and can be used directly without
/// creating a LinearSolver instance.
///
/// \code{.cpp}
/// // Setup
/// BlockSparseMatrix<real, 3> A{...}; // Linear system matrix
/// ColumnVector<real> b1, b2, x1, x2; // Right-hand sides and solution vectors
///
/// // Configure solver parameters
/// KrylovSolverParams params;
/// params.solverType = LinearSolverType::AugmentedCG;
/// params.preconditionerType = PreconditionerType::BlockJacobi;
///
/// // Create solver
/// LinearSolver<real> solver(params);
///
/// // First solve, with preconditioner block size of 3
/// auto status1 = solver.Solve<3>(A, b1, x1);
///
/// // Second solve, with a new right-hand side and preconditioner block size of 1
/// auto status2 = solver.Solve<1>(A, b2, x2, /*hasOperatorChanged*/ false);
/// @endcode
template <typename T>
class LinearSolver {
  // TODO
  // - Implement iterative refinement for direct solvers.
  // - Use SparseLDLt::Refactorize to avoid recomputing the symbolic factorization if the values
  //   change but the sparsity pattern does not between Solve calls. Such setting is currently not
  //   exposed in the Solve API.
 public:
  static_assert(!std::is_const_v<T>, "Scalar type must not be const");

  /// @brief Minimum number of DoFs to use sparse (instead of dense) LDLt factorization with LDLt
  /// solver.
  /// @note Only applies if the input matrix is sparse or block sparse. For other matrix and
  /// operator types, the dense LDLt factorization is always used.
  static constexpr int kSparseLdltDofThreshold = 200;

  LinearSolver(
      KrylovSolverParams const& params,
      std::shared_ptr<PreconditionerRecyclingManager<T>> precRecyclingMgr = nullptr)
      : _precRecyclingMgr(
            precRecyclingMgr ? precRecyclingMgr
                             : std::make_shared<PreconditionerRecyclingManager<T>>()) {
    SetParams(params);
  }

  virtual ~LinearSolver() = default;

  /// @brief Set the parameters of the solver.
  void SetParams(KrylovSolverParams const& params) {
    MOCHI_ASSERT(params.maxIter >= 0, "Maximum number of linear iterations must not be negative.");
    MOCHI_ASSERT(params.solverType != LinearSolverType::Auto, "Solver type must have been set.");
    if (params.preconditionerLifespan > 1 && details::IsCudaSolver(params.solverType)) {
      MOCHI_LOG_WARNING_ONCE("Preconditioner recycling not supported with CUDA solvers.");
    }
    if (details::IsCudaSolver(params.solverType) && details::IsIterativeSolver(params.solverType) &&
        params.preconditionerType != PreconditionerType::None &&
        params.preconditionerType != PreconditionerType::Jacobi &&
        params.preconditionerType != PreconditionerType::BlockJacobi) {
      MOCHI_LOG_ERROR_ONCE(
          "CUDA iterative solvers only support None, Jacobi and BlockJacobi preconditioners.");
    }
    _params = params;
  }

  /// @brief Retrieve the parameters of the solver.
  KrylovSolverParams GetParams() const {
    return _params;
  }

  /// @brief Solves the linear system 𝐀𝐱 = 𝐛 using the solver type specified by the solver
  /// parameters. Both direct and iterative solvers are supported.
  ///
  /// @param[in] A Matrix of the linear system.
  /// @param[in] b Vector with the right-hand side.
  /// @param[in,out] x Vector with the initial guess at input and the solution at output.
  /// @param[in] hasOperatorChanged Boolean flag for whether the operator has changed since the
  /// previous solve. Used as performance optimization for direct solvers, Krylov subspace recycling
  /// and preconditioner recycling if the operator has not changed. Default is true.
  template <int kPrecBlockSize = 3, typename MatrixType, typename RhsType, typename SolType>
  LinearSolverStatus
  Solve(MatrixType const& A, RhsType const& b, SolType& x, bool hasOperatorChanged = true);

 protected:
  template <typename MatrixType, typename RhsType, typename SolType, typename PrecType>
  LinearSolverStatus
  IterativeSolve(MatrixType const& A, RhsType const& b, SolType& x, PrecType const& prec);

  template <int kPrecBlockSize, typename MatrixType, typename RhsType, typename SolType>
  LinearSolverStatus CudaIterativeSolve(MatrixType const& A, RhsType const& b, SolType& x);

 protected:
  /// @brief Solver configuration parameters.
  KrylovSolverParams _params = {};

  /// @brief Dense LDLt factorization for the LDLt solver. Cached across solves if the operator is
  /// unchanged.
  std::unique_ptr<LDLt<T>> _denseLdlt = nullptr;

  /// @brief Sparse LDLt factorization for the LDLt solver. Cached across solves if the operator is
  /// unchanged.
  /// @note Only supports block sizes 1, 3 and 4.
  std::variant<
      std::monostate,
      std::unique_ptr<krylov::SparseLDLt<T, 1>>,
      std::unique_ptr<krylov::SparseLDLt<T, 3>>,
      std::unique_ptr<krylov::SparseLDLt<T, 4>>>
      _sparseLdlt;

  /// @brief Dense LU factorization for the LU solver. Cached across solves if the operator is
  /// unchanged.
  std::unique_ptr<LU<T>> _lu = nullptr;

  /// @brief Subspace recycling status for recycling-subspace iterative solvers. Stores Krylov
  /// subspaces from previous solves to accelerate convergence of subsequent solves.
  SubspaceRecyclingStatus<T> _subspaceRecycling = {};

  /// @brief Preconditioner recycling manager.
  /// @details This shared pointer manages the lifecycle of the preconditioner across multiple
  /// solves. Preconditioner recycling is useful when performing multiple solves with the same or a
  /// similar matrix. See PreconditionerRecyclingManager documentation for additional details.
  /// @note Preconditioner recycling is not supported with CUDA solvers.
  std::shared_ptr<PreconditionerRecyclingManager<T>> _precRecyclingMgr = nullptr;
};

template <typename T>
template <int kPrecBlockSize, typename MatrixType, typename RhsType, typename SolType>
LinearSolverStatus
LinearSolver<T>::Solve(MatrixType const& A, RhsType const& b, SolType& x, bool hasOperatorChanged) {
  if constexpr (IsLinearOperator<MatrixType>) {
    static_assert(!IsCuda<MatrixType>, "CUDA matrices not supported yet");
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
    MOCHI_ASSERT_VERBOSE((A.Cols() == x.Rows()) && (x.Rows() == b.Rows()), "Inconsistent sizes.");
    MOCHI_ASSERT_VERBOSE(_precRecyclingMgr, "Invalid preconditioner recycling pointer.");

    if (hasOperatorChanged) {
      // Factorizations are no longer valid.
      _denseLdlt.reset(nullptr);
      _sparseLdlt = std::monostate{};
      _lu.reset(nullptr);
      _subspaceRecycling.hasOperatorChanged = true;
    }

    if (!details::IsIterativeSolver(_params.solverType) &&
        (_params.preconditionerType != PreconditionerType::None)) {
      MOCHI_LOG_WARNING_ONCE(
          "Direct solvers don't support preconditioning. The requested preconditioner will be ignored.");
    }

    if (_params.solverType == LinearSolverType::LDLT) {
      int info = 0;

      // Use sparse LDLt if the matrix is sparse or block sparse and above the size threshold. Use
      // dense LDLt otherwise.
      if constexpr (IsBlockSparseMatrix<MatrixType> || IsSparseMatrix<MatrixType>) {
        if (A.Rows() >= kSparseLdltDofThreshold) {
          constexpr int kBlockSize = []() {
            if constexpr (IsBlockSparseMatrix<MatrixType>) {
              return MatrixType::kBlockSize;
            } else {
              return 1;
            }
          }();
          static_assert(
              kBlockSize == 1 || kBlockSize == 3 || kBlockSize == 4,
              "Only block sizes 1, 3 and 4 are currently supported");

          using SparseLDLtType = krylov::SparseLDLt<T, kBlockSize>;
          auto* existingLdlt = std::get_if<std::unique_ptr<SparseLDLtType>>(&_sparseLdlt);

          if (existingLdlt == nullptr || *existingLdlt == nullptr) {
            _sparseLdlt = std::make_unique<SparseLDLtType>(A, info);
            existingLdlt = &std::get<std::unique_ptr<SparseLDLtType>>(_sparseLdlt);
          }

          if (info == 0) {
            x = b;
            (*existingLdlt)->LeftSolveInPlace(x);
          } else {
            _sparseLdlt = std::monostate{};
          }
          return LinearSolverStatus{.converged = (info == 0)};
        }
      }

      if (_denseLdlt == nullptr) {
        if constexpr (IsMatrix<MatrixType>) {
          _denseLdlt = std::make_unique<LDLt<T>>(A, info);
        } else {
          if ((A.Rows() >= kSparseLdltDofThreshold)) {
            // TODO:
            // - For IslandOperators, it may be more efficient to convert to sparse (or block
            //   sparse) matrix and use the sparse LDLt.
            // - For LowRankAugmentedMatrix, it may be more efficient to use the (dense or sparse)
            //   LDLt of the underlying matrix and the Sherman-Morrison formula.
            MOCHI_LOG_WARNING_ONCE(
                "Using dense LDLt solver with a large non-dense matrix. Performance may degrade.");
          }
          _denseLdlt = std::make_unique<LDLt<T>>(ToMatrix(A), info);
        }
      }
      if (info == 0) {
        x = b;
        _denseLdlt->LeftSolveInPlace(x);
      } else {
        _denseLdlt.reset(nullptr);
      }
      return LinearSolverStatus{.converged = (info == 0)};

    } else if (_params.solverType == LinearSolverType::LU) {
      if (_lu == nullptr) {
        if constexpr (IsMatrix<MatrixType>) {
          _lu = std::make_unique<LU<T>>(A);
        } else {
          _lu = std::make_unique<LU<T>>(ToMatrix(A));
        }
      }
      x = b;
      _lu->LeftSolveInPlace(x);
      return LinearSolverStatus{.converged = true};
    } else if (_params.solverType == LinearSolverType::ExperimentalCudaSparseCholesky) {
#if MOCHI_USE_CUDA
      if constexpr (IsSparseMatrix<MatrixType> || IsBlockSparseMatrix<MatrixType>) {
        krylov::CudaSparseCholesky<T> Chol(A);
        CudaVector<T> bCuda(b), xCuda(x); // Transfer data to the GPU
        Chol(bCuda, xCuda);
        x = xCuda;
        return LinearSolverStatus{.converged = true};
      } else {
        MOCHI_ASSERT(false, "CudaSparseCholesky requires building a sparse matrix type.");
      }
#else
      MOCHI_ASSERT(false, "CudaSparseCholesky requires building with CUDA.");
#endif
    } else if (_params.solverType == LinearSolverType::ExperimentalCudaSparseLDLT) {
#if MOCHI_USE_CUDSS
      if constexpr (IsSparseMatrix<MatrixType> || IsBlockSparseMatrix<MatrixType>) {
        krylov::CudaSparseLDLt<T> LDLt(A);
        CudaVector<T> bCuda(b), xCuda(x); // Transfer data to the GPU
        LDLt(bCuda, xCuda);
        x = xCuda;
        return LinearSolverStatus{.converged = true};
      } else {
        MOCHI_ASSERT(false, "CudaSparseLDLt requires a sparse matrix type.");
      }
#else
      MOCHI_ASSERT(false, "CudaSparseLDLt requires building with CUDA and cudss.");
#endif
    } else if (_params.solverType == LinearSolverType::ExperimentalCudaSparseLU) {
#if MOCHI_USE_CUDA
      if constexpr (IsSparseMatrix<MatrixType> || IsBlockSparseMatrix<MatrixType>) {
        krylov::CudaSparseLU<T> LU(A);
        CudaVector<T> bCuda(b), xCuda(x); // Transfer data to the GPU
        LU(bCuda, xCuda);
        x = xCuda;
        return LinearSolverStatus{.converged = true};
      } else {
        MOCHI_ASSERT(false, "CudaSparseLU requires a sparse matrix type.");
      }
#else
      MOCHI_ASSERT(false, "CudaSparseLU requires building with CUDA.");
#endif
    } else if (
        _params.solverType == LinearSolverType::CG ||
        _params.solverType == LinearSolverType::GMRES ||
        _params.solverType == LinearSolverType::MINRES ||
        _params.solverType == LinearSolverType::ParallelCG ||
        _params.solverType == LinearSolverType::AsyncCG ||
        _params.solverType == LinearSolverType::AugmentedCG) {
      _precRecyclingMgr->template SetupPreconditioner<kPrecBlockSize>(
          A, hasOperatorChanged, _params.preconditionerType, _params.preconditionerLifespan);
      return IterativeSolve(
          A, b, x, details::PrecApplyer<T>{*_precRecyclingMgr->GetPreconditioner()});
    } else if (
        _params.solverType == LinearSolverType::CudaCG ||
        _params.solverType == LinearSolverType::CudaGMRES) {
      return CudaIterativeSolve<kPrecBlockSize>(A, b, x);
    }
    MOCHI_ASSERT(false, "Unsupported solver.");
    return {};
  } else {
    // Branch for std::variant's holding matrix and linear operator types. It resolves to the Solve
    // specialization with the actual implementation, i.e. the 'if' branch above.
    return std::visit(
        [&](auto const& mat) { return Solve<kPrecBlockSize>(mat, b, x, hasOperatorChanged); }, A);
  }
}

template <typename T>
template <typename MatrixType, typename RhsType, typename SolType, typename PrecType>
LinearSolverStatus LinearSolver<T>::IterativeSolve(
    MatrixType const& A,
    RhsType const& b,
    SolType& x,
    PrecType const& prec) {
  if (_params.solverType == LinearSolverType::CG) {
    return details::DispatchConvergenceNorm<T>(_params, [&](auto& stopCriterion) {
      return krylov::PCG(
          A, b, x, prec, _params.maxIter, stopCriterion, _params.abortIfNotSpd, _params.verbosity);
    });
  } else if (_params.solverType == LinearSolverType::ParallelCG) {
    if constexpr (!IsCuda<MatrixType>) {
      return details::DispatchConvergenceNorm<T>(_params, [&](auto& stopCriterion) {
        return krylov::ParallelPCG(
            A,
            b,
            x,
            prec,
            _params.maxIter,
            stopCriterion,
            _params.abortIfNotSpd,
            _params.verbosity);
      });
    } else {
      MOCHI_ASSERT(false, "Parallel PCG not supported for CUDA matrices.");
    }
  } else if (_params.solverType == LinearSolverType::AsyncCG) {
    if constexpr (!IsCuda<MatrixType>) {
      return details::DispatchConvergenceNorm<T>(_params, [&](auto& stopCriterion) {
        return krylov::AsyncPCG(
            A,
            b,
            x,
            prec,
            _params.maxIter,
            stopCriterion,
            _params.abortIfNotSpd,
            _params.verbosity);
      });
    } else {
      MOCHI_ASSERT(false, "Async PCG not supported for CUDA matrices.");
    }
  } else if (_params.solverType == LinearSolverType::AugmentedCG) {
    if constexpr (!IsCuda<MatrixType>) {
      if (_params.normType != LinearSolverConvergenceNorm::ResidualL2) {
        MOCHI_LOG_WARNING_ONCE(
            "Augmented PCG requires the L2-norm of the residual to monitor convergence. The requested "
            "convergence norm type (%i) is different and will be ignored.",
            static_cast<int>(_params.normType));
      }
      krylov::StatusResidualL2<krylov::UsualDot, T> stopCriterion(
          static_cast<T>(_params.relTol),
          static_cast<T>(_params.absTol),
          static_cast<T>(_params.relDivTol),
          _params.numRecyclingDir);
      auto const result = krylov::AugmentedPCG(
          A,
          b,
          x,
          prec,
          _params.maxIter,
          stopCriterion,
          RecyclingParams(_params),
          _subspaceRecycling,
          _params.abortIfNotSpd,
          _params.verbosity);
      _subspaceRecycling.hasOperatorChanged = false;
      return result;
    } else {
      MOCHI_ASSERT(false, "Augmented PCG not supported on CUDA.");
    }
  } else if (_params.solverType == LinearSolverType::GMRES) {
    krylov::StatusImplicitResidualNorm<T> stopCriterion(
        static_cast<T>(_params.relTol),
        static_cast<T>(_params.absTol),
        static_cast<T>(_params.relDivTol));
    return krylov::GMRes(
        A, b, x, prec, _params.maxIter, stopCriterion, _params.restartSize, _params.verbosity);
  } else if (_params.solverType == LinearSolverType::MINRES) {
    krylov::StatusImplicitResidualNorm<T> stopCriterion(
        static_cast<T>(_params.relTol),
        static_cast<T>(_params.absTol),
        static_cast<T>(_params.relDivTol));
    return krylov::MinRes(A, b, x, prec, _params.maxIter, stopCriterion, _params.verbosity);
  }
  MOCHI_ASSERT(false, "Solver type (%i) not supported.", static_cast<int>(_params.solverType));
  return {};
}

template <typename T>
template <int kPrecBlockSize, typename MatrixType, typename RhsType, typename SolType>
LinearSolverStatus LinearSolver<T>::CudaIterativeSolve(
    [[maybe_unused]] MatrixType const& A,
    [[maybe_unused]] RhsType const& b,
    [[maybe_unused]] SolType& x) {
  LinearSolverStatus status;
#if MOCHI_USE_CUDA
  if constexpr (IsAnyMatrix<MatrixType>) {
    auto ACuda = ToCuda(A);
    CudaVector<T> bCuda(b), xCuda(x); // Transfer data to the GPU
    auto P =
        details::CreateCudaPreconditioner<T, kPrecBlockSize>(ACuda, _params.preconditionerType);

    if (_params.solverType == LinearSolverType::CudaCG) {
      status = details::DispatchConvergenceNorm<T>(_params, [&](auto& stopCriterion) {
        return krylov::CudaPCG(
            ACuda,
            bCuda,
            xCuda,
            P,
            _params.maxIter,
            stopCriterion,
            _params.abortIfNotSpd,
            _params.verbosity);
      });
    } else {
      MOCHI_ASSERT(_params.solverType == LinearSolverType::CudaGMRES);
      status = krylov::CudaGMRes(
          ACuda,
          bCuda,
          xCuda,
          P,
          _params.maxIter,
          _params.absTol,
          _params.relTol,
          _params.relDivTol,
          _params.restartSize,
          _params.verbosity);
    }

    x = xCuda; // Transfer the result back to the CPU
  } else {
    MOCHI_ASSERT(false, "CUDA iterative solvers require a supported matrix type.");
  }
#else
  MOCHI_ASSERT(false, "CUDA iterative solvers require building with CUDA.");
#endif
  return status;
}

} // namespace mochi
