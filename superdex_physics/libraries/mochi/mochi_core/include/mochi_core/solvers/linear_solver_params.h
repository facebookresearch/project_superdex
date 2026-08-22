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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/verbosity_params.h>

namespace mochi {
/** @brief Linear solver types. */
enum struct LinearSolverType {
  /**
   * @brief Conjugate Gradient (CG) solver.
   *
   * @note Only valid for symmetric positive-definite systems.
   * @note Unlike @ref LinearSolverType::ParallelCG, only matrix-vector products are parallelized.
   * CG orthogonalization and preconditioner solves are not.
   */
  CG,

  /** @brief Generalized Minimal Residual (GMRES) solver. */
  GMRES,

  /**
   * @brief CUDA CG solver.
   *
   * @note Only valid for symmetric positive-definite systems.
   * @note Requires building with CUDA and a preconditioner with CUDA support.
   *
   * @see PreconditionerType
   */
  CudaCG,

  /**
   * @brief CUDA GMRES solver.
   *
   * @note Requires building with CUDA and a preconditioner with CUDA support.
   *
   * @see PreconditionerType
   */
  CudaGMRES,

  /**
   * @brief [Experimental] Augmented CG solver with Krylov subspace recycling.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Only valid for symmetric positive-definite systems.
   * @note Requires building with Eigen.
   */
  AugmentedCG,

  /**
   * @brief Direct solver based on the LDLt factorization without pivoting.
   *
   * @note Only valid for symmetric systems.
   * @note May be unstable for indefinite or ill-conditioned systems.
   * @note Uses a dense or a sparse factorization depending on the size and sparsity of the system.
   */
  LDLT,

  /**
   * @brief Direct solver based on the dense LU factorization without pivoting.
   *
   * @note May be unstable for ill-conditioned systems.
   * @note The cost of the factorization is O(n^3). Will be slow for large systems.
   * @note Consider using @ref LinearSolverType::LDLT if the system is symmetric.
   */
  LU,

  /**
   * @brief [Experimental] Asynchronous CG solver.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Only valid for symmetric positive-definite systems.
   * @note Async CG is a reformulation of the classical CG algorithm to compute the matrix-vector
   * products in parallel to the orthogonalization and convergence check.
   * @note Equivalent to the classical CG algorithm in exact-precision arithmetic, but has inferior
   * stability properties in finite-precision arithmetic.
   */
  AsyncCG,

  /**
   * @brief [Experimental] Parallel CG solver.
   *
   * @warning Not deterministic.
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Only valid for symmetric positive-definite systems.
   * @note Unlike @ref LinearSolverType::CG, all operations (matrix-vector products, preconditioner
   * solves, CG orthogonalization) are parallelized.
   * @note Requires a preconditioner with a parallel solve.
   */
  ParallelCG,

  /**
   * @brief Minimal Residual (MINRES) solver.
   *
   * @note Only valid for symmetric systems.
   */
  MINRES,

  /**
   * @brief [Experimental] CUDA sparse Cholesky factorization.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Only valid for sparse or block-sparse symmetric positive-definite systems.
   * @note Requires building with CUDA.
   */
  ExperimentalCudaSparseCholesky,

  /**
   * @brief [Experimental] CUDA sparse LDLt factorization.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Only valid for sparse or block-sparse symmetric systems.
   * @note Requires building with CUDA and cuDSS.
   */
  ExperimentalCudaSparseLDLT,

  /**
   * @brief [Experimental] CUDA sparse LU factorization.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Only valid for sparse or block-sparse systems.
   * @note Requires building with CUDA.
   */
  ExperimentalCudaSparseLU,

  /** @brief Let Mochi select the solver type based on the problem. */
  Auto,

  /** @brief Number of solver type enum values. */
  Count,

  /** @brief Default solver type. */
  Default = Auto
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::LinearSolverType)
MOCHI_ENUM_ITEM(CG)
MOCHI_ENUM_ITEM(GMRES)
MOCHI_ENUM_ITEM(CudaCG)
MOCHI_ENUM_ITEM(CudaGMRES)
MOCHI_ENUM_ITEM(AugmentedCG)
MOCHI_ENUM_ITEM(LDLT)
MOCHI_ENUM_ITEM(LU)
MOCHI_ENUM_ITEM(AsyncCG)
MOCHI_ENUM_ITEM(ParallelCG)
MOCHI_ENUM_ITEM(MINRES)
MOCHI_ENUM_ITEM(ExperimentalCudaSparseCholesky)
MOCHI_ENUM_ITEM(ExperimentalCudaSparseLDLT)
MOCHI_ENUM_ITEM(ExperimentalCudaSparseLU)
MOCHI_ENUM_ITEM(Auto)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/**
 * @brief Preconditioner types for the linear solver.
 *
 * @note Only used for iterative solvers.
 * @note CUDA iterative solvers only support @ref PreconditionerType::None, @ref
 * PreconditionerType::Jacobi and @ref PreconditionerType::BlockJacobi preconditioners.
 */
enum struct PreconditionerType {
  /** @brief No preconditioner. */
  None,

  /**
   * @brief Jacobi preconditioner.
   *
   * @note Only valid for matrices with non-zero diagonal entries.
   */
  Jacobi,

  /**
   * @brief Block Jacobi preconditioner with block size of 3.
   *
   * @note Only valid for matrices whose size is a multiple of 3 and whose 3x3 diagonal blocks are
   * non-singular.
   */
  BlockJacobi,

  /**
   * @brief Symmetric successive over-relaxation (SSOR) preconditioner.
   *
   * @note Only valid for symmetric positive-definite matrices.
   */
  SSOR,

  /**
   * @brief Block symmetric successive over-relaxation (SSOR) preconditioner with block size of 3.
   *
   * @note Only valid for symmetric positive-definite matrices whose size is a multiple of 3 and
   * whose 3x3 diagonal blocks are non-singular.
   */
  BlockSSOR,

  /**
   * @brief Dense LU factorization without pivoting.
   *
   * @note May be unstable for ill-conditioned matrices.
   * @note The iterative solver should converge in 1 iteration.
   */
  LU,

  /**
   * @brief Algebraic multigrid (AMG).
   *
   * @note Only valid for 3x3 block sparse symmetric positive-definite matrices with non-singular
   * 3x3 diagonal blocks.
   */
  AMG,

  /**
   * @brief Symmetric inverse preconditioner based on the dense LDLt factorization without pivoting.
   *
   * @note Only valid for symmetric matrices.
   * @note May be unstable for indefinite or ill-conditioned matrices.
   * @note The iterative solver should converge in 1 iteration.
   */
  SymInverse,

  /**
   * @brief Dense LDLt factorization without pivoting.
   *
   * @note Only valid for symmetric matrices.
   * @note May be unstable for indefinite or ill-conditioned matrices.
   * @note The iterative solver should converge in 1 iteration.
   */
  LDLT,

  /** @brief Incomplete LU factorization with zero fill-in. */
  ILU0,

  /**
   * @brief Incomplete Cholesky factorization with zero fill-in.
   *
   * @note Only valid for symmetric positive-definite matrices.
   */
  IC0,

  /**
   * @brief Domain decomposition preconditioner, where each actor is a subdomain.
   *
   * @note The most appropriate preconditioner type is used for each actor.
   */
  PerActor,

  /**
   * @brief [Experimental] Colored symmetric successive over-relaxation (SSOR) preconditioner.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Only valid for symmetric positive-definite matrices.
   */
  ColoredSSOR,

  /** @brief Number of preconditioner type enum values. */
  Count,

  /** @brief Default preconditioner type. */
  Default = PerActor
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::PreconditionerType)
MOCHI_ENUM_ITEM(None)
MOCHI_ENUM_ITEM(Jacobi)
MOCHI_ENUM_ITEM(BlockJacobi)
MOCHI_ENUM_ITEM(SSOR)
MOCHI_ENUM_ITEM(BlockSSOR)
MOCHI_ENUM_ITEM(LU)
MOCHI_ENUM_ITEM(AMG)
MOCHI_ENUM_ITEM(SymInverse)
MOCHI_ENUM_ITEM(LDLT)
MOCHI_ENUM_ITEM(ILU0)
MOCHI_ENUM_ITEM(IC0)
MOCHI_ENUM_ITEM(PerActor)
MOCHI_ENUM_ITEM(ColoredSSOR)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/**
 * @brief Norm types for the stopping criteria of the linear solver.
 *
 * @note This setting only applies to @ref LinearSolverType::CG, @ref LinearSolverType::CudaCG, @ref
 * LinearSolverType::ParallelCG and @ref LinearSolverType::AsyncCG. @ref LinearSolverType::GMRES,
 * @ref LinearSolverType::CudaGMRES, @ref LinearSolverType::MINRES, and @ref
 * LinearSolverType::AugmentedCG always use the L2-norm of the residual regardless of this setting.
 * Direct solvers do not have stop criteria.
 */
enum struct LinearSolverConvergenceNorm {
  ResidualL2, ///< L2-norm of the residual: \f$\|r\|_2\f$.
  PreconditionedResidualL2, ///< L2-norm of the preconditioned residual: \f$\|z\|_2\f$ where
                            ///< \f$z = M^{-1}r\f$.
  ResidualPreconditionerInduced, ///< Preconditioner-induced norm of the residual:
                                 ///< \f$\sqrt{\langle r, z \rangle}\f$ where \f$z = M^{-1}r\f$.
  Count, ///< Number of linear solver convergence norm enum values.
  Default = PreconditionedResidualL2 ///< Default linear solver convergence norm.
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::LinearSolverConvergenceNorm)
MOCHI_ENUM_ITEM(ResidualL2)
MOCHI_ENUM_ITEM(PreconditionedResidualL2)
MOCHI_ENUM_ITEM(ResidualPreconditionerInduced)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {

/** @brief Sentinel value for @ref LinearSolverParams::maxIter that lets Mochi automatically select
 * the maximum number of linear solver iterations. */
constexpr int kAutoLinearSolverMaxIter = -1;

/** @brief Parameters for the linear solve performed in each iteration of the non-linear solver. */
struct LinearSolverParams {
  /** @brief Linear solver type. */
  LinearSolverType solverType = LinearSolverType::Default;

  /**
   * @brief Preconditioner type.
   *
   * @note Applies only to iterative solvers.
   */
  PreconditionerType preconditionerType = PreconditionerType::Default;

  /**
   * @brief Norm used for the stopping criteria.
   *
   * @note Applies only to the solvers documented in @ref LinearSolverConvergenceNorm.
   */
  LinearSolverConvergenceNorm normType = LinearSolverConvergenceNorm::Default;

  /**
   * @brief Absolute residual norm tolerance for convergence.
   *
   * @note Applies only to iterative solvers.
   */
  real absTol = 1e-9_r;

  /**
   * @brief Relative residual norm tolerance for convergence, relative to the initial residual.
   *
   * @note Applies only to iterative solvers.
   * @note Ignored when using @ref LinearToleranceStrategy::EisenstatWalker1 or @ref
   * LinearToleranceStrategy::EisenstatWalker2 in the non-linear solver.
   * @note For @ref LinearToleranceStrategy::EisenstatWalker3, this is used as a lower bound on the
   * adaptive tolerance in every non-linear iteration.
   */
  real relTol = 1e-5_r;

  /**
   * @brief Relative residual norm tolerance for divergence, relative to the initial residual.
   *
   * @note Applies only to iterative solvers.
   */
  real relDivTol = 1e10_r;

  /**
   * @brief Maximum number of linear solver iterations.
   *
   * @note Applies only to iterative solvers.
   * @note Must be non-negative or @ref kAutoLinearSolverMaxIter.
   * @note @ref kAutoLinearSolverMaxIter lets Mochi select the maximum number of iterations based
   * on the problem.
   */
  int maxIter = kAutoLinearSolverMaxIter;

  /**
   * @brief Krylov subspace dimension before restarting.
   *
   * @note Applies only to GMRES solvers (e.g., @ref LinearSolverType::GMRES, @ref
   * LinearSolverType::CudaGMRES).
   */
  int restartSize = 1000;

  /**
   * @brief Abort solve if matrix is not SPD.
   *
   * @note Applies only to iterative SPD solvers (e.g., @ref LinearSolverType::CG, @ref
   * LinearSolverType::CudaCG, @ref LinearSolverType::AugmentedCG).
   */
  bool abortIfNotSpd = false;

  /** @brief Verbosity level for logging output. */
  VerbosityLevel verbosity = VerbosityLevel::Warning;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(LinearSolverParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::LinearSolverParams)
  MOCHI_FIELD(solverType)
  MOCHI_FIELD(preconditionerType)
  MOCHI_FIELD(normType)
  MOCHI_FIELD(absTol)
  MOCHI_FIELD(relTol)
  MOCHI_FIELD(relDivTol)
  MOCHI_FIELD(maxIter)
  MOCHI_FIELD(restartSize)
  MOCHI_FIELD(abortIfNotSpd)
  MOCHI_FIELD(verbosity)
  MOCHI_STRUCT_END()
};

} // namespace mochi
