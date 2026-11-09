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
   * @note By default, an appropriate preconditioner type is selected for each actor.
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

  /**
   * @brief [Experimental] @ref PreconditionerType::PerActor preconditioner that can switch the
   * preconditioner type of each actor during the simulation to speed up the linear solves.
   *
   * @warning Some scenes can run slower than with @ref PreconditionerType::PerActor, and some steps
   * can take longer than usual, causing latency spikes.
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   */
  AdaptivePerActor,

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
MOCHI_ENUM_ITEM(AdaptivePerActor)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/**
 * @brief Norm types for the stopping criteria of the linear solver.
 *
 * @note @ref LinearSolverType::CG, @ref LinearSolverType::ParallelCG and @ref
 * LinearSolverType::AsyncCG support every norm. @ref LinearSolverType::CudaCG supports every norm
 * except @ref LinearSolverConvergenceNorm::ResidualPerActorWeighted, for which it uses @ref
 * LinearSolverConvergenceNorm::PreconditionedResidualL2. @ref LinearSolverType::GMRES, @ref
 * LinearSolverType::CudaGMRES, and @ref LinearSolverType::AugmentedCG always use the L2-norm of the
 * residual regardless of this setting. @ref LinearSolverType::MINRES always uses the
 * preconditioner-induced norm of the residual. Direct solvers do not have stop criteria.
 */
enum struct LinearSolverConvergenceNorm {
  ResidualL2, ///< L2-norm of the residual: \f$\|r\|_2\f$.
  PreconditionedResidualL2, ///< L2-norm of the preconditioned residual: \f$\|z\|_2\f$ where
                            ///< \f$z = M^{-1}r\f$.
  ResidualPreconditionerInduced, ///< Preconditioner-induced norm of the residual:
                                 ///< \f$\sqrt{\langle r, z \rangle}\f$ where \f$z = M^{-1}r\f$.
  /**
   * @brief Per-actor weighted L2-norm of the residual, each actor normalized by its own non-linear
   * residual, so actors have comparable influence regardless of the magnitude of their residuals.
   *
   * @details \f$\|r\| = \sqrt{\sum_a \|r_a\|_W^2 / s_a^2}\f$, where
   * \f$\|r_a\|_W = \sqrt{\sum_i w_i r_i^2}\f$ uses actor \f$a\f$'s weights from @ref
   * NonLinearSolverConvergenceMode::PerActorWeighted, and \f$s_a\f$ is the larger of actor
   * \f$a\f$'s non-linear residual in that norm and @ref NonLinearSolverParams::absTol, or, if both
   * are zero, the largest actor's residual.
   *
   * @note In @ref NonLinearSolverConvergenceMode::Global mode, the solvers that support it use @ref
   * LinearSolverConvergenceNorm::ResidualL2 instead.
   */
  ResidualPerActorWeighted,
  Count, ///< Number of linear solver convergence norm enum values.
  Default = PreconditionedResidualL2 ///< Default linear solver convergence norm.
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::LinearSolverConvergenceNorm)
MOCHI_ENUM_ITEM(ResidualL2)
MOCHI_ENUM_ITEM(PreconditionedResidualL2)
MOCHI_ENUM_ITEM(ResidualPreconditionerInduced)
MOCHI_ENUM_ITEM(ResidualPerActorWeighted)
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
   * @note When the solver stops on @ref LinearSolverConvergenceNorm::ResidualPerActorWeighted, the
   * absolute stop ends the solve only once every actor's weighted residual \f$\|r_a\|_W\f$ is at
   * most @ref absTol. Otherwise, it bounds the solver's stopping norm.
   * @note 0 leaves only the relative test.
   * @note A negative value lets Mochi select it. When the solver stops on @ref
   * LinearSolverConvergenceNorm::ResidualPerActorWeighted, Mochi selects 1% of the residual at
   * which the non-linear solver accepts each actor: @ref NonLinearSolverParams::relTol times the
   * actor's initial residual, or @ref NonLinearSolverParams::absTol if larger. When it stops on
   * @ref LinearSolverConvergenceNorm::ResidualL2 in @ref NonLinearSolverConvergenceMode::Global
   * mode, Mochi selects 1% of the residual at which the non-linear solver accepts the island.
   * Otherwise, it selects 1e-9. The selection is 0 when @ref NonLinearSolverParams::absTol is 0 and
   * either @ref NonLinearSolverParams::relTol or the initial residual of an actor or the island is
   * 0, which leaves only the relative test.
   */
  real absTol = 1e-9_r;

  /**
   * @brief Relative residual norm tolerance for convergence, relative to the same norm of the
   * right-hand side.
   *
   * @note Applies only to iterative solvers.
   * @note Ignored when using @ref LinearToleranceStrategy::EisenstatWalker1 or @ref
   * LinearToleranceStrategy::EisenstatWalker2 in the non-linear solver.
   */
  real relTol = 1e-5_r;

  /**
   * @brief Relative residual norm tolerance for divergence, relative to the same norm of the
   * right-hand side.
   *
   * @note Applies only to iterative solvers.
   */
  real relDivTol = 1e10_r;

  /**
   * @brief Maximum number of iterations for iterative solvers.
   *
   * @note Applies only to iterative solvers.
   * @note Must be positive or @ref kAutoLinearSolverMaxIter.
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
