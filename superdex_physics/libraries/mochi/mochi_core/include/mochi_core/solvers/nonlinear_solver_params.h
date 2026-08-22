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

#include <limits>

namespace mochi {
/** @brief Convergence status of the non-linear solver. */
enum struct ConvergenceStatus {
  // WARNING: Values are ordered by severity: None < Converged < Stopped < Diverged. This ordering
  // enables worst-status accumulation via Max.

  /** @brief Convergence status has not been set. */
  None,

  /** @brief Solver converged to the requested tolerance. */
  Converged,

  /**
   * @brief Solver met at least one stopping criterion without converging to the requested
   * tolerance.
   */
  Stopped,

  /** @brief Solver diverged. Some form of solution reset may have been used. */
  Diverged,

  /** @brief Number of convergence status enum values. */
  Count
};
static_assert(
    static_cast<int>(ConvergenceStatus::Count) == 4 &&
        ConvergenceStatus::None < ConvergenceStatus::Converged &&
        ConvergenceStatus::Converged < ConvergenceStatus::Stopped &&
        ConvergenceStatus::Stopped < ConvergenceStatus::Diverged,
    "ConvergenceStatus must be ordered by severity.");
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::ConvergenceStatus)
MOCHI_ENUM_ITEM(None)
MOCHI_ENUM_ITEM(Converged)
MOCHI_ENUM_ITEM(Stopped)
MOCHI_ENUM_ITEM(Diverged)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/** @brief Positive Semi-Definite (PSD) projection modes for the dresidual matrix. */
// clang-format off
enum class PsdProjectionMode {
  Never,        ///< Never project to PSD.
  Always,       ///< Always project to PSD.
  IfFailRetry,  ///< If one non-linear iteration fails, retry the iteration projecting to PSD.
  IfFailAlways, ///< If one non-linear iteration fails, retry the iteration projecting to PSD and continue projecting in all the remaining iterations of the solve.
  Count,        ///< Number of PSD projection mode enum values.
  Default = Always ///< Default PSD projection mode.
};
// clang-format on
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::PsdProjectionMode)
MOCHI_ENUM_ITEM(Never)
MOCHI_ENUM_ITEM(Always)
MOCHI_ENUM_ITEM(IfFailRetry)
MOCHI_ENUM_ITEM(IfFailAlways)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/** @brief Non-linear solver types. */
enum struct NonLinearSolverType {
  /** @brief Newton's method. */
  Newton,

  /**
   * @brief Broyden-Fletcher-Goldfarb-Shanno (BFGS) method.
   *
   * @note The initial approximation of the dresidual is the actual dresidual (i.e., the first
   * iteration matches Newton's method).
   * @note The algorithm is restarted with the actual dresidual every @ref
   * NonLinearSolverParams::dResidualAssemblyPeriod iterations, or immediately if the line search
   * fails to improve the figure of merit it monitors or the linear solver fails to converge.
   *
   * @see NonLinearSolverParams::dResidualAssemblyPeriod
   */
  BFGS,

  /**
   * @brief Symmetric Rank-One (SR1) method.
   *
   * @note The initial approximation of the dresidual is the actual dresidual (i.e., the first
   * iteration matches Newton's method).
   * @note The algorithm is restarted with the actual dresidual every @ref
   * NonLinearSolverParams::dResidualAssemblyPeriod iterations, or immediately if the line search
   * fails to improve the figure of merit it monitors or the linear solver fails to converge.
   *
   * @see NonLinearSolverParams::dResidualAssemblyPeriod
   */
  SR1,

  /** @brief Number of non-linear solver type enum values. */
  Count,

  /** @brief Default non-linear solver type. */
  Default = Newton
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::NonLinearSolverType)
MOCHI_ENUM_ITEM(Newton)
MOCHI_ENUM_ITEM(BFGS)
MOCHI_ENUM_ITEM(SR1)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/** @brief Line search methods for the non-linear solver. */
enum struct LineSearchType {
  /** @brief No line search. May substantially degrade stability. */
  None,

  /**
   * @brief Simple line search. Accepts a step if the objective function does not increase by more
   * than a specified relative tolerance.
   *
   * @see NonLinearSolverParams::lineSearchMaxRelIncrease. */
  Simple,

  /**
   * @brief Line search with Armijo condition.
   *
   * @see NonLinearSolverParams::lineSearchWolfe1
   */
  Armijo,

  /**
   * @brief Line search with weak Wolfe conditions.
   *
   * @see NonLinearSolverParams::lineSearchWolfe1, NonLinearSolverParams::lineSearchWolfe2
   */
  WolfeWeak,

  /**
   * @brief Line search with strong Wolfe conditions.
   *
   * @see NonLinearSolverParams::lineSearchWolfe1, NonLinearSolverParams::lineSearchWolfe2
   */
  WolfeStrong,

  /**
   * @brief Line search with residual norm condition. Accepts a step if the residual norm decreases.
   *
   * @note More robust than objective-based line searches (@ref LineSearchType::Simple, @ref
   * LineSearchType::Armijo, @ref LineSearchType::WolfeWeak, @ref LineSearchType::WolfeStrong) but
   * may degrade non-linear solver convergence in non-convex problems. In highly non-convex regions
   * where the residual norm cannot be reduced, it may cause objects to move at slower velocity than
   * they should due to the inability to take valid steps. Consider using an objective-based line
   * search in that case.
   */
  ResidualNorm,

  /**
   * @brief Line search that accepts either the Armijo or the residual norm condition.
   *
   * @note It combines the robustness of objective-based criteria when the iteration is far from the
   * solution, with residual-based criteria when it is near the solution (particularly useful with
   * single precision). It requires slightly higher cost per line-search iteration.
   */
  ArmijoOrResidualNorm,

  /** @brief Number of line search type enum values. */
  Count,

  /** @brief Default line search type. */
  Default = ResidualNorm
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::LineSearchType)
MOCHI_ENUM_ITEM(None)
MOCHI_ENUM_ITEM(Simple)
MOCHI_ENUM_ITEM(Armijo)
MOCHI_ENUM_ITEM(WolfeWeak)
MOCHI_ENUM_ITEM(WolfeStrong)
MOCHI_ENUM_ITEM(ResidualNorm)
MOCHI_ENUM_ITEM(ArmijoOrResidualNorm)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/** @brief Strategies to select the relative tolerance of the linear solver (aka forcing term). */
enum struct LinearToleranceStrategy {
  /** @brief Constant relative tolerance. */
  Constant,

  /**
   * @brief Eisenstat-Walker strategy no. 1.
   *
   * @note Should be used with @ref LinearSolverConvergenceNorm::ResidualL2. Other norms may yield
   * suboptimal forcing terms.
   * @note Not recommended for use with quasi-Newton methods (e.g., @ref NonLinearSolverType::BFGS,
   * @ref NonLinearSolverType::SR1).
   *
   * @see [Choosing the Forcing Terms in an Inexact Newton Method, Choice 1 (Eisenstat and Walker,
   * 1994)](https://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR94463.pdf)
   */
  EisenstatWalker1,

  /**
   * @brief Eisenstat-Walker strategy no. 2.
   *
   * @note Not recommended for use with quasi-Newton methods (e.g., @ref NonLinearSolverType::BFGS,
   * @ref NonLinearSolverType::SR1).
   *
   * @see [Choosing the Forcing Terms in an Inexact Newton Method, Choice 2 (Eisenstat and Walker,
   * 1994)](https://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR94463.pdf)
   */
  EisenstatWalker2,

  /**
   * @brief Custom Eisenstat-Walker strategy: eta = max(min(1/(2+k), sqrt(|r|)), eta_0), where `k`
   * is the non-linear iteration number, `|r|` is the L2-norm of the nonlinear residual, and
   * `eta_0` is @ref LinearSolverParams::relTol.
   *
   * @note Should be used with @ref LinearSolverConvergenceNorm::ResidualL2. Other norms may yield
   * suboptimal forcing terms.
   * @note Not recommended for use with quasi-Newton methods (e.g., @ref NonLinearSolverType::BFGS,
   * @ref NonLinearSolverType::SR1).
   */
  EisenstatWalker3,

  /** @brief Number of linear tolerance strategy enum values. */
  Count,

  /** @brief Default linear tolerance strategy. */
  Default = EisenstatWalker2
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::LinearToleranceStrategy)
MOCHI_ENUM_ITEM(Constant)
MOCHI_ENUM_ITEM(EisenstatWalker1)
MOCHI_ENUM_ITEM(EisenstatWalker2)
MOCHI_ENUM_ITEM(EisenstatWalker3)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/**
 * @brief Convergence monitoring mode for the non-linear solver residual.
 *
 * @note Convergence is evaluated per simulation island, not scene-wide. Each island is solved
 * independently, and the selected mode determines how convergence is assessed within that island.
 * @note The norm used to monitor divergence via @ref NonLinearSolverParams::absDivTol and @ref
 * NonLinearSolverParams::relDivTol is always the L2 norm, irrespective of this setting.
 * @note The norm used to monitor stagnation via @ref NonLinearSolverParams::relStepTol is always
 * the L2 norm, irrespective of this setting.
 *
 * @see NonLinearSolverParams::absTol, NonLinearSolverParams::relTol
 */
enum struct NonLinearSolverConvergenceMode {
  /**
   * @brief Use global, unweighted residual norm for convergence checks within each simulation
   * island.
   *
   * @details Convergence is determined by the global residual norm within each simulation island:
   * |r| <= absTol or |r| <= relTol * |r0|
   */
  Global,

  /**
   * @brief Use per-actor weighted residual norms for convergence checks.
   *
   * @details Each actor uses a per-actor weighted L2 norm |r_a|_W = sqrt(Σᵢ wᵢ·rᵢ²), where
   * weights normalize force/torque residuals by characteristic force/torque. All actors must
   * satisfy their individual criteria: |r_a|_W <= absTol or |r_a|_W <= relTol * |r0_a|_W
   *
   * @note Weights are derived from inertia properties. Actors with zero inertia receive uniform
   * weights, which provide no physical normalization. For quasi-static problems, @ref
   * NonLinearSolverConvergenceMode::Global mode is recommended.
   */
  PerActorWeighted,

  /** @brief Number of convergence monitoring mode enum values. */
  Count,

  /** @brief Default convergence monitoring mode. */
  Default = PerActorWeighted,
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::NonLinearSolverConvergenceMode)
MOCHI_ENUM_ITEM(Global)
MOCHI_ENUM_ITEM(PerActorWeighted)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/**
 * @brief Default tolerance for the L2 norm of the non-linear solver's raw linear-solve increment.
 * Avoids attempting to solve beyond the floating-point noise floor.
 *
 * @see NonLinearSolverParams::relStepTol.
 */
constexpr real kDefaultRelStepTol = 10_r * std::numeric_limits<real>::epsilon();

/** @brief Parameters for the non-linear solver. */
struct NonLinearSolverParams {
  /** @brief Non-linear solver type. */
  NonLinearSolverType solverType = NonLinearSolverType::Default;

  /**
   * @brief Every how many non-linear iterations to assemble the dresidual matrix.
   *
   * @note For Newton's method, set to >1 to reuse the dresidual across iterations.
   * @note For quasi-Newton methods (e.g., BFGS, SR1), it must be >1 and indicates every how many
   * iterations to restart the algorithm with the actual dresidual.
   */
  int dResidualAssemblyPeriod = 1;

  // Stopping criteria. The solver terminates when any of the criteria is satisfied.

  /** @brief Maximum number of non-linear iterations. */
  int maxIter = 4;

  /**
   * @brief Maximum elapsed time [s].
   *
   * @note The solve terminates if the elapsed time exceeds this threshold.
   * @note 0 means no time limit.
   */
  double maxElapsedTimeSeconds = 0;

  /**
   * @brief Convergence monitoring mode.
   *
   * @note Applies to @ref absTol and @ref relTol.
   */
  NonLinearSolverConvergenceMode convergenceMode = NonLinearSolverConvergenceMode::Default;

  /** @brief Absolute residual norm tolerance for convergence. */
  real absTol = 1e-3_r;

  /** @brief Relative residual norm tolerance for convergence, relative to the initial residual. */
  real relTol = 1e-6_r;

  /**
   * @brief Relative tolerance on the L2 norm of the raw linear-solve increment before line search
   * scaling.
   *
   * @note The solve terminates with @ref ConvergenceStatus::Stopped status if |dx|/|x| is below
   * this threshold (i.e., the step norm is below this fraction of the current solution norm).
   * @note 0 disables this criterion.
   * @note Default is @ref kDefaultRelStepTol.
   */
  real relStepTol = kDefaultRelStepTol;

  /** @brief Stop the solve if the line search figure of merit does not improve from the previous
   * iteration. */
  bool stopIfNoImprovement = false;

  // Retries and fallbacks

  /** @brief Positive Semi-Definite (PSD) projection mode for the dresidual matrix. */
  PsdProjectionMode psdProjMode = PsdProjectionMode::Default;

  /** @brief Fall back to gradient descent direction if the solver search direction fails. */
  bool gradientDescentFallback = false;

  // Explosion control

  /** @brief Enable heuristic explosion prevention. */
  bool explosionControl = true;

  /**
   * @brief Absolute divergence tolerance.
   *
   * @note Triggers explosion control if residual norm exceeds this value.
   * @note Used only if @ref explosionControl is true.
   */
  real absDivTol = 1e9_r;

  /**
   * @brief Relative divergence tolerance, relative to the initial residual.
   *
   * @note Triggers explosion control if relative residual norm exceeds this value.
   * @note Used only if @ref explosionControl is true.
   */
  real relDivTol = 1e4_r;

  // Line search

  /**
   * @brief Maximum number of line search iterations.
   *
   * @note Must be >= 1 unless @ref lineSearchType is @ref LineSearchType::None.
   */
  int lineSearchMaxIter = 4;

  /**
   * @brief Step length reduction factor for line search.
   *
   * @note Must be in (0, 1).
   */
  real lineSearchAlpha = 0.5_r;

  /**
   * @brief Wolfe condition parameter c1 (sufficient decrease).
   *
   * @note Must be in (0, 1).
   * @note Used only by @ref LineSearchType::Armijo, @ref LineSearchType::WolfeWeak, @ref
   * LineSearchType::WolfeStrong, and @ref LineSearchType::ArmijoOrResidualNorm line search types.
   */
  real lineSearchWolfe1 = 1e-4_r;

  /**
   * @brief Wolfe condition parameter c2 (curvature).
   *
   * @note Must be in (@ref lineSearchWolfe1, 1).
   * @note Used only by @ref LineSearchType::WolfeWeak and @ref LineSearchType::WolfeStrong line
   * search types.
   */
  real lineSearchWolfe2 = 0.9_r;

  /**
   * @brief Maximum relative increase in the objective to accept the step.
   *
   * @note Only applies to @ref LineSearchType::Simple.
   */
  real lineSearchMaxRelIncrease = 0.0_r;

  /** @brief Line search type. */
  LineSearchType lineSearchType = LineSearchType::Default;

  // Linear tolerance strategy

  /** @brief Strategy for adaptive linear solver tolerance (forcing term). */
  LinearToleranceStrategy linearToleranceStrategy = LinearToleranceStrategy::Default;

  // Other

  /** @brief Verbosity level for logging output. */
  VerbosityLevel verbosity = VerbosityLevel::Warning;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(NonLinearSolverParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::NonLinearSolverParams)
  MOCHI_FIELD(solverType)
  MOCHI_FIELD(dResidualAssemblyPeriod)
  MOCHI_FIELD(maxIter)
  MOCHI_FIELD(maxElapsedTimeSeconds)
  MOCHI_FIELD(convergenceMode)
  MOCHI_FIELD(absTol)
  MOCHI_FIELD(relTol)
  MOCHI_FIELD(relStepTol)
  MOCHI_FIELD(stopIfNoImprovement)
  MOCHI_FIELD(psdProjMode)
  MOCHI_FIELD(gradientDescentFallback)
  MOCHI_FIELD(explosionControl)
  MOCHI_FIELD(absDivTol)
  MOCHI_FIELD(relDivTol)
  MOCHI_FIELD(lineSearchMaxIter)
  MOCHI_FIELD(lineSearchAlpha)
  MOCHI_FIELD(lineSearchWolfe1)
  MOCHI_FIELD(lineSearchWolfe2)
  MOCHI_FIELD(lineSearchMaxRelIncrease)
  MOCHI_FIELD(lineSearchType)
  MOCHI_FIELD(linearToleranceStrategy)
  MOCHI_FIELD(verbosity)
  MOCHI_STRUCT_END()
};

} // namespace mochi
