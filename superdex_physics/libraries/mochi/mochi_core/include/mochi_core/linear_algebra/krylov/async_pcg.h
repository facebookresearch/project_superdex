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

#include <mochi_core/linear_algebra/krylov/parallel_matrix_vector_product_pool.h>
#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_traits.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/utils/math_utils.h>

#include <functional>
#include <type_traits>
#include <utility>

namespace mochi::krylov {

/** @brief Solve a linear system using an asynchronous preconditioned CG algorithm. It is a
 * reformulation of the preconditioned CG algorithm to compute the matrix-vector products in
 * parallel to the orthogonalization and convergence check. It is equivalent to regular
 * preconditioned CG in exact-precision arithmetic, but has inferior stability properties in
 * finite-precision arithmetic. Stability is enhanced via restarting.
 *
 * @tparam MatType Type of the matrix.
 * @tparam RhsType Type of the right-hand side vector.
 * @tparam SolType Type of the solution vector.
 * @tparam Prec Type of the preconditioner.
 * @tparam Dot Type of the dot operation object/functor.
 * @tparam StopCriterion Type of the stop criteria checker.
 * @tparam VectorFactory Type of the vector factory.
 *
 * @param[in] A The matrix of the linear system.
 * @param[in] b The right-hand side vector of \f$ A x = b\f$.
 * @param[in,out] x Vector containing the initial guess at input and the solution at output.
 * @param[in] prec The preconditioner application functor.
 * @param[in] maxIter Maximum number of iterations.
 * @param[in,out] statusCheck A functor called every iteration to check the stop criteria. The norm
 * used in the stop criteria is determined by this object.
 * @param[in] abortIfNotSpd Boolean to abort the solve if the matrix is detected not to be symmetric
 * positive definite. Default is false.
 * @param[in] verbosity Verbosity level for logging output.
 * @param[in] usePolakRibiere Boolean to use the Polak-Ribiere formula for beta (if true) or the
 * Fletcher-Reeves formula (if false). Default is true.
 * @param[in] dot The dot operator. Must also handle matrix-vector operations.
 * @param[in] vectorFactory Factory to create vectors of a given type.
 * @param[in] restartPeriod Positive integer indicating every how many iterations to restart.
 * Default is 100.
 *
 * @return Linear solver status. Contains the number of iterations and the achieved absolute and
 * relative residuals. "maxIter+1" is used to indicate that the maximum number of iterations was
 * reached without convergence.
 *
 * @details The algorithm is designed to maximize the amount of work that can be performed in
 * parallel to the matrix-vector product. Similar variations of CG have been proposed in the
 * literature to minimize the number of global reductions and/or to maximize the amount of work that
 * can be performed in parallel to the global reduction(s), e.g., [Hiding global synchronization
 * latency in the preconditioned Conjugate Gradient algorithm (Ghysels and Vanroose,
 * 2014)](https://www.sciencedirect.com/science/article/abs/pii/S0167819113000719)
 *
 * @note The input matrix must be a supported matrix or linear operator type. Matrix application
 * functors are NOT supported.
 * @note It uses left preconditioning.
 * @note Parallel preconditioners are supported but NOT recommended: The parallel matrix-vector
 * product workers don't yield until the linear solve is complete, which may degrade performance of
 * the preconditioner solves if they are performed in parallel due to fewer available parallel
 * workers.
 * @note The norm used in the stop criteria is specified by the object 'statusCheck'.
 * @note Complex arithmetic is not supported.
 */
template <
    typename MatType,
    typename RhsType,
    typename SolType,
    typename Prec,
    typename Dot = UsualDot,
    typename StopCriterion = StatusResidualL2<Dot, typename MatType::NonConstScalar>,
    typename VectorFactory = MatrixFactoryType<RhsType>>
LinearSolverStatus AsyncPCG(
    MatType const& A,
    RhsType const& b,
    SolType& x,
    Prec const& prec,
    int maxIter,
    StopCriterion& statusCheck,
    bool abortIfNotSpd = false,
    VerbosityLevel verbosity = VerbosityLevel::Warning,
    bool usePolakRibiere = true,
    Dot dot = {},
    VectorFactory vectorFactory = {},
    int const restartPeriod = 100) {
  static_assert(IsLinearOperator<MatType>, "Unsupported matrix type");
  using NonConstScalar = typename MatType::NonConstScalar;
  using OwnerSolType = decltype(vectorFactory.GetSameAs(x));
  static_assert(
      std::is_same_v<StopCriterion, StatusResidualL2<Dot, NonConstScalar>> ||
          std::is_same_v<StopCriterion, StatusPreconditionedResidualL2<Dot, NonConstScalar>> ||
          std::is_same_v<StopCriterion, StatusResidualPreconditionerInduced<Dot, NonConstScalar>>,
      "Unsupported stop criterion");
  MOCHI_ASSERT(!abortIfNotSpd, "Asynchronous PCG does not support aborting if not SPD.");
  MOCHI_ASSERT_VERBOSE(restartPeriod > 0, "Restart period must be positive.");
  constexpr bool kStatusCheckNeedsPrecResidual =
      std::is_same_v<StopCriterion, StatusPreconditionedResidualL2<Dot, NonConstScalar>> ||
      std::is_same_v<StopCriterion, StatusResidualPreconditionerInduced<Dot, NonConstScalar>>;
  constexpr bool kCheckStatusComputesRTz =
      std::is_same_v<StopCriterion, StatusResidualPreconditionerInduced<Dot, NonConstScalar>>;

  auto opA = ParallelMatrixVectorProductPool(A, /*masterPerformsProduct*/ false);
  auto r = vectorFactory.GetCopy(b);
  auto p = vectorFactory.GetSameAs(x);
  auto z = vectorFactory.GetSameAs(x);
  auto Ap = vectorFactory.GetSameAs(x);
  auto ApPrev = vectorFactory.GetSameAs(x);
  auto ApNext = vectorFactory.GetSameAs(x);
  auto PAp = vectorFactory.GetSameAs(x);
  auto APAp = vectorFactory.GetSameAs(x);

  statusCheck.SetScaling(r, prec, z);

  Apply(opA, x, Ap); // A * x_0
  r -= Ap; // r_0 = b - A * x_0

  IterationStatus status = {};
  if constexpr (kStatusCheckNeedsPrecResidual) {
    Solve(prec, r, z); // z_0 = Prec^{-1} r_0
    status = statusCheck.CheckStatus(0, r, z, p, Ap); // p and Ap are not used if iter = 0
  } else {
    status = statusCheck.CheckStatus(0, r, z, p, Ap); // z, p and Ap are not used if iter = 0
    if (status == IterationStatus::Active) { // Skip if the stop criterion was met
      Solve(prec, r, z); // z_0 = Prec^{-1} r_0
    }
  }

  if (status != IterationStatus::Active) {
    return LinearSolverStatus{
        .numIterDone = 0,
        .residualNorm = static_cast<double>(statusCheck.GetLatestResidualNorm()),
        .relativeResidualNorm = static_cast<double>(statusCheck.GetLatestRelativeResidualNorm()),
        .converged = IsConverged(status)};
  }

  p = z; // p_0 = z_0
  NonConstScalar rTz{}; // r_0^T z_0
  if constexpr (kCheckStatusComputesRTz) {
    rTz = statusCheck.GetLatestResidualNormSqr();
  } else {
    rTz = dot(r, z);
  }

  Apply(opA, p, ApNext); // A * p_0
  SetZero(Ap);
  bool restart = false;
  bool isPrecDotZero = false;
  NonConstScalar beta = 0, betaPrev = 0, alpha = 0;
  int iter = 1;

  auto rotateBuffersAndUpdateX = [&]() {
    OwnerSolType temp = std::move(ApPrev);
    ApPrev = std::move(Ap);
    Ap = std::move(ApNext);
    ApNext = std::move(temp);
    betaPrev = beta;

    auto const pTAp = dot(p, Ap);
    if (pTAp < 0 && verbosity >= VerbosityLevel::Warning)
      MOCHI_UNLIKELY {
        MOCHI_LOG_WARNING(
            "Matrix A does not seem to be PSD (%e) at iteration %d.",
            static_cast<double>(pTAp),
            iter);
      }
    else if (pTAp == 0 && verbosity >= VerbosityLevel::Error)
      MOCHI_UNLIKELY {
        MOCHI_LOG_ERROR("Zero A-dot product at iteration %d.", iter);
      }

    alpha = rTz / pTAp;
    x += alpha * p; // x_i = x_{i-1} + alpha_i p_i
  };

  std::function<void()> beforeProduct = [&]() {
    if (!restart) {
      // TODO: The preconditioner solve could be performed in the matrix-vector product task
      // (instead of in the before-product work by the master thread), but this requires significant
      // changes to the parallelization routines.
      Solve(prec, ApNext, PAp); // Prec^{-1} * A * p_i
    } else {
      rotateBuffersAndUpdateX();
    }
  };

  std::function<void()> duringProduct = [&]() {
    if (!restart) {
      rotateBuffersAndUpdateX();
      r -= alpha * Ap; // r_i = r_{i-1} - alpha_i A * p_i

      beta = 0;
      auto updatePrecResidual = [&]() {
        if (usePolakRibiere && (iter > 1)) {
          beta = dot(r, z); // Compute r_i^T z_{i-1} before losing z_{i-1}
        }
        z -= alpha * PAp; // z_i = z_{i-1} - alpha_i Prec^{-1} * A * p_i
      };

      if constexpr (kStatusCheckNeedsPrecResidual) {
        updatePrecResidual();
        status = statusCheck.CheckStatus(iter, r, z, p, Ap);
      } else {
        status = statusCheck.CheckStatus(iter, r, z, p, Ap); // z is not used
        if (status == IterationStatus::Active) { // Skip if the stop criterion was met
          updatePrecResidual();
        }
      }

      if (status == IterationStatus::Active) {
        auto const rTzPrev = rTz;
        if constexpr (kCheckStatusComputesRTz) {
          rTz = statusCheck.GetLatestResidualNormSqr();
        } else {
          rTz = dot(r, z);
        }

        if (rTz == 0)
          MOCHI_UNLIKELY {
            if (verbosity >= VerbosityLevel::Error) {
              // The residual is not zero at this point. The preconditioner may be singular.
              MOCHI_LOG_ERROR("Zero preconditioner-dot product at iteration %d.", iter);
            }
            isPrecDotZero = true;
            return;
          }

        beta = (rTz - beta) / rTzPrev;
        p = z + beta * p;

        // Compute A * p_{i+1} as
        // (beta_i + 1) * A * p_i - beta_{i-1} * A * p_{i-1} - alpha_i * A * Prec^{-1} * A * p_i
        // This identity derives from:
        // 1. p_{i+1} = z_{i+1} + beta_i * p_i
        // 2. z_{i+1} = z_i - alpha_i * Prec^{-1} * A * p_i
        // 3. z_i = p_i - beta_{i-1} * p_{i-1}
        ApNext = (beta + 1) * Ap - betaPrev * ApPrev;
      }
    }
  };

  std::function<void()> afterProduct = [&]() {
    if (!restart) {
      if (status == IterationStatus::Active) {
        ApNext -= alpha * APAp;
      }
    } else {
      r = b - ApNext; // r_i = b - A * x_i

      if constexpr (kStatusCheckNeedsPrecResidual) {
        Solve(prec, r, z); // z_i = Prec^{-1} r_i
        status = statusCheck.CheckStatus(iter, r, z, p, Ap);
      } else {
        status = statusCheck.CheckStatus(iter, r, z, p, Ap); // z is not used
        if (status == IterationStatus::Active) { // Skip if the stop criterion was met
          Solve(prec, r, z); // z_i = Prec^{-1} r_i
        }
      }

      if (status == IterationStatus::Active) {
        if constexpr (kCheckStatusComputesRTz) {
          rTz = statusCheck.GetLatestResidualNormSqr();
        } else {
          rTz = dot(r, z);
        }

        if (rTz == 0)
          MOCHI_UNLIKELY {
            if (verbosity >= VerbosityLevel::Error) {
              // The residual is not zero at this point. The preconditioner may be singular.
              MOCHI_LOG_ERROR("Zero preconditioner-dot product at iteration %d.", iter);
            }
            isPrecDotZero = true;
            return;
          }

        beta = 0;
        p = z;
        Apply(opA, p, ApNext);
      }
    }
  };

  for (iter = 1; iter <= maxIter; ++iter) {
    restart = (iter % restartPeriod == 0);
    if (restart) {
      opA.AsyncApply(x, ApNext, beforeProduct, duringProduct, afterProduct);
    } else {
      opA.AsyncApply(PAp, APAp, beforeProduct, duringProduct, afterProduct);
    }
    if (status != IterationStatus::Active || isPrecDotZero) {
      return LinearSolverStatus{
          .numIterDone = iter,
          .residualNorm = static_cast<double>(statusCheck.GetLatestResidualNorm()),
          .relativeResidualNorm = static_cast<double>(statusCheck.GetLatestRelativeResidualNorm()),
          .converged = IsConverged(status)};
    }
  }

  return LinearSolverStatus{
      .numIterDone = maxIter + 1,
      .residualNorm = static_cast<double>(statusCheck.GetLatestResidualNorm()),
      .relativeResidualNorm = static_cast<double>(statusCheck.GetLatestRelativeResidualNorm()),
      .converged = false};
}

} // namespace mochi::krylov
