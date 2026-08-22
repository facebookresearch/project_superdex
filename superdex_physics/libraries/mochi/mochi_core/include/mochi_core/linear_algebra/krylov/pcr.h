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

#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_traits.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/utils/math_utils.h>

#include <type_traits>

namespace mochi::krylov {

/** @brief Solve a linear system using a preconditioned Conjugate Residual method.
 *
 * @tparam Op Type of the matrix application operator.
 * @tparam Vector Vector type for the RHS.
 * @tparam VSol Vector type for the solution.
 * @tparam Prec Type of the preconditioner.
 * @tparam Dot Type of the dot operation object/functor.
 * @tparam StopCriterion Type of the stop criteria checker.
 * @tparam VectorFactory Type of the vector factory.
 *
 * @param[in] A The matrix application operator.
 * @param[in] b The right-hand side vector of \f$ A x = b\f$.
 * @param[in,out] x Vector containing the initial guess at input and the solution at output.
 * @param[in] prec The preconditioner application functor.
 * @param[in] maxIter Maximum number of iterations.
 * @param[in,out] statusCheck A functor called at each iteration to check the stop criteria.
 * @param[in] verbosity Verbosity level for logging.
 * @param[in] dot The dot operator. Must also handle a matrix-vector operation.
 * @param[in] vectorFactory Factory to create vectors of a given type.
 *
 * @return Iteration data. Contains the number of iterations and the achieved absolute and relative
 * residuals. "maxIter+1" is used to indicate that the maximum number of iterations was reached
 * without convergence.
 *
 * @note It uses left preconditioning.
 * @note The norm used in the stop criteria is specified by the object 'statusCheck'.
 * @note Complex arithmetic is not supported.
 */
template <
    typename Op,
    typename Vector,
    typename VSol,
    typename Prec,
    typename Dot = UsualDot,
    typename StopCriterion = StatusResidualL2<Dot, real>,
    typename VectorFactory = MatrixFactoryType<Vector>>
LinearSolverStatus PCR(
    Op const& A,
    Vector const& b,
    VSol& x,
    Prec const& prec,
    int maxIter,
    StopCriterion& statusCheck,
    VerbosityLevel verbosity = VerbosityLevel::Warning,
    Dot dot = {},
    VectorFactory vectorFactory = {}) {
  auto r = vectorFactory.GetCopy(b);
  auto Ap = vectorFactory.GetSameAs(b);
  auto Mm1Ap = vectorFactory.GetSameAs(b);
  auto Az = vectorFactory.GetSameAs(b);

  auto p = vectorFactory.GetSameAs(x);
  auto z = vectorFactory.GetSameAs(x);

  using RealScalar = decltype(mochi::Abs(dot(r, r)));

  static_assert(
      std::is_same_v<StopCriterion, StatusResidualL2<Dot, RealScalar>> ||
          std::is_same_v<StopCriterion, StatusPreconditionedResidualL2<Dot, RealScalar>>,
      "The type 'StopCriterion' is currently not supported by PCR.");
  constexpr bool kNeedPrecResidual =
      std::is_same_v<StopCriterion, StatusPreconditionedResidualL2<Dot, RealScalar>>;

  statusCheck.SetScaling(r, prec, z);

  Apply(A, x, Ap);
  r -= Ap;

  IterationStatus myStatus{};

  if constexpr (kNeedPrecResidual) {
    Solve(prec, r, z); // z_0 = Prec^{-1} r_0
    //--- p and Ap will not be stored when iter = 0
    myStatus = statusCheck.CheckStatus(0, r, z, p, Ap);
  } else {
    //--- z, p, and Ap will not be accessed in CheckStatus when iter = 0
    myStatus = statusCheck.CheckStatus(0, r, z, p, Ap);
    //--- Skip the preconditioner application if we exit
    if (myStatus == IterationStatus::Active) {
      Solve(prec, r, z); // z_0 = Prec^{-1} r_0
    }
  }

  if (myStatus != IterationStatus::Active) {
    return LinearSolverStatus{
        .numIterDone = 0,
        .residualNorm = statusCheck.GetLatestResidualNorm(),
        .relativeResidualNorm = statusCheck.GetLatestRelativeResidualNorm(),
        .converged = IsConverged(myStatus)};
  }

  p = z;
  Apply(A, z, Az);
  Ap = Az;
  auto zTAz_current = dot(z, Az); // z_0^T A z_0
  for (int iter = 1; iter <= maxIter; ++iter) {
    Solve(prec, Ap, Mm1Ap);
    auto const ApTMm1Ap = dot(Ap, Mm1Ap);

    auto const alpha = zTAz_current / ApTMm1Ap;
    x += alpha * p; // x_i = x_{i-1} + alpha p
    r -= alpha * Ap; // r_i = r_{i-1} - alpha A*p
    z -= alpha * Mm1Ap; // z_i = z_{i-1} - alpha M^-1 A p_{i-1}

    Apply(A, z, Az); // Az := A z_i
    auto const zTAz_old = zTAz_current;
    zTAz_current = dot(z, Az);
    RealScalar beta = zTAz_current / zTAz_old;
    p = z + beta * p;
    Ap = Az + beta * Ap;

    myStatus = statusCheck.CheckStatus(iter, r, z, p, Ap);

    if (myStatus != IterationStatus::Active) {
      return LinearSolverStatus{
          .numIterDone = iter,
          .residualNorm = statusCheck.GetLatestResidualNorm(),
          .relativeResidualNorm = statusCheck.GetLatestRelativeResidualNorm(),
          .converged = IsConverged(myStatus)};
    }

    if (zTAz_current == 0)
      MOCHI_UNLIKELY {
        if (verbosity >= VerbosityLevel::Error) {
          // The residual is not zero at this point. The preconditioner may have a singularity.
          MOCHI_LOG_ERROR("Zero Preconditioner-dot product at iteration %d", iter);
        }
        return LinearSolverStatus{
            .numIterDone = iter,
            .residualNorm = statusCheck.GetLatestResidualNorm(),
            .relativeResidualNorm = statusCheck.GetLatestRelativeResidualNorm(),
            .converged = false};
      }
  } // for (int iter = 1; iter <= maxIter; ++iter)

  return {
      .numIterDone = maxIter + 1,
      .residualNorm = statusCheck.GetLatestResidualNorm(),
      .relativeResidualNorm = statusCheck.GetLatestRelativeResidualNorm(),
      .converged = false};
}

} // namespace mochi::krylov
