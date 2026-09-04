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
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/utils/math_utils.h>

#include <type_traits>
#include <utility>

namespace mochi::krylov {

/** @brief Solve a linear system of equations using a preconditioned MINRES method.
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
 * @param[in] rhs The right-hand side vector b of \f$ A x = b\f$.
 * @param[in,out] x Vector containing the initial guess at input and the solution at output.
 * @param[in] prec The preconditioner application functor.
 *            The preconditioner has to be symmetric positive definite.
 * @param[in] iterMax Maximum number of iterations.
 *            iterMax must be between 0 and the size of A.
 * @param[in] statusCheck A functor called at each iteration to check the stop criteria.
 * @param[in] verbosity Verbosity level for logging output.
 * @param[in] initialGuessHint Indicates whether @p x is known to be zero. The zero hint skips the
 * initial matrix-vector product and requires @p x to be exactly zero.
 * @param[in] dot The dot operator. Must also handle a matrix-vector operation.
 * @param[in] vectorFactory Factory to create vectors of a given type.
 *
 * @return Linear solver status. Contains the convergence status, number of iterations, and achieved
 * absolute and relative residuals.
 *
 * @note It uses right preconditioning.
 * @note It minimizes the ||.||_{prec^-1} norm of the residual.
 * @note Complex arithmetic is not supported.
 * @note This implementation follows the pseudo-code described in [A note on preconditioners and
 * scalar products in Krylov subspace methods for self-adjoint problems in Hilbert space (Gunnel et
 * al., 2014)](http://emis.icm.edu.pl/journals/ETNA/vol.41.2014/pp13-20.dir/pp13-20.pdf). Additional
 * information is available at [MINRES (Stanford
 * SOL)](https://web.stanford.edu/group/SOL/software/minres).
 */
template <
    typename Op,
    typename Prec,
    typename Vector,
    typename VSol,
    typename Dot = UsualDot,
    typename StopCriterion = StatusImplicitResidualNorm<real>,
    typename VectorFactory = MatrixFactoryType<Vector>>
LinearSolverStatus MinRes(
    Op const& A,
    Vector const& rhs,
    VSol& x,
    Prec const& prec,
    int iterMax,
    StopCriterion statusCheck = {},
    VerbosityLevel verbosity = VerbosityLevel::Warning,
    InitialGuessHint initialGuessHint = InitialGuessHint::Unknown,
    Dot dot = {},
    VectorFactory vectorFactory = {}) {
  using Scalar = decltype(dot(rhs, rhs));

  static_assert(
      std::is_same_v<StopCriterion, StatusImplicitResidualNorm<Scalar>>,
      "The type 'StopCriterion' is currently not supported by MinRes.");

  int n = static_cast<int>(NumRows(x));
  MOCHI_ASSERT_VERBOSE(NumRows(x) == NumRows(rhs));
  MOCHI_ASSERT_VERBOSE(
      initialGuessHint != InitialGuessHint::Zero || dot(x, x) == 0,
      "InitialGuessHint::Zero requires an exactly zero initial guess.");

  iterMax = Min(n, iterMax);

  auto z = vectorFactory.GetSameAs(x);
  Solve(prec, rhs, z);

  auto gammaSqr = dot(rhs, z);
  if (!IsFinite(gammaSqr) || gammaSqr <= 0)
    MOCHI_UNLIKELY {
      if (dot(rhs, rhs) == 0) {
        SetZero(x);
        return LinearSolverStatus{
            .numIterDone = 0,
            .residualNorm = 0.0,
            .relativeResidualNorm = 0.0,
            .convergence = LinearSolverConvergenceStatus::Converged};
      }
      if (verbosity >= VerbosityLevel::Warning) {
        MOCHI_LOG_WARNING(
            "Preconditioner does not seem to be SPD. P-dot product: %e",
            static_cast<double>(gammaSqr));
      }
      return LinearSolverStatus{
          .numIterDone = 0,
          .residualNorm = 0.0,
          .relativeResidualNorm = 0.0,
          .convergence = LinearSolverConvergenceStatus::Diverged};
    }
  statusCheck.SetScaling(Sqrt(gammaSqr));

  auto Az = vectorFactory.GetSameAs(x);
  auto v = vectorFactory.GetSameAs(x);
  if (initialGuessHint == InitialGuessHint::Zero) {
    // With x_0 = 0, v_0 = rhs, so the solve above already computed z_0 = Prec^{-1} v_0.
    v = rhs;
  } else {
    Apply(A, x, Az);
    v = rhs - Az;
    Solve(prec, v, z);
  }

  gammaSqr = dot(v, z);
  if (!IsFinite(gammaSqr) || gammaSqr <= 0)
    MOCHI_UNLIKELY {
      if (dot.Norm(v) == 0) {
        return LinearSolverStatus{
            .numIterDone = 0,
            .residualNorm = 0.0,
            .relativeResidualNorm = 0.0,
            .convergence = LinearSolverConvergenceStatus::Converged};
      }
      if (verbosity >= VerbosityLevel::Warning) {
        MOCHI_LOG_WARNING(
            "Preconditioner does not seem to be SPD. P-dot product: %e",
            static_cast<double>(gammaSqr));
      }
      return LinearSolverStatus{
          .numIterDone = 0,
          .residualNorm = 0.0,
          .relativeResidualNorm = 0.0,
          .convergence = LinearSolverConvergenceStatus::Diverged};
    }

  auto gamma = Sqrt(gammaSqr);
  auto myStatus = statusCheck.CheckStatus(0, gamma, z, Az);
  if (myStatus != IterationStatus::Active) {
    return LinearSolverStatus{
        .numIterDone = 0,
        .residualNorm = static_cast<double>(statusCheck.GetLatestResidualNorm()),
        .relativeResidualNorm = static_cast<double>(statusCheck.GetLatestRelativeResidualNorm()),
        .convergence = IsConverged(myStatus) ? LinearSolverConvergenceStatus::Converged
                                             : LinearSolverConvergenceStatus::Diverged};
  }
  auto invGamma = Scalar(1) / gamma;
  z *= invGamma;
  v *= invGamma;

  Scalar eta = gamma;

  auto v_old = vectorFactory.GetSameAs(x);
  SetZero(v_old);
  auto v_new = vectorFactory.GetSameAs(x);
  auto z_new = vectorFactory.GetSameAs(x);

  auto w_old = vectorFactory.GetSameAs(x);
  SetZero(w_old);
  auto w = vectorFactory.GetSameAs(x);
  SetZero(w);
  auto w_new = vectorFactory.GetSameAs(x);

  Scalar c_old = 1, c = 1, c_new = 1;
  Scalar s_old = 0, s = 0, s_new = 0;

  for (int iter = 1; iter <= iterMax; ++iter) {
    Apply(A, z, Az);
    auto delta = dot(z, Az);

    v_new = Az - delta * v - gamma * v_old;
    Solve(prec, v_new, z_new);

    gammaSqr = dot(v_new, z_new);
    if (!IsFinite(gammaSqr) || gammaSqr <= 0)
      MOCHI_UNLIKELY {
        // Failure: non-finite gammaSqr (NaN/Inf from a non-finite operator/preconditioner), a
        // non-SPD preconditioner (gammaSqr < 0), or a singular one (gammaSqr == 0, v_new != 0).
        // Continuing would poison the solution with NaN/Inf.
        if (!IsFinite(gammaSqr) || dot.Norm(v_new) > 0) {
          if (verbosity >= VerbosityLevel::Warning) {
            MOCHI_LOG_WARNING(
                "Preconditioner does not seem to be SPD. P-dot product: %e",
                static_cast<double>(gammaSqr));
          }
          return LinearSolverStatus{
              .numIterDone = iter - 1,
              .residualNorm = static_cast<double>(statusCheck.GetLatestResidualNorm()),
              .relativeResidualNorm =
                  static_cast<double>(statusCheck.GetLatestRelativeResidualNorm()),
              .convergence = LinearSolverConvergenceStatus::Diverged};
        } else {
          // Lucky breakdown: gammaSqr == 0 and ||v_new|| == 0, so the Krylov subspace is
          // exhausted. Perform the final Givens rotation to update x before returning.
          Scalar alpha0 = c * delta - c_old * s * gamma;
          Scalar alpha1 = Abs(alpha0);
          if (alpha1 != Scalar(0)) {
            Scalar alpha2 = s * delta + c_old * c * gamma;
            Scalar alpha3 = s_old * gamma;
            Scalar invAlpha1 = Scalar(1) / alpha1;
            c_new = alpha0 * invAlpha1;
            w_new = invAlpha1 * z - (alpha3 * invAlpha1) * w_old - (alpha2 * invAlpha1) * w;
            x += c_new * eta * w_new;
            eta = Scalar(0); // s_new = gammaNew / alpha1 = 0, so eta *= (-s_new) = 0
          }
          // eta is now the implicit residual: zero if the rotation was performed (nonsingular
          // case), unchanged from the previous iteration otherwise (singular operator).
          myStatus = statusCheck.CheckStatus(iter, Abs(eta), z, Az);
          return LinearSolverStatus{
              .numIterDone = iter,
              .residualNorm = static_cast<double>(statusCheck.GetLatestResidualNorm()),
              .relativeResidualNorm =
                  static_cast<double>(statusCheck.GetLatestRelativeResidualNorm()),
              .convergence = IsConverged(myStatus) ? LinearSolverConvergenceStatus::Converged
                                                   : LinearSolverConvergenceStatus::Diverged};
        }
      }

    Scalar gammaNew = Sqrt(gammaSqr);
    Scalar alpha0 = c * delta - c_old * s * gamma;
    Scalar alpha1 = Sqrt(Sqr(alpha0) + gammaSqr);
    Scalar alpha2 = s * delta + c_old * c * gamma;
    Scalar alpha3 = s_old * gamma;

    Scalar invAlpha1 = Scalar(1) / alpha1;
    c_new = alpha0 * invAlpha1;
    s_new = gammaNew * invAlpha1;

    w_new = invAlpha1 * z - (alpha3 * invAlpha1) * w_old - (alpha2 * invAlpha1) * w;
    x += c_new * eta * w_new;

    eta *= (-s_new);

    myStatus = statusCheck.CheckStatus(iter, Abs(eta), z, Az);
#if MOCHI_DEBUG
    {
      //
      // Expensive debugging section
      // Do not run in release form
      //
      auto Axtmp = vectorFactory.GetSameAs(x);
      Apply(A, x, Axtmp);
      auto residual = vectorFactory.GetSameAs(x);
      residual = rhs - Axtmp;
      auto Rr = vectorFactory.GetSameAs(x);
      Solve(prec, residual, Rr);
      auto rNormB = Sqrt(Abs(dot(residual, Rr)));
      auto estimateNormB = statusCheck.GetLatestResidualNorm();
      if (estimateNormB > Scalar(1.1) * rNormB && verbosity >= VerbosityLevel::Warning)
        MOCHI_UNLIKELY {
          MOCHI_LOG_WARNING(
              "P-norm estimate %e is exceeds value %e",
              static_cast<double>(estimateNormB),
              static_cast<double>(rNormB));
        }
    }
#endif
    if (myStatus != IterationStatus::Active) {
      return LinearSolverStatus{
          .numIterDone = iter,
          .residualNorm = static_cast<double>(statusCheck.GetLatestResidualNorm()),
          .relativeResidualNorm = static_cast<double>(statusCheck.GetLatestRelativeResidualNorm()),
          .convergence = IsConverged(myStatus) ? LinearSolverConvergenceStatus::Converged
                                               : LinearSolverConvergenceStatus::Diverged};
    }
    gamma = gammaNew;

    //
    // gamma is not zero else MINRES would have converged
    //
    z_new *= (Scalar(1) / gamma);
    v_new *= (Scalar(1) / gamma);

    std::swap(v_old, v);
    std::swap(v, v_new);

    std::swap(w_old, w);
    std::swap(w, w_new);

    std::swap(z, z_new);

    std::swap(c_old, c);
    std::swap(c, c_new);

    std::swap(s_old, s);
    std::swap(s, s_new);

  } // for (int iter = 1; iter <= iterMax; ++iter)

  return LinearSolverStatus{
      .numIterDone = iterMax,
      .residualNorm = static_cast<double>(statusCheck.GetLatestResidualNorm()),
      .relativeResidualNorm = static_cast<double>(statusCheck.GetLatestRelativeResidualNorm()),
      .convergence = LinearSolverConvergenceStatus::Stopped};
}

} // namespace mochi::krylov
