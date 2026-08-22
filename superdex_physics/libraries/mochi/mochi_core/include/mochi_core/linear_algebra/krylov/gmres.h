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
#include <vector>

/// @brief Namespace for utility functions used in GMRES
namespace mochi::krylov::details {

/** @brief Apply a Givens rotation for real or complex numbers. */
template <typename Scalar>
void ApplyGivens(Scalar c, Scalar s, Scalar& f, Scalar& g) {
  Scalar temp1 = c * f + s * g;
  g = -s * f + c * g;
  f = temp1;
}

/** @brief compute c and s of the Givens' rotation matrix.
 * @details The coefficients are such that the rotation zeroes out g.
 * @tparam Scalar
 * @param[in] f First term of the vector for which G zeroes out the second term.
 * @param[in] g Second term of the vector.
 * @param[in] verbosity Verbosity level for logging.
 * @return
 */
template <typename Scalar>
auto MakeGivens(Scalar f, Scalar g, VerbosityLevel verbosity) {
  auto givensNorm = Sqrt(Sqr<Scalar>(f) + Sqr<Scalar>(g));
  if (givensNorm == Scalar(0))
    MOCHI_UNLIKELY {
      if (verbosity >= VerbosityLevel::Error) {
        MOCHI_LOG_ERROR("Zero vector for Givens rotation matrix.");
      }
      return std::pair{Scalar(1), Scalar(0)};
    }
  else {
    return std::pair{f / givensNorm, g / givensNorm};
  }
}

/// @brief Apply a Gram Schmidt twice for stability.
///
/// @tparam Mat Parameter type for the set of vectors Q
/// @tparam Vec Parameter type for the vector v
/// @tparam ResVec Parameter type for the result h
/// @tparam Dot Parameter type for the dot product
/// @param[in] Q Set of orthonormal vectors
/// @param[in,out] v Vector to orthogonalize
/// @param[in,out] h Coordinates in Q-basis (i.e. h = Q^T v )
/// @param[in] dot Dot product
template <typename Mat, typename Vec, typename ResVec, typename Dot>
void Orthogonalize(Mat const& Q, Vec&& v, ResVec&& h, Dot& dot) {
  if constexpr (mochi::IsCuda<Mat>) {
    auto h1 = dot.MatrixWise(Q, v);
    v -= Q * h1;
    auto h2 = dot.MatrixWise(Q, v);
    v -= Q * h2;
    h1 += h2;
    //--- Note that we need to keep the pair (h1, h2) in case
    //--- (h1, h2) are on the GPU and h is on the CPU
    h = h1;
  } else {
    h = dot.MatrixWise(Q, v);
    v -= Q * h;
    auto h2 = dot.MatrixWise(Q, v);
    v -= Q * h2;
    h += h2;
  }
}

template <
    typename Mat,
    typename HRhs,
    typename PrecOp,
    typename Space,
    typename VWork,
    typename VSol,
    typename VectorFactory>
void SolveHessenbergSystem(
    int k,
    Mat const& H,
    HRhs& b,
    PrecOp const& prec,
    Space const& Q,
    VWork& z,
    VWork& Mz,
    VSol& x,
    [[maybe_unused]] VectorFactory& vectorFactory) {
  UpperSolveInPlace(Block(H, 0, 0, k, k), TopRows(b, k));
  auto Qleft = LeftCols(Q, k);
  if constexpr (mochi::IsCuda<decltype(Q)>) {
    auto bTop = vectorFactory.template CreateNew<true>(k);
    bTop = TopRows(b, k);
    z = Qleft * bTop;
  } else {
    auto bTop = TopRows(b, k);
    z = Qleft * bTop;
  }
  Solve(prec, z, Mz);
  x += Mz;
}

} // namespace mochi::krylov::details

namespace mochi::krylov {

/** @brief Solve a linear system of equations using a preconditioned GMRES method.
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
 * @param[in] iterMax Maximum number of iterations.
 * @param[in] statusCheck A functor called at each iteration to check the stop criteria.
 * @param[in] restartSize Size of Krylov space triggering a restart (default = no restart)
 * @param[in] verbosity Verbosity level for logging output.
 * @param[in] dot The dot operator. Must also handle a matrix-vector operation.
 * @param[in] vectorFactory Factory to create vectors of a given type.
 *
 * @return Linear solver status. Contains the number of iterations and the achieved absolute and
 * relative residuals. "iterMax+1" is used to indicate that the maximum number of iterations was
 * reached without convergence.
 *
 * @note It uses right preconditioning.
 * @note The norm used in the stop criteria is specified by the object 'statusCheck'.
 * @note Complex arithmetic is not supported.
 */
template <
    typename Op,
    typename Prec,
    typename Vector,
    typename VSol,
    typename Dot = UsualDot,
    typename StopCriterion = StatusImplicitResidualNorm<real>,
    typename VectorFactory = MatrixFactoryType<Vector>>
LinearSolverStatus GMRes(
    Op const& A,
    Vector const& rhs,
    VSol& x,
    Prec const& prec,
    int iterMax,
    StopCriterion statusCheck = {},
    int restartSize = 0,
    VerbosityLevel verbosity = VerbosityLevel::Warning,
    Dot dot = {},
    VectorFactory vectorFactory = {}) {
  using namespace details;

  auto residual = vectorFactory.GetCopy(rhs);
  using Scalar = decltype(dot(residual, residual));
  static_assert(
      std::is_same_v<StopCriterion, StatusImplicitResidualNorm<Scalar>>,
      "The type 'StopCriterion' is currently not supported by GMRes.");

  int n = static_cast<int>(NumRows(x));
  MOCHI_ASSERT_VERBOSE(NumRows(x) == NumRows(rhs));

  restartSize = Min(n, (restartSize <= 0) ? iterMax : restartSize, iterMax);

  auto Ap = vectorFactory.GetSameAs(rhs);
  auto z = vectorFactory.GetSameAs(rhs);

  //--- Will use the typename RealScalar for real non-negative scalars.
  using RealScalar = Scalar;

  RealScalar const bNorm = dot.Norm(residual);
  if (bNorm == 0) {
    SetZero(x);
    return LinearSolverStatus{
        .numIterDone = 0, .residualNorm = 0.0, .relativeResidualNorm = 0.0, .converged = true};
  }
  statusCheck.SetScaling(bNorm);

  Apply(A, x, Ap);
  residual -= Ap;

  auto resNorm = dot.Norm(residual);
  //--- when iter = 0, z and Ap are ignored
  auto myStatus = statusCheck.CheckStatus(0, resNorm, z, Ap);
  if (myStatus != IterationStatus::Active) {
    return LinearSolverStatus{
        .numIterDone = 0,
        .residualNorm = static_cast<double>(resNorm),
        .relativeResidualNorm = static_cast<double>(resNorm / bNorm),
        .converged = IsConverged(myStatus)};
  }

  // Note that the vector `b` will be stored on the CPU
  // because the algorithm needs entry-wise access.
  auto b = vectorFactory.template CreateNew<false>(restartSize + 1);
  b(0) = resNorm;

  // Note that the directions in `Q` will have
  // the same location as the input vectors.
  auto Q = vectorFactory.CreateNew(n, restartSize + 1);

  // Note that the matrix `H` will be stored on the CPU
  // because the algorithm needs entry-wise access.
  auto H = vectorFactory.template CreateNew<false>(restartSize + 1, restartSize);

  std::vector<Scalar> cs;
  cs.reserve(restartSize + 1);
  std::vector<Scalar> sn;
  sn.reserve(restartSize + 1);

  Col(Q, 0) = residual;
  Col(Q, 0) *= Scalar(1) / b(0);

  int iter = 1;
  for (int totalIter = 1; totalIter <= iterMax; ++totalIter) {
    // z = P^-1 v[iter-1]
    Solve(prec, Col(Q, iter - 1), z);
    // v[iter] = A P^-1 v[iter-1]
    Apply(A, z, Col(Q, iter));
    // Orthogonalize to the previous vectors
    Orthogonalize(LeftCols(Q, iter), Col(Q, iter), Block(H, 0, iter - 1, iter, 1), dot);
    H(iter, iter - 1) = dot.Norm(Col(Q, iter));

    // Copy the norm of Col(Q, iter) as H(iter, iter - 1) will be updated
    // We do not normalize Col(Q, iter) here as we may want
    // to store the vectors z and Az (= Col(Q, iter))
    auto const normQ = H(iter, iter - 1);

    // Apply the previous givens rotations
    for (int i = 0; i < iter - 1; ++i) {
      ApplyGivens<Scalar>(cs[i], sn[i], H(i, iter - 1), H(i + 1, iter - 1));
    }
    auto [c, s] = MakeGivens<Scalar>(H(iter - 1, iter - 1), H(iter, iter - 1), verbosity);
    cs.push_back(c);
    sn.push_back(s);
    ApplyGivens<Scalar>(c, s, H(iter - 1, iter - 1), H(iter, iter - 1));
    b(iter) = {};
    ApplyGivens<Scalar>(c, s, b(iter - 1), b(iter));

    myStatus = statusCheck.CheckStatus(totalIter, Abs(b(iter)), z, Col(Q, iter));
    if (myStatus != IterationStatus::Active) {
      details::SolveHessenbergSystem(iter, H, b, prec, Q, z, Ap, x, vectorFactory);
      return LinearSolverStatus{
          .numIterDone = totalIter,
          .residualNorm = statusCheck.GetLatestResidualNorm(),
          .relativeResidualNorm = statusCheck.GetLatestRelativeResidualNorm(),
          .converged = IsConverged(myStatus)};
    }

    if (normQ == 0)
      MOCHI_UNLIKELY {
        if (verbosity >= VerbosityLevel::Error) {
          MOCHI_LOG_ERROR(
              "Redundant Krylov direction without convergence at iteration %d.", totalIter);
        }
        // Krylov subspace exhausted. Solve with the current Hessenberg system.
        details::SolveHessenbergSystem(iter, H, b, prec, Q, z, Ap, x, vectorFactory);
        return LinearSolverStatus{
            .numIterDone = totalIter,
            .residualNorm = statusCheck.GetLatestResidualNorm(),
            .relativeResidualNorm = statusCheck.GetLatestRelativeResidualNorm(),
            .converged = false};
      }

    Col(Q, iter) *= Scalar(1) / normQ;

    if ((iter == restartSize) && (totalIter < iterMax)) {
      //--- Compute the latest approximation
      details::SolveHessenbergSystem(restartSize, H, b, prec, Q, z, Ap, x, vectorFactory);
      //--- Compute the updated residual
      residual = rhs;
      Apply(A, x, Ap);
      residual -= Ap;
      //--- Reset the data for GMRes
      b(0) = dot.Norm(residual);
      Col(Q, 0) = residual;
      Col(Q, 0) *= Scalar(1) / b(0);
      iter = 0;
      cs.clear();
      sn.clear();
    }
    iter += 1;

  } // for (int totalIter = 1; totalIter <= iterMax; ++totalIter)

  details::SolveHessenbergSystem(iter - 1, H, b, prec, Q, z, Ap, x, vectorFactory);
  return LinearSolverStatus{
      .numIterDone = iterMax + 1,
      .residualNorm = Abs(b(iter - 1)),
      .relativeResidualNorm = Abs(b(iter - 1)) / bNorm,
      .converged = false};
}

} // namespace mochi::krylov
