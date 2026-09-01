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

#include <mochi_core/mochi_config.h>

#if MOCHI_USE_CUDA

#include <mochi_core/linear_algebra/cuda/cuda_bsr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_csr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/krylov/iteration_status.h>
#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/utils/verbosity_params.h>

#include <functional>
#include <tuple>
#include <type_traits>

namespace mochi::krylov {

/**
 * @brief Function for the actual implementation of GMRes on device.
 * @return Tuple of number of iterations performed, final residual, final relative residual, and
 * convergence status.
 */
template <typename Scalar>
std::tuple<int, double, double, IterationStatus> CudaGMRes_impl(
    std::function<
        void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar>& Av)> const& A,
    mochi::CudaVectorView<Scalar const> bv,
    Scalar bNorm,
    mochi::CudaVectorView<Scalar> xv,
    std::function<
        void(mochi::CudaVectorView<Scalar> const& x, mochi::CudaVectorView<Scalar> Px)> const& prec,
    int iterMax,
    double aTol,
    double rTol,
    double dTol,
    int restartSize);

/** @brief Solve a linear system using a right-preconditioned GMRes method.
 *
 * @tparam Op Type of the matrix application operator.
 * @tparam Vector Vector type for the RHS.
 * @tparam VSol Vector type for the solution.
 * @tparam Prec Type of the preconditioner.
 *
 * @param[in] A The matrix application operator.
 * @param[in] b The right-hand side vector b of \f$ A x = b\f$.
 * @param[in,out] x Vector containing the initial guess at input and the solution at output.
 * @param[in] P The preconditioner application functor.
 * @param[in] iterMax Maximum number of iterations.
 * @param[in] aTol Absolute tolerance
 * @param[in] rTol Relative tolerance
 * @param[in] dTol Divergence tolerance
 * @param[in] restartSize Size of Krylov space triggering a restart (default = 0 = no restart)
 * @param[in] verbosity Verbosity level for logging output.
 *
 * @return Linear solver status. Contains the convergence status, number of iterations, and achieved
 * absolute and relative residuals.
 *
 * @note
 * It uses right preconditioning.
 *
 * @note
 * Complex arithmetic is not supported.
 *
 * @note
 * The implementation assumes that the application of A and of P use the CuBLAS static handle,
 * the CuSparse static handle, or the stream attached to one of these handles.
 *
 */
template <typename Op, typename Vector, typename VSol, typename Prec>
  requires(mochi::IsCuda<Op> && mochi::IsCuda<Vector> && mochi::IsCuda<VSol>)
LinearSolverStatus CudaGMRes(
    Op const& A,
    Vector const& b,
    VSol& x,
    Prec const& P,
    int iterMax,
    double aTol,
    double rTol,
    double dTol,
    int restartSize = 0,
    [[maybe_unused]] VerbosityLevel verbosity = VerbosityLevel::Warning) {
  using Scalar = std::remove_pointer_t<decltype(b.Data())>;
  using NonConstScalar = std::remove_const_t<Scalar>;
  //--- Represent the operator A with a function to "hide the type" of A.
  auto Afunc = [&](mochi::CudaVectorView<NonConstScalar> const& v,
                   mochi::CudaVectorView<NonConstScalar>& Av) { Apply(A, v, Av); };
  auto prec = [&](mochi::CudaVectorView<NonConstScalar> const& v,
                  mochi::CudaVectorView<NonConstScalar> Pv) { P(v, Pv); };

  restartSize = mochi::Min<int>(b.Rows(), (restartSize <= 0) ? iterMax : restartSize, iterMax);

  //--- Convert to CudaVectorView
  MOCHI_ASSERT(((b.Cols() == x.Cols()) && (x.Cols() == 1)), "Incompatible number of columns");
  mochi::CudaVectorView<Scalar> bv(b.Data(), b.Rows());
  mochi::CudaVectorView<NonConstScalar> xv(x.Data(), x.Rows());
  //--- Check whether the right hand side is the zero vector.
  auto const bNorm = bv.Norm();
  if (bNorm == 0) {
    SetZero(x);
    return LinearSolverStatus{
        .numIterDone = 0,
        .residualNorm = 0.0,
        .relativeResidualNorm = 0.0,
        .convergence = LinearSolverConvergenceStatus::Converged};
  }
  //--- Run the GMRes algorithm
  auto info = CudaGMRes_impl<NonConstScalar>(
      Afunc, bv, bNorm, xv, prec, iterMax, aTol, rTol, dTol, restartSize);
  auto const iterationStatus = std::get<3>(info);
  //
  return LinearSolverStatus{
      .numIterDone = std::get<0>(info),
      .residualNorm = std::get<1>(info),
      .relativeResidualNorm = std::get<2>(info),
      .convergence = iterationStatus == IterationStatus::Active
          ? LinearSolverConvergenceStatus::Stopped
          : (IsConverged(iterationStatus) ? LinearSolverConvergenceStatus::Converged
                                          : LinearSolverConvergenceStatus::Diverged),
  };
}

} // namespace mochi::krylov

#endif
