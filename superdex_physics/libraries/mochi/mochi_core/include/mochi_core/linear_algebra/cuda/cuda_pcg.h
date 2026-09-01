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

#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/utils/verbosity_params.h>

#include <functional>
#include <type_traits>
#include <utility>

namespace mochi::krylov {

/// @brief Function for the actual implementation of PCG with on device
///
/// @return Pair of the number of iterations performed and the convergence status.
template <typename Scalar>
std::pair<int, LinearSolverConvergenceStatus> CudaPCG_impl(
    std::function<
        void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar>& Av)> const& A,
    mochi::CudaVectorView<Scalar const> bv,
    mochi::CudaVectorView<Scalar> xv,
    std::function<
        void(mochi::CudaVectorView<Scalar> const& x, mochi::CudaVectorView<Scalar> Px)> const& prec,
    int maxIter,
    std::function<krylov::IterationStatus(
        int iter,
        mochi::CudaVectorView<Scalar> r,
        mochi::CudaVectorView<Scalar> z,
        mochi::CudaVectorView<Scalar> p,
        mochi::CudaVectorView<Scalar> Ap)> const& stopFunction,
    bool abortIfNotSpd,
    VerbosityLevel verbosity,
    bool usePolakRibiere);

/**
 * @brief Solve a linear system using a preconditioned CG method.
 *
 * @param[in] A The matrix application operator.
 * @param[in] b The right-hand side vector of \f$ A x = b\f$.
 * @param[in,out] x Vector containing the initial guess at input and the solution at output.
 * @param[in] P The preconditioner application functor.
 * @param[in] maxIter Maximum number of iterations.
 * @param[in,out] statusCheck A functor called at each iteration to check the stop criteria.
 * @param[in] abortIfNotSpd Boolean to abort the solve if the matrix is detected not to be symmetric
 * positive definite. Default is false.
 * @param[in] verbosity Verbosity level for logging output.
 * @param[in] usePolakRibiere Boolean for using the Polak-Ribiere definition of beta
 * as opposed to the Fletcher-Reeves formula (default = true)
 *
 * @return
 * Linear solver status.
 * Contains the convergence status, number of iterations, and achieved absolute and relative
 * residuals.
 *
 * @note The norm used in the stop criteria is specified by the object 'statusCheck'.
 * @note Complex arithmetic is not supported.
 * @note The implementation assumes that the application of A and of P use the CuBLAS static handle,
 * the CuSparse static handle, or the stream attached to one of these handles.
 * @note When using @ref StatusResidualPreconditionerInduced, the inner product \f$\langle r, z
 * \rangle\f$ is computed both inside the CUDA graph (for the CG recurrence) and again inside the
 * stop function (for the convergence check). This redundancy is a consequence of the type-erased
 * stop function interface, which prevents the implementation from sharing the on-device \f$r^T z\f$
 * with the criterion. The overhead is one cuBLAS dot product per convergence check (every @c
 * kGraphBurst iterations) and is negligible relative to the SpMV-dominated cost. If profiling shows
 * otherwise, a future optimization could extend the @ref CudaPCG_impl interface to pass the
 * on-device \f$r^T z\f$ to the stop function, avoiding the redundant computation.
 *
 */
template <typename Op, typename Vector, typename VSol, typename Prec, typename StopCriterion>
  requires(mochi::IsCuda<Op> && mochi::IsCuda<Vector> && mochi::IsCuda<VSol>)
LinearSolverStatus CudaPCG(
    Op const& A,
    Vector const& b,
    VSol& x,
    Prec const& P,
    int maxIter,
    StopCriterion& statusCheck,
    bool abortIfNotSpd = false,
    VerbosityLevel verbosity = VerbosityLevel::Warning,
    bool usePolakRibiere = true) {
  using Scalar = std::remove_pointer_t<decltype(b.Data())>;
  using NonConstScalar = std::remove_const_t<Scalar>;
  //--- Represent the operator A with a function to "hide the type" of A.
  auto Afunc = [&](mochi::CudaVectorView<NonConstScalar> const& v,
                   mochi::CudaVectorView<NonConstScalar>& Av) { Apply(A, v, Av); };
  auto prec = [&](mochi::CudaVectorView<NonConstScalar> const& v,
                  mochi::CudaVectorView<NonConstScalar> Pv) { P(v, Pv); };
  //--- Represent the main stopping function to "hide the type" of `StopCriterion`.
  auto stopFunction = [&](int iter,
                          mochi::CudaVectorView<NonConstScalar> r,
                          mochi::CudaVectorView<NonConstScalar> z,
                          mochi::CudaVectorView<NonConstScalar> p,
                          mochi::CudaVectorView<NonConstScalar> Ap) -> krylov::IterationStatus {
    return statusCheck.CheckStatus(iter, r, z, p, Ap);
  };
  //--- Convert to CudaVectorView
  MOCHI_ASSERT(((b.Cols() == x.Cols()) && (x.Cols() == 1)), "Incompatible number of columns");
  mochi::CudaVectorView<Scalar> bv(b.Data(), b.Rows());
  mochi::CudaVectorView<NonConstScalar> xv(x.Data(), x.Rows());
  //--- Set the "scaling" for the stopping criterion
  {
    CudaVector<NonConstScalar> r(bv);
    CudaVector<NonConstScalar> z(bv);
    statusCheck.SetScaling(r, prec, z);
  }
  //--- Run the PCG algorithm
  auto info = CudaPCG_impl<NonConstScalar>(
      Afunc, bv, xv, prec, maxIter, stopFunction, abortIfNotSpd, verbosity, usePolakRibiere);
  //
  return LinearSolverStatus{
      .numIterDone = std::get<0>(info),
      .residualNorm = statusCheck.GetLatestResidualNorm(),
      .relativeResidualNorm = statusCheck.GetLatestRelativeResidualNorm(),
      .convergence = std::get<1>(info)};
}

} // namespace mochi::krylov

#endif
