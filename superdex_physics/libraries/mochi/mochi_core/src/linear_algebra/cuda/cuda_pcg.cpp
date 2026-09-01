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

#if MOCHI_USE_CUDA

#include "cuda_pcg_kernel.h"

#include <mochi_core/linear_algebra/base_enums.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/utils/verbosity_params.h>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <functional>
#include <type_traits>
#include <utility>

int constexpr kGraphBurst = 8;

namespace {

template <typename Scalar>
struct SolverData {
  std::function<
      void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar>& Av)> const& A;
  std::function<
      void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar> Pv)> const& prec;

  Scalar* MOCHI_RESTRICT d_x = nullptr;
  Scalar* MOCHI_RESTRICT d_r = nullptr;
  Scalar* MOCHI_RESTRICT d_z = nullptr;
  Scalar* MOCHI_RESTRICT d_p = nullptr;
  Scalar* MOCHI_RESTRICT d_Ap = nullptr;

  Scalar* d_rTzCurrent = nullptr;
  Scalar* d_rTzOld = nullptr;

  /// @brief Device pointer to flag when the coefficient alpha is inadmissible.
  bool* d_isAlphaLeq0 = nullptr;

  int n = 0;

  /// @brief Constructor
  ///
  /// \param A_
  /// \param xv_
  /// \param prec_
  /// \param r_ Pointer on the device to store the residual
  /// \param z_ Pointer on the device to store the preconditioned residual
  /// \param p_ Pointer on the device to store the direction
  /// \param Ap_ Pointer on the device to store the image of the direction
  /// \param one_ Pointer on the device to the constant 1
  SolverData(
      std::function<
          void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar>& Av)> const&
          A_,
      mochi::CudaVectorView<Scalar> xv_,
      std::function<
          void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar> Pv)> const&
          prec_,
      Scalar* MOCHI_RESTRICT r_,
      Scalar* MOCHI_RESTRICT z_,
      Scalar* MOCHI_RESTRICT p_,
      Scalar* MOCHI_RESTRICT Ap_,
      Scalar* one_)
      : A(A_),
        prec(prec_),
        d_x(xv_.data()),
        d_r(r_),
        d_z(z_),
        d_p(p_),
        d_Ap(Ap_),
        n(xv_.Rows()),
        _one(one_) {
    //
    cudaStream_t mainStream{};
    MOCHI_CUDA_CHECK(cudaStreamCreateWithFlags(&mainStream, cudaStreamNonBlocking));
    MOCHI_CUDA_CHECK(cudaStreamCreateWithFlags(&_sideStream, cudaStreamNonBlocking));
    //
    MOCHI_CUDA_CHECK(cudaMallocAsync((void**)(&_scalarPool), 6 * sizeof(Scalar), mainStream));
    //
    d_rTzCurrent = _scalarPool;
    d_rTzOld = _scalarPool + 1;
    _pTAp = _scalarPool + 2;
    _alpha = _scalarPool + 3;
    _negAlpha = _scalarPool + 4;
    _beta = _scalarPool + 5;
    //
    MOCHI_CUDA_CHECK(cudaMallocAsync((void**)&d_isAlphaLeq0, sizeof(bool), mainStream));
    MOCHI_CUDA_CHECK(cudaMemsetAsync(d_isAlphaLeq0, 0, sizeof(bool), mainStream));
    //
    for (auto& epool : _eventPool) {
      MOCHI_CUDA_CHECK(cudaEventCreate(&epool));
    }
    MOCHI_CUDA_CHECK(cudaStreamSynchronize(mainStream));
    MOCHI_CUDA_CHECK(cudaStreamDestroy(mainStream));
  }

  ~SolverData() {
    MOCHI_CUDA_CHECK(cudaStreamSynchronize(_sideStream));
    MOCHI_CUDA_CHECK(cudaFree(_scalarPool));
    MOCHI_CUDA_CHECK(cudaFree(d_isAlphaLeq0));
    //  event
    for (auto& e : _eventPool) {
      MOCHI_CUDA_CHECK(cudaEventDestroy(e));
    }
    // stream
    MOCHI_CUDA_CHECK(cudaStreamDestroy(_sideStream));
  }

 protected:
  Scalar* _scalarPool = nullptr;
  Scalar* _pTAp = nullptr;
  Scalar* _alpha = nullptr;
  Scalar* _negAlpha = nullptr;
  Scalar* _beta = nullptr;
  Scalar* _one = nullptr;

  cudaEvent_t _eventPool[2]{};
  cudaStream_t _sideStream{};

 public:
  template <int kBurst, bool kUsePolakRibiere = true>
  void MakeGraph(
      cublasHandle_t& blasHandle,
      cusparseHandle_t& sparseHandle,
      cudaGraphExec_t& pcgIterGraphExec) {
    cudaStream_t iterStream{};
    cudaStreamCreateWithFlags(&iterStream, cudaStreamNonBlocking);
    cudaGraph_t pcgIterGraph{};
    //
    MOCHI_CUDA_CHECK(cudaStreamBeginCapture(iterStream, cudaStreamCaptureModeThreadLocal));
    cusparseSetStream(sparseHandle, iterStream);
    cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSetStream(blasHandle, iterStream);
    mochi::CudaVectorView<Scalar> rv(d_r, n, 1);
    mochi::CudaVectorView<Scalar> zv(d_z, n, 1);
    mochi::CudaVectorView<Scalar> pv(d_p, n, 1);
    mochi::CudaVectorView<Scalar> Apv(d_Ap, n, 1);
    for (int i = 0; i < kBurst; ++i) {
      //  Apply A to p
      A(pv, Apv);
      //
      cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE);
      if constexpr (std::is_same_v<Scalar, double>) {
        MOCHI_CUBLAS_CHECK(cublasDdot(blasHandle, n, d_p, 1, d_Ap, 1, _pTAp));
      } else {
        MOCHI_CUBLAS_CHECK(cublasSdot(blasHandle, n, d_p, 1, d_Ap, 1, _pTAp));
      }
      // Get alpha factor
      mochi::details::Alpha(d_rTzCurrent, _pTAp, _alpha, _negAlpha, d_isAlphaLeq0, iterStream);
      cudaEventRecord(_eventPool[0], iterStream);
      // Update x
      cudaStreamWaitEvent(_sideStream, _eventPool[0]);
      cublasSetStream(blasHandle, _sideStream);
      if constexpr (std::is_same_v<Scalar, double>) {
        cublasDaxpy(blasHandle, n, _alpha, d_p, 1, d_x, 1);
      } else {
        static_assert(std::is_same_v<Scalar, float>);
        cublasSaxpy(blasHandle, n, _alpha, d_p, 1, d_x, 1);
      }
      cudaEventRecord(_eventPool[1], _sideStream);
      cublasSetStream(blasHandle, iterStream);
      // Update r
      if constexpr (std::is_same_v<Scalar, double>) {
        cublasDaxpy(blasHandle, n, _negAlpha, d_Ap, 1, d_r, 1);
      } else {
        static_assert(std::is_same_v<Scalar, float>);
        cublasSaxpy(blasHandle, n, _negAlpha, d_Ap, 1, d_r, 1);
      }
      if constexpr (kUsePolakRibiere) {
        // Use Polak-Ribiere formula for stability
        if constexpr (std::is_same_v<Scalar, double>) {
          MOCHI_CUBLAS_CHECK(cublasDdot(blasHandle, n, d_r, 1, d_z, 1, _beta));
        } else {
          MOCHI_CUBLAS_CHECK(cublasSdot(blasHandle, n, d_r, 1, d_z, 1, _beta));
        }
      } else {
        cudaMemsetAsync(_beta, 0, sizeof(Scalar), iterStream);
      }
      // Apply preconditioner on iterStream
      prec(rv, zv);
      //
      cudaMemcpyAsync(d_rTzOld, d_rTzCurrent, sizeof(Scalar), cudaMemcpyDeviceToDevice, iterStream);
      cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE);
      if constexpr (std::is_same_v<Scalar, double>) {
        MOCHI_CUBLAS_CHECK(cublasDdot(blasHandle, n, d_r, 1, d_z, 1, d_rTzCurrent));
      } else {
        MOCHI_CUBLAS_CHECK(cublasSdot(blasHandle, n, d_r, 1, d_z, 1, d_rTzCurrent));
      }
      mochi::details::Beta(d_rTzOld, d_rTzCurrent, _beta, iterStream);
      //
      cudaStreamWaitEvent(iterStream, _eventPool[1]);
      if constexpr (std::is_same_v<Scalar, double>) {
        cublasDscal(blasHandle, n, _beta, d_p, 1);
        cublasDaxpy(blasHandle, n, _one, d_z, 1, d_p, 1);
      } else {
        static_assert(std::is_same_v<Scalar, float>);
        cublasSscal(blasHandle, n, _beta, d_p, 1);
        cublasSaxpy(blasHandle, n, _one, d_z, 1, d_p, 1);
      }
    }
    //
    MOCHI_CUDA_CHECK(cudaStreamEndCapture(iterStream, &pcgIterGraph));
    MOCHI_CUDA_CHECK(cudaGraphInstantiate(&pcgIterGraphExec, pcgIterGraph));
    MOCHI_CUDA_CHECK(cudaStreamDestroy(iterStream));
    MOCHI_CUDA_CHECK(cudaGraphDestroy(pcgIterGraph));
  }
};

template struct SolverData<double>;
template struct SolverData<float>;

} // namespace

namespace mochi::krylov {

/***** PCG Code *****/

/// ASSUMPTIONS:
///   1. The cuSPARSE and cuBLAS libraries have been initialized.
///
/// @return Pair of the number of iterations performed and the convergence status.
///
template <typename Scalar>
std::pair<int, LinearSolverConvergenceStatus> CudaPCG_impl(
    std::function<
        void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar>& Ax)> const& A,
    mochi::CudaVectorView<Scalar const> bv,
    mochi::CudaVectorView<Scalar> xv,
    std::function<
        void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar> Pv)> const& prec,
    int maxIter,
    std::function<krylov::IterationStatus(
        int iter,
        mochi::CudaVectorView<Scalar> r,
        mochi::CudaVectorView<Scalar> z,
        mochi::CudaVectorView<Scalar> p,
        mochi::CudaVectorView<Scalar> Ap)> const& stopFunction,
    bool abortIfNotSpd,
    VerbosityLevel verbosity,
    bool usePolakRibiere) {
  // Check that Scalar is either float or double
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);

  auto blasHandle = reinterpret_cast<cublasHandle_t>(mochi::details::GetCuBLASHandle());
  auto sparseHandle = reinterpret_cast<cusparseHandle_t>(mochi::details::GetCuSparseHandle());

  auto const n = xv.Rows();
  CudaVector<Scalar> workSpace(5 * n + 2);
  Scalar* MOCHI_RESTRICT const r = workSpace.data();
  Scalar* MOCHI_RESTRICT const z = r + n;
  Scalar* MOCHI_RESTRICT const p = z + n;
  Scalar* MOCHI_RESTRICT const Ap = p + n;
  Scalar* MOCHI_RESTRICT const xInput = Ap + n;
  Scalar* const d_one = xInput + n;
  Scalar* const d_negOne = d_one + 1;
  {
    Scalar vTmp[2] = {Scalar(1), Scalar(-1)};
    cudaMemcpy(d_one, &vTmp[0], 2 * sizeof(Scalar), cudaMemcpyHostToDevice);
  }

  mochi::CudaVectorView<Scalar> rv(r, n, 1);
  mochi::CudaVectorView<Scalar> zv(z, n, 1);
  mochi::CudaVectorView<Scalar> pv(p, n, 1);
  mochi::CudaVectorView<Scalar> Apv(Ap, n, 1);

  cudaStream_t mainStream{};
  cudaStreamCreateWithFlags(&mainStream, cudaStreamNonBlocking);
  MOCHI_CUSPARSE_CHECK(cusparseSetStream(sparseHandle, mainStream));
  MOCHI_CUBLAS_CHECK(cublasSetStream(blasHandle, mainStream));
  // Compute r = b - A x_0
  // Apply A to x_0 (it should not change the stream)
  A(xv, rv);
  //
  MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE));
  if constexpr (std::is_same_v<Scalar, double>) {
    MOCHI_CUBLAS_CHECK(cublasDscal(blasHandle, n, d_negOne, r, 1));
    MOCHI_CUBLAS_CHECK(cublasDaxpy(blasHandle, n, d_one, bv.data(), 1, r, 1));
  } else {
    MOCHI_CUBLAS_CHECK(cublasSscal(blasHandle, n, d_negOne, r, 1));
    MOCHI_CUBLAS_CHECK(cublasSaxpy(blasHandle, n, d_one, bv.data(), 1, r, 1));
  }
  // Apply preconditioner (it should not change the stream)
  prec(rv, zv);
  // Check for convergence
  auto myStatus = stopFunction(0, rv, zv, pv, Apv);
  if (myStatus != IterationStatus::Active) {
    // Free allocated memory
    //--- Reset cublas and cusparse
    MOCHI_CUSPARSE_CHECK(cusparseSetStream(sparseHandle, nullptr));
    MOCHI_CUSPARSE_CHECK(cusparseSetPointerMode(sparseHandle, CUSPARSE_POINTER_MODE_HOST));
    MOCHI_CUBLAS_CHECK(cublasSetStream(blasHandle, nullptr));
    MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_HOST));
    MOCHI_CUDA_CHECK(cudaStreamSynchronize(mainStream));
    MOCHI_CUDA_CHECK(cudaStreamDestroy(mainStream));
    return {
        0,
        IsConverged(myStatus) ? LinearSolverConvergenceStatus::Converged
                              : LinearSolverConvergenceStatus::Diverged};
  }
  SolverData<Scalar> cgInfo{A, xv, prec, r, z, p, Ap, d_one};
  Scalar* const d_rTzCurrent = cgInfo.d_rTzCurrent;
  // Create the graph
  cudaGraphExec_t pcgIterGraphExec{};
  if (usePolakRibiere) {
    cgInfo.template MakeGraph<kGraphBurst, true>(blasHandle, sparseHandle, pcgIterGraphExec);
  } else {
    cgInfo.template MakeGraph<kGraphBurst, false>(blasHandle, sparseHandle, pcgIterGraphExec);
  }

  cudaStream_t sideStream{};
  MOCHI_CUDA_CHECK(cudaStreamCreateWithFlags(&sideStream, cudaStreamNonBlocking));

  cudaEvent_t eventPool[2];
  for (auto& epool : eventPool) {
    MOCHI_CUDA_CHECK(cudaEventCreate(&epool));
  }
  auto& eventNewP = eventPool[0];
  auto& eventNewRtZ = eventPool[1];

  // Make a copy of input vector
  cudaMemcpyAsync(xInput, xv.data(), n * sizeof(Scalar), cudaMemcpyDeviceToDevice, mainStream);

  // Get the first direction p
  cudaMemcpyAsync(p, z, n * sizeof(Scalar), cudaMemcpyDeviceToDevice, mainStream);
  cudaEventRecord(eventNewP, mainStream);

  // Compute the latest value of r^T z
  // This computation does not depend on p.
  // So we start it on another stream.
  cublasSetStream(blasHandle, sideStream);
  MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE));
  if constexpr (std::is_same_v<Scalar, double>) {
    MOCHI_CUBLAS_CHECK(cublasDdot(blasHandle, n, r, 1, z, 1, d_rTzCurrent));
  } else {
    MOCHI_CUBLAS_CHECK(cublasSdot(blasHandle, n, r, 1, z, 1, d_rTzCurrent));
  }
  cudaEventRecord(eventNewRtZ, sideStream);

  // Wait until the value of p is up-to-date
  cudaStreamWaitEvent(sideStream, eventNewP);

  // Wait until the value of r^T z is up-to-date
  cudaStreamWaitEvent(mainStream, eventNewRtZ);
  cublasSetStream(blasHandle, mainStream);

  bool alphaBad = false;
  std::pair<int, LinearSolverConvergenceStatus> info{
      maxIter, LinearSolverConvergenceStatus::Stopped};
  int iter = 0;
  for (; iter < maxIter;) {
    MOCHI_CUDA_CHECK(cudaGraphLaunch(pcgIterGraphExec, mainStream));
    MOCHI_CUDA_CHECK(cudaStreamSynchronize(mainStream));
    iter += kGraphBurst;
    //
    if (abortIfNotSpd) {
      cudaMemcpy(&alphaBad, cgInfo.d_isAlphaLeq0, sizeof(bool), cudaMemcpyDeviceToHost);
      if (alphaBad) {
        if (verbosity >= VerbosityLevel::Warning) {
          MOCHI_LOG_WARNING("CudaPCG encountered a non-positive coefficient alpha during a burst");
        }
        break;
      }
    }

    // Check convergence. Note: for StatusResidualPreconditionerInduced, this recomputes <r, z>
    // which was already computed on-device (d_rTzCurrent) inside the graph. This is a consequence
    // of the type-erased stopFunction interface. See CudaPCG documentation for details.
    myStatus = stopFunction(iter, rv, zv, pv, Apv);
    if (myStatus != IterationStatus::Active) {
      std::get<0>(info) = iter;
      std::get<1>(info) = IsConverged(myStatus) ? LinearSolverConvergenceStatus::Converged
                                                : LinearSolverConvergenceStatus::Diverged;
      break;
    }
  }
  if (!alphaBad && myStatus == IterationStatus::Active) {
    std::get<0>(info) = iter;
  }
  // Remove the graph executor
  MOCHI_CUDA_CHECK(cudaGraphExecDestroy(pcgIterGraphExec));
  //
  if ((alphaBad) || (myStatus == IterationStatus::DivergedRes)) {
    if (verbosity >= VerbosityLevel::Warning) {
      MOCHI_LOG_WARNING("CudaPCG with burst encountered a problem. Will try without any burst");
    }
    cudaGraphExec_t pcgBackupExec{};
    if (usePolakRibiere) {
      cgInfo.template MakeGraph<1, true>(blasHandle, sparseHandle, pcgBackupExec);
    } else {
      cgInfo.template MakeGraph<1, false>(blasHandle, sparseHandle, pcgBackupExec);
    }
    cublasSetStream(blasHandle, mainStream);
    cusparseSetStream(sparseHandle, mainStream);
    // Reset state for the non-burst fallback
    iter = 1;
    myStatus = IterationStatus::Active;
    info = {maxIter, LinearSolverConvergenceStatus::Stopped};
    cudaMemsetAsync(cgInfo.d_isAlphaLeq0, 0, sizeof(bool), mainStream);
    // We will try a restart with the initial guess
    cudaMemcpyAsync(xv.data(), xInput, n * sizeof(Scalar), cudaMemcpyDeviceToDevice, mainStream);
    cudaStreamSynchronize(mainStream);
    // Restart PCG without burst
    A(xv, rv);
    // Synchronization needed before finalizing r
    MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE));
    if constexpr (std::is_same_v<Scalar, double>) {
      MOCHI_CUBLAS_CHECK(cublasDscal(blasHandle, n, d_negOne, r, 1));
      MOCHI_CUBLAS_CHECK(cublasDaxpy(blasHandle, n, d_one, bv.data(), 1, r, 1));
    } else {
      static_assert(std::is_same_v<Scalar, float>);
      MOCHI_CUBLAS_CHECK(cublasSscal(blasHandle, n, d_negOne, r, 1));
      MOCHI_CUBLAS_CHECK(cublasSaxpy(blasHandle, n, d_one, bv.data(), 1, r, 1));
    }
    prec(rv, zv);
    mochi::details::CudaDeviceCopy(p, z, n);
    MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE));
    if constexpr (std::is_same_v<Scalar, double>) {
      MOCHI_CUBLAS_CHECK(cublasDdot(blasHandle, n, r, 1, z, 1, d_rTzCurrent));
    } else {
      MOCHI_CUBLAS_CHECK(cublasSdot(blasHandle, n, r, 1, z, 1, d_rTzCurrent));
    }
    for (; iter <= maxIter; ++iter) {
      MOCHI_CUDA_CHECK(cudaGraphLaunch(pcgBackupExec, mainStream));
      MOCHI_CUDA_CHECK(cudaStreamSynchronize(mainStream));
      //
      if (abortIfNotSpd) {
        cudaMemcpy(&alphaBad, cgInfo.d_isAlphaLeq0, sizeof(bool), cudaMemcpyDeviceToHost);
        if (alphaBad) {
          std::get<0>(info) = iter;
          std::get<1>(info) = LinearSolverConvergenceStatus::Diverged;
          break;
        }
      }
      // Check convergence
      cublasSetStream(blasHandle, mainStream);
      cusparseSetStream(sparseHandle, mainStream);
      myStatus = stopFunction(iter, rv, zv, pv, Apv);
      if (myStatus != IterationStatus::Active) {
        std::get<0>(info) = iter;
        std::get<1>(info) = IsConverged(myStatus) ? LinearSolverConvergenceStatus::Converged
                                                  : LinearSolverConvergenceStatus::Diverged;
        break;
      }
    }
    cudaGraphExecDestroy(pcgBackupExec);
  }
  //--- Reset cublas and cusparse
  MOCHI_CUSPARSE_CHECK(cusparseSetStream(sparseHandle, nullptr));
  MOCHI_CUSPARSE_CHECK(cusparseSetPointerMode(sparseHandle, CUSPARSE_POINTER_MODE_HOST));
  MOCHI_CUBLAS_CHECK(cublasSetStream(blasHandle, nullptr));
  MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_HOST));

  //  event
  for (auto& e : eventPool) {
    MOCHI_CUDA_CHECK(cudaEventDestroy(e));
  }

  // stream
  MOCHI_CUDA_CHECK(cudaStreamDestroy(sideStream));

  MOCHI_CUDA_CHECK(cudaStreamSynchronize(mainStream));
  MOCHI_CUDA_CHECK(cudaStreamDestroy(mainStream));

  return info;
}

template std::pair<int, LinearSolverConvergenceStatus> CudaPCG_impl<double>(
    std::function<
        void(mochi::CudaVectorView<double> const& v, mochi::CudaVectorView<double>& Ax)> const& A,
    mochi::CudaVectorView<double const> bv,
    mochi::CudaVectorView<double> xv,
    std::function<
        void(mochi::CudaVectorView<double> const& v, mochi::CudaVectorView<double> Pv)> const& prec,
    int maxIter,
    std::function<krylov::IterationStatus(
        int iter,
        mochi::CudaVectorView<double> r,
        mochi::CudaVectorView<double> z,
        mochi::CudaVectorView<double> p,
        mochi::CudaVectorView<double> Ap)> const& stopFunction,
    bool abortIfNotSpd,
    VerbosityLevel verbosity,
    bool usePolakRibiere);

template std::pair<int, LinearSolverConvergenceStatus> CudaPCG_impl<float>(
    std::function<
        void(mochi::CudaVectorView<float> const& v, mochi::CudaVectorView<float>& Ax)> const& A,
    mochi::CudaVectorView<float const> bv,
    mochi::CudaVectorView<float> xv,
    std::function<
        void(mochi::CudaVectorView<float> const& v, mochi::CudaVectorView<float> Pv)> const& prec,
    int maxIter,
    std::function<krylov::IterationStatus(
        int iter,
        mochi::CudaVectorView<float> r,
        mochi::CudaVectorView<float> z,
        mochi::CudaVectorView<float> p,
        mochi::CudaVectorView<float> Ap)> const& stopFunction,
    bool abortIfNotSpd,
    VerbosityLevel verbosity,
    bool usePolakRibiere);

} // namespace mochi::krylov

#endif
