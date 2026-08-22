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

#include <mochi_core/linear_algebra/base_enums.h>
#include <mochi_core/linear_algebra/cuda/cuda_gmres_kernels.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_sparse_utils.h>
#include <mochi_core/linear_algebra/krylov/iteration_status.h>
#include <mochi_core/utils/defer.h>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <cmath>
#include <functional>
#include <tuple>
#include <type_traits>

namespace mochi::krylov {

/***** GMRes Code *****/

/// ASSUMPTIONS:
/// 1. The cuSPARSE and cuBLAS libraries have been initialized.
/// 2. The right-hand side is non-zero, i.e. bNorm > 0.
template <typename Scalar>
std::tuple<int, double, double, IterationStatus> CudaGMRes_impl(
    std::function<
        void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar>& Ax)> const& A,
    mochi::CudaVectorView<Scalar const> bv,
    Scalar bNorm,
    mochi::CudaVectorView<Scalar> xv,
    std::function<
        void(mochi::CudaVectorView<Scalar> const& v, mochi::CudaVectorView<Scalar> Pv)> const& prec,
    int maxIter,
    double aTol,
    double rTol,
    double dTol,
    int restartSize) {
  MOCHI_ASSERT(bNorm > 0, "CudaGMRes_impl requires non-zero initial RHS.");

  auto blasHandle = reinterpret_cast<cublasHandle_t>(mochi::details::GetCuBLASHandle());
  auto sparseHandle = reinterpret_cast<cusparseHandle_t>(mochi::details::GetCuSparseHandle());

  std::tuple<int, double, double, IterationStatus> info{};

  auto const n = xv.Rows();
  int const workSize = 3 * (restartSize + 1) + 8;
  CudaVector<Scalar> workSpace(workSize);
  Scalar* const d_b = workSpace.data();
  Scalar* const d_cs = d_b + (restartSize + 1);
  Scalar* const d_sn = d_cs + (restartSize + 1);
  Scalar* const d_normQc = d_sn + (restartSize + 1);
  Scalar* const d_invN = d_normQc + 1;
  Scalar* const d_one = d_invN + 1;
  Scalar* const d_zero = d_one + 1;
  Scalar* const d_negOne = d_zero + 1;
  Scalar* const d_aTol = d_negOne + 1;
  Scalar* const d_rTol = d_aTol + 1;
  Scalar* const d_dTol = d_rTol + 1;
  //
  // Start the initialization
  //
  cudaStream_t mainStream{};
  cudaStreamCreateWithFlags(&mainStream, cudaStreamNonBlocking);
  MOCHI_DEFER(cudaStreamDestroy(mainStream));
  //
  int nPadded = RoundUp(static_cast<int>(n * sizeof(Scalar)), 256) / sizeof(Scalar);
  CudaMatrix<Scalar> d_Q(nPadded, restartSize + 2);
  int const ldq = d_Q.LeadDim();
  Scalar* const Q = d_Q.data();
  //
  CudaVectorView<Scalar> wv(d_Q.Col(restartSize + 1).data(), n);
  //
  std::size_t bufferSize = mochi::details::gmres::GetBufferSize(n, Q, d_normQc, mainStream);
  CudaVector<Scalar> d_buffer(bufferSize + n);
  CudaVectorView<Scalar> zz(d_buffer.data(), n);
  //
  auto const ldh = restartSize + 1;
  CudaMatrix<Scalar> d_H(restartSize + 1, restartSize + 2);
  Scalar* const H = d_H.Data();
  CudaVectorView<Scalar> H1 = d_H.Col(restartSize);
  CudaVectorView<Scalar> H2 = d_H.Col(restartSize + 1);
  //
  MOCHI_CUBLAS_CHECK(cublasSetStream(blasHandle, mainStream));
  //
  int* d_int{};
  MOCHI_CUDA_CHECK(cudaMalloc((void**)&d_int, 2 * sizeof(int)));
  MOCHI_DEFER(cudaFree(d_int));
  MOCHI_CUDA_CHECK(cudaMemsetAsync(d_int, 0, 2 * sizeof(int), mainStream));
  int* const d_iter = d_int;
  int* const d_status = d_iter + 1;
  //
  int h_int[2] = {0, static_cast<int>(IterationStatus::Active)};
  int& h_iter = h_int[0];
  int& h_status = h_int[1];
  //
  Scalar const h_val[6] = {
      Scalar(1), 0, Scalar(-1), Scalar(aTol), Scalar(bNorm * rTol), Scalar(bNorm * dTol)};
  MOCHI_CUDA_CHECK(
      cudaMemcpyAsync(d_one, &h_val[0], 6 * sizeof(Scalar), cudaMemcpyHostToDevice, mainStream));
  //
  Scalar h_norm = 0;
  cudaGraphExec_t graphExec{};
  //
  MOCHI_CUBLAS_CHECK(cublasSetStream(blasHandle, mainStream));
  MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE));
  MOCHI_CUSPARSE_CHECK(cusparseSetStream(sparseHandle, mainStream));
  //
  int totalIter = 0;
  for (; totalIter < maxIter;) {
    MOCHI_CUDA_CHECK(cudaMemsetAsync(d_b, 0, (restartSize + 1) * sizeof(Scalar), mainStream));
    //--- Compute r = d_b - A x_0
    CudaVectorView<Scalar> rv(Q, n);
    // Apply A to x_0
    // A should not change the stream of BlasHandle and of SparseHandle
    A(xv, rv);
    // Finalize the residual
    MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE));
    if constexpr (std::is_same_v<Scalar, double>) {
      MOCHI_CUBLAS_CHECK(cublasDscal(blasHandle, n, d_negOne, rv.Data(), 1));
      MOCHI_CUBLAS_CHECK(cublasDaxpy(blasHandle, n, d_one, bv.data(), 1, rv.Data(), 1));
      MOCHI_CUBLAS_CHECK(cublasDnrm2(blasHandle, n, rv.data(), 1, d_b));
    } else {
      static_assert(std::is_same_v<Scalar, float>);
      MOCHI_CUBLAS_CHECK(cublasSscal(blasHandle, n, d_negOne, rv.Data(), 1));
      MOCHI_CUBLAS_CHECK(cublasSaxpy(blasHandle, n, d_one, bv.data(), 1, rv.Data(), 1));
      MOCHI_CUBLAS_CHECK(cublasSnrm2(blasHandle, n, rv.data(), 1, d_b));
    }
    // Check that the starting direction is not zero
    MOCHI_CUDA_CHECK(
        cudaMemcpyAsync(&h_norm, d_b, sizeof(Scalar), cudaMemcpyDeviceToHost, mainStream));
    MOCHI_CUDA_CHECK(cudaStreamSynchronize(mainStream));
    if (h_norm == Scalar{0}) {
      h_status = static_cast<int>(IterationStatus::ConvergedAtol);
      break;
    }
    //
    // Q(:, 0) = r / d_b(0)
    //
    mochi::details::gmres::GetInverse(d_b, d_invN, mainStream);
    if constexpr (std::is_same_v<Scalar, double>) {
      MOCHI_CUBLAS_CHECK(cublasDscal(blasHandle, n, d_invN, Q, 1));
    } else {
      static_assert(std::is_same_v<Scalar, float>);
      MOCHI_CUBLAS_CHECK(cublasSscal(blasHandle, n, d_invN, Q, 1));
    }
    //
    h_iter = 1;
    MOCHI_CUDA_CHECK(
        cudaMemcpyAsync(d_iter, &h_iter, sizeof(int), cudaMemcpyHostToDevice, mainStream));
    //
    if (totalIter == 0) {
      cudaGraph_t graph{};
      cudaGraphCreate(&graph, 0);
      cudaGraphNode_t conditionalNode{};
      cudaGraphConditionalHandle handle{};
      cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);
      cudaGraphNodeParams cParams = {.type = cudaGraphNodeTypeConditional};
      cParams.conditional.handle = handle;
      cParams.conditional.type = cudaGraphCondTypeWhile;
      cParams.conditional.size = 1;
      MOCHI_CUDA_CHECK(cudaGraphAddNode(&conditionalNode, graph, nullptr, 0, &cParams));
      cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];
      //
      cudaStream_t stream0{};
      MOCHI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));
      MOCHI_DEFER(cudaStreamDestroy(stream0));
      //
      cudaStreamBeginCaptureToGraph(
          stream0, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
      //
      cublasSetStream(blasHandle, stream0);
      cusparseSetStream(sparseHandle, stream0);
      // prec should not change the stream of BlasHandle and of SparseHandle
      prec(wv, zz);
      // A should not change the stream of BlasHandle and of SparseHandle
      A(zz, wv);
      //
      //--- Orthogonalize
      //
      mochi::details::gmres::GemvT(
          n, d_iter, restartSize + 1, Q, ldq, wv.Data(), H1.data(), stream0);
      mochi::details::gmres::Gemv(n, d_iter, Q, ldq, H1.data(), wv.Data(), stream0);
      //
      //--- Second Gram-Schmidt step
      //
      mochi::details::gmres::GemvT(
          n, d_iter, restartSize + 1, Q, ldq, wv.Data(), H2.data(), stream0);
      mochi::details::gmres::Gemv(n, d_iter, Q, ldq, H2.data(), wv.Data(), stream0);
      //
      mochi::details::gmres::UpdateHessenberg1(
          d_iter, d_cs, d_sn, H1.data(), H2.data(), H, ldh, stream0);
      //
      //--- Normalize and store new direction
      //
      mochi::details::gmres::NormL2(
          n, wv.Data(), d_normQc, d_buffer.data(), d_buffer.StorageSize(), stream0);
      //
      mochi::details::gmres::ScaleStore(n, d_iter, Q, ldq, wv.Data(), d_normQc, stream0);
      //
      // Do the Hessenberg rotation
      //
      mochi::details::gmres::UpdateHessenberg2(d_iter, d_cs, d_sn, d_normQc, H, ldh, d_b, stream0);
      //
      mochi::details::gmres::DoWhile(
          d_iter, restartSize, d_b, d_aTol, d_rTol, d_dTol, d_status, handle, stream0);
      //
      MOCHI_CUDA_CHECK(cudaStreamEndCapture(stream0, nullptr));
      MOCHI_CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
      //
      MOCHI_CUBLAS_CHECK(cublasSetStream(blasHandle, mainStream));
      MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE));
      //
      MOCHI_CUDA_CHECK(cudaGraphDestroy(graph));
    } // if (totalIter == 0)
    //
    mochi::details::CudaDeviceCopy(wv.data(), Q, n);
    MOCHI_CUDA_CHECK(cudaGraphLaunch(graphExec, mainStream));
    MOCHI_CUDA_CHECK(cudaStreamSynchronize(mainStream));
    // Copy both the number of iterations and the (int-valued) status
    MOCHI_CUDA_CHECK(
        cudaMemcpyAsync(h_int, d_int, 2 * sizeof(int), cudaMemcpyDeviceToHost, mainStream));
    //
    MOCHI_CUBLAS_CHECK(cublasSetStream(blasHandle, mainStream));
    MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE));
    MOCHI_CUDA_CHECK(cudaStreamSynchronize(mainStream));
    if constexpr (std::is_same_v<Scalar, double>) {
      MOCHI_CUBLAS_CHECK(cublasDtrsv(
          blasHandle,
          CUBLAS_FILL_MODE_UPPER,
          CUBLAS_OP_N,
          CUBLAS_DIAG_NON_UNIT,
          h_iter,
          H,
          ldh,
          d_b,
          1));
      MOCHI_CUBLAS_CHECK(cublasDgemv(
          blasHandle, CUBLAS_OP_N, n, h_iter, d_one, Q, ldq, d_b, 1, d_zero, wv.data(), 1));
    } else {
      MOCHI_CUBLAS_CHECK(cublasStrsv(
          blasHandle,
          CUBLAS_FILL_MODE_UPPER,
          CUBLAS_OP_N,
          CUBLAS_DIAG_NON_UNIT,
          h_iter,
          H,
          ldh,
          d_b,
          1));
      MOCHI_CUBLAS_CHECK(cublasSgemv(
          blasHandle, CUBLAS_OP_N, n, h_iter, d_one, Q, ldq, d_b, 1, d_zero, wv.data(), 1));
    }
    // prec should not change the stream of BlasHandle and of SparseHandle
    MOCHI_CUSPARSE_CHECK(cusparseSetStream(sparseHandle, mainStream));
    prec(wv, zz);
    //
    MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_DEVICE));
    if constexpr (std::is_same_v<Scalar, double>) {
      MOCHI_CUBLAS_CHECK(cublasDaxpy(blasHandle, n, d_one, zz.data(), 1, xv.data(), 1));
    } else if constexpr (std::is_same_v<Scalar, float>) {
      MOCHI_CUBLAS_CHECK(cublasSaxpy(blasHandle, n, d_one, zz.data(), 1, xv.data(), 1));
    }
    MOCHI_CUDA_CHECK(cudaStreamSynchronize(mainStream));
    //
    totalIter += h_iter;
    if ((h_status != static_cast<int>(IterationStatus::Active)) || (totalIter >= maxIter)) {
      MOCHI_CUDA_CHECK(cudaMemcpy(&h_norm, d_b + h_iter, sizeof(Scalar), cudaMemcpyDeviceToHost));
      break;
    }
  }

  std::get<0>(info) = totalIter;
  //--- Entries in the vector `d_b` are signed.
  std::get<1>(info) = std::abs(h_norm);
  std::get<2>(info) = std::abs(h_norm) / bNorm;
  std::get<3>(info) = static_cast<IterationStatus>(h_status);

  if (graphExec) {
    MOCHI_CUDA_CHECK(cudaGraphExecDestroy(graphExec));
  }

  //--- Reset cublas and cusparse
  MOCHI_CUSPARSE_CHECK(cusparseSetStream(sparseHandle, nullptr));
  MOCHI_CUSPARSE_CHECK(cusparseSetPointerMode(sparseHandle, CUSPARSE_POINTER_MODE_HOST));
  MOCHI_CUBLAS_CHECK(cublasSetStream(blasHandle, nullptr));
  MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_HOST));

  MOCHI_CUDA_CHECK_LAST();

  return info;
}

template std::tuple<int, double, double, IterationStatus> CudaGMRes_impl<double>(
    std::function<
        void(mochi::CudaVectorView<double> const& v, mochi::CudaVectorView<double>& Ax)> const& A,
    mochi::CudaVectorView<double const> bv,
    double bNorm,
    mochi::CudaVectorView<double> xv,
    std::function<
        void(mochi::CudaVectorView<double> const& v, mochi::CudaVectorView<double> Pv)> const& prec,
    int maxIter,
    double aTol,
    double rTol,
    double dTol,
    int restartSize);

template std::tuple<int, double, double, IterationStatus> CudaGMRes_impl<float>(
    std::function<
        void(mochi::CudaVectorView<float> const& v, mochi::CudaVectorView<float>& Ax)> const& A,
    mochi::CudaVectorView<float const> bv,
    float bNorm,
    mochi::CudaVectorView<float> xv,
    std::function<
        void(mochi::CudaVectorView<float> const& v, mochi::CudaVectorView<float> Pv)> const& prec,
    int maxIter,
    double aTol,
    double rTol,
    double dTol,
    int restartSize);

} // namespace mochi::krylov

#endif
