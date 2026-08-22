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
#include <mochi_core/mochi_config.h>
MOCHI_WARNING_IGNORE_MSVC(4505);
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#if MOCHI_USE_CUDSS
#include <cudss.h>
#endif
#include <cusolverSp.h>
#include <cusparse.h>

#include <mochi_core/linear_algebra/cuda/cuda_api.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/matrix.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace mochi::details {

void* CudaMalloc(std::size_t lenInBytes) {
  void* v = nullptr;
  MOCHI_CUDA_CHECK(cudaMalloc(&v, lenInBytes));
  return v;
}

void CudaFree(void* p) {
  MOCHI_CUDA_CHECK(cudaFree(p));
}

void CudaMemSetZero(void* ptr, std::size_t lenInBytes) {
  MOCHI_CUDA_CHECK(cudaMemset(ptr, 0, lenInBytes));
}

void CudaMemCopy(void* dest, void const* src, std::size_t lenInBytes) {
  MOCHI_CUDA_CHECK(cudaMemcpy(dest, src, lenInBytes, cudaMemcpyKind::cudaMemcpyDefault));
}

void CudaDeviceCopy(float* dest, float const* src, std::size_t len) {
  auto blasHandle = reinterpret_cast<cublasHandle_t>(GetCuBLASHandle());
  MOCHI_CUBLAS_CHECK(cublasScopy(blasHandle, int(len), src, 1, dest, 1));
}

void CudaDeviceCopy(double* dest, double const* src, std::size_t len) {
  auto blasHandle = reinterpret_cast<cublasHandle_t>(GetCuBLASHandle());
  MOCHI_CUBLAS_CHECK(cublasDcopy(blasHandle, int(len), src, 1, dest, 1));
}

void CudaMemCopy2D(
    void* dest,
    std::size_t destLdInBytes,
    void const* src,
    std::size_t srcLdInBytes,
    std::size_t widthInBytes,
    std::size_t height) {
  MOCHI_CUDA_CHECK(cudaMemcpy2D(
      dest,
      destLdInBytes,
      src,
      srcLdInBytes,
      widthInBytes,
      height,
      cudaMemcpyKind::cudaMemcpyDefault));
}

CudaBLASHandle GetCuBLASHandle() {
  static thread_local CudaBLASHandle thisHandle = [] {
    // Create the cuBLAS handle
    cublasHandle_t blasHandle = nullptr;
    MOCHI_CUBLAS_CHECK(cublasCreate(&blasHandle));
    return reinterpret_cast<CudaBLASHandle>(blasHandle);
  }();
  return thisHandle;
}

CudaSparseHandle GetCuSparseHandle() {
  static thread_local CudaSparseHandle thisHandle = [] {
    // Create the cuSparse handle
    cusparseHandle_t sparseHandle = nullptr;
    MOCHI_CUSPARSE_CHECK(cusparseCreate(&sparseHandle));
    return reinterpret_cast<CudaSparseHandle>(sparseHandle);
  }();
  return thisHandle;
}

CudaSolverSpHandle GetCuSolverSpHandle() {
  static thread_local CudaSolverSpHandle thisHandle = [] {
    // Create the cusolverSp handle
    cusolverSpHandle_t solverSpHandle = nullptr;
    MOCHI_CUSOLVER_CHECK(cusolverSpCreate(&solverSpHandle));
    return reinterpret_cast<CudaSolverSpHandle>(solverSpHandle);
  }();
  return thisHandle;
}

#if MOCHI_USE_CUDSS
/// @brief Returns the handle for calling cudss-specific functions
CudaDSSHandle GetCuDSSHandle() {
  static thread_local CudaDSSHandle thisHandle = [] {
    // Create the cuDSS handle
    cudssHandle_t dssHandle = nullptr;
    MOCHI_CUDSS_CHECK(cudssCreate(&dssHandle));
    return reinterpret_cast<CudaDSSHandle>(dssHandle);
  }();
  return thisHandle;
}
#endif

template <typename Scalar>
Scalar CudaDot(int n, Scalar const* x, Scalar const* y) {
  auto blasHandle = reinterpret_cast<cublasHandle_t>(GetCuBLASHandle());
  static_assert(std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>);
  Scalar result = 0;
  MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_HOST));
  if constexpr (std::is_same_v<Scalar, double>) {
    MOCHI_CUBLAS_CHECK(cublasDdot(blasHandle, n, x, 1, y, 1, &result));
  } else {
    MOCHI_CUBLAS_CHECK(cublasSdot(blasHandle, n, x, 1, y, 1, &result));
  }
  return result;
}

template double CudaDot<double>(int n, double const* x, double const* y);

template float CudaDot<float>(int n, float const* x, float const* y);

template <typename Scalar>
static void ExecuteCudaAssign_impl(
    MatDataDest<Scalar>&& dest,
    int nPos,
    ScaledMatData<Scalar const> const* posSrc,
    int nNeg,
    ScaledMatData<Scalar const> const* negSrc,
    int rows,
    int cols) {
  PackedScaleData<Scalar const> terms;
  int numTerms = 0;
  for (int i = 0; i < nPos && numTerms < kMaxNumTermsInExpression; ++i, ++numTerms) {
    terms.v[numTerms] = posSrc[i];
  }
  for (int i = 0; i < nNeg && numTerms < kMaxNumTermsInExpression; ++i, ++numTerms) {
    terms.v[numTerms] = negSrc[i];
    terms.v[numTerms].scale = -terms.v[numTerms].scale;
  }
  if (dest.rowStride != 1) {
    using namespace std;
    swap(dest.rowStride, dest.colStride);
    swap(rows, cols);
    for (int i = 0; i < numTerms; ++i) {
      swap(terms.v[i].rowStride, terms.v[i].colStride);
    }
  }

  int numRowPacked = 0;
  int numFull = dest.colStride == rows;
  for (int i = 0; i < numTerms; ++i) {
    numRowPacked += (terms.v[i].rowStride == 1);
    numFull += terms.v[i].colStride == rows;
  }

  if (numRowPacked == numTerms && (numFull == numTerms + 1 || cols == 1)) {
    int count = rows * cols;
    DoCudaScaledAssign(count, dest.v, numTerms, terms);
  } else {
    DoCudaScaled2DAssign(rows, cols, dest, numTerms, terms);
  }
}

void ExecuteCudaAssign(
    MatDataDest<double>&& dest,
    int nPos,
    ScaledMatData<double const> const* posSrc,
    int nNeg,
    ScaledMatData<double const> const* negSrc,
    int rows,
    int cols) {
  ExecuteCudaAssign_impl<double>(std::move(dest), nPos, posSrc, nNeg, negSrc, rows, cols);
}

void ExecuteCudaAssign(
    MatDataDest<float>&& dest,
    int nPos,
    ScaledMatData<float const> const* posSrc,
    int nNeg,
    ScaledMatData<float const> const* negSrc,
    int rows,
    int cols) {
  ExecuteCudaAssign_impl<float>(std::move(dest), nPos, posSrc, nNeg, negSrc, rows, cols);
}

static bool GetLeadDim(int rowStride, int colStride, int m, int n, int& ld) {
  bool colMajor = false;
  // NOLINTBEGIN(bugprone-branch-clone)
  if (rowStride > 1) {
    colMajor = false;
  } else if (colStride > 1) {
    colMajor = true;
  } else if (m == 1) {
    // Here rowStride = colStride = 1
    colMajor = true;
  }
  // NOLINTEND(bugprone-branch-clone)
  ld = (colMajor) ? std::max(colStride, m) : std::max(rowStride, n);
  return colMajor;
}

template <typename Scalar>
void ExecuteCudaProduct_impl(
    MatDataDest<Scalar>& C,
    ScaledMatData<Scalar const>& A,
    ScaledMatData<Scalar const>& B,
    int m,
    int n,
    int k,
    Scalar alpha,
    Scalar beta) {
  int ldA = 1, ldB = 1, ldC = 1;
  auto isAColMajor = GetLeadDim(A.rowStride, A.colStride, m, k, ldA);
  auto isBColMajor = GetLeadDim(B.rowStride, B.colStride, k, n, ldB);
  auto isCColMajor = GetLeadDim(C.rowStride, C.colStride, m, n, ldC);

  auto* Adata = A.v;
  auto* Bdata = B.v;
  if (!isCColMajor) {
    //--- Compute C^T = beta C^T + alpha B^T A^T
    isAColMajor = !isAColMajor;
    isBColMajor = !isBColMajor;
    std::swap(m, n);
    std::swap(ldA, ldB);
    std::swap(Adata, Bdata);
    std::swap(isAColMajor, isBColMajor);
  }
  cublasOperation_t transA =
      isAColMajor ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T;
  cublasOperation_t transB =
      isBColMajor ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T;
  auto blasHandle = reinterpret_cast<cublasHandle_t>(GetCuBLASHandle());
  MOCHI_CUBLAS_CHECK(cublasSetPointerMode(blasHandle, CUBLAS_POINTER_MODE_HOST));
  if constexpr (std::is_same_v<Scalar, double>) {
    MOCHI_CUBLAS_CHECK(cublasDgemm(
        blasHandle, transA, transB, m, n, k, &alpha, Adata, ldA, Bdata, ldB, &beta, C.v, ldC));
  } else if constexpr (std::is_same_v<Scalar, float>) {
    MOCHI_CUBLAS_CHECK(cublasSgemm(
        blasHandle, transA, transB, m, n, k, &alpha, Adata, ldA, Bdata, ldB, &beta, C.v, ldC));
  } else {
    static_assert(
        (std::is_same_v<Scalar, double>) || (std::is_same_v<Scalar, float>),
        "Scalar type not supported");
  }
}

void ExecuteCudaProduct(
    MatDataDest<double>& C,
    ScaledMatData<double const>& A,
    ScaledMatData<double const>& B,
    int m,
    int n,
    int k,
    double alpha,
    double beta) {
  ExecuteCudaProduct_impl<double>(C, A, B, m, n, k, alpha, beta);
}

void ExecuteCudaProduct(
    MatDataDest<float>& C,
    ScaledMatData<float const>& A,
    ScaledMatData<float const>& B,
    int m,
    int n,
    int k,
    float alpha,
    float beta) {
  ExecuteCudaProduct_impl<float>(C, A, B, m, n, k, alpha, beta);
}

} // namespace mochi::details

#endif // MOCHI_USE_CUDA
