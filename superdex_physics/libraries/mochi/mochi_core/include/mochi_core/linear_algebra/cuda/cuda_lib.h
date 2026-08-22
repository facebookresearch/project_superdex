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

//
// !!!  This header file should be handled carefully  !!!
// !!! Including this file could expose CUDA headers. !!!
//

#include <mochi_core/linear_algebra/cuda/cuda_api.h>
#include <mochi_core/linear_algebra/matrix_accessors_fwd.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/debug.h>

namespace mochi::details {

// !!! The next variables should be fine-tuned to the device
// for optimal performance. !!!
[[maybe_unused]] constexpr int kNThreads = 256;
[[maybe_unused]] constexpr int kTileDim = 32; // kTileDim must be a multiple of kBlockRows
[[maybe_unused]] constexpr int kBlockRows = 16;
static_assert(kTileDim % kBlockRows == 0, "kTileDim must be a multiple of kBlockRows");

// @brief CudaBLASHandle is hiding the CUDA type `cublasHandle_t`
// The type CudaBLASHandle will need to be `cast` into `cublasHandle_t`
// when calling a cublas-specific routine.
using CudaBLASHandle = void*;

// @brief CudaSparseHandle is hiding the CUDA type `cusparseHandle_t`
// The type CudaSparseHandle will need to be `cast` into `cusparseHandle_t`
// when calling a cusparse-specific routine.
using CudaSparseHandle = void*;

// @brief CudaSolverSpHandle is hiding the CUDA type `cusolverSpHandle_t`
// The type CudaSolverSpHandle will need to be `cast` into `cusolverSpHandle_t`
// when calling a cusolverSp-specific routine.
using CudaSolverSpHandle = void*;

#if MOCHI_USE_CUDSS
// @brief CudaDSSHandle is hiding the CUDA type `cudssHandle_t`
// The type CudaDSSHandle will need to be `cast` into `cudssHandle_t`
// when calling a cudss-specific routine.
using CudaDSSHandle = void*;
#endif

template <typename Scalar>
struct PackedScaleData {
  mochi::details::ScaledMatData<Scalar> v[kMaxNumTermsInExpression];
};

/// @brief Returns handle for CUBLAS functions
CudaBLASHandle GetCuBLASHandle();

/// @brief Returns handle for CuSparse functions
CudaSparseHandle GetCuSparseHandle();

/// @brief Returns handle for cusolverSp functions
CudaSolverSpHandle GetCuSolverSpHandle();

#if MOCHI_USE_CUDSS
/// @brief Returns handle for cudss functions
CudaDSSHandle GetCuDSSHandle();
#endif

template <typename Scalar>
void DoCudaScaledAssign(int N, Scalar* dest, int nPos, PackedScaleData<Scalar const>& posSrc);

template <typename Scalar>
void DoCudaScaled2DAssign(
    int nRows,
    int nCols,
    MatDataDest<Scalar> dest,
    int numTerms,
    PackedScaleData<Scalar const>& terms);

/// @brief Extract diagonal blocks of size (blockSize x blockSize)
/// from a sparse matrix stored in CSR format and invert each block
template <typename Scalar, typename ColIdx, typename RowPtr>
void CudaExtractInverseDiagonal(
    size_t nBlockRows,
    int blockSize,
    RowPtr const* rowPtr,
    ColIdx const* colIdx,
    Scalar const* values,
    Scalar* diagValues);

/// @brief Extract diagonal blocks of size (blockSize x blockSize)
/// from a sparse matrix stored in BSR format and invert each block
template <typename Scalar, typename ColIdx, typename RowPtr>
void CudaExtractInverseDiagonal(
    size_t nBlockRows,
    int blockSize,
    int bsrBlockSize,
    RowPtr const* bsrRowPtr,
    ColIdx const* bsrColIdx,
    Scalar const* bsrValues,
    Scalar* diagValues);

#if MOCHI_USE_CUDA

// Helper macro for calling a CUDA function. It will automatically assert
// the returned error code. Usage example:
// cudaMalloc(...) --> MOCHI_CUDA_CHECK(cudaMalloc(...))
#define MOCHI_CUDA_CHECK(EXPR)                                                 \
  do {                                                                         \
    [[maybe_unused]] cudaError_t err_code = EXPR;                              \
    MOCHI_ASSERT(err_code == cudaSuccess, "%s", cudaGetErrorString(err_code)); \
  } while (false)

// Helper macro for checking the last error in the CUDA runtime. Useful for
// testing if anything went wrong e.g. after a kernel launch.
#define MOCHI_CUDA_CHECK_LAST() MOCHI_CUDA_CHECK(cudaGetLastError())

// Helper macro for calling a cuBLAS function.
// It will automatically assert the returned error code.
#define MOCHI_CUBLAS_CHECK(EXPR)                                     \
  do {                                                               \
    [[maybe_unused]] cublasStatus_t err_code = EXPR;                 \
    MOCHI_ASSERT(err_code == CUBLAS_STATUS_SUCCESS, "%d", err_code); \
  } while (false)

// Helper macro for calling a cuSparse function.
// It will automatically assert the returned error code.
#define MOCHI_CUSPARSE_CHECK(EXPR)                                                             \
  do {                                                                                         \
    [[maybe_unused]] cusparseStatus_t err_code = EXPR;                                         \
    MOCHI_ASSERT(err_code == CUSPARSE_STATUS_SUCCESS, "%s", cusparseGetErrorString(err_code)); \
  } while (false)

// Helper macro for calling a cusolver function.
// It will automatically assert the returned error code.
#define MOCHI_CUSOLVER_CHECK(EXPR)                                     \
  do {                                                                 \
    [[maybe_unused]] cusolverStatus_t err_code = EXPR;                 \
    MOCHI_ASSERT(err_code == CUSOLVER_STATUS_SUCCESS, "%d", err_code); \
  } while (false)

#if MOCHI_USE_CUDSS
// Helper macro for calling a cudss function.
// It will automatically assert the returned error code.
#define MOCHI_CUDSS_CHECK(EXPR)                                     \
  do {                                                              \
    [[maybe_unused]] cudssStatus_t err_code = EXPR;                 \
    MOCHI_ASSERT(err_code == CUDSS_STATUS_SUCCESS, "%d", err_code); \
  } while (false)
#endif // MOCHI_USE_CUDSS

#endif // MOCHI_USE_CUDA

} // namespace mochi::details
