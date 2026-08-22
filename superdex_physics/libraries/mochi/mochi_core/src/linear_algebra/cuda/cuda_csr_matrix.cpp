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
#include <cuda_runtime.h>
#include <cusparse.h>

#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/cuda/cuda_sparse_utils.h>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace {
// Use CUSPARSE_SPMV_CSR_ALG2 for deterministic evaluations
// The default algorithm for CSR format (CUSPARSE_SPMV_CSR_ALG1)
// does not yield deterministic results.
// see https://docs.nvidia.com/cuda/cusparse/#cusparsespmv
// The algorithm CUSPARSE_SPMV_CSR_ALG1 can be faster.
// For a 3D Laplacian with 27 point stencil, the improvement is around 5 to 10 %.
cusparseSpMVAlg_t csrAlgSpMV = CUSPARSE_SPMV_CSR_ALG2;

// Use CUSPARSE_SPMM_CSR_ALG3 for deterministic evaluations
// The other algorithms for CSR format (CUSPARSE_SPMM_CSR_ALG1 and [..]_ALG2)
// do not yield deterministic results.
// see https://docs.nvidia.com/cuda/cusparse/#cusparsespmm
// Algorithm 3 for CSR/CSC sparse matrix format.
//
// Algorithm 3 supports only opA == CUSPARSE_OPERATION_NON_TRANSPOSE
// csrAlgSpMM could be overwritten at runtime when the simulation
// asks for the transpose matrix-vector product.
// In that case, the default algorithm CUSPARSE_SPMM_ALG_DEFAULT
// is selected (but it does not guarantee deterministic results).
cusparseSpMMAlg_t csrAlgSpMM = CUSPARSE_SPMM_CSR_ALG3;

} // namespace

namespace mochi::details {

template <typename Scalar, typename ColIdx, typename RowPtr>
std::unique_ptr<CudaConstSparseMatDescr, ReleaseSparseMatDescr> InitializeCsr(
    int64_t nRow,
    int64_t nCol,
    int64_t nnz,
    RowPtr const* ptr,
    ColIdx const* col,
    Scalar const* val) {
  static_assert(
      std::is_same_v<ColIdx, int32_t> || std::is_same_v<ColIdx, int64_t>,
      "Unsupported integral type");
  constexpr cusparseIndexType_t idxFlag =
      (std::is_same_v<ColIdx, int32_t>) ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
  static_assert(
      std::is_same_v<RowPtr, int32_t> || std::is_same_v<RowPtr, int64_t>,
      "Unsupported integral type");
  constexpr cusparseIndexType_t ptrFlag =
      (std::is_same_v<RowPtr, int32_t>) ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
  static_assert(
      std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>, "Unsupported scalar type");
  constexpr cudaDataType scalar = std::is_same_v<Scalar, double> ? CUDA_R_64F : CUDA_R_32F;

  //--- Create CSR descriptor for CUDA routines
  cusparseConstSpMatDescr_t matA = nullptr;
  MOCHI_CUSPARSE_CHECK(cusparseCreateConstCsr(
      &matA,
      nRow,
      nCol,
      nnz,
      reinterpret_cast<void const*>(ptr),
      reinterpret_cast<void const*>(col),
      reinterpret_cast<void const*>(val),
      ptrFlag,
      idxFlag,
      CUSPARSE_INDEX_BASE_ZERO,
      scalar));
  return std::unique_ptr<CudaConstSparseMatDescr, ReleaseSparseMatDescr>{
      reinterpret_cast<CudaConstSparseMatDescr*>(matA)};
}

#define MOCHI_INSTANTIATE_INITIALIZE_CSR(SCALAR_TYPE, COL_IDX_TYPE, ROW_PTR_TYPE) \
  template std::unique_ptr<CudaConstSparseMatDescr, ReleaseSparseMatDescr>        \
  InitializeCsr<SCALAR_TYPE, COL_IDX_TYPE, ROW_PTR_TYPE>(                         \
      int64_t nRow,                                                               \
      int64_t nCol,                                                               \
      int64_t nnz,                                                                \
      ROW_PTR_TYPE const* ptr,                                                    \
      COL_IDX_TYPE const* col,                                                    \
      SCALAR_TYPE const* val);

MOCHI_INSTANTIATE_INITIALIZE_CSR(float, int32_t, int32_t)
MOCHI_INSTANTIATE_INITIALIZE_CSR(float, int64_t, int64_t)
MOCHI_INSTANTIATE_INITIALIZE_CSR(double, int32_t, int32_t)
MOCHI_INSTANTIATE_INITIALIZE_CSR(double, int64_t, int64_t)

#undef MOCHI_INSTANTIATE_INITIALIZE_CSR

static cusparseOperation_t GetTransposeFlag(bool useTranspose) {
  return (useTranspose) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
}

static cusparseOrder_t GetMemoryLayout(bool isRowMajor) {
  return (isRowMajor) ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
}

template <typename Scalar>
void SizeBufferSpMM(
    bool useTransposeA,
    bool useTransposeX,
    CudaConstSparseMatDescr* csrDescr,
    Scalar const* x,
    int64_t xrows,
    int64_t xcols,
    int64_t ldx,
    bool isXRowMajor,
    Scalar* y,
    int64_t yrows,
    int64_t ycols,
    int64_t ldy,
    bool isYRowMajor,
    size_t& bufferSize,
    void** buffer) {
  auto matA = reinterpret_cast<cusparseConstSpMatDescr_t>(csrDescr);
  auto const one = Scalar(1), zero = Scalar(0);
  static_assert(
      std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>, "Unsupported scalar type");
  constexpr cudaDataType scalar = std::is_same_v<Scalar, double> ? CUDA_R_64F : CUDA_R_32F;
  cusparseConstDnMatDescr_t descrX = nullptr;
  cusparseDnMatDescr_t descrY = nullptr;
  MOCHI_CUSPARSE_CHECK(cusparseCreateConstDnMat(
      &descrX, xrows, xcols, ldx, x, scalar, GetMemoryLayout(isXRowMajor)));
  MOCHI_CUSPARSE_CHECK(
      cusparseCreateDnMat(&descrY, yrows, ycols, ldy, y, scalar, GetMemoryLayout(isYRowMajor)));
  auto cuspHandle = reinterpret_cast<cusparseHandle_t>(mochi::details::GetCuSparseHandle());
  MOCHI_CUSPARSE_CHECK(cusparseSetPointerMode(cuspHandle, CUSPARSE_POINTER_MODE_HOST));
  // allocate an external buffer if needed
  std::size_t newSize = 0;
  // CUSPARSE_SPMM_CSR_ALG3 supports only opA == CUSPARSE_OPERATION_NON_TRANSPOSE
  // When the simulation asks for the transpose matrix-vector product,
  // the default algorithm CUSPARSE_SPMM_ALG_DEFAULT
  // is selected (but it does not guarantee deterministic results).
  cusparseSpMMAlg_t const algo = (useTransposeA) ? CUSPARSE_SPMM_ALG_DEFAULT : ::csrAlgSpMM;
  MOCHI_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      cuspHandle,
      GetTransposeFlag(useTransposeA),
      GetTransposeFlag(useTransposeX),
      &one,
      matA,
      descrX,
      &zero,
      descrY,
      scalar,
      algo,
      &newSize));
  if (newSize > bufferSize) {
    bufferSize = newSize;
    MOCHI_CUDA_CHECK(cudaFree(*buffer));
    MOCHI_CUDA_CHECK(cudaMalloc(buffer, bufferSize));
  }
  // execute preprocess (optional)
  MOCHI_CUSPARSE_CHECK(cusparseSpMM_preprocess(
      cuspHandle,
      GetTransposeFlag(useTransposeA),
      GetTransposeFlag(useTransposeX),
      &one,
      matA,
      descrX,
      &zero,
      descrY,
      scalar,
      algo,
      *buffer));
  //--- Destroy descriptor
  MOCHI_CUSPARSE_CHECK(cusparseDestroyDnMat(descrX));
  MOCHI_CUSPARSE_CHECK(cusparseDestroyDnMat(descrY));
}

template void SizeBufferSpMM<double>(
    bool useTransposeA,
    bool useTransposeX,
    CudaConstSparseMatDescr* csrDescr,
    double const* x,
    int64_t xrows,
    int64_t xcols,
    int64_t ldx,
    bool isXRowMajor,
    double* y,
    int64_t yrows,
    int64_t ycols,
    int64_t ldy,
    bool isYRowMajor,
    size_t& bufferSize,
    void** buffer);

template void SizeBufferSpMM<float>(
    bool useTransposeA,
    bool useTransposeX,
    CudaConstSparseMatDescr* csrDescr,
    float const* x,
    int64_t xrows,
    int64_t xcols,
    int64_t ldx,
    bool isXRowMajor,
    float* y,
    int64_t yrows,
    int64_t ycols,
    int64_t ldy,
    bool isYRowMajor,
    size_t& bufferSize,
    void** buffer);

template <typename Scalar>
void SizeBufferSpMV(
    bool useTranspose,
    CudaConstSparseMatDescr* csrDescr,
    Scalar const* dX,
    int64_t n,
    Scalar* dY,
    int64_t m,
    size_t& bufferSize,
    void** buffer) {
  static_assert(
      std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>, "Unsupported scalar type");
  constexpr cudaDataType scalar = std::is_same_v<Scalar, double> ? CUDA_R_64F : CUDA_R_32F;
  // Create dense vector X
  cusparseConstDnVecDescr_t vecX;
  MOCHI_CUSPARSE_CHECK(cusparseCreateConstDnVec(&vecX, n, dX, scalar));
  // Create dense vector y
  cusparseDnVecDescr_t vecY;
  MOCHI_CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, m, dY, scalar));
  // Allocate an external buffer if needed
  auto matA = reinterpret_cast<cusparseConstSpMatDescr_t>(csrDescr);
  auto cuspHandle = reinterpret_cast<cusparseHandle_t>(mochi::details::GetCuSparseHandle());
  MOCHI_CUSPARSE_CHECK(cusparseSetPointerMode(cuspHandle, CUSPARSE_POINTER_MODE_HOST));
  auto const one = Scalar(1);
  auto const zero = Scalar(0);
  std::size_t newSize = 0;
  MOCHI_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      cuspHandle,
      GetTransposeFlag(useTranspose),
      &one,
      matA,
      vecX,
      &zero,
      vecY,
      scalar,
      ::csrAlgSpMV,
      &newSize));
  if (newSize > bufferSize) {
    bufferSize = newSize;
    MOCHI_CUDA_CHECK(cudaFree(*buffer));
    MOCHI_CUDA_CHECK(cudaMalloc(buffer, bufferSize));
  }
  // execute preprocess (optional)
  MOCHI_CUSPARSE_CHECK(cusparseSpMV_preprocess(
      cuspHandle,
      GetTransposeFlag(useTranspose),
      &one,
      matA,
      vecX,
      &zero,
      vecY,
      scalar,
      ::csrAlgSpMV,
      *buffer));
  //
  MOCHI_CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
  MOCHI_CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
}

template void SizeBufferSpMV<double>(
    bool useTranspose,
    CudaConstSparseMatDescr* csrDescr,
    double const* dX,
    int64_t n,
    double* dY,
    int64_t m,
    size_t& bufferSize,
    void** buffer);

template void SizeBufferSpMV<float>(
    bool useTranspose,
    CudaConstSparseMatDescr* csrDescr,
    float const* dX,
    int64_t n,
    float* dY,
    int64_t m,
    size_t& bufferSize,
    void** buffer);

template <typename Scalar>
void MultiplyCsrVec(
    bool useTranspose,
    CudaConstSparseMatDescr* csrDescr,
    void* buffer,
    Scalar const* x,
    int64_t xrows,
    Scalar* y,
    int64_t yrows) {
  static_assert(
      std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>, "Unsupported scalar type");
  constexpr cudaDataType scalar = std::is_same_v<Scalar, double> ? CUDA_R_64F : CUDA_R_32F;
  auto matA = reinterpret_cast<cusparseConstSpMatDescr_t>(csrDescr);
  auto const one = Scalar(1), zero = Scalar(0);
  cusparseConstDnVecDescr_t descrX = nullptr;
  cusparseDnVecDescr_t descrY = nullptr;
  auto const transposeFlag = GetTransposeFlag(useTranspose);
  MOCHI_CUSPARSE_CHECK(cusparseCreateConstDnVec(&descrX, xrows, x, scalar));
  MOCHI_CUSPARSE_CHECK(cusparseCreateDnVec(&descrY, yrows, y, scalar));
  auto cuspHandle = reinterpret_cast<cusparseHandle_t>(mochi::details::GetCuSparseHandle());
  MOCHI_CUSPARSE_CHECK(cusparseSetPointerMode(cuspHandle, CUSPARSE_POINTER_MODE_HOST));
  //--- Do the multiplication
  MOCHI_CUSPARSE_CHECK(cusparseSpMV(
      cuspHandle, transposeFlag, &one, matA, descrX, &zero, descrY, scalar, ::csrAlgSpMV, buffer));
  //--- Destroy descriptor
  MOCHI_CUSPARSE_CHECK(cusparseDestroyDnVec(descrX));
  MOCHI_CUSPARSE_CHECK(cusparseDestroyDnVec(descrY));
}

template void MultiplyCsrVec<double>(
    bool useTranspose,
    CudaConstSparseMatDescr* csrDescr,
    void* buffer,
    double const* x,
    int64_t xrows,
    double* y,
    int64_t yrows);

template void MultiplyCsrVec<float>(
    bool useTranspose,
    CudaConstSparseMatDescr* csrDescr,
    void* buffer,
    float const* x,
    int64_t xrows,
    float* y,
    int64_t yrows);

template <typename Scalar>
void MultiplyCsrMat(
    bool useTransposeA,
    bool useTransposeX,
    CudaConstSparseMatDescr* csrDescr,
    void* buffer,
    Scalar const* x,
    int64_t xrows,
    int64_t xcols,
    int64_t ldx,
    bool isXRowMajor,
    Scalar* y,
    int64_t yrows,
    int64_t ycols,
    int64_t ldy,
    bool isYRowMajor) {
  static_assert(
      std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>, "Unsupported scalar type");
  constexpr cudaDataType scalar = std::is_same_v<Scalar, double> ? CUDA_R_64F : CUDA_R_32F;
  auto matA = reinterpret_cast<cusparseConstSpMatDescr_t>(csrDescr);
  auto const one = Scalar(1), zero = Scalar(0);
  cusparseConstDnMatDescr_t descrX = nullptr;
  cusparseDnMatDescr_t descrY = nullptr;
  MOCHI_CUSPARSE_CHECK(cusparseCreateConstDnMat(
      &descrX, xrows, xcols, ldx, x, scalar, GetMemoryLayout(isXRowMajor)));
  MOCHI_CUSPARSE_CHECK(
      cusparseCreateDnMat(&descrY, yrows, ycols, ldy, y, scalar, GetMemoryLayout(isYRowMajor)));
  auto cuspHandle = reinterpret_cast<cusparseHandle_t>(mochi::details::GetCuSparseHandle());
  MOCHI_CUSPARSE_CHECK(cusparseSetPointerMode(cuspHandle, CUSPARSE_POINTER_MODE_HOST));
  // CUSPARSE_SPMM_CSR_ALG3 supports only opA == CUSPARSE_OPERATION_NON_TRANSPOSE
  // When the simulation asks for the transpose matrix-vector product,
  // the default algorithm CUSPARSE_SPMM_ALG_DEFAULT
  // is selected (but it does not guarantee deterministic results).
  cusparseSpMMAlg_t const algo = (useTransposeA) ? CUSPARSE_SPMM_ALG_DEFAULT : csrAlgSpMM;
  //--- Do the multiplication
  MOCHI_CUSPARSE_CHECK(cusparseSpMM(
      cuspHandle,
      GetTransposeFlag(useTransposeA),
      GetTransposeFlag(useTransposeX),
      &one,
      matA,
      descrX,
      &zero,
      descrY,
      scalar,
      algo,
      buffer));
  //--- Destroy descriptor
  MOCHI_CUSPARSE_CHECK(cusparseDestroyDnMat(descrX));
  MOCHI_CUSPARSE_CHECK(cusparseDestroyDnMat(descrY));
}

template void MultiplyCsrMat<double>(
    bool useTransposeA,
    bool useTransposeX,
    CudaConstSparseMatDescr* csrDescr,
    void* buffer,
    double const* x,
    int64_t xrows,
    int64_t xcols,
    int64_t ldx,
    bool isXRowMajor,
    double* y,
    int64_t yrows,
    int64_t ycols,
    int64_t ldy,
    bool isYRowMajor);

template void MultiplyCsrMat<float>(
    bool useTransposeA,
    bool useTransposeX,
    CudaConstSparseMatDescr* csrDescr,
    void* buffer,
    float const* x,
    int64_t xrows,
    int64_t xcols,
    int64_t ldx,
    bool isXRowMajor,
    float* y,
    int64_t yrows,
    int64_t ycols,
    int64_t ldy,
    bool isYRowMajor);

} // namespace mochi::details
#endif // MOCHI_USE_CUDA
