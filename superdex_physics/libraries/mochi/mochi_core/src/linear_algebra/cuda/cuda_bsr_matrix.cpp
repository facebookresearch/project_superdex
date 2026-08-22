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
#include <cusparse.h>

#include <mochi_core/linear_algebra/base_enums.h>
#include <mochi_core/linear_algebra/cuda/cuda_bsr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>

#include <type_traits>

namespace mochi::details {
static cusparseOperation_t GetTransposeFlag(bool useTranspose) {
  return (useTranspose) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
}

template <typename Scalar, krylov::Direction kBlockStorage>
void MultiplyBsrVec(
    CudaBsrMatDescr* descr,
    bool useTranspose,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    Scalar const* bsrVal,
    Scalar const* x,
    Scalar* y) {
  static_assert(std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>);
  auto cuspHandle = reinterpret_cast<cusparseHandle_t>(mochi::details::GetCuSparseHandle());
  auto matDescr = reinterpret_cast<cusparseMatDescr_t>(descr);
  auto const one = Scalar(1), zero = Scalar(0);
  MOCHI_CUSPARSE_CHECK(cusparseSetPointerMode(cuspHandle, CUSPARSE_POINTER_MODE_HOST));
  constexpr cusparseDirection_t kBlockDir = (kBlockStorage == mochi::krylov::Direction::ColMajor)
      ? CUSPARSE_DIRECTION_COLUMN
      : CUSPARSE_DIRECTION_ROW;
  //
  // Per Cuda documentation, on cusparse(*)bsrmv,
  // - only blockSize > 1 is supported
  // - only CUSPARSE_OPERATION_NON_TRANSPOSE is supported
  //
  if constexpr (std::is_same_v<Scalar, double>) {
    MOCHI_CUSPARSE_CHECK(cusparseDbsrmv(
        cuspHandle,
        kBlockDir,
        GetTransposeFlag(useTranspose),
        mb,
        nb,
        nnzb,
        &one,
        matDescr,
        bsrVal,
        bsrRowPtr,
        bsrColIdx,
        blockSize,
        x,
        &zero,
        y));
  } else {
    // float case (enforced by the static_assert above)
    MOCHI_CUSPARSE_CHECK(cusparseSbsrmv(
        cuspHandle,
        kBlockDir,
        GetTransposeFlag(useTranspose),
        mb,
        nb,
        nnzb,
        &one,
        matDescr,
        bsrVal,
        bsrRowPtr,
        bsrColIdx,
        blockSize,
        x,
        &zero,
        y));
  }
}

template void MultiplyBsrVec<double, krylov::Direction::ColMajor>(
    CudaBsrMatDescr* descr,
    bool useTranspose,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    double const* bsrVal,
    double const* x,
    double* y);

template void MultiplyBsrVec<double, krylov::Direction::RowMajor>(
    CudaBsrMatDescr* descr,
    bool useTranspose,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    double const* bsrVal,
    double const* x,
    double* y);

template void MultiplyBsrVec<float, krylov::Direction::ColMajor>(
    CudaBsrMatDescr* descr,
    bool useTranspose,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    float const* bsrVal,
    float const* x,
    float* y);

template void MultiplyBsrVec<float, krylov::Direction::RowMajor>(
    CudaBsrMatDescr* descr,
    bool useTranspose,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    float const* bsrVal,
    float const* x,
    float* y);

template <typename Scalar, krylov::Direction kBlockStorage>
void MultiplyBsrMat(
    CudaBsrMatDescr* descr,
    bool useTransposeA,
    bool useTransposeX,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    Scalar const* bsrVal,
    Scalar const* x,
    int colx,
    int ldx,
    Scalar* y,
    int ldy) {
  static_assert(std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>);
  // Use legacy cuSparse routine `cusparse<t>bsrmm()`
  // Only blockSize > 1 is supported
  // Nvidia CuSparse documentation - Section 9.1
  // https://docs.nvidia.com/cuda/pdf/CUSPARSE_Library.pdf
  //
  // CUDA v. 12.8 -- `cusparse<t>bsrmm()` is listed as deprecated.
  // The documentation recommends to use cusparseSpMM()
  // However, the algorithm CUSPARSE_SPMM_BSR_ALG1 does not support column-major blocks
  //
  auto cuspHandle = reinterpret_cast<cusparseHandle_t>(mochi::details::GetCuSparseHandle());
  auto matDescr = reinterpret_cast<cusparseMatDescr_t>(descr);
  auto const one = Scalar(1), zero = Scalar(0);
  MOCHI_CUSPARSE_CHECK(cusparseSetPointerMode(cuspHandle, CUSPARSE_POINTER_MODE_HOST));
  constexpr cusparseDirection_t kBlockDir = (kBlockStorage == mochi::krylov::Direction::ColMajor)
      ? CUSPARSE_DIRECTION_COLUMN
      : CUSPARSE_DIRECTION_ROW;
  if constexpr (std::is_same_v<Scalar, double>) {
    MOCHI_CUSPARSE_CHECK(cusparseDbsrmm(
        cuspHandle,
        kBlockDir,
        GetTransposeFlag(useTransposeA),
        GetTransposeFlag(useTransposeX),
        mb,
        colx,
        nb,
        nnzb,
        &one,
        matDescr,
        bsrVal,
        bsrRowPtr,
        bsrColIdx,
        blockSize,
        x,
        ldx,
        &zero,
        y,
        ldy));
  } else {
    // float case (enforced by the static_assert above)
    MOCHI_CUSPARSE_CHECK(cusparseSbsrmm(
        cuspHandle,
        kBlockDir,
        GetTransposeFlag(useTransposeA),
        GetTransposeFlag(useTransposeX),
        mb,
        colx,
        nb,
        nnzb,
        &one,
        matDescr,
        bsrVal,
        bsrRowPtr,
        bsrColIdx,
        blockSize,
        x,
        ldx,
        &zero,
        y,
        ldy));
  }
}

template void MultiplyBsrMat<double, krylov::Direction::ColMajor>(
    CudaBsrMatDescr* descr,
    bool useTransposeA,
    bool useTransposeX,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    double const* bsrVal,
    double const* x,
    int colx,
    int ldx,
    double* y,
    int ldy);

template void MultiplyBsrMat<double, krylov::Direction::RowMajor>(
    CudaBsrMatDescr* descr,
    bool useTransposeA,
    bool useTransposeX,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    double const* bsrVal,
    double const* x,
    int colx,
    int ldx,
    double* y,
    int ldy);

template void MultiplyBsrMat<float, krylov::Direction::ColMajor>(
    CudaBsrMatDescr* descr,
    bool useTransposeA,
    bool useTransposeX,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    float const* bsrVal,
    float const* x,
    int colx,
    int ldx,
    float* y,
    int ldy);

template void MultiplyBsrMat<float, krylov::Direction::RowMajor>(
    CudaBsrMatDescr* descr,
    bool useTransposeA,
    bool useTransposeX,
    int mb,
    int nb,
    int nnzb,
    int blockSize,
    int const* bsrRowPtr,
    int const* bsrColIdx,
    float const* bsrVal,
    float const* x,
    int colx,
    int ldx,
    float* y,
    int ldy);

} // namespace mochi::details

#endif // MOCHI_USE_CUDA
