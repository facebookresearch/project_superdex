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

#include <mochi_core/linear_algebra/base_enums.h>
#include <memory>

namespace mochi::details {

//
//--- CSR matrices
//

using CudaSparseMatDescr = void;
using CudaConstSparseMatDescr = CudaSparseMatDescr const;

void CudaSparseMatDescrRelease(CudaConstSparseMatDescr* p);
struct ReleaseSparseMatDescr {
  void operator()(CudaConstSparseMatDescr* p) const {
    CudaSparseMatDescrRelease(p);
  }
};

template <typename Scalar, typename ColIdx, typename RowPtr>
std::unique_ptr<CudaConstSparseMatDescr, ReleaseSparseMatDescr> InitializeCsr(
    int64_t nRow,
    int64_t nCol,
    int64_t nnz,
    RowPtr const* ptr,
    ColIdx const* col,
    Scalar const* val);

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
    void** buffer);

/// @brief Routine to compute matrix-vector product
/// The sparse matrix is stored with the CSR format.
/// The vector x is stored as a column-vector (i.e. row increment = 1)
/// The vector y is stored as a column-vector (i.e. row increment = 1)
///
/// @tparam Scalar
/// @param useTranspose
/// @param csrDescr
/// @param buffer
/// @param x Input dense vector on the device
/// @param xrows Number of rows in x
/// @param y Output dense vector on the device
/// @param yrows Number of rows in y
///
/// @note The routine calls [cusparseSpMV](https://docs.nvidia.com/cuda/cusparse/#cusparsespmv).
template <typename Scalar>
void MultiplyCsrVec(
    bool useTranspose,
    CudaConstSparseMatDescr* csrDescr,
    void* buffer,
    Scalar const* x,
    int64_t xrows,
    Scalar* y,
    int64_t yrows);

/// @brief Routine to compute matrix-matrix product
/// The sparse matrix is stored with the CSR format.
/// The matrix X has column-major storage (i.e. row increment = 1)
/// The matrix Y has column-major storage (i.e. row increment = 1)
///
/// @tparam Scalar
/// @param useTransposeA
/// @param useTransposeX
/// @param csrDescr
/// @param buffer
/// @param x Input dense vector on the device
/// @param xrows Number of rows in x
/// @param xcols Number of columns in x
/// @param ldx Leading dimension in x
/// @param isXRowMajor
/// @param y Output dense vector on the device
/// @param yrows Number of rows in y
/// @param ycols Number of columns y
/// @param ldy Leading dimension in y
/// @param isYRowMajor
///
/// @note The routine calls [cusparseSpMM](https://docs.nvidia.com/cuda/cusparse/#cusparsespmm).
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
    bool isYRowMajor);

//
//--- BSR matrices
//

using CudaBsrMatDescr = void;

void CudaBsrMatDescrRelease(CudaBsrMatDescr* p);
struct ReleaseCudaBsrMatDescr {
  void operator()(CudaBsrMatDescr* p) const {
    CudaBsrMatDescrRelease(p);
  }
};

std::unique_ptr<CudaBsrMatDescr, ReleaseCudaBsrMatDescr> CreateCudaBsrMatDescriptor();

/// @brief Routine to do sparse matrix vector product on the device
/// Use legacy cuSparse routine `cusparse<t>bsrmm()`
///
/// @tparam Scalar
/// @tparam kBlockStorage
/// @param descr
/// @param useTranspose
/// @param mb
/// @param nb
/// @param nnzb
/// @param blockSize
/// @param bsrRowPtr Integral offsets for the BSR sparse matrix on the device
/// @param bsrColIdx Integral column indices for the BSR sparse matrix on the device
/// @param bsrVal Scalar values for the BSR sparse matrix on the device
/// @param x Input dense vector on the device
/// @param y Output dense vector on the device
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
    Scalar* y);

/// @brief Routine to do sparse matrix vector product on the device
/// Use legacy cuSparse routine `cusparse<t>bsrmm()`
///
/// @tparam Scalar
/// @tparam kBlockStorage
/// @param descr
/// @param useTransposeA
/// @param useTransposeX
/// @param mb
/// @param nb
/// @param nnzb
/// @param blockSize
/// @param bsrRowPtr Integral offsets for the BSR sparse matrix on the device
/// @param bsrColIdx Integral column indices for the BSR sparse matrix on the device
/// @param bsrVal Scalar values for the BSR sparse matrix on the device
/// @param x Input dense vector on the device
/// @param colx Number of columns in x (and y)
/// @param ldx Leading dimension for x
/// @param y Output dense vector on the device
/// @param ldy Leading dimension for y
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
    int ldy);

} // namespace mochi::details
