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

#include <mochi_core/linear_algebra/cuda/cuda_api.h>
#include <mochi_core/linear_algebra/cuda/cuda_bsr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_csr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>

#include <type_traits>

namespace mochi::krylov {

template <typename Scalar, int kPrecBlockSize>
struct CudaBlockJacobiPrec final {
  static_assert(kPrecBlockSize > 0, "Incompatible requested block size for the preconditioner.");
  static_assert(
      std::is_same_v<Scalar const, double const> || std::is_same_v<Scalar const, float const>,
      "Unsupported scalar type");
  static_assert(
      (MOCHI_USE_CUDA) || std::is_same_v<Scalar, void>,
      "CudaBlockJacobiPrec requires building with CUDA. To enable CUDA, add the CUDA dependencies to your build configuration and define MOCHI_USE_CUDA=1");

  template <typename ScalarA, typename Index, typename PtrIndex>
  explicit CudaBlockJacobiPrec(CudaCsrMatrix<ScalarA, Index, PtrIndex> const& A);

  template <
      typename ScalarA,
      int kBlockSize,
      typename Index,
      typename PtrIndex,
      krylov::Direction kBlockStorage>
  explicit CudaBlockJacobiPrec(
      CudaBsrMatrix<ScalarA, kBlockSize, Index, PtrIndex, kBlockStorage> const& A);

  /**
   * @brief Constructor from an input dense matrix.
   *
   * @note
   * CUDA block Jacobi preconditioner is not currently supported for dense matrices.
   *
   */
  template <
      typename ScalarA,
      int kRowsAtCompileTime,
      int kColsAtCompileTime,
      Direction kMajorDirection,
      int kLeadingDim>
  explicit CudaBlockJacobiPrec(
      CudaMatrix<
          ScalarA,
          kRowsAtCompileTime,
          kColsAtCompileTime,
          kMajorDirection,
          kLeadingDim> const& A);

  /**
   * @brief Apply the preconditioner to a column vector (or a set of column vectors).
   * @param[in] x  Input column vector(s)
   * @param[out] y  Output column vector(s)
   *
   * @note
   * The implementation uses the static CuBLAS handle without modifying its stream.
   * When kPrecBlockSize > 1, CuBLAS routines are called for the operation.
   * When kPrecBlockSize is equal to 1, a basic kernel is run on the stream attached
   * to the CuBLAS handle.
   *
   */
  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const;

  // TODO Put new update here. MLX

 protected:
  /** @brief Protected function to verify several parameters from input matrix
   * @note
   * It is assumed that the user will guarantee that the types
   * 'ScalarA' and 'Matrix' are consistent.
   */
  template <typename ScalarA, typename Matrix>
  void VerifyParameters(Matrix const& A);

  CudaMatrix<Scalar, kPrecBlockSize, krylov::kDynamic> _inverseDiagBlocks;
};

template <typename Scalar>
using CudaJacobiPrec = CudaBlockJacobiPrec<Scalar, 1>;

} // namespace mochi::krylov

namespace mochi::details {

/// @brief Extract diagonal blocks of size (blockSize x blockSize) from a sparse matrix stored in
/// BSR format and invert each block
///
/// @tparam Scalar Scalar type
/// @tparam ColIdx Integral type for block column indices
/// @tparam RowPtr Integral type for offset indices
/// @tparam kBlockStorage Storage orientation for individual dense blocks
/// @param nBlockRows Number of block rows
/// @param blockSize Block size of output diagonal blocks
/// @param bsrBlockSize Block size of input BSR matrix
/// @param bsrRowPtr Array of offset indices (pointer on the device)
/// @param bsrColIdx Array of block column indices (pointer on the device)
/// @param bsrValues Array of scalar values (pointer on the device)
/// @param diagValues Array of diagonal blocks (pointer on the device)
template <typename Scalar, typename ColIdx, typename RowPtr, krylov::Direction kBlockStorage>
void ExtractInverseDiagBlocks(
    size_t nBlockRows,
    int blockSize,
    int bsrBlockSize,
    RowPtr const* bsrRowPtr,
    ColIdx const* bsrColIdx,
    Scalar const* bsrValues,
    Scalar* diagValues);

/// @brief Apply diagonal matrix on the device to input vector(s) on the device
///
/// @tparam Scalar Scalar type
/// @param nRows Number of block rows
/// @param D Diagonal entries (pointer on the device)
/// @param x Dense input vectors stored in column-major (pointer on the device)
/// @param ldx Leading dimension for x
/// @param colx Number of column-vectors in x
/// @param y Dense output vectors stored in column-major (pointer on the device)
/// @param ldy Leading dimension for y
/// @param coly Number of column-vectors in y
///
/// @note
/// coly must match colx.
///
/// @note
/// When colx = coly = 1, the application can not be in-place.
///
/// @note
/// When colx > 1, the routine `cublas<t>dgmm` is called.
///
template <typename Scalar>
void ApplyDiagonal(
    int nRows,
    Scalar const* D,
    Scalar const* x,
    int ldx,
    int colx,
    Scalar* y,
    int ldy,
    int coly);

/// @brief Apply block diagonal matrix on the device to input vector(s) on the device
///
/// @tparam Scalar
/// @param nRowBlocks
/// @param aRows
/// @param aCols
/// @param A Input block diagonal matrix (pointer on the device)
/// @param lda
/// @param strideA
/// @param x Dense input vectors stored in column-major (pointer on the device)
/// @param ldx Leading dimension for x
/// @param colx Number of column-vectors in x
/// @param strideX Stride for each block in x
/// @param y Dense output vectors stored in column-major (pointer on the device)
/// @param ldy Leading dimension for y
/// @param coly Number of column-vectors in y
/// @param strideY Stride for each block in y
///
/// @note coly must match colx.
/// @note When colx = coly = 1, the application cannot be in-place.
/// @note When colx > 1, the routine
/// [cublas<t>gemmStridedBatched](https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmstridedbatched)
/// is called.
template <typename Scalar>
void ApplyBatchedDiagonal(
    int nRowBlocks,
    int aRows,
    int aCols,
    Scalar const* A,
    int lda,
    int strideA,
    Scalar const* x,
    int ldx,
    int colx,
    int strideX,
    Scalar* y,
    int ldy,
    int coly,
    int strideY);

} // namespace mochi::details

//
//--- Implementation of functions
//

namespace mochi::krylov {

template <typename Scalar, int kPrecBlockSize>
template <typename ScalarA, typename Index, typename PtrIndex>
CudaBlockJacobiPrec<Scalar, kPrecBlockSize>::CudaBlockJacobiPrec(
    CudaCsrMatrix<ScalarA, Index, PtrIndex> const& A) {
  using Matrix = CudaCsrMatrix<ScalarA, Index, PtrIndex>;
  VerifyParameters<ScalarA, Matrix>(A);
  _inverseDiagBlocks.Resize(kPrecBlockSize, A.Rows());
  using NonConstIndex = std::remove_const_t<Index>;
  using NonConstPtrIndex = std::remove_const_t<PtrIndex>;
  using NonConstScalar = std::remove_const_t<Scalar>;
  mochi::details::ExtractInverseDiagBlocks<
      NonConstScalar,
      NonConstIndex,
      NonConstPtrIndex,
      krylov::Direction::ColMajor>(
      A.Rows() / kPrecBlockSize,
      kPrecBlockSize,
      1,
      A.Pointers().data(),
      A.Indices().data(),
      A.Values().data(),
      _inverseDiagBlocks.data());
}

template <typename Scalar, int kPrecBlockSize>
template <
    typename ScalarA,
    int kBlockSize,
    typename Index,
    typename PtrIndex,
    krylov::Direction kBlockStorage>
CudaBlockJacobiPrec<Scalar, kPrecBlockSize>::CudaBlockJacobiPrec(
    CudaBsrMatrix<ScalarA, kBlockSize, Index, PtrIndex, kBlockStorage> const& A) {
  using Matrix = CudaBsrMatrix<ScalarA, kBlockSize, Index, PtrIndex, kBlockStorage>;
  VerifyParameters<ScalarA, Matrix>(A);
  _inverseDiagBlocks.Resize(kPrecBlockSize, A.Rows());
  using NonConstIndex = std::remove_const_t<Index>;
  using NonConstPtrIndex = std::remove_const_t<PtrIndex>;
  using NonConstScalar = std::remove_const_t<Scalar>;
  mochi::details::
      ExtractInverseDiagBlocks<NonConstScalar, NonConstIndex, NonConstPtrIndex, kBlockStorage>(
          A.Rows() / kPrecBlockSize,
          kPrecBlockSize,
          kBlockSize,
          A.Pointers().data(),
          A.Indices().data(),
          A.Values().data(),
          _inverseDiagBlocks.data());
}

template <typename Scalar, int kPrecBlockSize>
template <
    typename ScalarA,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDirection,
    int kLeadingDim>
CudaBlockJacobiPrec<Scalar, kPrecBlockSize>::CudaBlockJacobiPrec(
    CudaMatrix<ScalarA, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kLeadingDim> const&
    /*A*/) {
  // TODO: Overload needed to compile but not supported yet.
  MOCHI_ASSERT(false, "CUDA block Jacobi preconditioner is not supported for dense matrices.");
}

template <typename Scalar, int kPrecBlockSize>
template <typename ScalarA, typename Matrix>
void CudaBlockJacobiPrec<Scalar, kPrecBlockSize>::VerifyParameters(
    [[maybe_unused]] Matrix const& A) {
  static_assert(std::is_same_v<Scalar const, ScalarA const>, "Incompatible scalar types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Only square matrices are supported");
  MOCHI_ASSERT_VERBOSE(
      A.Rows() % kPrecBlockSize == 0,
      "Incompatible pairing of matrix size (%d) and block size (%d)",
      A.Rows(),
      kPrecBlockSize);
}

template <typename Scalar, int kPrecBlockSize>
template <typename Input, typename Output>
void CudaBlockJacobiPrec<Scalar, kPrecBlockSize>::operator()(Input const& x, Output&& y) const {
  using Sx = std::remove_pointer_t<decltype(x.data())>;
  using Sy = std::remove_pointer_t<decltype(y.data())>;
  static_assert(
      std::is_same_v<std::remove_const_t<Sx>, std::remove_const_t<Sy>>,
      "Inconsistent scalar types");
  static_assert(std::is_same_v<std::remove_const_t<Sx>, Scalar>, "Inconsistent scalar types");
  MOCHI_ASSERT_VERBOSE(x.Rows() == _inverseDiagBlocks.Cols(), "Incompatible size of input vector");
  MOCHI_ASSERT_VERBOSE(y.Rows() == _inverseDiagBlocks.Cols(), "Incompatible size of output vector");
  MOCHI_ASSERT_VERBOSE(x.Cols() == y.Cols(), "Incompatible number of columns");
  constexpr krylov::Direction kDirX = mochi::krylov::details::MatTraits<Input>::kMajorDir;
  constexpr krylov::Direction kDirY = mochi::krylov::details::MatTraits<Output>::kMajorDir;
  static_assert(kDirX == kDirY, "Input and output must have same orientation");
  static_assert(
      kDirX == mochi::krylov::Direction::ColMajor, "Only column-major orientation is allowed");
  if constexpr (kPrecBlockSize == 1) {
    mochi::details::ApplyDiagonal(
        x.Rows(),
        _inverseDiagBlocks.data(),
        x.data(),
        x.LeadDim(),
        x.Cols(),
        y.data(),
        y.LeadDim(),
        y.Cols());
  } else {
    int const nRowBlocks = _inverseDiagBlocks.Cols() / kPrecBlockSize;
    mochi::details::ApplyBatchedDiagonal(
        nRowBlocks,
        kPrecBlockSize,
        kPrecBlockSize,
        _inverseDiagBlocks.data(),
        kPrecBlockSize,
        kPrecBlockSize * kPrecBlockSize,
        x.data(),
        x.LeadDim(),
        x.Cols(),
        kPrecBlockSize,
        y.data(),
        y.LeadDim(),
        y.Cols(),
        kPrecBlockSize);
  }
}

} // namespace mochi::krylov
