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
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_api.h>
#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_sparse_utils.h>
#include <mochi_core/linear_algebra/krylov_interop.h>

#include <memory>
#include <type_traits>

namespace mochi::krylov::details {

template <typename Scalar, krylov::Direction kMajorDir>
void ConvertCsrToBsr(
    int blockSize,
    int nBlockRows,
    int maxNNZBlockPerRow,
    int* d_csrRowPtr,
    Scalar const* d_csrVal,
    Scalar* d_bsrVal);

} // namespace mochi::krylov::details

namespace mochi::krylov {
/// @brief Class for interfacing with the Cuda BSR matrix
/// The implementation uses the CUDA language (with CUSparse library).
///
/// @tparam Scalar_  Template type for the scalar values
/// @tparam kBlockSize Block size
/// @tparam Index  Template type for the column indices
/// @tparam PtrIndex  Template type for the pointer in the CSR storage
///
/// @note
/// The current implementation supports Scalar_ = float or double, Index = int, PtrIndex = int
///
/// @note
/// The current implementation uses the CuSparse library, in particular
/// cusparseSbsrmv for A*x (where x is a single column-vector)
/// - the routine does not support A^T * x nor a different storage for x
/// https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-level-2-function-reference
/// cusparseSbsrmm for A*X (where X has multiple columns and it is column-major)
/// - the routine does not support A^T * X nor a different storage for X
/// https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-level-3-function-reference
///
/// @note
/// With experiments, it appears that storing individuals blocks
/// in column major yields a more efficient matrix-vector product.
/// We could not find any CUDA documentation supporting this observation.
///
/// @note
/// Requires building with MOCHI_USE_CUDA=1.
///
template <
    typename Scalar_,
    int kBlockSize,
    typename Index = int,
    typename PtrIndex = int,
    krylov::Direction kBlockStorage = krylov::Direction::ColMajor>
class CudaBsrMatrix {
 public:
  using Scalar = Scalar_;
  using NonConstIdx = std::remove_const_t<Index>;
  using NonConstPtrIdx = std::remove_const_t<PtrIndex>;
  using NonConstScalar = std::remove_const_t<Scalar>;

  static_assert(kBlockSize > 1, "Use CudaCsrMatrix for a block size of 1");
  static_assert(
      (MOCHI_USE_CUDA) || std::is_same_v<Scalar, void>,
      "CudaBsrMatrix requires building with CUDA. To enable CUDA, add the CUDA dependencies to your build configuration and define MOCHI_USE_CUDA=1");

  /// @note
  /// CuSparse create these limitations.
  ///
  static_assert(
      std::is_same_v<Index const, PtrIndex const>,
      "Unsupported combination of integral types");
  static_assert(std::is_same_v<NonConstIdx, int>, "Unsupported combination of integral types");

  template <template <typename, typename...> typename Storage>
  explicit CudaBsrMatrix(
      mochi::BlockSparseMatrix<Scalar, kBlockSize, Index, PtrIndex, Storage> const& A)
      : _mb(A.BlockRows()),
        _nb(A.BlockCols()),
        _nnzb(A.NumNonZeroBlocks()),
        _bsrRowPtr(AsConstView(A.Pointers())),
        _bsrColIdx(AsConstView(A.Indices())),
        _bsrValues(A.NumNonZeros()) {
    //--- Convert the CSR storage of values into CUDA BSR-format
    CudaVector<NonConstScalar> csrVal(AsConstView(A.Values()));
    details::ConvertCsrToBsr<NonConstScalar, kBlockStorage>(
        kBlockSize, _mb, A.MaxNnzPerRow(), _bsrRowPtr.data(), csrVal.data(), _bsrValues.data());
    _bsrDescr = mochi::details::CreateCudaBsrMatDescriptor();
  }

  ~CudaBsrMatrix() = default;

  /**
   * @brief Apply the block sparse matrix to a column vector (or a set of column vectors).
   * @param[in] x  Input column vector(s) on the device
   * @param[out] y  Output column vector(s) on the device
   *
   * @note
   * The implementation uses the static CuSparse handle
   * without modifying its stream.
   *
   */
  template <typename Input, typename Output>
  void Apply(Input const& X, Output&& AX) const {
    MOCHI_ASSERT_VERBOSE(
        X.Cols() == AX.Cols(), "Incompatible number of columns (X %d, AX %d)", X.Cols(), AX.Cols());
    MOCHI_ASSERT_VERBOSE(
        X.Rows() == kBlockSize * _nb,
        "Incompatible number of rows for X (%d) and columns for A (%d)",
        X.Rows(),
        kBlockSize * _nb);
    MOCHI_ASSERT_VERBOSE(
        AX.Rows() == kBlockSize * _mb,
        "Incompatible number of rows (AX %d, A %d)",
        AX.Rows(),
        kBlockSize * _mb);
    constexpr krylov::Direction kDirX =
        mochi::krylov::details::MatTraits<std::remove_const_t<Input>>::kMajorDir;
    constexpr krylov::Direction kDirAX = mochi::krylov::details::MatTraits<Output>::kMajorDir;
    static_assert(
        kDirX == krylov::Direction::ColMajor,
        "Only column major for input is implemented with CudaBsrMatrix");
    static_assert(
        kDirAX == krylov::Direction::ColMajor,
        "Only column major for output is supported with CudaBsrMatrix");
    static_assert(
        std::is_same_v<typename krylov::details::MatTraits<Input>::Scalar const, Scalar const>,
        "Incompatible input scalar type");
    static_assert(
        std::is_same_v<typename krylov::details::MatTraits<Output>::Scalar const, Scalar const>,
        "Incompatible output scalar type");
    if (X.Cols() == 1) {
      mochi::details::MultiplyBsrVec<NonConstScalar, kBlockStorage>(
          _bsrDescr.get(),
          false,
          static_cast<int>(_mb),
          static_cast<int>(_nb),
          static_cast<int>(_nnzb),
          kBlockSize,
          _bsrRowPtr.data(),
          _bsrColIdx.data(),
          _bsrValues.data(),
          X.data(),
          AX.data());
    } else {
      mochi::details::MultiplyBsrMat<NonConstScalar, kBlockStorage>(
          _bsrDescr.get(),
          false,
          false,
          static_cast<int>(_mb),
          static_cast<int>(_nb),
          static_cast<int>(_nnzb),
          kBlockSize,
          _bsrRowPtr.data(),
          _bsrColIdx.data(),
          _bsrValues.data(),
          X.data(),
          X.Cols(),
          X.LeadDim(),
          AX.Data(),
          AX.LeadDim());
    }
  }

  template <typename Input, typename Output>
  void TransposeApply(Input const& X, [[maybe_unused]] Output&& AX) const {
    MOCHI_ASSERT_VERBOSE(
        X.Cols() == AX.Cols(), "Incompatible number of columns (X %d, AX %d)", X.Cols(), AX.Cols());
    MOCHI_ASSERT_VERBOSE(
        X.Rows() == kBlockSize * _mb,
        "Incompatible number of rows (X %d, A %d)",
        X.Rows(),
        kBlockSize * _mb);
    MOCHI_ASSERT_VERBOSE(
        AX.Rows() == kBlockSize * _nb,
        "Incompatible number of rows for AX (%d) and columns for A (%d)",
        AX.Rows(),
        kBlockSize * _nb);
    constexpr krylov::Direction kDirX =
        mochi::krylov::details::MatTraits<std::remove_const_t<Input>>::kMajorDir;
    constexpr krylov::Direction kDirAX = mochi::krylov::details::MatTraits<Output>::kMajorDir;
    static_assert(kDirX == kDirAX, "Input and output must have same orientation");
    static_assert(
        kDirX == krylov::Direction::ColMajor, "Only column major is supported with CudaBsrMatrix");
    static_assert(
        std::is_same_v<typename krylov::details::MatTraits<Input>::Scalar const, Scalar const>,
        "Incompatible input scalar type");
    static_assert(
        std::is_same_v<typename krylov::details::MatTraits<Output>::Scalar const, Scalar const>,
        "Incompatible output scalar type");
    // Cuda routines do not support A^T * X when A is stored in BSR format
    MOCHI_ASSERT(X.Cols() == 0, "Matrix-Transpose times vector is not supported for CudaBsrMatrix");
  }

  Index Rows() const {
    return static_cast<Index>(_mb * kBlockSize);
  }

  Index Cols() const {
    return static_cast<Index>(_nb * kBlockSize);
  }

  Index BlockRows() const {
    return static_cast<Index>(_mb);
  }

  Index BlockCols() const {
    return static_cast<Index>(_nb);
  }

  auto& Values() const {
    return _bsrValues;
  }

  auto& Indices() const {
    return _bsrColIdx;
  }

  auto& Pointers() const {
    return _bsrRowPtr;
  }

  auto* Descriptor() const {
    return _bsrDescr.get();
  }

  auto Norm() const {
    return _bsrValues.Norm();
  }

 protected:
  //
  // Use the CUDA parameters for representing a matrix m x n where:
  // m (the number of rows) is equal to kBlockSize * _mb
  // n (the number of columns) is equal to kBlockSize * _nb

  int64_t _mb = 0;
  int64_t _nb = 0;

  /// @brief Number of non-zero blocks
  /// (Each block is of dimension kBlockSize x kBlockSize)
  int64_t _nnzb = 0;

  /// @brief Smart pointer to a Cuda-specific structure for representing the matrix in BSR format
  std::unique_ptr<mochi::details::CudaBsrMatDescr, mochi::details::ReleaseCudaBsrMatDescr>
      _bsrDescr;

  /// TODO The class CudaVector is too advanced for this usage.
  /// We could use a basic container like std::span
  CudaVector<NonConstPtrIdx, krylov::kDynamic> _bsrRowPtr;
  CudaVector<NonConstIdx, krylov::kDynamic> _bsrColIdx;
  CudaVector<NonConstScalar, krylov::kDynamic> _bsrValues;
};

} // namespace mochi::krylov

namespace mochi::details {
template <typename Scalar, int kBlockSize, typename Index, typename PtrIndex>
constexpr bool IsBlockSparseMatrixDef<krylov::CudaBsrMatrix<Scalar, kBlockSize, Index, PtrIndex>> =
    true;

template <typename Scalar, int kBlockSize, typename Index, typename PtrIndex>
constexpr bool IsCudaDef<krylov::CudaBsrMatrix<Scalar, kBlockSize, Index, PtrIndex>> = true;
} // namespace mochi::details
