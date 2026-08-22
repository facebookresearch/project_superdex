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
#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_sparse_utils.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace mochi::krylov {

/// @brief Class for a compressed-storage-row on the GPU device.
/// The implementation uses the CUDA language (with CUSparse library).
///
/// @tparam Scalar_  Template type for the scalar values
/// @tparam Index  Template type for the column indices
/// @tparam PtrIndex  Template type for the pointer in the CSR storage
///
/// @note
/// The current implementation supports Scalar_ = float, Index = PtrIndex = int32_t or int64_t
///
/// @note
/// Requires building with MOCHI_USE_CUDA=1.
///
template <typename Scalar_, typename Index = int, typename PtrIndex = int>
class CudaCsrMatrix {
 public:
  using Scalar = Scalar_;
  using NonConstIdx = std::remove_const_t<Index>;
  using NonConstPtrIdx = std::remove_const_t<PtrIndex>;
  using NonConstScalar = std::remove_const_t<Scalar>;

  static_assert(
      (MOCHI_USE_CUDA) || std::is_same_v<Scalar, void>,
      "CudaCsrMatrix requires building with CUDA. To enable CUDA, add the CUDA dependencies to your build configuration and define MOCHI_USE_CUDA=1");

  /// @note
  /// CuSparse create these limitations.
  ///
  static_assert(
      std::is_same_v<NonConstIdx, int32_t> || std::is_same_v<NonConstIdx, int64_t>,
      "Integer type for row index not supported yet.");
  static_assert(
      std::is_same_v<NonConstPtrIdx, int32_t> || std::is_same_v<NonConstPtrIdx, int64_t>,
      "Integer type for pointer index not supported yet.");
  /// @note
  /// The cuSparse documentation does not seem to limit the choice of integral types.
  /// However at runtime, an error is thrown when the integral types do not match
  ///
  static_assert(
      std::is_same_v<Index const, PtrIndex const>,
      "Unsupported combination of integral types");

  template <template <typename, typename...> typename Storage>
  explicit CudaCsrMatrix(mochi::SparseMatrix<Scalar, Index, PtrIndex, Storage> const& A)
      : _nRow(A.Rows()),
        _nCol(A.Cols()),
        _nnz(A.NumNonZeros()),
        _rowPtr(AsConstView(A.Pointers())),
        _colIdx(AsConstView(A.Indices())),
        _values(AsConstView(A.Values())) {
    //--- Create CSR descriptors
    //--- Per CUDA documentation, we associate one buffer per descriptor.
    //--- So we create a descriptor to compute A * x and one to compute A^T * x
    _csrDescr = mochi::details::InitializeCsr(
        _nRow, _nCol, _nnz, _rowPtr.Data(), _colIdx.Data(), _values.Data());
    _csrDescrTranspose = mochi::details::InitializeCsr(
        _nRow, _nCol, _nnz, _rowPtr.Data(), _colIdx.Data(), _values.Data());
    //--- Create buffer for SpMV (for A * x)
    /// Placing the allocation in the constructor removed an `if` statement in the `Apply` function
    /// and simplified the recording of CUDA graphs.
    /// The graph recording can not assume that the buffer size is not zero.
    /// Introducing the buffer "sizing" in `Apply` would potentially call `cudaFree` or `cudaMalloc`
    /// during the graph recording.
    /// Allocation or deallocation of memory while the CUDA graph is "being" recorded
    /// would need some specific care.
    /// TODO Review whether the allocation should be moved
    CudaVector<NonConstScalar> x(_nCol), y(_nRow);
    mochi::details::SizeBufferSpMV(
        false,
        _csrDescr.get(),
        x.data(),
        x.Rows(),
        y.data(),
        y.Rows(),
        _bufferSizeSpMv,
        &_bufferSpMv);
    //--- Create buffer for SpMV (for A^T * x)
    /// Placing the allocation in the constructor removed an `if` statement in the `TransposeApply`
    /// function and simplified the recording of CUDA graphs. The graph recording can not assume
    /// that the buffer size is not zero. Introducing the buffer "sizing" in `TransposeApply` would
    /// potentially call `cudaFree` or `cudaMalloc` during the graph recording. Allocation or
    /// deallocation of memory while the CUDA graph is "being" recorded would need some specific
    /// care.
    /// TODO Review whether the allocation should be moved
    mochi::details::SizeBufferSpMV(
        true,
        _csrDescrTranspose.get(),
        y.data(),
        y.Rows(),
        x.data(),
        x.Rows(),
        _bufferSizeSpMtv,
        &_bufferSpMtv);
  }

  ~CudaCsrMatrix() {
    mochi::details::CudaFree(_bufferSpMv);
    mochi::details::CudaFree(_bufferSpMtv);
    mochi::details::CudaFree(_bufferSpMM);
    mochi::details::CudaFree(_bufferSpMtM);
  }

  /**
   * @brief Apply the sparse matrix to a column vector.
   * @param[in] x  Input column vector on the device
   * @param[out] y  Output column vector on the device
   *
   * @note
   * The implementation uses the static CuSparse handle
   * without modifying its stream.
   *
   */
  template <int kRow, krylov::Ownership kOwner, int kLead>
    requires(krylov::IsCuda(kOwner))
  void Apply(
      Matrix<NonConstScalar, kRow, 1, krylov::Direction::ColMajor, kOwner, kLead> const& x,
      Matrix<NonConstScalar, kRow, 1, krylov::Direction::ColMajor, kOwner, kLead>& Ax) const {
    static_assert((kRow == krylov::kDynamic) || (kRow > 0));
    MOCHI_ASSERT_VERBOSE(
        x.Rows() == _nCol,
        "Incompatible number of rows for X (%d) and columns for A (%d)",
        x.Rows(),
        _nCol);
    MOCHI_ASSERT_VERBOSE(
        Ax.Rows() == _nRow, "Incompatible number of rows (AX %d, A %d)", Ax.Rows(), _nRow);
    mochi::details::MultiplyCsrVec(
        false, _csrDescr.get(), _bufferSpMv, x.data(), x.Rows(), Ax.data(), Ax.Rows());
  }

  /**
   * @brief Apply the sparse matrix to a vector (or a set of vectors).
   * @param[in] x  Input vector(s) on the device
   * @param[out] y  Output vector(s) on the device
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
        X.Rows() == _nCol,
        "Incompatible number of rows for X (%d) and columns for A (%d)",
        X.Rows(),
        _nCol);
    MOCHI_ASSERT_VERBOSE(
        AX.Rows() == _nRow, "Incompatible number of rows (AX %d, A %d)", AX.Rows(), _nRow);
    constexpr krylov::Direction kDirX =
        mochi::krylov::details::MatTraits<std::remove_const_t<Input>>::kMajorDir;
    static_assert(
        std::is_same_v<typename krylov::details::MatTraits<Input>::Scalar const, Scalar const>,
        "Incompatible input scalar type");
    static_assert(
        std::is_same_v<typename krylov::details::MatTraits<Output>::Scalar const, Scalar const>,
        "Incompatible output scalar type");
    if ((X.Cols() == 1) && (kDirX == krylov::Direction::ColMajor)) {
      mochi::details::MultiplyCsrVec(
          false, _csrDescr.get(), _bufferSpMv, X.data(), X.Rows(), AX.data(), AX.Rows());
    } else {
      mochi::details::SizeBufferSpMM(
          false,
          false,
          _csrDescr.get(),
          X.Data(),
          X.Rows(),
          X.Cols(),
          X.LeadDim(),
          X.kIsRowMajor,
          AX.Data(),
          AX.Rows(),
          AX.Cols(),
          AX.LeadDim(),
          AX.kIsRowMajor,
          _bufferSizeSpMM,
          &_bufferSpMM);
      mochi::details::MultiplyCsrMat(
          false,
          false,
          _csrDescr.get(),
          _bufferSpMM,
          X.Data(),
          X.Rows(),
          X.Cols(),
          X.LeadDim(),
          X.kIsRowMajor,
          AX.Data(),
          AX.Rows(),
          AX.Cols(),
          AX.LeadDim(),
          AX.kIsRowMajor);
    }
  }

  template <int kRow, krylov::Ownership kOwner, int kLead>
    requires(krylov::IsCuda(kOwner))
  void TransposeApply(
      Matrix<NonConstScalar, kRow, 1, krylov::Direction::ColMajor, kOwner, kLead> const& x,
      Matrix<NonConstScalar, kRow, 1, krylov::Direction::ColMajor, kOwner, kLead>& Atx) const {
    static_assert((kRow == krylov::kDynamic) || (kRow > 0));
    MOCHI_ASSERT_VERBOSE(
        x.Rows() == _nRow,
        "Incompatible number of rows for X (%d) and columns for A^T (%d)",
        x.Rows(),
        _nRow);
    MOCHI_ASSERT_VERBOSE(
        Atx.Rows() == _nCol, "Incompatible number of rows (AtX %d, A^T %d)", Atx.Rows(), _nCol);
    mochi::details::MultiplyCsrVec(
        true, _csrDescrTranspose.get(), _bufferSpMtv, x.data(), x.Rows(), Atx.data(), Atx.Rows());
  }

  template <typename Input, typename Output>
  void TransposeApply(Input const& X, Output&& AtX) const {
    MOCHI_ASSERT_VERBOSE(
        X.Cols() == AtX.Cols(),
        "Incompatible number of columns (X %d, AtX %d)",
        X.Cols(),
        AtX.Cols());
    MOCHI_ASSERT_VERBOSE(
        X.Rows() == _nRow, "Incompatible number of rows (X %d, A %d)", X.Rows(), _nRow);
    MOCHI_ASSERT_VERBOSE(
        AtX.Rows() == _nCol,
        "Incompatible number of rows for AtX (%d) and columns for A (%d)",
        AtX.Rows(),
        _nCol);
    constexpr krylov::Direction kDirX =
        mochi::krylov::details::MatTraits<std::remove_const_t<Input>>::kMajorDir;
    constexpr krylov::Direction kDirAtX = mochi::krylov::details::MatTraits<Output>::kMajorDir;
    static_assert(kDirX == kDirAtX, "Input and output must have same orientation");
    static_assert(
        kDirX == mochi::krylov::Direction::ColMajor, "Only column-major orientation is allowed");
    static_assert(
        std::is_same_v<typename krylov::details::MatTraits<Input>::Scalar const, Scalar const>,
        "Incompatible input scalar type");
    static_assert(
        std::is_same_v<typename krylov::details::MatTraits<Output>::Scalar const, Scalar const>,
        "Incompatible output scalar type");
    if ((X.Cols() == 1) && (kDirX == krylov::Direction::ColMajor)) {
      mochi::details::MultiplyCsrVec(
          true, _csrDescrTranspose.get(), _bufferSpMtv, X.data(), X.Rows(), AtX.data(), AtX.Rows());
    } else {
      mochi::details::SizeBufferSpMM(
          true,
          false,
          _csrDescrTranspose.get(),
          X.Data(),
          X.Rows(),
          X.Cols(),
          X.LeadDim(),
          X.kIsRowMajor,
          AtX.Data(),
          AtX.Rows(),
          AtX.Cols(),
          AtX.LeadDim(),
          AtX.kIsRowMajor,
          _bufferSizeSpMtM,
          &_bufferSpMtM);
      mochi::details::MultiplyCsrMat(
          true,
          false,
          _csrDescrTranspose.get(),
          _bufferSpMtM,
          X.Data(),
          X.Rows(),
          X.Cols(),
          X.LeadDim(),
          X.kIsRowMajor,
          AtX.Data(),
          AtX.Rows(),
          AtX.Cols(),
          AtX.LeadDim(),
          AtX.kIsRowMajor);
    }
  }

  Index Rows() const {
    return static_cast<Index>(_nRow);
  }

  Index Cols() const {
    return static_cast<Index>(_nCol);
  }

  auto& Values() const {
    return _values;
  }

  auto& Indices() const {
    return _colIdx;
  }

  auto& Pointers() const {
    return _rowPtr;
  }

  auto* Descriptor() const {
    return _csrDescr.get();
  }

  auto Norm() const {
    return _values.Norm();
  }

 protected:
  int64_t _nRow = 0;
  int64_t _nCol = 0;
  int64_t _nnz = 0;

  /// @brief Pointer to Cuda-specific structure to describe the matrix
  std::unique_ptr<mochi::details::CudaConstSparseMatDescr, mochi::details::ReleaseSparseMatDescr>
      _csrDescr;

  /// @brief Pointer to Cuda-specific structure to apply the transpose
  std::unique_ptr<mochi::details::CudaConstSparseMatDescr, mochi::details::ReleaseSparseMatDescr>
      _csrDescrTranspose;

  CudaVector<NonConstPtrIdx> _rowPtr;
  CudaVector<NonConstIdx> _colIdx;
  CudaVector<NonConstScalar> _values;

  /// @brief Storage to a workspace buffer for Cuda SpMv routine
  mutable void* _bufferSpMv = nullptr;
  mutable std::size_t _bufferSizeSpMv = 0;

  /// @brief Storage to a workspace buffer for Cuda SpMv routine (transpose operation)
  mutable void* _bufferSpMtv = nullptr;
  mutable std::size_t _bufferSizeSpMtv = 0;

  /// @brief Storage to a workspace buffer for Cuda SpMM routine
  mutable void* _bufferSpMM = nullptr;
  mutable std::size_t _bufferSizeSpMM = 0;

  /// @brief Storage to a workspace buffer for Cuda SpMM routine (transpose operation)
  mutable void* _bufferSpMtM = nullptr;
  mutable std::size_t _bufferSizeSpMtM = 0;
};

} // namespace mochi::krylov

namespace mochi::details {
template <typename Scalar, typename Index, typename PtrIndex>
constexpr bool IsSparseMatrixDef<mochi::krylov::CudaCsrMatrix<Scalar, Index, PtrIndex>> = true;

template <typename Scalar, typename Index, typename PtrIndex>
constexpr bool IsCudaDef<krylov::CudaCsrMatrix<Scalar, Index, PtrIndex>> = true;
} // namespace mochi::details
