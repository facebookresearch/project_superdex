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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>

#include <algorithm>
#include <concepts>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

namespace mochi::krylov {

template <typename Scalar, typename MatrixType>
struct SSORPrec final : Preconditioner<Scalar> {
  static_assert(!std::is_const_v<Scalar>, "Implementation assumes Scalar is non-const.");
  static constexpr auto kType = PreconditionerType::SSOR;

  /// @brief Constructor
  /// @param[in] A Input square matrix
  /// @param[in] omega Relaxation factor in (0, 2)
  explicit SSORPrec(MatrixType const& A, Scalar omega = Scalar(1));

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors)
  /// @param[in] xin Input column vector(s)
  /// @param[out] yout Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& xin, Output&& yout) const;

  /// @brief Apply the preconditioner to a column vector.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<Scalar> Px) const override {
    operator()(x, Px);
  }

  /// @brief Concurrent application of the preconditioner to a column vector by a pool of workers.
  /// @note The calling worker applies an SSOR approximation of the diagonal block
  /// [rowBegin, endRow) x [rowBegin, endRow). That is, 'ConcurrentSolve' corresponds to a domain
  /// decomposition SSOR and is in general different from the original SSOR preconditioner.
  /// @note The original SSOR preconditioner is recovered with one worker.
  void ConcurrentSolve(
      ColumnVectorView<Scalar const> x,
      ColumnVectorView<Scalar> Px,
      ParallelWorkerInfo const& data) const override;

  void Update(MatrixType const& A);

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

 protected:
  std::reference_wrapper<MatrixType const> _inputA;
  std::vector<int> _lowerEnd = {};
  std::vector<Scalar> _inverseDiagBlocks = {};
  Scalar _omega_s = Scalar(1);
  Scalar _ratio_s = Scalar(1);
  Matrix<Scalar> _workSpace = {};
};

} // namespace mochi::krylov

//
//--- Implementation of functions
//

namespace mochi::krylov::details {

template <
    typename ScalarA,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
void MakePositionArray(
    Matrix<ScalarA, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim> const&
    /*A*/,
    std::vector<int>& /*lowerEnd*/) {}

/// @brief Extract indices for lower and upper triangular part per row for BlockSparseMatrix
///
/// @tparam InputScalar
/// @tparam CRIdx
/// @tparam Ptr
/// @tparam AStorage
/// @param A Block sparse matrix
/// @param lowerEnd First index for a column not in the lower triangular part
///
/// @note This routine assumes (without checking) that the column indices
/// are sorted on each row.
template <
    typename InputScalar,
    int kBlockSizeA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename AStorage>
void MakePositionArray(
    BlockSparseMatrix<InputScalar, kBlockSizeA, CRIdx, Ptr, AStorage> const& A,
    std::vector<int>& lowerEnd) {
  MOCHI_ASSERT_VERBOSE(isize(lowerEnd) >= A.BlockRows(), "Insufficient span size");
  using NonConstIdx = std::remove_const_t<CRIdx>;
  for (NonConstIdx ir = 0; ir < A.BlockRows(); ++ir) {
    auto const colIdx = A.Indices(ir);
    auto const* ptr = std::lower_bound(colIdx.begin(), colIdx.end(), ir);
    lowerEnd[ir] = static_cast<int>(ptr - colIdx.begin());
    MOCHI_ASSERT_VERBOSE(
        colIdx[lowerEnd[ir]] == ir, "Diagonal entry seems to be missing on block row %d", ir);
  }
}

/// @brief Extract indices for lower and upper triangular part per row
///
/// @tparam InputScalar
/// @tparam CRIdx
/// @tparam Ptr
/// @tparam AStorage
/// @param A Sparse matrix
/// @param lowerEnd First index for a column not in the lower triangular part
///
/// @note This routine assumes (without checking) that the column indices
/// are sorted on each row.
template <
    typename InputScalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename AStorage>
void MakePositionArray(
    SparseMatrix<InputScalar, CRIdx, Ptr, AStorage> const& A,
    std::vector<int>& lowerEnd) {
  MOCHI_ASSERT_VERBOSE(isize(lowerEnd) >= A.Rows(), "Insufficient span size");
  using NonConstIdx = std::remove_const_t<CRIdx>;
  for (NonConstIdx row = 0; row < A.Rows(); ++row) {
    auto colIdx = A.Indices(row);
    auto const* ptr = std::lower_bound(colIdx.begin(), colIdx.end(), row);
    lowerEnd[row] = static_cast<int>(ptr - colIdx.begin());
    MOCHI_ASSERT_VERBOSE(colIdx[lowerEnd[row]] == row, "Diagonal entry seems to be missing.")
  }
}

/// @brief Apply the SSOR operator on the diagonal block [rBegin, rEnd[ x [rBegin, rEnd[
/// for a BlockSparseMatrix input
///
/// @note We assume that the matrix A is square.
/// @note We assume that the diagonal entry for the BlockSparseMatrix is non-zero.
/// @note We assume that the block-row entries in the BlockSparseMatrix are sorted
/// in increasing block-column index. No check is performed.
/// @note We assume that Input and Output are (derived) from Matrix
/// @note No assumption is made on the orientation of `x` or `y`
template <
    typename ScalarA,
    int kBlockSizeA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage,
    typename Input,
    typename Output>
void ApplySsorOperator(
    BlockSparseMatrix<ScalarA, kBlockSizeA, CRIdx, Ptr, Storage> const& A,
    std::vector<std::remove_const_t<ScalarA>> const& invDiagBlock,
    std::vector<int> const& lowerEnd,
    std::remove_const_t<ScalarA>* t,
    std::remove_const_t<ScalarA> omega_s,
    std::remove_const_t<ScalarA> ratio_s,
    Input const& x,
    Output&& y,
    int rBegin,
    int rEnd) {
  MOCHI_ASSERT_VERBOSE(rBegin % kBlockSizeA == 0, "Row beginning should be aligned with blocks");
  MOCHI_ASSERT_VERBOSE(rEnd % kBlockSizeA == 0, "Row end should be aligned with blocks");
  using Scalar = std::remove_const_t<ScalarA>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  ColumnVectorView<Scalar> tview(t, A.Rows());
  //--- Create storage space for column-major product
  [[maybe_unused]] ColumnVector<Scalar> xTmp;
  [[maybe_unused]] auto const xTmpRequiredSize = static_cast<NonConstIdx>(
      mochi::details::RowMultiplier<Scalar, kBlockSizeA>::GetWorkspaceSize(A.MaxNnzPerRow()));
  [[maybe_unused]] ColumnVector<Scalar, kBlockSizeA> Ly;
  [[maybe_unused]] auto aLy = mochi::details::GetAccessor(Ly);
  //
  auto bBegin = rBegin / kBlockSizeA;
  auto bEnd = rEnd / kBlockSizeA;
  //
  for (int jc = 0; jc < x.Cols(); ++jc) {
    tview.MiddleRows(rBegin, rEnd - rBegin) = x.Col(jc).MiddleRows(rBegin, rEnd - rBegin);
    auto yjc = y.Col(jc);
    [[maybe_unused]] auto aY = mochi::details::GetAccessor(yjc);
    for (NonConstIdx irb = bBegin; irb < bEnd; ++irb) {
      auto colIdx = A.Indices(irb);
      auto values = A.Values(irb);
      auto ti = tview.template MiddleRows<kBlockSizeA>(irb * kBlockSizeA, kBlockSizeA);
      // (D + L)^{-1} y = t
      // Do first t([i, ..., i + kBlockSizeA - 1]) <- t([i, ..., i + kBlockSizeA - 1]) - L * y
      size_t shift = 0;
      if (rBegin > 0) {
        auto const* ptr = std::lower_bound(colIdx.begin(), colIdx.end(), bBegin);
        shift = std::distance(colIdx.data(), ptr);
      }
      if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::RowMultiplier<Scalar, kBlockSizeA>::ApplyToColVector(
            colIdx.data() + shift,
            values.data() + shift * kBlockSizeA,
            lowerEnd[irb] - shift,
            values.LeadDim(),
            aY,
            aLy,
            /*br*/ 0,
            /*c*/ 0,
            xTmp,
            xTmpRequiredSize);
        ti -= Ly;
      } else {
        static_assert(mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
        for (size_t kk = shift; kk < lowerEnd[irb]; ++kk) {
          ti -= values[kk] *
              yjc.template MiddleRows<kBlockSizeA>(colIdx[kk] * kBlockSizeA, kBlockSizeA);
        }
      }
      auto const& D = values[lowerEnd[irb]];
      auto yi = yjc.template MiddleRows<kBlockSizeA>(irb * kBlockSizeA, kBlockSizeA);
      for (int ii = 0; ii < kBlockSizeA; ++ii) {
        for (int jj = 0; jj < ii; ++jj) {
          ti(ii, 0) -= D(ii, jj) * yi(jj, 0);
        }
        yi(ii, 0) = invDiagBlock[ii + irb * kBlockSizeA] * ti(ii, 0);
      }
    }
    if (Abs(omega_s * ratio_s - Scalar(1)) > 2 * std::numeric_limits<Scalar>::epsilon()) {
      tview.MiddleRows(rBegin, rEnd - rBegin) *= (omega_s * ratio_s);
    }
    for (int irb = bEnd - 1; irb >= bBegin; --irb) {
      auto colIdx = A.Indices(irb);
      auto values = A.Values(irb);
      auto ti = tview.template MiddleRows<kBlockSizeA>(irb * kBlockSizeA, kBlockSizeA);
      auto shift = static_cast<size_t>(lowerEnd[irb] + 1);
      auto len = colIdx.size() - shift;
      if (rEnd < A.Rows()) {
        auto const* ptr = std::lower_bound(colIdx.begin() + shift, colIdx.end(), bEnd);
        len = std::distance(colIdx.data() + shift, ptr);
      }
      if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::RowMultiplier<Scalar, kBlockSizeA>::ApplyToColVector(
            colIdx.data() + shift,
            values.data() + shift * kBlockSizeA,
            len,
            values.LeadDim(),
            aY,
            aLy,
            /*br*/ 0,
            /*c*/ 0,
            xTmp,
            xTmpRequiredSize);
        ti -= Ly;
      } else {
        static_assert(mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
        for (size_t kk = shift; kk < shift + len; ++kk) {
          ti -= values[kk] *
              yjc.template MiddleRows<kBlockSizeA>(colIdx[kk] * kBlockSizeA, kBlockSizeA);
        }
      }
      auto yi = yjc.template MiddleRows<kBlockSizeA>(irb * kBlockSizeA, kBlockSizeA);
      auto const& D = values[lowerEnd[irb]];
      for (int ii = kBlockSizeA - 1; ii >= 0; --ii) {
        for (int jj = ii + 1; jj < kBlockSizeA; ++jj) {
          ti(ii, 0) -= D(ii, jj) * yi(jj, 0);
        }
        yi(ii, 0) = invDiagBlock[ii + irb * kBlockSizeA] * ti(ii, 0);
      }
    } //
  }
}

/// @brief Apply the SSOR operator on the diagonal block [rBegin, rEnd[ x [rBegin, rEnd[
/// for a SparseMatrix input
///
/// @note We assume that the matrix A is square.
/// @note We assume that the diagonal entry for the SparseMatrix is non-zero.
/// @note We assume that the row entries in the SparseMatrix are sorted
/// in increasing column index. No check is performed.
/// @note We assume that Input and Output are (derived) from Matrix
/// @note We assume that `x` and `y` have the same orientation.
template <
    typename ScalarA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage,
    typename Input,
    typename Output>
void ApplySsorOperator(
    SparseMatrix<ScalarA, CRIdx, Ptr, Storage> const& A,
    std::vector<std::remove_const_t<ScalarA>> const& invDiagBlock,
    std::vector<int> const& lowerEnd,
    std::remove_const_t<ScalarA>* t,
    std::remove_const_t<ScalarA> omega_s,
    std::remove_const_t<ScalarA> ratio_s,
    Input const& x,
    Output&& y,
    int rBegin,
    int rEnd) {
  using NonConstIdx = std::remove_const_t<CRIdx>;
  NonConstIdx nRows = A.Rows();
  //
  static_assert(
      MatTraits<Input>::kMajorDir == MatTraits<Output>::kMajorDir,
      "Mixed orientation not implemented yet"); // Please update unit tests if mixed orientations
                                                // are supported in the future.
  //
  using NonConstScalar = std::remove_const_t<ScalarA>;
  auto aPtr = A.Pointers();
  auto aVal = A.Values();
  auto aCol = A.Indices();
  ColumnVectorView<NonConstScalar> tview(t, nRows);
  for (int jc = 0; jc < x.Cols(); ++jc) {
    tview.MiddleRows(rBegin, rEnd - rBegin) = x.Col(jc).MiddleRows(rBegin, rEnd - rBegin);
    auto yjc = y.Col(jc);
    // Define A = D + L + U
    // Forward: y <-- (D / w + L )^{-1} y
    for (NonConstIdx row = rBegin; row < rEnd; ++row) {
      size_t shift = 0;
      if (rBegin > 0) {
        auto const* start = aCol.data() + aPtr[row];
        auto const* ptr = std::lower_bound(start, start + lowerEnd[row], rBegin);
        shift = std::distance(aCol.data() + aPtr[row], ptr);
      }
      Span<CRIdx const> spanCol{aCol.data() + aPtr[row] + shift, size_t(lowerEnd[row]) - shift};
      Span<ScalarA const> spanVal{aVal.data() + aPtr[row] + shift, spanCol.size()};
      // We assume that `x` and `y` have the same orientation.
      // Call appropriate kernel depending on the orientation of `x`
      NonConstScalar sum = 0;
      if constexpr (MatTraits<Input>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::AddRowSparseTimesColumnVector<NonConstScalar, CRIdx>(
            spanCol, spanVal, yjc.Data(), sum);
      } else {
        mochi::details::AddRowSparseTimesVector<NonConstScalar, CRIdx>(spanCol, spanVal, yjc, sum);
      }
      t[row] -= sum;
      yjc(row, 0) = invDiagBlock[row] * t[row];
    }
    if (Abs(omega_s * ratio_s - 1) > 2 * std::numeric_limits<NonConstScalar>::epsilon()) {
      auto prod = omega_s * ratio_s;
      tview.MiddleRows(rBegin, rEnd - rBegin) *= prod;
    }
    // Backward: y <-- (D / w + U )^{-1} y
    for (NonConstIdx row = rEnd - 1; row >= rBegin; --row) {
      auto const upperPos = lowerEnd[row] + 1;
      auto shift = static_cast<size_t>(aPtr[row] + upperPos);
      auto len = static_cast<size_t>(aPtr[row + 1] - aPtr[row] - upperPos);
      if (rEnd < A.Rows()) {
        auto const* start = aCol.data() + shift;
        auto const* ptr = std::lower_bound(start, start + len, rEnd);
        len = std::distance(start, ptr);
      }
      Span<CRIdx const> spanCol{aCol.data() + shift, len};
      Span<ScalarA const> spanVal{aVal.data() + shift, spanCol.size()};
      // We assume that `x` and `y` have the same orientation.
      // Call appropriate kernel depending on the orientation of `x`
      NonConstScalar sum = 0;
      if constexpr (MatTraits<Input>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::AddRowSparseTimesColumnVector<NonConstScalar, CRIdx>(
            spanCol, spanVal, yjc.Data(), sum);
      } else {
        mochi::details::AddRowSparseTimesVector<NonConstScalar, CRIdx>(spanCol, spanVal, yjc, sum);
      }
      yjc(row, 0) = invDiagBlock[row] * (t[row] - sum);
    } // for (NonConstIdx row = rEnd - 1; row >= rBegin; --row)
  } // for (int jc = 0; jc < x.Cols(); ++jc)
}

/// @brief Apply the SSOR operator on the diagonal block [rBegin, rEnd[ x [rBegin, rEnd[
/// for a Matrix input
///
/// @note We assume that the matrix A is square.
/// @note We assume that Input and Output are (derived) from Matrix
/// @note The parameters Input and Output have no constraint on the orientation
/// @note No assumption is made on the orientation of `x` or `y`
template <
    typename ScalarA,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim,
    typename Input,
    typename Output>
void ApplySsorOperator(
    Matrix<ScalarA, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim> const&
        A,
    std::vector<std::remove_const_t<ScalarA>> const& invDiagBlock,
    [[maybe_unused]] std::vector<int> const& lowerEnd,
    std::remove_const_t<ScalarA>* t,
    std::remove_const_t<ScalarA> omega_s,
    std::remove_const_t<ScalarA> ratio_s,
    Input const& x,
    Output&& y,
    int rBegin,
    int rEnd) {
  using Scalar = std::remove_const_t<ScalarA>;
  auto nRows = A.Rows();
  ColumnVectorView<Scalar> tview(t, nRows);
  for (int jc = 0; jc < x.Cols(); ++jc) {
    tview.MiddleRows(rBegin, rEnd - rBegin) = x.Col(jc).MiddleRows(rBegin, rEnd - rBegin);
    auto yjc = y.Col(jc);
    // Define A = D + L + U
    // Forward: y <-- (D / w + L )^{-1} y
    for (int row = rBegin; row < rEnd; ++row) {
      auto L = A.template Block<1, krylov::kDynamic>(row, rBegin, 1, row - rBegin);
      tview.Row(row) -= L * yjc.MiddleRows(rBegin, L.Cols());
      // y <-- (w/d_{ii}) t
      yjc(row, 0) = invDiagBlock[row] * tview(row, 0);
    }
    if (Abs(omega_s * ratio_s - 1) > 2 * std::numeric_limits<Scalar>::epsilon()) {
      auto prod = omega_s * ratio_s;
      tview.MiddleRows(rBegin, rEnd - rBegin) *= prod;
    }
    // Backward: y <-- (D / w + U )^{-1} y
    for (int row = rEnd - 1; row >= rBegin; --row) {
      auto U = A.template Block<1, krylov::kDynamic>(row, row + 1, 1, rEnd - 1 - row);
      // y <-- (w/d_{ii}) t
      tview.Row(row) -= U * yjc.MiddleRows(row + 1, U.Cols());
      yjc(row, 0) = invDiagBlock[row] * tview(row, 0);
    } // for (int row =  rEnd - 1; row >= rBegin; --row)
  } // for (int jc = 0; jc < x.Cols(); ++jc)
}

} // namespace mochi::krylov::details

namespace mochi::krylov {

template <typename Scalar, typename MatrixType>
SSORPrec<Scalar, MatrixType>::SSORPrec(MatrixType const& A, Scalar omega)
    : _inputA(A), _omega_s(omega), _ratio_s((2 - _omega_s) / _omega_s) {
  if constexpr (IsSparseMatrix<MatrixType> || IsBlockSparseMatrix<MatrixType>) {
    using CRIdx = decltype(A.Rows());
    static_assert(
        std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
        "SSORPrec requires signed matrix indices representable as int.");
  }
  MOCHI_ASSERT_VERBOSE((_omega_s > 0) && (_omega_s < 2), "Invalid parameter 'omega'.");
  Update(A);
}

template <typename Scalar, typename MatrixType>
void SSORPrec<Scalar, MatrixType>::Update(MatrixType const& A) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(mochi::details::HasSortedRowIndices(A), "Row entries are not sorted.");
  _inputA = A;
  //--- Get position arrays
  //--- Note that these arrays remain empty for a (dense) `Matrix` input
  if constexpr (IsBlockSparseMatrix<MatrixType>) {
    _lowerEnd.resize(A.BlockRows());
  } else if constexpr (IsSparseMatrix<MatrixType>) {
    _lowerEnd.resize(A.Rows());
  } else {
    static_assert(IsMatrix<MatrixType>, "Unexpected matrix type");
  }
  details::MakePositionArray(A, _lowerEnd);
  //--- We will store 'scalar' entries
  _inverseDiagBlocks.resize(A.Rows());
  ExtractDiagonal(A, MakeSpan(_inverseDiagBlocks));
  ArrayInverts(MakeSpan(_inverseDiagBlocks));
  //--- Scale the diagonal blocks if needed
  if (Abs(_omega_s - 1) > 2 * std::numeric_limits<Scalar>::epsilon()) {
    //--- Scale the inverse diagonal blocks with _omega_s
    ArrayMulEquals(MakeSpan(_inverseDiagBlocks), _omega_s);
  }
  //
  _workSpace.Resize(A.Rows(), 1);
}

template <typename Scalar, typename MatrixType>
template <typename Input, typename Output>
void SSORPrec<Scalar, MatrixType>::operator()(Input const& xin, Output&& yout) const {
  Preconditioner<Scalar>::ValidateInputOutput(_inputA.get().Rows(), xin, yout);
  //--- Convert to MatrixView
  MatrixView<
      typename details::MatTraits<Input>::Scalar const,
      details::MatTraits<Input>::kNumRows,
      details::MatTraits<Input>::kNumCols,
      details::MatTraits<Input>::kMajorDir,
      krylov::kDynamic>
      x(xin.data(), xin.Rows(), xin.Cols(), xin.LeadDim());
  MatrixView<
      std::remove_const_t<typename details::MatTraits<Output>::Scalar>,
      details::MatTraits<Output>::kNumRows,
      details::MatTraits<Output>::kNumCols,
      details::MatTraits<Output>::kMajorDir,
      krylov::kDynamic>
      y(yout.data(), yout.Rows(), yout.Cols(), yout.LeadDim());
  //
  details::ApplySsorOperator(
      _inputA.get(),
      _inverseDiagBlocks,
      _lowerEnd,
      const_cast<Scalar*>(_workSpace.data()),
      _omega_s,
      _ratio_s,
      x,
      y,
      0,
      int(_inputA.get().Rows()));
}

template <typename Scalar, typename MatrixType>
void SSORPrec<Scalar, MatrixType>::ConcurrentSolve(
    ColumnVectorView<Scalar const> x,
    ColumnVectorView<Scalar> Px,
    ParallelWorkerInfo const& data) const {
  Preconditioner<Scalar>::ValidateInputOutput(_inputA.get().Rows(), x, Px);
  details::ApplySsorOperator(
      _inputA.get(),
      _inverseDiagBlocks,
      _lowerEnd,
      const_cast<Scalar*>(_workSpace.data()),
      _omega_s,
      _ratio_s,
      x,
      Px,
      data.rBegin,
      data.rEnd);
}

} // namespace mochi::krylov
