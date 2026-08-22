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

/// @brief Class for implementing block SSOR
///
/// @tparam Scalar Scalar for the output
/// @tparam kPrecBlockSize Block size for the preconditioner
/// @tparam MatrixType Type of input matrix
///
/// @note It is recommended to use SSORPrec for a pointwise implementation (and not to use
/// BlockSSORPrec<..., 1, ...>).
template <typename Scalar, int kPrecBlockSize, typename MatrixType>
struct BlockSSORPrec final : Preconditioner<Scalar> {
  static_assert(!std::is_const_v<Scalar>, "Implementation assumes Scalar is non-const.");
  static_assert(kPrecBlockSize > 0, "Preconditioner block size must be positive");
  static constexpr auto kType = PreconditionerType::BlockSSOR;

  /// @brief Constructor
  /// @param[in] A Input square matrix
  /// @param[in] omega Relaxation factor in (0, 2)
  explicit BlockSSORPrec(MatrixType const& A, Scalar omega = Scalar(1));

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

  void Update(MatrixType const& A);

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

 protected:
  using DiagType = Matrix<Scalar, kPrecBlockSize, kPrecBlockSize>;

  std::reference_wrapper<MatrixType const> _inputA;
  std::vector<int> _lowerEnd = {};
  std::vector<int> _upperStart = {};
  std::vector<DiagType> _inverseDiagBlocks = {};
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
    int kBlockSize,
    typename ScalarA,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
void MakePositionArrays(
    [[maybe_unused]] Matrix<
        ScalarA,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDir,
        kOwnership,
        kMajorDim> const& A,
    [[maybe_unused]] std::vector<int>& lowerEnd,
    [[maybe_unused]] std::vector<int>& upperStart) {}

/// @brief Extract indices for lower and upper triangular part per row for BlockSparseMatrix
///
/// @tparam kBlockSize
/// @tparam InputScalar
/// @tparam CRIdx
/// @tparam Ptr
/// @tparam AStorage
/// @param A Block sparse matrix
/// @param lowerEnd First index for a column not in the lower triangular part
/// @param upperStart First index for a column in the upper triangular part
///
/// @note This routine assumes (without checking) that the column indices are sorted on each row.
/// @note When kBlockSize is equal to 1 or kBlockSizeA, only the array lowerEnd is filled because we
/// have upperStart[*] = lowerEnd[*] + 1.
/// @note When kBlockSize is different from 1 and from kBlockSizeA, no array is filled.
template <
    int kBlockSize,
    typename InputScalar,
    int kBlockSizeA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename AStorage>
void MakePositionArrays(
    [[maybe_unused]] BlockSparseMatrix<InputScalar, kBlockSizeA, CRIdx, Ptr, AStorage> const& A,
    [[maybe_unused]] std::vector<int>& lowerEnd,
    [[maybe_unused]] std::vector<int>& upperStart) {
  if constexpr ((kBlockSize == 1) || (kBlockSize == kBlockSizeA)) {
    using NonConstIdx = std::remove_const_t<CRIdx>;
    MOCHI_ASSERT_VERBOSE(isize(lowerEnd) >= A.BlockRows(), "Insufficient span size.");
    MOCHI_ASSERT_VERBOSE(isize(upperStart) >= A.BlockRows(), "Insufficient span size.");
    for (NonConstIdx ir = 0; ir < A.BlockRows(); ++ir) {
      auto const colIdx = A.Indices(ir);
      auto ptr = std::lower_bound(colIdx.begin(), colIdx.end(), ir);
      lowerEnd[ir] = static_cast<int>(ptr - colIdx.begin());
      MOCHI_ASSERT_VERBOSE(colIdx[lowerEnd[ir]] == ir, "Diagonal entry is missing on block row.");
      upperStart[ir] = lowerEnd[ir] + 1;
    }
  }
}

/// @brief Extract indices for lower and upper triangular part per row
///
/// @tparam kBlockSize
/// @tparam InputScalar
/// @tparam CRIdx
/// @tparam Ptr
/// @tparam AStorage
/// @param A Sparse matrix
/// @param lowerEnd First index for a column not in the lower triangular part
/// @param upperStart First index for a column in the upper triangular part
///
/// @note This routine assumes (without checking) that the column indices are sorted on each row.
template <
    int kBlockSize,
    typename InputScalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename AStorage>
void MakePositionArrays(
    SparseMatrix<InputScalar, CRIdx, Ptr, AStorage> const& A,
    std::vector<int>& lowerEnd,
    std::vector<int>& upperStart) {
  MOCHI_ASSERT_VERBOSE(isize(lowerEnd) >= A.Rows(), "Insufficient span size.");
  MOCHI_ASSERT_VERBOSE(isize(upperStart) >= A.Rows(), "Insufficient span size.");
  using NonConstIdx = std::remove_const_t<CRIdx>;
  if constexpr (kBlockSize == 1) {
    for (NonConstIdx row = 0; row < A.Rows(); ++row) {
      auto colIdx = A.Indices(row);
      auto ptr = std::lower_bound(colIdx.begin(), colIdx.end(), row);
      lowerEnd[row] = static_cast<int>(ptr - colIdx.begin());
      MOCHI_ASSERT_VERBOSE(colIdx[lowerEnd[row]] == row, "Diagonal entry is missing.")
      upperStart[row] = lowerEnd[row] + 1;
    }
  } else {
    for (NonConstIdx start = 0; start < A.Rows(); start += kBlockSize) {
      for (int ir = 0; ir < kBlockSize; ++ir) {
        auto const row = start + ir;
        auto colIdx = A.Indices(row);
        auto ptr = std::lower_bound(colIdx.begin(), colIdx.end(), start);
        lowerEnd[row] = static_cast<int>(ptr - colIdx.begin());
        upperStart[row] = lowerEnd[row];
        for (;
             ((upperStart[row] < colIdx.size()) && (colIdx[upperStart[row]] < start + kBlockSize));
             ++upperStart[row]) {
        }
      }
    }
  }
}

/// @brief Apply the block SSOR operator on a BlockSparseMatrix
///
/// @param[in] A BlockSparseMatrix
/// @param[in] invDiagBlock Vector for inverse of diagonal blocks
/// @param[in] lowerEnd Array of local indices where the lower triangular part ends for each block
/// row. The array is used only when kPrecBlockSize matches kBlockSizeA.
/// @param[in] upperStart
/// @param[in] t Pointer to workspace
/// @param[in] omega_s Omega
/// @param[in] ratio_s (2 - omega) / omega
/// @param[in] x Input
/// @param[out] y Result of the application of preconditioner to x
///
/// @note We assume that the matrix A is square.
/// @note We assume that the diagonal entry for the BlockSparseMatrix is non-zero.
/// @note We assume that the block-row entries in the BlockSparseMatrix are sorted in increasing
/// block-column index. No check is performed.
/// @note We assume that Input and Output are (derived) from Matrix.
/// @note No assumption is made on the orientation of `x` or `y`.
template <
    int kPrecBlockSize,
    typename Scalar,
    typename ScalarA,
    int kBlockSizeA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage,
    typename InvDiag,
    typename Input,
    typename Output>
void ApplyBlockSsorOperator(
    BlockSparseMatrix<ScalarA, kBlockSizeA, CRIdx, Ptr, Storage> const& A,
    std::vector<InvDiag> const& invDiagBlock,
    [[maybe_unused]] std::vector<int> const& lowerEnd,
    [[maybe_unused]] std::vector<int> const& upperStart,
    Scalar* t,
    Scalar omega_s,
    Scalar ratio_s,
    Input const& x,
    Output&& y) {
  using NonConstIdx = std::remove_const_t<CRIdx>;
  auto nRows = A.Rows();
  ColumnVectorView<Scalar> tview(t, nRows);
  if constexpr (kPrecBlockSize == kBlockSizeA) {
    [[maybe_unused]] ColumnVector<Scalar> xTmp;
    [[maybe_unused]] auto const xTmpRequiredSize = static_cast<NonConstIdx>(
        mochi::details::RowMultiplier<Scalar, kBlockSizeA>::GetWorkspaceSize(A.MaxNnzPerRow()));
    [[maybe_unused]] ColumnVector<Scalar, kBlockSizeA> Ly;
    [[maybe_unused]] auto aLy = mochi::details::GetAccessor(Ly);
    for (int jc = 0; jc < x.Cols(); ++jc) {
      tview = x.Col(jc);
      auto yjc = y.Col(jc);
      yjc.template TopRows<kPrecBlockSize>(kPrecBlockSize) =
          invDiagBlock[0] * tview.template TopRows<kPrecBlockSize>(kPrecBlockSize);
      [[maybe_unused]] auto aY = mochi::details::GetAccessor(yjc);
      for (int irb = 1; irb < A.BlockRows(); ++irb) {
        auto colIdx = A.Indices(irb);
        auto values = A.Values(irb);
        auto ti = tview.template MiddleRows<kPrecBlockSize>(irb * kPrecBlockSize, kPrecBlockSize);
        if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
          mochi::details::RowMultiplier<Scalar, kBlockSizeA>::ApplyToColVector(
              colIdx.data(),
              values.data(),
              lowerEnd[irb],
              values.LeadDim(),
              aY,
              aLy,
              /*br*/ 0,
              /*c*/ 0,
              xTmp,
              xTmpRequiredSize);
          ti -= Ly;
        } else {
          static_assert(
              mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
          for (int kk = 0; kk < lowerEnd[irb]; ++kk) {
            ti -= values[kk] *
                yjc.template MiddleRows<kBlockSizeA>(colIdx[kk] * kBlockSizeA, kBlockSizeA);
          }
        }
        auto yi = yjc.template MiddleRows<kPrecBlockSize>(irb * kPrecBlockSize, kPrecBlockSize);
        yi = invDiagBlock[irb] * ti;
      }
      if (Abs(omega_s * ratio_s - Scalar(1)) > 2 * std::numeric_limits<Scalar>::epsilon()) {
        auto scaling = omega_s * ratio_s;
        tview *= scaling;
        yjc.template MiddleRows<kPrecBlockSize>(nRows - kPrecBlockSize, kPrecBlockSize) *= scaling;
      }
      for (int irb = A.BlockRows() - 2; irb >= 0; --irb) {
        auto colIdx = A.Indices(irb);
        auto values = A.Values(irb);
        auto ti = tview.template MiddleRows<kPrecBlockSize>(irb * kPrecBlockSize, kPrecBlockSize);
        auto const shift = upperStart[irb];
        if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
          mochi::details::RowMultiplier<Scalar, kBlockSizeA>::ApplyToColVector(
              colIdx.data() + shift,
              values.data() + shift * kBlockSizeA,
              static_cast<int>(colIdx.size() - shift),
              values.LeadDim(),
              aY,
              aLy,
              /*br*/ 0,
              /*c*/ 0,
              xTmp,
              xTmpRequiredSize);
          ti -= Ly;
        } else {
          static_assert(
              mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
          for (int kk = shift; kk < isize(colIdx); ++kk) {
            ti -= values[kk] *
                yjc.template MiddleRows<kBlockSizeA>(colIdx[kk] * kBlockSizeA, kBlockSizeA);
          }
        }
        auto yi = yjc.template MiddleRows<kPrecBlockSize>(irb * kPrecBlockSize, kPrecBlockSize);
        yi = invDiagBlock[irb] * ti;
      }
    }
  } else {
    auto nRowBlocks = static_cast<NonConstIdx>(nRows / kPrecBlockSize);
    // Define A = D + L + U
    for (int jc = 0; jc < x.Cols(); ++jc) {
      tview = x.Col(jc);
      auto yjc = y.Col(jc);
      // Forward: y <-- (D / w + L )^{-1} y
      for (NonConstIdx ir = 0, startRow = 0; ir < nRowBlocks; ++ir, startRow += kPrecBlockSize) {
        for (int ii = 0; ii < kPrecBlockSize;) {
          NonConstIdx row = startRow + ii;
          auto aRowBlock = static_cast<NonConstIdx>(row / kBlockSizeA);
          int aRowLocal = static_cast<int>(row - aRowBlock * kBlockSizeA);
          int len = std::min<int>(kBlockSizeA - aRowLocal, kPrecBlockSize - ii);
          auto const colIdx = A.Indices(aRowBlock);
          auto const values = A.Values(aRowBlock);
          auto tb = tview.MiddleRows(startRow + ii, len);
          for (NonConstIdx jj = 0; (jj < colIdx.size()) && (colIdx[jj] * kBlockSizeA < startRow);
               ++jj) {
            int const cLen =
                std::min(kBlockSizeA, static_cast<int>(startRow - colIdx[jj] * kBlockSizeA));
            auto const mat = values[jj].Block(aRowLocal, 0, len, cLen);
            tb -= mat * yjc.MiddleRows(colIdx[jj] * kBlockSizeA, cLen);
          }
          ii += len;
        } // for (int ii = 0; ii < kPrecBlockSize;)
        auto ytmp = yjc.template MiddleRows<kPrecBlockSize>(startRow, kPrecBlockSize);
        ytmp =
            invDiagBlock[ir] * tview.template MiddleRows<kPrecBlockSize>(startRow, kPrecBlockSize);
      }
      if (Abs(omega_s * ratio_s - Scalar(1)) > 2 * std::numeric_limits<Scalar>::epsilon()) {
        tview *= (omega_s * ratio_s);
      }
      // Backward: y <-- (D / w + U )^{-1} y
      for (NonConstIdx ir = nRowBlocks - 1, startRow = nRows - kPrecBlockSize; ir >= 0;
           --ir, startRow -= kPrecBlockSize) {
        for (int ii = 0; ii < kPrecBlockSize;) {
          NonConstIdx row = startRow + ii;
          auto aRowBlock = static_cast<NonConstIdx>(row / kBlockSizeA);
          auto aRowLocal = static_cast<int>(row - aRowBlock * kBlockSizeA);
          int len = std::min<int>(kBlockSizeA - aRowLocal, kPrecBlockSize - ii);
          auto const colIdx = A.Indices(aRowBlock);
          auto const values = A.Values(aRowBlock);
          auto tb = tview.MiddleRows(startRow + ii, len);
          for (NonConstIdx jj = 0; jj < colIdx.size(); ++jj) {
            if (colIdx[jj] * kBlockSizeA + kBlockSizeA <= startRow + kPrecBlockSize) {
              continue;
            }
            int const clen = std::min(
                kBlockSizeA, colIdx[jj] * kBlockSizeA + kBlockSizeA - startRow - kPrecBlockSize);
            auto const mat = values[jj].Block(aRowLocal, kBlockSizeA - clen, len, clen);
            tb -= mat * yjc.MiddleRows(colIdx[jj] * kBlockSizeA + kBlockSizeA - clen, clen);
          }
          ii += len;
        } // for (int ii = 0; ii < kPrecBlockSize;)
        auto ytmp = yjc.template MiddleRows<kPrecBlockSize>(startRow, kPrecBlockSize);
        ytmp =
            invDiagBlock[ir] * tview.template MiddleRows<kPrecBlockSize>(startRow, kPrecBlockSize);
      } // for (int ir = nRowBlocks - 1; ir >= 0; --ir)
    }
  }
}

/// @brief Apply the block SSOR operator on a SparseMatrix
///
/// @note We assume that the matrix A is square.
/// @note We assume that the diagonal entry for the SparseMatrix is non-zero.
/// @note We assume that the row entries in the SparseMatrix are sorted in increasing column index.
/// No check is performed.
/// @note We assume that Input and Output are (derived) from Matrix.
/// @note No assumption is made on the orientation of `x`.
template <
    int kPrecBlockSize,
    typename Scalar,
    typename ScalarA,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage,
    typename Input,
    typename Output>
void ApplyBlockSsorOperator(
    SparseMatrix<ScalarA, CRIdx, Ptr, Storage> const& A,
    std::vector<Matrix<Scalar, kPrecBlockSize, kPrecBlockSize>> const& invDiagBlock,
    std::vector<int> const& lowerEnd,
    std::vector<int> const& upperStart,
    Scalar* t,
    Scalar omega_s,
    Scalar ratio_s,
    Input const& x,
    Output&& y) {
  using NonConstIdx = std::remove_const_t<CRIdx>;
  static_assert(
      MatTraits<Input>::kMajorDir == MatTraits<Output>::kMajorDir,
      "Mixed orientation not implemented yet"); // Please update unit tests if mixed orientations
                                                // are supported in the future.
  NonConstIdx nRows = A.Rows();
  ColumnVectorView<Scalar> tview(t, A.Rows());
  auto nRowBlocks = static_cast<NonConstIdx>(nRows / kPrecBlockSize);
  auto aPtr = A.Pointers();
  auto aVal = A.Values();
  auto aCol = A.Indices();
  for (int jc = 0; jc < x.Cols(); ++jc) {
    // Initialize
    tview = x.Col(jc);
    auto yjc = y.Col(jc);
    // Define A = D + L + U
    // Forward: y <-- (D / w + L )^{-1} y
    yjc.template TopRows<kPrecBlockSize>(kPrecBlockSize) =
        invDiagBlock[0] * tview.template TopRows<kPrecBlockSize>(kPrecBlockSize);
    for (NonConstIdx ir = 1, startRow = kPrecBlockSize; ir < nRowBlocks;
         ++ir, startRow += kPrecBlockSize) {
      for (int ii = 0; ii < kPrecBlockSize; ++ii) {
        NonConstIdx row = startRow + ii;
        auto spanCol = aCol.subspan(aPtr[row], lowerEnd[row]);
        auto spanVal = aVal.subspan(aPtr[row], lowerEnd[row]);
        Scalar sum = 0;
        if constexpr (MatTraits<Input>::kMajorDir == krylov::Direction::ColMajor) {
          mochi::details::AddRowSparseTimesColumnVector<Scalar, CRIdx>(
              spanCol, spanVal, yjc.Data(), sum);
        } else {
          mochi::details::AddRowSparseTimesVector<Scalar, CRIdx>(spanCol, spanVal, yjc, sum);
        }
        tview[row] -= sum;
      }
      // y <-- w * D_{ir}^{-1} * t
      auto yBlock = yjc.template MiddleRows<kPrecBlockSize>(startRow, kPrecBlockSize);
      yBlock =
          invDiagBlock[ir] * tview.template MiddleRows<kPrecBlockSize>(startRow, kPrecBlockSize);
    }
    if (Abs(omega_s * ratio_s - 1) > 2 * std::numeric_limits<Scalar>::epsilon()) {
      tview *= (omega_s * ratio_s);
      yjc.template MiddleRows<kPrecBlockSize>(nRows - kPrecBlockSize, kPrecBlockSize) *=
          (omega_s * ratio_s);
    }
    // Backward: y <-- (D / w + U )^{-1} y
    for (NonConstIdx ir = nRowBlocks - 2, startRow = nRows - 2 * kPrecBlockSize; ir >= 0;
         --ir, startRow -= kPrecBlockSize) {
      for (int ii = 0; ii < kPrecBlockSize; ++ii) {
        NonConstIdx row = startRow + ii;
        size_t len = size_t(aPtr[row + 1] - aPtr[row] - upperStart[row]);
        auto spanCol = aCol.subspan(aPtr[row] + upperStart[row], len);
        auto spanVal = aVal.subspan(aPtr[row] + upperStart[row], len);
        Scalar sum = 0;
        if constexpr (MatTraits<Input>::kMajorDir == krylov::Direction::ColMajor) {
          mochi::details::AddRowSparseTimesColumnVector<Scalar, CRIdx>(
              spanCol, spanVal, yjc.Data(), sum);
        } else {
          mochi::details::AddRowSparseTimesVector<Scalar, CRIdx>(spanCol, spanVal, yjc, sum);
        }
        tview[row] -= sum;
      }
      auto yBlock = yjc.template MiddleRows<kPrecBlockSize>(startRow, kPrecBlockSize);
      // y <-- w * D_{ir}^{-1} * t
      yBlock =
          invDiagBlock[ir] * tview.template MiddleRows<kPrecBlockSize>(startRow, kPrecBlockSize);
    } // for (int ir = nRowBlocks - 2; ir >= 0; --ir)
  }
}

/// @brief Apply the block SSOR operator on a dense Matrix
///
/// @note We assume that the matrix A is square.
/// @note We assume that Input and Output are (derived) from Matrix.
/// @note No assumption is made on the orientation of `x` or `y`.
template <
    int kPrecBlockSize,
    typename Scalar,
    typename ScalarA,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim,
    typename Input,
    typename Output>
void ApplyBlockSsorOperator(
    Matrix<ScalarA, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim> const&
        A,
    std::vector<Matrix<Scalar, kPrecBlockSize, kPrecBlockSize>> const& invDiagBlock,
    [[maybe_unused]] std::vector<int> const& lowerEnd,
    [[maybe_unused]] std::vector<int> const& upperStart,
    Scalar* t,
    Scalar omega_s,
    Scalar ratio_s,
    Input const& x,
    Output&& y) {
  auto nRows = A.Rows();
  int nRowBlocks = static_cast<int>(nRows / kPrecBlockSize);
  ColumnVectorView<Scalar> tview(t, nRows);
  for (int jc = 0; jc < x.Cols(); ++jc) {
    auto yjc = y.Col(jc);
    // Define A = D + L + U
    // Forward: y <-- (D / w + L )^{-1} y
    tview = x.Col(jc);
    yjc.template TopRows<kPrecBlockSize>(kPrecBlockSize) =
        invDiagBlock[0] * tview.template TopRows<kPrecBlockSize>(kPrecBlockSize);
    for (int row = kPrecBlockSize, irb = 1; row < nRows; row += kPrecBlockSize, ++irb) {
      auto L = A.template Block<kPrecBlockSize, krylov::kDynamic>(row, 0, kPrecBlockSize, row);
      tview.template MiddleRows<kPrecBlockSize>(row, kPrecBlockSize) -= L * yjc.TopRows(row);
      // y <-- (w/d_{ii}) t
      yjc.template MiddleRows<kPrecBlockSize>(row, kPrecBlockSize) =
          invDiagBlock[irb] * tview.template MiddleRows<kPrecBlockSize>(row, kPrecBlockSize);
    }
    if (Abs(omega_s * ratio_s - 1) > 2 * std::numeric_limits<Scalar>::epsilon()) {
      auto prod = omega_s * ratio_s;
      tview *= prod;
      yjc.template MiddleRows<kPrecBlockSize>(nRows - kPrecBlockSize, kPrecBlockSize) *= prod;
    }
    // Backward: y <-- (D / w + U )^{-1} y
    for (int row = nRows - 2 * kPrecBlockSize, irb = nRowBlocks - 2; row >= 0;
         row -= kPrecBlockSize, --irb) {
      auto U = A.template Block<kPrecBlockSize, krylov::kDynamic>(
          row, row + kPrecBlockSize, kPrecBlockSize, nRows - kPrecBlockSize - row);
      // y <-- (w/d_{ii}) t
      yjc.template MiddleRows<kPrecBlockSize>(row, kPrecBlockSize) = invDiagBlock[irb] *
          (tview.template MiddleRows<kPrecBlockSize>(row, kPrecBlockSize) -
           U * yjc.MiddleRows(row + kPrecBlockSize, U.Cols()));
    } // for (int row = nRows - 2 * kPrecBlockSize; row >= 0; row -= kPrecBlockSize)
  } // for (int jc = 0; jc < x.Cols(); ++jc)
}

} // namespace mochi::krylov::details

namespace mochi::krylov {

template <typename Scalar, int kPrecBlockSize, typename MatrixType>
BlockSSORPrec<Scalar, kPrecBlockSize, MatrixType>::BlockSSORPrec(MatrixType const& A, Scalar omega)
    : _inputA(A), _omega_s(omega), _ratio_s((2 - _omega_s) / _omega_s) {
  if constexpr (IsSparseMatrix<MatrixType> || IsBlockSparseMatrix<MatrixType>) {
    using CRIdx = decltype(A.Rows());
    static_assert(
        std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
        "BlockSSORPrec requires signed matrix indices representable as int.");
  }
  MOCHI_ASSERT_VERBOSE((_omega_s > 0) && (_omega_s < 2), "Invalid parameter 'omega'.");
  Update(A);
}

template <typename Scalar, int kPrecBlockSize, typename MatrixType>
void BlockSSORPrec<Scalar, kPrecBlockSize, MatrixType>::Update(MatrixType const& A) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(
      A.Rows() % kPrecBlockSize == 0, "Incompatible pairing of matrix and block size.");
  MOCHI_ASSERT_VERBOSE(mochi::details::HasSortedRowIndices(A), "Row entries are not sorted.");
  _inputA = A;
  //--- Get position arrays
  //--- Note that these arrays remain empty for a (dense) `Matrix` input
  if constexpr (IsBlockSparseMatrix<MatrixType>) {
    _lowerEnd.resize(A.BlockRows());
    _upperStart.resize(A.BlockRows());
  } else if constexpr (IsSparseMatrix<MatrixType>) {
    _lowerEnd.resize(A.Rows());
    _upperStart.resize(A.Rows());
  } else {
    static_assert(IsMatrix<MatrixType>, "Unexpected matrix type");
  }
  details::MakePositionArrays<kPrecBlockSize>(A, _lowerEnd, _upperStart);
  //--- We will store general dense blocks
  auto numBlocks = A.Rows() / kPrecBlockSize;
  _inverseDiagBlocks.resize(numBlocks);
  ExtractBlockDiagonal(A, MakeSpan(_inverseDiagBlocks));
  // If needed for performance, BatchedInverse could be replaced by a BatchedSymInverse.
  BatchedInverse(MakeSpan(_inverseDiagBlocks));
  //--- Scale the diagonal blocks if needed
  if (Abs(_omega_s - 1) > 2 * std::numeric_limits<Scalar>::epsilon()) {
    //--- Scale the inverse diagonal blocks with _omega_s
    for (auto& invD : _inverseDiagBlocks) {
      invD *= _omega_s;
    }
  }
  //
  _workSpace.Resize(A.Rows(), 1);
}

template <typename Scalar, int kPrecBlockSize, typename MatrixType>
template <typename Input, typename Output>
void BlockSSORPrec<Scalar, kPrecBlockSize, MatrixType>::operator()(Input const& xin, Output&& yout)
    const {
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
  details::ApplyBlockSsorOperator<kPrecBlockSize, Scalar>(
      _inputA.get(),
      _inverseDiagBlocks,
      _lowerEnd,
      _upperStart,
      const_cast<Scalar*>(_workSpace.data()),
      _omega_s,
      _ratio_s,
      x,
      y);
}

} // namespace mochi::krylov
