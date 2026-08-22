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
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/graph_views.h>

#include <algorithm>
#include <concepts>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace mochi::krylov {

/// @brief Empty definition of colored SSOR for non-supported input matrix
template <typename MatrixType>
struct ColoredSSORPrec {
  ColoredSSORPrec() = delete;
};

/// @brief Implementation of colored SSOR preconditioner for BlockSparseMatrix input
///
/// @note The class will define a coloring of the blocks. Based on the coloring, it will compute the
/// SSOR preconditioner on the re-ordered matrix.
/// @note The lower and upper triangular part of the re-ordered matrix are stored explicitly.
///
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
struct ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>> final
    : Preconditioner<std::remove_const_t<Scalar>> {
  using BSpMat = BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>;
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  static constexpr auto kType = PreconditionerType::ColoredSSOR;

  /// @brief Constructor
  /// @param[in] A Input square matrix
  /// @param[in] omega Relaxation factor in (0, 2)
  explicit ColoredSSORPrec(BSpMat const& A, Scalar omega = Scalar(1));

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors)
  /// @param[in] xin Input column vector(s)
  /// @param[out] yout Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& xin, Output&& yout) const;

  /// @brief Apply the preconditioner to a column vector.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<NonConstScalar> Px) const override {
    operator()(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<Scalar const> x,
      ColumnVectorView<NonConstScalar> Px,
      ParallelWorkerInfo const& data) const override;

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

  /// @brief Update the numerical values of the preconditioner
  /// @param[in] A Input matrix for updated values
  ///
  /// @note
  /// The updating function assumes that the input matrix has the same sparsity pattern as the
  /// matrix used to construct the preconditioner.
  /// No check on the sparsity pattern is done and the coloring is not recomputed.
  void Update(BSpMat const& A) {
    SetNumericalValues(A);
  }

 protected:
  void SetNumericalValues(BSpMat const& A);

  template <typename InputType, typename OutputType>
  void CopyFromColored(
      InputType const& coloredY,
      NonConstIdx bBegin,
      NonConstIdx bEnd,
      OutputType& y) const;

  template <typename InputType, typename OutputType>
  void CopyToColored(InputType const& x, NonConstIdx bBegin, NonConstIdx bEnd, OutputType& coloredY)
      const;

  template <typename InputType, typename OutputType>
  void KernelBackward(InputType& t, OutputType& y, NonConstIdx brBegin, NonConstIdx brEnd) const;

  template <typename InputType, typename OutputType>
  void KernelForward(InputType& t, OutputType& y, NonConstIdx brBegin, NonConstIdx brEnd) const;

  /// @brief Lower block-triangular sparse matrix for the re-ordered matrix
  /// @note `_coloredL` block-column indices are NOT sorted per block-row
  BlockSparseMatrix<NonConstScalar, kBlockSize, NonConstIdx, NonConstPtr> _coloredL = {};

  /// @brief Upper block-triangular sparse matrix for the re-ordered matrix
  /// @note `_coloredU` block-column indices are NOT sorted per block-row
  BlockSparseMatrix<NonConstScalar, kBlockSize, NonConstIdx, NonConstPtr> _coloredU = {};
  DynamicArray<Matrix<NonConstScalar, kBlockSize, kBlockSize, Direction::RowMajor>>
      _coloredDiagBlock = {};
  NonConstScalar _omega_s = 1;
  NonConstScalar _ratio_s = 1;
  Graph<int, int> _colorToBlock = {};
  DynamicArray<NonConstIdx> _initToColored = {};
  mutable Matrix<NonConstScalar> _workSpace = {};
};

/// @brief Implementation of colored SSOR preconditioner for SparseMatrix input
///
/// @note The class will define a coloring of the rows. Based on the coloring, it will compute the
/// SSOR preconditioner on the re-ordered matrix.
/// @note The lower and upper triangular part of the re-ordered matrix are stored explicitly.
///
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
struct ColoredSSORPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>> final
    : Preconditioner<std::remove_const_t<Scalar>> {
  using SpMat = SparseMatrix<Scalar, CRIdx, Ptr, Storage>;
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  static constexpr auto kType = PreconditionerType::ColoredSSOR;

  /// @brief Constructor
  /// @param[in] A Input square matrix
  /// @param[in] omega Relaxation factor in (0, 2)
  explicit ColoredSSORPrec(SpMat const& A, Scalar omega = Scalar(1));

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors)
  /// @param[in] xin Input column vector(s)
  /// @param[out] yout Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& xin, Output&& yout) const;

  /// @brief Apply the preconditioner to a column vector.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<NonConstScalar> Px) const override {
    operator()(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<Scalar const> x,
      ColumnVectorView<NonConstScalar> Px,
      ParallelWorkerInfo const& data) const override;

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

  /// @brief Update the numerical values of the preconditioner
  /// @param[in] A Input matrix for updated values
  ///
  /// @note
  /// The updating function assumes that the input matrix has the same sparsity pattern as the
  /// matrix used to construct the preconditioner.
  /// No check on the sparsity pattern is done and the coloring is not recomputed.
  void Update(SpMat const& A) {
    SetNumericalValues(A);
  }

 protected:
  void SetNumericalValues(SpMat const& A);

  template <typename InputType, typename OutputType>
  void CopyFromColored(
      InputType const& coloredY,
      NonConstIdx rBegin,
      NonConstIdx rEnd,
      OutputType& y) const;

  template <typename InputType, typename OutputType>
  void CopyToColored(InputType const& x, NonConstIdx rBegin, NonConstIdx rEnd, OutputType& coloredY)
      const;

  template <typename InputType, typename OutputType>
  void KernelBackward(InputType& t, OutputType& y, NonConstIdx rBegin, NonConstIdx rEnd) const;

  template <typename InputType, typename OutputType>
  void KernelForward(InputType& t, OutputType& y, NonConstIdx rBegin, NonConstIdx rEnd) const;

  /// @brief Lower triangular sparse matrix for the re-ordered matrix
  /// @note `_coloredL` column indices are NOT sorted per row
  SparseMatrix<NonConstScalar, NonConstIdx, NonConstPtr> _coloredL = {};

  /// @brief Upper triangular sparse matrix for the re-ordered matrix
  /// @note `_coloredU` column indices are NOT sorted per row
  SparseMatrix<NonConstScalar, NonConstIdx, NonConstPtr> _coloredU = {};
  DynamicArray<NonConstScalar> _coloredInvDiag = {};
  NonConstScalar _omega_s = 1;
  NonConstScalar _ratio_s = 1;
  Graph<int, int> _colorToRow = {};
  DynamicArray<NonConstIdx> _initToColored = {};
  mutable Matrix<NonConstScalar> _workSpace = {};
};

/// @brief Implementation of colored SSOR preconditioner for Matrix input
///
/// @note The class will create a copy of the input matrix in SparseMatrix format (removing the zero
/// entries). Based on the coloring of the rows, it will compute the SSOR preconditioner on the
/// re-ordered SparseMatrix.
/// @note The lower and upper triangular part of the re-ordered matrix are stored explicitly.
///
template <
    typename Scalar,
    int kRows,
    int kCols,
    krylov::Direction kDir,
    krylov::Ownership kOwner,
    int kLeadDim>
struct ColoredSSORPrec<Matrix<Scalar, kRows, kCols, kDir, kOwner, kLeadDim>> final
    : Preconditioner<std::remove_const_t<Scalar>> {
  using DeMat = Matrix<Scalar, kRows, kCols, kDir, kOwner, kLeadDim>;
  using NonConstScalar = std::remove_const_t<Scalar>;
  static constexpr auto kType = PreconditionerType::ColoredSSOR;

  /// @brief Constructor
  /// @param[in] A Input square matrix
  /// @param[in] omega Relaxation factor in (0, 2)
  explicit ColoredSSORPrec(DeMat const& A, Scalar omega = Scalar(1));

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors)
  /// @param[in] xin Input column vector(s)
  /// @param[out] yout Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& xin, Output&& yout) const;

  /// @brief Apply the preconditioner to a column vector.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<NonConstScalar> Px) const override {
    operator()(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<Scalar const> x,
      ColumnVectorView<NonConstScalar> Px,
      ParallelWorkerInfo const& data) const override;

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

 protected:
  /// @brief Pointer to actual implementation
  ///
  /// @note The graph coloring will not find a good coloring for the dense matrix.
  /// To obtain small parallelism, we treat the input matrix as a sparse matrix.
  std::unique_ptr<ColoredSSORPrec<SparseMatrix<NonConstScalar>>> _spPrec = nullptr;
};

//
// Definition of member functions
//

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::ColoredSSORPrec(
    BSpMat const& A,
    Scalar omega)
    : _coloredDiagBlock(A.BlockRows()),
      _omega_s(omega),
      _ratio_s(static_cast<Scalar>((2.0 - _omega_s) / _omega_s)),
      _initToColored(A.BlockRows()) {
  static_assert(
      std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
      "ColoredSSORPrec requires signed matrix indices representable as int.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE((_omega_s > 0) && (_omega_s < 2), "Invalid parameter 'omega'.");
  //--- Color the graph
  _colorToBlock = GreedyColoring(AsGraphView(A));
  NonConstIdx count = 0;
  for (int ic = 0; ic < _colorToBlock.size(); ++ic) {
    MOCHI_ASSERT_VERBOSE(_colorToBlock.EdgeCount(ic) > 0, "Colors must not be empty.");
    auto bList = _colorToBlock[ic];
    for (auto b : bList) {
      _initToColored[b] = count++;
    }
  }
  //--- Extract the triangular parts
  DynamicArray<NonConstPtr> lptr, uptr;
  lptr.resize_noinit(A.BlockRows() + 1);
  uptr.resize_noinit(A.BlockRows() + 1);
  DynamicArray<NonConstIdx> lidx, uidx;
  lidx.reserve(A.NumNonZeroBlocks() / 2);
  uidx.reserve(A.NumNonZeroBlocks() / 2);
  count = 0;
  lptr[0] = 0;
  uptr[0] = 0;
  auto const cList = _colorToBlock.GetTargets();
  for (auto b : cList) {
    auto const aCol = A.Indices(b);
    auto const bColor = _initToColored[b];
    for (int k = 0; k < isize(aCol); ++k) {
      auto const color = _initToColored[aCol[k]];
      if (bColor > color) {
        lidx.push_back(color);
      } else if (bColor < color) {
        uidx.push_back(color);
      }
    }
    lptr[count + 1] = isize(lidx);
    uptr[count + 1] = isize(uidx);
    count += 1;
  }
  //
  DynamicArray<NonConstScalar> lval, uval;
  lval.resize_noinit(lidx.size() * kBlockSize * kBlockSize);
  uval.resize_noinit(uidx.size() * kBlockSize * kBlockSize);
  _coloredL.Reset(A.BlockCols(), std::move(lptr), std::move(lidx), std::move(lval));
  _coloredU.Reset(A.BlockCols(), std::move(uptr), std::move(uidx), std::move(uval));
  //
  SetNumericalValues(A);
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename InputType, typename OutputType>
void ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::CopyFromColored(
    InputType const& coloredY,
    NonConstIdx bBegin,
    NonConstIdx bEnd,
    OutputType& y) const {
  for (NonConstIdx ib = bBegin; ib < bEnd; ++ib) {
    y.template MiddleRows<kBlockSize>(ib * kBlockSize, kBlockSize) =
        coloredY.template MiddleRows<kBlockSize>(_initToColored[ib] * kBlockSize, kBlockSize);
  }
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::
    SetNumericalValues(BSpMat const& A) {
  // Parallelize over block rows - each thread gets its own position arrays to avoid race
  // conditions Use similar heuristics as other parts of the codebase for task size estimation
  auto const numBlockRows = A.BlockRows();
  auto const totalElements = A.NumNonZeroBlocks();
  constexpr int kMinBlocksPerTask = Max(1, 10000 / (kBlockSize * kBlockSize));
  auto const minBlockRowsPerTask = Clamp<NonConstIdx>(
      static_cast<NonConstIdx>(
          kMinBlocksPerTask * numBlockRows / Max<NonConstPtr>(1, totalElements)),
      1,
      numBlockRows);

  ParallelForRange(
      "ColoredSSORPrecSetup",
      0,
      numBlockRows,
      minBlockRowsPerTask,
      numBlockRows,
      [&](NonConstIdx ibBegin, NonConstIdx ibEnd) {
        for (NonConstIdx ib = ibBegin; ib < ibEnd; ++ib) {
          auto const aIdx = A.Indices(ib);
          auto const aVal = A.Values(ib);
          auto const newBlockRow = _initToColored[ib];
          //
          auto lVal = _coloredL.Values(newBlockRow);
          auto uVal = _coloredU.Values(newBlockRow);
          int lPos = 0, uPos = 0;
          MOCHI_ASSERT_VERBOSE(
              aVal.NumBlocks() - 1 == lVal.NumBlocks() + uVal.NumBlocks(),
              "Incompatible block dimensions");
          // Process each non-zero element in current block row
          for (int k = 0; k < isize(aIdx); ++k) {
            auto const thisBlockCol = _initToColored[aIdx[k]];
            if (thisBlockCol < newBlockRow) {
              // Block in the lower triangular part for the colored matrix
              lVal[lPos++] = aVal[k];
            } else if (thisBlockCol > newBlockRow) {
              // Block in the upper triangular part for the colored matrix
              uVal[uPos++] = aVal[k];
            } else {
              //--- Diagonal block
              _coloredDiagBlock[newBlockRow] = aVal[k];
              for (int i = 0; i < kBlockSize; ++i) {
                auto const dtmp = _coloredDiagBlock[newBlockRow](i, i);
                if (dtmp != Scalar{0})
                  MOCHI_LIKELY {
                    _coloredDiagBlock[newBlockRow](i, i) = _omega_s / dtmp;
                  }
              }
            }
          }
        }
      });
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename InputType, typename OutputType>
void ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::CopyToColored(
    InputType const& x,
    NonConstIdx bBegin,
    NonConstIdx bEnd,
    OutputType& coloredY) const {
  for (NonConstIdx ib = bBegin; ib < bEnd; ++ib) {
    coloredY.template MiddleRows<kBlockSize>(_initToColored[ib] * kBlockSize, kBlockSize) =
        x.template MiddleRows<kBlockSize>(ib * kBlockSize, kBlockSize);
  }
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename InputType, typename OutputType>
void ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::KernelBackward(
    InputType& t,
    OutputType& y,
    NonConstIdx brBegin,
    NonConstIdx brEnd) const {
  auto const rBegin = brBegin * kBlockSize;
  auto const rEnd = brEnd * kBlockSize;
  _coloredU.ApplyToRange(y, y, rBegin, rEnd);
  auto tr = t.MiddleRows(rBegin, rEnd - rBegin);
  auto yr = y.MiddleRows(rBegin, rEnd - rBegin);
  tr -= yr;
  for (NonConstIdx br = brBegin, rl = 0; br < brEnd; ++br, rl += kBlockSize) {
    auto const& D = _coloredDiagBlock[br];
    for (int i = kBlockSize - 1; i >= 0; --i) {
      for (int j = i + 1; j < kBlockSize; ++j) {
        tr.Row(rl + i) -= D(i, j) * yr.Row(rl + j);
      }
      yr.Row(rl + i) = D(i, i) * tr.Row(rl + i);
    }
  }
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename InputType, typename OutputType>
void ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::KernelForward(
    InputType& t,
    OutputType& y,
    NonConstIdx brBegin,
    NonConstIdx brEnd) const {
  auto const rBegin = brBegin * kBlockSize;
  auto const rEnd = brEnd * kBlockSize;
  _coloredL.ApplyToRange(y, t, rBegin, rEnd);
  auto tr = t.MiddleRows(rBegin, rEnd - rBegin);
  auto yr = y.MiddleRows(rBegin, rEnd - rBegin);
  tr = yr - tr;
  for (NonConstIdx br = brBegin, rl = 0; br < brEnd; ++br, rl += kBlockSize) {
    auto const& D = _coloredDiagBlock[br];
    for (int i = 0; i < kBlockSize; ++i) {
      for (int j = 0; j < i; ++j) {
        tr.Row(rl + i) -= D(i, j) * yr.Row(rl + j);
      }
      yr.Row(rl + i) = D(i, i) * tr.Row(rl + i);
    }
  }
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename Input, typename Output>
void ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::operator()(
    Input const& xin,
    Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_coloredL.Rows(), xin, yout);
  constexpr int kNumColsAtCT = (details::MatTraits<Input>::kNumCols >= 0)
      ? details::MatTraits<Input>::kNumCols
      : details::MatTraits<Output>::kNumCols;
  CRIdx const nCols = xin.Cols();
  MOCHI_ASSERT_VERBOSE(
      kNumColsAtCT == krylov::kDynamic || kNumColsAtCT == nCols, "Inconsistent number of columns");
  if (nCols == 0) {
    return;
  }
  //--- Convert to MatrixView
  MatrixView<
      typename details::MatTraits<Input>::Scalar const,
      details::MatTraits<Input>::kNumRows,
      kNumColsAtCT,
      details::MatTraits<Input>::kMajorDir,
      krylov::kDynamic>
      x(xin.data(), xin.Rows(), xin.Cols(), xin.LeadDim());
  MatrixView<
      std::remove_const_t<typename details::MatTraits<Output>::Scalar>,
      details::MatTraits<Output>::kNumRows,
      kNumColsAtCT,
      details::MatTraits<Output>::kMajorDir,
      krylov::kDynamic>
      y(yout.data(), yout.Rows(), yout.Cols(), yout.LeadDim());
  //
  if ((_workSpace.Rows() < xin.Rows()) || (_workSpace.Cols() < 2 * xin.Cols())) {
    _workSpace.Reset(xin.Rows(), 2 * xin.Cols());
  }
  //
  auto t = _workSpace.template Block<krylov::kDynamic, kNumColsAtCT>(0, 0, xin.Rows(), xin.Cols());
  auto coloredY = _workSpace.template Block<krylov::kDynamic, kNumColsAtCT>(
      0, xin.Cols(), xin.Rows(), xin.Cols());
  //
  CopyToColored(x, 0, _coloredL.BlockRows(), coloredY);
  //
  NonConstIdx pos = 0;
  for (int c = 0; c < _colorToBlock.size(); ++c) {
    auto const n = static_cast<CRIdx>(_colorToBlock.EdgeCount(c));
    // At least ~20000 FLOPs per worker.
    auto const minBlockRowsPerTask = Clamp<CRIdx>(
        static_cast<CRIdx>(
            10000 * n /
            ((_coloredL.NumNonZeroBlocksInBlockRowRange(pos, pos + n) + n) * kBlockSize *
             kBlockSize * nCols)),
        1,
        n);
    ParallelForRange(
        "ColoredSSORSolveL", pos, pos + n, minBlockRowsPerTask, n, [&](CRIdx brBegin, CRIdx brEnd) {
          KernelForward(t, coloredY, brBegin, brEnd);
        });
    pos += n;
  } // for (int c = 0; c < _colorToNode.size(); ++c)
  //
  if (Abs(_omega_s * _ratio_s - Scalar(1)) > 2 * std::numeric_limits<Scalar>::epsilon()) {
    t *= (_omega_s * _ratio_s);
  }
  //
  for (int c = _colorToBlock.size() - 1; c >= 0; --c) {
    auto const n = static_cast<CRIdx>(_colorToBlock.EdgeCount(c));
    pos -= n;
    // At least ~20000 FLOPs per worker.
    auto const minBlockRowsPerTask = Clamp<CRIdx>(
        static_cast<CRIdx>(
            10000 * n /
            ((_coloredU.NumNonZeroBlocksInBlockRowRange(pos, pos + n) + n) * kBlockSize *
             kBlockSize * nCols)),
        1,
        n);
    ParallelForRange(
        "ColoredSSORSolveU", pos, pos + n, minBlockRowsPerTask, n, [&](CRIdx brBegin, CRIdx brEnd) {
          KernelBackward(t, coloredY, brBegin, brEnd);
        });
  } // for (int c = _colorToNode.size() - 1; c >= 0; --c)
  //
  CopyFromColored(coloredY, 0, _coloredL.BlockRows(), y);
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::ConcurrentSolve(
    ColumnVectorView<Scalar const> x,
    ColumnVectorView<NonConstScalar> Px,
    ParallelWorkerInfo const& data) const {
  int const iWorker = data.workerId;
  int const numWorkers = data.numWorkers;
  auto const rowBegin = data.rBegin;
  auto const rowEnd = data.rEnd;
  //
  Preconditioner<NonConstScalar>::ValidateInputOutput(_coloredL.Rows(), x, Px);
  //
  if ((_workSpace.Rows() < x.Rows()) || (_workSpace.Cols() < 2 * x.Cols())) {
    if (iWorker == 0) {
      _workSpace.Reset(x.Rows(), 2 * x.Cols());
    }
  }
  data.BarrierWait();
  //
  auto t = _workSpace.template Block<krylov::kDynamic, 1>(0, 0, x.Rows(), 1);
  auto coloredY = _workSpace.template Block<krylov::kDynamic, 1>(0, 1, x.Rows(), 1);
  //
  NonConstIdx bBegin = rowBegin / kBlockSize;
  NonConstIdx bEnd = rowEnd / kBlockSize;
  CopyToColored(x, bBegin, bEnd, coloredY);
  data.BarrierWait();
  //
  NonConstIdx pos = 0;
  for (int c = 0; c < _colorToBlock.size(); ++c) {
    auto const n = static_cast<CRIdx>(_colorToBlock.EdgeCount(c));
    auto const cLen = int((n + numWorkers - 1)) / numWorkers;
    auto brBegin = pos + NonConstIdx(cLen * iWorker);
    auto brEnd = Min<NonConstIdx>(brBegin + cLen, pos + n);
    if (brBegin < brEnd) {
      KernelForward(t, coloredY, brBegin, brEnd);
    }
    pos += n;
    data.BarrierWait();
  } // for (int c = 0; c < _colorToNode.size(); ++c)
  //
  if (Abs(_omega_s * _ratio_s - Scalar(1)) > 2 * std::numeric_limits<Scalar>::epsilon()) {
    if (bBegin < bEnd) {
      t.MiddleRows(bBegin * kBlockSize, (bEnd - bBegin) * kBlockSize) *= (_omega_s * _ratio_s);
    }
    data.BarrierWait();
  }
  //
  for (int c = _colorToBlock.size() - 1; c >= 0; --c) {
    auto const n = static_cast<CRIdx>(_colorToBlock.EdgeCount(c));
    pos -= n;
    auto const cLen = int((n + numWorkers - 1)) / numWorkers;
    auto brBegin = pos + NonConstIdx(cLen * iWorker);
    auto brEnd = Min<NonConstIdx>(brBegin + cLen, pos + n);
    if (brBegin < brEnd) {
      KernelBackward(t, coloredY, brBegin, brEnd);
    }
    data.BarrierWait();
  } // for (int c = _colorToNode.size() - 1; c >= 0; --c)
  //
  CopyFromColored(coloredY, bBegin, bEnd, Px);
}

//
// Definition of member functions for ColoredSSORPrec<SparseMatrix<...>>
//

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
ColoredSSORPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::ColoredSSORPrec(
    SpMat const& A,
    Scalar omega)
    : _coloredInvDiag(A.Rows()),
      _omega_s(omega),
      _ratio_s(static_cast<Scalar>((2.0 - _omega_s) / _omega_s)),
      _initToColored(A.Rows()) {
  static_assert(
      std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
      "ColoredSSORPrec requires signed matrix indices representable as int.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE((_omega_s > 0) && (_omega_s < 2), "Invalid parameter 'omega'.");
  //--- Color the graph
  _colorToRow = GreedyColoring(AsGraphView(A));
  //--- Extract the renumbering
  NonConstIdx count = 0;
  for (int c = 0; c < _colorToRow.size(); ++c) {
    MOCHI_ASSERT_VERBOSE(_colorToRow.EdgeCount(c) > 0, "Colors must not be empty.");
    auto rList = _colorToRow[c];
    for (auto r : rList) {
      _initToColored[r] = count++;
    }
  }
  //--- Extract the triangular parts (of the renumbered matrix)
  DynamicArray<NonConstPtr> lptr, uptr;
  lptr.resize_noinit(A.Rows() + 1);
  uptr.resize_noinit(A.Rows() + 1);
  DynamicArray<NonConstIdx> lidx, uidx;
  lidx.reserve(A.NumNonZeros() / 2);
  uidx.reserve(A.NumNonZeros() / 2);
  count = 0;
  lptr[0] = 0;
  uptr[0] = 0;
  for (int c = 0; c < _colorToRow.size(); ++c) {
    auto rList = _colorToRow[c];
    for (auto r : rList) {
      auto aCol = A.Indices(r);
      auto const cRow = _initToColored[r];
      for (int k = 0; k < isize(aCol); ++k) {
        auto const color = _initToColored[aCol[k]];
        if (cRow > color) {
          lidx.push_back(color);
        } else if (cRow < color) {
          uidx.push_back(color);
        }
      }
      lptr[count + 1] = isize(lidx);
      uptr[count + 1] = isize(uidx);
      count += 1;
    }
  }
  //
  DynamicArray<NonConstScalar> lval;
  lval.resize_noinit(lidx.size());
  DynamicArray<NonConstScalar> uval;
  uval.resize_noinit(uidx.size());
  _coloredL.Reset(A.Cols(), std::move(lptr), std::move(lidx), std::move(lval));
  _coloredU.Reset(A.Cols(), std::move(uptr), std::move(uidx), std::move(uval));
  //
  SetNumericalValues(A);
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void ColoredSSORPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::SetNumericalValues(
    SpMat const& A) {
  auto const numRows = A.Rows();
  // Parallelize over rows - each thread gets its own position arrays to avoid race conditions
  // Use similar heuristics as other parts of the codebase for task size estimation
  auto const totalElements = A.NumNonZeros();
  int const kMinValuesPerTask = 10000;
  auto const minRowsPerTask = Clamp<NonConstIdx>(
      static_cast<NonConstIdx>(kMinValuesPerTask * numRows / Max<NonConstPtr>(1, totalElements)),
      1,
      numRows);
  ParallelForRange(
      "ColoredSSORPrecSetup",
      0,
      numRows,
      minRowsPerTask,
      numRows,
      [&](NonConstIdx ibBegin, NonConstIdx ibEnd) {
        for (NonConstIdx ib = ibBegin; ib < ibEnd; ++ib) {
          auto const aIdx = A.Indices(ib);
          auto const aVal = A.Values(ib);
          auto const newRow = _initToColored[ib];
          auto lVal = _coloredL.Values(newRow);
          auto uVal = _coloredU.Values(newRow);
          int uPos = 0, lPos = 0;
          // Process each non-zero element in current block row
          for (int k = 0; k < isize(aIdx); ++k) {
            auto const thisCol = _initToColored[aIdx[k]];
            if (thisCol < newRow) {
              // Lower triangular part for the colored matrix
              lVal[lPos++] = aVal[k];
            } else if (thisCol > newRow) {
              // Upper triangular part for the colored matrix
              uVal[uPos++] = aVal[k];
            } else {
              //--- Treat diagonal term
              if (aVal[k] == Scalar{0})
                MOCHI_UNLIKELY {
                  _coloredInvDiag[newRow] = 0;
                }
              else {
                _coloredInvDiag[newRow] = _omega_s / aVal[k];
              }
            }
          }
        }
      });
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename InputType, typename OutputType>
void ColoredSSORPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::CopyFromColored(
    InputType const& coloredY,
    NonConstIdx rBegin,
    NonConstIdx rEnd,
    OutputType& y) const {
  for (NonConstIdx ir = rBegin; ir < rEnd; ++ir) {
    y.Row(ir) = coloredY.Row(_initToColored[ir]);
  }
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename InputType, typename OutputType>
void ColoredSSORPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::CopyToColored(
    InputType const& x,
    NonConstIdx rBegin,
    NonConstIdx rEnd,
    OutputType& coloredY) const {
  for (NonConstIdx ir = rBegin; ir < rEnd; ++ir) {
    coloredY.Row(_initToColored[ir]) = x.Row(ir);
  }
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename InputType, typename OutputType>
void ColoredSSORPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::KernelBackward(
    InputType& t,
    OutputType& y,
    std::remove_const_t<CRIdx> rBegin,
    std::remove_const_t<CRIdx> rEnd) const {
  // We can use the same variable in ApplyToRange
  _coloredU.ApplyToRange(y, y, rBegin, rEnd);
  auto tr = t.MiddleRows(rBegin, rEnd - rBegin);
  auto yr = y.MiddleRows(rBegin, rEnd - rBegin);
  tr -= yr;
  for (NonConstIdx rl = 0; rl < (rEnd - rBegin); ++rl) {
    yr.Row(rl) = _coloredInvDiag[rl + rBegin] * tr.Row(rl);
  }
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename InputType, typename OutputType>
void ColoredSSORPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::KernelForward(
    InputType& t,
    OutputType& y,
    NonConstIdx rBegin,
    NonConstIdx rEnd) const {
  _coloredL.ApplyToRange(y, t, rBegin, rEnd);
  auto tr = t.MiddleRows(rBegin, rEnd - rBegin);
  auto yr = y.MiddleRows(rBegin, rEnd - rBegin);
  tr = yr - tr;
  for (NonConstIdx rl = 0; rl < (rEnd - rBegin); ++rl) {
    yr.Row(rl) = _coloredInvDiag[rl + rBegin] * tr.Row(rl);
  }
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename Input, typename Output>
void ColoredSSORPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::operator()(
    Input const& xin,
    Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_coloredL.Rows(), xin, yout);
  //
  constexpr int kNumColsAtCT = (details::MatTraits<Input>::kNumCols >= 0)
      ? details::MatTraits<Input>::kNumCols
      : details::MatTraits<Output>::kNumCols;
  CRIdx const nCols = xin.Cols();
  MOCHI_ASSERT_VERBOSE(
      kNumColsAtCT == krylov::kDynamic || kNumColsAtCT == nCols, "Inconsistent number of columns");
  if (nCols == 0) {
    return;
  }
  //--- Convert to MatrixView
  MatrixView<
      typename details::MatTraits<Input>::Scalar const,
      details::MatTraits<Input>::kNumRows,
      kNumColsAtCT,
      details::MatTraits<Input>::kMajorDir,
      krylov::kDynamic>
      x(xin.data(), xin.Rows(), xin.Cols(), xin.LeadDim());
  MatrixView<
      std::remove_const_t<typename details::MatTraits<Output>::Scalar>,
      details::MatTraits<Output>::kNumRows,
      kNumColsAtCT,
      details::MatTraits<Output>::kMajorDir,
      krylov::kDynamic>
      y(yout.data(), yout.Rows(), yout.Cols(), yout.LeadDim());
  //
  if ((_workSpace.Rows() < xin.Rows()) || (_workSpace.Cols() < 2 * xin.Cols())) {
    _workSpace.Reset(xin.Rows(), 2 * xin.Cols());
  }
  //
  auto t = _workSpace.template Block<krylov::kDynamic, kNumColsAtCT>(0, 0, xin.Rows(), xin.Cols());
  auto coloredY = _workSpace.template Block<krylov::kDynamic, kNumColsAtCT>(
      0, xin.Cols(), xin.Rows(), xin.Cols());
  //
  CopyToColored(x, 0, xin.Rows(), coloredY);
  //
  NonConstIdx pos = 0;
  for (int c = 0; c < _colorToRow.size(); ++c) {
    auto const n = static_cast<CRIdx>(_colorToRow.EdgeCount(c));
    // At least ~20000 FLOPs per worker.
    auto const minRowsPerTask = Clamp<CRIdx>(
        static_cast<CRIdx>(
            10000 * n / ((_coloredL.NumNonZerosInRowRange(pos, pos + n) + n) * nCols)),
        1,
        n);
    ParallelForRange(
        "ColoredSSORSolveL", pos, pos + n, minRowsPerTask, n, [&](CRIdx rBegin, CRIdx rEnd) {
          // _coloredL is strictly triangular.
          KernelForward(t, coloredY, rBegin, rEnd);
        });
    pos += n;
  } // for (int c = 0; c < _colorToNode.size(); ++c)
  //
  if (Abs(_omega_s * _ratio_s - Scalar(1)) > 2 * std::numeric_limits<Scalar>::epsilon()) {
    t *= (_omega_s * _ratio_s);
  }
  //
  for (int c = _colorToRow.size() - 1; c >= 0; --c) {
    auto const n = static_cast<CRIdx>(_colorToRow.EdgeCount(c));
    pos -= n;
    // At least ~20000 FLOPs per worker.
    auto const minRowsPerTask = Clamp<CRIdx>(
        static_cast<CRIdx>(
            10000 * n / ((_coloredU.NumNonZerosInRowRange(pos, pos + n) + n) * nCols)),
        1,
        n);
    ParallelForRange(
        "ColoredSSORSolveU", pos, pos + n, minRowsPerTask, n, [&](CRIdx rBegin, CRIdx rEnd) {
          // _coloredU is strictly triangular.
          KernelBackward(t, coloredY, rBegin, rEnd);
        });
  } // for (int c = _colorToNode.size() - 1; c >= 0; --c)
  //
  CopyFromColored(coloredY, 0, xin.Rows(), y);
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void ColoredSSORPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::ConcurrentSolve(
    ColumnVectorView<Scalar const> x,
    ColumnVectorView<NonConstScalar> Px,
    ParallelWorkerInfo const& data) const {
  int const iWorker = data.workerId;
  int const numWorkers = data.numWorkers;
  auto const rowBegin = data.rBegin;
  auto const rowEnd = data.rEnd;
  //
  Preconditioner<NonConstScalar>::ValidateInputOutput(_coloredL.Rows(), x, Px);
  //
  if ((_workSpace.Rows() < x.Rows()) || (_workSpace.Cols() < 2 * x.Cols())) {
    if (iWorker == 0) {
      _workSpace.Reset(x.Rows(), 2 * x.Cols());
    }
  }
  data.BarrierWait();
  //
  auto t = _workSpace.template Block<krylov::kDynamic, 1>(0, 0, x.Rows(), x.Cols());
  auto coloredY = _workSpace.template Block<krylov::kDynamic, 1>(0, x.Cols(), x.Rows(), x.Cols());
  //
  CopyToColored(x, rowBegin, rowEnd, coloredY);
  data.BarrierWait();
  //
  NonConstIdx pos = 0;
  for (int c = 0; c < _colorToRow.size(); ++c) {
    auto const n = static_cast<CRIdx>(_colorToRow.EdgeCount(c));
    auto const cLen = int((n + numWorkers - 1)) / numWorkers;
    auto rBegin = pos + NonConstIdx(cLen * iWorker);
    auto rEnd = Min<NonConstIdx>(rBegin + cLen, pos + n);
    if (rBegin < rEnd) {
      KernelForward(t, coloredY, rBegin, rEnd);
    }
    pos += n;
    data.BarrierWait();
  } // for (int c = 0; c < _colorToNode.size(); ++c)
  //
  if (Abs(_omega_s * _ratio_s - Scalar(1)) > 2 * std::numeric_limits<Scalar>::epsilon()) {
    if (rowBegin < rowEnd) {
      t.MiddleRows(rowBegin, rowEnd - rowBegin) *= (_omega_s * _ratio_s);
    }
    data.BarrierWait();
  }
  //
  for (int c = _colorToRow.size() - 1; c >= 0; --c) {
    auto const n = static_cast<CRIdx>(_colorToRow.EdgeCount(c));
    pos -= n;
    auto const cLen = int((n + numWorkers - 1)) / numWorkers;
    auto rBegin = pos + NonConstIdx(cLen * iWorker);
    auto rEnd = Min<NonConstIdx>(rBegin + cLen, pos + n);
    if (rBegin < rEnd) {
      KernelBackward(t, coloredY, rBegin, rEnd);
    }
    data.BarrierWait();
  } // for (int c = _colorToNode.size() - 1; c >= 0; --c)
  //
  CopyFromColored(coloredY, rowBegin, rowEnd, Px);
}

//
// Definition of member functions for ColoredSSORPrec<Matrix<...>>
//

template <
    typename Scalar,
    int kRows,
    int kCols,
    krylov::Direction kDir,
    krylov::Ownership kOwner,
    int kLeadDim>
ColoredSSORPrec<Matrix<Scalar, kRows, kCols, kDir, kOwner, kLeadDim>>::ColoredSSORPrec(
    DeMat const& A,
    Scalar omega)
    : _spPrec() {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE((omega > 0) && (omega < 2), "Invalid parameter 'omega'.");
  //
  auto ASp = ToSparseMatrix(A, true);
  _spPrec = std::make_unique<ColoredSSORPrec<SparseMatrix<NonConstScalar>>>(ASp, omega);
}

template <
    typename Scalar,
    int kRows,
    int kCols,
    krylov::Direction kDir,
    krylov::Ownership kOwner,
    int kLeadDim>
template <typename Input, typename Output>
void ColoredSSORPrec<Matrix<Scalar, kRows, kCols, kDir, kOwner, kLeadDim>>::operator()(
    Input const& xin,
    Output&& yout) const {
  _spPrec->operator()(xin, yout);
}

template <
    typename Scalar,
    int kRows,
    int kCols,
    krylov::Direction kDir,
    krylov::Ownership kOwner,
    int kLeadDim>
void ColoredSSORPrec<Matrix<Scalar, kRows, kCols, kDir, kOwner, kLeadDim>>::ConcurrentSolve(
    ColumnVectorView<Scalar const> x,
    ColumnVectorView<NonConstScalar> Px,
    ParallelWorkerInfo const& data) const {
  _spPrec->ConcurrentSolve(x, Px, data);
}

} // namespace mochi::krylov
