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
#include <mochi_core/linear_algebra/block_view_vector.h>
#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/sparsity_utils.h>

#include <algorithm>
#include <concepts>
#include <type_traits>

namespace mochi::krylov {

/// @brief Empty definition of relaxed ILU for non-supported input matrix.
template <typename MatrixType>
struct RelaxedILUPrec {
  RelaxedILUPrec() = delete;
};

/// @brief Implementation of relaxed ILU for BlockSparseMatrix input.
///
/// @note A relaxation coefficient of 0.95 was suggested by [Chan and van der
/// Vorst](https://ww3.math.ucla.edu/camreport/cam94-27.pdf) as a practical approximation for alpha
/// = 1 - c h^2.
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
struct RelaxedILUPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>> final
    : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  static constexpr auto kType = PreconditionerType::ILU0;

  /// @brief Constructor
  ///
  /// @param[in] A Input block sparse matrix
  /// @param[in] level Level of fill-in
  /// @param[in] alphaRelax Relaxation factor. 0 yields the plain ILU factorization and 1 yields the
  /// modified ILU.
  explicit RelaxedILUPrec(
      BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& A,
      int level,
      Scalar alphaRelax);

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors).
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

  /// @brief Concurrent solve for parallel execution.
  /// @todo Implement when linear solvers refactor is complete.
  void ConcurrentSolve(
      [[maybe_unused]] ColumnVectorView<Scalar const> x,
      [[maybe_unused]] ColumnVectorView<NonConstScalar> Px,
      [[maybe_unused]] ParallelWorkerInfo const& data) const override {
    MOCHI_ASSERT(false, "ConcurrentSolve not supported for incomplete LU preconditioner.");
  }

  constexpr PreconditionerType GetType() const override {
    MOCHI_ASSERT(_level == 0, "RelaxedILUPrec::GetType assumes ILU0.");
    return kType;
  }

  /// @brief Update the preconditioner from a matrix with the same sparsity pattern.
  /// @param[in] A Input block sparse matrix.
  /// @note The sparsity of the fill level is NOT recomputed. Only the values are updated.
  template <
      typename ScalarA,
      typename CRIdxA,
      typename PtrA,
      template <typename, typename...> typename StorageA>
  void Update(BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A);

 protected:
  /// @brief Perform the ILU factorization.
  /// @warning Assumes that _lu has already been initialized with the matrix values.
  void Factorize();

  /// @brief Storage for the incomplete factors L and U
  /// - The unit lower triangular matrix L is stored in the lower part of _lu
  /// but the diagonal entries of ones are not stored.
  /// - The upper triangular matrix U is stored in the upper part of _lu
  BlockSparseMatrix<NonConstScalar, kBlockSize, NonConstIdx, NonConstPtr> _lu;

  /// @brief Array pointing to the start of the upper part (inc. diagonal) for each block row
  /// The value for each block row is a "local" integer (local to the block row)
  DynamicArray<NonConstPtr> _uStart;

  /// @brief Inverse of the block diagonal for each block row
  DynamicArray<RowMatrix<NonConstScalar, kBlockSize, kBlockSize>> _invDiag;

  /// @brief Relaxation coefficient in [0, 1]
  /// _alpha = 0 yields plain ILU
  /// _alpha = 1 yields Modified ILU
  NonConstScalar _alpha;

  /// @brief Fill-in level.
  int _level = 0;
};

/// @brief Implementation of relaxed ILU for Matrix input.
///
/// @note A relaxation coefficient of 0.95 was suggested by [Chan and van der
/// Vorst](https://ww3.math.ucla.edu/camreport/cam94-27.pdf) as a practical approximation for alpha
/// = 1 - c h^2.
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
struct RelaxedILUPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>
    final : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  static constexpr auto kType = PreconditionerType::ILU0;

  /// @brief Constructor
  ///
  /// @param[in] A Input matrix
  /// @param[in] level Level of fill-in
  /// @param[in] alphaRelax Relaxation factor. 0 yields the plain ILU factorization and 1 yields the
  /// modified ILU.
  ///
  /// @warning Only zero fill-in level is currently supported.
  explicit RelaxedILUPrec(
      Matrix<
          Scalar,
          kRowsAtCompileTime,
          kColsAtCompileTime,
          kMajorDir,
          kOwnership,
          kMajorDim> const& A,
      int level,
      Scalar alphaRelax);

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors).
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

  /// @brief Concurrent solve for parallel execution.
  /// @todo Implement when linear solvers refactor is complete.
  void ConcurrentSolve(
      [[maybe_unused]] ColumnVectorView<Scalar const> x,
      [[maybe_unused]] ColumnVectorView<NonConstScalar> Px,
      [[maybe_unused]] ParallelWorkerInfo const& data) const override {
    MOCHI_ASSERT(false, "ConcurrentSolve not supported for incomplete LU preconditioner.");
  }

  constexpr PreconditionerType GetType() const override {
    MOCHI_ASSERT(_level == 0, "RelaxedILUPrec::GetType assumes ILU0.");
    return kType;
  }

  /// @brief Update the preconditioner from a matrix with the same sparsity pattern.
  /// @param[in] A Input matrix.
  /// @note The sparsity of the fill level is NOT recomputed. Only the values are updated.
  template <typename ScalarA, Direction kMajorDirA, Ownership kOwnershipA, int kMajorDimA>
  void Update(
      Matrix<
          ScalarA,
          kRowsAtCompileTime,
          kColsAtCompileTime,
          kMajorDirA,
          kOwnershipA,
          kMajorDimA> const& A);

 protected:
  /// @brief Perform the ILU factorization.
  /// @warning Assumes that _lu has already been initialized with the matrix values.
  void Factorize();

  /// @brief Initialize the sparsity mask from the current _lu values.
  /// Must be called before Factorize() to capture the original sparsity pattern.
  void InitializeSparsityMask();

  /// @brief Storage for the incomplete factors L and U
  /// - The unit lower triangular matrix L is stored in the lower part of _lu
  /// but the diagonal entries of ones are not stored.
  /// - The upper triangular matrix U is stored in the upper part of _lu
  RowMatrix<NonConstScalar, kRowsAtCompileTime, kColsAtCompileTime> _lu;

  /// @brief Sparsity mask from the original matrix (true = originally non-zero).
  /// Used to determine which entries are structural zeros vs fill-in during factorization. This is
  /// necessary because entries of _lu may become numerically zero during factorization even if the
  /// corresponding entry was originally non-zero.
  RowMatrix<bool, kRowsAtCompileTime, kColsAtCompileTime> _sparsityMask;

  /// @brief Relaxation coefficient in [0, 1]
  /// _alpha = 0 yields plain ILU(0)
  /// _alpha = 1 yields Modified ILU(0)
  NonConstScalar _alpha;

  /// @brief Fill-in level.
  int _level = 0;
};

/// @brief Implementation of relaxed ILU for SparseMatrix input.
///
/// @note Follows the CSR implementation from [Saad, Section 10.3.2, p.
/// 309](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf).
/// @note A relaxation coefficient of 0.95 was suggested by [Chan and van der
/// Vorst](https://ww3.math.ucla.edu/camreport/cam94-27.pdf) as a practical approximation for alpha
/// = 1 - c h^2.
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
struct RelaxedILUPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>> final
    : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  static constexpr auto kType = PreconditionerType::ILU0;

  /// @brief Constructor
  ///
  /// @param[in] A Input sparse matrix
  /// @param[in] level Level of fill-in
  /// @param[in] alphaRelax Relaxation factor. 0 yields the plain ILU factorization and 1 yields the
  /// modified ILU.
  explicit RelaxedILUPrec(
      SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A,
      int level,
      Scalar alphaRelax);

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors).
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

  /// @brief Concurrent solve for parallel execution.
  /// @todo Implement when linear solvers refactor is complete.
  void ConcurrentSolve(
      [[maybe_unused]] ColumnVectorView<Scalar const> x,
      [[maybe_unused]] ColumnVectorView<NonConstScalar> Px,
      [[maybe_unused]] ParallelWorkerInfo const& data) const override {
    MOCHI_ASSERT(false, "ConcurrentSolve not supported for incomplete LU preconditioner.");
  }

  constexpr PreconditionerType GetType() const override {
    MOCHI_ASSERT(_level == 0, "RelaxedILUPrec::GetType assumes ILU0.");
    return kType;
  }

  /// @brief Update the preconditioner from a matrix with the same sparsity pattern.
  /// @param[in] A Input sparse matrix.
  /// @note The sparsity of the fill level is NOT recomputed. Only the values are updated.
  template <
      typename ScalarA,
      typename CRIdxA,
      typename PtrA,
      template <typename, typename...> typename StorageA>
  void Update(SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& A);

 protected:
  /// @brief Perform the ILU factorization.
  /// @warning Assumes that _lu has already been initialized with the matrix values.
  void Factorize();

  /// @brief Storage for the incomplete factors L and U
  /// - The unit lower triangular matrix L is stored in the lower part of _lu
  /// but the diagonal entries of ones are not stored.
  /// - The upper triangular matrix U is stored in the upper part of _lu
  SparseMatrix<NonConstScalar, NonConstIdx, NonConstPtr> _lu;

  /// @brief Array pointing to the start of the upper part (inc. diagonal) for each row
  /// The value for each row is a "global" integer
  DynamicArray<NonConstPtr> _uStart;

  /// @brief Relaxation coefficient in [0, 1]
  /// _alpha = 0 yields plain ILU
  /// _alpha = 1 yields Modified ILU
  NonConstScalar _alpha;

  /// @brief Fill-in level.
  int _level = 0;
};

//
//--- Implementation of class member functions
//

//
//--- Specialization for BlockSparseMatrix
//

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
RelaxedILUPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::RelaxedILUPrec(
    BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& A,
    int level,
    Scalar alphaRelax)
    : _uStart(A.BlockRows()), _invDiag(A.BlockRows()), _alpha(alphaRelax), _level(level) {
  static_assert(
      std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
      "RelaxedILUPrec requires signed matrix indices representable as int.");
  MOCHI_ASSERT_VERBOSE(_level >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT_VERBOSE(
      (_alpha >= Scalar(0)) && (_alpha <= Scalar(1)), "Out-of-range relaxation factor.");
  mochi::details::ConvertToFillLevel(_level, A, _lu);
  Factorize();
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA>
void RelaxedILUPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::Update(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A) {
  static_assert(std::is_same_v<Scalar const, ScalarA const>, "Inconsistent scalar types");
  static_assert(std::is_same_v<CRIdx const, CRIdxA const>, "Inconsistent integer types");
  static_assert(std::is_same_v<Ptr const, PtrA const>, "Inconsistent integer types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _lu.Rows(), "Matrix size mismatch.");
  if (_level == 0) {
    auto srcValues = A.Values();
    auto dstValues = _lu.Values();
    MOCHI_ASSERT_VERBOSE(srcValues.size() == dstValues.size(), "Value array size mismatch.");
    MOCHI_ASSERT_VERBOSE(_lu.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(_lu.Indices() == A.Indices(), "Sparsity pattern mismatch.");
    std::copy(srcValues.begin(), srcValues.end(), dstValues.begin());
  } else {
    _lu.SetZero();
    _lu += A;
  }
  Factorize();
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void RelaxedILUPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::Factorize() {
  auto constexpr kDummyFlag = static_cast<Ptr>(-1);
  DynamicArray<NonConstPtr> iw(_lu.BlockRows(), kDummyFlag);
  //
  RowMatrix<NonConstScalar, kBlockSize, kBlockSize> sumDrops, scaling;
  for (NonConstIdx ib = 0; ib < _lu.BlockRows(); ++ib) {
    //--- Get inverse map for luIdx
    auto const colIdx = _lu.Indices(ib);
    auto values = _lu.Values(ib);
    MOCHI_ASSERT_VERBOSE(
        std::is_sorted(colIdx.begin(), colIdx.end()),
        "Block row does not have sorted block column indices");
    for (NonConstPtr p = 0; p < colIdx.size(); ++p) {
      iw[colIdx[p]] = p;
    }
    if (_alpha) {
      sumDrops.SetZero();
    }
    NonConstPtr p = 0;
    //-- Loop expects that the column indices are sorted
    for (; ((p < colIdx.size()) && (colIdx[p] < ib)); ++p) {
      auto kb = colIdx[p];
      scaling = values[p] * _invDiag[kb];
      values[p] = scaling;
      //
      auto jColIdx = _lu.Indices(kb);
      auto jValues = _lu.Values(kb);
      //
      if (_alpha) {
        for (NonConstPtr pp = _uStart[kb] + 1; pp < jColIdx.size(); ++pp) {
          auto pw = iw[jColIdx[pp]];
          if (pw == kDummyFlag) {
            sumDrops += scaling * jValues[pp];
          } else {
            values[pw] -= scaling * jValues[pp];
          }
        }
      } else {
        for (NonConstPtr pp = _uStart[kb] + 1; pp < jColIdx.size(); ++pp) {
          auto pw = iw[jColIdx[pp]];
          if (pw == kDummyFlag) {
            continue;
          }
          values[pw] -= scaling * jValues[pp];
        }
      }
    }
    if (_alpha) {
      values[p] -= _alpha * sumDrops;
    }
    if (colIdx[p] != ib)
      MOCHI_UNLIKELY {
        MOCHI_LOG_ERROR("Zero pivot at block row %d.", ib);
        break;
      }
    _uStart[ib] = p;
    _invDiag[ib] = Inverse(values[p]);
    //--- Reset entries of iw
    for (NonConstPtr pp = 0; pp < colIdx.size(); ++pp) {
      iw[colIdx[pp]] = kDummyFlag;
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
void RelaxedILUPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::operator()(
    Input const& xin,
    Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_lu.Rows(), xin, yout);
  //
  //--- Convert input and output to MatrixView
  //
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
  y = x;
  //
  [[maybe_unused]] ColumnVector<NonConstScalar> xTmp;
  [[maybe_unused]] auto const xTmpRequiredSize = static_cast<NonConstIdx>(
      mochi::details::RowMultiplier<NonConstScalar, kBlockSize>::GetWorkspaceSize(
          _lu.MaxNnzPerRow()));
  [[maybe_unused]] ColumnVector<NonConstScalar, kBlockSize> Ly;
  [[maybe_unused]] auto aLy = mochi::details::GetAccessor(Ly);
  for (int jc = 0; jc < x.Cols(); ++jc) {
    auto yjc = y.Col(jc);
    [[maybe_unused]] auto aY = mochi::details::GetAccessor(yjc);
    for (int irb = 1; irb < _lu.BlockRows(); ++irb) {
      auto colIdx = _lu.Indices(irb);
      auto values = _lu.Values(irb);
      if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::RowMultiplier<Scalar, kBlockSize>::ApplyToColVector(
            colIdx.data(),
            values.data(),
            _uStart[irb],
            values.LeadDim(),
            aY,
            aLy,
            /*br*/ 0,
            /*c*/ 0,
            xTmp,
            xTmpRequiredSize);
        yjc.template MiddleRows<kBlockSize>(irb * kBlockSize, kBlockSize) -= Ly;
      } else {
        static_assert(mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
        auto yij = yjc.template MiddleRows<kBlockSize>(irb * kBlockSize, kBlockSize);
        for (int kk = 0; kk < _uStart[irb]; ++kk) {
          yij -=
              values[kk] * yjc.template MiddleRows<kBlockSize>(colIdx[kk] * kBlockSize, kBlockSize);
        }
      }
    }
    //
    Ly = yjc.template MiddleRows<kBlockSize>((_lu.BlockRows() - 1) * kBlockSize, kBlockSize);
    yjc.template MiddleRows<kBlockSize>((_lu.BlockRows() - 1) * kBlockSize, kBlockSize) =
        _invDiag[_lu.BlockRows() - 1] * Ly;
    //
    for (int irb = _lu.BlockRows() - 2; irb >= 0; --irb) {
      auto colIdx = _lu.Indices(irb);
      auto values = _lu.Values(irb);
      Ly.SetZero();
      auto const shift = _uStart[irb] + 1;
      if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::RowMultiplier<Scalar, kBlockSize>::ApplyToColVector(
            colIdx.data() + shift,
            values.data() + shift * kBlockSize,
            isize(colIdx) - shift,
            values.LeadDim(),
            aY,
            aLy,
            /*br*/ 0,
            /*c*/ 0,
            xTmp,
            xTmpRequiredSize);
      } else {
        static_assert(mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
        for (int kk = shift; kk < colIdx.size(); ++kk) {
          Ly +=
              values[kk] * yjc.template MiddleRows<kBlockSize>(colIdx[kk] * kBlockSize, kBlockSize);
        }
      }
      Ly = yjc.template MiddleRows<kBlockSize>(irb * kBlockSize, kBlockSize) - Ly;
      yjc.template MiddleRows<kBlockSize>(irb * kBlockSize, kBlockSize) = _invDiag[irb] * Ly;
    }
  } // for (int jc = 0; jc < x.Cols(); ++jc)
}

//
//--- Specialization for Matrix
//

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
RelaxedILUPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
    RelaxedILUPrec(
        Matrix<
            Scalar,
            kRowsAtCompileTime,
            kColsAtCompileTime,
            kMajorDir,
            kOwnership,
            kMajorDim> const& A,
        int level,
        Scalar alphaRelax)
    : _alpha(alphaRelax), _level(level) {
  MOCHI_ASSERT_VERBOSE(_level >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT_VERBOSE(
      (_alpha >= Scalar(0)) && (_alpha <= Scalar(1)), "Out-of-range relaxation factor.")
  mochi::details::ConvertToFillLevel(_level, A, _lu);
  InitializeSparsityMask();
  Factorize();
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
template <typename ScalarA, Direction kMajorDirA, Ownership kOwnershipA, int kMajorDimA>
void RelaxedILUPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
    Update(
        Matrix<
            ScalarA,
            kRowsAtCompileTime,
            kColsAtCompileTime,
            kMajorDirA,
            kOwnershipA,
            kMajorDimA> const& A) {
  static_assert(std::is_same_v<Scalar const, ScalarA const>, "Inconsistent scalar types");
#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _lu.Rows(), "Matrix size mismatch.");
  for (int i = 0; i < A.Rows(); ++i) {
    for (int j = 0; j < A.Cols(); ++j) {
      MOCHI_ASSERT_VERBOSE(
          _sparsityMask(i, j) || (A(i, j) == Scalar{0}), "Sparsity pattern mismatch.");
    }
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
  _lu = A;
  Factorize();
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
void RelaxedILUPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
    InitializeSparsityMask() {
  _sparsityMask.Resize(_lu.Rows(), _lu.Cols());
  for (int i = 0; i < _lu.Rows(); ++i) {
    for (int j = 0; j < _lu.Cols(); ++j) {
      _sparsityMask(i, j) = (_lu(i, j) != Scalar{0});
    }
  }
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
void RelaxedILUPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
    Factorize() {
  for (int i = 0; i < _lu.Rows(); ++i) {
    for (int k = 0; k < i; ++k) {
      if (!_sparsityMask(i, k))
        MOCHI_UNLIKELY {
          continue;
        }
      _lu(i, k) *= _lu(k, k);
      for (int j = k + 1; j < _lu.Cols(); ++j) {
        if (!_sparsityMask(i, j))
          MOCHI_UNLIKELY {
            _lu(i, i) -= _alpha * _lu(i, k) * _lu(k, j);
          }
        else {
          _lu(i, j) -= _lu(i, k) * _lu(k, j);
        }
      }
    }
    if (_lu(i, i) == Scalar{0})
      MOCHI_UNLIKELY {
        MOCHI_LOG_ERROR("Zero pivot at row %d.", i);
        break;
      }
    _lu(i, i) = Scalar(1) / _lu(i, i);
  } // for (int i = 0; i < _lu.Rows(); ++i)
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
template <typename Input, typename Output>
void RelaxedILUPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
operator()(Input const& xin, Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_lu.Rows(), xin, yout);
  //
  //--- Convert input and output to MatrixView
  //
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
  y = x;
  //
  for (int ii = 1; ii < y.Rows(); ++ii) {
    y.Row(ii) -= _lu.template Block<1, krylov::kDynamic>(ii, 0, 1, ii) * y.TopRows(ii);
  }
  y.Row(y.Rows() - 1) *= _lu(y.Rows() - 1, y.Rows() - 1);
  for (int ii = y.Rows() - 2; ii >= 0; --ii) {
    y.Row(ii) -= _lu.template Block<1, krylov::kDynamic>(ii, ii + 1, 1, y.Rows() - ii - 1) *
        y.MiddleRows(ii + 1, y.Rows() - ii - 1);
    y.Row(ii) *= _lu(ii, ii);
  }
}

//
//--- Specialization for SparseMatrix
//

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
RelaxedILUPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::RelaxedILUPrec(
    SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A,
    int level,
    Scalar alphaRelax)
    : _uStart(A.Rows()), _alpha(alphaRelax), _level(level) {
  static_assert(
      std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
      "RelaxedILUPrec requires signed matrix indices representable as int.");
  MOCHI_ASSERT_VERBOSE(_level >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT_VERBOSE(
      (_alpha >= Scalar(0)) && (_alpha <= Scalar(1)), "Out-of-range relaxation factor.");
  mochi::details::ConvertToFillLevel(_level, A, _lu);
  Factorize();
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <
    typename ScalarA,
    typename CRIdxA,
    typename PtrA,
    template <typename, typename...> typename StorageA>
void RelaxedILUPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::Update(
    SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& A) {
  static_assert(std::is_same_v<Scalar const, ScalarA const>, "Inconsistent scalar types");
  static_assert(std::is_same_v<CRIdx const, CRIdxA const>, "Inconsistent integer types");
  static_assert(std::is_same_v<Ptr const, PtrA const>, "Inconsistent integer types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _lu.Rows(), "Matrix size mismatch.");
  if (_level == 0) {
    auto srcValues = A.Values();
    auto dstValues = _lu.Values();
    MOCHI_ASSERT_VERBOSE(srcValues.size() == dstValues.size(), "Value array size mismatch.");
    MOCHI_ASSERT_VERBOSE(_lu.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(_lu.Indices() == A.Indices(), "Sparsity pattern mismatch.");
    std::copy(srcValues.begin(), srcValues.end(), dstValues.begin());
  } else {
    _lu.SetZero();
    _lu += A;
  }
  Factorize();
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void RelaxedILUPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::Factorize() {
  auto const luPtr = _lu.Pointers();
  auto const luIdx = _lu.Indices();
  auto luVal = _lu.Values();
  auto constexpr kDummyFlag = static_cast<Ptr>(-1);
  DynamicArray<NonConstPtr> iw(_lu.Rows(), kDummyFlag);
  //
  // Algorithm I K J
  //
  for (NonConstIdx i = 0; i < _lu.Rows(); ++i) {
    //--- Get inverse map for luIdx
    [[maybe_unused]] auto colIdx = _lu.Indices(i);
    MOCHI_ASSERT_VERBOSE(
        std::is_sorted(colIdx.begin(), colIdx.end()), "Row does not have sorted column indices");
    for (NonConstPtr p = luPtr[i]; p < luPtr[i + 1]; ++p) {
      iw[luIdx[p]] = p;
      if (luIdx[p] == i) {
        _uStart[i] = p;
      }
    }
    //-- Loop expects that the column indices are sorted
    NonConstPtr p = luPtr[i];
    NonConstScalar sum{};
    for (; p < _uStart[i]; ++p) {
      auto k = luIdx[p];
      // Store "a_{ik} / a_{kk}"
      auto const tl = luVal[p] * luVal[_uStart[k]];
      luVal[p] = tl;
      //
      // TODO: SIMD Operations?
      //
      if (_alpha) {
        for (NonConstPtr pk = _uStart[k] + 1; pk < luPtr[k + 1]; ++pk) {
          auto const pi = iw[luIdx[pk]];
          if (pi == kDummyFlag) {
            sum += tl * luVal[pk];
          } else {
            // Update "a_{ij} = a_{ij} - (a_{ik} / a_{kk}) a_{kj}"
            luVal[pi] -= tl * luVal[pk];
          }
        }
      } else {
        for (NonConstPtr pk = _uStart[k] + 1; pk < luPtr[k + 1]; ++pk) {
          auto const pi = iw[luIdx[pk]];
          // Update "a_{ij} = a_{ij} - (a_{ik} / a_{kk}) a_{kj}"
          if (pi != kDummyFlag) {
            luVal[pi] -= tl * luVal[pk];
          }
        }
      }
    }
    if (_alpha) {
      luVal[p] -= _alpha * sum;
    }
    if (luVal[p] == Scalar{0})
      MOCHI_UNLIKELY {
        MOCHI_LOG_ERROR("Zero pivot at row %d.", i);
        break;
      }
    luVal[p] = NonConstScalar(1) / luVal[p];
    //--- Reset entries of iw
    for (p = luPtr[i]; p < luPtr[i + 1]; ++p) {
      iw[luIdx[p]] = kDummyFlag;
    }
  }
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename Input, typename Output>
void RelaxedILUPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::operator()(
    Input const& xin,
    Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_lu.Rows(), xin, yout);
  //
  //--- Convert input and output to MatrixView
  //
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
  y = x;
  // Solve "(I + L) z = x"
  auto luPtr = _lu.Pointers();
  auto luVal = _lu.Values();
  auto luCol = _lu.Indices();
  //
  for (int ir = 0; ir < y.Rows(); ++ir) {
    auto spanCol = luCol.subspan(luPtr[ir], int(_uStart[ir] - luPtr[ir]));
    auto spanVal = luVal.subspan(luPtr[ir], int(_uStart[ir] - luPtr[ir]));
    // We assume that `x` and `y` have the same orientation.
    // Call appropriate kernel depending on the orientation of `y`
    if constexpr (details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
      for (int jc = 0; jc < y.Cols(); ++jc) {
        NonConstScalar sum = 0;
        mochi::details::AddRowSparseTimesColumnVector<NonConstScalar, NonConstIdx>(
            spanCol, spanVal, y.Col(jc).Data(), sum);
        y(ir, jc) -= sum;
      }
    } else {
      for (int jc = 0; jc < y.Cols(); ++jc) {
        NonConstScalar sum = 0;
        mochi::details::AddRowSparseTimesVector<NonConstScalar, NonConstIdx>(
            spanCol, spanVal, y.Col(jc), sum);
        y(ir, jc) -= sum;
      }
    }
  }
  // Solve "U y = z"
  int ir = _lu.Rows() - 1;
  y.Row(ir) *= luVal[_uStart[ir]];
  for (ir = _lu.Rows() - 2; ir >= 0; --ir) {
    auto const upperPos = _uStart[ir] + 1;
    auto const len = int(luPtr[ir + 1] - upperPos);
    auto spanCol = luCol.subspan(upperPos, len);
    auto spanVal = luVal.subspan(upperPos, len);
    // We assume that `x` and `y` have the same orientation.
    // Call appropriate kernel depending on the orientation of `x`
    if constexpr (details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
      for (int jc = 0; jc < y.Cols(); ++jc) {
        NonConstScalar sum = 0;
        mochi::details::AddRowSparseTimesColumnVector<NonConstScalar, NonConstIdx>(
            spanCol, spanVal, y.Col(jc).Data(), sum);
        y(ir, jc) -= sum;
      }
    } else {
      for (int jc = 0; jc < y.Cols(); ++jc) {
        NonConstScalar sum = 0;
        mochi::details::AddRowSparseTimesVector<NonConstScalar, NonConstIdx>(
            spanCol, spanVal, y.Col(jc), sum);
        y(ir, jc) -= sum;
      }
    }
    y.Row(ir) *= luVal[_uStart[ir]];
  }
}

} // namespace mochi::krylov
