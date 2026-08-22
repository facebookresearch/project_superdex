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
#include <limits>
#include <type_traits>

namespace mochi::krylov {

/// @brief Empty definition of incomplete Cholesky for non-supported input matrix.
template <typename MatrixType>
struct IncompleteCholeskyPrec {
  IncompleteCholeskyPrec() = delete;
};

/// @brief Implementation of incomplete Cholesky for BlockSparseMatrix input
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
struct IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>
    : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  static constexpr auto kType = PreconditionerType::IC0;

  /// @brief Constructor
  ///
  /// @param[in] A Input block sparse matrix
  /// @param[in] level Level of fill-in
  /// @param[in] alphaShift Scaling factor for the shift
  ///
  /// @note When shifting the input matrix, the incomplete factor is the one for A + alphaShift *
  /// (trace(A) / N) * I where N is the dimension of A.
  /// @note Only the upper triangular part of the input is used.
  explicit IncompleteCholeskyPrec(
      BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& A,
      int level,
      Scalar alphaShift);

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors).
  /// @param[in] x Input column vector(s)
  /// @param[out] y Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const;

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
    MOCHI_ASSERT(false, "ConcurrentSolve not supported for incomplete Cholesky preconditioner.");
  }

  constexpr PreconditionerType GetType() const override {
    MOCHI_ASSERT(_level == 0, "IncompleteCholeskyPrec::GetType assumes IC0.");
    return kType;
  }

  /// @brief Update the preconditioner from a block sparse matrix with the same sparsity pattern.
  /// @param[in] A Input block sparse matrix.
  /// @note The sparsity of the fill level is NOT recomputed. Only the values are updated.
  template <
      typename ScalarA,
      typename CRIdxA,
      typename PtrA,
      template <typename, typename...> typename StorageA>
  void Update(BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A);

 protected:
  /// @brief Perform the incomplete Cholesky factorization.
  /// @warning Assumes that _rChol has already been initialized with the matrix values.
  void Factorize();

  /// @brief Storage for the incomplete factor R
  /// - The upper triangular part is calculated
  /// - The lower triangular part is obtained by symmetry
  /// - The diagonal entries are inverted for faster solve
  BlockSparseMatrix<NonConstScalar, kBlockSize, NonConstIdx, NonConstPtr> _rChol;

  /// @brief Array pointing to the start of the upper part (inc. diagonal) for each block row
  /// The value for each block row is a "local" integer (local to the block row)
  DynamicArray<NonConstPtr> _uStart;

  /// @brief Scalar for diagonal shift (A + _alpha * trace(A) / N * I)
  /// _alpha = 0 yields no shift
  NonConstScalar _alpha;

  /// @brief Original shift factor passed to the constructor.
  NonConstScalar _alphaShift;

  /// @brief Fill-in level.
  int _level = 0;
};

/// @brief Implementation of incomplete Cholesky for Matrix input.
///
/// @tparam Scalar  Scalar type for input matrix
/// @tparam kRowsAtCompileTime
/// @tparam kColsAtCompileTime
/// @tparam kMajorDir Storage direction for input matrix
/// @tparam kOwnership
/// @tparam kMajorDim
///
/// @note Only the fill-in level 0 is implemented.
/// @note The incomplete factor is identical to the one obtained from the sparse representation of
/// the pruned matrix (i.e. by removing the zeros).
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
struct IncompleteCholeskyPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>
    : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  static constexpr auto kType = PreconditionerType::IC0;

  /// @brief Constructor
  ///
  /// @param[in] A Input sparse matrix
  /// @param[in] level Level of fill-in
  /// @param[in] alphaShift Scaling factor for the shift
  ///
  /// @note When shifting the input matrix, the incomplete factor is the one for A + alphaShift *
  /// (trace(A) / N) * I where N is the dimension of A.
  /// @note The incomplete factor is identical to the one obtained from the sparse representation of
  /// the pruned matrix (i.e. by removing the zeros).
  /// @note Only the upper triangular part of the input matrix is used.
  explicit IncompleteCholeskyPrec(
      Matrix<
          Scalar,
          kRowsAtCompileTime,
          kColsAtCompileTime,
          kMajorDir,
          kOwnership,
          kMajorDim> const& A,
      int level,
      Scalar alphaShift);

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
    MOCHI_ASSERT(false, "ConcurrentSolve not supported for incomplete Cholesky preconditioner.");
  }

  constexpr PreconditionerType GetType() const override {
    MOCHI_ASSERT(_level == 0, "IncompleteCholeskyPrec::GetType assumes IC0.");
    return kType;
  }

  /// @brief Update the preconditioner from a matrix with the same size.
  /// @param[in] A Input matrix.
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
  /// @brief Perform the incomplete Cholesky factorization.
  /// @warning Assumes that _rIC has already been initialized with the matrix values.
  void Factorize();

  /// @brief Storage for the incomplete factors R^T and R
  /// - The lower triangular matrix R^T is stored in the lower part of _rIC
  /// - The upper triangular matrix R is stored in the upper part of _rIC
  RowMatrix<NonConstScalar, kRowsAtCompileTime, kColsAtCompileTime> _rIC;

  /// @brief Scalar for diagonal shift (A + _alpha * trace(A) / N * I)
  NonConstScalar _alpha;

  /// @brief Original shift factor passed to the constructor.
  NonConstScalar _alphaShift;

  /// @brief Fill-in level.
  int _level = 0;
};

/// @brief Implementation of incomplete Cholesky for SparseMatrix input
///
/// @note We use the CSR implementation from Saad (as a starting point)
/// https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf
/// (Section 10.3.2 -- p. 309 Fortran Code)
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
struct IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>
    : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  static constexpr auto kType = PreconditionerType::IC0;

  /// @brief Constructor
  ///
  /// @param[in] A Input sparse matrix
  /// @param[in] level Level of fill-in
  /// @param[in] alphaShift Scaling factor for the shift
  ///
  /// @note When shifting the input matrix, the incomplete factor is the one for A + alphaShift *
  /// (trace(A) / N) * I where N is the dimension of A.
  /// @note Only the upper triangular part of the input matrix is used.
  explicit IncompleteCholeskyPrec(
      SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A,
      int level,
      Scalar alphaShift);

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors).
  /// @param[in] x Input column vector(s)
  /// @param[out] y Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const;

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
    MOCHI_ASSERT(false, "ConcurrentSolve not supported for incomplete Cholesky preconditioner.");
  }

  constexpr PreconditionerType GetType() const override {
    MOCHI_ASSERT(_level == 0, "IncompleteCholeskyPrec::GetType assumes IC0.");
    return kType;
  }

  /// @brief Update the preconditioner from a sparse matrix with the same sparsity pattern.
  /// @param[in] A Input sparse matrix.
  /// @note The sparsity of the fill level is NOT recomputed. Only the values are updated.
  template <
      typename ScalarA,
      typename CRIdxA,
      typename PtrA,
      template <typename, typename...> typename StorageA>
  void Update(SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& A);

 protected:
  /// @brief Perform the incomplete Cholesky factorization.
  /// @warning Assumes that _rChol has already been initialized with the matrix values.
  void Factorize();

  /// @brief Storage for the incomplete factor R
  SparseMatrix<NonConstScalar, NonConstIdx, NonConstPtr> _rChol;

  /// @brief Array pointing to the start of the upper part (inc. diagonal) for each row
  /// The value for each row is a "global" integer
  DynamicArray<NonConstPtr> _uStart;

  /// @brief Scalar for diagonal shift (A + _alpha * trace(A) / N * I)
  NonConstScalar _alpha;

  /// @brief Original shift factor passed to the constructor.
  NonConstScalar _alphaShift;

  /// @brief Fill-in level.
  int _level = 0;
};

} // namespace mochi::krylov

//
//--- Implementation of class member functions
//

namespace mochi::krylov::details {

/// @brief Do a block-row update needed during a block Cholesky factorization
/// The steps involve a Cholesky factorization (computing an upper triangular matrix)
/// and a forward triangular solve along the block row.
///
/// @note We assume that the number of columns of R is a multiple of kBlockSize
/// @note The first (kBlockSize x kBlockSize) block in R comes from the diagonal block
/// in a block matrix.
template <int kBlockSize, typename NonConstScalar, int kLeadDim>
int UpdateBlockRow(RowMatrixView<NonConstScalar, kBlockSize, krylov::kDynamic, kLeadDim> R) {
  MOCHI_ASSERT_VERBOSE(
      (R.Cols() >= kBlockSize) && (R.Cols() % kBlockSize == 0), "Incompatible input matrix");
  //--- Do a Cholesky factorization of D
  RowMatrix<NonConstScalar, kBlockSize, kBlockSize> D =
      R.template Block<kBlockSize, kBlockSize>(0, 0, kBlockSize, kBlockSize);
  for (int i = 0; i < kBlockSize; ++i) {
    if ((D(i, i) < std::numeric_limits<NonConstScalar>::min()) || (!std::isfinite(D(i, i))))
      MOCHI_UNLIKELY {
        return i + 1;
      }
    D(i, i) = NonConstScalar(1) / Sqrt(D(i, i));
    D.template Block<1, krylov::kDynamic>(i, i + 1, 1, kBlockSize - i - 1) *= D(i, i);
    for (int k = i + 1; k < kBlockSize; ++k) {
      auto const d_ik = D(i, k);
      // Update D_{k,j} <- D_{k,j} - D_{k,i} D_{i,j}
      D.template Block<1, krylov::kDynamic>(k, k, 1, kBlockSize - k) -=
          d_ik * D.template Block<1, krylov::kDynamic>(i, k, 1, kBlockSize - k);
    }
  }
  //--- Solve triangular linear system with multiple columns in place
  R.Row(0) *= D(0, 0);
  for (int i = 1; i < kBlockSize; ++i) {
    auto Ri = R.Row(i);
    Ri -=
        D.template Block<krylov::kDynamic, 1>(0, i, i, 1).Transpose() * R.Block(0, 0, i, R.Cols());
    Ri *= D(i, i);
  }
  //--- Overwrite the diagonal of the block-diagonal in R
  auto R0 = R.template Block<kBlockSize, kBlockSize>(0, 0, kBlockSize, kBlockSize);
  for (int i = 0; i < kBlockSize; ++i) {
    R0(i, i) = D(i, i);
  }
  return 0;
}

} // namespace mochi::krylov::details

namespace mochi::krylov {

//
//--- Specialization for BlockSparseMatrix
//

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::
    IncompleteCholeskyPrec(
        BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& A,
        int level,
        Scalar alphaShift)
    : _uStart(A.BlockRows()), _alpha(alphaShift), _alphaShift(alphaShift), _level(level) {
  static_assert(
      std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
      "IncompleteCholeskyPrec requires signed matrix indices representable as int.");
  MOCHI_ASSERT_VERBOSE(_level >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT_VERBOSE(_alphaShift >= Scalar{0}, "Out-of-range shifting factor.")
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  mochi::details::ConvertToFillLevel(_level, A, _rChol);
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
void IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::Update(
    BlockSparseMatrix<ScalarA, kBlockSize, CRIdxA, PtrA, StorageA> const& A) {
  static_assert(std::is_same_v<Scalar const, ScalarA const>, "Inconsistent scalar types");
  static_assert(std::is_same_v<CRIdx const, CRIdxA const>, "Inconsistent integer types");
  static_assert(std::is_same_v<Ptr const, PtrA const>, "Inconsistent integer types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _rChol.Rows(), "Matrix size mismatch.");
  if (_level == 0) {
    auto srcValues = A.Values();
    auto dstValues = _rChol.Values();
    MOCHI_ASSERT_VERBOSE(srcValues.size() == dstValues.size(), "Value array size mismatch.");
    MOCHI_ASSERT_VERBOSE(_rChol.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(_rChol.Indices() == A.Indices(), "Sparsity pattern mismatch.");
    std::copy(srcValues.begin(), srcValues.end(), dstValues.begin());
  } else {
    _rChol.SetZero();
    _rChol += A;
  }
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  Factorize();
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void IncompleteCholeskyPrec<
    BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::Factorize() {
  auto const rPtr = _rChol.Pointers();
  for (NonConstIdx i = 0; i < _rChol.BlockRows(); ++i) {
    auto const cIdx = _rChol.Indices(i);
    auto values = _rChol.Values(i);
    MOCHI_ASSERT_VERBOSE(
        std::is_sorted(cIdx.begin(), cIdx.end()), "Column entries need to be sorted");
    auto ptr = std::lower_bound(cIdx.begin(), cIdx.end(), i);
    MOCHI_ASSERT_VERBOSE((ptr != cIdx.end()) && (*ptr == i), "Missing diagonal block");
    auto const k = static_cast<int>(ptr - cIdx.begin());
    _uStart[i] = k + rPtr[i];
    if (_alpha > NonConstScalar(0)) {
      auto mat = values[k];
      for (int ii = 0; ii < kBlockSize; ++ii) {
        mat(ii, ii) += _alpha;
      }
    }
  }
  //
  auto const dummyFlag = static_cast<NonConstPtr>(-1);
  auto zero = Matrix<NonConstScalar, kBlockSize, kBlockSize>::Zero();
  DynamicArray<NonConstPtr> iw(_rChol.BlockRows(), dummyFlag);
  NonConstIdx info = 0;
  //
  auto const rIdx = _rChol.Indices();
  for (NonConstIdx i = 0; i < _rChol.BlockRows(); ++i) {
    //--- Do a block-row update
    auto values = _rChol.Values(i);
    {
      auto start1 = (_uStart[i] - rPtr[i]) * kBlockSize;
      auto size = values.Underlying().Cols() - start1;
      info = krylov::details::UpdateBlockRow(
          values.Underlying().template Block<kBlockSize, krylov::kDynamic>(
              0, start1, kBlockSize, size));
      if (info != 0)
        MOCHI_UNLIKELY {
          info = i + 1;
          break;
        }
    }
    //--- Get inverse map of colIdx for the upper "triangular" part
    auto const colIdx = _rChol.Indices(i);
    for (int j = int(_uStart[i] - rPtr[i]); j < isize(colIdx); ++j) {
      iw[colIdx[j]] = NonConstPtr(j);
    }
    auto const lastBlockCol = colIdx.back();
    //--- Use 'p' as pointer integer
    //--- TODO Explore whether a `ParallelFor` can accelerate the computation
    NonConstPtr p = _uStart[i] + 1;
    for (; p < rPtr[i + 1]; ++p) {
      auto k = rIdx[p]; // <- Non-zero block column index in the block row 'i' (with k > i)
      auto rk = _rChol.Values(k);
      auto const rik_t = values[p - rPtr[i]].Transpose();
      //--- Explore the "upper part" block row 'k'
      //--- only for the blocks that could be updated by block row 'i'
      NonConstPtr pos = _uStart[k], localPos = _uStart[k] - rPtr[k];
      NonConstIdx j;
      for (; (pos < rPtr[k + 1]) && ((j = rIdx[pos]) <= lastBlockCol); pos += 1, localPos += 1) {
        auto jp = iw[j]; // <- Check whether block column 'j' is present in the block row 'i'
        // r_{kj} <- r_{kj} - r_{ik}^T r_{ij}
        rk[localPos] -= (jp == dummyFlag) ? zero : rik_t * values[jp];
      }
    }
    for (int j = int(_uStart[i] - rPtr[i]); j < isize(colIdx); ++j) {
      iw[colIdx[j]] = dummyFlag;
    }
  }
  if (info != 0) {
    MOCHI_LOG_ERROR("Non positive pivot at block %d", info - 1);
  }
  //
  /// Symmetrize the block sparse matrix
  //
  DynamicArray<int> countBlocks(_rChol.BlockRows(), 0);
  for (int i = 0; i < _rChol.BlockRows(); ++i) {
    auto const rColIdx = _rChol.Indices(i);
    for (int k = int(_uStart[i] - rPtr[i]) + 1; k < isize(rColIdx); ++k) {
      countBlocks[rColIdx[k]] += 1;
    }
  }
  DynamicArray<
      DynamicArray<RowMatrixView<NonConstScalar, kBlockSize, kBlockSize, krylov::kDynamic>>>
      transposeGraph(_rChol.BlockCols());
  for (int i = 0; i < _rChol.BlockRows(); ++i) {
    transposeGraph[i].reserve(countBlocks[i]);
  }
  for (int i = 0; i < _rChol.BlockRows(); ++i) {
    auto const rColIdx = _rChol.Indices(i);
    auto rValues = _rChol.Values(i);
    for (int k = int(_uStart[i] - rPtr[i]) + 1; k < isize(rColIdx); ++k) {
      transposeGraph[rColIdx[k]].push_back(rValues[k]);
    }
  }
  //
  // Fill the lower triangular part
  //
  for (int i = 0; i < _rChol.BlockRows(); ++i) {
    auto const& list = transposeGraph[i];
    MOCHI_ASSERT_VERBOSE(list.size() == _uStart[i] - rPtr[i], "Incompatible size");
    int k = 0;
    [[maybe_unused]] auto const rColIdx = _rChol.Indices(i);
    auto rValues = _rChol.Values(i);
    for (auto rval : list) {
      MOCHI_ASSERT_VERBOSE(rColIdx[k] <= i, "Incorrect location");
      rValues[k++] = Transpose(rval);
    }
    //--- Symmetrize the diagonal block
    auto diag = rValues[_uStart[i] - rPtr[i]];
    for (int ii = 0; ii < kBlockSize; ++ii) {
      for (int jj = 0; jj < ii; ++jj) {
        diag(ii, jj) = diag(jj, ii);
      }
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
void IncompleteCholeskyPrec<BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>>::operator()(
    Input const& xin,
    Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_rChol.Rows(), xin, yout);
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
          _rChol.MaxNnzPerRow()));
  [[maybe_unused]] ColumnVector<NonConstScalar, kBlockSize> Ly;
  [[maybe_unused]] auto aLy = mochi::details::GetAccessor(Ly);
  //
  auto rPtr = _rChol.Pointers();
  //
  for (int jc = 0; jc < x.Cols(); ++jc) {
    auto yjc = y.Col(jc);
    [[maybe_unused]] auto aY = mochi::details::GetAccessor(yjc);
    for (int irb = 0; irb < _rChol.BlockRows(); ++irb) {
      auto colIdx = _rChol.Indices(irb);
      auto values = _rChol.Values(irb);
      auto yij = yjc.MiddleRows(irb * kBlockSize, kBlockSize);
      auto const localShift = _uStart[irb] - rPtr[irb];
      if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::RowMultiplier<Scalar, kBlockSize>::ApplyToColVector(
            colIdx.data(),
            values.data(),
            localShift,
            values.LeadDim(),
            aY,
            aLy,
            /*br*/ 0,
            /*c*/ 0,
            xTmp,
            xTmpRequiredSize);
        yij -= Ly;
      } else {
        static_assert(mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
        for (int kk = 0; kk + rPtr[irb] < _uStart[irb]; ++kk) {
          yij -=
              values[kk] * yjc.template MiddleRows<kBlockSize>(colIdx[kk] * kBlockSize, kBlockSize);
        }
      }
      auto const& D = values[localShift];
      //
      // TODO Explore whether kernels from `LDLt.h` can be re-used
      //
      for (int i = 0; i < kBlockSize; ++i) {
        auto v = yij.Row(i);
        for (int j = 0; j < i; ++j) {
          v -= D(i, j) * yij.Row(j);
        }
        v *= D(i, i);
      }
    }
    //
    for (int irb = _rChol.BlockRows() - 1; irb >= 0; --irb) {
      auto colIdx = _rChol.Indices(irb);
      auto values = _rChol.Values(irb);
      Ly.SetZero();
      auto yij = yjc.template MiddleRows<kBlockSize>(irb * kBlockSize, kBlockSize);
      auto const localShift = _uStart[irb] - rPtr[irb];
      if constexpr (mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::ColMajor) {
        mochi::details::RowMultiplier<Scalar, kBlockSize>::ApplyToColVector(
            colIdx.data() + localShift + 1,
            values.data() + (localShift + 1) * kBlockSize,
            isize(colIdx) - (localShift + 1),
            values.LeadDim(),
            aY,
            aLy,
            /*br*/ 0,
            /*c*/ 0,
            xTmp,
            xTmpRequiredSize);
        yij = yij - Ly;
      } else {
        static_assert(mochi::details::MatTraits<Output>::kMajorDir == krylov::Direction::RowMajor);
        for (int kk = localShift + 1; kk < colIdx.size(); ++kk) {
          yij -=
              values[kk] * yjc.template MiddleRows<kBlockSize>(colIdx[kk] * kBlockSize, kBlockSize);
        }
      }
      //
      // TODO Explore whether kernels from `LDLt.h` can be re-used
      //
      auto const& D = values[localShift];
      for (int i = kBlockSize - 1; i >= 0; --i) {
        auto v = yij.Row(i);
        for (int j = i + 1; j < kBlockSize; ++j) {
          v -= D(i, j) * yij.Row(j);
        }
        v *= D(i, i);
      }
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
IncompleteCholeskyPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
    IncompleteCholeskyPrec(
        Matrix<
            Scalar,
            kRowsAtCompileTime,
            kColsAtCompileTime,
            kMajorDir,
            kOwnership,
            kMajorDim> const& A,
        int level,
        Scalar alphaShift)
    : _alpha(alphaShift), _alphaShift(alphaShift), _level(level) {
  MOCHI_ASSERT_VERBOSE(_level >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT_VERBOSE(_alphaShift >= Scalar{0}, "Out-of-range shifting factor.")
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  mochi::details::ConvertToFillLevel(_level, A, _rIC);
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
void IncompleteCholeskyPrec<
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
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _rIC.Rows(), "Matrix size mismatch.");
  _rIC = A;
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  Factorize();
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
void IncompleteCholeskyPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
    Factorize() {
  if (_alpha > Scalar{0}) {
    for (int i = 0; i < _rIC.Rows(); ++i) {
      _rIC(i, i) += _alpha;
    }
  }
  //
  int info = 0;
  auto const n = _rIC.Rows();
  for (int i = 0; i < n; ++i) {
    // Test for zero or negative entry on the diagonal
    if (_rIC(i, i) < std::numeric_limits<NonConstScalar>::min())
      MOCHI_UNLIKELY {
        info = i + 1;
        break;
      }
    //
    _rIC(i, i) = Sqrt(_rIC(i, i));
    //
    auto const scaling = NonConstScalar(1.0) / _rIC(i, i);
    _rIC.template Block<1, krylov::kDynamic>(i, i + 1, 1, n - i - 1) *= scaling;
    //
    for (int k = i + 1; k < n; ++k) {
      if (_rIC(i, k) == Scalar{0})
        MOCHI_UNLIKELY {
          continue;
        }
      auto const rik_t = _rIC(i, k);
      for (int j = k; j < n; ++j) {
        if (Abs(_rIC(k, j)) < std::numeric_limits<NonConstScalar>::min())
          MOCHI_UNLIKELY {
            continue;
          }
        _rIC(k, j) -= rik_t * _rIC(i, j);
      }
    }
  }
  if (info != 0) {
    MOCHI_LOG_ERROR("Non-positive pivot at position %d", info - 1);
  }
  //
  // Symmetrize _rIC
  //
  for (int i = 0; i < _rIC.Rows(); ++i) {
    _rIC.template Block<1, krylov::kDynamic>(i, 0, 1, i) =
        _rIC.template Block<krylov::kDynamic, 1>(0, i, i, 1).Transpose();
  }
  //
  for (int i = 0; i < _rIC.Rows(); ++i) {
    _rIC(i, i) = Scalar(1) / _rIC(i, i);
  }
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
template <typename Input, typename Output>
void IncompleteCholeskyPrec<
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim>>::
operator()(Input const& xin, Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_rIC.Rows(), xin, yout);
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
  y = x;
  for (int ii = 0; ii < y.Rows(); ++ii) {
    y.Row(ii) -= _rIC.template Block<1, krylov::kDynamic>(ii, 0, 1, ii) * y.TopRows(ii);
    y.Row(ii) *= _rIC(ii, ii);
  }
  y.Row(y.Rows() - 1) *= _rIC(y.Rows() - 1, y.Rows() - 1);
  for (int ii = y.Rows() - 2; ii >= 0; --ii) {
    y.Row(ii) -= _rIC.template Block<1, krylov::kDynamic>(ii, ii + 1, 1, y.Rows() - ii - 1) *
        y.MiddleRows(ii + 1, y.Rows() - ii - 1);
    y.Row(ii) *= _rIC(ii, ii);
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
IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::IncompleteCholeskyPrec(
    SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A,
    int level,
    Scalar alphaShift)
    : _uStart(A.Rows()), _alpha(alphaShift), _alphaShift(alphaShift), _level(level) {
  static_assert(
      std::signed_integral<CRIdx> && sizeof(CRIdx) <= sizeof(int),
      "IncompleteCholeskyPrec requires signed matrix indices representable as int.");
  MOCHI_ASSERT_VERBOSE(_level >= 0, "Fill-in level must not be negative.");
  MOCHI_ASSERT_VERBOSE(_alphaShift >= Scalar{0}, "Out-of-range shifting factor.")
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  mochi::details::ConvertToFillLevel(_level, A, _rChol);
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
void IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::Update(
    SparseMatrix<ScalarA, CRIdxA, PtrA, StorageA> const& A) {
  static_assert(std::is_same_v<Scalar const, ScalarA const>, "Inconsistent scalar types");
  static_assert(std::is_same_v<CRIdx const, CRIdxA const>, "Inconsistent integer types");
  static_assert(std::is_same_v<Ptr const, PtrA const>, "Inconsistent integer types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(A.Rows() == _rChol.Rows(), "Matrix size mismatch.");
  if (_level == 0) {
    auto srcValues = A.Values();
    auto dstValues = _rChol.Values();
    MOCHI_ASSERT_VERBOSE(srcValues.size() == dstValues.size(), "Value array size mismatch.");
    MOCHI_ASSERT_VERBOSE(_rChol.Pointers() == A.Pointers(), "Sparsity pattern mismatch.");
    MOCHI_ASSERT_VERBOSE(_rChol.Indices() == A.Indices(), "Sparsity pattern mismatch.");
    std::copy(srcValues.begin(), srcValues.end(), dstValues.begin());
  } else {
    _rChol.SetZero();
    _rChol += A;
  }
  if (_alphaShift > Scalar{0}) {
    _alpha = _alphaShift * (Trace(A) / A.Rows());
  }
  Factorize();
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
void IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::Factorize() {
  auto const rPtr = _rChol.Pointers();
  auto const rIdx = _rChol.Indices();
  auto rVal = _rChol.Values();
  //
  // Extract position of "diagonal" & shift diagonal if needed
  //
  for (NonConstIdx i = 0; i < _rChol.Rows(); ++i) {
    auto const cIdx = _rChol.Indices(i);
    auto rValues = _rChol.Values(i);
    MOCHI_ASSERT_VERBOSE(
        std::is_sorted(cIdx.begin(), cIdx.end()), "Column entries need to be sorted");
    auto ptr = std::lower_bound(cIdx.begin(), cIdx.end(), i);
    MOCHI_ASSERT_VERBOSE((ptr != cIdx.end()) && (*ptr == i), "Diagonal entry not found");
    auto const p = static_cast<int>(ptr - cIdx.begin());
    _uStart[i] = p + rPtr[i];
    if (_alpha > Scalar{0}) {
      rValues[p] += _alpha;
    }
  }
  //
  // Algorithm IKJ
  // I = row index
  // K = col index (K < I)
  // J = col index (J > I)
  //
  auto const dummyFlag = static_cast<NonConstPtr>(-1);
  DynamicArray<NonConstPtr> iw(_rChol.Rows(), dummyFlag);
  NonConstIdx info = 0;
  for (NonConstIdx i = 0; i < _rChol.Rows(); ++i) {
    // Test for zero or negative entry on the diagonal
    if (rVal[_uStart[i]] < std::numeric_limits<NonConstScalar>::min())
      MOCHI_UNLIKELY {
        info = i + 1;
        break;
      }
    //--- Get inverse map of rIdx only for the upper "triangular" part
    for (NonConstPtr p = _uStart[i]; p < rPtr[i + 1]; ++p) {
      iw[rIdx[p]] = p;
    }
    //
    RowVectorView<NonConstScalar> ri(&rVal[_uStart[i]], rPtr[i + 1] - _uStart[i]);
    ri *= NonConstScalar(1.0) / Sqrt(rVal[_uStart[i]]);
    //--- Use 'p' as pointer integer along the row 'i'
    //--- TODO Explore whether a `ParallelFor` can accelerate the computation
    NonConstPtr p = _uStart[i] + 1;
    // Extraction of _uStart guarantees that the row has, at least, one entry
    auto const lastCol = rIdx[rPtr[i + 1] - 1];
    for (; p < rPtr[i + 1]; ++p) {
      auto const k = rIdx[p];
      auto const rik_t = rVal[p];
      //--- Use 'pos' as pointer integer along the row 'k'
      NonConstPtr pos = _uStart[k];
      //--- Explore the "upper part" block row 'k'
      //--- only for the blocks that could be updated by block row 'i'
      for (; ((pos < rPtr[k + 1]) && (rIdx[pos] <= lastCol)); ++pos) {
        NonConstPtr jpos = iw[rIdx[pos]];
        // r_{kj} <- r_{kj} - r_{ik}^T (1 / r_{ii}) r_{ij}
        rVal[pos] -= (jpos == dummyFlag) ? Scalar(0) : rik_t * rVal[jpos];
      }
    }
    //--- Reset entries of iw
    for (p = _uStart[i]; p < rPtr[i + 1]; ++p) {
      iw[rIdx[p]] = dummyFlag;
    }
  } // for (NonConstIdx i = 0; i < _rChol.Rows(); ++i)
  //
  if (info != 0) {
    MOCHI_LOG_ERROR("Non-positive pivot at position %d", info - 1);
  }
  //
  // Symmetrize R
  //
  DynamicArray<int> countEntries(_rChol.Rows(), 0);
  for (int i = 0; i < _rChol.Rows(); ++i) {
    auto const rColIdx = _rChol.Indices(i);
    for (int k = int(_uStart[i] - rPtr[i]) + 1; k < isize(rColIdx); ++k) {
      countEntries[rColIdx[k]] += 1;
    }
  }
  DynamicArray<DynamicArray<NonConstScalar>> transposeGraph(_rChol.Cols());
  for (int i = 0; i < _rChol.Cols(); ++i) {
    transposeGraph[i].reserve(countEntries[i]);
  }
  for (int i = 0; i < _rChol.Rows(); ++i) {
    auto const rColIdx = _rChol.Indices(i);
    auto rValues = _rChol.Values(i);
    for (int k = int(_uStart[i] - rPtr[i]) + 1; k < isize(rColIdx); ++k) {
      transposeGraph[rColIdx[k]].push_back(rValues[k]);
    }
  }
  //
  // Fill the lower triangular part
  //
  for (int i = 0; i < _rChol.Rows(); ++i) {
    auto const& list = transposeGraph[i];
    MOCHI_ASSERT_VERBOSE(list.size() == _uStart[i] - rPtr[i], "Incompatible size");
    int k = 0;
    [[maybe_unused]] auto const rColIdx = _rChol.Indices(i);
    auto rValues = _rChol.Values(i);
    for (auto rval : list) {
      MOCHI_ASSERT_VERBOSE(rColIdx[k] <= i, "Incorrect location");
      rValues[k++] = rval;
    }
    //--- Invert the diagonal entry.
    rValues[_uStart[i] - rPtr[i]] = NonConstScalar(1) / rValues[_uStart[i] - rPtr[i]];
  }
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename Input, typename Output>
void IncompleteCholeskyPrec<SparseMatrix<Scalar, CRIdx, Ptr, Storage>>::operator()(
    Input const& xin,
    Output&& yout) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_rChol.Rows(), xin, yout);
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
  // Solve "R^T z = x"
  auto rPtr = _rChol.Pointers();
  auto rVal = _rChol.Values();
  auto rCol = _rChol.Indices();
  //
  for (int ir = 0; ir < y.Rows(); ++ir) {
    auto spanCol = rCol.subspan(rPtr[ir], int(_uStart[ir] - rPtr[ir]));
    auto spanVal = rVal.subspan(rPtr[ir], isize(spanCol));
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
    y.Row(ir) *= rVal[_uStart[ir]];
  }
  // Solve "R y = z"
  int ir = _rChol.Rows() - 1;
  y.Row(ir) *= rVal[_uStart[ir]];
  for (ir = _rChol.Rows() - 2; ir >= 0; --ir) {
    auto const upperPos = _uStart[ir] + 1;
    auto const len = int(rPtr[ir + 1] - upperPos);
    auto spanCol = rCol.subspan(upperPos, len);
    auto spanVal = rVal.subspan(upperPos, len);
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
    y.Row(ir) *= rVal[_uStart[ir]];
  }
}

} // namespace mochi::krylov
