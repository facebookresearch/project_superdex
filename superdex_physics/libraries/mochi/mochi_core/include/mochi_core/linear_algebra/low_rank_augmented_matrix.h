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

#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

#include <type_traits>
#include <utility>

namespace mochi {

/// @brief Linear operator representing a matrix A augmented with a low-rank update U * V^T.
/// @tparam MatrixType Matrix or linear operator type.
/// @details The operator is equivalent to A + U * V^T, where
/// A \in \mathbb{R}^{m \times n} is a matrix or linear operator, and
/// U \in \mathbb{R}^{m \times r} and V \in \mathbb{R}^{n \times r} are dense matrices representing
/// a rank-r update.
template <typename MatrixType>
struct LowRankAugmentedMatrix {
  static_assert(!IsCuda<MatrixType>, "CUDA matrices not supported");
  using Scalar = typename MatrixType::Scalar;
  using NonConstScalar = std::remove_const_t<Scalar>;

  LowRankAugmentedMatrix() = default;

  /// @brief Construct a LowRankAugmentedMatrix with an initial capacity for rank-one updates
  /// without specifying the rank-one updates at construction time. The rank-one updates can be
  /// specified after construction by calling 'AddRankOneUpdates' method.
  /// @param[in] A Unaugmented matrix.
  /// @param[in] initialCapacity Initial capacity for rank-one updates. The capacity can be modified
  /// after construction by calling 'Reserve' method. Default is 0.
  LowRankAugmentedMatrix(MatrixType const& A, int initialCapacity = 0) : _A(A) {
    _U.Resize(Rows(), initialCapacity);
    _V.Resize(Cols(), initialCapacity);
  }

  LowRankAugmentedMatrix(MatrixType&& A, int initialCapacity = 0) : _A(std::move(A)) {
    _U.Resize(Rows(), initialCapacity);
    _V.Resize(Cols(), initialCapacity);
  }

  /// @brief Construct a LowRankAugmentedMatrix from an unaugmented matrix and a set of rank-one
  /// updates.
  /// @param[in] A Unaugmented matrix.
  /// @param[in] U Dense matrix whose columns are the left vectors for the low-rank update.
  /// @param[in] V Dense matrix whose columns are the right vectors for the low-rank update.
  /// @note U and V must have the same number of columns.
  template <typename UType, typename VType>
  LowRankAugmentedMatrix(MatrixType const& A, UType&& U, VType&& V)
      : _A(A),
        _U(std::forward<decltype(U)>(U)),
        _V(std::forward<decltype(V)>(V)),
        _numRankOneUpdates(_U.Cols()) {
    MOCHI_ASSERT_VERBOSE(
        _U.Rows() == Rows() && _V.Rows() == Cols() && _U.Cols() == _V.Cols(),
        "Inconsistent sizes.");
  }

  template <typename UType, typename VType>
  LowRankAugmentedMatrix(MatrixType&& A, UType&& U, VType&& V)
      : _A(std::move(A)),
        _U(std::forward<decltype(U)>(U)),
        _V(std::forward<decltype(V)>(V)),
        _numRankOneUpdates(_U.Cols()) {
    MOCHI_ASSERT_VERBOSE(
        _U.Rows() == Rows() && _V.Rows() == Cols() && _U.Cols() == _V.Cols(),
        "Inconsistent sizes.");
  }

  /// @brief Reset this LowRankAugmentedMatrix using the arguments for any of its constructors.
  template <typename... Args>
  LowRankAugmentedMatrix& Reset(Args&&... args) {
    this->~LowRankAugmentedMatrix();
    new (this) LowRankAugmentedMatrix(std::forward<Args>(args)...);
    return *this;
  }

  /// @brief Reserve capacity for rank-one updates.
  void Reserve(int capacity) {
    MOCHI_ASSERT_VERBOSE(capacity >= 0, "Invalid capacity.");
    MOCHI_ASSERT_VERBOSE(_U.Cols() == _V.Cols(), "Inconsistent capacities.");
    if (capacity > _U.Cols()) {
      Matrix<NonConstScalar> Utmp(Rows(), capacity);
      Matrix<NonConstScalar> Vtmp(Cols(), capacity);
      Utmp.LeftCols(_numRankOneUpdates) = _U.LeftCols(_numRankOneUpdates);
      Vtmp.LeftCols(_numRankOneUpdates) = _V.LeftCols(_numRankOneUpdates);
      std::swap(_U, Utmp);
      std::swap(_V, Vtmp);
    }
  }

  /// @brief Add rank-one updates to the augmented matrix.
  /// @param[in] U Dense matrix whose columns are the left vectors for the new low-rank update.
  /// @param[in] V Dense matrix whose columns are the right vectors for the new low-rank update.
  /// @note U and V must have the same number of columns.
  template <typename UType, typename VType>
  void AddRankOneUpdates(UType const& U, VType const& V) {
    MOCHI_ASSERT_VERBOSE(
        U.Rows() == Rows() && V.Rows() == Cols() && U.Cols() == V.Cols(), "Inconsistent sizes.");
    if (_numRankOneUpdates + U.Cols() > _U.Cols()) {
      Reserve(Max(_numRankOneUpdates + U.Cols(), 2 * _U.Cols())); // At least double the capacity.
    }
    _U.MiddleCols(_numRankOneUpdates, U.Cols()) = U;
    _V.MiddleCols(_numRankOneUpdates, V.Cols()) = V;
    _numRankOneUpdates += U.Cols();
  }

  /// @brief Application of the augmented matrix to a dense matrix, including a column vector.
  /// @note The unaugmented matrix is applied in parallel if its type supports it (e.g., @ref
  /// IslandOperators). The low-rank term U (V^T x) is added on the calling thread.
  template <typename VectorIn, typename VectorOut>
  void Apply(VectorIn const& x, VectorOut&& Ax) const {
    krylov::Apply(_A, x, Ax);
    AddLowRankProductToRange(x, Ax, 0, Rows());
  }

  /// @brief Partial application of the augmented matrix to a dense matrix, including a column
  /// vector. Only the row range [rowBegin, rowEnd) of the output matrix is computed.
  /// @note Each call computes all of V^T x, so a product split into row ranges computes it once per
  /// range. Sharing it across the ranges of a parallel product has not been optimized yet.
  template <typename VectorIn, typename VectorOut, typename Idx>
  void ApplyToRange(VectorIn const& x, VectorOut&& Ax, Idx rowBegin, Idx rowEnd) const {
    krylov::ApplyToRange(_A, x, Ax, rowBegin, rowEnd);
    AddLowRankProductToRange(x, Ax, rowBegin, rowEnd);
  }

  auto Rows() const {
    return GetNumRows(_A);
  }

  auto Cols() const {
    return GetNumCols(_A);
  }

  int NumRankOneUpdates() const {
    return _numRankOneUpdates;
  }

  MatrixType const& GetUnaugmentedMatrix() const {
    return _A;
  }

  Matrix<NonConstScalar> GetAugmentedMatrix() const {
    Matrix<NonConstScalar> augMat = ToMatrix(_A);
    if (_numRankOneUpdates > 0) {
      augMat += _U.LeftCols(_numRankOneUpdates) * _V.LeftCols(_numRankOneUpdates).Transpose();
    }
    return augMat;
  }

 private:
  // Adds rows [rowBegin, rowEnd) of U (V^T x), where U is m x r, V is n x r, x is n x k and r =
  // _numRankOneUpdates. The r x k temporary V^T x keeps the cost at ~2 (numRows + n) r k FLOPs.
  template <typename VectorIn, typename VectorOut, typename Idx>
  void AddLowRankProductToRange(VectorIn const& x, VectorOut&& Ax, Idx rowBegin, Idx rowEnd) const {
    if (_numRankOneUpdates > 0) {
      Idx const numRows = rowEnd - rowBegin;
      Ax.MiddleRows(rowBegin, numRows) += _U.Block(rowBegin, 0, numRows, _numRankOneUpdates) *
          (_V.LeftCols(_numRankOneUpdates).Transpose() * x);
    }
  }

  MatrixType _A;
  Matrix<NonConstScalar> _U = {};
  Matrix<NonConstScalar> _V = {};
  int _numRankOneUpdates = 0;
};

/// @brief Approximate number of FLOPs to apply the augmented matrix to a column vector.
template <typename T>
inline auto FlopsPerApply(LowRankAugmentedMatrix<T> const& A) {
  return FlopsPerApply(A.GetUnaugmentedMatrix()) +
      2 * (A.Rows() + A.Cols()) * A.NumRankOneUpdates();
}

template <typename T>
inline auto GetRowRangesPerWorker(LowRankAugmentedMatrix<T> const& A, int numWorkers) {
  // Load balancing is based on the unaugmented matrix. It may not be optimal in corner cases.
  return GetRowRangesPerWorker(A.GetUnaugmentedMatrix(), numWorkers);
}

} // namespace mochi

namespace mochi::details {
template <typename T>
constexpr bool IsLowRankAugmentedMatrixDef<LowRankAugmentedMatrix<T>> = true;
} // namespace mochi::details
