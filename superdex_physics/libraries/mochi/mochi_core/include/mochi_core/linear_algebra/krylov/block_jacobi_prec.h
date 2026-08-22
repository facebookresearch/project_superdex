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

#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/span.h>

#include <type_traits>

namespace mochi::krylov {

template <typename Scalar, int kPrecBlockSize>
struct BlockJacobiPrec final : Preconditioner<Scalar> {
  static_assert(!std::is_const_v<Scalar>, "Implementation assumes Scalar is non-const.");
  static_assert(kPrecBlockSize > 0, "Preconditioner block size must be positive");
  static constexpr auto kType =
      (kPrecBlockSize > 1) ? PreconditionerType::BlockJacobi : PreconditionerType::Jacobi;

  template <typename MatrixType>
  explicit BlockJacobiPrec(MatrixType const& A);

  /// @brief Apply the preconditioner to a column vector (or a set of column vectors).
  /// @param[in] x Input column vector(s)
  /// @param[out] Px Output column vector(s)
  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& Px) const;

  /// @brief Apply the preconditioner to a column vector.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<Scalar> Px) const override {
    operator()(x, Px);
  }

  /// @brief Concurrent application of the preconditioner to a column vector by a pool of workers.
  /// The calling worker is responsible for applying its own preconditioner contribution.
  /// @param[in] x Input column vector.
  /// @param[out] Px Output column vector.
  /// @param[in] data Parallel information for each worker.
  /// @note The start and end rows must be a multiple of the preconditioner block size.
  void ConcurrentSolve(
      ColumnVectorView<Scalar const> x,
      ColumnVectorView<Scalar> Px,
      ParallelWorkerInfo const& data) const override;

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

  template <typename MatrixType>
  void Update(MatrixType const& A);

 private:
  using DiagType = std::
      conditional_t<kPrecBlockSize == 1, Scalar, Matrix<Scalar, kPrecBlockSize, kPrecBlockSize>>;

  /// @brief Vector of diagonal blocks inverted
  /// @note The number of blocks times the block size yields the input matrix size.
  DynamicArray<DiagType> _inverseDiagBlocks;
};

//
//--- Implementation of functions
//

template <typename Scalar, int kPrecBlockSize>
template <typename MatrixType>
BlockJacobiPrec<Scalar, kPrecBlockSize>::BlockJacobiPrec(MatrixType const& A) {
  Update(A);
}

template <typename Scalar, int kPrecBlockSize>
template <typename MatrixType>
void BlockJacobiPrec<Scalar, kPrecBlockSize>::Update(MatrixType const& A) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Only square matrices are supported");
  MOCHI_ASSERT_VERBOSE(
      A.Rows() % kPrecBlockSize == 0,
      "Incompatible pairing of matrix size (%d) and block size (%d)",
      A.Rows(),
      kPrecBlockSize);
  if constexpr (kIsResizeNoInitSafe<DiagType>) {
    _inverseDiagBlocks.resize_noinit(A.Rows() / kPrecBlockSize);
  } else {
    // TODO: Investigate if it's safe to use resize_noinit with compile-time matrices and, if so,
    // specialize kIsResizeNoInitSafe to use the faster codepath.
    _inverseDiagBlocks.resize(A.Rows() / kPrecBlockSize);
  }
  if constexpr (kPrecBlockSize == 1) {
    ExtractDiagonal(A, MakeSpan(_inverseDiagBlocks));
    ArrayInverts(MakeSpan(_inverseDiagBlocks));
  } else {
    ExtractBlockDiagonal(A, MakeSpan(_inverseDiagBlocks));
    BatchedInverse(MakeSpan(_inverseDiagBlocks));
  }
}

template <typename Scalar, int kPrecBlockSize>
template <typename Input, typename Output>
void BlockJacobiPrec<Scalar, kPrecBlockSize>::operator()(Input const& x, Output&& Px) const {
  int const opN = kPrecBlockSize * isize(_inverseDiagBlocks);
  Preconditioner<Scalar>::ValidateInputOutput(opN, x, Px);
  ApplyBlockDiagonal<Scalar>(MakeConstSpan(_inverseDiagBlocks), x, Px);
}

template <typename Scalar, int kPrecBlockSize>
void BlockJacobiPrec<Scalar, kPrecBlockSize>::ConcurrentSolve(
    ColumnVectorView<Scalar const> x,
    ColumnVectorView<Scalar> Px,
    ParallelWorkerInfo const& data) const {
  int const opN = kPrecBlockSize * isize(_inverseDiagBlocks);
  Preconditioner<Scalar>::ValidateInputOutput(opN, x, Px);
  MOCHI_ASSERT_VERBOSE(
      data.rBegin >= 0 && data.rEnd <= x.Rows() && data.rBegin <= data.rEnd, "Invalid row range");
  auto const rowBegin = data.rBegin;
  auto const rowEnd = data.rEnd;
  MOCHI_ASSERT(
      rowBegin % kPrecBlockSize == 0 && rowEnd % kPrecBlockSize == 0,
      "The start and end rows must be a multiple of the preconditioner block size");
  auto const numRows = rowEnd - rowBegin;
  if (numRows == 0)
    MOCHI_UNLIKELY {
      return;
    }
  Span<DiagType const> invD(
      &_inverseDiagBlocks[rowBegin / kPrecBlockSize], numRows / kPrecBlockSize);
  ApplyBlockDiagonal<Scalar>(
      invD, x.MiddleRows(rowBegin, numRows), Px.MiddleRows(rowBegin, numRows));
}

template <typename Scalar>
using JacobiPrec = BlockJacobiPrec<Scalar, 1>;

} // namespace mochi::krylov
