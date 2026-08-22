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
#include <mochi_core/linear_algebra/ldlt.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>

#include <type_traits>

namespace mochi::krylov {

/** @brief Preconditioner based on the inverse of a symmetric matrix. */
template <typename Scalar>
struct SymInversePrec final : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  static constexpr auto kType = PreconditionerType::SymInverse;

  template <typename MatrixType>
  explicit SymInversePrec(MatrixType const& A);

  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const;

  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<NonConstScalar> Px) const override {
    operator()(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<Scalar const> x,
      ColumnVectorView<NonConstScalar> Px,
      ParallelWorkerInfo const& data) const override {
    Preconditioner<NonConstScalar>::ValidateInputOutput(_inverse.Rows(), x, Px);
    auto const rowBegin = data.rBegin;
    auto const rowEnd = data.rEnd;
    MOCHI_ASSERT_VERBOSE(
        rowBegin >= 0 && rowBegin <= rowEnd && rowEnd <= _inverse.Rows(), "Invalid row range.");
    Px.MiddleRows(rowBegin, rowEnd - rowBegin) =
        _inverse.MiddleRows(rowBegin, rowEnd - rowBegin) * x;
  }

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

  template <typename MatrixType>
  void Update(MatrixType const& A);

 protected:
  Matrix<NonConstScalar> _inverse;
};

template <typename Scalar>
template <typename MatrixType>
SymInversePrec<Scalar>::SymInversePrec(MatrixType const& A) {
  Update(A);
}

template <typename Scalar>
template <typename MatrixType>
void SymInversePrec<Scalar>::Update(MatrixType const& A) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  if constexpr (IsMatrix<MatrixType>) {
    _inverse = SymInverse(A);
  } else {
    _inverse = SymInverse(ToMatrix(A));
  }
}

template <typename Scalar>
template <typename Input, typename Output>
void SymInversePrec<Scalar>::operator()(Input const& x, Output&& y) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_inverse.Rows(), x, y);
  y = _inverse * x;
}

} // namespace mochi::krylov
