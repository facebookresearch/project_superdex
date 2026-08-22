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
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>

#include <type_traits>

namespace mochi::krylov {

template <typename Scalar>
struct LDLtPrec final : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  static constexpr auto kType = PreconditionerType::LDLT;

  template <typename MatrixType>
  explicit LDLtPrec(MatrixType const& A);

  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const;

  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<NonConstScalar> Px) const override {
    operator()(x, Px);
  }

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

  // TODO Put new update here. MLX

 private:
  /** @brief Prepares @p A for consumption by @ref LDLt. If @p A is already a dense @ref Matrix,
   * returns a dynamic-size view to avoid a copy. Otherwise, materializes it into a dense matrix via
   * @ref ToMatrix. */
  template <typename MatrixType>
  static auto ToLDLtInput(MatrixType const& A) {
    if constexpr (IsMatrix<MatrixType>) {
      // Create a view to avoid an unnecessary copy. Note that _ldlt currently requires dynamic
      // size.
      return AsConstView<krylov::kDynamic, krylov::kDynamic>(A);
    } else {
      return ToMatrix(A);
    }
  }

 private:
  int _info = 0;
  LDLt<NonConstScalar> _ldlt;
};

template <typename Scalar>
template <typename MatrixType>
LDLtPrec<Scalar>::LDLtPrec(MatrixType const& A) : _ldlt(ToLDLtInput(A), _info) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  if (_info != 0) {
    MOCHI_LOG_ERROR("LDLt factorization failed with status flag %i.", _info);
  }
}

template <typename Scalar>
template <typename Input, typename Output>
void LDLtPrec<Scalar>::operator()(Input const& x, Output&& y) const {
  Preconditioner<NonConstScalar>::ValidateInputOutput(_ldlt.GetStorage().Rows(), x, y);
  y = x;
  _ldlt.LeftSolveInPlace(y);
}

} // namespace mochi::krylov
