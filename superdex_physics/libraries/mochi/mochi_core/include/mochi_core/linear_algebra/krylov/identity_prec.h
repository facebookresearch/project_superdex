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

#include <type_traits>

namespace mochi::krylov {

template <typename Scalar>
struct IdentityPrec final : Preconditioner<std::remove_const_t<Scalar>> {
  using NonConstScalar = std::remove_const_t<Scalar>;
  static constexpr auto kType = PreconditionerType::None;

  template <typename MatrixType>
  IdentityPrec(MatrixType const& A) {
    Update(A);
  }

  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& Px) const {
    Preconditioner<NonConstScalar>::ValidateInputOutput(_size, x, Px);
    Px = x;
  }

  void Solve(ColumnVectorView<Scalar const> x, ColumnVectorView<NonConstScalar> Px) const override {
    operator()(x, Px);
  }

  void ConcurrentSolve(
      ColumnVectorView<Scalar const> x,
      ColumnVectorView<NonConstScalar> Px,
      ParallelWorkerInfo const& data) const override {
    Preconditioner<NonConstScalar>::ValidateInputOutput(_size, x, Px);
    auto const rowBegin = data.rBegin;
    auto const rowEnd = data.rEnd;
    MOCHI_ASSERT_VERBOSE(
        rowBegin >= 0 && rowBegin <= rowEnd && rowEnd <= x.Rows(), "Invalid row range.");
    Px.MiddleRows(rowBegin, rowEnd - rowBegin) = x.MiddleRows(rowBegin, rowEnd - rowBegin);
  }

  template <typename MatrixType>
  void Update(MatrixType const& A) {
    MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
    _size = A.Rows();
  }

  constexpr PreconditionerType GetType() const override {
    return kType;
  }

 protected:
  int _size = 0;
};

} // namespace mochi::krylov
