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

#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_accessors.h>
#include <mochi_core/linear_algebra/triangular_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

#include <type_traits>

namespace mochi {

template <typename Scalar, bool kUseCuda = false>
struct MochiMatrixFactory {
  //
  using NonConstScalar = std::remove_const_t<Scalar>;

  template <bool kOnGPU = kUseCuda>
  static auto CreateNew(int mrow) {
    if constexpr (kOnGPU) {
      return CudaVector<NonConstScalar, krylov::kDynamic>(mrow);
    } else {
      return Matrix<NonConstScalar, krylov::kDynamic, 1>(mrow);
    }
  }

  template <bool kOnGPU = kUseCuda>
  static auto CreateNew(int mrow, int mcols) {
    if constexpr (kOnGPU) {
      return CudaMatrix<NonConstScalar>(mrow, mcols);
    } else {
      return Matrix<NonConstScalar, krylov::kDynamic, krylov::kDynamic>(mrow, mcols);
    }
  }

  //
  template <
      typename InputScalar,
      int kRow,
      int kCol,
      krylov::Direction kMajorDirection,
      krylov::Ownership kOwnership,
      int kLdDim>
  static auto GetSameAs(
      Matrix<InputScalar, kRow, kCol, kMajorDirection, kOwnership, kLdDim> const& M) {
    static_assert(std::is_same_v<InputScalar const, Scalar const>, "Incompatible scalar types.");
    if constexpr (krylov::IsCuda(kOwnership)) {
      return CudaMatrix<NonConstScalar, kRow, kCol, kMajorDirection>{M.Rows(), M.Cols()};
    } else {
      return Matrix<NonConstScalar, kRow, kCol, kMajorDirection>{M.Rows(), M.Cols()};
    }
  }

  //
  template <
      typename InputScalar,
      int kRow,
      int kCol,
      krylov::Direction kMajorDirection,
      krylov::Ownership kOwnership,
      int kLdDim>
  static auto GetCopy(
      Matrix<InputScalar, kRow, kCol, kMajorDirection, kOwnership, kLdDim> const& M) {
    static_assert(std::is_same_v<InputScalar const, Scalar const>, "Incompatible scalar types.");
    if constexpr (std::is_same_v<NonConstScalar, InputScalar>) {
      if constexpr (krylov::IsCuda(kOwnership)) {
        return CudaMatrix<NonConstScalar, kRow, kCol, kMajorDirection>(M);
      } else {
        return Matrix<NonConstScalar, kRow, kCol, kMajorDirection>(M);
      }
    } else {
      auto copyM = GetSameAs(M);
      copyM = M;
      return copyM;
    }
  }
};

} // namespace mochi

namespace mochi::krylov::customization {

template <IsMatrix M>
auto GetFactory(M const& /*mat*/) {
  return MochiMatrixFactory<
      typename details::MatTraits<M>::Scalar,
      details::MatTraits<M>::kIsCuda>{};
}

// Put customization of SetZero
template <IsMatrix M>
void SetZero(M& m) {
  m.SetZero();
}

} // namespace mochi::krylov::customization

namespace mochi::krylov {

template <IsMatrix Mtx, typename VectorType>
void UpperSolveInPlace(Mtx const& M, VectorType&& v1) {
  auto const& upperMat = UpperTriangularView(M);
  upperMat.SolveInPlace(v1);
}

} // namespace mochi::krylov
