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
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

#include <type_traits>
#include <variant>
#include <vector>

namespace mochi {

// A variant that can store any of our commonly used matrix types.
template <typename Scalar>
using AnyMatrix = std::variant<
    BlockSparseMatrix<Scalar, 3>,
    BlockSparseMatrix<Scalar, 4>,
    SparseMatrix<Scalar>,
    Matrix<Scalar>>;

// A variant that can store any of our commonly used matrix VIEW types.
template <typename Scalar>
using AnyMatrixView = std::variant<
    BlockSparseMatrixView<Scalar, 3>,
    BlockSparseMatrixView<Scalar, 4>,
    SparseMatrixView<Scalar>,
    MatrixView<Scalar>>;

// AnyMatrix and AnyMatrixView must be in sync.
static_assert(std::variant_size_v<AnyMatrix<real>> == std::variant_size_v<AnyMatrixView<real>>);
} // namespace mochi

namespace mochi::details {
/// @note It does not include AnyMatrixView objects.
template <typename Scalar>
constexpr bool IsAnyMatrixVariantDef<AnyMatrix<Scalar>> = true;
} // namespace mochi::details

namespace mochi::krylov::details {
/// @note It does not cover AnyMatrixView objects.
template <typename T>
struct MatTraitsDef<mochi::AnyMatrix<T>> {
  using Scalar = T;
};
} // namespace mochi::krylov::details

namespace mochi {
// Type trait to determine if T is a type of std::variant (in the style of std type traits).
template <typename T>
struct is_variant : std::false_type {};
template <typename... Args>
struct is_variant<std::variant<Args...>> : std::true_type {};
template <typename T>
struct is_variant<T const> : is_variant<T> {};
template <typename T>
inline constexpr bool is_variant_v = is_variant<T>::value;

// Return the number of rows in any matrix type, including the variant types above.
template <typename MatrixT>
inline auto GetNumRows(MatrixT const& mat) {
  if constexpr (is_variant_v<MatrixT>) {
    return std::visit([](auto const& m) { return m.Rows(); }, mat);
  } else {
    return mat.Rows();
  }
}

// Return the number of columns in any matrix type, including the variant types above.
template <typename MatrixT>
inline auto GetNumCols(MatrixT const& mat) {
  if constexpr (is_variant_v<MatrixT>) {
    return std::visit([](auto const& m) { return m.Cols(); }, mat);
  } else {
    return mat.Cols();
  }
}

// Return the number of block rows in a matrix. For dense and sparse matrices, this is simply the
// number of rows.
template <typename MatrixT>
inline auto GetNumBlockRows(MatrixT const& mat) {
  if constexpr (is_variant_v<MatrixT>) {
    return std::visit([](auto const& m) { return GetNumBlockRows(m); }, mat);
  } else if constexpr (IsBlockSparseMatrix<MatrixT>) {
    return mat.BlockRows();
  } else {
    static_assert(IsMatrix<MatrixT> || IsSparseMatrix<MatrixT>);
    return mat.Rows();
  }
}

// Return the value array as a Span. For a sparse matrix, this is just the non-zero values.
template <typename MatrixT>
inline auto GetValues(MatrixT& mat) {
  if constexpr (is_variant_v<MatrixT>) {
    return std::visit([&](auto& m) { return GetValues(m); }, mat);
  } else if constexpr (IsSparseMatrix<MatrixT> || IsBlockSparseMatrix<MatrixT>) {
    return mat.Values();
  } else {
    static_assert(IsMatrix<MatrixT>);
    MOCHI_ASSERT_VERBOSE(
        mat.Rows() * mat.Cols() == mat.StorageSize(),
        "This matrix does not store a contiguous span of values");
    return Span{mat.data(), static_cast<size_t>(mat.StorageSize())}; // dense storage
  }
}

// Return the number of entries in the sparsity pattern of a matrix.
template <typename MatrixT>
inline auto GetNumValues(MatrixT const& mat) {
  return isize(GetValues(mat));
}

// Approximate number of FLOPs per matrix-vector product.
template <typename MatrixT>
inline auto FlopsPerApply(MatrixT const& mat) {
  return 2 * GetNumValues(mat);
}

template <typename MatrixT>
auto GetRowRangesPerWorker(MatrixT const& A, int numWorkers) {
  static_assert(IsAnyMatrix<MatrixT>, "Matrix type not supported");
  MOCHI_ASSERT_VERBOSE(numWorkers > 0, "Number of workers must be positive.");
  using Idx = std::remove_const_t<decltype(GetNumBlockRows(A))>;
  auto const numBlockRows = GetNumBlockRows(A);
  auto const blockSize = GetNumRows(A) / numBlockRows;
  std::vector<Idx> workerRowRanges(numWorkers + 1, 0);
  for (int i = 1; i < numWorkers; ++i) {
    // Ensure row range is consistent with the block size.
    workerRowRanges[i] =
        blockSize * static_cast<Idx>(static_cast<int64_t>(i * numBlockRows) / numWorkers);
  }
  workerRowRanges[numWorkers] = blockSize * numBlockRows;
  return workerRowRanges;
}

// Set values to zero in AnyMatrix.
template <typename Scalar>
inline void SetZero(AnyMatrix<Scalar>& mat) {
  std::visit([](auto& m) { m.SetZero(); }, mat);
}

// Set values to zero in AnyMatrixView.
template <typename Scalar>
inline void SetZero(AnyMatrixView<Scalar>& mat) {
  std::visit([](auto& m) { m.SetZero(); }, mat);
}

// Create a view of AnyMatrix
template <typename Scalar>
AnyMatrixView<Scalar> AsView(AnyMatrix<Scalar>& anyMat) {
  return std::visit([](auto& m) { return AnyMatrixView<Scalar>{m}; }, anyMat);
}

// Create a const view of AnyMatrix
template <typename Scalar>
AnyMatrixView<Scalar const> AsConstView(AnyMatrix<Scalar> const& anyMat) {
  return std::visit([](auto const& m) { return AnyMatrixView<Scalar const>{m}; }, anyMat);
}

// Create a const view of AnyMatrix, from a non-const view
template <typename Scalar>
AnyMatrixView<Scalar const> AsConstView(AnyMatrixView<Scalar> const& anyMat) {
  return std::visit([](auto const& m) { return AnyMatrixView<Scalar const>{m}; }, anyMat);
}

} // namespace mochi
