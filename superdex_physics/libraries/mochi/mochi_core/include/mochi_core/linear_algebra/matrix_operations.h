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

#include <mochi_core/linear_algebra/actor_pseudo_matrix.h>
#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/krylov/tools/custom_matrix_traits.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_traits.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/ldlt.h>
#include <mochi_core/linear_algebra/lu.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/vmatrix.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mochi::details {

template <typename InputType>
bool HasSortedRowIndices(InputType const& /*A*/) {
  static_assert(std::is_void_v<InputType>, "Matrix type is not supported currently.");
  return false;
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDir,
    krylov::Ownership kOwnership,
    int kMajorDim>
bool HasSortedRowIndices(
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim> const&
    /*A*/) {
  return true;
}

template <
    typename InputScalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename AStorage>
bool HasSortedRowIndices(
    BlockSparseMatrix<InputScalar, kBlockSize, CRIdx, Ptr, AStorage> const& A) {
  bool isSorted = true;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  for (NonConstIdx ir = 0; ((ir < A.BlockRows()) && (isSorted)); ++ir) {
    auto const colIndices = A.Indices(ir);
    isSorted = std::is_sorted(colIndices.begin(), colIndices.end());
  }
  return isSorted;
}

template <
    typename InputScalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename AStorage>
bool HasSortedRowIndices(SparseMatrix<InputScalar, CRIdx, Ptr, AStorage> const& A) {
  bool isSorted = true;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  auto aPtr = A.Pointers();
  auto first = A.Indices().begin();
  for (NonConstIdx ir = 0; ((ir < A.Rows()) && (isSorted)); ++ir) {
    auto last = first + aPtr[ir + 1] - aPtr[ir];
    isSorted = std::is_sorted(first, last);
    first = last;
  }
  return isSorted;
}

template <typename InputMatType, int i, int j>
inline auto Cofactor_3x3(InputMatType const& A) {
  MOCHI_ASSERT_VERBOSE((A.Rows() >= 3) && (A.Cols() >= 3), "Matrix is too small.");
  constexpr char i1 = (i + 1) % 3;
  constexpr char i2 = (i + 2) % 3;
  constexpr char j1 = (j + 1) % 3;
  constexpr char j2 = (j + 2) % 3;
  return A(i1, j1) * A(i2, j2) - A(i1, j2) * A(i2, j1);
}

template <typename InputMatType>
inline auto Determinant3x3(InputMatType const& A) {
  MOCHI_ASSERT_VERBOSE((A.Rows() == 3) && (A.Cols() == 3), "Incorrect matrix size.");
  using NonConstScalar = std::remove_const_t<typename MatTraits<InputMatType>::Scalar>;
  NonConstScalar det = 0;
  det += A(0, 0) * details::Cofactor_3x3<InputMatType, 0, 0>(A);
  det += A(1, 0) * details::Cofactor_3x3<InputMatType, 1, 0>(A);
  det += A(2, 0) * details::Cofactor_3x3<InputMatType, 2, 0>(A);
  return det;
}

template <typename InputMatType>
inline auto Determinant4x4(InputMatType const& A) {
  // This implementation outperforms cofactor and block-inverse implementations. Additional
  // performance may be achieved via SIMD instructions.
  MOCHI_ASSERT_VERBOSE((A.Rows() == 4) && (A.Cols() == 4), "Incorrect matrix size.");
  using Scalar = typename MatTraits<InputMatType>::Scalar;
  Scalar const s0 = A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1);
  Scalar const s1 = A(0, 0) * A(1, 2) - A(1, 0) * A(0, 2);
  Scalar const s2 = A(0, 0) * A(1, 3) - A(1, 0) * A(0, 3);
  Scalar const s3 = A(0, 1) * A(1, 2) - A(1, 1) * A(0, 2);
  Scalar const s4 = A(0, 1) * A(1, 3) - A(1, 1) * A(0, 3);
  Scalar const s5 = A(0, 2) * A(1, 3) - A(1, 2) * A(0, 3);

  Scalar const c5 = A(2, 2) * A(3, 3) - A(3, 2) * A(2, 3);
  Scalar const c4 = A(2, 1) * A(3, 3) - A(3, 1) * A(2, 3);
  Scalar const c3 = A(2, 1) * A(3, 2) - A(3, 1) * A(2, 2);
  Scalar const c2 = A(2, 0) * A(3, 3) - A(3, 0) * A(2, 3);
  Scalar const c1 = A(2, 0) * A(3, 2) - A(3, 0) * A(2, 2);
  Scalar const c0 = A(2, 0) * A(3, 1) - A(3, 0) * A(2, 1);

  Scalar const det = (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
  return det;
}

template <
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline void Inverse1x1(
    Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A,
    Matrix<std::remove_const_t<Scalar>, kRowsAtCT, kColsAtCT, kMajorDirection>& invA) {
  MOCHI_ASSERT_VERBOSE((A.Rows() == 1) && (A.Cols() == 1), "Incorrect matrix size.");
  MOCHI_ASSERT_VERBOSE((invA.Rows() == 1) && (invA.Cols() == 1), "Incorrect matrix size.");
  if (A(0, 0) == Scalar{0})
    MOCHI_UNLIKELY {
      MOCHI_LOG_ERROR("Matrix is numerically singular (%e)", static_cast<double>(A(0, 0)));
      invA(0, 0) = Scalar(0);
      return;
    }
  invA(0, 0) = Scalar(1) / A(0, 0);
}

template <
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline void Inverse2x2(
    Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A,
    Matrix<std::remove_const_t<Scalar>, kRowsAtCT, kColsAtCT, kMajorDirection>& invA) {
  MOCHI_ASSERT_VERBOSE((A.Rows() == 2) && (A.Cols() == 2), "Incorrect matrix size.");
  MOCHI_ASSERT_VERBOSE((invA.Rows() == 2) && (invA.Cols() == 2), "Incorrect matrix size.");
  auto const detA = Determinant(A);
  if (detA == Scalar{0})
    MOCHI_UNLIKELY {
      MOCHI_LOG_ERROR("Matrix is numerically singular (%e)", static_cast<double>(detA));
      invA.SetZero();
      return;
    }
  invA(0, 0) = A(1, 1);
  invA(0, 1) = -A(0, 1);
  invA(1, 0) = -A(1, 0);
  invA(1, 1) = A(0, 0);
  invA *= Scalar(1) / detA; // For 2x2, "invA /= detA" is faster than "invA *= Scalar(1) / detA" or
  // "invDetA = Scalar(1) / detA; invA *= invDetA".
}

template <
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline void Inverse3x3(
    Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A,
    Matrix<std::remove_const_t<Scalar>, kRowsAtCT, kColsAtCT, kMajorDirection>& invA) {
  // Longer but more performant implementation: P785454111. Use it as starting point if additional
  // performance is needed.
  MOCHI_ASSERT_VERBOSE((A.Rows() == 3) && (A.Cols() == 3), "Incorrect matrix size.");
  MOCHI_ASSERT_VERBOSE((invA.Rows() == 3) && (invA.Cols() == 3), "Incorrect matrix size.");

  using ResultType = Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim>;
  invA(0, 0) = details::Cofactor_3x3<ResultType, 0, 0>(A);
  invA(0, 1) = details::Cofactor_3x3<ResultType, 1, 0>(A);
  invA(0, 2) = details::Cofactor_3x3<ResultType, 2, 0>(A);

  invA(1, 0) = details::Cofactor_3x3<ResultType, 0, 1>(A);
  invA(1, 1) = details::Cofactor_3x3<ResultType, 1, 1>(A);
  invA(1, 2) = details::Cofactor_3x3<ResultType, 2, 1>(A);

  invA(2, 0) = details::Cofactor_3x3<ResultType, 0, 2>(A);
  invA(2, 1) = details::Cofactor_3x3<ResultType, 1, 2>(A);
  invA(2, 2) = details::Cofactor_3x3<ResultType, 2, 2>(A);

  Scalar const detA = invA(0, 0) * A(0, 0) + invA(0, 1) * A(1, 0) + invA(0, 2) * A(2, 0);
  if (detA == Scalar{0})
    MOCHI_UNLIKELY {
      MOCHI_LOG_ERROR("Matrix is numerically singular (%e)", static_cast<double>(detA));
      invA.SetZero();
      return;
    }
  invA *= Scalar(1) / detA; // For 3x3, no performance difference between "invA /= detA" and
  // "invA *= Scalar(1) / detA".
}

template <
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline void Inverse4x4(
    Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A,
    Matrix<std::remove_const_t<Scalar>, kRowsAtCT, kColsAtCT, kMajorDirection>& invA) {
  // Reference:
  // https://stackoverflow.com/questions/2624422/efficient-4x4-matrix-inverse-affine-transform/9614511#9614511
  // It outperforms cofactor and block-inverse implementations. Additional performance may be
  // achieved via SIMD instructions.
  MOCHI_ASSERT_VERBOSE((A.Rows() == 4) && (A.Cols() == 4), "Incorrect matrix size.");
  MOCHI_ASSERT_VERBOSE((invA.Rows() == 4) && (invA.Cols() == 4), "Incorrect matrix size.");

  Scalar const s0 = A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1);
  Scalar const s1 = A(0, 0) * A(1, 2) - A(1, 0) * A(0, 2);
  Scalar const s2 = A(0, 0) * A(1, 3) - A(1, 0) * A(0, 3);
  Scalar const s3 = A(0, 1) * A(1, 2) - A(1, 1) * A(0, 2);
  Scalar const s4 = A(0, 1) * A(1, 3) - A(1, 1) * A(0, 3);
  Scalar const s5 = A(0, 2) * A(1, 3) - A(1, 2) * A(0, 3);

  Scalar const c5 = A(2, 2) * A(3, 3) - A(3, 2) * A(2, 3);
  Scalar const c4 = A(2, 1) * A(3, 3) - A(3, 1) * A(2, 3);
  Scalar const c3 = A(2, 1) * A(3, 2) - A(3, 1) * A(2, 2);
  Scalar const c2 = A(2, 0) * A(3, 3) - A(3, 0) * A(2, 3);
  Scalar const c1 = A(2, 0) * A(3, 2) - A(3, 0) * A(2, 2);
  Scalar const c0 = A(2, 0) * A(3, 1) - A(3, 0) * A(2, 1);

  Scalar const detA = (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);

  if (detA == Scalar{0})
    MOCHI_UNLIKELY {
      MOCHI_LOG_ERROR("Matrix is numerically singular (%e)", static_cast<double>(detA));
      invA.SetZero();
      return;
    }
  Scalar const invDetA = Scalar(1) / detA;

  // Performance notes:
  // - Setting the inverse in (0, 0) -> (0, 1) -> ... order is marginally faster for row-major
  //   matrices. The code below can be gated by major direction if additional performance is needed.
  // - No noticeable improvement when using the initializer list constructor.
  // clang-format off
        invA(0, 0) = (A(1, 1) * c5 - A(1, 2) * c4 + A(1, 3) * c3) * invDetA;
        invA(1, 0) = (-A(1, 0) * c5 + A(1, 2) * c2 - A(1, 3) * c1) * invDetA;
        invA(2, 0) = (A(1, 0) * c4 - A(1, 1) * c2 + A(1, 3) * c0) * invDetA;
        invA(3, 0) = (-A(1, 0) * c3 + A(1, 1) * c1 - A(1, 2) * c0) * invDetA;

        invA(0, 1) = (-A(0, 1) * c5 + A(0, 2) * c4 - A(0, 3) * c3) * invDetA;
        invA(1, 1) = (A(0, 0) * c5 - A(0, 2) * c2 + A(0, 3) * c1) * invDetA;
        invA(2, 1) = (-A(0, 0) * c4 + A(0, 1) * c2 - A(0, 3) * c0) * invDetA;
        invA(3, 1) = (A(0, 0) * c3 - A(0, 1) * c1 + A(0, 2) * c0) * invDetA;

        invA(0, 2) = (A(3, 1) * s5 - A(3, 2) * s4 + A(3, 3) * s3) * invDetA;
        invA(1, 2) = (-A(3, 0) * s5 + A(3, 2) * s2 - A(3, 3) * s1) * invDetA;
        invA(2, 2) = (A(3, 0) * s4 - A(3, 1) * s2 + A(3, 3) * s0) * invDetA;
        invA(3, 2) = (-A(3, 0) * s3 + A(3, 1) * s1 - A(3, 2) * s0) * invDetA;

        invA(0, 3) = (-A(2, 1) * s5 + A(2, 2) * s4 - A(2, 3) * s3) * invDetA;
        invA(1, 3) = (A(2, 0) * s5 - A(2, 2) * s2 + A(2, 3) * s1) * invDetA;
        invA(2, 3) = (-A(2, 0) * s4 + A(2, 1) * s2 - A(2, 3) * s0) * invDetA;
        invA(3, 3) = (A(2, 0) * s3 - A(2, 1) * s1 + A(2, 2) * s0) * invDetA;
  // clang-format on
}

} // namespace mochi::details

namespace mochi {

/// @brief Function to compute X^T Y.
/// @return Matrix storing X^T Y (even when X and Y have 1 column).
template <
    typename ScalarX,
    int kRowsAtCompileTimeX,
    int kColsAtCompileTimeX,
    krylov::Direction kMajorDirectionX,
    krylov::Ownership kOwnershipX,
    int kLeadingDimX,
    typename ScalarY,
    int kRowsAtCompileTimeY,
    int kColsAtCompileTimeY,
    krylov::Direction kMajorDirectionY,
    krylov::Ownership kOwnershipY,
    int kLeadingDimY>
auto Dot(
    Matrix<
        ScalarX,
        kRowsAtCompileTimeX,
        kColsAtCompileTimeX,
        kMajorDirectionX,
        kOwnershipX,
        kLeadingDimX> const& X,
    Matrix<
        ScalarY,
        kRowsAtCompileTimeY,
        kColsAtCompileTimeY,
        kMajorDirectionY,
        kOwnershipY,
        kLeadingDimY> const& Y) {
  static_assert(
      std::is_same_v<std::remove_const_t<ScalarX>, std::remove_const_t<ScalarY>>,
      "X and Y must have the same scalar type for computing X^T Y.");
  MOCHI_ASSERT_VERBOSE(X.Rows() == Y.Rows(), "Dimensions do not match");
  static_assert(
      krylov::Owning<kOwnershipX> == krylov::Owning<kOwnershipY>,
      "Mixing of ownership types is not supported yet");

  using NonConstScalar = std::remove_const_t<ScalarX>;
  using ResultMatrix = Matrix<
      NonConstScalar,
      kColsAtCompileTimeX,
      kColsAtCompileTimeY,
      krylov::Direction::ColMajor,
      krylov::Owning<kOwnershipX>>;
  ResultMatrix XtY(X.Cols(), Y.Cols());

  if constexpr (krylov::Owning<kOwnershipX> != krylov::Ownership::Cuda) {
    if constexpr ((kColsAtCompileTimeX == 1) && (kColsAtCompileTimeY == 1)) {
      XtY(0, 0) = X.Dot(Y);
    } else {
      if ((X.Cols() == 1) && (Y.Cols() == 1)) {
        XtY(0, 0) = X.Dot(Y);
      } else {
        auto Xt = X.Transpose();
        XtY = Xt * Y;
      }
    }
  } else {
    auto Xt = X.Transpose();
    XtY = Xt * Y;
  }

  return XtY;
}

/// @brief Computes the trace of a block sparse matrix, that is, the sum of the elements along the
/// main diagonal.
/// @param[in] A Block sparse matrix.
/// @return Trace of the matrix.
/// @note The matrix must be square.
template <
    typename Scalar,
    int kBlockSize,
    typename Idx,
    typename Ptr,
    template <typename, typename...> typename InputStorage>
std::remove_const_t<Scalar> Trace(
    BlockSparseMatrix<Scalar, kBlockSize, Idx, Ptr, InputStorage> const& A) {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<Idx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  MOCHI_ASSERT_VERBOSE(A.BlockRows() == A.BlockCols(), "Trace defined only for square matrix");
  NonConstScalar trace{};
  for (NonConstIdx i = 0; i < A.BlockRows(); ++i) {
    auto colIdx = A.Indices(i);
    auto values = A.Values(i);
    for (NonConstPtr p = 0; p < static_cast<NonConstPtr>(colIdx.size()); ++p) {
      if (colIdx[p] == i) {
        auto const& D = values[p];
        for (int k = 0; k < kBlockSize; ++k) {
          trace += D(k, k);
        }
        break;
      }
    }
  }
  return trace;
}

/// @brief Computes the trace of a (dense) matrix, that is, the sum of the elements along the main
/// diagonal.
/// @param[in] A (Dense) matrix.
/// @return Trace of the matrix.
/// @note The matrix must be square.
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDir,
    krylov::Ownership kOwnership,
    int kMajorDim>
std::remove_const_t<Scalar> Trace(
    Matrix<Scalar, kRowsAtCompileTime, kColsAtCompileTime, kMajorDir, kOwnership, kMajorDim> const&
        A) {
  static_assert(!krylov::IsCuda(kOwnership), "Invalid input matrix");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Trace defined only for square matrix");
  using NonConstScalar = std::remove_const_t<Scalar>;
  NonConstScalar trace{};
  if constexpr ((kRowsAtCompileTime <= 0) || (kColsAtCompileTime <= 0)) {
    for (int i = 0; i < A.Rows(); ++i) {
      trace += A(i, i);
    }
  } else {
    // Both dimensions are set at compile time
    trace += A(0, 0);
    if constexpr ((kRowsAtCompileTime >= 2) && (kColsAtCompileTime >= 2)) {
      trace += A(1, 1);
    }
    if constexpr ((kRowsAtCompileTime >= 3) && (kColsAtCompileTime >= 3)) {
      trace += A(2, 2);
    }
    if constexpr ((kRowsAtCompileTime >= 4) && (kColsAtCompileTime >= 4)) {
      trace += A(3, 3);
    }
    if constexpr ((kRowsAtCompileTime >= 5) && (kColsAtCompileTime >= 5)) {
      for (int i = 4; i < A.Rows(); ++i) {
        trace += A(i, i);
      }
    }
  }
  return trace;
}

/// @brief Computes the trace of a sparse matrix, that is, the sum of the elements along the main
/// diagonal.
/// @param[in] A Sparse matrix.
/// @return Trace of the matrix.
/// @note The matrix must be square.
template <
    typename Scalar,
    typename Idx,
    typename Ptr,
    template <typename, typename...> typename InputStorage>
std::remove_const_t<Scalar> Trace(SparseMatrix<Scalar, Idx, Ptr, InputStorage> const& A) {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<Idx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  NonConstScalar trace{};
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Trace defined only for square matrix");
  for (NonConstIdx i = 0; i < A.Rows(); ++i) {
    auto colIdx = A.Indices(i);
    auto values = A.Values(i);
    for (NonConstPtr p = 0; p < static_cast<NonConstPtr>(colIdx.size()); ++p) {
      if (colIdx[p] == i) {
        trace += values[p];
        break;
      }
    }
  }
  return trace;
}

/// @brief Compute the determinant of a 1x1 fixed-size matrix.
/// @param[in] A Input matrix.
/// @return The determinant of the input matrix.
template <
    typename Scalar,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline auto Determinant(Matrix<Scalar, 1, 1, kMajorDirection, kOwnership, kLeadDim> const& A) {
  return A(0, 0);
}

/// @brief Compute the determinant of a 2x2 fixed-size matrix.
/// @param[in] A Input matrix.
/// @return The determinant of the input matrix.
template <
    typename Scalar,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline auto Determinant(Matrix<Scalar, 2, 2, kMajorDirection, kOwnership, kLeadDim> const& A) {
  return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
}

/// @brief Compute the determinant of a 3x3 fixed-size matrix.
/// @param[in] A Input matrix.
/// @return The determinant of the input matrix.
template <
    typename Scalar,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline auto Determinant(Matrix<Scalar, 3, 3, kMajorDirection, kOwnership, kLeadDim> const& A) {
  using InputType = Matrix<Scalar, 3, 3, kMajorDirection, kOwnership, kLeadDim>;
  return details::Determinant3x3<InputType>(A);
}

/// @brief Compute the determinant of a 4x4 fixed-size matrix.
/// @param[in] A Input matrix.
/// @return The determinant of the input matrix.
template <
    typename Scalar,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline auto Determinant(Matrix<Scalar, 4, 4, kMajorDirection, kOwnership, kLeadDim> const& A) {
  using InputType = Matrix<Scalar, 4, 4, kMajorDirection, kOwnership, kLeadDim>;
  return details::Determinant4x4<InputType>(A);
}

/// @brief Compute the determinant of a matrix.
/// @param[in] A Input matrix.
/// @return The determinant of the input matrix.
///
/// @remark For sizes up to 4, the determinant is computed via exact formula. For larger sizes, the
/// determinant is computed via block LU factorization without pivoting.
template <
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
auto Determinant(
    Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A) {
  if (A.Rows() != A.Cols()) {
    return Scalar(0);
  }
  //--- If size < 5, use the exact formula.
  if (A.Rows() < 5) {
    using InputType = Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim>;
    switch (A.Rows()) {
      case 1:
        return A(0, 0);
      case 2:
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
      case 3:
        return details::Determinant3x3<InputType>(A);
      case 4:
        return details::Determinant4x4<InputType>(A);
    }
  }
  //--- Otherwise, use the LU factorization.
  LU<Scalar, kRowsAtCT, kColsAtCT, PermuteAlg::None> lu(A);
  return lu.ScalarDeterminant();
}

/// @brief Compute the inverse of a 1x1 fixed-size matrix.
/// @param[in] A Input matrix.
/// @return The inverse of the input matrix.
///
/// @remark If the matrix is numerically singular, an error message is logged and the zero matrix is
/// returned.
template <
    typename Scalar,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline auto Inverse(Matrix<Scalar, 1, 1, kMajorDirection, kOwnership, kLeadDim> const& A) {
  Matrix<std::remove_const_t<Scalar>, 1, 1, kMajorDirection> invA;
  details::Inverse1x1(A, invA);
  return invA;
}

/// @brief Compute the inverse of a 2x2 fixed-size matrix.
/// @param[in] A Input matrix.
/// @return The inverse of the input matrix.
///
/// @remark If the matrix is numerically singular, an error message is logged and the zero matrix is
/// returned.
template <
    typename Scalar,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline auto Inverse(Matrix<Scalar, 2, 2, kMajorDirection, kOwnership, kLeadDim> const& A) {
  Matrix<std::remove_const_t<Scalar>, 2, 2, kMajorDirection> invA;
  details::Inverse2x2(A, invA);
  return invA;
}

/// @brief Compute the inverse of a 3x3 fixed-size matrix.
/// @param[in] A Input matrix.
/// @return The inverse of the input matrix.
///
/// @remark If the matrix is numerically singular, an error message is logged and the zero matrix is
/// returned.
template <
    typename Scalar,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline auto Inverse(Matrix<Scalar, 3, 3, kMajorDirection, kOwnership, kLeadDim> const& A) {
  Matrix<std::remove_const_t<Scalar>, 3, 3, kMajorDirection> invA;
  details::Inverse3x3(A, invA);
  return invA;
}

/// @brief Compute the inverse of a 4x4 fixed-size matrix.
/// @param[in] A Input matrix.
/// @return The inverse of the input matrix.
///
/// @remark If the matrix is numerically singular, an error message is logged and the zero matrix is
/// returned.
template <
    typename Scalar,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
inline auto Inverse(Matrix<Scalar, 4, 4, kMajorDirection, kOwnership, kLeadDim> const& A) {
  Matrix<std::remove_const_t<Scalar>, 4, 4, kMajorDirection> invA;
  details::Inverse4x4(A, invA);
  return invA;
}

/// @brief Stable computation of the inverse of a matrix.
/// @param[in] A Input matrix.
/// @return The inverse of the input matrix.
///
/// @remark The inverse is computed via block LU factorization with Rook pivoting at the block
/// level.
template <
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
auto StableInverse(
    Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A) {
  Matrix<std::remove_const_t<Scalar>, kRowsAtCT, kColsAtCT, kMajorDirection> invA(
      A.Rows(), A.Cols());
  details::LuInverse<PermuteAlg::Rook>(A, invA);
  return invA;
}

/// @brief Compute the inverse of a matrix.
/// @param[in] A Input matrix.
/// @return The inverse of the input matrix.
///
/// @remark For sizes up to 4, the inverse is computed via exact formula. For larger sizes, the
/// inverse is computed via block LU factorization without pivoting.
/// @remark For symmetric matrices, prefer 'SymInverse' over 'Inverse' to improve performance.
/// @remark For matrices that are ill-conditioned or require pivoting, prefer 'StableInverse' over
/// 'Inverse' to improve stability.
template <
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
auto Inverse(Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A) {
  Matrix<std::remove_const_t<Scalar>, kRowsAtCT, kColsAtCT, kMajorDirection> invA(
      A.Rows(), A.Cols());
  //--- If size < 5, use the exact formula.
  if (A.Rows() < 5) {
    switch (A.Rows()) {
      case 1: {
        details::Inverse1x1(A, invA);
        return invA;
      }
      case 2: {
        details::Inverse2x2(A, invA);
        return invA;
      }
      case 3: {
        details::Inverse3x3(A, invA);
        return invA;
      }
      case 4: {
        details::Inverse4x4(A, invA);
        return invA;
      }
    }
  }
  //--- Otherwise, use the LU factorization without pivoting.
  details::LuInverse<PermuteAlg::None>(A, invA);
  return invA;
}

/// @brief Compute the inverse of a symmetric matrix.
/// @param[in] A Input symmetric matrix.
/// @return The inverse of the input matrix.
///
/// @remark If the matrix is numerically singular or needs pivoting, an error message is logged and
/// the zero matrix is returned.
/// @remark For sizes up to 4, the inverse is computed via exact formula. For larger sizes, the
/// inverse is computed via block LDLt factorization without pivoting.
/// @remark For symmetric matrices, prefer 'SymInverse' over 'Inverse' to improve performance.
/// @remark For matrices that are ill-conditioned or require pivoting, prefer 'StableInverse' over
/// 'SymInverse' to improve stability.
template <
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadDim>
auto SymInverse(
    Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadDim> const& A) {
  using NonConstScalar = std::remove_const_t<Scalar>;
  Matrix<NonConstScalar, kRowsAtCT, kColsAtCT, kMajorDirection> invA(A.Rows(), A.Cols());
  //--- If size < 5, use the exact formula.
  if (A.Rows() < 5) {
    switch (A.Rows()) {
      case 1: {
        details::Inverse1x1(A, invA);
        return invA;
      }
      case 2: {
        details::Inverse2x2(A, invA);
        return invA;
      }
      case 3: {
        details::Inverse3x3(A, invA);
        return invA;
      }
      case 4: {
        details::Inverse4x4(A, invA);
        return invA;
      }
    }
  }
  //--- Otherwise, use the LDLt factorization.
  int const info = details::LDLtInverse(A, invA);
  if (info != 0)
    MOCHI_UNLIKELY {
      invA.SetZero();
    }
  return invA;
}

/// @brief Compute the dense matrix-matrix product C = A * B in parallel. If kAdd = true,
/// compute C += A * B.
/// @remark Each worker computes the contribution to the product from a subset of the entries along
/// the contraction direction (i.e. along 'k').
template <bool kAdd = false, typename MatA, typename MatB, typename MatC>
inline void ParallelMatMatAlongK(MatA const& A, MatB const& B, MatC&& C) {
  constexpr int kFlopsPerTask = 2500000; // ~50 μs per task assuming ~50 GFLOPs.
  auto const numFlops = static_cast<int64_t>(A.Rows()) * B.Cols() * (2 * B.Rows() - 1);
  int const numTasks =
      Min(static_cast<int>((numFlops + kFlopsPerTask - 1) / kFlopsPerTask),
          B.Rows(),
          TaskScheduler::StaticGetNumOtherThreads() + 1);
  if (numTasks > 1) {
    // Compute per-worker contributions.
    using MatCOwning = decltype(krylov::MatrixFactoryType<MatC>{}.GetSameAs(C));
    std::vector<MatCOwning> Cworker(numTasks - 1);
    ParallelForN("ParallelMatMatAlongK", numTasks, 1, [&](int iTask) {
      int const kBegin = (B.Rows() * iTask) / numTasks;
      int const kEnd = (B.Rows() * (iTask + 1)) / numTasks;
      if (iTask == 0) {
        if constexpr (kAdd) {
          C += A.MiddleCols(kBegin, kEnd - kBegin) * B.MiddleRows(kBegin, kEnd - kBegin);
        } else {
          C = A.MiddleCols(kBegin, kEnd - kBegin) * B.MiddleRows(kBegin, kEnd - kBegin);
        }
      } else {
        MOCHI_ASSERT_VERBOSE(iTask > 0, "Invalid task index.");
        Cworker[iTask - 1] = // Move assignment resizes the LHS.
            MatCOwning(A.MiddleCols(kBegin, kEnd - kBegin) * B.MiddleRows(kBegin, kEnd - kBegin));
      }
    });

    // Add per-worker contributions.
    for (int iTask = 1; iTask < numTasks; ++iTask) {
      C += Cworker[iTask - 1];
    }
  } else {
    // Dedicated branch to avoid unnecessary work.
    if constexpr (kAdd) {
      C += A * B;
    } else {
      C = A * B;
    }
  }
}

} // namespace mochi

namespace mochi::krylov {

/// @brief Routine to compute the eigendecomposition of a pencil (A, M), where A is a symmetric
/// matrix and M is a symmetric positive definite matrix.
/// @note It is a wrapper around Eigen's generalized self-adjoint eigendecomposition.
void GeneralizedSelfAdjointEigenDecomposition(
    MatrixView<float> const& A,
    MatrixView<float> const& M,
    Matrix<float>& V,
    ColumnVector<float>& D);

/// @brief Routine to compute the eigendecomposition of a pencil (A, M), where A is a symmetric
/// matrix and M is a symmetric positive definite matrix.
/// @note It is a wrapper around Eigen's generalized self-adjoint eigendecomposition.
void GeneralizedSelfAdjointEigenDecomposition(
    MatrixView<double> const& A,
    MatrixView<double> const& M,
    Matrix<double>& V,
    ColumnVector<double>& D);

/// @brief Routine to compute the eigendecomposition of a symmetric matrix A.
/// @note It is a wrapper around Eigen's self-adjoint eigendecomposition.
void SelfAdjointEigenDecomposition(
    MatrixView<float const> const& A,
    Matrix<float>& V,
    ColumnVector<float>& D);

/// @brief Routine to compute the eigendecomposition of a symmetric matrix A.
/// @note It is a wrapper around Eigen's self-adjoint eigendecomposition.
void SelfAdjointEigenDecomposition(
    MatrixView<double const> const& A,
    Matrix<double>& V,
    ColumnVector<double>& D);

/// @brief Extract a square block starting on a diagonal entry.
/// @param[in] A Block sparse matrix to extract from.
/// @param[in] startRow Starting row of the block to extract.
/// @param[in] len Length for indicating that the block is of size len x len.
/// @return Dense matrix storing the block.
///
/// @note The block size for the output matrix does not need to match the block size from the block
/// sparse matrix A.
/// @note The function does not assume that the sparse entries are sorted by block-row.
/// @note If we can assume that the entries are sorted, some speed-up could be obtained.
template <
    int kBlockSizeOut,
    Direction kMajorDirOut,
    typename Scalar,
    int kBlockSizeIn,
    typename Idx,
    typename Ptr,
    template <typename, typename...> typename InputStorage>
auto GetBlockDiagonal(
    BlockSparseMatrix<Scalar, kBlockSizeIn, Idx, Ptr, InputStorage> const& A,
    std::remove_const_t<Idx> startRow,
    int len) {
  if constexpr (kBlockSizeOut != kDynamic) {
    MOCHI_ASSERT_VERBOSE(len == kBlockSizeOut, "Inconsistent length.");
  }
  MOCHI_ASSERT_VERBOSE(
      (startRow + len <= A.Rows()) && (startRow + len <= A.Cols()),
      "Block dimensions are out of range");
  using NonConstScalar = std::remove_const_t<Scalar>;
  Matrix<NonConstScalar, kBlockSizeOut, kBlockSizeOut, kMajorDirOut> diagBlock(len, len);
  if constexpr ((kBlockSizeOut == kBlockSizeIn) || (kBlockSizeOut == krylov::kDynamic)) {
    auto const bsmRowBlock = static_cast<Idx>(startRow / kBlockSizeIn);
    if ((bsmRowBlock * kBlockSizeIn == startRow) && (len == kBlockSizeIn)) {
      auto const blockColIdx = A.Indices(bsmRowBlock);
      auto ptr = std::find(blockColIdx.begin(), blockColIdx.end(), bsmRowBlock);
      if (ptr == blockColIdx.end()) {
        diagBlock.SetZero();
      } else {
        auto const blockValues = A.Values(bsmRowBlock);
        diagBlock = blockValues[static_cast<int>(ptr - blockColIdx.begin())];
      }
      return diagBlock;
    }
  }

  // General case
  using NonConstIdx = std::remove_const_t<Idx>;
  diagBlock.SetZero();
  int br = 0;
  for (; br < len;) {
    auto const globalRow = startRow + static_cast<Idx>(br);
    auto const bsmRowBlock = static_cast<Idx>(globalRow / kBlockSizeIn);
    auto bsmLocalRow = static_cast<Idx>(globalRow - bsmRowBlock * kBlockSizeIn);
    auto rWidth = Min(len - br, kBlockSizeIn - bsmLocalRow);
    auto const colIndices = A.Indices(bsmRowBlock);
    auto const blockValues = A.Values(bsmRowBlock);
    int fillCount = 0;
    for (NonConstIdx ic = 0; ((ic < colIndices.size()) && (fillCount < len)); ++ic) {
      auto const newShift = static_cast<Idx>(colIndices[ic] * kBlockSizeIn);
      if ((newShift + kBlockSizeIn <= startRow) || (newShift >= startRow + len)) {
        continue;
      }
      auto cstart = Max<int>(0, startRow - newShift);
      auto cend = Min<int>(kBlockSizeIn, len + startRow - newShift);
      auto clen = cend - cstart;
      diagBlock.Block(br, cstart + newShift - startRow, rWidth, clen) =
          blockValues[ic].Block(bsmLocalRow, cstart, rWidth, clen);
      fillCount += clen;
    }
    br += rWidth;
  }
  return diagBlock;
}

/// @brief Extract a square block starting on a diagonal entry.
/// @param[in] A Dense matrix to extract from.
/// @param[in] startRow Starting row of the block to extract.
/// @param[in] len Length for indicating that the block is of size len x len.
/// @return Dense matrix storing the block.
template <
    int kBlockSizeOut,
    Direction kMajorDirOut,
    typename Scalar,
    int kRows,
    int kCols,
    Direction kMajorDirIn,
    Ownership kOwnership,
    int kMajorDim>
auto GetBlockDiagonal(
    Matrix<Scalar, kRows, kCols, kMajorDirIn, kOwnership, kMajorDim> const& A,
    int startRow,
    int len) {
  if constexpr (kBlockSizeOut != kDynamic) {
    MOCHI_ASSERT_VERBOSE(len == kBlockSizeOut, "Inconsistent length.");
  }
  MOCHI_ASSERT_VERBOSE(
      (startRow + len <= A.Rows()) && (startRow + len <= A.Cols()),
      "Block dimensions are out of range");
  return Matrix<std::remove_const_t<Scalar>, kBlockSizeOut, kBlockSizeOut, kMajorDirOut>{
      A.template Block<kBlockSizeOut, kBlockSizeOut>(startRow, startRow, len, len)};
}

/// @brief Extract a square block starting on a diagonal entry.
/// @param[in] A Sparse matrix to extract from.
/// @param[in] startRow Starting row of the block to extract.
/// @param[in] len Length for indicating that the block is of size len x len.
/// @return Dense matrix storing the block.
///
/// @note The function does not assume that the sparse entries are sorted by row.
/// @note If we can assume that the entries are sorted, some speed-up could be obtained.
template <
    int kBlockSizeOut,
    Direction kMajorDirOut,
    typename ScalarA,
    typename Idx,
    typename Ptr,
    template <typename, typename...> typename InputStorage>
auto GetBlockDiagonal(
    SparseMatrix<ScalarA, Idx, Ptr, InputStorage> const& A,
    std::remove_const_t<Idx> startRow,
    int len) {
  if constexpr (kBlockSizeOut != kDynamic) {
    MOCHI_ASSERT_VERBOSE(len == kBlockSizeOut, "Inconsistent length.");
  }
  MOCHI_ASSERT_VERBOSE(
      (startRow + len <= A.Rows()) && (startRow + len <= A.Cols()),
      "Block dimensions are out of range");
  using Scalar = std::remove_const_t<ScalarA>;
  auto diagBlock = Matrix<Scalar, kBlockSizeOut, kBlockSizeOut, kMajorDirOut>::Zero(len, len);
  for (int ir = 0; ir < len; ++ir) {
    auto const row = static_cast<Idx>(ir + startRow);
    auto const colIndices = A.Indices(row);
    auto const colValues = A.Values(row);
    int fillCount = 0;
    for (int ic = 0; ((ic < colIndices.size()) && (fillCount < len)); ++ic) {
      auto const localCol = static_cast<int>(colIndices[ic] - startRow);
      if ((localCol < 0) || (localCol >= len)) {
        continue;
      }
      diagBlock(ir, localCol) = colValues[ic];
      fillCount += 1;
    }
  } // for (int ir = 0; ir < len; ++ir)
  return diagBlock;
}

/// @brief Extract a square block starting on a diagonal entry.
/// @param[in] A AnyMatrixView to extract from.
/// @param[in] startRow Starting row of the block to extract.
/// @param[in] len Length of the block to extract, i.e. the extracted block is of size len x len.
/// @return Dense matrix storing the block.
template <int kBlockSizeOut, Direction kMajorDirOut, typename Scalar, typename Idx>
auto GetBlockDiagonal(AnyMatrixView<Scalar const> const& A, Idx startRow, int len) {
  return std::visit(
      [&](auto const& mat) {
        return GetBlockDiagonal<kBlockSizeOut, kMajorDirOut>(mat, startRow, len);
      },
      A);
}

/// @brief Extract a square block starting on a diagonal entry.
/// @param[in] A ActorPseudoMatrix to extract from.
/// @param[in] startRow Starting row of the block to extract, relative to the actor offset.
/// @param[in] len Length of the block to extract, i.e. the extracted block is of size len x len.
/// @return Dense matrix storing the block.
/// @remark Interaction matrices that overlap with the block diagonal must be square and have the
/// same row and col offset. Other layouts are not supported.
template <int kBlockSizeOut, Direction kMajorDirOut, typename Scalar, typename Idx>
auto GetBlockDiagonal(ActorPseudoMatrix<Scalar> const& A, Idx startRow, int len) {
  MOCHI_ASSERT_VERBOSE(
      len >= 0 && startRow >= 0 && startRow + len <= A.Rows(), "Invalid diagonal range.");
  auto D = GetBlockDiagonal<kBlockSizeOut, kMajorDirOut>(A.actorMatrix, startRow, len);
  for (auto const& [rOffset, cOffset, anyMat, _] : A.interactionMatrices) {
    // Skip matrices that do not overlap with the block diagonal.
    // TODO[T175051452]: Deal with too small blocks
    if ((A.offset + startRow + len <= Max(rOffset, cOffset)) ||
        (A.offset + startRow >= rOffset + GetNumRows(anyMat)) ||
        (A.offset + startRow >= cOffset + GetNumCols(anyMat))) {
      continue;
    }
    MOCHI_ASSERT(
        rOffset == cOffset && GetNumRows(anyMat) == GetNumCols(anyMat),
        "Unsupported interaction matrix layout.");
    int const rBegin = Max(A.offset + startRow, rOffset);
    int const rEnd = Min(A.offset + startRow + len, rOffset + GetNumRows(anyMat));
    int const lenToExtract = rEnd - rBegin;
    if (lenToExtract == kBlockSizeOut) {
      D += GetBlockDiagonal<kBlockSizeOut, kMajorDirOut>(anyMat, rBegin - rOffset, kBlockSizeOut);
    } else {
      D.Block(
          rBegin - (A.offset + startRow),
          rBegin - (A.offset + startRow),
          lenToExtract,
          lenToExtract) +=
          GetBlockDiagonal<krylov::kDynamic, kMajorDirOut>(anyMat, rBegin - rOffset, lenToExtract);
    }
  }
  return D;
}

/// @brief Extract blocks of size kBlockSize x kBlockSize along the diagonal.
/// @tparam Scalar Scalar type.
/// @tparam kBlockSize Block size.
/// @tparam kDir Orientation direction for the extracted block.
/// @tparam InputType Type of the matrix to get the blocks from.
/// @param[in] A Matrix to get the blocks from.
/// @param[out] diagonal Span of matrices with the diagonal blocks.
template <typename Scalar, int kBlockSize, Direction kDir, typename InputType>
void ExtractBlockDiagonal(
    InputType const& A,
    Span<Matrix<Scalar, kBlockSize, kBlockSize, kDir>> diagonal) {
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix should be square");
  MOCHI_ASSERT_VERBOSE(A.Rows() % kBlockSize == 0, "Incompatible matrix size");
  using Idx = std::remove_const_t<decltype(A.Rows())>;
  Idx const numBlockRows = A.Rows() / kBlockSize;
  MOCHI_ASSERT_VERBOSE(diagonal.size() >= numBlockRows, "Insufficient memory in the span");
  if (numBlockRows == 0)
    MOCHI_UNLIKELY {
      return;
    }
  // TODO: Fine tune value of minPerTask.
  auto const minPerTask = Clamp(9216 / (kBlockSize * kBlockSize), 1, numBlockRows);
  ParallelForRange(
      "ExtractBlockDiagonal",
      0,
      numBlockRows,
      minPerTask,
      numBlockRows,
      [&](Idx brBegin, Idx brEnd) {
        for (Idx br = brBegin; br < brEnd; ++br) {
          diagonal[br] = GetBlockDiagonal<kBlockSize, kDir>(A, kBlockSize * br, kBlockSize);
        }
      });
}

template <typename Scalar, typename InputType>
void ExtractDiagonal(InputType const& A, Span<Scalar> diagonal) {
  using Idx = std::remove_const_t<decltype(A.Rows())>;
  auto const rMin = Min(A.Rows(), A.Cols());
  MOCHI_ASSERT_VERBOSE(isize(diagonal) >= rMin, "Insufficient span size");
  for (Idx ir = 0; ir < rMin; ++ir) {
    diagonal[ir] = A(ir, ir);
  }
}

template <
    typename Scalar,
    typename ScalarA,
    typename Idx,
    typename Ptr,
    template <typename, typename...> typename InputStorage>
void ExtractDiagonal(
    SparseMatrix<ScalarA, Idx, Ptr, InputStorage> const& A,
    Span<Scalar> diagonal) {
  using NonConstIdx = std::remove_const_t<Idx>;
  auto const rMin = Min(A.Rows(), A.Cols());
  MOCHI_ASSERT_VERBOSE(isize(diagonal) >= rMin, "Insufficient span size");
  auto aPtr = A.Pointers();
  auto aIdx = A.Indices();
  auto aVal = A.Values();
  auto first = aIdx.data();
  for (NonConstIdx ir = 0; ir < rMin; ++ir) {
    auto last = first + aPtr[ir + 1] - aPtr[ir];
    auto iter = std::find(first, last, ir);
    diagonal[ir] = (iter == last) ? Scalar(0) : aVal[aPtr[ir] + static_cast<int>(iter - first)];
    first = last;
  }
}

template <
    typename Scalar,
    typename InputScalar,
    int kBlockSize,
    typename InputPtr,
    typename InputIdx,
    template <typename, typename...> typename Storage>
void ExtractDiagonal(
    BlockSparseMatrix<InputScalar, kBlockSize, InputIdx, InputPtr, Storage> const& A,
    Span<Scalar> diagonal) {
  using NonConstIdx = std::remove_const_t<InputIdx>;
  auto const rMin = Min(A.BlockRows(), A.BlockCols());
  MOCHI_ASSERT_VERBOSE(isize(diagonal) >= rMin * kBlockSize, "Insufficient span size");
  auto aPtr = A.Pointers();
  auto aIdx = A.Indices();
  auto first = aIdx.data();
  for (NonConstIdx i = 0; i < rMin; ++i) {
    auto last = first + aPtr[i + 1] - aPtr[i];
    auto iter = std::find(first, last, i);
    auto values = A.Values(i);
    if (iter == last) {
      for (int k = 0; k < kBlockSize; ++k) {
        diagonal[k + i * kBlockSize] = Scalar(0);
      }
    } else {
      auto j = static_cast<int>(iter - first);
      auto diagBlock = values[j];
      for (int k = 0; k < kBlockSize; ++k) {
        diagonal[k + i * kBlockSize] = diagBlock(k, k);
      }
    }
    first = last;
  }
}

namespace details {

template <int kBlockSize, typename ScalarA, typename ScalarD>
bool TryExtractDiagonalBlockSparse(
    ActorPseudoMatrix<ScalarA> const& A,
    int rOffset,
    AnyMatrixView<ScalarA const> const& interactionMatrix,
    Span<ScalarD> diagonal) {
  if (auto const* bsp =
          std::get_if<BlockSparseMatrixView<ScalarA const, kBlockSize>>(&interactionMatrix)) {
    for (int br = Max(0, (A.offset - rOffset) / kBlockSize); br < bsp->BlockRows(); ++br) {
      int const shift = rOffset + br * kBlockSize - A.offset;
      if (shift >= A.Rows()) {
        MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "ActorPseudoMatrix is assumed to be square.");
        break;
      }
      auto const colIdx = bsp->Indices(br);
      auto const values = bsp->Values(br);
      auto ptr = std::lower_bound(colIdx.begin(), colIdx.end(), br);
      if ((ptr == colIdx.end()) || (*ptr != br)) {
        ptr = std::find(colIdx.begin(), colIdx.end(), br);
        if (ptr == colIdx.end()) {
          continue;
        }
      }
      auto const block = values[static_cast<int>(ptr - colIdx.begin())];
      for (int k = Max(0, -shift), r = k + shift; (k < kBlockSize) && (r < A.Rows()); ++k, ++r) {
        diagonal[r] += block(k, k);
      }
    }
    return true;
  } else {
    return false;
  }
}

} // namespace details

/// @remark Interaction matrices that overlap with the diagonal must be square and have the same row
/// and col offset. Other layouts are not supported.
template <typename ScalarD, typename ScalarA>
void ExtractDiagonal(ActorPseudoMatrix<ScalarA> const& A, Span<ScalarD> diagonal) {
  // TODO[T175051452]: Optimize implementation.
  static_assert(std::is_same_v<std::remove_const_t<ScalarA>, ScalarD>, "Inconsistent scalar types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Actor pseudo matrix must be square.");
  MOCHI_ASSERT_VERBOSE(isize(diagonal) >= A.Rows(), "Insufficient span size");
  std::visit([&](auto const& mat) { ExtractDiagonal(mat, diagonal); }, A.actorMatrix);
  for (auto const& [rOffset, cOffset, interactionMatrix, _] : A.interactionMatrices) {
    if ((A.offset + A.Rows() <= rOffset) || (A.offset + A.Cols() <= cOffset) ||
        (A.offset >= rOffset + GetNumRows(interactionMatrix)) ||
        (A.offset >= cOffset + GetNumCols(interactionMatrix))) {
      continue;
    }
    MOCHI_ASSERT(
        rOffset == cOffset && GetNumRows(interactionMatrix) == GetNumCols(interactionMatrix),
        "Unsupported interaction matrix layout.");

    static_assert(
        std::variant_size_v<decltype(interactionMatrix)> == 4,
        "Please update the if statement below if the interaction matrix types change");
    if (auto const* dense = std::get_if<MatrixView<ScalarA const>>(&interactionMatrix)) {
      int const rBegin = Max(A.offset, rOffset);
      int const rEnd = Min(A.offset + A.Rows(), rOffset + dense->Rows());
      for (int r = rBegin; r < rEnd; ++r) {
        diagonal[r - A.offset] += (*dense)(r - rOffset, r - rOffset);
      }
    } else if (auto const* sp = std::get_if<SparseMatrixView<ScalarA const>>(&interactionMatrix)) {
      int const rBegin = Max(A.offset, rOffset);
      int const rEnd = Min(A.offset + A.Rows(), rOffset + sp->Rows());
      for (int r = rBegin; r < rEnd; ++r) {
        diagonal[r - A.offset] += (*sp)(r - rOffset, r - rOffset);
      }
    } else if (details::TryExtractDiagonalBlockSparse<3>(A, rOffset, interactionMatrix, diagonal)) {
    } else if (details::TryExtractDiagonalBlockSparse<4>(A, rOffset, interactionMatrix, diagonal)) {
    } else {
      MOCHI_ASSERT(false, "Unexpected matrix type.");
    }
  }
}

template <typename Scalar>
bool HasOverlap(Scalar const* left, size_t leftLen, Scalar const* right, size_t rightLen) {
  auto const leftBegin = reinterpret_cast<std::uintptr_t>(left);
  auto const leftEnd = leftBegin + leftLen * sizeof(Scalar);
  auto const rightBegin = reinterpret_cast<std::uintptr_t>(right);
  auto const rightEnd = rightBegin + rightLen * sizeof(Scalar);
  return (leftBegin < rightEnd) && (rightBegin < leftEnd);
}

/// @brief Apply a block-diagonal matrix to a column vector (or a set of column vectors).
/// @param[in] diagBlock Storage of blocks as a vector of dense matrices.
/// @param[in] x Input column vector(s).
/// @param[out] y Output column vector(s).
///
/// @note x and y could represent the same object (i.e. application in-place).
template <typename Scalar, int kBlockSize, Direction kDir, typename Input, typename Output>
void ApplyBlockDiagonal(
    Span<Matrix<Scalar, kBlockSize, kBlockSize, kDir> const> diagBlock,
    Input const& xin,
    Output&& yout) {
  using Sx = std::remove_pointer_t<decltype(xin.data())>;
  using Sy = std::remove_pointer_t<decltype(yout.data())>;
  static_assert(std::is_same_v<Sx const, Sy const>, "Inconsistent scalar types");
  static_assert(std::is_same_v<Sx const, Scalar const>, "Inconsistent scalar types");
  //
  MOCHI_ASSERT_VERBOSE(
      diagBlock.size() * kBlockSize == xin.Rows(), "Inconsistent number of blocks");
  MOCHI_ASSERT_VERBOSE(
      xin.Rows() % kBlockSize == 0, "Inconsistent number of rows for input vector");
  MOCHI_ASSERT_VERBOSE(
      yout.Rows() % kBlockSize == 0, "Inconsistent number of rows for output vector");
  MOCHI_ASSERT_VERBOSE(xin.Rows() == yout.Rows(), "Inconsistent number of rows");
  MOCHI_ASSERT_VERBOSE(xin.Cols() == yout.Cols(), "Inconsistent number of columns");
  //
  MatrixView<
      typename details::MatTraits<Output>::Scalar,
      details::MatTraits<Output>::kNumRows,
      details::MatTraits<Output>::kNumCols,
      details::MatTraits<Output>::kMajorDir,
      kDynamic>
      yview(yout.data(), yout.Rows(), yout.Cols(), yout.LeadDim());
  //
  //--- Check whether xin or yout overlaps
  //
  bool needsTmp = HasOverlap(xin.data(), xin.StorageSize(), yview.data(), yview.StorageSize());
  //
  auto nRowBlocks = static_cast<int>(xin.Rows() / kBlockSize);
  if (needsTmp) {
    Matrix<
        std::remove_const_t<typename details::MatTraits<Input>::Scalar>,
        krylov::kDynamic,
        krylov::kDynamic,
        details::MatTraits<Input>::kMajorDir>
        x(xin);
    for (int ib = 0; ib < nRowBlocks; ++ib) {
      auto xBlock = x.template Block<kBlockSize>(ib * kBlockSize, 0, kBlockSize, x.Cols());
      auto yBlock = yview.template Block<kBlockSize>(ib * kBlockSize, 0, kBlockSize, yview.Cols());
      yBlock = diagBlock[ib] * xBlock;
    }
  } else {
    MatrixView<
        typename details::MatTraits<Input>::Scalar const,
        details::MatTraits<Input>::kNumRows,
        details::MatTraits<Input>::kNumCols,
        details::MatTraits<Input>::kMajorDir,
        kDynamic>
        x(xin.data(), xin.Rows(), xin.Cols(), xin.LeadDim());
    for (int ib = 0; ib < nRowBlocks; ++ib) {
      auto xBlock = x.template Block<kBlockSize>(ib * kBlockSize, 0, kBlockSize, x.Cols());
      auto yBlock = yview.template Block<kBlockSize>(ib * kBlockSize, 0, kBlockSize, yview.Cols());
      yBlock = diagBlock[ib] * xBlock;
    }
  }
}

/// @brief Apply a block-diagonal matrix to a column vector (or a set of column vectors).
/// @param[in] diagValues Storage of "blocks" as a vector of scalar (i.e. like 1x1 matrices).
/// @param[in] x Input column vector(s).
/// @param[out] y Output column vector(s).
///
/// @note x and y could represent the same object (i.e. application in-place).
template <typename Scalar, typename Input, typename Output>
void ApplyBlockDiagonal(Span<Scalar const> diagValues, Input const& x, Output&& y) {
  using Sx = std::remove_pointer_t<decltype(x.data())>;
  using Sy = std::remove_pointer_t<decltype(y.data())>;
  static_assert(std::is_same_v<Sx const, Sy const>, "Inconsistent scalar types");
  static_assert(std::is_same_v<Sx const, Scalar const>, "Inconsistent scalar types");

  MOCHI_ASSERT_VERBOSE(x.Rows() == y.Rows(), "Inconsistent number of rows");
  MOCHI_ASSERT_VERBOSE(x.Cols() == y.Cols(), "Inconsistent number of columns");
  MOCHI_ASSERT_VERBOSE(x.Rows() == diagValues.size(), "Inconsistent number of rows");

  for (int jc = 0; jc < x.Cols(); ++jc) {
    auto yj = y.Col(jc);
    auto xj = x.Col(jc);
    int irow = 0;
    using VType = Simd<std::remove_const_t<Scalar>>; // Native SIMD size.
    if constexpr (
        VType::kIsSupported && (details::MatTraits<Input>::kMajorDir == Direction::ColMajor) &&
        (details::MatTraits<Output>::kMajorDir == Direction::ColMajor)) {
      // TODO(T158480383): Introduce minimum SIMD size to favor the SIMD implementation.
      auto const* xptr = xj.data();
      auto* yptr = yj.data();
      using VTypeLarge = Simd<typename VType::Scalar, 3 * VType::kSize>; // 3x the native SIMD size.
      static_assert(
          VTypeLarge::kIsSupported, "Larger-than-native SIMD vector expected to be supported");
      for (; irow + VTypeLarge::kSize <= x.Rows(); irow += VTypeLarge::kSize) {
        auto vx = Load<VTypeLarge>(xptr + irow);
        auto vd = Load<VTypeLarge>(&diagValues[irow]);
        Store(yptr + irow, vd * vx);
      }
      for (; irow + VType::kSize <= x.Rows(); irow += VType::kSize) {
        auto vx = Load<VType>(xptr + irow);
        auto vd = Load<VType>(&diagValues[irow]);
        Store(yptr + irow, vd * vx);
      }
    }
    for (; irow < x.Rows(); ++irow) {
      yj(irow, 0) = diagValues[irow] * xj(irow, 0);
    }
  }
}

/// @brief Invert a set of matrices stored in a span.
/// @param[in,out] dSpan Span of matrices to be inverted in place.
/// @remark The matrices are inverted in place.
template <
    typename InputScalar,
    int kBlockSize,
    Direction kMajorDir,
    Ownership kOwnership,
    int kMajorDim>
void BatchedInverse(
    Span<Matrix<InputScalar, kBlockSize, kBlockSize, kMajorDir, kOwnership, kMajorDim>> dSpan) {
  using Idx = int;
  //-- At least ~5 us worth of work per thread.
  auto numBlocks = isize(dSpan);
  auto const minBlocksPerTask =
      Clamp<Idx>(15000 / (kBlockSize * kBlockSize * kBlockSize), 1, numBlocks);
  ParallelForRange(
      "BatchedInverse",
      0,
      numBlocks,
      minBlocksPerTask,
      numBlocks,
      [&](Idx blkRowBegin, Idx blkRowEnd) {
        for (auto iBlkRow = blkRowBegin; iBlkRow < blkRowEnd; ++iBlkRow) {
          dSpan[iBlkRow] = Inverse(dSpan[iBlkRow]);
        }
      });
}

} // namespace mochi::krylov
