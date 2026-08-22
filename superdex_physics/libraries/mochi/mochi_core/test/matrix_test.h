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

/**
 * @file matrix_test.h
 * @brief Parametric test infrastructure for the Matrix class (matrix.h).
 *
 * @details Provides template functions that test element access, views, blocks, norms, and
 * arithmetic operators across all combinations of size (fixed/dynamic), direction
 * (ColMajor/RowMajor), and scalar type.
 */

#pragma once

#include <mochi_core/linear_algebra/matrix.h>

#include <mochi_core/linear_algebra/krylov/tools/custom_matrix_traits.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_traits.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace mochi::test {

/// Verifies sub-matrix elements against counter pattern: mat(r,c) = r * stride + c.
template <typename T, typename MatT>
inline void ExpectCounterSubMatrix(MatT const& view, int stride, int rowOffset, int colOffset) {
  for (int r = 0; r < view.Rows(); ++r) {
    for (int c = 0; c < view.Cols(); ++c) {
      EXPECT_EQ(static_cast<T>((r + rowOffset) * stride + (c + colOffset)), view(r, c));
    }
  }
}

template <typename T, bool kTestViews, typename MatrixT>
inline void TestMatrix(int numRows, int numCols, MatrixT&& mat) {
  // This function expects a Matrix filled by a counter value, which increases across
  // the rows. Example:
  // | 0 1 2 |
  // | 3 4 5 |

  EXPECT_EQ(numRows, mat.Rows());
  EXPECT_EQ(numCols, mat.Cols());
  EXPECT_EQ(numRows * numCols, mat.size());
  EXPECT_EQ(numRows == 0 || numCols == 0, mat.empty());

  // Test values individually
  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      EXPECT_EQ(static_cast<T>(r * numCols + c), mat(r, c));
    }
  }

  // Test the row vectors
  for (int r = 0; r < numRows; ++r) {
    auto row = mat.Row(r);
    EXPECT_EQ(1, row.Rows());
    EXPECT_EQ(numCols, row.Cols());
    for (int c = 0; c < numCols; ++c) {
      EXPECT_EQ(static_cast<T>(r * numCols + c), row[c]);
    }
  }

  // Test the column vectors
  for (int c = 0; c < numCols; ++c) {
    auto col = mat.Col(c);
    EXPECT_EQ(numRows, col.Rows());
    EXPECT_EQ(1, col.Cols());
    for (int r = 0; r < numRows; ++r) {
      EXPECT_EQ(static_cast<T>(r * numCols + c), col[r]);
    }
  }

  // Test Norm and NormSqr
  using RawMatrixT = std::remove_reference_t<MatrixT>;
  using Scalar = typename RawMatrixT::NonConstScalar;
  if constexpr (std::is_floating_point_v<Scalar>) {
    auto normTestRunner = [&](auto&& A) {
      auto norm = A.Norm();
      auto normSqr = A.NormSqr();
      EXPECT_NEAR_EQ(norm, Sqrt(normSqr));
      Scalar expectedNormSqr = {};
      for (int r = 0; r < A.Rows(); ++r) {
        for (int c = 0; c < A.Cols(); ++c) {
          expectedNormSqr += A(r, c) * A(r, c);
        }
      }
      EXPECT_NEAR_EQ(expectedNormSqr, normSqr);
    };

    normTestRunner(mat);
    if (numRows > 2 && numCols > 2) { // Test with leading dimension larger than size
      constexpr int kBlockRows =
          krylov::details::MatTraits<RawMatrixT>::kNumRows == krylov::kDynamic
          ? krylov::kDynamic
          : krylov::details::MatTraits<RawMatrixT>::kNumRows - 2;
      constexpr int kBlockCols =
          krylov::details::MatTraits<RawMatrixT>::kNumCols == krylov::kDynamic
          ? krylov::kDynamic
          : krylov::details::MatTraits<RawMatrixT>::kNumCols - 2;
      normTestRunner(mat.template Block<kBlockRows, kBlockCols>(1, 1, numRows - 2, numCols - 2));
    }
  }

  // Test operator+=, operator-=, operator*= and operator/= accessor-based assignments. Only
  // supported for floating point types.
  if constexpr (std::is_floating_point_v<Scalar>) {
    using OwningMatrix = decltype(krylov::MatrixFactoryType<RawMatrixT>{}.GetSameAs(mat));
    using OwningMatrixTranspose =
        decltype(krylov::MatrixFactoryType<decltype(Transpose(mat))>{}.GetSameAs(Transpose(mat)));
    OwningMatrix mat0 = mat, mat1 = mat;
    OwningMatrixTranspose mat2 = Transpose(mat), mat3 = Transpose(mat);
    mat0 += Scalar{2} * mat;
    mat1 -= Scalar{2} * mat;
    mat2 += Scalar{2} * Transpose(mat);
    mat3 -= Scalar{2} * Transpose(mat);
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        EXPECT_NEAR_EQ(Scalar{3} * mat(r, c), mat0(r, c));
        EXPECT_NEAR_EQ(Scalar{-1} * mat(r, c), mat1(r, c));
        EXPECT_NEAR_EQ(Scalar{3} * mat(r, c), mat2(c, r));
        EXPECT_NEAR_EQ(Scalar{-1} * mat(r, c), mat3(c, r));
      }
    }

    auto scaled = mat.Duplicate();
    scaled *= Scalar{2};
    for (int c = 0; c < numCols; ++c) {
      for (int r = 0; r < numRows; ++r) {
        EXPECT_NEAR_EQ(scaled(r, c), mat(r, c) * Scalar{2});
      }
    }
    scaled /= Scalar{2};
    for (int c = 0; c < numCols; ++c) {
      for (int r = 0; r < numRows; ++r) {
        EXPECT_NEAR_EQ(scaled(r, c), mat(r, c));
      }
    }
  }

  if constexpr (kTestViews) {
    // MiddleRows
    if (numRows >= 3) {
      auto midR = mat.MiddleRows(1, numRows - 2);
      EXPECT_EQ(numRows - 2, midR.Rows());
      EXPECT_EQ(numCols, midR.Cols());
      EXPECT_EQ(&mat(1, 0), &midR(0, 0));
      ExpectCounterSubMatrix<T>(midR, numCols, 1, 0);

      auto midR2 = mat.template MiddleRows<2>(1, 2);
      static_assert(std::is_same_v<decltype(midR2.CERows()), mochi::details::IntOrEmpty<2>>);
      static_assert(
          krylov::details::MatTraits<decltype(midR2)>::kNumCols ==
          krylov::details::MatTraits<RawMatrixT>::kNumCols);
      EXPECT_EQ(&mat(1, 0), &midR2(0, 0));
    }

    // MiddleCols
    if (numCols >= 3) {
      auto midC = mat.MiddleCols(1, numCols - 2);
      EXPECT_EQ(numRows, midC.Rows());
      EXPECT_EQ(numCols - 2, midC.Cols());
      EXPECT_EQ(&mat(0, 1), &midC(0, 0));
      ExpectCounterSubMatrix<T>(midC, numCols, 0, 1);

      auto midC2 = mat.template MiddleCols<2>(1, 2);
      static_assert(std::is_same_v<decltype(midC2.CECols()), mochi::details::IntOrEmpty<2>>);
      static_assert(
          krylov::details::MatTraits<decltype(midC2)>::kNumRows ==
          krylov::details::MatTraits<RawMatrixT>::kNumRows);
      EXPECT_EQ(&mat(0, 1), &midC2(0, 0));
    }

    // TopRows / BottomRows
    if (numRows >= 2) {
      auto top = mat.TopRows(2);
      EXPECT_EQ(2, top.Rows());
      EXPECT_EQ(numCols, top.Cols());
      EXPECT_EQ(&mat(0, 0), &top(0, 0));
      ExpectCounterSubMatrix<T>(top, numCols, 0, 0);

      auto bot = mat.BottomRows(2);
      EXPECT_EQ(2, bot.Rows());
      EXPECT_EQ(numCols, bot.Cols());
      EXPECT_EQ(&mat(numRows - 2, 0), &bot(0, 0));
      ExpectCounterSubMatrix<T>(bot, numCols, numRows - 2, 0);

      auto top2 = mat.template TopRows<2>(2);
      static_assert(std::is_same_v<decltype(top2.CERows()), mochi::details::IntOrEmpty<2>>);
      EXPECT_EQ(&mat(0, 0), &top2(0, 0));

      auto bot2 = mat.template BottomRows<2>(2);
      static_assert(std::is_same_v<decltype(bot2.CERows()), mochi::details::IntOrEmpty<2>>);
      EXPECT_EQ(&mat(numRows - 2, 0), &bot2(0, 0));
    }

    // LeftCols / RightCols
    if (numCols >= 2) {
      auto left = mat.LeftCols(2);
      EXPECT_EQ(numRows, left.Rows());
      EXPECT_EQ(2, left.Cols());
      EXPECT_EQ(&mat(0, 0), &left(0, 0));
      ExpectCounterSubMatrix<T>(left, numCols, 0, 0);

      auto right = mat.RightCols(2);
      EXPECT_EQ(numRows, right.Rows());
      EXPECT_EQ(2, right.Cols());
      EXPECT_EQ(&mat(0, numCols - 2), &right(0, 0));
      ExpectCounterSubMatrix<T>(right, numCols, 0, numCols - 2);

      auto left2 = mat.template LeftCols<2>(2);
      static_assert(std::is_same_v<decltype(left2.CECols()), mochi::details::IntOrEmpty<2>>);
      EXPECT_EQ(&mat(0, 0), &left2(0, 0));

      auto right2 = mat.template RightCols<2>(2);
      static_assert(std::is_same_v<decltype(right2.CECols()), mochi::details::IntOrEmpty<2>>);
      EXPECT_EQ(&mat(0, numCols - 2), &right2(0, 0));
    }

    // Block
    if (numRows >= 3 && numCols >= 3) {
      auto blk = mat.Block(1, 1, numRows - 2, numCols - 2);
      EXPECT_EQ(numRows - 2, blk.Rows());
      EXPECT_EQ(numCols - 2, blk.Cols());
      EXPECT_EQ(&mat(1, 1), &blk(0, 0));
      ExpectCounterSubMatrix<T>(blk, numCols, 1, 1);

      auto blk2 = mat.template Block<2, 2>(1, 1, 2, 2);
      static_assert(std::is_same_v<decltype(blk2.CERows()), mochi::details::IntOrEmpty<2>>);
      static_assert(std::is_same_v<decltype(blk2.CECols()), mochi::details::IntOrEmpty<2>>);
      EXPECT_EQ(&mat(1, 1), &blk2(0, 0));
      ExpectCounterSubMatrix<T>(blk2, numCols, 1, 1);

      // View-of-view: exercises non-automatic leading dimension codepaths.
      auto blkWide = mat.Block(1, 0, numRows - 2, numCols);
      auto midROnBlk = blkWide.MiddleRows(0, 1);
      EXPECT_EQ(&mat(1, 0), &midROnBlk(0, 0));
      ExpectCounterSubMatrix<T>(midROnBlk, numCols, 1, 0);

      auto midCOnBlk = blkWide.MiddleCols(1, numCols - 2);
      EXPECT_EQ(&mat(1, 1), &midCOnBlk(0, 0));
      ExpectCounterSubMatrix<T>(midCOnBlk, numCols, 1, 1);

      // Chaining: MiddleRows -> MiddleCols.
      auto midRC = mat.MiddleRows(1, numRows - 2).MiddleCols(1, numCols - 2);
      EXPECT_EQ(&mat(1, 1), &midRC(0, 0));
      ExpectCounterSubMatrix<T>(midRC, numCols, 1, 1);
    }

    // Diagonal
    auto diag = mat.Diagonal();
    static_assert(std::is_same_v<decltype(diag.CECols()), mochi::details::IntOrEmpty<1>>);
    static_assert(krylov::details::MatTraits<decltype(diag)>::kNumCols == 1);
    int const diagSize = Min(numRows, numCols);
    EXPECT_EQ(diagSize, diag.Rows());
    EXPECT_EQ(1, diag.Cols());
    EXPECT_EQ(&mat(0, 0), &diag[0]);
    for (int i = 0; i < diagSize; ++i) {
      EXPECT_EQ(T(i * numCols + i), diag[i]);
    }
  }
}

/// Tests Slice on column vectors (constrained to ColMajor + kCols==1).
template <typename T, typename VecT>
inline void TestSlice(int numRows, VecT&& vec) {
  if (numRows >= 3) {
    auto sl = vec.Slice(1, numRows - 2);
    EXPECT_EQ(numRows - 2, sl.Rows());
    EXPECT_EQ(1, sl.Cols());
    EXPECT_EQ(&vec[1], &sl[0]);
    for (int r = 0; r < sl.Rows(); ++r) {
      EXPECT_EQ(T(r + 1), sl[r]);
    }

    auto sl2 = vec.template Slice<2>(1, 2);
    static_assert(std::is_same_v<decltype(sl2.CERows()), mochi::details::IntOrEmpty<2>>);
    EXPECT_EQ(&vec[1], &sl2[0]);
    EXPECT_EQ(T(1), sl2[0]);
    EXPECT_EQ(T(2), sl2[1]);
  }
}

template <
    typename T,
    int kNumRowsAtCompileTime,
    int kNumColsAtCompileTime,
    krylov::Direction kDirection,
    bool kTestViews>
inline void TestMatrix(int numRows, int numCols) {
  // Construct the matrix
  using MatrixT = Matrix<T, kNumRowsAtCompileTime, kNumColsAtCompileTime, kDirection>;
  MatrixT mat(numRows, numCols);

  // Fill the matrix with a counter which increases across the rows, then cols.
  T* data = mat.data();
  int value = 0;
  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      if constexpr (kDirection == krylov::Direction::RowMajor) {
        data[r * numCols + c] = static_cast<T>(value);
      } else {
        data[c * numRows + r] = static_cast<T>(value);
      }
      ++value;
    }
  }

  TestMatrix<T, kTestViews>(numRows, numCols, mat);
  TestMatrix<T, kTestViews>(numRows, numCols, AsView(mat));
  TestMatrix<T, kTestViews>(numRows, numCols, AsConstView(mat));
  TestMatrix<T, kTestViews>(numRows, numCols, std::as_const(mat));

  // Test Slice on column vectors (Slice is only available for ColMajor + kCols==1).
  if constexpr (kNumColsAtCompileTime == 1 && kDirection == krylov::Direction::ColMajor) {
    TestSlice<T>(numRows, mat);
    TestSlice<T>(numRows, AsView(mat));
    TestSlice<T>(numRows, AsConstView(mat));
    TestSlice<T>(numRows, std::as_const(mat));
  }

  // Make a copy and prove that it is independent of the original
  auto matCopy = mat.Duplicate();
  memset(data, 0, numRows * numCols * sizeof(T));
  TestMatrix<T, kTestViews>(numRows, numCols, matCopy);
}

template <
    typename T,
    int kNumRows,
    int kNumCols,
    krylov::Direction kDirection,
    bool kTestViews = true>
inline void TestMatrixSizes() {
  // Static vs dynamic dimension
  TestMatrix<T, kNumRows, kNumCols, kDirection, kTestViews>(kNumRows, kNumCols);
  TestMatrix<T, kNumRows, krylov::kDynamic, kDirection, kTestViews>(kNumRows, kNumCols);
  TestMatrix<T, krylov::kDynamic, kNumCols, kDirection, kTestViews>(kNumRows, kNumCols);
  TestMatrix<T, krylov::kDynamic, krylov::kDynamic, kDirection, kTestViews>(kNumRows, kNumCols);
}

/// Tests SetZero across owner (memset path) and block view (strided loop path).
template <typename T, int kRows, int kCols, krylov::Direction kDir>
inline void TestSetZero(int numRows, int numCols) {
  // Owner with automatic lead dim (exercises contiguous memset path).
  Matrix<T, kRows, kCols, kDir> mat(numRows, numCols);
  mat.SetConstant(T(42));
  mat.SetZero();
  for (int c = 0; c < mat.Cols(); ++c) {
    for (int r = 0; r < mat.Rows(); ++r) {
      EXPECT_EQ(mat(r, c), T(0));
    }
  }

  // Block view with non-automatic lead dim (exercises strided loop path).
  Matrix<T, krylov::kDynamic, krylov::kDynamic, kDir> big(numRows + 4, numCols + 4);
  big.SetConstant(T(42));
  auto view = big.template Block<kRows, kCols>(0, 0, numRows, numCols);
  view.SetZero();
  for (int c = 0; c < numCols; ++c) {
    for (int r = 0; r < numRows; ++r) {
      EXPECT_EQ(view(r, c), T(0));
    }
  }
  // Verify padding was not touched.
  EXPECT_EQ(big(numRows + 1, numCols + 1), T(42));
}

template <typename T, krylov::Direction kDirection>
inline void TestMatrixSmallSizes() {
  // Various sizes
  TestMatrixSizes<T, 1, 1, kDirection>();
  TestMatrixSizes<T, 1, 3, kDirection>();
  TestMatrixSizes<T, 3, 1, kDirection>();
  TestMatrixSizes<T, 3, 3, kDirection>();
  TestMatrixSizes<T, 4, 4, kDirection>();
  TestMatrixSizes<T, 4, 5, kDirection>();
  TestMatrixSizes<T, 5, 4, kDirection>();
  TestMatrixSizes<T, 8, 8, kDirection>();
}

template <typename T, krylov::Direction kDirection>
inline void TestMatrixNx8_A() {
  // Test all SIMD codepaths along rows. View tests skipped (size-independent).
  TestMatrixSizes<T, 9, 8, kDirection, false>();
  TestMatrixSizes<T, 10, 8, kDirection, false>();
  TestMatrixSizes<T, 11, 8, kDirection, false>();
  TestMatrixSizes<T, 12, 8, kDirection, false>();
  TestMatrixSizes<T, 13, 8, kDirection, false>();
  TestMatrixSizes<T, 14, 8, kDirection, false>();
  TestMatrixSizes<T, 15, 8, kDirection, false>();
}

template <typename T, krylov::Direction kDirection>
inline void TestMatrixNx8_B() {
  // Test all SIMD codepaths along rows. View tests skipped (size-independent).
  TestMatrixSizes<T, 16, 8, kDirection, false>();
  TestMatrixSizes<T, 17, 8, kDirection, false>();
  TestMatrixSizes<T, 23, 8, kDirection, false>();
  TestMatrixSizes<T, 24, 8, kDirection, false>();
  TestMatrixSizes<T, 25, 8, kDirection, false>();
  TestMatrixSizes<T, 31, 8, kDirection, false>();
  TestMatrixSizes<T, 32, 8, kDirection, false>();
  TestMatrixSizes<T, 33, 8, kDirection, false>();
}

template <typename T, krylov::Direction kDirection>
inline void TestMatrix8xM_A() {
  // Test all SIMD codepaths along columns. View tests skipped (size-independent).
  TestMatrixSizes<T, 8, 9, kDirection, false>();
  TestMatrixSizes<T, 8, 10, kDirection, false>();
  TestMatrixSizes<T, 8, 11, kDirection, false>();
  TestMatrixSizes<T, 8, 12, kDirection, false>();
  TestMatrixSizes<T, 8, 13, kDirection, false>();
  TestMatrixSizes<T, 8, 14, kDirection, false>();
  TestMatrixSizes<T, 8, 15, kDirection, false>();
}

template <typename T, krylov::Direction kDirection>
inline void TestMatrix8xM_B() {
  // Test all SIMD codepaths along columns. View tests skipped (size-independent).
  TestMatrixSizes<T, 8, 16, kDirection, false>();
  TestMatrixSizes<T, 8, 17, kDirection, false>();
  TestMatrixSizes<T, 8, 23, kDirection, false>();
  TestMatrixSizes<T, 8, 24, kDirection, false>();
  TestMatrixSizes<T, 8, 25, kDirection, false>();
  TestMatrixSizes<T, 8, 31, kDirection, false>();
  TestMatrixSizes<T, 8, 32, kDirection, false>();
  TestMatrixSizes<T, 8, 33, kDirection, false>();
}

template <typename T, krylov::Direction kDirection>
inline void TestMatrixNx1() {
  // Column vector sizes — exercises Slice and MiddleRows col-vector codepaths.
  TestMatrixSizes<T, 5, 1, kDirection>();
  TestMatrixSizes<T, 9, 1, kDirection>();
  TestMatrixSizes<T, 17, 1, kDirection>();
}

template <typename T, krylov::Direction kDirection>
inline void TestMatrix1xM() {
  // Row vector sizes — exercises MiddleCols row-vector codepath.
  TestMatrixSizes<T, 1, 5, kDirection>();
  TestMatrixSizes<T, 1, 9, kDirection>();
  TestMatrixSizes<T, 1, 17, kDirection>();
}

template <typename T, krylov::Direction kDirection>
inline void TestMatrixSetZero() {
  // SetZero — exercises both contiguous (memset) and strided (loop) codepaths.
  TestSetZero<T, 3, 3, kDirection>(3, 3);
  TestSetZero<T, 4, 5, kDirection>(4, 5);
  TestSetZero<T, 4, krylov::kDynamic, kDirection>(4, 7);
  TestSetZero<T, krylov::kDynamic, 3, kDirection>(5, 3);
  TestSetZero<T, krylov::kDynamic, krylov::kDynamic, kDirection>(5, 7);
}

} // namespace mochi::test
