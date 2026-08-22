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

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <algorithm>
#include <iterator>
#include <tuple>
#include <variant>

using namespace mochi;

TEST(AnyMatrix, GetNumRowsCols) {
  // Matrix<real>
  {
    auto m = Matrix<real>{2, 3};
    EXPECT_EQ(2, GetNumRows(m));
    EXPECT_EQ(3, GetNumCols(m));
    EXPECT_EQ(2, GetNumBlockRows(m));
    EXPECT_EQ(6, GetNumValues(m));
    EXPECT_EQ(m.data(), GetValues(m).data());
    EXPECT_EQ(6, GetValues(m).size());

    auto any = AnyMatrix<real>{m}; // copy
    EXPECT_EQ(2, GetNumRows(any));
    EXPECT_EQ(3, GetNumCols(any));
    EXPECT_EQ(2, GetNumBlockRows(any));
    EXPECT_EQ(6, GetNumValues(any));
    EXPECT_EQ(std::get<Matrix<real>>(any).data(), GetValues(any).data());
    EXPECT_EQ(6, GetValues(any).size());

    auto anyView = AnyMatrixView<real>{m}; // View
    EXPECT_EQ(2, GetNumRows(anyView));
    EXPECT_EQ(3, GetNumCols(anyView));
    EXPECT_EQ(2, GetNumBlockRows(anyView));
    EXPECT_EQ(6, GetNumValues(anyView));
    EXPECT_EQ(m.data(), GetValues(anyView).data());
    EXPECT_EQ(6, GetValues(anyView).size());
  }

  // SparseMatrix<real>
  {
    auto m1 = SparseMatrix<real>{3, DynamicArray<int>{0, 0, 0}, {}, {}};
    auto m2 = SparseMatrix<real>{
        3,
        DynamicArray<int>{0, 3, 4},
        DynamicArray<int>{0, 1, 2, 2},
        DynamicArray<real>{0_r, 1_r, 2_r, 3_r}};
    EXPECT_EQ(2, GetNumRows(m1));
    EXPECT_EQ(3, GetNumCols(m1));
    EXPECT_EQ(2, GetNumBlockRows(m1));
    EXPECT_EQ(0, GetNumValues(m1));
    EXPECT_EQ(4, GetNumValues(m2));
    EXPECT_EQ(m1.Values().data(), GetValues(m1).data());
    EXPECT_EQ(m1.Values().size(), GetValues(m1).size());

    auto any1 = AnyMatrix<real>{m1}; // copy
    EXPECT_EQ(2, GetNumRows(any1));
    EXPECT_EQ(3, GetNumCols(any1));
    EXPECT_EQ(2, GetNumBlockRows(any1));
    EXPECT_EQ(0, GetNumValues(any1));
    EXPECT_EQ(std::get<SparseMatrix<real>>(any1).Values().data(), GetValues(any1).data());
    EXPECT_EQ(std::get<SparseMatrix<real>>(any1).Values().size(), GetValues(any1).size());

    auto any2 = AnyMatrixView<real>{m2}; // View
    EXPECT_EQ(2, GetNumRows(any2));
    EXPECT_EQ(3, GetNumCols(any2));
    EXPECT_EQ(2, GetNumBlockRows(any2));
    EXPECT_EQ(4, GetNumValues(any2));
    EXPECT_EQ(m2.Values().data(), GetValues(any2).data());
    EXPECT_EQ(m2.Values().size(), GetValues(any2).size());
  }

  // BlockSparseMatrix<real, 3>
  {
    auto m1 = BlockSparseMatrix<real, 3>{3, DynamicArray<int>{0, 0, 0}, {}, {}};
    auto m2 = BlockSparseMatrix<real, 3>{
        3,
        DynamicArray<int>{0, 1, 1},
        DynamicArray<int>{0},
        DynamicArray<real>{0_r, 1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r}};
    EXPECT_EQ(6, GetNumRows(m1));
    EXPECT_EQ(9, GetNumCols(m1));
    EXPECT_EQ(2, GetNumBlockRows(m1));
    EXPECT_EQ(0, GetNumValues(m1));
    EXPECT_EQ(9, GetNumValues(m2));
    EXPECT_EQ(m1.Values().data(), GetValues(m1).data());
    EXPECT_EQ(m1.Values().size(), GetValues(m1).size());

    auto any1 = AnyMatrix<real>{m1}; // copy
    EXPECT_EQ(6, GetNumRows(any1));
    EXPECT_EQ(9, GetNumCols(any1));
    EXPECT_EQ(2, GetNumBlockRows(any1));
    EXPECT_EQ(0, GetNumValues(any1));
    EXPECT_EQ((std::get<BlockSparseMatrix<real, 3>>(any1).Values().data()), GetValues(any1).data());
    EXPECT_EQ((std::get<BlockSparseMatrix<real, 3>>(any1).Values().size()), GetValues(any1).size());

    auto any2 = AnyMatrixView<real>{m2}; // View
    EXPECT_EQ(6, GetNumRows(any2));
    EXPECT_EQ(9, GetNumCols(any2));
    EXPECT_EQ(2, GetNumBlockRows(any2));
    EXPECT_EQ(9, GetNumValues(any2));
    EXPECT_EQ(m2.Values().data(), GetValues(any2).data());
    EXPECT_EQ(m2.Values().size(), GetValues(any2).size());
  }
}

TEST(AnyMatrix, AsView) {
  // | 1 0 2 0 3 0 |
  // | 0 4 0 5 0 6 |
  // | 7 0 8 0 9 0 |
  int constexpr kNumRows = 3;
  int constexpr kNumCols = 6;
  real constexpr kValuesDenseRowMajor[] = {
      1_r, 0_r, 2_r, 0_r, 3_r, 0_r, 0_r, 4_r, 0_r, 5_r, 0_r, 6_r, 7_r, 0_r, 8_r, 0_r, 9_r, 0_r};
  real constexpr kValuesDenseColMajor[] = {
      1_r, 0_r, 7_r, 0_r, 4_r, 0_r, 2_r, 0_r, 8_r, 0_r, 5_r, 0_r, 3_r, 0_r, 9_r, 0_r, 6_r, 0_r};

  // Expect the matrix above
  // Capture kNumRows and kNumCols explicitly (workaround for VS2019 + CPP20 bug)
  auto checkMat = [&, numRows = kNumRows, numCols = kNumCols](auto const& any) {
    std::visit(
        [&](auto const& m) {
          int i = 0;
          for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
              EXPECT_NEAR_EQ(kValuesDenseRowMajor[i++], m(r, c));
            }
          }
        },
        any);
  };

  // Matrix<real> (column major)
  {
    Matrix<real> mat{kNumRows, kNumCols};
    std::copy(std::begin(kValuesDenseColMajor), std::end(kValuesDenseColMajor), mat.data());
    AnyMatrix<real> any = mat;
    checkMat(any);
    checkMat(AsView(any));
    checkMat(AsConstView(any));
    checkMat(AsConstView(AsView(any)));
  }

  // SparseMatrix<real>
  {
    AnyMatrix<real> any = SparseMatrix<real>{
        kNumCols,
        DynamicArray<int>{0, 3, 6, 9},
        DynamicArray<int>{0, 2, 4, 1, 3, 5, 0, 2, 4},
        DynamicArray<real>{1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r, 8_r, 9_r}};
    checkMat(any);
    checkMat(AsView(any));
    checkMat(AsConstView(any));
    checkMat(AsConstView(AsView(any)));
  }

  // BlockSparse<real, 3>
  {
    AnyMatrix<real> any = BlockSparseMatrix<real, 3>{
        kNumCols / 3,
        DynamicArray<int>{0, 2},
        DynamicArray<int>{0, 1},
        DynamicArray<real>{
            1_r,
            0_r,
            2_r,
            0_r,
            3_r,
            0_r,
            0_r,
            4_r,
            0_r,
            5_r,
            0_r,
            6_r,
            7_r,
            0_r,
            8_r,
            0_r,
            9_r,
            0_r}};
    checkMat(any);
    checkMat(AsView(any));
    checkMat(AsConstView(any));
    checkMat(AsConstView(AsView(any)));
  }
}

TEST(AnyMatrix, SetZero) {
  // Matrix<real>
  {
    auto any = AnyMatrix<real>{Matrix<real>{2, 3}};
    auto& m = std::get<Matrix<real>>(any);
    m.SetConstant(1_r);
    EXPECT_TRUE(test::NearEqualMatrices(m, Matrix<real>{{1, 1}, {1, 1}, {1, 1}}));
    SetZero(any);
    EXPECT_TRUE(test::NearEqualMatrices(m, Matrix<real>{{0, 0}, {0, 0}, {0, 0}}));
  }

  // SparseMatrix<real>
  {
    auto any = AnyMatrix<real>{
        SparseMatrix<real>{2, {0, 1, 2}, {0, 1}, DynamicArray<real>(2)}}; // diagonal
    auto& m = std::get<SparseMatrix<real>>(any);
    std::fill(m.Values().begin(), m.Values().end(), 1_r);
    EXPECT_TRUE(test::NearEqualMatrices(m, Matrix<real>{{1, 0}, {0, 1}}));
    SetZero(any);
    EXPECT_TRUE(test::NearEqualMatrices(m, Matrix<real>{{0, 0}, {0, 0}}));
  }

  // BlockSparseMatrix<real, 3>
  {
    auto any = AnyMatrix<real>{BlockSparseMatrix<real, 3>{
        2, {0, 1, 2}, {0, 1}, DynamicArray<real>(3 * 3 * 2)}}; // diagonal blocks
    auto& m = std::get<BlockSparseMatrix<real, 3>>(any);
    std::fill(m.Values().begin(), m.Values().end(), 1_r);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            m,
            Matrix<real>{
                {1, 1, 1, 0, 0, 0},
                {1, 1, 1, 0, 0, 0},
                {1, 1, 1, 0, 0, 0},
                {0, 0, 0, 1, 1, 1},
                {0, 0, 0, 1, 1, 1},
                {0, 0, 0, 1, 1, 1}}));
    SetZero(any);
    EXPECT_TRUE(
        test::NearEqualMatrices(
            m,
            Matrix<real>{
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0}}));
  }
}

namespace mochi::test::details {

template <typename MatrixType>
void CheckGetBlockDiagonal(MatrixType const& mat) {
  AnyMatrix<real> any = mat;
  auto anyView = AsConstView(any);
  using mochi::krylov::GetBlockDiagonal;
  for (int startRow = 0; startRow < mat.Rows(); ++startRow) {
    for (int len = 0; len < mat.Rows() - startRow; ++len) {
      auto dMat =
          GetBlockDiagonal<krylov::kDynamic, krylov::Direction::ColMajor>(mat, startRow, len);
      auto D =
          GetBlockDiagonal<krylov::kDynamic, krylov::Direction::ColMajor>(anyView, startRow, len);
      EXPECT_TRUE(test::NearEqualMatrices(D, dMat));
      auto Dr =
          GetBlockDiagonal<krylov::kDynamic, krylov::Direction::RowMajor>(anyView, startRow, len);
      EXPECT_TRUE(test::NearEqualMatrices(Dr, dMat));
      if (len == 3) {
        auto D3 = GetBlockDiagonal<3, krylov::Direction::ColMajor>(anyView, startRow, len);
        EXPECT_TRUE(test::NearEqualMatrices(D3, dMat));
        auto D3r = GetBlockDiagonal<3, krylov::Direction::RowMajor>(anyView, startRow, len);
        EXPECT_TRUE(test::NearEqualMatrices(D3r, dMat));
      } else if (len == 2) {
        auto D2 = GetBlockDiagonal<2, krylov::Direction::ColMajor>(anyView, startRow, len);
        EXPECT_TRUE(test::NearEqualMatrices(D2, dMat));
        auto D2r = GetBlockDiagonal<2, krylov::Direction::RowMajor>(anyView, startRow, len);
        EXPECT_TRUE(test::NearEqualMatrices(D2r, dMat));
      } else if (len == 1) {
        auto D1 = GetBlockDiagonal<1, krylov::Direction::ColMajor>(anyView, startRow, len);
        EXPECT_TRUE(test::NearEqualMatrices(D1, dMat));
        auto D1r = GetBlockDiagonal<1, krylov::Direction::RowMajor>(anyView, startRow, len);
        EXPECT_TRUE(test::NearEqualMatrices(D1r, dMat));
      }
    }
  }
}

} // namespace mochi::test::details

TEST(AnyMatrixView, GetBlockDiagonal) {
  // |  1    -2.2  2   |
  // | -1.1   4   -4.4 |
  // |  7    -3.3  8   |
  int constexpr kNumRows = 3;
  real constexpr kValuesDenseColMajor[] = {1_r, -1.1_r, 7_r, -2.2_r, 4_r, -3.3_r, 2_r, -4.4_r, 8_r};
  Matrix<real> mat{kNumRows, kNumRows};
  std::copy(std::begin(kValuesDenseColMajor), std::end(kValuesDenseColMajor), mat.data());
  using test::details::CheckGetBlockDiagonal;

  // Matrix<real> (column major)
  CheckGetBlockDiagonal(mat);

  // Matrix<real> (row major)
  {
    auto const matT = mat.Transpose();
    CheckGetBlockDiagonal(matT);
  }

  // SparseMatrix<real>
  {
    auto spMat = ToSparseMatrix(mat);
    CheckGetBlockDiagonal(spMat);
  }

  // BlockSparse<real, 3>
  {
    auto bspMat = ToBlockSparseMatrix<3>(mat);
    CheckGetBlockDiagonal(bspMat);
    CheckGetBlockDiagonal(bspMat);
  }
}
