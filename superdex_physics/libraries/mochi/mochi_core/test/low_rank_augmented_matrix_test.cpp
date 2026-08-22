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

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/low_rank_augmented_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <limits>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::test;

template <typename T>
static Matrix<T> GetTestMatrix(int numRows, int numCols) {
  Matrix<T> mat(numRows, numCols);
  mat.SetRandom(1);
  return mat;
}

template <typename MatrixType>
static void TestLowRankAugmentedMatrix(MatrixType const& A0) {
  int const numRows = A0.Rows();
  int const numCols = A0.Cols();
  int const numRankOneUpdates = 6;
  Matrix<real> U(numRows, numRankOneUpdates), V(numCols, numRankOneUpdates);
  U.SetRandom(1);
  V.SetRandom(2);

  Matrix<real> const asMatrix = ToMatrix(A0) + U * Transpose(V);
  LowRankAugmentedMatrix<MatrixType> augMat(A0, /*initialCapacity*/ numRankOneUpdates);

  // Sizes.
  EXPECT_EQ(numRows, augMat.Rows());
  EXPECT_EQ(numCols, augMat.Cols());
  EXPECT_EQ(0, augMat.NumRankOneUpdates());

  // Reset.
  augMat.Reset(A0, U, V);
  EXPECT_EQ(numRows, augMat.Rows());
  EXPECT_EQ(numCols, augMat.Cols());
  EXPECT_EQ(numRankOneUpdates, augMat.NumRankOneUpdates());

  // GetUnaugmentedMatrix.
  EXPECT_TRUE(NearEqualMatrices(ToMatrix(A0), ToMatrix(augMat.GetUnaugmentedMatrix())));

  // GetAugmentedMatrix.
  EXPECT_TRUE(NearEqualMatrices(asMatrix, augMat.GetAugmentedMatrix()));

  // Constructors involving rvalue's.
  augMat.Reset(MatrixType(A0)); // rvalue
  EXPECT_TRUE(NearEqualMatrices(ToMatrix(A0), augMat.GetAugmentedMatrix()));
  augMat.Reset(A0, Matrix<real>(U), Matrix<real>(V)); // lvalue, rvalue, rvalue
  EXPECT_TRUE(NearEqualMatrices(asMatrix, augMat.GetAugmentedMatrix()));
  augMat.Reset(MatrixType(A0), U, V); // rvalue, lvalue, lvalue
  EXPECT_TRUE(NearEqualMatrices(asMatrix, augMat.GetAugmentedMatrix()));
  augMat.Reset(MatrixType(A0), Matrix<real>(U), Matrix<real>(V)); // rvalue, rvalue, rvalue
  EXPECT_TRUE(NearEqualMatrices(asMatrix, augMat.GetAugmentedMatrix()));

  // Apply.
  real const applyTol = numCols * numRankOneUpdates * std::numeric_limits<real>::epsilon();
  int const numRhs = 4;
  Matrix<real> x(numCols, numRhs), y1(numRows, numRhs), y2(numRows, numRhs);
  x.SetRandom(1);
  y1.SetRandom(2);
  y2.SetRandom(3);

  Apply(augMat, x, y1);
  augMat.Apply(x, y2);
  Matrix<real> yExpected = asMatrix * x;

  EXPECT_TRUE(NearEqualMatrices(yExpected, y1, applyTol));
  EXPECT_TRUE(NearEqualMatrices(yExpected, y2, applyTol));

  // ApplyToRange.
  int const rowBegin = 3 * (numRows / 9); // Multiple of 3 for BlockSparseMatrix tests.
  int const rowEnd = 6 * (numRows / 9); // Multiple of 3 for BlockSparseMatrix tests.
  EXPECT_TRUE(rowBegin > 0 && rowBegin < rowEnd && rowEnd < numRows); // All codepaths are tested.

  y1.SetZero();
  y2.SetZero();
  ApplyToRange(augMat, x, y1, rowBegin, rowEnd);
  augMat.ApplyToRange(x, y2, rowBegin, rowEnd);

  yExpected.TopRows(rowBegin).SetZero();
  yExpected.BottomRows(numRows - rowEnd).SetZero();
  EXPECT_TRUE(NearEqualMatrices(yExpected, y1, applyTol));
  EXPECT_TRUE(NearEqualMatrices(yExpected, y2, applyTol));

  // ToMatrix.
  EXPECT_TRUE(NearEqualMatrices(asMatrix, ToMatrix(LowRankAugmentedMatrix<MatrixType>(A0, U, V))));

  // AddRankOneUpdates: All rank-one updates at once.
  LowRankAugmentedMatrix augMat2(
      A0, numRankOneUpdates); // With initial capacity and automatic template deduction.
  augMat2.AddRankOneUpdates(U, V);
  EXPECT_TRUE(NearEqualMatrices(ToMatrix(augMat), ToMatrix(augMat2)));

  // AddRankOneUpdates: One rank-one update at a time.
  augMat2.Reset(A0); // Without initial capacity.
  for (int i = 0; i < numRankOneUpdates; ++i) {
    augMat2.AddRankOneUpdates(U.Col(i), V.Col(i));
    EXPECT_TRUE(NearEqualMatrices(
        ToMatrix(LowRankAugmentedMatrix<MatrixType>(A0, U.LeftCols(i + 1), V.LeftCols(i + 1))),
        ToMatrix(augMat2)));
  }
}

TEST(LowRankAugmentedMatrix, kIsLowRankAugmentedMatrix) {
  // clang-format off
  static_assert(!IsLowRankAugmentedMatrix<real>);
  static_assert(!IsLowRankAugmentedMatrix<Matrix<real>>);
  static_assert(!IsLowRankAugmentedMatrix<SparseMatrix<real>>);
  static_assert(!IsLowRankAugmentedMatrix<BlockSparseMatrix<real, 3>>);
  static_assert(!IsLowRankAugmentedMatrix<IslandOperators<real>>);
  static_assert(IsLowRankAugmentedMatrix<LowRankAugmentedMatrix<Matrix<real>>>);
  static_assert(IsLowRankAugmentedMatrix<LowRankAugmentedMatrix<RowMatrix<real>>>);
  static_assert(IsLowRankAugmentedMatrix<LowRankAugmentedMatrix<MatrixView<real>>>);
  static_assert(IsLowRankAugmentedMatrix<LowRankAugmentedMatrix<MatrixView<real const>>>);
  static_assert(IsLowRankAugmentedMatrix<LowRankAugmentedMatrix<SparseMatrix<real>>>);
  static_assert(IsLowRankAugmentedMatrix<LowRankAugmentedMatrix<SparseMatrixView<real>>>);
  static_assert(IsLowRankAugmentedMatrix<LowRankAugmentedMatrix<BlockSparseMatrix<real, 3>>>);
  static_assert(IsLowRankAugmentedMatrix<LowRankAugmentedMatrix<BlockSparseMatrixView<real, 3>>>);
  // clang-format on
}

TEST(LowRankAugmentedMatrix, kIsLinearOperator) {
  static_assert(IsLinearOperator<LowRankAugmentedMatrix<Matrix<real>>>);
  static_assert(IsLinearOperator<LowRankAugmentedMatrix<RowMatrix<real>>>);
  static_assert(IsLinearOperator<LowRankAugmentedMatrix<MatrixView<real>>>);
  static_assert(IsLinearOperator<LowRankAugmentedMatrix<MatrixView<real const>>>);
  static_assert(IsLinearOperator<LowRankAugmentedMatrix<SparseMatrix<real>>>);
  static_assert(IsLinearOperator<LowRankAugmentedMatrix<SparseMatrixView<real>>>);
  static_assert(IsLinearOperator<LowRankAugmentedMatrix<BlockSparseMatrix<real, 3>>>);
  static_assert(IsLinearOperator<LowRankAugmentedMatrix<BlockSparseMatrixView<real, 3>>>);
}

TEST(LowRankAugmentedMatrix, FromMatrix) {
  Matrix<real> mat = GetTestMatrix<real>(/*numRows*/ 50, /*numCols*/ 40);

  // Col-major.
  TestLowRankAugmentedMatrix(mat);
  TestLowRankAugmentedMatrix(AsView(mat));
  TestLowRankAugmentedMatrix(AsConstView(mat));

  // Row-major.
  RowMatrix<real> rowMajor = mat.Transpose();
  TestLowRankAugmentedMatrix(rowMajor);
  TestLowRankAugmentedMatrix(AsView(rowMajor));
  TestLowRankAugmentedMatrix(AsConstView(rowMajor));
}

TEST(LowRankAugmentedMatrix, FromSparseMatrix) {
  Matrix<real> mat = GetTestMatrix<real>(/*numRows*/ 40, /*numCols*/ 50);
  SparseMatrix<real> sparse = ToSparseMatrix(mat);
  TestLowRankAugmentedMatrix(sparse);
  TestLowRankAugmentedMatrix(AsView(sparse));
  TestLowRankAugmentedMatrix(AsConstView(sparse));
}

TEST(LowRankAugmentedMatrix, FromBlockSparseMatrix) {
  Matrix<real> mat = GetTestMatrix<real>(/*numRows*/ 17 * 3, /*numCols*/ 11 * 3);
  BlockSparseMatrix<real, 3> blockSparse = ToBlockSparseMatrix<3>(mat);
  TestLowRankAugmentedMatrix(blockSparse);
  TestLowRankAugmentedMatrix(AsView(blockSparse));
  TestLowRankAugmentedMatrix(AsConstView(blockSparse));
}

TEST(LowRankAugmentedMatrix, FromIslandOperators) {
  // Actor matrices must be square and symmetric.
  Matrix<real> half = GetTestMatrix<real>(/*numRows*/ 50, /*numCols*/ 50);
  Matrix<real> mat = half + half.Transpose();
  std::vector<std::pair<int, AnyMatrixView<real const>>> actorMatrices = {{0, AsConstView(mat)}};
  IslandOperators<real> ops(actorMatrices, {}, {});
  TestLowRankAugmentedMatrix(ops);
}
