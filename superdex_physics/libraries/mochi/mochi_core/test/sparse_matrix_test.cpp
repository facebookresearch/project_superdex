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

#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix_utils.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/memory/monotonic_allocator.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mochi;

/**************************************************************************************
 * SparseMatrix Examples
 */

struct TestCase_3x3 {
  // |1 2|
  // | 3 |
  // |4 5|

  static constexpr int kNumCols = 3;
  static constexpr int kPointers[] = {0, 2, 3, 5};
  static constexpr int kIndices[] = {0, 2, 1, 0, 2};
  static constexpr real kValues[] = {1_r, 2_r, 3_r, 4_r, 5_r};
  static constexpr real kValuesDense[3][3] = {{1_r, 0_r, 2_r}, {0_r, 3_r, 0_r}, {4_r, 0_r, 5_r}};
};

struct TestCase_3x1 {
  // |1|
  // | |
  // |2|

  static constexpr int kNumCols = 1;
  static constexpr int kPointers[] = {0, 1, 1, 2};
  static constexpr int kIndices[] = {0, 0};
  static constexpr real kValues[] = {1_r, 2_r};
  static constexpr real kValuesDense[3][1] = {{1_r}, {0_r}, {2_r}};
};

struct TestCase_1x3 {
  // | 1 |

  static constexpr int kNumCols = 3;
  static constexpr int kPointers[] = {0, 1};
  static constexpr int kIndices[] = {1};
  static constexpr real kValues[] = {1_r};
  static constexpr real kValuesDense[1][3] = {{0_r, 1_r, 0_r}};
};

/**************************************************************************************
 * Helper Functions
 */

template <class TC>
static auto MakeMat() {
  return SparseMatrix<real>{
      TC::kNumCols,
      DynamicArray<int>(std::begin(TC::kPointers), std::end(TC::kPointers)),
      DynamicArray<int>(std::begin(TC::kIndices), std::end(TC::kIndices)),
      DynamicArray<real>(std::begin(TC::kValues), std::end(TC::kValues))};
}

// Return a new SparseMatrixView pointing to the data
template <class TC>
static auto MakeMatConstView() {
  return SparseMatrixView<real const>{
      TC::kNumCols, MakeSpan(TC::kPointers), MakeSpan(TC::kIndices), MakeSpan(TC::kValues)};
}

// Check that a SparseMatrix has the above data
template <class TC, class Mat>
static void CheckMat(Mat const& mat) {
  EXPECT_EQ(isize(TC::kValues), mat.NumNonZeros());
  EXPECT_EQ(isize(TC::kPointers) - 1, mat.Rows());
  EXPECT_EQ(TC::kNumCols, mat.Cols());
  EXPECT_EQ(mat.Rows(), mat.CERows().iVal());
  EXPECT_EQ(mat.Cols(), mat.CECols().iVal());
  EXPECT_SPAN_EQ(MakeSpan(TC::kPointers), mat.Pointers());
  EXPECT_SPAN_EQ(MakeSpan(TC::kIndices), mat.Indices());
  EXPECT_SPAN_EQ(MakeSpan(TC::kValues), mat.Values());
  for (int r = 0; r < mat.Rows(); ++r) {
    for (int c = 0; c < mat.Cols(); ++c) {
      EXPECT_NEAR_EQ(TC::kValuesDense[r][c], mat(r, c));
    }
  }
}

template <class Mat>
static void ExpectEmpty(Mat const& mat) {
  EXPECT_EQ(0, mat.NumNonZeros());
  EXPECT_EQ(0, mat.Rows());
  EXPECT_EQ(0, mat.Cols());
  EXPECT_EQ(0, mat.CERows().iVal());
  EXPECT_EQ(0, mat.CECols().iVal());
  EXPECT_EQ(0, isize(mat.Pointers()));
  EXPECT_EQ(0, isize(mat.Indices()));
  EXPECT_EQ(0, isize(mat.Values()));
}

template <class MatA, class MatB>
static bool HasSameAddresses(MatA const& a, MatB const& b) {
  return (a.Pointers().data() == b.Pointers().data()) &&
      (a.Indices().data() == b.Indices().data()) && (a.Values().data() == b.Values().data());
}

template <class TC, class Mat>
static void CheckPerRowAccessors(Mat const& mat) {
  for (int r = 0; r < mat.Rows(); ++r) {
    auto const indices = mat.Indices(r);
    auto const values = mat.Values(r);
    int const nnz = TC::kPointers[r + 1] - TC::kPointers[r];
    EXPECT_SPAN_EQ(Span(TC::kIndices + TC::kPointers[r], nnz), indices);
    EXPECT_SPAN_EQ(Span(TC::kValues + TC::kPointers[r], nnz), values);
    EXPECT_EQ(mat.Indices().data() + TC::kPointers[r], indices.data()); // Same address
    EXPECT_EQ(mat.Values().data() + TC::kPointers[r], values.data()); // Same address
  }
}

template <typename Scalar, typename CRIdx, typename Ptr, krylov::Direction kDenseMajorDir>
static void TestSparseMatrixVectorProduct() {
  // The test tolerances are designed for the worst case to prevent false positives, but are
  // sufficiently small to prevent false negatives. Absolute tolerances are used since the relative
  // error of the individual components is potentially unbounded.
  using MatDir = Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, kDenseMajorDir>;
  using MatOpDir = Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, ~kDenseMajorDir>;

  // Sparse matrix for testing.
  constexpr CRIdx kNumRows = 123;
  constexpr CRIdx kNumCols = kNumRows + 5; // Rectangular to increase test coverage.
  DynamicArray<Ptr> rowPtr;
  DynamicArray<CRIdx> colIdx;
  rowPtr.reserve(kNumRows + 1);
  colIdx.reserve(kNumRows * (kNumRows + 1) / 2);
  rowPtr.push_back(0);
  CRIdx maxNumNonZerosPerRow = 0;
  for (CRIdx ii = 0; ii < kNumRows; ++ii) {
    CRIdx const numNonZerosInRow = ii; // Different in each row to increase test coverage
    maxNumNonZerosPerRow = std::max(maxNumNonZerosPerRow, numNonZerosInRow);
    std::set<CRIdx> tmpColIdx;
    for (CRIdx jj = 0; jj < numNonZerosInRow; ++jj) {
      tmpColIdx.insert(
          static_cast<CRIdx>(ii + static_cast<int64_t>(jj) * kNumCols / numNonZerosInRow) %
          kNumCols);
    }
    for (auto idx : tmpColIdx) {
      colIdx.push_back(idx);
    }
    rowPtr.push_back(rowPtr.back() + numNonZerosInRow);
  }
  EXPECT_LE(maxNumNonZerosPerRow, kNumCols);

  Ptr const numNonZeros = rowPtr.back();
  DynamicArray<Scalar> values;
  values.reserve(numNonZeros);
  for (Ptr ii = 0; ii < numNonZeros; ++ii) {
    values.push_back(Scalar(2 * ii) / numNonZeros - Scalar(1));
  }

  SparseMatrix<Scalar, CRIdx, Ptr> C(kNumCols, rowPtr, colIdx, values);
  auto CAsView = AsView(C);
  auto CAsConstView = AsConstView(C);

  SparseMatrix<Scalar, CRIdx, Ptr> CT = Transpose(C);
  auto CTAsView = AsView(CT);
  auto CTAsConstView = AsConstView(CT);

  // Dense matrix analog of C. Used to generate the expected results in "Sparse matrix" x "Matrix"
  // tests.
  Matrix<Scalar> CD = ToMatrix(C);

  // Parameters for ApplyToRange tests.
  CRIdx const numRowsApply = kNumRows / 2;
  CRIdx const rowStartApply = 3;
  CRIdx const rowEndApply = rowStartApply + numRowsApply;
  EXPECT_LT(rowEndApply, kNumRows); // Not out of bounds.
  EXPECT_GT(rowStartApply, 0); // To increase test coverage.
  EXPECT_GT(numRowsApply, 0); // Not a dummy test.

  // "Sparse matrix" x "Column vector" tests.
  {
    ColumnVector<Scalar, kNumCols> x;
    ColumnVector<Scalar> y1(C.Rows()), y2(C.Rows()), y3(C.Rows()), y4(C.Rows());
    x.SetRandom(1, -1, 1); // [-1, 1] is the default. Just being explicit about this requirement.
    y1.SetRandom(2, -1, 1);
    y2.SetRandom(3, -1, 1);
    y3.SetRandom(4, -1, 1);
    y4.SetRandom(5, -1, 1);
    ColumnVector<Scalar> y40 = y4;
    C.Apply(x, y1);
    CAsView.Apply(x, y2);
    CAsConstView.Apply(x, y3);
    C.ApplyToRange(x, y4, rowStartApply, rowEndApply);
    ColumnVector<Scalar> y5 = C * x;
    ColumnVector<Scalar> y6 = CAsView * x;
    ColumnVector<Scalar> y7 = CAsConstView * x;
    ColumnVector<Scalar> y0 = CD * x;
    for (CRIdx ii = 0; ii < kNumRows; ++ii) {
      bool const isInApplyRange = (ii >= rowStartApply && ii < rowEndApply);
      CRIdx const nonZerosInRow = ii;
      Scalar const absTol = Scalar(nonZerosInRow + nonZerosInRow * (nonZerosInRow + 1) / 2) *
          std::numeric_limits<Scalar>::epsilon();
      EXPECT_NEAR(y1[ii], y0[ii], absTol);
      EXPECT_NEAR(y2[ii], y0[ii], absTol);
      EXPECT_NEAR(y3[ii], y0[ii], absTol);
      EXPECT_NEAR(y4[ii], isInApplyRange ? y0[ii] : y40[ii], absTol);
      EXPECT_NEAR(y5[ii], y0[ii], absTol);
      EXPECT_NEAR(y6[ii], y0[ii], absTol);
      EXPECT_NEAR(y7[ii], y0[ii], absTol);
    }
  }

  // "Sparse matrix transpose" x "Column vector" tests.
  {
    ColumnVector<Scalar, kNumCols> x;
    ColumnVector<Scalar> y1(CT.Cols()), y2(CT.Cols()), y3(CT.Cols());
    x.SetRandom(1, -1, 1);
    y1.SetRandom(2, -1, 1);
    y2.SetRandom(3, -1, 1);
    y3.SetRandom(4, -1, 1);
    CT.TransposeApply(x, y1);
    CTAsView.TransposeApply(x, y2);
    CTAsConstView.TransposeApply(x, y3);
    ColumnVector<Scalar> y0 = CD * x;
    for (CRIdx ii = 0; ii < kNumRows; ++ii) {
      CRIdx const nonZerosInRow = ii;
      Scalar const absTol = Scalar(nonZerosInRow + nonZerosInRow * (nonZerosInRow + 1) / 2) *
          std::numeric_limits<Scalar>::epsilon();
      EXPECT_NEAR(y1[ii], y0[ii], absTol);
      EXPECT_NEAR(y2[ii], y0[ii], absTol);
      EXPECT_NEAR(y3[ii], y0[ii], absTol);
    }
  }

  // "Sparse matrix" x "Matrix" tests.
  for (auto numColsDense : {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 64}) {
    MatDir U(CD.Cols(), numColsDense);
    MatDir X1(CD.Rows(), numColsDense), X2(CD.Rows(), numColsDense), X3(CD.Rows(), numColsDense);
    MatOpDir X4(CD.Rows(), numColsDense), X5(CD.Rows(), numColsDense), X6(CD.Rows(), numColsDense);
    MatDir Y1(CD.Rows(), numColsDense), Y2(CD.Rows(), numColsDense), Y3(CD.Rows(), numColsDense);
    MatOpDir Y4(CD.Rows(), numColsDense), Y5(CD.Rows(), numColsDense), Y6(CD.Rows(), numColsDense);

    // Expected result.
    U.SetRandom(1, -1, 1);
    MatDir XD = CD * U;

    // Apply() tests.
    X1.SetRandom(1, -1, 1);
    X2.SetRandom(2, -1, 1);
    X3.SetRandom(3, -1, 1);
    X4.SetRandom(4, -1, 1);
    X5.SetRandom(5, -1, 1);
    X6.SetRandom(6, -1, 1);
    C.Apply(U, X1);
    CAsView.Apply(U, X2);
    CAsConstView.Apply(U, X3);
    C.Apply(U, X4);
    CAsView.Apply(U, X5);
    CAsConstView.Apply(U, X6);
    for (CRIdx ii = 0; ii < XD.Rows(); ++ii) {
      CRIdx const nonZerosInRow = ii;
      Scalar const absTol = Scalar(nonZerosInRow + nonZerosInRow * (nonZerosInRow + 1) / 2) *
          std::numeric_limits<Scalar>::epsilon();
      for (CRIdx jj = 0; jj < XD.Cols(); ++jj) {
        EXPECT_NEAR(X1(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X2(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X3(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X4(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X5(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X6(ii, jj), XD(ii, jj), absTol);
      }
    }

    // ApplyToRange() tests.
    X1.SetRandom(7, -1, 1);
    X2.SetRandom(8, -1, 1);
    X3.SetRandom(9, -1, 1);
    X4.SetRandom(10, -1, 1);
    X5.SetRandom(11, -1, 1);
    X6.SetRandom(12, -1, 1);
    MatDir X10 = X1, X20 = X2, X30 = X3;
    MatOpDir X40 = X4, X50 = X5, X60 = X6;
    C.ApplyToRange(U, X1, rowStartApply, rowEndApply);
    CAsView.ApplyToRange(U, X2, rowStartApply, rowEndApply);
    CAsConstView.ApplyToRange(U, X3, rowStartApply, rowEndApply);
    C.ApplyToRange(U, X4, rowStartApply, rowEndApply);
    CAsView.ApplyToRange(U, X5, rowStartApply, rowEndApply);
    CAsConstView.ApplyToRange(U, X6, rowStartApply, rowEndApply);
    for (CRIdx ii = 0; ii < XD.Rows(); ++ii) {
      bool const isInApplyRange = (ii >= rowStartApply && ii < rowEndApply);
      CRIdx const nonZerosInRow = ii;
      Scalar const absTol = isInApplyRange
          ? Scalar(nonZerosInRow + nonZerosInRow * (nonZerosInRow + 1) / 2) *
              std::numeric_limits<Scalar>::epsilon()
          : Scalar(0);
      for (CRIdx jj = 0; jj < XD.Cols(); ++jj) {
        EXPECT_NEAR(X1(ii, jj), isInApplyRange ? XD(ii, jj) : X10(ii, jj), absTol);
        EXPECT_NEAR(X2(ii, jj), isInApplyRange ? XD(ii, jj) : X20(ii, jj), absTol);
        EXPECT_NEAR(X3(ii, jj), isInApplyRange ? XD(ii, jj) : X30(ii, jj), absTol);
        EXPECT_NEAR(X4(ii, jj), isInApplyRange ? XD(ii, jj) : X40(ii, jj), absTol);
        EXPECT_NEAR(X5(ii, jj), isInApplyRange ? XD(ii, jj) : X50(ii, jj), absTol);
        EXPECT_NEAR(X6(ii, jj), isInApplyRange ? XD(ii, jj) : X60(ii, jj), absTol);
      }
    }

    // operator* tests.
    X1.SetRandom(13, -1, 1);
    X2.SetRandom(14, -1, 1);
    X3.SetRandom(15, -1, 1);
    X4.SetRandom(16, -1, 1);
    X5.SetRandom(17, -1, 1);
    X6.SetRandom(18, -1, 1);
    X1 = C * U;
    X2 = CAsView * U;
    X3 = CAsConstView * U;
    X4 = C * U;
    X5 = CAsView * U;
    X6 = CAsConstView * U;
    for (CRIdx ii = 0; ii < XD.Rows(); ++ii) {
      CRIdx const nonZerosInRow = ii;
      Scalar const absTol = Scalar(nonZerosInRow + nonZerosInRow * (nonZerosInRow + 1) / 2) *
          std::numeric_limits<Scalar>::epsilon();
      for (CRIdx jj = 0; jj < XD.Cols(); ++jj) {
        EXPECT_NEAR(X1(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X2(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X3(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X4(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X5(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(X6(ii, jj), XD(ii, jj), absTol);
      }
    }

    // TransposeApply() tests.
    Y1.SetRandom(1, -1, 1);
    Y2.SetRandom(2, -1, 1);
    Y3.SetRandom(3, -1, 1);
    Y4.SetRandom(4, -1, 1);
    Y5.SetRandom(5, -1, 1);
    Y6.SetRandom(6, -1, 1);
    CT.TransposeApply(U, Y1);
    CTAsView.TransposeApply(U, Y2);
    CTAsConstView.TransposeApply(U, Y3);
    CT.TransposeApply(U, Y4);
    CTAsView.TransposeApply(U, Y5);
    CTAsConstView.TransposeApply(U, Y6);
    for (CRIdx ii = 0; ii < XD.Rows(); ++ii) {
      CRIdx const nonZerosInRow = ii;
      Scalar const absTol = Scalar(nonZerosInRow + nonZerosInRow * (nonZerosInRow + 1) / 2) *
          std::numeric_limits<Scalar>::epsilon();
      for (CRIdx jj = 0; jj < XD.Cols(); ++jj) {
        EXPECT_NEAR(Y1(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(Y2(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(Y3(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(Y4(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(Y5(ii, jj), XD(ii, jj), absTol);
        EXPECT_NEAR(Y6(ii, jj), XD(ii, jj), absTol);
      }
    }
  }
}

/**************************************************************************************
 * SparseMatrix Test Cases
 */

TEST(SparseMatrix, DefaultConstructor) {
  SparseMatrix<real> mat;
  ExpectEmpty(mat);
  SparseMatrixView<real> view;
  ExpectEmpty(view);
}

TEST(SparseMatrix, ConstructFromArrays) {
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();
  CheckMat<TC>(mat);
  auto view = MakeMatConstView<TC>();
  CheckMat<TC>(view);
}

TEST(SparseMatrix, MoveConstructor) {
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();
  auto mat2 = std::move(mat); // move construct from same type
  CheckMat<TC>(mat2);
  EXPECT_FALSE(HasSameAddresses(mat, mat2)); // NOLINT(bugprone-use-after-move)
  SparseMatrix<real, int, int, std::vector> mat3(std::move(mat2));
  CheckMat<TC>(mat3);
  CheckMat<TC>(mat2); // NOLINT(bugprone-use-after-move)
}

TEST(SparseMatrix, CopyConstructor) {
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();
  auto mat2 = mat; // copy construct from same type
  CheckMat<TC>(mat2);
  EXPECT_FALSE(HasSameAddresses(mat, mat2));
  auto view = MakeMatConstView<TC>();
  auto view2 = view; // copy construct from same type
  CheckMat<TC>(view2);
  EXPECT_TRUE(HasSameAddresses(view, view2));
  //
  SparseMatrix<real, int, int, std::vector> mat3(
      std::move(mat2)); // Move constructor from default to non-default storage type.
  CheckMat<TC>(mat3);
  // Move not possible from a different storage type. mat2 must have been copied.
  CheckMat<TC>(mat2); // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(HasSameAddresses(mat, mat3));
  //
  SparseMatrix<real, int, int> mat5(
      std::move(mat3)); // Move constructor from non-default to default storage type.
  CheckMat<TC>(mat5);
  // Move not possible from a different storage type. mat3 must have been copied.
  CheckMat<TC>(mat3); // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(HasSameAddresses(mat3, mat5));
}

TEST(SparseMatrix, ConstructFromOther) {
  using TC = TestCase_3x3;
  using SMat = SparseMatrix<real>;
  using SMatVector = SparseMatrix<real, int, int, std::vector>;
  using SMatView = SparseMatrixView<real>;
  using SMatConstView = SparseMatrixView<real const>;
  SMat mat = MakeMat<TC>();
  SMatVector matvector = mat;
  SMatView view = AsView(mat);
  SMatConstView cview = AsConstView(mat);

  // Owning from "std::vector"-matrix
  {
    SMat mat2 = matvector;
    CheckMat<TC>(mat2);
    EXPECT_FALSE(HasSameAddresses(mat2, matvector));
  }

  // Owning from view
  {
    SMat mat2 = view;
    CheckMat<TC>(mat2);
    EXPECT_FALSE(HasSameAddresses(mat2, view));
  }

  // Owning from const view
  {
    SMat mat2 = cview;
    CheckMat<TC>(mat2);
    EXPECT_FALSE(HasSameAddresses(mat2, cview));
  }

  // View from owning & default storage
  {
    SMatView view2 = mat;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(mat, view2));
  }

  // Const view from owning & default storage
  {
    SMatConstView view2 = mat;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(mat, view2));
  }

  // View from owning & "std::vector" storage
  {
    SMatView view2 = matvector;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(matvector, view2));
  }

  // Const view from owning & "std::vector" storage
  {
    SMatConstView view2 = matvector;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(matvector, view2));
  }

  // Const view from view
  {
    SMatConstView view2 = view;
    CheckMat<TC>(view2);
    EXPECT_TRUE(HasSameAddresses(view, view2));
  }
}

TEST(SparseMatrix, ConstructFromGraph) {
  // From empty graph
  {
    SparseMatrix<real> mat(0, Graph<int, int>{{}, {}});
    EXPECT_EQ(0, mat.Rows());
    EXPECT_EQ(0, mat.Cols());
  }

  // From 3x3 graph
  {
    using TC = TestCase_3x3;
    std::vector<int> pointers(std::begin(TC::kPointers), std::end(TC::kPointers));
    std::vector<int> indices(std::begin(TC::kIndices), std::end(TC::kIndices));
    Graph<int, int, std::vector> graph(pointers, indices);
    SparseMatrix<real, int, int, std::vector> mat(3, graph);
    EXPECT_EQ(graph.NumTargets(), mat.NumNonZeros());
    EXPECT_EQ(graph.NumTargets(), isize(mat.Values()));
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat.Values().begin());
    CheckMat<TC>(mat);

    // Repeat, but this time use std::move
    SparseMatrix<real, int, int, std::vector> mat2(3, std::move(graph));
    EXPECT_EQ(0, isize(graph.GetPointers())); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(0, isize(graph.GetTargets())); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(isize(mat.Values()), isize(mat2.Values()));
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat2.Values().begin());
    CheckMat<TC>(mat2);

    // If we do not specify the number of columns, then it will default to be a square matrix.
    // The number of rows still comes from the graph.
    SparseMatrix<real, int, int, std::vector> mat3(Graph<int, int, std::vector>(pointers, indices));
    EXPECT_EQ(3, mat3.Rows());
    EXPECT_EQ(3, mat3.Cols());
  }

  // From 3x3 graph
  {
    using TC = TestCase_3x3;
    int constexpr N = 2048;
    std::array<int, N> buffer1{};
    MonotonicAllocator mem1(buffer1.data(), N * sizeof(int));
    DynamicArray<int> pointers(std::begin(TC::kPointers), std::end(TC::kPointers), &mem1);
    DynamicArray<int> indices(std::begin(TC::kIndices), std::end(TC::kIndices), &mem1);
    EXPECT_GE(int(pointers.data() - buffer1.data()), 0);
    EXPECT_LT(int(pointers.data() - buffer1.data()), N);
    EXPECT_GE(int(indices.data() - buffer1.data()), 0);
    EXPECT_LT(int(indices.data() - buffer1.data()), N);

    //--- Make a copy of pointers and indices to preserve the original values
    DynamicArray<int> pg(pointers, &mem1), ig(indices, &mem1);
    EXPECT_GE(int(pg.data() - buffer1.data()), 0);
    EXPECT_LT(int(pg.data() - buffer1.data()), N);
    EXPECT_GE(int(ig.data() - buffer1.data()), 0);
    EXPECT_LT(int(ig.data() - buffer1.data()), N);

    //--- Move the copies into the graph
    //--- The graph should storage its data on "mem1"-pace
    Graph<int, int> graph(std::move(pg), std::move(ig));
    EXPECT_GE(int(graph.GetPointers().data() - buffer1.data()), 0);
    EXPECT_LT(int(graph.GetPointers().data() - buffer1.data()), N);
    EXPECT_GE(int(graph.GetTargets().data() - buffer1.data()), 0);
    EXPECT_LT(int(graph.GetTargets().data() - buffer1.data()), N);

    std::array<char, N * sizeof(real)> buffer2{};
    MonotonicAllocator mem2(buffer2.data(), N * sizeof(real));
    //--- The sparse matrix will store its integral containers on "new-delete"-space
    //--- and the scalar values on "mem2"-space
    SparseMatrix<real> mat(3, graph, &mem2);
    EXPECT_EQ(graph.NumTargets(), mat.NumNonZeros());
    EXPECT_EQ(graph.NumTargets(), isize(mat.Values()));
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat.Values().begin());
    CheckMat<TC>(mat);
    EXPECT_GE(int(mat.Values().begin() - (real*)(buffer2.data())), 0);
    EXPECT_LT(int(mat.Values().begin() - (real*)(buffer2.data())), N);

    // Repeat, but this time use std::move
    //--- The sparse matrix will store its integral containers on "mem1"-space
    //--- and the scalar values on "mem2"-space
    SparseMatrix<real> mat2(3, std::move(graph), &mem2);
    EXPECT_EQ(0, isize(graph.GetPointers())); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(0, isize(graph.GetTargets())); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(isize(mat.Values()), isize(mat2.Values()));
    std::copy(std::begin(TC::kValues), std::end(TC::kValues), mat2.Values().begin());
    CheckMat<TC>(mat2);
    EXPECT_GE(int(mat2.Pointers().data() - buffer1.data()), 0);
    EXPECT_LT(int(mat2.Pointers().data() - buffer1.data()), N);
    EXPECT_GE(int(mat2.Indices().data() - buffer1.data()), 0);
    EXPECT_LT(int(mat2.Indices().data() - buffer1.data()), N);
    EXPECT_GE(int(mat2.Values().begin() - (real*)(buffer2.data())), 0);
    EXPECT_LT(int(mat2.Values().begin() - (real*)(buffer2.data())), N);

    // If we do not specify the number of columns, then it will default to be a square matrix.
    // The number of rows still comes from the graph.
    //--- The sparse matrix will store its integral containers on "new-delete"-space
    //--- and the scalar values on "new-delete"-space
    SparseMatrix<real> mat3(Graph<int, int>(pointers, indices));
    EXPECT_EQ(3, mat3.Rows());
    EXPECT_EQ(3, mat3.Cols());
  }
}

TEST(SparseMatrix, AsView) {
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();
  auto view = AsView(mat);
  static_assert(std::is_same_v<decltype(view), SparseMatrixView<real>>);
  CheckMat<TC>(view);
  auto cview = AsConstView(mat);
  static_assert(
      std::is_same_v<decltype(cview), SparseMatrixView<real const, int const, int const>>);
  CheckMat<TC>(cview);
}

TEST(SparseMatrix, OtherSizes) {
  CheckMat<TestCase_3x1>(MakeMat<TestCase_3x1>());
  CheckMat<TestCase_1x3>(MakeMat<TestCase_1x3>());
}

TEST(SparseMatrix, MatrixVectorProduct) {
  // Specializations tested: Scalar={float,double}, CRIdx=int, Ptr={int, int64_t}, Matrix
  // Direction={ColMajor,RowMajor}.
  TestSparseMatrixVectorProduct<real, int, int, krylov::Direction::ColMajor>();
  TestSparseMatrixVectorProduct<real, int, int, krylov::Direction::RowMajor>();
  TestSparseMatrixVectorProduct<real, int, int64_t, krylov::Direction::ColMajor>();
  TestSparseMatrixVectorProduct<real, int, int64_t, krylov::Direction::RowMajor>();
// Skip it in non-debug builds.
// Edit this line to locally run it in non-debug builds.
#if MOCHI_DEBUG
  TestSparseMatrixVectorProduct<real, int, int16_t, krylov::Direction::ColMajor>();
  TestSparseMatrixVectorProduct<real, int, int16_t, krylov::Direction::RowMajor>();
#endif
}

namespace mochi::test::transpose {

// Specializations tested: Scalar=real, CRIdx=int, Ptr=int
template <krylov::Direction xDir, krylov::Direction yDir>
void TransposeMatVecProduct(int nCols) {
  int constexpr n = 4;
  DynamicArray<int> eRowPtr({0, 2, 5, 8, 9});
  DynamicArray<int> eColIdx({1, 2, 0, 1, 2, 1, 2, 3, 0});
  DynamicArray<real> eValues({1_r, -2_r, 3_r, -4_r, 5_r, -5.8_r, 6.1_r, -7.3_r, 0.5_r});
  SparseMatrix<real, int, int> E(n, eRowPtr, eColIdx, eValues);
  auto Edense = ToMatrix(E);
  auto Etdense = Transpose(Edense);
  Matrix<real, krylov::kDynamic, krylov::kDynamic, xDir> x(n, nCols);
  Matrix<real, krylov::kDynamic, krylov::kDynamic, yDir> y(n, nCols);
  Matrix<real, krylov::kDynamic, krylov::kDynamic, yDir> yref(n, nCols);
  x.SetRandom(11);
  y.SetRandom(13);
  E.TransposeApply(x, y);
  yref = Etdense * x;
  EXPECT_TRUE(test::NearEqualMatrices(y, yref));
}

} // namespace mochi::test::transpose

TEST(SparseMatrix, TransposeMatrixVectorProduct) {
  // Direction={ColMajor,RowMajor} x {ColMajor, RowMajor}
  using mochi::test::transpose::TransposeMatVecProduct;
  for (auto k : {1, 2, 3, 4, 8}) {
    TransposeMatVecProduct<krylov::Direction::ColMajor, krylov::Direction::ColMajor>(k);
    TransposeMatVecProduct<krylov::Direction::RowMajor, krylov::Direction::RowMajor>(k);
    TransposeMatVecProduct<krylov::Direction::ColMajor, krylov::Direction::RowMajor>(k);
    TransposeMatVecProduct<krylov::Direction::RowMajor, krylov::Direction::ColMajor>(k);
  }
}

TEST(SparseMatrix, CsrCsrProduct) {
  {
    using Scalar = real;
    int numRows = 3, numCols = numRows;
    DynamicArray<int> rowPtr({0, 2, 5, 7});
    DynamicArray<int> colIdx({0, 1, 0, 1, 2, 1, 2});
    auto const two = Scalar(2);
    auto const minusOne = Scalar(-1);
    DynamicArray<Scalar> values({two, minusOne, minusOne, two, minusOne, minusOne, two});

    SparseMatrix<Scalar, int, int> C(numCols, rowPtr, colIdx, values);
    auto D = C * C;
    auto row = D.Values(0);
    EXPECT_NEAR_EQ(Scalar(5), row[0]);
    EXPECT_NEAR_EQ(Scalar(-4), row[1]);
    EXPECT_NEAR_EQ(Scalar(1), row[2]);
    row = D.Values(1);
    EXPECT_NEAR_EQ(Scalar(-4), row[0]);
    EXPECT_NEAR_EQ(Scalar(6), row[1]);
    EXPECT_NEAR_EQ(Scalar(-4), row[2]);
    row = D.Values(2);
    EXPECT_NEAR_EQ(Scalar(1), row[0]);
    EXPECT_NEAR_EQ(Scalar(-4), row[1]);
    EXPECT_NEAR_EQ(Scalar(5), row[2]);
  }
  {
    using Scalar = real;
    int numRows = 3;
    DynamicArray<int> rowPtr({0, 2, 5, 7});
    DynamicArray<int> colIdx({0, 1, 0, 1, 2, 1, 2});
    auto const two = Scalar(2);
    auto const minusOne = Scalar(-1);
    DynamicArray<Scalar> values({two, minusOne, minusOne, two, minusOne, minusOne, two});
    SparseMatrix<Scalar, int, int> C(numRows, rowPtr, colIdx, values);

    DynamicArray<int> rowPtr2({0, 1, 3, 6});
    DynamicArray<int> colIdx2({1, 1, 2, 0, 1, 2});
    DynamicArray<Scalar> values2(
        {Scalar(0.5), Scalar(0.5), Scalar(0.4), Scalar(0.3), Scalar(0.1), Scalar(0.2)});
    SparseMatrix<Scalar, int, int> D(numRows + 1, rowPtr2, colIdx2, values2);

    auto E = C * D;
    EXPECT_EQ(8, E.NumNonZeros());
    auto row = E.Values(0);
    auto col = E.Indices(0);
    EXPECT_EQ(2, row.size());
    EXPECT_EQ(2, col.size());
    EXPECT_NEAR_EQ(Scalar(0.5), row[0]);
    EXPECT_EQ(1, col[0]);
    EXPECT_NEAR_EQ(Scalar(-0.4), row[1]);
    EXPECT_EQ(2, col[1]);
    row = E.Values(1);
    col = E.Indices(1);
    EXPECT_EQ(3, row.size());
    EXPECT_EQ(3, col.size());
    EXPECT_NEAR_EQ(Scalar(-0.3), row[0]);
    EXPECT_EQ(0, col[0]);
    EXPECT_NEAR_EQ(Scalar(0.4), row[1]);
    EXPECT_EQ(1, col[1]);
    EXPECT_NEAR_EQ(Scalar(0.6), row[2]);
    EXPECT_EQ(2, col[2]);
    row = E.Values(2);
    col = E.Indices(2);
    EXPECT_EQ(3, row.size());
    EXPECT_EQ(3, col.size());
    EXPECT_NEAR_EQ(Scalar(0.6), row[0]);
    EXPECT_EQ(0, col[0]);
    EXPECT_NEAR_EQ(Scalar(-0.3), row[1]);
    EXPECT_EQ(1, col[1]);
    //--- Entry (2, 2) is zero because of the numerical values
    //--- But the sparsity indicates a non-zero potential
    EXPECT_NEAR_EQ(Scalar(0), row[2]);
    EXPECT_EQ(2, col[2]);
  }
}

TEST(SparseMatrix, SparseMatProduct) {
  using Scalar = real;
  {
    int numRows = 3, numCols = numRows;
    DynamicArray<int> rowPtr({0, 2, 5, 7});
    DynamicArray<int> colIdx({0, 1, 0, 1, 2, 1, 2});
    auto const two = Scalar(2);
    auto const minusOne = Scalar(-1);
    DynamicArray<Scalar> values({two, minusOne, minusOne, two, minusOne, minusOne, two});
    SparseMatrix<Scalar, int, int> C(numCols, rowPtr, colIdx, values);
    DynamicArray<int> dRowPtr({0, 3, 6, 9});
    DynamicArray<int> dColIdx({0, 1, 2, 0, 1, 2, 0, 1, 2});
    DynamicArray<Scalar> dValues(9, Scalar(0));
    SparseMatrix<Scalar, int, int> D(numCols, dRowPtr, dColIdx, dValues);
    mochi::details::SparseMatProduct(C, C, D);
    auto row = D.Values(0);
    EXPECT_NEAR_EQ(Scalar(5), row[0]);
    EXPECT_NEAR_EQ(Scalar(-4), row[1]);
    EXPECT_NEAR_EQ(Scalar(1), row[2]);
    row = D.Values(1);
    EXPECT_NEAR_EQ(Scalar(-4), row[0]);
    EXPECT_NEAR_EQ(Scalar(6), row[1]);
    EXPECT_NEAR_EQ(Scalar(-4), row[2]);
    row = D.Values(2);
    EXPECT_NEAR_EQ(Scalar(1), row[0]);
    EXPECT_NEAR_EQ(Scalar(-4), row[1]);
    EXPECT_NEAR_EQ(Scalar(5), row[2]);
  }
  {
    int numRows = 3;
    DynamicArray<int> rowPtr({0, 2, 5, 7});
    DynamicArray<int> colIdx({0, 1, 0, 1, 2, 1, 2});
    auto const two = Scalar(2);
    auto const minusOne = Scalar(-1);
    DynamicArray<Scalar> values({two, minusOne, minusOne, two, minusOne, minusOne, two});
    SparseMatrix<Scalar, int, int> C(numRows, rowPtr, colIdx, values);
    //
    DynamicArray<int> rowPtr2({0, 1, 3, 6});
    DynamicArray<int> colIdx2({1, 1, 2, 0, 1, 2});
    DynamicArray<Scalar> values2(
        {Scalar(0.5), Scalar(0.5), Scalar(0.4), Scalar(0.3), Scalar(0.1), Scalar(0.2)});
    SparseMatrix<Scalar, int, int> D(numRows + 1, rowPtr2, colIdx2, values2);
    //
    DynamicArray<int> eRowPtr({0, 2, 5, 8});
    DynamicArray<int> eColIdx({1, 2, 0, 1, 2, 0, 1, 2});
    DynamicArray<Scalar> eValues(8, 0.0);
    SparseMatrix<Scalar, int, int> E(D.Cols(), eRowPtr, eColIdx, eValues);
    details::SparseMatProduct(C, D, E);
    EXPECT_EQ(8, E.NumNonZeros());
    auto row = E.Values(0);
    auto col = E.Indices(0);
    EXPECT_NEAR_EQ(Scalar(0.5), row[0]);
    EXPECT_NEAR_EQ(Scalar(-0.4), row[1]);
    row = E.Values(1);
    col = E.Indices(1);
    EXPECT_NEAR_EQ(Scalar(-0.3), row[0]);
    EXPECT_NEAR_EQ(Scalar(0.4), row[1]);
    EXPECT_NEAR_EQ(Scalar(0.6), row[2]);
    row = E.Values(2);
    col = E.Indices(2);
    EXPECT_NEAR_EQ(Scalar(0.6), row[0]);
    EXPECT_NEAR_EQ(Scalar(-0.3), row[1]);
    //--- Entry (2, 2) is zero because of the numerical values
    //--- But the sparsity indicates a non-zero potential
    EXPECT_NEAR_EQ(Scalar(0), row[2]);
  }
  //--- Larger matrices to test all codepaths.
  {
    for (auto m : {1, 2, 3, 4, 6, 8, 9, 10}) {
      for (auto k : {1, 2, 4, 5, 16, 17, 18, 19, 20, 21, 22, 23}) {
        for (auto n : {1, 2, 4, 5, 10}) {
          Matrix<Scalar> A(m, k);
          Matrix<Scalar> B(k, n);
          A.SetRandom(1);
          B.SetRandom(2);
          SparseMatrix<Scalar> AB =
              ToSparseMatrix(A, /* pruneZeros */ false) * ToSparseMatrix(B, /* pruneZeros */ false);
          Matrix<Scalar> ABdense = A * B;
          Scalar const absTol = 2 * k * std::numeric_limits<Scalar>::epsilon();
          EXPECT_TRUE(test::NearEqualMatrices(ABdense, AB, absTol));
        }
      }
    }
  }
}

TEST(SparseMatrix, Empty) {
  // Default constructed
  {
    SparseMatrix<real> mat;
    EXPECT_TRUE(mat.empty());
    EXPECT_FALSE(mat); // operator bool
    EXPECT_EQ(0, mat.NumNonZeros());
  }

  // Zero rows
  {
    SparseMatrix<real> mat(3, {}, {}, {});
    EXPECT_TRUE(mat.empty());
    EXPECT_FALSE(mat); // operator bool
    EXPECT_EQ(0, mat.NumNonZeros());
  }

  // Zero non-zeros
  {
    SparseMatrix<real> mat(3, {0, 0, 0, 0}, {}, {});
    EXPECT_FALSE(mat.empty());
    EXPECT_TRUE(mat); // operator bool
    EXPECT_EQ(0, mat.NumNonZeros());
  }

  // Non-Zero non-zeros
  {
    SparseMatrix<real> mat(3, {0, 1, 2, 3}, {0, 1, 2}, {0_r, 1_r, 2_r});
    EXPECT_FALSE(mat.empty());
    EXPECT_TRUE(mat); // operator bool
    EXPECT_EQ(3, mat.NumNonZeros());
  }
}

TEST(SparseMatrix, Reset) {
  // Reset an owning SparseMatrix
  {
    // Reset from default
    SparseMatrix<real> mat;
    EXPECT_EQ(0, mat.Rows());
    mat.Reset(MakeMat<TestCase_3x3>());
    CheckMat<TestCase_3x3>(mat);

    // Change dimensions and values
    mat.Reset(MakeMat<TestCase_1x3>());
    CheckMat<TestCase_1x3>(mat);

    // Change dimensions and values using a view
    mat.Reset(MakeMatConstView<TestCase_3x1>());
    CheckMat<TestCase_3x1>(mat);

    // Change dimensions using a different constructor
    DynamicArray<int> pointers = {0, 0, 0, 0};
    mat.Reset(3, pointers, DynamicArray<int>{}, DynamicArray<real>{});
    EXPECT_EQ(3, mat.Rows());
    EXPECT_EQ(3, mat.Cols());
    EXPECT_EQ(0, isize(mat.Values()));
    EXPECT_EQ(0, isize(mat.Indices()));
  }

  // Reset a SparseMatrixView
  {
    SparseMatrixView<real const> view;
    EXPECT_EQ(0, view.Rows());

    // Reset from owning matrix
    auto mat = MakeMat<TestCase_3x3>();
    view.Reset(mat);
    EXPECT_EQ(mat.Values().data(), view.Values().data()); // same address
    CheckMat<TestCase_3x3>(view);

    // Reset from const view
    mat = MakeMat<TestCase_3x1>();
    view.Reset(AsConstView(mat));
    EXPECT_EQ(mat.Values().data(), view.Values().data()); // same address
    CheckMat<TestCase_3x1>(view);

    // Reset using a different constructor
    mat = MakeMat<TestCase_3x3>();
    view.Reset(3, mat.Pointers(), mat.Indices(), mat.Values());
    EXPECT_EQ(mat.Values().data(), view.Values().data()); // same address
    CheckMat<TestCase_3x3>(view);
  }
}

TEST(SparseMatrix, Duplicate) {
  using TC = TestCase_3x3;
  SparseMatrix<real> mat = MakeMat<TC>();
  SparseMatrix<real> mat2 = mat.Duplicate();
  SparseMatrix<real> mat3 = AsConstView(mat).Duplicate();
  CheckMat<TC>(mat);
  CheckMat<TC>(mat2);
  CheckMat<TC>(mat3);
  EXPECT_NE(mat.Values().data(), mat2.Values().data()); // Not the same address
  EXPECT_NE(mat.Values().data(), mat3.Values().data()); // Not the same address
  EXPECT_NE(mat2.Values().data(), mat3.Values().data()); // Not the same address
}

TEST(SparseMatrix, Transpose) {
  // Create matrix for testing. Use a rectangular matrix with some empty rows and columns to
  // increase test coverage.
  int const numRows = 33;
  int const numCols = numRows + 3;
  DynamicArray<int> rowPtr;
  DynamicArray<int> colIdx;
  rowPtr.reserve(numRows + 1);
  colIdx.reserve(numRows * (numRows + 1) / 2);
  rowPtr.push_back(0);
  for (int ii = 0; ii < numRows; ++ii) {
    int const numNonZerosInRow = ii; // Different in each row.
    for (int jj = 0; jj < numNonZerosInRow; ++jj) {
      colIdx.push_back(jj);
    }
    rowPtr.push_back(rowPtr.back() + numNonZerosInRow);
  }

  int const numNonZeros = rowPtr.back();
  DynamicArray<real> values;
  values.reserve(numNonZeros);
  for (int ii = 0; ii < numNonZeros; ++ii) {
    values.push_back(real(2 * ii) / numNonZeros - 1_r); // Different in each entry.
  }

  SparseMatrix<real> A(numCols, rowPtr, colIdx, values);
  auto Av = AsView(A);
  auto Acv = AsConstView(A);
  auto AT0 = Transpose(A);
  auto AT1 = Transpose(Av);
  auto AT2 = Transpose(Acv);

  static_assert(std::is_same_v<SparseMatrix<real>, decltype(AT0)>);
  static_assert(std::is_same_v<SparseMatrix<real>, decltype(AT1)>);
  static_assert(std::is_same_v<SparseMatrix<real>, decltype(AT2)>);

  EXPECT_EQ(A.Cols(), AT0.Rows());
  EXPECT_EQ(A.Rows(), AT0.Cols());
  for (int r = 0; r < AT0.Rows(); ++r) {
    for (int c = 0; c < AT0.Cols(); ++c) {
      EXPECT_EQ(A(c, r), AT0(r, c));
    }
  }

  real const absTol = 0_r;
  EXPECT_TRUE(test::NearEqualMatrices(AT0, AT1, absTol));
  EXPECT_TRUE(test::NearEqualMatrices(AT0, AT2, absTol));
}

TEST(SparseMatrix, IsRowEmpty) {
  int const numCols = 6;
  DynamicArray<int> rowPtr = {0, 2, 2, 4, 5, 5};
  DynamicArray<int> colIdx = {0, 1, 0, 5, 3};
  DynamicArray<real> values = {1_r, 2_r, 3_r, 4_r, 5_r};
  SparseMatrix<real> A(numCols, rowPtr, colIdx, values);

  EXPECT_FALSE(A.IsRowEmpty(0));
  EXPECT_TRUE(A.IsRowEmpty(1));
  EXPECT_FALSE(A.IsRowEmpty(2));
  EXPECT_FALSE(A.IsRowEmpty(3));
  EXPECT_TRUE(A.IsRowEmpty(4));

  EXPECT_FALSE(AsView(A).IsRowEmpty(0));
  EXPECT_TRUE(AsView(A).IsRowEmpty(1));
  EXPECT_FALSE(AsView(A).IsRowEmpty(2));
  EXPECT_FALSE(AsConstView(A).IsRowEmpty(3));
  EXPECT_TRUE(AsConstView(A).IsRowEmpty(4));
}

TEST(SparseMatrix, Norm) {
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();
  real expectedNormSqr = 0_r;
  for (int r = 0; r < mat.Rows(); ++r) {
    for (int c = 0; c < mat.Cols(); ++c) {
      expectedNormSqr += mat(r, c) * mat(r, c);
    }
  }
  EXPECT_NEAR_EQ(expectedNormSqr, mat.NormSqr());
  EXPECT_NEAR_EQ(Sqrt(expectedNormSqr), mat.Norm());
}

TEST(SparseMatrix, MaxAbsDifference) {
  using mochi::test::MaxAbsDifference;
  int constexpr kTestSizes[] = {1, 2, 3, 4, 7, 8, 9, 31, 32, 33};
  for (int numRows : kTestSizes) {
    for (int numCols : kTestSizes) {
      // Start with dense data. Use std::vector for this test because DynamicArray does not support
      // arbitrary insert and erase from the middle.
      std::vector<int> pointers(numRows + 1);
      std::vector<int> indices(numRows * numCols);
      std::vector<real> values(numRows * numCols);
      pointers[0] = 0;
      for (int r = 0; r < numRows; ++r) {
        pointers[r + 1] = pointers[r] + numCols;
        for (int c = 0; c < numCols; ++c) {
          indices[r * numCols + c] = c;
          values[r * numCols + c] = static_cast<real>(r * numCols + c) /
              static_cast<real>(numRows * numCols); // arbitrary values between 0 and 1
        }
      }

      // Zero difference
      SparseMatrixView<real const> matA(numCols, pointers, indices, values);
      SparseMatrix<real> matB = matA; // copy
      EXPECT_EQ(0_r, MaxAbsDifference(matA, matB));

      // Modify a single value on each row of matA
      for (int r = 0; r < numRows; ++r) {
        int c = (r % numCols);
        auto& val = values[r * numCols + c]; // value on row r of matA
        auto prev = val;
        val = prev + 0.1_r;
        EXPECT_NEAR_EQ(0.1_r, MaxAbsDifference(matA, matB));
        EXPECT_NEAR_EQ(0.1_r, MaxAbsDifference(matB, matA));
        val = prev - 0.2_r;
        EXPECT_NEAR_EQ(0.2_r, MaxAbsDifference(matA, matB));
        EXPECT_NEAR_EQ(0.2_r, MaxAbsDifference(matB, matA));
        val = prev; // revert
        EXPECT_EQ(0_r, MaxAbsDifference(matA, matB));
      }

      // Remove a single value from the sparsity of matA
      for (int r = 0; r < numRows; ++r) {
        // Remove matA(r, c) from the sparsity pattern
        int c = (r % numCols);
        int prev = values[pointers[r] + c];
        indices.erase(indices.begin() + pointers[r] + c);
        values.erase(values.begin() + pointers[r] + c);
        for (int r2 = r + 1; r2 <= numRows; ++r2) {
          pointers[r2]--;
        }
        matA.Reset(numCols, pointers, indices, values);
        // The non-zero value at matB(r, c) is the max difference
        EXPECT_NEAR_EQ(matB(r, c), MaxAbsDifference(matA, matB));
        // But if we set that vale to zero, then MaxAbsDifference should once again return zero,
        // despite the difference in sparsity patterns
        matB.Values(r)[c] = 0_r;
        EXPECT_EQ(0_r, MaxAbsDifference(matA, matB));
        // Revert
        indices.insert(indices.begin() + pointers[r] + c, c);
        values.insert(values.begin() + pointers[r] + c, prev);
        for (int r2 = r + 1; r2 <= numRows; ++r2) {
          pointers[r2]++;
        }
        matA.Reset(numCols, pointers, indices, values);
        matB.Values(r)[c] = prev;
        EXPECT_EQ(0_r, MaxAbsDifference(matA, matB));
      }
    }
  }
}

TEST(SparseMatrix, SetZero) {
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();
  EXPECT_NE(0_r, mat(0, 0));
  mat.SetZero();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_EQ(0_r, mat(r, c));
    }
  }
}

TEST(SparseMatrix, SetConstant) {
  // |1 2|
  // | 3 |
  // |4 5|
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();
  mat.SetConstant(9_r);
  EXPECT_EQ(9_r, mat(0, 0));
  EXPECT_EQ(0_r, mat(0, 1));
  EXPECT_EQ(9_r, mat(0, 2));
  EXPECT_EQ(0_r, mat(1, 0));
  EXPECT_EQ(9_r, mat(1, 1));
  EXPECT_EQ(0_r, mat(1, 2));
  EXPECT_EQ(9_r, mat(2, 0));
  EXPECT_EQ(0_r, mat(2, 1));
  EXPECT_EQ(9_r, mat(2, 2));
}

TEST(SparseMatrix, SetValue) {
  // |1 2|
  // | 3 |
  // |4 5|
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();

  EXPECT_EQ(1_r, mat(0, 0));
  mat.SetValue(0, 0, 8_r);
  EXPECT_EQ(8_r, mat(0, 0));

  EXPECT_EQ(2_r, mat(0, 2));
  mat.SetValue(0, 2, 9_r);
  EXPECT_EQ(9_r, mat(0, 2));

  EXPECT_EQ(3_r, mat(1, 1));
  mat.SetValue(1, 1, 10_r);
  EXPECT_EQ(10_r, mat(1, 1));

  EXPECT_EQ(4_r, mat(2, 0));
  mat.SetValue(2, 0, 11_r);
  EXPECT_EQ(11_r, mat(2, 0));

  EXPECT_EQ(5_r, mat(2, 2));
  mat.SetValue(2, 2, 12_r);
  EXPECT_EQ(12_r, mat(2, 2));
}

TEST(SparseMatrix, AddValue) {
  // |1 2|
  // | 3 |
  // |4 5|
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();

  EXPECT_EQ(1_r, mat(0, 0));
  mat.AddValue(0, 0, 8_r);
  EXPECT_NEAR_EQ(9_r, mat(0, 0));

  EXPECT_EQ(2_r, mat(0, 2));
  mat.AddValue(0, 2, 9_r);
  EXPECT_NEAR_EQ(11_r, mat(0, 2));

  EXPECT_EQ(3_r, mat(1, 1));
  mat.AddValue(1, 1, 10_r);
  EXPECT_NEAR_EQ(13_r, mat(1, 1));

  EXPECT_EQ(4_r, mat(2, 0));
  mat.AddValue(2, 0, 11_r);
  EXPECT_NEAR_EQ(15_r, mat(2, 0));

  EXPECT_EQ(5_r, mat(2, 2));
  mat.AddValue(2, 2, 12_r);
  EXPECT_NEAR_EQ(17_r, mat(2, 2));
}

TEST(SparseMatrix, PlusEquals) {
  // |1 2|
  // | 3 |
  // |4 5|
  using TC = TestCase_3x3;

  // SparseMatrix += self
  {
    auto a = MakeMat<TC>();
    a += a;
    for (int r = 0; r < a.Rows(); ++r) {
      for (int c = 0; c < a.Cols(); ++c) {
        EXPECT_NEAR_EQ(TC::kValuesDense[r][c] * 2_r, a(r, c));
      }
    }
  }

  // SparseMatrix += SparseMatrixView with same sparsity
  {
    auto a = MakeMat<TC>();
    auto const b = MakeMat<TC>();
    a += AsConstView(b);
    for (int r = 0; r < a.Rows(); ++r) {
      for (int c = 0; c < a.Cols(); ++c) {
        EXPECT_NEAR_EQ(TC::kValuesDense[r][c] * 2_r, a(r, c));
      }
    }
  }

  // SparseMatrix += SparseMatrix with no non-zero values
  {
    auto a = MakeMat<TC>();
    auto const b = SparseMatrix<real>(3, {0, 0, 0, 0}, {}, {});
    CheckMat<TC>(a);
    a += b;
  }

  // SparseMatrix += SparseMatrix with same dimensions, but fewer non-zero values
  {
    // |1 2|    |  1|
    // | 3 | += |   |
    // |4 5|    |2 3|
    auto a = MakeMat<TC>();
    auto const b = SparseMatrix<real>(
        3,
        {0, 1, 1, 3}, // pointers
        {2, 0, 2}, // indices
        {1_r, 2_r, 3_r}); // values
    a += b;
    RowMatrix<real> expected = {{1, 0, 3}, {0, 3, 0}, {6, 0, 8}};
    EXPECT_TRUE(test::NearEqualMatrices(a, expected));
  }

  // SparseMatrix += SparseMatrix with fewer rows.
  {
    // |1 2|    |  1|
    // | 3 | +=
    // |4 5|
    auto a = MakeMat<TC>();
    auto const b = SparseMatrix<real>(
        2,
        {0, 1}, // pointers
        {2}, // indices
        {1_r}); // values
    a += b;
    RowMatrix<real> expected = {{1, 0, 3}, {0, 3, 0}, {4, 0, 5}};
    EXPECT_TRUE(test::NearEqualMatrices(a, expected));
  }

  // SparseMatrix += SparseMatrix with fewer columns.
  {
    // |1 2|    | |
    // | 3 | += | |
    // |4 5|    |1|
    auto a = MakeMat<TC>();
    auto const b = SparseMatrix<real>(
        1,
        {0, 0, 0, 1}, // pointers
        {0}, // indices
        {1_r}); // values
    a += b;
    RowMatrix<real> expected = {{1, 0, 2}, {0, 3, 0}, {5, 0, 5}};
    EXPECT_TRUE(test::NearEqualMatrices(a, expected));
  }
}

TEST(SparseMatrix, IndicesAndValuesPerRow) {
  using TC = TestCase_3x3;
  auto mat = MakeMat<TC>();
  CheckPerRowAccessors<TC>(mat);
  CheckPerRowAccessors<TC>(AsView(mat));
  CheckPerRowAccessors<TC>(AsConstView(mat));

  // Mutability: write through mutable view's Values(r)
  AsView(mat).Values(1)[0] = 99_r;
  EXPECT_EQ(99_r, mat(1, 1));

  // Empty row (TestCase_3x1: row 1 has no entries)
  auto mat2 = MakeMat<TestCase_3x1>();
  EXPECT_TRUE(mat2.Indices(1).empty());
  EXPECT_TRUE(mat2.Values(1).empty());
}
