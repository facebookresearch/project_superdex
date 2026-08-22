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

#include <mochi_core/linear_algebra/krylov/relaxed_ilu_prec.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <limits>

using namespace mochi;
using namespace mochi::krylov;

// Use real instead of float or double to reduce build time. Both are checked by CI.
using Scalar = real;

template <typename MatrixType, typename Input, typename Output>
static void Example1(mochi::krylov::RelaxedILUPrec<MatrixType> const& P, Input& X, Output& PX) {
  //
  // Input matrix
  // A = [   3   0  -1  -1   0  -1
  //         0   2   0  -1   0   0
  //        -1   0   3   0  -1   0
  //        -1  -1   0   2   0  -1
  //         0   0  -1   0   3  -1
  //        -1   0   0   0   0   4 ]
  //
  // Compute ILU(0) preconditioner with octave
  //
  Matrix<Scalar, 6, 6> L;
  L.SetIdentity();
  L(2, 0) = -Scalar(1.0) / Scalar(3.0);
  L(3, 0) = L(2, 0);
  L(5, 0) = L(2, 0);
  L(3, 1) = -Scalar(1.0) / Scalar(2.0);
  L(4, 2) = -Scalar(3.0) / Scalar(8.0);
  Matrix<Scalar, 6, 6> U;
  U(0, 0) = Scalar(3.0);
  U(0, 2) = -Scalar(1.0);
  U(0, 3) = U(0, 2);
  U(0, 5) = U(0, 2);
  U(1, 1) = Scalar(2.0);
  U(1, 3) = -Scalar(1.0);
  U(2, 2) = Scalar(8.0) / Scalar(3.0);
  U(2, 4) = -Scalar(1.0);
  U(3, 3) = Scalar(7.0) / Scalar(6.0);
  U(3, 5) = -Scalar(4.0) / Scalar(3.0);
  U(4, 4) = Scalar(21.0) / Scalar(8.0);
  U(4, 5) = -Scalar(1.0);
  U(5, 5) = Scalar(11.0) / Scalar(3.0);
  auto TU = mochi::krylov::UpperTriangularView(U);
  //
  X.SetRandom(123);
  PX.SetRandom(456);
  //
  P(X, PX);
  //
  Matrix<Scalar> TPX(X.Rows(), X.Cols());
  TPX = X;
  //
  for (int ii = 1; ii < X.Rows(); ++ii) {
    TPX.Row(ii) -= L.Block(ii, 0, 1, ii) * TPX.TopRows(ii);
  }
  //
  TU.SolveInPlace(TPX);
  //
  auto tol = 2 * X.Rows() * std::numeric_limits<Scalar>::epsilon();
  //
  for (int i = 0; i < X.Rows(); ++i) {
    for (int j = 0; j < X.Cols(); ++j) {
      EXPECT_NEAR_RTOL(PX(i, j), TPX(i, j), tol);
    }
  }
}

// This example tests whether P is an exact factorization of A
template <mochi::krylov::Direction kDir, typename Scalar, typename MatrixType>
static void Example2(mochi::krylov::RelaxedILUPrec<MatrixType> const& P, MatrixType const& A) {
  auto tol = Scalar(4 * A.Rows()) * std::numeric_limits<Scalar>::epsilon();
  Matrix<Scalar, mochi::krylov::kDynamic, mochi::krylov::kDynamic, kDir> X(
      A.Rows(), Max(1, A.Cols() / 2));
  X.SetRandom(123);
  Matrix<Scalar, mochi::krylov::kDynamic, mochi::krylov::kDynamic, kDir> Y(X), PY(X);
  Y = A * X;
  P(Y, PY);
  //--- PY should match X
  PY -= X;
  EXPECT_NEAR_RTOL(PY.Norm() / X.Norm(), Scalar(0), tol);
}

// This example tests whether P(A*e) = e where e = [1, 1, ..., 1]^T
// This check can be used for modified ILU(0)
template <typename Scalar, typename MatrixType>
static void Example3(mochi::krylov::RelaxedILUPrec<MatrixType> const& P, MatrixType const& A) {
  auto tol = Scalar(4 * A.Rows()) * std::numeric_limits<Scalar>::epsilon();
  ColumnVector<Scalar> e(A.Rows()), Ae(e);
  e.SetConstant(Scalar(1));
  Ae = A * e;
  ColumnVector<Scalar> PAe(A.Rows());
  P(Ae, PAe);
  PAe -= e;
  EXPECT_NEAR_RTOL(PAe.Norm(), Scalar(0), tol);
}

template <typename MatrixType, typename Input, typename Output>
static void Example4(mochi::krylov::RelaxedILUPrec<MatrixType> const& P, Input& X, Output& PX) {
  //
  // Input matrix
  // A = [   3   0  -1  -1   0  -1
  //         0   2   0  -1   0   0
  //        -1   0   3   0  -1   0
  //        -1  -1   0   2   0  -1
  //         0   0  -1   0   3  -1
  //        -1   0   0   0   0   4 ]
  //
  // Compute modified ILU(0) preconditioner with octave
  //
  Matrix<Scalar, 6, 6> L;
  L.SetIdentity();
  L(2, 0) = -Scalar(1.0) / Scalar(3.0);
  L(3, 0) = L(2, 0);
  L(5, 0) = L(2, 0);
  L(3, 1) = -Scalar(1.0) / Scalar(2.0);
  L(4, 2) = -Scalar(1.0) / Scalar(2.0);
  Matrix<Scalar, 6, 6> U;
  U(0, 0) = Scalar(3.0);
  U(0, 2) = -Scalar(1.0);
  U(0, 3) = U(0, 2);
  U(0, 5) = U(0, 2);
  U(1, 1) = Scalar(2.0);
  U(1, 3) = -Scalar(1.0);
  U(2, 2) = Scalar(2.0);
  U(2, 4) = -Scalar(1.0);
  U(3, 3) = Scalar(5.0) / Scalar(6.0);
  U(3, 5) = -Scalar(4.0) / Scalar(3.0);
  U(4, 4) = Scalar(2.5);
  U(4, 5) = -Scalar(1.0);
  U(5, 5) = Scalar(3.0);
  auto TU = mochi::krylov::UpperTriangularView(U);
  //
  X.SetRandom(123);
  PX.SetRandom(456);
  //
  P(X, PX);
  //
  Matrix<Scalar> TPX(X.Rows(), X.Cols());
  TPX = X;
  //
  for (int ii = 1; ii < X.Rows(); ++ii) {
    TPX.Row(ii) -= L.Block(ii, 0, 1, ii) * TPX.TopRows(ii);
  }
  //
  TU.SolveInPlace(TPX);
  //
  auto tol = 2 * X.Rows() * std::numeric_limits<Scalar>::epsilon();
  //
  for (int i = 0; i < X.Rows(); ++i) {
    for (int j = 0; j < X.Cols(); ++j) {
      EXPECT_NEAR_RTOL(PX(i, j), TPX(i, j), tol);
    }
  }
}

TEST(RelaxedILUPrec, Example1) {
  //
  // Input matrix
  // A = [   3   0  -1  -1   0  -1
  //         0   2   0  -1   0   0
  //        -1   0   3   0  -1   0
  //        -1  -1   0   2   0  -1
  //         0   0  -1   0   3  -1
  //        -1   0   0   0   0   4 ]
  //
  // Test ILU(0) preconditioner
  //
  DynamicArray<int> rowPtr({0, 4, 6, 9, 13, 16, 18});
  DynamicArray<int> colIdx({0, 2, 3, 5, 1, 3, 0, 2, 4, 0, 1, 3, 5, 2, 4, 5, 0, 5});
  DynamicArray<Scalar> values({3, -1, -1, -1, 2, -1, -1, 3, -1, -1, -1, 2, -1, -1, 3, -1, -1, 4});
  SparseMatrix<Scalar> SpA(6, rowPtr, colIdx, values);
  {
    mochi::krylov::RelaxedILUPrec<SparseMatrix<Scalar>> P(
        SpA, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar{0});
    {
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> X, PX;
      Example1(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> PX2;
      Example1(P, X, PX2);
    }
    {
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> X, PX;
      Example1(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> PX2;
      Example1(P, X, PX2);
    }
  }
  {
    auto bSpA = ToBlockSparseMatrix<1>(SpA);
    mochi::krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, 1>> P(
        bSpA, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar{0});
    {
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> X, PX;
      Example1(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> PX2;
      Example1(P, X, PX2);
    }
    {
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> X, PX;
      Example1(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> PX2;
      Example1(P, X, PX2);
    }
  }
  {
    auto dA = ToMatrix(SpA);
    mochi::krylov::RelaxedILUPrec<Matrix<Scalar>> P(
        dA, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar{0});
    {
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> X, PX;
      Example1(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> PX2;
      Example1(P, X, PX2);
    }
    {
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> X, PX;
      Example1(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> PX2;
      Example1(P, X, PX2);
    }
  }
}

TEST(RelaxedILUPrec, Example4) {
  //
  // Input matrix
  // A = [   3   0  -1  -1   0  -1
  //         0   2   0  -1   0   0
  //        -1   0   3   0  -1   0
  //        -1  -1   0   2   0  -1
  //         0   0  -1   0   3  -1
  //        -1   0   0   0   0   4 ]
  //
  // Test modified-ILU(0) preconditioner
  //
  DynamicArray<int> rowPtr({0, 4, 6, 9, 13, 16, 18});
  DynamicArray<int> colIdx({0, 2, 3, 5, 1, 3, 0, 2, 4, 0, 1, 3, 5, 2, 4, 5, 0, 5});
  DynamicArray<Scalar> values({3, -1, -1, -1, 2, -1, -1, 3, -1, -1, -1, 2, -1, -1, 3, -1, -1, 4});
  SparseMatrix<Scalar> SpA(6, rowPtr, colIdx, values);
  {
    mochi::krylov::RelaxedILUPrec<SparseMatrix<Scalar>> P(
        SpA, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar(1));
    Example3<Scalar>(P, SpA);
    {
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> X, PX;
      Example4(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> PX2;
      Example4(P, X, PX2);
    }
    {
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> X, PX;
      Example4(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> PX2;
      Example4(P, X, PX2);
    }
  }
  {
    auto bSpA = ToBlockSparseMatrix<1>(SpA);
    mochi::krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, 1>> P(bSpA, 0, Scalar(1));
    Example3<Scalar>(P, bSpA);
    {
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> X, PX;
      Example4(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> PX2;
      Example4(P, X, PX2);
    }
    {
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> X, PX;
      Example4(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> PX2;
      Example4(P, X, PX2);
    }
  }
  {
    auto dA = ToMatrix(SpA);
    mochi::krylov::RelaxedILUPrec<Matrix<Scalar>> P(dA, 0, Scalar(1));
    Example3<Scalar>(P, dA);
    {
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> X, PX;
      Example4(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> PX2;
      Example4(P, X, PX2);
    }
    {
      Matrix<Scalar, 6, 4, krylov::Direction::RowMajor> X, PX;
      Example4(P, X, PX);
      Matrix<Scalar, 6, 4, krylov::Direction::ColMajor> PX2;
      Example4(P, X, PX2);
    }
  }
}

TEST(RelaxedILUPrec, BlockSparseMatrix) {
  {
    DynamicArray<int> rowPtr({0, 3, 6, 8});
    DynamicArray<int> colIdx({0, 1, 2, 0, 1, 2, 0, 2});
    DynamicArray<Scalar> values({3,  0, -1, -1, 0, -1, 0, 2,  0, -1, 0, 0,  -1, 0, 3,  0,
                                 -1, 0, -1, -1, 0, 2,  0, -1, 0, 0,  3, -1, -1, 0, -1, 4});
    BlockSparseMatrix<Scalar, 2> A(3, rowPtr, colIdx, values);
    //
    // L(2, 0) = L(3, 0) = L(5, 0) = -1/3 and L(3, 1) = -0.5
    // the rest of the entries of L should match the identity entries
    //
    // U should match L^{-1} * A except for the block ([2, 3], [4, 5]) which is set to 0
    //
    {
      mochi::krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, 2>> P(
          A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar{0});
      auto tol = Scalar(2 * A.Rows()) * std::numeric_limits<Scalar>::epsilon();
      {
        Matrix<Scalar, 6, 1, krylov::Direction::ColMajor> x, px;
        x(0, 0) = Scalar(0);
        x(1, 0) = Scalar(1);
        x(2, 0) = Scalar(1);
        x(3, 0) = Scalar(-1);
        x(4, 0) = Scalar(2);
        x(5, 0) = Scalar(8.0 / 3.0);
        P(x, px);
        for (int ii = 0; ii < x.Rows(); ++ii) {
          EXPECT_NEAR_RTOL(px(ii, 0), Scalar(1.0), tol);
        }
      }
      {
        Matrix<Scalar, 6, 1, krylov::Direction::RowMajor> x, px;
        x(0, 0) = Scalar(0);
        x(1, 0) = Scalar(1);
        x(2, 0) = Scalar(1);
        x(3, 0) = Scalar(-1);
        x(4, 0) = Scalar(2);
        x(5, 0) = Scalar(8.0 / 3.0);
        P(x, px);
        for (int ii = 0; ii < x.Rows(); ++ii) {
          EXPECT_NEAR_RTOL(px(ii, 0), Scalar(1.0), tol);
        }
      }
    }
    {
      mochi::krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, 2>> P(A, 0, Scalar(1));
      Example3<Scalar>(P, A);
    }
  }
  {
    DynamicArray<int> rowPtr({0, 3, 6, 9});
    DynamicArray<int> colIdx({0, 1, 2, 0, 1, 2, 0, 1, 2});
    DynamicArray<Scalar> values({3,  0,  -1, -1, 0, -1, 0, 2, 0,  -1, 0, 0,  -1, 0, 3, 0,  -1, 0,
                                 -1, -1, 0,  2,  0, -1, 0, 0, -1, 0,  3, -1, -1, 0, 0, -1, -1, 4});
    BlockSparseMatrix<Scalar, 2> A(3, rowPtr, colIdx, values);
    //
    // This factorization should be exact
    // Any relaxation coefficient should work
    //
    mochi::krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, 2>> P(
        A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar(0.95));
    Example2<krylov::Direction::ColMajor, Scalar>(P, A);
    Example2<krylov::Direction::RowMajor, Scalar>(P, A);
  }
  {
    DynamicArray<int> rowPtr({0, 2, 4});
    DynamicArray<int> colIdx({0, 1, 0, 1});
    DynamicArray<Scalar> values({3,  0,  -1, -1, 0, -1, 0, 2, 0,  -1, 0, 0,  -1, 0, 3, 0,  -1, 0,
                                 -1, -1, 0,  2,  0, -1, 0, 0, -1, 0,  3, -1, -1, 0, 0, -1, -1, 4});
    BlockSparseMatrix<Scalar, 3> A(2, rowPtr, colIdx, values);
    //
    // This factorization should be exact
    // Any relaxation coefficient should work
    //
    mochi::krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, 3>> P(
        A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar(0.95));
    Example2<krylov::Direction::ColMajor, Scalar>(P, A);
    Example2<krylov::Direction::RowMajor, Scalar>(P, A);
  }
}

TEST(RelaxedILUPrec, Matrix) {
  {
    for (auto n : {2, 3, 4, 5, 6, 7, 8, 9, 17, 31, 43}) {
      Matrix<Scalar> A(n, n);
      A.SetRandom(123, Scalar(-1.0), Scalar(1.0));
      //
      // Making the matrix diagonal dominant reduces the risk of small diagonal entries
      // during the incomplete factorization.
      //
      for (int i = 0; i < n; ++i) {
        A(i, i) += Scalar(4.3);
      }
      mochi::krylov::RelaxedILUPrec<Matrix<Scalar>> P(
          A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar(0.95));
      Example2<krylov::Direction::ColMajor, Scalar>(P, A);
      Example2<krylov::Direction::RowMajor, Scalar>(P, A);
    }
  }
}

TEST(RelaxedILUPrec, SparseMatrix) {
  {
    DynamicArray<int> rowPtr({0, 2, 5, 8, 11, 14, 17, 19});
    DynamicArray<int> colIdx({0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6});
    DynamicArray<Scalar> values(
        {2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 2});
    SparseMatrix<Scalar> A(7, rowPtr, colIdx, values);
    mochi::krylov::RelaxedILUPrec<SparseMatrix<Scalar>> P(
        A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar(0.95));
    //--- ILU(0) should be exact
    Example2<krylov::Direction::ColMajor, Scalar>(P, A);
    Example2<krylov::Direction::RowMajor, Scalar>(P, A);
  }
  {
    for (auto n : {2, 3, 4, 5, 6, 7, 8, 9, 17, 31, 43}) {
      Matrix<Scalar> dA(n, n);
      dA.SetRandom(789, Scalar(-1.0), Scalar(1.0));
      //
      // Making the matrix diagonal dominant reduces the risk of small diagonal entries
      // during the incomplete factorization.
      //
      for (int i = 0; i < n; ++i) {
        dA(i, i) += Scalar(5.1);
      }
      auto A = ToSparseMatrix(dA);
      mochi::krylov::RelaxedILUPrec<SparseMatrix<Scalar>> P(
          A, /*fillInLevel*/ 0, /*alphaRelax*/ Scalar(0.95));
      //--- ILU(0) factorization should be exact
      Example2<krylov::Direction::ColMajor, Scalar>(P, A);
      Example2<krylov::Direction::RowMajor, Scalar>(P, A);
    }
  }
}

TEST(RelaxedILUPrec, Update) {
  // Test that Update produces the same result as creating a new preconditioner.

  for (int level = 0; level <= 2; ++level) {
    // Test with SparseMatrix
    {
      DynamicArray<int> rowPtr({0, 4, 6, 9, 13, 16, 18});
      DynamicArray<int> colIdx({0, 2, 3, 5, 1, 3, 0, 2, 4, 0, 1, 3, 5, 2, 4, 5, 0, 5});
      DynamicArray<Scalar> values1(
          {3, -1, -1, -1, 2, -1, -1, 3, -1, -1, -1, 2, -1, -1, 3, -1, -1, 4});
      DynamicArray<Scalar> values2(
          {4, -1, -1, -1, 3, -1, -1, 4, -1, -1, -1, 3, -1, -1, 4, -1, -1, 5});
      SparseMatrix<Scalar> A1(6, rowPtr, colIdx, values1);
      SparseMatrix<Scalar> A2(6, DynamicArray<int>(rowPtr), DynamicArray<int>(colIdx), values2);

      krylov::RelaxedILUPrec<SparseMatrix<Scalar>> P1(A1, level, Scalar(0.5));
      krylov::RelaxedILUPrec<SparseMatrix<Scalar>> P2(A2, level, Scalar(0.5));

      // Update P1 with A2
      P1.Update(A2);

      // P1 after update should give the same result as P2
      ColumnVector<Scalar> x(6), Px1(6), Px2(6);
      x.SetRandom(123);
      P1(x, Px1);
      P2(x, Px2);

      auto tol = Scalar(8 * 6) * std::numeric_limits<Scalar>::epsilon();
      EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2, tol));
    }

    // Test with BlockSparseMatrix
    {
      DynamicArray<int> rowPtr({0, 2, 5, 8, 11, 14, 16});
      DynamicArray<int> ci({0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5});
      DynamicArray<Scalar> va1({2, -1, -1, 3, -1, -1, 4, -1, -1, 2, -1, -1, 2, -1, -1, 2});
      DynamicArray<Scalar> va2({3, -1, -1, 4, -1, -1, 5, -1, -1, 3, -1, -1, 3, -1, -1, 3});
      SparseMatrix<Scalar, int, int> SpA1(6, rowPtr, ci, va1);
      SparseMatrix<Scalar, int, int> SpA2(6, DynamicArray<int>(rowPtr), DynamicArray<int>(ci), va2);

      auto bSpA1 = ToBlockSparseMatrix<2>(SpA1);
      auto bSpA2 = ToBlockSparseMatrix<2>(SpA2);

      krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, 2>> P1(bSpA1, level, Scalar(0.5));
      krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, 2>> P2(bSpA2, level, Scalar(0.5));

      P1.Update(bSpA2);

      ColumnVector<Scalar> x(6), Px1(6), Px2(6);
      x.SetRandom(456);
      P1(x, Px1);
      P2(x, Px2);

      auto tol = Scalar(8 * 6) * std::numeric_limits<Scalar>::epsilon();
      EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2, tol));
    }

    // Test with dense Matrix
    if (level == 0) { // Only supported without fill-in
      Matrix<Scalar> A1(6, 6), A2(6, 6);
      A1.SetRandom(123, 1_r, 2_r);
      A2.SetRandom(456, 1_r, 2_r);
      // Make diagonally dominant
      for (int i = 0; i < 6; ++i) {
        A1(i, i) += Scalar(5);
        A2(i, i) += Scalar(5);
      }

      krylov::RelaxedILUPrec<Matrix<Scalar>> P1(A1, level, Scalar(0.5));
      krylov::RelaxedILUPrec<Matrix<Scalar>> P2(A2, level, Scalar(0.5));

      P1.Update(A2);

      ColumnVector<Scalar> x(6), Px1(6), Px2(6);
      x.SetRandom(789);
      P1(x, Px1);
      P2(x, Px2);

      auto tol = Scalar(8 * 6) * std::numeric_limits<Scalar>::epsilon();
      EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2, tol));
    }
  }
}

/// @brief Test ConcurrentSolve gives same result as Solve for ILU0.
/// @todo Re-enable when implementing ConcurrentSolve and extend to actual multi-threaded tests.
TEST(RelaxedILUPrec, DISABLED_ConcurrentSolve) {
  // Test with SparseMatrix
  {
    DynamicArray<int> rowPtr({0, 4, 6, 9, 13, 16, 18});
    DynamicArray<int> colIdx({0, 2, 3, 5, 1, 3, 0, 2, 4, 0, 1, 3, 5, 2, 4, 5, 0, 5});
    DynamicArray<Scalar> values({3, -1, -1, -1, 2, -1, -1, 3, -1, -1, -1, 2, -1, -1, 3, -1, -1, 4});
    SparseMatrix<Scalar> A(6, rowPtr, colIdx, values);

    krylov::RelaxedILUPrec<SparseMatrix<Scalar>> P(A, 0, Scalar(0));

    ColumnVector<Scalar> x(6), Px1(6), Px2(6);
    x.SetRandom(123);
    Px1.SetRandom(456);
    Px2.SetRandom(789);

    P.Solve(x, Px1);
    ParallelBarrier barrier(1);
    P.ConcurrentSolve(x, Px2, {0, 1, 0, 6, barrier});

    EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2));
  }

  // Test with BlockSparseMatrix
  {
    DynamicArray<int> rowPtr({0, 2, 5, 8, 11, 14, 16});
    DynamicArray<int> ci({0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5});
    DynamicArray<Scalar> va({2, -1, -1, 3, -1, -1, 4, -1, -1, 2, -1, -1, 2, -1, -1, 2});
    SparseMatrix<Scalar, int, int> SpA(6, rowPtr, ci, va);
    auto bSpA = ToBlockSparseMatrix<2>(SpA);

    krylov::RelaxedILUPrec<BlockSparseMatrix<Scalar, 2>> P(bSpA, 0, Scalar(0));

    ColumnVector<Scalar> x(6), Px1(6), Px2(6);
    x.SetRandom(123);
    Px1.SetRandom(456);
    Px2.SetRandom(789);

    P.Solve(x, Px1);
    ParallelBarrier barrier(1);
    P.ConcurrentSolve(x, Px2, {0, 1, 0, 6, barrier});

    EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2));
  }

  // Test with dense Matrix
  {
    Matrix<Scalar> A(6, 6);
    A.SetRandom(123);
    for (int i = 0; i < 6; ++i) {
      A(i, i) += Scalar(5);
    }

    krylov::RelaxedILUPrec<Matrix<Scalar>> P(A, 0, Scalar(0));

    ColumnVector<Scalar> x(6), Px1(6), Px2(6);
    x.SetRandom(456);
    Px1.SetRandom(789);
    Px2.SetRandom(111);

    P.Solve(x, Px1);
    ParallelBarrier barrier(1);
    P.ConcurrentSolve(x, Px2, {0, 1, 0, 6, barrier});

    EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2));
  }
}
