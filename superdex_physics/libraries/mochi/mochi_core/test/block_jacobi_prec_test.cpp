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

#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
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

template <typename Scalar, typename InType, typename OutType, typename MatrixType>
static void EvalBlockJacobiPrec(MatrixType& A) {
  constexpr int kNumColsInAtCT = mochi::krylov::details::MatTraits<InType>::kNumCols;
  int const numColsB = (kNumColsInAtCT == mochi::krylov::kDynamic) ? 11 : kNumColsInAtCT;
  int numRows = A.Rows();
  InType B(numRows, numColsB);
  for (int ii = 0; ii < numRows; ++ii) {
    for (int jj = 0; jj < numColsB; ++jj) {
      B(ii, jj) = Scalar(jj + 1) * (Scalar(ii) + 1);
    }
  }
  OutType JB(numRows, numColsB);

  {
    constexpr real kRelTol = 10_r * std::numeric_limits<real>::epsilon();
    mochi::krylov::JacobiPrec<Scalar> J(A);
    JB.SetRandom(123);
    J(B, JB);
    for (int jj = 0; jj < numColsB; ++jj) {
      EXPECT_NEAR_RTOL(JB(0, jj), Scalar(jj + 1) * Scalar(0.5), kRelTol);
      EXPECT_NEAR_RTOL(JB(1, jj), Scalar(jj + 1) * Scalar(2) / Scalar(3), kRelTol);
      EXPECT_NEAR_RTOL(JB(2, jj), Scalar(jj + 1) * Scalar(3) / Scalar(4), kRelTol);
      EXPECT_NEAR_RTOL(JB(3, jj), Scalar(jj + 1) * Scalar(4) / Scalar(2), kRelTol);
      EXPECT_NEAR_RTOL(JB(4, jj), Scalar(jj + 1) * Scalar(5) / Scalar(2), kRelTol);
      EXPECT_NEAR_RTOL(JB(5, jj), Scalar(jj + 1) * Scalar(6) / Scalar(2), kRelTol);
    }
  }
  {
    mochi::krylov::BlockJacobiPrec<Scalar, 2> J(A);
    JB.SetRandom(234);
    J(B, JB);
    for (int jj = 0; jj < numColsB; ++jj) {
      EXPECT_NEAR_EQ(JB(0, jj), Scalar(jj + 1) * Scalar(1));
      EXPECT_NEAR_EQ(JB(1, jj), Scalar(jj + 1) * Scalar(1));
      EXPECT_NEAR_RTOL(JB(2, jj), Scalar(jj + 1) * Scalar(1.4286), 0.0001);
      EXPECT_NEAR_RTOL(JB(3, jj), Scalar(jj + 1) * Scalar(2.7143), 0.0001);
      EXPECT_NEAR_RTOL(JB(4, jj), Scalar(jj + 1) * Scalar(5.3333), 0.0001);
      EXPECT_NEAR_RTOL(JB(5, jj), Scalar(jj + 1) * Scalar(5.6666), 0.0001);
    }
  }
  {
    mochi::krylov::BlockJacobiPrec<Scalar, 3> J(A);
    JB.SetRandom(345);
    J(B, JB);
    for (int jj = 0; jj < numColsB; ++jj) {
      EXPECT_NEAR_RTOL(JB(0, jj), Scalar(jj + 1) * Scalar(1.2222), 0.0001);
      EXPECT_NEAR_RTOL(JB(1, jj), Scalar(jj + 1) * Scalar(1.4444), 0.0001);
      EXPECT_NEAR_RTOL(JB(2, jj), Scalar(jj + 1) * Scalar(1.1111), 0.0001);
      EXPECT_NEAR_EQ(JB(3, jj), Scalar(jj + 1) * Scalar(7));
      EXPECT_NEAR_EQ(JB(4, jj), Scalar(jj + 1) * Scalar(10));
      EXPECT_NEAR_EQ(JB(5, jj), Scalar(jj + 1) * Scalar(8));
    }
  }
  {
    mochi::krylov::BlockJacobiPrec<Scalar, 6> J(A);
    JB.SetRandom(456);
    J(B, JB);
    for (int jj = 0; jj < numColsB; ++jj) {
      EXPECT_NEAR_RTOL(JB(0, jj), Scalar(jj + 1) * Scalar(1.7719), 0.0001);
      EXPECT_NEAR_RTOL(JB(1, jj), Scalar(jj + 1) * Scalar(2.5439), 0.0001);
      EXPECT_NEAR_RTOL(JB(2, jj), Scalar(jj + 1) * Scalar(3.8596), 0.0001);
      EXPECT_NEAR_RTOL(JB(3, jj), Scalar(jj + 1) * Scalar(9.8947), 0.0001);
      EXPECT_NEAR_RTOL(JB(4, jj), Scalar(jj + 1) * Scalar(11.9298), 0.0001);
      EXPECT_NEAR_RTOL(JB(5, jj), Scalar(jj + 1) * Scalar(8.9649), 0.0001);
    }
  }

  // Test Solve and ConcurrentSolve() methods. Only supported for column vectors.
  {
    constexpr int kPrecBlockSize = 2;
    BlockJacobiPrec<Scalar, kPrecBlockSize> J(A);

    ColumnVector<Scalar> X(numRows), JX1(numRows), JX2(numRows);
    X.SetRandom(1);
    JX1.SetRandom(2);
    JX2.SetRandom(3);

    J(X, JX1);
    J.Solve(X, JX2);
    EXPECT_TRUE(mochi::test::NearEqualMatrices(JX1, JX2));

    int const concurrentSolveRowBegin = kPrecBlockSize;
    auto const concurrentSolveRowEnd = numRows - kPrecBlockSize;
    EXPECT_GT(concurrentSolveRowEnd, concurrentSolveRowBegin); // Not a dummy test.
    JX2.SetZero();
    J.ConcurrentSolve(
        X, JX2, {0, 1, concurrentSolveRowBegin, concurrentSolveRowEnd, ParallelBarrier(1)});
    EXPECT_EQ(0, JX2.TopRows(concurrentSolveRowBegin).Norm());
    EXPECT_EQ(0, JX2.BottomRows(numRows - concurrentSolveRowEnd).Norm());
    EXPECT_TRUE(
        mochi::test::NearEqualMatrices(
            JX1.MiddleRows(
                concurrentSolveRowBegin, concurrentSolveRowEnd - concurrentSolveRowBegin),
            JX2.MiddleRows(
                concurrentSolveRowBegin, concurrentSolveRowEnd - concurrentSolveRowBegin)));

    JX2.SetRandom(4);
    ColumnVector<Scalar> const expectedJX2 = JX2;
    J.ConcurrentSolve(X, JX2, {0, 1, numRows, numRows, ParallelBarrier(1)});
    EXPECT_TRUE(mochi::test::NearEqualMatrices(expectedJX2, JX2, Scalar{0}));
  }
}

TEST(BlockJacobiPrec, BlockJacobiPrec) {
  DynamicArray<int> rp{0, 2, 5, 8, 11, 14, 16};
  DynamicArray<int> ci{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5};
  DynamicArray<Scalar> va{2, -1, -1, 3, -1, -1, 4, -1, -1, 2, -1, -1, 2, -1, -1, 2};
  SparseMatrix<Scalar, int, int> C(6, rp, ci, va);
  {
    EvalBlockJacobiPrec<Scalar, ColumnVector<Scalar>, ColumnVector<Scalar>>(C);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(C);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, RowMatrix<Scalar>>(C);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, Matrix<Scalar>>(C);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(C);
  }
  {
    auto bSpC = ToBlockSparseMatrix<2>(C);
    EvalBlockJacobiPrec<Scalar, ColumnVector<Scalar>, ColumnVector<Scalar>>(bSpC);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(bSpC);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, RowMatrix<Scalar>>(bSpC);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, Matrix<Scalar>>(bSpC);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(bSpC);
  }
  {
    auto bSpC = ToBlockSparseMatrix<3>(C);
    EvalBlockJacobiPrec<Scalar, ColumnVector<Scalar>, ColumnVector<Scalar>>(bSpC);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(bSpC);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, RowMatrix<Scalar>>(bSpC);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, Matrix<Scalar>>(bSpC);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(bSpC);
  }
  {
    auto dC = ToMatrix(C);
    EvalBlockJacobiPrec<Scalar, ColumnVector<Scalar>, ColumnVector<Scalar>>(dC);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(dC);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, RowMatrix<Scalar>>(dC);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, Matrix<Scalar>>(dC);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(dC);
  }
  {
    RowMatrix<Scalar> dC(6, 6); // Row-major, dynamic
    dC = ToMatrix(C);
    EvalBlockJacobiPrec<Scalar, ColumnVector<Scalar>, ColumnVector<Scalar>>(dC);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(dC);
    EvalBlockJacobiPrec<Scalar, Matrix<Scalar>, RowMatrix<Scalar>>(dC);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, Matrix<Scalar>>(dC);
    EvalBlockJacobiPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(dC);
  }
}
