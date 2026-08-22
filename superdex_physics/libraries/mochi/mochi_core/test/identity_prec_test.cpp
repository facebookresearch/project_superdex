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

#include <mochi_core/linear_algebra/krylov/identity_prec.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::krylov;

// Use real instead of float or double to reduce build time. Both are checked by CI.
using Scalar = real;

template <typename Scalar, typename InType, typename OutType, typename MatrixType>
static void EvalIdentityPrec(MatrixType const& A) {
  constexpr int kNumColsInAtCT = krylov::details::MatTraits<InType>::kNumCols;
  int const numRowsB = A.Rows();
  int const numColsB = (kNumColsInAtCT == krylov::kDynamic) ? 11 : kNumColsInAtCT;
  EXPECT_EQ(A.Rows(), A.Cols());
  krylov::IdentityPrec<Scalar> P(A);

  // Test operator() method.
  {
    InType B(numRowsB, numColsB);
    OutType PB(numRowsB, numColsB);
    B.SetRandom(1);
    PB.SetRandom(2);
    P(B, PB);
    EXPECT_TRUE(mochi::test::NearEqualMatrices(B, PB, Scalar(0)));
  }

  // Test Solve and ConcurrentSolve() methods. Only supported for column vectors.
  {
    ColumnVector<Scalar> B(numRowsB);
    ColumnVector<Scalar> PB(numRowsB);
    B.SetRandom(1);
    PB.SetRandom(2);

    int const numConcurrentSolveRows = numRowsB / 2;
    EXPECT_GT(numConcurrentSolveRows, 0); // Not a dummy test.
    P.ConcurrentSolve(B, PB, {0, 1, 0, numConcurrentSolveRows, ParallelBarrier(1)});
    EXPECT_TRUE(
        mochi::test::NearEqualMatrices(
            B.TopRows(numConcurrentSolveRows), PB.TopRows(numConcurrentSolveRows), Scalar(0)));
    for (int r = numConcurrentSolveRows; r < numRowsB; ++r) {
      EXPECT_FALSE(mochi::test::NearEqualMatrices(B.Row(r), PB.Row(r), Scalar(0)));
    }

    PB.SetRandom(3);
    P.Solve(B, PB);
    EXPECT_TRUE(mochi::test::NearEqualMatrices(B, PB, Scalar(0)));
  }
}

TEST(IdentityPrec, IdentityPrec) {
  int const n = 10;
  Matrix<Scalar> A(n, n); // Implementation of IdentityPrec is independent of the matrix type. No
                          // need to test with other matrix types.
  A.SetRandom(123);
  EvalIdentityPrec<Scalar, ColumnVector<Scalar>, ColumnVector<Scalar>>(A);
  EvalIdentityPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(A);
  EvalIdentityPrec<Scalar, Matrix<Scalar>, RowMatrix<Scalar>>(A);
  EvalIdentityPrec<Scalar, RowMatrix<Scalar>, Matrix<Scalar>>(A);
  EvalIdentityPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(A);
}
