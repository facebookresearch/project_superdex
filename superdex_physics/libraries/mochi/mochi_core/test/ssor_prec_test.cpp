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

#include <mochi_core/linear_algebra/krylov/block_ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/ssor_prec.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::krylov;

template <typename Scalar, typename InType, typename OutType, typename MatrixType>
static void EvalBlockSSORPrec(MatrixType& A) {
  int nrows = A.Rows();
  InType b(nrows, 2);
  for (int ii = 0; ii < nrows; ++ii) {
    b(ii, 0) = Scalar(ii) + 1;
    b(ii, 1) = Scalar(nrows - ii);
  }
  OutType jb(nrows, 2);
  {
    mochi::krylov::SSORPrec<Scalar, MatrixType> J(A, 0.7);
    J(b, jb);
    Matrix<Scalar, 6, 2> ref(
        {Scalar(0.822151),
         Scalar(1.049),
         Scalar(1.44072),
         Scalar(3.61986),
         Scalar(4.33522),
         Scalar(3.78381),
         Scalar(3.6243),
         Scalar(2.55514),
         Scalar(1.72059),
         Scalar(2.4783),
         Scalar(1.89396),
         Scalar(0.995888)});
    EXPECT_TRUE(mochi::test::NearEqualMatrices<OutType>(jb, ref, Scalar(0.001)));
  }
  {
    //
    // Test for correctness
    // SSORPrec must be preferred over BlockSSORPrec<.., 1, ...>
    //
    mochi::krylov::BlockSSORPrec<Scalar, 1, MatrixType> J(A, 0.7);
    J(b, jb);
    Matrix<Scalar, 6, 2> ref(
        {Scalar(0.822151),
         Scalar(1.049),
         Scalar(1.44072),
         Scalar(3.61986),
         Scalar(4.33522),
         Scalar(3.78381),
         Scalar(3.6243),
         Scalar(2.55514),
         Scalar(1.72059),
         Scalar(2.4783),
         Scalar(1.89396),
         Scalar(0.995888)});
    EXPECT_TRUE(mochi::test::NearEqualMatrices<OutType>(jb, ref, Scalar(0.001)));
  }
  {
    mochi::krylov::BlockSSORPrec<Scalar, 2, MatrixType> J(A, 0.7);
    J(b, jb);
    EXPECT_NEAR_RTOL(jb(0, 0), Scalar(1.20216), 0.001);
    EXPECT_NEAR_RTOL(jb(1, 0), Scalar(1.49432), 0.001);
    EXPECT_NEAR_RTOL(jb(2, 0), Scalar(2.08685), 0.001);
    EXPECT_NEAR_RTOL(jb(3, 0), Scalar(4.98039), 0.001);
    EXPECT_NEAR_RTOL(jb(4, 0), Scalar(6.04847), 0.001);
    EXPECT_NEAR_RTOL(jb(5, 0), Scalar(5.75423), 0.001);
    EXPECT_NEAR_RTOL(jb(0, 1), Scalar(4.50446), 0.001);
    EXPECT_NEAR_RTOL(jb(1, 1), Scalar(3.54892), 0.001);
    EXPECT_NEAR_RTOL(jb(2, 1), Scalar(2.27472), 0.001);
    EXPECT_NEAR_RTOL(jb(3, 1), Scalar(3.42049), 0.001);
    EXPECT_NEAR_RTOL(jb(4, 1), Scalar(2.62323), 0.001);
    EXPECT_NEAR_RTOL(jb(5, 1), Scalar(1.76661), 0.001);
  }
  {
    mochi::krylov::BlockSSORPrec<Scalar, 3, MatrixType> J(A, 0.7);
    J(b, jb);
    EXPECT_NEAR_RTOL(jb(0, 0), Scalar(1.38059), 0.001);
    EXPECT_NEAR_RTOL(jb(1, 0), Scalar(1.85118), 0.001);
    EXPECT_NEAR_RTOL(jb(2, 0), Scalar(2.35294), 0.001);
    EXPECT_NEAR_RTOL(jb(3, 0), Scalar(6.90083), 0.001);
    EXPECT_NEAR_RTOL(jb(4, 0), Scalar(9.45389), 0.001);
    EXPECT_NEAR_RTOL(jb(5, 0), Scalar(7.45694), 0.001);
    EXPECT_NEAR_RTOL(jb(0, 1), Scalar(4.71102), 0.001);
    EXPECT_NEAR_RTOL(jb(1, 1), Scalar(3.96204), 0.001);
    EXPECT_NEAR_RTOL(jb(2, 1), Scalar(2.6251), 0.001);
    EXPECT_NEAR_RTOL(jb(3, 1), Scalar(4.1405), 0.001);
    EXPECT_NEAR_RTOL(jb(4, 1), Scalar(4.277), 0.001);
    EXPECT_NEAR_RTOL(jb(5, 1), Scalar(2.5935), 0.001);
  }
  {
    mochi::krylov::BlockSSORPrec<Scalar, 6, MatrixType> J(A);
    J(b, jb);
    EXPECT_NEAR_RTOL(jb(0, 0), Scalar(1.7719), 0.0001);
    EXPECT_NEAR_RTOL(jb(1, 0), Scalar(2.5439), 0.0001);
    EXPECT_NEAR_RTOL(jb(2, 0), Scalar(3.8596), 0.0001);
    EXPECT_NEAR_RTOL(jb(3, 0), Scalar(9.8947), 0.0001);
    EXPECT_NEAR_RTOL(jb(4, 0), Scalar(11.9298), 0.0001);
    EXPECT_NEAR_RTOL(jb(5, 0), Scalar(8.9649), 0.0001);
    EXPECT_NEAR_RTOL(jb(0, 1), Scalar(5.3509), 0.0001);
    EXPECT_NEAR_RTOL(jb(1, 1), Scalar(4.7018), 0.0001);
    EXPECT_NEAR_RTOL(jb(2, 1), Scalar(3.7544), 0.0001);
    EXPECT_NEAR_RTOL(jb(3, 1), Scalar(6.3158), 0.0001);
    EXPECT_NEAR_RTOL(jb(4, 1), Scalar(5.8772), 0.0001);
    EXPECT_NEAR_RTOL(jb(5, 1), Scalar(3.4386), 0.0001);
  }
}

TEST(BlockSSORPrec, BlockSSORPrec) {
  // Note SSOR and block SSOR preconditioners don't support mixed input and output orientations yet.
  // Float tested in single-precision build. Double tested in double-precision build.
  using Scalar = real;
  DynamicArray<int> rp{0, 2, 5, 8, 11, 14, 16};
  DynamicArray<int> ci{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5};
  DynamicArray<Scalar> va{2, -1, -1, 3, -1, -1, 4, -1, -1, 2, -1, -1, 2, -1, -1, 2};
  SparseMatrix<Scalar, int, int> SpC(6, rp, ci, va);
  {
    EvalBlockSSORPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(SpC);
    EvalBlockSSORPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(SpC);
  }
  {
    auto bSpC = ToBlockSparseMatrix<2>(SpC);
    EvalBlockSSORPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(bSpC);
    EvalBlockSSORPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(bSpC);
  }
  {
    auto bSpC = ToBlockSparseMatrix<3>(SpC);
    EvalBlockSSORPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(bSpC);
    EvalBlockSSORPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(bSpC);
  }
  {
    Matrix<Scalar, 6, 6> dC = ToMatrix(SpC); // Compile-time, row-major
    EvalBlockSSORPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(dC);
    EvalBlockSSORPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(dC);
  }
  {
    RowMatrix<Scalar> dC(6, 6); // Dynamic, row-major
    dC = ToMatrix(SpC);
    EvalBlockSSORPrec<Scalar, Matrix<Scalar>, Matrix<Scalar>>(dC);
    EvalBlockSSORPrec<Scalar, RowMatrix<Scalar>, RowMatrix<Scalar>>(dC);
  }
}

TEST(SSORPrec, SSORPrec) {
  using Scalar = real;
  DynamicArray<int> rp{0, 2, 5, 8, 11, 14, 16};
  DynamicArray<int> ci{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5};
  DynamicArray<Scalar> va{2, -1, -1, 3, -1, -1, 4, -1, -1, 2, -1, -1, 2, -1, -1, 2};
  SparseMatrix<Scalar, int, int> SpC(6, rp, ci, va);
  auto DeC = ToMatrix(SpC);
  int const n = DeC.Rows();
  ColumnVector<Scalar> x(n), y(n);
  x.SetRandom(12);
  mochi::krylov::SSORPrec<Scalar, SparseMatrix<Scalar, int, int>> prec(SpC, 0.7);
  prec(x, y);
  auto tol = Scalar(0.0001);
  for (int i = 1; i < 7; ++i) {
    Matrix<Scalar> K(n * i, n * i);
    K.SetRandom(33);
    for (int j = 0; j < i; ++j) {
      K.Block(j * n, j * n, n, n) = DeC;
    }
    auto KSp = ToSparseMatrix(K, /*pruneZeros*/ true);
    mochi::krylov::SSORPrec<Scalar, SparseMatrix<Scalar, int, int>> precl(KSp, 0.7);
    ColumnVector<Scalar> xl(i * n), yl(i * n);
    for (int j = 0; j < i; ++j) {
      xl.MiddleRows(j * n, n) = x;
    }
    yl.SetZero();
    for (int j = 0; j < i; ++j) {
      precl.ConcurrentSolve(xl, yl, {0, 1, j * n, (j + 1) * n, ParallelBarrier(1)});
    }
    for (int j = 0; j < i; ++j) {
      EXPECT_TRUE(mochi::test::NearEqualMatrices(y, yl.MiddleRows(j * n, n), tol));
    }
  }
}
