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
 * @file matrix_operations_test.cpp
 * @brief Tests free functions defined in matrix_operations.h.
 *
 * @note Inverse, StableInverse, SymInverse, and Determinant (also defined in matrix_operations.h)
 * are tested in ldlt_lu_test.cpp alongside the LU/LDLt decompositions that implement them.
 */

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>

using namespace mochi;
using namespace mochi::test;

template <int N, typename VecA, typename VecB>
static void TestDotHelper(VecA const& a, VecB const& b) {
  ASSERT_TRUE(N <= a.Rows() && a.Cols() == 1);
  ASSERT_TRUE(N <= b.Rows() && b.Cols() == 1);
  auto const expectedDot = static_cast<real>(N * (N + 1));

  // Test all combinations of compile-time and dynamic number of rows.
  EXPECT_NEAR_EQ(expectedDot, (a.TopRows(N).Dot(b.TopRows(N))));
  EXPECT_NEAR_EQ(expectedDot, (a.template TopRows<N>(N).Dot(b.TopRows(N))));
  EXPECT_NEAR_EQ(expectedDot, (a.TopRows(N).Dot(b.template TopRows<N>(N))));
  EXPECT_NEAR_EQ(expectedDot, (a.template TopRows<N>(N).Dot(b.template TopRows<N>(N))));

  // Using Dot utility function.
  EXPECT_NEAR_EQ(expectedDot, Dot(a.TopRows(N), b.TopRows(N))(0, 0));
  EXPECT_NEAR_EQ(expectedDot, Dot(a.template TopRows<N>(N), b.TopRows(N))(0, 0));
  EXPECT_NEAR_EQ(expectedDot, Dot(a.TopRows(N), b.template TopRows<N>(N))(0, 0));
  EXPECT_NEAR_EQ(expectedDot, Dot(a.template TopRows<N>(N), b.template TopRows<N>(N))(0, 0));
}

template <int N>
static void TestDot(Matrix<real> const& Ac, Matrix<real> const& Bc) {
  RowMatrix<real> Ar = Ac;
  RowMatrix<real> Br = Bc;

  // Test all combinations of col-major and row-major views.
  TestDotHelper<N>(Ac.Col(0), Bc.Col(0));
  TestDotHelper<N>(Ac.Col(0), Br.Col(0));
  TestDotHelper<N>(Ar.Col(0), Bc.Col(0));
  TestDotHelper<N>(Ar.Col(0), Br.Col(0));

  // Now with owning column vectors.
  TestDotHelper<N>(ColumnVector<real>(Ac.Col(0)), ColumnVector<real>(Bc.Col(0)));
}

TEST(MatrixOperations, Dot) {
  constexpr int kMaxSizeTested = 33;

  // Only the 1st column is relevant. The last 4 columns are unused.
  Matrix<real> A(kMaxSizeTested, 5), B(kMaxSizeTested, 5);
  A.SetRandom(123);
  B.SetRandom(234);
  for (int i = 0; i < kMaxSizeTested; ++i) {
    A(i, 0) = Sqrt(static_cast<real>(i + 1));
    B(i, 0) = 2_r * Sqrt(static_cast<real>(i + 1));
  }

  TestDot<1>(A, B);
  TestDot<2>(A, B);
  TestDot<3>(A, B);
  TestDot<4>(A, B);
  TestDot<5>(A, B);
  TestDot<6>(A, B);
  TestDot<7>(A, B);
  TestDot<8>(A, B);
  TestDot<9>(A, B);
  TestDot<10>(A, B);
  TestDot<11>(A, B);
  TestDot<12>(A, B);
  TestDot<13>(A, B);
  TestDot<14>(A, B);
  TestDot<15>(A, B);
  TestDot<16>(A, B);
  TestDot<17>(A, B);
  TestDot<23>(A, B);
  TestDot<24>(A, B);
  TestDot<25>(A, B);
  TestDot<31>(A, B);
  TestDot<32>(A, B);
  TestDot<33>(A, B);
}

TEST(MatrixOperations, ParallelMatMatAlongK) {
  TaskScheduler scheduler(TaskScheduler::GetNumSupportedLogicalProcessors());
  int const m = 100;
  int const n = 100;
  std::vector<int> kList = {10}; // Tests the single-threaded implementation.
#if MOCHI_OPTIMIZED
  kList.push_back(269); // Tests the multi-threaded implementation. Prime number to ensure uneven
                        // load among workers so that that all codepaths are tested.
#endif
  for (int k : kList) {
    real const relTol = 4 * k * std::numeric_limits<real>::epsilon();
    Matrix<real> A(m, k);
    Matrix<real> B(k, n);
    A.SetRandom(1);
    B.SetRandom(2);
    Matrix<real> Cref = A * B;
    // Col-major.
    Matrix<real> C(m, n);
    ParallelMatMatAlongK(A, B, C); // Owning
    EXPECT_TRUE(test::NearEqualMatrices(Cref, C, relTol));
    C.SetRandom(3);
    ParallelMatMatAlongK(A, B, AsView(C)); // View
    EXPECT_TRUE(test::NearEqualMatrices(Cref, C, relTol));
    // Row-major.
    C.SetRandom(4);
    ParallelMatMatAlongK(B.Transpose(), A.Transpose(), C.Transpose());
    EXPECT_TRUE(test::NearEqualMatrices(Cref, C, relTol));
    // += overload.
    ParallelMatMatAlongK<true>(A, B, C);
    EXPECT_TRUE(test::NearEqualMatrices(Matrix<real>(2_r * Cref), C, 2 * relTol));
  }
}

TEST(MatrixOperations, HasOverlap) {
  std::array<real, 8> values{};
  auto* v = values.data();

  EXPECT_TRUE(krylov::HasOverlap(v, 4, v, 4));
  EXPECT_TRUE(krylov::HasOverlap(v, 5, v + 4, 4));
  EXPECT_TRUE(krylov::HasOverlap(v + 4, 4, v, 5));

  EXPECT_FALSE(krylov::HasOverlap(v, 4, v + 4, 4));
  EXPECT_FALSE(krylov::HasOverlap(v + 4, 4, v, 4));
  EXPECT_FALSE(krylov::HasOverlap(v, 2, v + 4, 2));
  EXPECT_FALSE(krylov::HasOverlap(v + 4, 2, v, 2));

  // A zero-length range never overlaps, even at a shared endpoint.
  EXPECT_FALSE(krylov::HasOverlap(v, 0, v, 4));
  EXPECT_FALSE(krylov::HasOverlap(v, 4, v, 0));
}

TEST(MatrixOperations, Trace) {
  RowMatrix<real, 3, 3> A{1.0_r, 0.0_r, 0.0_r, 2.0_r, 3.0_r, -2.0_r, 3.0_r, 4.0_r, -3.0_r};
  EXPECT_NEAR_EQ(Trace(A), 1.0_r);
  EXPECT_NEAR_EQ(Trace(A.Transpose()), 1.0_r);
  auto ASp = ToSparseMatrix(A, true);
  EXPECT_NEAR_EQ(Trace(ASp), 1.0_r);
  auto ABSp = ToBlockSparseMatrix<1>(A, true);
  EXPECT_NEAR_EQ(Trace(ABSp), 1.0_r);

  Matrix<real, 6, 6> B{2.0_r,  0.0_r, -1.5_r, 0.0_r,  0.0_r,  -1.2_r, 0.0_r, 3.0_r,  0.0_r,
                       0.0_r,  0.0_r, -0.5_r, -1.5_r, 0.0_r,  4.0_r,  0.0_r, 0.0_r,  0.0_r,
                       0.0_r,  0.0_r, 0.0_r,  2.0_r,  -1.0_r, 0.0_r,  0.0_r, 0.0_r,  0.0_r,
                       -1.0_r, 2.0_r, -1.0_r, -1.2_r, -0.5_r, 0.0_r,  0.0_r, -1.0_r, 2.0_r};
  EXPECT_NEAR_EQ(Trace(B), 15.0_r);
  EXPECT_NEAR_EQ(Trace(B.Transpose()), 15.0_r);
  auto BSp = ToSparseMatrix(B, true);
  EXPECT_NEAR_EQ(Trace(BSp), 15.0_r);
  auto BBSp1 = ToBlockSparseMatrix<1>(B, true);
  EXPECT_NEAR_EQ(Trace(BBSp1), 15.0_r);
  auto BBSp2 = ToBlockSparseMatrix<2>(B, true);
  EXPECT_NEAR_EQ(Trace(BBSp2), 15.0_r);
  auto BBSp3 = ToBlockSparseMatrix<3>(B, true);
  EXPECT_NEAR_EQ(Trace(BBSp3), 15.0_r);
  auto BBSp6 = ToBlockSparseMatrix<6>(B, true);
  EXPECT_NEAR_EQ(Trace(BBSp6), 15.0_r);
}
