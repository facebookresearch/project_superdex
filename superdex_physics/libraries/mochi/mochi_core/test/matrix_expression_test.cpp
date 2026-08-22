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
 * @file matrix_expression_test.cpp
 * @brief Tests expression template evaluation (matrix_expressions.h, matrix_assignment.h,
 * host_matrix_eval.h).
 */

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>

#include "matrix_expression_test.h"

#include <gtest/gtest.h>

#include <numbers>
#include <vector>

using namespace mochi;

template <
    typename Scalar,
    int kDestRowsAtCT,
    int kDestColsAtCT,
    krylov::Direction kDestDir,
    int kDestLeadDim,
    typename SrcMatType>
static void TestMatrixCopyGivenSrcMatAndDestParams(SrcMatType const& A, int m, int n) {
  constexpr auto kOwner = krylov::Ownership::Owner;

  // Determine runtime leading dimension of the destination matrix.
  constexpr bool kIsDestDynamicSize = (kDestRowsAtCT < 0) || (kDestColsAtCT < 0);
  int const destLeadDim = kDestLeadDim > 0
      ? kDestLeadDim
      : ((kDestLeadDim == krylov::kAutomaticLeadDim || !kIsDestDynamicSize)
             ? (kDestDir == krylov::Direction::ColMajor ? m : n) // Only LeadDim = Size is legal.
             : m + n + A.LeadDim()); // LeadDim > Size is legal. Use DestLeadDim > Size, SrcLeadDim
                                     // to increase test coverage.

  // View of the source matrix.
  auto Av = AsConstView(A);

  // Copy constructor tests.
  {
    Matrix<Scalar, kDestRowsAtCT, kDestColsAtCT, kDestDir, kOwner, kDestLeadDim> B1(A);
    EXPECT_TRUE(test::IsNear(A, B1, Scalar(0)));
    Matrix<Scalar, kDestRowsAtCT, kDestColsAtCT, kDestDir, kOwner, kDestLeadDim> B2(Av);
    EXPECT_TRUE(test::IsNear(A, B2, Scalar(0)));
  }

  // Copy assignment tests.
  {
    Matrix<Scalar, kDestRowsAtCT, kDestColsAtCT, kDestDir, kOwner, kDestLeadDim> B1(
        m, n, destLeadDim);
    B1 = A;
    EXPECT_TRUE(test::IsNear(A, B1, Scalar(0)));
    Matrix<Scalar, kDestRowsAtCT, kDestColsAtCT, kDestDir, kOwner, kDestLeadDim> B2(
        m, n, destLeadDim);
    B2 = Av;
    EXPECT_TRUE(test::IsNear(A, B2, Scalar(0)));
  }
}

template <
    typename Scalar,
    int m,
    int n,
    int kSrcRowsAtCT,
    int kSrcColsAtCT,
    krylov::Direction kSrcDir,
    int kSrcLeadDim>
static void TestMatrixCopyGivenSrcParams() {
  constexpr auto kColMajor = krylov::Direction::ColMajor;
  constexpr auto kRowMajor = krylov::Direction::RowMajor;
  constexpr auto kDynamic = krylov::kDynamic;
  constexpr auto kAutoLeadDim = krylov::kAutomaticLeadDim;

  // Determine runtime leading dimension of the source matrix.
  constexpr bool kIsSrcDynamicSize = (kSrcRowsAtCT < 0) || (kSrcColsAtCT < 0);
  int const srcLeadDim = kSrcLeadDim > 0
      ? kSrcLeadDim
      : ((kSrcLeadDim == kAutoLeadDim || !kIsSrcDynamicSize)
             ? (kSrcDir == kColMajor ? m : n) // Only LeadDim = Size is legal.
             : (m + n + 1)); // LeadDim > Size is legal. Use it to increase test coverage.

  // Construct source matrix.
  Matrix<Scalar, kSrcRowsAtCT, kSrcColsAtCT, kSrcDir, krylov::Ownership::Owner, kSrcLeadDim> A(
      m, n, srcLeadDim);
  A.SetRandom(1);

  // Destination matrix with automatic leading dimension.
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, m, n, kColMajor, kAutoLeadDim>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, m, kDynamic, kColMajor, kAutoLeadDim>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, kDynamic, n, kColMajor, kAutoLeadDim>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, kDynamic, kDynamic, kColMajor, kAutoLeadDim>(
      A, m, n);

  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, m, n, kRowMajor, kAutoLeadDim>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, m, kDynamic, kRowMajor, kAutoLeadDim>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, kDynamic, n, kRowMajor, kAutoLeadDim>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, kDynamic, kDynamic, kRowMajor, kAutoLeadDim>(
      A, m, n);

  // Destination matrix with dynamic leading dimension.
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, m, kDynamic, kColMajor, kDynamic>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, kDynamic, n, kColMajor, kDynamic>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, kDynamic, kDynamic, kColMajor, kDynamic>(A, m, n);

  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, m, kDynamic, kRowMajor, kDynamic>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, kDynamic, n, kRowMajor, kDynamic>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, kDynamic, kDynamic, kRowMajor, kDynamic>(A, m, n);

  // Destination matrix with compile-time leading dimension. This case requires the size along the
  // storage direction to be compile-time. If both sizes are compile-time, the leading dimension
  // must be the same as (and not greater than) the size along the storage direction. All other
  // cases are illegal.
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, m, n, kColMajor, m>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, m, kDynamic, kColMajor, 2 * m + 1>(A, m, n);

  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, m, n, kRowMajor, n>(A, m, n);
  TestMatrixCopyGivenSrcMatAndDestParams<Scalar, kDynamic, n, kRowMajor, 2 * n + 1>(A, m, n);
}

template <
    typename Scalar,
    krylov::Direction aDir,
    krylov::Direction bDir,
    krylov::Direction cDir,
    krylov::Ownership kOwner = krylov::Ownership::Owner>
static void TestLargeProduct() {
  struct Sizes {
    int m, n, k;
  };

  std::vector<Sizes> combinations = {
      {16, 16, 16},
      {16, 32, 16},
      {16, 16, 32},
      {32, 16, 16},
      {13, 5, 7},
      {13, 7, 5},
      {7, 5, 13},
      {5, 7, 13},
      {7, 13, 5},
      {5, 13, 7},
      {13, 5, 1},
      {13, 1, 5},
      {1, 5, 13},
      {5, 1, 13},
      {1, 13, 5},
      {5, 13, 1},
      {33, 34, 256}};

#if MOCHI_OPTIMIZED
  for (int i = 0; i < 16; ++i) {
    combinations.push_back({64 + i, 64, 64});
    combinations.push_back({64, 64 + i, 64});
    combinations.push_back({64, 64, 64 + i});
  }
  for (int i = 0; i < 16; ++i) {
    combinations.push_back({1 + i, 64, 64});
    combinations.push_back({64, 1 + i, 64});
    combinations.push_back({64, 64, 1 + i});
  }
#endif

  for (auto [m, n, k] : combinations) {
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, aDir, kOwner> A(m, k);
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, bDir, kOwner> B(k, n);
    A.SetRandom(123);
    B.SetRandom(234);
    //
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, cDir, kOwner> C = A * B;
    Matrix<Scalar> D(m, n), F(m, n);
    test::MultiplyMatrices(Scalar(1), A, B, D);
    F = D; // F is A*B
    EXPECT_TRUE(test::IsNear(C, D, test::GetTol<Scalar>(k)));
    //
    C.SetRandom(345);
    EXPECT_FALSE(test::IsNear(C, D, test::GetTol<Scalar>(k)));
    Apply(A, B, C);
    EXPECT_TRUE(test::IsNear(C, D, test::GetTol<Scalar>(k)));
    //
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, cDir, kOwner> CT =
        Transpose(B) * Transpose(A);
    EXPECT_TRUE(test::IsNear(Transpose(CT), D, test::GetTol<Scalar>(k)));
    //
    A.SetRandom(456);
    B.SetRandom(567);
    C += Scalar(2) * A * B;
    Matrix<Scalar> E(m, n);
    test::MultiplyMatrices(Scalar(1), A, B, E);
    test::AddMatrices(Scalar(2), E, D, D);
    EXPECT_TRUE(test::IsNear(C, D, 4 * test::GetTol<Scalar>(k)));
    //
    C -= Scalar(2) * A * B;
    EXPECT_TRUE(test::IsNear(C, F, 5 * test::GetTol<Scalar>(k)));
    //
    C = Scalar(2) * A * B + F;
    test::AddMatrices(Scalar(2), E, F, D);
    EXPECT_TRUE(test::IsNear(C, D, 3 * test::GetTol<Scalar>(k)));
    //
    C -= Scalar(2) * A * B + F - F;
    EXPECT_TRUE(test::IsNear(C, F, 4 * test::GetTol<Scalar>(k)));
    //
    C = A * B - A * B;
    F.SetZero();
    EXPECT_TRUE(test::IsNear(C, F, 4 * test::GetTol<Scalar>(k)));
    //
    F.SetRandom(678);
    C = A * B - F;
    D = -(F - A * B);
    EXPECT_TRUE(test::IsNear(C, D, 4 * test::GetTol<Scalar>(k)));
    //
    C = A * B - F;
    D = F - A * B;
    D = -D;
    EXPECT_TRUE(test::IsNear(C, D, 4 * test::GetTol<Scalar>(k)));
    //
    C = A * B - Scalar(2.34) * A * B;
    D = (1 - Scalar(2.34)) * A * B;
    EXPECT_TRUE(test::IsNear(C, D, 4 * test::GetTol<Scalar>(k)));
    //
    C = Scalar(2.34) * A;
    D = A * Scalar(2.34);
    EXPECT_TRUE(test::IsNear(C, D, 4 * test::GetTol<Scalar>(k)));
    //
    F.Resize(n, n);
    F.SetRandom(789);
    C = A * (-B);
    D = (-A) * B;
    EXPECT_TRUE(test::IsNear(C, D, 4 * test::GetTol<Scalar>(k)));
    //
    F.Resize(n, n);
    F.SetRandom(890);
    C = A * B * F;
    E = A * B;
    D = E * F;
    EXPECT_TRUE(test::IsNear(C, D, 24 * test::GetTol<Scalar>(k)));
    //
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, aDir, kOwner> G(k, n);
    G.SetRandom(901);
    F.Resize(k, n);
    C = Scalar(1.5) * A * (Scalar(2) * B + G);
    test::AddMatrices(Scalar(2), B, G, F);
    test::MultiplyMatrices(Scalar(1.5), A, F, D);
    EXPECT_TRUE(test::IsNear(C, D, 10 * test::GetTol<Scalar>(k)));
    //
    Matrix<Scalar, krylov::kDynamic, krylov::kDynamic, aDir, kOwner> H(n, n);
    H.SetRandom(012);
    E.Resize(k, n);
    C = (-A - Scalar(0) + Scalar(2) * A) *
        (-B * Scalar(1) * H + Scalar(0.5) * B * H - (Scalar(0) - B) * H * Scalar(1.5)) *
        Scalar(-1) * Scalar(1) * (Scalar(2) * H - H + Scalar(0));
    test::MultiplyMatrices(Scalar(1), B, H, E);
    test::MultiplyMatrices(Scalar(1), E, H, F);
    test::MultiplyMatrices(Scalar(-1), A, F, D);
    EXPECT_TRUE(test::IsNear(C, D, Scalar(n * n) * test::GetTol<Scalar>(k)));
    //
    {
      constexpr auto kDyn = krylov::kDynamic;
      Matrix<Scalar, kDyn, kDyn, aDir, kOwner, kDyn> AA(m, k, m + k);
      Matrix<Scalar, kDyn, kDyn, bDir, kOwner, kDyn> BB(k, n, k + n);
      AA.SetRandom(123);
      BB.SetRandom(234);
      Matrix<Scalar, kDyn, kDyn, cDir, kOwner, kDyn> CC(m, n, m + n);
      CC = AA * BB;
      Matrix<Scalar> DD(m, n);
      test::MultiplyMatrices(Scalar(1), AA, BB, DD);
      EXPECT_TRUE(test::IsNear(CC, DD, test::GetTol<Scalar>(k)));
      //
      CC.SetRandom(345);
      EXPECT_FALSE(test::IsNear(CC, DD, test::GetTol<Scalar>(k)));
      Apply(AA, BB, CC);
      EXPECT_TRUE(test::IsNear(CC, DD, test::GetTol<Scalar>(k)));
    }
  }
}

// Compilation is slow. Skip it in non-debug builds. Edit this line to locally run it in non-debug
// builds.
#if MOCHI_DEBUG
TEST(MatrixExpression, MatrixCopy) {
  using Scalar = float;
  constexpr int m = 19, n = 21;
  constexpr auto kColMajor = krylov::Direction::ColMajor;
  constexpr auto kRowMajor = krylov::Direction::RowMajor;
  constexpr auto kDynamic = krylov::kDynamic;
  constexpr auto kAutoLeadDim = krylov::kAutomaticLeadDim;

  // Source matrix with automatic leading dimension.

  TestMatrixCopyGivenSrcParams<Scalar, m, n, kDynamic, kDynamic, kColMajor, kAutoLeadDim>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, m, kDynamic, kColMajor, kAutoLeadDim>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, kDynamic, n, kColMajor, kAutoLeadDim>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, m, n, kColMajor, kAutoLeadDim>();

  TestMatrixCopyGivenSrcParams<Scalar, m, n, kDynamic, kDynamic, kRowMajor, kAutoLeadDim>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, m, kDynamic, kRowMajor, kAutoLeadDim>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, kDynamic, n, kRowMajor, kAutoLeadDim>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, m, n, kRowMajor, kAutoLeadDim>();

  // Source matrix with dynamic leading dimension.
  TestMatrixCopyGivenSrcParams<Scalar, m, n, kDynamic, kDynamic, kColMajor, kDynamic>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, m, kDynamic, kColMajor, kDynamic>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, kDynamic, n, kColMajor, kDynamic>();

  TestMatrixCopyGivenSrcParams<Scalar, m, n, kDynamic, kDynamic, kRowMajor, kDynamic>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, m, kDynamic, kRowMajor, kDynamic>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, kDynamic, n, kRowMajor, kDynamic>();

  // Source matrix with compile-time leading dimension. This case requires the size along the
  // storage direction to be compile-time. If both sizes are compile-time, the leading dimension
  // must be the same as (and not greater than) the size along the storage direction. All other
  // cases are illegal.
  TestMatrixCopyGivenSrcParams<Scalar, m, n, m, kDynamic, kColMajor, m + 1>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, m, n, kColMajor, m>();

  TestMatrixCopyGivenSrcParams<Scalar, m, n, kDynamic, n, kRowMajor, n + 1>();
  TestMatrixCopyGivenSrcParams<Scalar, m, n, m, n, kRowMajor, n>();
}
#endif

TEST(MatrixExpression, FixedSizeMatrixScaling) {
  {
    ColumnVector<float, 2> A{2.0f, 3.0f};
    A *= -1.1f;
    EXPECT_FLOAT_EQ(A(0, 0), -2.2f);
    EXPECT_FLOAT_EQ(A(1, 0), -3.3f);
  }

  {
    Matrix<double, 2, 2> A{double(2), double(3), double(4), double(5)};
    A *= 1.2;
    EXPECT_DOUBLE_EQ(A(0, 0), 2.4);
    EXPECT_DOUBLE_EQ(A(1, 0), 3.6);
    EXPECT_DOUBLE_EQ(A(0, 1), 4.8);
    EXPECT_DOUBLE_EQ(A(1, 1), 6.0);
  }
}

TEST(MatrixExpression, FixedSizeMatrixLinearCombination) {
  {
    ColumnVector<float, 2> A{2.0f, 3.0f};
    ColumnVector<float, 2> C;
    C = -3.1f * A;
    EXPECT_FLOAT_EQ(C(0, 0), -6.2f);
    EXPECT_FLOAT_EQ(C(1, 0), -9.3f);
  }

  {
    ColumnVector<float, 2> A(2.0f, 3.0f);
    ColumnVector<float, 2> B{4.0f, 5.0f};
    ColumnVector<float, 2> C;
    C = -3.1f * A + 1.1f * B;
    EXPECT_FLOAT_EQ(C(0, 0), -1.8f);
    EXPECT_FLOAT_EQ(C(1, 0), -3.8f);
  }

  {
    ColumnVector<double, 2> A{double(2), double(3)};
    ColumnVector<double, 2> C;
    C = -3.1 * A;
    EXPECT_DOUBLE_EQ(C(0, 0), -6.2);
    EXPECT_DOUBLE_EQ(C(1, 0), -9.3);
  }

  {
    ColumnVector<double, 2> A(2.0, 3.0);
    ColumnVector<double, 2> B{4.0, 5.0};
    ColumnVector<double, 2> C = -3.1 * A + 1.1 * B;
    EXPECT_DOUBLE_EQ(C(0, 0), -1.8);
    EXPECT_DOUBLE_EQ(C(1, 0), -3.8);
  }

  {
    constexpr Matrix<float, 2, 3> A{{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}};
    Matrix<float, 2, 3> C;
    C.SetConstant(-0.2f);
    C += A;
    EXPECT_FLOAT_EQ(C(0, 0), 0.8f);
    EXPECT_FLOAT_EQ(C(0, 1), 1.8f);
    EXPECT_FLOAT_EQ(C(0, 2), 2.8f);
    EXPECT_FLOAT_EQ(C(1, 0), 3.8f);
    EXPECT_FLOAT_EQ(C(1, 1), 4.8f);
    EXPECT_FLOAT_EQ(C(1, 2), 5.8f);
  }

  {
    constexpr Matrix<float, 2, 3> A{{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}};
    Matrix<float, 2, 3> B;
    B.SetConstant(1.0f);
    Matrix<float, 2, 3> C;
    C.SetConstant(-0.2f);
    C += A + B;
    EXPECT_FLOAT_EQ(C(0, 0), 1.8f);
    EXPECT_FLOAT_EQ(C(1, 0), 4.8f);
    EXPECT_FLOAT_EQ(C(0, 1), 2.8f);
    EXPECT_FLOAT_EQ(C(1, 1), 5.8f);
    EXPECT_FLOAT_EQ(C(0, 2), 3.8f);
    EXPECT_FLOAT_EQ(C(1, 2), 6.8f);
  }

  {
    constexpr Matrix<double, 2, 3> A(1.0, 4.0, 2.0, 5.0, 3.0, 6.0);
    Matrix<double, 2, 3> C;
    C.SetConstant(-0.2);
    C += A;
    EXPECT_DOUBLE_EQ(C(0, 0), 0.8);
    EXPECT_DOUBLE_EQ(C(0, 1), 1.8);
    EXPECT_DOUBLE_EQ(C(0, 2), 2.8);
    EXPECT_DOUBLE_EQ(C(1, 0), 3.8);
    EXPECT_DOUBLE_EQ(C(1, 1), 4.8);
    EXPECT_DOUBLE_EQ(C(1, 2), 5.8);
  }

  {
    constexpr Matrix<real, 2, 3> A(1_r, 4_r, 2_r, 5_r, 3_r, 6_r);
    Matrix<real, 2, 3> B;
    B.SetConstant(1_r);
    Matrix<real, 2, 3> C;
    C.SetConstant(-0.2_r);
    C += A + B;
    EXPECT_NEAR_EQ(C(0, 0), 1.8_r);
    EXPECT_NEAR_EQ(C(1, 0), 4.8_r);
    EXPECT_NEAR_EQ(C(0, 1), 2.8_r);
    EXPECT_NEAR_EQ(C(1, 1), 5.8_r);
    EXPECT_NEAR_EQ(C(0, 2), 3.8_r);
    EXPECT_NEAR_EQ(C(1, 2), 6.8_r);
  }

  {
    constexpr Matrix<real, 2, 2> A{1_r, 4_r, 2_r, 5_r};
    Matrix<real, 2, 2> C;
    C.SetConstant(-0.3_r);
    C -= 1.1_r * A;
    EXPECT_NEAR_EQ(C(0, 0), -1.4_r);
    EXPECT_NEAR_EQ(C(1, 0), -4.7_r);
    EXPECT_NEAR_EQ(C(0, 1), -2.5_r);
    EXPECT_NEAR_EQ(C(1, 1), -5.8_r);
  }

  {
    constexpr Matrix<real, 2, 2> A{1_r, 4_r, 2_r, 5_r};
    Matrix<real, 2, 2> B;
    B.SetConstant(1_r);
    Matrix<real, 2, 2> C;
    C.SetConstant(-0.3_r);
    C -= 1.1_r * A + 1.3_r * B;
    EXPECT_NEAR_EQ(C(0, 0), -2.7_r);
    EXPECT_NEAR_EQ(C(1, 0), -6.0_r);
    EXPECT_NEAR_EQ(C(0, 1), -3.8_r);
    EXPECT_NEAR_EQ(C(1, 1), -7.1_r);
  }

  {
    constexpr Matrix<real, 2, 2, krylov::Direction::RowMajor> A{{1_r, 2_r}, {4_r, 5_r}};
    Matrix<real, 2, 2, krylov::Direction::RowMajor> C;
    C.SetConstant(-0.123_r);
    C -= 1.1_r * A;
    EXPECT_NEAR_EQ(C(0, 0), -1.223_r);
    EXPECT_NEAR_EQ(C(1, 0), -4.523_r);
    EXPECT_NEAR_EQ(C(0, 1), -2.323_r);
    EXPECT_NEAR_EQ(C(1, 1), -5.623_r);
  }

  {
    constexpr Matrix<real, 2, 2> A{1_r, 4_r, 2_r, 5_r};
    Matrix<real, 2, 2> B;
    B.SetConstant(1_r);
    Matrix<real, 2, 2> C;
    C.SetConstant(-0.123_r);
    C -= 1.1_r * A + -1.3_r * B;
    EXPECT_NEAR_EQ(C(0, 0), 0.077_r);
    EXPECT_NEAR_EQ(C(1, 0), -3.223_r);
    EXPECT_NEAR_EQ(C(0, 1), -1.023_r);
    EXPECT_NEAR_EQ(C(1, 1), -4.323_r);
  }

  {
    Matrix<real, 1, 1> A;
    auto dest = -details::SetDest(details::GetAccessor(A));
    dest.Store(0, 0, 3_r);
    EXPECT_NEAR_EQ(A(0, 0), -3_r);
  }
}

// Expression templates must behave identically whether sub-expressions are consumed inline or
// stored in a named auto variable.
TEST(MatrixExpression, NamedSubExpressions) {
  constexpr int m = 17, n = 19, k = 13;
  Matrix<real> A(m, k), B(k, n);
  A.SetRandom(123);
  B.SetRandom(234);
  real const alpha = 3_r;
  real const tol = 10 * test::GetTol<real>(k);
  Matrix<real> C(m, n);
  Matrix<real> truth(m, n);

  // Scaled sub-expression on the LHS of a product.
  {
    auto scaled = alpha * A;
    C = scaled * B;
    test::MultiplyMatrices(alpha, A, B, truth);
    EXPECT_TRUE(test::IsNear(C, truth, tol));
  }

  // Scaled sub-expression on the RHS of a product (const lvalue).
  {
    auto const scaled = alpha * B;
    C = A * scaled;
    test::MultiplyMatrices(alpha, A, B, truth);
    EXPECT_TRUE(test::IsNear(C, truth, tol));
  }

  // Negated sub-expression on the LHS of a product (const lvalue).
  {
    auto const neg = -A;
    C = neg * B;
    test::MultiplyMatrices(-1_r, A, B, truth);
    EXPECT_TRUE(test::IsNear(C, truth, tol));
  }

  // Negated sub-expression on the RHS of a product (const lvalue).
  {
    auto const neg = -B;
    C = A * neg;
    test::MultiplyMatrices(-1_r, A, B, truth);
    EXPECT_TRUE(test::IsNear(C, truth, tol));
  }

  // Scaled sub-expressions on both sides of a product.
  {
    real const beta = 5_r;
    auto scaledA = alpha * A;
    auto scaledB = beta * B;
    C = scaledA * scaledB;
    test::MultiplyMatrices(alpha * beta, A, B, truth);
    EXPECT_TRUE(test::IsNear(C, truth, tol));
  }

  // Negated sub-expressions on both sides of a product.
  {
    auto negA = -A;
    auto negB = -B;
    C = negA * negB;
    test::MultiplyMatrices(1_r, A, B, truth);
    EXPECT_TRUE(test::IsNear(C, truth, tol));
  }

  // Scaled and negated sub-expressions on both sides of a product.
  {
    auto scaledA = alpha * A;
    auto negB = -B;
    C = scaledA * negB;
    test::MultiplyMatrices(-alpha, A, B, truth);
    EXPECT_TRUE(test::IsNear(C, truth, tol));
  }

  // Sum sub-expression composed into a further product.
  {
    auto sum = A + A;
    C = sum * B;
    test::MultiplyMatrices(2_r, A, B, truth);
    EXPECT_TRUE(test::IsNear(C, truth, tol));
  }
}

TEST(MatrixExpression, LargeProduct) {
  // Unit tests for matrix-matrix products involving 'LargeProduct' specializations. All
  // combinations of storage directions are tested.
  // TODO: Include tests for matrices in which one or several dimensions are compile-time.
  constexpr krylov::Direction col = krylov::Direction::ColMajor;
  constexpr krylov::Direction row = krylov::Direction::RowMajor;
  TestLargeProduct<real, col, col, col>();
  TestLargeProduct<real, row, col, col>();
  TestLargeProduct<real, col, row, col>();
  TestLargeProduct<real, col, col, row>();
  TestLargeProduct<real, row, row, col>();
  TestLargeProduct<real, row, col, row>();
  TestLargeProduct<real, col, row, row>();
  TestLargeProduct<real, row, row, row>();
}

/* Test that the triple product is working as well as unary negation operator.
 The underlying product operation is the same
 as general product, so not all sizes have to be tested to ensure that the assignment is
 accepted and that the correct sub-products are executed. */
TEST(MatrixExpression, TripleMatrixProduct) {
  int m = 133;
  int n = 178;
  int k = 112;
  int l = 59;

  Matrix<float> A(m, n);
  Matrix<float> B(n, k);
  Matrix<float> C(k, l);
  A.SetRandom(m);
  B.SetRandom(n);
  C.SetRandom(k);

  Matrix<float> D;
  D = A * B * C; // Equivalent to (A*B)*C
  Matrix<float> D2;
  D2 = A * (B * C);
  // Reference computation in the same order as D2.
  Matrix<float> E = B * C;
  Matrix<float> R = A * E;
  // In float, the alternative formula may end up with a wider difference.
  EXPECT_TRUE(test::IsNear(D, R, 9 * n * test::GetTol<float>(n + k)));
  EXPECT_TRUE(test::IsNear(D2, R, 9 * test::GetTol<float>(n + k)));
  D = -D2;
  R = -1.0 * D2;
  EXPECT_TRUE(test::IsNear(D, R, 1 * test::GetTol<float>(1)));
}

TEST(MatrixExpression, MatrixScaling) {
  int m = 133;
  int n = 59;
  {
    Matrix<float> A(m, n);
    Matrix<float> B(m, n);
    A.SetRandom(m);
    B.SetRandom(n);
    auto alpha = std::numbers::sqrt2_v<float>;
    for (int ii = 0; ii < m; ++ii) {
      for (int jj = 0; jj < n; ++jj) {
        B(ii, jj) = alpha * A(ii, jj);
      }
    }
    A *= alpha;
    EXPECT_TRUE(test::IsNear(A, B, 2 * test::GetTol<float>(m + n)));
    auto beta = std::numbers::sqrt3_v<float>;
    for (int ii = 0; ii < m; ++ii) {
      for (int jj = 0; jj < n; ++jj) {
        B(ii, jj) = B(ii, jj) / beta;
      }
    }
    A /= beta;
    EXPECT_TRUE(test::IsNear(A, B, 2 * test::GetTol<float>(m + n)));
  }
  {
    Matrix<double> A(m, n);
    Matrix<double> B(m, n);
    A.SetRandom(m);
    B.SetRandom(n);
    double alpha = std::numbers::sqrt3;
    for (int ii = 0; ii < m; ++ii) {
      for (int jj = 0; jj < n; ++jj) {
        B(ii, jj) = alpha * A(ii, jj);
      }
    }
    A *= alpha;
    EXPECT_TRUE(test::IsNear(A, B, 2 * test::GetTol<double>(m + n)));
    double beta = std::numbers::sqrt2;
    for (int ii = 0; ii < m; ++ii) {
      for (int jj = 0; jj < n; ++jj) {
        B(ii, jj) = B(ii, jj) / beta;
      }
    }
    A /= beta;
    EXPECT_TRUE(test::IsNear(A, B, 2 * test::GetTol<double>(m + n)));
  }
}

TEST(MatrixExpression, ViewAssign) {
  // Test of assignment, making sure the view does not attempt a resizing.
  Matrix<float> A(256, 35);
  Matrix<float> B(16, 12);
  A.SetRandom(256);
  B.SetRandom(123);
  auto block = A.Block(133, 6, 16, 12);
  block = B;
  EXPECT_EQ(A(133, 6), B(0, 0));
}
