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
 * @file qr_test.cpp
 * @brief Tests ThinQR decomposition (qr.h).
 */

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/qr.h>
#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <limits>

using namespace mochi;
using namespace mochi::test;

template <QrAlgorithm kAlgo, typename MatType>
static void TestThinQR(MatType const& A, bool expectUnitaryQ = true) {
  using Scalar = typename MatType::NonConstScalar;
  constexpr auto kNumRows = krylov::details::MatTraits<MatType>::kNumRows;
  constexpr auto kNumCols = krylov::details::MatTraits<MatType>::kNumCols;
  constexpr bool kIsOwner = IsOwner(krylov::details::MatTraits<MatType>::kOwner);
  auto const absTol =
      Scalar(5e2) * Max(A.Rows(), A.Cols()) * std::numeric_limits<Scalar>::epsilon();

  ThinQR<Scalar, kNumRows, kNumCols, kAlgo> qr(A);
  auto Q = qr.Q();
  auto R = qr.R();

  // Check Q() and R() return views.
  static_assert(IsView(krylov::details::MatTraits<decltype(qr.Q())>::kOwner));
  static_assert(IsView(krylov::details::MatTraits<decltype(qr.R())>::kOwner));

  // Check sizes.
  EXPECT_EQ(A.Rows(), Q.Rows());
  EXPECT_EQ(A.Cols(), Q.Cols());
  EXPECT_EQ(A.Cols(), R.Rows());
  EXPECT_EQ(A.Cols(), R.Cols());

  // Check A = Q * R.
  EXPECT_TRUE(NearEqualMatrices(A, Matrix<Scalar>(Q * R), absTol));

  // Check Q is unitary.
  if (expectUnitaryQ) {
    Matrix<Scalar> I(A.Cols(), A.Cols());
    I.SetIdentity();
    EXPECT_TRUE(NearEqualMatrices(I, Matrix<Scalar>(Q.Transpose() * Q), absTol));
  }

  // Check R is upper triangular.
  for (int i = 0; i < R.Rows(); ++i) {
    for (int j = 0; j < i; ++j) {
      EXPECT_EQ(Scalar(0), R(i, j));
    }
  }

  // Check factorization of a view.
  if constexpr (kIsOwner) {
    TestThinQR<kAlgo>(AsConstView(A), expectUnitaryQ);
  }
}

TEST(ThinQR, ThinQR) {
  // Dynamic matrices.
  for (int nCols :
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
    for (int nRows : {nCols, nCols + 1, nCols + 2, nCols + 3, nCols + 4, nCols + 16}) {
      Matrix<real> A(nRows, nCols);
      A.SetRandom(nRows + nCols);

      TestThinQR<QrAlgorithm::MGS>(A);
      TestThinQR<QrAlgorithm::MGS>(RowMatrix<real>(A));

      // Classical Gram-Schmidt is unstable. Check unitary Q only with fewer than 32 columns.
      TestThinQR<QrAlgorithm::CGS>(A, /*expectUnitaryQ*/ (nCols < 32));
      TestThinQR<QrAlgorithm::CGS>(RowMatrix<real>(A), /*expectUnitaryQ*/ (nCols < 32));

      TestThinQR<QrAlgorithm::ICGS>(A);
      TestThinQR<QrAlgorithm::ICGS>(RowMatrix<real>(A));
    }
  }

  // Fixed-size matrices.
  {
    constexpr int kCols = 10;
    constexpr int kRows = kCols + 1;
    Matrix<real> A(kRows, kCols);
    A.SetRandom(kRows + kCols);

    TestThinQR<QrAlgorithm::MGS>(Matrix<real, kRows, kCols>(A));
    TestThinQR<QrAlgorithm::MGS>(RowMatrix<real, kRows, kCols>(A));
    TestThinQR<QrAlgorithm::MGS>(Matrix<real, krylov::kDynamic, kCols>(A));
    TestThinQR<QrAlgorithm::MGS>(RowMatrix<real, krylov::kDynamic, kCols>(A));
    TestThinQR<QrAlgorithm::MGS>(Matrix<real, kRows, krylov::kDynamic>(A));
    TestThinQR<QrAlgorithm::MGS>(RowMatrix<real, kRows, krylov::kDynamic>(A));

    TestThinQR<QrAlgorithm::CGS>(Matrix<real, kRows, kCols>(A));
    TestThinQR<QrAlgorithm::CGS>(RowMatrix<real, kRows, kCols>(A));
    TestThinQR<QrAlgorithm::CGS>(Matrix<real, krylov::kDynamic, kCols>(A));
    TestThinQR<QrAlgorithm::CGS>(RowMatrix<real, krylov::kDynamic, kCols>(A));
    TestThinQR<QrAlgorithm::CGS>(Matrix<real, kRows, krylov::kDynamic>(A));
    TestThinQR<QrAlgorithm::CGS>(RowMatrix<real, kRows, krylov::kDynamic>(A));

    TestThinQR<QrAlgorithm::ICGS>(Matrix<real, kRows, kCols>(A));
    TestThinQR<QrAlgorithm::ICGS>(RowMatrix<real, kRows, kCols>(A));
    TestThinQR<QrAlgorithm::ICGS>(Matrix<real, krylov::kDynamic, kCols>(A));
    TestThinQR<QrAlgorithm::ICGS>(RowMatrix<real, krylov::kDynamic, kCols>(A));
    TestThinQR<QrAlgorithm::ICGS>(Matrix<real, kRows, krylov::kDynamic>(A));
    TestThinQR<QrAlgorithm::ICGS>(RowMatrix<real, kRows, krylov::kDynamic>(A));
  }

  // Rank-deficient cases.
  {
    int const nCols = 16;
    for (int nRows : {nCols, nCols + 1, nCols + 2, nCols + 3, nCols + 4, nCols + 16}) {
      Matrix<real> A(nRows, nCols);
      A.SetRandom(nRows + nCols);
      A.LeftCols(nCols / 2) = A.RightCols(nCols / 2);

      // Expect unitary Q only with ICGS.
      TestThinQR<QrAlgorithm::MGS>(A, /*expectUnitaryQ*/ false);
      TestThinQR<QrAlgorithm::MGS>(RowMatrix<real>(A), /*expectUnitaryQ*/ false);

      TestThinQR<QrAlgorithm::CGS>(A, /*expectUnitaryQ*/ false);
      TestThinQR<QrAlgorithm::CGS>(RowMatrix<real>(A), /*expectUnitaryQ*/ false);

      TestThinQR<QrAlgorithm::ICGS>(A);
      TestThinQR<QrAlgorithm::ICGS>(RowMatrix<real>(A));

      // Test with 2 zero columns. Do not expect unitary Q with any of MGS, CGS and ICGS.
      A.Col(0).SetZero();
      A.Col(nCols / 2).SetZero();

      TestThinQR<QrAlgorithm::MGS>(A, /*expectUnitaryQ*/ false);
      TestThinQR<QrAlgorithm::MGS>(RowMatrix<real>(A), /*expectUnitaryQ*/ false);

      TestThinQR<QrAlgorithm::CGS>(A, /*expectUnitaryQ*/ false);
      TestThinQR<QrAlgorithm::CGS>(RowMatrix<real>(A), /*expectUnitaryQ*/ false);

      TestThinQR<QrAlgorithm::ICGS>(A, /*expectUnitaryQ*/ false);
      TestThinQR<QrAlgorithm::ICGS>(RowMatrix<real>(A), /*expectUnitaryQ*/ false);
    }
  }

  static_assert(
      static_cast<int>(QrAlgorithm::Count) == 3,
      "Please update unit tests to cover all algorithms");
}
