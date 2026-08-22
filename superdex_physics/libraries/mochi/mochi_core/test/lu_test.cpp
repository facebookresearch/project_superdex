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
 * @file lu_test.cpp
 * @brief Tests LU decomposition class and Inverse, StableInverse, and Determinant free functions
 * (implemented via LU).
 */

#include "ldlt_lu_test.h"

using namespace mochi;
using namespace mochi::test;

template <krylov::Direction kDirR, krylov::Direction kDirX>
static void TestBackSubstitutionInPlace(int n, int nCols) {
  real const relTol = 100 * Pow(n, 2) * std::numeric_limits<real>::epsilon();
  Matrix<real, krylov::kDynamic, krylov::kDynamic, kDirR> R(n, n);
  Matrix<real> X0(n, nCols);
  R.SetRandom(n);
  X0.SetRandom(2 * n);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < r; ++c) {
      R(r, c) = 0_r;
    }
  }

  Matrix<real, krylov::kDynamic, krylov::kDynamic, kDirX> X = X0;
  kernel::BackSubstitutionInPlace(R, X);
  EXPECT_LE(Matrix<real>(R * X - X0).Norm(), relTol * X0.Norm());
}

/// @brief Tests the Determinant() free function and LU::ScalarDeterminant() against Eigen.
template <krylov::Direction kDirection>
static void TestDeterminant() {
  CompileTimeDeterminantTests<kDirection>();

  DynamicDeterminantTestsVsEigen<kDirection>([](auto const& A, auto const& M, real absTol) {
    real const detA = Determinant(A);
    real const detM = M.determinant();
    LU<real> lu(A);
    real const detLu = lu.ScalarDeterminant();

    if (IsFinite(detM) || IsFinite(detA) || IsFinite(detLu)) {
      // Determinant of large matrices may overflow.
      EXPECT_NEAR_RTOL(detA, detM, absTol);
      EXPECT_NEAR_RTOL(detA, detLu, absTol);
    }
  });
}

/// @brief Verify that LU with pivoting produces the same results as LU without pivoting.
template <PermuteAlg kPermAlg>
static void TestPivotingConsistency(MatrixView<real const> A) {
  static_assert(
      kPermAlg != PermuteAlg::None, "Only test non-None pivoting strategies against None");

  int const n = A.Rows();
  real const absTol = 2_r * real(n * n * n) * std::numeric_limits<real>::epsilon();

  // With pivoting.
  LU<real, kDynamic, kDynamic, kPermAlg> lu(A);

  // Reference: no pivoting.
  LU<real, kDynamic, kDynamic, PermuteAlg::None> luRef(A);

  TestDecompositionConsistency(lu, luRef, n, absTol);
}

TEST(LU, BackSubstitutionInPlace) {
  for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17}) {
    for (int nCols : {1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17}) {
      TestBackSubstitutionInPlace<kColMajor, kColMajor>(n, nCols);
      TestBackSubstitutionInPlace<kRowMajor, kColMajor>(n, nCols);
      TestBackSubstitutionInPlace<kColMajor, kRowMajor>(n, nCols);
      TestBackSubstitutionInPlace<kRowMajor, kRowMajor>(n, nCols);
    }
  }
}

TEST(LU, Determinant) {
  TestDeterminant<kColMajor>();
  TestDeterminant<kRowMajor>();
}

TEST(LU, Inverse) {
  TestInverse<kColMajor, InverseMode::Inverse>();
  TestInverse<kRowMajor, InverseMode::Inverse>();
  TestInverse<kColMajor, InverseMode::StableInverse>();
  TestInverse<kRowMajor, InverseMode::StableInverse>();
}

TEST(LU, PivotingConsistency) {
  static_assert(
      static_cast<int>(PermuteAlg::Count) == 3,
      "Please update unit tests when adding pivoting algorithms");

  for (auto const n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 24, 32, 48, 63, 64, 65}) {
    // Create a random invertible matrix (diagonal dominance ensures invertibility without
    // pivoting).
    Matrix<real> A(n, n);
    A.SetRandom(/*seed*/ 42 + n);
    for (int i = 0; i < n; ++i) {
      A(i, i) += real(n);
    }

    // Test non-None pivoting strategies.
    TestPivotingConsistency<PermuteAlg::PartialRow>(A);
    TestPivotingConsistency<PermuteAlg::Rook>(A);
  }
}

TEST(LU, SolveInPlace) {
  auto testLUPivoting = []<typename Permutation>(Permutation) {
    auto makeLU = [](auto&& A) { return LU(Permutation{}, std::forward<decltype(A)>(A)); };
    TestSolveInPlace<kColMajor>(makeLU, /*isSymmetricSolver*/ false);
    TestSolveInPlace<kRowMajor>(makeLU, /*isSymmetricSolver*/ false);

    auto makeSmallLU = [](auto&& A) {
      using MatType = decltype(A);
      return SmallLU<
          real,
          krylov::details::MatTraits<MatType>::kNumRows,
          krylov::details::MatTraits<MatType>::kNumCols,
          Permutation::value>(A);
    };
    TestSolveInPlace<kColMajor>(makeSmallLU, /*isSymmetricSolver*/ false);
    TestSolveInPlace<kRowMajor>(makeSmallLU, /*isSymmetricSolver*/ false);
  };

  static_assert(
      static_cast<int>(PermuteAlg::Count) == 3,
      "Please update unit tests when adding permutation algorithms");
  testLUPivoting(NoPermutation);
  testLUPivoting(PartialPermutation);
  testLUPivoting(RookPermutation);
}
