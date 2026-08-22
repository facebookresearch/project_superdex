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
 * @file ldlt_lu_test.h
 * @brief Shared template helpers for LDLt and LU tests.
 */

#pragma once

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/ldlt.h>
#include <mochi_core/linear_algebra/lu.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/math_utils.h>

#include <gtest/gtest.h>

#if MOCHI_USE_EIGEN
#include <Eigen/Dense>
#endif // MOCHI_USE_EIGEN

#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace mochi::test {

inline constexpr auto kColMajor = krylov::Direction::ColMajor;
inline constexpr auto kRowMajor = krylov::Direction::RowMajor;

enum class InverseMode { Inverse, StableInverse, SymInverse };

/// @brief Returns the set of matrix sizes used by determinant tests.
[[maybe_unused]] inline std::set<int> MakeDeterminantNList() {
  std::set<int> nList{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 64};
  // Cases with matrix size equal or close to a block size multiple.
  constexpr int kDefaultBlockSize = LU<real>::kBlockSz;
  constexpr int kSmallLuBlockSize = SmallLU<real>::kBlockSz;
  constexpr int kSmallLdltBlockSize = SmallLDLt<real>::kBlockSz;
  static_assert(kDefaultBlockSize > 2 && kSmallLuBlockSize > 2 && kSmallLdltBlockSize > 2);
  static_assert(
      LU<real>::kBlockSz == LDLt<real>::kBlockSz, "Please update unit tests to cover all cases");
  for (int k : {1, 2, 3}) {
    nList.insert(k * kDefaultBlockSize - 2);
    nList.insert(k * kDefaultBlockSize - 1);
    nList.insert(k * kDefaultBlockSize);
    nList.insert(k * kDefaultBlockSize + 1);
    nList.insert(k * kDefaultBlockSize + 2);
    nList.insert(k * kSmallLuBlockSize - 2);
    nList.insert(k * kSmallLuBlockSize - 1);
    nList.insert(k * kSmallLuBlockSize);
    nList.insert(k * kSmallLuBlockSize + 1);
    nList.insert(k * kSmallLuBlockSize + 2);
    nList.insert(k * kSmallLdltBlockSize - 2);
    nList.insert(k * kSmallLdltBlockSize - 1);
    nList.insert(k * kSmallLdltBlockSize);
    nList.insert(k * kSmallLdltBlockSize + 1);
    nList.insert(k * kSmallLdltBlockSize + 2);
  }
  return nList;
}

/// @brief Compile-time size determinant tests.
template <krylov::Direction kDirection>
void CompileTimeDeterminantTests() {
  EXPECT_EQ(-3_r, Determinant(Matrix<real, 1, 1, kDirection>{{-3}}));
  EXPECT_EQ(8_r, Determinant(Matrix<real, 2, 2, kDirection>{{4, 2}, {2, 3}}));
  EXPECT_EQ(8_r, Determinant(Matrix<real, 3, 3, kDirection>{{2, 1, 0}, {1, 3, 1}, {0, 1, 2}}));
  EXPECT_NEAR_RTOL(
      245_r,
      Determinant(
          Matrix<real, 4, 4, kDirection>{{5, 1, 0, 2}, {1, 4, 1, 0}, {0, 1, 3, 1}, {2, 0, 1, 6}}),
      4 * 4 * std::numeric_limits<real>::epsilon());
  EXPECT_NEAR_RTOL(
      1012_r,
      Determinant(
          Matrix<real, 5, 5, kDirection>{
              {4, 1, -1, 0, 2},
              {1, 5, 2, -1, 0},
              {-1, 2, 6, 1, -2},
              {0, -1, 1, 3, 1},
              {2, 0, -2, 1, 7}}),
      5 * 5 * std::numeric_limits<real>::epsilon());
}

/// @brief Dynamic-size determinant tests against Eigen, parametrized by a test callback.
///
/// @param testFn Called for each matrix size n with the Mochi matrix A, its Eigen copy M, and the
/// absolute tolerance. The callback should compute determinants via solver-specific paths and
/// compare them against the Eigen reference.
template <krylov::Direction kDirection, typename Ftor>
void DynamicDeterminantTestsVsEigen([[maybe_unused]] Ftor&& testFn) {
#if MOCHI_USE_EIGEN
  for (auto const n : MakeDeterminantNList()) {
    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> M(n, n);
    Matrix<real, krylov::kDynamic, krylov::kDynamic, kDirection> A(n, n);
    A.SetRandom(123);
    for (int ii = 0; ii < n; ++ii) {
      for (int jj = 0; jj <= ii; ++jj) {
        A(jj, ii) = A(ii, jj); // Symmetrize (harmless for LU, required for LDLt).
        M(ii, jj) = A(ii, jj);
        M(jj, ii) = A(jj, ii);
      }
    }

    real const absTol = 2 * n * n * n * std::numeric_limits<real>::epsilon(); // O(n^3) flops
    testFn(A, M, absTol);
  }
#endif // MOCHI_USE_EIGEN
}

template <krylov::Direction kDirection, InverseMode kMode>
void TestInverse() {
  // Parameterized tester for inverse utility functions. It requires the implementation of the
  // matrix-matrix product (validated in other unit tests) to be correct.

  constexpr bool kIsSymmetricSolver = (kMode == InverseMode::SymInverse);

  auto symmetrize = [](auto& A) {
    EXPECT_EQ(A.Rows(), A.Cols());
    int const n = A.Rows();
    for (int ii = 0; ii < n; ++ii) {
      for (int jj = 0; jj < ii; ++jj) {
        A(ii, jj) = A(jj, ii);
      }
    }
  };

  auto computeInverse = [mode = kMode](auto const& A) {
    Matrix<real> invA(A.Rows(), A.Cols());
    if (mode == InverseMode::Inverse) {
      invA = Inverse(A);
    } else if (mode == InverseMode::StableInverse) {
      invA = StableInverse(A);
    } else {
      EXPECT_TRUE(mode == InverseMode::SymInverse);
      invA = SymInverse(A);
    }
    return invA;
  };

  auto checkIsNearIdentity = [](auto const& A) {
    EXPECT_EQ(A.Rows(), A.Cols());
    int const n = A.Rows();
    for (int ii = 0; ii < n; ++ii) {
      for (int jj = 0; jj < n; ++jj) {
        EXPECT_NEAR(
            A(ii, jj),
            ii == jj ? 1_r : 0_r,
            real(15 * n * n) * std::numeric_limits<real>::epsilon());
      }
    }
  };

  // Tests for compile-time size matrices.
  {
    Matrix<real, 1, 1, kDirection> A{{2_r}};
    auto invA = computeInverse(A);
    EXPECT_EQ(invA(0, 0), 0.5_r);
  }

  {
    constexpr int n = 2;
    Matrix<real, n, n, kDirection> A;
    A.SetRandom(n, -1_r, 1_r);
    if (kIsSymmetricSolver) {
      symmetrize(A);
    }
    auto invA = computeInverse(A);
    Matrix<real> AinvA = A * invA;
    checkIsNearIdentity(AinvA);
  }

  {
    constexpr int n = 3;
    Matrix<real, n, n, kDirection> A;
    A.SetRandom(n, -1_r, 1_r);
    if (kIsSymmetricSolver) {
      symmetrize(A);
    }
    auto invA = computeInverse(A);
    Matrix<real> AinvA = A * invA;
    checkIsNearIdentity(AinvA);
  }

  {
    constexpr int n = 4;
    Matrix<real, n, n, kDirection> A;
    A.SetRandom(n, -1_r, 1_r);
    if (kIsSymmetricSolver) {
      symmetrize(A);
    }
    auto invA = computeInverse(A);
    Matrix<real> AinvA = A * invA;
    checkIsNearIdentity(AinvA);
  }

  {
    constexpr int n = 5;
    Matrix<real, n, n, kDirection> A;
    A.SetRandom(n, -1_r, 1_r);
    if (kIsSymmetricSolver) {
      symmetrize(A);
    }
    auto invA = computeInverse(A);
    Matrix<real> AinvA = A * invA;
    checkIsNearIdentity(AinvA);
  }

  // Tests for dynamic matrices.
  std::set<int> nList{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                      31, 32, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65};
  // Cases with matrix size equal or close to a block size multiple.
  constexpr int kDefaultBlockSize = LU<real>::kBlockSz;
  constexpr int kSmallLuBlockSize = SmallLU<real>::kBlockSz;
  constexpr int kSmallLdltBlockSize = SmallLDLt<real>::kBlockSz;
  static_assert(kDefaultBlockSize > 2 && kSmallLuBlockSize > 2 && kSmallLdltBlockSize > 2);
  static_assert(
      LU<real>::kBlockSz == LDLt<real>::kBlockSz, "Please update unit tests to cover all cases");
  for (int k : {1, 2, 3, 10}) {
    nList.insert(k * kDefaultBlockSize - 2);
    nList.insert(k * kDefaultBlockSize - 1);
    nList.insert(k * kDefaultBlockSize);
    nList.insert(k * kDefaultBlockSize + 1);
    nList.insert(k * kDefaultBlockSize + 2);
    nList.insert(k * kSmallLuBlockSize - 2);
    nList.insert(k * kSmallLuBlockSize - 1);
    nList.insert(k * kSmallLuBlockSize);
    nList.insert(k * kSmallLuBlockSize + 1);
    nList.insert(k * kSmallLuBlockSize + 2);
    nList.insert(k * kSmallLdltBlockSize - 2);
    nList.insert(k * kSmallLdltBlockSize - 1);
    nList.insert(k * kSmallLdltBlockSize);
    nList.insert(k * kSmallLdltBlockSize + 1);
    nList.insert(k * kSmallLdltBlockSize + 2);
  }
#if MOCHI_OPTIMIZED // Expensive cases. Only in optimized builds.
  nList.insert(500);
  nList.insert(1000);
#endif
  for (auto const n : nList) {
    // Create diagonally dominant matrix.
    Matrix<real, krylov::kDynamic, krylov::kDynamic, kDirection> A(n, n);
    ColumnVector<real> D(n);
    A.SetRandom(123, -1_r, 1_r);
    D.SetRandom(456, real(n), real(2 * n));
    for (int i = 0; i < n; ++i) {
      A(i, i) = (i % 2 == 0) ? D(i) : -D(i);
      if (kIsSymmetricSolver) {
        for (int j = 0; j < i; ++j) {
          A(i, j) = A(j, i);
        }
      }
    }
    auto invA = computeInverse(A);

    // Check A * invA recovers the identity.
    Matrix<real> C = A * invA;
    double maxError = 0;
    for (int ii = 0; ii < n; ++ii) {
      for (int jj = 0; jj < n; ++jj) {
        maxError =
            Max<double>(maxError, Abs(static_cast<double>(C(ii, jj)) - (jj == ii ? 1.0 : 0.0)));
      }
    }
    EXPECT_LT(maxError, n * n * std::numeric_limits<real>::epsilon());
  }
}

template <krylov::Direction kDirection, typename Ftor>
void TestSolveInPlace(Ftor&& makeSolver, bool isSymmetricSolver) {
  // Parameterized tester for 'LeftSolveInPlace' and 'RightSolveInPlace' methods of a dense solver.

  std::set<int> nList{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,  12,  13,
                      14, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 200, 300};
  // Cases with matrix size equal or close to a block size multiple.
  using SolverType = decltype(makeSolver(Matrix<real>{}));
  static_assert(SolverType::kBlockSz > 2);
  for (int k : {1, 2, 3}) {
    nList.insert(k * SolverType::kBlockSz - 2);
    nList.insert(k * SolverType::kBlockSz - 1);
    nList.insert(k * SolverType::kBlockSz);
    nList.insert(k * SolverType::kBlockSz + 1);
    nList.insert(k * SolverType::kBlockSz + 2);
  }
#if MOCHI_OPTIMIZED // Expensive cases. Only in optimized builds.
  nList.insert(500);
  nList.insert(1000);
#endif
  std::vector<int> mList{1, 2, 3, 4, 7};

  auto test = [&](auto& A, int n) {
    A.SetZero();
    for (int ii = 0; ii < n; ++ii) {
      A(ii, ii) = 2_r;
      if (ii + 1 < n) {
        A(ii, ii + 1) = -1_r;
      }
      if (ii - 1 >= 0) {
        A(ii, ii - 1) = isSymmetricSolver ? -1_r : 1_r;
      }
    }

    // Factorize matrix
    auto mySolver = makeSolver(A);
    for (auto const m : mList) {
      // Compute expected solution
      Matrix<real> XL(n, m);
      XL.SetRandom(123, 0.5_r, 1_r);
      Matrix<real> XR(m, n);
      XR.SetRandom(234, 0.5_r, 1_r);

      // Left solve on same major direction as A
      Matrix<real, krylov::kDynamic, krylov::kDynamic, kDirection> BL1(n, m);
      ColumnVector<real> bL1(n, 1);
      BL1 = A * XL;
      bL1 = BL1.Col(0);
      mySolver.LeftSolveInPlace(BL1);
      mySolver.LeftSolveInPlace(bL1);

      // Left solve on different major direction than A
      Matrix<real, krylov::kDynamic, krylov::kDynamic, ~kDirection> BL2(n, m);
      BL2 = A * XL;
      mySolver.LeftSolveInPlace(BL2);

      // Right solve on same major direction as A
      Matrix<real, krylov::kDynamic, krylov::kDynamic, kDirection> BR1(m, n);
      RowVector<real> bR1(1, n);
      BR1 = XR * A;
      bR1 = BR1.Row(0);
      mySolver.RightSolveInPlace(BR1);
      mySolver.RightSolveInPlace(bR1);

      // Right solve on different major direction than A
      Matrix<real, krylov::kDynamic, krylov::kDynamic, ~kDirection> BR2(m, n);
      BR2 = XR * A;
      mySolver.RightSolveInPlace(BR2);

      // Check results
      int numErrorLeftSolve = 0, numErrorRightSolve = 0;
      for (int ii = 0; ii < n; ++ii) {
        for (int jj = 0; jj < m; ++jj) {
          real const absTolL = n * n * std::numeric_limits<real>::epsilon() *
              std::abs(XL(ii, jj)); // Cond(A) = O(n^2)
          real const absTolR = n * n * std::numeric_limits<real>::epsilon() *
              std::abs(XR(jj, ii)); // Cond(A) = O(n^2)
          if ((std::abs(bL1(ii, 0) - XL(ii, 0)) > absTolL) ||
              (std::abs(BL1(ii, jj) - XL(ii, jj)) > absTolL) ||
              (std::abs(BL2(ii, jj) - XL(ii, jj)) > absTolL)) {
            numErrorLeftSolve += 1;
          }
          if ((std::abs(bR1(0, ii) - XR(0, ii)) > absTolR) ||
              (std::abs(BR1(jj, ii) - XR(jj, ii)) > absTolR) ||
              (std::abs(BR2(jj, ii) - XR(jj, ii)) > absTolR)) {
            numErrorRightSolve += 1;
          }
        }
      }
      EXPECT_EQ(numErrorLeftSolve, 0);
      EXPECT_EQ(numErrorRightSolve, 0);
    }
  };

  for (auto const n : nList) {
    Matrix<real, krylov::kDynamic, krylov::kDynamic, kDirection> A(n, n);
    test(A, n);
  }

  Matrix<real, 8, 8, kDirection> A_8;
  test(A_8, 8);

  Matrix<real, 32, 32, kDirection> A_32;
  test(A_32, 32);

  Matrix<real, 65, 65, kDirection> A_65;
  test(A_65, 65);
}

template <typename Decomp, typename DecompRef>
void TestDecompositionConsistency(
    Decomp const& decomp,
    DecompRef const& decompRef,
    int n,
    real absTol) {
  // Test Determinant().
  {
    auto const detME = decomp.Determinant();
    auto const detRefME = decompRef.Determinant();
    real const det = std::ldexp(detME.mantissa, detME.exponent);
    real const detRef = std::ldexp(detRefME.mantissa, detRefME.exponent);
    if (IsFinite(det) || IsFinite(detRef)) {
      EXPECT_TRUE(IsFinite(det) && IsFinite(detRef));
      EXPECT_NEAR_TOL(det, detRef, absTol);
    }
  }

  // Test ScalarDeterminant().
  {
    // Disable warnings about determinant overflow. They are expected for ScalarDeterminant.
    bool wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
    EnableLogChannel(LogChannel::Warning, false);
    MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, wasWarningEnabled));
    real const det = decomp.ScalarDeterminant();
    real const detRef = decompRef.ScalarDeterminant();
    if (IsFinite(det) || IsFinite(detRef)) {
      EXPECT_TRUE(IsFinite(det) && IsFinite(detRef));
      EXPECT_NEAR_TOL(det, detRef, absTol);
    }
  }

  // Test Inverse().
  {
    Matrix<real> inv(n, n), invRef(n, n);
    decomp.Inverse(inv);
    decompRef.Inverse(invRef);
    EXPECT_TRUE(NearEqualMatrices(inv, invRef, absTol));
  }

  // Test LeftSolveInPlace().
  {
    ColumnVector<real> b(n);
    b.SetRandom(/*seed*/ 123);
    ColumnVector<real> x = b;
    ColumnVector<real> xRef = b;
    decomp.LeftSolveInPlace(x);
    decompRef.LeftSolveInPlace(xRef);
    EXPECT_TRUE(NearEqualMatrices(x, xRef, absTol));
  }

  // Test RightSolveInPlace().
  {
    RowVector<real> b(n);
    b.SetRandom(/*seed*/ 456);
    RowVector<real> x = b;
    RowVector<real> xRef = b;
    decomp.RightSolveInPlace(x);
    decompRef.RightSolveInPlace(xRef);
    EXPECT_TRUE(NearEqualMatrices(x, xRef, absTol));
  }
}

} // namespace mochi::test
