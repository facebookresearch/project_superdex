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
 * @file ldlt_test.cpp
 * @brief Tests LDLt decomposition class and SymInverse free function (implemented via LDLt).
 */

#include "ldlt_lu_test.h"

using namespace mochi;
using namespace mochi::test;

template <int kTrSize>
static void TestTriangularSolves(int n) {
  Matrix<real, kTrSize, kTrSize> L;
  L.SetRandom(6);
  for (int r = 0; r < kTrSize; ++r) {
    for (int c = r; c < kTrSize; ++c) {
      L(r, c) = (r == c) ? 1_r : 0_r;
    }
  }

  MatrixView<real, kTrSize, kTrSize> Lv{L};
  Matrix<real, krylov::kDynamic, kTrSize> P(n, kTrSize);
  MatrixView<real, krylov::kDynamic, kTrSize> Pv{P};
  P.SetRandom(7);
  auto Q = P;
  kernel::RightTriangularUnitTransposeSolve<real, kTrSize>(Lv, Pv);

  auto ref = Q.Norm();
  Q -= P * L.Transpose();
  auto delta = Q.Norm();
  EXPECT_LE(delta, 1e-3 * ref);
}

template <int kTrSize, int kFullSize>
static void TestBlockedTriangularSolves(int n) {
  Matrix<real, kFullSize, kFullSize> L;
  L.SetRandom(8);
  for (int r = 0; r < kFullSize; ++r) {
    for (int c = r; c < kFullSize; ++c) {
      L(r, c) = (r == c) ? 1_r : 0_r;
    }
  }

  MatrixView<real, kFullSize, kFullSize> Lv{L};
  Matrix<real, krylov::kDynamic, kFullSize> P(n, kFullSize);
  MatrixView<real, krylov::kDynamic, kFullSize> Pv{P};
  P.SetRandom(9);
  auto Q = P;
  kernel::ApplyLmtOnRight<real, kTrSize, kFullSize>(Lv, Pv);

  auto ref = Q.Norm();
  Q -= P * L.Transpose();
  auto delta = Q.Norm();
  EXPECT_LE(delta, 1e-3 * ref);
}

template <int kBlockSz>
static void TestBlockFactor() {
  Matrix<real, kBlockSz, kBlockSz> A;
  A.SetRandom(10);
  // Symmetrize
  for (int r = 0; r < kBlockSz; ++r) {
    for (int c = r + 1; c < kBlockSz; ++c) {
      A(r, c) = A(c, r);
    }
  }
  auto B = A;
  int info = 0;
  kernel::FactorBlock<real, kBlockSz>(A, info);
  EXPECT_EQ(info, 0);
  Matrix<real, kBlockSz, kBlockSz> L;
  Matrix<real, kBlockSz, kBlockSz> D;
  for (int r = 0; r < kBlockSz; ++r) {
    for (int c = 0; c < kBlockSz; ++c) {
      D(r, c) = (r == c) ? (1_r / A(r, c)) : 0_r;
      L(r, c) = (r < c) ? 0_r : (r == c ? 1_r : A(r, c));
    }
  }
  Matrix<real, kBlockSz, kBlockSz> LD = L * D;
  auto ref = B.Norm();
  B -= LD * L.Transpose();
  auto delta = B.Norm();
  EXPECT_LE(delta, 1e-3 * ref);
}

/// @brief Tests LDLt::ScalarDeterminant() on symmetric matrices against Eigen.
template <krylov::Direction kDirection>
static void TestDeterminant() {
  CompileTimeDeterminantTests<kDirection>();

  DynamicDeterminantTestsVsEigen<kDirection>([](auto const& A, auto const& M, real absTol) {
    real const detA = Determinant(A);
    real const detM = M.determinant();
    int info = {};
    LDLt<real> ldlt(A, info);
    EXPECT_EQ(0, info);
    real const detLdlt = ldlt.ScalarDeterminant();

    if (IsFinite(detM) || IsFinite(detA) || IsFinite(detLdlt)) {
      // Determinant of large matrices may overflow.
      EXPECT_NEAR_RTOL(detA, detM, absTol);
      EXPECT_NEAR_RTOL(detA, detLdlt, absTol);
    }
  });
}

/// @brief Verify that equilibrated LDLt produces the same results as non-equilibrated.
template <LDLtEquilibration kEquilibration>
static void TestEquilibrationConsistency(MatrixView<real const> A) {
  static_assert(
      kEquilibration != LDLtEquilibration::None, "Only test non-None equilibration strategies");

  int const n = A.Rows();
  real const absTol = 2_r * real(n * n * n) * std::numeric_limits<real>::epsilon();
  int info = {};

  // Equilibrated.
  LDLt<real, kDynamic, kDynamic, kEquilibration> ldlt(A, info);
  EXPECT_EQ(0, info);

  // Reference: non-equilibrated.
  LDLt<real> ldltRef(A, info);
  EXPECT_EQ(0, info);

  TestDecompositionConsistency(ldlt, ldltRef, n, absTol);
}

TEST(LDLt, TriangularSolves) {
  TestTriangularSolves<6>(128);
  TestTriangularSolves<6>(133);
  TestTriangularSolves<6>(135);
  TestTriangularSolves<8>(251);
  TestTriangularSolves<8>(255);
  TestTriangularSolves<8>(256);
}

TEST(LDLt, BlockedTriangularSolves) {
  TestBlockedTriangularSolves<6, 24>(128);
  TestBlockedTriangularSolves<6, 24>(133);
  TestBlockedTriangularSolves<6, 24>(135);
  TestBlockedTriangularSolves<8, 32>(251);
  TestBlockedTriangularSolves<8, 48>(255);
  TestBlockedTriangularSolves<8, 24>(256);
}

TEST(LDLt, BlockFactor) {
  TestBlockFactor<8>();
  TestBlockFactor<48>();
}

TEST(LDLt, Determinant) {
  TestDeterminant<kColMajor>();
  TestDeterminant<kRowMajor>();
}

TEST(LDLt, SymInverse) {
  TestInverse<kColMajor, InverseMode::SymInverse>();
  TestInverse<kRowMajor, InverseMode::SymInverse>();
}

TEST(LDLt, EquilibrationConsistency) {
  static_assert(
      static_cast<int>(LDLtEquilibration::Count) == 2,
      "Please update unit tests when adding LDLt equilibration methods");

  for (auto const n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 24, 32, 48, 63, 64, 65}) {
    // Create symmetric positive definite matrix.
    Matrix<real> Asqrt(n, n);
    Asqrt.SetRandom(/*seed*/ 42 + n);
    Matrix<real> A = Asqrt * Asqrt.Transpose();
    for (int i = 0; i < n; ++i) {
      A(i, i) += real(n);
    }

    // Test non-None equilibration strategies.
    TestEquilibrationConsistency<LDLtEquilibration::Diagonal>(A);
  }
}

TEST(LDLt, SolveInPlace) {
  auto testLDLtEquilibration = []<typename Equilibration>(Equilibration) {
    auto makeLDLt = [](auto&& A) {
      int info = {};
      auto ldlt = LDLt(Equilibration{}, std::forward<decltype(A)>(A), info);
      EXPECT_EQ(0, info);
      return ldlt;
    };
    TestSolveInPlace<kColMajor>(makeLDLt, /*isSymmetricSolver*/ true);
    TestSolveInPlace<kRowMajor>(makeLDLt, /*isSymmetricSolver*/ true);
    auto makeSmallLDLt = [](auto&& A) {
      using MatType = decltype(A);
      int info = {};
      auto smallLdlt = SmallLDLt<
          real,
          krylov::details::MatTraits<MatType>::kNumRows,
          krylov::details::MatTraits<MatType>::kNumCols,
          Equilibration::value>(A, info);
      EXPECT_EQ(0, info);
      return smallLdlt;
    };
    TestSolveInPlace<kColMajor>(makeSmallLDLt, /*isSymmetricSolver*/ true);
    TestSolveInPlace<kRowMajor>(makeSmallLDLt, /*isSymmetricSolver*/ true);
  };

  static_assert(
      static_cast<int>(LDLtEquilibration::Count) == 2,
      "Please update unit tests when adding LDLt equilibration methods");
  testLDLtEquilibration(NoEquilibration);
  testLDLtEquilibration(DiagonalEquilibration);
}

TEST(LDLt, RoundToPowerOfTwo) {
  using details::RoundToPowerOfTwo;

  // Helper: Check if a value is exactly a power of two.
  auto IsPowerOfTwo = [](real x) {
    int exp{};
    real const mantissa = std::frexp(x, &exp);
    return mantissa == 0.5_r; // Powers of two have mantissa exactly 0.5.
  };

  // Helper: Check if result is the closest power of two (in geometric/log sense).
  // The geometric midpoint between 2^(exp-1) and 2^exp is sqrt(2) * 2^(exp-1).
  // Values below midpoint should round down, values at or above should round up.
  auto IsClosestPowerOfTwo = [](real x, real result) {
    int exp{};
    std::frexp(x, &exp);
    real const lower = std::ldexp(1_r, exp - 1);
    real const upper = std::ldexp(1_r, exp);
    // Geometric distance: x/result vs otherCandidate/x (or result/x vs x/otherCandidate).
    real const otherCandidate = (result == lower) ? upper : lower;
    return (x / result <= otherCandidate / x) || (result / x <= x / otherCandidate);
  };

  real constexpr kMin = std::numeric_limits<real>::min();
  real constexpr kMax = std::numeric_limits<real>::max();

  // Test exact powers of two: should return themselves.
  for (real powerOfTwo = kMin; powerOfTwo <= kMax; powerOfTwo *= 2_r) {
    EXPECT_EQ(RoundToPowerOfTwo(powerOfTwo), powerOfTwo);
    if (powerOfTwo > kMax / 2_r) {
      // Safe termination: break before multiplying the maximum exact power of two by 2.0,
      // preventing an FENV exception.
      break;
    }
  }

  // Test values at geometric midpoints.
  for (real lower = kMin; lower < kMax / 2_r; lower *= 2_r) {
    real const upper = lower * 2_r;
    real const midpoint = lower * kSqrt2; // Geometric midpoint.

    // Just below midpoint -> rounds to lower.
    real const belowMid = std::nextafter(midpoint, 0_r);
    EXPECT_EQ(RoundToPowerOfTwo(belowMid), lower);

    // Just above midpoint -> rounds to upper.
    real const aboveMid = std::nextafter(midpoint, upper);
    EXPECT_EQ(RoundToPowerOfTwo(aboveMid), upper);
  }

  // Verify properties hold for sampled mantissa values across the exponent range.
  for (real powerOfTwo = kMin * 2_r;;) {
    for (real mantissa : {0.5_r, 0.6_r, 0.7_r, 0.8_r, 0.9_r, 0.99_r}) {
      real const x = mantissa * powerOfTwo;
      real const result = RoundToPowerOfTwo(x);
      EXPECT_TRUE(IsPowerOfTwo(result));
      EXPECT_TRUE(IsClosestPowerOfTwo(x, result));
    }

    if (powerOfTwo > kMax / 2_r) {
      // Safe termination: break before multiplying the maximum exact power of two by 2.0,
      // preventing an FENV exception.
      break;
    }
    powerOfTwo *= 2_r;
  }
}
