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

#include <mochi_core/test/batch_helpers.h>
#include <mochi_core/test/decomposition_utils_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <array>
#include <limits>

using namespace mochi;
using namespace mochi::test;

static constexpr auto kScalings =
    kPI * NdArray<real, 9>{1e-12_r, 1e-9_r, 1e-6_r, 1e-3_r, 1e0_r, 1e3_r, 1e6_r, 1e9_r, 1e12_r};
static real const eigenRelTol = 10_r * Sqrt(std::numeric_limits<real>::epsilon());
static real const svdRelTol = 50_r * Sqrt(std::numeric_limits<real>::epsilon());
static constexpr real kPsdRelTol = 2e2_r * std::numeric_limits<real>::epsilon();
static constexpr real kPsdAbsTol = kPsdRelTol;
static constexpr real kCholeskyRelTol = 1e2_r * std::numeric_limits<real>::epsilon();

// clang-format off
  static constexpr Matrix3x3r kTestMatrices3x3[] = {
    // Problematic?
    Matrix3x3r{
      Real3{ 11.9742670_r,   8.36378384_r, -8.73394966_r},
      Real3{-14.0197401_r,  -5.03967428_r,  5.76973772_r},
      Real3{ 0.152402133_r, -3.11131549_r,  4.15363169_r}},
    // Problematic?
    Matrix3x3r{
      Real3{ 0.962555885_r,   0.271199137_r, -0.0117054982_r},
      Real3{-0.266829342_r,   0.953240335_r,  0.143183127_r},
      Real3{ 0.0498688221_r, -0.134830400_r,  0.989717662_r}},
    // Scaled identity
    Matrix3x3r{
        Real3{1_r, 0_r, 0_r},
        Real3{0_r, 1_r, 0_r},
        Real3{0_r, 0_r, 1_r}},
    Matrix3x3r{
        Real3{-4_r,  0_r,  0_r},
        Real3{ 0_r, -4_r,  0_r},
        Real3{ 0_r,  0_r, -4_r}},
    Matrix3x3r{
        Real3{0.25_r, 0_r,    0_r},
        Real3{0_r,    0.25_r, 0_r},
        Real3{0_r,    0_r,    0.25_r}},
    // Diagonal with non-monotonic and non-equal diagonal entries
    Matrix3x3r{
        Real3{-1_r, 0_r, 0_r},
        Real3{ 0_r, 2_r, 0_r},
        Real3{ 0_r, 0_r, 3_r}},
    Matrix3x3r{
        Real3{1_r, 0_r,  0_r},
        Real3{0_r, 2_r,  0_r},
        Real3{0_r, 0_r, -0.5_r}},
    Matrix3x3r{
        Real3{2_r,  0_r, 0_r},
        Real3{0_r, -2_r, 0_r},
        Real3{0_r,  0_r, 2_r}},
    // Random
    Matrix3x3r{
        Real3{1.757904e+00_r, 8.375675e-01_r, 8.940403e-01_r},
        Real3{7.781580e-01_r, 1.293040e+00_r, 8.082672e-01_r},
        Real3{9.259674e-01_r, 3.419398e-01_r, 1.178375e+00_r}},
    // Isochoric
    Matrix3x3r{
        Real3{1.852487e+00_r, 3.954069e-01_r, 8.651419e-01_r},
        Real3{1.217431e-01_r, 1.074060e+00_r, 2.034998e-01_r},
        Real3{1.669658e-01_r, 6.375304e-01_r, 1.053620e+00_r}},
    // Symmetric
    Matrix3x3r{
        Real3{3.474406e+00_r, 9.696913e-01_r, 1.803357e+00_r},
        Real3{9.696913e-01_r, 1.716397e+00_r, 1.232369e+00_r},
        Real3{1.803357e+00_r, 1.232369e+00_r, 1.899997e+00_r}},
    // Shear
    Matrix3x3r{
        Real3{1_r,   0.5_r, 0_r},
        Real3{0.5_r, 1_r,   0_r},
        Real3{0_r,   0_r,   1_r}},
    // Near-identity (clustered singular values).
    Matrix3x3r{
        Real3{1.0001_r, 1e-4_r,  0_r},
        Real3{0_r,       0.9999_r, 0_r},
        Real3{0_r,       0_r,      1_r}},
    Matrix3x3r{
        Real3{ 1_r,    0_r, 1e-4_r},
        Real3{-1e-4_r, 1_r, 0_r},
        Real3{ 0_r,    0_r, 1.0001_r}},
    Matrix3x3r{
        Real3{1.0002_r, 0_r, 0_r},
        Real3{0_r,       1_r, 0_r},
        Real3{0_r,       0_r, 0.9998_r}},
    // Rank-2 cases
    Matrix3x3r{
        Real3{2_r, 0_r, 0_r},
        Real3{0_r, 0_r, 0_r},
        Real3{0_r, 0_r, 2_r}},
    Matrix3x3r{
        Real3{2_r, 0_r,  0_r},
        Real3{0_r, 0_r, -1_r},
        Real3{0_r, 0_r,  0_r}},
    Matrix3x3r{
        Real3{0_r, 1_r,  0_r},
        Real3{0_r, 0_r, -1_r},
        Real3{0_r, 0_r,  0_r}},
    Matrix3x3r{
        Real3{1_r, -1_r,  0_r},
        Real3{0_r, -1_r, -1_r},
        Real3{2_r, -1_r,  1_r}},
    Matrix3x3r{
        Real3{1_r,  2_r,  3_r},
        Real3{3_r,  3_r,  5_r},
        Real3{1_r, -1_r, -1_r}},
    // Rank-1 cases
    Matrix3x3r{
        Real3{0_r, 0_r, 0_r},
        Real3{0_r, 0_r, 1_r},
        Real3{0_r, 0_r, 3_r}},
    Matrix3x3r{
        Real3{0_r, 0_r, 0_r},
        Real3{0_r, 0_r, 2_r},
        Real3{0_r, 0_r, 0_r}},
    Matrix3x3r{
        Real3{0_r, 0_r, -1_r},
        Real3{0_r, 0_r,  0_r},
        Real3{0_r, 0_r,  0_r}},
    Matrix3x3r{
        Real3{0_r,  0_r,   0_r},
        Real3{0_r, -0.5_r, 0_r},
        Real3{0_r,  1_r,   0_r}},
    Matrix3x3r{
        Real3{1_r, 0_r, 0_r},
        Real3{3_r, 0_r, 0_r},
        Real3{0_r, 0_r, 0_r}},
    Matrix3x3r{
        Real3{2_r, 0_r, 1_r},
        Real3{4_r, 0_r, 2_r},
        Real3{2_r, 0_r, 1_r}},
    Matrix3x3r{
        Real3{ 1_r,    1_r,    1_r},
        Real3{ 0_r,    0_r,    0_r},
        Real3{-2.5_r, -2.5_r, -2.5_r}},
    // Rank-0 cases
    Matrix3x3r{
        Real3{0_r, 0_r, 0_r},
        Real3{0_r, 0_r, 0_r},
        Real3{0_r, 0_r, 0_r}},
    // Large entries → validates that the fused-SVD path's internal normalization remains
    // accurate for large inputs. Excluded from split-SVD tests since those paths do not
    // normalize G = FᵀF and lose accuracy in single precision.
    Matrix3x3r{
        Real3{1e5_r, 2e4_r, 3e4_r},
        Real3{4e4_r, 5e5_r, 6e4_r},
        Real3{7e4_r, 8e4_r, 9e5_r}},
};

static constexpr Matrix2x2r kTestMatricesSym2x2[] = {
  // Full-rank
  Matrix2x2r{Real2{1.0_r, 3.0_r}, Real2{3.0_r, 2.0_r}},
  // Rank incomplete.
  Matrix2x2r{Real2{1.0_r, 2.0_r}, Real2{2.0_r, 4.0_r}},
  // Diagonal (a < b → sort triggers swap).
  Matrix2x2r{Real2{1.0_r, 0.0_r}, Real2{0.0_r, 2.0_r}},
  // Diagonal (a > b → sort does not swap).
  Matrix2x2r{Real2{2.0_r, 0.0_r}, Real2{0.0_r, 1.0_r}},
  // Diagonal (a == b → degenerate eigenvalues, sort is a no-op).
  Matrix2x2r{Real2{1.0_r, 0.0_r}, Real2{0.0_r, 1.0_r}},
};

static constexpr Matrix3x3r kTestMatricesSym3x3[] = {
  // Full-rank
  Matrix3x3r{Real3{1.0_r, 4.0_r, 5.0_r}, Real3{4.0_r, 2.0_r, 6.0_r}, Real3{5.0_r, 6.0_r, 3.0_r}},
  // Rank incomplete.
  Matrix3x3r{Real3{1.0_r, 2.0_r, 3.0_r}, Real3{2.0_r, 4.0_r, 6.0_r}, Real3{3.0_r, 6.0_r, 9.0_r}},
  // Tri-diagonal.
  Matrix3x3r{Real3{1.0_r, 4.0_r, 0.0_r}, Real3{4.0_r, 2.0_r, 6.0_r}, Real3{0.0_r, 6.0_r, 3.0_r}},
  // Diagonal (a < b < c → sort triggers all swaps).
  Matrix3x3r{Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 2.0_r, 0.0_r}, Real3{0.0_r, 0.0_r, 3.0_r}},
  // Diagonal (non-monotonic → sort triggers a partial swap: swaps 2 & 3, not 1).
  Matrix3x3r{Real3{2.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}, Real3{0.0_r, 0.0_r, 3.0_r}},
  // Edge case with x2 = 0.
  Matrix3x3r{Real3{0_r,                  0.3333333333333333_r, 0_r},
             Real3{0.3333333333333333_r, 0_r,                  0_r},
             Real3{0_r,                  0_r,                  1_r}},
  // Identity.
  Matrix3x3r{Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 1_r}},
  // Near-identity (clustered eigenvalues).
  Matrix3x3r{Real3{1.0001_r, 1e-4_r, 0_r}, Real3{1e-4_r, 0.9999_r, 0_r}, Real3{0_r, 0_r, 1_r}},
  Matrix3x3r{Real3{1_r, 0_r, 1e-4_r}, Real3{0_r, 1.0001_r, 1e-4_r}, Real3{1e-4_r, 1e-4_r, 0.9999_r}},
  Matrix3x3r{Real3{1.0002_r, 0_r, 0_r}, Real3{0_r, 1_r, 0_r}, Real3{0_r, 0_r, 0.9998_r}},
  // Zero.
  Matrix3x3r{Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}},
  // Negative-definite.
  Matrix3x3r{Real3{-3_r, -0.5_r, -0.1_r}, Real3{-0.5_r, -2_r, -0.3_r}, Real3{-0.1_r, -0.3_r, -1_r}},
};

// Adversarial symmetric matrices targeting distinct eigenvalue codepaths for mixed-lane batched tests.
static constexpr Matrix3x3r kMixedLaneMatsSym3x3[] = {
  // Diagonal (non-monotonic) → sort within diagonal override.
  Matrix3x3r{Real3{1_r, 0_r, 0_r}, Real3{0_r, 3_r, 0_r}, Real3{0_r, 0_r, 2_r}},
  // Large entries → overflow fallback.
  Matrix3x3r{
    Real3{1e5_r, 1e4_r, 1e4_r},
    Real3{1e4_r, 2e5_r, 1e4_r},
    Real3{1e4_r, 1e4_r, 3e5_r}},
  // x₂ = 0 exactly → x2NearZero fallback.
  Matrix3x3r{
    Real3{0_r,                  0.3333333333333333_r, 0_r},
    Real3{0.3333333333333333_r, 0_r,                  0_r},
    Real3{0_r,                  0_r,                  1_r}},
  // Cluster at bottom (eigenvalues ≈ 2, 1, 1 after sort) → no flip.
  Matrix3x3r{
    Real3{1.0001_r, 1e-4_r,   0_r},
    Real3{1e-4_r,   0.9999_r, 0_r},
    Real3{0_r,      0_r,      2_r}},
  // Cluster at top (eigenvalues ≈ 2.01, 1.99, 1) → flip.
  Matrix3x3r{
    Real3{2_r,    0.01_r, 0_r},
    Real3{0.01_r, 2_r,    0_r},
    Real3{0_r,    0_r,    1_r}},
  // x₂ < 0 → psi sign correction.
  Matrix3x3r{
    Real3{4_r,   0.1_r, 0.1_r},
    Real3{0.1_r, 1_r,   0.1_r},
    Real3{0.1_r, 0.1_r, 1_r}},
};

// Adversarial matrices targeting distinct SVD codepaths for mixed-lane batched tests.
static constexpr Matrix3x3r kMixedLaneMats3x3[] = {
  // Rank-0 → u0zero + u1zero, diagonal mask.
  Matrix3x3r{Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}},
  // Rank-1 → u1zero fallback.
  Matrix3x3r{Real3{1_r, 0_r, 0_r}, Real3{3_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}},
  // Negative determinant → Sg[2] sign flip, diagonal G → diagonal mask.
  Matrix3x3r{Real3{-1_r, 0_r, 0_r}, Real3{0_r, 2_r, 0_r}, Real3{0_r, 0_r, 3_r}},
  // Near-identity → clustered singular values, flip.
  Matrix3x3r{
    Real3{1.0001_r, 1e-5_r,    0_r},
    Real3{0_r,      0.99999_r, 0_r},
    Real3{0_r,      0_r,       1_r}},
};

// Adversarial symmetric 2x2 matrices for metric-aware batched PSD projection.
static constexpr NdArray<real, 4> kMixedLaneMatsPsd2x2[] = {
  {2_r, 0.5_r, 0.5_r, 3_r},     // Already PSD.
  {-5_r, -1_r, -1_r, -8_r},     // Negative-definite.
  {1_r, 0_r, 0_r, -1_r},        // Indefinite (eigenvalues +1, -1) → partial clamp.
  {0_r, 0_r, 0_r, 0_r},         // Zero.
  {100_r, 0_r, 0_r, 0.01_r},    // Ill-conditioned diagonal.
};

// Adversarial symmetric 3x3 matrices for batched PSD projection: different lanes require different
// projection behavior.
static constexpr Matrix3x3r kMixedLaneMatsPsd3x3[] = {
  // Already PSD → no projection needed.
  Matrix3x3r{Real3{2_r, 0.5_r, 0_r}, Real3{0.5_r, 3_r, 0.1_r}, Real3{0_r, 0.1_r, 1_r}},
  // Negative-definite → all eigenvalues clamped.
  Matrix3x3r{Real3{-5_r, -1_r, 0_r}, Real3{-1_r, -8_r, -0.5_r}, Real3{0_r, -0.5_r, -3_r}},
  // Mixed eigenvalues (some positive, some negative).
  Matrix3x3r{Real3{1_r, 4_r, 5_r}, Real3{4_r, 2_r, 6_r}, Real3{5_r, 6_r, 3_r}},
  // Zero.
  Matrix3x3r{Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}},
  // Ill-conditioned diagonal.
  Matrix3x3r{Real3{100_r, 0_r, 0_r}, Real3{0_r, 0.01_r, 0_r}, Real3{0_r, 0_r, -0.001_r}},
};

// SPD metrics for metric-aware PSD projection tests.
static constexpr NdArray<real, 4> kTestMetrics2x2[] = {
    {1_r, 0_r, 0_r, 1_r},               // Identity.
    {2_r, 0.5_r, 0.5_r, 1_r},           // Well-conditioned SPD.
    {4_r, -1_r, -1_r, 0.5_r},           // Moderate condition number.
    {1_r, 0.99_r, 0.99_r, 1_r},         // Near-singular SPD (high condition number).
};
// clang-format on

static constexpr int kNumMats3x3 = isize(kTestMatrices3x3);
// Excludes the large-entry matrix at the end of kTestMatrices3x3. Use for non-normalizing SVD paths
// (split vals/vecs).
static constexpr int kNumMatsNormalized3x3 = kNumMats3x3 - 1;
static_assert(
    NormSqr(kTestMatrices3x3[kNumMatsNormalized3x3]) > 1e10_r,
    "Large-entry matrix must remain last. Please update kNumMatsNormalized3x3.");
static constexpr int kNumMatsSym3x3 = isize(kTestMatricesSym3x3);
static constexpr int kNumMixedLaneMatsSym3x3 = isize(kMixedLaneMatsSym3x3);
static constexpr int kNumMixedLaneMats3x3 = isize(kMixedLaneMats3x3);
static constexpr int kNumMixedLaneMatsPsd2x2 = isize(kMixedLaneMatsPsd2x2);
static constexpr int kNumMixedLaneMatsPsd3x3 = isize(kMixedLaneMatsPsd3x3);
static constexpr int kNumTestMetrics2x2 = isize(kTestMetrics2x2);

// Mixed-lane arrays must not exceed the max batch size in MOCHI_BATCH_TEST to ensure all entries
// are tested with at least one batch size.
static_assert(kNumMixedLaneMatsSym3x3 <= kBatchTestMaxSize);
static_assert(kNumMixedLaneMats3x3 <= kBatchTestMaxSize);
static_assert(kNumMixedLaneMatsPsd2x2 <= kBatchTestMaxSize);
static_assert(kNumMixedLaneMatsPsd3x3 <= kBatchTestMaxSize);
static_assert(kNumTestMetrics2x2 <= kBatchTestMaxSize);

/**************************************************************************************************
  Verification Helpers
*/

// Verify 2x2 eigendecomposition. Eigenvectors are stored as rows of QT (i.e., QT = Qᵀ).
static void
VerifyEigendecomp2x2(Matrix2x2r const& mat, Real2 const& eigvals, Matrix2x2r const& QT) {
  EXPECT_GE(eigvals[0], eigvals[1]);
  Matrix2x2r const reconstructed = Dot(Transpose(QT), Dot(DiagonalMatrix(eigvals), QT));
  EXPECT_LE(Norm(reconstructed - mat), eigenRelTol * Norm(mat));
  EXPECT_LE(Norm(Dot(QT, Transpose(QT)) - Eye<2>()), eigenRelTol);
}

// Verify 3x3 eigendecomposition. Eigenvectors are stored as rows of QT (i.e., QT = Qᵀ).
static void
VerifyEigendecomp3x3(Matrix3x3r const& mat, Real3 const& eigvals, Matrix3x3r const& QT) {
  EXPECT_GE(eigvals[0], eigvals[1]);
  EXPECT_GE(eigvals[1], eigvals[2]);
  Matrix3x3r const reconstructed = Dot(Transpose(QT), Dot(DiagonalMatrix(eigvals), QT));
  EXPECT_LE(Norm(reconstructed - mat), eigenRelTol * Norm(mat));
  EXPECT_LE(Norm(Dot(QT, Transpose(QT)) - Eye<3>()), eigenRelTol);
  EXPECT_NEAR(Det(QT), 1_r, eigenRelTol);
}

// Verify SVD result for a single matrix.
static void
VerifySvd3x3(Matrix3x3r const& mat, Matrix3x3r const& U, Real3 const& sigma, Matrix3x3r const& VT) {
  EXPECT_GE(Abs(sigma[0]), Abs(sigma[1]));
  EXPECT_GE(Abs(sigma[1]), Abs(sigma[2]));
  EXPECT_GE(Det(U), 0.0_r);
  EXPECT_GE(Det(VT), 0.0_r);
  Matrix3x3r const reconstructed = Dot(U, Dot(DiagonalMatrix(sigma), VT));
  // NOTE: The absolute floor (svdRelTol) handles near-zero matrices at extreme scalings where the
  // relative term vanishes but reconstruction error remains O(eps × singular values).
  real const absTol = svdRelTol;
  EXPECT_LE(Norm(reconstructed - mat), svdRelTol * Norm(mat) + absTol);
  // Orthogonality check: UᵀU ≈ I and VVᵀ ≈ I. For rank-deficient matrices (|σ₂| ≈ 0), the
  // null-space columns of U/V are not uniquely determined, so orthogonality may not hold.
  // TODO: Enable unconditionally once the SVD re-orthogonalizes U for rank-deficient matrices.
  if (Abs(sigma[2]) > svdRelTol * Abs(sigma[0])) {
    EXPECT_LE(Norm(Dot(Transpose(U), U) - Eye<3>()), svdRelTol);
    EXPECT_LE(Norm(Dot(VT, Transpose(VT)) - Eye<3>()), svdRelTol);
  }
  real const matNorm = Norm(mat);
  if (matNorm > 0_r) {
    real const detF = Det(mat);
    real const detTol = svdRelTol * matNorm * matNorm * matNorm;
    if (detF < -detTol) {
      EXPECT_LT(sigma[2], 0.0_r);
    } else if (detF > detTol) {
      EXPECT_GE(sigma[2], 0.0_r);
    }
  }
}

static void VerifyPsdWithMetric(
    Matrix2x2r const& S,
    Matrix2x2r const& M,
    real epsCholesky,
    real epsEigenvalue) {
  VMatrix2x2r const L = CholeskySym2x2(ToSimdMatrix(M), epsCholesky);
  Matrix2x2r const LNd = ToNdArray2x2(L);
  real const l00 = LNd[0][0];
  real const l10 = LNd[1][0];
  real const l11 = LNd[1][1];
  Matrix2x2r const Linv{Real2{1_r / l00, 0_r}, Real2{-l10 / (l00 * l11), 1_r / l11}};
  Matrix2x2r const B = Dot(Linv, Dot(S, Transpose(Linv)));

  Real2 eigvals;
  AnalyticalEigendecompSym(B, eigvals);
  EXPECT_GE(eigvals[0], -epsEigenvalue);
  EXPECT_GE(eigvals[1], -epsEigenvalue);
}

/**************************************************************************************************
  Eigendecomposition: 2x2
*/

TEST(DecompositionUtils, AnalyticalEigendecompSym2x2_Scalar) {
  for (real scaling : kScalings) {
    for (auto const& mat0 : kTestMatricesSym2x2) {
      auto const mat = mat0 * scaling;
      Real2 eigvals;
      Matrix2x2r QT;
      AnalyticalEigendecompSym(mat, eigvals, &QT, /*transpose*/ true);
      VerifyEigendecomp2x2(mat, eigvals, QT);
    }
  }
}

TEST(DecompositionUtils, AnalyticalEigendecompSym2x2_SIMD) {
  for (real scaling : kScalings) {
    for (auto const& mat0 : kTestMatricesSym2x2) {
      auto const mat = mat0 * scaling;
      Vec4r vEigvals;
      VMatrix2x2r vEigvecs;
      AnalyticalEigendecompSym2x2(ToSimdSymMatrix(mat), vEigvals, &vEigvecs);
      VerifyEigendecomp2x2(mat, ToReal2(vEigvals), ToNdArray2x2(vEigvecs));
    }
  }
}

/**************************************************************************************************
  Eigendecomposition: 3x3
*/

TEST(DecompositionUtils, AnalyticalEigendecompSym3x3_Scalar) {
  for (real scaling : kScalings) {
    for (auto const& mat0 : kTestMatricesSym3x3) {
      auto const mat = mat0 * scaling;
      Real3 eigvals;
      Matrix3x3r QT;
      AnalyticalEigendecompSym(mat, eigvals, &QT, /*transpose*/ true);
      VerifyEigendecomp3x3(mat, eigvals, QT);
    }
  }
}

TEST(DecompositionUtils, AnalyticalEigendecompSym3x3_SIMD) {
  for (real scaling : kScalings) {
    for (auto const& mat0 : kTestMatricesSym3x3) {
      auto const mat = mat0 * scaling;
      Vec4r vEigvals;
      VMatrix3x3r vEigvecs;
      AnalyticalEigendecompSym3x3(ToSimdSymMatrix(mat), vEigvals, &vEigvecs);
      VerifyEigendecomp3x3(mat, ToReal3(vEigvals), ToNdArray3x3(vEigvecs));
    }
  }
}

/**************************************************************************************************
  Eigendecomposition: Eigenvalues-only (eigvecs == nullptr)
*/

TEST(DecompositionUtils, EigendecompSym_ValsOnly) {
  // Eigenvalues are semantically identical with and without eigenvectors, but codegen (FMA
  // fusion, register allocation) may differ, so the two paths agree only to within a few ULP.
  // The error of a floating-point eigenvalue scales with the eigenvalue itself, so the tolerance
  // is relative to Norm(mat), an upper bound on |lambda_max| for a symmetric matrix; `scaling`
  // is the floor, for the zero matrix.
  real constexpr kUlpTol = 4_r * std::numeric_limits<real>::epsilon();

  for (real scaling : kScalings) {
    for (auto const& mat0 : kTestMatricesSym2x2) {
      auto const mat = mat0 * scaling;
      real const tol = kUlpTol * Max(scaling, Norm(mat));
      Real2 fullEigvals, valsOnly;
      Matrix2x2r eigvecs;
      AnalyticalEigendecompSym(mat, fullEigvals, &eigvecs);
      AnalyticalEigendecompSym(mat, valsOnly);
      EXPECT_NEAR_TOL(valsOnly, fullEigvals, tol);

      Vec4r vFull, vOnly;
      VMatrix2x2r vEigvecs;
      AnalyticalEigendecompSym2x2(ToSimdSymMatrix(mat), vFull, &vEigvecs);
      AnalyticalEigendecompSym2x2(ToSimdSymMatrix(mat), vOnly);
      EXPECT_NEAR_TOL(ToReal2(vOnly), ToReal2(vFull), tol);
    }
    for (auto const& mat0 : kTestMatricesSym3x3) {
      auto const mat = mat0 * scaling;
      real const tol = kUlpTol * Max(scaling, Norm(mat));
      Real3 fullEigvals, valsOnly;
      Matrix3x3r eigvecs;
      AnalyticalEigendecompSym(mat, fullEigvals, &eigvecs);
      AnalyticalEigendecompSym(mat, valsOnly);
      EXPECT_NEAR_TOL(valsOnly, fullEigvals, tol);

      Vec4r vFull, vOnly;
      VMatrix3x3r vEigvecs;
      AnalyticalEigendecompSym3x3(ToSimdSymMatrix(mat), vFull, &vEigvecs);
      AnalyticalEigendecompSym3x3(ToSimdSymMatrix(mat), vOnly);
      EXPECT_NEAR_TOL(ToReal3(vOnly), ToReal3(vFull), tol);
    }
  }
}

/**************************************************************************************************
  Eigendecomposition: Transpose flag consistency (scalar only — SIMD has no transpose parameter)
*/

TEST(DecompositionUtils, EigendecompSym_TransposeFlag) {
  real constexpr kUlpTol = 4_r * std::numeric_limits<real>::epsilon();
  for (real scaling : kScalings) {
    for (auto const& mat0 : kTestMatricesSym2x2) {
      auto const mat = mat0 * scaling;
      Real2 eigvals0, eigvals1;
      Matrix2x2r Q, QT;
      AnalyticalEigendecompSym(mat, eigvals0, &Q, /*transpose*/ false);
      AnalyticalEigendecompSym(mat, eigvals1, &QT, /*transpose*/ true);
      EXPECT_NEAR_TOL(eigvals0, eigvals1, kUlpTol * Max(scaling, Norm(mat)));
      // Compare reconstructions (eigenvector sign is not unique).
      Matrix2x2r const rec0 = Dot(Q, Dot(DiagonalMatrix(eigvals0), Transpose(Q)));
      Matrix2x2r const rec1 = Dot(Transpose(QT), Dot(DiagonalMatrix(eigvals1), QT));
      EXPECT_LE(Norm(rec0 - rec1), eigenRelTol * Norm(mat));
    }
    for (auto const& mat0 : kTestMatricesSym3x3) {
      auto const mat = mat0 * scaling;
      Real3 eigvals0, eigvals1;
      Matrix3x3r Q, QT;
      AnalyticalEigendecompSym(mat, eigvals0, &Q, /*transpose*/ false);
      AnalyticalEigendecompSym(mat, eigvals1, &QT, /*transpose*/ true);
      EXPECT_NEAR_TOL(eigvals0, eigvals1, kUlpTol * Max(scaling, Norm(mat)));
      // Compare reconstructions (eigenvector sign is not unique).
      Matrix3x3r const rec0 = Dot(Q, Dot(DiagonalMatrix(eigvals0), Transpose(Q)));
      Matrix3x3r const rec1 = Dot(Transpose(QT), Dot(DiagonalMatrix(eigvals1), QT));
      EXPECT_LE(Norm(rec0 - rec1), eigenRelTol * Norm(mat));
    }
  }
}

/**************************************************************************************************
  SVD: Fused
*/

TEST(DecompositionUtils, RotationVariantSvd3x3_Scalar) {
  // TODO: Make RotationVariantSvd3x3 robust to arbitrary scales and enable kScalings loop.
  for (auto const& mat : kTestMatrices3x3) {
    Real3 sigma;
    Matrix3x3r U, VT;
    RotationVariantSvd(mat, U, sigma, VT);
    VerifySvd3x3(mat, U, sigma, VT);
  }
}

TEST(DecompositionUtils, RotationVariantSvd3x3_SIMD) {
  // TODO: Make RotationVariantSvd3x3 robust to arbitrary scales and enable kScalings loop.
  for (auto const& mat : kTestMatrices3x3) {
    Vec4r sigma;
    VMatrix3x3r U, VT;
    RotationVariantSvd3x3(ToSimdMatrix(mat), U, sigma, VT);
    VerifySvd3x3(mat, ToNdArray3x3(U), ToReal3(sigma), ToNdArray3x3(VT));
  }
}

/**************************************************************************************************
  SVD: Split Vals/Vecs
*/

TEST(DecompositionUtils, RotationVariantSvd3x3_SplitValsVecs_Scalar) {
  // TODO: Make RotationVariantSvd3x3 robust to arbitrary scales and enable kScalings loop.
  // NOTE: Uses kNumMatsNormalized3x3. The split path does not normalize G, so the large-entry
  // matrix produces inaccurate results.
  for (int m = 0; m < kNumMatsNormalized3x3; ++m) {
    auto const& mat = kTestMatrices3x3[m];
    Real3 sigma;
    Int3 order;
    RotationVariantSvdVals(mat, sigma, order);

    Matrix3x3r U, VT;
    RotationVariantSvdVecs(mat, order, sigma, U, VT);

    VerifySvd3x3(mat, U, sigma, VT);
  }
}

TEST(DecompositionUtils, RotationVariantSvd3x3_SplitValsVecs_SIMD) {
  // TODO: Make RotationVariantSvd3x3 robust to arbitrary scales and enable kScalings loop.
  // NOTE: Uses kNumMatsNormalized3x3 — the split path does not normalize G, so the large-entry
  // matrix produces inaccurate results in single precision.
  for (int m = 0; m < kNumMatsNormalized3x3; ++m) {
    auto const& mat = kTestMatrices3x3[m];
    VMatrix3x3r const F = ToSimdMatrix(mat);
    Vec4r sigma;
    Int3 order;
    RotationVariantSvdVals3x3(F, sigma, order);

    VMatrix3x3r U, VT;
    RotationVariantSvdVecs3x3(F, order, sigma, U, VT);

    VerifySvd3x3(mat, ToNdArray3x3(U), ToReal3(sigma), ToNdArray3x3(VT));
  }
}

/**************************************************************************************************
  Polar Decomposition
*/

TEST(DecompositionUtils, LeftPolarDecomposition3x3) {
  real const relTol = 50_r * Sqrt(std::numeric_limits<real>::epsilon());
  // TODO: Make LeftPolarDecomposition3x3 robust to arbitrary scales and enable kScalings loop.
  for (auto const& mat : kTestMatrices3x3) {
    VMatrix3x3r A = ToSimdMatrix(mat);
    VMatrix3x3r U, P;
    LeftPolarDecomposition3x3(A, U, P);

    // Check A = U * P.
    EXPECT_LE(Norm3x3(Dot3x3(U, P) - A), relTol * Norm3x3(A));

    // Check U is unitary.
    EXPECT_LE(Norm3x3(Dot3x3(Transpose3x3(U), U) - VEye<3>()), relTol);

    // Check P is symmetric positive semi-definite.
    EXPECT_LE(Norm3x3(Transpose3x3(P) - P), relTol * Norm3x3(P));
    Vec4r eigsP;
    AnalyticalEigendecompSym3x3(SimdFullToSym(P), eigsP, /*Q*/ nullptr);
    EXPECT_GE(HMin<3>(eigsP), -relTol * Norm3x3(P)); // PSD to within tolerance.
  }
}

/**************************************************************************************************
  PSD Check and Projection
*/

TEST(DecompositionUtils, IsSpd) {
  constexpr real kEpsilon = 1e2_r * std::numeric_limits<real>::epsilon();

  // Targeted Sylvester criterion branches.
  // Branch 1: A₁₁ < 0 (first check fails).
  EXPECT_FALSE(
      details::IsSpd(
          VMatrix3x3r{Vec4r{-1_r, 0_r, 0_r}, Vec4r{0_r, 100_r, 0_r}, Vec4r{0_r, 0_r, 100_r}}));
  // Branch 2: A₁₁ > 0, 2×2 leading minor < 0 (second check fails).
  EXPECT_FALSE(
      details::IsSpd(
          VMatrix3x3r{Vec4r{1_r, 2_r, 0_r}, Vec4r{2_r, 1_r, 0_r}, Vec4r{0_r, 0_r, 100_r}}));
  // Branch 3: A₁₁ > 0, 2×2 minor > 0, det(3x3) < 0 (third check fails).
  EXPECT_FALSE(
      details::IsSpd(
          VMatrix3x3r{Vec4r{2_r, 1_r, 2_r}, Vec4r{1_r, 2_r, 1_r}, Vec4r{2_r, 1_r, 1_r}}));
  // All pass: identity.
  EXPECT_TRUE(details::IsSpd(VEye<3>()));

  // Indefinite matrix with non-negative leading principal minors but negative non-leading principal
  // minors.
  EXPECT_FALSE(details::IsSpd(VDiagonalMatrix<3>(Vec4r{1_r, 0_r, -1_r})));

  // Rotated matrices with known eigenvalues.
  for (real scaling : kScalings) {
    for (auto const& A : kTestMatricesSym3x3) {
      // Use the SVD to create an orthogonal matrix.
      Vec4r sv;
      VMatrix3x3r U, VT;
      RotationVariantSvd3x3(ToSimdMatrix(A), U, sv, VT);

      // Test symmetric matrices with known eigenvalues.
      EXPECT_TRUE(
          details::IsSpd(Dot3x3(
              U, Dot3x3(VDiagonalMatrix<3>(scaling * Vec4r{1_r, 2_r, 3_r}), Transpose3x3(U)))));
      EXPECT_TRUE(
          details::IsSpd(Dot3x3(
              U,
              Dot3x3(VDiagonalMatrix<3>(scaling * Vec4r{kEpsilon, 1_r, 1_r}), Transpose3x3(U)))));
      EXPECT_TRUE(
          details::IsSpd(Dot3x3(
              U,
              Dot3x3(VDiagonalMatrix<3>(scaling * Vec4r{1_r, kEpsilon, 1_r}), Transpose3x3(U)))));
      EXPECT_TRUE(
          details::IsSpd(Dot3x3(
              U,
              Dot3x3(VDiagonalMatrix<3>(scaling * Vec4r{1_r, 1_r, kEpsilon}), Transpose3x3(U)))));

      EXPECT_FALSE(
          details::IsSpd(Dot3x3(
              U, Dot3x3(VDiagonalMatrix<3>(scaling * Vec4r{-1_r, 2_r, 3_r}), Transpose3x3(U)))));
      EXPECT_FALSE(
          details::IsSpd(Dot3x3(
              U,
              Dot3x3(VDiagonalMatrix<3>(scaling * Vec4r{-kEpsilon, 1_r, 1_r}), Transpose3x3(U)))));
      EXPECT_FALSE(
          details::IsSpd(Dot3x3(
              U,
              Dot3x3(VDiagonalMatrix<3>(scaling * Vec4r{1_r, -kEpsilon, 1_r}), Transpose3x3(U)))));
      EXPECT_FALSE(
          details::IsSpd(Dot3x3(
              U,
              Dot3x3(VDiagonalMatrix<3>(scaling * Vec4r{1_r, 1_r, -kEpsilon}), Transpose3x3(U)))));
    }
  }
}

TEST(DecompositionUtils, ProjectSymPsd2x2) {
  for (real scaling : kScalings) {
    for (auto const& mat : kTestMatricesSym2x2) {
      // Perform eigendecomposition to get an orthogonal matrix.
      // Use transpose=true to get eigenvectors as rows (QT), matching the SIMD version
      // used internally by ProjectSymPsd.
      Real2 sv;
      Matrix2x2r QT;
      AnalyticalEigendecompSym(mat, sv, &QT, true);

      // Create a symmetric matrix with known eigenvalues: A = Q * D * Q^T = Q^T^T * D * Q^T
      Real2 const eigs = scaling * Real2{1_r, -2_r};
      Matrix2x2r const input = Dot(Transpose(QT), Dot(DiagonalMatrix(eigs), QT));
      real const inputNorm = Norm(input);

      // Compute expected result: same eigenvectors, eigenvalues clamped to eps.
      real const eps = scaling * kPsdRelTol;
      Real2 const clampedEigs{Max(eigs[0], eps), Max(eigs[1], eps)};
      Matrix2x2r const expected = Dot(Transpose(QT), Dot(DiagonalMatrix(clampedEigs), QT));

      // Check the projection matches the expected value.
      VMatrix2x2r computed = ToSimdMatrix(input);
      ProjectSymPsd(computed, eps);
      EXPECT_NEAR_TOL(ToNdArray2x2(computed), expected, kPsdRelTol * inputNorm);

      // Check the projection is idempotent.
      VMatrix2x2r const beforeSecondProjection = computed;
      ProjectSymPsd(computed, eps);
      EXPECT_NEAR_TOL(
          ToNdArray2x2(computed), ToNdArray2x2(beforeSecondProjection), kPsdRelTol * inputNorm);
    }
  }

  // Already-PSD inputs must remain unchanged up to eigendecomposition/reconstruction roundoff.
  for (real scaling : kScalings) {
    Matrix2x2r const input = scaling * Matrix2x2r{Real2{2_r, 0.5_r}, Real2{0.5_r, 1_r}};
    VMatrix2x2r projected = ToSimdMatrix(input);
    ProjectSymPsd(projected, scaling * kPsdRelTol);
    EXPECT_NEAR_TOL(ToNdArray2x2(projected), input, kPsdRelTol * Norm(input));
  }
}

TEST(DecompositionUtils, ProjectSymPsd3x3) {
  for (real scaling : kScalings) {
    for (auto const& mat : kTestMatricesSym3x3) {
      // Use the SVD to create an orthogonal matrix.
      Vec4r sv;
      VMatrix3x3r U, VT;
      RotationVariantSvd3x3(ToSimdMatrix(mat), U, sv, VT);

      // Create a symmetric matrix with known eigenvalues.
      Vec4r const eigs = scaling * Vec4r{1_r, -2_r, 3_r};
      VMatrix3x3r computed = Dot3x3(U, Dot3x3(VDiagonalMatrix<3>(eigs), Transpose3x3(U)));

      // Check the projection matches the expected value.
      ProjectSymPsd(computed, scaling * kPsdRelTol);
      VMatrix3x3r const expected =
          Dot3x3(U, Dot3x3(VDiagonalMatrix<3>(Max(eigs, SimdZero<Vec4r>())), Transpose3x3(U)));
      EXPECT_NEAR_TOL(
          ToNdArray3x3(computed), ToNdArray3x3(expected), kPsdRelTol * Norm3x3(expected));

      // Check the projection is idempotent.
      ProjectSymPsd(computed, scaling * kPsdRelTol);
      EXPECT_NEAR_TOL(
          ToNdArray3x3(computed), ToNdArray3x3(expected), kPsdRelTol * Norm3x3(expected));
    }
  }

  // Bit-exact preservation: an already-PSD input must be returned unchanged.
  for (real scaling : kScalings) {
    VMatrix3x3r const psdInput3 = VDiagonalMatrix<3>(scaling * Vec4r{1_r, 2_r, 3_r});
    VMatrix3x3r projected3 = psdInput3;
    ProjectSymPsd(projected3, scaling * kPsdRelTol);
    EXPECT_EQ(ToNdArray3x3(projected3), ToNdArray3x3(psdInput3));
  }
}

/**************************************************************************************************
  Cholesky Decomposition
*/

TEST(DecompositionUtils, CholeskySym2x2) {
  // Near-singular and non-PD matrices: epsilon clamping must produce a valid lower-triangular
  // factor with positive diagonal. The result L*Lᵀ is a PSD approximation (not a faithful
  // factorization of the input).
  // clang-format off
  VMatrix2x2r const kNearSingular[] = {
    VMatrix2x2r{0_r, 0_r, 0_r, 1_r},            // a00 = 0 → l00 clamp
    VMatrix2x2r{1_r, 1_r, 1_r, 1_r},             // Schur complement = 0 → l11 clamp
    VMatrix2x2r{0_r, 0_r, 0_r, 0_r},             // Both clamp
    VMatrix2x2r{-1_r, 0_r, 0_r, 2_r},            // a00 < 0 → l00 clamp
    VMatrix2x2r{1e-20_r, 0_r, 0_r, 1_r},         // a00 tiny → l00 clamp
  };
  // clang-format on

  for (real scaling : kScalings) {
    real const eps = scaling * kCholeskyRelTol;

    // PSD matrices.
    for (auto const& mat : kTestMatricesSym2x2) {
      VMatrix2x2r A = ToSimdMatrix(mat * scaling);
      ProjectSymPsd(A, eps);

      VMatrix2x2r const L = CholeskySym2x2(A, eps);

      Matrix2x2r const LNd = ToNdArray2x2(L);
      Matrix2x2r const reconstructed = Dot(LNd, Transpose(LNd));
      Matrix2x2r const ANd = ToNdArray2x2(A);

      EXPECT_LE(Norm(reconstructed - ANd), kCholeskyRelTol * Norm(ANd));
      EXPECT_NEAR(LNd[0][1], 0_r, kCholeskyRelTol * Norm(ANd));
      EXPECT_GT(LNd[0][0], 0_r);
      EXPECT_GT(LNd[1][1], 0_r);
    }

    // Near-singular and non-PD matrices at this scale.
    for (auto const& A0 : kNearSingular) {
      VMatrix2x2r const A = A0 * scaling;
      VMatrix2x2r const L = CholeskySym2x2(A, eps);
      Matrix2x2r const LNd = ToNdArray2x2(L);

      EXPECT_NEAR(LNd[0][1], 0_r, kCholeskyRelTol);
      EXPECT_GT(LNd[0][0], 0_r);
      EXPECT_GT(LNd[1][1], 0_r);

      // L*Lᵀ must be PSD.
      Matrix2x2r const reconstructed = Dot(LNd, Transpose(LNd));
      VMatrix2x2r projected = ToSimdMatrix(reconstructed);
      ProjectSymPsd(projected, eps);
      EXPECT_LE(
          Norm(ToNdArray2x2(projected) - reconstructed), kCholeskyRelTol * Norm(reconstructed));
    }
  }
}

/**************************************************************************************************
  Metric-Aware PSD Projection
*/

TEST(DecompositionUtils, ProjectPsdWithMetric) {
  for (auto const& metric : kTestMetrics2x2) {
    for (real scaling : kScalings) {
      Matrix2x2r const MNd =
          scaling * Matrix2x2r{Real2{metric[0], metric[1]}, Real2{metric[2], metric[3]}};
      VMatrix2x2r const M = ToSimdMatrix(MNd);

      for (auto const& mat : kTestMatricesSym2x2) {
        // Create a symmetric tensor S (not necessarily PSD)
        Real2 sv;
        Matrix2x2r QT;
        AnalyticalEigendecompSym(mat, sv, &QT, true);
        Real2 const eigs = scaling * Real2{2_r, -1_r};
        Matrix2x2r const SNd = Dot(Transpose(QT), Dot(DiagonalMatrix(eigs), QT));
        VMatrix2x2r const S = ToSimdMatrix(SNd);

        // epsCholesky scales with ||M||; epsEigenvalue scales with generalized eigenvalue
        // magnitude.
        real const epsCholesky = kPsdRelTol * Norm(MNd);
        real const epsEigenvalue = kPsdRelTol * Norm(SNd) / Norm(MNd);
        VMatrix2x2r const SProj = ProjectPsdWithMetric(S, M, epsCholesky, epsEigenvalue);

        // Verify the result is symmetric and PSD in the generalized metric sense.
        Matrix2x2r const SProjNd = ToNdArray2x2(SProj);
        EXPECT_LE(Norm(SProjNd - Transpose(SProjNd)), kPsdRelTol * Norm(SProjNd));
        VerifyPsdWithMetric(SProjNd, MNd, epsCholesky, epsEigenvalue);

        // Verify the projection is idempotent.
        VMatrix2x2r const SProjAgain = ProjectPsdWithMetric(SProj, M, epsCholesky, epsEigenvalue);
        EXPECT_LE(Norm(ToNdArray2x2(SProjAgain - SProj)), kPsdRelTol * Norm(SProjNd));
      }
    }
  }

  // Test that already M-PSD matrices are unchanged.
  for (auto const& metric : kTestMetrics2x2) {
    for (real scaling : kScalings) {
      Matrix2x2r const MNd =
          scaling * Matrix2x2r{Real2{metric[0], metric[1]}, Real2{metric[2], metric[3]}};
      VMatrix2x2r const M = ToSimdMatrix(MNd);

      // Cholesky: M = L * Lᵀ.
      VMatrix2x2r const L = CholeskySym2x2(M, scaling * kCholeskyRelTol);
      Matrix2x2r const LNd = ToNdArray2x2(L);

      for (auto const& mat : kTestMatricesSym2x2) {
        // Create a standard-PSD matrix A from mat's eigenvectors with positive eigenvalues.
        Real2 sv;
        Matrix2x2r QT;
        AnalyticalEigendecompSym(mat, sv, &QT, true);
        Matrix2x2r const ANd = Dot(Transpose(QT), Dot(DiagonalMatrix(Real2{1_r, 2_r}), QT));

        // S = Lᵀ * A * L is guaranteed M-PSD (generalized eigenvalues = eigenvalues of A > 0).
        Matrix2x2r const SNd = Dot(Transpose(LNd), Dot(ANd, LNd));
        VMatrix2x2r const S = ToSimdMatrix(SNd);

        // Projection should preserve S since it's already M-PSD.
        real const epsCholesky = kPsdRelTol * Norm(MNd);
        real const epsEigenvalue = kPsdRelTol * Norm(SNd) / Norm(MNd);
        VMatrix2x2r const SProj = ProjectPsdWithMetric(S, M, epsCholesky, epsEigenvalue);
        EXPECT_LE(Norm(ToNdArray2x2(SProj - S)), kPsdRelTol * Norm(SNd));
      }
    }
  }

  // Test with different scales for S and M, across all test metrics.
  for (auto const& metric : kTestMetrics2x2) {
    for (real mScaling : {1e-3_r, 1_r, 1e3_r}) {
      for (real sScaling : {1e-3_r, 1_r, 1e3_r}) {
        // Create a PSD metric matrix M
        Matrix2x2r const MNd =
            mScaling * Matrix2x2r{Real2{metric[0], metric[1]}, Real2{metric[2], metric[3]}};
        VMatrix2x2r const M = ToSimdMatrix(MNd);

        // Create a symmetric tensor S (not necessarily PSD, with eigenvalues 2 and -1 in S-scale)
        Real2 sv;
        Matrix2x2r QT;
        AnalyticalEigendecompSym(MNd, sv, &QT, true);
        Real2 const eigs = sScaling * Real2{2_r, -1_r};
        Matrix2x2r const SNd = Dot(Transpose(QT), Dot(DiagonalMatrix(eigs), QT));
        VMatrix2x2r const S = ToSimdMatrix(SNd);

        // epsCholesky scales with ||M||.
        // epsEigenvalue: generalized eigenvalues are O(||S||/||M||), so scale accordingly.
        real const epsCholesky = kPsdRelTol * Norm(MNd);
        real const epsEigenvalue = kPsdRelTol * Norm(SNd) / Norm(MNd);
        VMatrix2x2r const SProj = ProjectPsdWithMetric(S, M, epsCholesky, epsEigenvalue);

        // Verify the result is symmetric and PSD in the generalized metric sense.
        Matrix2x2r const SProjNd = ToNdArray2x2(SProj);
        EXPECT_LE(Norm(SProjNd - Transpose(SProjNd)), kPsdRelTol * Norm(SProjNd));
        VerifyPsdWithMetric(SProjNd, MNd, epsCholesky, epsEigenvalue);

        // Verify the projection is idempotent.
        VMatrix2x2r const SProjAgain = ProjectPsdWithMetric(SProj, M, epsCholesky, epsEigenvalue);
        EXPECT_LE(Norm(ToNdArray2x2(SProjAgain - SProj)), kPsdRelTol * Norm(SProjNd));
      }
    }
  }
}

/**************************************************************************************************
  Batched Eigendecomposition Tests
*/

template <int kBatchSize>
static void TestBatchedEigendecompSym3x3() {
  for (real scaling : kScalings) {
    std::array<Matrix3x3r, kBatchSize> mats{};
    for (int i = 0; i < kBatchSize; ++i) {
      mats[i] = kTestMatricesSym3x3[i % kNumMatsSym3x3] * scaling;
    }

    std::array<Real3, kBatchSize> eigvals{};
    std::array<Matrix3x3r, kBatchSize> eigvecs{};
    BatchedAnalyticalEigendecompSym3x3<kBatchSize>(
        MakeConstSpan(mats), MakeSpan(eigvals), MakeSpan(eigvecs));

    for (int i = 0; i < kBatchSize; ++i) {
      VerifyEigendecomp3x3(mats[i], eigvals[i], eigvecs[i]);
    }
  }
}

MOCHI_BATCH_TEST(DecompositionUtils, BatchedEigendecompSym3x3, TestBatchedEigendecompSym3x3)

template <int kBatchSize>
static void TestBatchedEigendecompSym3x3ValsOnly() {
  for (real scaling : kScalings) {
    std::array<Matrix3x3r, kBatchSize> mats{};
    for (int i = 0; i < kBatchSize; ++i) {
      mats[i] = kTestMatricesSym3x3[i % kNumMatsSym3x3] * scaling;
    }

    // Compute with eigenvectors (reference).
    std::array<Real3, kBatchSize> refEigvals{};
    std::array<Matrix3x3r, kBatchSize> refEigvecs{};
    BatchedAnalyticalEigendecompSym3x3<kBatchSize>(
        MakeConstSpan(mats), MakeSpan(refEigvals), MakeSpan(refEigvecs));

    // Compute vals-only (empty eigvecs span).
    std::array<Real3, kBatchSize> eigvals{};
    BatchedAnalyticalEigendecompSym3x3<kBatchSize>(
        MakeConstSpan(mats), MakeSpan(eigvals), Span<Matrix3x3r>{});

    // Vals-only must match the full version exactly (both code paths share the same eigenvalue
    // computation).
    for (int i = 0; i < kBatchSize; ++i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_EQ(eigvals[i][j], refEigvals[i][j]);
      }
    }
  }
}

MOCHI_BATCH_TEST(
    DecompositionUtils,
    BatchedEigendecompSym3x3_ValsOnly,
    TestBatchedEigendecompSym3x3ValsOnly)

template <int kBatchSize>
static void TestBatchedEigendecompSym3x3MixedLane() {
  std::array<Matrix3x3r, kBatchSize> mats{};
  for (int i = 0; i < kBatchSize; ++i) {
    mats[i] = kMixedLaneMatsSym3x3[i % kNumMixedLaneMatsSym3x3];
  }

  std::array<Real3, kBatchSize> eigvals{};
  std::array<Matrix3x3r, kBatchSize> eigvecs{};
  BatchedAnalyticalEigendecompSym3x3<kBatchSize>(
      MakeConstSpan(mats), MakeSpan(eigvals), MakeSpan(eigvecs));

  for (int i = 0; i < kBatchSize; ++i) {
    VerifyEigendecomp3x3(mats[i], eigvals[i], eigvecs[i]);
  }
}

MOCHI_BATCH_TEST(
    DecompositionUtils,
    BatchedEigendecompSym3x3_MixedLane,
    TestBatchedEigendecompSym3x3MixedLane)

/**************************************************************************************************
  Batched SVD Tests
*/

template <int kBatchSize>
static void TestBatchedSvd3x3() {
  // TODO: Make BatchedRotationVariantSvd3x3 robust to arbitrary scales and enable kScalings loop.
  for (int batch = 0; batch < kNumMats3x3; batch += kBatchSize) {
    std::array<Matrix3x3r, kBatchSize> mats{};
    for (int i = 0; i < kBatchSize; ++i) {
      mats[i] = kTestMatrices3x3[(batch + i) % kNumMats3x3];
    }

    std::array<Matrix3x3r, kBatchSize> U{}, VT{};
    std::array<Real3, kBatchSize> sigma{};
    BatchedRotationVariantSvd3x3<kBatchSize>(
        MakeConstSpan(mats), MakeSpan(U), MakeSpan(sigma), MakeSpan(VT));

    for (int i = 0; i < kBatchSize; ++i) {
      VerifySvd3x3(mats[i], U[i], sigma[i], VT[i]);
    }
  }
}

MOCHI_BATCH_TEST(DecompositionUtils, BatchedSvd3x3, TestBatchedSvd3x3)

template <int kBatchSize>
static void TestBatchedSvd3x3ValsVecs() {
  // TODO: Make BatchedRotationVariantSvd3x3 robust to arbitrary scales and enable kScalings loop.
  for (int batch = 0; batch < kNumMats3x3; batch += kBatchSize) {
    std::array<Matrix3x3r, kBatchSize> mats{};
    for (int i = 0; i < kBatchSize; ++i) {
      mats[i] = kTestMatrices3x3[(batch + i) % kNumMats3x3];
    }

    std::array<Matrix3x3r, kBatchSize> U{}, VT{};
    std::array<Real3, kBatchSize> sigma{};
    BatchedRotationVariantSvdValsVecs3x3<kBatchSize>(
        MakeConstSpan(mats), MakeSpan(U), MakeSpan(sigma), MakeSpan(VT));

    for (int i = 0; i < kBatchSize; ++i) {
      VerifySvd3x3(mats[i], U[i], sigma[i], VT[i]);
    }

    // Check the 2-argument overload of BatchedRotationVariantSvdVals3x3.
    std::array<Real3, kBatchSize> sigmaValsOnly{};
    BatchedRotationVariantSvdVals3x3<kBatchSize>(MakeConstSpan(mats), MakeSpan(sigmaValsOnly));

    for (int i = 0; i < kBatchSize; ++i) {
      EXPECT_NEAR_EQ(sigma[i], sigmaValsOnly[i]);
    }
  }
}

MOCHI_BATCH_TEST(DecompositionUtils, BatchedSvd3x3_ValsVecs, TestBatchedSvd3x3ValsVecs);

template <int kBatchSize>
static void TestBatchedSvd3x3MixedLane() {
  std::array<Matrix3x3r, kBatchSize> mats{};
  for (int i = 0; i < kBatchSize; ++i) {
    mats[i] = kMixedLaneMats3x3[i % kNumMixedLaneMats3x3];
  }

  // Fused path.
  std::array<Matrix3x3r, kBatchSize> U{}, VT{};
  std::array<Real3, kBatchSize> sigma{};
  BatchedRotationVariantSvd3x3<kBatchSize>(
      MakeConstSpan(mats), MakeSpan(U), MakeSpan(sigma), MakeSpan(VT));

  for (int i = 0; i < kBatchSize; ++i) {
    VerifySvd3x3(mats[i], U[i], sigma[i], VT[i]);
  }

  // Split vals/vecs path.
  std::array<Matrix3x3r, kBatchSize> USplit{}, VTSplit{};
  std::array<Real3, kBatchSize> sigmaSplit{};
  BatchedRotationVariantSvdValsVecs3x3<kBatchSize>(
      MakeConstSpan(mats), MakeSpan(USplit), MakeSpan(sigmaSplit), MakeSpan(VTSplit));

  for (int i = 0; i < kBatchSize; ++i) {
    VerifySvd3x3(mats[i], USplit[i], sigmaSplit[i], VTSplit[i]);
  }

  // Cross-validate: fused and split paths must produce the same singular values.
  for (int i = 0; i < kBatchSize; ++i) {
    for (int k = 0; k < 3; ++k) {
      EXPECT_NEAR(sigma[i][k], sigmaSplit[i][k], svdRelTol * Abs(sigma[i][k]) + svdRelTol);
    }
  }
}

MOCHI_BATCH_TEST(DecompositionUtils, BatchedSvd3x3_MixedLane, TestBatchedSvd3x3MixedLane)

/**************************************************************************************************
  Batched PSD Projection Tests
*/

template <int kBatchSize>
static void TestBatchedProjectSymPsd() {
  std::array<Matrix3x3r, kBatchSize> mats{};
  for (int i = 0; i < kBatchSize; ++i) {
    mats[i] = kMixedLaneMatsPsd3x3[i % kNumMixedLaneMatsPsd3x3];
  }

  // Scalar reference.
  std::array<Matrix3x3r, kBatchSize> expected{};
  for (int i = 0; i < kBatchSize; ++i) {
    VMatrix3x3r v = ToSimdMatrix(mats[i]);
    ProjectSymPsd(v);
    expected[i] = ToNdArray3x3(v);
  }

  // Batched.
  using V = BatchReal<kBatchSize>;
  BatchReal3x3<kBatchSize> aBatch{};
  alignas(alignof(V)) real staging[V::kSize]{};
  for (int k = 0; k < 9; ++k) {
    int const row = k / 3;
    int const col = k % 3;
    for (int i = 0; i < kBatchSize; ++i) {
      staging[i] = mats[i][row][col];
    }
    aBatch[row][col] = Load<V>(staging);
  }
  BatchedProjectSymPsd<kBatchSize>(aBatch);

  for (int i = 0; i < kBatchSize; ++i) {
    for (int k = 0; k < 9; ++k) {
      int const row = k / 3;
      int const col = k % 3;
      EXPECT_NEAR(
          aBatch[row][col][i],
          expected[i][row][col],
          Max(kPsdRelTol * Abs(expected[i][row][col]), kPsdAbsTol));
    }
  }
}

MOCHI_BATCH_TEST(DecompositionUtils, BatchedProjectSymPsd, TestBatchedProjectSymPsd)

template <int kBatchSize>
static void TestBatchedProjectPsdWithMetric() {
  constexpr int kNumCombinations = kNumMixedLaneMatsPsd2x2 * kNumTestMetrics2x2;
  for (int offset = 0; offset < kNumCombinations; offset += kBatchSize) {
    NdArray<NdArray<real, 4>, kBatchSize> sFlat, mFlat;
    for (int i = 0; i < kBatchSize; ++i) {
      int const combo = (offset + i) % kNumCombinations;
      sFlat[i] = kMixedLaneMatsPsd2x2[combo / kNumTestMetrics2x2];
      mFlat[i] = kTestMetrics2x2[combo % kNumTestMetrics2x2];
    }

    // Scalar reference.
    NdArray<NdArray<real, 4>, kBatchSize> expected;
    for (int i = 0; i < kBatchSize; ++i) {
      VMatrix2x2r const s{sFlat[i][0], sFlat[i][1], sFlat[i][2], sFlat[i][3]};
      VMatrix2x2r const m{mFlat[i][0], mFlat[i][1], mFlat[i][2], mFlat[i][3]};
      VMatrix2x2r const proj = ProjectPsdWithMetric(s, m);
      expected[i] = {proj[0], proj[1], proj[2], proj[3]};
    }

    // Batched.
    using V = BatchReal<kBatchSize>;
    BatchReal2x2<kBatchSize> sBatch{}, mBatch{};
    alignas(alignof(V)) real staging[V::kSize]{};
    for (int k = 0; k < 4; ++k) {
      int const row = k / 2;
      int const col = k % 2;
      for (int i = 0; i < kBatchSize; ++i) {
        staging[i] = sFlat[i][k];
      }
      sBatch[row][col] = Load<V>(staging);
      for (int i = 0; i < kBatchSize; ++i) {
        staging[i] = mFlat[i][k];
      }
      mBatch[row][col] = Load<V>(staging);
    }
    auto const result = BatchedProjectPsdWithMetric<kBatchSize>(sBatch, mBatch);

    for (int i = 0; i < kBatchSize; ++i) {
      for (int k = 0; k < 4; ++k) {
        int const row = k / 2;
        int const col = k % 2;
        EXPECT_NEAR(
            result[row][col][i],
            expected[i][k],
            Max(kPsdRelTol * Max(Abs(expected[i][k]), Norm(sFlat[i])), kPsdAbsTol));
      }
    }
  }
}

MOCHI_BATCH_TEST(DecompositionUtils, BatchedProjectPsdWithMetric, TestBatchedProjectPsdWithMetric)

/**************************************************************************************************
  Cross-Validation: Batched vs Non-Batched Reference
*/

TEST(DecompositionUtils, BatchedEigendecompSym3x3_CrossValidation) {
  constexpr int kBatch = 4;

  for (real scaling : kScalings) {
    std::array<Matrix3x3r, kBatch> mats{};
    for (int i = 0; i < kBatch; ++i) {
      mats[i] = kTestMatricesSym3x3[i % kNumMatsSym3x3] * scaling;
    }

    std::array<Real3, kBatch> batchEigvals{};
    std::array<Matrix3x3r, kBatch> batchEigvecs{};
    BatchedAnalyticalEigendecompSym3x3<kBatch>(
        MakeConstSpan(mats), MakeSpan(batchEigvals), MakeSpan(batchEigvecs));

    for (int i = 0; i < kBatch; ++i) {
      Real3 scalarEigvals;
      AnalyticalEigendecompSym(mats[i], scalarEigvals);
      for (int j = 0; j < 3; ++j) {
        EXPECT_LE(Abs(batchEigvals[i][j] - scalarEigvals[j]), 2_r * eigenRelTol * Norm(mats[i]));
      }
    }
  }
}

// TODO: Make RotationVariantSvd3x3 robust to arbitrary scales and enable kScalings loop.
TEST(DecompositionUtils, BatchedSvd3x3_CrossValidation) {
  constexpr int kBatch = 4;
  for (int batch = 0; batch < kNumMats3x3; batch += kBatch) {
    std::array<Matrix3x3r, kBatch> mats{};
    for (int i = 0; i < kBatch; ++i) {
      mats[i] = kTestMatrices3x3[(batch + i) % kNumMats3x3];
    }

    std::array<Matrix3x3r, kBatch> batchU{}, batchVT{};
    std::array<Real3, kBatch> batchSigma{};
    BatchedRotationVariantSvd3x3<kBatch>(
        MakeConstSpan(mats), MakeSpan(batchU), MakeSpan(batchSigma), MakeSpan(batchVT));

    for (int i = 0; i < kBatch; ++i) {
      Real3 scalarSigma;
      Matrix3x3r scalarU, scalarVT;
      RotationVariantSvd(mats[i], scalarU, scalarSigma, scalarVT);
      for (int j = 0; j < 3; ++j) {
        EXPECT_LE(
            Abs(batchSigma[i][j] - scalarSigma[j]),
            2_r * svdRelTol * Max(Norm(mats[i]), std::numeric_limits<real>::min()));
      }
    }
  }
}

TEST(DecompositionUtils, BatchedProjectSymPsd_CrossValidation) {
  constexpr int kBatch = 4;
  auto rng = RandomGenerator(42);

  // Edge-case 3x3 symmetric matrices (row-major 9 components).
  // Already PSD, identity, negative-definite, diagonal (3x), zero.
  NdArray<NdArray<real, 9>, 7> const edgeCases = {
      NdArray<real, 9>{2_r, 0.5_r, 0_r, 0.5_r, 3_r, 0.1_r, 0_r, 0.1_r, 1_r},
      NdArray<real, 9>{1_r, 0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r},
      NdArray<real, 9>{-5_r, -1_r, 0_r, -1_r, -8_r, -0.5_r, 0_r, -0.5_r, -3_r},
      NdArray<real, 9>{100_r, 0_r, 0_r, 0_r, 0.1_r, 0_r, 0_r, 0_r, 0.01_r},
      NdArray<real, 9>{0.1_r, 0_r, 0_r, 0_r, 0.01_r, 0_r, 0_r, 0_r, 100_r},
      NdArray<real, 9>{0.01_r, 0_r, 0_r, 0_r, 100_r, 0_r, 0_r, 0_r, 0.1_r},
      NdArray<real, 9>{0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r}};

  for (int trial = 0; trial < 24; ++trial) {
    NdArray<NdArray<real, 9>, kBatch> aScalar;
    for (int i = 0; i < kBatch; ++i) {
      if (trial < isize(edgeCases)) {
        aScalar[i] = edgeCases[trial];
      } else {
        // Random symmetric 3x3.
        real const a00 = RandomUniformValue<real>(rng, -3_r, 3_r);
        real const a01 = RandomUniformValue<real>(rng, -2_r, 2_r);
        real const a02 = RandomUniformValue<real>(rng, -2_r, 2_r);
        real const a11 = RandomUniformValue<real>(rng, -3_r, 3_r);
        real const a12 = RandomUniformValue<real>(rng, -2_r, 2_r);
        real const a22 = RandomUniformValue<real>(rng, -3_r, 3_r);
        aScalar[i] = {a00, a01, a02, a01, a11, a12, a02, a12, a22};
      }
    }

    // Scalar reference.
    NdArray<NdArray<real, 9>, kBatch> scalarResult;
    for (int i = 0; i < kBatch; ++i) {
      VMatrix3x3r mat{
          Vec4r{aScalar[i][0], aScalar[i][1], aScalar[i][2]},
          Vec4r{aScalar[i][3], aScalar[i][4], aScalar[i][5]},
          Vec4r{aScalar[i][6], aScalar[i][7], aScalar[i][8]}};
      ProjectSymPsd(mat);
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          scalarResult[i][row * 3 + col] = mat[row][col];
        }
      }
    }

    // Batched.
    using V = BatchReal<kBatch>;
    BatchReal3x3<kBatch> aBatch{};
    alignas(alignof(V)) real staging[V::kSize]{};
    for (int k = 0; k < 9; ++k) {
      for (int i = 0; i < kBatch; ++i) {
        staging[i] = aScalar[i][k];
      }
      aBatch[k / 3][k % 3] = Load<V>(staging);
    }
    BatchedProjectSymPsd<kBatch>(aBatch);

    // Compare.
    for (int i = 0; i < kBatch; ++i) {
      for (int k = 0; k < 9; ++k) {
        EXPECT_NEAR(
            aBatch[k / 3][k % 3][i],
            scalarResult[i][k],
            Max(kPsdRelTol * Abs(scalarResult[i][k]), kPsdAbsTol));
      }
    }
  }
}

TEST(DecompositionUtils, BatchedProjectPsdWithMetric_CrossValidation) {
  constexpr int kBatch = 4;
  auto rng = RandomGenerator(99);

  // Edge-case S matrices: already-PSD, identity, negative-definite, diagonal (2x), zero.
  NdArray<NdArray<real, 4>, 6> const edgeCaseS = {
      NdArray<real, 4>{2_r, 0.5_r, 0.5_r, 3_r},
      NdArray<real, 4>{1_r, 0_r, 0_r, 1_r},
      NdArray<real, 4>{-5_r, -1_r, -1_r, -8_r},
      NdArray<real, 4>{100_r, 0_r, 0_r, 0.01_r},
      NdArray<real, 4>{0.01_r, 0_r, 0_r, 100_r},
      NdArray<real, 4>{0_r, 0_r, 0_r, 0_r}};

  for (int trial = 0; trial < 24; ++trial) {
    NdArray<NdArray<real, 4>, kBatch> sScalar, mScalar;
    for (int i = 0; i < kBatch; ++i) {
      if (trial < isize(edgeCaseS)) {
        // Use edge-case S with identity metric.
        sScalar[i] = edgeCaseS[trial];
        mScalar[i] = {1_r, 0_r, 0_r, 1_r};
      } else {
        // Random S (symmetric) and M (SPD).
        real const s00 = RandomUniformValue<real>(rng, -2_r, 2_r);
        real const s01 = RandomUniformValue<real>(rng, -2_r, 2_r);
        real const s11 = RandomUniformValue<real>(rng, -2_r, 2_r);
        sScalar[i] = {s00, s01, s01, s11};

        real const l00 = RandomUniformValue<real>(rng, 0.5_r, 2_r);
        real const l10 = RandomUniformValue<real>(rng, -1_r, 1_r);
        real const l11 = RandomUniformValue<real>(rng, 0.5_r, 2_r);
        mScalar[i] = {l00 * l00, l00 * l10, l00 * l10, l10 * l10 + l11 * l11};
      }
    }

    // Scalar reference.
    NdArray<NdArray<real, 4>, kBatch> scalarResult;
    for (int i = 0; i < kBatch; ++i) {
      VMatrix2x2r const sSimd{sScalar[i][0], sScalar[i][1], sScalar[i][2], sScalar[i][3]};
      VMatrix2x2r const mSimd{mScalar[i][0], mScalar[i][1], mScalar[i][2], mScalar[i][3]};
      VMatrix2x2r const projected = ProjectPsdWithMetric(sSimd, mSimd);
      scalarResult[i] = {projected[0], projected[1], projected[2], projected[3]};
    }

    // Batched.
    using V = BatchReal<kBatch>;
    BatchReal2x2<kBatch> sBatch{}, mBatch{};
    alignas(alignof(V)) real staging[V::kSize]{};
    for (int k = 0; k < 4; ++k) {
      for (int i = 0; i < kBatch; ++i) {
        staging[i] = sScalar[i][k];
      }
      sBatch[k / 2][k % 2] = Load<V>(staging);
      for (int i = 0; i < kBatch; ++i) {
        staging[i] = mScalar[i][k];
      }
      mBatch[k / 2][k % 2] = Load<V>(staging);
    }
    auto const batchResult = BatchedProjectPsdWithMetric<kBatch>(sBatch, mBatch);

    // Compare.
    for (int i = 0; i < kBatch; ++i) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(
            batchResult[k / 2][k % 2][i],
            scalarResult[i][k],
            Max(kPsdRelTol * Abs(scalarResult[i][k]), kPsdAbsTol));
      }
    }
  }
}

/**************************************************************************************************
  Overflow-Path Coverage

  The overflow branch in ComputePsiSym3x3 (scalar) guards `x1 > 1e10 || |x2| > 1e10`. This condition
  is only ever true when the detail function receives an unnormalized G = 𝐅ᵀ𝐅 with very large
  entries — the public eigendecomp APIs normalize before calling detail, so the overflow path is
  unreachable through them. It is reachable only through the scalar and SIMD split-SVD paths
  (RotationVariantSvdVals, RotationVariantSvdVals3x3), which call the detail function directly on
  raw G. The batched split path normalizes G first, so its detail function has no overflow branch.

  These tests invoke the split paths on a large-entry F (entries ~O(1e5)) so that G = 𝐅ᵀ𝐅 has
  entries ~O(1e10) and thus x₁ ~O(1e20) >> 1e10. The fused SVD (which normalizes G internally,
  avoiding the overflow path) serves as the reference. Individual singular values from the
  split path are cross-validated against this reference.
*/

// Relative tolerance for cross-validating overflow-path singular values against the fused SVD
// reference.
static constexpr real kOverflowSvdTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 5e-2_r;

TEST(DecompositionUtils, SplitSvd_OverflowPath_Scalar) {
  auto const& F = kTestMatrices3x3[kNumMats3x3 - 1];
  EXPECT_GE(Norm(F), 1e5_r); // Confirm F has large entries.

  // Split path (non-normalizing → triggers overflow guard in ComputePsiSym3x3).
  Real3 sigma;
  Int3 order;
  RotationVariantSvdVals(F, sigma, order);
  EXPECT_TRUE(IsFinite(sigma));

  // Cross-validate against fused path (normalizes internally → avoids overflow).
  Real3 refSigma;
  Matrix3x3r U, VT;
  RotationVariantSvd(F, U, refSigma, VT);
  for (int j = 0; j < 3; ++j) {
    EXPECT_NEAR(sigma[j], refSigma[j], kOverflowSvdTol * Abs(refSigma[j]));
  }
}

TEST(DecompositionUtils, SplitSvd_OverflowPath_SIMD) {
  auto const& F = kTestMatrices3x3[kNumMats3x3 - 1];
  EXPECT_GE(Norm(F), 1e5_r); // Confirm F has large entries.

  // Split path.
  Vec4r sigma;
  Int3 order;
  RotationVariantSvdVals3x3(ToSimdMatrix(F), sigma, order);
  Real3 const s = ToReal3(sigma);
  EXPECT_TRUE(IsFinite(s));

  // Cross-validate against fused path.
  Vec4r refSigma;
  VMatrix3x3r U, VT;
  RotationVariantSvd3x3(ToSimdMatrix(F), U, refSigma, VT);
  Real3 const ref = ToReal3(refSigma);
  for (int j = 0; j < 3; ++j) {
    EXPECT_NEAR(s[j], ref[j], kOverflowSvdTol * Abs(ref[j]));
  }
}
