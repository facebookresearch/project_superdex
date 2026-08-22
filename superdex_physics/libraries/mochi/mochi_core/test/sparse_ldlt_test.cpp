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

#include <mochi_core/linear_algebra/krylov/sparse_ldlt.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <utility>

namespace mochi {

template <int kBlockSize, bool kUseBlockSparseMatrix>
static void TestSparseLDLtSolve(Matrix<real> const& A, unsigned int seed) {
  static_assert(
      kUseBlockSparseMatrix || kBlockSize == 1, "SparseMatrix only supports block size 1");

  int const n = A.Rows();
  ColumnVector<real> x(n), f(n), y(n);
  x.SetRandom(seed);

  int info = 0;
  real constexpr kTolerance = 5_r * kDefaultNearEqualEpsilon<real>;
  mochi_default_random_engine rng(seed);
  if constexpr (kUseBlockSparseMatrix) {
    auto Absp = ToBlockSparseMatrix<kBlockSize>(A, /*pruneZeros*/ true);
    Absp.Apply(x, f);

    // Test factorization.
    krylov::SparseLDLt ldlt(Absp, info);
    EXPECT_EQ(info, 0);

    y = f;
    ldlt.LeftSolveInPlace(y);
    EXPECT_TRUE(test::NearEqualMatrices(x, y, kTolerance));

    // Test refactorization.
    auto AbspRandom = Absp;
    SetRandom(rng, -1_r, 1_r, AbspRandom.Values());

    krylov::SparseLDLt ldltRefactorize(AbspRandom, info);
    ldltRefactorize.Refactorize(Absp, info);
    EXPECT_EQ(info, 0);

    y = f;
    ldltRefactorize.LeftSolveInPlace(y);
    EXPECT_TRUE(test::NearEqualMatrices(x, y, kTolerance));
  } else {
    auto Asp = ToSparseMatrix(A, /*pruneZeros*/ true);
    Asp.Apply(x, f);

    // Test factorization.
    krylov::SparseLDLt ldlt(Asp, info);
    EXPECT_EQ(info, 0);

    y = f;
    ldlt.LeftSolveInPlace(y);
    EXPECT_TRUE(test::NearEqualMatrices(x, y, kTolerance));

    // Test refactorization.
    auto AspRandom = Asp;
    SetRandom(rng, -1_r, 1_r, AspRandom.Values());

    krylov::SparseLDLt ldltRefactorize(AspRandom, info);
    ldltRefactorize.Refactorize(Asp, info);
    EXPECT_EQ(info, 0);

    y = f;
    ldltRefactorize.LeftSolveInPlace(y);
    EXPECT_TRUE(test::NearEqualMatrices(x, y, kTolerance));
  }
}

/// @brief Test SparseLDLt with a 2D Laplacian matrix.
template <int kBlockSize, bool kUseBlockSparseMatrix>
static void TestSparseLDLtLaplacian() {
  int const nx = 7;
  int const ny = 3;
  unsigned int seed = 123;
  auto A = test::Create2dLaplacianMatrix<real, kBlockSize>(nx, ny);
  TestSparseLDLtSolve<kBlockSize, kUseBlockSparseMatrix>(A, seed);
}

TEST(SparseLDLt, Laplacian) {
  TestSparseLDLtLaplacian<1, false>();
  TestSparseLDLtLaplacian<1, true>();
  TestSparseLDLtLaplacian<3, true>();
  TestSparseLDLtLaplacian<4, true>();
}

/**************************************************************************************
 * Comprehensive SparseLDLt Tests
 *
 * Tests for the sparse LDL^T solver with various:
 * - Sparsity patterns (block-banded with symmetric permutations)
 * - Numerical values
 * - Matrix sizes
 * - Block sizes
 */

/// @brief Configuration for SparseLDLt test cases.
struct SparseLDLtTestConfig {
  int numBlockRows;
  int bandwidth;
  unsigned int seed;
  bool enforceDiagonallyDominant;
  bool applyPermutation;
};

/// @brief Test SparseLDLt with given configuration.
template <int kBlockSize, bool kUseBlockSparseMatrix>
static void TestSparseLDLtWithConfig(SparseLDLtTestConfig const& config) {
  static_assert(
      kUseBlockSparseMatrix || kBlockSize == 1, "SparseMatrix only supports block size 1");
  if (!config.enforceDiagonallyDominant) {
    // TODO: Enable once pivoting is implemented.
    return;
  }

  mochi_default_random_engine rng(config.seed);

  // Create dense block-banded matrix
  auto A = test::CreateBlockBandedMatrix<real, kBlockSize>(
      config.numBlockRows, config.bandwidth, rng, config.enforceDiagonallyDominant);

  // Optionally apply symmetric permutation at the block level
  if (config.applyPermutation) {
    auto blockPerm = test::CreateRandomPermutation(config.numBlockRows, rng);
    A = test::ApplySymmetricPermutation<kBlockSize>(A, MakeConstSpan(blockPerm));
  }

  TestSparseLDLtSolve<kBlockSize, kUseBlockSparseMatrix>(A, config.seed);
}

TEST(SparseLDLt, ParameterizedConfigs) {
  // Size -> bandwidths mapping.
  DynamicArray<std::pair<int, DynamicArray<int>>> sizeBandwidths = {
      {2, {0, 1}}, {5, {0, 1, 2}}, {10, {0, 1, 3, 5}}, {50, {0, 2, 7}}, {100, {0, 2, 8}}};
#if MOCHI_OPTIMIZED // Expensive cases. Only in optimized builds.
  auto constexpr kLDLtBlockSize = krylov::SparseLDLt<real, 3>::kLDLtBlockSize;
  static_assert(kLDLtBlockSize > 2);
  for (int numBlockRows :
       {kLDLtBlockSize - 2,
        kLDLtBlockSize - 1,
        kLDLtBlockSize,
        kLDLtBlockSize + 1,
        kLDLtBlockSize + 2,
        2 * kLDLtBlockSize - 1,
        2 * kLDLtBlockSize,
        2 * kLDLtBlockSize + 1,
        3 * kLDLtBlockSize}) {
    sizeBandwidths.push_back({numBlockRows, {0, 3, 10}});
  }
#endif // MOCHI_OPTIMIZED

  unsigned int seed = 100;
  DynamicArray<SparseLDLtTestConfig> configs;
  for (auto const& [numBlockRows, bandwidths] : sizeBandwidths) {
    for (int bandwidth : bandwidths) {
      // Non-permuted case.
      configs.emplace_back(SparseLDLtTestConfig{numBlockRows, bandwidth, ++seed, true, false});
      // Permuted cases with different random seeds.
      for (int i = 0; i < 4; ++i) {
        configs.emplace_back(SparseLDLtTestConfig{numBlockRows, bandwidth, ++seed, true, true});
      }
      // Non-diagonally dominant case.
      configs.emplace_back(SparseLDLtTestConfig{numBlockRows, bandwidth, ++seed, false, true});
    }
  }

  for (auto const& cfg : configs) {
    TestSparseLDLtWithConfig<1, false>(cfg);
    TestSparseLDLtWithConfig<1, true>(cfg);
    TestSparseLDLtWithConfig<3, true>(cfg);
    TestSparseLDLtWithConfig<4, true>(cfg);
  }
}

} // namespace mochi
