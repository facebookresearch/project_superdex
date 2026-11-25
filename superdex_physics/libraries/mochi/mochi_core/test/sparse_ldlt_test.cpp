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
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <limits>
#include <utility>

namespace mochi {

template <typename LDLt, typename M>
static void ExpectRefactorizationSolves(LDLt& ldlt, M const& A) {
  int info = 0;
  ldlt.Refactorize(A, info);
  EXPECT_EQ(info, 0);

  ColumnVector<real> expected(A.Rows()), rhs(A.Rows());
  expected.SetRandom(42);
  A.Apply(expected, rhs);
  ldlt.LeftSolveInPlace(rhs);
  EXPECT_TRUE(test::NearEqualMatrices(expected, rhs, 5_r * kDefaultNearEqualEpsilon<real>));
}

template <int kBlockSize, bool kUseBlockSparseMatrix>
static void TestSparseLDLtSolve(Matrix<real> const& A, unsigned int seed) {
  static_assert(
      kUseBlockSparseMatrix || kBlockSize == 1, "SparseMatrix only supports block size 1");

  int const n = A.Rows();
  ColumnVector<real> x(n), f(n), y(n);
  x.SetRandom(seed);

  int info = 0;
  real constexpr kTolerance = 5_r * kDefaultNearEqualEpsilon<real>;
  if constexpr (kUseBlockSparseMatrix) {
    auto Absp = ToBlockSparseMatrix<kBlockSize>(A, /*pruneZeros*/ true);
    Absp.Apply(x, f);

    // Test factorization.
    krylov::SparseLDLt ldlt(Absp, info);
    EXPECT_EQ(info, 0);

    y = f;
    ldlt.LeftSolveInPlace(y);
    EXPECT_TRUE(test::NearEqualMatrices(x, y, kTolerance));

    // Test refactorizing a factorization of other values.
    auto AbspScaled = Absp;
    AsView(AbspScaled.Values()) *= 2_r;
    krylov::SparseLDLt ldltRefactorize(AbspScaled, info);
    ExpectRefactorizationSolves(ldltRefactorize, Absp);
  } else {
    auto Asp = ToSparseMatrix(A, /*pruneZeros*/ true);
    Asp.Apply(x, f);

    // Test factorization.
    krylov::SparseLDLt ldlt(Asp, info);
    EXPECT_EQ(info, 0);

    y = f;
    ldlt.LeftSolveInPlace(y);
    EXPECT_TRUE(test::NearEqualMatrices(x, y, kTolerance));

    // Test refactorizing a factorization of other values.
    auto AspScaled = Asp;
    AsView(AspScaled.Values()) *= 2_r;
    krylov::SparseLDLt ldltRefactorize(AspScaled, info);
    ExpectRefactorizationSolves(ldltRefactorize, Asp);
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

/**************************************************************************************
 * Scheduler-Backed and Adaptive Factorization Tests
 *
 * Verifies factorization with and without a scheduler. Large block Laplacians exceed the adaptive
 * work cutoff and exercise parallel factorization; the smaller and disconnected cases exercise
 * sequential fallbacks and multiple roots.
 */

// For block sizes 3 and 4, the block Laplacian's root gets a split plan when built with a worker
// besides the calling thread.
template <int kBlockSize>
static auto ParallelLaplacian() {
  return test::Create2dLaplacianMatrix<real, kBlockSize>(24, 24);
}

/// @brief Create the block diagonal matrix diag(A, B).
static Matrix<real> CreateBlockDiagonal(Matrix<real> const& A, Matrix<real> const& B) {
  int const nA = A.Rows();
  int const nB = B.Rows();
  auto C = Matrix<real>::Zero(nA + nB, nA + nB);
  C.Block(0, 0, nA, nA) = A;
  C.Block(nA, nA, nB, nB) = B;
  return C;
}

/// @brief Test factorization, solves and refactorization without a scheduler, then with worker
/// requests of one and four, each capped at the supported logical processor count.
template <int kBlockSize>
static void TestSparseLDLtWithAndWithoutScheduler(Matrix<real> const& A, unsigned int seed) {
  ASSERT_EQ(nullptr, TaskScheduler::TryGet());
  TestSparseLDLtSolve<kBlockSize, true>(A, seed);

  for (int numWorkers : {1, 4}) {
    TaskScheduler scheduler(Min(TaskScheduler::GetNumSupportedLogicalProcessors(), numWorkers));
    TestSparseLDLtSolve<kBlockSize, true>(A, seed);
  }
}

TEST(SparseLDLt, WithAndWithoutScheduler) {
  // Small disconnected components are factorized directly, one root at a time.
  TestSparseLDLtWithAndWithoutScheduler<1>(
      CreateBlockDiagonal(
          test::Create2dLaplacianMatrix<real, 1>(5, 3),
          test::Create2dLaplacianMatrix<real, 1>(4, 2)),
      42);

  // A dense matrix has enough work to parallelize, but its single supernode is a leaf, which
  // PickBranches returns whole.
  mochi_default_random_engine rng(42);
  TestSparseLDLtWithAndWithoutScheduler<3>(
      test::CreateBlockBandedMatrix<real, 3>(32, 31, rng, true), 43);

  // One factorization handles a parallel root and a sequential one.
  TestSparseLDLtWithAndWithoutScheduler<3>(
      CreateBlockDiagonal(ParallelLaplacian<3>(), test::Create2dLaplacianMatrix<real, 3>(3, 2)),
      42);

  // 4-DoF blocks only add work, so this root is also factorized in parallel.
  TestSparseLDLtWithAndWithoutScheduler<4>(ParallelLaplacian<4>(), 42);
}

// Return the parallel block Laplacian with a diagonal shift.
static auto ShiftedLaplacian(real shift) {
  auto dense = ParallelLaplacian<3>();
  for (int i = 0; i < dense.Rows(); ++i) {
    dense(i, i) += shift;
  }
  return ToBlockSparseMatrix<3>(dense, /*pruneZeros*/ true);
}

// Each refactorization uses a new shift, so a skipped refactorization fails its solve.
TEST(SparseLDLt, RefactorizeAcrossSchedulerModes) {
  ASSERT_EQ(nullptr, TaskScheduler::TryGet());

  TaskScheduler scheduler(Min(TaskScheduler::GetNumSupportedLogicalProcessors(), 4));
  auto const initial = ShiftedLaplacian(0_r);

  // Initialize with parallel branch picks.
  int info = 0;
  krylov::SparseLDLt initializedMultithreaded(initial, info);
  EXPECT_EQ(info, 0);

  // Refactorize with the parallel plans created during initialization.
  ExpectRefactorizationSolves(initializedMultithreaded, ShiftedLaplacian(1_r));

  // Local single-threaded mode temporarily bypasses existing parallel plans.
  {
    ScopedLocalSingleThreadedMode const singleThreaded;
    ExpectRefactorizationSolves(initializedMultithreaded, ShiftedLaplacian(2_r));
  }

  // Global single-threaded mode also bypasses existing parallel plans.
  scheduler.SetGlobalSingleThreadedMode(true);
  ExpectRefactorizationSolves(initializedMultithreaded, ShiftedLaplacian(3_r));

  // Plans omitted during single-threaded initialization remain unavailable after restoring
  // multithreading.
  krylov::SparseLDLt initializedSingleThreaded(initial, info);
  scheduler.SetGlobalSingleThreadedMode(false);
  EXPECT_EQ(info, 0);
  ExpectRefactorizationSolves(initializedSingleThreaded, ShiftedLaplacian(1_r));
}

// Refactorizing after the scheduler is gone must bypass the parallel plans.
TEST(SparseLDLt, RefactorizeAfterSchedulerDestroyed) {
  ASSERT_EQ(nullptr, TaskScheduler::TryGet());

  auto const initial = ShiftedLaplacian(0_r);
  int info = 0;
  auto ldlt = [&] {
    TaskScheduler scheduler(Min(TaskScheduler::GetNumSupportedLogicalProcessors(), 4));
    return krylov::SparseLDLt(initial, info);
  }();
  EXPECT_EQ(info, 0);
  ASSERT_EQ(nullptr, TaskScheduler::TryGet());

  ExpectRefactorizationSolves(ldlt, ShiftedLaplacian(1_r));
}

// Simulation factorizes inside scheduler tasks, so factorize from two of them.
TEST(SparseLDLt, FactorizeFromWorkers) {
  ASSERT_EQ(nullptr, TaskScheduler::TryGet());

  TaskScheduler scheduler(Min(TaskScheduler::GetNumSupportedLogicalProcessors(), 4));
  auto const initial = ShiftedLaplacian(0_r);
  auto const shifted = ShiftedLaplacian(1_r);
  TaskSemaphore done;
  for (int task = 0; task < 2; ++task) {
    scheduler.AddTask(done, "FactorizeFromWorker", [&] {
      EXPECT_TRUE(TaskScheduler::IsCurrentThreadAWorker());
      int info = 0;
      krylov::SparseLDLt ldlt(initial, info);
      EXPECT_EQ(info, 0);
      ExpectRefactorizationSolves(ldlt, shifted);
    });
  }
  done.Wait();
}

TEST(SparseLDLt, SumsDuplicateEntries) {
  // [[4, 1, 0], [1, 5, 2], [0, 2, 6]], with each entry split in two and the rows unsorted.
  SparseMatrix<real> const A(
      3,
      DynamicArray<int>{0, 4, 10, 14},
      DynamicArray<int>{1, 0, 0, 1, 2, 1, 0, 1, 0, 2, 2, 1, 2, 1},
      DynamicArray<real>{
          0.5_r, 3_r, 1_r, 0.5_r, 1_r, 2_r, 0.25_r, 3_r, 0.75_r, 1_r, 4_r, 1.5_r, 2_r, 0.5_r});

  // Factorizing other values first checks that refactorization assembles them all again.
  auto doubled = A;
  AsView(doubled.Values()) *= 2_r;

  int info = 0;
  krylov::SparseLDLt ldlt(doubled, info);
  EXPECT_EQ(info, 0);
  ExpectRefactorizationSolves(ldlt, A);
}

TEST(SparseLDLt, ReportsUnusablePivots) {
  auto const expectUnusablePivots = [](auto const& A) {
    int info = 0;
    krylov::SparseLDLt ldlt(A, info);
    EXPECT_NE(info, 0);

    info = 0;
    ldlt.Refactorize(A, info);
    EXPECT_NE(info, 0);
  };

  // [[0, 1], [1, 0]] has a zero pivot under every order.
  expectUnusablePivots(
      SparseMatrix<real>(
          2,
          DynamicArray<int>{0, 2, 4},
          DynamicArray<int>{0, 1, 0, 1},
          DynamicArray<real>{0_r, 1_r, 1_r, 0_r}));

  // A row without stored entries, not even its diagonal, has a zero pivot.
  expectUnusablePivots(
      SparseMatrix<real>(
          2, DynamicArray<int>{0, 1, 1}, DynamicArray<int>{0}, DynamicArray<real>{1_r}));

  // An infinite diagonal entry gives an infinite pivot.
  expectUnusablePivots(
      SparseMatrix<real>(
          1,
          DynamicArray<int>{0, 1},
          DynamicArray<int>{0},
          DynamicArray<real>{std::numeric_limits<real>::infinity()}));

  // A NaN diagonal entry gives a NaN pivot, also when the root is factorized in parallel.
  TaskScheduler scheduler(Min(TaskScheduler::GetNumSupportedLogicalProcessors(), 4));
  auto withNaN = ParallelLaplacian<3>();
  withNaN(0, 0) = std::numeric_limits<real>::quiet_NaN();
  expectUnusablePivots(ToBlockSparseMatrix<3>(withNaN, /*pruneZeros*/ true));
}

} // namespace mochi
