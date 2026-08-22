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

#include <mochi_core/linear_algebra/krylov/parallel_matrix_vector_product_pool.h>

#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
#include <mochi_core/linear_algebra/krylov/gmres.h>
#include <mochi_core/linear_algebra/krylov/pcg.h>
#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/utils/task_scheduler.h>

#include <limits>

#include <gtest/gtest.h>

using namespace mochi;

namespace mochi::test {

template <typename Scalar>
static void TestParallelMatrixVectorProductPool(
    bool singleThreadedMode,
    bool masterPerformsProduct) {
  // TODO(@pabfer): Once the task scheduler has a reliable way to schedule a batch of tasks in
  // different threads, add checks to confirm that both the parallel and the fallback
  // implementations are tested.

  int const numThreads = TaskScheduler::GetNumSupportedLogicalProcessors();
  auto scheduler = TaskScheduler(numThreads);
  EXPECT_EQ(&scheduler, TaskScheduler::TryGet());
  EXPECT_EQ(numThreads, scheduler.GetNumThreads());

  if (singleThreadedMode) {
    // Force tasks to be scheduled on the calling thread, even if more threads are available.
    scheduler.SetGlobalSingleThreadedMode(true);
  }

  for (auto n : {1, 2, 5}) {
    // Create matrix in all supported formats.
    BlockSparseMatrix<Scalar, 3> ABSp = MakeBlockSparseMatrix<Scalar, 3>(n, n, n);
    SparseMatrix<Scalar> ASp = ToSparseMatrix(ABSp);
    Matrix<Scalar> ADense = ToMatrix(ABSp);
    RowMatrix<Scalar> ADenseRow = ADense;

    int const numRows = ABSp.Rows();
    int const numCols = ABSp.Cols();
    EXPECT_EQ(numRows, numCols); // Square matrix
    EXPECT_GT(numRows, 0); // Not a dummy test

    // Solver settings.
    int const maxIter =
        2 * numRows; // Larger than numRows to check convergence in at most numRows iterations
    Scalar const relTol = Scalar(10) * std::numeric_limits<Scalar>::epsilon();

    Scalar constexpr kAbsTol = 1e-17;
    Scalar constexpr kRelDivTol = 1e10;

    // Parallel matrix-vector pools.
    auto opABSp = krylov::ParallelMatrixVectorProductPool(ABSp, masterPerformsProduct);
    auto opASp = krylov::ParallelMatrixVectorProductPool(ASp, masterPerformsProduct);
    auto opADense = krylov::ParallelMatrixVectorProductPool(ADense, masterPerformsProduct);
    auto opADenseRow = krylov::ParallelMatrixVectorProductPool(ADenseRow, masterPerformsProduct);

    // Preconditioner.
    krylov::JacobiPrec<Scalar> P(ABSp);
    auto opP = [&P](auto const& x, auto& Px) { P(x, Px); };

    // Right-hand side and solution vector.
    ColumnVector<Scalar> b(numRows);
    ColumnVector<Scalar> xBSp(numCols), xSp(numCols), xDense(numCols), xDenseRow(numCols),
        xRef(numCols);
    b.SetRandom(123);

    // Solve with PCG.
    krylov::StatusResidualL2<krylov::UsualDot, Scalar> stopperPcg{relTol, kAbsTol, kRelDivTol};
    xBSp.SetZero();
    auto infoPcg = krylov::PCG(opABSp, b, xBSp, opP, maxIter, stopperPcg);
    if (singleThreadedMode) {
      EXPECT_FALSE(opABSp.IsPoolEnabled());
    }
    EXPECT_TRUE(infoPcg.converged);
    EXPECT_LT(infoPcg.numIterDone, maxIter);
    EXPECT_LT(infoPcg.relativeResidualNorm, relTol);
    opABSp.Reset();

    // Check solution matches that with the reference matrix-vector product.
    xRef.SetZero();
    auto opARef = [&ABSp](auto const& x, auto& Ax) { Apply(ABSp, x, Ax); };
    krylov::PCG(opARef, b, xRef, opP, maxIter, stopperPcg);
    ColumnVector<Scalar> xDiff = xRef - xBSp;
    EXPECT_LE(xDiff.Norm(), xDiff.Rows() * relTol);

    // Sparse matrix specialization.
    xSp.SetZero();
    infoPcg = krylov::PCG(opASp, b, xSp, opP, maxIter, stopperPcg);
    xDiff = xBSp - xSp;
    if (singleThreadedMode) {
      EXPECT_FALSE(opASp.IsPoolEnabled());
    }
    EXPECT_TRUE(infoPcg.converged);
    EXPECT_LT(infoPcg.numIterDone, maxIter);
    EXPECT_LT(infoPcg.relativeResidualNorm, relTol);
    EXPECT_LT(xDiff.Norm(), relTol * numRows * xBSp.Norm());
    opASp.Reset();

    // Col-major dense matrix specialization.
    xDense.SetZero();
    infoPcg = krylov::PCG(opADense, b, xDense, opP, maxIter, stopperPcg);
    xDiff = xBSp - xDense;
    if (singleThreadedMode) {
      EXPECT_FALSE(opADense.IsPoolEnabled());
    }
    EXPECT_TRUE(infoPcg.converged);
    EXPECT_LT(infoPcg.numIterDone, maxIter);
    EXPECT_LT(infoPcg.relativeResidualNorm, relTol);
    EXPECT_LT(xDiff.Norm(), relTol * numRows * xBSp.Norm());
    opADense.Reset();

    // Row-major dense matrix specialization.
    xDenseRow.SetZero();
    infoPcg = krylov::PCG(opADenseRow, b, xDenseRow, opP, maxIter, stopperPcg);
    xDiff = xBSp - xDenseRow;
    if (singleThreadedMode) {
      EXPECT_FALSE(opADenseRow.IsPoolEnabled());
    }
    EXPECT_TRUE(infoPcg.converged);
    EXPECT_LT(infoPcg.numIterDone, maxIter);
    EXPECT_LT(infoPcg.relativeResidualNorm, relTol);
    EXPECT_LT(xDiff.Norm(), relTol * numRows * xBSp.Norm());
    opADenseRow.Reset();

    // Solve with the inverse as preconditioner and check convergence in 1 iteration.
    Matrix<Scalar> invA = Inverse(ADense);
    xBSp.SetZero();
    auto opInvA = [&invA](auto const& x, auto& Px) { Px = invA * x; };
    infoPcg = krylov::PCG(opABSp, b, xBSp, opInvA, maxIter, stopperPcg);
    if (singleThreadedMode) {
      EXPECT_FALSE(opABSp.IsPoolEnabled());
    }
    EXPECT_TRUE(infoPcg.converged);
    EXPECT_EQ(infoPcg.numIterDone, 1);
    EXPECT_LE(infoPcg.relativeResidualNorm, relTol);
    opABSp.Reset();

    // Solve with GMRES.
    krylov::StatusImplicitResidualNorm<Scalar> stopperGmres{relTol, kAbsTol, kRelDivTol};
    xBSp.SetZero();
    auto infoGmres = krylov::GMRes(opABSp, b, xBSp, opP, maxIter, stopperGmres);
    if (singleThreadedMode) {
      EXPECT_FALSE(opABSp.IsPoolEnabled());
    }
    EXPECT_TRUE(infoGmres.converged);
    EXPECT_LT(infoGmres.numIterDone, maxIter);
    EXPECT_LT(infoGmres.relativeResidualNorm, relTol);
    opABSp.Reset();

    // Test 'ReleaseWorkers'.
    opABSp.ReleaseWorkers();
    opASp.ReleaseWorkers();
    opADense.ReleaseWorkers();
    opADenseRow.ReleaseWorkers();
    EXPECT_FALSE(opABSp.IsPoolEnabled());
    EXPECT_FALSE(opASp.IsPoolEnabled());
    EXPECT_FALSE(opADense.IsPoolEnabled());
    EXPECT_FALSE(opADenseRow.IsPoolEnabled());
  }
}

} // namespace mochi::test

TEST(ParallelMatrixVectorProductPool, SingleThreadedSolve) {
  mochi::test::TestParallelMatrixVectorProductPool<real>(true, true);
  mochi::test::TestParallelMatrixVectorProductPool<real>(true, false);
}

TEST(ParallelMatrixVectorProductPool, MultiThreadedSolve) {
  mochi::test::TestParallelMatrixVectorProductPool<real>(false, true);
  mochi::test::TestParallelMatrixVectorProductPool<real>(false, false);
}
