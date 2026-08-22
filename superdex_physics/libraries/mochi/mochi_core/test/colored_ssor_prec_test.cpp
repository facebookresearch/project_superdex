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

#include <mochi_core/linear_algebra/krylov/colored_ssor_prec.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>

using namespace mochi;
using namespace mochi::krylov;

// Use real instead of float or double to reduce build time. Both are checked by CI.
using Scalar = real;

template <typename Scalar, typename InType, typename OutType>
static void Example1() {
  //
  // This matrix is similar to the 1D Laplacian problem
  // The coloring will yield red-black ordering.
  //
  DynamicArray<int> rp{0, 2, 5, 8, 11, 14, 16};
  DynamicArray<int> ci{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5};
  DynamicArray<Scalar> va{2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, 7};
  SparseMatrix<Scalar, int, int> C(6, rp, ci, va);
  InType x(C.Rows(), 2);
  for (int i = 0; i < C.Rows(); ++i) {
    x(i, 0) = Scalar(i + 1);
    x(i, 1) = Scalar(C.Rows() - i);
  }
  auto const tol = Scalar(0.000001);
  {
    krylov::ColoredSSORPrec<SparseMatrix<Scalar, int, int>> prec(C, Scalar(0.7));
    auto y = OutType::Zero(C.Rows(), 2);
    prec(x, y);
    EXPECT_NEAR_RTOL(y(0, 0), Scalar(6.673333333333333e-01), tol);
    EXPECT_NEAR_RTOL(y(1, 0), Scalar(9.761266666666666e-01), tol);
    EXPECT_NEAR_RTOL(y(2, 0), Scalar(9.160666666666666e-01), tol);
    EXPECT_NEAR_RTOL(y(3, 0), Scalar(9.870466666666666e-01), tol);
    EXPECT_NEAR_RTOL(y(4, 0), Scalar(9.342666666666667e-01), tol);
    EXPECT_NEAR_RTOL(y(5, 0), Scalar(8.734266666666668e-01), tol);
    //
    EXPECT_NEAR_RTOL(y(0, 1), Scalar(3.260833333333333e+00), tol);
    EXPECT_NEAR_RTOL(y(1, 1), Scalar(2.574086666666667e+00), tol);
    EXPECT_NEAR_RTOL(y(2, 1), Scalar(1.270966666666667e+00), tol);
    EXPECT_NEAR_RTOL(y(3, 1), Scalar(7.774433333333333e-01), tol);
    EXPECT_NEAR_RTOL(y(4, 1), Scalar(3.822000000000000e-01), tol);
    EXPECT_NEAR_RTOL(y(5, 1), Scalar(1.682200000000001e-01), tol);
  }
  {
    auto BSpC = ToBlockSparseMatrix<1>(C);
    krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, 1, int, int>> prec(BSpC, Scalar(0.7));
    auto y = OutType::Zero(C.Rows(), 2);
    prec(x, y);
    EXPECT_NEAR_RTOL(y(0, 0), Scalar(6.673333333333333e-01), tol);
    EXPECT_NEAR_RTOL(y(1, 0), Scalar(9.761266666666666e-01), tol);
    EXPECT_NEAR_RTOL(y(2, 0), Scalar(9.160666666666666e-01), tol);
    EXPECT_NEAR_RTOL(y(3, 0), Scalar(9.870466666666666e-01), tol);
    EXPECT_NEAR_RTOL(y(4, 0), Scalar(9.342666666666667e-01), tol);
    EXPECT_NEAR_RTOL(y(5, 0), Scalar(8.734266666666668e-01), tol);
    //
    EXPECT_NEAR_RTOL(y(0, 1), Scalar(3.260833333333333e+00), tol);
    EXPECT_NEAR_RTOL(y(1, 1), Scalar(2.574086666666667e+00), tol);
    EXPECT_NEAR_RTOL(y(2, 1), Scalar(1.270966666666667e+00), tol);
    EXPECT_NEAR_RTOL(y(3, 1), Scalar(7.774433333333333e-01), tol);
    EXPECT_NEAR_RTOL(y(4, 1), Scalar(3.822000000000000e-01), tol);
    EXPECT_NEAR_RTOL(y(5, 1), Scalar(1.682200000000001e-01), tol);
  }
  {
    auto BSpC = ToBlockSparseMatrix<2>(C);
    krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, 2, int, int>> prec(BSpC, Scalar(0.7));
    auto y = OutType::Zero(C.Rows(), 2);
    prec(x, y);
    EXPECT_NEAR_RTOL(y(0, 0), Scalar(7.602291666666667e-01), tol);
    EXPECT_NEAR_RTOL(y(1, 0), Scalar(8.720833333333333e-01), tol);
    EXPECT_NEAR_RTOL(y(2, 0), Scalar(1.002642700465972e+00), tol);
    EXPECT_NEAR_RTOL(y(3, 0), Scalar(9.573035264722222e-01), tol);
    EXPECT_NEAR_RTOL(y(4, 0), Scalar(9.553823319444443e-01), tol);
    EXPECT_NEAR_RTOL(y(5, 0), Scalar(8.654414166666669e-01), tol);
    //
    EXPECT_NEAR_RTOL(y(0, 1), Scalar(3.558100000000000e+00), tol);
    EXPECT_NEAR_RTOL(y(1, 1), Scalar(2.366000000000000e+00), tol);
    EXPECT_NEAR_RTOL(y(2, 1), Scalar(1.451732210463889e+00), tol);
    EXPECT_NEAR_RTOL(y(3, 1), Scalar(7.296126312222223e-01), tol);
    EXPECT_NEAR_RTOL(y(4, 1), Scalar(4.015187944444444e-01), tol);
    EXPECT_NEAR_RTOL(y(5, 1), Scalar(1.681896666666667e-01), tol);
  }
}

template <typename Scalar, typename InType, typename OutType>
static void Example2() {
  //
  // This matrix is similar to the 1D Laplacian problem
  // The coloring will yield red-black ordering.
  //
  DynamicArray<int> rp{0, 2, 5, 8, 11, 14, 16};
  DynamicArray<int> ci{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5};
  DynamicArray<Scalar> va{2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, 7};
  SparseMatrix<Scalar, int, int> C(6, rp, ci, va);
  InType x(C.Rows(), 1);
  for (int i = 0; i < C.Rows(); ++i) {
    x(i, 0) = Scalar(i + 1);
  }
  auto const tol = Scalar(0.000001);
  {
    ParallelBarrier barrier(1);
    krylov::ColoredSSORPrec<SparseMatrix<Scalar, int, int>> prec(C, Scalar(0.7));
    auto y = OutType::Zero(C.Rows(), 1);
    prec(x, y);
    auto yp = OutType::Zero(C.Rows(), 1);
    prec.ConcurrentSolve(x, yp, {0, 1, 0, 6, barrier});
    EXPECT_TRUE(mochi::test::NearEqualMatrices(y, yp, tol));
  }
  {
    ParallelBarrier barrier(1);
    auto BSpC = ToBlockSparseMatrix<1>(C);
    krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, 1, int, int>> prec(BSpC, Scalar(0.7));
    auto y = OutType::Zero(C.Rows(), 1);
    prec(x, y);
    auto yp = OutType::Zero(C.Rows(), 1);
    prec.ConcurrentSolve(x, yp, {0, 1, 0, 6, barrier});
    EXPECT_TRUE(mochi::test::NearEqualMatrices(y, yp, tol));
  }
  {
    ParallelBarrier barrier(1);
    auto BSpC = ToBlockSparseMatrix<2>(C);
    krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, 2, int, int>> prec(BSpC, Scalar(0.7));
    auto y = OutType::Zero(C.Rows(), 1);
    prec(x, y);
    auto yp = OutType::Zero(C.Rows(), 1);
    prec.ConcurrentSolve(x, yp, {0, 1, 0, 6, barrier});
    EXPECT_TRUE(mochi::test::NearEqualMatrices(y, yp, tol));
  }
}

template <typename Fn>
static bool ParallelFor(int numWorkers, Fn const& forEach) {
  TaskSemaphore sem;
  TaskScheduler::BatchTaskFn task = [sem, &forEach](int workerIdx, int /*numWorkers*/) {
    TaskScheduler::PushLocalSingleThreadedMode();
    forEach(workerIdx);
    TaskScheduler::PopLocalSingleThreadedMode();
    sem.Done();
  };
  TaskScheduler scheduler(numWorkers);
  if (scheduler.BatchEnqueueOnAvailableWorkers(
          sem,
          std::move(task),
          /*minWorkers*/ numWorkers,
          /*targetWorkers*/ numWorkers,
          /*includeSelf*/ true) == numWorkers) {
    sem.Wait();
    return true;
  } else {
    return false;
  }
}

template <typename Scalar, typename InType, typename OutType>
static void Example3() {
  //
  // This matrix is similar to the 1D Laplacian problem
  // The coloring will yield red-black ordering.
  //
  DynamicArray<int> rp{0, 2, 5, 8, 11, 14, 16};
  DynamicArray<int> ci{0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5};
  DynamicArray<Scalar> va{2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, 7};
  SparseMatrix<Scalar, int, int> C(6, rp, ci, va);
  InType x(C.Rows(), 1);
  for (int i = 0; i < C.Rows(); ++i) {
    x(i, 0) = Scalar(i + 1);
  }
  auto const tol = Scalar(0.000001);
  //
  int countParallel = 0;
  //
  for (auto numWorkers : {1, 2, 3, 4, 5, 6}) {
    if (numWorkers > TaskScheduler::GetNumSupportedLogicalProcessors()) {
      break;
    }
    krylov::ColoredSSORPrec<SparseMatrix<Scalar, int, int>> prec(C, Scalar(0.7));
    auto y = OutType::Zero(C.Rows(), 1);
    prec(x, y);
    auto yp = OutType::Zero(C.Rows(), 1);
    ParallelBarrier barrier(numWorkers);
    auto success = ParallelFor(numWorkers, [&](int workerId) {
      int rBegin = workerId * ((C.Rows() + numWorkers - 1) / numWorkers);
      int rEnd = std::min(C.Rows(), (workerId + 1) * ((C.Rows() + numWorkers - 1) / numWorkers));
      auto workerBarrier = barrier;
      prec.ConcurrentSolve(x, yp, {workerId, numWorkers, rBegin, rEnd, workerBarrier});
    });
    if (success) {
      EXPECT_TRUE(mochi::test::NearEqualMatrices(y, yp, tol));
      countParallel += 1;
    }
  }
  EXPECT_GT(countParallel, 0);
  //
  {
    auto BSpC = ToBlockSparseMatrix<1>(C);
    countParallel = 0;
    for (auto numWorkers : {1, 2, 3, 4, 5, 6}) {
      if (numWorkers > TaskScheduler::GetNumSupportedLogicalProcessors()) {
        break;
      }
      krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, 1, int, int>> prec(BSpC, Scalar(0.7));
      auto y = OutType::Zero(C.Rows(), 1);
      prec(x, y);
      auto yp = OutType::Zero(C.Rows(), 1);
      ParallelBarrier barrier(numWorkers);
      auto success = ParallelFor(numWorkers, [&](int workerId) {
        int rBegin = workerId * ((C.Rows() + numWorkers - 1) / numWorkers);
        int rEnd = std::min(C.Rows(), (workerId + 1) * ((C.Rows() + numWorkers - 1) / numWorkers));
        auto workerBarrier = barrier;
        prec.ConcurrentSolve(x, yp, {workerId, numWorkers, rBegin, rEnd, workerBarrier});
      });
      if (success) {
        EXPECT_TRUE(mochi::test::NearEqualMatrices(y, yp, tol));
        countParallel += 1;
      }
    }
    EXPECT_GT(countParallel, 0);
  }
  {
    constexpr int kBlockSize = 2;
    auto BSpC = ToBlockSparseMatrix<2>(C);
    int numWorkers = 3;
    if (numWorkers <= TaskScheduler::GetNumSupportedLogicalProcessors()) {
      krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, 2, int, int>> prec(BSpC, Scalar(0.7));
      auto y = OutType::Zero(C.Rows(), 1);
      prec(x, y);
      auto yp = OutType::Zero(C.Rows(), 1);
      ParallelBarrier barrier(numWorkers);
      auto success = ParallelFor(numWorkers, [&](int workerId) {
        int rBegin = workerId * ((BSpC.BlockRows() + numWorkers - 1) / numWorkers) * kBlockSize;
        int rEnd = std::min(
            C.Rows(),
            (workerId + 1) * ((BSpC.BlockRows() + numWorkers - 1) / numWorkers) * kBlockSize);
        auto workerBarrier = barrier;
        prec.ConcurrentSolve(x, yp, {workerId, numWorkers, rBegin, rEnd, workerBarrier});
      });
      if (success) {
        EXPECT_TRUE(mochi::test::NearEqualMatrices(y, yp, tol));
      }
    }
  }
}

template <typename Scalar, typename InType, typename OutType, int kBlockSize>
static void Example4() {
  //
  int nx = 13, ny = 15, nz = 17;
  auto C = mochi::test::MakeBlockSparseMatrix<Scalar, kBlockSize>(nx, ny, nz);
  auto CSp = ToSparseMatrix(C);
  InType x(C.Rows(), 1);
  x.SetRandom(123);
  auto const tol = Scalar(0.000001);
  int count = 0;
  //
  for (auto numWorkers : {1, 2, 3, 4, 5, 6}) {
    if (numWorkers > TaskScheduler::GetNumSupportedLogicalProcessors()) {
      break;
    }
    krylov::ColoredSSORPrec<SparseMatrix<Scalar, int, int>> prec(CSp);
    auto y = OutType::Zero(C.Rows(), 1);
    prec(x, y);
    auto yp = OutType::Zero(C.Rows(), 1);
    ParallelBarrier barrier(numWorkers);
    auto success = ParallelFor(numWorkers, [&](int workerId) {
      int rBegin = workerId * ((C.Rows() + numWorkers - 1) / numWorkers);
      int rEnd = std::min(C.Rows(), (workerId + 1) * ((C.Rows() + numWorkers - 1) / numWorkers));
      auto workerBarrier = barrier;
      prec.ConcurrentSolve(x, yp, {workerId, numWorkers, rBegin, rEnd, workerBarrier});
    });
    if (success) {
      EXPECT_TRUE(mochi::test::NearEqualMatrices(y, yp, tol));
      count += 1;
    }
  }
  EXPECT_GT(count, 0);
  //
  count = 0;
  for (auto numWorkers : {1, 2, 3, 4, 5, 6}) {
    if (numWorkers > TaskScheduler::GetNumSupportedLogicalProcessors()) {
      break;
    }
    krylov::ColoredSSORPrec<BlockSparseMatrix<Scalar, kBlockSize, int, int>> prec(C);
    auto y = OutType::Zero(C.Rows(), 1);
    prec(x, y);
    auto yp = OutType::Zero(C.Rows(), 1);
    ParallelBarrier barrier(numWorkers);
    auto success = ParallelFor(numWorkers, [&](int workerId) {
      int rBegin = workerId * ((C.BlockRows() + numWorkers - 1) / numWorkers) * kBlockSize;
      int rEnd = std::min(
          C.Rows(), (workerId + 1) * ((C.BlockRows() + numWorkers - 1) / numWorkers) * kBlockSize);
      auto workerBarrier = barrier;
      prec.ConcurrentSolve(x, yp, {workerId, numWorkers, rBegin, rEnd, workerBarrier});
    });
    if (success) {
      EXPECT_TRUE(mochi::test::NearEqualMatrices(y, yp, tol));
      count += 1;
    }
  }
  EXPECT_GT(count, 0);
}

// Workaround for a really bizarre compiler bug only seen with VS2022 on Windows using MSBuild
namespace {
class ColoredSSORPrecTest : public testing::Test {};
} // namespace

TEST_F(ColoredSSORPrecTest, Example1) {
  using ColMat = Matrix<Scalar>;
  using RowMat = RowMatrix<Scalar>;
  Example1<Scalar, ColMat, ColMat>();
  Example1<Scalar, RowMat, RowMat>();
  Example1<Scalar, RowMat, ColMat>();
  Example1<Scalar, ColMat, RowMat>();
}

TEST(ColoredSSORPrec, Example2) {
  using ColType = ColumnVector<Scalar>;
  Example2<Scalar, ColType, ColType>();
}

TEST(ColoredSSORPrec, Example3) {
  using ColType = ColumnVector<Scalar>;
  if (TaskScheduler::GetNumSupportedLogicalProcessors() >= 1) {
    Example3<Scalar, ColType, ColType>();
  }
}

TEST(ColoredSSORPrec, Example4) {
  using ColType = ColumnVector<Scalar>;
  if (TaskScheduler::GetNumSupportedLogicalProcessors() >= 1) {
    Example4<Scalar, ColType, ColType, 1>();
    Example4<Scalar, ColType, ColType, 2>();
    Example4<Scalar, ColType, ColType, 3>();
    Example4<Scalar, ColType, ColType, 4>();
  }
}
