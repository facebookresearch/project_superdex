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

#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/time.h>

#include <gtest/gtest.h>

using namespace mochi;

static void TestParallelDot(bool singleThreadedMode) {
  int const numThreads = TaskScheduler::GetNumSupportedLogicalProcessors();
  TaskScheduler scheduler(numThreads);
  scheduler.SetGlobalSingleThreadedMode(singleThreadedMode);

  for (int numWorkers : {1, 2, numThreads, 2 * numThreads}) {
    EXPECT_GT(numWorkers, 0);
    TaskSemaphore sem1(numWorkers), sem2(numWorkers);
    auto areWorkersReady = [=](TimePoint timeoutTime) {
      sem1.Done();
      while (!sem1.IsDone() && (Timer::Now() < timeoutTime)) {
      }

      if (!sem1.IsDone()) {
        sem1.Add(1);
      } else {
        sem2.Done();
      }

      while (sem1.IsDone() && !sem2.IsDone()) {
      }
      return static_cast<bool>(sem1.IsDone());
    };

    krylov::ParallelDot<real> parDot(
        numWorkers + 3); // Overallocate workers to test 'ReduceNumWorkers'.
    ParallelBarrier barrier(numWorkers);
    krylov::UsualDot dot = {};
    TaskSemaphore sem(numWorkers);
    ColumnVector<real> x(2 * numWorkers);
    ColumnVector<real> y(2 * numWorkers);
    auto const timeoutTime = Timer::Now() +
        TimeSpanFromSeconds(/* Large to increase likelihood of success */
                            0.01);
    for (int workerId = 0; workerId < numWorkers; ++workerId) {
      scheduler.AddTask("ParallelDot", [&, sem, parDot, barrier, workerId]() mutable {
        int const startRow = 2 * workerId;
        int const endRow = 2 * (workerId + 1);
        if (areWorkersReady(timeoutTime)) {
          parDot.ReduceNumWorkers(numWorkers, workerId == 0);
          for (int iter = 0; iter < 5; ++iter) {
            for (int r = startRow; r < endRow; ++r) {
              x[r] = 1_r;
              y[r] = real(iter);
            }
            auto result = parDot.Dot(dot, x, y, startRow, endRow, workerId);
            EXPECT_NEAR_EQ(static_cast<real>(2 * numWorkers * iter), result);
          }

          if (numWorkers > 1) {
            barrier.Wait();
            int const reducedNumWorkers = numWorkers - 1;
            if (workerId < reducedNumWorkers) {
              parDot.ReduceNumWorkers(reducedNumWorkers, workerId == 0);
              for (int r = startRow; r < endRow; ++r) {
                x[r] = 1_r;
                y[r] = 1_r;
              }
              auto result = parDot.Dot(dot, x, y, startRow, endRow, workerId);
              EXPECT_NEAR_EQ(static_cast<real>(2 * reducedNumWorkers), result);
            }
          }
        }
        sem.Done();
      });
    }
    sem.Wait();
  }
}

TEST(TensorFunctions, ParallelDot) {
  TestParallelDot(/*singleThreadedMode*/ true);
  TestParallelDot(/*singleThreadedMode*/ false);
}
