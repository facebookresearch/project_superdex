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

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/time.h>

#include <gtest/gtest.h>
#include <marl/conditionvariable.h> // Only required for use of TaskConditionVariable
#include <marl/mutex.h> // Only required for use of TaskMutex and TaskLock

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using namespace mochi;

static void TestSingleTask(TaskScheduler& scheduler) {
  std::atomic<int> counter = 0;
  TaskSemaphore sem(1);
  auto const callingThreadId = std::this_thread::get_id();
  bool const shouldRunOnCallingThread = (scheduler.GetNumThreads() == 0);
  scheduler.AddTask("Test", [&, sem]() {
    if (shouldRunOnCallingThread) {
      EXPECT_EQ(callingThreadId, std::this_thread::get_id());
    }
    ++counter;
    sem.Done();
  });
  sem.Wait();
  EXPECT_EQ(1, counter.load());
}

static void TestMultipleTasks(TaskScheduler& threadPool) {
  std::atomic<int> counter = 0;

  auto Increment = [&](int n) {
    for (int i = 0; i < n; ++i) {
      ++counter;
    }
  };

  TaskSemaphore sem;
  threadPool.AddTask(sem, "Test", [&Increment]() { Increment(10); });
  threadPool.AddTask(sem, "Test", [&Increment]() { Increment(100); });
  threadPool.AddTask(sem, "Test", [&Increment]() { Increment(1000); });
  threadPool.AddTask(sem, "Test", [&Increment]() { Increment(10000); });
  sem.Wait();
  EXPECT_EQ(11110, counter.load());
}

static void TestTasksWithinTasks(TaskScheduler& scheduler) {
  std::string message;
  TaskSemaphore sem(1);

  auto job3 = [&, sem]() {
    message += " worked!";
    sem.Done();
  };

  auto job2 = [&, sem]() {
    message += " It";
    scheduler.AddTask("job3", job3);
  };

  auto job1 = [&]() {
    message += "Cool!";
    scheduler.AddTask("job2", job2);
  };

  // Start the chain of jobs
  scheduler.AddTask("job1", job1);

  // Wait for the last job in the chain
  sem.Wait();

  EXPECT_STREQ("Cool! It worked!", message.c_str());
}

static void TestTaskSemaphore(TaskScheduler& scheduler) {
  // Default
  {
    TaskSemaphore sem;
    EXPECT_TRUE(sem.IsDone());
    sem.Wait(); // instant
  }

  // Initialize count
  {
    TaskSemaphore sem(4);
    EXPECT_FALSE(sem.IsDone());
    sem.Done();
    EXPECT_FALSE(sem.IsDone());
    sem.Done();
    EXPECT_FALSE(sem.IsDone());
    EXPECT_EQ(false, sem.WaitFor(TimeSpanFromSeconds(0.01))); // will time out
    sem.Done(2);
    EXPECT_TRUE(sem.IsDone());
    sem.Wait(); // instant
  }

  // Add count
  {
    TaskSemaphore sem(1);
    EXPECT_FALSE(sem.IsDone());
    sem.Add(3);
    EXPECT_FALSE(sem.IsDone());
    sem.Done(2);
    EXPECT_FALSE(sem.IsDone());
    sem.Done();
    EXPECT_FALSE(sem.IsDone());
    EXPECT_EQ(false, sem.WaitFor(TimeSpanFromSeconds(0.0001))); // will time out
    sem.Done();
    EXPECT_TRUE(sem.IsDone());
    sem.Wait(); // instant
  }

  // Wait for empty tasks
  {
    int constexpr kCount = 10;
    TaskSemaphore sem(kCount);
    for (int i = 0; i < kCount; ++i) {
      scheduler.AddTask("TestTaskSemaphore", [sem]() { sem.Done(); });
    }
    sem.Wait();
  }

  // Wait with timeout
  {
    TaskSemaphore startSem(1);
    TaskSemaphore stopSem(1);
    scheduler.AddTask("TestTaskSemaphore", [=]() {
      startSem.Wait(); // Wait for the main thread to tell us to start
      stopSem.Done(); // Tell the main thread we're done
    });

    // Wait and fail because of timeout
    EXPECT_EQ(false, stopSem.WaitFor(TimeSpanFromSeconds(0.01)));

    // Tell the task to stop waiting
    startSem.Done();

    // Wait and expect success. Use a very generous timeout to be safe
    EXPECT_EQ(true, stopSem.WaitFor(TimeSpanFromSeconds(1.0)));
  }

  // Busy wait
  if (scheduler.GetNumThreads() > 1) {
    // If we only added one task, then there is a chance that it could be enqueued onto the calling
    // thread. That would result in a deadlock because the calling thread will be busy waiting. Even
    // if we added multiple tasks, there is still some chance that they could all be enqueued onto
    // the calling thread, because Marl uses a `spinningWorkers` queue which could contain duplicate
    // entries. Therefore, we enqueue more taks that worker threads. This will exhaust the
    // `spinningWorkers` queue and force the scheduler to fall back on round-robin task assignment.
    // If we enqueue enough tasks, then we can ensure that other worker(s) will be awake to do the
    // work, and to steal any tasks that are (unfortunately) assigned to this thread.
    int count = 2 * scheduler.GetNumThreads();
    TaskSemaphore sem(count);
    for (int i = 0; i < count; ++i) {
      scheduler.AddTask("TestTaskSemaphore", [sem]() { sem.Done(); });
    }
    while (!sem.IsDone()) {
      // busy wait
    }
  }
}

static void TestTaskMutex(TaskScheduler& scheduler) {
  TaskMutex mutex;
  TaskSemaphore sem(10);
  std::vector<int> numbers;

  // Start 10 tasks which will each add 1000 numbers to the same
  // std::vector guareded by a mutex. This is a terrible use of the
  // task system. Don't do this is real code!
  for (int i = 0; i < 10; ++i) {
    scheduler.AddTask("Test", [i, sem, &mutex, &numbers]() {
      int const beginIndex = i * 1000;
      int const endIndex = beginIndex + 1000;
      for (int j = beginIndex; j < endIndex; ++j) {
        TaskLock lock(mutex);
        numbers.push_back(j);
      }
      sem.Done();
    });
  }

  sem.Wait();

  // Verify that we got all 10,000 unique numbers
  EXPECT_EQ(10000, numbers.size());
  std::sort(numbers.begin(), numbers.end());
  for (int i = 0; i < 10000; ++i) {
    EXPECT_EQ(i, numbers[i]);
  }
}

static void TestTaskConditionVariable(TaskScheduler& scheduler) {
  // This is similar to TestTaskMutex, but instead of waiting for a
  // TaskSemaphore, we will use a TestConditionVariable to wait until
  // the 'numbers' array is the right size.
  auto mutex = std::make_shared<TaskMutex>();
  auto cv = std::make_shared<TaskConditionVariable>();
  std::vector<int> numbers;
  for (int i = 0; i < 10; ++i) {
    // Capture cv by value to increment the reference count. This ensures that the cv will last long
    // enough for the task to call cv->notify_all(), even if the calling function has returned. That
    // is possible because the calling function can acquires the lock and checks numbers.size()
    // immediately after the last value was pushed and before the last cv->notify_all.
    scheduler.AddTask("Test", [i, mutex, cv, &numbers]() {
      int const beginIndex = i * 1000;
      int const endIndex = beginIndex + 1000;
      for (int j = beginIndex; j < endIndex; ++j) {
        TaskLock lock(*mutex);
        numbers.push_back(j);
      }
      cv->notify_all();
    });
  }

  // Lock the same mutex that guards the numbers array.
  // Wait for the array size to grow to 10 * 1000.
  TaskLock lock(*mutex);
  cv->wait(lock, [&numbers]() { return numbers.size() == 10000; });

  // Verify that we got all 10,000 unique numbers
  EXPECT_EQ(10000, numbers.size());
  std::sort(numbers.begin(), numbers.end());
  for (int i = 0; i < 10000; ++i) {
    EXPECT_EQ(i, numbers[i]);
  }
}

static void TestBatchEnqueueOnAvailableWorkers(TaskScheduler& scheduler) {
  int const numSchedulerThreads = TaskScheduler::StaticGetNumThreads();
  int const numOtherThreads = TaskScheduler::StaticGetNumOtherThreads();
  bool const shouldRunOnCallingThread = (scheduler.GetNumThreads() == 0);
  auto const callingThreadId = std::this_thread::get_id();
  for (bool includeSelf : {false, true}) {
    for (int numTargetWorkers : {1, 2, numSchedulerThreads, 2 * numSchedulerThreads}) {
      for (int numMinWorkers : {1, numTargetWorkers}) {
        if (numTargetWorkers == 0) {
          // Nothing to test. Calling BatchEnqueueOnAvailableWorkers with 0 target workers is
          // illegal.
          continue;
        }

        TaskSemaphore sem;
        std::atomic<int> counter = 0;
        TaskScheduler::BatchTaskFn task = [&, sem](int workerId, int numWorkers) {
          EXPECT_TRUE(numWorkers >= numMinWorkers && numWorkers <= numTargetWorkers);
          EXPECT_TRUE(workerId >= 0 && workerId < numWorkers);
          if (shouldRunOnCallingThread) {
            EXPECT_TRUE(workerId == 0 && std::this_thread::get_id() == callingThreadId);
          }
          EXPECT_EQ(workerId == 0 && includeSelf, std::this_thread::get_id() == callingThreadId);
          counter++;
          sem.Done();
        };

        int const numWorkers = scheduler.BatchEnqueueOnAvailableWorkers(
            sem, std::move(task), numMinWorkers, numTargetWorkers, includeSelf);
        // Always call sem.Wait() even if the batch enqueu failed. That way, we can prove whether or
        // not tasks were executed, and ensure that they can never continue exeucint after this
        // point.
        sem.Wait();
        if (numWorkers >= numMinWorkers) {
          EXPECT_LE(numWorkers, numTargetWorkers);
          EXPECT_EQ(numWorkers, counter);
          EXPECT_LE(numWorkers, includeSelf ? numOtherThreads + 1 : numOtherThreads);
          if (shouldRunOnCallingThread) {
            int const numOtherWorkers = includeSelf ? (numWorkers - 1) : numWorkers;
            EXPECT_LT(numOtherWorkers, 1);
          }
        } else {
          EXPECT_EQ(0, numWorkers);
          EXPECT_EQ(0, counter);
        }
      }
    }
  }
}

static void TestUtil_Schedule() {
  MOCHI_PROFILE_SCOPE();
  auto* scheduler = TaskScheduler::TryGet();
  int numOtherThreads = scheduler ? scheduler->GetNumOtherThreads() : 0;

  // If there are no other threads available, then expect the task to run immediately
  // on the calling thread.
  auto mainThreadId = std::this_thread::get_id();
  auto checkThreadId = [&]() {
    if (numOtherThreads == 0) {
      EXPECT_EQ(std::this_thread::get_id(), mainThreadId);
    }
  };

  std::atomic<int> counter = 0;
  TaskSemaphore sem(1);
  mochi::Schedule("Test", [&, sem]() {
    checkThreadId();
    ++counter;
    sem.Done();
  });
  if (numOtherThreads == 0) {
    // Should have finished already
    EXPECT_TRUE(sem.IsDone());
    EXPECT_EQ(1, counter.load());
  }
  sem.Wait();
  EXPECT_EQ(1, counter.load());

  // This overload uses the TaskSemaphore that we provide
  counter = 0;
  mochi::Schedule(sem, "Test", [&]() {
    checkThreadId();
    ++counter;
  });
  mochi::Schedule(sem, "Test", [&]() {
    checkThreadId();
    ++counter;
  });
  mochi::Schedule(sem, "Test", [&]() {
    checkThreadId();
    ++counter;
  });
  mochi::Schedule(sem, "Test", [&]() {
    checkThreadId();
    ++counter;
  });
  if (numOtherThreads == 0) {
    // Should have finished already
    EXPECT_TRUE(sem.IsDone());
    EXPECT_EQ(4, counter.load());
  }
  sem.Wait(); // Wait for all of them
  EXPECT_EQ(4, counter.load());
}

static void TestUtil_ParallelForN() {
  MOCHI_PROFILE_SCOPE();
  std::vector<int> data;
  auto runTest = [&](int count, int minPerTask) {
    data.clear();
    data.resize(count + 1, 0);
    data[count] = 12345; // sentinel value

    // Set each item of the data equal to its index for each
    // index in the range 0 to (count-1).
    ParallelForN("Test", count, minPerTask, [&](int i) { data[i] = i; });

    // Check each value
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(i, data[i]);
    }

    // Check for buffer overrun
    EXPECT_EQ(12345, data[count]);
  };

  runTest(50, 1);

  // Repeat with smaller counts (including cases where count < numThreads)
  for (int count = 0; count < 32; ++count) {
    for (int minPerTask = 1; minPerTask < 32; ++minPerTask) {
      runTest(count, minPerTask);
    }
  }
}

static void TestUtil_ParallelForEach() {
  MOCHI_PROFILE_SCOPE();
  std::vector<int> data;
  auto runTest = [&](int count, int minPerTask) {
    data.clear();
    data.resize(count + 1, 0);
    data[count] = 12345; // sentinel value

    // Set each item of the data equal to its index for each
    // index in the range 0 to (count-1).
    ParallelForN("Test", count, minPerTask, [&data](int i) { data[i] = i; });

    // Use ParallelForEach to double each value with a mochi::Span
    ParallelForEach(
        "Test", mochi::Span<int>(data.data(), count), minPerTask, [](int& x) { x *= 2; });

    // Check each value
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(2 * i, data[i]);
    }

    // Check for buffer overrun
    EXPECT_EQ(12345, data[count]);

    // Repeat, but this time pass the span as a tuple
    auto mySpan = mochi::Span<int>(data.data(), count);
    ParallelForEach("Test", std::tie(mySpan), minPerTask, [](int& x) { x *= 2; });
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(4 * i, data[i]);
    }
    EXPECT_EQ(12345, data[count]);

    // Now pass the std::vector to ParallelForEach, to double every element including the sentinel
    ParallelForEach("Test", data, minPerTask, [](int& x) { x *= 2; });
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(8 * i, data[i]);
    }
    EXPECT_EQ(12345 * 2, data[count]);

    // Repeat, but this time pass the vector as a tuple
    ParallelForEach("Test", std::tie(data), minPerTask, [](int& x) { x *= 2; });
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(16 * i, data[i]);
    }
    EXPECT_EQ(12345 * 4, data[count]);

    // If the count can be evenly cut in half, then tie two sub-spans so that we iterate
    // both halves in lock-step.
    if ((count % 2) == 0) {
      Span<int, int> const firstHalf{data.data(), count / 2};
      Span<int, int> const secondHalf{data.data() + count / 2, count / 2};
      ParallelForEach("Test", std::tie(firstHalf, secondHalf), minPerTask, [](int& x, int& y) {
        x *= 2;
        y *= 2;
      });
      for (int i = 0; i < count; ++i) {
        EXPECT_EQ(32 * i, data[i]);
      }
      EXPECT_EQ(12345 * 4, data[count]); // no overflow
    }

    // Now pass a const std::vector to ParallelForEach to calculate the sum (in a very inefficient
    // way)
    std::vector<int> const& cref = data;
    std::atomic<int> sum = 0;
    ParallelForEach("Test", cref, minPerTask, [&](int const& x) { sum += x; });
    int expectedSum = 0;
    for (int x : data) {
      expectedSum += x;
    }
    EXPECT_EQ(expectedSum, sum);
  };

  runTest(50, 1);

  // Repeat with smaller counts (including cases where count < numThreads)
  for (int count = 0; count < 16; ++count) {
    for (int minPerTask = 1; minPerTask < 16; ++minPerTask) {
      runTest(count, minPerTask);
    }
  }
}

static void TestUtil_ParallelForRange() {
  MOCHI_PROFILE_SCOPE();
  auto const* scheduler = TaskScheduler::TryGet();
  std::vector<int> data;
  auto runTest = [&](int rangeBegin, int rangeEnd, int minPerTask, int maxPerTask) {
    int count = rangeEnd - rangeBegin;
    data.clear();
    data.resize(count + 1, 0);
    data[count] = 12345; // sentinel value

    // Set each item of the data equal to its index for each
    // index in the range 0 to (count-1).
    auto const callingThreadId = std::this_thread::get_id();
    bool const shouldRunOnCallingThread = !scheduler || (scheduler->GetNumThreads() == 0);
    ParallelForRange("Test", rangeBegin, rangeEnd, minPerTask, maxPerTask, [&](int begin, int end) {
      if (shouldRunOnCallingThread) {
        EXPECT_EQ(callingThreadId, std::this_thread::get_id());
      }
      for (int i = begin; i < end; ++i) {
        int dataIndex = i - rangeBegin;
        data[dataIndex] = dataIndex;
      }
    });

    // Check each value
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(i, data[i]);
    }

    // Check for buffer overrun
    EXPECT_EQ(12345, data[count]);
  };

  int const kTestCaseSizes[] = {0, 1, 15, 16, 63, 64, 1000};
  int const kTestCaseOffsets[] = {0, 1, 2};
  int const kTestCaseItemsPerTask[] = {1, 3, 7, 8, 15, 16};

  for (int rangeSize : kTestCaseSizes) {
    for (int rangeOffset : kTestCaseOffsets) {
      for (int i = 0; i < (int)std::size(kTestCaseItemsPerTask); ++i) {
        for (int j = i; j < (int)std::size(kTestCaseItemsPerTask); ++j) {
          int minPerTask = kTestCaseItemsPerTask[i];
          int maxPerTask = kTestCaseItemsPerTask[j];
          runTest(rangeOffset, rangeOffset + rangeSize, minPerTask, maxPerTask);
        }
      }
    }
  }
}

static void TestUtil_ParallelSort() {
  MOCHI_PROFILE_SCOPE();
  // Create a large data set
  auto gen = RandomGenerator(123UL); // arbitrary fixed seed
  std::uniform_int_distribution dist(
      std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
  static constexpr size_t kNumValues = 500; // arbitrary large size
  std::vector<int> randomData(kNumValues);
  std::generate(randomData.begin(), randomData.end(), [&]() { return dist(gen); });

  // Repeat this test for various data sizes
  auto runTest = [&](size_t len) {
    // Sort using ParallelSort
    std::vector<int> sorted(randomData.begin(), randomData.begin() + len);
    ParallelSort(sorted.begin(), sorted.end());
    EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));

    // Repeat with a custom comparison function
    auto compare = [](int a, int b) { return b < a; };
    sorted.assign(randomData.begin(), randomData.begin() + len);
    ParallelSort(sorted.begin(), sorted.end(), compare);
    EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end(), compare));
  };

  static constexpr size_t kTestSizes[] = {0, 1, 2, 3, 16, 32, 64, 128, 255, 256, 257, kNumValues};
  for (size_t sz : kTestSizes) {
    runTest(sz);
  }
}

static void TestPendingJobsAtTimeOfShutdown(
    int numThreads,
    bool globalSingleThreadedMode,
    bool localSingleThreadedMode) {
  MOCHI_PROFILE_SCOPE();
  std::atomic<int> counter = 0;
  constexpr int kNumJobs = 5;

  {
    TaskScheduler scheduler(numThreads);
    scheduler.SetGlobalSingleThreadedMode(globalSingleThreadedMode);
    if (localSingleThreadedMode) {
      TaskScheduler::PushLocalSingleThreadedMode();
    }

    auto func = [&]() {
      // Don't do this normally! It will put the whole worker thread
      // to sleep, not just this one task!
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      ++counter;
    };

    // Start some jobs that will take non-zero time
    for (int i = 0; i < kNumJobs; ++i) {
      scheduler.AddTask("Test", func);
    }

    // Let the TaskScheduler go out of scope
  }

  // Expect that all tasks completed
  EXPECT_EQ(kNumJobs, counter.load());
}

static void TestUtilityFunctions() {
  MOCHI_PROFILE_SCOPE();
  TestUtil_Schedule();
  TestUtil_ParallelForN();
  TestUtil_ParallelForEach();
  TestUtil_ParallelForRange();
  TestUtil_ParallelSort();
}

static void TestLocalSingleThreadedMode(TaskScheduler& scheduler, bool expectLocalSingleThreaded) {
  EXPECT_EQ(expectLocalSingleThreaded, TaskScheduler::IsLocalSingleThreaded());
  TaskScheduler::PushLocalSingleThreadedMode();
  EXPECT_TRUE(TaskScheduler::IsLocalSingleThreaded());
  EXPECT_EQ(0, scheduler.GetNumThreads());
  TaskScheduler::PushLocalSingleThreadedMode();
  TaskScheduler::PopLocalSingleThreadedMode();
  EXPECT_TRUE(TaskScheduler::IsLocalSingleThreaded());
  TaskScheduler::PopLocalSingleThreadedMode();
  EXPECT_EQ(expectLocalSingleThreaded, TaskScheduler::IsLocalSingleThreaded());
}

static void TestTaskScheduler(TaskScheduler& scheduler, bool localSingleThreadedMode) {
  MOCHI_PROFILE_SCOPE();
  // The scheduler should already be bound to the calling thread
  EXPECT_EQ(&scheduler, TaskScheduler::TryGet());

  // Set local single-threaded mode.
  if (localSingleThreadedMode) {
    TaskScheduler::PushLocalSingleThreadedMode();
  }

  // Basic functionality:
  TestLocalSingleThreadedMode(scheduler, localSingleThreadedMode);
  TestSingleTask(scheduler);
  TestMultipleTasks(scheduler);
  TestTasksWithinTasks(scheduler);
  TestTaskSemaphore(scheduler);
  TestTaskMutex(scheduler);
  TestTaskConditionVariable(scheduler);
  TestBatchEnqueueOnAvailableWorkers(scheduler);

  // Utility functions built on TaskScheduler
  TestUtilityFunctions();

  // Reset local single-threaded mode.
  if (localSingleThreadedMode) {
    TaskScheduler::PopLocalSingleThreadedMode();
  }
}

static void TestTaskScheduler(
    uint32_t numThreads,
    bool globalSingleThreadedMode,
    bool localSingleThreadedMode) {
  {
    TaskScheduler scheduler(numThreads);
    EXPECT_EQ(numThreads, scheduler.GetNumThreads());

    scheduler.SetGlobalSingleThreadedMode(globalSingleThreadedMode);
    if (globalSingleThreadedMode) {
      EXPECT_EQ(0, scheduler.GetNumThreads());
    }

    // Run a variety of tests from this thread which is already bound to the scheduler,
    // but is not a worker thread. Synchronization primitives like TaskSemaphore, TaskMutex, and
    // TaskConditionVariable may put this thread to sleep while waiting for tasks.
    TestTaskScheduler(scheduler, localSingleThreadedMode);

    // Repeat the test from inside a different OS thread. Again, the synchronization primitives
    // may put the other thread to sleep while waiting for completion of tasks.
    {
      std::thread otherThread([&]() {
        scheduler.BindThisThread();
        TestTaskScheduler(scheduler, localSingleThreadedMode);
        scheduler.UnbindThisThread();
      });
      otherThread.join();
    }

    // Repeat the tests from inside of a task. In this context, the synchronization primitives
    // within TestTaskScheduler will NOT put the task's worker thread to sleep. They will simply
    // cause the coroutine to get rescheduled.
    TaskSemaphore sem;
    scheduler.AddTask(
        sem, "TestTaskScheduler", [&]() { TestTaskScheduler(scheduler, localSingleThreadedMode); });
    sem.Wait();
  }

  TestPendingJobsAtTimeOfShutdown(numThreads, globalSingleThreadedMode, localSingleThreadedMode);
}

static void TestTaskScheduler(uint32_t numThreads) {
  bool const wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  if (numThreads > TaskScheduler::GetNumSupportedLogicalProcessors()) {
    // Disable warnings about using more threads than can run concurrently on the device.
    EnableLogChannel(LogChannel::Warning, false);
  }
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, wasWarningEnabled));

  TestTaskScheduler(numThreads, false, false);
  TestTaskScheduler(numThreads, true, false);
  TestTaskScheduler(numThreads, false, true);
  TestTaskScheduler(numThreads, true, true);
}

static void TestParallelBarrier(bool globalSingleThreadedMode, bool localSingleThreadedMode) {
  int const numThreads = TaskScheduler::GetNumSupportedLogicalProcessors();
  TaskScheduler scheduler(numThreads);
  scheduler.SetGlobalSingleThreadedMode(globalSingleThreadedMode);
  if (localSingleThreadedMode) {
    TaskScheduler::PushLocalSingleThreadedMode();
  }

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

    ParallelBarrier barrier(numWorkers + 3); // Overallocate workers to test 'ReduceNumWorkers'.
    TaskSemaphore sem(numWorkers);
    auto const timeoutTime = Timer::Now() +
        TimeSpanFromSeconds(/* Large to increase likelihood of success */
                            0.01);
    for (int i = 0; i < numWorkers; ++i) {
      scheduler.AddTask("ParallelBarrier", [&, sem, barrier, i]() mutable {
        if (areWorkersReady(timeoutTime)) {
          barrier.ReduceNumWorkers(numWorkers, i == 0);
          Timer timer = {};
          for (int iter = 0; iter < 5; ++iter) {
            barrier.Wait();
            timer.Reset();
            while (ToMilliseconds(timer.GetElapsed()) < 1.0) {
            }
            barrier.Wait();
          }
        }
        sem.Done();
      });
    }
    sem.Wait();
  }
}

TEST(TaskScheduler, GetNumSupportedPhysicalProcessors) {
  int numPhysicalProcessors = TaskScheduler::GetNumSupportedPhysicalProcessors();
  int numLogicalProcessors = TaskScheduler::GetNumSupportedLogicalProcessors();
  EXPECT_GE(numPhysicalProcessors, 1); // At least 1
  EXPECT_GE(numLogicalProcessors, numPhysicalProcessors);
}

TEST(TaskScheduler, GetNumSupportedLogicalProcessors) {
  int numLogicalProcessors = TaskScheduler::GetNumSupportedLogicalProcessors();
  EXPECT_GE(numLogicalProcessors, 1); // At least 1
}

TEST(TaskScheduler, UnboundUtils) {
  // The standalone utility functions are expected to operate with or without a TaskScheduler,
  // falling back on single-threaded implementations.
  EXPECT_EQ((TaskScheduler*)nullptr, TaskScheduler::TryGet());
  TestUtilityFunctions();
}

TEST(TaskScheduler, NumThreads0) {
  TestTaskScheduler(0);
}

TEST(TaskScheduler, NumThreads1) {
  TestTaskScheduler(1);
}

TEST(TaskScheduler, NumThreads2) {
  TestTaskScheduler(2);
}

TEST(TaskScheduler, NumThreads3) {
  TestTaskScheduler(3);
}

TEST(TaskScheduler, IsCurrentThreadAWorker) {
  TaskScheduler scheduler(1);
  EXPECT_FALSE(TaskScheduler::IsCurrentThreadAWorker());

  std::thread controlThread([&]() {
    scheduler.BindThisThread();
    EXPECT_FALSE(TaskScheduler::IsCurrentThreadAWorker());
    scheduler.UnbindThisThread();
  });
  controlThread.join();

  TaskSemaphore sem;
  scheduler.AddTask(sem, "IsCurrentThreadAWorker", []() {
    EXPECT_TRUE(TaskScheduler::IsCurrentThreadAWorker());
  });
  sem.Wait();
}

TEST(TaskScheduler, GetNumOtherThreads) {
  bool const wasWarningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  if (TaskScheduler::GetNumSupportedLogicalProcessors() < 3) {
    // Disable warnings about using more threads than can run concurrently on the device.
    EnableLogChannel(LogChannel::Warning, false);
  }
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, wasWarningEnabled));

  // Zero workers
  {
    TaskScheduler scheduler(0);
    EXPECT_EQ(0, scheduler.GetNumThreads());
    EXPECT_EQ(0, scheduler.GetNumOtherThreads());
  }

  // Global single threaded
  {
    TaskScheduler scheduler(1);
    EXPECT_EQ(1, scheduler.GetNumThreads());
    EXPECT_EQ(1, scheduler.GetNumOtherThreads());
    scheduler.SetGlobalSingleThreadedMode(true);
    EXPECT_EQ(0, scheduler.GetNumThreads());
    EXPECT_EQ(0, scheduler.GetNumOtherThreads());
  }

  // Local single threaded
  {
    TaskScheduler scheduler(1);
    EXPECT_EQ(1, scheduler.GetNumThreads());
    EXPECT_EQ(1, scheduler.GetNumOtherThreads());
    scheduler.PushLocalSingleThreadedMode();
    EXPECT_EQ(0, scheduler.GetNumThreads());
    EXPECT_EQ(0, scheduler.GetNumOtherThreads());
    scheduler.PopLocalSingleThreadedMode();
    EXPECT_EQ(1, scheduler.GetNumThreads());
    EXPECT_EQ(1, scheduler.GetNumOtherThreads());
  }

  // Multi-threaded
  for (int numThreads : {1, 2, 3}) {
    TaskScheduler scheduler(numThreads);
    EXPECT_EQ(numThreads, scheduler.GetNumThreads());
    EXPECT_EQ(numThreads, scheduler.GetNumOtherThreads());
    auto osThread = std::thread([&]() {
      EXPECT_EQ(numThreads, scheduler.GetNumThreads());
      EXPECT_EQ(numThreads, scheduler.GetNumOtherThreads());
    });
    osThread.join();
    TaskSemaphore sem;
    for (int i = 0; i < 8; ++i) {
      scheduler.AddTask(sem, "test", [&]() {
        EXPECT_EQ(numThreads, scheduler.GetNumThreads());
        EXPECT_EQ(numThreads - 1, scheduler.GetNumOtherThreads());
      });
    }
    sem.Wait();
  }
}

TEST(TaskScheduler, ParallelBarrier) {
  TestParallelBarrier(true, false);
  TestParallelBarrier(false, false);
  TestParallelBarrier(true, true);
  TestParallelBarrier(false, true);
}
