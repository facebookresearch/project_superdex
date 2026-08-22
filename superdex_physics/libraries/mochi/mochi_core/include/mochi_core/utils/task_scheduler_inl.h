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

#pragma once

#include "task_scheduler.h" // Reverse include for Intellisense

#include <algorithm>
#include <functional>
#include <iterator>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>

namespace mochi {

/**************************************************************************************************
  TaskScheduler
*/

template <typename FN>
MOCHI_FORCE_INLINE void TaskScheduler::AddTask(
    [[maybe_unused]] std::string_view debugNameStringLiteral,
    FN&& fn) {
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
  AddTaskVerboseProfile(debugNameStringLiteral, std::forward<FN>(fn));
#else
  bool const isSingleThreaded = (GetNumThreads() == 0);
  AddTaskNoProfile(
      [fn = std::forward<FN>(fn)]() mutable {
#if MOCHI_TASK_PROFILE_VERBOSITY != MOCHI_TASK_PROFILE_VERBOSITY_NONE
        MOCHI_PROFILE_SCOPE_N("RunTask");
#endif
        fn();
      },
      isSingleThreaded);
#endif
}

template <typename FN>
MOCHI_FORCE_INLINE void TaskScheduler::AddTask(
    TaskSemaphore sem,
    [[maybe_unused]] std::string_view debugNameStringLiteral,
    FN&& fn) {
  sem.Add(1);
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
  AddTaskVerboseProfile(
      debugNameStringLiteral, [sem = std::move(sem), fn = std::forward<FN>(fn)]() mutable {
        fn();
        sem.Done();
      });
#else
  bool const isSingleThreaded = (GetNumThreads() == 0);
  AddTaskNoProfile(
      [sem = std::move(sem), fn = std::forward<FN>(fn)]() mutable {
#if MOCHI_TASK_PROFILE_VERBOSITY != MOCHI_TASK_PROFILE_VERBOSITY_NONE
        MOCHI_PROFILE_SCOPE_N("RunTask");
#endif
        fn();
        sem.Done();
      },
      isSingleThreaded);
#endif
}

/**************************************************************************************************
  Utility Functions
*/

template <typename FN>
MOCHI_FORCE_INLINE void Schedule(std::string_view debugNameStringLiteral, FN&& fn) {
  auto* scheduler = TaskScheduler::TryGet();
  if (scheduler && (scheduler->GetNumOtherThreads() > 0)) {
    scheduler->AddTask(debugNameStringLiteral, std::forward<FN>(fn));
  } else { // Single threaded fallback
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
    MOCHI_PROFILE_SCOPE();
    MOCHI_PROFILE_LABEL(debugNameStringLiteral);
#endif
    fn();
  }
}

template <typename FN>
MOCHI_FORCE_INLINE void
Schedule(TaskSemaphore sem, std::string_view debugNameStringLiteral, FN&& fn) {
  auto* scheduler = TaskScheduler::TryGet();
  if (scheduler && (scheduler->GetNumOtherThreads() > 0)) {
    scheduler->AddTask(std::move(sem), debugNameStringLiteral, std::forward<FN>(fn));
  } else { // Single threaded fallback
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
    MOCHI_PROFILE_SCOPE();
    MOCHI_PROFILE_LABEL(debugNameStringLiteral);
#endif
    fn();
  }
}

template <typename FN>
void ParallelForRange(
    [[maybe_unused]] std::string_view debugName,
    int globalRangeBegin,
    int globalRangeEnd,
    int minPerTask,
    int maxPerTask,
    FN const& forRange) {
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
  MOCHI_PROFILE_SCOPE();
  MOCHI_PROFILE_LABEL_2("ParallelFor ", debugName);
#endif

  MOCHI_ASSERT_VERBOSE(globalRangeEnd >= globalRangeBegin);
  MOCHI_ASSERT_VERBOSE(minPerTask > 0);
  MOCHI_ASSERT_VERBOSE(maxPerTask >= minPerTask);
  auto* scheduler = TaskScheduler::TryGet();
  int numOtherThreads = scheduler ? scheduler->GetNumOtherThreads() : 0;
  if (numOtherThreads == 0) {
    // Single threaded fallback
    forRange(globalRangeBegin, globalRangeEnd);
    return;
  }

  // We have to decide how to subdivide the index range into tasks. If all threads are idle and
  // all callbacks are equally fast, then the optimal subdivision would split the work evenly with
  // one task per thread. However, some threads may already be busy or some callbacks may take
  // longer than others. In that case, an even subdivision would be sub-optimal because we could
  // end up waiting for one task long after the others finish.
  //
  // Therefore, lets shoot for two tasks per worker thread as a general target, unless we are forced
  // to do otherwise by minPerTask, maxPerTask, or count.
  int count = globalRangeEnd - globalRangeBegin;
  int numThreads = numOtherThreads + 1; // +1 for the current thread
  int numTasksTarget = std::min(count, 2 * numThreads);
  int numPerTask =
      numTasksTarget ? std::min(std::max(count / numTasksTarget, minPerTask), maxPerTask) : count;

  // Schedule tasks to handle each sub-range except the last one, which this thread will handle.
  int numTasks = numPerTask ? (count - 1) / numPerTask : 0;
  int localRangeBegin = globalRangeBegin;
  TaskSemaphore sem;
  if (numTasks > 0) {
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
    MOCHI_PROFILE_DESCRIPTION_F("%d items, %d tasks", count, numTasks);
#endif
    sem.Add(numTasks);
    for (int i = 0; i < numTasks; ++i) {
      int localRangeEnd = localRangeBegin + numPerTask;
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
      // TaskScheduler will wrap our std::function in another one, which will add custom formatted
      // profile labels.
      scheduler->AddTask(debugName, [sem, &forRange, localRangeBegin, localRangeEnd]() {
        forRange(localRangeBegin, localRangeEnd);
        sem.Done();
      });
#else
      scheduler->AddTaskNoProfile(
          [sem, &forRange, localRangeBegin, localRangeEnd]() {
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_LOW
            MOCHI_PROFILE_SCOPE_N("RunTask");
#endif
            forRange(localRangeBegin, localRangeEnd);
            sem.Done();
          },
          false);
#endif
      localRangeBegin = localRangeEnd;
    }
  }

  // If the user sets minPerTask to 1, then each item is probably quite expensive to compute. In
  // such cases, we really want each task to be processed in parallel on different worker threads.
  // Unfortunately, marl's task scheduling algorithm sometimes doubles them up on a single thread.
  // Hacky work-around: schedule dummy jobs to wake up more worker threads. They will then enter
  // the spin-for-work state and have the opportunity to steal work from the doubled up thread(s).
  if (minPerTask == 1) {
    scheduler->TryToWakeUpMoreWorkers(numTasks / 2); // Wake up more workers please
  }

  // Since this is a blocking call, we might as well do part of the work here.
  {
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
    MOCHI_PROFILE_SCOPE();
    MOCHI_PROFILE_LABEL(debugName);
#elif MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_LOW
    MOCHI_PROFILE_SCOPE_N("RunTask");
#endif
    forRange(localRangeBegin, globalRangeEnd);
  }

  if (numTasks > 0) {
    sem.Wait();
  }
}

template <typename FN>
MOCHI_FORCE_INLINE void
ParallelForN(std::string_view debugNameStringLiteral, int n, int minPerTask, FN const& forEach) {
  ParallelForRange(
      debugNameStringLiteral, 0, n, minPerTask, INT_MAX, [&](int rangeBegin, int rangeEnd) {
        for (int i = rangeBegin; i < rangeEnd; ++i) {
          forEach(i);
        }
      });
}

template <typename ContainerT, typename FN>
MOCHI_FORCE_INLINE void ParallelForEach(
    std::string_view debugNameStringLiteral,
    ContainerT&& container,
    int minPerTask,
    FN const& forEach) {
  ParallelForRange(
      debugNameStringLiteral,
      0,
      static_cast<int>(std::size(container)),
      minPerTask,
      INT_MAX,
      [&](int rangeBegin, int rangeEnd) {
        for (int i = rangeBegin; i < rangeEnd; ++i) {
          forEach(container[i]);
        }
      });
}

template <typename FN, typename... ContainerT>
MOCHI_FORCE_INLINE void ParallelForEach(
    std::string_view debugNameStringLiteral,
    std::tuple<ContainerT&...> containerTuple,
    int minPerTask,
    FN const& forEach) {
  size_t const itemCount = std::size(std::get<0>(containerTuple));
#if MOCHI_ASSERT_ENABLED
  auto checkSizes = [itemCount](ContainerT&... container) {
    MOCHI_ASSERT(((std::size(container) == itemCount) && ... && true), "Container size mismatch");
  };
  std::apply(checkSizes, containerTuple);
#endif // MOCHI_ASSERT_ENABLED
  ParallelForRange(
      debugNameStringLiteral,
      0,
      static_cast<int>(itemCount),
      minPerTask,
      INT_MAX,
      [&](int rangeBegin, int rangeEnd) {
        for (int i = rangeBegin; i < rangeEnd; ++i) {
          auto callForEach = [i, &forEach](ContainerT&... container) { forEach(container[i]...); };
          std::apply(callForEach, containerTuple);
        }
      });
}

template <class Iter, class Less>
inline void ParallelSort(Iter begin, Iter end, Less comp, int taskDepthCoutdown) {
  auto len = std::distance(begin, end);
  if ((len <= 256) || (taskDepthCoutdown == 0)) {
    // Do the rest single-threaded
    std::sort(begin, end, comp);
  } else {
    // Divide and conquer
    Iter middle = std::next(begin, len / 2);
    TaskSemaphore sem;
    Schedule(sem, "MergeSort", [=]() { ParallelSort(begin, middle, comp, taskDepthCoutdown - 1); });
    ParallelSort(middle, end, comp, taskDepthCoutdown - 1);
    sem.Wait();
    std::inplace_merge(begin, middle, end, comp);
  }
}

template <class Predicate>
MOCHI_FORCE_INLINE void BusyWaitFor(Predicate&& condition, TimeSpan yieldPeriod) {
  Timer timer;
  while (!condition()) {
    if (timer.GetElapsed() >= yieldPeriod) {
      std::this_thread::yield();
      timer.Reset();
    }
    MOCHI_NOP_50();
  }
}

} // namespace mochi
