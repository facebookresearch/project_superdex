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

#include <mochi_core/utils/basic_utils.h>

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
    TaskSemaphore const& sem,
    [[maybe_unused]] std::string_view debugNameStringLiteral,
    FN&& fn) {
  sem.Add(1);
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
  AddTaskVerboseProfile(debugNameStringLiteral, [sem = sem, fn = std::forward<FN>(fn)]() mutable {
    fn();
    sem.Done();
  });
#else
  bool const isSingleThreaded = (GetNumThreads() == 0);
  AddTaskNoProfile(
      [sem = sem, fn = std::forward<FN>(fn)]() mutable {
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
Schedule(TaskSemaphore const& sem, std::string_view debugNameStringLiteral, FN&& fn) {
  auto* scheduler = TaskScheduler::TryGet();
  if (scheduler && (scheduler->GetNumOtherThreads() > 0)) {
    scheduler->AddTask(sem, debugNameStringLiteral, std::forward<FN>(fn));
  } else { // Single threaded fallback
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
    MOCHI_PROFILE_SCOPE();
    MOCHI_PROFILE_LABEL(debugNameStringLiteral);
#endif
    fn();
  }
}

namespace details {
// Claim state of one ParallelForRange call, shared by the calling thread and its tasks.
struct ParallelForClaims {
  explicit ParallelForClaims(int numUnfinished) : unfinished(numUnfinished) {}
  TaskSemaphore const unfinished;
  std::atomic<int> next{0};
};
} // namespace details

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

  MOCHI_ASSERT_VERBOSE(globalRangeEnd >= globalRangeBegin, "Invalid range.");
  MOCHI_ASSERT_VERBOSE(minPerTask > 0, "Non-positive minPerTask.");
  MOCHI_ASSERT_VERBOSE(maxPerTask >= minPerTask, "maxPerTask below minPerTask.");
  auto* scheduler = TaskScheduler::TryGet();
  int numOtherThreads = scheduler ? scheduler->GetNumOtherThreads() : 0;
  if (numOtherThreads == 0) {
    // Single threaded fallback
    forRange(globalRangeBegin, globalRangeEnd);
    return;
  }

  // We have to decide how to subdivide the index range into sub-ranges. If all threads are idle and
  // all callbacks are equally fast, then the optimal subdivision would split the work evenly with
  // one sub-range per thread. However, some threads may already be busy or some callbacks may take
  // longer than others. In that case, an even subdivision would be sub-optimal because we could
  // end up waiting for one sub-range long after the others finish.
  //
  // Therefore, let's shoot for two sub-ranges per thread as a general target, unless we are forced
  // to do otherwise by minPerTask, maxPerTask, or count.
  int count = globalRangeEnd - globalRangeBegin;
  int numThreads = numOtherThreads + 1; // +1 for the current thread
  int const numSubRangesTarget = Min(count, 2 * numThreads);
  int const numPerSubRange =
      numSubRangesTarget ? Clamp(count / numSubRangesTarget, minPerTask, maxPerTask) : count;

  // Split off every sub-range except the last one, which this thread will handle, and add one task
  // for each of up to numClaimableSubRanges other threads. Each task claims unclaimed sub-ranges
  // until none is left. After its own sub-range, this thread claims and handles the rest too, so it
  // never waits for a busy or sleeping worker to start one. A task that starts after every
  // sub-range is claimed does nothing, so it never touches forRange after this call returns. The
  // tasks keep the claim state alive until they finish.
  int const numClaimableSubRanges = numPerSubRange ? (count - 1) / numPerSubRange : 0;
  int const numTasks = Min(numClaimableSubRanges, numOtherThreads);
  std::shared_ptr<mochi::details::ParallelForClaims> claims;
  auto claimRanges = [&forRange, globalRangeBegin, numPerSubRange, numClaimableSubRanges](
                         mochi::details::ParallelForClaims& state) {
    while (true) {
      int const range = state.next.fetch_add(1, std::memory_order_relaxed);
      if (range >= numClaimableSubRanges) {
        return;
      }
      int const localRangeBegin = globalRangeBegin + range * numPerSubRange;
      forRange(localRangeBegin, localRangeBegin + numPerSubRange);
      state.unfinished.Done();
    }
  };
  if (numTasks > 0) {
    claims = std::make_shared<mochi::details::ParallelForClaims>(numClaimableSubRanges);
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
    MOCHI_PROFILE_DESCRIPTION_F(
        "%d items, %d sub-ranges, %d tasks", count, numClaimableSubRanges + 1, numTasks);
#endif
    for (int i = 0; i < numTasks; ++i) {
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
      // TaskScheduler will wrap our std::function in another one, which will add custom formatted
      // profile labels.
      scheduler->AddTask(debugName, [claimRanges, claims]() { claimRanges(*claims); });
#else
      scheduler->AddTaskNoProfile(
          [claimRanges, claims]() {
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_LOW
            MOCHI_PROFILE_SCOPE_N("RunTask");
#endif
            claimRanges(*claims);
          },
          false);
#endif
    }
  }

  // If the user sets minPerTask to 1, then each item is probably quite expensive to compute. In
  // such cases, we really want each task to be processed in parallel on different worker threads.
  // Unfortunately, marl's task scheduling algorithm sometimes doubles them up on a single thread.
  // Hacky work-around: schedule dummy jobs to wake up more worker threads. They will then enter
  // the spin-for-work state and have the opportunity to steal work from the doubled up thread(s).
  // Waking more than numOtherThreads - numTasks threads cost more than it saved in benchmarks.
  if (minPerTask == 1) {
    scheduler->TryToWakeUpMoreWorkers(
        Min(numClaimableSubRanges / 2, numOtherThreads - numTasks)); // Wake up more workers please
  }

  // Since this is a blocking call, we might as well do part of the work here.
  {
#if MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_HIGH
    MOCHI_PROFILE_SCOPE();
    MOCHI_PROFILE_LABEL(debugName);
#elif MOCHI_TASK_PROFILE_VERBOSITY == MOCHI_TASK_PROFILE_VERBOSITY_LOW
    MOCHI_PROFILE_SCOPE_N("RunTask");
#endif
    forRange(globalRangeBegin + numClaimableSubRanges * numPerSubRange, globalRangeEnd);
    if (numTasks > 0) {
      claimRanges(*claims);
    }
  }

  if (numTasks > 0) {
    claims->unfinished.Wait();
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
