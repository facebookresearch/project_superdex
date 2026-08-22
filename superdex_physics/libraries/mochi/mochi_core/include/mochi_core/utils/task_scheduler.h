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

#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/time.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// Forwards:
namespace marl {
class ConditionVariable;
class Event;
class Scheduler;
class lock;
class mutex;
class WaitGroup;
} // namespace marl

namespace mochi {

/**************************************************************************************************
  Synchronization primitives are available if you take a direct dependency on Marl and include the
  corresponding Marl headers. These work well in both threads and coroutines.
*/
using TaskConditionVariable = marl::ConditionVariable; // #include <marl/conditionvariable.h>
using TaskLock = marl::lock; // #include <marl/mutex.h>
using TaskMutex = marl::mutex; // #include <marl/mutex.h>
class TaskSemaphore; // See below

/**************************************************************************************************
  TaskScheduler:
    - Manages the execution of tasks across multiple threads. A "task" is simply a std::function.
    - Can be restricted to single-threaded execution for debugging.
    - Each task is called from a coroutine.
    - Tasks can schedule additional tasks.
    - Tasks can wait for other tasks (see WaitGroup) without blocking the worker threads.

  IMPLEMENTATION NOTE:
      This class is currently implemented using Marl, an open-source task scheduler. The
      implementation details are hidden in the corresponding cpp file to help reduce our dependence
      on third-party code. See: https://github.com/google/marl
*/
class TaskScheduler final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(TaskScheduler);

 public:
  using TaskFn = std::function<void()>;

  struct Config {
    Config();
    explicit Config(int numThreads) : numThreads(numThreads) {}

    // Number of threads to create. Can use all logical processors by default.
    int numThreads = 0;

    // Stack size for each fiber (coroutine). May be rounded up to next power-of-two.
    int fiberStackSize = 256 * 1024;
  };

  explicit TaskScheduler(int numThreads);
  explicit TaskScheduler(Config const& config = {});
  ~TaskScheduler();

  /**
    Get the number of independent CPU cores in this machine, capped to the maximum worker thread
    count supported by the task scheduler implementation.
  */
  static int GetNumSupportedPhysicalProcessors();

  /**
    Get the number of logical processors capped to the maximum worker thread count supported by the
    task scheduler implementation.
    This number may be greater than GetNumSupportedPhysicalProcessors() due to technologies like
    hyperthreading.
  */
  static int GetNumSupportedLogicalProcessors();

  /**
    If there is a TaskScheduler bound to this thread, then get a pointer to it. Else return nullptr.
  */
  static TaskScheduler* TryGet();

  /**
    Assert that a TaskScheduler is already bound to this thread and return a reference to it.
  */
  static TaskScheduler& Get();

  /**
    Call once from each thread that uses the Scheduler API. Must be called before other API calls
    for the given thread. This sets up some static thread-local state. On Windows, static variables
    are not shared between DLLs. If you use TaskScheduler directly from multiple DLLs, then you may
    need to call BindthisThread() and UnbindThisThread() from your other DLLs as well, even through
    the same thread has already been bound. The internals are reference counted, so this is OK.
  */
  void BindThisThread();

  /**
    Call once for each thread that called BindThisThread().
    Must be called before the Scheduler is destroyed.
  */
  void UnbindThisThread();

  /**
    Return true if the calling thread is a @ref TaskScheduler worker thread.
  */
  [[nodiscard]] static bool IsCurrentThreadAWorker();

  /**
    Get the number of worker threads that this Scheduler was configured with, or zero if
    in single-threaded mode (either global or thread-local).
  */
  [[nodiscard]] int GetNumThreads() const;

  /**
    Get the number of other threads to which concurrent work could be scheduled.
    Return zero is there are no workers, or if the calling thread is the only worker
    thread, or if single-threaded mode is enabled.
  */
  [[nodiscard]] int GetNumOtherThreads() const;

  /**
    Static function to get the number of threads. Returns 0 if there is no TaskScheduler.
  */
  [[nodiscard]] static int StaticGetNumThreads() {
    auto const* scheduler = TryGet();
    return scheduler ? scheduler->GetNumThreads() : 0;
  }

  /**
    Static function to get the number of other threads, not including the calling thread.
    Returns 0 if there is no TaskScheduler.
  */
  [[nodiscard]] static int StaticGetNumOtherThreads() {
    auto const* scheduler = TryGet();
    return scheduler ? scheduler->GetNumOtherThreads() : 0;
  }

  /**
    When forced into global single-threaded mode, all NEW tasks will execute on the same thread that
    schedules them. This does NOT affect existing tasks that have already been scheduled. Handy for
    debugging.
  */
  void SetGlobalSingleThreadedMode(bool singleThreaded);

  /**
    When forced into local single-threaded mode, all NEW tasks scheduled by the calling thread will
    execute on the calling thread.

    Example:
      TaskScheduler::PushLocalSingleThreadedMode();
      DoSingleThreadedWork();
      TaskScheduler::PopLocalSingleThreadedMode();
  */
  static void PushLocalSingleThreadedMode();
  static void PopLocalSingleThreadedMode();

  /**
    Return true if the calling thread is in local single-thread mode.
  */
  [[nodiscard]] static bool IsLocalSingleThreaded();

  /**
    Schedule a function to be called on an available thread/fiber. If you need to wait for
    completion, then you can use your own TaskSemaphore inside the function.

    PERFORMANCE: The per-task overhead is significant. You may want to call the function directly
    instead of adding a task if there are no other threads that could perform the work concurrently.
    See GetNumOtherThreads.
  */
  template <typename FN>
  void AddTask(std::string_view debugNameStringLiteral, FN&& fn);

  /**
    Schedule a function to be called on an available thread/fiber. You can use the provided
    TaskSemaphore to wait for completion.

    PERFORMANCE: The per-task overhead is significant. You may want to call the function directly
    instead of adding a task if there are no other threads that could perform the work concurrently.
    See GetNumOtherThreads.
  */
  template <typename FN>
  void AddTask(TaskSemaphore sem, std::string_view debugNameStringLiteral, FN&& fn);

  /**
    Add a task with no profiling overhead. Used to implement parallel algorithms that handle the
    profiling labels for themselves.
  */
  void AddTaskNoProfile(TaskFn&& fn, bool isSingleThreaded);

  /**
    Add a task with verbose profiling. Used to implement parallel algorithms that handle the
    other verbosity levels for themselves.
  */
  void AddTaskVerboseProfile(std::string_view debugNameStringLiteral, TaskFn&& fn);

  /**
    Add a task to pool of up to 'targetWorkers' workers that are ready to start executing it
    immediately. If fewer than 'minWorkers' are available, no tasks are scheduled. 'targetWorkers'
    must greater than or equal to 'minWorkers'. The return is the number of scheduled tasks. The
    counter of the TaskSemaphore is increased by the same amount as the number of scheduled tasks.

    EXAMPLE:
    TaskSemaphore sem;
    auto task = [sem](int workerIdx, int numWorkers) {
      TaskScheduler::PushLocalSingleThreadedMode();
      // Do work in local single-threaded mode.
      TaskScheduler::PopLocalSingleThreadedMode();
      sem.Done();
    }
    int minWorkers = 1;
    int targetWorkers = 10;
    int scheduledTasks = scheduler.BatchEnqueueOnAvailableWorkers(sem, std::move(task), minWorkers,
          targetWorkers, false);
    sem.Wait();
    bool success = (scheduledTasks > 0) ? true : false;
  */
  using BatchTaskFn = std::function<void(int, int)>;
  int BatchEnqueueOnAvailableWorkers(
      TaskSemaphore sem,
      BatchTaskFn&& task,
      int minWorkers,
      int targetWorkers,
      bool includeSelf);

  /**
    Schedule dummy tasks to wake up more worker threads. Workers will then enter the spin-for-work
    state and have the opportunity to steal tasks from other worker(s).
  */
  void TryToWakeUpMoreWorkers(int numWorkers, TimeSpan dummyTaskDuration = TimeSpanFromSeconds(0));

 private:
  static thread_local TaskScheduler* s_currentScheduler;
  static thread_local int s_currentSchedulerRefCount;
  static thread_local bool s_marlAlreadyBound;
  static thread_local bool s_isCurrentThreadAWorker;
  static thread_local int s_localSingleThreadedCount;
  marl::Scheduler* _marl = nullptr;
  std::atomic<bool> _isGlobalSingleThreaded{false};
  std::atomic<uint64_t> _profileTaskCounter{0};
};

/**************************************************************************************************
  Profiling

  The TaskScheduler is designed to handle a huge number of tiny tasks. As such, it is possible for
  the overhead of detailed profile scopes to adversely affect performance when the profiling tool is
  connected. This may lead to incorrect conclusions about the efficiency of concurrent algorithms,
  so you can adjust the verbosity as needed.

    MOCHI_TASK_PROFILE_VERBOSITY_NONE:
      Zero overhead. Zero info unless the task has its own profile scopes.

    MOCHI_TASK_PROFILE_VERBOSITY_LOW:
      All tasks will have the label "RunTask" to ensure that they are visible in the tool.
      There will be no way to tell which task it was unless the task has its own profile scopes.

    MOCHI_TASK_PROFILE_VERBOSITY_HIGH:
      Every task will have a descriptive label in the form "RunTask NAME (ID)". The name lets you
      know what it is and the ID lets you match it to the corresponding "AddTask NAME (ID)" scope.
      This is nice for debugging but significantly degrades performance when the profiling tool is
      connected.
*/
#define MOCHI_TASK_PROFILE_VERBOSITY_NONE 0
#define MOCHI_TASK_PROFILE_VERBOSITY_LOW 1
#define MOCHI_TASK_PROFILE_VERBOSITY_HIGH 2

#ifndef MOCHI_TASK_PROFILE_VERBOSITY
#define MOCHI_TASK_PROFILE_VERBOSITY MOCHI_TASK_PROFILE_VERBOSITY_LOW
#endif

/**************************************************************************************************
  TaskSemaphore is a synchronization primitive that holds an internal counter that can incremented,
  decremented and waited on until it reaches 0 (like a semaphore). TaskSemaphore can be used as a
  simple mechanism for waiting on a number of concurrently execute a number of tasks to complete.
  See marl::WaitGroup: https://github.com/google/marl

  WARNING:
      If you create a TaskSemaphore on the stack, then pass it to lambdas BY VALUE not by
      reference. Otherwise it might be destroyed prematurely. Note that the internal state is
      reference counted.

  Example:
      TaskSemaphore sem(1);
      scheduler.AddTask([sem]() { // capture semaphore BY VALUE
        // Do work
        sem.Done();
      });
      sem.Wait();
*/
class TaskSemaphore {
 public:
  // Constructs the WaitGroup with the specified initial count.
  TaskSemaphore(int initialCount = 0);

  // Increments the internal counter by count.
  void Add(int count = 1) const;

  // Decrements the internal counter by numTasksDone. Returns true if the internal counter reaches
  // zero.
  bool Done(int numTasksDone = 1) const;

  // Return true if the internal counter is already zero.
  bool IsDone() const;

  // Blocks until the internal counter reaches zero.
  void Wait() const;

  // Blocks until the internal counter reaches zero, then returns true.
  // If it takes longer than the specified duration, then it will give up and return false.
  bool WaitFor(TimeSpan duration) const;

  struct Data {
    virtual ~Data() = default;
  };

 private:
  std::shared_ptr<Data> const _data;
};

/**************************************************************************************************
  Utility Functions

    If a TaskScheduler is bound to the current thread, then they will use it. Otherwise, they fall
    back on single-threaded implementations to ensure they are valid to use in any context.
*/

// Add a task to call a function. If you need to wait for completion, you can use your own
// TaskSemaphore inside the function.
//
// WARNING: The function will be called IMMEDIATELY if a TaskScheduler is not bound to this thread,
// or if the TaskScheduler has no other threads that could performance the work concurrently. If you
// have a task that must be scheduled to avoid deadlocks, then use TaskScheduler::AddTask directly.
template <typename FN>
void Schedule(std::string_view debugNameStringLiteral, FN&& fn);

// Add a task to call a function. You can use the provided TaskSemaphore to wait for completion.
//
// WARNING: The function will be called IMMEDIATELY if a TaskScheduler is not bound to this thread,
// or if the TaskScheduler has no other threads that could performance the work concurrently. If you
// have a task that must be scheduled to avoid deadlocks, then use TaskScheduler::AddTask directly.
template <typename FN>
void Schedule(TaskSemaphore sem, std::string_view debugNameStringLiteral, FN&& fn);

/**
  Subdivide a range of indices into one or more sub-ranges. Then, calls forRange(subRangeBegin,
  subRangeEnd) one or more time. If it distributes work to any async tasks, then it will wait for
  completion before returning. Thus the operation is synchronous, from the caller's perspective.

  If each call is very expensive then set minPerTask to 1 to indicate it is OK to create a task for
  each index. If each call is very cheap, then set minPerTask to a larger value to reduce overhead.
*/
template <typename FN>
void ParallelForRange(
    [[maybe_unused]] std::string_view debugName,
    int globalRangeBegin,
    int globalRangeEnd,
    int minPerTask,
    int maxPerTask,
    FN const& forRange);

/**
  Calls forEach(i) for each i in the half-open range [0, n). If it distributes work to any async
  tasks, then it will wait for completion before returning. Thus the operation is synchronous, from
  the caller's perspective.

  If each call is very expensive then set minPerTask to 1 to indicate it is OK to create a task for
  each index. If each call is very cheap, then set minPerTask to a larger value to reduce overhead.

  Example:
    ParallelForN("DoSomething", 100, 1, [](int i) { DoSomething(i); });

  Equivalent To:
    for (int i = 0; i < 100; ++i) {
      DoSomething(i);
    }
*/
template <typename FN>
void ParallelForN(
    std::string_view debugNameStringLiteral,
    int n,
    int minPerTask,
    FN const& forEach);

/**
  Calls forEach(container[i]) for each element of the container, which must support random access
  (e.g. Span, std::vector, std::array, etc...). If it distributes work to any async tasks, then it
  will wait for completion before returning. Thus the operation is synchronous, from the caller's
  perspective.

  If each call is very expensive then set minPerTask to 1 to indicate it is OK to create a task for
  each index. If each call is very cheap, then set minPerTask to a larger value to reduce overhead.

  Example:
    std::vector<MyClass> vec;
    ParallelForEach("DoSomething", vec, 1, [](MyClass& x) { DoSomething(x); });

  Equivalent To:
    for (auto&& x : container) {
      DoSomething(x);
    }
*/
template <typename ContainerT, typename FN>
void ParallelForEach(
    std::string_view debugNameStringLiteral,
    ContainerT&& container,
    int minPerTask,
    FN const& forEach);

/**
  Similar to ParallelForEach with a single container (see above), but this overload takes a
  std::tuple of references to containers. This provides a convenient way to iterate over multiple
  containers at the same time. They must all have equal size.

  Example:
    std::vector<MyClass> vec1;
    std::vector<int> vec2;
    ParallelForEach(
        "DoSomething", std::tie(vec1, vec2), 1, [](MyClass& x, int& y) { DoSomething(x, y); });

  Example Equivalent To:
    for (int i = 0; i < std::size(vec1); ++i) {
      DoSomething(vec1[i], vec2[i]);
    }
*/
template <typename FN, typename... ContainerT>
void ParallelForEach(
    std::string_view debugNameStringLiteral,
    std::tuple<ContainerT&...> containerTuple,
    int minPerTask,
    FN const& forEach);

/**
  Simple parallel merge sort. Use it just like std::sort.

  Example:
    ParallelSort(std::begin(myData), std::end(myData), optionalCompare);

  Notes:
    The maximum number of async tasks is 2^taskDepthCoutdown.
    The default value should be sufficient to avoid oversubscription.
    If we need something faster in the future, then consider PARADIS or other third-party solutions.

*/
template <class Iter, class Less = std::less<>>
void ParallelSort(
    Iter begin,
    Iter end,
    Less comp = Less{},
    int taskDepthCoutdown = (TaskScheduler::StaticGetNumThreads() > 0)
        ? static_cast<int>(std::log2(TaskScheduler::StaticGetNumThreads()))
        : 0);

/**
  Busy wait until a condition is met. It is a busy wait in the sense that the worker does NOT yield
  to Mochi's task scheduler, but it may yield to the OS scheduler to prevent starvation of non-Mochi
  tasks. It is meant to be used in concurrent parallel algorithms that involve sequential steps.
*/
template <class Predicate>
void BusyWaitFor(Predicate&& condition, TimeSpan yieldPeriod = TimeSpanFromSeconds(1e-3));

/**************************************************************************************************
  ParallelBarrier is a synchronization primitive that enables multiple workers to run concurrently a
  parallel algorithm that involves sequential steps. Each worker runs until it reaches a barrier
  point in the code. The barrier represents the end of one phase of work. When a worker reaches a
  barrier, it waits until all workers reach the same barrier point.

  WARNINGS:
    - Workers busy-wait at barrier points. It's the responsibility of the caller to ensure each
      worker is allocated to a different thread (which can be accomplished through
      TaskScheduler::BatchEnqueueOnAvailableWorkers) to prevent deadlocks.
    - Each worker must own a COPY (not a reference) of the barrier.

  EXAMPLE:
      ParallelBarrier barrier(5);
      for (int workerId = 0; workerId < 5; ++workerId) {
        scheduler.AddTask([barrier]() { // Capture barrier BY VALUE
          // Do work
          barrier.Wait();
          // Do more work
          for (int iter = 0; iter < 100; ++iter) {
            barrier.Wait();
            // Do more work
          }
        });
      }
*/
class ParallelBarrier final {
 public:
  ParallelBarrier() = delete;
  ~ParallelBarrier() = default;
  ParallelBarrier(ParallelBarrier const&) = default;
  ParallelBarrier(ParallelBarrier&&) = default;
  MOCHI_DECLARE_NO_ASSIGN(ParallelBarrier);

  explicit ParallelBarrier(int numWorkers) : _numWorkers(numWorkers), _target(numWorkers) {
    MOCHI_ASSERT_VERBOSE(_numWorkers > 0, "Number of workers must be positive.");
  }

  // Reduce the number of workers that use the parallel barrier.
  void ReduceNumWorkers(int numWorkers, bool /*isMaster*/) {
    MOCHI_ASSERT_VERBOSE(numWorkers <= _numWorkers, "Invalid new number of workers.");
    _target -= (_numWorkers - numWorkers);
    _numWorkers = numWorkers;
  }

  void Wait() const {
    if (_numWorkers == 1) {
      return; // No need to wait for other workers.
    }

    if (_count->fetch_add(1, std::memory_order_acq_rel) + 1 < _target) {
      auto exitCondition = [&]() { return (_count->load(std::memory_order_acquire) >= _target); };
      BusyWaitFor(exitCondition);
    }

    _target += _numWorkers;
  }

 private:
  int _numWorkers = {};
  std::shared_ptr<std::atomic<uint64_t>> const _count = std::make_shared<std::atomic<uint64_t>>();
  mutable uint64_t _target = 0;
};

} // namespace mochi

#include "task_scheduler_inl.h"
