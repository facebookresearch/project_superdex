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
#include <mochi_core/async/task.h>
#include <mochi_core/mochi_platform.h>

#include <atomic>
#include <coroutine>
#include <functional>
#include <memory>

namespace mochi {
namespace details {

// Workarounds for Clang bugs:
// - Do not inline to avoid accessing a destroyed coroutine frame when await_suspend is inlined:
//   https://github.com/llvm/llvm-project/issues/56301
//
// The why:
//    ParallelFactor<E>::Awaiter is designed to be strongly asynchronous for maximum performance:
//      - Each task is started asynchronously
//      - The work can complete even if not all tasks have finished
//      - Unfinished tasks after the factorization complete do not hinder the destruction
//        of the Trunk because they hold a weak_ptr<Trunk> and do nothing if they cannot lock it.
//      - Tasks that still run after the completion of the factorization will only refer to local
//        variables. They will clean up correctly even if the main code reaches the point of
//        destroying the TaskScheduler. The latter's destructor is forced to wait for the
//        marl::Scheduler which can complete only when there are no more executing tasks.
//
// By putting all the actions in a function that cannot be inlined, we guarantee that all
// the variables straddling `localExec(...)` are on the stack and not the coroutine frame.
// The clang++ bug is an illegal optimization that bypasses the copying of the variables and
// referring to the coroutine frame, thus potentially referring to deleted data.
#if MOCHI_CLANG_AWAIT_SUSPEND_BUG
MOCHI_NO_INLINE
#endif
std::coroutine_handle<>
TrunkAwaitSuspend(auto tr, int const localNumTasks, auto localExec) noexcept {
  for (int i = 1; i < localNumTasks; ++i) {
    localExec([tr]() {
      std::coroutine_handle<> toResume{nullptr};
      {
        // The trunk work may have been completed by another task.
        auto workTrunk = tr.lock();
        if (!workTrunk || !workTrunk->TryEnter()) {
          return;
        }
        bool isCompleter = workTrunk->func();
        auto mustResume = workTrunk->Exit(isCompleter);
        if (mustResume) {
          toResume = workTrunk->suspended;
        }
      }
      if (toResume) {
        toResume.resume();
      }
    });
  }
  // By now the trunk work may have been completed by the above tasks.
  auto trunkCopy = tr.lock();
  if (trunkCopy && trunkCopy->TryEnter()) {
    bool isCompleter = trunkCopy->func();
    auto mustResume = trunkCopy->Exit(isCompleter);
    if (mustResume) {
      return trunkCopy->suspended;
    }
  }
  return std::noop_coroutine();
}

} // namespace details

/**
 * @brief Represents a trunk in the multi-frontal solver.
 * @details One trunk can be entered by at most maxAvailable workers. That number is typically set
 * to the number of panels in the TrunkWork. When work is ready to start, available will be
 * set to that value.
 * To enter a trunk, a worker must successfully decrement available to a non-negative number.
 * The worker that completes the TrunkWork must zero `available` with an atomic swap
 * and add its previous value to the `exited` atomic.
 * When exiting, every worker must increment exited. The worker that brings `exited`'s value to
 * `maxAvailable` is the last to exit and must resume the suspended coroutine.
 *
 * @note The reason the worker that completes the TrunkWork is not necessarily the one resuming the
 * suspended handle is that as long as not all workers have exited, other workers are still
 * holding references to the TrunkWork. Resuming the suspended handle might destroy it
 * before they exit.
 */
struct Trunk {
  std::atomic<uint32_t> available{0};
  uint32_t maxAvailable{0};
  std::function<bool()> func;
  std::coroutine_handle<> suspended{};
  std::atomic<uint32_t> exited{0};

  bool TryEnter() {
    auto av = available.load(std::memory_order::relaxed);
    while (av > 0 && !available.compare_exchange_strong(av, av - 1)) {
    }
    return av > 0;
  }

  /// @brief Mark the exit of a worker from the Trunk.
  /// @details The TrunkWork associated with a Trunk should not be destroyed
  /// until all workers have exited.
  /// @param isCompleter True if the worker completing the Trunk is the one exiting.
  /// @return True if the worker is the last to exit.
  bool Exit(bool isCompleter);
};

/**
 * @brief Coordinates parallel panel factorization of a @ref Trunk using coroutines.
 * @details This function creates an awaiter that distributes panel factorization
 * work across multiple threads. The Trunk's internal synchronization primitives
 * are used to coordinate workers and ensure proper cleanup.
 *
 * The function returns an Awaiter that:
 * - If only one task is needed (numTasks <= 1), executes the work function directly
 *   and completes synchronously (await_ready returns true)
 * - Otherwise, suspends the calling coroutine and dispatches worker tasks to the executor.
 *   Each worker enters the trunk, executes the work function until completion, then exits.
 *   The last worker to exit resumes the suspended coroutine.
 *
 * @note The design maximizes efficiency by eliminating possible synchronizations between the
 * resumption of the calling coroutine and the completion of the Awaiter's work as well as any of
 * of the sub-tasks.
 * @note The Executor must be copyable but does not need to be movable. It must not be a reference.
 * This is because the awaiter may be still using the executor when the calling coroutine is resumed
 * and at the end of that coroutine, the frame containing the executor being referenced may be
 * destroyed resulting in a use after delete of the executor.
 * @note To be memory-efficient as well as time-efficient, a `weak_ptr` to the trunk is created,
 * allowing the trunk to be destroyed before all the sub-tasks are done. When a task wants to start
 * its work, it attempts to lock the pointer and in case of failure, returns immediately.
 *
 * @tparam Executor Type of the task executor (must not be a reference type).
 * @param trp Shared pointer to the Trunk synchronization object.
 * @param executor Executor for dispatching parallel worker tasks.
 * @param fct Work function to execute. Returns true when all work is complete.
 * @param numPanel Number of panels to process (sets trunk capacity).
 * @param numThreads Maximum number of threads to use (actual tasks = min(numThreads, numPanel)).
 * @return An awaitable object that completes when all panel work is finished.
 *
 * @note The Trunk's available counter is initialized to numPanel, limiting concurrent
 * workers to the number of panels. The actual number of tasks spawned is
 * min(numThreads, numPanel).
 * @note Workers use a weak_ptr to the trunk to avoid extending its lifetime beyond
 * necessity. If the trunk is destroyed, workers exit early.
 */
template <typename Executor>
  requires(!std::is_reference_v<Executor>)
auto ParallelFactor(
    std::shared_ptr<Trunk> trp,
    Executor executor,
    std::function<bool()> fct,
    int numPanel,
    int numThreads) {
  struct Awaiter {
    std::shared_ptr<Trunk> trunk;
    Executor exec;
    int numTasks{};

    /**
     * @brief Check if the awaitable is ready to complete synchronously.
     * @details If only one task is required (numTasks <= 1), executes the work function
     * directly on the current thread and returns true to indicate immediate completion.
     * Otherwise returns false to trigger suspension and parallel execution.
     * @return true if work was completed synchronously, false if suspension is needed.
     */
    bool await_ready() const noexcept {
      if (numTasks <= 1) {
        trunk->func();
        return true;
      }
      return false;
    }

    /**
     * @brief Suspend the coroutine and dispatch parallel workers.
     * @details Stores the coroutine handle in the trunk, then spawns (numTasks - 1)
     * worker tasks via the executor. Each worker:
     * 1. Attempts to enter the trunk (decrementing available counter)
     * 2. Executes the work function until it returns true (completion)
     * 3. Exits the trunk and potentially resumes the suspended coroutine
     *
     * The calling thread also attempts to enter and do work. If it is the last to exit,
     * it returns the awaiting coroutine's handle resuming it after await_suspend returns.
     * Otherwise, returning std::noop_coroutine() leaves resumption to the last worker, which may
     * already have resumed it.
     *
     * @param h Handle to the suspended coroutine.
     * @return The coroutine handle to resume (suspended handle if this thread has completed the
     * work, noop_coroutine otherwise).
     */
#if MOCHI_CLANG_AWAIT_SUSPEND_BUG
    MOCHI_NO_INLINE
#endif
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> h) const noexcept {
      trunk->suspended = h;
      auto tr = std::weak_ptr{trunk};
      auto localExec = exec;
      return details::TrunkAwaitSuspend(tr, numTasks, localExec);
    }

    /**
     * @brief Called after the coroutine is resumed.
     * @details No-op since all work is complete when resumed.
     */
    void await_resume() const noexcept {}
  };
  trp->func = std::move(fct);
  trp->maxAvailable = numPanel;
  trp->available.store(numPanel, std::memory_order::release);
  int numTasks = std::min(numThreads, numPanel);
  return Awaiter{std::move(trp), executor, numTasks};
}

/**
 * @brief Mark that a worker that entered a trunk has completed its work.
 *
 * @details This method is called when a worker cannot find work
 * to do on a trunk anymore.
 *
 * @param isCompleter Whether the worker is the one completing the trunk work.
 * @return True if the worker is the last to exit. (i.e., must resume the suspended coroutine).
 */
inline bool Trunk::Exit(bool isCompleter) {
  if (isCompleter) {
    // avoid having other workers enter the trunk.
    auto nonEntered = available.exchange(0);
    exited.fetch_add(nonEntered, std::memory_order::acq_rel);
  }
  return exited.fetch_add(1, std::memory_order::acq_rel) == maxAvailable - 1;
}

} // namespace mochi
