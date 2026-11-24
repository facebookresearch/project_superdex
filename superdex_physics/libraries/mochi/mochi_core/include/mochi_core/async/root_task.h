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

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/task_scheduler.h>

#include <coroutine>
#include <exception>
#include <string_view>
#include <utility>

namespace mochi {

/**
 * @brief Lazy coroutine root that runs on @ref TaskScheduler until completion.
 *
 * @details A RootTask owns its coroutine frame until @ref RunAndWait starts it. The coroutine then
 * destroys its frame on completion.
 *
 * @warning RootTask does not own objects referenced by pointer or reference parameters, a member
 * coroutine's object, or a coroutine lambda's closure. They must outlive @ref RunAndWait. Calling
 * RunAndWait in the full-expression that creates the task, e.g. `Coro(args).RunAndWait(scheduler)`,
 * keeps that expression's temporaries alive until it returns.
 */
class [[nodiscard]] RootTask {
 public:
  struct promise_type;

  explicit RootTask(std::coroutine_handle<promise_type> handle) : _handle(handle) {}

  RootTask(RootTask&& other) noexcept : _handle(other._handle) {
    other._handle = nullptr;
  }

  RootTask(RootTask const&) = delete;
  RootTask& operator=(RootTask const&) = delete;
  RootTask& operator=(RootTask&& other) noexcept {
    std::swap(_handle, other._handle);
    return *this;
  }

  ~RootTask() {
    if (_handle) { // If the coroutine was never started, the handle will be non-null.
      _handle.destroy();
    }
  }

  /// @brief Dispatch the coroutine and synchronously wait for completion.
  ///
  /// @details Dispatching consumes the task. The coroutine destroys its frame on the thread that
  /// completes it, then RunAndWait rethrows any escaping exception. Once the coroutine starts, it
  /// may destroy this RootTask. In local single-threaded mode, dispatch starts the coroutine
  /// inline, and the coroutine must complete before dispatch returns.
  ///
  /// @param[in] taskScheduler Scheduler used to dispatch the coroutine.
  /// @param[in] name Name used to identify the scheduled task.
  void RunAndWait(TaskScheduler& taskScheduler, std::string_view name = "RootCoroutine");

 private:
  /// @brief State of a @ref RunAndWait call, which outlives the coroutine frame.
  struct Completion {
    TaskSemaphore finished{1};
    std::exception_ptr exception;
  };

  std::coroutine_handle<promise_type> _handle;
};

struct RootTask::promise_type {
  /// @brief Return the lazy root task for this coroutine.
  RootTask get_return_object() {
    return RootTask{std::coroutine_handle<promise_type>::from_promise(*this)};
  }

  /// @brief RootTask is lazy and does not start at creation.
  std::suspend_always initial_suspend() noexcept {
    return {};
  }

  /// @brief Destroy the frame on the completing thread, then signal @ref RootTask::RunAndWait.
  struct FinalAwaiter : std::suspend_always {
    // Not inlining keeps `finished` below off the frame it destroys, working around
    // https://github.com/llvm/llvm-project/issues/56301.
#if MOCHI_CLANG_AWAIT_SUSPEND_BUG
    MOCHI_NO_INLINE
#endif
    void await_suspend(std::coroutine_handle<promise_type> handle) const noexcept {
      // Signal through a copy, since RunAndWait may return, destroying its Completion, as soon as
      // it is signaled.
      auto const finished = handle.promise()._completion->finished;
      handle.destroy();
      finished.Done();
    }
  };

  FinalAwaiter final_suspend() noexcept {
    return {};
  }

  void unhandled_exception() noexcept {
    _completion->exception = std::current_exception();
  }

  void return_void() {}

 private:
  friend class RootTask;

  Completion* _completion = nullptr;
};

inline void RootTask::RunAndWait(TaskScheduler& taskScheduler, std::string_view name) {
  MOCHI_ASSERT_VERBOSE(_handle != nullptr, "RootTask has already been started.");

  Completion completion;
  // Release ownership first, since the coroutine destroys its frame and may destroy this RootTask.
  // If AddTask throws, the coroutine never ran.
  auto const handle = std::exchange(_handle, {});
  handle.promise()._completion = &completion;
  try {
    taskScheduler.AddTask(name, handle);
  } catch (...) {
    _handle = handle;
    throw;
  }

  // In local single-threaded mode, waiting for an unfinished coroutine can suspend this thread's
  // fiber, letting unrelated tasks run on the thread in that mode.
  MOCHI_ASSERT_VERBOSE(
      !TaskScheduler::IsLocalSingleThreaded() || completion.finished.IsDone(),
      "In local single-threaded mode, a RootTask must complete before dispatch returns.");
  completion.finished.Wait();

  if (completion.exception) {
    std::rethrow_exception(completion.exception);
  }
}

} // namespace mochi
