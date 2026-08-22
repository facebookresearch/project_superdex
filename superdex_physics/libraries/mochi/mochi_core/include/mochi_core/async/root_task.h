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
#include <coroutine>

#include <mochi_core/utils/task_scheduler.h>

namespace mochi {

/** @brief Coroutine type that can be scheduled onto TaskScheduler.
 *  @details The task is lazy but it is self-destructive:
 *   - lazy: creating a RootTask does not start the coroutine. It needs to be scheduled
 *     by use of Schedule(ts, "name");
 *   - self-destructive: when it is finished, its destroy() method is automatically called
 *     by the coroutine system.
 *   If the task is never scheduled, the RootTask object will destroy the coroutine frame.
 *   If the first argument is a TaskSemaphore, the semaphore's `Done()` method will automatically
 *   be called at the completion of the coroutine, even in case of an unexpected exception
 *   being thrown. The `Done()` method should not be manually called on the first argument
 *   semaphore.
 */
class RootTask {
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
    if (_handle) { // If the coroutine was never scheduled, the handle will be non-null.
      _handle.destroy();
    }
  }
  /// @brief Schedule the coroutine onto the given TaskScheduler.
  /// @details The coroutine will destroy its frame automatically when it completes.
  void Schedule(TaskScheduler& ts, std::string_view name = "RootCoroutine") {
    ts.AddTask(name, _handle);
    _handle = {};
  }

 private:
  std::coroutine_handle<promise_type> _handle;
};

struct RootTask::promise_type {
  promise_type(TaskSemaphore sem, [[maybe_unused]] auto&&... args) : _sem(sem) {}
  /// @brief The coroutine system will call handle.resume() when the coroutine
  /// is resumed.
  /// @param handle The coroutine handle.
  /// @return The coroutine's return value.
  /// @note The coroutine system will call handle.destroy() as a result of the
  /// suspend_never return type.
  RootTask get_return_object() {
    return RootTask{std::coroutine_handle<promise_type>::from_promise(*this)};
  }
  /// @brief RootTask is lazy and does not start at creation.
  std::suspend_always initial_suspend() noexcept {
    return {};
  }
  /// @brief RootTasks are self-destructive, i.e. the coroutine system will call
  /// handle.destroy() as a result of the suspend_never return type.
  std::suspend_never final_suspend() noexcept {
    _sem.Done();
    return {};
  }
  // TODO Figure out desired handling of any exception.
  void unhandled_exception() noexcept {
    _exception = std::current_exception();
  }
  void return_void() {}

 private:
  std::exception_ptr _exception{};
  TaskSemaphore _sem;
};
} // namespace mochi
