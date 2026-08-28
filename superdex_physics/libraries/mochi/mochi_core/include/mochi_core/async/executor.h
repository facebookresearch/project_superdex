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
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/task_scheduler.h>

#include <atomic>
#include <concepts>
#include <coroutine>
#include <type_traits>
#include <utility>

namespace mochi {
namespace async {
template <typename T>
constexpr T& NextValue(T& t) {
  return t;
}

template <typename T>
constexpr T const& NextValue(T const& t) {
  return t;
}

/** @brief Function returning whether t is being iterated over and has reached the end
 *  of its range.
 *  @details By default arguments are not being iterated over. Only ForEach<T> do iterate
 *  over their content range. */
template <typename T>
bool EndOfRange(T const& /* t */) {
  return false;
}

/** @brief A utility class to iterate over a range once.
 *
 * @details This class is used to indicate to the Execute method that the task generating
 * functor should be called once for each value in the range object.
 *
 * @tparam T Type of an object over which to iterate
 * @tparam It Iterator type for T (type returned by std::begin(T))
 * @tparam Snt `end` sentinel type for T
 */
template <typename T, typename It, typename Snt>
class ForEach {
 public:
  template <typename U>
    requires(!std::is_same_v<U, ForEach>) // Avoids conflict with the move constructor.
  explicit ForEach(U&& v) : _v(std::forward<U>(v)), _it(std::begin(_v)), _sentinel(std::end(_v)) {}

  ForEach(ForEach const&) = delete;

  ForEach(ForEach&&) noexcept = default;

  ~ForEach() = default;

  ForEach& operator=(ForEach const&) = delete;
  ForEach& operator=(ForEach&&) = delete;

  friend decltype(auto) NextValue(ForEach<T, It, Snt>& f) {
    return *f._it++;
  }
  /** @brief Test if the iterator has reached the end of the content. */
  friend bool EndOfRange(ForEach<T, It, Snt> const& f) {
    return f._it == f._sentinel;
  }

 private:
  T _v;
  It _it;
  Snt _sentinel;
};

template <typename T>
ForEach(T&&)
    -> ForEach<T, decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>;

/** @brief Awaiter that puts a coroutine into fully suspended mode before
 * calling a functor.
 * @details To notify a thread that started a coroutine, It is typical to
 * decrement a semaphore. Doing so, however is not thread safe, as the notified
 * thread may be holding the coroutine's Task object and destroy it before
 * the coroutine has finished its work. To safely notify the thread, use:
 *    co_await EndWith([sem]{sem.Done()});
 * This call will never return, so there should not be any code after it.
 * The coroutine will be fully suspended before the functor is called.
 * and the thread can only resume after the coroutine is in a safe (suspended)
 * state to be destroyed.
 */
template <std::move_constructible T>
  requires requires(T&& t) { t(); }
auto EndWith(T&& t) {
  struct awaiter {
    T t;
    static bool await_ready() noexcept {
      return false;
    }
    /** @brief Call t when the coroutine is a fully suspended state.
     *  @details This call moves the functor before calling it.
     *  The reason being that the functor can allow restarting the owner
     *  of the coroutine handle to restart on another thread, potentially destroying
     *  the coroutine frame containing `t`. If the functor still needs access to data
     *  inside `t`, this would result in a race condition.
     *  With the move, on the one hand, `t` is left in a consistent state for destruction and
     *  on the other hand, `moved_t` stays alive on the stack until its `operator()` is finished.
     */
    std::coroutine_handle<> await_suspend(std::coroutine_handle<>) {
      T moved_t{std::move(t)};
      moved_t();
      return std::noop_coroutine();
    }

    void await_resume() {}
  };
  return awaiter{std::forward<T>(t)};
}
} // namespace async

namespace detail {
using async::ForEach;
using async::NextValue;

template <typename T>
constexpr bool IsForEachDef = false;

template <typename T, typename It, typename Snt>
constexpr bool IsForEachDef<ForEach<T, It, Snt>> = true;

template <typename T>
using NextValue_t = decltype(NextValue(std::declval<T>()));

/** @brief A counter allowing a parent subroutine to be resumed when all its children
 * reach their end of life.
 * @details The parent uses `co_await WhenCountIs(n);` where n is the number of its children.
 * Each child decrements the count from its `final_suspend()` rather than its body: a child is a
 * `Task<T, DeathHandler>`, and @ref DeathHandler::FinalSuspendAwaiter returns @ref EndOfLife's
 * awaiter. Routing the decrement through `final_suspend()` guarantees it runs on both normal
 * completion and unhandled exception (so a throwing child cannot deadlock the parent). The
 * awaiter suspends the child and never continues; the parent destroys the children's frames.
 */
class DeathCount {
 public:
  DeathCount() = default;

  std::coroutine_handle<> MarkEndOfLife() {
    if (--_count == 0) {
      return _awaitingCoroutine;
    } else {
      return std::noop_coroutine();
    }
  }

  auto EndOfLife() {
    struct Awaiter {
      DeathCount& counter;
      bool await_ready() const noexcept {
        return false;
      }

      /// @note The coroutine that called EndOfLife will never be resumed.
      std::coroutine_handle<> await_suspend(std::coroutine_handle<>) noexcept {
        return counter.MarkEndOfLife();
      }

      void await_resume() const noexcept {}
    };
    return Awaiter{*this};
  }

  /// @brief One coroutine should call `co_await counter.WhenCountIs(n);`.
  auto WhenCountIs(size_t cnt) {
    struct Awaiter {
      DeathCount& counter;
      size_t count;

      static bool await_ready() noexcept {
        return false;
      }

      std::coroutine_handle<> await_suspend(std::coroutine_handle<> suspending_coroutine) noexcept {
        counter._awaitingCoroutine = suspending_coroutine;
        if ((counter._count += count) == 0) {
          return suspending_coroutine; // If count was already 0, we resume immediately.
        } else {
          return std::noop_coroutine();
        }
      }
      void await_resume() const noexcept {}
    };
    return Awaiter{.counter = *this, .count = cnt};
  }

 private:
  std::atomic<size_t> _count;
  std::coroutine_handle<> _awaitingCoroutine;
};

/** @brief Promise base that decrements a @ref DeathCount as part of `final_suspend()`.
 *  @details Captures the @ref DeathCount through the coroutine promise-constructor
 *  mechanism (it derives from @ref PromiseArgConsumer so the @ref Task forwarding
 *  constructor passes it the coroutine arguments) and exposes a `FinalSuspendAwaiter()`
 *  hook returning @ref DeathCount::EndOfLife's awaiter. Because `final_suspend()` runs on
 *  both normal completion and unhandled exception, the death-count decrement and parent
 *  resumption are guaranteed to happen — eliminating the deadlock structurally.
 *
 *  @warning A `Task<T, DeathHandler>` must be scheduled directly and never `co_await`ed:
 *  the hook bypasses the continuation logic used by the default `final_suspend()`.
 */
struct DeathHandler : PromiseArgConsumer {
  DeathHandler() = default;

  /// @brief Captures the @ref DeathCount; ignores the remaining coroutine arguments.
  DeathHandler(DeathCount& dc, auto&&...) : _dc(&dc) {}

  /// @brief Hook invoked by @ref Task::promise_type::final_suspend.
  auto FinalSuspendAwaiter() const noexcept {
    return _dc->EndOfLife();
  }

 private:
  DeathCount* _dc{nullptr};
};

template <typename T>
struct AwaitStrip {
  using type = T;
};

template <IsTask T>
struct AwaitStrip<T> {
  using type = typename std::decay_t<T>::value_type;
};

template <typename FTor, typename... Args>
struct WorkGenerator {
  using TaskResult_t = decltype(std::declval<FTor>()(NextValue(std::declval<Args&>())...));
  static constexpr bool resultIsTask = IsTask<TaskResult_t>;
  using WorkResult_t = typename AwaitStrip<TaskResult_t>::type;
  static constexpr bool isVoidResult = std::is_same_v<WorkResult_t, void>;
  using CompoundResult_t = std::conditional_t<isVoidResult, void, DynamicArray<WorkResult_t>>;

  static Task<WorkResult_t, DeathHandler>
  NextTask([[maybe_unused]] DeathCount& dc, FTor& f, auto&&... args)
    requires(resultIsTask)
  {
    co_return co_await f(args...);
  }

  static Task<WorkResult_t, DeathHandler>
  NextTask([[maybe_unused]] DeathCount& dc, FTor& f, auto&&... args)
    requires(!resultIsTask)
  {
    co_return f(args...);
  }
};

} // namespace detail

template <typename T>
concept IsForEach = detail::IsForEachDef<std::decay_t<T>>;

/**
 * @brief Execute a functor for each value in one or more ForEach ranges, dispatching work via exec.
 *
 * @tparam Exec Callable type that schedules a coroutine or task.
 * @tparam FTor Type of the task-running functor. Can be a coroutine type.
 * @tparam Args Argument types for the functor's input. At least one must be a @ref ForEach range.
 * @param exec Callable that schedules a coroutine or task.
 * @param f Functor invoked per iteration.
 * @param args Arguments to pass to f. For every @ref ForEach type argument, f is given one value
 * in the range.
 * @return Task yielding void if f returns void, or a DynamicArray of results otherwise.
 */
template <typename Exec, typename FTor, typename... Args>
  requires(IsForEach<Args> || ...)
auto Execute(Exec exec, FTor f, Args... args)
    -> Task<typename detail::WorkGenerator<FTor, Args...>::CompoundResult_t> {
  using async::EndOfRange;
  using async::NextValue;
  using WorkGenerator = detail::WorkGenerator<FTor, Args...>;
  using WorkResult = typename WorkGenerator::WorkResult_t;

  detail::DeathCount counter;

  DynamicArray<Task<WorkResult, detail::DeathHandler>> workers;
  while (!(EndOfRange(args) || ...)) {
    workers.emplace_back(WorkGenerator::NextTask(counter, f, NextValue(args)...));
    exec(workers.back());
  }
  co_await counter.WhenCountIs(workers.size());
  // All workers have reached their end of life (success or failure) by this point
  // because the death-count decrement runs in each worker's final_suspend(), which the
  // standard guarantees on both normal completion and unhandled exception. If any worker
  // captured an exception, propagate the first one so the caller sees the real error.
  // Further exceptions (if any) are currently discarded; if multi-exception
  // aggregation is ever needed, this is the place to add it.
  if constexpr (WorkGenerator::isVoidResult) {
    for (auto& worker : workers) {
      if (auto const& ex = worker.Handle().promise().GetException()) {
        std::rethrow_exception(ex);
      }
    }
    co_return;
  } else {
    DynamicArray<WorkResult> result{};
    result.reserve(workers.size());
    for (auto& worker : workers) {
      auto& promise = worker.Handle().promise();
      if (auto const& ex = promise.GetException()) {
        std::rethrow_exception(ex);
      }
      result.emplace_back(std::move(promise.result.value()));
    }
    co_return std::move(result);
  }
}

class Executor {
 public:
  Executor(TaskScheduler& taskScheduler) : _taskScheduler(taskScheduler) {}
  Executor() : Executor(TaskScheduler::Get()) {}
  Executor(Executor const&) = default;
  Executor(Executor&&) = default;
  ~Executor() = default;

  Executor& operator=(Executor const&) = delete;
  Executor& operator=(Executor&&) = delete;

  /**
   * @tparam FTor Type of the task-running functor. Can be a coroutine type
   * @tparam Args Arguments type for the functor's input. At least one must be `Each` range type.
   * @param f Functor running the tasks
   * @param args Arguments to pass to f. For every `Each` type argument, f is given one value in the
   * range.
   * @return Either void if f returns void or a DynamicArray made of the result from each task.
   */
  template <typename FTor, typename... Args>
    requires(IsForEach<Args> || ...)
  auto Execute(FTor&& f, Args&&... args) {
    return mochi::Execute(
        [this](auto&& t) { this->schedule(std::forward<decltype(t)>(t)); },
        std::forward<FTor>(f),
        std::forward<Args>(args)...);
  }

  template <typename F>
    requires requires(F f) {
      { f.Handle() } -> std::convertible_to<std::coroutine_handle<>>;
    }
  void schedule(F&& f) const {
    _taskScheduler.AddTask("coroutine", f.Handle());
  }

  template <typename F>
  void schedule(F&& f) const {
    _taskScheduler.AddTask("fct", f);
  }

 private:
  TaskScheduler& _taskScheduler;
};

} // namespace mochi
