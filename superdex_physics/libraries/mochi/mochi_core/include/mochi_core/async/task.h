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

#include <concepts>
#include <coroutine>
#include <exception>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace mochi {

/**
 * @brief Marker base for promise base types that consume the coroutine arguments.
 *
 * @details A `Task` promise may be assembled from several `PromiseBaseTypes`. Some bases
 * (allocators, restart executors) must be default-constructed, while others need the
 * coroutine's arguments forwarded to them. A base advertises "forward the coroutine
 * arguments to my constructor" by deriving from this marker; the `WithArguments` concept
 * detects it. This is explicit and self-documenting, avoiding accidental matches that an
 * `std::is_constructible` probe could produce.
 */
struct PromiseArgConsumer {};

/// @brief Detects promise base types that opt into receiving the coroutine arguments.
template <typename B>
concept WithArguments = std::derived_from<B, PromiseArgConsumer>;

/// @brief Detects a promise that provides a `noexcept` `FinalSuspendAwaiter()` hook.
/// @details When present, `final_suspend()` delegates to this hook instead of the default
/// continuation-based awaiter.
template <typename P>
concept HasFinalSuspendHook = requires(P const& p) {
  { p.FinalSuspendAwaiter() } noexcept;
};

namespace detail {
/// @brief Constructs a promise base, forwarding the coroutine args only to opt-in bases.
/// @details Returns a prvalue so the base subobject is initialized via guaranteed copy
/// elision (no copy/move ctor required; supports move-only / non-default-constructible
/// arg-consuming bases).
template <typename B, typename... Args>
B MakeBase(Args&... args) {
  if constexpr (WithArguments<B>) {
    return B(args...);
  } else {
    return B{};
  }
}

template <typename T, typename PromiseType>
struct TaskAwaiter {
  bool await_ready() const noexcept {
    return handle.done();
  }

  std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept {
    handle.promise().SetContinuation(awaiting_coroutine);
    // start the coroutine
    return handle;
  }

  T await_resume() {
    PromiseType& promise = handle.promise();
    auto exception = promise.GetException();
    if (exception) {
      std::rethrow_exception(exception);
    }
    if constexpr (std::is_same_v<T, void>) {
      return;
    } else {
      return std::move(handle.promise().result.value());
    }
  }

  std::coroutine_handle<PromiseType> handle;
};
} // namespace detail

/**
 * @brief Lazy task coroutine returning a T.
 *
 * @details As a lazy task, the task does not start immediately at creation. It must either
 * be scheduled on an executor (such as mochi::TaskScheduler) or awaited by another coroutine.
 *
 * The PromiseBaseTypes template arguments can be used to further tune the behavior of the
 * coroutine. Examples of possible behavior tuning are:
 *   - Supplying an allocation mechanism for the promise type
 *   - Supplying a restart executor.
 * Note that there are no examples of such a tuning within the scope of this file. These are
 * present so that any user can create specific Task types for their particular use.
 *
 * @tparam T Type returned on exit of the coroutine.
 * @tparam PromiseBaseTypes A (possibly empty) set of base classes for behavior customization.
 */
template <typename T, typename... PromiseBaseTypes>
class [[nodiscard]] Task {
 public:
  using value_type = T;
  /// @brief The task promise_type as required for C++ stackless coroutines.
  struct promise_type;

  Task(Task&& t) noexcept : _handle(t._handle) {
    t._handle = nullptr;
  }

  Task(Task const&) = delete;

  ~Task() {
    if (_handle) {
      _handle.destroy();
    }
  }

  Task& operator=(Task const&) = delete;

  Task& operator=(Task&& t) = delete;

  /// Co-awaiting a task starts it if it is not done yet.
  /// This is an easy mechanism but care should be taken to not call co_await on different threads.
  auto operator co_await() noexcept {
    return detail::TaskAwaiter<T, promise_type>{_handle};
  }

  /// @brief Returns the value of the task if it is done.
  /// @details This should only be called after the task has finished and should be called only
  /// once. if T is a movable object owning data. This method is for advanced use cases only. Do not
  /// call if the task was `co_await`ed.
  T GetValue() {
    auto& promise = _handle.promise();
    auto exception = promise.GetException();
    if (exception) {
      std::rethrow_exception(exception);
    }
    if constexpr (std::is_same_v<T, void>) {
      return;
    } else {
      return std::move(_handle.promise().result.value());
    }
  }

  auto Handle() const {
    return _handle;
  }

  void SetContinuation(std::coroutine_handle<> continuation) {
    _handle.promise().SetContinuation(continuation);
  }

  void SetContinuation(std::coroutine_handle<> (*f)(void*), void* arg) {
    _handle.promise().SetContinuation(f, arg);
  }

 private:
  /// @brief The promise object calls the constructor of the Task with its handle.
  explicit Task(std::coroutine_handle<promise_type> handle) : _handle(std::move(handle)) {}

  std::coroutine_handle<promise_type> _handle;
};

/**
 * @brief Base class for the required coroutine `Task<T>::promise_type`.
 *
 * @details The `promise_type` must define return_value when T is non-void and return_void
 * when there is no return type. It is not allowed to define both functions, and trying to use
 * a `requires` to shunt out the unused one is not allowed either.
 * This base type is specialized for the `Task<void>` case, thus avoiding duplicating the
 * rest of the promise_type.
 *
 * @tparam T The return type of `Task`
 */
template <typename T>
struct PromiseBase {
  std::optional<T> result;

  void return_value(auto&& v) {
    result.emplace(std::forward<decltype(v)>(v));
  }
};

/** @brief Specialization of the PromiseBase in case the return type of the task is void. */
template <>
struct PromiseBase<void> {
  void return_void() {}
};

/**
 * @brief The C++20 coroutine mandated `promise_type` of `Task`.
 *
 * @details The promise_type controls many aspects of a coroutine's behavior, including
 * the possibility of using a custom allocator. It must provide a minimum of expected functions,
 * in order of call:
 *  - get_return_object() which builds the coroutine object. (Task)
 *  - initial_suspend() which is called when the coroutine is created
 *  - return_value(...) or return_void() which is called from the coroutine `co_return`.
 *  - final_suspend() which is called at the exit of the coroutine
 * If exception can be thrown and not caught from within the execution of the coroutine body,
 * it must also supply unhandled_exception().
 */
template <typename T, typename... BaseTypes>
struct Task<T, BaseTypes...>::promise_type : PromiseBase<T>, BaseTypes... {
  /// @brief Forwarding constructor used by the coroutine machinery for tasks with bases.
  /// @details Each base is constructed via @ref detail::MakeBase, which forwards the
  /// coroutine arguments only to bases deriving from @ref PromiseArgConsumer and
  /// default-constructs the rest. `PromiseBase<T>` is not in `BaseTypes...`, so it is
  /// default-constructed automatically. Disabled for `Task<T>` (no bases), which keeps the
  /// implicit default constructor.
  template <typename... Args>
    requires(sizeof...(BaseTypes) > 0)
  promise_type(Args&... args) : BaseTypes(detail::MakeBase<BaseTypes>(args...))... {}

  promise_type() = default;

  Task get_return_object() noexcept {
    return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
  }

  /// @brief Task is lazy and does not start at creation.
  std::suspend_always initial_suspend() noexcept {
    return {};
  }

  /// @brief Called after the return of the coroutine. Returns an awaiter/awaitable.
  /// @details If a base provides a `noexcept` `FinalSuspendAwaiter()` hook (see
  /// @ref HasFinalSuspendHook), `final_suspend()` delegates to it. This is how a task can
  /// guarantee teardown work (e.g. decrementing a death count) runs on both normal
  /// completion and unhandled exception. Otherwise the default continuation-based awaiter
  /// is used.
  auto final_suspend() const noexcept {
    if constexpr (HasFinalSuspendHook<promise_type>) {
      return this->FinalSuspendAwaiter();
    } else {
      struct Awaiter {
        /// @brief The current coroutine is done, so it is not ready to restart.
        /// @details await_suspend will be called next allowing to return the handle of the
        /// continuation.
        static bool await_ready() noexcept {
          return false;
        }

        void await_resume() const noexcept {}
        /**
         * @brief Suspension method for the ending co-routine.
         * @details The returned handle will be resumed.
         * Either this co-routine is at a root of a complete task tree,
         * such as when it is one of a group of parallel execution tasks,
         * or it is a branch. In the first case, a noop handle is returned,
         * otherwise the continuation handle is returned.
         *
         * @param h Handle of the current and ending co-routine.
         * @return The handle of the coroutine that continues the work if any.
         */
        std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> h) noexcept {
          auto& continuation = h.promise()._continuation;
          return std::visit<std::coroutine_handle<>>(
              []<typename VT>(VT const& cb) -> std::coroutine_handle<> {
                if constexpr (std::is_same_v<VT, CallBack>) {
                  return cb.f(cb.arg);
                } else {
                  return cb;
                }
              },
              continuation);
        }
      };
      return Awaiter{};
    }
  }

  void unhandled_exception() noexcept {
    _exception = std::current_exception();
  }

  void SetContinuation(std::coroutine_handle<> continuation) {
    _continuation = continuation;
  }

  void SetContinuation(std::coroutine_handle<> (*f)(void*), void* arg) {
    _continuation = CallBack{f, arg};
  }

  auto GetException() const noexcept {
    return _exception;
  }

 private:
  struct CallBack {
    std::coroutine_handle<> (*f)(void*);
    void* arg;
  };
  std::exception_ptr _exception{};
  std::variant<std::coroutine_handle<>, CallBack> _continuation{std::noop_coroutine()};
};

template <typename T>
constexpr bool IsTaskDef = false;

template <typename T, typename... BaseTypes>
constexpr bool IsTaskDef<Task<T, BaseTypes...>> = true;

template <typename T>
concept IsTask = IsTaskDef<std::decay_t<T>>;

} // namespace mochi
