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
#include <utility>

namespace mochi {

/** @brief A coroutine-based generator type that yields values of type T.
 *  @details This class implements a generator pattern using C++20 coroutines.
 *   It allows for lazy evaluation of sequences, where values are computed
 *   on-demand as they are requested by the consumer.
 *
 *   Usage example:
 *   ```
 *   Generator<int> fibonacci() {
 *     int a = 0, b = 1;
 *     while (true) {
 *       co_yield a;
 *       int tmp = a;
 *       a = b;
 *       b = tmp + b;
 *     }
 *   }
 *   ```
 *
 *   The generator can be consumed using range-based for loops or
 *   by manually calling next() to advance the generator.
 */
template <typename T>
class Generator {
 public:
  /** @brief Promise type required by the coroutine machinery.
   *  @details This class implements the promise interface required by C++20
   *   coroutines. It handles the yielding of values and manages the lifetime
   *   of the coroutine.
   */
  struct promise_type {
    /** @brief Get the return object for the coroutine.
     *  @return A Generator instance that wraps the coroutine handle.
     */
    Generator get_return_object() {
      return Generator(std::coroutine_handle<promise_type>::from_promise(*this));
    }

    /** @brief Initial suspend point for the coroutine.
     *  @return A suspend_always object to make the generator lazy.
     */
    std::suspend_always initial_suspend() noexcept {
      return {};
    }

    /** @brief Final suspend point for the coroutine.
     *  @return A suspend_always object to allow the consumer to detect completion.
     */
    std::suspend_always final_suspend() noexcept {
      return {};
    }

    /** @brief Handle unhandled exceptions in the coroutine.
     *  @details This method is called when an exception is thrown in the coroutine
     *   and not caught. It stores the exception to be rethrown when the consumer
     *   tries to advance the generator.
     */
    void unhandled_exception() {
      _exception = std::current_exception();
    }

    void RethrowIfException() const {
      if (_exception) {
        std::rethrow_exception(_exception);
      }
    }

    /** @brief Handle the co_yield expression.
     *  @param value The value being yielded by the coroutine.
     *  @return A suspend_always object to pause the coroutine until the next value is requested.
     */
    std::suspend_always yield_value(std::remove_reference_t<T>& value) noexcept
      requires(!std::is_rvalue_reference_v<T>)
    {
      _currentValue = std::addressof(value);
      return {};
    }

    std::suspend_always yield_value(std::remove_reference_t<T>&& value) noexcept {
      _currentValue = std::addressof(value);
      return {};
    }
    /** @brief Handle the co_return statement (with no value).
     *  @details This is called when the coroutine reaches a co_return statement
     *   with no value, indicating the end of the sequence.
     */
    std::suspend_never return_void() {
      return {};
    }

    // Current value being yielded
    T* _currentValue{};
    // Exception pointer for propagating exceptions to the consumer
    std::exception_ptr _exception{};
  };

  /** @brief Constructor that takes a coroutine handle.
   *  @param handle The coroutine handle to wrap.
   */
  explicit Generator(std::coroutine_handle<promise_type> handle) {
    _handle = handle;
  }

  /** @brief Move constructor.
   *  @param other The generator to move from.
   */
  Generator(Generator&& other) noexcept {
    _handle = other._handle;
    other._handle = nullptr;
  }

  /** @brief Destructor that cleans up the coroutine if it's still active.
   */
  ~Generator() {
    if (_handle) {
      _handle.destroy();
    }
  }

  // Delete copy operations
  Generator(Generator const&) = delete;
  Generator& operator=(Generator const&) = delete;

  /** @brief Move assignment operator.
   *  @param other The generator to move from.
   *  @return Reference to this generator.
   */
  Generator& operator=(Generator&& other) noexcept;

  /** @brief Advance the generator to the next value.
   *  @return true if a new value is available, false if the generator is exhausted.
   *  @throws Any exception that was thrown inside the coroutine.
   */
  bool Next() {
    _handle.resume();
    _handle.promise().RethrowIfException();
    return !_handle.done();
  }

  /** @brief Get the current value of the generator.
   *  @return Reference to the current value.
   *  @note This method should only be called after next() returns true.
   */
  T const& Value() const {
    return *_handle.promise()._currentValue;
  }

  /** @brief Check if the generator has been exhausted.
   *  @return true if the generator is done, false otherwise.
   */
  bool Done() const {
    return _handle.done();
  }

  /** @brief Iterator class for the generator. */
  class Iterator {
   public:
    Iterator(std::coroutine_handle<promise_type> coro, bool done) : _handle(coro), _done(done) {}

    Iterator& operator++() {
      _handle.resume();
      _handle.promise().RethrowIfException();
      _done = _handle.done();
      return *this;
    }

    bool operator==(Iterator const& right) const {
      return _done == right._done;
    }

    T& operator*() const {
      return *_handle.promise()._currentValue;
    }
    T* operator->() const {
      return &(operator*());
    }

   private:
    std::coroutine_handle<promise_type> _handle;
    bool _done;
  };

  Iterator begin() const {
    _handle.resume();
    _handle.promise().RethrowIfException();
    return Iterator(_handle, _handle.done());
  }

  Iterator end() const noexcept {
    return Iterator(_handle, true);
  }

 private:
  std::coroutine_handle<promise_type> _handle;
};

} // namespace mochi
