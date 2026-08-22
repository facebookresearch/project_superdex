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

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <functional>
#include <iterator>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mochi {

using CallbackId = uint64_t;
constexpr CallbackId kInvalidCallbackId = 0;
constexpr int kCallbackPriorityMax = INT32_MIN;
constexpr int kCallbackPriorityHigh = 1000;
constexpr int kCallbackPriorityNormal = 2000;
constexpr int kCallbackPriorityLow = 3000;
constexpr int kCallbackPriorityLowest = INT32_MAX;

namespace details {

template <typename T>
class CallbackSetBase;

template <typename RetT, typename... ParamTs>
class CallbackSetBase<RetT(ParamTs...)> {
 public:
  using func_sig = RetT(ParamTs...);
  using function_type = std::function<func_sig>;

  static_assert(
      ((std::is_lvalue_reference<ParamTs>::value &&
        std::is_const<std::remove_reference_t<ParamTs>>::value) &&
       ...),
      "CallbackSet parameters must be const lvalue references.");

 protected:
  struct Callback {
    function_type callback;
    int priority;
    std::atomic<bool> isRegistered{true};
    Callback(function_type&& callback, int priority)
        : callback{std::move(callback)}, priority{priority} {}
    Callback(Callback&& rhs) noexcept
        : callback(std::move(rhs.callback)),
          priority(std::move(rhs.priority)),
          isRegistered(rhs.isRegistered.exchange(false)) {
      rhs.callback = {};
    }
  };

 protected:
  // Takes unique_locks as arguments so you can't forget to lock the data mutex and calling mutex
  void EraseDeregisteredCallbacks(
      std::unique_lock<std::recursive_mutex> const& /*callingLock*/,
      std::unique_lock<std::mutex> const& /*dataLock*/
  ) {
    if (!_deregisteredCallbacksDirty) {
      return;
    }
    MOCHI_ASSERT(_callDepth == 0, "Can't erase deregistered callbacks while calling!");
    // Erase any callbacks that were deregistered.
    for (auto it{_callbacks.cbegin()}; it != _callbacks.cend();) {
      if (!it->second.isRegistered) {
        it = _callbacks.erase(it);
      } else {
        ++it;
      }
    }
    _deregisteredCallbacksDirty = false;
    _orderedArrayDirty = true; // Ordered array should be compacted too.
  }

  // Takes a unique_lock as an argument so you can't forget to lock the data mutex
  void RebuildOrderedArray(
      std::unique_lock<std::recursive_mutex> const& callingLock,
      std::unique_lock<std::mutex> const& dataLock) {
    if (_orderedArrayDirty) {
      if (_deregisteredCallbacksDirty && (_callDepth == 0)) {
        EraseDeregisteredCallbacks(callingLock, dataLock);
      }

      _callbacksInPriorityOrder.clear();
      _callbacksInPriorityOrder.reserve(_callbacks.size());

      // Pull the callback functions out of the CallbackId->Callback map
      std::transform(
          _callbacks.cbegin(),
          _callbacks.cend(),
          std::back_inserter(_callbacksInPriorityOrder),
          [](auto& a) { return &a.second; });

      std::sort(
          _callbacksInPriorityOrder.begin(),
          _callbacksInPriorityOrder.end(),
          [](Callback const* a, Callback const* b) { return a->priority < b->priority; });

      _orderedArrayDirty = false;
    }
  }

 public:
  /**
   * @brief Default constructor
   */
  CallbackSetBase() = default;

  /**
   * @brief Move constructor
   *
   * @param[in,out] rhs Another @ref CallbackSetBase.
   *
   * @details Takes ownership of callback registration from the rhs object.
   *
   * @warning This operation is illegal while callbacks are executing.
   */
  CallbackSetBase(CallbackSetBase&& rhs) noexcept {
    *this = std::move(rhs);
  }

  /**
   * @brief Move assignment
   *
   * @param[in,out] rhs Another @ref CallbackSetBase.
   *
   * @details Deregisters all previous callbacks and takes ownership of callback registration from
   * the rhs object.
   *
   * @warning This operation is illegal while callbacks are executing.
   */
  CallbackSetBase& operator=(CallbackSetBase&& rhs) noexcept {
    if (&rhs != this) {
      // Lock all 4 mutexes at the same time.
      auto rhsCallingLock = std::unique_lock{rhs._callbackCallingMutex, std::defer_lock};
      auto rhsDataLock = std::unique_lock{rhs._callbackDataMutex, std::defer_lock};
      auto thisCallingLock = std::unique_lock{_callbackCallingMutex, std::defer_lock};
      auto thisDataLock = std::unique_lock{_callbackDataMutex, std::defer_lock};
      std::lock(rhsCallingLock, rhsDataLock, thisCallingLock, thisDataLock);
      MOCHI_ASSERT(
          rhs._callDepth == 0 && _callDepth == 0, "Can't move assign a CallbackSet while calling!");
      _deregisteredCallbacksDirty.store(rhs._deregisteredCallbacksDirty.exchange(false));
      _callbacksInPriorityOrder = std::move(rhs._callbacksInPriorityOrder);
      _callbacks = std::move(rhs._callbacks);
      _orderedArrayDirty.store(rhs._orderedArrayDirty.exchange(false));
      // Clear rhs arrays to ensure portable behavior after move assignment.
      rhs._callbacksInPriorityOrder.clear();
      rhs._callbacks.clear();
    }
    return *this;
  }

  /**
   * @brief Check if there are zero callbacks registered.
   *
   * @return True iff there are zero callbacks registered.
   */
  bool IsEmpty() const {
    std::unique_lock<std::mutex> const dataLock{_callbackDataMutex};
    return _callbacks.empty();
  }

  /**
   * @brief Deregister all callbacks immediately.
   *
   * @note If a callback has already started on this thread, then it will run to completion.
   * @note If a callback is executing on another thread, then @ref Clear will wait (blocking).
   * @note After @ref Clear, no previously registered callbacks will start again on any thread.
   */
  void Clear() {
    std::lock(_callbackCallingMutex, _callbackDataMutex);
    std::unique_lock<std::recursive_mutex> const callingLock{
        _callbackCallingMutex, std::adopt_lock};
    std::unique_lock<std::mutex> const dataLock{_callbackDataMutex, std::adopt_lock};

    if (_callDepth > 0) {
      for (auto& cb : _callbacks) {
        cb.second.isRegistered = false;
      }
      _deregisteredCallbacksDirty = true;
    } else {
      _callbacks.clear();
      _deregisteredCallbacksDirty = false;
    }
    _orderedArrayDirty = true;

    // locks go out of scope and unlock
  }

  /**
   * @brief Destructor
   *
   * @warning It is illegal to destroy a @ref CallbackSetBase while callbacks are executing.
   */
  virtual ~CallbackSetBase() {
    MOCHI_ASSERT(_callDepth == 0);
    Clear();
  }

  /**
   * @brief Register a callback with the given @ref CallbackId and priority.
   *
   * @param[in] newId    Unique @ref CallbackId for this callback. Must not already be registered.
   * @param[in] func     Callback function to invoke on each @ref Call.
   * @param[in] priority Execution priority; lower values run first.
   *                     Defaults to @ref kCallbackPriorityNormal.
   *
   * @note Registering during a @ref Call is safe and does not block.
   * @note Newly registered callbacks are not executed until the next @ref Call.
   */
  void Register(CallbackId newId, function_type func, int priority = kCallbackPriorityNormal) {
    std::unique_lock<std::mutex> const dataLock{_callbackDataMutex};
    MOCHI_ASSERT(newId != kInvalidCallbackId);
    MOCHI_ASSERT(_callbacks.find(newId) == _callbacks.end());

    // Insert into map.
    _callbacks.emplace(newId, Callback{std::move(func), priority});
    _orderedArrayDirty = true;
  }

  /**
   * @brief Deregister a single callback by @ref CallbackId.
   *
   * @param[in,out] id Identifies a callback to deregister. Will be set to @ref
   * kInvalidCallbackId if the callback was found.
   *
   * @note If id is @ref kInvalidCallbackId, then nothing happens.
   * @note If a callback function deregisters itself, then that function will still run to
   * completion.
   * @note If a callback is executing on another thread, then @ref Deregister will wait (blocking).
   * @note After @ref Deregister, the callback will not start again on any thread.
   */
  void Deregister(CallbackId& id) {
    if (id == kInvalidCallbackId) {
      return;
    }
    std::lock(_callbackCallingMutex, _callbackDataMutex);
    std::unique_lock<std::recursive_mutex> const callingLock{
        _callbackCallingMutex, std::adopt_lock};
    std::unique_lock<std::mutex> const dataLock{_callbackDataMutex, std::adopt_lock};

    if (_callbacks.count(id)) {
      // If _callDepth > 0, callbacks are currently executing; defer erasure.
      if (_callDepth > 0) {
        _callbacks.at(id).isRegistered = false;
        _deregisteredCallbacksDirty = true;
      } else {
        _callbacks.erase(id);
      }
      _orderedArrayDirty = true;
      id = kInvalidCallbackId;
    }

    // locks go out of scope and unlock
  }

  /**
   * @brief Execute all registered callbacks in priority order.
   *
   * @note Callback arguments must be const lvalue references.
   * @note If a callback function deregisters itself, that callback function will run to completion.
   * @note If a callback function deregisters another callback that has not started yet, then the
   * other callback will be skipped.
   * @note If a callback function registers additional callbacks, they will not be executed until
   * the next @ref Call or @ref operator().
   * @note If a callback function calls @ref Call or @ref operator(), then a new round of callback
   * execution will begin immediately. Afterward, the original list of callbacks will resume where
   * they left off.
   */
  void Call(ParamTs... param) {
    Call(param..., [](auto&&...) {}); // Use empty predicate
  }

  /**
   * @brief Execute all registered callbacks in priority order. Same as @ref Call.
   */
  void operator()(ParamTs... param) {
    Call(param..., [](auto&&...) {}); // Use empty predicate
  }

  /**
   * @brief Execute all registered callbacks in priority order. Follow each callback with a call to
   * the specified predicate function.
   *
   * @param[in] pred Predicate function to execute after each callback. If the callback returns a
   * value (not void), that value is forwarded to the predicate.
   *
   * @note Callback arguments must be const lvalue references.
   * @note If a callback function deregisters itself, that callback function will run to completion.
   * @note If a callback function deregisters another callback that has not started yet, then the
   * other callback will be skipped.
   * @note If a callback function registers additional callbacks, they will not be executed until
   * the next @ref Call or @ref operator().
   * @note If a callback function calls @ref Call or @ref operator(), then a new round of callback
   * execution will begin immediately. Afterward, the original list of callbacks will resume where
   * they left off.
   */
  template <class Predicate>
  void Call(ParamTs... param, Predicate&& pred) {
    std::vector<Callback const*> callbacksToCall;
    {
      // Lock _callbackCallingMutex and _callbackDataMutex at the same time.
      // Then capture the list of callbacks in priority order.
      // Then unlock _callbackDataMutex only.
      auto callingLock = std::unique_lock{_callbackCallingMutex, std::defer_lock};
      {
        auto dataLock = std::unique_lock{_callbackDataMutex, std::defer_lock};
        std::lock(callingLock, dataLock);
        RebuildOrderedArray(callingLock, dataLock);
        callbacksToCall = _callbacksInPriorityOrder;
      }

      // Increment _callDepth to detect re-entry (on any thread)
      ++_callDepth;

      // Cleanup will run at the end of scope, even if a callback throws.
      auto cleanup = [&]() {
        // Decrement _callDepth. If we decrement the value to zero, then it means that
        // there are no other callers (on any thread), and we know that new callers
        // cannot start until we release _callbackCallingMutex. This is a safe time to
        // clean up any deregistered callbacks.
        bool const wasLastCaller = ((--_callDepth) == 0);
        if (wasLastCaller && _deregisteredCallbacksDirty) {
          // callingLock is already held — only reacquire _callbackDataMutex.
          auto dataLock = std::unique_lock{_callbackDataMutex};
          EraseDeregisteredCallbacks(callingLock, dataLock);
        }
      };
      MOCHI_DEFER(cleanup());

      // Fire callbacks
      for (auto& callback : callbacksToCall) {
        if (callback->isRegistered) {
          if constexpr (std::is_void<RetT>::value) {
            callback->callback(param...);
            pred();
          } else {
            pred(callback->callback(param...));
          }
        }
      }

      // Mutexes will be unlocked here.
    }
    // callbacksToCall will be destroyed here.
  }

  /**
   * @brief Execute all registered callbacks in priority order. Follow each callback with a call to
   * the specified predicate function. Same as @ref Call.
   */
  template <class Predicate>
  void operator()(ParamTs... param, Predicate&& pred) {
    Call(param..., std::forward<Predicate>(pred));
  }

 protected:
  // Separate mutexes since it's safe to register a callback while the CallbackSet is being called.
  mutable std::recursive_mutex _callbackCallingMutex;
  mutable std::mutex _callbackDataMutex;

  mutable std::atomic<int> _callDepth{0};
  mutable std::atomic<bool> _deregisteredCallbacksDirty{false};

  mutable std::vector<Callback const*> _callbacksInPriorityOrder;

 private:
  std::unordered_map<CallbackId, Callback> _callbacks;
  mutable std::atomic<bool> _orderedArrayDirty{false};
};

} // namespace details

template <typename T>
class CallbackSet : public mochi::details::CallbackSetBase<T> {};

template <typename RetT, typename... ParamTs>
class CallbackSet<RetT(ParamTs...)> : public mochi::details::CallbackSetBase<RetT(ParamTs...)> {};

} // namespace mochi
