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

#include <functional>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <utility>

namespace mochi {

namespace guarded_details {

template <typename ValueT, typename Traits>
class GuardedImpl;

/// @brief Trait that detects whether a type is any @ref GuardedImpl specialization.
template <typename T>
struct IsGuardedImpl : std::false_type {};

template <typename U, typename OtherTraits>
struct IsGuardedImpl<GuardedImpl<U, OtherTraits>> : std::true_type {};

template <typename T>
inline constexpr bool kIsGuardedImpl = IsGuardedImpl<std::decay_t<T>>::value;

/**
 * @brief Lock policy where both reads and writes take an exclusive lock.
 *
 * @tparam MutexT The underlying mutex type (e.g. @c std::mutex, @c std::recursive_mutex).
 */
template <typename MutexT>
struct GuardTraits {
  using GuardType = MutexT;
  using WriteLockType = std::unique_lock<MutexT>;
  using ReadLockType = std::unique_lock<MutexT>;
};

/**
 * @brief Lock policy where writes are exclusive but reads may run concurrently.
 *
 * @tparam MutexT The underlying shared mutex type (e.g. @c std::shared_mutex).
 */
template <typename MutexT>
struct SharedGuardTraits {
  using GuardType = MutexT;
  using WriteLockType = std::unique_lock<MutexT>;
  using ReadLockType = std::shared_lock<MutexT>;
};

/**
 * @brief Wraps a value together with the lock that guards it, so the mutex<->data relationship is
 * explicit and access is only possible while the lock is held.
 *
 * @tparam ValueT The guarded value type.
 * @tparam Traits Lock policy supplying the mutex and read/write lock types
 * (see @ref GuardTraits, @ref SharedGuardTraits).
 *
 * @note Prefer the @ref Guarded, @ref SharedGuarded, and @ref RecursiveGuarded aliases over
 * naming this type directly.
 */
template <typename ValueT, typename Traits>
class GuardedImpl final {
  using Self = GuardedImpl<ValueT, Traits>;

  // Different GuardedImpl instantiations need to access each other's members for cross-traits
  // moves.
  template <typename U, typename OtherTraits>
  friend class GuardedImpl;

 public:
  using ValueType = ValueT;
  using GuardType = typename Traits::GuardType;
  using WriteLockType = typename Traits::WriteLockType;
  using ReadLockType = typename Traits::ReadLockType;

  /**
   * @brief Constructs the guarded value in place by forwarding the arguments to its constructor.
   *
   * @note The single same-type case is excluded so that copy/move of @ref GuardedImpl resolve to
   * the dedicated (copy-deleted / move) constructors instead of being hijacked here.
   */
  template <
      typename... Params,
      typename = std::enable_if_t<
          !(sizeof...(Params) == 1 && (std::is_same_v<std::decay_t<Params>, Self> && ...))>>
  GuardedImpl(Params&&... params) : _value(std::forward<Params>(params)...) {}

  /**
   * @brief Copy-assigns @p rhs into the value under an exclusive lock.
   */
  Self& operator=(ValueType const& rhs) {
    WriteLockType lk(_guard);
    _value = rhs;
    return *this;
  }

  /**
   * @brief Move-assigns @p rhs into the value under an exclusive lock.
   */
  Self& operator=(ValueType&& rhs) {
    WriteLockType lk(_guard);
    _value = std::move(rhs);
    return *this;
  }

  /** @brief Copy construction is deleted; the guard is not copyable. */
  GuardedImpl(Self const&) = delete;

  /** @brief Move constructs from @p other, locking @p other for the move. */
  GuardedImpl(Self&& other) noexcept(std::is_nothrow_move_constructible_v<ValueType>)
      : GuardedImpl(std::move(other), WriteLockType(other._guard)) {}

  ~GuardedImpl() = default;

  /** @brief Copy assignment is deleted; the guard is not copyable. */
  Self& operator=(Self const&) = delete;

  /** @brief Move assigns from @p rhs, locking both guards. */
  Self& operator=(Self&& rhs) noexcept(std::is_nothrow_move_assignable_v<ValueType>) {
    if (this != &rhs) {
      std::scoped_lock lock(_guard, rhs._guard); // Prevents deadlocks
      _value = std::move(rhs._value);
    }
    return *this;
  }

  /** @brief Move assigns from a guard with a different lock policy, locking both guards. */
  template <typename OtherGuardTraits>
  Self& operator=(GuardedImpl<ValueT, OtherGuardTraits>&& rhs) {
    std::scoped_lock lock(_guard, rhs._guard); // Prevents deadlocks
    _value = std::move(rhs._value);
    return *this;
  }

  /** @brief Copy assigns from a guard with a different lock policy, locking both guards. */
  template <typename OtherGuardTraits>
  Self& operator=(GuardedImpl<ValueT, OtherGuardTraits> const& rhs) {
    std::scoped_lock lock(_guard, rhs._guard); // Prevents deadlocks
    _value = rhs._value;
    return *this;
  }

  /** @brief Assigns a new value under an exclusive lock. */
  template <
      typename U,
      typename = std::enable_if_t<std::is_convertible_v<U, ValueType> && !kIsGuardedImpl<U>>>
  Self& operator=(U&& rhs) {
    WriteLockType lock(_guard);
    _value = std::forward<U>(rhs);
    return *this;
  }

  /** @brief Returns a copy of the value, taken under a read lock. */
  operator ValueType() const {
    ReadLockType lock(_guard);
    return _value;
  }

  /**
   * @brief Runs @p func on the value under an exclusive lock, allowing in-place mutation.
   *
   * @return Whatever @p func returns, by value (references are decayed to copies).
   */
  template <typename Function, typename... Params>
  auto Mutate(Function func, Params&&... params) {
    WriteLockType lock(_guard);
    return std::invoke(func, _value, std::forward<Params>(params)...);
  }

  /**
   * @brief Like @ref Mutate, but also passes the held lock to @p func (e.g. to release it early).
   */
  template <typename Function, typename... Params>
  auto MutateWithLock(Function func, Params&&... params) {
    WriteLockType lock(_guard);
    return std::invoke(func, lock, _value, std::forward<Params>(params)...);
  }

  /**
   * @brief Tries to lock and, only if successful, runs @p func on the value.
   *
   * @return @c true if the lock was acquired and @p func ran, @c false otherwise.
   */
  template <typename Function, typename... Params>
  bool TryMutate(Function func, Params&&... params) {
    WriteLockType lock(_guard, std::try_to_lock);
    if (lock.owns_lock()) {
      std::invoke(func, _value, std::forward<Params>(params)...);
      return true;
    }
    return false;
  }

  /**
   * @brief Runs @p func on the value under a read lock, without copying it.
   *
   * @return Whatever @p func returns, by value (references are decayed to copies).
   */
  template <typename Function, typename... Params>
  auto Read(Function func, Params&&... params) const {
    ReadLockType lock(_guard);
    return std::invoke(func, _value, std::forward<Params>(params)...);
  }

  /** @brief Like @ref Read, but also passes the held lock to @p func (e.g. to release it early). */
  template <typename Function, typename... Params>
  auto ReadWithLock(Function func, Params&&... params) const {
    ReadLockType lock(_guard);
    return std::invoke(func, lock, _value, std::forward<Params>(params)...);
  }

  /**
   * @brief Runs @p func on the value WITHOUT locking. Only safe when no concurrent mutation
   * is possible.
   */
  template <typename Function, typename... Params>
  auto UnsafeRead(Function func, Params&&... params) const {
    return std::invoke(func, _value, std::forward<Params>(params)...);
  }

  /**
   * @brief Reads a single data member of the value under a read lock.
   */
  template <
      typename MemberObjectPointer,
      typename = std::enable_if_t<std::is_member_object_pointer_v<MemberObjectPointer>>>
  auto LoadMemberObject(MemberObjectPointer mop) const {
    return Read(mop);
  }

  /**
   * @brief Writes a single data member of the value under an exclusive lock.
   */
  template <typename MemberObjectPointer, typename U>
  auto StoreMemberObject(MemberObjectPointer mop, U&& newVal)
      -> std::enable_if_t<std::is_member_object_pointer_v<MemberObjectPointer>> {
    WriteLockType lock(_guard);
    std::invoke(mop, _value) = std::forward<U>(newVal);
  }

  /**
   * @brief Atomically swaps a single data member of the value, returning its previous value.
   */
  template <typename MemberObjectPointer, typename U>
  auto ExchangeMemberObject(MemberObjectPointer mop, U&& newVal) {
    static_assert(std::is_member_object_pointer_v<MemberObjectPointer>);
    WriteLockType lock(_guard);
    return std::exchange(std::invoke(mop, _value), std::forward<U>(newVal));
  }

  /** @brief Replaces the value with @p v under an exclusive lock and returns the old value. */
  ValueType Exchange(ValueType v = ValueType()) {
    WriteLockType lock(_guard);
    return std::exchange(_value, std::move(v));
  }

  /** @brief Returns a copy of the value, taken under a read lock. */
  ValueType Load() const {
    return *this;
  }

  /** @brief Assigns @p v to the value under an exclusive lock. */
  void Store(ValueType v) {
    operator=(std::move(v));
  }

 private:
  // Delegated-to by the public move constructor. Taking the lock by value keeps @p other locked for
  // the whole move.
  GuardedImpl(Self&& other, WriteLockType /*lockOther*/) noexcept(
      std::is_nothrow_move_constructible_v<ValueType>)
      : _value(std::move(other._value)) {}

  mutable GuardType _guard;
  ValueType _value;
};

} // namespace guarded_details

/// @brief A value guarded by an exclusive mutex; all reads and writes are serialized.
template <typename ValueType, typename LockableType = std::mutex>
using Guarded = guarded_details::GuardedImpl<ValueType, guarded_details::GuardTraits<LockableType>>;

/// @brief A value guarded by a shared mutex; @ref Read may proceed concurrently while
/// @ref Mutate is exclusive.
template <typename ValueType, typename LockableType = std::shared_mutex>
using SharedGuarded =
    guarded_details::GuardedImpl<ValueType, guarded_details::SharedGuardTraits<LockableType>>;

/// @brief A value guarded by a recursive mutex; the same thread may re-enter the lock.
template <typename ValueType>
using RecursiveGuarded =
    guarded_details::GuardedImpl<ValueType, guarded_details::GuardTraits<std::recursive_mutex>>;

} // namespace mochi
