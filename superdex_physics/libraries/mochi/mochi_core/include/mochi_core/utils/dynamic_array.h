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

#include <mochi_core/memory/allocator.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/reflection.h>

#if MOCHI_LANGUAGE_CPP20
#include <concepts>
#endif
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mochi {

// Define MOCHI_DARRAY_DEBUG to 1 to enable safety checks in DynamicArray
#ifndef MOCHI_DARRAY_DEBUG
#define MOCHI_DARRAY_DEBUG MOCHI_DEBUG
#endif

#if MOCHI_DARRAY_DEBUG && MOCHI_ASSERT_ENABLED
#define MOCHI_DARRAY_ASSERT(cond, ...) MOCHI_ASSERT(cond, __VA_ARGS__)
#else
#define MOCHI_DARRAY_ASSERT(cond, ...)
#endif

#if MOCHI_LANGUAGE_CPP20
// Only until unrealios-android-mochiunreal gets an updated compiler that supports the
// std::forward_iterator concept.
template <class From, class To>
concept ConvertibleTo =
    std::is_convertible_v<From, To> && requires { static_cast<To>(std::declval<From>()); };
template <typename T>
concept HasDistance = requires(T t) {
  { std::distance(t, t) } -> ConvertibleTo<std::ptrdiff_t>;
};
#define MOCHI_DARRAY_ASSERT_FORWARD_ITERATOR(IteratorType)                                       \
  static_assert(                                                                                 \
      HasDistance<IteratorType>,                                                                 \
      "This type of input iterator is not supported by DynamicArray because there is no way to " \
      "reserve sufficient memory up front. We would have to loop calling emplace_back, which "   \
      "would be inefficient. Consider initializing this DynamicArray in a different way.");
#else
#define MOCHI_DARRAY_ASSERT_FORWARD_ITERATOR(IteratorType)                                       \
  static_assert(                                                                                 \
      std::is_base_of_v<                                                                         \
          std::forward_iterator_tag,                                                             \
          typename std::iterator_traits<IteratorType>::iterator_category>,                       \
      "This type of input iterator is not supported by DynamicArray because there is no way to " \
      "reserve sufficient memory up front. We would have to loop calling emplace_back, which "   \
      "would be inefficient. Consider initializing this DynamicArray in a different way.");
#endif

/** @brief Trait to determine if a type is safe for use with @ref DynamicArray::resize_noinit. */
template <typename T>
struct IsResizeNoInitSafe : std::bool_constant<std::is_trivially_copyable_v<T>> {};

/** @brief Helper variable template for @ref IsResizeNoInitSafe. */
template <typename T>
inline constexpr bool kIsResizeNoInitSafe = IsResizeNoInitSafe<T>::value;

/**
 * @brief A dynamically resizable array with syntax and behavior similar to std::pmr::vector
 *
 * @remarks
 *   - Optionally provide your own polymorphic allocator.
 *   - Compatible with FILO allocators as long as you reserve or resize sufficient memory.
 *   - Extends the std::pmr::vector syntax with new methods including resize_noinit and append.
 *
 * @warning DynamicArray does not provide equivalent exception safety guarantees as std::vector.
 * Exception-safe use assumes that element construction, assignment, destruction, move operations,
 * and allocator operations do not throw. For example:
 *   - Reallocation via emplace_back, push_back, append, assign, reserve, resize, or shrink_to_fit
 *     can leak memory or leave the array in an invalid state if T's move constructor throws.
 *   - Construction, append, assign, and resize can leak memory or leave the array in an invalid
 *     state if T's constructors or assignment operators throw.
 *   - Moving from a DynamicArray with a different allocator does not provide the same rollback or
 *     cleanup guarantees as std::vector if moving elements throws.
 *   - Allocator exceptions may leave the array in an invalid state in code paths that release old
 *     storage before allocating new storage.
 *
 * @tparam T Element value type.
 */
template <class T>
class DynamicArray {
 public:
  static_assert(
      !std::is_same_v<T, Allocator> && !std::is_same_v<T, Allocator*>,
      "DynamicArray of allocators is not allowed because of overload ambiguity.");

  using const_iterator = T const*;
  using iterator = T*;
  using value_type = T;
  using size_type = size_t;

  /**
   * @brief Construct an empty DynamicArray with the default allocator.
   */
  DynamicArray() = default;

  /**
   * @brief Construct an empty DynamicArray with a specific allocator.
   *
   * @param allocator Pointer to a polymorphic allocator. Must outlive this DynamicArray object.
   */
  explicit DynamicArray(Allocator* allocator) : _allocator(allocator) {}

  /**
   * @brief Construct a DynamicArray with an initial size. All elements will be default constructed
   * or zero-initialized (for POD types).
   *
   * @param size Initial size (number of elements, not bytes)
   * @param allocator Pointer to a polymorphic allocator. Must outlive this DynamicArray object.
   */
  explicit DynamicArray(size_type size, Allocator* allocator = GetDefaultAllocator())
      : _begin(static_cast<T*>(allocator->allocate(size * sizeof(T), alignof(T)))),
        _end(_begin + size),
        _endCapacity(_end),
        _allocator(allocator) {
    DefaultConstructRange(_begin, _end);
  }

  /**
   * @brief Construct a DynamicArray filled with copies of the specified value.
   *
   * @tparam FromT Input value type. Must be same or convertible to type T.
   * @param size Initial size (number of elements, not bytes)
   * @param defaultValue Value to copy to all new array elements.
   * @param allocator Pointer to a polymorphic allocator. Must outlive this DynamicArray object.
   */
  template <typename FromT, MOCHI_CONCEPT((std::is_convertible_v<FromT, T>))>
  DynamicArray(
      size_type size,
      FromT const& defaultValue,
      Allocator* allocator = GetDefaultAllocator())
      : _begin(static_cast<T*>(allocator->allocate(size * sizeof(T), alignof(T)))),
        _end(_begin + size),
        _endCapacity(_end),
        _allocator(allocator) {
    CopyConstructRangeFromValue(_begin, _end, defaultValue);
  }

  /**
   * @brief Construct a DynamicArray and copy values from the half-open range:
   * [rangeBegin, rangeEnd)
   *
   * @tparam InputIt Input forward iterator type (typically deduced).
   * @param rangeBegin Iterator pointing to the first input value to copy.
   * @param rangeEnd Iterator pointing ONE PAST the last input value to copy.
   * @param allocator Pointer to a polymorphic allocator. Must outlive this DynamicArray object.
   */
  template <
      typename InputIt,
      MOCHI_CONCEPT((std::is_base_of_v<
                     std::input_iterator_tag,
                     typename std::iterator_traits<InputIt>::iterator_category>))>
  DynamicArray(InputIt rangeBegin, InputIt rangeEnd, Allocator* allocator = GetDefaultAllocator())
      : _allocator(allocator) {
    MOCHI_DARRAY_ASSERT_FORWARD_ITERATOR(InputIt);
    auto size = static_cast<size_type>(std::distance(rangeBegin, rangeEnd));
    _begin = static_cast<T*>(allocator->allocate(size * sizeof(T), alignof(T)));
    _end = _endCapacity = _begin + size;
    CopyConstructRange(_begin, rangeBegin, rangeEnd);
  }

  /**
   * @brief Construct a DynamicArray and copy values from a std::initializer_list.
   *
   * @param list List from which values will be copied.
   * @param allocator Pointer to a polymorphic allocator. Must outlive this DynamicArray object.
   */
  DynamicArray(std::initializer_list<T> const& list, Allocator* allocator = GetDefaultAllocator())
      : DynamicArray(std::begin(list), std::end(list), allocator) {}

  /**
   * @brief Construct a DynamicArray and copy values from another iterable container type like
   * mochi::Span, std::vector, std::list, etc...
   *
   * @tparam InputContainerT Input container type. Must support std::begin and std::end.
   * @param other Container from which values will be copied.
   * @param allocator Pointer to a polymorphic allocator. Must outlive this DynamicArray object.
   */
  template <
      typename InputContainerT,
      MOCHI_CONCEPT(
          (!std::is_same_v<std::decay_t<InputContainerT>, DynamicArray> &&
           sizeof(decltype(std::begin(std::declval<InputContainerT const&>()))) &&
           sizeof(decltype(std::end(std::declval<InputContainerT const&>())))))>
  explicit DynamicArray(InputContainerT const& other, Allocator* allocator = GetDefaultAllocator())
      : DynamicArray(std::begin(other), std::end(other), allocator) {}

  /**
   * @brief Construct a DynamicArray and copy values from another one.
   *
   * @param other Another DynamicArray from which values will be copied.
   * @param allocator Pointer to a polymorphic allocator. Must outlive this DynamicArray object.
   */
  DynamicArray(DynamicArray const& other, Allocator* allocator)
      : DynamicArray(other.begin(), other.end(), allocator) {}

  /**
   * @brief Copy construct from another DynamicArray. Use the same allocator.
   *
   * @param other Another DynamicArray from which the allocator and the values will be copied.
   */
  DynamicArray(DynamicArray const& other)
      : DynamicArray(other.begin(), other.end(), other._allocator) {}

  /**
   * @brief Construct a DynamicArray by moving memory or values from another one.
   *
   * @remarks If the allocators are equal, then array memory ownership will be transferred from
   * the other array to this one. Otherwise, new memory will be allocated for this array and then
   * the other array's values will be moved into it. Either way, the other array will be empty
   * after this call.
   *
   * @param other Another DynamicArray from which memory or values will be moved
   * @param allocator Pointer to a polymorphic allocator. Must outlive this DynamicArray object.
   */
  DynamicArray(DynamicArray&& other, Allocator* allocator) : _allocator(allocator) {
    if (other._allocator->is_equal(*_allocator))
      MOCHI_LIKELY {
        _begin = other._begin;
        _end = other._end;
        _endCapacity = other._endCapacity;
        other._begin = other._end = other._endCapacity = nullptr;
      }
    else {
      auto size = other.size();
      _begin = static_cast<T*>(_allocator->allocate(size * sizeof(T), alignof(T)));
      _end = _begin;
      _endCapacity = _begin + size;
      // Move construct each element individually, advancing _end after each successful
      // construction.
      for (auto src = other._begin; src < other._end; ++src, ++_end) {
        new (_end) T(std::move(*src)); // Move construct
      }
      other.clear();
    }
  }

  /**
   * @brief Move construct from another DynamicArray. Use the same allocator.
   * @remarks Memory ownership is always transferred. No new allocation.
   *
   * @param other Another DynamicArray from which memory will be moved. Will be empty after this
   * call.
   */
  DynamicArray(DynamicArray&& other) noexcept
      : _begin(other._begin),
        _end(other._end),
        _endCapacity(other._endCapacity),
        _allocator(other._allocator) {
    other._begin = other._end = other._endCapacity = nullptr;
  }

  /**
   * @brief Destroy the DynamicArray object and all of its elements. Deallocate any memory that
   * was used.
   */
  ~DynamicArray() {
    if (_begin != nullptr) {
      DestroyRange(_begin, _end);
      _allocator->deallocate(_begin, capacity() * sizeof(T), alignof(T));
#if MOCHI_DARRAY_DEBUG
      _begin = _end = _endCapacity = nullptr;
      _allocator = nullptr;
#endif // MOCHI_DARRAY_DEBUG
    }
  }

  /**
   * @brief Get an iterator pointing to the beginning of the array.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE iterator begin() {
    return _begin;
  }

  /**
   * @brief Get a const iterator pointing to the beginning of the array.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE const_iterator begin() const {
    return _begin;
  }

  /**
   * @brief Get a const iterator pointing to the beginning of the array.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE const_iterator cbegin() const {
    return _begin;
  }

  /**
   * @brief Get an iterator pointing to the next element AFTER the end of the array.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE iterator end() {
    return _end;
  }

  /**
   * @brief Get a const iterator pointing to the next element AFTER the end of the array.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE const_iterator end() const {
    return _end;
  }

  /**
   * @brief Get a const iterator pointing to the next element AFTER the end of the array.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE const_iterator cend() const {
    return _end;
  }

  /**
   * @brief Get a reference to the first element. Array must not be empty.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE T& front() {
    MOCHI_DARRAY_ASSERT(!empty(), "Cannot get front of empty array");
    return *_begin;
  }

  /**
   * @brief Get a const reference to the first element. Array must not be empty.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE T const& front() const {
    MOCHI_DARRAY_ASSERT(!empty(), "Cannot get front of empty array");
    return *_begin;
  }

  /**
   * @brief Get a reference to the last element. Array must not be empty.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE T& back() {
    MOCHI_DARRAY_ASSERT(!empty(), "Cannot get back of empty array");
    return *(_end - 1);
  }

  /**
   * @brief Get a const reference to the last element. Array must not be empty.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE T const& back() const {
    MOCHI_DARRAY_ASSERT(!empty(), "Cannot get back of empty array");
    return *(_end - 1);
  }

  /**
   * @brief Get a pointer to the first element. May be nullptr if this array is empty.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE T* data() {
    return _begin;
  }

  /**
   * @brief Get a const pointer to the first element. May be nullptr if this array is empty.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE T const* data() const {
    return _begin;
  }

  /**
   * @brief Get the number of elements in the array.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE size_type size() const {
    return _end - _begin;
  }

  /**
   * @brief Get the number of elements that could fit in the array without allocating more memory.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE size_type capacity() const {
    return _endCapacity - _begin;
  }

  /**
   * @brief Return true if the array is empty (zero size).
   */
  [[nodiscard]] MOCHI_FORCE_INLINE bool empty() const {
    return _end == _begin;
  }

  /**
   * @brief Set the size. If larger than the current size, new elements will be default constructed
   * (zero-initialized for POD types).
   *
   * @param newSize New number of elements
   *
   * @remarks The first time you call resize, capacity will be allocated for the exact size
   * requested. However, subsequent calls to resize may allocate more than requested, similar to
   * emplace_back.
   */
  void resize(size_type newSize) {
    if (newSize > size()) {
      if (newSize > capacity()) {
        GrowCapacity(_begin ? GetNextCapacity(newSize) : newSize);
      }
      DefaultConstructRange(_end, _begin + newSize);
    } else {
      DestroyRange(_begin + newSize, _end);
    }
    _end = _begin + newSize;
  }

  /**
   * @brief Set the size. If larger than the current size, new elements will be copy constructed
   * from the given value.
   *
   * @param newSize New number of elements
   * @param value Value to copy to new elements (if any).
   *
   * @warning The value parameter is not allowed to be a reference to an element within this
   * same DynamicArray. In that case, consider using a copy of the value instead.
   *
   * @remarks The first time you call resize, capacity will be allocated for the exact size
   * requested. However, subsequent calls to resize may allocate more than requested, similar to
   * emplace_back.
   */
  void resize(size_type newSize, T const& value) {
    if (newSize > size()) {
      if (newSize > capacity()) {
        GrowCapacity(_begin ? GetNextCapacity(newSize) : newSize);
      }
      CopyConstructRangeFromValue(_end, _begin + newSize, value);
    } else {
      DestroyRange(_begin + newSize, _end);
    }
    _end = _begin + newSize;
  }

  /**
   * @brief Set the size but do not initialize any new elements. They will be in an undefined state!
   *
   * @param newSize New number of elements
   *
   * @warning This method may be used as an optimization, but only if you can guarantee that any new
   * elements will be initialized before being used.
   *
   * @note Only supported for types satisfying @ref kIsResizeNoInitSafe.
   * @note The first time you call resize_noinit, capacity will be allocated for the exact size
   * requested. However, subsequent calls to resize_noinit may allocate more than requested, similar
   * to emplace_back.
   */
  void resize_noinit(size_type newSize) {
    static_assert(
        kIsResizeNoInitSafe<T>, "DynamicArray::resize_noinit is not supported for this type");
    size_type prevSize = size();
    if (newSize > prevSize) {
      if (newSize > capacity()) {
        GrowCapacity(_begin ? GetNextCapacity(newSize) : newSize);
      }
      _end = _begin + newSize;
      DebugFillWithNaN(_begin + prevSize, _end);
    } else {
      DestroyRange(_begin + newSize, _end);
      _end = _begin + newSize;
    }
  }

  /**
   * @brief Reset the state of this DynamicArray using the arguments from any constructor. Any
   * previous elements will be destroyed. Any previous memory will be deallocated.
   *
   * @remarks This method can be used to assign a new allocator pointer to a DynamicArray object
   * that has already been constructed (unlike move assignment, which does not move the allocator).
   *
   * @tparam Args Any argument types supported by one of the DynamicArray constructors.
   * @param args Any parameter types supported by one of the DynamicArray constructors.
   */
  template <typename... Args>
  void reset(Args&&... args) {
    this->~DynamicArray<T>();
    new (this) DynamicArray<T>(std::forward<Args>(args)...);
  }

  /**
   * @brief Ensure that this array has at least the specified capacity.
   *
   * @param newCapacity Desired capacity measured by number of elements (not bytes).
   *
   * @warning If you call reserve repeatedly with incrementally larger values, it will reallocate
   * the memory every time. It will not overallocate the memory the way emplace_back does.
   */
  void reserve(size_type newCapacity) {
    if (newCapacity > capacity()) {
      GrowCapacity(newCapacity);
    }
  }

  /**
   * @brief Set the size to zero. Does not deallocate memory.
   * @see shrink_to_fit
   */
  void clear() {
    DestroyRange(_begin, _end);
    _end = _begin;
  }

  /**
   * @brief Ensure that the capacity (memory allocated) is no larger than the size (memory used).
   */
  void shrink_to_fit() {
    if (_endCapacity > _end) {
      if (empty()) {
        _allocator->deallocate(_begin, capacity() * sizeof(T), alignof(T));
        _begin = _end = _endCapacity = nullptr;
      } else {
        auto const n = size();
        auto* newBegin = static_cast<T*>(_allocator->allocate(n * sizeof(T), alignof(T)));
        MoveConstructRange(newBegin, _begin, n);
        DestroyRange(_begin, _end);
        _allocator->deallocate(_begin, capacity() * sizeof(T), alignof(T));
        _begin = newBegin;
        _end = _endCapacity = newBegin + n;
      }
    }
  }

  /**
   * @brief Emplace a new element at the end of the array.
   *
   * @remarks If people use reserve or resize, then we will allocate the exact capacity requested.
   * However, if people call push_back or emplace_back repeatedly, we will grow the capacity by 50%
   * each time more memory is needed. This achieves amortized O(1) cost, similar to std::vector.
   *
   * Unlike std::vector, we skip past the first few reallocations by reserving a capacity of
   * kGrowthPatternStartSize the first time. See GetNextCapacity() for details.
   *
   * @warning The argument is not allowed to be a reference to an element within this same
   * DynamicArray. In that case, consider emplacing a copy of the element instead.
   *
   * @tparam Args Any types that can be passed to type T's constructor.
   * @param args Any arguments that can be passed to type T's constructor.
   */
  template <class... Args>
  void emplace_back(Args&&... args) {
    if constexpr (sizeof...(Args) == 1) {
      if constexpr (std::is_same_v<
                        T,
                        std::remove_cv_t<std::remove_reference_t<decltype(std::get<0>(
                            std::forward_as_tuple(args...)))>>>) {
        MOCHI_DARRAY_ASSERT(
            !IsInThisArray(&std::get<0>(std::forward_as_tuple(args...))),
            "You are not allowed to push an element of a DynamicArray onto the end of the same DynamicArray. "
            "Consider pushing a copy of the value instead.");
      }
    }
    if (_endCapacity == _end)
      MOCHI_UNLIKELY {
        GrowCapacity(GetNextCapacity(size() + 1));
      }
    new (_end) T(std::forward<Args>(args)...);
    ++_end;
  }

  /**
   * @brief Copy the value to a new element at the end of the array.
   *
   * @remarks May over allocate to achieve amortized O(1) cost like emplace_back.
   *
   * @warning The argument is not allowed to be a reference to an element within this same
   * DynamicArray. In that case, consider pushing a copy of the element instead.
   *
   * @param value Value to be copied
   * @see emplace_back
   */
  MOCHI_FORCE_INLINE void push_back(T const& value) {
    emplace_back(value); // emplace via copy constructor
  }

  /**
   * @brief Move the value to a new element at the end of the array.
   *
   * @remarks May over allocate to achieve amortized O(1) cost like emplace_back.
   *
   * @warning The argument is not allowed to be a reference to an element within this same
   * DynamicArray. In that case, consider pushing a copy of the element instead.
   *
   * @param value Value to be moved
   * @see emplace_back
   */
  void push_back(T&& value) {
    emplace_back(std::move(value)); // emplace via move constructor
  }

  /**
   * @brief Default construct a new element at the end of the array.
   * @remarks May over allocate to achieve amortized O(1) cost like emplace_back.
   *
   * @code{.cpp}
   * auto& obj = myArray.push_back();
   * obj.value = someValue;
   * @endcode
   *
   * @return T& A reference to the new element
   */
  T& push_back() {
    emplace_back();
    return back();
  }

  /**
   * @brief Remove the last element of the array (must not be empty).
   */
  void pop_back() {
    MOCHI_DARRAY_ASSERT(!empty(), "Cannot pop_back an empty array");
    --_end;
    if constexpr (!std::is_trivially_destructible_v<T>) {
      _end->~T();
    }
  }

  /**
   * @brief Remove the half-open range [rangeBegin, rangeEnd) from this array. Any later elements
   * will be shifted down to preserve order.
   *
   * @remarks Size will be reduced by (rangeEnd - rangeBegin).
   *
   * @param rangeBegin_ Iterator pointing to the first element to remove.
   * @param rangeEnd_ Iterator pointing ONE PAST the last element to remove.
   */
  void erase(const_iterator rangeBegin_, const_iterator rangeEnd_) {
    // Const cast so we don't have to implement both erase(iterator, iterator) and
    // erase(const_iterator, const_iterator).
    auto rangeBegin = const_cast<iterator>(rangeBegin_);
    auto rangeEnd = const_cast<iterator>(rangeEnd_);
    if (rangeEnd == _end) {
      MOCHI_DARRAY_ASSERT(rangeBegin >= _begin && rangeBegin <= _end, "Invalid iterator");
      // Erase from the end. This is the most common case.
      DestroyRange(rangeBegin, rangeEnd);
      _end = rangeBegin;
    } else if (rangeEnd != rangeBegin) {
      MOCHI_DARRAY_ASSERT(rangeBegin <= rangeEnd, "Invalid iterator range");
      MOCHI_DARRAY_ASSERT(IsInThisArray(rangeBegin, rangeEnd), "Invalid iterator");
      // Erase from the middle and shift elements down
      if constexpr (std::is_trivially_move_assignable_v<T>) {
        memmove(rangeBegin, rangeEnd, (_end - rangeEnd) * sizeof(T));
      } else {
        T* dst = rangeBegin;
        T* src = const_cast<T*>(rangeEnd); // First value to move, after the range to erase
        for (; src != _end; ++dst, ++src) {
          *dst = std::move(*src); // Move assignment
        }
      }
      auto numRemoved = (rangeEnd - rangeBegin);
      auto newSize = size() - numRemoved;
      DestroyRange(_begin + newSize, _end);
      _end = _begin + newSize;
    }
  }

  /**
   * @brief Remove one element from this array. Any later elements will be shifted down to preserve
   * order.
   *
   * @param it_ Iterator pointing to the element to remove.
   */
  void erase(const_iterator it_) {
    // Const cast so we don't have to implement both erase(iterator) and erase(const_iterator).
    auto it = const_cast<iterator>(it_);
    MOCHI_DARRAY_ASSERT(it >= _begin && it < _end, "Invalid iterator");
    if (it + 1 == _end) {
      // Erase from the end. This is the most common case.
      Destroy(it);
      _end = it;
    } else {
      erase(it, it + 1);
    }
  }

  /**
   * @brief Remove one element from this array and swap the last element into its place. This
   * achieves O(1) cost but does not preserve order.
   *
   * @param it Iterator pointing to the element to remove.
   */
  void erase_unordered(iterator it) {
    MOCHI_DARRAY_ASSERT(it >= _begin && it < _end, "Invalid iterator");
    auto* last = _end - 1;
    if (it != last) {
      *it = std::move(*last);
    }
    Destroy(last);
    _end = last;
  }

  /**
   * @brief Replace the contents of this DynamicArray by copying values from the half-open range
   * [rangeBegin, rangeEnd).
   *
   * @warning The iterators are not allowed to point to elements within this same DynamicArray.
   *
   * @tparam InputIt Input iterator type (typically deduced).
   * @param rangeBegin Iterator pointing to the first value to copy from.
   * @param rangeEnd Iterator pointing ONE PAST the last value to copy from.
   */
  template <
      typename InputIt,
      MOCHI_CONCEPT((std::is_base_of_v<
                     std::input_iterator_tag,
                     typename std::iterator_traits<InputIt>::iterator_category>))>
  void assign(InputIt rangeBegin, InputIt rangeEnd) {
    MOCHI_DARRAY_ASSERT_FORWARD_ITERATOR(InputIt);
    if constexpr (std::is_pointer_v<InputIt>) {
      MOCHI_DARRAY_ASSERT(rangeBegin <= rangeEnd, "Invalid input range");
      MOCHI_DARRAY_ASSERT(
          !IsInThisArray(rangeBegin, rangeEnd),
          "The input range is not allowed to point to memory within the DynamicArray that is being modified.");
    }
    auto const oldSize = size();
    auto const newSize = static_cast<size_type>(std::distance(rangeBegin, rangeEnd));
    MOCHI_DARRAY_ASSERT(
        _allocator || (newSize == oldSize),
        "It is illegal to change the size of a DynamicArray that does not own the memory.");
    if (newSize <= oldSize) {
      // This array is getting smaller
      _end = _begin + newSize;
      DestroyRange(_begin + newSize, _begin + oldSize);
      std::copy(rangeBegin, rangeEnd, _begin);
    } else if (newSize <= capacity()) {
      // This array is getting larger. Sufficient capacity exists.
      _end += (newSize - oldSize);
      if constexpr (std::is_trivially_copyable_v<T>) {
        // We can copy the full range without worrying about a user-defined constructor or
        // assignment operator.
        std::copy(rangeBegin, rangeEnd, _begin);
      } else {
        // Copy assign existing values
        auto src = rangeBegin;
        std::copy_n(src, oldSize, _begin);
        // Copy construct new values
        std::advance(src, oldSize);
        for (size_type i = oldSize; i < newSize; ++i, ++src) {
          new (_begin + i) T(*src);
        }
      }
    } else {
      // This array is getting larger and needs to allocate new capacity
      if constexpr (
          std::is_trivially_copyable_v<T> && std::is_trivially_move_constructible_v<T> &&
          std::is_trivially_move_assignable_v<T> && std::is_trivially_destructible_v<T>) {
        // We can deallocate the old memory before allocating the new memory because we don't need
        // to read nor destruct the old values. We can simply copy the full range to the new
        // memory location.
        if (_begin) {
          _allocator->deallocate(_begin, capacity() * sizeof(T), alignof(T));
        }
        _begin = static_cast<T*>(_allocator->allocate(newSize * sizeof(T), alignof(T)));
        _end = _endCapacity = _begin + newSize;
        std::copy(rangeBegin, rangeEnd, _begin);
      } else {
        // Allocate memory and move existing values.
        GrowCapacity(newSize); // exact capacity
        _end = _begin + newSize;
        // Copy assign over existing values
        auto src = rangeBegin;
        std::copy_n(src, oldSize, _begin);
        // Copy construct new values
        std::advance(src, oldSize);
        CopyConstructRange(_begin + oldSize, src, rangeEnd);
      }
    }
  }

  /**
   * @brief Add values to the end of this DynamicArray by copying them from the half-open range
   * [rangeBegin, rangeEnd).
   *
   * @warning The iterators are not allowed to point to elements within this same DynamicArray.
   * Consider appending a copy of the elements instead.
   *
   * @tparam InputIt Input iterator type (typically deduced).
   * @param rangeBegin Iterator pointing to the first input value to copy.
   * @param rangeEnd Iterator pointing ONE PAST the last input value to copy.
   */
  template <class InputIt>
  void append(InputIt rangeBegin, InputIt rangeEnd) {
    MOCHI_DARRAY_ASSERT_FORWARD_ITERATOR(InputIt);
    MOCHI_DARRAY_ASSERT(
        !IsInThisArray(rangeBegin, rangeEnd),
        "The input range is not allowed to point to memory within the DynamicArray that is being modified.");
    auto count = std::distance(rangeBegin, rangeEnd);
    MOCHI_DARRAY_ASSERT(count >= 0, "Invalid input range");
    auto prevSize = size();
    auto newSize = prevSize + count;
    if (newSize > capacity())
      MOCHI_UNLIKELY {
        GrowCapacity(GetNextCapacity(newSize));
      }
    _end = _begin + newSize;
    CopyConstructRange(_begin + prevSize, rangeBegin, rangeEnd);
  }

  /**
   * @brief Add values to the end of this DynamicArray by copying them from another iterable
   * container.
   *
   * @warning It is illegal to append a DynamicArray to itself.
   *
   * @tparam InputContainerT Another iterable container type. Must support std::begin and
   * std::end.
   * @param container Container from which value will be copied.
   */
  template <class InputContainerT>
  MOCHI_FORCE_INLINE void append(InputContainerT const& container) {
    append(std::begin(container), std::end(container));
  }

  /**
   * @brief Get a pointer to the polymorphic allocator.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE Allocator* get_allocator() const {
    return _allocator;
  }

  /**
   * @brief Get a reference to the value at index i.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE T& operator[](size_type i) {
    MOCHI_DARRAY_ASSERT(i < size(), "Index out of range");
    return _begin[i];
  }

  /**
   * @brief Get a reference to the value at index i.
   */
  [[nodiscard]] MOCHI_FORCE_INLINE T const& operator[](size_type i) const {
    MOCHI_DARRAY_ASSERT(i < size(), "Index out of range");
    return _begin[i];
  }

  /**
   * @brief Copy Assignment: Replace the contents of this DynamicArray by copying values from
   * another one.
   *
   * @param other Another DynamicArray from which values will be copied.
   * @return *this
   */
  DynamicArray& operator=(DynamicArray const& other) {
    if (&other != this)
      MOCHI_LIKELY {
        assign(other.begin(), other.end());
      }
    return *this;
  }

  /**
   * @brief Move Assignment: Replace the contents of this DynamicArray by moving memory or values
   * from another one.
   *
   * @remarks If the allocators are equal, then array memory ownership will be transferred from
   * the other array to this one. Otherwise, new memory will be allocated for this array (if
   * necessary) and then the other array's values will be moved into it. Either way, the other
   * array will be empty after this call.
   *
   * @param other Another DynamicArray from which memory or values will be moved.
   * @return *this
   */
  DynamicArray& operator=(DynamicArray&& other) {
    if (&other != this)
      MOCHI_LIKELY {
        clear();
        if (_allocator->is_equal(*other._allocator))
          MOCHI_LIKELY {
            // Release any previously allocated memory
            if (_begin != nullptr) {
              _allocator->deallocate(_begin, capacity() * sizeof(T), alignof(T));
            }
            // Move ownership of the other memory to this array
            _begin = other._begin;
            _end = other._end;
            _endCapacity = other._endCapacity;
            other._begin = other._end = other._endCapacity = nullptr;
          }
        else if (!other.empty()) {
          // Ensure sufficient capacity using our allocator
          auto const newSize = other.size();
          if (newSize > capacity()) {
            GrowCapacity(newSize);
          }
          // Move construct each element individually, advancing _end after each successful
          // construction. If T's move constructor throws, the destructor will only destroy elements
          // that were successfully constructed.
          for (auto src = other._begin; src < other._end; ++src, ++_end) {
            new (_end) T(std::move(*src)); // Move construct
          }
          // Destroy the elements in the rhs array, which have already been moved.
          // No need to release memory in the rhs array at this time.
          other.clear();
        }
      }
    return *this;
  }

  /**
   * @brief Copy Assignment: Replace the contents of this DynamicArray by copying values from
   * a std::initializer_list.
   *
   * @param list List of values to copy
   * @return *this
   */
  DynamicArray& operator=(std::initializer_list<T> const& list) {
    assign(std::begin(list), std::end(list));
    return *this;
  }

  /**
   * @brief Copy Assignment: Replace the contents of this DynamicArray by copying values from
   * another iterable container of compatible type.
   *
   * @tparam InputContainerT Another container type. Must support std::begin and std::end.
   * @param other Container from which values will be copied.
   * @return *this
   */
  template <
      typename InputContainerT,
      MOCHI_CONCEPT(
          (!std::is_same_v<std::decay_t<InputContainerT>, DynamicArray> &&
           sizeof(decltype(std::begin(std::declval<InputContainerT const&>()))) &&
           sizeof(decltype(std::end(std::declval<InputContainerT const&>())))))>
  DynamicArray& operator=(InputContainerT const& other) {
    assign(std::begin(other), std::end(other));
    return *this;
  }

  /**
   * @brief Comparison returns true if the other array has the same size and all values are equal.
   */
  bool operator==(DynamicArray const& other) const {
    auto s = size();
    if (other.size() == s) {
      for (size_type i = 0; i < s; ++i) {
        if (other._begin[i] != _begin[i]) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  /**
   * @brief Comparison returns false unless the other array has the same size and all values are
   * equal.
   */
  MOCHI_FORCE_INLINE bool operator!=(DynamicArray const& other) const {
    return !(*this == other);
  }

 private:
  void GrowCapacity(size_type newCapacity) {
    MOCHI_DARRAY_ASSERT(newCapacity > capacity(), "This method only increases capacity");
    if (empty()) {
      if (_begin) {
        // Free previous allocation first, in case of a FILO allocator.
        _allocator->deallocate(_begin, capacity() * sizeof(T), alignof(T));
      }
      _begin = _end = static_cast<T*>(_allocator->allocate(newCapacity * sizeof(T), alignof(T)));
      _endCapacity = _begin + newCapacity;
    } else {
      auto oldSize = size();
      auto* newBegin = static_cast<T*>(_allocator->allocate(newCapacity * sizeof(T), alignof(T)));
      for (auto i = static_cast<ptrdiff_t>(oldSize) - 1; i >= 0; --i) {
        new (&newBegin[i]) T(std::move(_begin[i])); // Move via placement new
        if constexpr (!std::is_trivially_destructible_v<T>) {
          _begin[i].~T(); // Destroy
        }
      }
      _allocator->deallocate(_begin, capacity() * sizeof(T), alignof(T));
      _begin = newBegin;
      _end = newBegin + oldSize;
      _endCapacity = newBegin + newCapacity;
    }
  }

  size_type GetNextCapacity(size_type minCapacity) const {
    // Jump to kGrowthPatternStartSize, then grow by 50% each time after that.
    return std::max(std::max(minCapacity, kGrowthPatternStartSize), capacity() * 3 / 2);
  }

  static void DebugFillWithNaN(
      [[maybe_unused]] void* rangeBegin,
      [[maybe_unused]] void const* rangeEnd) {
#if MOCHI_DARRAY_DEBUG
    // Fill as much of the range as we can with NaN values of type real.
    // This will help catch mistakes in case someone reads memory that they shouldn't.
    auto fillBeginAddr =
        Min(reinterpret_cast<size_type>(rangeEnd),
            RoundUp(reinterpret_cast<size_type>(rangeBegin), alignof(real)));
    auto fillEndAddr =
        Max(reinterpret_cast<size_type>(rangeBegin),
            RoundDown(reinterpret_cast<size_type>(rangeEnd), alignof(real)));
    auto* fillBegin = reinterpret_cast<std::byte*>(fillBeginAddr);
    auto* fillEnd = reinterpret_cast<std::byte*>(fillEndAddr);
    auto const nan = std::numeric_limits<real>::signaling_NaN();
    for (auto it = fillBegin; it < fillEnd; it += sizeof(nan)) {
      // Use memcpy instead of dereferencing a real* because of strict aliasing rules.
      memcpy(it, &nan, sizeof(nan));
    }
    // Because of alignment, the fill range might be smaller than the input range.
    // In that case, write zeros to the remaining bytes on either end.
    auto paddingSizeFront = fillBegin - reinterpret_cast<std::byte*>(rangeBegin);
    if (paddingSizeFront) {
      memset(rangeBegin, 0, paddingSizeFront);
    }
    auto paddingSizeBack = reinterpret_cast<std::byte const*>(rangeEnd) - fillEnd;
    if (paddingSizeBack) {
      memset(fillEnd, 0, paddingSizeBack);
    }
#endif // MOCHI_DARRAY_DEBUG
  }

  static void DefaultConstructRange(T* rangeBegin, T const* rangeEnd) {
    if constexpr (std::is_trivially_default_constructible_v<T>) {
      auto numBytes =
          reinterpret_cast<std::intptr_t>(rangeEnd) - reinterpret_cast<std::intptr_t>(rangeBegin);
      memset(rangeBegin, 0, numBytes); // Zero-initialize
    } else {
      for (auto* it = rangeBegin; it < rangeEnd; ++it) {
        new (it) T; // Default construct
      }
    }
  }

  static void CopyConstructRangeFromValue(T* rangeBegin, T const* rangeEnd, T const& value) {
    // TODO: Use mochi::Fill for cases where copy construction is equivalent to copy assignment.
    for (auto* it = rangeBegin; it < rangeEnd; ++it) {
      new (it) T(value); // Copy construct
    }
  }

  template <class InputIt>
  static void CopyConstructRange(T* dstBegin, InputIt srcBegin, InputIt srcEnd) {
    // TODO(C++20): Replace by std::iter_value_t<InputIt>.
    using SrcT = std::remove_cv_t<typename std::iterator_traits<InputIt>::value_type>;
    if constexpr (
        std::is_trivially_copyable_v<T> && std::is_copy_assignable_v<T> &&
        std::is_same_v<SrcT, T>) {
      std::copy(srcBegin, srcEnd, dstBegin); // Copy assign. Works for any iterator type.
    } else {
      auto src = srcBegin;
      auto dst = dstBegin;
      for (; src != srcEnd; ++src, ++dst) {
        new (dst) T(*src); // Copy construct
      }
    }
  }

  static void MoveConstructRange(T* dstBegin, T* srcBegin, size_type count) {
    if constexpr (std::is_trivially_move_constructible_v<T>) {
      if (count > 0)
        MOCHI_LIKELY {
          memcpy(dstBegin, srcBegin, count * sizeof(T));
        }
    } else {
      for (auto i = 0; i < count; ++i) {
        new (&dstBegin[i]) T(std::move(srcBegin[i])); // Move construct
      }
    }
  }

  static void Destroy(T* it) {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      it->~T();
    }
    DebugFillWithNaN(it, it + 1);
  }

  static void DestroyRange(T* rangeBegin, T const* rangeEnd) {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      if (rangeEnd > rangeBegin)
        MOCHI_LIKELY {
          // Destroy in reverse order in case of FILO allocator used within element
          auto* it = const_cast<T*>(rangeEnd);
          while (it != rangeBegin) {
            --it;
            it->~T();
          }
        }
    }
    DebugFillWithNaN(rangeBegin, rangeEnd);
  }

#if MOCHI_DARRAY_DEBUG
  template <class InputIt>
  bool IsInThisArray(InputIt rangeBegin, InputIt rangeEnd) const {
    if constexpr (std::is_pointer_v<std::remove_reference_t<InputIt>>) {
      // Return true if the input range overlaps the memory used by this DynamicArray
      return !(
          (reinterpret_cast<T const*>(rangeBegin)) >= _endCapacity ||
          (reinterpret_cast<T const*>(rangeEnd) <= _begin));
    } else {
      // If input iterators are used (not raw pointers) then we'll assume they come from somewhere
      // else since DynamicArray does not use an iterator class.
      return false;
    }
  }

  bool IsInThisArray(T const* ptr) const {
    return (ptr >= _begin) && (ptr < _end);
  }
#endif // MOCHI_DARRAY_DEBUG

  // When overallocating, start with at least this capacity.
  static constexpr size_type kGrowthPatternStartSize = 8;

  T* _begin = nullptr;
  T* _end = nullptr;
  T* _endCapacity = nullptr;
  Allocator* _allocator = GetDefaultAllocator();
};

/**
 Utility Functions
*/

// Specialization of Append found in container_utils.h
template <typename T, typename InBeginIterator, typename InEndIterator>
MOCHI_FORCE_INLINE void
Append(DynamicArray<T>& out, InBeginIterator const& inBegin, InEndIterator const& inEnd) {
  out.append(inBegin, inEnd);
}

// Specialization of Append found in container_utils.h
template <typename T, typename InContainer>
MOCHI_FORCE_INLINE void Append(DynamicArray<T>& out, InContainer const& inContainer) {
  out.append(std::begin(inContainer), std::end(inContainer));
}

// Specialization of Append found in container_utils.h
// For each input value, add valueToAdd and append to result to the output.
template <typename T, typename ContainerIn>
void AppendSum(DynamicArray<T>& out, ContainerIn const& in, T valueToAdd) {
  if (valueToAdd == T(0)) {
    Append(out, in);
  } else {
    auto outIndex = out.size();
    out.resize_noinit(outIndex + std::size(in));
    for (auto const& x : in) {
      out[outIndex++] = x + valueToAdd;
    }
  }
}

/**
 Type Traits
*/

namespace details {
template <class ContainerT>
struct IsDynamicArrayDef : public std::false_type {};
template <class T>
struct IsDynamicArrayDef<DynamicArray<T>> : public std::true_type {};
} // namespace details

// IsDynamicArray<ContainerT> is true iff ContainerT is a type of the form DynamicArray<T>
template <class ContainerT>
static constexpr bool kIsDynamicArray = // Note: nvcc thinks details without mochi:: is ambiguous.
    mochi::details::IsDynamicArrayDef<std::decay_t<ContainerT>>::value;

// Class Template Argument Deduction (CTAD) guides
template <
    typename InputIt,
    MOCHI_CONCEPT((std::is_base_of_v<
                   std::input_iterator_tag,
                   typename std::iterator_traits<InputIt>::iterator_category>))>
DynamicArray(InputIt, InputIt, Allocator* = GetDefaultAllocator())
    -> DynamicArray<typename std::iterator_traits<InputIt>::value_type>;

} // namespace mochi

// Reflection support
#if MOCHI_USE_REFLECTION
template <class T>
struct SReflectTypeTraits<mochi::DynamicArray<T>> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_array;
  static SReflect::ArrayTypeInfo const& GetTypeInfo() {
    using ArrT = mochi::DynamicArray<T>;
    static auto* s_typeInfo =
        SReflect::MakeDynamicArrayTypeInfo<SReflect::VectorTypeInfo<ArrT>, ArrT, T>(
            "mochi::DynamicArray", true);
    return *s_typeInfo;
  }
};
#endif // MOCHI_USE_REFLECTION
