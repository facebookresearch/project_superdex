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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/range_by_iterators.h>

#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>

namespace mochi {

/**************************************************************************************************
  @brief Integral half-open interval.
  @details The interval is half-open [begin, end). The state variables _begin and _end are not
  directly exposed. Note that _begin can be larger than _end for use in generic code where it
  allows to represent a fully non-constraining null interval. The Null interval is such an interval.
  It is non constraining because Union( Null(), Interval(a, b) ) with a <= b will return
  Interval(a,b). This is contrast to Union( Interval(0,0), Interval(3, 5) ) which returns
  Interval(0,5) despite the first argument being an empty interval.

  A Interval where _begin > _end is valid and represents an empty interval with various possibility
  of constraining the Union operation.
*/
template <typename T>
struct Interval final {
  static_assert(std::is_integral<T>::value, "Interval only accepts integral types");

 public:
  using value_type = T;

 public:
  constexpr Interval() = default;
  constexpr Interval(int count) : _begin(0), _end(count) {}
  constexpr Interval(T begin, T end) : _begin(begin), _end(end) {}

  /// @brief Check if the interval is non empty, or empty but constraining the Union operation.
  [[nodiscard]] MOCHI_FORCE_INLINE constexpr bool Valid() const {
    return _begin <= _end;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE constexpr T Size() const {
    return mochi::Max(T(0), _end - _begin);
  }

  /// @brief For compatibility with STL algorithms.
  [[nodiscard]] MOCHI_FORCE_INLINE constexpr auto size() const {
    return static_cast<size_t>(Size());
  }

  [[nodiscard]] MOCHI_FORCE_INLINE constexpr T Min() const {
    return _begin;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE constexpr T& Min() {
    return _begin;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE constexpr T Max() const {
    return _end - 1;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE constexpr bool Within(value_type x) const {
    return x >= _begin && x < _end;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE constexpr bool Overlaps(Interval const& other) const {
    return _begin < other._end && other._begin < _end;
  }

  /// @brief Form an Interval covering the union of two intervals.
  /// @note It is not the mathematical union of the intervals as it may include values
  /// that are neither in *this nor in other.
  [[nodiscard]] MOCHI_FORCE_INLINE constexpr Interval Union(Interval const& other) const {
    return {mochi::Min(_begin, other._begin), mochi::Max(_end, other._end)};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE constexpr Interval Intersect(Interval const& other) const {
    return {mochi::Max(_begin, other._begin), mochi::Min(_end, other._end)};
  }

  /// @brief Returns an interval that starts at the beginning of the left interval and ends at
  /// the end of the right one.
  /// @note It can result in the Union the Intersection or a hybrid
  /// of both of those if left._begin > right._begin and/or right._end > left._end.
  [[nodiscard]] MOCHI_FORCE_INLINE friend constexpr Interval Straddle(
      Interval const& left,
      Interval const& right) {
    return Interval{left._begin, right._end};
  }

  template <typename It>
  [[nodiscard]] MOCHI_FORCE_INLINE friend constexpr auto operator+(
      It iterator,
      Interval const& interval) {
    auto theEnd = interval._begin > interval._end ? interval._begin : interval._end;
    return RangeByIterators<It>{iterator + interval._begin, iterator + theEnd};
  }

  /** @brief Take a slice of a general indexable range object. */
  template <typename Rng>
  [[nodiscard]] MOCHI_FORCE_INLINE friend constexpr auto Slice(
      Rng& rng,
      Interval const& integerInterval) {
    MOCHI_ASSERT_VERBOSE(rng.begin() + integerInterval._end <= rng.end());
    return rng.begin() + integerInterval;
  }

  struct Iterator {
    /// Note, for C++20, thanks to the standard library use of concepts,
    /// all the "using..." lines below can be removed
    using iterator_category = std::forward_iterator_tag;
    using difference_type = T;
    using value_type = T;
    using pointer = T const*;
    using reference = T const&;
    [[nodiscard]] MOCHI_FORCE_INLINE T operator*() const {
      return n;
    }
    MOCHI_FORCE_INLINE Iterator& operator++() {
      ++n;
      return *this;
    }

    MOCHI_FORCE_INLINE Iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Iterator const& b) const {
      return n == b.n;
    }
    // Not necessary in C++20
    [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Iterator const& b) const {
      return n != b.n;
    }
    [[nodiscard]] MOCHI_FORCE_INLINE T operator-(Iterator const& b) const {
      return n - b.n;
    }
    T n;
  };

  [[nodiscard]] MOCHI_FORCE_INLINE Iterator begin() const {
    return Iterator{_begin};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Iterator end() const {
    return Iterator{_begin > _end ? _begin : _end};
  }

  /// @brief Output the range to a container.
  /// @note Using lowercase `to` to be able to use the standard range library when switching to
  /// C++20/C++23.
  template <typename Container>
  [[nodiscard]] MOCHI_FORCE_INLINE auto to() const {
    return Container{begin(), end()};
  }

 private:
  T _begin = T{0};
  T _end = T{0};
};

} // namespace mochi
