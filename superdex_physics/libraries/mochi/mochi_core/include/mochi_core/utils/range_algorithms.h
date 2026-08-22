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

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/predicates.h>
#include <mochi_core/utils/range_by_iterators.h>

#if !MOCHI_LANGUAGE_CPP20
#error This file requires C++20. Do not include it in the public API.
#endif

#include <algorithm>
#include <concepts>
#include <functional>
#include <iterator>
#include <ranges>

namespace mochi {

template <std::ranges::bidirectional_range R>
[[nodiscard]] MOCHI_FORCE_INLINE auto ReverseRange(R&& range) {
  return RangeByIterators{std::reverse_iterator{range.end()}, std::reverse_iterator{range.begin()}};
}

template <typename Range, typename Pred>
MOCHI_FORCE_INLINE auto partition(Range&& range, Pred&& predicate) {
  return std::partition(range.begin(), range.end(), predicate);
}

template <typename Range, typename Pred>
[[nodiscard]] MOCHI_FORCE_INLINE auto find(Range&& range, Pred&& predicate) {
  return std::find_if(range.begin(), range.end(), predicate);
}

template <typename Range, typename Comp = std::less<>, typename Projection = IdentityProjection>
[[nodiscard]] auto max_element(Range&& range, Comp&& less = {}, Projection&& projection = {}) {
  auto maxIt = std::begin(range);
  auto endSentinel = std::end(range);
  for (auto it = maxIt; it != endSentinel; ++it) {
    if (less(projection(*maxIt), projection(*it))) {
      maxIt = it;
    }
  }
  return maxIt;
}

template <typename Range, typename Comp = std::less<>>
MOCHI_FORCE_INLINE void sort(Range&& range, Comp&& less = {}) {
  std::sort(range.begin(), range.end(), less);
}

} // namespace mochi
