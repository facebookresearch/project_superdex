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
#include <mochi_core/utils/debug.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mochi {

// Append values from an iterator range to a std::vector (or similar) container.
template <class OutContainer, typename InBeginIterator, typename InEndIterator>
MOCHI_FORCE_INLINE void
Append(OutContainer& out, InBeginIterator const& inBegin, InEndIterator const& inEnd) {
  out.insert(std::end(out), inBegin, inEnd);
}

// Append values from a Span (or similar) to a std::vector (or similar) container.
template <class OutContainer, typename InContainer>
MOCHI_FORCE_INLINE void Append(OutContainer& out, InContainer const& in) {
  Append(out, std::begin(in), std::end(in));
}

// For each input value, add valueToAdd and append to result to the output.
template <typename ContainerOut, typename ContainerIn, typename Scalar>
void AppendSum(ContainerOut& out, ContainerIn const& in, Scalar valueToAdd) {
  if (valueToAdd == Scalar(0)) {
    Append(out, in);
  } else {
    std::transform(in.begin(), in.end(), std::back_inserter(out), [valueToAdd](auto& v) {
      return v + valueToAdd;
    });
  }
}

// For each pair of input values, add them and append the result to the output. The two input
// containers must be the same type and length.
template <typename ContainerOut, typename ContainerIn>
void AppendSum(ContainerOut& out, ContainerIn const& in1, ContainerIn const& in2) {
  using NonConstScalar = std::decay_t<decltype(out[0])>;
  MOCHI_ASSERT_VERBOSE(in1.size() == in2.size(), "Input containers must be the same size.");
  std::transform(
      in1.begin(), in1.end(), in2.begin(), std::back_inserter(out), std::plus<NonConstScalar>());
}

// Erase index i from a dynamic array like std::vector (or similar) by swapping the last element
// into its place. This is fast, but it does NOT preserve the order of the later elements.
template <class ArrayT, class IndexT>
void EraseIndexUnordered(ArrayT& arr, IndexT i) {
  static_assert(std::is_integral_v<IndexT>, "Not an index");
  using SizeT = decltype(arr.size());
  auto idx = static_cast<SizeT>(i);
  MOCHI_ASSERT_VERBOSE(idx >= 0 && idx < arr.size(), "Index out-of-bounds");
  arr[idx] = std::move(arr[arr.size() - 1]); // Should be safe even for the last index
  arr.pop_back();
}

namespace details {
// IsAssociative<T> is true_type iff type T::key_type is defined
template <typename T, typename V = std::void_t<>>
struct IsAssociative : std::false_type {};
template <typename T>
struct IsAssociative<T, std::void_t<typename T::key_type>> : std::true_type {};
} // namespace details

// Return true if a value can be found within a container. Works for arrays like std::vector and
// Span. Also works for associative containers like std::unordered_map (in which case it finds a
// key).
template <class Container, class Value>
[[nodiscard]] bool Contains(Container const& container, Value const& value) {
  if constexpr (details::IsAssociative<Container>::value) {
    return container.find(value) != container.end();
  } else {
    auto itr = std::find(std::begin(container), std::end(container), value);
    return itr != std::end(container);
  }
}

// Hash functor for std::pair<First, Second>. Can be used in std library unordered containers.
// Example: using MySetOfPairs = std::unordered_set<std::pair<int, int>, PairHash<int, int>>;
template <typename First, typename Second>
struct PairHash {
  size_t operator()(std::pair<First, Second> const& pair) const {
    static_assert(
        sizeof(First) <= sizeof(uint32_t) && sizeof(Second) <= sizeof(uint32_t),
        "Currently only supports types that are <= 4 bytes and convertible to uint64_t");
    return std::hash<uint64_t>{}(
        static_cast<uint64_t>(pair.first) | (static_cast<uint64_t>(pair.second) << 32));
  }
};

// Randomly selects N elements from an array-like container without replacement, discarding the
// rest. Uses the Fisher-Yates (Knuth) shuffle algorithm to efficiently select elements in O(N)
// time.
//
// Parameters:
// - v: The container to be modified in place.
// - N: The number of elements to keep. If N >= v.size(), the container remains unchanged.
// - rng: A random number generator function that returns random integers. Should satisfy the
//   minimal requirements of returning unsigned integers with uniform distribution.
//
// Time complexity: O(N) - or O(1) if N >= v.size()
// Space complexity: O(1) - performed in-place
//
// Example:
//   std::vector<int> v = {0, 1, 2, 3, 4};
//   RandomSubset(v, 3, RandomGenerator(seed)); // v might become {2, 0, 4}
template <typename ArrayT, typename SizeT, typename RngT>
void RandomSubset(ArrayT& v, SizeT N, RngT&& rng) {
  static_assert(std::is_integral_v<SizeT>, "Invalid size type");
  MOCHI_ASSERT_VERBOSE(N >= 0, "Invalid size.");
  auto const N0 = v.size();
  if (N0 <= N) {
    return;
  }
  for (SizeT i = 0; i < N; ++i) {
    SizeT j = i + static_cast<SizeT>(rng() % (N0 - i));
    std::swap(v[i], v[j]);
  }
  v.resize(N);
}

} // namespace mochi
