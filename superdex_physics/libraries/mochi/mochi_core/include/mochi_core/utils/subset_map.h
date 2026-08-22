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
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace mochi {
/*
 *   A utility class used to map back and forth between a subset and a full set
 *   and perform different array manipulations. The indices of the full set are
 *   given by {0, 1, 2, ..., N - 1} while the indices of the subset are written
 *   in the comments below as {a_1, a_2, ..., a_k} \subset {0, 1, 2, ..., N - 1}.
 *   Note that the entries a_1, ..., a_k are stored in ascending order in an internal
 *   vector, which makes it possible to check if an item is inside the set via
 *   binary search.
 */
class SubsetMap {
 private:
  std::vector<int> subset;

  // The value of N
  int originalRangeCount = 0;

 public:
  // Extracts the objects corresponding to indices {a_1, a_2, ..., a_k} and
  // writes them to output.
  template <typename object_t>
  void Extract(Span<object_t const> input, Span<object_t> output) const;

  // Same as above except returns a vector
  template <typename object_t>
  std::vector<object_t> Extract(Span<object_t const> input) const;

  // Prepares an output array to have the correct size for extraction.
  template <typename object_t>
  void ResizeForExtract(std::vector<object_t>& output) const;

  /*
    Given an index i in the set {0, 1, 2, ..., N - 1}, return the corresponding
    index in the subset {a_1, a_2, ..., a_k} or nullopt if i is not a member of
    the subset.
  */
  std::optional<int> GetSubsetIndexFromFullIndex(int fullId) const;

  // Batch version of GetSubsetIndexFromFullIndex
  // Raises an error if any full indices are not part of the subset
  // Input and output iterators must be over type int.
  template <typename InputIt, typename OutputIt>
  void GetSubsetIndicesFromFullIndices(
      InputIt inBegin,
      InputIt inEnd,
      OutputIt outputIt,
      Error& error) const;

  /*
    Given an index i in the subset {a_1, a_2, ..., a_k} return a_i.
  */
  int GetFullIndexFromSubsetIndex(int remappedId) const;

  // Batch version of GetFullIndexFromSubsetIndex
  // Input and output iterators must be over type int.
  template <typename InputIt, typename OutputIt>
  void GetFullIndicesFromSubsetIndices(InputIt inBegin, InputIt inEnd, OutputIt outputIt) const;

  // Returns true if the subset contains the specified ID, or false otherwise
  inline bool Contains(int Id) const {
    return GetSubsetIndexFromFullIndex(Id).has_value();
  }

  inline int GetSubsetSize() const {
    return isize(subset);
  }

  inline int GetFullSetSize() const {
    return originalRangeCount;
  }

  /*
    Constructs a subset map from a potentially unsorted list with potential duplicates.
  */
  template <typename IterableT>
  static SubsetMap FromUnsortedList(IterableT const& subset, int N);
  template <typename IteratorIt>
  static SubsetMap FromUnsortedList(IteratorIt begin, IteratorIt end, int N);

  inline bool IsEmpty() const {
    return subset.empty();
  }

  SubsetMap() = default;

  // Constructs a SubsetMap directly from a vector.
  // The vector must be sorted and have only unique entries!
  // If you want to create a subset map from a potentially
  // unsorted or list with duplicates, use SubsetMap::FromUnsortedList
  // instead!
  inline SubsetMap(std::vector<int> subset_, int N)
      : subset(std::move(subset_)), originalRangeCount(N) {
    MOCHI_ASSERT(std::is_sorted(subset.begin(), subset.end()));
  }

  auto begin() {
    return subset.begin();
  }

  auto end() {
    return subset.end();
  }

  auto begin() const {
    return subset.begin();
  }

  auto end() const {
    return subset.end();
  }

  inline int operator[](int idx) const {
    return subset[idx];
  }

  inline std::vector<int> const& GetStorage() const {
    return subset;
  }

  SubsetMap Union(SubsetMap const& A) const;
  SubsetMap Intersection(SubsetMap const& A) const;
  SubsetMap Complement() const;

  // Selects every Nth element in the subset
  SubsetMap EveryNth(int N) const;

  // Create an empty subset from this subset
  SubsetMap EmptySubset() const {
    return {std::vector<int>{}, GetFullSetSize()};
  }
};

SubsetMap Union(SubsetMap const& A, SubsetMap const& B);
SubsetMap Intersection(SubsetMap const& A, SubsetMap const& B);

template <typename object_t, typename index_t>
void Extract(Span<object_t const> input, Span<object_t> output, Span<index_t const> indices) {
  MOCHI_ASSERT(output.size() == indices.size());

  for (int i = 0; i < output.size(); ++i) {
    output[i] = input[indices[i]];
  }
}

template <typename object_t, typename index_t>
std::vector<object_t> Extract(Span<object_t const> input, Span<index_t const> indices) {
  std::vector<object_t> result(indices.size());
  Extract(input, MakeSpan(result), indices);
  return result;
}

/*
    Implementation
*/
template <typename object_t>
void SubsetMap::Extract(Span<object_t const> input, Span<object_t> output) const {
  MOCHI_ASSERT(output.size() == subset.size());
  mochi::Extract<object_t, int>(input, output, MakeSpan(subset));
}

// Applies the subset
template <typename object_t>
std::vector<object_t> SubsetMap::Extract(Span<object_t const> input) const {
  std::vector<object_t> result;
  ResizeForExtract(result);
  Extract<object_t>(input, MakeSpan(result));
  return result;
}

template <typename object_t>
void SubsetMap::ResizeForExtract(std::vector<object_t>& output) const {
  output.resize(subset.size());
}

inline std::optional<int> SubsetMap::GetSubsetIndexFromFullIndex(int fullId) const {
  auto it = std::lower_bound(subset.begin(), subset.end(), fullId);
  if (it == subset.end() || *it != fullId) {
    return std::nullopt;
  } else {
    return static_cast<int>(it - subset.begin());
  }
}

inline int SubsetMap::GetFullIndexFromSubsetIndex(int remappedId) const {
  return subset[remappedId];
}

template <typename InputIt, typename OutputIt>
void SubsetMap::GetSubsetIndicesFromFullIndices(
    InputIt inBegin,
    InputIt inEnd,
    OutputIt outputIt,
    Error& error) const {
  std::transform(inBegin, inEnd, outputIt, [&](int i) {
    auto result = GetSubsetIndexFromFullIndex(i);
    MOCHI_ERROR_IF(!result, error, "Full index has no corresponding subset index.");
    return result.value_or(-1);
  });
}

template <typename InputIt, typename OutputIt>
void SubsetMap::GetFullIndicesFromSubsetIndices(InputIt inBegin, InputIt inEnd, OutputIt outputIt)
    const {
  std::transform(inBegin, inEnd, outputIt, [&](int i) { return GetFullIndexFromSubsetIndex(i); });
}

template <typename IterableT>
SubsetMap SubsetMap::FromUnsortedList(IterableT const& subset, int N) {
  return FromUnsortedList(subset.begin(), subset.end(), N);
}

template <typename IteratorIt>
SubsetMap SubsetMap::FromUnsortedList(IteratorIt begin, IteratorIt end, int N) {
  std::vector<int> subset;
  std::copy(begin, end, std::back_inserter(subset));

  std::sort(subset.begin(), subset.end());
  auto onePast = std::unique(subset.begin(), subset.end());
  subset.erase(onePast, subset.end());

  return {std::move(subset), N};
}

} // namespace mochi
