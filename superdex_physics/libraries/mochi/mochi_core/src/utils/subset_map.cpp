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

#include <mochi_core/utils/subset_map.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

using namespace mochi;

SubsetMap mochi::Union(SubsetMap const& A, SubsetMap const& B) {
  MOCHI_ASSERT(A.GetFullSetSize() == B.GetFullSetSize());
  std::vector<int> result;
  result.reserve(A.GetSubsetSize() + B.GetSubsetSize());
  std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(result));
  return {std::move(result), A.GetFullSetSize()};
}

SubsetMap mochi::Intersection(SubsetMap const& A, SubsetMap const& B) {
  MOCHI_ASSERT(A.GetFullSetSize() == B.GetFullSetSize());

  auto const sizeGuess =
      int64_t{2} * A.GetSubsetSize() * B.GetSubsetSize() / Max(1, A.GetFullSetSize());
  std::vector<int> result;
  result.reserve(sizeGuess);
  std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(result));
  return {std::move(result), A.GetFullSetSize()};
}

SubsetMap SubsetMap::Complement() const {
  std::vector<int> result;
  result.reserve(GetFullSetSize() - GetSubsetSize());

  int current = 0;
  for (auto idx : *this) {
    // Add all indices before this one in A that were not in A
    for (; current < idx; ++current) {
      result.emplace_back(current);
    }
    ++current;
  }

  // Add all remaining indices to A
  for (; current < GetFullSetSize(); ++current) {
    result.emplace_back(current);
  }

  return {std::move(result), GetFullSetSize()};
}

SubsetMap SubsetMap::EveryNth(int N) const {
  std::vector<int> result;
  result.reserve(GetSubsetSize() / N + 1);
  for (int i = 0; i < GetSubsetSize(); i += N) {
    result.emplace_back(subset[i]);
  }
  return {std::move(result), GetFullSetSize()};
}

SubsetMap SubsetMap::Union(SubsetMap const& A) const {
  return ::Union(*this, A);
}

SubsetMap SubsetMap::Intersection(SubsetMap const& A) const {
  return ::Intersection(*this, A);
}
