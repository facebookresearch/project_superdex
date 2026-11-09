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

#include <cstdint>
#include <iterator>
#include <type_traits>

namespace mochi {

// The splitmix64 finalizer, a bijective mix of the bits of x.
[[nodiscard]] constexpr uint64_t MixBits(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Hash of the integer or enum values in a range, independent of their order: the sum of their mixed
// bits.
template <typename Range>
[[nodiscard]] constexpr uint64_t HashUnordered(Range const& values) {
  using Value = std::decay_t<decltype(*std::begin(values))>;
  static_assert(
      std::is_integral_v<Value> || std::is_enum_v<Value>, "Values must be integers or enums.");
  // The splitmix64 increment, added before mixing so that 0 counts too, since MixBits(0) is 0.
  constexpr uint64_t kIncrement = 0x9e3779b97f4a7c15ull;
  uint64_t hash = 0;
  for (auto const value : values) {
    hash += MixBits(static_cast<uint64_t>(value) + kIncrement);
  }
  return hash;
}

} // namespace mochi
