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

#include <string_view>

namespace mochi {

/**************************************************************************************************
Tag literal based on compile-time string hashing. Used to identify different backends.
*/
using Tag = size_t;

inline constexpr Tag MakeTag(char const* str, size_t len) {
  // NOTE: The result of this operation may lead to integer overflow. We can safely ignore this
  // as unsigned integer arithmetic is guaranteed to be modulo 2n, so this is well-defined no
  // matter what.

  static_assert(sizeof(size_t) == 8, "Unexpected size for size_t");
  constexpr auto kBasis = static_cast<size_t>(1381);
  constexpr auto kPrime = static_cast<size_t>(2467);

  size_t result = kBasis;
  for (size_t i = 0; i < len; ++i) {
    result ^= str[i];
    result *= kPrime;
  }

  return result;
}

inline constexpr Tag MakeTag(std::string_view const& str) {
  return MakeTag(str.data(), str.length());
}

inline constexpr Tag operator""_tag(char const* str, size_t len) {
  return MakeTag(str, len);
}

} // namespace mochi
