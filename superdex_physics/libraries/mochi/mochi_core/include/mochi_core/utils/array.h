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

#include <cstddef>

#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>

namespace mochi {

template <typename T, size_t N>
struct Array {
  static_assert(N > 0, "Array size must be positive");
  using value_type = T;
  using size_type = size_t;

  // clang-format off
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T& operator[](size_type i) { return v[i]; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T const& operator[](size_type i) const { return v[i]; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T& front() { return v[0]; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T const& front() const { return v[0]; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T& back() { return v[N - 1]; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T const& back() const { return v[N - 1]; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T* data() { return v; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T const* data() const { return v; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T* begin() { return v; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T const* begin() const { return v; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T* end() { return v + N; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T const* end() const { return v + N; }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr static size_type size() { return N; }
  T v[N];
  // clang-format on
};

} // namespace mochi
