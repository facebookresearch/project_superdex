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

#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace mochi {
template <typename T, class LambdaT>
auto Transform(T const* input, LambdaT&& func) {
  if (input) {
    return std::make_optional(func(*input));
  } else {
    using result_t = std::decay_t<decltype(func(*input))>;
    return std::optional<result_t>{};
  }
}

template <typename T, class LambdaT>
auto AndThen(std::optional<T> const& input, LambdaT&& func) {
  if (input) {
    return func(*input);
  } else {
    using result_t = std::decay_t<decltype(func(*input))>;
    return result_t{};
  }
}

template <typename T1, typename T2, class LambdaT>
auto AndThen(std::optional<T1> const& input1, std::optional<T2> const& input2, LambdaT&& func) {
  if (input1 && input2) {
    return func(*input1, *input2);
  } else {
    using result_t = std::decay_t<decltype(func(*input1, *input2))>;
    return result_t{};
  }
}

template <typename T, class LambdaT>
auto AndThen(std::optional<T>&& input, LambdaT&& func) {
  if (input) {
    return func(std::move(*input));
  } else {
    using result_t = std::decay_t<decltype(func(std::move(*input)))>;
    return result_t{};
  }
}

template <typename T, class LambdaT>
auto AndThen(T const* input, LambdaT&& func) {
  if (input) {
    return func(*input);
  } else {
    using result_t = std::decay_t<decltype(func(*input))>;
    return result_t{};
  }
}

template <typename T, class LambdaT>
auto Transform(std::optional<T> const& input, LambdaT&& func) {
  if (input) {
    return std::make_optional(func(*input));
  } else {
    using result_t = std::decay_t<decltype(func(*input))>;
    return std::optional<result_t>{};
  }
}

template <typename T1, typename T2, class LambdaT>
auto Transform(std::optional<T1> const& input1, std::optional<T2> const& input2, LambdaT&& func) {
  if (input1 && input2) {
    return std::make_optional(func(*input1, *input2));
  } else {
    using result_t = std::decay_t<decltype(func(*input1, *input2))>;
    return std::optional<result_t>{};
  }
}

template <typename T, class LambdaT>
auto Transform(std::optional<T>&& input, LambdaT&& func) {
  if (input) {
    return std::make_optional(func(std::move(*input)));
  } else {
    using result_t = std::decay_t<decltype(func(std::move(*input)))>;
    return std::optional<result_t>{};
  }
}

} // namespace mochi
