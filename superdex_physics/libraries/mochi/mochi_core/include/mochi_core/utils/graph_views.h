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

#include <mochi_core/utils/graph.h>

#include <type_traits>

namespace mochi {

namespace details {
template <typename T>
using SpanCI = Span<T, int const>;
} // namespace details

template <typename Mat>
auto AsGraphView(Mat const& A) {
  using namespace details;
  auto const& idx = A.Indices();
  using CRIdx = std::decay_t<decltype(idx[0])>;
  auto const& ptr = A.Pointers();
  using Ptr = std::decay_t<decltype(ptr[0])>;
  SpanCI<Ptr const> ptr_s(ptr);
  return Graph<CRIdx const, Ptr const, details::SpanCI>(ptr_s, idx);
}

} // namespace mochi
