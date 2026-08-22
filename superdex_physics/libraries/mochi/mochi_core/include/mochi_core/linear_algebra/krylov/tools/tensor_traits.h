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

#include <mochi_core/linear_algebra/krylov/tools/custom_matrix_traits.h>

#include <type_traits>
#include <utility>

namespace mochi::krylov::customization {
//
// This 'customization' namespace is a space where the user
// could add its own customization points when using the Krylov library.
//

} // namespace mochi::krylov::customization

namespace mochi::krylov::details {

template <typename T, typename = void>
constexpr bool IsGetFactoryAvailable{};

template <typename T>
constexpr bool
    IsGetFactoryAvailable<T, std::void_t<decltype(customization::GetFactory(std::declval<T>()))>> =
        true;

} // namespace mochi::krylov::details

namespace mochi::krylov {
template <typename T, typename Y = decltype(customization::GetFactory(std::declval<T>()))>
struct MatFType {
  using type = Y;
};

template <typename Matrix>
using MatrixFactoryType = typename MatFType<std::decay_t<Matrix>>::type;

} // namespace mochi::krylov
