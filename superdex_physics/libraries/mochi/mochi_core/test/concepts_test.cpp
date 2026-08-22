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

#include <mochi_core/utils/concepts.h>
#include <mochi_core/utils/half.h>

#include <type_traits>

using namespace mochi;

static_assert(std::is_same_v<ScalarType<float>, float>);
static_assert(std::is_same_v<ScalarType<double>, double>);
static_assert(std::is_same_v<ScalarType<int>, int>);
static_assert(std::is_same_v<ScalarType<int64_t>, int64_t>);
static_assert(std::is_same_v<ScalarType<bool>, bool>);
static_assert(std::is_same_v<ScalarType<Half>, Half>);

static_assert(std::is_same_v<ScalarType<float const>, float>);
static_assert(std::is_same_v<ScalarType<float&>, float>);
static_assert(std::is_same_v<ScalarType<float const&>, float>);
static_assert(std::is_same_v<ScalarType<float&&>, float>);
