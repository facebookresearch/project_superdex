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

#include <filesystem>
#include <optional>
#include <string_view>

namespace superdex::robotics {

/* @brief @ref ResolveBotPath for a string that may not be a bot path at all: returns std::nullopt
 * instead of logging and setting an error when @p input is not a valid reference from
 * @p basePath. For callers probing arbitrary strings, such as the type-agnostic walk over sensor
 * params that archiving does. A malformed .superdex_root is still logged, since that is broken
 * regardless of the input. */
[[nodiscard]] std::optional<std::filesystem::path> TryResolveBotPath(
    std::string_view input,
    std::filesystem::path const& basePath,
    int maxParentDepth);

} // namespace superdex::robotics
