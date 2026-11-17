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

#include <mochi_core/utils/span.h>
#include <simple_reflection/simple_reflection_console.h>

#include <cstddef>
#include <string>

namespace mochi::dbg {

/// @brief Terminal layout options. Widths count UTF-8 characters. Zero maximum means unlimited.
struct ConsoleFormatParams {
  size_t _minNameColumnWidth = 16;
  size_t _maxNameColumnWidth = 0;
  size_t _minTypeColumnWidth = 16;
  size_t _maxTypeColumnWidth = 0;
  size_t _maxLineWidth = 0;
  size_t _columnSpacing = 4;
  bool _sortAlphabetically = true;
  bool _showTypes = false;
  bool _includeNamespaces = false;
};

/// @brief Format @ref SReflect::ConsoleListEntry data as terminal rows.
///
/// @details Fields show name, optional type and value. Methods follow in a separate block,
/// with signatures appended to their names. Entries are sorted alphabetically by default.
/// Truncated values and lines end in an ellipsis. Lines have no trailing padding.
[[nodiscard]] std::string FormatConsoleList(
    Span<SReflect::ConsoleListEntry const> entries,
    ConsoleFormatParams const& params = {});

} // namespace mochi::dbg
