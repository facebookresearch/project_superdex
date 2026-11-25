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
#include <string_view>

namespace mochi::dbg {

/// @brief Terminal layout options. Widths count display cells. Zero maximum means unlimited.
struct ConsoleFormatParams {
  size_t _minNameColumnWidth = 32;
  size_t _maxNameColumnWidth = 0;
  size_t _minTypeColumnWidth = 32;
  size_t _maxTypeColumnWidth = 0;
  size_t _maxLineWidth = 0; ///< Shorten property values to fit this line width.
  size_t _columnSpacing = 4;
  bool _sortAlphabetically = true;
  bool _showTypes = false;
  bool _includeNamespaces = false;
};

/// @brief Format a type name or method signature as listings show it: ReflectionData suffixes
/// are dropped, and a space follows each comma.
[[nodiscard]] std::string FormatConsoleTypeName(std::string_view text);

/// @brief Format @ref SReflect::ConsoleListEntry data as terminal rows.
///
/// @details Methods precede properties with a blank line between the blocks.
/// Fields show name, optional type and value. Methods show names in the first column and signatures
/// in the second. Entries are sorted alphabetically by default.
/// Getter errors leave values blank. Otherwise, missing values show container size when available.
/// Type names drop ReflectionData suffixes when a name remains. Argument names are unchanged.
/// Truncated values end in an ellipsis. Lines have no trailing padding.
[[nodiscard]] std::string FormatConsoleList(
    Span<SReflect::ConsoleListEntry const> entries,
    ConsoleFormatParams const& params = {});

} // namespace mochi::dbg
