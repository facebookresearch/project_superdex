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
#include <string>
#include <string_view>

namespace mochi::dbg {

/// @brief Number of terminal display cells occupied by UTF-8 text.
size_t CountTerminalCells(std::string_view text);

/// @brief Byte offset after the longest UTF-8 prefix that fits the given cell width.
size_t TerminalPrefixBytes(std::string_view text, size_t cells);

struct TerminalTextView {
  std::string_view text;
  size_t cursorColumn{};
};

/// @brief Keep a byte-positioned cursor visible without splitting UTF-8 characters or wrapping.
TerminalTextView MakeTerminalTextView(std::string_view text, size_t cursor, size_t columns);

/// @brief Fit a completion preview and its annotation, separated by a space, into the given cells.
/// The annotation is dropped first, then the preview is cut to end in "...".
std::string FitSuggestion(std::string_view preview, std::string_view annotation, size_t cells);

} // namespace mochi::dbg
