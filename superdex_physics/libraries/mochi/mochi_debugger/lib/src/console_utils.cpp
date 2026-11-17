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

#include <mochi_debugger/lib/console_utils.h>

#include <mochi_core/utils/dynamic_array.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <string_view>
#include <utility>

namespace mochi::dbg {
namespace {

size_t CountCharacters(std::string_view text) {
  return static_cast<size_t>(std::count_if(text.begin(), text.end(), [](char c) {
    return (static_cast<unsigned char>(c) & 0xC0) != 0x80;
  }));
}

std::string Truncate(std::string text, size_t width) {
  if (width == 0 || CountCharacters(text) <= width) {
    return text;
  }
  size_t const dots = std::min(size_t{3}, width);
  size_t end = 0;
  for (size_t i = 0; i < width - dots; ++i) {
    ++end;
    while (end < text.size() && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) {
      ++end;
    }
  }
  text.resize(end);
  text.append(dots, '.');
  return text;
}

std::string Beautify(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  bool quoted = false;
  bool escaped = false;
  for (size_t i = 0; i < text.size(); ++i) {
    char const c = text[i];
    result += c;
    if (escaped) {
      escaped = false;
    } else if (quoted && c == '\\') {
      escaped = true;
    } else if (c == '"') {
      quoted = !quoted;
    } else if (
        !quoted && i + 1 < text.size() &&
        (c == ',' || (c == ':' && text[i + 1] != ':' && (i == 0 || text[i - 1] != ':'))) &&
        !std::isspace(static_cast<unsigned char>(text[i + 1]))) {
      result += ' ';
    }
  }
  return result;
}

} // namespace

std::string FormatConsoleList(
    Span<SReflect::ConsoleListEntry const> entries,
    ConsoleFormatParams const& params) {
  DynamicArray<SReflect::ConsoleListEntry const*> ordered;
  ordered.reserve(entries.size());
  for (auto const& entry : entries) {
    ordered.push_back(&entry);
  }
  if (params._sortAlphabetically) {
    std::stable_sort(ordered.begin(), ordered.end(), [](auto const* a, auto const* b) {
      return a->_name < b->_name;
    });
  }
  DynamicArray<std::array<std::string, 3>> fields;
  DynamicArray<std::string> methods;
  std::array<size_t, 2> widths{};
  size_t const numColumns = params._showTypes ? 3 : 2;
  for (auto const* entry : ordered) {
    std::string const type = Beautify(
        params._includeNamespaces && !entry->_typeNameWithNamespace.empty()
            ? entry->_typeNameWithNamespace
            : entry->_typeName);
    if (entry->_kind == SReflect::ConsoleEntryKind::Method) {
      methods.push_back(Truncate(entry->_name + type, params._maxNameColumnWidth));
      continue;
    }
    std::string value = Beautify(entry->_value);
    if (!entry->_valueError.empty()) {
      value = "<error: " + entry->_valueError + ">";
      std::replace_if(
          value.begin(),
          value.end(),
          [](char c) { return static_cast<unsigned char>(c) < 0x20 || c == 0x7F; },
          ' ');
    } else if (entry->_isValueTruncated) {
      value += "...";
    }
    std::array<std::string, 3> row{
        Truncate(entry->_name, params._maxNameColumnWidth),
        params._showTypes ? Truncate(type, params._maxTypeColumnWidth) : std::move(value),
        params._showTypes ? std::move(value) : std::string{}};
    for (size_t i = 0; i + 1 < numColumns; ++i) {
      widths[i] = std::max(widths[i], CountCharacters(row[i]));
    }
    fields.push_back(std::move(row));
  }
  std::array<size_t, 2> const minWidths{params._minNameColumnWidth, params._minTypeColumnWidth};
  std::array<size_t, 2> const maxWidths{params._maxNameColumnWidth, params._maxTypeColumnWidth};
  for (size_t i = 0; i < widths.size(); ++i) {
    widths[i] = std::max(widths[i], minWidths[i]);
    if (maxWidths[i] != 0) {
      widths[i] = std::min(widths[i], maxWidths[i]);
    }
  }
  std::string result;
  auto const appendLine = [&](std::string line) {
    if (!result.empty()) {
      result += '\n';
    }
    result += Truncate(std::move(line), params._maxLineWidth);
  };
  for (auto const& row : fields) {
    size_t lastColumn = numColumns - 1;
    while (lastColumn > 0 && row[lastColumn].empty()) {
      --lastColumn;
    }
    std::string line = row[0];
    for (size_t i = 1; i <= lastColumn; ++i) {
      line.append(widths[i - 1] - CountCharacters(row[i - 1]) + params._columnSpacing, ' ');
      line += row[i];
    }
    appendLine(std::move(line));
  }
  if (!fields.empty() && !methods.empty()) {
    result += '\n';
  }
  for (auto& method : methods) {
    appendLine(std::move(method));
  }
  return result;
}

} // namespace mochi::dbg
