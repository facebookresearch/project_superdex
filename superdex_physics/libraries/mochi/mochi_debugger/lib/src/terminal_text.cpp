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

#include <mochi_debugger/lib/terminal_text.h>

#include <mochi_core/mochi_platform.h>

#include <algorithm>
#include <cstdint>

#if !MOCHI_PLATFORM_WINDOWS
#include <locale.h>
#include <wchar.h>
#endif

namespace mochi::dbg {
namespace {

struct Character {
  uint32_t value;
  size_t bytes;
};

Character Decode(std::string_view text) {
  auto const first = static_cast<unsigned char>(text[0]);
  if (first < 0x80) {
    return {first, 1};
  }
  size_t const bytes = first >= 0xF0 && first <= 0xF4 ? 4 : first >= 0xE0 ? 3 : 2;
  if (first < 0xC2 || first > 0xF4 || text.size() < bytes) {
    return {0xFFFD, 1};
  }
  uint32_t value = first & (0x7F >> bytes);
  for (size_t i = 1; i < bytes; ++i) {
    auto const byte = static_cast<unsigned char>(text[i]);
    if ((byte & 0xC0) != 0x80) {
      return {0xFFFD, 1};
    }
    value = (value << 6) | (byte & 0x3F);
  }
  uint32_t const minimum = bytes == 2 ? 0x80 : bytes == 3 ? 0x800 : 0x10000;
  if (value < minimum || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
    return {0xFFFD, 1};
  }
  return {value, bytes};
}

size_t Cells(uint32_t c) {
  if (c < 0x80) {
    return c >= 0x20 && c != 0x7F ? 1 : 0;
  }
#if !MOCHI_PLATFORM_WINDOWS
  struct Utf8Locale {
    Utf8Locale(Utf8Locale const&) = delete;
    Utf8Locale& operator=(Utf8Locale const&) = delete;
    Utf8Locale(Utf8Locale&&) = delete;
    Utf8Locale& operator=(Utf8Locale&&) = delete;
    locale_t value = newlocale(LC_CTYPE_MASK, "C.UTF-8", nullptr);
    Utf8Locale() {
      if (!value) {
        value = newlocale(LC_CTYPE_MASK, "en_US.UTF-8", nullptr);
      }
    }
    ~Utf8Locale() {
      if (value) {
        freelocale(value);
      }
    }
  };
  static Utf8Locale const locale;
  if (locale.value) {
    auto const previous = uselocale(locale.value);
    int const width = wcwidth(static_cast<wchar_t>(c));
    uselocale(previous);
    if (width >= 0) {
      return static_cast<size_t>(width);
    }
  }
#endif
  if ((c >= 0x300 && c <= 0x36F) || (c >= 0x1AB0 && c <= 0x1AFF) || (c >= 0x1DC0 && c <= 0x1DFF) ||
      (c >= 0x200B && c <= 0x200F) || (c >= 0x20D0 && c <= 0x20FF) ||
      (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xFE20 && c <= 0xFE2F) || c == 0xFEFF) {
    return 0;
  }
  bool const wide = (c >= 0x1100 && c <= 0x115F) || c == 0x2329 || c == 0x232A ||
      (c >= 0x2E80 && c <= 0xA4CF && c != 0x303F) || (c >= 0xAC00 && c <= 0xD7A3) ||
      (c >= 0xF900 && c <= 0xFAFF) || (c >= 0xFE10 && c <= 0xFE19) ||
      (c >= 0xFE30 && c <= 0xFE6F) || (c >= 0xFF00 && c <= 0xFF60) ||
      (c >= 0xFFE0 && c <= 0xFFE6) || (c >= 0x1F300 && c <= 0x1FAFF) ||
      (c >= 0x20000 && c <= 0x3FFFD);
  return wide ? 2 : 1;
}

} // namespace

size_t CountTerminalCells(std::string_view text) {
  size_t width = 0;
  while (!text.empty()) {
    auto const character = Decode(text);
    width += Cells(character.value);
    text.remove_prefix(character.bytes);
  }
  return width;
}

size_t TerminalPrefixBytes(std::string_view text, size_t cells) {
  size_t offset = 0;
  while (offset < text.size()) {
    auto const character = Decode(text.substr(offset));
    size_t const width = Cells(character.value);
    if (width > cells) {
      break;
    }
    cells -= width;
    offset += character.bytes;
  }
  return offset;
}

TerminalTextView MakeTerminalTextView(std::string_view text, size_t cursor, size_t columns) {
  cursor = std::min(cursor, text.size());
  size_t remaining = CountTerminalCells(text.substr(0, cursor));
  size_t start = 0;
  size_t end = text.size();
  if (columns != 0) {
    size_t const width = std::max(size_t{1}, columns - 1);
    while (remaining >= width && start < cursor) {
      auto const character = Decode(text.substr(start));
      remaining -= Cells(character.value);
      start += character.bytes;
    }
    end = start + TerminalPrefixBytes(text.substr(start), width);
  }
  return {text.substr(start, end - start), remaining};
}

std::string FitSuggestion(std::string_view preview, std::string_view annotation, size_t cells) {
  std::string text{preview};
  if (!annotation.empty()) {
    text += ' ';
    text += annotation;
  }
  if (CountTerminalCells(text) <= cells) {
    return text;
  }
  if (CountTerminalCells(preview) <= cells) {
    return std::string{preview};
  }
  static constexpr std::string_view kEllipsis = "...";
  if (cells <= kEllipsis.size()) {
    return {};
  }
  return std::string{preview.substr(0, TerminalPrefixBytes(preview, cells - kEllipsis.size()))} +
      std::string{kEllipsis};
}

} // namespace mochi::dbg
