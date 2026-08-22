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

#include <mochi_core/utils/string_utils.h>

#include <cctype>
#include <optional>
#include <utility>

using namespace mochi;

template <typename STR>
static std::string JoinImpl(Span<STR const> input, std::string_view separator) {
  if (input.empty()) {
    return {};
  } else {
    size_t len = (input.size() - 1) * separator.length();
    for (auto const& str : input) {
      len += str.length();
    }
    std::string output;
    output.reserve(len);
    output.append(input[0]);
    for (size_t i = 1; i < input.size(); ++i) {
      output.append(separator);
      output.append(input[i]);
    }
    return output;
  }
}

std::string mochi::Join(Span<std::string_view const> input, std::string_view separator) {
  return JoinImpl(input, separator);
}

std::string mochi::Join(Span<std::string const> input, std::string_view separator) {
  return JoinImpl(input, separator);
}

DynamicArray<std::string_view> mochi::Split(
    std::string_view input,
    std::string_view separators,
    bool keepEmptyTokens,
    Allocator* allocator) {
  DynamicArray<std::string_view> output(allocator);
  if (input.empty()) {
    return output;
  }

  size_t start = 0;
  size_t end = 0;

  while (end != std::string_view::npos) {
    end = input.find_first_of(separators, start);

    if (end != std::string_view::npos) {
      if ((end > start) || keepEmptyTokens) {
        output.emplace_back(input.substr(start, end - start));
      }
      start = end + 1;
    } else {
      // No more separators - add the remaining part
      if (start < input.size() || keepEmptyTokens) {
        output.emplace_back(input.substr(start));
      }
    }
  }

  return output;
}

// Translate the character following a backslash into its escape-sequence value. Returns nullopt if
// `c` is not a supported escape.
static std::optional<char> EscapeChar(char c) {
  switch (c) {
    case 'n':
      return '\n';
    case 't':
      return '\t';
    case 'r':
      return '\r';
    case 'a':
      return '\a';
    case 'b':
      return '\b';
    case 'f':
      return '\f';
    case 'v':
      return '\v';
    case '"':
      return '"';
    case '\\':
      return '\\';
    case '\'':
      return '\'';
    default:
      return std::nullopt;
  }
}

// Append the contents of a quoted section to token, resolving escapes. On entry, i points at the
// opening quote. Returns the index just past the matching closing quote (or the end of input).
// Set outError if an unsupported escape is encountered.
static size_t AppendQuoted(
    std::string_view input,
    size_t i,
    char quote,
    std::string& token,
    std::string& outError) {
  auto setError = [&](size_t slashIdx) {
    MOCHI_ASSERT_VERBOSE(input[slashIdx] == '\\');
    if (outError.empty()) {
      if (slashIdx + 1 < input.size()) {
        outError = Format(
            R"(Unsupported escape sequence: "\%c" (index %zu))", input[slashIdx + 1], slashIdx);
      } else {
        outError = Format(R"(Unsupported escape sequence: "\" (index %zu))", slashIdx);
      }
    }
  };
  size_t const n = input.size();
  ++i; // Skip the opening quote.
  while (i < n && input[i] != quote) {
    if (input[i] == '\\') {
      if (i + 1 < n) {
        auto const escaped = EscapeChar(input[i + 1]);
        if (escaped) {
          token.push_back(*escaped);
        } else {
          setError(i);
          token.push_back(input[i + 1]);
        }
        i += 2;
      } else {
        setError(i);
        ++i;
      }
    } else {
      token.push_back(input[i]);
      ++i;
    }
  }
  if (i < n) {
    ++i; // Skip the closing quote.
  }
  return i;
}

DynamicArray<std::string> mochi::ConsoleTokenize(std::string_view input, std::string& outError) {
  outError.clear();
  DynamicArray<std::string> tokens;
  std::string token;
  bool inToken = false;
  size_t const n = input.size();
  for (size_t i = 0; i < n;) {
    char const c = input[i];
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (inToken) {
        tokens.push_back(std::move(token));
        token.clear();
        inToken = false;
      }
      ++i;
    } else if (c == '\'' || c == '"') {
      // Quotes group whitespace and honor backslash escape sequences.
      inToken = true;
      i = AppendQuoted(input, i, c, token, outError);
    } else {
      // Everything else, including a backslash outside quotes, is literal.
      inToken = true;
      token.push_back(c);
      ++i;
    }
  }
  if (inToken) {
    tokens.push_back(std::move(token));
  }
  return tokens;
}

bool mochi::EndsWithCaseInsensitive(std::string_view str, std::string_view suffix) {
  if (str.size() < suffix.size()) {
    return false;
  }
  auto tail = str.substr(str.size() - suffix.size());
  for (size_t i = 0; i < suffix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(tail[i])) !=
        std::tolower(static_cast<unsigned char>(suffix[i]))) {
      return false;
    }
  }
  return true;
}

std::string_view mochi::Trim(std::string_view input) {
  std::string_view constexpr kWhitespace = " \t\n\r\f\v";
  auto const start = input.find_first_not_of(kWhitespace);
  if (start == std::string_view::npos) {
    return input.substr(0, 0);
  } else {
    auto const end = input.find_last_not_of(kWhitespace);
    return input.substr(start, end - start + 1);
  }
}
