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

#include <simple_reflection/simple_reflection_console.h>

#if SIMPLE_REFLECTION_ENABLE

#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SReflect {

// Translate the character following a backslash into its escape-sequence value. Returns nullopt if
// c is not a supported escape.
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

// Append the contents of a quoted section to token, resolving escapes inside double quotes. On
// entry, i points at the opening quote. Returns the index just past the matching closing quote (or
// the end of input). Set outError if an unsupported escape is encountered.
static size_t AppendQuoted(
    std::string_view input,
    size_t i,
    char quote,
    std::string& token,
    std::string& outError) {
  auto setError = [&](size_t slashIdx) {
    assert(input[slashIdx] == '\\');
    if (outError.empty()) {
      if (slashIdx + 1 < input.size()) {
        outError = std::string{"Unsupported escape sequence: \"\\"} + input[slashIdx + 1] + '"';
      } else {
        outError = "Backslash at the end of the input";
      }
      outError += " (index " + std::to_string(slashIdx) + ")";
    }
  };
  size_t const n = input.size();
  ++i;
  while (i < n && input[i] != quote) {
    if (quote == '"' && input[i] == '\\') {
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
    ++i;
  }
  return i;
}

// Append a JSON array or object to token as-is. On entry, i points at the opening bracket. Returns
// the index just past the matching closing bracket. If the input ends first, append a closing quote
// for an open string and the missing closing brackets, and return the end of input.
static size_t AppendBracketGroup(std::string_view input, size_t i, std::string& token) {
  assert(input[i] == '[' || input[i] == '{');
  std::string closers;
  bool inString = false;
  size_t const n = input.size();
  while (i < n) {
    char const c = input[i++];
    token.push_back(c);
    if (inString) {
      if (c == '\\' && i < n) {
        token.push_back(input[i++]);
      } else if (c == '"') {
        inString = false;
      }
    } else if (c == '"') {
      inString = true;
    } else if (c == '[') {
      closers.push_back(']');
    } else if (c == '{') {
      closers.push_back('}');
    } else if (c == closers.back()) {
      closers.pop_back();
      if (closers.empty()) {
        return i;
      }
    }
  }
  if (inString) {
    token.push_back('"');
  }
  token.append(closers.rbegin(), closers.rend());
  return i;
}

// The whitespace characters of the "C" locale, independent of the global locale.
static bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

std::vector<std::string> ConsoleTokenize(std::string_view input, std::string& outError) {
  outError.clear();
  std::vector<std::string> tokens;
  std::string token;
  bool inToken = false;
  size_t const n = input.size();
  for (size_t i = 0; i < n;) {
    char const c = input[i];
    if (IsSpace(c)) {
      if (inToken) {
        tokens.push_back(std::move(token));
        token.clear();
        inToken = false;
      }
      ++i;
    } else if (c == '\'' || c == '"') {
      // Quotes group whitespace. Only double quotes honor backslash escape sequences.
      inToken = true;
      i = AppendQuoted(input, i, c, token, outError);
    } else if (!inToken && (c == '[' || c == '{')) {
      inToken = true;
      i = AppendBracketGroup(input, i, token);
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

} // namespace SReflect

#endif // SIMPLE_REFLECTION_ENABLE
