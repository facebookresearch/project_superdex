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

#include <gtest/gtest.h>

#include <cctype>
#include <string>
#include <string_view>

namespace mochi::test {

inline std::string RemoveUnquotedWhitespace(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  char quote = '\0';
  bool escaped = false;
  for (char c : text) {
    if (quote != '\0') {
      result += c;
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == quote) {
        quote = '\0';
      }
    } else if (c == '"' || c == '\'') {
      quote = c;
      result += c;
    } else if (!std::isspace(static_cast<unsigned char>(c))) {
      result += c;
    }
  }
  return result;
}

inline void ExpectConsoleListEq(std::string_view expected, std::string_view actual) {
  EXPECT_EQ(RemoveUnquotedWhitespace(expected), RemoveUnquotedWhitespace(actual)) << actual;
}

} // namespace mochi::test
