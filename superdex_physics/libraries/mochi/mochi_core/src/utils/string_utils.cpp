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
