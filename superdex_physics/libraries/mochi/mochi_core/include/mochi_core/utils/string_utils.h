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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/span.h>

#include <string>
#include <string_view>

namespace mochi {

/**
 * @brief Return a formatted string using printf-style formatting.
 *
 * @param format Printf-style format string.
 * @param ... Variable argument list
 * @return std::string
 *
 * @code{.cpp}
 * auto str = Format("Hello %s", "World");
 * @endcode
 *
 * @note Implemented inline in <mochi_core/utils/log.h>. Also declared here for discoverability.
 */
[[nodiscard]] std::string Format(char const* format, ...);

/**
 * @brief Join an array of strings into a single string, with a separator string between each.
 *
 * @param input Input array of strings to join
 * @param separator String
 *
 * @code{.cpp}
 *    std::string input[] = {"Hello", "World"};
 *    std::string output = Join(input, " "); // output = "Hello World"
 * @endcode
 */
[[nodiscard]] std::string Join(Span<std::string_view const> input, std::string_view separator);
[[nodiscard]] std::string Join(Span<std::string const> input, std::string_view separator);

/**
 * @brief Type deduction helper for various array types that are convertible to Span
 */
template <class ArrayT>
[[nodiscard]] inline std::string Join(ArrayT const& input, std::string_view separator) {
  return Join(MakeConstSpan(input), separator);
}

/**
 * @brief Split a string into tokens using a given set of separator characters.
 *
 * @param input Input string to be split.
 * @param separators String of separator characters.
 * @param keepEmptyTokens If true, the output may include empty tokens.
 * @param allocator Allocator to use for the output tokens.
 * @return Array of tokens. Each token is string_view within the input string.
 *
 * @code{.cpp}
 *   auto tokens = Split("Hello World", " "); // returns {"Hello", "World"}
 * @endcode
 */
[[nodiscard]] DynamicArray<std::string_view> Split(
    std::string_view input,
    std::string_view separators,
    bool keepEmptyTokens = false,
    Allocator* allocator = GetDefaultAllocator());

/**
 * @brief Split a string into tokens the way a command-line shell would.
 *
 * @details Tokens are separated by whitespace. If a token is wrapped in single or double quotes,
 * then it may contain whitespace and backslash escape sequences.
 *
 * @param input Input string to tokenize.
 * @param[out] outError Empty on success. Error message if failure to resolve escape sequence.
 * @return Array of tokens with quotes removed and escapes resolved.
 */
[[nodiscard]] DynamicArray<std::string> ConsoleTokenize(
    std::string_view input,
    std::string& outError);

/**
 * @brief Trim leading and trailing whitespace characters from a string.
 *
 * @param input Input string to be trimmed.
 * @return Trimmed string (a view of part or all of the input string).
 *
 * @code{.cpp}
 * auto trimmed = Trim("\tHello World \n"); // returns "Hello World"
 * @endcode
 */
[[nodiscard]] std::string_view Trim(std::string_view input);

/**
 * @brief Check whether a string ends with the given suffix, case-insensitive.
 *
 * @param str Input string.
 * @param suffix Suffix to check for.
 * @return True if @ref str ends with @ref suffix (ignoring case).
 *
 * @code{.cpp}
 * EndsWithCaseInsensitive("Model.OBJ", ".obj") == true
 * @endcode
 */
[[nodiscard]] bool EndsWithCaseInsensitive(std::string_view str, std::string_view suffix);

} // namespace mochi
