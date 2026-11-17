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

#include <simple_reflection/simple_reflection.h>

#if SIMPLE_REFLECTION_ENABLE

#include <string>
#include <string_view>
#include <vector>

namespace SReflect {

/**
Split a command line into tokens:

- Whitespace separates tokens. Quotes group text and are removed. Text in single quotes is
  literal. Double quotes resolve backslash escapes. Backslashes outside quotes are literal.
- Tokens starting with '[' or '{' keep bracketed text intact, including nested groups,
  whitespace and quotes. Brackets inside strings in double quotes and mismatched closing brackets
  are literal. Text after the group stays in the token until whitespace.
- Unclosed quotes consume the remaining input. Unclosed groups get a closing double quote
  if needed, followed by missing closing brackets. Other JSON errors are left to the caller.

Clear outError on success. Unsupported escapes inside double quotes set outError.
*/
[[nodiscard]] std::vector<std::string> ConsoleTokenize(
    std::string_view input,
    std::string& outError);

} // namespace SReflect

#endif // SIMPLE_REFLECTION_ENABLE
