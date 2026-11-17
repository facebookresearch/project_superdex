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
Split a string into tokens the way a command-line shell would.

Tokens are separated by whitespace. If a token is wrapped in single or double quotes, then it may
contain whitespace and backslash escape sequences. Returns the tokens with quotes removed and
escapes resolved. outError is cleared on success, or set to an error message if an escape sequence
is not supported.
*/
[[nodiscard]] std::vector<std::string> ConsoleTokenize(
    std::string_view input,
    std::string& outError);

} // namespace SReflect

#endif // SIMPLE_REFLECTION_ENABLE
