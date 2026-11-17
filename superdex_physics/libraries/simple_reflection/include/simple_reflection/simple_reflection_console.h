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

namespace SReflect {

/**
Execute a console command on a reflected object. Commands have the form: <path> [args...]
The path names a member, using '/' to reach nested values. Arguments supply a value to set or
method parameters. Whitespace separates tokens. Quotes group text, and JSON arrays and objects
stay together.

Examples:
  count                           Read the "count" field
  parts/2/mass                    Read the "mass" field of index 2 of the "parts" array.
  parts/2/mass 3.5                Set that field
  parts/2 {"label": "wheel"}      Update a struct, keeping other members
  parts/2/add 1.5 "front wheel"   Call a method with two arguments
  parts/2/clear                   Call a method with no arguments

Path resolution:
  - obj must be a class or struct described by typeInfo.
  - Paths follow detail::SplitConsolePath syntax, traversing optionals and variants through their
    contained values.
  - Methods must be last.

Argument resolution:
  - A value takes no arguments to read or one to set.
  - A method takes one argument per parameter.
  - Strings and enums take literal argument text.
  - Other arguments must be one complete JSON value.
  - If any part of the path is read-only, then the command is read-only.
  - If the path contains a property with a getter/setter pair, then it executes on a temporary
    variable. It then calls the setter if the command was a set or non-const call.
  - If a setter or method throws an exception, ConsoleExecute returns an error string. Partial
    modification is possible in this case, depending on the method internals.

Outputs:
  - Returns true on success. Empty or whitespace-only commands fail.
  - On failure, outError receives the error description if non-null. Else, the error is logged.
  - If non-null, outString receives the value read, the value read back after a set, or a method's
    return value as pretty JSON. Empty for void methods.
  - Readback is skipped if outString is null. Otherwise it happens after enclosing setters run.
    A readback failure reports an error without undoing the set.
*/
bool ConsoleExecute(
    void* obj,
    TypeInfo const& typeInfo,
    std::string_view command,
    std::string* outString = nullptr,
    std::string* outError = nullptr);

// Uses GetFinalTypeInfo(*obj)
template <typename T>
bool ConsoleExecute(
    T* obj,
    std::string_view command,
    std::string* outString = nullptr,
    std::string* outError = nullptr);

} // namespace SReflect

#include "simple_reflection_console_inl.h"

#endif // SIMPLE_REFLECTION_ENABLE
