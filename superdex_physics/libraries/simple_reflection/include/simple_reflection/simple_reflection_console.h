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

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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

// A Directory can be listed with ConsoleList. A Method has a callable signature.
enum class ConsoleEntryKind { Field, Directory, Method };

} // namespace SReflect

SR_BeginEnum(SReflect::ConsoleEntryKind);
SR_EnumItem(Field);
SR_EnumItem(Directory);
SR_EnumItem(Method);
SR_EndEnum();

namespace SReflect {

struct ConsoleListParams {
  size_t _maxValueLength = 1024; // Max bytes to capture for each entry's value
  bool _serializePrimitives = true; // Serialize the value of strings, bools, numbers and enums
  bool _serializePrimitiveArrays = true; // Serialize fixed-size arrays and matrices of primitives

  SR_BeginStruct(SReflect::ConsoleListParams);
  SR_Field_Name(_maxValueLength, "maxValueLength");
  SR_Field_Name(_serializePrimitives, "serializePrimitives");
  SR_Field_Name(_serializePrimitiveArrays, "serializePrimitiveArrays");
  SR_EndStruct()
};

/**
One member, array element, map value or method listed by ConsoleList.
*/
struct ConsoleListEntry {
  ConsoleEntryKind _kind = ConsoleEntryKind::Field;
  std::string _name; // Path step, quoted when needed
  std::string _typeName; // Type name or method signature, without namespaces
  std::string _typeNameWithNamespace;
  std::string _value; // Captured value
  std::string _valueError; // Error reading or serializing the entry, empty if none
  bool _isValueTruncated = false; // The captured value is incomplete. No ellipsis is appended.
  CoreType _coreType = CoreType::X_Invalid; // Unwrapped value's core type. Invalid for methods.
  std::optional<size_t> _containerSize = {}; // Array/matrix elements or map entries, if applicable.

  SR_BeginStruct(SReflect::ConsoleListEntry);
  SR_Field_Name(_kind, "kind");
  SR_Field_Name(_name, "name");
  SR_Field_Name(_typeName, "typeName");
  SR_Field_Name(_typeNameWithNamespace, "typeNameWithNamespace");
  SR_Field_Name(_value, "value");
  SR_Field_Name(_valueError, "valueError");
  SR_Field_Name(_isValueTruncated, "isValueTruncated");
  SR_Field_Name(_coreType, "coreType");
  SR_Field_Name(_containerSize, "containerSize");
  SR_EndStruct()
};

/**
Follow path as in ConsoleExecute, then list the entries that can be referenced there. obj must be a
class or struct described by typeInfo. pathStart skips a prefix already resolved by the caller.

See ConsoleListParams for options to customize what is captured and ConsoleListEntry for the
output. Nothing is written through obj.

Return an empty vector on failure and report the error as in ConsoleExecute. On success, clear
outError.
*/
[[nodiscard]] std::vector<ConsoleListEntry> ConsoleList(
    void const* obj,
    TypeInfo const& typeInfo,
    std::string_view path,
    ConsoleListParams const& params = {},
    std::string* outError = nullptr,
    size_t pathStart = 0);

// Uses GetFinalTypeInfo(*obj)
template <typename T>
[[nodiscard]] std::vector<ConsoleListEntry> ConsoleList(
    T const* obj,
    std::string_view path,
    ConsoleListParams const& params = {},
    std::string* outError = nullptr,
    size_t pathStart = 0) {
  return ConsoleList(
      static_cast<void const*>(obj), GetFinalTypeInfo(*obj), path, params, outError, pathStart);
}

/**
Options for ConsoleAutoComplete.
*/
struct ConsoleAutoCompleteParams {
  size_t _maxCandidates = 1000; // Candidates beyond this many are left out
  size_t _maxValueLength = 256; // Longer argument values are not offered
  bool _directoriesOnly = false; // Offer only path steps that ConsoleList can list

  SR_BeginStruct(SReflect::ConsoleAutoCompleteParams);
  SR_Field_Name(_maxCandidates, "maxCandidates");
  SR_Field_Name(_maxValueLength, "maxValueLength");
  SR_Field_Name(_directoriesOnly, "directoriesOnly");
  SR_EndStruct()
};

/**
Ways to complete the path step or argument at the end of a console command.
*/
struct ConsoleAutoCompleteResult {
  std::vector<std::string> _candidates; // Full text of the step or argument, as it would be typed
  size_t _defaultIndex = 0; // The candidate to offer first, if there are any
  std::string _note; // The value's type, or a method argument's type and name. Empty for steps.
  std::vector<std::string> _laterArgumentNotes; // Notes for the method's arguments after this one
  bool _isTruncated = false; // More candidates match than params._maxCandidates allowed

  SR_BeginStruct(SReflect::ConsoleAutoCompleteResult);
  SR_Field_Name(_candidates, "candidates");
  SR_Field_Name(_defaultIndex, "defaultIndex");
  SR_Field_Name(_note, "note");
  SR_Field_Name(_laterArgumentNotes, "laterArgumentNotes");
  SR_Field_Name(_isTruncated, "isTruncated");
  SR_EndStruct()
};

/**
Complete the path step or argument at the end of a console command, which has the form of a
ConsoleExecute command. Only candidates that start with the text typed so far are returned. Nothing
is written through obj.

Path steps:
  - The first token is a path, and only its last step is completed: a field, property or method
    name or a map key. Array indices do not complete.
  - The default is the first candidate.

Arguments:
  - A value takes one argument, which completes to its current value as the default. Nothing
    completes if the current value cannot be read or represented as text within
    params._maxValueLength, or when a set would fail because the value, a step on its path or one
    of its members is read-only, or the value cannot be copied.
  - A method argument completes to a default-constructed value of its type. Nothing completes for a
    method that cannot be called because an argument or its return type cannot be default
    constructed, or because it is non-const and a step on its path is read-only.
    An enum argument selects its zero item when present, otherwise the first alphabetical match.
  - Enums also complete to their item names, bools to true and false, and optionals to null.
    Argument candidates longer than params._maxValueLength are left out.
  - The note describes the argument, for example "Fruit" for a value or "Fruit fruit" for a method
    argument. A method argument also has notes for the arguments after it, in order. Nothing
    completes after a value's argument or a method's last argument.

Candidates are text that ConsoleTokenize reads back as the step or argument. A value is one line of
JSON with a space after each ',' and ':', except that a string or enum is its text, quoted if
needed. Candidates made only of digits come first in numeric order, then other candidates in
alphabetical order. At most params._maxCandidates are returned, always including the default.
A path step typed with quotes or escapes does not complete.

Return the available completion data. Errors are reported only through outError, when provided,
and are never logged. On success, clear outError.
*/
[[nodiscard]] ConsoleAutoCompleteResult ConsoleAutoComplete(
    void const* obj,
    TypeInfo const& typeInfo,
    std::string_view command,
    ConsoleAutoCompleteParams const& params = {},
    std::string* outError = nullptr);

// Uses GetFinalTypeInfo(*obj)
template <typename T>
[[nodiscard]] ConsoleAutoCompleteResult ConsoleAutoComplete(
    T const* obj,
    std::string_view command,
    ConsoleAutoCompleteParams const& params = {},
    std::string* outError = nullptr) {
  return ConsoleAutoComplete(
      static_cast<void const*>(obj), GetFinalTypeInfo(*obj), command, params, outError);
}

} // namespace SReflect

#include "simple_reflection_console_inl.h"

#endif // SIMPLE_REFLECTION_ENABLE
