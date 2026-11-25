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

#include "simple_reflection_console.h" // Reverse include for intellisense

#if SIMPLE_REFLECTION_ENABLE

#include <optional>
#include <vector>

namespace SReflect {

template <typename T>
bool ConsoleExecute(
    T* obj,
    std::string_view command,
    std::string* outString,
    std::string* outError) {
  return ConsoleExecute(
      static_cast<void*>(obj), GetFinalTypeInfo(*obj), command, outString, outError);
}

namespace detail {

/**
Execute a command from a path and arguments. See SReflect::ConsoleExecute for execution and output
behavior. pathStart skips a prefix already resolved to reach obj, ending just after '/' or at
path.size().
*/
bool ConsoleExecute(
    void* obj,
    TypeInfo const& typeInfo,
    std::string_view path,
    Span<std::string const> args,
    std::string* outString = nullptr,
    std::string* outError = nullptr,
    size_t pathStart = 0);

template <typename T>
bool ConsoleExecute(
    T* obj,
    std::string_view path,
    Span<std::string const> args,
    std::string* outString = nullptr,
    std::string* outError = nullptr,
    size_t pathStart = 0) {
  return ConsoleExecute(
      static_cast<void*>(obj), GetFinalTypeInfo(*obj), path, args, outString, outError, pathStart);
}

/**
A token from ConsoleTokenize and the part of the input it came from.
*/
struct ConsoleToken {
  std::string text; // With quotes removed and escapes resolved
  size_t offset = 0; // Byte offset of the token's first character in the input
  size_t length = 0; // Length in bytes of the token as typed, including quotes and escapes
};

/**
Split a command line into tokens:

- Whitespace separates tokens. Quotes group text and are removed. Text in single quotes is
  literal. Double quotes resolve backslash escapes. Backslashes outside quotes are literal.
- Tokens starting with '[' or '{' keep bracketed text intact, including nested groups,
  whitespace and quotes. Brackets inside strings in double quotes and mismatched closing brackets
  are literal. Text after the group stays in the token until whitespace.
- Unclosed quotes consume the remaining input. Unclosed groups get a closing double quote
  if needed, followed by missing closing brackets. Other JSON errors are left to the caller.

Each token reports the part of the input it came from. Only whitespace outside quotes and groups
ends a token, so the last token reaches the end of the input unless such whitespace follows it.
The reported length covers the original input, even when missing closing brackets are added to text.

Unsupported escapes inside double quotes set outError to a description and outErrorOffset to the
byte offset of the first offending backslash. On success, clear outError and set outErrorOffset to
zero.
*/
[[nodiscard]] std::vector<ConsoleToken>
ConsoleTokenize(std::string_view input, std::string& outError, size_t* outErrorOffset = nullptr);

/**
One step of a console path (see SplitConsolePath).
*/
struct ConsolePathStep {
  std::string name; // Decoded name: a quoted step's string value, otherwise the text as written
  size_t offset = 0; // Position of the step's first character in the path
  size_t length = 0; // Length of the step as written, including quotes
  bool isQuoted = false;
};

/**
Split a console path into steps at each '/'. Paths use the syntax of IssueReport paths: a step is a
field, property or method name, an array index, or a map key, and a step that starts with '"' is a
JSON string, so it can hold any name. An empty path gives no steps. Empty, "." and ".." steps are
returned as they are, so callers decide what they mean.

Return false if a quoted step is unterminated, is not a valid JSON string, or is followed by
anything but '/'. Then outError is set to an error message, outErrorOffset to the position of the
problem in the path, and outSteps is unspecified.
*/
[[nodiscard]] bool SplitConsolePath(
    std::string_view path,
    std::vector<ConsolePathStep>& outSteps,
    std::string* outError = nullptr,
    size_t* outErrorOffset = nullptr);

/**
Format an error in the original command text. offset is a byte offset into command. The command
and caret lines start with "  | "; control characters in the command are escaped.
*/
[[nodiscard]] std::string
FormatConsoleError(std::string_view message, std::string_view command, size_t offset);

/**
Format an error in a console command as three lines joined with '\n': the message, the command (the
path followed by the arguments, with tokens quoted and escaped when needed), and a '^' under the
problem. The command and caret lines start with "  | ". tokenIndex 0 is the path and i is
args[i - 1]. offset is the position of the problem in that token.
*/
[[nodiscard]] std::string FormatConsoleError(
    std::string_view message,
    std::string_view path,
    Span<std::string const> args,
    size_t tokenIndex,
    size_t offset);

enum class ConsoleCompletionTarget { PathStep, Argument };

/**
What the end of a console command completes (see FindConsoleCompletion).
*/
struct ConsoleCompletion {
  ConsoleCompletionTarget target = ConsoleCompletionTarget::PathStep;
  size_t offset = 0; // Position in the input where the step or argument being completed starts
  std::string path; // The path up to that step, or the path of the member taking the argument
  std::vector<std::string> args; // Arguments before the one being completed
};

/**
Find what the end of input completes. Token pathTokenIndex is the path, and the tokens before it
are ignored, so a caller can pass 1 for commands such as "ls <path>". The tokens after the path are
arguments. Return nullopt while a token before the path is typed, and while typing a path that has
quotes or escapes, because its steps cannot be located in the input.
*/
[[nodiscard]] std::optional<ConsoleCompletion> FindConsoleCompletion(
    std::string_view input,
    size_t pathTokenIndex = 0);

/**
Complete the step or argument described by completion, keeping the candidates that start with
typed, as ConsoleAutoComplete does. completion.path may end with '/' after a step. pathStart skips a
prefix already resolved to reach obj, as in ConsoleExecute. Return false on failure, retaining any
available completion data. Errors are reported only through outError, when provided, and are never
logged.
*/
bool ConsoleAutoComplete(
    void const* obj,
    TypeInfo const& typeInfo,
    ConsoleCompletion const& completion,
    std::string_view typed,
    ConsoleAutoCompleteParams const& params,
    ConsoleAutoCompleteResult& outResult,
    std::string* outError = nullptr,
    size_t pathStart = 0);

/**
Add directories named outside of reflection, such as built-in directories, to a result for a path
step. Names are quoted for the terminal, then the candidates are matched, ordered and limited as in
ConsoleAutoComplete. The default becomes the first candidate.
*/
void AddConsoleStepCandidates(
    ConsoleAutoCompleteResult& result,
    std::vector<std::string> const& names,
    std::string_view typed,
    ConsoleAutoCompleteParams const& params);

} // namespace detail

} // namespace SReflect

#endif // SIMPLE_REFLECTION_ENABLE
