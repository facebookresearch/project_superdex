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

#include "simple_reflection_utils.h"

SR_WARNING_PUSH()
SR_WARNING_IGNORE_MSVC(4459) // declaration of 'last' hides global declaration
SR_WARNING_IGNORE_MSVC(4706) // assignment within conditional expression
#include <picojson/picojson.h> // picojson (third-party)
SR_WARNING_POP()

#include <algorithm>
#include <cassert>
#include <charconv>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace SReflect {

using namespace detail;

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
    std::string& outError,
    size_t* outErrorOffset) {
  auto setError = [&](size_t slashIdx) {
    assert(input[slashIdx] == '\\');
    if (outError.empty()) {
      outError = slashIdx + 1 < input.size() ? "Unsupported escape sequence"
                                             : "Incomplete escape sequence";
      if (outErrorOffset != nullptr) {
        *outErrorOffset = slashIdx;
      }
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

std::vector<std::string>
detail::ConsoleTokenize(std::string_view input, std::string& outError, size_t* outErrorOffset) {
  outError.clear();
  if (outErrorOffset != nullptr) {
    *outErrorOffset = 0;
  }
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
      i = AppendQuoted(input, i, c, token, outError, outErrorOffset);
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

/////////////////////////////////////////////////////////////////////////////
// Paths, quoting and errors
/////////////////////////////////////////////////////////////////////////////

bool detail::SplitConsolePath(
    std::string_view path,
    std::vector<ConsolePathStep>& outSteps,
    std::string* outError,
    size_t* outErrorOffset) {
  auto const fail = [&](std::string_view message, size_t offset) {
    if (outError != nullptr) {
      *outError = message;
    }
    if (outErrorOffset != nullptr) {
      *outErrorOffset = offset;
    }
    return false;
  };
  outSteps.clear();
  if (path.empty()) {
    return true;
  }
  for (size_t start = 0;;) {
    ConsolePathStep step;
    step.offset = start;
    size_t end = start;
    if (start < path.size() && path[start] == '"') {
      ++end;
      while (end < path.size() && path[end] != '"') {
        end += path[end] == '\\' ? 2 : 1;
      }
      if (end >= path.size()) {
        return fail("Unterminated quotation", start);
      }
      ++end;
      std::string_view const quoted = path.substr(start, end - start);
      picojson::value name;
      std::string parseError;
      picojson::parse(name, quoted.begin(), quoted.end(), &parseError);
      if (!parseError.empty()) {
        return fail("Invalid argument: not valid JSON", start);
      }
      if (end < path.size() && path[end] != '/') {
        return fail("Expected \"/\" or end of path after closing quotation mark", end);
      }
      step.name = name.get<std::string>();
      step.isQuoted = true;
    } else {
      end = std::min(path.find('/', start), path.size());
      step.name = path.substr(start, end - start);
    }
    step.length = end - start;
    outSteps.push_back(std::move(step));
    if (end == path.size()) {
      return true;
    }
    start = end + 1;
  }
}

// Append text for use inside double quotes, escaping every character that ConsoleTokenize
// unescapes there.
static void AppendEscaped(std::string& out, std::string_view text) {
  for (char c : text) {
    for (char const escape : std::string_view{"ntrabfv\"\\"}) {
      if (EscapeChar(escape) == c) {
        out += '\\';
        c = escape;
        break;
      }
    }
    out += c;
  }
}

/**
Return a form of token that ConsoleTokenize turns back into exactly token: token itself if that
works and it contains no control characters, otherwise token in single quotes, or in double quotes
with escapes if it contains a single quote or a control character.
*/
static std::string QuoteConsoleToken(std::string_view token) {
  // Escapes keep a control character from breaking the line.
  bool const hasControlCharacter = std::any_of(
      token.begin(), token.end(), [](char c) { return static_cast<unsigned char>(c) < 0x20; });
  std::string error;
  auto const tokens = ConsoleTokenize(token, error);
  if (!hasControlCharacter && tokens.size() == 1 && tokens[0] == token) {
    return std::string{token};
  }
  if (!hasControlCharacter && token.find('\'') == std::string_view::npos) {
    return "'" + std::string{token} + "'";
  }
  std::string result = "\"";
  AppendEscaped(result, token);
  result += '"';
  return result;
}

// The number of UTF-8 characters in text. Continuation bytes look like 10xxxxxx.
static size_t CountCharacters(std::string_view text) {
  return static_cast<size_t>(std::count_if(text.begin(), text.end(), [](char c) {
    return (static_cast<unsigned char>(c) & 0xC0) != 0x80;
  }));
}

// Return the position in quoted, which is QuoteConsoleToken(token), of the character at offset in
// token.
static size_t CaretColumn(std::string_view token, std::string_view quoted, size_t offset) {
  offset = std::min(offset, token.size());
  if (quoted == token) {
    return offset;
  } else if (token.empty()) {
    return 0;
  } else if (quoted[0] == '\'') {
    return 1 + offset;
  }
  std::string escaped;
  AppendEscaped(escaped, token.substr(0, offset));
  return 1 + escaped.size();
}

std::string
detail::FormatConsoleError(std::string_view message, std::string_view command, size_t offset) {
  std::string displayed;
  size_t column = std::string::npos;
  for (size_t i = 0; i < command.size(); ++i) {
    if (i == offset) {
      column = displayed.size();
    }
    char const c = command[i];
    unsigned char const byte = static_cast<unsigned char>(c);
    if (byte < 0x20 || byte == 0x7F) {
      size_t const start = displayed.size();
      AppendEscaped(displayed, std::string_view{&c, 1});
      if (displayed.size() == start + 1) {
        displayed.pop_back();
        constexpr char hex[] = "0123456789abcdef";
        displayed += "\\x";
        displayed += hex[byte >> 4];
        displayed += hex[byte & 0xF];
      }
    } else {
      displayed += c;
    }
  }
  size_t const width = CountCharacters(std::string_view{displayed}.substr(0, column));
  return std::string{message} + "\n  | " + displayed + "\n  | " + std::string(width, ' ') + "^";
}

std::string detail::FormatConsoleError(
    std::string_view message,
    std::string_view path,
    Span<std::string const> args,
    size_t tokenIndex,
    size_t offset) {
  std::string command;
  size_t column = std::string::npos;
  for (size_t i = 0; i <= args.size(); ++i) {
    std::string_view const token = i == 0 ? path : std::string_view{args.data()[i - 1]};
    std::string const quoted = QuoteConsoleToken(token);
    if (i > 0) {
      command += ' ';
    }
    if (i == tokenIndex) {
      column = command.size() + CaretColumn(token, quoted, offset);
    }
    command += quoted;
  }
  return FormatConsoleError(message, command, column);
}

/////////////////////////////////////////////////////////////////////////////
// ConsoleExecute
/////////////////////////////////////////////////////////////////////////////

// A name as a JSON string, for messages
static std::string Quoted(std::string_view name) {
  return picojson::value{std::string{name}}.serialize();
}

// Return the last step of path, ignoring a trailing '/', or nullopt if there is none.
static std::optional<ConsolePathStep> LastStep(std::string_view path) {
  std::vector<ConsolePathStep> steps;
  if (!SplitConsolePath(path, steps)) {
    return std::nullopt;
  } else if (!steps.empty() && steps.back().length == 0) {
    steps.pop_back();
  }
  if (steps.empty()) {
    return std::nullopt;
  }
  return std::move(steps.back());
}

namespace {

constexpr size_t kNoStep = std::numeric_limits<size_t>::max();

// A value reached by walking a path. value is null when the walk is const.
struct WalkNode {
  void* value = nullptr;
  void const* constValue = nullptr;
  TypeInfo const* type = nullptr;
};

// What a path step names
struct Member {
  WalkNode parent; // Holds the member, with optionals and variants unwrapped
  FieldTypeInfo const* field = nullptr;
  MethodInfo const* method = nullptr;
  WalkNode element; // An array element or map value, when there is no field or method
};

struct WalkContext {
  std::string_view path;
  Span<std::string const> args;
  size_t pathStart = 0;
  TypeInfo const* rootType = nullptr;
  std::optional<ConsolePathStep> rootStep; // The last step of path[0, pathStart)
  std::vector<ConsolePathStep> steps; // Of path[pathStart, end), with offsets into path
  size_t step = kNoStep; // The step being processed, which exceptions are reported at
  size_t readOnlyStep = kNoStep; // The read-only member that made the rest of the walk const
  bool modified = false; // A set or non-const call succeeded
  bool isMethod = false; // The path resolves to a method
  std::string error;
  size_t errorToken = 0;
  size_t errorOffset = 0;

  bool Fail(size_t token, size_t offset, std::string message) {
    error = std::move(message);
    errorToken = token;
    errorOffset = offset;
    return false;
  }

  // Step i, or for kNoStep the step that reached the object, if any.
  [[nodiscard]] ConsolePathStep const* Step(size_t i) const {
    if (i != kNoStep) {
      assert(i < steps.size());
      return &steps[i];
    }
    return rootStep ? &*rootStep : nullptr;
  }

  // Report an error at step i, or at the object itself for kNoStep.
  bool FailAtStep(size_t i, std::string message) {
    ConsolePathStep const* const pathStep = Step(i);
    return Fail(0, pathStep != nullptr ? pathStep->offset : pathStart, std::move(message));
  }

  // The object itself is named by its type when no step reached it.
  [[nodiscard]] std::string StepName(size_t i) const {
    ConsolePathStep const* const pathStep = Step(i);
    return pathStep != nullptr ? Quoted(pathStep->name) : std::string{rootType->_name};
  }
};

// Owns values made by TypeInfo::New or Clone, and deletes them in reverse order.
class OwnedValues {
 public:
  OwnedValues() = default;
  OwnedValues(OwnedValues const&) = delete;
  OwnedValues& operator=(OwnedValues const&) = delete;
  OwnedValues(OwnedValues&&) = delete;
  OwnedValues& operator=(OwnedValues&&) = delete;
  ~OwnedValues() {
    for (auto it = _values.rbegin(); it != _values.rend(); ++it) {
      it->first->Delete(it->second);
    }
  }

  // Return nullptr if type cannot be default constructed.
  void* New(TypeInfo const& type) {
    auto& entry = _values.emplace_back(&type, nullptr);
    entry.second = type.New();
    return entry.second;
  }

  // Return nullptr if type cannot be copied.
  void* Clone(TypeInfo const& type, void const* src) {
    auto& entry = _values.emplace_back(&type, nullptr);
    entry.second = type.Clone(src);
    return entry.second;
  }

 private:
  std::vector<std::pair<TypeInfo const*, void*>> _values;
};

} // namespace

// Split path[pathStart, end) into ctx.steps. Unquoted empty, "." and ".." steps are errors.
static bool ParsePath(WalkContext& ctx) {
  std::string error;
  size_t offset = 0;
  if (!SplitConsolePath(ctx.path.substr(ctx.pathStart), ctx.steps, &error, &offset)) {
    return ctx.Fail(0, ctx.pathStart + offset, std::move(error));
  }
  for (size_t i = 0; i < ctx.steps.size(); ++i) {
    ConsolePathStep& step = ctx.steps[i];
    step.offset += ctx.pathStart;
    if (step.isQuoted) {
      continue;
    } else if (step.name.empty()) {
      return ctx.Fail(
          0, std::min(step.offset, ctx.path.size() - 1), "Unexpected \"/\" after empty string");
    } else if (step.name == "." || step.name == "..") {
      return ctx.FailAtStep(i, "Unsupported " + Quoted(step.name) + " in command path");
    }
  }
  return true;
}

static bool IsReadOnlyMember(FieldTypeInfo const& field) {
  return field.IsReadOnly() || field.HasAttribute<Attribute_ReadOnly>();
}

static bool HasReadOnlyDescendants(TypeInfo const& type, std::vector<TypeInfo const*>& visited) {
  if (std::find(visited.begin(), visited.end(), &type) != visited.end()) {
    return false;
  }
  visited.push_back(&type);
  if (type._coreType == CoreType::CT_struct) {
    for (FieldTypeInfo const* field : static_cast<StructTypeInfo const&>(type)._fields) {
      if (IsReadOnlyMember(*field) || HasReadOnlyDescendants(*field->_innerTypeInfo, visited)) {
        return true;
      }
    }
  } else if (type._coreType == CoreType::CT_array) {
    return HasReadOnlyDescendants(*static_cast<ArrayTypeInfo const&>(type)._innerTypeInfo, visited);
  } else if (type._coreType == CoreType::CT_matrix) {
    return HasReadOnlyDescendants(
        *static_cast<MatrixTypeInfo const&>(type)._innerTypeInfo, visited);
  } else if (type._coreType == CoreType::CT_map) {
    auto const& map = static_cast<MapTypeInfo const&>(type);
    return HasReadOnlyDescendants(*map._keyTypeInfo, visited) ||
        HasReadOnlyDescendants(*map._valueTypeInfo, visited);
  } else if (type._coreType == CoreType::CT_optional) {
    return HasReadOnlyDescendants(
        *static_cast<OptionalTypeInfo const&>(type)._innerTypeInfo, visited);
  } else if (type._coreType == CoreType::CT_variant) {
    for (TypeInfo const* inner : static_cast<VariantTypeInfo const&>(type)._innerTypes) {
      if (HasReadOnlyDescendants(*inner, visited)) {
        return true;
      }
    }
  } else if (auto const* pair = dynamic_cast<PairTypeInfo const*>(&type)) {
    return HasReadOnlyDescendants(*pair->_infoT, visited) ||
        HasReadOnlyDescendants(*pair->_infoU, visited);
  }
  return false;
}

// Return the text that names a map key in a path: its JSON text, as in MapTypeInfo::SerializeInner.
static std::string MapKeyText(MapTypeInfo const& mapType, void const* key) {
  picojson::value json;
  mapType._keyTypeInfo->SerializeInner(key, json);
  return json.is<std::string>() ? json.get<std::string>() : json.serialize();
}

// Replace node with the value inside its optionals and variants. Return false, with node at the
// optional or variant, if an optional is empty or a variant is valueless.
static bool Unwrap(WalkNode& node) {
  for (;;) {
    if (node.type->_coreType == CoreType::CT_optional) {
      auto const& optional = static_cast<OptionalTypeInfo const&>(*node.type);
      void const* const inner = optional.GetOptionalValue(node.constValue);
      if (inner == nullptr) {
        return false;
      }
      void* const value = node.value != nullptr ? optional.GetOptionalValue(node.value) : nullptr;
      node = {value, inner, optional._innerTypeInfo};
    } else if (node.type->_coreType == CoreType::CT_variant) {
      auto const& variant = static_cast<VariantTypeInfo const&>(*node.type);
      size_t const index = variant.GetInnerTypeIndex(node.constValue);
      if (index >= variant._innerTypes.size()) {
        return false;
      }
      void* const value = node.value != nullptr ? variant.GetInnerObject(node.value) : nullptr;
      node = {value, variant.GetInnerObject(node.constValue), variant._innerTypes[index]};
    } else {
      return true;
    }
  }
}

// Report that step reached node, an empty optional or a valueless variant.
static bool FailUnwrap(WalkContext& ctx, size_t step, WalkNode const& node) {
  bool const isOptional = node.type->_coreType == CoreType::CT_optional;
  return ctx.FailAtStep(
      step,
      std::string{isOptional ? "Optional " : "Variant "} + ctx.StepName(step) + " has no value");
}

// Unwrap node, then find what step i names in it.
static bool FindMember(WalkContext& ctx, WalkNode node, size_t i, Member& out) {
  ctx.step = i;
  if (!Unwrap(node)) {
    assert(i > 0); // The walk starts at a struct
    return FailUnwrap(ctx, i - 1, node);
  }
  out.parent = node;
  assert(i < ctx.steps.size());
  std::string const& name = ctx.steps[i].name;
  TypeInfo const& type = *node.type;
  if (type._coreType == CoreType::CT_struct) {
    auto const& structType = static_cast<StructTypeInfo const&>(type);
    out.field = structType.FindField(name);
    out.method = out.field != nullptr ? nullptr : structType.FindMethod(name);
    if (out.field != nullptr || out.method != nullptr) {
      return true;
    }
  } else if (type._coreType == CoreType::CT_array) {
    auto const& arrayType = static_cast<ArrayTypeInfo const&>(type);
    size_t index = 0;
    char const* const nameEnd = name.data() + name.size();
    auto const [indexEnd, indexError] = std::from_chars(name.data(), nameEnd, index);
    if (indexError == std::errc{} && indexEnd == nameEnd &&
        index < arrayType.GetNumElements(node.constValue)) {
      void* const value = node.value != nullptr ? arrayType.GetElement(node.value, index) : nullptr;
      out.element = {value, arrayType.GetElement(node.constValue, index), arrayType._innerTypeInfo};
      return true;
    }
  } else if (type._coreType == CoreType::CT_map) {
    auto const& mapType = static_cast<MapTypeInfo const&>(type);
    // Enumerate may ignore the callback's request to stop, so later keys must not undo a match.
    bool found = false;
    auto const find = [&](void const* key, auto* value) {
      if (!found && MapKeyText(mapType, key) == name) {
        found = true;
        if constexpr (std::is_same_v<decltype(value), void*>) {
          out.element = {value, value, mapType._valueTypeInfo};
        } else {
          out.element = {nullptr, value, mapType._valueTypeInfo};
        }
      }
      return !found;
    };
    if (node.value != nullptr) {
      mapType.Enumerate(node.value, find);
    } else {
      mapType.Enumerate(node.constValue, find);
    }
    if (found) {
      return true;
    }
  } else {
    assert(i > 0);
    return ctx.Fail(
        0, ctx.steps[i].offset - 1, "Unexpected \"/\" after value " + ctx.StepName(i - 1));
  }
  return ctx.FailAtStep(i, Quoted(name) + " not found");
}

using OnWalkEnd = FunctionRef<bool(WalkNode const& node)>;

// Walk steps [i, end) from node, then call onEnd with the value reached. Each member is stepped
// into inside the previous member's callback, so property values stay alive for the rest of the
// walk. A writable property is written back after the rest of the walk only if it modified
// something.
static bool Walk(WalkContext& ctx, WalkNode const& node, size_t i, size_t end, OnWalkEnd onEnd) {
  if (i == end) {
    return onEnd(node);
  }
  Member member;
  if (!FindMember(ctx, node, i, member)) {
    return false;
  }
  if (member.method != nullptr) {
    return ctx.Fail(
        0,
        ctx.steps[i + 1].offset - 1,
        "Unexpected \"/\" after method " + Quoted(member.method->_name));
  } else if (member.field == nullptr) {
    return Walk(ctx, member.element, i + 1, end, onEnd);
  }
  FieldTypeInfo const& field = *member.field;
  TypeInfo const* const type = field._innerTypeInfo;
  bool ok = false;
  if (member.parent.value != nullptr && !IsReadOnlyMember(field)) {
    field.ModifyValue(member.parent.value, [&](void* value) {
      ok = Walk(ctx, {value, value, type}, i + 1, end, onEnd);
      ctx.step = i; // An exception from the setter is reported at this step
      return ctx.modified;
    });
  } else {
    if (member.parent.value != nullptr) {
      ctx.readOnlyStep = i;
    }
    field.VisitValue(member.parent.constValue, [&](void const* value) {
      ok = Walk(ctx, {nullptr, value, type}, i + 1, end, onEnd);
    });
  }
  return ok;
}

// Serialize value into output as pretty JSON, reporting a NaN or infinity at the current step.
static bool
ToOutput(WalkContext& ctx, TypeInfo const& type, void const* value, std::string& output) {
  try {
    output = SaveToJsonString(value, type, true);
  } catch (std::overflow_error const& e) {
    if (std::string_view{e.what()} != "json: !finite") {
      throw;
    }
    return ctx.FailAtStep(ctx.step, "Value contains NaN or infinity, which JSON cannot show");
  }
  output.pop_back(); // Pretty JSON ends with a newline
  return true;
}

// Parse text as exactly one JSON value. Return the offset where parsing failed, or nullopt on
// success. picojson::parse ignores trailing text, so it would read "2kg" as 2.
static std::optional<size_t> ParseJson(std::string const& text, picojson::value& out) {
  picojson::default_parse_context context(&out);
  picojson::input<std::string::const_iterator> in(text.begin(), text.end());
  if (picojson::_parse(context, in)) {
    in.skip_ws();
    if (in.getc() == -1) {
      return std::nullopt;
    }
    in.ungetc();
  }
  // getc also returns a character the parser pushed back, which is where it stopped.
  size_t remaining = 0;
  while (in.getc() != -1) {
    ++remaining;
  }
  return text.size() - remaining;
}

// Load args[tokenIndex - 1] into value, an existing value of the given type. "null" clears an
// optional, a string or enum takes the argument as it is, and anything else must be JSON.
static bool LoadArgument(
    WalkContext& ctx,
    size_t tokenIndex,
    TypeInfo const& type,
    std::string const& label,
    void* value) {
  std::string const& arg = ctx.args.data()[tokenIndex - 1];
  bool const isOptional = type._coreType == CoreType::CT_optional;
  TypeInfo const& target =
      isOptional ? *static_cast<OptionalTypeInfo const&>(type)._innerTypeInfo : type;
  picojson::value json;
  if (isOptional && arg == "null") {
    // json is already null
  } else if (target._coreType == CoreType::CT_string || target._coreType == CoreType::CT_enum) {
    json = picojson::value{arg};
  } else {
    std::optional<size_t> errorOffset;
    try {
      errorOffset = ParseJson(arg, json);
    } catch (std::overflow_error const&) {
      // picojson throws for a number that a double cannot hold
      return ctx.Fail(tokenIndex, 0, "Invalid argument: number is out of range");
    }
    if (errorOffset) {
      return ctx.Fail(tokenIndex, *errorOffset, "Invalid argument: not valid JSON");
    }
  }
  std::vector<IssueReport> issues;
  type.Deserialize(json, value, DeserializeFlags::Default, &issues);
  if (!issues.empty()) {
    return ctx.Fail(tokenIndex, 0, label + ": " + issues.front().ToString());
  }
  return true;
}

// Call cb with the current value of a member that is not a method.
static void VisitMemberValue(Member const& member, FunctionRef<void(void const*)> cb) {
  if (member.field == nullptr) {
    cb(member.element.constValue);
  } else {
    member.field->VisitValue(member.parent.constValue, cb);
  }
}

// Read or set a value: a field or property, an array element or map value, or the object itself.
static bool ExecuteValue(WalkContext& ctx, Member const& member, std::string& output) {
  FieldTypeInfo const* const field = member.field;
  TypeInfo const& type = field != nullptr ? *field->_innerTypeInfo : *member.element.type;
  if (ctx.args.size() == 0) {
    bool ok = false;
    VisitMemberValue(member, [&](void const* value) { ok = ToOutput(ctx, type, value, output); });
    return ok;
  } else if (ctx.steps.empty()) {
    return ctx.FailAtStep(kNoStep, "Cannot set the root object. Set individual members instead.");
  }
  size_t const last = ctx.steps.size() - 1;
  if (ctx.args.size() > 1) {
    return ctx.Fail(2, 0, "Unexpected 2nd argument");
  }
  bool const isReadOnlyField = field != nullptr && IsReadOnlyMember(*field);
  if (isReadOnlyField || member.parent.value == nullptr) {
    size_t const readOnlyStep = isReadOnlyField ? last : ctx.readOnlyStep;
    return ctx.FailAtStep(
        readOnlyStep, "Cannot set value because " + ctx.StepName(readOnlyStep) + " is read-only");
  }
  std::vector<TypeInfo const*> visited;
  if (HasReadOnlyDescendants(type, visited)) {
    return ctx.FailAtStep(
        last,
        "Cannot set the whole value because it contains read-only members. Set individual members "
        "instead.");
  }
  // Load the argument onto a copy, so that a load failure changes nothing.
  OwnedValues owned;
  void* copy = nullptr;
  VisitMemberValue(member, [&](void const* value) { copy = owned.Clone(type, value); });
  if (copy == nullptr) {
    return ctx.FailAtStep(
        last, "Cannot set value because type " + Quoted(type._name) + " cannot be copied");
  } else if (!LoadArgument(ctx, 1, type, "Invalid argument", copy)) {
    return false;
  }
  if (field == nullptr) {
    type.Set(copy, member.element.value);
  } else if (!field->SetValue(member.parent.value, copy)) {
    return ctx.FailAtStep(last, "Failed to set value");
  }
  ctx.modified = true;
  return true;
}

static bool CallMethod(WalkContext& ctx, Member const& member, std::string& output) {
  MethodInfo const& method = *member.method;
  size_t const last = ctx.steps.size() - 1;
  size_t const numArgs = method._args.size();
  if (ctx.args.size() != numArgs) {
    std::string argNames;
    for (MethodArgInfo const& arg : method._args) {
      if (!argNames.empty()) {
        argNames += ", ";
      }
      argNames += arg._name;
    }
    std::string const message = "Expected " + std::to_string(numArgs) +
        (numArgs == 1 ? " argument" : " arguments") +
        (argNames.empty() ? "" : " (" + argNames + ")") + " but got " +
        std::to_string(ctx.args.size());
    // Point at the first extra argument, or at the method when arguments are missing.
    return ctx.args.size() > numArgs ? ctx.Fail(numArgs + 1, 0, message)
                                     : ctx.FailAtStep(last, message);
  } else if (!method._isConst && member.parent.value == nullptr) {
    return ctx.FailAtStep(
        ctx.readOnlyStep,
        "Cannot call non-const method " + Quoted(method._name) + " because " +
            ctx.StepName(ctx.readOnlyStep) + " is read-only");
  }
  OwnedValues owned;
  std::vector<void*> args;
  for (size_t i = 0; i < numArgs; ++i) {
    MethodArgInfo const& arg = method._args[i];
    TypeInfo const& type = arg.GetTypeInfo();
    std::string const argName = std::to_string(i + 1) + " (" + arg._name + ")";
    void* const value = owned.New(type);
    if (value == nullptr) {
      return ctx.FailAtStep(
          last,
          "Cannot call method: argument " + argName + " has type " + Quoted(type._name) +
              " which cannot be default constructed");
    } else if (!LoadArgument(ctx, i + 1, type, "Argument " + argName, value)) {
      return false;
    }
    args.push_back(value);
  }
  TypeInfo const* const returnType = method.GetReturnTypeInfo();
  void* returnValue = nullptr;
  if (returnType != nullptr) {
    returnValue = owned.New(*returnType);
    if (returnValue == nullptr) {
      return ctx.FailAtStep(
          last,
          "Cannot call method: return type " + Quoted(returnType->_name) +
              " cannot be default constructed");
    }
  }
  bool const called = member.parent.value != nullptr
      ? method.Call(member.parent.value, returnValue, args)
      : method.Call(member.parent.constValue, returnValue, args);
  if (!called) {
    return ctx.FailAtStep(last, "Failed to call method");
  }
  ctx.modified = !method._isConst;
  return returnType == nullptr || ToOutput(ctx, *returnType, returnValue, output);
}

static bool Execute(WalkContext& ctx, WalkNode const& root, std::string& output) {
  try {
    if (root.type->_coreType != CoreType::CT_struct) {
      return ctx.FailAtStep(kNoStep, "Requires a class or struct");
    } else if (!ParsePath(ctx)) {
      return false;
    } else if (ctx.steps.empty()) {
      Member member;
      member.element = root;
      return ExecuteValue(ctx, member, output);
    }
    size_t const last = ctx.steps.size() - 1;
    return Walk(ctx, root, 0, last, [&](WalkNode const& parent) {
      Member member;
      if (!FindMember(ctx, parent, last, member)) {
        return false;
      }
      ctx.isMethod = member.method != nullptr;
      return ctx.isMethod ? CallMethod(ctx, member, output) : ExecuteValue(ctx, member, output);
    });
  } catch (std::exception const& e) {
    return ctx.FailAtStep(ctx.step, std::string{"Exception: "} + e.what());
  } catch (...) {
    return ctx.FailAtStep(ctx.step, "Exception: unknown");
  }
}

bool detail::ConsoleExecute(
    void* obj,
    TypeInfo const& typeInfo,
    std::string_view path,
    Span<std::string const> args,
    std::string* outString,
    std::string* outError,
    size_t pathStart) {
  WalkContext ctx;
  ctx.path = path;
  ctx.args = args;
  ctx.pathStart = std::min(pathStart, path.size());
  ctx.rootType = &typeInfo;
  ctx.rootStep = LastStep(path.substr(0, ctx.pathStart));
  std::string output;
  bool ok = Execute(ctx, {obj, obj, &typeInfo}, output);
  if (ok && ctx.modified && !ctx.isMethod && outString != nullptr) {
    // Read after enclosing setters finish so their adjustments are reflected in the output.
    ctx.args = {};
    ok = Execute(ctx, {nullptr, obj, &typeInfo}, output);
  }
  if (outString != nullptr) {
    *outString = ok ? std::move(output) : std::string{};
  }
  if (ok) {
    if (outError != nullptr) {
      outError->clear();
    }
    return true;
  }
  std::string error = FormatConsoleError(ctx.error, path, args, ctx.errorToken, ctx.errorOffset);
  if (outError != nullptr) {
    *outError = std::move(error);
  } else {
    SR_LOG("%s", error.c_str());
  }
  return false;
}

bool ConsoleExecute(
    void* obj,
    TypeInfo const& typeInfo,
    std::string_view command,
    std::string* outString,
    std::string* outError) {
  std::string error;
  size_t errorOffset = 0;
  auto const tokens = detail::ConsoleTokenize(command, error, &errorOffset);
  if (error.empty() && !tokens.empty()) {
    return detail::ConsoleExecute(
        obj, typeInfo, tokens.front(), {tokens.data() + 1, tokens.size() - 1}, outString, outError);
  }
  if (error.empty()) {
    error = "Empty command";
  } else {
    error = detail::FormatConsoleError(error, command, errorOffset);
  }
  if (outString != nullptr) {
    outString->clear();
  }
  if (outError != nullptr) {
    *outError = std::move(error);
  } else {
    SR_LOG("%s", error.c_str());
  }
  return false;
}

} // namespace SReflect

#endif // SIMPLE_REFLECTION_ENABLE
