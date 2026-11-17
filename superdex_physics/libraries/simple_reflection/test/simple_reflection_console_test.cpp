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

#include <simple_reflection/simple_reflection_console_inl.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// Ignore warnings about semicolon after SR_EndStruct()
SR_WARNING_IGNORE_CLANG(clang diagnostic ignored "-Wextra-semi")

static void ExpectTokens(
    std::string_view input,
    std::vector<std::string> const& expected,
    std::string_view expectedError = {},
    size_t expectedErrorOffset = 0) {
  std::string error;
  size_t offset = std::string::npos;
  EXPECT_EQ(SReflect::detail::ConsoleTokenize(input, error, &offset), expected)
      << "input: " << input;
  EXPECT_EQ(error, expectedError) << "input: " << input;
  EXPECT_EQ(offset, expectedErrorOffset) << "input: " << input;
}

TEST(SReflectConsole, Tokenize) {
  // Empty and whitespace-only input yield no tokens.
  ExpectTokens("", {});
  ExpectTokens(" \t\n\r\f\v ", {});

  // Varying numbers of unquoted tokens, separated by assorted whitespace.
  ExpectTokens("hello", {"hello"});
  ExpectTokens("one two three", {"one", "two", "three"});
  ExpectTokens("  one \t two\nthree  ", {"one", "two", "three"});

  // Quotes group whitespace into a single token. Empty quotes give an empty token.
  ExpectTokens(R"(connect "one two three")", {"connect", "one two three"});
  ExpectTokens(R"('single quoted arg')", {"single quoted arg"});
  ExpectTokens(R"(a '' "" b)", {"a", "", "", "b"});

  // Bytes above 0x7F never separate tokens, so UTF-8 text stays intact.
  ExpectTokens("voil\xC3\xA0 caf\xC3\xA9", {"voil\xC3\xA0", "caf\xC3\xA9"});

  // Adjacent quoted and unquoted segments form a single token.
  ExpectTokens(R"(foo"bar baz"qux)", {"foobar bazqux"});

  // Backslash is literal outside quotes, so Windows-style paths survive intact.
  ExpectTokens(R"(C:\foo\bar)", {R"(C:\foo\bar)"});
  ExpectTokens(R"(a\b c)", {R"(a\b)", "c"});

  // Escapes are resolved inside double quotes.
  ExpectTokens(R"("a\"b")", {R"(a"b)"});
  ExpectTokens(R"("a\\b")", {R"(a\b)"});
  ExpectTokens(R"("\n\t\r\a\b\f\v")", {"\n\t\r\a\b\f\v"});
  ExpectTokens(R"("it\'s")", {"it's"});

  // Single quotes are literal, as in a POSIX shell, so a backslash does not escape the closing
  // quote. A single quote can be written in double quotes.
  ExpectTokens(R"('C:\temp\new')", {R"(C:\temp\new)"});
  ExpectTokens(R"('a\' b)", {R"(a\)", "b"});
  ExpectTokens(R"('a'"'"'b')", {"a'b"});
  ExpectTokens(R"(cd 'myMap/"a\"b"')", {"cd", R"(myMap/"a\"b")"});

  // An unsupported escape drops the backslash, emits the offending char literally, and reports
  // the first such occurrence via outError.
  ExpectTokens(R"("a\qb")", {"aqb"}, "Unsupported escape sequence", 2);

  // Only the first unsupported escape is reported.
  ExpectTokens(R"("\q\z")", {"qz"}, "Unsupported escape sequence", 1);
  ExpectTokens("\"caf\xC3\xA9\\q\"", {"caf\xC3\xA9q"}, "Unsupported escape sequence", 6);

  // An unterminated quoted segment is captured up to end of input.
  ExpectTokens(R"(connect "one two)", {"connect", "one two"});
  ExpectTokens(R"('unterminated single)", {"unterminated single"});

  // A backslash at the end of the input inside an unterminated double quote is dropped and
  // reported.
  ExpectTokens(R"("foo\)", {"foo"}, "Incomplete escape sequence", 4);

  // An error left over from an earlier call is cleared on success.
  std::string err = "stale";
  EXPECT_EQ(SReflect::detail::ConsoleTokenize("a b", err), (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(err, "");
}

TEST(SReflectConsole, TokenizeBrackets) {
  // A token that starts with a bracket runs to the matching bracket, keeping whitespace and quotes.
  ExpectTokens("gravity [0, -9.81, 0]", {"gravity", "[0, -9.81, 0]"});
  ExpectTokens(R"(names ["a b", "c"])", {"names", R"(["a b", "c"])"});

  // Brackets inside JSON strings, which may contain escaped quotes and backslashes, do not count.
  ExpectTokens(R"(p {"a": [1, 2], "s": "]"})", {"p", R"({"a": [1, 2], "s": "]"})"});
  ExpectTokens(R"(["a[\"]\\", "]"] x)", {R"(["a[\"]\\", "]"])", "x"});

  // Escapes that only JSON supports pass through unchanged, without an error.
  ExpectTokens(R"({"s": "\u00e9\/"})", {R"({"s": "\u00e9\/"})"});

  // A closing bracket of the wrong kind is plain text.
  ExpectTokens("[1} x", {"[1} x]"});

  // Brackets only group at the start of a token and outside quotes. The token continues after the
  // group until whitespace.
  ExpectTokens("a[1, 2]", {"a[1,", "2]"});
  ExpectTokens("'[1, 2'", {"[1, 2"});
  ExpectTokens(R"([1]x"a b" y)", {"[1]xa b", "y"});

  // At the end of the input, an open JSON string and then the open brackets are closed.
  ExpectTokens("x [1, [2, 3", {"x", "[1, [2, 3]]"});
  ExpectTokens(R"({"a": [1, "b)", {R"({"a": [1, "b"]})"});

  // Closing only appends, so this string stays unterminated for the JSON parser to report.
  ExpectTokens(R"(["a\)", {R"(["a\"])"});
}

namespace SReflectConsoleTest {
namespace {

enum class Kind { Rigid, Soft };

} // namespace
} // namespace SReflectConsoleTest

SR_BeginEnum(SReflectConsoleTest::Kind);
SR_EnumItem(Rigid);
SR_EnumItem(Soft);
SR_EndEnum();

namespace SReflectConsoleTest {

static int g_enabledChangedCount = 0;

namespace {

// A supported type with no default constructor
struct NoDefault {
  explicit NoDefault(float v) : value(v) {}
  float value;

  SR_BeginStruct(SReflectConsoleTest::NoDefault);
  SR_Field(value);
  SR_EndStruct();
};

// Its constructor can throw, so emplacing it can leave a variant valueless. The string keeps it
// from being trivially copyable, which would let libstdc++ avoid the valueless state.
struct Fragile {
  Fragile() = default;
  explicit Fragile(bool fail) {
    if (fail) {
      throw std::runtime_error("Fragile");
    }
  }
  std::string label;

  SR_BeginStruct(SReflectConsoleTest::Fragile);
  SR_Field(label);
  SR_EndStruct();
};

// Holds a variant that tests make valueless
struct Box {
  std::variant<int32_t, Fragile> content;

  SR_BeginStruct(SReflectConsoleTest::Box);
  SR_Field(content);
  SR_EndStruct();
};

struct Part {
  float mass = 1.0f;
  std::string label;
  int32_t limitStorage = 0;

  SR_BeginStruct(SReflectConsoleTest::Part);
  SR_Field(mass);
  SR_Field(label);
  // Accessors that throw: the getter for a corrupt value, and the setter to reject a negative one
  SR_Property(
      "limit",
      [](Part const& p) {
        if (p.limitStorage < 0) {
          throw std::runtime_error("limit is corrupted");
        }
        return p.limitStorage;
      },
      [](Part& p, int32_t limit) {
        if (limit < 0) {
          throw std::invalid_argument("limit must not be negative");
        }
        p.limitStorage = limit;
      });
  // Const, with an argument and a return value
  SR_Method("scaled(factor)", [](Part const& p, float factor) { return p.mass * factor; });
  // Non-const, with arguments and a return value, and throws to reject an argument
  SR_Method("add(amount, note)", [](Part& p, float amount, std::string const& note) {
    if (amount < 0) {
      throw std::invalid_argument("amount must not be negative");
    }
    p.mass += amount;
    p.label = note;
    return p.mass;
  });
  // Returns void
  SR_Method("clear()", [](Part& p) {
    p.mass = 0;
    p.label.clear();
  });
  // A return type that isn't default constructible
  SR_Method("snapshot()", [](Part const& p) { return NoDefault{p.mass}; });
  SR_EndStruct();
};

// Shared by the ConsoleExecute tests. Each member covers a different kind of step or value.
struct Machine {
  int32_t count = 3;
  std::string name = "m1";
  Kind kind = Kind::Rigid;
  bool enabled = true;
  Part factory{9.0f, "factory"};
  float ratio = 0.5f;
  std::vector<float> weights{0.5f, 1.5f};
  Part base;
  std::vector<Part> parts{Part{2.0f, "p0"}, Part{3.0f, "p1"}};
  std::unordered_map<std::string, Part> named{{"a/b", Part{4.0f, "ab"}}, {"..", Part{5.0f, "up"}}};
  std::optional<Part> spare;
  std::variant<int32_t, Part> slot{Part{6.0f, "slot"}};
  Part coreStorage;
  mutable int32_t coreGetCount = 0;
  int32_t coreSetCount = 0;

  SR_BeginStruct(SReflectConsoleTest::Machine);
  SR_Field(count);
  SR_Field(name);
  SR_Field(kind);
  SR_Field(enabled);
  SRA_OnChanged([] { ++g_enabledChangedCount; });
  SR_Field(factory);
  SRA_ReadOnly();
  SR_Field(ratio);
  SR_Field(weights);
  SR_Field(base);
  SR_Field(parts);
  SR_Field(named);
  SR_Field(spare);
  SR_Field(slot);
  // A writable property that counts accessor calls and keeps mass between 0 and 10
  SR_Property(
      "core",
      [](Machine const& m) {
        ++m.coreGetCount;
        return m.coreStorage;
      },
      [](Machine& m, Part const& core) {
        if (core.mass < 0) {
          throw std::invalid_argument("core mass must not be negative");
        }
        m.coreStorage = core;
        m.coreStorage.mass = std::min(core.mass, 10.0f);
        ++m.coreSetCount;
      });
  // A read-only property
  SR_Property("view", [](Machine const& m) { return m.base; });
  SR_EndStruct();
};

struct ReadOnlyValue {
  int32_t value = 1;

  SR_BeginStruct(SReflectConsoleTest::ReadOnlyValue);
  SR_Field(value);
  SRA_ReadOnly();
  SR_EndStruct();
};

struct ReadOnlyProperty {
  int32_t value = 1;

  SR_BeginStruct(SReflectConsoleTest::ReadOnlyProperty);
  SR_Property("value", [](ReadOnlyProperty const& p) { return p.value; });
  SR_EndStruct();
};

struct BulkValues {
  Machine object;
  std::vector<Machine> array{Machine{}};
  std::unordered_map<std::string, Machine> map{{"item", Machine{}}};
  std::optional<Machine> optional;
  std::variant<int32_t, Machine> variant;
  ReadOnlyProperty property;
  std::pair<ReadOnlyValue, int32_t> pairKey{ReadOnlyValue{}, 1};
  std::pair<int32_t, ReadOnlyValue> pairValue{1, ReadOnlyValue{}};
  std::vector<std::pair<int32_t, ReadOnlyValue>> pairArray{{1, ReadOnlyValue{}}};
  std::pair<int32_t, int32_t> writablePair{1, 2};
  mutable int32_t getterCalls = 0;
  int32_t setterCalls = 0;

  SR_BeginStruct(SReflectConsoleTest::BulkValues);
  SR_Field(object);
  SR_Field(array);
  SR_Field(map);
  SR_Field(optional);
  SR_Field(variant);
  SR_Field(property);
  SR_Field(pairKey);
  SR_Field(pairValue);
  SR_Field(pairArray);
  SR_Field(writablePair);
  SR_Property(
      "copy",
      [](BulkValues const& values) {
        ++values.getterCalls;
        return values.object;
      },
      [](BulkValues& values, Machine const& object) {
        values.object = object;
        ++values.setterCalls;
      });
  SR_EndStruct();
};

using Step = std::tuple<std::string, size_t, size_t, bool>; // name, offset, length, isQuoted

} // namespace

// SReflect::ToJsonString without its final newline
template <typename T>
static std::string PrettyJson(T const& value) {
  std::string json = SReflect::ToJsonString(value, /*pretty*/ true);
  json.pop_back();
  return json;
}

// Run a command that should succeed and return its output.
template <typename T>
static std::string
Execute(T& obj, std::string_view path, std::vector<std::string> const& args = {}) {
  std::string output;
  std::string error = "not cleared";
  EXPECT_TRUE(SReflect::detail::ConsoleExecute(&obj, path, args, &output, &error)) << error;
  EXPECT_EQ(error, "");
  return output;
}

// Run a command that should fail and return its error.
template <typename T>
static std::string
ExecuteError(T& obj, std::string_view path, std::vector<std::string> const& args = {}) {
  std::string output = "not cleared";
  std::string error;
  EXPECT_FALSE(SReflect::detail::ConsoleExecute(&obj, path, args, &output, &error)) << output;
  EXPECT_EQ(output, "");
  return error;
}

static std::vector<Step> Split(std::string_view path) {
  std::vector<SReflect::detail::ConsolePathStep> steps;
  std::string error;
  EXPECT_TRUE(SReflect::detail::SplitConsolePath(path, steps, &error)) << error;
  std::vector<Step> result;
  result.reserve(steps.size());
  for (SReflect::detail::ConsolePathStep const& step : steps) {
    result.emplace_back(step.name, step.offset, step.length, step.isQuoted);
  }
  return result;
}

// Return "offset: error" for a path that fails to split.
static std::string SplitError(std::string_view path) {
  std::vector<SReflect::detail::ConsolePathStep> steps;
  std::string error;
  size_t offset = 0;
  EXPECT_FALSE(SReflect::detail::SplitConsolePath(path, steps, &error, &offset));
  return std::to_string(offset) + ": " + error;
}

static void ExpectQuoted(std::string const& token, std::string const& expected) {
  std::string const error = SReflect::detail::FormatConsoleError("", token, {}, 0, 0);
  size_t const start = std::string_view{"\n  | "}.size();
  std::string const quoted = error.substr(start, error.find('\n', start) - start);
  EXPECT_EQ(quoted, expected);
  ExpectTokens(quoted, {token});
}

TEST(SReflectConsole, SplitPath) {
  EXPECT_EQ(Split(""), std::vector<Step>{});
  EXPECT_EQ(
      Split(R"(named/"a/b"/mass)"),
      (std::vector<Step>{{"named", 0, 5, false}, {"a/b", 6, 5, true}, {"mass", 12, 4, false}}));
  EXPECT_EQ(Split("\"a\\\"b\""), (std::vector<Step>{{"a\"b", 0, 6, true}}));

  // Empty, "." and ".." steps are returned as they are.
  EXPECT_EQ(
      Split("a//b/"),
      (std::vector<Step>{
          {"a", 0, 1, false}, {"", 2, 0, false}, {"b", 3, 1, false}, {"", 5, 0, false}}));
  EXPECT_EQ(Split("./.."), (std::vector<Step>{{".", 0, 1, false}, {"..", 2, 2, false}}));

  EXPECT_EQ(SplitError("named/\"a/b"), "6: Unterminated quotation");
  EXPECT_EQ(SplitError("\"\\q\""), "0: Invalid argument: not valid JSON");
  EXPECT_EQ(SplitError("\"a\"b"), "3: Expected \"/\" or end of path after closing quotation mark");
}

TEST(SReflectConsole, QuoteToken) {
  // A token stays as it is when the tokenizer gives it back unchanged.
  ExpectQuoted(R"(C:\a\b)", R"(C:\a\b)");
  ExpectQuoted("[0, 1]", "[0, 1]");

  ExpectQuoted("", "''");
  ExpectQuoted(R"(say "hi")", R"('say "hi"')");
  ExpectQuoted("[1, 2", "'[1, 2'");
  ExpectQuoted(R"(C:\a b)", R"('C:\a b')");

  // Double quotes when the token has a single quote, or a control character that would break the
  // line
  ExpectQuoted(R"(it's C:\)", R"("it's C:\\")");
  ExpectQuoted("a\nb", R"("a\nb")");
  ExpectQuoted("[1,\n\t2]", R"("[1,\n\t2]")");
}

TEST(SReflectConsole, FormatError) {
  std::vector<std::string> const args{"1", R"(it's "x")"};
  std::string const command = R"(parts/9/mass 1 "it's \"x\"")";
  EXPECT_EQ(
      SReflect::detail::FormatConsoleError("Bad step", "parts/9/mass", args, 0, 6),
      "Bad step\n  | " + command + "\n  |       ^");

  // The caret allows for the opening quote and the escapes before it.
  EXPECT_EQ(
      SReflect::detail::FormatConsoleError("Bad value", "parts/9/mass", args, 2, 6),
      "Bad value\n  | " + command + "\n  |                        ^");

  // The caret counts UTF-8 characters, and an offset past the end of its token stops at the end.
  std::vector<std::string> const one{"1"};
  EXPECT_EQ(
      SReflect::detail::FormatConsoleError("Bad value", "caf\xC3\xA9", one, 1, 0),
      "Bad value\n  | caf\xC3\xA9 1\n  |      ^");
  EXPECT_EQ(
      SReflect::detail::FormatConsoleError("Bad step", "ab", one, 0, 9),
      "Bad step\n  | ab 1\n  |   ^");

  std::vector<std::string> const multiline{"[1,\n\tinvalid]"};
  EXPECT_EQ(
      SReflect::detail::FormatConsoleError("Bad value", "weights", multiline, 1, 5),
      "Bad value\n"
      "  | weights \"[1,\\n\\tinvalid]\"\n"
      "  |                 ^");
}

TEST(SReflectConsole, FormatControlCharacters) {
  std::vector<std::tuple<char, std::string>> const cases{
      {'\0', "\\x00"},
      {'\x01', "\\x01"},
      {'\x1B', "\\x1b"},
      {'\x1F', "\\x1f"},
      {'\x7F', "\\x7f"},
      {'\n', "\\n"},
      {'\t', "\\t"},
      {'\r', "\\r"},
      {'\a', "\\a"},
      {'\b', "\\b"},
      {'\f', "\\f"},
      {'\v', "\\v"}};
  for (auto const& [control, escaped] : cases) {
    SCOPED_TRACE(static_cast<unsigned char>(control));
    std::string const command = "a" + std::string(1, control) + "b";
    std::string const display = "a" + escaped + "b";
    EXPECT_EQ(
        SReflect::detail::FormatConsoleError("Bad value", command, 1),
        "Bad value\n  | " + display + "\n  |  ^");
    EXPECT_EQ(
        SReflect::detail::FormatConsoleError("Bad value", command, 2),
        "Bad value\n  | " + display + "\n  | " + std::string(1 + escaped.size(), ' ') + "^");
  }

  std::vector<std::string> const args{std::string{"a\0b", 3}};
  EXPECT_EQ(
      SReflect::detail::FormatConsoleError("Bad value", "x", args, 1, 2),
      "Bad value\n  | x \"a\\x00b\"\n  |         ^");
}

TEST(SReflectConsole, Command) {
  Machine m;
  std::string output;
  std::string error;
  auto execute = [&](std::string_view command) {
    error = "not cleared";
    EXPECT_TRUE(SReflect::ConsoleExecute(&m, command, &output, &error)) << error;
    EXPECT_EQ(error, "");
    return output;
  };

  EXPECT_EQ(execute("count"), "3");
  EXPECT_EQ(execute("  count\t5\n"), "5");
  EXPECT_EQ(m.count, 5);
  EXPECT_EQ(execute(R"(name "front wheel")"), "\"front wheel\"");
  EXPECT_EQ(m.name, "front wheel");
  EXPECT_EQ(execute(R"(name "")"), "\"\"");
  EXPECT_EQ(m.name, "");
  EXPECT_EQ(execute("kind Soft"), "\"Soft\"");
  EXPECT_EQ(m.kind, Kind::Soft);
  EXPECT_EQ(execute("weights [1, 2, 3]"), PrettyJson(std::vector<float>{1, 2, 3}));
  EXPECT_EQ(m.weights, (std::vector<float>{1, 2, 3}));
  std::string const base = execute(R"(base {"mass": 2, "label": "front wheel"})");
  EXPECT_EQ(base, PrettyJson(m.base));
  EXPECT_EQ(m.base.mass, 2.0f);
  EXPECT_EQ(m.base.label, "front wheel");
  EXPECT_EQ(execute(R"(base/add 1.5 "rear wheel")"), "3.5");
  EXPECT_EQ(m.base.label, "rear wheel");
  EXPECT_EQ(execute("base/clear"), "");
  EXPECT_EQ(m.base.mass, 0.0f);
  EXPECT_EQ(execute(R"('named/"a/b"/label')"), "\"ab\"");
  EXPECT_EQ(execute("core/mass 20"), "10");
  EXPECT_EQ(m.coreStorage.mass, 10.0f);

  EXPECT_TRUE(
      SReflect::ConsoleExecute(
          static_cast<void*>(&m), SReflect::GetTypeInfo<Machine>(), "count", &output, &error));
  EXPECT_EQ(output, "5");
  EXPECT_TRUE(SReflect::ConsoleExecute(&m, "count"));
}

TEST(SReflectConsole, CommandErrors) {
  Machine m;
  std::vector<std::tuple<std::string, std::string>> const cases{
      {"", "Empty command"},
      {" \t\n ", "Empty command"},
      {R"(name "bad\q")",
       "Unsupported escape sequence\n"
       "  | name \"bad\\q\"\n"
       "  |          ^"},
      {R"("na\qme")",
       "Unsupported escape sequence\n"
       "  | \"na\\qme\"\n"
       "  |    ^"},
      {R"(name "bad\q\z")",
       "Unsupported escape sequence\n"
       "  | name \"bad\\q\\z\"\n"
       "  |          ^"},
      {R"(name "abc\)",
       "Incomplete escape sequence\n"
       "  | name \"abc\\\n"
       "  |          ^"},
      {R"(name "ok\nbad\q")",
       "Unsupported escape sequence\n"
       "  | name \"ok\\nbad\\q\"\n"
       "  |              ^"},
      {"name \"caf\xC3\xA9\\q\"",
       "Unsupported escape sequence\n"
       "  | name \"caf\xC3\xA9\\q\"\n"
       "  |           ^"},
      {"\tname\n\"bad\\q\"",
       "Unsupported escape sequence\n"
       "  | \\tname\\n\"bad\\q\"\n"
       "  |             ^"},
      {"count 2kg", "Invalid argument: not valid JSON\n  | count 2kg\n  |        ^"}};
  for (auto const& [command, expected] : cases) {
    SCOPED_TRACE(command);
    std::string output = "not cleared";
    std::string error = "not cleared";
    EXPECT_FALSE(SReflect::ConsoleExecute(&m, command, &output, &error));
    EXPECT_EQ(output, "");
    EXPECT_EQ(error, expected);
  }
  EXPECT_EQ(m.count, 3);
  EXPECT_EQ(m.name, "m1");

  testing::internal::CaptureStdout();
  EXPECT_FALSE(SReflect::ConsoleExecute(&m, ""));
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "[Simple Reflection] Empty command\n");
}

TEST(SReflectConsole, CommandWithoutOutput) {
  Machine m;
  std::string error = "not cleared";
  EXPECT_TRUE(SReflect::ConsoleExecute(&m, "core/mass 20", nullptr, &error)) << error;
  EXPECT_EQ(error, "");
  EXPECT_EQ(m.coreStorage.mass, 10.0f);
  EXPECT_EQ(m.coreGetCount, 1);
  EXPECT_EQ(m.coreSetCount, 1);

  EXPECT_TRUE(SReflect::ConsoleExecute(&m, "ratio 1e100", nullptr, &error)) << error;
  EXPECT_EQ(error, "");
  EXPECT_EQ(m.ratio, std::numeric_limits<float>::infinity());
}

TEST(SReflectConsole, Read) {
  Machine m;
  EXPECT_EQ(Execute(m, "name"), "\"m1\"");
  EXPECT_EQ(Execute(m, "kind"), "\"Rigid\"");
  EXPECT_EQ(Execute(m, "base/mass"), "1");
  EXPECT_EQ(Execute(m, "weights/1"), "1.5");
  EXPECT_EQ(Execute(m, R"(parts/"0"/mass)"), "2"); // A quoted step can be an index too
  EXPECT_EQ(Execute(m, R"(named/"a/b"/label)"), "\"ab\"");
  EXPECT_EQ(Execute(m, R"(named/".."/mass)"), "5");
  EXPECT_EQ(Execute(m, "slot/label"), "\"slot\"");
  EXPECT_EQ(Execute(m, "spare"), "null");

  // A read calls a property's getter once and never its setter.
  EXPECT_EQ(Execute(m, "core"), PrettyJson(m.coreStorage));
  EXPECT_EQ(m.coreGetCount, 1);
  EXPECT_EQ(m.coreSetCount, 0);

  EXPECT_EQ(Execute(m, ""), PrettyJson(m));
}

TEST(SReflectConsole, Set) {
  Machine m;
  m.base.label = "keep";

  // Strings and enums take the argument as it is.
  EXPECT_EQ(Execute(m, "name", {"123"}), "\"123\"");
  EXPECT_EQ(m.name, "123");
  EXPECT_EQ(Execute(m, "name", {""}), "\"\"");
  EXPECT_EQ(m.name, "");
  EXPECT_EQ(Execute(m, "kind", {"Soft"}), "\"Soft\"");
  EXPECT_EQ(m.kind, Kind::Soft);

  // Anything else is JSON.
  EXPECT_EQ(Execute(m, "count", {"5"}), "5");
  EXPECT_EQ(m.count, 5);
  EXPECT_EQ(Execute(m, "weights", {"[1, 2, 3]"}), PrettyJson(std::vector<float>{1, 2, 3}));
  EXPECT_EQ(m.weights, (std::vector<float>{1, 2, 3}));

  // Members that a JSON object leaves out keep their values.
  Execute(m, "base", {"{\"mass\": 2}"});
  EXPECT_EQ(m.base.mass, 2.0f);
  EXPECT_EQ(m.base.label, "keep");
  Execute(m, "parts/1", {"{\"label\": \"x\"}"});
  EXPECT_EQ(m.parts[1].mass, 3.0f);
  EXPECT_EQ(m.parts[1].label, "x");
  Execute(m, "parts", {R"([{"mass": 8, "label": "q"}])"});
  ASSERT_EQ(m.parts.size(), 1u);
  EXPECT_EQ(m.parts[0].mass, 8.0f);
  EXPECT_EQ(m.parts[0].label, "q");
  EXPECT_EQ(Execute(m, "named/\"..\"/mass", {"6"}), "6");
  EXPECT_EQ(m.named[".."].mass, 6.0f);

  // An object fills an empty optional, and its members can then be set.
  Execute(m, "spare", {"{\"label\": \"s\"}"});
  ASSERT_TRUE(m.spare.has_value());
  EXPECT_EQ(m.spare->mass, 1.0f);
  EXPECT_EQ(Execute(m, "spare/label"), "\"s\"");
  Execute(m, "spare/mass", {"2"});
  EXPECT_EQ(m.spare->mass, 2.0f);
  // "null" clears an optional.
  EXPECT_EQ(Execute(m, "spare", {"null"}), "null");
  EXPECT_FALSE(m.spare.has_value());
  // The members of a variant's current alternative can be set too.
  EXPECT_EQ(Execute(m, "slot/mass", {"7"}), "7");
  EXPECT_EQ(std::get<Part>(m.slot).mass, 7.0f);

  // A set fires Attribute_OnChanged, and a failed one doesn't.
  g_enabledChangedCount = 0;
  EXPECT_EQ(Execute(m, "enabled", {"false"}), "false");
  EXPECT_FALSE(m.enabled);
  EXPECT_EQ(g_enabledChangedCount, 1);
  EXPECT_EQ(
      ExecuteError(m, "enabled", {"2"}),
      "Invalid argument: Expected JSON boolean, found integer\n"
      "  | enabled 2\n"
      "  |         ^");
  EXPECT_EQ(g_enabledChangedCount, 1);
}

TEST(SReflectConsole, SetReadback) {
  Machine m;
  EXPECT_EQ(m.coreStorage.mass, 1.0f); // default value
  std::string const output = Execute(m, "core", {R"({"mass": 20})"});
  EXPECT_EQ(m.coreStorage.mass, 10.0f); // mass property setter clamps to 10.0
  EXPECT_EQ(output, PrettyJson(m.coreStorage));
  EXPECT_EQ(m.coreGetCount, 2);
  EXPECT_EQ(m.coreSetCount, 1);

  EXPECT_EQ(Execute(m, "core/mass", {"20"}), "10");
  EXPECT_EQ(m.coreStorage.mass, 10.0f);
  EXPECT_EQ(m.coreGetCount, 4);
  EXPECT_EQ(m.coreSetCount, 2);
}

TEST(SReflectConsole, SetWithoutOutput) {
  Machine m;
  std::string error = "not cleared";
  std::vector<std::string> const mass{"20"};
  EXPECT_TRUE(SReflect::detail::ConsoleExecute(&m, "core/mass", mass, nullptr, &error)) << error;
  EXPECT_EQ(error, "");
  EXPECT_EQ(m.coreStorage.mass, 10.0f);
  EXPECT_EQ(m.coreGetCount, 1);
  EXPECT_EQ(m.coreSetCount, 1);

  std::vector<std::string> const ratio{"1e100"};
  EXPECT_TRUE(SReflect::detail::ConsoleExecute(&m, "ratio", ratio, nullptr, &error)) << error;
  EXPECT_EQ(error, "");
  EXPECT_EQ(m.ratio, std::numeric_limits<float>::infinity());
}

TEST(SReflectConsole, SetReadbackFailure) {
  Machine m;
  EXPECT_EQ(
      ExecuteError(m, "ratio", {"1e100"}),
      "Value contains NaN or infinity, which JSON cannot show\n"
      "  | ratio 1e100\n"
      "  | ^");
  EXPECT_EQ(m.ratio, std::numeric_limits<float>::infinity());
}

TEST(SReflectConsole, FailedSets) {
  Machine m;

  // A set below a writable property reads it again after the setter runs.
  Execute(m, "core/mass", {"2"});
  EXPECT_EQ(m.coreStorage.mass, 2.0f);
  EXPECT_EQ(m.coreGetCount, 2);
  EXPECT_EQ(m.coreSetCount, 1);
  Execute(m, "core", {"{\"label\": \"c\"}"});
  EXPECT_EQ(m.coreStorage.mass, 2.0f);
  EXPECT_EQ(m.coreStorage.label, "c");
  EXPECT_EQ(m.coreGetCount, 4);
  EXPECT_EQ(m.coreSetCount, 2);

  // Bad JSON, a wrong type, or an exception from a setter or method below the property, or from the
  // property's own setter, leaves it unchanged.
  std::string const core = PrettyJson(m.coreStorage);
  EXPECT_EQ(
      ExecuteError(m, "core/mass", {"heavy"}),
      "Invalid argument: not valid JSON\n"
      "  | core/mass heavy\n"
      "  |           ^");
  EXPECT_EQ(
      ExecuteError(m, "core/mass", {"\"heavy\""}),
      "Invalid argument: Expected JSON number, found string\n"
      "  | core/mass '\"heavy\"'\n"
      "  |            ^");
  EXPECT_EQ(
      ExecuteError(m, "core/limit", {"-1"}),
      "Exception: limit must not be negative\n"
      "  | core/limit -1\n"
      "  |      ^");
  EXPECT_EQ(
      ExecuteError(m, "core/add", {"-1", "x"}),
      "Exception: amount must not be negative\n"
      "  | core/add -1 x\n"
      "  |      ^");
  EXPECT_EQ(
      ExecuteError(m, "core/mass", {"-1"}),
      "Exception: core mass must not be negative\n"
      "  | core/mass -1\n"
      "  | ^");
  EXPECT_EQ(PrettyJson(m.coreStorage), core);
  EXPECT_EQ(m.coreSetCount, 2);

  // Only a non-const call writes the property back.
  EXPECT_EQ(Execute(m, "core/scaled", {"2"}), "4");
  EXPECT_EQ(m.coreSetCount, 2);
  EXPECT_EQ(Execute(m, "core/add", {"1", "x"}), "3");
  EXPECT_EQ(m.coreSetCount, 3);

  // A value that fails to load leaves a field unchanged.
  EXPECT_EQ(
      ExecuteError(m, "base", {"{\"mass\": 9, \"label\": 5}"}),
      "Invalid argument: label: Expected JSON string, found integer\n"
      "  | base {\"mass\": 9, \"label\": 5}\n"
      "  |      ^");
  EXPECT_EQ(m.base.mass, 1.0f);

  // But a struct field is copied back member by member, so an accessor that throws during the copy
  // leaves the earlier members written.
  m.base.limitStorage = -1;
  EXPECT_EQ(
      ExecuteError(m, "base", {"{\"mass\": 9}"}),
      "Exception: limit is corrupted\n"
      "  | base {\"mass\": 9}\n"
      "  | ^");
  EXPECT_EQ(m.base.mass, 9.0f);
}

TEST(SReflectConsole, ReadOnly) {
  Machine m;
  EXPECT_EQ(
      ExecuteError(m, "factory", {"{\"mass\": 2}"}),
      "Cannot set value because \"factory\" is read-only\n"
      "  | factory {\"mass\": 2}\n"
      "  | ^");
  EXPECT_EQ(
      ExecuteError(m, "factory/mass", {"2"}),
      "Cannot set value because \"factory\" is read-only\n"
      "  | factory/mass 2\n"
      "  | ^");
  EXPECT_EQ(
      ExecuteError(m, "view", {"{\"mass\": 2}"}),
      "Cannot set value because \"view\" is read-only\n"
      "  | view {\"mass\": 2}\n"
      "  | ^");
  EXPECT_EQ(
      ExecuteError(m, "view/mass", {"2"}),
      "Cannot set value because \"view\" is read-only\n"
      "  | view/mass 2\n"
      "  | ^");
}

TEST(SReflectConsole, ReadOnlyBulkSets) {
  BulkValues values;
  std::string const before = PrettyJson(values);
  int32_t const getterCalls = values.getterCalls;
  std::vector<std::tuple<std::string, std::string>> const cases{
      {"object", R"({"factory": {"mass": 2}})"},
      {"object", R"({"count": 9})"},
      {"array", "[]"},
      {"array/0", "{}"},
      {"map", "{}"},
      {"map/item", "{}"},
      {"optional", "null"},
      {"variant", R"({"int32": 7})"},
      {"property", R"({"value": 2})"},
      {"pairKey", R"({"key": {"value": 2}, "value": 1})"},
      {"pairValue", R"({"key": 1, "value": {"value": 2}})"},
      {"pairArray/0", R"({"key": 1, "value": {"value": 2}})"},
      {"pairArray", "[]"},
      {"copy", R"({"count": 9})"}};
  for (auto const& [path, arg] : cases) {
    SCOPED_TRACE(path);
    std::string const error = ExecuteError(values, path, {arg});
    EXPECT_NE(
        error.find("contains read-only members. Set individual members instead"), std::string::npos)
        << error;
  }
  EXPECT_EQ(values.getterCalls, getterCalls);
  EXPECT_EQ(values.setterCalls, 0);
  EXPECT_EQ(PrettyJson(values), before);

  Execute(values, "object/count", {"8"});
  EXPECT_EQ(values.object.count, 8);
  Execute(values, "copy/count", {"9"});
  EXPECT_EQ(values.object.count, 9);
  EXPECT_EQ(values.setterCalls, 1);

  Execute(values, "writablePair", {R"({"key": 3, "value": 4})"});
  EXPECT_EQ(values.writablePair, (std::pair<int32_t, int32_t>{3, 4}));
}

TEST(SReflectConsole, Call) {
  Machine m;
  // A const method can be called through a read-only property.
  EXPECT_EQ(Execute(m, "view/scaled", {"3"}), "3");
  EXPECT_EQ(Execute(m, "base/add", {"1.5", "heavier"}), "2.5");
  EXPECT_EQ(m.base.mass, 2.5f);
  EXPECT_EQ(m.base.label, "heavier");
  EXPECT_EQ(Execute(m, "base/clear"), "");
  EXPECT_EQ(m.base.mass, 0.0f);
  EXPECT_EQ(m.base.label, "");
  EXPECT_EQ(Execute(m, "slot/add", {"1", "x"}), "7");
  EXPECT_EQ(std::get<Part>(m.slot).label, "x");

  EXPECT_EQ(
      ExecuteError(m, "view/add", {"1", "x"}),
      "Cannot call non-const method \"add\" because \"view\" is read-only\n"
      "  | view/add 1 x\n"
      "  | ^");
  EXPECT_EQ(
      ExecuteError(m, "base/add", {"1"}),
      "Expected 2 arguments (amount, note) but got 1\n"
      "  | base/add 1\n"
      "  |      ^");
  EXPECT_EQ(
      ExecuteError(m, "base/add", {"1", "x", "y"}),
      "Expected 2 arguments (amount, note) but got 3\n"
      "  | base/add 1 x y\n"
      "  |              ^");
  EXPECT_EQ(
      ExecuteError(m, "base/add", {"heavy", "x"}),
      "Invalid argument: not valid JSON\n"
      "  | base/add heavy x\n"
      "  |          ^");
  EXPECT_EQ(
      ExecuteError(m, "base/snapshot"),
      "Cannot call method: return type \"NoDefault\" cannot be default constructed\n"
      "  | base/snapshot\n"
      "  |      ^");
}

TEST(SReflectConsole, Errors) {
  Machine m;
  EXPECT_EQ(
      ExecuteError(m, "base/nope"),
      "\"nope\" not found\n"
      "  | base/nope\n"
      "  |      ^");
  EXPECT_EQ(
      ExecuteError(m, "base/mass/x"),
      "Unexpected \"/\" after value \"mass\"\n"
      "  | base/mass/x\n"
      "  |          ^");
  EXPECT_EQ(
      ExecuteError(m, "base/add/x"),
      "Unexpected \"/\" after method \"add\"\n"
      "  | base/add/x\n"
      "  |         ^");
  EXPECT_EQ(
      ExecuteError(m, R"(base/"add"/x)"),
      "Unexpected \"/\" after method \"add\"\n"
      "  | 'base/\"add\"/x'\n"
      "  |            ^");
  EXPECT_EQ(
      ExecuteError(m, "weights/5"),
      "\"5\" not found\n"
      "  | weights/5\n"
      "  |         ^");
  EXPECT_EQ(
      ExecuteError(m, "weights/99999999999999999999"),
      "\"99999999999999999999\" not found\n"
      "  | weights/99999999999999999999\n"
      "  |         ^");
  EXPECT_EQ(
      ExecuteError(m, "weights/x"),
      "\"x\" not found\n"
      "  | weights/x\n"
      "  |         ^");
  EXPECT_EQ(
      ExecuteError(m, "weights", {"[1,\n\tinvalid]"}),
      "Invalid argument: not valid JSON\n"
      "  | weights \"[1,\\n\\tinvalid]\"\n"
      "  |                 ^");
  EXPECT_EQ(
      ExecuteError(m, "named/nope"),
      "\"nope\" not found\n"
      "  | named/nope\n"
      "  |       ^");
  EXPECT_EQ(
      ExecuteError(m, "named/../mass"),
      "Unsupported \"..\" in command path\n"
      "  | named/../mass\n"
      "  |       ^");
  EXPECT_EQ(
      ExecuteError(m, "base//mass"),
      "Unexpected \"/\" after empty string\n"
      "  | base//mass\n"
      "  |      ^");
  EXPECT_EQ(
      ExecuteError(m, "base/"),
      "Unexpected \"/\" after empty string\n"
      "  | base/\n"
      "  |     ^");
  EXPECT_EQ(
      ExecuteError(m, "spare/mass"),
      "Optional \"spare\" has no value\n"
      "  | spare/mass\n"
      "  | ^");
  EXPECT_EQ(
      ExecuteError(m, "named/\"a/b"),
      "Unterminated quotation\n"
      "  | 'named/\"a/b'\n"
      "  |        ^");
  EXPECT_EQ(
      ExecuteError(m, "named/\"\\q\""),
      "Invalid argument: not valid JSON\n"
      "  | 'named/\"\\q\"'\n"
      "  |        ^");
  EXPECT_EQ(
      ExecuteError(m, R"(named/"a"b)"),
      "Expected \"/\" or end of path after closing quotation mark\n"
      "  | 'named/\"a\"b'\n"
      "  |           ^");
  EXPECT_EQ(
      ExecuteError(m, "count", {"1", "2"}),
      "Unexpected 2nd argument\n"
      "  | count 1 2\n"
      "  |         ^");
  EXPECT_EQ(
      ExecuteError(m, "", {"5"}),
      "Cannot set the root object. Set individual members instead.\n"
      "  | '' 5\n"
      "  | ^");
  EXPECT_EQ(
      ExecuteError(m, "count", {"2kg"}),
      "Invalid argument: not valid JSON\n"
      "  | count 2kg\n"
      "  |        ^");
  EXPECT_EQ(
      ExecuteError(m, "ratio", {"1e999"}),
      "Invalid argument: number is out of range\n"
      "  | ratio 1e999\n"
      "  |       ^");
  EXPECT_EQ(
      ExecuteError(m, "base", {"{mass: 2}"}),
      "Invalid argument: not valid JSON\n"
      "  | base {mass: 2}\n"
      "  |       ^");
  EXPECT_EQ(
      ExecuteError(m, "kind", {"Medium"}),
      "Invalid argument: Unknown enum value \"Medium\"\n"
      "  | kind Medium\n"
      "  |      ^");

  m.ratio = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(
      ExecuteError(m, "ratio"),
      "Value contains NaN or infinity, which JSON cannot show\n"
      "  | ratio\n"
      "  | ^");
  m.base.limitStorage = -1;
  EXPECT_EQ(
      ExecuteError(m, "base/limit"),
      "Exception: limit is corrupted\n"
      "  | base/limit\n"
      "  |      ^");

  // Without outError, the error is logged.
  testing::internal::CaptureStdout();
  EXPECT_FALSE(SReflect::detail::ConsoleExecute(&m, "nope", {}));
  EXPECT_EQ(
      testing::internal::GetCapturedStdout(),
      "[Simple Reflection] \"nope\" not found\n  | nope\n  | ^\n");

  float notAStruct = 0;
  std::string error;
  EXPECT_FALSE(
      SReflect::detail::ConsoleExecute(
          &notAStruct, SReflect::GetTypeInfo<float>(), "x", {}, nullptr, &error));
  EXPECT_EQ(error, "Requires a class or struct\n  | x\n  | ^");
}

TEST(SReflectConsole, PathStart) {
  // The caller resolved "machine/base/" to reach m.base.
  Machine m;
  std::string output;
  std::string error;
  EXPECT_TRUE(
      SReflect::detail::ConsoleExecute(&m.base, "machine/base/mass", {}, &output, &error, 13));
  EXPECT_EQ(output, "1");
  EXPECT_FALSE(
      SReflect::detail::ConsoleExecute(&m.base, "machine/base/nope", {}, &output, &error, 13));
  EXPECT_EQ(
      error,
      "\"nope\" not found\n"
      "  | machine/base/nope\n"
      "  |              ^");

  // An error about the object itself points at the step that reached it.
  std::vector<std::string> const args{"{}"};
  EXPECT_FALSE(
      SReflect::detail::ConsoleExecute(&m.base, "machine/base", args, &output, &error, 12));
  EXPECT_EQ(
      error,
      "Cannot set the root object. Set individual members instead.\n"
      "  | machine/base {}\n"
      "  |         ^");
}

// Make the variant in box valueless.
static void MakeValueless(Box& box) {
  try {
    box.content.emplace<Fragile>(true);
  } catch (std::runtime_error const&) {
  }
}

TEST(SReflectConsole, ValuelessVariant) {
  Box box;
  MakeValueless(box);
  ASSERT_TRUE(box.content.valueless_by_exception());
  std::string error;
  EXPECT_FALSE(SReflect::detail::ConsoleExecute(&box, "content/label", {}, nullptr, &error));
  EXPECT_EQ(error, "Variant \"content\" has no value\n  | content/label\n  | ^");
}

} // namespace SReflectConsoleTest
