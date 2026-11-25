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
#include <array>
#include <cstddef>
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
  std::vector<std::string> texts;
  for (auto const& token : SReflect::detail::ConsoleTokenize(input, error, &offset)) {
    texts.push_back(token.text);
  }
  EXPECT_EQ(texts, expected) << "input: " << input;
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
  auto const tokens = SReflect::detail::ConsoleTokenize("a b", err);
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].text, "a");
  EXPECT_EQ(tokens[1].text, "b");
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

TEST(SReflectConsole, TokenizeOffsets) {
  // text, offset and length of each token
  using Tokens = std::vector<std::tuple<std::string, size_t, size_t>>;
  auto const tokenize = [](std::string_view input) {
    std::string error;
    Tokens tokens;
    for (auto& token : SReflect::detail::ConsoleTokenize(input, error)) {
      tokens.emplace_back(std::move(token.text), token.offset, token.length);
    }
    return tokens;
  };

  // Whitespace ends a token, so only a token with nothing after it reaches the end of the input.
  EXPECT_EQ(tokenize("  ab  c"), (Tokens{{"ab", 2, 2}, {"c", 6, 1}}));
  EXPECT_EQ(tokenize("ab "), (Tokens{{"ab", 0, 2}}));

  // Removed quotes and escapes make text shorter than the input it came from.
  EXPECT_EQ(tokenize("'a b'/c \"d\\\"e\""), (Tokens{{"a b/c", 0, 7}, {"d\"e", 8, 6}}));
  EXPECT_EQ(tokenize("a\"b c\"d [1]x y"), (Tokens{{"ab cd", 0, 7}, {"[1]x", 8, 4}, {"y", 13, 1}}));

  // An unclosed quote or group runs to the end of the input, including trailing whitespace.
  EXPECT_EQ(tokenize("x \"a "), (Tokens{{"x", 0, 1}, {"a ", 2, 3}}));
  EXPECT_EQ(tokenize("[1, "), (Tokens{{"[1, ]", 0, 4}}));
}

namespace SReflectConsoleTest {
namespace {

enum class Kind { Rigid, Soft };
enum class Level { Low = 1, High = 2 }; // No item for zero, the default value
enum class Mode { Active = 1, Disabled = 0 };

// A row-major float matrix with fixed dimensions
template <size_t kRows, size_t kColumns>
struct Matrix {
  std::array<float, kRows * kColumns> values;
};

template <size_t kRows, size_t kColumns>
class MatrixInfo final : public SReflect::MatrixTypeInfo {
 public:
  MatrixInfo() {
    _coreType = SReflect::CoreType::CT_matrix;
    _sizeInBytes = sizeof(Matrix<kRows, kColumns>);
    _alignment = alignof(Matrix<kRows, kColumns>);
    _name = _nameStorage.c_str();
    _nameWithNamespace = _name;
    _typeId = SReflect::ComputeTypeId(_name);
    _innerTypeInfo = &SReflect::GetTypeInfo<float>();
    _isRowMajor = true;
    _fixedNumRows = kRows;
    _fixedNumColumns = kColumns;
  }

  void* GetData(void* obj) const override {
    return static_cast<Matrix<kRows, kColumns>*>(obj)->values.data();
  }

  bool TryResize(void* /*obj*/, size_t rows, size_t columns) const override {
    return rows == kRows && columns == kColumns;
  }

 protected:
  Layout GetLayoutImpl(void const* /*obj*/) const override {
    return {kRows, kColumns, kColumns};
  }

 private:
  std::string const _nameStorage =
      "Matrix" + std::to_string(kRows) + "x" + std::to_string(kColumns);
};

} // namespace
} // namespace SReflectConsoleTest

SR_BeginEnum(SReflectConsoleTest::Kind);
SR_EnumItem(Rigid);
SR_EnumItem(Soft);
SR_EndEnum();

SR_BeginEnum(SReflectConsoleTest::Level);
SR_EnumItem(Low);
SR_EnumItem(High);
SR_EndEnum();

SR_BeginEnum(SReflectConsoleTest::Mode);
SR_EnumItem(Active);
SR_EnumItem(Disabled);
SR_EndEnum();

template <size_t kRows, size_t kColumns>
struct SReflectTypeTraits<SReflectConsoleTest::Matrix<kRows, kColumns>> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_matrix;
  static SReflect::MatrixTypeInfo const& GetTypeInfo() {
    static SReflectConsoleTest::MatrixInfo<kRows, kColumns> const info;
    return info;
  }
};

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

// Holds values that a listing cannot show. Tests make the variant valueless.
struct Box {
  std::variant<int32_t, Fragile> content;
  std::unordered_map<float, int32_t> weights{
      {1.5f, 1},
      {std::numeric_limits<float>::quiet_NaN(), 2}};

  SR_BeginStruct(SReflectConsoleTest::Box);
  SR_Field(content);
  SR_Field(weights);
  // Throws a message with two lines while the variant is valueless
  SR_Property("note", [](Box const& box) {
    if (box.content.valueless_by_exception()) {
      throw std::runtime_error("line one\nline two");
    }
    return std::string{};
  });
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
          throw std::runtime_error(p.limitStorage == -2 ? "" : "limit is corrupted");
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
  SR_Method("debugDump()", [](Part const&) {});
  SRA_HideFromEditor();
  SR_EndStruct();
};

// Shared by the console tests. Each member covers a different kind of step or value.
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
  std::vector<std::unordered_map<std::string, int32_t>> tables{{{"x", 1}}};
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
  SR_Field(tables);
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
  // A read-only property, left out of listings
  SR_Property("view", [](Machine const& m) { return m.base; });
  SRA_HideFromEditor();
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

struct ArrayCases {
  mutable int32_t primitiveGets = 0;
  std::array<float, 3> positionStorage{1, 2, 3};
  mutable int32_t positionGets = 0;
  int32_t positionSets = 0;
  std::array<std::array<int32_t, 2>, 2> nested{{{1, 2}, {3, 4}}};
  std::vector<std::optional<Part>> parts{Part{}, std::nullopt};
  std::vector<bool> flags{true, false};

  SR_BeginStruct(SReflectConsoleTest::ArrayCases);
  SR_Property("count", [](ArrayCases const& obj) {
    ++obj.primitiveGets;
    return int32_t{5};
  });
  SR_Property(
      "position",
      [](ArrayCases const& obj) {
        ++obj.positionGets;
        return obj.positionStorage;
      },
      [](ArrayCases& obj, std::array<float, 3> const& value) {
        obj.positionStorage = value;
        ++obj.positionSets;
      });
  SR_Field(nested);
  SR_Field(parts);
  SR_Field(flags);
  SR_EndStruct();
};

struct MatrixCases {
  Matrix<2, 3> grid{{1, 2, 3, 4, 5, 6}};
  Matrix<3, 1> column{{7, 8, 9}};

  SR_BeginStruct(SReflectConsoleTest::MatrixCases);
  SR_Field(grid);
  SR_Field(column);
  SR_EndStruct();
};

using Step = std::tuple<std::string, size_t, size_t, bool>; // name, offset, length, isQuoted
// A ConsoleListEntry: the initial of its kind, then name, type and value
using Row = std::tuple<char, std::string, std::string, std::string>;

// Completion cases that Machine does not cover. A leading '_' has no special meaning.
struct Completions {
  int32_t b = 0;
  int32_t _underscore = 0;
  int32_t a = 0;
  std::vector<int32_t> items = std::vector<int32_t>(12);
  std::unordered_map<std::string, int32_t> keys{{"9", 0}, {"10", 0}, {"1a", 0}};
  std::optional<int32_t> limit = 5;
  float nan = std::numeric_limits<float>::quiet_NaN();
  std::string quote = "it's";

  SR_BeginStruct(SReflectConsoleTest::Completions);
  SR_Field(b);
  SR_Field(_underscore);
  SR_Field(a);
  SR_Field(items);
  SR_Field(keys);
  SR_Field(limit);
  SR_Field(nan);
  SR_Field(quote);
  SR_Method("setLevel(level)", [](Completions&, Level) {});
  SR_Method("setMode(mode)", [](Completions&, Mode) {});
  SR_Method("noDefault(value)", [](Completions&, int32_t) { return NoDefault{1}; });
  SR_EndStruct();
};

struct UnreadableProperty {
  bool readable = false;

  SR_BeginStruct(SReflectConsoleTest::UnreadableProperty);
  SR_Property(
      "enabled",
      [](UnreadableProperty const& property) {
        if (!property.readable) {
          throw std::runtime_error("unavailable");
        }
        return false;
      },
      [](UnreadableProperty&, bool) {});
  SR_EndStruct();
};

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

// List a path that should succeed, as rows.
template <typename T>
static std::vector<Row> List(
    T const& obj,
    std::string_view path,
    size_t pathStart = 0,
    SReflect::ConsoleListParams const& params = {}) {
  std::string error = "not cleared";
  auto const entries = SReflect::ConsoleList(&obj, path, params, &error, pathStart);
  EXPECT_EQ(error, "");
  std::vector<Row> rows;
  rows.reserve(entries.size());
  for (SReflect::ConsoleListEntry const& entry : entries) {
    rows.emplace_back(
        SReflect::EnumToString(entry._kind)[0], entry._name, entry._typeName, entry._value);
  }
  return rows;
}

using ValueError = std::pair<std::string, std::string>;

template <typename T>
static std::vector<ValueError> ListValueErrors(
    T const& obj,
    std::string_view path,
    SReflect::ConsoleListParams const& params = {}) {
  std::string error = "not cleared";
  auto const entries = SReflect::ConsoleList(&obj, path, params, &error);
  EXPECT_TRUE(error.empty());
  std::vector<ValueError> errors;
  for (auto const& entry : entries) {
    if (!entry._valueError.empty()) {
      EXPECT_TRUE(entry._value.empty());
      EXPECT_FALSE(entry._isValueTruncated);
      errors.emplace_back(entry._name, entry._valueError);
    }
  }
  return errors;
}

// List a path that should fail and return the error.
template <typename T>
static std::string ListError(T const& obj, std::string_view path, size_t pathStart = 0) {
  std::string error;
  auto const entries = SReflect::ConsoleList(&obj, path, {}, &error, pathStart);
  EXPECT_FALSE(error.empty());
  EXPECT_TRUE(entries.empty());
  return error;
}

// Replace the row that has the same name as row.
static void ReplaceRow(std::vector<Row>& rows, Row const& row) {
  auto const it = std::find_if(
      rows.begin(), rows.end(), [&](Row const& r) { return std::get<1>(r) == std::get<1>(row); });
  ASSERT_NE(it, rows.end()) << std::get<1>(row);
  *it = row;
}

// The listing of a Part with limit 0
static std::vector<Row> PartRows(std::string const& mass, std::string const& label) {
  return {
      {'F', "mass", "float", mass},
      {'F', "label", "string", "\"" + label + "\""},
      {'F', "limit", "int32", "0"},
      {'M', "scaled", "(float factor) -> float", ""},
      {'M', "add", "(float amount, string note) -> float", ""},
      {'M', "clear", "() -> void", ""},
      {'M', "snapshot", "() -> NoDefault", ""},
  };
}

TEST(SReflectConsole, List) {
  Machine m;
  // Fields and properties in declaration order, leaving out the hidden "view"
  std::vector<Row> rows{
      {'F', "count", "int32", "3"},
      {'F', "name", "string", "\"m1\""},
      {'F', "kind", "Kind", "\"Rigid\""},
      {'F', "enabled", "bool", "true"},
      {'D', "factory", "Part", ""},
      {'F', "ratio", "float", "0.5"},
      {'D', "weights", "vector<float>", ""},
      {'D', "base", "Part", ""},
      {'D', "parts", "vector<Part>", ""},
      {'D', "named", "unordered_map<string,Part>", ""},
      {'F', "spare", "optional<Part>", "null"},
      {'D', "slot", "variant<int32,Part>", ""},
      {'D', "tables", "vector<unordered_map<string,int32>>", ""},
      {'D', "core", "Part", ""},
  };
  EXPECT_EQ(List(m, ""), rows);
  // A member whose type is always a directory is not read.
  EXPECT_EQ(m.coreGetCount, 0);

  // Methods come after fields, leaving out the hidden "debugDump".
  EXPECT_EQ(List(m, "base"), PartRows("1", ""));

  // Hidden members can still be named in a path.
  EXPECT_EQ(List(m, "view"), PartRows("1", ""));
  EXPECT_EQ(Execute(m, "base/debugDump"), "");

  // Listing a property reads it once and never writes it.
  EXPECT_EQ(List(m, "core"), PartRows("1", ""));
  EXPECT_EQ(m.coreGetCount, 1);
  EXPECT_EQ(m.coreSetCount, 0);

  // An optional or variant is a directory while it holds one, and keeps its declared type.
  m.spare = Part{};
  m.slot = 7;
  ReplaceRow(rows, {'D', "spare", "optional<Part>", ""});
  ReplaceRow(rows, {'F', "slot", "variant<int32,Part>", "{\"int32\":7}"});
  EXPECT_EQ(List(m, ""), rows);
}

TEST(SReflectConsole, ListContainers) {
  Machine m;
  EXPECT_EQ(List(m, "parts"), (std::vector<Row>{{'D', "0", "Part", ""}, {'D', "1", "Part", ""}}));
  EXPECT_EQ(List(m, "parts/1"), PartRows("3", "p1"));

  // Map keys are sorted, and quoted when needed.
  m.named["b"] = Part{};
  m.named["a"] = Part{};
  EXPECT_EQ(
      List(m, "named"),
      (std::vector<Row>{
          {'D', "\"..\"", "Part", ""},
          {'D', "\"a/b\"", "Part", ""},
          {'D', "a", "Part", ""},
          {'D', "b", "Part", ""}}));

  // Quoted keys, variants and optionals lead to the value they hold.
  EXPECT_EQ(List(m, "named/\"a/b\""), PartRows("4", "ab"));
  EXPECT_EQ(List(m, "named/\"..\""), PartRows("5", "up"));
  EXPECT_EQ(List(m, "slot"), PartRows("6", "slot"));
  m.spare = Part{7.0f, "s"};
  EXPECT_EQ(List(m, "spare"), PartRows("7", "s"));

  // An array of maps, whose values are not directories
  EXPECT_EQ(List(m, "tables"), (std::vector<Row>{{'D', "0", "unordered_map<string,int32>", ""}}));
  EXPECT_EQ(List(m, "tables/0"), (std::vector<Row>{{'F', "x", "int32", "1"}}));

  // The caller resolved "machine/" to reach m.
  EXPECT_EQ(List(m, "machine/base", 8), PartRows("1", ""));
}

TEST(SReflectConsole, ListValues) {
  Machine m;
  std::vector<Row> rows = List(m, "");

  // A NaN or a getter that throws only affects that entry.
  m.ratio = std::numeric_limits<float>::quiet_NaN();
  ReplaceRow(rows, {'F', "ratio", "float", ""});
  EXPECT_EQ(List(m, ""), rows);
  EXPECT_EQ(
      ListValueErrors(m, ""),
      (std::vector<ValueError>{
          {"ratio", "Value contains NaN or infinity, which JSON cannot show"}}));
  m.base.limitStorage = -1;
  std::vector<Row> base = PartRows("1", "");
  ReplaceRow(base, {'F', "limit", "int32", ""});
  EXPECT_EQ(List(m, "base"), base);
  for (size_t cap : {0, 4, 1024}) {
    EXPECT_EQ(
        ListValueErrors(m, "base", {cap}),
        (std::vector<ValueError>{{"limit", "limit is corrupted"}}));
  }
  EXPECT_TRUE(ListValueErrors(m, "base", {0, false, false}).empty());
  m.base.limitStorage = -2;
  EXPECT_EQ(
      ListValueErrors(m, "base", {0}),
      (std::vector<ValueError>{{"limit", "exception with no message"}}));
}

TEST(SReflectConsole, ListErrors) {
  Machine m;
  EXPECT_EQ(
      ListError(m, "count"),
      "Not a directory\n"
      "  | count\n"
      "  | ^");
  EXPECT_EQ(
      ListError(m, "weights/0"),
      "Not a directory\n"
      "  | weights/0\n"
      "  |         ^");
  EXPECT_EQ(
      ListError(m, "base/add"),
      "Not a directory\n"
      "  | base/add\n"
      "  |      ^");
  EXPECT_EQ(
      ListError(m, "spare"),
      "Optional \"spare\" has no value\n"
      "  | spare\n"
      "  | ^");
  EXPECT_EQ(
      ListError(m, "base/nope"),
      "\"nope\" not found\n"
      "  | base/nope\n"
      "  |      ^");
  EXPECT_EQ(
      ListError(m, "machine/count", 8),
      "Not a directory\n"
      "  | machine/count\n"
      "  |         ^");
  m.slot = 7;
  EXPECT_EQ(
      ListError(m, "slot"),
      "Not a directory\n"
      "  | slot\n"
      "  | ^");

  // Without outError, the error is logged and the listing is cleared.
  testing::internal::CaptureStdout();
  EXPECT_TRUE(SReflect::ConsoleList(&m, "nope").empty());
  EXPECT_EQ(
      testing::internal::GetCapturedStdout(),
      "[Simple Reflection] \"nope\" not found\n  | nope\n  | ^\n");

  float const notAStruct = 0;
  std::string error;
  EXPECT_TRUE(
      SReflect::ConsoleList(&notAStruct, SReflect::GetTypeInfo<float>(), "", {}, &error).empty());
  EXPECT_EQ(error, "Requires a class or struct\n  | ''\n  | ^");
}

TEST(SReflectConsole, ListValuesThatCannotBeShown) {
  Box box;
  MakeValueless(box);
  ASSERT_TRUE(box.content.valueless_by_exception());

  // Entry errors preserve the exception message.
  EXPECT_EQ(
      List(box, ""),
      (std::vector<Row>{
          {'F', "content", "variant<int32,Fragile>", ""},
          {'D', "weights", "unordered_map<float,int32>", ""},
          {'F', "note", "string", ""}}));
  EXPECT_EQ(
      ListValueErrors(box, ""),
      (std::vector<ValueError>{{"content", "Invalid type index"}, {"note", "line one\nline two"}}));
  EXPECT_EQ(
      ListError(box, "content"),
      "Variant \"content\" has no value\n"
      "  | content\n"
      "  | ^");

  // A key that cannot be shown gets an entry of its own.
  EXPECT_EQ(
      List(box, "weights"),
      (std::vector<Row>{{'F', "1.5", "int32", "1"}, {'F', "?", "int32", ""}}));
  EXPECT_EQ(
      ListValueErrors(box, "weights"),
      (std::vector<ValueError>{{"?", "The key contains NaN or infinity, which JSON cannot show"}}));
}

TEST(SReflectConsole, ListPrimitiveArrays) {
  ArrayCases obj;
  EXPECT_EQ(
      List(obj, ""),
      (std::vector<Row>{
          {'F', "count", "int32", "5"},
          {'D', "position", "array<float,3>", "[1,2,3]"},
          {'D', "nested", "array<array<int32,2>,2>", "[[1,2],[3,4]]"},
          {'D', "parts", "vector<optional<Part>>", ""},
          {'F', "flags", "vector<bool>", ""}}));
  EXPECT_EQ(
      List(obj, "position"),
      (std::vector<Row>{
          {'F', "0", "float", "1"}, {'F', "1", "float", "2"}, {'F', "2", "float", "3"}}));
  EXPECT_EQ(
      List(obj, "nested"),
      (std::vector<Row>{
          {'D', "0", "array<int32,2>", "[1,2]"}, {'D', "1", "array<int32,2>", "[3,4]"}}));
  EXPECT_EQ(
      List(obj, "parts"),
      (std::vector<Row>{{'D', "0", "optional<Part>", ""}, {'F', "1", "optional<Part>", "null"}}));
  EXPECT_EQ(ListError(obj, "flags"), "Not a directory\n  | flags\n  | ^");

  EXPECT_EQ(Execute(obj, "position/0", {"42.0"}), "42");
  EXPECT_EQ(obj.positionStorage, (std::array<float, 3>{42, 2, 3}));
  EXPECT_EQ(obj.positionSets, 1);
  obj.parts.clear();
  std::string error = "not cleared";
  EXPECT_TRUE(SReflect::ConsoleList(&obj, "parts", {}, &error).empty());
  EXPECT_TRUE(error.empty());
}

TEST(SReflectConsole, ListPrimitiveMatrices) {
  MatrixCases obj;
  for (bool primitives : {false, true}) {
    for (bool arrays : {false, true}) {
      EXPECT_EQ(
          List(obj, "", 0, {1024, primitives, arrays}),
          (std::vector<Row>{
              {'F', "grid", "Matrix2x3", arrays ? "[[1,2,3],[4,5,6]]" : ""},
              {'F', "column", "Matrix3x1", arrays ? "[7,8,9]" : ""}}));
    }
  }
  EXPECT_EQ(ListError(obj, "grid"), "Not a directory\n  | grid\n  | ^");
  auto const entries = SReflect::ConsoleList(&obj, "", {4});
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0]._value, "[[1,");
  EXPECT_EQ(entries[1]._value, "[7,8");
  EXPECT_TRUE(entries[0]._isValueTruncated);
  EXPECT_TRUE(entries[1]._isValueTruncated);
}

TEST(SReflectConsole, ListSerializationOptions) {
  for (bool primitives : {false, true}) {
    for (bool arrays : {false, true}) {
      Machine obj;
      SReflect::ConsoleListParams const params{1024, primitives, arrays};
      auto const entries = SReflect::ConsoleList(&obj, "", params);
      for (auto const& entry : entries) {
        if (entry._name == "count") {
          EXPECT_EQ(entry._value, primitives ? "3" : "");
        } else if (entry._name == "name") {
          EXPECT_EQ(entry._value, primitives ? "\"m1\"" : "");
        } else if (entry._name == "kind") {
          EXPECT_EQ(entry._value, primitives ? "\"Rigid\"" : "");
        } else if (entry._name == "weights") {
          EXPECT_TRUE(entry._value.empty());
          EXPECT_EQ(entry._coreType, SReflect::CoreType::CT_array);
          EXPECT_EQ(entry._containerSize, 2);
        } else if (entry._kind == SReflect::ConsoleEntryKind::Directory) {
          EXPECT_TRUE(entry._value.empty());
        }
        EXPECT_FALSE(entry._isValueTruncated);
        EXPECT_TRUE(entry._valueError.empty());
      }
      EXPECT_EQ(obj.coreGetCount, 0);
      ArrayCases arrayObj;
      EXPECT_FALSE(SReflect::ConsoleList(&arrayObj, "", params).empty());
      EXPECT_EQ(arrayObj.primitiveGets, primitives ? 1 : 0);
      EXPECT_EQ(arrayObj.positionGets, 1);
    }
  }
  ArrayCases obj;
  SReflect::ConsoleListParams const params{1024, false, false};
  auto const entries = SReflect::ConsoleList(&obj, "", params);
  EXPECT_EQ(obj.positionGets, 1);
  for (auto const& entry : entries) {
    EXPECT_TRUE(entry._value.empty());
  }
  EXPECT_EQ(
      List(obj, "position", 0, params),
      (std::vector<Row>{
          {'F', "0", "float", ""}, {'F', "1", "float", ""}, {'F', "2", "float", ""}}));
  EXPECT_EQ(obj.positionGets, 2);
  EXPECT_EQ(obj.positionSets, 0);
}

TEST(SReflectConsole, ListValueTruncation) {
  Machine obj;
  auto const nameEntry = [&](size_t limit) {
    auto entries = SReflect::ConsoleList(&obj, "", {limit});
    return *std::find_if(
        entries.begin(), entries.end(), [](auto const& entry) { return entry._name == "name"; });
  };
  obj.name = "abc";
  for (size_t limit : {0, 1, 4, 5, 6}) {
    auto const entry = nameEntry(limit);
    EXPECT_EQ(entry._value, std::string{"\"abc\""}.substr(0, limit));
    EXPECT_EQ(entry._isValueTruncated, limit < 5);
  }
  // The listing cap falls inside a two-byte character, which must be removed in full.
  size_t const limit = SReflect::ConsoleListParams{}._maxValueLength;
  obj.name = std::string(limit - 2, 'x') + "\xC3\xA9\xC3\xA9";
  auto const entry = nameEntry(limit);
  EXPECT_EQ(entry._value, "\"" + std::string(limit - 2, 'x'));
  EXPECT_TRUE(entry._isValueTruncated);
  obj.name = "\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80";
  for (size_t cap : {2, 4, 8}) {
    auto const captured = nameEntry(cap);
    EXPECT_EQ(
        captured._value,
        cap == 2       ? "\""
            : cap == 4 ? "\"\xC3\xA9"
                       : "\"\xC3\xA9\xE4\xB8\xAD");
    EXPECT_TRUE(captured._isValueTruncated);
  }
}

TEST(SReflectConsole, ListTypeNames) {
  Machine obj;
  auto const entries = SReflect::ConsoleList(&obj, "");
  auto const factory = std::find_if(
      entries.begin(), entries.end(), [](auto const& entry) { return entry._name == "factory"; });
  ASSERT_NE(factory, entries.end());
  EXPECT_EQ(factory->_typeName, "Part");
  EXPECT_EQ(factory->_typeNameWithNamespace, "SReflectConsoleTest::Part");
  auto const methods = SReflect::ConsoleList(&obj, "base");
  for (auto const& entry : methods) {
    if (entry._name == "add") {
      EXPECT_EQ(entry._typeName, "(float amount, string note) -> float");
      EXPECT_EQ(entry._typeNameWithNamespace, "(float amount, std::string note) -> float");
    } else if (entry._name == "snapshot") {
      EXPECT_EQ(entry._typeName, "() -> NoDefault");
      EXPECT_EQ(entry._typeNameWithNamespace, "() -> SReflectConsoleTest::NoDefault");
    }
  }
}

TEST(SReflectConsole, ListEntriesJson) {
  using SReflect::ConsoleEntryKind;
  std::vector<SReflect::ConsoleListEntry> const entries{
      {ConsoleEntryKind::Directory, "base", "Part", "SReflectConsoleTest::Part", "", "", false},
      {ConsoleEntryKind::Field, "count", "int32", "int32", "3", "", true},
      {ConsoleEntryKind::Method, "clear", "() -> void", "() -> void", "", "", false},
      {ConsoleEntryKind::Field, "broken", "int32", "int32", "", "getter failed", false}};
  std::string const json =
      "[{\"coreType\":\"X_Invalid\",\"isValueTruncated\":false,\"kind\":\"Directory\",\"name\":\"base\",\"typeName\":\"Part\","
      "\"typeNameWithNamespace\":\"SReflectConsoleTest::Part\",\"value\":\"\",\"valueError\":\"\"},"
      "{\"coreType\":\"X_Invalid\",\"isValueTruncated\":true,\"kind\":\"Field\",\"name\":\"count\",\"typeName\":\"int32\","
      "\"typeNameWithNamespace\":\"int32\",\"value\":\"3\",\"valueError\":\"\"},"
      "{\"coreType\":\"X_Invalid\",\"isValueTruncated\":false,\"kind\":\"Method\",\"name\":\"clear\",\"typeName\":\"() -> void\","
      "\"typeNameWithNamespace\":\"() -> void\",\"value\":\"\",\"valueError\":\"\"},"
      "{\"coreType\":\"X_Invalid\",\"isValueTruncated\":false,\"kind\":\"Field\",\"name\":\"broken\",\"typeName\":\"int32\","
      "\"typeNameWithNamespace\":\"int32\",\"value\":\"\",\"valueError\":\"getter failed\"}]";
  EXPECT_EQ(SReflect::ToJsonString(entries, false), json);
  std::vector<SReflect::ConsoleListEntry> loaded;
  EXPECT_TRUE(SReflect::FromJsonString(loaded, json));
  EXPECT_EQ(SReflect::ToJsonString(loaded, false), json);
}

TEST(SReflectConsole, ListContainerMetadata) {
  Machine object;
  object.spare = Part{};
  auto const entries = SReflect::ConsoleList(&object, "");
  struct Case {
    char const* name;
    SReflect::CoreType type;
    std::optional<size_t> size;
  };
  Case const cases[] = {
      {"weights", SReflect::CoreType::CT_array, 2},
      {"named", SReflect::CoreType::CT_map, 2},
      {"spare", SReflect::CoreType::CT_struct, std::nullopt},
      {"slot", SReflect::CoreType::CT_struct, std::nullopt}};
  for (auto const& c : cases) {
    SCOPED_TRACE(c.name);
    auto const entry = std::find_if(
        entries.begin(), entries.end(), [&](auto const& entry) { return entry._name == c.name; });
    ASSERT_NE(entries.end(), entry);
    EXPECT_EQ(c.type, entry->_coreType);
    EXPECT_EQ(c.size, entry->_containerSize);
  }
  MatrixCases matrix;
  auto const matrices = SReflect::ConsoleList(&matrix, "", {1024, true, false});
  ASSERT_EQ(2, matrices.size());
  EXPECT_EQ(SReflect::CoreType::CT_matrix, matrices[0]._coreType);
  EXPECT_EQ(6, matrices[0]._containerSize);
  EXPECT_EQ(3, matrices[1]._containerSize);
}

TEST(SReflectConsole, ListNonDefaultMetadataJson) {
  SReflect::ConsoleListEntry entry;
  entry._kind = SReflect::ConsoleEntryKind::Directory;
  entry._name = "items";
  entry._coreType = SReflect::CoreType::CT_array;
  entry._containerSize = 7;
  SReflect::ConsoleListEntry restored;
  ASSERT_TRUE(SReflect::FromJsonString(restored, SReflect::ToJsonString(entry, false)));
  EXPECT_EQ(entry._coreType, restored._coreType);
  EXPECT_EQ(entry._containerSize, restored._containerSize);
}

TEST(SReflectConsole, ListSerializationErrorClearsContainerSize) {
  ArrayCases object;
  object.positionStorage[0] = std::numeric_limits<float>::quiet_NaN();
  auto const entries = SReflect::ConsoleList(&object, "");
  auto const entry = std::find_if(
      entries.begin(), entries.end(), [](auto const& entry) { return entry._name == "position"; });
  ASSERT_NE(entries.end(), entry);
  EXPECT_FALSE(entry->_valueError.empty());
  EXPECT_TRUE(entry->_value.empty());
  EXPECT_FALSE(entry->_containerSize.has_value());
}

using Completion = std::tuple<std::vector<std::string>, size_t, std::string, bool>;

template <class T>
static Completion
Complete(T const& obj, std::string_view command, SReflect::ConsoleAutoCompleteParams params = {}) {
  std::string error;
  auto result = SReflect::ConsoleAutoComplete(&obj, command, params, &error);
  EXPECT_EQ(error, "") << command;
  return {result._candidates, result._defaultIndex, result._note, result._isTruncated};
}

TEST(SReflectConsole, FindCompletion) {
  using SReflect::detail::ConsoleCompletionTarget;
  using Args = std::vector<std::string>;
  using Found = std::optional<std::tuple<ConsoleCompletionTarget, size_t, std::string, Args>>;
  auto const find = [](std::string_view input, size_t pathTokenIndex = 0) -> Found {
    auto const completion = SReflect::detail::FindConsoleCompletion(input, pathTokenIndex);
    if (!completion) {
      return std::nullopt;
    }
    return std::tuple{completion->target, completion->offset, completion->path, completion->args};
  };
  auto constexpr kStep = ConsoleCompletionTarget::PathStep;
  auto constexpr kArgument = ConsoleCompletionTarget::Argument;

  // The last step of the path
  EXPECT_EQ(find(""), Found(std::tuple(kStep, 0, "", Args{})));
  EXPECT_EQ(find("parts/1/ma"), Found(std::tuple(kStep, 8, "parts/1/", Args{})));
  EXPECT_EQ(find("parts/1/"), Found(std::tuple(kStep, 8, "parts/1/", Args{})));

  // A path typed with quotes or escapes cannot be completed.
  EXPECT_EQ(find("'named/\"a/"), std::nullopt);
  EXPECT_EQ(find("\"a b\"/c"), std::nullopt);

  // Arguments after the path, including one that has not started
  EXPECT_EQ(find("parts/0/add 1 "), Found(std::tuple(kArgument, 14, "parts/0/add", Args{"1"})));
  EXPECT_EQ(find("parts/0/add [1, "), Found(std::tuple(kArgument, 12, "parts/0/add", Args{})));

  // A command word before the path
  EXPECT_EQ(find("ls parts/", 1), Found(std::tuple(kStep, 9, "parts/", Args{})));
  EXPECT_EQ(find("ls ", 1), Found(std::tuple(kStep, 3, "", Args{})));
  EXPECT_EQ(find("ls", 1), std::nullopt);
}

TEST(SReflectConsole, CandidateOrder) {
  std::vector<std::string> names{"b", "10", "1a", "9", "_x", "A", "0"};
  SReflect::ConsoleAutoCompleteResult result;
  SReflect::detail::AddConsoleStepCandidates(result, names, "", {});
  EXPECT_EQ(result._candidates, (std::vector<std::string>{"0", "9", "10", "1a", "A", "_x", "b"}));
}

TEST(SReflectConsole, CompleteSteps) {
  Machine m;
  using Steps = std::vector<std::string>;
  // Members leave out the hidden "view", and the default is the first candidate.
  Steps const root{
      "base",
      "core",
      "count",
      "enabled",
      "factory",
      "kind",
      "name",
      "named",
      "parts",
      "ratio",
      "slot",
      "spare",
      "tables",
      "weights"};
  EXPECT_EQ(Complete(m, ""), Completion(root, 0, "", false));
  EXPECT_EQ(Complete(m, "na"), Completion(Steps{"name", "named"}, 0, "", false));
  EXPECT_EQ(Complete(m, "parts/"), Completion(Steps{}, 0, "", false));

  // Methods follow the path rules too, leaving out the hidden "debugDump".
  Steps const part{"add", "clear", "label", "limit", "mass", "scaled", "snapshot"};
  EXPECT_EQ(Complete(m, "parts/1/"), Completion(part, 0, "", false));
  EXPECT_EQ(Complete(m, "core/"), Completion(part, 0, "", false));

  // Keys that need quotes in a path, then in the terminal
  EXPECT_EQ(Complete(m, "named/"), Completion(Steps{"'\"..\"'", "'\"a/b\"'"}, 0, "", false));

  // Only directories, including an optional or variant that could hold one
  SReflect::ConsoleAutoCompleteParams directories;
  directories._directoriesOnly = true;
  EXPECT_EQ(
      Complete(m, "", directories),
      Completion(
          Steps{"base", "core", "factory", "named", "parts", "slot", "spare", "tables", "weights"},
          0,
          "",
          false));

  // Names are sorted alphabetically.
  Completions c;
  EXPECT_EQ(
      Complete(c, ""),
      Completion(
          Steps{
              "_underscore",
              "a",
              "b",
              "items",
              "keys",
              "limit",
              "nan",
              "noDefault",
              "quote",
              "setLevel",
              "setMode"},
          0,
          "",
          false));
  EXPECT_EQ(Complete(c, "keys/"), Completion(Steps{"9", "10", "1a"}, 0, "", false));

  // The limit applies after the typed text is matched.
  SReflect::ConsoleAutoCompleteParams three;
  three._maxCandidates = 3;
  EXPECT_EQ(Complete(c, "", three), Completion(Steps{"_underscore", "a", "b"}, 0, "", true));
  EXPECT_EQ(Complete(c, "keys/", three), Completion(Steps{"9", "10", "1a"}, 0, "", false));
  SReflect::ConsoleAutoCompleteParams two;
  two._maxCandidates = 2;
  EXPECT_EQ(Complete(c, "keys/", two), Completion(Steps{"9", "10"}, 0, "", true));
  EXPECT_EQ(Complete(c, "keys/1", two), Completion(Steps{"10", "1a"}, 0, "", false));
  SReflect::ConsoleAutoCompleteParams zero;
  zero._maxCandidates = 0;
  EXPECT_EQ(Complete(c, "keys/", zero), Completion(Steps{}, 0, "", true));

  // Array indices never complete, even in a large array or after typing a numeric prefix.
  c.items.resize(100000);
  for (std::string_view const command : {"items/", "items/0", "items/1", "items/01", "items/x"}) {
    EXPECT_EQ(Complete(c, command), Completion(Steps{}, 0, "", false)) << command;
  }

  // A value has no steps, and a bare "/" is not a path.
  EXPECT_EQ(Complete(m, "count/"), Completion(Steps{}, 0, "", false));
  std::string error;
  EXPECT_TRUE(SReflect::ConsoleAutoComplete(&m, "/", {}, &error)._candidates.empty());
  EXPECT_NE(error, "");
}

TEST(SReflectConsole, CompleteArguments) {
  Machine m;
  m.name = "a b";
  m.parts[0].limitStorage = -1; // Its getter throws
  Completions c;
  UnreadableProperty unreadable;
  using Values = std::vector<std::string>;
  Completion const none{Values{}, 0, "", false};
  std::tuple<std::string_view, Completion> const machineCases[] = {
      // A value completes to its current value, plus the names of its type's values.
      {"count ", {Values{"3"}, 0, "int32", false}},
      {"name ", {Values{"'a b'"}, 0, "string", false}},
      {"weights ", {Values{"[0.5, 1.5]"}, 0, "vector<float>", false}},
      {"weights/0 ", {Values{"0.5"}, 0, "float", false}},
      {"tables/0/x ", {Values{"1"}, 0, "int32", false}},
      {"enabled ", {Values{"false", "true"}, 1, "bool", false}},
      {"kind ", {Values{"Rigid", "Soft"}, 0, "Kind", false}},
      {"kind S", {Values{"Soft"}, 0, "Kind", false}},
      {"spare ", {Values{"null"}, 0, "optional<Part>", false}},
      // A getter that throws leaves only the note.
      {"parts/0/limit ", {Values{}, 0, "int32", false}},
      // A method argument completes to a default-constructed value of its type.
      {"base/add ", {Values{"0"}, 0, "float amount", false}},
      {"base/add 1 ", {Values{"''"}, 0, "string note", false}},
      // A const method can be called on a read-only path.
      {"factory/scaled ", {Values{"0"}, 0, "float factor", false}},
      // Nothing after the last argument, or for what cannot be set or called
      {"count 4 ", none},
      {"base/add 1 x ", none},
      {"base/clear ", none},
      {"factory ", none},
      {"factory/mass ", none},
      {"factory/add ", none},
      {"view/mass ", none},
  };
  for (auto const& [command, expected] : machineCases) {
    EXPECT_EQ(Complete(m, command), expected) << command;
  }

  std::tuple<std::string_view, Completion> const completionCases[] = {
      {"limit ", {Values{"5", "null"}, 0, "optional<int32>", false}},
      {"quote ", {Values{R"("it's")"}, 0, "string", false}},
      // A value that cannot be written as text leaves only the note.
      {"nan ", {Values{}, 0, "float", false}},
      // An enum without a name for zero defaults to its first alphabetical item.
      {"setLevel ", {Values{"High", "Low"}, 0, "Level level", false}},
      // A named zero value stays selected in an alphabetical list, until the prefix excludes it.
      {"setMode ", {Values{"Active", "Disabled"}, 1, "Mode mode", false}},
      {"setMode A", {Values{"Active"}, 0, "Mode mode", false}},
      // A method whose return type cannot be default constructed cannot be called.
      {"noDefault ", none},
  };
  for (auto const& [command, expected] : completionCases) {
    EXPECT_EQ(Complete(c, command), expected) << command;
  }
  EXPECT_EQ(Complete(unreadable, "enabled "), Completion(Values{}, 0, "bool", false));

  // Values with read-only members cannot be set as a whole.
  BulkValues bulk;
  EXPECT_EQ(Complete(bulk, "pairValue "), none);
  EXPECT_EQ(std::get<0>(Complete(bulk, "writablePair ")).size(), 1);

  // A value too long to edit leaves only the note.
  SReflect::ConsoleAutoCompleteParams shortValues;
  shortValues._maxValueLength = 9;
  EXPECT_EQ(Complete(m, "weights ", shortValues), Completion(Values{}, 0, "vector<float>", false));
  m.name = std::string(256, 'x');
  EXPECT_EQ(Complete(m, "name "), Completion(Values{m.name}, 0, "string", false));
  m.name += 'x';
  EXPECT_EQ(Complete(m, "name "), Completion(Values{}, 0, "string", false));
  c.items.resize(100000);
  EXPECT_EQ(Complete(c, "items "), Completion(Values{}, 0, "vector<int32>", false));
  shortValues._maxValueLength = 3;
  EXPECT_EQ(
      Complete(c, "setLevel ", shortValues), Completion(Values{"Low"}, 0, "Level level", false));
  shortValues._maxValueLength = 1;
  EXPECT_EQ(Complete(m, "base/add 1 ", shortValues), Completion(Values{}, 0, "string note", false));

  // Without current text, an optional or enum must not suggest a replacement value.
  c.limit = 12345;
  shortValues._maxValueLength = 4;
  EXPECT_EQ(Complete(c, "limit ", shortValues), Completion(Values{}, 0, "optional<int32>", false));
  c.limit.reset();
  EXPECT_EQ(
      Complete(c, "limit ", shortValues), Completion(Values{"null"}, 0, "optional<int32>", false));
  m.kind = static_cast<Kind>(99);
  EXPECT_EQ(Complete(m, "kind "), Completion(Values{}, 0, "Kind", false));

  // The default is kept within the limit.
  SReflect::ConsoleAutoCompleteParams one;
  one._maxCandidates = 1;
  EXPECT_EQ(Complete(m, "enabled ", one), Completion(Values{"true"}, 0, "bool", true));
}

TEST(SReflectConsole, CompleteListsLaterArguments) {
  Machine m;
  using Notes = std::vector<std::string>;
  auto const laterNotes = [&](std::string_view command) {
    return SReflect::ConsoleAutoComplete(&m, command)._laterArgumentNotes;
  };
  EXPECT_EQ(laterNotes("base/add "), Notes{"string note"});
  EXPECT_EQ(laterNotes("base/add 1 "), Notes{});
  EXPECT_EQ(laterNotes("count "), Notes{});
}

TEST(SReflectConsole, CompleteCandidateKinds) {
  using Kind = SReflect::ConsoleCandidateKind;
  using Kinds = std::vector<std::pair<std::string, Kind>>;
  auto const kinds = [](SReflect::ConsoleAutoCompleteResult const& result) {
    EXPECT_EQ(result._candidateKinds.size(), result._candidates.size());
    Kinds found;
    for (size_t i = 0; i < result._candidates.size() && i < result._candidateKinds.size(); ++i) {
      found.emplace_back(result._candidates[i], result._candidateKinds[i]);
    }
    return found;
  };
  Machine m;
  EXPECT_EQ(
      kinds(SReflect::ConsoleAutoComplete(&m, "co")),
      (Kinds{{"core", Kind::Directory}, {"count", Kind::Leaf}}));
  EXPECT_EQ(kinds(SReflect::ConsoleAutoComplete(&m, "base/a")), (Kinds{{"add", Kind::Leaf}}));
  EXPECT_EQ(
      kinds(SReflect::ConsoleAutoComplete(&m, "named/")),
      (Kinds{{"'\"..\"'", Kind::Directory}, {"'\"a/b\"'", Kind::Directory}}));

  // An optional or variant may hold a directory or a value.
  EXPECT_EQ(
      kinds(SReflect::ConsoleAutoComplete(&m, "s")),
      (Kinds{{"slot", Kind::DirectoryOrLeaf}, {"spare", Kind::DirectoryOrLeaf}}));

  // A fixed-size array of primitives is listed with its value, which can be set as a whole too.
  ArrayCases arrays;
  EXPECT_EQ(
      kinds(SReflect::ConsoleAutoComplete(&arrays, "")),
      (Kinds{
          {"count", Kind::Leaf},
          {"flags", Kind::Leaf},
          {"nested", Kind::DirectoryOrLeaf},
          {"parts", Kind::Directory},
          {"position", Kind::DirectoryOrLeaf}}));

  // A command that takes a directory completes only directories.
  SReflect::ConsoleAutoCompleteParams directories;
  directories._directoriesOnly = true;
  EXPECT_EQ(
      kinds(SReflect::ConsoleAutoComplete(&m, "s", directories)),
      (Kinds{{"slot", Kind::Directory}, {"spare", Kind::Directory}}));

  // Arguments end their token.
  EXPECT_EQ(
      kinds(SReflect::ConsoleAutoComplete(&m, "kind ")),
      (Kinds{{"Rigid", Kind::Leaf}, {"Soft", Kind::Leaf}}));
  EXPECT_EQ(kinds(SReflect::ConsoleAutoComplete(&m, "base/add ")), (Kinds{{"0", Kind::Leaf}}));

  // Steps named outside of reflection are directories, and keep the kinds already found.
  auto result = SReflect::ConsoleAutoComplete(&m, "na");
  SReflect::detail::AddConsoleStepCandidates(result, {"nav"}, "na", {});
  EXPECT_EQ(
      kinds(result),
      (Kinds{{"name", Kind::Leaf}, {"named", Kind::Directory}, {"nav", Kind::Directory}}));
}

TEST(SReflectConsole, CompleteFromPathStart) {
  Machine m;
  SReflect::detail::ConsoleCompletion completion;
  completion.path = "machine/parts/1/";
  SReflect::ConsoleAutoCompleteResult result;
  std::string error;
  EXPECT_TRUE(
      SReflect::detail::ConsoleAutoComplete(
          &m, SReflect::GetTypeInfo<Machine>(), completion, "ma", {}, result, &error, 8));
  EXPECT_EQ(result._candidates, std::vector<std::string>{"mass"});
}

TEST(SReflectConsole, AddStepCandidates) {
  using Kind = SReflect::ConsoleCandidateKind;
  SReflect::ConsoleAutoCompleteResult result;
  result._candidates = {"base", "count"};
  result._candidateKinds = {Kind::DirectoryOrLeaf, Kind::Leaf};
  result._defaultIndex = 1;
  SReflect::ConsoleAutoCompleteParams params;
  params._maxCandidates = 3;
  SReflect::detail::AddConsoleStepCandidates(result, {"a b", "box", "core"}, "b", params);
  EXPECT_EQ(result._candidates, (std::vector<std::string>{"base", "box"}));
  EXPECT_EQ(result._candidateKinds, (std::vector<Kind>{Kind::DirectoryOrLeaf, Kind::Directory}));
  EXPECT_EQ(result._defaultIndex, 0);
  EXPECT_FALSE(result._isTruncated);
  SReflect::detail::AddConsoleStepCandidates(result, {"bat", "bee"}, "", params);
  EXPECT_EQ(result._candidates, (std::vector<std::string>{"base", "bat", "bee"}));
  EXPECT_TRUE(result._isTruncated);
}

TEST(SReflectConsole, CompleteErrors) {
  Machine m;
  std::string error;
  auto const result = SReflect::ConsoleAutoComplete(&m, "nope/", {}, &error);
  EXPECT_TRUE(result._candidates.empty());
  EXPECT_EQ(error, "\"nope\" not found\n  | nope\n  | ^");
}

} // namespace SReflectConsoleTest
