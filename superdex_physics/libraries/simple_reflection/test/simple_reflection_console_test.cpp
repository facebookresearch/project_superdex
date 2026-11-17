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

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

static void ExpectTokens(
    std::string_view input,
    std::vector<std::string> const& expected,
    std::string_view expectedError = {}) {
  std::string error;
  EXPECT_EQ(SReflect::ConsoleTokenize(input, error), expected) << "input: " << input;
  EXPECT_EQ(error, expectedError) << "input: " << input;
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
  ExpectTokens(R"("a\qb")", {"aqb"}, R"(Unsupported escape sequence: "\q" (index 2))");

  // Only the first unsupported escape is reported.
  ExpectTokens(R"("\q\z")", {"qz"}, R"(Unsupported escape sequence: "\q" (index 1))");

  // An unterminated quoted segment is captured up to end of input.
  ExpectTokens(R"(connect "one two)", {"connect", "one two"});
  ExpectTokens(R"('unterminated single)", {"unterminated single"});

  // A backslash at the end of the input inside an unterminated double quote is dropped and
  // reported.
  ExpectTokens(R"("foo\)", {"foo"}, "Backslash at the end of the input (index 4)");

  // An error left over from an earlier call is cleared on success.
  std::string err = "stale";
  EXPECT_EQ(SReflect::ConsoleTokenize("a b", err), (std::vector<std::string>{"a", "b"}));
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
