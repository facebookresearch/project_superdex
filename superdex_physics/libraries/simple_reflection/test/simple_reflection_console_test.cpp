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

TEST(SReflectConsole, Tokenize) {
  auto testTokenize = [](std::string_view input,
                         std::vector<std::string> const& expected,
                         std::string_view expectedError = {}) {
    std::string err;
    auto tokens = SReflect::ConsoleTokenize(input, err);
    EXPECT_EQ(err, expectedError) << "input: " << input;
    EXPECT_EQ(tokens, expected) << "input: " << input;
  };

  // Empty and whitespace-only input yield no tokens.
  testTokenize("", {});
  testTokenize(" \t\n\r\f\v ", {});

  // Varying numbers of unquoted tokens, separated by assorted whitespace.
  testTokenize("hello", {"hello"});
  testTokenize("one two three", {"one", "two", "three"});
  testTokenize("  one \t two\nthree  ", {"one", "two", "three"});

  // Quotes group whitespace into a single token.
  testTokenize(R"(connect "one two three")", {"connect", "one two three"});
  testTokenize(R"('single quoted arg')", {"single quoted arg"});

  // Empty quotes give an empty token.
  testTokenize(R"(a '' "" b)", {"a", "", "", "b"});

  // Bytes above 0x7F never separate tokens, so UTF-8 text stays intact.
  testTokenize("voil\xC3\xA0 caf\xC3\xA9", {"voil\xC3\xA0", "caf\xC3\xA9"});

  // Adjacent quoted and unquoted segments form a single token.
  testTokenize(R"(foo"bar baz"qux)", {"foobar bazqux"});

  // Backslash is literal outside quotes, so Windows-style paths survive intact.
  testTokenize(R"(C:\foo\bar)", {R"(C:\foo\bar)"});
  testTokenize(R"(a\b c)", {R"(a\b)", "c"});

  // Escapes are resolved inside quotes (single or double).
  testTokenize(R"("a\"b")", {R"(a"b)"});
  testTokenize(R"("a\\b")", {R"(a\b)"});
  testTokenize(R"('a\'b')", {"a'b"});
  testTokenize(R"('a\\b')", {R"(a\b)"});

  // Backslash escape sequences are translated to their control characters inside quotes.
  testTokenize(R"("line1\nline2")", {"line1\nline2"});
  testTokenize(R"("col1\tcol2")", {"col1\tcol2"});
  testTokenize(R"("a\rb")", {"a\rb"});
  testTokenize(R"("a\ab")", {"a\ab"});
  testTokenize(R"("a\bb")", {"a\bb"});
  testTokenize(R"("a\fb")", {"a\fb"});
  testTokenize(R"("a\vb")", {"a\vb"});

  // An unsupported escape drops the backslash, emits the offending char literally, and reports
  // the first such occurrence via outError.
  testTokenize(R"("a\qb")", {"aqb"}, R"(Unsupported escape sequence: "\q" (index 2))");
  testTokenize(R"("a\0b")", {"a0b"}, R"(Unsupported escape sequence: "\0" (index 2))");

  // Only the first unsupported escape is reported.
  testTokenize(R"("\q\z")", {"qz"}, R"(Unsupported escape sequence: "\q" (index 1))");

  // An unterminated quoted segment is captured up to end of input.
  testTokenize(R"("abc)", {"abc"});
  testTokenize(R"(connect "one two)", {"connect", "one two"});
  testTokenize(R"('unterminated single)", {"unterminated single"});

  // A trailing backslash at end of input inside an unterminated quote is unsupported, so the
  // backslash is dropped and outError is set.
  testTokenize(R"("foo\)", {"foo"}, R"(Unsupported escape sequence: "\" (index 4))");

  // An error left over from an earlier call is cleared on success.
  std::string err = "stale";
  EXPECT_EQ(SReflect::ConsoleTokenize("a b", err), (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(err, "");
}
