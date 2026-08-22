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

#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/string_utils.h>

#include <utility>

using namespace mochi;

TEST(StringUtils, Format) {
  EXPECT_STREQ("", Format().c_str());
  EXPECT_STREQ("", Format("").c_str());
  EXPECT_STREQ("word", Format("word").c_str());
  EXPECT_STREQ("Cool 1", Format("Cool %d", 1).c_str());
  EXPECT_STREQ("Cool 1!", Format("Cool %d%s", 1, "!").c_str());
}

TEST(StringUtils, Join) {
  auto testJoin = [](DynamicArray<std::string_view> const& input,
                     std::string_view separator,
                     std::string_view expectedOutput) {
    std::string output = Join(input, separator);
    EXPECT_STREQ(output.c_str(), std::string{expectedOutput}.c_str());

    // Now test the Span<std::string const> overload
    DynamicArray<std::string> inputCopy(input.begin(), input.end());
    output = Join(inputCopy, separator);
    EXPECT_STREQ(output.c_str(), std::string{expectedOutput}.c_str());
  };

  testJoin({}, "", "");
  testJoin({}, "xxx", "");
  testJoin({""}, "", "");
  testJoin({""}, "xxx", "");
  testJoin({"", ""}, "xxx", "xxx");
  testJoin({"Hello"}, "", "Hello");
  testJoin({"Hello"}, "xxx", "Hello");
  testJoin({"Hello", "World"}, "", "HelloWorld");
  testJoin({"Hello", "World"}, "xxx", "HelloxxxWorld");
  testJoin({"One", "Two", "Three"}, ",", "One,Two,Three");
  testJoin({"One,", "Two,", "Three,"}, ",", "One,,Two,,Three,");
}

TEST(StringUtils, Split) {
  FiloAllocator alloc;
  auto testSplit = [&alloc](
                       std::string_view input,
                       std::string_view separators,
                       bool keepEmptyTokens,
                       DynamicArray<std::pair<int, int>> const& expectedOffsets) {
    auto result = Split(input, separators, keepEmptyTokens, &alloc);
    EXPECT_EQ(result.get_allocator(), &alloc);
    EXPECT_EQ(expectedOffsets.size(), result.size());
    size_t i = 0;
    for (auto const& [expectedBegin, expectedEnd] : expectedOffsets) {
      auto tok = result[i++];
      EXPECT_EQ(input.data() + expectedBegin, tok.data());
      EXPECT_EQ(input.data() + expectedEnd, tok.data() + tok.length());
    }
  };

  testSplit("", "", true, {});
  testSplit("", "", false, {});
  testSplit("Hello", "", true, {{0, 5}});
  testSplit("Hello", "", false, {{0, 5}});
  testSplit("Hello", ",", true, {{0, 5}});
  testSplit("Hello", ",", false, {{0, 5}});
  testSplit(",,Hello,World,,", ",", true, {{0, 0}, {1, 1}, {2, 7}, {8, 13}, {14, 14}, {15, 15}});
  testSplit(",,Hello,World,,", ",", false, {{2, 7}, {8, 13}});
  testSplit("$#Hello@World@#", "@#$", true, {{0, 0}, {1, 1}, {2, 7}, {8, 13}, {14, 14}, {15, 15}});
  testSplit("$#Hello@World@#", "@#$", false, {{2, 7}, {8, 13}});

  // Test with default arguments
  {
    DynamicArray<std::string_view> result = Split(",Hello,World,", ",");
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result.get_allocator(), GetDefaultAllocator());
    EXPECT_STREQ("Hello", std::string{result[0]}.c_str());
    EXPECT_STREQ("World", std::string{result[1]}.c_str());
  }
}

TEST(StringUtils, ConsoleTokenize) {
  auto testTokenize = [](std::string_view input,
                         DynamicArray<std::string> const& expected,
                         std::string_view expectedError = {}) {
    std::string err;
    auto tokens = ConsoleTokenize(input, err);
    EXPECT_EQ(err, expectedError) << "input: " << input;
    ASSERT_EQ(tokens.size(), expected.size()) << "input: " << input;
    for (size_t i = 0; i < tokens.size(); ++i) {
      EXPECT_EQ(tokens[i], expected[i]) << "input: " << input << " index: " << i;
    }
  };

  // Whitespace-only input yields no tokens.
  testTokenize(" \t\n\r\f\v ", {});

  // Varying numbers of unquoted tokens, separated by assorted whitespace.
  testTokenize("hello", {"hello"});
  testTokenize("one two three", {"one", "two", "three"});
  testTokenize("  one \t two\nthree  ", {"one", "two", "three"});

  // Quotes group whitespace into a single token.
  testTokenize(R"(connect "one two three")", {"connect", "one two three"});
  testTokenize(R"('single quoted arg')", {"single quoted arg"});

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

  // A trailing backslash at end of input inside an unterminated quote is unsupported; the
  // backslash is dropped and outError is set.
  testTokenize(R"("foo\)", {"foo"}, R"(Unsupported escape sequence: "\" (index 4))");
}

TEST(StringUtils, EndsWithCaseInsensitive) {
  // Exact match
  EXPECT_TRUE(EndsWithCaseInsensitive("model.obj", ".obj"));
  EXPECT_TRUE(EndsWithCaseInsensitive("model.OBJ", ".obj"));
  EXPECT_TRUE(EndsWithCaseInsensitive("model.obj", ".OBJ"));
  EXPECT_TRUE(EndsWithCaseInsensitive("model.ObJ", ".oBj"));

  // Non-matching suffix
  EXPECT_FALSE(EndsWithCaseInsensitive("model.obj", ".stl"));
  EXPECT_FALSE(EndsWithCaseInsensitive("model.obj", ".ob"));

  // Suffix longer than string
  EXPECT_FALSE(EndsWithCaseInsensitive(".obj", "model.obj"));

  // Empty cases
  EXPECT_TRUE(EndsWithCaseInsensitive("anything", ""));
  EXPECT_TRUE(EndsWithCaseInsensitive("", ""));
  EXPECT_FALSE(EndsWithCaseInsensitive("", "x"));

  // Entire string matches suffix
  EXPECT_TRUE(EndsWithCaseInsensitive(".obj", ".OBJ"));
}

TEST(StringUtils, Trim) {
  auto testTrim = [](std::string_view input, size_t expectedBegin, size_t expectedEnd) {
    std::string_view result = Trim(input);
    EXPECT_EQ(input.data() + expectedBegin, result.data());
    EXPECT_EQ(input.data() + expectedEnd, result.data() + result.length());
  };

  testTrim("", 0, 0);
  testTrim(" \t\n\r\f\v", 0, 0);
  testTrim("Hello World! ", 0, 12);
  testTrim(" Hello World!", 1, 13);
  testTrim(" Hello World! ", 1, 13);
  testTrim(" \t\n\r\f\vHello \t\n\r\f\vWorld! \t\n\r\f\v", 6, 23);
}
