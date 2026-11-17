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

#include "console_test_utils.h"

#include <mochi_debugger/lib/console_utils.h>

#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <string_view>

using namespace mochi;
using namespace mochi::dbg;
using SReflect::ConsoleEntryKind;
using SReflect::ConsoleListEntry;

TEST(ConsoleUtils, RemovesOnlyUnquotedWhitespace) {
  struct Case {
    std::string_view input;
    std::string_view expected;
  };
  Case const cases[] = {
      {" \t\r\n\v\f", ""},
      {" a \t b\n c ", "abc"},
      {" item  \"a \tb\" ", "item\"a \tb\""},
      {R"( x 'a b' "c d" )", R"(x'a b'"c d")"},
      {R"( x "a \" b" y )", R"(x"a \" b"y)"},
      {R"( x "a \\" y )", R"(x"a \\"y)"}};
  for (auto const& c : cases) {
    SCOPED_TRACE(c.input);
    EXPECT_EQ(c.expected, test::RemoveUnquotedWhitespace(c.input));
  }
}

TEST(ConsoleUtils, GroupsAndSortsFieldsAndMethods) {
  DynamicArray<ConsoleListEntry> const entries{
      {ConsoleEntryKind::Directory,
       "position",
       "array<float,3>",
       "std::array<float,3>",
       "[1,2,3]",
       "",
       false},
      {ConsoleEntryKind::Method, "reset", "() -> void", "() -> void", "", "", false},
      {ConsoleEntryKind::Directory, "actors", "", "", "", "", false},
      {ConsoleEntryKind::Field, "id", "string", "std::string", "\"a b\"", "", false},
      {ConsoleEntryKind::Method, "clear", "() -> void", "() -> void", "", "", false}};
  std::string const output = FormatConsoleList(entries);
  test::ExpectConsoleListEq(
      "actors\nid \"a b\"\nposition [1,2,3]\n\nclear() -> void\nreset() -> void", output);
  EXPECT_NE(std::string::npos, output.find("\n\n"));
  test::ExpectConsoleListEq(
      "position [1,2,3]\nactors\nid \"a b\"\n\nreset() -> void\nclear() -> void",
      FormatConsoleList(entries, {._sortAlphabetically = false}));
}

TEST(ConsoleUtils, ShowsOptionalTypesAndNamespaces) {
  DynamicArray<ConsoleListEntry> const entries{
      {ConsoleEntryKind::Directory,
       "items",
       "array<float,3>",
       "std::array<float,3>",
       "[1,2,3]",
       "",
       false},
      {ConsoleEntryKind::Method,
       "set",
       "(array<float,3> v) -> string",
       "(std::array<float,3> v) -> std::string",
       "",
       "",
       false}};
  struct Case {
    bool showTypes;
    bool includeNamespaces;
    std::string_view expected;
  };
  Case const cases[] = {
      {false, false, "items [1,2,3]\nset(array<float,3> v) -> string"},
      {true, false, "items array<float,3> [1,2,3]\nset(array<float,3> v) -> string"},
      {true, true, "items std::array<float,3> [1,2,3]\nset(std::array<float,3> v) -> std::string"}};
  for (auto const& c : cases) {
    ConsoleFormatParams params;
    params._maxNameColumnWidth = params._maxTypeColumnWidth = 0;
    params._showTypes = c.showTypes;
    params._includeNamespaces = c.includeNamespaces;
    test::ExpectConsoleListEq(c.expected, FormatConsoleList(entries, params));
  }
}

TEST(ConsoleUtils, BeautifiesValuesAndTypesWithoutChangingQuotedText) {
  DynamicArray<ConsoleListEntry> const entries{
      {ConsoleEntryKind::Directory,
       "items",
       "map<string,string>",
       "std::map<std::string,std::string>",
       R"({"key":"a,b:c \"x, y\"","values":[1,2]})",
       "",
       false}};
  std::string const output = FormatConsoleList(
      entries, {._maxTypeColumnWidth = 0, ._showTypes = true, ._includeNamespaces = true});
  EXPECT_NE(std::string::npos, output.find("std::map<std::string, std::string>"));
  EXPECT_NE(std::string::npos, output.find(R"({"key": "a,b:c \"x, y\"", "values": [1, 2]})"));
}

TEST(ConsoleUtils, ReportsValueTruncationAndAppliesIndependentWidthLimits) {
  DynamicArray<ConsoleListEntry> const entries{
      {ConsoleEntryKind::Field, "abcdef", "string", "std::string", "abcdef", "", true}};
  test::ExpectConsoleListEq("abcdef abcdef...", FormatConsoleList(entries));
  test::ExpectConsoleListEq(
      "ab... string abcdef...",
      FormatConsoleList(entries, {._maxNameColumnWidth = 5, ._showTypes = true}));
  test::ExpectConsoleListEq(
      "abcdef st... abcdef...",
      FormatConsoleList(entries, {._maxTypeColumnWidth = 5, ._showTypes = true}));
}

TEST(ConsoleUtils, AppliesMinimumWidthsAndSpacingWithoutTrailingPadding) {
  DynamicArray<ConsoleListEntry> const entries{
      {ConsoleEntryKind::Directory, "dir", "Short", "Short", "", "", false},
      {ConsoleEntryKind::Field,
       "fieldlong",
       "longTypeName",
       "longTypeName",
       "long value",
       "",
       false},
      {ConsoleEntryKind::Field, "a", "x", "x", "1", "", false}};
  ConsoleFormatParams params{
      ._minNameColumnWidth = 6,
      ._maxNameColumnWidth = 7,
      ._minTypeColumnWidth = 5,
      ._maxTypeColumnWidth = 6,
      ._columnSpacing = 3,
      ._showTypes = true};
  EXPECT_EQ(
      "a         x        1\ndir       Short\nfiel...   lon...   long value",
      FormatConsoleList(entries, params));
  DynamicArray<ConsoleListEntry> const single{entries.back()};
  EXPECT_EQ("a        x       1", FormatConsoleList(single, params));
  params._showTypes = false;
  EXPECT_EQ("a        1", FormatConsoleList(single, params));
  params._columnSpacing = 0;
  params._minNameColumnWidth = 0;
  EXPECT_EQ("a1", FormatConsoleList(single, params));
}

TEST(ConsoleUtils, AlignsUtf8AndTruncatesAtCharacterBoundaries) {
  DynamicArray<ConsoleListEntry> entries{
      {ConsoleEntryKind::Field, "caf\xC3\xA9", "int32", "int32", "1", "", false},
      {ConsoleEntryKind::Field, "B", "int32", "int32", "2", "", false}};
  ConsoleFormatParams params{
      ._minNameColumnWidth = 0, ._maxNameColumnWidth = 0, ._columnSpacing = 2};
  EXPECT_EQ("B     2\ncaf\xC3\xA9  1", FormatConsoleList(entries, params));
  entries = {
      {ConsoleEntryKind::Field,
       "a",
       "string",
       "std::string",
       "\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80"
       "abcd",
       "",
       false}};
  struct Case {
    size_t width;
    std::string_view expected;
  };
  Case const cases[] = {
      {1, "."},
      {2, ".."},
      {3, "..."},
      {7, "a  \xC3\xA9..."},
      {8, "a  \xC3\xA9\xE4\xB8\xAD..."},
      {9, "a  \xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80..."}};
  for (auto const& c : cases) {
    SCOPED_TRACE(c.width);
    params._maxLineWidth = c.width;
    test::ExpectConsoleListEq(c.expected, FormatConsoleList(entries, params));
  }
}

TEST(ConsoleUtils, HandlesEmptyAndMetadataOnlyEntries) {
  EXPECT_EQ("", FormatConsoleList({}));
  DynamicArray<ConsoleListEntry> const entries{
      {ConsoleEntryKind::Directory,
       "position",
       "array<float,3>",
       "std::array<float,3>",
       "",
       "",
       false}};
  EXPECT_EQ("position", FormatConsoleList(entries));
  test::ExpectConsoleListEq(
      "position array<float,3>", FormatConsoleList(entries, {._showTypes = true}));
  DynamicArray<ConsoleListEntry> const truncated{
      {ConsoleEntryKind::Directory,
       "position",
       "array<float,3>",
       "std::array<float,3>",
       "",
       "",
       true}};
  test::ExpectConsoleListEq("position ...", FormatConsoleList(truncated));
  DynamicArray<ConsoleListEntry> const methods{
      {ConsoleEntryKind::Method, "reset", "() -> void", "() -> void", "", "", false}};
  EXPECT_EQ("reset() -> void", FormatConsoleList(methods));
}

TEST(ConsoleUtils, ShowsEntryErrorsOnOneLine) {
  DynamicArray<ConsoleListEntry> const entries{
      {ConsoleEntryKind::Field, "ok", "int32", "int32", "3", "", false},
      {ConsoleEntryKind::Field,
       "broken",
       "string",
       "std::string",
       "",
       "line one\nline two",
       false}};
  std::string const output = FormatConsoleList(entries);
  test::ExpectConsoleListEq("broken <error: line one line two>\nok 3", output);
  EXPECT_EQ(1, std::count(output.begin(), output.end(), '\n'));
}
