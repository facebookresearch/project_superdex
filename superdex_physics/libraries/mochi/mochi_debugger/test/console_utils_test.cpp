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
#include <optional>
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
      "clear () -> void\nreset () -> void\n\nactors\nid \"a b\"\nposition [1,2,3]", output);
  EXPECT_NE(std::string::npos, output.find("\n\n"));
  test::ExpectConsoleListEq(
      "reset () -> void\nclear () -> void\n\nposition [1,2,3]\nactors\nid \"a b\"",
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
      {false, false, "set (array<float,3> v) -> string\n\nitems [1,2,3]"},
      {true, false, "set (array<float,3> v) -> string\n\nitems array<float,3> [1,2,3]"},
      {true,
       true,
       "set (std::array<float,3> v) -> std::string\n\nitems std::array<float,3> [1,2,3]"}};
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
       "\xC3\xA9"
       "abcdef",
       "",
       false}};
  struct Case {
    size_t width;
    std::string_view expected;
  };
  Case const cases[] = {
      {1, "a"},
      {2, "a"},
      {3, "a"},
      {4, "a  ."},
      {5, "a  .."},
      {6, "a  ..."},
      {7, "a  \xC3\xA9..."},
      {8,
       "a  \xC3\xA9"
       "a..."},
      {9,
       "a  \xC3\xA9"
       "ab..."},
      {10,
       "a  \xC3\xA9"
       "abcdef"}};
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
  test::ExpectConsoleListEq("reset () -> void", FormatConsoleList(methods));
}

TEST(ConsoleUtils, OmitsEntryErrors) {
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
  test::ExpectConsoleListEq("broken\nok 3", output);
  EXPECT_EQ(1, std::count(output.begin(), output.end(), '\n'));
}

TEST(ConsoleUtils, SimplifiesAdapterTypeNames) {
  struct Case {
    std::string_view input;
    std::string_view expected;
  };
  Case const cases[] = {
      {"std::vector<std::array<FooReflectionData,2>>", "std::vector<std::array<Foo,2>>"},
      {"ns::FooReflectionData", "ns::Foo"},
      {"FooReflectionDataReflectionData", "FooReflectionData"},
      {"ReflectionData", "ReflectionData"},
      {"ns::ReflectionData", "ns::ReflectionData"},
      {"FooReflectionDataOther", "FooReflectionDataOther"}};
  for (auto const& c : cases) {
    SCOPED_TRACE(c.input);
    DynamicArray<ConsoleListEntry> const entries{
        {._kind = ConsoleEntryKind::Field,
         ._name = "field",
         ._typeName = std::string{c.input},
         ._value = "7"}};
    test::ExpectConsoleListEq(
        "field " + std::string{c.expected} + " 7",
        FormatConsoleList(entries, {._showTypes = true}));
  }
}

TEST(ConsoleUtils, SimplifiesSignaturesWithoutChangingArgumentNames) {
  DynamicArray<ConsoleListEntry> const entries{
      {._kind = ConsoleEntryKind::Method,
       ._name = "call",
       ._typeName = "(ActorHandle inputReflectionData, FooReflectionData resultReflectionData)"
                    " -> ns::BarReflectionData"}};
  test::ExpectConsoleListEq(
      "call (ActorHandle inputReflectionData, Foo resultReflectionData) -> ns::Bar",
      FormatConsoleList(entries));
}

TEST(ConsoleUtils, AlignsMethodNamesWithoutWideningPropertyTypeColumns) {
  DynamicArray<ConsoleListEntry> const entries{
      {._kind = ConsoleEntryKind::Method,
       ._name = "reset",
       ._typeName = "(VeryLongArgumentType argument) -> AnotherLongReturnType"},
      {._kind = ConsoleEntryKind::Field, ._name = "x", ._typeName = "int", ._value = "7"}};
  EXPECT_EQ(
      "reset  (VeryLongArgumentType argument) -> AnotherLongReturnType\n\nx      int  7",
      FormatConsoleList(
          entries,
          {._minNameColumnWidth = 0,
           ._minTypeColumnWidth = 0,
           ._maxTypeColumnWidth = 3,
           ._columnSpacing = 2,
           ._showTypes = true}));
}

TEST(ConsoleUtils, LimitsValuesWithoutTruncatingMethodSignatures) {
  DynamicArray<ConsoleListEntry> const entries{
      {._kind = ConsoleEntryKind::Method,
       ._name = "doThing",
       ._typeName = "(LongArgumentType argument) -> LongReturnType"},
      {._kind = ConsoleEntryKind::Field, ._name = "name", ._value = "abcdefgh"}};
  struct Case {
    size_t width;
    std::string_view value;
  };
  Case const cases[] = {
      {3, ""}, {10, "."}, {11, ".."}, {12, "..."}, {14, "ab..."}, {0, "abcdefgh"}};
  for (auto const& c : cases) {
    SCOPED_TRACE(c.width);
    test::ExpectConsoleListEq(
        "doThing (LongArgumentType argument) -> LongReturnType\n\nname " + std::string{c.value},
        FormatConsoleList(
            entries, {._minNameColumnWidth = 0, ._maxLineWidth = c.width, ._columnSpacing = 2}));
  }
}

TEST(ConsoleUtils, ShowsSizeOnlyWhenNoValueWasCaptured) {
  struct Case {
    std::string_view value;
    std::string_view error;
    bool truncated;
    std::optional<size_t> size;
    std::string_view expected;
  };
  Case const cases[] = {
      {"", "", false, 0, "(size=0)"},
      {"", "", false, 9999, "(size=9999)"},
      {"[]", "", false, 0, "[]"},
      {"null", "", false, std::nullopt, "null"},
      {"", "", false, std::nullopt, ""},
      {"abc", "", true, 9999, "abc..."},
      {"stale", "failed", false, 5, ""}};
  for (auto const& c : cases) {
    SCOPED_TRACE(c.expected);
    DynamicArray<ConsoleListEntry> const entries{
        {._kind = ConsoleEntryKind::Directory,
         ._name = "data",
         ._value = std::string{c.value},
         ._valueError = std::string{c.error},
         ._isValueTruncated = c.truncated,
         ._containerSize = c.size}};
    test::ExpectConsoleListEq("data " + std::string{c.expected}, FormatConsoleList(entries));
  }
}

TEST(ConsoleUtils, FitsWideCharacterValuesToTerminalCells) {
  DynamicArray<ConsoleListEntry> const entries{
      {._kind = ConsoleEntryKind::Field, ._name = "a", ._value = "\"\xE4\xB8\xAD\xE6\x96\x87\""}};
  EXPECT_EQ(
      "a" + std::string(35, ' ') + "\"...", FormatConsoleList(entries, {._maxLineWidth = 40}));
}
