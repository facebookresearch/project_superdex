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

#include <gtest/gtest.h>
#include <mochi_debugger/lib/terminal_text.h>

#include <string>
#include <vector>

using namespace mochi::dbg;

TEST(TerminalText, MeasuresCellsAndKeepsUtf8Boundaries) {
  EXPECT_EQ(4, CountTerminalCells("\xE4\xB8\xAD\xE6\x96\x87"));
  EXPECT_EQ(1, CountTerminalCells("e\xCC\x81"));
  EXPECT_EQ(2, CountTerminalCells("\xF0\x9F\x98\x80"));
  EXPECT_EQ(0, TerminalPrefixBytes("\xE4\xB8\xAD", 1));
  EXPECT_EQ(3, TerminalPrefixBytes("\xE4\xB8\xAD", 2));
  EXPECT_EQ(3, TerminalPrefixBytes("e\xCC\x81", 1));
}

TEST(TerminalText, KeepsTheCursorVisibleForUnicodePrompts) {
  std::string const text = "/caf\xC3\xA9> abcdef";
  for (size_t columns : {size_t{1}, size_t{5}, size_t{9}, size_t{80}}) {
    for (size_t cursor = 7; cursor <= text.size(); ++cursor) {
      auto const view = MakeTerminalTextView(text, cursor, columns);
      EXPECT_LT(view.cursorColumn, columns);
      EXPECT_LE(CountTerminalCells(view.text), columns == 1 ? 1 : columns - 1);
      if (!view.text.empty()) {
        EXPECT_NE(0x80, static_cast<unsigned char>(view.text.front()) & 0xC0);
      }
    }
  }
}

TEST(TerminalText, FitsSuggestions) {
  using Annotations = std::vector<std::string>;
  Annotations const none;
  Annotations const kind{"(+1)", "<Kind>"};
  Annotations const arguments{"<bool enable>", "<float speed>"};
  Annotations const real3{"<Real3>"};
  // Spaces separate the preview and annotations. Without a preview, a space separates the first
  // annotation from the input, unless the input ends with one.
  EXPECT_EQ("enes (+1) <Kind>", FitSuggestion("kind Soft", "enes", kind, 80));
  EXPECT_EQ("<bool enable> <float speed>", FitSuggestion("enable ", "", arguments, 27));
  EXPECT_EQ(" (+1) <Kind>", FitSuggestion("kind Soft", "", kind, 80));
  EXPECT_EQ("(+1) <Kind>", FitSuggestion("", "", kind, 80));
  // Text that does not fit is cut to end in "...", using all the cells.
  EXPECT_EQ("<bool enable> <float sp...", FitSuggestion("enable ", "", arguments, 26));
  EXPECT_EQ("enes (+1) <K...", FitSuggestion("kind Soft", "enes", kind, 15));
  EXPECT_EQ("<bool ena...", FitSuggestion("enable ", "", arguments, 12));
  EXPECT_EQ("<...", FitSuggestion("enable ", "", arguments, 4));
  EXPECT_EQ("", FitSuggestion("enable ", "", arguments, 3));
  EXPECT_EQ("[0, ...", FitSuggestion("gravity ", "[0, -9.8, 0]", real3, 7));
  EXPECT_EQ("", FitSuggestion("gravity ", "[0, -9.8, 0]", none, 3));
  EXPECT_EQ("\xE4\xB8\xAD...", FitSuggestion("", "\xE4\xB8\xAD\xE6\x96\x87\xE4\xB8\xAD", none, 5));
}
