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

#include <mochi_renderer/path.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <unordered_map>

namespace {

TEST(MochiPathTest, EmptyDefaultConstructed) {
  mochi::Path p;
  EXPECT_TRUE(p.IsEmpty());
  EXPECT_TRUE(p.ToString().empty());
}

TEST(MochiPathTest, EmptyFromEmptyString) {
  mochi::Path p{std::string{}};
  EXPECT_TRUE(p.IsEmpty());
}

TEST(MochiPathTest, ConstructFromCStr) {
  mochi::Path p{"foo.txt"};
  EXPECT_FALSE(p.IsEmpty());
}

TEST(MochiPathTest, ConstructFromStringView) {
  std::string_view sv{"foo.txt"};
  mochi::Path p{sv};
  EXPECT_FALSE(p.IsEmpty());
}

TEST(MochiPathTest, ConstructFromFilesystemPath) {
  std::filesystem::path fsp{"foo.txt"};
  mochi::Path p{fsp};
  EXPECT_FALSE(p.IsEmpty());
}

TEST(MochiPathTest, ComparesCaseInsensitively) {
  // Mixed-case input compares equal to its lowercased form (the comparable
  // form is internal — we observe its effect through `==`).
  mochi::Path mixed{"/Foo/Bar.txt"};
  mochi::Path lower{"/foo/bar.txt"};
  EXPECT_EQ(mixed, lower);
}

TEST(MochiPathTest, EqualityCaseInsensitive) {
  mochi::Path a{"/Foo/Bar.txt"};
  mochi::Path b{"/foo/bar.txt"};
  EXPECT_EQ(a, b);
  EXPECT_FALSE(a != b);
}

TEST(MochiPathTest, EqualitySeparatorInsensitive) {
  // Constructing via std::filesystem::path on POSIX collapses backslashes
  // differently. Test the canonical generic-string equivalence.
  mochi::Path a{"/foo/bar/baz"};
  mochi::Path b{"/foo//bar/./baz"};
  EXPECT_EQ(a, b);
}

TEST(MochiPathTest, TrailingSlashCanonicalized) {
  // A trailing separator must not affect the canonical form: a dialog-provided
  // path like "/foo/bar/" must compare equal (and hash equal) to "/foo/bar".
  mochi::Path withSlash{"/foo/bar/"};
  mochi::Path noSlash{"/foo/bar"};
  EXPECT_EQ(withSlash, noSlash);
  EXPECT_EQ(std::hash<mochi::Path>{}(withSlash), std::hash<mochi::Path>{}(noSlash));
  // The stored string itself should not retain the trailing slash.
  EXPECT_EQ(withSlash.ToString().back(), 'r');
}

TEST(MochiPathTest, TrailingSlashParentIsDescendant) {
  // Regression: a parent with a trailing slash must still recognize its
  // children (the asset browser opens workspace roots with a trailing slash
  // while subfolders come from directory iteration without one).
  mochi::Path parent{"/foo/bar/"};
  mochi::Path child{"/foo/bar/baz"};
  EXPECT_TRUE(child.IsDescendantOf(parent));
  EXPECT_TRUE(parent.IsDescendantOf(parent));
  EXPECT_EQ(child.RelativeToParent(parent), "baz");
}

TEST(MochiPathTest, HashMatchesEquality) {
  mochi::Path a{"/Foo/Bar.txt"};
  mochi::Path b{"/foo/bar.txt"};
  EXPECT_EQ(std::hash<mochi::Path>{}(a), std::hash<mochi::Path>{}(b));
}

TEST(MochiPathTest, StorablePreservesOriginalCase) {
  mochi::Path p{"/Foo/Bar.txt"};
  // The storable form is the absolute path. It should contain "Foo" / "Bar" verbatim.
  auto const s = p.ToString();
  EXPECT_NE(s.find("Foo"), std::string::npos);
  EXPECT_NE(s.find("Bar"), std::string::npos);
}

TEST(MochiPathTest, IsDescendantOfBasic) {
  mochi::Path parent{"/foo/bar"};
  mochi::Path child{"/foo/bar/baz.txt"};
  mochi::Path same{"/foo/bar"};
  mochi::Path sibling{"/foo/qux"};
  EXPECT_TRUE(child.IsDescendantOf(parent));
  EXPECT_TRUE(same.IsDescendantOf(parent));
  EXPECT_FALSE(sibling.IsDescendantOf(parent));
}

TEST(MochiPathTest, IsDescendantOfBoundary) {
  // "/foo/bar" must NOT be considered a descendant of "/foo/ba"
  // (string-prefix-only would be a bug here).
  mochi::Path parent{"/foo/ba"};
  mochi::Path notChild{"/foo/bar"};
  EXPECT_FALSE(notChild.IsDescendantOf(parent));
}

TEST(MochiPathTest, RelativeTo) {
  mochi::Path base{"/foo/bar"};
  mochi::Path child{"/foo/bar/baz/qux.txt"};
  EXPECT_EQ(child.RelativeToParent(base), "baz/qux.txt");
}

TEST(MochiPathTest, RelativeToSelfIsEmpty) {
  mochi::Path p{"/foo/bar"};
  EXPECT_TRUE(p.RelativeToParent(p).empty());
}

TEST(MochiPathTest, RelativeToNonDescendantIsEmpty) {
  mochi::Path base{"/foo/bar"};
  mochi::Path sibling{"/foo/qux/baz.txt"};
  EXPECT_TRUE(sibling.RelativeToParent(base).empty());
}

TEST(MochiPathTest, OperatorSlash) {
  mochi::Path a{"/foo"};
  auto combined = a / "bar/baz.txt";
  EXPECT_TRUE(combined.IsDescendantOf(a));
}

TEST(MochiPathTest, ParentAndFilename) {
  mochi::Path p{"/foo/bar/baz.txt"};
  EXPECT_EQ(p.GetFilename(), "baz.txt");
  EXPECT_FALSE(p.GetParentPath().IsEmpty());
}

TEST(MochiPathTest, Extension) {
  // Extension preserves original case (it is a string fragment, not a `Path`).
  mochi::Path p{"/foo/bar/baz.TXT"};
  EXPECT_EQ(p.GetExtension(), ".TXT");
}

TEST(MochiPathTest, ExtensionLowercase) {
  mochi::Path p{"/foo/bar/baz.TXT"};
  EXPECT_EQ(p.GetExtensionLowercase(), ".txt");
}

TEST(MochiPathTest, ReplaceExtension) {
  mochi::Path p{"/foo/bar/baz.txt"};
  p.ReplaceExtension(".png");
  EXPECT_EQ(p.GetExtension(), ".png");
}

TEST(MochiPathTest, UsableInOrderedMap) {
  std::map<mochi::Path, int> m;
  m[mochi::Path{"/Foo/A"}] = 1;
  m[mochi::Path{"/foo/a"}] = 2; // overwrites — same key after normalization
  EXPECT_EQ(m.size(), 1u);
  EXPECT_EQ(m[mochi::Path{"/FOO/A"}], 2);
}

TEST(MochiPathTest, UsableInUnorderedMap) {
  std::unordered_map<mochi::Path, int> m;
  m[mochi::Path{"/Foo/A"}] = 1;
  m[mochi::Path{"/foo/a"}] = 2;
  EXPECT_EQ(m.size(), 1u);
}

TEST(MochiPathTest, UsableInSet) {
  std::set<mochi::Path> s;
  s.insert(mochi::Path{"/Foo/A"});
  s.insert(mochi::Path{"/foo/a"});
  EXPECT_EQ(s.size(), 1u);
}

} // namespace
