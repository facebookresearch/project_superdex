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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/span.h>
#include <superdex_robotics/utils/file_utils.h>

/* Expectations are built with NormalizeBotPath, the same normalization path resolution itself
 * applies, rather than with std::filesystem::weakly_canonical. Bot paths are deliberately compared
 * as written and symlinks are not resolved, so canonicalizing an expectation disagrees wherever the
 * temp directory sits behind a link -- on macOS /var is a symlink to /private/var, which is exactly
 * where this showed up. The assertions still pin which directory a path resolves to; only the
 * spelling convention is shared with the code under test. */

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

namespace fs = std::filesystem;

class PathResolutionTest : public testing::Test {
 protected:
  void SetUp() override {
    // ResolveBotPath / ParseRootFile call MOCHI_LOG_ERROR before returning an
    // Error. The test framework's default log callback fails the test on any
    // error-channel log. Suppress for the duration of the test; negative-path
    // tests still verify the error via EXPECT_NOT_OK.
    _prevLogFn = GetLogCallback();
    SetLogCallback([](LogChannel, char const*, char const*, int) {});
  }

  void TearDown() override {
    SetLogCallback(_prevLogFn);
  }

  // Create an empty .superdex_root marker in `dir`.
  void WriteEmptyRootMarker(fs::path const& dir) {
    mochi::WriteFile(dir / kRootMarkerFile, mochi::Span<char const>{}, ExpectOK{});
  }

  // Write a .superdex_root with the given JSON contents in `dir`.
  void WriteRootFile(fs::path const& dir, std::string_view contents) {
    mochi::WriteFile(dir / kRootMarkerFile, contents, ExpectOK{});
  }

  // Touch a file (create empty) at `path`, creating parents as needed.
  void TouchFile(fs::path const& path) {
    mochi::WriteFile(path, mochi::Span<char const>{}, ExpectOK{});
  }

  // Temp directory with automatic cleanup.
  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("path_resolution_test", ExpectOK{});
  fs::path _tempDir = _tempDirCleanup.Path();
  LogFn _prevLogFn{};
};

} // namespace

// ----------------------------------------------------------------------------
// ParseRootFile
// ----------------------------------------------------------------------------

// File structure (under _tempDir):
//   empty/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ParseRootFile_EmptyFileIsValid) {
  auto const dir = _tempDir / "empty";
  WriteEmptyRootMarker(dir);
  auto const parsed = ParseRootFile(dir / kRootMarkerFile, ExpectOK{});
  EXPECT_TRUE(parsed.tags.empty());
  EXPECT_EQ(parsed.rootDir, dir);
}

// File structure (under _tempDir):
//   ws/
//     .superdex_root  (whitespace only)
TEST_F(PathResolutionTest, ParseRootFile_WhitespaceOnlyIsValid) {
  auto const dir = _tempDir / "ws";
  WriteRootFile(dir, "   \n\t\r\n  ");
  auto const parsed = ParseRootFile(dir / kRootMarkerFile, ExpectOK{});
  EXPECT_TRUE(parsed.tags.empty());
}

/* Reaching a root through a symlink must not silently relocate what it resolves to.
 *
 * Bot paths are compared as written and symlinks are deliberately not followed, so a path reached
 * via a link stays on the link's side rather than jumping to the target. This reproduces on any
 * platform the shape that macOS produces for free, where the temp directory lives under /var, a
 * symlink to /private/var: expectations built with weakly_canonical used to pass on Linux and fail
 * there. Skipped where the platform will not create symlinks.
 */
TEST_F(PathResolutionTest, ParseRootFile_ThroughSymlinkKeepsTheSpellingUsed) {
  auto const real = _tempDir / "real_root";
  WriteRootFile(real, R"({"@bots": "."})");

  std::error_code ec;
  fs::create_directory_symlink(real, _tempDir / "link_root", ec);
  if (ec) {
    GTEST_SKIP() << "Platform cannot create symlinks: " << ec.message();
  }
  auto const viaLink = _tempDir / "link_root";

  auto const parsed = ParseRootFile(viaLink / kRootMarkerFile, ExpectOK{});
  ASSERT_EQ(parsed.tags.size(), 1u);
  // The tag resolves on the side it was reached from, not the link's target.
  EXPECT_EQ(parsed.tags.at("@bots"), NormalizeBotPath(viaLink));
  EXPECT_NE(parsed.tags.at("@bots"), NormalizeBotPath(real));
}

// File structure (under _tempDir):
//   single/
//     .superdex_root  ({"@bots": "."})
TEST_F(PathResolutionTest, ParseRootFile_SingleTag) {
  auto const dir = _tempDir / "single";
  WriteRootFile(dir, R"({"@bots": "."})");
  auto const parsed = ParseRootFile(dir / kRootMarkerFile, ExpectOK{});
  ASSERT_EQ(parsed.tags.size(), 1u);
  EXPECT_EQ(parsed.tags.at("@bots"), NormalizeBotPath(dir));
}

// File structure (under _tempDir):
//   r1/
//     .superdex_root  ({"@self": ".", "@other": "../r2"})
//   r2/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ParseRootFile_MultipleTagsRelativeAndCrossRoot) {
  auto const r1 = _tempDir / "r1";
  auto const r2 = _tempDir / "r2";
  WriteEmptyRootMarker(r2);
  WriteRootFile(r1, R"({"@self": ".", "@other": "../r2"})");
  auto const parsed = ParseRootFile(r1 / kRootMarkerFile, ExpectOK{});
  ASSERT_EQ(parsed.tags.size(), 2u);
  EXPECT_EQ(parsed.tags.at("@self"), NormalizeBotPath(r1));
  EXPECT_EQ(parsed.tags.at("@other"), NormalizeBotPath(r2));
}

// File structure (under _tempDir):
//   bad/
//     .superdex_root  ({not json)
TEST_F(PathResolutionTest, ParseRootFile_MalformedJsonFails) {
  auto const dir = _tempDir / "bad";
  WriteRootFile(dir, R"({not json)");
  Error error;
  (void)ParseRootFile(dir / kRootMarkerFile, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   arr/
//     .superdex_root  (["not", "an", "object"])
TEST_F(PathResolutionTest, ParseRootFile_NonObjectRootFails) {
  auto const dir = _tempDir / "arr";
  WriteRootFile(dir, R"(["not", "an", "object"])");
  Error error;
  (void)ParseRootFile(dir / kRootMarkerFile, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   nonstr/
//     .superdex_root  ({"@bots": 42})
TEST_F(PathResolutionTest, ParseRootFile_NonStringValueFails) {
  auto const dir = _tempDir / "nonstr";
  WriteRootFile(dir, R"({"@bots": 42})");
  Error error;
  (void)ParseRootFile(dir / kRootMarkerFile, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   noat/
//     .superdex_root  ({"bots": "."})  // key missing leading '@'
TEST_F(PathResolutionTest, ParseRootFile_InvalidKeyMissingAtFails) {
  auto const dir = _tempDir / "noat";
  WriteRootFile(dir, R"({"bots": "."})");
  Error error;
  (void)ParseRootFile(dir / kRootMarkerFile, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   badchar/
//     .superdex_root  ({"@bo-ts": "."})  // '-' is not a valid identifier char
TEST_F(PathResolutionTest, ParseRootFile_InvalidKeyBadCharFails) {
  auto const dir = _tempDir / "badchar";
  WriteRootFile(dir, R"({"@bo-ts": "."})");
  Error error;
  (void)ParseRootFile(dir / kRootMarkerFile, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   missingmarker/
//     .superdex_root  ({"@x": "../no_marker_here"})
//   no_marker_here/    (no .superdex_root marker)
TEST_F(PathResolutionTest, ParseRootFile_TagWithoutMarkerWarnsButSucceeds) {
  auto const dir = _tempDir / "missingmarker";
  fs::create_directories(_tempDir / "no_marker_here");
  WriteRootFile(dir, R"({"@x": "../no_marker_here"})");
  auto const parsed = ParseRootFile(dir / kRootMarkerFile, ExpectOK{});
  ASSERT_EQ(parsed.tags.size(), 1u);
  EXPECT_EQ(parsed.tags.at("@x"), NormalizeBotPath(_tempDir / "no_marker_here"));
}

// File structure (under _tempDir):
//   mixed/
//     .superdex_root  ({"@good": ".", "@bad": "../no_marker_here"})
//   no_marker_here/    (no marker)
TEST_F(PathResolutionTest, ParseRootFile_OneBadTagWarnsButStoresBoth) {
  auto const dir = _tempDir / "mixed";
  fs::create_directories(_tempDir / "no_marker_here");
  WriteRootFile(dir, R"({"@good": ".", "@bad": "../no_marker_here"})");
  auto const parsed = ParseRootFile(dir / kRootMarkerFile, ExpectOK{});
  ASSERT_EQ(parsed.tags.size(), 2u);
  EXPECT_EQ(parsed.tags.at("@good"), NormalizeBotPath(dir));
  EXPECT_EQ(parsed.tags.at("@bad"), NormalizeBotPath(_tempDir / "no_marker_here"));
}

// File structure (under _tempDir):
//   r1/
//     .superdex_root  ({"@bots": "../r2"})
//   r2/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ParseRootFile_TagValueMayUseDotDot) {
  // Tag *values* are exempt from the .. restriction — that's the whole point of cross-root tags.
  auto const r1 = _tempDir / "r1";
  auto const r2 = _tempDir / "r2";
  WriteEmptyRootMarker(r2);
  WriteRootFile(r1, R"({"@bots": "../r2"})");
  auto const parsed = ParseRootFile(r1 / kRootMarkerFile, ExpectOK{});
  EXPECT_EQ(parsed.tags.at("@bots"), NormalizeBotPath(r2));
}

// ----------------------------------------------------------------------------
// FindBotsRoot
// ----------------------------------------------------------------------------

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     a/
//       b/
//         bot.superdex_bot  (empty)
TEST_F(PathResolutionTest, FindBotsRoot_LocatesNearestMarker) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const sub = root / "a" / "b";
  fs::create_directories(sub);
  TouchFile(sub / "bot.superdex_bot");
  auto const found = FindBotsRoot(sub / "bot.superdex_bot");
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(*found, NormalizeBotPath(root));
}

// File structure (under _tempDir):
//   nomarker/
//     deep/    (no .superdex_root anywhere in or above _tempDir)
TEST_F(PathResolutionTest, FindBotsRoot_NoMarkerReturnsNullopt) {
  auto const sub = _tempDir / "nomarker" / "deep";
  fs::create_directories(sub);
  // _tempDir is under temp_directory_path, which won't contain a .superdex_root.
  // Note: this test is technically environment-dependent, but in practice the system
  // temp dir doesn't have a marker file walking all the way up.
  auto const found = FindBotsRoot(sub);
  // We can't strictly assert nullopt because some unusual CI environment might place a
  // marker upstream; instead assert that if found, it's not within _tempDir.
  if (found.has_value()) {
    EXPECT_FALSE(
        NormalizeBotPath(*found).string().starts_with(NormalizeBotPath(_tempDir).string()));
  }
}

// ----------------------------------------------------------------------------
// ResolveBotPath
// ----------------------------------------------------------------------------

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_EmptyInput) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const result =
      ResolveBotPath("", root / "x.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_TRUE(result.empty());
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     thing.glb  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_AbsoluteInputFails) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const target = root / "thing.glb";
  TouchFile(target);
  auto const absStr = target.generic_string();
  Error error;
  (void)ResolveBotPath(absStr, root / "bot.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     sub/
TEST_F(PathResolutionTest, ResolveBotPath_RelativeJoinsWithBaseParent) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const sub = root / "sub";
  fs::create_directories(sub);
  auto const result =
      ResolveBotPath("foo.glb", sub / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(result, NormalizeBotPath(sub / "foo.glb"));
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     deep/
//       deeper/
TEST_F(PathResolutionTest, ResolveBotPath_RootPrefixResolvesToRoot) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root); // empty root file is OK for // resolution
  auto const sub = root / "deep" / "deeper";
  fs::create_directories(sub);
  auto const result =
      ResolveBotPath("//foo/bar.glb", sub / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(result, NormalizeBotPath(root / "foo" / "bar.glb"));
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  ({"@bots": "."})
//     deep/
TEST_F(PathResolutionTest, ResolveBotPath_TaggedPathResolves) {
  auto const root = _tempDir / "root";
  WriteRootFile(root, R"({"@bots": "."})");
  auto const sub = root / "deep";
  fs::create_directories(sub);
  auto const result = ResolveBotPath(
      "@bots/x.superdex_bot", sub / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(result, NormalizeBotPath(root / "x.superdex_bot"));
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  ({"@bots": "."})
TEST_F(PathResolutionTest, ResolveBotPath_UnknownTagFails) {
  auto const root = _tempDir / "root";
  WriteRootFile(root, R"({"@bots": "."})");
  Error error;
  (void)ResolveBotPath("@nope/x", root / "ref.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   no_root_dir/    (no .superdex_root anywhere)
TEST_F(PathResolutionTest, ResolveBotPath_TaggedPathWithNoRootFails) {
  auto const sub = _tempDir / "no_root_dir";
  fs::create_directories(sub);
  // Don't create any .superdex_root anywhere under _tempDir.
  Error error;
  (void)ResolveBotPath("@bots/x", sub / "ref.superdex_bot", kBotPathMaxParentDepth, error);
  // Whether the surrounding filesystem has a root upstream is environment-dependent;
  // on a clean temp dir without an upstream marker, this must error.
  // If the environment has an upstream marker, the lookup of @bots will still fail
  // because the upstream tag table almost certainly doesn't define @bots.
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_RejectsParentDirComponentRelative) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  Error error;
  (void)ResolveBotPath("../foo.glb", root / "bot.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_RejectsParentDirComponentMidPath) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  Error error;
  (void)ResolveBotPath("foo/../bar.glb", root / "bot.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_RejectsTrailingParentDirComponent) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  Error error;
  (void)ResolveBotPath("foo/bar/..", root / "bot.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_RejectsParentDirWithBackslashSeparator) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  Error error;
  (void)ResolveBotPath(
      "foo\\..\\bar.glb", root / "bot.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_RejectsParentDirInRootPrefix) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  Error error;
  (void)ResolveBotPath("//../foo", root / "bot.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  ({"@bots": "."})
TEST_F(PathResolutionTest, ResolveBotPath_RejectsParentDirInTaggedPath) {
  auto const root = _tempDir / "root";
  WriteRootFile(root, R"({"@bots": "."})");
  Error error;
  (void)ResolveBotPath("@bots/../foo", root / "bot.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_AllowsDotDotAsFilenameSubstring) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  // ..foo and foo..bar.glb are not standalone ".." components -> should be accepted.
  auto const result1 =
      ResolveBotPath("..foo", root / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(result1, NormalizeBotPath(root / "..foo"));
  auto const result2 =
      ResolveBotPath("foo..bar.glb", root / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(result2, NormalizeBotPath(root / "foo..bar.glb"));
}

// File structure (under _tempDir):
//   R1/
//     .superdex_root  ({"@bots": "../R2"})
//   R2/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_TagDefinitionMayUseDotDotEvenThoughInputCannot) {
  // Mirrors the spec: tag *definitions* may use "..", input paths may not.
  auto const r1 = _tempDir / "R1";
  auto const r2 = _tempDir / "R2";
  WriteEmptyRootMarker(r2);
  WriteRootFile(r1, R"({"@bots": "../R2"})");
  auto const result =
      ResolveBotPath("@bots/x", r1 / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(result, NormalizeBotPath(r2 / "x"));
}

// File structure (under _tempDir):
//   R1/
//     .superdex_root  ({"@bots": "../R2"})
//   R2/
//     .superdex_root  ({"@bots": "."})
TEST_F(PathResolutionTest, ResolveBotPath_EvaluatesIndependentlyPerBasePath) {
  // Two roots: R1 has @bots -> ../R2; R2 has @bots -> ".".
  // Same input "@bots/x" must resolve differently depending on the basePath used.
  auto const r1 = _tempDir / "R1";
  auto const r2 = _tempDir / "R2";
  WriteEmptyRootMarker(r2); // create R2 first so R1's tag can validate against it
  WriteRootFile(r2, R"({"@bots": "."})");
  WriteRootFile(r1, R"({"@bots": "../R2"})");

  auto const fromR1 =
      ResolveBotPath("@bots/x", r1 / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(fromR1, NormalizeBotPath(r2 / "x"));

  auto const fromR2 =
      ResolveBotPath("@bots/x", r2 / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(fromR2, NormalizeBotPath(r2 / "x"));
}

// ----------------------------------------------------------------------------
// Multi-root scenarios
// ----------------------------------------------------------------------------

// File structure (under _tempDir):
//   root/
//     .superdex_root  ({"@self": "."})
//     deep/
TEST_F(PathResolutionTest, ResolveBotPath_SelfReferentialTagResolvesToOwnRoot) {
  // A root that tags itself via "." should resolve "@self/x" to its own directory.
  auto const root = _tempDir / "root";
  WriteRootFile(root, R"({"@self": "."})");
  auto const sub = root / "deep";
  fs::create_directories(sub);
  auto const result =
      ResolveBotPath("@self/x", sub / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(result, NormalizeBotPath(root / "x"));
}

// File structure (under _tempDir):
//   R1/
//     .superdex_root  ({"@mid": "../R2"})
//   R2/
//     .superdex_root  ({"@end": "../R3"})
//   R3/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_ChainedTagsAreNotFollowed) {
  // R1 tags @mid -> R2; R2 tags @end -> R3.
  // Resolving "@mid/x" from R1 must produce R2/x — NOT follow into R3 via @end.
  auto const r1 = _tempDir / "R1";
  auto const r2 = _tempDir / "R2";
  auto const r3 = _tempDir / "R3";
  WriteEmptyRootMarker(r3);
  WriteRootFile(r2, R"({"@end": "../R3"})");
  WriteRootFile(r1, R"({"@mid": "../R2"})");

  auto const result =
      ResolveBotPath("@mid/x", r1 / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(result, NormalizeBotPath(r2 / "x"));

  // And @end is not visible from R1 — only R2's table defines it.
  Error error;
  (void)ResolveBotPath("@end/x", r1 / "ref.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   outer/
//     .superdex_root  ({"@here": "."})
//     inner/
//       .superdex_root  ({"@here": "."})
TEST_F(PathResolutionTest, ResolveBotPath_NestedRootsResolveToNearest) {
  // outer/.superdex_root and outer/inner/.superdex_root both exist.
  // // and @ resolutions from inside `inner` must use `inner`'s root.
  auto const outer = _tempDir / "outer";
  auto const inner = outer / "inner";
  WriteRootFile(outer, R"({"@here": "."})");
  WriteRootFile(inner, R"({"@here": "."})");

  auto const fromInner =
      ResolveBotPath("//x.glb", inner / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(fromInner, NormalizeBotPath(inner / "x.glb"));

  auto const fromOuter =
      ResolveBotPath("//x.glb", outer / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(fromOuter, NormalizeBotPath(outer / "x.glb"));
}

// File structure (under _tempDir):
//   outer/
//     .superdex_root  ({"@bots": "."})
//     inner/
//       .superdex_root  ({"@bots": "."})
TEST_F(PathResolutionTest, ResolveBotPath_NestedRootTagShadowsOuter) {
  // Both outer and inner roots define @bots, pointing to themselves.
  // Resolution from `inner` must use inner's @bots; resolution from `outer` must use outer's.
  auto const outer = _tempDir / "outer";
  auto const inner = outer / "inner";
  WriteRootFile(outer, R"({"@bots": "."})");
  WriteRootFile(inner, R"({"@bots": "."})");

  auto const fromInner =
      ResolveBotPath("@bots/x", inner / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(fromInner, NormalizeBotPath(inner / "x"));

  auto const fromOuter =
      ResolveBotPath("@bots/x", outer / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(fromOuter, NormalizeBotPath(outer / "x"));
}

// File structure (under _tempDir):
//   R1/
//     .superdex_root  ({"@bots": "../R2"})
//   R2/
//     .superdex_root  ({"@bots": "../R3"})
//   R3/
//     .superdex_root  (empty)
TEST_F(PathResolutionTest, ResolveBotPath_TaggedTargetTagTableIsNotConsulted) {
  // R1 tags @bots -> R2. R2 also defines @bots -> R3 (a *different* directory).
  // Resolving "@bots/x" from R1 must use R1's @bots (-> R2/x), not chase R2's @bots into R3.
  auto const r1 = _tempDir / "R1";
  auto const r2 = _tempDir / "R2";
  auto const r3 = _tempDir / "R3";
  WriteEmptyRootMarker(r3);
  WriteRootFile(r2, R"({"@bots": "../R3"})");
  WriteRootFile(r1, R"({"@bots": "../R2"})");

  auto const result =
      ResolveBotPath("@bots/x", r1 / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(result, NormalizeBotPath(r2 / "x"));
  // Sanity: starting from R2, @bots resolves into R3.
  auto const fromR2 =
      ResolveBotPath("@bots/x", r2 / "ref.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(fromR2, NormalizeBotPath(r3 / "x"));
}

// ----------------------------------------------------------------------------
// UnresolveBotPath
// ----------------------------------------------------------------------------

TEST_F(PathResolutionTest, UnresolveBotPath_EmptyInput) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const result =
      UnresolveBotPath({}, root / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_TRUE(result.empty());
}

TEST_F(PathResolutionTest, UnresolveBotPath_AlreadyRelativeIsPassedThrough) {
  // Defensive: an already-relative input has no canonical absolute form to derive.
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const result = UnresolveBotPath(
      fs::path("foo.glb"), root / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(std::string(result), "foo.glb");
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     sub/
//       bot.superdex_bot   (referencing file)
//       _render/foo.glb (descendant of basePath dir)
TEST_F(PathResolutionTest, UnresolveBotPath_DescendantOfBaseDirEmitsPlainRelative) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const sub = root / "sub";
  fs::create_directories(sub / "render");
  auto const target = sub / "render" / "foo.glb";
  TouchFile(target);
  auto const result =
      UnresolveBotPath(target, sub / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(std::string(result), "render/foo.glb");
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     other/asset.glb       (under root, NOT under sub/)
//     sub/bot.superdex_bot
TEST_F(PathResolutionTest, UnresolveBotPath_UnderRootNotUnderBaseDirEmitsRootPrefix) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const sub = root / "sub";
  fs::create_directories(sub);
  auto const target = root / "other" / "asset.glb";
  TouchFile(target);
  auto const result =
      UnresolveBotPath(target, sub / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(std::string(result), "//other/asset.glb");
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  ({"@here": "."})
//     sub/bot.superdex_bot
//     thing.glb
TEST_F(PathResolutionTest, UnresolveBotPath_RootPrefixWinsOverSelfTag) {
  // A self-tag like {"@here": "."} means @here resolves to root.
  // For a path under root but not under sub/, // and @here both apply.
  // // is preferred because it doesn't depend on a tag table.
  auto const root = _tempDir / "root";
  WriteRootFile(root, R"({"@here": "."})");
  auto const sub = root / "sub";
  fs::create_directories(sub);
  auto const target = root / "thing.glb";
  TouchFile(target);
  auto const result =
      UnresolveBotPath(target, sub / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(std::string(result), "//thing.glb");
}

// File structure (under _tempDir):
//   R1/
//     .superdex_root  ({"@other": "../R2"})
//     bot.superdex_bot
//   R2/
//     .superdex_root  (empty)
//     foo.glb
TEST_F(PathResolutionTest, UnresolveBotPath_TaggedWhenOutsideBotsRoot) {
  auto const r1 = _tempDir / "R1";
  auto const r2 = _tempDir / "R2";
  WriteEmptyRootMarker(r2);
  WriteRootFile(r1, R"({"@other": "../R2"})");
  auto const target = r2 / "foo.glb";
  TouchFile(target);
  auto const result =
      UnresolveBotPath(target, r1 / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(std::string(result), "@other/foo.glb");
}

// File structure (under _tempDir):
//   R1/
//     .superdex_root  ({"@a": "../R2", "@b": "../R2/inner"})
//     bot.superdex_bot
//   R2/
//     .superdex_root  (empty)
//     inner/
//       .superdex_root (empty)
//       deep/foo.glb
TEST_F(PathResolutionTest, UnresolveBotPath_LongestTagWins) {
  auto const r1 = _tempDir / "R1";
  auto const r2 = _tempDir / "R2";
  auto const inner = r2 / "inner";
  WriteEmptyRootMarker(r2);
  WriteEmptyRootMarker(inner);
  WriteRootFile(r1, R"({"@a": "../R2", "@b": "../R2/inner"})");
  auto const target = inner / "deep" / "foo.glb";
  TouchFile(target);
  auto const result =
      UnresolveBotPath(target, r1 / "bot.superdex_bot", kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(std::string(result), "@b/deep/foo.glb");
}

// File structure (under _tempDir):
//   root/
//     .superdex_root (empty)
//     bot.superdex_bot
//   outside/asset.glb        (sibling of `root`, not reachable via root or tag)
TEST_F(PathResolutionTest, UnresolveBotPath_OutsideEverythingFails) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const target = _tempDir / "outside" / "asset.glb";
  TouchFile(target);
  Error error;
  (void)UnresolveBotPath(target, root / "bot.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

// ----------------------------------------------------------------------------
// Round-trip: UnresolveBotPath(ResolveBotPath(s)) == s
// ----------------------------------------------------------------------------

// File structure (under _tempDir):
//   root/
//     .superdex_root ({"@assets": "."})
//     other/asset.glb
//     sub/
//       bot.superdex_bot
//       _render/local.glb
TEST_F(PathResolutionTest, RoundTrip_AllSupportedInputForms) {
  auto const root = _tempDir / "root";
  WriteRootFile(root, R"({"@assets": "."})");
  auto const sub = root / "sub";
  auto const base = sub / "bot.superdex_bot";
  fs::create_directories(sub / "render");
  TouchFile(base);
  TouchFile(sub / "render" / "local.glb");
  TouchFile(root / "other" / "asset.glb");

  auto const fileRel = "render/local.glb";
  auto const rootRel = "//other/asset.glb";
  auto const tagRel = "@assets/other/asset.glb";

  // file-relative round-trip
  {
    auto const abs = ResolveBotPath(fileRel, base, kBotPathMaxParentDepth, ExpectOK{});
    auto const back = UnresolveBotPath(abs, base, kBotPathMaxParentDepth, ExpectOK{});
    EXPECT_EQ(std::string(back), fileRel);
  }
  // // round-trip — re-derives // since the asset is under root but not under sub/
  {
    auto const abs = ResolveBotPath(rootRel, base, kBotPathMaxParentDepth, ExpectOK{});
    auto const back = UnresolveBotPath(abs, base, kBotPathMaxParentDepth, ExpectOK{});
    EXPECT_EQ(std::string(back), rootRel);
  }
  // @tag round-trip — for an asset inside the bots root, // wins over @tag,
  // so the canonical form for "@assets/other/asset.glb" round-trips as "//other/asset.glb".
  // ResolveBotPath accepts both; UnresolveBotPath returns the // form per priority.
  {
    auto const abs = ResolveBotPath(tagRel, base, kBotPathMaxParentDepth, ExpectOK{});
    auto const back = UnresolveBotPath(abs, base, kBotPathMaxParentDepth, ExpectOK{});
    EXPECT_EQ(std::string(back), "//other/asset.glb");
    // But the result is itself a valid ResolveBotPath input that resolves to the same place.
    auto const reabs = ResolveBotPath(back.c_str(), base, kBotPathMaxParentDepth, ExpectOK{});
    EXPECT_EQ(reabs, abs);
  }
}

// File structure (under _tempDir):
//   R1/
//     .superdex_root ({"@other": "../R2"})
//     bot.superdex_bot
//   R2/
//     .superdex_root (empty)
//     foo.glb
TEST_F(PathResolutionTest, RoundTrip_TagPreservedWhenOutsideBotsRoot) {
  auto const r1 = _tempDir / "R1";
  auto const r2 = _tempDir / "R2";
  WriteEmptyRootMarker(r2);
  WriteRootFile(r1, R"({"@other": "../R2"})");
  auto const base = r1 / "bot.superdex_bot";
  TouchFile(base);
  TouchFile(r2 / "foo.glb");

  // The tag is the *only* form that can refer to a path outside the basePath's bots root.
  std::string const original = "@other/foo.glb";
  auto const abs = ResolveBotPath(original, base, kBotPathMaxParentDepth, ExpectOK{});
  auto const back = UnresolveBotPath(abs, base, kBotPathMaxParentDepth, ExpectOK{});
  EXPECT_EQ(std::string(back), original);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root (empty)
//     sub/
//       bot.superdex_bot
//       render/m.glb
//       collision/m.h5
//     other/extra.glb
TEST_F(PathResolutionTest, RoundTrip_BotPrefab_ViaMakePathsAbsoluteAndRelative) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const sub = root / "sub";
  auto const botFile = sub / "bot.superdex_bot";
  fs::create_directories(sub / "render");
  fs::create_directories(sub / "collision");
  fs::create_directories(root / "other");
  TouchFile(botFile);
  TouchFile(sub / "render" / "m.glb");
  TouchFile(sub / "collision" / "m.h5");
  TouchFile(root / "other" / "extra.glb");

  BotPrefab prefab;
  prefab.links.resize(2);
  prefab.links[0].renderModelFile = DynamicString{"render/m.glb"};
  prefab.links[0].shapeFile = DynamicString{"collision/m.h5"};
  prefab.links[1].renderModelFile = DynamicString{"//other/extra.glb"};
  prefab.links[1].shapeFile = DynamicString{};

  // Snapshot original strings.
  std::string const r0 = std::string(prefab.links[0].renderModelFile);
  std::string const s0 = std::string(prefab.links[0].shapeFile);
  std::string const r1 = std::string(prefab.links[1].renderModelFile);

  MakePathsAbsolute(prefab, botFile, ExpectOK{});
  // After absolutizing, all paths must be absolute.
  EXPECT_TRUE(fs::path(prefab.links[0].renderModelFile.c_str()).is_absolute());
  EXPECT_TRUE(fs::path(prefab.links[0].shapeFile.c_str()).is_absolute());
  EXPECT_TRUE(fs::path(prefab.links[1].renderModelFile.c_str()).is_absolute());
  EXPECT_TRUE(prefab.links[1].shapeFile.empty());

  MakePathsRelative(prefab, botFile, ExpectOK{});
  EXPECT_EQ(std::string(prefab.links[0].renderModelFile), r0);
  EXPECT_EQ(std::string(prefab.links[0].shapeFile), s0);
  EXPECT_EQ(std::string(prefab.links[1].renderModelFile), r1);
  EXPECT_TRUE(prefab.links[1].shapeFile.empty());
}

// File structure (under _tempDir):
//   root/
//     .superdex_root (empty)
//     sub/
//       bot.superdex_bot
//       base.superdex_bot
//       render/m.glb
//     other/attached.superdex_bot
TEST_F(PathResolutionTest, RoundTrip_ModBotPrefab_ViaMakePathsAbsoluteAndRelative) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const sub = root / "sub";
  auto const botFile = sub / "bot.superdex_bot";
  fs::create_directories(sub / "render");
  fs::create_directories(root / "other");
  TouchFile(botFile);
  TouchFile(sub / "base.superdex_bot");
  TouchFile(sub / "render" / "m.glb");
  TouchFile(root / "other" / "attached.superdex_bot");

  ModBotPrefab bp;
  bp.base = DynamicString{"base.superdex_bot"};
  bp.modifications.resize(2);
  AttachBot ab;
  ab.path = DynamicString{"//other/attached.superdex_bot"};
  bp.modifications[0] = ab;
  AttachLink al;
  al.link.renderModelFile = DynamicString{"render/m.glb"};
  bp.modifications[1] = al;

  std::string const baseStr = std::string(bp.base);
  std::string const attachPath = std::string(std::get<AttachBot>(bp.modifications[0]).path);
  std::string const linkRender =
      std::string(std::get<AttachLink>(bp.modifications[1]).link.renderModelFile);

  MakePathsAbsolute(bp, botFile, ExpectOK{});
  EXPECT_TRUE(fs::path(bp.base.c_str()).is_absolute());
  EXPECT_TRUE(fs::path(std::get<AttachBot>(bp.modifications[0]).path.c_str()).is_absolute());
  EXPECT_TRUE(
      fs::path(std::get<AttachLink>(bp.modifications[1]).link.renderModelFile.c_str())
          .is_absolute());

  MakePathsRelative(bp, botFile, ExpectOK{});
  EXPECT_EQ(std::string(bp.base), baseStr);
  EXPECT_EQ(std::string(std::get<AttachBot>(bp.modifications[0]).path), attachPath);
  EXPECT_EQ(
      std::string(std::get<AttachLink>(bp.modifications[1]).link.renderModelFile), linkRender);
}

// ----------------------------------------------------------------------------
// MakeParamsPath{Absolute,Relative}: a `params` field is path-or-inline-JSON
// ----------------------------------------------------------------------------

// Inline JSON must survive resolution untouched in both directions. Regression test for a
// controller/sensor `params` holding inline JSON being rewritten into "<dir>/{...}", which then
// takes the file branch and fails to load once the scene is read back from disk.
TEST_F(PathResolutionTest, MakeParamsPath_InlineJsonIsLeftUntouched) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const base = root / "scene.mochi_bot_scene";

  std::string const objectJson = R"({"kp": 1.0, "kd": 0.1})";
  std::string const arrayJson = R"([1, 2, 3])";
  std::string const whitespaceLeadingJson = "  \n{\"a\": 1}";

  for (std::string const& original : {objectJson, arrayJson, whitespaceLeadingJson}) {
    DynamicString params{original.c_str()};
    MakeParamsPathAbsolute(params, base, ExpectOK{});
    EXPECT_EQ(std::string(params), original);
    MakeParamsPathRelative(params, base, ExpectOK{});
    EXPECT_EQ(std::string(params), original);
  }
}

// A file-path `params` must still resolve to absolute and round-trip back to its relative form,
// exactly like other path fields.
TEST_F(PathResolutionTest, MakeParamsPath_FilePathRoundTrips) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const sub = root / "sub";
  auto const base = sub / "scene.mochi_bot_scene";
  fs::create_directories(sub / "params");
  TouchFile(base);
  TouchFile(sub / "params" / "ctrl.superdex_sensor");

  std::string const original = "params/ctrl.superdex_sensor";
  DynamicString params{original.c_str()};

  MakeParamsPathAbsolute(params, base, ExpectOK{});
  EXPECT_TRUE(fs::path(params.c_str()).is_absolute());

  MakeParamsPathRelative(params, base, ExpectOK{});
  EXPECT_EQ(std::string(params), original);
}

// Empty `params` is a no-op: not inline JSON, and MakePath* skips empty input.
TEST_F(PathResolutionTest, MakeParamsPath_EmptyIsNoOp) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const base = root / "scene.mochi_bot_scene";
  DynamicString params{};
  MakeParamsPathAbsolute(params, base, ExpectOK{});
  EXPECT_TRUE(params.empty());
  MakeParamsPathRelative(params, base, ExpectOK{});
  EXPECT_TRUE(params.empty());
}

// ----------------------------------------------------------------------------
// maxParentDepth: file-relative paths may ascend up to N parents (0 = default,
// descendant-only). // and @tag remainders stay descendant-only regardless.
// ----------------------------------------------------------------------------

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     intermediates/   (base file dir)
//     render/foo.glb   (sibling of intermediates)
TEST_F(PathResolutionTest, ResolveBotPath_AllowsParentDirWithinMaxDepth) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const inter = root / "intermediates";
  fs::create_directories(inter);
  fs::create_directories(root / "render");
  auto const base = inter / "m.StudioProcessing.json";
  auto const result = ResolveBotPath("../render/foo.glb", base, /*maxParentDepth=*/2, ExpectOK{});
  EXPECT_EQ(result, NormalizeBotPath(root / "render" / "foo.glb"));
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     a/b/            (base file dir)
TEST_F(PathResolutionTest, ResolveBotPath_RejectsParentDirBeyondMaxDepth) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const deep = root / "a" / "b";
  fs::create_directories(deep);
  Error error;
  // Three ".." exceeds maxParentDepth=2.
  (void)ResolveBotPath(
      "../../../x.glb", deep / "m.StudioProcessing.json", /*maxParentDepth=*/2, error);
  EXPECT_NOT_OK(error);
}

// // remainders stay descendant-only even when maxParentDepth > 0.
TEST_F(PathResolutionTest, ResolveBotPath_RootPrefixRejectsParentDirEvenWithMaxDepth) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  Error error;
  (void)ResolveBotPath("//../foo", root / "bot.superdex_bot", /*maxParentDepth=*/2, error);
  EXPECT_NOT_OK(error);
}

// @tag remainders stay descendant-only even when maxParentDepth > 0.
TEST_F(PathResolutionTest, ResolveBotPath_TaggedRejectsParentDirEvenWithMaxDepth) {
  auto const root = _tempDir / "root";
  WriteRootFile(root, R"({"@bots": "."})");
  Error error;
  (void)ResolveBotPath("@bots/../foo", root / "bot.superdex_bot", /*maxParentDepth=*/2, error);
  EXPECT_NOT_OK(error);
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     intermediates/m.StudioProcessing.json  (base)
//     render/foo.glb
TEST_F(PathResolutionTest, UnresolveBotPath_EmitsParentRelativeWithinMaxDepth) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const inter = root / "intermediates";
  fs::create_directories(inter);
  fs::create_directories(root / "render");
  auto const base = inter / "m.StudioProcessing.json";
  auto const target = root / "render" / "foo.glb";
  TouchFile(target);
  // Default (descendant-only) can't reach a sibling folder, so it uses //; depth 2 prefers the
  // local ../ form, which survives relocating the whole cluster.
  EXPECT_EQ(
      std::string(UnresolveBotPath(target, base, kBotPathMaxParentDepth, ExpectOK{})),
      "//render/foo.glb");
  EXPECT_EQ(
      std::string(UnresolveBotPath(target, base, /*maxParentDepth=*/2, ExpectOK{})),
      "../render/foo.glb");
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     a/b/c/m.StudioProcessing.json  (base, 3 levels below root)
//     x.glb
TEST_F(PathResolutionTest, UnresolveBotPath_FallsBackToRootBeyondMaxDepth) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const deep = root / "a" / "b" / "c";
  fs::create_directories(deep);
  auto const base = deep / "m.StudioProcessing.json";
  auto const target = root / "x.glb";
  TouchFile(target);
  // Reaching root/x.glb from a/b/c needs three "..", exceeding maxDepth=2 -> // fallback.
  EXPECT_EQ(
      std::string(UnresolveBotPath(target, base, /*maxParentDepth=*/2, ExpectOK{})), "//x.glb");
}

// File structure (under _tempDir):
//   root/
//     .superdex_root  (empty)
//     intermediates/m.StudioProcessing.json  (base)
//     render/foo.glb
TEST_F(PathResolutionTest, RoundTrip_ParentRelativeWithMaxDepth) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  auto const inter = root / "intermediates";
  fs::create_directories(inter);
  fs::create_directories(root / "render");
  auto const base = inter / "m.StudioProcessing.json";
  TouchFile(root / "render" / "foo.glb");
  std::string const original = "../render/foo.glb";
  auto const abs = ResolveBotPath(original, base, /*maxParentDepth=*/2, ExpectOK{});
  auto const back = UnresolveBotPath(abs, base, /*maxParentDepth=*/2, ExpectOK{});
  EXPECT_EQ(std::string(back), original);
}
