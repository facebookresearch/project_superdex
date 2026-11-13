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

#include <mochi_physics/utils/mochi_path_remap.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <mochi_core/test/log_suppression.h>

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

using namespace mochi;

namespace {

namespace fs = std::filesystem;

std::array<PathRemapEntry, 1> const kEntries = {
    PathRemapEntry{
        .from = DynamicString("/assets/tools/wrench.glb"),
        .to = DynamicString("/assets/tools/wrench-5.glb")},
};

class PathRemapTest : public testing::Test {
 protected:
  void TearDown() override {
    ClearPathRemap();
  }
};

} // namespace

TEST_F(PathRemapTest, InactiveUntilATableIsInstalled) {
  EXPECT_FALSE(IsPathRemapActive());
  SetPathRemap(kEntries);
  EXPECT_TRUE(IsPathRemapActive());
  ClearPathRemap();
  EXPECT_FALSE(IsPathRemapActive());
}

TEST_F(PathRemapTest, UnmappedPathIsUnchanged) {
  SetPathRemap(kEntries);
  EXPECT_EQ(std::string(RemapPath("/assets/tools/hammer.glb")), "/assets/tools/hammer.glb");
}

TEST_F(PathRemapTest, MappedPathIsRemapped) {
  SetPathRemap(kEntries);
  EXPECT_EQ(std::string(RemapPath("/assets/tools/wrench.glb")), "/assets/tools/wrench-5.glb");
}

TEST_F(PathRemapTest, LookupIgnoresSpelling) {
  SetPathRemap(kEntries);
  EXPECT_EQ(
      std::string(RemapPath("/assets/./meshes/../tools/wrench.glb")), "/assets/tools/wrench-5.glb");
}

TEST_F(PathRemapTest, RelativeLookupMatchesAbsoluteKey) {
  auto const absolute = fs::current_path() / "tools" / "wrench.glb";
  std::array<PathRemapEntry, 1> const entries = {
      PathRemapEntry{
          .from = DynamicString(absolute.generic_string()), .to = DynamicString("/pinned.glb")},
  };
  SetPathRemap(entries);
  EXPECT_EQ(std::string(RemapPath("tools/wrench.glb")), "/pinned.glb");
}

TEST_F(PathRemapTest, LookupThroughSymlinkMatchesKey) {
  auto const dir = fs::temp_directory_path() /
      ("path_remap_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  fs::create_directories(dir / "real");
  std::error_code ec;
  fs::create_directory_symlink(dir / "real", dir / "link", ec);
  if (ec) {
    fs::remove_all(dir);
    GTEST_SKIP() << "Cannot create symlinks here: " << ec.message();
  }
  std::array<PathRemapEntry, 1> const entries = {
      PathRemapEntry{
          .from = DynamicString((dir / "real" / "wrench.glb").generic_string()),
          .to = DynamicString("/pinned.glb")},
  };
  SetPathRemap(entries);
  EXPECT_EQ(std::string(RemapPath((dir / "link" / "wrench.glb").generic_string())), "/pinned.glb");
  fs::remove_all(dir);
}

TEST_F(PathRemapTest, PrefabReferencesAreRemapped) {
  SetPathRemap(kEntries);
  EXPECT_EQ(
      std::string(prefab::GetPrefabFullPath("./wrench.glb", "", "/assets/tools/kit.mochi_prefab")),
      "/assets/tools/wrench-5.glb");
}

class StrictPathRemapTest : public PathRemapTest {
 protected:
  /* RemapPath of @p path, which a strict root refuses. Refusals log an error, and the test
   * framework fails any test that logs one, so errors are silenced for the lookup. */
  static std::string RemapRefused(std::string_view path) {
    auto const suppressError = test::SuppressLogError();
    return std::string(RemapPath(path));
  }

  std::array<DynamicString, 1> const _strictRoots = {DynamicString("/assets")};
};

TEST_F(StrictPathRemapTest, UnmappedFileUnderAStrictRootIsRefused) {
  SetPathRemap(kEntries, _strictRoots);
  EXPECT_EQ(RemapRefused("/assets/tools/hammer.glb"), "");
}

TEST_F(StrictPathRemapTest, StrictRootsWithoutEntriesRefuseFilesUnderThem) {
  SetPathRemap({}, _strictRoots);
  EXPECT_TRUE(IsPathRemapActive());
  EXPECT_EQ(RemapRefused("/assets/tools/hammer.glb"), "");
}

/* A symlink under a strict root is under it even when it points outside every root: an unpinned
 * one is refused, and a pinned one still remaps. */
TEST_F(StrictPathRemapTest, SymlinkUnderAStrictRootIsCoveredWhereverItPoints) {
  auto const dir = fs::temp_directory_path() /
      ("strict_symlink_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  fs::create_directories(dir / "root");
  fs::create_directories(dir / "outside");
  std::ofstream(dir / "outside" / "a.glb") << "tip";
  std::ofstream(dir / "outside" / "b.glb") << "tip";
  std::ofstream(dir / "root" / "a-1.glb") << "pinned";
  std::error_code ec;
  fs::create_symlink(dir / "outside" / "a.glb", dir / "root" / "a.glb", ec);
  if (!ec) {
    fs::create_symlink(dir / "outside" / "b.glb", dir / "root" / "b.glb", ec);
  }
  if (ec) {
    fs::remove_all(dir);
    GTEST_SKIP() << "Cannot create symlinks here: " << ec.message();
  }
  std::array<PathRemapEntry, 1> const entries = {
      PathRemapEntry{
          .from = DynamicString((dir / "root" / "a.glb").generic_string()),
          .to = DynamicString((dir / "root" / "a-1.glb").generic_string())},
  };
  std::array<DynamicString, 1> const strictRoots = {DynamicString((dir / "root").generic_string())};
  SetPathRemap(entries, strictRoots);
  EXPECT_EQ(
      std::string(RemapPath((dir / "root" / "a.glb").generic_string())),
      (dir / "root" / "a-1.glb").generic_string());
  EXPECT_EQ(RemapRefused((dir / "root" / "b.glb").generic_string()), "");
  fs::remove_all(dir);
}

TEST_F(StrictPathRemapTest, MappedFileAndItsTargetAreNotRefused) {
  SetPathRemap(kEntries, _strictRoots);
  EXPECT_EQ(std::string(RemapPath("/assets/tools/wrench.glb")), "/assets/tools/wrench-5.glb");
  EXPECT_EQ(std::string(RemapPath("/assets/tools/wrench-5.glb")), "/assets/tools/wrench-5.glb");
}

TEST_F(StrictPathRemapTest, FileOutsideTheStrictRootsIsUnchanged) {
  SetPathRemap(kEntries, _strictRoots);
  EXPECT_EQ(std::string(RemapPath("/other/hammer.glb")), "/other/hammer.glb");
  EXPECT_EQ(std::string(RemapPath("/assets-old/hammer.glb")), "/assets-old/hammer.glb");
}

TEST_F(StrictPathRemapTest, DirectoryUnderAStrictRootIsNotRefused) {
  auto const dir = fs::temp_directory_path() /
      ("strict_path_remap_test_" +
       std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  fs::create_directories(dir / "meshes");
  std::array<PathRemapEntry, 1> const entries = {
      PathRemapEntry{
          .from = DynamicString((dir / "a.glb").generic_string()),
          .to = DynamicString((dir / "a-5.glb").generic_string())},
  };
  std::array<DynamicString, 1> const strictRoots = {DynamicString(dir.generic_string())};
  SetPathRemap(entries, strictRoots);
  auto const meshes = (dir / "meshes").generic_string();
  EXPECT_EQ(std::string(RemapPath(meshes)), meshes);
  fs::remove_all(dir);
}

TEST_F(StrictPathRemapTest, ClearingTheTableDropsItsStrictRoots) {
  SetPathRemap(kEntries, _strictRoots);
  ClearPathRemap();
  std::array<PathRemapEntry, 1> const elsewhere = {
      PathRemapEntry{.from = DynamicString("/other/a.glb"), .to = DynamicString("/other/a-5.glb")},
  };
  SetPathRemap(elsewhere);
  EXPECT_EQ(std::string(RemapPath("/assets/tools/hammer.glb")), "/assets/tools/hammer.glb");
}

TEST_F(StrictPathRemapTest, PrefabReferenceUnderAStrictRootIsRefused) {
  SetPathRemap(kEntries, _strictRoots);
  auto const suppressError = test::SuppressLogError();
  EXPECT_EQ(
      std::string(prefab::GetPrefabFullPath("./hammer.glb", "", "/assets/tools/kit.mochi_prefab")),
      "");
}

// Windows has no permission bits that make path resolution fail.
#if !MOCHI_PLATFORM_WINDOWS
/* A path whose symlinks can't be resolved might not match a strict root's spelling, so it is
 * refused rather than risk loading a file the table does not pin. */
TEST_F(StrictPathRemapTest, UnresolvablePathIsRefusedWhileStrictRootsAreInstalled) {
  auto const dir = fs::temp_directory_path() /
      ("strict_unresolvable_test_" +
       std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  fs::create_directories(dir / "locked" / "inner");
  fs::permissions(dir / "locked", fs::perms::none);
  auto const lookup = (dir / "locked" / "inner" / "a.glb").generic_string();
  std::error_code ec;
  (void)fs::weakly_canonical(lookup, ec);
  if (!ec) {
    fs::permissions(dir / "locked", fs::perms::owner_all);
    fs::remove_all(dir);
    GTEST_SKIP() << "Cannot make path resolution fail here (for example, running as root).";
  }
  std::array<DynamicString, 1> const strictRoots = {DynamicString("/elsewhere")};
  SetPathRemap(kEntries, strictRoots);
  EXPECT_EQ(RemapRefused(lookup), "");
  // Without strict roots, the same lookup passes through unchanged.
  SetPathRemap(kEntries);
  EXPECT_EQ(std::string(RemapPath(lookup)), lookup);
  fs::permissions(dir / "locked", fs::perms::owner_all);
  fs::remove_all(dir);
}
#endif // !MOCHI_PLATFORM_WINDOWS
