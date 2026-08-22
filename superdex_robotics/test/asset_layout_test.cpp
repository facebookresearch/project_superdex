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
#include <superdex_robotics/utils/file_utils.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

using namespace superdex::robotics;
using namespace mochi::test;

namespace {

namespace fs = std::filesystem;

/* Expectations are normalized the same way the code under test normalizes, rather than
 * canonicalized: a temp directory reached through a symlink (macOS /var) would otherwise disagree
 * on spelling while naming the same directory. */
class AssetLayoutTest : public testing::Test {
 protected:
  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("asset_layout_test", ExpectOK{});
  fs::path _tempDir = _tempDirCleanup.Path();

  // A base folder with the given role subfolders created inside it.
  fs::path MakeAsset(std::string const& name, std::vector<std::string> const& roleSubdirs) {
    auto base = _tempDir / name;
    fs::create_directories(base);
    for (auto const& subdir : roleSubdirs) {
      fs::create_directories(base / subdir);
    }
    return base;
  }

  static void ExpectSamePath(fs::path const& actual, fs::path const& expected) {
    EXPECT_EQ(NormalizeBotPath(actual), NormalizeBotPath(expected));
  }
};

TEST_F(AssetLayoutTest, RoleSubdirNamesAreRecognizedCaseInsensitively) {
  EXPECT_TRUE(IsAssetRoleSubdirName(kCadSubdir));
  EXPECT_TRUE(IsAssetRoleSubdirName(kCollisionSubdir));
  EXPECT_TRUE(IsAssetRoleSubdirName(kRenderSubdir));
  EXPECT_TRUE(IsAssetRoleSubdirName(kIntermediatesSubdir));
  EXPECT_TRUE(IsAssetRoleSubdirName("Render"));
  EXPECT_FALSE(IsAssetRoleSubdirName("visual"));
  EXPECT_FALSE(IsAssetRoleSubdirName(""));
}

// The role-subfolder layout: a file opened out of `cad/` belongs to the folder above it.
TEST_F(AssetLayoutTest, BaseFolderOfAFileInARoleFolderIsTheFolderAbove) {
  auto const base = MakeAsset("openarm_torso", {"cad", "collision", "render"});
  ExpectSamePath(FindAssetBaseFolder(base / kCadSubdir), base);
  ExpectSamePath(FindAssetBaseFolder(base / kRenderSubdir), base);
  EXPECT_EQ(DetectAssetFolderLayout(base / kCadSubdir), AssetFolderLayout::RoleSubdirs);
}

/* The regression this anchoring exists for: a model sitting directly in its asset's folder must
 * resolve to that folder, not to the category folder holding unrelated sibling assets. Counting a
 * fixed number of levels up put one asset's saved pipeline in `prefabs/internal/intermediates/`,
 * pooled with every other asset under `internal/`. */
TEST_F(AssetLayoutTest, BaseFolderOfAFlatAssetDoesNotEscapeIntoTheCategoryFolder) {
  auto const category = _tempDir / "internal";
  auto const base = category / "paper_cups";
  fs::create_directories(base);

  ExpectSamePath(FindAssetBaseFolder(base), base);
  EXPECT_EQ(DetectAssetFolderLayout(base), AssetFolderLayout::Flat);
  ExpectSamePath(AssetRoleFolderForWrite(base, kIntermediatesSubdir), base / kIntermediatesSubdir);
}

// Opening out of a role folder is proof enough that the asset uses them; a missing sibling is
// created rather than sidestepped, so a STEP in `cad/` exports to `render/` and `collision/`.
TEST_F(AssetLayoutTest, WritesTargetRoleFoldersThatDoNotExistYet) {
  auto const base = MakeAsset("fr3", {"cad"});
  auto const originDir = base / kCadSubdir;

  ASSERT_FALSE(fs::exists(base / kRenderSubdir));
  ASSERT_FALSE(fs::exists(base / kCollisionSubdir));
  ExpectSamePath(AssetRoleFolderForWrite(originDir, kRenderSubdir), base / kRenderSubdir);
  ExpectSamePath(AssetRoleFolderForWrite(originDir, kCollisionSubdir), base / kCollisionSubdir);
  // Resolving a write target must not be what creates the folder.
  EXPECT_FALSE(fs::exists(base / kRenderSubdir));
}

// A file loose in a base folder that already has role folders belongs to that layout too.
TEST_F(AssetLayoutTest, LooseFileInAnAssetThatUsesRoleFoldersWritesIntoThem) {
  auto const base = MakeAsset("jenga", {"render"});
  EXPECT_EQ(DetectAssetFolderLayout(base), AssetFolderLayout::RoleSubdirs);
  ExpectSamePath(AssetRoleFolderForWrite(base, kCollisionSubdir), base / kCollisionSubdir);
}

// A flat asset stays flat: exports land beside the model rather than growing role folders.
TEST_F(AssetLayoutTest, FlatAssetWritesStayInTheBaseFolder) {
  auto const base = MakeAsset("axis_gizmos", {});
  ExpectSamePath(AssetRoleFolderForWrite(base, kRenderSubdir), base);
  ExpectSamePath(AssetRoleFolderForWrite(base, kCollisionSubdir), base);
  ExpectSamePath(AssetRoleFolderForWrite(base, ""), base);
}

// Intermediates are the exception: generated pipelines and scratch exports are bookkeeping, and
// stay out of a flat asset's folder under either layout.
TEST_F(AssetLayoutTest, IntermediatesAreAlwaysASubfolder) {
  auto const flat = MakeAsset("flat", {});
  auto const structured = MakeAsset("structured", {"cad"});
  ExpectSamePath(AssetRoleFolderForWrite(flat, kIntermediatesSubdir), flat / kIntermediatesSubdir);
  ExpectSamePath(
      AssetRoleFolderForWrite(structured / kCadSubdir, kIntermediatesSubdir),
      structured / kIntermediatesSubdir);
}

/* A role folder may be partitioned -- `cad/internal/` holds CAD sources that must not be
 * open-sourced (bots/internal/ur7e is laid out this way). A file in there still belongs to the
 * asset, so the anchor ascends to the nearest role folder rather than testing only the immediate
 * directory name. Getting this wrong is not merely a failed lookup: the subfolder would be taken
 * for a flat asset, and exports would default into the very folder being stripped. */
TEST_F(AssetLayoutTest, BaseFolderOfAFileInARoleSubfolderIsStillTheAsset) {
  auto const base = MakeAsset("ur7e", {"cad", "render"});
  auto const partitioned = base / std::string(kCadSubdir) / "internal";
  fs::create_directories(partitioned);

  ExpectSamePath(FindAssetBaseFolder(partitioned), base);
  EXPECT_EQ(DetectAssetFolderLayout(partitioned), AssetFolderLayout::RoleSubdirs);
  // The export of a partitioned CAD source belongs in the role folder itself, not beside it.
  ExpectSamePath(AssetRoleFolderForWrite(partitioned, kRenderSubdir), base / kRenderSubdir);
  ExpectSamePath(
      AssetRoleFolderForWrite(partitioned, kIntermediatesSubdir), base / kIntermediatesSubdir);
}

// Nesting depth is not fixed: the anchor ascends until it finds a role folder.
TEST_F(AssetLayoutTest, BaseFolderResolvesThroughSeveralSubfolderLevels) {
  auto const base = MakeAsset("vendor_bot", {"cad"});
  auto const deep = base / std::string(kCadSubdir) / "internal" / "vendor";
  fs::create_directories(deep);

  ExpectSamePath(FindAssetBaseFolder(deep), base);
}

// Bots that split into per-side folders make each side its own asset.
TEST_F(AssetLayoutTest, PerSideSubfolderIsItsOwnBaseFolder) {
  auto const hand = _tempDir / "dg5f_short";
  auto const right = hand / "right";
  fs::create_directories(right / std::string(kCollisionSubdir));

  ExpectSamePath(FindAssetBaseFolder(right / kCollisionSubdir), right);
  ExpectSamePath(
      AssetRoleFolderForWrite(right / kCollisionSubdir, kRenderSubdir), right / kRenderSubdir);
}

} // namespace
