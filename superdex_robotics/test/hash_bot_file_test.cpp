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
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

class HashBotFileTest : public testing::Test {
 protected:
  std::filesystem::path WriteTempFile(std::string_view name, std::string_view content) {
    auto path = _tempDir / name;
    mochi::WriteFile(path, content, ExpectOK{});
    return path;
  }

  void SetUpFR3TempDir(std::string const& testName) {
    _dirPath = _tempDir / testName;
    // Drop a .superdex_root marker so HashBotFile can anchor relative paths. Without
    // this, the temp tree has no recognizable bots root and HashBotFile fails.
    mochi::WriteFile(_dirPath / kRootMarkerFile, std::string_view{}, ExpectOK{});
    /* Drop a .superdex_root marker so HashBotFile can anchor relative paths. Without
     * this, the temp tree has no recognizable bots root and HashBotFile fails. */
    {
      std::ofstream(_tempDir / std::string(kRootMarkerFile)).close();
    }
    auto const fr3BotPath = std::filesystem::path(GetAssetPath("bots/arms/fr3/fr3.superdex_bot"));
    // Canonicalize via the bot FILE (which is a symlink in Buck sandbox) to get the
    // real repo directory. Canonicalizing the directory alone doesn't follow the
    // per-file symlinks inside it, leaving render/ full of broken relative symlinks.
    auto const fr3Dir = std::filesystem::weakly_canonical(fr3BotPath).parent_path();
    auto const copyOpts = std::filesystem::copy_options::recursive |
        std::filesystem::copy_options::overwrite_existing;
    std::filesystem::copy(fr3Dir / "render", _dirPath / "render", copyOpts);
    std::filesystem::copy(fr3Dir / "collision", _dirPath / "collision", copyOpts);
    // Ensure copied files are writable. Buck's remote sandbox stages test_assets as
    // read-only, and std::filesystem::copy preserves those permissions. Tests that
    // modify file content (e.g. HashBotFile_ModifiedModelFileContent) need write access.
    for (auto const& entry : std::filesystem::recursive_directory_iterator(_dirPath)) {
      if (entry.is_regular_file()) {
        std::filesystem::permissions(
            entry.path(), std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
      }
    }
  }

  BotPrefab LoadFR3ForTempDir() {
    auto const fr3Path = GetAssetPath("bots/arms/fr3/fr3.superdex_bot");
    auto botPrefab = LoadBotPrefabFromFile(fr3Path, ExpectOK{});
    auto const fr3Dir =
        std::filesystem::weakly_canonical(std::filesystem::path(fr3Path)).parent_path();
    for (auto& link : botPrefab.links) {
      if (!link.renderModelFile.empty()) {
        auto rel =
            std::filesystem::relative(std::filesystem::path(link.renderModelFile.c_str()), fr3Dir);
        link.renderModelFile = DynamicString((_dirPath / rel).generic_string());
      }
      if (!link.shapeFile.empty()) {
        auto rel = std::filesystem::relative(std::filesystem::path(link.shapeFile.c_str()), fr3Dir);
        link.shapeFile = DynamicString((_dirPath / rel).generic_string());
      }
    }
    return botPrefab;
  }

  std::string SaveToTempDir(BotPrefab const& botPrefab, std::string const& filename) {
    auto path = (_dirPath / filename).string();
    SaveToFile(botPrefab, path, ExpectOK{});
    return path;
  }

  // Temp directory with automatic cleanup
  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("hash_bot_file_test", ExpectOK{});
  std::filesystem::path _tempDir = _tempDirCleanup.Path();
  std::filesystem::path _dirPath = _tempDir;
};

} // namespace

// --- Basic tests ---

TEST_F(HashBotFileTest, HashBotFile_ValidFile) {
  auto path = GetAssetPath("bots/arms/fr3/fr3.superdex_bot");
  auto result = HashBotFile(path, ExpectOK{});
  EXPECT_EQ(result.size(), 16u);
  for (auto c : std::string_view(result.c_str(), result.size())) {
    EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
  }
}

TEST_F(HashBotFileTest, HashBotFile_Deterministic) {
  auto path = GetAssetPath("bots/arms/fr3/fr3.superdex_bot");
  auto result1 = HashBotFile(path, ExpectOK{});
  auto result2 = HashBotFile(path, ExpectOK{});
  EXPECT_EQ(std::string_view(result1.c_str()), std::string_view(result2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_DifferentBaseBots) {
  auto path1 = GetAssetPath("bots/arms/fr3/fr3.superdex_bot");
  auto path2 = GetAssetPath("bots/hands/dg5f_long/right/dg5f_long_right.superdex_bot");
  auto result1 = HashBotFile(path1, ExpectOK{});
  auto result2 = HashBotFile(path2, ExpectOK{});
  EXPECT_NE(std::string_view(result1.c_str()), std::string_view(result2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_DifferentModBots) {
  auto path1 =
      GetAssetPath("bots/arm_hand_combos/fr3_dg5f_short/left/fr3_dg5f_short_left.superdex_bot");
  auto path2 =
      GetAssetPath("bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot");
  auto result1 = HashBotFile(path1, ExpectOK{});
  auto result2 = HashBotFile(path2, ExpectOK{});
  EXPECT_NE(std::string_view(result1.c_str()), std::string_view(result2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_DifferentBotTypes) {
  auto path1 = GetAssetPath("bots/arms/fr3/fr3.superdex_bot");
  auto path2 =
      GetAssetPath("bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot");
  auto result1 = HashBotFile(path1, ExpectOK{});
  auto result2 = HashBotFile(path2, ExpectOK{});
  EXPECT_NE(std::string_view(result1.c_str()), std::string_view(result2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_InvalidFile) {
  auto path = WriteTempFile("hash_test_invalid.txt", "some content");
  auto result = HashBotFile(path.string(), ExpectNotOK{});
  EXPECT_EQ(result.size(), 0u);
}

TEST_F(HashBotFileTest, HashBotFile_NonexistentFile) {
  auto result = HashBotFile("nonexistent_file_that_does_not_exist.xyz", ExpectNotOK{});
  EXPECT_EQ(result.size(), 0u);
}

// --- Copy and modification tests ---

TEST_F(HashBotFileTest, HashBotFile_IdenticalCopy) {
  SetUpFR3TempDir("identical_copy");
  auto const botPrefab = LoadFR3ForTempDir();
  auto const baselinePath = SaveToTempDir(botPrefab, "fr3_baseline.superdex_bot");
  auto const copyPath = SaveToTempDir(botPrefab, "fr3_copy.superdex_bot");
  auto const hash1 = HashBotFile(baselinePath, ExpectOK{});
  auto const hash2 = HashBotFile(copyPath, ExpectOK{});
  EXPECT_EQ(std::string_view(hash1.c_str()), std::string_view(hash2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_ModifiedLinkParam) {
  SetUpFR3TempDir("modified_link");
  auto botPrefab = LoadFR3ForTempDir();
  auto const baselinePath = SaveToTempDir(botPrefab, "fr3_baseline.superdex_bot");
  botPrefab.links[1].name = "has_test_name";
  auto const modifiedPath = SaveToTempDir(botPrefab, "fr3_modified_link.superdex_bot");
  auto const hash1 = HashBotFile(baselinePath, ExpectOK{});
  auto const hash2 = HashBotFile(modifiedPath, ExpectOK{});
  EXPECT_NE(std::string_view(hash1.c_str()), std::string_view(hash2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_ModifiedJointParam) {
  SetUpFR3TempDir("modified_joint");
  auto botPrefab = LoadFR3ForTempDir();
  auto const baselinePath = SaveToTempDir(botPrefab, "fr3_baseline.superdex_bot");
  botPrefab.joints[2].name = "has_test_name";
  auto const modifiedPath = SaveToTempDir(botPrefab, "fr3_modified_joint.superdex_bot");
  auto const hash1 = HashBotFile(baselinePath, ExpectOK{});
  auto const hash2 = HashBotFile(modifiedPath, ExpectOK{});
  EXPECT_NE(std::string_view(hash1.c_str()), std::string_view(hash2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_ModifiedDefaultPose) {
  SetUpFR3TempDir("modified_pose");
  auto botPrefab = LoadFR3ForTempDir();
  auto const baselinePath = SaveToTempDir(botPrefab, "fr3_baseline.superdex_bot");
  botPrefab.defaultPose[0] = botPrefab.defaultPose[0] + 0.1f;
  auto const modifiedPath = SaveToTempDir(botPrefab, "fr3_modified_pose.superdex_bot");
  auto const hash1 = HashBotFile(baselinePath, ExpectOK{});
  auto const hash2 = HashBotFile(modifiedPath, ExpectOK{});
  EXPECT_NE(std::string_view(hash1.c_str()), std::string_view(hash2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_SwappedMochiModels) {
  SetUpFR3TempDir("swapped_mochi");
  auto botPrefab = LoadFR3ForTempDir();
  auto const baselinePath = SaveToTempDir(botPrefab, "fr3_baseline.superdex_bot");
  std::swap(botPrefab.links[1].shapeFile, botPrefab.links[2].shapeFile);
  std::swap(botPrefab.links[1].shapeScale, botPrefab.links[2].shapeScale);
  std::swap(botPrefab.links[1].shapeRotation, botPrefab.links[2].shapeRotation);
  std::swap(botPrefab.links[1].shapeTranslation, botPrefab.links[2].shapeTranslation);
  auto const modifiedPath = SaveToTempDir(botPrefab, "fr3_swapped_mochi.superdex_bot");
  auto const hash1 = HashBotFile(baselinePath, ExpectOK{});
  auto const hash2 = HashBotFile(modifiedPath, ExpectOK{});
  EXPECT_NE(std::string_view(hash1.c_str()), std::string_view(hash2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_SwappedRenderModels) {
  SetUpFR3TempDir("swapped_render");
  auto botPrefab = LoadFR3ForTempDir();
  auto const baselinePath = SaveToTempDir(botPrefab, "fr3_baseline.superdex_bot");
  std::swap(botPrefab.links[1].renderModelFile, botPrefab.links[2].renderModelFile);
  std::swap(botPrefab.links[1].renderModelScale, botPrefab.links[2].renderModelScale);
  std::swap(botPrefab.links[1].renderModelRotation, botPrefab.links[2].renderModelRotation);
  std::swap(botPrefab.links[1].renderModelTranslation, botPrefab.links[2].renderModelTranslation);
  auto const modifiedPath = SaveToTempDir(botPrefab, "fr3_swapped_render.superdex_bot");
  auto const hash1 = HashBotFile(baselinePath, ExpectOK{});
  auto const hash2 = HashBotFile(modifiedPath, ExpectOK{});
  EXPECT_NE(std::string_view(hash1.c_str()), std::string_view(hash2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_ModifiedModelFileContent) {
  SetUpFR3TempDir("modified_content");
  auto const botPrefab = LoadFR3ForTempDir();
  auto const path = SaveToTempDir(botPrefab, "fr3_content.superdex_bot");
  auto const hash1 = HashBotFile(path, ExpectOK{});
  for (auto const& link : botPrefab.links) {
    if (!link.shapeFile.empty()) {
      std::ofstream ofs(link.shapeFile.c_str(), std::ios::binary | std::ios::app);
      ofs << "extra_bytes";
      break;
    }
  }
  auto const hash2 = HashBotFile(path, ExpectOK{});
  EXPECT_NE(std::string_view(hash1.c_str()), std::string_view(hash2.c_str()));
}

TEST_F(HashBotFileTest, HashBotFile_RemovedLastLinkAndJoint) {
  SetUpFR3TempDir("removed_last");
  auto botPrefab = LoadFR3ForTempDir();
  auto const baselinePath = SaveToTempDir(botPrefab, "fr3_baseline.superdex_bot");
  botPrefab.links.pop_back();
  botPrefab.joints.pop_back();
  auto const modifiedPath = SaveToTempDir(botPrefab, "fr3_removed_last.superdex_bot");
  auto const hash1 = HashBotFile(baselinePath, ExpectOK{});
  auto const hash2 = HashBotFile(modifiedPath, ExpectOK{});
  EXPECT_NE(std::string_view(hash1.c_str()), std::string_view(hash2.c_str()));
}
