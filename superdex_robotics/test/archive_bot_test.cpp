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

#include "mochi_bots_test_helpers.h"

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>

#include <filesystem>
#include <string>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

// Parameterized so each bot combo is a separate test case — archiving touches disk
// and this allows the test runner to shard work across processes for parallel execution.
class ArchiveBotTest : public ::testing::TestWithParam<std::filesystem::path> {};

TEST_P(ArchiveBotTest, ArchiveAndLoadRoundTrip) {
  auto const& botPath = GetParam();

  // Load the original bot to establish the expected link count for round-trip comparison.
  auto originalPrefab = LoadBotPrefabFromFile(botPath.string(), ExpectOK{});
  ASSERT_FALSE(originalPrefab.links.empty());

  // Set up a unique temp directory for this bot. Include the precision suffix so the
  // single- and double-precision test binaries do not collide on the same path.
  constexpr std::string_view kPrecision = std::is_same_v<real, double> ? "double" : "float";
  std::string stem = botPath.stem().string();
  auto const label = "archive_bot_test_" + std::string(kPrecision) + "_" + stem;
  auto const tempDir = mochi::CreateTempDirectory(label, ExpectOK{});
  auto const archiveFile = tempDir.Path() / (stem + ".superdex_bot_archive");

  /* Some shipped bots declare a sensor whose params file this build does not have -- the internal
   * sensors' params are stripped from the open-source export while the bots that declare them are
   * kept. Archiving and hashing warn about each such reference and carry on, so the warning is
   * expected here rather than a fault. */
  auto const suppressWarning = SuppressLogWarning();

  // Archive the bot. ArchiveBot creates the destination's parent directory itself.
  ArchiveParams params;
  params.src = botPath.string();
  params.dst = archiveFile.string();
  ArchiveBot(params, ExpectOK{});

  // The destination must be a single regular file (zip-format) with the .superdex_bot_archive
  // extension, not a directory.
  ASSERT_TRUE(std::filesystem::is_regular_file(archiveFile));

  auto extractedDir = ExtractBotArchiveToCache(archiveFile.string(), ExpectOK{});
  auto targetPath = GetExtractedBotArchiveTarget(extractedDir, ExpectOK{});
  auto loadedPrefab = LoadBotPrefabFromFile(archiveFile.string(), ExpectOK{});

  auto originalHash = HashBotFile(botPath.string(), ExpectOK{});
  auto loadedHash = HashBotFile(targetPath, ExpectOK{});
  EXPECT_EQ(originalHash, loadedHash);
}

INSTANTIATE_TEST_SUITE_P(
    AllBots,
    ArchiveBotTest,
    ::testing::ValuesIn(FindAllBotFiles()),
    [](::testing::TestParamInfo<std::filesystem::path> const& info) {
      auto const& path = info.param;
      auto const botsRoot = FindBotsRoot(path);
      std::string name = botsRoot.has_value()
          ? std::filesystem::relative(path, botsRoot.value()).replace_extension().string()
          : path.stem().string();
      for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
          c = '_';
        }
      }
      return name;
    });

// Regression test: a stale/incomplete extraction occupying the content-addressed
// cache directory (missing its metadata file) must not dead-lock extraction.
// Before the fix, the cache-hit check failed (no metadata) yet the atomic-rename
// publish could not replace the non-empty destination, so every subsequent
// ExtractBotArchiveToCache — and therefore HashBotFile / reimport — failed
// permanently with "Failed to publish extracted archive cache."
TEST(ArchiveBotCacheTest, RecoversFromStaleIncompleteCacheDir) {
  auto const bots = FindAllBotFiles();
  ASSERT_FALSE(bots.empty());
  auto const botPath = bots.front();

  // Unique-per-precision archive stem so the single- and double-precision test
  // binaries key into distinct content-addressed cache directories and do not
  // race on the destructive metadata removal below.
  constexpr std::string_view kPrecision = std::is_same_v<real, double> ? "double" : "float";
  auto const tempDir = mochi::CreateTempDirectory("archive_bot_cache_test", ExpectOK{});
  auto const archiveFile =
      tempDir.Path() / ("archive_cache_stale_" + std::string(kPrecision) + ".superdex_bot_archive");

  ArchiveParams params;
  params.src = botPath.string();
  params.dst = archiveFile.string();
  ArchiveBot(params, ExpectOK{});

  // First extraction publishes a complete cache directory.
  auto const extractedDir = ExtractBotArchiveToCache(archiveFile.string(), ExpectOK{});
  std::filesystem::path const dstDir{extractedDir.c_str()};
  auto const metadataPath = dstDir / kArchiveMetadataFile;
  ASSERT_TRUE(std::filesystem::exists(metadataPath));

  // Simulate an interrupted/older extraction: the directory exists and is
  // non-empty but has lost its metadata file, so it is neither a valid cache
  // hit nor a rename-able destination.
  std::filesystem::remove(metadataPath);
  ASSERT_FALSE(std::filesystem::exists(metadataPath));
  ASSERT_TRUE(std::filesystem::exists(dstDir));

  // Re-extraction must self-heal: clear the stale directory and re-publish.
  auto const recoveredDir = ExtractBotArchiveToCache(archiveFile.string(), ExpectOK{});
  EXPECT_EQ(recoveredDir, extractedDir);
  EXPECT_TRUE(std::filesystem::exists(metadataPath));

  // The recovered cache is fully usable end-to-end: it hashes identically to
  // the original source bot.
  auto const target = GetExtractedBotArchiveTarget(recoveredDir, ExpectOK{});
  auto const originalHash = HashBotFile(botPath.string(), ExpectOK{});
  auto const recoveredHash = HashBotFile(target, ExpectOK{});
  EXPECT_EQ(originalHash, recoveredHash);
}

// ----------------------------------------------------------------------------
// Multi-root archive tests
// ----------------------------------------------------------------------------

namespace {

namespace fs = std::filesystem;

class ArchiveBotMultiRootTest : public testing::Test {
 protected:
  // Write an empty .superdex_root marker in `dir`.
  static void WriteEmptyRootMarker(fs::path const& dir) {
    mochi::WriteFile(dir / kRootMarkerFile, std::string_view{}, ExpectOK{});
  }

  // Write a .superdex_root JSON file in `dir` with `contents`.
  static void WriteRootFile(fs::path const& dir, std::string_view contents) {
    mochi::WriteFile(dir / kRootMarkerFile, contents, ExpectOK{});
  }

  // Recursively copy a bot directory (e.g. fr3 or dg5f_long_right) from the
  // assets tree into `dstDir`. After the copy, `dstDir/<botName>.superdex_bot` is
  // loadable.
  static void CopyBotDirInto(fs::path const& assetsRelDir, fs::path const& dstDir) {
    fs::path const src = fs::path(GetAssetsDir()) / assetsRelDir;
    fs::create_directories(dstDir.parent_path());
    fs::copy(src, dstDir, fs::copy_options::recursive);
    for (auto const& entry : fs::recursive_directory_iterator(dstDir)) {
      if (entry.is_regular_file()) {
        fs::permissions(entry.path(), fs::perms::owner_write, fs::perm_options::add);
      }
    }
  }

  // Write a ModBotPrefab to `path` that uses an fr3 arm as the
  // base and attaches a dg5f hand to the arm's tip link (`fr3_link8`). `armBot`
  // and `handBot` are absolute on-disk paths to existing .superdex_bot files;
  // SaveToFile internally rewrites them to whichever canonical form (file-
  // relative, "//"-prefixed, or "@tag/...") is appropriate for the surrounding
  // root layout. Passing absolute paths avoids platform-specific parsing
  // (notably, on POSIX `"//x"` parses as an absolute path, defeating
  // MakePathsRelative).
  static void
  WriteFr3WithDg5fModBot(fs::path const& path, fs::path const& armBot, fs::path const& handBot) {
    ModBotPrefab params;
    params.name = DynamicString{"fr3_dg5f_mod_bot"};
    params.base = DynamicString{armBot.generic_string()};
    AttachBot attach;
    attach.name = DynamicString{"dg5f"};
    attach.parentLinkName = DynamicString{"fr3_link8"};
    attach.joint.name = DynamicString{"dg5f_to_fr3"};
    attach.joint.type = ArticulatedJointType::Hard;
    attach.path = DynamicString{handBot.generic_string()};
    params.modifications.push_back(attach);
    SaveToFile(params, path.string(), ExpectOK{});
  }

  // Temp directory with automatic cleanup.
  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("archive_bot_multi_root", ExpectOK{});
  fs::path _tempDir = _tempDirCleanup.Path();
};

} // namespace

// Helper: archive the mod bot at `srcBot` and verify the extracted bot
// hashes identically to the source via HashBotFile.
static void ExpectArchiveHashRoundTrip(fs::path const& srcBot, fs::path const& tempDir) {
  auto const archiveFile = tempDir / "out.superdex_bot_archive";
  ArchiveParams params;
  params.src = srcBot.string();
  params.dst = archiveFile.string();
  ArchiveBot(params, ExpectOK{});
  ASSERT_TRUE(fs::is_regular_file(archiveFile));

  auto const extractedDir = ExtractBotArchiveToCache(archiveFile.string(), ExpectOK{});
  auto const target = GetExtractedBotArchiveTarget(extractedDir, ExpectOK{});

  auto const originalHash = HashBotFile(srcBot.string(), ExpectOK{});
  auto const loadedHash = HashBotFile(target, ExpectOK{});
  EXPECT_EQ(originalHash, loadedHash);
}

// True if a regular file named `filename` exists anywhere under `dir`.
static bool ContainsFileNamed(fs::path const& dir, std::string_view filename) {
  for (auto const& entry : fs::recursive_directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().filename() == filename) {
      return true;
    }
  }
  return false;
}

// Asset-relative paths to the canonical fr3 arm and dg5f_long right hand bots.
static constexpr std::string_view kFr3AssetDir = "bots/arms/fr3";
static constexpr std::string_view kDg5fAssetDir = "bots/hands/dg5f_long/right";
static constexpr std::string_view kFr3BotName = "fr3.superdex_bot";
static constexpr std::string_view kDg5fBotName = "dg5f_long_right.superdex_bot";

// Two-root, sibling depth: mod bot + fr3 arm copy live in root A; the dg5f
// hand copy lives in sibling root B. A's .superdex_root names B via @b.
// SaveToFile rewrites the absolute base path as a plain file-relative
// reference (rule 1) and the absolute attach path as `@b/...` (rule 3); the
// archive must mirror both root subtrees so those forms still resolve after
// extraction.
//
// File structure (under _tempDir):
//   A/
//     .superdex_root  ({"@b": "../B"})
//     fr3/              (copy of bots/arms/fr3)
//     mod_bot.superdex_bot
//   B/
//     .superdex_root  (empty)
//     dg5f/             (copy of bots/hands/dg5f_long/right)
TEST_F(ArchiveBotMultiRootTest, TwoRoots_SiblingDepth) {
  auto const prevLogFn = GetLogCallback();
  SetLogCallback(nullptr);
  MOCHI_DEFER(SetLogCallback(prevLogFn));

  auto const rootA = _tempDir / "A";
  auto const rootB = _tempDir / "B";
  WriteRootFile(rootA, R"({"@b": "../B"})");
  WriteEmptyRootMarker(rootB);
  CopyBotDirInto(kFr3AssetDir, rootA / "fr3");
  CopyBotDirInto(kDg5fAssetDir, rootB / "dg5f");
  auto const modBot = rootA / "mod_bot.superdex_bot";
  WriteFr3WithDg5fModBot(modBot, rootA / "fr3" / kFr3BotName, rootB / "dg5f" / kDg5fBotName);

  ExpectArchiveHashRoundTrip(modBot, _tempDir);
}

// Two-root, asymmetric depth: mod bot + fr3 arm in shallow root A, dg5f hand
// in deeply-nested root B (tempDir/deep/nested/B). The archive must preserve
// the relative depth difference so A's @b (originally "../deep/nested/B")
// still resolves after extraction.
//
// File structure (under _tempDir):
//   A/
//     .superdex_root  ({"@b": "../deep/nested/B"})
//     fr3/              (copy of bots/arms/fr3)
//     mod_bot.superdex_bot
//   deep/
//     nested/
//       B/
//         .superdex_root  (empty)
//         dg5f/             (copy of bots/hands/dg5f_long/right)
TEST_F(ArchiveBotMultiRootTest, TwoRoots_AsymmetricDepth) {
  auto const prevLogFn = GetLogCallback();
  SetLogCallback(nullptr);
  MOCHI_DEFER(SetLogCallback(prevLogFn));

  auto const rootA = _tempDir / "A";
  auto const rootB = _tempDir / "deep" / "nested" / "B";
  WriteRootFile(rootA, R"({"@b": "../deep/nested/B"})");
  WriteEmptyRootMarker(rootB);
  CopyBotDirInto(kFr3AssetDir, rootA / "fr3");
  CopyBotDirInto(kDg5fAssetDir, rootB / "dg5f");
  auto const modBot = rootA / "mod_bot.superdex_bot";
  WriteFr3WithDg5fModBot(modBot, rootA / "fr3" / kFr3BotName, rootB / "dg5f" / kDg5fBotName);

  ExpectArchiveHashRoundTrip(modBot, _tempDir);
}

// An asset tree containing a symlink still archives into something that loads. The archiver treats
// paths as written rather than resolving them, so the link is bundled at the path the bot names --
// the file is carried at both spellings if both are referenced, which costs bytes but leaves no
// reference dangling on extraction. Skipped where the platform will not create symlinks.
TEST_F(ArchiveBotMultiRootTest, ArchivesThroughASymlinkedAssetPath) {
  auto const root = _tempDir / "symlink_root";
  WriteEmptyRootMarker(root);
  CopyBotDirInto(kFr3AssetDir, root / "fr3");

  /* A second name for the collision directory, inside the bot's own folder. Asset references are
   * descendant-only, so this is the only shape a symlinked reference can legally take. */
  std::error_code ec;
  fs::create_directory_symlink(root / "fr3" / "collision", root / "fr3" / "collision_link", ec);
  if (ec) {
    GTEST_SKIP() << "Platform cannot create symlinks: " << ec.message();
  }

  /* Point one link's collision mesh at the symlinked spelling. This is the case that
   * discriminates: the bot is reached directly while one asset it names is reached through the
   * link, so resolving paths and writing them as authored disagree about where that asset belongs
   * in the archive. */
  auto const bot = root / "fr3" / std::string(kFr3BotName);
  auto botJson = mochi::ReadFileString(bot, ExpectOK{});
  auto const direct = std::string(R"("shape": "collision/fr3_link0.mochi.h5")");
  auto const viaLink = std::string(R"("shape": "collision_link/fr3_link0.mochi.h5")");
  ASSERT_NE(botJson.find(direct), std::string::npos);
  botJson.replace(botJson.find(direct), direct.size(), viaLink);
  mochi::WriteFile(bot, botJson, ExpectOK{});

  auto const archiveFile = _tempDir / "symlinked.superdex_bot_archive";
  ArchiveParams params;
  params.src = bot.string();
  params.dst = archiveFile.string();
  ArchiveBot(params, ExpectOK{});
  ASSERT_TRUE(fs::is_regular_file(archiveFile));

  // The archive must load, and every asset the bot names must be present inside it: that is the
  // property that matters, not whether the hash matches the pre-archive bot.
  auto const extractedDir = ExtractBotArchiveToCache(archiveFile.string(), ExpectOK{});
  auto const targetPath = GetExtractedBotArchiveTarget(extractedDir, ExpectOK{});
  auto const loaded = LoadBotPrefabFromFile(targetPath, ExpectOK{});
  EXPECT_FALSE(loaded.links.empty());
  /* The hash also survives the round trip. Nothing requires that of a symlinked tree -- two
   * spellings of one file are deliberately distinct now, so a bot naming the link hashes
   * differently from one naming the target -- but a bot must still hash the same before and after
   * being archived, which is the property the round trip is for. */
  EXPECT_EQ(HashBotFile(bot.string(), ExpectOK{}), HashBotFile(targetPath, ExpectOK{}));
  for (auto const& link : loaded.links) {
    if (!link.shapeFile.empty()) {
      EXPECT_TRUE(fs::exists(std::string(link.shapeFile)))
          << "shape missing from archive: " << std::string(link.shapeFile);
    }
    if (!link.renderModelFile.empty()) {
      EXPECT_TRUE(fs::exists(std::string(link.renderModelFile)))
          << "render model missing from archive: " << std::string(link.renderModelFile);
    }
  }
}

/* Give an already-copied bot an inline sensor whose params name a companion file.
 *
 * The sensor type is never registered and never needs to be: collecting a sensor's assets is
 * deliberately type-agnostic, so a made-up name exercises the same path a real one would, and the
 * test does not depend on which sensor types a given build happens to ship. */
static void AddSensorWithParams(
    fs::path const& bot,
    fs::path const& botDir,
    std::string const& paramsContents,
    bool writeWeights) {
  auto botJson = mochi::ReadFileString(bot, ExpectOK{});
  auto const anchor = std::string(R"("shape": "collision/fr3_link0.mochi.h5")");
  auto const withSensor =
      anchor + R"(, "sensors": [{"type": "TEST_SENSOR", "params": "params/test.superdex_sensor"}])";
  auto const at = botJson.find(anchor);
  ASSERT_NE(at, std::string::npos) << "fr3 asset no longer has the link this test anchors on";
  botJson.replace(at, anchor.size(), withSensor);
  mochi::WriteFile(bot, botJson, ExpectOK{});

  fs::create_directories(botDir / "params");
  mochi::WriteFile(botDir / "params" / "test.superdex_sensor", paramsContents, ExpectOK{});
  if (writeWeights) {
    fs::create_directories(botDir / "params" / "weights");
    mochi::WriteFile(
        botDir / "params" / "weights" / "test_weights.json", R"({"w": [1.0]})", ExpectOK{});
  }
}

// A link that declares a sensor must archive the sensor's params file AND the companion assets that
// file references. This guards against the regression where archives bundled the sensor declaration
// but neither its params nor the files they name, so the sensor failed to load from the archive.
TEST_F(ArchiveBotMultiRootTest, CollectsSensorParamsAndReferencedAssets) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  CopyBotDirInto(kFr3AssetDir, root / "fr3");
  auto const bot = root / "fr3" / std::string(kFr3BotName);
  AddSensorWithParams(
      bot, root / "fr3", R"({"weightsPath": "weights/test_weights.json"})", /*writeWeights*/ true);

  auto const archiveFile = _tempDir / "with_sensor.superdex_bot_archive";
  ArchiveParams params;
  params.src = bot.string();
  params.dst = archiveFile.string();
  ArchiveBot(params, ExpectOK{});
  ASSERT_TRUE(fs::is_regular_file(archiveFile));

  // The sensor's params file and the weights it names must both be bundled.
  auto const extractedDir = ExtractBotArchiveToCache(archiveFile.string(), ExpectOK{});
  EXPECT_TRUE(ContainsFileNamed(extractedDir, "test.superdex_sensor"));
  EXPECT_TRUE(ContainsFileNamed(extractedDir, "test_weights.json"));
}

// A sensor params file that cannot be parsed still archives. The archive mirrors the assets it was
// built from, broken ones included: the unparseable file is bundled as-is, so loading from the
// archive fails at the same parse that loading from the source tree does. Refusing to archive would
// make a source tree that loads badly impossible to archive at all.
TEST_F(ArchiveBotMultiRootTest, UnparseableSensorParamsStillArchives) {
  auto const suppressWarning = SuppressLogWarning();

  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  CopyBotDirInto(kFr3AssetDir, root / "fr3");
  auto const bot = root / "fr3" / std::string(kFr3BotName);
  AddSensorWithParams(bot, root / "fr3", "not valid json {", /*writeWeights*/ true);

  auto const archiveFile = _tempDir / "out.superdex_bot_archive";
  ArchiveParams params;
  params.src = bot.string();
  params.dst = archiveFile.string();
  ArchiveBot(params, ExpectOK{});
  ASSERT_TRUE(fs::is_regular_file(archiveFile));

  // The broken file travels with the archive; the weights it would have named do not, because the
  // file naming them cannot be read.
  auto const extractedDir = ExtractBotArchiveToCache(archiveFile.string(), ExpectOK{});
  EXPECT_TRUE(ContainsFileNamed(extractedDir, "test.superdex_sensor"));
  EXPECT_FALSE(ContainsFileNamed(extractedDir, "test_weights.json"));
}

// A .superdex_root tag pointing outside the archived set is dropped, because it cannot be
// rewritten to anything inside. That is recorded: "@tag/..." references resolve in the source tree
// and will not from the archive, which is as invisible at extraction time as a missing file.
TEST_F(ArchiveBotMultiRootTest, DroppedTagIsRecorded) {
  auto const suppressWarning = SuppressLogWarning();

  auto const rootA = _tempDir / "A";
  auto const rootB = _tempDir / "B";
  // A names B via @b, but the bot in A does not reference anything in B, so B is not archived.
  WriteRootFile(rootA, R"({"@b": "../B"})");
  WriteEmptyRootMarker(rootB);
  CopyBotDirInto(kFr3AssetDir, rootA / "fr3");

  auto const archiveFile = _tempDir / "dropped_tag.superdex_bot_archive";
  ArchiveParams params;
  params.src = (rootA / "fr3" / std::string(kFr3BotName)).string();
  params.dst = archiveFile.string();
  ArchiveBot(params, ExpectOK{});

  auto const extractedDir = ExtractBotArchiveToCache(archiveFile.string(), ExpectOK{});
  auto const metadata = ReadBotArchiveMetadata(extractedDir, ExpectOK{});
  ASSERT_EQ(isize(metadata.warnings), 1);
  EXPECT_NE(std::string(metadata.warnings[0]).find("@b"), std::string::npos)
      << "warning did not name the dropped tag: " << std::string(metadata.warnings[0]);
}

// An archive built from a complete tree records no warnings, so a non-empty list is a real signal
// rather than routine noise.
TEST_F(ArchiveBotMultiRootTest, CompleteTreeArchivesWithNoWarnings) {
  auto const root = _tempDir / "root";
  WriteEmptyRootMarker(root);
  CopyBotDirInto(kFr3AssetDir, root / "fr3");

  auto const archiveFile = _tempDir / "complete.superdex_bot_archive";
  ArchiveParams params;
  params.src = (root / "fr3" / std::string(kFr3BotName)).string();
  params.dst = archiveFile.string();
  ArchiveBot(params, ExpectOK{});

  auto const extractedDir = ExtractBotArchiveToCache(archiveFile.string(), ExpectOK{});
  auto const metadata = ReadBotArchiveMetadata(extractedDir, ExpectOK{});
  EXPECT_TRUE(metadata.warnings.empty());
}

// Three-root: mod bot in A, fr3 arm in B, dg5f hand in C. A's
// .superdex_root names both via @b and @c. SaveToFile picks `@b/...` and
// `@c/...` for the two paths (rule 3: only `@tag` form avoids `..`). All
// three roots must appear in the archive so those references still resolve
// after extraction.
//
// File structure (under _tempDir):
//   A/
//     .superdex_root  ({"@b": "../B", "@c": "../C"})
//     mod_bot.superdex_bot
//   B/
//     .superdex_root  (empty)
//     fr3/              (copy of bots/arms/fr3)
//   C/
//     .superdex_root  (empty)
//     dg5f/             (copy of bots/hands/dg5f_long/right)
TEST_F(ArchiveBotMultiRootTest, ThreeRoots_BaseAndAttachInSeparateRoots) {
  auto const prevLogFn = GetLogCallback();
  SetLogCallback(nullptr);
  MOCHI_DEFER(SetLogCallback(prevLogFn));

  auto const rootA = _tempDir / "A";
  auto const rootB = _tempDir / "B";
  auto const rootC = _tempDir / "C";
  WriteEmptyRootMarker(rootB);
  WriteEmptyRootMarker(rootC);
  WriteRootFile(rootA, R"({"@b": "../B", "@c": "../C"})");
  CopyBotDirInto(kFr3AssetDir, rootB / "fr3");
  CopyBotDirInto(kDg5fAssetDir, rootC / "dg5f");
  auto const modBot = rootA / "mod_bot.superdex_bot";
  WriteFr3WithDg5fModBot(modBot, rootB / "fr3" / kFr3BotName, rootC / "dg5f" / kDg5fBotName);

  ExpectArchiveHashRoundTrip(modBot, _tempDir);
}
