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

#include <superdex_robotics/utils/asset_snapshot.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_physics/utils/mochi_path_remap.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <string>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

namespace fs = std::filesystem;

class AssetSnapshotTest : public testing::Test {
 protected:
  void TearDown() override {
    ClearPathRemap();
  }

  /* A snapshot of `//scenes/s.mochi_bot_scene` pinning it and `//params/p.json`. */
  static AssetSnapshot MakeSnapshot() {
    AssetSnapshot snapshot;
    snapshot.scene = "//scenes/s.mochi_bot_scene";
    snapshot.createdAt = 1792520967;
    snapshot.sourceRevision = "abc123";
    snapshot.files.push_back(
        {.file = DynamicString("//scenes/s.mochi_bot_scene"),
         .version = DynamicString("//scenes/s-1792441200.mochi_bot_scene")});
    snapshot.files.push_back(
        {.file = DynamicString("//params/p.json"),
         .version = DynamicString("//params/p-1791002000.json")});
    return snapshot;
  }

  /* Lay out the root with every file MakeSnapshot names, tips and versions. */
  void WriteSnapshotFiles() {
    mochi::WriteFile(_root / kRootMarkerFile, std::string_view{}, ExpectOK{});
    for (auto const* file :
         {"scenes/s.mochi_bot_scene",
          "scenes/s-1792441200.mochi_bot_scene",
          "params/p.json",
          "params/p-1791002000.json"}) {
      mochi::WriteFile(_root / file, std::string_view{"{}"}, ExpectOK{});
    }
  }

  /* Applying `snapshot` fails. Path resolution logs before it errors, and the test framework fails
   * any test that logs an error, so logging is silenced for the attempt. */
  void ExpectApplyFails(AssetSnapshot const& snapshot) {
    auto const prevLogFn = GetLogCallback();
    SetLogCallback([](LogChannel, char const*, char const*, int) {});
    Error error;
    (void)ApplyAssetSnapshot(snapshot, _root, error);
    SetLogCallback(prevLogFn);
    EXPECT_NOT_OK(error);
    EXPECT_FALSE(IsPathRemapActive());
  }

  mochi::TempDirCleanup _tempDirCleanup =
      mochi::CreateTempDirectory("asset_snapshot_test", ExpectOK{});
  fs::path _root = fs::path(_tempDirCleanup.Path()) / "root";
};

} // namespace

TEST_F(AssetSnapshotTest, SaveThenLoadRoundTrips) {
  auto const path = (fs::path(_tempDirCleanup.Path()) / "s.superdex_scene_snapshot").string();
  auto const original = MakeSnapshot();
  SaveAssetSnapshot(original, path, ExpectOK{});

  auto const loaded = LoadAssetSnapshot(path, ExpectOK{});
  EXPECT_EQ(std::string(loaded.scene), std::string(original.scene));
  EXPECT_EQ(loaded.createdAt, original.createdAt);
  EXPECT_EQ(std::string(loaded.sourceRevision), std::string(original.sourceRevision));
  // Files are a JSON object, so they come back keyed rather than in insertion order.
  auto const asMap = [](AssetSnapshot const& snapshot) {
    std::map<std::string, std::string> files;
    for (auto const& entry : snapshot.files) {
      files.emplace(std::string(entry.file), std::string(entry.version));
    }
    return files;
  };
  EXPECT_EQ(asMap(loaded), asMap(original));
}

TEST_F(AssetSnapshotTest, ApplyReturnsTheScenesPath) {
  WriteSnapshotFiles();
  auto const scene = ApplyAssetSnapshot(MakeSnapshot(), _root, ExpectOK{});
  EXPECT_EQ(
      std::string(scene), NormalizeBotPath(_root / "scenes/s.mochi_bot_scene").generic_string());
}

TEST_F(AssetSnapshotTest, ApplyRoutesReferencesToVersionedCopies) {
  WriteSnapshotFiles();
  (void)ApplyAssetSnapshot(MakeSnapshot(), _root, ExpectOK{});
  auto const referencing = _root / "bots" / "b.superdex_bot";
  EXPECT_EQ(
      ResolveBotPath("//params/p.json", referencing, kBotPathMaxParentDepth, ExpectOK{}),
      NormalizeBotPath(_root / "params" / "p-1791002000.json"));
}

/* A file under the asset root that the snapshot does not pin fails rather than loading its tip. */
TEST_F(AssetSnapshotTest, ApplyRefusesAFileItDoesNotPin) {
  WriteSnapshotFiles();
  mochi::WriteFile(_root / "params/other.json", std::string_view{"{}"}, ExpectOK{});
  (void)ApplyAssetSnapshot(MakeSnapshot(), _root, ExpectOK{});
  auto const suppressError = SuppressLogError();
  Error error;
  (void)ResolveBotPath(
      "//params/other.json", _root / "bots" / "b.superdex_bot", kBotPathMaxParentDepth, error);
  EXPECT_NOT_OK(error);
}

TEST_F(AssetSnapshotTest, ApplyLeavesFilesOutsideItsRootsUnchanged) {
  WriteSnapshotFiles();
  auto const outside = fs::path(_tempDirCleanup.Path()) / "outside" / "o.json";
  mochi::WriteFile(outside, std::string_view{"{}"}, ExpectOK{});
  (void)ApplyAssetSnapshot(MakeSnapshot(), _root, ExpectOK{});
  EXPECT_EQ(std::string(RemapPath(outside.generic_string())), outside.generic_string());
}

TEST_F(AssetSnapshotTest, LoadRejectsAMissingOrUnknownFormatVersion) {
  auto const path = (fs::path(_tempDirCleanup.Path()) / "s.superdex_scene_snapshot").string();
  SaveAssetSnapshot(MakeSnapshot(), path, ExpectOK{});
  auto const written = mochi::ReadFileString(path, ExpectOK{});
  auto const versionKey = std::string("\"formatVersion\": 1");
  ASSERT_NE(written.find(versionKey), std::string::npos);
  for (auto const* replacement : {"\"formatVersion\": 2", "\"formatVersionX\": 1"}) {
    auto edited = written;
    edited.replace(edited.find(versionKey), versionKey.size(), replacement);
    mochi::WriteFile(path, std::string_view{edited}, ExpectOK{});
    SCOPED_TRACE(replacement);
    Error error;
    (void)LoadAssetSnapshot(path, error);
    EXPECT_NOT_OK(error);
  }
}

TEST_F(AssetSnapshotTest, ApplyRejectsAVersionThatIsAlsoAPinnedFile) {
  WriteSnapshotFiles();
  auto snapshot = MakeSnapshot();
  snapshot.files.push_back(
      {.file = DynamicString("//params/p-1791002000.json"),
       .version = DynamicString("//params/p-1791002000-1792000000.json")});
  mochi::WriteFile(
      _root / "params/p-1791002000-1792000000.json", std::string_view{"{}"}, ExpectOK{});
  ExpectApplyFails(snapshot);
}

/* A snapshot that pins nothing still covers its asset root, so no tip under the root loads. */
TEST_F(AssetSnapshotTest, AnEmptySnapshotRefusesFilesUnderTheRoot) {
  WriteSnapshotFiles();
  auto snapshot = MakeSnapshot();
  snapshot.files.clear();
  (void)ApplyAssetSnapshot(snapshot, _root, ExpectOK{});
  EXPECT_TRUE(IsPathRemapActive());
  auto const suppressError = SuppressLogError();
  EXPECT_EQ(std::string(RemapPath((_root / "params/p.json").generic_string())), "");
}

/* A snapshot pinning the .superdex_root applies while the root matches its version. */
TEST_F(AssetSnapshotTest, ApplyAcceptsAnUnchangedPinnedRootMarker) {
  WriteSnapshotFiles();
  mochi::WriteFile(_root / "-1790000000.superdex_root", std::string_view{}, ExpectOK{});
  auto snapshot = MakeSnapshot();
  snapshot.files.push_back(
      {.file = DynamicString("//.superdex_root"),
       .version = DynamicString("//-1790000000.superdex_root")});
  (void)ApplyAssetSnapshot(snapshot, _root, ExpectOK{});
  EXPECT_TRUE(IsPathRemapActive());
}

/* Retargeting a tag (or any other edit to a pinned .superdex_root) makes the snapshot fail. */
TEST_F(AssetSnapshotTest, ApplyRejectsAChangedPinnedRootMarker) {
  WriteSnapshotFiles();
  mochi::WriteFile(_root / "-1790000000.superdex_root", std::string_view{}, ExpectOK{});
  mochi::WriteFile(_root / kRootMarkerFile, std::string_view{"{}"}, ExpectOK{});
  auto snapshot = MakeSnapshot();
  snapshot.files.push_back(
      {.file = DynamicString("//.superdex_root"),
       .version = DynamicString("//-1790000000.superdex_root")});
  ExpectApplyFails(snapshot);
}

TEST_F(AssetSnapshotTest, ApplyRejectsAMissingVersion) {
  WriteSnapshotFiles();
  fs::remove(_root / "params/p-1791002000.json");
  ExpectApplyFails(MakeSnapshot());
}

/* A version's own `./` references resolve from its directory, so it must sit beside its tip. */
TEST_F(AssetSnapshotTest, ApplyRejectsAVersionInAnotherDirectory) {
  WriteSnapshotFiles();
  mochi::WriteFile(_root / "elsewhere/p-1791002000.json", std::string_view{"{}"}, ExpectOK{});
  auto snapshot = MakeSnapshot();
  snapshot.files.back().version = "//elsewhere/p-1791002000.json";
  ExpectApplyFails(snapshot);
}
