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

#include <superdex_robotics/domain_randomization/domain_randomization_episode.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/json_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/reflection.h>

#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;

namespace {

// A small but structurally complete list: a scalar draw, a vector draw, and a
// per-spawn pose draw (a vector value under a dr/spawn/<name>/<class>/pose key) —
// enough to exercise every serialized field of
// DREpisodeList / DREpisode / DREpisodeEntry.
DREpisodeList MakeSampleList() {
  DREpisodeEntry scalarEntry;
  scalarEntry.key = "dr/scene/light";
  scalarEntry.scalar = true;
  scalarEntry.value.push_back(0.5_r);

  DREpisodeEntry vectorEntry;
  vectorEntry.key = "dr/robot/joint_offsets";
  vectorEntry.scalar = false;
  vectorEntry.value.push_back(0.25_r);
  vectorEntry.value.push_back(0.5_r);
  vectorEntry.value.push_back(0.75_r);

  // Per-spawn pose DR now rides as a flat [tx, ty, tz, qx, qy, qz, qw] value entry.
  DREpisodeEntry spawnPoseEntry;
  spawnPoseEntry.key = "dr/spawn/cube/object_pose/pose";
  spawnPoseEntry.scalar = false;
  spawnPoseEntry.value.push_back(1.0_r);
  spawnPoseEntry.value.push_back(2.0_r);
  spawnPoseEntry.value.push_back(3.0_r);
  spawnPoseEntry.value.push_back(0.0_r);
  spawnPoseEntry.value.push_back(0.0_r);
  spawnPoseEntry.value.push_back(0.5_r);
  spawnPoseEntry.value.push_back(0.8660254_r);

  DREpisode episode;
  episode.index = 0;
  episode.episodeSeed = "9999";
  episode.values.push_back(scalarEntry);
  episode.values.push_back(vectorEntry);
  episode.values.push_back(spawnPoseEntry);

  DREpisodeList list;
  list.schemaVersion = 2;
  list.specVersion = "spec_abc";
  list.masterSeed = "1234";
  list.episodes.push_back(episode);
  return list;
}

std::string TempPath(std::string_view name) {
  static auto const directory = CreateTempDirectory("dr_episode", ErrorAssert{});
  return (directory.Path() / std::string(name)).string();
}

} // namespace

TEST(DrEpisodeTest, SaveThenLoadRoundTrips) {
  DREpisodeList const original = MakeSampleList();
  std::string const path = TempPath("dr_episode_roundtrip.json");

  Error saveError;
  SaveDREpisodeListToFile(original, path, saveError);
  EXPECT_OK(saveError);

  Error loadError;
  DREpisodeList const loaded = LoadDREpisodeListFromFile(path, loadError);
  EXPECT_OK(loadError);

  // Top-level fields survive.
  EXPECT_EQ(loaded.schemaVersion, original.schemaVersion);
  EXPECT_EQ(std::string(loaded.specVersion), std::string(original.specVersion));
  EXPECT_EQ(std::string(loaded.masterSeed), std::string(original.masterSeed));
  ASSERT_EQ(loaded.episodes.size(), 1u);

  DREpisode const& episode = loaded.episodes[0];
  EXPECT_EQ(episode.index, 0);
  EXPECT_EQ(std::string(episode.episodeSeed), std::string("9999"));

  // Scalar, vector, and spawn-pose value entries keep key, shape flag, and payload.
  ASSERT_EQ(episode.values.size(), 3u);
  EXPECT_EQ(std::string(episode.values[0].key), std::string("dr/scene/light"));
  EXPECT_TRUE(episode.values[0].scalar);
  ASSERT_EQ(episode.values[0].value.size(), 1u);
  EXPECT_EQ(episode.values[0].value[0], 0.5);

  EXPECT_EQ(std::string(episode.values[1].key), std::string("dr/robot/joint_offsets"));
  EXPECT_FALSE(episode.values[1].scalar);
  ASSERT_EQ(episode.values[1].value.size(), 3u);
  std::array<real, 3> const expectedOffsets{0.25_r, 0.5_r, 0.75_r};
  EXPECT_TRUE(mochi::test::NearEqualSpan(episode.values[1].value, expectedOffsets));

  // The per-spawn pose draw survives as a flat [tx, ty, tz, qx, qy, qz, qw] value entry.
  EXPECT_EQ(std::string(episode.values[2].key), std::string("dr/spawn/cube/object_pose/pose"));
  EXPECT_FALSE(episode.values[2].scalar);
  ASSERT_EQ(episode.values[2].value.size(), 7u);
  std::array<real, 7> const expectedPose{1.0_r, 2.0_r, 3.0_r, 0.0_r, 0.0_r, 0.5_r, 0.8660254_r};
  EXPECT_TRUE(mochi::test::NearEqualSpan(episode.values[2].value, expectedPose));
}

TEST(DrEpisodeTest, SaveRejectsInvalidList) {
  // A scalar entry must carry exactly one value; two is a structural error the
  // save-time validation must catch before writing anything to disk.
  DREpisodeList list = MakeSampleList();
  list.episodes[0].values[0].value.push_back(1.0_r); // scalar entry now has two

  std::string const path = TempPath("dr_episode_invalid.json");
  Error error;
  SaveDREpisodeListToFile(list, path, error);
  EXPECT_NOT_OK(error);
}

TEST(DrEpisodeTest, SaveRejectsUnsupportedSchemaVersion) {
  DREpisodeList list = MakeSampleList();
  list.schemaVersion = 3; // only schema 2 is public
  Error error;
  SaveDREpisodeListToFile(list, TempPath("dr_episode_badschema.json"), error);
  EXPECT_NOT_OK(error);
}

TEST(DrEpisodeTest, SaveRejectsUnparseableSeed) {
  // ParseSeed must reject a non-uint64 seed string (the seed rides as a decimal string).
  DREpisodeList list = MakeSampleList();
  list.masterSeed = "not-a-uint64";
  Error error;
  SaveDREpisodeListToFile(list, TempPath("dr_episode_badseed.json"), error);
  EXPECT_NOT_OK(error);
}

TEST(DrEpisodeTest, SaveRejectsDuplicateValueKey) {
  DREpisodeList list = MakeSampleList();
  DREpisodeEntry duplicate;
  duplicate.key = "dr/scene/light"; // already present in MakeSampleList
  duplicate.scalar = true;
  duplicate.value.push_back(1.0_r);
  list.episodes[0].values.push_back(duplicate);
  Error error;
  SaveDREpisodeListToFile(list, TempPath("dr_episode_dupkey.json"), error);
  EXPECT_NOT_OK(error);
}

TEST(DrEpisodeTest, SchemaTwoPreservesOpaqueMetadataAndNativeValues) {
  auto list = MakeSampleList();
  list.schemaVersion = 2;
  list.specVersion = "ant-dr-v2";
  list.provenance = R"({"run_id":"ant-example","configuration_sha256":"abc"})";
  list.metadata =
      R"({"sampler":{"generator":"numpy-pcg64-v1"},"quantities":[],"lineage":{"parent":"training-1","external_id":"18446744073709551615"}})";
  real const value = static_cast<real>(1.0000000000000002);
  list.episodes[0].values[0].value[0] = value;
  auto const path = TempPath("dr_episode_metadata.json");
  Error error;
  SaveDREpisodeListToFile(list, path, error);
  auto const loaded = LoadDREpisodeListFromFile(path, error);
  EXPECT_OK(error);
  EXPECT_EQ(loaded.schemaVersion, 2);
  EXPECT_EQ(loaded.episodes[0].values[0].value[0], value);
  auto const root = ParseJsonFromFile(path, error);
  EXPECT_OK(error);
  EXPECT_EQ(
      root.get<picojson::object>()
          .at("provenance")
          .get<picojson::object>()
          .at("run_id")
          .get<std::string>(),
      "ant-example");
  EXPECT_EQ(
      root.get<picojson::object>()
          .at("metadata")
          .get<picojson::object>()
          .at("lineage")
          .get<picojson::object>()
          .at("parent")
          .get<std::string>(),
      "training-1");
  auto const& metadata = root.get<picojson::object>().at("metadata").get<picojson::object>();
  EXPECT_TRUE(metadata.at("quantities").is<picojson::array>());
  EXPECT_EQ(root.get<picojson::object>().count("lineage"), 0u);
  EXPECT_EQ(
      metadata.at("lineage").get<picojson::object>().at("external_id").get<std::string>(),
      "18446744073709551615");
  EXPECT_FALSE(loaded.metadata.empty());
  EXPECT_EQ(ParseJsonFromString(loaded.metadata, error), ParseJsonFromString(list.metadata, error));
  picojson::value reflected;
  SReflect::ToJsonValue(list, reflected);
  EXPECT_EQ(reflected.get<picojson::object>().at("metadata"), picojson::value(metadata));
  EXPECT_OK(error);
}

TEST(DrEpisodeTest, SchemaTwoRejectsTopLevelExtrasAndNonobjectMetadata) {
  bool const warningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, warningEnabled));
  for (auto const* field : {R"("profile":{})", R"("metadata":[])"}) {
    auto const path = TempPath("dr_episode_invalid_metadata.json");
    {
      std::ofstream stream(path);
      stream << R"({"schema_version":2,"spec_version":"test","master_seed":"1","episodes":[],)"
             << field << "}";
    }
    Error error;
    (void)LoadDREpisodeListFromFile(path, error);
    EXPECT_NOT_OK(error);
  }
}

TEST(DrEpisodeTest, RejectsNonfiniteValuesAndInvalidIdentity) {
  auto list = MakeSampleList();
  list.episodes[0].values[0].value[0] = std::numeric_limits<real>::infinity();
  Error finiteError;
  ValidateDREpisodeList(list, finiteError);
  EXPECT_NOT_OK(finiteError);
  list = MakeSampleList();
  auto const duplicate = list.episodes[0];
  list.episodes.push_back(duplicate);
  Error duplicateError;
  ValidateDREpisodeList(list, duplicateError);
  EXPECT_NOT_OK(duplicateError);
  list = MakeSampleList();
  list.episodes[0].index = -1;
  Error negativeError;
  ValidateDREpisodeList(list, negativeError);
  EXPECT_NOT_OK(negativeError);
}

TEST(DrEpisodeTest, LoadRejectsValuesOutsideNativeFiniteRange) {
  bool const warningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, warningEnabled));
  auto const path = TempPath("dr_episode_value_overflow.json");
  {
    std::ofstream stream(path);
    stream
        << R"({"schema_version":2,"spec_version":"test","master_seed":"1","episodes":[{"index":0,"episode_seed":"2","values":[{"key":"dr/test/value","scalar":true,"value":[)"
        << (sizeof(real) == sizeof(float) ? "1e39" : "1e309") << "]}]}]}";
  }
  Error error;
  (void)LoadDREpisodeListFromFile(path, error);
  EXPECT_NOT_OK(error);
}

TEST(DrEpisodeTest, RejectsMalformedProvenanceAndPreservesOpaqueMetadata) {
  bool const warningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, warningEnabled));
  auto list = MakeSampleList();
  for (auto const* text : {"[]", "{", R"({"run_id":1})", R"({"run_id":1e400})"}) {
    list.provenance = text;
    Error error;
    ValidateDREpisodeList(list, error);
    EXPECT_NOT_OK(error);
    if (std::string_view(text) == "[]") {
      EXPECT_NE(error.ToString().find("provenance must be an object"), std::string::npos);
    }
  }
  list.provenance.clear();
  list.metadata = "[]";
  Error metadataError;
  ValidateDREpisodeList(list, metadataError);
  EXPECT_NOT_OK(metadataError);
  EXPECT_NE(metadataError.ToString().find("metadata must be an object"), std::string::npos);
  list.metadata = R"({"schema_version":2})";
  Error error;
  ValidateDREpisodeList(list, error);
  EXPECT_OK(error);
  Error nestedError;
  ValidateDREpisodeList(list, nestedError);
  EXPECT_OK(nestedError);
}

TEST(DrEpisodeTest, SeedParsingRejectsInvalidAndOverflowValues) {
  Error error;
  EXPECT_EQ(ParseSeed("18446744073709551615", error), std::numeric_limits<uint64_t>::max());
  EXPECT_OK(error);
  EXPECT_EQ(ParseSeed("01", error), 1u);
  EXPECT_OK(error);
  for (auto const* seed : {"", "-1", "+1", "1 ", "18446744073709551616"}) {
    Error invalid;
    (void)ParseSeed(seed, invalid);
    EXPECT_NOT_OK(invalid);
  }
}

TEST(DrEpisodeTest, LoadRejectsInvalidStructuresAndIntegerOverflow) {
  bool const warningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, warningEnabled));
  for (
      auto const* json :
      {R"({"schema_version":1,"spec_version":"test","master_seed":"1","episodes":[]})",
       R"({"schema_version":4294967298,"spec_version":"test","master_seed":"1","episodes":[]})",
       R"({"schema_version":2,"spec_version":"test","master_seed":"1","episodes":[],"metadata":{"extra":1e400}})",
       R"({"schema_version":2,"spec_version":"test","master_seed":"1","episodes":[{"episode_seed":"2","values":[]}]})",
       R"({"schema_version":2,"spec_version":"test","master_seed":"1","episodes":[{"index":0,"episode_seed":"2","values":[{"key":"dr/test","scalar":true}]}]})",
       R"({"schema_version":2,"spec_version":"test","master_seed":"1","episodes":[{"index":0,"episode_seed":"2","values":"invalid"}]})"}) {
    auto const path = TempPath("dr_episode_missing_fields.json");
    std::ofstream(path) << json;
    Error error;
    EXPECT_NO_THROW((void)LoadDREpisodeListFromFile(path, error));
    EXPECT_NOT_OK(error);
  }
}

TEST(DrEpisodeTest, AppliedCoverageRoundTripsUnknownEmptyAndNonempty) {
  for (int state = 0; state < 3; ++state) {
    auto list = MakeSampleList();
    if (state) {
      list.episodes[0].appliedKeys.emplace();
      if (state == 2) {
        list.episodes[0].appliedKeys->push_back("dr/scene/light");
      }
    }
    auto const path = TempPath("dr_episode_applied.json");
    Error error;
    SaveDREpisodeListToFile(list, path, error);
    auto const loaded = LoadDREpisodeListFromFile(path, error);
    EXPECT_OK(error);
    EXPECT_EQ(bool(loaded.episodes[0].appliedKeys), state != 0);
    if (state) {
      ASSERT_TRUE(loaded.episodes[0].appliedKeys);
      EXPECT_EQ(loaded.episodes[0].appliedKeys->size(), state == 2 ? 1u : 0u);
    }
    auto root = ParseJsonFromFile(path, error);
    EXPECT_OK(error);
    auto const& fields = root.get<picojson::object>()
                             .at("episodes")
                             .get<picojson::array>()[0]
                             .get<picojson::object>();
    EXPECT_EQ(fields.count("applied_keys"), state != 0 ? 1u : 0u);
  }
}

TEST(DrEpisodeTest, RejectsAppliedKeysWithoutValuesOrWithDuplicates) {
  for (auto const* key : {"dr/missing", "dr/scene/light"}) {
    auto list = MakeSampleList();
    list.episodes[0].appliedKeys.emplace();
    list.episodes[0].appliedKeys->push_back("dr/scene/light");
    list.episodes[0].appliedKeys->push_back(key);
    Error error;
    ValidateDREpisodeList(list, error);
    EXPECT_NOT_OK(error);
  }
}

TEST(DrEpisodeTest, CanonicalSeedsRequiredButDistinctIndicesMayShareSeeds) {
  auto list = MakeSampleList();
  list.masterSeed = "001234";
  Error paddedMaster;
  ValidateDREpisodeList(list, paddedMaster);
  EXPECT_NOT_OK(paddedMaster);
  list.masterSeed = "1234";
  list.episodes[0].episodeSeed = "009999";
  Error paddedEpisode;
  ValidateDREpisodeList(list, paddedEpisode);
  EXPECT_NOT_OK(paddedEpisode);
  list.episodes[0].episodeSeed = "9999";
  auto duplicate = list.episodes[0];
  duplicate.index = 1;
  list.episodes.push_back(duplicate);
  Error sharedSeed;
  ValidateDREpisodeList(list, sharedSeed);
  EXPECT_OK(sharedSeed);
}

TEST(DrEpisodeTest, SignedIndicesRoundTripThroughStandardReflection) {
  for (int64_t index :
       {int64_t{0},
        int64_t{2147483648},
        int64_t{9007199254740993},
        std::numeric_limits<int64_t>::max()}) {
    auto list = MakeSampleList();
    list.schemaVersion = 2;
    list.episodes[0].index = index;
    auto const path = TempPath("dr_episode_wide_index.json");
    Error error;
    SaveDREpisodeListToFile(list, path, error);
    auto const loaded = LoadDREpisodeListFromFile(path, error);
    ASSERT_TRUE(error.IsOK());
    ASSERT_EQ(loaded.episodes.size(), 1u);
    EXPECT_EQ(loaded.episodes[0].index, index);
    auto const root = ParseJsonFromFile(path, error);
    EXPECT_OK(error);
    auto const& wireIndex = root.get<picojson::object>()
                                .at("episodes")
                                .get<picojson::array>()[0]
                                .get<picojson::object>()
                                .at("index");
    ASSERT_TRUE(wireIndex.is<int64_t>());
    EXPECT_EQ(wireIndex.get<int64_t>(), index);
    picojson::value reflected;
    SReflect::ToJsonValue(list, reflected);
    EXPECT_EQ(reflected, root);
  }
}

TEST(DrEpisodeTest, SchemaTwoRejectsMalformedAndDuplicateIndices) {
  bool const warningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, warningEnabled));
  for (auto const* index :
       {"-1",
        "\"0\"",
        "1.5",
        "null",
        "true",
        "\"\"",
        "\"01\"",
        "\"-1\"",
        "\"+1\"",
        "\"1 \"",
        "\" 1\"",
        "\"1e3\"",
        "9223372036854775808"}) {
    auto const path = TempPath("dr_episode_invalid_wide_index.json");
    {
      std::ofstream stream(path);
      stream
          << R"({"schema_version":2,"spec_version":"test","master_seed":"1","episodes":[{"index":)"
          << index << R"(,"episode_seed":"2","values":[]}]})";
    }
    Error error;
    (void)LoadDREpisodeListFromFile(path, error);
    EXPECT_NOT_OK(error);
  }
  auto list = MakeSampleList();
  list.schemaVersion = 2;
  list.episodes[0].index = int64_t{9007199254740993};
  auto duplicate = list.episodes[0];
  duplicate.episodeSeed = "10000";
  list.episodes.push_back(duplicate);
  Error error;
  ValidateDREpisodeList(list, error);
  EXPECT_NOT_OK(error);
}

TEST(DrEpisodeTest, JsonStringReflectionAcceptsObjectTextAndEmitsObjects) {
  auto const path = TempPath("dr_episode_jsonstring.json");
  std::ofstream(path)
      << R"({"schema_version":2,"spec_version":"test","master_seed":"1","episodes":[],"metadata":"{}","provenance":"{}"})";
  Error error;
  auto const list = LoadDREpisodeListFromFile(path, error);
  EXPECT_OK(error);
  SaveDREpisodeListToFile(list, path, error);
  auto const root = ParseJsonFromFile(path, error);
  EXPECT_OK(error);
  EXPECT_TRUE(root.get<picojson::object>().at("metadata").is<picojson::object>());
  EXPECT_TRUE(root.get<picojson::object>().at("provenance").is<picojson::object>());
}
