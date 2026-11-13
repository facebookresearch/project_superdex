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

#include <superdex_recorder/episode_reader.h>
#include <superdex_recorder/recorder.h>
#include <superdex_recorder/series.h>

#include <gtest/gtest.h>
#include <rerun.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace superdex::recorder {
namespace {

using mochi::operator""_r;

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    auto const suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
        ("superdex_episode_reader_test_" + std::to_string(suffix) + "_" +
         std::to_string(std::random_device{}()));
    std::filesystem::create_directories(_path);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }

  [[nodiscard]] std::filesystem::path const& Path() const {
    return _path;
  }

 private:
  std::filesystem::path _path;
};

/* Records and commits one episode, returning its file. */
std::filesystem::path RecordEpisode(
    TemporaryDirectory const& directory,
    std::function<void(Episode&, mochi::Error&)> const& record) {
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  EXPECT_NE(episode, nullptr) << error.ToString();
  if (episode == nullptr) {
    return {};
  }
  record(*episode, error);
  episode->Commit(error);
  EXPECT_TRUE(error.IsOK()) << error.ToString();
  return episode->GetOutputPath();
}

/* Writes a file with the Rerun SDK, which can write rows that the recorder never writes. */
std::filesystem::path WriteRerunFile(
    TemporaryDirectory const& directory,
    std::function<void(rerun::RecordingStream const&)> const& write) {
  auto const path = directory.Path() / "rerun.rrd";
  rerun::RecordingStream const stream{"superdex_episode_reader_test"};
  EXPECT_TRUE(stream.save(path.string()).is_ok());
  write(stream);
  EXPECT_TRUE(stream.flush_blocking().is_ok());
  return path;
}

template <typename Component>
rerun::ComponentBatch MakeBatch(Component const& value, char const* component) {
  return rerun::ComponentBatch::from_loggable(value, rerun::ComponentDescriptor{component}).value;
}

std::unique_ptr<EpisodeReader> OpenEpisode(std::filesystem::path const& path) {
  mochi::Error error;
  auto reader = EpisodeReader::Open(path, error);
  EXPECT_TRUE(error.IsOK()) << error.ToString();
  return reader;
}

TEST(EpisodeReaderTest, ReadsScalarSeries) {
  TemporaryDirectory directory;
  auto const path = RecordEpisode(directory, [](Episode& episode, mochi::Error& error) {
    auto* const series = episode.CreateScalarSeries(
        "data/test", {{"superdex.archetypes.Test", "superdex.test.values"}}, error);
    ASSERT_NE(series, nullptr);
    std::vector<std::vector<mochi::real>> const rows{{1_r, 2_r}, {}, {3_r}};
    for (uint64_t frame = 0; frame < rows.size(); ++frame) {
      std::array<mochi::Span<mochi::real const>, 1> const fields{rows[frame]};
      series->Append({.frame = frame, .simulationTimeSec = 0.25 * frame}, fields, error);
    }
  });
  auto const reader = OpenEpisode(path);
  ASSERT_NE(reader, nullptr);
  mochi::Error error;

  ScalarRows const rows = reader->ReadScalars("data/test", "superdex.test.values", error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(rows.frames, (std::vector<int64_t>{0, 1, 2}));
  EXPECT_EQ(rows.simulationTimesNs, (std::vector<int64_t>{0, 250'000'000, 500'000'000}));
  EXPECT_EQ(rows.offsets, (std::vector<size_t>{0, 2, 2, 3}));
  EXPECT_EQ(rows.values, (std::vector<double>{1, 2, 3}));
  EXPECT_TRUE(reader->HasComponent("data/test", "superdex.test.values", false));
  EXPECT_FALSE(reader->HasComponent("data/test", "superdex.test.values", true));
}

TEST(EpisodeReaderTest, ReadsPlotSeriesAtDoublePrecision) {
  TemporaryDirectory directory;
  // Not representable as a float, so a 32-bit round trip would change it.
  constexpr double kValue = 0.1;
  auto const path = RecordEpisode(directory, [](Episode& episode, mochi::Error& error) {
    auto* const series = episode.CreatePlotSeries("data/plot", error);
    ASSERT_NE(series, nullptr);
    series->Append({.frame = 0, .simulationTimeSec = 0.0}, kValue, error);
    series->Append({.frame = 1, .simulationTimeSec = 0.5}, -kValue, error);
  });
  auto const reader = OpenEpisode(path);
  ASSERT_NE(reader, nullptr);
  mochi::Error error;

  ScalarRows const rows = reader->ReadScalars("data/plot", "Scalars:scalars", error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(rows.frames, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(rows.values, (std::vector<double>{kValue, -kValue}));
}

TEST(EpisodeReaderTest, ReadsTransformsAsFlatRows) {
  TemporaryDirectory directory;
  auto const path = RecordEpisode(directory, [](Episode& episode, mochi::Error& error) {
    auto* const series = episode.CreateTransformSeries("world/box", error);
    ASSERT_NE(series, nullptr);
    series->Append(
        {.frame = 0, .simulationTimeSec = 0.0}, {1_r, 2_r, 3_r}, {0_r, 0_r, 0_r, 1_r}, error);
    series->Append(
        {.frame = 1, .simulationTimeSec = 0.5}, {4_r, 5_r, 6_r}, {0_r, 0_r, 1_r, 0_r}, error);
  });
  auto const reader = OpenEpisode(path);
  ASSERT_NE(reader, nullptr);
  mochi::Error error;

  ScalarRows const translations =
      reader->ReadScalars("world/box", "Transform3D:translation", error);
  ScalarRows const rotations = reader->ReadScalars("world/box", "Transform3D:quaternion", error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(translations.values, (std::vector<double>{1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(rotations.values, (std::vector<double>{0, 0, 0, 1, 0, 0, 1, 0}));
}

TEST(EpisodeReaderTest, ReadsConstants) {
  TemporaryDirectory directory;
  std::vector<std::string> const names{"left", "right"};
  std::vector<double> const values{0.5, -1.5};
  auto const path = RecordEpisode(directory, [&](Episode& episode, mochi::Error& error) {
    episode.WriteConstantTexts(
        "data/constants", {"superdex.archetypes.Test", "superdex.test.names"}, names, error);
    episode.WriteConstantScalars(
        "data/constants", {"superdex.archetypes.Test", "superdex.test.values"}, values, error);
  });
  auto const reader = OpenEpisode(path);
  ASSERT_NE(reader, nullptr);
  mochi::Error error;

  auto const readNames = reader->ReadStaticTexts("data/constants", "superdex.test.names", error);
  auto const readValues =
      reader->ReadStaticScalars("data/constants", "superdex.test.values", error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(readNames, names);
  EXPECT_EQ(readValues, values);
}

TEST(EpisodeReaderTest, RejectsMissingAndMistypedComponents) {
  TemporaryDirectory directory;
  auto const path = RecordEpisode(directory, [](Episode& episode, mochi::Error& error) {
    episode.WriteConstantText(
        "data/test", {"superdex.archetypes.Test", "superdex.test.name"}, "name", error);
  });
  auto const reader = OpenEpisode(path);
  ASSERT_NE(reader, nullptr);
  mochi::Error missingError;
  mochi::Error staticAsTemporalError;
  mochi::Error textAsNumbersError;

  (void)reader->ReadScalars("data/test", "superdex.test.other", missingError);
  (void)reader->ReadScalars("data/test", "superdex.test.name", staticAsTemporalError);
  (void)reader->ReadStaticScalars("data/test", "superdex.test.name", textAsNumbersError);

  EXPECT_STREQ(
      missingError.GetDescription(),
      "Episode has no temporal rows of 'superdex.test.other' on 'data/test'.");
  EXPECT_STREQ(
      staticAsTemporalError.GetDescription(),
      "Episode has no temporal rows of 'superdex.test.name' on 'data/test'.");
  EXPECT_STREQ(
      textAsNumbersError.GetDescription(),
      "Component 'superdex.test.name' on 'data/test' does not hold floats or doubles.");
}

TEST(EpisodeReaderTest, OrdersRowsByFrameKeepingRecordedOrder) {
  TemporaryDirectory directory;
  auto const path = WriteRerunFile(directory, [](rerun::RecordingStream const& stream) {
    stream.log_static(
        "episode",
        MakeBatch(rerun::components::Text{"superdex.episode"}, "superdex.episode.schema_name"),
        MakeBatch(rerun::components::Text{"1.7"}, "superdex.episode.schema_version"));
    // Two chunks, the first of them out of frame order.
    for (auto const& [frames, values] :
         {std::pair{std::vector<int64_t>{3, 1}, std::vector<double>{30, 10}},
          std::pair{std::vector<int64_t>{1}, std::vector<double>{11}}}) {
      auto const column = rerun::ComponentColumn::from_loggable(
          rerun::Collection<rerun::components::Scalar>::take_ownership(
              std::vector<rerun::components::Scalar>{values.begin(), values.end()}),
          rerun::ComponentDescriptor{"superdex.test.value"});
      stream.send_columns(
          "data/test",
          {rerun::TimeColumn::from_sequence("frame", frames),
           rerun::TimeColumn::from_duration_secs("sim_time", std::vector<double>(frames.size()))},
          rerun::Collection<rerun::ComponentColumn>{column.value});
    }
  });
  auto const reader = OpenEpisode(path);
  ASSERT_NE(reader, nullptr);
  mochi::Error error;

  ScalarRows const rows = reader->ReadScalars("data/test", "superdex.test.value", error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_EQ(rows.frames, (std::vector<int64_t>{1, 1, 3}));
  EXPECT_EQ(rows.values, (std::vector<double>{10, 11, 30}));
}

TEST(EpisodeReaderTest, RejectsFilesThatAreNotWholeEpisodes) {
  TemporaryDirectory directory;
  auto const episodePath = RecordEpisode(directory, [](Episode&, mochi::Error&) {});
  std::ifstream episodeFile{episodePath, std::ios::binary};
  std::vector<char> const bytes{
      std::istreambuf_iterator<char>{episodeFile}, std::istreambuf_iterator<char>{}};
  ASSERT_GT(bytes.size(), 64U);
  auto const writeFile = [&](std::string const& name, std::vector<char> const& content) {
    auto const path = directory.Path() / name;
    std::ofstream{path, std::ios::binary}.write(
        content.data(), static_cast<std::streamsize>(content.size()));
    return path;
  };
  std::vector<char> otherRerunVersion = bytes;
  // The stream header holds the writer's Rerun version at bytes 4 to 6.
  otherRerunVersion[5] = 34;
  otherRerunVersion[6] = 0;
  std::vector<std::filesystem::path> const rejected{
      directory.Path() / "missing.rrd",
      writeFile("text.rrd", {'n', 'o', 't', ' ', 'a', 'n', ' ', 'r', 'r', 'd', '.', '.'}),
      writeFile("half.rrd", {bytes.begin(), bytes.begin() + bytes.size() / 2}),
      // Rerun ends a stream with an end-of-stream message and a 32-byte footer.
      writeFile("no_end.rrd", {bytes.begin(), bytes.end() - 33}),
      writeFile("version.rrd", otherRerunVersion),
      WriteRerunFile(
          directory,
          [](rerun::RecordingStream const& stream) {
            stream.log("points", rerun::Points3D{{{0.0F, 0.0F, 0.0F}}});
          }),
  };

  for (auto const& path : rejected) {
    mochi::Error error;

    EXPECT_EQ(EpisodeReader::Open(path, error), nullptr) << path;
    EXPECT_FALSE(error.IsOK()) << path;
  }
  // Each description names the file and what is wrong with it.
  auto const describe = [](std::filesystem::path const& path) {
    mochi::Error error;
    (void)EpisodeReader::Open(path, error);
    return std::string{error.GetDescription()};
  };
  std::string const versionPath = (directory.Path() / "version.rrd").string();
  EXPECT_EQ(
      describe(versionPath),
      "'" + versionPath +
          "' was written by Rerun 0.34.0, but this reader reads Rerun 0.33 files. "
          "`rerun rrd migrate` upgrades older files.");
  std::string const halfPath = (directory.Path() / "half.rrd").string();
  EXPECT_EQ(describe(halfPath), "'" + halfPath + "' was cut short.");
}

} // namespace
} // namespace superdex::recorder
