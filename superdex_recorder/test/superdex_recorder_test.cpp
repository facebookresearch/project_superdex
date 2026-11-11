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

#include <superdex_recorder/recorder.h>
#include <superdex_recorder/series.h>
#include <superdex_recorder/source_recorder.h>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <random>
#include <regex>
#include <string>
#include <vector>

namespace superdex::recorder {
namespace {

using mochi::operator""_r;

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    auto const suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
        ("superdex_recorder_test_" + std::to_string(suffix) + "_" +
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

uint64_t UnixMilliseconds() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

void AppendSample(Episode& episode, mochi::Error& error) {
  auto* const series = episode.CreateScalarSeries(
      "data/test", {{"superdex.archetypes.Test", "superdex.test.value"}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  std::array<mochi::real, 1> const value{static_cast<mochi::real>(42)};
  std::array<mochi::Span<mochi::real const>, 1> const fields{value};
  series->Append({.frame = 0, .simulationTimeSec = 0.0}, fields, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
}

TEST(SuperDexRecorderTest, CommitPublishesOnlyFinalFile) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  auto episode = recorder.BeginEpisode({.properties = {{"task", "test"}}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto const outputPath = episode->GetOutputPath();
  EXPECT_FALSE(std::filesystem::exists(outputPath));
  ASSERT_NO_FATAL_FAILURE(AppendSample(*episode, error));

  episode->Commit(error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_FALSE(episode->IsActive());
  std::vector<std::filesystem::path> const expectedFiles{outputPath};
  std::vector<std::filesystem::path> const files{
      std::filesystem::directory_iterator{directory.Path()}, std::filesystem::directory_iterator{}};
  EXPECT_EQ(files, expectedFiles);
  EXPECT_GT(std::filesystem::file_size(outputPath), 0);
}

TEST(SuperDexRecorderTest, CommitFailureLeavesNoPartialFile) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  std::filesystem::path outputPath;
  {
    auto episode = recorder.BeginEpisode({}, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_NE(episode, nullptr);
    ASSERT_NO_FATAL_FAILURE(AppendSample(*episode, error));
    outputPath = episode->GetOutputPath();
    // No platform renames a file over a non-empty directory.
    std::filesystem::create_directories(outputPath / "blocker");

    episode->Commit(error);

    EXPECT_FALSE(error.IsOK());
    EXPECT_FALSE(episode->IsActive());
  }

  std::vector<std::filesystem::path> const expectedFiles{outputPath};
  std::vector<std::filesystem::path> const files{
      std::filesystem::directory_iterator{directory.Path()}, std::filesystem::directory_iterator{}};
  EXPECT_EQ(files, expectedFiles);
}

TEST(SuperDexRecorderTest, AbortDiscardsEpisode) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto const outputPath = episode->GetOutputPath();

  episode->Abort("rejected", error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_FALSE(episode->IsActive());
  EXPECT_FALSE(std::filesystem::exists(outputPath));
  EXPECT_TRUE(std::filesystem::is_empty(directory.Path()));
}

TEST(SuperDexRecorderTest, AbortDiscardsEpisodeAfterRecordingError) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  error.SetFirstError("recording failed", __FILE__, __LINE__);

  episode->Abort("rejected after recording failure", error);

  EXPECT_FALSE(episode->IsActive());
  EXPECT_TRUE(std::filesystem::is_empty(directory.Path()));
}

TEST(SuperDexRecorderTest, AppendAfterAbortIsRejected) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series = episode->CreateScalarSeries(
      "data/test", {{"superdex.archetypes.Test", "superdex.test.value"}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  episode->Abort("rejected", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  std::array<mochi::real, 1> const value{static_cast<mochi::real>(42)};
  std::array<mochi::Span<mochi::real const>, 1> const fields{value};

  series->Append({.frame = 0, .simulationTimeSec = 0.0}, fields, error);

  EXPECT_FALSE(error.IsOK());
}

TEST(SuperDexRecorderTest, DestructionDiscardsUncommittedEpisode) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  {
    auto episode = recorder.BeginEpisode({}, error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    ASSERT_NE(episode, nullptr);
    ASSERT_NO_FATAL_FAILURE(AppendSample(*episode, error));
  }

  EXPECT_TRUE(std::filesystem::is_empty(directory.Path()));
}

TEST(SuperDexRecorderTest, EpisodesHaveIndependentIdsAndFiles) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  auto first = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(first, nullptr);
  auto second = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first->GetEpisodeId(), second->GetEpisodeId());

  first->Commit(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  second->Commit(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(std::filesystem::exists(first->GetOutputPath()));
  EXPECT_TRUE(std::filesystem::exists(second->GetOutputPath()));
}

TEST(SuperDexRecorderTest, EpisodeIdIsUuidV7WithStartTimestamp) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;

  uint64_t const before = UnixMilliseconds();
  auto episode = recorder.BeginEpisode({}, error);
  uint64_t const after = UnixMilliseconds();
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);

  std::string const& id = episode->GetEpisodeId();
  ASSERT_TRUE(
      std::regex_match(
          id, std::regex{"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"}))
      << id;
  uint64_t const timestamp = std::stoull(id.substr(0, 8) + id.substr(9, 4), nullptr, 16);
  EXPECT_GE(timestamp, before);
  EXPECT_LE(timestamp, after);
}

TEST(SuperDexRecorderTest, BeginEpisodeRejectsEmptyPropertyKeyBeforeCreatingDirectory) {
  TemporaryDirectory directory;
  auto const outputDirectory = directory.Path() / "recordings";
  Recorder recorder{{.outputDirectory = outputDirectory}};
  mochi::Error error;

  auto episode = recorder.BeginEpisode({.properties = {{"", "value"}}}, error);

  EXPECT_EQ(episode, nullptr);
  EXPECT_FALSE(error.IsOK());
  EXPECT_FALSE(std::filesystem::exists(outputDirectory));
}

TEST(SuperDexRecorderTest, ScalarSeriesRejectsRepeatedFrames) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series = episode->CreateScalarSeries(
      "data/test", {{"superdex.archetypes.Test", "superdex.test.value"}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  std::array<mochi::real, 1> const value{static_cast<mochi::real>(42)};
  std::array<mochi::Span<mochi::real const>, 1> const fields{value};
  series->Append({.frame = 1, .simulationTimeSec = 0.1}, fields, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  mochi::Error appendError;
  series->Append({.frame = 1, .simulationTimeSec = 0.2}, fields, appendError);

  EXPECT_FALSE(appendError.IsOK());
}

TEST(SuperDexRecorderTest, SeriesRejectsMixingSamplesWithAndWithoutWallTime) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  ComponentField const field{"superdex.archetypes.Test", "superdex.test.value"};
  auto* const timed = episode->CreateScalarSeries("data/timed", {field}, error);
  auto* const untimed = episode->CreateScalarSeries("data/untimed", {field}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  std::array<mochi::real, 1> const value{static_cast<mochi::real>(42)};
  std::array<mochi::Span<mochi::real const>, 1> const fields{value};
  timed->Append({.frame = 1, .simulationTimeSec = 0.1, .wallTimeSec = 0.1}, fields, error);
  untimed->Append({.frame = 1, .simulationTimeSec = 0.1}, fields, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  mochi::Error timedError;
  timed->Append({.frame = 2, .simulationTimeSec = 0.2}, fields, timedError);
  mochi::Error untimedError;
  untimed->Append({.frame = 2, .simulationTimeSec = 0.2, .wallTimeSec = 0.1}, fields, untimedError);

  EXPECT_FALSE(timedError.IsOK());
  EXPECT_FALSE(untimedError.IsOK());
}

TEST(SuperDexRecorderTest, SeriesRejectsDecreasingWallTime) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series = episode->CreateScalarSeries(
      "data/test", {{"superdex.archetypes.Test", "superdex.test.value"}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  std::array<mochi::real, 1> const value{static_cast<mochi::real>(42)};
  std::array<mochi::Span<mochi::real const>, 1> const fields{value};
  series->Append({.frame = 1, .simulationTimeSec = 0.1, .wallTimeSec = 0.2}, fields, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  mochi::Error appendError;
  series->Append({.frame = 2, .simulationTimeSec = 0.2, .wallTimeSec = 0.1}, fields, appendError);

  EXPECT_FALSE(appendError.IsOK());
}

TEST(SuperDexRecorderTest, EpisodeWallTimeStartsAtTheEpisodeAndNeverDecreases) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);

  double const first = episode->GetWallTimeSec();
  double const second = episode->GetWallTimeSec();

  EXPECT_GE(first, 0.0);
  EXPECT_GE(second, first);
}

TEST(SuperDexRecorderTest, CreateSeriesRejectsExistingEntityPath) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  ASSERT_NE(episode->CreateTransformSeries("world/test", error), nullptr);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  mochi::Error scalarError;
  auto* const scalarSeries = episode->CreateScalarSeries(
      "world/test", {{"superdex.archetypes.Test", "superdex.test.value"}}, scalarError);
  mochi::Error transformError;
  auto* const transformSeries = episode->CreateTransformSeries("world/test", transformError);
  mochi::Error arrowError;
  auto* const arrowSeries = episode->CreateArrow3DSeries("world/test", arrowError);
  mochi::Error pointsError;
  auto* const pointsSeries = episode->CreatePoints3DSeries("world/test", pointsError);
  mochi::Error textError;
  auto* const textSeries = episode->CreateTextSeries(
      "world/test", {"rerun.archetypes.TextLog", "TextLog:text"}, textError);

  EXPECT_EQ(scalarSeries, nullptr);
  EXPECT_FALSE(scalarError.IsOK());
  EXPECT_EQ(transformSeries, nullptr);
  EXPECT_FALSE(transformError.IsOK());
  EXPECT_EQ(arrowSeries, nullptr);
  EXPECT_FALSE(arrowError.IsOK());
  EXPECT_EQ(pointsSeries, nullptr);
  EXPECT_FALSE(pointsError.IsOK());
  EXPECT_EQ(textSeries, nullptr);
  EXPECT_FALSE(textError.IsOK());
}

TEST(SuperDexRecorderTest, TransformSeriesChecksTheTimeOfUnchangedTransforms) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series = episode->CreateTransformSeries("world/test", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  auto const transform = mochi::TransformRT::Identity();
  series->Append({.frame = 1, .simulationTimeSec = 0.1}, transform, error);
  series->Append({.frame = 2, .simulationTimeSec = 0.2}, transform, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(series->GetNumSamples(), 1U);

  mochi::Error repeatedFrameError;
  series->Append({.frame = 2, .simulationTimeSec = 0.3}, transform, repeatedFrameError);
  mochi::Error earlierTimeError;
  series->Append({.frame = 3, .simulationTimeSec = 0.15}, transform, earlierTimeError);

  EXPECT_FALSE(repeatedFrameError.IsOK());
  EXPECT_FALSE(earlierTimeError.IsOK());
}

TEST(SuperDexRecorderTest, TransformSeriesRejectsUnchangedTransformsAfterCommit) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series = episode->CreateTransformSeries("world/test", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  auto const transform = mochi::TransformRT::Identity();
  series->Append({.frame = 1, .simulationTimeSec = 0.1}, transform, error);
  episode->Commit(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  mochi::Error appendError;
  series->Append({.frame = 2, .simulationTimeSec = 0.2}, transform, appendError);

  EXPECT_FALSE(appendError.IsOK());
}

TEST(SuperDexRecorderTest, WriteMeshRejectsOutOfRangeIndices) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  std::array<float, 9> const vertices{0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
  std::array<uint32_t, 3> const triangles{0, 1, 3};

  episode->WriteMesh("world/test", vertices, triangles, {255, 255, 255, 255}, error);

  EXPECT_FALSE(error.IsOK());
}

TEST(SuperDexRecorderTest, Points3DSeriesStoresOnlyChangedBatches) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series = episode->CreatePoints3DSeries("world/test", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  std::array<mochi::Real3, 2> const points{mochi::Real3{0_r, 0_r, 0_r}, {0.1_r, 0_r, 0_r}};
  auto const firstPoint = mochi::Span<mochi::Real3 const>{points}.subspan(0, 1);
  auto moved = points;
  moved[1][2] = 0.01_r;

  series->Append({.frame = 1, .simulationTimeSec = 0.1}, points, error);
  series->Append({.frame = 2, .simulationTimeSec = 0.2}, points, error);
  EXPECT_EQ(series->GetNumSamples(), 1U);
  // moved repeats the last batch but not the earlier batch of the same length.
  series->Append({.frame = 3, .simulationTimeSec = 0.3}, moved, error);
  series->Append({.frame = 4, .simulationTimeSec = 0.4}, moved, error);
  EXPECT_EQ(series->GetNumSamples(), 2U);
  series->Append({.frame = 5, .simulationTimeSec = 0.5}, firstPoint, error);
  EXPECT_EQ(series->GetNumSamples(), 3U);
  series->Append({.frame = 6, .simulationTimeSec = 0.6}, {}, error);
  series->Append({.frame = 7, .simulationTimeSec = 0.7}, {}, error);
  EXPECT_EQ(series->GetNumSamples(), 4U);
  series->Append({.frame = 8, .simulationTimeSec = 0.8}, points, error);
  EXPECT_EQ(series->GetNumSamples(), 5U);
  episode->Commit(error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
}

TEST(SuperDexRecorderTest, Points3DSeriesChecksTheTimeOfUnchangedBatches) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series = episode->CreatePoints3DSeries("world/test", error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  series->Append({.frame = 1, .simulationTimeSec = 0.1}, {}, error);
  series->Append({.frame = 2, .simulationTimeSec = 0.2}, {}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_EQ(series->GetNumSamples(), 1U);

  mochi::Error repeatedFrameError;
  series->Append({.frame = 2, .simulationTimeSec = 0.3}, {}, repeatedFrameError);
  mochi::Error earlierTimeError;
  series->Append({.frame = 3, .simulationTimeSec = 0.15}, {}, earlierTimeError);

  EXPECT_FALSE(repeatedFrameError.IsOK());
  EXPECT_FALSE(earlierTimeError.IsOK());
}

TEST(SuperDexRecorderTest, WriteKeypointSkeletonRejectsOutOfRangeConnectionsAndRadii) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  std::array<std::string, 2> const names{"root", "tip"};
  std::array<std::array<uint16_t, 2>, 1> const connection{{{0, 1}}};
  std::array<std::array<uint16_t, 2>, 1> const outOfRange{{{0, 2}}};
  std::array<uint8_t, 4> const color{255, 255, 255, 255};

  episode->WriteKeypointSkeleton("world/test", "test", names, connection, color, 0.01F, error);
  mochi::Error connectionError;
  episode->WriteKeypointSkeleton(
      "world/test", "test", names, outOfRange, color, 0.01F, connectionError);
  mochi::Error radiusError;
  episode->WriteKeypointSkeleton("world/test", "test", names, connection, color, 0.0F, radiusError);

  EXPECT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_FALSE(connectionError.IsOK());
  EXPECT_FALSE(radiusError.IsOK());
}

TEST(SuperDexRecorderTest, TextSeriesRecordsSeveralTextsPerFrame) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series =
      episode->CreateTextSeries("events/test", {"rerun.archetypes.TextLog", "TextLog:text"}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  std::array<std::string, 2> const texts{"trial_start=1", "grasp=left"};
  std::array<std::string, 1> const text{"trial_end=1"};

  series->Append({.frame = 1, .simulationTimeSec = 0.1}, texts, error);
  series->Append({.frame = 2, .simulationTimeSec = 0.2}, {}, error);
  series->Append({.frame = 3, .simulationTimeSec = 0.3}, text, error);
  EXPECT_EQ(series->GetNumSamples(), 3U);
  episode->Commit(error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_TRUE(std::filesystem::exists(episode->GetOutputPath()));
}

TEST(SuperDexRecorderTest, TextSeriesRejectsRepeatedFramesAndIncompleteFields) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  auto* const series =
      episode->CreateTextSeries("events/test", {"rerun.archetypes.TextLog", "TextLog:text"}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(series, nullptr);
  std::array<std::string, 1> const text{"event"};
  series->Append({.frame = 1, .simulationTimeSec = 0.1}, text, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  mochi::Error repeatedFrameError;
  series->Append({.frame = 1, .simulationTimeSec = 0.2}, text, repeatedFrameError);
  mochi::Error fieldError;
  auto* const incomplete =
      episode->CreateTextSeries("events/incomplete", {"rerun.archetypes.TextLog", ""}, fieldError);

  EXPECT_FALSE(repeatedFrameError.IsOK());
  EXPECT_EQ(incomplete, nullptr);
  EXPECT_FALSE(fieldError.IsOK());
}

TEST(SuperDexRecorderTest, SourceRecorderCreatesEachStreamOnFirstUse) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  SourceRecorder sources{*episode};
  sources.AddSkeleton(
      "hand/left", {.keypointNames = {"wrist", "tip"}, .parentIndices = {-1, 0}}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  std::array<mochi::Real3, 2> const keypoints{mochi::Real3{0_r, 0_r, 0_r}, {0.1_r, 0_r, 0_r}};
  std::array<std::string, 2> const events{"trial_start", "grasp"};

  for (uint64_t frame = 1; frame <= 2; ++frame) {
    SampleTime const time{.frame = frame, .simulationTimeSec = 0.1 * static_cast<double>(frame)};
    sources.RecordTransform(time, "head", mochi::TransformRT{}, error);
    sources.RecordSkeleton(time, "hand/left", keypoints, error);
    sources.RecordTexts(time, "study", events, error);
    sources.RecordScalar(time, "score", static_cast<mochi::real>(frame), error);
  }
  episode->Commit(error);

  EXPECT_TRUE(error.IsOK()) << error.ToString();
}

TEST(SuperDexRecorderTest, SourceRecorderRecordsScalarListsOnTheScalarSeries) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  SourceRecorder sources{*episode};
  std::array<mochi::real, 3> const forces{1_r, 2_r, 3_r};
  SampleTime const first{.frame = 1, .simulationTimeSec = 0.1};
  SampleTime const second{.frame = 2, .simulationTimeSec = 0.2};

  sources.RecordScalars(first, "forces", forces, error);
  sources.RecordScalar(second, "forces", 4_r, error);
  mochi::Error repeatedFrameError;
  sources.RecordScalars(second, "forces", forces, repeatedFrameError);
  episode->Commit(error);

  EXPECT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_FALSE(repeatedFrameError.IsOK());
}

TEST(SuperDexRecorderTest, SourceRecorderRejectsInvalidSkeletonsAndRepeatedFrames) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path()}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(episode, nullptr);
  SourceRecorder sources{*episode};
  SampleTime const time{.frame = 1, .simulationTimeSec = 0.1};
  std::array<mochi::Real3, 1> const keypoint{mochi::Real3{0_r, 0_r, 0_r}};
  sources.RecordScalar(time, "score", 1_r, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  mochi::Error parentError;
  sources.AddSkeleton("bad", {.keypointNames = {"a", "b"}, .parentIndices = {1, -1}}, parentError);
  mochi::Error undeclaredError;
  sources.RecordSkeleton(time, "missing", keypoint, undeclaredError);
  mochi::Error repeatedFrameError;
  sources.RecordScalar(time, "score", 2_r, repeatedFrameError);

  EXPECT_FALSE(parentError.IsOK());
  EXPECT_FALSE(undeclaredError.IsOK());
  EXPECT_FALSE(repeatedFrameError.IsOK());
}

} // namespace
} // namespace superdex::recorder
