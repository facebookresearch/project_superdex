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
    std::function<void(Episode&, mochi::Error&)> const& record,
    size_t chunkMaxBytes = RecorderConfig{}.chunkMaxBytes,
    double chunkMaxDurationSec = RecorderConfig{}.chunkMaxDurationSec) {
  Recorder recorder{
      {.outputDirectory = directory.Path(),
       .chunkMaxBytes = chunkMaxBytes,
       .chunkMaxDurationSec = chunkMaxDurationSec}};
  mochi::Error error;
  auto episode = recorder.BeginEpisode({}, error);
  EXPECT_NE(episode, nullptr) << error.ToString();
  if (episode == nullptr) {
    return {};
  }
  record(*episode, error);
  episode->CommitSync(error);
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

TEST(EpisodeReaderTest, ReadsSeriesWrittenAsManyChunks) {
  // Transforms, points and meshes repeat at frames 1, 3 and 5, so only frames 0, 2 and 4 store a
  // row.
  auto const record = [](Episode& episode, mochi::Error& error) {
    auto* const scalars = episode.CreateScalarSeries(
        "data/test", {{"superdex.archetypes.Test", "superdex.test.values"}}, error);
    auto* const transforms = episode.CreateTransformSeries("world/box", error);
    auto* const points = episode.CreatePoints3DSeries("world/points", error);
    std::array<uint32_t, 3> const triangle{0, 1, 2};
    auto* const mesh =
        episode.CreateDeformableMeshSeries("world/mesh", 3, triangle, {255, 255, 255, 255}, error);
    auto* const plot = episode.CreatePlotSeries("data/plot", error);
    ASSERT_NE(scalars, nullptr);
    ASSERT_NE(transforms, nullptr);
    ASSERT_NE(points, nullptr);
    ASSERT_NE(mesh, nullptr);
    ASSERT_NE(plot, nullptr);
    for (uint64_t frame = 0; frame < 6; ++frame) {
      SampleTime const time{.frame = frame, .simulationTimeSec = 0.5 * frame};
      std::vector<mochi::real> const values(frame, static_cast<mochi::real>(frame));
      std::array<mochi::Span<mochi::real const>, 1> const fields{values};
      scalars->Append(time, fields, error);
      uint64_t const step = frame / 2;
      auto const position = static_cast<mochi::real>(step);
      transforms->Append(time, {position, 0_r, 0_r}, {0_r, 0_r, 0_r, 1_r}, error);
      std::array<mochi::Real3, 1> const point{mochi::Real3{0_r, position, 0_r}};
      points->Append(time, point, error);
      std::array<mochi::real, 9> const vertices{
          0_r, position, 0_r, 1_r, position, 0_r, 0_r, position, 1_r};
      std::array<mochi::real, 9> const normals{0_r, 1_r, 0_r, 0_r, 1_r, 0_r, 0_r, 1_r, 0_r};
      mesh->Append(time, vertices, normals, error);
      plot->Append(time, 0.1 * static_cast<double>(frame), error);
    }
  };
  TemporaryDirectory directory;
  // One chunk per series, then one chunk per row.
  auto const onePath = RecordEpisode(directory, record, 0, 0.0);
  auto const manyPath = RecordEpisode(directory, record, 1, 0.0);
  auto const one = OpenEpisode(onePath);
  auto const many = OpenEpisode(manyPath);
  ASSERT_NE(one, nullptr);
  ASSERT_NE(many, nullptr);
  mochi::Error error;

  for (auto const& [entityPath, component] :
       {std::pair{"data/test", "superdex.test.values"},
        std::pair{"world/box", "Transform3D:translation"},
        std::pair{"world/points", "Points3D:positions"},
        std::pair{"world/mesh", "Mesh3D:vertex_positions"},
        std::pair{"data/plot", "Scalars:scalars"}}) {
    ScalarRows const expected = one->ReadScalars(entityPath, component, error);
    ScalarRows const actual = many->ReadScalars(entityPath, component, error);

    ASSERT_TRUE(error.IsOK()) << error.ToString();
    EXPECT_EQ(actual.frames, expected.frames) << component;
    EXPECT_EQ(actual.simulationTimesNs, expected.simulationTimesNs) << component;
    EXPECT_EQ(actual.offsets, expected.offsets) << component;
    EXPECT_EQ(actual.values, expected.values) << component;
  }
  EXPECT_EQ(
      many->ReadScalars("world/box", "Transform3D:translation", error).frames,
      (std::vector<int64_t>{0, 2, 4}));
  EXPECT_EQ(
      many->ReadScalars("world/points", "Points3D:positions", error).frames,
      (std::vector<int64_t>{0, 2, 4}));
  EXPECT_EQ(
      many->ReadScalars("world/mesh", "Mesh3D:vertex_positions", error).frames,
      (std::vector<int64_t>{0, 2, 4}));
  EXPECT_EQ(
      many->ReadScalars("data/test", "superdex.test.values", error).values,
      (std::vector<double>{1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5}));
  EXPECT_TRUE(error.IsOK()) << error.ToString();
}

TEST(EpisodeReaderTest, LaterEpisodesOfARecorderReadBackOnlyTheirOwnSamples) {
  // Values carry their episode's offset, so a sample left in a reused buffer would show up.
  auto const record = [](Episode& episode, mochi::real offset, mochi::Error& error) {
    auto* const scalars = episode.CreateScalarSeries(
        "data/test", {{"superdex.archetypes.Test", "superdex.test.values"}}, error);
    auto* const transforms = episode.CreateTransformSeries("world/box", error);
    auto* const arrows = episode.CreateArrow3DSeries("world/arrows", error);
    auto* const points = episode.CreatePoints3DSeries("world/points", error);
    std::array<uint32_t, 3> const triangle{0, 1, 2};
    auto* const mesh =
        episode.CreateDeformableMeshSeries("world/mesh", 3, triangle, {255, 255, 255, 255}, error);
    auto* const plot = episode.CreatePlotSeries("data/plot", error);
    ASSERT_NE(scalars, nullptr);
    ASSERT_NE(transforms, nullptr);
    ASSERT_NE(arrows, nullptr);
    ASSERT_NE(points, nullptr);
    ASSERT_NE(mesh, nullptr);
    ASSERT_NE(plot, nullptr);
    for (uint64_t frame = 0; frame < 4; ++frame) {
      SampleTime const time{.frame = frame, .simulationTimeSec = 0.5 * frame};
      mochi::real const value = offset + static_cast<mochi::real>(frame);
      std::vector<mochi::real> const values(frame + 1, value);
      std::array<mochi::Span<mochi::real const>, 1> const fields{values};
      scalars->Append(time, fields, error);
      transforms->Append(time, {value, 0_r, 0_r}, {0_r, 0_r, value, 1_r}, error);
      std::array<mochi::Real3, 1> const origin{mochi::Real3{value, 0_r, 0_r}};
      std::array<mochi::Real3, 1> const vector{mochi::Real3{0_r, 0_r, value}};
      arrows->Append(time, origin, vector, error);
      std::array<mochi::Real3, 1> const point{mochi::Real3{0_r, value, 0_r}};
      points->Append(time, point, error);
      std::array<mochi::real, 9> const vertices{0_r, value, 0_r, 1_r, value, 0_r, 0_r, value, 1_r};
      std::array<mochi::real, 9> const normals{0_r, 1_r, 0_r, 0_r, 1_r, 0_r, 0_r, 1_r, 0_r};
      mesh->Append(time, vertices, normals, error);
      plot->Append(time, value, error);
    }
  };
  TemporaryDirectory directory;
  // Chunks of one to three rows, small enough to keep, so that later episodes reuse buffers.
  Recorder recorder{{.outputDirectory = directory.Path(), .chunkMaxBytes = 64}};
  // The first episode is discarded mid-chunk; the others continue in its buffers.
  std::vector<std::pair<mochi::real, std::filesystem::path>> committed;
  for (mochi::real const offset : {10_r, 20_r, 30_r}) {
    mochi::Error error;
    auto episode = recorder.BeginEpisode({}, error);
    ASSERT_NE(episode, nullptr) << error.ToString();
    record(*episode, offset, error);
    if (offset == 10_r) {
      episode->Abort("discarded", error);
    } else {
      episode->CommitSync(error);
      committed.emplace_back(offset, episode->GetOutputPath());
    }
    ASSERT_TRUE(error.IsOK()) << error.ToString();
  }

  for (auto const& [offset, path] : committed) {
    auto const expectedPath = RecordEpisode(
        directory, [&](Episode& episode, mochi::Error& error) { record(episode, offset, error); });
    auto const actual = OpenEpisode(path);
    auto const expected = OpenEpisode(expectedPath);
    ASSERT_NE(actual, nullptr);
    ASSERT_NE(expected, nullptr);
    mochi::Error error;
    for (auto const& [entityPath, component] :
         {std::pair{"data/test", "superdex.test.values"},
          std::pair{"world/box", "Transform3D:translation"},
          std::pair{"world/box", "Transform3D:quaternion"},
          std::pair{"world/arrows", "Arrows3D:origins"},
          std::pair{"world/arrows", "Arrows3D:vectors"},
          std::pair{"world/points", "Points3D:positions"},
          std::pair{"world/mesh", "Mesh3D:vertex_positions"},
          std::pair{"world/mesh", "Mesh3D:vertex_normals"},
          std::pair{"data/plot", "Scalars:scalars"}}) {
      ScalarRows const expectedRows = expected->ReadScalars(entityPath, component, error);
      ScalarRows const actualRows = actual->ReadScalars(entityPath, component, error);

      ASSERT_TRUE(error.IsOK()) << error.ToString();
      EXPECT_EQ(actualRows.frames, expectedRows.frames) << component;
      EXPECT_EQ(actualRows.offsets, expectedRows.offsets) << component;
      EXPECT_EQ(actualRows.values, expectedRows.values) << component;
    }
  }
}

TEST(EpisodeReaderTest, ReusedChunkBuffersTakeTheLayoutOfTheNewSeries) {
  TemporaryDirectory directory;
  Recorder recorder{{.outputDirectory = directory.Path(), .chunkMaxBytes = 64}};
  ComponentField const x{.archetype = "superdex.archetypes.Test", .component = "superdex.test.x"};
  ComponentField const y{.archetype = "superdex.archetypes.Test", .component = "superdex.test.y"};
  ComponentField const z{.archetype = "superdex.archetypes.Test", .component = "superdex.test.z"};
  mochi::Error error;
  // Leaves two-field scalar buffers for data/a and transform buffers for data/b.
  {
    auto episode = recorder.BeginEpisode({}, error);
    ASSERT_NE(episode, nullptr) << error.ToString();
    auto* const a = episode->CreateScalarSeries("data/a", {x, y}, error);
    auto* const b = episode->CreateTransformSeries("data/b", error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    std::array<mochi::real, 1> const one{1_r};
    std::array<mochi::Span<mochi::real const>, 2> const fields{one, one};
    for (uint64_t frame = 0; frame < 2; ++frame) {
      SampleTime const time{.frame = frame, .simulationTimeSec = 0.1 * frame};
      a->Append(time, fields, error);
      b->Append(time, {static_cast<mochi::real>(frame), 0_r, 0_r}, {0_r, 0_r, 0_r, 1_r}, error);
    }
    episode->CommitSync(error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
  }

  // Records data/a with one other field, and data/b as scalars.
  auto episode = recorder.BeginEpisode({}, error);
  ASSERT_NE(episode, nullptr) << error.ToString();
  auto* const a = episode->CreateScalarSeries("data/a", {z}, error);
  auto* const b = episode->CreateScalarSeries("data/b", {x}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  std::array<mochi::real, 1> const two{2_r};
  std::array<mochi::Span<mochi::real const>, 1> const fields{two};
  for (uint64_t frame = 0; frame < 2; ++frame) {
    SampleTime const time{.frame = frame, .simulationTimeSec = 0.1 * frame};
    a->Append(time, fields, error);
    b->Append(time, fields, error);
  }
  episode->CommitSync(error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  auto const reader = OpenEpisode(episode->GetOutputPath());
  ASSERT_NE(reader, nullptr);

  std::vector<double> const expected{2, 2};
  EXPECT_EQ(reader->ReadScalars("data/a", "superdex.test.z", error).values, expected);
  EXPECT_EQ(reader->ReadScalars("data/b", "superdex.test.x", error).values, expected);
  EXPECT_TRUE(error.IsOK()) << error.ToString();
  EXPECT_FALSE(reader->HasComponent("data/a", "superdex.test.x", /*isStatic=*/false));
  EXPECT_FALSE(reader->HasComponent("data/b", "Transform3D:translation", /*isStatic=*/false));
}

TEST(EpisodeReaderTest, OverlappingEpisodesOfARecorderRecordTheirOwnSamples) {
  TemporaryDirectory directory;
  // Chunks of one to three rows, small enough to keep, so that later episodes reuse buffers.
  Recorder recorder{{.outputDirectory = directory.Path(), .chunkMaxBytes = 64}};
  ComponentField const field{
      .archetype = "superdex.archetypes.Test", .component = "superdex.test.values"};
  ComponentField const text{.archetype = "rerun.archetypes.TextLog", .component = "TextLog:text"};
  constexpr uint64_t kNumFrames = 10;
  std::vector<std::filesystem::path> paths;
  // Each episode commits in the background and lives until the next one commits, as in Unreal.
  std::unique_ptr<Episode> committing;
  for (int index = 0; index < 3; ++index) {
    mochi::Error error;
    auto episode = recorder.BeginEpisode({}, error);
    ASSERT_NE(episode, nullptr) << error.ToString();
    auto* const values = episode->CreateScalarSeries("data/test", {field}, error);
    // One path alternates between text and scalars.
    TextSeries* const texts =
        index % 2 == 0 ? episode->CreateTextSeries("data/mixed", text, error) : nullptr;
    ScalarSeries* const scalars =
        index % 2 == 1 ? episode->CreateScalarSeries("data/mixed", {field}, error) : nullptr;
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    for (uint64_t frame = 0; frame < kNumFrames; ++frame) {
      SampleTime const time{
          .frame = frame,
          .simulationTimeSec = 0.1 * frame,
          .wallTimeSec = episode->GetWallTimeSec()};
      std::array<mochi::real, 1> const value{static_cast<mochi::real>(100 * index + frame)};
      std::array<mochi::Span<mochi::real const>, 1> const fields{value};
      values->Append(time, fields, error);
      if (texts != nullptr) {
        std::array<std::string, 1> const message{std::to_string(frame)};
        texts->Append(time, message, error);
      }
      if (scalars != nullptr) {
        scalars->Append(time, fields, error);
      }
    }
    episode->CommitAsync(error);
    ASSERT_TRUE(error.IsOK()) << error.ToString();
    paths.push_back(episode->GetOutputPath());
    committing = std::move(episode);
  }
  committing.reset();

  for (int index = 0; index < 3; ++index) {
    auto const reader = OpenEpisode(paths[index]);
    ASSERT_NE(reader, nullptr);
    mochi::Error error;
    std::vector<double> expected;
    expected.reserve(kNumFrames);
    for (uint64_t frame = 0; frame < kNumFrames; ++frame) {
      expected.push_back(static_cast<double>(100 * index + frame));
    }
    EXPECT_EQ(reader->ReadScalars("data/test", "superdex.test.values", error).values, expected);
    if (index % 2 == 1) {
      EXPECT_EQ(reader->ReadScalars("data/mixed", "superdex.test.values", error).values, expected);
    } else {
      EXPECT_TRUE(reader->HasComponent("data/mixed", "TextLog:text", /*isStatic=*/false));
    }
    EXPECT_TRUE(error.IsOK()) << error.ToString();
  }
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
