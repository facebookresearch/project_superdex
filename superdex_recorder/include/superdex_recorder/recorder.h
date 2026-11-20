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

#pragma once

#include <superdex_recorder/export.h>

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/* Unreal builds against this header and the recorders' headers without Rerun on its include path.
 * Only series.h may include Rerun. */
namespace rerun {
class RecordingStream;
} // namespace rerun

namespace superdex::recorder {

namespace detail {
class ChunkPool;
class ChunkWriter;
struct SeriesChunk;
} // namespace detail

class Series;
class ScalarSeries;
class TransformSeries;
class Arrow3DSeries;
class Points3DSeries;
class DeformableMeshSeries;
class TextSeries;
class PlotSeries;

inline constexpr std::string_view kSchemaName = "superdex.episode";
inline constexpr uint32_t kSchemaVersionMajor = 1;
inline constexpr uint32_t kSchemaVersionMinor = 1;
inline constexpr std::string_view kSchemaCoordinateSpace = "FLU";
inline constexpr double kSchemaUnitsPerMeter = 1.0;

struct SampleTime {
  uint64_t frame;
  double simulationTimeSec;
  /* @brief Wall-clock time of the sample since its episode began [s], from Episode::GetWallTimeSec,
   * recorded on the wall_time timeline. Unset records no wall time. Compared with the simulation
   * time, it shows when samples were late or missing. Every sample of a series has one or none
   * does. */
  std::optional<double> wallTimeSec;
};

/* @brief Identifies one Rerun component column and its archetype. */
struct ComponentField {
  std::string archetype;
  std::string component;
};

struct RecorderConfig {
  std::filesystem::path outputDirectory;
  /* @brief A series writes its buffered rows as one RRD chunk once they reach either limit. Zero
   * disables a limit; with both disabled, each series is one chunk written at commit. The byte
   * limit is at most 1 GB, since the native reader rejects chunks over 2 GiB. A zero byte limit
   * also disables reusing chunk buffers (see Recorder). */
  size_t chunkMaxBytes = 8'000'000; // 8 MB
  double chunkMaxDurationSec = 10.0;
};

struct EpisodeMetadata {
  std::map<std::string, std::string> properties;
};

/* @brief Owns one transactional RRD and its sample series. A writer thread writes the series in
 * chunks; constants are buffered until commit. */
class SUPERDEX_RECORDER_API Episode final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(Episode)

 public:
  ~Episode();

  /* Keep allocation and deletion in the recorder library's heap, even when a module with its own
   * allocator deletes the episode that BeginEpisode returned. */
  static void* operator new(size_t size);
  static void operator delete(void* episode);

  ScalarSeries* CreateScalarSeries(
      std::string_view entityPath,
      std::vector<ComponentField> const& fields,
      mochi::Error& error);
  TransformSeries* CreateTransformSeries(std::string_view entityPath, mochi::Error& error);
  Arrow3DSeries* CreateArrow3DSeries(std::string_view entityPath, mochi::Error& error);
  Points3DSeries* CreatePoints3DSeries(std::string_view entityPath, mochi::Error& error);
  /* Writes the mesh's triangles, which index its numVertices vertices, and its color once; the
   * series stores the vertices at each sample. */
  DeformableMeshSeries* CreateDeformableMeshSeries(
      std::string_view entityPath,
      size_t numVertices,
      mochi::Span<uint32_t const> triangleIndices,
      std::array<uint8_t, 4> const& color,
      mochi::Error& error);
  /* Records text over time, such as the events an application raises. Each sample holds every
   * text of one frame, so several events in the same step share a sample. */
  TextSeries*
  CreateTextSeries(std::string_view entityPath, ComponentField const& field, mochi::Error& error);
  /* Records one value per sample that the Rerun viewer plots; its component is "Scalars:scalars".
   */
  PlotSeries* CreatePlotSeries(std::string_view entityPath, mochi::Error& error);
  void WriteConstantText(
      std::string_view entityPath,
      ComponentField const& field,
      std::string_view value,
      mochi::Error& error);
  void WriteConstantTexts(
      std::string_view entityPath,
      ComponentField const& field,
      mochi::Span<std::string const> values,
      mochi::Error& error);
  void WriteConstantScalars(
      std::string_view entityPath,
      ComponentField const& field,
      mochi::Span<double const> values,
      mochi::Error& error);
  void WriteConstantPose(
      std::string_view entityPath,
      ComponentField const& translationField,
      ComponentField const& quaternionField,
      std::array<double, 3> const& translation,
      std::array<double, 4> const& quaternionXyzw,
      mochi::Error& error);
  void WriteConstantPose(
      std::string_view entityPath,
      ComponentField const& translationField,
      ComponentField const& quaternionField,
      mochi::TransformRT const& transform,
      mochi::Error& error);
  void WriteMesh(
      std::string_view entityPath,
      mochi::Span<float const> vertexPositions,
      mochi::Span<uint32_t const> triangleIndices,
      std::array<uint8_t, 4> const& color,
      mochi::Error& error);
  /* Labels the points of a Points3DSeries at entityPath as one keypoint skeleton: point i of every
   * sample is keypoint i, named keypointNames[i], and the viewer draws a line for each connection
   * between two keypoint indices. */
  void WriteKeypointSkeleton(
      std::string_view entityPath,
      std::string_view label,
      mochi::Span<std::string const> keypointNames,
      mochi::Span<std::array<uint16_t, 2> const> connections,
      std::array<uint8_t, 4> const& color,
      float keypointRadiusMeters,
      mochi::Error& error);

  /* Writes the remaining rows and the constants, then publishes the file. CommitSync, CommitAsync
   * and Abort may run on any thread once appending has stopped. */
  void CommitSync(mochi::Error& error);
  /* Like CommitSync, but returns without waiting for the file: error reports only the failures
   * found before it returns. The episode's writer thread then publishes the file, or logs the
   * failure and deletes it. IsCommitFinished and WaitForCommit tell when that is done; destroying
   * the episode waits for it. */
  void CommitAsync(mochi::Error& error);
  /* True once the commit is done, published or failed: right after CommitSync, and once the writer
   * thread has finished after CommitAsync. Destroying the episode then does not wait. */
  [[nodiscard]] bool IsCommitFinished() const;
  /* Waits until the commit is done, then sets its failure, if any. Returns at once after
   * CommitSync, which reported its own failures. Call it on the thread that committed. */
  void WaitForCommit(mochi::Error& error);
  void Abort(std::string_view reason, mochi::Error& error);

  [[nodiscard]] std::string const& GetEpisodeId() const;
  [[nodiscard]] std::filesystem::path const& GetOutputPath() const;
  [[nodiscard]] bool IsActive() const;
  /* @brief Returns the time since the episode began [s], on a monotonic clock: a sample's
   * SampleTime::wallTimeSec. */
  [[nodiscard]] double GetWallTimeSec() const;

 private:
  friend class Recorder;

  enum class State {
    Active,
    CommitFailed,
    Committed,
    Aborted,
  };

  struct StaticRows;
  struct SeriesPaths;

  Episode(
      std::string episodeId,
      std::filesystem::path temporaryPath,
      std::filesystem::path outputPath,
      std::unique_ptr<rerun::RecordingStream> stream,
      RecorderConfig const& config,
      std::shared_ptr<detail::ChunkPool> pool);

  void ValidateWrite(mochi::Error& error) const;
  void ValidateSeriesPath(std::string_view entityPath, mochi::Error& error) const;
  void AddSeries(std::unique_ptr<Series> series);
  void SealSeries();
  /* Seals the series and submits their remaining rows to the writer. */
  void BeginCommit(mochi::Error& error);
  /* Writes the constants and renames the file into place once every chunk is written. */
  void PublishFile(mochi::Error& error);
  void DiscardTemporaryFile(mochi::Error* error);
  [[nodiscard]] bool RemoveTemporaryFile(mochi::Error* error);

  std::string _episodeId;
  std::filesystem::path _temporaryPath;
  std::filesystem::path _outputPath;
  std::unique_ptr<rerun::RecordingStream> _stream;
  std::unique_ptr<StaticRows> _staticRows;
  std::vector<std::unique_ptr<Series>> _series;
  // The series' entity paths, so that creating a series checks for a duplicate in constant time.
  std::unique_ptr<SeriesPaths> _seriesPaths;
  // Declared after _stream so that it stops before the stream is destroyed.
  std::unique_ptr<detail::ChunkWriter> _writer;
  State _state = State::Active;
  // The failure of a CommitAsync, set on the writer thread and read once it has stopped.
  mochi::Error _commitError;
  std::chrono::steady_clock::time_point _startTime = std::chrono::steady_clock::now();
};

/* @brief Starts transactional episode recordings in one output directory.
 * Keep one recorder for consecutive episodes: their series continue in the buffers of earlier
 * episodes' chunks rather than growing new ones while recording. The recorder keeps the buffers of
 * up to two chunks per entity path, each of at most about four times chunkMaxBytes, while it or
 * any of its episodes exists; it frees those of paths that its episodes stop recording. A zero
 * chunkMaxBytes disables this reuse. */
class SUPERDEX_RECORDER_API Recorder final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(Recorder)

 public:
  explicit Recorder(RecorderConfig const& config);
  ~Recorder();

  std::unique_ptr<Episode> BeginEpisode(EpisodeMetadata const& metadata, mochi::Error& error) const;

 private:
  RecorderConfig _config;
  // Written chunks' buffers, which the series of its episodes continue in.
  std::shared_ptr<detail::ChunkPool> _pool;
};

/* @brief Sets Mochi's log callback inside this library.
 * A shared library can hold its own copy of the callback, and on Windows always does, so a host
 * that loads this library passes its callback here. */
SUPERDEX_RECORDER_API void SetLogCallback(mochi::LogFn const& callback);

namespace detail {
// Not exported: they return std::string, which must not cross the DLL boundary.
std::string EncodeEntityPathSegment(std::string_view segment);
/* Names a robot's entities after its articulated actor's name, or "robot" if that is null or
 * empty. */
std::string RobotEntityId(char const* actorName);
/* Returns world/shadows/<robot>/<id>. Every kind of shadow builds its path here, so the episode
 * rejects two shadows with one ID on one robot. */
std::string ShadowEntityPath(char const* actorName, std::string_view id, mochi::Error& error);
} // namespace detail

} // namespace superdex::recorder
