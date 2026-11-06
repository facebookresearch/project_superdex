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
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

/* Unreal builds against this header and the recorders' headers without Rerun on its include path.
 * Only series.h may include Rerun. */
namespace rerun {
class RecordingStream;
} // namespace rerun

namespace superdex::recorder {

class Series;
class ScalarSeries;
class TransformSeries;
class Arrow3DSeries;
class Points3DSeries;
class DeformableMeshSeries;

inline constexpr std::string_view kSchemaName = "superdex.episode";
inline constexpr uint32_t kSchemaVersionMajor = 1;
inline constexpr uint32_t kSchemaVersionMinor = 0;
inline constexpr std::string_view kSchemaCoordinateSpace = "FLU";
inline constexpr double kSchemaUnitsPerMeter = 1.0;

struct SampleTime {
  uint64_t frame;
  double simulationTimeSec;
};

/* @brief Identifies one Rerun component column and its archetype. */
struct ComponentField {
  std::string archetype;
  std::string component;
};

struct RecorderConfig {
  std::filesystem::path outputDirectory;
};

struct EpisodeMetadata {
  std::map<std::string, std::string> properties;
};

/* @brief Owns one transactional RRD and the constants and sample series buffered until commit. */
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

  /* Every sample stays in memory until Commit, which blocks the calling thread while it writes the
   * episode; a two-arm teleoperation scene records about 1 MB/s, so keep episodes to minutes. */
  void Commit(mochi::Error& error);
  void Abort(std::string_view reason, mochi::Error& error);

  [[nodiscard]] std::string const& GetEpisodeId() const;
  [[nodiscard]] std::filesystem::path const& GetOutputPath() const;
  [[nodiscard]] bool IsActive() const;

 private:
  friend class Recorder;

  enum class State {
    Active,
    CommitFailed,
    Committed,
    Aborted,
  };

  struct StaticRows;

  Episode(
      std::string episodeId,
      std::filesystem::path temporaryPath,
      std::filesystem::path outputPath,
      std::unique_ptr<rerun::RecordingStream> stream);

  void ValidateWrite(mochi::Error& error) const;
  void ValidateSeriesPath(std::string_view entityPath, mochi::Error& error) const;
  void SealSeries();
  void DiscardTemporaryFile(mochi::Error* error);

  std::string _episodeId;
  std::filesystem::path _temporaryPath;
  std::filesystem::path _outputPath;
  std::unique_ptr<rerun::RecordingStream> _stream;
  std::unique_ptr<StaticRows> _staticRows;
  std::vector<std::unique_ptr<Series>> _series;
  State _state = State::Active;
};

/* @brief Starts transactional episode recordings in one output directory. */
class SUPERDEX_RECORDER_API Recorder final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(Recorder)

 public:
  explicit Recorder(RecorderConfig const& config);
  ~Recorder();

  std::unique_ptr<Episode> BeginEpisode(EpisodeMetadata const& metadata, mochi::Error& error) const;

 private:
  RecorderConfig _config;
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
