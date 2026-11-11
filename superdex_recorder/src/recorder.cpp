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

#include <mochi_core/utils/log.h>

#include <rerun.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <random>
#include <system_error>
#include <utility>

namespace superdex::recorder {
namespace {

constexpr std::string_view kApplicationId = "superdex";
constexpr std::string_view kEpisodeArchetype = "superdex.archetypes.Episode";
constexpr std::string_view kShadowsPath = "world/shadows";

bool HasValidPropertyKeys(std::map<std::string, std::string> const& properties) {
  return std::ranges::none_of(
      properties, [](auto const& property) { return property.first.empty(); });
}

bool HasValidField(ComponentField const& field) {
  return !field.archetype.empty() && !field.component.empty();
}

// Readers look components up by name alone, so a name must be unique across archetypes.
bool HasDuplicateComponents(mochi::Span<ComponentField const> fields) {
  std::vector<std::string_view> componentNames;
  componentNames.reserve(fields.size());
  for (auto const& field : fields) {
    componentNames.push_back(field.component);
  }
  std::ranges::sort(componentNames);
  return std::ranges::adjacent_find(componentNames) != componentNames.end();
}

void SetRerunError(rerun::Error const& rerunError, mochi::Error& error, char const* description) {
  if (rerunError.is_ok()) {
    return;
  }
  MOCHI_LOG_ERROR("%s: %s", description, rerunError.description.c_str());
  error.SetFirstError(description, __FILE__, __LINE__);
}

// RFC 9562 UUIDv7: the leading Unix-millisecond timestamp makes episode IDs, and the file names
// built from them, sort by start time.
std::string GenerateUuidV7(mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::array<uint8_t, 16> bytes{};
  try {
    std::random_device device;
    for (size_t i = 6; i < bytes.size(); ++i) {
      bytes[i] = static_cast<uint8_t>(device());
    }
  } catch (std::exception const&) {
    MOCHI_ERROR_SET(error, "Failed to generate an episode UUID.");
    return {};
  }
  auto const milliseconds =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count());
  for (size_t i = 0; i < 6; ++i) {
    bytes[i] = static_cast<uint8_t>(milliseconds >> (40 - 8 * i));
  }
  bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0F) | 0x70); // Version 7.
  bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3F) | 0x80); // RFC 9562 variant.
  constexpr std::string_view kHexDigits = "0123456789abcdef";
  std::string uuid;
  uuid.reserve(36);
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      uuid.push_back('-');
    }
    uuid.push_back(kHexDigits[bytes[i] >> 4]);
    uuid.push_back(kHexDigits[bytes[i] & 0x0F]);
  }
  return uuid;
}

std::string UtcNowNanoseconds() {
  return std::to_string(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

} // namespace

// Rerun splits an entity's separately logged static rows into chunks by datatype and by flush
// timing, so each entity's constants are logged as one row at commit. A component written twice on
// one entity keeps its last batch.
struct Episode::StaticRows {
  std::map<std::string, std::vector<rerun::ComponentBatch>> batchesByEntity;
};

Episode::Episode(
    std::string episodeId,
    std::filesystem::path temporaryPath,
    std::filesystem::path outputPath,
    std::unique_ptr<rerun::RecordingStream> stream)
    : _episodeId{std::move(episodeId)},
      _temporaryPath{std::move(temporaryPath)},
      _outputPath{std::move(outputPath)},
      _stream{std::move(stream)},
      _staticRows{std::make_unique<StaticRows>()} {}

Episode::~Episode() {
  if (_state == State::Active || _state == State::CommitFailed) {
    DiscardTemporaryFile(nullptr);
  }
}

void* Episode::operator new(size_t size) {
  return ::operator new(size);
}

void Episode::operator delete(void* episode) {
  ::operator delete(episode);
}

ScalarSeries* Episode::CreateScalarSeries(
    std::string_view entityPath,
    std::vector<ComponentField> const& fields,
    mochi::Error& error) {
  ValidateWrite(error);
  ValidateSeriesPath(entityPath, error);
  MOCHI_ERROR_IF(fields.empty(), error, "A scalar series must have at least one field.");
  MOCHI_ERROR_IF_NOT(
      std::ranges::all_of(fields, HasValidField),
      error,
      "Scalar series fields must have archetype and component names.");
  MOCHI_ERROR_IF(
      HasDuplicateComponents(fields), error, "Scalar series component names must be unique.");
  MOCHI_ERROR_RETURN(error, nullptr);

  auto series = std::make_unique<ScalarSeries>(entityPath, fields);
  auto* const result = series.get();
  _series.push_back(std::move(series));
  return result;
}

TransformSeries* Episode::CreateTransformSeries(std::string_view entityPath, mochi::Error& error) {
  ValidateWrite(error);
  ValidateSeriesPath(entityPath, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  auto series = std::make_unique<TransformSeries>(entityPath);
  auto* const result = series.get();
  _series.push_back(std::move(series));
  return result;
}

Arrow3DSeries* Episode::CreateArrow3DSeries(std::string_view entityPath, mochi::Error& error) {
  ValidateWrite(error);
  ValidateSeriesPath(entityPath, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  auto series = std::make_unique<Arrow3DSeries>(entityPath);
  auto* const result = series.get();
  _series.push_back(std::move(series));
  return result;
}

Points3DSeries* Episode::CreatePoints3DSeries(std::string_view entityPath, mochi::Error& error) {
  ValidateWrite(error);
  ValidateSeriesPath(entityPath, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  auto series = std::make_unique<Points3DSeries>(entityPath);
  auto* const result = series.get();
  _series.push_back(std::move(series));
  return result;
}

DeformableMeshSeries* Episode::CreateDeformableMeshSeries(
    std::string_view entityPath,
    size_t numVertices,
    mochi::Span<uint32_t const> triangleIndices,
    std::array<uint8_t, 4> const& color,
    mochi::Error& error) {
  ValidateWrite(error);
  ValidateSeriesPath(entityPath, error);
  MOCHI_ERROR_IF(
      numVertices > std::numeric_limits<uint32_t>::max(),
      error,
      "Mesh series has too many vertices to record.");
  MOCHI_ERROR_IF(
      triangleIndices.size() % 3 != 0,
      error,
      "Mesh triangle indices must contain complete triples.");
  MOCHI_ERROR_IF_NOT(
      std::ranges::all_of(
          triangleIndices, [numVertices](uint32_t index) { return index < numVertices; }),
      error,
      "Mesh triangle index exceeds the vertex count.");
  MOCHI_ERROR_RETURN(error, nullptr);

  std::vector<rerun::components::TriangleIndices> triangles;
  triangles.reserve(triangleIndices.size() / 3);
  for (size_t i = 0; i < triangleIndices.size(); i += 3) {
    triangles.emplace_back(triangleIndices[i], triangleIndices[i + 1], triangleIndices[i + 2]);
  }
  auto batches = rerun::AsComponents<rerun::archetypes::Mesh3D>::as_batches(
      rerun::archetypes::Mesh3D::update_fields()
          .with_triangle_indices(std::move(triangles))
          .with_albedo_factor(
              rerun::components::AlbedoFactor{
                  rerun::datatypes::Rgba32{color[0], color[1], color[2], color[3]}}));
  if (batches.is_err()) {
    SetRerunError(batches.error, error, "Failed to encode constant mesh triangles.");
    return nullptr;
  }
  auto& row = _staticRows->batchesByEntity[std::string{entityPath}];
  row.insert(row.end(), batches.value.begin(), batches.value.end());

  auto series = std::make_unique<DeformableMeshSeries>(entityPath, numVertices);
  auto* const result = series.get();
  _series.push_back(std::move(series));
  return result;
}

TextSeries* Episode::CreateTextSeries(
    std::string_view entityPath,
    ComponentField const& field,
    mochi::Error& error) {
  ValidateWrite(error);
  ValidateSeriesPath(entityPath, error);
  MOCHI_ERROR_IF_NOT(
      HasValidField(field), error, "A text series field must have archetype and component names.");
  MOCHI_ERROR_RETURN(error, nullptr);

  auto series = std::make_unique<TextSeries>(entityPath, field);
  auto* const result = series.get();
  _series.push_back(std::move(series));
  return result;
}

void Episode::WriteConstantText(
    std::string_view entityPath,
    ComponentField const& field,
    std::string_view value,
    mochi::Error& error) {
  std::array<std::string, 1> const values{std::string{value}};
  WriteConstantTexts(entityPath, field, values, error);
}

void Episode::WriteConstantTexts(
    std::string_view entityPath,
    ComponentField const& field,
    mochi::Span<std::string const> values,
    mochi::Error& error) {
  ValidateWrite(error);
  MOCHI_ERROR_IF(entityPath.empty(), error, "Constant entity path must not be empty.");
  MOCHI_ERROR_IF_NOT(HasValidField(field), error, "Constant field is incomplete.");
  MOCHI_ERROR_RETURN(error);

  std::vector<rerun::components::Text> texts;
  texts.reserve(values.size());
  for (auto const& value : values) {
    texts.emplace_back(value);
  }
  auto batch = rerun::ComponentBatch::from_loggable(
      rerun::Collection<rerun::components::Text>{std::move(texts)},
      rerun::ComponentDescriptor{
          field.archetype,
          field.component,
          rerun::Loggable<rerun::components::Text>::ComponentType});
  if (batch.is_err()) {
    SetRerunError(batch.error, error, "Failed to encode constant text data.");
    return;
  }
  _staticRows->batchesByEntity[std::string{entityPath}].push_back(std::move(batch.value));
}

void Episode::WriteConstantScalars(
    std::string_view entityPath,
    ComponentField const& field,
    mochi::Span<double const> values,
    mochi::Error& error) {
  ValidateWrite(error);
  MOCHI_ERROR_IF(entityPath.empty(), error, "Constant entity path must not be empty.");
  MOCHI_ERROR_IF_NOT(HasValidField(field), error, "Constant field is incomplete.");
  MOCHI_ERROR_RETURN(error);

  std::vector<rerun::components::Scalar> scalars;
  scalars.reserve(values.size());
  for (double const value : values) {
    scalars.emplace_back(value);
  }
  auto batch = rerun::ComponentBatch::from_loggable(
      rerun::Collection<rerun::components::Scalar>{std::move(scalars)},
      rerun::ComponentDescriptor{
          field.archetype,
          field.component,
          rerun::Loggable<rerun::components::Scalar>::ComponentType});
  if (batch.is_err()) {
    SetRerunError(batch.error, error, "Failed to encode constant scalar data.");
    return;
  }
  _staticRows->batchesByEntity[std::string{entityPath}].push_back(std::move(batch.value));
}

void Episode::WriteConstantPose(
    std::string_view entityPath,
    ComponentField const& translationField,
    ComponentField const& quaternionField,
    std::array<double, 3> const& translation,
    std::array<double, 4> const& quaternionXyzw,
    mochi::Error& error) {
  WriteConstantScalars(entityPath, translationField, translation, error);
  WriteConstantScalars(entityPath, quaternionField, quaternionXyzw, error);
}

void Episode::WriteConstantPose(
    std::string_view entityPath,
    ComponentField const& translationField,
    ComponentField const& quaternionField,
    mochi::TransformRT const& transform,
    mochi::Error& error) {
  auto const translation = transform.GetTranslation();
  auto const& rotation = transform.GetRotation();
  std::array<double, 3> const translationArray{
      static_cast<double>(translation[0]),
      static_cast<double>(translation[1]),
      static_cast<double>(translation[2])};
  std::array<double, 4> const quaternionArray{
      static_cast<double>(rotation.data[0]),
      static_cast<double>(rotation.data[1]),
      static_cast<double>(rotation.data[2]),
      static_cast<double>(rotation.data[3])};
  WriteConstantPose(
      entityPath, translationField, quaternionField, translationArray, quaternionArray, error);
}

void Episode::WriteMesh(
    std::string_view entityPath,
    mochi::Span<float const> vertexPositions,
    mochi::Span<uint32_t const> triangleIndices,
    std::array<uint8_t, 4> const& color,
    mochi::Error& error) {
  ValidateWrite(error);
  MOCHI_ERROR_IF(entityPath.empty(), error, "Mesh entity path must not be empty.");
  MOCHI_ERROR_IF(
      vertexPositions.size() % 3 != 0,
      error,
      "Mesh vertex positions must contain complete XYZ triples.");
  MOCHI_ERROR_IF(
      triangleIndices.size() % 3 != 0,
      error,
      "Mesh triangle indices must contain complete triples.");
  size_t const vertexCount = vertexPositions.size() / 3;
  MOCHI_ERROR_IF_NOT(
      std::ranges::all_of(
          triangleIndices, [vertexCount](uint32_t index) { return index < vertexCount; }),
      error,
      "Mesh triangle index exceeds the vertex count.");
  MOCHI_ERROR_RETURN(error);

  // Rerun shades with per-vertex normals, so flat shading needs each triangle's own vertices.
  // Zero-area triangles cover no pixels and have no normal.
  std::vector<rerun::Position3D> vertices;
  std::vector<rerun::components::Vector3D> normals;
  vertices.reserve(triangleIndices.size());
  normals.reserve(triangleIndices.size());
  for (size_t i = 0; i < triangleIndices.size(); i += 3) {
    std::array<float const*, 3> const corners{
        &vertexPositions[size_t{triangleIndices[i]} * 3],
        &vertexPositions[size_t{triangleIndices[i + 1]} * 3],
        &vertexPositions[size_t{triangleIndices[i + 2]} * 3]};
    std::array<float, 3> const ab{
        corners[1][0] - corners[0][0],
        corners[1][1] - corners[0][1],
        corners[1][2] - corners[0][2]};
    std::array<float, 3> const ac{
        corners[2][0] - corners[0][0],
        corners[2][1] - corners[0][1],
        corners[2][2] - corners[0][2]};
    std::array<float, 3> normal{
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0]};
    float const length =
        std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (length == 0.0f) {
      continue;
    }
    for (float& component : normal) {
      component /= length;
    }
    for (float const* corner : corners) {
      vertices.emplace_back(corner[0], corner[1], corner[2]);
      normals.emplace_back(normal[0], normal[1], normal[2]);
    }
  }

  auto mesh = rerun::archetypes::Mesh3D{std::move(vertices)}
                  .with_vertex_normals(std::move(normals))
                  .with_albedo_factor(
                      rerun::components::AlbedoFactor{
                          rerun::datatypes::Rgba32{color[0], color[1], color[2], color[3]}});
  auto batches = rerun::AsComponents<rerun::archetypes::Mesh3D>::as_batches(mesh);
  if (batches.is_err()) {
    SetRerunError(batches.error, error, "Failed to encode constant mesh data.");
    return;
  }
  auto& row = _staticRows->batchesByEntity[std::string{entityPath}];
  row.insert(row.end(), batches.value.begin(), batches.value.end());
}

void Episode::WriteKeypointSkeleton(
    std::string_view entityPath,
    std::string_view label,
    mochi::Span<std::string const> keypointNames,
    mochi::Span<std::array<uint16_t, 2> const> connections,
    std::array<uint8_t, 4> const& color,
    float keypointRadiusMeters,
    mochi::Error& error) {
  ValidateWrite(error);
  MOCHI_ERROR_IF(entityPath.empty(), error, "Keypoint skeleton entity path must not be empty.");
  MOCHI_ERROR_IF(
      keypointNames.size() > size_t{std::numeric_limits<uint16_t>::max()} + 1,
      error,
      "Keypoint skeleton has too many keypoints.");
  MOCHI_ERROR_IF_NOT(
      std::ranges::all_of(
          connections,
          [&](auto const& connection) {
            return connection[0] < keypointNames.size() && connection[1] < keypointNames.size();
          }),
      error,
      "Keypoint skeleton connection index exceeds the keypoint count.");
  MOCHI_ERROR_IF_NOT(
      std::isfinite(keypointRadiusMeters) && keypointRadiusMeters > 0.0f,
      error,
      "Keypoint radius must be finite and positive.");
  MOCHI_ERROR_RETURN(error);

  // The viewer joins points by their class and keypoint IDs, and draws the connections in the
  // class color rather than the point color.
  constexpr uint16_t kClassId = 0;
  rerun::datatypes::Rgba32 const rgba{color[0], color[1], color[2], color[3]};
  std::vector<rerun::datatypes::AnnotationInfo> keypoints;
  std::vector<rerun::components::KeypointId> keypointIds;
  keypoints.reserve(keypointNames.size());
  keypointIds.reserve(keypointNames.size());
  for (size_t i = 0; i < keypointNames.size(); ++i) {
    keypoints.emplace_back(static_cast<uint16_t>(i), keypointNames[i]);
    keypointIds.emplace_back(static_cast<uint16_t>(i));
  }
  std::vector<rerun::datatypes::KeypointPair> pairs;
  pairs.reserve(connections.size());
  for (auto const& connection : connections) {
    pairs.emplace_back(connection[0], connection[1]);
  }
  std::vector<rerun::datatypes::ClassDescriptionMapElem> classes;
  classes.emplace_back(
      rerun::datatypes::ClassDescription{
          rerun::datatypes::AnnotationInfo{kClassId, std::string{label}, rgba},
          std::move(keypoints),
          std::move(pairs)});
  auto contextBatches = rerun::AsComponents<rerun::archetypes::AnnotationContext>::as_batches(
      rerun::archetypes::AnnotationContext{
          rerun::components::AnnotationContext{std::move(classes)}});
  auto pointBatches = rerun::AsComponents<rerun::archetypes::Points3D>::as_batches(
      rerun::archetypes::Points3D::update_fields()
          .with_class_ids(std::vector<rerun::components::ClassId>(keypointNames.size(), kClassId))
          .with_keypoint_ids(std::move(keypointIds))
          .with_colors(rerun::components::Color{rgba})
          .with_radii(rerun::components::Radius::scene_units(keypointRadiusMeters))
          .with_show_labels(rerun::components::ShowLabels{false}));
  if (contextBatches.is_err() || pointBatches.is_err()) {
    SetRerunError(
        contextBatches.is_err() ? contextBatches.error : pointBatches.error,
        error,
        "Failed to encode constant keypoint skeleton data.");
    return;
  }
  auto& row = _staticRows->batchesByEntity[std::string{entityPath}];
  row.insert(row.end(), contextBatches.value.begin(), contextBatches.value.end());
  row.insert(row.end(), pointBatches.value.begin(), pointBatches.value.end());
}

void Episode::Commit(mochi::Error& error) {
  ValidateWrite(error);
  MOCHI_ERROR_RETURN(error);
  SealSeries();

  WriteConstantText(
      "episode",
      {std::string{kEpisodeArchetype}, "superdex.episode.committed_utc_ns"},
      UtcNowNanoseconds(),
      error);
  _state = State::CommitFailed;
  MOCHI_ERROR_RETURN(error);

  for (auto& [entityPath, batches] : _staticRows->batchesByEntity) {
    SetRerunError(
        _stream->try_log_serialized_batches(entityPath, /*static_=*/true, std::move(batches)),
        error,
        "Failed to write constant data.");
  }
  for (auto& ownedSeries : _series) {
    Series* const series = ownedSeries.get();
    if (series == nullptr) {
      MOCHI_ERROR_SET(error, "Episode contains an invalid series.");
      break;
    }
    series->Flush(*_stream, error);
  }
  MOCHI_ERROR_RETURN(error);
  SetRerunError(_stream->flush_blocking(), error, "Failed to flush the episode RRD.");
  MOCHI_ERROR_RETURN(error);
  _stream.reset();

  std::error_code filesystemError;
  std::filesystem::rename(_temporaryPath, _outputPath, filesystemError);
  if (filesystemError) {
    MOCHI_LOG_ERROR(
        "Failed to publish episode '%s': %s",
        _episodeId.c_str(),
        filesystemError.message().c_str());
    MOCHI_ERROR_SET(error, "Failed to atomically publish the episode RRD.");
    return;
  }
  _state = State::Committed;
}

void Episode::Abort(std::string_view reason, mochi::Error& error) {
  if (_state != State::Active && _state != State::CommitFailed) {
    MOCHI_ERROR_SET(error, "Cannot abort a committed or discarded episode.");
    return;
  }

  MOCHI_LOG_VERBOSE("Discarding episode '%s': %s", _episodeId.c_str(), std::string{reason}.c_str());
  DiscardTemporaryFile(&error);
}

std::string const& Episode::GetEpisodeId() const {
  return _episodeId;
}

std::filesystem::path const& Episode::GetOutputPath() const {
  return _outputPath;
}

bool Episode::IsActive() const {
  return _state == State::Active;
}

double Episode::GetWallTimeSec() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - _startTime).count();
}

std::string detail::EncodeEntityPathSegment(std::string_view segment) {
  return rerun::escape_entity_path_part(segment);
}

std::string detail::RobotEntityId(char const* actorName) {
  return actorName == nullptr || *actorName == '\0' ? "robot" : actorName;
}

std::string
detail::ShadowEntityPath(char const* actorName, std::string_view id, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(id.empty(), error, "Shadow ID must not be empty.");
  MOCHI_ERROR_RETURN(error, {});
  return std::string{kShadowsPath} + "/" + EncodeEntityPathSegment(RobotEntityId(actorName)) + "/" +
      EncodeEntityPathSegment(id);
}

void Episode::ValidateWrite(mochi::Error& error) const {
  MOCHI_ERROR_IF(_state != State::Active, error, "Cannot write to an inactive episode.");
}

void Episode::ValidateSeriesPath(std::string_view entityPath, mochi::Error& error) const {
  MOCHI_ERROR_IF(entityPath.empty(), error, "Series entity path must not be empty.");
  MOCHI_ERROR_RETURN(error);
  for (auto const& ownedSeries : _series) {
    Series const* const series = ownedSeries.get();
    if (series != nullptr && series->_entityPath == entityPath) {
      MOCHI_LOG_ERROR(
          "A series already exists at entity path '%s'.", std::string{entityPath}.c_str());
      MOCHI_ERROR_SET(error, "A series already exists at this entity path.");
      return;
    }
  }
}

void Episode::SealSeries() {
  for (auto const& ownedSeries : _series) {
    Series* const series = ownedSeries.get();
    if (series != nullptr) {
      series->_sealed = true;
    }
  }
}

void Episode::DiscardTemporaryFile(mochi::Error* error) {
  _state = State::CommitFailed;
  SealSeries();
  _stream.reset();
  std::error_code filesystemError;
  std::filesystem::remove(_temporaryPath, filesystemError);
  if (filesystemError) {
    MOCHI_LOG_ERROR(
        "Failed to remove episode '%s': %s", _episodeId.c_str(), filesystemError.message().c_str());
    if (error != nullptr) {
      error->SetFirstError("Failed to remove the episode RRD.", __FILE__, __LINE__);
    }
    return;
  }
  _state = State::Aborted;
}

Recorder::Recorder(RecorderConfig const& config) : _config(config) {}

Recorder::~Recorder() = default;

std::unique_ptr<Episode> Recorder::BeginEpisode(
    EpisodeMetadata const& metadata,
    mochi::Error& error) const {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(_config.outputDirectory.empty(), error, "Recorder output directory is empty.");
  MOCHI_ERROR_IF_NOT(
      HasValidPropertyKeys(metadata.properties), error, "Episode property keys must not be empty.");
  MOCHI_ERROR_RETURN(error, nullptr);

  std::error_code filesystemError;
  std::filesystem::create_directories(_config.outputDirectory, filesystemError);
  if (filesystemError) {
    MOCHI_LOG_ERROR(
        "Failed to create recorder output directory '%s': %s",
        _config.outputDirectory.string().c_str(),
        filesystemError.message().c_str());
    MOCHI_ERROR_SET(error, "Failed to create the recorder output directory.");
    return nullptr;
  }

  std::string episodeId = GenerateUuidV7(error);
  MOCHI_ERROR_RETURN(error, nullptr);
  auto outputPath = _config.outputDirectory / (episodeId + ".rrd");
  auto temporaryPath = _config.outputDirectory / (episodeId + ".rrd.partial");
  auto stream = std::make_unique<rerun::RecordingStream>(kApplicationId, episodeId);
  if (!stream->is_enabled()) {
    MOCHI_ERROR_SET(error, "Failed to create the episode Rerun recording stream.");
    return nullptr;
  }

  auto episode = std::unique_ptr<Episode>{new Episode{
      std::move(episodeId), std::move(temporaryPath), std::move(outputPath), std::move(stream)}};
  SetRerunError(
      episode->_stream->save(episode->_temporaryPath.string()),
      error,
      "Failed to open the temporary episode RRD.");
  MOCHI_ERROR_RETURN(error, nullptr);

  auto const episodeField = [](std::string component) {
    return ComponentField{std::string{kEpisodeArchetype}, std::move(component)};
  };
  std::array<double, 1> const unitsPerMeter{kSchemaUnitsPerMeter};
  episode->WriteConstantText(
      "episode", episodeField("superdex.episode.schema_name"), kSchemaName, error);
  episode->WriteConstantText(
      "episode",
      episodeField("superdex.episode.schema_version"),
      std::to_string(kSchemaVersionMajor) + "." + std::to_string(kSchemaVersionMinor),
      error);
  episode->WriteConstantText(
      "episode", episodeField("superdex.episode.id"), episode->GetEpisodeId(), error);
  episode->WriteConstantText(
      "episode", episodeField("superdex.episode.started_utc_ns"), UtcNowNanoseconds(), error);
  episode->WriteConstantText(
      "episode", episodeField("superdex.episode.coordinate_space"), kSchemaCoordinateSpace, error);
  episode->WriteConstantScalars(
      "episode", episodeField("superdex.episode.units_per_meter"), unitsPerMeter, error);
  for (auto const& [key, value] : metadata.properties) {
    std::string const path = "episode/properties/" + detail::EncodeEntityPathSegment(key);
    episode->WriteConstantText(path, episodeField("superdex.episode.property_key"), key, error);
    episode->WriteConstantText(path, episodeField("superdex.episode.property_value"), value, error);
  }
  MOCHI_ERROR_RETURN(error, nullptr);
  return episode;
}

void SetLogCallback(mochi::LogFn const& callback) {
  mochi::SetLogCallback(callback);
}

} // namespace superdex::recorder
