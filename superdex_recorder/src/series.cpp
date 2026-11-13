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

#include <superdex_recorder/series.h>

#include <mochi_core/utils/log.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>

namespace superdex::recorder {
namespace {

constexpr std::string_view kFrameTimeline = "frame";
constexpr std::string_view kSimulationTimeTimeline = "sim_time";
constexpr std::string_view kWallTimeTimeline = "wall_time";

void SetRerunError(rerun::Error const& rerunError, mochi::Error& error, char const* description) {
  if (rerunError.is_ok()) {
    return;
  }
  MOCHI_LOG_ERROR("%s: %s", description, rerunError.description.c_str());
  error.SetFirstError(description, __FILE__, __LINE__);
}

} // namespace

Series::Series(std::string_view entityPath) : _entityPath{entityPath} {}

Series::~Series() = default;

size_t Series::GetNumSamples() const {
  return _frames.size();
}

void Series::AdvanceTime(SampleTime time, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(_sealed, error, "Cannot append to a committed or aborted series.");
  MOCHI_ERROR_IF(
      time.frame > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      error,
      "Series frame exceeds the supported range.");
  MOCHI_ERROR_IF_NOT(
      std::isfinite(time.simulationTimeSec) && time.simulationTimeSec >= 0.0,
      error,
      "Series time must be finite and non-negative.");
  MOCHI_ERROR_IF(
      time.wallTimeSec.has_value() &&
          !(std::isfinite(*time.wallTimeSec) && *time.wallTimeSec >= 0.0),
      error,
      "Series wall time must be finite and non-negative.");
  if (_latestTime.has_value()) {
    MOCHI_ERROR_IF_NOT(time.frame > _latestTime->frame, error, "Series frames must increase.");
    MOCHI_ERROR_IF_NOT(
        time.simulationTimeSec >= _latestTime->simulationTimeSec,
        error,
        "Series time must not decrease.");
    MOCHI_ERROR_IF(
        time.wallTimeSec.has_value() && _latestTime->wallTimeSec.has_value() &&
            *time.wallTimeSec < *_latestTime->wallTimeSec,
        error,
        "Series wall time must not decrease.");
  }
  MOCHI_ERROR_RETURN(error);
  _latestTime = time;
}

void Series::AppendTime(SampleTime time, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  // The wall_time timeline needs a value in every row.
  MOCHI_ERROR_IF(
      !_frames.empty() && time.wallTimeSec.has_value() == _wallTimesSec.empty(),
      error,
      "Every sample of a series must have a wall time, or none.");
  AdvanceTime(time, error);
  MOCHI_ERROR_RETURN(error);
  _frames.push_back(static_cast<int64_t>(time.frame));
  _simulationTimesSec.push_back(time.simulationTimeSec);
  if (time.wallTimeSec.has_value()) {
    _wallTimesSec.push_back(*time.wallTimeSec);
  }
}

void Series::Flush(rerun::RecordingStream& recording, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  _sealed = true;
  if (_frames.empty()) {
    return;
  }

  std::vector<rerun::ComponentColumn> componentColumns;
  BuildColumns(componentColumns, error);
  MOCHI_ERROR_RETURN(error);

  std::vector<rerun::TimeColumn> timeColumns;
  timeColumns.reserve(3);
  timeColumns.emplace_back(
      rerun::TimeColumn::from_sequence(
          std::string{kFrameTimeline}, std::move(_frames), rerun::SortingStatus::Sorted));
  timeColumns.emplace_back(
      rerun::TimeColumn::from_duration_secs(
          std::string{kSimulationTimeTimeline},
          std::move(_simulationTimesSec),
          rerun::SortingStatus::Sorted));
  if (!_wallTimesSec.empty()) {
    timeColumns.emplace_back(
        rerun::TimeColumn::from_duration_secs(
            std::string{kWallTimeTimeline},
            std::move(_wallTimesSec),
            rerun::SortingStatus::Sorted));
  }
  SetRerunError(
      recording.try_send_columns(
          _entityPath,
          rerun::Collection<rerun::TimeColumn>::take_ownership(std::move(timeColumns)),
          rerun::Collection<rerun::ComponentColumn>::take_ownership(std::move(componentColumns))),
      error,
      "Failed to write sampled series.");
}

ScalarSeries::ScalarSeries(std::string_view entityPath, std::vector<ComponentField> const& fields)
    : Series{entityPath} {
  _columns.reserve(fields.size());
  for (auto const& field : fields) {
    _columns.push_back({.field = field});
  }
}

void ScalarSeries::Append(
    SampleTime time,
    mochi::Span<mochi::Span<mochi::real const> const> values,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(values.size() != _columns.size(), error, "Scalar series field count changed.");
  for (auto const& value : values) {
    MOCHI_ERROR_IF(
        value.size() > std::numeric_limits<uint32_t>::max(),
        error,
        "A scalar series field is too large to record.");
  }
  AppendTime(time, error);
  MOCHI_ERROR_RETURN(error);

  auto const* input = values.begin();
  for (auto& column : _columns) {
    column.lengths.push_back(static_cast<uint32_t>(input->size()));
    for (mochi::real const value : *input) {
      column.values.emplace_back(value);
    }
    ++input;
  }
}

void ScalarSeries::BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) {
  // `rerun rrd verify` rejects a datatype as a component type, so 32-bit columns declare none.
  constexpr auto kComponentType = std::is_same_v<Value, rerun::components::Scalar>
      ? std::optional{rerun::Loggable<Value>::ComponentType}
      : std::nullopt;
  output.reserve(_columns.size());
  for (auto const& column : _columns) {
    auto encoded = rerun::ComponentColumn::from_loggable_with_lengths(
        rerun::Collection<Value>::borrow(column.values),
        rerun::Collection<uint32_t>::borrow(column.lengths),
        rerun::ComponentDescriptor{column.field.archetype, column.field.component, kComponentType});
    if (encoded.is_err()) {
      SetRerunError(encoded.error, error, "Failed to encode a scalar series.");
      return;
    }
    output.emplace_back(std::move(encoded.value));
  }
}

TransformSeries::TransformSeries(std::string_view entityPath) : Series{entityPath} {}

void TransformSeries::Append(
    SampleTime time,
    std::array<mochi::real, 3> const& translation,
    std::array<mochi::real, 4> const& quaternionXyzw,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::array<float, 7> const transform{
      static_cast<float>(translation[0]),
      static_cast<float>(translation[1]),
      static_cast<float>(translation[2]),
      static_cast<float>(quaternionXyzw[0]),
      static_cast<float>(quaternionXyzw[1]),
      static_cast<float>(quaternionXyzw[2]),
      static_cast<float>(quaternionXyzw[3])};
  if (transform == _lastTransform) {
    AdvanceTime(time, error);
    return;
  }
  AppendTime(time, error);
  MOCHI_ERROR_RETURN(error);
  _translations.emplace_back(transform[0], transform[1], transform[2]);
  _rotations.emplace_back(
      rerun::datatypes::Quaternion::from_xyzw(
          transform[3], transform[4], transform[5], transform[6]));
  _lastTransform = transform;
}

void TransformSeries::Append(
    SampleTime time,
    mochi::TransformRT const& transform,
    mochi::Error& error) {
  auto const translation = transform.GetTranslation();
  auto const& rotation = transform.GetRotation();
  std::array<mochi::real, 3> const translationArray{translation[0], translation[1], translation[2]};
  std::array<mochi::real, 4> const quaternion{
      rotation.data[0], rotation.data[1], rotation.data[2], rotation.data[3]};
  Append(time, translationArray, quaternion, error);
}

void TransformSeries::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) {
  output.reserve(2);
  auto translations = rerun::ComponentColumn::from_loggable(
      rerun::Collection<rerun::components::Translation3D>::borrow(_translations),
      rerun::archetypes::Transform3D::Descriptor_translation);
  if (translations.is_err()) {
    SetRerunError(translations.error, error, "Failed to encode a transform translation series.");
    return;
  }
  auto rotations = rerun::ComponentColumn::from_loggable(
      rerun::Collection<rerun::components::RotationQuat>::borrow(_rotations),
      rerun::archetypes::Transform3D::Descriptor_quaternion);
  if (rotations.is_err()) {
    SetRerunError(rotations.error, error, "Failed to encode a transform rotation series.");
    return;
  }
  output.emplace_back(std::move(translations.value));
  output.emplace_back(std::move(rotations.value));
}

Arrow3DSeries::Arrow3DSeries(std::string_view entityPath) : Series{entityPath} {}

void Arrow3DSeries::Append(
    SampleTime time,
    mochi::Span<mochi::Real3 const> origins,
    mochi::Span<mochi::Real3 const> vectors,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      origins.size() != vectors.size(), error, "Arrow origins and vectors differ in size.");
  MOCHI_ERROR_IF(
      origins.size() > std::numeric_limits<uint32_t>::max(),
      error,
      "Arrow series sample is too large to record.");
  MOCHI_ERROR_RETURN(error);
  AppendTime(time, error);
  MOCHI_ERROR_RETURN(error);

  _lengths.push_back(static_cast<uint32_t>(origins.size()));
  for (size_t i = 0; i < origins.size(); ++i) {
    _origins.emplace_back(
        static_cast<float>(origins[i][0]),
        static_cast<float>(origins[i][1]),
        static_cast<float>(origins[i][2]));
    _vectors.emplace_back(
        static_cast<float>(vectors[i][0]),
        static_cast<float>(vectors[i][1]),
        static_cast<float>(vectors[i][2]));
  }
}

void Arrow3DSeries::BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) {
  output.reserve(2);
  auto origins = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Position3D>::borrow(_origins),
      rerun::Collection<uint32_t>::borrow(_lengths),
      rerun::archetypes::Arrows3D::Descriptor_origins);
  if (origins.is_err()) {
    SetRerunError(origins.error, error, "Failed to encode arrow origins.");
    return;
  }
  auto vectors = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Vector3D>::borrow(_vectors),
      rerun::Collection<uint32_t>::borrow(_lengths),
      rerun::archetypes::Arrows3D::Descriptor_vectors);
  if (vectors.is_err()) {
    SetRerunError(vectors.error, error, "Failed to encode arrow vectors.");
    return;
  }
  output.emplace_back(std::move(origins.value));
  output.emplace_back(std::move(vectors.value));
}

Points3DSeries::Points3DSeries(std::string_view entityPath) : Series{entityPath} {}

void Points3DSeries::Append(
    SampleTime time,
    mochi::Span<mochi::Real3 const> positions,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      positions.size() > std::numeric_limits<uint32_t>::max(),
      error,
      "Point series sample is too large to record.");
  MOCHI_ERROR_RETURN(error);
  auto const toPosition = [](mochi::Real3 const& position) {
    return rerun::components::Position3D{
        static_cast<float>(position[0]),
        static_cast<float>(position[1]),
        static_cast<float>(position[2])};
  };
  bool const repeatsLastSample = !_lengths.empty() && _lengths.back() == positions.size() &&
      std::equal(
          positions.begin(),
          positions.end(),
          _positions.end() - static_cast<std::ptrdiff_t>(positions.size()),
          [&](mochi::Real3 const& position, rerun::components::Position3D const& stored) {
            return toPosition(position).xyz.xyz == stored.xyz.xyz;
          });
  if (repeatsLastSample) {
    AdvanceTime(time, error);
    return;
  }
  AppendTime(time, error);
  MOCHI_ERROR_RETURN(error);

  _lengths.push_back(static_cast<uint32_t>(positions.size()));
  std::ranges::transform(positions, std::back_inserter(_positions), toPosition);
}

void Points3DSeries::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) {
  auto positions = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Position3D>::borrow(_positions),
      rerun::Collection<uint32_t>::borrow(_lengths),
      rerun::archetypes::Points3D::Descriptor_positions);
  if (positions.is_err()) {
    SetRerunError(positions.error, error, "Failed to encode point positions.");
    return;
  }
  output.emplace_back(std::move(positions.value));
}

DeformableMeshSeries::DeformableMeshSeries(std::string_view entityPath, size_t numVertices)
    : Series{entityPath}, _numVertices{numVertices} {}

void DeformableMeshSeries::Append(
    SampleTime time,
    mochi::Span<mochi::real const> vertexPositions,
    mochi::Span<mochi::real const> vertexNormals,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      vertexPositions.size() != 3 * _numVertices || vertexNormals.size() != 3 * _numVertices,
      error,
      "Mesh sample does not have one position and one normal per vertex.");
  MOCHI_ERROR_RETURN(error);
  bool repeatsLastSample = GetNumSamples() != 0;
  size_t const lastSample = _positions.size() - (repeatsLastSample ? _numVertices : 0);
  for (size_t i = 0; i < 3 * _numVertices && repeatsLastSample; ++i) {
    repeatsLastSample =
        static_cast<float>(vertexPositions[i]) == _positions[lastSample + i / 3].xyz.xyz[i % 3] &&
        static_cast<float>(vertexNormals[i]) == _normals[lastSample + i / 3].vector.xyz[i % 3];
  }
  if (repeatsLastSample) {
    AdvanceTime(time, error);
    return;
  }
  AppendTime(time, error);
  MOCHI_ERROR_RETURN(error);

  for (size_t i = 0; i < _numVertices; ++i) {
    _positions.emplace_back(
        static_cast<float>(vertexPositions[3 * i]),
        static_cast<float>(vertexPositions[3 * i + 1]),
        static_cast<float>(vertexPositions[3 * i + 2]));
    _normals.emplace_back(
        static_cast<float>(vertexNormals[3 * i]),
        static_cast<float>(vertexNormals[3 * i + 1]),
        static_cast<float>(vertexNormals[3 * i + 2]));
  }
}

void DeformableMeshSeries::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) {
  std::vector<uint32_t> const lengths(GetNumSamples(), static_cast<uint32_t>(_numVertices));
  auto positions = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Position3D>::borrow(_positions),
      rerun::Collection<uint32_t>::borrow(lengths),
      rerun::archetypes::Mesh3D::Descriptor_vertex_positions);
  if (positions.is_err()) {
    SetRerunError(positions.error, error, "Failed to encode mesh vertex positions.");
    return;
  }
  auto normals = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Vector3D>::borrow(_normals),
      rerun::Collection<uint32_t>::borrow(lengths),
      rerun::archetypes::Mesh3D::Descriptor_vertex_normals);
  if (normals.is_err()) {
    SetRerunError(normals.error, error, "Failed to encode mesh vertex normals.");
    return;
  }
  output.reserve(2);
  output.emplace_back(std::move(positions.value));
  output.emplace_back(std::move(normals.value));
}

TextSeries::TextSeries(std::string_view entityPath, ComponentField field)
    : Series{entityPath}, _field{std::move(field)} {}

void TextSeries::Append(
    SampleTime time,
    mochi::Span<std::string const> texts,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      texts.size() > std::numeric_limits<uint32_t>::max(),
      error,
      "Text series sample is too large to record.");
  AppendTime(time, error);
  MOCHI_ERROR_RETURN(error);

  _lengths.push_back(static_cast<uint32_t>(texts.size()));
  for (auto const& text : texts) {
    _texts.emplace_back(text);
  }
}

void TextSeries::BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) {
  auto texts = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Text>::borrow(_texts),
      rerun::Collection<uint32_t>::borrow(_lengths),
      rerun::ComponentDescriptor{
          _field.archetype,
          _field.component,
          rerun::Loggable<rerun::components::Text>::ComponentType});
  if (texts.is_err()) {
    SetRerunError(texts.error, error, "Failed to encode a text series.");
    return;
  }
  output.emplace_back(std::move(texts.value));
}

PlotSeries::PlotSeries(std::string_view entityPath) : Series{entityPath} {}

void PlotSeries::Append(SampleTime time, double value, mochi::Error& error) {
  AppendTime(time, error);
  MOCHI_ERROR_RETURN(error);
  _values.emplace_back(value);
}

void PlotSeries::BuildColumns(std::vector<rerun::ComponentColumn>& output, mochi::Error& error) {
  auto values = rerun::ComponentColumn::from_loggable(
      rerun::Collection<rerun::components::Scalar>::borrow(_values),
      rerun::archetypes::Scalars::Descriptor_scalars);
  if (values.is_err()) {
    SetRerunError(values.error, error, "Failed to encode a plot series.");
    return;
  }
  output.emplace_back(std::move(values.value));
}

} // namespace superdex::recorder
