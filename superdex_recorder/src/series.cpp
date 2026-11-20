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

#include "chunk_writer.h"

#include <mochi_core/utils/log.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>

namespace superdex::recorder {
namespace {

void SetRerunError(rerun::Error const& rerunError, mochi::Error& error, char const* description) {
  if (rerunError.is_ok()) {
    return;
  }
  MOCHI_LOG_ERROR("%s: %s", description, rerunError.description.c_str());
  error.SetFirstError(description, __FILE__, __LINE__);
}

/* Returns reuse as Rows if it holds Rows, or else new Rows. */
template <typename Rows>
std::unique_ptr<Rows> ReuseRows(std::unique_ptr<detail::SeriesRows> reuse) {
  if (dynamic_cast<Rows*>(reuse.get()) == nullptr) {
    return std::make_unique<Rows>();
  }
  return std::unique_ptr<Rows>{static_cast<Rows*>(reuse.release())};
}

} // namespace

detail::SeriesRows::~SeriesRows() = default;

Series::Series(std::string_view entityPath) : _entityPath{entityPath} {}

Series::~Series() = default;

size_t Series::GetNumSamples() const {
  return _numSamples;
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
  if (_writer != nullptr) {
    _writer->CheckError(error);
  }
  MOCHI_ERROR_RETURN(error);
  _latestTime = time;
}

void Series::AppendTime(SampleTime time, size_t numBytes, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  // The wall_time timeline needs a value in every row.
  MOCHI_ERROR_IF(
      !_frames.empty() && time.wallTimeSec.has_value() == _wallTimesSec.empty(),
      error,
      "Every sample of a series must have a wall time, or none.");
  AdvanceTime(time, error);
  MOCHI_ERROR_RETURN(error);
  if (_writer != nullptr && !_frames.empty() &&
      _writer->IsChunkFull(_bufferedBytes, time.simulationTimeSec - _simulationTimesSec.front())) {
    SubmitChunk();
  }
  _frames.push_back(static_cast<int64_t>(time.frame));
  _simulationTimesSec.push_back(time.simulationTimeSec);
  if (time.wallTimeSec.has_value()) {
    _wallTimesSec.push_back(*time.wallTimeSec);
    numBytes += sizeof(double);
  }
  _bufferedBytes += numBytes + sizeof(int64_t) + sizeof(double);
  ++_numSamples;
}

detail::SeriesChunk Series::TakeChunk(detail::SeriesChunk next) {
  std::swap(_frames, next.frames);
  std::swap(_simulationTimesSec, next.simulationTimesSec);
  std::swap(_wallTimesSec, next.wallTimesSec);
  next.rows = TakeRows(std::move(next.rows));
  next.entityPath = _entityPath;
  next.numBytes = std::exchange(_bufferedBytes, 0);
  return next;
}

void Series::SubmitChunk() {
  if (_frames.empty() || _writer == nullptr) {
    return;
  }
  _writer->Submit(TakeChunk(_writer->TakeEmptyChunk(_entityPath)));
}

ScalarSeries::ScalarSeries(std::string_view entityPath, std::vector<ComponentField> const& fields)
    : Series{entityPath} {
  _rows->columns.reserve(fields.size());
  for (auto const& field : fields) {
    _rows->columns.push_back({.field = field, .lengths = {}, .values = {}});
  }
}

void ScalarSeries::Append(
    SampleTime time,
    mochi::Span<mochi::Span<mochi::real const> const> values,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      values.size() != _rows->columns.size(), error, "Scalar series field count changed.");
  for (auto const& value : values) {
    MOCHI_ERROR_IF(
        value.size() > std::numeric_limits<uint32_t>::max(),
        error,
        "A scalar series field is too large to record.");
  }
  size_t numBytes = 0;
  for (auto const& value : values) {
    numBytes += sizeof(uint32_t) + value.size() * sizeof(Value);
  }
  AppendTime(time, numBytes, error);
  MOCHI_ERROR_RETURN(error);

  auto const* input = values.begin();
  for (auto& column : _rows->columns) {
    column.lengths.push_back(static_cast<uint32_t>(input->size()));
    for (mochi::real const value : *input) {
      column.values.emplace_back(value);
    }
    ++input;
  }
}

std::unique_ptr<detail::SeriesRows> ScalarSeries::TakeRows(
    std::unique_ptr<detail::SeriesRows> reuse) {
  auto next = ReuseRows<Rows>(std::move(reuse));
  next->columns.resize(_rows->columns.size());
  for (size_t i = 0; i < next->columns.size(); ++i) {
    next->columns[i].field = _rows->columns[i].field;
  }
  return std::exchange(_rows, std::move(next));
}

void ScalarSeries::Rows::Clear() {
  for (auto& column : columns) {
    column.lengths.clear();
    column.values.clear();
  }
}

void ScalarSeries::Rows::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) const {
  // `rerun rrd verify` rejects a datatype as a component type, so 32-bit columns declare none.
  constexpr auto kComponentType = std::is_same_v<Value, rerun::components::Scalar>
      ? std::optional{rerun::Loggable<Value>::ComponentType}
      : std::nullopt;
  output.reserve(columns.size());
  for (auto const& column : columns) {
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
  AppendTime(time, sizeof(transform), error);
  MOCHI_ERROR_RETURN(error);
  _rows->translations.emplace_back(transform[0], transform[1], transform[2]);
  _rows->rotations.emplace_back(
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

std::unique_ptr<detail::SeriesRows> TransformSeries::TakeRows(
    std::unique_ptr<detail::SeriesRows> reuse) {
  return std::exchange(_rows, ReuseRows<Rows>(std::move(reuse)));
}

void TransformSeries::Rows::Clear() {
  translations.clear();
  rotations.clear();
}

void TransformSeries::Rows::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) const {
  output.reserve(2);
  auto encodedTranslations = rerun::ComponentColumn::from_loggable(
      rerun::Collection<rerun::components::Translation3D>::borrow(translations),
      rerun::archetypes::Transform3D::Descriptor_translation);
  if (encodedTranslations.is_err()) {
    SetRerunError(
        encodedTranslations.error, error, "Failed to encode a transform translation series.");
    return;
  }
  auto encodedRotations = rerun::ComponentColumn::from_loggable(
      rerun::Collection<rerun::components::RotationQuat>::borrow(rotations),
      rerun::archetypes::Transform3D::Descriptor_quaternion);
  if (encodedRotations.is_err()) {
    SetRerunError(encodedRotations.error, error, "Failed to encode a transform rotation series.");
    return;
  }
  output.emplace_back(std::move(encodedTranslations.value));
  output.emplace_back(std::move(encodedRotations.value));
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
  AppendTime(time, sizeof(uint32_t) + origins.size() * 6 * sizeof(float), error);
  MOCHI_ERROR_RETURN(error);

  _rows->lengths.push_back(static_cast<uint32_t>(origins.size()));
  for (size_t i = 0; i < origins.size(); ++i) {
    _rows->origins.emplace_back(
        static_cast<float>(origins[i][0]),
        static_cast<float>(origins[i][1]),
        static_cast<float>(origins[i][2]));
    _rows->vectors.emplace_back(
        static_cast<float>(vectors[i][0]),
        static_cast<float>(vectors[i][1]),
        static_cast<float>(vectors[i][2]));
  }
}

std::unique_ptr<detail::SeriesRows> Arrow3DSeries::TakeRows(
    std::unique_ptr<detail::SeriesRows> reuse) {
  return std::exchange(_rows, ReuseRows<Rows>(std::move(reuse)));
}

void Arrow3DSeries::Rows::Clear() {
  lengths.clear();
  origins.clear();
  vectors.clear();
}

void Arrow3DSeries::Rows::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) const {
  output.reserve(2);
  auto encodedOrigins = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Position3D>::borrow(origins),
      rerun::Collection<uint32_t>::borrow(lengths),
      rerun::archetypes::Arrows3D::Descriptor_origins);
  if (encodedOrigins.is_err()) {
    SetRerunError(encodedOrigins.error, error, "Failed to encode arrow origins.");
    return;
  }
  auto encodedVectors = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Vector3D>::borrow(vectors),
      rerun::Collection<uint32_t>::borrow(lengths),
      rerun::archetypes::Arrows3D::Descriptor_vectors);
  if (encodedVectors.is_err()) {
    SetRerunError(encodedVectors.error, error, "Failed to encode arrow vectors.");
    return;
  }
  output.emplace_back(std::move(encodedOrigins.value));
  output.emplace_back(std::move(encodedVectors.value));
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
  bool const repeatsLastSample = GetNumSamples() != 0 &&
      _lastPositions.size() == positions.size() &&
      std::equal(positions.begin(),
                 positions.end(),
                 _lastPositions.begin(),
                 [&](mochi::Real3 const& position, rerun::components::Position3D const& stored) {
                   return toPosition(position).xyz.xyz == stored.xyz.xyz;
                 });
  if (repeatsLastSample) {
    AdvanceTime(time, error);
    return;
  }
  AppendTime(time, sizeof(uint32_t) + positions.size() * 3 * sizeof(float), error);
  MOCHI_ERROR_RETURN(error);

  _rows->lengths.push_back(static_cast<uint32_t>(positions.size()));
  _lastPositions.clear();
  std::transform(
      positions.begin(), positions.end(), std::back_inserter(_lastPositions), toPosition);
  _rows->positions.insert(_rows->positions.end(), _lastPositions.begin(), _lastPositions.end());
}

std::unique_ptr<detail::SeriesRows> Points3DSeries::TakeRows(
    std::unique_ptr<detail::SeriesRows> reuse) {
  return std::exchange(_rows, ReuseRows<Rows>(std::move(reuse)));
}

void Points3DSeries::Rows::Clear() {
  lengths.clear();
  positions.clear();
}

void Points3DSeries::Rows::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) const {
  auto encodedPositions = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Position3D>::borrow(positions),
      rerun::Collection<uint32_t>::borrow(lengths),
      rerun::archetypes::Points3D::Descriptor_positions);
  if (encodedPositions.is_err()) {
    SetRerunError(encodedPositions.error, error, "Failed to encode point positions.");
    return;
  }
  output.emplace_back(std::move(encodedPositions.value));
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
  for (size_t i = 0; i < 3 * _numVertices && repeatsLastSample; ++i) {
    repeatsLastSample =
        static_cast<float>(vertexPositions[i]) == _lastPositions[i / 3].xyz.xyz[i % 3] &&
        static_cast<float>(vertexNormals[i]) == _lastNormals[i / 3].vector.xyz[i % 3];
  }
  if (repeatsLastSample) {
    AdvanceTime(time, error);
    return;
  }
  AppendTime(time, sizeof(uint32_t) + _numVertices * 6 * sizeof(float), error);
  MOCHI_ERROR_RETURN(error);

  _lastPositions.clear();
  _lastNormals.clear();
  for (size_t i = 0; i < _numVertices; ++i) {
    _lastPositions.emplace_back(
        static_cast<float>(vertexPositions[3 * i]),
        static_cast<float>(vertexPositions[3 * i + 1]),
        static_cast<float>(vertexPositions[3 * i + 2]));
    _lastNormals.emplace_back(
        static_cast<float>(vertexNormals[3 * i]),
        static_cast<float>(vertexNormals[3 * i + 1]),
        static_cast<float>(vertexNormals[3 * i + 2]));
  }
  _rows->lengths.push_back(static_cast<uint32_t>(_numVertices));
  _rows->positions.insert(_rows->positions.end(), _lastPositions.begin(), _lastPositions.end());
  _rows->normals.insert(_rows->normals.end(), _lastNormals.begin(), _lastNormals.end());
}

std::unique_ptr<detail::SeriesRows> DeformableMeshSeries::TakeRows(
    std::unique_ptr<detail::SeriesRows> reuse) {
  return std::exchange(_rows, ReuseRows<Rows>(std::move(reuse)));
}

void DeformableMeshSeries::Rows::Clear() {
  lengths.clear();
  positions.clear();
  normals.clear();
}

void DeformableMeshSeries::Rows::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) const {
  auto encodedPositions = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Position3D>::borrow(positions),
      rerun::Collection<uint32_t>::borrow(lengths),
      rerun::archetypes::Mesh3D::Descriptor_vertex_positions);
  if (encodedPositions.is_err()) {
    SetRerunError(encodedPositions.error, error, "Failed to encode mesh vertex positions.");
    return;
  }
  auto encodedNormals = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Vector3D>::borrow(normals),
      rerun::Collection<uint32_t>::borrow(lengths),
      rerun::archetypes::Mesh3D::Descriptor_vertex_normals);
  if (encodedNormals.is_err()) {
    SetRerunError(encodedNormals.error, error, "Failed to encode mesh vertex normals.");
    return;
  }
  output.reserve(2);
  output.emplace_back(std::move(encodedPositions.value));
  output.emplace_back(std::move(encodedNormals.value));
}

TextSeries::TextSeries(std::string_view entityPath, ComponentField field) : Series{entityPath} {
  _rows->field = std::move(field);
}

void TextSeries::Append(
    SampleTime time,
    mochi::Span<std::string const> texts,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      texts.size() > std::numeric_limits<uint32_t>::max(),
      error,
      "Text series sample is too large to record.");
  size_t numBytes = sizeof(uint32_t);
  for (auto const& text : texts) {
    numBytes += text.size();
  }
  AppendTime(time, numBytes, error);
  MOCHI_ERROR_RETURN(error);

  _rows->lengths.push_back(static_cast<uint32_t>(texts.size()));
  for (auto const& text : texts) {
    _rows->texts.emplace_back(text);
  }
}

std::unique_ptr<detail::SeriesRows> TextSeries::TakeRows(
    std::unique_ptr<detail::SeriesRows> reuse) {
  auto next = ReuseRows<Rows>(std::move(reuse));
  next->field = _rows->field;
  return std::exchange(_rows, std::move(next));
}

void TextSeries::Rows::Clear() {
  lengths.clear();
  texts.clear();
}

void TextSeries::Rows::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) const {
  auto encodedTexts = rerun::ComponentColumn::from_loggable_with_lengths(
      rerun::Collection<rerun::components::Text>::borrow(texts),
      rerun::Collection<uint32_t>::borrow(lengths),
      rerun::ComponentDescriptor{
          field.archetype,
          field.component,
          rerun::Loggable<rerun::components::Text>::ComponentType});
  if (encodedTexts.is_err()) {
    SetRerunError(encodedTexts.error, error, "Failed to encode a text series.");
    return;
  }
  output.emplace_back(std::move(encodedTexts.value));
}

PlotSeries::PlotSeries(std::string_view entityPath) : Series{entityPath} {}

void PlotSeries::Append(SampleTime time, double value, mochi::Error& error) {
  AppendTime(time, sizeof(double), error);
  MOCHI_ERROR_RETURN(error);
  _rows->values.emplace_back(value);
}

std::unique_ptr<detail::SeriesRows> PlotSeries::TakeRows(
    std::unique_ptr<detail::SeriesRows> reuse) {
  return std::exchange(_rows, ReuseRows<Rows>(std::move(reuse)));
}

void PlotSeries::Rows::Clear() {
  values.clear();
}

void PlotSeries::Rows::BuildColumns(
    std::vector<rerun::ComponentColumn>& output,
    mochi::Error& error) const {
  auto encodedValues = rerun::ComponentColumn::from_loggable(
      rerun::Collection<rerun::components::Scalar>::borrow(values),
      rerun::archetypes::Scalars::Descriptor_scalars);
  if (encodedValues.is_err()) {
    SetRerunError(encodedValues.error, error, "Failed to encode a plot series.");
    return;
  }
  output.emplace_back(std::move(encodedValues.value));
}

} // namespace superdex::recorder
