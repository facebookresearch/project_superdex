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

#include <superdex_recorder/source_recorder.h>

#include <superdex_recorder/series.h>

#include <mochi_core/utils/log.h>

#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace superdex::recorder {
namespace {

constexpr std::string_view kSourcesPath = "world/sources/";
constexpr std::string_view kEventsPath = "data/events/";
constexpr std::string_view kScalarsPath = "data/scalars/";

ComponentField const kTextField{"rerun.archetypes.TextLog", "TextLog:text"};
ComponentField const kScalarField{"superdex.archetypes.SourceScalar", "superdex.sources.scalar"};

std::string StreamPath(std::string_view prefix, std::string_view name, mochi::Error& error) {
  MOCHI_ERROR_IF(name.empty(), error, "A source stream name must not be empty.");
  MOCHI_ERROR_RETURN(error, {});
  return std::string{prefix} + detail::EncodeEntityPathSegment(name);
}

// Finds the series recorded under `name`, creating it with `create(path)` the first time.
template <typename SeriesType, typename Create>
SeriesType* FindOrCreate(
    std::map<std::string, SeriesType*, std::less<>>& seriesByName,
    std::string_view prefix,
    std::string_view name,
    mochi::Error& error,
    Create&& create) {
  MOCHI_ERROR_RETURN(error, nullptr);
  if (auto const found = seriesByName.find(name); found != seriesByName.end()) {
    return found->second;
  }
  std::string const path = StreamPath(prefix, name, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  SeriesType* const series = create(path);
  MOCHI_ERROR_RETURN(error, nullptr);
  seriesByName.emplace(std::string{name}, series);
  return series;
}

} // namespace

SourceRecorder::SourceRecorder(Episode& episode) : _episode{episode} {}

SourceRecorder::~SourceRecorder() = default;

void SourceRecorder::AddSkeleton(
    std::string_view id,
    SourceSkeletonConfig const& config,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::string const path = StreamPath(kSourcesPath, id, error);
  MOCHI_ERROR_RETURN(error);
  size_t const numKeypoints = config.keypointNames.size();
  MOCHI_ERROR_IF(
      _skeletons.find(id) != _skeletons.end(),
      error,
      "A source skeleton with this ID already exists.");
  MOCHI_ERROR_IF(numKeypoints == 0, error, "A source skeleton must have at least one keypoint.");
  // Rerun keypoint IDs are uint16.
  MOCHI_ERROR_IF(
      numKeypoints > size_t{std::numeric_limits<uint16_t>::max()} + 1,
      error,
      "A source skeleton has too many keypoints.");
  MOCHI_ERROR_IF(
      config.parentIndices.size() != numKeypoints,
      error,
      "Source skeleton parent indices and keypoint names differ in size.");
  std::set<std::string_view> const uniqueNames(
      config.keypointNames.begin(), config.keypointNames.end());
  MOCHI_ERROR_IF(
      uniqueNames.size() != numKeypoints || uniqueNames.count("") != 0,
      error,
      "Source skeleton keypoint names must be unique and not empty.");
  MOCHI_ERROR_RETURN(error);

  std::vector<std::array<uint16_t, 2>> connections;
  for (size_t i = 0; i < numKeypoints; ++i) {
    int32_t const parent = config.parentIndices[i];
    if (parent < -1 || parent >= static_cast<int64_t>(i)) {
      MOCHI_LOG_ERROR(
          "Keypoint %zu of the source skeleton at '%s' has parent %d.", i, path.c_str(), parent);
      MOCHI_ERROR_SET(error, "A source skeleton keypoint's parent must precede it or be -1.");
      return;
    }
    if (parent >= 0) {
      connections.push_back({static_cast<uint16_t>(parent), static_cast<uint16_t>(i)});
    }
  }

  Points3DSeries* const series = _episode.CreatePoints3DSeries(path, error);
  MOCHI_ERROR_RETURN(error);
  _episode.WriteKeypointSkeleton(
      path,
      config.label.empty() ? id : std::string_view{config.label},
      config.keypointNames,
      connections,
      config.colorRgba,
      config.keypointRadiusMeters,
      error);
  // Without its skeleton, the series would record bare points.
  MOCHI_ERROR_RETURN(error);
  _skeletons.emplace(std::string{id}, Skeleton{numKeypoints, series});
}

void SourceRecorder::RecordSkeleton(
    SampleTime time,
    std::string_view id,
    mochi::Span<mochi::Real3 const> worldKeypoints,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const found = _skeletons.find(id);
  MOCHI_ERROR_IF(found == _skeletons.end(), error, "Source skeleton was not added.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      worldKeypoints.size() != found->second.numKeypoints,
      error,
      "Source skeleton keypoint count changed.");
  for (mochi::Real3 const& keypoint : worldKeypoints) {
    MOCHI_ERROR_IF_NOT(
        std::isfinite(keypoint[0]) && std::isfinite(keypoint[1]) && std::isfinite(keypoint[2]),
        error,
        "Source skeleton keypoints must be finite.");
  }
  MOCHI_ERROR_RETURN(error);
  found->second.series->Append(time, worldKeypoints, error);
}

void SourceRecorder::RecordSkeletonUntracked(
    SampleTime time,
    std::string_view id,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const found = _skeletons.find(id);
  MOCHI_ERROR_IF(found == _skeletons.end(), error, "Source skeleton was not added.");
  MOCHI_ERROR_RETURN(error);
  found->second.series->Append(time, {}, error);
}

void SourceRecorder::RecordTransform(
    SampleTime time,
    std::string_view id,
    mochi::TransformRT const& worldTransform,
    mochi::Error& error) {
  TransformSeries* const series =
      FindOrCreate(_transforms, kSourcesPath, id, error, [&](std::string const& path) {
        return _episode.CreateTransformSeries(path, error);
      });
  MOCHI_ERROR_RETURN(error);
  series->Append(time, worldTransform, error);
}

void SourceRecorder::RecordTexts(
    SampleTime time,
    std::string_view stream,
    mochi::Span<std::string const> texts,
    mochi::Error& error) {
  TextSeries* const series =
      FindOrCreate(_texts, kEventsPath, stream, error, [&](std::string const& path) {
        return _episode.CreateTextSeries(path, kTextField, error);
      });
  MOCHI_ERROR_RETURN(error);
  series->Append(time, texts, error);
}

void SourceRecorder::RecordScalar(
    SampleTime time,
    std::string_view name,
    mochi::real value,
    mochi::Error& error) {
  std::array<mochi::real, 1> const values{value};
  RecordScalars(time, name, values, error);
}

void SourceRecorder::RecordScalars(
    SampleTime time,
    std::string_view name,
    mochi::Span<mochi::real const> values,
    mochi::Error& error) {
  ScalarSeries* const series =
      FindOrCreate(_scalars, kScalarsPath, name, error, [&](std::string const& path) {
        return _episode.CreateScalarSeries(path, {kScalarField}, error);
      });
  MOCHI_ERROR_RETURN(error);
  std::array<mochi::Span<mochi::real const>, 1> const fields{values};
  series->Append(time, fields, error);
}

} // namespace superdex::recorder
