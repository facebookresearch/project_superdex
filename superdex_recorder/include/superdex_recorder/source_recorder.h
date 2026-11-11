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

#include <superdex_recorder/recorder.h>

#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::recorder {

/* A keypoint skeleton that an application records on its own, such as a tracked hand. */
struct SourceSkeletonConfig {
  /* Labels the skeleton in the viewer; empty uses the source ID. */
  std::string label;
  /* Names each keypoint, in the order RecordSkeleton takes them. */
  std::vector<std::string> keypointNames;
  /* Each keypoint's parent keypoint index, which must be lower than its own, or -1 for none. */
  std::vector<int32_t> parentIndices;
  std::array<uint8_t, 4> colorRgba{242, 140, 38, 255};
  float keypointRadiusMeters = 0.004f;
};

/*
 * Records streams an application provides beside the physics and robots, such as a headset pose,
 * a tracked hand, the events of a study, or values it computes:
 * - transforms and keypoint skeletons under "world/sources/<id>"
 * - texts under "data/events/<stream>", as a Rerun text log
 * - scalars, or lists of scalars, under "data/scalars/<name>"
 *
 * Each stream's series is created the first time the stream is recorded. Like every series, a
 * stream takes at most one sample per frame, and its frames must increase.
 */
class SUPERDEX_RECORDER_API SourceRecorder final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(SourceRecorder)

 public:
  explicit SourceRecorder(Episode& episode);
  ~SourceRecorder();

  /* Declares the keypoint skeleton of source `id`, before its first RecordSkeleton. */
  void AddSkeleton(std::string_view id, SourceSkeletonConfig const& config, mochi::Error& error);
  /* Records one finite world position per keypoint of a declared skeleton, in configuration
   * order. */
  void RecordSkeleton(
      SampleTime time,
      std::string_view id,
      mochi::Span<mochi::Real3 const> worldKeypoints,
      mochi::Error& error);
  /* Hides a declared skeleton until its next RecordSkeleton. */
  void RecordSkeletonUntracked(SampleTime time, std::string_view id, mochi::Error& error);

  void RecordTransform(
      SampleTime time,
      std::string_view id,
      mochi::TransformRT const& worldTransform,
      mochi::Error& error);
  /* Records every text of one frame on `stream`. */
  void RecordTexts(
      SampleTime time,
      std::string_view stream,
      mochi::Span<std::string const> texts,
      mochi::Error& error);
  void RecordScalar(SampleTime time, std::string_view name, mochi::real value, mochi::Error& error);
  /* Records a list of scalars as one sample of `name`. The recording doesn't fix the list's length;
   * readers interpret the values in the order the application wrote them. */
  void RecordScalars(
      SampleTime time,
      std::string_view name,
      mochi::Span<mochi::real const> values,
      mochi::Error& error);

 private:
  struct Skeleton {
    size_t numKeypoints = 0;
    Points3DSeries* series = nullptr;
  };

  Episode& _episode;
  std::map<std::string, Skeleton, std::less<>> _skeletons;
  std::map<std::string, TransformSeries*, std::less<>> _transforms;
  std::map<std::string, TextSeries*, std::less<>> _texts;
  std::map<std::string, ScalarSeries*, std::less<>> _scalars;
};

} // namespace superdex::recorder
