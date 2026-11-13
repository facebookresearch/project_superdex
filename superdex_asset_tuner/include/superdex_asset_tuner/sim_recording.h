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

#include "superdex_asset_tuner/run_writer.h"
#include "superdex_asset_tuner/tuner_session.h"
#include "superdex_asset_tuner/tuner_spec.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace superdex::tuner {

// Where a run's sim.rrd keeps the tuner's own data. The scene's actors are under data/physics, as
// superdex_recorder's PhysicsRecorder writes them.
inline constexpr std::string_view kRecordingTunerPath = "data/tuner";
// One plot series ("Scalars:scalars") per channel, named after the channel.
inline constexpr std::string_view kRecordingChannelsPath = "data/tuner/channels";
// One series per load, named after the load, with the kRecordingLoad* components.
inline constexpr std::string_view kRecordingLoadsPath = "data/tuner/loads";
// Static texts on kRecordingTunerPath.
inline constexpr std::string_view kRecordingSpecJson = "superdex.tuner.resolved_spec_json";
inline constexpr std::string_view kRecordingSummaryJson = "superdex.tuner.summary_json";
inline constexpr std::string_view kRecordingChannelNames = "superdex.tuner.channel_names";
inline constexpr std::string_view kRecordingLoadNames = "superdex.tuner.load_names";
// AppliedLoad per step: xyz, xyz, xyz and xyzw.
inline constexpr std::string_view kRecordingLoadPoint = "superdex.tuner.load_point_world";
inline constexpr std::string_view kRecordingLoadVector = "superdex.tuner.load_vector_world";
inline constexpr std::string_view kRecordingLoadTargetTranslation =
    "superdex.tuner.load_target_translation";
inline constexpr std::string_view kRecordingLoadTargetRotation =
    "superdex.tuner.load_target_quaternion_xyzw";

// Records a tuner run to sim.rrd, a SuperDex episode to open in the Rerun viewer: the scene's
// actors, every channel as a series the viewer plots, each load as it was applied, the resolved
// spec and the run summary. Not thread-safe.
class SimRecording final {
 public:
  SimRecording();
  ~SimRecording();
  SimRecording(SimRecording const&) = delete;
  SimRecording& operator=(SimRecording const&) = delete;
  SimRecording(SimRecording&&) = delete;
  SimRecording& operator=(SimRecording&&) = delete;

  // False when built without superdex_recorder: Start fails, and runs write only their CSV and JSON
  // files.
  [[nodiscard]] static bool IsSupported();

  // Starts recording the scene's current actors, `session`'s channels and loads, and
  // `resolvedSpec`. Call after TunerSession::Bind, on the thread that steps the scene. The file is
  // written by Finish.
  bool Start(
      std::filesystem::path const& path,
      mochi::Scene& scene,
      TunerSession const& session,
      TunerSpec const& resolvedSpec,
      std::string& outError);

  // Records the scene and `sample` after a step, on the scene's thread. Steps keep being counted
  // until StopSampling, so call it for every step that should be recorded.
  void Record(Sample const& sample);

  // Releases the scene, on its thread. The scene may then be destroyed; Finish may follow on any
  // thread.
  void StopSampling();

  // Stops sampling if needed, then writes the file with `summary`. On failure no file is left.
  bool Finish(RunSummary const& summary, std::string& outError);

  // Stops sampling if needed and discards the recording.
  void Abort();

  // Between Start and Finish or Abort.
  [[nodiscard]] bool IsActive() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace superdex::tuner
