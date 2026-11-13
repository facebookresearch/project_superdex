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

#include "superdex_asset_tuner/sim_recording.h"

#include <superdex_recorder/physics_recorder.h>
#include <superdex_recorder/recorder.h>
#include <superdex_recorder/series.h>

#include <rerun.hpp>

#include <algorithm>
#include <array>

namespace superdex::tuner {

namespace {

using recorder::ComponentField;

constexpr std::string_view kTunerArchetype = "superdex.archetypes.TunerRun";
constexpr std::string_view kLoadArchetype = "superdex.archetypes.TunerLoad";

ComponentField TunerField(std::string_view component) {
  return {std::string{kTunerArchetype}, std::string{component}};
}

ComponentField LoadField(std::string_view component) {
  return {std::string{kLoadArchetype}, std::string{component}};
}

std::string ChannelPath(std::string_view channel) {
  return std::string(kRecordingChannelsPath) + "/" + rerun::escape_entity_path_part(channel);
}

std::string LoadPath(std::string_view load) {
  return std::string(kRecordingLoadsPath) + "/" + rerun::escape_entity_path_part(load);
}

std::array<real, 3> Xyz(Real3 const& v) {
  return {v[0], v[1], v[2]};
}

std::array<real, 4> Xyzw(mochi::Quaternion const& q) {
  mochi::Real4 const v = q.ToReal4();
  return {v[0], v[1], v[2], v[3]};
}

void AbortEpisode(recorder::Episode& episode, std::string_view reason) {
  mochi::ErrorLog error;
  episode.Abort(reason, error);
}

} // namespace

struct SimRecording::Impl {
  std::filesystem::path path;
  std::unique_ptr<recorder::Episode> episode;
  std::unique_ptr<recorder::PhysicsRecorder> physics;
  std::vector<recorder::PlotSeries*> channels;
  std::vector<recorder::ScalarSeries*> loads;
  uint64_t nextFrame = 0;
  // The first failure; later writes are skipped and Finish reports it.
  mochi::Error error;
};

SimRecording::SimRecording() = default;

SimRecording::~SimRecording() {
  Abort();
}

bool SimRecording::IsSupported() {
  return true;
}

bool SimRecording::Start(
    std::filesystem::path const& path,
    mochi::Scene& scene,
    TunerSession const& session,
    TunerSpec const& resolvedSpec,
    std::string& outError) {
  Abort();
  auto impl = std::make_unique<Impl>();
  impl->path = path;
  mochi::Error& error = impl->error;
  recorder::Recorder const episodes{{.outputDirectory = path.parent_path()}};
  impl->episode = episodes.BeginEpisode({}, error);
  if (impl->episode != nullptr) {
    recorder::Episode& episode = *impl->episode;
    impl->physics = std::make_unique<recorder::PhysicsRecorder>(
        episode,
        scene,
        recorder::PhysicsRecorderConfig{
            .recordContactForces = resolvedSpec.run.record.contactForces},
        error);

    std::vector<std::string> channelNames;
    for (DynamicString const& name : session.GetChannelNames()) {
      channelNames.emplace_back(std::string_view(name));
      impl->channels.push_back(episode.CreatePlotSeries(ChannelPath(channelNames.back()), error));
    }
    std::vector<std::string> loadNames;
    for (LoadSpec const& load : session.GetLoads()) {
      loadNames.emplace_back(std::string_view(load.name));
      impl->loads.push_back(episode.CreateScalarSeries(
          LoadPath(loadNames.back()),
          {LoadField(kRecordingLoadPoint),
           LoadField(kRecordingLoadVector),
           LoadField(kRecordingLoadTargetTranslation),
           LoadField(kRecordingLoadTargetRotation)},
          error));
    }
    episode.WriteConstantText(
        kRecordingTunerPath,
        TunerField(kRecordingSpecJson),
        SaveTunerSpecToJson(resolvedSpec),
        error);
    episode.WriteConstantTexts(
        kRecordingTunerPath, TunerField(kRecordingChannelNames), channelNames, error);
    episode.WriteConstantTexts(
        kRecordingTunerPath, TunerField(kRecordingLoadNames), loadNames, error);
  }
  if (!error.IsOK()) {
    outError = std::string("Cannot record '") + path.string() + "': " + error.GetDescription();
    impl->physics.reset();
    if (impl->episode != nullptr) {
      AbortEpisode(*impl->episode, "Setup failed");
    }
    return false;
  }
  _impl = std::move(impl);
  return true;
}

void SimRecording::Record(Sample const& sample) {
  if (_impl == nullptr || _impl->physics == nullptr || !_impl->error.IsOK()) {
    return;
  }
  mochi::Error& error = _impl->error;
  recorder::SampleTime const time{.frame = _impl->nextFrame++, .simulationTimeSec = sample.time};
  _impl->physics->Record(time, error);
  size_t const numChannels = std::min(sample.values.size(), _impl->channels.size());
  for (size_t i = 0; i < numChannels; ++i) {
    _impl->channels[i]->Append(time, static_cast<double>(sample.values[i]), error);
  }
  size_t const numLoads = std::min(sample.loads.size(), _impl->loads.size());
  for (size_t i = 0; i < numLoads; ++i) {
    AppliedLoad const& load = sample.loads[i];
    auto const point = Xyz(load.pointWorld);
    auto const vector = Xyz(load.vectorWorld);
    auto const targetTranslation = Xyz(load.targetPose.GetTranslation());
    auto const targetRotation = Xyzw(load.targetPose.GetRotation());
    std::array<Span<real const>, 4> const fields{
        mochi::MakeConstSpan(point),
        mochi::MakeConstSpan(vector),
        mochi::MakeConstSpan(targetTranslation),
        mochi::MakeConstSpan(targetRotation)};
    _impl->loads[i]->Append(time, fields, error);
  }
}

void SimRecording::StopSampling() {
  if (_impl != nullptr) {
    _impl->physics.reset();
  }
}

bool SimRecording::Finish(RunSummary const& summary, std::string& outError) {
  if (_impl == nullptr) {
    outError = "No recording is active.";
    return false;
  }
  std::unique_ptr<Impl> const impl = std::move(_impl);
  impl->physics.reset();
  mochi::Error& error = impl->error;
  impl->episode->WriteConstantText(
      kRecordingTunerPath,
      TunerField(kRecordingSummaryJson),
      SReflect::ToJsonString(summary, true),
      error);
  if (!error.IsOK()) {
    outError =
        std::string("Cannot record '") + impl->path.string() + "': " + error.GetDescription();
    AbortEpisode(*impl->episode, outError);
    return false;
  }
  impl->episode->Commit(error);
  // The recorder names files after their episode ID; tuner runs keep a fixed name.
  std::error_code renameError;
  if (error.IsOK()) {
    std::filesystem::rename(impl->episode->GetOutputPath(), impl->path, renameError);
    if (renameError) {
      std::error_code ignored;
      std::filesystem::remove(impl->episode->GetOutputPath(), ignored);
    }
  }
  if (!error.IsOK() || renameError) {
    outError = std::string("Cannot write '") + impl->path.string() +
        "': " + (renameError ? renameError.message() : error.GetDescription());
    return false;
  }
  return true;
}

void SimRecording::Abort() {
  if (_impl == nullptr) {
    return;
  }
  std::unique_ptr<Impl> const impl = std::move(_impl);
  impl->physics.reset();
  AbortEpisode(*impl->episode, "Recording discarded");
}

bool SimRecording::IsActive() const {
  return _impl != nullptr;
}

} // namespace superdex::tuner
