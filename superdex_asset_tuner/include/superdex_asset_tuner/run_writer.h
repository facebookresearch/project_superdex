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

#include "superdex_asset_tuner/tuner_session.h"
#include "superdex_asset_tuner/tuner_spec.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace superdex::tuner {

struct ChannelSummary {
  DynamicString name;
  real min = 0;
  real max = 0;
  real final = 0;

  MOCHI_STRUCT_BEGIN(superdex::tuner::ChannelSummary)
  MOCHI_FIELD(name)
  MOCHI_FIELD(min)
  MOCHI_FIELD(max)
  MOCHI_FIELD(final)
  MOCHI_STRUCT_END()
};

struct RunSummary {
  // "ok", "stopped" (a stop condition ended the run early), "diverged" or "error".
  DynamicString status = "ok";
  DynamicString message;
  // Name of the stop condition that ended the run; empty unless status is "stopped".
  DynamicString stopCondition;
  int steps = 0;
  double simTime = 0.0;
  double wallTimeSec = 0.0;
  DynamicString runDir;
  bool simRecording = false;
  DynamicArray<ChannelSummary> channels;

  MOCHI_STRUCT_BEGIN(superdex::tuner::RunSummary)
  MOCHI_FIELD(status)
  MOCHI_FIELD(message)
  MOCHI_FIELD(stopCondition)
  MOCHI_FIELD(steps)
  MOCHI_FIELD(simTime) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD(wallTimeSec) MOCHI_ATTRIBUTE(Units("s"));
  MOCHI_FIELD(runDir)
  MOCHI_FIELD(simRecording)
  MOCHI_FIELD(channels)
  MOCHI_STRUCT_END()
};

// Writes one run directory:
//   tuner.resolved.json  spec after overrides, plus the resolved prefab path
//   signals.csv          time + one column per channel
//   summary.json         RunSummary with per-channel min/max/final
//   sim.rrd              SuperDex episode of the run (written by SimRecording, see
//   GetSimRecordingPath)
class RunWriter {
 public:
  // Creates `<outDir>/<UTC timestamp>_<tunerName>[_<tag>]` and writes tuner.resolved.json.
  bool Open(
      std::filesystem::path const& outDir,
      std::string_view tunerName,
      std::string_view tag,
      TunerSpec const& resolvedSpec,
      std::filesystem::path const& prefabPath,
      std::string& outError);

  void WriteHeader(Span<DynamicString const> channelNames);
  void WriteSample(Sample const& sample);

  // Sets runDir in `inOutSummary`, then writes summary.json.
  bool Close(RunSummary& inOutSummary, std::string& outError);

  [[nodiscard]] bool IsOpen() const {
    return _signals.is_open();
  }
  [[nodiscard]] std::filesystem::path const& GetRunDir() const {
    return _runDir;
  }
  [[nodiscard]] std::filesystem::path GetSimRecordingPath() const {
    return _runDir / "sim.rrd";
  }

 private:
  std::filesystem::path _runDir;
  std::ofstream _signals;
};

// Updates per-channel min/max/final with one sample. `first` resets min/max.
void UpdateChannelSummaries(
    Span<real const> values,
    bool first,
    Span<ChannelSummary> inOutChannels);

} // namespace superdex::tuner
