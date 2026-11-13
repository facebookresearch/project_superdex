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

// SimRecording for builds without superdex_recorder: runs write only their CSV and JSON files.

#include "superdex_asset_tuner/sim_recording.h"

namespace superdex::tuner {

namespace {

constexpr std::string_view kUnsupported =
    "Built without superdex_recorder, so sim.rrd recordings are not supported.";

} // namespace

struct SimRecording::Impl {};

SimRecording::SimRecording() = default;

SimRecording::~SimRecording() = default;

bool SimRecording::IsSupported() {
  return false;
}

bool SimRecording::Start(
    std::filesystem::path const& /*path*/,
    mochi::Scene& /*scene*/,
    TunerSession const& /*session*/,
    TunerSpec const& /*resolvedSpec*/,
    std::string& outError) {
  outError = kUnsupported;
  return false;
}

void SimRecording::Record(Sample const& /*sample*/) {}

void SimRecording::StopSampling() {}

bool SimRecording::Finish(RunSummary const& /*summary*/, std::string& outError) {
  outError = kUnsupported;
  return false;
}

void SimRecording::Abort() {}

bool SimRecording::IsActive() const {
  return false;
}

} // namespace superdex::tuner
