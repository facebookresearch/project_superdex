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

#include <superdex_robotics/sensors/sensor_gaussian_noise.h>

#include <superdex_robotics/domain_randomization/derive.h>
#include <superdex_robotics/domain_randomization/random.h>
#include <superdex_robotics/sensors/sensor_recording.h>
#include <superdex_robotics/utils/file_utils.h>

#include <algorithm>
#include <cmath>
#include <string>

using namespace mochi;
using namespace superdex::robotics;

SensorGaussianNoise::SensorGaussianNoise(Actor* actor, std::string_view paramArgs, Error& error)
    : SensorBase(actor, error) {
  MOCHI_ERROR_RETURN(error);
  auto const params = LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  MOCHI_ERROR_RETURN(error);
  SetParams(params, error);
}
SensorGaussianNoise::SensorGaussianNoise(Actor* actor, Params const& params, Error& error)
    : SensorBase(actor, error) {
  SetParams(params, error);
}
SensorGaussianNoise::SensorGaussianNoise(Params const& params, Error& error)
    : SensorGaussianNoise(nullptr, params, error) {}

void SensorGaussianNoise::SetParams(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (params.std.empty() ||
      (!_values.empty() && params.std.size() != 1 && params.std.size() != _values.size())) {
    MOCHI_ERROR_SET(
        error, "SensorGaussianNoise std must be scalar or match the input element count");
    return;
  }
  for (double const value : params.std) {
    if (!std::isfinite(value) || value < 0.0) {
      MOCHI_ERROR_SET(error, "SensorGaussianNoise std must be finite and nonnegative");
      return;
    }
  }
  _params = params;
}

SensorFormat SensorGaussianNoise::InitSensor(SensorScope const& /*scope*/, Error& error) {
  if (Inputs().size() != 1 || Inputs()[0]->GetFormat().dtype != SensorDType::Float32) {
    MOCHI_ERROR_SET(error, "SensorGaussianNoise requires one Float32 input");
    return {};
  }
  SensorFormat const& format = Inputs()[0]->GetFormat();
  auto const count = static_cast<std::size_t>(Inputs()[0]->Latest().elementCount);
  if (_params.std.size() != 1 && _params.std.size() != count) {
    MOCHI_ERROR_SET(
        error, "SensorGaussianNoise std must be scalar or match the input element count");
    return {};
  }
  _values.resize(count);
  ResetSensor();
  return format;
}

SensorStepResult SensorGaussianNoise::Step(StepTime const& now, Error& error) {
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  SensorSample const source = Inputs()[0]->Latest();
  if (source.captureStep < 0 || source.captureStep == _lastCaptureStep) {
    return SensorStepResult::NoUpdate;
  }
  Span<float const> const values = source.Floats();
  if (values.size() != _values.size()) {
    MOCHI_ERROR_SET(error, "SensorGaussianNoise input size changed after initialization");
    return SensorStepResult::NoUpdate;
  }
  uint64_t state = Derive(
      EpisodeSeed(_params.random.episodeSeed),
      SensorStreamKey(_params.random, GraphName()),
      static_cast<uint64_t>(source.captureStep));
  for (std::size_t i = 0; i < values.size(); ++i) {
    double const sigma = _params.std[_params.std.size() == 1 ? 0 : i];
    _values[i] = static_cast<float>(values[i] + sigma * Normal(state));
  }
  bool const published = PublishSample(now, _values, error, &source);
  if (published) {
    _lastCaptureStep = source.captureStep;
  }
  return published ? SensorStepResult::Published : SensorStepResult::NoUpdate;
}

void SensorGaussianNoise::ResetSensor() {
  _lastCaptureStep = -1;
}
Span<std::byte const> SensorGaussianNoise::TryGetMetadata(std::string_view key) const {
  return Inputs().size() == 1 ? Inputs()[0]->TryGetMetadata(key) : Span<std::byte const>{};
}
void SensorGaussianNoise::ConfigureRecording(SensorRecordingSink& sink, Error& error) const {
  SensorBase::ConfigureRecording(sink, error);
  MOCHI_ERROR_RETURN(error);
  DynamicArray<double> const standardDeviations(_params.std.begin(), _params.std.end());
  sink.WriteConstantScalars("process", "std", standardDeviations, error);
  sink.WriteConstantText(
      "process", "episode_seed", std::to_string(EpisodeSeed(_params.random.episodeSeed)), error);
  sink.WriteConstantText("process", "key", SensorStreamKey(_params.random, GraphName()), error);
  sink.WriteConstantText(
      "process", "algorithm", "independent_gaussian_splitmix64_box_muller.v1", error);
}
