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

#include <superdex_robotics/sensors/sensor_uniform_noise.h>

#include <superdex_robotics/domain_randomization/derive.h>
#include <superdex_robotics/domain_randomization/random.h>
#include <superdex_robotics/utils/file_utils.h>

#include <cmath>

using namespace mochi;
using namespace superdex::robotics;

SensorUniformNoise::SensorUniformNoise(Actor* actor, std::string_view paramArgs, Error& error)
    : SensorBase(actor, error) {
  MOCHI_ERROR_RETURN(error);
  Params const params = LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  SetParams(params, error);
}

SensorUniformNoise::SensorUniformNoise(Params const& params, Error& error) : SensorBase(error) {
  SetParams(params, error);
}

void SensorUniformNoise::SetParams(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      std::isfinite(params.low) && std::isfinite(params.high) && params.low <= params.high,
      error,
      "SensorUniformNoise needs finite bounds with low <= high");
  MOCHI_ERROR_RETURN(error);
  _params = params;
}

SensorFormat SensorUniformNoise::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_IF_NOT(
      Inputs().size() == 1 && Inputs()[0]->GetFormat().dtype == SensorDType::Float32,
      error,
      "SensorUniformNoise requires one Float32 input");
  MOCHI_ERROR_RETURN(error, {});
  _values.resize(static_cast<std::size_t>(Inputs()[0]->Latest().elementCount));
  ResetSensor();
  return Inputs()[0]->GetFormat();
}

SensorStepResult SensorUniformNoise::Step(StepTime const& now, Error& error) {
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  SensorSample const source = Inputs()[0]->Latest();
  if (source.captureStep < 0 || source.captureStep == _lastCaptureStep) {
    return SensorStepResult::NoUpdate;
  }
  Span<float const> const values = source.Floats();
  MOCHI_ASSERT_VERBOSE(values.size() == _values.size());
  uint64_t state = Derive(
      EpisodeSeed(_params.random.episodeSeed),
      SensorStreamKey(_params.random, GraphName()),
      static_cast<uint64_t>(source.captureStep));
  for (std::size_t i = 0; i < values.size(); ++i) {
    _values[i] = values[i] + static_cast<float>(Uniform(state, _params.low, _params.high));
  }
  if (!PublishSample(now, MakeConstSpan(_values), error, &source)) {
    return SensorStepResult::NoUpdate;
  }
  _lastCaptureStep = source.captureStep;
  return SensorStepResult::Published;
}

void SensorUniformNoise::ResetSensor() {
  _lastCaptureStep = -1;
}

Span<std::byte const> SensorUniformNoise::TryGetMetadata(std::string_view key) const {
  return Inputs().size() == 1 ? Inputs()[0]->TryGetMetadata(key) : Span<std::byte const>{};
}
