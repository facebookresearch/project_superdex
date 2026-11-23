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

#include <superdex_robotics/sensors/sensor_concat.h>
#include <superdex_robotics/sensors/sensor_recording.h>
#include <superdex_robotics/utils/file_utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

using namespace mochi;
using namespace superdex::robotics;

SensorConcat::SensorConcat(Actor* actor, std::string_view paramArgs, Error& error)
    : SensorBase(actor, error) {
  MOCHI_ERROR_RETURN(error);
  _params = paramArgs.empty()
      ? Params{}
      : LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
}

SensorConcat::SensorConcat(Params const& params, Error& error)
    : SensorBase(error), _params(params) {}

SensorFormat SensorConcat::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto const inputs = Inputs();
  MOCHI_ERROR_IF(
      inputs.empty() || inputs.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      error,
      "SensorConcat requires a nonempty, representable input count");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(
      _params.axisName.empty() ||
          (!_params.inputLabels.empty() && _params.inputLabels.size() != inputs.size()),
      error,
      "SensorConcat requires an axis name and one label per input when labels are supplied");
  MOCHI_ERROR_RETURN(error, {});
  SensorFormat const& source = inputs[0]->GetFormat();
  for (SensorBase const* input : inputs) {
    MOCHI_ERROR_IF_NOT(
        source.dtype == SensorDType::Float32 && input->GetFormat() == source,
        error,
        "SensorConcat requires identically formatted Float32 inputs");
    MOCHI_ERROR_RETURN(error, {});
  }
  SensorFormat format{.kind = source.kind, .dtype = source.dtype};
  format.shape.push_back(static_cast<int32_t>(inputs.size()));
  for (auto dimension : source.shape) {
    format.shape.push_back(dimension);
  }
  format.axes.push_back({.name = _params.axisName, .labels = _params.inputLabels});
  for (auto const& axis : source.axes) {
    format.axes.push_back(axis);
  }
  _inputPublishedSteps.resize(inputs.size());
  _inputCaptureSteps.resize(inputs.size());
  _inputCaptureTimes.resize(inputs.size());
  _values.clear();
  ResetSensor();
  return format;
}

void SensorConcat::ResetSensor() {
  std::ranges::fill(_inputPublishedSteps, -1);
  std::ranges::fill(_inputCaptureSteps, -1);
  std::ranges::fill(_inputCaptureTimes, 0.0);
}

SensorStepResult SensorConcat::Step(StepTime const& now, Error& error) {
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  auto const inputs = Inputs();
  bool changed = false;
  SensorSample newest;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto const sample = inputs[i]->Latest();
    if (sample.captureStep < 0) {
      return SensorStepResult::NoUpdate;
    }
    changed |= sample.publishedStep != _inputPublishedSteps[i];
    if (newest.captureStep < 0 || sample.captureTime > newest.captureTime) {
      newest = sample;
    }
  }
  if (!changed) {
    return SensorStepResult::NoUpdate;
  }
  _values.resize(static_cast<size_t>(Latest().elementCount));
  auto* destination = _values.begin();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto const sample = inputs[i]->Latest();
    destination = std::ranges::copy(sample.Floats(), destination).out;
    _inputPublishedSteps[i] = sample.publishedStep;
    _inputCaptureSteps[i] = sample.captureStep;
    _inputCaptureTimes[i] = sample.captureTime;
  }
  if (!PublishSample(now, _values, error, &newest)) {
    return SensorStepResult::NoUpdate;
  }
  return SensorStepResult::Published;
}

double SensorConcat::GetPublicationFrequencyHzEstimate() const {
  double frequency = 0.0;
  for (SensorBase const* input : Inputs()) {
    double const estimate = input->GetPublicationFrequencyHzEstimate();
    if (!std::isfinite(estimate) || estimate <= 0.0) {
      return 0.0;
    }
    frequency += estimate;
  }
  return frequency;
}

void SensorConcat::ConfigureRecording(SensorRecordingSink& sink, Error& error) const {
  MOCHI_ERROR_RETURN(error);
  sink.DeclareInt64SampleField("input_capture_steps", Inputs().size(), error);
  sink.DeclareFloat64SampleField("input_capture_times", Inputs().size(), error);
}

void SensorConcat::Record(SensorRecordingSink& sink, Error& error) const {
  std::array<SensorRecordingInt64Values, 1> const steps{{
      {.name = "input_capture_steps", .values = _inputCaptureSteps},
  }};
  std::array<SensorRecordingFloat64Values, 1> const times{{
      {.name = "input_capture_times", .values = _inputCaptureTimes},
  }};
  sink.WriteSampleWithFields(Latest(), steps, times, error);
}
