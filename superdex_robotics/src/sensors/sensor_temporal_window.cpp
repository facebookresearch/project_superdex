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

#include <superdex_robotics/sensors/sensor_temporal_window.h>

#include <superdex_robotics/sensors/sensor_recording.h>
#include <superdex_robotics/utils/file_utils.h>

#include <algorithm>
#include <array>

using namespace mochi;
using namespace superdex::robotics;

SensorTemporalWindow::SensorTemporalWindow(Actor* actor, std::string_view paramArgs, Error& error)
    : SensorBase(actor, error) {
  MOCHI_ERROR_RETURN(error);
  Params const params = paramArgs.empty()
      ? Params{}
      : LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  MOCHI_ERROR_RETURN(error);
  Initialize(params, error);
}

SensorTemporalWindow::SensorTemporalWindow(Actor* actor, Params const& params, Error& error)
    : SensorBase(actor, error) {
  MOCHI_ERROR_RETURN(error);
  Initialize(params, error);
}

SensorTemporalWindow::SensorTemporalWindow(Params const& params, Error& error)
    : SensorTemporalWindow(nullptr, params, error) {}

SensorTemporalWindow::~SensorTemporalWindow() = default;

void SensorTemporalWindow::Initialize(Params const& params, Error& error) {
  MOCHI_ERROR_IF(
      params.historyLength < 1, error, "SensorTemporalWindow historyLength must be positive");
  MOCHI_ERROR_RETURN(error);
  _params = params;
}

SensorFormat SensorTemporalWindow::InitSensor(SensorScope const& /*scope*/, Error& error) {
  Span<SensorBase const* const> const inputs = Inputs();
  MOCHI_ERROR_IF_NOT(inputs.size() == 1, error, "SensorTemporalWindow requires one input");
  MOCHI_ERROR_RETURN(error, {});
  SensorFormat const& source = inputs[0]->GetFormat();
  MOCHI_ERROR_IF_NOT(
      source.dtype == SensorDType::Float32 && !source.shape.empty(),
      error,
      "SensorTemporalWindow requires one fixed Float32 input");
  MOCHI_ERROR_RETURN(error, {});

  _elementsPerSample = static_cast<std::size_t>(inputs[0]->Latest().elementCount);
  auto const history = static_cast<std::size_t>(_params.historyLength);
  MOCHI_ERROR_IF(
      _elementsPerSample > std::numeric_limits<std::size_t>::max() / history,
      error,
      "SensorTemporalWindow output shape overflows");
  MOCHI_ERROR_RETURN(error, {});
  std::size_t const elements = history * _elementsPerSample;

  _ring.resize(elements);
  _ringCaptureSteps.resize(history);
  _ringCaptureTimes.resize(history);
  _ringPublishedSteps.resize(history);
  _ringPublishedTimes.resize(history);
  _output.resize(elements);
  _outputCaptureSteps.resize(history);
  _outputCaptureTimes.resize(history);
  _outputPublishedSteps.resize(history);
  _outputPublishedTimes.resize(history);

  SensorFormat format = source;
  DynamicArray<int32_t> outputShape;
  outputShape.reserve(source.shape.size() + 1);
  outputShape.push_back(_params.historyLength);
  for (int32_t const dimension : source.shape) {
    outputShape.push_back(dimension);
  }
  format.shape = std::move(outputShape);
  DynamicArray<SensorAxis> outputAxes;
  outputAxes.reserve(source.axes.size() + 1);
  outputAxes.push_back({.name = "history", .labels = {}, .units = {}});
  for (SensorAxis const& axis : source.axes) {
    outputAxes.push_back(axis);
  }
  format.axes = std::move(outputAxes);
  ResetSensor();
  return format;
}

void SensorTemporalWindow::MaterializeOutput(SensorSample const& source, Span<float const> values) {
  auto const history = static_cast<std::size_t>(_params.historyLength);
  std::size_t const nextValidCount = std::min(history, _validCount + 1);
  std::size_t const firstOutput = history - nextValidCount;
  std::ranges::fill(_output.begin(), _output.begin() + firstOutput * _elementsPerSample, 0.0F);
  std::ranges::fill(_outputCaptureSteps.begin(), _outputCaptureSteps.begin() + firstOutput, -1);
  std::ranges::fill(_outputCaptureTimes.begin(), _outputCaptureTimes.begin() + firstOutput, 0.0);
  std::ranges::fill(_outputPublishedSteps.begin(), _outputPublishedSteps.begin() + firstOutput, -1);
  std::ranges::fill(
      _outputPublishedTimes.begin(), _outputPublishedTimes.begin() + firstOutput, 0.0);

  std::size_t const previousRows = nextValidCount - 1;
  std::size_t const oldest = (_nextWrite + history + 1 - nextValidCount) % history;
  for (std::size_t i = 0; i < previousRows; ++i) {
    std::size_t const ringRow = (oldest + i) % history;
    std::size_t const outputRow = firstOutput + i;
    std::copy_n(
        _ring.data() + ringRow * _elementsPerSample,
        _elementsPerSample,
        _output.data() + outputRow * _elementsPerSample);
    _outputCaptureSteps[outputRow] = _ringCaptureSteps[ringRow];
    _outputCaptureTimes[outputRow] = _ringCaptureTimes[ringRow];
    _outputPublishedSteps[outputRow] = _ringPublishedSteps[ringRow];
    _outputPublishedTimes[outputRow] = _ringPublishedTimes[ringRow];
  }
  std::size_t const newestOutput = history - 1;
  std::ranges::copy(values, _output.begin() + newestOutput * _elementsPerSample);
  _outputCaptureSteps[newestOutput] = source.captureStep;
  _outputCaptureTimes[newestOutput] = source.captureTime;
  _outputPublishedSteps[newestOutput] = source.publishedStep;
  _outputPublishedTimes[newestOutput] = source.publishedTime;
}

SensorStepResult SensorTemporalWindow::Step(StepTime const& now, Error& error) {
  Span<SensorBase const* const> const inputs = Inputs();
  MOCHI_ERROR_IF_NOT(inputs.size() == 1, error, "SensorTemporalWindow requires one input");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  SensorSample const source = inputs[0]->Latest();
  if (source.publishedStep < 0 || source.publishedStep == _lastInputPublishedStep) {
    return SensorStepResult::NoUpdate;
  }
  Span<float const> const values = source.Floats();
  MOCHI_ERROR_IF_NOT(
      values.size() == _elementsPerSample,
      error,
      "SensorTemporalWindow input size changed after graph build");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  MaterializeOutput(source, values);
  PublishSample(now, Span<float const>{_output}, error, &source);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);

  std::ranges::copy(values, _ring.begin() + _nextWrite * _elementsPerSample);
  _ringCaptureSteps[_nextWrite] = source.captureStep;
  _ringCaptureTimes[_nextWrite] = source.captureTime;
  _ringPublishedSteps[_nextWrite] = source.publishedStep;
  _ringPublishedTimes[_nextWrite] = source.publishedTime;
  auto const history = static_cast<std::size_t>(_params.historyLength);
  _nextWrite = (_nextWrite + 1) % history;
  _validCount = std::min(history, _validCount + 1);
  _lastInputPublishedStep = source.publishedStep;
  return SensorStepResult::Published;
}

void SensorTemporalWindow::ConfigureRecording(SensorRecordingSink& sink, Error& error) const {
  SensorBase::ConfigureRecording(sink, error);
  auto const history = static_cast<std::size_t>(_params.historyLength);
  std::array<double, 1> const historyLength{static_cast<double>(history)};
  sink.WriteConstantScalars({}, "temporal_history_length", historyLength, error);
  sink.DeclareInt64SampleField("window_capture_steps", history, error);
  sink.DeclareInt64SampleField("window_published_steps", history, error);
  sink.DeclareFloat64SampleField("window_capture_times_sec", history, error);
  sink.DeclareFloat64SampleField("window_published_times_sec", history, error);
}

void SensorTemporalWindow::Record(SensorRecordingSink& sink, Error& error) const {
  std::array<SensorRecordingInt64Values, 2> const int64Fields{{
      {.name = "window_capture_steps", .values = _outputCaptureSteps},
      {.name = "window_published_steps", .values = _outputPublishedSteps},
  }};
  std::array<SensorRecordingFloat64Values, 2> const float64Fields{{
      {.name = "window_capture_times_sec", .values = _outputCaptureTimes},
      {.name = "window_published_times_sec", .values = _outputPublishedTimes},
  }};
  sink.WriteSampleWithFields(Latest(), int64Fields, float64Fields, error);
}

void SensorTemporalWindow::ResetSensor() {
  std::ranges::fill(_ring, 0.0F);
  std::ranges::fill(_ringCaptureSteps, int64_t{-1});
  std::ranges::fill(_ringCaptureTimes, 0.0);
  std::ranges::fill(_ringPublishedSteps, int64_t{-1});
  std::ranges::fill(_ringPublishedTimes, 0.0);
  std::ranges::fill(_output, 0.0F);
  std::ranges::fill(_outputCaptureSteps, int64_t{-1});
  std::ranges::fill(_outputCaptureTimes, 0.0);
  std::ranges::fill(_outputPublishedSteps, int64_t{-1});
  std::ranges::fill(_outputPublishedTimes, 0.0);
  _nextWrite = 0;
  _validCount = 0;
  _lastInputPublishedStep = -1;
}
