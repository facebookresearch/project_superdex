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

#include <superdex_robotics/sensors/sensor_delay_line.h>

#include <superdex_robotics/utils/file_utils.h>

#include <algorithm>
#include <cmath>
#include <string_view>

using namespace mochi;
using namespace superdex::robotics;

SensorDelayLine::SensorDelayLine(Actor* actor, std::string_view paramArgs, Error& error)
    : SensorBase(actor, error) {
  MOCHI_ERROR_RETURN(error);
  Params const params = paramArgs.empty()
      ? Params{}
      : LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  MOCHI_ERROR_RETURN(error);
  Initialize(params, error);
}

SensorDelayLine::SensorDelayLine(Actor* actor, Params const& params, Error& error)
    : SensorBase(actor, error) {
  MOCHI_ERROR_RETURN(error);
  Initialize(params, error);
}

SensorDelayLine::SensorDelayLine(Params const& params, Error& error)
    : SensorDelayLine(nullptr, params, error) {}

SensorDelayLine::~SensorDelayLine() = default;

void SensorDelayLine::Initialize(Params const& params, Error& error) {
  SetParams(params, error);
}

void SensorDelayLine::SetParams(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (!std::isfinite(params.delayMs) || params.delayMs < 0.0 || !std::isfinite(params.bufferMs) ||
      params.bufferMs < 0.0 || !std::isfinite(params.bufferAssumedFrequencyHzEstimate) ||
      params.bufferAssumedFrequencyHzEstimate <= 0.0) {
    MOCHI_ERROR_SET(
        error, "SensorDelayLine requires finite nonnegative times and a positive rate estimate");
    return;
  }
  if (_capacity != 0 &&
      (params.bufferMs != _params.bufferMs ||
       params.bufferAssumedFrequencyHzEstimate != _params.bufferAssumedFrequencyHzEstimate)) {
    MOCHI_ERROR_SET(error, "SensorDelayLine buffer sizing cannot change after initialization");
    return;
  }
  _params = params;
}

SensorFormat SensorDelayLine::InitSensor(SensorScope const& /*scope*/, Error& error) {
  Span<SensorBase const* const> const inputs = Inputs();
  MOCHI_ERROR_IF_NOT(inputs.size() == 1, error, "SensorDelayLine requires one input");
  MOCHI_ERROR_RETURN(error, {});
  SensorFormat const& source = inputs[0]->GetFormat();
  MOCHI_ERROR_IF_NOT(
      source.dtype == SensorDType::Float32 && !source.shape.empty(),
      error,
      "SensorDelayLine requires one fixed Float32 input");
  MOCHI_ERROR_RETURN(error, {});

  _elementsPerSample = static_cast<std::size_t>(inputs[0]->Latest().elementCount);
  double frequencyHzEstimate = inputs[0]->GetPublicationFrequencyHzEstimate();
  if (!std::isfinite(frequencyHzEstimate) || frequencyHzEstimate <= 0.0) {
    frequencyHzEstimate = _params.bufferAssumedFrequencyHzEstimate;
  }
  double const sampleCountEstimate = std::ceil((_params.bufferMs / 1000.0) * frequencyHzEstimate);
  MOCHI_ERROR_IF(
      !std::isfinite(sampleCountEstimate) ||
          sampleCountEstimate >= static_cast<double>(std::numeric_limits<std::size_t>::max()),
      error,
      "SensorDelayLine buffer capacity estimate overflows");
  MOCHI_ERROR_RETURN(error, {});
  _capacity = static_cast<std::size_t>(sampleCountEstimate) + 1;
  MOCHI_ERROR_IF(
      _elementsPerSample > std::numeric_limits<std::size_t>::max() / _capacity,
      error,
      "SensorDelayLine storage size overflows");
  MOCHI_ERROR_RETURN(error, {});
  std::size_t const elements = _capacity * _elementsPerSample;
  _floatRing.resize(elements);
  _captureSteps.resize(_capacity);
  _captureTimes.resize(_capacity);
  _dueTimes.resize(_capacity);
  ResetSensor();
  return source;
}

void SensorDelayLine::Enqueue(SensorSample const& source, Error& error) {
  MOCHI_ERROR_IF(
      _size >= _capacity,
      error,
      "SensorDelayLine queue exceeded its configured capacity; increase bufferMs or "
      "bufferAssumedFrequencyHzEstimate to cover the delay and actual input publication rate");
  MOCHI_ERROR_RETURN(error);
  std::size_t const slot = (_head + _size) % _capacity;
  Span<float const> const values = source.Floats();
  MOCHI_ERROR_IF_NOT(
      values.size() == _elementsPerSample,
      error,
      "SensorDelayLine input size changed after graph build");
  MOCHI_ERROR_RETURN(error);
  std::ranges::copy(values, _floatRing.begin() + slot * _elementsPerSample);
  _captureSteps[slot] = source.captureStep;
  _captureTimes[slot] = source.captureTime;
  _dueTimes[slot] = source.publishedTime + static_cast<double>(_params.delayMs) / 1000.0;
  ++_size;
}

bool SensorDelayLine::PublishDue(StepTime const& now, Error& error) {
  double const inclusiveBoundary = std::nextafter(now.t, std::numeric_limits<double>::infinity());
  if (_size == 0 || _dueTimes[_head] > inclusiveBoundary) {
    return false;
  }
  SensorSample const sourceIdentity = {
      .captureStep = _captureSteps[_head],
      .captureTime = _captureTimes[_head],
  };
  std::size_t const offset = _head * _elementsPerSample;
  bool const published = PublishSample(
      now,
      Span<float const>{_floatRing.data() + offset, _elementsPerSample},
      error,
      &sourceIdentity);
  if (published) {
    _head = (_head + 1) % _capacity;
    --_size;
  }
  return published;
}

Span<std::byte const> SensorDelayLine::TryGetMetadata(std::string_view key) const {
  Span<SensorBase const* const> const inputs = Inputs();
  return inputs.size() == 1 ? inputs[0]->TryGetMetadata(key) : Span<std::byte const>{};
}

SensorStepResult SensorDelayLine::Step(StepTime const& now, Error& error) {
  Span<SensorBase const* const> const inputs = Inputs();
  MOCHI_ERROR_IF_NOT(inputs.size() == 1, error, "SensorDelayLine requires one input");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  bool published = false;
  while (PublishDue(now, error)) {
    published = true;
  }
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  SensorSample const source = inputs[0]->Latest();
  if (source.publishedStep >= 0 && source.publishedStep != _lastInputPublishedStep) {
    Enqueue(source, error);
    MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
    _lastInputPublishedStep = source.publishedStep;
  }
  while (PublishDue(now, error)) {
    published = true;
  }
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  return published ? SensorStepResult::Published : SensorStepResult::NoUpdate;
}

void SensorDelayLine::ResetSensor() {
  std::ranges::fill(_floatRing, 0.0F);
  std::ranges::fill(_captureSteps, int64_t{-1});
  std::ranges::fill(_captureTimes, 0.0);
  std::ranges::fill(_dueTimes, 0.0);
  _head = 0;
  _size = 0;
  _lastInputPublishedStep = -1;
}
