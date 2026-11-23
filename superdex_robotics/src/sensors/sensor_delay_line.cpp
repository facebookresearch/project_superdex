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

#include <superdex_robotics/domain_randomization/derive.h>
#include <superdex_robotics/domain_randomization/random.h>
#include <superdex_robotics/utils/file_utils.h>

#include <sensors/sensor_timing.h>

#include <algorithm>
#include <cmath>
#include <limits>
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
  if (!std::isfinite(params.delayMs) || params.delayMs < 0.0 || !std::isfinite(params.jitterMs) ||
      params.jitterMs < 0.0 || !std::isfinite(params.bufferMs) || params.bufferMs < 0.0 ||
      !std::isfinite(params.bufferAssumedFrequencyHzEstimate) ||
      params.bufferAssumedFrequencyHzEstimate <= 0.0) {
    MOCHI_ERROR_SET(
        error, "SensorDelayLine requires finite nonnegative times and a positive rate estimate");
    return;
  }
  MOCHI_ERROR_IF(
      params.delayMs + params.jitterMs > params.bufferMs,
      error,
      "SensorDelayLine delayMs + jitterMs must fit in bufferMs; raise bufferMs to cover the "
      "largest delay");
  MOCHI_ERROR_IF(
      !_ring.empty() &&
          (params.bufferMs != _params.bufferMs ||
           params.bufferAssumedFrequencyHzEstimate != _params.bufferAssumedFrequencyHzEstimate),
      error,
      "SensorDelayLine buffer sizing cannot change after initialization");
  MOCHI_ERROR_RETURN(error);
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
  std::size_t const capacity = static_cast<std::size_t>(sampleCountEstimate) + 1;
  MOCHI_ERROR_IF(
      _elementsPerSample > std::numeric_limits<std::size_t>::max() / capacity,
      error,
      "SensorDelayLine storage size overflows");
  MOCHI_ERROR_RETURN(error, {});
  _ring.resize(capacity);
  _ringValues.resize(capacity * _elementsPerSample);
  ResetSensor();
  return source;
}

void SensorDelayLine::Enqueue(SensorSample const& source, Error& error) {
  MOCHI_ERROR_IF(
      _size >= _ring.size(),
      error,
      "SensorDelayLine queue exceeded its configured capacity; increase bufferMs or "
      "bufferAssumedFrequencyHzEstimate to cover the delay and actual input publication rate");
  MOCHI_ERROR_RETURN(error);
  Span<float const> const values = source.Floats();
  MOCHI_ERROR_IF_NOT(
      values.size() == _elementsPerSample,
      error,
      "SensorDelayLine input size changed after graph build");
  MOCHI_ERROR_RETURN(error);
  double delayMs = _params.delayMs;
  if (_params.jitterMs > 0.0) {
    uint64_t state = Derive(
        EpisodeSeed(_params.random.episodeSeed),
        SensorStreamKey(_params.random, GraphName()),
        static_cast<uint64_t>(source.captureStep));
    delayMs += Uniform(state, 0.0, _params.jitterMs);
  }
  std::size_t const slot = (_head + _size) % _ring.size();
  std::copy(values.begin(), values.end(), _ringValues.begin() + slot * _elementsPerSample);
  _ring[slot] = {
      .captureStep = source.captureStep,
      .captureTime = source.captureTime,
      .dueTime = source.publishedTime + delayMs / 1000.0,
  };
  ++_size;
}

bool SensorDelayLine::PublishDue(StepTime const& now, Error& error) {
  // A sample due within the time tolerance after the step's time publishes at that step.
  if (_size == 0 || _ring[_head].dueTime > now.t + detail::kSensorTimeToleranceSec) {
    return false;
  }
  SensorSample const sourceIdentity = {
      .captureStep = _ring[_head].captureStep,
      .captureTime = _ring[_head].captureTime,
  };
  bool const published = PublishSample(
      now,
      Span<float const>{_ringValues.data() + _head * _elementsPerSample, _elementsPerSample},
      error,
      &sourceIdentity);
  if (published) {
    _head = (_head + 1) % _ring.size();
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
  // Due samples free their slots before the input's new sample takes one.
  while (PublishDue(now, error)) {
    published = true;
  }
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  SensorSample const source = inputs[0]->Latest();
  if (source.publishedStep >= 0 && source.publishedStep != _lastInputPublishedStep) {
    if (_lastInputPublishedStep < 0) {
      // The first sample since the reset publishes at once.
      published = PublishSample(now, source.Floats(), error, &source) || published;
    } else {
      Enqueue(source, error);
    }
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
  _head = 0;
  _size = 0;
  _lastInputPublishedStep = -1;
}
