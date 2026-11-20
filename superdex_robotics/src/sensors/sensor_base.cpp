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

#include <superdex_robotics/sensors/sensor_base.h>

#include <superdex_robotics/sensors/sensor_recording.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

using namespace mochi;
using namespace superdex::robotics;

SensorBase::SensorBase(Actor* actor, Error& /*error*/) : ComponentBase(actor) {}
SensorBase::SensorBase(Error& /*error*/) : ComponentBase(nullptr) {}
SensorBase::~SensorBase() {
  MOCHI_ASSERT(_graphBorrowCount == 0, "Clear every SensorGraph before destroying its sensors");
}

Span<float const> SensorSample::Floats() const {
  MOCHI_ASSERT(data != nullptr || elementCount == 0);
  return {static_cast<float const*>(data), static_cast<std::size_t>(elementCount)};
}

DynamicArray<real> SensorBase::CopyLatestValues() const {
  DynamicArray<real> values;
  values.resize(_output.size());
  std::ranges::transform(
      _output, values.begin(), [](float value) { return static_cast<real>(value); });
  return values;
}

double SensorBase::GetPublicationFrequencyHzEstimate() const {
  double frequency = 0.0;
  for (SensorBase const* input : Inputs()) {
    double const estimate = input->GetPublicationFrequencyHzEstimate();
    if (!std::isfinite(estimate) || estimate <= 0.0) {
      return 0.0;
    }
    frequency = std::max(frequency, estimate);
  }
  return frequency;
}

namespace {

std::size_t ValidateFormatAndGetElementCount(SensorFormat const& format, Error& error) {
  if (format.kind.empty()) {
    MOCHI_ERROR_SET(error, "Sensor output kind must not be empty");
    return 0;
  }
  if (format.shape.empty()) {
    MOCHI_ERROR_SET(error, "Sensor output shape must not be empty");
    return 0;
  }
  if (format.axes.size() != format.shape.size()) {
    MOCHI_ERROR_SET(error, "Sensor output axes must match the shape rank");
    return 0;
  }

  std::size_t elementCount = 1;
  bool hasUnits = false;
  for (std::size_t i = 0; i < format.shape.size(); ++i) {
    SensorDimension const dimension = format.shape[i];
    if (dimension <= 0) {
      MOCHI_ERROR_SET(error, "Sensor output dimensions must be positive");
      return 0;
    }
    auto const size = static_cast<std::size_t>(dimension);
    if (elementCount > std::numeric_limits<std::size_t>::max() / size) {
      MOCHI_ERROR_SET(error, "Sensor output element count overflows size_t");
      return 0;
    }
    elementCount *= size;

    SensorAxis const& axis = format.axes[i];
    if (axis.name.empty()) {
      MOCHI_ERROR_SET(error, "Sensor output axis names must not be empty");
      return 0;
    }
    if (!axis.labels.empty() && axis.labels.size() != size) {
      MOCHI_ERROR_SET(error, "Sensor output axis labels must match their dimension");
      return 0;
    }
    if (!axis.units.empty()) {
      if (hasUnits || (axis.units.size() != 1 && axis.units.size() != size)) {
        MOCHI_ERROR_SET(error, "Sensor output units must belong to one compatible axis");
        return 0;
      }
      hasUnits = true;
    }
  }
  return elementCount;
}

} // namespace

void SensorBase::Init(
    Span<SensorBase const* const> inputs,
    SensorScope const& scope,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  _inputs.assign(inputs.begin(), inputs.end());
  PrepareOutput(InitSensor(scope, error), error);
}

void SensorBase::PrepareOutput(SensorFormat format, Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::size_t const elementCount = ValidateFormatAndGetElementCount(format, error);
  MOCHI_ERROR_RETURN(error);
  if (elementCount > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    MOCHI_ERROR_SET(error, "Sensor output element count overflows int64_t");
    return;
  }

  _format = std::move(format);
  _output.resize(elementCount);
  std::ranges::fill(_output, 0.0F);
  _latest = {
      .data = _output.data(),
      .elementCount = static_cast<int64_t>(elementCount),
  };
}

void SensorBase::Reset() {
  ClearOutput();
  ResetSensor();
}

void SensorBase::ClearOutput() {
  std::ranges::fill(_output, 0.0F);
  _latest.captureStep = -1;
  _latest.captureTime = 0.0;
  _latest.publishedStep = -1;
  _latest.publishedTime = 0.0;
  _latest.data = _output.data();
  _latest.elementCount = static_cast<int64_t>(_output.size());
}

bool SensorBase::PublishSample(
    StepTime const& now,
    Span<float const> sample,
    Error& error,
    SensorSample const* source) {
  MOCHI_ERROR_RETURN(error, false);
  if (source != nullptr && source->captureStep < 0) {
    return false;
  }
  MOCHI_ERROR_IF_NOT(
      sample.size() == _output.size(),
      error,
      "Sensor sample element count does not match its format");
  MOCHI_ERROR_RETURN(error, false);
  std::ranges::copy(sample, _output.begin());
  _latest.captureStep = source != nullptr ? source->captureStep : now.step;
  _latest.captureTime = source != nullptr ? source->captureTime : now.t;
  _latest.publishedStep = now.step;
  _latest.publishedTime = now.t;
  return true;
}

Span<std::byte const> SensorBase::TryGetMetadata(std::string_view /*key*/) const {
  return {};
}

void SensorBase::ConfigureRecording(SensorRecordingSink& /*sink*/, Error& /*error*/) const {}

void SensorBase::Record(SensorRecordingSink& sink, Error& error) const {
  sink.WriteSample(Latest(), error);
}

TransformRT SensorBase::GetWorldTransform() const {
  if (Actor* const actor = GetActor()) {
    // Actor transforms are already world-resolved; composing a scene transform here would apply
    // the articulation hierarchy twice.
    return actor->GetRootTransform() * _parentFromSensor;
  }
  // An unattached sensor uses the scene root as its parent. The root-to-world transform remains an
  // identity surrogate until scene spawn transforms are supported.
  return _parentFromSensor;
}
