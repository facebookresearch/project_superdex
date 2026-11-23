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

#include <superdex_robotics/sensors/sensor_function.h>

#include <utility>

using namespace mochi;
using namespace superdex::robotics;

SensorFunction::SensorFunction(SensorFormat format, Read read, Error& error)
    : SensorBase(error), _outputFormat{std::move(format)}, _read{std::move(read)} {
  std::size_t const elements = GetSensorFormatElementCount(_outputFormat, error);
  MOCHI_ERROR_IF(!_read, error, "SensorFunction needs a function");
  MOCHI_ERROR_RETURN(error);
  _values.resize(elements);
}

SensorFunction::~SensorFunction() = default;

SensorFormat SensorFunction::ValuesFormat(int32_t width) {
  return {
      .kind = "Values",
      .shape = {width},
      .axes = {{.name = "value", .labels = {}, .units = {}}},
  };
}

SensorFormat SensorFunction::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "SensorFunction accepts no inputs");
  MOCHI_ERROR_RETURN(error, {});
  return _outputFormat;
}

SensorStepResult SensorFunction::Step(StepTime const& now, Error& error) {
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  std::optional<SensorSample> const source = _read(MakeSpan(_values));
  return PublishSample(now, MakeConstSpan(_values), error, source ? &*source : nullptr)
      ? SensorStepResult::Published
      : SensorStepResult::NoUpdate;
}

void SensorFunction::ResetSensor() {}
