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

#include <superdex_robotics/sensors/sensor_base.h>

namespace superdex::robotics {

struct SensorConcatParams {
  DynamicString axisName = "input";
  DynamicArray<DynamicString> inputLabels;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorConcatParams)
  MOCHI_FIELD(axisName);
  MOCHI_FIELD(inputLabels);
  MOCHI_STRUCT_END()
};

/* @brief Stack equally formatted inputs along a new leading axis. Publish on any input update,
 * holding the other inputs' latest values. The sample capture time is the newest input capture;
 * per-input capture times remain available separately and are included in recording.
 * The output retains the inputs' format kind, units and component axes. */
class MOCHI_API SensorConcat final : public SensorBase {
 public:
  using Params = SensorConcatParams;

  static constexpr std::string_view TypeName() {
    return "SENSOR_CONCAT";
  }

  SensorConcat(Actor* actor, std::string_view paramArgs, superdex::Error& error);
  explicit SensorConcat(Params const& params, superdex::Error& error);

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

  [[nodiscard]] double GetPublicationFrequencyHzEstimate() const override;

  void ConfigureRecording(SensorRecordingSink& sink, superdex::Error& error) const override;
  void Record(SensorRecordingSink& sink, superdex::Error& error) const override;

  [[nodiscard]] Params const& GetParams() const {
    return _params;
  }

  [[nodiscard]] Span<double const> GetInputCaptureTimes() const {
    return _inputCaptureTimes;
  }

  [[nodiscard]] Span<int64_t const> GetInputCaptureSteps() const {
    return _inputCaptureSteps;
  }

 private:
  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;
  void ResetSensor() override;

  Params _params;
  DynamicArray<float> _values;
  DynamicArray<int64_t> _inputPublishedSteps;
  DynamicArray<int64_t> _inputCaptureSteps;
  DynamicArray<double> _inputCaptureTimes;
};

} // namespace superdex::robotics
