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

#include <cstdint>

namespace superdex::robotics {

struct SensorTemporalWindowParams {
  /* Number of native-rate samples in the oldest-first output window. */
  int32_t historyLength = 1;
  /* Rows before the first sample since a reset repeat that sample, its capture and publication
   * included, instead of holding zeros. */
  bool padWithFirst = false;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorTemporalWindowParams)
  MOCHI_FIELD(historyLength);
  MOCHI_FIELD(padWithFirst);
  MOCHI_STRUCT_END()
};

/* @brief Fixed oldest-first rolling window over one native-rate CPU float sensor. */
class MOCHI_API SensorTemporalWindow final : public SensorBase {
 public:
  using Params = SensorTemporalWindowParams;

  static constexpr std::string_view TypeName() {
    return "SENSOR_TEMPORAL_WINDOW";
  }

  SensorTemporalWindow(Actor* actor, std::string_view paramArgs, superdex::Error& error);
  SensorTemporalWindow(Actor* actor, Params const& params, superdex::Error& error);
  explicit SensorTemporalWindow(Params const& params, superdex::Error& error);
  ~SensorTemporalWindow() override;

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

  void ConfigureRecording(SensorRecordingSink& sink, superdex::Error& error) const override;
  void Record(SensorRecordingSink& sink, superdex::Error& error) const override;

  [[nodiscard]] Params const& GetParams() const {
    return _params;
  }

  /* Row metadata follows the output's oldest-first leading axis. Invalid rows have step -1; with
   * padWithFirst, rows before the first sample carry its metadata. */
  [[nodiscard]] Span<int64_t const> GetRowCaptureSteps() const {
    return _outputCaptureSteps;
  }

  [[nodiscard]] Span<double const> GetRowCaptureTimes() const {
    return _outputCaptureTimes;
  }

  [[nodiscard]] Span<int64_t const> GetRowPublishedSteps() const {
    return _outputPublishedSteps;
  }

  [[nodiscard]] Span<double const> GetRowPublishedTimes() const {
    return _outputPublishedTimes;
  }

 private:
  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;
  void ResetSensor() override;
  void Initialize(Params const& params, superdex::Error& error);
  void MaterializeOutput(SensorSample const& source, Span<float const> values);

  Params _params;
  std::size_t _elementsPerSample = 0;
  DynamicArray<float> _ring;
  DynamicArray<int64_t> _ringCaptureSteps;
  DynamicArray<double> _ringCaptureTimes;
  DynamicArray<int64_t> _ringPublishedSteps;
  DynamicArray<double> _ringPublishedTimes;
  DynamicArray<float> _output;
  DynamicArray<int64_t> _outputCaptureSteps;
  DynamicArray<double> _outputCaptureTimes;
  DynamicArray<int64_t> _outputPublishedSteps;
  DynamicArray<double> _outputPublishedTimes;
  std::size_t _nextWrite = 0;
  std::size_t _validCount = 0;
  int64_t _lastInputPublishedStep = -1;
};

} // namespace superdex::robotics
