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

struct SensorDelayLineParams {
  /* Time from input publication to output publication [ms]. */
  double delayMs = 0.0;

  /* @brief Pending-history duration used to size storage at Init [ms].
   * Choose a duration covering the maximum delay; underestimated capacity reports overflow. */
  double bufferMs = 1000.0;

  /* @brief Fallback publication rate [Hz] when the input has no positive rate estimate. */
  double bufferAssumedFrequencyHzEstimate = 100.0;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorDelayLineParams)
  MOCHI_FIELD(delayMs) MOCHI_ATTRIBUTE(Units("ms"));
  MOCHI_FIELD(bufferMs) MOCHI_ATTRIBUTE(Units("ms"));
  MOCHI_FIELD(bufferAssumedFrequencyHzEstimate) MOCHI_ATTRIBUTE(Units("1/s"));
  MOCHI_STRUCT_END()
};

/* @brief Deterministic time-domain delay over one fixed CPU sensor output. */
class MOCHI_API SensorDelayLine final : public SensorBase {
 public:
  using Params = SensorDelayLineParams;

  static constexpr std::string_view TypeName() {
    return "SENSOR_DELAY_LINE";
  }

  SensorDelayLine(Actor* actor, std::string_view paramArgs, superdex::Error& error);
  SensorDelayLine(Actor* actor, Params const& params, superdex::Error& error);
  explicit SensorDelayLine(Params const& params, superdex::Error& error);
  ~SensorDelayLine() override;

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;
  [[nodiscard]] Span<std::byte const> TryGetMetadata(std::string_view key) const override;

  [[nodiscard]] Params GetParams() const {
    return _params;
  }

  /* Apply episode parameters before graph Reset; initialized capacity is immutable. */
  void SetParams(Params const& params, superdex::Error& error);

 private:
  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;
  void ResetSensor() override;
  void Initialize(Params const& params, superdex::Error& error);
  void Enqueue(SensorSample const& source, superdex::Error& error);
  [[nodiscard]] bool PublishDue(StepTime const& now, superdex::Error& error);

  Params _params;
  std::size_t _elementsPerSample = 0;
  std::size_t _capacity = 0;
  std::size_t _head = 0;
  std::size_t _size = 0;
  DynamicArray<float> _floatRing;
  DynamicArray<int64_t> _captureSteps;
  DynamicArray<double> _captureTimes;
  DynamicArray<double> _dueTimes;
  int64_t _lastInputPublishedStep = -1;
};

} // namespace superdex::robotics
