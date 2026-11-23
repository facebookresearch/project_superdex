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
#include <superdex_robotics/sensors/sensor_random_params.h>

namespace superdex::robotics {

struct SensorUniformNoiseParams {
  /* Bounds in input element units: each value gains a draw uniform in [low, high). */
  double low = 0.0;
  double high = 0.0;
  SensorRandomParams random;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorUniformNoiseParams)
  MOCHI_FIELD(low);
  MOCHI_FIELD(high);
  MOCHI_FIELD(random);
  MOCHI_STRUCT_END()
};

/* @brief Independent additive uniform noise, drawn once per input publication from the episode
 * seed, the stream key and the input's published step. */
class MOCHI_API SensorUniformNoise final : public SensorBase {
 public:
  using Params = SensorUniformNoiseParams;
  static constexpr std::string_view TypeName() {
    return "SENSOR_UNIFORM_NOISE";
  }
  SensorUniformNoise(Actor* actor, std::string_view paramArgs, Error& error);
  explicit SensorUniformNoise(Params const& params, Error& error);
  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }
  [[nodiscard]] Params GetParams() const {
    return _params;
  }
  /* Install episode parameters before the caller resets the graph. */
  void SetParams(Params const& params, Error& error);
  [[nodiscard]] SensorStepResult Step(StepTime const& now, Error& error) override;
  [[nodiscard]] Span<std::byte const> TryGetMetadata(std::string_view key) const override;

 private:
  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, Error& error) override;
  void ResetSensor() override;

  Params _params;
  DynamicArray<float> _values;
  int64_t _lastInputPublishedStep = -1;
};

} // namespace superdex::robotics
