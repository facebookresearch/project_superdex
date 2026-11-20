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

struct SensorGaussianNoiseParams {
  /* Standard deviation in input element units; scalar broadcast or one value per element. */
  DynamicArray<real> std = {0_r};
  SensorRandomParams random;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorGaussianNoiseParams)
  MOCHI_FIELD(std);
  MOCHI_FIELD(random);
  MOCHI_STRUCT_END()
};

/* @brief Independent zero-mean Gaussian innovations, once per new input capture. */
class MOCHI_API SensorGaussianNoise final : public SensorBase {
 public:
  using Params = SensorGaussianNoiseParams;
  static constexpr std::string_view TypeName() {
    return "SENSOR_GAUSSIAN_NOISE";
  }
  SensorGaussianNoise(Actor* actor, std::string_view paramArgs, Error& error);
  SensorGaussianNoise(Actor* actor, Params const& params, Error& error);
  explicit SensorGaussianNoise(Params const& params, Error& error);
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
  void ConfigureRecording(SensorRecordingSink& sink, Error& error) const override;

 private:
  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, Error& error) override;
  void ResetSensor() override;
  Params _params;
  DynamicArray<float> _values;
  int64_t _lastCaptureStep = -1;
};

} // namespace superdex::robotics
