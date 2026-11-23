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

struct SensorOutlierInjectionParams {
  /* Probability that an input sample is replaced by an outlier. */
  double probability = 0.0;
  /* An outlier's values are uniform in [-range, range). */
  double range = 0.0;
  /* With rotation, the input is two rows of a rotation matrix (6 values) and an outlier is two rows
   * of a uniformly random rotation instead. */
  bool rotation = false;
  SensorRandomParams random;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorOutlierInjectionParams)
  MOCHI_FIELD(probability);
  MOCHI_FIELD(range);
  MOCHI_FIELD(rotation);
  MOCHI_FIELD(random);
  MOCHI_STRUCT_END()
};

/* @brief Replaces whole input samples by outliers with a fixed probability, drawn once per input
 * publication from the episode seed, the stream key and the input's published step. Other samples
 * pass unchanged. */
class MOCHI_API SensorOutlierInjection final : public SensorBase {
 public:
  using Params = SensorOutlierInjectionParams;
  static constexpr std::string_view TypeName() {
    return "SENSOR_OUTLIER_INJECTION";
  }
  SensorOutlierInjection(Actor* actor, std::string_view paramArgs, Error& error);
  explicit SensorOutlierInjection(Params const& params, Error& error);
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
  DynamicArray<float> _outlier;
  int64_t _lastInputPublishedStep = -1;
};

} // namespace superdex::robotics
