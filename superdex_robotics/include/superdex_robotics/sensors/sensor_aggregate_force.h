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

#include <superdex_robotics/sensors/interaction_wrench_metadata.h>
#include <superdex_robotics/sensors/sensor_base.h>
#include <superdex_robotics/sensors/tactile_observation.h>

namespace superdex::robotics {

struct SensorAggregateForceParams {
  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorAggregateForceParams)
  /* Register parameter fields here with MOCHI_FIELD. */
  MOCHI_STRUCT_END()
};

/* @brief Sum one borrowed upstream Force3D [N,3] tensor into a [3] force vector. */
class MOCHI_API SensorAggregateForce final : public SensorBase {
 public:
  using Params = SensorAggregateForceParams;

  static constexpr std::string_view TypeName() {
    return "SENSOR_AGGREGATE_FORCE";
  }

  SensorAggregateForce(Actor* actor, std::string_view paramArgs, superdex::Error& error);
  explicit SensorAggregateForce(Params const& params, superdex::Error& error);

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

  [[nodiscard]] Span<std::byte const> TryGetMetadata(std::string_view key) const override;

  [[nodiscard]] Params const& GetParams() const {
    return _params;
  }

 private:
  void ResetSensor() override {}

  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;

  void Initialize(Params const& params, superdex::Error& error);

  Params _params;
  InteractionWrenchMeta _interactionMetadata;
};

} // namespace superdex::robotics
