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

#include <optional>

namespace superdex::robotics {

struct SensorLinkContactForceParams {
  /* Zero captures every graph step; a positive rate uses a drift-free acquisition-time schedule. */
  real sampleRateHz = 0_r;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorLinkContactForceParams)
  MOCHI_FIELD(sampleRateHz) MOCHI_ATTRIBUTE(Units("1/s"));
  MOCHI_STRUCT_END()
};

/* @brief Total contact force on a link, as a Force3D [3] environment-on-sensor force in the sensor
 * frame.
 *
 * Reads @ref QueryType::TotalContactForce, which includes every contact on the link whichever body
 * supplies the contact samples. Use this sensor for a whole-link force without requesting a
 * per-vertex distribution. */
class MOCHI_API SensorLinkContactForce final : public SensorBase {
 public:
  using Params = SensorLinkContactForceParams;

  static constexpr std::string_view TypeName() {
    return "SENSOR_LINK_CONTACT_FORCE";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  SensorLinkContactForce(Actor* linkActor, std::string_view paramArgs, superdex::Error& error);
  SensorLinkContactForce(Actor* linkActor, Params const& params, superdex::Error& error);
  ~SensorLinkContactForce() override;

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

  [[nodiscard]] Span<std::byte const> TryGetMetadata(std::string_view key) const override;

  [[nodiscard]] Params const& GetParams() const {
    return _params;
  }

  [[nodiscard]] double GetPublicationFrequencyHzEstimate() const override {
    return static_cast<double>(_params.sampleRateHz);
  }

 private:
  void ResetSensor() override;

  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;

  void Initialize(Params const& params, superdex::Error& error);

  Params _params;
  InteractionWrenchMeta _interactionMetadata;
  QueryHandle _query;
  bool _queryRegistered = false;
  std::optional<int64_t> _lastSamplePeriod;
  std::optional<double> _sampleEpochTime;
};

} // namespace superdex::robotics
