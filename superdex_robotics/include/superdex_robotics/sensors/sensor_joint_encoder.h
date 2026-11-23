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

#include <string_view>

namespace superdex::robotics {

struct SensorJointEncoderParams {
  /* Names of the joints to read, in output order. Empty reads every one-DOF joint of the owning bot
   * but its root, in the bot's order. */
  DynamicArray<DynamicString> joints;
  /* Calibration offset added to each joint's position [rad or m], in output order; empty is zeros.
   */
  DynamicArray<real> offset;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorJointEncoderParams)
  MOCHI_FIELD(joints) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(offset) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};

/* @brief The owning bot's one-DOF joint states, published at every graph step, the reset step
 * included, as a [2, N] tensor: row 0 the positions plus per-joint calibration offsets, row 1 the
 * velocities, columns the joints. Declared on any link of the bot. */
class MOCHI_API SensorJointEncoder final : public SensorBase {
 public:
  using Params = SensorJointEncoderParams;

  static constexpr std::string_view TypeName() {
    return "SENSOR_JOINT_ENCODER";
  }

  SensorJointEncoder(Actor* linkActor, std::string_view paramArgs, superdex::Error& error);
  SensorJointEncoder(Actor* linkActor, Params const& params, superdex::Error& error);
  ~SensorJointEncoder() override;

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

  [[nodiscard]] Params const& GetParams() const {
    return _params;
  }

  /* @brief Install new offsets, for example randomized ones. The joints cannot change once the
   * graph has initialized the sensor. */
  void SetParams(Params const& params, superdex::Error& error);

 private:
  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;
  void ResetSensor() override;

  Params _params;
  /* Each output joint's DOF in the articulated actor's pose and velocities. */
  DynamicArray<int> _dofs;
  DynamicArray<real> _pose;
  DynamicArray<real> _velocities;
  DynamicArray<float> _values;
};

} // namespace superdex::robotics
