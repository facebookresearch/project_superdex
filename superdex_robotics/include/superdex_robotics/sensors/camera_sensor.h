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

#include <superdex_physics.h>
#include <superdex_robotics/sensors/sensor_base.h>

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/reflection.h>

namespace superdex::robotics {

/* @brief Parameters for a camera sensor (fixed or wrist-mounted).
 * Loaded from a .superdex_sensor JSON file via SReflect.
 * Fields with value 0 mean "use engine default". */
struct CameraSensorParams {
  DynamicString name;

  int imageWidth = 0;
  int imageHeight = 0;

  real fovVerticalDeg = 0_r;
  real nearClip = 0_r;
  real farClip = 0_r;

  /* Look-at target in world space (fixed cameras only). */
  Real3 lookAt = {};

  /* Offset from EE frame origin in local coordinates (wrist cameras only). */
  Real3 offsetLocal = {};

  /* Forward direction in EE-local frame (wrist cameras only). */
  Real3 forwardAxis = {};

  /* Up direction in EE-local frame (wrist cameras only). */
  Real3 upAxisLocal = {};

  /* Distance ahead of offset point to place look-at target (wrist cameras only). */
  real lookDistance = 0_r;

  /* @brief Load sensor parameters from a JSON file.
   * @param path Path to a .superdex_sensor JSON file.
   * @param error Error status.
   * @return Loaded parameters. */
  static MOCHI_API CameraSensorParams LoadFromFile(std::string_view path, superdex::Error& error);

  /* @brief Save sensor parameters to a JSON file.
   * @param path Destination file path.
   * @param error Error status. */
  MOCHI_API void SaveToFile(std::string_view path, superdex::Error& error) const;

  MOCHI_STRUCT_BEGIN(superdex::robotics::CameraSensorParams)
  MOCHI_FIELD(name)
  MOCHI_FIELD(imageWidth) MOCHI_ATTRIBUTE(Units("px"));
  MOCHI_FIELD(imageHeight) MOCHI_ATTRIBUTE(Units("px"));
  MOCHI_FIELD(fovVerticalDeg) MOCHI_ATTRIBUTE(Units("deg"));
  MOCHI_FIELD(nearClip) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(farClip) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(lookAt) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(offsetLocal) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(forwardAxis)
  MOCHI_FIELD(upAxisLocal)
  MOCHI_FIELD(lookDistance) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_STRUCT_END()
};

/* @brief Camera sensor runtime object.
 * Stores intrinsics/extrinsics for use by renderers (UE, etc.).
 * No ComputeSignal — cameras don't produce data in the mochi physics loop;
 * renderers query GetParams() to configure their camera actors. */
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API CameraSensor : public SensorBase {
 public:
  /* @brief Registration type name for this sensor (see RoboticsContext::RegisterSensorType).
   * @return This sensor's registration type name. */
  static constexpr std::string_view TypeName() {
    return "SENSOR_CAMERA";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  /* Nothing to clear between episodes: a camera holds only its params and pose. */
  void Reset() override {}

  using Params = CameraSensorParams;

  /* @brief Construct from @p paramArgs -- either a params file path or an inline JSON string (empty
   * uses defaults); the sensor loads its own params. This is the factory
   * RoboticsContext::RegisterSensor uses. */
  CameraSensor(Actor* linkActor, std::string_view paramArgs, superdex::Error& error);

  /* @brief Construct from a pre-loaded params struct (programmatic / non-filesystem construction).
   */
  CameraSensor(Actor* linkActor, Params const& params, superdex::Error& error);

  /* @brief Get sensor parameters.
   * @return The params. */
  [[nodiscard]] Params const& GetParams() const {
    return _params;
  }

  /* @brief Set sensor parameters.
   * @param params Params.
   * @param error Error status. */
  void SetParams(Params const& params, [[maybe_unused]] superdex::Error& error) {
    _params = params;
  }

 private:
  Params _params;
};

} // namespace superdex::robotics
