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
#include <superdex_robotics/controllers/controller_base.h>

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/reflection.h>

namespace superdex::robotics {

/* @brief Joint-space PD controller parameters. */
struct ControllerBasicJscPdParams {
  // joint space gains
  DynamicArray<real> Kp = {}; // The position gain [Nm/rad]
  DynamicArray<real> Kd = {}; // The position damping gain [Nms/rad]

  // The saturation magnitude of the controller output for each joint.
  // Values < 0 are interpreted as infinitiy
  DynamicArray<real> saturation = {}; // [N or Nm]

  // The deadband is removed from error towards 0 (limited to 0) before multiplying by Kp.
  // Damping is always applied regardless of the deadband.
  // The deadband results in a force profile about the target like this \___/ rather than \/
  DynamicArray<real> deadband = {};

  /**
   * @brief Load controller parameters from a JSON file.
   * @param[in] path Path to a .superdex_controller JSON file.
   * @param[in,out] error Error status.
   * @return Loaded parameters, or default-constructed on failure.
   */
  static MOCHI_API ControllerBasicJscPdParams
  LoadFromFile(std::string_view path, superdex::Error& error);

  /**
   * @brief Save controller parameters to a JSON file.
   * @param[in] path Destination file path.
   * @param[in,out] error Error status.
   */
  MOCHI_API void SaveToFile(std::string_view path, superdex::Error& error) const;

  MOCHI_STRUCT_BEGIN(superdex::robotics::ControllerBasicJscPdParams)
  MOCHI_FIELD(Kp) MOCHI_ATTRIBUTE(Units("N/m or Nm/rad"));
  MOCHI_FIELD(Kd) MOCHI_ATTRIBUTE(Units("Ns/m or Nms/rad"));
  MOCHI_FIELD(saturation) MOCHI_ATTRIBUTE(Units("N or Nm"));
  MOCHI_FIELD(deadband) MOCHI_ATTRIBUTE(Units("m or rad"));
  MOCHI_STRUCT_END()
};

/* @brief Observation state — robot state variables read each control step.
 * Fill it with GetCurrentObservationsFromMochi to read the live simulation, or populate it
 * directly from an external state source (e.g., a real robot). */
struct ControllerBasicJscPdObsv {
  DynamicArray<real> dofPositions; /* All DOF positions [rad or m] */
  DynamicArray<real> dofVelocities; /* All DOF velocities [rad/s or m/s] */
  real dt = 0_r; /* Time since the last ComputeOutput call [s] */
};

/* @brief Target setpoint for the controller. */
struct ControllerBasicJscPdTarget {
  DynamicArray<real> targetPose; /* Target joint positions [m or rad] */
};

/* @brief A simple PD controller written in joint space. */

// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API ControllerBasicJscPd : public ControllerBase {
 public:
  /* @brief Registration type name for this controller (see
   * RoboticsContext::RegisterControllerType).
   * @return This controller's registration type name. */
  static constexpr std::string_view TypeName() {
    return "BASIC_JSC_PD";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  using Params = ControllerBasicJscPdParams;
  using Obsv = ControllerBasicJscPdObsv;
  using Target = ControllerBasicJscPdTarget;

  /* @brief Construct a joint-space PD controller for the given articulated robot, optionally
   * holding a borrowed robot-model description (see GetBotPrefab).
   * @param prefab Optional borrowed BotPrefab model (may be null; must outlive the controller).
   * @param robot The articulated robot actor to control.
   * @param error Error status. */
  ControllerBasicJscPd(BotPrefab const* prefab, Actor* robot, superdex::Error& error);

  /* @brief No-op initialization for pattern consistency with OSC.
   * JSC requires no link resolution; params are set via SetParams(). */
  void Initialize(superdex::Error& /*error*/) {}

  /* @brief Get controller parameters.
   * @return Reference to parameters. */
  Params const& GetParams() const;

  /* @brief Set controller parameters.
   * @param params New parameters.
   * @param error Error status. */
  void SetParams(Params const& params, superdex::Error& error);

  /* @brief Configure from a scene-file controller entry (loads params; needs no init args).
   * Override of ControllerBase. */
  void ConfigureFromSceneEntry(
      std::string_view paramArgs,
      std::string_view initArgs,
      superdex::Error& error) override;

  /* @brief Read current robot state from the Mochi actor.
   * @param error Error status.
   * @return Observation struct populated from the actor's current state. */
  Obsv GetCurrentObservationsFromMochi(superdex::Error& error);

  /* @brief Compute PD control efforts from observation state and target setpoint.
   * @param obsv Current robot state (positions, velocities, dt).
   * @param target Target setpoint.
   * @param error Error status.
   * @return Efforts for all DOFs [N or Nm]. Returns empty span on error. */
  Span<real const> ComputeOutput(Obsv const& obsv, Target const& target, superdex::Error& error);

  /* @brief Reset internal controller state (e.g., between episodes). */
  void Reset() override;

 protected:
  void CheckParams(Params const& params, superdex::Error& error);

  /* The controller parameters */
  Params _params;

  /* The efforts to be applied */
  ColumnVector<real> _efforts;

  /* The prior ComputeOutput call targetPose are stored here for computing target velocity */
  DynamicArray<real> _prevTargetPose;
  bool _prevTargetPoseInitialized = false;

  ~ControllerBasicJscPd() override = default;
};

} // namespace superdex::robotics
