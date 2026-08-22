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

#include <superdex_robotics/controllers/controller_base.h>

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/transform_rt.h>

namespace superdex::robotics {

/* @brief Basic operational-space PD controller parameters.
 *
 * Minimal parameter set for a robot-agnostic task-space impedance controller.
 * See ControllerBasicOscPd for details. */
struct ControllerBasicOscPdParams {
  /* Tracking gains */
  real Kp_p = 0_r; /* Position gain [N/m] */
  real Kd_p = 0_r; /* Position damping gain [Ns/m] */
  real Kp_r = 0_r; /* Rotation gain [Nm/rad] */
  real Kd_r = 0_r; /* Rotation damping gain [Nms/rad] */

  /* If true, renormalize the task-space error so translation and rotation errors stay below their
   * configured maximums, keeping the force direction consistent. Requires positive
   * maxTranslationError and maxRotationError. */
  bool bApplyMaxErrorMagnitudeNormalization = true;

  /* Max translation error for normalization in control space [m]. Must be positive when enabled. */
  real maxTranslationError = 0_r;
  /* Max rotation error for normalization in control space [rad]. Must be positive when enabled. */
  real maxRotationError = 0_r;

  /* The transform of the actual end-effector in the end-effector link space */
  TransformRT EELinkFromEE = TransformRT::Identity();

  /* If true, scale the output efforts down uniformly so that no controlled DOF exceeds the
   * per-joint effort limit harvested from the controller's BotPrefab (see Initialize), preserving
   * the task-space force direction. Requires the controller to have been constructed with a
   * BotPrefab; only joints with a finite (positive) effortLimit constrain the scaling — unbounded
   * (negative) and non-actuated (zero) joints are ignored. */
  bool bApplyMaxOSCTorqueNormalization = true;

  /**
   * @brief Load controller parameters from a JSON file.
   * @param[in] path Path to a .superdex_controller JSON file.
   * @param[in,out] error Error status.
   * @return Loaded parameters, or default-constructed on failure.
   */
  static MOCHI_API ControllerBasicOscPdParams
  LoadFromFile(std::string_view path, superdex::Error& error);

  /**
   * @brief Save controller parameters to a JSON file.
   * @param[in] path Destination file path.
   * @param[in,out] error Error status.
   */
  MOCHI_API void SaveToFile(std::string_view path, superdex::Error& error) const;

  MOCHI_STRUCT_BEGIN(superdex::robotics::ControllerBasicOscPdParams)
  MOCHI_FIELD(Kp_p) MOCHI_ATTRIBUTE(Units("N/m"));
  MOCHI_FIELD(Kd_p) MOCHI_ATTRIBUTE(Units("Ns/m"));
  MOCHI_FIELD(Kp_r) MOCHI_ATTRIBUTE(Units("Nm/rad"));
  MOCHI_FIELD(Kd_r) MOCHI_ATTRIBUTE(Units("Nms/rad"));
  MOCHI_FIELD(bApplyMaxErrorMagnitudeNormalization);
  MOCHI_FIELD(maxTranslationError) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(maxRotationError) MOCHI_ATTRIBUTE(Units("rad"));
  MOCHI_FIELD(EELinkFromEE);
  MOCHI_FIELD(bApplyMaxOSCTorqueNormalization);
  MOCHI_STRUCT_END()
};

/* @brief Observation state — robot state variables read each control step.
 * Fill it with GetCurrentObservationsFromMochi to read the live simulation, or populate it
 * directly from an external state source (e.g., a real robot). */
struct ControllerBasicOscPdObsv {
  DynamicArray<real> dofPositions; /* All DOF positions [rad or m] */
  DynamicArray<real> dofVelocities; /* All DOF velocities [rad/s or m/s] */
  TransformRT worldFromRoot = TransformRT::Identity(); /* Root link pose in world frame */
  TransformRT worldFromEELink = TransformRT::Identity(); /* EE link pose in world frame */
  DynamicArray<real> eeJacobian; /* 6×numDofs EE Jacobian in world frame (row-major) */
};

/* @brief Target setpoint for the controller. */
struct ControllerBasicOscPdTarget {
  TransformRT rootFromTargetEE = TransformRT::Identity(); /* Target EE pose in root frame */
};

/* @brief Scene-entry init arguments for ControllerBasicOscPd, deserialized from the controller's
 * JSON object in ConfigureFromSceneEntry. Unrecognized keys (type/name/params, and any fields
 * belonging to other controller types) are ignored on load. */
struct ControllerBasicOscPdInitArgs {
  DynamicString baseLinkName; /* Base link name (root of the control chain). */
  DynamicString eeLinkName; /* End-effector link name. */

  MOCHI_STRUCT_BEGIN(superdex::robotics::ControllerBasicOscPdInitArgs)
  MOCHI_FIELD(baseLinkName)
  MOCHI_FIELD(eeLinkName)
  MOCHI_STRUCT_END()
};

/* @brief Basic operational-space PD (impedance) controller for robotic arms.
 *
 * Computes joint torques via the Jacobian transpose method: tau = J^T * F, where
 * F = Kp * pose_error + Kd * velocity_error combines position/rotation tracking errors with PD
 * gains. This is the minimal, robot-agnostic core of operational-space control:
 *
 *   - No nullspace or posture bias. Redundant (extra-DOF) arms are free to drift in the nullspace;
 *     only the end-effector pose is regulated.
 *   - Per-DOF effort clamping (direction-preserving vector scaling), enabled by default; the
 *     per-joint effort limits are harvested from the controller's BotPrefab (see Initialize).
 *   - No joint/velocity-limit avoidance, friction/gravity compensation, or inertia decoupling;
 *     those belong to richer controllers (see the internal OSC variants).
 *
 * Intended as a clean, open-sourceable baseline that works on any articulated arm without
 * robot-specific tuning tables. */

// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API ControllerBasicOscPd : public ControllerBase {
 public:
  /* @brief Registration type name for this controller (see
   * RoboticsContext::RegisterControllerType).
   * @return This controller's registration type name. */
  static constexpr std::string_view TypeName() {
    return "BASIC_OSC_PD";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  using Params = ControllerBasicOscPdParams;
  using Obsv = ControllerBasicOscPdObsv;
  using Target = ControllerBasicOscPdTarget;

  /* @brief Construct a basic operational-space PD controller for the given articulated robot,
   * optionally holding a borrowed robot-model description (see GetBotPrefab). After construction,
   * call Initialize(baseLinkName, eeLinkName, error) to resolve links.
   * @param prefab Optional borrowed BotPrefab model (may be null; must outlive the controller).
   * @param robot The articulated robot actor to control.
   * @param error Error status. */
  ControllerBasicOscPd(BotPrefab const* prefab, Actor* robot, superdex::Error& error);

  /* @brief Two-phase initialization: resolve base and end-effector links by name.
   * Must be called after creation and before @ref ComputeOutput.
   * @param baseLinkName Name of the base link (root of control chain).
   * @param eeLinkName Name of the end-effector link.
   * @param error Error status. */
  void
  Initialize(std::string_view baseLinkName, std::string_view eeLinkName, superdex::Error& error);

  /* @brief Get controller parameters.
   * @return The params. */
  [[nodiscard]] Params const& GetParams() const;

  /* @brief Set controller parameters.
   * @param params Params.
   * @param error Error status. */
  void SetParams(Params const& params, superdex::Error& error);

  /* @brief Configure from a scene-file controller entry: reads "baseLinkName"/"eeLinkName" from the
   * init args and resolves them, then loads params. Override of ControllerBase. */
  void ConfigureFromSceneEntry(
      std::string_view paramArgs,
      std::string_view initArgs,
      superdex::Error& error) override;
  /* @brief Get the controlled DOF indices (between base and end-effector links).
   * @return The controlled dofs. */
  Span<int const> GetControlledDofs();

  /* @brief Get the controlled link indices (between base and end-effector links).
   * @return The controlled links. */
  Span<int const> GetControlledLinks();

  /* @brief Get the end-effector link index.
   * @return The eelink index. */
  [[nodiscard]] int GetEELinkIndex() const;

  /* @brief Read current robot state from the Mochi actor.
   * @param error Error status.
   * @return Observation struct populated from the actor's current state. */
  Obsv GetCurrentObservationsFromMochi(superdex::Error& error);

  /* @brief Compute control efforts from observation state and target setpoint.
   * @param obsv Current robot state (positions, velocities, transforms).
   * @param target Target setpoint.
   * @param error Error status.
   * @return Efforts for all DOFs [Nm]. Uncontrolled DOFs are set to 0. Empty span on error. */
  Span<real const> ComputeOutput(Obsv const& obsv, Target const& target, superdex::Error& error);

  /* @brief Reset internal controller state (e.g., between episodes).
   * This controller is stateless between steps, so this is a no-op. */
  void Reset() override;

 protected:
  static void CheckParams(Params const& params, superdex::Error& error);

  /* The controller parameters */
  Params _params;

  /* Per-controlled-DOF effort limits [Nm], harvested from the BotPrefab in Initialize (empty when
   * the controller was constructed without a prefab). Consumed by ComputeOutput when
   * bApplyMaxOSCTorqueNormalization is enabled. Indexed like GetControlledDofs(). */
  DynamicArray<real> _effortLimits;

  /* The rigid link that serves as the base of the control chain (generally the 0th link in the
   * articulation), held as a stale-safe handle plus its index in Actor::GetNestedLinkActors order,
   * kept symmetric with the end-effector link. Currently only the index is consumed (ancestor walk
   * / DOF mapping); the handle is retained (like _eeLinkHandle) for future base-frame queries. */
  ActorHandle _baseLinkHandle;
  int _baseLinkIndex = -1;

  /* The rigid link to be controlled as end-effector, held as a stale-safe handle and resolved on
   * demand via ComponentBase::GetScene()->GetActor() so it never dangles if the scene is destroyed
   * before this controller. */
  ActorHandle _eeLinkHandle;
  int _eeLinkIndex = -1;

  /* The indices of the dofs between the eeLink and baseLink */
  DynamicArray<int> _dofIndices;
  /* The indices of the links between the eeLink and baseLink */
  DynamicArray<int> _linkIndices;

  /* The Jacobian of the forward kinematics function (controlled DOFs only) */
  Matrix<real> _jArm;
  /* The efforts to be applied (all DOFs; uncontrolled DOFs are 0) */
  ColumnVector<real> _efforts;

  /* The robot DOFs for the controlled links only */
  ColumnVector<real> _positionsArm;
  /* The velocity of the controlled links only at the current timestep */
  ColumnVector<real> _velocitiesArm;

  /* Pre-allocated temporaries for ComputeOutput (sized in Initialize) */
  Matrix<real> _tmpJArmRoot; /* 6 × numArm */
  ColumnVector<real> _effortsArm; /* numArm */

  ~ControllerBasicOscPd() override = default;
};

} // namespace superdex::robotics
