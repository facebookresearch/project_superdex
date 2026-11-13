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

#include <mochi_core/utils/dynamic_string.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <memory>

namespace superdex::robotics {

struct BotPrefab;
struct IBotLoader;
struct BotIKGoalConfig;

/** @brief Creation-time soft penalty qB ~= ratio*qA + offset on the private output actor.
 * Legacy runtime callers may use mixed revolute/prismatic pairs and repeated pairs.
 * Disabled or zero-stiffness entries need no model binding, but names and numbers must be valid.
 */
struct BotIKJointCouplingParams {
  mochi::DynamicString jointA;
  mochi::DynamicString jointB;
  mochi::real ratio = mochi::real(1);
  mochi::real offset = mochi::real(0);
  mochi::real stiffness = mochi::real(10);
  bool enabled = true;
};

/** @brief Options applied to the private IK-only copy of a prepared @ref BotPrefab. */
struct BotIKCreateParams {
  /** @brief Weld the root to @ref BotPrefab::worldFromRoot; otherwise retain its joint type. */
  bool fixedBase = false;
  /** @brief Omit collision geometry and skin. Contact is disabled even when geometry is retained.
   */
  bool excludeCollisionShapes = true;
  /** @brief Numerical termination and goal-reached thresholds of the experimental solver. */
  mochi::experimental::IKSolverParams solverParams;
};

/**
 * @brief Synchronous, owning bot IK convenience API using @ref mochi::experimental::IKSolver.
 *
 * @details Owns a dedicated scene, private robotics context, bot, goals, and pose snapshots.
 * Sensors and per-link actuator components are omitted; authored transmissions, tendons, cycles,
 * and joint limits remain. Goals and mechanical constraints are soft penalties, not guarantees.
 * Operations must be externally serialized. The borrowed Mochi context must outlive this object.
 */
class MOCHI_API BotIK {
 public:
  /**
   * @brief Create an IK instance without mutating or reordering @p prefab.
   *
   * @param context Borrowed Mochi context; must outlive the returned instance.
   * @param prefab Prepared prefab, normally returned by a bot loader after @ref RebuildBotData.
   * Inconsistent cached topology/DOF data is rejected; the seed is never silently clamped.
   * @param params Creation options.
   * @param loader Shape loader, borrowed only for this call.
   * @param error Error status.
   * @return Owning instance to release with @ref Destroy, or nullptr on error, with partial
   * resources released.
   */
  [[nodiscard]] static BotIK* Create(
      mochi::Context* context,
      BotPrefab const& prefab,
      BotIKCreateParams const& params,
      IBotLoader const& loader,
      mochi::Error& error);

  /** @brief Convenience creation overload using @ref FileBotLoader. */
  [[nodiscard]] static BotIK* Create(
      mochi::Context* context,
      BotPrefab const& prefab,
      BotIKCreateParams const& params,
      mochi::Error& error);

  /** @brief Prevalidate the entire runtime batch without creating or modifying an actor.
   * Ratio must be finite and nonzero; offset finite; stiffness finite and nonnegative.
   * Active joints must bind uniquely by exact name to nonroot revolute or prismatic joints.
   */
  static void ValidateJointCouplings(
      BotPrefab const& prefab,
      mochi::Span<BotIKJointCouplingParams const> couplings,
      mochi::Error& error);

  /** @brief Create with private output-actor couplings; prefab and input bots are unchanged.
   */
  [[nodiscard]] static BotIK* Create(
      mochi::Context* context,
      BotPrefab const& prefab,
      BotIKCreateParams const& params,
      mochi::Span<BotIKJointCouplingParams const> couplings,
      IBotLoader const& loader,
      mochi::Error& error);

  /** @brief Coupling overload using @ref FileBotLoader. */
  [[nodiscard]] static BotIK* Create(
      mochi::Context* context,
      BotPrefab const& prefab,
      BotIKCreateParams const& params,
      mochi::Span<BotIKJointCouplingParams const> couplings,
      mochi::Error& error);

  /** @brief Release an instance in the native library, leaving its borrowed context untouched.
   *
   * @param ik Instance returned by @ref Create, or nullptr. Do not delete the pointer directly.
   */
  static void Destroy(BotIK* ik);

  BotIK(BotIK const&) = delete;
  BotIK& operator=(BotIK const&) = delete;
  BotIK(BotIK&&) = delete;
  BotIK& operator=(BotIK&&) = delete;

  /** @brief Forward numerical and goal-reached parameters to the experimental solver. */
  void SetSolverParams(mochi::experimental::IKSolverParams const& params);
  /** @brief Return the current experimental solver parameters. */
  [[nodiscard]] mochi::experimental::IKSolverParams GetSolverParams() const;

  /**
   * @brief Upsert a link-local point [m] targeting a world-space point [m].
   *
   * @details One position goal per output-prefab link index. Geometry must be finite and
   * stiffness finite and nonnegative [N/m]. Zero stiffness keeps the goal present. An invalid
   * update preserves the previous goal. Reuses the constraint when the local pivot is unchanged.
   */
  void SetPositionGoal(
      int linkIndex,
      mochi::Real3 const& localPosition,
      mochi::Real3 const& worldPosition,
      mochi::real stiffness,
      mochi::Error& error);
  /**
   * @brief Upsert a link-local pivot rotation targeting a world-space pivot rotation.
   *
   * @details One independent rotation goal per output-prefab link index. Quaternions must be
   * finite and nonzero, and are normalized. Stiffness must be finite and nonnegative [N m/rad].
   * Invalid updates preserve the previous goal. Damping is zero for both goal types.
   */
  void SetRotationGoal(
      int linkIndex,
      mochi::Quaternion const& localRotation,
      mochi::Quaternion const& worldRotation,
      mochi::real stiffness,
      mochi::Error& error);

  /** @brief Idempotently remove a position goal; an out-of-range index reports an error. */
  void ClearPositionGoal(int linkIndex, mochi::Error& error);
  /** @brief Idempotently remove a rotation goal; an out-of-range index reports an error. */
  void ClearRotationGoal(int linkIndex, mochi::Error& error);
  /** @brief Idempotently remove all position and rotation goals. */
  void ClearAllGoals(mochi::Error& error);

  /** @brief Replace IK goals with goals resolved from another bot's forward kinematics.
   *
   * @details Empty configuration uses the input bot's default pose. Root placement is explicit
   * in world space. Bound goal frames follow inputBotLink; unbound frames use this IK bot's
   * authored initial base. Zero stiffness disables a component. Invalid input preserves goals.
   * Meant to run every frame: the input bot must be prepared (@ref RebuildBotData) and valid
   * (@ref Validate), and the goals valid (@ref ValidateIKConfig), as neither is rechecked here.
   */
  void SetInputBotConfiguration(
      BotPrefab const& inputBot,
      mochi::Span<mochi::real const> configuration,
      mochi::TransformRT const& worldFrameFromInputBaseFrame,
      mochi::Span<BotIKGoalConfig const> goals,
      mochi::Error& error);

  /* @brief Scale-aware retargeting; the original overload uses unit retarget scale.
   * Only input-bot root-relative FK translations and bound goal offsets are scaled.
   * Native input bot scale is respected separately. Invalid input preserves existing goals.
   */
  void SetInputBotConfiguration(
      BotPrefab const& inputBot,
      mochi::Span<mochi::real const> configuration,
      mochi::TransformRT const& worldFrameFromInputBaseFrame,
      mochi::real inputBotScale,
      mochi::Span<BotIKGoalConfig const> goals,
      mochi::Error& error);

  /**
   * @brief Warm-start from the current configuration and solve synchronously.
   *
   * @return Native goals-reached result, separately from @p error. Error-free solves refresh
   * snapshots even when goals are unreachable. No-goal solves report an error and preserve
   * snapshots. Getters never initiate a solve.
   */
  [[nodiscard]] bool SolveIK(mochi::Error& error);
  /**
   * @brief Native nonlinear-solver statistics of the most recent solve step.
   *
   * @note Reports the IK scene's last step: iterations, line-search iterations, residual norm and
   * convergence status. A solve rejected before stepping (for example, with no goals) leaves the
   * previous statistics in place.
   */
  [[nodiscard]] mochi::SolverStats GetLastSolverStats() const;

  /**
   * @brief Snapshot of joint positions [m/rad], excluding root DOFs for both Free and Hard roots.
   *
   * @note Initially contains the created seed. Reacquire views after mutation; views must not
   * outlive this object. The same lifetime rule applies to @ref GetWorldFromLinks.
   */
  [[nodiscard]] mochi::Span<mochi::real const> GetJointPositions() const;
  /** @brief World-from-link snapshots in output-prefab order, including solved base placement.
   */
  [[nodiscard]] mochi::Span<mochi::TransformRT const> GetWorldFromLinks() const;
  /** @brief Replace bot-space joint positions [m/rad], preserving the current base placement. */
  void SetJointPositions(mochi::Span<mochi::real const> positions, mochi::Error& error);

  /** @brief Number of links in the output prefab. */
  [[nodiscard]] int GetNumLinks() const;
  /** @brief Number of bot-space DOFs, excluding the root joint. */
  [[nodiscard]] int GetNumDofs() const;
  /** @brief Whether the link connects to the root through Hard joints only; false out of range. */
  [[nodiscard]] bool IsLinkRigidlyAttachedToRootLink(int linkIndex) const;

 private:
  struct Impl;
  explicit BotIK(std::unique_ptr<Impl> impl);
  ~BotIK();
  std::unique_ptr<Impl> _impl;
};

} // namespace superdex::robotics
