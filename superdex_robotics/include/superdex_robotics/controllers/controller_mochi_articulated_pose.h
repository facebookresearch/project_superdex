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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_physics/cpp_api/mochi_structs.h>

namespace superdex::robotics {

/* @brief ControllerMochiArticulatedPose parameters. */
struct ControllerMochiArticulatedPoseParams {
  PoseControllerParams poseControllerParams;

  /**
   * @brief Load controller parameters from a JSON file.
   * @param[in] path Path to a .superdex_controller JSON file.
   * @param[in,out] error Error status.
   * @return Loaded parameters, or default-constructed on failure.
   */
  static MOCHI_API ControllerMochiArticulatedPoseParams
  LoadFromFile(std::string_view path, superdex::Error& error);

  /**
   * @brief Save controller parameters to a JSON file.
   * @param[in] path Destination file path.
   * @param[in,out] error Error status.
   */
  MOCHI_API void SaveToFile(std::string_view path, superdex::Error& error) const;

  MOCHI_STRUCT_BEGIN(superdex::robotics::ControllerMochiArticulatedPoseParams)
  MOCHI_FIELD(poseControllerParams)
  MOCHI_STRUCT_END()
};

/* @brief Observation state (empty — this controller is target-driven). */
struct ControllerMochiArticulatedPoseObsv {};

/* @brief Target to drive the ArticulatedPoseController toward.
 * Provide exactly one of localToParentTransforms or poseDofs. */
struct ControllerMochiArticulatedPoseTarget {
  TransformRT worldFromRoot = TransformRT::Identity(); /* Root link pose in world frame */
  DynamicArray<TransformRT> localToParentTransforms; /* Per-link LocalToParent transforms */
  DynamicArray<real> poseDofs; /* Non-root joint DOFs; root DOFs are derived from worldFromRoot */
};

/* @brief ControllerMochiArticulatedPose -- drives a mochi ArticulatedPoseController
 * via either per-link LocalToParent transforms (converted to world-space targets)
 * or non-root pose DOFs combined with a world-space root transform. */

// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API ControllerMochiArticulatedPose : public ControllerBase {
 public:
  /* @brief Registration type name for this controller (see
   * RoboticsContext::RegisterControllerType).
   * @return This controller's registration type name. */
  static constexpr std::string_view TypeName() {
    return "MOCHI_ARTICULATED_POSE";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  using Params = ControllerMochiArticulatedPoseParams;
  using Obsv = ControllerMochiArticulatedPoseObsv;
  using Target = ControllerMochiArticulatedPoseTarget;

  /* @brief Construct an ArticulatedPoseController for the given articulated robot, optionally
   * holding a borrowed robot-model description (see GetBotPrefab).
   * @param prefab Optional borrowed BotPrefab model (may be null; must outlive the controller).
   * @param robot The articulated robot actor to control.
   * @param error Error status. */
  ControllerMochiArticulatedPose(BotPrefab const* prefab, Actor* robot, superdex::Error& error);

  /* This controller owns a singleton ArticulatedPoseController registration on the
   * actor (removed in the destructor), so it is neither copyable nor movable. */
  ControllerMochiArticulatedPose(ControllerMochiArticulatedPose const&) = delete;
  ControllerMochiArticulatedPose& operator=(ControllerMochiArticulatedPose const&) = delete;
  ControllerMochiArticulatedPose(ControllerMochiArticulatedPose&&) = delete;
  ControllerMochiArticulatedPose& operator=(ControllerMochiArticulatedPose&&) = delete;

  /* @brief Add a mochi ArticulatedPoseController to the actor using the default/current params.
   * Mochi ArticulatedPoseController is a singleton per articulation so only one controller can be
   * active at a time. Calling commands on multiple controllers at the same time will result in
   * overwriting commands. Must be called after SetParams and before ComputeOutput.
   * @param removeExisting If true, removes any existing ArticulatedPoseController on the actor
   * before adding a new one. If false, an existing controller causes an error.
   * @param error Error status. */
  void Initialize(bool removeExisting, superdex::Error& error);

  /* @brief Get controller parameters.
   * @return The params. */
  [[nodiscard]] Params const& GetParams() const;

  /* @brief Set controller parameters.
   * @param params Params.
   * @param error Error status. */
  void SetParams(Params const& params, superdex::Error& error);

  /* @brief Configure from a scene entry: load params (if @p paramArgs is non-empty), then add the
   * pose controller to the actor so it is ready to receive targets. Override of ControllerBase. */
  void ConfigureFromSceneEntry(
      std::string_view paramArgs,
      std::string_view initArgs,
      superdex::Error& error) override;

  /* @brief Forward targets to the internal pose controller.
   * @param obsv Unused (empty).
   * @param target Target poses: either worldFromRoot + localToParentTransforms (link transforms
   * path) or worldFromRoot + poseDofs (pose DOFs path). Exactly one must be provided.
   * @param error Error status.
   * @return Empty span — control is applied internally via the pose controller. */
  Span<real const> ComputeOutput(Obsv const& obsv, Target const& target, superdex::Error& error);

  /* @brief Reset internal controller state (e.g., between episodes). */
  void Reset() override;

 protected:
  Params _params;

  int _numLinks = 0;
  DynamicArray<int> _parentIndices;

  DynamicArray<TransformRT> _worldFromLinks;

  int _rootTransDofsCount = 0;
  int _rootRotDofsCount = 0;
  int _totalDofCount = 0;
  DynamicArray<real> _combinedDofs;

  bool _poseControllerAdded = false;
  bool _needsReset = true;
  bool _hasUserParams = false;

  ~ControllerMochiArticulatedPose() override;
};

} // namespace superdex::robotics
