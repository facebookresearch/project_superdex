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

#include <mochi_physics/mochi_physics_experimental.h>

#include <unordered_map>

// Forwards
namespace mochi {
class Actor;
class Context;
class Scene;
class SceneImpl;
class DebugDrawInternal;
class DebugDrawCollector;

class IKSolverImpl final : public experimental::IKSolver {
 public:
  IKSolverImpl(Scene* scene, Error& error);

  ~IKSolverImpl() override;

  void SetSolverParams(experimental::IKSolverParams const& ikParams) override;
  experimental::IKSolverParams GetSolverParams() const override;

  Constraint* CreatePositionTarget(
      ActorHandle actor,
      Real3 localPosition,
      Real3 targetPosition,
      real weight,
      Error& error) override;
  void ClearPositionTarget(ActorHandle actor, Error& error) override;

  Constraint* CreateRotationTarget(
      ActorHandle actor,
      Real3 localRotation,
      Real3 targetRotation,
      real weight,
      Error& error) override;
  void ClearRotationTarget(ActorHandle actor, Error& error) override;

  bool SolveIK(Error& error) override;

 private:
  // The scene will be invoked with infinity timestep size, which will pollute many of the data
  // structures there. Therefore, this scene is designed to be used internally for solving IK only,
  // not for general simulation.
  SceneImpl* _scene = nullptr;

  // The set of end-effector position targets
  std::unordered_map<ActorHandle, Constraint*> _positionTargets;

  // The set of end-effector rotation targets
  std::unordered_map<ActorHandle, Constraint*> _rotationTargets;

  // The position and rotation error thresholds for deciding whether the IK targets are reachable.
  real _positionErrorThres = 1e-2_r;
  real _rotationErrorThres = 1e-2_r;
};
} // namespace mochi
