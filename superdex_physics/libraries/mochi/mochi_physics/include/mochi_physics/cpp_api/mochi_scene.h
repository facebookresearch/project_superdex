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

#include <mochi_physics/cpp_api/mochi_actor.h>
#include <mochi_physics/cpp_api/mochi_handle.h>
#include <mochi_physics/cpp_api/mochi_structs.h>

/********************************************************************************
 IMPORTANT: PLEASE KEEP HEADER INCLUDES TO A MINIMUM.
    If you must include a mochi_core header, then please make sure that it only
    declares the data types (not containing other implementation details).
*********************************************************************************/
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <cstdint>
#include <functional>
#include <string_view>

namespace mochi {

class Constraint;
class DebugDraw;
class Context;

inline constexpr Real3 kDefaultGravity = {0_r, -9.8_r, 0_r};

class Scene {
 public:
  [[nodiscard]] virtual Context* GetContext() = 0;

  [[nodiscard]] virtual Context const* GetContext() const = 0;

  virtual SceneHandle GetHandle() const = 0;

  [[nodiscard]] virtual char const* GetName() const = 0;

  [[nodiscard]] virtual Real3 GetGravity() const = 0;

  virtual void SetGravity(Real3 const& gravity) = 0;

  [[nodiscard]] virtual SolverParams GetSolverParams() const = 0;

  virtual void SetSolverParams(SolverParams const& params, Error& error) = 0;

  virtual void Step(double timeStepSec) = 0;

  [[nodiscard]] virtual double GetLastTimeStep() const = 0;

  [[nodiscard]] virtual double GetTotalSimulationTime() const = 0;

  [[nodiscard]] virtual SolverStats GetSolverStats() const = 0;

  [[nodiscard]] virtual PerformanceStats GetPerformanceStats() const = 0;

  virtual void SetForceSingleIsland(bool forceSingleIsland) = 0;

  [[nodiscard]] virtual bool GetForceSingleIsland() const = 0;

  // State Snapshots *******************************************************************************

  [[nodiscard]] virtual StateHandle CaptureState(Error& error) = 0;

  virtual void RestoreState(StateHandle handle, bool releaseImmediately, Error& error) = 0;

  virtual void CaptureStateToBytes(DynamicArray<uint8_t>& outData, Error& error) = 0;

  virtual void RestoreStateFromBytes(Span<uint8_t const> data, Error& error) = 0;

  virtual void ReleaseState(StateHandle handle) = 0;

  virtual void ReleaseAllStates() = 0;

  [[nodiscard]] virtual bool IsEqualState(StateHandle stateA, StateHandle stateB) const = 0;

  virtual void CaptureStateToFile(std::string_view filePath, Error& error) = 0;

  // Actors ****************************************************************************************

  virtual Actor* CreateRigidActor(RigidActorParams const& params, Error& error) = 0;

  virtual Actor* CreateSoftActor(SoftActorParams const& params, Error& error) = 0;

  virtual Actor* CreateArticulatedActor(ArticulatedActorParams const& params, Error& error) = 0;

  virtual Actor* CreateSoftSkinnedActor(SoftSkinnedActorParams const& params, Error& error) = 0;

  virtual void DestroyActor(ActorHandle actor) = 0;

  void DestroyActor(Actor* actor);

  [[nodiscard]] virtual Actor* GetActor(ActorHandle actor) = 0;

  [[nodiscard]] virtual Actor const* GetActor(ActorHandle actor) const = 0;

  [[nodiscard]] virtual int GetNumActors() const = 0;

  virtual void GetActors(Span<Actor*> outActors, Error& error) = 0;

  virtual void GetActors(Span<Actor const*> outActors, Error& error) const = 0;

  virtual void ForEachActor(std::function<void(Actor*)> const& callback) = 0;

  virtual void ForEachActor(std::function<void(Actor const*)> const& callback) const = 0;

  // Constraints ***********************************************************************************

  virtual Constraint* CreateArticulatedSingleDofRangeConstraint(
      ArticulatedSingleDofRangeConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateArticulated3dRotationRangeConstraint(
      Articulated3dRotationRangeConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateArticulatedSingleDofTargetConstraint(
      ArticulatedSingleDofTargetConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateArticulated3dRotationTargetConstraint(
      Articulated3dRotationTargetConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateJointRotationRangeConstraint(
      JointRotationRangeConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateJointRotationTrackingConstraint(
      JointRotationTrackingConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateRodElementRotationToRigidConstraint(
      RodElementRotationToRigidConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateRigidPivotPositionConstraint(
      RigidPivotPositionConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateRigidPivotToRigidTargetConstraint(
      RigidPivotToRigidTargetConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateRigidPivotRotationConstraint(
      RigidPivotRotationConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateRigidPrismaticJointConstraint(
      RigidPrismaticJointConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateRigidSphericalJointConstraint(
      RigidSphericalJointConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateDeformableNodePositionConstraint(
      DeformableNodePositionConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateDeformableNodeToDeformableNodeConstraint(
      DeformableNodeToDeformableNodeConstraintParams const& params,
      Error& error) = 0;

  virtual Constraint* CreateDeformableNodeToRigidConstraint(
      DeformableNodeToRigidConstraintParams const& params,
      Error& error) = 0;

  [[nodiscard]] virtual Constraint* GetConstraint(ConstraintHandle constraint) = 0;

  [[nodiscard]] virtual Constraint const* GetConstraint(ConstraintHandle constraint) const = 0;

  virtual void DestroyConstraint(Constraint* constraint) = 0;

  virtual void DestroyConstraint(ConstraintHandle constraint) = 0;

  [[nodiscard]] virtual int GetNumConstraints() const = 0;

  virtual void ForEachConstraint(std::function<void(Constraint*)> const& callback) = 0;

  virtual void ForEachConstraint(std::function<void(Constraint const*)> const& callback) const = 0;

  // Debugging ***********************************************************************************

  [[nodiscard]] virtual DebugDraw& GetDebugDraw() = 0;

  [[nodiscard]] virtual DebugDraw const& GetDebugDraw() const = 0;

  virtual void UpdateDebugger() = 0;

  // Recording ***********************************************************************************

  [[nodiscard]] virtual bool IsRecording() const = 0;

  virtual void
  StartRecording(std::string_view filePath, RecordingParams const& params, Error& error) = 0;

  virtual void StopRecording() = 0;

  // Contact Layers *****************************************************************************

  virtual void EnableLayerContactAsymmetric(
      std::string_view layerA,
      std::string_view layerB,
      bool enable,
      Error& error) = 0;

  virtual void EnableLayerContactSymmetric(
      std::string_view layerA,
      std::string_view layerB,
      bool enable,
      Error& error) = 0;

  [[nodiscard]] virtual bool IsLayerContactEnabled(std::string_view layerA, std::string_view layerB)
      const = 0;

  [[nodiscard]] virtual int GetNumContactLayers() const = 0;

  virtual void EnumerateContactLayerNames(
      std::function<void(std::string_view name)> const& callback) const = 0;

  virtual void EnableActorContactAsymmetric(
      ActorHandle colliding,
      ActorHandle collider,
      bool enable,
      IncludeNestedActors includeNestedActors,
      Error& error) = 0;

  virtual void EnableActorContactSymmetric(
      ActorHandle actorA,
      ActorHandle actorB,
      bool enable,
      IncludeNestedActors includeNestedActors,
      Error& error) = 0;

  // Callbacks **********************************************************************************

  static constexpr int kDefaultCallbackPriority = 100;

  [[nodiscard]] virtual CallbackHandle RegisterPreStepCallback(
      std::string_view debugName,
      std::function<void(StepInfo const&)> callback,
      int priority = kDefaultCallbackPriority) = 0;

  [[nodiscard]] virtual CallbackHandle RegisterPostStepCallback(
      std::string_view debugName,
      std::function<void(StepInfo const&)> callback,
      int priority = kDefaultCallbackPriority) = 0;

  virtual void CancelCallback(CallbackHandle handle) = 0;

 protected:
  /// Don't delete the Scene pointer. Call @ref Context::DestroyScene.
  virtual ~Scene() = default;
};

} // namespace mochi

#include "mochi_scene_inl.h"
