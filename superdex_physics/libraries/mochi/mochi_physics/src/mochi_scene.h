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

#include "mochi_callback_set.h"
#include "mochi_common_components.h"
#include "mochi_ecs.h"
#include "mochi_scene_recorder.h"

#include <mochi_core/utils/guarded.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace mochi {

class DebugDrawInternal;
class ContextImpl;
struct BodyForceArgs;
class Shape;
class TetrahedralMeshShape;
class TriangularMeshShape;
class ArticulatedBodyShape;
class PolylineShape;

namespace dbg {
class SceneDebugger;
}

class SceneImpl final : public Scene {
 public:
  explicit SceneImpl(ContextImpl* context, std::string_view name, uint64_t uniqueId);
  ~SceneImpl() override;
  SceneImpl(SceneImpl const&) = delete;
  SceneImpl& operator=(SceneImpl const&) = delete;
  SceneImpl(SceneImpl&&) = delete;
  SceneImpl& operator=(SceneImpl&&) = delete;

  // Scene API:
  Context* GetContext() override;
  Context const* GetContext() const override;
  SceneHandle GetHandle() const override;
  char const* GetName() const override;
  Real3 GetGravity() const override;
  void SetGravity(Real3 const& gravityAccelWorld) override;
  SolverParams GetSolverParams() const override;
  void SetSolverParams(SolverParams const& params, Error& error) override;
  void Step(double timeStepSec) override;
  double GetLastTimeStep() const override;
  double GetTotalSimulationTime() const override;
  SolverStats GetSolverStats() const override;
  PerformanceStats GetPerformanceStats() const override;
  bool GetForceSingleIsland() const override;
  void SetForceSingleIsland(bool forceSingleIsland) override;
  StateHandle CaptureState(Error& error) override;
  void RestoreState(StateHandle handle, bool releaseImmediately, Error& error) override;
  void CaptureStateToBytes(DynamicArray<uint8_t>& outData, Error& error) override;
  void RestoreStateFromBytes(Span<uint8_t const> data, Error& error) override;
  void ReleaseState(StateHandle handle) override;
  void ReleaseAllStates() override;
  bool IsEqualState(StateHandle stateA, StateHandle stateB) const override;
  void CaptureStateToFile(std::string_view filePath, Error& error) override;
  Actor* CreateRigidActor(RigidActorParams const& params, Error& error) override;
  Actor* CreateSoftActor(SoftActorParams const& params, Error& error) override;
  Actor* CreateArticulatedActor(ArticulatedActorParams const& params, Error& error) override;
  Actor* CreateSoftSkinnedActor(SoftSkinnedActorParams const& params, Error& error) override;
  void DestroyActor(ActorHandle actor) override;
  Actor* GetActor(ActorHandle actor) override;
  Actor const* GetActor(ActorHandle actor) const override;
  int GetNumActors() const override;
  void GetActors(Span<Actor*> outActors, Error& error) override;
  void GetActors(Span<Actor const*> outActors, Error& error) const override;
  void ForEachActor(std::function<void(Actor*)> const& callback) override;
  Constraint* CreateRigidSphericalJointConstraint(
      RigidSphericalJointConstraintParams const& params,
      Error& error) override;
  void ForEachActor(std::function<void(Actor const*)> const& callback) const override;
  Constraint* CreateRigidPrismaticJointConstraint(
      RigidPrismaticJointConstraintParams const& params,
      Error& error) override;
  Constraint* CreateDeformableNodeToDeformableNodeConstraint(
      DeformableNodeToDeformableNodeConstraintParams const& params,
      Error& error) override;
  Constraint* CreateDeformableNodeToRigidConstraint(
      DeformableNodeToRigidConstraintParams const& params,
      Error& error) override;
  Constraint* CreateRodElementRotationToRigidConstraint(
      RodElementRotationToRigidConstraintParams const& params,
      Error& error) override;
  Constraint* CreateJointRotationRangeConstraint(
      JointRotationRangeConstraintParams const& params,
      Error& error) override;
  Constraint* CreateJointRotationTrackingConstraint(
      JointRotationTrackingConstraintParams const& params,
      Error& error) override;
  Constraint* CreateRigidPivotPositionConstraint(
      RigidPivotPositionConstraintParams const& params,
      Error& error) override;
  Constraint* CreateRigidPivotToRigidTargetConstraint(
      RigidPivotToRigidTargetConstraintParams const& params,
      Error& error) override;
  Constraint* CreateRigidPivotRotationConstraint(
      RigidPivotRotationConstraintParams const& params,
      Error& error) override;
  Constraint* CreateDeformableNodePositionConstraint(
      DeformableNodePositionConstraintParams const& params,
      Error& error) override;
  Constraint* CreateArticulatedSingleDofTargetConstraint(
      ArticulatedSingleDofTargetConstraintParams const& params,
      Error& error) override;
  Constraint* CreateArticulated3dRotationTargetConstraint(
      Articulated3dRotationTargetConstraintParams const& params,
      Error& error) override;
  Constraint* CreateArticulatedSingleDofRangeConstraint(
      ArticulatedSingleDofRangeConstraintParams const& params,
      Error& error) override;
  Constraint* CreateArticulated3dRotationRangeConstraint(
      Articulated3dRotationRangeConstraintParams const& params,
      Error& error) override;
  Constraint* GetConstraint(ConstraintHandle constraint) override;
  Constraint const* GetConstraint(ConstraintHandle constraint) const override;
  void DestroyConstraint(Constraint* constraint) override;
  void DestroyConstraint(ConstraintHandle constraint) override;
  int GetNumConstraints() const override;
  void ForEachConstraint(std::function<void(Constraint* constraint)> const& callback) override;
  void ForEachConstraint(
      std::function<void(Constraint const* constraint)> const& callback) const override;
  DebugDraw& GetDebugDraw() override;
  DebugDraw const& GetDebugDraw() const override;
  bool IsRecording() const override;
  void StartRecording(std::string_view filePath, RecordingParams const& params, Error& error)
      override;
  void StopRecording() override;
  void EnableLayerContactAsymmetric(
      std::string_view layerA,
      std::string_view layerB,
      bool enable,
      Error& error) override;
  void EnableLayerContactSymmetric(
      std::string_view layerA,
      std::string_view layerB,
      bool enable,
      Error& error) override;
  MOCHI_API bool IsLayerContactEnabled(std::string_view layerA, std::string_view layerB)
      const override;
  int GetNumContactLayers() const override;
  void EnumerateContactLayerNames(
      std::function<void(std::string_view name)> const& callback) const override;
  void EnableActorContactAsymmetric(
      ActorHandle colliding,
      ActorHandle collider,
      bool enable,
      IncludeNestedActors includeNestedActors,
      Error& error) override;
  void EnableActorContactSymmetric(
      ActorHandle actorA,
      ActorHandle actorB,
      bool enable,
      IncludeNestedActors includeNestedActors,
      Error& error) override;
  CallbackHandle RegisterPreStepCallback(
      std::string_view debugName,
      std::function<void(StepInfo const&)> callback,
      int priority) override; // thread-safe
  CallbackHandle RegisterPostStepCallback(
      std::string_view debugName,
      std::function<void(StepInfo const&)> callback,
      int priority) override; // thread-safe
  void CancelCallback(CallbackHandle handle) override; // thread-safe
  void UpdateDebugger() override;

  // For internal use only:
  void SetThreadAffinity();
  QueryHandle NewQueryHandle(QueryType type); // thread-safe
  void RegisterActorQuery(
      Actor* actor,
      QueryHandle preallocatedHandle,
      bool computeImmediately,
      Error& error);
  void ValidateNewActorComposition(entt::entity e) const;
  Actor* CreateShellActorImpl(
      experimental::ShellActorParams const& params,
      std::shared_ptr<TriangularMeshShape const> shapePtr,
      Error& error);
  Actor* CreateRodActorImpl(
      experimental::RodActorParams const& params,
      std::shared_ptr<PolylineShape const> shapePtr,
      Error& error);
  Span<uint8_t const> FindState(StateHandle handle, Error& error) const;
  void RestorePartialState(
      StateHandle handle,
      bool releaseImmediately,
      Span<SReflect::TypeId const> excludedAttributes,
      Error& error);
  [[nodiscard]] experimental::DebugStats GetDebugStats() const;
  void ApplyImprovedConvergenceSettings(bool logWarnings = false);
  void WarnIfNotImprovedConvergenceSettings() const;
  // Restore two captured states so that curr becomes the TimeStep::Current state and prev becomes
  // the TimeStep::Previous state, with constraint/controller targets set as they would be at the
  // beginning of a step. Unlike RestoreState (which restores a single post-step snapshot), this
  // reconstructs the full pre-step pair by restoring prev, advancing time via PreStepEcs, then
  // restoring curr. Used by BackPropagate and GetStepJacobian to evaluate residuals across two
  // consecutive states.
  void RestoreStatePair(StateHandle curr, StateHandle prev, Error& err);
  void ResetBackPropagation();
  void PrepareBackPropagate(StateHandle stateNew, StateHandle stateOld, Error& error);
  void BackPropagate(Error& error);
  void GetStepJacobian(
      StateHandle stateNew,
      StateHandle stateCurr,
      StateHandle stateOld,
      Span<real> outJacCurr,
      Span<real> outJacOld,
      Error& error);

  // For unit test access:
  entt::registry& GetRegistry() {
    return _registry;
  }
  entt::registry const& GetRegistry() const {
    return _registry;
  }

  /**
   * @brief Check if collision detection is enabled for an ordered pair of actor handles, according
   * to the actor-vs-actor contact filter.
   *
   * @warning This does NOT consider the layer-vs-layer contact filter.
   * @warning Order matters, as with @ref EnableActorContactAsymmetric.
   *
   * @note This method could be in the public API, but we are keeping it internal for now because of
   * the fear that people might mistake the meaning.
   *
   * @note Actor-vs-actor contact is enabled by default.
   *
   * @param[in] colliding ActorHandle of the "colliding" actor (the one checking for contact).
   * @param[in] collider ActorHandle of the "collider" actor (the one with a @ref ColliderType being
   * tested).
   * @param[in,out] error Fails if either actor handle is invalid. Check Error::IsOK() for success.
   *
   * @return True if contact is enabled for this ordered pair, according to actor-vs-actor
   * contact filtering.
   *
   * @see EnableActorContactAsymmetric, EnableActorContactSymmetric
   */
  [[nodiscard]] MOCHI_API bool
  IsActorContactEnabled(ActorHandle colliding, ActorHandle collider, Error& error) const;

  Actor* CreateSoftActorImpl(
      SoftActorParams const& params,
      experimental::ExperimentalSoftActorParams const& experimentalParams,
      bool isSkinned,
      std::shared_ptr<TetrahedralMeshShape const> shapePtr,
      Error& error);

  Actor* CreateSoftSkinnedActorImpl(
      SoftSkinnedActorParams const& params,
      experimental::ExperimentalSoftSkinnedActorParams const& experimentalParams,
      std::shared_ptr<ArticulatedBodyShape const> articulatedShapePtr,
      Error& error);

  // Called by the DebugServer to install a SceneDebugger, or clear via null pointer. Thread-safe.
  void SetDebugger(std::shared_ptr<dbg::SceneDebugger> debugger);

  // Return a shared_ptr to the current debugger (if any). For unit tests. Thread-safe.
  MOCHI_API std::shared_ptr<dbg::SceneDebugger> GetDebugger() const;

 private:
  // Private Members:
  ContextImpl* const _context;
  std::string const _name;
  uint64_t const _sceneId;
  entt::registry _registry;
  PerformanceStats _lastPerformanceStats;
  SolverStats _lastSolverStats;
  experimental::DebugStats _lastDebugStats;
  int _numActors = 0;
  int _numConstraints = 0;

  Constraint* CreateConstraintImpl(
      std::function<void(entt::registry&, entt::entity, Error&)> const& init,
      Error& error);
  Actor* CreateRigidActorImpl(
      RigidActorParams const& params,
      bool isArticulatedLink,
      bool useContact,
      std::shared_ptr<Shape const> shapePtr,
      Error& error);
  Actor* CreateArticulatedActorImpl(
      ArticulatedActorParams const& params,
      std::shared_ptr<ArticulatedBodyShape const> shapePtr,
      Error& error);
  Actor* CreateArticulatedActorImpl(
      ArticulatedActorParams const& params,
      bool useContact,
      std::shared_ptr<ArticulatedBodyShape const> shapePtr,
      Span<ActorHandle const> links,
      std::shared_ptr<Shape const> skinShape,
      Error& error);
  void CreateArticulatedLinkActorsImpl(
      std::string_view parentActorName,
      Span<ArticulatedLinkParams const> params,
      bool useContact,
      std::shared_ptr<ArticulatedBodyShape const> shapePtr,
      TransformRT const& rootTransform,
      Span<ActorHandle> outLinks,
      Span<ShapeHandle> outLinkShapes,
      Error& error);
  void CreateArticulatedActorJointLimitsImpl(
      entt::entity e,
      ArticulatedActorParams const& params,
      std::shared_ptr<ArticulatedBodyShape const> shape,
      Error& error);
  void CreateArticulatedActorCycleJointsImpl(
      ArticulatedActorParams const& params,
      std::shared_ptr<ArticulatedBodyShape const> shape,
      entt::entity articulatedActorEntity,
      Span<ActorHandle const> links,
      Error& error);

  enum CallbackType : uint8_t { PreStep, PostStep };
  CallbackHandle GenerateNewCallbackHandle(CallbackType type);

  // Extract the CallbackType from a CallbackHandle.
  static CallbackType GetCallbackType(CallbackHandle handle);

  // Some bytes of a scene's ID are packed into the CallbackHandle.
  // Return true if those bytes match this scene's ID.
  bool IsProbablyMyCallbackHandle(CallbackHandle handle) const;

  // Destroy all actors and constraints in an articulated actor.
  void DestroyAllItemsInArticulatedActor(entt::entity e);

  // Callbacks:
  uint32_t _nextCallbackId = 1;
  CallbackSet<void(StepInfo const&)> _preStepCallbacks;
  CallbackSet<void(StepInfo const&)> _postStepCallbacks;

  // Debugging:
  std::unique_ptr<DebugDrawInternal> _debugDraw; // ptr so we don't have to include the header here
  struct DebuggerInfo {
    std::shared_ptr<dbg::SceneDebugger> debugger;
    DynamicArray<std::shared_ptr<dbg::SceneDebugger>> pendingShutdown;
  };
  Guarded<DebuggerInfo> _debugger;

  // Recording:
  std::unique_ptr<SceneRecorder> _recorder;

  // Captured State:
  uint64_t _nextStateHandleValue = 1;
  size_t _prevStateBufferSize = 0;
  DynamicArray<uint8_t> _stateBufferForRecycling;
  std::unordered_map<StateHandle, DynamicArray<uint8_t>> _capturedState;
};

inline Real3 SceneImpl::GetGravity() const {
  return ToReal3(_registry.ctx<CSceneGravity>().accel);
}

/// @brief Create a new scene containing only a clone of the given actor.
///
/// @details The actor is exported to a temporary prefab and re-imported into a fresh scene
/// that inherits the original scene's gravity and solver parameters. This is useful when a
/// computation requires an isolated actor without interference from other actors in the original
/// scene. Currently only articulated actors are supported.
///
/// @param[in]  actor   The actor to clone. Must be an articulated actor.
/// @param[in]  context The context that will own the new scene.
/// @param[out] error   Error status.
/// @return Pointer to the cloned actor in the newly created scene, or null on error.
Actor* CopyActorToSingletonScene(Actor const* actor, Context* context, Error& error);

// Internal implementation of the API function MakeSceneDifferentiable.
void MakeSceneDifferentiableInternal(Scene* scene, Error& error);
} // namespace mochi
