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

#include "core/settings.h"
#include "simulation/mochi_async_scene.h"

#include <mochi_core/utils/coordinate_space_converter.h>

#include <mochi_physics/mochi_physics.h>

#include <mochi_renderer/buffer.h>

#include <math/vec3.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mochi_renderer {
class SceneObject;
class DebugDraw;
} // namespace mochi_renderer

namespace superdex::studio {

struct SceneStage;
class Viewport;
class DebugText;

// Interactive mouse force-drag for the physics editors: while simulating, a grab springs a point on
// a rigid body / articulated link (a @ref mochi::RigidPivotPositionConstraint) or a cluster of
// soft-body nodes (@ref mochi::DeformableNodePositionConstraint) toward a moving world-space
// target, removed on release.
//
// One controller is bound to one simulation session (constructed on start, destroyed on stop), so
// there is no cross-session state to reset.
//
// Threading: @ref mochi::Scene access happens on the sim thread in a pre-step callback; the UI
// thread only writes the latest target/actor under a mutex. Points arrive in Filament space and are
// converted to Mochi space here.
class PhysicsDragController {
 public:
  // Binds to @p mochiScene's current session and to @p stage (which maps a picked object to an
  // actor name). @p settings is snapshotted for this session's tuning, clamped to a usable range
  // (it is persisted and hand-editable). @p rendererToEditor converts the viewport's Filament-space
  // points into Mochi space. Must be constructed while simulating; builds its own
  // name -> ActorHandle table.
  PhysicsDragController(
      MochiAsyncScene& mochiScene,
      SceneStage const* stage,
      PhysicsDragSettings settings,
      mochi::CoordinateSpaceConverter rendererToEditor);
  ~PhysicsDragController();

  PhysicsDragController(PhysicsDragController const&) = delete;
  PhysicsDragController& operator=(PhysicsDragController const&) = delete;
  PhysicsDragController(PhysicsDragController&&) = delete;
  PhysicsDragController& operator=(PhysicsDragController&&) = delete;

  // UI-thread entry points wired to the Viewport's onSceneObjectDrag* hooks (points in Filament
  // space). BeginDrag returns true if the picked object mapped to a grabbable actor.
  bool BeginDrag(mochi_renderer::SceneObject* object, filament::math::float3 filamentHitPoint);
  void UpdateDrag(filament::math::float3 filamentTargetPoint);
  void EndDrag();

  // Draws the active grab (spheres at the grab point and target, a line between, and a label with
  // the grab force at the target). Call once per frame from OnRender before RenderScene; no-op when
  // not dragging. @p converter maps Mochi -> Filament.
  void DrawDebug(
      mochi_renderer::DebugDraw* debugDraw,
      DebugText* debugText,
      mochi::CoordinateSpaceConverter const& converter) const;

  // Replaces the session's tuning while simulating (UI thread). Values are sanitized as at
  // construction. Most gains are baked into the constraints when a grab is set up, so if one of
  // those changed during an active grab the grab is invalidated and rebuilt on the next step.
  void SetSettings(PhysicsDragSettings settings);

 private:
  // Fills _nameToHandle with every grabbable actor (dynamic rigid bodies and articulated links) by
  // name. Runs once on the sim thread at construction.
  void BuildActorMap(mochi::Scene* scene);

  // Maps a picked object to a grabbable actor (object -> stage name -> handle), or nullopt. UI
  // thread.
  std::optional<mochi::ActorHandle> ResolveActor(mochi_renderer::SceneObject* object) const;

  // Per-step reconciliation of the UI-requested drag state with the live constraint. Sim thread.
  void OnPreStep(mochi::StepInfo const& info);

  mochi::Real3 FilamentToMochi(filament::math::float3 point) const;

  MochiAsyncScene* _mochiScene = nullptr;
  // The session's async scene; used to check the session is still live before cancelling in the
  // dtor.
  mochi::AsyncScene* _sessionScene = nullptr;
  // Maps a picked render object to a physics-actor name (object -> StagedActor.name). Non-owning.
  SceneStage const* _stage = nullptr;
  // Converts viewport (Filament-space) points into Mochi space; injected so it cannot drift from
  // the app's space definitions.
  mochi::CoordinateSpaceConverter _rendererToEditor;
  // name -> grabbable ActorHandle; built once at construction, read-only (and UI-readable) after.
  std::unordered_map<std::string, mochi::ActorHandle> _nameToHandle;

  // --- UI-thread-only state ---
  bool _dragging = false;
  mochi::CallbackHandle _preStepCb = {};

  // --- Shared UI -> sim state (guarded by _mutex) ---
  mutable std::mutex _mutex;
  // Tuning, clamped to a usable range (see PhysicsDragSettings). Seeded at construction and
  // replaceable via SetSettings, so it is read under the lock: the sim thread snapshots it once per
  // step, and DrawDebug (UI thread) reads the gizmo radius.
  PhysicsDragSettings _settings;
  bool _active = false;
  mochi::ActorHandle _actor = {};
  mochi::Real3 _targetMochi = {};
  // Bumped on every BeginDrag, and on a settings change that invalidates a live grab, so the sim
  // side rebuilds the constraint.
  uint64_t _generation = 0;

  // --- Sim-thread-only state (touched only inside OnPreStep) ---
  mochi::ConstraintHandle _simConstraint = {};
  bool _simConstraintValid = false;
  // Optional companion rotation constraint pinning the grab-time orientation.
  mochi::ConstraintHandle _simRotConstraint = {};
  bool _simRotConstraintValid = false;
  // True when the current grab is on a static body: no constraints are created (they would error)
  // and the gizmo is drawn gray.
  bool _simStatic = false;
  // Set when constraint creation failed for _simCreatedGeneration. Latches the failure so the grab
  // is not torn down and rebuilt (a whole-scene mass scan, two passes over the soft nodes, and the
  // same error log) on every step for as long as the button is held.
  bool _simSetupFailed = false;
  // True when the current grab is on a soft (deformable) actor; the spring targets a cluster of
  // nodes near the pick.
  bool _simSoft = false;
  // The soft-body anchor node (nearest to the pick); drives the leash and debug gizmo. Valid only
  // when _simSoft.
  int _simNodeIndex = -1;
  // One position constraint per grabbed soft node, each offset from the anchor so the cluster
  // translates rigidly with the drag (avoids pinching a single vertex).
  struct SoftNodeConstraint {
    mochi::ConstraintHandle handle;
    mochi::Real3 offset; // node's world offset from the anchor node at grab time
  };
  std::vector<SoftNodeConstraint> _simSoftConstraints;
  uint64_t _simCreatedGeneration = 0;
  // Local-frame pivot anchor, used to recompute the anchor's world position for the debug viz.
  mochi::Real3 _simLocalPosition = {};
  // The rate-limited, leashed "virtual hand" position the spring targets (chases, not snaps to, the
  // cursor).
  mochi::Real3 _simTarget = {};

  // Sim thread publishes the current anchor/target each step; DrawDebug (UI thread) consumes them.
  struct DragDebugData {
    bool active = false;
    // False when the grabbed body is static (not draggable) — DrawDebug renders it gray.
    bool grabbable = true;
    mochi::Real3 anchor = {};
    mochi::Real3 target = {};
    // Magnitude of the net translational force the grab constraint(s) applied last step [N]. Zero
    // until a step has completed with the ConstraintForce query registered.
    mochi::real forceNewtons = 0;
  };
  mutable mochi_renderer::ProducerConsumerBuffer<DragDebugData> _debugData;
};

// Routes @p viewport's scene-object drag hooks to the controller held in @p controller. The slot is
// read through its address on each event, so the hooks stay correct across the controller's
// per-session create/destroy cycle. Both arguments must outlive @p viewport's hooks.
void BindSceneObjectDragHooks(
    Viewport& viewport,
    std::unique_ptr<PhysicsDragController> const& controller);

} // namespace superdex::studio
