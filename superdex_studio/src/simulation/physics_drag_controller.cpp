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

#include "simulation/physics_drag_controller.h"

#include "rendering/debug_text.h"
#include "rendering/scene_stage.h"
#include "rendering/viewport.h"

#include <mochi_renderer/debug.h>
#include <mochi_renderer/type_conversions.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/math_utils.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

namespace superdex::studio {

using mochi::operator""_r; // real-valued literals (0_r, 2_r, ...)

// Grab tuning now lives in @ref PhysicsDragSettings (snapshotted per session as _settings). The
// gains are frequency-based: stiffness = m*omega^2, damping = 2*zeta*m*omega, omega = 2*pi*f, so
// the feel is mass-independent and heavy bodies still move. Mochi's saturation is a DISTANCE, so
// saturation = a_max/omega^2 caps the elastic force at m*a_max and the grab yields to collisions.
static constexpr mochi::real kTwoPi = 2_r * mochi::kPI;

// Match the viewport's selection/hover highlight orange (opaque), shared by both spheres and the
// connecting line.
static constexpr filament::math::float4 kDebugColor{
    kViewportSelectionColor[0],
    kViewportSelectionColor[1],
    kViewportSelectionColor[2],
    1.0f};
// Gray gizmo shown when the grabbed body is static (not draggable).
static constexpr filament::math::float4 kDebugGrayColor{0.5f, 0.5f, 0.5f, 1.0f};
// Screen-space nudge that lifts the force label clear of the target sphere.
static constexpr ImVec2 kForceLabelOffset{0.0f, -18.0f};

// Returns @p v scaled so its magnitude is at most @p maxLen.
static mochi::Real3 ClampMagnitude(mochi::Real3 const& v, mochi::real maxLen) {
  mochi::real const len = mochi::Norm(v);
  if (len > maxLen && len > 0_r) {
    return v * (maxLen / len);
  }
  return v;
}

// PhysicsDragSettings is reflected and persisted, so a hand-edited settings file can reach values
// that break the grab: a non-positive node cap makes resize() allocate a huge buffer or divides by
// zero when splitting mass, and a non-positive or non-finite response frequency makes omega^2 zero
// so saturation (a_max/omega^2) is non-finite and the solver rejects the constraint on every step.
// Clamp the session snapshot once here so the per-step path can trust it.
static PhysicsDragSettings SanitizeSettings(PhysicsDragSettings settings) {
  PhysicsDragSettings const defaults;
  auto positive = [](float value, float fallback) {
    return (mochi::IsFinite(value) && value > 0.0f) ? value : fallback;
  };
  auto nonNegative = [](float value, float fallback) {
    return (mochi::IsFinite(value) && value >= 0.0f) ? value : fallback;
  };
  settings.responseFrequencyHz =
      positive(settings.responseFrequencyHz, defaults.responseFrequencyHz);
  settings.rotResponseFrequencyHz =
      positive(settings.rotResponseFrequencyHz, defaults.rotResponseFrequencyHz);
  settings.dampingRatio = nonNegative(settings.dampingRatio, defaults.dampingRatio);
  settings.rotDampingRatio = nonNegative(settings.rotDampingRatio, defaults.rotDampingRatio);
  settings.maxAcceleration = nonNegative(settings.maxAcceleration, defaults.maxAcceleration);
  settings.maxAngularAcceleration =
      nonNegative(settings.maxAngularAcceleration, defaults.maxAngularAcceleration);
  settings.maxTargetSpeed = nonNegative(settings.maxTargetSpeed, defaults.maxTargetSpeed);
  settings.maxStretch = nonNegative(settings.maxStretch, defaults.maxStretch);
  settings.softGrabRadiusFraction =
      nonNegative(settings.softGrabRadiusFraction, defaults.softGrabRadiusFraction);
  settings.debugSphereRadius = nonNegative(settings.debugSphereRadius, defaults.debugSphereRadius);
  settings.softGrabMaxNodes = mochi::Max(1, settings.softGrabMaxNodes);
  return settings;
}

// Mass the grab must accelerate at @p grabbed: the body's own @p ownMass, or for an articulated
// link the whole articulation's mass so a light distal link can still move the chain behind it.
// Runs on the sim thread.
static mochi::real
EffectiveGrabMass(mochi::Scene* scene, mochi::ActorHandle grabbed, mochi::real ownMass) {
  mochi::real effective = ownMass;
  scene->ForEachActor([&](mochi::Actor* actor) {
    if (actor->GetType() != mochi::ActorType::Articulated) {
      return;
    }
    mochi::Error error;
    mochi::Span<mochi::ActorHandle const> const links = actor->GetNestedLinkActors(error);
    if (!error.IsOK()) {
      return;
    }
    bool containsGrabbed = false;
    mochi::real total = 0_r;
    for (mochi::ActorHandle const& link : links) {
      if (link == grabbed) {
        containsGrabbed = true;
      }
      if (mochi::Actor* linkActor = scene->GetActor(link)) {
        mochi::Error massError;
        mochi::real const m = linkActor->GetMass(massError);
        if (massError.IsOK() && m > 0_r) {
          total += m;
        }
      }
    }
    if (containsGrabbed && total > 0_r) {
      effective = total;
    }
  });
  return effective;
}

PhysicsDragController::PhysicsDragController(
    MochiAsyncScene& mochiScene,
    SceneStage const* stage,
    PhysicsDragSettings settings,
    mochi::CoordinateSpaceConverter rendererToEditor)
    : _mochiScene(&mochiScene),
      _sessionScene(mochiScene.GetAsyncScene()),
      _stage(stage),
      _rendererToEditor(rendererToEditor),
      _settings(SanitizeSettings(settings)) {
  if (_sessionScene == nullptr) {
    return; // not simulating; nothing to bind to
  }
  // Build the name -> handle table on the sim thread (it needs a Scene*), synchronized before
  // returning so the UI thread can read it safely for the rest of the session.
  _sessionScene->QueueCommand([this](mochi::Scene* scene) { BuildActorMap(scene); });
  _sessionScene->WaitForQueuedCommands();
  _preStepCb = _sessionScene->RegisterPreStepCallback(
      "PhysicsDragController", [this](mochi::StepInfo const& info) { OnPreStep(info); });
}

PhysicsDragController::~PhysicsDragController() {
  // Cancel our callback only if this controller's session is still live (a stopped/restarted
  // session makes GetAsyncScene() differ from _sessionScene). A controller lives for one session,
  // so this can't false-cancel a different session's callback.
  if (_sessionScene != nullptr && _mochiScene->GetAsyncScene() == _sessionScene) {
    _sessionScene->CancelCallback(_preStepCb);
  }
}

void PhysicsDragController::BuildActorMap(mochi::Scene* scene) {
  _nameToHandle.clear();
  scene->ForEachActor([this, scene](mochi::Actor* actor) {
    mochi::ActorType const type = actor->GetType();
    if (type == mochi::ActorType::Rigid) {
      // Include static bodies too: the sim side detects them and shows a gray, non-draggable gizmo
      // (rather than erroring on a constraint).
      if (char const* name = actor->GetName()) {
        _nameToHandle[name] = actor->GetHandle();
      }
    } else if (type == mochi::ActorType::Soft) {
      // Soft bodies are grabbable via a single-node constraint (chosen at grab time). Register a
      // NodePositions query so GetNodePositionsLocal() (used to find the nearest node and the
      // anchor) returns data during simulation; it's freed when the scene is destroyed on stop.
      mochi::Error queryError;
      actor->RegisterQuery(mochi::QueryType::NodePositions, queryError);
      if (char const* name = actor->GetName()) {
        _nameToHandle[name] = actor->GetHandle();
      }
    } else if (type == mochi::ActorType::Articulated) {
      // Each link carries rigid state and is individually grabbable; the articulated root is not.
      mochi::Error error;
      mochi::Span<mochi::ActorHandle const> const links = actor->GetNestedLinkActors(error);
      if (!error.IsOK()) {
        return;
      }
      for (mochi::ActorHandle const& link : links) {
        if (mochi::Actor* linkActor = scene->GetActor(link)) {
          if (char const* name = linkActor->GetName()) {
            _nameToHandle[name] = link;
          }
        }
      }
    }
  });
}

std::optional<mochi::ActorHandle> PhysicsDragController::ResolveActor(
    mochi_renderer::SceneObject* object) const {
  if (object == nullptr || _stage == nullptr) {
    return std::nullopt;
  }
  int const idx = _stage->GetSceneObjectIndex(object);
  if (idx < 0 || idx >= _stage->GetNumActors()) {
    return std::nullopt;
  }
  auto const it = _nameToHandle.find(_stage->GetActors()[idx].name);
  if (it == _nameToHandle.end()) {
    return std::nullopt;
  }
  return it->second;
}

mochi::Real3 PhysicsDragController::FilamentToMochi(filament::math::float3 point) const {
  return _rendererToEditor.TranslationToOutput(mochi_renderer::ToMochi<mochi::real>(point));
}

bool PhysicsDragController::BeginDrag(
    mochi_renderer::SceneObject* object,
    filament::math::float3 filamentHitPoint) {
  std::optional<mochi::ActorHandle> const handle = ResolveActor(object);
  if (!handle.has_value()) {
    return false; // not a grabbable object
  }
  mochi::Real3 const target = FilamentToMochi(filamentHitPoint);
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _actor = *handle;
    _targetMochi = target;
    _active = true;
    ++_generation;
  }
  _dragging = true;
  return true;
}

void PhysicsDragController::UpdateDrag(filament::math::float3 filamentTargetPoint) {
  if (!_dragging) {
    return;
  }
  mochi::Real3 const target = FilamentToMochi(filamentTargetPoint);
  std::lock_guard<std::mutex> lock(_mutex);
  _targetMochi = target;
}

void PhysicsDragController::EndDrag() {
  if (!_dragging) {
    return;
  }
  _dragging = false;
  // Leave the pre-step callback registered for the session; on the next step it observes the
  // cleared active flag and destroys the constraint.
  std::lock_guard<std::mutex> lock(_mutex);
  _active = false;
}

void PhysicsDragController::OnPreStep(mochi::StepInfo const& info) {
  bool active = false;
  mochi::ActorHandle actor = {};
  mochi::Real3 target = {};
  uint64_t generation = 0;
  PhysicsDragSettings settings;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    active = _active;
    actor = _actor;
    target = _targetMochi;
    generation = _generation;
    settings = _settings;
  }

  mochi::Scene* scene = info.scene;

  auto destroyConstraints = [this, scene]() {
    if (_simConstraintValid) {
      scene->DestroyConstraint(_simConstraint);
      _simConstraint = {};
      _simConstraintValid = false;
    }
    if (_simRotConstraintValid) {
      scene->DestroyConstraint(_simRotConstraint);
      _simRotConstraint = {};
      _simRotConstraintValid = false;
    }
    for (SoftNodeConstraint const& c : _simSoftConstraints) {
      scene->DestroyConstraint(c.handle);
    }
    _simSoftConstraints.clear();
  };
  auto publishDebug = [this](
                          bool isActive,
                          bool grabbable,
                          mochi::Real3 const& anchor,
                          mochi::Real3 const& tgt,
                          mochi::real forceNewtons) {
    DragDebugData& data = _debugData.GetProducerData();
    data.active = isActive;
    data.grabbable = grabbable;
    data.anchor = anchor;
    data.target = tgt;
    data.forceNewtons = forceNewtons;
    _debugData.Produce();
  };
  // Magnitude of the net translational force the grab applied during the last step. GetForce
  // errors until a step has completed with the query registered, so the first step of every grab
  // legitimately has no data and reads zero -- swallow that rather than logging it per grab.
  auto currentGrabForce = [this, scene]() -> mochi::real {
    auto translationalForce = [](mochi::Constraint const* constraint) -> mochi::Real3 {
      mochi::Error error;
      // RigidPivotPosition reports 6 values (force [N] then torque [N.m]);
      // DeformableNodePosition reports 3 (force only). Either way the first 3 are the force.
      mochi::Span<mochi::real const> const force = constraint->GetForce(error);
      if (!error.IsOK() || force.size() < 3) {
        return {};
      }
      return mochi::Real3{force[0], force[1], force[2]};
    };
    if (_simSoft) {
      // Vector-sum the cluster: the per-node mass split was designed to conserve exactly this net
      // force, so summing (rather than adding magnitudes) keeps it comparable to a rigid grab.
      mochi::Real3 total{};
      for (SoftNodeConstraint const& c : _simSoftConstraints) {
        if (mochi::Constraint const* constraint = scene->GetConstraint(c.handle)) {
          total = total + translationalForce(constraint);
        }
      }
      return mochi::Norm(total);
    }
    if (mochi::Constraint const* constraint = scene->GetConstraint(_simConstraint)) {
      return mochi::Norm(translationalForce(constraint));
    }
    return 0_r;
  };
  // Current world position of the grab point: the tracked node for soft bodies, else the fixed
  // local pivot on the (rigid/static) body.
  auto currentAnchorWorld = [this](mochi::Actor* grabbed) -> mochi::Real3 {
    mochi::TransformRT const worldFromLocal = grabbed->GetRootTransform();
    if (_simSoft) {
      mochi::Error error;
      mochi::Span<mochi::real const> const local = grabbed->GetNodePositionsLocal(error);
      int const base = _simNodeIndex * 3;
      if (error.IsOK() && _simNodeIndex >= 0 && base + 2 < mochi::isize(local)) {
        return worldFromLocal.TransformPoint(
            mochi::Real3{local[base], local[base + 1], local[base + 2]});
      }
      return _simTarget; // node unavailable; fall back to the target
    }
    return worldFromLocal.TransformPoint(_simLocalPosition);
  };

  if (!active) {
    destroyConstraints();
    _simStatic = false;
    _simSoft = false;
    _simSetupFailed = false;
    publishDebug(false, true, {}, {}, 0_r);
    return;
  }

  // A grab is "set up" for this generation once its constraint(s) exist, it was found to be static
  // (which needs none), or creation already failed for it. Failures latch: without that, every step
  // would tear the grab down and rebuild it — a whole-scene mass scan, two passes over the soft
  // nodes, and the same error log at step rate — for as long as the button is held.
  bool const setUp = _simCreatedGeneration == generation &&
      (_simConstraintValid || _simStatic || !_simSoftConstraints.empty() || _simSetupFailed);
  if (!setUp) {
    destroyConstraints();
    _simStatic = false;
    _simSoft = false;
    _simSetupFailed = false;
    _simNodeIndex = -1;
    mochi::Actor* grabbed = scene->GetActor(actor);
    if (grabbed == nullptr) {
      publishDebug(
          false, true, {}, {}, 0_r); // actor gone (e.g. scene rebuilt); wait for a fresh grab
      return;
    }
    mochi::TransformRT const worldFromLocal = grabbed->GetRootTransform();
    _simTarget = target; // virtual hand starts at the grab point (no initial lag)
    _simCreatedGeneration = generation;

    if (grabbed->IsStatic()) {
      // Static bodies can't be moved; skip the constraints (creating them would error) and let the
      // gizmo render gray as feedback.
      _simStatic = true;
      _simLocalPosition = worldFromLocal.TransformPointInverse(target);
    } else if (grabbed->GetType() == mochi::ActorType::Soft) {
      // Grab a cluster of nodes near the pick (soft bodies have no rigid pivot). A single node
      // pinches, so spring every node within a radius and translate them rigidly with the drag.
      mochi::Error nodeError;
      mochi::Span<mochi::real const> const local = grabbed->GetNodePositionsLocal(nodeError);
      if (!nodeError.IsOK() || local.size() < 3) {
        publishDebug(false, true, {}, {}, 0_r); // node data unavailable (query not ready)
        return;
      }
      int const nodeCount = mochi::isize(local) / 3;
      auto nodeWorldOf = [&](int i) {
        return worldFromLocal.TransformPoint(
            mochi::Real3{local[3 * i], local[3 * i + 1], local[3 * i + 2]});
      };
      // First pass: nearest node to the pick + the body's world bounds (for a scale-relative
      // radius).
      int anchor = -1;
      mochi::real bestSq = std::numeric_limits<mochi::real>::max();
      mochi::Real3 boundsMin = nodeWorldOf(0);
      mochi::Real3 boundsMax = boundsMin;
      for (int i = 0; i < nodeCount; ++i) {
        mochi::Real3 const p = nodeWorldOf(i);
        boundsMin = mochi::Min(boundsMin, p);
        boundsMax = mochi::Max(boundsMax, p);
        mochi::real const distSq = mochi::NormSqr(p - target);
        if (distSq < bestSq) {
          bestSq = distSq;
          anchor = i;
        }
      }
      if (anchor < 0) {
        publishDebug(false, true, {}, {}, 0_r);
        return;
      }
      _simSoft = true;
      _simNodeIndex = anchor;
      mochi::Real3 const anchorWorld = nodeWorldOf(anchor);
      _simTarget = anchorWorld; // virtual hand starts on the anchor node (no initial jump)

      mochi::Real3 const extent = boundsMax - boundsMin;
      mochi::real const radius = settings.softGrabRadiusFraction * mochi::Norm(extent);
      mochi::real const radiusSq = mochi::Sqr(radius);

      // Second pass: gather nodes within the radius of the pick, closest first (capped).
      std::vector<std::pair<mochi::real, int>> within;
      for (int i = 0; i < nodeCount; ++i) {
        mochi::real const distSq = mochi::NormSqr(nodeWorldOf(i) - target);
        if (distSq <= radiusSq) {
          within.emplace_back(distSq, i);
        }
      }
      if (within.empty()) {
        within.emplace_back(bestSq, anchor); // radius too small; at least grab the anchor node
      }
      std::sort(within.begin(), within.end());
      if (mochi::isize(within) > settings.softGrabMaxNodes) {
        within.resize(settings.softGrabMaxNodes);
      }

      mochi::Error massError;
      mochi::real softMass = grabbed->GetMass(massError);
      if (!massError.IsOK() || !(softMass > 0_r)) {
        softMass = 1_r; // fallback (undefined mass); harmless
      }
      // Split the total mass across the cluster so the combined authority matches a single
      // total-mass grab, just spread out to avoid pinching a single vertex.
      mochi::real const perNodeMass = softMass / static_cast<mochi::real>(within.size());
      mochi::real const omega = kTwoPi * settings.responseFrequencyHz;
      mochi::real const omegaSq = omega * omega;
      for (auto const& entry : within) {
        int const i = entry.second;
        mochi::Real3 const nodeWorld = nodeWorldOf(i);
        mochi::DeformableNodePositionConstraintParams params;
        params.actor = actor;
        params.nodeIndex = i;
        params.position = nodeWorld;
        params.stiffness = perNodeMass * omegaSq;
        params.damping = 2_r * settings.dampingRatio * perNodeMass * omega;
        params.saturation =
            settings.maxAcceleration / omegaSq; // distance; elastic force caps at m*a_max
        mochi::ErrorLog error;
        if (mochi::Constraint* constraint =
                scene->CreateDeformableNodePositionConstraint(params, error)) {
          // No CancelQuery: the query is a component on the constraint's entity, so it is destroyed
          // along with the constraint in destroyConstraints().
          mochi::ErrorLog queryError;
          constraint->RegisterQuery(mochi::QueryType::ConstraintForce, queryError);
          _simSoftConstraints.push_back({constraint->GetHandle(), nodeWorld - anchorWorld});
        }
      }
      _simSetupFailed = _simSoftConstraints.empty();
    } else {
      // Anchor the spring at the grabbed surface point in the actor's local frame (no initial
      // jump).
      _simLocalPosition = worldFromLocal.TransformPointInverse(target);
      mochi::Error massError;
      mochi::real linkMass = grabbed->GetMass(massError);
      if (!massError.IsOK() || !(linkMass > 0_r)) {
        linkMass = 1_r; // fallback (undefined mass); harmless
      }
      // For an articulated link, use the whole articulation's mass so a light distal link can still
      // drag the chain; massScale (>= 1) boosts the rotation stabilizer to match.
      mochi::real const effectiveMass = EffectiveGrabMass(scene, actor, linkMass);
      mochi::real const massScale = effectiveMass / linkMass;
      mochi::real const omega = kTwoPi * settings.responseFrequencyHz;
      mochi::real const omegaSq = omega * omega;

      mochi::RigidPivotPositionConstraintParams params;
      params.actor = actor;
      params.targetPosition = target;
      params.localPosition = _simLocalPosition;
      params.stiffness = effectiveMass * omegaSq;
      params.damping = 2_r * settings.dampingRatio * effectiveMass * omega;
      params.saturation =
          settings.maxAcceleration / omegaSq; // distance; elastic force caps at m_eff*a_max
      mochi::ErrorLog error;
      if (mochi::Constraint* constraint =
              scene->CreateRigidPivotPositionConstraint(params, error)) {
        // No CancelQuery: the query is a component on the constraint's entity, so it is destroyed
        // along with the constraint in destroyConstraints(). The companion rotation constraint is
        // deliberately left unqueried -- it reports torque, which the label does not show.
        mochi::ErrorLog queryError;
        constraint->RegisterQuery(mochi::QueryType::ConstraintForce, queryError);
        _simConstraint = constraint->GetHandle();
        _simConstraintValid = true;
      }
      _simSetupFailed = !_simConstraintValid;

      // Optional soft rotation constraint pinning the orientation the body had at grab time.
      if (settings.enableRotationConstraint && _simConstraintValid) {
        mochi::Error moiError;
        mochi::Real6 const moi = grabbed->GetRigidMomentOfInertiaLocal(moiError);
        // Real6 upper triangle [ixx, ixy, ixz, iyy, iyz, izz]; mean of the diagonal (0, 3, 5) is a
        // characteristic scalar inertia, scaled up for articulated links to match the position
        // gain.
        mochi::real const inertia = (moi[0] + moi[3] + moi[5]) / 3_r * massScale;
        if (moiError.IsOK() && inertia > 0_r) {
          mochi::real const omegaRot = kTwoPi * settings.rotResponseFrequencyHz;
          mochi::real const omegaRotSq = omegaRot * omegaRot;
          mochi::RigidPivotRotationConstraintParams rotParams;
          rotParams.actor = actor;
          rotParams.localRotation = {};
          // Target the current world orientation via params (not SetTargetRotation) so both the
          // current and stage-start targets init to it; otherwise it would drive toward identity.
          rotParams.targetRotation = worldFromLocal.GetRotation().ToRotationVector();
          rotParams.stiffness = inertia * omegaRotSq;
          rotParams.damping = 2_r * settings.rotDampingRatio * inertia * omegaRot;
          rotParams.saturation =
              settings.maxAngularAcceleration / omegaRotSq; // angle; torque caps at I*a_max
          mochi::ErrorLog rotError;
          if (mochi::Constraint* rot =
                  scene->CreateRigidPivotRotationConstraint(rotParams, rotError)) {
            _simRotConstraint = rot->GetHandle();
            _simRotConstraintValid = true;
          }
        }
      }
    }
  } else if (!_simStatic) {
    // Advance the virtual hand toward the cursor: rate-limit its speed, then leash it to the grab
    // point so a jammed body can't let the target (and thus the damping force) run away.
    mochi::real const dt = static_cast<mochi::real>(info.timeStepSec);
    _simTarget = _simTarget + ClampMagnitude(target - _simTarget, settings.maxTargetSpeed * dt);
    if (mochi::Actor* grabbed = scene->GetActor(actor)) {
      mochi::Real3 const anchorWorld = currentAnchorWorld(grabbed);
      _simTarget = anchorWorld + ClampMagnitude(_simTarget - anchorWorld, settings.maxStretch);
    }
    if (_simSoft) {
      // Translate the whole grabbed cluster: each node targets the virtual hand plus its fixed
      // offset from the anchor.
      mochi::ErrorLog error;
      for (SoftNodeConstraint const& c : _simSoftConstraints) {
        if (mochi::Constraint* constraint = scene->GetConstraint(c.handle)) {
          constraint->SetTargetPosition(_simTarget + c.offset, error);
        }
      }
    } else if (mochi::Constraint* constraint = scene->GetConstraint(_simConstraint)) {
      mochi::ErrorLog error;
      constraint->SetTargetPosition(_simTarget, error);
    }
  }

  mochi::Actor* grabbed = scene->GetActor(actor);
  if (grabbed == nullptr) {
    publishDebug(false, true, {}, {}, 0_r);
    return;
  }
  mochi::Real3 const anchorWorld = currentAnchorWorld(grabbed);
  if (_simStatic) {
    // Fixed grab point on the static body, line drawn to the cursor; gray = not draggable.
    publishDebug(true, false, anchorWorld, target, 0_r);
  } else if (_simConstraintValid || !_simSoftConstraints.empty()) {
    publishDebug(true, true, anchorWorld, _simTarget, currentGrabForce());
  } else {
    publishDebug(false, true, {}, {}, 0_r);
  }
}

// Fields baked into the constraints when a grab is set up. Changing any of them has no effect on a
// live grab unless it is torn down and rebuilt, so SetSettings bumps the generation for these.
static bool GrabGainsDiffer(PhysicsDragSettings const& a, PhysicsDragSettings const& b) {
  return a.responseFrequencyHz != b.responseFrequencyHz || a.dampingRatio != b.dampingRatio ||
      a.maxAcceleration != b.maxAcceleration ||
      a.enableRotationConstraint != b.enableRotationConstraint ||
      a.rotResponseFrequencyHz != b.rotResponseFrequencyHz ||
      a.rotDampingRatio != b.rotDampingRatio ||
      a.maxAngularAcceleration != b.maxAngularAcceleration ||
      a.softGrabRadiusFraction != b.softGrabRadiusFraction ||
      a.softGrabMaxNodes != b.softGrabMaxNodes;
}

void PhysicsDragController::SetSettings(PhysicsDragSettings settings) {
  PhysicsDragSettings const sanitized = SanitizeSettings(settings);
  std::lock_guard<std::mutex> lock(_mutex);
  // Only invalidate a live grab when a gain that is baked in at setup actually moved: the rebuild
  // is a whole-scene mass scan plus two passes over the soft nodes. maxTargetSpeed, maxStretch and
  // debugSphereRadius are read fresh each step (or frame), so they need no rebuild.
  if (_active && GrabGainsDiffer(_settings, sanitized)) {
    ++_generation;
  }
  _settings = sanitized;
}
void PhysicsDragController::DrawDebug(
    mochi_renderer::DebugDraw* debugDraw,
    DebugText* debugText,
    mochi::CoordinateSpaceConverter const& converter) const {
  if (debugDraw == nullptr || !_dragging) {
    return;
  }
  _debugData.Consume();
  DragDebugData const& data = _debugData.GetConsumerData();
  if (!data.active) {
    return;
  }
  auto toRender = [&converter](mochi::Real3 const& p) {
    return mochi_renderer::ToFilament<float>(converter.TranslationToOutput(p));
  };
  filament::math::float3 const anchor = toRender(data.anchor);
  filament::math::float3 const target = toRender(data.target);
  filament::math::float4 const color = data.grabbable ? kDebugColor : kDebugGrayColor;
  float sphereRadius = 0.0f;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    sphereRadius = _settings.debugSphereRadius;
  }
  debugDraw->DrawSolidSphere(anchor, sphereRadius, color);
  debugDraw->DrawSolidSphere(target, sphereRadius, color);
  debugDraw->DrawLine(anchor, target, color);
  if (debugText != nullptr && data.grabbable) {
    char label[32];
    std::snprintf(label, sizeof(label), "%.3f N", static_cast<double>(data.forceNewtons));
    debugText->Draw(target, label, color, kForceLabelOffset);
  }
}

void BindSceneObjectDragHooks(
    Viewport& viewport,
    std::unique_ptr<PhysicsDragController> const& controller) {
  // The controller is created on simulation start and destroyed on stop, so the hooks read the slot
  // through its address rather than capturing the controller itself.
  auto const* slot = &controller;
  viewport.onSceneObjectDragStart =
      [slot](mochi_renderer::SceneObject* object, filament::math::float3 hitPoint) {
        return *slot && (*slot)->BeginDrag(object, hitPoint);
      };
  viewport.onSceneObjectDragUpdate = [slot](filament::math::float3 target) {
    if (*slot) {
      (*slot)->UpdateDrag(target);
    }
  };
  viewport.onSceneObjectDragEnd = [slot]() {
    if (*slot) {
      (*slot)->EndDrag();
    }
  };
}

} // namespace superdex::studio
