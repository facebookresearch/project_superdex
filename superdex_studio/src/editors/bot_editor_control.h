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

#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/articulated_mass_matrix.h>

#include <mochi_physics/mochi_physics.h>

#include <atomic>
#include <string>

namespace mochi {
class AsyncScene;
class Path;
} // namespace mochi

namespace superdex::robotics {
class ControllerMochiArticulatedPose;
} // namespace superdex::robotics

namespace superdex::studio {

// The "Bot Control" window together with its simulation-side state. Three sections:
//
//  - Pose control: a joint-space PD that holds the bot at a commanded pose. Gains are derived from
//    the bot's own mass properties so a single knob -- the closed-loop bandwidth [Hz] -- behaves
//    the same across bots of wildly different scale.
//  - Gravity compensation: spawns the bot with gravity disabled on every link.
//  - Effort control: a per-DOF effort applied as an external DOF force.
//
// Pose and effort control are active simultaneously; the pose controller's constraint forces and
// the effort forces simply add.
//
// Threading: the UI thread edits the commanded pose/efforts and the knobs while the sim thread
// reads them in the pre-step callback. The knobs are atomics; _poseTarget and _efforts are read
// unsynchronized, which is tolerant of slightly stale slider values (they are only ever resized
// while stopped). The inertia state and the controller are touched only by the sim thread, and all
// controller creation, destruction, and stepping happens there.
class BotControl {
 public:
  BotControl() = default;
  ~BotControl() = default;

  // The pre-step callback captures `this`, so the object must not move.
  BotControl(BotControl const&) = delete;
  BotControl& operator=(BotControl const&) = delete;
  BotControl(BotControl&&) = delete;
  BotControl& operator=(BotControl&&) = delete;

  // Binds the simulated bot (borrowed; null when there is none) and caches everything derived from
  // it: the DOF mapping, the per-joint inertia coefficients, and the mass-matrix workspace. Passing
  // null tears down the pose controller. Sim thread: called from the editor's physics-actor
  // create/destroy hooks, so the bot's mass properties are readable.
  void SetBot(superdex::robotics::Bot* bot);

  // Sizes the per-DOF state for the bound bot and registers the per-step callback. UI thread.
  [[nodiscard]] mochi::CallbackHandle RegisterPreStepCallback(mochi::AsyncScene* scene);

  // Clears the session's commanded pose and efforts. UI thread.
  void OnStopPhysics();

  // Renders the "Bot Control" window. @p editPrefab is the bot being edited (which is also the bot
  // being simulated, since editing is disabled during simulation); it supplies the DOF list the
  // sliders are indexed by. @p botPath supplies the default export folder and filename, and @p
  // scene runs the export against simulation-owned controller state. The slider sections and
  // controller export are disabled while not simulating, the gravity checkbox is disabled while
  // simulating, and the pose-control knobs stay live throughout. UI thread.
  void ShowWindow(
      bool* open,
      superdex::robotics::BotPrefab const& editPrefab,
      bool isSimulating,
      mochi::AsyncScene* scene,
      mochi::Path const& botPath);

  // True when the bot should spawn with gravity disabled on every link. Mochi cannot toggle gravity
  // on a live actor, so the spawn code bakes this into its copy of the prefab and it is fixed for
  // the session. Safe to read from the sim thread: the checkbox is disabled while simulating, and
  // the spawn command is issued and waited on by the UI thread.
  [[nodiscard]] bool IsGravityCompensated() const {
    return _gravityCompensation;
  }

 private:
  // Per-step reconciliation of the UI-requested pose control with the live controller, followed by
  // the effort push. Sim thread.
  void OnPreStep();

  // Caches everything SetBot derives from the bot and takes the first inertia reading. Sim thread.
  void PrepareInertiaState();

  // Recomputes _effectiveInertias from the actor's joint-space mass matrix at its current
  // configuration. Returns false if the actor could not be read. Sim thread.
  [[nodiscard]] bool RefreshEffectiveInertias();

  // Creates the pose controller with gains for the current bandwidth, or leaves it null on failure.
  void CreatePoseController();
  void DestroyPoseController();
  // Applies any pending bandwidth and configuration-dependent gain update. Requires a live
  // _poseController. Sim thread.
  void UpdatePoseControllerParams();

  // Saves the complete live pose-controller params on the simulation thread.
  void ExportPoseController(mochi::AsyncScene* scene, std::string path);

  // Per-link PD gains for the current bandwidth, derived from the cached effective inertias.
  [[nodiscard]] mochi::PoseControllerParams BuildPoseControllerParams() const;

  // Borrowed simulated bot; set on physics start, cleared on physics stop.
  superdex::robotics::Bot* _bot = nullptr;

  // --- Pose control ---
  // UI writes, sim thread reads.
  std::atomic<bool> _poseEnabled{false};
  // Undamped natural frequency of the joint PD [Hz]; the single tuning knob. Presented as the
  // closed-loop bandwidth, which for a critically damped response is the slightly lower 0.64x of
  // this.
  std::atomic<float> _frequencyHz{5.0f};
  // Set by the UI when the bandwidth changes; consumed by the sim thread to hot-swap the gains.
  std::atomic<bool> _gainsDirty{false};
  // When set, the effective inertias (and hence the gains) are recomputed from M(q) every step
  // instead of staying frozen at the configuration the bot spawned in.
  std::atomic<bool> _configDependentGains{false};
  // Commanded pose, indexed by bot DOF. Also the backing storage for the controller target, which
  // holds a non-owning span.
  mochi::DynamicArray<mochi::real> _poseTarget;

  // --- Effective inertia ---
  // Per-DOF effective inertia [kg*m^2] for rotational DOFs, or mass [kg] for prismatic ones: the
  // diagonal of the actor's joint-space mass matrix plus the joint's own inertia coefficient.
  mochi::DynamicArray<mochi::real> _effectiveInertias;
  // Per-DOF joint inertia coefficient (rotor/armature), which the mass matrix does not include.
  // Constant in the configuration, so it is cached with the bot.
  mochi::DynamicArray<mochi::real> _armature;
  // Mass-matrix workspace. Link inertias are constant in the configuration; the link transforms and
  // the matrix itself are refreshed on every reading.
  mochi::DynamicArray<superdex::robotics::ArticulatedLinkInertia> _linkInertias;
  mochi::DynamicArray<mochi::TransformRT> _linkTransforms;
  mochi::DynamicArray<mochi::real> _massMatrix;
  int _numActorDofs = 0;
  // Latches a failed inertia reading so it is not retried (and re-logged) on every step. Sim thread
  // only; the last good inertias are kept.
  bool _inertiaRefreshFailed = false;

  superdex::robotics::ControllerHandle _poseControllerHandle;
  superdex::robotics::ControllerMochiArticulatedPose* _poseController = nullptr;
  // Latches a failed controller setup so it is not retried (and re-logged) on every step. Sim
  // thread only; cleared when the controller is torn down.
  bool _poseSetupFailed = false;

  // --- Gravity compensation ---
  // UI-thread state, read once per session by the spawn code (see IsGravityCompensated).
  bool _gravityCompensation = false;

  // --- Effort control ---
  // Commanded effort per bot DOF.
  mochi::DynamicArray<mochi::real> _efforts;
  // Actor DOF index per bot DOF (offset past the root joint's DOFs when the root is Free). Also
  // indexes the mass matrix, which is sized in actor DOFs. Left empty when the offset could not be
  // established, which disables both the effort push and the pose gains.
  mochi::DynamicArray<int> _actorDofIndices;
};

} // namespace superdex::studio
