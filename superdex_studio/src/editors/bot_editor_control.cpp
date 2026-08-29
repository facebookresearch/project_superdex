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

#include "editors/bot_editor_control.h"

#include "app/app.h"
#include "ui/imgui_widgets.h"

#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>

#include <tinyfiledialogs.h> // tinyfd_messageBox (normalized-path overwrite confirmation)

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

namespace superdex::studio {

namespace {

// Bounds on the pose controller's closed-loop bandwidth knob. The lower bound keeps omega > 0.
constexpr float kMinFrequencyHz = 0.1f;
constexpr float kMaxFrequencyHz = 100.0f;
constexpr std::string_view kControllerExtension = ".superdex_controller";

} // namespace

void BotControl::SetBot(superdex::robotics::Bot* bot) {
  // Destroy against the outgoing bot's context before rebinding. DestroyBot would cascade-destroy
  // the controller anyway, but this keeps the cached handle from dangling.
  DestroyPoseController();
  _bot = bot;
  _actorDofIndices.clear();
  _armature.clear();
  _effectiveInertias.clear();
  _linkInertias.clear();
  _linkTransforms.clear();
  _massMatrix.clear();
  _numActorDofs = 0;
  _inertiaRefreshFailed = false;
  if (_bot != nullptr) {
    PrepareInertiaState();
  }
}

void BotControl::PrepareInertiaState() {
  using namespace mochi;
  Actor* const actor = _bot->GetArticulatedActor();
  auto const& botPrefab = _bot->GetBotPrefab();
  int const numDofs = botPrefab._numDofs;
  if (actor == nullptr || numDofs <= 0) {
    return; // Nothing this panel can drive.
  }

  // A bot's DOFs exclude the root joint's, but the actor's include them, so the two spaces differ
  // by a constant offset. Everything indexed in actor DOFs (the mass matrix, the external-force
  // push) goes through _actorDofIndices.
  int const numBaseDofs =
      (!botPrefab.joints.empty() && botPrefab.joints[0].type == ArticulatedJointType::Free) ? 6 : 0;
  _actorDofIndices.resize(numDofs);
  _armature.resize(numDofs);
  for (int i = 0; i < numDofs; ++i) {
    _actorDofIndices[i] = i + numBaseDofs;
    // BotPrefab::_dofIndices holds joint indices, and only Revolute/Prismatic/Spherical joints
    // appear in it. The joint inertia coefficient is the rotor/armature inertia: it adds to the
    // mass-matrix diagonal but appears in no link's rigid inertia, so ComputeArticulatedMassMatrix
    // does not see it. It follows the DOF's units ([kg*m^2] rotational, [kg] prismatic), so it can
    // be summed with the diagonal directly. For distal joints it dominates -- on the FR3 wrist it
    // is ~100x the link inertia, and omitting it makes the joint that much too soft. A spherical
    // joint's single coefficient applies isotropically to all three of its DOFs.
    _armature[i] = Max(0_r, botPrefab.joints[botPrefab._dofIndices[i]].inertia.value_or(0_r));
  }
  _effectiveInertias.resize(numDofs, 0_r);

  _numActorDofs = actor->GetNumDofs();
  // Guards the constant-offset assumption above. Dropping the mapping disables both the effort push
  // and the pose gains, which is the safe outcome: the indices would otherwise run off the end of
  // the mass matrix and address the wrong actor DOFs.
  if (numDofs + numBaseDofs != _numActorDofs) {
    MOCHI_LOG_WARNING(
        "BotControl: bot has %d DOFs plus %d root DOFs but the actor reports %d; pose and effort "
        "control are disabled",
        numDofs,
        numBaseDofs,
        _numActorDofs);
    _actorDofIndices.clear();
    _inertiaRefreshFailed = true;
    return;
  }

  // Per-link inertias are constant in the configuration; only the transforms change.
  ErrorLog error;
  superdex::robotics::BuildArticulatedLinkInertias(actor, _linkInertias, error);
  if (!error.IsOK() || _linkInertias.empty()) {
    _inertiaRefreshFailed = true;
    return;
  }
  _linkTransforms.resize(_linkInertias.size());
  _massMatrix.resize(static_cast<size_t>(_numActorDofs) * _numActorDofs);
  if (!RefreshEffectiveInertias()) {
    _inertiaRefreshFailed = true;
  }
}

bool BotControl::RefreshEffectiveInertias() {
  using namespace mochi;
  Actor* const actor = _bot != nullptr ? _bot->GetArticulatedActor() : nullptr;
  if (actor == nullptr || _numActorDofs <= 0 || _linkInertias.empty()) {
    return false;
  }
  ErrorLog error;
  // These are link-origin transforms, not COM ones. That is what ComputeArticulatedMassMatrix
  // documents as its input and all it needs: it uses only their rotation, which the two frames
  // share. Do not "fix" this by shifting them to the COM.
  actor->GetArticulatedLinkTransforms(_linkTransforms, error);
  superdex::robotics::ComputeArticulatedMassMatrix(
      actor, _linkInertias, _linkTransforms, _numActorDofs, _massMatrix, error);
  if (!error.IsOK()) {
    return false;
  }
  // The diagonal of the joint-space mass matrix is each DOF's effective inertia (or mass, for a
  // prismatic DOF) at the current configuration -- for a revolute joint, the distal subtree's
  // moment about the joint axis. Units follow the DOF, matching PoseTrackingParams.
  for (int iDof = 0; iDof < isize(_effectiveInertias); ++iDof) {
    int const actorDof = _actorDofIndices[iDof];
    real const diagonal = _massMatrix[static_cast<size_t>(actorDof) * _numActorDofs + actorDof];
    _effectiveInertias[iDof] = Max(0_r, diagonal + _armature[iDof]);
  }
  return true;
}

mochi::CallbackHandle BotControl::RegisterPreStepCallback(mochi::AsyncScene* scene) {
  using namespace mochi;
  if (!_bot) {
    return {};
  }
  // Size the per-session state from the prefab (matching the slider iteration order used by
  // ShowWindow, which walks BotPrefab::_dofIndices). The DOF mapping and inertias were already
  // cached by SetBot.
  auto const& botPrefab = _bot->GetBotPrefab();
  int const numDofs = botPrefab._numDofs;
  _efforts.clear();
  _efforts.resize(numDofs, 0_r);
  // Command the spawn pose, so enabling pose control holds the bot where it appeared.
  _poseTarget = botPrefab.defaultPose;
  _poseTarget.resize(numDofs, 0_r);
  _gainsDirty.store(false, std::memory_order_relaxed);
  return scene->RegisterPreStepCallback(
      "BotControl", [this](mochi::StepInfo const& /* info */) { OnPreStep(); });
}

void BotControl::OnStopPhysics() {
  // Drop the session's commanded state. The window re-seeds the pose from the prefab's default pose
  // while stopped, so the next session starts from the spawn pose at zero effort.
  std::fill(_efforts.begin(), _efforts.end(), mochi::real{0});
  _poseTarget.clear();
}

void BotControl::OnPreStep() {
  using namespace mochi;
  if (_bot == nullptr) {
    return;
  }
  Actor* const actor = _bot->GetArticulatedActor();
  if (actor == nullptr) {
    return;
  }
  auto const& botPrefab = _bot->GetBotPrefab();

  // Reconcile the UI's pose-control request against the live controller here rather than at physics
  // start, so a mid-session toggle and the start-up case are the same code path. ComputeOutput
  // errors on an empty target, so a bot with no actuated DOFs is skipped entirely.
  bool const wantPose = _poseEnabled.load(std::memory_order_relaxed) && botPrefab._numDofs > 0 &&
      static_cast<int>(_poseTarget.size()) == botPrefab._numDofs;
  if (!wantPose) {
    DestroyPoseController();
  } else {
    if (_poseController == nullptr) {
      if (!_poseSetupFailed) {
        CreatePoseController();
      }
    } else {
      UpdatePoseControllerParams();
    }
    if (_poseController != nullptr) {
      // _poseTarget backs the target's non-owning span, so it must outlive this call.
      superdex::robotics::ControllerMochiArticulatedPose::Target target;
      target.worldFromRoot = botPrefab.worldFromRoot;
      target.poseDofs = MakeConstSpan(_poseTarget);
      _poseController->ComputeOutput({}, target, ErrorLog{});
    }
  }

  // Skipped when the DOF mapping was rejected in PrepareInertiaState, which leaves the two arrays
  // at different sizes.
  if (_actorDofIndices.size() == _efforts.size()) {
    actor->SetExternalForcesOnDofs(
        MakeConstSpan(_actorDofIndices), MakeConstSpan(_efforts), ErrorLog{});
  }
}

void BotControl::CreatePoseController() {
  using namespace superdex::robotics;
  RoboticsContext* const context = _bot->GetBotContext();
  mochi::ErrorLog error;
  ControllerHandle const handle = context->CreateController(
      ControllerMochiArticulatedPose::TypeName(),
      &_bot->GetBotPrefab(),
      _bot->GetArticulatedActor(),
      "BotEditorPoseControl",
      error);
  auto* const controller =
      static_cast<ControllerMochiArticulatedPose*>(context->GetController(handle));
  if (controller != nullptr) {
    // Acquire-paired with the UI's release store, so the params below see the latest frequency.
    _gainsDirty.exchange(false, std::memory_order_acquire);
    // Ordering is load-bearing: Initialize() before SetParams() silently installs zero-gain
    // defaults. Reset() makes the first ComputeOutput take the reset path, so re-enabling mid-sim
    // does not deliver a velocity kick.
    ControllerMochiArticulatedPose::Params params;
    params.poseControllerParams = BuildPoseControllerParams();
    controller->SetParams(params, error);
    controller->Initialize(/* removeExisting */ true, error);
    controller->Reset();
  }
  if (controller == nullptr || !error.IsOK()) {
    // Latch the failure so we don't retry (and re-log) every step for the rest of the session.
    context->DestroyController(handle);
    _poseSetupFailed = true;
    return;
  }
  _poseControllerHandle = handle;
  _poseController = controller;
}

void BotControl::DestroyPoseController() {
  // Clearing the latch lets a toggle-off/toggle-on retry a failed setup.
  _poseSetupFailed = false;
  if (_poseController == nullptr) {
    return;
  }
  if (_bot != nullptr) {
    _bot->GetBotContext()->DestroyController(_poseControllerHandle);
  }
  _poseController = nullptr;
  _poseControllerHandle = {};
}

void BotControl::UpdatePoseControllerParams() {
  // Effective inertia is configuration-dependent, so gains designed at one pose drift from critical
  // damping at another. Re-reading M(q) keeps them honest at the cost of a mass-matrix evaluation
  // and a params upload. Export calls this too, so a paused scene includes the current UI settings.
  bool inertiaRefreshed = false;
  if (_configDependentGains.load(std::memory_order_relaxed) && !_inertiaRefreshFailed) {
    inertiaRefreshed = RefreshEffectiveInertias();
    _inertiaRefreshFailed = !inertiaRefreshed;
  }
  bool const bandwidthChanged = _gainsDirty.exchange(false, std::memory_order_acquire);
  if (inertiaRefreshed || bandwidthChanged) {
    superdex::robotics::ControllerMochiArticulatedPose::Params params;
    params.poseControllerParams = BuildPoseControllerParams();
    _poseController->SetParams(params, mochi::ErrorLog{});
  }
}

void BotControl::ExportPoseController(mochi::AsyncScene* scene, std::string path) {
  if (scene == nullptr || _bot == nullptr) {
    return;
  }
  scene->QueueCommand([this, path = std::move(path)](mochi::Scene*) {
    if (_poseController == nullptr && !_poseSetupFailed) {
      CreatePoseController();
    }
    if (_poseController == nullptr) {
      MOCHI_LOG_WARNING("BotControl: failed to build the pose controller for export");
      return;
    }
    UpdatePoseControllerParams();
    mochi::ErrorLog error;
    superdex::robotics::EnsureDirectoriesCreated(path, error);
    _poseController->GetParams().SaveToFile(path, error);
  });
  scene->WaitForQueuedCommands();
}

mochi::PoseControllerParams BotControl::BuildPoseControllerParams() const {
  using namespace mochi;
  auto const& botPrefab = _bot->GetBotPrefab();
  int const numLinks = static_cast<int>(botPrefab.links.size());
  // Pre-sized so every untracked link (the root, Hard joints, cycle joints) keeps a zero gain.
  // Cartesian tracking is left at zero throughout; it would fight the joint-space PD.
  PoseControllerParams params{numLinks};
  // Critically damped (zeta = 1) second-order response at the commanded natural frequency:
  // k = Q * omega^2 and d = 2 * Q * omega, with Q the DOF's effective inertia (or mass). Solving
  // from omega rather than from a hand-picked stiffness is what makes every joint track with the
  // same bandwidth regardless of its inertia.
  real const omega = 2_r * kPI * static_cast<real>(_frequencyHz.load(std::memory_order_relaxed));
  size_t const numDofs = std::min(botPrefab._dofIndices.size(), _effectiveInertias.size());
  for (size_t iDof = 0; iDof < numDofs; ++iDof) {
    // Joint index == link index in both the bot and the articulated actor.
    int const iLink = botPrefab._dofIndices[iDof];
    real const effective = _effectiveInertias[iDof];
    if (iLink < 0 || iLink >= numLinks || effective <= 0_r) {
      continue;
    }
    real const stiffness = effective * omega * omega;
    PoseTrackingParams& tracking = params.jointTracking[iLink];
    // A spherical joint's three DOFs share one rotation constraint and so one slot; keep the
    // largest of the three so the heaviest axis is not left under-damped.
    if (stiffness <= tracking.stiffness) {
      continue;
    }
    tracking.stiffness = stiffness;
    tracking.damping = 2_r * effective * omega;
    // Saturate the elastic term at the joint's rated effort. Unbounded (< 0) and non-actuated (0)
    // limits leave saturation disabled.
    real const effortLimit = botPrefab.joints[iLink].effortLimit;
    tracking.saturation = effortLimit > 0_r ? effortLimit / stiffness : -1_r;
  }
  return params;
}

void BotControl::ShowWindow(
    bool* open,
    superdex::robotics::BotPrefab const& editPrefab,
    bool isSimulating,
    mochi::AsyncScene* scene,
    mochi::Path const& botPath) {
  using namespace mochi;
  if (!ImGui::Begin("Bot Control", open)) {
    ImGui::End();
    return;
  }

  if (editPrefab._dofIndices.empty()) {
    ImGui::TextDisabled("Bot has no actuated DOFs");
    ImGui::End();
    return;
  }

  ImGui::HoverableSeparatorText("Emulate Gravity Compensation");
  ImGui::PushID("Emulate GravityCompensation");
  // Mochi has no runtime gravity toggle, so this can only be applied when the bot spawns.
  ImGui::BeginDisabled(isSimulating);
  ImGui::Checkbox("Enable", &_gravityCompensation);
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip(
        "Spawn the bot with gravity disabled on every link to emulate gravity compensation.\n"
        "Applied when the simulation starts, so it cannot be changed while simulating.");
  }
  ImGui::PopID();

  ImGui::HoverableSeparatorText("Pose Control");
  ImGui::PushID("PoseControl");
  bool poseEnabled = _poseEnabled.load(std::memory_order_relaxed);
  if (ImGui::Checkbox("Enable", &poseEnabled)) {
    _poseEnabled.store(poseEnabled, std::memory_order_relaxed);
  }
  float frequencyHz = _frequencyHz.load(std::memory_order_relaxed);
  if (ImGui::DragFloat(
          "Closed-Loop Bandwidth",
          &frequencyHz,
          0.05f,
          kMinFrequencyHz,
          kMaxFrequencyHz,
          "%.3f Hz",
          ImGuiSliderFlags_AlwaysClamp)) {
    _frequencyHz.store(frequencyHz, std::memory_order_relaxed);
    // Release-paired with the sim thread's acquire exchange, so it cannot consume the flag and
    // still read the pre-edit frequency.
    _gainsDirty.store(true, std::memory_order_release);
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Closed-loop bandwidth of the pose controller. Gains are derived from the\n"
        "bot's own mass properties for a critically damped response at this frequency.");
  }
  bool configDependentGains = _configDependentGains.load(std::memory_order_relaxed);
  if (ImGui::Checkbox("Configuration-Dependent Gains", &configDependentGains)) {
    _configDependentGains.store(configDependentGains, std::memory_order_relaxed);
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Recompute gains every step from the bot's mass matrix, following each joint's\n"
        "changing effective inertia. Off: gains stay frozen at the spawn pose.");
  }
  bool const canExport = isSimulating && poseEnabled && scene != nullptr;
  ImGui::BeginDisabled(!canExport);
  if (ImGui::Button("Export Controller...")) {
    mochi::Path const controllerDir = botPath.GetParentPath() / "control";
    mochi::Path const defaultPath =
        controllerDir / (botPath.GetStem() + "_pose" + std::string{kControllerExtension});
    // tinyfd falls back to its last-used directory when the default's parent does not exist.
    std::error_code createError;
    std::filesystem::create_directories(controllerDir.AsFilesystemPath(), createError);
    if (createError) {
      MOCHI_LOG_WARNING("BotControl: failed to create default controller export directory");
    }
    std::array<char const*, 1> const filters = {"*.superdex_controller"};
    mochi::Path outputPath = SuperDexStudio::GetFileDialogPath(
        "Export Pose Controller",
        filters.data(),
        static_cast<int>(filters.size()),
        "SuperDex Controller (*.superdex_controller)",
        /*isSaveDialog=*/true,
        defaultPath);
    if (!outputPath.IsEmpty()) {
      std::string const selectedPath = outputPath.ToString();
      outputPath.ReplaceExtension(kControllerExtension);
      bool canOverwrite = true;
      if (selectedPath != outputPath.ToString()) {
        std::error_code existsError;
        bool const normalizedPathExists =
            std::filesystem::exists(outputPath.AsFilesystemPath(), existsError);
        if (existsError) {
          MOCHI_LOG_WARNING("BotControl: failed to inspect the controller export destination");
          canOverwrite = false;
        } else if (normalizedPathExists) {
          canOverwrite = tinyfd_messageBox(
                             "Overwrite Controller",
                             "The normalized .superdex_controller file already exists. Replace it?",
                             "yesno",
                             "warning",
                             /*aDefaultButton=*/0) == 1;
        }
      }
      if (canOverwrite) {
        ExportPoseController(scene, outputPath.ToString());
      }
    }
  }
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip(
        canExport ? "Export the fully built pose controller's current parameters."
                  : "Start the simulation and enable pose control before exporting.");
  }
  // While simulating, the per-DOF arrays are owned by the sim thread (sized at physics start and
  // read every step), so they are only ever resized here while stopped.
  if (!isSimulating && _poseTarget.size() != editPrefab._dofIndices.size()) {
    _poseTarget = editPrefab.defaultPose;
    _poseTarget.resize(editPrefab._dofIndices.size(), 0_r);
  }
  if (_poseTarget.size() == editPrefab._dofIndices.size()) {
    ImGui::BeginDisabled(!isSimulating || !poseEnabled);
    ImGui::JointPoseEditor(editPrefab, _poseTarget);
    ImGui::EndDisabled();
  }
  ImGui::PopID();

  ImGui::HoverableSeparatorText("Effort Control");
  if (!isSimulating && static_cast<int>(_efforts.size()) != editPrefab._numDofs) {
    _efforts.clear();
    _efforts.resize(editPrefab._numDofs, 0_r);
  }
  if (static_cast<int>(_efforts.size()) != editPrefab._numDofs) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: DOF count mismatch");
    ImGui::TextDisabled(
        "Bot has %d DOFs but control state has %zu. Simulation may need restart.",
        editPrefab._numDofs,
        _efforts.size());
    ImGui::End();
    return;
  }
  ImGui::BeginDisabled(!isSimulating);
  for (size_t iDof = 0; iDof < editPrefab._dofIndices.size(); ++iDof) {
    int const iJoint = editPrefab._dofIndices[iDof];
    auto const& joint = editPrefab.joints[iJoint];
    ImGui::PushID(static_cast<int>(iDof));
    std::array<char, 256> label{};
    snprintf(label.data(), label.size(), "[%zu] %s", iDof, joint.name.c_str());
    char const* unitFmt =
        GetUnitFormat(UnitFormat::Effort, joint.type, /* unused for format */ 0.0f);
    auto value = static_cast<float>(_efforts[iDof]);
    auto const limit = static_cast<float>(joint.effortLimit);
    if (limit > 0.0f) {
      // Finite limit: bounded slider [-limit, +limit].
      if (ImGui::SliderFloat(label.data(), &value, -limit, limit, unitFmt)) {
        _efforts[iDof] = static_cast<mochi::real>(value);
      }
    } else if (limit < 0.0f) {
      // Unbounded: no finite range, so drag any magnitude instead of a bounded slider.
      if (ImGui::DragFloat(label.data(), &value, 1.0f, 0.0f, 0.0f, unitFmt)) {
        _efforts[iDof] = static_cast<mochi::real>(value);
      }
    } else {
      // Non-actuated (0): no effort can be applied; force to zero and disable.
      _efforts[iDof] = static_cast<mochi::real>(0.0f);
      ImGui::BeginDisabled(true);
      float zero = 0.0f;
      ImGui::SliderFloat(label.data(), &zero, 0.0f, 0.0f, unitFmt);
      ImGui::EndDisabled();
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Non-actuated joint (effortLimit 0): no effort can be applied");
      }
    }
    ImGui::PopID();
  }
  ImGui::EndDisabled();
  ImGui::End();
}

} // namespace superdex::studio
