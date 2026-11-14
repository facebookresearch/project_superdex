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

#include "editors/ik_editor.h"
#include <imgui_internal.h>
#include <implot.h>
#include "app/app.h"
#include "assets/asset_manager.h"
#include "assets/bot_asset.h"
#include "assets/ik_asset.h"
#include "core/ik_design_session.h"
#include "rendering/measure_tool.h"
#include "rendering/stats.h"
#include "ui/asset_browser.h"
#include "ui/imgui_widgets.h"

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_renderer/mesh.h>
#include <mochi_renderer/type_conversions.h>

#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>

namespace superdex::studio {

namespace {
constexpr double kPreviewIntervalSeconds = 1.0 / 60.0;
constexpr float kGoalMarkerRadius = 0.012f;
constexpr mochi::real kDefaultOutputLinkAxisLength = mochi::real(0.08);
constexpr mochi::real kGoalAxisRatio = mochi::real(0.8);
constexpr int kInputBotSelection = -2;
// Solve plot history cap (~55 min at the 60 Hz preview rate); the oldest tenth is dropped when
// full.
constexpr size_t kMaxSolvePlotSamples = 200'000;
constexpr float kSolvePlotLineWeight = 2.0f;

char const* ConvergenceLabel(mochi::ConvergenceStatus status) {
  switch (status) {
    case mochi::ConvergenceStatus::Converged:
      return "converged";
    case mochi::ConvergenceStatus::Stopped:
      return "stopped before converging";
    case mochi::ConvergenceStatus::Diverged:
      return "diverged";
    default:
      return "no status";
  }
}

bool IsUnconverged(mochi::ConvergenceStatus status) {
  return status == mochi::ConvergenceStatus::Stopped ||
      status == mochi::ConvergenceStatus::Diverged;
}

template <typename T>
void DropOldest(std::vector<T>& values, size_t count) {
  values.erase(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count));
}

// Picker choices only; native ValidateIKJointCouplings remains the binding authority.
bool CouplingJointChoice(robotics::BotPrefab const& bot, size_t joint) {
  auto const type = bot.joints[joint].type;
  return joint != 0 &&
      (type == mochi::ArticulatedJointType::Revolute ||
       type == mochi::ArticulatedJointType::Prismatic) &&
      std::count(bot._dofIndices.begin(), bot._dofIndices.end(), static_cast<int>(joint)) == 1;
}
mochi::ArticulatedJointType CouplingJointKind(
    robotics::BotPrefab const* bot,
    std::string_view name) {
  if (bot) {
    for (size_t i = 1; i < bot->joints.size(); ++i) {
      if (bot->joints[i].name == name && CouplingJointChoice(*bot, i)) {
        return bot->joints[i].type;
      }
    }
  }
  return mochi::ArticulatedJointType::Invalid;
}
bool CouplingJointPicker(
    char const* label,
    mochi::DynamicString& value,
    mochi::DynamicString const& other,
    robotics::BotPrefab const* bot,
    float width = 0) {
  std::string shown = value.empty() ? "Choose joint" : std::string{value};
  if (!value.empty() && bot &&
      CouplingJointKind(bot, value) == mochi::ArticulatedJointType::Invalid) {
    bool const exists = std::any_of(bot->joints.begin(), bot->joints.end(), [&](auto const& joint) {
      return joint.name == std::string_view{value};
    });
    shown += exists ? " [unsupported]" : " [missing]";
  }
  bool changed = false;
  ImGui::BeginDisabled(!bot);
  ImGui::SetNextItemWidth(width > 0 ? width : -std::numeric_limits<float>::min());
  if (ImGui::BeginCombo(label, shown.c_str())) {
    if (bot) {
      auto const otherKind = CouplingJointKind(bot, other);
      for (size_t i = 1; i < bot->joints.size(); ++i) {
        auto const& joint = bot->joints[i];
        if (!CouplingJointChoice(*bot, i) || joint.name == std::string_view{other} ||
            (otherKind != mochi::ArticulatedJointType::Invalid && joint.type != otherKind)) {
          continue;
        }
        ImGui::PushID(static_cast<int>(i));
        bool const selected = joint.name == std::string_view{value};
        if (ImGui::Selectable(joint.name.c_str(), selected) && !selected) {
          value = joint.name;
          changed = true;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
        ImGui::PopID();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  return changed;
}

std::unique_ptr<mochi_renderer::Mesh> CreateGoalMarker(
    filament::Engine* engine,
    std::shared_ptr<mochi_renderer::MaterialInstance> const& material) {
  // An octahedron is a small, closed, renderer-only picking surface.
  constexpr std::array<filament::math::float3, 6> vertices = {
      filament::math::float3{1, 0, 0},
      filament::math::float3{-1, 0, 0},
      filament::math::float3{0, 1, 0},
      filament::math::float3{0, -1, 0},
      filament::math::float3{0, 0, 1},
      filament::math::float3{0, 0, -1}};
  constexpr std::array<int, 24> faces = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
                                         2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
  std::array<float, 72> positions{};
  std::array<float, 72> normals{};
  std::array<int, 24> indices{};
  for (int face = 0; face < 8; ++face) {
    auto const a = vertices[faces[3 * face]];
    auto const b = vertices[faces[3 * face + 1]];
    auto const c = vertices[faces[3 * face + 2]];
    auto const normal = normalize(cross(b - a, c - a));
    for (int corner = 0; corner < 3; ++corner) {
      int const i = 3 * face + corner;
      auto const vertex = vertices[faces[i]] * kGoalMarkerRadius;
      for (int axis = 0; axis < 3; ++axis) {
        positions[3 * i + axis] = vertex[axis];
        normals[3 * i + axis] = normal[axis];
      }
      indices[i] = i;
    }
  }
  return mochi_renderer::Mesh::CreateMesh(
      engine,
      mochi::MakeConstSpan(positions),
      mochi::MakeConstSpan(normals),
      mochi::MakeConstSpan(indices),
      material);
}
} // namespace

IKEditor::IKEditor(SuperDexStudio* studio, IKAsset* asset)
    : AssetEditor(studio, asset),
      _ikAsset(asset),
      _outputBotStage(studio, "IKOutputBotStage"),
      _inputBotStage(studio, "IKInputBotStage") {}

IKEditor::~IKEditor() {
  Shutdown();
}

void IKEditor::Initialize() {
  ResetUndoBaseline();
  _viewport = Viewport::Create(_studio, _studio->GetViewSettings());
  _viewport->enableScaleGizmo = false;
  _viewport->enableSceneObjectDrag = false;
  _viewport->onSceneSelectionChanged =
      [this](std::vector<mochi_renderer::SceneObject*> const& objects) {
        OnSceneSelectionChanged(objects);
      };
  _viewport->showTransformGizmoTarget = [this] {
    return _prepared && !IsReadOnly() &&
        ((_showGoals && HasGoalSelection()) ||
         (!_recordingOwnsPose && _selectInputBotRoot && _preview.GetInputBot()));
  };
  _viewport->getTransformGizmoTarget =
      [this]() -> std::optional<std::pair<mochi::TransformRT, mochi::Real3>> {
    if (!_prepared || IsReadOnly()) {
      return std::nullopt;
    }
    if (_selectInputBotRoot && _preview.GetInputBot()) {
      if (_recordingOwnsPose) {
        return std::nullopt;
      }
      return std::pair{
          _preview.GetWorldFrameFromOutputBaseFrame() *
              _ikAsset->GetConfig().outputBaseFrameFromInputBaseFrame,
          mochi::Real3{mochi::real(1), mochi::real(1), mochi::real(1)}};
    }
    if (!HasGoalSelection() || !_showGoals) {
      return std::nullopt;
    }
    auto const worldFrameFromConstraintFrame =
        GoalWorldConstraintFrame(_ikAsset->GetConfig().goals[_selectedGoal]);
    if (!worldFrameFromConstraintFrame) {
      return std::nullopt;
    }
    return std::pair{
        *worldFrameFromConstraintFrame,
        mochi::Real3{mochi::real(1), mochi::real(1), mochi::real(1)}};
  };
  _viewport->onTransformGizmoStarted = [this] { PushUndoSnapshot(); };
  _viewport->onTransformGizmoDelta = [this](mochi::TransformRT const& delta, mochi::Real3 const&) {
    if (!_prepared || IsReadOnly()) {
      return;
    }
    if (_selectInputBotRoot && _preview.GetInputBot()) {
      if (_recordingOwnsPose) {
        return;
      }
      auto& outputBaseFrameFromInputBaseFrame =
          _ikAsset->GetConfig().outputBaseFrameFromInputBaseFrame;
      auto const worldFrameFromOutputBaseFrame =
          mochi::NormalizeRotation(_preview.GetWorldFrameFromOutputBaseFrame());
      auto const worldFrameFromInputBaseFrame = mochi::NormalizeRotation(
          _preview.GetWorldFrameFromOutputBaseFrame() * outputBaseFrameFromInputBaseFrame);
      auto const next = mochi::NormalizeRotation(
          mochi::Invert(worldFrameFromOutputBaseFrame) * delta * worldFrameFromInputBaseFrame);
      if (mochi::IsFinite(next)) {
        outputBaseFrameFromInputBaseFrame = next;
        MarkEdited();
      }
      return;
    }
    if (!HasGoalSelection()) {
      return;
    }
    auto& goal = _ikAsset->GetConfig().goals[_selectedGoal];
    auto const worldFrameFromBaseOrInputLinkFrame = GoalConstraintAnchor(goal);
    if (!worldFrameFromBaseOrInputLinkFrame) {
      return;
    }
    auto const worldFrameFromConstraintFrame = GoalWorldConstraintFrame(goal);
    if (!worldFrameFromConstraintFrame) {
      return;
    }
    auto const next = mochi::NormalizeRotation(
        robotics::IKGoalLocalConstraintTransform(
            goal,
            mochi::NormalizeRotation(*worldFrameFromBaseOrInputLinkFrame),
            delta * mochi::NormalizeRotation(*worldFrameFromConstraintFrame),
            _ikAsset->GetConfig().inputBotScale));
    if (mochi::IsFinite(next)) {
      SetIKGoalConstraintTransform(goal, next);
      MarkEdited();
    }
  };
  _outputBotStage.BindRenderScene(_viewport->GetRenderScene());
  _inputBotStage.BindRenderScene(_viewport->GetRenderScene());
  _viewport->SetSceneStage(&_outputBotStage);
  BindSceneStageMeasureTargets(
      *_viewport, _outputBotStage, [this] { return _running; }, [this] { return !_running; });
  _viewport->SetStatsProvider([this] {
    auto const& config = _ikAsset->GetConfig();
    auto const count = [](auto const& items, auto const& predicate) {
      return static_cast<int64_t>(std::ranges::count_if(items, predicate));
    };
    StatsText text;
    // The preview session keeps its last output bot after a failed prepare.
    text.Add("Output DOFs", _prepared ? FormatBotDofs(_preview.GetOutputBot()) : "None");
    auto const* inputBot = _preview.GetInputBot();
    text.Add("Input DOFs", inputBot ? FormatBotDofs(*inputBot) : "None");
    char scale[32];
    std::snprintf(scale, sizeof(scale), "%.3f", static_cast<double>(config.inputBotScale));
    text.Add("Input Scale", scale);
    text.Add(
        "IK Goals",
        FormatBreakdown(
            static_cast<int64_t>(config.goals.size()),
            {{count(config.goals, [](auto const& g) { return !g.enabled; }), "Disabled"},
             {count(config.goals, [](auto const& g) { return g.inputBotLink.empty(); }),
              "Unbound"}}));
    return text.Add(
        "Joint Couplings",
        FormatBreakdown(
            static_cast<int64_t>(config.jointCouplings.size()),
            {{count(config.jointCouplings, [](auto const& c) { return !c.enabled; }),
              "Disabled"}}));
  });
  _viewport->RegisterShowCommand(
      {.name = "IK Constraint Frames",
       .onToggle =
           [this] {
             _showGoals = !_showGoals;
             UpdateMarkers();
           },
       .getState = [this] { return _showGoals; },
       .shortcut = ImGuiKey_T});
  auto registerMode = [this](char const* name, StageType type, ImGuiKeyChord shortcut) {
    _viewport->RegisterShowCommand(
        {.name = name,
         .onToggle =
             [this, type] {
               _stageType = type;
               RestageBots();
             },
         .getState = [this, type] { return _stageType == type; },
         .shortcut = shortcut});
  };
  registerMode("Render Only", StageType::RenderModelOnly, ImGuiKey_1);
  registerMode("Collision Only", StageType::MochiModelOnly, ImGuiKey_2);
  registerMode(
      "Render (Collision Fallback)", StageType::RenderModelFallbackToMochiModel, ImGuiKey_3);
  auto& resources = _studio->GetResourceManager();
  _markerMaterial = resources.CreateFlatLitOpaqueMaterial({0.85f, 0.85f, 0.85f});
  _selectedMarkerMaterial = resources.CreateFlatLitOpaqueMaterial(kViewportSelectionColor);
  _disabledMarkerMaterial = resources.CreateFlatLitOpaqueMaterial({0.28f, 0.28f, 0.28f});
  RestoreRecordingSelection();
  PrepareOutputBot();
  RebuildMarkers();
  _viewport->FocusCameraOnScene();
}

void IKEditor::ResetUndoBaseline() {
  if (!IsReadOnly()) {
    if (!_undoStack.IsInitialized()) {
      _undoStack.Initialize(
          [this] { return SerializeStudioIKConfig(_ikAsset->GetConfig()); },
          [this](std::string const& json, int selected) { RestoreUndoSnapshot(json, selected); });
      _undoStack.SetSelectionFn(
          [this] { return _selectInputBotRoot ? kInputBotSelection : _selectedGoal; });
    } else {
      _undoStack.Reset();
    }
    _lastUndoSnapshot = SerializeStudioIKConfig(_ikAsset->GetConfig());
    _pendingEdit = false;
    // A dirty refresh's baseline is not the persisted document. UndoStack's reset saved index
    // must not make that baseline clean; only an observed successful save can establish one.
    if (IsDirty()) {
      _savedSnapshot.reset();
    } else {
      _savedSnapshot = _lastUndoSnapshot;
    }
  }
}

void IKEditor::PushUndoSnapshot() {
  if (IsReadOnly() || !_undoStack.IsInitialized()) {
    return;
  }
  auto snapshot = SerializeStudioIKConfig(_ikAsset->GetConfig());
  bool const savedDuringInteraction = !IsDirty();
  if (savedDuringInteraction) {
    _savedSnapshot = snapshot;
  }
  if (snapshot != _lastUndoSnapshot) {
    _undoStack.PushNow();
    _lastUndoSnapshot = std::move(snapshot);
  }
  _pendingEdit = false;
  _ikAsset->SetDirty(!_savedSnapshot || _lastUndoSnapshot != *_savedSnapshot);
}

void IKEditor::CaptureSavedSnapshot() {
  // Save is owned by AssetEditor. Capture its exact clean document before another UI edit;
  // Ctrl+S runs after the UI pass and is captured by OnRender, even while a drag stays active.
  if (!IsReadOnly() && !IsDirty() &&
      (_pendingEdit || !_savedSnapshot || _lastUndoSnapshot != *_savedSnapshot)) {
    PushUndoSnapshot();
  }
}

bool IKEditor::HasGoalSelection() const {
  return _selectedGoal >= 0 && _selectedGoal < static_cast<int>(_ikAsset->GetConfig().goals.size());
}

void IKEditor::SelectGoal(int index) {
  _selectInputBotRoot = false;
  _selectedGoal = index;
  if (!HasGoalSelection()) {
    _selectedGoal = -1;
  }
  _nameBufferGoal = -1;
  SyncViewportSelection();
  UpdateMarkers();
}

void IKEditor::SelectInputBot() {
  _selectedGoal = -1;
  _selectInputBotRoot = _prepared && _preview.GetInputBot();
  _nameBufferGoal = -1;
  SyncViewportSelection();
  UpdateMarkers();
}

void IKEditor::SyncViewportSelection() {
  if (!_viewport) {
    return;
  }
  std::vector<mochi_renderer::SceneObject*> selection;
  if (_selectInputBotRoot) {
    for (auto const& actor : _inputBotStage.GetActors()) {
      if (actor.sceneObject) {
        selection.push_back(actor.sceneObject);
      }
    }
  } else if (
      HasGoalSelection() && _showGoals && _selectedGoal < static_cast<int>(_markers.size()) &&
      _markers[_selectedGoal]) {
    selection.push_back(_markers[_selectedGoal]);
  }
  _viewport->SetSceneStage(_selectInputBotRoot ? &_inputBotStage : &_outputBotStage);
  _viewport->SetSelectedSceneObjects(std::move(selection), false);
}

void IKEditor::OnSceneSelectionChanged(std::vector<mochi_renderer::SceneObject*> const& objects) {
  if (objects.empty()) {
    SelectGoal(-1);
    return;
  }
  auto const* object = objects.back();
  if (_inputBotStage.GetSceneObjectIndex(object) >= 0) {
    SelectInputBot();
    FocusWindowIfOpen("IK Solver");
    return;
  }
  if (auto const it = _markerGoals.find(object); it != _markerGoals.end()) {
    SelectGoal(it->second);
    FocusWindowIfOpen("IK Goals");
    return;
  }
  int const actorIndex = _outputBotStage.GetSceneObjectIndex(object);
  if (actorIndex >= 0) {
    auto const& actor = _outputBotStage.GetActors()[actorIndex];
    auto const& goals = _ikAsset->GetConfig().goals;
    for (int i = 0; i < static_cast<int>(goals.size()); ++i) {
      int const outputBotLinkIndex = FindOutputBotLink(goals[i].outputBotLink);
      if (outputBotLinkIndex >= 0 &&
          outputBotLinkIndex < static_cast<int>(_outputBotStagedLinkNames.size()) &&
          _outputBotStagedLinkNames[outputBotLinkIndex] == actor.name) {
        SelectGoal(i);
        FocusWindowIfOpen("IK Goals");
        return;
      }
    }
    _editDiagnostic = "This robot link has no goal. Add one from the link picker.";
  }
  SelectGoal(-1);
}

void IKEditor::RestoreUndoSnapshot(std::string const& json, int selectionIndex) {
  if (IsReadOnly()) {
    return;
  }
  BotIKConfig config;
  std::string diagnostic;
  if (!ParseIKConfig(json, config, diagnostic)) {
    _editDiagnostic = "Cannot restore IK snapshot: " + diagnostic;
    return;
  }
  bool const rebind = config.outputBot != _ikAsset->GetConfig().outputBot ||
      config.inputBot != _ikAsset->GetConfig().inputBot;
  bool const outputBotPoseChanged =
      config.outputBotConfiguration != _ikAsset->GetConfig().outputBotConfiguration;
  bool const couplingsChanged = config.jointCouplings != _ikAsset->GetConfig().jointCouplings;
  _ikAsset->GetConfig() = std::move(config);
  SyncRecording();
  _lastUndoSnapshot = SerializeStudioIKConfig(_ikAsset->GetConfig());
  _pendingEdit = false;
  _editDiagnostic.clear();
  if (rebind) {
    PrepareOutputBot();
  } else {
    if ((outputBotPoseChanged || couplingsChanged) && _prepared) {
      StopPreview();
      _running = false;
      _previewDiagnostic.clear();
      if (outputBotPoseChanged) {
        _preview.SetOutputBotConfiguration(
            _ikAsset->GetConfig().outputBotConfiguration, _outputBotDiagnostic);
        ApplyPreviewPose();
      }
    }
    UpdateInputBotPose();
  }
  RebuildMarkers();
  if (selectionIndex == kInputBotSelection) {
    SelectInputBot();
  } else {
    SelectGoal(selectionIndex);
  }
  _ikAsset->SetDirty(!_savedSnapshot || _lastUndoSnapshot != *_savedSnapshot);
  // Do not gate undo on Run. The next UI-thread tick applies the restored settings.
}

void IKEditor::MarkEdited(bool discrete, bool structureChanged) {
  if (IsReadOnly()) {
    return;
  }
  _ikAsset->SetDirty(true);
  _pendingEdit = true;
  SyncRecording();
  UpdateInputBotPose();
  if (structureChanged) {
    RebuildMarkers();
  } else {
    UpdateMarkers();
  }
  ValidateIKConfig(_ikAsset->GetConfig(), _editDiagnostic);
  if (discrete) {
    PushUndoSnapshot();
  }
}

bool IKEditor::CommitJointCouplings(
    mochi::DynamicArray<robotics::BotIKJointCouplingConfig> couplings,
    bool discrete,
    std::optional<size_t> editedRow) {
  if (IsReadOnly()) {
    return false;
  }
  auto& config = _ikAsset->GetConfig();
  BotIKConfig candidate = config;
  candidate.jointCouplings = couplings;
  if (!ValidateIKConfig(candidate, _editDiagnostic)) {
    return false;
  }
  if (editedRow) {
    auto const& coupling = couplings[*editedRow];
    if (!_prepared && coupling.enabled && coupling.stiffness > 0) {
      _editDiagnostic = "Prepare a valid output bot before enabling a joint coupling.";
      return false;
    }
    if (_prepared) {
      mochi::DynamicString diagnostic;
      if (!robotics::ValidateIKJointCouplings(
              _preview.GetOutputBot(), {&coupling, 1}, diagnostic)) {
        _editDiagnostic.assign(diagnostic.begin(), diagnostic.end());
        return false;
      }
    }
  }
  if (config.jointCouplings == couplings) {
    return true;
  }
  if (discrete) {
    PushUndoSnapshot();
  }
  config.jointCouplings = std::move(couplings);
  // These are solve-only output constraints. Stop resets both preview displays/clock but retains
  // the authored pose and loaded input-bot clip; MarkEdited's ordinary sync does not reload the
  // file.
  StopPreview();
  MarkEdited(discrete);
  return true;
}

void IKEditor::PrepareOutputBot() {
  CancelRecording();
  _pendingCouplingA.clear();
  _pendingCouplingB.clear();
  // Output-bot edits, undo, and reference refresh all enter here before the next UI frame.
  auto& assets = _studio->GetAssetManager();
  assets.ResyncReferencer(_ikAsset);
  assets.ResyncReferencer(this);
  _running = false;
  _preview.Stop();
  _prepared = false;
  _outputBotDiagnostic.clear();
  _previewDiagnostic.clear();
  _authoredOutputBaseFrameFromLinkFrames.clear();
  _outputBotLinkIndices.clear();
  _outputBotStagedLinkNames.clear();
  _outputBotSnapshot.clear();
  _inputBotStage.Clear();
  _inputBotStagedLinkNames.clear();
  _worldFrameFromInputLinkFrames.clear();
  _inputBotDiagnostic.clear();
  _inputBotSnapshot.clear();
  _selectInputBotRoot = false;
  _observedInputBotReference = _ikAsset->GetConfig().inputBot;
  _preview.SetInputBot(nullptr, _inputBotDiagnostic);
  _observedOutputBotReference = _ikAsset->GetConfig().outputBot;
  _observedDocumentPath = _ikAsset->GetPath();
  if (_viewport) {
    _viewport->SetSelectedSceneObjects({}, false);
  }
  _outputBotStage.Clear();
  auto* outputBotAsset = _ikAsset->GetOutputBotAsset();
  if (!outputBotAsset) {
    _outputBotDiagnostic = _ikAsset->GetDiagnostic();
    if (_outputBotDiagnostic.empty()) {
      _outputBotDiagnostic = "Output bot could not be resolved. Select a Bot asset.";
    }
    UpdateMarkers();
    return;
  }
  outputBotAsset->Rebuild(_studio->GetBotLoader());
  if (!outputBotAsset->IsBuildOk() || !outputBotAsset->IsValidateOk()) {
    _outputBotDiagnostic = "The output bot failed rebuild or validation. Open Bot to inspect it.";
    UpdateMarkers();
    return;
  }
  auto const& outputBotPrefab = outputBotAsset->GetBotPrefab();
  _outputBotSnapshot = SReflect::ToJsonString(outputBotPrefab, false);
  if (!_preview.Prepare(outputBotPrefab, _outputBotDiagnostic)) {
    UpdateMarkers();
    return;
  }
  _preview.SetOutputBotConfiguration(
      _ikAsset->GetConfig().outputBotConfiguration, _outputBotDiagnostic);
  mochi::Error error;
  superdex::robotics::ComputeLinkTransformsFromPose(
      _preview.GetOutputBot(),
      mochi::MakeConstSpan(_preview.GetOutputBot().defaultPose),
      _authoredOutputBaseFrameFromLinkFrames,
      superdex::robotics::LinkTransformSpace::RootFromParent,
      error);
  if (!error.IsOK()) {
    _outputBotDiagnostic = error.GetDescription();
    _preview.Stop();
    UpdateMarkers();
    return;
  }
  auto const& names = _preview.GetOutputBotLinkNames();
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    _outputBotLinkIndices.emplace(names[i], i);
  }
  _prepared = true;
  PrepareInputBot();
  SyncRecording(true, true);
  RestageBots();
  UpdateMarkers();
}

void IKEditor::RestageBots() {
  _displayedInputPose.reset();
  if (!_prepared || !_viewport) {
    return;
  }
  // Robot restaging may destroy objects. Marker identity and input-bot selection remain
  // independent.
  _viewport->SetSelectedSceneObjects({}, false);
  _outputBotStage.StageBot(_preview.GetOutputBot(), _stageType, /*hideSkinCoveredLinks=*/false);
  UpdateFrameSize();
  if (auto const* inputBot = _preview.GetInputBot()) {
    auto staged = *inputBot;
    staged.name = std::string(inputBot->name) + " (IK Input Bot)";
    _inputBotStage.StageBot(staged, _stageType, /*hideSkinCoveredLinks=*/false);
    _inputBotStagedLinkNames.clear();
    for (auto const& link : inputBot->links) {
      _inputBotStagedLinkNames.push_back(std::string(staged.name) + "/" + std::string(link.name));
    }
    auto material =
        _studio->GetResourceManager().CreateLitSeeThroughMaterial({0.25f, 0.65f, 0.95f}, 0.35f);
    for (auto const& actor : _inputBotStage.GetActors()) {
      for (auto const* instance : {&actor.render, &actor.shape}) {
        if (instance->sceneObject) {
          instance->sceneObject->SetMaterial(material);
          instance->sceneObject->SetShadows(false, false);
        }
      }
    }
    UpdateInputBotPose();
  }
  _outputBotStagedLinkNames.clear();
  std::string const prefix = std::string(std::string_view(_preview.GetOutputBot().name)) + "/";
  for (auto const& name : _preview.GetOutputBotLinkNames()) {
    _outputBotStagedLinkNames.push_back(prefix + name);
  }
  ApplyPreviewPose();
  SyncViewportSelection();
  _viewport->UpdateGroundPlane();
}

void IKEditor::PrepareInputBot() {
  _displayedInputPose.reset();
  auto const& config = _ikAsset->GetConfig();
  if (config.inputBot.empty()) {
    return;
  }
  auto* asset = _ikAsset->GetInputBotAsset();
  if (!asset) {
    _inputBotDiagnostic = "Input bot could not be resolved. Select a Bot asset.";
    return;
  }
  asset->Rebuild(_studio->GetBotLoader());
  if (!asset->IsBuildOk() || !asset->IsValidateOk()) {
    _inputBotDiagnostic = "The input bot failed rebuild or validation.";
    return;
  }
  auto const& inputBot = asset->GetBotPrefab();
  if (_preview.SetInputBot(&inputBot, _inputBotDiagnostic)) {
    _inputBotSnapshot = SReflect::ToJsonString(inputBot, false);
  }
}

void IKEditor::CancelRecording() {
  _displayedInputPose.reset();
  if (_recordingTasks) {
    _recordingTasks->Cancel(1);
  }
  ++_recordingGeneration;
  _recordingClock.Stop();
  _recordingOwnsPose = false;
  _resumeRecording = false;
  _recordingPose.reset();
  _recordingClip.reset();
  _recordingLoad.reset();
  _recordingReference.clear();
  _recordingPath.clear();
  _recordingDiagnostic.clear();
}

void IKEditor::SyncRecording(bool force, bool checkFile) {
  auto const& memories = _studio->GetAppSettings().ikRecordings;
  auto const found = memories.find(_recordingDocumentKey);
  std::string const remembered = found == memories.end() ? std::string{} : found->second;
  if (remembered != _observedRememberedRecording) {
    RestoreRecordingSelection();
    force = true;
  }
  std::string const reference = _selectedRecording;
  std::string const path = reference;
  std::error_code error;
  auto stamp = _recordingStamp;
  bool hasStamp = _recordingHasStamp;
  bool const changedReference = reference != _recordingReference || path != _recordingPath;
  if (force || changedReference || checkFile) {
    stamp = std::filesystem::last_write_time(std::filesystem::u8path(path), error);
    hasStamp = !error;
  }
  if (force || changedReference ||
      (checkFile && (hasStamp != _recordingHasStamp || (hasStamp && stamp != _recordingStamp)))) {
    if (_running || _preview.HasSolver() || _recordingOwnsPose) {
      StopPreview();
    }
    CancelRecording();
    _recordingReference = reference;
    _recordingPath = path;
    _recordingStamp = stamp;
    _recordingHasStamp = hasStamp;
    if (reference.empty()) {
      return;
    }
    auto const* inputBot = _preview.GetInputBot();
    if (!_prepared || !inputBot) {
      _recordingDiagnostic = "Assign a valid input bot before loading a recording";
      return;
    }
    _studio->GetAssetManager().ResyncReferencer(_ikAsset);
    _studio->GetAssetManager().ResyncReferencer(this);
    if (!_recordingTasks) {
      _recordingTasks = std::make_unique<BackgroundTaskQueue>();
    }
    auto result = std::make_shared<RecordingLoad>();
    result->generation = _recordingGeneration;
    _recordingLoad = result;
    _recordingTasks->Enqueue(
        1, [result, path, inputBotSnapshot = *inputBot](std::atomic_bool const& cancelled) {
          try {
            result->clip =
                IKRecordingClip::Load(path, inputBotSnapshot, result->diagnostic, &cancelled);
          } catch (std::exception const& failure) {
            result->diagnostic = failure.what();
          } catch (...) {
            result->diagnostic = "Unexpected native recording load failure";
          }
          result->done.store(true, std::memory_order_release);
        });
  }
  if (_recordingLoad && _recordingLoad->done.load(std::memory_order_acquire)) {
    if (_recordingLoad->generation == _recordingGeneration) {
      _recordingClip = _recordingLoad->clip;
      _recordingDiagnostic = _recordingLoad->diagnostic;
      if (_recordingClip) {
        _recordingClock.Configure(*_recordingClip);
      }
    }
    _recordingLoad.reset();
  }
}

void IKEditor::SampleRecording() {
  _recordingPose.reset();
  if (!_recordingOwnsPose || !_recordingClip) {
    return;
  }
  if (_running && _active) {
    _recordingClock.Tick();
  }
  auto const* sample = _recordingClip->SampleAt(_recordingClock.GetCursor());
  if (!sample || !sample->hasPose) {
    return;
  }
  auto const& config = _ikAsset->GetConfig();
  _recordingPose = IKInputBotPose{
      mochi::MakeConstSpan(sample->inputBotConfiguration),
      _recordingClip->InputBotRoot(
          *sample,
          _preview.GetWorldFrameFromOutputBaseFrame() * config.outputBaseFrameFromInputBaseFrame,
          config.createParams.fixedBase)};
}

void IKEditor::RestoreRecordingSelection() {
  _selectionDiagnostic.clear();
  _recordingDocumentKey = IKDesignPathKey(_ikAsset->GetPath());
  auto const remembered =
      RememberedIKRecording(_studio->GetAppSettings().ikRecordings, _ikAsset->GetPath());
  _observedRememberedRecording = remembered;
  if (remembered.empty() || !kIKRecordingPlaybackAvailable) {
    _selectedRecording.clear();
    return;
  }
  std::error_code error;
  if (std::filesystem::is_regular_file(mochi::Path{remembered}.AsFilesystemPath(), error) &&
      !error) {
    _selectedRecording = remembered;
  } else {
    _selectedRecording.clear();
    _selectionDiagnostic =
        "Last-used RRD is unavailable; manual input-bot pose is active: " + remembered;
  }
}

void IKEditor::SelectRecording(mochi::Path const& path) {
  if (!path.IsEmpty()) {
    auto extension = path.AsFilesystemPath().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
      return char(std::tolower(c));
    });
    std::error_code error;
    if (extension != ".rrd" || !std::filesystem::is_regular_file(path.AsFilesystemPath(), error) ||
        error) {
      _recordingDiagnostic = "Select one existing .rrd recording for this Studio session.";
      return;
    }
    if (_ikAsset->GetConfig().inputBot.empty()) {
      _recordingDiagnostic = "Assign a input bot before selecting a session recording.";
      return;
    }
  }
  _selectionDiagnostic.clear();
  StopPreview();
  _selectedRecording = IKDesignPathKey(path);
  if (RememberIKRecording(_studio->GetAppSettings().ikRecordings, _ikAsset->GetPath(), path)) {
    _studio->SaveSettings();
  }
  _recordingDocumentKey = IKDesignPathKey(_ikAsset->GetPath());
  _observedRememberedRecording =
      RememberedIKRecording(_studio->GetAppSettings().ikRecordings, _ikAsset->GetPath());
  SyncRecording(true, true);
  UpdateInputBotPose();
  UpdateMarkers();
}

void IKEditor::ShowRecordingDropTarget() {
  ImVec2 const position = ImGui::GetWindowPos();
  ImRect const area{position, position + ImGui::GetWindowSize()};
  if (!ImGui::BeginDragDropTargetCustom(area, ImGui::GetID("SessionRecordingDrop"))) {
    return;
  }
  auto const* peek =
      ImGui::AcceptDragDropPayload(kAssetBrowserDragDropType, ImGuiDragDropFlags_AcceptPeekOnly);
  if (peek && peek->DataSize == sizeof(AssetBrowserDragDropPayload*)) {
    auto const* payload = *static_cast<AssetBrowserDragDropPayload* const*>(peek->Data);
    std::string extension = payload && payload->items.size() == 1
        ? payload->items[0].fullPath.AsFilesystemPath().extension().string()
        : std::string{};
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
      return char(std::tolower(c));
    });
    bool const valid = kIKRecordingPlaybackAvailable && payload && payload->items.size() == 1 &&
        !payload->items[0].isFolder && extension == ".rrd" &&
        !_ikAsset->GetConfig().inputBot.empty();
    if (valid) {
      if (ImGui::AcceptDragDropPayload(kAssetBrowserDragDropType)) {
        SelectRecording(payload->items[0].fullPath);
      }
    } else {
      ImGui::SetMouseCursor(ImGuiMouseCursor_NotAllowed);
    }
  }
  ImGui::EndDragDropTarget();
}

void IKEditor::ShowRecordingControls() {
  auto& config = _ikAsset->GetConfig();
  ImGui::SeparatorText("Session Recording");
  if constexpr (!kIKRecordingPlaybackAvailable) {
    ImGui::BeginDisabled();
    ImGui::Button("Browse RRD");
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      ImGui::SetTooltip("%s", kIKRecordingPlaybackUnavailable);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Playback unavailable in this build");
    return;
  }
  ImGui::TextWrapped(
      "Drop an RRD from the Asset Browser onto the viewport. This design-only choice is remembered locally, not saved in the IK document.");
  std::string reference = _selectedRecording;
  ImGui::InputText("RRD Path", &reference, ImGuiInputTextFlags_ReadOnly);
  ImGui::BeginDisabled(config.inputBot.empty());
  if (ImGui::Button("Browse RRD")) {
    char const* filters[] = {"*.rrd"};
    auto path = SuperDexStudio::GetFileDialogPath(
        "Studio Session Recording",
        filters,
        1,
        "RRD recording (*.rrd)",
        false,
        mochi::Path{_selectedRecording});
    if (!path.IsEmpty()) {
      SelectRecording(path);
    }
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(
      _selectedRecording.empty() &&
      RememberedIKRecording(_studio->GetAppSettings().ikRecordings, _ikAsset->GetPath()).empty());
  if (ImGui::Button("Clear RRD")) {
    SelectRecording({});
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(_selectedRecording.empty() || !_preview.GetInputBot());
  if (ImGui::Button("Reload RRD")) {
    SyncRecording(true, true);
  }
  ImGui::EndDisabled();
  if (_recordingLoad) {
    ImGui::TextDisabled("Loading native RRD (eager read)...");
  }
  if (!_selectionDiagnostic.empty()) {
    ImGui::TextWrapped("%s", _selectionDiagnostic.c_str());
  }
  if (!_recordingDiagnostic.empty()) {
    ImGui::TextWrapped("Recording: %s", _recordingDiagnostic.c_str());
  }
  if (_recordingClip) {
    ImGui::TextWrapped("%s", _recordingClip->GetCompatibility().c_str());
    double const seconds = static_cast<double>(
                               static_cast<uint64_t>(_recordingClock.GetCursor()) -
                               static_cast<uint64_t>(_recordingClip->GetFirstTime())) /
        1e9;
    ImGui::Text("%.3f / %.3f s | 1x looping", seconds, _recordingClip->GetDurationSeconds());
    ImGui::TextDisabled(
        config.createParams.fixedBase ? "Joint-only root (Fixed Base)"
                                      : "Simulated wrist-relative root (Floating Base)");
    if (_recordingOwnsPose) {
      auto const* sample = _recordingClip->SampleAt(_recordingClock.GetCursor());
      ImGui::TextWrapped(
          !sample || !sample->hasPose
              ? "No complete articulation snapshot: authored input-bot pose retained"
              : "Physics-scene articulation pose (simulated; no raw tracking confidence)");
      ImGui::TextDisabled("Stop restores pose authoring. Pause retains both previews.");
    }
    bool const bound = std::any_of(config.goals.begin(), config.goals.end(), [](auto const& goal) {
      return goal.enabled && !goal.inputBotLink.empty() &&
          (goal.positionStiffness > 0 || goal.rotationStiffness > 0);
    });
    if (!bound) {
      ImGui::TextWrapped(
          "Warning: no enabled IK goals are bound to input-bot links; recording motion will not drive output IK.");
    }
  } else if (_selectedRecording.empty()) {
    ImGui::TextDisabled("Manual input-bot pose. Assign a input bot to select an RRD.");
  }
}

void IKEditor::UpdateInputBotPose() {
  _displayedInputPose.reset();
  auto const* inputBot = _preview.GetInputBot();
  if (!_prepared || !inputBot) {
    return;
  }
  auto const& config = _ikAsset->GetConfig();
  SampleRecording();
  auto const worldFrameFromInputBaseFrame = _recordingPose
      ? _recordingPose->worldFrameFromInputBaseFrame
      : _preview.GetWorldFrameFromOutputBaseFrame() * config.outputBaseFrameFromInputBaseFrame;
  auto const inputBotConfiguration = _recordingPose
      ? _recordingPose->inputBotConfiguration
      : mochi::MakeConstSpan(config.inputBotConfiguration);
  mochi::Error error;
  robotics::ComputeIKInputBotTransforms(
      *inputBot,
      inputBotConfiguration,
      worldFrameFromInputBaseFrame,
      config.inputBotScale,
      _worldFrameFromInputLinkFrames,
      error);
  if (!error.IsOK()) {
    _inputBotDiagnostic = error.GetDescription();
    _worldFrameFromInputLinkFrames.clear();
    return;
  }
  _inputBotDiagnostic.clear();
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  mochi::Real3 const scale{config.inputBotScale, config.inputBotScale, config.inputBotScale};
  for (int i = 0; i < mochi::isize(_inputBotStagedLinkNames); ++i) {
    _inputBotStage.ApplyWorldTransform(
        _inputBotStagedLinkNames[i], _worldFrameFromInputLinkFrames[i], scale, converter);
  }
  if (inputBot->skin) {
    SkinnedPoseUpdate update;
    update.name = SkinStagedName(std::string(inputBot->name) + " (IK Input Bot)");
    mochi::DynamicArray<mochi::TransformRT> worldFrameFromInputBoneFrames;
    if (MapIKSkinTransforms(
            *inputBot,
            _worldFrameFromInputLinkFrames,
            worldFrameFromInputBoneFrames,
            _inputBotDiagnostic)) {
      update.linkWorldTransforms.assign(
          worldFrameFromInputBoneFrames.begin(), worldFrameFromInputBoneFrames.end());
      int const index = _inputBotStage.FindIndexByName(update.name);
      if (index >= 0) {
        auto inputBaseFrameFromSkinFrame = mochi::Invert(inputBot->worldFromRoot) *
            _inputBotStage.GetActors()[index].defaultWorldTransform;
        inputBaseFrameFromSkinFrame.SetTranslation(
            inputBaseFrameFromSkinFrame.GetTranslation() * config.inputBotScale);
        _inputBotStage.ApplyWorldTransform(
            update.name,
            worldFrameFromInputBaseFrame * inputBaseFrameFromSkinFrame,
            scale,
            converter);
      }
      std::array<SkinnedPoseUpdate, 1> const updates{std::move(update)};
      _inputBotStage.ApplySkinnedPose(mochi::MakeConstSpan(updates), converter);
    }
  }
  bool const staged = !_worldFrameFromInputLinkFrames.empty() &&
      _inputBotStagedLinkNames.size() == inputBot->links.size() &&
      _worldFrameFromInputLinkFrames.size() == inputBot->links.size() &&
      std::any_of(
          _inputBotStage.GetActors().begin(),
          _inputBotStage.GetActors().end(),
          [](auto const& actor) { return actor.sceneObject != nullptr; });
  if (staged && _inputBotDiagnostic.empty()) {
    // Capture exactly the q/root just applied above. Do not sample the playback clock on click.
    auto const positions = inputBotConfiguration.empty()
        ? mochi::MakeConstSpan(inputBot->defaultPose)
        : inputBotConfiguration;
    _displayedInputPose = IKInputPoseSnapshot{
        mochi::DynamicArray<mochi::real>{positions}, _worldFrameFromInputLinkFrames.front()};
  }
}

bool IKEditor::CanSnapshotInputBotPose(BotIKConfig& candidate, std::string& diagnostic) const {
  diagnostic.clear();
  if (IsReadOnly()) {
    diagnostic = "Read-only document";
    return false;
  }
  auto const& config = _ikAsset->GetConfig();
  if (!_prepared || !_preview.GetInputBot() || !_displayedInputPose || _inputBotSnapshot.empty() ||
      _observedInputBotReference != config.inputBot ||
      _observedOutputBotReference != config.outputBot ||
      _observedDocumentPath != _ikAsset->GetPath()) {
    diagnostic = "No complete input viewport pose for the current bot and document";
    return false;
  }
  // Inspect only existing live assets: no rebuild/load from a capture availability check.
  // Same-reference model edits must not let an old staged pose be authored against a new model.
  auto const* inputAsset =
      _studio->GetAssetManager().FindAssetByPath<BotAsset>(_ikAsset->GetInputBotPath());
  auto const* outputAsset =
      _studio->GetAssetManager().FindAssetByPath<BotAsset>(_ikAsset->GetOutputBotPath());
  if (!inputAsset || !outputAsset || !inputAsset->IsBuildOk() || !inputAsset->IsValidateOk() ||
      !outputAsset->IsBuildOk() || !outputAsset->IsValidateOk() ||
      SReflect::ToJsonString(inputAsset->GetBotPrefab(), false) != _inputBotSnapshot ||
      SReflect::ToJsonString(outputAsset->GetBotPrefab(), false) != _outputBotSnapshot) {
    diagnostic = "Input or output model changed; refresh the IK editor before taking a snapshot";
    return false;
  }
  if (!config.outputBotConfiguration.empty() &&
      mochi::isize(config.outputBotConfiguration) != _preview.GetOutputBot()._numDofs) {
    diagnostic = "Output joint configuration does not match the prepared bot";
    return false;
  }
  return BuildIKInputPoseSnapshot(
      config,
      _preview.GetInputBot(),
      *_displayedInputPose,
      _preview.GetWorldFrameFromOutputBaseFrame(),
      candidate,
      diagnostic);
}

void IKEditor::SnapshotInputBotPose() {
  if (IsReadOnly()) {
    return;
  }
  BotIKConfig candidate;
  std::string diagnostic;
  if (!CanSnapshotInputBotPose(candidate, diagnostic)) {
    _editDiagnostic = std::move(diagnostic);
    return;
  }
  auto& config = _ikAsset->GetConfig();
  bool const changed = config.inputBotConfiguration != candidate.inputBotConfiguration ||
      config.outputBaseFrameFromInputBaseFrame != candidate.outputBaseFrameFromInputBaseFrame;
  if (changed) {
    PushUndoSnapshot(); // Commit any preceding drag separately from this atomic input-only edit.
    config.inputBotConfiguration = std::move(candidate.inputBotConfiguration);
    config.outputBaseFrameFromInputBaseFrame = candidate.outputBaseFrameFromInputBaseFrame;
  }
  // Do not StopPreview: its output-seed restoration would discard the visible output solve.
  // Restarting from the first cursor also avoids applying the captured wrist delta twice.
  _recordingClock.Stop();
  _recordingOwnsPose = false;
  _recordingPose.reset();
  _resumeRecording = false;
  _running = false;
  if (changed) {
    MarkEdited(true);
  } else {
    UpdateInputBotPose();
    UpdateMarkers();
  }
}

std::optional<mochi::TransformRT> IKEditor::GoalConstraintAnchor(
    BotIKGoalConfig const& goal) const {
  if (goal.inputBotLink.empty()) {
    return _preview.GetWorldFrameFromOutputBaseFrame();
  }
  auto const* inputBot = _preview.GetInputBot();
  if (inputBot && _worldFrameFromInputLinkFrames.size() == inputBot->links.size()) {
    for (int i = 0; i < mochi::isize(inputBot->links); ++i) {
      if (inputBot->links[i].name == std::string_view(goal.inputBotLink)) {
        return _worldFrameFromInputLinkFrames[i];
      }
    }
  }
  return std::nullopt;
}

std::optional<mochi::TransformRT> IKEditor::GoalWorldConstraintFrame(
    BotIKGoalConfig const& goal) const {
  if (!_prepared) {
    return std::nullopt;
  }
  mochi::TransformRT worldFrameFromConstraintFrame;
  mochi::Error error;
  robotics::ComputeIKGoalWorldConstraintTransform(
      goal,
      _preview.GetWorldFrameFromOutputBaseFrame(),
      _preview.GetInputBot(),
      mochi::MakeConstSpan(_worldFrameFromInputLinkFrames),
      _ikAsset->GetConfig().inputBotScale,
      worldFrameFromConstraintFrame,
      error);
  return error.IsOK() ? std::optional{worldFrameFromConstraintFrame} : std::nullopt;
}

void IKEditor::UpdateFrameSize() {
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  filament::Box bounds;
  bool hasBounds = false;
  for (auto const& actor : _outputBotStage.GetActors()) {
    if (!actor.sceneObject) {
      continue;
    }
    auto const box = actor.sceneObject->GetAABB();
    if (box.isEmpty()) {
      continue;
    }
    bool finite = true;
    for (int axis = 0; axis < 3; ++axis) {
      finite &= std::isfinite(box.center[axis]) && std::isfinite(box.halfExtent[axis]);
    }
    if (finite) {
      if (hasBounds) {
        bounds.unionSelf(box);
      } else {
        bounds = box;
        hasBounds = true;
      }
    }
  }
  mochi::real size = mochi::real(0);
  if (hasBounds) {
    auto const extent = bounds.halfExtent * 2.0f;
    size = mochi::real(std::max({extent.x, extent.y, extent.z})) / mochi::Abs(converter.GetScale());
  }
  if (!(size > mochi::real(0))) {
    // Shapeless bots still have useful kinematic extents. Cache before the solved pose is applied.
    if (!_authoredOutputBaseFrameFromLinkFrames.empty()) {
      auto lo = _authoredOutputBaseFrameFromLinkFrames.front().GetTranslation();
      auto hi = lo;
      for (auto const& link : _authoredOutputBaseFrameFromLinkFrames) {
        auto const p = link.GetTranslation();
        for (int axis = 0; axis < 3; ++axis) {
          lo[axis] = std::min(lo[axis], p[axis]);
          hi[axis] = std::max(hi[axis], p[axis]);
        }
      }
      auto const extent = hi - lo;
      size = std::max({extent[0], extent[1], extent[2]}) * _preview.GetOutputBot().scale;
    }
  }
  _outputLinkAxisLength = mochi::IsFinite(size) && size > mochi::real(0)
      ? size * mochi::real(0.1)
      : kDefaultOutputLinkAxisLength;
}

void IKEditor::ClearMarkers() {
  if (!_viewport) {
    return;
  }
  _viewport->SetSelectedSceneObjects({}, false);
  _markerGoals.clear();
  for (auto* marker : _markers) {
    if (marker) {
      _viewport->GetRenderScene()->DestroySceneObject(marker);
    }
  }
  _markers.clear();
}

void IKEditor::RebuildMarkers() {
  if (!_viewport) {
    return;
  }
  ClearMarkers();
  auto const& goals = _ikAsset->GetConfig().goals;
  _markers.reserve(goals.size());
  for (int i = 0; i < static_cast<int>(goals.size()); ++i) {
    auto mesh = CreateGoalMarker(_studio->GetEngine(), _markerMaterial);
    auto* marker = _viewport->GetRenderScene()->AddSceneObjectToScene(std::move(mesh));
    _markers.push_back(marker);
    if (marker) {
      marker->SetShadows(false, false);
      _markerGoals.emplace(marker, i);
    }
  }
  UpdateMarkers();
  SyncViewportSelection();
}

void IKEditor::UpdateMarkers() {
  if (!_viewport) {
    return;
  }
  auto const& goals = _ikAsset->GetConfig().goals;
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  for (int i = 0; i < static_cast<int>(_markers.size()); ++i) {
    auto* marker = _markers[i];
    if (!marker || i >= static_cast<int>(goals.size())) {
      continue;
    }
    auto const& goal = goals[i];
    auto const worldFrameFromConstraintFrame = GoalWorldConstraintFrame(goal);
    marker->SetVisible(_showGoals && worldFrameFromConstraintFrame.has_value());
    if (!worldFrameFromConstraintFrame) {
      continue;
    }
    marker->SetName("IK Goal: " + std::string(goal.name));
    mochi::real const markerScale =
        _outputLinkAxisLength * kGoalAxisRatio / kDefaultOutputLinkAxisLength;
    marker->SetLocalTransform(
        *worldFrameFromConstraintFrame,
        mochi::Real3{markerScale, markerScale, markerScale},
        &converter);
    bool const disabled =
        !goal.enabled || (goal.positionStiffness == 0 && goal.rotationStiffness == 0);
    marker->SetMaterial(
        i == _selectedGoal ? _selectedMarkerMaterial
                           : (disabled ? _disabledMarkerMaterial : _markerMaterial));
  }
}

void IKEditor::DrawGoals() {
  if (!_prepared || !_showGoals) {
    return;
  }
  auto* draw = _viewport->GetRenderScene()->GetDebugDraw();
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  auto const& goals = _ikAsset->GetConfig().goals;
  auto const& links = _preview.GetOutputBotWorldFromLinks();
  constexpr std::array<filament::math::float4, 3> linkColors = {
      filament::math::float4{0.9f, 0.18f, 0.18f, 1},
      filament::math::float4{0.5f, 0.8f, 0.06f, 1},
      filament::math::float4{0.18f, 0.53f, 0.91f, 1}};
  auto goalColors = linkColors;
  for (auto& color : goalColors) {
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.x, color.y, color.z, h, s, v);
    ImGui::ColorConvertHSVtoRGB(std::fmod(h + 1.0f / 12.0f, 1.0f), s, v, color.x, color.y, color.z);
  }
  auto drawFrame = [&](mochi::TransformRT const& worldFrameFromConstraintFrame,
                       mochi::real length,
                       std::array<filament::math::float4, 3> const& colors,
                       bool disabled) {
    if (!draw || !mochi::IsFinite(worldFrameFromConstraintFrame)) {
      return;
    }
    auto const origin = mochi_renderer::ToFilament<float>(
        converter.TranslationToOutput(worldFrameFromConstraintFrame.GetTranslation()));
    float const radius = static_cast<float>(length * mochi::Abs(converter.GetScale())) * 0.025f;
    for (int axis = 0; axis < 3; ++axis) {
      mochi::Real3 direction{};
      direction[axis] = length;
      auto const vector = mochi_renderer::ToFilament<float>(
          converter.DirectionToOutput(worldFrameFromConstraintFrame.TransformDirection(direction)));
      auto color = colors[axis];
      if (disabled) {
        color = {color.x * 0.4f, color.y * 0.4f, color.z * 0.4f, 1};
      }
      draw->DrawSolidCylinder(origin, vector, radius, color, 8, true, true);
    }
  };
  for (int i = 0; i < static_cast<int>(goals.size()); ++i) {
    auto const& goal = goals[i];
    bool const disabled =
        !goal.enabled || (goal.positionStiffness == 0 && goal.rotationStiffness == 0);
    mochi::real const length = _outputLinkAxisLength * mochi::real(i == _selectedGoal ? 1.3 : 1.0);
    int const link = FindOutputBotLink(goal.outputBotLink);
    if (link >= 0 && link < mochi::isize(links)) {
      drawFrame(links[link] * goal.linkFrameFromConstraintFrame, length, linkColors, disabled);
    }
    auto const worldFrameFromConstraintFrame = GoalWorldConstraintFrame(goal);
    if (!worldFrameFromConstraintFrame) {
      continue;
    }
    drawFrame(*worldFrameFromConstraintFrame, length * kGoalAxisRatio, goalColors, disabled);
    auto const origin = mochi_renderer::ToFilament<float>(
        converter.TranslationToOutput(worldFrameFromConstraintFrame->GetTranslation()));
    auto const labelColor = i == _selectedGoal
        ? filament::math::float4{kViewportSelectionColor, 1}
        : (disabled ? filament::math::float4{0.55f, 0.55f, 0.55f, 1}
                    : filament::math::float4{1, 1, 1, 1});
    std::string label{goal.name};
    if (disabled) {
      label += " (disabled)";
    }
    if (link < 0) {
      label += " (missing link)";
    }
    _viewport->GetDebugText()->Draw(origin, label, labelColor, {0, -20});
  }
}

void IKEditor::ApplyPreviewPose() {
  auto const& transforms = _preview.GetOutputBotWorldFromLinks();
  auto const& names = _preview.GetOutputBotLinkNames();
  if (transforms.size() != names.size() || transforms.empty()) {
    return;
  }
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  _outputBotStage.ApplyWorldTransforms(
      mochi::MakeConstSpan(_outputBotStagedLinkNames), mochi::MakeConstSpan(transforms), converter);
  auto const& outputBot = _preview.GetOutputBot();
  if (!outputBot.skin.has_value()) {
    return;
  }
  SkinnedPoseUpdate update;
  update.name = SkinStagedName(superdex::robotics::GetBotSkinName(outputBot));
  mochi::DynamicArray<mochi::TransformRT> worldFrameFromBones;
  if (!MapIKSkinTransforms(
          outputBot, mochi::MakeConstSpan(transforms), worldFrameFromBones, _previewDiagnostic)) {
    _running = false;
    return;
  }
  update.linkWorldTransforms.assign(worldFrameFromBones.begin(), worldFrameFromBones.end());
  int const skinIndex = _outputBotStage.FindIndexByName(update.name);
  if (skinIndex >= 0 && !_authoredOutputBaseFrameFromLinkFrames.empty()) {
    // Retain the staged bind frame and inverse binds. Only the solved root's rigid displacement
    // moves that frame; following current bone 0 would latch a new, incompatible skin origin.
    auto const worldFrameFromSolvedOutputBaseFrame =
        transforms.front() * mochi::Invert(_authoredOutputBaseFrameFromLinkFrames.front());
    auto const displacement = worldFrameFromSolvedOutputBaseFrame *
        mochi::Invert(_preview.GetWorldFrameFromOutputBaseFrame());
    _outputBotStage.ApplyWorldTransform(
        update.name,
        displacement * _outputBotStage.GetActors()[skinIndex].defaultWorldTransform,
        converter);
  }
  std::array<SkinnedPoseUpdate, 1> const updates{std::move(update)};
  _outputBotStage.ApplySkinnedPose(mochi::MakeConstSpan(updates), converter);
}

void IKEditor::StopPreview() {
  _recordingClock.Stop();
  _recordingOwnsPose = false;
  _recordingPose.reset();
  _resumeRecording = false;
  _running = false;
  _preview.Stop();
  _previewDiagnostic.clear();
  UpdateInputBotPose();
  if (_viewport) {
    _outputBotStage.ResetWorldTransforms(_studio->GetEditorToRendererSpaceConverter());
    ApplyPreviewPose();
    UpdateMarkers();
  }
}

bool IKEditor::CanPreview(std::string& diagnostic) const {
  if (!_prepared) {
    diagnostic = _outputBotDiagnostic;
    return false;
  }
  if (!_selectedRecording.empty() && !_recordingClip) {
    diagnostic = _recordingLoad ? "Loading input-bot recording" : _recordingDiagnostic;
    if (diagnostic.empty()) {
      diagnostic = "Input-bot recording is not ready";
    }
    return false;
  }
  auto pose = _recordingPose;
  if (!pose && _recordingClip) {
    auto const* sample = _recordingClip->SampleAt(_recordingClock.GetCursor());
    if (sample && sample->hasPose) {
      auto const& config = _ikAsset->GetConfig();
      pose = IKInputBotPose{
          mochi::MakeConstSpan(sample->inputBotConfiguration),
          _recordingClip->InputBotRoot(
              *sample,
              _preview.GetWorldFrameFromOutputBaseFrame() *
                  config.outputBaseFrameFromInputBaseFrame,
              config.createParams.fixedBase)};
    }
  }
  return _preview.CanSolve(_ikAsset->GetConfig(), diagnostic, pose ? &*pose : nullptr);
}

void IKEditor::SolveOnce() {
  if (!_prepared) {
    _running = false;
    return;
  }
  auto const& config = _ikAsset->GetConfig();
  // The session recreates creation-only settings safely; do not reset the last valid display
  // before a successful Step (including when creating the replacement solver fails).
  std::string diagnostic;
  UpdateInputBotPose();
  if (!_preview.Step(
          _studio->GetMochiContext(),
          _studio->GetBotLoader(),
          config,
          diagnostic,
          _recordingPose ? &*_recordingPose : nullptr)) {
    _running = false;
    _recordingClock.Pause();
    _previewDiagnostic = diagnostic;
    return;
  }
  _previewDiagnostic.clear();
  ApplyPreviewPose();
  RecordSolveSample();
}

void IKEditor::OnHandleInputs() {
  if (!ImGui::GetIO().WantTextInput && !ImGui::IsAnyItemActive()) {
    std::string diagnostic;
    HandlePlayToolbarAction(GetPlayToolbarHotkey(
        _running || _preview.HasSolver() || _recordingOwnsPose, _running, CanPreview(diagnostic)));
  }
  if (!IsReadOnly() && !ImGui::GetIO().WantTextInput && HasGoalSelection() &&
      ImGui::IsKeyChordPressed(ImGuiKey_Delete)) {
    RemoveGoal();
  }
}

void IKEditor::CommitPendingEdits() {
  if (_pendingEdit) {
    PushUndoSnapshot();
  }
}

void IKEditor::OnRender(Renderer const* renderer) {
  CaptureSavedSnapshot();
  if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGuizmo::IsUsing() &&
      !ImGui::IsAnyItemActive()) {
    CommitPendingEdits();
  }
  if (!_viewport) {
    return;
  }
  SyncRecording();
  auto const now = Clock::now();
  if (_running && _active && now >= _nextSolve) {
    // Schedule from this UI frame, not the previous deadline. One due solve, no catch-up loop.
    _nextSolve = now +
        std::chrono::duration_cast<Clock::duration>(
                     std::chrono::duration<double>(kPreviewIntervalSeconds));
    SolveOnce();
  }
  UpdateMarkers();
  DrawGoals();
  // The viewport handles highlights for the selected bot; reconcile the other stage too.
  if (_selectInputBotRoot) {
    _outputBotStage.UpdateHighlights();
  } else {
    _inputBotStage.UpdateHighlights();
  }
  _viewport->RenderScene(renderer);
}

void IKEditor::Shutdown() {
  CancelRecording();
  _running = false;
  _active = false;
  _preview.Stop();
  ClearMarkers();
  _outputBotStage.Clear();
  _inputBotStage.Clear();
  _outputBotStage.BindRenderScene(nullptr);
  _inputBotStage.BindRenderScene(nullptr);
  _viewport.reset();
  _markerMaterial.reset();
  _selectedMarkerMaterial.reset();
  _disabledMarkerMaterial.reset();
  _prepared = false;
}

void IKEditor::OnActivate() {
  CaptureSavedSnapshot();
  _active = true;
  // Look up by path each time: another editor may have changed the authored output bot in memory.
  auto* outputBot = _ikAsset->GetOutputBotAsset();
  if (outputBot) {
    outputBot->Rebuild(_studio->GetBotLoader());
    if (!outputBot->IsBuildOk() || !outputBot->IsValidateOk()) {
      _running = false;
      _preview.Stop();
      _prepared = false;
      _outputBotDiagnostic = "The output bot failed rebuild or validation. Open Bot to inspect it.";
      return;
    }
  }
  auto* inputBot = _ikAsset->GetInputBotAsset();
  if (inputBot) {
    inputBot->Rebuild(_studio->GetBotLoader());
  }
  bool const inputBotChanged = inputBot
      ? (!inputBot->IsBuildOk() || !inputBot->IsValidateOk() ||
         SReflect::ToJsonString(inputBot->GetBotPrefab(), false) != _inputBotSnapshot)
      : _preview.GetInputBot() != nullptr;
  if (!_prepared || _observedOutputBotReference != _ikAsset->GetConfig().outputBot ||
      _observedInputBotReference != _ikAsset->GetConfig().inputBot || inputBotChanged ||
      _observedDocumentPath != _ikAsset->GetPath() ||
      (outputBot &&
       SReflect::ToJsonString(outputBot->GetBotPrefab(), false) != _outputBotSnapshot) ||
      (!outputBot && _prepared)) {
    if (_observedDocumentPath != _ikAsset->GetPath()) {
      RestoreRecordingSelection();
    }
    PrepareOutputBot();
  }
  SyncRecording(false, true);
  if (_resumeRecording && _recordingClip) {
    _resumeRecording = false;
    _running = true;
    _recordingClock.Play();
    _nextSolve = Clock::now();
  }
}

void IKEditor::OnDeactivate() {
  _resumeRecording = _running && _recordingOwnsPose;
  _recordingClock.Pause();
  _active = false;
  _running = false;
  PushUndoSnapshot();
}

void IKEditor::Refresh() {
  if (!_viewport) {
    return;
  }
  // Recording choice is Studio state, not part of authored snapshots or asset references.
  RestoreRecordingSelection();
  PrepareOutputBot();
  RebuildMarkers();
  // A reference rewrite changes snapshot paths out-of-band; discard stale path-bearing history.
  ResetUndoBaseline();
}

bool IKEditor::CanUndoRedo() const {
  return !IsReadOnly();
}

Viewport* IKEditor::GetViewport() {
  return _viewport.get();
}

void IKEditor::ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& settings) {
  if (_viewport) {
    _viewport->GetRenderScene()->ApplyViewSettings(settings);
  }
}

void IKEditor::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  AssetEditor::ForEachReferencedPath(callback);
  _ikAsset->ForEachReferencedPath(callback);
}

int IKEditor::FindOutputBotLink(std::string_view name) const {
  auto const it = _outputBotLinkIndices.find(std::string(name));
  return it == _outputBotLinkIndices.end() ? -1 : it->second;
}

bool IKEditor::OutputBotLinkUsed(std::string_view name, int except) const {
  auto const& goals = _ikAsset->GetConfig().goals;
  for (int i = 0; i < static_cast<int>(goals.size()); ++i) {
    if (i != except && goals[i].outputBotLink == name) {
      return true;
    }
  }
  return false;
}

bool IKEditor::NameUsed(std::string_view name, int except) const {
  auto const& goals = _ikAsset->GetConfig().goals;
  for (int i = 0; i < static_cast<int>(goals.size()); ++i) {
    if (i != except && goals[i].name == name) {
      return true;
    }
  }
  return false;
}

void IKEditor::AddGoal(std::string const& outputBotLink) {
  int const index = FindOutputBotLink(outputBotLink);
  if (IsReadOnly() || index < 0 || OutputBotLinkUsed(outputBotLink) ||
      index >= static_cast<int>(_authoredOutputBaseFrameFromLinkFrames.size())) {
    return;
  }
  PushUndoSnapshot();
  BotIKGoalConfig goal;
  goal.name = outputBotLink;
  for (int suffix = 2; NameUsed(goal.name); ++suffix) {
    goal.name = outputBotLink + "_" + std::to_string(suffix);
  }
  goal.enabled = true;
  goal.outputBotLink = outputBotLink;
  goal.positionStiffness = kIKInitialPositionStiffness;
  goal.rotationStiffness = kIKInitialRotationStiffness;
  SetIKGoalConstraintTransform(
      goal, _authoredOutputBaseFrameFromLinkFrames[index] * goal.linkFrameFromConstraintFrame);
  auto& goals = _ikAsset->GetConfig().goals;
  goals.push_back(std::move(goal));
  _selectedGoal = static_cast<int>(goals.size()) - 1;
  _selectInputBotRoot = false;
  _nameBufferGoal = -1;
  MarkEdited(true, true);
}

void IKEditor::RemoveGoal() {
  if (IsReadOnly() || !HasGoalSelection()) {
    return;
  }
  PushUndoSnapshot();
  auto& goals = _ikAsset->GetConfig().goals;
  goals.erase(goals.begin() + _selectedGoal);
  _selectedGoal = std::min(_selectedGoal, static_cast<int>(goals.size()) - 1);
  _nameBufferGoal = -1;
  MarkEdited(true, true);
}

void IKEditor::ResetGoal() {
  if (IsReadOnly() || !_prepared || !HasGoalSelection()) {
    return;
  }
  auto& goal = _ikAsset->GetConfig().goals[_selectedGoal];
  int const index = FindOutputBotLink(goal.outputBotLink);
  if (index < 0 || index >= static_cast<int>(_authoredOutputBaseFrameFromLinkFrames.size())) {
    return;
  }
  auto const initialConstraintFrame =
      _authoredOutputBaseFrameFromLinkFrames[index] * goal.linkFrameFromConstraintFrame;
  if (!mochi::IsFinite(initialConstraintFrame)) {
    _editDiagnostic = "The attachment frame produces a non-finite initial goal pose.";
    return;
  }
  PushUndoSnapshot();
  auto const worldFrameFromBaseOrInputLinkFrame = GoalConstraintAnchor(goal);
  if (!worldFrameFromBaseOrInputLinkFrame) {
    return;
  }
  SetIKGoalConstraintTransform(
      goal,
      robotics::IKGoalLocalConstraintTransform(
          goal,
          *worldFrameFromBaseOrInputLinkFrame,
          _preview.GetWorldFrameFromOutputBaseFrame() * initialConstraintFrame,
          _ikAsset->GetConfig().inputBotScale));
  MarkEdited(true);
}

std::vector<AssetEditor::WindowDeclaration> IKEditor::GetDefaultWindows() {
  return {
      {"IK Goals", true, DockRegion::SidePanelTop, false},
      {"IK Solver", true, DockRegion::SidePanelBottom, false},
      {"IK Solve Plots", true, DockRegion::BottomBar, false},
      MeasureWindowDeclaration()};
}

std::vector<AssetEditor::WindowDeclaration> IKEditor::GetAuxiliaryWindows() const {
  return GetDefaultWindows();
}

void IKEditor::ShowAuxiliaryWindows() {
  CaptureSavedSnapshot();
  if (bool& open = _studio->GetWindowVisible("IK Goals")) {
    ShowGoalsWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("IK Solver")) {
    ShowSolverWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("IK Solve Plots")) {
    ShowSolvePlotsWindow(&open);
  }
  ShowMeasureWindow();
}

void IKEditor::ShowGoalsWindow(bool* open) {
  if (!ImGui::Begin("IK Goals", open)) {
    ImGui::End();
    return;
  }
  if (IsReadOnly()) {
    ImGui::TextDisabled("Read-only document: inspection and preview are available.");
  }
  ImGui::TextWrapped(
      "Larger RGB axes: output constraint frame. Smaller hue-shifted axes: resolved input constraint frame.");
  auto& config = _ikAsset->GetConfig();
  for (int i = 0; i < static_cast<int>(config.goals.size()); ++i) {
    auto& goal = config.goals[i];
    ImGui::PushID(i);
    ImGui::BeginDisabled(IsReadOnly());
    bool enabled = goal.enabled;
    if (ImGui::Checkbox("##Enabled", &enabled)) {
      PushUndoSnapshot();
      goal.enabled = enabled;
      MarkEdited(true);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    std::string label = std::string(goal.name) + " : " + std::string(goal.inputBotLink) + " -> " +
        std::string(goal.outputBotLink);
    if (_prepared && FindOutputBotLink(goal.outputBotLink) < 0) {
      label += " [missing]";
    }
    if (NameUsed(goal.name, i) || OutputBotLinkUsed(goal.outputBotLink, i)) {
      label += " [duplicate]";
    }
    if (ImGui::Selectable(label.c_str(), _selectedGoal == i)) {
      SelectGoal(i);
    }
    ImGui::PopID();
  }
  if (config.goals.empty()) {
    ImGui::TextDisabled("No goals. Add a link goal below.");
  }
  ImGui::BeginDisabled(IsReadOnly() || !_prepared);
  if (ImGui::BeginCombo("Add Output Link Goal", "Choose an unused link")) {
    for (auto const& name : _preview.GetOutputBotLinkNames()) {
      ImGui::BeginDisabled(OutputBotLinkUsed(name));
      if (ImGui::Selectable(name.c_str())) {
        AddGoal(name);
      }
      ImGui::EndDisabled();
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  if (HasGoalSelection()) {
    auto& goal = config.goals[_selectedGoal];
    ImGui::SeparatorText("Selected Goal");
    if (_nameBufferGoal != _selectedGoal) {
      _nameBuffer = goal.name;
      _nameBufferGoal = _selectedGoal;
    }
    ImGui::BeginDisabled(IsReadOnly());
    ImGui::InputText("Name", &_nameBuffer);
    bool const badName = _nameBuffer.empty() || NameUsed(_nameBuffer, _selectedGoal);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
      if (badName) {
        _editDiagnostic = "Goal names must be nonempty and unique.";
        _nameBuffer = goal.name;
      } else if (_nameBuffer != std::string_view(goal.name)) {
        PushUndoSnapshot();
        goal.name = _nameBuffer;
        MarkEdited(true);
      }
    }
    ImGui::BeginDisabled(!_prepared);
    if (ImGui::BeginCombo("Output Bot Link", goal.outputBotLink.c_str())) {
      for (auto const& name : _preview.GetOutputBotLinkNames()) {
        ImGui::BeginDisabled(OutputBotLinkUsed(name, _selectedGoal));
        if (ImGui::Selectable(name.c_str(), name == std::string_view(goal.outputBotLink)) &&
            name != std::string_view(goal.outputBotLink)) {
          PushUndoSnapshot();
          goal.outputBotLink = name;
          MarkEdited(true);
        }
        ImGui::EndDisabled();
      }
      ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    if (auto const* inputBot = _preview.GetInputBot()) {
      char const* selected =
          goal.inputBotLink.empty() ? "Output Bot Base" : goal.inputBotLink.c_str();
      if (ImGui::BeginCombo("Input Bot Link", selected)) {
        auto selectAnchor = [&](std::string_view name) {
          auto const oldWorldFrameFromConstraintFrame = GoalWorldConstraintFrame(goal);
          PushUndoSnapshot();
          goal.inputBotLink = name;
          if (auto const worldFrameFromBaseOrInputLinkFrame = GoalConstraintAnchor(goal);
              worldFrameFromBaseOrInputLinkFrame && oldWorldFrameFromConstraintFrame) {
            goal.baseOrInputLinkFrameFromConstraintFrame = robotics::IKGoalLocalConstraintTransform(
                goal,
                *worldFrameFromBaseOrInputLinkFrame,
                *oldWorldFrameFromConstraintFrame,
                config.inputBotScale);
          }
          MarkEdited(true);
        };
        if (ImGui::Selectable("Output Bot Base", goal.inputBotLink.empty())) {
          selectAnchor({});
        }
        for (auto const& link : inputBot->links) {
          if (ImGui::Selectable(
                  link.name.c_str(), link.name == std::string_view(goal.inputBotLink))) {
            selectAnchor(link.name);
          }
        }
        ImGui::EndCombo();
      }
    }
    ImGui::TextDisabled(
        goal.inputBotLink.empty()
            ? "Maps the constraint frame into the output bot's initial base frame."
            : "Maps the constraint frame into the selected input bot link's frame.");
    BotIKGoalConfig edited = goal;
    int const outputBotLinkIndex = FindOutputBotLink(edited.outputBotLink);
    auto const& links = _preview.GetOutputBotWorldFromLinks();
    bool const hasLink =
        _prepared && outputBotLinkIndex >= 0 && outputBotLinkIndex < mochi::isize(links);
    bool discrete = false;
    ImGui::SeparatorText("Output Base / Input Link Frame From Constraint Frame");
    auto baseOrInputLinkFrameFromConstraintFrame = edited.baseOrInputLinkFrameFromConstraintFrame;
    bool changed = ImGui::DragTransformRT(
        "Output Base / Input Link Frame From Constraint Frame",
        baseOrInputLinkFrameFromConstraintFrame);
    if (changed) {
      edited.baseOrInputLinkFrameFromConstraintFrame = baseOrInputLinkFrameFromConstraintFrame;
    }
    if (ImGui::Button("Zero##BaseOrInputLinkFrameFromConstraintFrame")) {
      edited.baseOrInputLinkFrameFromConstraintFrame = {};
      changed = discrete = true;
    }
    ImGui::SameLine();
    auto const worldFrameFromBaseOrInputLinkFrame = GoalConstraintAnchor(edited);
    ImGui::BeginDisabled(!hasLink || !worldFrameFromBaseOrInputLinkFrame);
    if (ImGui::Button("Match Output Constraint Frame") && hasLink &&
        worldFrameFromBaseOrInputLinkFrame) {
      edited.baseOrInputLinkFrameFromConstraintFrame = mochi::NormalizeRotation(
          robotics::IKGoalLocalConstraintTransform(
              edited,
              mochi::NormalizeRotation(*worldFrameFromBaseOrInputLinkFrame),
              mochi::NormalizeRotation(links[outputBotLinkIndex]) *
                  mochi::NormalizeRotation(edited.linkFrameFromConstraintFrame),
              config.inputBotScale));
      changed = discrete = true;
    }
    ImGui::EndDisabled();
    ImGui::SeparatorText("Output Link Frame From Constraint Frame");
    ImGui::TextDisabled("Maps the constraint frame into the selected output bot link's frame.");
    auto linkFrameFromConstraintFrame = edited.linkFrameFromConstraintFrame;
    if (ImGui::DragTransformRT(
            "Output Link Frame From Constraint Frame", linkFrameFromConstraintFrame)) {
      edited.linkFrameFromConstraintFrame = linkFrameFromConstraintFrame;
      changed = true;
    }
    if (ImGui::Button("Zero##LinkFrameFromConstraintFrame")) {
      edited.linkFrameFromConstraintFrame = {};
      changed = discrete = true;
    }
    ImGui::SameLine();
    auto const worldFrameFromConstraintFrame = GoalWorldConstraintFrame(edited);
    ImGui::BeginDisabled(!hasLink || !worldFrameFromConstraintFrame);
    if (ImGui::Button("Match Input Constraint Frame") && hasLink && worldFrameFromConstraintFrame) {
      edited.linkFrameFromConstraintFrame = mochi::NormalizeRotation(
          mochi::Invert(mochi::NormalizeRotation(links[outputBotLinkIndex])) *
          mochi::NormalizeRotation(*worldFrameFromConstraintFrame));
      changed = discrete = true;
    }
    ImGui::EndDisabled();
    ImGui::SeparatorText("Stiffness");
    changed |= ImGui::DragReal(
        "Position Stiffness",
        &edited.positionStiffness,
        1.0f,
        0,
        std::numeric_limits<float>::max(),
        GetUnitFormat(
            UnitFormat::Stiffness,
            mochi::ArticulatedJointType::Prismatic,
            edited.positionStiffness));
    changed |= ImGui::DragReal(
        "Rotation Stiffness",
        &edited.rotationStiffness,
        0.01f,
        0,
        std::numeric_limits<float>::max(),
        GetUnitFormat(
            UnitFormat::Stiffness,
            mochi::ArticulatedJointType::Revolute,
            edited.rotationStiffness));
    if (changed) {
      if (mochi::IsFinite(edited.baseOrInputLinkFrameFromConstraintFrame) &&
          mochi::NearEqual(
              mochi::Norm(edited.baseOrInputLinkFrameFromConstraintFrame.GetRotation().ToReal4()),
              mochi::real(1),
              mochi::real(1e-4)) &&
          mochi::IsFinite(edited.positionStiffness) && edited.positionStiffness >= 0 &&
          mochi::IsFinite(edited.rotationStiffness) && edited.rotationStiffness >= 0 &&
          mochi::IsFinite(IKGoalConstraintTransform(edited)) &&
          mochi::IsFinite(edited.linkFrameFromConstraintFrame) &&
          mochi::NearEqual(
              mochi::Norm(edited.linkFrameFromConstraintFrame.GetRotation().ToReal4()),
              mochi::real(1),
              mochi::real(1e-4))) {
        if (discrete) {
          PushUndoSnapshot();
        }
        goal = std::move(edited);
        MarkEdited(discrete);
      } else {
        _editDiagnostic =
            "Both constraint frames must define finite transforms with valid rotations; "
            "stiffness must be finite and nonnegative. Previous goal retained.";
      }
    }
    ImGui::TextDisabled("Zero stiffness disables that component.");
    ImGui::BeginDisabled(!_prepared || FindOutputBotLink(goal.outputBotLink) < 0);
    if (ImGui::Button("Reset Goal to Initial Output Constraint Pose")) {
      ResetGoal();
    }
    ImGui::EndDisabled();
    if (ImGui::Button("Remove Goal")) {
      RemoveGoal();
    }
    ImGui::EndDisabled();
  }
  ShowJointCouplings();
  if (!_outputBotDiagnostic.empty()) {
    ImGui::TextWrapped("%s", _outputBotDiagnostic.c_str());
  }
  if (!_editDiagnostic.empty()) {
    ImGui::TextWrapped("%s", _editDiagnostic.c_str());
  }
  ImGui::End();
}

void IKEditor::ShowJointCouplings() {
  using Coupling = robotics::BotIKJointCouplingConfig;
  auto& config = _ikAsset->GetConfig();
  auto const* outputBot = _prepared ? &_preview.GetOutputBot() : nullptr;
  ImGui::SeparatorText("Joint Couplings");
  ImGui::TextWrapped(
      "Soft constraints on the relative positions of two joints: q_B = ratio * q_A."
      "Revolute stiffness is N*m/rad; prismatic stiffness is N/m. Signed ratios are allowed.");
  ImGui::PushID("JointCouplings");
  int remove = -1;
  if (ImGui::BeginTable(
          "Rows",
          6,
          ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable |
              ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Enabled", ImGuiTableColumnFlags_WidthFixed, 55.0f);
    ImGui::TableSetupColumn("Joint A", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Joint B", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Ratio", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("Stiffness", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 65.0f);
    ImGui::TableHeadersRow();
    for (size_t i = 0; i < config.jointCouplings.size(); ++i) {
      Coupling edited = config.jointCouplings[i];
      ImGui::PushID(static_cast<int>(i));
      ImGui::TableNextRow();
      ImGui::BeginDisabled(IsReadOnly());
      ImGui::TableNextColumn();
      bool changed = ImGui::Checkbox("##Enabled", &edited.enabled);
      bool discrete = changed;
      ImGui::TableNextColumn();
      if (CouplingJointPicker("##JointA", edited.jointA, edited.jointB, outputBot)) {
        changed = discrete = true;
      }
      ImGui::TableNextColumn();
      if (CouplingJointPicker("##JointB", edited.jointB, edited.jointA, outputBot)) {
        changed = discrete = true;
      }
      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-std::numeric_limits<float>::min());
      changed |= ImGui::DragReal("##Ratio", &edited.ratio, 0.01f);
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(
            "q[%s] = ratio * q[%s] (ratio must be finite and nonzero)",
            edited.jointB.c_str(),
            edited.jointA.c_str());
      }
      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-std::numeric_limits<float>::min());
      auto couplingKind = CouplingJointKind(outputBot, edited.jointB);
      if (couplingKind == mochi::ArticulatedJointType::Invalid) {
        couplingKind = CouplingJointKind(outputBot, edited.jointA);
      }
      char const* stiffnessFormat = couplingKind == mochi::ArticulatedJointType::Invalid
          ? "%.4f (unbound)"
          : GetUnitFormat(UnitFormat::Stiffness, couplingKind, edited.stiffness);
      changed |= ImGui::DragReal(
          "##Stiffness",
          &edited.stiffness,
          0.01f,
          0,
          std::numeric_limits<mochi::real>::max(),
          stiffnessFormat);
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(
            "Stiffness (%s); zero is inactive. Large values can worsen conditioning.",
            couplingKind == mochi::ArticulatedJointType::Invalid ? "bind joints to resolve units"
                : couplingKind == mochi::ArticulatedJointType::Prismatic ? "N/m"
                                                                         : "N*m/rad");
      }
      ImGui::TableNextColumn();
      if (ImGui::Button("Remove")) {
        remove = static_cast<int>(i);
      }
      ImGui::EndDisabled();
      if (changed) {
        auto couplings = config.jointCouplings;
        couplings[i] = std::move(edited);
        CommitJointCouplings(std::move(couplings), discrete, i);
      }
      ImGui::PopID();
    }
    ImGui::EndTable();
  }
  if (remove >= 0) {
    auto couplings = config.jointCouplings;
    couplings.erase(couplings.begin() + remove);
    CommitJointCouplings(std::move(couplings), true);
  }
  if (config.jointCouplings.empty()) {
    ImGui::TextDisabled("No joint couplings.");
  }
  ImGui::PushID("Add");
  ImGui::BeginDisabled(IsReadOnly() || !outputBot);
  Coupling pending;
  pending.jointA = _pendingCouplingA;
  pending.jointB = _pendingCouplingB;
  bool canAdd = outputBot && !pending.jointA.empty() && !pending.jointB.empty();
  std::string addDiagnostic;
  if (canAdd) {
    BotIKConfig candidate = config;
    candidate.jointCouplings.push_back(pending);
    canAdd = ValidateIKConfig(candidate, addDiagnostic);
    if (canAdd) {
      mochi::DynamicString diagnostic;
      canAdd = robotics::ValidateIKJointCouplings(*outputBot, {&pending, 1}, diagnostic);
      addDiagnostic.assign(diagnostic.begin(), diagnostic.end());
    }
  }
  ImGui::BeginDisabled(!canAdd);
  if (ImGui::Button("Add Coupling")) {
    auto couplings = config.jointCouplings;
    couplings.push_back(std::move(pending));
    auto const row = couplings.size() - 1;
    if (CommitJointCouplings(std::move(couplings), true, row)) {
      _pendingCouplingA.clear();
      _pendingCouplingB.clear();
    }
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  float const pickerWidth =
      std::max(40.0f, (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f);
  CouplingJointPicker("##AddJointA", _pendingCouplingA, _pendingCouplingB, outputBot, pickerWidth);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Joint A (qB = ratio * qA)");
  }
  ImGui::SameLine();
  CouplingJointPicker("##AddJointB", _pendingCouplingB, _pendingCouplingA, outputBot, pickerWidth);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Joint B (qB = ratio * qA)");
  }
  ImGui::EndDisabled();
  ImGui::PopID();
  if (!addDiagnostic.empty()) {
    ImGui::TextWrapped("Add coupling: %s", addDiagnostic.c_str());
  }
  if (outputBot) {
    mochi::DynamicString diagnostic;
    if (!robotics::ValidateIKJointCouplings(*outputBot, config.jointCouplings, diagnostic)) {
      ImGui::TextWrapped("Couplings: %s", diagnostic.c_str());
    }
  }
  ImGui::TextDisabled(
      "Edits stop/reset the preview; authored pose and session recording are retained.");
  ImGui::PopID();
}

void IKEditor::ShowSolverWindow(bool* open) {
  if (!ImGui::Begin("IK Solver", open)) {
    ImGui::End();
    return;
  }
  auto& config = _ikAsset->GetConfig();
  auto const outputBotPath = _ikAsset->GetOutputBotPath();
  mochi::DynamicString outputBotSelection{outputBotPath.ToString()};
  if (ImGui::AssetSlot(
          "Output Bot",
          outputBotSelection,
          _studio->GetAssetManager(),
          _studio,
          AssetType::Bot,
          !IsReadOnly(),
          false) &&
      !IsReadOnly() && mochi::Path{std::string(outputBotSelection)} != outputBotPath) {
    PushUndoSnapshot();
    config.outputBot =
        MakeIKBotReference(_ikAsset->GetPath(), mochi::Path{std::string(outputBotSelection)});
    config.outputBotConfiguration.clear();
    MarkEdited(true);
    PrepareOutputBot();
  }
  if (_prepared && ImGui::CollapsingHeader("Output Bot Pose")) {
    ImGui::PushID("BotPose");
    ImGui::BeginDisabled(IsReadOnly() || _running || _recordingOwnsPose);
    auto const& outputBot = _preview.GetOutputBot();
    auto pose = config.outputBotConfiguration.empty() ||
            config.outputBotConfiguration.size() != outputBot.defaultPose.size()
        ? outputBot.defaultPose
        : config.outputBotConfiguration;
    auto commitPose = [&](mochi::DynamicArray<mochi::real> next, bool discrete) {
      std::string diagnostic;
      if (!_preview.SetOutputBotConfiguration(next, diagnostic)) {
        _editDiagnostic = std::move(diagnostic);
        return;
      }
      _running = false;
      _previewDiagnostic.clear();
      _outputBotDiagnostic.clear();
      config.outputBotConfiguration = std::move(next);
      MarkEdited(discrete);
      ApplyPreviewPose();
    };
    ImGui::TextWrapped(
        "Starting joint configuration. Stop restores this pose; Pause keeps the solve result.");
    if (ImGui::JointPoseEditor(outputBot, pose)) {
      commitPose(std::move(pose), false);
    }
    if (outputBot.defaultPose.empty()) {
      ImGui::TextDisabled("This output bot has no editable joints.");
    }
    if (ImGui::Button("Reset Output Bot Pose")) {
      PushUndoSnapshot();
      commitPose({}, true);
    }
    ImGui::EndDisabled();
    ImGui::PopID();
  }
  auto const inputBotPath = _ikAsset->GetInputBotPath();
  mochi::DynamicString inputBotSelection{inputBotPath.ToString()};
  if (ImGui::AssetSlot(
          "Input Bot",
          inputBotSelection,
          _studio->GetAssetManager(),
          _studio,
          AssetType::Bot,
          !IsReadOnly(),
          !IsReadOnly()) &&
      !IsReadOnly() && mochi::Path{std::string(inputBotSelection)} != inputBotPath) {
    PushUndoSnapshot();
    config.inputBot =
        MakeIKBotReference(_ikAsset->GetPath(), mochi::Path{std::string(inputBotSelection)});
    config.inputBotConfiguration.clear();
    if (config.inputBot.empty()) {
      SelectRecording({});
      for (auto& goal : config.goals) {
        if (auto const worldFrameFromConstraintFrame = GoalWorldConstraintFrame(goal)) {
          goal.baseOrInputLinkFrameFromConstraintFrame =
              mochi::Invert(_preview.GetWorldFrameFromOutputBaseFrame()) *
              *worldFrameFromConstraintFrame;
        }
        goal.inputBotLink.clear();
      }
    }
    PrepareOutputBot();
    MarkEdited(true);
  }
  if (!config.inputBot.empty()) {
    ImGui::BeginDisabled(IsReadOnly() || _recordingOwnsPose);
    ImGui::SeparatorText("Output Base Frame From Input Base Frame");
    auto outputBaseFrameFromInputBaseFrame = config.outputBaseFrameFromInputBaseFrame;
    if (ImGui::DragTransformRT(
            "Output Base Frame From Input Base Frame", outputBaseFrameFromInputBaseFrame)) {
      BotIKConfig candidate = config;
      candidate.outputBaseFrameFromInputBaseFrame = outputBaseFrameFromInputBaseFrame;
      std::string diagnostic;
      if (ValidateIKConfig(candidate, diagnostic)) {
        config.outputBaseFrameFromInputBaseFrame = outputBaseFrameFromInputBaseFrame;
        MarkEdited();
      } else {
        _editDiagnostic = diagnostic;
      }
    }
    ImGui::EndDisabled();
    ImGui::BeginDisabled(IsReadOnly());
    auto scale = config.inputBotScale;
    if (ImGui::DragReal(
            "Input Bot Uniform Scale",
            &scale,
            0.01f,
            mochi::real(0),
            std::numeric_limits<float>::max())) {
      BotIKConfig candidate = config;
      candidate.inputBotScale = scale;
      std::string diagnostic;
      if (ValidateIKConfig(candidate, diagnostic)) {
        config.inputBotScale = scale;
        MarkEdited();
      } else {
        _editDiagnostic = diagnostic;
      }
    }
    ImGui::TextDisabled(
        "Maps the input bot's base frame into the output bot's initial base frame.");
    ImGui::TextWrapped("Click the input bot in the viewport to select, translate, or rotate it.");
    ImGui::EndDisabled();
    if (!_inputBotDiagnostic.empty()) {
      ImGui::TextWrapped("%s", _inputBotDiagnostic.c_str());
    }
  }
  if (ImGui::CollapsingHeader("Input Bot Pose")) {
    ImGui::PushID("InputBotPose");
    auto const* inputBot = _preview.GetInputBot();
    ImGui::BeginDisabled(IsReadOnly() || _recordingOwnsPose || !inputBot);
    if (inputBot) {
      auto pose = config.inputBotConfiguration.empty() ? inputBot->defaultPose
                                                       : config.inputBotConfiguration;
      if (ImGui::JointPoseEditor(*inputBot, pose)) {
        mochi::DynamicArray<mochi::TransformRT> links;
        mochi::Error error;
        robotics::ComputeIKInputBotTransforms(
            *inputBot,
            pose,
            _preview.GetWorldFrameFromOutputBaseFrame() * config.outputBaseFrameFromInputBaseFrame,
            config.inputBotScale,
            links,
            error);
        if (error.IsOK()) {
          config.inputBotConfiguration = std::move(pose);
          MarkEdited();
        } else {
          _editDiagnostic = error.GetDescription();
        }
      }
    }
    if (ImGui::Button("Reset Input Bot Pose")) {
      PushUndoSnapshot();
      config.inputBotConfiguration.clear();
      MarkEdited(true);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    BotIKConfig snapshotCandidate;
    std::string snapshotDiagnostic;
    ImGui::BeginDisabled(!CanSnapshotInputBotPose(snapshotCandidate, snapshotDiagnostic));
    if (ImGui::Button("Snapshot Input Bot Pose")) {
      SnapshotInputBotPose();
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      if (snapshotDiagnostic.empty()) {
        ImGui::SetTooltip(
            "Capture displayed input pose into document and stop playback; recording selection kept.");
      } else {
        ImGui::SetTooltip("%s", snapshotDiagnostic.c_str());
      }
    }
    ImGui::EndDisabled();
    ImGui::PopID();
  }
  ImGui::SeparatorText("Creation and Numerical Settings");
  ImGui::BeginDisabled(IsReadOnly());
  auto createParams = config.createParams;
  if (ImGui::SimpleReflectionStruct(createParams)) {
    BotIKConfig candidate = config;
    candidate.createParams = createParams;
    std::string diagnostic;
    if (ValidateIKConfig(candidate, diagnostic)) {
      config.createParams = createParams;
      MarkEdited();
    } else {
      _editDiagnostic = std::move(diagnostic);
    }
  }
  ImGui::EndDisabled();
  if (config.createParams.solverParams.maxElapsedTimeSeconds <= 0) {
    ImGui::TextWrapped(
        "Warning: an unlimited solve budget may block viewport input on the UI thread.");
  }
  if (!_editDiagnostic.empty()) {
    ImGui::TextWrapped("%s", _editDiagnostic.c_str());
  }
  ShowRecordingControls();
  ImGui::End();
}

void IKEditor::ClearSolvePlots() {
  _solvePlots = {};
  // Start from the preview's current count: only solves after the clear are plotted.
  _plottedSolveCount = _preview.GetSolveCount();
  _solvePlotMin = 0;
  _solvePlotMax = 1;
}

void IKEditor::RecordSolveSample() {
  uint64_t const count = _preview.GetSolveCount();
  if (count == _plottedSolveCount) {
    return;
  }
  _plottedSolveCount = count;
  auto& plots = _solvePlots;
  if (plots.solves.size() >= kMaxSolvePlotSamples) {
    size_t const drop = kMaxSolvePlotSamples / 10;
    double const firstKept = plots.solves[drop];
    DropOldest(plots.solves, drop);
    DropOldest(plots.milliseconds, drop);
    DropOldest(plots.iterations, drop);
    auto const kept =
        std::lower_bound(plots.unconvergedSolves.begin(), plots.unconvergedSolves.end(), firstKept);
    size_t const unconvergedDrop = static_cast<size_t>(kept - plots.unconvergedSolves.begin());
    DropOldest(plots.unconvergedSolves, unconvergedDrop);
    DropOldest(plots.unconvergedIterations, unconvergedDrop);
  }
  double const solve = plots.nextSolve;
  plots.nextSolve += 1;
  double const iterations = _preview.GetLastSolveIterations();
  plots.solves.push_back(solve);
  plots.milliseconds.push_back(_preview.GetLastSolveMilliseconds());
  plots.iterations.push_back(iterations);
  plots.lastConvergence = _preview.GetLastSolveConvergence();
  if (IsUnconverged(plots.lastConvergence)) {
    plots.unconvergedSolves.push_back(solve);
    plots.unconvergedIterations.push_back(iterations);
  }
}

void IKEditor::ShowSolvePlotsWindow(bool* open) {
  // Layouts saved before this window existed have no place for it: on first use, tab it with the
  // bottom bar's Log Console instead of floating it over the viewport. Saved placements win.
  if (ImGuiWindow const* logConsole = ImGui::FindWindowByName("Log Console");
      logConsole && logConsole->DockId != 0) {
    ImGui::SetNextWindowDockID(logConsole->DockId, ImGuiCond_FirstUseEver);
  }
  if (!ImGui::Begin("IK Solve Plots", open)) {
    ImGui::End();
    return;
  }
  auto const& plots = _solvePlots;
  ImGui::BeginDisabled(plots.solves.empty());
  if (ImGui::Button("Clear")) {
    ClearSolvePlots();
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::Checkbox("Auto-fit", &_autoFitSolvePlots);
  ImGui::SameLine();
  if (plots.solves.empty()) {
    ImGui::TextDisabled(
        "Press Play or Step to plot each IK solve's time and nonlinear solver iterations.");
    ImGui::End();
    return;
  }
  size_t const count = plots.solves.size();
  double totalMilliseconds = 0;
  double maxMilliseconds = 0;
  for (double const milliseconds : plots.milliseconds) {
    totalMilliseconds += milliseconds;
    maxMilliseconds = std::max(maxMilliseconds, milliseconds);
  }
  double const maxIterations = *std::max_element(plots.iterations.begin(), plots.iterations.end());
  auto const& limits = _ikAsset->GetConfig().createParams.solverParams;
  char budget[32] = "unlimited";
  if (limits.maxElapsedTimeSeconds > 0) {
    std::snprintf(budget, sizeof(budget), "%.3g ms", limits.maxElapsedTimeSeconds * 1000.0);
  }
  ImGui::Text(
      "%zu solves | last: %.3f ms, %d iterations, %s | mean %.3f ms, max %.3f ms | max %d iterations | %zu not converged | limits: %s, %d iterations",
      count,
      plots.milliseconds.back(),
      static_cast<int>(plots.iterations.back()),
      ConvergenceLabel(plots.lastConvergence),
      totalMilliseconds / static_cast<double>(count),
      maxMilliseconds,
      static_cast<int>(maxIterations),
      plots.unconvergedSolves.size(),
      budget,
      limits.maxIter);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Solve time is the native solver call, excluding solver creation. Iterations are the "
        "nonlinear solver's; a solve that stops at the iteration limit or time budget before "
        "reaching its tolerance is marked not converged.");
  }

  auto const& params = _ikAsset->GetConfig().createParams.solverParams;
  if (_autoFitSolvePlots) {
    _solvePlotMin = plots.solves.front();
    _solvePlotMax = std::max(plots.solves.back(), plots.solves.front() + 1);
  }
  int const samples = static_cast<int>(count);
  ImPlotFlags const flags = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect;
  ImPlotAxisFlags const valueFit =
      _autoFitSolvePlots ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None;
  // One plot, two y-axes: solve time on the left, iterations on the right. Each limit line is a
  // translucent copy of its series' color on that series' axis; limits never drive the fit.
  // Solve time is the primary signal: prominent color, drawn after (on top of) the noisier
  // iterations, which are muted and thinner. Non-converged markers stay on top of both.
  ImVec4 const timeColor{1.00f, 0.62f, 0.20f, 1.0f};
  ImVec4 const iterationColor{0.45f, 0.60f, 0.80f, 0.75f};
  auto const lineSpec = [](ImVec4 color, float weight) {
    ImPlotSpec spec;
    spec.LineColor = color;
    spec.LineWeight = weight;
    return spec;
  };
  auto const limitSpec = [](ImVec4 color) {
    ImPlotSpec spec;
    spec.LineColor = ImVec4(color.x, color.y, color.z, 0.45f);
    spec.Flags = static_cast<ImPlotItemFlags>(ImPlotInfLinesFlags_Horizontal) |
        static_cast<ImPlotItemFlags>(ImPlotItemFlags_NoFit);
    return spec;
  };
  if (ImGui::BeginChild("##IKSolvePlotArea", ImVec2(0, 0))) {
    if (ImPlot::BeginPlot("##IKSolveMetrics", ImVec2(-1, -1), flags)) {
      ImPlot::SetupAxis(ImAxis_X1, "Solve");
      ImPlot::SetupAxis(ImAxis_Y1, "Solve time [ms]", valueFit);
      ImPlot::SetupAxis(ImAxis_Y2, "Iterations", ImPlotAxisFlags_AuxDefault | valueFit);
      ImPlot::SetupAxisLinks(ImAxis_X1, &_solvePlotMin, &_solvePlotMax);
      ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_Horizontal);

      ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
      ImPlot::PlotLine(
          "Iterations",
          plots.solves.data(),
          plots.iterations.data(),
          samples,
          lineSpec(iterationColor, 0.75f * kSolvePlotLineWeight));
      double const iterationLimit = params.maxIter;
      ImPlot::PlotInfLines("Iteration limit", &iterationLimit, 1, limitSpec(iterationColor));

      ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
      ImPlot::PlotLine(
          "Solve time",
          plots.solves.data(),
          plots.milliseconds.data(),
          samples,
          lineSpec(timeColor, kSolvePlotLineWeight));
      if (params.maxElapsedTimeSeconds > 0) {
        double const budgetMilliseconds = params.maxElapsedTimeSeconds * 1000.0;
        ImPlot::PlotInfLines("Time budget", &budgetMilliseconds, 1, limitSpec(timeColor));
      }

      if (!plots.unconvergedSolves.empty()) {
        ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
        ImPlotSpec markers;
        markers.Marker = ImPlotMarker_Circle;
        markers.MarkerSize = 3.0f;
        markers.MarkerFillColor = ImVec4(0.95f, 0.3f, 0.25f, 1.0f);
        markers.MarkerLineColor = markers.MarkerFillColor;
        ImPlot::PlotScatter(
            "Not converged",
            plots.unconvergedSolves.data(),
            plots.unconvergedIterations.data(),
            static_cast<int>(plots.unconvergedSolves.size()),
            markers);
      }
      ImPlot::EndPlot();
    }
  }
  ImGui::EndChild();
  ImGui::End();
}

void IKEditor::HandlePlayToolbarAction(PlayToolbarAction action) {
  switch (action) {
    case PlayToolbarAction::Play:
      if (!_selectedRecording.empty()) {
        if (!_recordingClip) {
          break;
        }
        _recordingOwnsPose = true;
        _recordingClock.Play();
      }
      if (!_preview.HasSolver()) {
        ClearSolvePlots(); // A fresh run, not a resume from Pause.
      }
      _running = true;
      _previewDiagnostic.clear();
      _nextSolve = Clock::now();
      break;
    case PlayToolbarAction::Pause:
      _recordingClock.Pause();
      _running = false;
      break;
    case PlayToolbarAction::Step:
      _running = false;
      if (!_selectedRecording.empty()) {
        if (!_recordingClip) {
          break;
        }
        _recordingOwnsPose = true;
        _recordingClock.Step();
      }
      if (!_preview.HasSolver()) {
        ClearSolvePlots();
      }
      SolveOnce();
      break;
    case PlayToolbarAction::Stop:
      StopPreview();
      break;
    case PlayToolbarAction::None:
      break;
  }
}

void IKEditor::ShowTabContents() {
  CaptureSavedSnapshot();
  if (!_viewport) {
    return;
  }
  ImGui::BeginChild("IKViewport", ImVec2(0, 0), 0, ImGuiWindowFlags_NoMove);
  _viewport->ShowViewportContents(true);
  ShowRecordingDropTarget();
  _viewport->ShowStatsOverlay(std::nullopt, std::nullopt, _preview.GetLastSolveMilliseconds());
  std::string diagnostic;
  HandlePlayToolbarAction(ShowPlayToolbarOverViewport(
      _running || _preview.HasSolver() || _recordingOwnsPose, _running, CanPreview(diagnostic)));
  std::string status = !_previewDiagnostic.empty() ? _previewDiagnostic
      : !diagnostic.empty()                        ? diagnostic
                                                   : _preview.GetStatus();
  if (!_selectedRecording.empty()) {
    status = "Session RRD: " + mochi::Path{_selectedRecording}.GetFilename() + " | " + status;
  }
  auto const origin = ImGui::GetWindowPos();
  auto const size = ImGui::GetWindowSize();
  ImGui::GetWindowDrawList()->AddText(
      {origin.x + 12, origin.y + size.y - ImGui::GetTextLineHeight() - 12},
      ImGui::GetColorU32(ImGuiCol_Text),
      status.c_str());
  ImGui::EndChild();
}

} // namespace superdex::studio
