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

#include "editors/tuner_editor.h"

#include "app/app.h"
#include "assets/mochi_prefab_asset.h"
#include "assets/tuner_asset.h"
#include "rendering/debug_text.h"
#include "rendering/measure_tool.h"
#include "ui/imgui_widgets.h"

#include <superdex_asset_tuner/load_math.h>
#include <superdex_asset_tuner/scene_targets.h>
#include <superdex_asset_tuner/tuner_runner.h>

#include <mochi_core/utils/math_utils.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_renderer/debug.h>
#include <mochi_renderer/type_conversions.h>

#include <imguios/fonts/icons_font_awesome5.h>
#include <implot.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>

namespace superdex::studio {

namespace tuner = superdex::tuner;
using mochi::DynamicString;
using mochi::real;
using mochi::Real3;
using mochi_renderer::SceneObject;

namespace {

struct NamedOption {
  char const* name;
  char const* description;
};

constexpr std::array<NamedOption, 4> kLoadTypes{{
    {"Force",
     "Force at a point on a rigid body or articulated link. A point off the center of mass also "
     "applies the matching torque. On a link it becomes joint forces through the link Jacobian."},
    {"Torque",
     "Pure torque on a rigid body or articulated link. On a link it becomes joint forces through "
     "the link Jacobian."},
    {"Joint Effort",
     "Generalized force on a joint's DoFs: N for prismatic, N-m for revolute. DoF Weights scale "
     "the profile per DoF."},
    {"Pose Path",
     "Pulls a point on a rigid body or articulated link, and optionally the body's orientation, along timed waypoints with a "
     "spring-damper, similar to in-viewport Physics Drag."},
}};
constexpr std::array<NamedOption, 7> kProbeKinds{{
    {"Body Pose", "Position and rotation vector of the body origin, in world frame."},
    {"Body Velocity", "Linear and angular velocity of the body, in world frame."},
    {"Joint State", "Position and velocity of each of the joint's DoFs."},
    {"Contact Force", "Total contact force on the body, in world frame."},
    {"Solver Stats",
     "Nonlinear iterations, residual, line-search iterations and convergence status (0 none, 1 "
     "converged, 2 stopped, 3 diverged) of each step."},
    {"Performance", "Wall-clock solve time of each step. Can vary between runs."},
    {"Penetration",
     "Approximate volume (m^3) by which the body overlaps other actors: depth times contact area, "
     "summed over its overlapping contact points."},
}};
constexpr std::array<NamedOption, 4> kProfileKinds{{
    {"Constant", "Constant magnitude from Start until End."},
    {"Envelope",
     "Rises linearly to Magnitude over Rise Time, holds, then falls to zero over Fall Time to End. "
     "Setting End before Starts results in holding forever."},
    {"Sine",
     "Offset + Magnitude * sin(2 * pi * Frequency * (t - Start) + Phase), from Start until End."},
    {"Keyframes", "Curve through timed values, interpolated."},
}};
constexpr std::array<NamedOption, 3> kInterpolations{{
    {"Step", "Holds until the next key/waypoint."},
    {"Linear", "Straight line to the next one (constant-speed turn for orientations)."},
    {"Smooth", "Smooth curve through the neighbors with easing."},
}};
constexpr std::array<NamedOption, 3> kDriveModes{{
    {"On",
     "Pulls body using the Grab Point along the path. With Drive Orientation on, also orients the body "
     "along the path."},
    {"Position Only", "Pulls only the Grab Point; the body turns freely about it."},
    {"Off", "Releases the Grab Point, until turned on again."},
}};
constexpr std::array<NamedOption, 3> kLoadFrames{{
    {"World", "Direction is fixed in the world frame with the point following the body."},
    {"Local", "Direction is in the body's frame and turns with the body."},
    {"World-Fixed",
     "Direction and point are both fixed in the world. "
     "Applies forces even if body is no longer near the point."},
}};
constexpr std::array<filament::math::float4, 4> kLoadColors{{
    {1.0f, 0.6f, 0.1f, 1.0f}, // Force
    {0.3f, 0.6f, 1.0f, 1.0f}, // Torque
    {0.7f, 0.3f, 1.0f, 1.0f}, // JointEffort
    {0.2f, 0.85f, 0.6f, 1.0f}, // PosePath
}};
constexpr filament::math::float4 kSelectedLoadColor{1.0f, 1.0f, 0.2f, 1.0f};
constexpr ImVec4 kWarningColor{1.0f, 0.7f, 0.2f, 1.0f};
constexpr ImVec4 kErrorColor{1.0f, 0.45f, 0.35f, 1.0f};
// Plot history cap, so a long session cannot grow without bound (~1.5 h at 100 Hz).
constexpr size_t kMaxPlotSamples = 500'000;
constexpr float kPlotLineWeight = 2.0f;
// Fraction of the data range added when fitting Y, so values at the extremes stay visible.
constexpr float kPlotFitPaddingY = 0.1f;
constexpr double kTimeEpsilon = 1e-9;
// Waypoint and keyframe tables.
constexpr ImGuiTableFlags kListTableFlags = ImGuiTableFlags_SizingStretchProp |
    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_PadOuterX;

bool IsForce(tuner::LoadType type) {
  return type == tuner::LoadType::Force;
}

// Joint Effort

// Joint Effort weights 1-3 DoFs (see TunerSession::ResolveLoad).
bool TakesJointEffort(int numDofs) {
  return numDofs >= 1 && numDofs <= 3;
}

// The key a load's keyed direction and point are shown and edited at while stopped: `key` if it is
// one of the profile's keys, else the first.
size_t ClampKey(tuner::LoadSpec const& load, int key) {
  return key >= 0 && key < static_cast<int>(load.profile.keys.size()) ? static_cast<size_t>(key)
                                                                      : 0;
}

// The load's direction (in its frame) as the editor shows and edits it: key `key`'s when the keys
// animate it, else the load's own.
template <typename Load>
auto& DirectionAt(Load& load, int key) {
  return tuner::AnimatesDirection(load) ? load.profile.keys[ClampKey(load, key)].direction
                                        : load.direction;
}

// The load's point (body-local, or world for WorldPoint) as the editor shows and edits it: key
// `key`'s when the keys animate it, else the load's own.
template <typename Load>
auto& PointAt(Load& load, int key) {
  if (tuner::AnimatesPoint(load)) {
    return load.profile.keys[ClampKey(load, key)].point;
  }
  return load.frame == tuner::LoadFrame::WorldPoint ? load.pointWorld : load.pointLocal;
}

// Where a load acts for its body at `worldFromBody`: the application point for forces (key `key`'s
// when keyed), the driven point for pose paths, else the body origin.
Real3 LoadPointWorld(
    tuner::LoadSpec const& load,
    int key,
    mochi::TransformRT const& worldFromBody) {
  if (load.type == tuner::LoadType::PosePath) {
    return worldFromBody.TransformPoint(load.pointLocal);
  }
  if (!IsForce(load.type)) {
    return worldFromBody.GetTranslation();
  }
  Real3 const& point = PointAt(load, key);
  return load.frame == tuner::LoadFrame::WorldPoint ? point : worldFromBody.TransformPoint(point);
}

// A load as previewed on its staged body at `worldFromBody` while stopped: at key `key` for a
// keyframed profile (its value as the magnitude, and its direction and point when keyed), else at
// the profile's magnitude.
tuner::AppliedLoad
PreviewLoad(tuner::LoadSpec const& load, int key, mochi::TransformRT const& worldFromBody) {
  tuner::AppliedLoad applied;
  if (load.type == tuner::LoadType::PosePath) {
    applied.pointWorld = LoadPointWorld(load, key, worldFromBody);
    return applied;
  }
  auto const& keys = load.profile.keys;
  bool const keyframed = load.profile.kind == tuner::ProfileKind::Keyframes && !keys.empty();
  applied.magnitude = keyframed ? keys[ClampKey(load, key)].value : load.profile.magnitude;
  applied.pointWorld = LoadPointWorld(load, key, worldFromBody);
  Real3 const& shown = DirectionAt(load, key);
  if (load.type == tuner::LoadType::JointEffort) {
    applied.vectorWorld = shown * applied.magnitude;
    return applied;
  }
  Real3 const direction =
      load.frame == tuner::LoadFrame::Local ? worldFromBody.TransformDirection(shown) : shown;
  real const norm = mochi::Norm(direction);
  applied.vectorWorld = norm > 0 ? direction * (applied.magnitude / norm) : Real3{0.0f, 0.0f, 0.0f};
  return applied;
}

// A waypoint's pose: the driven point's position and the body's orientation.
mochi::TransformRT WaypointPose(tuner::PoseWaypoint const& waypoint) {
  return mochi::TransformRT(
      mochi::Quaternion::FromRotationVector(waypoint.rotation), waypoint.position);
}

// `items` sorted by time, for previews of keys or waypoints mid-edit (the spec is sorted once an
// edit ends).
template <typename T>
std::vector<T> SortedByTime(mochi::DynamicArray<T> const& items) {
  std::vector<T> sorted(items.begin(), items.end());
  std::stable_sort(
      sorted.begin(), sorted.end(), [](T const& a, T const& b) { return a.time < b.time; });
  return sorted;
}

// Stable-sorts `items` (waypoints or keys) by time and returns where item `tracked` ended up.
template <typename T>
int SortByTimeTracking(mochi::DynamicArray<T>& items, size_t tracked) {
  // After every earlier time, and after equal times that came before it.
  double const time = items[tracked].time;
  int sortedIndex = 0;
  for (size_t i = 0; i < items.size(); ++i) {
    sortedIndex += (items[i].time < time || (items[i].time == time && i < tracked)) ? 1 : 0;
  }
  std::stable_sort(
      items.begin(), items.end(), [](T const& a, T const& b) { return a.time < b.time; });
  return sortedIndex;
}

std::string_view ChannelGroup(std::string_view channel) {
  size_t const dot = channel.rfind('.');
  return dot == std::string_view::npos ? channel : channel.substr(0, dot);
}

void Tooltip(char const* text) {
  ImGui::ItemTooltipWrapped(text);
}

// Combo over `options`, with each option's description as its tooltip (and the current option's on
// the closed combo).
template <typename Enum, size_t N>
bool DescribedCombo(char const* label, Enum& value, std::array<NamedOption, N> const& options) {
  auto const current = static_cast<size_t>(value);
  bool changed = false;
  if (ImGui::BeginCombo(label, current < N ? options[current].name : "?")) {
    for (size_t i = 0; i < N; ++i) {
      if (ImGui::Selectable(options[i].name, i == current)) {
        value = static_cast<Enum>(i);
        changed = true;
      }
      Tooltip(options[i].description);
      if (i == current) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  if (current < N) {
    Tooltip(options[current].description);
  }
  return changed;
}

// Bubble tint for a card: the model editor's neutral bubble grey with a faint shade of `accent`,
// darker when disabled or dimmed (another card is selected), and lighter when selected.
ImU32 CardColor(ImVec4 const& accent, bool enabled, bool selected = false, bool dimmed = false) {
  ImVec4 const neutral = !enabled ? ImVec4(0.13f, 0.13f, 0.15f, 1.0f)
      : selected                  ? ImVec4(0.28f, 0.28f, 0.32f, 1.0f)
      : dimmed                    ? ImVec4(0.16f, 0.16f, 0.18f, 1.0f)
                                  : ImVec4(0.22f, 0.22f, 0.25f, 1.0f);
  float const tint = !enabled ? 0.12f : selected ? 0.32f : dimmed ? 0.16f : 0.25f;
  // Darkened so the tint shifts the hue more than it lifts the brightness.
  constexpr float kAccentScale = 0.6f;
  auto const mix = [&](float n, float a) { return n + (a * kAccentScale - n) * tint; };
  return ImGui::GetColorU32(
      ImVec4(mix(neutral.x, accent.x), mix(neutral.y, accent.y), mix(neutral.z, accent.z), 1.0f));
}

ImVec4 LoadAccent(tuner::LoadType type) {
  auto const& c = kLoadColors[static_cast<size_t>(type)];
  return {c.x, c.y, c.z, 1.0f};
}

// A pose path span's color, on the waypoint timeline and the viewport path. On is the load color.
ImVec4 DriveModeColor(tuner::PathDriveMode mode) {
  switch (mode) {
    case tuner::PathDriveMode::On:
      break;
    case tuner::PathDriveMode::PositionOnly:
      return {0.95f, 0.75f, 0.25f, 1.0f};
    case tuner::PathDriveMode::Off:
      return {0.5f, 0.5f, 0.55f, 1.0f};
  }
  return LoadAccent(tuner::LoadType::PosePath);
}

constexpr ImVec4 kNeutralAccent{0.35f, 0.35f, 0.45f, 1.0f};

// Moves element `from` to index `to`, shifting the ones between (works for any contiguous array).
template <typename Container>
void MoveItem(Container& items, size_t from, size_t to) {
  auto const begin = items.begin();
  if (from < to) {
    std::rotate(begin + from, begin + from + 1, begin + to + 1);
  } else if (from > to) {
    std::rotate(begin + to, begin + from, begin + from + 1);
  }
}

// Automatic ground height shown by DrawRunFieldOverride. The reflection widget takes a plain
// function pointer, so the value is passed through here (UI thread only), set just before drawing.
float sAutoGroundHeightForRunUi = 0.0f;

// Run Settings widget override: Ground Height shows the automatic height until overridden, and an
// override starts from it.
bool DrawRunFieldOverride(char const* label, SReflect::FieldTypeInfo const& field, void* fieldPtr) {
  if (std::string_view(field._name) != "groundHeight") {
    return false;
  }
  auto& value = *static_cast<std::optional<real>*>(fieldPtr);
  ImGui::PushID("groundHeight");
  bool isOverridden = value.has_value();
  if (ImGui::Checkbox("##override", &isOverridden)) {
    value = isOverridden ? std::optional<real>(sAutoGroundHeightForRunUi) : std::nullopt;
  }
  Tooltip(
      "Override the ground height. Unchecked: the prefab's lowest point, like the prefab editor.");
  ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::SetNextItemWidth(
      ImGui::CalcItemWidth() - ImGui::GetFrameHeight() - ImGui::GetStyle().ItemInnerSpacing.x);
  real shown = value.value_or(sAutoGroundHeightForRunUi);
  ImGui::BeginDisabled(!isOverridden);
  if (ImGui::DragReal(
          label, &shown, 0.001f, 0.0f, 0.0f, isOverridden ? "%.4f m" : "auto: %.4f m") &&
      isOverridden) {
    value = shown;
  }
  ImGui::EndDisabled();
  Tooltip(
      isOverridden ? "Ground plane height; the viewport ground moves with it."
                   : "Automatic: the lowest point of the prefab. Check the box to override.");
  ImGui::PopID();
  return true;
}

// Channel suffixes a probe produces (as in signals.csv), for its card body.
std::string ProbeChannelHint(tuner::ProbeKind kind) {
  switch (kind) {
    case tuner::ProbeKind::BodyPose:
      return "pos.{x,y,z}, rotvec.{x,y,z}, rotvecUnwrapped.{x,y,z}";
    case tuner::ProbeKind::BodyVelocity:
      return "vel.{x,y,z}, angvel.{x,y,z}";
    case tuner::ProbeKind::JointState:
      return "q<i>, qd<i> per DoF";
    case tuner::ProbeKind::ContactForce:
      return "contactForce.{x,y,z}";
    case tuner::ProbeKind::SolverStats:
      return "solver.iters, solver.residual, solver.lineSearchIters, solver.convergence";
    case tuner::ProbeKind::Performance:
      return "perf.solveMs";
    case tuner::ProbeKind::Penetration:
      return "penetration.volume";
  }
  return {};
}

bool ProbeHasTarget(tuner::ProbeKind kind) {
  return kind != tuner::ProbeKind::SolverStats && kind != tuner::ProbeKind::Performance;
}
// Dropdown of `candidates` with a filter box. A target that is not a candidate (e.g. renamed in the
// prefab) is kept and flagged rather than silently replaced.
bool TargetPicker(
    char const* label,
    DynamicString& target,
    std::vector<std::string> const& candidates) {
  bool const known =
      std::find(candidates.begin(), candidates.end(), std::string_view(target)) != candidates.end();
  std::string const preview = target.empty() ? std::string("(pick)")
      : known                                ? std::string(std::string_view(target))
              : std::string(std::string_view(target)) + "  (not found)";
  if (!known && !target.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.35f, 1.0f));
  }
  bool const open = ImGui::BeginCombo(label, preview.c_str(), ImGuiComboFlags_HeightLarge);
  if (!known && !target.empty()) {
    ImGui::PopStyleColor();
  }
  if (!open) {
    return false;
  }
  // Only one combo is open at a time, so one filter serves every picker.
  static ImGuiTextFilter filter;
  if (ImGui::IsWindowAppearing()) {
    filter.Clear();
    ImGui::SetKeyboardFocusHere();
  }
  filter.Draw("##filter", -FLT_MIN);
  bool changed = false;
  for (std::string const& candidate : candidates) {
    if (!filter.PassFilter(candidate.c_str())) {
      continue;
    }
    if (ImGui::Selectable(candidate.c_str(), std::string_view(target) == candidate)) {
      target = candidate;
      changed = true;
    }
  }
  if (candidates.empty()) {
    ImGui::TextDisabled("No valid targets in this prefab.");
  }
  ImGui::EndCombo();
  return changed;
}

} // namespace

//--------------------------------------------------------------------------------------------------
// AssetEditor
//--------------------------------------------------------------------------------------------------

TunerEditor::TunerEditor(SuperDexStudio* studio, TunerAsset* asset)
    : AssetEditor(studio, asset),
      _tunerAsset(asset),
      _stage(studio, "TunerEditorStage"),
      _simSync(studio->GetEditorToRendererSpaceConverter()),
      _outputDir((asset->GetPath().GetParentPath().AsFilesystemPath() / "tuner_runs").string()) {}

void TunerEditor::Initialize() {
  if (!_tunerAsset->IsReadOnly()) {
    _undoStack.Initialize(
        [this] { return TakeUndoSnapshot(); },
        [this](std::string const& json, int) { RestoreUndoSnapshot(json); });
  }
  _viewport = Viewport::Create(_studio, _studio->GetViewSettings());
  // The transform gizmo edits the selected load: translate moves its point, rotate its direction.
  _viewport->enableScaleGizmo = false;
  // The gizmo edits the selected load, or the box of the selected box stop condition (move
  // and scale; boxes stay axis-aligned).
  _viewport->showTransformGizmoTarget = [this] {
    bool const box = SelectedStopBoxIndex() >= 0;
    _viewport->enableScaleGizmo = box;
    return !_mochiScene.IsSimulating() && _placingLoad < 0 &&
        (box || SelectedLoadGizmoFrame().has_value());
  };
  _viewport->getTransformGizmoTarget =
      [this]() -> std::optional<std::pair<mochi::TransformRT, mochi::Real3>> {
    if (int const box = SelectedStopBoxIndex(); box >= 0) {
      tuner::StopCondition const& stop =
          _tunerAsset->GetSpec().stopConditions[static_cast<size_t>(box)];
      Real3 const center = (stop.boxMin + stop.boxMax) * real{0.5};
      Real3 const size = stop.boxMax - stop.boxMin;
      return std::pair{mochi::TransformRT(mochi::Quaternion::Identity(), center), size};
    }
    std::optional<mochi::TransformRT> const frame = SelectedLoadGizmoFrame();
    if (!frame) {
      return std::nullopt;
    }
    return std::pair{*frame, Real3{1.0f, 1.0f, 1.0f}};
  };
  _viewport->onTransformGizmoDelta =
      [this](mochi::TransformRT const& delta, mochi::Real3 const& scale) {
        if (int const box = SelectedStopBoxIndex(); box >= 0) {
          tuner::StopCondition& stop =
              _tunerAsset->GetSpec().stopConditions[static_cast<size_t>(box)];
          Real3 const center = delta.TransformPoint((stop.boxMin + stop.boxMax) * real{0.5});
          Real3 half = (stop.boxMax - stop.boxMin) * real{0.5};
          constexpr real kMinHalfExtent = 1e-4f;
          for (int d = 0; d < 3; ++d) {
            half[d] = std::max(half[d] * scale[d], kMinHalfExtent);
          }
          stop.boxMin = center - half;
          stop.boxMax = center + half;
          OnSpecEdited(false);
          return;
        }
        OnLoadGizmoDelta(delta);
      };
  // Clicking a body selects the first load acting on it; clicking empty space deselects.
  _viewport->onSceneSelectionChanged = [this](std::vector<SceneObject*> const& objects) {
    if (_placingLoad >= 0) {
      return;
    }
    int const index = objects.empty() ? -1 : _stage.GetSceneObjectIndex(objects.back());
    StagedActor const* picked =
        index < 0 ? nullptr : &_stage.GetActors()[static_cast<size_t>(index)];
    auto const& loads = _tunerAsset->GetSpec().loads;
    _loadCards.Resize(loads.size());
    for (size_t i = 0; picked != nullptr && i < loads.size(); ++i) {
      if (FindStagedBody(loads[i].target) == picked) {
        SelectLoad(static_cast<int>(i));
        _loadCards.cards[i].collapsed = false;
        return;
      }
    }
    SelectLoad(-1);
    _selectedStopCard = 0;
  };
  _viewport->onSceneObjectClicked = [this](SceneObject* object, filament::math::float3 hit) {
    OnScenePressed(object, hit);
  };
  _viewport->RegisterShowCommand(
      {.name = "Physics Debug Draw",
       .onToggle = [this] { _mochiScene.ToggleDebugDraw(); },
       .getState = [this] { return _mochiScene.IsDebugDrawEnabled(); },
       .shortcut = ImGuiKey_P});
  auto const registerStageType = [this](char const* name, StageType type, ImGuiKey key) {
    _viewport->RegisterShowCommand(
        {.name = name,
         .onToggle =
             [this, type] {
               _stageType = type;
               Restage();
             },
         .getState = [this, type] { return _stageType == type; },
         .shortcut = key});
  };
  registerStageType("Render Only", StageType::RenderModelOnly, ImGuiKey_1);
  registerStageType("Collision Only", StageType::MochiModelOnly, ImGuiKey_2);
  registerStageType(
      "Render (Collision Fallback)", StageType::RenderModelFallbackToMochiModel, ImGuiKey_3);

  _mochiScene.Initialize(_studio->GetMochiContext(), "TunerEditor");
  ApplyRunSettings();
  _mochiScene.onStartPhysics = [this]() { _simSync.CaptureStagedSurfaces(_stage); };
  _mochiScene.createPhysicsActors = [this](mochi::Scene* s) { CreatePhysicsActors(s); };
  _mochiScene.destroyPhysicsActors = [this](mochi::Scene* s) { DestroyPhysicsActors(s); };
  // Runs on the UI thread once the actors exist: start the per-session drag controller and reset
  // the UI view of the session.
  _mochiScene.registerPreStepCallback = [this](mochi::AsyncScene*) {
    _dragController = std::make_unique<PhysicsDragController>(
        _mochiScene,
        &_stage,
        _studio->GetAppSettings().physicsDrag,
        _studio->GetRendererToEditorSpaceConverter());
    auto const loads = _session.GetLoads();
    _boundLoads.assign(loads.begin(), loads.end());
    _channels.clear();
    for (auto const& name : _session.GetChannelNames()) {
      ChannelHistory& channel = _channels.emplace_back();
      channel.name = std::string(std::string_view(name));
    }
    _sampleTimes.clear();
    _hasSample = false;
    _stopEvent.reset();
    return mochi::CallbackHandle{};
  };
  _mochiScene.registerPostStepCallback = [this](mochi::AsyncScene* scene) {
    return scene->RegisterPostStepCallback(
        "TunerEditor::ExtractActors",
        [this](mochi::StepInfo const& info) { _simSync.Produce(info.scene); });
  };
  _mochiScene.onStopPhysics = [this]() { OnStopPhysics(); };
  BindSceneObjectDragHooks(*_viewport, _dragController);
  _stage.BindRenderScene(_viewport->GetRenderScene());
  _viewport->SetSceneStage(&_stage);
  BindSceneStageMeasureTargets(
      *_viewport,
      _stage,
      [this] { return _mochiScene.IsSimulating(); },
      [this] { return _mochiScene.IsPaused(); });
  Restage();
  _viewport->FocusCameraOnScene();
}

void TunerEditor::OnHandleInputs() {
  if (_tunerAsset->GetPrefabAsset() != nullptr) {
    _mochiScene.HandleHotkeys();
  }
  if (_placingLoad >= 0 && ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    _placingLoad = -1;
  }
}

void TunerEditor::OnRender(Renderer const* renderer) {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.UpdateStats();
    _simSync.Apply(_stage, _studio->GetEditorToRendererSpaceConverter());
    DrainSamples();
    // A tuner whose targets do not resolve would silently run without its loads.
    if (!_bindError.empty()) {
      MOCHI_LOG_ERROR("Asset Tuner: %s", _bindError.c_str());
      _mochiScene.DestroyMochiScene();
    }
  }
  auto* debugDraw = _viewport->GetRenderScene()->GetDebugDraw();
  _mochiScene.DrawDebug(debugDraw, _studio->GetEditorToRendererSpaceConverter());
  if (_dragController) {
    _dragController->DrawDebug(
        debugDraw, _viewport->GetDebugText(), _studio->GetEditorToRendererSpaceConverter());
  }
  DrawLoadArrows();
  DrawStopBoxes();
  _viewport->RenderScene(renderer);
}

void TunerEditor::Shutdown() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  _stage.Clear();
  _viewport.reset();
}

void TunerEditor::OnActivate() {
  if (!_mochiScene.IsSimulating()) {
    Restage();
  }
}

void TunerEditor::OnDeactivate() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
}

void TunerEditor::Refresh() {
  if (_mochiScene.IsSimulating()) {
    _mochiScene.DestroyMochiScene();
  }
  Restage();
}

void TunerEditor::ShowTabContents() {
  ImGui::BeginChild("Viewport_Child", ImVec2(0, 0), 0, ImGuiWindowFlags_NoMove);
  _viewport->enableSceneObjectDrag = _mochiScene.IsSimulating() || _placingLoad >= 0;
  _viewport->ShowViewportContents(true);
  _viewport->ShowStatsOverlay(
      _mochiScene.GetStepsPerSecond(),
      _hasSample ? std::optional<float>(static_cast<float>(_lastSample.time)) : std::nullopt);
  ImGui::BeginDisabled(_tunerAsset->GetPrefabAsset() == nullptr);
  _mochiScene.ShowPlayToolbarOverViewport();
  ImGui::EndDisabled();
  ShowStopBanner();
  if (_placingLoad >= 0 || IsRecording()) {
    // Bottom-right, clear of the stats overlay in the bottom-left corner.
    std::string status;
    if (IsRecording()) {
      status = "REC";
    }
    if (_placingLoad >= 0) {
      status += status.empty() ? "" : "   ";
      status += "Click a body to place the load point (Esc cancels)";
    }
    constexpr float kMargin = 12.0f;
    ImVec2 const size = ImGui::CalcTextSize(status.c_str());
    ImVec2 const pos{
        ImGui::GetWindowWidth() - size.x - kMargin, ImGui::GetWindowHeight() - size.y - kMargin};
    ImVec2 const screen{ImGui::GetWindowPos().x + pos.x, ImGui::GetWindowPos().y + pos.y};
    ImVec2 const pad = ImGui::GetStyle().FramePadding;
    ImGui::GetWindowDrawList()->AddRectFilled(
        ImVec2(screen.x - pad.x, screen.y - pad.y),
        ImVec2(screen.x + size.x + pad.x, screen.y + size.y + pad.y),
        ImGui::GetColorU32(ImGuiCol_Button),
        size.y * 0.5f + pad.y);
    ImGui::SetCursorPos(pos);
    if (IsRecording()) {
      ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "REC");
      if (_placingLoad >= 0) {
        ImGui::SameLine(0.0f, ImGui::CalcTextSize("   ").x);
      }
    }
    if (_placingLoad >= 0) {
      ImGui::TextUnformatted("Click a body to place the load point (Esc cancels)");
    }
  }
  ImGui::EndChild();
}

std::vector<AssetEditor::WindowDeclaration> TunerEditor::GetDefaultWindows() {
  using Dock = AssetEditor::DockRegion;
  return {
      {"Asset Tuner Loads", true, Dock::SidePanelTop, false},
      {"Asset Tuner Run", true, Dock::SidePanelBottom, false},
      {"Asset Tuner Prefab Overrides", false, Dock::SidePanelBottom, false},
      {"Asset Tuner Plots", true, Dock::BottomBar, false},
      {"Physics Settings", false, Dock::SidePanelBottom, false},
      MeasureWindowDeclaration()};
}

std::vector<AssetEditor::WindowDeclaration> TunerEditor::GetAuxiliaryWindows() const {
  return GetDefaultWindows();
}

void TunerEditor::ShowAuxiliaryWindows() {
  RefreshTargets();
  if (bool& open = _studio->GetWindowVisible("Asset Tuner Loads")) {
    ShowLoadsWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Asset Tuner Run")) {
    ShowRunWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Asset Tuner Prefab Overrides")) {
    ShowOverridesWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Asset Tuner Plots")) {
    ShowPlotsWindow(&open);
  }
  if (bool& open = _studio->GetWindowVisible("Physics Settings")) {
    ShowPhysicsSettingsWindow(&open);
  }
  ShowMeasureWindow();
}

// Only debug draw is editable here: the time step, gravity, solver and ground come from the tuner
// (Asset Tuner Run), so a Studio run matches the CLI.
void TunerEditor::ShowPhysicsSettingsWindow(bool* open) {
  if (!ImGui::Begin("Physics Settings", open)) {
    ImGui::End();
    return;
  }
  ImGui::TextWrapped(
      "Time step, Gravity and Ground Plane settings are in the Asset Tuner Run tab.\n"
      "Solver settings are set from the prefab.");
  if (ImGui::CollapsingHeader("Debug Draw", ImGuiTreeNodeFlags_DefaultOpen)) {
    _mochiScene.ShowDebugDrawSection();
  }
  ImGui::End();
}

bool TunerEditor::CanUndoRedo() const {
  return !_mochiScene.IsSimulating();
}

void TunerEditor::ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) {
  if (_viewport && _viewport->GetRenderScene()) {
    _viewport->GetRenderScene()->ApplyViewSettings(viewSettings);
  }
}

void TunerEditor::OnAppSettingsChanged(AppSettings const& settings) {
  if (_dragController) {
    _dragController->SetSettings(settings.physicsDrag);
  }
}

void TunerEditor::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  AssetEditor::ForEachReferencedPath(callback);
  callback(_tunerAsset->GetPrefabPath());
}

//--------------------------------------------------------------------------------------------------
// Staging and simulation
//--------------------------------------------------------------------------------------------------

void TunerEditor::Restage() {
  _describedTargetsStale = true;
  MochiPrefabAsset* prefabAsset = _tunerAsset->GetPrefabAsset();
  if (prefabAsset == nullptr) {
    _stage.Clear();
    return;
  }
  // Nested prefabs edited in their own editors since the prefab was loaded (saved or not).
  ResolveNestedPrefabsFromMemory(
      prefabAsset->GetPrefab(), _studio->GetAssetManager(), prefabAsset->GetAssetsRoot());
  ValidateOverrides();
  _stage.StagePrefab(prefabAsset->GetPrefab(), prefabAsset->GetAssetsRoot(), _stageType);
  // The ground stays put mid-run (it only changes the view there).
  if (_mochiScene.IsSimulating()) {
    return;
  }
  // As in the prefab editor: the automatic ground sits at the lowest point of what is shown.
  _autoGroundHeight = _viewport->UpdateGroundPlane();
  ApplyRunSettings();
}

void TunerEditor::RefreshTargets() {
  // Wait until no widget is being edited. Otherwise typing an override would build a new scene on
  // every key press.
  if (!_describedTargetsStale || ImGui::IsAnyItemActive()) {
    return;
  }
  _describedTargetsStale = false;
  _describedTargets.reset();
  _targets = {};
  if (_tunerAsset->GetPrefabAsset() == nullptr) {
    return;
  }
  // In a temporary scene of the studio's context (never a second context; see DescribeTuner).
  tuner::SceneTargets described;
  std::string error;
  if (!tuner::DescribeTuner(
          _tunerAsset->GetSpec(),
          _tunerAsset->GetPath().AsFilesystemPath(),
          _studio->GetMochiContext(),
          described,
          error)) {
    MOCHI_LOG_ERROR("Asset Tuner: could not load the prefab into Mochi: %s", error.c_str());
    return;
  }
  // Unnamed actors cannot be targeted by name.
  for (tuner::RigidTarget const& rigid : described.rigid) {
    if (!rigid.isStatic && !rigid.name.empty()) {
      _targets.rigidBodies.emplace_back(std::string_view(rigid.name));
    }
  }
  for (tuner::ArticulatedTarget const& articulated : described.articulated) {
    std::string const actor(std::string_view(articulated.name));
    for (size_t i = 0; i < articulated.links.size(); ++i) {
      if (actor.empty() || articulated.links[i].empty()) {
        continue;
      }
      std::string link = actor + "/" + std::string(std::string_view(articulated.links[i]));
      if (i < articulated.joints.size() && !articulated.joints[i].name.empty()) {
        _targets.joints.push_back(
            {.name = actor + "/" + std::string(std::string_view(articulated.joints[i].name)),
             .link = link,
             .numDofs = articulated.joints[i].dofCount});
      }
      _targets.links.push_back(std::move(link));
    }
  }
  _describedTargets = std::move(described);
}

// Mirrors the superdex_asset_tuner CLI setup so a run in Studio matches the same run on the command
// line: fixed dt, default gravity and solver (the prefab and RunSpec override them), and the
// RunSpec ground plane. Debug draw is only visual, so the session's own choices are kept.
void TunerEditor::ApplyRunSettings() {
  tuner::RunSpec const& run = _tunerAsset->GetSpec().run;
  PhysicsSettings settings = _studio->GetAppSettings().physics;
  if (_debugSettingsInitialized) {
    settings.debug = _mochiScene.GetDebugSettings();
  }
  _debugSettingsInitialized = true;
  settings.scene.gravity = tuner::kDefaultGravity;
  settings.scene.useFixedTimeStep = true;
  settings.scene.fixedTimeStepSeconds = run.dt;
  settings.scene.throttleToRealTime = true;
  settings.solver = mochi::SolverParams{};
  // The tuner resolves the ground height itself (the RunSpec override or the automatic height), so
  // the scene gets it as a custom height.
  float const groundHeight = EffectiveGroundHeight();
  settings.studio.groundPlaneMode =
      run.groundPlane ? GroundPlaneMode::Custom : GroundPlaneMode::Off;
  settings.studio.groundPlaneHeight = groundHeight;
  _mochiScene.SetSettings(settings);
  // The viewport ground and the physics plane always share one height, so what is drawn is what the
  // bodies land on. Not moved mid-run (the settings are locked while simulating).
  _mochiScene.SetGroundPlaneContact(run.groundContact.value_or(mochi::ContactParams{}));
  if (_viewport && !_mochiScene.IsSimulating()) {
    _viewport->SetGroundPlaneHeight(groundHeight);
  }
}

float TunerEditor::EffectiveGroundHeight() const {
  auto const& override = _tunerAsset->GetSpec().run.groundHeight;
  return override ? static_cast<float>(*override) : _autoGroundHeight;
}

// Sim thread; the UI thread is blocked in MochiAsyncScene::CreateScene.
void TunerEditor::CreatePhysicsActors(mochi::Scene* scene) {
  _bindError.clear();
  _stopPausedOnSim = false;
  // The context is shared by the whole app. Studio runs one simulation at a time, which lets this
  // run set the threading mode. A run that fails to bind puts it back here, any other run in
  // DestroyPhysicsActors.
  mochi::Context* context = _studio->GetMochiContext();
  _savedSingleThreaded = context->IsSingleThreaded();
  context->SetIsSingleThreaded(_tunerAsset->GetSpec().run.singleThreaded);
  if (!BindPhysicsActors(scene)) {
    RestoreThreadingMode();
  }
}

// Sim thread, from CreatePhysicsActors.
bool TunerEditor::BindPhysicsActors(mochi::Scene* scene) {
  MochiPrefabAsset* prefabAsset = _tunerAsset->GetPrefabAsset();
  if (prefabAsset == nullptr) {
    _bindError = "Prefab '" + _tunerAsset->GetPrefabPath().ToString() + "' is not loaded.";
    return false;
  }
  tuner::TunerSpec const& spec = _tunerAsset->GetSpec();
  mochi::prefab::ScenePrefab overridden;
  mochi::prefab::ScenePrefab* prefab = &prefabAsset->GetPrefab();
  if (!spec.prefabOverrides.empty()) {
    overridden = *prefab;
    if (!tuner::ApplyPrefabOverrides(overridden, spec.prefabOverrides, _bindError)) {
      return false;
    }
    prefab = &overridden;
  }
  mochi::ErrorLog error;
  mochi::prefab::EnsureFullyLoaded(
      *prefab, prefabAsset->GetAssetsRoot(), _studio->GetMochiContext(), error);
  auto const result = mochi::prefab::AddToScene(*prefab, scene, {}, error);
  if (!error.IsOK()) {
    _bindError = "Failed to add the prefab to the scene (see log).";
    return false;
  }
  tuner::ApplyRunSpecOverrides(*scene, spec.run);
  _simSync.Track(mochi::MakeConstSpan(result.actors));
  if (!_session.Bind(*scene, spec, _bindError)) {
    return false;
  }
  _loadsCallback = scene->RegisterPreStepCallback(
      "TunerEditor::Loads", [this](mochi::StepInfo const& info) { _session.OnPreStep(info); });
  _probesCallback = scene->RegisterPostStepCallback(
      "TunerEditor::Probes", [this, sample = tuner::Sample{}](mochi::StepInfo const& info) mutable {
        _session.OnPostStep(info, sample);
        // Pause on this step, like the CLI stops on it, so the scene the viewport shows is the one
        // that triggered. Only the first trigger pauses; resuming runs on without stop checks.
        if (sample.stopCondition >= 0 && !_stopPausedOnSim.exchange(true)) {
          _mochiScene.GetAsyncScene()->Pause(true);
        }
        std::lock_guard const lock(_samplesMutex);
        _pendingSamples.push_back(sample);
      });
  if (_recordNextRun && OpenRunWriter()) {
    mochi::experimental::RecordingParams recording;
    recording.recordContactPoints = spec.run.record.contactPoints;
    recording.recordStaticActorState = spec.run.record.staticActors;
    mochi::Error recordError;
    mochi::experimental::StartRecording(
        scene, _writer.GetSimRecordingPath().string(), recording, recordError);
    _runSummary.simRecording = recordError.IsOK();
  }
  return true;
}

// Sim thread; the UI thread is blocked in MochiAsyncScene::DestroyMochiScene.
void TunerEditor::DestroyPhysicsActors(mochi::Scene* scene) {
  if (mochi::experimental::IsRecording(scene, mochi::ErrorAssert{})) {
    mochi::experimental::StopRecording(scene, mochi::ErrorAssert{});
  }
  for (mochi::CallbackHandle* handle : {&_loadsCallback, &_probesCallback}) {
    if (handle->IsValid()) {
      scene->CancelCallback(*handle);
      *handle = {};
    }
  }
  _session.Unbind(*scene);
  _simSync.Untrack(scene);
  RestoreThreadingMode();
}

void TunerEditor::RestoreThreadingMode() {
  if (_savedSingleThreaded) {
    _studio->GetMochiContext()->SetIsSingleThreaded(*_savedSingleThreaded);
    _savedSingleThreaded.reset();
  }
}

void TunerEditor::OnStopPhysics() {
  _dragController.reset();
  DrainSamples();
  StopRecording();
  _simSync.Reset();
  _stage.ResetWorldTransforms(_studio->GetEditorToRendererSpaceConverter());
  _hasSample = false;
  _stopEvent.reset();
  _bindError.clear();
}

void TunerEditor::DrainSamples() {
  std::vector<tuner::Sample> samples;
  {
    std::lock_guard const lock(_samplesMutex);
    samples.swap(_pendingSamples);
  }
  double const duration = _tunerAsset->GetSpec().run.duration;
  bool stopNow = false;
  for (tuner::Sample& sample : samples) {
    bool const inRun = !_stopAtDuration || sample.time <= duration + kTimeEpsilon;
    // The first triggered stop condition ends the run like on the CLI: the sim thread paused on
    // this step, and the recording ends with this sample.
    bool const stopsHere = sample.stopCondition >= 0 && !_stopEvent;
    if (IsRecording() && inRun) {
      _writer.WriteSample(sample);
      tuner::UpdateChannelSummaries(
          sample.values, _runSummary.steps == 0, mochi::MakeSpan(_runSummary.channels));
      ++_runSummary.steps;
      _runSummary.simTime = sample.time;
    }
    if (stopsHere) {
      _stopEvent = StopEvent{
          .condition = sample.stopCondition,
          .message = std::string(std::string_view(sample.stopMessage)),
          .time = sample.time};
      auto const& stops = _tunerAsset->GetSpec().stopConditions;
      if (IsRecording()) {
        _runSummary.status = "stopped";
        _runSummary.message = sample.stopMessage;
        if (sample.stopCondition < static_cast<int>(stops.size())) {
          _runSummary.stopCondition = stops[static_cast<size_t>(sample.stopCondition)].name;
        }
      }
      MOCHI_LOG("Asset Tuner stopped at t = %.3f s: %s", sample.time, sample.stopMessage.c_str());
      stopNow = true;
    }
    if (_sampleTimes.size() < kMaxPlotSamples && sample.values.size() == _channels.size()) {
      _sampleTimes.push_back(static_cast<float>(sample.time));
      for (size_t i = 0; i < _channels.size(); ++i) {
        _channels[i].values.push_back(static_cast<float>(sample.values[i]));
      }
    }
    _lastSample = std::move(sample);
    _hasSample = true;
    if (stopNow) {
      break; // keep the stop sample as the last one shown
    }
  }
  if (stopNow) {
    StopRecording();
  }
  if (_stopAtDuration && _hasSample && _lastSample.time >= duration - kTimeEpsilon &&
      _mochiScene.IsSimulating() && !_mochiScene.IsPaused()) {
    _mochiScene.Pause(true);
    StopRecording();
  }
}

//--------------------------------------------------------------------------------------------------
// Edits
//--------------------------------------------------------------------------------------------------

void TunerEditor::OnSpecEdited(bool structural) {
  _tunerAsset->SetDirty(true);
  if (structural) {
    GetUndoStack().PushNow();
  } else {
    GetUndoStack().MarkEdited();
  }
  if (!_mochiScene.IsSimulating()) {
    ApplyRunSettings();
    return;
  }
  if (LiveLoadsCompatible()) {
    _mochiScene.GetAsyncScene()->QueueCommand(
        [this, loads = _tunerAsset->GetSpec().loads](mochi::Scene*) {
          std::string error;
          _session.UpdateLoads(loads, error);
        });
  }
}

bool TunerEditor::LiveLoadsCompatible() const {
  auto const& loads = _tunerAsset->GetSpec().loads;
  if (loads.size() != _boundLoads.size()) {
    return false;
  }
  for (size_t i = 0; i < loads.size(); ++i) {
    // A pose path's drive is attached at its point, so moving the point needs a new run.
    if (loads[i].name != _boundLoads[i].name || loads[i].type != _boundLoads[i].type ||
        loads[i].target != _boundLoads[i].target ||
        (loads[i].type == tuner::LoadType::PosePath &&
         loads[i].pointLocal != _boundLoads[i].pointLocal)) {
      return false;
    }
  }
  return true;
}

std::string TunerEditor::TakeUndoSnapshot() const {
  return tuner::SaveTunerSpecToJson(_tunerAsset->GetSpec());
}

void TunerEditor::RestoreUndoSnapshot(std::string const& json) {
  tuner::TunerSpec spec;
  std::string error;
  if (!tuner::LoadTunerSpecFromJson(json, {}, spec, error)) {
    MOCHI_LOG_ERROR("Asset Tuner undo failed: %s", error.c_str());
    return;
  }
  _tunerAsset->GetSpec() = std::move(spec);
  _placingLoad = -1;
  _tunerAsset->SetDirty(!GetUndoStack().IsAtSavedState());
  ValidateOverrides();
  ApplyRunSettings();
}

void TunerEditor::ValidateOverrides() {
  // This runs when the prefab or its overrides change. Masses may have changed too.
  _describedTargetsStale = true;
  auto const& overrides = _tunerAsset->GetSpec().prefabOverrides;
  _overrideErrors.assign(overrides.size(), {});
  MochiPrefabAsset const* prefabAsset = _tunerAsset->GetPrefabAsset();
  if (prefabAsset == nullptr) {
    return;
  }
  // Each override is checked alone so every broken one gets its own message.
  for (size_t i = 0; i < overrides.size(); ++i) {
    if (overrides[i].path.empty()) {
      _overrideErrors[i] = "Empty path.";
      continue;
    }
    mochi::prefab::ScenePrefab copy = prefabAsset->GetPrefab();
    tuner::ApplyPrefabOverrides(
        copy, Span<tuner::PrefabOverride const>(&overrides[i], 1), _overrideErrors[i]);
  }
}

std::string TunerEditor::MakeUniqueLoadName(std::string_view base) const {
  auto const& loads = _tunerAsset->GetSpec().loads;
  auto const taken = [&](std::string const& name) {
    return std::any_of(loads.begin(), loads.end(), [&](tuner::LoadSpec const& load) {
      return std::string_view(load.name) == name;
    });
  };
  std::string name(base);
  for (int suffix = 1; taken(name); ++suffix) {
    name = std::string(base) + "_" + std::to_string(suffix);
  }
  return name;
}

void TunerEditor::CardListState::Resize(size_t count) {
  while (cards.size() < count) {
    cards.push_back({.id = nextId++, .collapsed = true});
  }
  cards.resize(count);
}

void TunerEditor::CardListState::AddExpanded() {
  cards.push_back({.id = nextId++, .collapsed = false});
}

//--------------------------------------------------------------------------------------------------
// Recording
//--------------------------------------------------------------------------------------------------

bool TunerEditor::OpenRunWriter() {
  // Record the ground height actually used, so rerunning the resolved spec reproduces this run.
  tuner::TunerSpec spec = _tunerAsset->GetSpec();
  if (spec.run.groundPlane) {
    spec.run.groundHeight = EffectiveGroundHeight();
  }
  _runSummary = {};
  for (auto const& name : _session.GetChannelNames()) {
    tuner::ChannelSummary channel;
    channel.name = name;
    _runSummary.channels.push_back(std::move(channel));
  }
  std::string error;
  if (!_writer.Open(
          std::filesystem::path(_outputDir),
          _tunerAsset->GetName(),
          "studio",
          spec,
          _tunerAsset->GetPrefabPath().AsFilesystemPath(),
          error)) {
    MOCHI_LOG_ERROR("Asset Tuner recording: %s", error.c_str());
    return false;
  }
  _writer.WriteHeader(_session.GetChannelNames());
  return true;
}

void TunerEditor::StartRecording() {
  if (IsRecording() || !_mochiScene.IsSimulating() || !OpenRunWriter()) {
    return;
  }
  tuner::RecordSpec const& record = _tunerAsset->GetSpec().run.record;
  mochi::experimental::RecordingParams recording;
  recording.recordContactPoints = record.contactPoints;
  recording.recordStaticActorState = record.staticActors;
  mochi::AsyncScene* asyncScene = _mochiScene.GetAsyncScene();
  asyncScene->QueueCommand(
      [this, path = _writer.GetSimRecordingPath().string(), recording](mochi::Scene* scene) {
        mochi::Error error;
        mochi::experimental::StartRecording(scene, path, recording, error);
        _runSummary.simRecording = error.IsOK();
      });
  asyncScene->WaitForQueuedCommands();
}

void TunerEditor::StopRecording() {
  if (!IsRecording()) {
    return;
  }
  if (_mochiScene.IsSimulating()) {
    mochi::AsyncScene* asyncScene = _mochiScene.GetAsyncScene();
    asyncScene->QueueCommand([](mochi::Scene* scene) {
      if (mochi::experimental::IsRecording(scene, mochi::ErrorAssert{})) {
        mochi::experimental::StopRecording(scene, mochi::ErrorAssert{});
      }
    });
    asyncScene->WaitForQueuedCommands();
  }
  std::string error;
  if (!_writer.Close(_runSummary, error)) {
    MOCHI_LOG_ERROR("Asset Tuner recording: %s", error.c_str());
  }
  _lastRunDir = std::string(std::string_view(_runSummary.runDir));
  MOCHI_LOG("Asset Tuner run recorded to %s", _lastRunDir.c_str());
}

//--------------------------------------------------------------------------------------------------
// Viewport
//--------------------------------------------------------------------------------------------------

StagedActor const* TunerEditor::FindStagedBody(std::string_view target) const {
  // A joint target is drawn on the link it drives.
  for (auto const& joint : _targets.joints) {
    if (joint.name == target) {
      target = joint.link;
      break;
    }
  }
  int const index = _stage.FindIndexByName(target);
  return index < 0 ? nullptr : &_stage.GetActors()[static_cast<size_t>(index)];
}

std::vector<std::string> TunerEditor::GetTargetCandidates(tuner::LoadType type) const {
  switch (type) {
    case tuner::LoadType::Force:
    case tuner::LoadType::Torque:
    case tuner::LoadType::PosePath: {
      std::vector<std::string> bodies = _targets.rigidBodies;
      bodies.insert(bodies.end(), _targets.links.begin(), _targets.links.end());
      return bodies;
    }
    case tuner::LoadType::JointEffort: {
      std::vector<std::string> joints;
      for (auto const& joint : _targets.joints) {
        if (TakesJointEffort(joint.numDofs)) {
          joints.push_back(joint.name);
        }
      }
      return joints;
    }
  }
  return {};
}

std::optional<std::string> TunerEditor::SelectedLoadTarget(tuner::LoadType type) const {
  auto const contains = [](std::vector<std::string> const& names, std::string const& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
  };
  auto const& selection = _viewport->GetSelectedSceneObjects();
  for (auto it = selection.rbegin(); it != selection.rend(); ++it) {
    int const index = _stage.GetSceneObjectIndex(*it);
    if (index < 0) {
      continue;
    }
    std::string const& name = _stage.GetActors()[static_cast<size_t>(index)].name;
    if (type == tuner::LoadType::JointEffort) {
      for (auto const& joint : _targets.joints) {
        if (TakesJointEffort(joint.numDofs) && joint.link == name) {
          return joint.name;
        }
      }
    } else if (contains(_targets.rigidBodies, name) || contains(_targets.links, name)) {
      return name;
    }
  }
  return std::nullopt;
}

void TunerEditor::OnScenePressed(SceneObject* object, filament::math::float3 worldHit) {
  auto& loads = _tunerAsset->GetSpec().loads;
  if (_placingLoad < 0 || _placingLoad >= static_cast<int>(loads.size()) ||
      _mochiScene.IsSimulating()) {
    return;
  }
  int const index = _stage.GetSceneObjectIndex(object);
  if (index < 0) {
    return;
  }
  StagedActor const& staged = _stage.GetActors()[static_cast<size_t>(index)];
  tuner::LoadSpec& load = loads[static_cast<size_t>(_placingLoad)];
  auto const contains = [&](std::vector<std::string> const& names) {
    return std::find(names.begin(), names.end(), staged.name) != names.end();
  };
  if (!contains(_targets.links) && !contains(_targets.rigidBodies)) {
    MOCHI_LOG_WARNING(
        "Asset Tuner: '%s' cannot take loads (only dynamic rigid bodies and articulated links can).",
        staged.name.c_str());
    return;
  }
  load.target = staged.name;
  Real3 const hitEditor = _studio->GetRendererToEditorSpaceConverter().TranslationToOutput(
      mochi_renderer::ToMochi<mochi::real>(worldHit));
  Real3 const hitLocal = staged.worldTransform.TransformPointInverse(hitEditor);
  if (tuner::AnimatesPoint(load)) {
    // The shown key's point, in the load's frame.
    PointAt(load, KeyShownFor(static_cast<size_t>(_placingLoad))) =
        load.frame == tuner::LoadFrame::WorldPoint ? hitEditor : hitLocal;
  } else {
    load.pointLocal = hitLocal;
    load.pointWorld = hitEditor;
  }
  _placingLoad = -1;
  OnSpecEdited(true);
}

int TunerEditor::SelectedLoadIndex() const {
  if (_selectedLoadCard == 0) {
    return -1;
  }
  auto const& cards = _loadCards.cards;
  auto const it = std::find_if(cards.begin(), cards.end(), [&](CardListState::Card const& card) {
    return card.id == _selectedLoadCard;
  });
  int const index = it == cards.end() ? -1 : static_cast<int>(it - cards.begin());
  return index < static_cast<int>(_tunerAsset->GetSpec().loads.size()) ? index : -1;
}

void TunerEditor::SelectLoad(int index) {
  bool const valid = index >= 0 && index < static_cast<int>(_loadCards.cards.size());
  uint64_t const card = valid ? _loadCards.cards[static_cast<size_t>(index)].id : 0;
  if (card != _selectedLoadCard) {
    // A pose path starts with its first waypoint on the gizmo.
    _selectedWaypoint = 0;
    _selectedKey = -1;
  }
  _selectedLoadCard = card;
  if (valid) {
    _selectedStopCard = 0;
  }
}

int TunerEditor::SelectedWaypointIndex() const {
  int const index = SelectedLoadIndex();
  if (index < 0) {
    return -1;
  }
  tuner::LoadSpec const& load = _tunerAsset->GetSpec().loads[static_cast<size_t>(index)];
  bool const valid = load.type == tuner::LoadType::PosePath && _selectedWaypoint >= 0 &&
      _selectedWaypoint < static_cast<int>(load.waypoints.size());
  return valid ? _selectedWaypoint : -1;
}

int TunerEditor::SelectedStopBoxIndex() const {
  if (_selectedStopCard == 0) {
    return -1;
  }
  auto const& stops = _tunerAsset->GetSpec().stopConditions;
  auto const& cards = _stopCards.cards;
  for (size_t i = 0; i < cards.size() && i < stops.size(); ++i) {
    if (cards[i].id == _selectedStopCard) {
      return stops[i].kind != tuner::StopKind::ChannelThreshold ? static_cast<int>(i) : -1;
    }
  }
  return -1;
}

void TunerEditor::DrawStopBoxes() {
  auto* debugDraw = _viewport->GetRenderScene()->GetDebugDraw();
  auto const& stops = _tunerAsset->GetSpec().stopConditions;
  if (debugDraw == nullptr || stops.empty()) {
    return;
  }
  using filament::math::float3;
  using filament::math::float4;
  auto const& toRenderer = _studio->GetEditorToRendererSpaceConverter();
  auto const point = [&](Real3 const& p) {
    return mochi_renderer::ToFilament<float>(toRenderer.TranslationToOutput(p));
  };
  filament::math::double3 const camera = _viewport->GetRenderScene()->GetCameraPosition();
  float3 const cameraPos{
      static_cast<float>(camera.x), static_cast<float>(camera.y), static_cast<float>(camera.z)};
  int const selected = SelectedStopBoxIndex();
  DebugText* debugText = _viewport->GetDebugText();
  for (size_t i = 0; i < stops.size(); ++i) {
    tuner::StopCondition const& stop = stops[i];
    if (!stop.enabled) {
      continue;
    }
    bool const triggered = _stopEvent && _stopEvent->condition == static_cast<int>(i);
    if (stop.kind == tuner::StopKind::ChannelThreshold) {
      // Position thresholds ("<body>.pos.x|y|z") are drawn as a translucent plane at the threshold,
      // centered on the body; other channels have no place in the scene.
      std::string_view const channel = stop.channel;
      constexpr std::string_view kPos = ".pos.";
      size_t const at = channel.rfind(kPos);
      if (at == std::string_view::npos || at + kPos.size() + 1 != channel.size()) {
        continue;
      }
      int const axis = channel.back() - 'x';
      StagedActor const* body = FindStagedBody(channel.substr(0, at));
      if (axis < 0 || axis > 2 || body == nullptr) {
        continue;
      }
      Real3 center = body->worldTransform.GetTranslation();
      center[axis] = stop.threshold;
      int const u = (axis + 1) % 3;
      int const v = (axis + 2) % 3;
      float const half = 0.12f * length(cameraPos - point(center));
      auto const corner = [&](float su, float sv) {
        Real3 p = center;
        p[u] += su * half;
        p[v] += sv * half;
        return point(p);
      };
      std::array<float3, 4> const quad{corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)};
      float4 const fill =
          triggered ? float4{1.0f, 0.2f, 0.15f, 0.30f} : float4{1.0f, 0.65f, 0.1f, 0.18f};
      float4 const edge =
          triggered ? float4{1.0f, 0.2f, 0.15f, 1.0f} : float4{1.0f, 0.65f, 0.1f, 0.9f};
      // Both windings, so the plane shows from either side.
      debugDraw->DrawSolidTriangle(quad[0], quad[1], quad[2], fill);
      debugDraw->DrawSolidTriangle(quad[0], quad[2], quad[3], fill);
      debugDraw->DrawSolidTriangle(quad[0], quad[2], quad[1], fill);
      debugDraw->DrawSolidTriangle(quad[0], quad[3], quad[2], fill);
      for (size_t e = 0; e < 4; ++e) {
        float3 const& p0 = quad[e];
        float3 const& p1 = quad[(e + 1) % 4];
        debugDraw->DrawSolidCylinder(
            p0, p1 - p0, 0.0014f * length(cameraPos - p0), edge, 8, true, false);
      }
      char label[160];
      std::snprintf(
          label,
          sizeof(label),
          ICON_FA_RULER_VERTICAL " %s: %s %s %.3g",
          stop.name.c_str(),
          stop.channel.c_str(),
          stop.comparison == tuner::StopComparison::Above ? ">" : "<",
          static_cast<double>(stop.threshold));
      debugText->Draw(quad[2], label, edge, ImVec2{0, -14});
      continue;
    }
    bool const isSelected = static_cast<int>(i) == selected;
    float4 const color = triggered ? float4{1.0f, 0.2f, 0.15f, 1.0f}
        : isSelected               ? kSelectedLoadColor
                                   : float4{1.0f, 0.65f, 0.1f, 0.9f};
    std::array<float3, 8> corners;
    for (int c = 0; c < 8; ++c) {
      Real3 const p{
          (c & 1) ? stop.boxMax[0] : stop.boxMin[0],
          (c & 2) ? stop.boxMax[1] : stop.boxMin[1],
          (c & 4) ? stop.boxMax[2] : stop.boxMin[2]};
      corners[static_cast<size_t>(c)] = point(p);
    }
    // Edges as thin tubes (camera-scaled like the load arrows) so they read over the scene.
    for (int a = 0; a < 8; ++a) {
      for (int bit : {1, 2, 4}) {
        int const b = a | bit;
        if (b == a) {
          continue;
        }
        float3 const& p0 = corners[static_cast<size_t>(a)];
        float3 const& p1 = corners[static_cast<size_t>(b)];
        float const radius =
            (isSelected || triggered ? 0.0022f : 0.0014f) * length(cameraPos - (p0 + p1) * 0.5f);
        debugDraw->DrawSolidCylinder(p0, p1 - p0, radius, color, 8, true, false);
      }
    }
    debugText->Draw(
        corners[7], std::string(ICON_FA_CUBE " ") + stop.name.c_str(), color, ImVec2{0, -14});
  }
}

void TunerEditor::ShowStopBanner() {
  if (!_stopEvent || !_mochiScene.IsSimulating()) {
    return;
  }
  // Top-center, below the play toolbar.
  std::string const text = std::string(ICON_FA_HAND_PAPER " Stopped at t = ") +
      std::to_string(_stopEvent->time).substr(0, 5) + " s: " + _stopEvent->message;
  char const* const hint = "Play resumes (stop conditions stay off for this run); Stop resets.";
  ImVec2 const textSize = ImGui::CalcTextSize(text.c_str());
  ImVec2 const hintSize = ImGui::CalcTextSize(hint);
  float const width = std::max(textSize.x, hintSize.x);
  ImVec2 const pad{12.0f, 8.0f};
  ImVec2 const windowPos = ImGui::GetWindowPos();
  float const x = windowPos.x + 0.5f * (ImGui::GetWindowWidth() - width);
  float const y = windowPos.y + 64.0f;
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  drawList->AddRectFilled(
      ImVec2(x - pad.x, y - pad.y),
      ImVec2(x + width + pad.x, y + textSize.y + hintSize.y + 4.0f + pad.y),
      ImGui::GetColorU32(ImVec4(0.45f, 0.08f, 0.06f, 0.92f)),
      8.0f);
  drawList->AddText(
      ImVec2(x, y), ImGui::GetColorU32(ImVec4(1.0f, 0.9f, 0.85f, 1.0f)), text.c_str());
  drawList->AddText(
      ImVec2(x, y + textSize.y + 4.0f), ImGui::GetColorU32(ImVec4(1.0f, 0.75f, 0.7f, 0.85f)), hint);
}

std::optional<mochi::TransformRT> TunerEditor::SelectedLoadGizmoFrame() const {
  int const index = SelectedLoadIndex();
  if (index < 0) {
    return std::nullopt;
  }
  tuner::LoadSpec const& load = _tunerAsset->GetSpec().loads[static_cast<size_t>(index)];
  if (load.type == tuner::LoadType::PosePath) {
    int const waypoint = SelectedWaypointIndex();
    return waypoint < 0
        ? std::nullopt
        : std::optional(WaypointPose(load.waypoints[static_cast<size_t>(waypoint)]));
  }
  StagedActor const* staged =
      load.type == tuner::LoadType::JointEffort ? nullptr : FindStagedBody(load.target);
  if (staged == nullptr) {
    return std::nullopt;
  }
  mochi::TransformRT const& worldFromBody = staged->worldTransform;
  int const key = KeyShownFor(static_cast<size_t>(index));
  Real3 const point = LoadPointWorld(load, key, worldFromBody);
  Real3 const& shown = DirectionAt(load, key);
  Real3 const direction =
      load.frame == tuner::LoadFrame::Local ? worldFromBody.TransformDirection(shown) : shown;
  return mochi::TransformRT(tuner::ShortestRotation(Real3{1.0f, 0.0f, 0.0f}, direction), point);
}

int TunerEditor::KeyShownFor(size_t index) const {
  return SelectedLoadIndex() == static_cast<int>(index) ? _selectedKey : -1;
}

void TunerEditor::OnLoadGizmoDelta(mochi::TransformRT const& delta) {
  int const index = SelectedLoadIndex();
  std::optional<mochi::TransformRT> const frame = SelectedLoadGizmoFrame();
  if (index < 0 || !frame) {
    return;
  }
  tuner::LoadSpec& load = _tunerAsset->GetSpec().loads[static_cast<size_t>(index)];
  if (load.type == tuner::LoadType::PosePath) {
    tuner::PoseWaypoint& waypoint = load.waypoints[static_cast<size_t>(SelectedWaypointIndex())];
    waypoint.position = delta.TransformPoint(waypoint.position);
    waypoint.rotation =
        (delta.GetRotation() * mochi::Quaternion::FromRotationVector(waypoint.rotation))
            .ToRotationVector();
    OnSpecEdited(false);
    return;
  }
  mochi::TransformRT const& worldFromBody = FindStagedBody(load.target)->worldTransform;
  // A keyed direction or point is edited at the shown key.
  int const key = KeyShownFor(static_cast<size_t>(index));
  if (IsForce(load.type)) {
    Real3 const pointWorld = delta.TransformPoint(frame->GetTranslation());
    PointAt(load, key) = load.frame == tuner::LoadFrame::WorldPoint
        ? pointWorld
        : worldFromBody.TransformPointInverse(pointWorld);
  }
  // Rotate the direction, keeping its length (it scales joint-effort DoFs, and users may rely on
  // it) and its frame.
  Real3& direction = DirectionAt(load, key);
  real const length = mochi::Norm(direction);
  Real3 const axisWorld = frame->TransformDirection(Real3{1.0f, 0.0f, 0.0f});
  Real3 const rotatedWorld = delta.TransformDirection(axisWorld) * (length > 0 ? length : real{1});
  direction = load.frame == tuner::LoadFrame::Local
      ? worldFromBody.TransformDirectionInverse(rotatedWorld)
      : rotatedWorld;
  OnSpecEdited(false);
}

void TunerEditor::DrawLoadArrows() {
  auto* debugDraw = _viewport->GetRenderScene()->GetDebugDraw();
  if (debugDraw == nullptr) {
    return;
  }
  DebugText* debugText = _viewport->GetDebugText();
  auto const& toRenderer = _studio->GetEditorToRendererSpaceConverter();
  auto const point = [&](Real3 const& p) {
    return mochi_renderer::ToFilament<float>(toRenderer.TranslationToOutput(p));
  };
  using filament::math::float3;
  using filament::math::float4;
  // Arrows are sized in renderer space relative to the camera distance (like gizmos), so they read
  // the same whatever the scene scale or zoom; only their length carries magnitude.
  filament::math::double3 const camera = _viewport->GetRenderScene()->GetCameraPosition();
  float3 const cameraPos{
      static_cast<float>(camera.x), static_cast<float>(camera.y), static_cast<float>(camera.z)};
  constexpr float kThicknessPerDistance = 0.0035f; // shaft radius / camera distance
  constexpr float kHeadLength = 5.0f; // x shaft radius
  constexpr float kHeadRadius = 2.2f; // x shaft radius
  constexpr float kRingRadius = 7.0f; // x shaft radius
  constexpr float kOverlayDepthOff = true;
  auto const thicknessAt = [&](float3 const& p) {
    return kThicknessPerDistance * std::max(length(cameraPos - p), 1e-4f);
  };
  // Label placement. Directions are found on screen, so labels clear the arrows from any view.
  filament::math::mat4 const viewProj = _viewport->GetRenderScene()->GetCameraProjectionMatrix() *
      _viewport->GetRenderScene()->GetCameraViewMatrix();
  int targetWidth = 1;
  int targetHeight = 1;
  _viewport->GetRenderTarget()->GetSize(targetWidth, targetHeight);
  auto const toScreen = [&](float3 const& p) -> std::optional<ImVec2> {
    filament::math::double4 const clip = viewProj * filament::math::double4{p.x, p.y, p.z, 1.0};
    if (clip.w <= 0.0) {
      return std::nullopt;
    }
    // Screen pixels (y down); the origin does not matter, only directions are used.
    return ImVec2{
        static_cast<float>(clip.x / clip.w) * 0.5f * static_cast<float>(targetWidth),
        -static_cast<float>(clip.y / clip.w) * 0.5f * static_cast<float>(targetHeight)};
  };
  // Half size of a label chip (DebugText pads the text by 5 x 2 px) plus the gap to the arrow.
  constexpr ImVec2 kChipPadding{5.0f, 2.0f};
  constexpr float kLabelGapPx = 8.0f;
  auto const chipHalfSize = [&](char const* text) {
    ImVec2 const textSize = ImGui::CalcTextSize(text);
    return ImVec2{0.5f * textSize.x + kChipPadding.x, 0.5f * textSize.y + kChipPadding.y};
  };
  auto const labelOffsetAbove = [&](char const* text) {
    return ImVec2{0.0f, -(chipHalfSize(text).y + kLabelGapPx + 4.0f)};
  };
  // Offset that moves a label centered on `anchor` off the arrow running from `anchor` to
  // `alongArrow`: to the opposite side, far enough that the chip clears the anchor.
  auto const labelOffsetAwayFrom =
      [&](float3 const& anchor, float3 const& alongArrow, char const* text) {
        std::optional<ImVec2> const from = toScreen(anchor);
        std::optional<ImVec2> const to = toScreen(alongArrow);
        if (!from || !to) {
          return labelOffsetAbove(text);
        }
        ImVec2 direction{to->x - from->x, to->y - from->y};
        float const directionLength = std::hypot(direction.x, direction.y);
        if (directionLength < 1.0f) {
          return labelOffsetAbove(text); // arrow points at the camera
        }
        direction = {direction.x / directionLength, direction.y / directionLength};
        ImVec2 const half = chipHalfSize(text);
        // Distance from the chip center to its edge along `direction` (support of a box).
        float const extent = half.x * std::abs(direction.x) + half.y * std::abs(direction.y);
        float const distance = extent + kLabelGapPx;
        return ImVec2{-direction.x * distance, -direction.y * distance};
      };
  // Straight arrow from `tail` to `tip`. The head keeps its proportions; an arrow too short for a
  // full head shrinks as a whole so the head never exceeds half its length.
  auto const drawArrow = [&](float3 const& tail, float3 const& tip, float radius, float4 const& c) {
    float3 const axis = tip - tail;
    float const axisLength = length(axis);
    if (axisLength < 1e-7f) {
      return;
    }
    float3 const unit = axis / axisLength;
    float const shrink = std::min(1.0f, 0.5f * axisLength / (kHeadLength * radius));
    float const shaftRadius = radius * shrink;
    float const headLength = kHeadLength * shaftRadius;
    debugDraw->DrawSolidCylinder(
        tail, unit * (axisLength - headLength), shaftRadius, c, 12, true, kOverlayDepthOff);
    debugDraw->DrawSolidCone(
        tail + unit * (axisLength - headLength),
        unit * headLength,
        kHeadRadius * shaftRadius,
        c,
        16,
        true,
        kOverlayDepthOff);
  };
  // Curved arrow (a tube around 3/4 of a circle, plus a head) turning about `axisUnit` by the
  // right-hand rule.
  auto const drawRotationArrow = [&](float3 const& center,
                                     float3 const& axisUnit,
                                     float radius,
                                     float4 const& c) {
    constexpr int kSegments = 24;
    constexpr float kSweep = 1.5f * std::numbers::pi_v<float>;
    float const ringRadius = kRingRadius * radius;
    float const tubeRadius = 0.8f * radius;
    float const headLength = kHeadLength * tubeRadius;
    float3 const helper =
        std::abs(axisUnit.x) < 0.9f ? float3{1.0f, 0.0f, 0.0f} : float3{0.0f, 1.0f, 0.0f};
    float3 const u = normalize(cross(axisUnit, helper));
    // u -> v turns positively about axisUnit; mirror it if renderer space flips handedness so
    // the arrow keeps the editor-space (physical) sense of rotation.
    float3 const v = cross(axisUnit, u) * (toRenderer.FlipsHandedness() ? -1.0f : 1.0f);
    auto const onRing = [&](float angle) {
      return center + (u * std::cos(angle) + v * std::sin(angle)) * ringRadius;
    };
    float const bodySweep = kSweep - headLength / ringRadius;
    float3 previous = onRing(0.0f);
    for (int s = 1; s <= kSegments; ++s) {
      float3 const next = onRing(bodySweep * static_cast<float>(s) / kSegments);
      debugDraw->DrawSolidCylinder(
          previous, next - previous, tubeRadius, c, 8, true, kOverlayDepthOff);
      previous = next;
    }
    float3 const tangent = -u * std::sin(bodySweep) + v * std::cos(bodySweep);
    debugDraw->DrawSolidCone(
        previous, tangent * headLength, kHeadRadius * tubeRadius, c, 16, true, kOverlayDepthOff);
  };
  bool const live = _mochiScene.IsSimulating() && _hasSample;
  int const selected = SelectedLoadIndex();
  auto const& overlays = _studio->GetAppSettings().viewport.tuner;
  // Whether an overlay with `visibility` is drawn for load `i`; `active` is whether it applies
  // something this step.
  auto const shows = [&](TunerOverlayVisibility visibility, size_t i, bool active) {
    switch (visibility) {
      case TunerOverlayVisibility::All:
        return true;
      case TunerOverlayVisibility::OnlyActive:
        return !live || active;
      case TunerOverlayVisibility::OnlySelected:
        return static_cast<int>(i) == selected;
      case TunerOverlayVisibility::HideWhileSimulating:
        return !_mochiScene.IsSimulating();
      case TunerOverlayVisibility::None:
        break;
    }
    return false;
  };
  // Straight tube from `a` to `b` (skipped when they coincide).
  auto const drawSegment = [&](float3 const& a, float3 const& b, float radius, float4 const& c) {
    if (length(b - a) > 1e-7f) {
      debugDraw->DrawSolidCylinder(a, b - a, radius, c, 8, true, kOverlayDepthOff);
    }
  };
  // Orientation axes of `pose` (X red, Y green, Z blue), drawn at its position.
  auto const drawTriad = [&](mochi::TransformRT const& pose) {
    float3 const at = point(pose.GetTranslation());
    real const axisLength = 14.0f * thicknessAt(at);
    for (int axis = 0; axis < 3; ++axis) {
      Real3 unit{0.0f, 0.0f, 0.0f};
      unit[axis] = 1.0f;
      float4 c{0.25f, 0.25f, 0.25f, 1.0f};
      c[axis] = 1.0f;
      drawSegment(
          at,
          point(pose.GetTranslation() + pose.TransformDirection(unit) * axisLength),
          0.35f * thicknessAt(at),
          c);
    }
  };
  // A load's arrow and label: an arrow toward its point for forces and the pose path drive, from
  // the body origin for torques (with a curl), and a dot for joint efforts and loads too small to
  // show.
  auto const drawLoadVector = [&](tuner::LoadSpec const& load,
                                  tuner::AppliedLoad const& applied,
                                  float4 const& color,
                                  char const* label,
                                  bool showArrow,
                                  bool showLabel) {
    bool const pullsPoint = IsForce(load.type) || load.type == tuner::LoadType::PosePath;
    float3 const anchor = point(applied.pointWorld);
    real const magnitude = mochi::Norm(applied.vectorWorld);
    real const arrowLength =
        magnitude * real{_studio->GetAppSettings().viewport.tuner.arrowMetersPerNewton};
    constexpr float kMinArrowLength = 1e-3f;
    if (load.type == tuner::LoadType::JointEffort || arrowLength < kMinArrowLength) {
      if (showArrow) {
        debugDraw->DrawSolidSphere(anchor, 1.5f * thicknessAt(anchor), color);
      }
      if (showLabel) {
        debugText->Draw(anchor, label, color, labelOffsetAbove(label));
      }
      return;
    }
    Real3 const unit = applied.vectorWorld * (1 / magnitude);
    // Point loads pull on their point (the head ends there); torques start at the body origin.
    Real3 const tailEditor =
        pullsPoint ? applied.pointWorld - unit * arrowLength : applied.pointWorld;
    float3 const tail = point(tailEditor);
    float3 const tip = point(tailEditor + unit * arrowLength);
    float const radius = thicknessAt(pullsPoint ? tip : tail);
    if (showArrow) {
      drawArrow(tail, tip, radius, color);
      if (load.type == tuner::LoadType::Torque) {
        // Curved arrow around the torque axis, turning by the right-hand rule, wrapped around the
        // middle of the axis arrow.
        float3 const axisUnit = normalize(tip - tail);
        drawRotationArrow(tail + (tip - tail) * 0.5f, axisUnit, radius, color);
      }
    }
    if (showLabel) {
      // The label sits at the anchor, pushed away from the arrow body so it never covers it.
      float3 const alongArrow = pullsPoint ? tail : tip;
      debugText->Draw(
          anchor,
          label,
          color,
          showArrow ? labelOffsetAwayFrom(anchor, alongArrow, label) : labelOffsetAbove(label));
    }
  };
  auto const& loads = _tunerAsset->GetSpec().loads;
  for (size_t i = 0; i < loads.size(); ++i) {
    tuner::LoadSpec const& load = loads[i];
    if (!load.enabled) {
      continue;
    }
    tuner::AppliedLoad applied;
    if (live && i < _lastSample.loads.size()) {
      applied = _lastSample.loads[i];
    } else {
      StagedActor const* staged = FindStagedBody(load.target);
      if (staged == nullptr) {
        continue;
      }
      applied = PreviewLoad(load, KeyShownFor(i), staged->worldTransform);
    }
    auto const type = static_cast<size_t>(load.type);
    bool const highlighted = static_cast<int>(i) == _hoveredLoad || static_cast<int>(i) == selected;
    filament::math::float4 const color = highlighted ? kSelectedLoadColor : kLoadColors[type];
    bool const isPath = load.type == tuner::LoadType::PosePath;
    bool const active =
        isPath ? applied.driveMode != tuner::PathDriveMode::Off : applied.magnitude != 0;
    bool const showPath = isPath && shows(overlays.paths, i, active);
    bool const showArrow = shows(overlays.arrows, i, active);
    bool const showLabel = shows(overlays.labels, i, active);
    if (showPath && !load.waypoints.empty()) {
      // The path, sampled densely enough to look smooth and colored by drive mode, then each
      // waypoint with its orientation.
      float4 const pathColor = kLoadColors[type];
      std::vector<tuner::PoseWaypoint> const path = SortedByTime(load.waypoints);
      double const first = path.front().time;
      double const last = path.back().time;
      int const samples = 48 * static_cast<int>(path.size() - 1);
      float3 previous = point(tuner::EvaluatePosePath(path, first).GetTranslation());
      for (int s = 1; s <= samples && last > first; ++s) {
        double const t = first + (last - first) * s / samples;
        float3 const next = point(tuner::EvaluatePosePath(path, t).GetTranslation());
        double const mid = first + (last - first) * (s - 0.5) / samples;
        ImVec4 const c = DriveModeColor(tuner::EvaluatePathDrive(path, mid));
        drawSegment(previous, next, 0.5f * thicknessAt(next), float4{c.x, c.y, c.z, 1.0f});
        previous = next;
      }
      int const selectedWaypoint = SelectedWaypointIndex();
      for (size_t w = 0; w < load.waypoints.size(); ++w) {
        tuner::PoseWaypoint const& waypoint = load.waypoints[w];
        bool const isSelected = static_cast<int>(w) == selectedWaypoint;
        float4 const waypointColor = isSelected ? kSelectedLoadColor : pathColor;
        float3 const at = point(waypoint.position);
        debugDraw->DrawSolidSphere(at, (isSelected ? 2.2f : 1.4f) * thicknessAt(at), waypointColor);
        drawTriad(WaypointPose(waypoint));
        char time[32];
        std::snprintf(time, sizeof(time), "%.2f s", waypoint.time);
        debugText->Draw(at, time, waypointColor, labelOffsetAbove(time));
      }
    }
    if (showPath && live && applied.driveMode != tuner::PathDriveMode::Off) {
      // Where the path wants the body now, tied to the driven point. Hidden while released.
      float3 const target = point(applied.targetPose.GetTranslation());
      drawSegment(point(applied.pointWorld), target, 0.4f * thicknessAt(target), color);
      drawTriad(applied.targetPose);
    }
    if (!showArrow && !showLabel) {
      continue;
    }
    // Forces and the pose path drive pull on their point; torques and efforts act on the body.
    bool const pullsPoint = IsForce(load.type) || isPath;
    char label[128];
    std::snprintf(
        label,
        sizeof(label),
        "%s %s %.3g %s",
        isPath           ? ICON_FA_ROUTE
            : pullsPoint ? ICON_FA_LONG_ARROW_ALT_DOWN
                         : ICON_FA_REDO,
        load.name.c_str(),
        static_cast<double>(applied.magnitude),
        pullsPoint ? "N" : "N m");
    drawLoadVector(load, applied, color, label, showArrow, showLabel);

    // A selected load whose keys animate its direction or point also shows its other keys, dimmed
    // and numbered, on the staged body (only while stopped: the body has moved during a run).
    if (live || static_cast<int>(i) != selected ||
        !(tuner::AnimatesDirection(load) || tuner::AnimatesPoint(load))) {
      continue;
    }
    StagedActor const* staged = FindStagedBody(load.target);
    if (staged == nullptr) {
      continue;
    }
    float4 const& base = kLoadColors[type];
    float4 const dimmed{0.55f * base.x, 0.55f * base.y, 0.55f * base.z, 1.0f};
    size_t const shownKey = ClampKey(load, KeyShownFor(i));
    for (size_t k = 0; k < load.profile.keys.size(); ++k) {
      if (k == shownKey) {
        continue; // drawn above
      }
      char number[16];
      std::snprintf(number, sizeof(number), "%d", static_cast<int>(k));
      drawLoadVector(
          load,
          PreviewLoad(load, static_cast<int>(k), staged->worldTransform),
          dimmed,
          number,
          showArrow,
          showLabel);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Windows
//--------------------------------------------------------------------------------------------------

void TunerEditor::ShowProfileEditor(size_t index, bool& changed) {
  using tuner::ProfileKind;
  tuner::LoadSpec& load = _tunerAsset->GetSpec().loads[index];
  tuner::LoadProfile& profile = load.profile;
  changed |= DescribedCombo("Profile", profile.kind, kProfileKinds);

  if (profile.kind == ProfileKind::Keyframes) {
    // Laid out like a pose path's waypoints: the curve with draggable key markers, a key table,
    // then Add Key.
    auto& keys = profile.keys;
    auto const select = [&](int key) {
      SelectLoad(static_cast<int>(index));
      _selectedKey = key;
    };

    // The keys can also carry the direction (Force, Torque, Joint Effort weights) and the point
    // (Force). Turning one on starts every key from the load's own value, so nothing jumps.
    bool const isEffort = load.type == tuner::LoadType::JointEffort;
    Real3 const staticPoint =
        load.frame == tuner::LoadFrame::WorldPoint ? load.pointWorld : load.pointLocal;
    if (ImGui::Checkbox(
            isEffort ? "Animate DoF Weights" : "Animate Direction", &profile.animateDirection)) {
      if (profile.animateDirection) {
        for (tuner::ProfileKey& key : keys) {
          key.direction = load.direction;
        }
      }
      changed = true;
    }
    Tooltip(
        isEffort ? "Give each key its own DoF weights, blended between keys."
                 : "Give each key its own direction. It turns between keys along the shortest arc, "
                   "as the key's To Next says.");
    if (IsForce(load.type)) {
      ImGui::SameLine();
      if (ImGui::Checkbox("Animate Point", &profile.animatePoint)) {
        if (profile.animatePoint) {
          for (tuner::ProfileKey& key : keys) {
            key.point = staticPoint;
          }
        }
        changed = true;
      }
      Tooltip(
          "Give each key its own application point, in the load's frame. With World Point the line "
          "of action moves through the world.");
    }
    if (keys.empty()) {
      ImGui::Text("No keys yet. Add one below.");
    } else {
      ShowProfilePreview(index, false, changed);
    }
    bool const isSelectedLoad = SelectedLoadIndex() == static_cast<int>(index);
    int const current =
        isSelectedLoad && _selectedKey < static_cast<int>(keys.size()) ? _selectedKey : -1;

    int removeKey = -1;
    int sortAfterEdit = -1;
    if (!keys.empty() && ImGui::BeginTable("##keys", 5, kListTableFlags)) {
      ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthStretch, 0.15f);
      ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthStretch, 1.0f);
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
      ImGui::TableSetupColumn("To Next", ImGuiTableColumnFlags_WidthStretch, 1.0f);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableHeadersRow();
      double const minTime = 0.0;
      for (size_t k = 0; k < keys.size(); ++k) {
        tuner::ProfileKey& key = keys[k];
        int const id = static_cast<int>(k);
        ImGui::PushID(id);
        ImGui::TableNextRow();
        // Editing any cell of a row selects its key, like clicking the row.
        auto const selectOnActivate = [&] {
          if (ImGui::IsItemActivated()) {
            select(id);
          }
        };

        ImGui::TableNextColumn();
        char number[16];
        std::snprintf(number, sizeof(number), "%d", id);
        if (ImGui::Selectable(
                number,
                current == id,
                ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap,
                ImVec2(0.0f, ImGui::GetFrameHeight()))) {
          select(id);
        }

        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= ImGui::DragScalar(
            "##time", ImGuiDataType_Double, &key.time, 0.01f, &minTime, nullptr, "%.3f s");
        selectOnActivate();
        if (ImGui::IsItemDeactivatedAfterEdit()) {
          sortAfterEdit = id;
        }

        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= ImGui::DragReal("##value", &key.value, 0.05f);
        selectOnActivate();
        Tooltip("N for forces and prismatic joints, N-m for torques and revolute joints.");

        ImGui::TableNextColumn();
        if (k + 1 < keys.size()) {
          ImGui::SetNextItemWidth(-FLT_MIN);
          changed |= DescribedCombo("##interpolation", key.interpolation, kInterpolations);
          selectOnActivate();
        } else {
          ImGui::AlignTextToFramePadding();
          ImGui::TextDisabled("end");
          Tooltip("The last key's value holds until the run ends.");
        }

        ImGui::TableNextColumn();
        if (ImGui::SmallButton(ICON_FA_TRASH)) {
          removeKey = id;
        }
        Tooltip("Remove this key.");
        ImGui::PopID();
      }
      ImGui::EndTable();
    }
    if (removeKey >= 0) {
      keys.erase(keys.begin() + removeKey);
      if (isSelectedLoad) {
        _selectedKey =
            std::min(current - (current > removeKey ? 1 : 0), static_cast<int>(keys.size()) - 1);
      }
      changed = true;
    } else if (sortAfterEdit >= 0) {
      _selectedKey = SortByTimeTracking(keys, static_cast<size_t>(sortAfterEdit));
    }

    if (ImGui::Button(ICON_FA_PLUS " Add Key")) {
      tuner::ProfileKey key;
      if (!keys.empty()) {
        key = keys.back();
        key.time += 0.5;
      } else {
        key.direction = load.direction;
        key.point = staticPoint;
      }
      keys.push_back(key);
      select(static_cast<int>(keys.size()) - 1);
      changed = true;
    }
    Tooltip("Add a key half a second after the last, with the same value.");

    // The selected key's direction and point, when the keys carry them.
    bool const directionKeyed = tuner::AnimatesDirection(load);
    bool const pointKeyed = tuner::AnimatesPoint(load);
    int const selected = SelectedLoadIndex() == static_cast<int>(index) &&
            _selectedKey < static_cast<int>(keys.size())
        ? _selectedKey
        : -1;
    if (selected >= 0 && (directionKeyed || pointKeyed)) {
      tuner::ProfileKey& key = keys[static_cast<size_t>(selected)];
      ImGui::Spacing();
      ImGui::Text("Keyframe %d", selected);
      ImGui::SameLine();
      ImGui::Text("at %.3f s", key.time);
      if (directionKeyed) {
        changed |= ImGui::DragRealXYZ(isEffort ? "DoF Weights" : "Direction", key.direction, 0.01f);
        Tooltip(
            isEffort ? "This key's per-DoF weights."
                     : "This key's direction, in the load's frame. Normalized when applied; the "
                       "viewport gizmo rotates it.");
      }
      if (pointKeyed) {
        changed |= ImGui::DragRealXYZ("Point", key.point, 0.001f, 0.0f, 0.0f, "%.4f m");
        Tooltip(
            load.frame == tuner::LoadFrame::WorldPoint
                ? "This key's application point in world space. The viewport gizmo moves it."
                : "This key's application point in the body's (or link's) local frame. The "
                  "viewport gizmo moves it.");
      }
    }
    return;
  }

  auto const dragSeconds = [&](char const* label, double& value, char const* tip) {
    double const min = 0.0;
    changed |=
        ImGui::DragScalar(label, ImGuiDataType_Double, &value, 0.01f, &min, nullptr, "%.3f s");
    Tooltip(tip);
  };
  changed |= ImGui::DragReal("Magnitude", &profile.magnitude, 0.05f);
  Tooltip(
      "Peak value: N for forces and prismatic joints, N m for torques and revolute joints. "
      "Negative values reverse the direction.");
  if (profile.kind == ProfileKind::Sine) {
    changed |= ImGui::DragReal("Offset", &profile.offset, 0.05f);
    Tooltip("Constant added to the sine wave.");
  }
  dragSeconds("Start", profile.start, "Simulation time the load turns on.");
  dragSeconds("End", profile.end, "Simulation time the load turns off. 0 (or <= Start): never.");
  if (profile.kind == ProfileKind::Envelope) {
    dragSeconds("Rise Time", profile.riseTime, "Time to ramp from 0 up to Magnitude.");
    bool const hasEnd = profile.end > profile.start;
    ImGui::BeginDisabled(!hasEnd);
    dragSeconds(
        "Fall Time",
        profile.fallTime,
        hasEnd ? "Time to ramp from Magnitude back to 0, ending exactly at End."
               : "Set an End to ramp down; without one the envelope holds forever.");
    ImGui::EndDisabled();
  }
  if (profile.kind == ProfileKind::Sine) {
    changed |= ImGui::DragReal("Frequency", &profile.freqHz, 0.01f, 0.0f, 0.0f, "%.3f Hz");
    Tooltip("Oscillation frequency.");
    changed |= ImGui::DragReal("Phase", &profile.phase, 0.01f, 0.0f, 0.0f, "%.3f rad");
    Tooltip("Phase at Start.");
  }
  ShowProfilePreview(index, false, changed);
}

void TunerEditor::ShowProfilePreview(size_t index, bool compact, bool& changed) {
  tuner::LoadProfile& profile = _tunerAsset->GetSpec().loads[index].profile;
  bool const isKeyframes = profile.kind == tuner::ProfileKind::Keyframes;
  // Keys may be mid-edit (unsorted), so the curve comes from a sorted copy.
  tuner::LoadProfile preview = profile;
  double end = std::max(_tunerAsset->GetSpec().run.duration, 1e-6);
  if (isKeyframes) {
    std::vector<tuner::ProfileKey> const sorted = SortedByTime(profile.keys);
    preview.keys.assign(sorted.begin(), sorted.end());
    if (!sorted.empty()) {
      end = std::max(end, sorted.back().time);
    }
  }
  constexpr int kPreviewSamples = 256;
  std::array<float, kPreviewSamples> times{};
  std::array<float, kPreviewSamples> values{};
  for (int s = 0; s < kPreviewSamples; ++s) {
    double const t = end * s / (kPreviewSamples - 1);
    times[s] = static_cast<float>(t);
    values[s] = static_cast<float>(tuner::EvaluateProfile(preview, t));
  }

  bool const isSelectedLoad = SelectedLoadIndex() == static_cast<int>(index);
  int const current = isSelectedLoad ? _selectedKey : -1;
  bool const draggingKey = !compact && isSelectedLoad && _keyDragIndex >= 0;
  ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.0f, kPlotFitPaddingY));
  // Compact: a read-only strip for a collapsed card, without tick labels.
  ImPlotFlags const flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMenus |
      (compact ? ImPlotFlags_NoInputs | ImPlotFlags_NoMouseText : ImPlotFlags_None);
  ImPlotAxisFlags const decorations =
      compact ? ImPlotAxisFlags_NoDecorations : ImPlotAxisFlags_None;
  // While a key is dragged the value axis holds still: refitting it to the dragged key would move
  // the axis under the cursor and run away.
  ImPlotAxisFlags const valueFit = draggingKey ? ImPlotAxisFlags_None : ImPlotAxisFlags_AutoFit;
  bool anyHeld = false;
  if (ImPlot::BeginPlot(
          compact ? "##profileCompact" : "##profile", ImVec2(-1, compact ? 40 : 132), flags)) {
    ImPlot::SetupAxes(nullptr, nullptr, decorations, decorations | valueFit);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, end, ImPlotCond_Always);
    ImPlot::PlotLine(
        "profile",
        times.data(),
        values.data(),
        kPreviewSamples,
        {ImPlotProp_LineWeight, kPlotLineWeight});
    ImVec4 const lineColor = ImPlot::GetLastItemColor();
    if (_hasSample) {
      double now = _lastSample.time;
      ImPlot::DragLineX(0, &now, ImVec4(1, 1, 0, 1), 1, ImPlotDragToolFlags_NoInputs);
    }

    ImVec4 const selectedColor{
        kSelectedLoadColor.x, kSelectedLoadColor.y, kSelectedLoadColor.z, kSelectedLoadColor.w};
    auto& keys = profile.keys;
    for (size_t k = 0; isKeyframes && k < keys.size(); ++k) {
      tuner::ProfileKey& key = keys[k];
      int const id = static_cast<int>(k);
      bool const isSelected = id == current;
      ImVec4 const color = isSelected ? selectedColor : lineColor;
      double x = key.time;
      double y = key.value;
      if (compact) {
        ImPlot::DragPoint(
            id, &x, &y, color, 3.0f, ImPlotDragToolFlags_NoInputs | ImPlotDragToolFlags_NoFit);
        continue;
      }
      bool clicked = false;
      bool hovered = false;
      bool held = false;
      if (ImPlot::DragPoint(
              id,
              &x,
              &y,
              color,
              isSelected ? 6.0f : 4.5f,
              ImPlotDragToolFlags_NoFit,
              &clicked,
              &hovered,
              &held)) {
        key.time = mochi::Clamp(x, 0.0, end);
        key.value = static_cast<real>(y);
        changed = true;
      }
      if (clicked || held) {
        SelectLoad(static_cast<int>(index));
        _selectedKey = id;
      }
      if (held) {
        _keyDragIndex = id;
        anyHeld = true;
      }
      char number[16];
      std::snprintf(number, sizeof(number), "%d", id);
      ImVec2 const at = ImPlot::PlotToPixels(key.time, key.value);
      ImVec2 const textSize = ImGui::CalcTextSize(number);
      ImPlot::GetPlotDrawList()->AddText(
          ImVec2(at.x - 0.5f * textSize.x, at.y - 9.0f - textSize.y),
          ImGui::GetColorU32(color),
          number);
      if (hovered && !held) {
        ImGui::SetTooltip(
            "Key %d at %.3f s: %.4g. Drag to move.", id, key.time, static_cast<double>(key.value));
      }
    }
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleVar();

  // The dragged key's load is the selected one, so other cards' previews leave it alone.
  if (!compact && isSelectedLoad && _keyDragIndex >= 0 && !anyHeld) {
    if (_keyDragIndex < static_cast<int>(profile.keys.size())) {
      _selectedKey = SortByTimeTracking(profile.keys, static_cast<size_t>(_keyDragIndex));
    }
    _keyDragIndex = -1;
  }
}

bool TunerEditor::ShowLoadBody(size_t index, bool& structural) {
  auto& loads = _tunerAsset->GetSpec().loads;
  tuner::LoadSpec& load = loads[index];
  bool changed = false;

  bool const nameCollides = std::count_if(loads.begin(), loads.end(), [&](auto const& other) {
                              return other.name == load.name;
                            }) > 1;
  if (nameCollides || load.name.empty()) {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.45f, 0.12f, 0.12f, 1.0f));
  }
  structural |= ImGui::InputText("Name", &load.name, ImGuiInputTextFlags_CharsNoBlank);
  if (nameCollides || load.name.empty()) {
    ImGui::PopStyleColor();
  }
  Tooltip(
      nameCollides ? "Load names must be unique"
          : load.name.empty()
          ? "Load names cannot be empty"
          : "Load's name as shown in plots and recordings (load.<name>.mag). Must be unique.");

  structural |= DescribedCombo("Type", load.type, kLoadTypes);
  structural |= TargetPicker("Target", load.target, GetTargetCandidates(load.type));
  bool const isPath = load.type == tuner::LoadType::PosePath;
  Tooltip(
      load.type == tuner::LoadType::JointEffort
          ? "Joint to drive"
          : "Non-static rigid body, or articulated link. Can be in a nested prefab.");

  if (IsForce(load.type)) {
    tuner::LoadFrame const previousFrame = load.frame;
    if (DescribedCombo("Frame", load.frame, kLoadFrames)) {
      changed = true;
      // Keep the point (and each key's) where it is in the world (at the staged pose) across the
      // switch.
      bool const toWorldPoint = load.frame == tuner::LoadFrame::WorldPoint;
      if (toWorldPoint != (previousFrame == tuner::LoadFrame::WorldPoint)) {
        if (StagedActor const* staged = FindStagedBody(load.target)) {
          mochi::TransformRT const& worldFromBody = staged->worldTransform;
          if (toWorldPoint) {
            load.pointWorld = worldFromBody.TransformPoint(load.pointLocal);
          } else {
            load.pointLocal = worldFromBody.TransformPointInverse(load.pointWorld);
          }
          for (tuner::ProfileKey& key : load.profile.keys) {
            key.point = toWorldPoint ? worldFromBody.TransformPoint(key.point)
                                     : worldFromBody.TransformPointInverse(key.point);
          }
        }
      }
    }
  } else if (load.type == tuner::LoadType::Torque) {
    // A torque has no application point, so World Point would mean the same as World.
    std::array<NamedOption, 2> const torqueFrames{kLoadFrames[0], kLoadFrames[1]};
    changed |= DescribedCombo("Frame", load.frame, torqueFrames);
  }
  if (!isPath) {
    bool const isEffort = load.type == tuner::LoadType::JointEffort;
    if (tuner::AnimatesDirection(load)) {
      // Read-only: shows the key the gizmo edits.
      Real3 shown = DirectionAt(load, KeyShownFor(index));
      ImGui::BeginDisabled();
      ImGui::DragRealXYZ(
          isEffort ? "DoF Weights (Keyframe controlled)" : "Direction (Keyframe controlled)",
          shown,
          0.01f);
      ImGui::EndDisabled();
      Tooltip(
          "Set per key in the Profile's key table, or with the gizmo on the selected key. Shows "
          "the selected key's value (the first key's when none is selected).");
    } else {
      changed |= ImGui::DragRealXYZ(isEffort ? "DoF Weights" : "Direction", load.direction, 0.01f);
      Tooltip(
          isEffort
              ? "Per-DoF multiplier on the profile. A revolute or prismatic joint uses only X."
              : "Direction of the force or torque axis. Normalized when applied; the profile sets "
                "the magnitude.");
    }
  }

  if (IsForce(load.type) || isPath) {
    bool const worldPoint = !isPath && load.frame == tuner::LoadFrame::WorldPoint;
    bool const pointKeyed = tuner::AnimatesPoint(load);
    if (pointKeyed) {
      // Read-only: shows the key the gizmo, Pick Point and Center of Mass edit.
      Real3 shown = PointAt(load, KeyShownFor(index));
      ImGui::BeginDisabled();
      ImGui::DragRealXYZ("Point (Keyframe controlled)", shown, 0.001f, 0.0f, 0.0f, "%.4f m");
      ImGui::EndDisabled();
      Tooltip(
          "Set per key in the Profile's key table, with the gizmo, or with Pick Point and Center "
          "of Mass, which set the selected key's point. Shows the selected key's value (the first "
          "key's when none is selected).");
    } else {
      changed |= ImGui::DragRealXYZ(
          isPath ? "Grab Point" : "Point",
          worldPoint ? load.pointWorld : load.pointLocal,
          0.001f,
          0.0f,
          0.0f,
          "%.4f m");
      Tooltip(
          isPath ? "Point on the body, in its local frame, that follows the path. A change "
                   "during a run applies on the next run."
              : worldPoint ? "Application point in world space; it stays put while the body moves."
                           : "Application point in the body's (or link's) local frame.");
    }
    bool const placing = _placingLoad == static_cast<int>(index);
    ImGui::BeginDisabled(_mochiScene.IsSimulating());
    if (ImGui::Button(
            placing ? ICON_FA_CROSSHAIRS " Click a Body... (Esc)"
                    : ICON_FA_CROSSHAIRS " Pick Point")) {
      _placingLoad = placing ? -1 : static_cast<int>(index);
    }
    ImGui::EndDisabled();
    Tooltip(
        _mochiScene.IsSimulating()
            ? "Stop the simulation to pick a point."
            : "Click on a body in the viewport to set both Target and Point");
    ImGui::SameLine();
    ImGui::BeginDisabled(load.target.empty());
    if (ImGui::Button(ICON_FA_DOT_CIRCLE " Center of Mass")) {
      std::optional<Real3> const com = _describedTargets
          ? tuner::FindCenterOfMassLocal(*_describedTargets, load.target)
          : std::nullopt;
      if (!com) {
        MOCHI_LOG_ERROR("Asset Tuner: no center of mass for '%s'", load.target.c_str());
      } else {
        if (isPath) {
          // Shift each waypoint by the point's move at that waypoint's orientation, so the body
          // still follows the same motion.
          Real3 const offset = *com - load.pointLocal;
          for (tuner::PoseWaypoint& waypoint : load.waypoints) {
            waypoint.position = waypoint.position +
                mochi::Quaternion::FromRotationVector(waypoint.rotation) * offset;
          }
        }
        StagedActor const* staged = FindStagedBody(load.target);
        if (pointKeyed) {
          // The shown key's point, in the load's frame.
          if (!worldPoint) {
            PointAt(load, KeyShownFor(index)) = *com;
          } else if (staged != nullptr) {
            PointAt(load, KeyShownFor(index)) = staged->worldTransform.TransformPoint(*com);
          }
        } else {
          load.pointLocal = *com;
          if (staged != nullptr) {
            load.pointWorld = staged->worldTransform.TransformPoint(*com);
          }
        }
        changed = true;
      }
    }
    ImGui::EndDisabled();
    Tooltip(
        isPath ? "Move Point to the body's center of mass, shifting every waypoint so the body "
                 "follows the same motion. Driving at the center of mass keeps the position spring "
                 "from also twisting the body. During a run this applies on the next run."
            : worldPoint
            ? "Move Point to where the body's center of mass is now."
            : "Move Point to the body's center of mass, so the force applies no torque.");
  }

  if (isPath) {
    ShowPosePathEditor(index, changed);
    return changed || structural;
  }
  ImGui::SeparatorText("Profile");
  ShowProfileEditor(index, changed);
  return changed || structural;
}

namespace {

// Predicts the most the drive pulls on a body that cannot move. Shows the spring and the damper
// apart: the spring pulls the whole time, the damper only while the path moves.
void ShowDrivePull(
    tuner::LoadSpec const& load,
    std::optional<tuner::DriveScale> const& scale,
    double dt) {
  tuner::PathDrive const& drive = load.drive;
  if (!scale) {
    ImGui::TextDisabled("Pick a dynamic body or link as Target to estimate the pull.");
    return;
  }
  if (load.waypoints.empty() || !(drive.responseHz > 0)) {
    ImGui::TextDisabled(
        load.waypoints.empty() ? "Add waypoints to estimate the pull."
                               : "Response is 0: the drive does not pull.");
    return;
  }
  // Waypoints can be out of order while one is being dragged on the timeline.
  tuner::LoadSpec sorted = load;
  tuner::SortByTime(sorted);
  tuner::DrivePullEstimate const pull = tuner::EstimateDrivePull(sorted, *scale, dt);
  bool const turning = drive.rotation && drive.rotResponseHz > 0;

  if (ImGui::BeginTable("##drivePull", turning ? 3 : 2, kListTableFlags)) {
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 1.4f);
    ImGui::TableSetupColumn("Force", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    if (turning) {
      ImGui::TableSetupColumn("Torque", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    }
    ImGui::TableHeadersRow();
    // One table row. Negative values show as the `negative` text.
    auto const row = [&](char const* label,
                         char const* tip,
                         bool textDisabled,
                         std::array<double, 2> values,
                         std::array<char const*, 2> units,
                         char const* negative) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      textDisabled ? ImGui::TextDisabled("%s", label) : ImGui::TextUnformatted(label);
      Tooltip(tip);
      for (int column = 0; column < (turning ? 2 : 1); ++column) {
        ImGui::TableNextColumn();
        char text[64];
        if (values[column] < 0) {
          std::snprintf(text, sizeof(text), "%s", negative);
        } else {
          std::snprintf(text, sizeof(text), "%.3g %s", values[column], units[column]);
        }
        textDisabled ? ImGui::TextDisabled("%s", text) : ImGui::TextUnformatted(text);
      }
    };
    auto const value = [](real v) { return static_cast<double>(v); };
    double const radToDeg = 180.0 / std::numbers::pi;
    row("Spring",
        "Theoretical max force the spring applies on a body that is fully blocked from moving.",
        false,
        {value(pull.springForce), value(pull.springTorque)},
        {"N", "N m"},
        "no cap");
    row("   reached at",
        "How far behind the path the body is when the spring reaches its max.",
        true,
        {value(pull.capLag) * 1e3, value(pull.capAngle) * radToDeg},
        {"mm", "deg"},
        "-");
    row("Damper",
        "Theoretical max force damping adds, depends on how fast the target point moves.",
        false,
        {value(pull.damperForce), value(pull.damperTorque)},
        {"N", "N m"},
        "-");
    row("   at path velocity",
        "The path's fastest velocity along the entire path.",
        true,
        {value(pull.peakVelocity), value(pull.peakAngularVelocity)},
        {"m/s", "rad/s"},
        "-");
    auto const peak = [&](real spring, real damper) {
      return spring < 0 ? -1.0 : value(spring + damper);
    };
    row("Peak",
        "Maximum combined force between spring and damper",
        false,
        {peak(pull.springForce, pull.damperForce), peak(pull.springTorque, pull.damperTorque)},
        {"N", "N m"},
        "unbounded");
    row("Estimated drive mass, inertia",
        "The drive's strength scales with these. For an articulation link, this is the whole articulation's mass.",
        false,
        {value(scale->mass), value(scale->inertia)},
        {"kg", "kg m^2"},
        "-");
    ImGui::EndTable();
  }
}

} // namespace

void TunerEditor::ShowPosePathEditor(size_t index, bool& changed) {
  tuner::LoadSpec& load = _tunerAsset->GetSpec().loads[index];
  tuner::PathDrive& drive = load.drive;
  std::optional<tuner::DriveScale> const scale =
      _describedTargets ? tuner::FindDriveScale(*_describedTargets, load.target) : std::nullopt;
  ImGui::SeparatorText("Drive");
  auto const dragGain = [&](char const* label, real& value, char const* format, char const* tip) {
    changed |= ImGui::DragReal(label, &value, 0.05f, 0.0f, 10000.0f, format);
    Tooltip(tip);
  };
  // Shows a max acceleration as a force (or torque). Edits convert back using the mass (or
  // inertia).
  auto const dragCap =
      [&](char const* label, real& acceleration, real per, char const* format, char const* tip) {
        ImGui::BeginDisabled(!(per > 0));
        real cap = acceleration * per;
        float const velocity =
            std::max(0.005f * static_cast<float>(cap), 0.01f * static_cast<float>(per));
        char const* const shown = acceleration > 0 ? format : "no cap";
        if (ImGui::DragReal(label, &cap, velocity, 0.0f, FLT_MAX, shown) && per > 0) {
          acceleration = cap / per;
          changed = true;
        }
        ImGui::EndDisabled();
        Tooltip(tip);
      };
  dragGain(
      "Response",
      drive.responseHz,
      "%.2f Hz",
      "How quickly the Grab Point is pulled onto the path: higher follows more stiffly. Gains scale "
      "with the body's mass (for a link, its whole articulation's)");
  dragGain("Damping Ratio", drive.dampingRatio, "%.2f", "1: critically damped, no overshoot");
  dragGain(
      "Max Acceleration",
      drive.maxAcceleration,
      "%.1f m/s^2",
      "Limits the spring's pull to mass * Max Acceleration. The damper is not limited and adds to "
      "it while the path moves. 0: no limit.");
  dragCap(
      "Max Pull",
      drive.maxAcceleration,
      scale ? scale->mass : real{0},
      "%.4g N",
      "The spring's max pull in newtons. Sets Max Acceleration to this / mass. Needs a Target with "
      "a known mass.");
  changed |= ImGui::Checkbox("Drive Orientation", &drive.rotation);
  Tooltip(
      "Also turn the body to each waypoint's orientation. Off: only the point follows the path "
      "and the body turns freely about it.");
  ImGui::BeginDisabled(!drive.rotation);
  dragGain(
      "Rotation Response",
      drive.rotResponseHz,
      "%.2f Hz",
      "How quickly the body is turned to the path's orientation.");
  dragGain("Rotation Damping", drive.rotDampingRatio, "%.2f", "1: critically damped, no overshoot");
  dragGain(
      "Max Angular Accel.",
      drive.maxAngularAcceleration,
      "%.1f rad/s^2",
      "Limits the turning spring's torque to inertia * Max Angular Accel. The damper is not "
      "limited and adds to it while the path turns. 0: no limit.");
  dragCap(
      "Max Torque",
      drive.maxAngularAcceleration,
      scale ? scale->inertia : real{0},
      "%.4g N m",
      "The turning spring's max torque in N-m. Sets Max Angular Accel. to this / "
      "inertia. Needs a Target with a known inertia.");
  ImGui::EndDisabled();

  ImGui::SeparatorText("Max Pull Force Prediction");
  ShowDrivePull(load, scale, _tunerAsset->GetSpec().run.dt);

  ImGui::SeparatorText("Waypoints");
  auto& waypoints = load.waypoints;
  bool const isSelectedLoad = SelectedLoadIndex() == static_cast<int>(index);
  int const current = isSelectedLoad ? SelectedWaypointIndex() : -1;
  auto const select = [&](int waypoint) {
    SelectLoad(static_cast<int>(index));
    _selectedWaypoint = waypoint;
  };
  if (waypoints.empty()) {
    ImGui::TextDisabled("No waypoints yet. Add one below.");
  } else {
    ShowWaypointTimeline(index, false, changed);
  }

  int removeWaypoint = -1;
  int sortAfterEdit = -1;
  if (!waypoints.empty() && ImGui::BeginTable("##waypoints", 5, kListTableFlags)) {
    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthStretch, 0.15f);
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("Drive", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("To Next", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableHeadersRow();
    double const minTime = 0.0;
    for (size_t w = 0; w < waypoints.size(); ++w) {
      tuner::PoseWaypoint& waypoint = waypoints[w];
      int const id = static_cast<int>(w);
      ImGui::PushID(id);
      ImGui::TableNextRow();
      // Editing any cell of a row selects its waypoint, like clicking the row.
      auto const selectOnActivate = [&] {
        if (ImGui::IsItemActivated()) {
          select(id);
        }
      };

      ImGui::TableNextColumn();
      char number[16];
      std::snprintf(number, sizeof(number), "%d", id);
      ImGui::PushStyleColor(ImGuiCol_Text, DriveModeColor(waypoint.drive));
      if (ImGui::Selectable(
              number,
              current == id,
              ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap,
              ImVec2(0.0f, ImGui::GetFrameHeight()))) {
        select(id);
      }
      ImGui::PopStyleColor();

      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-FLT_MIN);
      changed |= ImGui::DragScalar(
          "##time", ImGuiDataType_Double, &waypoint.time, 0.01f, &minTime, nullptr, "%.3f s");
      selectOnActivate();
      if (ImGui::IsItemDeactivatedAfterEdit()) {
        sortAfterEdit = id;
      }
      Tooltip("Simulation time the body should reach this pose");

      ImGui::TableNextColumn();
      ImGui::SetNextItemWidth(-FLT_MIN);
      changed |= DescribedCombo("##drive", waypoint.drive, kDriveModes);
      selectOnActivate();

      ImGui::TableNextColumn();
      if (w + 1 < waypoints.size()) {
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= DescribedCombo("##interpolation", waypoint.interpolation, kInterpolations);
        selectOnActivate();
      } else {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("end");
        Tooltip("The last waypoint's pose and drive hold until the run ends.");
      }

      ImGui::TableNextColumn();
      ImGui::BeginDisabled(waypoints.size() <= 1);
      if (ImGui::SmallButton(ICON_FA_TRASH)) {
        removeWaypoint = id;
      }
      ImGui::EndDisabled();
      Tooltip(
          waypoints.size() <= 1 ? "A path needs at least one waypoint." : "Remove this waypoint.");
      ImGui::PopID();
    }
    ImGui::EndTable();
  }
  if (removeWaypoint >= 0) {
    waypoints.erase(waypoints.begin() + removeWaypoint);
    if (isSelectedLoad) {
      _selectedWaypoint = std::min(
          current - (current > removeWaypoint ? 1 : 0), static_cast<int>(waypoints.size()) - 1);
    }
    changed = true;
  } else if (sortAfterEdit >= 0) {
    _selectedWaypoint = SortByTimeTracking(waypoints, static_cast<size_t>(sortAfterEdit));
  }

  StagedActor const* staged = FindStagedBody(load.target);
  auto const bodyPose = [&](tuner::PoseWaypoint& waypoint) {
    waypoint.position = staged->worldTransform.TransformPoint(load.pointLocal);
    waypoint.rotation = staged->worldTransform.GetRotation().ToRotationVector();
  };
  bool const simulating = _mochiScene.IsSimulating();
  if (ImGui::Button(ICON_FA_PLUS " Add Waypoint")) {
    tuner::PoseWaypoint waypoint;
    if (!waypoints.empty()) {
      waypoint = waypoints.back();
      waypoint.time += 1.0;
    }
    if (simulating && _hasSample &&
        (waypoints.empty() || _lastSample.time > waypoints.back().time)) {
      waypoint.time = _lastSample.time;
    }
    if (staged != nullptr && (simulating || waypoints.empty())) {
      bodyPose(waypoint);
    }
    waypoints.push_back(waypoint);
    select(static_cast<int>(waypoints.size()) - 1);
    changed = true;
  }
  Tooltip(
      "Add a waypoint after the last.\n"
      "While simulating will take the current body's pose.\n"
      "Otherwise will copy the last waypoint +1 second.");
  // Re-read: Add Waypoint may just have selected this load and its new waypoint.
  int const selected =
      SelectedLoadIndex() == static_cast<int>(index) ? SelectedWaypointIndex() : -1;
  ImGui::SameLine();
  ImGui::BeginDisabled(selected < 0 || staged == nullptr);
  if (ImGui::Button(ICON_FA_CROSSHAIRS " Set to Current Pose")) {
    bodyPose(waypoints[static_cast<size_t>(selected)]);
    changed = true;
  }
  ImGui::EndDisabled();
  Tooltip(
      "Set the selected waypoint to the body's current pose. Will use live pose while simulating.");

  if (selected >= 0) {
    tuner::PoseWaypoint& waypoint = waypoints[static_cast<size_t>(selected)];
    ImGui::Spacing();
    ImGui::Text("Waypoint %d", selected);
    ImGui::SameLine();
    ImGui::Text("at %.3f s", waypoint.time);
    changed |= ImGui::DragRealXYZ("Position", waypoint.position, 0.001f, 0.0f, 0.0f, "%.4f m");
    Tooltip("Where the Grab Point should be, in world space.");
    mochi::Quaternion rotation = mochi::Quaternion::FromRotationVector(waypoint.rotation);
    if (ImGui::DragQuaternion("Rotation", rotation)) {
      waypoint.rotation = rotation.ToRotationVector();
      changed = true;
    }
    Tooltip("Body orientation in world space, as roll, pitch and yaw.");
  }
}

void TunerEditor::ShowWaypointTimeline(size_t index, bool compact, bool& changed) {
  tuner::LoadSpec& load = _tunerAsset->GetSpec().loads[index];
  auto& waypoints = load.waypoints;
  bool const isSelectedLoad = SelectedLoadIndex() == static_cast<int>(index);
  int const current = isSelectedLoad ? SelectedWaypointIndex() : -1;
  // Mid-drag the waypoints may be out of order, so the spans come from a sorted copy.
  std::vector<tuner::PoseWaypoint> const sorted = SortedByTime(waypoints);
  double const end =
      std::max({_tunerAsset->GetSpec().run.duration, sorted.back().time, kTimeEpsilon});

  // Compact: a read-only strip for a collapsed card, without tick labels or waypoint numbers.
  float const height = compact ? 32.0f : 76.0f;
  ImPlotFlags const flags = ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect |
      ImPlotFlags_NoMouseText | (compact ? ImPlotFlags_NoInputs : ImPlotFlags_None);
  if (!ImPlot::BeginPlot(
          compact ? "##waypointTimelineCompact" : "##waypointTimeline",
          ImVec2(-1, height),
          flags)) {
    return;
  }
  ImPlot::SetupAxes(
      nullptr,
      nullptr,
      ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoMenus |
          (compact ? ImPlotAxisFlags_NoDecorations : ImPlotAxisFlags_None),
      ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoDecorations);
  ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, end, ImPlotCond_Always);
  ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);

  // Each waypoint's drive mode holds until the next; the first's also covers the time before it.
  ImDrawList* drawList = ImPlot::GetPlotDrawList();
  ImPlot::PushPlotClipRect();
  for (size_t w = 0; w < sorted.size(); ++w) {
    double const from = w == 0 ? 0.0 : sorted[w].time;
    double const to = w + 1 < sorted.size() ? sorted[w + 1].time : end;
    ImVec4 color = DriveModeColor(sorted[w].drive);
    color.w = 0.35f;
    drawList->AddRectFilled(
        ImPlot::PlotToPixels(from, 0.8), ImPlot::PlotToPixels(to, 0.2), ImGui::GetColorU32(color));
  }
  ImPlot::PopPlotClipRect();

  if (_hasSample) {
    double now = _lastSample.time;
    ImPlot::DragLineX(0, &now, ImVec4(1, 1, 0, 1), 1, ImPlotDragToolFlags_NoInputs);
  }

  ImVec4 const selectedColor{
      kSelectedLoadColor.x, kSelectedLoadColor.y, kSelectedLoadColor.z, kSelectedLoadColor.w};
  bool anyHeld = false;
  for (size_t w = 0; w < waypoints.size(); ++w) {
    tuner::PoseWaypoint& waypoint = waypoints[w];
    int const id = static_cast<int>(w);
    bool const isSelected = id == current;
    ImVec4 const color = isSelected ? selectedColor : DriveModeColor(waypoint.drive);
    double x = waypoint.time;
    double y = 0.5;
    if (compact) {
      ImPlot::DragPoint(id, &x, &y, color, 4.0f, ImPlotDragToolFlags_NoInputs);
      continue;
    }
    bool clicked = false;
    bool hovered = false;
    bool held = false;
    // Delayed: drawn at the passed-in position, so the marker stays on its row while dragged.
    if (ImPlot::DragPoint(
            id,
            &x,
            &y,
            color,
            isSelected ? 7.0f : 5.0f,
            ImPlotDragToolFlags_Delayed,
            &clicked,
            &hovered,
            &held)) {
      waypoint.time = mochi::Clamp(x, 0.0, end);
      changed = true;
    }
    if (clicked || held) {
      SelectLoad(static_cast<int>(index));
      _selectedWaypoint = id;
    }
    if (held) {
      _timelineDragWaypoint = id;
      anyHeld = true;
    }
    char number[16];
    std::snprintf(number, sizeof(number), "%d", id);
    ImVec2 const at = ImPlot::PlotToPixels(waypoint.time, 0.5);
    ImVec2 const textSize = ImGui::CalcTextSize(number);
    drawList->AddText(
        ImVec2(at.x - 0.5f * textSize.x, at.y - 10.0f - textSize.y),
        ImGui::GetColorU32(color),
        number);
  }
  ImPlot::EndPlot();

  // The dragged waypoint's load is the selected one, so other paths' timelines leave it alone.
  if (!compact && isSelectedLoad && _timelineDragWaypoint >= 0 && !anyHeld) {
    if (_timelineDragWaypoint < static_cast<int>(waypoints.size())) {
      _selectedWaypoint = SortByTimeTracking(waypoints, static_cast<size_t>(_timelineDragWaypoint));
    }
    _timelineDragWaypoint = -1;
  }
}

void TunerEditor::ShowLoadsWindow(bool* open) {
  if (!ImGui::Begin("Asset Tuner Loads", open)) {
    ImGui::End();
    return;
  }
  auto& loads = _tunerAsset->GetSpec().loads;
  _loadCards.Resize(loads.size());
  bool changed = false;
  bool structural = false;

  if (_mochiScene.IsSimulating()) {
    if (LiveLoadsCompatible()) {
      ImGui::TextDisabled("Edits apply live while simulating.");
    } else {
      ImGui::TextColored(
          kWarningColor,
          ICON_FA_EXCLAMATION_TRIANGLE
          " Added, removed, renamed, reordered or retargeted loads, and moved pose path points, "
          "apply on the next run.");
    }
  }
  ImGui::Separator();
  if (loads.empty()) {
    ImGui::TextDisabled("No loads yet. Add one below.");
  }

  _hoveredLoad = -1;
  int const selectedIndex = SelectedLoadIndex();
  int removeIndex = -1;
  int duplicateIndex = -1;
  ReorderableBubbleList list;
  list.count = loads.size();
  list.id = [&](size_t i) { return _loadCards.cards[i].id; };
  list.bubbleColor = [&](size_t i, bool isPlaceholder) {
    bool const isSelected = static_cast<int>(i) == selectedIndex;
    return CardColor(
        LoadAccent(loads[i].type),
        !isPlaceholder && loads[i].enabled,
        !isPlaceholder && isSelected,
        selectedIndex >= 0 && !isSelected);
  };
  list.ghostLabel = [&](size_t i) { return std::string(std::string_view(loads[i].name)); };
  list.move = [&](size_t from, size_t to) {
    if (from != to) {
      MoveItem(loads, from, to);
      MoveItem(_loadCards.cards, from, to);
      _placingLoad = -1;
      structural = true;
    }
  };
  list.drawContent = [&](size_t i) {
    tuner::LoadSpec& load = loads[i];
    float const top = ImGui::GetItemRectMin().y; // the drag grip, first item of the row
    std::string const label = std::string(std::string_view(load.name)) + "  [" +
        kLoadTypes[static_cast<size_t>(load.type)].name + ": " +
        (load.target.empty() ? std::string("no target")
                             : std::string(std::string_view(load.target))) +
        "]";
    bool const wasEnabled = load.enabled;
    bool remove = false;
    bool const bodyOpen = DrawBubbleHeader(
        i,
        &load.enabled,
        "When unchecked the load is not applied (its channel reads 0).",
        label,
        _loadCards.cards[i].collapsed,
        "Remove this load.",
        remove);
    changed |= load.enabled != wasEnabled;
    if (remove) {
      removeIndex = static_cast<int>(i);
    }
    if (bodyOpen) {
      ImGui::BeginDisabled(!load.enabled);
      changed |= ShowLoadBody(i, structural);
      ImGui::Separator();
      if (ImGui::Button(ICON_FA_COPY " Duplicate")) {
        duplicateIndex = static_cast<int>(i);
      }
      Tooltip("Add a copy of this load right below it.");
      ImGui::EndDisabled();
    } else {
      // Collapsed: a short read-only preview of what the load does over the run.
      ImGui::BeginDisabled(!load.enabled);
      if (load.type != tuner::LoadType::PosePath) {
        bool unused = false;
        ShowProfilePreview(i, true, unused);
      } else if (!load.waypoints.empty()) {
        bool unused = false;
        ShowWaypointTimeline(i, true, unused);
      }
      ImGui::EndDisabled();
    }
    float const bottom = ImGui::GetCursorScreenPos().y;
    ImVec2 const windowPos = ImGui::GetWindowPos();
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
        ImGui::IsMouseHoveringRect(
            ImVec2(windowPos.x, top),
            ImVec2(windowPos.x + ImGui::GetWindowWidth(), bottom),
            false)) {
      _hoveredLoad = static_cast<int>(i);
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        SelectLoad(static_cast<int>(i));
      }
    }
  };
  DrawReorderableBubbleList(_loadCards.drag, list);

  if (removeIndex >= 0) {
    loads.erase(loads.begin() + removeIndex);
    _loadCards.cards.erase(_loadCards.cards.begin() + removeIndex);
    _placingLoad = -1;
    structural = true;
  } else if (duplicateIndex >= 0) {
    tuner::LoadSpec copy = loads[static_cast<size_t>(duplicateIndex)];
    copy.name = MakeUniqueLoadName(std::string_view(copy.name));
    loads.push_back(std::move(copy));
    _loadCards.AddExpanded();
    MoveItem(loads, loads.size() - 1, static_cast<size_t>(duplicateIndex) + 1);
    MoveItem(_loadCards.cards, loads.size() - 1, static_cast<size_t>(duplicateIndex) + 1);
    SelectLoad(duplicateIndex + 1);
    _placingLoad = -1;
    structural = true;
  }

  if (ImGui::BeginCombo(
          "##addload",
          ICON_FA_PLUS " Add Load",
          ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge)) {
    constexpr std::array<char const*, 4> kDefaultNames{"force", "torque", "effort", "path"};
    for (size_t t = 0; t < kLoadTypes.size(); ++t) {
      if (ImGui::Selectable(kLoadTypes[t].name)) {
        tuner::LoadSpec load;
        load.type = static_cast<tuner::LoadType>(t);
        if (auto target = SelectedLoadTarget(load.type)) {
          load.target = std::move(*target);
        } else if (auto const candidates = GetTargetCandidates(load.type); !candidates.empty()) {
          load.target = candidates.front();
        }
        load.name = MakeUniqueLoadName(kDefaultNames[static_cast<size_t>(load.type)]);
        StagedActor const* staged = FindStagedBody(load.target);
        if (load.type == tuner::LoadType::PosePath && staged != nullptr) {
          // Starts where the body is, then lifts 10 cm over a second.
          tuner::PoseWaypoint start;
          start.position = staged->worldTransform.TransformPoint(load.pointLocal);
          start.rotation = staged->worldTransform.GetRotation().ToRotationVector();
          tuner::PoseWaypoint lifted = start;
          lifted.time = 1.0;
          lifted.position = lifted.position + Real3{0.0f, 0.0f, 0.1f};
          load.waypoints.push_back(start);
          load.waypoints.push_back(lifted);
        }
        loads.push_back(std::move(load));
        _loadCards.AddExpanded();
        SelectLoad(static_cast<int>(loads.size()) - 1);
        structural = true;
      }
      Tooltip(kLoadTypes[t].description);
    }
    ImGui::EndCombo();
  }
  Tooltip("Adds a load, targets the highlighted body if applicable.");

  if (changed || structural) {
    OnSpecEdited(structural);
  }
  ImGui::End();
}

namespace {
constexpr char kChannelPayload[] = "TUNER_CHANNELS";
constexpr float kMinPlotHeight = 80.0f;

// Drag payload: the source plot index (-1 for the channel list), then one channel name per line.
struct ChannelDrop {
  int fromPlot = -1;
  std::vector<std::string> names;
};

void SetChannelPayload(int fromPlot, std::vector<std::string> const& names) {
  std::string data = std::to_string(fromPlot);
  for (std::string const& name : names) {
    data += '\n';
    data += name;
  }
  ImGui::SetDragDropPayload(kChannelPayload, data.data(), data.size());
  if (names.size() == 1) {
    ImGui::TextUnformatted(names.front().c_str());
  } else {
    ImGui::Text("%d channels", static_cast<int>(names.size()));
  }
}

std::optional<ChannelDrop> AcceptChannelPayload() {
  ImGuiPayload const* payload = ImGui::AcceptDragDropPayload(kChannelPayload);
  if (payload == nullptr) {
    return std::nullopt;
  }
  std::string_view data(
      static_cast<char const*>(payload->Data), static_cast<size_t>(payload->DataSize));
  ChannelDrop drop;
  bool first = true;
  while (!data.empty() || first) {
    size_t const end = data.find('\n');
    std::string_view const part = data.substr(0, end);
    if (first) {
      std::from_chars(part.data(), part.data() + part.size(), drop.fromPlot);
      first = false;
    } else if (!part.empty()) {
      drop.names.emplace_back(part);
    }
    data = end == std::string_view::npos ? std::string_view() : data.substr(end + 1);
  }
  return drop;
}

bool PlotHasChannel(tuner::PlotSpec const& plot, std::string_view name) {
  return std::any_of(plot.channels.begin(), plot.channels.end(), [&](DynamicString const& c) {
    return std::string_view(c) == name;
  });
}

void AddChannel(tuner::PlotSpec& plot, std::string_view name) {
  if (!PlotHasChannel(plot, name)) {
    plot.channels.push_back(DynamicString(name));
  }
}

void RemoveChannel(tuner::PlotSpec& plot, std::string_view name) {
  auto const it =
      std::find_if(plot.channels.begin(), plot.channels.end(), [&](DynamicString const& c) {
        return std::string_view(c) == name;
      });
  if (it != plot.channels.end()) {
    plot.channels.erase(it);
  }
}

} // namespace

mochi::DynamicArray<tuner::PlotSpec>& TunerEditor::EditablePlots() {
  auto& plots = _tunerAsset->GetSpec().plots;
  if (plots.empty()) {
    tuner::PlotSpec plot;
    for (ChannelHistory const& channel : _channels) {
      if (channel.name.starts_with("load.")) {
        plot.channels.push_back(DynamicString(channel.name));
      }
    }
    plots.push_back(std::move(plot));
  }
  return plots;
}

bool TunerEditor::IsChannelPlotted(std::string_view channel) const {
  auto const& plots = _tunerAsset->GetSpec().plots;
  return std::any_of(plots.begin(), plots.end(), [&](tuner::PlotSpec const& plot) {
    return PlotHasChannel(plot, channel);
  });
}

void TunerEditor::ShowChannelList(bool& structural) {
  auto& plots = EditablePlots();
  ImGui::Checkbox("Auto-fit All", &_autoFitPlots);
  Tooltip(
      "Fit every plot to its data each frame. Turn off to pan and zoom (all plots share time).");
  if (ImGui::Button(ICON_FA_PLUS " Add Plot", ImVec2(-FLT_MIN, 0))) {
    plots.emplace_back();
    structural = true;
  }
  Tooltip("Add an empty plot area below the others, then drag channels onto it.");
  if (ImGui::Button("Clear", ImVec2(-FLT_MIN, 0))) {
    for (tuner::PlotSpec& plot : plots) {
      plot.channels.clear();
    }
    structural = true;
  }
  Tooltip("Remove every channel from every plot (the plot areas are kept).");
  ImGui::Separator();

  float const trashWidth =
      ImGui::CalcTextSize(ICON_FA_TRASH).x + ImGui::GetStyle().FramePadding.x * 2.0f;
  std::string_view openGroup;
  bool groupOpen = false;
  for (size_t c = 0; c < _channels.size(); ++c) {
    ChannelHistory const& channel = _channels[c];
    std::string_view const group = ChannelGroup(channel.name);
    if (group != openGroup) {
      if (groupOpen) {
        ImGui::TreePop();
      }
      openGroup = group;
      // Framed so group headers match the buttons below them (same height, rounded corners).
      groupOpen = ImGui::TreeNodeEx(
          std::string(group).c_str(),
          ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth);
      if (ImGui::BeginDragDropSource()) {
        std::vector<std::string> names;
        for (ChannelHistory const& other : _channels) {
          if (ChannelGroup(other.name) == group) {
            names.push_back(other.name);
          }
        }
        SetChannelPayload(-1, names);
        ImGui::EndDragDropSource();
      }
      Tooltip("Drag onto a plot to add all of these channels.");
    }
    if (!groupOpen) {
      continue;
    }
    ImGui::PushID(static_cast<int>(c));
    bool const isPlotted = IsChannelPlotted(channel.name);
    char const* const shortName =
        channel.name.c_str() + group.size() + (group.size() < channel.name.size() ? 1 : 0);
    // The row is only a drag handle; drawn as a left-aligned button to match the other controls.
    float const rowWidth = std::max(
        1.0f, ImGui::GetContentRegionAvail().x - trashWidth - ImGui::GetStyle().ItemSpacing.x);
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
    ImGui::Button(shortName, ImVec2(rowWidth, 0));
    ImGui::PopStyleVar();
    if (ImGui::BeginDragDropSource()) {
      SetChannelPayload(-1, {channel.name});
      ImGui::EndDragDropSource();
    }
    Tooltip(
        isPlotted ? "Drag onto another plot to add it there too." : "Drag onto a plot to plot it.");
    ImGui::SameLine();
    ImGui::BeginDisabled(!isPlotted);
    if (ImGui::Button(ICON_FA_TRASH)) {
      for (tuner::PlotSpec& plot : plots) {
        RemoveChannel(plot, channel.name);
      }
      structural = true;
    }
    ImGui::EndDisabled();
    Tooltip(isPlotted ? "Remove this channel from every plot." : "Not plotted.");
    ImGui::PopID();
  }
  if (groupOpen) {
    ImGui::TreePop();
  }
}

void TunerEditor::ShowPlotArea(size_t index, bool& structural, bool& outRemove) {
  auto& plots = _tunerAsset->GetSpec().plots;
  tuner::PlotSpec& plot = plots[index];

  // A left column of the bubble list's drag grip (the last item) with the remove button under it,
  // then the plot to its right.
  ImVec2 const gripMin = ImGui::GetItemRectMin();
  ImVec2 const gripMax = ImGui::GetItemRectMax();
  PushHeaderRowFramePadding();
  ImGui::BeginDisabled(plots.size() <= 1);
  if (ImGui::Button(ICON_FA_TRASH, ImVec2(gripMax.x - gripMin.x, 0.0f))) {
    outRemove = true;
  }
  ImGui::EndDisabled();
  ImGui::PopStyleVar();
  Tooltip(plots.size() <= 1 ? "The last plot cannot be removed." : "Remove this plot area.");
  ImGui::SetCursorScreenPos(ImVec2(gripMax.x + ImGui::GetStyle().ItemSpacing.x, gripMin.y));
  ImGui::BeginGroup();

  std::string pendingRemoval;
  float const height = std::max(kMinPlotHeight, plot.height);
  ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.0f, kPlotFitPaddingY));
  if (ImPlot::BeginPlot("##plot", ImVec2(-1, height))) {
    ImPlotAxisFlags const fit = _autoFitPlots ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None;
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_None, fit);
    ImPlot::SetupAxisLinks(ImAxis_X1, &_plotTimeMin, &_plotTimeMax);
    ImPlot::SetupLegend(
        ImPlotLocation_North, ImPlotLegendFlags_Outside | ImPlotLegendFlags_Horizontal);
    int const count = static_cast<int>(_sampleTimes.size());
    for (DynamicString const& name : plot.channels) {
      auto const it =
          std::find_if(_channels.begin(), _channels.end(), [&](ChannelHistory const& c) {
            return c.name == std::string_view(name);
          });
      if (it == _channels.end()) {
        continue; // not produced by the current run (e.g. its probe was removed)
      }
      ImPlot::PlotLine(
          it->name.c_str(),
          _sampleTimes.data(),
          it->values.data(),
          count,
          {ImPlotProp_LineWeight, kPlotLineWeight});
      if (ImPlot::BeginDragDropSourceItem(it->name.c_str())) {
        SetChannelPayload(static_cast<int>(index), {it->name});
        ImPlot::EndDragDropSource();
      }
      if (ImPlot::BeginLegendPopup(it->name.c_str())) {
        if (ImGui::MenuItem("Remove from plot")) {
          pendingRemoval = it->name;
        }
        ImPlot::EndLegendPopup();
      }
    }
    if (plot.channels.empty()) {
      ImVec2 const pos = ImPlot::GetPlotPos();
      ImVec2 const size = ImPlot::GetPlotSize();
      char const* const hint = "Drag channels here";
      ImVec2 const textSize = ImGui::CalcTextSize(hint);
      ImPlot::GetPlotDrawList()->AddText(
          ImVec2(pos.x + 0.5f * (size.x - textSize.x), pos.y + 0.5f * (size.y - textSize.y)),
          ImGui::GetColorU32(ImGuiCol_TextDisabled),
          hint);
    }
    auto const acceptDrop = [&] {
      if (auto drop = AcceptChannelPayload()) {
        for (std::string const& name : drop->names) {
          AddChannel(plot, name);
          if (drop->fromPlot >= 0 && drop->fromPlot != static_cast<int>(index) &&
              drop->fromPlot < static_cast<int>(plots.size())) {
            RemoveChannel(plots[static_cast<size_t>(drop->fromPlot)], name);
          }
        }
        structural = true;
      }
    };
    if (ImPlot::BeginDragDropTargetPlot()) {
      acceptDrop();
      ImPlot::EndDragDropTarget();
    }
    if (ImPlot::BeginDragDropTargetLegend()) {
      acceptDrop();
      ImPlot::EndDragDropTarget();
    }
    ImPlot::EndPlot();
  }
  ImPlot::PopStyleVar();
  if (!pendingRemoval.empty()) {
    RemoveChannel(plot, pendingRemoval);
    structural = true;
  }

  // Resize handle below the plot.
  ImGui::InvisibleButton("##resize", ImVec2(-1, 6));
  if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    ImVec2 const min = ImGui::GetItemRectMin();
    ImVec2 const max = ImGui::GetItemRectMax();
    float const y = 0.5f * (min.y + max.y);
    ImGui::GetWindowDrawList()->AddLine(
        ImVec2(min.x, y), ImVec2(max.x, y), ImGui::GetColorU32(ImGuiCol_SeparatorHovered), 2.0f);
  }
  if (ImGui::IsItemActive()) {
    plot.height = std::max(kMinPlotHeight, height + ImGui::GetIO().MouseDelta.y);
  }
  if (ImGui::IsItemDeactivated()) {
    structural = true;
  }
  ImGui::EndGroup();
}

void TunerEditor::ShowNewPlotDropZone(bool& structural) {
  ImVec2 const size(
      ImGui::GetContentRegionAvail().x, std::max(ImGui::GetContentRegionAvail().y, kMinPlotHeight));
  if (size.x <= 0.0f) {
    return;
  }
  ImGui::InvisibleButton("##newPlotDrop", size);
  ImGuiPayload const* const dragged = ImGui::GetDragDropPayload();
  if (dragged != nullptr && dragged->IsDataType(kChannelPayload)) {
    ImVec2 const min = ImGui::GetItemRectMin();
    ImVec2 const max = ImGui::GetItemRectMax();
    ImDrawList* const drawList = ImGui::GetWindowDrawList();
    drawList->AddRect(min, max, ImGui::GetColorU32(ImGuiCol_TextDisabled), 4.0f, 0, 1.5f);
    char const* const hint = ICON_FA_PLUS " Drop to add a new plot";
    ImVec2 const textSize = ImGui::CalcTextSize(hint);
    drawList->AddText(
        ImVec2(
            0.5f * (min.x + max.x - textSize.x),
            min.y + std::min(0.5f * (size.y - textSize.y), kMinPlotHeight * 0.5f)),
        ImGui::GetColorU32(ImGuiCol_TextDisabled),
        hint);
  }
  if (!ImGui::BeginDragDropTarget()) {
    return;
  }
  if (auto drop = AcceptChannelPayload()) {
    auto& plots = _tunerAsset->GetSpec().plots;
    tuner::PlotSpec plot;
    for (std::string const& name : drop->names) {
      AddChannel(plot, name);
      if (drop->fromPlot >= 0 && drop->fromPlot < static_cast<int>(plots.size())) {
        RemoveChannel(plots[static_cast<size_t>(drop->fromPlot)], name);
      }
    }
    plots.push_back(std::move(plot));
    structural = true;
  }
  ImGui::EndDragDropTarget();
}

void TunerEditor::ShowPlotsWindow(bool* open) {
  if (!ImGui::Begin("Asset Tuner Plots", open)) {
    ImGui::End();
    return;
  }
  if (_channels.empty()) {
    ImGui::TextDisabled(
        "Press Play to see load and probe channels. The plot layout is saved with the asset tuner.");
    ImGui::End();
    return;
  }
  bool structural = false;
  EditablePlots();
  if (_autoFitPlots) {
    _plotTimeMin = 0.0;
    _plotTimeMax =
        std::max(_sampleTimes.empty() ? 0.0 : static_cast<double>(_sampleTimes.back()), 1e-3);
  }

  ImGui::BeginChild(
      "##channels", ImVec2(220, 0), ImGuiChildFlags_ResizeX | ImGuiChildFlags_Borders);
  ShowChannelList(structural);
  ImGui::EndChild();
  // Dropping a plotted trace back on the channel list removes it from its plot.
  if (ImGui::BeginDragDropTarget()) {
    if (auto drop = AcceptChannelPayload(); drop && drop->fromPlot >= 0) {
      auto& plots = _tunerAsset->GetSpec().plots;
      if (drop->fromPlot < static_cast<int>(plots.size())) {
        for (std::string const& name : drop->names) {
          RemoveChannel(plots[static_cast<size_t>(drop->fromPlot)], name);
        }
        structural = true;
      }
    }
    ImGui::EndDragDropTarget();
  }

  ImGui::SameLine();
  ImGui::BeginChild("##plots", ImVec2(0, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
  int removePlot = -1;
  auto& plots = _tunerAsset->GetSpec().plots;
  _plotCards.Resize(plots.size());
  ReorderableBubbleList list;
  list.count = plots.size();
  list.id = [&](size_t i) { return _plotCards.cards[i].id; };
  list.bubbleColor = [&](size_t, bool isPlaceholder) {
    return CardColor(kNeutralAccent, !isPlaceholder);
  };
  list.ghostLabel = [&](size_t i) { return "Plot " + std::to_string(i + 1); };
  // As wide as the remove button below it, so the two form a column.
  list.gripWidth = ImGui::CalcTextSize(ICON_FA_TRASH).x + 2.0f * ImGui::GetStyle().FramePadding.x;
  list.move = [&](size_t from, size_t to) {
    if (from != to) {
      MoveItem(plots, from, to);
      MoveItem(_plotCards.cards, from, to);
      structural = true;
    }
  };
  list.drawContent = [&](size_t i) {
    bool remove = false;
    ShowPlotArea(i, structural, remove);
    if (remove) {
      removePlot = static_cast<int>(i);
    }
  };
  DrawReorderableBubbleList(_plotCards.drag, list);
  ImGui::TextDisabled(
      "Drag channels or probe groups onto a plot, or below the plots to add a new one. Drag a "
      "legend entry to move it, or back to the list to remove it. Right-click a legend entry for "
      "more. Drag a plot's = grip to reorder the plots.");
  ShowNewPlotDropZone(structural);
  ImGui::EndChild();

  if (removePlot >= 0 && plots.size() > 1) {
    plots.erase(plots.begin() + removePlot);
    _plotCards.cards.erase(_plotCards.cards.begin() + removePlot);
    structural = true;
  }
  if (structural) {
    _tunerAsset->SetDirty(true);
    GetUndoStack().PushNow();
  }
  ImGui::End();
}

void TunerEditor::ShowStopConditionCards(bool& structural) {
  tuner::TunerSpec& spec = _tunerAsset->GetSpec();
  auto& stops = spec.stopConditions;
  _stopCards.Resize(stops.size());

  // Channels a threshold can watch: the last run's, plus those the spec's loads and probes will
  // produce (so conditions can be set up before the first Play).
  std::vector<std::string> channels;
  auto const addChannel = [&](std::string name) {
    if (std::find(channels.begin(), channels.end(), name) == channels.end()) {
      channels.push_back(std::move(name));
    }
  };
  for (tuner::LoadSpec const& load : spec.loads) {
    std::string const prefix = "load." + std::string(std::string_view(load.name));
    addChannel(prefix + ".mag");
    if (load.type == tuner::LoadType::PosePath) {
      addChannel(prefix + ".error");
      addChannel(prefix + ".angleError");
    }
  }
  for (tuner::ProbeSpec const& probe : spec.probes) {
    std::string const target(std::string_view(probe.target));
    auto const add = [&](std::initializer_list<char const*> suffixes) {
      for (char const* suffix : suffixes) {
        addChannel(target + "." + suffix);
      }
    };
    switch (probe.kind) {
      case tuner::ProbeKind::BodyPose:
        add(
            {"pos.x",
             "pos.y",
             "pos.z",
             "rotvec.x",
             "rotvec.y",
             "rotvec.z",
             "rotvecUnwrapped.x",
             "rotvecUnwrapped.y",
             "rotvecUnwrapped.z"});
        break;
      case tuner::ProbeKind::BodyVelocity:
        add({"vel.x", "vel.y", "vel.z", "angvel.x", "angvel.y", "angvel.z"});
        break;
      case tuner::ProbeKind::ContactForce:
        add({"contactForce.x", "contactForce.y", "contactForce.z"});
        break;
      case tuner::ProbeKind::SolverStats:
        addChannel("solver.iters");
        addChannel("solver.residual");
        addChannel("solver.lineSearchIters");
        addChannel("solver.convergence");
        break;
      case tuner::ProbeKind::Performance:
        addChannel("perf.solveMs");
        break;
      case tuner::ProbeKind::Penetration:
        add({"penetration.volume"});
        break;
      case tuner::ProbeKind::JointState:
        break; // DoF count is only known once bound; the last run's channels cover these
    }
  }
  for (ChannelHistory const& channel : _channels) {
    addChannel(channel.name);
  }
  std::vector<std::string> const bodies = GetTargetCandidates(tuner::LoadType::Force);
  auto const bodyPosition = [&](std::string_view target) -> std::optional<Real3> {
    StagedActor const* staged = FindStagedBody(target);
    return staged ? std::optional(staged->worldTransform.GetTranslation()) : std::nullopt;
  };
  auto const centerBoxOn = [&](tuner::StopCondition& stop, Real3 const& center) {
    Real3 const half = (stop.boxMax - stop.boxMin) * real{0.5};
    stop.boxMin = center - half;
    stop.boxMax = center + half;
  };
  // A threshold on a probe channel the spec does not produce yet: the probe that would.
  auto const probeFor = [](std::string_view channel) -> std::optional<tuner::ProbeSpec> {
    size_t const dot =
        channel.find('.', channel.rfind('/') == std::string_view::npos ? 0 : channel.rfind('/'));
    if (dot == std::string_view::npos || channel.starts_with("load.")) {
      return std::nullopt;
    }
    std::string_view const quantity = channel.substr(dot + 1);
    tuner::ProbeSpec probe;
    probe.target = DynamicString(channel.substr(0, dot));
    if (quantity.starts_with("pos.") || quantity.starts_with("rotvec.") ||
        quantity.starts_with("rotvecUnwrapped.")) {
      probe.kind = tuner::ProbeKind::BodyPose;
    } else if (quantity.starts_with("vel.") || quantity.starts_with("angvel.")) {
      probe.kind = tuner::ProbeKind::BodyVelocity;
    } else if (quantity.starts_with("contactForce.")) {
      probe.kind = tuner::ProbeKind::ContactForce;
    } else if (quantity.starts_with("penetration.")) {
      probe.kind = tuner::ProbeKind::Penetration;
    } else {
      return std::nullopt;
    }
    return probe;
  };

  if (stops.empty()) {
    ImGui::TextDisabled("None: runs last the full duration.");
  }
  int const selectedBox = SelectedStopBoxIndex();
  int removeIndex = -1;
  ReorderableBubbleList list;
  list.count = stops.size();
  list.id = [&](size_t i) { return _stopCards.cards[i].id; };
  list.bubbleColor = [&](size_t i, bool isPlaceholder) {
    bool const triggered = _stopEvent && _stopEvent->condition == static_cast<int>(i);
    return CardColor(
        triggered ? kErrorColor : kWarningColor,
        !isPlaceholder && stops[i].enabled,
        !isPlaceholder && _stopCards.cards[i].id == _selectedStopCard);
  };
  list.ghostLabel = [&](size_t i) { return std::string(std::string_view(stops[i].name)); };
  list.move = [&](size_t from, size_t to) {
    if (from != to) {
      MoveItem(stops, from, to);
      MoveItem(_stopCards.cards, from, to);
      structural = true;
    }
  };
  list.drawContent = [&](size_t i) {
    tuner::StopCondition& stop = stops[i];
    float const top = ImGui::GetItemRectMin().y;
    bool const isBox = stop.kind != tuner::StopKind::ChannelThreshold;
    bool const enters = stop.kind == tuner::StopKind::EntersBox;
    bool const triggered = _stopEvent && _stopEvent->condition == static_cast<int>(i);
    std::string label = (triggered ? std::string(ICON_FA_HAND_PAPER " ") : std::string()) +
        std::string(std::string_view(stop.name)) + "  [";
    if (isBox) {
      label += (enters ? "enters box: " : "leaves box: ") +
          (stop.target.empty() ? std::string("no target")
                               : std::string(std::string_view(stop.target)));
    } else {
      char threshold[32];
      std::snprintf(threshold, sizeof(threshold), "%.4g", static_cast<double>(stop.threshold));
      label += (stop.channel.empty() ? std::string("no channel")
                                     : std::string(std::string_view(stop.channel))) +
          (stop.comparison == tuner::StopComparison::Above ? " > " : " < ") + threshold;
    }
    label += "]";
    bool const wasEnabled = stop.enabled;
    bool remove = false;
    bool const bodyOpen = DrawBubbleHeader(
        i,
        &stop.enabled,
        "When unchecked this condition is not checked.",
        label,
        _stopCards.cards[i].collapsed,
        "Remove this stop condition.",
        remove);
    structural |= stop.enabled != wasEnabled;
    if (remove) {
      removeIndex = static_cast<int>(i);
    }
    if (bodyOpen) {
      ImGui::BeginDisabled(!stop.enabled);
      structural |= ImGui::InputText("Name", &stop.name);
      Tooltip("Shown in the viewport and in summary.json when this condition stops a run.");
      constexpr std::array<NamedOption, 3> kStopKinds{{
          {"Leaves Box", "Stops once the target's origin leaves an axis-aligned box."},
          {"Enters Box", "Stops once the target's origin enters an axis-aligned box."},
          {"Channel Threshold",
           "Stops once a channel (e.g. block.pos.z) goes above or below a threshold."},
      }};
      structural |= DescribedCombo("Kind", stop.kind, kStopKinds);
      if (isBox) {
        structural |= TargetPicker("Target", stop.target, bodies);
        Tooltip(
            enters ? "Rigid body or articulated link whose origin must stay outside the box."
                   : "Rigid body or articulated link whose origin must stay inside the box.");
        structural |= ImGui::DragRealXYZ("Box Min", stop.boxMin, 0.005f, 0.0f, 0.0f, "%.3f m");
        structural |= ImGui::DragRealXYZ("Box Max", stop.boxMax, 0.005f, 0.0f, 0.0f, "%.3f m");
        Tooltip(
            "World-space corners of the box. Select the card to move and scale it with the "
            "viewport gizmo (W, R).");
        for (int d = 0; d < 3; ++d) {
          if (stop.boxMin[d] > stop.boxMax[d]) {
            std::swap(stop.boxMin[d], stop.boxMax[d]);
          }
        }
        std::optional<Real3> const position = bodyPosition(stop.target);
        ImGui::BeginDisabled(!position || _mochiScene.IsSimulating());
        if (ImGui::Button(ICON_FA_CROSSHAIRS " Center on Target")) {
          centerBoxOn(stop, *position);
          structural = true;
        }
        ImGui::EndDisabled();
        Tooltip("Move the box so it is centered on the target's current position.");
        if (static_cast<int>(i) == selectedBox && !_mochiScene.IsSimulating()) {
          ImGui::SameLine();
          ImGui::TextDisabled("Gizmo: W move, R scale");
        }
      } else {
        structural |= TargetPicker("Channel", stop.channel, channels);
        Tooltip("Channel to watch, as in Asset Tuner Plots and signals.csv.");
        constexpr std::array<NamedOption, 2> kComparisons{{
            {"Above", "Stops once the channel is greater than the threshold."},
            {"Below", "Stops once the channel is less than the threshold."},
        }};
        structural |= DescribedCombo("When", stop.comparison, kComparisons);
        // Step relative to the value, so tiny thresholds (e.g. overlap volumes in m^3) stay
        // editable.
        float const thresholdSpeed =
            stop.threshold != 0 ? 0.01f * std::abs(static_cast<float>(stop.threshold)) : 0.001f;
        structural |=
            ImGui::DragReal("Threshold", &stop.threshold, thresholdSpeed, 0.0f, 0.0f, "%.4g");
        Tooltip(
            "In the channel's units (m, m/s, N, m^3, ...). Position thresholds are drawn as a "
            "plane in the viewport. Ctrl+click to type a value.");
        bool const produced =
            std::find(channels.begin(), channels.end(), std::string_view(stop.channel)) !=
            channels.end();
        if (!stop.channel.empty() && !produced) {
          ImGui::TextColored(
              kWarningColor, ICON_FA_EXCLAMATION_TRIANGLE " No probe produces this channel.");
          if (std::optional<tuner::ProbeSpec> probe = probeFor(stop.channel)) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Add Probe")) {
              spec.probes.push_back(std::move(*probe));
              _probeCards.AddExpanded();
              structural = true;
            }
            Tooltip("Add the probe that produces this channel.");
          }
        }
      }
      double const minStart = 0.0;
      structural |= ImGui::DragScalar(
          "Check From", ImGuiDataType_Double, &stop.start, 0.01f, &minStart, nullptr, "%.3f s");
      Tooltip("Not checked before this simulation time, e.g. to let the scene settle.");
      structural |= ImGui::DragScalar(
          "Hold For", ImGuiDataType_Double, &stop.holdTime, 0.01f, &minStart, nullptr, "%.3f s");
      Tooltip(
          "Only stop once the condition has held continuously this long, e.g. to ignore brief "
          "spikes on impact. 0: stop at once.");
      ImGui::EndDisabled();
    }
    float const bottom = ImGui::GetCursorScreenPos().y;
    ImVec2 const windowPos = ImGui::GetWindowPos();
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        ImGui::IsMouseHoveringRect(
            ImVec2(windowPos.x, top),
            ImVec2(windowPos.x + ImGui::GetWindowWidth(), bottom),
            false)) {
      _selectedStopCard = _stopCards.cards[i].id;
      _selectedLoadCard = 0;
    }
  };
  DrawReorderableBubbleList(_stopCards.drag, list);

  if (removeIndex >= 0) {
    stops.erase(stops.begin() + removeIndex);
    _stopCards.cards.erase(_stopCards.cards.begin() + removeIndex);
    structural = true;
  }
  if (ImGui::BeginCombo(
          "##addstop",
          ICON_FA_PLUS " Add Stop Condition",
          ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge)) {
    std::string const firstBody = bodies.empty() ? std::string() : bodies.front();
    auto const add = [&](tuner::StopCondition stop) {
      stops.push_back(std::move(stop));
      _stopCards.AddExpanded();
      _selectedStopCard = _stopCards.cards.back().id;
      _selectedLoadCard = 0;
      structural = true;
    };
    if (ImGui::Selectable("Leaves Box")) {
      tuner::StopCondition stop;
      stop.name = "escaped";
      stop.kind = tuner::StopKind::LeavesBox;
      stop.target = DynamicString(firstBody);
      stop.boxMin = {-0.25f, -0.25f, -0.25f};
      stop.boxMax = {0.25f, 0.25f, 0.25f};
      if (std::optional<Real3> const position = bodyPosition(firstBody)) {
        centerBoxOn(stop, *position);
      }
      add(std::move(stop));
    }
    Tooltip("Stop once a body leaves a box (0.5 m, centered on it; adjust in the viewport).");
    if (ImGui::Selectable("Enters Box")) {
      tuner::StopCondition stop;
      stop.name = "reached";
      stop.kind = tuner::StopKind::EntersBox;
      stop.target = DynamicString(firstBody);
      stop.boxMin = {-0.25f, -0.25f, -0.25f};
      stop.boxMax = {0.25f, 0.25f, 0.25f};
      // Beside the body rather than around it, so it does not trigger on the first step.
      if (std::optional<Real3> const position = bodyPosition(firstBody)) {
        centerBoxOn(stop, *position + Real3{0.6f, 0.0f, 0.0f});
      }
      add(std::move(stop));
    }
    Tooltip(
        "Stop once a body enters a box (0.5 m, placed beside it along +X; move it in the "
        "viewport).");
    if (ImGui::Selectable("Falls Below Height")) {
      // A threshold on the body's height, 5 cm under the ground, with the probe it needs.
      tuner::StopCondition stop;
      stop.name = "fell";
      stop.kind = tuner::StopKind::ChannelThreshold;
      stop.channel = DynamicString(firstBody + ".pos.z");
      stop.comparison = tuner::StopComparison::Below;
      stop.threshold = EffectiveGroundHeight() - 0.05f;
      if (std::optional<tuner::ProbeSpec> probe = probeFor(stop.channel); probe &&
          std::find(channels.begin(), channels.end(), firstBody + ".pos.z") == channels.end()) {
        spec.probes.push_back(std::move(*probe));
        _probeCards.AddExpanded();
      }
      add(std::move(stop));
    }
    Tooltip(
        "Stop once a body's height drops 5 cm below the ground (adds its Body Pose probe if "
        "needed).");
    if (ImGui::Selectable("Interpenetration")) {
      // A threshold on the body's overlap volume, held briefly so impacts do not trigger it.
      std::string const channel = firstBody + ".penetration.volume";
      tuner::StopCondition stop;
      stop.name = "interpenetrated";
      stop.kind = tuner::StopKind::ChannelThreshold;
      stop.channel = DynamicString(channel);
      stop.comparison = tuner::StopComparison::Above;
      stop.threshold = 1e-5f;
      stop.holdTime = 0.05;
      if (std::optional<tuner::ProbeSpec> probe = probeFor(stop.channel);
          probe && std::find(channels.begin(), channels.end(), channel) == channels.end()) {
        spec.probes.push_back(std::move(*probe));
        _probeCards.AddExpanded();
      }
      add(std::move(stop));
    }
    Tooltip(
        "Stop once a body overlaps other actors by more than 10 cm^3 for 50 ms (adds its "
        "Penetration probe if needed). Normal contact already overlaps a little, and how much "
        "scales with body size and load (a resting 10 cm, 1 kg cube reads ~3 cm^3), so plot the "
        "channel first and set the threshold above it.");
    if (ImGui::Selectable("Channel Threshold")) {
      tuner::StopCondition stop;
      stop.name = "threshold";
      stop.kind = tuner::StopKind::ChannelThreshold;
      stop.channel = channels.empty() ? DynamicString() : DynamicString(channels.front());
      add(std::move(stop));
    }
    Tooltip("Stop once any channel goes above or below a value.");
    ImGui::EndCombo();
  }
  Tooltip(
      "Stop conditions end a run early: the CLI stops (status \"stopped\" in summary.json), "
      "Studio pauses and shows why.");
  if (_mochiScene.IsSimulating()) {
    ImGui::TextDisabled("Stop condition changes apply on the next run.");
  }
}

void TunerEditor::ShowProbeCards(bool& structural) {
  auto& probes = _tunerAsset->GetSpec().probes;
  _probeCards.Resize(probes.size());
  auto const candidatesFor = [&](tuner::ProbeKind kind) {
    if (kind != tuner::ProbeKind::JointState) {
      return GetTargetCandidates(tuner::LoadType::Force);
    }
    std::vector<std::string> joints;
    for (auto const& joint : _targets.joints) {
      if (joint.numDofs > 0) {
        joints.push_back(joint.name);
      }
    }
    return joints;
  };

  int removeIndex = -1;
  ReorderableBubbleList list;
  list.count = probes.size();
  list.id = [&](size_t i) { return _probeCards.cards[i].id; };
  list.bubbleColor = [&](size_t, bool isPlaceholder) {
    return CardColor(kNeutralAccent, !isPlaceholder);
  };
  list.ghostLabel = [&](size_t i) {
    return std::string(kProbeKinds[static_cast<size_t>(probes[i].kind)].name);
  };
  list.move = [&](size_t from, size_t to) {
    if (from != to) {
      MoveItem(probes, from, to);
      MoveItem(_probeCards.cards, from, to);
      structural = true;
    }
  };
  list.drawContent = [&](size_t i) {
    tuner::ProbeSpec& probe = probes[i];
    bool const hasTarget = ProbeHasTarget(probe.kind);
    std::string label = kProbeKinds[static_cast<size_t>(probe.kind)].name;
    if (hasTarget) {
      label += "  [" +
          (probe.target.empty() ? std::string("no target")
                                : std::string(std::string_view(probe.target))) +
          "]";
    }
    bool remove = false;
    bool const bodyOpen = DrawBubbleHeader(
        i, nullptr, nullptr, label, _probeCards.cards[i].collapsed, "Remove this probe.", remove);
    if (remove) {
      removeIndex = static_cast<int>(i);
    }
    if (bodyOpen) {
      structural |= DescribedCombo("Kind", probe.kind, kProbeKinds);
      if (hasTarget) {
        structural |= TargetPicker("Target", probe.target, candidatesFor(probe.kind));
        Tooltip(
            probe.kind == tuner::ProbeKind::JointState
                ? "Joint to sample, as <articulated actor>/<joint>."
                : "Rigid body or articulated link to sample.");
      }
      if (probe.kind == tuner::ProbeKind::Penetration) {
        structural |= ImGui::Checkbox("Ignore Ground", &probe.ignoreGround);
        Tooltip("Leave out overlap with ground planes (static, unbounded actors).");
      }
      ImGui::TextDisabled("Channels: %s", ProbeChannelHint(probe.kind).c_str());
    }
  };
  DrawReorderableBubbleList(_probeCards.drag, list);

  if (removeIndex >= 0) {
    probes.erase(probes.begin() + removeIndex);
    _probeCards.cards.erase(_probeCards.cards.begin() + removeIndex);
    structural = true;
  }
  if (ImGui::BeginCombo(
          "##addprobe",
          ICON_FA_PLUS " Add Probe",
          ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge)) {
    for (size_t k = 0; k < kProbeKinds.size(); ++k) {
      if (ImGui::Selectable(kProbeKinds[k].name)) {
        tuner::ProbeSpec probe;
        probe.kind = static_cast<tuner::ProbeKind>(k);
        if (ProbeHasTarget(probe.kind)) {
          auto const candidates = candidatesFor(probe.kind);
          if (!candidates.empty()) {
            probe.target = candidates.front();
          }
        }
        probes.push_back(std::move(probe));
        _probeCards.AddExpanded();
        structural = true;
      }
      Tooltip(kProbeKinds[k].description);
    }
    ImGui::EndCombo();
  }
  Tooltip("Add a probe. Its channels appear in Asset Tuner Plots and in the recorded signals.csv.");
  if (_mochiScene.IsSimulating()) {
    ImGui::TextDisabled("Probe changes apply on the next run.");
  }
}

void TunerEditor::ShowRunWindow(bool* open) {
  if (!ImGui::Begin("Asset Tuner Run", open)) {
    ImGui::End();
    return;
  }
  tuner::TunerSpec& spec = _tunerAsset->GetSpec();
  bool changed = false;
  bool structural = false;

  ImGui::SeparatorText("Prefab");
  {
    mochi::DynamicString prefabPath(_tunerAsset->GetPrefabPath().ToString());
    ImGui::InputText("Prefab", &prefabPath, ImGuiInputTextFlags_ReadOnly);
    Tooltip("The prefab this asset tuner simulates. Solver settings are from the prefab");
    bool const prefabLoaded = _tunerAsset->GetPrefabAsset() != nullptr;
    ImGui::BeginDisabled(!prefabLoaded);
    if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open Prefab")) {
      _studio->OpenPath(_tunerAsset->GetPrefabPath());
    }
    ImGui::EndDisabled();
    Tooltip("Open the prefab in its own editor tab.");
    if (!prefabLoaded) {
      ImGui::SameLine();
      ImGui::TextColored(kErrorColor, ICON_FA_EXCLAMATION_TRIANGLE " Failed to load (see log).");
    }
  }

  ImGui::SeparatorText("Recording");
  bool const recording = IsRecording();
  bool const armed = recording || _recordNextRun;
  ImGui::PushStyleColor(
      ImGuiCol_Button,
      armed ? ImVec4(0.70f, 0.12f, 0.12f, 1.0f) : ImGui::GetStyleColorVec4(ImGuiCol_Button));
  char const* const recordLabel = recording ? ICON_FA_STOP " Stop Recording###record"
      : armed                               ? ICON_FA_CIRCLE " Armed###record"
                                            : ICON_FA_CIRCLE " Record###record";
  if (ImGui::Button(recordLabel)) {
    if (armed) {
      // Stops the current recording (if any) and disarms.
      StopRecording();
      _recordNextRun = false;
    } else {
      _recordNextRun = true;
      StartRecording(); // no-op until Play when not simulating
    }
  }
  ImGui::PopStyleColor();
  Tooltip(
      recording ? "Recording. Click to stop, write the run folder, and disarm (later runs are not "
                  "recorded)."
          : armed
          ? "Armed: every run from Play is recorded. Click to disarm."
          : "Record the current run (or, when stopped, every run from Play) to a tuner_run folder"
            "specified below.\nClick again to disarm.");
  ImGui::SameLine();
  if (recording) {
    ImGui::TextColored(kErrorColor, "Recording");
  } else if (armed) {
    ImGui::TextUnformatted("Recording each run");
  } else {
    ImGui::TextDisabled("Not recording");
  }
  ImGui::Checkbox("Stop at Duration", &_stopAtDuration);
  Tooltip("Pauses the simulation (and end the recording) when reaching the run duration.");

  mochi::DynamicString outputDir(_outputDir);
  if (ImGui::InputText("Output Folder", &outputDir)) {
    _outputDir = std::string(std::string_view(outputDir));
  }
  Tooltip("Each recording creates a timestamped run folder in here.");
  if (ImGui::Button("Browse##output")) {
    mochi::Path const chosen =
        SuperDexStudio::GetFolderDialogPath("Asset Tuner Output Folder", mochi::Path{_outputDir});
    if (!chosen.IsEmpty()) {
      _outputDir = chosen.ToString();
    }
  }
  Tooltip("Pick the folder recordings are written to.");
  if (!_lastRunDir.empty()) {
    mochi::DynamicString lastRun(_lastRunDir);
    ImGui::InputText("Last Run", &lastRun, ImGuiInputTextFlags_ReadOnly);
    Tooltip("Folder of the most recent recording.");
    if (ImGui::Button(ICON_FA_COPY " Copy Path")) {
      ImGui::SetClipboardText(_lastRunDir.c_str());
    }
    Tooltip("Copy the run folder path, e.g. to pass to analysis scripts.");
  }

  ImGui::SeparatorText("Run Settings");
  ImGui::BeginDisabled(_mochiScene.IsSimulating());
  std::optional<real> const groundHeightBefore = spec.run.groundHeight;
  sAutoGroundHeightForRunUi = _autoGroundHeight;
  changed |= ImGui::SimpleReflectionStruct(spec.run, DrawRunFieldOverride);
  changed |= spec.run.groundHeight != groundHeightBefore;
  ImGui::EndDisabled();
  if (_mochiScene.IsSimulating()) {
    ImGui::TextDisabled("Stop the simulation to change run settings.");
  }

  // Each card list numbers its cards from 1, so each gets its own ID scope.
  ImGui::SeparatorText("Probes");
  ImGui::PushID("probes");
  ShowProbeCards(structural);
  ImGui::PopID();

  ImGui::SeparatorText("Stop Conditions");
  ImGui::PushID("stopConditions");
  ShowStopConditionCards(structural);
  ImGui::PopID();

  if (changed || structural) {
    OnSpecEdited(structural);
  }
  ImGui::End();
}

void TunerEditor::ShowOverridesWindow(bool* open) {
  if (!ImGui::Begin("Asset Tuner Prefab Overrides", open)) {
    ImGui::End();
    return;
  }
  auto& overrides = _tunerAsset->GetSpec().prefabOverrides;
  _overrideCards.Resize(overrides.size());
  if (_overrideErrors.size() != overrides.size()) {
    ValidateOverrides();
  }
  ImGui::TextWrapped(
      "Values set on the prefab before each Play, without changing the prefab file. The viewport "
      "shows the unmodified prefab.");
  ImGui::Separator();
  bool changed = false;
  bool structural = false;

  int removeIndex = -1;
  ReorderableBubbleList list;
  list.count = overrides.size();
  list.id = [&](size_t i) { return _overrideCards.cards[i].id; };
  list.bubbleColor = [&](size_t i, bool isPlaceholder) {
    bool const broken = i < _overrideErrors.size() && !_overrideErrors[i].empty();
    return CardColor(broken ? kErrorColor : kNeutralAccent, !isPlaceholder);
  };
  list.ghostLabel = [&](size_t i) { return std::string(std::string_view(overrides[i].path)); };
  list.move = [&](size_t from, size_t to) {
    if (from != to) {
      MoveItem(overrides, from, to);
      MoveItem(_overrideCards.cards, from, to);
      MoveItem(_overrideErrors, from, to);
      structural = true;
    }
  };
  list.drawContent = [&](size_t i) {
    tuner::PrefabOverride& entry = overrides[i];
    std::string const& error = _overrideErrors[i];
    std::string label = (error.empty() ? "" : ICON_FA_EXCLAMATION_TRIANGLE " ") +
        (entry.path.empty() ? std::string("(no path)")
                            : std::string(std::string_view(entry.path))) +
        " = " + std::string(std::string_view(entry.value));
    bool remove = false;
    bool const bodyOpen = DrawBubbleHeader(
        i,
        nullptr,
        nullptr,
        label,
        _overrideCards.cards[i].collapsed,
        "Remove this override.",
        remove);
    if (remove) {
      removeIndex = static_cast<int>(i);
    }
    if (bodyOpen) {
      changed |= ImGui::InputText("Path", &entry.path);
      Tooltip(
          "Dot-separated JSON path into the prefab, with [index] or [name=...] selectors.\n\n"
          "Examples:\n"
          "  actors.rigid[name=block].contact.coulombFrictionCoefficient\n"
          "  prefabs[name=Nested1].actors.rigid[name=Rigid1].isStatic");
      changed |= ImGui::InputText("Value", &entry.value);
      Tooltip(
          "JSON value, e.g. 0.8, true or [0, 0, 1]. Text that is not valid JSON is passed as a string.");
      if (!error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, kErrorColor);
        ImGui::TextWrapped("%s", error.c_str());
        ImGui::PopStyleColor();
      } else {
        ImGui::TextDisabled("Applies on the next Play.");
      }
    }
  };
  DrawReorderableBubbleList(_overrideCards.drag, list);

  if (removeIndex >= 0) {
    overrides.erase(overrides.begin() + removeIndex);
    _overrideCards.cards.erase(_overrideCards.cards.begin() + removeIndex);
    structural = true;
  }
  if (ImGui::Button(ICON_FA_PLUS " Add Override", ImVec2(-FLT_MIN, 0))) {
    overrides.emplace_back();
    _overrideCards.AddExpanded();
    structural = true;
  }
  Tooltip("Add a prefab override.");
  if (changed || structural) {
    ValidateOverrides();
    OnSpecEdited(structural);
  }
  ImGui::End();
}

} // namespace superdex::studio
