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

#include "simulation/mochi_async_scene.h"
#include "ui/imgui_widgets.h"

#include <imguios/fonts/icons_font_awesome5.h>

#include <mochi_core/mochi_platform.h>
#include <mochi_renderer/debug.h>
#include <mochi_renderer/type_conversions.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

using namespace mochi_renderer;

namespace superdex::studio {

// The intended DebugDraw here is mochi_renderer's. Disambiguate it from mochi::DebugDraw,
// which is now visible in `superdex` via superdex_physics.h's `using namespace mochi;`.
using mochi_renderer::DebugDraw;

// Marks PhysicsSceneSettings::gravity read-only, for assets that supply their own.
static bool IsGravityField(SReflect::FieldTypeInfo const& field) {
  return std::strcmp(field._name, "gravity") == 0;
}

//--------------------------------------------------------------------------------------------------
// DEBUG DRAW STATE
//--------------------------------------------------------------------------------------------------

void MochiDebugRenderData::Clear() {
  debugLines.positions.clear();
  debugLines.colors.clear();
  debugSpheres.positions.clear();
  debugSpheres.radii.clear();
  debugSpheres.colors.clear();
}

//--------------------------------------------------------------------------------------------------
// DEBUG DRAW
//--------------------------------------------------------------------------------------------------

std::vector<MochiDebugDrawFeature> const& GetMochiDebugDrawFeatures(mochi::Context* context) {
  static std::vector<MochiDebugDrawFeature> features;
  if (!features.empty() || context == nullptr) {
    return features;
  }
  // The feature list is a property of the mochi build, not of any one scene, but it can only be
  // read off a scene's DebugDraw -- so query a throwaway one.
  auto* scene = context->CreateScene("debug_draw_feature_query");
  if (!scene) {
    return features;
  }
  auto& debugDraw = scene->GetDebugDraw();
  int const numFeatures = debugDraw.GetNumFeatures();
  features.reserve(numFeatures);
  for (int i = 0; i < numFeatures; ++i) {
    features.push_back(
        {std::string{debugDraw.GetFeatureName(i)},
         std::string{debugDraw.GetFeatureDescription(i)},
         debugDraw.IsFeatureEnabled(i)});
  }
  context->DestroyScene(scene);
  return features;
}

bool IsDebugDrawFeatureEnabled(
    PhysicsDebugSettings const& settings,
    MochiDebugDrawFeature const& feature) {
  auto const it = settings.features.find(feature.name);
  return it != settings.features.end() ? it->second : feature.defaultEnabled;
}

DebugDrawSettingsEdit ShowDebugDrawSettings(
    mochi::Context* context,
    PhysicsDebugSettings& settings) {
  DebugDrawSettingsEdit edit;
  edit.masterToggled = ImGui::Checkbox("Enable Debug Draw", &settings.enabled);
  // Indent the feature rows to the checkbox's label so they read as its children rather than as a
  // differently-shaped control sitting next to it.
  float const featureIndent = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x;
  ImGui::Indent(featureIndent);
  // Features stay editable while debug draw is off -- you can set up what you want to see before
  // turning it on -- but are dimmed to show that nothing is being drawn yet.
  if (!settings.enabled) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  }
  auto const& features = GetMochiDebugDrawFeatures(context);
  for (size_t i = 0; i < features.size(); ++i) {
    auto const& feature = features[i];
    bool enabled = IsDebugDrawFeatureEnabled(settings, feature);
    if (ImGui::Selectable(feature.name.c_str(), &enabled)) {
      settings.features[feature.name] = enabled;
      edit.toggledFeature = static_cast<int>(i);
    }
    if (!feature.description.empty()) {
      ImGui::ItemTooltipWrapped(feature.description.c_str());
    }
  }
  if (!settings.enabled) {
    ImGui::PopStyleColor(); // ImGuiCol_Text
  }
  ImGui::Unindent(featureIndent);
  return edit;
}

//--------------------------------------------------------------------------------------------------
// ASYNC SCENE
//--------------------------------------------------------------------------------------------------

// Minimum positive timestep used as a safety floor. Mirrors the 0.001 s default from
// FingerController4Dof::GetSamplePeriodSec and prevents a 0-dt spin when stepSeconds is
// unset or SetFixedTimeStepSeconds(0) is called.
constexpr double kMinRealTimeStepSeconds = 0.001;

// Wall-clock schedule for real-time pacing. Owned by the pacing callback via shared_ptr so it
// can safely outlive scene teardown. A fresh pacer is allocated each time step params change
// (ApplySceneSettings), so this state is effectively immutable per-pacer and only touched on the
// simulation thread inside the pacing callback.
struct RealTimePacer {
  double stepSeconds{kMinRealTimeStepSeconds};
  std::optional<std::chrono::steady_clock::time_point> nextStepTime;
};

namespace {
// std::this_thread::sleep_* only wakes on an OS timer tick (~15 ms on default Windows; sub-ms on
// Linux/macOS), so it is far too coarse to hit a 1 kHz cadence on its own. To land on `target`
// accurately we coarse-sleep up to a margin above that granularity, then busy-wait the remainder.
// The margin exceeds the worst-case tick so a single overshooting sleep never passes `target`.
#if defined(_WIN32)
constexpr auto kPacingSpinMargin = std::chrono::milliseconds(16);
#else
constexpr auto kPacingSpinMargin = std::chrono::microseconds(500);
#endif

// Waits until `target` with sub-millisecond accuracy. The busy-wait tail keeps the end time precise
// at the cost of briefly occupying the core (bounded by kPacingSpinMargin, except for step
// durations shorter than the margin, which are spun in full).
void WaitUntilPrecise(std::chrono::steady_clock::time_point target) {
  using Clock = std::chrono::steady_clock;
  auto const remaining = target - Clock::now();
  if (remaining > kPacingSpinMargin) {
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(remaining - kPacingSpinMargin);
  }
  while (Clock::now() < target) {
    MOCHI_NOP_50();
  }
}

// Returns a timeStepCallback that paces fixed stepping to real time: it waits until the next
// scheduled wall-clock instant before returning the (constant) step duration. Because a callback
// takes priority over useFixedTimeStep in mochi's step loop, returning stepSeconds keeps dt fixed.
std::function<double()> MakeRealTimePacingCallback(std::shared_ptr<RealTimePacer> pacer) {
  return [pacer = std::move(pacer)]() -> double {
    using Clock = std::chrono::steady_clock;
    double const rawStepSeconds = pacer->stepSeconds;
    // Clamp to a small positive minimum to avoid a 0-dt spin if the value is transiently 0
    // (e.g. during reconfig) or if SetFixedTimeStepSeconds(0) was called.
    double const stepSeconds = std::max(rawStepSeconds, kMinRealTimeStepSeconds);
    auto const stepDuration =
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(stepSeconds));
    Clock::time_point const now = Clock::now();
    // Anchor on the first step, or re-anchor if we have fallen more than one full step behind so a
    // hitch does not trigger a burst of catch-up steps.
    if (!pacer->nextStepTime || now > *pacer->nextStepTime + stepDuration) {
      pacer->nextStepTime = now;
    }
    WaitUntilPrecise(*pacer->nextStepTime);
    *pacer->nextStepTime += stepDuration;
    return stepSeconds;
  };
}
} // namespace

MochiAsyncScene::~MochiAsyncScene() {
  DestroyMochiScene();
}

void MochiAsyncScene::Initialize(mochi::Context* context, std::string const& sceneName) {
  _context = context;
  _sceneName = sceneName;
}

void MochiAsyncScene::SetSettings(PhysicsSettings const& settings) {
  _settings = settings;
}

void MochiAsyncScene::SetGroundPlaneHeight(float height) {
  _groundPlaneHeight = height;
}

void MochiAsyncScene::UpdateStats() {
  if (_asyncScene == nullptr || _mochiSteps <= _previousMochiSteps) {
    return;
  }
  auto const now = std::chrono::steady_clock::now();
  double const elapsedSec = std::chrono::duration<double>(now - _lastStatsTime).count();
  if (elapsedSec > 0.0) {
    // Actual step rate = steps completed since the last sample over the wall-clock elapsed. This
    // reflects real-time throttling (the sleep between steps), unlike per-step compute duration.
    double const currentSPS = static_cast<double>(_mochiSteps - _previousMochiSteps) / elapsedSec;
    if (_mochiSPS <= 0.0f) {
      _mochiSPS = static_cast<float>(currentSPS);
    } else {
      constexpr double filterAlpha = 0.9;
      _mochiSPS = static_cast<float>(_mochiSPS * filterAlpha + (1.0 - filterAlpha) * currentSPS);
    }
  }
  _previousMochiSteps = _mochiSteps;
  _lastStatsTime = now;
}

bool MochiAsyncScene::IsSimulating() const {
  return _asyncScene != nullptr;
}

mochi::AsyncScene* MochiAsyncScene::GetAsyncScene() const {
  return _asyncScene;
}

void MochiAsyncScene::Pause(bool pause) {
  if (_asyncScene) {
    _asyncScene->Pause(pause);
  }
}

void MochiAsyncScene::RequestStepThenPause() {
  if (_asyncScene) {
    _asyncScene->RequestStepThenPause();
  }
}

bool MochiAsyncScene::IsPaused() const {
  if (_asyncScene) {
    return _asyncScene->IsPaused();
  }
  return true;
}

void MochiAsyncScene::CreateScene(std::string_view name, bool startPaused) {
  if (_context == nullptr) {
    MOCHI_LOG_ERROR(
        "Failed to create async scene because context was null! Did you forget to call Initialize?");
    return;
  }
  // Destroy the existing one if necessary
  DestroyMochiScene();
  // create context in paused or unpaused state
  mochi::ErrorLog error;
  _asyncScene = _context->CreateAsyncScene(name, error);
  if (!error.IsOK() || !_asyncScene) {
    MOCHI_LOG_ERROR("Failed to create async scene");
    return;
  }
  if (startPaused) {
    _asyncScene->Pause(true);
  }
  // Apply the scene settings before the actors are created: an asset that carries its own
  // gravity/solver (prefab::AddToScene) applies them from createPhysicsActors, and wins.
  ApplySceneSettings();

  _asyncScene->QueueCommand([this,
                             addGroundPlane = _settings.studio.groundPlane,
                             groundHeight = _groundPlaneHeight](mochi::Scene* physicsScene) {
    if (addGroundPlane) {
      mochi::ErrorLog planeError;
      mochi::RigidActorParams groundParams;
      groundParams.isStatic = true;
      groundParams.name = "StudioGroundPlane";
      groundParams.shape = _context->CreatePlaneShape({0.0f, 0.0f, 1.0f}, groundHeight, planeError);
      physicsScene->CreateRigidActor(groundParams, planeError);
    }
    if (createPhysicsActors) {
      createPhysicsActors(physicsScene);
    }
  });

  auto const& features = GetMochiDebugDrawFeatures(_context);
  std::vector<uint8_t> featureEnables(features.size());
  for (size_t i = 0; i < features.size(); ++i) {
    featureEnables[i] = IsDebugDrawFeatureEnabled(_settings.debug, features[i]) ? 1 : 0;
  }
  _asyncScene->QueueCommand([enabled = _settings.debug.enabled,
                             featureEnables = std::move(featureEnables)](mochi::Scene* scene) {
    auto& debugDraw = scene->GetDebugDraw();
    debugDraw.Enable(enabled);
    int const numFeatures = debugDraw.GetNumFeatures();
    for (int i = 0; i < numFeatures && i < static_cast<int>(featureEnables.size()); ++i) {
      debugDraw.EnableFeature(i, featureEnables[i] != 0);
    }
  });
  _asyncScene->WaitForQueuedCommands();

  if (registerPreStepCallback) {
    _preStepCb = registerPreStepCallback(_asyncScene);
  }

  if (registerPostStepCallback) {
    _postStepCb = registerPostStepCallback(_asyncScene);
  }

  _debugDrawCb = _asyncScene->RegisterPostStepCallback(
      "MochiBots::DebugDraw", [this](mochi::StepInfo const& info) {
        MochiDebugRenderData& data = _debugData.GetProducerData();
        data.Clear();
        auto const& drawData = info.scene->GetDebugDraw().GatherData();
        data.debugLines.positions.assign(
            drawData.lineVertices.positions.begin(), drawData.lineVertices.positions.end());
        data.debugLines.colors.assign(
            drawData.lineVertices.colors.begin(), drawData.lineVertices.colors.end());
        data.debugSpheres.positions.assign(
            drawData.spheres.positions.begin(), drawData.spheres.positions.end());
        data.debugSpheres.radii.assign(
            drawData.spheres.radii.begin(), drawData.spheres.radii.end());
        data.debugSpheres.colors.assign(
            drawData.spheres.colors.begin(), drawData.spheres.colors.end());
        _debugData.Produce();
      });

  // Reset step-rate stats so the first sample after (re)starting measures from a clean baseline.
  // Safe here: the previous scene (and its stat callback) was destroyed above, and the new stat
  // callback is registered just below, so nothing writes _mochiSteps concurrently.
  _mochiSteps = 0;
  _previousMochiSteps = 0;
  _mochiSPS = 0.0f;
  _lastStatsTime = std::chrono::steady_clock::now();

  _statCb = _asyncScene->RegisterPostStepCallback(
      "MochiBots::UpdateStats", [this](mochi::StepInfo const&) { _mochiSteps++; });
}

void MochiAsyncScene::DestroyMochiScene() {
  if (!_asyncScene) {
    return;
  }
  // cancel any registered callbacks
  if (_preStepCb.IsValid()) {
    _asyncScene->CancelCallback(_preStepCb);
    _preStepCb = {};
  }
  if (_postStepCb.IsValid()) {
    _asyncScene->CancelCallback(_postStepCb);
    _postStepCb = {};
  }
  // destroy actors
  if (destroyPhysicsActors) {
    _asyncScene->QueueCommand([this](mochi::Scene* scene) { destroyPhysicsActors(scene); });
    _asyncScene->WaitForQueuedCommands();
  }
  // destroy scene
  auto context = _asyncScene->GetContext();
  context->DestroyAsyncScene(_asyncScene);
  _asyncScene = nullptr;
  // notify user
  if (onStopPhysics) {
    onStopPhysics();
  }
  _debugData.Consume();
}

void MochiAsyncScene::EnableDebugDraw(bool enable) {
  _settings.debug.enabled = enable;
  if (IsSimulating()) {
    _asyncScene->QueueCommand(
        [enable](mochi::Scene* scene) { scene->GetDebugDraw().Enable(enable); });
  }
}

void MochiAsyncScene::ToggleDebugDraw() {
  EnableDebugDraw(!_settings.debug.enabled);
}

bool MochiAsyncScene::IsDebugDrawEnabled() const {
  return _settings.debug.enabled;
}

void MochiAsyncScene::ApplySceneSettings() {
  if (!IsSimulating()) {
    return;
  }
  mochi::ErrorLog stepParamsError;
  mochi::AsyncStepParams stepParams = _asyncScene->GetAsyncStepParams();
  stepParams.useFixedTimeStep = _settings.scene.useFixedTimeStep;
  stepParams.fixedTimeStepSeconds = _settings.scene.fixedTimeStepSeconds;
  if (_settings.scene.throttleToRealTime && stepParams.useFixedTimeStep) {
    // Allocate a fresh pacer each time so the previously-installed callback (which holds a
    // shared_ptr to the old pacer) cannot race with the UI thread's reset of nextStepTime.
    // The old callback keeps its own isolated state until it is replaced by SetAsyncStepParams.
    auto pacer = std::make_shared<RealTimePacer>();
    pacer->stepSeconds = stepParams.fixedTimeStepSeconds;
    _pacer = pacer;
    stepParams.timeStepCallback = MakeRealTimePacingCallback(pacer);
  } else {
    // GetAsyncStepParams returns the live params, so an earlier pacing callback must be cleared
    // explicitly or it would survive turning throttling off.
    stepParams.timeStepCallback = {};
    _pacer.reset();
  }
  _asyncScene->SetAsyncStepParams(stepParams, stepParamsError);
  if (!stepParamsError.IsOK()) {
    MOCHI_LOG_ERROR("Failed to set async step params on scene");
  }
  _asyncScene->QueueCommand(
      [scene = _settings.scene, solver = _settings.solver](mochi::Scene* physicsScene) {
        physicsScene->SetGravity(scene.gravity);
        mochi::ErrorLog solverError;
        physicsScene->SetSolverParams(solver, solverError);
      });
}

void MochiAsyncScene::DrawDebug(
    DebugDraw* debugDraw,
    mochi::CoordinateSpaceConverter const& converter) {
  if (IsSimulating() && _settings.debug.enabled) {
    if (!debugDraw) {
      return;
    }
    _debugData.Consume();
    MochiDebugRenderData const& data = _debugData.GetConsumerData();
    // Draw lines (positions come in pairs)
    for (size_t i = 0; i + 1 < data.debugLines.positions.size(); i += 2) {
      auto const& p0 = data.debugLines.positions[i];
      auto const& p1 = data.debugLines.positions[i + 1];
      auto const& c = data.debugLines.colors[i];
      auto p0f = ToFilament<float>(converter.TranslationToOutput(p0));
      auto p1f = ToFilament<float>(converter.TranslationToOutput(p1));
      debugDraw->DrawLine(p0f, p1f, {c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, c[3] / 255.0f});
    }
    // Draw spheres
    for (size_t i = 0; i < data.debugSpheres.positions.size(); ++i) {
      auto const& p = data.debugSpheres.positions[i];
      auto r = data.debugSpheres.radii[i];
      auto const& c = data.debugSpheres.colors[i];
      auto pf = ToFilament<float>(converter.TranslationToOutput(p));
      debugDraw->DrawSphere(
          pf, float(r), {c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, c[3] / 255.0f});
    }
  }
}

float MochiAsyncScene::GetStepsPerSecond() const {
  return IsSimulating() ? _mochiSPS : 0.0f;
}

void MochiAsyncScene::SetFixedTimeStepSeconds(double seconds) {
  _settings.scene.fixedTimeStepSeconds = seconds;
  ApplySceneSettings();
}

void MochiAsyncScene::ShowPhysicsSettingsWindow(
    char const* name,
    bool* open,
    AssetSceneOverrides assetOverrides) {
  if (!ImGui::Begin(name, open)) {
    ImGui::End();
    return;
  }
  // Scene and Solver are siblings here, in the app Settings window and in the prefab's own scene
  // section; keep the three in step.
  if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Indent();
    if (assetOverrides.gravity) {
      ImGui::TextWrapped(
          "Gravity comes from the asset, and is applied when the simulation starts.");
    }
    // Only gravity can come from the asset; the time step is always the editor's own.
    if (ImGui::SimpleReflectionStruct(
            _settings.scene, nullptr, assetOverrides.gravity ? IsGravityField : nullptr)) {
      ApplySceneSettings();
    }
    ImGui::Unindent();
  }
  if (ImGui::CollapsingHeader("Solver")) {
    ImGui::Indent();
    if (assetOverrides.solver) {
      ImGui::TextWrapped(
          "The solver comes from the asset, and is applied when the simulation starts.");
      ImGui::BeginDisabled();
      (void)ImGui::SimpleReflectionStruct(_settings.solver);
      ImGui::EndDisabled();
    } else if (ImGui::SimpleReflectionStruct(_settings.solver)) {
      ApplySceneSettings();
    }
    ImGui::Unindent();
  }
  if (ImGui::CollapsingHeader("Debug Draw", ImGuiTreeNodeFlags_DefaultOpen)) {
    DebugDrawSettingsEdit const edit = ShowDebugDrawSettings(_context, _settings.debug);
    if (edit.masterToggled) {
      EnableDebugDraw(_settings.debug.enabled);
    }
    if (edit.toggledFeature >= 0 && IsSimulating()) {
      int const index = edit.toggledFeature;
      bool const enabled =
          IsDebugDrawFeatureEnabled(_settings.debug, GetMochiDebugDrawFeatures(_context)[index]);
      _asyncScene->QueueCommand([index, enabled](mochi::Scene* scene) {
        scene->GetDebugDraw().EnableFeature(index, enabled);
      });
    }
  }
  if (ImGui::CollapsingHeader("Studio")) {
    ImGui::Indent();
    (void)ImGui::SimpleReflectionStruct(_settings.studio);
    ImGui::Unindent();
  }
  ImGui::End();
}

void MochiAsyncScene::ShowPlayToolbarOverViewport() {
  ImVec2 childPos = ImGui::GetWindowPos();
  ImVec2 childSize = ImGui::GetWindowSize();
  constexpr float kButtonPadding = 10.0f;
  constexpr int kNumButtons = 3;
  float buttonSize = ImGui::CalcTextSize(ICON_FA_PLAY).x + ImGui::GetStyle().FramePadding.x * 2;
  float spacing = ImGui::GetStyle().ItemSpacing.x;
  float totalWidth = buttonSize * kNumButtons + spacing * (kNumButtons - 1);
  float startX = childPos.x + (childSize.x - totalWidth) * 0.5f;
  float startY = childPos.y + kButtonPadding;

  ImGui::SetCursorScreenPos({startX, startY});
  bool simulating = IsSimulating();
  bool playing = simulating && !IsPaused();
  if (!simulating) {
    if (ImGui::Button(ICON_FA_PLAY "###play")) {
      CreateScene(_sceneName.c_str(), false);
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Play (Space)");
    }
  } else if (!playing) {
    if (ImGui::Button(ICON_FA_PLAY "###play")) {
      Pause(false);
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Resume (Space)");
    }
  } else {
    if (ImGui::Button(ICON_FA_PAUSE "###play")) {
      Pause(true);
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Pause (Space)");
    }
  }
  ImGui::BeginDisabled(playing);
  ImGui::SameLine();
  if (ImGui::Button(ICON_FA_STEP_FORWARD "###step")) {
    if (!simulating) {
      CreateScene(_sceneName.c_str(), true);
    }
    if (IsSimulating()) {
      RequestStepThenPause();
    }
  }
  // Show the tooltip even while disabled so users can discover the shortcut before starting.
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip("Step (Right Arrow)");
  }
  ImGui::EndDisabled(); // playing
  ImGui::BeginDisabled(!simulating);
  ImGui::SameLine();
  if (ImGui::Button(ICON_FA_STOP "###stop")) {
    DestroyMochiScene();
  }
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip("Stop (Esc)");
  }
  ImGui::EndDisabled(); // !simulating
}

void MochiAsyncScene::HandleHotkeys() {
  bool const simulating = IsSimulating();
  bool const playing = simulating && !IsPaused();
  // Space toggles play/pause, starting the simulation if it is currently stopped.
  if (ImGui::IsKeyChordPressed(ImGuiKey_Space)) {
    if (!simulating) {
      CreateScene(_sceneName, false);
    } else {
      Pause(playing);
    }
  }
  // Right Arrow single-steps; matches the toolbar, which disables stepping while playing.
  if (!playing && ImGui::IsKeyChordPressed(ImGuiKey_RightArrow)) {
    if (!simulating) {
      CreateScene(_sceneName, true);
    }
    if (IsSimulating()) {
      RequestStepThenPause();
    }
  }
  // Escape stops; matches the toolbar, which disables stop while not simulating.
  if (simulating && ImGui::IsKeyChordPressed(ImGuiKey_Escape)) {
    DestroyMochiScene();
  }
}

} // namespace superdex::studio
