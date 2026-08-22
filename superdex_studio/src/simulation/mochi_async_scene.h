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

#include <superdex_robotics/superdex_robotics.h>

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/transform_srt.h>

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/utils/mochi_prefab.h>
#include <mochi_renderer/buffer.h>

#include <chrono>
#include <memory>

namespace mochi_renderer {
class SceneObject;
class DebugDraw;
} // namespace mochi_renderer

namespace superdex::studio {

class SuperDexStudio;

// Holds the wall-clock schedule used to pace fixed-step simulation to real time. Defined in
// the .cpp; owned via shared_ptr so the stepping callback can safely outlive scene teardown.
struct RealTimePacer;

//--------------------------------------------------------------------------------------------------
// DEBUG DRAW
//--------------------------------------------------------------------------------------------------

// One of mochi's debug draw features, as reported by the running mochi build.
struct MochiDebugDrawFeature {
  std::string name;
  std::string description;
  // What mochi enables by default, used for features the settings have no entry for.
  bool defaultEnabled = false;
};

// Mochi's debug draw features, queried once from a throwaway scene and cached. Features are added
// and removed as mochi evolves, so this live list -- never the saved settings -- decides what
// exists; saved entries for features that are gone are kept but ignored.
std::vector<MochiDebugDrawFeature> const& GetMochiDebugDrawFeatures(mochi::Context* context);

// Whether @p feature is enabled by @p settings, falling back to mochi's default for a feature the
// settings say nothing about (i.e. one added since they were saved).
bool IsDebugDrawFeatureEnabled(
    PhysicsDebugSettings const& settings,
    MochiDebugDrawFeature const& feature);

// What the user changed in a ShowDebugDrawSettings() call.
struct DebugDrawSettingsEdit {
  bool masterToggled = false;
  int toggledFeature = -1; //< index into GetMochiDebugDrawFeatures, or -1 if none

  bool AnyChange() const {
    return masterToggled || toggledFeature >= 0;
  }
};

// Draws the debug draw enable checkbox and the feature list, editing @p settings. Shared by the app
// Settings window (which edits the defaults) and each editor's Physics Settings window (which edits
// a session's copy and pushes the result into its live scene).
DebugDrawSettingsEdit ShowDebugDrawSettings(
    mochi::Context* context,
    PhysicsDebugSettings& settings);

struct MochiDebugDrawLineVerticesBuffer {
  std::vector<mochi::Real3> positions;
  std::vector<mochi::Color> colors;
};

struct MochiDebugDrawSpheresBuffer {
  std::vector<mochi::Real3> positions;
  std::vector<mochi::real> radii;
  std::vector<mochi::Color> colors;
};

struct MochiDebugRenderData {
  MochiDebugDrawLineVerticesBuffer debugLines;
  MochiDebugDrawSpheresBuffer debugSpheres;
  void Clear();
};

//--------------------------------------------------------------------------------------------------
// ASYNC SCENE
//--------------------------------------------------------------------------------------------------

// Which of the session's scene settings the loaded asset supplies itself. Those are shown but not
// editable in the Physics Settings window: the asset's values are applied when the simulation
// starts and would overwrite an edit.
struct AssetSceneOverrides {
  bool gravity = false;
  bool solver = false;
};

class MochiAsyncScene {
 public:
  std::function<void(mochi::Scene* scene)> createPhysicsActors;
  std::function<void(mochi::Scene* scene)> destroyPhysicsActors;
  std::function<mochi::CallbackHandle(mochi::AsyncScene* asyncScene)> registerPreStepCallback;
  std::function<mochi::CallbackHandle(mochi::AsyncScene* asyncScene)> registerPostStepCallback;
  std::function<void()> onStopPhysics;

  ~MochiAsyncScene();

  // Scene
  void Initialize(mochi::Context* context, std::string const& sceneName);
  void SetSettings(PhysicsSettings const& settings);
  void SetGroundPlaneHeight(float height);
  bool IsSimulating() const;
  mochi::AsyncScene* GetAsyncScene() const;
  void Pause(bool pause);
  void RequestStepThenPause();
  bool IsPaused() const;
  void CreateScene(std::string_view name, bool startPaused);
  void DestroyMochiScene();

  // Sets the fixed-step duration (seconds) and applies it live if a scene is simulating.
  void SetFixedTimeStepSeconds(double seconds);

  // Debug
  void EnableDebugDraw(bool enable);
  void ToggleDebugDraw();
  bool IsDebugDrawEnabled() const;
  void DrawDebug(
      mochi_renderer::DebugDraw* debugDraw,
      mochi::CoordinateSpaceConverter const& converter);

  // Stats
  void UpdateStats();
  float GetStepsPerSecond() const;

  // ImGui
  void
  ShowPhysicsSettingsWindow(char const* name, bool* open, AssetSceneOverrides assetOverrides = {});
  void ShowPlayToolbarOverViewport();

  // Keyboard shortcuts mirroring the play toolbar
  void HandleHotkeys();

 private:
  // Pushes the scene settings (step params and real-time pacing, gravity, solver) into the
  // scene. No-op when stopped.
  void ApplySceneSettings();

  mochi::Context* _context = nullptr;
  std::string _sceneName;
  mochi::AsyncScene* _asyncScene = nullptr;
  mochi::CallbackHandle _preStepCb = {};
  mochi::CallbackHandle _postStepCb = {};
  float _mochiSPS = 0.0f;
  int _mochiSteps = 0;
  int _previousMochiSteps = 0;
  // Wall-clock time of the last stats sample, used to derive the actual (throttled) step rate.
  std::chrono::steady_clock::time_point _lastStatsTime = {};
  mochi::CallbackHandle _statCb = {};
  mochi_renderer::ProducerConsumerBuffer<MochiDebugRenderData> _debugData;
  mochi::CallbackHandle _debugDrawCb = {};
  PhysicsSettings _settings;
  float _groundPlaneHeight = 0.0f;
  std::shared_ptr<RealTimePacer> _pacer;
};

} // namespace superdex::studio
