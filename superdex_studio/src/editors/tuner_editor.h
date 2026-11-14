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

#include "editors/asset_editor.h"
#include "rendering/scene_stage.h"
#include "rendering/stats.h"
#include "simulation/mochi_async_scene.h"
#include "simulation/physics_drag_controller.h"
#include "simulation/prefab_sim_sync.h"
#include "ui/reorderable_list.h"

#include <superdex_asset_tuner/run_writer.h>
#include <superdex_asset_tuner/scene_targets.h>
#include <superdex_asset_tuner/sim_recording.h>
#include <superdex_asset_tuner/tuner_session.h>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace superdex::studio {

class TunerAsset;

// Physics test tuner: simulates a tuner's prefab with its loads applied, plots its probes live and
// records runs in the same format as the superdex_asset_tuner CLI.
class TunerEditor : public AssetEditor {
 public:
  TunerEditor(SuperDexStudio* studio, TunerAsset* asset);

  // AssetEditor
  void Initialize() override;
  void OnHandleInputs() override;
  void OnRender(Renderer const* renderer) override;
  void Shutdown() override;
  void OnActivate() override;
  void OnDeactivate() override;
  void Refresh() override;
  void ShowTabContents() override;
  static std::vector<WindowDeclaration> GetDefaultWindows();
  std::vector<WindowDeclaration> GetAuxiliaryWindows() const override;
  void ShowAuxiliaryWindows() override;
  bool CanUndoRedo() const override;
  void ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) override;
  void OnAppSettingsChanged(AppSettings const& settings) override;
  Viewport* GetViewport() override {
    return _viewport.get();
  }
  void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const override;

 private:
  // Plot history of one channel.
  struct ChannelHistory {
    std::string name;
    std::vector<float> values;
  };
  // A previous run, kept to compare with the current one in the plots.
  struct RunTrace {
    // Runs are numbered from 1 in each editor session.
    uint64_t id = 0;
    // What changed in the spec from the run before, one "path: before -> after" line each.
    std::string changes;
    // Why it ended early, and whether its loads were edited while it ran.
    std::string status;
    std::vector<float> times;
    std::vector<ChannelHistory> channels;
    bool visible = true;
    // Kept beyond AssetTunerSettings::runHistory.
    bool pinned = false;
  };
  // Everything in the tuner's scene that loads and probes can target, with the names Mochi gives
  // the actors: "<nested>/<actor>" and "<articulated actor>/<link or joint>". Built from
  // _describedTargets.
  struct PrefabTargets {
    std::vector<std::string> rigidBodies; // dynamic rigid actors
    std::vector<std::string> links;
    struct Joint {
      std::string name;
      // The link it drives (joint i drives link i).
      std::string link;
      // Its DoF count, as the run counts them (0 for Hard joints). Joint Effort takes 1-3.
      int numDofs = 0;
    };
    std::vector<Joint> joints;
  };
  // Editor-only state of one card list (loads, probes or overrides), kept parallel to the spec
  // array: stable ids for ImGui and drag-reorder, and each card's collapsed state.
  struct CardListState {
    struct Card {
      uint64_t id = 0;
      bool collapsed = true;
    };
    std::vector<Card> cards;
    DragReorderState drag;
    uint64_t nextId = 1;

    // Matches the card count to the spec array (after load, undo or an external edit).
    void Resize(size_t count);
    // Appends an expanded card for an item the user just added.
    void AddExpanded();
  };

  // Staging and simulation
  void Restage();
  void ApplyRunSettings();
  // The run's ground height: its override, else the automatic (lowest point of the prefab) one.
  [[nodiscard]] float EffectiveGroundHeight() const;
  void CreatePhysicsActors(mochi::Scene* scene);
  // Adds the prefab and binds the session; false with _bindError set on failure.
  bool BindPhysicsActors(mochi::Scene* scene);
  void DestroyPhysicsActors(mochi::Scene* scene);
  // Puts back the shared context's threading mode if this run overrode it.
  void RestoreThreadingMode();
  void OnStopPhysics();
  void DrainSamples();

  // Edits
  void OnSpecEdited(bool structural);
  [[nodiscard]] bool LiveLoadsCompatible() const;
  std::string TakeUndoSnapshot() const;
  void RestoreUndoSnapshot(std::string const& json);
  // Checks each prefab override against the loaded prefab; fills _overrideErrors.
  void ValidateOverrides();
  // Rebuilds _describedTargets and _targets from a temporary Mochi scene of the tuner (prefab plus
  // overrides). Runs once per frame, and only after the prefab or overrides change.
  void RefreshTargets();
  [[nodiscard]] std::string MakeUniqueLoadName(std::string_view base) const;

  // Run history
  // Moves the current run's traces into _runHistory and starts the next run's, on the UI thread
  // when a run starts.
  void BeginRunTrace();
  // Drops the oldest unpinned runs beyond AssetTunerSettings::runHistory.
  void TrimRunHistory();
  // Run list of the plots window: which runs are drawn and pinned, and what each changed.
  void ShowRunHistory();

  // Recording
  bool OpenRunWriter();
  void StartRecording();
  void StopRecording();
  // Scene thread (or UI thread while it is blocked): sim.rrd of the run OpenRunWriter opened.
  void StartSimRecording(mochi::Scene& scene);
  void RecordSimSample(superdex::tuner::Sample const& sample, bool firstStop);
  void StopSimSampling();
  [[nodiscard]] bool IsRecording() const {
    return _writer.IsOpen();
  }

  // Viewport
  void DrawLoadArrows();
  // Wireframe boxes of box stop conditions and planes of position thresholds (red once triggered).
  void DrawStopBoxes();
  // Banner shown after a stop condition paused the run.
  void ShowStopBanner();
  void OnScenePressed(mochi_renderer::SceneObject* object, filament::math::float3 worldHit);
  // Index of the load whose card is selected (the transform gizmo edits it), or -1.
  [[nodiscard]] int SelectedLoadIndex() const;
  void SelectLoad(int index);
  // Index of the selected waypoint of the selected pose path (the gizmo edits it), or -1.
  [[nodiscard]] int SelectedWaypointIndex() const;
  // The selected key if load `index` is the selected load, else -1. Keyed directions and points are
  // shown and edited at this key (the first when none is selected; see ClampKey).
  [[nodiscard]] int KeyShownFor(size_t index) const;
  // Gizmo frame of the selected load in editor space: at its application point, X along its
  // direction, or a pose path's selected waypoint. Empty when the selected load has no gizmo (joint
  // efforts, unresolved targets, no waypoint selected).
  [[nodiscard]] std::optional<mochi::TransformRT> SelectedLoadGizmoFrame() const;
  void OnLoadGizmoDelta(mochi::TransformRT const& delta);
  // Index of the stop condition whose box the gizmo edits, or -1 (box conditions only).
  [[nodiscard]] int SelectedStopBoxIndex() const;
  // The staged body a load or probe target refers to, or null.
  StagedActor const* FindStagedBody(std::string_view target) const;
  // Target names for loads of `type` from the prefab's top-level actors.
  std::vector<std::string> GetTargetCandidates(superdex::tuner::LoadType type) const;
  // The most recently selected viewport body that can take a load of `type`: the body itself, or
  // for Joint Effort the joint driving it. Nullopt when nothing selected fits.
  std::optional<std::string> SelectedLoadTarget(superdex::tuner::LoadType type) const;

  // Windows
  void ShowLoadsWindow(bool* open);
  // Draws one load card's body. Returns true on any edit; sets `structural` for edits that need a
  // new run (name, type, target).
  bool ShowLoadBody(size_t index, bool& structural);
  // Pose path settings of a load card: drive gains and the waypoint editor.
  void ShowPosePathEditor(size_t index, bool& changed);
  // Strip of the path's waypoints over run time, with each span's drive mode shaded. Markers select
  // a waypoint on click and retime it on drag. `compact` draws a short read-only strip instead.
  void ShowWaypointTimeline(size_t index, bool compact, bool& changed);
  // Profile settings of a load card. Keyframes get a key table and draggable markers on the
  // preview.
  void ShowProfileEditor(size_t index, bool& changed);
  // Load `index`'s profile over the run, with the sim time marked and keyframes as markers that
  // select a key on click and move it on drag. `compact` draws a short read-only strip.
  void ShowProfilePreview(size_t index, bool compact, bool& changed);
  void ShowPlotsWindow(bool* open);
  // Channel list: checkboxes plus drag sources for the plot areas.
  void ShowChannelList(bool& structural);
  // One plot area; `outRemove` is set when its remove button is pressed.
  void ShowPlotArea(size_t index, bool& structural, bool& outRemove);
  // Empty space below the plots; dropping channels there adds a new plot with them.
  void ShowNewPlotDropZone(bool& structural);
  // The saved plot layout, creating the default (every load channel in one plot) on first edit.
  mochi::DynamicArray<superdex::tuner::PlotSpec>& EditablePlots();
  [[nodiscard]] bool IsChannelPlotted(std::string_view channel) const;
  void ShowRunWindow(bool* open);
  void ShowProbeCards(bool& structural);
  void ShowStopConditionCards(bool& structural);
  void ShowOverridesWindow(bool* open);
  void ShowPhysicsSettingsWindow(bool* open);

 private:
  TunerAsset* _tunerAsset = nullptr;
  std::unique_ptr<Viewport> _viewport;
  SceneStage _stage;
  // The stage's render and collision stats, taken after each staging at the rest pose.
  StageStats _stageStats;
  MochiAsyncScene _mochiScene;
  PrefabSimSync _simSync;
  std::unique_ptr<PhysicsDragController> _dragController;

  // Sim-thread state. Touched from the UI thread only while the sim thread is blocked (scene
  // creation/destruction) or through AsyncScene::QueueCommand.
  superdex::tuner::TunerSession _session;
  mochi::CallbackHandle _loadsCallback;
  mochi::CallbackHandle _probesCallback;
  std::string _bindError;

  // Samples produced on the sim thread, drained each frame on the UI thread.
  std::mutex _samplesMutex;
  // Set on the sim thread when a stop condition first pauses the run; reset for each run.
  std::atomic<bool> _stopPausedOnSim = false;
  // The shared context's threading mode before this run set it; empty while the run is not
  // overriding it.
  std::optional<bool> _savedSingleThreaded;
  std::vector<superdex::tuner::Sample> _pendingSamples;

  // UI-thread view of the running session.
  std::vector<superdex::tuner::LoadSpec> _boundLoads;
  std::vector<float> _sampleTimes;
  std::vector<ChannelHistory> _channels;
  // Shared time-axis range of every plot area (ImPlot axis links), so they pan and zoom together.
  double _plotTimeMin = 0.0;
  double _plotTimeMax = 1.0;
  superdex::tuner::Sample _lastSample;
  bool _hasSample = false;

  // Run history, newest first, and the current run's id, spec, changes from the run before and
  // status.
  std::deque<RunTrace> _runHistory;
  uint64_t _runId = 0;
  superdex::tuner::TunerSpec _runSpec;
  std::string _runChanges;
  std::string _runStatus;
  bool _runEditedLive = false;
  // The loads last sent to the running session, as JSON, to tell load edits from other edits.
  std::string _liveLoadsJson;

  // Recording
  bool _recordNextRun = false;
  bool _stopAtDuration = true;
  std::string _outputDir;
  superdex::tuner::RunWriter _writer;
  superdex::tuner::RunSummary _runSummary;
  // The spec the recorded run resolves to (the ground height it uses).
  superdex::tuner::TunerSpec _recordedSpec;
  std::string _lastRunDir;
  // Scene-thread state of sim.rrd: the recording, whether it still samples, and the sim time it
  // ends at. The UI thread touches them only while the scene thread is blocked or idle.
  superdex::tuner::SimRecording _simRecording;
  bool _simRecordingOnSim = false;
  double _simRecordEnd = 0.0;

  // UI state
  PrefabTargets _targets;
  StageType _stageType = StageType::RenderModelFallbackToMochiModel;
  // Whether the session's debug-draw settings were seeded from the app defaults; after that they
  // belong to this editor and survive ApplyRunSettings.
  bool _debugSettingsInitialized = false;
  // Ground height the viewport found for the staged prefab (its lowest point).
  float _autoGroundHeight = 0.0f;
  CardListState _loadCards;
  CardListState _probeCards;
  CardListState _overrideCards;
  // One entry per prefab override; empty when the override applies cleanly.
  std::vector<std::string> _overrideErrors;
  // The tuner's bodies as a run builds them: names, masses and centers of mass. Empty if the prefab
  // failed to load into Mochi.
  std::optional<superdex::tuner::SceneTargets> _describedTargets;
  bool _describedTargetsStale = true;
  // Load whose card is under the mouse (its arrow is highlighted), or -1.
  int _hoveredLoad = -1;
  // Card id of the selected load (0 = none); an id so it survives reordering and removal.
  uint64_t _selectedLoadCard = 0;
  // Waypoint of the selected pose path the gizmo edits (-1 = none).
  int _selectedWaypoint = -1;
  // Waypoint whose timeline marker is being dragged (-1 = none); sorted by time on release.
  int _timelineDragWaypoint = -1;
  // Keyframe of the selected load's profile (-1 = none), and the one whose preview marker is being
  // dragged (-1 = none; sorted by time on release).
  int _selectedKey = -1;
  int _keyDragIndex = -1;
  // Stable ids of the plot areas, for drag-to-reorder.
  CardListState _plotCards;
  CardListState _stopCards;
  // Card id of the stop condition whose box the transform gizmo moves (0 = none).
  uint64_t _selectedStopCard = 0;
  // The stop condition that paused this run: spec index, message and sim time. Further triggers in
  // the same run are ignored so it can be resumed.
  struct StopEvent {
    int condition = -1;
    std::string message;
    double time = 0.0;
  };
  std::optional<StopEvent> _stopEvent;
  int _placingLoad = -1;
  bool _autoFitPlots = true;
};

} // namespace superdex::studio
