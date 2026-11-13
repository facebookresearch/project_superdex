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

#include "assets/ik_config.h"
#include "core/async_task.h"
#include "editors/asset_editor.h"
#include "rendering/scene_stage.h"
#include "simulation/ik_preview_session.h"
#include "simulation/ik_recording_player.h"

#include <filesystem>

#include <chrono>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace superdex::studio {

class IKAsset;
enum class PlayToolbarAction;

class IKEditor : public AssetEditor {
 public:
  IKEditor(SuperDexStudio* studio, IKAsset* asset);
  ~IKEditor() override;
  void Initialize() override;
  void OnHandleInputs() override;
  void OnRender(Renderer const* renderer) override;
  // Called before app Undo/Redo dispatch so a completed drag is part of the history.
  void CommitPendingEdits();
  void Shutdown() override;
  void OnActivate() override;
  void OnDeactivate() override;
  void Refresh() override;
  void ShowTabContents() override;
  static std::vector<WindowDeclaration> GetDefaultWindows();
  std::vector<WindowDeclaration> GetAuxiliaryWindows() const override;
  void ShowAuxiliaryWindows() override;
  bool CanUndoRedo() const override;
  void ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& settings) override;
  Viewport* GetViewport() override;
  void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const override;

 private:
  using Clock = std::chrono::steady_clock;
  bool HasGoalSelection() const;
  void SelectGoal(int index);
  void SelectInputBot();
  void SyncViewportSelection();
  void OnSceneSelectionChanged(std::vector<mochi_renderer::SceneObject*> const& objects);
  void RestoreUndoSnapshot(std::string const& json, int selectionIndex);
  void MarkEdited(bool discrete = false, bool structureChanged = false);
  void ResetUndoBaseline();
  void PushUndoSnapshot();
  void CaptureSavedSnapshot();
  void PrepareOutputBot();
  void RestageBots();
  void PrepareInputBot();
  void UpdateInputBotPose();
  bool CanSnapshotInputBotPose(BotIKConfig& candidate, std::string& diagnostic) const;
  void SnapshotInputBotPose();
  void CancelRecording();
  void SyncRecording(bool force = false, bool checkFile = false);
  void SampleRecording();
  void ShowRecordingControls();
  void SelectRecording(mochi::Path const& path);
  void RestoreRecordingSelection();
  void ShowRecordingDropTarget();
  std::optional<mochi::TransformRT> GoalConstraintAnchor(BotIKGoalConfig const&) const;
  std::optional<mochi::TransformRT> GoalWorldConstraintFrame(BotIKGoalConfig const&) const;
  void UpdateFrameSize();
  void ClearMarkers();
  void RebuildMarkers();
  void UpdateMarkers();
  void DrawGoals();
  void ApplyPreviewPose();
  void StopPreview();
  void SolveOnce();
  bool CanPreview(std::string& diagnostic) const;
  int FindOutputBotLink(std::string_view name) const;
  bool OutputBotLinkUsed(std::string_view name, int except = -1) const;
  bool NameUsed(std::string_view name, int except = -1) const;
  void AddGoal(std::string const& link);
  void RemoveGoal();
  void ResetGoal();
  void ShowGoalsWindow(bool* open);
  void ShowJointCouplings();
  bool CommitJointCouplings(
      mochi::DynamicArray<robotics::BotIKJointCouplingConfig> couplings,
      bool discrete,
      std::optional<size_t> editedRow = std::nullopt);
  void ShowSolverWindow(bool* open);
  void ShowSolvePlotsWindow(bool* open);
  // Appends the preview's latest solve to the plot history once; ticks without a solve add nothing.
  void RecordSolveSample();
  void ClearSolvePlots();
  void HandlePlayToolbarAction(PlayToolbarAction action);

  IKAsset* _ikAsset = nullptr;
  std::unique_ptr<Viewport> _viewport;
  SceneStage _outputBotStage;
  SceneStage _inputBotStage;
  StageType _stageType = StageType::RenderModelFallbackToMochiModel;
  IKPreviewSession _preview;
  // Live solve metrics, one entry per published solve since the plots were last cleared. X is the
  // solve number; non-converged solves are also kept separately for highlighting.
  struct SolvePlotHistory {
    std::vector<double> solves;
    std::vector<double> milliseconds;
    std::vector<double> iterations;
    std::vector<double> unconvergedSolves;
    std::vector<double> unconvergedIterations;
    mochi::ConvergenceStatus lastConvergence = mochi::ConvergenceStatus::None;
    double nextSolve = 1;
  };
  SolvePlotHistory _solvePlots;
  uint64_t _plottedSolveCount = 0;
  // Solve-axis range, kept across frames (ImPlot axis link) so panning and zooming persist while
  // auto-fit is off.
  double _solvePlotMin = 0;
  double _solvePlotMax = 1;
  bool _autoFitSolvePlots = true;
  mochi::DynamicArray<mochi::TransformRT> _authoredOutputBaseFrameFromLinkFrames;
  mochi::DynamicArray<mochi::TransformRT> _worldFrameFromInputLinkFrames;
  // Owned only after a successful stage application; invalidated on rebind, restage, or bad FK.
  std::optional<IKInputPoseSnapshot> _displayedInputPose;
  mochi::DynamicArray<std::string> _inputBotStagedLinkNames;
  bool _selectInputBotRoot = false;
  std::string _inputBotDiagnostic;
  std::string _inputBotSnapshot;
  mochi::DynamicString _observedInputBotReference;
  std::unordered_map<std::string, int> _outputBotLinkIndices;
  mochi::DynamicArray<std::string> _outputBotStagedLinkNames;
  // Markers are owned by the render scene, independently of SceneStage's robot objects.
  mochi::DynamicArray<mochi_renderer::SceneObject*> _markers;
  std::unordered_map<mochi_renderer::SceneObject const*, int> _markerGoals;
  std::shared_ptr<mochi_renderer::MaterialInstance> _markerMaterial;
  std::shared_ptr<mochi_renderer::MaterialInstance> _selectedMarkerMaterial;
  std::shared_ptr<mochi_renderer::MaterialInstance> _disabledMarkerMaterial;
  int _selectedGoal = -1;
  mochi::real _outputLinkAxisLength = mochi::real(0.08);
  bool _prepared = false;
  bool _running = false;
  bool _active = false;
  bool _showGoals = true;
  Clock::time_point _nextSolve{};
  std::string _outputBotDiagnostic;
  std::string _editDiagnostic;
  std::string _previewDiagnostic;
  // Pending Add selections are UI-only; no incomplete coupling is written to the document.
  mochi::DynamicString _pendingCouplingA;
  mochi::DynamicString _pendingCouplingB;
  std::string _nameBuffer;
  int _nameBufferGoal = -1;
  std::string _lastUndoSnapshot;
  bool _pendingEdit = false;
  // A dirty reference refresh invalidates the saved JSON, even if UndoStack resets to index zero.
  std::optional<std::string> _savedSnapshot;
  mochi::DynamicString _observedOutputBotReference;
  std::string _outputBotSnapshot;
  mochi::Path _observedDocumentPath;
  struct RecordingLoad {
    std::atomic_bool done{false};
    uint64_t generation = 0;
    std::shared_ptr<IKRecordingClip const> clip;
    std::string diagnostic;
  };
  std::shared_ptr<RecordingLoad> _recordingLoad;
  std::shared_ptr<IKRecordingClip const> _recordingClip;
  IKRecordingClock _recordingClock;
  std::optional<IKInputBotPose> _recordingPose;
  std::string _recordingDocumentKey;
  std::string _observedRememberedRecording;
  std::string _selectedRecording;
  std::string _recordingReference;
  std::string _recordingPath;
  std::string _selectionDiagnostic;
  std::string _recordingDiagnostic;
  std::filesystem::file_time_type _recordingStamp{};
  bool _recordingHasStamp = false;
  bool _recordingOwnsPose = false;
  bool _resumeRecording = false;
  uint64_t _recordingGeneration = 0;
  // Destroy/join the worker before its result sinks; jobs never capture this editor.
  std::unique_ptr<BackgroundTaskQueue> _recordingTasks;
};

} // namespace superdex::studio
