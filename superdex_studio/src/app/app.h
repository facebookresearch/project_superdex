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

#include "assets/asset_manager.h"
#include "core/common.h"
#include "core/settings.h"
#include "editors/asset_editor.h"
#include "editors/bot_editor.h"
#include "io/bot_loader.h"
#include "io/importer.h"
#include "rendering/renderer.h"
#include "ui/asset_browser.h"
#include "ui/log_console.h"

#include <mochi_renderer/mochi_renderer.h>
#include <mochi_renderer/render_space.h>
#include <mochi_renderer/resource.h>

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/error.h>

#include <superdex_robotics/core/context.h>

#include <filament/Engine.h>
#include <imguios/imguios.h>

#include <chrono>
#include <filesystem>

namespace mochi_renderer {
class IBL;
} // namespace mochi_renderer

namespace superdex::studio {

class Viewport;

class SuperDexStudio : public ImGuios::Application {
 public:
  //------------------------------------------------------------------------------------------------
  // ImGuios::Application
  //------------------------------------------------------------------------------------------------
  SuperDexStudio();
  void OnInitialize() override;
  void OnUpdate() override;
  void OnShutdown() override;

  //------------------------------------------------------------------------------------------------
  // Accessors
  //------------------------------------------------------------------------------------------------
  filament::Engine* GetEngine() const;
  mochi::Context* GetMochiContext() const;
  superdex::robotics::RoboticsContext* GetRoboticsContext() const;
  AssetManager& GetAssetManager() const;
  mochi_renderer::ResourceManager& GetResourceManager() const;
  AssetBrowser& GetAssetBrowser() const;
  AppSettings& GetAppSettings();
  mochi::CoordinateSpaceConverter const& GetEditorToRendererSpaceConverter() const;
  mochi::CoordinateSpaceConverter const& GetRendererToEditorSpaceConverter() const;
  SuperDexStudioBotLoader const& GetBotLoader() const;
  AssetEditor* GetActiveAssetEditor() const;
  bool& GetWindowVisible(std::string const& name);
  static std::string const& GetDefaultIblPath();
  mochi_renderer::IBL* GetDefaultIbl() const;
  mochi_renderer::SceneViewSettings const& GetViewSettings() const;

  //------------------------------------------------------------------------------------------------
  // Asset Editors
  //------------------------------------------------------------------------------------------------
  void OpenAssetEditor(Asset* asset);
  void ActivateAssetEditor(int index);
  void CloseAssetEditor(int index);
  int FindAssetEditorIndex(Asset const* asset) const;
  void SelectAssetEditor(int index);
  void RefreshEditors(std::vector<IAssetReferencer*> const& referencers);
  void SaveActiveAssetEditor();
  void SaveAllAssetEditors();

  //------------------------------------------------------------------------------------------------
  // Screenshots
  //------------------------------------------------------------------------------------------------

  void
  SaveAssetThumbnail(Asset& asset, int sizePx, mochi::Path const& outFile, mochi::Error& error);
  void SaveViewportScreenshot(Viewport& viewport, mochi::Path const& outFile, mochi::Error& error);

  //------------------------------------------------------------------------------------------------
  // File
  //------------------------------------------------------------------------------------------------
  static std::filesystem::path GetExecutableDir();
  static mochi::Path GetFileDialogPath(
      char const* title,
      char const* const* filters,
      int numFilters,
      char const* filterDesc,
      bool isSaveDialog,
      mochi::Path const& defaultPathAndFile = {});
  static mochi::Path GetFolderDialogPath(char const* title, mochi::Path const& defaultPath = {});
  // General entry point for "open whatever the user pointed at" -- the file dialogs, Open Recent,
  // MOCHI_AUTO_LOAD and OS file drops all route through here. A directory (or a root marker file,
  // which stands for the directory containing it) goes to @ref AddFolderToWorkspace; anything else
  // is imported or loaded as an asset via @ref OpenFile. Records the matching Recent entry on
  // success. Returns false if nothing could be opened.
  bool OpenPath(mochi::Path const& path);
  // The one way a folder enters the workspace: adds @p path to the Asset Browser as a root,
  // navigates to it, records it under Recent Folders and brings the browser forward. Additive --
  // the folders already open stay open, and are removed deliberately via the tree's "Remove from
  // Workspace". Returns false, leaving the browser untouched, if the Asset Browser refused @p path
  // for exceeding the file/folder limit.
  bool AddFolderToWorkspace(mochi::Path const& path);
  // Imports or loads @p path as an asset and opens an editor for it. Folders are not handled here;
  // call @ref OpenPath when @p path may be either.
  bool OpenFile(mochi::Path const& path);
  void AddRecentFile(mochi::Path const& path);
  void AddRecentFolder(mochi::Path const& path);
  static void CreateRootMarker(mochi::Path const& dir, mochi::Error& error);

  //------------------------------------------------------------------------------------------------
  // Settings
  //------------------------------------------------------------------------------------------------
  void LoadSettings();
  void SaveSettings();
  // Restores every Settings-window category to its defaults. Recent files, window visibility and
  // the open workspace roots are preserved.
  void ResetAllSettings();
  // Writes a pending Settings-window edit and runs its category's commit hook, once the edit is
  // released. Called every frame; a no-op when nothing is pending.
  void CommitPendingSettings();
  // Pushes AppSettings::viewport::selection::highlightOverlayOpacity into every open viewport.
  void ApplyHighlightOverlayOpacity();
  // Pushes AppSettings::graphics into every open editor and the renderer's clear color.
  void ApplyGraphicsSettings();
  // Broadcasts a committed settings change to every open editor (see
  // AssetEditor::OnAppSettingsChanged).
  void NotifyAppSettingsChanged();

  //------------------------------------------------------------------------------------------------
  // Importers
  //------------------------------------------------------------------------------------------------
  void RegisterImporter(std::unique_ptr<Importer> importer);
  Importer* FindImporterForPath(mochi::Path const& path) const;
  bool CanImport() const;
  void BeginImport(Importer* importer, mochi::Path const& path);

  //------------------------------------------------------------------------------------------------
  // Tasks
  //------------------------------------------------------------------------------------------------
  bool BeginAsyncTasks(
      std::string title,
      std::vector<AsyncTask> tasks,
      std::function<void(bool allSucceeded)> onComplete,
      bool serial = false);
  // True while an async task batch (and its progress modal) is active. Only one batch runs at a
  // time, so callers use this to disable controls that would start another.
  bool IsAsyncTasksRunning() const;

  //------------------------------------------------------------------------------------------------
  // ImGui
  //------------------------------------------------------------------------------------------------
  void ShowMainMenu();
  void ShowAssetEditorWindow();
  void ShowSettingsWindow(bool* open);
  bool ShowGraphicsSettings();
  bool ShowPhysicsSettings();
  void BuildDefaultDockLayout();

  //------------------------------------------------------------------------------------------------
  // Close guard
  //------------------------------------------------------------------------------------------------
  bool GuardUnsavedAssets(char const* prompt, std::function<void()> onProceed);
  bool RequestAppClose();
  void ShowUnsavedChangesModal();

 protected:
  // Function to check if the Log Console should be brough to the front of its group on error.
  void MaybeRaiseLogConsole();

  std::unique_ptr<AssetBrowser> _assetBrowser;
  std::unique_ptr<LogConsole> _logConsole;
  std::unique_ptr<AssetManager> _assetManager;
  std::unique_ptr<mochi_renderer::MochiRenderer> _mochiRenderer;
  mochi_renderer::IBL* _defaultIbl =
      nullptr; // owned by the ResourceManager (inside _mochiRenderer)
  std::unique_ptr<Renderer> _renderer;
  std::unique_ptr<SuperDexStudioBotLoader> _botLoader;
  std::vector<std::unique_ptr<AssetEditor>> _assetEditors;
  std::vector<std::unique_ptr<Importer>> _importers;
  struct UnsavedAssetEntry {
    Asset* asset = nullptr;
    bool save = true;
  };
  bool _openUnsavedChangesModal = false;
  char const* _unsavedPrompt = nullptr;
  std::vector<UnsavedAssetEntry> _unsavedEntries;
  std::function<void()> _onUnsavedProceed;
  Importer* _activeImporter = nullptr; // importer whose modal is currently open, if any
  AsyncTaskRunner _asyncTasks; // runs importer background work + its progress modal
  int _activeAssetEditorIdx = -1;
  bool _needDefaultDockLayout = false;
  bool _needDefaultWindowFocus = false;
  bool _focusMode = false; // hides docked panels and editor tabs (F10); not OS fullscreen
  int _settingsCategoryIdx = 0;
  // Category whose Settings-window value changed, or -1 when nothing is pending. The save is
  // deferred until the edit is released, so it must outlive a category switch.
  int _settingsPendingSaveCategory = -1;
  bool _openResetSettingsModal = false;
  bool _logConsolePendingRaise = false;
  std::chrono::steady_clock::time_point _lastLogConsoleRaise{};
  mochi::CoordinateSpace _editorSpace = mochi::CoordinateSpace::Default();
  mochi::CoordinateSpace _renderSpace = mochi_renderer::RenderSpace();
  mochi::CoordinateSpaceConverter _editorToRendererConverter;
  mochi::CoordinateSpaceConverter _renderToEditorConverter;
  mochi::Context* _mochiContext = nullptr;
  superdex::robotics::RoboticsContext* _botsContext = nullptr;
  AppSettings _appSettings;
};

} // namespace superdex::studio
