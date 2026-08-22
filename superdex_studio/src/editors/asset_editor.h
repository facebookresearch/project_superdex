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
#include "core/settings.h"
#include "rendering/viewport.h"

namespace superdex::studio {

class AssetEditor : public IAssetReferencer {
 public:
  //------------------------------------------------------------------------------------------------
  // General
  //------------------------------------------------------------------------------------------------

  // Returns the asset this AssetEditor is targeting.
  Asset* GetAsset() const;

  // Whether this editor already represents @p asset, so opening @p asset should focus this editor
  // rather than spawn a new one. Defaults to the single target asset; editors that hold several
  // related assets (e.g. the Model Editor's CAD / render / mochi slots) override this to include
  // all of them.
  virtual bool RepresentsAsset(Asset const* asset) const {
    return asset != nullptr && GetAsset() == asset;
  }

  // Called when OpenAssetEditor re-focuses this already-open editor for @p asset (instead of
  // spawning a new one; see RepresentsAsset). Lets an editor that holds several related assets
  // react to which one was opened -- e.g. the Model Editor focuses that asset's slot visualization,
  // as if it had just been opened. Default: no-op.
  virtual void OnReopenedFor(Asset* /*asset*/) {}

  // Rebuild/restage the editor's live view after an asset it depends on was rewritten out-of-band
  // -- e.g. the asset manager rewrote references during an asset replace / rename / move, or (in
  // the future) the backing file changed on disk. Default: no-op.
  virtual void Refresh() {}

  //------------------------------------------------------------------------------------------------
  // App Loop Hooks
  //------------------------------------------------------------------------------------------------

  // Called immediately after the AssetEditor is created.
  virtual void Initialize() {}
  // Called once per frame to handle inputs (i.e. keyboard shortcuts) before UI is drawn.
  virtual void OnHandleInputs() {}
  // Called once per frame before UI is drawn. Passed global renderer for Viewport/Scene rendering.
  virtual void OnRender(Renderer const*) {}
  // Called when the AssetEditor is about to be destroyed.
  virtual void Shutdown() {}

  //------------------------------------------------------------------------------------------------
  // Main Editor Tab
  //------------------------------------------------------------------------------------------------

  // Return the name to display for the AssetEditor's main tab (default is target asset name).
  virtual std::string const& GetTabDisplayName() const;
  // Return a color to decorate the AssetEditor's main tab (default is target asset color).
  virtual ImU32 GetTabColor() const;
  // Called when the editor tab selected and when it is first opened.
  virtual void OnActivate() {}
  // Called when the editor tab is deselected because another editor tab was selected.
  virtual void OnDeactivate() {}
  // Show the contents of the AssetEditor's main tab with ImGui calls (required).
  virtual void ShowTabContents() = 0;

  //------------------------------------------------------------------------------------------------
  // Auxiliary Windows (Hierarchy, Details, etc.)
  //------------------------------------------------------------------------------------------------

  // Where a window is placed in the app's default dock layout (see
  // SuperDexStudio::BuildDefaultDockLayout). Regions map to nodes of the default layout; Floating
  // windows are left undocked.
  enum class DockRegion {
    Floating,
    SidePanelTop,
    SidePanelBottom,
    BottomBar,
  };

  struct WindowDeclaration {
    char const* name = nullptr; //< the name of the window and menu entries for it
    bool showByDefault = true; //< show the window by default when the editor is initialized
    DockRegion dock = DockRegion::SidePanelTop; //< default dock location in the layout
    bool debug = false; //< organize the window in the Debug section of the main menu bar
  };

  // Return a list of auxiliary ImGui windows and default open states this AssetEditor will display.
  // This can be used to show hierarchical side panels, details, utility windows, etc. The windows
  // will be openable/closeable from the main app's Window menu.
  virtual std::vector<WindowDeclaration> GetAuxiliaryWindows() const;
  // Implement to show each window declared above with ImGui::Begin/End calls. Pass the return value
  // of _studio->GetWindowVisible("WindowName") to each ImGui::Begin() call for close buttons to
  // work.
  virtual void ShowAuxiliaryWindows();

  // Raises an auxiliary window to the front of its dock node, but only if it is currently open.
  // Use instead of ImGui::SetWindowFocus so a closed window is never brought back.
  void FocusWindowIfOpen(char const* name);

  //------------------------------------------------------------------------------------------------
  // Main Menu
  //------------------------------------------------------------------------------------------------

  // Each AssetEditor is given the opportunity to show main menu items. Use this to implement asset
  // specific actions or functions. Begin and end each menu with ImGui::BeginMenu/EndMenu.
  virtual void ShowMainMenuItems() {}

  //------------------------------------------------------------------------------------------------
  //  Save/Undo/Redo Hooks
  //------------------------------------------------------------------------------------------------

  // Return true if the AssetEditor is in a read-only state (default returns Asset::IsReadOnly).
  bool IsReadOnly() const;
  // Return true if the AssetEditor is in a dity state (default returns Asset::IsDirty).
  bool IsDirty() const;
  // Implement logic to save the asset (and other state) and return true on success (default calls
  // Asset::Save and Asset::SetDirty(false) if it succeeds)
  bool Save();

  // Returns the UndoStack associated with the AssetEditor. Implementations must implement
  // _undoStack if they want to opt-in for Undo/Redo actions managed by the main application.
  UndoStack& GetUndoStack();
  // Return false to gate undo/redo operations (e.g. prevent undo/redo while simulating).
  virtual bool CanUndoRedo() const;

  //------------------------------------------------------------------------------------------------
  // Misc.
  //------------------------------------------------------------------------------------------------

  // Called when app level SceneViewSettings are edited. Implement to apply settings to any owned
  // scenes that should have a consistent appearance.
  virtual void ApplySceneViewSettings(mochi_renderer::SceneViewSettings const& settings);

  // Called after any app setting is committed (including a reset). Implement to push settings an
  // editor has already copied into live state -- notably anything snapshotted into a running
  // simulation, which cannot re-read AppSettings itself because it steps on another thread.
  virtual void OnAppSettingsChanged(AppSettings const& settings);

  // The editor's primary viewport, if it has one (default: none). Lets app-level UI (e.g. the View
  // Settings window) reach per-viewport state such as the highlight overlay opacity.
  virtual Viewport* GetViewport() {
    return nullptr;
  }

  //------------------------------------------------------------------------------------------------
  // IAssetReferencer
  //------------------------------------------------------------------------------------------------

  // Returns GetDisplayName() by default. This only effects what is displayed in delete modals.
  std::string const& GetReferencerName() const override;
  // Each AssetEditor by default references its target asset so they cannot be deleted while open.
  // If an AssetEditor needs to reference additional assets, it should override this function.
  void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const override;
  // Returns true by default so that AssetEditor always references its asset if the asset moves.
  // Override if the AssetEditor needs to update paths to assets other than the target asset when
  // they change.
  bool RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) override;

 protected:
  AssetEditor(SuperDexStudio* studio, Asset* asset);

 protected:
  SuperDexStudio* _studio = nullptr;
  Asset* _asset = nullptr;
  UndoStack _undoStack;
  friend class SuperDexStudio;
  bool _needsSelect = false;
};

} // namespace superdex::studio
