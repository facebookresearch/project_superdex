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

#include "assets/bot_asset.h"
#include "ui/imgui_widgets.h"

#include <mochi_renderer/path.h>

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace superdex::studio {

class Asset;
class SuperDexStudio;
class AssetManager;
class Importer;

inline constexpr char kAssetBrowserDragDropType[] = "AssetBrowserItem";

struct AssetBrowserDragDropItem {
  bool isFolder = false;
  AssetType type = AssetType::Unknown;
  mochi::Path fullPath;
  Asset* asset = nullptr;
};

struct AssetBrowserDragDropPayload {
  std::vector<AssetBrowserDragDropItem> items;
};

struct FileEntry {
  std::string name;
  std::string stem;
  mochi::Path fullPath;
  AssetType type = AssetType::Unknown;
  Importer* importer = nullptr;
};

struct DirectoryNode {
  std::string name;
  mochi::Path fullPath;
  std::vector<DirectoryNode> children;
  std::vector<FileEntry> files;
  bool isMochiBotsRoot = false;
};

enum class ClipboardMode { None, Copy, Cut };

struct BatchRenameEntry {
  mochi::Path path;
  std::string originalStem;
  std::string extension;
  bool isFolder = false;
};

struct TileItem {
  mochi::Path fullPath;
  std::string name;
  bool isFolder = false;
  FileEntry const* file = nullptr;
  DirectoryNode const* dir = nullptr;
};

class AssetBrowser {
 public:
  static std::unique_ptr<AssetBrowser> Create(SuperDexStudio* studio, AssetManager* assetManager);

  AssetBrowser(AssetBrowser const&) = delete;
  AssetBrowser& operator=(AssetBrowser const&) = delete;
  AssetBrowser(AssetBrowser&&) = delete;
  AssetBrowser& operator=(AssetBrowser&&) = delete;
  ~AssetBrowser() = default;

  void ShowWindow(bool* open);
  void SelectAsset(Asset* asset);
  // Adds @p root as an additional workspace root, keeping the roots already open. Returns whether
  // @p root is in the workspace afterwards: false only when it is refused for exceeding the
  // file/folder limit (which leaves the existing roots unchanged); a @p root already covered by an
  // existing root is not added again but still returns true.
  bool AddRootPath(mochi::Path const& root);
  void RemoveRootPath(mochi::Path const& root);
  void ClearRootPaths();
  void SetCurrentPath(mochi::Path const& path);
  mochi::Path const& GetCurrentPath() const;
  void Refresh();

 private:
  AssetBrowser(SuperDexStudio* studio, AssetManager* assetManager);

  // Adds @p root without the size check -- callers that already validated it (AddRootPath via
  // RootExceedsEntryLimit) use this so the folder isn't walked twice. Handles dedup/fold with
  // existing roots, tag-dir expansion, current-path/selection, and persistence.
  void AddRootPathUnchecked(mochi::Path const& root);
  // Bounded probe: logs and returns true when @p root has more than the configured file/folder
  // limit (too large to use as a root). Checked before any destructive change so a refusal leaves
  // the browser's current roots untouched.
  bool RootExceedsEntryLimit(mochi::Path const& root) const;

  bool MoveItems(std::set<mochi::Path> const& sources, mochi::Path const& destDir);
  bool CopyItems(std::set<mochi::Path> const& sources, mochi::Path const& destDir);
  bool DeleteItems(std::set<mochi::Path> paths);
  bool CreateFolder(mochi::Path const& parentDir, std::string const& name);
  bool RenameFolder(mochi::Path const& folderPath, std::string const& newName);
  bool RenameAsset(mochi::Path const& assetPath, std::string const& newStem);
  void DuplicateItem(TileItem const& item);
  void CreateModBotAsset(mochi::Path const& dir, std::string const& base = {});

  DirectoryNode const* FindNode(mochi::Path const& path) const;
  mochi::Path const* FindOwningRoot(mochi::Path const& path) const;
  void TryRefreshDirectoryTree();
  void CollectFilesRecursive(DirectoryNode const& node, std::vector<TileItem>& out) const;
  // Recursively collects the file paths under @p dir (for folder copy / delete / drag), bounded by
  // the configured file/folder limit (AssetBrowserSettings.fileFolderLimit); logs a warning and
  // stops if the limit is hit.
  std::vector<mochi::Path> CollectFilePaths(mochi::Path const& dir) const;
  bool IsFilterActive() const;

  // Folder navigation history for the mouse back/forward buttons (web-browser style).
  // ApplyCurrentPath switches folders without touching history; SetCurrentPath (the public entry
  // point every navigation goes through) records history on top of it.
  void ApplyCurrentPath(mochi::Path const& path);
  void GoBack();
  void GoForward();

  void ShowFilter(float width);
  void ShowFolderTree(DirectoryNode const& node, bool isRoot);
  void ShowBreadcrumb();
  void ShowSettingsCog();
  void ShowDeleteBlockedModal();
  void ShowBatchRenameModal();
  void ShowReplaceWithModal();
  void ShowItemContextMenuContents(TileItem const& item, Asset const* asset);
  void TryApplyPendingAssetAction();
  void TryShowPasteSelectable(mochi::Path const& destDir);
  void TryAcceptAssetBrowserDrop(mochi::Path const& destDir, bool setCursorNotAllowed);
  void ClearSelection();
  void ExecutePaste(mochi::Path const& destDir);
  void BeginRenaming(
      mochi::Path const& path,
      std::string const& displayName,
      std::function<void(std::string const&)> onFinished);

 private:
  SuperDexStudio* _studio = nullptr;
  AssetManager* _assetManager = nullptr;
  bool _assetTypeFilters[kAssetTypeCount] = {};
  bool _importableFilter = false;
  ImGuiTextFilter _assetNameFilter;
  std::set<mochi::Path> _selectedPaths;
  int _lastClickedIndex = -1;
  bool _forceAssetFocus = false;
  std::vector<mochi::Path> _rootPaths;
  std::vector<DirectoryNode> _rootNodes;
  mochi::Path _currentPath;
  bool _expandTreeToCurrentPath = false;
  // Visited-folder history and the current position within it. Back/forward move _navIndex without
  // truncating; a fresh navigation drops any entries after _navIndex.
  std::vector<mochi::Path> _navHistory;
  int _navIndex = -1;
  bool _needsRefresh = true;
  std::vector<TileItem> _tileItems;
  std::vector<int> _visibleIndices;
  ClipboardMode _clipboardMode = ClipboardMode::None;
  std::set<mochi::Path> _clipboardPaths;
  // Asset-lifetime actions requested from the item context menu. These destroy Asset objects (and
  // with them the thumbnail textures already recorded into this frame's ImGui draw list), while the
  // menu and tile code still hold raw Asset pointers, so they run at the start of the next frame.
  enum class PendingAssetAction { None, Unload, Reload, Delete };
  PendingAssetAction _pendingAssetAction = PendingAssetAction::None;
  std::set<mochi::Path> _pendingAssetActionPaths;
  mochi::Path _renamingPath;
  ImGui::TileRenameState _tileRename;
  AssetBrowserDragDropPayload _dragPayload;
  bool _openDeleteBlockedPopup = false;
  struct DeleteBlocker {
    mochi::Path file;
    std::vector<IAssetReferencer*> referencers;
  };
  std::vector<DeleteBlocker> _deleteBlockers;
  bool _openBatchRenamePopup = false;
  std::vector<BatchRenameEntry> _batchRenameEntries;
  BatchRenameParams _batchRenameInputs;
  bool _openReplaceWithPopup = false;
  std::set<mochi::Path> _replaceTargets;
  AssetType _replaceType = AssetType::Unknown;
  mochi::DynamicString _replaceSelection;
};

} // namespace superdex::studio
