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

#include "ui/asset_browser.h"
#include "app/app.h"
#include "assets/asset.h"

#include <mochi_physics/utils/mochi_prefab.h>
#include <superdex_robotics/utils/file_utils.h>

#include <imguios/fonts/icons_font_awesome5.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <unordered_set>
#include <utility>

namespace superdex::studio {

constexpr float kFolderTreeIndent = 10.0f;
constexpr float kSidebarWidth = 200.0f;
constexpr float kFilterWidth = 150.0f;

// Pixel sizes offered by the "Save Thumbnail(s)" menus. Shared so the batch and single-asset menus
// stay in sync.
constexpr std::array<int, 6> kThumbnailExportSizesPx = {64, 128, 256, 512, 1024, 2048};

// Recursively scans @p dir into @p node, decrementing @p budget once per visited entry and stopping
// once it is exhausted (shared across the whole refresh, so it bounds wide and deep trees alike).
// The budget is a safety net: a newly-added root is refused up front if it exceeds the cap (see
// AddRootPath), so this only trips for a root persisted before that check existed -- a partial tree
// then, which the caller reports, rather than a hang.
static void ScanDirectory(
    mochi::Path const& dir,
    DirectoryNode& node,
    SuperDexStudio const* studio,
    int& budget) {
  node.fullPath = dir;
  node.name = dir.GetFilename();
  if (node.name.empty() || node.name == "/") {
    node.name = dir.GetParentPath().GetFilename();
  }
  std::error_code ec;
  node.isMochiBotsRoot = superdex::robotics::FindRootMarker(dir.AsFilesystemPath()).has_value();
  for (auto const& entry : std::filesystem::directory_iterator(dir.AsFilesystemPath(), ec)) {
    if (budget <= 0) {
      break;
    }
    --budget;
    if (entry.is_directory(ec)) {
      // Skip hidden directories
      auto const name = entry.path().filename().string();
      if (!name.empty() && name[0] == '.') {
        continue;
      }
      DirectoryNode child;
      ScanDirectory(entry.path(), child, studio, budget);
      node.children.push_back(std::move(child));
    } else if (entry.is_regular_file(ec)) {
      auto type = ClassifyAssetTypeByFilename(entry.path().filename().string());
      FileEntry file;
      file.name = entry.path().filename().string();
      file.stem = GetAssetNameFromPath(entry.path());
      file.fullPath = entry.path();
      file.type = type;
      if (type == AssetType::Unknown) {
        file.importer = studio->FindImporterForPath(file.fullPath);
      }
      node.files.push_back(std::move(file));
    }
  }
  std::ranges::sort(node.children, [](auto const& a, auto const& b) { return a.name < b.name; });
  std::ranges::sort(node.files, [](auto const& a, auto const& b) { return a.name < b.name; });
}

static DirectoryNode const* FindNodeRecursive(DirectoryNode const& node, mochi::Path const& path) {
  if (node.fullPath == path) {
    return &node;
  }
  for (auto const& child : node.children) {
    if (auto const* found = FindNodeRecursive(child, path)) {
      return found;
    }
  }
  return nullptr;
}

std::vector<mochi::Path> AssetBrowser::CollectFilePaths(mochi::Path const& dir) const {
  std::vector<mochi::Path> filePaths;
  std::error_code ec;
  int const limit = _studio->GetAppSettings().assetBrowser.fileFolderLimit;
  int budget = limit;
  for (auto const& entry :
       std::filesystem::recursive_directory_iterator(dir.AsFilesystemPath(), ec)) {
    if (budget <= 0) {
      MOCHI_LOG_WARNING(
          "Asset Browser: folder '%s' exceeds the %d-entry limit; only the first %d were processed.",
          dir.ToString().c_str(),
          limit,
          limit);
      break;
    }
    --budget;
    if (entry.is_regular_file(ec)) {
      filePaths.emplace_back(entry.path());
    }
  }
  return filePaths;
}

// Returns true if @p dir contains more than @p limit filesystem entries (files + folders). Bounded:
// the walk stops the moment the limit is exceeded, so it never traverses a huge tree in full (used
// to refuse an over-large folder as a root before it is scanned or persisted).
static bool DirectoryExceedsEntryLimit(mochi::Path const& dir, int limit) {
  std::error_code ec;
  int count = 0;
  for (auto const& entry : std::filesystem::recursive_directory_iterator(
           dir.AsFilesystemPath(),
           std::filesystem::directory_options::skip_permission_denied,
           ec)) {
    (void)entry;
    if (++count > limit) {
      return true;
    }
  }
  return false;
}

AssetBrowser::AssetBrowser(SuperDexStudio* studio, AssetManager* assetManager)
    : _studio(studio), _assetManager(assetManager) {}

std::unique_ptr<AssetBrowser> AssetBrowser::Create(
    SuperDexStudio* studio,
    AssetManager* assetManager) {
  if (studio && assetManager) {
    return std::unique_ptr<AssetBrowser>(new AssetBrowser(studio, assetManager));
  }
  MOCHI_LOG_ERROR("AssetBrowser requires valid studio and asset manager references!");
  return nullptr;
}

void AssetBrowser::TryRefreshDirectoryTree() {
  if (!_needsRefresh) {
    return;
  }
  _needsRefresh = false;

  if (_rootPaths.empty()) {
    auto const& saved = _studio->GetAppSettings().assetBrowser.rootPaths;
    for (auto const& s : saved) {
      if (!s.empty() && std::filesystem::exists(s)) {
        _rootPaths.emplace_back(s);
      }
    }
    if (!_rootPaths.empty() && _currentPath.IsEmpty()) {
      SetCurrentPath(_rootPaths.front());
    }
  }

  // One budget shared across all roots: the scan stops once the total entry count is hit, so even a
  // persisted over-large root (added before this limit existed, or set outside AddRootPath) yields
  // a partial tree instead of hanging the UI.
  int const limit = _studio->GetAppSettings().assetBrowser.fileFolderLimit;
  int budget = limit;
  _rootNodes.resize(_rootPaths.size());
  for (size_t i = 0; i < _rootPaths.size(); ++i) {
    _rootNodes[i] = {};
    ScanDirectory(_rootPaths[i], _rootNodes[i], _studio, budget);
  }
  if (budget <= 0) {
    MOCHI_LOG_WARNING(
        "Asset Browser: directory scan hit the %d-entry limit; the folder view is partial. Point at "
        "a more specific folder (or remove an over-large root).",
        limit);
  }

  if (_currentPath.IsEmpty() && !_rootPaths.empty()) {
    SetCurrentPath(_rootPaths.front());
  }
}

static void SyncRootPathsToSettings(
    std::vector<mochi::Path> const& rootPaths,
    AssetBrowserSettings& settings) {
  settings.rootPaths.clear();
  settings.rootPaths.reserve(rootPaths.size());
  for (auto const& p : rootPaths) {
    settings.rootPaths.push_back(p.ToString());
  }
}

bool AssetBrowser::RootExceedsEntryLimit(mochi::Path const& root) const {
  // Refuse a folder too large to scan (e.g. the repo root, picked by mistake): a bounded probe
  // stops as soon as it exceeds the cap, so this never walks the whole tree.
  int const entryLimit = _studio->GetAppSettings().assetBrowser.fileFolderLimit;
  if (DirectoryExceedsEntryLimit(root, entryLimit)) {
    MOCHI_LOG_ERROR(
        "Not adding '%s' as an Asset Browser root: it contains more than %d files/folders. Point at "
        "a more specific folder.",
        root.ToString().c_str(),
        entryLimit);
    return true;
  }
  return false;
}

bool AssetBrowser::AddRootPath(mochi::Path const& root) {
  if (RootExceedsEntryLimit(root)) {
    return false;
  }
  AddRootPathUnchecked(root);
  return true;
}

void AssetBrowser::AddRootPathUnchecked(mochi::Path const& root) {
  // If root is a child of (or equal to) an existing root, skip
  for (auto const& existing : _rootPaths) {
    if (root.IsDescendantOf(existing)) {
      return;
    }
  }
  // If root is a parent of existing roots, fold them
  std::erase_if(_rootPaths, [&](auto const& existing) { return existing.IsDescendantOf(root); });
  _rootPaths.push_back(root);
  _needsRefresh = true;
  ClearSelection();
  // If this directory contains a .superdex_root with tags, recursively add them
  auto rootFile = superdex::robotics::FindRootMarker(root.AsFilesystemPath());
  if (rootFile) {
    mochi::ErrorLog error;
    auto parsed = superdex::robotics::ParseRootFile(*rootFile, error);
    if (error.IsOK()) {
      for (auto const& [tag, tagDir] : parsed.tags) {
        AddRootPath(tagDir);
      }
    }
  }
  SetCurrentPath(root);
  SyncRootPathsToSettings(_rootPaths, _studio->GetAppSettings().assetBrowser);
  _studio->SaveSettings();
}

void AssetBrowser::RemoveRootPath(mochi::Path const& root) {
  std::erase_if(_rootPaths, [&](auto const& p) { return p == root; });
  _rootNodes.clear();
  // If current path was under the removed root, reset
  if (!_currentPath.IsEmpty() && _currentPath.IsDescendantOf(root)) {
    SetCurrentPath(_rootPaths.empty() ? mochi::Path{} : _rootPaths.front());
  }
  _needsRefresh = true;
  ClearSelection();
  SyncRootPathsToSettings(_rootPaths, _studio->GetAppSettings().assetBrowser);
  _studio->SaveSettings();
}

void AssetBrowser::ClearRootPaths() {
  _rootPaths.clear();
  _rootNodes.clear();
  _currentPath = {};
  _navHistory.clear();
  _navIndex = -1;
  _needsRefresh = true;
  ClearSelection();
  _studio->GetAppSettings().assetBrowser.rootPaths.clear();
  _studio->SaveSettings();
}

void AssetBrowser::ApplyCurrentPath(mochi::Path const& path) {
  _currentPath = path;
  _expandTreeToCurrentPath = true;
  _studio->GetAppSettings().assetBrowser.lastPath = path.ToString();
}

void AssetBrowser::SetCurrentPath(mochi::Path const& path) {
  ApplyCurrentPath(path);
  // Record folder navigation history for the mouse back/forward buttons. Skip a no-op
  // re-navigation to the folder we are already on, and drop any forward entries -- a fresh
  // navigation starts a new forward branch, matching web-browser behavior.
  if (_navIndex >= 0 && _navIndex < static_cast<int>(_navHistory.size()) &&
      _navHistory[static_cast<size_t>(_navIndex)] == path) {
    return;
  }
  _navHistory.resize(static_cast<size_t>(_navIndex) + 1);
  _navHistory.push_back(path);
  _navIndex = static_cast<int>(_navHistory.size()) - 1;
}

void AssetBrowser::GoBack() {
  if (_navIndex > 0) {
    --_navIndex;
    ApplyCurrentPath(_navHistory[static_cast<size_t>(_navIndex)]);
  }
}

void AssetBrowser::GoForward() {
  if (_navIndex >= 0 && _navIndex + 1 < static_cast<int>(_navHistory.size())) {
    ++_navIndex;
    ApplyCurrentPath(_navHistory[static_cast<size_t>(_navIndex)]);
  }
}

mochi::Path const& AssetBrowser::GetCurrentPath() const {
  return _currentPath;
}

void AssetBrowser::Refresh() {
  _needsRefresh = true;
}

void AssetBrowser::ClearSelection() {
  _selectedPaths.clear();
  _lastClickedIndex = -1;
}

void AssetBrowser::ExecutePaste(mochi::Path const& destDir) {
  if (_clipboardMode == ClipboardMode::Cut) {
    MoveItems(_clipboardPaths, destDir);
  } else if (_clipboardMode == ClipboardMode::Copy) {
    CopyItems(_clipboardPaths, destDir);
  }
  _clipboardPaths.clear();
  _clipboardMode = ClipboardMode::None;
}

void AssetBrowser::BeginRenaming(
    mochi::Path const& path,
    std::string const& displayName,
    std::function<void(std::string const&)> onFinished) {
  _needsRefresh = true;
  _selectedPaths.clear();
  _selectedPaths.insert(path);
  _renamingPath = path;
  _tileRename.buffer = displayName;
  _tileRename.onFinished = [this, onFinished = std::move(onFinished)](std::string const& newName) {
    if (onFinished) {
      onFinished(newName);
    }
    _renamingPath = {};
  };
  _tileRename.onCanceled = [this]() {
    DeleteItems({_renamingPath});
    _renamingPath = {};
  };
}

void AssetBrowser::ShowFilter(float width) {
  ImGui::BeginChild(
      "##AssetFilters", ImVec2(width, 0), ImGuiChildFlags_Border | ImGuiChildFlags_ResizeX);
  float const buttonWidth = ImGui::GetContentRegionAvail().x;
  ImGui::SetNextItemWidth(buttonWidth);
  if (ImGui::InputTextWithHint(
          "##NameFilter",
          ICON_FA_FILTER " Filter",
          _assetNameFilter.InputBuf,
          IM_ARRAYSIZE(_assetNameFilter.InputBuf))) {
    _assetNameFilter.Build();
  }
  ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));

  bool const showUnknownFiles = _studio->GetAppSettings().assetBrowser.showUnknownFiles;
  bool const showImportableFiles = _studio->GetAppSettings().assetBrowser.showImportableFiles;
  constexpr int kUnknownIndex = kAssetTypeCount - 1;
  auto drawTypeSwatch = [&](int i) {
    auto type = static_cast<AssetType>(i);
    if (ImGui::ColorSwatchButton(
            GetAssetTypeLabel(type), GetAssetTypeColor(type), !_assetTypeFilters[i], buttonWidth)) {
      if (ImGui::GetIO().KeyCtrl) {
        for (int j = 0; j < kAssetTypeCount; ++j) {
          _assetTypeFilters[j] = (j == i);
        }
        _importableFilter = false;
      } else {
        _assetTypeFilters[i] = !_assetTypeFilters[i];
      }
    }
  };
  // Real-type swatches (Unknown handled separately so Importable can precede it).
  for (int i = 0; i < kUnknownIndex; ++i) {
#if !MOCHI_INTERNAL
    if (static_cast<AssetType>(i) == AssetType::BotScene) {
      continue;
    }
#endif
    drawTypeSwatch(i);
  }
  // Importable is a cross-cutting category (driven by the importer registry), so
  // it is a separate toggle rather than an AssetType-indexed entry.
  if (showImportableFiles) {
    if (ImGui::ColorSwatchButton(
            "Importable", IM_COL32(255, 255, 255, 255), !_importableFilter, buttonWidth)) {
      if (ImGui::GetIO().KeyCtrl) {
        for (int j = 0; j < kAssetTypeCount; ++j) {
          _assetTypeFilters[j] = false;
        }
        _importableFilter = true;
      } else {
        _importableFilter = !_importableFilter;
      }
    }
  }
  if (showUnknownFiles) {
    drawTypeSwatch(kUnknownIndex);
  }
  ImGui::EndChild();
}

bool AssetBrowser::IsFilterActive() const {
  if (!_studio->GetAppSettings().assetBrowser.showFilters) {
    return false;
  }
  if (_assetNameFilter.IsActive()) {
    return true;
  }
  bool const showUnknownFiles = _studio->GetAppSettings().assetBrowser.showUnknownFiles;
  int const numTypeFilters = showUnknownFiles ? kAssetTypeCount : kAssetTypeCount - 1;
  for (int i = 0; i < numTypeFilters; ++i) {
    if (_assetTypeFilters[i]) {
      return true;
    }
  }
  return _importableFilter && _studio->GetAppSettings().assetBrowser.showImportableFiles;
}

void AssetBrowser::CollectFilesRecursive(DirectoryNode const& node, std::vector<TileItem>& out)
    const {
  for (auto const& file : node.files) {
    out.push_back({file.fullPath, file.name, false, &file, nullptr});
  }
  for (auto const& child : node.children) {
    CollectFilesRecursive(child, out);
  }
}

void AssetBrowser::ShowFolderTree(DirectoryNode const& node, bool isRoot) {
  bool const isSelected =
      !_currentPath.IsEmpty() && !node.fullPath.IsEmpty() && node.fullPath == _currentPath;
  bool const isLeaf = node.children.empty();

  // Auto-expand nodes on the path to _currentPath
  bool const isAncestorOfCurrent = !_currentPath.IsEmpty() && !node.fullPath.IsEmpty() &&
      _currentPath != node.fullPath && _currentPath.IsDescendantOf(node.fullPath);

  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanFullWidth |
      ImGuiTreeNodeFlags_OpenOnDoubleClick;
  if (isLeaf) {
    flags |= ImGuiTreeNodeFlags_Leaf;
  }
  if (isSelected) {
    flags |= ImGuiTreeNodeFlags_Selected;
  }
  if (isRoot) {
    flags |= ImGuiTreeNodeFlags_DefaultOpen;
  }
  if (_expandTreeToCurrentPath && isAncestorOfCurrent) {
    ImGui::SetNextItemOpen(true);
  }

  // Determine if the tree node is currently open so we can pick the right folder icon.
  ImGuiID const nodeId = ImGui::GetID(node.fullPath.ToString().c_str());
  int const defaultOpenInt = (isRoot || (_expandTreeToCurrentPath && isAncestorOfCurrent)) ? 1 : 0;
  bool const isOpen = !isLeaf && ImGui::GetStateStorage()->GetInt(nodeId, defaultOpenInt) != 0;

  char const* icon = node.isMochiBotsRoot ? ICON_FA_DATABASE
      : isOpen                            ? ICON_FA_FOLDER_OPEN
                                          : ICON_FA_FOLDER;

  if (isRoot) {
    ImGui::PushFont(_studio->GetFont("Roboto Bold"));
  }
  bool nodeOpen =
      ImGui::TreeNodeEx(node.fullPath.ToString().c_str(), flags, "%s %s", icon, node.name.c_str());
  if (isRoot) {
    ImGui::PopFont();
  }

  if (_expandTreeToCurrentPath && isSelected) {
    ImGui::ScrollToItem();
  }

  if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
    SetCurrentPath(node.fullPath);
  }

  // Drop target on tree node
  TryAcceptAssetBrowserDrop(node.fullPath, true);

  if (isRoot && ImGui::BeginItemTooltip()) {
    ImGui::Text("%s", node.fullPath.ToString().c_str());
    ImGui::EndTooltip();
  }

  if (isRoot &&
      ImGui::BeginPopupContextItem("##RootContextMenu", ImGuiPopupFlags_MouseButtonRight)) {
    if (ImGui::IconSelectable("Remove from Workspace", ICON_FA_TIMES)) {
      RemoveRootPath(node.fullPath);
    }
    ImGui::EndPopup();
  }

  if (nodeOpen) {
    for (auto const& child : node.children) {
      ShowFolderTree(child, false);
    }
    ImGui::TreePop();
  }
}

void AssetBrowser::ShowBreadcrumb() {
  if (_rootPaths.empty() || _currentPath.IsEmpty()) {
    ImGui::TextDisabled("No folder open.");
    return;
  }

  auto const* owningRoot = FindOwningRoot(_currentPath);
  if (!owningRoot) {
    ImGui::TextDisabled("No folder open.");
    return;
  }

  // Find the root node for the owning root
  DirectoryNode const* rootNode = nullptr;
  for (auto const& rn : _rootNodes) {
    if (rn.fullPath == *owningRoot) {
      rootNode = &rn;
      break;
    }
  }

  if (!rootNode) {
    ImGui::TextDisabled("No folder open.");
    return;
  }

  auto const relPath = _currentPath.RelativeToParent(*owningRoot);

  // Shows a caret button which opens a popup listing the child folders of `folderPath`.
  // Selecting a child navigates to it.
  auto showCaret = [this](mochi::Path const& folderPath) {
    ImGui::PushID(folderPath.ToString().c_str());
    ImGui::PushID("##caret");
    ImGui::PushFont(_studio->GetFont("Roboto Regular Large"));
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    if (ImGui::TextButton(ICON_FA_CARET_RIGHT)) {
      ImGui::OpenPopup("##BreadcrumbChildren");
    }
    ImGui::PopStyleColor();
    ImGui::PopFont();
    if (ImGui::BeginPopup("##BreadcrumbChildren")) {
      auto const* folderNode = FindNode(folderPath);
      if (folderNode && !folderNode->children.empty()) {
        for (auto const& child : folderNode->children) {
          if (ImGui::IconSelectable(child.name.c_str(), ICON_FA_FOLDER)) {
            SetCurrentPath(child.fullPath);
          }
        }
      } else {
        ImGui::TextDisabled("(no subfolders)");
      }
      ImGui::EndPopup();
    }
    ImGui::PopID();
    ImGui::PopID();
  };

  if (ImGui::TextButton(rootNode->name.c_str())) {
    SetCurrentPath(*owningRoot);
  }
  TryAcceptAssetBrowserDrop(*owningRoot, true);

  mochi::Path finalPath = *owningRoot;
  if (!relPath.empty()) {
    mochi::Path accumulated = *owningRoot;
    mochi::Path parent = *owningRoot;
    // Iterate path components via std::filesystem::path. The relative path is
    // pure lexical (no FS touch), and component iteration is read-only.
    std::filesystem::path const relFs{relPath};
    for (auto const& component : relFs) {
      ImGui::SameLine();
      showCaret(parent);
      ImGui::SameLine();
      accumulated = accumulated / component.string();
      ImGui::PushID(accumulated.ToString().c_str());
      if (ImGui::TextButton(component.string().c_str())) {
        SetCurrentPath(accumulated);
      }
      TryAcceptAssetBrowserDrop(accumulated, true);
      ImGui::PopID();
      parent = accumulated;
    }
    finalPath = accumulated;
  }

  // Trailing caret if the final folder is not a leaf.
  auto const* finalNode = FindNode(finalPath);
  if (finalNode && !finalNode->children.empty()) {
    ImGui::SameLine();
    showCaret(finalPath);
  }

  ImGui::Separator();
}

void AssetBrowser::ShowSettingsCog() {
  auto backup = ImGui::GetCursorPos();
  float const gearWidth = ImGui::CalcTextSize(ICON_FA_COG).x;
  ImGui::SetCursorPos(ImGui::GetCursorStartPos());
  ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - gearWidth);
  if (ImGui::TextButton(ICON_FA_COG)) {
    ImGui::OpenPopup("##AssetBrowserSettings");
  }
  ImGui::SetCursorPos(backup);
  if (ImGui::BeginPopup("##AssetBrowserSettings")) {
    ImGui::MenuItem(
        "Show Directory Tree", nullptr, &_studio->GetAppSettings().assetBrowser.showDirectoryTree);
    ImGui::MenuItem("Show Filters", nullptr, &_studio->GetAppSettings().assetBrowser.showFilters);
    ImGui::MenuItem(
        "Show Importable Files",
        nullptr,
        &_studio->GetAppSettings().assetBrowser.showImportableFiles);
    ImGui::MenuItem(
        "Show Unknown Files", nullptr, &_studio->GetAppSettings().assetBrowser.showUnknownFiles);
    ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
    auto& sortByType = _studio->GetAppSettings().assetBrowser.sortByType;
    if (ImGui::MenuItem("Sort by Path", nullptr, !sortByType)) {
      sortByType = false;
    }
    if (ImGui::MenuItem("Sort by Type", nullptr, sortByType)) {
      sortByType = true;
    }
    ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
    if (ImGui::MenuItem("Load All Visible Assets")) {
      for (int const idx : _visibleIndices) {
        auto const& item = _tileItems[idx];
        if (item.isFolder) {
          continue;
        }
        if (!_assetManager->FindAssetByPath(item.fullPath)) {
          _assetManager->LoadAsset(item.fullPath);
        }
      }
    }
    // Save Thumbnails: write a transparent <asset name>.png for every visible, loaded asset that
    // renders a thumbnail into a chosen folder (defaulting to the current browser folder). Honors
    // active filtering via _visibleIndices (the same set "Load All Visible Assets" uses).
    if (ImGui::BeginMenu("Save Thumbnails")) {
      for (int const size : kThumbnailExportSizesPx) {
        char label[32];
        snprintf(label, sizeof(label), "%d x %d", size, size);
        if (ImGui::MenuItem(label)) {
          mochi::Path const destDir =
              SuperDexStudio::GetFolderDialogPath("Save Thumbnails", _currentPath);
          if (!destDir.IsEmpty()) {
            // Visible assets can share a name (same stem but different extension, or same-named
            // files in different folders under recursive filtering), which would map to the same
            // output file. Track the names already written (case-insensitively, since the target
            // filesystem may be case-insensitive) and append a numeric suffix on collision so each
            // asset gets its own PNG instead of silently overwriting a previous one.
            std::unordered_set<std::string> usedNames;
            auto toLowerKey = [](std::string s) {
              std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
              });
              return s;
            };
            for (int const idx : _visibleIndices) {
              auto const& item = _tileItems[idx];
              if (item.isFolder) {
                continue;
              }
              Asset* asset = _assetManager->FindAssetByPath(item.fullPath);
              if (!asset || !asset->RendersThumbnail()) {
                continue;
              }
              std::string const baseName = asset->GetName();
              std::string fileName = baseName + ".png";
              for (int n = 2; !usedNames.insert(toLowerKey(fileName)).second; ++n) {
                fileName = baseName + "_" + std::to_string(n) + ".png";
              }
              mochi::Path const outFile = destDir / fileName;
              _studio->SaveAssetThumbnail(*asset, size, outFile, mochi::ErrorLog{});
            }
          }
        }
      }
      ImGui::EndMenu();
    }
    if (ImGui::MenuItem("Refresh")) {
      _needsRefresh = true;
    }
    ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
    if (ImGui::BeginMenu("Stats")) {
      int totalFolders = 0;
      int totalAssets = 0;
      int loadedAssets = 0;
      auto countNode = [&](auto& self, DirectoryNode const& node) -> void {
        totalFolders += static_cast<int>(node.children.size());
        for (auto const& file : node.files) {
          ++totalAssets;
          if (_assetManager->FindAssetByPath(file.fullPath)) {
            ++loadedAssets;
          }
        }
        for (auto const& child : node.children) {
          self(self, child);
        }
      };
      for (auto const& rootNode : _rootNodes) {
        countNode(countNode, rootNode);
      }
      ImGui::Text("Total Folders: %d", totalFolders);
      ImGui::Text("Total Assets: %d", totalAssets);
      ImGui::Text("Total Loaded Assets: %d", loadedAssets);
      ImGui::EndMenu();
    }
    ImGui::EndPopup();
  }
}

void AssetBrowser::ShowItemContextMenuContents(TileItem const& item, Asset const* asset) {
  // Compute selection summary flags
  int selectedCount = static_cast<int>(_selectedPaths.size());
  int folderCount = 0;
  int loadedAssetCount = 0;
  int unloadedFileCount = 0;
  int mochiBotCount = 0;
  bool allSavable = true;
  bool allUnloadable = true;
  // Common type across all selected loaded assets (for "Replace With..."). Stays valid only while
  // every selected item is a loaded asset sharing one type.
  AssetType commonLoadedType = AssetType::Unknown;
  bool allLoadedSameType = true;
  for (auto const& selPath : _selectedPaths) {
    std::error_code ec;
    if (std::filesystem::is_directory(selPath.AsFilesystemPath(), ec)) {
      ++folderCount;
    } else if (auto* selAsset = _assetManager->FindAssetByPath(selPath)) {
      if (loadedAssetCount == 0) {
        commonLoadedType = selAsset->GetType();
      } else if (selAsset->GetType() != commonLoadedType) {
        allLoadedSameType = false;
      }
      ++loadedAssetCount;
      if (dynamic_cast<BotAsset*>(selAsset)) {
        ++mochiBotCount;
      }
      if (!selAsset->IsSavable() || !selAsset->IsDirty()) {
        allSavable = false;
      }
      if (selAsset->GetReferenceCount() != 0) {
        allUnloadable = false;
      }
    } else if (ClassifyAssetTypeByFilename(selPath.GetFilename()) != AssetType::Unknown) {
      ++unloadedFileCount;
    }
  }
  // Check whether any selected file (or file inside a selected folder) is
  // referenced by another asset. Used to disable Unload/Delete.
  bool anyReferenced = false;
  auto isReferencedByOther = [&](mochi::Path const& filePath) {
    auto refs = _assetManager->GetReferencersToPath(filePath);
    // Only exclude assets that are themselves being deleted; their references go
    // away with them. Editors are NOT excluded — deleting an editors's open file does
    // not close the editor, so the editor remains a legitimate blocker.
    std::erase_if(refs, [&](IAssetReferencer* a) {
      if (auto* asAsset = dynamic_cast<Asset*>(a)) {
        return asAsset && _selectedPaths.contains(asAsset->GetPath());
      }
      return false;
    });
    return !refs.empty();
  };
  for (auto const& selPath : _selectedPaths) {
    if (anyReferenced) {
      break;
    }
    std::error_code ec;
    if (std::filesystem::is_directory(selPath.AsFilesystemPath(), ec)) {
      for (auto const& filePath : CollectFilePaths(selPath)) {
        if (isReferencedByOther(filePath)) {
          anyReferenced = true;
          break;
        }
      }
    } else if (isReferencedByOther(selPath)) {
      anyReferenced = true;
    }
  }
  // TODO: Abstract behind registered asset context menu actions.
  bool showedAssetActions = false;
  if ((selectedCount == 1) && (mochiBotCount == 1) &&
      (dynamic_cast<BotAsset const*>(asset) != nullptr)) {
    if (ImGui::IconSelectable("Create Mod Bot...", ICON_FA_PLUS)) {
      CreateModBotAsset(item.fullPath.GetParentPath(), item.fullPath.ToString());
    }
    showedAssetActions = true;
  }
  if (showedAssetActions) {
    ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
  }

  // Load: files only, at least one unloaded file
  bool showedGenericAssetActions = false;
  if (folderCount == 0 && unloadedFileCount > 0) {
    if (ImGui::IconSelectable("Load", ICON_FA_DOWNLOAD)) {
      for (auto const& selPath : _selectedPaths) {
        if (!_assetManager->FindAssetByPath(selPath)) {
          _assetManager->LoadAsset(selPath);
        }
      }
    }
    showedGenericAssetActions = true;
  }
  // Save: files only, at least one loaded asset
  if (folderCount == 0 && loadedAssetCount > 0) {
    ImGui::BeginDisabled(!allSavable);
    if (ImGui::IconSelectable("Save", ICON_FA_SAVE)) {
      for (auto const& selPath : _selectedPaths) {
        if (auto* selAsset = _assetManager->FindAssetByPath(selPath)) {
          if (selAsset->Save()) {
            selAsset->SetDirty(false);
          }
        }
      }
    }
    ImGui::EndDisabled();
    showedGenericAssetActions = true;
  }
  // Unload: files only, at least one loaded asset
  if (folderCount == 0 && loadedAssetCount > 0) {
    ImGui::BeginDisabled(!allUnloadable || anyReferenced);
    if (ImGui::IconSelectable("Unload", ICON_FA_EJECT)) {
      _pendingAssetAction = PendingAssetAction::Unload;
      _pendingAssetActionPaths = _selectedPaths;
    }
    if (anyReferenced && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      ImGui::SetTooltip("Cannot unload: still referenced by another asset.");
    }
    if (ImGui::IconSelectable("Reload", ICON_FA_SYNC)) {
      _pendingAssetAction = PendingAssetAction::Reload;
      _pendingAssetActionPaths = _selectedPaths;
    }
    if (anyReferenced && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      ImGui::SetTooltip("Cannot reload: still referenced by another asset.");
    }
    ImGui::EndDisabled();
    showedGenericAssetActions = true;
  }
  if (showedGenericAssetActions) {
    ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
  }

  // Folder-specific: paste into this folder (single folder only)
  if (selectedCount == 1 && folderCount == 1) {
    TryShowPasteSelectable(item.fullPath);
  }
  // Rename: single item only
  if (selectedCount == 1) {
    if (ImGui::IconSelectable("Rename", ICON_FA_PENCIL_ALT)) {
      _renamingPath = item.fullPath;
      if (item.isFolder) {
        _tileRename.buffer = item.name;
        _tileRename.onFinished = [this](std::string const& newName) {
          RenameFolder(_renamingPath, newName);
          _renamingPath = {};
        };
      } else {
        _tileRename.buffer = item.file->stem;
        _tileRename.onFinished = [this](std::string const& newName) {
          RenameAsset(_renamingPath, newName);
          _renamingPath = {};
        };
      }
      _tileRename.onCanceled = [this]() { _renamingPath = {}; };
    }
    // Duplicate: single item only
    if (ImGui::IconSelectable("Duplicate", ICON_FA_CLONE)) {
      DuplicateItem(item);
    }
  }
  // Batch Rename: multiple items only
  if (selectedCount > 1) {
    if (ImGui::IconSelectable("Batch Rename...", ICON_FA_EDIT)) {
      _batchRenameEntries.clear();
      for (auto const& selPath : _selectedPaths) {
        BatchRenameEntry entry;
        entry.path = selPath;
        std::error_code ec;
        entry.isFolder = std::filesystem::is_directory(selPath.AsFilesystemPath(), ec);
        if (entry.isFolder) {
          entry.originalStem = selPath.GetFilename();
          entry.extension = {};
        } else {
          auto const filename = selPath.GetFilename();
          entry.originalStem = GetAssetNameFromPath(selPath);
          entry.extension = filename.substr(entry.originalStem.size());
        }
        _batchRenameEntries.push_back(std::move(entry));
      }
      std::ranges::sort(_batchRenameEntries, [](auto const& a, auto const& b) {
        return a.originalStem < b.originalStem;
      });
      _batchRenameInputs = {};
      _openBatchRenamePopup = true;
    }
  }
  // Replace With: all selected items are loaded assets sharing a single type (single or multi).
  // Repoints every reference to the selected asset(s) at another loaded asset of the same type.
  if (selectedCount >= 1 && folderCount == 0 && loadedAssetCount == selectedCount &&
      allLoadedSameType && commonLoadedType != AssetType::Unknown) {
    if (ImGui::IconSelectable("Replace With...", ICON_FA_EXCHANGE_ALT)) {
      _replaceTargets = _selectedPaths;
      _replaceType = commonLoadedType;
      _replaceSelection = "";
      _openReplaceWithPopup = true;
    }
  }
  // Cut, Copy, Delete: always available
  if (ImGui::IconSelectable("Cut", ICON_FA_CUT)) {
    _clipboardPaths = _selectedPaths;
    _clipboardMode = ClipboardMode::Cut;
  }
  if (ImGui::IconSelectable("Copy", ICON_FA_COPY)) {
    _clipboardPaths = _selectedPaths;
    _clipboardMode = ClipboardMode::Copy;
  }
  ImGui::BeginDisabled(anyReferenced);
  if (ImGui::IconSelectable("Delete", ICON_FA_TRASH)) {
    _pendingAssetAction = PendingAssetAction::Delete;
    _pendingAssetActionPaths = _selectedPaths;
  }
  if (anyReferenced && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip("Cannot delete: still referenced by another asset.");
  }
  ImGui::EndDisabled();
  // Copy Paths: always available
  ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
  char copyFilenameLabel[32];
  snprintf(
      copyFilenameLabel,
      sizeof(copyFilenameLabel),
      selectedCount <= 1 ? "Copy Filename" : "Copy %d Filenames",
      selectedCount);
  if (ImGui::IconSelectable(copyFilenameLabel, ICON_FA_CLIPBOARD)) {
    std::string joined;
    for (auto const& selPath : _selectedPaths) {
      if (!joined.empty()) {
        joined += '\n';
      }
      joined += selPath.GetFilename();
    }
    ImGui::SetClipboardText(joined.c_str());
  }
  char copyPathLabel[32];
  snprintf(
      copyPathLabel,
      sizeof(copyPathLabel),
      selectedCount <= 1 ? "Copy Path" : "Copy %d Paths",
      selectedCount);
  if (ImGui::IconSelectable(copyPathLabel, ICON_FA_CLIPBOARD)) {
    std::string joined;
    for (auto const& selPath : _selectedPaths) {
      if (!joined.empty()) {
        joined += '\n';
      }
      joined += selPath.ToString();
    }
    ImGui::SetClipboardText(joined.c_str());
  }
  // Save Thumbnail: single loaded asset that renders a thumbnail. Each size opens a Save dialog
  // defaulting to <asset name>.png in the asset's own folder, then writes a transparent PNG.
  if (selectedCount == 1 && asset && asset->RendersThumbnail()) {
    if (ImGui::BeginMenu(ICON_FA_IMAGE "   Save Thumbnail")) {
      for (int const size : kThumbnailExportSizesPx) {
        char label[32];
        snprintf(label, sizeof(label), "%d x %d", size, size);
        if (ImGui::MenuItem(label)) {
          mochi::Path const defaultPath =
              item.fullPath.GetParentPath() / (asset->GetName() + ".png");
          char const* filters[] = {"*.png"};
          mochi::Path const chosen = SuperDexStudio::GetFileDialogPath(
              "Save Thumbnail",
              filters,
              1,
              "PNG Image (*.png)",
              /*isSaveDialog=*/true,
              defaultPath);
          // FindAssetByPath yields a mutable Asset (the menu's `asset` is const); the render path
          // needs a non-const reference.
          if (!chosen.IsEmpty()) {
            if (Asset* saveable = _assetManager->FindAssetByPath(item.fullPath)) {
              _studio->SaveAssetThumbnail(*saveable, size, chosen, mochi::ErrorLog{});
            }
          }
        }
      }
      ImGui::EndMenu();
    }
  }
  // Open in File Browser: single item only
  if (selectedCount == 1) {
    ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
    if (ImGui::IconSelectable("Open in File Browser", ICON_FA_FOLDER_OPEN)) {
      auto dir = (item.isFolder ? item.fullPath : item.fullPath.GetParentPath()).ToString();
      auto& pio = ImGui::GetPlatformIO();
      if (pio.Platform_OpenInShellFn) {
        pio.Platform_OpenInShellFn(ImGui::GetCurrentContext(), dir.c_str());
      }
    }
  }
}

DirectoryNode const* AssetBrowser::FindNode(mochi::Path const& path) const {
  for (auto const& rootNode : _rootNodes) {
    if (auto const* found = FindNodeRecursive(rootNode, path)) {
      return found;
    }
  }
  return nullptr;
}

mochi::Path const* AssetBrowser::FindOwningRoot(mochi::Path const& path) const {
  for (auto const& root : _rootPaths) {
    if (path.IsDescendantOf(root)) {
      return &root;
    }
  }
  return nullptr;
}

bool AssetBrowser::MoveItems(std::set<mochi::Path> const& sources, mochi::Path const& destDir) {
  bool anyMoved = false;
  std::error_code ec;
  std::vector<std::pair<mochi::Path, mochi::Path>> refMoves;
  for (auto const& source : sources) {
    auto dest = destDir / source.GetFilename();
    if (std::filesystem::exists(dest.AsFilesystemPath(), ec)) {
      continue;
    }
    bool const isDir = std::filesystem::is_directory(source.AsFilesystemPath(), ec);
    auto const filePaths = isDir ? CollectFilePaths(source) : std::vector<mochi::Path>{};
    std::filesystem::rename(source.AsFilesystemPath(), dest.AsFilesystemPath(), ec);
    if (ec) {
      continue;
    }
    if (isDir) {
      for (auto const& oldFilePath : filePaths) {
        auto rel = oldFilePath.RelativeToParent(source);
        refMoves.emplace_back(oldFilePath, dest / rel);
      }
    } else {
      refMoves.emplace_back(source, dest);
    }
    _selectedPaths.erase(source);
    _selectedPaths.insert(dest);
    anyMoved = true;
  }
  if (anyMoved) {
    _assetManager->RewriteAssetPaths(refMoves);
    _needsRefresh = true;
  }
  return anyMoved;
}

bool AssetBrowser::CopyItems(std::set<mochi::Path> const& sources, mochi::Path const& destDir) {
  bool anyCopied = false;
  std::error_code ec;
  for (auto const& source : sources) {
    auto dest = destDir / source.GetFilename();
    std::filesystem::copy(
        source.AsFilesystemPath(),
        dest.AsFilesystemPath(),
        std::filesystem::copy_options::recursive | std::filesystem::copy_options::skip_existing,
        ec);
    if (!ec) {
      anyCopied = true;
    }
  }
  if (anyCopied) {
    _needsRefresh = true;
  }
  return anyCopied;
}

bool AssetBrowser::DeleteItems(std::set<mochi::Path> paths) {
  if (paths.empty()) {
    return false;
  }
  auto& assetManager = _studio->GetAssetManager();
  // Pre-flight: collect any files that are still referenced. Hard-block deletion
  // if any referenced files would be removed.
  std::error_code ec;
  std::vector<DeleteBlocker> blockers;
  auto checkFile = [&](mochi::Path const& filePath) {
    auto refs = assetManager.GetReferencersToPath(filePath);
    // Only exclude assets that are themselves in the delete set; their refs
    // disappear with them. Tabs remain blockers.
    std::erase_if(refs, [&](IAssetReferencer* a) {
      if (auto* asAsset = dynamic_cast<Asset*>(a)) {
        return asAsset && paths.contains(asAsset->GetPath());
      }
      return false;
    });
    if (!refs.empty()) {
      DeleteBlocker blocker;
      blocker.file = filePath;
      blocker.referencers.reserve(refs.size());
      for (IAssetReferencer* a : refs) {
        blocker.referencers.push_back(a);
      }
      blockers.push_back(std::move(blocker));
    }
  };
  for (auto const& path : paths) {
    if (std::filesystem::is_directory(path.AsFilesystemPath(), ec)) {
      for (auto const& filePath : CollectFilePaths(path)) {
        checkFile(filePath);
      }
    } else {
      checkFile(path);
    }
  }
  if (!blockers.empty()) {
    _deleteBlockers = std::move(blockers);
    _openDeleteBlockedPopup = true;
    return false;
  }

  // Unload all loaded assets across the selection in dependency order (referencers
  // before referencees) so a referenced model is not left stale in memory when its
  // referencing bot is deleted in the same operation. A per-path unload would bail
  // on a still-referenced asset whenever a dependency sorts ahead of its dependent.
  std::set<mochi::Path> assetPaths;
  for (auto const& path : paths) {
    if (std::filesystem::is_directory(path.AsFilesystemPath(), ec)) {
      for (auto const& filePath : CollectFilePaths(path)) {
        if (_assetManager->FindAssetByPath(filePath)) {
          assetPaths.insert(filePath);
        }
      }
    } else if (_assetManager->FindAssetByPath(path)) {
      assetPaths.insert(path);
    }
  }
  _assetManager->UnloadAssets(assetPaths);

  for (auto const& path : paths) {
    if (std::filesystem::is_directory(path.AsFilesystemPath(), ec)) {
      // Check if we're viewing this folder or a subfolder BEFORE deleting
      bool const isViewing =
          !_currentPath.IsEmpty() && !path.IsEmpty() && _currentPath.IsDescendantOf(path);
      std::filesystem::remove_all(path.AsFilesystemPath(), ec);
      if (isViewing) {
        SetCurrentPath(path.GetParentPath());
      }
    } else {
      std::filesystem::remove(path.AsFilesystemPath(), ec);
    }
  }
  _needsRefresh = true;
  return true;
}

bool AssetBrowser::CreateFolder(mochi::Path const& parentDir, std::string const& name) {
  if (name.empty()) {
    return false;
  }
  auto folderPath = parentDir / name;
  std::error_code ec;
  if (std::filesystem::exists(folderPath.AsFilesystemPath(), ec)) {
    return false;
  }
  bool const created = std::filesystem::create_directory(folderPath.AsFilesystemPath(), ec);
  if (created) {
    _needsRefresh = true;
  }
  return created;
}

bool AssetBrowser::RenameFolder(mochi::Path const& folderPath, std::string const& newName) {
  if (newName.empty()) {
    MOCHI_LOG_ERROR(
        "RenameFolder: new name is empty (folder='%s').", folderPath.ToString().c_str());
    return false;
  }
  auto const newPath = folderPath.GetParentPath() / newName;
  if (newPath == folderPath) {
    // Name unchanged (e.g. accepting the default name) - nothing to do.
    return true;
  }
  std::error_code ec;
  if (std::filesystem::exists(newPath.AsFilesystemPath(), ec)) {
    MOCHI_LOG_ERROR(
        "RenameFolder: target already exists ('%s' -> '%s').",
        folderPath.ToString().c_str(),
        newPath.ToString().c_str());
    return false;
  }
  // Check if current path is inside the folder being renamed (before the rename)
  bool updateCurrentPath = false;
  std::string relToFolder;
  if (!_currentPath.IsEmpty() && !folderPath.IsEmpty() && _currentPath.IsDescendantOf(folderPath)) {
    if (_currentPath == folderPath) {
      relToFolder = {};
    } else {
      relToFolder = _currentPath.RelativeToParent(folderPath);
    }
    updateCurrentPath = true;
  }
  auto const filePaths = CollectFilePaths(folderPath);
  std::filesystem::rename(folderPath.AsFilesystemPath(), newPath.AsFilesystemPath(), ec);
  if (ec) {
    MOCHI_LOG_ERROR(
        "RenameFolder: filesystem rename failed ('%s' -> '%s'): %s",
        folderPath.ToString().c_str(),
        newPath.ToString().c_str(),
        ec.message().c_str());
    return false;
  }
  std::vector<std::pair<mochi::Path, mochi::Path>> refMoves;
  refMoves.reserve(filePaths.size());
  for (auto const& oldFilePath : filePaths) {
    auto rel = oldFilePath.RelativeToParent(folderPath);
    refMoves.emplace_back(oldFilePath, newPath / rel);
  }
  _assetManager->RewriteAssetPaths(refMoves);
  if (updateCurrentPath) {
    SetCurrentPath(relToFolder.empty() ? newPath : newPath / relToFolder);
  }
  _needsRefresh = true;
  return true;
}

bool AssetBrowser::RenameAsset(mochi::Path const& assetPath, std::string const& newStem) {
  if (newStem.empty()) {
    MOCHI_LOG_ERROR("RenameAsset: new stem is empty (asset='%s').", assetPath.ToString().c_str());
    return false;
  }
  // Reconstruct full filename: newStem + original extension(s)
  auto const filename = assetPath.GetFilename();
  auto const oldStem = GetAssetNameFromPath(assetPath);
  auto const ext = filename.substr(oldStem.size());
  auto const newPath = assetPath.GetParentPath() / (newStem + ext);
  if (newPath == assetPath) {
    // Name unchanged (e.g. accepting the default name) - nothing to do.
    return true;
  }
  std::error_code ec;
  if (std::filesystem::exists(newPath.AsFilesystemPath(), ec)) {
    MOCHI_LOG_ERROR(
        "RenameAsset: target already exists ('%s' -> '%s').",
        assetPath.ToString().c_str(),
        newPath.ToString().c_str());
    return false;
  }
  std::filesystem::rename(assetPath.AsFilesystemPath(), newPath.AsFilesystemPath(), ec);
  if (ec) {
    MOCHI_LOG_ERROR(
        "RenameAsset: filesystem rename failed ('%s' -> '%s'): %s",
        assetPath.ToString().c_str(),
        newPath.ToString().c_str(),
        ec.message().c_str());
    return false;
  }
  _assetManager->RewriteAssetPath(assetPath, newPath);
  if (_selectedPaths.erase(assetPath)) {
    _selectedPaths.insert(newPath);
  }
  _needsRefresh = true;
  return true;
}

void AssetBrowser::TryShowPasteSelectable(mochi::Path const& destDir) {
  if (_clipboardMode == ClipboardMode::None) {
    return;
  }
  auto const n = _clipboardPaths.size();
  char pasteLabel[64];
  snprintf(
      pasteLabel,
      sizeof(pasteLabel),
      n <= 1 ? "Paste Item" : "Paste %d Items",
      static_cast<int>(n));
  if (ImGui::IconSelectable(pasteLabel, ICON_FA_PASTE)) {
    ExecutePaste(destDir);
  }
  ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
}

void AssetBrowser::TryAcceptAssetBrowserDrop(mochi::Path const& destDir, bool setCursorNotAllowed) {
  if (ImGui::BeginDragDropTarget()) {
    // Peek at the payload without drawing the highlight rect
    if (auto const* peek = ImGui::AcceptDragDropPayload(
            kAssetBrowserDragDropType, ImGuiDragDropFlags_AcceptPeekOnly)) {
      auto const* dragPayload = *static_cast<AssetBrowserDragDropPayload* const*>(peek->Data);
      // Validate: build the set of items that would actually move/copy
      std::set<mochi::Path> sources;
      if (dragPayload && !dragPayload->items.empty()) {
        for (auto const& item : dragPayload->items) {
          if (item.fullPath == destDir) {
            continue;
          }
          if (item.fullPath.GetParentPath() == destDir) {
            continue;
          }
          if (item.isFolder && destDir.IsDescendantOf(item.fullPath)) {
            // Cannot drop a folder into itself or its own descendants.
            continue;
          }
          sources.insert(item.fullPath);
        }
      }
      if (sources.empty()) {
        // Invalid drop — show not-allowed cursor, no highlight
        if (setCursorNotAllowed) {
          ImGui::SetMouseCursor(ImGuiMouseCursor_NotAllowed);
        }
      } else {
        // Valid drop — accept for real (draws highlight rect, delivers on release)
        if (ImGui::AcceptDragDropPayload(kAssetBrowserDragDropType) != nullptr) {
          bool const ctrl = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeySuper;
          if (ctrl) {
            CopyItems(sources, destDir);
          } else {
            MoveItems(sources, destDir);
          }
        }
      }
    }
    ImGui::EndDragDropTarget();
  }
}

void AssetBrowser::DuplicateItem(TileItem const& item) {
  auto dir = item.fullPath.GetParentPath();
  if (item.isFolder) {
    std::string dupName = MakeUniqueFileName(item.name, dir);
    auto dupPath = dir / dupName;
    std::error_code ec;
    std::filesystem::copy(
        item.fullPath.AsFilesystemPath(),
        dupPath.AsFilesystemPath(),
        std::filesystem::copy_options::recursive,
        ec);
    if (!ec) {
      BeginRenaming(dupPath, dupName, [this, dupName](std::string const& newName) {
        if (newName != dupName) {
          RenameFolder(_renamingPath, newName);
        }
      });
    }
  } else if (item.file) {
    auto const filename = item.fullPath.GetFilename();
    auto const ext = filename.substr(item.file->stem.size());
    std::string dupStem = MakeUniqueFileName(item.file->stem, dir, ext);
    auto dupPath = dir / (dupStem + ext);
    std::error_code ec;
    std::filesystem::copy_file(item.fullPath.AsFilesystemPath(), dupPath.AsFilesystemPath(), ec);
    if (!ec) {
      BeginRenaming(dupPath, dupStem, [this, dupStem](std::string const& newName) {
        if (newName != dupStem) {
          RenameAsset(_renamingPath, newName);
        }
      });
    }
  }
}

void AssetBrowser::CreateModBotAsset(mochi::Path const& dir, std::string const& base) {
  std::string const extension{superdex::robotics::kBotExtension};
  std::string const botName = MakeUniqueFileName("Mod Bot", dir, extension);
  auto const newPath = dir / (botName + extension);
  superdex::robotics::ModBotPrefab params;
  params.name = botName;
  params.base = base;
  superdex::robotics::SaveToFile(params, newPath.ToString(), mochi::ErrorLog{});
  BeginRenaming(newPath, botName, [this, extension](std::string const& newName) {
    RenameAsset(_renamingPath, newName);
    auto renamedPath = _renamingPath.GetParentPath() / (newName + extension);
    auto* bot = _assetManager->LoadBotAsset(renamedPath);
    if (bot) {
      bot->GetBotName() = newName;
      // The file was written with a placeholder name before the user picked one.
      bot->Save();
    }
  });
}

void AssetBrowser::ShowDeleteBlockedModal() {
  if (_openDeleteBlockedPopup) {
    ImGui::OpenPopup("Cannot Delete");
    _openDeleteBlockedPopup = false;
  }
  // Center on the main viewport every frame (handles window resize / dpi changes).
  ImVec2 const center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(600, 0), ImGuiCond_Appearing);
  if (ImGui::BeginPopupModal(
          "Cannot Delete",
          nullptr,
          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
              ImGuiWindowFlags_NoSavedSettings)) {
    bool closePopup = false;
    ImGui::TextWrapped(
        "Cannot delete the following asset(s) because they are still being referenced:");
    ImGui::Spacing();
    int id = 0;
    for (auto const& blocker : _deleteBlockers) {
      ImGui::PushID(id++);
      ImGui::Bullet();
      if (ImGui::TextButton(blocker.file.GetFilename().c_str())) {
        if (auto* a = _assetManager->FindAssetByPath(blocker.file)) {
          SelectAsset(a);
          _deleteBlockers.clear();
          ImGui::CloseCurrentPopup();
        }
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", blocker.file.ToString().c_str());
      }
      ImGui::Indent();
      for (auto const& referencer : blocker.referencers) {
        ImGui::PushID(id++);
        ImGui::Bullet();
        std::string tooltip = referencer->GetReferencerName();
        // specialization for editors
        if (auto* asEditor = dynamic_cast<AssetEditor*>(referencer)) {
          ImGui::TextDisabled("Open in Asset Editor:");
          ImGui::SameLine();
          if (ImGui::TextButton(asEditor->GetReferencerName().c_str())) {
            int const editorIdx = _studio->FindAssetEditorIndex(asEditor->GetAsset());
            if (editorIdx >= 0) {
              _studio->SelectAssetEditor(editorIdx);
              closePopup = true;
            }
          }
          tooltip = asEditor->GetAsset()->GetPath().ToString();
        }
        // specialization for assets
        else if (auto* asAsset = dynamic_cast<Asset*>(referencer)) {
          ImGui::TextDisabled("Referenced by Asset:");
          ImGui::SameLine();
          if (ImGui::TextButton(asAsset->GetPath().GetFilename().c_str())) {
            SelectAsset(asAsset);
            closePopup = true;
          }
          tooltip = asAsset->GetPath().ToString();
        }
        // all other types
        else {
          ImGui::TextDisabled("Referenced by:");
          ImGui::SameLine();
          ImGui::TextUnformatted(referencer->GetReferencerName().c_str());
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("%s", tooltip.c_str());
        }
        ImGui::PopID();
      }
      ImGui::Unindent();
      ImGui::Spacing();
      ImGui::PopID();
    }
    ImGui::Separator();
    ImGui::Spacing();
    if (ImGui::Button("OK", ImVec2(120, 0)) || closePopup) {
      _deleteBlockers.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void AssetBrowser::ShowBatchRenameModal() {
  if (_openBatchRenamePopup) {
    ImGui::OpenPopup("Batch Rename");
    _openBatchRenamePopup = false;
  }
  ImVec2 const center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(800, 0), ImGuiCond_Appearing);
  if (ImGui::BeginPopupModal(
          "Batch Rename",
          nullptr,
          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
              ImGuiWindowFlags_NoSavedSettings)) {
    int trimMax = 0;
    for (auto const& e : _batchRenameEntries) {
      trimMax = std::max(trimMax, static_cast<int>(e.originalStem.size()));
    }
    ImGui::BatchRenameInputs(_batchRenameInputs, trimMax);
    ImGui::Spacing();

    // Compute new stems and validate.
    std::vector<std::string> oldNames;
    std::vector<std::string> newStems;
    std::vector<bool> rowInvalid;
    oldNames.reserve(_batchRenameEntries.size());
    newStems.reserve(_batchRenameEntries.size());
    rowInvalid.reserve(_batchRenameEntries.size());
    int changedCount = 0;
    int invalidCount = 0;
    // Gather siblings outside this batch for collision detection.
    std::set<mochi::Path> batchPathSet;
    for (auto const& e : _batchRenameEntries) {
      batchPathSet.insert(e.path);
    }
    for (auto const& entry : _batchRenameEntries) {
      auto newStem = ComputeBatchRenamedName(entry.originalStem, _batchRenameInputs);
      bool invalid = false;
      if (newStem.empty()) {
        invalid = true;
      } else if (newStem != entry.originalStem) {
        ++changedCount;
        // Check collision with existing sibling not in batch. A case-only
        // rename on a case-insensitive filesystem resolves to entry.path
        // itself; std::filesystem::equivalent excludes that self-collision.
        auto const newFullPath = entry.path.GetParentPath() / (newStem + entry.extension);
        std::error_code ec;
        if (std::filesystem::exists(newFullPath.AsFilesystemPath(), ec) &&
            !batchPathSet.contains(newFullPath) &&
            !std::filesystem::equivalent(
                newFullPath.AsFilesystemPath(), entry.path.AsFilesystemPath(), ec)) {
          invalid = true;
        }
      }
      oldNames.push_back(entry.originalStem);
      newStems.push_back(std::move(newStem));
      rowInvalid.push_back(invalid);
    }
    // Within-batch collision detection: two entries producing the same target path.
    for (size_t i = 0; i < _batchRenameEntries.size(); ++i) {
      if (rowInvalid[i]) {
        continue;
      }
      auto const& ei = _batchRenameEntries[i];
      auto const pathI = ei.path.GetParentPath() / (newStems[i] + ei.extension);
      for (size_t j = i + 1; j < _batchRenameEntries.size(); ++j) {
        auto const& ej = _batchRenameEntries[j];
        auto const pathJ = ej.path.GetParentPath() / (newStems[j] + ej.extension);
        if (pathI == pathJ) {
          rowInvalid[i] = true;
          rowInvalid[j] = true;
        }
      }
    }
    invalidCount = static_cast<int>(std::count(rowInvalid.begin(), rowInvalid.end(), true));

    ImGui::BatchRenamePreviewTable(oldNames, newStems, rowInvalid);

    ImGui::Separator();
    bool const canApply = invalidCount == 0 && changedCount > 0;
    ImGui::BeginDisabled(!canApply);
    if (ImGui::Button("Apply", ImVec2(120, 0))) {
      // Two-phase rename to safely handle case-only changes on
      // case-insensitive filesystems, chains (A->B, B->C), and swaps
      // (A->B, B->A): first move each source to a unique temp name, then
      // from the temp name to the final name. This guarantees no target
      // exists at the moment of the final rename.
      struct PendingRename {
        mochi::Path tempPath;
        std::string finalStem;
        bool isFolder;
      };
      std::vector<PendingRename> pending;
      pending.reserve(_batchRenameEntries.size());
      bool allOk = true;
      for (size_t i = 0; i < _batchRenameEntries.size(); ++i) {
        auto const& entry = _batchRenameEntries[i];
        if (newStems[i] == entry.originalStem) {
          continue;
        }
        auto const tempStem = "__mochi_batch_rename_tmp_" + std::to_string(i);
        bool const renamedToTemp =
            entry.isFolder ? RenameFolder(entry.path, tempStem) : RenameAsset(entry.path, tempStem);
        if (!renamedToTemp) {
          allOk = false;
          continue;
        }
        auto const tempPath = entry.isFolder
            ? entry.path.GetParentPath() / tempStem
            : entry.path.GetParentPath() / (tempStem + entry.extension);
        pending.push_back({tempPath, newStems[i], entry.isFolder});
      }
      for (auto const& p : pending) {
        bool const renamedToFinal = p.isFolder ? RenameFolder(p.tempPath, p.finalStem)
                                               : RenameAsset(p.tempPath, p.finalStem);
        if (!renamedToFinal) {
          allOk = false;
        }
      }
      if (!allOk) {
        MOCHI_LOG_WARNING(
            "Batch rename: one or more rename operations failed; "
            "some files may remain at temporary names. See preceding errors for details.");
      }
      _batchRenameEntries.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      _batchRenameEntries.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void AssetBrowser::ShowReplaceWithModal() {
  if (_openReplaceWithPopup) {
    ImGui::OpenPopup("Replace With");
    _openReplaceWithPopup = false;
  }
  ImVec2 const center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(500, 0), ImGuiCond_Appearing);
  if (ImGui::BeginPopupModal(
          "Replace With",
          nullptr,
          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
              ImGuiWindowFlags_NoSavedSettings)) {
    int const targetCount = static_cast<int>(_replaceTargets.size());
    char const* const typeLabel = GetAssetTypeLabel(_replaceType);
    if (targetCount == 1) {
      ImGui::TextWrapped(
          "Replace all references to \"%s\" (%s) with:",
          _replaceTargets.begin()->GetFilename().c_str(),
          typeLabel);
    } else {
      ImGui::TextWrapped(
          "Replace all references to %d selected %s assets with:", targetCount, typeLabel);
    }
    ImGui::Spacing();

    // Replacement picker: same thumbnail-tile asset slot the Bot/Prefab editors use for render and
    // Mochi models (ImGui::AssetSlot), so the selected asset is shown as a 64px thumbnail tile.
    ImGui::AssetSlot(
        "Replacement",
        _replaceSelection,
        *_assetManager,
        _studio,
        _replaceType,
        /*acceptDragDropPayload=*/true);
    Asset* const replacement =
        _replaceSelection.empty() ? nullptr : _assetManager->FindAssetByPath(_replaceSelection);
    // A target can't replace itself; the slot lists all assets of the type, so guard against it.
    bool const replacementIsTarget =
        replacement != nullptr && _replaceTargets.contains(replacement->GetPath());
    if (replacementIsTarget) {
      ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(230, 100, 100, 255));
      ImGui::TextWrapped("Pick an asset that is not among the ones being replaced.");
      ImGui::PopStyleColor();
    }

    // Count the distinct referencers that will be repointed (excluding the targets themselves,
    // whose references disappear if they are also deleted).
    std::set<IAssetReferencer*> affected;
    for (auto const& target : _replaceTargets) {
      for (IAssetReferencer* r : _assetManager->GetReferencersToPath(target)) {
        if (auto* asAsset = dynamic_cast<Asset*>(r);
            asAsset && _replaceTargets.contains(asAsset->GetPath())) {
          continue;
        }
        affected.insert(r);
      }
    }
    ImGui::Spacing();
    if (affected.empty()) {
      ImGui::TextDisabled("No other assets currently reference the selection.");
    } else {
      ImGui::TextWrapped(
          "%d referencing asset(s) will be updated:", static_cast<int>(affected.size()));
      ImGui::Spacing();
      int id = 0;
      for (IAssetReferencer* referencer : affected) {
        ImGui::PushID(id++);
        ImGui::Bullet();
        std::string tooltip = referencer->GetReferencerName();
        // Mirror the "Cannot Delete" modal: a descriptor prefix plus the referencer's name, with
        // the full path on hover. Read-only here so a click doesn't navigate mid-operation.
        if (auto* asEditor = dynamic_cast<AssetEditor*>(referencer)) {
          ImGui::TextDisabled("Open in Asset Editor:");
          ImGui::SameLine();
          ImGui::TextUnformatted(asEditor->GetReferencerName().c_str());
          tooltip = asEditor->GetAsset()->GetPath().ToString();
        } else if (auto* asAsset = dynamic_cast<Asset*>(referencer)) {
          ImGui::TextDisabled("Referenced by Asset:");
          ImGui::SameLine();
          ImGui::TextUnformatted(asAsset->GetPath().GetFilename().c_str());
          tooltip = asAsset->GetPath().ToString();
        } else {
          ImGui::TextDisabled("Referenced by:");
          ImGui::SameLine();
          ImGui::TextUnformatted(referencer->GetReferencerName().c_str());
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("%s", tooltip.c_str());
        }
        ImGui::PopID();
      }
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    mochi::Path const replacementPath = replacement ? replacement->GetPath() : mochi::Path{};
    auto repointAll = [&]() {
      for (auto const& target : _replaceTargets) {
        _assetManager->RepointReferences(target, replacementPath);
      }
    };

    bool const canReplace = replacement != nullptr && !replacementIsTarget;
    ImGui::BeginDisabled(!canReplace);
    if (ImGui::Button("Replace", ImVec2(140, 0))) {
      repointAll();
      _replaceTargets.clear();
      _replaceSelection = "";
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Replace and Delete", ImVec2(160, 0))) {
      // Repoint first so the targets are no longer referenced, then delete them. DeleteItems'
      // reference pre-flight will pass because every referencer now points at the replacement.
      repointAll();
      DeleteItems(_replaceTargets);
      ClearSelection();
      _replaceTargets.clear();
      _replaceSelection = "";
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      _replaceTargets.clear();
      _replaceSelection = "";
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void AssetBrowser::TryApplyPendingAssetAction() {
  if (_pendingAssetAction == PendingAssetAction::None) {
    return;
  }
  auto const action = std::exchange(_pendingAssetAction, PendingAssetAction::None);
  auto const paths = std::exchange(_pendingAssetActionPaths, {});
  switch (action) {
    case PendingAssetAction::None:
      break;
    case PendingAssetAction::Unload:
      for (auto const& path : paths) {
        if (_assetManager->FindAssetByPath(path)) {
          _assetManager->UnloadAssetByPath(path);
        }
      }
      break;
    case PendingAssetAction::Reload:
      for (auto const& path : paths) {
        if (_assetManager->FindAssetByPath(path)) {
          _assetManager->UnloadAssetByPath(path);
          _assetManager->LoadAsset(path);
        }
      }
      break;
    case PendingAssetAction::Delete:
      DeleteItems(paths);
      ClearSelection();
      break;
  }
}

void AssetBrowser::ShowWindow(bool* open) {
  TryApplyPendingAssetAction();
  TryRefreshDirectoryTree();

  ImGui::Begin("Asset Browser", open);

  // Mouse back/forward buttons (buttons 4/5, ImGui indices 3/4) navigate the folder history like a
  // web browser, while the Asset Browser or one of its panels is hovered.
  if (ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows)) {
    if (ImGui::IsMouseClicked(3)) {
      GoBack();
    } else if (ImGui::IsMouseClicked(4)) {
      GoForward();
    }
  }

  // Folder Tree
  if (_studio->GetAppSettings().assetBrowser.showDirectoryTree) {
    ImGui::BeginChild(
        "##FolderTree", ImVec2(kSidebarWidth, 0), ImGuiChildFlags_Border | ImGuiChildFlags_ResizeX);
    if (!_rootPaths.empty()) {
      ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, kFolderTreeIndent);
      ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
      for (auto const& rootNode : _rootNodes) {
        ShowFolderTree(rootNode, true);
      }
      ImGui::PopStyleVar();
      ImGui::PopStyleColor();
      _expandTreeToCurrentPath = false;
    } else {
      ImGui::TextDisabled("No folders in workspace.");
    }
    ImGui::EndChild();
    ImGui::SameLine();
  }

  // Filter
  if (_studio->GetAppSettings().assetBrowser.showFilters) {
    ShowFilter(kFilterWidth);
    ImGui::SameLine();
  }

  // Right Group
  ImGui::BeginGroup();

  // Breadcrumb bar
  ShowBreadcrumb();

  // Settings Cog
  ShowSettingsCog();

  bool const filtering = IsFilterActive();

  // Content Area
  ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32_BLACK_TRANS);
  // AlwaysVerticalScrollbar keeps the usable width constant whether or not the
  // content overflows, preventing a layout feedback loop with the tile grid.
  ImGui::BeginChild(
      "##AssetContent",
      ImVec2(0, 0),
      ImGuiChildFlags_None,
      ImGuiWindowFlags_AlwaysVerticalScrollbar);

  auto const* currentNode = FindNode(_currentPath);
  bool assetClicked = false;
  Asset* assetToOpen = nullptr;

  auto const& style = ImGui::GetStyle();

  // Build unified tile list
  _tileItems.clear();
  if (currentNode) {
    if (filtering) {
      // When filtering, only show files (recursively)
      CollectFilesRecursive(*currentNode, _tileItems);
    } else {
      // Folders first, then files
      for (auto const& child : currentNode->children) {
        _tileItems.push_back({child.fullPath, child.name, true, nullptr, &child});
      }
      for (auto const& file : currentNode->files) {
        _tileItems.push_back({file.fullPath, file.name, false, &file, nullptr});
      }
    }
  }

  // Build visible indices (folders always pass when not filtering; files filtered by type/name)
  _visibleIndices.clear();
  bool const showUnknownFiles = _studio->GetAppSettings().assetBrowser.showUnknownFiles;
  bool const showImportableFiles = _studio->GetAppSettings().assetBrowser.showImportableFiles;
  int const numTypeFilters = showUnknownFiles ? kAssetTypeCount : kAssetTypeCount - 1;
  bool const anyTypeFilter = filtering &&
      std::any_of(_assetTypeFilters, _assetTypeFilters + numTypeFilters, [](bool v) { return v; });
  bool const importableFilterActive = filtering && _importableFilter && showImportableFiles;
  bool const anyCategoryFilter = anyTypeFilter || importableFilterActive;
  for (int idx = 0; idx < static_cast<int>(_tileItems.size()); ++idx) {
    auto const& item = _tileItems[idx];
    if (item.isFolder) {
      // Folders only shown when not filtering (they were excluded above already)
      _visibleIndices.push_back(idx);
      continue;
    }
    // Selected items always pass the filter
    bool const isSelected = _selectedPaths.contains(item.fullPath);
    // Hide unknown (non-importable) files unless explicitly enabled
    if (!isSelected && !showUnknownFiles && item.file->type == AssetType::Unknown &&
        item.file->importer == nullptr) {
      continue;
    }
    // Hide importable files unless explicitly enabled
    if (!isSelected && !showImportableFiles && item.file->importer != nullptr) {
      continue;
    }
    // Apply category filter (asset type and/or importable)
    if (!isSelected && anyCategoryFilter) {
      // Importable files are categorized as Importable, not by their classified
      // type, so they only match the Importable filter (e.g. a URDF classified as
      // Unknown must not appear under the Unknown filter).
      bool const passesType = anyTypeFilter && item.file->importer == nullptr &&
          _assetTypeFilters[static_cast<int>(item.file->type)];
      bool const passesImportable = importableFilterActive && item.file->importer != nullptr;
      if (!passesType && !passesImportable) {
        continue;
      }
    }
    // Apply name filter (matches full filename, including extension)
    if (!isSelected && _assetNameFilter.IsActive() &&
        !_assetNameFilter.PassFilter(item.file->name.c_str())) {
      continue;
    }
    _visibleIndices.push_back(idx);
  }

  // Sort visible items by type (folders first, then by AssetType), preserving alphabetical
  // order within each group.
  if (_studio->GetAppSettings().assetBrowser.sortByType) {
    std::ranges::stable_sort(_visibleIndices, [this](int a, int b) {
      auto const& ia = _tileItems[a];
      auto const& ib = _tileItems[b];
      if (ia.isFolder != ib.isFolder) {
        return ia.isFolder;
      }
      if (ia.isFolder) {
        return false;
      }
      return static_cast<int>(ia.file->type) < static_cast<int>(ib.file->type);
    });
  }

  // Pick the largest column count whose tiles (at min size) still fit, then grow
  // each tile to fill the row, clamped to [minTile, maxTile]. If we don't have
  // enough visible tiles to fill a row, render at minTile to avoid jitter.
  float const fbScale = ImGui::GetIO().DisplayFramebufferScale.x;
  float const dpiScale = (fbScale > 1.0f ? 1.0f : _studio->GetDpiScale());
  constexpr float minTileSize = 100;
  constexpr float maxTileSize = 120;
  float const minTile = minTileSize * dpiScale;
  float const maxTile = maxTileSize * dpiScale;
  float const spacing = style.ItemSpacing.x;
  float const availWidth = ImGui::GetContentRegionAvail().x - spacing;
  int const maxColumns =
      std::max(1, static_cast<int>((availWidth + spacing) / (minTile + spacing)));
  int const numColumns =
      std::min(maxColumns, std::max(1, static_cast<int>(_visibleIndices.size())));
  float const tileSize = (static_cast<int>(_visibleIndices.size()) < maxColumns)
      ? minTile
      : std::clamp((availWidth - (numColumns - 1) * spacing) / numColumns, minTile, maxTile);

  // Unified tile loop
  for (int vi = 0; vi < static_cast<int>(_visibleIndices.size()); ++vi) {
    int const itemIdx = _visibleIndices[vi];
    auto const& item = _tileItems[itemIdx];

    // Skip items that were moved/deleted this frame (avoids one-frame stale render)
    if (_needsRefresh) {
      std::error_code ec;
      if (!std::filesystem::exists(item.fullPath.AsFilesystemPath(), ec)) {
        continue;
      }
    }

    bool const selected = _selectedPaths.contains(item.fullPath);

    Asset* asset = nullptr;
    if (!item.isFolder) {
      asset = _assetManager->FindAssetByPath(item.fullPath);
    }

    ImGui::PushID(item.fullPath.ToString().c_str());

    bool const isRenaming = (_renamingPath == item.fullPath);

    // Validate rename buffer against existing items in the directory
    if (isRenaming) {
      _tileRename.valid = true;
      mochi::Path targetPath;
      if (item.isFolder) {
        targetPath = _renamingPath.GetParentPath() / _tileRename.buffer;
      } else {
        auto filename = _renamingPath.GetFilename();
        auto ext = filename.substr(item.file->stem.size());
        targetPath = _renamingPath.GetParentPath() / (_tileRename.buffer + ext);
      }
      if (targetPath != _renamingPath) {
        for (auto const& other : _tileItems) {
          if (other.fullPath == targetPath) {
            _tileRename.valid = false;
            break;
          }
        }
      }
    }

    bool const isCut =
        _clipboardMode == ClipboardMode::Cut && _clipboardPaths.contains(item.fullPath);

    if (isCut) {
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
    }
    // Render Tile
    if (item.isFolder) {
      ImGui::FolderTile(
          item.name.c_str(),
          tileSize,
          _studio->GetFont("Folder Tile Icons"),
          item.name.c_str(),
          _studio->GetFont("Roboto Regular Small"),
          IM_COL32(230, 190, 100, 255),
          selected,
          isRenaming ? &_tileRename : nullptr);
    } else if (asset) {
      if (_forceAssetFocus && selected) {
        ImGui::SetScrollHereY(0.5f);
        _forceAssetFocus = false;
      }
      bool const isBotInvalid =
          asset->GetType() == AssetType::Bot && !static_cast<BotAsset*>(asset)->IsValidateOk();
      ImGui::AssetTile(
          asset->GetName().c_str(),
          asset->GetThumbnailImage(),
          tileSize,
          asset->GetName().c_str(),
          asset->GetTypeLabel(),
          _studio->GetFont("Roboto Regular Small"),
          asset->GetColor(),
          asset->IsReadOnly()    ? ImGui::AssetTileState_ReadOnly
              : asset->IsDirty() ? ImGui::AssetTileState_Unsaved
                                 : ImGui::AssetTileState_None,
          selected,
          isBotInvalid ? ICON_FA_EXCLAMATION_TRIANGLE : nullptr,
          isBotInvalid ? IM_COL32(255, 255, 0, 255) : IM_COL32_BLACK_TRANS,
          ImGui::kDefaultRenderTargetUV0,
          ImGui::kDefaultRenderTargetUV1,
          10,
          isRenaming ? &_tileRename : nullptr);
    } else {
      ImGui::SimpleAssetTile(
          item.file->name.c_str(),
          tileSize,
          ICON_FA_FILE,
          _studio->GetFont("Asset Tile Icons"),
          item.file->stem.c_str(),
          item.file->importer ? item.file->importer->GetDisplayName().c_str()
              : item.file->type == AssetType::Unknown ? item.fullPath.GetExtension().c_str()
                                                      : GetAssetTypeLabel(item.file->type),
          _studio->GetFont("Roboto Regular Small"),
          item.file->importer ? item.file->importer->GetColor()
                              : GetAssetTypeColor(item.file->type),
          selected,
          isRenaming ? &_tileRename : nullptr);
    }
    if (isCut) {
      ImGui::PopStyleVar();
    }

    // Item Context Menu
    if (ImGui::BeginPopupContextItem("##TilePopup", ImGuiPopupFlags_MouseButtonRight)) {
      if (ImGui::IsWindowAppearing() && !selected) {
        _selectedPaths.clear();
        _selectedPaths.insert(item.fullPath);
        _lastClickedIndex = vi;
      }
      ShowItemContextMenuContents(item, asset);
      ImGui::EndPopup();
    }

    // Item Tooltip
    if (ImGui::BeginItemTooltip()) {
      ImGui::Text("Path: %s", item.fullPath.ToString().c_str());
      if (!item.isFolder) {
        ImGui::Text("References: %d", _assetManager->GetPathReferenceCount(item.fullPath));
      }
      if (!item.isFolder && asset) {
        if (asset->GetPath() != item.fullPath) {
          ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 0, 255));
          ImGui::Text("Asset Path: %s", asset->GetPath().ToString().c_str());
          ImGui::PopStyleColor();
        }
        asset->ShowAssetTileTooltipItems();
      } else if (!item.isFolder) {
        ImGui::TextDisabled("(double-click to load)");
      }
      ImGui::EndTooltip();
    }

    // Drag-drop
    if (ImGui::BeginDragDropSource()) {
      // Build payload items: if dragged item is part of a multi-selection, include all
      bool const isMultiDrag = selected && _selectedPaths.size() > 1;
      _dragPayload.items.clear();
      if (isMultiDrag) {
        for (auto const& selPath : _selectedPaths) {
          AssetBrowserDragDropItem ddi;
          ddi.fullPath = selPath;
          std::error_code ec;
          ddi.isFolder = std::filesystem::is_directory(selPath.AsFilesystemPath(), ec);
          ddi.asset = _assetManager->FindAssetByPath(selPath);
          ddi.type = ddi.asset ? ddi.asset->GetType() : AssetType::Unknown;
          _dragPayload.items.push_back(std::move(ddi));
        }
      } else if (item.isFolder) {
        _dragPayload.items.push_back({true, AssetType::Unknown, item.fullPath, nullptr});
      } else {
        _dragPayload.items.push_back(
            {false, asset ? asset->GetType() : item.file->type, item.fullPath, asset});
      }
      auto* payloadPtr = &_dragPayload;
      ImGui::SetDragDropPayload(kAssetBrowserDragDropType, &payloadPtr, sizeof(payloadPtr));

      // Drag preview
      if (isMultiDrag) {
        char countStr[16];
        snprintf(countStr, sizeof(countStr), "%d", static_cast<int>(_selectedPaths.size()));
        ImGui::SimpleAssetThumbnail(
            "##MultiDrag",
            tileSize,
            countStr,
            _studio->GetFont("Asset Tile Icons"),
            IM_COL32(255, 255, 255, 255));
      } else if (item.isFolder) {
        ImGui::FolderThumbnail(
            item.name.c_str(),
            tileSize,
            _studio->GetFont("Folder Tile Icons"),
            IM_COL32(230, 190, 100, 255));
      } else if (asset) {
        ImGui::AssetThumbnail(
            asset->GetName().c_str(), asset->GetThumbnailImage(), tileSize, asset->GetColor());
      } else {
        ImGui::SimpleAssetThumbnail(
            item.name.c_str(),
            tileSize,
            ICON_FA_FILE,
            _studio->GetFont("Asset Tile Icons"),
            GetAssetTypeColor(item.file->type));
      }
      bool const ctrl = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeySuper;
      if (ctrl) {
        ImGui::PushFont(_studio->GetFont("Roboto Bold Large"));
        ImVec2 const thumbMin = ImGui::GetItemRectMin();
        ImVec2 const thumbMax = ImGui::GetItemRectMax();
        ImVec2 const iconSize = ImGui::CalcTextSize(ICON_FA_PLUS);
        float const pad = 5.0f;
        ImVec2 const iconPos = {thumbMax.x - iconSize.x - pad, thumbMin.y + pad};
        ImGui::GetWindowDrawList()->AddText(iconPos, IM_COL32(255, 255, 255, 255), ICON_FA_PLUS);
        ImGui::PopFont();
      }
      ImGui::EndDragDropSource();
    }

    // Drop target (folder tiles only)
    if (item.isFolder) {
      TryAcceptAssetBrowserDrop(item.fullPath, true);
    }

    // Click handling: double-click takes priority over single-click
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      assetClicked = true;
      if (!isRenaming && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        ClearSelection();
        if (item.isFolder) {
          SetCurrentPath(item.fullPath);
        } else if (item.file->importer != nullptr) {
          _studio->BeginImport(item.file->importer, item.fullPath);
        } else if (asset) {
          assetToOpen = asset;
        } else {
          if (auto* loaded = _assetManager->LoadAsset(item.fullPath)) {
            assetToOpen = loaded;
          }
        }
      } else {
        bool const ctrl = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeySuper;
        bool const shift = ImGui::GetIO().KeyShift;
        if (shift && _lastClickedIndex >= 0 &&
            _lastClickedIndex < static_cast<int>(_visibleIndices.size())) {
          if (!ctrl) {
            _selectedPaths.clear();
          }
          int const lo = ImMin(_lastClickedIndex, vi);
          int const hi = ImMax(_lastClickedIndex, vi);
          for (int r = lo; r <= hi; ++r) {
            _selectedPaths.insert(_tileItems[_visibleIndices[r]].fullPath);
          }
        } else if (ctrl) {
          if (selected) {
            _selectedPaths.erase(item.fullPath);
          } else {
            _selectedPaths.insert(item.fullPath);
          }
          _lastClickedIndex = vi;
        } else if (!selected) {
          // Not yet selected: select immediately on mouse-down
          _selectedPaths.clear();
          _selectedPaths.insert(item.fullPath);
          _lastClickedIndex = vi;
        }
        // If already selected without modifiers, defer to mouse-release
        // so multi-selection drag works (see below).
      }
    }
    // Deferred single-select on mouse-release: if the item was already selected
    // and the user released without dragging, narrow selection to just this item.
    if (ImGui::IsItemHovered() && selected && ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
        !ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
      bool const ctrl = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeySuper;
      bool const shift = ImGui::GetIO().KeyShift;
      if (!ctrl && !shift && _selectedPaths.size() > 1) {
        _selectedPaths.clear();
        _selectedPaths.insert(item.fullPath);
        _lastClickedIndex = vi;
      }
    }

    // Wrap deterministically on the computed column count rather than a
    // floating-point lookahead, which avoids both right-edge gaps and the
    // possibility of disagreement with the tile-size math above.
    if ((vi + 1) % numColumns != 0) {
      ImGui::SameLine();
    }

    ImGui::PopID();
  }

  ImGui::PopStyleColor();

  // Context menu for empty content area
  if (ImGui::BeginPopupContextWindow(
          "##ContentAreaContextMenu",
          ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
    TryShowPasteSelectable(_currentPath);
    if (ImGui::IconSelectable("New Folder...", ICON_FA_FOLDER_PLUS)) {
      std::string folderName = MakeUniqueFileName("New Folder", _currentPath);
      if (CreateFolder(_currentPath, folderName)) {
        auto const newPath = _currentPath / folderName;
        BeginRenaming(newPath, folderName, [this](std::string const& newName) {
          RenameFolder(_renamingPath, newName);
        });
      }
    }
    // TODO: Abstract behind registered asset factories
    if (ImGui::BeginMenu(ICON_FA_PLUS "    Create")) {
      if (ImGui::Selectable("Bot...")) {
        using namespace mochi;
        auto dir = _currentPath;
        std::string botName = MakeUniqueFileName("Bot", dir, ".superdex_bot");
        auto newPath = dir / (botName + ".superdex_bot");
        superdex::robotics::BotPrefab prefab;
        prefab.name = botName;
        prefab.joints.resize(2);
        prefab.joints[0].name = "world_joint";
        prefab.joints[1].name = "joint_1";
        prefab.joints[1].type = mochi::ArticulatedJointType::Revolute;
        prefab.joints[1].axis = {0_r, 0_r, 1_r};
        prefab.links.resize(2);
        prefab.links[0].name = "base_link";
        prefab.links[1].name = "link_1";
        prefab.links[1].parentLink = 0;
        superdex::robotics::SaveToFile(prefab, newPath.ToString(), mochi::ErrorLog{});
        BeginRenaming(newPath, botName, [this](std::string const& newName) {
          RenameAsset(_renamingPath, newName);
          auto renamedPath = _renamingPath.GetParentPath() / (newName + ".superdex_bot");
          auto* bot = _assetManager->LoadBotAsset(renamedPath);
          if (bot) {
            bot->GetBotPrefab().name = newName;
          }
        });
      }
      if (ImGui::Selectable("Mod Bot...")) {
        CreateModBotAsset(_currentPath);
      }
      if (ImGui::Selectable("Prefab...")) {
        auto dir = _currentPath;
        std::string prefabName = MakeUniqueFileName("Prefab", dir, ".mochi_prefab");
        auto newPath = dir / (prefabName + ".mochi_prefab");
        mochi::prefab::ScenePrefab prefab;
        mochi::prefab::SaveToJsonFile(prefab, newPath.ToString(), mochi::ErrorLog{});
        BeginRenaming(newPath, prefabName, [this](std::string const& newName) {
          RenameAsset(_renamingPath, newName);
        });
      }
      // A bots root marks a folder as the anchor for root-relative (`//`) and tag (`@`) bot
      // references. At most one per folder, so disable when this folder already has one.
      bool const hasRoot = _currentPath.IsEmpty() ||
          superdex::robotics::FindRootMarker(_currentPath.AsFilesystemPath()).has_value();
      ImGui::BeginDisabled(hasRoot);
      if (ImGui::Selectable("Root")) {
        mochi::ErrorLog error;
        _studio->CreateRootMarker(_currentPath, error);
        if (error.IsOK()) {
          _needsRefresh = true;
        }
      }
      ImGui::EndDisabled();
      if (hasRoot && !_currentPath.IsEmpty() &&
          ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("This folder is already a root.");
      }
#if MOCHI_INTERNAL
      if (ImGui::Selectable("Bot Scene...")) {
        auto dir = _currentPath;
        std::string const extension{superdex::robotics::kBotSceneExtension};
        std::string sceneName = MakeUniqueFileName("Bot Scene", dir, extension);
        auto newPath = dir / (sceneName + extension);
        superdex::robotics::BotScenePrefab prefab;
        prefab.metadata.name = sceneName;
        superdex::robotics::SaveToFile(prefab, newPath.ToString(), mochi::ErrorLog{});
        BeginRenaming(newPath, sceneName, [this, extension](std::string const& newName) {
          RenameAsset(_renamingPath, newName);
          auto renamedPath = _renamingPath.GetParentPath() / (newName + extension);
          auto* botScene = _assetManager->LoadBotSceneAsset(renamedPath);
          if (botScene) {
            botScene->GetPrefab().metadata.name = newName;
            // The file was written with a placeholder name before the user picked one.
            botScene->Save();
          }
        });
      }
#endif // MOCHI_INTERNAL
      ImGui::EndMenu();
    }
    ImGui::Separator(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
    if (!_currentPath.IsEmpty() &&
        ImGui::IconSelectable("Open in File Browser", ICON_FA_FOLDER_OPEN)) {
      auto& pio = ImGui::GetPlatformIO();
      if (pio.Platform_OpenInShellFn) {
        pio.Platform_OpenInShellFn(ImGui::GetCurrentContext(), _currentPath.ToString().c_str());
      }
    }
    ImGui::EndPopup();
  }

  if (ImGui::IsWindowFocused() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !assetClicked) {
    ClearSelection();
  }
  if (_renamingPath.IsEmpty() && _clipboardMode != ClipboardMode::None &&
      ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    _clipboardPaths.clear();
    _clipboardMode = ClipboardMode::None;
  }

  // Keyboard shortcuts (only when content area is focused and not renaming)
  if (ImGui::IsWindowFocused() && _renamingPath.IsEmpty()) {
    // Ctrl+A: select all visible items
    if (ImGui::IsKeyChordPressed(ImGuiKey_A | ImGuiMod_Ctrl)) {
      _selectedPaths.clear();
      for (int const idx : _visibleIndices) {
        _selectedPaths.insert(_tileItems[idx].fullPath);
      }
    }
    // Delete/Backspace: delete selected items
    if ((ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace)) &&
        !_selectedPaths.empty()) {
      DeleteItems(_selectedPaths);
      ClearSelection();
    }
    // Ctrl+C: copy selection to clipboard
    if (ImGui::IsKeyChordPressed(ImGuiKey_C | ImGuiMod_Ctrl) && !_selectedPaths.empty()) {
      _clipboardPaths = _selectedPaths;
      _clipboardMode = ClipboardMode::Copy;
    }
    // Ctrl+X: cut selection to clipboard
    if (ImGui::IsKeyChordPressed(ImGuiKey_X | ImGuiMod_Ctrl) && !_selectedPaths.empty()) {
      _clipboardPaths = _selectedPaths;
      _clipboardMode = ClipboardMode::Cut;
    }
    // Ctrl+V: paste clipboard into current directory
    if (ImGui::IsKeyChordPressed(ImGuiKey_V | ImGuiMod_Ctrl) &&
        _clipboardMode != ClipboardMode::None) {
      ExecutePaste(_currentPath);
    }
    // Ctrl+D: duplicate single selected item
    if (ImGui::IsKeyChordPressed(ImGuiKey_D | ImGuiMod_Ctrl) && _selectedPaths.size() == 1) {
      for (auto const& ti : _tileItems) {
        if (ti.fullPath == *_selectedPaths.begin()) {
          DuplicateItem(ti);
          break;
        }
      }
    }
  }

  ImGui::EndChild();
  // Content area drop target if not filtering (drop into current directory)
  if (!filtering) {
    TryAcceptAssetBrowserDrop(_currentPath, false);
  }

  ImGui::EndGroup();

  ShowDeleteBlockedModal();
  ShowBatchRenameModal();
  ShowReplaceWithModal();

  if (assetToOpen) {
    _studio->OpenAssetEditor(assetToOpen);
  }

  ImGui::End();
}

void AssetBrowser::SelectAsset(Asset* asset) {
  ClearSelection();
  if (asset == nullptr) {
    return;
  }
  _forceAssetFocus = true;
  auto const assetPath = asset->GetPath();
  if (_assetManager->FindAssetByPath(assetPath)) {
    _selectedPaths.insert(assetPath);
    auto parentDir = assetPath.GetParentPath();
    if (!parentDir.IsEmpty() && std::filesystem::is_directory(parentDir.AsFilesystemPath())) {
      SetCurrentPath(parentDir);
    }
  }
}

} // namespace superdex::studio
