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

#if MOCHI_INTERNAL

#include "assets/bot_scene_asset.h"
#include "app/app.h"
#include "assets/asset.h"
#include "assets/asset_manager.h"
#include "assets/bot_asset.h"
#include "assets/mochi_prefab_asset.h"
#include "editors/bot_scene_editor.h"
#include "rendering/scene_stage.h"

#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/path.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <random>
#include <set>
#include <system_error>

namespace superdex::studio {

namespace {

int constexpr kMaxUniqueNameAttempts = 16;

std::string RandomNameSuffix() {
  thread_local std::mt19937_64 generator{std::random_device{}()};
  std::array<char, 17> buffer{};
  std::snprintf(
      buffer.data(), buffer.size(), "%016llx", static_cast<unsigned long long>(generator()));
  return buffer.data();
}

// Creates a new directory, private to the current user, directly in the system temp directory.
// Creation fails rather than reusing an existing path, and there is no shared intermediate
// directory another user could own and use to swap the workspace out.
bool CreateUniqueTemporaryDirectory(std::string_view stem, std::filesystem::path& directory) {
  std::error_code error;
  auto const parent = std::filesystem::temp_directory_path(error);
  for (int attempt = 0; !error && attempt < kMaxUniqueNameAttempts; ++attempt) {
    auto const candidate =
        parent / ("superdex_studio_" + std::string(stem) + "_" + RandomNameSuffix());
    if (std::filesystem::create_directory(candidate, error)) {
      std::filesystem::permissions(candidate, std::filesystem::perms::owner_all, error);
      directory = candidate;
      return !error;
    }
  }
  MOCHI_LOG_ERROR(
      "Failed to create a temporary directory under '%s': %s",
      parent.generic_string().c_str(),
      error ? error.message().c_str() : "no unused name found");
  return false;
}

// Returns a hidden path beside @p destination that does not exist yet.
std::filesystem::path MakeUniqueSibling(
    std::filesystem::path const& destination,
    std::string_view suffix) {
  std::filesystem::path candidate;
  std::error_code error;
  for (int attempt = 0; attempt < kMaxUniqueNameAttempts; ++attempt) {
    candidate = destination.parent_path() /
        ("." + destination.filename().string() + "_" + RandomNameSuffix() + std::string(suffix));
    if (!std::filesystem::exists(candidate, error) && !error) {
      break;
    }
  }
  return candidate;
}

bool CopyDirectoryContents(
    std::filesystem::path const& source,
    std::filesystem::path const& destination) {
  std::error_code error;
  std::filesystem::create_directories(destination, error);
  if (error) {
    MOCHI_LOG_ERROR(
        "Failed to create directory '%s': %s",
        destination.generic_string().c_str(),
        error.message().c_str());
    return false;
  }
  for (auto const& entry : std::filesystem::directory_iterator(source, error)) {
    if (error) {
      break;
    }
    std::filesystem::copy(
        entry.path(),
        destination / entry.path().filename(),
        std::filesystem::copy_options::recursive | std::filesystem::copy_options::copy_symlinks,
        error);
    if (error) {
      break;
    }
  }
  if (error) {
    MOCHI_LOG_ERROR(
        "Failed to copy archive workspace from '%s' to '%s': %s",
        source.generic_string().c_str(),
        destination.generic_string().c_str(),
        error.message().c_str());
    return false;
  }
  return true;
}

bool IsPathInside(std::filesystem::path const& path, std::filesystem::path const& root) {
  auto const relative = path.lexically_normal().lexically_relative(root.lexically_normal());
  if (relative.empty() || relative.is_absolute()) {
    return false;
  }
  auto const first = relative.begin();
  return first == relative.end() || *first != "..";
}

bool RebasePath(
    mochi::DynamicString& value,
    std::filesystem::path const& oldRoot,
    std::filesystem::path const& newRoot,
    bool pathOrJson = false) {
  if (value.empty() ||
      (pathOrJson &&
       superdex::robotics::IsInlineJson(std::string_view(value.c_str(), value.size())))) {
    return true;
  }
  std::filesystem::path const path{value.c_str()};
  if (!path.is_absolute()) {
    return true;
  }
  // External references keep their absolute path (see ImportExternalReferences).
  if (!IsPathInside(path, oldRoot)) {
    return true;
  }
  value = (newRoot / path.lexically_normal().lexically_relative(oldRoot.lexically_normal()))
              .generic_string();
  return true;
}

bool RebaseScenePaths(
    superdex::robotics::BotScenePrefab& prefab,
    std::filesystem::path const& oldRoot,
    std::filesystem::path const& newRoot) {
  if (!RebasePath(prefab.scene.baseScene, oldRoot, newRoot)) {
    return false;
  }
  for (auto& entry : prefab.scene.spawnablePrefabs) {
    if (!RebasePath(entry.path, oldRoot, newRoot)) {
      return false;
    }
  }
  for (auto& bot : prefab.bots) {
    if (!RebasePath(bot.path, oldRoot, newRoot)) {
      return false;
    }
    for (auto& controller : bot.controllers) {
      if (!RebasePath(controller.params, oldRoot, newRoot, true)) {
        return false;
      }
    }
  }
  for (auto& sensor : prefab.sensors) {
    if (!RebasePath(sensor.params, oldRoot, newRoot, true)) {
      return false;
    }
  }
  return true;
}

bool HaveSameContents(std::filesystem::path const& a, std::filesystem::path const& b) {
  std::error_code error;
  if (std::filesystem::equivalent(a, b, error)) {
    return true;
  }
  if (std::filesystem::file_size(a, error) != std::filesystem::file_size(b, error) || error) {
    return false;
  }
  mochi::Error readError;
  return mochi::ReadFileString(a, readError) == mochi::ReadFileString(b, readError) &&
      readError.IsOK();
}

// Copies files the scene references from outside @p workspace into it and points the scene at the
// copies. Each file keeps its path relative to its own .superdex_root, placed under the root of
// @p sceneFile, so the relative and root-relative paths inside an imported prefab still resolve. A
// .mochi_scene/.mochi_prefab brings the files it references with it.
bool ImportExternalReferences(
    superdex::robotics::BotScenePrefab& prefab,
    std::filesystem::path const& workspace,
    std::filesystem::path const& sceneFile) {
  // References are canonicalized before checking whether they are inside the workspace, so the
  // workspace must be too (macOS's temp directory is reached through a symlink). Copies are placed
  // under the uncanonicalized root so the scene saved beside them can refer to them.
  auto const workspaceRoot = std::filesystem::weakly_canonical(workspace);
  auto const sceneRoot = superdex::robotics::FindBotsRoot(sceneFile);
  if (!sceneRoot.has_value()) {
    MOCHI_LOG_ERROR("Bot scene archive target has no .superdex_root");
    return false;
  }
  auto const importFile = [&](std::filesystem::path const& file, std::filesystem::path& copy) {
    auto const root = superdex::robotics::FindBotsRoot(file);
    if (!root.has_value()) {
      MOCHI_LOG_ERROR(
          "Cannot add '%s' to the archive: it is not under a .superdex_root",
          file.generic_string().c_str());
      return false;
    }
    copy = *sceneRoot / file.lexically_relative(*root);
    std::error_code error;
    if (std::filesystem::exists(copy, error)) {
      if (HaveSameContents(file, copy)) {
        return true;
      }
      MOCHI_LOG_ERROR(
          "Cannot add '%s' to the archive: it already contains a different '%s'",
          file.generic_string().c_str(),
          file.lexically_relative(*root).generic_string().c_str());
      return false;
    }
    std::filesystem::create_directories(copy.parent_path(), error);
    if (!error) {
      std::filesystem::copy_file(file, copy, error);
    }
    if (error) {
      MOCHI_LOG_ERROR(
          "Failed to add '%s' to the archive: %s",
          file.generic_string().c_str(),
          error.message().c_str());
      return false;
    }
    return true;
  };
  auto const import = [&](mochi::DynamicString& value, bool pathOrJson) {
    if (value.empty() ||
        (pathOrJson &&
         superdex::robotics::IsInlineJson(std::string_view(value.c_str(), value.size())))) {
      return true;
    }
    std::filesystem::path const path = std::filesystem::weakly_canonical(value.c_str());
    if (!path.is_absolute() || IsPathInside(path, workspaceRoot)) {
      return true;
    }
    if (superdex::robotics::IsBotPath(path.generic_string())) {
      MOCHI_LOG_ERROR(
          "Cannot add bot '%s' to the archive; reference its .superdex_bot_archive instead",
          path.generic_string().c_str());
      return false;
    }
    std::set<std::filesystem::path> files{path};
    auto const extension = path.extension();
    if (extension == ".mochi_prefab" || extension == ".mochi_scene") {
      mochi::ErrorLog error;
      files = superdex::robotics::CollectPrefabFiles(path, error);
      if (!error.IsOK()) {
        return false;
      }
    }
    for (auto const& file : files) {
      std::filesystem::path copy;
      if (!importFile(file, copy)) {
        return false;
      }
      if (file == path) {
        value = copy.generic_string();
      }
    }
    return true;
  };

  bool ok = import(prefab.scene.baseScene, false);
  for (auto& entry : prefab.scene.spawnablePrefabs) {
    ok = ok && import(entry.path, false);
  }
  for (auto& bot : prefab.bots) {
    ok = ok && import(bot.path, false);
    for (auto& controller : bot.controllers) {
      ok = ok && import(controller.params, true);
    }
  }
  for (auto& sensor : prefab.sensors) {
    ok = ok && import(sensor.params, true);
  }
  return ok;
}

bool ReplaceFileAfterSuccessfulWrite(
    std::filesystem::path const& temporary,
    std::filesystem::path const& destination) {
  std::error_code error;
  if (!std::filesystem::exists(destination, error)) {
    std::filesystem::rename(temporary, destination, error);
    if (!error) {
      return true;
    }
    std::error_code removeError;
    std::filesystem::remove(temporary, removeError);
    MOCHI_LOG_ERROR(
        "Failed to move saved archive to '%s': %s",
        destination.generic_string().c_str(),
        error.message().c_str());
    return false;
  }

  auto const backup = MakeUniqueSibling(destination, ".backup");
  std::filesystem::rename(destination, backup, error);
  if (error) {
    std::error_code removeError;
    std::filesystem::remove(temporary, removeError);
    MOCHI_LOG_ERROR("Failed to preserve existing archive: %s", error.message().c_str());
    return false;
  }

  std::filesystem::rename(temporary, destination, error);
  if (error) {
    std::error_code restoreError;
    std::filesystem::rename(backup, destination, restoreError);
    std::error_code removeError;
    std::filesystem::remove(temporary, removeError);
    MOCHI_LOG_ERROR(
        "Failed to replace archive '%s': %s",
        destination.generic_string().c_str(),
        error.message().c_str());
    if (restoreError) {
      MOCHI_LOG_ERROR(
          "Failed to restore original archive from '%s': %s",
          backup.generic_string().c_str(),
          restoreError.message().c_str());
    }
    return false;
  }
  std::filesystem::remove(backup, error);
  return true;
}

} // namespace

std::unique_ptr<BotSceneAsset>
BotSceneAsset::Create(std::string const& name, mochi::Path const& path, AssetManager* manager) {
  mochi::ErrorLog error;

  // Archive case: load from the shared extraction cache while keeping the archive as asset
  // identity.
  bool const isArchive = superdex::robotics::IsSceneArchivePath(path.GetExtensionLowercase());
  std::string loadPath = path.ToString();
  std::string extractedDirPath;
  std::optional<mochi::DynamicString> archiveComment;
  if (isArchive) {
    auto extractedDir = superdex::robotics::ExtractBotSceneArchiveToCache(path.ToString(), error);
    auto targetPath = superdex::robotics::GetExtractedBotSceneArchiveTarget(extractedDir, error);
    auto metadata = superdex::robotics::ReadBotSceneArchiveMetadata(extractedDir, error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to extract BotSceneAsset: %s", path.ToString().c_str());
      return nullptr;
    }
    extractedDirPath = std::string(extractedDir);
    loadPath = std::string(targetPath);
    archiveComment = std::move(metadata.comment);
  }

  auto prefab = superdex::robotics::LoadBotScenePrefabFromFile(loadPath, error);
  if (!error.IsOK()) {
    MOCHI_LOG_ERROR("Failed to load BotSceneAsset: %s", path.ToString().c_str());
    return nullptr;
  }

  // Recursively load referenced assets so they show up in the AssetManager and
  // participate in reference tracking.
  if (!prefab.scene.baseScene.empty()) {
    manager->LoadMochiPrefabAsset(prefab.scene.baseScene);
  }
  for (auto const& entry : prefab.scene.spawnablePrefabs) {
    if (!entry.path.empty()) {
      // Spawnable prefabs can be either .mochi_scene prefabs or bot archives
      if (superdex::robotics::IsBotArchivePath(entry.path) ||
          superdex::robotics::IsBotPath(entry.path)) {
        manager->LoadBotAsset(entry.path);
      } else {
        manager->LoadMochiPrefabAsset(entry.path);
      }
    }
  }
  for (auto const& bot : prefab.bots) {
    if (!bot.path.empty()) {
      manager->LoadBotAsset(bot.path);
    }
  }

  auto asset =
      std::unique_ptr<BotSceneAsset>(new BotSceneAsset(name, path, AssetType::BotScene, manager));
  asset->_prefab = std::move(prefab);
  asset->_isArchive = isArchive;
  if (isArchive) {
    asset->_archiveExtractedDir = mochi::Path{extractedDirPath};
    asset->_archiveTargetPath = mochi::Path{loadPath};
    asset->_archiveComment = std::move(archiveComment);
  }

  // Resolve the base directory used for the base scene's relative shape/prefab paths: the
  // .superdex_root if present, otherwise the base scene's own directory.
  auto botsRoot = superdex::robotics::FindBotsRoot(loadPath);
  asset->_botsRootPath = botsRoot
      ? botsRoot->string()
      : mochi::Path{std::string(asset->_prefab.scene.baseScene)}.GetParentPath().ToString();

  manager->RegisterReferencer(asset.get());
  return asset;
}

char const* BotSceneAsset::GetTypeLabel() const {
  return _isArchive ? "Bot Scene Arch." : "Bot Scene";
}

bool BotSceneAsset::RendersThumbnail() const {
  return true;
}

void BotSceneAsset::StageThumbnailScene(mochi_renderer::Scene& scene) {
  SceneStage stage(_manager->GetStudio(), "BotSceneThumbnailStage");
  stage.BindRenderScene(&scene);
  stage.StageBotScene(_prefab, StageType::RenderModelFallbackToMochiModel);
}

bool BotSceneAsset::IsSavable() const {
  return !IsReadOnly();
}

bool BotSceneAsset::Save() {
  if (IsReadOnly()) {
    MOCHI_LOG_ERROR("Attempting to save read-only BotSceneAsset");
    return false;
  }
  if (!SaveControllerParams((_isArchive ? _archiveTargetPath : _path).GetParentPath())) {
    return false;
  }
  if (std::string error; !_taskCache.SaveModified(error)) {
    MOCHI_LOG_ERROR("%s", error.c_str());
    return false;
  }
  if (_isArchive) {
    return SaveArchiveCopy(_path);
  }
  mochi::ErrorLog error;
  superdex::robotics::SaveToFile(_prefab, _path.ToString(), error);
  return error.IsOK();
}

std::unique_ptr<AssetEditor> BotSceneAsset::CreateEditor(SuperDexStudio* studio) {
  return std::make_unique<BotSceneEditor>(studio, this);
}

std::string const& BotSceneAsset::GetReferencerName() const {
  return GetName();
}

void BotSceneAsset::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  auto visit = [&](mochi::DynamicString const& s) {
    if (!s.empty()) {
      callback(mochi::Path{s.c_str()});
    }
  };
  visit(_prefab.scene.baseScene);
  for (auto const& entry : _prefab.scene.spawnablePrefabs) {
    visit(entry.path);
  }
  for (auto const& bot : _prefab.bots) {
    visit(bot.path);
    for (auto const& controller : bot.controllers) {
      if (!superdex::robotics::IsInlineJson(controller.params)) {
        visit(controller.params);
      }
    }
  }
  for (auto const& sensor : _prefab.sensors) {
    if (!superdex::robotics::IsInlineJson(sensor.params)) {
      visit(sensor.params);
    }
  }
}

namespace {

bool MaybeRewrite(
    mochi::DynamicString& field,
    mochi::Path const& oldPath,
    mochi::Path const& newPath) {
  if (field.empty()) {
    return false;
  }
  if (mochi::Path{field.c_str()} == oldPath) {
    field = newPath.ToString();
    return true;
  }
  return false;
}

} // namespace

bool BotSceneAsset::RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) {
  bool changed = false;
  changed |= MaybeRewrite(_prefab.scene.baseScene, oldPath, newPath);
  for (auto& entry : _prefab.scene.spawnablePrefabs) {
    changed |= MaybeRewrite(entry.path, oldPath, newPath);
  }
  for (auto& bot : _prefab.bots) {
    changed |= MaybeRewrite(bot.path, oldPath, newPath);
    for (auto& controller : bot.controllers) {
      if (!superdex::robotics::IsInlineJson(controller.params)) {
        changed |= MaybeRewrite(controller.params, oldPath, newPath);
      }
    }
  }
  for (auto& sensor : _prefab.sensors) {
    if (!superdex::robotics::IsInlineJson(sensor.params)) {
      changed |= MaybeRewrite(sensor.params, oldPath, newPath);
    }
  }
  if (changed) {
    SetDirty(true);
  }
  return changed;
}

superdex::robotics::BotScenePrefab const& BotSceneAsset::GetPrefab() const {
  return _prefab;
}

superdex::robotics::BotScenePrefab& BotSceneAsset::GetPrefab() {
  return _prefab;
}

ControllerParamsCache& BotSceneAsset::GetControllerParams() {
  return _controllerParams;
}

BotTaskCache& BotSceneAsset::GetTaskCache() {
  return _taskCache;
}

std::string const& BotSceneAsset::GetBotsRootPath() const {
  return _botsRootPath;
}

bool BotSceneAsset::IsArchive() const {
  return _isArchive;
}

bool BotSceneAsset::SaveControllerParams(mochi::Path const& sceneDirectory) {
  auto const choosePath = [&](superdex::robotics::BotEntry const& bot,
                              superdex::robotics::ControllerEntry const& controller) {
    std::array<char const*, 1> const filters{{"*.superdex_controller"}};
#if MOCHI_PLATFORM_MACOS
    // Custom extensions without a registered UTI are disabled when the native filter is set.
    int constexpr numFilters = 0;
#else
    int constexpr numFilters = static_cast<int>(filters.size());
#endif
    return SuperDexStudio::GetFileDialogPath(
               "Save Controller Parameters",
               filters.data(),
               numFilters,
               "SuperDex Controller (*.superdex_controller)",
               true,
               sceneDirectory /
                   (std::string(bot.name) + "_" + std::string(controller.name) +
                    ".superdex_controller"))
        .ToString();
  };
  std::string paramsError;
  bool const externalized =
      ExternalizeInlineParams(_prefab, _controllerParams, choosePath, paramsError);
  if (externalized) {
    _manager->ResyncReferencer(this);
  }
  if (!externalized || !_controllerParams.SaveModified(paramsError)) {
    MOCHI_LOG_ERROR("Failed to save controller parameters: %s", paramsError.c_str());
    return false;
  }
  return true;
}

bool BotSceneAsset::SupportsSaveAs() const {
  return true;
}

bool BotSceneAsset::SaveAs(mochi::Path const& path) {
  if (!SaveControllerParams((_isArchive ? _archiveTargetPath : path).GetParentPath())) {
    return false;
  }
  if (_isArchive) {
    return SaveArchiveCopy(path);
  }
  mochi::ErrorLog error;
  superdex::robotics::SaveToFile(_prefab, path.ToString(), error);
  return error.IsOK();
}

bool BotSceneAsset::SaveArchiveCopy(mochi::Path const& path) const {
  if (!_isArchive || !superdex::robotics::IsSceneArchivePath(path.GetExtensionLowercase())) {
    MOCHI_LOG_ERROR("Bot scene archive output must end with .superdex_scene_archive");
    return false;
  }

  std::filesystem::path const extractedRoot = _archiveExtractedDir.AsFilesystemPath();
  std::filesystem::path const extractedTarget = _archiveTargetPath.AsFilesystemPath();
  if (!IsPathInside(extractedTarget, extractedRoot)) {
    MOCHI_LOG_ERROR("Bot scene archive target is outside its extracted root");
    return false;
  }

  std::filesystem::path workspace;
  if (!CreateUniqueTemporaryDirectory("scene_archive_save", workspace)) {
    return false;
  }
  MOCHI_DEFER({
    std::error_code cleanupError;
    std::filesystem::remove_all(workspace, cleanupError);
  });
  if (!CopyDirectoryContents(extractedRoot, workspace)) {
    return false;
  }

  auto const targetRelative = extractedTarget.lexically_relative(extractedRoot);
  auto const workspaceTarget = workspace / targetRelative;
  auto stagedPrefab = _prefab;
  if (!RebaseScenePaths(stagedPrefab, extractedRoot, workspace) ||
      !ImportExternalReferences(stagedPrefab, workspace, workspaceTarget)) {
    return false;
  }

  mochi::ErrorLog error;
  superdex::robotics::SaveToFile(stagedPrefab, workspaceTarget.generic_string(), error);
  if (!error.IsOK()) {
    MOCHI_LOG_ERROR("Failed to stage edited bot scene for archiving");
    return false;
  }

  std::filesystem::path const& destination = path.AsFilesystemPath();
  auto const temporaryArchive =
      MakeUniqueSibling(destination, superdex::robotics::kSceneArchiveExtension);
  std::error_code removeError;
  std::filesystem::remove(temporaryArchive, removeError);

  superdex::robotics::ArchiveParams params;
  params.src = workspaceTarget.generic_string();
  params.dst = temporaryArchive.generic_string();
  params.comment = _archiveComment;
  superdex::robotics::ArchiveBotScene(params, error);
  if (!error.IsOK()) {
    std::filesystem::remove(temporaryArchive, removeError);
    MOCHI_LOG_ERROR(
        "Failed to create bot scene archive '%s'", destination.generic_string().c_str());
    return false;
  }
  return ReplaceFileAfterSuccessfulWrite(temporaryArchive, destination);
}

bool BotSceneAsset::ExtractArchiveToLooseFiles(mochi::Path const& directory, mochi::Path& scenePath)
    const {
  if (!_isArchive) {
    return false;
  }
  std::filesystem::path const& destination = directory.AsFilesystemPath();
  std::error_code error;
  bool const isDirectory = std::filesystem::is_directory(destination, error);
  if (error || !isDirectory) {
    MOCHI_LOG_ERROR("Extraction destination must be an existing directory");
    return false;
  }
  bool const isEmpty = std::filesystem::is_empty(destination, error);
  if (error || !isEmpty) {
    MOCHI_LOG_ERROR("Extraction destination must be an empty directory");
    return false;
  }

  std::filesystem::path const extractedRoot = _archiveExtractedDir.AsFilesystemPath();
  std::filesystem::path const extractedTarget = _archiveTargetPath.AsFilesystemPath();
  if (destination.lexically_normal() == extractedRoot.lexically_normal() ||
      IsPathInside(destination, extractedRoot)) {
    MOCHI_LOG_ERROR("Extraction destination cannot be inside the shared archive cache");
    return false;
  }
  if (!CopyDirectoryContents(extractedRoot, destination)) {
    return false;
  }

  auto const cleanup = [&] {
    std::error_code cleanupError;
    for (auto const& entry : std::filesystem::directory_iterator(destination, cleanupError)) {
      std::filesystem::remove_all(entry.path(), cleanupError);
    }
  };
  auto const targetRelative = extractedTarget.lexically_relative(extractedRoot);
  auto const editableTarget = destination / targetRelative;
  auto editablePrefab = _prefab;
  if (!RebaseScenePaths(editablePrefab, extractedRoot, destination)) {
    cleanup();
    return false;
  }

  mochi::ErrorLog saveError;
  superdex::robotics::SaveToFile(editablePrefab, editableTarget.generic_string(), saveError);
  if (!saveError.IsOK()) {
    cleanup();
    MOCHI_LOG_ERROR("Failed to save extracted editable bot scene");
    return false;
  }
  std::filesystem::remove(destination / superdex::robotics::kSceneArchiveMetadataFile, error);
#if MOCHI_INTERNAL
  std::filesystem::remove(destination / superdex::robotics::kSceneArchiveMetadataFileLegacy, error);
#endif
  scenePath = mochi::Path{editableTarget};
  return true;
}

} // namespace superdex::studio

#endif // MOCHI_INTERNAL
