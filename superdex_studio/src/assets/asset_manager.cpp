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

#include "assets/asset_manager.h"
#include "app/app.h"
#include "core/common.h"
#include "editors/asset_editor.h"

#include <mochi_core/utils/path.h>
#include <mochi_renderer/type_conversions.h>

#include <functional>
#include <set>
#include <vector>

#include "superdex_robotics/utils/archive_utils.h"

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// ASSET
//--------------------------------------------------------------------------------------------------

AssetManager::AssetManager(SuperDexStudio* studio) : _studio(studio) {
  mochi_renderer::SceneViewSettings viewSettings;
  _thumbnailScene = mochi_renderer::Scene::Create(_studio->GetEngine(), viewSettings);
  _thumbnailScene->SetViewport(kDefaultThumbnailSize * 2, kDefaultThumbnailSize * 2);
  _thumbnailScene->CreateSkybox();
  _thumbnailScene->CreateSunlight();
  _thumbnailScene->CreateIndirectLight();
  _thumbnailScene->SetIbl(_studio->GetCurrentIbl());
  _thumbnailScene->SetSkyboxVisible(false);
}

std::unique_ptr<AssetManager> AssetManager::Create(SuperDexStudio* studio) {
  MOCHI_ASSERT(studio != nullptr);
  return std::unique_ptr<AssetManager>(new AssetManager(studio));
}

SuperDexStudio* AssetManager::GetStudio() const {
  return _studio;
}

Asset* AssetManager::LoadAsset(mochi::Path const& path) {
  if (auto it = _assets.find(path); it != _assets.end()) {
    MOCHI_LOG_VERBOSE("Asset is already loaded from path: %s", path.ToString().c_str());
    return it->second.get();
  }
  AssetType const type = ClassifyAssetTypeByPath(path);
  switch (type) {
    case AssetType::RenderModel:
      return LoadRenderModelAsset(path);
    case AssetType::CadModel:
      return LoadCadModelAsset(path);
    case AssetType::MochiModel:
      return LoadMochiModelAsset(path);
    case AssetType::MochiPrefab:
      return LoadMochiPrefabAsset(path);
    case AssetType::Bot:
      return LoadBotAsset(path);
#if MOCHI_INTERNAL
    case AssetType::BotScene:
      return LoadBotSceneAsset(path);
#else
    case AssetType::BotScene:
#endif
    case AssetType::Unknown:
    case AssetType::Count:
      break;
  }
  MOCHI_LOG_WARNING("Failed to load unsupported asset type: %s", path.ToString().c_str());
  return nullptr;
}

RenderModelAsset* AssetManager::LoadRenderModelAsset(mochi::Path const& path) {
  if (auto* existing = FindAssetByPath(path)) {
    if (existing->GetType() == AssetType::RenderModel) {
      MOCHI_LOG_VERBOSE("Asset is already loaded from path: %s", path.ToString().c_str());
      return static_cast<RenderModelAsset*>(existing);
    }
    MOCHI_LOG_ERROR(
        "Failed to cast previously loaded asset to RenderModelAsset: %s", path.ToString().c_str());
    return nullptr;
  }
  auto name = GetAssetNameFromPath(path);
  auto asset = RenderModelAsset::Create(name, path, this, _studio->GetResourceManager());
  if (asset) {
    return static_cast<RenderModelAsset*>(RegisterAsset(path, std::move(asset)));
  }
  MOCHI_LOG_ERROR("Failed to load RenderModelAsset: %s", path.ToString().c_str());
  return nullptr;
}

CadModelAsset* AssetManager::LoadCadModelAsset(mochi::Path const& path) {
  if (auto* existing = FindAssetByPath(path)) {
    if (existing->GetType() == AssetType::CadModel) {
      MOCHI_LOG_VERBOSE("Asset is already loaded from path: %s", path.ToString().c_str());
      return static_cast<CadModelAsset*>(existing);
    }
    MOCHI_LOG_ERROR(
        "Failed to cast previously loaded asset to CadModelAsset: %s", path.ToString().c_str());
    return nullptr;
  }
  auto name = GetAssetNameFromPath(path);
  auto asset = CadModelAsset::Create(name, path, this, _studio->GetResourceManager());
  if (asset) {
    return static_cast<CadModelAsset*>(RegisterAsset(path, std::move(asset)));
  }
  MOCHI_LOG_ERROR("Failed to load CadModelAsset: %s", path.ToString().c_str());
  return nullptr;
}

MochiModelAsset* AssetManager::LoadMochiModelAsset(mochi::Path const& path) {
  if (auto* existing = FindAssetByPath(path)) {
    if (existing->GetType() == AssetType::MochiModel) {
      MOCHI_LOG_VERBOSE("Asset is already loaded from path: %s", path.ToString().c_str());
      return static_cast<MochiModelAsset*>(existing);
    }
    MOCHI_LOG_ERROR(
        "Failed to cast previously loaded asset to MochiModelAsset: %s", path.ToString().c_str());
    return nullptr;
  }
  auto name = GetAssetNameFromPath(path);
  auto asset = MochiModelAsset::Create(name, path, this, _studio->GetResourceManager());
  if (asset) {
    return static_cast<MochiModelAsset*>(RegisterAsset(path, std::move(asset)));
  }
  MOCHI_LOG_ERROR("Failed to load MochiModelAsset: %s", path.ToString().c_str());
  return nullptr;
}

MochiPrefabAsset* AssetManager::LoadMochiPrefabAsset(mochi::Path const& path) {
  if (auto* existing = FindAssetByPath(path)) {
    if (existing->GetType() == AssetType::MochiPrefab) {
      MOCHI_LOG_VERBOSE("Asset is already loaded from path: %s", path.ToString().c_str());
      return static_cast<MochiPrefabAsset*>(existing);
    }
    MOCHI_LOG_ERROR(
        "Failed to cast previously loaded asset to MochiPrefabAsset: %s", path.ToString().c_str());
    return nullptr;
  }
  auto name = GetAssetNameFromPath(path);
  auto asset = MochiPrefabAsset::Create(name, path, this);
  if (asset) {
    return static_cast<MochiPrefabAsset*>(RegisterAsset(path, std::move(asset)));
  }
  MOCHI_LOG_ERROR("Failed to load MochiPrefabAsset: %s", path.ToString().c_str());
  return nullptr;
}

BotAsset* AssetManager::LoadBotAsset(mochi::Path const& path) {
  if (auto* existing = FindAssetByPath(path)) {
    if (existing->GetType() == AssetType::Bot) {
      MOCHI_LOG_VERBOSE("Asset is already loaded from path: %s", path.ToString().c_str());
      return static_cast<BotAsset*>(existing);
    }
    MOCHI_LOG_ERROR(
        "Failed to cast previously loaded asset to BotAsset: %s", path.ToString().c_str());
    return nullptr;
  }
  auto name = GetAssetNameFromPath(path);
  auto asset = BotAsset::Create(name, path, this);
  if (asset) {
    auto registerPath = asset->GetPath();
    return static_cast<BotAsset*>(RegisterAsset(registerPath, std::move(asset)));
  }
  MOCHI_LOG_ERROR("Failed to load BotAsset: %s", path.ToString().c_str());
  return nullptr;
}

#if MOCHI_INTERNAL
BotSceneAsset* AssetManager::LoadBotSceneAsset(mochi::Path const& path) {
  if (auto* existing = FindAssetByPath(path)) {
    if (existing->GetType() == AssetType::BotScene) {
      MOCHI_LOG_VERBOSE("Asset is already loaded from path: %s", path.ToString().c_str());
      return static_cast<BotSceneAsset*>(existing);
    }
    MOCHI_LOG_ERROR(
        "Failed to cast previously loaded asset to BotSceneAsset: %s", path.ToString().c_str());
    return nullptr;
  }
  auto name = GetAssetNameFromPath(path);
  auto asset = BotSceneAsset::Create(name, path, this);
  if (asset) {
    return static_cast<BotSceneAsset*>(RegisterAsset(path, std::move(asset)));
  }
  MOCHI_LOG_ERROR("Failed to load BotSceneAsset: %s", path.ToString().c_str());
  return nullptr;
}
#endif // MOCHI_INTERNAL

bool AssetManager::UnloadAssetByPath(mochi::Path const& path) {
  auto it = _assets.find(path);
  if (it == _assets.end()) {
    return false;
  }
  if (GetPathReferenceCount(path) != 0) {
    MOCHI_LOG_WARNING("Attempted to unload asset that is still referenced in memory!");
    return false;
  }
  it->second->OnUnload(_studio->GetResourceManager());
  UnregisterReferencer(dynamic_cast<IAssetReferencer*>(it->second.get()));
  // DANGLING-POINTER DISCIPLINE: this destroys the Asset object. A "reload" via this path is an
  // unload here followed by LoadAsset(), which creates a *new* Asset with a different address. Any
  // Asset* cached across such a reload (e.g. an editor's target asset, per-slot asset pointers,
  // scene objects holding an asset) therefore dangles and MUST be re-fetched after the reload --
  // see ModelEditor::PollSlotFileChanges, which re-points its slot pointers and base _asset. (An
  // asset that is still referenced cannot be unloaded here at all; callers that must refresh it in
  // that case use the in-place Asset::ReloadFromDisk, which re-reads into the same object and has
  // no such hazard.)
  _assets.erase(it);
  return true;
}

bool AssetManager::UnloadAssets(std::set<mochi::Path> const& paths) {
  // Unloads each requested path together with any in-memory assets that reference
  // it, recursively, so referencers are torn down before the assets they depend
  // on. This ordering is required because an asset cannot be safely unloaded while
  // another loaded asset still references it (UnloadAssetByPath bails in that
  // case). Callers performing a multi-select delete are responsible for ensuring
  // those referencers are also intended for removal. Non-asset referencers (e.g.
  // open editor tabs) still block unloading of the referenced asset.
  auto& resourceManager = _studio->GetResourceManager();
  std::set<mochi::Path> visiting;
  std::function<bool(mochi::Path const&)> tryUnload = [&](mochi::Path const& path) {
    auto it = _assets.find(path);
    if (it == _assets.end()) {
      return true; // not loaded; nothing to unload
    }
    if (!visiting.insert(path).second) {
      return false; // cycle
    }
    std::vector<mochi::Path> referencerPaths;
    for (auto* referencer : GetReferencersToPath(path)) {
      auto* refAsset = dynamic_cast<Asset*>(referencer);
      if (refAsset == nullptr) {
        visiting.erase(path);
        return false;
      }
      referencerPaths.push_back(refAsset->GetPath());
    }
    for (auto const& refPath : referencerPaths) {
      if (!tryUnload(refPath)) {
        visiting.erase(path);
        return false;
      }
    }
    visiting.erase(path);
    it = _assets.find(path); // re-find: recursion may have invalidated the iterator
    if (it == _assets.end()) {
      return true;
    }
    it->second->OnUnload(resourceManager);
    UnregisterReferencer(dynamic_cast<IAssetReferencer*>(it->second.get()));
    _assets.erase(it);
    return true;
  };
  bool allUnloaded = true;
  for (auto const& path : paths) {
    if (!tryUnload(path)) {
      MOCHI_LOG_WARNING(
          "Failed to unload asset still referenced in memory: %s", path.ToString().c_str());
      allUnloaded = false;
    }
  }
  return allUnloaded;
}

bool AssetManager::UnloadAllAssets() {
  // Collect paths up front since UnloadAssets mutates _assets during traversal.
  std::set<mochi::Path> paths;
  for (auto const& [path, asset] : _assets) {
    paths.insert(path);
  }
  UnloadAssets(paths);
  return _assets.empty();
}

Asset* AssetManager::FindAssetByPath(mochi::Path const& path, bool silent) const {
  auto it = _assets.find(path);
  if (it != _assets.end()) {
    return it->second.get();
  }
  if (!silent) {
    MOCHI_LOG_ERROR("Asset not found: %s", path.ToString().c_str());
  }
  return nullptr;
}

void AssetManager::ForEachAsset(
    std::function<void(Asset*, mochi::Path const&)> const& callback) const {
  for (auto const& [path, asset] : _assets) {
    callback(asset.get(), path);
  }
}

std::vector<Asset*> AssetManager::GetAllAssetsOfType(AssetType type) const {
  std::vector<Asset*> result;
  for (auto const& [path, asset] : _assets) {
    if (asset->GetType() == type) {
      result.push_back(asset.get());
    }
  }
  return result;
}

void AssetManager::SetThumbnailIbl(mochi_renderer::IBL* ibl) {
  if (_thumbnailScene) {
    _thumbnailScene->SetIbl(ibl);
    _thumbnailScene->SetSkyboxVisible(false);
  }
}

void AssetManager::RenderAssetThumbnails(Renderer& renderer, int maxThumbnails) {
  auto engine = _studio->GetEngine();
  int thumbnailsRendered = 0;
  ForEachAsset([&, this](Asset* asset, mochi::Path const&) {
    if (!asset->RendersThumbnail()) {
      return;
    }
    if (thumbnailsRendered < maxThumbnails) {
      if (!asset->_thumbRenderTarget) {
        asset->_thumbRenderTarget =
            RenderTarget::Create(engine, kDefaultThumbnailSize * 2, kDefaultThumbnailSize * 2);
        asset->_thumbDirty = true;
      }
      if (asset->_thumbDirty) {
        RenderAssetToTarget(*asset, *asset->_thumbRenderTarget, renderer);
        asset->_thumbDirty = false;
        asset->_thumbRendered = true;
        ++thumbnailsRendered;
      }
    }
  });
}

void AssetManager::RenderAssetToTarget(Asset& asset, RenderTarget& target, Renderer& renderer) {
  // Match the shared thumbnail scene's viewport (and thus the ortho camera aspect) to the target
  // size. The scene defaults to the tile resolution, so without this an export to a larger target
  // would only fill a corner.
  int width = 0;
  int height = 0;
  target.GetSize(width, height);
  _thumbnailScene->SetViewport(width, height);

  _thumbnailScene->DestroyAllSceneObjects();
  asset.StageThumbnailScene(*_thumbnailScene);
  auto const& converter = _studio->GetEditorToRendererSpaceConverter();
  auto from =
      mochi_renderer::ToFilament(converter.TranslationToOutput(mochi::Double3{1.0, 1.0, 0.5}));
  auto to =
      mochi_renderer::ToFilament(converter.TranslationToOutput(mochi::Double3{0.0, 0.0, 0.0}));
  auto const up =
      mochi_renderer::ToFilament(converter.DirectionToOutput(mochi::Double3{0.0, 0.0, 1.0}));
  float orthoHeight = 10;
  _thumbnailScene->CameraLookAt(from, to, up);
  if (_thumbnailScene->GetCameraFocusOnAllSceneObjects(from, to, orthoHeight)) {
    _thumbnailScene->CameraLookAt(from, to, up);
  }
  auto bak = renderer.GetClearColor();
  renderer.SetClearColor({});
  renderer.Render(_thumbnailScene.get(), &target, true);
  renderer.SetClearColor(bak);
  _thumbnailScene->DestroyAllSceneObjects();
}

Asset* AssetManager::RegisterAsset(mochi::Path const& path, std::unique_ptr<Asset> asset) {
  // Returns the live asset stored at `path`: either the existing entry on a duplicate registration
  // (the incoming `asset` is then dropped on scope exit) or the newly-inserted asset. Never returns
  // nullptr, so callers can safely use the return value as their raw pointer.
  auto [it, inserted] = _assets.try_emplace(path, std::move(asset));
  if (!inserted) {
    MOCHI_LOG_WARNING("Asset already registered at path: %s", path.ToString().c_str());
  }
  // Any assets in the OS temp directory (where archive cache lives) are always marked read only for
  // sneaky users that try to add the cache root folder to the asset browser workspace.
  mochi::Path const tempRoot{std::filesystem::temp_directory_path()};
  auto* assetPtr = it->second.get();
  if (path.IsDescendantOf(tempRoot)) {
    assetPtr->SetReadOnly(true);
  }
  return assetPtr;
}

void AssetManager::RewriteAssetPathInMap(mochi::Path const& oldPath, mochi::Path const& newPath) {
  auto it = _assets.find(oldPath);
  if (it == _assets.end()) {
    // No asset at oldPath: legitimate when the caller is only rewriting tab/referencer
    // references for a file that has no in-memory Asset. Not an error.
    return;
  }
  if (_assets.contains(newPath)) {
    MOCHI_LOG_ERROR(
        "Cannot rewrite asset path: destination already occupied: %s -> %s",
        oldPath.ToString().c_str(),
        newPath.ToString().c_str());
    return;
  }
  auto node = _assets.extract(it);
  Asset* asset = node.mapped().get();
  asset->_path = newPath;
  asset->_name = GetAssetNameFromPath(newPath);
  asset->OnRewritePath(oldPath, newPath, _studio->GetResourceManager());
  node.key() = newPath;
  _assets.insert(std::move(node));
}

void AssetManager::RewriteAssetPath(mochi::Path const& oldPath, mochi::Path const& newPath) {
  std::array paths{std::pair{oldPath, newPath}};
  RewriteAssetPaths(paths);
}

void AssetManager::RewriteAssetPaths(
    mochi::Span<std::pair<mochi::Path, mochi::Path>> const& moves) {
  for (auto const& [oldPath, newPath] : moves) {
    RewriteAssetPathInMap(oldPath, newPath);
  }
  auto const affected = NotifyPathsMoved(moves);
  RefreshReferencers(affected);
}

void AssetManager::RepointReferences(mochi::Path const& oldPath, mochi::Path const& newPath) {
  // Repoint referencers only: unlike RewriteAssetPaths, neither asset is moved in _assets, so both
  // the replaced and replacement assets stay loaded at their own paths.
  //
  // Skip any editor that represents the asset at oldPath: such an editor edits that asset itself
  // rather than holding an external reference to it. AssetEditor::RewriteReferencedPath does not
  // retarget the editor's asset pointer, so redirecting its index entry onto newPath would both
  // desync reference tracking and drop the editor as a deletion blocker -- letting a subsequent
  // "Replace and Delete" unload the asset out from under the still-open editor.
  Asset* const target = FindAssetByPath(oldPath);
  auto const shouldNotify = [target](IAssetReferencer* referencer) {
    if (auto* editor = dynamic_cast<AssetEditor*>(referencer)) {
      return !editor->RepresentsAsset(target);
    }
    return true;
  };
  auto const affected = NotifyPathMoved(oldPath, newPath, shouldNotify);
  RefreshReferencers(affected);
}

void AssetManager::RefreshReferencers(std::vector<IAssetReferencer*> const& referencers) {
  if (referencers.empty()) {
    return;
  }
  // Refresh the affected assets themselves (rebuild derived data / thumbnail) before refreshing
  // editors, so an editor that restages from an asset reads the already-updated state.
  for (IAssetReferencer* referencer : referencers) {
    if (auto* asset = dynamic_cast<Asset*>(referencer)) {
      asset->Refresh();
    }
  }
  if (_studio) {
    _studio->RefreshEditors(referencers);
  }
}

int AssetManager::GetPathReferenceCount(mochi::Path const& path) const {
  auto it = _reverseRefs.find(path);
  if (it == _reverseRefs.end()) {
    return 0;
  }
  return static_cast<int>(it->second.size());
}

std::vector<IAssetReferencer*> AssetManager::GetReferencersToPath(mochi::Path const& path) const {
  auto it = _reverseRefs.find(path);
  if (it == _reverseRefs.end()) {
    return {};
  }
  return {it->second.begin(), it->second.end()};
}

std::vector<mochi::Path> AssetManager::GetPathsFromReferencer(IAssetReferencer* asset) const {
  auto it = _forwardRefs.find(asset);
  if (it == _forwardRefs.end()) {
    return {};
  }
  return {it->second.begin(), it->second.end()};
}

void AssetManager::RegisterReferencer(IAssetReferencer* referencer) {
  if (!referencer) {
    return;
  }
  // Insert empty forward entry if not present; Resync fills it in.
  _forwardRefs.try_emplace(referencer);
  ResyncReferencer(referencer);
}

void AssetManager::UnregisterReferencer(IAssetReferencer* referencer) {
  if (!referencer) {
    return;
  }
  auto it = _forwardRefs.find(referencer);
  if (it == _forwardRefs.end()) {
    return;
  }
  for (auto const& path : it->second) {
    auto rIt = _reverseRefs.find(path);
    if (rIt != _reverseRefs.end()) {
      rIt->second.erase(referencer);
      if (rIt->second.empty()) {
        _reverseRefs.erase(rIt);
      }
    }
  }
  _forwardRefs.erase(it);
}

/// Notify the manager that `oldPath` was renamed/moved to `newPath`. Every
/// referencing referencer's RewriteReferencedPath is invoked and the maps are
/// updated incrementally.
/// @return The list of referencers whose references were rewritten.
std::vector<IAssetReferencer*> AssetManager::NotifyPathMoved(
    mochi::Path const& oldPath,
    mochi::Path const& newPath,
    std::function<bool(IAssetReferencer*)> const& shouldNotify) {
  auto rIt = _reverseRefs.find(oldPath);
  if (rIt == _reverseRefs.end()) {
    return {};
  }
  // Snapshot referencing referencers (RewriteReferencedPath may mutate them).
  std::vector<IAssetReferencer*> const candidates(rIt->second.begin(), rIt->second.end());
  std::vector<IAssetReferencer*> rewritten;
  rewritten.reserve(candidates.size());
  for (IAssetReferencer* asset : candidates) {
    // Let the caller exclude referencers that should not be repointed (e.g. an editor that edits
    // the moved asset itself); such referencers stay indexed against oldPath.
    if (shouldNotify && !shouldNotify(asset)) {
      continue;
    }
    // Only update the index when the implementer actually rewrote something.
    // A no-op rewrite means the asset's true reference set never included
    // oldPath (or has already drifted); mutating the index would silently
    // desynchronize it.
    if (!asset->RewriteReferencedPath(oldPath, newPath)) {
      continue;
    }
    auto& fwd = _forwardRefs[asset];
    fwd.erase(oldPath);
    fwd.insert(newPath);
    _reverseRefs[newPath].insert(asset);
    // Drop this asset from the old reverse entry. If it was the last referer,
    // the bucket will be empty and we erase it below.
    rIt->second.erase(asset);
    rewritten.push_back(asset);
  }
  if (rIt->second.empty()) {
    _reverseRefs.erase(rIt);
  }
  return rewritten;
}

/// Batch variant for folder moves. Each pair is processed independently.
/// @return The deduplicated list of all referencers whose references were rewritten.
std::vector<IAssetReferencer*> AssetManager::NotifyPathsMoved(
    mochi::Span<std::pair<mochi::Path, mochi::Path>> const& moves) {
  std::set<IAssetReferencer*> affected;
  for (auto const& [oldPath, newPath] : moves) {
    auto const rewritten = NotifyPathMoved(oldPath, newPath);
    affected.insert(rewritten.begin(), rewritten.end());
  }
  return {affected.begin(), affected.end()};
}

void AssetManager::ResyncReferencer(IAssetReferencer* referencer) {
  if (!referencer) {
    return;
  }
  auto fwdIt = _forwardRefs.find(referencer);
  if (fwdIt == _forwardRefs.end()) {
    return;
  }
  // Collect current paths from the referencer.
  std::set<mochi::Path> newPaths;
  referencer->ForEachReferencedPath([&newPaths](mochi::Path const& p) {
    if (!p.IsEmpty()) {
      newPaths.insert(p);
    }
  });
  auto& oldPaths = fwdIt->second;
  if (newPaths == oldPaths) {
    return;
  }
  // Diff: remove old-not-in-new, add new-not-in-old.
  for (auto const& oldPath : oldPaths) {
    if (!newPaths.contains(oldPath)) {
      auto rIt = _reverseRefs.find(oldPath);
      if (rIt != _reverseRefs.end()) {
        rIt->second.erase(referencer);
        if (rIt->second.empty()) {
          _reverseRefs.erase(rIt);
        }
      }
    }
  }
  for (auto const& newPath : newPaths) {
    if (!oldPaths.contains(newPath)) {
      _reverseRefs[newPath].insert(referencer);
    }
  }
  oldPaths = std::move(newPaths);
}

} // namespace superdex::studio
