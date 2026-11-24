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

#include "assets/tuner_asset.h"

#include "assets/asset_manager.h"
#include "assets/asset_types.h"
#include "assets/mochi_prefab_asset.h"
#include "assets/scene_asset.h"
#include "editors/tuner_editor.h"

#include <mochi_core/utils/path.h>

#include <filesystem>
#include <optional>

namespace superdex::studio {

std::unique_ptr<TunerAsset>
TunerAsset::Create(std::string const& name, mochi::Path const& path, AssetManager* manager) {
  superdex::tuner::TunerSpec spec;
  std::string error;
  if (!superdex::tuner::LoadTunerSpecFromFile(path.AsFilesystemPath(), {}, spec, error)) {
    MOCHI_LOG_ERROR("Failed to load asset tuner: %s", error.c_str());
    return nullptr;
  }
  auto asset = std::unique_ptr<TunerAsset>(new TunerAsset(name, path, AssetType::Tuner, manager));
  asset->_spec = std::move(spec);
  manager->RegisterReferencer(asset.get());
  asset->RefreshReferencesFromSpec();
  return asset;
}

char const* TunerAsset::GetTypeLabel() const {
  return "Asset Tuner";
}

bool TunerAsset::RendersThumbnail() const {
  return GetPrefabAsset() != nullptr;
}

void TunerAsset::StageThumbnailScene(mochi_renderer::Scene& scene) {
  if (MochiPrefabAsset* prefab = GetPrefabAsset()) {
    prefab->StageThumbnailScene(scene);
  }
}

bool TunerAsset::IsSavable() const {
  return !IsReadOnly();
}

bool TunerAsset::Save() {
  std::string error;
  if (!superdex::tuner::SaveTunerSpecToFile(_spec, _path.AsFilesystemPath(), error)) {
    MOCHI_LOG_ERROR("Failed to save asset tuner: %s", error.c_str());
    return false;
  }
  return true;
}

std::unique_ptr<AssetEditor> TunerAsset::CreateEditor(SuperDexStudio* studio) {
  return std::make_unique<TunerEditor>(studio, this);
}

MochiPrefabAsset* TunerAsset::GetPrefabAsset() const {
  return _manager->FindAssetByPath<MochiPrefabAsset>(_prefabPath);
}

SceneAsset* TunerAsset::GetSceneAsset() const {
  return _scenePath.IsEmpty() ? nullptr : _manager->FindAssetByPath<SceneAsset>(_scenePath);
}

std::string const& TunerAsset::GetReferencerName() const {
  return GetName();
}

void TunerAsset::ForEachReferencedPath(
    std::function<void(mochi::Path const&)> const& callback) const {
  if (!_prefabPath.IsEmpty()) {
    callback(_prefabPath);
  }
  if (!_scenePath.IsEmpty()) {
    callback(_scenePath);
  }
  if (!_taskPath.IsEmpty()) {
    callback(_taskPath);
  }
}

bool TunerAsset::RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) {
  if (_prefabPath == oldPath) {
    _prefabPath = newPath;
    _spec.prefab = ToSpecPath(newPath);
  } else if (_scenePath == oldPath) {
    _scenePath = newPath;
    _spec.scene = ToSpecPath(newPath);
  } else if (_taskPath == oldPath) {
    _taskPath = newPath;
    _spec.task = ToSpecPath(newPath);
  } else {
    return false;
  }
  SetDirty(true);
  return true;
}

void TunerAsset::SetPrefabOrScene(mochi::Path const& path) {
  bool const isScene = ClassifyAssetTypeByFilename(path.GetFilename()) == AssetType::Scene;
  mochi::DynamicString const specPath(ToSpecPath(path));
  if (path != _scenePath) {
    // Both belong to the scene being replaced.
    _spec.task = mochi::DynamicString();
    _spec.bots = {};
  }
  _spec.prefab = isScene ? mochi::DynamicString() : specPath;
  _spec.scene = isScene ? specPath : mochi::DynamicString();
  SetDirty(true);
  RefreshReferencesFromSpec();
}

void TunerAsset::SetTaskPath(mochi::Path const& path) {
  _spec.task = path.IsEmpty() ? mochi::DynamicString() : mochi::DynamicString(ToSpecPath(path));
  SetDirty(true);
  RefreshReferencesFromSpec();
}

bool TunerAsset::RefreshReferencesFromSpec() {
  std::filesystem::path const file = _path.AsFilesystemPath();
  mochi::Path const prefab(superdex::tuner::ResolvePrefabPath(_spec, file));
  mochi::Path const scene(superdex::tuner::ResolveScenePath(_spec, file));
  mochi::Path const task(superdex::tuner::ResolveTaskPath(_spec, file));
  if (prefab == _prefabPath && scene == _scenePath && task == _taskPath) {
    return false;
  }
  _prefabPath = prefab;
  _scenePath = scene;
  _taskPath = task;
  if (!_prefabPath.IsEmpty() && _manager->LoadMochiPrefabAsset(_prefabPath) == nullptr) {
    MOCHI_LOG_ERROR(
        "Asset Tuner %s: failed to load prefab %s",
        _path.ToString().c_str(),
        _prefabPath.ToString().c_str());
  }
  if (!_scenePath.IsEmpty() && _manager->LoadSceneAsset(_scenePath) == nullptr) {
    MOCHI_LOG_ERROR(
        "Asset Tuner %s: failed to load scene %s",
        _path.ToString().c_str(),
        _scenePath.ToString().c_str());
  }
  // Keeps the new assets loaded and lets the old ones unload.
  _manager->ResyncReferencer(this);
  return true;
}

std::string TunerAsset::ToSpecPath(mochi::Path const& path) const {
  auto const relativeIfBelow = [&](std::filesystem::path const& dir) -> std::optional<std::string> {
    std::filesystem::path const relative = path.AsFilesystemPath().lexically_relative(dir);
    if (relative.empty() || *relative.begin() == "..") {
      return std::nullopt;
    }
    return relative.generic_string();
  };
  std::filesystem::path const tunerPath = _path.AsFilesystemPath();
  if (auto const relative = relativeIfBelow(tunerPath.parent_path())) {
    return "./" + *relative;
  }
  if (auto const relative =
          relativeIfBelow(superdex::tuner::ResolveAssetsRoot(_spec, tunerPath, tunerPath))) {
    return *relative;
  }
  return path.ToString();
}

} // namespace superdex::studio
