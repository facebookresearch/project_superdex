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
#include "assets/asset.h"
#include "assets/asset_manager.h"
#include "assets/bot_asset.h"
#include "assets/mochi_prefab_asset.h"
#include "editors/bot_scene_editor.h"
#include "rendering/scene_stage.h"

#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/path.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

namespace superdex::studio {

std::unique_ptr<BotSceneAsset>
BotSceneAsset::Create(std::string const& name, mochi::Path const& path, AssetManager* manager) {
  mochi::ErrorLog error;

  // Archive case: extract to cache, load from the extracted target, mark read-only.
  bool const isArchive = path.GetExtensionLowercase() == superdex::robotics::kSceneArchiveExtension;
  std::string loadPath = path.ToString();
  if (isArchive) {
    auto extractedDir = superdex::robotics::ExtractBotSceneArchiveToCache(path.ToString(), error);
    auto targetPath = superdex::robotics::GetExtractedBotSceneArchiveTarget(extractedDir, error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to extract BotSceneAsset: %s", path.ToString().c_str());
      return nullptr;
    }
    loadPath = std::string(targetPath);
    // first load the scene from the unzipped target path so that it is also a valid asset
    manager->LoadBotSceneAsset(loadPath);
    // now continue to internalize the archived scene as a valid asset too
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
    asset->SetReadOnly(true);
  }

  // Resolve the base directory used for the base scene's relative shape/prefab paths: the
  // .superdex_root if present, otherwise the base scene's own directory.
  auto botsRoot = superdex::robotics::FindBotsRoot(path.ToString());
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
  return !_isArchive && !IsReadOnly();
}

bool BotSceneAsset::Save() const {
  if (IsReadOnly()) {
    MOCHI_LOG_ERROR("Attempting to save read-only BotSceneAsset");
    return false;
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

std::string const& BotSceneAsset::GetBotsRootPath() const {
  return _botsRootPath;
}

} // namespace superdex::studio

#endif // MOCHI_INTERNAL
