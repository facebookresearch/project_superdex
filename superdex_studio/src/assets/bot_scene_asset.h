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

#include "assets/asset.h"
#include "assets/asset_referencer.h"
#include "assets/bot_task_cache.h"
#include "assets/controller_params.h"

#include <superdex_robotics/scene/bot_scene.h>

#include <optional>

namespace superdex::studio {

class AssetManager;

class BotSceneAsset : public Asset, public IAssetReferencer {
 public:
  ~BotSceneAsset() override = default;

  // Asset
  char const* GetTypeLabel() const override;
  bool RendersThumbnail() const override;
  void StageThumbnailScene(mochi_renderer::Scene& scene) override;
  bool IsSavable() const override;
  bool Save() override;
  [[nodiscard]] bool SupportsSaveAs() const override;
  [[nodiscard]] bool SaveAs(mochi::Path const& path) override;
  std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio* studio) override;

  // IAssetReferencer
  std::string const& GetReferencerName() const override;
  void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const override;
  bool RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) override;

  superdex::robotics::BotScenePrefab const& GetPrefab() const;
  superdex::robotics::BotScenePrefab& GetPrefab();
  // Params of controllers that reference .superdex_controller files, written by Save.
  ControllerParamsCache& GetControllerParams();
  // Tasks edited in this scene's editor, written by Save.
  BotTaskCache& GetTaskCache();
  std::string const& GetBotsRootPath() const;
  [[nodiscard]] bool IsArchive() const;
  [[nodiscard]] bool SaveArchiveCopy(mochi::Path const& path) const;
  bool ExtractArchiveToLooseFiles(mochi::Path const& directory, mochi::Path& scenePath) const;

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<BotSceneAsset>
  Create(std::string const& name, mochi::Path const& path, AssetManager* manager);
  // Moves inline controller params to files (asking where, starting in @p sceneDirectory) and
  // writes modified params files. Runs before the scene is written, so an archive bundles them.
  bool SaveControllerParams(mochi::Path const& sceneDirectory);

 private:
  superdex::robotics::BotScenePrefab _prefab;
  ControllerParamsCache _controllerParams;
  BotTaskCache _taskCache;
  std::string _botsRootPath;
  mochi::Path _archiveExtractedDir;
  mochi::Path _archiveTargetPath;
  std::optional<mochi::DynamicString> _archiveComment;
  bool _isArchive = false;
};

} // namespace superdex::studio
