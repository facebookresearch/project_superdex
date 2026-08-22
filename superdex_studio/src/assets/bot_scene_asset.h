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

#if MOCHI_INTERNAL

#include "assets/asset.h"
#include "assets/asset_referencer.h"

#include <superdex_robotics/internal/bot_scene.h>

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
  bool Save() const override;
  std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio* studio) override;

  // IAssetReferencer
  std::string const& GetReferencerName() const override;
  void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const override;
  bool RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) override;

  superdex::robotics::BotScenePrefab const& GetPrefab() const;
  superdex::robotics::BotScenePrefab& GetPrefab();
  std::string const& GetBotsRootPath() const;

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<BotSceneAsset>
  Create(std::string const& name, mochi::Path const& path, AssetManager* manager);

 private:
  superdex::robotics::BotScenePrefab _prefab;
  std::string _botsRootPath;
  bool _isArchive = false;
};

} // namespace superdex::studio

#endif // MOCHI_INTERNAL
