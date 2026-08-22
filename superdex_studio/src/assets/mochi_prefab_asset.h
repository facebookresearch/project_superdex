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

#include <mochi_renderer/resource.h>
#include <mochi_renderer/scene.h>

#include <mochi_physics/utils/mochi_prefab.h>

#include <optional>

namespace superdex::studio {

class AssetManager;

class MochiPrefabAsset : public Asset, public IAssetReferencer {
 public:
  ~MochiPrefabAsset() override = default;

  // Asset
  unsigned int GetColor() const override;
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

  mochi::prefab::ScenePrefab const& GetPrefab() const;
  mochi::prefab::ScenePrefab& GetPrefab();
  std::string const& GetAssetsRoot() const;
  // Scene settings removed by unchecking "Scene Prefab", held so re-checking restores them rather
  // than fabricating defaults. Lives on the asset so it survives closing the editor.
  std::optional<mochi::prefab::SceneParams>& GetStashedSceneParams();

 protected:
  void OnRewritePath(
      mochi::Path const& oldPath,
      mochi::Path const& newPath,
      mochi_renderer::ResourceManager& resourceManager) override;

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<MochiPrefabAsset>
  Create(std::string const& name, mochi::Path const& path, AssetManager* manager);

 private:
  mochi::prefab::ScenePrefab _prefab;
  std::string _assetsRoot;
  std::optional<mochi::prefab::SceneParams> _stashedSceneParams;
};

} // namespace superdex::studio
