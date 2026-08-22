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

#include <mochi_renderer/resource.h>
#include <mochi_renderer/scene.h>

namespace superdex::studio {

class AssetManager;

//--------------------------------------------------------------------------------------------------
// RENDER MODEL ASSET
//--------------------------------------------------------------------------------------------------

class RenderModelAsset : public Asset {
 public:
  std::unique_ptr<mochi_renderer::SceneObject> GetRenderModelInstance() const;
  bool RendersThumbnail() const override;
  void StageThumbnailScene(mochi_renderer::Scene& scene) override;
  void ShowAssetTileTooltipItems() const override;
  std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio* studio) override;

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<RenderModelAsset> Create(
      std::string const& name,
      mochi::Path const& path,
      AssetManager* manager,
      mochi_renderer::ResourceManager& resourceManager);
  void OnUnload(mochi_renderer::ResourceManager& resourceManager) override;
  void OnRewritePath(
      mochi::Path const& oldPath,
      mochi::Path const& newPath,
      mochi_renderer::ResourceManager& resourceManager) override;

 private:
  mochi_renderer::RenderModel* _renderModel = nullptr;
};

} // namespace superdex::studio
