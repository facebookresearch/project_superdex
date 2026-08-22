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

#include "assets/render_model_asset.h"
#include "app/app.h"
#include "assets/asset_manager.h"
#include "editors/model_editor.h"

#include <mochi_renderer/resource_manager.h>

using namespace mochi_renderer;

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// RENDER MODEL ASSET
//--------------------------------------------------------------------------------------------------

std::unique_ptr<RenderModelAsset> RenderModelAsset::Create(
    std::string const& name,
    mochi::Path const& path,
    AssetManager* manager,
    ResourceManager& resourceManager) {
  auto asset = std::unique_ptr<RenderModelAsset>(
      new RenderModelAsset(name, path, AssetType::RenderModel, manager));
  asset->_renderModel = resourceManager.LoadRenderModel(path);
  if (asset->_renderModel == nullptr) {
    MOCHI_LOG_ERROR("Failed to create RenderModelAsset because RenderModel could not be loaded.");
    return nullptr;
  }
  return asset;
}

std::unique_ptr<SceneObject> RenderModelAsset::GetRenderModelInstance() const {
  return _renderModel ? _renderModel->GetInstance() : nullptr;
}

bool RenderModelAsset::RendersThumbnail() const {
  return true;
}

void RenderModelAsset::StageThumbnailScene(mochi_renderer::Scene& scene) {
  if (auto instance = GetRenderModelInstance()) {
    scene.AddSceneObjectToScene(std::move(instance));
  }
}

void RenderModelAsset::ShowAssetTileTooltipItems() const {
  if (_renderModel) {
    ImGui::Text(
        "Render Model Instances: %d / %d",
        _renderModel->GetInstanceCount(),
        _renderModel->GetMaxInstances());
  }
}

std::unique_ptr<AssetEditor> RenderModelAsset::CreateEditor(SuperDexStudio* studio) {
  return std::make_unique<ModelEditor>(studio, this);
}

void RenderModelAsset::OnUnload(ResourceManager& resourceManager) {
  if (_renderModel) {
    resourceManager.UnloadResource(_renderModel->GetPath());
  }
}

void RenderModelAsset::OnRewritePath(
    mochi::Path const& oldPath,
    mochi::Path const& newPath,
    ResourceManager& resourceManager) {
  resourceManager.RewriteResourcePath(oldPath, newPath);
}

} // namespace superdex::studio
