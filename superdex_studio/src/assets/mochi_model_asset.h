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

#include <mochi_physics/cpp_api/mochi_context.h>
#include <mochi_renderer/resource.h>
#include <mochi_renderer/scene.h>

#include <mochi_core/utils/span.h>

#include <math/vec3.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mochi_renderer {
class Mesh;
class WireframeMesh;
} // namespace mochi_renderer

namespace superdex::studio {

class AssetManager;

//--------------------------------------------------------------------------------------------------
// MOCHI MODEL ASSET
//--------------------------------------------------------------------------------------------------

class MochiModelAsset : public Asset {
 public:
  ~MochiModelAsset() override;
  std::unique_ptr<mochi_renderer::SceneObject> GetRenderModelInstance() const;
  mochi::ShapeHandle GetShape(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& bakeTransform,
      mochi::Error& error);
  void ClearShapeCache();
  void UpdateRenderModel();
  mochi::ModelData const& GetModelData() const;
  mochi::ModelData& GetModelData();
  filament::math::float3 GetSurfaceColor() const;

  // Soft Dynamic Mesh Utils
  int GetSoftSurfaceVertexCount();
  bool CreateSoftDynamicMeshes(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      std::unique_ptr<mochi_renderer::Mesh>& outSolid,
      std::unique_ptr<mochi_renderer::WireframeMesh>& outWireframe);
  void UpdateSoftDynamicMeshes(
      mochi_renderer::Mesh* solid,
      mochi_renderer::WireframeMesh* wireframe,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform);

  // Asset overrides
  char const* GetTypeLabel() const override;
  bool RendersThumbnail() const override;
  bool IsSavable() const override;
  bool Save() const override;
  bool ReloadFromDisk() override;
  void StageThumbnailScene(mochi_renderer::Scene& scene) override;
  void ShowAssetTileTooltipItems() const override;
  std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio* studio) override;

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<MochiModelAsset> Create(
      std::string const& name,
      mochi::Path const& path,
      AssetManager* manager,
      mochi_renderer::ResourceManager& resourceManager);

  // Load the model's tetrahedral boundary surface (native, unbaked); empty view + one warning if
  // the model is not a tetrahedral mesh.
  mochi::MeshDataView GetNativeSoftSurface(mochi::Error& error);
  // Bake the native surface for `bakeScale` + `shapeTransform` into flat renderer-space buffers
  // (area-weighted normals). Returns false if the model is not a tetrahedral mesh.
  bool BakeSoftSurface(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      std::vector<float>& positions,
      std::vector<float>& normals,
      std::vector<int>& indices);

 private:
  std::unique_ptr<mochi_renderer::WireframeMesh> _renderModel;
  filament::math::float3 _color = {0.5f, 0.7f, 1.0f};
  filament::math::float3 _wireframeColor = {1.0f, 1.0f, 1.0f};
  mochi::ModelData _modelData;
  std::unordered_map<std::string, mochi::ShapeHandle> _shapeCache;
};

} // namespace superdex::studio
