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

#include "assets/mochi_model_asset.h"
#include "app/app.h"
#include "assets/asset_manager.h"
#include "editors/model_editor.h"
#include "ui/imgui_widgets.h" // HashStringToColor

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/string_utils.h>
#include <mochi_physics/utils/mochi_model_utils.h>
#include <mochi_renderer/mesh.h>
#include <mochi_renderer/resource_manager.h>
#include <mochi_renderer/utils.h>

#include <superdex_robotics/utils/file_utils.h>

#include <imgui_internal.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

using namespace mochi_renderer;

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// MOCHI MODEL ASSET
//--------------------------------------------------------------------------------------------------

std::unique_ptr<MochiModelAsset> MochiModelAsset::Create(
    std::string const& name,
    mochi::Path const& path,
    AssetManager* manager,
    ResourceManager& resourceManager) {
  auto asset = std::unique_ptr<MochiModelAsset>(
      new MochiModelAsset(name, path, AssetType::MochiModel, manager));
  mochi::ErrorLog error;
  asset->_modelData = mochi::model_utils::LoadFromFile(path.ToString(), error);
  MOCHI_ERROR_RETURN(error, nullptr);
  // Validate the model has usable render geometry (replaces the old RenderModel load check).
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  if (!BuildMochiModelGeometry(
          asset->_modelData, resourceManager.GetSpaceConverter(), positions, normals, indices)) {
    MOCHI_LOG_ERROR("Failed to create MochiModelAsset: model has no usable render geometry.");
    return nullptr;
  }
  // Give each Mochi Model a stable, distinct color derived from its path (see HashStringToColor).
  std::string colorSeed = path.ToString();
  constexpr bool kDeriveColorFromFileHash = false;
  if (kDeriveColorFromFileHash) {
    // TODO: Currently disabled because it makes the asset change color in ModelEditor when the data
    // changes. Leaving code here for future reference since the alternate (path) below is not ideal
    // either.
    mochi::ErrorLog e;
    auto hash = superdex::robotics::HashGenericFile(path.ToString(), e);
    if (e.IsOK()) {
      colorSeed = hash;
    }
  }
  ImVec4 const color = HashStringToColor(colorSeed);
  asset->_color = {color.x, color.y, color.z};
  asset->_wireframeColor = WireframeColorForSurface(asset->_color);

  // Build the one and only flat-lit + wireframe mesh; GetRenderModelInstance() hands out instances.
  asset->_renderModel = mochi_renderer::WireframeMesh::CreateWireframeMesh(
      resourceManager.GetEngine(),
      mochi::Span<float const>(positions.data(), positions.size()),
      mochi::Span<float const>(normals.data(), normals.size()),
      mochi::Span<int const>(indices.data(), indices.size()),
      resourceManager.CreateWireframeMaterial(
          {asset->_wireframeColor.x, asset->_wireframeColor.y, asset->_wireframeColor.z, 1.0f}),
      resourceManager.CreateFlatLitOpaqueMaterial(asset->_color),
      /*isClosed=*/true,
      /*castShadows=*/true,
      /*isDynamic=*/true);
  if (asset->_renderModel == nullptr) {
    MOCHI_LOG_ERROR("Failed to create MochiModelAsset: could not build render mesh.");
    return nullptr;
  }
  return asset;
}

MochiModelAsset::~MochiModelAsset() = default;

std::unique_ptr<SceneObject> MochiModelAsset::GetRenderModelInstance() const {
  return _renderModel ? _renderModel->GetInstance() : nullptr;
}
mochi::ShapeHandle MochiModelAsset::GetShape(
    mochi::Real3 const& bakeScale,
    mochi::TransformRT const& bakeTransform,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Mirror mochi's native file cache key (see GetFileCacheKey in mochi_context.cpp), minus the path
  // component since the path is implicit per asset.
  auto const key = mochi::Format(
      "%s:%s:%s",
      SReflect::ToJsonString(bakeScale).c_str(),
      SReflect::ToJsonString(bakeTransform.GetRotation()).c_str(),
      SReflect::ToJsonString(bakeTransform.GetTranslation()).c_str());

  if (auto it = _shapeCache.find(key); it != _shapeCache.end()) {
    return it->second;
  }

  // Note: experimental/ROM data (loaded from .h5 by the file-based path via
  // hdf5::LoadExperimentalModelDataFromFile) is intentionally not handled here. It is out of scope
  // for the in-memory editor path.
  mochi::ModelData modelCopy = _modelData;
  mochi::model_utils::BakeTransform(modelCopy, bakeScale, bakeTransform, error);
  MOCHI_ERROR_RETURN(error, {});
  auto shape = _manager->GetStudio()->GetMochiContext()->CreateModelShape(modelCopy, error);
  MOCHI_ERROR_RETURN(error, {});

  _shapeCache.emplace(key, shape);
  return shape;
}

void MochiModelAsset::ClearShapeCache() {
  _shapeCache.clear();
}

mochi::MeshDataView MochiModelAsset::GetNativeSoftSurface(mochi::Error& error) {
  using namespace mochi;
  // Only tetrahedral (volumetric) models have a soft-body boundary surface; anything else (surface
  // meshes, analytic shapes) cannot be simulated as a soft actor.
  if (!_modelData.mesh || _modelData.mesh->nodesPerElement != 4) {
    MOCHI_LOG_WARNING_ONCE(
        "MochiModelAsset: model is not a tetrahedral mesh; cannot build a soft surface.");
    return {};
  }
  // Load the native (unbaked) shape once (cached) and read its boundary surface -- the same
  // surface-mesh data the physics engine exposes, so the vertex ordering/count matches the
  // per-frame SurfaceNodePositions query during simulation.
  ShapeHandle const shape = GetShape(Real3{1_r, 1_r, 1_r}, TransformRT::Identity(), error);
  return _manager->GetStudio()->GetMochiContext()->GetShapeSurfaceMesh(shape, error);
}

bool MochiModelAsset::BakeSoftSurface(
    mochi::Real3 const& bakeScale,
    mochi::TransformRT const& shapeTransform,
    std::vector<float>& positions,
    std::vector<float>& normals,
    std::vector<int>& indices) {
  using namespace mochi;
  ErrorLog e;
  MeshDataView const surface = GetNativeSoftSurface(e);
  if (!e.IsOK() || surface.IsEmpty()) {
    return false;
  }
  // Bake scale then the shape offset into the rest positions on the CPU (topology is
  // scale-independent) rather than baking a new physics shape per scale, which would create and
  // cache a shape on every frame of a scale drag.
  auto const& converter = _manager->GetStudio()->GetEditorToRendererSpaceConverter();
  int const numNodes = surface.GetNumNodes();
  positions.resize(static_cast<size_t>(numNodes) * 3);
  for (int n = 0; n < numNodes; ++n) {
    Real3 const native{
        surface.coordinates[3 * n + 0],
        surface.coordinates[3 * n + 1],
        surface.coordinates[3 * n + 2]};
    Real3 const baked = shapeTransform.TransformPoint(bakeScale * native);
    auto const p = StaticCast<Float3>(converter.TranslationToOutput(baked));
    positions[3 * n + 0] = p[0];
    positions[3 * n + 1] = p[1];
    positions[3 * n + 2] = p[2];
  }
  indices.assign(surface.connectivity.begin(), surface.connectivity.end());
  ComputeVertexNormalsAreaWeighted(positions, indices, normals);
  return true;
}

int MochiModelAsset::GetSoftSurfaceVertexCount() {
  mochi::ErrorLog e;
  mochi::MeshDataView const surface = GetNativeSoftSurface(e);
  return (e.IsOK() && !surface.IsEmpty()) ? surface.GetNumNodes() : 0;
}

bool MochiModelAsset::CreateSoftDynamicMeshes(
    mochi::Real3 const& bakeScale,
    mochi::TransformRT const& shapeTransform,
    std::unique_ptr<Mesh>& outSolid,
    std::unique_ptr<WireframeMesh>& outWireframe) {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  if (!BakeSoftSurface(bakeScale, shapeTransform, positions, normals, indices)) {
    return false;
  }
  ResourceManager& rm = _manager->GetStudio()->GetResourceManager();
  auto const p = mochi::MakeConstSpan(positions);
  auto const n = mochi::MakeConstSpan(normals);
  auto const idx = mochi::MakeConstSpan(indices);
  // Solid: flat-lit opaque, deforming. Wireframe: shaded surface (surface color) + edge overlay
  // (wireframe color), deforming. Both share the actor's hashed colors.
  outSolid = Mesh::CreateMesh(
      rm.GetEngine(),
      p,
      n,
      idx,
      rm.CreateFlatLitOpaqueMaterial(_color),
      /*isDynamic=*/true,
      /*isClosed=*/true);
  outWireframe = WireframeMesh::CreateWireframeMesh(
      rm.GetEngine(),
      p,
      n,
      idx,
      rm.CreateWireframeMaterial({_wireframeColor.x, _wireframeColor.y, _wireframeColor.z, 1.0f}),
      rm.CreateFlatLitOpaqueMaterial(_color),
      /*isClosed=*/true,
      /*castShadows=*/true,
      /*isDynamic=*/true);
  return true;
}

void MochiModelAsset::UpdateSoftDynamicMeshes(
    Mesh* solid,
    WireframeMesh* wireframe,
    mochi::Real3 const& bakeScale,
    mochi::TransformRT const& shapeTransform) {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  if (!BakeSoftSurface(bakeScale, shapeTransform, positions, normals, indices)) {
    return;
  }
  auto const p = mochi::MakeConstSpan(positions);
  auto const n = mochi::MakeConstSpan(normals);
  if (solid) {
    solid->Update(p, n);
  }
  if (wireframe) {
    wireframe->Update(p, n);
  }
}

void MochiModelAsset::UpdateRenderModel() {
  if (!_renderModel) {
    return;
  }
  ResourceManager& resourceManager = _manager->GetStudio()->GetResourceManager();
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  if (!BuildMochiModelGeometry(
          _modelData, resourceManager.GetSpaceConverter(), positions, normals, indices)) {
    MOCHI_LOG_ERROR("Failed to rebuild MochiModelAsset render geometry.");
    return;
  }
  // Re-points the single mesh and all its live instances (e.g. open editors, staged scenes).
  _renderModel->UpdateGeometry(
      mochi::Span<float const>(positions.data(), positions.size()),
      mochi::Span<float const>(normals.data(), normals.size()),
      mochi::Span<int const>(indices.data(), indices.size()));
}

bool MochiModelAsset::ReloadFromDisk() {
  mochi::ErrorLog error;
  mochi::ModelData reloaded = mochi::model_utils::LoadFromFile(_path.ToString(), error);
  if (!error.IsOK()) {
    MOCHI_LOG_ERROR("Failed to reload MochiModelAsset from disk: %s", _path.ToString().c_str());
    return false;
  }
  ResourceManager& resourceManager = _manager->GetStudio()->GetResourceManager();
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
  if (!BuildMochiModelGeometry(
          reloaded, resourceManager.GetSpaceConverter(), positions, normals, indices)) {
    MOCHI_LOG_ERROR("Failed to reload MochiModelAsset: model has no usable render geometry.");
    return false;
  }
  _modelData = std::move(reloaded);
  // Physics shapes were baked from the previous model data; drop them so consumers re-bake from the
  // reloaded data on their next GetShape.
  ClearShapeCache();
  // Re-point the single render mesh (and every live instance -- open editors, staged bot scenes) at
  // the reloaded geometry.
  if (_renderModel) {
    _renderModel->UpdateGeometry(
        mochi::Span<float const>(positions.data(), positions.size()),
        mochi::Span<float const>(normals.data(), normals.size()),
        mochi::Span<int const>(indices.data(), indices.size()));
  }
  MarkThumbnailDirty();
  return true;
}

mochi::ModelData const& MochiModelAsset::GetModelData() const {
  return _modelData;
}

mochi::ModelData& MochiModelAsset::GetModelData() {
  return _modelData;
}

filament::math::float3 MochiModelAsset::GetSurfaceColor() const {
  return _color;
}

char const* MochiModelAsset::GetTypeLabel() const {
  return "Collision Model";
}

bool MochiModelAsset::RendersThumbnail() const {
  return true;
}

bool MochiModelAsset::IsSavable() const {
  return !IsReadOnly();
}

bool MochiModelAsset::Save() const {
  if (IsReadOnly()) {
    MOCHI_LOG_ERROR("Attempting to save read-only MochiModelAsset");
    return false;
  }
  std::string ext = _path.GetExtension();
  std::transform(
      ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
  mochi::FileFormat const format = ext == ".h5" ? mochi::FileFormat::H5 : mochi::FileFormat::JSON;
  mochi::ErrorLog error;
  mochi::model_utils::SaveToFile(_modelData, _path.ToString(), format, error);
  return error.IsOK();
}

void MochiModelAsset::StageThumbnailScene(mochi_renderer::Scene& scene) {
  if (auto instance = GetRenderModelInstance()) {
    scene.AddSceneObjectToScene(std::move(instance));
  }
}

void MochiModelAsset::ShowAssetTileTooltipItems() const {
  ImGui::Text("Elements: %d", _modelData.mesh ? _modelData.mesh->GetNumElements() : 0);
  if (_modelData.sdf) {
    auto const& dims = _modelData.sdf->dims;
    auto const size = static_cast<int>(_modelData.sdf->values.size());
    ImGui::Text("SDF Grid: %d x %d x %d (%d)", dims[0], dims[1], dims[2], size);
  } else {
    ImGui::TextUnformatted("SDF Grid: [none]");
  }
  ImGui::Text("Shapes Cached: %d", static_cast<int>(_shapeCache.size()));
}

std::unique_ptr<AssetEditor> MochiModelAsset::CreateEditor(SuperDexStudio* studio) {
  return std::make_unique<ModelEditor>(studio, this);
}

} // namespace superdex::studio
