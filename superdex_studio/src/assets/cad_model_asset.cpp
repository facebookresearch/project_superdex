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

#include "assets/cad_model_asset.h"
#include "app/app.h"
#include "assets/asset_manager.h"
#include "editors/model_editor.h"

#include <mochi_mesh/step_tessellation.h>

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>

#include <mochi_renderer/resource_manager.h>
#include <mochi_renderer/utils.h>

#include <utility>
#include <vector>

using namespace mochi_renderer;

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// CAD MODEL ASSET
//--------------------------------------------------------------------------------------------------

std::vector<MeshSection> TessellateCadModelFile(
    mochi::Path const& path,
    mochi::mesh::StepTessellationParams const& params,
    mochi::Error& error) {
  mochi::MeshData const mesh = mochi::mesh::TessellateStep(path.ToString(), params, error);
  if (!error.IsOK() || mesh.nodesPerElement != 3 || mesh.GetNumElements() == 0) {
    MOCHI_LOG_ERROR("Failed to tessellate CAD model: %s", path.ToString().c_str());
    return {};
  }

  MeshSection section;
  section.positions.reserve(mesh.coordinates.size());
  for (mochi::real const coordinate : mesh.coordinates) {
    section.positions.push_back(static_cast<float>(coordinate));
  }
  section.indices.reserve(mesh.connectivity.size());
  for (int const index : mesh.connectivity) {
    section.indices.push_back(index);
  }
  section.hasNormals = false;

  MOCHI_LOG(
      "Tessellated CAD model %s: %d vertices, %d triangles",
      path.ToString().c_str(),
      static_cast<int>(section.positions.size() / 3),
      static_cast<int>(section.indices.size() / 3));

  // No space conversion here, unlike the .obj/.stl readers: the superdex_mesh_cli helper already
  // converts OCCT's Z-up mm to renderer-space (Y-up) m.
  std::vector<MeshSection> sections;
  sections.push_back(std::move(section));
  return sections;
}

std::unique_ptr<CadModelAsset> CadModelAsset::Create(
    std::string const& name,
    mochi::Path const& path,
    AssetManager* manager,
    ResourceManager& /*resourceManager*/) {
  // The CAD geometry is tessellated lazily and asynchronously by the Model Editor (OpenCascade runs
  // in the superdex_mesh_cli helper) when the asset is opened; see ModelEditor::RegenerateCadModel,
  // which calls SetTessellation when it completes. Until then the asset is a lightweight reference:
  // no render model and no thumbnail.
  return std::unique_ptr<CadModelAsset>(
      new CadModelAsset(name, path, AssetType::CadModel, manager));
}

std::unique_ptr<SceneObject> CadModelAsset::GetRenderModelInstance() const {
  return _renderModel ? _renderModel->GetInstance() : nullptr;
}

std::vector<mochi_renderer::MeshSection> const& CadModelAsset::GetMeshSections() const {
  return _meshSections;
}

void CadModelAsset::SetTessellation(
    std::vector<MeshSection> sections,
    ResourceManager& resourceManager) {
  if (_renderModel) {
    resourceManager.UnloadResource(_renderModel->GetPath());
    _renderModel = nullptr;
  }
  _meshSections = std::move(sections);
  MarkThumbnailDirty();
  if (_meshSections.empty()) {
    return;
  }
  _renderModel =
      resourceManager.LoadModelFromMeshSections(GetPath(), _meshSections, RenderModelFormat::Cad);
}

bool CadModelAsset::RendersThumbnail() const {
  return _renderModel != nullptr;
}

void CadModelAsset::StageThumbnailScene(mochi_renderer::Scene& scene) {
  if (auto instance = GetRenderModelInstance()) {
    scene.AddSceneObjectToScene(std::move(instance));
  }
}

std::unique_ptr<AssetEditor> CadModelAsset::CreateEditor(SuperDexStudio* studio) {
  return std::make_unique<ModelEditor>(studio, this);
}

void CadModelAsset::OnUnload(ResourceManager& resourceManager) {
  if (_renderModel) {
    resourceManager.UnloadResource(_renderModel->GetPath());
  }
}

void CadModelAsset::OnRewritePath(
    mochi::Path const& oldPath,
    mochi::Path const& newPath,
    ResourceManager& resourceManager) {
  if (_renderModel) {
    resourceManager.RewriteResourcePath(oldPath, newPath);
  }
}

} // namespace superdex::studio
