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

#include <mochi_mesh/step_tessellation.h>

#include <mochi_core/utils/error.h>

#include <mochi_renderer/resource.h>
#include <mochi_renderer/scene.h>
#include <mochi_renderer/utils.h>

#include <vector>

namespace superdex::studio {

class AssetManager;

// Tessellates a STEP/CAD file into renderer mesh sections (one section). Calls into the
// superdex_mesh_cli helper (OpenCascade) via mochi_mesh and performs no renderer work, so it is
// safe to run off the main thread (e.g. inside an AsyncTask). Returns empty on failure (with @p
// error set).
std::vector<mochi_renderer::MeshSection> TessellateCadModelFile(
    mochi::Path const& path,
    mochi::mesh::StepTessellationParams const& params,
    mochi::Error& error);

//--------------------------------------------------------------------------------------------------
// CAD MODEL ASSET
//--------------------------------------------------------------------------------------------------

// A CAD model: a STEP file (.step/.stp) or an STL (.stl). Distinct from RenderModel and MochiModel:
// a CAD model cannot be used as either the render model or the mochi model of a bot. A STEP is
// tessellated into a triangle mesh (via the superdex_mesh_cli helper) for display in the Model
// Editor viewport; an STL is already a triangle mesh and is read directly.
class CadModelAsset : public Asset {
 public:
  std::unique_ptr<mochi_renderer::SceneObject> GetRenderModelInstance() const;
  // The tessellated geometry retained as mesh sections (the working backing used by the Model
  // Editor for wireframe rendering and, later, mesh processing/export).
  std::vector<mochi_renderer::MeshSection> const& GetMeshSections() const;
  // Stores the tessellated geometry and (re)builds the thumbnail render model from it. The Model
  // Editor calls this when its asynchronous tessellation completes (CAD models are not tessellated
  // at load time, so there is no thumbnail until the asset is opened and tessellated).
  void SetTessellation(
      std::vector<mochi_renderer::MeshSection> sections,
      mochi_renderer::ResourceManager& resourceManager);
  bool RendersThumbnail() const override;
  void StageThumbnailScene(mochi_renderer::Scene& scene) override;
  std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio* studio) override;

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<CadModelAsset> Create(
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
  std::vector<mochi_renderer::MeshSection> _meshSections;
};

} // namespace superdex::studio
