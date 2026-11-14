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
#include "rendering/soft_surface.h"
#include "rendering/stats.h"

#include <mochi_physics/cpp_api/mochi_context.h>
#include <mochi_renderer/resource.h>
#include <mochi_renderer/scene.h>

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/span.h>

#include <math/vec3.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mochi_renderer {
class DeformableModelInstance;
class Mesh;
class WireframeMesh;
} // namespace mochi_renderer

namespace superdex::studio {

class AssetManager;

// Role suffix hashed with a mesh's color seed to color the SDF surface reconstructed from it (see
// MochiModelAsset::GetSdfSurfaceColor), so SDF surfaces read alike everywhere they are shown.
inline constexpr std::string_view kSdfSurfaceColorRole = " output SDF";

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
  // Computed on first use and again after the model reloads.
  ModelStats const& GetStats() const;
  filament::math::float3 GetSurfaceColor() const;

  // Soft Dynamic Mesh Utils
  int GetSoftSurfaceVertexCount();
  // Node count of one of the shape's triangle meshes, regardless of element type, or 0 if the shape
  // has no such mesh. SoftSurface::Surface is the tetrahedral boundary or the triangle surface
  // itself.
  int GetSurfaceVertexCount(SoftSurface surface);
  // The meshes a soft actor of this model is drawn with (see ChooseSoftSurfaces).
  SoftSurfaces GetSoftSurfaces(bool useContactSkin) const;
  // A wireframe of `surfaces.collision`, plus a solid mesh of `surfaces.rendered` when
  // @p createSolid is true, both deformable.
  bool CreateSoftDynamicMeshes(
      SoftSurfaces const& surfaces,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      bool createSolid,
      std::unique_ptr<mochi_renderer::Mesh>& outSolid,
      std::unique_ptr<mochi_renderer::WireframeMesh>& outWireframe);
  // Restores meshes made by CreateSoftDynamicMeshes with the same `surfaces` to their rest pose.
  void UpdateSoftDynamicMeshes(
      SoftSurfaces const& surfaces,
      mochi_renderer::Mesh* solid,
      mochi_renderer::WireframeMesh* wireframe,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform);
  // Sets @p renderModel, a render model linked to this model's visual mesh (see
  // CheckRenderModelLink), to the visual mesh's rest pose, baked like UpdateSoftDynamicMeshes.
  void UpdateSoftRenderModel(
      mochi_renderer::DeformableModelInstance& renderModel,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform);

  // Build a GPU-skinned wireframe of the shape's collision surface, for previewing an articulated
  // skin's deformation at editor time (and driving it at sim time) without per-vertex CPU updates.
  // The surface must carry skinning (Shape::GetSurfaceMeshData exposes it for skinned mesh shapes);
  // returns false otherwise. `boneCount` is the number of bones the caller will drive (articulation
  // link count); the renderable allocates at least enough bones to cover the referenced indices.
  // Deform it via WireframeMesh::SetBoneMatrices.
  bool CreateSkinnedCollisionWireframe(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      int boneCount,
      std::unique_ptr<mochi_renderer::WireframeMesh>& outWireframe);

  // The zero isosurface of the model's SDF grid, in renderer space in the same frame as the model's
  // own render geometry.
  struct SdfSurface {
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<int> indices;
  };
  // Reconstructs the surface of @p sdf through the mesh CLI. Touches no asset state, so it can run
  // off the main thread. Returns nullopt if reconstruction fails or yields no triangles.
  static std::optional<SdfSurface> ReconstructSdfSurface(
      mochi::GridSdfData const& sdf,
      mochi::CoordinateSpaceConverter const& converter);

  // The SDF surface is reconstructed on demand and cached, as one mesh that every viewport
  // instances (see GetSdfRenderModelInstance). ReloadFromDisk rebuilds it in place when it was
  // built, so every staged instance follows a re-meshed or re-baked model.
  bool HasSdf() const;
  // True once the SDF surface was reconstructed (successfully or not) for the current model data.
  bool IsSdfSurfaceBuilt() const;
  // Bumped whenever the model data is reloaded, so a surface reconstructed in the background from
  // older data can be told apart and dropped (see SetSdfSurface).
  int GetSdfGeneration() const;
  // Reconstructs the SDF surface now unless already built. Returns the cached surface, or null when
  // the model has no SDF or reconstruction failed.
  SdfSurface const* EnsureSdfSurface();
  // Installs a surface reconstructed in the background from this asset's SDF at @p generation.
  // Ignored if the model was reloaded since.
  void SetSdfSurface(int generation, std::optional<SdfSurface> surface);
  // An instance of the shared SDF surface mesh, or null until a surface is built.
  std::unique_ptr<mochi_renderer::SceneObject> GetSdfRenderModelInstance() const;
  // The SDF surface color: hashed from the path like GetSurfaceColor, with an extra role suffix, so
  // the SDF is distinct from the collision surface yet stable everywhere it is shown.
  filament::math::float3 GetSdfSurfaceColor() const;

  // Asset overrides
  char const* GetTypeLabel() const override;
  bool RendersThumbnail() const override;
  bool IsSavable() const override;
  bool Save() override;
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

  // Load one of the shape's triangle meshes (native, unbaked). SoftSurface::Surface is the
  // tetrahedral boundary for a tet mesh, or the triangle surface itself for a surface mesh. Vertex
  // ordering matches the physics engine's node queries for that mesh (e.g. SurfaceNodePositions),
  // so the same buffer can be updated in place during simulation. Empty if the shape has no such
  // mesh.
  mochi::MeshDataView GetNativeSurfaceMesh(SoftSurface surface, mochi::Error& error);
  // Load the model's tetrahedral boundary surface (native, unbaked); empty view + one warning if
  // the model is not a tetrahedral mesh.
  mochi::MeshDataView GetNativeSoftSurface(mochi::Error& error);
  // Bake a native triangle mesh for `bakeScale` + `shapeTransform` into flat renderer-space buffers
  // (area-weighted normals). Works for both tetrahedral boundaries and triangle surfaces (soft
  // callers are already gated to tet meshes upstream). Returns false if the shape has no such mesh.
  bool BakeSoftSurface(
      SoftSurface surface,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      std::vector<float>& positions,
      std::vector<float>& normals,
      std::vector<int>& indices);
  // Read the (surface-aligned) skinning from the native surface mesh into per-node bone indices and
  // weights (numNodes * weightsPerNode each, in the same node order as BakeSoftSurface). Returns
  // false if the shape carries no surface skinning. `outMaxBoneIndex` is the largest bone index
  // referenced (-1 if none).
  bool BakeSoftSurfaceSkinning(
      std::vector<int>& boneIndices,
      std::vector<float>& boneWeights,
      int& weightsPerNode,
      int& outMaxBoneIndex);
  // Caches @p surface as the built SDF surface and points the shared SDF mesh (and so every live
  // instance) at it.
  void InstallSdfSurface(std::optional<SdfSurface> surface);

 private:
  std::unique_ptr<mochi_renderer::WireframeMesh> _renderModel;
  std::unique_ptr<mochi_renderer::WireframeMesh> _sdfRenderModel;
  std::optional<SdfSurface> _sdfSurface;
  bool _sdfSurfaceBuilt = false;
  int _sdfGeneration = 0;
  filament::math::float3 _sdfColor = {1.0f, 0.0f, 0.0f};
  filament::math::float3 _color = {0.5f, 0.7f, 1.0f};
  filament::math::float3 _wireframeColor = {1.0f, 1.0f, 1.0f};
  mochi::ModelData _modelData;
  mutable std::optional<ModelStats> _stats;
  std::unordered_map<std::string, mochi::ShapeHandle> _shapeCache;
};

} // namespace superdex::studio
